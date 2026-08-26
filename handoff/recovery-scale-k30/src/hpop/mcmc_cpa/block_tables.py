"""Candidate block scores over a CPA vocabulary, with per-skill role supports.

`hpop.mcmc_original.stage6e_block_table.BlockScoreTable` scores every legal candidate block
under every skill, reading observed symbols directly as role indices. That is only correct
when `A = m`. This module does the same job when a skill owns `m` of `A` symbols through an
injective role map:

* an observed CPA is translated to that skill's role index before the recurrent arithmetic
  runs, so the arithmetic itself is unchanged;
* a block containing **any** CPA outside the skill's support scores `-inf`, because that
  skill cannot have produced it.

`mcmc_original` is sealed and is not edited. The recurrent loop below is the same
recurrence, reproduced so the translation and the mask can sit inside it, and
`assert_matches_sealed_scorer` pins it against the sealed per-block scorer on relabelled
traces.

## What the mask does to the candidate set

With `m = 10` of `A = 50` and unrelated supports, a block of width `w` survives for a given
skill only if all `w` of its CPAs fall in that skill's ten. Most candidates therefore die,
which is the point: it is what makes skills identifiable when there are thirty of them.
It also changes the candidate geometry completely, so runtime measured under `A = m` does
not transfer -- recalibrate rather than extrapolate.
"""

from __future__ import annotations

import numpy as np

from hpop.mcmc_original.stage6e_frozen import MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH

from .role_maps import NOT_IN_SUPPORT, RoleMaps

__all__ = ["CPABlockScoreTable", "assert_matches_sealed_scorer"]


class CPABlockScoreTable:
    """`log p_RFS(x[a:b) | U_k, ell_k, scalars)` for every legal candidate block.

    The dense per-trace layout is `(J, J+1, K)`, the same one the frozen backward sampler
    already accepts, so nothing downstream changes.
    """

    def __init__(self, traces, epsilon: float, role_maps: RoleMaps,
                 min_width: int = MIN_BLOCK_WIDTH, max_width: int = MAX_BLOCK_WIDTH):
        self.traces = tuple(tuple(int(v) for v in t) for t in traces)
        self.epsilon = float(epsilon)
        self.role_maps = role_maps
        self.n_skills = int(role_maps.n_skills)
        self.n_roles = int(role_maps.n_roles)
        self.min_width = int(min_width)
        self.max_width = int(max_width)
        self.version = 0
        self.refresh_calls = 0
        self.skill_rebuilds = 0
        self._built_u = None
        self._built_params = None

        for n, trace in enumerate(self.traces):
            if trace and (min(trace) < 0 or max(trace) >= role_maps.n_cpa):
                raise ValueError(f"trace {n} contains a CPA outside the vocabulary")

        # Per skill, the whole corpus translated once: role index, or -1 out of support.
        self._roles = [role_maps.to_roles(t) for t in self.traces]      # each (K, J)

        # Width-bucketed candidate rows, exactly as the sealed builder buckets them.
        self._buckets: dict = {}
        self._row: dict = {}
        offset = 0
        for width in range(self.min_width, self.max_width + 1):
            rows = [(n, a) for n, trace in enumerate(self.traces)
                    for a in range(0, len(trace) - width + 1)]
            if not rows:
                continue
            self._buckets[width] = (np.array(rows, dtype=np.int64), offset)
            for index, (n, a) in enumerate(rows):
                self._row[(n, a, width)] = offset + index
            offset += len(rows)
        self.n_blocks = offset
        self._table = np.full((self.n_skills, offset), -np.inf)
        self._live = np.zeros(self.n_skills, dtype=np.int64)

        self.tables = [np.full((len(t), len(t) + 1, self.n_skills), -np.inf)
                       for t in self.traces]
        self._index = self._dense_index()

    def _dense_index(self) -> list:
        out = []
        for n in range(len(self.traces)):
            rows, a_ix, b_ix = [], [], []
            for (tn, a, width), row in self._row.items():
                if tn != n:
                    continue
                rows.append(row)
                a_ix.append(a)
                b_ix.append(a + width)
            out.append((np.array(rows, dtype=np.int64), np.array(a_ix, dtype=np.int64),
                        np.array(b_ix, dtype=np.int64)))
        return out

    # ------------------------------------------------------------- the only writer
    def _build_skill(self, skill: int, u, beta: float, omega: float, lambda_rep: float,
                     lambda_back: float) -> int:
        """Recompute exactly one skill's row of `_table`. Returns its live-block count.

        The whole arithmetic of the candidate score lives here and nowhere else, so the
        all-skills path and the skill-local path cannot drift apart: they are the same
        code called over different index sets, not two implementations kept in step.
        """
        kappa = 1.0 / (1.0 + np.exp(-float(omega)))
        self._table[skill].fill(-np.inf)
        live_here = 0
        if True:
            u = np.asarray(u, dtype=float)
            m = u.shape[0]
            precedence = np.all(u[:, None, :] > u[None, :, :], axis=2)
            succ = precedence.astype(float)
            succ_off = succ.copy()
            np.fill_diagonal(succ_off, 0.0)
            predecessors = [np.flatnonzero(precedence[:, x]) for x in range(m)]

            for width, (rows, offset) in self._buckets.items():
                # translate this bucket's blocks into skill-k role indices
                roles = np.stack([self._roles[n][skill, a:a + width] for n, a in rows]) \
                    if rows.size else np.zeros((0, width), dtype=np.int64)
                in_support = (roles != NOT_IN_SUPPORT).all(axis=1)
                if not in_support.any():
                    continue
                roles = roles[in_support]
                n = roles.shape[0]
                live_here += n

                q = np.zeros((n, m))
                total = np.zeros(n)
                index = np.arange(n)
                feasibility = np.empty((n, m))
                for t in range(width):
                    for x in range(m):
                        if predecessors[x].size == 0:
                            feasibility[:, x] = 1.0          # empty product
                        else:
                            column = q[:, predecessors[x][0]].copy()
                            for z in predecessors[x][1:]:
                                column *= q[:, z]
                            feasibility[:, x] = column
                    utilities = np.log1p((1.0 - q) @ succ.T)
                    back = kappa * (q @ succ_off.T)
                    exponent = (float(beta) * utilities - float(lambda_rep) * q
                                - float(lambda_back) * back)
                    exponent -= exponent.max(axis=1, keepdims=True)
                    weights = feasibility * np.exp(exponent)
                    mixed = ((1.0 - self.epsilon)
                             * (weights / weights.sum(axis=1, keepdims=True))
                             + self.epsilon / m)
                    observed = roles[:, t]
                    total += np.log(mixed[index, observed])
                    gate = np.where(precedence[observed], kappa, 0.0)
                    q = q * (1.0 - gate)
                    q[index, observed] = 1.0

                target = np.flatnonzero(in_support) + offset
                self._table[skill, target] = total
        return live_here

    def refresh(self, u_by_skill, beta: float, omega: float, lambda_rep: float,
                lambda_back: float, skills=None) -> dict:
        """Rebuild the candidate table. `skills=None` rebuilds every skill.

        Passing an explicit `skills` rebuilds only those columns and leaves the rest
        exactly as they were. That is sound only because a skill's score column depends on
        `U[skill]` alone -- the roles, the buckets and the corpus are fixed at
        construction, and `beta`, `omega`, `lambda_rep`, `lambda_back` are shared. When
        any shared parameter moves, every column must be rebuilt; `refresh_changed` is the
        caller that gets that decision right, and this method trusts what it is told.
        """
        u_by_skill = np.asarray(u_by_skill, dtype=float)
        if u_by_skill.shape[:2] != (self.n_skills, self.n_roles):
            raise ValueError(
                f"u_by_skill must be (K={self.n_skills}, m={self.n_roles}, d), got "
                f"{u_by_skill.shape}; U is per-skill over its OWN roles, not over the "
                f"{self.role_maps.n_cpa}-symbol vocabulary")
        if skills is None:
            rebuilt = list(range(self.n_skills))
        else:
            rebuilt = sorted({int(k) for k in skills})
            if rebuilt and (rebuilt[0] < 0 or rebuilt[-1] >= self.n_skills):
                raise ValueError(f"skills out of range for K={self.n_skills}: {rebuilt}")

        for skill in rebuilt:
            self._live[skill] = self._build_skill(skill, u_by_skill[skill], beta, omega,
                                                  lambda_rep, lambda_back)

        for n, (rows, a_ix, b_ix) in enumerate(self._index):
            for skill in rebuilt:
                self.tables[n][a_ix, b_ix, skill] = self._table[skill, rows]

        self._built_u = np.array(u_by_skill, copy=True)
        self._built_params = (float(beta), float(omega), float(lambda_rep),
                              float(lambda_back))
        self.version += 1
        self.refresh_calls += 1
        self.skill_rebuilds += len(rebuilt)
        live = self._live
        total_candidates = self.n_blocks * self.n_skills
        return {"live_blocks_per_skill": live.tolist(),
                "live_block_skill_pairs": int(live.sum()),
                "candidate_block_skill_pairs": int(total_candidates),
                "rebuilt_skills": rebuilt,
                "live_fraction": float(live.sum() / total_candidates)
                if total_candidates else 0.0}

    def refresh_changed(self, u_by_skill, beta: float, omega: float, lambda_rep: float,
                        lambda_back: float) -> dict:
        """Rebuild only the skills whose `U` actually moved. Exact, not approximate.

        A skill's column is a deterministic function of `U[skill]` and the four shared
        parameters. If a shared parameter moved, every column is stale and this rebuilds
        all of them; otherwise it rebuilds precisely the skills whose `U` differs, bit for
        bit, from the `U` the table was last built at. There is no tolerance and no
        heuristic here -- a column is either built at the current inputs or it is not.
        """
        u_by_skill = np.asarray(u_by_skill, dtype=float)
        params = (float(beta), float(omega), float(lambda_rep), float(lambda_back))
        if (self._built_u is None or self._built_params != params
                or self._built_u.shape != u_by_skill.shape):
            return self.refresh(u_by_skill, *params, skills=None)
        moved = [k for k in range(self.n_skills)
                 if not np.array_equal(self._built_u[k], u_by_skill[k])]
        return self.refresh(u_by_skill, *params, skills=moved)

    # ------------------------------------------------------------------------ lookup
    def score(self, trace: int, start: int, end: int, skill: int) -> float:
        key = (int(trace), int(start), int(end) - int(start))
        if key not in self._row:
            return float("-inf")
        return float(self._table[int(skill), self._row[key]])

    def width_is_legal(self, start: int, end: int) -> bool:
        return self.min_width <= end - start <= self.max_width


def assert_matches_sealed_scorer(table: CPABlockScoreTable, u_by_skill, beta, omega,
                                 lambda_rep, lambda_back, tolerance: float = 1e-12,
                                 limit: int | None = None) -> dict:
    """Every finite entry must equal the sealed per-block scorer on the relabelled trace.

    The sealed `RecurrentBlockScorer` reads symbols as role indices, so feeding it the
    already-translated role sequence makes it the reference for the translated model. Any
    disagreement beyond floating-point noise is a defect in the translation layer, not a
    modelling choice.
    """
    from hpop.mcmc_original.recurrent_segmentation import RecurrentBlockScorer

    role_traces = tuple(
        tuple(int(v) for v in table.role_maps.to_roles(t)[skill])
        for skill in range(table.n_skills) for t in table.traces)
    n_traces = len(table.traces)

    worst, checked, out_of_support = 0.0, 0, 0
    for (n, a, width), row in sorted(table._row.items()):
        for skill in range(table.n_skills):
            got = float(table._table[skill, row])
            roles = table.role_maps.to_roles(table.traces[n])[skill, a:a + width]
            if (roles == NOT_IN_SUPPORT).any():
                if np.isfinite(got):
                    raise AssertionError(
                        f"block ({n}, {a}, {width}) under skill {skill} contains a CPA "
                        f"outside that skill's support but scored {got}, not -inf")
                out_of_support += 1
                continue
            scorer = RecurrentBlockScorer(
                traces=(role_traces[skill * n_traces + n],), epsilon=table.epsilon,
                u_by_skill=np.asarray(u_by_skill)[skill][None, ...], beta=beta,
                omega=omega, lambda_rep=lambda_rep, lambda_back=lambda_back,
                max_width=table.max_width, min_width=table.min_width)
            expected = scorer.replay(0, a, a + width, 0)
            worst = max(worst, abs(got - expected))
            checked += 1
            if limit is not None and checked >= limit:
                break
        if limit is not None and checked >= limit:
            break
    return {"in_support_blocks_checked": checked,
            "out_of_support_blocks_verified_neg_inf": out_of_support,
            "max_absolute_difference": worst, "tolerance": tolerance,
            "pass": bool(worst < tolerance)}
