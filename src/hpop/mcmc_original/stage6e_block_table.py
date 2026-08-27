"""Stage 6E — every candidate block score for the current parameters, in one batch.

`RecurrentBlockScorer` is the registered per-block scorer and stays the reference. It is
the right shape for a *proposal*: a split touches two blocks, so scoring two blocks is what
the move needs. But the Stage 6E2 corpus makes that shape expensive, because each call pays
NumPy's per-call overhead on an array with a single row.

Within one sweep the segmentation phase runs at **fixed** parameters — the registered sweep
order updates `(S, z)` first and only then touches `(pi, P)`, `U`, `rho` and the scalars. So
every candidate block score the phase could possibly ask for is determined before the phase
begins, and there are only

    sum over traces of (J_n - w + 1) for each legal width w, times K skills

of them: about 10^5 for the registered corpus, which one batched replay computes in tens of
milliseconds. Every proposal then costs a dictionary lookup instead of a replay.

This changes *when* the arithmetic happens, never *what* it is:

* every block is still replayed from `q_0 = 0`, along a batch axis rather than alone;
* blocks are bucketed by width so the replay stays rectangular, and no state crosses a
  block, a trace, a skill or a bucket;
* the table is written only by `refresh`, never by a lookup, so a rejected proposal cannot
  leave anything behind;
* `assert_table_matches_scorer` compares every entry against `RecurrentBlockScorer.replay`,
  and the Stage 6E tests run it.

The table is rebuilt at the start of every segmentation phase, which is the same
invalidation rule the versioned scorer uses — any parameter change discards everything.
"""

from __future__ import annotations

import numpy as np

from hpop.mcmc_original.stage6e_frozen import MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH

__all__ = ["BlockScoreTable", "assert_table_matches_scorer"]


class BlockScoreTable:
    """`log p_RFS(x[a:b) | U_k, scalars, epsilon)` for every legal candidate block."""

    def __init__(self, traces, epsilon: float, n_skills: int,
                 min_width: int = MIN_BLOCK_WIDTH, max_width: int = MAX_BLOCK_WIDTH):
        self.traces = tuple(tuple(int(v) for v in t) for t in traces)
        self.epsilon = float(epsilon)
        self.n_skills = int(n_skills)
        self.min_width = int(min_width)
        self.max_width = int(max_width)
        self.version = 0
        self.refresh_calls = 0

        # bucket every legal (trace, start, width) by width, once
        self._buckets: dict = {}
        self._row: dict = {}
        offset = 0
        for width in range(self.min_width, self.max_width + 1):
            rows = [(n, a) for n, trace in enumerate(self.traces)
                    for a in range(0, len(trace) - width + 1)]
            if not rows:
                continue
            self._buckets[width] = (
                np.array([self.traces[n][a:a + width] for n, a in rows], dtype=int),
                offset)
            for index, (n, a) in enumerate(rows):
                self._row[(n, a, width)] = offset + index
            offset += len(rows)
        self.n_blocks = offset
        self._table = np.full((self.n_skills, offset), np.nan)

    # -- the only writer -----------------------------------------------------------------
    def refresh(self, u_by_skill, beta: float, omega: float, lambda_rep: float,
                lambda_back: float) -> None:
        u_by_skill = np.asarray(u_by_skill, dtype=float)
        kappa = 1.0 / (1.0 + np.exp(-float(omega)))
        for skill in range(self.n_skills):
            u = u_by_skill[skill]
            m = u.shape[0]
            precedence = np.all(u[:, None, :] > u[None, :, :], axis=2)
            succ = precedence.astype(float)
            succ_off = succ.copy()
            np.fill_diagonal(succ_off, 0.0)
            # Predecessor lists rather than a mask. `F[b, x]` is the product of `q[b, z]`
            # over the predecessors of `x`, and the registered order has about six
            # relations over five roles — so this is roughly six elementwise products on
            # (n,) arrays, against a materialised (n, m, m) temporary and a reduction over
            # its last axis. Same numbers, and the profile showed the reduction was a
            # quarter of the whole refresh.
            predecessors = [np.flatnonzero(precedence[:, x]) for x in range(m)]
            for width, (roles, offset) in self._buckets.items():
                n = roles.shape[0]
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
                self._table[skill, offset:offset + n] = total
        self.version += 1
        self.refresh_calls += 1

    # -- lookup ---------------------------------------------------------------------------
    def score(self, trace: int, start: int, end: int, skill: int) -> float:
        """Same signature as `RecurrentBlockScorer.score`, so the target is duck-typed."""
        return float(self._table[skill, self._row[(int(trace), int(start),
                                                   int(end) - int(start))]])

    def width_is_legal(self, start: int, end: int) -> bool:
        return self.min_width <= end - start <= self.max_width

    @property
    def n_skills_(self) -> int:                      # parity with the scorer's property
        return self.n_skills


def assert_table_matches_scorer(table: BlockScoreTable, scorer,
                                tolerance: float = 1e-9, limit: int | None = None) -> dict:
    """Every table entry must equal an uncached `RecurrentBlockScorer` replay.

    The two compute the same per-block quantity by the same equations; only the loop order
    differs, so any disagreement beyond floating-point noise is a real defect. `limit`
    samples the block set when it is large, and the sample is deterministic (the first
    `limit` blocks in the table's own fixed order), never random.
    """
    worst = 0.0
    checked = 0
    for (n, a, width), row in sorted(table._row.items()):
        for skill in range(table.n_skills):
            worst = max(worst, abs(float(table._table[skill, row])
                                   - scorer.replay(n, a, a + width, skill)))
            checked += 1
        if limit is not None and checked >= limit:
            break
    return {"blocks_checked": checked, "n_blocks_total": table.n_blocks * table.n_skills,
            "max_absolute_difference": worst, "tolerance": tolerance,
            "pass": bool(worst < tolerance)}
