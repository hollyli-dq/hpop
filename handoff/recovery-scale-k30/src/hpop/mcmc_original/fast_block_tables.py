"""Step 7B2 optimisation — one recurrent trajectory per start, every width by prefix sum.

This module changes *when and in what shape* the arithmetic happens, never what the
arithmetic is. Each entry it produces is the same number the validated builders produce:

    log_block_scores[a, b, k] = log p_RFS(x[a:b] | U_k, beta, omega, lambda_rep,
                                          lambda_back, epsilon)   from q_0 = 0

and `test_stage7b2_optimisation.py` pins it against both `BlockScoreTable` (the Stage 6E
width-bucketed builder) and `RecurrentBlockScorer.replay` (the registered per-block
reference), entry for entry, on every problem shape the corpus and the tests use.

## The redundancy this removes

A block score is a sum of per-step emission terms along a recurrent trajectory that starts
at `a` with `q_0 = 0`:

    log p_RFS(x[a:b] | ...) = sum over t in [a, b) of  log p(x_t | q_t, U_k, ...)

and **nothing in that trajectory depends on `b`**. The state `q_t` is a function of
`(a, x, U_k, omega)` alone, so the candidates `[a, a+3)`, `[a, a+4)`, ..., `[a, a+12)` all
walk the *same* path and differ only in where they stop.

Every previous builder replayed that shared prefix once per width. `BlockScoreTable`
buckets by width and replays each bucket, so on the registered corpus it performs
`sum_w n_w * w` candidate-steps — about 175,000 — to fill 25,000 candidates. Here each
start is replayed **once** for `max_width` steps, the per-step emissions are accumulated
with `cumsum`, and every width is read off that cumulative row:

    score(a, a + w, k) = cumulative_emissions[row(a), w - 1]

which is about 35,000 candidate-steps for the same table: the same arithmetic per step,
performed once instead of once per width.

This is exact, not an approximation. It relies on one property of the model — that the
recurrent state is a function of the start and the observations, not of the block end —
and `test_prefix_sum_identity_holds_block_by_block` checks that property directly against
`RecurrentBlockScorer.replay` rather than assuming it.

## Why the rows are sorted by descending length

A start near the end of a trace has fewer than `max_width` steps available. Sorting the
starts by descending step count makes the set still running at step `t` a **contiguous
prefix** `[0 : n_active[t])`, so each step operates on a numpy view — no mask, no copy, no
wasted element, and a finished start is simply not in the prefix any more.

## Skill-local invalidation

`rho`, `pi` and `P` do not enter a block score, so they invalidate nothing. `beta`,
`omega`, `lambda_rep` and `lambda_back` enter every score, so they invalidate every skill
column. `U_k` enters only skill `k`'s column, so a refresh that sees only `U_k` move
recomputes column `k` and keeps the rest. The fingerprint is per skill, and the reuse
decision is made from it rather than from a float comparison of the parameters.

In the registered Stage 7B2 sweep all four scalars move on most sweeps, so this buys
little there; it is implemented because the invalidation matrix has to be right anyway, and
`reuse_fraction` reports how often it actually helped rather than assuming it did.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CandidateLayout", "FastBlockScoreTable", "layout_for"]

_LAYOUT_CACHE: dict = {}


class CandidateLayout:
    """The parameter-independent geometry of the candidate set, built once and reused.

    Everything here is a function of the traces and the width bounds — never of `U`, the
    scalars or `epsilon` — so it is computed on the first use of a given
    `(traces, min_width, max_width)` and cached for the life of the process.
    """

    __slots__ = ("traces", "min_width", "max_width", "n_roles", "n_starts", "n_candidates",
                 "roles", "n_steps", "n_active", "start_trace", "start_position",
                 "scatter", "candidate_steps", "naive_candidate_steps")

    def __init__(self, traces, min_width: int, max_width: int, n_roles: int):
        self.traces = tuple(tuple(int(v) for v in t) for t in traces)
        self.min_width = int(min_width)
        self.max_width = int(max_width)
        self.n_roles = int(n_roles)

        # one row per (trace, start) that can host at least one legal block
        starts = []
        for index, trace in enumerate(self.traces):
            length = len(trace)
            for position in range(0, length - self.min_width + 1):
                starts.append((index, position, min(self.max_width, length - position)))
        # descending step count: the starts still running at step t are a contiguous prefix
        starts.sort(key=lambda row: (-row[2], row[0], row[1]))

        self.n_starts = len(starts)
        self.start_trace = np.array([s[0] for s in starts], dtype=np.int32)
        self.start_position = np.array([s[1] for s in starts], dtype=np.int32)
        self.n_steps = np.array([s[2] for s in starts], dtype=np.int32)

        self.roles = np.zeros((self.n_starts, self.max_width), dtype=np.int64)
        for row, (index, position, steps) in enumerate(starts):
            self.roles[row, :steps] = self.traces[index][position:position + steps]

        self.n_active = np.array(
            [int((self.n_steps > t).sum()) for t in range(self.max_width)], dtype=np.int64)

        # the scatter: for each trace, the (row, width-1) -> (a, b) map that reads every
        # candidate score off the cumulative emission rows
        self.scatter = []
        n_candidates = 0
        for index in range(len(self.traces)):
            rows, widths, a_ix, b_ix = [], [], [], []
            for row in np.flatnonzero(self.start_trace == index):
                position = int(self.start_position[row])
                for width in range(self.min_width, int(self.n_steps[row]) + 1):
                    rows.append(row)
                    widths.append(width - 1)
                    a_ix.append(position)
                    b_ix.append(position + width)
            self.scatter.append((np.array(rows, dtype=np.int64),
                                 np.array(widths, dtype=np.int64),
                                 np.array(a_ix, dtype=np.int64),
                                 np.array(b_ix, dtype=np.int64)))
            n_candidates += len(rows)
        self.n_candidates = n_candidates
        self.candidate_steps = int(self.n_steps.sum())
        # what a width-bucketed builder would have to do for the same table
        self.naive_candidate_steps = int(sum(
            (int(self.n_steps[row]) - self.min_width + 1) * 0 + sum(
                range(self.min_width, int(self.n_steps[row]) + 1))
            for row in range(self.n_starts)))


def layout_for(traces, min_width: int, max_width: int, n_roles: int) -> CandidateLayout:
    """The layout for this trace set and width bound, built at most once per process."""
    key = (tuple(tuple(int(v) for v in t) for t in traces), int(min_width),
           int(max_width), int(n_roles))
    layout = _LAYOUT_CACHE.get(key)
    if layout is None:
        layout = CandidateLayout(traces, min_width, max_width, n_roles)
        _LAYOUT_CACHE[key] = layout
    return layout


class FastBlockScoreTable:
    """Dense per-trace `(J, J+1, K)` candidate scores, one trajectory per start."""

    def __init__(self, traces, epsilon: float, n_skills: int, min_width: int,
                 max_width: int, n_roles: int | None = None):
        self.epsilon = float(epsilon)
        self.n_skills = int(n_skills)
        self.min_width = int(min_width)
        self.max_width = int(max_width)
        self.traces = tuple(tuple(int(v) for v in t) for t in traces)
        roles = int(n_roles) if n_roles is not None else int(
            max(max(t) for t in self.traces) + 1)
        self.layout = layout_for(self.traces, min_width, max_width, roles)

        # the dense tables are allocated once and written in place; the -inf entries are
        # written at construction and never touched again, because legality is a property
        # of the widths rather than of the parameters
        self.tables = [np.full((len(t), len(t) + 1, self.n_skills), -np.inf)
                       for t in self.traces]
        self._cumulative = np.zeros((self.n_skills, self.layout.n_starts,
                                     self.layout.max_width))
        self._fingerprint: list = [None] * self.n_skills
        self.refreshes = 0
        self.skill_columns_built = 0
        self.skill_columns_reused = 0

    # -------------------------------------------------------------- the batched replay
    def _emissions_for_skill(self, u_k, beta: float, omega: float, lambda_rep: float,
                             lambda_back: float) -> np.ndarray:
        """Per-step emission log probabilities, `(n_starts, max_width)`.

        One trajectory per start, all starts advanced together. Row `i`, column `t` is the
        log probability of the observation at step `t` of the trajectory that begins at
        that start — the quantity every candidate beginning there shares.
        """
        layout = self.layout
        u_k = np.asarray(u_k, dtype=float)
        m = u_k.shape[0]
        kappa = 1.0 / (1.0 + np.exp(-float(omega)))

        precedence = np.all(u_k[:, None, :] > u_k[None, :, :], axis=2)
        succ = precedence.astype(float)
        succ_off = succ.copy()
        np.fill_diagonal(succ_off, 0.0)
        predecessors = [np.flatnonzero(precedence[:, x]) for x in range(m)]

        n_total = layout.n_starts
        index = np.arange(n_total)
        q = np.zeros((n_total, m))
        feasibility = np.empty((n_total, m))
        emissions = np.zeros((n_total, layout.max_width))

        for t in range(layout.max_width):
            n = int(layout.n_active[t])
            if n == 0:
                break
            # rows [0, n) are exactly the starts still inside the trace, because the
            # layout is sorted by descending step count. Every array below is a view.
            q_t = q[:n]
            rows = index[:n]
            for x in range(m):
                if predecessors[x].size == 0:
                    feasibility[:n, x] = 1.0
                else:
                    column = q_t[:, predecessors[x][0]].copy()
                    for z in predecessors[x][1:]:
                        column *= q_t[:, z]
                    feasibility[:n, x] = column
            utilities = np.log1p((1.0 - q_t) @ succ.T)
            back = kappa * (q_t @ succ_off.T)
            exponent = (float(beta) * utilities - float(lambda_rep) * q_t
                        - float(lambda_back) * back)
            exponent -= exponent.max(axis=1, keepdims=True)
            weights = feasibility[:n] * np.exp(exponent)
            mixed = ((1.0 - self.epsilon) * (weights / weights.sum(axis=1, keepdims=True))
                     + self.epsilon / m)
            observed = layout.roles[:n, t]
            emissions[:n, t] = np.log(mixed[rows, observed])
            gate = np.where(precedence[observed], kappa, 0.0)
            q_t *= (1.0 - gate)                      # in place, on the active prefix only
            q_t[rows, observed] = 1.0
        return emissions

    # ------------------------------------------------------------------------ lifecycle
    @staticmethod
    def _skill_fingerprint(u_k, beta, omega, lambda_rep, lambda_back) -> tuple:
        return (np.asarray(u_k, dtype=float).tobytes(), float(beta), float(omega),
                float(lambda_rep), float(lambda_back))

    def refresh(self, u_by_skill, beta: float, omega: float, lambda_rep: float,
                lambda_back: float) -> dict:
        """Rebuild the columns whose parameters moved; keep the rest.

        Returns which skills were rebuilt, so the caller can record reuse rather than
        assume it. A skill is reused only when its complete fingerprint — its own `U_k`
        and all four global scalars — is unchanged.
        """
        u_by_skill = np.asarray(u_by_skill, dtype=float)
        rebuilt, reused = [], []
        for skill in range(self.n_skills):
            fingerprint = self._skill_fingerprint(u_by_skill[skill], beta, omega,
                                                  lambda_rep, lambda_back)
            if self._fingerprint[skill] == fingerprint:
                reused.append(skill)
                self.skill_columns_reused += 1
                continue
            emissions = self._emissions_for_skill(u_by_skill[skill], beta, omega,
                                                  lambda_rep, lambda_back)
            # every width, from one trajectory: score(a, a+w) = sum of the first w steps
            np.cumsum(emissions, axis=1, out=self._cumulative[skill])
            self._fingerprint[skill] = fingerprint
            rebuilt.append(skill)
            self.skill_columns_built += 1

        if rebuilt:
            for index, (rows, widths, a_ix, b_ix) in enumerate(self.layout.scatter):
                if rows.size == 0:
                    continue
                table = self.tables[index]
                for skill in rebuilt:
                    table[a_ix, b_ix, skill] = self._cumulative[skill][rows, widths]
        self.refreshes += 1
        return {"rebuilt_skills": rebuilt, "reused_skills": reused}

    # ------------------------------------------------------------------------ accessors
    def score(self, trace: int, start: int, end: int, skill: int) -> float:
        """Same signature as the registered scorer, so the target stays duck-typed."""
        return float(self.tables[int(trace)][int(start), int(end), int(skill)])

    @property
    def reuse_fraction(self) -> float:
        total = self.skill_columns_built + self.skill_columns_reused
        return float(self.skill_columns_reused / total) if total else 0.0

    def stats(self) -> dict:
        layout = self.layout
        return {"refreshes": int(self.refreshes),
                "skill_columns_built": int(self.skill_columns_built),
                "skill_columns_reused": int(self.skill_columns_reused),
                "reuse_fraction": self.reuse_fraction,
                "n_candidates": int(layout.n_candidates),
                "n_starts": int(layout.n_starts),
                "candidate_steps_per_skill": int(layout.candidate_steps),
                "width_bucketed_candidate_steps_per_skill": int(
                    layout.naive_candidate_steps),
                "arithmetic_reduction": (layout.naive_candidate_steps
                                         / max(1, layout.candidate_steps)),
                "n_traces": len(self.traces), "n_skills": int(self.n_skills)}
