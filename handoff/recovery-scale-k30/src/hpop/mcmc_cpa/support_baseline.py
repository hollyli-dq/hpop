"""The support-only baseline, and candidate survival stratified by what drives it.

## Why this baseline is not optional

With `A = 50` and `m = 10`, the probability that a *wrong* skill's support happens to
contain `d` given distinct CPAs is `C(A-d, m-d) / C(A, m)`. At `K = 30` the expected number
of the 29 wrong skills that remain support-compatible is

    d = 1   5.80        d = 3   0.18
    d = 2   1.07        d = 4   0.03

So a block with three distinct actions is very nearly identified by its support alone. If
the ladder reports strong recovery at `K = 30` without this baseline, the result cannot be
attributed to partial-order inference: support matching may have done the work, and the
recurrent likelihood only refined an answer that was already almost unique.

The baseline is the same semi-Markov machinery with the recurrent score replaced by

    0      if every CPA in the block lies in skill k's support
    -inf   otherwise

so any difference between it and the full model is exactly what the partial-order component
contributes. It also isolates a second effect: the `-inf` support rule alone forbids many
cross-boundary candidates, which shapes segmentation before any likelihood is consulted.
"""

from __future__ import annotations

import numpy as np

from hpop.mcmc_original.stage6e_frozen import MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH

from .role_maps import NOT_IN_SUPPORT, RoleMaps

__all__ = ["SupportOnlyBlockScoreTable", "candidate_survival",
           "expected_compatible_wrong_skills"]


class SupportOnlyBlockScoreTable:
    """`0` for a support-compatible block, `-inf` otherwise. Same layout as the full table.

    A flat score inside the support is deliberate: it removes every contribution of the
    partial order, the recurrent state and the utilities, leaving only "could this skill
    have emitted these actions at all". Paired with the ordinary segmentation prior and
    transition matrix, it is a complete, runnable model.
    """

    def __init__(self, traces, role_maps: RoleMaps,
                 min_width: int = MIN_BLOCK_WIDTH, max_width: int = MAX_BLOCK_WIDTH):
        self.traces = tuple(tuple(int(v) for v in t) for t in traces)
        self.role_maps = role_maps
        self.n_skills = int(role_maps.n_skills)
        self.min_width, self.max_width = int(min_width), int(max_width)
        self._roles = [role_maps.to_roles(t) for t in self.traces]
        self.tables = [np.full((len(t), len(t) + 1, self.n_skills), -np.inf)
                       for t in self.traces]
        self.live = np.zeros(self.n_skills, dtype=np.int64)
        self._build()

    def _build(self) -> None:
        for n, trace in enumerate(self.traces):
            roles = self._roles[n]
            for width in range(self.min_width, self.max_width + 1):
                for a in range(0, len(trace) - width + 1):
                    window = roles[:, a:a + width]
                    compatible = (window != NOT_IN_SUPPORT).all(axis=1)
                    self.tables[n][a, a + width, compatible] = 0.0
                    self.live += compatible.astype(np.int64)

    def score(self, trace: int, start: int, end: int, skill: int) -> float:
        return float(self.tables[int(trace)][int(start), int(end), int(skill)])


def expected_compatible_wrong_skills(n_skills: int, n_roles: int, n_cpa: int,
                                     distinct: int) -> float:
    """`(K-1) * C(A-d, m-d) / C(A, m)` — how many wrong skills survive on support alone."""
    from math import comb
    if distinct > n_roles:
        return 0.0
    return (n_skills - 1) * comb(n_cpa - distinct, n_roles - distinct) / comb(n_cpa,
                                                                              n_roles)


def candidate_survival(corpus, full_table, support_table) -> dict:
    """Survival of every candidate block-skill pair, stratified by what actually drives it.

    A single pooled survival figure hides the mechanism. Every candidate is labelled by
    block length, number of distinct CPAs, whether the skill is the true one for that
    block, and whether the block crosses a true boundary -- then survival is reported
    within each stratum for both models.

    `corpus` supplies the latent labels and boundaries, so this runs only after unsealing.
    """
    n_skills = int(support_table.n_skills)
    strata: dict = {}

    for n, trace_record in enumerate(corpus.train):
        trace = trace_record.cpa
        # true skill covering each position, and the set of true boundary positions
        true_skill = np.empty(len(trace), dtype=np.int64)
        boundaries, cursor = set(), 0
        for width, skill in zip(trace_record.widths, trace_record.labels):
            true_skill[cursor:cursor + width] = skill
            cursor += width
            boundaries.add(cursor)
        boundaries.discard(len(trace))

        for width in range(support_table.min_width, support_table.max_width + 1):
            for a in range(0, len(trace) - width + 1):
                block = trace[a:a + width]
                distinct = len(set(block))
                crosses = any(a < b < a + width for b in boundaries)
                exact = (not crosses) and true_skill[a] == true_skill[a + width - 1]
                for k in range(n_skills):
                    is_true = bool(exact and k == int(true_skill[a]))
                    key = (width, distinct, is_true, crosses)
                    row = strata.setdefault(key, {"candidates": 0, "support_live": 0,
                                                  "full_live": 0})
                    row["candidates"] += 1
                    row["support_live"] += int(
                        np.isfinite(support_table.tables[n][a, a + width, k]))
                    row["full_live"] += int(
                        np.isfinite(full_table.tables[n][a, a + width, k]))

    rows = []
    for (width, distinct, is_true, crosses), counts in sorted(strata.items()):
        rows.append({
            "block_length": width, "distinct_cpas": distinct,
            "true_skill_pair": is_true, "crosses_true_boundary": crosses,
            "candidates": counts["candidates"],
            "support_only_survival": counts["support_live"] / counts["candidates"],
            "full_model_survival": counts["full_live"] / counts["candidates"],
        })

    total = sum(r["candidates"] for r in rows)
    true_rows = [r for r in rows if r["true_skill_pair"]]
    false_rows = [r for r in rows if not r["true_skill_pair"]]

    def pooled(subset, field):
        n = sum(r["candidates"] for r in subset)
        return (sum(r[field] * r["candidates"] for r in subset) / n) if n else float("nan")

    return {
        "n_skills": n_skills,
        "total_candidate_pairs": total,
        "pooled_support_only_survival": pooled(rows, "support_only_survival"),
        "pooled_full_model_survival": pooled(rows, "full_model_survival"),
        "true_pair_support_survival": pooled(true_rows, "support_only_survival"),
        "false_pair_support_survival": pooled(false_rows, "support_only_survival"),
        "cross_boundary_support_survival": pooled(
            [r for r in rows if r["crosses_true_boundary"]], "support_only_survival"),
        "strata": rows,
        "NOTE": "a pooled survival figure is not enough: the support rule alone can nearly "
                "identify a skill once a block holds three distinct CPAs, so recovery must "
                "be compared against the support-only baseline before it is attributed to "
                "the partial-order component",
    }
