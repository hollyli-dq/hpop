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

__all__ = ["SupportOnlyBlockScoreTable", "candidate_survival", "block_ambiguity",
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
                                     distinct: int, pairwise_distinct: bool = True) -> float:
    """Expected number of WRONG skills whose support contains `d` given CPAs.

    The supports of the master library are required to be pairwise distinct, so a wrong
    skill's support cannot equal the true one. Conditioning on that:

        (K-1) * [ C(A-d, m-d) - 1 ] / [ C(A, m) - 1 ]

    when the `d` CPAs come from the true support. The unconditional form differs
    negligibly at small `d` but matters at `d = m`, where it reports a small positive
    probability for an event that is **exactly impossible**: at `d = 10 = m` the only
    support containing all ten is the true one, which distinctness excludes.
    """
    from math import comb
    K, m, A, d = int(n_skills), int(n_roles), int(n_cpa), int(distinct)
    if d > m:
        return 0.0
    if not pairwise_distinct:
        return (K - 1) * comb(A - d, m - d) / comb(A, m)
    return (K - 1) * (comb(A - d, m - d) - 1) / (comb(A, m) - 1)


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
        "non_true_pair_survival_ALL_BLOCKS": pooled(false_rows, "support_only_survival"),
        "wrong_skill_survival_CONTAINED_BLOCKS_ONLY": pooled(
            [r for r in false_rows if not r["crosses_true_boundary"]],
            "support_only_survival"),
        "cross_boundary_support_survival": pooled(
            [r for r in rows if r["crosses_true_boundary"]], "support_only_survival"),
        "strata": rows,
        "NOTE": "`non_true_pair_survival_ALL_BLOCKS` counts every candidate whose skill is "
                "not the block's own, which for a boundary-crossing block is ALL of them. "
                "It therefore mixes two populations and does NOT satisfy "
                "E[C_b] = (K-1) * p. Use `wrong_skill_survival_CONTAINED_BLOCKS_ONLY` for "
                "that identity; `block_ambiguity` reports the same population.",
    }


def block_ambiguity(corpus, support_table) -> dict:
    """How many skills a block is compatible with — the quantity per-pair survival hides.

    ## Why per-pair survival is the wrong statistic

    A falling per-pair survival rate looks like sharper discrimination, but the number of
    competing skills grows as `K - 1` at the same time. What a segmentation actually faces
    is the **count of compatible skills for the block in front of it**, `C_b`. Measured on
    this design, per-pair survival falls from 0.0368 to 0.0071 between K = 3 and K = 30
    while the expected number of *false* compatible skills per block rises from 0.074 to
    0.206. Ambiguity increases with K; the per-pair figure points the other way and is
    misleading on its own.

    Reported separately for blocks contained inside one true segment and for blocks that
    cross a true boundary, because those play different roles: the first governs skill
    identification, the second governs whether the support mask forbids a wrong cut before
    any likelihood is consulted.
    """
    n_skills = int(support_table.n_skills)
    contained, crossing = [], []

    for n, record in enumerate(corpus.train):
        trace = record.cpa
        true_skill = np.empty(len(trace), dtype=np.int64)
        boundaries, cursor = set(), 0
        for width, skill in zip(record.widths, record.labels):
            true_skill[cursor:cursor + width] = skill
            cursor += width
            boundaries.add(cursor)
        boundaries.discard(len(trace))

        for width in range(support_table.min_width, support_table.max_width + 1):
            for a in range(0, len(trace) - width + 1):
                finite = np.isfinite(support_table.tables[n][a, a + width, :])
                count = int(finite.sum())
                crosses = any(a < b < a + width for b in boundaries)
                if crosses:
                    crossing.append(count)
                else:
                    owner = int(true_skill[a])
                    # false compatible skills: exclude the block's own skill
                    crossing_free = count - int(finite[owner])
                    contained.append((count, crossing_free))

    def summarise(counts):
        counts = np.asarray(counts, dtype=float)
        if counts.size == 0:
            return {}
        return {
            "n_blocks": int(counts.size),
            "mean_compatible_skills": float(counts.mean()),
            "P_zero": float((counts == 0).mean()),
            "P_exactly_one": float((counts == 1).mean()),
            "P_two_or_more": float((counts >= 2).mean()),
            "max_compatible_skills": int(counts.max()),
        }

    total = np.asarray([c for c, _ in contained], dtype=float)
    false = np.asarray([f for _, f in contained], dtype=float)
    return {
        "n_skills": n_skills,
        "blocks_inside_one_true_segment": {
            **summarise(total),
            "mean_FALSE_compatible_skills": float(false.mean()) if false.size else 0.0,
            "P_no_false_skill": float((false == 0).mean()) if false.size else 0.0,
            "P_at_least_one_false_skill": float((false >= 1).mean()) if false.size else 0.0,
        },
        "blocks_crossing_a_true_boundary": summarise(crossing),
        "NOTE": "per-pair survival falls with K while the number of competing skills grows "
                "as K-1; the block-level count is what a segmentation actually faces, and "
                "it moves the other way. Do not read falling per-pair survival as sharper "
                "discrimination.",
    }


def pairwise_compatibility(n_skills: int, n_roles: int, n_cpa: int,
                           distinct: int) -> dict:
    """Theory for one block with `d` distinct CPAs, under pairwise-distinct supports.

    Note what is **absent** from `p_d`: `K`. A single wrong skill's chance of being
    support-compatible is set by `d`, `m` and `A` alone. Only the *number* of wrong skills
    grows with `K`. So any measured fall in per-pair survival across rungs is a change in
    the block population being averaged over -- typically the mix of `d` values, or the
    share of boundary-crossing blocks -- and not `K` making an individual pair less
    compatible.
    """
    from math import comb
    K, m, A, d = int(n_skills), int(n_roles), int(n_cpa), int(distinct)
    if d > m:
        return {"distinct_cpas": d, "p_pair": 0.0, "expected_wrong": 0.0,
                "p_at_least_one_wrong": 0.0}
    p = (comb(A - d, m - d) - 1) / (comb(A, m) - 1)
    return {
        "distinct_cpas": d,
        "p_pair": p,
        "expected_wrong": (K - 1) * p,
        "p_at_least_one_wrong": 1.0 - (1.0 - p) ** (K - 1),
    }


def accounting_check(corpus, support_table) -> dict:
    """`E[C_b] = (K-1) * p_pair` must hold exactly on one shared block population.

    An earlier report placed a per-pair rate and a per-block expectation in the same table
    when they had been averaged over different populations -- the per-pair figure counted
    boundary-crossing candidates as "wrong skill" pairs, the per-block one did not. The
    identity is arithmetic, so a mismatch is always a bookkeeping error, and this makes it
    impossible to reintroduce silently.
    """
    K = int(support_table.n_skills)
    pairs = live = 0
    counts = []
    for n, record in enumerate(corpus.train):
        trace = record.cpa
        true_skill = np.empty(len(trace), dtype=np.int64)
        boundaries, cursor = set(), 0
        for width, skill in zip(record.widths, record.labels):
            true_skill[cursor:cursor + width] = skill
            cursor += width
            boundaries.add(cursor)
        boundaries.discard(len(trace))
        for width in range(support_table.min_width, support_table.max_width + 1):
            for a in range(0, len(trace) - width + 1):
                if any(a < b < a + width for b in boundaries):
                    continue                      # contained blocks only
                if true_skill[a] != true_skill[a + width - 1]:
                    continue
                finite = np.isfinite(support_table.tables[n][a, a + width, :])
                owner = int(true_skill[a])
                wrong = int(finite.sum()) - int(finite[owner])
                counts.append(wrong)
                pairs += K - 1
                live += wrong
    counts = np.asarray(counts, dtype=float)
    p_pair = live / pairs if pairs else float("nan")
    return {
        "population": "blocks contained in one true segment; wrong-skill pairs only",
        "n_blocks": int(counts.size), "n_wrong_skill_pairs": int(pairs),
        "p_pair": p_pair,
        "expected_wrong_per_block": float(counts.mean()) if counts.size else float("nan"),
        "identity_lhs_minus_rhs": float(counts.mean() - (K - 1) * p_pair)
        if counts.size else float("nan"),
        "P_at_least_one_wrong": float((counts >= 1).mean()) if counts.size else 0.0,
        "mean_wrong_GIVEN_at_least_one": float(counts[counts >= 1].mean())
        if (counts >= 1).any() else 0.0,
    }
