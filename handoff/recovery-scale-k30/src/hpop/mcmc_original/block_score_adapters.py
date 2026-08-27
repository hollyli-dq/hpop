"""Step 7A — turning a per-block scorer into the dense table FFBS consumes.

`semi_markov_ffbs` is deliberately model-blind: it wants `log_block_scores[a, b, k]` and
three prior pieces. This module is the only place where a *model* meets that interface,
and it exists so the engine never has to know what a recurrent block score is.

## The one invariant that matters

Every candidate block `[a, b)_k` is scored **from `q_0 = 0`**. A candidate block is a
hypothesis that a skill execution begins at `a`; letting the recurrent state leak in from
the block to the left would score a different model, and would make the table depend on
the order in which its entries happened to be filled. `RecurrentBlockScorer.replay`
rebuilds the trajectory from zero on every call, and `assert_no_recurrent_state_leak`
below turns that into evidence rather than a claim: it scores A, then B, then A again, and
requires both A scores to be bit-identical.

Order invariance is checked the same way — `build_log_block_scores` accepts an explicit
iteration order, and `assert_order_invariance` requires the complete tables from two
different legal orders to be identical, not merely close.

Illegal blocks are `-inf`, which is how the width bounds reach an engine that only knows
`max_width`: a block narrower than `min_width` is simply an entry the recursion finds at
`-inf`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ITERATION_ORDERS", "build_log_block_scores", "legal_blocks",
    "assert_no_recurrent_state_leak", "assert_order_invariance",
    "assert_table_matches_uncached_replay",
]

ITERATION_ORDERS = ("by_start", "by_width", "by_skill_then_start", "reverse_by_start")


def legal_blocks(trace_length: int, n_skills: int, min_width: int, max_width: int,
                 order: str = "by_start") -> list:
    """Every `(a, b, k)` with a legal width, in one of several equivalent orders."""
    J, K = int(trace_length), int(n_skills)
    lo, hi = int(min_width), int(max_width)
    if order == "by_start":
        out = [(a, b, k) for a in range(J)
               for b in range(a + lo, min(J, a + hi) + 1) for k in range(K)]
    elif order == "by_width":
        out = [(a, a + w, k) for w in range(lo, hi + 1)
               for a in range(0, J - w + 1) for k in range(K)]
    elif order == "by_skill_then_start":
        out = [(a, b, k) for k in range(K) for a in range(J)
               for b in range(a + lo, min(J, a + hi) + 1)]
    elif order == "reverse_by_start":
        out = [(a, b, k) for a in range(J - 1, -1, -1)
               for b in range(min(J, a + hi), a + lo - 1, -1) for k in range(K - 1, -1, -1)]
    else:
        raise ValueError(f"unknown iteration order {order!r}; "
                         f"expected one of {ITERATION_ORDERS}")
    return out


def build_log_block_scores(scorer, trace_index: int, trace_length: int, n_skills: int,
                           min_width: int, max_width: int, order: str = "by_start",
                           uncached: bool = True) -> np.ndarray:
    """`(J, J+1, K)` of block log scores; `-inf` wherever the width bound forbids a block.

    `uncached=True` calls `scorer.replay`, which always replays from `q_0 = 0` and never
    consults a cache, so the table cannot inherit anything from an earlier evaluation.
    `uncached=False` calls `scorer.score`, the versioned-cache path the sampler uses; the
    two must agree exactly, and `test_cached_and_uncached_tables_are_identical` requires it.
    """
    J, K = int(trace_length), int(n_skills)
    table = np.full((J, J + 1, K), -np.inf)
    evaluate = scorer.replay if uncached else scorer.score
    for a, b, k in legal_blocks(J, K, min_width, max_width, order):
        table[a, b, k] = float(evaluate(int(trace_index), int(a), int(b), int(k)))
    return table


# ------------------------------------------------------------------------- the audits
def assert_no_recurrent_state_leak(scorer, trace_index: int, block_a: tuple,
                                   block_b: tuple) -> dict:
    """Score A, score B, score A again. Both A scores must be bit-identical.

    If any recurrent state survived a call, the second A would differ from the first: B
    ends in a different `q`. Bit equality is the right requirement here rather than a
    tolerance, because the two calls run identical arithmetic on identical inputs.
    """
    a_start, a_end, a_skill = (int(v) for v in block_a)
    b_start, b_end, b_skill = (int(v) for v in block_b)
    first = float(scorer.replay(int(trace_index), a_start, a_end, a_skill))
    other = float(scorer.replay(int(trace_index), b_start, b_end, b_skill))
    again = float(scorer.replay(int(trace_index), a_start, a_end, a_skill))
    return {"block_a": [a_start, a_end, a_skill], "block_b": [b_start, b_end, b_skill],
            "score_a_first": first, "score_b": other, "score_a_again": again,
            "difference": abs(first - again),
            "bit_identical": bool(first == again),
            "pass": bool(first == again)}


def assert_order_invariance(scorer, trace_index: int, trace_length: int, n_skills: int,
                            min_width: int, max_width: int,
                            orders=("by_start", "by_width")) -> dict:
    """Two legal iteration orders must produce the *identical* table, entry for entry."""
    tables = {order: build_log_block_scores(scorer, trace_index, trace_length, n_skills,
                                            min_width, max_width, order=order)
              for order in orders}
    reference = tables[orders[0]]
    worst, identical = 0.0, True
    for order in orders[1:]:
        other = tables[order]
        finite = np.isfinite(reference) & np.isfinite(other)
        worst = max(worst, float(np.abs(reference[finite] - other[finite]).max())
                    if finite.any() else 0.0)
        identical &= bool(np.array_equal(reference, other, equal_nan=False))
        identical &= bool((np.isfinite(reference) == np.isfinite(other)).all())
    return {"orders": list(orders), "max_absolute_difference": worst,
            "identical": identical, "pass": identical,
            "n_finite_entries": int(np.isfinite(reference).sum())}


def assert_table_matches_uncached_replay(table, scorer, trace_index: int,
                                         min_width: int, max_width: int,
                                         tolerance: float = 0.0) -> dict:
    """Every finite entry of `table` must equal a fresh uncached replay of that block."""
    J, _, K = table.shape
    worst, checked = 0.0, 0
    for a in range(J):
        for b in range(a + int(min_width), min(J, a + int(max_width)) + 1):
            for k in range(K):
                value = float(table[a, b, k])
                if not np.isfinite(value):
                    continue
                worst = max(worst, abs(value - float(
                    scorer.replay(int(trace_index), a, b, k))))
                checked += 1
    return {"blocks_checked": checked, "max_absolute_difference": worst,
            "tolerance": tolerance, "pass": bool(worst <= tolerance)}
