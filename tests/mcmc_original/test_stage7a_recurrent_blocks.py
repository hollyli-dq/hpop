"""Step 7A — the recurrent block-score adapter, and the `q_0 = 0` guarantee.

The adapter is the only place where the model reaches the engine, so every way the table
could be wrong is checked here: recurrent state leaking across candidates, the fill order
mattering, the cache disagreeing with a fresh replay, and the table disagreeing with the
independent batched builder Stage 6E already validated.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_original.block_score_adapters import (
    ITERATION_ORDERS, assert_no_recurrent_state_leak, assert_order_invariance,
    assert_table_matches_uncached_replay, build_log_block_scores, legal_blocks,
)
from hpop.mcmc_original.recurrent_segmentation import RecurrentBlockScorer
from hpop.mcmc_original.semi_markov_ffbs import forward
from hpop.mcmc_original.stage6e_block_table import BlockScoreTable

J = 8
K = 3
M_ROLES = 3
MIN_WIDTH = 3
MAX_WIDTH = 12
EPSILON = 0.02
SCALARS = {"beta": 1.5, "omega": 1.7346, "lambda_rep": 0.8, "lambda_back": 0.25}
U_BY_SKILL = np.array([
    [[2.0, 2.0], [1.0, 0.0], [0.0, 1.0]],
    [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0]],
    [[3.0, 3.0], [2.0, 2.0], [0.0, 0.0]],
], dtype=float)
TRACE = (0, 0, 0, 1, 2, 1, 1, 1)                  # the Stage 6E1A trace


def make_scorer(traces=(TRACE,)) -> RecurrentBlockScorer:
    return RecurrentBlockScorer(traces=traces, epsilon=EPSILON, u_by_skill=U_BY_SKILL,
                                beta=SCALARS["beta"], omega=SCALARS["omega"],
                                lambda_rep=SCALARS["lambda_rep"],
                                lambda_back=SCALARS["lambda_back"],
                                min_width=MIN_WIDTH, max_width=MAX_WIDTH)


# ------------------------------------------------------------------------ q_0 = 0
@pytest.mark.parametrize("block_a,block_b", [
    ((0, 3, 0), (3, 8, 1)),
    ((0, 4, 2), (4, 8, 0)),
    ((2, 6, 1), (0, 8, 2)),
    ((0, 8, 0), (0, 3, 0)),
    ((5, 8, 1), (0, 5, 1)),
])
def test_score_a_then_b_then_a_again_is_bit_identical(block_a, block_b):
    audit = assert_no_recurrent_state_leak(make_scorer(), 0, block_a, block_b)
    assert audit["bit_identical"]
    assert audit["score_a_first"] == audit["score_a_again"]
    assert audit["difference"] == 0.0


def test_a_block_score_depends_only_on_its_own_window():
    """The same role window scored at two different offsets, in two different traces.

    `q_0 = 0` means a candidate block is a fresh execution: its score is a function of the
    window alone. If any state leaked in from the left the two would differ.
    """
    window = (1, 2, 1, 0)
    left = (0, 0) + window + (2, 2)
    right = (1, 1, 1) + window + (0,)
    scorer = make_scorer(traces=(left, right))
    for skill in range(K):
        first = scorer.replay(0, 2, 6, skill)
        second = scorer.replay(1, 3, 7, skill)
        assert first == second


# ------------------------------------------------------------------ order invariance
def test_every_pair_of_iteration_orders_gives_an_identical_table():
    audit = assert_order_invariance(make_scorer(), 0, J, K, MIN_WIDTH, MAX_WIDTH,
                                    orders=ITERATION_ORDERS)
    assert audit["pass"]
    assert audit["max_absolute_difference"] == 0.0
    assert audit["n_finite_entries"] == len(
        legal_blocks(J, K, MIN_WIDTH, MAX_WIDTH))


def test_every_legal_block_appears_exactly_once_in_every_order():
    for order in ITERATION_ORDERS:
        blocks = legal_blocks(J, K, MIN_WIDTH, MAX_WIDTH, order)
        assert len(blocks) == len(set(blocks))
        assert set(blocks) == set(legal_blocks(J, K, MIN_WIDTH, MAX_WIDTH, "by_start"))


def test_unknown_iteration_order_is_rejected():
    with pytest.raises(ValueError, match="unknown iteration order"):
        legal_blocks(J, K, MIN_WIDTH, MAX_WIDTH, "whatever")


# --------------------------------------------------------------------------- the table
def test_illegal_widths_are_minus_infinity():
    table = build_log_block_scores(make_scorer(), 0, J, K, MIN_WIDTH, MAX_WIDTH)
    assert table.shape == (J, J + 1, K)
    for a in range(J):
        for b in range(J + 1):
            width = b - a
            legal = MIN_WIDTH <= width <= MAX_WIDTH and b <= J
            for k in range(K):
                assert np.isfinite(table[a, b, k]) == legal


def test_cached_and_uncached_tables_are_identical():
    scorer = make_scorer()
    uncached = build_log_block_scores(scorer, 0, J, K, MIN_WIDTH, MAX_WIDTH,
                                      uncached=True)
    cached = build_log_block_scores(scorer, 0, J, K, MIN_WIDTH, MAX_WIDTH, uncached=False)
    # the first cached pass only fills the cache; the second reads it
    reread = build_log_block_scores(scorer, 0, J, K, MIN_WIDTH, MAX_WIDTH, uncached=False)
    assert np.array_equal(uncached, cached)
    assert np.array_equal(cached, reread)
    assert scorer.cached_calls == len(legal_blocks(J, K, MIN_WIDTH, MAX_WIDTH))


def test_table_matches_a_fresh_uncached_replay_of_every_entry():
    scorer = make_scorer()
    table = build_log_block_scores(scorer, 0, J, K, MIN_WIDTH, MAX_WIDTH)
    audit = assert_table_matches_uncached_replay(table, scorer, 0, MIN_WIDTH, MAX_WIDTH)
    assert audit["pass"]
    assert audit["max_absolute_difference"] == 0.0
    assert audit["blocks_checked"] == len(legal_blocks(J, K, MIN_WIDTH, MAX_WIDTH))


def test_table_matches_the_independent_batched_builder():
    """`BlockScoreTable` computes the same scores by a completely different loop order —
    bucketed by width, batched over starts. Agreement is a cross-implementation check."""
    scorer = make_scorer()
    table = build_log_block_scores(scorer, 0, J, K, MIN_WIDTH, MAX_WIDTH)
    batched = BlockScoreTable(traces=(TRACE,), epsilon=EPSILON, n_skills=K,
                              min_width=MIN_WIDTH, max_width=MAX_WIDTH)
    batched.refresh(U_BY_SKILL, SCALARS["beta"], SCALARS["omega"],
                    SCALARS["lambda_rep"], SCALARS["lambda_back"])
    worst = 0.0
    for a, b, k in legal_blocks(J, K, MIN_WIDTH, MAX_WIDTH):
        worst = max(worst, abs(float(table[a, b, k]) - batched.score(0, a, b, k)))
    assert worst < 1e-9


def test_a_parameter_change_changes_the_table_and_the_normaliser():
    """The table is a function of the parameters; nothing may survive a change."""
    scorer = make_scorer()
    log_pi = np.log(np.full(K, 1.0 / K))
    transition = (np.ones((K, K)) - np.eye(K)) / (K - 1)
    with np.errstate(divide="ignore"):
        log_p = np.log(transition)
    before = build_log_block_scores(scorer, 0, J, K, MIN_WIDTH, MAX_WIDTH)
    first = forward(before, log_pi, log_p, 0.15, MAX_WIDTH, MIN_WIDTH).log_normalizer

    scorer.set_parameters(beta=SCALARS["beta"] + 0.5)
    after = build_log_block_scores(scorer, 0, J, K, MIN_WIDTH, MAX_WIDTH)
    second = forward(after, log_pi, log_p, 0.15, MAX_WIDTH, MIN_WIDTH).log_normalizer
    assert not np.allclose(before[np.isfinite(before)], after[np.isfinite(after)])
    assert first != second
