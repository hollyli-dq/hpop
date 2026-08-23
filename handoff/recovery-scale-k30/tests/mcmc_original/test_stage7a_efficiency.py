"""Step 7A — the properties the efficiency claim depends on.

No wall-clock threshold is asserted here: timings belong in the result artifact, where
they are reported with the machine they were measured on. What is asserted is the
*structural* content of the claim — that draws from a fixed chart are independent, that
the chart is built once and reused, and that the block table costs exactly one replay per
legal block.
"""

from __future__ import annotations

import numpy as np

from hpop.mcmc_original.block_score_adapters import build_log_block_scores, legal_blocks
from hpop.mcmc_original.diagnostics import autocorrelation
from hpop.mcmc_original.semi_markov_ffbs import (
    SemiMarkovFFBS, backward_sample_many, forward,
)
from hpop.mcmc_original.stage6b_mcmc_diagnostics import bulk_ess
from tests.mcmc_original.test_stage7a_exact_posterior import path_prior
from tests.mcmc_original.test_stage7a_recurrent_blocks import (
    J, K, MAX_WIDTH, MIN_WIDTH, make_scorer,
)

DELTA_B = 0.15


def chart_for_the_frozen_problem():
    log_pi, log_p = path_prior()
    table = build_log_block_scores(make_scorer(), 0, J, K, MIN_WIDTH, MAX_WIDTH)
    return forward(table, log_pi, log_p, DELTA_B, MAX_WIDTH, MIN_WIDTH)


def test_draws_from_a_fixed_chart_are_serially_independent():
    chart = chart_for_the_frozen_problem()
    draws = backward_sample_many(chart, 40_000, np.random.default_rng(7_071_002))
    counts = np.array([len(d) for d in draws], dtype=float)
    acf = autocorrelation(counts, max_lag=10)
    # iid draws: every non-zero lag sits inside a 5/sqrt(N) band. The band is deliberately
    # wider than 2 sigma because the statistic is a maximum over ten lags, and it is still
    # two orders of magnitude tighter than the local kernel's lag-1 autocorrelation.
    band = 5.0 / np.sqrt(len(counts))
    assert np.abs(acf[1:]).max() < band
    # and the ESS of an iid sequence is its length, to within sampling noise
    ess = bulk_ess(counts.reshape(4, -1))
    assert ess > 0.85 * len(counts)


def test_a_boundary_indicator_is_also_serially_independent():
    chart = chart_for_the_frozen_problem()
    draws = backward_sample_many(chart, 40_000, np.random.default_rng(7_071_003))
    series = np.array([1.0 if any(b == 4 for _, b, _ in d[:-1]) else 0.0 for d in draws])
    assert 0.0 < series.mean() < 1.0                  # the statistic actually varies
    acf = autocorrelation(series, max_lag=10)
    assert np.abs(acf[1:]).max() < 5.0 / np.sqrt(len(series))


def test_the_chart_is_built_once_and_every_draw_reuses_it():
    log_pi, log_p = path_prior()
    table = build_log_block_scores(make_scorer(), 0, J, K, MIN_WIDTH, MAX_WIDTH)
    engine = SemiMarkovFFBS(table, log_pi, log_p, DELTA_B, MAX_WIDTH, MIN_WIDTH)
    chart = engine.chart()
    alpha = chart.alpha.copy()
    engine.sample_many(1_000, np.random.default_rng(3))
    assert engine.chart() is chart
    assert np.array_equal(engine.chart().alpha, alpha)


def test_the_block_table_costs_exactly_one_replay_per_legal_block():
    scorer = make_scorer()
    before = int(scorer.full_replay_calls)
    build_log_block_scores(scorer, 0, J, K, MIN_WIDTH, MAX_WIDTH)
    spent = int(scorer.full_replay_calls) - before
    assert spent == len(legal_blocks(J, K, MIN_WIDTH, MAX_WIDTH))


def test_chart_construction_scales_with_j_times_max_width_not_with_the_path_count():
    """A longer trace must not cost exponentially more — the whole point of the recursion."""
    log_pi, log_p = path_prior()
    for length in (8, 24, 48):
        rng = np.random.default_rng(length)
        trace = tuple(int(v) for v in rng.integers(3, size=length))
        scorer = make_scorer(traces=(trace,))
        table = build_log_block_scores(scorer, 0, length, K, MIN_WIDTH, MAX_WIDTH)
        chart = forward(table, log_pi, log_p, DELTA_B, MAX_WIDTH, MIN_WIDTH)
        expected_blocks = K * sum(max(0, length - w + 1)
                                  for w in range(MIN_WIDTH, MAX_WIDTH + 1))
        assert int(np.isfinite(table).sum()) == expected_blocks
        assert chart.alpha.size == (length + 1) * K
        # the same problem enumerated has combinatorially many paths; the chart does not
        assert expected_blocks <= K * length * (MAX_WIDTH - MIN_WIDTH + 1)
