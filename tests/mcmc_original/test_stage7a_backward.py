"""Step 7A — backward sampling: legality, exactness, and the joint predecessor draw.

The gate that matters is distributional: draws from the chart must reproduce the exact
path distribution obtained by brute-force enumeration. Two structural properties are
tested alongside it because they fail in ways a coarse frequency check can absorb —
sampling `a` and `h` independently, and reading a predecessor set that differs from the
one the forward pass summed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import logsumexp

from hpop.mcmc_original.semi_markov_ffbs import (
    SemiMarkovFFBS, backward_sample, backward_sample_many, forward, predecessor_terms,
)
from tests.mcmc_original.test_stage7a_forward import (
    all_segmentations, brute_force_log_z, random_problem, segmentation_log_weight,
)


def exact_path_distribution(table, log_pi, log_p, delta_b, J, K, min_width, max_width):
    blocks = list(all_segmentations(J, K, min_width, max_width))
    weights = np.array([segmentation_log_weight(b, table, log_pi, log_p, delta_b, J)
                        for b in blocks])
    return blocks, np.exp(weights - logsumexp(weights))


def empirical_distribution(draws, blocks):
    index = {b: i for i, b in enumerate(blocks)}
    counts = np.zeros(len(blocks))
    for draw in draws:
        counts[index[draw]] += 1.0
    return counts / counts.sum()


# --------------------------------------------------------------------------- legality
@pytest.mark.parametrize("seed,J,K,min_width,max_width", [
    (30, 8, 3, 3, 12), (31, 9, 2, 2, 4), (32, 6, 4, 1, 3),
])
def test_every_draw_is_a_legal_segmentation(seed, J, K, min_width, max_width):
    table, log_pi, log_p = random_problem(seed, J, K, min_width, max_width)
    chart = forward(table, log_pi, log_p, 0.15, max_width, min_width)
    rng = np.random.default_rng(seed)
    for draw in backward_sample_many(chart, 300, rng):
        assert draw[0][0] == 0 and draw[-1][1] == J           # covers [0, J)
        for (_, b, _), (a2, _, _) in zip(draw[:-1], draw[1:]):
            assert b == a2                                     # contiguous, no gaps
        for a, b, k in draw:
            assert min_width <= b - a <= max_width
            assert np.isfinite(table[a, b, k])                 # never a forbidden block
        for (_, _, left), (_, _, right) in zip(draw[:-1], draw[1:]):
            assert np.isfinite(log_p[left, right])             # never a forbidden transition
        assert np.isfinite(log_pi[draw[0][2]])                 # never a forbidden start


def test_draws_are_chronological():
    table, log_pi, log_p = random_problem(33, 9, 3, 2, 4)
    chart = forward(table, log_pi, log_p, 0.15, 4, 2)
    rng = np.random.default_rng(33)
    for draw in backward_sample_many(chart, 100, rng):
        assert list(draw) == sorted(draw, key=lambda block: block[0])


# ------------------------------------------------------------------------- exactness
@pytest.mark.parametrize("seed,J,K,min_width,max_width,n_draws", [
    (40, 6, 2, 2, 3, 60_000),
    (41, 8, 3, 3, 12, 60_000),
    (42, 7, 2, 1, 3, 60_000),
])
def test_empirical_distribution_matches_exact_enumeration(seed, J, K, min_width,
                                                          max_width, n_draws):
    table, log_pi, log_p = random_problem(seed, J, K, min_width, max_width)
    chart = forward(table, log_pi, log_p, 0.15, max_width, min_width)
    blocks, exact = exact_path_distribution(table, log_pi, log_p, 0.15, J, K,
                                            min_width, max_width)
    rng = np.random.default_rng(seed + 900)
    draws = backward_sample_many(chart, n_draws, rng)
    empirical = empirical_distribution(draws, blocks)
    total_variation = 0.5 * float(np.abs(empirical - exact).sum())
    # a crude but honest bound: TV error of a multinomial with |support| cells
    assert total_variation < 6.0 * math.sqrt(len(blocks) / n_draws)
    assert total_variation < 0.01


def test_a_and_h_are_drawn_jointly_not_independently():
    """A table where the start and the previous skill are strongly dependent.

    If the sampler drew `a` from its marginal and then `h` from its own marginal, the
    joint frequency of `(a, h)` would factorise. Here `(a=3, h=0)` and `(a=5, h=1)` carry
    almost all the mass, so the product of the marginals would put about a quarter of the
    mass on `(a=3, h=1)`, which the correct sampler never visits.
    """
    J, K, delta = 8, 2, 0.15
    table = np.full((J, J + 1, K), -np.inf)
    # two prefixes of very different shapes, then one final block from each
    table[0, 3, 0] = 0.0                      # [0,3) skill 0
    table[0, 5, 1] = 0.0                      # [0,5) skill 1
    table[3, 8, 1] = 0.0                      # [3,8) skill 1, only reachable after h=0
    table[5, 8, 0] = 0.0                      # [5,8) skill 0, only reachable after h=1
    log_pi = np.log([0.5, 0.5])
    with np.errstate(divide="ignore"):
        log_p = np.log([[0.0, 1.0], [1.0, 0.0]])
    chart = forward(table, log_pi, log_p, delta, 5, 3)
    rng = np.random.default_rng(44)
    seen = {(draw[0][1], draw[0][2]) for draw in backward_sample_many(chart, 2_000, rng)}
    assert seen == {(3, 0), (5, 1)}           # never (3, 1) or (5, 0)


def test_backward_reads_the_same_predecessor_set_the_forward_pass_summed():
    """Every option the sampler can choose must be a term of `alpha[b, k]`, and the
    weights it normalises must be those terms exactly."""
    J, K, min_width, max_width, delta = 8, 3, 2, 4, 0.15
    table, log_pi, log_p = random_problem(45, J, K, min_width, max_width)
    chart = forward(table, log_pi, log_p, delta, max_width, min_width)
    log_db, log_1mdb = math.log(delta), math.log1p(-delta)
    for b in range(min_width, J + 1):
        for k in range(K):
            starts, prev, terms = predecessor_terms(
                chart.alpha, b, k, table, log_pi, log_p, log_db, log_1mdb, max_width,
                min_width)
            if terms.size == 0:
                continue
            # each term rebuilt from the definition, independently of the helper
            for a, h, value in zip(starts, prev, terms):
                if h < 0:
                    expected = (float(log_pi[k]) + float(table[0, b, k])
                                + (b - 1) * log_1mdb)
                else:
                    expected = (float(chart.alpha[a, h]) + log_db + float(log_p[h, k])
                                + float(table[a, b, k]) + (b - a - 1) * log_1mdb)
                assert abs(float(value) - expected) < 1e-12
            assert abs(float(logsumexp(terms)) - float(chart.alpha[b, k])) < 1e-12


# ---------------------------------------------------------------------------- plumbing
def test_the_same_seed_gives_the_same_draws():
    table, log_pi, log_p = random_problem(46, 8, 3, 2, 4)
    chart = forward(table, log_pi, log_p, 0.15, 4, 2)
    first = backward_sample_many(chart, 200, np.random.default_rng(7))
    second = backward_sample_many(chart, 200, np.random.default_rng(7))
    assert first == second
    third = backward_sample_many(chart, 200, np.random.default_rng(8))
    assert first != third


def test_an_explicit_generator_is_required():
    table, log_pi, log_p = random_problem(47, 6, 2, 2, 3)
    chart = forward(table, log_pi, log_p, 0.15, 3, 2)
    with pytest.raises(TypeError, match="Generator"):
        backward_sample(chart, 12345)
    with pytest.raises(TypeError, match="Generator"):
        backward_sample(chart, np.random.RandomState(0))       # legacy global-style state


def test_sampling_does_not_mutate_the_chart():
    table, log_pi, log_p = random_problem(48, 8, 3, 3, 12)
    chart = forward(table, log_pi, log_p, 0.15, 12, 3)
    before = chart.alpha.copy()
    log_z = chart.log_normalizer
    backward_sample_many(chart, 500, np.random.default_rng(1))
    assert np.array_equal(chart.alpha, before)
    assert chart.log_normalizer == log_z


def test_facade_builds_the_chart_once_and_reuses_it():
    table, log_pi, log_p = random_problem(49, 8, 3, 3, 12)
    engine = SemiMarkovFFBS(table, log_pi, log_p, 0.15, 12, 3)
    first = engine.chart()
    assert engine.chart() is first
    expected = brute_force_log_z(table, log_pi, log_p, 0.15, 8, 3, 3, 12)
    assert abs(engine.log_normalizer - expected) < 1e-12
    draws = engine.sample_many(50, np.random.default_rng(2))
    assert len(draws) == 50
