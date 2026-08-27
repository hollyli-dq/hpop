"""Step 7A — the generic forward recursion, against brute force and at the edges.

Everything here works on *synthetic* block tables. The engine is model-agnostic, so its
tests should be too: if a check needs a recurrent scorer to state, it belongs in
`test_stage7a_recurrent_blocks.py` or `test_stage7a_exact_posterior.py` instead.

`brute_force_log_z` below enumerates segmentations and sums their weights directly from
the definition. It shares no line with `semi_markov_ffbs`, which is the point.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from hpop.mcmc_original.semi_markov_ffbs import (
    ForwardChart, forward, posterior_log_marginals, predecessor_terms,
)


# ------------------------------------------------------------------ independent oracle
def all_segmentations(J: int, K: int, min_width: int, max_width: int):
    """Every `((a, b, k), ...)` covering `[0, J)`, without reference to the engine."""
    def widths(remaining):
        if remaining == 0:
            yield ()
            return
        for w in range(min_width, min(max_width, remaining) + 1):
            for tail in widths(remaining - w):
                yield (w,) + tail

    for parts in widths(J):
        starts, running = [], 0
        for w in parts:
            starts.append((running, running + w))
            running += w
        for labels in itertools.product(range(K), repeat=len(parts)):
            yield tuple((a, b, k) for (a, b), k in zip(starts, labels))


def segmentation_log_weight(blocks, table, log_pi, log_p, delta_b, J):
    log_db, log_1mdb = math.log(delta_b), math.log1p(-delta_b)
    total = float(log_pi[blocks[0][2]])
    for a, b, k in blocks:
        total += float(table[a, b, k])
    total += (len(blocks) - 1) * log_db + (J - len(blocks)) * log_1mdb
    for (_, _, left), (_, _, right) in zip(blocks[:-1], blocks[1:]):
        total += float(log_p[left, right])
    return total


def brute_force_log_z(table, log_pi, log_p, delta_b, J, K, min_width, max_width):
    weights = [segmentation_log_weight(blocks, table, log_pi, log_p, delta_b, J)
               for blocks in all_segmentations(J, K, min_width, max_width)]
    finite = [w for w in weights if np.isfinite(w)]
    return float(logsumexp(finite)) if finite else -np.inf


# ------------------------------------------------------------------------- fixtures
def random_problem(seed, J, K, min_width=1, max_width=None, forbid_self=True):
    max_width = J if max_width is None else max_width
    rng = np.random.default_rng(seed)
    table = np.full((J, J + 1, K), -np.inf)
    for a in range(J):
        for b in range(a + min_width, min(J, a + max_width) + 1):
            for k in range(K):
                table[a, b, k] = float(rng.normal(scale=2.0))
    log_pi = np.log(rng.dirichlet(np.ones(K)))
    transition = rng.dirichlet(np.ones(K), size=K)
    if forbid_self:
        np.fill_diagonal(transition, 0.0)
        transition = transition / transition.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        log_p = np.log(transition)
    return table, log_pi, log_p


# ------------------------------------------------------------------------- the tests
@pytest.mark.parametrize("seed,J,K,min_width,max_width", [
    (0, 6, 2, 1, 6),
    (1, 8, 3, 3, 12),          # the registered Stage 6E shape
    (2, 7, 3, 2, 4),
    (3, 5, 4, 1, 2),
    (4, 9, 2, 2, 9),
])
def test_forward_log_z_equals_brute_force(seed, J, K, min_width, max_width):
    table, log_pi, log_p = random_problem(seed, J, K, min_width, max_width)
    chart = forward(table, log_pi, log_p, 0.15, max_width, min_width)
    expected = brute_force_log_z(table, log_pi, log_p, 0.15, J, K, min_width, max_width)
    assert abs(chart.log_normalizer - expected) < 1e-12


def test_j_equals_one():
    table = np.full((1, 2, 2), -np.inf)
    table[0, 1, 0], table[0, 1, 1] = -0.5, -1.5
    log_pi = np.log([0.25, 0.75])
    with np.errstate(divide="ignore"):
        log_p = np.log([[0.0, 1.0], [1.0, 0.0]])
    chart = forward(table, log_pi, log_p, 0.15, 1)
    # exactly two paths, each one block wide, and no boundary term at all
    expected = logsumexp([log_pi[0] - 0.5, log_pi[1] - 1.5])
    assert abs(chart.log_normalizer - expected) < 1e-14
    assert chart.J == 1 and chart.K == 2


def test_k_equals_one_forces_a_single_block_when_self_transitions_are_forbidden():
    J = 4
    table = np.full((J, J + 1, 1), -np.inf)
    for a in range(J):
        for b in range(a + 1, J + 1):
            table[a, b, 0] = -0.1 * (b - a)
    log_pi = np.zeros(1)
    log_p = np.array([[-np.inf]])
    chart = forward(table, log_pi, log_p, 0.15, J)
    expected = table[0, J, 0] + (J - 1) * math.log1p(-0.15)
    assert abs(chart.log_normalizer - expected) < 1e-14


def test_max_width_one_forces_every_position_to_be_a_block():
    J, K = 5, 2
    table, log_pi, log_p = random_problem(11, J, K, 1, 1)
    chart = forward(table, log_pi, log_p, 0.3, 1, 1)
    expected = brute_force_log_z(table, log_pi, log_p, 0.3, J, K, 1, 1)
    assert abs(chart.log_normalizer - expected) < 1e-12
    # every draw has J blocks, so the boundary prior contributes (J-1) log delta_B only
    assert all(len(blocks) == J for blocks in all_segmentations(J, K, 1, 1))


def test_max_width_at_or_above_j():
    J, K = 6, 3
    table, log_pi, log_p = random_problem(12, J, K, 1, J)
    wide = forward(table, log_pi, log_p, 0.2, 3 * J, 1)
    exact = forward(table, log_pi, log_p, 0.2, J, 1)
    assert abs(wide.log_normalizer - exact.log_normalizer) < 1e-14


def test_forbidden_initial_skill_is_excluded():
    J, K = 6, 3
    table, log_pi, log_p = random_problem(13, J, K, 2, 3)
    log_pi = log_pi.copy()
    log_pi[1] = -np.inf                       # skill 1 can never start a sequence
    chart = forward(table, log_pi, log_p, 0.15, 3, 2)
    expected = brute_force_log_z(table, log_pi, log_p, 0.15, J, K, 2, 3)
    assert abs(chart.log_normalizer - expected) < 1e-12


def test_forbidden_transition_is_excluded():
    J, K = 8, 3
    table, log_pi, log_p = random_problem(14, J, K, 2, 4)
    log_p = log_p.copy()
    log_p[0, 2] = -np.inf                     # 0 -> 2 is forbidden as well as the diagonal
    chart = forward(table, log_pi, log_p, 0.15, 4, 2)
    expected = brute_force_log_z(table, log_pi, log_p, 0.15, J, K, 2, 4)
    assert abs(chart.log_normalizer - expected) < 1e-12


def test_forbidden_block_is_excluded():
    J, K = 7, 2
    table, log_pi, log_p = random_problem(15, J, K, 2, 3)
    table = table.copy()
    table[2, 5, 1] = -np.inf                  # this one candidate block is illegal
    chart = forward(table, log_pi, log_p, 0.15, 3, 2)
    expected = brute_force_log_z(table, log_pi, log_p, 0.15, J, K, 2, 3)
    assert abs(chart.log_normalizer - expected) < 1e-12


def test_asymmetric_transition_matrix():
    J, K = 8, 3
    table, log_pi, _ = random_problem(16, J, K, 2, 4)
    transition = np.array([[0.0, 0.9, 0.1], [0.2, 0.0, 0.8], [0.7, 0.3, 0.0]])
    with np.errstate(divide="ignore"):
        log_p = np.log(transition)
    assert not np.allclose(transition, transition.T)
    chart = forward(table, log_pi, log_p, 0.15, 4, 2)
    expected = brute_force_log_z(table, log_pi, log_p, 0.15, J, K, 2, 4)
    assert abs(chart.log_normalizer - expected) < 1e-12


def test_min_width_is_an_optimisation_only():
    """Widths below `min_width` are already `-inf`; passing the bound must change nothing."""
    J, K, min_width, max_width = 9, 3, 3, 5
    table, log_pi, log_p = random_problem(17, J, K, min_width, max_width)
    with_bound = forward(table, log_pi, log_p, 0.15, max_width, min_width)
    without = forward(table, log_pi, log_p, 0.15, max_width, 1)
    assert with_bound.log_normalizer == without.log_normalizer
    finite = np.isfinite(with_bound.alpha)
    assert np.array_equal(finite, np.isfinite(without.alpha))
    assert np.allclose(with_bound.alpha[finite], without.alpha[finite], rtol=0, atol=0)


def test_boundary_prior_bookkeeping_is_exact():
    """Every internal position costs log(1-delta_B), every cut costs log delta_B, and the
    endpoint costs neither — checked against a hand-computed two-block weight."""
    J, K, delta = 6, 2, 0.15
    table = np.full((J, J + 1, K), -np.inf)
    table[0, 3, 0] = -1.0
    table[3, 6, 1] = -2.0
    log_pi = np.log([1.0, 1e-300])
    with np.errstate(divide="ignore"):
        log_p = np.log([[0.0, 1.0], [1.0, 0.0]])
    chart = forward(table, log_pi, log_p, delta, 3, 3)
    expected = (float(log_pi[0]) - 1.0 - 2.0 + math.log(delta)
                + (J - 2) * math.log1p(-delta))
    assert abs(chart.log_normalizer - expected) < 1e-12


def test_dp_marginals_match_brute_force():
    J, K, min_width, max_width = 8, 3, 2, 4
    table, log_pi, log_p = random_problem(18, J, K, min_width, max_width)
    chart = forward(table, log_pi, log_p, 0.15, max_width, min_width)
    marginals = posterior_log_marginals(chart)

    weights, blocks_of = [], []
    for blocks in all_segmentations(J, K, min_width, max_width):
        weights.append(segmentation_log_weight(blocks, table, log_pi, log_p, 0.15, J))
        blocks_of.append(blocks)
    weights = np.array(weights)
    probability = np.exp(weights - logsumexp(weights))
    expected: dict = {}
    for blocks, p in zip(blocks_of, probability):
        for block in blocks:
            expected[block] = expected.get(block, 0.0) + float(p)

    assert abs(marginals["log_normalizer_from_beta"] - chart.log_normalizer) < 1e-12
    worst = max(abs(marginals["labelled_block_marginals"].get(b, 0.0) - p)
                for b, p in expected.items())
    assert worst < 1e-12


# --------------------------------------------------------------------------- validation
def test_validation_rejects_bad_input():
    table, log_pi, log_p = random_problem(19, 5, 2, 1, 5)
    with pytest.raises(ValueError, match="boundary_prob"):
        forward(table, log_pi, log_p, 0.0, 5)
    with pytest.raises(ValueError, match="boundary_prob"):
        forward(table, log_pi, log_p, 1.0, 5)
    with pytest.raises(ValueError, match="max_width"):
        forward(table, log_pi, log_p, 0.15, 0)
    with pytest.raises(ValueError, match="min_width"):
        forward(table, log_pi, log_p, 0.15, 3, 4)
    with pytest.raises(ValueError, match=r"\(J, J\+1, K\)"):
        forward(table[:, :-1, :], log_pi, log_p, 0.15, 5)
    with pytest.raises(ValueError, match="log_initial_probs"):
        forward(table, log_pi[:1], log_p, 0.15, 5)
    with pytest.raises(ValueError, match="log_transition_matrix"):
        forward(table, log_pi, log_p[:1], 0.15, 5)
    nan_table = table.copy()
    nan_table[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        forward(nan_table, log_pi, log_p, 0.15, 5)
    inf_table = table.copy()
    inf_table[0, 1, 0] = np.inf
    with pytest.raises(ValueError, match=r"\+inf"):
        forward(inf_table, log_pi, log_p, 0.15, 5)


def test_no_finite_complete_path_is_an_error_not_a_silent_minus_inf():
    J, K = 4, 2
    table = np.full((J, J + 1, K), -np.inf)
    table[0, 2, 0] = -1.0                     # covers [0,2) only; [2,4) has no legal block
    log_pi = np.zeros(K)
    with np.errstate(divide="ignore"):
        log_p = np.log([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="no finite complete path"):
        forward(table, log_pi, log_p, 0.15, 2, 2)


def test_predecessor_terms_is_the_only_recurrence_and_reproduces_alpha():
    """The forward chart must be exactly the log-sum of the helper's own terms."""
    J, K, min_width, max_width = 8, 3, 2, 4
    table, log_pi, log_p = random_problem(20, J, K, min_width, max_width)
    delta = 0.15
    chart = forward(table, log_pi, log_p, delta, max_width, min_width)
    log_db, log_1mdb = math.log(delta), math.log1p(-delta)
    for b in range(1, J + 1):
        for k in range(K):
            _, _, terms = predecessor_terms(chart.alpha, b, k, table, log_pi, log_p,
                                            log_db, log_1mdb, max_width, min_width)
            if terms.size == 0:
                assert chart.alpha[b, k] == -np.inf
            else:
                assert abs(float(logsumexp(terms)) - float(chart.alpha[b, k])) < 1e-12


def test_chart_is_a_frozen_record():
    table, log_pi, log_p = random_problem(21, 6, 2, 2, 3)
    chart = forward(table, log_pi, log_p, 0.15, 3, 2)
    assert isinstance(chart, ForwardChart)
    summary = chart.summary()
    assert summary["J"] == 6 and summary["K"] == 2
    assert summary["max_width"] == 3 and summary["min_width"] == 2
    assert summary["boundary_prob"] == 0.15
    assert math.isfinite(summary["log_normalizer"])
    with pytest.raises(Exception):
        chart.log_normalizer = 0.0            # frozen dataclass
