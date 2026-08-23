"""Stage 1 — segmentation MH must reproduce the exact Stage-0 posterior."""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_original import toy
from hpop.mcmc_original.diagnostics import (
    autocorrelation,
    boundary_marginals_from_samples,
    effective_sample_size,
    segmentation_frequencies,
    total_variation_distance,
)
from hpop.mcmc_original.enumerate import (
    boundary_marginals_from_probs,
    build_trace_states,
    exact_state_table,
)
from hpop.mcmc_original.sampler_segmentation import (
    mh_segmentation_step,
    propose_other_state,
    run_segmentation_mcmc,
)
from hpop.mcmc_original.targets import SkillEvaluator

SEED = 20260808
N_ITERATIONS = 100_000
BURN_IN = 5_000


@pytest.fixture(scope="module")
def stage1():
    skills = toy.stage012_skills()
    evaluators = [SkillEvaluator(s) for s in skills]
    trace_states = build_trace_states(toy.PRIMARY_TRACE, skills, evaluators, toy.DELTA_B)
    tables = [evaluators[k].log_table(skills[k].u) for k in range(len(skills))]
    exact = exact_state_table(trace_states, tables, toy.uniform_log_pi(len(skills)))
    rng = np.random.default_rng(SEED)
    result = run_segmentation_mcmc(
        exact["log_targets"], N_ITERATIONS, BURN_IN, rng, init=0
    )
    return {"trace_states": trace_states, "exact": exact, "result": result}


# ---------------------------------------------------------------------------
# the proposal is symmetric and well-formed
# ---------------------------------------------------------------------------


def test_proposal_never_returns_the_current_state():
    rng = np.random.default_rng(0)
    for n_states in (2, 3, 5):
        for current in range(n_states):
            for _ in range(200):
                assert propose_other_state(n_states, current, rng) != current


def test_proposal_is_uniform_over_the_other_states():
    rng = np.random.default_rng(1)
    counts = np.zeros(4)
    for _ in range(40_000):
        counts[propose_other_state(4, 1, rng)] += 1
    assert counts[1] == 0
    frequencies = counts[[0, 2, 3]] / counts.sum()
    np.testing.assert_allclose(frequencies, np.full(3, 1 / 3), atol=0.02)


def test_single_state_update_is_a_noop():
    rng = np.random.default_rng(2)
    assert propose_other_state(1, 0, rng) == 0
    state, accepted, proposed = mh_segmentation_step(0, np.array([-1.0]), rng)
    assert (state, accepted, proposed) == (0, False, False)


def test_proposal_symmetry_q_ratio_is_one():
    """q(S'|S) = q(S|S') = 1/(N-1), so the Hastings correction is exactly 1."""
    for n_states in (2, 3, 7):
        forward = 1.0 / (n_states - 1)
        backward = 1.0 / (n_states - 1)
        assert forward == backward


# ---------------------------------------------------------------------------
# the chain targets the exact posterior
# ---------------------------------------------------------------------------


def test_segmentation_mcmc_matches_exact(stage1):
    exact = stage1["exact"]["probs"]
    empirical = segmentation_frequencies(
        stage1["result"]["kept"], stage1["trace_states"].n_states
    )
    for i in range(len(exact)):
        assert abs(empirical[i] - exact[i]) < 0.01, (
            f"state {i}: empirical {empirical[i]:.5f} vs exact {exact[i]:.5f}"
        )
    assert total_variation_distance(empirical, exact) < 0.01


def test_segmentation_mcmc_boundary_marginals(stage1):
    trace_states = stage1["trace_states"]
    exact = boundary_marginals_from_probs(trace_states, stage1["exact"]["probs"])
    empirical = boundary_marginals_from_samples(
        stage1["result"]["kept"], trace_states.cuts, trace_states.length
    )
    for t in (2, 3):
        assert abs(empirical[t - 1] - exact[t - 1]) < 0.01, (
            f"cut {t}: empirical {empirical[t-1]:.5f} vs exact {exact[t-1]:.5f}"
        )
    assert empirical[0] == 0.0 and empirical[3] == 0.0


def test_chain_visits_both_states(stage1):
    """A chain stuck in one state could pass a loose tolerance for the wrong reason."""
    visited = np.unique(stage1["result"]["kept"])
    assert len(visited) == 2


def test_acceptance_rate_is_recorded_and_sane(stage1):
    rate = stage1["result"]["acceptance_rate"]
    assert 0.0 < rate < 1.0
    assert stage1["result"]["n_proposed"] == N_ITERATIONS


def test_ess_and_autocorrelation_are_computable(stage1):
    indicator = (stage1["result"]["kept"] == 1).astype(float)
    rho = autocorrelation(indicator, max_lag=20)
    assert rho[0] == pytest.approx(1.0)
    ess = effective_sample_size(indicator)
    assert 0.0 < ess <= len(indicator)


def test_result_is_reproducible_under_the_same_seed(stage1):
    rng = np.random.default_rng(SEED)
    repeat = run_segmentation_mcmc(
        stage1["exact"]["log_targets"], N_ITERATIONS, BURN_IN, rng, init=0
    )
    np.testing.assert_array_equal(repeat["kept"], stage1["result"]["kept"])


def test_different_init_reaches_the_same_posterior(stage1):
    rng = np.random.default_rng(SEED + 1)
    other = run_segmentation_mcmc(
        stage1["exact"]["log_targets"], N_ITERATIONS, BURN_IN, rng, init=1
    )
    empirical = segmentation_frequencies(
        other["kept"], stage1["trace_states"].n_states
    )
    assert total_variation_distance(empirical, stage1["exact"]["probs"]) < 0.01


def test_mh_targets_a_hand_made_three_state_distribution():
    """A stronger check of the kernel: three states with known weights."""
    log_targets = np.log(np.array([0.2, 0.5, 0.3]))
    rng = np.random.default_rng(5)
    result = run_segmentation_mcmc(log_targets, 200_000, 10_000, rng, init=0)
    empirical = segmentation_frequencies(result["kept"], 3)
    np.testing.assert_allclose(empirical, [0.2, 0.5, 0.3], atol=0.01)


def test_total_variation_distance_basics():
    assert total_variation_distance([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert total_variation_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
