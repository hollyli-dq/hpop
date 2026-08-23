"""Stage 6C — diagnostics (§17 areas 32, 33, 34, 35, 36).

Two conventions are enforced here because they are the ones most likely to produce a
confidently wrong report:

* **a constant trace is not convergence.** When the poset posterior is effectively a
  point mass the relation-count trace never moves, and R-hat / ESS are undefined rather
  than excellent. `convergence_block` must say so.
* **rho recovery is NOT APPLICABLE, not PASS.** `U_TRUE` is hand specified rather than
  drawn from `p(U | rho_true)`, so no generating `rho` exists. A finite posterior
  interval is not recovery, and the diagnostics must refuse to imply otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.stage6c_diagnostics import (
    SMOKE_REQUIRED_KEYS, convergence_block, full_u_total_variation, mixed_comparison,
    mode_summary, poset_distribution, recovery_metrics, relation_marginals,
    scalar_diagnostics, smoke_summary, structural_diagnostics,
)
from hpop.mcmc_original.stage6c_exact_reference import build_catalogue
from hpop.mcmc_original.stage6c_frozen import load_stage6c_dataset


@pytest.fixture(scope="module")
def catalogue():
    return build_catalogue(4, 2)


@pytest.fixture(scope="module")
def frozen():
    return load_stage6c_dataset()


# --------------------------------------------------------- area 32: TV over U states
def test_total_variation_is_zero_against_the_matching_distribution(catalogue):
    ids = np.array([0, 0, 1, 2, 2, 2])
    empirical = poset_distribution(ids, catalogue.size)
    result = full_u_total_variation(ids, empirical, catalogue.size)
    assert result["total_variation"] == pytest.approx(0.0, abs=1e-12)


def test_total_variation_is_one_for_disjoint_support(catalogue):
    ids = np.array([0, 0, 0])
    reference = np.zeros(catalogue.size)
    reference[1] = 1.0
    result = full_u_total_variation(ids, reference, catalogue.size)
    assert result["total_variation"] == pytest.approx(1.0, abs=1e-12)


def test_total_variation_matches_a_hand_computed_value(catalogue):
    ids = np.array([0, 0, 0, 1])                      # empirical: 0.75 / 0.25
    reference = np.zeros(catalogue.size)
    reference[0], reference[1] = 0.5, 0.5
    result = full_u_total_variation(ids, reference, catalogue.size)
    assert result["total_variation"] == pytest.approx(0.25, abs=1e-12)
    assert result["n_states_visited"] == 2


def test_unmatched_draws_are_an_error_not_a_silent_drop(catalogue):
    with pytest.raises(ValueError):
        poset_distribution(np.array([0, -1, 2]), catalogue.size)


def test_reference_is_renormalised_before_comparison(catalogue):
    """A reference that does not quite sum to one must not inflate the distance."""
    ids = np.array([0, 1])
    reference = np.zeros(catalogue.size)
    reference[0], reference[1] = 1.0, 1.0             # sums to 2
    result = full_u_total_variation(ids, reference, catalogue.size)
    assert result["total_variation"] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------- area 33: relation marginals
def test_relation_marginal_of_a_single_state_is_that_state(catalogue):
    for index in (0, 3, catalogue.size - 1):
        marginals = relation_marginals(np.array([index] * 7), catalogue)
        expected = catalogue.closures[index].reshape(-1).astype(float)
        assert np.array_equal(marginals["relation_marginal"], expected)


def test_relation_marginal_is_the_weighted_average_of_the_visited_states(catalogue):
    ids = np.array([0, 0, 0, 1])
    marginals = relation_marginals(ids, catalogue)
    expected = (0.75 * catalogue.closures[0].reshape(-1).astype(float)
                + 0.25 * catalogue.closures[1].reshape(-1).astype(float))
    assert np.allclose(marginals["relation_marginal"], expected)


def test_reduction_marginal_differs_from_the_closure_marginal_somewhere(catalogue):
    """They answer different questions and must not be silently interchangeable."""
    ids = np.arange(catalogue.size)
    marginals = relation_marginals(ids, catalogue)
    assert not np.allclose(marginals["relation_marginal"],
                           marginals["reduction_marginal"])


def test_structural_diagnostics_reports_per_chain_agreement(catalogue):
    reference_probability = poset_distribution(np.array([0, 1]), catalogue.size)
    arrays = {
        "poset_probability": reference_probability,
        "relation_marginal": relation_marginals(np.array([0, 1]),
                                                catalogue)["relation_marginal"],
        "reduction_marginal": relation_marginals(np.array([0, 1]),
                                                 catalogue)["reduction_marginal"],
    }
    ids = np.array([[0, 1, 0, 1], [0, 1, 0, 1]])
    result = structural_diagnostics(ids, catalogue, arrays)
    assert result["full_u_total_variation"] == pytest.approx(0.0, abs=1e-12)
    assert result["max_relation_marginal_error"] == pytest.approx(0.0, abs=1e-12)
    assert len(result["per_chain"]) == 2
    assert result["worst_chain_relation_error"] == pytest.approx(0.0, abs=1e-12)


def test_a_disagreeing_chain_is_visible_in_the_per_chain_report(catalogue):
    reference_probability = poset_distribution(np.array([0, 1]), catalogue.size)
    arrays = {
        "poset_probability": reference_probability,
        "relation_marginal": relation_marginals(np.array([0, 1]),
                                                catalogue)["relation_marginal"],
        "reduction_marginal": relation_marginals(np.array([0, 1]),
                                                 catalogue)["reduction_marginal"],
    }
    ids = np.array([[0, 1, 0, 1], [2, 2, 2, 2]])       # chain 1 is somewhere else
    result = structural_diagnostics(ids, catalogue, arrays)
    assert result["worst_chain_relation_error"] > 0.5
    assert result["per_chain"][1]["total_variation"] > 0.9


# ------------------------------- area 34: closure vs reduction recovery, kept separate
def test_perfect_recovery_scores_one_on_both_representations(frozen):
    catalogue = build_catalogue(5, 2)
    true_index = catalogue.index_of(precedence_from_u(frozen.u_true))
    ids = np.array([true_index] * 200)
    result = recovery_metrics(ids, catalogue, frozen.u_true)

    assert result["map_is_true"]
    assert result["posterior_probability_of_true"] == pytest.approx(1.0)
    assert result["posterior_rank_of_true"] == 1
    assert result["closure"]["f1"] == pytest.approx(1.0)
    assert result["reduction"]["f1"] == pytest.approx(1.0)
    assert result["closure"]["structural_hamming"] == 0
    assert result["reduction"]["structural_hamming"] == 0
    assert result["min_true_relation_probability"] == pytest.approx(1.0)
    assert result["max_false_relation_probability"] == pytest.approx(0.0)


def test_closure_and_reduction_scores_can_disagree(frozen):
    """A chain sitting on a total order recovers no reduction edge it should not.

    The two representations are scored separately precisely because a state can look
    close on one and far on the other.
    """
    catalogue = build_catalogue(5, 2)
    total_order = np.array([[5.0, 5.0], [4.0, 4.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0]])
    index = catalogue.index_of(precedence_from_u(total_order))
    result = recovery_metrics(np.array([index] * 50), catalogue, frozen.u_true)
    assert result["closure"]["structural_hamming"] != result["reduction"][
        "structural_hamming"]


def test_recovery_reports_the_rank_of_the_true_state_when_it_is_not_the_map(frozen):
    catalogue = build_catalogue(5, 2)
    true_index = catalogue.index_of(precedence_from_u(frozen.u_true))
    other = 0 if true_index != 0 else 1
    ids = np.array([other] * 70 + [true_index] * 30)
    result = recovery_metrics(ids, catalogue, frozen.u_true)
    assert not result["map_is_true"]
    assert result["posterior_rank_of_true"] == 2
    assert result["posterior_probability_of_true"] == pytest.approx(0.3)


def test_mode_summary_counts_visits_and_transitions(catalogue):
    ids = np.array([[0, 0, 1, 1, 0], [2, 2, 2, 2, 2]])
    result = mode_summary(ids, catalogue, top=3)
    assert set(result["major_modes"]) == {0, 1, 2}
    assert result["visits_by_chain"]["0"] == [3, 0]
    assert result["transitions_between_major_modes"] == 2       # 0->1 and 1->0, chain 0


# ------------------------------------- degenerate traces must not be read as convergence
def test_a_constant_trace_is_reported_as_degenerate_not_converged():
    result = convergence_block(np.full((4, 500), 6.0), "relation count")
    assert result["degenerate"] is True
    assert result["rhat"] is None
    assert result["bulk_ess"] is None
    assert "undefined" in result["note"]


def test_a_varying_trace_gets_real_convergence_numbers():
    rng = np.random.default_rng(0)
    result = convergence_block(rng.normal(size=(4, 800)), "test")
    assert result["degenerate"] is False
    assert result["rhat"] < 1.05
    assert result["bulk_ess"] > 100
    assert result["mcse"] > 0


# ----------------------------------------- area 36: rho recovery is NOT APPLICABLE
def test_rho_is_reported_as_not_applicable_when_no_true_value_exists():
    rng = np.random.default_rng(1)
    chains = {"rho": rng.uniform(0.1, 0.6, size=(4, 400))}
    grid = np.linspace(0.0, 1.0, 101)
    reference = {"rho": {"grid": grid, "cdf": grid, "mean": 0.35, "sd": 0.18}}
    result = scalar_diagnostics(chains, reference, truth={},
                                acceptance_total={"rho": 0.4},
                                acceptance_post={"rho": 0.41})
    assert result["rho"]["true_value"] is None
    assert result["rho"]["recovery"] == "NOT APPLICABLE — no generating value exists"
    assert "truth_in_95_interval" not in result["rho"]
    assert result["rho"]["ks_distance_to_reference"] >= 0.0


def test_a_scalar_with_a_true_value_does_get_a_recovery_verdict():
    rng = np.random.default_rng(2)
    chains = {"beta": 1.5 + 0.03 * rng.normal(size=(4, 400))}
    grid = np.linspace(1.2, 1.8, 201)
    reference = {"beta": {"grid": grid, "cdf": np.linspace(0, 1, 201),
                          "mean": 1.5, "sd": 0.03}}
    result = scalar_diagnostics(chains, reference, truth={"beta": 1.5},
                                acceptance_total={"beta": 0.3},
                                acceptance_post={"beta": 0.3})
    assert result["beta"]["true_value"] == 1.5
    assert result["beta"]["truth_in_95_interval"] is True
    assert "recovery" not in result["beta"]


def test_explicitly_passing_none_as_a_truth_is_treated_as_not_applicable():
    rng = np.random.default_rng(3)
    chains = {"rho": rng.uniform(0.1, 0.6, size=(2, 200))}
    result = scalar_diagnostics(chains, {}, truth={"rho": None},
                                acceptance_total={}, acceptance_post={})
    assert result["rho"]["recovery"] == "NOT APPLICABLE — no generating value exists"


# --------------------------------------------------- area 35: smoke-output schemas
def test_smoke_summary_requires_every_registered_check():
    partial = {k: True for k in SMOKE_REQUIRED_KEYS[:-1]}
    with pytest.raises(ValueError, match="missing required checks"):
        smoke_summary(partial)


def test_smoke_summary_rejects_non_boolean_checks():
    checks = {k: True for k in SMOKE_REQUIRED_KEYS}
    checks["rho_moved"] = "yes"
    with pytest.raises(ValueError, match="must be booleans"):
        smoke_summary(checks)


def test_smoke_summary_reports_which_checks_failed():
    checks = {k: True for k in SMOKE_REQUIRED_KEYS}
    checks["rejection_safe"] = False
    summary = smoke_summary(checks)
    assert summary["all_passed"] is False
    assert summary["failed_checks"] == ["rejection_safe"]


def test_a_fully_passing_smoke_summary_has_the_expected_schema():
    checks = {k: True for k in SMOKE_REQUIRED_KEYS}
    summary = smoke_summary(checks, extra={"n_sweeps": 200})
    assert set(summary["checks"]) == set(SMOKE_REQUIRED_KEYS)
    assert summary["all_passed"] is True
    assert summary["failed_checks"] == []
    assert summary["n_sweeps"] == 200


# ------------------------------------------------- mixed discrete/continuous comparison
def test_mixed_comparison_passes_when_both_sides_come_from_the_same_law(catalogue):
    rng = np.random.default_rng(4)
    n = 3000
    ids = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
    reference_ids = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
    reference = {
        "relations": catalogue.closures[reference_ids].reshape(n, -1).astype(float),
        "rho": rng.uniform(0.1, 0.6, size=n)}
    result = mixed_comparison(ids, {"rho": rng.uniform(0.1, 0.6, size=n)}, catalogue,
                              reference, seed=1, n_compare=500, n_replicates=20)
    assert result["pass"] is True


def test_mixed_comparison_fails_when_the_laws_differ(catalogue):
    rng = np.random.default_rng(5)
    n = 3000
    ids = np.zeros(n, dtype=int)
    reference_ids = rng.choice([1, 2], size=n)
    reference = {
        "relations": catalogue.closures[reference_ids].reshape(n, -1).astype(float),
        "rho": rng.uniform(0.1, 0.6, size=n)}
    result = mixed_comparison(ids, {"rho": rng.uniform(0.6, 0.9, size=n)}, catalogue,
                              reference, seed=1, n_compare=500, n_replicates=20)
    assert result["pass"] is False


def test_constant_reference_coordinates_are_dropped_not_turned_into_infinities(catalogue):
    """A relation that never varies has zero scale; standardising it would divide by zero."""
    n = 500
    ids = np.zeros(n, dtype=int)
    reference = {"relations": catalogue.closures[np.zeros(n, dtype=int)].reshape(n, -1)
                 .astype(float)}
    result = mixed_comparison(ids, {}, catalogue, reference, seed=1, n_compare=200,
                              n_replicates=10)
    assert result["dropped_constant_coordinates"] > 0
    assert np.isfinite(result["observed"])
