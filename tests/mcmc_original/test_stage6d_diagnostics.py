"""Stage 6D — diagnostics for the joint latent-`U` + four-scalar chains.

Four properties are enforced here because each one, if wrong, produces a report that is
confidently wrong rather than merely imprecise:

* **`H` labels are canonical.** Stage 6D has no fixed poset catalogue: the small 6D1
  model and the full 6D2 corpus induce different order spaces, so MCMC and reference are
  aligned by the closure's own bytes. Two draws with the same induced order must receive
  the same label whatever their `U`, and two draws with different orders must not.
* **A constant trace is not convergence.** On the full corpus the induced order is a
  point mass, so the relation-count trace never moves. The diagnostics must say
  `degenerate`, never report an R-hat of 1.0.
* **A correlation with a constant coordinate is undefined, not zero.** Emitting 0.0 there
  would read as "no dependence" when the truth is "no information".
* **Column permutation is a symmetry, not a failure.** `h(U)` is the intersection of the
  column orderings, so the target is column-exchangeable and entrywise `U` traces may
  swap labels between chains. The audit must detect that and say what it means.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.sampler_u import sigma_rho_matrix
from hpop.mcmc_original.stage6d_diagnostics import (
    align_h_distributions, column_permutation_audit, dependence_block, h_labels_from_u,
    h_total_variation, mixed_envelope, relation_marginal_from_u, scalar_block,
    structural_block, weighted_quantile,
)

ANTICHAIN = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
TOTAL_ORDER = np.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
ONE_OVER_TWO = np.array([[2.0, 2.0], [1.0, 0.0], [0.0, 1.0]])


# ------------------------------------------------------------------ induced H labels
def test_same_induced_order_from_different_u_gets_the_same_label():
    """Labels track `h(U)`, not `U`. A monotone rescale of a column cannot relabel."""
    stretched = TOTAL_ORDER * np.array([3.0, 1.0]) + np.array([-5.0, 0.25])
    labels, keys, _ = h_labels_from_u(np.array([TOTAL_ORDER, stretched]))
    assert labels[0] == labels[1]
    assert len(keys) == 1


def test_different_induced_orders_get_different_labels():
    labels, keys, closures = h_labels_from_u(
        np.array([ANTICHAIN, TOTAL_ORDER, ONE_OVER_TWO, ANTICHAIN]))
    assert labels[0] == labels[3]
    assert len({int(l) for l in labels}) == 3
    assert len(keys) == 3
    # the closure stack is aligned with the key order, first-appearance ordered
    assert np.array_equal(closures[0], precedence_from_u(ANTICHAIN))
    assert np.array_equal(closures[1], precedence_from_u(TOTAL_ORDER))
    assert closures[0].sum() == 0                     # the antichain really is empty
    assert closures[1].sum() == 3                     # 3 relations on 3 elements


def test_column_permutation_leaves_the_label_unchanged():
    """`h(U)` intersects the column orderings, so swapping columns cannot change it."""
    labels, keys, _ = h_labels_from_u(np.array([ONE_OVER_TWO, ONE_OVER_TWO[:, ::-1]]))
    assert labels[0] == labels[1]
    assert len(keys) == 1


# ---------------------------------------------------------------------- TV alignment
def test_alignment_uses_the_union_of_keys_not_a_shared_index():
    a, b, union = align_h_distributions([b"x", b"y"], [0.6, 0.4],
                                        [b"y", b"z"], [0.5, 0.5])
    assert union == [b"x", b"y", b"z"]
    assert a.tolist() == [0.6, 0.4, 0.0]
    assert b.tolist() == [0.0, 0.5, 0.5]


def test_total_variation_is_zero_for_identical_distributions_and_one_for_disjoint():
    assert h_total_variation([0.2, 0.8], [0.2, 0.8]) == pytest.approx(0.0, abs=1e-12)
    assert h_total_variation([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0, abs=1e-12)


def test_total_variation_matches_a_hand_computed_value():
    # |0.75-0.5|/2 + |0.25-0.5|/2 = 0.25
    assert h_total_variation([0.75, 0.25], [0.5, 0.5]) == pytest.approx(0.25, abs=1e-12)


def test_total_variation_renormalises_so_an_unnormalised_input_cannot_inflate_it():
    assert h_total_variation([1.5, 0.5], [0.75, 0.25]) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------ relation marginals
def test_relation_marginal_counts_the_fraction_of_draws_holding_each_relation():
    draws = np.array([TOTAL_ORDER, TOTAL_ORDER, ANTICHAIN, ANTICHAIN])
    marginal = relation_marginal_from_u(draws).reshape(3, 3)
    assert marginal[0, 1] == pytest.approx(0.5)
    assert marginal[1, 0] == pytest.approx(0.0)
    assert np.allclose(np.diag(marginal), 0.0)


def test_relation_marginal_honours_importance_weights():
    draws = np.array([TOTAL_ORDER, ANTICHAIN])
    marginal = relation_marginal_from_u(draws, weights=[3.0, 1.0]).reshape(3, 3)
    assert marginal[0, 1] == pytest.approx(0.75)


def test_weighted_quantile_reduces_to_the_ordinary_one_under_equal_weights():
    values = np.linspace(0.0, 1.0, 101)
    weights = np.ones_like(values)
    for q in (0.025, 0.5, 0.975):
        assert weighted_quantile(values, weights, q) == pytest.approx(
            float(np.quantile(values, q)), abs=2e-2)


def test_weighted_quantile_moves_with_the_weights():
    values = np.array([0.0, 1.0, 2.0])
    balanced = weighted_quantile(values, [1.0, 1.0, 1.0], 0.5)
    low_heavy = weighted_quantile(values, [10.0, 1.0, 1.0], 0.5)
    high_heavy = weighted_quantile(values, [1.0, 1.0, 10.0], 0.5)
    assert low_heavy < balanced < high_heavy
    assert balanced == pytest.approx(0.5)


# ------------------------------------------------------------------- structural block
def _chains(*orders, per_chain=6):
    """Two chains, each cycling through the given U matrices."""
    return np.array([[orders[i % len(orders)] for i in range(per_chain)],
                     [orders[(i + 1) % len(orders)] for i in range(per_chain)]],
                    dtype=float)


def test_structural_block_reports_a_constant_order_as_degenerate_not_as_converged():
    """The full-corpus case: the induced order never moves, so R-hat is undefined."""
    block = structural_block(_chains(TOTAL_ORDER))
    assert block["n_h_states_visited"] == 1
    assert block["relation_count_convergence"]["degenerate"] is True
    assert block["relation_count_convergence"]["rhat"] is None
    assert block["n_uncertain_relations"] == 0
    assert block["max_relation_rhat"] is None
    assert block["min_relation_bulk_ess"] is None


def test_structural_block_scores_only_the_relations_that_actually_vary():
    """A relation that is always absent has no R-hat; averaging it in would flatter."""
    block = structural_block(_chains(TOTAL_ORDER, ANTICHAIN))
    assert block["n_h_states_visited"] == 2
    # 3 relations vary (the total order's), the reverse 3 and the diagonal never do
    assert block["n_uncertain_relations"] == 3
    scored = {r["relation"] for r in block["uncertain_relations"]}
    varying = {1, 2, 5}                                # (0>1), (0>2), (1>2) flattened
    assert scored == varying
    assert all(r["rhat"] is not None for r in block["uncertain_relations"])


def test_uncertain_relations_carry_a_usable_r_hat_and_ess_on_a_realistic_trace():
    rng = np.random.default_rng(0)
    orders = (TOTAL_ORDER, ANTICHAIN)
    chains = np.array([[orders[int(x)] for x in rng.integers(0, 2, 400)]
                       for _ in range(4)], dtype=float)
    block = structural_block(chains)
    assert block["n_uncertain_relations"] == 3
    assert np.isfinite(block["min_relation_bulk_ess"])
    assert block["min_relation_bulk_ess"] > 100.0
    assert block["max_relation_rhat"] < 1.01
    assert block["relation_count_convergence"]["degenerate"] is False


def test_structural_block_incomparability_is_the_complement_of_the_order():
    block = structural_block(_chains(ANTICHAIN))
    inc = np.array(block["incomparability_marginal"]).reshape(3, 3)
    assert np.allclose(np.diag(inc), 0.0)
    assert inc[0, 1] == pytest.approx(1.0)            # a pure antichain
    assert np.array(block["relation_marginal"]).sum() == pytest.approx(0.0)


def test_structural_block_relation_count_distribution_is_a_probability_vector():
    block = structural_block(_chains(TOTAL_ORDER, ANTICHAIN, ONE_OVER_TWO))
    dist = np.array(block["relation_count_distribution"])
    assert dist.sum() == pytest.approx(1.0)
    assert dist.size == 3 * 2 // 2 + 1
    assert dist[0] > 0 and dist[2] > 0 and dist[3] > 0


def test_structural_block_compares_against_a_reference_by_key_not_by_position():
    """The reference lists its states in a different order; alignment must not care."""
    u_by_chain = _chains(TOTAL_ORDER, ANTICHAIN)
    reference_keys = [precedence_from_u(ANTICHAIN).tobytes(),
                      precedence_from_u(TOTAL_ORDER).tobytes()]
    reference_relation = precedence_from_u(TOTAL_ORDER).reshape(-1).astype(float) * 0.5
    block = structural_block(u_by_chain, reference_keys, [0.5, 0.5], reference_relation)
    assert block["h_total_variation"] == pytest.approx(0.0, abs=1e-12)
    assert block["max_relation_marginal_error"] == pytest.approx(0.0, abs=1e-12)


def test_structural_block_reports_a_real_discrepancy_rather_than_hiding_it():
    u_by_chain = np.array([[TOTAL_ORDER] * 4, [TOTAL_ORDER] * 4], dtype=float)
    reference_keys = [precedence_from_u(TOTAL_ORDER).tobytes(),
                      precedence_from_u(ANTICHAIN).tobytes()]
    block = structural_block(u_by_chain, reference_keys, [0.5, 0.5],
                             precedence_from_u(TOTAL_ORDER).reshape(-1) * 0.5)
    assert block["h_total_variation"] == pytest.approx(0.5, abs=1e-12)
    assert block["max_relation_marginal_error"] == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------------- scalar block
def _scalar_input(seed=0):
    rng = np.random.default_rng(seed)
    return {"beta": 1.5 + 0.1 * rng.normal(size=(4, 500)),
            "rho": rng.uniform(0.0, 0.99, size=(4, 500))}


def test_scalar_block_marks_a_parameter_with_no_generating_value_not_applicable():
    """`rho` has no truth: `U_TRUE` was hand specified, not drawn from `p(U | rho)`."""
    out = scalar_block(_scalar_input(), {}, {"beta": 1.5, "rho": None}, {}, {})
    assert out["rho"]["true_value"] is None
    assert "NOT APPLICABLE" in out["rho"]["recovery"]
    assert "truth_in_95_interval" not in out["rho"]
    assert out["beta"]["true_value"] == 1.5
    assert out["beta"]["truth_in_95_interval"] is True


def test_scalar_block_reports_the_reference_error_in_reference_standard_deviations():
    out = scalar_block(_scalar_input(), {"beta": {"mean": 1.4, "sd": 0.1}},
                       {"beta": 1.5, "rho": None}, {}, {})
    assert out["beta"]["mean_error"] == pytest.approx(
        out["beta"]["posterior_mean"] - 1.4)
    assert out["beta"]["mean_error_in_reference_sd"] == pytest.approx(
        out["beta"]["mean_error"] / 0.1)


def test_scalar_block_carries_both_acceptance_rates_through():
    out = scalar_block(_scalar_input(), {}, {"beta": 1.5, "rho": None},
                       {"beta": 0.9}, {"beta": 0.45})
    assert out["beta"]["acceptance_total"] == 0.9
    assert out["beta"]["acceptance_post_burn_in"] == 0.45


def test_scalar_block_reports_a_constant_scalar_as_degenerate():
    out = scalar_block({"beta": np.full((4, 100), 1.5)}, {}, {"beta": 1.5}, {}, {})
    assert out["beta"]["degenerate"] is True
    assert out["beta"]["rhat"] is None


# ------------------------------------------------------------------ dependence block
def test_a_correlation_with_a_constant_coordinate_is_undefined_with_a_reason():
    series = {"relation_count": np.full(50, 6.0), "beta": np.linspace(1.0, 2.0, 50)}
    out = dependence_block(series, [("relation_count", "beta")])
    entry = out["relation_count_vs_beta"]
    assert entry["correlation"] is None
    assert entry["undefined_because_constant"] == ["relation_count"]
    assert "not zero" in entry["note"]


def test_an_ordinary_correlation_is_reported_as_a_number():
    x = np.linspace(0.0, 1.0, 200)
    out = dependence_block({"a": x, "b": 2.0 * x + 1.0}, [("a", "b")])
    assert out["a_vs_b"]["correlation"] == pytest.approx(1.0, abs=1e-12)
    assert "undefined_because_constant" not in out["a_vs_b"]


def test_both_constant_coordinates_are_named():
    out = dependence_block({"a": np.ones(10), "b": np.zeros(10)}, [("a", "b")])
    assert out["a_vs_b"]["undefined_because_constant"] == ["a", "b"]


# --------------------------------------------------------------- mixed energy envelope
def _prior_cloud(n, seed, shift=0.0, fixed_rho=None):
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(sigma_rho_matrix(2, 0.4))
    u = rng.normal(size=(n, 3, 2)) @ chol.T + shift
    scalars = {"rho": (np.full(n, fixed_rho) if fixed_rho is not None
                       else rng.uniform(0.0, 0.99, size=n)),
               "beta": rng.gamma(2.0, 0.5, size=n) + shift,
               "omega": rng.normal(0.0, 2.0, size=n),
               "lambda_rep": rng.gamma(2.0, 0.5, size=n),
               "lambda_back": rng.gamma(2.0, 0.5, size=n)}
    return u, scalars


@pytest.mark.parametrize("seed", (5, 13, 21))
def test_two_samples_from_the_same_law_sit_inside_the_envelope(seed):
    x_u, x_s = _prior_cloud(4000, 1)
    y_u, y_s = _prior_cloud(12000, 2)
    out = mixed_envelope(x_u, x_s, y_u, y_s, seed=seed, n_compare=1500)
    assert out["pass"] is True
    assert out["observed"] <= out["envelope"]
    assert out["n_coordinates"] > 0


def test_a_genuinely_shifted_sample_is_caught():
    x_u, x_s = _prior_cloud(4000, 3, shift=1.2)
    y_u, y_s = _prior_cloud(12000, 4)
    out = mixed_envelope(x_u, x_s, y_u, y_s, seed=6, n_compare=1500)
    assert out["pass"] is False
    assert out["observed"] > 10.0 * out["envelope"]


def test_the_weighted_reference_is_resampled_before_the_unweighted_statistic():
    """Comparing a weighted cloud against an unweighted statistic mis-states the null.

    Here the reference's second half is the shifted one and carries almost no weight, so
    after resampling the reference must look like the *unshifted* MCMC cloud. Ignoring
    the weights would make the same comparison fail.
    """
    x_u, x_s = _prior_cloud(4000, 7)
    plain_u, plain_s = _prior_cloud(6000, 8)
    shifted_u, shifted_s = _prior_cloud(6000, 9, shift=1.5)
    y_u = np.concatenate([plain_u, shifted_u])
    y_s = {k: np.concatenate([plain_s[k], shifted_s[k]]) for k in plain_s}
    weights = np.concatenate([np.full(6000, 1.0), np.full(6000, 1e-6)])

    weighted = mixed_envelope(x_u, x_s, y_u, y_s, weights, seed=9, n_compare=1500)
    ignored = mixed_envelope(x_u, x_s, y_u, y_s, None, seed=9, n_compare=1500)
    assert weighted["pass"] is True
    assert ignored["pass"] is False
    assert weighted["observed"] < ignored["observed"]


def test_constant_coordinates_are_dropped_and_counted_rather_than_dividing_by_zero():
    """A `rho` that never moves carries no information and must not divide by its sd."""
    x_u, x_s = _prior_cloud(2000, 10, fixed_rho=0.4)
    y_u, y_s = _prior_cloud(4000, 11, fixed_rho=0.4)
    out = mixed_envelope(x_u, x_s, y_u, y_s, seed=12, n_compare=800, n_replicates=20)
    assert out["dropped_constant_coordinates"] >= 1
    assert np.isfinite(out["observed"]) and np.isfinite(out["envelope"])


# ------------------------------------------------------- §13 column-permutation audit
def test_the_audit_detects_chains_sitting_in_opposite_column_labellings():
    rng = np.random.default_rng(21)
    base = rng.normal(size=(400, 4, 2)) + np.array([1.5, -1.5])
    swapped = base[:, :, ::-1]
    audit = column_permutation_audit(np.array([base, base, swapped, swapped]))
    assert audit["target_is_column_exchangeable"] is True
    assert audit["chains_in_opposite_labellings"] is True
    signed = audit["per_chain_signed_column_contrast"]
    assert signed[0] > 0 and signed[2] < 0
    # the invariant summary is untouched by the relabelling
    absolute = audit["per_chain_absolute_column_contrast"]
    assert absolute[0] == pytest.approx(absolute[2], rel=1e-12)


def test_label_switching_inflates_the_signed_rhat_only():
    rng = np.random.default_rng(22)
    base = rng.normal(size=(400, 4, 2)) + np.array([1.5, -1.5])
    audit = column_permutation_audit(np.array([base, base[:, :, ::-1]]))
    assert audit["signed_contrast_rhat"] > 1.5
    assert "label switching, not non-convergence" in audit["note"]


def test_chains_in_one_labelling_are_not_reported_as_switched():
    rng = np.random.default_rng(23)
    chains = rng.normal(size=(4, 300, 4, 2)) + np.array([1.5, -1.5])
    audit = column_permutation_audit(chains)
    assert audit["chains_in_opposite_labellings"] is False
    assert audit["signed_contrast_rhat"] < 1.05
