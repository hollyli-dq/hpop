"""Stage 6D1 — the independent continuous-latent QMC reference.

The reference exists to be *wrong in different ways* from the sampler, so these tests
concentrate on the places where a shared error could sneak back in:

* **the non-centred construction must reproduce the centred density.** `U = Z L(rho)^T`
  removes the Gaussian determinant from the importance weight entirely, which is the
  point — a determinant error in the sampler then cannot hide behind the same error in
  the reference. The change of variables is checked against `sampler_u.log_u_prior`
  rather than asserted in prose.
* **the proposal must be exactly the prior**, so the weight collapses to the likelihood
  alone. If it were not, the weights would silently carry leftover prior ratios.
* **`rqmc_se` must shrink with R.** The superseded statistic — the maximum spread across
  replicates — estimates the dispersion of a *single* replicate and therefore samples
  further into the tail as R grows. The registered precision statistic is the standard
  error of the averaged reference, and the difference between the two is enforced here
  rather than left as a comment.
* **replicate combination must align by canonical key**, because independent scrambles
  visit different induced-order sets.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_joint_scalar_mcmc import vectorized_state_features
from hpop.mcmc_original.recurrent_scalar_posterior import (
    PRIORS, cached_batch_log_likelihood,
)
from hpop.mcmc_original.sampler_u import log_u_prior, sigma_rho_matrix
from hpop.mcmc_original.stage6d_frozen import RHO_UPPER, log_det_sigma_rho
from hpop.mcmc_original.stage6d_joint_reference import (
    build_state, centred_matches_non_centred, combine_replicates, induced_h_labels,
    prior_inverse_cdf, qmc_dimension, qmc_replicate, replicate_summary,
    rqmc_standard_error, small_model, sobol_points,
)

SCALARS = ("beta", "omega", "lambda_rep", "lambda_back")


@pytest.fixture(scope="module")
def model():
    return small_model()


# ------------------------------------------------------------- the registered problem
def test_the_small_model_dimensions_are_the_registered_ones(model):
    """Frozen before any MCMC comparison existed; a silent change invalidates the gate."""
    assert (model.m, model.d, model.n_blocks, model.T) == (3, 2, 3, 5)
    assert model.n_skills == 1
    assert model.epsilon == 0.02
    assert model.roles.shape == (3, 5)
    assert set(model.truth) == set(SCALARS)


def test_the_small_model_induced_order_is_neither_an_antichain_nor_a_total_order(model):
    """The reference needs genuine structural uncertainty to represent."""
    closure = precedence_from_u(model.u_true)
    assert closure[0, 1] and closure[0, 2]
    assert not closure[1, 2] and not closure[2, 1]
    assert 0 < closure.sum() < 3


def test_the_qmc_dimension_is_one_rho_plus_m_times_d_normals_plus_four_scalars(model):
    assert qmc_dimension(3, 2) == 1 + 3 * 2 + 4 == 11
    assert model.qmc_dimension == 11
    assert qmc_dimension(5, 2) == 15


def test_the_model_is_deterministic_given_its_seed():
    assert np.array_equal(small_model().roles, small_model().roles)
    assert not np.array_equal(small_model(1).roles, small_model(2).roles)


# ------------------------------------------------------------------ prior inverse CDFs
def test_the_rho_inverse_cdf_is_the_truncated_uniform_the_frozen_prior_registers():
    """Beta(1,1) truncated at 1 - 5e-3 and renormalised is Uniform(0, RHO_UPPER)."""
    u = np.array([0.0, 0.25, 0.5, 1.0])
    assert np.allclose(prior_inverse_cdf("rho", u), u * RHO_UPPER)
    assert prior_inverse_cdf("rho", 1.0) == pytest.approx(RHO_UPPER)
    assert prior_inverse_cdf("rho", 0.999999) < 1.0


@pytest.mark.parametrize("name", SCALARS)
def test_each_scalar_inverse_cdf_inverts_its_registered_prior_cdf(name):
    spec = PRIORS[name]
    dist = (stats.gamma(a=spec["shape"], scale=1.0 / spec["rate"])
            if spec["family"] == "gamma"
            else stats.norm(loc=spec["mean"], scale=spec["sd"]))
    u = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    assert np.allclose(dist.cdf(prior_inverse_cdf(name, u)), u, atol=1e-12)


def test_inverse_cdf_transformed_uniforms_reproduce_the_prior_moments():
    """A quadrature check that the mapping really is the prior, not merely monotone."""
    u = (np.arange(200_000) + 0.5) / 200_000
    for name in SCALARS:
        spec = PRIORS[name]
        expected = (spec["shape"] / spec["rate"] if spec["family"] == "gamma"
                    else spec["mean"])
        assert prior_inverse_cdf(name, u).mean() == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------- the non-centred construction
def test_the_non_centred_construction_reproduces_the_centred_density(model):
    """`log p_U(U) = log p_Z(Z) - (m/2) log|Sigma_rho|`, to machine precision."""
    points = sobol_points(64, model.qmc_dimension, seed=3)
    state = build_state(points, model)
    for i in range(0, 64, 7):
        gap = centred_matches_non_centred(state["u"][i], state["z"][i],
                                          float(state["rho"][i]))
        assert gap < 1e-9


def test_u_equals_z_times_the_cholesky_factor_of_sigma_rho(model):
    points = sobol_points(32, model.qmc_dimension, seed=4)
    state = build_state(points, model)
    for i in (0, 5, 17, 31):
        chol = np.linalg.cholesky(sigma_rho_matrix(model.d, float(state["rho"][i])))
        assert np.allclose(state["u"][i], state["z"][i] @ chol.T, atol=1e-12)


def test_the_rows_of_u_carry_the_registered_equicorrelation(model):
    """`Var(U[j,k]) = 1` and `Cov(U[j,0], U[j,1]) = rho`, empirically."""
    rng = np.random.default_rng(7)
    rho = 0.6
    z = rng.normal(size=(200_000, model.m, model.d))
    u = z @ np.linalg.cholesky(sigma_rho_matrix(model.d, rho)).T
    flat = u.reshape(-1, model.d)
    assert flat.var(axis=0) == pytest.approx([1.0, 1.0], abs=0.02)
    assert np.corrcoef(flat.T)[0, 1] == pytest.approx(rho, abs=0.01)


def test_the_change_of_variables_term_is_the_half_log_determinant(model):
    """Deleting `-(m/2) log|Sigma_rho|` must break the identity — a negative control."""
    points = sobol_points(16, model.qmc_dimension, seed=5)
    state = build_state(points, model)
    i = 3
    rho = float(state["rho"][i])
    correct = float(stats.norm.logpdf(state["z"][i]).sum()) \
        - 0.5 * model.m * log_det_sigma_rho(model.d, rho)
    without = float(stats.norm.logpdf(state["z"][i]).sum())
    assert log_u_prior(state["u"][i], rho) == pytest.approx(correct, abs=1e-9)
    assert abs(log_u_prior(state["u"][i], rho) - without) > 1e-3


def test_build_state_maps_the_scalar_columns_in_the_registered_order(model):
    """rho, then m*d normals, then beta, omega, lambda_rep, lambda_back."""
    points = np.full((1, model.qmc_dimension), 0.5)
    points[0, 1 + model.m * model.d + 1] = 0.9        # the omega column
    state = build_state(points, model)
    assert state["omega"][0] == pytest.approx(prior_inverse_cdf("omega", 0.9))
    for k, name in enumerate(SCALARS):
        if name != "omega":
            assert state[name][0] == pytest.approx(prior_inverse_cdf(name, 0.5))


# --------------------------------------------------------------------- Sobol mechanics
def test_sobol_points_stay_strictly_inside_the_open_cube():
    """Sobol emits exact 0 and 1; an inverse CDF at either endpoint is infinite."""
    points = sobol_points(1024, 11, seed=1)
    assert points.shape == (1024, 11)
    assert points.min() > 0.0 and points.max() < 1.0
    assert np.all(np.isfinite(stats.norm.ppf(points)))


def test_independent_scrambles_give_different_points_but_the_same_uniformity():
    a, b = sobol_points(4096, 11, seed=1), sobol_points(4096, 11, seed=2)
    assert not np.allclose(a, b)
    assert a.mean(axis=0) == pytest.approx(0.5, abs=0.02)
    assert b.mean(axis=0) == pytest.approx(0.5, abs=0.02)


def test_the_same_seed_reproduces_the_same_replicate(model):
    a = qmc_replicate(model, 512, seed=11)
    b = qmc_replicate(model, 512, seed=11)
    assert a["log_evidence"] == pytest.approx(b["log_evidence"], rel=0, abs=0)
    assert np.array_equal(a["weights"], b["weights"])


# ------------------------------------------------------------------ importance weights
def test_the_importance_weight_is_the_likelihood_alone(model):
    """The proposal is exactly the prior, so every prior density cancels analytically."""
    replicate = qmc_replicate(model, 256, seed=13)
    for i in (0, 63, 200):
        features = vectorized_state_features(model.roles, replicate["u"][i],
                                             float(replicate["omega"][i]))
        expected = float(cached_batch_log_likelihood(
            features, float(replicate["beta"][i]), model.epsilon,
            float(replicate["lambda_rep"][i]), float(replicate["lambda_back"][i])))
        assert replicate["log_weights"][i] == pytest.approx(expected, abs=1e-12)


def test_the_weights_are_normalised_and_the_ess_is_bounded_by_the_point_count(model):
    replicate = qmc_replicate(model, 1024, seed=17)
    assert replicate["weights"].sum() == pytest.approx(1.0, abs=1e-12)
    assert 0.0 < replicate["ess"] <= 1024
    assert replicate["relative_ess"] == pytest.approx(replicate["ess"] / 1024)
    assert replicate["max_normalised_weight"] < 1.0


def test_the_maximum_normalised_weight_falls_as_the_point_count_grows(model):
    """The registered quality criterion: no single point may come to dominate."""
    small = qmc_replicate(model, 512, seed=19)["max_normalised_weight"]
    large = qmc_replicate(model, 8192, seed=19)["max_normalised_weight"]
    assert large < small


def test_the_log_evidence_converges_as_the_point_count_grows(model):
    """Stability is a property of `n_points`, not an unconditional claim."""
    coarse = [qmc_replicate(model, 1024, seed=s)["log_evidence"] for s in (21, 22, 23)]
    fine = [qmc_replicate(model, 16384, seed=s)["log_evidence"] for s in (21, 22, 23)]
    assert all(math.isfinite(v) for v in coarse + fine)
    assert max(fine) - min(fine) < (max(coarse) - min(coarse)) / 4.0
    assert max(fine) - min(fine) < 0.05


def test_the_log_evidence_agrees_with_the_frozen_reference(model):
    """The committed reference reports -15.24703 +/- 0.00185 over 32 replicates."""
    fine = [qmc_replicate(model, 16384, seed=s)["log_evidence"] for s in (21, 22, 23)]
    assert float(np.mean(fine)) == pytest.approx(-15.247028617891512, abs=0.02)


# -------------------------------------------------------------------- replicate summary
def test_the_summary_probabilities_and_marginals_are_coherent(model):
    summary = replicate_summary(qmc_replicate(model, 2048, seed=31), model)
    assert summary["h_probability"].sum() == pytest.approx(1.0, abs=1e-12)
    assert summary["relation_count_distribution"].sum() == pytest.approx(1.0, abs=1e-12)
    assert summary["n_induced_h_states"] == len(summary["h_keys"])
    marginal = summary["relation_marginal"].reshape(model.m, model.m)
    assert np.allclose(np.diag(marginal), 0.0)
    assert np.all((marginal >= 0.0) & (marginal <= 1.0))
    # a strict order cannot hold i>j and j>i at once
    assert np.all(marginal + marginal.T <= 1.0 + 1e-12)


def test_the_weighted_scalar_summaries_are_ordered_and_finite(model):
    summary = replicate_summary(qmc_replicate(model, 2048, seed=33), model)
    for name in ("rho",) + SCALARS:
        s = summary["scalars"][name]
        assert s["q025"] <= s["median"] <= s["q975"]
        assert s["sd"] > 0.0
        assert math.isfinite(s["mean"])
    assert 0.0 < summary["scalars"]["rho"]["mean"] < RHO_UPPER


def test_the_correlation_matrix_is_a_correlation_matrix(model):
    summary = replicate_summary(qmc_replicate(model, 2048, seed=35), model)
    corr = summary["correlation"]
    assert corr.shape == (6, 6)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-9)
    assert np.allclose(corr, corr.T, atol=1e-12)
    assert np.all(np.abs(corr) <= 1.0 + 1e-9)
    assert summary["correlation_names"][-1] == "relation_count"


def test_induced_h_labels_are_canonical_and_first_appearance_ordered(model):
    u = np.array([model.u_true, model.u_true * 3.0, -model.u_true, model.u_true])
    labels, keys, closures = induced_h_labels(u)
    assert labels[0] == labels[1] == labels[3]      # a positive rescale cannot relabel
    assert labels[2] != labels[0]                   # negation reverses the order
    assert len(keys) == 2 == closures.shape[0]
    assert np.array_equal(closures[0], precedence_from_u(model.u_true))


# ----------------------------------------------------- the registered precision statistic
def test_rqmc_se_is_the_standard_error_of_the_averaged_reference():
    rng = np.random.default_rng(41)
    estimates = rng.normal(size=(32, 5))
    result = rqmc_standard_error(estimates)
    assert np.allclose(result["standard_error"],
                       estimates.std(axis=0, ddof=1) / math.sqrt(32))
    assert result["n_replicates"] == 32
    assert result["t_multiplier"] == pytest.approx(float(stats.t.ppf(0.975, 31)))
    assert np.allclose(result["half_width_95"],
                       result["t_multiplier"] * result["standard_error"])


def test_rqmc_se_shrinks_with_R_while_the_max_over_replicates_does_not():
    """This is the supersession, enforced rather than described.

    `rqmc_se` estimates how precisely the *averaged* reference is known and falls as
    1/sqrt(R). The maximum departure of any single replicate from the mean estimates the
    dispersion of one replicate: it can only grow with R, because more replicates sample
    further into the tail. Reading the second as an uncertainty on the first is the error
    that the Stage 6D1 registration was corrected for.
    """
    rng = np.random.default_rng(43)
    draws = rng.normal(size=(4096, 1))
    se, spread = [], []
    for r in (16, 64, 256, 1024, 4096):
        block = draws[:r]
        se.append(rqmc_standard_error(block)["max_standard_error"])
        spread.append(float(np.abs(block - block.mean(axis=0)).max()))
    assert se == sorted(se, reverse=True)                 # strictly shrinking
    assert se[0] / se[-1] > 8.0                           # about sqrt(4096/16) = 16
    assert spread == sorted(spread)                       # never shrinks
    assert spread[-1] > spread[0]


def test_rqmc_se_refuses_a_single_replicate():
    with pytest.raises(ValueError):
        rqmc_standard_error(np.array([[1.0, 2.0]]))


# ------------------------------------------------------------------ combining replicates
def test_replicates_are_combined_on_the_union_of_their_induced_order_keys(model):
    """Independent scrambles visit different order sets; positional merging would be wrong."""
    summaries = [replicate_summary(qmc_replicate(model, 1024, seed=s), model)
                 for s in (51, 52, 53)]
    combined = combine_replicates(summaries)
    union = set(combined["h_keys"])
    for s in summaries:
        assert set(s["h_keys"]) <= union
    assert combined["per_replicate_h_probability"].shape == (3, len(union))
    assert combined["pooled_h_probability"].sum() == pytest.approx(1.0, abs=1e-12)
    # every replicate's mass is preserved, wherever its keys landed in the union
    assert np.allclose(combined["per_replicate_h_probability"].sum(axis=1), 1.0,
                       atol=1e-12)


def test_the_combination_reports_both_the_primary_precision_and_the_secondary_dispersion(
        model):
    summaries = [replicate_summary(qmc_replicate(model, 1024, seed=s), model)
                 for s in (61, 62, 63, 64)]
    combined = combine_replicates(summaries)
    precision = combined["precision"]
    assert "rqmc_se" in precision["definition"]
    assert precision["max_structural_half_width_95"] == pytest.approx(max(
        precision["h_probability"]["max_half_width_95"],
        precision["relation_marginal"]["max_half_width_95"]))
    dispersion = combined["replicate_dispersion"]
    assert len(dispersion["per_replicate_h_total_variation"]) == 4
    assert dispersion["max_h_total_variation_from_mean"] == pytest.approx(
        max(dispersion["per_replicate_h_total_variation"]))
    # the two are different quantities and must not be confused
    assert (precision["max_structural_half_width_95"]
            != dispersion["max_relation_departure_from_mean"])


def test_the_pooled_relation_marginal_is_the_mean_over_replicates(model):
    summaries = [replicate_summary(qmc_replicate(model, 1024, seed=s), model)
                 for s in (71, 72)]
    combined = combine_replicates(summaries)
    expected = np.mean([s["relation_marginal"] for s in summaries], axis=0)
    assert np.allclose(combined["pooled_relation_marginal"], expected, atol=1e-12)
    assert combined["n_replicates"] == 2
    assert combined["log_evidence"]["mean"] == pytest.approx(
        float(np.mean([s["log_evidence"] for s in summaries])))
