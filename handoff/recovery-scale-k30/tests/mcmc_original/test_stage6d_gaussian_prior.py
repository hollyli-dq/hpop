"""Stage 6D — the equicorrelated Gaussian prior and the rho update
(§18 areas 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14).

This is the §3 hard gate. The `rho`-dependent normaliser `-(m/2) log|Sigma_rho|` must be
present, must cancel in a `U` update at fixed `rho`, and must NOT cancel in a `rho` update
at fixed `U`. The load-bearing test is the negative control at the bottom: an
implementation with the determinant deleted must give a different, wrong `rho` target.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from hpop.mcmc_original.sampler_u import log_u_prior, sigma_rho_matrix
from hpop.mcmc_original.stage6d_frozen import (
    RHO_UPPER, log_det_sigma_rho, log_mvn_equicorrelated, log_rho_prior,
    rho_is_in_support, scaling_proposal, scaling_proposal_log_ratio, sigma_rho_inverse,
)

RHOS = (0.02, 0.15, 0.4, 0.65, 0.9, 0.98)
DIMS = (2, 3, 5, 8)


# ------------------------------------------------- area 3: rho is a COLUMN correlation
def test_rho_is_the_within_row_correlation_between_latent_columns():
    """`Var(U[j,k]) = 1` and `Cov(U[j,k], U[j,l]) = rho`, empirically, for k != l."""
    rng = np.random.default_rng(0)
    d, rho, n = 3, 0.6, 400_000
    draws = rng.normal(size=(n, d)) @ np.linalg.cholesky(sigma_rho_matrix(d, rho)).T
    covariance = np.cov(draws, rowvar=False)
    assert np.allclose(np.diag(covariance), 1.0, atol=0.01)
    off = covariance[~np.eye(d, dtype=bool)]
    assert np.allclose(off, rho, atol=0.01)


def test_rows_are_independent_under_the_prior():
    rng = np.random.default_rng(1)
    u = rng.normal(size=(5, 2))
    for rho in RHOS:
        by_row = sum(log_u_prior(u[[j]], rho) for j in range(u.shape[0]))
        assert log_u_prior(u, rho) == pytest.approx(by_row, abs=1e-10)


# --------------------------------------- areas 4, 5: analytic determinant and inverse
def test_analytic_log_determinant_matches_slogdet():
    for d in DIMS:
        for rho in RHOS:
            _, slogdet = np.linalg.slogdet(sigma_rho_matrix(d, rho))
            assert log_det_sigma_rho(d, rho) == pytest.approx(slogdet, abs=1e-11)


def test_analytic_log_determinant_matches_the_closed_form():
    """`|Sigma| = (1-rho)^(d-1) [1 + (d-1) rho]`, written out independently."""
    for d in DIMS:
        for rho in RHOS:
            expected = (d - 1) * math.log(1 - rho) + math.log(1 + (d - 1) * rho)
            assert log_det_sigma_rho(d, rho) == pytest.approx(expected, abs=1e-12)


def test_analytic_inverse_matches_numpy():
    for d in DIMS:
        for rho in RHOS:
            observed = sigma_rho_inverse(d, rho)
            assert np.allclose(observed, np.linalg.inv(sigma_rho_matrix(d, rho)),
                               atol=1e-10)
            assert np.allclose(observed @ sigma_rho_matrix(d, rho), np.eye(d), atol=1e-10)


def test_inverse_rejects_a_non_positive_definite_rho():
    with pytest.raises(ValueError):
        sigma_rho_inverse(3, -0.9)          # below -1/(d-1)
    with pytest.raises(ValueError):
        sigma_rho_inverse(3, 1.0)


# ------------------------------------ area 6: parity with a trusted dense computation
def test_density_matches_scipy_dense_multivariate_normal():
    """Many deterministic (d, rho, U): the frozen density vs scipy's dense MVN."""
    rng = np.random.default_rng(2)
    worst = 0.0
    for d in DIMS:
        for rho in RHOS:
            sigma = sigma_rho_matrix(d, rho)
            u = rng.normal(size=(6, d)) @ np.linalg.cholesky(sigma).T
            expected = float(stats.multivariate_normal(mean=np.zeros(d), cov=sigma)
                             .logpdf(u).sum())
            worst = max(worst, abs(log_u_prior(u, rho) - expected))
    assert worst < 1e-9


def test_analytic_route_matches_the_cholesky_route():
    """`log_mvn_equicorrelated` uses the closed-form det and inverse; `log_u_prior` a
    Cholesky solve. Agreement means the determinant is present in both."""
    rng = np.random.default_rng(3)
    worst = 0.0
    for d in DIMS:
        for rho in RHOS:
            u = rng.normal(size=(5, d))
            worst = max(worst, abs(log_mvn_equicorrelated(u, rho) - log_u_prior(u, rho)))
    assert worst < 1e-9


def test_assessor_residual_density_is_not_applicable():
    """§18 area 7: this model has no assessor level and no tau, stated not skipped."""
    from hpop.mcmc_original.stage6d_frozen import frozen_config, load_stage6d_dataset
    config = frozen_config()
    assert config["tau_in_model"] is False
    assert config["assessor_level_in_model"] is False
    assert load_stage6d_dataset().n_assessors == 0


# ------------------------- areas 8, 9: cancellation in U, non-cancellation in rho
def test_determinant_cancels_in_a_u_move_at_fixed_rho():
    rng = np.random.default_rng(4)
    d, rho = 2, 0.45
    a, b = rng.normal(size=(5, d)), rng.normal(size=(5, d))

    def quadratic(u):
        return float(np.einsum("ij,jk,ik->", u, sigma_rho_inverse(d, rho), u))

    difference = log_u_prior(b, rho) - log_u_prior(a, rho)
    assert difference == pytest.approx(-0.5 * (quadratic(b) - quadratic(a)), abs=1e-10)


def test_determinant_does_not_cancel_in_a_rho_move_at_fixed_u():
    rng = np.random.default_rng(5)
    d, m = 2, 5
    u = rng.normal(size=(m, d))
    lo, hi = 0.2, 0.8

    def quadratic(rho):
        return float(np.einsum("ij,jk,ik->", u, sigma_rho_inverse(d, rho), u))

    determinant_part = -0.5 * m * (log_det_sigma_rho(d, hi) - log_det_sigma_rho(d, lo))
    quadratic_part = -0.5 * (quadratic(hi) - quadratic(lo))
    assert abs(determinant_part) > 1e-3
    assert (log_u_prior(u, hi) - log_u_prior(u, lo)) == pytest.approx(
        determinant_part + quadratic_part, abs=1e-10)


# ------------------------------- area 10: the omitted-determinant negative control
def _no_determinant_log_prior(u, rho):
    """Deliberately wrong: the quadratic form only."""
    u = np.asarray(u, dtype=float)
    m, d = u.shape
    quadratic = float(np.einsum("ij,jk,ik->", u, sigma_rho_inverse(d, rho), u))
    return -0.5 * (m * d * math.log(2 * math.pi) + quadratic)


def _rho_posterior_mean(prior_fn, u, grid):
    density = np.array([prior_fn(u, float(r)) + log_rho_prior(float(r)) for r in grid])
    w = np.exp(density - density.max())
    return float(np.trapezoid(grid * w, grid) / np.trapezoid(w, grid))


def test_omitting_the_determinant_changes_the_rho_target():
    rng = np.random.default_rng(2026)
    grid = np.linspace(1e-3, RHO_UPPER - 1e-3, 400)
    shifts = []
    for _ in range(12):
        u = rng.normal(size=(5, 2))
        shifts.append(abs(_rho_posterior_mean(log_u_prior, u, grid)
                          - _rho_posterior_mean(_no_determinant_log_prior, u, grid)))
    assert min(shifts) > 0.005
    assert float(np.mean(shifts)) > 0.03, (
        f"deleting -(m/2)log|Sigma_rho| barely moved the rho posterior: {shifts}")


def test_the_two_priors_differ_by_exactly_the_determinant_term():
    rng = np.random.default_rng(6)
    for rho in RHOS:
        differences = [log_u_prior(u, rho) - _no_determinant_log_prior(u, rho)
                       for u in (rng.normal(size=(5, 2)) for _ in range(4))]
        assert np.allclose(differences, differences[0], atol=1e-10)
        assert differences[0] == pytest.approx(-0.5 * 5 * log_det_sigma_rho(2, rho),
                                               abs=1e-10)


# ------------------------------------------- areas 11, 12, 13, 14: the rho prior/proposal
def test_the_registered_rho_prior_is_uniform_not_beta_1_one_sixth():
    """The brief names Beta(1, 1/6); the frozen configuration establishes Beta(1, 1)."""
    from hpop.mcmc_original.stage6d_frozen import frozen_config
    spec = frozen_config()["inherits_from_stage6c"]["rho_prior"]
    assert spec["family"] == "beta" and spec["a"] == 1.0 and spec["b"] == 1.0
    assert spec["truncated_at"] == pytest.approx(0.995)

    values = [log_rho_prior(r) for r in (0.05, 0.3, 0.5, 0.8, 0.99)]
    assert np.allclose(values, values[0], atol=1e-12)         # flat => Uniform
    assert math.exp(values[0]) == pytest.approx(1.0 / RHO_UPPER, abs=1e-9)

    # and it is NOT Beta(1, 1/6), which would be strongly decreasing
    beta_one_sixth = stats.beta(1.0, 1.0 / 6.0)
    assert not np.allclose([beta_one_sixth.logpdf(r) for r in (0.05, 0.8)],
                           values[0], atol=1e-6)


def test_the_registered_divergences_are_recorded():
    from hpop.mcmc_original.stage6d_frozen import frozen_config
    sections = {d["section"] for d in frozen_config()["spec_divergences"]}
    assert {"2", "4"} <= sections


def test_scaling_proposal_ratio_is_minus_log_delta():
    """§18 area 13, for the brief's proposal — implemented, tested, not production."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        rho = float(rng.uniform(0.01, 0.98))
        proposed, delta, ratio = scaling_proposal(rho, 0.1, rng)
        assert ratio == pytest.approx(-math.log(delta), abs=1e-12)
        # the map is the one the brief states
        assert proposed == pytest.approx(1.0 - (1.0 - rho) * delta, abs=1e-12)
        assert delta == pytest.approx((1.0 - proposed) / (1.0 - rho), abs=1e-10)


def test_scaling_proposal_density_is_symmetric_under_inversion():
    """Forward with `delta` and reverse with `1/delta` must give opposite log ratios."""
    for delta in (0.15, 0.5, 1.0, 2.0, 7.5):
        assert scaling_proposal_log_ratio(delta) == pytest.approx(
            -scaling_proposal_log_ratio(1.0 / delta), abs=1e-12)


def test_scaling_proposal_rejects_a_bad_step_parameter():
    rng = np.random.default_rng(8)
    for bad in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(ValueError):
            scaling_proposal(0.4, bad, rng)


def test_invalid_rho_is_outside_the_registered_support():
    assert not rho_is_in_support(0.0)
    assert not rho_is_in_support(RHO_UPPER)
    assert not rho_is_in_support(0.999)
    assert not rho_is_in_support(-0.2)
    assert rho_is_in_support(0.5)
    assert log_rho_prior(0.999) == -math.inf


def test_an_invalid_rho_proposal_is_rejected_immediately_and_counted():
    """The sweep must count an out-of-support proposal separately from a rejection."""
    from hpop.mcmc_original.recurrent_latent_poset_mcmc import LatentPosetEvaluator
    from hpop.mcmc_original.recurrent_oracle_joint_mcmc import (
        Stage6DTarget, initial_state, sweep_once,
    )
    from hpop.mcmc_original.stage6d_frozen import REGISTERED_SCALES, load_stage6d_dataset

    frozen = load_stage6d_dataset()
    evaluator = LatentPosetEvaluator(frozen.train[:10], epsilon=frozen.epsilon,
                                     omega=frozen.truth["omega"])
    target = Stage6DTarget(evaluator)
    rng = np.random.default_rng(0)
    values = {"rho": 0.99, **{k: float(v) for k, v in frozen.truth.items()}}
    state = initial_state(target, frozen.u_true, values, rng)

    # a huge rho step from rho = 0.99 lands outside the support very often
    scales = {**REGISTERED_SCALES, "rho": 6.0}
    for _ in range(60):
        state = sweep_once(state, target, scales, rng)
    assert state.invalid["rho"] > 0
    assert state.invalid["rho"] + state.accepted["rho"] <= state.proposed["rho"]
