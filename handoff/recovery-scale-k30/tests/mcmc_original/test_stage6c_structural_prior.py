"""Stage 6C — the structural prior `p(U | rho)` (§17 areas 8, 9, 10).

This is the §2.1 hard gate. The registered prior is a *density on `R^{m x d}`*, not a
score over a set of partial orders, so the failure mode the brief anticipates — an
unnormalised relation-counting score whose `rho`-dependent constant is missing — cannot
arise in the form it describes. The `rho`-dependent normaliser that does exist is the
Gaussian determinant

    log|Sigma_rho| = (d-1) log(1-rho) + log(1 + (d-1) rho),

contributing `-(m/2) log|Sigma_rho|`. It cancels in a `U` update at fixed `rho` and does
NOT cancel in a `rho` update at fixed `U`.

The negative control at the bottom is the load-bearing test: an implementation with that
term deleted is shown to give a *different and wrong* `rho` posterior. Without it, the
normalisation tests above would pass just as happily against a decorative constant.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, stats

from hpop.mcmc_original.sampler_u import log_u_prior, sigma_rho_matrix
from hpop.mcmc_original.stage6c_frozen import (
    RHO_UPPER, log_det_sigma_rho, log_rho_prior, log_structural_prior,
)

RHO_GRID = (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)


# ------------------------------------------------------- area 8: p(U|rho) normalised
def test_matches_scipy_multivariate_normal_row_by_row():
    rng = np.random.default_rng(0)
    for rho in RHO_GRID:
        sigma = sigma_rho_matrix(2, rho)
        u = rng.normal(size=(5, 2)) @ np.linalg.cholesky(sigma).T
        expected = float(stats.multivariate_normal(mean=np.zeros(2), cov=sigma)
                         .logpdf(u).sum())
        assert log_u_prior(u, rho) == pytest.approx(expected, abs=1e-9)


def test_integrates_to_one_by_quadrature_for_one_row():
    """`m = 1, d = 2` is small enough for genuine 2-D quadrature, not a Monte Carlo proxy."""
    for rho in (0.1, 0.5, 0.9):
        def density(x, y, rho=rho):
            return math.exp(log_u_prior(np.array([[x, y]]), rho))

        mass, error = integrate.dblquad(
            density, -12.0, 12.0, lambda _: -12.0, lambda _: 12.0,
            epsabs=1e-11, epsrel=1e-11)
        assert mass == pytest.approx(1.0, abs=1e-9)
        assert error < 1e-8


def test_normalisation_for_five_rows_follows_exactly_from_row_factorisation():
    """`m = 5` needs no Monte Carlo: it is five independent copies of the `m = 1` density.

    A 10-dimensional importance estimate would carry enough variance to be a weak check.
    The exact argument is stronger: the joint is the product of the row densities (asserted
    here), and each row density integrates to 1 by the quadrature test above, so the joint
    integrates to 1 identically.
    """
    rng = np.random.default_rng(7)
    for rho in (0.1, 0.5, 0.9):
        sigma = sigma_rho_matrix(2, rho)
        u = rng.normal(size=(5, 2)) @ np.linalg.cholesky(sigma).T
        joint = log_u_prior(u, rho)
        product_of_rows = sum(log_u_prior(u[[i]], rho) for i in range(5))
        assert joint == pytest.approx(product_of_rows, abs=1e-11)

        # and each row density is itself a normalised 2-D Gaussian
        def density(x, y, rho=rho):
            return math.exp(log_u_prior(np.array([[x, y]]), rho))

        mass, _ = integrate.dblquad(density, -12.0, 12.0, lambda _: -12.0,
                                    lambda _: 12.0, epsabs=1e-11, epsrel=1e-11)
        assert mass == pytest.approx(1.0, abs=1e-9)


def test_rows_are_independent_so_the_prior_factorises():
    rng = np.random.default_rng(3)
    u = rng.normal(size=(5, 2))
    for rho in RHO_GRID:
        by_row = sum(log_u_prior(u[[i]], rho) for i in range(5))
        assert log_u_prior(u, rho) == pytest.approx(by_row, abs=1e-10)


# ------------------------------------- area 9: the rho-dependent normaliser is present
def test_closed_form_log_determinant_matches_numpy():
    for d in (2, 3, 5):
        for rho in RHO_GRID:
            _, slogdet = np.linalg.slogdet(sigma_rho_matrix(d, rho))
            assert log_det_sigma_rho(d, rho) == pytest.approx(slogdet, abs=1e-12)


def test_log_u_prior_contains_the_determinant_term():
    """Strip the quadratic form and what remains must be exactly the Gaussian constant."""
    rng = np.random.default_rng(11)
    m, d = 5, 2
    u = rng.normal(size=(m, d))
    for rho in RHO_GRID:
        sigma = sigma_rho_matrix(d, rho)
        quadratic = float(np.einsum("ij,jk,ik->", u, np.linalg.inv(sigma), u))
        constant = log_u_prior(u, rho) + 0.5 * quadratic
        expected = -0.5 * (m * d * math.log(2 * math.pi) + m * log_det_sigma_rho(d, rho))
        assert constant == pytest.approx(expected, abs=1e-9)


def test_determinant_cancels_in_a_u_move_but_not_in_a_rho_move():
    """The precise asymmetry the audit turns on, asserted rather than described."""
    rng = np.random.default_rng(5)
    u_a, u_b = rng.normal(size=(5, 2)), rng.normal(size=(5, 2))
    rho_a, rho_b = 0.3, 0.7

    def quadratic(u, rho):
        return float(np.einsum("ij,jk,ik->", u, np.linalg.inv(sigma_rho_matrix(2, rho)),
                               u))

    # U move at fixed rho: the difference is purely the quadratic forms
    u_move = log_u_prior(u_b, rho_a) - log_u_prior(u_a, rho_a)
    assert u_move == pytest.approx(-0.5 * (quadratic(u_b, rho_a) - quadratic(u_a, rho_a)),
                                   abs=1e-10)

    # rho move at fixed U: the determinant term survives and is non-zero
    rho_move = log_u_prior(u_a, rho_b) - log_u_prior(u_a, rho_a)
    determinant_part = -0.5 * 5 * (log_det_sigma_rho(2, rho_b)
                                   - log_det_sigma_rho(2, rho_a))
    assert abs(determinant_part) > 1e-3
    quadratic_part = -0.5 * (quadratic(u_a, rho_b) - quadratic(u_a, rho_a))
    assert rho_move == pytest.approx(determinant_part + quadratic_part, abs=1e-10)


def test_structural_prior_rejects_rho_outside_the_positive_definite_range():
    u = np.zeros((5, 2))
    assert log_structural_prior(u, 1.5) == -math.inf
    assert log_structural_prior(u, -1.5) == -math.inf
    assert math.isfinite(log_structural_prior(u, 0.5))


# ---------------------------------------------- area 10: the negative control must fail
def _omitted_normaliser_log_prior(u, rho):
    """Deliberately wrong: the quadratic form only, with `m log|Sigma_rho|` deleted."""
    u = np.asarray(u, dtype=float)
    m, d = u.shape
    sigma = sigma_rho_matrix(d, rho)
    quadratic = float(np.einsum("ij,jk,ik->", u, np.linalg.inv(sigma), u))
    return -0.5 * (m * d * math.log(2 * math.pi) + quadratic)


def _rho_posterior_mean(log_prior_fn, u, grid):
    log_density = np.array([log_prior_fn(u, float(r)) + log_rho_prior(float(r))
                            for r in grid])
    weights = np.exp(log_density - log_density.max())
    return float(np.trapezoid(grid * weights, grid) / np.trapezoid(weights, grid))


def test_omitting_the_normaliser_changes_the_rho_posterior():
    """The control: deleting the determinant term must visibly move the rho posterior.

    If this test ever passes-by-agreeing, the normaliser is not doing any work and the
    §2.1 gate would be vacuous.
    """
    rng = np.random.default_rng(2024)
    grid = np.linspace(1e-3, RHO_UPPER - 1e-3, 400)
    shifts = []
    for _ in range(12):
        u = rng.normal(size=(5, 2))
        correct = _rho_posterior_mean(log_u_prior, u, grid)
        broken = _rho_posterior_mean(_omitted_normaliser_log_prior, u, grid)
        shifts.append(abs(correct - broken))

    # The shift is systematic, not incidental: the broken variant is biased for every
    # draw, and on average by a large fraction of the rho posterior's own spread
    # (the reference rho posterior has sd about 0.19).
    assert min(shifts) > 0.005, (
        f"omitting -(m/2)log|Sigma_rho| left the rho posterior essentially unmoved for "
        f"some draw: {shifts}; the normaliser must be load-bearing")
    assert float(np.mean(shifts)) > 0.03, (
        f"mean rho posterior shift {np.mean(shifts):.4f} is too small to demonstrate "
        f"that the normaliser matters; shifts were {shifts}")


def test_omitted_normaliser_does_not_integrate_to_one():
    """The broken variant is not a density — the direct reason it must not be used."""
    rho = 0.8
    def density(x, y):
        return math.exp(_omitted_normaliser_log_prior(np.array([[x, y]]), rho))

    mass, _ = integrate.dblquad(density, -12.0, 12.0, lambda _: -12.0, lambda _: 12.0,
                                epsabs=1e-10, epsrel=1e-10)
    assert abs(mass - 1.0) > 0.1


def test_the_correct_prior_and_the_broken_one_agree_only_up_to_a_rho_constant():
    """Their difference must be exactly `-(m/2) log|Sigma_rho|` — free of `U`."""
    rng = np.random.default_rng(99)
    for rho in RHO_GRID:
        differences = []
        for _ in range(4):
            u = rng.normal(size=(5, 2))
            differences.append(log_u_prior(u, rho) - _omitted_normaliser_log_prior(u, rho))
        assert np.allclose(differences, differences[0], atol=1e-10)
        assert differences[0] == pytest.approx(-0.5 * 5 * log_det_sigma_rho(2, rho),
                                               abs=1e-10)
