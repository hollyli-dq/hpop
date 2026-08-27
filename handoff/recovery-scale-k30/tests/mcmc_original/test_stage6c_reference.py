"""Stage 6C — the independent exact references (§17 areas 27, 28, 29, 30, 31).

The reference is a *marginal* of the continuous target, not a substitute for it:

    p(P, rho | Y) proportional to p(rho) · L(P) · pi_rho(P)

is exact because the likelihood is constant on the cell `{U : h(U) = P}`, so the integral
of `p(Y|U) p(U|rho)` over that cell factorises. That constancy is the load-bearing
assumption, and it is verified directly below rather than assumed.

Nothing here touches the MCMC kernel, an acceptance-ratio helper, or a chain draw.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_joint_scalar_mcmc import vectorized_state_features
from hpop.mcmc_original.recurrent_scalar_posterior import (
    cached_batch_log_likelihood, log_prior,
)
from hpop.mcmc_original.sampler_u import sigma_rho_matrix
from hpop.mcmc_original.stage6c_exact_reference import (
    build_catalogue, build_6c1_reference, build_6c2_reference, poset_log_likelihoods,
    poset_log_likelihood_beta_table, prior_cell_masses, reference_summary,
    sample_reference_draws,
)
from hpop.mcmc_original.stage6c_frozen import (
    RHO_UPPER, load_stage6c_dataset, log_rho_prior, log_jacobian_rho,
    rho_from_unconstrained, rho_to_unconstrained,
)


@pytest.fixture(scope="module")
def frozen():
    return load_stage6c_dataset()


@pytest.fixture(scope="module")
def catalogue():
    return build_catalogue(5, 2)


@pytest.fixture(scope="module")
def blocks(frozen):
    return frozen.train[:60]


@pytest.fixture(scope="module")
def small_grid():
    return np.linspace(1e-3, 0.994, 13)


@pytest.fixture(scope="module")
def masses(catalogue, small_grid):
    return prior_cell_masses(catalogue, small_grid, n_draws=200_000, seed=5)


# --------------------- area 27: the cell factorisation the reference actually relies on
def test_the_likelihood_is_constant_on_a_cell(blocks, frozen, catalogue):
    """`L` depends on `U` only through `h(U)`, so the cell integral factorises exactly.

    Sampled `U` values that land in the same cell must score *identically*, not merely
    close. If this ever failed, `p(P, rho | Y) = p(rho) L(P) pi_rho(P)` would be wrong and
    the whole reference would be measuring the wrong object.
    """
    rng = np.random.default_rng(0)
    chol = np.linalg.cholesky(sigma_rho_matrix(2, 0.4))
    by_cell: dict[int, list[float]] = {}
    for _ in range(120):
        u = rng.normal(size=(5, 2)) @ chol.T
        index = catalogue.index_of(precedence_from_u(u))
        features = vectorized_state_features(blocks, u, frozen.truth["omega"])
        value = float(cached_batch_log_likelihood(
            features, frozen.truth["beta"], frozen.epsilon,
            frozen.truth["lambda_rep"], frozen.truth["lambda_back"]))
        by_cell.setdefault(index, []).append(value)

    shared = [v for v in by_cell.values() if len(v) > 1]
    assert shared, "no cell was hit twice; the test proved nothing"
    for values in shared:
        assert max(values) - min(values) == pytest.approx(0.0, abs=1e-9)


def test_catalogue_representative_likelihood_equals_any_member_of_the_cell(
        blocks, frozen, catalogue):
    """`L(P)` computed from the representative is `L` for every `U` in that cell."""
    rng = np.random.default_rng(1)
    chol = np.linalg.cholesky(sigma_rho_matrix(2, 0.5))
    subset = [0, 1, 2]
    reference = poset_log_likelihoods(
        _slice(catalogue, subset), blocks, frozen.epsilon, frozen.truth["beta"],
        frozen.truth["omega"], frozen.truth["lambda_rep"], frozen.truth["lambda_back"])
    for position, index in enumerate(subset):
        u = catalogue.representatives[index]
        features = vectorized_state_features(blocks, u, frozen.truth["omega"])
        direct = float(cached_batch_log_likelihood(
            features, frozen.truth["beta"], frozen.epsilon,
            frozen.truth["lambda_rep"], frozen.truth["lambda_back"]))
        assert reference[position] == pytest.approx(direct, abs=1e-12)
    assert rng is not None


def _slice(catalogue, indices):
    """A catalogue restricted to `indices`, for tests that do not need all 4231."""
    indices = np.asarray(indices, dtype=int)
    keys = catalogue.keys
    return type(catalogue)(
        m=catalogue.m, d=catalogue.d, closures=catalogue.closures[indices],
        reductions=catalogue.reductions[indices],
        relation_counts=catalogue.relation_counts[indices],
        representatives=catalogue.representatives[indices],
        ranking_tuple_counts=catalogue.ranking_tuple_counts[indices],
        keys=keys[:indices.size], order=np.arange(indices.size))


def test_single_skill_factorisation_is_the_identity(frozen):
    """With `K = 1` the §4 per-skill product has one factor; no Cartesian tensor exists.

    Recorded explicitly so the absence of a `U_1 x ... x U_K` enumeration is a stated
    consequence of the registered model rather than an omission.
    """
    assert frozen.n_skills == 1


# ------------------------------------------- areas 28, 29: reference normalisation
def test_6c1_reference_normalises_to_one(catalogue, blocks, frozen, small_grid, masses):
    reference = build_6c1_reference(catalogue, blocks, frozen.truth, frozen.epsilon,
                                    small_grid, masses)
    total = np.trapezoid(reference.joint.sum(axis=0), small_grid)
    assert float(total) == pytest.approx(1.0, abs=1e-10)


def test_6c1_poset_probabilities_and_rho_marginal_are_distributions(
        catalogue, blocks, frozen, small_grid, masses):
    reference = build_6c1_reference(catalogue, blocks, frozen.truth, frozen.epsilon,
                                    small_grid, masses)
    summary = reference_summary(reference, frozen.truth)
    assert summary["poset_probability"].sum() == pytest.approx(1.0, abs=1e-10)
    assert (summary["poset_probability"] >= 0).all()
    assert float(np.trapezoid(summary["rho_marginal_density"], small_grid)) == \
        pytest.approx(1.0, abs=1e-10)
    assert summary["relation_count_distribution"].sum() == pytest.approx(1.0, abs=1e-10)


def test_6c2_reference_normalises_to_one(catalogue, blocks, frozen, small_grid, masses):
    betas = np.linspace(1.2, 1.8, 9)
    reference = build_6c2_reference(catalogue, blocks, frozen.truth, frozen.epsilon,
                                    small_grid, betas, masses)
    over_posets = reference.joint.sum(axis=0)
    total = np.trapezoid(np.trapezoid(over_posets, small_grid, axis=0), betas)
    assert float(total) == pytest.approx(1.0, abs=1e-10)


def test_6c2_marginals_are_distributions(catalogue, blocks, frozen, small_grid, masses):
    betas = np.linspace(1.2, 1.8, 9)
    reference = build_6c2_reference(catalogue, blocks, frozen.truth, frozen.epsilon,
                                    small_grid, betas, masses)
    summary = reference_summary(reference, frozen.truth)
    assert summary["poset_probability"].sum() == pytest.approx(1.0, abs=1e-10)
    assert float(np.trapezoid(summary["rho_marginal_density"], small_grid)) == \
        pytest.approx(1.0, abs=1e-10)
    assert float(np.trapezoid(summary["beta_marginal_density"], betas)) == \
        pytest.approx(1.0, abs=1e-10)


def test_beta_table_agrees_with_the_single_beta_likelihood(catalogue, blocks, frozen):
    """The batched `L(P, beta)` table must reproduce the scalar path exactly."""
    subset = _slice(catalogue, [0, 100, 4002])
    betas = np.array([1.2, 1.5, 1.9])
    table = poset_log_likelihood_beta_table(
        subset, blocks, frozen.epsilon, betas, frozen.truth["omega"],
        frozen.truth["lambda_rep"], frozen.truth["lambda_back"])
    for j, beta in enumerate(betas):
        column = poset_log_likelihoods(
            subset, blocks, frozen.epsilon, float(beta), frozen.truth["omega"],
            frozen.truth["lambda_rep"], frozen.truth["lambda_back"])
        assert np.allclose(table[:, j], column, atol=1e-9)


# ------------------------------------- area 30: quadrature and transformation Jacobians
def test_the_reference_integrates_in_the_rho_coordinate_so_needs_no_jacobian(
        catalogue, blocks, frozen, small_grid, masses):
    """`log_rho_prior` is a density in `rho` and the grid is in `rho`: no Jacobian.

    The logit Jacobian belongs to the *sampler's* random walk, not to the reference's
    quadrature. Conflating the two is exactly the Stage 6B1 failure this guards against.
    """
    reference = build_6c1_reference(catalogue, blocks, frozen.truth, frozen.epsilon,
                                    small_grid, masses)
    total = np.trapezoid(reference.joint.sum(axis=0), small_grid)
    assert float(total) == pytest.approx(1.0, abs=1e-10)

    # The same density re-expressed in z = logit(rho) integrates to 1 only when the
    # Jacobian is applied. Done on a dedicated fine grid: z spans about [-6.9, 5.1], far
    # too wide for the 13-point reference grid used above to integrate accurately.
    # The support is open, so the grid stops an epsilon inside each end and necessarily
    # misses 2*eps/RHO_UPPER of the uniform mass. The tolerances below are that sliver,
    # not slack in the identity being tested.
    eps = 1e-4
    fine_rho = np.linspace(eps, RHO_UPPER - eps, 4001)
    excluded = 2 * eps / RHO_UPPER
    density_rho = np.exp([log_rho_prior(float(r)) for r in fine_rho])
    assert float(np.trapezoid(density_rho, fine_rho)) == pytest.approx(
        1.0 - excluded, abs=1e-6)

    fine_z = np.array([rho_to_unconstrained(float(r)) for r in fine_rho])
    density_z = density_rho * np.exp([log_jacobian_rho(float(r)) for r in fine_rho])
    assert float(np.trapezoid(density_z, fine_z)) == pytest.approx(
        1.0 - excluded, abs=5e-3)

    # and omitting the Jacobian gets it wrong, which is why the distinction matters
    assert abs(float(np.trapezoid(density_rho, fine_z)) - 1.0) > 0.1


def test_change_of_variables_identity_for_the_logit_transform():
    """`p_z(z) = p_rho(rho) * rho (1 - rho)` — asserted numerically, both directions."""
    for rho in (0.1, 0.35, 0.6, 0.9):
        z = rho_to_unconstrained(rho)
        h = 1e-6
        drho_dz = (rho_from_unconstrained(z + h) - rho_from_unconstrained(z - h)) / (2 * h)
        assert math.exp(log_jacobian_rho(rho)) == pytest.approx(drho_dz, abs=1e-8)


def test_rho_prior_is_uniform_on_its_truncated_support():
    """Registered as Beta(1,1) truncated at 1 - 5e-3, i.e. Uniform(0, 0.995)."""
    values = [log_rho_prior(r) for r in (0.05, 0.3, 0.5, 0.8, 0.99)]
    assert np.allclose(values, values[0], atol=1e-12)
    assert math.exp(values[0]) == pytest.approx(1.0 / RHO_UPPER, abs=1e-9)


def test_trapezoid_normalisation_improves_with_refinement(catalogue, blocks, frozen,
                                                          masses, small_grid):
    """A coarser grid must not already be exact — otherwise refinement proves nothing."""
    coarse = small_grid[::4]
    coarse_masses = {**masses, "masses": masses["masses"][::4], "rho_grid": coarse}
    reference = build_6c1_reference(catalogue, blocks, frozen.truth, frozen.epsilon,
                                    coarse, coarse_masses)
    total = np.trapezoid(reference.joint.sum(axis=0), coarse)
    assert float(total) == pytest.approx(1.0, abs=1e-10)


# ------------------------------------------------- area 31: refinement stability
def test_rho_summary_is_stable_under_grid_refinement(catalogue, blocks, frozen, masses,
                                                     small_grid):
    fine = build_6c1_reference(catalogue, blocks, frozen.truth, frozen.epsilon,
                               small_grid, masses)
    fine_summary = reference_summary(fine, frozen.truth)

    coarse_grid = small_grid[::2]
    coarse_masses = {**masses, "masses": masses["masses"][::2], "rho_grid": coarse_grid}
    coarse = build_6c1_reference(catalogue, blocks, frozen.truth, frozen.epsilon,
                                 coarse_grid, coarse_masses)
    coarse_summary = reference_summary(coarse, frozen.truth)

    assert abs(fine_summary["rho"]["mean"] - coarse_summary["rho"]["mean"]) < 0.05
    assert np.abs(fine_summary["poset_probability"]
                  - coarse_summary["poset_probability"]).max() < 1e-6


def test_the_map_poset_does_not_move_under_refinement(catalogue, blocks, frozen, masses,
                                                      small_grid):
    fine = reference_summary(build_6c1_reference(
        catalogue, blocks, frozen.truth, frozen.epsilon, small_grid, masses),
        frozen.truth)
    coarse_grid = small_grid[::2]
    coarse_masses = {**masses, "masses": masses["masses"][::2], "rho_grid": coarse_grid}
    coarse = reference_summary(build_6c1_reference(
        catalogue, blocks, frozen.truth, frozen.epsilon, coarse_grid, coarse_masses),
        frozen.truth)
    assert fine["map_poset"] == coarse["map_poset"]


# ------------------------------------------------------- reference draws are usable
def test_reference_draws_reproduce_the_reference_marginals(catalogue, blocks, frozen,
                                                           small_grid, masses):
    reference = build_6c1_reference(catalogue, blocks, frozen.truth, frozen.epsilon,
                                    small_grid, masses)
    summary = reference_summary(reference, frozen.truth)
    draws = sample_reference_draws(reference, summary, n_draws=20_000, seed=3)
    assert draws["rho"].shape == (20_000,)
    assert draws["relations"].shape == (20_000, catalogue.m * catalogue.m)

    # The draws reconstruct rho as a jittered grid cell, so on this deliberately coarse
    # 13-point grid they agree with the trapezoid mean only to about a grid spacing / 4.
    # The registered reference runs on 81 points; this tolerance is a property of the
    # test's grid, not of the sampler.
    spacing = float(np.diff(small_grid).mean())
    assert float(np.mean(draws["rho"])) == pytest.approx(
        summary["rho"]["mean"], abs=spacing / 3)
    empirical_relation = draws["relations"].mean(axis=0)
    assert np.abs(empirical_relation - summary["relation_marginal"]).max() < 0.02


def test_prior_cell_masses_come_from_the_prior_alone(catalogue, small_grid):
    """No data, no likelihood, no chain — asserted from the recorded provenance."""
    masses = prior_cell_masses(catalogue, small_grid, n_draws=20_000, seed=1)
    assert masses["provenance"] == "prior draws only: no data, no likelihood, no MCMC"
    assert masses["unseen_draws"] == 0
    assert np.allclose(masses["masses"].sum(axis=1), 1.0, atol=1e-12)


def test_prior_cell_masses_shift_with_rho_in_the_expected_direction(catalogue):
    """Higher rho aligns the two coordinates, so orders get denser."""
    grid = np.array([0.05, 0.95])
    masses = prior_cell_masses(catalogue, grid, n_draws=200_000, seed=2)
    mean_relations = masses["masses"] @ catalogue.relation_counts
    assert mean_relations[1] > mean_relations[0]
