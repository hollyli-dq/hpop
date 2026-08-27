"""Stage 6B — the independent joint references.

The references are what the sampler-correctness claim rests on, so they get their own
validation: the collapsed evaluator must be exact, the transformed-coordinate density must
carry the right Jacobian, the grids must normalise, and the summaries must be resolved
rather than merely computed.

The Jacobian test is the sharpest one available here. With a single active coordinate the
joint machinery — which works in `log theta` and adds `+ log theta` for the change of
variables — must reproduce the Stage 6B0 one-dimensional reference, which works directly
in `theta` and adds nothing. Agreement between those two routes can only happen if the
Jacobian is right.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hpop.mcmc_original.recurrent_joint_scalar_mcmc import (
    RecurrentJointEvaluator, vectorized_state_features,
)
from hpop.mcmc_original.recurrent_scalar_posterior import reference_posterior
from hpop.mcmc_original.stage6b_frozen import load_frozen_dataset
from hpop.mcmc_original.stage6b_joint_reference import (
    build_reference, collapse_features, collapsed_log_likelihood_beta_grid,
    curvature_scales, find_map, reference_summary, sample_reference_draws,
    transformed_log_posterior,
)

ACTIVE_3D = ("beta", "omega", "lambda_rep")
ACTIVE_4D = ("beta", "omega", "lambda_rep", "lambda_back")


@pytest.fixture(scope="module")
def frozen():
    return load_frozen_dataset()


@pytest.fixture(scope="module")
def small(frozen):
    """A 40-block slice: the same model, small enough to grid repeatedly in a test."""
    return frozen.train[:40]


@pytest.fixture(scope="module")
def evaluator(small, frozen):
    return RecurrentJointEvaluator(small, frozen.u_true, frozen.epsilon)


# ------------------------------------------------------------------- collapsed evaluator
def test_collapsing_identical_steps_is_exact(small, frozen, evaluator):
    features = vectorized_state_features(small, frozen.u_true, frozen.truth["omega"])
    collapsed = collapse_features(features)
    assert collapsed["n_steps"] == small.size
    assert collapsed["n_unique"] < collapsed["n_steps"], "the corpus should collapse at all"

    rng = np.random.default_rng(0)
    for _ in range(10):
        beta = float(rng.uniform(0.5, 3.0))
        lrep = float(rng.uniform(0.05, 2.0))
        lback = float(rng.uniform(0.01, 1.0))
        got = collapsed_log_likelihood_beta_grid(
            collapsed, [beta], frozen.epsilon, lrep, lback)[0]
        exact = evaluator.full_replay_log_likelihood(
            beta, frozen.truth["omega"], lrep, lback)
        assert got == pytest.approx(exact, abs=1e-8)


def test_the_beta_batch_matches_one_beta_at_a_time(small, frozen):
    collapsed = collapse_features(
        vectorized_state_features(small, frozen.u_true, frozen.truth["omega"]))
    betas = np.linspace(0.8, 2.4, 9)
    batched = collapsed_log_likelihood_beta_grid(collapsed, betas, frozen.epsilon, 0.8, 0.25)
    one_by_one = [collapsed_log_likelihood_beta_grid(
        collapsed, [b], frozen.epsilon, 0.8, 0.25)[0] for b in betas]
    assert np.abs(batched - np.array(one_by_one)).max() < 1e-10


def test_the_collapse_is_rebuilt_for_every_omega(small, frozen):
    """The grouping values depend on kappa, so two omegas must not share a collapse."""
    a = collapse_features(vectorized_state_features(small, frozen.u_true, 0.5))
    b = collapse_features(vectorized_state_features(small, frozen.u_true, 2.5))
    assert a["omega"] != b["omega"]
    assert not np.array_equal(a["q"], b["q"]) or not np.array_equal(a["C_back"], b["C_back"])


# --------------------------------------------------------------------- Jacobian (item 19)
def test_the_transformed_reference_reproduces_the_stage_6b0_one_dimensional_grid(
        small, frozen, evaluator):
    """One active coordinate: transformed-with-Jacobian must equal direct-in-theta."""
    fixed = {n: frozen.truth[n] for n in ("omega", "lambda_rep", "lambda_back")}
    grid = build_reference(("beta",), evaluator, fixed, frozen.truth, frozen.epsilon,
                           n_points=401, radius=8.0)
    summary = reference_summary(grid)

    direct = reference_posterior(
        "beta", [tuple(r) for r in small], frozen.u_true, frozen.truth, frozen.epsilon,
        initial_range=(0.2, 5.0), n_points=2001)

    assert summary["mean"]["beta"] == pytest.approx(direct["mean"], rel=2e-3)
    assert summary["sd"]["beta"] == pytest.approx(direct["sd"], rel=5e-3)
    assert summary["median"]["beta"] == pytest.approx(direct["median"], rel=2e-3)
    assert summary["q025"]["beta"] == pytest.approx(direct["q025"], rel=5e-3)
    assert summary["q975"]["beta"] == pytest.approx(direct["q975"], rel=5e-3)


def test_dropping_the_jacobian_would_shift_the_answer(small, frozen, evaluator):
    """The agreement above is not automatic: without `+ log theta` it breaks."""
    fixed = {n: frozen.truth[n] for n in ("omega", "lambda_rep", "lambda_back")}
    z = np.linspace(math.log(0.6), math.log(3.0), 241)

    with_jacobian = np.array([transformed_log_posterior([zi], ("beta",), evaluator, fixed)
                              for zi in z])
    without = with_jacobian - z            # remove the log-theta Jacobian term

    def mean_on(log_density):
        d = np.exp(log_density - log_density.max())
        d = d / np.trapezoid(d, z)
        return float(np.trapezoid(np.exp(z) * d, z))

    correct = mean_on(with_jacobian)
    broken = mean_on(without)
    assert abs(correct - broken) > 1e-4, "the Jacobian must actually matter here"


# ------------------------------------------------------------- normalisation (items 17/18)
def test_the_three_dimensional_reference_normalises(evaluator, frozen):
    fixed = {"lambda_back": frozen.truth["lambda_back"]}
    grid = build_reference(ACTIVE_3D, evaluator, fixed, frozen.truth, frozen.epsilon,
                           n_points=25, radius=6.0)
    summary = reference_summary(grid)
    assert summary["integral_check"] == pytest.approx(1.0, rel=1e-8)
    assert summary["grid_points"] == 25 ** 3
    assert summary["outer_face_mass"] < 1e-4
    assert np.all(np.isfinite(np.array(summary["correlation"])))
    assert np.allclose(np.diag(np.array(summary["correlation"])), 1.0)


def test_the_four_dimensional_reference_normalises(evaluator, frozen):
    grid = build_reference(ACTIVE_4D, evaluator, {}, frozen.truth, frozen.epsilon,
                           n_points=17, radius=6.0)
    summary = reference_summary(grid)
    assert summary["integral_check"] == pytest.approx(1.0, rel=1e-8)
    assert summary["grid_points"] == 17 ** 4
    assert summary["outer_face_mass"] < 1e-3
    correlation = np.array(summary["correlation"])
    assert correlation.shape == (4, 4)
    assert np.allclose(correlation, correlation.T)


def test_reference_summaries_are_ordered_and_bracket_the_median(evaluator, frozen):
    grid = build_reference(ACTIVE_3D, evaluator,
                           {"lambda_back": frozen.truth["lambda_back"]},
                           frozen.truth, frozen.epsilon, n_points=25)
    summary = reference_summary(grid)
    for name in ACTIVE_3D:
        assert summary["q025"][name] < summary["median"][name] < summary["q975"][name]
        assert summary["sd"][name] > 0
        assert summary["q025"][name] <= summary["mean"][name] <= summary["q975"][name]


# ---------------------------------------------------------------- refinement (item 20)
def test_refining_the_grid_does_not_move_the_summaries(evaluator, frozen):
    fixed = {"lambda_back": frozen.truth["lambda_back"]}
    # max_expansions=0 keeps both grids on the same domain, so this measures resolution
    # rather than the difference between two differently-expanded boxes
    coarse = reference_summary(build_reference(
        ACTIVE_3D, evaluator, fixed, frozen.truth, frozen.epsilon, n_points=31,
        radius=6.0, max_expansions=0))
    fine = reference_summary(build_reference(
        ACTIVE_3D, evaluator, fixed, frozen.truth, frozen.epsilon, n_points=45,
        radius=6.0, max_expansions=0))
    for name in ACTIVE_3D:
        sd = fine["sd"][name]
        assert abs(fine["mean"][name] - coarse["mean"][name]) / sd < 0.05
        assert abs(fine["median"][name] - coarse["median"][name]) / sd < 0.05
        assert abs(fine["q025"][name] - coarse["q025"][name]) / sd < 0.15
        assert abs(fine["q975"][name] - coarse["q975"][name]) / sd < 0.15
    assert np.abs(np.array(fine["correlation"])
                  - np.array(coarse["correlation"])).max() < 0.05


def test_the_map_is_found_deterministically(evaluator, frozen):
    fixed = {"lambda_back": frozen.truth["lambda_back"]}
    start = {n: frozen.truth[n] for n in ACTIVE_3D}
    first = find_map(ACTIVE_3D, evaluator, fixed, start)
    second = find_map(ACTIVE_3D, evaluator, fixed, start)
    assert np.array_equal(first["z"], second["z"])
    assert first["log_posterior"] == second["log_posterior"]


def test_curvature_is_a_valid_covariance(evaluator, frozen):
    fixed = {"lambda_back": frozen.truth["lambda_back"]}
    mode = find_map(ACTIVE_3D, evaluator, fixed, {n: frozen.truth[n] for n in ACTIVE_3D})
    curvature = curvature_scales(ACTIVE_3D, evaluator, fixed, mode["z"])
    covariance = curvature["covariance"]
    assert np.allclose(covariance, covariance.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(covariance) > 0), "must be positive definite"
    assert np.all(curvature["sd"] > 0)


# ------------------------------------------------------------------- reference draws
def test_reference_draws_reproduce_the_reference_summaries(evaluator, frozen):
    grid = build_reference(ACTIVE_3D, evaluator,
                           {"lambda_back": frozen.truth["lambda_back"]},
                           frozen.truth, frozen.epsilon, n_points=61, radius=6.0,
                           max_expansions=0)
    summary = reference_summary(grid)
    draws = sample_reference_draws(grid, 40_000, seed=3)
    assert draws.shape == (40_000, 3)
    for i, name in enumerate(ACTIVE_3D):
        assert draws[:, i].mean() == pytest.approx(summary["mean"][name],
                                                   abs=0.05 * summary["sd"][name])
        # cells are sampled with their mass and jittered uniformly inside, which inflates
        # the spread by O(h^2); at the registered resolution that is ~1%, and it biases
        # the two-sample test towards declaring a difference, never towards hiding one
        ratio = draws[:, i].std(ddof=1) / summary["sd"][name]
        assert 1.0 <= ratio < 1.03, f"{name} draw spread ratio {ratio:.4f}"
    assert np.all(draws[:, 0] > 0) and np.all(draws[:, 2] > 0), "positive support respected"


def test_reference_draws_are_reproducible_and_seed_dependent(evaluator, frozen):
    grid = build_reference(ACTIVE_3D, evaluator,
                           {"lambda_back": frozen.truth["lambda_back"]},
                           frozen.truth, frozen.epsilon, n_points=21)
    a = sample_reference_draws(grid, 500, seed=1)
    b = sample_reference_draws(grid, 500, seed=1)
    c = sample_reference_draws(grid, 500, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_the_reference_never_touches_mcmc_code():
    """Provenance, enforced by import inspection rather than by assertion in prose."""
    import ast
    import hpop.mcmc_original.stage6b_joint_reference as module

    tree = ast.parse(open(module.__file__).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}

    forbidden = {"scalar_mh_step", "run_joint_scalar_mcmc", "run_scalar_mh", "sweep_once",
                 "build_proposal", "tune_proposal_scale"}
    assert not (called & forbidden), f"reference calls MCMC code: {called & forbidden}"
    for name in imported:
        assert name.rsplit(".", 1)[-1] not in forbidden, f"reference imports {name}"
    # the only sampler-adjacent import allowed is the pure state-feature builder
    assert any("vectorized_state_features" in n for n in imported)
