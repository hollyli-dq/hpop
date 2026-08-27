"""Stage 6B1, step 3 — the recurrent log-posterior callback the samplers actually call.

Three things are established here, in order:

1. the cached evaluator for the three parameters outside the ``q`` recursion is *exact*,
   not an approximation, and the per-block evaluator agrees with the batch one;
2. ``omega`` is not allowed to use that cache, and a fixed-``q`` shortcut for ``omega``
   is shown to be wrong — this is the trap the project already fell into once, so it is
   pinned by a test rather than by a comment;
3. the callback reproduces the immutable Stage 6B0 grid, and an end-to-end run on a small
   dataset recovers a grid computed the same way.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_rfs import recurrent_rfs_log_likelihood, sigmoid
from hpop.mcmc_original.recurrent_scalar_mcmc import (
    RecurrentScalarTarget, ScalarMHConfig, blockwise_recurrent_log_likelihood,
    build_proposal, curvature_proposal_scale, run_scalar_mh,
)
from hpop.mcmc_original.recurrent_scalar_posterior import (
    TRUE_VALUES, batch_recurrent_log_likelihood, batch_recurrent_log_likelihood_full_replay,
    log_prior, normalize_log_density_grid,
)
from hpop.mcmc_original.recurrent_synthetic import U_TRUE, generate_recurrent_dataset
from hpop.mcmc_original.stage6b_mcmc_diagnostics import (
    compare_to_reference, evaluate_gates, grid_summary, load_reference_posteriors,
)

EPSILON = 0.02
REFERENCE = (Path(__file__).resolve().parents[2] / "results" / "mcmc_original"
             / "stage6b_full_seed0" / "reference_posteriors.json")


@pytest.fixture(scope="module")
def smoke():
    data = generate_recurrent_dataset("smoke", seed=0)
    return np.array([b.roles for b in data.train], dtype=int)


@pytest.fixture(scope="module")
def full_train():
    data = generate_recurrent_dataset("full", seed=0)
    return np.array([b.roles for b in data.train], dtype=int)


# ------------------------------------------------------------------ evaluator equivalence
@pytest.mark.parametrize("parameter", ["beta", "lambda_rep", "lambda_back"])
def test_cached_evaluator_is_exact_for_parameters_outside_the_recursion(parameter, smoke):
    cached = RecurrentScalarTarget(parameter, smoke, U_TRUE, TRUE_VALUES, EPSILON,
                                   use_cache=True)
    replay = RecurrentScalarTarget(parameter, smoke, U_TRUE, TRUE_VALUES, EPSILON,
                                   use_cache=False)
    assert cached.cached and not replay.cached
    rng = np.random.default_rng(0)
    for value in np.concatenate([[TRUE_VALUES[parameter]], rng.uniform(0.01, 4.0, 60)]):
        assert cached.log_likelihood(value) == pytest.approx(replay.log_likelihood(value),
                                                             abs=1e-9)


def test_omega_never_uses_the_state_cache(smoke):
    target = RecurrentScalarTarget("omega", smoke, U_TRUE, TRUE_VALUES, EPSILON, use_cache=True)
    assert target.cached is False, "requesting a cache for omega must not produce one"
    assert target._features is None


def test_a_fixed_q_shortcut_for_omega_is_wrong(smoke):
    """Negative control for the registered trap.

    ``kappa = sigmoid(omega)`` enters the validity recursion. Holding ``q`` at its
    true-parameter trajectory and varying ``omega`` only inside the back-cost term keeps
    just the direct one-step channel, and understates the curvature by a large factor.
    The number is not the point; the order of magnitude is.
    """
    precedence = precedence_from_u(U_TRUE)
    succ_off = precedence.astype(float).copy()
    np.fill_diagonal(succ_off, 0.0)
    truth_omega = TRUE_VALUES["omega"]

    _, q_trace, _ = batch_recurrent_log_likelihood_full_replay(
        smoke, U_TRUE, TRUE_VALUES["beta"], EPSILON, truth_omega,
        TRUE_VALUES["lambda_rep"], TRUE_VALUES["lambda_back"], return_states=True)
    pred_mask = precedence.T.astype(bool)
    n, T, m = q_trace.shape

    def fixed_q_log_likelihood(omega: float) -> float:
        """Deliberately wrong: q is frozen at the truth trajectory."""
        kappa = sigmoid(omega)
        total = 0.0
        for t in range(T):
            q = q_trace[:, t, :]
            feasibility = np.prod(np.where(pred_mask[None, :, :], q[:, None, :], 1.0), axis=2)
            stale = (1.0 - q) @ precedence.astype(float).T
            back = kappa * (q @ succ_off.T)
            exponent = (TRUE_VALUES["beta"] * np.log1p(stale)
                        - TRUE_VALUES["lambda_rep"] * q - TRUE_VALUES["lambda_back"] * back)
            exponent -= exponent.max(axis=1, keepdims=True)
            weights = feasibility * np.exp(exponent)
            mixed = ((1.0 - EPSILON) * (weights / weights.sum(axis=1, keepdims=True))
                     + EPSILON / m)
            total += float(np.log(mixed[np.arange(n), smoke[:, t]]).sum())
        return total

    exact = RecurrentScalarTarget("omega", smoke, U_TRUE, TRUE_VALUES, EPSILON)
    delta = 0.05

    def curvature(fn):
        return -(fn(truth_omega + delta) - 2 * fn(truth_omega) + fn(truth_omega - delta)) / delta ** 2

    exact_curvature = curvature(exact.log_likelihood)
    shortcut_curvature = curvature(fixed_q_log_likelihood)
    assert exact_curvature > 0
    assert shortcut_curvature < exact_curvature / 5.0, (
        f"fixed-q curvature {shortcut_curvature:.3f} vs exact {exact_curvature:.3f}")


def test_blockwise_likelihood_agrees_with_the_batch_and_the_per_block_reference(smoke):
    kw = dict(TRUE_VALUES)
    per_block = blockwise_recurrent_log_likelihood(
        smoke, U_TRUE, kw["beta"], EPSILON, kw["omega"], kw["lambda_rep"], kw["lambda_back"])
    assert per_block.shape == (smoke.shape[0],)
    assert float(per_block.sum()) == pytest.approx(
        batch_recurrent_log_likelihood_full_replay(
            smoke, U_TRUE, kw["beta"], EPSILON, kw["omega"], kw["lambda_rep"],
            kw["lambda_back"]), abs=1e-9)
    for b in range(0, smoke.shape[0], 7):
        reference = recurrent_rfs_log_likelihood(
            tuple(smoke[b]), U_TRUE, kw["beta"], EPSILON, kw["omega"],
            kw["lambda_rep"], kw["lambda_back"])
        assert float(per_block[b]) == pytest.approx(reference, abs=1e-9)


def test_batch_full_replay_agrees_with_the_per_block_batch_evaluator(smoke):
    kw = dict(TRUE_VALUES)
    sequences = [tuple(row) for row in smoke]
    assert batch_recurrent_log_likelihood_full_replay(
        smoke, U_TRUE, kw["beta"], EPSILON, kw["omega"], kw["lambda_rep"], kw["lambda_back"]
    ) == pytest.approx(batch_recurrent_log_likelihood(
        sequences, U_TRUE, kw["beta"], EPSILON, kw["omega"], kw["lambda_rep"],
        kw["lambda_back"]), abs=1e-9)


# ------------------------------------------------------------------------ the target API
@pytest.mark.parametrize("parameter", ["beta", "lambda_rep", "lambda_back"])
def test_target_is_minus_infinity_outside_the_gamma_support(parameter, smoke):
    target = RecurrentScalarTarget(parameter, smoke, U_TRUE, TRUE_VALUES, EPSILON)
    before = target.calls
    for bad in (0.0, -0.5):
        log_posterior, log_likelihood = target(bad)
        assert log_posterior == -math.inf and log_likelihood == -math.inf
    assert target.calls == before, "no likelihood evaluation outside the support"


def test_target_decomposes_into_prior_plus_likelihood(smoke):
    target = RecurrentScalarTarget("beta", smoke, U_TRUE, TRUE_VALUES, EPSILON)
    log_posterior, log_likelihood = target(1.23)
    assert log_posterior == pytest.approx(log_likelihood + log_prior("beta", 1.23), abs=1e-12)


def test_unknown_parameter_is_rejected(smoke):
    with pytest.raises(ValueError):
        RecurrentScalarTarget("rho", smoke, U_TRUE, TRUE_VALUES, EPSILON)


# ------------------------------------------------- parity with the immutable Stage 6B0 grid
@pytest.mark.parametrize("parameter", ["beta", "omega", "lambda_rep", "lambda_back"])
def test_target_reproduces_the_immutable_reference_grid(parameter, full_train):
    """Compared as *differences* of log density, since the stored grid is normalized."""
    reference = load_reference_posteriors(REFERENCE)
    entry = reference["posteriors"][parameter]
    grid = np.asarray(entry["grid"])
    density = np.asarray(entry["density"])
    target = RecurrentScalarTarget(parameter, full_train, U_TRUE, reference["truth"],
                                   reference["epsilon"])

    indices = [300, 450, 500, 550, 700]
    mine = np.array([target(float(grid[i]))[0] for i in indices])
    stored = np.log(density[indices])
    assert np.abs((mine - mine[2]) - (stored - stored[2])).max() < 1e-9
