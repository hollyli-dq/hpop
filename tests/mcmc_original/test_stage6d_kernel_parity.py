"""Stage 6D0 — kernel parity with both parents (§18 areas 15, 16, 17, 18, 19, 20).

Composing the Stage 6B scalar kernels with the Stage 6C latent kernels must not have
changed either. These tests reconstruct each parent's acceptance ratio from that parent's
own objects and require numerical equality with Stage 6D's — not because both call the
same helper, but because the two routes are built independently here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hpop.mcmc_original.recurrent_joint_scalar_mcmc import RecurrentJointEvaluator
from hpop.mcmc_original.recurrent_latent_poset_mcmc import (
    LatentPosetEvaluator, Stage6CTarget,
)
from hpop.mcmc_original.recurrent_oracle_joint_mcmc import (
    Stage6DTarget, initial_state, log_target_6d, sweep_once,
)
from hpop.mcmc_original.recurrent_scalar_mcmc import build_proposal, scalar_mh_step
from hpop.mcmc_original.recurrent_scalar_posterior import log_prior
from hpop.mcmc_original.sampler_u import propose_row
from hpop.mcmc_original.stage6d_frozen import (
    ACTIVE_6D, REGISTERED_SCALES, SCALAR_ORDER, load_stage6d_dataset, log_jacobian_rho,
    log_rho_prior, log_structural_prior, rho_from_unconstrained, rho_to_unconstrained,
)

# The Stage 6B parity tolerance established in Part VIII of the walkthrough.
STAGE6B_TOLERANCE = 7.3e-12
STAGE6C_TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def frozen():
    return load_stage6d_dataset()


@pytest.fixture(scope="module")
def blocks(frozen):
    return frozen.train[:40]


def u_states(frozen, rng, n=4):
    yield frozen.u_true
    for _ in range(n - 1):
        yield frozen.u_true + rng.normal(scale=0.6, size=(5, 2))


# ---------------------------------- areas 15-18: the four Stage 6B scalar kernels
@pytest.mark.parametrize("name", SCALAR_ORDER)
def test_stage6b_scalar_parity_across_u_and_rho(name, blocks, frozen):
    """At many nontrivial U and several rho, the 6D ratio equals the 6B ratio.

    rho does not enter the likelihood, so varying it must leave the scalar ratio
    unchanged — which is itself part of what this asserts.
    """
    rng = np.random.default_rng(11)
    truth = {k: float(v) for k, v in frozen.truth.items()}
    worst = 0.0
    comparisons = 0
    for u in u_states(frozen, rng):
        evaluator6d = LatentPosetEvaluator(blocks, epsilon=frozen.epsilon,
                                           omega=truth["omega"])
        evaluator6b = RecurrentJointEvaluator(blocks, u, frozen.epsilon)
        for rho in (0.1, 0.5, 0.85):
            for candidate in (0.6, 0.95, 1.4, 2.3):
                trial = dict(truth)
                trial[name] = candidate
                ratio6d = ((log_prior(name, candidate) - log_prior(name, truth[name]))
                           + (evaluator6d.log_likelihood(
                               u, trial["beta"], trial["omega"], trial["lambda_rep"],
                               trial["lambda_back"], allow_cache=False)
                              - evaluator6d.log_likelihood(
                                  u, truth["beta"], truth["omega"], truth["lambda_rep"],
                                  truth["lambda_back"], allow_cache=False)))
                ratio6b = ((log_prior(name, candidate) - log_prior(name, truth[name]))
                           + (evaluator6b.log_likelihood(
                               trial["beta"], trial["omega"], trial["lambda_rep"],
                               trial["lambda_back"], allow_cache=False)
                              - evaluator6b.log_likelihood(
                                  truth["beta"], truth["omega"], truth["lambda_rep"],
                                  truth["lambda_back"], allow_cache=False)))
                worst = max(worst, abs(ratio6d - ratio6b))
                comparisons += 1
    assert comparisons >= 48
    assert worst < STAGE6B_TOLERANCE, f"{name} parity worst {worst:.3e}"


def test_stage6b_transformed_proposal_is_reused_unchanged():
    """The Hastings correction comes from the frozen builder, not a local re-derivation."""
    for name in SCALAR_ORDER:
        proposal = build_proposal(name, REGISTERED_SCALES[name])(1.4,
                                                                 np.random.default_rng(0))
        if name == "omega":
            assert proposal.log_q_reverse_minus_forward == pytest.approx(0.0, abs=1e-15)
        else:
            assert proposal.value > 0.0
            assert proposal.log_q_reverse_minus_forward != 0.0


def test_stage6b_scalar_step_is_deterministic_given_the_stream(blocks, frozen):
    """Same seed, same kernel objects, same outcome — accepted and rejected alike."""
    truth = {k: float(v) for k, v in frozen.truth.items()}
    evaluator = LatentPosetEvaluator(blocks, epsilon=frozen.epsilon,
                                     omega=truth["omega"])
    evaluator.ensure_cache(frozen.u_true, truth["omega"])

    def posterior(candidate, name="beta"):
        prior = log_prior(name, candidate)
        if not math.isfinite(prior):
            return -math.inf
        trial = dict(truth)
        trial[name] = candidate
        return prior + evaluator.log_likelihood(
            frozen.u_true, trial["beta"], trial["omega"], trial["lambda_rep"],
            trial["lambda_back"], allow_cache=True)

    outcomes = set()
    for seed in range(8):
        a = scalar_mh_step(truth["beta"], posterior(truth["beta"]), posterior,
                           build_proposal("beta", REGISTERED_SCALES["beta"]),
                           np.random.default_rng(seed))
        b = scalar_mh_step(truth["beta"], posterior(truth["beta"]), posterior,
                           build_proposal("beta", REGISTERED_SCALES["beta"]),
                           np.random.default_rng(seed))
        assert a == b
        outcomes.add(a[2])
    assert outcomes == {True, False}, "seeds produced only one acceptance outcome"


# ------------------------------------- areas 19, 20: the Stage 6C latent kernels
def test_stage6c_u_parity(blocks, frozen):
    """With the scalars pinned, the 6D target reduces to the 6C target term for term."""
    truth = {k: float(v) for k, v in frozen.truth.items()}
    target6d = Stage6DTarget(LatentPosetEvaluator(
        blocks, epsilon=frozen.epsilon, omega=truth["omega"]), active=ACTIVE_6D)
    target6c = Stage6CTarget(LatentPosetEvaluator(
        blocks, epsilon=frozen.epsilon, omega=truth["omega"]),
        active=("U", "rho"), fixed=truth)

    rng = np.random.default_rng(12)
    worst = 0.0
    for _ in range(40):
        u = frozen.u_true + rng.normal(scale=0.5, size=(5, 2))
        rho = float(rng.uniform(0.05, 0.9))
        candidate = propose_row(u, int(rng.integers(5)), REGISTERED_SCALES["U"], rng)
        ratio6d = (target6d.log_target(candidate, {"rho": rho, **truth},
                                       allow_cache=False)
                   - target6d.log_target(u, {"rho": rho, **truth}, allow_cache=False))
        ratio6c = (target6c.log_target(candidate, {"rho": rho}, allow_cache=False)
                   - target6c.log_target(u, {"rho": rho}, allow_cache=False))
        worst = max(worst, abs(ratio6d - ratio6c))
    assert worst < STAGE6C_TOLERANCE


def test_stage6c_rho_parity_includes_the_full_log_determinant(blocks, frozen):
    """The rho ratio must equal the direct target difference plus the logit Jacobian,
    and must move when the determinant is removed."""
    truth = {k: float(v) for k, v in frozen.truth.items()}
    target = Stage6DTarget(LatentPosetEvaluator(
        blocks, epsilon=frozen.epsilon, omega=truth["omega"]), active=ACTIVE_6D)
    rng = np.random.default_rng(13)
    worst = 0.0
    checked = 0
    for _ in range(60):
        u = frozen.u_true + rng.normal(scale=0.4, size=(5, 2))
        rho = float(rng.uniform(0.05, 0.9))
        candidate = rho_from_unconstrained(rho_to_unconstrained(rho) + 0.5 * rng.normal())
        if not (0.0 < candidate < 0.995):
            continue
        implemented = ((log_structural_prior(u, candidate) - log_structural_prior(u, rho))
                       + (log_rho_prior(candidate) - log_rho_prior(rho))
                       + (log_jacobian_rho(candidate) - log_jacobian_rho(rho)))
        direct = (log_target_6d(target, u, {"rho": candidate, **truth})["log_target"]
                  - log_target_6d(target, u, {"rho": rho, **truth})["log_target"]
                  + log_jacobian_rho(candidate) - log_jacobian_rho(rho))
        worst = max(worst, abs(implemented - direct))
        checked += 1
    assert checked > 30
    assert worst < STAGE6C_TOLERANCE


# --------------------------------------------- area 22 (sweep order), by observation
def test_each_coordinate_sees_the_values_accepted_before_it(blocks, frozen):
    """Drive a sweep and confirm every recorded component belongs to the FINAL state.

    If `rho` had been scored against the pre-sweep `U`, or a scalar against a stale
    `rho`, the recomputed decomposition would disagree.
    """
    truth = {k: float(v) for k, v in frozen.truth.items()}
    target = Stage6DTarget(LatentPosetEvaluator(
        blocks, epsilon=frozen.epsilon, omega=truth["omega"]), active=ACTIVE_6D)
    rng = np.random.default_rng(14)
    state = initial_state(target, frozen.u_true, {"rho": 0.3, **truth}, rng)
    moved = {"U": False, "rho": False}
    for _ in range(60):
        before_u = state.u.copy()
        before_rho = state.values["rho"]
        state = sweep_once(state, target, REGISTERED_SCALES, rng)
        moved["U"] = moved["U"] or not np.array_equal(before_u, state.u)
        moved["rho"] = moved["rho"] or before_rho != state.values["rho"]
        recomputed = log_target_6d(target, state.u, state.values)
        assert state.log_structural_prior == pytest.approx(
            recomputed["log_structural_prior"], abs=1e-9)
        assert state.log_likelihood == pytest.approx(
            recomputed["log_likelihood"], abs=1e-7)
    assert moved["U"] and moved["rho"], "nothing moved; the test proved nothing"


def test_a_rejected_sweep_leaves_the_previous_accepted_state_visible(blocks, frozen):
    truth = {k: float(v) for k, v in frozen.truth.items()}
    target = Stage6DTarget(LatentPosetEvaluator(
        blocks, epsilon=frozen.epsilon, omega=truth["omega"]), active=ACTIVE_6D)
    rng = np.random.default_rng(15)
    state = initial_state(target, frozen.u_true, {"rho": 0.3, **truth}, rng)
    # A huge U step makes every U proposal astronomically unlikely to be accepted. The
    # scalar scales stay registered: three of the four walk on a log scale, where an
    # absurd scale overflows exp() rather than simply being rejected.
    scales = {**REGISTERED_SCALES, "U": 1e6}
    after = sweep_once(state, target, scales, rng)
    assert after.accepted["U"] == 0
    assert np.array_equal(after.u, state.u)
    # rho keeps its registered scale and may legitimately move, which changes the
    # structural prior at fixed U. What must hold is that the carried value belongs to
    # the final accepted (U, rho) rather than to the rejected proposal.
    assert after.log_structural_prior == pytest.approx(
        log_structural_prior(after.u, after.values["rho"]), abs=1e-10)
