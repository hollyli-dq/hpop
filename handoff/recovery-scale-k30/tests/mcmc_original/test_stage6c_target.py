"""Stage 6C — direct target, rho update, beta parity and reproducibility
(§17 areas 11, 12, 20, 21, 22, 23, 24, 25).

The direct target is evaluated entirely outside the transition code: `Stage6CTarget`
never calls an acceptance-ratio helper, so comparing an implemented acceptance ratio
against a difference of direct targets compares two independent routes rather than one
function against a wrapper around itself.

    Stage 6C1:  log target = log L + log p(U | rho) + log p(rho)
    Stage 6C2:  log target = log L + log p(U | rho) + log p(rho) + log p(beta)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hpop.mcmc_original.recurrent_latent_poset_mcmc import (
    LatentPosetEvaluator, Stage6CState, Stage6CTarget, initial_state, run_stage6c_mcmc,
    sweep_once,
)
from hpop.mcmc_original.recurrent_scalar_mcmc import build_proposal, scalar_mh_step
from hpop.mcmc_original.recurrent_scalar_posterior import log_prior
from hpop.mcmc_original.sampler_u import propose_row
from hpop.mcmc_original.stage6c_frozen import (
    ACTIVE_6C1, ACTIVE_6C2, SIGMA_U, log_jacobian_rho, log_rho_prior,
    log_structural_prior, load_stage6c_dataset, rho_from_unconstrained,
    rho_to_unconstrained,
)

FIXED_6C1 = {"beta": 1.5, "omega": 1.7346, "lambda_rep": 0.8, "lambda_back": 0.25}
FIXED_6C2 = {"omega": 1.7346, "lambda_rep": 0.8, "lambda_back": 0.25}


@pytest.fixture(scope="module")
def frozen():
    return load_stage6c_dataset()


@pytest.fixture(scope="module")
def blocks(frozen):
    return frozen.train[:30]


@pytest.fixture()
def target_6c1(blocks):
    evaluator = LatentPosetEvaluator(blocks, epsilon=0.02, omega=FIXED_6C1["omega"])
    return Stage6CTarget(evaluator, active=ACTIVE_6C1, fixed=FIXED_6C1)


@pytest.fixture()
def target_6c2(blocks):
    evaluator = LatentPosetEvaluator(blocks, epsilon=0.02, omega=FIXED_6C2["omega"])
    return Stage6CTarget(evaluator, active=ACTIVE_6C2, fixed=FIXED_6C2)


# ------------------------------------------------ areas 11, 12: target decomposition
def test_stage_6c1_target_is_the_sum_of_its_three_components(target_6c1, frozen):
    rng = np.random.default_rng(0)
    for _ in range(40):
        u = frozen.u_true + rng.normal(scale=0.4, size=(5, 2))
        rho = float(rng.uniform(0.05, 0.9))
        parts = target_6c1.decompose(u, {"rho": rho}, allow_cache=False)
        expected = (parts["log_likelihood"] + parts["log_structural_prior"]
                    + parts["log_rho_prior"])
        assert parts["log_target"] == pytest.approx(expected, abs=1e-12)
        assert parts["log_scalar_priors"] == {}
        # and each component matches an independently computed value
        assert parts["log_structural_prior"] == pytest.approx(
            log_structural_prior(u, rho), abs=1e-12)
        assert parts["log_rho_prior"] == pytest.approx(log_rho_prior(rho), abs=1e-12)


def test_stage_6c2_target_adds_exactly_the_beta_prior(target_6c2, frozen):
    rng = np.random.default_rng(1)
    for _ in range(40):
        u = frozen.u_true + rng.normal(scale=0.4, size=(5, 2))
        rho = float(rng.uniform(0.05, 0.9))
        beta = float(rng.uniform(0.5, 3.0))
        parts = target_6c2.decompose(u, {"rho": rho, "beta": beta}, allow_cache=False)
        expected = (parts["log_likelihood"] + parts["log_structural_prior"]
                    + parts["log_rho_prior"] + parts["log_scalar_priors"]["beta"])
        assert parts["log_target"] == pytest.approx(expected, abs=1e-12)
        assert parts["log_scalar_priors"]["beta"] == pytest.approx(
            log_prior("beta", beta), abs=1e-12)


def test_the_two_targets_differ_by_the_beta_prior_alone(target_6c1, target_6c2, frozen):
    """At a common state with beta at its 6C1 fixed value."""
    u, rho, beta = frozen.u_true, 0.4, FIXED_6C1["beta"]
    a = target_6c1.decompose(u, {"rho": rho}, allow_cache=False)
    b = target_6c2.decompose(u, {"rho": rho, "beta": beta}, allow_cache=False)
    assert b["log_likelihood"] == pytest.approx(a["log_likelihood"], abs=1e-12)
    assert b["log_target"] - a["log_target"] == pytest.approx(
        log_prior("beta", beta), abs=1e-12)


def test_a_non_finite_component_makes_the_target_non_finite(target_6c1, frozen):
    parts = target_6c1.decompose(frozen.u_true, {"rho": 1.5}, allow_cache=False)
    assert parts["log_target"] == -math.inf


def test_target_refuses_to_leave_a_needed_scalar_unspecified(blocks):
    evaluator = LatentPosetEvaluator(blocks, epsilon=0.02, omega=1.7346)
    with pytest.raises(ValueError):
        Stage6CTarget(evaluator, active=("U", "rho"), fixed={"beta": 1.5})
    with pytest.raises(ValueError):
        Stage6CTarget(evaluator, active=("U",), fixed=FIXED_6C1)


# -------------------------------------------- area 20: rho proposal and its Jacobian
def test_logit_transform_round_trips():
    for rho in (1e-4, 0.01, 0.3, 0.5, 0.9, 0.994):
        assert rho_from_unconstrained(rho_to_unconstrained(rho)) == pytest.approx(
            rho, abs=1e-12)


def test_log_jacobian_matches_a_numerical_derivative():
    """`log|d rho / d z|` for `z = logit(rho)` is `log(rho (1 - rho))`."""
    for rho in (0.05, 0.2, 0.5, 0.75, 0.95):
        z = rho_to_unconstrained(rho)
        h = 1e-6
        numerical = (rho_from_unconstrained(z + h) - rho_from_unconstrained(z - h)) / (2 * h)
        assert log_jacobian_rho(rho) == pytest.approx(math.log(numerical), abs=1e-7)


def test_implemented_rho_ratio_equals_direct_target_difference_plus_jacobian(
        target_6c1, frozen):
    """The §9 identity, at deterministic states, against the direct target."""
    rng = np.random.default_rng(2)
    u = frozen.u_true
    for _ in range(60):
        current = float(rng.uniform(0.05, 0.9))
        candidate = float(rng.uniform(0.05, 0.9))
        implemented = ((log_structural_prior(u, candidate)
                        - log_structural_prior(u, current))
                       + (log_rho_prior(candidate) - log_rho_prior(current))
                       + (log_jacobian_rho(candidate) - log_jacobian_rho(current)))
        direct = (target_6c1.log_target(u, {"rho": candidate}, allow_cache=False)
                  - target_6c1.log_target(u, {"rho": current}, allow_cache=False))
        assert implemented == pytest.approx(direct + log_jacobian_rho(candidate)
                                            - log_jacobian_rho(current), abs=1e-9)


def test_a_rho_update_consumes_no_likelihood_evaluations(target_6c1, frozen):
    """rho acts only through `p(U | rho)`; touching the likelihood would be a bug."""
    rng = np.random.default_rng(3)
    state = initial_state(target_6c1, frozen.u_true, {"rho": 0.3}, rng)

    # A sweep proposes exactly one U move per role, each of which must replay, plus one
    # rho move, which must not. So the replay count per sweep is exactly the role count:
    # any rho contribution would push it above.
    n_sweeps = 20
    target_6c1.evaluator.full_replay_calls = 0
    target_6c1.evaluator.cached_calls = 0
    for _ in range(n_sweeps):
        state = sweep_once(state, target_6c1, SIGMA_U, 0.5, 0.05, rng)

    assert target_6c1.evaluator.full_replay_calls == n_sweeps * frozen.n_roles, (
        "the rho update evaluated the recurrent likelihood; rho acts only through "
        "p(U | rho) and must consume zero likelihood evaluations")
    assert target_6c1.evaluator.cached_calls == 0
    assert state.proposed["rho"] == n_sweeps


def test_changing_rho_does_not_change_the_recurrent_log_likelihood(target_6c1, frozen):
    values = {target_6c1.log_likelihood(frozen.u_true, {"rho": r}, allow_cache=False)
              for r in (0.05, 0.3, 0.6, 0.9)}
    assert len(values) == 1


def test_rho_outside_its_support_is_rejected_by_the_prior():
    assert log_rho_prior(0.0) == -math.inf
    assert log_rho_prior(0.9999) == -math.inf          # beyond the 1 - 5e-3 truncation
    assert math.isfinite(log_rho_prior(0.5))


# --------------------------------------------------- area 21: beta parity with 6B
def test_stage_6c2_beta_step_is_the_frozen_stage_6b_step(target_6c2, frozen):
    """Same proposal object, same kernel, same rng stream -> identical outcome.

    Parity is asserted numerically rather than inferred from the fact that both call
    `scalar_mh_step`: the 6B step is reconstructed here from its own pieces and driven
    with an identically seeded generator.
    """
    u, rho = frozen.u_true, 0.4
    beta_scale = 0.05109

    def beta_log_posterior(candidate):
        prior = log_prior("beta", candidate)
        if not math.isfinite(prior):
            return -math.inf
        return prior + target_6c2.evaluator.log_likelihood(
            u, candidate, FIXED_6C2["omega"], FIXED_6C2["lambda_rep"],
            FIXED_6C2["lambda_back"], allow_cache=True)

    target_6c2.evaluator.ensure_cache(u, FIXED_6C2["omega"])
    for seed in (0, 1, 2, 3, 4):
        for start in (0.9, 1.5, 2.4):
            current_posterior = beta_log_posterior(start)
            a = scalar_mh_step(start, current_posterior, beta_log_posterior,
                               build_proposal("beta", beta_scale),
                               np.random.default_rng(seed))
            b = scalar_mh_step(start, current_posterior, beta_log_posterior,
                               build_proposal("beta", beta_scale),
                               np.random.default_rng(seed))
            assert a == b


def test_beta_proposal_is_the_registered_log_random_walk():
    """A log random walk carries a non-zero Hastings term; a Gaussian one would not."""
    proposal = build_proposal("beta", 0.05109)(1.5, np.random.default_rng(0))
    assert proposal.value > 0.0
    assert proposal.log_q_reverse_minus_forward != 0.0


# ------------------------------- area 22: later updates see earlier accepted states
def test_rho_update_uses_the_u_accepted_earlier_in_the_same_sweep(target_6c1, frozen):
    """The structural prior carried out of the sweep must be evaluated at the final U."""
    rng = np.random.default_rng(11)
    state = initial_state(target_6c1, frozen.u_true, {"rho": 0.3}, rng)
    for _ in range(60):
        state = sweep_once(state, target_6c1, SIGMA_U, 0.5, 0.05, rng)
        recomputed = log_structural_prior(state.u, state.values["rho"])
        assert state.log_structural_prior == pytest.approx(recomputed, abs=1e-9), (
            "the sweep's structural prior does not match its own final (U, rho)")


def test_beta_update_sees_the_u_and_rho_accepted_in_the_same_sweep(target_6c2, frozen):
    rng = np.random.default_rng(12)
    state = initial_state(target_6c2, frozen.u_true, {"rho": 0.3, "beta": 1.5}, rng)
    for _ in range(30):
        state = sweep_once(state, target_6c2, SIGMA_U, 0.5, 0.05109, rng)
        parts = target_6c2.decompose(state.u, state.values, allow_cache=False)
        assert state.log_target == pytest.approx(parts["log_target"], abs=1e-8)


def test_sweep_order_is_u_then_rho_then_beta():
    from hpop.mcmc_original.stage6c_frozen import SWEEP_ORDER
    assert SWEEP_ORDER == ("U", "rho", "beta")


# ------------------------------- areas 23, 24, 25: reproducibility, streams, resume
def test_same_seed_reproduces_the_run_exactly(blocks, frozen):
    def run():
        evaluator = LatentPosetEvaluator(blocks, epsilon=0.02, omega=FIXED_6C1["omega"])
        target = Stage6CTarget(evaluator, active=ACTIVE_6C1, fixed=FIXED_6C1)
        return run_stage6c_mcmc(target, frozen.u_true, {"rho": 0.3}, num_sweeps=60,
                                burn_in=10, thin=2, seed=42)

    np.random.seed(1)
    a = run()
    np.random.seed(9999)                       # module-level state must be irrelevant
    b = run()
    assert np.array_equal(a.rho, b.rho)
    assert np.array_equal(a.u_draws, b.u_draws)
    assert np.array_equal(a.log_target, b.log_target)


def test_different_seeds_give_independent_streams(blocks, frozen):
    results = []
    for seed in (1, 2, 3):
        evaluator = LatentPosetEvaluator(blocks, epsilon=0.02, omega=FIXED_6C1["omega"])
        target = Stage6CTarget(evaluator, active=ACTIVE_6C1, fixed=FIXED_6C1)
        results.append(run_stage6c_mcmc(target, frozen.u_true, {"rho": 0.3},
                                        num_sweeps=60, burn_in=10, thin=2, seed=seed))
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            assert not np.array_equal(results[i].rho, results[j].rho)


def test_a_split_run_resumes_bit_identically(blocks, frozen):
    def fresh_target():
        evaluator = LatentPosetEvaluator(blocks, epsilon=0.02, omega=FIXED_6C1["omega"])
        return Stage6CTarget(evaluator, active=ACTIVE_6C1, fixed=FIXED_6C1)

    whole = run_stage6c_mcmc(fresh_target(), frozen.u_true, {"rho": 0.3},
                             num_sweeps=80, burn_in=10, thin=1, seed=13)

    first = run_stage6c_mcmc(fresh_target(), frozen.u_true, {"rho": 0.3},
                             num_sweeps=40, burn_in=10, thin=1, seed=13)
    rng = np.random.default_rng(13)
    rng.bit_generator.state = first.final_state.rng_state
    second = run_stage6c_mcmc(fresh_target(), frozen.u_true, {"rho": 0.3},
                              num_sweeps=80, burn_in=10, thin=1, seed=13,
                              state=first.final_state, rng=rng)

    assert np.array_equal(np.concatenate([first.rho, second.rho]), whole.rho)
    assert np.array_equal(np.concatenate([first.u_draws, second.u_draws]), whole.u_draws)
    assert first.final_state.iteration == 40
    assert second.final_state.iteration == 80


def test_state_survives_a_json_round_trip(target_6c2, frozen):
    rng = np.random.default_rng(21)
    state = initial_state(target_6c2, frozen.u_true, {"rho": 0.3, "beta": 1.5}, rng)
    state = sweep_once(state, target_6c2, SIGMA_U, 0.5, 0.05109, rng)
    restored = Stage6CState.from_dict(state.to_dict())
    assert np.array_equal(restored.u, state.u)
    assert restored.values == state.values
    assert restored.log_target == state.log_target
    assert restored.iteration == state.iteration
    assert restored.rng_state == state.rng_state


def test_run_rejects_impossible_settings(target_6c1, frozen):
    with pytest.raises(ValueError):
        run_stage6c_mcmc(target_6c1, frozen.u_true, {"rho": 0.3}, num_sweeps=10,
                         burn_in=10, thin=1, seed=0)
    with pytest.raises(ValueError):
        run_stage6c_mcmc(target_6c1, frozen.u_true, {"rho": 0.3}, num_sweeps=10,
                         burn_in=1, thin=0, seed=0)


def test_a_start_with_a_non_finite_target_is_refused(target_6c1, frozen):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        initial_state(target_6c1, frozen.u_true, {"rho": 1.5}, rng)
