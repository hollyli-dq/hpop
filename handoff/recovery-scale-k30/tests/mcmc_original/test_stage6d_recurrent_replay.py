"""Stage 6D — recurrent replay and cache semantics (§18 areas 23-33).

`U` and `omega` are the dangerous pair. A changed `U` can change `H = h(U)` and with it
the frontier and the whole `q` trajectory; a changed `omega` changes `kappa` inside the
`q` recursion itself. Both must be scored by a complete replay from `q_0 = 0`, and the
cache — keyed on `(induced order, omega)` — must never be readable at a different value
of either, nor writable by an evaluation.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_joint_scalar_mcmc import vectorized_state_features
from hpop.mcmc_original.recurrent_latent_poset_mcmc import LatentPosetEvaluator, poset_key
from hpop.mcmc_original.recurrent_oracle_joint_mcmc import (
    Stage6DTarget, initial_state, run_oracle_joint_mcmc, sweep_once,
)
from hpop.mcmc_original.recurrent_scalar_posterior import cached_batch_log_likelihood
from hpop.mcmc_original.sampler_u import propose_row
from hpop.mcmc_original.stage6d_frozen import (
    ACTIVE_6D, REGISTERED_SCALES, load_stage6d_dataset,
)


@pytest.fixture(scope="module")
def frozen():
    return load_stage6d_dataset()


@pytest.fixture(scope="module")
def blocks(frozen):
    return frozen.train[:40]


def evaluator_for(blocks, frozen):
    return LatentPosetEvaluator(blocks, epsilon=frozen.epsilon,
                                omega=frozen.truth["omega"])


def score(evaluator, u, frozen, **overrides):
    values = {**{k: float(v) for k, v in frozen.truth.items()}, **overrides}
    return evaluator.full_replay_log_likelihood(
        u, values["beta"], values["omega"], values["lambda_rep"],
        values["lambda_back"])


# --------------------------------------- areas 23, 24: U and omega force full replay
def test_a_u_proposal_triggers_a_complete_replay(blocks, frozen):
    evaluator = evaluator_for(blocks, frozen)
    evaluator.refresh_cache(frozen.u_true, frozen.truth["omega"])
    before = evaluator.full_replay_calls
    moved = frozen.u_true.copy()
    moved[1] = [10.0, 10.0]
    evaluator.log_likelihood(moved, frozen.truth["beta"], frozen.truth["omega"],
                             frozen.truth["lambda_rep"], frozen.truth["lambda_back"],
                             allow_cache=True)
    assert evaluator.full_replay_calls == before + 1, (
        "a U at a different induced order read the cache instead of replaying")


def test_an_omega_proposal_triggers_a_complete_replay(blocks, frozen):
    evaluator = evaluator_for(blocks, frozen)
    evaluator.refresh_cache(frozen.u_true, frozen.truth["omega"])
    before = evaluator.full_replay_calls
    evaluator.log_likelihood(frozen.u_true, frozen.truth["beta"],
                             frozen.truth["omega"] + 0.3, frozen.truth["lambda_rep"],
                             frozen.truth["lambda_back"], allow_cache=True)
    assert evaluator.full_replay_calls == before + 1, (
        "a proposed omega read a cache built at a different omega")


def test_the_three_scalars_outside_the_q_recursion_may_use_the_cache(blocks, frozen):
    evaluator = evaluator_for(blocks, frozen)
    evaluator.refresh_cache(frozen.u_true, frozen.truth["omega"])
    before = evaluator.full_replay_calls
    for name in ("beta", "lambda_rep", "lambda_back"):
        values = {k: float(v) for k, v in frozen.truth.items()}
        values[name] = values[name] * 1.4
        evaluator.log_likelihood(frozen.u_true, values["beta"], values["omega"],
                                 values["lambda_rep"], values["lambda_back"],
                                 allow_cache=True)
    assert evaluator.full_replay_calls == before


def test_a_sweep_replays_exactly_m_plus_one_times(blocks, frozen):
    """m U rows plus one omega. Any other coordinate adding a replay is a bug."""
    target = Stage6DTarget(evaluator_for(blocks, frozen), active=ACTIVE_6D)
    rng = np.random.default_rng(0)
    state = initial_state(target, frozen.u_true,
                          {"rho": 0.3, **{k: float(v) for k, v in frozen.truth.items()}},
                          rng)
    target.evaluator.full_replay_calls = 0
    for _ in range(15):
        state = sweep_once(state, target, REGISTERED_SCALES, rng)
    assert target.evaluator.full_replay_calls == 15 * (frozen.n_roles + 1)


# ------------------------------------------- area 32/33: order invariance and parity
def test_score_a_then_b_then_a_again(blocks, frozen):
    evaluator = evaluator_for(blocks, frozen)
    rng = np.random.default_rng(1)
    a = frozen.u_true
    b = propose_row(a, 1, 2.0, rng)
    first = score(evaluator, a, frozen)
    score(evaluator, b, frozen)
    assert score(evaluator, a, frozen) == first


def test_scores_do_not_depend_on_request_order(blocks, frozen):
    rng = np.random.default_rng(2)
    candidates = [frozen.u_true] + [propose_row(frozen.u_true, int(rng.integers(5)),
                                                1.5, rng) for _ in range(5)]
    isolated = [score(evaluator_for(blocks, frozen), u, frozen) for u in candidates]
    shared = evaluator_for(blocks, frozen)
    forward = [score(shared, u, frozen) for u in candidates]
    backward = list(reversed([score(shared, u, frozen) for u in reversed(candidates)]))
    assert forward == isolated
    assert backward == isolated


def test_cached_and_uncached_evaluations_agree_exactly(blocks, frozen):
    evaluator = evaluator_for(blocks, frozen)
    evaluator.refresh_cache(frozen.u_true, frozen.truth["omega"])
    cached = evaluator.log_likelihood(
        frozen.u_true, frozen.truth["beta"], frozen.truth["omega"],
        frozen.truth["lambda_rep"], frozen.truth["lambda_back"], allow_cache=True)
    fresh = vectorized_state_features(blocks, frozen.u_true, frozen.truth["omega"])
    direct = float(cached_batch_log_likelihood(
        fresh, frozen.truth["beta"], frozen.epsilon, frozen.truth["lambda_rep"],
        frozen.truth["lambda_back"]))
    assert cached == pytest.approx(direct, abs=1e-12)


def test_repeated_scoring_is_bitwise_identical(blocks, frozen):
    evaluator = evaluator_for(blocks, frozen)
    assert len({score(evaluator, frozen.u_true, frozen) for _ in range(5)}) == 1


# --------------------------------------------------- areas 25, 26, 27: q_0 resets
def test_q_starts_at_zero_for_every_block(blocks, frozen):
    features = vectorized_state_features(blocks, frozen.u_true, frozen.truth["omega"])
    assert np.all(features["q"][:, 0, :] == 0.0)


def test_q_starts_at_zero_at_every_omega(blocks, frozen):
    for omega in (-1.0, 0.0, frozen.truth["omega"], 3.0):
        features = vectorized_state_features(blocks, frozen.u_true, omega)
        assert np.all(features["q"][:, 0, :] == 0.0)


def test_q_does_not_leak_between_blocks(blocks, frozen):
    order = np.array([5, 0, 11, 2, 9])
    subset = vectorized_state_features(blocks[order], frozen.u_true,
                                       frozen.truth["omega"])
    full = vectorized_state_features(blocks, frozen.u_true, frozen.truth["omega"])
    for position, source in enumerate(order):
        assert np.allclose(subset["q"][position], full["q"][source])


def test_q_reset_between_skills_is_vacuous_with_one_skill(frozen):
    """Stated rather than faked: K = 1, so the condition reduces to the block one."""
    assert frozen.n_skills == 1


def test_q_does_not_leak_between_chains(blocks, frozen):
    results = []
    for seed in (1, 2):
        target = Stage6DTarget(evaluator_for(blocks, frozen), active=ACTIVE_6D)
        results.append(run_oracle_joint_mcmc(
            target, frozen.u_true,
            {"rho": 0.3, **{k: float(v) for k, v in frozen.truth.items()}},
            25, 5, 1, seed=seed))
    assert not np.array_equal(results[0].scalars["rho"], results[1].scalars["rho"])
    assert not np.array_equal(results[0].u_draws, results[1].u_draws)


# ------------------------------ areas 28, 29, 30, 31: rejection and cache invalidation
def test_evaluation_never_writes_the_cache(blocks, frozen):
    evaluator = evaluator_for(blocks, frozen)
    evaluator.refresh_cache(frozen.u_true, frozen.truth["omega"])
    key, builds = evaluator.cache_key, evaluator.cache_builds
    rng = np.random.default_rng(3)
    for _ in range(20):
        score(evaluator, propose_row(frozen.u_true, int(rng.integers(5)), 3.0, rng),
              frozen)
        score(evaluator, frozen.u_true, frozen, omega=frozen.truth["omega"] + 0.5)
    assert evaluator.cache_key == key
    assert evaluator.cache_builds == builds


def test_an_accepted_u_invalidates_the_cache(blocks, frozen):
    target = Stage6DTarget(evaluator_for(blocks, frozen), active=ACTIVE_6D)
    rng = np.random.default_rng(4)
    # Start from an antichain rather than the truth. Started at U_TRUE the chain accepts
    # U moves but they wander *inside* the true order's cell — the likelihood separates
    # that order from its neighbours by hundreds of nats — so the induced order would
    # never change and the invalidation path would go untested.
    antichain = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])
    state = initial_state(target, antichain,
                          {"rho": 0.3, **{k: float(v) for k, v in frozen.truth.items()}},
                          rng)
    for _ in range(60):
        target.evaluator.ensure_cache(state.u, state.values["omega"])
        new_state = sweep_once(state, target, REGISTERED_SCALES, rng)
        order_changed = not np.array_equal(precedence_from_u(state.u),
                                           precedence_from_u(new_state.u))
        if new_state.accepted["U"] > state.accepted["U"] and order_changed:
            # The sweep invalidates on acceptance; the later cache-safe scalars may then
            # legitimately rebuild at the NEW state. What must never happen is a cache
            # still valid for the superseded order.
            assert not target.evaluator.cache_is_valid_for(
                state.u, state.values["omega"]), (
                "an accepted U left a cache built at the previous induced order")
            if target.evaluator.cache_key is not None:
                assert target.evaluator.cache_is_valid_for(
                    new_state.u, new_state.values["omega"])
            break
        state = new_state
    else:
        pytest.fail("no U proposal changed the induced order in 60 sweeps")


def test_an_accepted_omega_invalidates_the_cache(blocks, frozen):
    target = Stage6DTarget(evaluator_for(blocks, frozen), active=ACTIVE_6D)
    rng = np.random.default_rng(5)
    state = initial_state(target, frozen.u_true,
                          {"rho": 0.3, **{k: float(v) for k, v in frozen.truth.items()}},
                          rng)
    for _ in range(40):
        before = state.accepted["omega"]
        stale_u, stale_omega = state.u.copy(), state.values["omega"]
        target.evaluator.ensure_cache(stale_u, stale_omega)
        state = sweep_once(state, target, REGISTERED_SCALES, rng)
        if state.accepted["omega"] > before:
            # omega is accepted mid-sweep and the cache is invalidated there; lambda_rep
            # and lambda_back then rebuild it at the new omega. The invariant is that the
            # cache is never valid at the superseded omega afterwards.
            assert not target.evaluator.cache_is_valid_for(stale_u, stale_omega)
            if target.evaluator.cache_key is not None:
                assert target.evaluator.cache_is_valid_for(state.u,
                                                           state.values["omega"])
            break
    else:
        pytest.fail("no omega proposal was accepted in 40 sweeps")


def test_cache_key_tracks_the_induced_order_not_the_matrix(blocks, frozen):
    evaluator = evaluator_for(blocks, frozen)
    evaluator.refresh_cache(frozen.u_true, frozen.truth["omega"])
    rescaled = frozen.u_true * 3.0
    assert poset_key(rescaled) == poset_key(frozen.u_true)
    assert evaluator.cache_is_valid_for(rescaled, frozen.truth["omega"])
    assert not evaluator.cache_is_valid_for(rescaled, frozen.truth["omega"] + 0.05)


def test_likelihood_depends_on_u_only_through_the_induced_order(blocks, frozen):
    evaluator = evaluator_for(blocks, frozen)
    rescaled = frozen.u_true * 2.5 + 1.0
    assert np.array_equal(precedence_from_u(frozen.u_true),
                          precedence_from_u(rescaled))
    assert score(evaluator, frozen.u_true, frozen) == pytest.approx(
        score(evaluator, rescaled, frozen), abs=1e-9)
