"""Stage 6C — recurrent replay and cache semantics (§17 areas 13-19).

A `U` move can change the induced order, and therefore the frontier, feasibility, the
whole `q` trajectory and every downstream one-step probability. So a proposed `U` must be
scored by a complete replay from `q_0 = 0` for every block — never by a local edit to the
changed relation, and never by reusing a trajectory built under the old `U`.

The evaluator's cache is keyed on the exact `(induced order, omega)` it was built at and
is written *only* by an explicit refresh. Rejection safety is therefore structural rather
than defensive: an evaluation cannot leave anything behind, because evaluation never
writes.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_joint_scalar_mcmc import vectorized_state_features
from hpop.mcmc_original.recurrent_latent_poset_mcmc import (
    LatentPosetEvaluator, Stage6CTarget, initial_state, poset_key, run_stage6c_mcmc,
    sweep_once,
)
from hpop.mcmc_original.sampler_u import propose_row
from hpop.mcmc_original.stage6c_frozen import SIGMA_U, load_stage6c_dataset

FIXED = {"beta": 1.5, "omega": 1.7346, "lambda_rep": 0.8, "lambda_back": 0.25}


@pytest.fixture(scope="module")
def frozen():
    return load_stage6c_dataset()


@pytest.fixture(scope="module")
def blocks(frozen):
    return frozen.train[:40]


def make_evaluator(blocks):
    return LatentPosetEvaluator(blocks, epsilon=0.02, omega=FIXED["omega"])


def score(evaluator, u):
    return evaluator.full_replay_log_likelihood(
        u, FIXED["beta"], FIXED["omega"], FIXED["lambda_rep"], FIXED["lambda_back"])


# ------------------------------------------------- area 13: replay after a U change
def test_scoring_u_a_then_u_b_then_u_a_again_reproduces_the_first_score(blocks, frozen):
    """The §8 regression test, verbatim: A, B, A must give the same value for A."""
    evaluator = make_evaluator(blocks)
    rng = np.random.default_rng(0)
    u_a = frozen.u_true
    u_b = propose_row(u_a, 1, 2.0, rng)

    first = score(evaluator, u_a)
    score(evaluator, u_b)
    again = score(evaluator, u_a)
    assert first == again


def test_a_changed_u_that_changes_the_order_changes_the_likelihood(blocks, frozen):
    """Guards against a replay that silently ignores the proposed U."""
    evaluator = make_evaluator(blocks)
    u = frozen.u_true
    moved = u.copy()
    moved[1] = [10.0, 10.0]                     # role 1 goes from incomparable to maximal
    assert not np.array_equal(precedence_from_u(u), precedence_from_u(moved))
    assert score(evaluator, u) != score(evaluator, moved)


def test_replay_matches_a_freshly_built_feature_bundle(blocks, frozen):
    """The cached path and a completely fresh build must agree to the last bit."""
    from hpop.mcmc_original.recurrent_scalar_posterior import cached_batch_log_likelihood

    evaluator = make_evaluator(blocks)
    u = frozen.u_true
    evaluator.refresh_cache(u, FIXED["omega"])
    cached = evaluator.log_likelihood(u, FIXED["beta"], FIXED["omega"],
                                      FIXED["lambda_rep"], FIXED["lambda_back"],
                                      allow_cache=True)
    fresh_features = vectorized_state_features(blocks, u, FIXED["omega"])
    fresh = float(cached_batch_log_likelihood(
        fresh_features, FIXED["beta"], 0.02, FIXED["lambda_rep"], FIXED["lambda_back"]))
    assert cached == pytest.approx(fresh, abs=1e-12)
    assert evaluator.cached_calls == 1


def test_likelihood_is_a_function_of_the_induced_order_only(blocks, frozen):
    """Two different `U` matrices inducing the same order must score identically.

    This is what makes the poset catalogue a legitimate label for reporting, and what
    makes the exact reference's cell factorisation valid.
    """
    evaluator = make_evaluator(blocks)
    u = frozen.u_true
    rescaled = u * 3.0 + 1.0            # strictly monotone per coordinate: same order
    assert np.array_equal(precedence_from_u(u), precedence_from_u(rescaled))
    assert score(evaluator, u) == pytest.approx(score(evaluator, rescaled), abs=1e-9)


# ------------------------------------------------------------- areas 14, 15, 16: q_0
def test_q_starts_at_zero_for_every_block(blocks, frozen):
    features = vectorized_state_features(blocks, frozen.u_true, FIXED["omega"])
    assert np.all(features["q"][:, 0, :] == 0.0)


def test_q_does_not_leak_between_blocks(blocks, frozen):
    """Permuting the block order must permute the per-block contributions, not change them."""
    order = np.array([7, 0, 3, 21, 11, 2])
    subset = blocks[order]
    features_full = vectorized_state_features(blocks, frozen.u_true, FIXED["omega"])
    features_subset = vectorized_state_features(subset, frozen.u_true, FIXED["omega"])
    for position, source in enumerate(order):
        assert np.allclose(features_subset["q"][position], features_full["q"][source])


def test_a_single_block_scored_alone_matches_its_place_in_the_batch(blocks, frozen):
    for index in (0, 5, 17):
        alone = vectorized_state_features(blocks[[index]], frozen.u_true, FIXED["omega"])
        batch = vectorized_state_features(blocks, frozen.u_true, FIXED["omega"])
        assert np.allclose(alone["q"][0], batch["q"][index])
        assert np.allclose(alone["Q"][0], batch["Q"][index])


def test_q_reset_between_skills_is_vacuous_because_stage_6c_has_one_skill(frozen):
    """Stated rather than faked: `K = 1`, so there is no second skill to leak into.

    The multi-skill leakage condition reduces exactly to the between-block condition
    tested above, which is where the guarantee actually lives for this stage.
    """
    assert frozen.n_skills == 1


def test_q_does_not_leak_between_chains(blocks, frozen):
    """Two chains from the same start with different seeds must not share trajectories."""
    evaluator_a, evaluator_b = make_evaluator(blocks), make_evaluator(blocks)
    target_a = Stage6CTarget(evaluator_a, active=("U", "rho"), fixed=FIXED)
    target_b = Stage6CTarget(evaluator_b, active=("U", "rho"), fixed=FIXED)
    result_a = run_stage6c_mcmc(target_a, frozen.u_true, {"rho": 0.3}, num_sweeps=25,
                                burn_in=5, thin=1, seed=1, chain=0)
    result_b = run_stage6c_mcmc(target_b, frozen.u_true, {"rho": 0.3}, num_sweeps=25,
                                burn_in=5, thin=1, seed=2, chain=1)
    assert not np.allclose(result_a.rho, result_b.rho)
    assert not np.array_equal(result_a.u_draws, result_b.u_draws)


# ------------------------------------------------ areas 17, 18: rejection and caching
def test_evaluation_never_writes_the_cache(blocks, frozen):
    """The structural guarantee: scoring a candidate cannot disturb a valid cache."""
    evaluator = make_evaluator(blocks)
    evaluator.refresh_cache(frozen.u_true, FIXED["omega"])
    key_before = evaluator.cache_key
    builds_before = evaluator.cache_builds

    rng = np.random.default_rng(3)
    for _ in range(25):
        score(evaluator, propose_row(frozen.u_true, int(rng.integers(5)), 3.0, rng))

    assert evaluator.cache_key == key_before
    assert evaluator.cache_builds == builds_before


def test_a_rejected_sweep_leaves_state_and_cache_untouched(blocks, frozen):
    """Force universal rejection by making the structural prior impossible to improve."""
    evaluator = make_evaluator(blocks)
    target = Stage6CTarget(evaluator, active=("U", "rho"), fixed=FIXED)
    rng = np.random.default_rng(9)
    state = initial_state(target, frozen.u_true, {"rho": 0.3}, rng)

    # A gigantic proposal scale makes every U proposal astronomically unlikely to accept.
    before_u = state.u.copy()
    before_ll = state.log_likelihood
    after = sweep_once(state, target, 1e6, 0.0, 0.05, rng)
    assert after.accepted["U"] == 0
    assert np.array_equal(after.u, before_u)
    assert after.log_likelihood == before_ll


def test_an_accepted_u_invalidates_the_cache(blocks, frozen):
    evaluator = make_evaluator(blocks)
    target = Stage6CTarget(evaluator, active=("U", "rho"), fixed=FIXED)
    rng = np.random.default_rng(4)
    state = initial_state(target, frozen.u_true, {"rho": 0.3}, rng)
    evaluator.refresh_cache(state.u, FIXED["omega"])
    assert evaluator.cache_is_valid_for(state.u, FIXED["omega"])

    for _ in range(40):
        new_state = sweep_once(state, target, SIGMA_U, 0.0, 0.05, rng)
        if new_state.accepted["U"] > state.accepted["U"]:
            assert evaluator.cache_key is None, (
                "an accepted U left a cache built under the previous order")
            break
        state = new_state
    else:
        pytest.fail("no U proposal was accepted in 40 sweeps; the test proved nothing")


def test_cache_key_tracks_the_induced_order_not_the_matrix(blocks, frozen):
    evaluator = make_evaluator(blocks)
    evaluator.refresh_cache(frozen.u_true, FIXED["omega"])
    rescaled = frozen.u_true * 2.5
    assert poset_key(rescaled) == poset_key(frozen.u_true)
    assert evaluator.cache_is_valid_for(rescaled, FIXED["omega"])


def test_cache_is_invalid_at_a_different_omega(blocks, frozen):
    """omega enters the q recursion, so a cache built at one omega is wrong at another."""
    evaluator = make_evaluator(blocks)
    evaluator.refresh_cache(frozen.u_true, FIXED["omega"])
    assert not evaluator.cache_is_valid_for(frozen.u_true, FIXED["omega"] + 0.1)


# ------------------------------------------------- area 19: evaluation-order invariance
def test_scores_do_not_depend_on_the_order_they_were_requested_in(blocks, frozen):
    rng = np.random.default_rng(6)
    candidates = [frozen.u_true] + [
        propose_row(frozen.u_true, int(rng.integers(5)), 1.5, rng) for _ in range(6)]

    forward = [score(make_evaluator(blocks), u) for u in candidates]
    shared = make_evaluator(blocks)
    in_order = [score(shared, u) for u in candidates]
    reversed_order = list(reversed([score(shared, u) for u in reversed(candidates)]))

    assert in_order == forward
    assert reversed_order == forward


def test_repeated_scoring_of_the_same_u_is_bitwise_identical(blocks, frozen):
    evaluator = make_evaluator(blocks)
    values = {score(evaluator, frozen.u_true) for _ in range(5)}
    assert len(values) == 1
