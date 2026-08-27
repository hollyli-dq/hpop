"""Stage 6B — omega replay, q resets, and the cache's acceptance/rejection semantics.

`kappa = sigmoid(omega)` *is* the validity recursion, so an omega proposal invalidates the
whole `q` trajectory. Section 4's requirements are all failure modes that would leave the
chain sampling a subtly wrong target while every convergence diagnostic still looked
healthy, so each is pinned here:

* every omega evaluation replays from `q_0 = 0`, for every block;
* `q` never leaks across blocks, across evaluations, or across chains;
* a rejected omega leaves both the scalar state and every cached value untouched;
* an accepted omega invalidates the cache rather than silently reusing it;
* the same state evaluates identically from a clean evaluator, after a rejection, after
  an acceptance, and under a different evaluation order.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_original.recurrent_joint_scalar_mcmc import (
    ACTIVE_B3, JointScalarTarget, RecurrentJointEvaluator, run_joint_scalar_mcmc,
    vectorized_state_features,
)
from hpop.mcmc_original.recurrent_rfs import recurrent_rfs_log_likelihood
from hpop.mcmc_original.recurrent_scalar_posterior import precompute_state_features
from hpop.mcmc_original.stage6b_frozen import load_frozen_dataset

SCALES = {"beta": 0.05109, "omega": 0.27891, "lambda_rep": 0.07086,
          "lambda_back": 0.21734}


@pytest.fixture(scope="module")
def frozen():
    return load_frozen_dataset()


@pytest.fixture(scope="module")
def small(frozen):
    """A 40-block slice — same model, small enough for per-block reference loops."""
    return frozen.train[:40]


def make_evaluator(roles, frozen):
    return RecurrentJointEvaluator(roles, frozen.u_true, frozen.epsilon)


# --------------------------------------------------------------------- full replay from 0
def test_every_omega_evaluation_replays_from_q0_zero(small, frozen):
    """Feature trajectories always start at zero, whatever was evaluated before."""
    evaluator = make_evaluator(small, frozen)
    for omega in (0.4, 1.7346, 3.1):
        features = vectorized_state_features(small, frozen.u_true, omega)
        assert np.abs(features["q"][:, 0, :]).max() == 0.0, "q_0 must be exactly zero"


def test_vectorized_features_match_the_stage_6b1_loop_builder(small, frozen):
    omega = 1.1
    reference = precompute_state_features([tuple(r) for r in small], frozen.u_true, omega)
    mine = vectorized_state_features(small, frozen.u_true, omega)
    for key in ("F", "Q", "q", "C_back"):
        assert np.abs(reference[key] - mine[key]).max() == 0.0
    assert np.array_equal(reference["obs"], mine["obs"])


def test_q_does_not_leak_between_traces(small, frozen):
    """The batch total is the sum of independently replayed blocks, in any order."""
    evaluator = make_evaluator(small, frozen)
    kw = dict(frozen.truth)
    total = evaluator.full_replay_log_likelihood(
        kw["beta"], kw["omega"], kw["lambda_rep"], kw["lambda_back"])
    per_block = sum(recurrent_rfs_log_likelihood(
        tuple(row), frozen.u_true, kw["beta"], frozen.epsilon, kw["omega"],
        kw["lambda_rep"], kw["lambda_back"]) for row in small)
    assert total == pytest.approx(per_block, abs=1e-9)

    shuffled = small[::-1].copy()
    other = make_evaluator(shuffled, frozen)
    assert other.full_replay_log_likelihood(
        kw["beta"], kw["omega"], kw["lambda_rep"], kw["lambda_back"]
    ) == pytest.approx(total, abs=1e-9)


def test_a_single_block_is_unaffected_by_its_neighbours(small, frozen):
    kw = dict(frozen.truth)
    alone = make_evaluator(small[3:4], frozen).full_replay_log_likelihood(
        kw["beta"], kw["omega"], kw["lambda_rep"], kw["lambda_back"])
    reference = recurrent_rfs_log_likelihood(
        tuple(small[3]), frozen.u_true, kw["beta"], frozen.epsilon, kw["omega"],
        kw["lambda_rep"], kw["lambda_back"])
    assert alone == pytest.approx(reference, abs=1e-10)


def test_evaluation_order_does_not_change_any_result(small, frozen):
    """Interleaving different omegas must not let one contaminate another."""
    evaluator = make_evaluator(small, frozen)
    kw = dict(frozen.truth)
    omegas = [0.5, 1.7346, 2.6, 1.0]
    forward = [evaluator.full_replay_log_likelihood(
        kw["beta"], w, kw["lambda_rep"], kw["lambda_back"]) for w in omegas]
    backward = [evaluator.full_replay_log_likelihood(
        kw["beta"], w, kw["lambda_rep"], kw["lambda_back"]) for w in reversed(omegas)]
    assert forward == pytest.approx(list(reversed(backward)), abs=0.0)

    fresh = [make_evaluator(small, frozen).full_replay_log_likelihood(
        kw["beta"], w, kw["lambda_rep"], kw["lambda_back"]) for w in omegas]
    assert forward == pytest.approx(fresh, abs=0.0)


# ------------------------------------------------------------------------ cache semantics
def test_a_cache_built_at_one_omega_is_never_read_at_another(small, frozen):
    """The key check falls through to full replay, so a stale read is impossible."""
    evaluator = make_evaluator(small, frozen)
    kw = dict(frozen.truth)
    evaluator.refresh_cache(kw["omega"])
    assert evaluator.cache_is_valid_for(kw["omega"])
    assert not evaluator.cache_is_valid_for(kw["omega"] + 0.3)

    # a large offset, so the "stale value would have been different" check below has
    # real force on this 40-block slice rather than sitting inside numerical noise
    other = kw["omega"] + 1.5
    with_cache_allowed = evaluator.log_likelihood(
        kw["beta"], other, kw["lambda_rep"], kw["lambda_back"], allow_cache=True)
    exact = make_evaluator(small, frozen).full_replay_log_likelihood(
        kw["beta"], other, kw["lambda_rep"], kw["lambda_back"])
    assert with_cache_allowed == pytest.approx(exact, abs=0.0)

    # and the stale entry was genuinely different, so this was not a vacuous check
    stale = evaluator.log_likelihood(kw["beta"], kw["omega"], kw["lambda_rep"],
                                     kw["lambda_back"], allow_cache=True)
    assert abs(stale - exact) > 1.0


def test_evaluation_never_writes_the_cache(small, frozen):
    evaluator = make_evaluator(small, frozen)
    kw = dict(frozen.truth)
    assert evaluator.cache_omega is None
    evaluator.log_likelihood(kw["beta"], kw["omega"], kw["lambda_rep"],
                             kw["lambda_back"], allow_cache=True)
    assert evaluator.cache_omega is None, "an evaluation must not populate the cache"
    evaluator.refresh_cache(kw["omega"])
    builds = evaluator.cache_builds
    evaluator.log_likelihood(kw["beta"], kw["omega"] + 1.0, kw["lambda_rep"],
                             kw["lambda_back"], allow_cache=True)
    assert evaluator.cache_omega == kw["omega"] and evaluator.cache_builds == builds


def test_invalidate_clears_the_omega_dependent_cache(small, frozen):
    evaluator = make_evaluator(small, frozen)
    evaluator.refresh_cache(frozen.truth["omega"])
    assert evaluator.cache_omega is not None
    evaluator.invalidate()
    assert evaluator.cache_omega is None and evaluator._features is None


def test_cached_and_full_replay_agree_exactly_at_the_cache_omega(small, frozen):
    evaluator = make_evaluator(small, frozen)
    kw = dict(frozen.truth)
    evaluator.refresh_cache(kw["omega"])
    rng = np.random.default_rng(0)
    for _ in range(15):
        beta = float(rng.uniform(0.3, 3.0))
        lrep = float(rng.uniform(0.05, 2.0))
        lback = float(rng.uniform(0.01, 1.5))
        cached = evaluator.log_likelihood(beta, kw["omega"], lrep, lback, allow_cache=True)
        exact = evaluator.full_replay_log_likelihood(beta, kw["omega"], lrep, lback)
        assert cached == pytest.approx(exact, abs=1e-9)


# --------------------------------------------------- acceptance / rejection state hygiene
def _state_snapshot(evaluator, target, values):
    return {"cache_omega": evaluator.cache_omega,
            "log_posterior": target.log_posterior(values, allow_cache=False),
            "features": None if evaluator._features is None
            else {k: np.array(v, copy=True) for k, v in evaluator._features.items()
                  if isinstance(v, np.ndarray)}}


def test_a_rejected_omega_leaves_scalar_and_cached_state_unchanged(small, frozen):
    """The registered requirement: after a rejection nothing may have moved."""
    from hpop.mcmc_original.recurrent_joint_scalar_mcmc import sweep_once, JointChainState
    from hpop.mcmc_original.recurrent_scalar_mcmc import build_proposal, Proposal

    evaluator = make_evaluator(small, frozen)
    target = JointScalarTarget(evaluator, ACTIVE_B3, {})
    values = {n: frozen.truth[n] for n in ACTIVE_B3}
    evaluator.refresh_cache(values["omega"])
    parts = target.decompose(values)
    state = JointChainState(values=dict(values), log_likelihood=parts["log_likelihood"],
                            log_prior_components=parts["log_prior_components"],
                            log_posterior=parts["log_posterior"],
                            proposed={n: 0 for n in ACTIVE_B3},
                            accepted={n: 0 for n in ACTIVE_B3})
    before = _state_snapshot(evaluator, target, values)

    # an omega proposal so bad it is certain to be rejected
    hopeless = {n: build_proposal(n, 1e-9) for n in ACTIVE_B3}
    hopeless["omega"] = lambda current, rng: Proposal(current + 40.0, 0.0)
    after_state = sweep_once(state, target, hopeless, np.random.default_rng(0))

    assert after_state.accepted["omega"] == 0, "the probe must actually reject"
    assert after_state.values["omega"] == values["omega"]
    after = _state_snapshot(evaluator, target, values)
    assert after["cache_omega"] == before["cache_omega"]
    assert after["log_posterior"] == before["log_posterior"]
    for key, array in before["features"].items():
        assert np.array_equal(after["features"][key], array), f"{key} changed on rejection"


def test_an_accepted_omega_invalidates_the_cache(small, frozen):
    from hpop.mcmc_original.recurrent_joint_scalar_mcmc import sweep_once, JointChainState
    from hpop.mcmc_original.recurrent_scalar_mcmc import build_proposal, Proposal

    evaluator = make_evaluator(small, frozen)
    target = JointScalarTarget(evaluator, ("omega",),
                               {n: frozen.truth[n] for n in ("beta", "lambda_rep",
                                                             "lambda_back")})
    values = {"omega": frozen.truth["omega"]}
    evaluator.refresh_cache(values["omega"])
    parts = target.decompose(values)
    state = JointChainState(values=dict(values), log_likelihood=parts["log_likelihood"],
                            log_prior_components=parts["log_prior_components"],
                            log_posterior=parts["log_posterior"],
                            proposed={"omega": 0}, accepted={"omega": 0})

    # a tiny move towards the mode, accepted with near-certainty under a fixed uniform
    tiny = {"omega": lambda current, rng: Proposal(current + 1e-4, 0.0)}

    class _AlwaysAccept:
        """Forces the accept branch while still exposing a real bit generator."""

        def __init__(self):
            self._rng = np.random.default_rng(0)

        def random(self):
            return 1e-300

        @property
        def bit_generator(self):
            return self._rng.bit_generator

    after = sweep_once(state, target, tiny, _AlwaysAccept())
    assert after.accepted["omega"] == 1
    assert evaluator.cache_omega is None, "an accepted omega must invalidate the cache"


def test_the_same_state_evaluates_identically_from_clean_rejected_and_accepted(small,
                                                                              frozen):
    """Section 4's four-way agreement, on one probe state."""
    evaluator = make_evaluator(small, frozen)
    kw = dict(frozen.truth)
    probe = (kw["beta"], kw["omega"], kw["lambda_rep"], kw["lambda_back"])

    clean = make_evaluator(small, frozen).full_replay_log_likelihood(*probe)

    # after a rejected proposal at a different omega
    evaluator.full_replay_log_likelihood(kw["beta"], kw["omega"] + 0.9,
                                         kw["lambda_rep"], kw["lambda_back"])
    after_rejection = evaluator.full_replay_log_likelihood(*probe)

    # after an acceptance, cache refreshed at the new omega then brought back
    evaluator.refresh_cache(kw["omega"] + 0.9)
    evaluator.invalidate()
    evaluator.refresh_cache(kw["omega"])
    after_acceptance = evaluator.log_likelihood(*probe, allow_cache=True)

    reordered = make_evaluator(small[::-1].copy(), frozen).full_replay_log_likelihood(*probe)

    assert after_rejection == pytest.approx(clean, abs=0.0)
    assert after_acceptance == pytest.approx(clean, abs=1e-9)
    assert reordered == pytest.approx(clean, abs=1e-9)


def test_q_does_not_leak_between_chains(small, frozen):
    """Two chains run through the same evaluator must not influence each other."""
    evaluator = make_evaluator(small, frozen)
    target = JointScalarTarget(evaluator, ACTIVE_B3, {})
    start = {n: frozen.truth[n] for n in ACTIVE_B3}

    alone = run_joint_scalar_mcmc(target, start, SCALES, 120, 20, 2, seed=5, chain=0)
    run_joint_scalar_mcmc(target, {**start, "omega": start["omega"] + 1.0}, SCALES,
                          120, 20, 2, seed=99, chain=1)
    again = run_joint_scalar_mcmc(target, start, SCALES, 120, 20, 2, seed=5, chain=0)

    for name in ACTIVE_B3:
        assert np.array_equal(alone.draws[name], again.draws[name]), (
            f"{name} changed after an unrelated chain ran through the same evaluator")
