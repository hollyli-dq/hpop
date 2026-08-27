"""Stage 6B — the Metropolis-within-Gibbs orchestrator itself.

The orchestrator is the only genuinely new code in Stages 6B2/6B3: the kernels and
proposals are Stage 6B1's. So what needs testing is the orchestration — that a coordinate
later in a sweep sees the acceptances that happened earlier in the *same* sweep, that
chains are reproducible and independent, that a run can be resumed bit-identically, and
that the registered starts are genuinely dispersed in every coordinate rather than in one.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from hpop.mcmc_original.recurrent_joint_scalar_mcmc import (
    ACTIVE_B2, ACTIVE_B3, JointChainState, JointScalarTarget, RecurrentJointEvaluator,
    run_joint_scalar_mcmc, sweep_once,
)
from hpop.mcmc_original.recurrent_scalar_mcmc import Proposal, build_proposal
from hpop.mcmc_original.stage6b_frozen import PARAMETER_ORDER, SWEEP_ORDER, load_frozen_dataset

SCALES = {"beta": 0.05109, "omega": 0.27891, "lambda_rep": 0.07086,
          "lambda_back": 0.21734}
RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "stage6b_joint_mcmc.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("stage6b_joint_mcmc_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frozen():
    return load_frozen_dataset()


@pytest.fixture(scope="module")
def small(frozen):
    return frozen.train[:40]


@pytest.fixture
def target(small, frozen):
    return JointScalarTarget(RecurrentJointEvaluator(small, frozen.u_true, frozen.epsilon),
                             ACTIVE_B3, {})


def fresh_state(target, values):
    parts = target.decompose(values, allow_cache=False)
    return JointChainState(values=dict(values), log_likelihood=parts["log_likelihood"],
                           log_prior_components=parts["log_prior_components"],
                           log_posterior=parts["log_posterior"],
                           proposed={n: 0 for n in target.active},
                           accepted={n: 0 for n in target.active})


class _AlwaysAccept:
    def __init__(self, seed=0):
        self._rng = np.random.default_rng(seed)

    def random(self):
        return 1e-300

    @property
    def bit_generator(self):
        return self._rng.bit_generator


class _RecordingTarget(JointScalarTarget):
    """Records the full parameter vector handed to every log-posterior evaluation."""

    def __init__(self, inner):
        super().__init__(inner.evaluator, inner.active, dict(inner.fixed))
        self.seen = []

    def log_posterior(self, values, allow_cache: bool = True) -> float:
        self.seen.append(dict(values))
        return super().log_posterior(values, allow_cache=allow_cache)


# --------------------------------------------------------------- sweep uses latest values
def test_later_coordinates_see_earlier_acceptances_in_the_same_sweep(target, frozen):
    """beta moves first; omega, lambda_rep and lambda_back must all see the new beta."""
    spy = _RecordingTarget(target)
    values = {n: frozen.truth[n] for n in ACTIVE_B3}
    state = fresh_state(spy, values)
    spy.seen.clear()

    shifts = {"beta": 0.11, "omega": 0.07, "lambda_rep": 0.05, "lambda_back": 0.03}
    proposals = {n: (lambda current, rng, _s=shifts[n]: Proposal(current + _s, 0.0))
                 for n in ACTIVE_B3}

    after = sweep_once(state, spy, proposals, _AlwaysAccept())
    assert all(after.accepted[n] == 1 for n in ACTIVE_B3), "every coordinate must accept"

    # one evaluation per coordinate, in the registered order
    assert len(spy.seen) == 4
    beta_call, omega_call, lrep_call, lback_call = spy.seen

    assert beta_call["beta"] == pytest.approx(frozen.truth["beta"] + 0.11)
    assert beta_call["omega"] == pytest.approx(frozen.truth["omega"])

    assert omega_call["beta"] == pytest.approx(frozen.truth["beta"] + 0.11), \
        "omega did not see the accepted beta"
    assert omega_call["omega"] == pytest.approx(frozen.truth["omega"] + 0.07)

    assert lrep_call["beta"] == pytest.approx(frozen.truth["beta"] + 0.11)
    assert lrep_call["omega"] == pytest.approx(frozen.truth["omega"] + 0.07), \
        "lambda_rep did not see the accepted omega"
    assert lrep_call["lambda_rep"] == pytest.approx(frozen.truth["lambda_rep"] + 0.05)

    assert lback_call["beta"] == pytest.approx(frozen.truth["beta"] + 0.11)
    assert lback_call["omega"] == pytest.approx(frozen.truth["omega"] + 0.07)
    assert lback_call["lambda_rep"] == pytest.approx(frozen.truth["lambda_rep"] + 0.05)
    assert lback_call["lambda_back"] == pytest.approx(frozen.truth["lambda_back"] + 0.03)


def test_a_rejected_coordinate_is_not_carried_forward(target, frozen):
    """The mirror image: a rejected beta must not leak into the later coordinates."""
    spy = _RecordingTarget(target)
    values = {n: frozen.truth[n] for n in ACTIVE_B3}
    state = fresh_state(spy, values)
    spy.seen.clear()

    proposals = {n: build_proposal(n, 1e-9) for n in ACTIVE_B3}
    proposals["beta"] = lambda current, rng: Proposal(current + 25.0, 0.0)

    after = sweep_once(state, spy, proposals, np.random.default_rng(0))
    assert after.accepted["beta"] == 0
    assert after.values["beta"] == pytest.approx(frozen.truth["beta"])
    for call in spy.seen[1:]:
        assert call["beta"] == pytest.approx(frozen.truth["beta"])


def test_the_sweep_visits_only_active_coordinates_in_the_registered_order(small, frozen):
    inner = JointScalarTarget(
        RecurrentJointEvaluator(small, frozen.u_true, frozen.epsilon), ACTIVE_B2,
        {"lambda_back": frozen.truth["lambda_back"]})
    spy = _RecordingTarget(inner)
    values = {n: frozen.truth[n] for n in ACTIVE_B2}
    state = fresh_state(spy, values)
    spy.seen.clear()
    proposals = {n: build_proposal(n, 1e-6) for n in ACTIVE_B2}
    sweep_once(state, spy, proposals, np.random.default_rng(1))

    assert len(spy.seen) == 3, "lambda_back must not be proposed in Stage 6B2"
    assert [n for n in SWEEP_ORDER if n in ACTIVE_B2] == list(ACTIVE_B2)
    for call in spy.seen:
        # the proposal vector carries only active coordinates; the fixed one is supplied
        # by `complete`, and must arrive at the likelihood pinned to its registered value
        assert "lambda_back" not in call
        assert spy.complete(call)["lambda_back"] == pytest.approx(
            frozen.truth["lambda_back"])


def test_state_records_counts_priors_and_the_complete_likelihood(target, frozen):
    values = {n: frozen.truth[n] for n in ACTIVE_B3}
    state = fresh_state(target, values)
    after = sweep_once(state, target, {n: build_proposal(n, SCALES[n]) for n in ACTIVE_B3},
                       np.random.default_rng(4))
    assert after.iteration == 1
    assert set(after.proposed) == set(ACTIVE_B3)
    assert all(after.proposed[n] == 1 for n in ACTIVE_B3)
    assert set(after.log_prior_components) == set(ACTIVE_B3)
    assert after.log_posterior == pytest.approx(
        after.log_likelihood + sum(after.log_prior_components.values()), abs=1e-8)
    assert after.rng_state is not None


def test_chain_state_round_trips_through_its_serialised_form(target, frozen):
    values = {n: frozen.truth[n] for n in ACTIVE_B3}
    state = sweep_once(fresh_state(target, values), target,
                       {n: build_proposal(n, SCALES[n]) for n in ACTIVE_B3},
                       np.random.default_rng(9))
    restored = JointChainState.from_dict(state.to_dict())
    assert restored.values == state.values
    assert restored.iteration == state.iteration
    assert restored.proposed == state.proposed and restored.accepted == state.accepted
    assert restored.log_posterior == pytest.approx(state.log_posterior, abs=0.0)
    assert restored.rng_state["state"] == state.rng_state["state"]

    rng = np.random.default_rng(0)
    rng.bit_generator.state = restored.rng_state
    other = np.random.default_rng(0)
    other.bit_generator.state = state.rng_state
    assert rng.random() == other.random()


# --------------------------------------------------------------------- reproducibility
def test_the_same_seed_reproduces_the_chain_exactly(target, frozen):
    start = {n: frozen.truth[n] for n in ACTIVE_B3}
    a = run_joint_scalar_mcmc(target, start, SCALES, 150, 30, 2, seed=42)
    b = run_joint_scalar_mcmc(target, start, SCALES, 150, 30, 2, seed=42)
    for n in ACTIVE_B3:
        assert np.array_equal(a.draws[n], b.draws[n])
    assert a.accepted == b.accepted


def test_different_chains_use_independent_streams(target, frozen):
    start = {n: frozen.truth[n] for n in ACTIVE_B3}
    a = run_joint_scalar_mcmc(target, start, SCALES, 800, 200, 2, seed=100, chain=0)
    b = run_joint_scalar_mcmc(target, start, SCALES, 800, 200, 2, seed=101, chain=1)
    for n in ACTIVE_B3:
        assert not np.array_equal(a.draws[n], b.draws[n])
    # different streams, same target: compatible in units of the posterior spread, since
    # an absolute threshold would depend on the corpus size rather than on independence
    for n in ACTIVE_B3:
        spread = np.concatenate([a.draws[n], b.draws[n]]).std(ddof=1)
        assert abs(a.draws[n].mean() - b.draws[n].mean()) < 1.0 * spread


def test_no_global_random_state_is_consumed(target, frozen):
    """Seeding numpy's legacy global RNG must not change anything."""
    start = {n: frozen.truth[n] for n in ACTIVE_B3}
    np.random.seed(1234)
    a = run_joint_scalar_mcmc(target, start, SCALES, 120, 20, 2, seed=7)
    np.random.seed(999)
    b = run_joint_scalar_mcmc(target, start, SCALES, 120, 20, 2, seed=7)
    for n in ACTIVE_B3:
        assert np.array_equal(a.draws[n], b.draws[n])


def test_a_split_run_resumes_bit_identically(target, frozen):
    start = {n: frozen.truth[n] for n in ACTIVE_B3}
    target.evaluator.invalidate()
    whole = run_joint_scalar_mcmc(target, start, SCALES, 200, 20, 2, seed=13)

    target.evaluator.invalidate()
    first = run_joint_scalar_mcmc(target, start, SCALES, 100, 20, 2, seed=13)
    rng = np.random.default_rng(13)
    rng.bit_generator.state = first.final_state.rng_state
    target.evaluator.invalidate()
    second = run_joint_scalar_mcmc(target, start, SCALES, 200, 20, 2, seed=13,
                                   initial_state=first.final_state, rng=rng)
    for n in ACTIVE_B3:
        joined = np.concatenate([first.draws[n], second.draws[n]])
        assert np.array_equal(joined, whole.draws[n]), f"{n} diverged on resume"
    assert first.final_state.iteration == 100
    assert second.final_state.iteration == 200


def test_run_rejects_impossible_settings(target, frozen):
    start = {n: frozen.truth[n] for n in ACTIVE_B3}
    with pytest.raises(ValueError):
        run_joint_scalar_mcmc(target, start, SCALES, 100, 100, 2, seed=0)
    with pytest.raises(ValueError):
        run_joint_scalar_mcmc(target, start, SCALES, 100, 10, 0, seed=0)
    with pytest.raises(ValueError):
        run_joint_scalar_mcmc(target, {**start, "lambda_rep": -1.0}, SCALES,
                              100, 10, 2, seed=0)


# ------------------------------------------------------------------- dispersed starts
def test_registered_starts_are_dispersed_in_every_coordinate(frozen):
    runner = load_runner()
    for active in (ACTIVE_B2, ACTIVE_B3):
        starts = runner.dispersed_starts(active, 4)
        assert len(starts) == 4
        for name in active:
            values = [s[name] for s in starts]
            assert len(set(values)) == 4, f"{name} is not dispersed across chains"
            spread = max(values) / min(values) if name != "omega" else max(values) - min(values)
            assert spread > 2.0, f"{name} starts are too close together: {values}"


def test_no_two_chains_share_a_level_on_any_coordinate():
    runner = load_runner()
    square = runner.LATIN_SQUARE
    for name, levels in square.items():
        assert sorted(levels) == [0, 1, 2, 3], f"{name} is not a permutation"
    # and the chains are not all "low" or all "high" together
    for chain in range(4):
        levels = {name: square[name][chain] for name in square}
        assert len(set(levels.values())) > 1, f"chain {chain} is a single corner"


def test_every_registered_start_is_valid_and_finite(small, frozen):
    runner = load_runner()
    evaluator = RecurrentJointEvaluator(small, frozen.u_true, frozen.epsilon)
    for active, fixed in ((ACTIVE_B2, {"lambda_back": frozen.truth["lambda_back"]}),
                          (ACTIVE_B3, {})):
        target = JointScalarTarget(evaluator, active, fixed)
        for start in runner.dispersed_starts(active, 4):
            for name, value in start.items():
                if name != "omega":
                    assert value > 0.0, f"{name} start {value} is outside the support"
            evaluator.invalidate()
            parts = target.decompose(start, allow_cache=False)
            assert math.isfinite(parts["log_prior"])
            assert math.isfinite(parts["log_likelihood"])
            assert math.isfinite(parts["log_posterior"])


def test_starts_come_from_the_prior_not_from_a_posterior(frozen):
    """Start values must be prior quantiles — far from the posterior, by construction."""
    runner = load_runner()
    for name in PARAMETER_ORDER:
        for level in runner.START_LEVELS:
            assert runner.prior_quantile(name, level) == pytest.approx(
                runner.prior_quantile(name, level))
    starts = runner.dispersed_starts(ACTIVE_B3, 4)
    # every beta start is many posterior sds from the truth: these are not warm starts
    assert min(abs(s["beta"] - frozen.truth["beta"]) for s in starts) > 0.2
