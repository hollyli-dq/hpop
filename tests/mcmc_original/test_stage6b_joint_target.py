"""Stage 6B2 / 6B3 — the frozen model, and the joint log posterior.

The joint target must decompose exactly into the complete recurrent log likelihood plus
the ACTIVE scalar log priors, and each coordinate's acceptance ratio must be numerically
identical to the Stage 6B1 ratio when the other coordinates are held at the same values.
That identity is what licenses reusing the Stage 6B1 validation for the joint stages, so
it is tested rather than argued.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hpop.mcmc_original.recurrent_joint_scalar_mcmc import (
    ACTIVE_B2, ACTIVE_B3, JointScalarTarget, RecurrentJointEvaluator,
)
from hpop.mcmc_original.recurrent_scalar_mcmc import RecurrentScalarTarget, build_proposal
from hpop.mcmc_original.recurrent_scalar_posterior import log_prior
from hpop.mcmc_original.stage6b_frozen import (
    PARAMETER_ORDER, STAGE6B_MODEL_ID, SWEEP_ORDER, TRANSFORM, assert_likelihood_branch,
    config_hash, frozen_config, from_unconstrained, load_frozen_dataset,
    log_jacobian_to_unconstrained, to_unconstrained,
)

# The Stage 6B1 log-density parity tolerance, reused unchanged.
PARITY_TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def frozen():
    return load_frozen_dataset()


@pytest.fixture(scope="module")
def evaluator(frozen):
    return RecurrentJointEvaluator(frozen.train, frozen.u_true, frozen.epsilon)


# ------------------------------------------------------------------------- the freeze
def test_the_frozen_model_is_the_utility_weighted_frontier():
    report = assert_likelihood_branch()
    assert report["model_id"] == STAGE6B_MODEL_ID
    assert report["feasibility_q0"] == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert report["stale_counts_q0"] == [3, 0, 2, 1, 0]
    # beta genuinely reweights the frontier; a uniform-frontier branch would not
    assert report["beta_weight_ratio_high"] > 1.5 * report["beta_weight_ratio_low"]


def test_config_hash_is_stable_and_covers_the_registered_quantities():
    first, second = config_hash(), config_hash()
    assert first == second and len(first) == 64
    config = frozen_config()
    for key in ("model_id", "likelihood_branch", "dataset", "u_true", "epsilon", "truth",
                "priors", "transform", "boundaries", "numerics", "seeds"):
        assert key in config
    assert config["epsilon"] == 0.02
    assert config["dataset"] == {"mode": "full", "generator_seed": 0, "n_train": 500,
                                 "n_heldout": 200, "T": 20}


def test_n_roles_is_not_the_latent_dimension(frozen):
    """Five roles embedded in two dimensions — the two must never be conflated."""
    assert frozen.n_roles == 5
    assert frozen.latent_dimension == 2
    assert frozen.n_roles != frozen.latent_dimension
    assert frozen.u_true.shape == (5, 2)
    assert frozen_config()["n_roles"] == 5
    assert frozen_config()["latent_dimension"] == 2


def test_stage_6b_never_infers_the_structural_objects():
    never = frozen_config()["never_inferred_in_stage_6b"]
    for name in ("U", "rho", "P", "segmentation_boundaries", "skill_labels", "epsilon"):
        assert name in never
    assert set(frozen_config()["inferred_in_stage_6b"]) == set(PARAMETER_ORDER)


def test_the_frozen_corpus_matches_the_stage_6b1_dataset(frozen):
    assert frozen.train.shape == (500, 20)
    assert frozen.heldout.shape == (200, 20)
    assert frozen.epsilon == 0.02
    assert frozen.truth["lambda_back"] == 0.25
    assert frozen.truth["omega"] == pytest.approx(math.log(0.85 / 0.15))


def test_sweep_order_is_the_registered_one():
    assert SWEEP_ORDER == ("beta", "omega", "lambda_rep", "lambda_back")


# ------------------------------------------------------------------- coordinate transforms
def test_transforms_round_trip_and_carry_the_right_jacobian():
    for name in PARAMETER_ORDER:
        for value in (0.05, 0.8, 1.5, 3.0):
            if TRANSFORM[name] == "identity" and name != "omega":
                continue
            z = to_unconstrained(name, value)
            assert from_unconstrained(name, z) == pytest.approx(value, rel=1e-12)
            if TRANSFORM[name] == "log":
                # d value / d z = value, so log|J| = log value
                assert log_jacobian_to_unconstrained(name, value) == pytest.approx(
                    math.log(value), abs=1e-12)
            else:
                assert log_jacobian_to_unconstrained(name, value) == 0.0


def test_a_density_in_lambda_is_not_a_density_in_log_lambda():
    """The Jacobian is what separates them; a numerical check, not an assertion."""
    from scipy import integrate
    shape, rate = 2.0, 2.0
    density_lambda = lambda x: x ** (shape - 1) * math.exp(-rate * x)      # noqa: E731
    # same measure expressed in z = log lambda must integrate to the same total
    in_lambda, _ = integrate.quad(density_lambda, 1e-9, 60.0)
    in_z, _ = integrate.quad(
        lambda z: density_lambda(math.exp(z)) * math.exp(z), -30.0, 6.0)
    assert in_z == pytest.approx(in_lambda, rel=1e-8)
    # and dropping the Jacobian changes the answer materially
    without, _ = integrate.quad(lambda z: density_lambda(math.exp(z)), -30.0, 6.0)
    assert abs(without - in_lambda) / in_lambda > 0.2


# ------------------------------------------------------------------- joint decomposition
@pytest.mark.parametrize("active", [ACTIVE_B2, ACTIVE_B3])
def test_joint_log_posterior_is_likelihood_plus_active_priors(active, evaluator, frozen):
    target = JointScalarTarget(evaluator, active,
                               {"lambda_back": frozen.truth["lambda_back"]})
    values = {n: frozen.truth[n] * 1.05 for n in active}
    if "omega" in values:
        values["omega"] = frozen.truth["omega"] + 0.1
    parts = target.decompose(values, allow_cache=False)
    expected_prior = sum(log_prior(n, values[n]) for n in active)
    assert parts["log_prior"] == pytest.approx(expected_prior, abs=1e-12)
    assert parts["log_posterior"] == pytest.approx(
        parts["log_likelihood"] + expected_prior, abs=1e-12)
    assert set(parts["log_prior_components"]) == set(active)


def test_the_b2_target_excludes_the_fixed_lambda_back_prior(evaluator, frozen):
    """Stage 6B2 is not Stage 6B3 with a coordinate pinned: the prior term is absent."""
    b2 = JointScalarTarget(evaluator, ACTIVE_B2,
                           {"lambda_back": frozen.truth["lambda_back"]})
    b3 = JointScalarTarget(evaluator, ACTIVE_B3, {})
    values = {n: frozen.truth[n] for n in ACTIVE_B2}
    all_values = dict(values); all_values["lambda_back"] = frozen.truth["lambda_back"]
    assert "lambda_back" not in b2.log_priors(values)
    assert "lambda_back" in b3.log_priors(all_values)
    difference = b3.log_posterior(all_values, allow_cache=False) \
        - b2.log_posterior(values, allow_cache=False)
    assert difference == pytest.approx(
        log_prior("lambda_back", frozen.truth["lambda_back"]), abs=1e-9)


def test_the_target_is_callable_outside_the_mcmc_code(evaluator, frozen):
    target = JointScalarTarget(evaluator, ACTIVE_B3, {})
    value = target.log_posterior({n: frozen.truth[n] for n in ACTIVE_B3},
                                 allow_cache=False)
    assert math.isfinite(value)


def test_inactive_coordinates_must_be_given_fixed_values(evaluator):
    with pytest.raises(ValueError):
        JointScalarTarget(evaluator, ACTIVE_B2, {})
    with pytest.raises(ValueError):
        JointScalarTarget(evaluator, ("beta", "rho"), {})


def test_out_of_support_values_short_circuit_to_minus_infinity(evaluator, frozen):
    target = JointScalarTarget(evaluator, ACTIVE_B3, {})
    values = {n: frozen.truth[n] for n in ACTIVE_B3}
    values["lambda_rep"] = -0.5
    assert target.log_posterior(values, allow_cache=False) == -math.inf


# ---------------------------------------------------------- acceptance-ratio parity (6)
@pytest.mark.parametrize("active,name", [(ACTIVE_B2, "beta"), (ACTIVE_B2, "omega"),
                                         (ACTIVE_B2, "lambda_rep"), (ACTIVE_B3, "beta"),
                                         (ACTIVE_B3, "omega"), (ACTIVE_B3, "lambda_rep"),
                                         (ACTIVE_B3, "lambda_back")])
def test_coordinate_acceptance_ratio_matches_stage_6b1(active, name, evaluator, frozen):
    """With the other coordinates held equal, the joint ratio IS the Stage 6B1 ratio.

    The inactive priors are identical on both sides of the ratio and cancel, so the two
    differ only by floating-point regrouping.
    """
    joint = JointScalarTarget(evaluator, active,
                              {"lambda_back": frozen.truth["lambda_back"]})
    scalar = RecurrentScalarTarget(name, frozen.train, frozen.u_true, frozen.truth,
                                   frozen.epsilon)
    rng = np.random.default_rng(20250812)
    worst = 0.0
    for _ in range(10):
        if name == "omega":
            current = frozen.truth[name] + rng.normal(0.0, 0.4)
            proposed = frozen.truth[name] + rng.normal(0.0, 0.4)
        else:
            current = frozen.truth[name] * math.exp(rng.normal(0.0, 0.25))
            proposed = frozen.truth[name] * math.exp(rng.normal(0.0, 0.25))
        stage6b1_ratio = scalar(proposed)[0] - scalar(current)[0]

        base = {n: frozen.truth[n] for n in active}
        a = dict(base); a[name] = proposed
        b = dict(base); b[name] = current
        evaluator.invalidate()
        joint_ratio = (joint.log_posterior(a, allow_cache=False)
                       - joint.log_posterior(b, allow_cache=False))
        worst = max(worst, abs(stage6b1_ratio - joint_ratio))
    assert worst < PARITY_TOLERANCE, f"{name}: worst ratio difference {worst:.3e}"


class _FixedUniform:
    """A stand-in generator returning a chosen uniform, so both MH branches are forced.

    `scalar_mh_step` consumes randomness only through ``rng.random()``; fixing it makes
    the accept and reject paths deterministic instead of hoping a random probe hits both.
    """

    def __init__(self, value: float):
        self._value = value

    def random(self) -> float:
        return self._value


@pytest.mark.parametrize("name", PARAMETER_ORDER)
def test_log_acceptance_ratio_equals_posterior_difference_plus_hastings(name, evaluator,
                                                                       frozen):
    """The identity the kernel relies on, checked on both an accepted and a rejected move."""
    from hpop.mcmc_original.recurrent_scalar_mcmc import scalar_mh_step

    target = JointScalarTarget(evaluator, ACTIVE_B3, {})
    values = {n: frozen.truth[n] for n in ACTIVE_B3}
    evaluator.invalidate()
    current_lp = target.log_posterior(values, allow_cache=False)

    def log_posterior_fn(candidate):
        trial = dict(values); trial[name] = candidate
        evaluator.invalidate()
        return target.log_posterior(trial, allow_cache=False)

    near = build_proposal(name, 0.002)(values[name], np.random.default_rng(3))
    far = build_proposal(name, 2.5)(values[name], np.random.default_rng(5))

    # 1e-300 rather than 0.0: log(0) would emit a divide-by-zero warning from the kernel
    for proposal, uniform, should_accept in ((near, 1e-300, True),
                                             (far, 1.0 - 1e-12, False)):
        proposed_lp = log_posterior_fn(proposal.value)
        expected = proposed_lp - current_lp + proposal.log_q_reverse_minus_forward

        new_value, new_lp, accepted = scalar_mh_step(
            values[name], current_lp, log_posterior_fn,
            lambda c, r, _p=proposal: _p, _FixedUniform(uniform))

        assert accepted is should_accept, (name, expected)
        if should_accept:
            assert new_value == proposal.value
            assert new_lp == pytest.approx(proposed_lp, abs=1e-9)
            # a forced accept must still be a genuine move, not a no-op
            assert new_value != values[name]
        else:
            assert new_value == values[name]
            assert new_lp == current_lp
            # rejection was earned: the far proposal is genuinely much worse
            assert expected < -1.0
