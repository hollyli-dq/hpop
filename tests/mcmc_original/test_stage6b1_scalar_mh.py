"""Stage 6B1, step 1 — the generic scalar MH engine, on targets whose answer is known.

Nothing in this file touches the recurrent model. That is the point: if the engine is
wrong, it must fail here, on a Normal and a Gamma, before the recurrent likelihood is
ever involved. Otherwise a sampler bug and a model bug are indistinguishable later.

The last test is the one that gives the rest teeth. On a log-scale random walk the MH
ratio carries ``log theta' - log theta``. Removing it does not merely slow the chain
down: it silently changes the stationary distribution from ``Gamma(a, b)`` to
``Gamma(a - 1, b)``. The test asserts the broken sampler lands on that *specific* wrong
distribution, so a passing correct sampler cannot be an accident of a loose tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from hpop.mcmc_original.recurrent_scalar_mcmc import (
    Proposal, ScalarMHConfig, gaussian_random_walk, log_scale_random_walk,
    run_scalar_mh, scalar_mh_step,
)


def normal_target(mean: float, sd: float):
    return lambda x: -0.5 * ((x - mean) / sd) ** 2


def gamma_target(shape: float, rate: float):
    def log_density(x: float) -> float:
        if x <= 0:
            return -math.inf
        return (shape - 1.0) * math.log(x) - rate * x
    return log_density


def ks_statistic(samples, cdf) -> float:
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    f = cdf(x)
    return float(max((np.arange(1, n + 1) / n - f).max(), (f - np.arange(n) / n).max()))


# ------------------------------------------------------------------ engine mechanics
def test_burn_in_and_thinning_shape_the_output():
    config = ScalarMHConfig("x", 1.0, 0.0, 1000, 200, 4, 0)
    result = run_scalar_mh(config, normal_target(0.0, 1.0), gaussian_random_walk(1.0))
    assert result.samples.shape == (len(range(200, 1000, 4)),)
    assert result.proposed == 1000
    assert 0.0 < result.acceptance_rate < 1.0
    assert result.runtime_seconds >= 0.0


def test_same_seed_reproduces_the_chain_exactly():
    config = ScalarMHConfig("x", 0.8, 0.5, 500, 100, 1, 7)
    a = run_scalar_mh(config, normal_target(0.0, 1.0), gaussian_random_walk(0.8))
    b = run_scalar_mh(config, normal_target(0.0, 1.0), gaussian_random_walk(0.8))
    assert np.array_equal(a.samples, b.samples)


def test_rejection_costs_exactly_one_evaluation():
    """The current value's log posterior is cached, so a rejected step evaluates once."""
    calls = {"n": 0}

    def counted(x):
        calls["n"] += 1
        return normal_target(0.0, 1.0)(x)

    config = ScalarMHConfig("x", 3.0, 0.0, 400, 0, 1, 3)
    result = run_scalar_mh(config, counted, gaussian_random_walk(3.0))
    assert result.acceptance_rate < 0.5, "scale chosen so that most steps reject"
    assert calls["n"] == config.num_iterations + 1   # one per iteration, plus the start


def test_invalid_configurations_are_rejected():
    with pytest.raises(ValueError):
        ScalarMHConfig("x", 1.0, 0.0, 100, 100, 1, 0)
    with pytest.raises(ValueError):
        ScalarMHConfig("x", 1.0, 0.0, 100, 10, 0, 0)
    with pytest.raises(ValueError):
        gaussian_random_walk(0.0)
    with pytest.raises(ValueError):
        log_scale_random_walk(-1.0)
    with pytest.raises(ValueError):
        run_scalar_mh(ScalarMHConfig("x", 1.0, -1.0, 100, 10, 1, 0),
                      gamma_target(2.0, 2.0), log_scale_random_walk(0.5))


def test_step_returns_the_current_state_when_the_proposal_is_impossible():
    rng = np.random.default_rng(0)
    reject_everything = lambda current, r: Proposal(-1.0, 0.0)   # noqa: E731
    value, lp, accepted = scalar_mh_step(
        1.0, gamma_target(2.0, 2.0)(1.0), gamma_target(2.0, 2.0), reject_everything, rng)
    assert not accepted and value == 1.0


# ------------------------------------------------------------------- known targets
def test_gaussian_random_walk_recovers_a_normal_target():
    mean, sd = 2.0, 0.5
    config = ScalarMHConfig("x", 2.4 * sd, 0.0, 60_000, 5_000, 5, 11)
    result = run_scalar_mh(config, normal_target(mean, sd), gaussian_random_walk(2.4 * sd))
    draws = result.samples
    assert 0.15 < result.acceptance_rate < 0.75
    assert abs(draws.mean() - mean) < 0.02 * sd * 3
    assert abs(draws.std(ddof=1) / sd - 1.0) < 0.05
    assert ks_statistic(draws, stats.norm(mean, sd).cdf) < 0.02


def test_log_scale_walk_with_the_jacobian_recovers_a_gamma_target():
    shape, rate = 2.0, 2.0
    scale = 0.9
    config = ScalarMHConfig("theta", scale, 1.0, 60_000, 5_000, 5, 13)
    result = run_scalar_mh(config, gamma_target(shape, rate), log_scale_random_walk(scale))
    draws = result.samples
    reference = stats.gamma(a=shape, scale=1.0 / rate)
    assert (draws > 0).all()
    assert 0.15 < result.acceptance_rate < 0.75
    assert abs(draws.mean() - reference.mean()) < 0.03 * reference.std()  * 3
    assert abs(draws.std(ddof=1) / reference.std() - 1.0) < 0.05
    assert ks_statistic(draws, reference.cdf) < 0.02


def test_dropping_the_log_jacobian_shifts_the_stationary_distribution():
    """Negative control: the broken sampler targets Gamma(a-1, b), not Gamma(a, b).

    With ``a = 2`` the wrong stationary law is Exponential(rate 2): mean 0.5 rather than
    1.0. Both directions are asserted — far from the right answer, and close to the
    specific wrong one — so this cannot pass by accident.
    """
    shape, rate, scale = 2.0, 2.0, 0.9

    def uncorrected(current, rng):
        value = float(current * math.exp(scale * rng.normal()))
        return Proposal(value, 0.0)          # the log-Jacobian is deliberately omitted

    config = ScalarMHConfig("theta", scale, 1.0, 60_000, 5_000, 5, 13)
    broken = run_scalar_mh(config, gamma_target(shape, rate), uncorrected).samples
    correct = run_scalar_mh(config, gamma_target(shape, rate),
                            log_scale_random_walk(scale)).samples

    right = stats.gamma(a=shape, scale=1.0 / rate)
    wrong = stats.gamma(a=shape - 1.0, scale=1.0 / rate)

    assert ks_statistic(correct, right.cdf) < 0.02
    assert ks_statistic(broken, right.cdf) > 0.20
    assert ks_statistic(broken, wrong.cdf) < 0.02
    assert abs(broken.mean() - wrong.mean()) < 0.05
    assert broken.mean() < 0.75 * right.mean()
