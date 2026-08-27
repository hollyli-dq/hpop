"""Stage 6B1, step 2 — the proposal ratios, verified against explicit densities.

The samplers use a one-line correction, ``log theta' - log theta``, in place of the two
lognormal densities it stands for. These tests evaluate both densities in full and check
the identity holds, rather than trusting the algebra, and check that what the proposal
*samples* is the law whose density is being differenced.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from hpop.mcmc_original.recurrent_scalar_mcmc import (
    PROPOSAL_KIND, build_proposal, gaussian_random_walk, log_scale_random_walk,
    lognormal_log_density, normal_log_density,
)

POSITIVE_PARAMETERS = ("beta", "lambda_rep", "lambda_back")


@pytest.mark.parametrize("scale", [0.05, 0.3, 1.2])
@pytest.mark.parametrize("current,proposed", [(1.5, 1.7), (0.25, 0.05), (0.8, 3.4), (2.0, 2.0)])
def test_lognormal_ratio_equals_the_log_difference(scale, current, proposed):
    explicit = (lognormal_log_density(current, proposed, scale)
                - lognormal_log_density(proposed, current, scale))
    assert explicit == pytest.approx(math.log(proposed) - math.log(current), abs=1e-12)


@pytest.mark.parametrize("parameter", POSITIVE_PARAMETERS)
def test_positive_parameters_use_a_log_scale_walk_and_carry_the_jacobian(parameter):
    assert PROPOSAL_KIND[parameter] == "log"
    propose = build_proposal(parameter, 0.4)
    rng = np.random.default_rng(0)
    for current in (0.05, 0.8, 1.5, 6.0):
        for _ in range(50):
            proposal = propose(current, rng)
            assert proposal.value > 0.0
            expected = (lognormal_log_density(current, proposal.value, 0.4)
                        - lognormal_log_density(proposal.value, current, 0.4))
            assert proposal.log_q_reverse_minus_forward == pytest.approx(expected, abs=1e-12)


def test_omega_uses_a_symmetric_walk_with_no_correction():
    assert PROPOSAL_KIND["omega"] == "identity"
    propose = build_proposal("omega", 0.3)
    rng = np.random.default_rng(1)
    for current in (-2.0, 0.0, 1.7346, 5.0):
        for _ in range(50):
            proposal = propose(current, rng)
            explicit = (normal_log_density(current, proposal.value, 0.3)
                        - normal_log_density(proposal.value, current, 0.3))
            assert explicit == pytest.approx(0.0, abs=1e-12)
            assert proposal.log_q_reverse_minus_forward == 0.0


def test_log_scale_walk_refuses_a_non_positive_state():
    """No ``max(value, 0.01)`` floor: the state is validated instead of quietly clipped."""
    propose = log_scale_random_walk(0.3)
    rng = np.random.default_rng(0)
    for bad in (0.0, -1e-9, -2.0):
        with pytest.raises(ValueError):
            propose(bad, rng)


def test_sampled_log_scale_proposals_follow_the_claimed_lognormal_law():
    scale, current, n = 0.35, 1.5, 40_000
    propose = log_scale_random_walk(scale)
    rng = np.random.default_rng(4)
    draws = np.array([propose(current, rng).value for _ in range(n)])
    reference = stats.lognorm(s=scale, scale=current)
    assert stats.kstest(draws, reference.cdf).statistic < 0.01
    assert np.log(draws).mean() == pytest.approx(math.log(current), abs=4.0 * scale / math.sqrt(n))


def test_sampled_gaussian_proposals_follow_the_claimed_normal_law():
    scale, current, n = 0.4, 1.7346, 40_000
    propose = gaussian_random_walk(scale)
    rng = np.random.default_rng(5)
    draws = np.array([propose(current, rng).value for _ in range(n)])
    assert stats.kstest(draws, stats.norm(current, scale).cdf).statistic < 0.01


def test_the_reverse_move_is_reachable_and_the_ratio_negates():
    """``log q(a|b) - log q(b|a)`` must be the exact negative of the reverse direction."""
    rng = np.random.default_rng(9)
    propose = log_scale_random_walk(0.5)
    for _ in range(200):
        a = float(rng.lognormal(0.0, 1.0))
        forward = propose(a, rng)
        b = forward.value
        back = (lognormal_log_density(b, a, 0.5) - lognormal_log_density(a, b, 0.5))
        assert forward.log_q_reverse_minus_forward == pytest.approx(-back, abs=1e-12)
