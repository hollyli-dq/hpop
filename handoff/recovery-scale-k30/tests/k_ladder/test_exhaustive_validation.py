"""Exhaustive validation of the Monte-Carlo diagnostics, at sizes small enough to enumerate.

A Monte-Carlo diagnostic that is never checked against an exact answer is a number with no
warrant. At `m = 3` roles and width `w = 2` or `3` the role sequence space is `m^w` — 9 or
27 outcomes — so the exact expected count and exact probability of at least one occurrence
can be computed by summing over every path, and the estimator compared against them.

This also pins the two quantities apart. Expected count and probability-of-at-least-one are
different numbers whenever repeats are possible, and conflating them is easy.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_cpa import corpus as C                                  # noqa: E402
from hpop.mcmc_original.latent_poset import precedence_from_u          # noqa: E402
from hpop.mcmc_original.recurrent_rfs import (RecurrentRFSParameters,  # noqa: E402
                                              recurrent_step_probabilities)

PARAMS = RecurrentRFSParameters(beta=1.5, epsilon=0.02, shared_omega=1.7346010553881064,
                                lambda_rep=0.8, lambda_back=0.25)


def exact_sequence_distribution(u, width: int, params):
    """Every role sequence of the given width, with its exact probability.

    Replays the same recurrence the sampler uses -- `q` starts at zero, a gate multiplies
    `q` by `1 - kappa` on the successors of the observed role, and the observed role's `q`
    is set to one -- summing over all `m^w` paths instead of drawing one.
    """
    u = np.asarray(u, dtype=float)
    m = u.shape[0]
    precedence = np.asarray(precedence_from_u(u))
    kappa = 1.0 / (1.0 + np.exp(-params.shared_omega))

    out = {}
    for sequence in itertools.product(range(m), repeat=width):
        q = np.zeros(m)
        probability = 1.0
        for role in sequence:
            probability *= float(recurrent_step_probabilities(u, q, params)[role])
            gate = np.where(precedence[role], kappa, 0.0)
            q = q * (1.0 - gate)
            q[role] = 1.0
        out[sequence] = probability
    return out


def exact_moments(u, width: int, params):
    """Exact expected count per role, and exact P(role appears at least once)."""
    m = np.asarray(u).shape[0]
    distribution = exact_sequence_distribution(u, width, params)
    expected = np.zeros(m)
    at_least_once = np.zeros(m)
    for sequence, probability in distribution.items():
        counts = np.bincount(sequence, minlength=m)
        expected += probability * counts
        at_least_once += probability * (counts > 0)
    return expected, at_least_once, sum(distribution.values())


# ------------------------------------------------------------------ the enumeration
@pytest.mark.parametrize("width", (2, 3))
def test_the_enumeration_is_a_proper_distribution(width):
    u = np.random.default_rng(1).standard_normal((3, 2))
    _, _, total = exact_moments(u, width, PARAMS)
    assert total == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("width", (2, 3))
def test_expected_count_sums_to_the_width(width):
    """Every step emits exactly one role, so the counts must sum to the width."""
    u = np.random.default_rng(2).standard_normal((3, 2))
    expected, _, _ = exact_moments(u, width, PARAMS)
    assert expected.sum() == pytest.approx(float(width), abs=1e-12)


@pytest.mark.parametrize("width", (2, 3))
def test_probability_never_exceeds_expected_count(width):
    """P(at least one) <= E[count], with equality only when repeats are impossible."""
    u = np.random.default_rng(3).standard_normal((3, 2))
    expected, at_least_once, _ = exact_moments(u, width, PARAMS)
    assert np.all(at_least_once <= expected + 1e-12)
    assert np.all(at_least_once <= 1.0 + 1e-12)


# ------------------------------------------- the Monte-Carlo estimator against exact
@pytest.mark.parametrize("width", (2, 3))
def test_role_exposure_matches_the_exact_answer(width):
    """The estimator must land on the enumerated truth, within its own stated error."""
    u = np.random.default_rng(7).standard_normal((1, 3, 2))
    expected, at_least_once, _ = exact_moments(u[0], width, PARAMS)

    report = C.role_exposure(u, PARAMS, widths=(width, width), samples=20_000, seed=11)
    estimated = np.asarray(report["expected_count_per_segment"])[0]
    estimated_p = np.asarray(report["probability_at_least_once_per_segment"])[0]
    stderr = np.asarray(report["expected_count_mc_stderr"])[0]

    # within four standard errors of the exact value, per role
    for role in range(3):
        assert abs(estimated[role] - expected[role]) <= 4 * stderr[role] + 1e-9, (
            f"role {role}: estimate {estimated[role]:.4f} vs exact {expected[role]:.4f}")
    assert np.allclose(estimated_p, at_least_once, atol=0.02)
    # the estimator's own bookkeeping must be self-consistent
    assert estimated.sum() == pytest.approx(float(width), abs=0.02)


def test_the_two_quantities_are_genuinely_different():
    """If they coincided, reporting both would be pointless -- confirm they do not."""
    u = np.random.default_rng(5).standard_normal((1, 3, 2))
    report = C.role_exposure(u, PARAMS, widths=(6, 6), samples=4_000, seed=2)
    counts = np.asarray(report["expected_count_per_segment"])[0]
    probability = np.asarray(report["probability_at_least_once_per_segment"])[0]
    assert counts.max() > probability.max() + 0.2, (
        "at width 6 some role must repeat, so E[count] must exceed P(at least one)")


def test_widths_are_integrated_over_the_requested_range():
    """A range must not silently collapse to one width."""
    u = np.random.default_rng(9).standard_normal((1, 4, 2))
    narrow = C.role_exposure(u, PARAMS, widths=(3, 3), samples=3_000, seed=4)
    wide = C.role_exposure(u, PARAMS, widths=(3, 12), samples=3_000, seed=4)
    assert np.asarray(narrow["expected_count_per_segment"]).sum() == pytest.approx(3.0,
                                                                                   abs=.05)
    assert np.asarray(wide["expected_count_per_segment"]).sum() > 5.0
    assert wide["widths_sampled_uniformly_over"] == [3, 12]
