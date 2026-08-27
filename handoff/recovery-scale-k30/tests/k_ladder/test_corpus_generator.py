"""Corpus generation, and the two preregistration criteria that cannot be met as written.

Both failures were found by running the generator, not by reading the prompt, and both
would have stopped the study on the target machine after the machine had been booked. They
are pinned here so a later change cannot quietly reintroduce either, and so the diagnosis
travels with the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_cpa import corpus as C                                  # noqa: E402
from hpop.mcmc_cpa.nested_library import draw_master_library           # noqa: E402
from hpop.mcmc_original.recurrent_rfs import RecurrentRFSParameters    # noqa: E402
from hpop.mcmc_original.recurrent_scalar_posterior import TRUE_VALUES  # noqa: E402

PARAMS = RecurrentRFSParameters(
    beta=float(TRUE_VALUES["beta"]), epsilon=0.02,
    shared_omega=float(TRUE_VALUES["omega"]),
    lambda_rep=float(TRUE_VALUES["lambda_rep"]),
    lambda_back=float(TRUE_VALUES["lambda_back"]))


@pytest.fixture(scope="module")
def library():
    return draw_master_library(0)[0]


# ------------------------------------------------ pi = nu(P): stationary from segment 1
@pytest.mark.parametrize("K", (3, 5, 10, 20, 30))
def test_pi_equals_the_stationary_law(K):
    """An unconstrained pi leaves a K-dependent per-skill imbalance the ladder cannot carry.

    Each of the 5K traces contributes a fresh first segment, so skill k gets about
    5*K*pi_k instances from first segments alone -- a term linear in K, against a balanced
    total of about 5*E[L] ~ 71 that does not grow with K at all. Setting pi = nu(P) makes
    every segment index have marginal nu with no reliance on mixing.
    """
    pi, transition = C.draw_pi_p(K, 6_510_000 + 100 * K)
    stationary = C.stationary_of(transition, K)
    assert np.allclose(pi, stationary), "pi must be the stationary law of P"
    low, high = 0.5 / K, 1.5 / K
    assert np.all((pi >= low) & (pi <= high)), "stationary band violated"
    assert np.array_equal(np.diag(transition), np.zeros(K))
    # stationary means pi P == pi
    assert np.allclose(pi @ transition, pi, atol=1e-10)


def test_the_pi_band_that_was_removed_was_genuinely_unreachable():
    """Kept so the reason for the change stays checkable, not as a live criterion."""
    for K, ceiling in ((10, 1e-3), (20, 1e-6)):
        rate = C.band_acceptance_rate(K, trials=300, seed=2)
        assert rate["pi_band_rate"] < ceiling
        assert rate["stationary_band_rate"] > 0.3, "the stationary band is reachable"


# ------------------------------------------- no realised-count rejection, anywhere
@pytest.mark.parametrize("K", (3, 10, 30))
def test_the_corpus_is_generated_exactly_once(library, K):
    """Rejecting on realised counts samples p(D | theta, A) while the scorer uses
    p(D | theta), with no -log P_theta(A) term. That is a likelihood mismatch."""
    corpus = C.generate_ladder_corpus(library, K, 0)
    assert corpus.attempts == [{"generated_once": True}]
    assert len(corpus.train) == 5 * K
    assert len(corpus.heldout) == 2 * K


def test_coverage_is_recorded_but_never_enforced(library):
    corpus = C.generate_ladder_corpus(library, 30, 0)
    coverage = corpus.coverage
    for key in ("train_instances", "train_occurrences", "train_per_role_min",
                "train_roles_never_seen", "heldout_instances", "heldout_occurrences"):
        assert key in coverage, key
    assert "bands_met_AS_A_DIAGNOSTIC_ONLY" in coverage
    assert "unmet_conditions" in coverage
    assert "not enforced" in coverage["NOTE"]
    # a corpus that misses the reference bands is still returned, not rejected
    assert coverage["bands_met_AS_A_DIAGNOSTIC_ONLY"] in (True, False)


def test_no_acceptance_loop_survives_in_the_source():
    """A structural guard: the generator must not regain a rejection loop by edit."""
    source = (ROOT / "src" / "hpop" / "mcmc_cpa" / "corpus.py").read_text()
    body = source[source.index("def generate_ladder_corpus("):]
    assert "for attempt in range" not in body, "an acceptance loop reappeared"
    assert "max_attempts" not in body.split('"""')[2], "attempt cap back in the body"


# ------------------------------------- role exposure: two quantities, with uncertainty
def test_role_exposure_reports_count_and_probability_with_error(library):
    u, _ = library.prefix(3)
    report = C.role_exposure(u, PARAMS, samples=150, seed=1)
    counts = np.asarray(report["expected_count_per_segment"])
    probability = np.asarray(report["probability_at_least_once_per_segment"])
    assert counts.shape == probability.shape == (3, 10)
    assert np.all(probability <= 1.0) and np.all(probability >= 0.0)
    # a count can exceed 1 (repeats); a probability cannot
    assert counts.max() > probability.max()
    assert np.all(np.asarray(report["expected_count_mc_stderr"]) >= 0)
    assert report["q0_reset_each_segment"] is True
    assert "never as a recovery ceiling" in report["note"]


def test_first_step_probability_is_provably_useless_as_a_criterion(library):
    """The impossibility that killed the withdrawn Amendment B.

    A role with any predecessor has feasibility F = 0 at q = 0, so its first-step
    probability is EXACTLY epsilon/m. Requiring every role above any multiple of that
    floor forces an antichain, which criterion 3 forbids.
    """
    from hpop.mcmc_original.latent_poset import precedence_from_u
    from hpop.mcmc_original.recurrent_rfs import (recurrent_feasibility,
                                                  recurrent_step_probabilities)

    u, _ = library.prefix(30)
    floor = PARAMS.epsilon / u.shape[1]
    checked = 0
    for k in range(u.shape[0]):
        precedence = np.asarray(precedence_from_u(u[k]))
        strict = precedence.copy()
        np.fill_diagonal(strict, False)
        assert strict.sum() >= 1
        feasibility = recurrent_feasibility(precedence, np.zeros(u.shape[1]))
        probabilities = recurrent_step_probabilities(u[k], np.zeros(u.shape[1]), PARAMS)
        for role in np.flatnonzero(strict.sum(axis=0) > 0):
            assert feasibility[role] == 0.0
            assert probabilities[role] == pytest.approx(floor, rel=1e-12)
            checked += 1
    assert checked > 0


# ------------------------------------------- off-frontier contamination, characterised
def test_off_frontier_events_are_not_in_almost_every_trace(library):
    """Mean 1.0 per trace does not mean most traces have one. Measured: 41% have none."""
    from hpop.mcmc_original.latent_poset import precedence_from_u
    from hpop.mcmc_original.recurrent_rfs import (recurrent_feasibility,
                                                  sample_recurrent_rfs_sequence)
    from hpop.mcmc_original.matched_segmentation_prior import (sample_segmentation_widths,
                                                               width_sampling_tables)

    u, _ = library.prefix(10)
    tables = width_sampling_tables(96, 0.15, 3, 12)
    rng = np.random.default_rng(11)
    kappa = 1.0 / (1.0 + np.exp(-PARAMS.shared_omega))
    per_trace = []
    for _ in range(150):
        events = 0
        for width in sample_segmentation_widths(rng, 96, 0.15, 3, 12, tables):
            k = int(rng.integers(10))
            precedence = np.asarray(precedence_from_u(u[k]))
            q = np.zeros(10)
            for role in sample_recurrent_rfs_sequence(rng, width, u[k], PARAMS):
                role = int(role)
                if recurrent_feasibility(precedence, q)[role] == 0.0:
                    events += 1
                gate = np.where(precedence[role], kappa, 0.0)
                q = q * (1.0 - gate)
                q[role] = 1.0
        per_trace.append(events)
    per_trace = np.asarray(per_trace)
    assert per_trace.mean() < 3.0
    assert (per_trace == 0).mean() > 0.2, (
        "a substantial share of traces must be free of off-frontier events")


# --------------------------------------------------- the emission path is correct
def test_every_emitted_cpa_lies_in_its_skills_support(library):
    from hpop.mcmc_original.matched_segmentation_prior import width_sampling_tables

    u, maps = library.prefix(5)
    pi, transition = C.draw_pi_p(3, 6_510_300)
    pi = np.full(5, 0.2)
    transition = (np.ones((5, 5)) - np.eye(5)) / 4.0
    tables = width_sampling_tables(96, 0.15, 3, 12)
    traces = [C._emit_trace(4242, "train", i, 96, u, maps, pi, transition,
                            PARAMS, tables, 0.15, 3, 12) for i in range(6)]
    for trace in traces:
        assert trace.length == 96
        assert sum(trace.widths) == 96
        cursor = 0
        for width, skill in zip(trace.widths, trace.labels):
            support = set(maps.forward[skill].tolist())
            block = trace.cpa[cursor:cursor + width]
            assert set(block) <= support, (
                f"skill {skill} emitted a CPA outside its own support")
            cursor += width
        # the path never repeats a skill across a boundary
        assert all(a != b for a, b in zip(trace.labels[:-1], trace.labels[1:]))


def test_digests_separate_observed_data_from_sealed_truth(library):
    from hpop.mcmc_original.matched_segmentation_prior import width_sampling_tables

    u, maps = library.prefix(3)
    pi = np.full(3, 1 / 3)
    transition = (np.ones((3, 3)) - np.eye(3)) / 2.0
    tables = width_sampling_tables(96, 0.15, 3, 12)
    traces = [C._emit_trace(11, "train", i, 96, u, maps, pi, transition,
                            PARAMS, tables, 0.15, 3, 12) for i in range(3)]
    corpus = C.LadderCorpus(3, 0, traces, traces, maps, pi, transition, {}, {}, [])
    observed, truth = corpus.observed_digest(), corpus.truth_digest()
    assert observed != truth
    manifest = corpus.manifest()
    # a manifest may expose hashes and verdicts, never latent values
    text = str(manifest)
    assert "labels" not in text and "widths" not in text
    assert manifest["observed_sha256"] == observed
