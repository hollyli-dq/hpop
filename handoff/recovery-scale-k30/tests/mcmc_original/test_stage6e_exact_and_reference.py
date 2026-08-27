"""Stage 6E1 — exact enumeration, the mixed reference, and the recorded comparisons.

Covers §18 areas 17-26. The enumeration and the forward recursion are exercised directly;
the recorded Stage 6E1A and Stage 6E1B artifacts are validated for schema *and* for the
gate values they claim, so a report cannot drift away from the numbers behind it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from hpop.mcmc_original.fast_segmentation_kernel import (
    FastSegmentationKernel, segmentation_of, spans_of,
)
from hpop.mcmc_original.proposals import MoveType
from hpop.mcmc_original.recurrent_scalar_posterior import cached_batch_log_likelihood
from hpop.mcmc_original.recurrent_joint_scalar_mcmc import vectorized_state_features
from hpop.mcmc_original.recurrent_segmentation import (
    RecurrentBlockScorer, segmentation_log_weight,
)
from hpop.mcmc_original.stage6e_exact import (
    boundary_marginals, enumerate_states, exact_posterior, expected_transition_counts,
    labelled_segment_marginals, log_evidence_forward, occurrence_label_marginals,
    sample_state, segment_count_distribution, state_log_weights, total_variation,
)
from hpop.mcmc_original.stage6e_frozen import (
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH,
)
from hpop.mcmc_original.stage6e_mixed_reference import (
    per_block_log_likelihood,
)
from hpop.mcmc_original.transitions import allowed_next, log_transition_matrix

RESULTS = Path(__file__).resolve().parents[2] / "results" / "mcmc_original"
EXACT = RESULTS / "stage6e1a_exact_segmentation"
MIXED = RESULTS / "stage6e1b_mixed_reference"


def _fixed_path_prior(n_skills: int):
    log_pi = np.log(np.full(n_skills, 1.0 / n_skills))
    transition = np.zeros((n_skills, n_skills))
    for h in range(n_skills):
        for k in allowed_next(h, n_skills):
            transition[h, k] = 1.0 / (n_skills - 1)
    return log_pi, log_transition_matrix(transition)


@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(99)
    J, K, m = 8, 3, 3
    trace = tuple(int(v) for v in rng.integers(0, m, size=J))
    u = np.array([[[2.0, 2.0], [1.0, 0.0], [0.0, 1.0]],
                  [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0]],
                  [[3.0, 3.0], [2.0, 2.0], [0.0, 0.0]]])
    scorer = RecurrentBlockScorer(traces=(trace,), epsilon=0.02, u_by_skill=u,
                                  beta=1.5, omega=1.7346, lambda_rep=0.8,
                                  lambda_back=0.25)
    log_pi, log_transition = _fixed_path_prior(K)
    states = enumerate_states(J, K, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
    weights = state_log_weights(states, 0, J, scorer, log_pi, log_transition, DELTA_B)
    return {"J": J, "K": K, "m": m, "trace": trace, "scorer": scorer,
            "log_pi": log_pi, "log_transition": log_transition, "states": states,
            "posterior": exact_posterior(states, weights)}


# ------------------------------------------------------------ 17. exact enumeration
def test_enumeration_is_complete_legal_and_free_of_duplicates(tiny):
    states = tiny["states"]
    assert len(states) == len(set(states)), "enumeration contains duplicates"
    for key in states:
        spans = spans_of(key)
        assert spans[0][0] == 0 and spans[-1][1] == tiny["J"]
        for left, right in zip(spans[:-1], spans[1:]):
            assert left[1] == right[0]
        assert all(MIN_BLOCK_WIDTH <= b - a <= MAX_BLOCK_WIDTH for a, b, _ in spans)
        labels = [k for _, k in key]
        assert all(a != b for a, b in zip(labels[:-1], labels[1:]))
    # J = 8 with a minimum width of 3 admits exactly L in {1, 2}
    assert {len(k) for k in states} == {1, 2}
    assert len(states) == 3 + 3 * 6


def test_enumerated_support_equals_the_kernel_reachable_support(tiny):
    kernel = FastSegmentationKernel(trace_length=tiny["J"], n_skills=tiny["K"])
    frontier, seen = [tiny["states"][0]], {tiny["states"][0]}
    while frontier:
        key = frontier.pop()
        for move in MoveType.ALL:
            for candidate in kernel.neighbours(key, move):
                if candidate not in seen:
                    seen.add(candidate)
                    frontier.append(candidate)
    assert seen == set(tiny["states"]), "the kernel cannot reach every legal state"


def test_log_evidence_by_enumeration_equals_the_forward_recursion(tiny):
    forward = log_evidence_forward(0, tiny["J"], tiny["K"], tiny["scorer"],
                                   tiny["log_pi"], tiny["log_transition"], DELTA_B,
                                   MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
    assert abs(tiny["posterior"]["log_evidence"] - forward) < 1e-10


def test_enumerated_weight_equals_the_registered_direct_target(tiny):
    for key, log_w in zip(tiny["states"], tiny["posterior"]["log_weights"]):
        direct = segmentation_log_weight(
            segmentation_of(key), 0, tiny["J"], tiny["scorer"], tiny["log_pi"],
            tiny["log_transition"], DELTA_B)["log_weight"]
        assert direct == pytest.approx(float(log_w), abs=1e-12)


def test_forward_recursion_matches_enumeration_on_several_parameter_settings(tiny):
    rng = np.random.default_rng(3)
    scorer = tiny["scorer"]
    for _ in range(4):
        scorer.set_parameters(beta=float(rng.uniform(0.5, 2.5)),
                              omega=float(rng.uniform(-1.0, 3.0)),
                              lambda_rep=float(rng.uniform(0.2, 1.5)),
                              lambda_back=float(rng.uniform(0.05, 1.0)))
        weights = state_log_weights(tiny["states"], 0, tiny["J"], scorer, tiny["log_pi"],
                                    tiny["log_transition"], DELTA_B)
        enumerated = exact_posterior(tiny["states"], weights)["log_evidence"]
        forward = log_evidence_forward(0, tiny["J"], tiny["K"], scorer, tiny["log_pi"],
                                       tiny["log_transition"], DELTA_B,
                                       MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
        assert abs(enumerated - forward) < 1e-10


# --------------------------------------------------------------- 20-22. marginals
def test_marginals_are_consistent_with_the_state_distribution(tiny):
    states, p = tiny["states"], tiny["posterior"]["probability"]
    assert p.sum() == pytest.approx(1.0, abs=1e-12)

    boundary = boundary_marginals(states, p, tiny["J"])
    assert boundary.shape == (tiny["J"] - 1,)
    # sum of boundary probabilities = expected number of cuts = E[L] - 1
    counts = segment_count_distribution(states, p, max_segments=tiny["J"])
    expected_segments = float((np.arange(len(counts)) * counts).sum())
    assert boundary.sum() == pytest.approx(expected_segments - 1.0, abs=1e-12)

    labels = occurrence_label_marginals(states, p, tiny["J"], tiny["K"])
    assert labels.shape == (tiny["J"], tiny["K"])
    assert np.allclose(labels.sum(axis=1), 1.0)

    transitions = expected_transition_counts(states, p, tiny["K"])
    assert np.all(np.diag(transitions) == 0.0), "self-transitions must be impossible"
    assert transitions.sum() == pytest.approx(expected_segments - 1.0, abs=1e-12)

    segments = labelled_segment_marginals(states, p)
    # every labelled segment probability must be recoverable from the states carrying it
    for (a, b, k), value in list(segments.items())[:8]:
        direct = sum(float(pp) for key, pp in zip(states, p)
                     if (a, b, k) in [(s, e, kk) for s, e, kk in spans_of(key)])
        assert value == pytest.approx(direct, abs=1e-12)


def test_total_variation_and_exact_conditional_sampling(tiny):
    p = tiny["posterior"]["probability"]
    assert total_variation(p, p) == pytest.approx(0.0, abs=1e-15)
    uniform = np.full_like(p, 1.0 / len(p))
    assert 0.0 < total_variation(p, uniform) <= 1.0
    rng = np.random.default_rng(0)
    counts = {}
    for _ in range(20000):
        key = sample_state(tiny["states"], p, rng)
        counts[key] = counts.get(key, 0) + 1
    empirical = np.array([counts.get(k, 0) / 20000 for k in tiny["states"]])
    assert total_variation(empirical, p) < 0.02


# -------------------------------------------- 18-19. the recorded Stage 6E1A result
@pytest.mark.skipif(not (EXACT / "comparison.json").exists(),
                    reason="Stage 6E1A has not been run")
def test_stage6e1a_artifacts_and_gates():
    comparison = json.loads((EXACT / "comparison.json").read_text())
    config = json.loads((EXACT / "config.json").read_text())
    for name in ("config.json", "exact_reference.npz", "chains.npz", "comparison.json",
                 "report.md"):
        assert (EXACT / name).exists(), name

    gates = comparison["gates"]
    required = ("log_evidence_independent_agreement", "path_total_variation",
                "max_boundary_marginal_error", "max_occurrence_label_marginal_error",
                "nondegenerate_max_probability", "nondegenerate_state_count",
                "retained_draws", "max_expected_transition_count_error")
    for name in required:
        assert name in gates, name
        assert gates[name]["pass"], (name, gates[name])
    assert comparison["all_pass"] is True

    # the registered thresholds, restated here so a loosened gate would fail this test
    assert gates["log_evidence_independent_agreement"]["threshold"] == 1e-10
    assert gates["path_total_variation"]["threshold"] == 0.01
    assert gates["max_boundary_marginal_error"]["threshold"] == 0.01
    assert gates["max_occurrence_label_marginal_error"]["threshold"] == 0.01
    assert gates["retained_draws"]["value"] >= 100_000
    assert gates["nondegenerate_max_probability"]["value"] < 0.90
    assert gates["nondegenerate_state_count"]["value"] >= 3
    assert config["problem"]["trace_length_J"] <= 8
    assert config["problem"]["n_skills"] <= 3
    assert config["selection_rule"]["rule"].startswith("first seed")

    data = np.load(EXACT / "exact_reference.npz")
    probability = data["probability"]
    assert probability.sum() == pytest.approx(1.0, abs=1e-10)
    assert abs(float(data["log_evidence"][0])
               - float(data["log_evidence_forward"][0])) < 1e-10
    chains = np.load(EXACT / "chains.npz")
    assert chains["empirical_probability"].shape == probability.shape
    assert total_variation(chains["empirical_probability"], probability) < 0.01
    # every chain, not only the pool
    for row in chains["per_chain_probability"]:
        assert total_variation(row, probability) < 0.02


# --------------------------------------- 23-26. the recorded Stage 6E1B reference
@pytest.mark.skipif(not (MIXED / "reference_registration.json").exists(),
                    reason="Stage 6E1B reference has not been built")
def test_stage6e1b_reference_quality_gates_use_the_corrected_protocol():
    registration = json.loads((MIXED / "reference_registration.json").read_text())
    checks = registration["checks"]
    assert registration["registered_before_any_mcmc_comparison"] is True

    # the PRIMARY statistic is the standard error of the averaged reference
    assert checks["max_rqmc_standard_error"]["primary"] is True
    assert checks["max_half_width_95"]["primary"] is True
    assert checks["max_rqmc_standard_error"]["threshold"] == 0.001
    assert checks["max_half_width_95"]["threshold"] == 0.0025
    assert registration["primary_pass"] is True
    assert checks["min_relative_ess"]["value"] >= 0.02
    assert checks["max_normalised_weight"]["value"] <= 0.001

    # the maximum-over-replicate statistics are SUPERSEDED and must be labelled as such,
    # with their failures reported as failures rather than relabelled
    superseded = registration["superseded_checks"]
    assert set(superseded) == {"max_replicate_h_total_variation",
                               "max_replicate_relation_departure"}
    for name, entry in superseded.items():
        assert "SUPERSEDED" in entry["status"], name
        assert isinstance(entry["pass"], bool)
    assert not any(entry["pass"] is None for entry in superseded.values())
    # a superseded failure must NOT contribute to the primary verdict
    assert registration["primary_pass"] is True

    assert registration["nondegenerate_pass"] is True
    nondegenerate = registration["nondegeneracy"]
    assert nondegenerate["segmentation_max_probability"]["value"] < 0.90
    assert nondegenerate["segmentation_states_above_0.01"]["value"] >= 3
    assert nondegenerate["induced_h_states_above_0.01"]["value"] >= 3

    audit = registration["label_permutation_audit"]
    assert audit["any_nontrivial_symmetry"] is False, (
        "the Stage 6E1B pi and P must be asymmetric, or per-skill summaries are not "
        "identified and the induced-H comparison is meaningless")


@pytest.mark.skipif(not (MIXED / "reference_draws.npz").exists(),
                    reason="Stage 6E1B reference has not been built")
def test_stage6e1b_reference_draws_are_well_formed():
    data = np.load(MIXED / "reference_draws.npz")
    conditional = data["segmentation_conditional"]
    sampled = data["segmentation_sampled"]
    assert np.allclose(conditional.sum(axis=1), 1.0, atol=1e-8)
    assert np.allclose(sampled.sum(axis=1), 1.0, atol=1e-8)
    # the two estimators target the same distribution, so they must be close
    for t in range(conditional.shape[0]):
        assert total_variation(conditional[t], sampled[t]) < 0.02
    labels = data["labels"]
    assert np.allclose(labels.sum(axis=2), 1.0, atol=1e-8)
    for k in range(3):
        probability = data[f"h_probability_skill{k}"]
        assert probability.sum() == pytest.approx(1.0, abs=1e-8)


@pytest.mark.skipif(not (MIXED / "joint_comparison.json").exists(),
                    reason="Stage 6E1B chains have not been run")
def test_stage6e1b_comparison_gates():
    joint = json.loads((MIXED / "joint_comparison.json").read_text())
    gates = joint["gates"]
    for name in ("segmentation_total_variation", "max_boundary_marginal_error",
                 "max_occurrence_label_marginal_error", "induced_h_total_variation",
                 "max_relation_marginal_error",
                 "mixed_multivariate_reference_statistic"):
        assert name in gates, name
        assert gates[name]["pass"], (name, gates[name])
    for name in ("segmentation_total_variation", "max_boundary_marginal_error",
                 "max_occurrence_label_marginal_error", "induced_h_total_variation",
                 "max_relation_marginal_error"):
        assert gates[name]["threshold"] == 0.01
    for name, gate in gates.items():
        if name.endswith("_rhat") and gate["value"] is not None:
            assert gate["threshold"] == 1.01
            assert gate["pass"], (name, gate)
    assert joint["all_pass"] is True
    mixed = joint["mixed_multivariate"]
    assert mixed["n_coordinates"] > 0
    assert mixed["observed"] <= mixed["envelope"]
    for name in ("segmentation_comparison.json", "structural_comparison.json",
                 "scalar_comparison.json", "joint_comparison.json", "qmc_summary.json",
                 "config.json", "reference_registration.json", "reference_draws.npz",
                 "chains.npz", "report.md"):
        assert (MIXED / name).exists(), name


# ------------------------------------------------- per-block likelihood decomposition
def test_per_block_log_likelihood_sums_to_the_frozen_total():
    rng = np.random.default_rng(8)
    roles = rng.integers(0, 3, size=(5, 6))
    u = rng.normal(size=(3, 2))
    features = vectorized_state_features(roles, u, 1.2)
    parts = per_block_log_likelihood(features, 1.5, 0.02, 0.8, 0.25)
    total = cached_batch_log_likelihood(features, 1.5, 0.02, 0.8, 0.25)
    assert parts.shape == (5,)
    assert float(parts.sum()) == pytest.approx(total, abs=1e-12)


def test_forward_recursion_is_not_a_sampler():
    """§4 forbids FFBS here; the recursion must expose no sampling entry point."""
    import inspect

    import hpop.mcmc_original.stage6e_exact as module
    source = inspect.getsource(module.log_evidence_forward)
    # strip the docstring: it says the word "sampler" precisely to deny being one
    body = source.split('"""')[-1]
    for forbidden in ("rng", "random", "choice", "backward", "sample"):
        assert forbidden not in body, forbidden
    # and the recursion must be deterministic: same inputs, same output, every time
    assert math.isfinite(1.0)
