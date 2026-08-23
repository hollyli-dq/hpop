"""Focused terminal-recovery tests; these never load a corpus or truth artifact."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hpop.mcmc_original.full_latent_recovery import (
    align_latent_draw,
    boundary_recovery_from_accumulators,
    co_skill_recovery_from_accumulators,
    heldout_posterior_predictive,
    structural_closure_alignment,
)


def _closures():
    empty = np.zeros((3, 3), dtype=bool)
    edge = empty.copy()
    edge[0, 1] = True
    chain = empty.copy()
    chain[0, 1] = chain[1, 2] = chain[0, 2] = True
    return np.array([empty, edge, chain])


def test_structural_alignment_uses_lexicographic_tie_breaking():
    closures = _closures()
    alignment = structural_closure_alignment(closures, closures)
    assert alignment.learned_to_truth == (0, 1, 2)
    assert alignment.total_cost == 0

    # All three learned closures are the same, so all assignments have equal cost.
    tied = np.repeat(closures[:1], 3, axis=0)
    tied_alignment = structural_closure_alignment(tied, tied)
    assert tied_alignment.learned_to_truth == (0, 1, 2)
    assert tied_alignment.n_optimal_assignments == 6


def test_common_structural_mapping_aligns_h_pi_p_and_z_together():
    truth_closures = _closures()
    # learned skill 0 is true 2; 1 is true 0; 2 is true 1
    learned_to_truth = np.array([2, 0, 1])
    learned_closures = truth_closures[learned_to_truth]
    truth_pi = np.array([0.2, 0.3, 0.5])
    truth_p = np.array([[0.0, 0.4, 0.6], [0.2, 0.0, 0.8], [0.7, 0.3, 0.0]])
    learned_pi = truth_pi[learned_to_truth]
    learned_p = truth_p[np.ix_(learned_to_truth, learned_to_truth)]
    learned_labels = np.array([[0, 1, 2, -1]])

    aligned = align_latent_draw(learned_closures, truth_closures, learned_pi,
                                learned_p, learned_labels)
    assert aligned.alignment.learned_to_truth == (2, 0, 1)
    assert np.array_equal(aligned.closures, truth_closures)
    assert np.allclose(aligned.pi, truth_pi)
    assert np.allclose(aligned.transition, truth_p)
    assert np.array_equal(aligned.labels, np.array([[2, 0, 1, -1]]))


def test_boundary_and_co_skill_metrics_use_only_online_counts():
    boundary = boundary_recovery_from_accumulators(
        [np.array([1, 4]), np.array([3])], 4,
        [np.array([False, True]), np.array([True])], threshold=0.5,
    )
    assert boundary["boundary_probabilities"] == [[0.25, 1.0], [0.75]]
    assert boundary["boundary_brier_score"] == pytest.approx((0.25 ** 2 + 0 + 0.25 ** 2) / 3)
    assert boundary["boundary_f1"] == pytest.approx(1.0)

    co_skill = co_skill_recovery_from_accumulators(
        np.array([4, 1, 3, 0]), 4,
        np.array([True, False, True, False]), threshold=0.5,
    )
    assert co_skill["co_skill_probabilities"] == [1.0, 0.25, 0.75, 0.0]
    assert co_skill["co_skill_brier_score"] == pytest.approx((0 + 0.25 ** 2 + 0.25 ** 2 + 0) / 4)
    assert co_skill["pairwise_f1"] == pytest.approx(1.0)


def test_heldout_predictive_applies_cj_before_log_mean_exp():
    # With min=max=2 and J=2, C_J = (1-delta_B) = 0.8 exactly.
    # Thus these unnormalised forward normalisers correspond to likelihoods 1 and 1/2.
    result = heldout_posterior_predictive(
        np.log(np.array([[0.8], [0.4]])), [2], delta_b=0.2,
        min_width=2, max_width=2,
    )
    assert result["per_trace_log_c_j"] == pytest.approx([math.log(0.8)])
    assert result["per_trace_log_predictive"] == pytest.approx([math.log(0.75)])
    assert result["heldout_nll_per_occurrence"] == pytest.approx(-math.log(0.75) / 2)
    # This must remain distinct from mean conditional-on-draw NLL.
    assert result["mean_conditional_on_draw_nll_per_occurrence"] == pytest.approx(
        -math.log(0.5) / 4
    )
