"""Terminal-only recovery helpers for the unanchored FULL-LATENT experiment.

This module deliberately has no corpus, checkpoint, sampler, or sealed-truth
dependency.  It is intended to be imported only after the registered stopping
rule permits recovery.  Truth is always supplied explicitly to the functions
that need it; importing this module cannot open or inspect a truth artifact.

The central convention is made explicit because it is easy to get backwards:
``learned_to_truth[k] == j`` means learned skill ``k`` is matched to true skill
``j``.  Consequently labels are mapped with ``learned_to_truth`` while a
learned ``pi`` or transition matrix is put into true-label order with its
inverse.  The *same* structural mapping is used for every quantity.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


__all__ = [
    "N_SKILLS",
    "StructuralAlignment",
    "AlignedLatentDraw",
    "closure_stack_from_u",
    "structural_closure_alignment",
    "align_closures",
    "align_pi",
    "align_transition",
    "align_labels",
    "align_latent_draw",
    "closure_recovery_metrics",
    "pi_recovery_metrics",
    "transition_recovery_metrics",
    "boundary_truth_from_keys",
    "co_skill_truth_from_labels",
    "boundary_recovery_from_accumulators",
    "co_skill_recovery_from_accumulators",
    "log_segmentation_normalizer",
    "heldout_posterior_predictive",
]


N_SKILLS = 3


@dataclass(frozen=True)
class StructuralAlignment:
    """One deterministic K=3 learned-label -> truth-label structural match.

    ``cost_matrix[k, j]`` is the closure Hamming cost for learned ``k`` and
    true ``j``.  ``learned_to_truth`` is lexicographically smallest among all
    cost-minimising assignments, which makes ties reproducible across SciPy
    versions and platforms.
    """

    learned_to_truth: tuple[int, int, int]
    truth_to_learned: tuple[int, int, int]
    cost_matrix: np.ndarray
    total_cost: int
    n_optimal_assignments: int


@dataclass(frozen=True)
class AlignedLatentDraw:
    """A draw expressed in true skill order after one structural alignment."""

    alignment: StructuralAlignment
    closures: np.ndarray
    pi: np.ndarray
    transition: np.ndarray
    labels: np.ndarray | None


def _as_closure_stack(value, name: str) -> np.ndarray:
    """Validate a stack of three square strict-relation closure matrices."""
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[0] != N_SKILLS:
        raise ValueError(f"{name} must have shape (3, m, m), got {array.shape}")
    if array.shape[1] < 1 or array.shape[1] != array.shape[2]:
        raise ValueError(f"{name} must contain nonempty square matrices, got {array.shape}")
    if array.dtype != np.bool_:
        if not (np.issubdtype(array.dtype, np.number)
                and np.all(np.isfinite(array))
                and np.all((array == 0) | (array == 1))):
            raise ValueError(f"{name} must be boolean or a finite 0/1 array")
        array = array.astype(bool)
    else:
        array = np.array(array, dtype=bool, copy=True)
    if np.any(np.diagonal(array, axis1=1, axis2=2)):
        raise ValueError(f"{name} must be irreflexive (false diagonal)")
    return array


def _mapping_tuple(learned_to_truth) -> tuple[int, int, int]:
    mapping = tuple(int(v) for v in learned_to_truth)
    if len(mapping) != N_SKILLS or set(mapping) != set(range(N_SKILLS)):
        raise ValueError("learned_to_truth must be a permutation of (0, 1, 2)")
    return mapping


def _truth_to_learned(learned_to_truth) -> tuple[int, int, int]:
    mapping = _mapping_tuple(learned_to_truth)
    inverse = [0] * N_SKILLS
    for learned, truth in enumerate(mapping):
        inverse[truth] = learned
    return tuple(inverse)


def _as_probability_vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (N_SKILLS,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-{N_SKILLS} vector")
    if np.any(array < -1e-12) or not np.isclose(array.sum(), 1.0, atol=1e-8):
        raise ValueError(f"{name} must be a probability vector summing to one")
    return array


def _as_transition(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (N_SKILLS, N_SKILLS) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite ({N_SKILLS}, {N_SKILLS}) matrix")
    if np.any(array < -1e-12) or not np.allclose(array.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError(f"{name} must have nonnegative rows summing to one")
    if not np.allclose(np.diag(array), 0.0, atol=1e-10):
        raise ValueError(f"{name} must satisfy the zero-self-transition constraint")
    return array


def closure_stack_from_u(u_by_skill) -> np.ndarray:
    """Return the transitive product-order closures induced by three U matrices.

    This mirrors ``latent_poset.precedence_from_u`` without importing a model,
    corpus, or truth container.  The returned relation is the closure, not a
    transitive reduction.
    """
    u = np.asarray(u_by_skill, dtype=float)
    if u.ndim != 3 or u.shape[0] != N_SKILLS:
        raise ValueError(f"u_by_skill must have shape (3, m, d), got {u.shape}")
    if u.shape[1] < 1 or u.shape[2] < 1:
        raise ValueError("u_by_skill must have at least one role and one coordinate")
    if not np.all(np.isfinite(u)):
        raise ValueError("u_by_skill must not contain NaN or inf")
    closures = np.all(u[:, :, None, :] > u[:, None, :, :], axis=-1)
    for closure in closures:
        np.fill_diagonal(closure, False)
    return closures


def structural_closure_alignment(learned_closures, truth_closures) -> StructuralAlignment:
    """Hungarian K=3 closure alignment with deterministic lexicographic ties.

    The primary cost is the exact Hamming distance between transitive closures.
    ``linear_sum_assignment`` obtains the Hungarian optimum; all six assignments
    are then checked so that a tied optimum is resolved by the lexicographically
    smallest learned-to-truth tuple.  Enumerating six candidates is exact for
    the preregistered K=3 setting and avoids an implementation-dependent tie.
    """
    learned = _as_closure_stack(learned_closures, "learned_closures")
    truth = _as_closure_stack(truth_closures, "truth_closures")
    if learned.shape[1:] != truth.shape[1:]:
        raise ValueError("learned and truth closures must have the same role dimensions")

    cost = np.count_nonzero(
        learned[:, None, :, :] != truth[None, :, :, :], axis=(2, 3)
    ).astype(np.int64)
    rows, columns = linear_sum_assignment(cost)
    hungarian_cost = int(cost[rows, columns].sum())

    candidates = []
    for permutation in itertools.permutations(range(N_SKILLS)):
        candidate_cost = int(sum(cost[k, permutation[k]] for k in range(N_SKILLS)))
        if candidate_cost == hungarian_cost:
            candidates.append(tuple(int(v) for v in permutation))
    if not candidates:  # Defensive: impossible if scipy returned a valid assignment.
        raise RuntimeError("failed to enumerate a Hungarian-optimal K=3 assignment")
    learned_to_truth = min(candidates)
    return StructuralAlignment(
        learned_to_truth=learned_to_truth,
        truth_to_learned=_truth_to_learned(learned_to_truth),
        cost_matrix=np.array(cost, dtype=np.int64, copy=True),
        total_cost=hungarian_cost,
        n_optimal_assignments=len(candidates),
    )


def align_closures(learned_closures, learned_to_truth) -> np.ndarray:
    """Put learned closures into truth-label order under one common mapping."""
    learned = _as_closure_stack(learned_closures, "learned_closures")
    return learned[np.asarray(_truth_to_learned(learned_to_truth), dtype=int)]


def align_pi(learned_pi, learned_to_truth) -> np.ndarray:
    """Put a learned initial-skill vector into truth-label order."""
    pi = _as_probability_vector(learned_pi, "learned_pi")
    return pi[np.asarray(_truth_to_learned(learned_to_truth), dtype=int)]


def align_transition(learned_transition, learned_to_truth) -> np.ndarray:
    """Put both axes of P into truth-label order under the structural mapping."""
    transition = _as_transition(learned_transition, "learned_transition")
    inverse = np.asarray(_truth_to_learned(learned_to_truth), dtype=int)
    return transition[np.ix_(inverse, inverse)]


def align_labels(learned_labels, learned_to_truth) -> np.ndarray:
    """Map occurrence/segment labels from learned indices to true indices.

    Negative values are retained as padding sentinels, which permits direct use
    with padded occurrence-label arrays.
    """
    labels = np.asarray(learned_labels)
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("learned_labels must have an integer dtype")
    if np.any(labels >= N_SKILLS):
        raise ValueError("learned_labels contains an out-of-range skill index")
    mapping = np.asarray(_mapping_tuple(learned_to_truth), dtype=int)
    aligned = np.array(labels, dtype=int, copy=True)
    valid = aligned >= 0
    aligned[valid] = mapping[aligned[valid]]
    return aligned


def align_latent_draw(learned_closures, truth_closures, learned_pi,
                      learned_transition, learned_labels=None) -> AlignedLatentDraw:
    """Align H, pi, P, and optionally z using one structural assignment only."""
    alignment = structural_closure_alignment(learned_closures, truth_closures)
    labels = (None if learned_labels is None
              else align_labels(learned_labels, alignment.learned_to_truth))
    return AlignedLatentDraw(
        alignment=alignment,
        closures=align_closures(learned_closures, alignment.learned_to_truth),
        pi=align_pi(learned_pi, alignment.learned_to_truth),
        transition=align_transition(learned_transition, alignment.learned_to_truth),
        labels=labels,
    )


def _closure_pair_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict:
    """Closure F1 and Hamming summary for one aligned skill, off diagonal."""
    n_roles = predicted.shape[0]
    off_diagonal = ~np.eye(n_roles, dtype=bool)
    predicted = predicted[off_diagonal]
    truth = truth[off_diagonal]
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    precision = 1.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 1.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = (0.0 if precision + recall == 0.0
          else 2.0 * precision * recall / (precision + recall))
    hamming = int(np.count_nonzero(predicted != truth))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "closure_f1": float(f1),
        "closure_hamming": hamming,
        "normalized_closure_hamming": float(hamming / max(1, predicted.size)),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


def closure_recovery_metrics(aligned_closures, truth_closures) -> dict:
    """Per-skill and aggregate closure metrics after a common alignment."""
    aligned = _as_closure_stack(aligned_closures, "aligned_closures")
    truth = _as_closure_stack(truth_closures, "truth_closures")
    if aligned.shape != truth.shape:
        raise ValueError("aligned and truth closures must have identical shapes")
    per_skill = [_closure_pair_metrics(aligned[k], truth[k]) for k in range(N_SKILLS)]
    exact = bool(np.array_equal(aligned, truth))
    total_tp = sum(row["true_positive"] for row in per_skill)
    total_fp = sum(row["false_positive"] for row in per_skill)
    total_fn = sum(row["false_negative"] for row in per_skill)
    aggregate_precision = 1.0 if total_tp + total_fp == 0 else total_tp / (total_tp + total_fp)
    aggregate_recall = 1.0 if total_tp + total_fn == 0 else total_tp / (total_tp + total_fn)
    aggregate_f1 = (0.0 if aggregate_precision + aggregate_recall == 0.0 else
                    2.0 * aggregate_precision * aggregate_recall /
                    (aggregate_precision + aggregate_recall))
    total_hamming = int(sum(row["closure_hamming"] for row in per_skill))
    total_off_diagonal = N_SKILLS * aligned.shape[1] * (aligned.shape[1] - 1)
    return {
        "exact_unordered_library": exact,
        "per_skill": per_skill,
        "mean_closure_f1": float(np.mean([row["closure_f1"] for row in per_skill])),
        "mean_normalized_closure_hamming": float(
            np.mean([row["normalized_closure_hamming"] for row in per_skill])
        ),
        "total_closure_hamming": total_hamming,
        "aggregate_closure": {
            "precision": float(aggregate_precision),
            "recall": float(aggregate_recall),
            "closure_f1": float(aggregate_f1),
            "normalized_closure_hamming": float(total_hamming /
                                                  max(1, total_off_diagonal)),
        },
    }


def pi_recovery_metrics(aligned_pi, truth_pi) -> dict:
    """L1, total-variation, and RMSE error for already aligned pi vectors."""
    learned = _as_probability_vector(aligned_pi, "aligned_pi")
    truth = _as_probability_vector(truth_pi, "truth_pi")
    difference = learned - truth
    l1 = float(np.abs(difference).sum())
    return {"l1_error": l1, "total_variation_error": 0.5 * l1,
            "rmse": float(np.sqrt(np.mean(difference * difference)))}


def transition_recovery_metrics(aligned_transition, truth_transition) -> dict:
    """Off-diagonal RMSE, Frobenius, and row-TV error for one common alignment."""
    learned = _as_transition(aligned_transition, "aligned_transition")
    truth = _as_transition(truth_transition, "truth_transition")
    difference = learned - truth
    off_diagonal = ~np.eye(N_SKILLS, dtype=bool)
    row_tv = 0.5 * np.abs(difference).sum(axis=1)
    return {
        "off_diagonal_rmse": float(np.sqrt(np.mean(difference[off_diagonal] ** 2))),
        "frobenius_error": float(np.linalg.norm(difference)),
        "row_total_variation": row_tv.tolist(),
        "mean_row_total_variation": float(row_tv.mean()),
        "max_row_total_variation": float(row_tv.max()),
    }


def boundary_truth_from_keys(truth_keys: Sequence[Sequence[Sequence[int]]],
                             trace_lengths: Sequence[int] | None = None) -> list[np.ndarray]:
    """Make internal-boundary truth arrays from terminal truth segmentation keys.

    Each key is a sequence of ``(end, skill)`` pairs.  This helper is terminal
    only because callers provide the truth keys explicitly; it has no truth I/O.
    """
    if trace_lengths is not None and len(trace_lengths) != len(truth_keys):
        raise ValueError("trace_lengths and truth_keys must have the same length")
    result = []
    for index, key in enumerate(truth_keys):
        key = tuple(key)
        if not key:
            raise ValueError("each truth key must contain at least one segment")
        ends = [int(segment[0]) for segment in key]
        length = int(trace_lengths[index]) if trace_lengths is not None else ends[-1]
        if ends[-1] != length or any(a >= b for a, b in zip((0,) + tuple(ends[:-1]), ends)):
            raise ValueError("truth keys must be contiguous, positive, and end at trace length")
        boundary = np.zeros(max(0, length - 1), dtype=bool)
        for end in ends[:-1]:
            boundary[end - 1] = True
        result.append(boundary)
    return result


def co_skill_truth_from_labels(truth_labels: Sequence[np.ndarray], pairs) -> np.ndarray:
    """Truth same-skill indicators for a fixed occurrence-pair probe set.

    A pair can be ``(trace, i, j)`` for two positions in one trace or
    ``(left_trace, left_i, right_trace, right_j)`` for a cross-trace pair.
    """
    arrays = [np.asarray(labels) for labels in truth_labels]
    truth = np.empty(len(pairs), dtype=bool)
    for index, pair in enumerate(pairs):
        if len(pair) == 3:
            left_trace, left, right = (int(v) for v in pair)
            right_trace = left_trace
        elif len(pair) == 4:
            left_trace, left, right_trace, right = (int(v) for v in pair)
        else:
            raise ValueError("each co-skill pair must have three or four integer entries")
        if not (0 <= left_trace < len(arrays) and 0 <= right_trace < len(arrays)):
            raise ValueError(f"pair {index} has an invalid trace index")
        left_labels, right_labels = arrays[left_trace], arrays[right_trace]
        if not (0 <= left < left_labels.size and 0 <= right < right_labels.size):
            raise ValueError(f"pair {index} has an out-of-range occurrence index")
        truth[index] = left_labels[left] == right_labels[right]
    return truth


def _accumulator_probabilities(sums, n_retained_draws: int, name: str) -> np.ndarray:
    if isinstance(n_retained_draws, bool) or int(n_retained_draws) != n_retained_draws \
            or int(n_retained_draws) < 1:
        raise ValueError("n_retained_draws must be a positive integer")
    count = int(n_retained_draws)
    values = np.asarray(sums, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < -1e-10) \
            or np.any(values > count + 1e-10):
        raise ValueError(f"{name} must contain finite counts in [0, n_retained_draws]")
    return values / count


def _binary_metrics(probability: np.ndarray, truth: np.ndarray, threshold: float) -> dict:
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must lie in [0, 1]")
    if probability.shape != truth.shape:
        raise ValueError("accumulator and truth shapes must match")
    truth = np.asarray(truth, dtype=bool)
    predicted = probability >= float(threshold)
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    precision = 1.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 1.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "brier_score": float(np.mean((probability - truth.astype(float)) ** 2)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "n_true_positive": int(np.count_nonzero(truth)),
        "n_predicted_positive": int(np.count_nonzero(predicted)),
    }


def boundary_recovery_from_accumulators(boundary_sums, n_retained_draws: int,
                                        truth_boundaries, *, threshold: float = 0.5) -> dict:
    """Boundary recovery from online per-boundary cut counts, not saved paths.

    ``boundary_sums[n][j]`` must be the retained-draw count of a segment cut
    after occurrence ``j`` in trace ``n``.  ``truth_boundaries`` is the matching
    terminal-only sequence of boolean arrays, one of length ``J_n - 1`` per
    trace.
    """
    if len(boundary_sums) != len(truth_boundaries):
        raise ValueError("boundary_sums and truth_boundaries must have the same number of traces")
    probabilities, truth = [], []
    for index, (counts, target) in enumerate(zip(boundary_sums, truth_boundaries)):
        probability = _accumulator_probabilities(counts, n_retained_draws,
                                                  f"boundary_sums[{index}]")
        target = np.asarray(target, dtype=bool)
        if probability.ndim != 1 or target.ndim != 1:
            raise ValueError("each boundary accumulator and truth vector must be 1-D")
        if probability.shape != target.shape:
            raise ValueError(f"boundary accumulator/truth mismatch for trace {index}")
        probabilities.append(probability)
        truth.append(target)
    flat_probability = (np.concatenate(probabilities) if probabilities
                        else np.empty(0, dtype=float))
    flat_truth = np.concatenate(truth) if truth else np.empty(0, dtype=bool)
    if flat_probability.size == 0:
        raise ValueError("at least one internal boundary position is required")
    metrics = _binary_metrics(flat_probability, flat_truth, threshold)
    return {
        "n_retained_draws": int(n_retained_draws),
        "n_traces": len(probabilities),
        "n_internal_positions": int(flat_probability.size),
        "threshold": float(threshold),
        "boundary_probabilities": [row.tolist() for row in probabilities],
        "boundary_brier_score": metrics.pop("brier_score"),
        "boundary_precision": metrics.pop("precision"),
        "boundary_recall": metrics.pop("recall"),
        "boundary_f1": metrics.pop("f1"),
        "expected_total_boundaries": float(flat_probability.sum()),
        "expected_total_segments": float(len(probabilities) + flat_probability.sum()),
        **metrics,
    }


def co_skill_recovery_from_accumulators(co_skill_sums, n_retained_draws: int,
                                        truth_same_skill, *, threshold: float = 0.5) -> dict:
    """Co-skill recovery from online counts on a fixed occurrence-pair probe set."""
    probability = _accumulator_probabilities(co_skill_sums, n_retained_draws,
                                              "co_skill_sums")
    truth = np.asarray(truth_same_skill, dtype=bool)
    if probability.ndim != 1 or truth.ndim != 1:
        raise ValueError("co-skill accumulators and truth must be 1-D")
    if probability.size == 0:
        raise ValueError("at least one co-skill probe pair is required")
    metrics = _binary_metrics(probability, truth, threshold)
    return {
        "n_retained_draws": int(n_retained_draws),
        "n_pairs": int(probability.size),
        "threshold": float(threshold),
        "co_skill_probabilities": probability.tolist(),
        "co_skill_brier_score": metrics.pop("brier_score"),
        "pairwise_precision": metrics.pop("precision"),
        "pairwise_recall": metrics.pop("recall"),
        "pairwise_f1": metrics.pop("f1"),
        **metrics,
    }


def log_segmentation_normalizer(trace_length: int, delta_b: float,
                                min_width: int, max_width: int) -> float:
    """Exact ``log C_J(delta_B)`` for the registered legal-width prior.

    ``C_J`` is the sum over legal segmentations of
    ``delta_B**(L - 1) * (1-delta_B)**(J - L)``.  This independent log-space DP
    is used by held-out reporting to turn a forward *unnormalized* ``log Z``
    into the proper observation likelihood.
    """
    J = int(trace_length)
    min_width, max_width = int(min_width), int(max_width)
    if J < 1:
        raise ValueError("trace_length must be positive")
    if not (0.0 < float(delta_b) < 1.0):
        raise ValueError("delta_b must lie strictly between zero and one")
    if not (1 <= min_width <= max_width):
        raise ValueError("need 1 <= min_width <= max_width")
    log_delta = math.log(float(delta_b))
    log_non_boundary = math.log1p(-float(delta_b))
    suffix = np.full(J + 1, -np.inf, dtype=float)
    suffix[0] = 0.0
    for remaining in range(1, J + 1):
        total = -np.inf
        for width in range(min_width, min(max_width, remaining) + 1):
            tail = suffix[remaining - width]
            if not np.isfinite(tail):
                continue
            term = ((width - 1) * log_non_boundary
                    + (log_delta if width < remaining else 0.0) + tail)
            total = np.logaddexp(total, term)
        suffix[remaining] = total
    if not np.isfinite(suffix[J]):
        raise ValueError(f"no legal segmentation of J={J} for widths "
                         f"[{min_width}, {max_width}]")
    return float(suffix[J])


def heldout_posterior_predictive(log_forward_normalizers, trace_lengths: Sequence[int], *,
                                 delta_b: float, min_width: int,
                                 max_width: int) -> dict:
    """Posterior-predictive held-out NLL with the required exact ``-log C_J``.

    ``log_forward_normalizers[d, n]`` must be the forward-DP log sum of the
    *unnormalized* legal segmentation weights for posterior draw ``d`` and
    held-out trace ``n``.  For every draw and trace this helper evaluates

    ``log p(x_n | theta_d) = log Z_n(theta_d) - log C_{J_n}(delta_B)``.

    It then computes the primary posterior-predictive quantity by log-mean-exp
    over draws separately for each trace, before summing traces and dividing by
    total held-out occurrences.  The conditional-on-draw NLL is returned under
    a separate, explicitly labelled key and is not the predictive metric.
    """
    log_z = np.asarray(log_forward_normalizers, dtype=float)
    if log_z.ndim != 2 or log_z.shape[0] < 1:
        raise ValueError("log_forward_normalizers must have shape (n_draws, n_traces)")
    if np.any(np.isnan(log_z)) or np.any(np.isposinf(log_z)):
        raise ValueError("log_forward_normalizers may be finite or -inf, never NaN/+inf")
    lengths = np.asarray(trace_lengths, dtype=int)
    if lengths.ndim != 1 or lengths.size != log_z.shape[1] or np.any(lengths < 1):
        raise ValueError("trace_lengths must be positive and match the trace axis")

    normalizer_by_length = {
        int(length): log_segmentation_normalizer(int(length), delta_b,
                                                  min_width, max_width)
        for length in np.unique(lengths)
    }
    log_c = np.asarray([normalizer_by_length[int(length)] for length in lengths], dtype=float)
    log_likelihood = log_z - log_c[None, :]
    log_predictive = np.logaddexp.reduce(log_likelihood, axis=0) - math.log(log_z.shape[0])
    total_occurrences = int(lengths.sum())
    per_draw_total = log_likelihood.sum(axis=1)
    nll_per_occurrence = float(-log_predictive.sum() / total_occurrences)
    return {
        "n_draws": int(log_z.shape[0]),
        "n_traces": int(log_z.shape[1]),
        "total_heldout_occurrences": total_occurrences,
        "per_trace_log_c_j": log_c.tolist(),
        "per_trace_log_predictive": log_predictive.tolist(),
        "per_trace_nll": (-log_predictive).tolist(),
        "total_log_predictive": float(log_predictive.sum()),
        "heldout_nll_per_occurrence": nll_per_occurrence,
        # Kept as a stage-6E-compatible alias; both name the primary predictive metric.
        "nll_per_occurrence": nll_per_occurrence,
        "nll_per_trace": float(-log_predictive.mean()),
        "mean_conditional_on_draw_nll_per_occurrence": float(
            -per_draw_total.mean() / total_occurrences
        ),
        "method": (
            "per draw: log Z_forward - log C_J(delta_B); per trace: "
            "log-mean-exp over posterior draws"
        ),
    }
