"""Stage 6E2 — segmentation, label, structural and predictive diagnostics.

Everything that already exists is reused rather than reimplemented: rank-normalised split
R-hat, bulk/tail ESS and MCSE come through `stage6c_diagnostics.convergence_block`, the
closure map is `latent_poset.precedence_from_u`, and the marginal segmentation likelihood
is `stage6e_exact.log_evidence_forward`.

Three conventions are held throughout, each inherited from a stage that had to learn it.

**Correctness, convergence and recovery are separate verdicts.** A recovery failure is not
evidence that the sampler is wrong, and a converged chain is not evidence that the model
recovers anything. They are computed by different functions here and reported separately.

**Closure and transitive reduction answer different questions.** `precedence_from_u`
returns the transitive closure. Scoring recovery on the closure rewards a method for
getting implied relations right; scoring it on the reduction asks about the cover relation.
Both are reported, never averaged together.

**Degenerate coordinates are reported as degenerate.** A constant trace has no R-hat, and
saying "1.0" would present the absence of variation as evidence of convergence.

## Label switching

`skill_alignment` implements §10: a deterministic Hungarian assignment per retained draw,
on a **frozen cost** — the draw's occurrence-level confusion against the truth. The truth is
used *only* here and only for reporting; it never touches the target, the proposals or any
convergence statistic. Convergence is measured on permutation-invariant summaries instead
(`co_clustering_sample`, segment counts, relation counts), because a raw per-skill label
trace has no meaning if the target is label-exchangeable and is misleading if it is only
nearly so.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.stage6c_diagnostics import convergence_block

__all__ = [
    "labels_to_key", "boundary_indicators", "boundary_recovery", "transitive_reduction",
    "skill_alignment", "skill_recovery", "partial_order_recovery",
    "co_clustering_sample", "segment_length_distribution", "calibration_table",
    "heldout_predictive", "adjusted_rand_index", "normalised_mutual_information",
]


# ------------------------------------------------------------------- representations
def labels_to_key(labels: np.ndarray) -> tuple:
    """Occurrence labels -> `((end, skill), ...)`.

    Exact, not approximate: `P` forbids self-transitions, so adjacent segments always
    carry different labels and a cut is precisely a label change.
    """
    labels = np.asarray(labels)
    labels = labels[labels >= 0]
    out, start = [], 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            out.append((i, int(labels[start])))
            start = i
    out.append((len(labels), int(labels[start])))
    return tuple(out)


def boundary_indicators(labels: np.ndarray) -> np.ndarray:
    """`b[t] = 1` iff a cut falls at internal position `t + 1`, for `t = 0 .. J-2`."""
    labels = np.asarray(labels)
    labels = labels[labels >= 0]
    return (labels[1:] != labels[:-1]).astype(np.int8)


# ---------------------------------------------------------------- boundary recovery
def boundary_recovery(label_draws, true_keys, lengths, threshold: float = 0.5) -> dict:
    """Posterior boundary probabilities and every §16 boundary statistic.

    `label_draws` is `(n_draws, n_traces, max_J)` padded with -1.
    """
    n_draws, n_traces, _ = label_draws.shape
    probability, truth = [], []
    for n in range(n_traces):
        J = int(lengths[n])
        indicators = np.array([boundary_indicators(label_draws[d, n])
                               for d in range(n_draws)])
        probability.append(indicators.mean(axis=0))
        true_cuts = {end for end, _ in true_keys[n][:-1]}
        truth.append(np.array([1.0 if t + 1 in true_cuts else 0.0
                               for t in range(J - 1)]))

    flat_p = np.concatenate(probability)
    flat_y = np.concatenate(truth)
    predicted = flat_p >= threshold
    true_positive = float(np.sum(predicted & (flat_y > 0.5)))
    false_positive = float(np.sum(predicted & (flat_y < 0.5)))
    false_negative = float(np.sum((~predicted) & (flat_y > 0.5)))
    precision = true_positive / max(1e-12, true_positive + false_positive)
    recall = true_positive / max(1e-12, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    true_counts = np.array([len(k) for k in true_keys], dtype=float)
    posterior_counts = np.array([[len(labels_to_key(label_draws[d, n]))
                                  for n in range(n_traces)] for d in range(n_draws)],
                                dtype=float)
    mean_counts = posterior_counts.mean(axis=0)

    return {
        "threshold": threshold,
        "n_internal_positions": int(flat_y.size),
        "n_true_boundaries": int(flat_y.sum()),
        "n_predicted_boundaries": int(predicted.sum()),
        "boundary_precision": precision, "boundary_recall": recall, "boundary_f1": f1,
        "boundary_brier_score": float(np.mean((flat_p - flat_y) ** 2)),
        "calibration": calibration_table(flat_p, flat_y),
        "segment_count_error": {
            "mean_absolute_error": float(np.abs(mean_counts - true_counts).mean()),
            "mean_signed_error": float((mean_counts - true_counts).mean()),
            "true_total": int(true_counts.sum()),
            "posterior_mean_total": float(mean_counts.sum()),
            "posterior_sd_total": float(posterior_counts.sum(axis=1).std(ddof=1)),
        },
        "segment_length_distribution": segment_length_distribution(
            label_draws, true_keys, lengths),
        "mean_posterior_boundary_probability_at_true_cuts": float(
            flat_p[flat_y > 0.5].mean()) if flat_y.sum() else None,
        "mean_posterior_boundary_probability_elsewhere": float(
            flat_p[flat_y < 0.5].mean()),
        "per_trace_boundary_probability": [p.tolist() for p in probability[:5]],
    }


def calibration_table(probability, truth, n_bins: int = 10) -> dict:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probability >= lo) & (probability < hi if i < n_bins - 1
                                      else probability <= hi)
        if not mask.any():
            rows.append({"bin": [float(lo), float(hi)], "n": 0,
                         "mean_probability": None, "empirical_frequency": None})
            continue
        rows.append({"bin": [float(lo), float(hi)], "n": int(mask.sum()),
                     "mean_probability": float(probability[mask].mean()),
                     "empirical_frequency": float(truth[mask].mean())})
    populated = [r for r in rows if r["n"] > 0]
    gaps = [abs(r["mean_probability"] - r["empirical_frequency"]) for r in populated]
    weights = np.array([r["n"] for r in populated], dtype=float)
    return {"bins": rows,
            "expected_calibration_error": float(
                (weights * np.array(gaps)).sum() / weights.sum()) if populated else None,
            "max_calibration_gap": float(max(gaps)) if gaps else None}


def segment_length_distribution(label_draws, true_keys, lengths) -> dict:
    n_draws, n_traces, _ = label_draws.shape
    posterior, true = [], []
    for d in range(n_draws):
        for n in range(n_traces):
            key = labels_to_key(label_draws[d, n])
            start = 0
            for end, _ in key:
                posterior.append(end - start)
                start = end
    for n, key in enumerate(true_keys):
        start = 0
        for end, _ in key:
            true.append(end - start)
            start = end
    posterior = np.array(posterior, dtype=float)
    true = np.array(true, dtype=float)
    support = np.arange(1, int(max(posterior.max(), true.max())) + 1)
    p_hist = np.array([(posterior == w).mean() for w in support])
    t_hist = np.array([(true == w).mean() for w in support])
    return {"widths": support.tolist(),
            "posterior_probability": p_hist.tolist(),
            "true_probability": t_hist.tolist(),
            "total_variation": float(0.5 * np.abs(p_hist - t_hist).sum()),
            "posterior_mean_width": float(posterior.mean()),
            "true_mean_width": float(true.mean())}


# ------------------------------------------------------------------- label recovery
def skill_alignment(labels: np.ndarray, true_labels: np.ndarray, n_skills: int) -> tuple:
    """Deterministic Hungarian assignment on one draw's occurrence confusion.

    The cost is frozen: `-confusion[inferred, true]`, so the assignment maximises the
    number of correctly aligned occurrences. `linear_sum_assignment` is deterministic, so
    the same draw always yields the same permutation.
    """
    confusion = np.zeros((n_skills, n_skills))
    valid = (labels >= 0) & (true_labels >= 0)
    np.add.at(confusion, (labels[valid], true_labels[valid]), 1.0)
    rows, columns = linear_sum_assignment(-confusion)
    permutation = np.empty(n_skills, dtype=int)
    permutation[rows] = columns
    return permutation, confusion


def adjusted_rand_index(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    classes, class_index = np.unique(a, return_inverse=True)
    clusters, cluster_index = np.unique(b, return_inverse=True)
    table = np.zeros((len(classes), len(clusters)))
    np.add.at(table, (class_index, cluster_index), 1)
    def comb2(x):
        return x * (x - 1) / 2.0
    sum_ij = comb2(table).sum()
    sum_i = comb2(table.sum(axis=1)).sum()
    sum_j = comb2(table.sum(axis=0)).sum()
    total = comb2(table.sum())
    expected = sum_i * sum_j / total
    maximum = 0.5 * (sum_i + sum_j)
    return float((sum_ij - expected) / (maximum - expected)) if maximum != expected else 1.0


def normalised_mutual_information(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(table, (ai, bi), 1)
    joint = table / table.sum()
    pa, pb = joint.sum(axis=1), joint.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mutual = np.nansum(joint * np.log(joint / np.outer(pa, pb)))
        ha = -np.nansum(pa * np.log(pa))
        hb = -np.nansum(pb * np.log(pb))
    denominator = math.sqrt(max(ha * hb, 1e-300))
    return float(mutual / denominator) if denominator > 0 else 1.0


def co_clustering_sample(label_draws, pair_sample, chain_sizes=None) -> dict:
    """`p(occurrence i and occurrence j carry the same skill)` on a fixed pair sample.

    Permutation invariant, so it is a legitimate convergence summary even if the target is
    label-exchangeable. `pair_sample` is `[(trace, i, j), ...]`, drawn once with a fixed
    seed and reused across chains.
    """
    n_draws = label_draws.shape[0]
    values = np.empty((n_draws, len(pair_sample)))
    for index, (trace, i, j) in enumerate(pair_sample):
        values[:, index] = (label_draws[:, trace, i] == label_draws[:, trace, j])
    out = {"n_pairs": len(pair_sample),
           "posterior_probability": values.mean(axis=0).tolist()}
    if chain_sizes is not None:
        offset, per_chain = 0, []
        for size in chain_sizes:
            per_chain.append(values[offset:offset + size])
            offset += size
        rhats, esss = [], []
        for index in range(len(pair_sample)):
            block = convergence_block(
                np.array([c[:, index] for c in per_chain]), f"co-cluster {index}")
            if not block.get("degenerate"):
                rhats.append(block["rhat"])
                esss.append(block["bulk_ess"])
        out.update({"max_rhat": float(max(rhats)) if rhats else None,
                    "min_bulk_ess": float(min(esss)) if esss else None,
                    "n_varying_pairs": len(rhats),
                    "n_degenerate_pairs": len(pair_sample) - len(rhats)})
    return out


def skill_recovery(label_draws, true_label_arrays, n_skills: int,
                   true_keys=None) -> dict:
    """Everything §16 asks about skill labels, aligned per draw by Hungarian assignment."""
    n_draws, n_traces, _ = label_draws.shape
    flat_true = np.concatenate([np.asarray(a) for a in true_label_arrays])

    accuracies, aris, nmis, permutations = [], [], [], []
    confusion_total = np.zeros((n_skills, n_skills))
    for d in range(n_draws):
        drawn = np.concatenate([label_draws[d, n][:len(true_label_arrays[n])]
                                for n in range(n_traces)])
        permutation, confusion = skill_alignment(drawn, flat_true, n_skills)
        aligned = permutation[drawn]
        accuracies.append(float((aligned == flat_true).mean()))
        aris.append(adjusted_rand_index(drawn, flat_true))
        nmis.append(normalised_mutual_information(drawn, flat_true))
        permutations.append(tuple(permutation.tolist()))
        np.add.at(confusion_total, (aligned, flat_true), 1.0)

    unique_permutations: dict = {}
    for p in permutations:
        unique_permutations[p] = unique_permutations.get(p, 0) + 1
    switches = sum(1 for a, b in zip(permutations[:-1], permutations[1:]) if a != b)

    confusion_total /= confusion_total.sum()
    row_normalised = confusion_total / confusion_total.sum(axis=0, keepdims=True)
    off_diagonal = row_normalised.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    worst = np.unravel_index(int(np.argmax(off_diagonal)), off_diagonal.shape)

    segment_accuracy = repeated_accuracy = None
    if true_keys is not None:
        correct = total = repeat_correct = repeat_total = 0
        for d in range(n_draws):
            for n in range(n_traces):
                drawn = label_draws[d, n][:len(true_label_arrays[n])]
                permutation, _ = skill_alignment(
                    drawn, np.asarray(true_label_arrays[n]), n_skills)
                aligned = permutation[drawn]
                start = 0
                seen: dict = {}
                for end, skill in true_keys[n]:
                    majority = np.bincount(aligned[start:end],
                                           minlength=n_skills).argmax()
                    hit = int(majority == skill)
                    correct += hit
                    total += 1
                    seen[skill] = seen.get(skill, 0) + 1
                    if seen[skill] > 1:               # a repeated invocation of this skill
                        repeat_correct += hit
                        repeat_total += 1
                    start = end
        segment_accuracy = correct / max(1, total)
        repeated_accuracy = repeat_correct / max(1, repeat_total)

    return {
        "occurrence_aligned_accuracy": {
            "mean": float(np.mean(accuracies)), "sd": float(np.std(accuracies, ddof=1)),
            "q025": float(np.quantile(accuracies, 0.025)),
            "q975": float(np.quantile(accuracies, 0.975))},
        "adjusted_rand_index": {
            "mean": float(np.mean(aris)), "sd": float(np.std(aris, ddof=1)),
            "q025": float(np.quantile(aris, 0.025)),
            "q975": float(np.quantile(aris, 0.975))},
        "normalised_mutual_information": {
            "mean": float(np.mean(nmis)), "sd": float(np.std(nmis, ddof=1))},
        "segment_level_aligned_accuracy": segment_accuracy,
        "repeated_invocation_aligned_accuracy": repeated_accuracy,
        "repeated_invocation_note": "restricted to true segments that are the second or "
                                    "later invocation of their skill within their trace",
        "confusion_matrix_aligned": confusion_total.tolist(),
        "confusion_matrix_column_normalised": row_normalised.tolist(),
        "worst_confused_pair": {"inferred": int(worst[0]), "true": int(worst[1]),
                                "probability": float(off_diagonal[worst])},
        "alignment_permutations": {str(list(k)): int(v)
                                   for k, v in sorted(unique_permutations.items(),
                                                      key=lambda kv: -kv[1])},
        "n_distinct_alignment_permutations": len(unique_permutations),
        "label_permutation_mode_switches": switches,
        "label_permutation_switch_rate": switches / max(1, n_draws - 1),
        "alignment_rule": "deterministic Hungarian assignment per retained draw on the "
                          "frozen cost -confusion[inferred, true]; used for RECOVERY "
                          "REPORTING ONLY and never by the target, the proposals or any "
                          "convergence statistic",
    }


# --------------------------------------------------------------- structural recovery
def transitive_reduction(closure: np.ndarray) -> np.ndarray:
    """Cover relation of a transitively closed strict order.

    `i -> j` survives iff no `k` has `i -> k -> j`. Distinct from the closure and reported
    separately, because recovery scored on the two answers different questions.
    """
    closure = np.asarray(closure, dtype=bool)
    return closure & ~((closure.astype(int) @ closure.astype(int)) > 0)


def _prf1(predicted: np.ndarray, truth: np.ndarray) -> dict:
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = float((predicted & truth).sum())
    fp = float((predicted & ~truth).sum())
    fn = float((~predicted & truth).sum())
    precision = tp / max(1e-12, tp + fp) if (tp + fp) else 1.0
    recall = tp / max(1e-12, tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / max(1e-12, precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positive": int(tp), "false_positive": int(fp), "false_negative": int(fn)}


def partial_order_recovery(u_draws, permutations, u_true, n_skills: int) -> dict:
    """Per aligned skill: MAP order, `p(true H)`, relation marginals and both P/R/F1s.

    `permutations[d][k_inferred] = k_true`, the per-draw Hungarian assignment, so the
    inferred skill that plays the role of true skill `k` is looked up rather than assumed.
    """
    n_draws = u_draws.shape[0]
    m = u_draws.shape[2]
    out = []
    for true_skill in range(n_skills):
        closures, keys = [], {}
        for d in range(n_draws):
            permutation = permutations[d]
            matches = [i for i in range(n_skills) if permutation[i] == true_skill]
            if not matches:
                continue
            closure = precedence_from_u(u_draws[d, matches[0]])
            closures.append(closure)
            keys[closure.tobytes()] = keys.get(closure.tobytes(), 0) + 1
        closures = np.array(closures)
        truth = precedence_from_u(u_true[true_skill])
        marginal = closures.mean(axis=0)
        map_key = max(keys, key=keys.get)
        map_closure = np.frombuffer(map_key, dtype=bool).reshape(m, m)
        map_reduction = transitive_reduction(map_closure)
        true_reduction = transitive_reduction(truth)

        off = ~np.eye(m, dtype=bool)
        true_mask = truth & off
        false_mask = (~truth) & off
        out.append({
            "true_skill": true_skill,
            "n_draws_matched": int(len(closures)),
            "true_closure": truth.tolist(),
            "map_closure": map_closure.tolist(),
            "probability_of_true_order": float(keys.get(truth.tobytes(), 0)
                                               / max(1, len(closures))),
            "probability_of_map_order": float(keys[map_key] / max(1, len(closures))),
            "map_equals_truth": bool(np.array_equal(map_closure, truth)),
            "n_distinct_orders_visited": len(keys),
            "relation_marginal": marginal.tolist(),
            "closure": _prf1(map_closure, truth),
            "transitive_reduction": _prf1(map_reduction, true_reduction),
            "structural_hamming_distance": int(np.sum(map_closure != truth)),
            "min_probability_over_true_relations": (
                float(marginal[true_mask].min()) if true_mask.any() else None),
            "max_probability_over_false_relations": (
                float(marginal[false_mask].max()) if false_mask.any() else None),
            "true_relation_count": int(truth.sum()),
            "posterior_mean_relation_count": float(
                closures.reshape(len(closures), -1).sum(axis=1).mean()),
        })
    return {"per_skill": out,
            "closure_f1_min": min(r["closure"]["f1"] for r in out),
            "reduction_f1_min": min(r["transitive_reduction"]["f1"] for r in out),
            "max_structural_hamming": max(r["structural_hamming_distance"] for r in out),
            "min_probability_of_true_order": min(r["probability_of_true_order"]
                                                 for r in out),
            "convention": "precedence_from_u returns the transitive CLOSURE; the "
                          "reduction is the cover relation. Both are scored, never "
                          "averaged together."}


# ---------------------------------------------------------------- held-out prediction
def heldout_predictive(heldout_traces, draws, epsilon: float, delta_b: float,
                       n_skills: int, min_width: int, max_width: int,
                       progress: int = 0) -> dict:
    """Posterior-predictive held-out NLL, integrating over `(S, z)` analytically.

    For one parameter draw, `log p(x* | theta)` is the *marginal* segmentation likelihood
    of the held-out trace — every legal `(S, z)` summed — which is exactly what
    `log_evidence_forward` computes without enumerating a state. The predictive is then

        log p(x*) = log mean over draws of exp( log p(x* | theta_d) )

    so the segmentation, the labels, `U`, the four scalars and `(pi, P)` are all integrated
    over. `theta` includes whatever the draw carries, so a draw set from the unknown-boundary
    posterior and one from the oracle-boundary control are compared on the same footing:
    the held-out boundaries are unknown to both, which is the honest predictive question.
    """
    from hpop.mcmc_original.stage6e_block_table import BlockScoreTable
    from hpop.mcmc_original.stage6e_exact import log_evidence_forward

    table = BlockScoreTable(traces=heldout_traces, epsilon=epsilon, n_skills=n_skills,
                            min_width=min_width, max_width=max_width)
    n_draws = len(draws)
    per_trace = np.empty((n_draws, len(heldout_traces)))
    for d, draw in enumerate(draws):
        table.refresh(draw["u_by_skill"], draw["beta"], draw["omega"],
                      draw["lambda_rep"], draw["lambda_back"])
        log_pi = np.log(np.asarray(draw["pi"], dtype=float))
        with np.errstate(divide="ignore"):
            log_transition = np.log(np.asarray(draw["transition"], dtype=float))
        for n in range(len(heldout_traces)):
            per_trace[d, n] = log_evidence_forward(
                n, len(heldout_traces[n]), n_skills, table, log_pi, log_transition,
                delta_b, min_width, max_width)
        if progress and (d + 1) % progress == 0:
            print(f"    predictive draw {d + 1}/{n_draws}", flush=True)

    predictive = logsumexp(per_trace, axis=0) - math.log(n_draws)   # per trace
    occurrences = np.array([len(t) for t in heldout_traces], dtype=float)
    return {
        "n_draws": n_draws, "n_traces": len(heldout_traces),
        "total_log_predictive": float(predictive.sum()),
        "nll_per_trace": float(-predictive.mean()),
        "nll_per_occurrence": float(-predictive.sum() / occurrences.sum()),
        "per_trace_log_predictive": predictive.tolist(),
        "per_trace_nll": (-predictive).tolist(),
        "predictive_interval_per_trace": {
            "q025": float(np.quantile(-predictive, 0.025)),
            "median": float(np.median(-predictive)),
            "q975": float(np.quantile(-predictive, 0.975))},
        "draw_level_log_likelihood_spread": {
            "min": float(per_trace.sum(axis=1).min()),
            "max": float(per_trace.sum(axis=1).max()),
            "mean": float(per_trace.sum(axis=1).mean()),
            # a single-draw plug-in has no spread; reporting None beats reporting NaN
            "sd": (float(per_trace.sum(axis=1).std(ddof=1)) if n_draws > 1 else None)},
        "integrated_over": ["segmentation S", "labels z", "U", "beta", "omega",
                            "lambda_rep", "lambda_back", "pi", "P"],
        "method": "exact marginalisation over every legal (S, z) by forward recursion, "
                  "then a Monte Carlo average over posterior draws",
    }
