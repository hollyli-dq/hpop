"""Condition A — path identifiability under oracle structures. Exact inference.

Run:  PYTHONPATH=src .venv/bin/python scripts/run_matched_condition_a.py

Target per trace:  p(S, z | x, U*, vartheta*, pi*, P*, delta_B*, epsilon*).
Only (S, z) is latent. NO MCMC: exact semi-Markov forward-backward, exact
Viterbi, and iid FFBS draws for nonlinear summaries only. Preregistration,
corpus manifest and metric definitions are written BEFORE any posterior or
recovery statistic is computed; tiny-trace exactness gates run before the
formal corpus is touched.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_generator_diagnostics as mgd            # noqa: E402
from hpop.mcmc_original import matched_synthetic_generator as msg              # noqa: E402
from hpop.mcmc_original.fast_segmentation_kernel import spans_of               # noqa: E402
from hpop.mcmc_original.matched_condition_a import (                           # noqa: E402
    NullScorer, SemiMarkovPosterior, adjusted_rand_index, auroc,
    average_precision, boundary_f1, calibration_table,
    expected_calibration_error, normalized_mutual_information,
    segmentation_voi,
)
from hpop.mcmc_original.recurrent_segmentation import RecurrentBlockScorer     # noqa: E402
from hpop.mcmc_original.stage6e_exact import enumerate_states, state_log_weights  # noqa: E402
from hpop.mcmc_original.targets import logsumexp                               # noqa: E402

CORPUS_DIR = ROOT / "results" / "mcmc_original" / "matched_synthetic_formal_corpus"
OUT = ROOT / "results" / "mcmc_original" / "matched_condition_a"

FFBS_MASTER_SEED = 6_201_001       # verified unused before registration
TINY_SEED = 6_200_777              # verified unused before registration
FFBS_DRAWS_PER_TRACE = 5_000
TINY_FFBS_DRAWS = 200_000
TINY_LENGTHS = (6, 7, 10, 6, 7, 10, 6, 7, 10)
CALIBRATION_BINS = 10
FIXED_THRESHOLD = 0.5
CLIP = 1e-300

VERDICT_RULE = {
    "strong_requires_all": {
        "boundary_auroc_all": ">= 0.85",
        "boundary_auroc_heldout": ">= 0.80",
        "boundary_brier_relative_reduction_vs_prior": ">= 0.40",
        "boundary_ece_all": "<= 0.05",
        "occurrence_modal_accuracy_all": ">= 0.80",
        "mean_posterior_prob_of_true_label_all": ">= 0.70",
        "mean_per_trace_ari": ">= 0.50",
        "map_segment_count_accuracy": ">= 0.60",
        "path_evidence": "MAP full labelled-path recovery >= 0.25 OR median "
                         "true-path posterior >= 0.02",
    },
    "not_identifiable_if_any": {
        "boundary_auroc_all": "< 0.65",
        "boundary_brier_relative_reduction_vs_prior": "< 0.10",
        "occurrence_modal_accuracy_all": "< 0.55",
        "mean_per_trace_ari": "< 0.15",
        "occurrence_nll_relative_reduction_vs_prior": "< 0.10",
    },
    "otherwise": "PATH PARTIALLY IDENTIFIABLE",
}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _dump(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_corpus():
    truth = msg.supplied_truth()
    corpus = msg.generate_corpus(6_200_001,
                                 tuple((24, 32, 40, 48)[i % 4] for i in range(100)),
                                 tuple((24, 32, 40, 48)[i % 4] for i in range(45)),
                                 truth)
    recorded = json.loads((CORPUS_DIR / "corpus_hash.json").read_text())
    regenerated = msg.corpus_hash(corpus)
    if regenerated != recorded["corpus_hash_sha256"]:
        raise SystemExit("frozen corpus hash mismatch: refusing to run "
                         f"({regenerated} != {recorded['corpus_hash_sha256']})")
    return truth, corpus, recorded


def _summ(values) -> dict:
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                   dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()),
            "median": float(np.median(a)), "sd": float(a.std(ddof=1)) if a.size > 1
            else 0.0, "q05": float(np.quantile(a, 0.05)),
            "q95": float(np.quantile(a, 0.95)),
            "min": float(a.min()), "max": float(a.max())}


# =============================================================== phase 1: freeze
def write_preregistration(recorded_hashes: dict) -> None:
    prereg = {
        "condition": "A — path identifiability under oracle structures",
        "target": "p(S, z | x, U*, rho*, beta*, omega*, lambda_rep*, "
                  "lambda_back*, pi*, P*, delta_B*, epsilon*) — per trace, "
                  "conditionally independent; everything except (S, z) fixed to "
                  "generating truth",
        "no_mcmc": "exact semi-Markov forward-backward + Viterbi; FFBS iid "
                   "draws only for nonlinear summaries; no burn-in, no "
                   "thinning, no R-hat",
        "corpus_hash_sha256": recorded_hashes["corpus_hash_sha256"],
        "truth_hash_sha256": recorded_hashes["truth_hash_sha256"],
        "ffbs": {"draws_per_trace": FFBS_DRAWS_PER_TRACE,
                 "master_seed": FFBS_MASTER_SEED,
                 "child_seed_rule": "SeedSequence(entropy=FFBS_MASTER_SEED, "
                                    "spawn_key=(global_trace_index,)) with "
                                    "global index 0..144 over train then "
                                    "heldout in frozen order",
                 "mcse_rule": "sd/sqrt(n) reported for every sample-based mean"},
        "tiny_corpus": {"seed": TINY_SEED, "trace_lengths": list(TINY_LENGTHS),
                        "ffbs_draws_per_trace": TINY_FFBS_DRAWS},
        "tiny_gates": {"log_z_dp_vs_enum": 1e-10,
                       "boundary_marginal_max_error": 1e-10,
                       "occurrence_label_max_error": 1e-10,
                       "path_probability_max_error": 1e-10,
                       "ffbs_path_tv": 0.01},
        "map_rule": "exact semi-Markov max-product; ties broken deterministically "
                    "by first-encountered candidate (lowest start, then lowest "
                    "skill index)",
        "calibration_bins": CALIBRATION_BINS,
        "fixed_threshold": FIXED_THRESHOLD,
        "threshold_note": "0.5 is fixed and never tuned on this corpus",
        "label_alignment": "NONE — fixed generating skill identities; no "
                           "Hungarian or other alignment. Symmetry check: the "
                           "three induced orders are pairwise distinct and "
                           "pi/P entries are asymmetric, so no exact label "
                           "symmetry exists; if one were found it would be "
                           "reported, not silently aligned",
        "verdict_rule": VERDICT_RULE,
        "stop_condition": "no Condition B/C/D, no U/rho/scalar inference, no "
                          "collapsed-U move, regardless of classification",
    }
    _dump("preregistration.json", prereg)
    _dump("corpus_manifest.json", {
        "corpus_dir": "results/mcmc_original/matched_synthetic_formal_corpus",
        **recorded_hashes,
        "n_train": 100, "n_heldout": 45,
        "trace_lengths": "train 25x{24,32,40,48} cycling; heldout cycling -> "
                         "12/11/11/11",
        "generation_seed": 6_200_001,
        "generator_commit": "8ca828153e8e263bf4ea4823e45a53fa454037ad",
    })
    _dump("metric_definitions.json", {
        "boundary": {
            "b_hat": "exact Pr(B_nt = 1 | x_n, truth) from forward-backward "
                     "block marginals",
            "outcome": "hidden true internal cut positions",
            "brier": "mean (b_hat - outcome)^2",
            "nll": "mean -log(b_hat if outcome else 1 - b_hat), clipped at 1e-300",
            "auroc": "Mann-Whitney with average ranks for ties",
            "auprc": "average precision (sum of precision x recall increments)",
            "ece": f"{CALIBRATION_BINS} equal-width bins, count-weighted "
                   "|mean predicted - empirical|",
            "prf_at_0.5": "precision/recall/F1 at fixed threshold 0.5",
            "prior_baseline": "p(B_t = 1 | J, delta_B) from the prior-only DP "
                              "(NullScorer), no observations",
        },
        "segment_count": "exact p(L | x) from an L-augmented forward recursion; "
                         "posterior mean/SD, p(L*), MAP L, |mean - L*|, MAP "
                         "accuracy",
        "occurrence_labels": {
            "marginal": "exact Pr(c_t = k | x) by summing block marginals over "
                        "blocks of skill k covering t",
            "metrics": "mean p(c*), modal accuracy, log score, multiclass "
                       "Brier, per-trace ARI and NMI (fixed identities), "
                       "pooled 3x3 confusion (true vs modal)",
        },
        "joint_path": "exact p(S*, z* | x); exact p(S* | x) with z "
                      "marginalized (label-chain DP); Viterbi MAP path; exact "
                      "posterior path entropy log Z - E[log w] via the linear "
                      "decomposition of log w; note: with a zero-diagonal P the "
                      "occurrence-label vector determines (S, z) bijectively, "
                      "so p(c* vector) = p(S*, z*)",
        "ffbs_summaries": "per-draw boundary F1, occurrence accuracy, and "
                          "segmentation variation of information vs truth; "
                          "means with MCSE and quantiles",
        "information_gain": "prior-to-posterior reduction in boundary "
                            "Brier/NLL, occurrence NLL, and exact path-entropy "
                            "reduction H_prior - H_posterior",
        "stratification": "train / heldout / all, and by J in {24,32,40,48}",
    })


# ============================================================ phase 2: tiny gates
def run_tiny_gates(truth) -> dict:
    corpus = msg.generate_corpus(TINY_SEED, TINY_LENGTHS, (), truth)
    scorer = RecurrentBlockScorer(
        traces=tuple(t.cpa for t in corpus.train), epsilon=truth.epsilon,
        u_by_skill=truth.u_by_skill, beta=truth.beta, omega=truth.omega,
        lambda_rep=truth.lambda_rep, lambda_back=truth.lambda_back,
        max_width=truth.max_width, min_width=truth.min_width)
    with np.errstate(divide="ignore"):
        log_pi = np.log(truth.pi)
        log_p = np.log(truth.transition)
    worst = {"log_z": 0.0, "boundary": 0.0, "occurrence": 0.0, "path": 0.0,
             "ffbs_tv": 0.0}
    for i, trace in enumerate(corpus.train):
        post = SemiMarkovPosterior(i, trace.length, scorer, log_pi, log_p,
                                   truth.delta_b, truth.min_width,
                                   truth.max_width)
        states = enumerate_states(trace.length, truth.n_skills,
                                  truth.min_width, truth.max_width)
        weights = state_log_weights(states, i, trace.length, scorer, log_pi,
                                    log_p, truth.delta_b)
        log_z = float(logsumexp(weights))
        probs = np.exp(weights - log_z)
        worst["log_z"] = max(worst["log_z"], abs(post.log_z - log_z))
        bm = np.zeros(trace.length - 1)
        occ = np.zeros((trace.length, truth.n_skills))
        for key, p in zip(states, probs):
            for end, _ in key[:-1]:
                bm[end - 1] += p
            for a, b, k in spans_of(key):
                occ[a:b, k] += p
        worst["boundary"] = max(worst["boundary"], float(
            np.abs(post.boundary_marginals() - bm).max()))
        worst["occurrence"] = max(worst["occurrence"], float(
            np.abs(post.occurrence_label_marginals() - occ).max()))
        for key, weight in zip(states, weights):
            ends = [e for e, _ in key]
            labels = [k for _, k in key]
            worst["path"] = max(worst["path"], abs(
                math.exp(post.true_path_log_posterior(ends, labels))
                - math.exp(weight - log_z)))
        rng = np.random.default_rng(
            np.random.SeedSequence(entropy=TINY_SEED, spawn_key=(9000 + i,)))
        counts = Counter(tuple(zip(*post.ffbs_draw(rng)))
                         for _ in range(TINY_FFBS_DRAWS))
        tv = 0.5 * sum(abs(counts.get(s, 0) / TINY_FFBS_DRAWS - p)
                       for s, p in zip(states, probs))
        tv += 0.5 * sum(c / TINY_FFBS_DRAWS for s, c in counts.items()
                        if s not in set(states))
        worst["ffbs_tv"] = max(worst["ffbs_tv"], tv)
    gates = {
        "log_z_dp_vs_enum": {"worst": float(worst["log_z"]), "gate": 1e-10,
                             "pass": bool(worst["log_z"] < 1e-10)},
        "boundary_marginal_max_error": {"worst": float(worst["boundary"]),
                                        "gate": 1e-10,
                                        "pass": bool(worst["boundary"] < 1e-10)},
        "occurrence_label_max_error": {"worst": float(worst["occurrence"]),
                                       "gate": 1e-10,
                                       "pass": bool(worst["occurrence"] < 1e-10)},
        "path_probability_max_error": {"worst": float(worst["path"]),
                                       "gate": 1e-10,
                                       "pass": bool(worst["path"] < 1e-10)},
        "ffbs_path_tv": {"worst": float(worst["ffbs_tv"]), "gate": 0.01,
                         "pass": bool(worst["ffbs_tv"] < 0.01)},
    }
    for name, row in gates.items():
        print(f"  [{'PASS' if row['pass'] else 'FAIL'}] tiny {name}: "
              f"{row['worst']:.3e}")
    return gates


# ============================================================ phase 3: inference
def run_inference(truth, corpus) -> dict:
    traces = list(corpus.train) + list(corpus.heldout)
    scorer = RecurrentBlockScorer(
        traces=tuple(t.cpa for t in traces), epsilon=truth.epsilon,
        u_by_skill=truth.u_by_skill, beta=truth.beta, omega=truth.omega,
        lambda_rep=truth.lambda_rep, lambda_back=truth.lambda_back,
        max_width=truth.max_width, min_width=truth.min_width)
    with np.errstate(divide="ignore"):
        log_pi = np.log(truth.pi)
        log_p = np.log(truth.transition)
    priors = {J: SemiMarkovPosterior(0, J, NullScorer(), log_pi, log_p,
                                     truth.delta_b, truth.min_width,
                                     truth.max_width)
              for J in sorted({t.length for t in traces})}
    prior_cache = {J: {
        "boundary": priors[J].boundary_marginals(),
        "occurrence": priors[J].occurrence_label_marginals(),
        "counts": priors[J].segment_count_posterior(),
        "entropy": priors[J].path_entropy(),
    } for J in priors}

    results = []
    consistency = {"occ_row_sum": 0.0, "count_sum": 0.0, "boundary_vs_counts": 0.0,
                   "ffbs_illegal_draws": 0}
    t_forward = t_ffbs = 0.0
    n_draws_total = 0
    for i, trace in enumerate(traces):
        start = time.perf_counter()
        post = SemiMarkovPosterior(i, trace.length, scorer, log_pi, log_p,
                                   truth.delta_b, truth.min_width,
                                   truth.max_width)
        boundary = post.boundary_marginals()
        occurrence = post.occurrence_label_marginals()
        counts = post.segment_count_posterior()
        transitions = post.expected_transition_counts()
        map_ends, map_labels, map_log_post = post.map_path()
        true_ends = tuple(list(trace.boundaries) + [trace.length])
        true_path_lp = post.true_path_log_posterior(true_ends, trace.labels)
        true_seg_lp = post.segmentation_log_posterior(true_ends)
        entropy = post.path_entropy()
        t_forward += time.perf_counter() - start

        consistency["occ_row_sum"] = max(consistency["occ_row_sum"], float(
            np.abs(occurrence.sum(axis=1) - 1.0).max()))
        consistency["count_sum"] = max(consistency["count_sum"],
                                       abs(float(counts.sum()) - 1.0))
        expected_cuts = float(np.dot(np.arange(len(counts)), counts)) - 1.0
        consistency["boundary_vs_counts"] = max(
            consistency["boundary_vs_counts"],
            abs(float(boundary.sum()) - expected_cuts))

        start = time.perf_counter()
        rng = np.random.default_rng(np.random.SeedSequence(
            entropy=FFBS_MASTER_SEED, spawn_key=(i,)))
        true_boundary_set = set(trace.boundaries)
        true_occ = np.repeat(np.asarray(trace.labels),
                             np.asarray(trace.widths))
        f1_draws = np.empty(FFBS_DRAWS_PER_TRACE)
        acc_draws = np.empty(FFBS_DRAWS_PER_TRACE)
        voi_draws = np.empty(FFBS_DRAWS_PER_TRACE)
        for d in range(FFBS_DRAWS_PER_TRACE):
            ends, labels = post.ffbs_draw(rng)
            widths = np.diff([0] + list(ends))
            if (ends[-1] != trace.length
                    or any(not truth.min_width <= w <= truth.max_width
                           for w in widths)
                    or any(a == b for a, b in zip(labels[:-1], labels[1:]))):
                consistency["ffbs_illegal_draws"] += 1
            f1_draws[d] = boundary_f1(set(ends[:-1]), true_boundary_set)
            acc_draws[d] = float((np.repeat(labels, widths) == true_occ).mean())
            voi_draws[d] = segmentation_voi(ends, true_ends, trace.length)
        t_ffbs += time.perf_counter() - start
        n_draws_total += FFBS_DRAWS_PER_TRACE

        results.append({
            "index": i, "split": trace.split, "J": trace.length,
            "log_z": post.log_z, "entropy": entropy,
            "prior_entropy": prior_cache[trace.length]["entropy"],
            "boundary": boundary,
            "prior_boundary": prior_cache[trace.length]["boundary"],
            "occurrence": occurrence,
            "prior_occurrence": prior_cache[trace.length]["occurrence"],
            "counts": counts,
            "prior_counts": prior_cache[trace.length]["counts"],
            "transitions": transitions,
            "map_ends": map_ends, "map_labels": map_labels,
            "map_log_post": map_log_post,
            "true_ends": true_ends, "true_labels": trace.labels,
            "true_widths": trace.widths,
            "true_occ": true_occ,
            "true_path_log_post": true_path_lp,
            "true_seg_log_post": true_seg_lp,
            "ffbs_f1": f1_draws, "ffbs_acc": acc_draws, "ffbs_voi": voi_draws,
        })
        if (i + 1) % 29 == 0:
            print(f"  inference {i + 1}/145 traces")
    return {"results": results, "consistency": consistency,
            "t_forward": t_forward, "t_ffbs": t_ffbs,
            "n_draws_total": n_draws_total}


# ============================================================== phase 4: metrics
def _boundary_pool(results, use_prior: bool):
    probs, outcomes = [], []
    for r in results:
        source = r["prior_boundary"] if use_prior else r["boundary"]
        truth_set = set(r["true_ends"][:-1])
        for t in range(1, r["J"]):
            probs.append(float(source[t - 1]))
            outcomes.append(1 if t in truth_set else 0)
    return np.asarray(probs), np.asarray(outcomes, dtype=int)


def boundary_metric_block(results, use_prior: bool = False) -> dict:
    probs, outcomes = _boundary_pool(results, use_prior)
    p_clip = np.clip(probs, CLIP, 1 - 1e-16)
    nll = float(-np.mean(np.where(outcomes == 1, np.log(p_clip),
                                  np.log1p(-p_clip))))
    predicted = probs >= FIXED_THRESHOLD
    tp = int((predicted & (outcomes == 1)).sum())
    precision = tp / int(predicted.sum()) if predicted.sum() else float("nan")
    recall = tp / int(outcomes.sum()) if outcomes.sum() else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and np.isfinite(precision)
          and np.isfinite(recall) and (precision + recall) > 0 else 0.0)
    return {
        "n_gaps": int(len(probs)), "n_true_cuts": int(outcomes.sum()),
        "brier": float(np.mean((probs - outcomes) ** 2)),
        "nll": nll,
        "auroc": auroc(probs, outcomes),
        "auprc": average_precision(probs, outcomes),
        "ece": expected_calibration_error(probs, outcomes, CALIBRATION_BINS),
        "reliability_table": calibration_table(probs, outcomes,
                                               CALIBRATION_BINS),
        "threshold_0.5": {"precision": precision, "recall": recall, "f1": f1},
    }


def occurrence_metric_block(results, use_prior: bool = False) -> dict:
    p_true, modal_hits, log_scores, briers = [], [], [], []
    ari, nmi = [], []
    confusion = np.zeros((3, 3))
    for r in results:
        marginals = r["prior_occurrence"] if use_prior else r["occurrence"]
        true_occ = r["true_occ"]
        modal = np.argmax(marginals, axis=1)
        one_hot = np.zeros_like(marginals)
        one_hot[np.arange(len(true_occ)), true_occ] = 1.0
        p_star = marginals[np.arange(len(true_occ)), true_occ]
        p_true.extend(p_star)
        modal_hits.extend(modal == true_occ)
        log_scores.extend(-np.log(np.clip(p_star, CLIP, None)))
        briers.extend(((marginals - one_hot) ** 2).sum(axis=1))
        ari.append(adjusted_rand_index(true_occ, modal))
        nmi.append(normalized_mutual_information(true_occ, modal))
        for t_label, m_label in zip(true_occ, modal):
            confusion[int(t_label), int(m_label)] += 1
    return {
        "n_occurrences": len(p_true),
        "mean_posterior_prob_true_label": float(np.mean(p_true)),
        "modal_accuracy": float(np.mean(modal_hits)),
        "log_score": float(np.mean(log_scores)),
        "brier_multiclass": float(np.mean(briers)),
        "per_trace_ari": _summ(ari),
        "per_trace_nmi": _summ(nmi),
        "confusion_true_rows_modal_cols": confusion.astype(int).tolist(),
    }


def stratified(results, block_fn) -> dict:
    out = {"all": block_fn(results),
           "train": block_fn([r for r in results if r["split"] == "train"]),
           "heldout": block_fn([r for r in results if r["split"] == "heldout"])}
    for J in (24, 32, 40, 48):
        out[f"J={J}"] = block_fn([r for r in results if r["J"] == J])
    return out


def compute_all_metrics(inference: dict) -> dict:
    results = inference["results"]

    boundary = stratified(results, boundary_metric_block)
    boundary_prior = stratified(results,
                                lambda rs: boundary_metric_block(rs, True))
    labels = stratified(results, occurrence_metric_block)
    labels_prior = stratified(results,
                              lambda rs: occurrence_metric_block(rs, True))

    count_rows = []
    for r in results:
        counts = r["counts"]
        true_l = len(r["true_widths"])
        mean_l = float(np.dot(np.arange(len(counts)), counts))
        sd_l = float(math.sqrt(max(0.0, np.dot(np.arange(len(counts)) ** 2,
                                               counts) - mean_l ** 2)))
        map_l = int(np.argmax(counts))
        prior = r["prior_counts"]
        count_rows.append({
            "split": r["split"], "J": r["J"], "true_L": true_l,
            "posterior_mean_L": mean_l, "posterior_sd_L": sd_l,
            "p_true_L": float(counts[true_l]) if true_l < len(counts) else 0.0,
            "prior_p_true_L": float(prior[true_l]) if true_l < len(prior)
            else 0.0,
            "map_L": map_l, "abs_error_mean": abs(mean_l - true_l),
            "map_correct": map_l == true_l,
        })
    segment_counts = {
        "posterior_mean_abs_error": _summ([r["abs_error_mean"]
                                           for r in count_rows]),
        "p_true_L": _summ([r["p_true_L"] for r in count_rows]),
        "prior_p_true_L": _summ([r["prior_p_true_L"] for r in count_rows]),
        "map_accuracy": float(np.mean([r["map_correct"] for r in count_rows])),
        "map_accuracy_train": float(np.mean(
            [r["map_correct"] for r in count_rows if r["split"] == "train"])),
        "map_accuracy_heldout": float(np.mean(
            [r["map_correct"] for r in count_rows if r["split"] == "heldout"])),
        "posterior_sd_L": _summ([r["posterior_sd_L"] for r in count_rows]),
        "rows": count_rows,
    }

    path_rows = []
    for r in results:
        map_key = tuple(zip(r["map_ends"], r["map_labels"]))
        true_key = tuple(zip(r["true_ends"], r["true_labels"]))
        map_widths = np.diff([0] + list(r["map_ends"]))
        map_occ = np.repeat(r["map_labels"], map_widths)
        n = FFBS_DRAWS_PER_TRACE
        path_rows.append({
            "split": r["split"], "J": r["J"],
            "true_path_log_post": r["true_path_log_post"],
            "true_path_prob": math.exp(r["true_path_log_post"]),
            "true_seg_log_post": r["true_seg_log_post"],
            "true_seg_prob": math.exp(r["true_seg_log_post"]),
            "map_log_post": r["map_log_post"],
            "map_equals_truth": map_key == true_key,
            "map_boundary_f1": boundary_f1(set(r["map_ends"][:-1]),
                                           set(r["true_ends"][:-1])),
            "map_occ_accuracy": float((map_occ == r["true_occ"]).mean()),
            "entropy": r["entropy"], "prior_entropy": r["prior_entropy"],
            "ffbs_f1_mean": float(r["ffbs_f1"].mean()),
            "ffbs_f1_mcse": float(r["ffbs_f1"].std(ddof=1) / math.sqrt(n)),
            "ffbs_acc_mean": float(r["ffbs_acc"].mean()),
            "ffbs_acc_mcse": float(r["ffbs_acc"].std(ddof=1) / math.sqrt(n)),
            "ffbs_voi_mean": float(r["ffbs_voi"].mean()),
            "ffbs_voi_mcse": float(r["ffbs_voi"].std(ddof=1) / math.sqrt(n)),
        })

    def path_block(rows):
        if not rows:
            return {"n": 0}
        return {
            "n": len(rows),
            "map_exact_recovery_rate": float(np.mean(
                [r["map_equals_truth"] for r in rows])),
            "true_path_prob": _summ([r["true_path_prob"] for r in rows]),
            "true_path_log_post": _summ([r["true_path_log_post"] for r in rows]),
            "true_seg_prob": _summ([r["true_seg_prob"] for r in rows]),
            "map_boundary_f1": _summ([r["map_boundary_f1"] for r in rows]),
            "map_occ_accuracy": _summ([r["map_occ_accuracy"] for r in rows]),
            "posterior_entropy": _summ([r["entropy"] for r in rows]),
            "entropy_reduction_vs_prior": _summ(
                [r["prior_entropy"] - r["entropy"] for r in rows]),
            "ffbs_boundary_f1_mean": _summ([r["ffbs_f1_mean"] for r in rows]),
            "ffbs_occ_accuracy_mean": _summ([r["ffbs_acc_mean"] for r in rows]),
            "ffbs_voi_mean": _summ([r["ffbs_voi_mean"] for r in rows]),
        }

    joint = {"all": path_block(path_rows),
             "train": path_block([r for r in path_rows
                                  if r["split"] == "train"]),
             "heldout": path_block([r for r in path_rows
                                    if r["split"] == "heldout"])}
    for J in (24, 32, 40, 48):
        joint[f"J={J}"] = path_block([r for r in path_rows if r["J"] == J])
    joint["per_trace_rows"] = path_rows

    prior_vs_posterior = {}
    for stratum in ("all", "train", "heldout"):
        post_b, prior_b = boundary[stratum], boundary_prior[stratum]
        post_l, prior_l = labels[stratum], labels_prior[stratum]
        rows = [r for r in path_rows
                if stratum == "all" or r["split"] == stratum]
        prior_vs_posterior[stratum] = {
            "boundary_brier": {"prior": prior_b["brier"],
                               "posterior": post_b["brier"],
                               "absolute_reduction": prior_b["brier"]
                               - post_b["brier"],
                               "relative_reduction": 1 - post_b["brier"]
                               / prior_b["brier"]},
            "boundary_nll": {"prior": prior_b["nll"],
                             "posterior": post_b["nll"],
                             "absolute_reduction": prior_b["nll"] - post_b["nll"],
                             "relative_reduction": 1 - post_b["nll"]
                             / prior_b["nll"]},
            "occurrence_nll": {"prior": prior_l["log_score"],
                               "posterior": post_l["log_score"],
                               "absolute_reduction": prior_l["log_score"]
                               - post_l["log_score"],
                               "relative_reduction": 1 - post_l["log_score"]
                               / prior_l["log_score"]},
            "occurrence_prob_true": {
                "prior": prior_l["mean_posterior_prob_true_label"],
                "posterior": post_l["mean_posterior_prob_true_label"]},
            "path_entropy_reduction_nats": _summ(
                [r["prior_entropy"] - r["entropy"] for r in rows]),
        }

    return {"boundary": boundary, "boundary_prior": boundary_prior,
            "labels": labels, "labels_prior": labels_prior,
            "segment_counts": segment_counts, "joint": joint,
            "prior_vs_posterior": prior_vs_posterior}


# =============================================================== phase 5: verdict
def classify(metrics: dict) -> dict:
    b_all = metrics["boundary"]["all"]
    b_ho = metrics["boundary"]["heldout"]
    l_all = metrics["labels"]["all"]
    pvp = metrics["prior_vs_posterior"]["all"]
    joint = metrics["joint"]["all"]
    counts = metrics["segment_counts"]
    observed = {
        "boundary_auroc_all": b_all["auroc"],
        "boundary_auroc_heldout": b_ho["auroc"],
        "boundary_brier_relative_reduction_vs_prior":
            pvp["boundary_brier"]["relative_reduction"],
        "boundary_ece_all": b_all["ece"],
        "occurrence_modal_accuracy_all": l_all["modal_accuracy"],
        "mean_posterior_prob_of_true_label_all":
            l_all["mean_posterior_prob_true_label"],
        "mean_per_trace_ari": l_all["per_trace_ari"]["mean"],
        "map_segment_count_accuracy": counts["map_accuracy"],
        "map_exact_recovery_rate": joint["map_exact_recovery_rate"],
        "median_true_path_posterior": joint["true_path_prob"]["median"],
        "occurrence_nll_relative_reduction_vs_prior":
            pvp["occurrence_nll"]["relative_reduction"],
    }
    strong = (observed["boundary_auroc_all"] >= 0.85
              and observed["boundary_auroc_heldout"] >= 0.80
              and observed["boundary_brier_relative_reduction_vs_prior"] >= 0.40
              and observed["boundary_ece_all"] <= 0.05
              and observed["occurrence_modal_accuracy_all"] >= 0.80
              and observed["mean_posterior_prob_of_true_label_all"] >= 0.70
              and observed["mean_per_trace_ari"] >= 0.50
              and observed["map_segment_count_accuracy"] >= 0.60
              and (observed["map_exact_recovery_rate"] >= 0.25
                   or observed["median_true_path_posterior"] >= 0.02))
    not_identifiable = (
        observed["boundary_auroc_all"] < 0.65
        or observed["boundary_brier_relative_reduction_vs_prior"] < 0.10
        or observed["occurrence_modal_accuracy_all"] < 0.55
        or observed["mean_per_trace_ari"] < 0.15
        or observed["occurrence_nll_relative_reduction_vs_prior"] < 0.10)
    if strong:
        verdict = "PATH STRONGLY IDENTIFIABLE"
    elif not_identifiable:
        verdict = "PATH NOT IDENTIFIABLE UNDER ORACLE STRUCTURES"
    else:
        verdict = "PATH PARTIALLY IDENTIFIABLE"
    return {"verdict": verdict, "rule": VERDICT_RULE, "observed": observed}


# ================================================================== persistence
def save_arrays(inference: dict) -> None:
    results = inference["results"]

    def pack(name, getter, dtype=np.float64):
        return {f"t{r['index']:03d}_{name}": np.asarray(getter(r), dtype=dtype)
                for r in results}

    np.savez_compressed(OUT / "exact_forward_results.npz", **{
        "index": np.asarray([r["index"] for r in results]),
        "J": np.asarray([r["J"] for r in results]),
        "is_heldout": np.asarray([r["split"] == "heldout" for r in results]),
        "log_z": np.asarray([r["log_z"] for r in results]),
        "entropy": np.asarray([r["entropy"] for r in results]),
        "prior_entropy": np.asarray([r["prior_entropy"] for r in results]),
        "true_path_log_post": np.asarray([r["true_path_log_post"]
                                          for r in results]),
        "true_seg_log_post": np.asarray([r["true_seg_log_post"]
                                         for r in results]),
        "map_log_post": np.asarray([r["map_log_post"] for r in results]),
    }, **pack("transitions", lambda r: r["transitions"]))
    np.savez_compressed(OUT / "boundary_marginals.npz",
                        **pack("post", lambda r: r["boundary"]),
                        **pack("prior", lambda r: r["prior_boundary"]))
    np.savez_compressed(OUT / "occurrence_label_marginals.npz",
                        **pack("post", lambda r: r["occurrence"]),
                        **pack("prior", lambda r: r["prior_occurrence"]))
    np.savez_compressed(OUT / "segment_count_posteriors.npz",
                        **pack("post", lambda r: r["counts"]),
                        **pack("prior", lambda r: r["prior_counts"]))
    np.savez_compressed(OUT / "map_paths.npz",
                        **pack("ends", lambda r: r["map_ends"], np.int16),
                        **pack("labels", lambda r: r["map_labels"], np.int8),
                        **pack("true_ends", lambda r: r["true_ends"], np.int16),
                        **pack("true_labels", lambda r: r["true_labels"],
                               np.int8))


def main() -> int:
    wall_start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    truth, corpus, recorded = _load_corpus()

    print("== phase 1: freeze analysis ==")
    write_preregistration(recorded)

    print("== phase 2: tiny exactness gates ==")
    tiny = run_tiny_gates(truth)
    if not all(row["pass"] for row in tiny.values()):
        _dump("correctness.json", {"tiny_gates": tiny,
                                   "status": "TINY GATE FAILURE — STOPPED"})
        print("TINY GATE FAILURE — STOP")
        return 1

    print("== phase 3: exact inference on 145 traces ==")
    inference = run_inference(truth, corpus)
    consistency = inference["consistency"]
    correctness = {
        "tiny_gates": tiny,
        "marginal_consistency": {
            "occurrence_rows_sum_to_one_max_error":
                float(consistency["occ_row_sum"]),
            "segment_count_sums_to_one_max_error":
                float(consistency["count_sum"]),
            "boundary_sum_equals_E_L_minus_1_max_error":
                float(consistency["boundary_vs_counts"]),
            "ffbs_illegal_draws": int(consistency["ffbs_illegal_draws"]),
        },
        "pass": bool(consistency["occ_row_sum"] < 1e-9
                     and consistency["count_sum"] < 1e-9
                     and consistency["boundary_vs_counts"] < 1e-8
                     and consistency["ffbs_illegal_draws"] == 0),
    }
    _dump("correctness.json", correctness)
    if not correctness["pass"]:
        print("CONSISTENCY FAILURE — STOP")
        return 1

    print("== phase 4: metrics ==")
    metrics = compute_all_metrics(inference)
    save_arrays(inference)
    _dump("boundary_metrics.json", {"posterior": metrics["boundary"],
                                    "prior": metrics["boundary_prior"]})
    _dump("label_metrics.json", {"posterior": metrics["labels"],
                                 "prior": metrics["labels_prior"],
                                 "label_symmetry": "none exists: the three "
                                 "induced orders are pairwise distinct and "
                                 "pi/P are asymmetric; identities used "
                                 "directly, no alignment"})
    _dump("segment_count_posteriors_summary.json",
          {k: v for k, v in metrics["segment_counts"].items() if k != "rows"})
    _dump("joint_path_metrics.json", metrics["joint"])
    _dump("prior_vs_posterior.json", metrics["prior_vs_posterior"])
    _dump("true_path_probabilities.json", [
        {"index": r["index"], "split": r["split"], "J": r["J"],
         "true_path_log_post": r["true_path_log_post"],
         "true_seg_log_post": r["true_seg_log_post"]}
        for r in inference["results"]])
    _dump("ffbs_sample_summaries.json", {
        "draws_per_trace": FFBS_DRAWS_PER_TRACE,
        "master_seed": FFBS_MASTER_SEED,
        "boundary_f1": metrics["joint"]["all"]["ffbs_boundary_f1_mean"],
        "occurrence_accuracy": metrics["joint"]["all"]["ffbs_occ_accuracy_mean"],
        "variation_of_information": metrics["joint"]["all"]["ffbs_voi_mean"],
        "per_trace": [{k: r[k] for k in
                       ("split", "J", "ffbs_f1_mean", "ffbs_f1_mcse",
                        "ffbs_acc_mean", "ffbs_acc_mcse", "ffbs_voi_mean",
                        "ffbs_voi_mcse")}
                      for r in metrics["joint"]["per_trace_rows"]],
    })

    verdict = classify(metrics)
    _dump("final_verdict.json", verdict)
    runtime = {
        "forward_backward_seconds": inference["t_forward"],
        "ffbs_seconds": inference["t_ffbs"],
        "ffbs_draws_total": inference["n_draws_total"],
        "ffbs_draws_per_second": inference["n_draws_total"]
        / inference["t_ffbs"],
        "wall_seconds_total": time.perf_counter() - wall_start,
    }
    _dump("runtime.json", runtime)

    b, l, j, pvp = (metrics["boundary"]["all"], metrics["labels"]["all"],
                    metrics["joint"]["all"],
                    metrics["prior_vs_posterior"]["all"])
    (OUT / "report.md").write_text("\n".join([
        "# Condition A — path identifiability under oracle structures",
        "",
        f"Source commit: `{_git('rev-parse', 'HEAD')}` &middot; corpus "
        f"`{recorded['corpus_hash_sha256'][:16]}…` &middot; truth "
        f"`{recorded['truth_hash_sha256'][:16]}…`",
        "",
        f"## Verdict: **{verdict['verdict']}**",
        "",
        "Exact per-trace semi-Markov posterior over (S, z); everything else "
        "fixed to generating truth. No MCMC.",
        "",
        "| metric (all traces) | posterior | prior |",
        "|---|---|---|",
        f"| boundary Brier | {b['brier']:.4f} | "
        f"{metrics['boundary_prior']['all']['brier']:.4f} |",
        f"| boundary NLL | {b['nll']:.4f} | "
        f"{metrics['boundary_prior']['all']['nll']:.4f} |",
        f"| boundary AUROC | {b['auroc']:.4f} | "
        f"{metrics['boundary_prior']['all']['auroc']:.4f} |",
        f"| boundary AUPRC | {b['auprc']:.4f} | "
        f"{metrics['boundary_prior']['all']['auprc']:.4f} |",
        f"| boundary ECE | {b['ece']:.4f} | "
        f"{metrics['boundary_prior']['all']['ece']:.4f} |",
        f"| occurrence mean p(c*) | "
        f"{l['mean_posterior_prob_true_label']:.4f} | "
        f"{metrics['labels_prior']['all']['mean_posterior_prob_true_label']:.4f} |",
        f"| occurrence modal accuracy | {l['modal_accuracy']:.4f} | "
        f"{metrics['labels_prior']['all']['modal_accuracy']:.4f} |",
        "",
        f"- boundary Brier reduction vs prior: "
        f"{pvp['boundary_brier']['relative_reduction']:.1%}; NLL reduction "
        f"{pvp['boundary_nll']['relative_reduction']:.1%}; occurrence NLL "
        f"reduction {pvp['occurrence_nll']['relative_reduction']:.1%}",
        f"- mean per-trace ARI {l['per_trace_ari']['mean']:.3f}, NMI "
        f"{l['per_trace_nmi']['mean']:.3f}",
        f"- MAP segment-count accuracy "
        f"{metrics['segment_counts']['map_accuracy']:.3f}; mean |E[L] - L*| "
        f"{metrics['segment_counts']['posterior_mean_abs_error']['mean']:.3f}",
        f"- MAP labelled path exactly equals truth on "
        f"{j['map_exact_recovery_rate']:.1%} of traces; median true-path "
        f"posterior {j['true_path_prob']['median']:.4f}; median true-"
        f"segmentation posterior {j['true_seg_prob']['median']:.4f}",
        f"- exact posterior path entropy mean {j['posterior_entropy']['mean']:.2f} "
        f"nats vs prior {metrics['joint']['all']['entropy_reduction_vs_prior']['mean'] + j['posterior_entropy']['mean']:.2f} "
        f"(mean reduction {j['entropy_reduction_vs_prior']['mean']:.2f} nats)",
        f"- FFBS: {FFBS_DRAWS_PER_TRACE} iid draws/trace, "
        f"{runtime['ffbs_draws_per_second']:.0f} draws/s, no burn-in, no "
        f"thinning, no R-hat; {consistency['ffbs_illegal_draws']} illegal draws",
        "",
        "Note on interpretation: with a zero-diagonal P the occurrence-label "
        "vector determines the labelled path bijectively, so p(c*-vector) = "
        "p(S*, z*). Low exact-path probability on long traces coexists with "
        "accurate marginals when near-equivalent paths share mass; the verdict "
        "rule was frozen in preregistration.json before inference.",
        "",
        "STOPPED as registered: no Condition B/C/D, no U/rho/scalar inference.",
    ]) + "\n")

    print(f"\nVERDICT: {verdict['verdict']}")
    print(f"artifacts in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
