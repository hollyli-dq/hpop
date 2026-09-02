#!/usr/bin/env python3
"""Open the sealed truth and score recovery -- ONLY for cells whose verdict is PASS.

Refuses to run unless the coordinator has written ALL_DONE, and scores only PASS cells:
a FAIL cell's recovery is undefined by design (inference FAIL is the result). Alignment
is deterministic Hungarian on closure agreement; metrics follow the registered list:

    structure:     macro closure F1, macro incomparability F1, exact-skill fraction
    segmentation:  boundary AUROC (posterior boundary frequency vs true boundaries)
    reuse:         occurrence ARI (co-assignment vs truth)
    cost:          pulled from the checkpoints (the machines already recorded it)

Held-out NLL uses the existing confirmatory machinery and is run separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_cpa.recovery_regime import REGIME                          # noqa: E402
from hpop.mcmc_original.latent_poset import precedence_from_u             # noqa: E402


def hungarian(cost: np.ndarray):
    from scipy.optimize import linear_sum_assignment
    return linear_sum_assignment(cost)


def closure_of(u_skill) -> np.ndarray:
    return np.asarray(precedence_from_u(np.asarray(u_skill, dtype=float)), dtype=bool)


def f1(true: np.ndarray, pred: np.ndarray) -> float:
    tp = int((true & pred).sum())
    fp = int((~true & pred).sum())
    fn = int((true & ~pred).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path,
                   default=ROOT / "dataset" / "recovery_scale_v1")
    p.add_argument("--work", type=Path, default=ROOT / "results" / "recovery_scale")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    work = args.work
    if not (work / "ALL_DONE").exists():
        raise SystemExit("REFUSED: coordinator has not written ALL_DONE; the truth "
                         "stays sealed until every cell's verdict is frozen")
    verdicts = json.loads((work / "verdicts.json").read_text())
    selection = json.loads((work / "sigma_selection.json").read_text())

    results = []
    for key, verdict in sorted(verdicts.items()):
        if verdict["final"] != "PASS":
            results.append({"cell": key, "verdict": "INFERENCE FAIL",
                            "recovery": None,
                            "note": "recovery undefined; failure of inference at this "
                                    "budget, not a model claim"})
            continue
        replicate = int(key.split("_")[0][3:])
        k = int(key.split("_K")[1])
        sigma = float(selection[str(k)]["sigma"])
        truth = json.loads((args.dataset / "truth" /
                            f"rep{replicate}_K{k}.json").read_text())
        true_closures = [closure_of(u) for u in truth["u_by_skill"]]

        # posterior mean closure per skill from the last half of U draws, all chains
        cell = work / "phaseB" / f"rep{replicate}_K{k}_s{sigma:g}"
        edge_prob = None
        n_draws = 0
        boundary_hits = defaultdict(float)
        co_pairs = []
        label_draws = []
        for chain in range(REGIME.CHAINS):
            files = sorted((cell / f"chain{chain}").glob("checkpoint_*.json"))
            u_draws, labels, bounds = [], [], []
            for f in files:
                payload = json.loads(f.read_text())
                u_draws.extend(payload["draws"]["u"])
                labels.extend(payload["draws"]["labels"])
                bounds.extend(payload["draws"]["boundaries"])
            u_draws = u_draws[len(u_draws) // 2:]
            labels = labels[len(labels) // 2:]
            bounds = bounds[len(bounds) // 2:]
            label_draws.append((labels, bounds))
            for u in u_draws:
                closures = np.stack([closure_of(s) for s in u])
                edge_prob = closures.astype(float) if edge_prob is None \
                    else edge_prob + closures
                n_draws += 1
        edge_prob /= max(n_draws, 1)

        # deterministic Hungarian alignment on closure disagreement
        K = k
        cost = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                cost[i, j] = np.abs(true_closures[i].astype(float)
                                    - edge_prob[j]).sum()
        rows, cols = hungarian(cost)
        aligned = {int(i): int(j) for i, j in zip(rows, cols)}

        closure_f1s, incomp_f1s, exact = [], [], 0
        for i in range(K):
            pred = edge_prob[aligned[i]] >= 0.5
            true = true_closures[i]
            closure_f1s.append(f1(true, pred))
            true_inc = ~(true | true.T) & ~np.eye(true.shape[0], dtype=bool)
            pred_inc = ~(pred | pred.T) & ~np.eye(true.shape[0], dtype=bool)
            incomp_f1s.append(f1(true_inc, pred_inc))
            exact += int(np.array_equal(true, pred))

        # boundary AUROC: posterior boundary frequency at each interior position
        seg_truth = truth["train_segmentations"]
        true_bounds = []
        for t in seg_truth:
            ends, pos = set(), 0
            for w in t["widths"][:-1]:
                pos += w
                ends.add(pos)
            true_bounds.append(ends)
        freq = defaultdict(float)
        total_draws = 0
        for labels, bounds in label_draws:
            for draw in bounds:
                total_draws += 1
                for trace_index, ends in enumerate(draw):
                    for e in ends:
                        freq[(trace_index, int(e))] += 1.0
        scores, ys = [], []
        n_traces = len(seg_truth)
        trace_length = REGIME.TRACE_LENGTH
        for trace_index in range(n_traces):
            for position in range(1, trace_length):
                scores.append(freq.get((trace_index, position), 0.0)
                              / max(total_draws, 1))
                ys.append(1 if position in true_bounds[trace_index] else 0)
        scores, ys = np.array(scores), np.array(ys)
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(scores))
        n_pos, n_neg = int(ys.sum()), int((1 - ys).sum())
        auroc = (ranks[ys == 1].sum() - n_pos * (n_pos - 1) / 2) / max(n_pos * n_neg, 1)

        # occurrence ARI against truth, on the modal assignment, aligned
        def ari(a, b):
            from math import comb
            contingency = defaultdict(int)
            for x, y in zip(a, b):
                contingency[(x, y)] += 1
            a_counts, b_counts = defaultdict(int), defaultdict(int)
            for (x, y), n in contingency.items():
                a_counts[x] += n
                b_counts[y] += n
            n_total = len(a)
            sum_ij = sum(comb(n, 2) for n in contingency.values())
            sum_a = sum(comb(n, 2) for n in a_counts.values())
            sum_b = sum(comb(n, 2) for n in b_counts.values())
            expected = sum_a * sum_b / max(comb(n_total, 2), 1)
            max_index = (sum_a + sum_b) / 2
            return (sum_ij - expected) / max(max_index - expected, 1e-12)

        true_occ = []
        for t in seg_truth:
            for w, s in zip(t["widths"], t["labels"]):
                true_occ.extend([s] * w)
        # modal per-position assignment from the last draws of chain 0
        labels, bounds = label_draws[0]
        pred_occ = []
        last_labels, last_bounds = labels[-1], bounds[-1]
        for trace_index in range(n_traces):
            ends = list(last_bounds[trace_index]) + [trace_length]
            start = 0
            for seg_index, end in enumerate(ends):
                pred_occ.extend([aligned_inv(aligned, last_labels[trace_index]
                                             [seg_index])] * (end - start))
                start = end
        occurrence_ari = ari(true_occ, pred_occ)

        results.append({"cell": key, "verdict": "PASS", "recovery": {
            "macro_closure_f1": float(np.mean(closure_f1s)),
            "macro_incomparability_f1": float(np.mean(incomp_f1s)),
            "exact_skill_fraction": exact / K,
            "boundary_auroc": float(auroc),
            "occurrence_ari": float(occurrence_ari),
        }})
        print(f"{key}: closure F1 {np.mean(closure_f1s):.3f}  "
              f"incomp F1 {np.mean(incomp_f1s):.3f}  exact {exact}/{K}  "
              f"AUROC {auroc:.3f}  ARI {occurrence_ari:.3f}")

    out = args.out or (work / "recovery_results.json")
    out.write_text(json.dumps({"schema": "recovery-scale-results/1.0.0",
                               "results": results}, indent=1, default=str))
    print(f"wrote {out}")
    return 0


def aligned_inv(aligned: dict, label: int) -> int:
    for true_skill, inferred in aligned.items():
        if inferred == label:
            return true_skill
    return label


if __name__ == "__main__":
    raise SystemExit(main())
