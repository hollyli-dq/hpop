"""Truth recovery for the confirmatory experiment. Runs ONLY after the truth-free commit.

    PYTHONPATH=src python scripts/confirmatory_recovery.py

Unseals truth under the frozen protocol, applies ONE common global label permutation to
H, z, pi and P, and reports the registered recovery metrics. Recovery is descriptive and
cannot revise the registered verdicts (FULL-COND = FAIL, FULL-MARG = FAIL).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_full_latent as mfl                     # noqa: E402
from hpop.mcmc_original import matched_synthetic_generator as msg             # noqa: E402
from hpop.mcmc_original.fast_block_tables import FastBlockScoreTable          # noqa: E402
from hpop.mcmc_original.latent_poset import precedence_from_u                 # noqa: E402
from hpop.mcmc_original.transitions import log_transition_matrix              # noqa: E402
from hpop.mcmc_optimized.forward import forward_batched_group                 # noqa: E402

CORP = ROOT / "results" / "mcmc_optimized" / "confirmatory_corpus"
RUN = ROOT / "results" / "mcmc_optimized" / "confirmatory_run"
OUT = RUN / "recovery"
NLL_PER_CHAIN = 1_000                       # frozen in PREREG section 10


def canonical(u) -> str:
    u = np.asarray(u, dtype=float)
    return hashlib.sha256(b"".join(sorted(
        np.ascontiguousarray(precedence_from_u(u[k])).tobytes()
        for k in range(u.shape[0])))).hexdigest()[:16]


def load_arm(tag):
    return [np.load(RUN / "chains" / f"{tag}_{i}.npz", allow_pickle=False)
            for i in range(4)]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ unseal
    truth = msg.truth_from_jsonable(json.loads((CORP / "truth_SEALED.json").read_text()))
    truth_hash = json.loads((CORP / "truth_SEAL.json").read_text())["truth_sha256"]
    draw_hashes = {f"{tag}_{i}": hashlib.sha256(
        (RUN / "chains" / f"{tag}_{i}.npz").read_bytes()).hexdigest()
        for tag in ("full_cond", "full_marg") for i in range(4)}
    (OUT / "TRUTH_UNSEAL.json").write_text(json.dumps({
        "unsealed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "truth_sha256": truth_hash,
        "authorised_by": "frozen protocol PREREG section 3: unsealing is permitted once, "
                         "after the terminal verdict is recorded",
        "truth_free_verdict_commit": "f9799f9f9f89508b09e6fd6a1ca45610bd2d9068",
        "registered_verdicts_at_unseal": {"FULL-COND": "FAIL", "FULL-MARG": "FAIL"},
        "terminal_draw_set_sha256": draw_hashes,
    }, indent=2, sort_keys=True) + "\n")

    K = truth.u_by_skill.shape[0]
    # Match `relation_indicator_vector`: off-diagonal (i, j) in fixed order, i != j.
    _m = truth.u_by_skill.shape[1]
    _off = [(i, j) for i in range(_m) for j in range(_m) if i != j]
    truth_H = np.array([[bool(np.asarray(truth.precedence(k))[i, j]) for i, j in _off]
                        for k in range(K)])          # (K, m(m-1))
    truth_library = canonical(truth.u_by_skill)

    # true segmentation labels per training trace
    train = np.load(CORP / "train_traces.npz", allow_pickle=False)
    n_train = int(train["n_traces"][0])
    true_labels, true_boundary = [], []
    for n in range(n_train):
        widths = np.asarray(train[f"t{n:03d}_widths"], dtype=int)
        labels = np.asarray(train[f"t{n:03d}_labels"], dtype=int)
        J = int(widths.sum())
        per_pos = np.concatenate([np.full(w, l) for w, l in zip(widths, labels)])
        true_labels.append(per_pos)
        b = np.zeros(J - 1, dtype=float)
        b[np.cumsum(widths)[:-1] - 1] = 1.0
        true_boundary.append(b)

    corpus = mfl.load_frozen_observed_corpus(CORP)
    fixed = mfl.FullLatentFixed()
    model = mfl.build_full_latent_model(corpus.train, fixed)
    probes = mfl.select_truth_free_probes(model.traces, corpus.corpus_hash)
    heldout = mfl.load_frozen_observed_corpus(CORP).heldout

    report = {"truth_hash_sha256": truth_hash, "truth_library": truth_library,
              "registered_verdicts": {"FULL-COND": "FAIL", "FULL-MARG": "FAIL"},
              "note": "descriptive; cannot revise the registered verdicts", "arms": {}}

    for arm, tag in (("FULL-COND", "full_cond"), ("FULL-MARG", "full_marg")):
        data = load_arm(tag)
        M = int(data[0]["u_draws"].shape[0])
        u = np.concatenate([d["u_draws"] for d in data])          # (4M, K, m, d)
        pi = np.concatenate([d["pi_draws"] for d in data])
        P = np.concatenate([d["p_draws"] for d in data])
        rel = np.concatenate([d["relation_indicators"] for d in data])
        total = u.shape[0]

        # ---- exact unordered-library recovery (permutation-invariant) ----
        libs = np.array([canonical(u[i]) == truth_library for i in range(total)])
        p_library = float(libs.mean())

        # ---- one common global alignment over all six permutations ----
        m_roles = truth.u_by_skill.shape[1]
        per_perm = []
        for perm in itertools.permutations(range(K)):
            pm = np.array(perm)
            H_post = rel.reshape(total, K, m_roles * (m_roles - 1))[:, pm, :].mean(axis=0)
            H_true = truth_H.astype(float)
            score = -np.abs(H_post - H_true).sum()            # closure agreement
            score += -np.abs(pi[:, pm].mean(axis=0) - truth.pi).sum()
            per_perm.append({"perm": list(perm), "score": float(score)})
        best = max(per_perm, key=lambda r: r["score"])
        pm = np.array(best["perm"])

        # ---- closure F1 / Hamming under the common alignment ----
        H_prob = rel.reshape(total, K, m_roles * (m_roles - 1))[:, pm, :].mean(axis=0)
        H_mode = H_prob >= 0.5
        H_true_flat = truth_H
        tp = float((H_mode & H_true_flat).sum())
        fp = float((H_mode & ~H_true_flat).sum())
        fn = float((~H_mode & H_true_flat).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-12)
        hamming = int((H_mode != H_true_flat).sum())
        rel_prob_true = float(H_prob[H_true_flat].mean()) if H_true_flat.any() else float("nan")

        # ---- supplementary: per-draw alignment, to separate structure recovery from
        # ---- within-chain label switching. NOT the registered metric.
        blocks = rel.reshape(total, K, m_roles * (m_roles - 1))
        perms = list(itertools.permutations(range(K)))
        best_per_draw = np.empty(total, dtype=np.int8)
        exact_per_draw = np.zeros(total, dtype=bool)
        ham_per_draw = np.empty(total, dtype=np.int16)
        for i in range(total):
            hs = [int((blocks[i][list(q)] != truth_H).sum()) for q in perms]
            j = int(np.argmin(hs))
            best_per_draw[i] = j
            ham_per_draw[i] = hs[j]
            exact_per_draw[i] = hs[j] == 0
        switch_counts = np.bincount(best_per_draw, minlength=len(perms)).tolist()
        dominant = float(max(switch_counts)) / total
        per_chain_perm = []
        for c in range(4):
            seg = best_per_draw[c * M:(c + 1) * M]
            vals = np.unique(seg)
            per_chain_perm.append({"chain": c,
                                   "permutations_used": [list(perms[v]) for v in vals],
                                   "constant_within_chain": bool(len(vals) == 1)})
        within_chain_switching = any(not r["constant_within_chain"]
                                     for r in per_chain_perm)

        # ---- boundary Brier ----
        nb = sum(len(b) for b in true_boundary)
        brier_b = 0.0
        for n in range(n_train):
            post = np.sum([d[f"boundary__{n:03d}"] for d in data], axis=0) / total
            brier_b += float(((post - true_boundary[n]) ** 2).sum())
        brier_b /= nb

        # ---- co-skill Brier over the 256 recovery probes ----
        cos = np.sum([d["recovery_coskill_sums"] for d in data], axis=0) / total
        truth_cos = np.array([float(true_labels[n][a] == true_labels[n][b])
                              for n, a, b in probes["recovery_coskill"]])
        brier_c = float(((cos - truth_cos) ** 2).mean())

        # ---- pi and P under the same alignment ----
        pi_al = pi[:, pm]
        P_al = P[:, pm][:, :, pm]
        pi_mean = pi_al.mean(axis=0)
        P_mean = P_al.mean(axis=0)
        pi_tv = float(0.5 * np.abs(pi_mean - truth.pi).sum())
        pi_rmse = float(np.sqrt(((pi_mean - truth.pi) ** 2).mean()))
        P_fro = float(np.linalg.norm(P_mean - truth.transition))
        off = ~np.eye(K, dtype=bool)
        P_off_rmse = float(np.sqrt(((P_mean[off] - truth.transition[off]) ** 2).mean()))
        mcse = lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))
        report_arm = {
            "alignment": {"chosen_permutation": best["perm"],
                          "all_six": per_perm,
                          "criterion": "closure agreement + pi agreement; one permutation "
                                       "applied jointly to H, z, pi and P"},
            "library_recovery": {"posterior_probability_exact_true_library": p_library,
                                 "truth_library": truth_library},
            "closure": {"f1": f1, "hamming": hamming,
                        "mean_posterior_prob_of_true_relations": rel_prob_true},
            "supplementary_per_draw_alignment": {
                "note": "NOT the registered metric. Aligns each draw separately, which "
                        "removes within-chain label switching that one global permutation "
                        "cannot undo.",
                "posterior_probability_exact_labelled_closure": float(exact_per_draw.mean()),
                "mean_closure_hamming_per_draw": float(ham_per_draw.mean()),
                "best_permutation_counts": switch_counts,
                "dominant_permutation_share": dominant,
                "per_chain_permutation": per_chain_perm,
                "within_chain_label_switching": within_chain_switching,
                "between_chain_label_switching": bool(dominant < 0.999
                                                      and not within_chain_switching)},
            "boundary_brier": brier_b, "coskill_brier": brier_c,
            "pi": {"tv": pi_tv, "rmse": pi_rmse,
                   "mcse_components": [mcse(pi_al[:, k]) for k in range(K)]},
            "P": {"frobenius": P_fro, "off_diagonal_rmse": P_off_rmse},
            "n_draws": total,
        }
        report["arms"][arm] = report_arm
        print(f"{arm}: P(true library) {p_library:.4f}  closure F1 {f1:.4f}  "
              f"Hamming {hamming}  boundary Brier {brier_b:.5f}  "
              f"co-skill Brier {brier_c:.5f}  pi TV {pi_tv:.5f}  P Fro {P_fro:.5f}",
              flush=True)

    (OUT / "recovery.json").write_text(json.dumps(report, indent=2, sort_keys=True,
                                                  default=str) + "\n")
    print(f"\nwrote {OUT/'recovery.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
