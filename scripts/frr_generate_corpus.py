"""Full Recurrent Release — fresh sealed corpus with recurrent-excitation admissibility.

    PYTHONPATH=src python scripts/frr_generate_corpus.py --design-pilot
    PYTHONPATH=src python scripts/frr_generate_corpus.py --formal

Two phases, deliberately separated.

`--design-pilot` draws candidate truths from a SEPARATE seed family and reports the
distribution of the five excitation statistics. It is not part of the formal result and
its draws are discarded. Its only purpose is to show the thresholds are feasible.

`--formal` draws the formal truth under thresholds already frozen in
EXCITATION_THRESHOLDS below, and writes the sealed corpus.

Excitation exists because the four recurrent parameters are only identifiable if the
training corpus actually exercises them:

  A  progress variation  (beta)        decisions with >= 2 feasible roles whose Q differ
  B  repetition          (lambda_rep)  selected roles with q_{u-1}(y_u) > 0
  C  backward disruption (lambda_back) selected roles with C_back_u(y_u) > 0
  D  partial invalidation(omega)       downstream validity knocked down by re-execution
  E  skill coverage                    minimum instances and occurrences per skill

Each uses only generating truth, generated latent paths and generated recurrent states.
None uses an inferred parameter, a posterior, a recovery metric, convergence, held-out
prediction or a method comparison.

TRUTH IS SEALED: the formal phase prints hashes only, never a truth value.
"""

from __future__ import annotations

import argparse
import hashlib
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
from hpop.mcmc_original.recurrent_joint_scalar_mcmc import (                   # noqa: E402
    vectorized_state_features)
from hpop.mcmc_original.recurrent_rfs import recurrent_rfs_log_likelihood      # noqa: E402
from hpop.mcmc_original.recurrent_segmentation import RecurrentBlockScorer     # noqa: E402

OUT = ROOT / "results" / "full_recurrent_release" / "corpus"
PILOT_OUT = ROOT / "results" / "full_recurrent_release" / "pilots"

# ------------------------------------------------------------- frozen design
MASTER_SEED = 6_310_001            # corpus draw
TRUTH_SEED = 6_310_002             # formal truth draw
DESIGN_PILOT_SEED = 6_319_001      # SEPARATE family; discarded
DESIGN_PILOT_DRAWS = 40
PINNED_RHO = 0.5
MAX_TRUTH_ATTEMPTS = 200
N_TRAIN, N_HELDOUT = 100, 45
DESIGN_J = (24, 32, 40, 48)
TRACE_LENGTHS_TRAIN = tuple(DESIGN_J[i % 4] for i in range(N_TRAIN))
TRACE_LENGTHS_HELDOUT = tuple(DESIGN_J[i % 4] for i in range(N_HELDOUT))

# Structural admissibility, unchanged from the confirmatory corpus.
# The strict-partial-order maximum is m(m-1)/2, not m(m-1).
# EXCITATION thresholds are filled in from the design pilot and frozen BEFORE --formal.
EXCITATION_THRESHOLDS = json.loads(
    (Path(__file__).parent / "frr_excitation_thresholds.json").read_text()
) if (Path(__file__).parent / "frr_excitation_thresholds.json").exists() else None


def draw_truth(seed: int) -> msg.MatchedTruth:
    """Prior draw at pinned rho; identical construction to the confirmatory corpus."""
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(msg.sigma_rho_matrix(msg.LATENT_DIM, PINNED_RHO))
    u = np.array([[chol @ rng.standard_normal(msg.LATENT_DIM)
                   for _ in range(msg.N_ROLES)] for _ in range(msg.N_SKILLS)])
    pi = rng.dirichlet(np.full(msg.N_SKILLS, float(msg.ETA_INITIAL)))
    transition = np.zeros((msg.N_SKILLS, msg.N_SKILLS))
    for h in range(msg.N_SKILLS):
        allowed = msg.allowed_next(h, msg.N_SKILLS)
        for k, v in zip(allowed, rng.dirichlet(np.full(len(allowed),
                                                       float(msg.ETA_TRANSITION)))):
            transition[h, k] = v
    s = dict(msg.TRUE_VALUES)
    return msg.MatchedTruth(
        u_by_skill=u, pi=pi, transition=transition, beta=s["beta"], omega=s["omega"],
        lambda_rep=s["lambda_rep"], lambda_back=s["lambda_back"], epsilon=0.02,
        delta_b=msg.DELTA_B, min_width=msg.MIN_BLOCK_WIDTH, max_width=msg.MAX_BLOCK_WIDTH,
        role_maps=tuple(tuple(range(msg.N_ROLES)) for _ in range(msg.N_SKILLS)),
        rho=PINNED_RHO, mode="prior_draw_pinned_rho")


def structural_ok(truth) -> tuple[bool, dict]:
    try:
        msg.validate_truth(truth)
    except AssertionError as exc:
        return False, {"reason": f"validate_truth: {exc}"}
    m = truth.u_by_skill.shape[1]
    cap = m * (m - 1) // 2
    closures, counts = [], []
    for k in range(truth.u_by_skill.shape[0]):
        p = np.asarray(truth.precedence(k), dtype=bool)
        closures.append(p.tobytes())
        counts.append(int(p.sum()))
    distinct = len(set(closures)) == len(closures)
    nondeg = all(0 < c < cap for c in counts)
    return distinct and nondeg, {"pairwise_distinct": distinct, "nondegenerate": nondeg}


def excitation_stats(truth, train_traces) -> dict:
    """The five excitation statistics, from truth and the generated latent paths only."""
    K = truth.u_by_skill.shape[0]
    by_skill: dict = {k: [] for k in range(K)}
    instances = Counter()
    occurrences = Counter()
    for t in train_traces:
        for block, skill in zip(t.role_blocks, t.labels):
            by_skill[int(skill)].append(list(block))
            instances[int(skill)] += 1
            occurrences[int(skill)] += len(block)

    a_var = a_tot = b_rep = c_back = d_inval = tot_steps = 0
    for k in range(K):
        blocks = by_skill[k]
        if not blocks:
            continue
        width = max(len(b) for b in blocks)
        padded = np.array([b + [b[-1]] * (width - len(b)) for b in blocks], dtype=int)
        f = vectorized_state_features(padded, truth.u_by_skill[k], truth.omega)
        F, Q, q, CB, obs = f["F"], f["Q"], f["q"], f["C_back"], f["obs"]
        for n, blk in enumerate(blocks):
            T = len(blk)
            for u_i in range(T):
                tot_steps += 1
                feas = F[n, u_i] > 0
                if feas.sum() >= 2 and np.ptp(Q[n, u_i][feas]) > 1e-9:
                    a_var += 1
                a_tot += 1
                y = obs[n, u_i]
                if q[n, u_i, y] > 0:
                    b_rep += 1
                if CB[n, u_i, y] > 0:
                    c_back += 1
                if u_i > 0 and q[n, u_i, y] > 0:
                    # re-executing y knocks down every strict successor that was valid
                    succ = np.asarray(truth.precedence(k))[y]
                    if np.any(succ & (q[n, u_i] > 0)):
                        d_inval += 1
    return {
        "A_progress_variation_frac": a_var / max(a_tot, 1),
        "A_progress_variation_count": int(a_var),
        "B_repetition_frac": b_rep / max(tot_steps, 1),
        "B_repetition_count": int(b_rep),
        "C_backward_disruption_frac": c_back / max(tot_steps, 1),
        "C_backward_disruption_count": int(c_back),
        "D_partial_invalidation_count": int(d_inval),
        "E_min_instances_per_skill": int(min(instances[k] for k in range(K))),
        "E_min_occurrences_per_skill": int(min(occurrences[k] for k in range(K))),
        "total_steps": int(tot_steps),
    }


def build_corpus(truth, master_seed):
    return msg.generate_corpus(master_seed, TRACE_LENGTHS_TRAIN,
                               TRACE_LENGTHS_HELDOUT, truth)


def design_pilot() -> int:
    PILOT_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(DESIGN_PILOT_DRAWS):
        truth = draw_truth(DESIGN_PILOT_SEED + i)
        ok, detail = structural_ok(truth)
        if not ok:
            rows.append({"draw": i, "structural_ok": False, **detail})
            continue
        corpus = build_corpus(truth, DESIGN_PILOT_SEED + 500_000 + i)
        rows.append({"draw": i, "structural_ok": True,
                     **excitation_stats(truth, corpus.train)})
    ok_rows = [r for r in rows if r.get("structural_ok")]
    keys = ["A_progress_variation_frac", "B_repetition_frac",
            "C_backward_disruption_frac", "D_partial_invalidation_count",
            "E_min_instances_per_skill", "E_min_occurrences_per_skill"]
    summary = {k: {"min": float(np.min([r[k] for r in ok_rows])),
                   "p10": float(np.percentile([r[k] for r in ok_rows], 10)),
                   "median": float(np.median([r[k] for r in ok_rows])),
                   "max": float(np.max([r[k] for r in ok_rows]))} for k in keys}
    (PILOT_OUT / "design_pilot_excitation.json").write_text(json.dumps(
        {"seed_family": DESIGN_PILOT_SEED, "draws": DESIGN_PILOT_DRAWS,
         "structurally_admissible": len(ok_rows), "per_draw": rows,
         "summary": summary,
         "note": "design pilot only; discarded, not part of the formal result"},
        indent=2, sort_keys=True) + "\n")
    print(f"design pilot: {len(ok_rows)}/{DESIGN_PILOT_DRAWS} structurally admissible")
    for k in keys:
        s = summary[k]
        print(f"  {k:<36} min {s['min']:.4g}  p10 {s['p10']:.4g}  "
              f"median {s['median']:.4g}  max {s['max']:.4g}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design-pilot", action="store_true")
    ap.add_argument("--formal", action="store_true")
    args = ap.parse_args()
    if args.design_pilot:
        return design_pilot()
    if not args.formal:
        raise SystemExit("choose --design-pilot or --formal")
    if EXCITATION_THRESHOLDS is None:
        raise SystemExit("frr_excitation_thresholds.json missing: thresholds must be "
                         "frozen before the formal draw")

    OUT.mkdir(parents=True, exist_ok=True)
    attempts, log, truth, stats = 0, [], None, None
    for offset in range(MAX_TRUTH_ATTEMPTS):
        attempts = offset + 1
        cand = draw_truth(TRUTH_SEED + offset)
        ok, detail = structural_ok(cand)
        if not ok:
            log.append({"attempt": attempts, "accepted": False,
                        "reason": {"structural": detail}})
            continue
        corpus = build_corpus(cand, MASTER_SEED)
        st = excitation_stats(cand, corpus.train)
        fails = {k: [st[k], v] for k, v in EXCITATION_THRESHOLDS.items() if st[k] < v}
        if fails:
            log.append({"attempt": attempts, "accepted": False,
                        "reason": {"excitation_below_threshold": fails}})
            continue
        truth, stats = cand, st
        log.append({"attempt": attempts, "accepted": True, "reason": None})
        break
    if truth is None:
        raise SystemExit(f"no truth met the frozen admissibility event in "
                         f"{MAX_TRUTH_ATTEMPTS} attempts; thresholds NOT lowered")

    corpus = build_corpus(truth, MASTER_SEED)
    digest = msg.corpus_hash(corpus)
    truth_json = msg.canonical_json(msg.truth_to_jsonable(truth))
    truth_hash = msg.sha256_hex(truth_json)
    traces = corpus.train + corpus.heldout

    checks = {"illegal_width": 0, "cover_mismatch": 0, "self_transition": 0,
              "nonfinite_block_ll": 0}
    for t in traces:
        for w in t.widths:
            if not (truth.min_width <= w <= truth.max_width):
                checks["illegal_width"] += 1
        if sum(t.widths) != t.length or len(t.cpa) != t.length:
            checks["cover_mismatch"] += 1
        for a, b in zip(t.labels[:-1], t.labels[1:]):
            if a == b:
                checks["self_transition"] += 1
        if not all(math.isfinite(v) for v in t.block_log_likelihoods):
            checks["nonfinite_block_ll"] += 1
    q0 = max(abs(recurrent_rfs_log_likelihood(
        blk, truth.u_by_skill[s], truth.beta, truth.epsilon, truth.omega,
        truth.lambda_rep, truth.lambda_back) - rec)
        for t in traces for blk, s, rec in zip(t.role_blocks, t.labels,
                                               t.block_log_likelihoods))
    scorer = RecurrentBlockScorer(
        traces=tuple(t.cpa for t in traces), epsilon=truth.epsilon,
        u_by_skill=truth.u_by_skill, beta=truth.beta, omega=truth.omega,
        lambda_rep=truth.lambda_rep, lambda_back=truth.lambda_back,
        max_width=truth.max_width, min_width=truth.min_width)
    log_c = {J: mgd.exact_normalizer(J, truth.delta_b, truth.min_width, truth.max_width)
             for J in set(DESIGN_J)}
    parity = max(abs(msg.generator_complete_data_log_prob(t)
                     - msg.inference_complete_data_log_prob(t, truth, scorer, i,
                                                            log_c[t.length]))
                 for i, t in enumerate(traces))
    if not (all(v == 0 for v in checks.values()) and q0 < 1e-10 and parity < 1e-10):
        raise SystemExit(f"corpus validation failed: {checks} q0={q0} parity={parity}")

    def to_npz(tr, path):
        payload = {}
        for t in tr:
            tag = f"t{t.trace_index:03d}"
            payload[f"{tag}_cpa"] = np.asarray(t.cpa, dtype=np.int8)
            payload[f"{tag}_widths"] = np.asarray(t.widths, dtype=np.int16)
            payload[f"{tag}_boundaries"] = np.asarray(t.boundaries, dtype=np.int16)
            payload[f"{tag}_labels"] = np.asarray(t.labels, dtype=np.int8)
            payload[f"{tag}_block_ll"] = np.asarray(t.block_log_likelihoods, dtype=float)
            payload[f"{tag}_logs"] = np.asarray([t.log_seg_prior, t.log_label_prior])
        payload["n_traces"] = np.asarray([len(tr)])
        np.savez_compressed(path, **payload)

    to_npz(corpus.train, OUT / "train_traces.npz")
    to_npz(corpus.heldout, OUT / "heldout_traces.npz")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (OUT / "truth_SEALED.json").write_text(truth_json + "\n")
    (OUT / "corpus_hash.json").write_text(json.dumps({
        "corpus_hash_sha256": digest, "truth_hash_sha256": truth_hash,
        "train_npz_sha256": sha(OUT / "train_traces.npz"),
        "heldout_npz_sha256": sha(OUT / "heldout_traces.npz")},
        indent=2, sort_keys=True) + "\n")
    (OUT / "config.json").write_text(json.dumps({
        **corpus.config, "n_train_traces": N_TRAIN, "n_heldout_traces": N_HELDOUT,
        "master_seed": MASTER_SEED, "truth_seed_base": TRUTH_SEED,
        "generator_commit": "8ca828153e8e263bf4ea4823e45a53fa454037ad",
        "purpose": "FULL RECURRENT RELEASE sealed corpus"},
        indent=2, sort_keys=True, default=str) + "\n")
    (OUT / "truth_SEAL.json").write_text(json.dumps({
        "truth_sha256": truth_hash, "sealed": True,
        "seal_rule": "opened only after the truth-free full-joint terminal report is "
                     "committed",
        "truth_seed_base": TRUTH_SEED, "attempts_used": attempts,
        "attempt_log": log, "rejections": [r for r in log if not r["accepted"]],
        "frozen_excitation_thresholds": EXCITATION_THRESHOLDS,
        "excitation_of_accepted_truth_SEALED": stats,
        "validation": {"section4": checks, "q0": q0, "parity": parity}},
        indent=2, sort_keys=True) + "\n")
    print("FULL RECURRENT RELEASE CORPUS FROZEN")
    print(f"  corpus_hash = {digest}")
    print(f"  truth_hash  = {truth_hash}   [SEALED]")
    print(f"  attempts    = {attempts}, rejections = {len(log)-1}")
    print(f"  validation  = {checks}, q0={q0:.2e}, parity={parity:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
