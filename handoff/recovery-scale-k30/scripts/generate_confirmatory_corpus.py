"""Freeze the CONFIRMATORY matched-synthetic corpus for the optimized FULL-LATENT run.

Run:  PYTHONPATH=src python scripts/generate_confirmatory_corpus.py

This mirrors `generate_matched_formal_corpus.py`, which is sealed
(Condition C `SHARED_SOURCES` -> `formal_corpus_loader`) and therefore not edited.
Three things differ, each deliberate and preregistered here BEFORE generation:

1. A NEW master seed, 6_300_001, verified unused across code, results manifests, both
   worktrees and project memory. The 6_2xx_xxx band is exhausted by the old programme.

2. The truth is DRAWN FROM THE REGISTERED PRIOR, not supplied. `supplied_truth()` is a
   hardcoded configuration and is the truth of the terminated run, which was UNSEALED on
   2026-08-22 (`TRUTH_UNSEAL_midrun.json`). Reusing it would make a "sealed" confirmatory
   experiment sealed in name only -- the answer is already known to the analyst.

3. `rho` is PINNED at 0.5, not drawn. `sample_prior_truth` draws rho ~ U(0, 0.995), but
   FULL-LATENT inference holds `FIXED_RHO_0 = 0.5`. Pinning makes the generative prior for
   U exactly the prior the sampler assumes, removing a misspecification that an
   unconstrained draw would introduce silently. Everything else -- pi ~ Dir(1),
   P rows ~ Dir(1) off-diagonal, scalars at registered TRUE_VALUES -- is the registered
   prior unchanged.

Preregistered acceptance rule for the drawn truth. The draw is accepted iff
`validate_truth` passes AND the three induced partial orders are pairwise distinct AND
each is nondegenerate (at least one relation, and not a total order on the 5 roles).
On rejection the truth seed increments by 1 and the draw repeats, up to 100 attempts.
The rule and the attempt cap are fixed here before the first draw; the number of attempts
used is recorded. This preserves the controlled configuration class of the old programme
-- three distinct nondegenerate induced orders -- while leaving the specific structure
unknown.

TRUTH IS SEALED. This script writes the truth to `truth_SEALED.json` and prints only its
hash. No truth value is printed, returned or logged.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_generator_diagnostics as mgd            # noqa: E402
from hpop.mcmc_original import matched_synthetic_generator as msg              # noqa: E402
from hpop.mcmc_original.recurrent_rfs import recurrent_rfs_log_likelihood      # noqa: E402
from hpop.mcmc_original.recurrent_segmentation import RecurrentBlockScorer     # noqa: E402

OUT = ROOT / "results" / "mcmc_optimized" / "confirmatory_corpus"

# --------------------------------------------------------------- frozen design
MASTER_SEED = 6_300_001          # corpus draw
TRUTH_SEED = 6_300_002           # truth draw (separate stream)
PINNED_RHO = 0.5                 # == matched_full_latent.FIXED_RHO_0
MAX_TRUTH_ATTEMPTS = 100
N_TRAIN, N_HELDOUT = 100, 45
DESIGN_J = (24, 32, 40, 48)
TRACE_LENGTHS_TRAIN = tuple(DESIGN_J[i % 4] for i in range(N_TRAIN))
TRACE_LENGTHS_HELDOUT = tuple(DESIGN_J[i % 4] for i in range(N_HELDOUT))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _draw_truth_at_pinned_rho(seed: int) -> msg.MatchedTruth:
    """`sample_prior_truth`, with rho pinned instead of drawn."""
    rng = np.random.default_rng(seed)
    sigma = msg.sigma_rho_matrix(msg.LATENT_DIM, PINNED_RHO)
    chol = np.linalg.cholesky(sigma)
    u = np.array([[chol @ rng.standard_normal(msg.LATENT_DIM)
                   for _ in range(msg.N_ROLES)] for _ in range(msg.N_SKILLS)],
                 dtype=float)
    pi = rng.dirichlet(np.full(msg.N_SKILLS, float(msg.ETA_INITIAL)))
    transition = np.zeros((msg.N_SKILLS, msg.N_SKILLS))
    for h in range(msg.N_SKILLS):
        allowed = msg.allowed_next(h, msg.N_SKILLS)
        draw = rng.dirichlet(np.full(len(allowed), float(msg.ETA_TRANSITION)))
        for k, value in zip(allowed, draw):
            transition[h, k] = value
    scalars = dict(msg.TRUE_VALUES)
    return msg.MatchedTruth(
        u_by_skill=u, pi=pi, transition=transition,
        beta=scalars["beta"], omega=scalars["omega"],
        lambda_rep=scalars["lambda_rep"], lambda_back=scalars["lambda_back"],
        epsilon=0.02, delta_b=msg.DELTA_B,
        min_width=msg.MIN_BLOCK_WIDTH, max_width=msg.MAX_BLOCK_WIDTH,
        role_maps=tuple(tuple(range(msg.N_ROLES)) for _ in range(msg.N_SKILLS)),
        rho=PINNED_RHO, mode="prior_draw_pinned_rho")


def _accept(truth) -> tuple[bool, dict]:
    """The preregistered acceptance rule. Returns (accepted, sealed_detail)."""
    try:
        msg.validate_truth(truth)
    except AssertionError as exc:
        return False, {"reason": f"validate_truth: {exc}"}
    m = truth.u_by_skill.shape[1]
    # A STRICT partial order is antisymmetric, so its closure holds at most one of
    # (i, j) / (j, i) for each unordered pair: the maximum is m(m-1)/2, not m(m-1).
    # The earlier bound m(m-1) was vacuous -- no strict partial order can reach it --
    # so it excluded nothing. m(m-1)/2 is attained exactly by a total order.
    total_pairs = m * (m - 1) // 2
    closures, counts = [], []
    for k in range(truth.u_by_skill.shape[0]):
        p = np.asarray(truth.precedence(k), dtype=bool)
        closures.append(p.tobytes())
        counts.append(int(p.sum()))
    distinct = len(set(closures)) == len(closures)
    nondegenerate = all(0 < c < total_pairs for c in counts)
    return (distinct and nondegenerate,
            {"pairwise_distinct": distinct, "nondegenerate": nondegenerate,
             "relations_per_skill": counts})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    truth = None
    attempts = 0
    acceptance_detail = None
    attempt_log = []
    for offset in range(MAX_TRUTH_ATTEMPTS):
        attempts = offset + 1
        candidate = _draw_truth_at_pinned_rho(TRUTH_SEED + offset)
        accepted, detail = _accept(candidate)
        attempt_log.append({"attempt": attempts, "truth_seed": TRUTH_SEED + offset,
                            "accepted": bool(accepted),
                            "rejection_reason": None if accepted else detail})
        if accepted:
            truth, acceptance_detail = candidate, detail
            break
    if truth is None:
        raise SystemExit(f"no truth satisfied the acceptance rule in "
                         f"{MAX_TRUTH_ATTEMPTS} attempts")
    truth_checks = msg.validate_truth(truth)

    corpus = msg.generate_corpus(MASTER_SEED, TRACE_LENGTHS_TRAIN,
                                 TRACE_LENGTHS_HELDOUT, truth)
    digest = msg.corpus_hash(corpus)
    truth_json = msg.canonical_json(msg.truth_to_jsonable(truth))
    truth_hash = msg.sha256_hex(truth_json)
    traces = corpus.train + corpus.heldout

    # --------------------------------------------- the same Section 4 checks
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

    q0_worst = 0.0
    for t in traces:
        for block, skill, recorded in zip(t.role_blocks, t.labels,
                                          t.block_log_likelihoods):
            replay = recurrent_rfs_log_likelihood(
                block, truth.u_by_skill[skill], truth.beta, truth.epsilon,
                truth.omega, truth.lambda_rep, truth.lambda_back)
            q0_worst = max(q0_worst, abs(replay - recorded))

    scorer = RecurrentBlockScorer(
        traces=tuple(t.cpa for t in traces), epsilon=truth.epsilon,
        u_by_skill=truth.u_by_skill, beta=truth.beta, omega=truth.omega,
        lambda_rep=truth.lambda_rep, lambda_back=truth.lambda_back,
        max_width=truth.max_width, min_width=truth.min_width)
    log_c = {J: mgd.exact_normalizer(J, truth.delta_b, truth.min_width,
                                     truth.max_width) for J in set(DESIGN_J)}
    parity = max(abs(msg.generator_complete_data_log_prob(t)
                     - msg.inference_complete_data_log_prob(
                         t, truth, scorer, i, log_c[t.length]))
                 for i, t in enumerate(traces))

    def _traces_to_npz(tr, path):
        payload = {}
        for t in tr:
            tag = f"t{t.trace_index:03d}"
            payload[f"{tag}_cpa"] = np.asarray(t.cpa, dtype=np.int8)
            payload[f"{tag}_widths"] = np.asarray(t.widths, dtype=np.int16)
            payload[f"{tag}_boundaries"] = np.asarray(t.boundaries, dtype=np.int16)
            payload[f"{tag}_labels"] = np.asarray(t.labels, dtype=np.int8)
            payload[f"{tag}_block_ll"] = np.asarray(t.block_log_likelihoods,
                                                    dtype=np.float64)
            payload[f"{tag}_logs"] = np.asarray(
                [t.log_seg_prior, t.log_label_prior], dtype=np.float64)
        payload["n_traces"] = np.asarray([len(tr)])
        np.savez_compressed(path, **payload)

    _traces_to_npz(corpus.train, OUT / "train_traces.npz")
    _traces_to_npz(corpus.heldout, OUT / "heldout_traces.npz")

    passed = (all(v == 0 for v in checks.values())
              and q0_worst < 1e-10 and parity < 1e-10)
    if not passed:
        raise SystemExit(f"confirmatory corpus failed validation: {checks}, "
                         f"q0={q0_worst}, parity={parity}")

    # ------------------------------------------------------------ SEALED truth
    (OUT / "truth_SEALED.json").write_text(truth_json + "\n")
    (OUT / "truth_SEAL.json").write_text(json.dumps({
        "truth_sha256": truth_hash,
        "sealed": True,
        "seal_rule": "opened only at formal termination of the confirmatory run",
        "truth_seed_base": TRUTH_SEED,
        "truth_attempts_used": attempts,
        "attempt_log": attempt_log,
        "rejections": [a for a in attempt_log if not a["accepted"]],
        "outcome": ("the first prior draw satisfied the preregistered admissibility "
                    "conditions" if attempts == 1 else
                    f"admissible truth found on attempt {attempts}"),
        "admissibility_criteria": [
            "validate_truth: K >= 2",
            "validate_truth: each induced relation is a strict partial order "
            "(irreflexive, transitively closed)",
            "validate_truth: pi is a length-K probability vector",
            "validate_truth: P nonnegative, exactly zero diagonal, rows sum to 1",
            "validate_truth: all scalars inside their registered support",
            "validate_truth: 0 < delta_b < 1",
            "validate_truth: 1 <= min_width <= max_width",
            "validate_truth: rho strictly inside (RHO_LOWER, RHO_UPPER)",
            "validate_truth: every role map injective over m roles",
            "structural: the three induced closures are pairwise distinct",
            "structural: each closure has >= 1 relation (not empty)",
            "structural: each closure has < m(m-1)/2 relations (not a total order); "
            "m(m-1)/2 is the maximum for a strict partial order and is attained only "
            "by a total order",
        ],
        "criteria_use_no_data": (
            "Every criterion above is a function of the truth parameters alone. None "
            "evaluates a likelihood, a recovery metric, a convergence diagnostic, a "
            "held-out prediction, or any comparison between FULL-COND and FULL-MARG. "
            "No corpus is drawn until a truth is admitted, so no criterion can depend "
            "on realised data."),
        "acceptance_rule": "validate_truth passes AND the three induced partial "
                           "orders are pairwise distinct AND each has at least one "
                           "relation and is not a total order",
        "acceptance_detail_SEALED": acceptance_detail,
        "mode": truth.mode, "rho_pinned": PINNED_RHO,
    }, indent=2, sort_keys=True) + "\n")

    total_blocks = sum(t.n_segments for t in traces)
    config = {
        **corpus.config,
        "purpose": "CONFIRMATORY matched-synthetic corpus for the optimized "
                   "FULL-LATENT experiment. Frozen before any inference was run.",
        "design": {"n_train": N_TRAIN, "n_heldout": N_HELDOUT,
                   "trace_lengths_train": list(TRACE_LENGTHS_TRAIN),
                   "trace_lengths_heldout": list(TRACE_LENGTHS_HELDOUT),
                   "design_rule": "identical to the terminated programme: train cycles "
                                  "(24,32,40,48) 25 times; heldout cycles 45 times"},
        "master_seed": MASTER_SEED,
        "truth_seed_base": TRUTH_SEED,
        "truth_mode": truth.mode,
        "rho_pinned_to": PINNED_RHO,
        "seed_provenance": "6_300_001 / 6_300_002 verified unused across code, results "
                           "manifests, both worktrees and project memory; the 6_2xx_xxx "
                           "band is exhausted by the terminated programme",
        "differs_from_terminated_programme": [
            "new master seed",
            "truth drawn from the registered prior rather than supplied, because the "
            "supplied truth was unsealed on 2026-08-22",
            "rho pinned at 0.5 to match FIXED_RHO_0 rather than drawn from U(0, 0.995)",
        ],
        "generator_commit": "8ca828153e8e263bf4ea4823e45a53fa454037ad",
        "backend_commit": "564995efd056d7d33984f0ca1532386e6140ea0c",
        "source_commit": _git("rev-parse", "HEAD"),
        "corpus_hash_sha256": digest,
        "truth_hash_sha256": truth_hash,
        "validation": {"section4_checks": checks, "q0_reset_worst": q0_worst,
                       "generator_inference_parity": parity,
                       "truth_checks": truth_checks, "passed": passed},
        "descriptive": {
            "total_true_blocks": total_blocks,
            "block_width_distribution": {str(k): v for k, v in sorted(
                Counter(w for t in traces for w in t.widths).items())},
        },
    }
    (OUT / "corpus_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=str) + "\n")

    # ---- the two files `load_frozen_observed_corpus` requires, in its schema ----
    train_npz_sha = hashlib.sha256((OUT / "train_traces.npz").read_bytes()).hexdigest()
    heldout_npz_sha = hashlib.sha256((OUT / "heldout_traces.npz").read_bytes()).hexdigest()
    (OUT / "corpus_hash.json").write_text(json.dumps({
        "corpus_hash_sha256": digest,
        "train_npz_sha256": train_npz_sha,
        "heldout_npz_sha256": heldout_npz_sha,
        "truth_hash_sha256": truth_hash,
        "hash_covers": "canonical JSON of config + truth + all traces (observed and "
                       "hidden); no timestamps, no paths",
    }, indent=2, sort_keys=True) + "\n")
    (OUT / "config.json").write_text(json.dumps({
        **{k: v for k, v in config.items() if k != "validation"},
        "n_train_traces": N_TRAIN, "n_heldout_traces": N_HELDOUT,
    }, indent=2, sort_keys=True, default=str) + "\n")

    print("CONFIRMATORY CORPUS FROZEN")
    print(f"  corpus_hash_sha256 = {digest}")
    print(f"  truth_hash_sha256  = {truth_hash}   [SEALED, values not printed]")
    print(f"  train_hash         = {msg.sha256_hex(msg.canonical_json([msg.trace_to_jsonable(t) for t in corpus.train]))}")
    print(f"  heldout_hash       = {msg.sha256_hex(msg.canonical_json([msg.trace_to_jsonable(t) for t in corpus.heldout]))}")
    print(f"  truth attempts     = {attempts} (cap {MAX_TRUTH_ATTEMPTS})")
    print(f"  validation         = {checks}, q0={q0_worst:.2e}, parity={parity:.2e}")
    print(f"  train_npz_sha256   = {train_npz_sha}")
    print(f"  heldout_npz_sha256 = {heldout_npz_sha}")
    print(f"  written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
