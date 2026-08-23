"""Freeze the FORMAL matched-synthetic corpus for the Condition A--D program.

Run:  PYTHONPATH=src .venv/bin/python scripts/generate_matched_formal_corpus.py

100 training + 45 held-out traces from the validated matched generator
(commit 8ca8281) in supplied-truth mode, with the trace-length design, seed and
split all frozen in this file BEFORE generation. The smoke corpus and the Stage
6E2 corpus are not reused. The corpus is validated (Section 4 checks) and then
frozen with hashes; it must never be regenerated because its realized draws
look inconvenient.
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

OUT = ROOT / "results" / "mcmc_original" / "matched_synthetic_formal_corpus"

# ------------------------------------------------------------- frozen design
# Seed 6_200_001 was verified UNUSED across all three worktrees, every results
# manifest, and the project memory before being registered here (Stage 6E2 used
# 6_053_000/+10_000; generator validation 700_024..750_002; smoke 6_100_001;
# collapsed-U 815xxxx/816xxxx).
MASTER_SEED = 6_200_001
N_TRAIN, N_HELDOUT = 100, 45
DESIGN_J = (24, 32, 40, 48)
# Frozen ordered length vectors: train cycles the design 25 times (25 traces per
# J exactly); held-out cycles it 45 times -> counts (12, 11, 11, 11).
TRACE_LENGTHS_TRAIN = tuple(DESIGN_J[i % 4] for i in range(N_TRAIN))
TRACE_LENGTHS_HELDOUT = tuple(DESIGN_J[i % 4] for i in range(N_HELDOUT))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _traces_to_npz(traces, path: Path) -> None:
    payload = {}
    for t in traces:
        tag = f"t{t.trace_index:03d}"
        payload[f"{tag}_cpa"] = np.asarray(t.cpa, dtype=np.int8)
        payload[f"{tag}_widths"] = np.asarray(t.widths, dtype=np.int16)
        payload[f"{tag}_boundaries"] = np.asarray(t.boundaries, dtype=np.int16)
        payload[f"{tag}_labels"] = np.asarray(t.labels, dtype=np.int8)
        payload[f"{tag}_block_ll"] = np.asarray(t.block_log_likelihoods,
                                                dtype=np.float64)
        payload[f"{tag}_logs"] = np.asarray(
            [t.log_seg_prior, t.log_label_prior], dtype=np.float64)
    payload["n_traces"] = np.asarray([len(traces)])
    np.savez_compressed(path, **payload)


def _reload_and_compare(traces, path: Path) -> bool:
    data = np.load(path)
    for t in traces:
        tag = f"t{t.trace_index:03d}"
        if (tuple(int(v) for v in data[f"{tag}_cpa"]) != t.cpa
                or tuple(int(v) for v in data[f"{tag}_widths"]) != t.widths
                or tuple(int(v) for v in data[f"{tag}_boundaries"]) != t.boundaries
                or tuple(int(v) for v in data[f"{tag}_labels"]) != t.labels
                or not np.array_equal(data[f"{tag}_block_ll"],
                                      np.asarray(t.block_log_likelihoods))):
            return False
    return int(data["n_traces"][0]) == len(traces)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    truth = msg.supplied_truth()
    truth_checks = msg.validate_truth(truth)

    corpus = msg.generate_corpus(MASTER_SEED, TRACE_LENGTHS_TRAIN,
                                 TRACE_LENGTHS_HELDOUT, truth)
    digest = msg.corpus_hash(corpus)
    truth_hash = msg.sha256_hex(msg.canonical_json(msg.truth_to_jsonable(truth)))
    traces = corpus.train + corpus.heldout

    # ---------------------------------------------- Section 4 correctness checks
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

    # q_0 reset: production replay from zeros must reproduce the recorded
    # per-block log-likelihood for every block of every trace.
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

    _traces_to_npz(corpus.train, OUT / "train_traces.npz")
    _traces_to_npz(corpus.heldout, OUT / "heldout_traces.npz")
    save_load_ok = (_reload_and_compare(corpus.train, OUT / "train_traces.npz")
                    and _reload_and_compare(corpus.heldout,
                                            OUT / "heldout_traces.npz"))

    passed = (checks["illegal_width"] == 0 and checks["cover_mismatch"] == 0
              and checks["self_transition"] == 0
              and checks["nonfinite_block_ll"] == 0
              and q0_worst < 1e-10 and parity < 1e-10 and save_load_ok)
    if not passed:
        raise SystemExit(f"formal corpus failed validation: {checks}, "
                         f"q0={q0_worst}, parity={parity}, "
                         f"save_load={save_load_ok}")

    # -------------------------------------------------- descriptive summary only
    block_counts = Counter(t.n_segments for t in traces)
    widths = Counter(w for t in traces for w in t.widths)
    skills = Counter(k for t in traces for k in t.labels)
    transitions = Counter((a, b) for t in traces
                          for a, b in zip(t.labels[:-1], t.labels[1:]))
    repeats = total_steps = 0
    for t in traces:
        for block in t.role_blocks:
            seen = set()
            for role in block:
                total_steps += 1
                if role in seen:
                    repeats += 1
                seen.add(role)
    total_blocks = sum(t.n_segments for t in traces)
    summary = {
        "trace_length_counts": {
            "train": dict(Counter(TRACE_LENGTHS_TRAIN)),
            "heldout": dict(Counter(TRACE_LENGTHS_HELDOUT))},
        "total_true_blocks": {"train": sum(t.n_segments for t in corpus.train),
                              "heldout": sum(t.n_segments for t in corpus.heldout),
                              "all": total_blocks},
        "block_count_distribution": {str(k): v for k, v in
                                     sorted(block_counts.items())},
        "block_width_distribution": {str(k): v for k, v in
                                     sorted(widths.items())},
        "true_skill_frequencies": {str(k): v / total_blocks for k, v in
                                   sorted(skills.items())},
        "true_transition_frequencies": {f"{a}->{b}": c for (a, b), c in
                                        sorted(transitions.items())},
        "repeat_occurrence_frequency": repeats / total_steps,
        "note": "descriptive only; the corpus is frozen regardless of these "
                "values (registered rule)",
    }
    (OUT / "corpus_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")

    config = {
        **corpus.config,
        "purpose": "FORMAL matched-synthetic recovery corpus for Conditions "
                   "A-D. Frozen before any inference result was inspected.",
        "design": {"n_train": N_TRAIN, "n_heldout": N_HELDOUT,
                   "trace_lengths_train": list(TRACE_LENGTHS_TRAIN),
                   "trace_lengths_heldout": list(TRACE_LENGTHS_HELDOUT),
                   "design_rule": "train: cycle (24,32,40,48) 25 times -> "
                                  "exactly 25 per length; heldout: cycle 45 "
                                  "times -> counts (12,11,11,11)"},
        "seed_provenance": "6_200_001 verified unused in Stage 6E2, generator "
                           "validation, smoke corpus, collapsed-U validation "
                           "and all worktree manifests before registration",
        "generator_commit": "8ca828153e8e263bf4ea4823e45a53fa454037ad",
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True)
                                     + "\n")
    manifest = {
        "source_commit": _git("rev-parse", "HEAD"),
        "generator_commit": "8ca828153e8e263bf4ea4823e45a53fa454037ad",
        "generator_version": msg.GENERATOR_VERSION,
        "master_seed": MASTER_SEED,
        "truth": msg.truth_to_jsonable(truth),
        "truth_hash_sha256": truth_hash,
        "truth_validation": truth_checks,
        "rho_star": None,
        "rho_note": "supplied-truth mode fixes U* directly; rho* is not part of "
                    "the registered Stage 6E2 truth and is recorded null",
        "observed_fields_available_to_inference":
            ["split", "trace_index", "length", "cpa occurrence sequence"],
        "hidden_truth_used_only_for_evaluation":
            ["widths", "boundaries", "labels", "role_blocks",
             "block log-likelihoods", "truth object"],
        "split_assignment": {"level": "trace",
                             "train": [t.trace_index for t in corpus.train],
                             "heldout": [t.trace_index for t in corpus.heldout]},
        "corpus_hash_sha256": digest,
    }
    (OUT / "truth_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUT / "corpus_hash.json").write_text(json.dumps({
        "corpus_hash_sha256": digest,
        "truth_hash_sha256": truth_hash,
        "train_npz_sha256": hashlib.sha256(
            (OUT / "train_traces.npz").read_bytes()).hexdigest(),
        "heldout_npz_sha256": hashlib.sha256(
            (OUT / "heldout_traces.npz").read_bytes()).hexdigest(),
        "hash_covers": "canonical JSON of config + truth + all traces (observed "
                       "and hidden); no timestamps, no paths",
    }, indent=2, sort_keys=True) + "\n")

    (OUT / "report.md").write_text("\n".join([
        "# Formal matched-synthetic corpus (frozen)",
        "",
        f"Source commit: `{_git('rev-parse', 'HEAD')}` &middot; generator commit "
        "`8ca8281`",
        f"Master seed: {MASTER_SEED} (registered before generation; never used "
        "by any prior run)",
        f"Corpus hash: `{digest}`",
        f"Truth hash: `{truth_hash}`",
        "",
        f"- {N_TRAIN} training traces: exactly 25 each at J = 24, 32, 40, 48",
        f"- {N_HELDOUT} held-out traces: lengths cycling (24,32,40,48) -> "
        "counts 12/11/11/11",
        f"- {total_blocks} true blocks; skill frequencies "
        f"{ {k: round(v/total_blocks, 4) for k, v in sorted(skills.items())} }",
        f"- repeat-occurrence frequency {repeats / total_steps:.4f}",
        "",
        "## Section 4 validation (all PASS)",
        f"- illegal widths / cover mismatches / self-transitions / non-finite "
        f"block likelihoods: 0 / 0 / 0 / 0",
        f"- q_0-reset replay max error: {q0_worst:.3e} (< 1e-10)",
        f"- generator/scorer complete-data log-probability parity: "
        f"{parity:.3e} (< 1e-10)",
        f"- deterministic save/load parity: {save_load_ok}",
        "",
        "This corpus is FROZEN. Do not regenerate it because realized block "
        "counts, skill counts, or recovery difficulty look inconvenient.",
    ]) + "\n")
    print(f"formal corpus frozen at {OUT}")
    print(f"corpus hash: {digest}")
    print(f"truth hash:  {truth_hash}")
    print(f"q0 replay:   {q0_worst:.3e}   parity: {parity:.3e}   "
          f"save/load: {save_load_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
