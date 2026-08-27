"""Generate the deterministic matched-generator SMOKE corpus.

Run:  PYTHONPATH=src .venv/bin/python scripts/generate_matched_smoke_corpus.py

Refuses to run unless every gate in
results/mcmc_original/matched_generator_validation/report.md passed.

This corpus exists for serialization, manifest validation, scorer parity and
later pipeline smoke tests ONLY. It is NOT the headline matched-synthetic
corpus; the final Condition A--D corpus size, seed and truth configuration are
deliberately NOT chosen here. No inference is run on it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_generator_diagnostics as mgd            # noqa: E402
from hpop.mcmc_original import matched_synthetic_generator as msg              # noqa: E402
from hpop.mcmc_original.recurrent_segmentation import RecurrentBlockScorer     # noqa: E402

VALIDATION = ROOT / "results" / "mcmc_original" / "matched_generator_validation"
OUT = ROOT / "results" / "mcmc_original" / "matched_generator_smoke_corpus"

# ------------------------------------------------------------- registered smoke design
MASTER_SEED = 6_100_001                     # fixed before generation; never searched
N_TRAIN = 20
N_HELDOUT = 10
# Fixed design lengths assigned by cycling the validated J design {24, 32, 40, 48}.
TRACE_LENGTHS_TRAIN = tuple([24, 32, 40, 48][i % 4] for i in range(N_TRAIN))
TRACE_LENGTHS_HELDOUT = tuple([24, 32, 40, 48][i % 4] for i in range(N_HELDOUT))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def require_validation_passed() -> dict:
    report = VALIDATION / "report.md"
    if not report.exists():
        raise SystemExit("validation artifacts missing; run "
                         "scripts/validate_matched_segmentation_generator.py first")
    if "ALL GATES PASS" not in report.read_text():
        raise SystemExit("a validation gate failed; the smoke corpus must not be "
                         "generated")
    return {"validation_report": str(report.relative_to(ROOT)),
            "validation_verdict": "ALL GATES PASS"}


def main() -> int:
    provenance = require_validation_passed()
    OUT.mkdir(parents=True, exist_ok=True)

    truth = msg.supplied_truth()
    truth_checks = msg.validate_truth(truth)
    corpus = msg.generate_corpus(MASTER_SEED, TRACE_LENGTHS_TRAIN,
                                 TRACE_LENGTHS_HELDOUT, truth)
    digest = msg.corpus_hash(corpus)

    # scorer-parity sanity on the exact corpus being shipped
    traces = corpus.train + corpus.heldout
    scorer = RecurrentBlockScorer(
        traces=tuple(t.cpa for t in traces), epsilon=truth.epsilon,
        u_by_skill=truth.u_by_skill, beta=truth.beta, omega=truth.omega,
        lambda_rep=truth.lambda_rep, lambda_back=truth.lambda_back,
        max_width=truth.max_width, min_width=truth.min_width)
    log_c = {J: mgd.exact_normalizer(J, truth.delta_b, truth.min_width,
                                     truth.max_width)
             for J in {t.length for t in traces}}
    parity = max(abs(msg.generator_complete_data_log_prob(t)
                     - msg.inference_complete_data_log_prob(
                         t, truth, scorer, i, log_c[t.length]))
                 for i, t in enumerate(traces))
    if parity >= 1e-10:
        raise SystemExit(f"smoke-corpus scorer parity failed: {parity}")

    config = {
        **corpus.config,
        "purpose": "smoke corpus: serialization, manifest validation, scorer "
                   "parity, pipeline smoke tests. NOT the headline "
                   "matched-synthetic corpus; no inference is run on it.",
        "scorer_parity_max_abs_error": parity,
        **provenance,
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True)
                                     + "\n")

    manifest = {
        "generator": {
            "source_commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "generator_version": msg.GENERATOR_VERSION,
            "generator_file_sha256": hashlib.sha256(
                (ROOT / "src/hpop/mcmc_original/matched_synthetic_generator.py")
                .read_bytes()).hexdigest(),
            "segmentation_prior_file_sha256": hashlib.sha256(
                (ROOT / "src/hpop/mcmc_original/matched_segmentation_prior.py")
                .read_bytes()).hexdigest(),
        },
        "seeds": {
            "master_seed": MASTER_SEED,
            "child_stream_scheme": corpus.config["rng_scheme"],
            "split_codes": corpus.config["split_codes"],
        },
        "observed_fields_available_to_inference": {
            "description": "per trace: split, trace_index, length J, and the CPA "
                           "occurrence sequence x^(n) under 'observed'. Nothing "
                           "else.",
            "fields": ["split", "trace_index", "length", "observed.cpa"],
        },
        "hidden_truth_used_only_for_evaluation": {
            "fields": ["hidden_truth.widths", "hidden_truth.boundaries",
                       "hidden_truth.labels", "hidden_truth.role_blocks",
                       "hidden_truth.log_seg_prior",
                       "hidden_truth.log_label_prior",
                       "hidden_truth.block_log_likelihoods",
                       "truth (entire object)"],
        },
        "truth": msg.truth_to_jsonable(truth),
        "truth_validation": truth_checks,
        "trace_lengths": {"train": list(TRACE_LENGTHS_TRAIN),
                          "heldout": list(TRACE_LENGTHS_HELDOUT)},
        "split_assignment": {"train": [t.trace_index for t in corpus.train],
                             "heldout": [t.trace_index for t in corpus.heldout],
                             "level": "trace"},
        "corpus_hash_sha256": digest,
    }
    (OUT / "truth_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    for name, split in (("train_traces.jsonl", corpus.train),
                        ("heldout_traces.jsonl", corpus.heldout)):
        with (OUT / name).open("w") as handle:
            for trace in split:
                handle.write(msg.canonical_json(msg.trace_to_jsonable(trace))
                             + "\n")

    (OUT / "corpus_hash.json").write_text(json.dumps({
        "corpus_hash_sha256": digest,
        "hash_covers": "canonical JSON of config + truth + all traces "
                       "(observed and hidden), sorted keys, no timestamps, "
                       "no paths",
        "train_traces_jsonl_sha256": hashlib.sha256(
            (OUT / "train_traces.jsonl").read_bytes()).hexdigest(),
        "heldout_traces_jsonl_sha256": hashlib.sha256(
            (OUT / "heldout_traces.jsonl").read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True) + "\n")

    n_blocks_train = sum(t.n_segments for t in corpus.train)
    n_blocks_heldout = sum(t.n_segments for t in corpus.heldout)
    (OUT / "report.md").write_text("\n".join([
        "# Matched-generator smoke corpus",
        "",
        f"Source commit: `{_git('rev-parse', 'HEAD')}`",
        f"Master seed: {MASTER_SEED} (fixed before generation, never searched)",
        f"Corpus hash: `{digest}`",
        "",
        f"- {len(corpus.train)} training traces "
        f"({n_blocks_train} blocks, lengths {sorted(set(TRACE_LENGTHS_TRAIN))})",
        f"- {len(corpus.heldout)} held-out traces "
        f"({n_blocks_heldout} blocks), split at the trace level",
        f"- truth: supplied mode (Stage 6E2 registered configuration), "
        f"K={truth.n_skills}, m={truth.n_roles}, d={truth.latent_dim}, "
        f"delta_B={truth.delta_b}, epsilon={truth.epsilon}, "
        f"widths [{truth.min_width}, {truth.max_width}]",
        f"- generator/inference complete-data log-probability parity on this "
        f"exact corpus: max |diff| = {parity:.3e} (< 1e-10)",
        "",
        "This is a SMOKE corpus for serialization, manifest and scorer-parity "
        "checks. It is not the headline matched-synthetic corpus, and no "
        "inference has been or should be run on it under this task.",
    ]) + "\n")
    print(f"smoke corpus written to {OUT}")
    print(f"corpus hash: {digest}")
    print(f"scorer parity max abs error: {parity:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
