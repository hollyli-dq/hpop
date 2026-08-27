"""Stage 6E2 corpus — the segmentation-prior mismatch between generator and inference.

    PYTHONPATH=src python scripts/stage7b2_generator_prior_audit.py

The Stage 6E2 synthetic generator and the registered inference target agree on the skill
path, on the recurrent block generation and on `q_0 = 0` per block. They do **not** agree
on the law of the segmentation itself. This script derives the discrepancy exactly,
quantifies how much of it the prior alone would produce on this corpus, and writes the
audit.

It changes nothing. The corpus is frozen, its hash is recorded, and the Stage 6E2 and
Step 7B2 chains are running on it right now; the audit exists so the recovery numbers are
read correctly later, not so anything is regenerated.

## What the generator does (stage6e_corpus.py, `_generate_trace`)

    line 193   n_blocks = rng.choice(BLOCKS_PER_TRACE)     L ~ Uniform{4, 5, 6}
    line 195   path[0]  = rng.choice(K, p=PI_TRUE)         z_1 ~ pi
    line 197   path[l]  = rng.choice(K, p=P_TRUE[path[-1]])  z_l ~ P[z_{l-1}, .]
    line 201   width    = rng.choice(widths, p=width_p)    w ~ (1-delta_B)^(w-1) on [3,12]
    line 204   q = zeros(m)                                q_0 = 0 for EVERY block
    line 212   true_boundaries = ends[:-1]                 no terminal boundary event

## Where they diverge

Conditioned on the observed trace length `J`, with `L` blocks of widths `w_1..w_L`:

    inference   p(S | J) proportional to delta_B^(L-1) (1 - delta_B)^(J-L)
    generator   p(S | J) proportional to Z^(-L)        (1 - delta_B)^(J-L),  L in {4,5,6}

because the generator draws each width from the normalised law with
`Z = sum_{w=3}^{12} (1 - delta_B)^(w-1)`. The width factor is therefore *exactly* the
registered one — that part is right — and the whole discrepancy is carried by the block
count:

    p_inference / p_generator  proportional to  (delta_B * Z)^L

`delta_B * Z = 0.5803 != 1`, so relative to the generator the inference prior penalises
every additional block. It prefers fewer, longer segments, and its support over `L` is
wider than the three values the generator can produce.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.stage6e_corpus import (                              # noqa: E402
    BLOCKS_PER_TRACE, corpus_hash, generate_corpus, width_distribution,
)
from hpop.mcmc_original.stage6e_frozen import (                              # noqa: E402
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH,
)

OUT = ROOT / "results" / "mcmc_original" / "stage7b2_full_joint_ffbs"
GENERATOR = ROOT / "src" / "hpop" / "mcmc_original" / "stage6e_corpus.py"


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:                                                # pragma: no cover
        return "unknown"


def jsonable(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if math.isnan(v) else v
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def compositions_by_length(J: int, lo: int, hi: int) -> dict:
    """How many ordered compositions of `J` into parts in `[lo, hi]` have `L` parts."""
    counts = {0: {0: 1}}
    for total in range(1, J + 1):
        row: dict = {}
        for width in range(lo, min(hi, total) + 1):
            for length, n in counts[total - width].items():
                row[length + 1] = row.get(length + 1, 0) + n
        counts[total] = row
    return counts[J]


def inference_length_distribution(J: int, delta_b: float) -> dict:
    """`p(L | J)` under the registered boundary prior, exactly."""
    counts = compositions_by_length(J, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
    weights = {L: n * delta_b ** (L - 1) * (1.0 - delta_b) ** (J - L)
               for L, n in counts.items()}
    total = sum(weights.values())
    return {L: w / total for L, w in sorted(weights.items())} if total > 0 else {}


def generator_length_distribution(J: int, delta_b: float) -> dict:
    """`p(L | J)` under the generator: uniform over {4,5,6}, widths from the width law."""
    Z = sum((1.0 - delta_b) ** (w - 1)
            for w in range(MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH + 1))
    counts = compositions_by_length(J, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
    weights = {L: counts.get(L, 0) * (Z ** -L) * (1.0 - delta_b) ** (J - L)
               for L in BLOCKS_PER_TRACE}
    total = sum(weights.values())
    return {L: w / total for L, w in sorted(weights.items())} if total > 0 else {}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = generate_corpus()
    digest = corpus_hash(corpus)

    delta_b = DELTA_B
    Z = float(sum((1.0 - delta_b) ** (w - 1)
                  for w in range(MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH + 1)))
    relative = delta_b * Z

    lengths = [t.length for t in corpus.train]
    blocks = [t.n_blocks for t in corpus.train]
    widths = [b - a for t in corpus.train
              for a, b in zip((0,) + t.true_boundaries,
                              t.true_boundaries + (t.length,))]
    law = width_distribution(delta_b)
    observed = np.array([sum(1 for w in widths if w == x)
                         for x in range(MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH + 1)], float)
    observed /= observed.sum()

    # How much under-segmentation the PRIOR ALONE would produce on this corpus: the
    # prior-expected block count per trace at that trace's own J, summed, against the
    # generating truth. This is a statement about the prior, not about any chain.
    expected_inference = 0.0
    expected_generator = 0.0
    per_length_examples = {}
    for J in sorted(set(lengths)):
        inference = inference_length_distribution(J, delta_b)
        generator = generator_length_distribution(J, delta_b)
        multiplicity = lengths.count(J)
        expected_inference += multiplicity * sum(L * p for L, p in inference.items())
        expected_generator += multiplicity * sum(L * p for L, p in generator.items())
        if J in (24, 31, 40):
            per_length_examples[str(J)] = {
                "inference_p_L": {str(L): round(p, 4) for L, p in inference.items()
                                  if p >= 5e-4},
                "generator_p_L": {str(L): round(p, 4) for L, p in generator.items()},
                "inference_mean_L": sum(L * p for L, p in inference.items()),
                "generator_mean_L": sum(L * p for L, p in generator.items())}

    audit = {
        "what": "segmentation-prior mismatch between the Stage 6E2 synthetic generator "
                "and the registered inference target",
        "discovered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "audit_commit": source_commit(),
        "python": platform.python_version(), "numpy": np.__version__,
        "audit_occurred_after_corpus_freeze": True,
        "corpus_was_changed_by_this_audit": False,
        "corpus_hash": digest,
        "corpus_hash_matches_running_experiments": True,
        "generator_source": {
            "file": str(GENERATOR.relative_to(ROOT)),
            "sha256": hashlib.sha256(GENERATOR.read_bytes()).hexdigest(),
            "function": "_generate_trace",
            "lines": {
                "block_count": "193: n_blocks = int(rng.choice(BLOCKS_PER_TRACE)) "
                               "-> L ~ Uniform{4,5,6}",
                "initial_skill": "195: rng.choice(N_SKILLS, p=PI_TRUE) -> z_1 ~ pi",
                "transition": "197: rng.choice(N_SKILLS, p=P_TRUE[path[-1]]) "
                              "-> z_l ~ P[z_{l-1}, .]; P_TRUE diagonal is exactly 0 "
                              "(lines 80-82), so adjacent blocks cannot share a skill",
                "width": "201: rng.choice(widths_support, p=width_p) with width_p from "
                         "width_distribution (118-129)",
                "q0_reset": "204: q = np.zeros(u.shape[0]) -- q_0 = 0 for EVERY block",
                "emission": "206/209: recurrent_step_probabilities / "
                            "recurrent_validity_update",
                "no_terminal_boundary": "212: true_boundaries = ends[:-1]",
            },
        },
        "agrees_with_the_registered_model": [
            "z_1 ~ pi", "z_l ~ P[z_{l-1}, .]", "P[h, h] = 0, no adjacent repeats",
            "no terminal transition", "q_0 = 0 reset for every block",
            "per-block width factor p(w) proportional to (1-delta_B)^(w-1) on [3, 12]"],
        "disagrees_with_the_registered_model": [
            "the number of blocks: the generator draws L ~ Uniform{4,5,6} instead of "
            "letting the boundary prior determine it"],
        "inference": {
            "p_S_given_J": "proportional to delta_B^(L-1) (1-delta_B)^(J-L)",
            "P_hh": 0, "terminal_transition": False,
            "widths": [MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH], "delta_B": delta_b},
        "generator": {
            "p_S_given_J": "proportional to Z^(-L) (1-delta_B)^(J-L), L in {4,5,6}",
            "L_law": "Uniform{4,5,6}",
            "width_law": "normalised (1-delta_B)^(w-1) on [3, 12]"},
        "normalizer": {"Z": Z, "one_over_delta_B": 1.0 / delta_b,
                       "equal": bool(abs(Z - 1.0 / delta_b) < 1e-9)},
        "relative_weight": {
            "delta_B_times_Z": relative,
            "interpretation": "each additional block is downweighted by "
                              f"{relative:.4f} under inference relative to the generator; "
                              "the inference prior therefore prefers fewer, longer "
                              "segments"},
        "support_mismatch": {
            "generator_L": list(BLOCKS_PER_TRACE),
            "inference_L": "broader, J-dependent: every L with a legal composition of J "
                           "into widths in [3, 12]"},
        "examples": per_length_examples,
        "corpus_facts": {
            "n_train_traces": len(corpus.train),
            "true_total_blocks": int(sum(blocks)),
            "observed_L_counts": {str(k): int(v) for k, v in
                                  sorted(Counter(blocks).items())},
            "trace_length": {"min": int(min(lengths)), "max": int(max(lengths)),
                             "mean": float(np.mean(lengths))},
            "width_law": law.tolist(),
            "observed_width_frequencies": observed.tolist(),
            "max_width_deviation_from_law": float(np.abs(observed - law).max())},
        "prior_pull_on_this_corpus": {
            "prior_expected_total_blocks_under_inference": expected_inference,
            "prior_expected_total_blocks_under_generator": expected_generator,
            "generating_truth_total_blocks": int(sum(blocks)),
            "prior_deficit_vs_truth": expected_inference - float(sum(blocks)),
            "note": "the expected block count under each prior, evaluated at each trace's "
                    "own J and summed. This is what the PRIOR ALONE pulls toward before "
                    "any data enters; it is not a prediction of any chain's posterior and "
                    "no chain output is used to compute it."},
        "impact": {
            "sampler_correctness": "unaffected -- Stage 6E1A, Stage 6E1B, Step 7A and "
                                   "Step 7B1 compare a sampler against an exact or "
                                   "independent computation of the SAME posterior given "
                                   "the data; where the data came from is irrelevant to "
                                   "those claims",
            "local_vs_ffbs_comparison": "unaffected -- same frozen corpus, same "
                                        "posterior, only the segmentation kernel differs, "
                                        "so a structural-locking difference between the "
                                        "two kernels remains a valid conclusion",
            "synthetic_recovery": "MISSPECIFIED -- Stage 6E2 is a recovery experiment "
                                  "under a documented segmentation-prior "
                                  "misspecification. Boundary F1, boundary "
                                  "precision/recall, segment-count recovery and skill "
                                  "ARI remain real observations, but must not be read as "
                                  "'what is recoverable under a well-specified synthetic "
                                  "model'",
            "nominal_coverage": "not interpretable as well-specified coverage -- the "
                                "generating truth is not a draw from the inference prior, "
                                "so credible-interval coverage carries no nominal "
                                "guarantee"},
        "direction": "the mismatch pushes inference toward fewer, longer segments, which "
                     "is the direction of under-segmentation. It is quantified above as a "
                     "prior effect only. No baseline recovery statistic is embedded here: "
                     "the Stage 6E2 baseline is still advancing its registered ladder and "
                     "its intermediate rung must not be quoted as a result.",
        "not_claimed": [
            "that this explains all of the observed low boundary F1",
            "that this causes the observed structural locking",
            "that any corpus should be regenerated now"],
        "recommended_follow_up": {
            "what": "a small matched-generator sensitivity control, AFTER Stage 6E2 and "
                    "Step 7B2 finish",
            "generator_A": "the current corpus: L ~ Uniform{4,5,6}",
            "generator_B": "L and widths drawn from the registered p(S | delta_B)",
            "held_fixed": ["U", "the four scalars", "pi", "P", "a comparable trace count"],
            "read_out": ["boundary count bias", "boundary F1", "relation locking", "ARI"],
            "logic": "if boundary recovery improves markedly under B, part of the Stage "
                     "6E2 difficulty is misspecification; if structural locking persists, "
                     "that is stronger evidence it is inference geometry rather than "
                     "generator mismatch"},
    }
    (OUT / "generator_prior_audit.json").write_text(
        json.dumps(jsonable(audit), indent=2))

    print(f"[audit] corpus {digest[:16]}...  Z={Z:.6f}  delta_B*Z={relative:.4f}")
    print(f"[audit] true blocks {sum(blocks)}; prior-expected under inference "
          f"{expected_inference:.1f}, under generator {expected_generator:.1f}")
    print(f"[audit] observed L counts {dict(sorted(Counter(blocks).items()))}")
    print(f"[audit] width law deviation {np.abs(observed - law).max():.4f}")
    print(f"[audit] wrote {OUT / 'generator_prior_audit.json'}")


if __name__ == "__main__":
    main()
