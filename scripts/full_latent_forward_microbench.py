"""FULL-LATENT step 1b — inside the forward pass: what shapes, and what the LSE costs.

Measurement only.  `full_latent_sweep_profile.py` shows the semi-Markov forward
recursion dominates a FULL-LATENT sweep; this script says what that time is made of, by

  1. recording the exact size of every `predecessor_terms` result on one real forward
     pass over the frozen 100-trace corpus (no timing, so no observer effect), and
  2. benchmarking `scipy.special.logsumexp` against a plain-numpy shift-exp-sum-log on
     exactly that size distribution.

(2) is a *counterfactual measurement*, not a change: nothing in `src/` is edited and the
sampler is not asked to use the alternative.  It exists so the cost of the reduction can
be separated from the cost of the library call that currently performs it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_full_latent as mfl                     # noqa: E402
from hpop.mcmc_original import semi_markov_ffbs as smf                        # noqa: E402
from hpop.mcmc_original.transitions import log_transition_matrix              # noqa: E402

CORPUS_DIR = ROOT / "results" / "mcmc_original" / "matched_synthetic_formal_corpus"
OUT = ROOT / "results" / "mcmc_original" / "full_latent_perf"
SEED = 909_112_002


def inline_logsumexp(values: np.ndarray) -> float:
    """The same reduction, written out.  Kept here only to be timed."""
    top = values.max()
    return float(top + np.log(np.exp(values - top).sum()))


def collect_shapes(checkpoint: Path | None):
    corpus = mfl.load_frozen_observed_corpus(CORPUS_DIR)
    fixed = mfl.FullLatentFixed()
    model = mfl.build_full_latent_model(corpus.train, fixed)
    config = mfl.FullLatentConfig(arm=mfl.FULL_COND, structural_cadence=10,
                                  structural_scale=0.5, table_source="batched")
    sampler = mfl.FullLatentSampler(model=model, fixed=fixed, config=config)
    if checkpoint is not None and Path(checkpoint).exists():
        state = mfl.FullLatentChain.load(checkpoint, sampler).state.copy()
    else:
        pi, transition = mfl.draw_initial_pi_p(model, SEED)
        state = mfl.initial_full_latent_state(
            model, mfl.make_u_start(0, SEED, 1.0, fixed, model), pi, transition)

    sampler.tables.refresh(state)
    tables = sampler.tables.tables_for(state)
    log_pi = np.log(state.pi)
    log_p = log_transition_matrix(state.transition)

    sizes, finite_sizes, samples = [], [], []
    original = smf.predecessor_terms

    def recording(*args, **kwargs):
        starts, prev, terms = original(*args, **kwargs)
        sizes.append(int(terms.size))
        finite = terms[np.isfinite(terms)]
        finite_sizes.append(int(finite.size))
        if finite.size and len(samples) < 4000:
            samples.append(np.array(finite, copy=True))
        return starts, prev, terms

    smf.predecessor_terms = recording
    try:
        for table in tables:
            smf.forward(table, log_pi, log_p, model.delta_b, model.max_width,
                        model.min_width)
    finally:
        smf.predecessor_terms = original
    return sizes, finite_sizes, samples, len(tables)


def bench(samples, repeats: int = 3) -> dict:
    """Per-call cost of the two reductions over the recorded arrays."""
    results = {}
    for name, function in (("scipy_logsumexp", lambda a: float(logsumexp(a))),
                           ("inline_numpy", inline_logsumexp)):
        best = None
        for _ in range(repeats):
            began = time.perf_counter()
            for array in samples:
                function(array)
            elapsed = time.perf_counter() - began
            best = elapsed if best is None else min(best, elapsed)
        results[name] = {"total_seconds": float(best),
                         "microseconds_per_call": float(1e6 * best / len(samples))}
    # they must agree, or the counterfactual is meaningless
    worst = max(abs(float(logsumexp(a)) - inline_logsumexp(a)) for a in samples[:500])
    results["max_absolute_disagreement"] = float(worst)
    results["n_arrays"] = int(len(samples))
    return results


def main() -> None:
    checkpoint = None
    if len(sys.argv) > 1:
        checkpoint = Path(sys.argv[1])
    sizes, finite_sizes, samples, n_traces = collect_shapes(checkpoint)
    histogram = Counter(finite_sizes)
    report = {
        "n_traces": n_traces,
        "predecessor_terms_calls_per_forward_pass": len(sizes),
        "terms_per_call": {"mean": float(np.mean(sizes)),
                           "median": float(np.median(sizes)),
                           "max": int(np.max(sizes)), "total": int(np.sum(sizes))},
        "finite_terms_per_call": {"mean": float(np.mean(finite_sizes)),
                                  "median": float(np.median(finite_sizes)),
                                  "max": int(np.max(finite_sizes)),
                                  "total": int(np.sum(finite_sizes))},
        "finite_size_histogram": {str(k): int(v)
                                  for k, v in sorted(histogram.items())},
        "logsumexp_counterfactual": bench(samples),
        "scipy_version": __import__("scipy").__version__,
        "numpy_version": np.__version__,
    }
    calls = report["predecessor_terms_calls_per_forward_pass"]
    lse = report["logsumexp_counterfactual"]
    report["projected_per_sweep_seconds"] = {
        "scipy": float(calls * lse["scipy_logsumexp"]["microseconds_per_call"] / 1e6),
        "inline": float(calls * lse["inline_numpy"]["microseconds_per_call"] / 1e6),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "forward_microbench.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True)[:2000])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
