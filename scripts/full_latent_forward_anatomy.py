"""FULL-LATENT steps B/C/F — the anatomy of one forward pass.

    PYTHONPATH=src python scripts/full_latent_forward_anatomy.py [checkpoint.npz]

Measurement only.  Nothing in `src/` is edited: the census wraps the sampler's own
callables for the duration of this process and restores them, and every benchmark runs on
COPIES of real checkpoint tables.  No formal chain file is opened for writing.

B  census of `predecessor_terms` over one real forward pass across all traces: calls,
   option counts, block-score reads, and the empirical scaling in K on synthetic tables.
C  what the logsumexp time is actually made of, on the real size distribution:
   scipy call -> array-API shim -> allocation -> the reduction arithmetic itself.
F  attribution of one forward pass by cProfile self time, grouped into
   arithmetic / dispatch / Python-interpreter / allocation.

Timings take the MINIMUM over repeats: the box is running an 8-worker formal chain, and
the minimum is the statistic least contaminated by a competing process.
"""

from __future__ import annotations

import cProfile
import json
import math
import os
import pstats
import sys
import time
from collections import Counter
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_full_latent as mfl                     # noqa: E402
from hpop.mcmc_original import semi_markov_ffbs as smf                        # noqa: E402
from hpop.mcmc_original.transitions import log_transition_matrix              # noqa: E402

CORPUS_DIR = ROOT / "results" / "mcmc_original" / "matched_synthetic_formal_corpus"
OUT = ROOT / "results" / "mcmc_original" / "full_latent_perf"
SEED = 909_112_002


# --------------------------------------------------------------------------- fixture
def fixture(checkpoint: Path | None):
    corpus = mfl.load_frozen_observed_corpus(CORPUS_DIR)
    fixed = mfl.FullLatentFixed()
    model = mfl.build_full_latent_model(corpus.train, fixed)
    config = mfl.FullLatentConfig(arm=mfl.FULL_COND, structural_cadence=10,
                                  structural_scale=0.5, table_source="batched")
    sampler = mfl.FullLatentSampler(model=model, fixed=fixed, config=config)
    if checkpoint is not None and Path(checkpoint).exists():
        state = mfl.FullLatentChain.load(checkpoint, sampler).state.copy()
        origin = f"checkpoint:{checkpoint.name}"
    else:
        pi, transition = mfl.draw_initial_pi_p(model, SEED)
        state = mfl.initial_full_latent_state(
            model, mfl.make_u_start(0, SEED, 1.0, fixed, model), pi, transition)
        origin = "fresh_start"
    sampler.tables.refresh(state)
    tables = list(sampler.tables.tables_for(state))
    return {"model": model, "state": state, "tables": tables, "origin": origin,
            "log_pi": np.log(state.pi),
            "log_p": log_transition_matrix(state.transition)}


def run_forward_all(fx) -> None:
    for table in fx["tables"]:
        smf.forward(table, fx["log_pi"], fx["log_p"], fx["model"].delta_b,
                    fx["model"].max_width, fx["model"].min_width)


# ------------------------------------------------------------------------- B: census
def census(fx) -> dict:
    """One real forward pass with counting-only wrappers.  No timing, no observer bias
    on the numbers that matter (they are integers)."""
    model = fx["model"]
    sizes, block_reads, inner_pairs = [], [], []
    per_b_k = Counter()
    start_visits = Counter()
    original = smf.predecessor_terms

    def recording(alpha, b, k, scores, log_pi, log_p, log_db, log_1mdb,
                  max_width, min_width=1):
        out = original(alpha, b, k, scores, log_pi, log_p, log_db, log_1mdb,
                       max_width, min_width)
        sizes.append(int(out[2].size))
        # replicate the loop bounds to count work attempted, not just work returned
        lowest = max(0, int(b) - int(max_width))
        reads = 0
        for a in range(max(1, lowest), int(b) - int(min_width) + 1):
            reads += 1
            start_visits[a] += 1
        block_reads.append(reads)
        inner_pairs.append(reads * int(alpha.shape[1]))
        per_b_k[(int(b), int(k))] += 1
        return out

    smf.predecessor_terms = recording
    try:
        run_forward_all(fx)
    finally:
        smf.predecessor_terms = original

    J_values = [len(t) for t in model.traces]
    K, D, Dmin = model.n_skills, model.max_width, model.min_width
    predicted_pairs = 0
    predicted_calls = 0
    for J in J_values:
        predicted_calls += J * K
        for b in range(1, J + 1):
            lo, hi = max(1, b - D), b - Dmin
            predicted_pairs += max(0, hi - lo + 1) * K * K
    return {
        "n_traces": len(fx["tables"]),
        "J_values": dict(sorted(Counter(J_values).items())),
        "K": K, "max_width": D, "min_width": Dmin, "n_roles": model.n_roles,
        "calls": len(sizes),
        "calls_predicted_J_times_K": predicted_calls,
        "options_returned_total": int(np.sum(sizes)),
        "options_per_call": {"mean": float(np.mean(sizes)),
                             "median": float(np.median(sizes)),
                             "max": int(np.max(sizes)), "min": int(np.min(sizes))},
        "block_score_reads_total": int(np.sum(block_reads)),
        "inner_a_h_pairs_total": int(np.sum(inner_pairs)),
        "inner_a_h_pairs_predicted_JDK2": int(predicted_pairs),
        "pruned_fraction": float(1.0 - np.sum(sizes) / max(1, np.sum(inner_pairs))),
        "distinct_b_k_visited": len(per_b_k),
        "each_b_k_visited_once": all(v == 1 for v in per_b_k.values()),
        "start_position_visits_top10": [[int(a), int(n)] for a, n
                                        in start_visits.most_common(10)],
        "size_histogram": {str(k): int(v) for k, v in sorted(Counter(sizes).items())},
    }


# ------------------------------------------- B2: empirical scaling in K (synthetic)
def scaling_in_K(J: int, D: int, Dmin: int, K_values, repeats: int = 3) -> dict:
    """Same recursion, synthetic tables, K varied.  Measures the exponent in K."""
    rng = np.random.default_rng(SEED)
    out = {}
    for K in K_values:
        scores = rng.normal(-2.0, 1.0, size=(J, J + 1, K))
        mask = np.ones((J, J + 1), dtype=bool)
        for a in range(J):
            for b in range(J + 1):
                if not (Dmin <= b - a <= D):
                    mask[a, b] = False
        scores[~mask] = -np.inf
        log_pi = np.log(np.full(K, 1.0 / K))
        p = rng.dirichlet(np.ones(K), size=K)
        log_p = np.log(p)
        best = math.inf
        for _ in range(repeats):
            began = time.perf_counter()
            smf.forward(scores, log_pi, log_p, 0.15, D, Dmin)
            best = min(best, time.perf_counter() - began)
        out[str(K)] = {"seconds": best,
                       "predicted_JDK2": J * D * K * K,
                       "predicted_JK2_plus_JDK": J * K * K + J * D * K}
    ks = np.array([float(k) for k in out], dtype=float)
    ts = np.array([out[k]["seconds"] for k in out], dtype=float)
    slope = float(np.polyfit(np.log(ks), np.log(ts), 1)[0])
    return {"per_K": out, "empirical_exponent_in_K": slope, "J": J, "D": D}


# ------------------------------------------------------- C: logsumexp decomposition
def collect_arrays(fx, cap: int = 6000):
    arrays = []
    original = smf.predecessor_terms

    def recording(*a, **kw):
        out = original(*a, **kw)
        terms = out[2]
        if terms.size and len(arrays) < cap:
            finite = terms[np.isfinite(terms)]
            if finite.size:
                arrays.append(np.array(finite, copy=True))
        return out

    smf.predecessor_terms = recording
    try:
        run_forward_all(fx)
    finally:
        smf.predecessor_terms = original
    return arrays


def logsumexp_decomposition(arrays, repeats: int = 5) -> dict:
    n = len(arrays)

    def noop(a):
        return a

    def inline(a):
        top = a.max()
        return top + np.log(np.exp(a - top).sum())

    buf = np.empty(max(a.size for a in arrays), dtype=float)

    def inline_prealloc(a):
        m = a.size
        view = buf[:m]
        top = a.max()
        np.subtract(a, top, out=view)
        np.exp(view, out=view)
        return top + math.log(view.sum())

    def max_only(a):
        return a.max()

    def exp_only(a):
        return np.exp(a)

    def sum_only(a):
        return a.sum()

    variants = [("00_python_call_overhead", noop),
                ("01_max_only", max_only),
                ("02_exp_only_allocating", exp_only),
                ("03_sum_only", sum_only),
                ("10_inline_numpy", inline),
                ("11_inline_prealloc", inline_prealloc),
                ("20_scipy_logsumexp", lambda a: logsumexp(a))]
    results = {}
    for name, fn in variants:
        best = math.inf
        for _ in range(repeats):
            began = time.perf_counter()
            for a in arrays:
                fn(a)
            best = min(best, time.perf_counter() - began)
        results[name] = {"total_s": best, "us_per_call": 1e6 * best / n}

    ref = [float(logsumexp(a)) for a in arrays[:800]]
    alt = [float(inline(a)) for a in arrays[:800]]
    pre = [float(inline_prealloc(a)) for a in arrays[:800]]
    results["max_abs_disagreement_inline"] = float(max(abs(x - y) for x, y in zip(ref, alt)))
    results["max_abs_disagreement_prealloc"] = float(max(abs(x - y) for x, y in zip(ref, pre)))
    results["n_arrays"] = n
    results["mean_array_size"] = float(np.mean([a.size for a in arrays]))

    call = results["00_python_call_overhead"]["us_per_call"]
    scipy_us = results["20_scipy_logsumexp"]["us_per_call"]
    inline_us = results["10_inline_numpy"]["us_per_call"]
    prealloc_us = results["11_inline_prealloc"]["us_per_call"]
    results["split_us_per_call"] = {
        "python_call_overhead": call,
        "reduction_arithmetic_prealloc_minus_call": prealloc_us - call,
        "allocation_copy_inline_minus_prealloc": inline_us - prealloc_us,
        "scipy_array_api_shim_scipy_minus_inline": scipy_us - inline_us,
        "scipy_total": scipy_us,
    }
    return results


# --------------------------------------------------- F: attribution of a forward pass
ARITHMETIC = ("method 'reduce' of 'numpy.ufunc' objects", "method 'max' of 'numpy.ndarray'",
              "method 'sum' of 'numpy.ndarray'", "'exp'", "'log'", "logaddexp")
DISPATCH = ("xp_promote", "isdtype", "_preprocess_dtype", "_wrapreduction", "_asarray",
            "array_namespace", "_logsumexp", "logsumexp", "amax", "_amax", "sum",
            "xp_size", "_ureduce")
INTERPRETER = ("predecessor_terms", "forward", "method 'append' of 'list' objects",
               "isinf", "isfinite")
ALLOC = ("asarray", "array", "empty", "full", "zeros", "copy")


def bucket_for(name: str) -> str:
    for token in ARITHMETIC:
        if token in name:
            return "arithmetic"
    for token in DISPATCH:
        if token in name:
            return "dispatch"
    for token in INTERPRETER:
        if token in name:
            return "interpreter"
    for token in ALLOC:
        if token in name:
            return "allocation"
    return "other"


def attribute_forward(fx, top: int = 30) -> dict:
    profiler = cProfile.Profile()
    profiler.enable()
    run_forward_all(fx)
    profiler.disable()
    stats = pstats.Stats(profiler)
    rows = []
    for func, (calls, _, tottime, cumtime, _) in stats.stats.items():
        name = f"{Path(func[0]).name}:{func[1]}({func[2]})"
        rows.append({"function": name, "ncalls": int(calls),
                     "tottime_s": float(tottime), "cumtime_s": float(cumtime),
                     "bucket": bucket_for(name)})
    rows.sort(key=lambda r: -r["tottime_s"])
    total = sum(r["tottime_s"] for r in rows)
    buckets = {}
    for r in rows:
        buckets.setdefault(r["bucket"], 0.0)
        buckets[r["bucket"]] += r["tottime_s"]
    return {"total_self_time_s": total,
            "bucket_seconds": buckets,
            "bucket_share": {k: v / total for k, v in buckets.items()},
            "top": rows[:top]}


def main() -> None:
    checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    fx = fixture(checkpoint)

    # an uninstrumented reference time for one forward pass over every trace
    best = math.inf
    for _ in range(3):
        began = time.perf_counter()
        run_forward_all(fx)
        best = min(best, time.perf_counter() - began)

    report = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "state_origin": fx["origin"],
        "loadavg": list(os.getloadavg()),
        "forward_all_traces_seconds_best_of_3": best,
        "B_census": census(fx),
        "C_logsumexp": logsumexp_decomposition(collect_arrays(fx)),
        "F_attribution": attribute_forward(fx),
        "scipy_version": __import__("scipy").__version__,
        "numpy_version": np.__version__,
    }
    b = report["B_census"]
    report["B_scaling_in_K"] = scaling_in_K(J=40, D=b["max_width"], Dmin=b["min_width"],
                                            K_values=(3, 5, 10, 20))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "forward_anatomy.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
