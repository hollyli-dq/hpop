"""Measured speedup of the optimized backend over the frozen reference.

    PYTHONPATH=src python scripts/optimized_backend_bench.py <checkpoint_dir> [--reps 15]

The baseline is the REFERENCE sampler and sweep, not the optimized backend with its flags
off -- though that configuration is measured too, because it is the only way to show the
backend adds no overhead of its own.

Isolated speedups are not multiplied. O1, O3 and O4 all attack the same scipy dispatch
cost, so the product of their separate speedups overstates the stack badly. Each
configuration is measured directly and the MARGINAL contribution of each addition reported.

Protocol: all configurations in ONE process, interleaved one sweep each per round with the
order rotating, every sweep from an identical state copy under an identically seeded rng,
statistic = median over rounds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_full_latent as mfl                     # noqa: E402
from hpop.mcmc_optimized import COUNTERS, FLAGS, OptimizedFullLatentSampler   # noqa: E402
from hpop.mcmc_optimized import sweep_once as optimized_sweep_once            # noqa: E402

CORPUS_DIR = ROOT / "results" / "mcmc_original" / "matched_synthetic_formal_corpus"
OUT = ROOT / "results" / "mcmc_original" / "full_latent_perf"
SEED = 909_112_002

INDEPENDENT = [("reference", None), ("optimized[none]", []),
               ("O1_inline", ["inline_logsumexp"]),
               ("O2_cache", ["emission_hash_cache"]),
               ("O3_factorised", ["factorised_forward"]),
               ("O4_batched", ["batched_forward"])]
CUMULATIVE = [("reference", None), ("+O1", ["inline_logsumexp"]),
              ("+O2", ["inline_logsumexp", "emission_hash_cache"]),
              ("+O3", ["inline_logsumexp", "emission_hash_cache", "factorised_forward"]),
              ("+O4", ["inline_logsumexp", "emission_hash_cache", "factorised_forward",
                       "batched_forward"])]


def build(arm, checkpoint):
    corpus = mfl.load_frozen_observed_corpus(CORPUS_DIR)
    fixed = mfl.FullLatentFixed()
    model = mfl.build_full_latent_model(corpus.train, fixed)
    config = mfl.FullLatentConfig(arm=arm, structural_cadence=10, structural_scale=0.5,
                                  table_source="batched")
    reference = mfl.FullLatentSampler(model=model, fixed=fixed, config=config)
    optimized = OptimizedFullLatentSampler(model=model, fixed=fixed, config=config)
    state = mfl.FullLatentChain.load(checkpoint, reference).state.copy()
    return reference, optimized, state


def measure(reference, optimized, base_state, flags, seed):
    state = base_state.copy()
    rng = np.random.default_rng(seed)
    if flags is None:
        sampler, sweep = reference, mfl.full_latent_sweep_once
    else:
        FLAGS.all_off()
        FLAGS.apply(**{f: True for f in flags})
        sampler, sweep = optimized, optimized_sweep_once
    sampler.tables.refresh(state)          # warm as a running chain would be
    sampler.tables.mark_stale()
    COUNTERS.reset()
    wall0, cpu0 = time.perf_counter(), time.process_time()
    sweep(state, sampler, rng)
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    FLAGS.reset()
    return wall, cpu, COUNTERS.snapshot()


def run_block(reference, optimized, base_state, configs, reps, tag):
    walls, cpus, counters = ({n: [] for n, _ in configs}, {n: [] for n, _ in configs}, {})
    for rep in range(reps):
        order = configs[rep % len(configs):] + configs[:rep % len(configs)]
        for name, flags in order:
            wall, cpu, snap = measure(reference, optimized, base_state, flags, SEED)
            walls[name].append(wall)
            cpus[name].append(cpu)
            counters[name] = snap
        print(f"    [{tag}] round {rep + 1}/{reps}", flush=True)
    rows, previous = {}, None
    base = float(np.median(walls[configs[0][0]]))
    for name, _ in configs:
        w, c = np.array(walls[name]), np.array(cpus[name])
        med = float(np.median(w))
        rows[name] = {"wall_median_s": med, "wall_min_s": float(w.min()),
                      "wall_p90_s": float(np.percentile(w, 90)),
                      "cpu_median_s": float(np.median(c)), "n": int(w.size),
                      "speedup_vs_reference": base / med,
                      "marginal_vs_previous": (previous / med) if previous else 1.0,
                      "marginal_ms_saved": (1000.0 * (previous - med)) if previous else 0.0,
                      "counters": counters[name]}
        previous = med
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--reps", type=int, default=15)
    args = parser.parse_args()
    directory = Path(args.checkpoint_dir)

    report = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "reps": args.reps, "loadavg_at_start": list(os.getloadavg()), "arms": {}}
    for arm, name in ((mfl.FULL_COND, "full_cond_0.npz"),
                      (mfl.FULL_MARG, "full_marg_0.npz")):
        reference, optimized, state = build(arm, directory / name)
        plain = state.copy()
        structural = state.copy()
        structural.iteration = int(state.iteration) + (
            9 - int(state.iteration) % 10) % 10
        while (structural.iteration + 1) % 10 != 0:
            structural.iteration += 1
        assert (plain.iteration + 1) % 10 != 0
        print(f"  {arm}: independent", flush=True)
        ind = run_block(reference, optimized, plain, INDEPENDENT, args.reps, f"{arm}/ind")
        print(f"  {arm}: cumulative", flush=True)
        cum = run_block(reference, optimized, plain, CUMULATIVE, args.reps, f"{arm}/cum")
        print(f"  {arm}: cumulative structural", flush=True)
        stc = run_block(reference, optimized, structural, CUMULATIVE,
                        max(4, args.reps // 3), f"{arm}/struct")
        report["arms"][arm] = {"independent_plain": ind, "cumulative_plain": cum,
                               "cumulative_structural": stc,
                               "base_iteration": int(state.iteration)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "optimized_backend_bench.json").write_text(json.dumps(report, indent=2,
                                                                 sort_keys=True))
    print("wrote optimized_backend_bench.json")


if __name__ == "__main__":
    main()
