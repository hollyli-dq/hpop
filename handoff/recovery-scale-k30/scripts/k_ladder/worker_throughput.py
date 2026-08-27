#!/usr/bin/env python3
"""Measure EFFECTIVE workers by running them, then project wall-clock honestly.

    projected_wall_time >= max(longest_chain_time, total_chain_hours / effective_workers)

Both terms matter and they fail in different ways. The first is a floor no parallelism
beats: a chain is sequential, so a plan whose longest single chain exceeds the ceiling is
infeasible at any worker count. The second is the aggregate.

**`effective_workers` must be measured, never assumed.** The count of independent chains
says nothing about throughput, and neither does the core count: memory bandwidth, shared
last-level cache, SMT siblings competing for one physical core, and thermal or power limits
all mean that `W` concurrent workers routinely deliver appreciably less than `W` times one
worker's rate. This script measures single-worker throughput, then measures it again under
concurrency, and reports the ratio it actually achieved.

    python scripts/k_ladder/worker_throughput.py --workers 1 2 4 8 --K 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

WORKER_SNIPPET = """
import os, sys, time, resource
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
try:                       # only if the environment actually has it
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
sys.path.insert(0, {src!r})
from hpop.mcmc_cpa.nested_library import draw_master_library
from hpop.mcmc_cpa.corpus import generate_ladder_corpus
from hpop.mcmc_cpa.ladder_runner import run_ladder_chain, ORACLE_ORDER
from hpop.mcmc_original.stage6e_state import Stage6EModel
lib, _ = draw_master_library(0)
K = {k}
c = generate_ladder_corpus(lib, K, 0)
u, maps = lib.prefix(K)
m = Stage6EModel(traces=c.traces("train"), epsilon=0.02, delta_b=0.15, n_skills=K,
                 n_roles=lib.n_roles, min_width=3, max_width=12, infer_pi_P=True,
                 eta_initial=1.0, eta_transition=1.0)
began = time.perf_counter()
run_ladder_chain(ORACLE_ORDER, m, maps, u, chain={chain}, sweeps={sweeps},
                 warmup={sweeps}, seed=4242, thin=1)
_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
_gib = _rss / (1024**3) if sys.platform == "darwin" else _rss / (1024**2)
print(({sweeps}) / (time.perf_counter() - began), _gib)
"""


def one_batch(workers: int, k: int, sweeps: int) -> tuple:
    """Aggregate sweeps/second and total RSS from `workers` concurrent single-thread jobs.

    Every threading knob is pinned to one, in the environment AND inside the worker. A
    benchmark that launches `W` "single-threaded" workers whose BLAS quietly opens eight
    threads each measures an oversubscribed machine, and the production run built on it
    would be oversubscribed too -- the throughput number would be wrong in the direction
    that matters, making the plan look feasible when it is not.
    """
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1")
    procs = [subprocess.Popen(
        [sys.executable, "-c", WORKER_SNIPPET.format(
            src=str(ROOT / "src"), k=k, chain=i, sweeps=sweeps)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
        for i in range(workers)]
    rates, rss = [], []
    for p in procs:
        out, err = p.communicate()
        if p.returncode != 0:
            raise SystemExit(f"worker failed:\n{err[-2000:]}")
        rate, gib = out.strip().splitlines()[-1].split()
        rates.append(float(rate))
        rss.append(float(gib))
    return sum(rates), sum(rss), max(rss)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--sweeps", type=int, default=30)
    p.add_argument("--total-chain-hours", type=float, default=660.0)
    p.add_argument("--longest-chain-hours", type=float, default=55.7)
    p.add_argument("--ceiling-hours", type=float, default=132.0)
    p.add_argument("--slowdowns", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "k_ladder_pilot" / "worker_throughput.json")
    args = p.parse_args()

    print(f"logical CPUs visible: {os.cpu_count()}")
    print("thread knobs pinned to 1 per worker: OMP, MKL, OPENBLAS, NUMEXPR, VECLIB"
          " (+ torch if present)")
    print(f"measuring aggregate throughput at K={args.K}, {args.sweeps} sweeps/worker\n")
    print(f"{'workers':>8} {'aggregate sw/s':>15} {'effective':>10} {'efficiency':>11} "
          f"{'aggregate RSS':>14} {'peak/worker':>12}")
    print("-" * 78)
    base, rows = None, []
    for w in sorted(args.workers):
        rate, total_rss, peak_rss = one_batch(w, args.K, args.sweeps)
        if base is None:
            base = rate
        effective = rate / base
        rows.append({"workers": w, "aggregate_sweeps_per_second": rate,
                     "speedup": effective, "efficiency": effective / w,
                     "aggregate_rss_gib": total_rss, "peak_rss_per_worker_gib": peak_rss})
        print(f"{w:>8} {rate:>15.3f} {effective:>10.2f} {100*effective/w:>10.0f}% "
              f"{total_rss:>13.2f}G {peak_rss:>11.2f}G")

    best = max(rows, key=lambda r: r["speedup"])
    print(f"\nbest measured effective workers: {best['speedup']:.2f} "
          f"at {best['workers']} concurrent processes "
          f"({100*best['efficiency']:.0f}% efficiency)")

    print(f"\nprojection: wall >= max(longest_chain, total/effective_workers)")
    print(f"  total_chain_hours   = {args.total_chain_hours:.0f}")
    print(f"  longest_chain_hours = {args.longest_chain_hours:.1f}")
    print(f"  ceiling             = {args.ceiling_hours:.0f} h\n")
    print(f"{'slowdown':>9} {'longest':>9} {'min effective workers for ceiling':>35}")
    print("-" * 58)
    needs = {}
    for s in args.slowdowns:
        longest = args.longest_chain_hours * s
        total = args.total_chain_hours * s
        if longest > args.ceiling_hours:
            needs[s] = None
            print(f"{s:>8.1f}x {longest:>9.1f} "
                  f"{'INFEASIBLE - longest chain alone exceeds ceiling':>35}")
            continue
        need = math.ceil(total / args.ceiling_hours)
        needs[s] = need
        print(f"{s:>8.1f}x {longest:>9.1f} {need:>35}")

    print("\nA worker count is only usable if the machine actually delivers it. "
          "Compare the\n'min effective workers' above against the measured effective "
          "workers, not against\ncore count and not against the number of independent "
          "chains.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema": "k-ladder-worker-throughput/1.0.0", "namespace": "PILOT",
        "settings": vars(args) | {"out": str(args.out)},
        "logical_cpus": os.cpu_count(), "measurements": rows,
        "best_effective_workers": best["speedup"],
        "min_effective_workers_by_slowdown": needs,
        "formula": "wall >= max(longest_chain_time, total_chain_hours/effective_workers)",
        "note": ("effective_workers is measured by running concurrent workers, never "
                 "inferred from core count or from the number of independent chains"),
    }, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
