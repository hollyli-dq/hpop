#!/usr/bin/env python3
"""Split the pilot manifest across machines and show what each will actually do.

`run_pilot_job.py --slice I/N` takes every job whose manifest index is congruent to `I`
modulo `N`. That is not an arbitrary choice. The manifest's innermost loop is the chain,
so with `N = 4` machine `i` receives **chain `i` of every cell** -- an exactly equal share
of the expensive `K = 30` work and of the trivial `K = 3` work. Slicing by contiguous
blocks instead would hand one machine most of the `K = 30` chains and leave another idle.

This script prints the split, estimates each machine's runtime from measured per-move
costs, and verifies the slices are disjoint and cover the manifest exactly.

    python scripts/k_ladder/fleet_plan.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Seconds per collapsed U move, measured on an Apple M4 at one thread. Used ONLY to
#: estimate wall-clock; it never affects which jobs run or how they are assigned.
MEASURED_U_MOVE_SECONDS = {3: 0.139, 10: 0.607, 30: 3.219}
MEASURED_FFBS_SECONDS = {3: 0.0156, 10: 0.0972, 30: 0.7893}


def interpolate(table: dict, k: int) -> float:
    if k in table:
        return table[k]
    keys = sorted(table)
    lo = max([x for x in keys if x < k], default=keys[0])
    hi = min([x for x in keys if x > k], default=keys[-1])
    if lo == hi:
        return table[lo]
    e = math.log(table[hi] / table[lo]) / math.log(hi / lo)
    return table[lo] * (k / lo) ** e


def job_seconds(job: dict) -> float:
    k = int(job["K"])
    return (job["M_K"] * interpolate(MEASURED_U_MOVE_SECONDS, k)
            + job["sweeps"] * interpolate(MEASURED_FFBS_SECONDS, k))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path,
                   default=ROOT / "results" / "k_ladder_pilot" / "pilot_manifest.json")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--effective-workers-per-machine", type=float, default=4.22,
                   help="measure this with worker_throughput.py on each machine")
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    jobs = manifest["jobs"]
    n = int(args.workers)

    slices = {i: [j for idx, j in enumerate(jobs) if idx % n == i] for i in range(n)}
    covered = sum(len(v) for v in slices.values())
    keys = [j["key"] for i in range(n) for j in slices[i]]
    if covered != len(jobs) or len(set(keys)) != len(jobs):
        raise SystemExit("slices do not partition the manifest exactly")

    print(f"manifest: {len(jobs)} jobs   workers: {n}\n")
    print(f"{'slice':>6} {'jobs':>6} {'chain-hours':>12} {'longest job (min)':>18} "
          f"{'K mix':>28}")
    print("-" * 76)
    totals = []
    for i in range(n):
        secs = [job_seconds(j) for j in slices[i]]
        total_h = sum(secs) / 3600
        totals.append(total_h)
        mix = Counter(j["K"] for j in slices[i])
        mix_s = " ".join(f"K{k}:{mix[k]}" for k in sorted(mix))
        print(f"{i:>6} {len(slices[i]):>6} {total_h:>12.1f} {max(secs)/60:>18.1f} "
              f"{mix_s:>28}")

    spread = max(totals) - min(totals)
    print(f"\nload balance: {min(totals):.1f}-{max(totals):.1f} chain-hours per machine "
          f"(spread {spread:.2f} h, {100*spread/max(totals):.1f}%)")

    eff = float(args.effective_workers_per_machine)
    longest_min = max(job_seconds(j) for j in jobs) / 60
    print(f"\nassuming {eff:.2f} effective workers per machine "
          f"(measure with worker_throughput.py):")
    print(f"{'slice':>6} {'wall-clock h':>13}")
    for i, t in enumerate(totals):
        print(f"{i:>6} {max(t / eff, longest_min / 60):>13.1f}")
    print(f"\nwall-clock floor is the longest single chain: {longest_min:.1f} min "
          f"({longest_min/60:.2f} h). No worker count beats it.")

    print("\ncommands, one per machine:\n")
    for i in range(n):
        print(f"  machine {i}:  OMP_NUM_THREADS=1 PYTHONPATH=src \\\n"
              f"                python3 scripts/k_ladder/run_pilot_job.py --slice {i}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
