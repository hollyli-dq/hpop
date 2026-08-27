#!/usr/bin/env python3
"""Measure per-sweep cost on THIS machine and project the formal wall-clock.

Section 11 of the prompt requires the schedule A / schedule B choice to be made from
timing and memory alone, before any formal chain starts, and never from recovery or truth.
This script produces that estimate. It must be run on the machine that will do the work:
per-sweep cost depends on core speed, BLAS and memory bandwidth, and a projection carried
over from another machine is not evidence about this one.

It measures the rungs you ask for and log-interpolates the rest of the ladder, marking
which rows were interpolated. Interpolation is honest here because the cost curve in `K`
is smooth and monotone; it is not a substitute for measuring the rung you actually care
about, and `--rungs 3 5 10 20 30` measures all five if you can spare the minutes.

    python scripts/k_ladder/timing_probe.py --rungs 3 10 30 --sweeps 20
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_cpa.corpus import generate_ladder_corpus                # noqa: E402
from hpop.mcmc_cpa.ladder_runner import (LEARNED_ORDER, ORACLE_ORDER,  # noqa: E402
                                         SUPPORT_ONLY, run_ladder_chain)
from hpop.mcmc_cpa.nested_library import draw_master_library, K_LADDER  # noqa: E402
from hpop.mcmc_original.stage6e_state import Stage6EModel              # noqa: E402

ARMS = ((SUPPORT_ONLY, 0), (ORACLE_ORDER, 0), (LEARNED_ORDER, 10))


def peak_rss_gib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 3) if sys.platform == "darwin" else rss / (1024 ** 2)


def measure(library, k: int, sweeps: int, replicate: int) -> dict:
    corpus = generate_ladder_corpus(library, k, replicate)
    u_by_skill, role_maps = library.prefix(k)
    model = Stage6EModel(traces=corpus.traces("train"), epsilon=0.02, delta_b=0.15,
                         n_skills=k, n_roles=library.n_roles, min_width=3, max_width=12,
                         infer_pi_P=True, eta_initial=1.0, eta_transition=1.0)
    out = {}
    for arm, every in ARMS:
        began = time.perf_counter()
        run_ladder_chain(arm, model, role_maps, u_by_skill, chain=0, sweeps=sweeps,
                         warmup=sweeps, seed=5150 + k, thin=1,
                         u_every=every or 1, u_moves=1, replicate=replicate)
        out[arm] = (time.perf_counter() - began) / sweeps
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rungs", type=int, nargs="+", default=[3, 10, 30])
    p.add_argument("--sweeps", type=int, default=20)
    p.add_argument("--ladder", type=int, nargs="+", default=list(K_LADDER))
    p.add_argument("--schedule-a", type=int, default=50_000)
    p.add_argument("--schedule-b", type=int, default=40_000)
    p.add_argument("--replicates", type=int, default=2)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--cores", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    p.add_argument("--budget-hours", type=float, default=132.0)
    p.add_argument("--fallback-budget-hours", type=float, default=144.0)
    p.add_argument("--library-seed", type=int, default=0)
    p.add_argument("--replicate", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "k_ladder" / "timing_probe.json")
    args = p.parse_args()

    library, _ = draw_master_library(args.library_seed)
    measured = {}
    print(f"measuring {args.sweeps} sweeps per arm per rung "
          f"(single thread; pin OMP_NUM_THREADS=1)\n")
    print(f"{'K':>4} {'support s/sw':>13} {'oracle s/sw':>12} {'learned s/sw':>13}")
    print("-" * 46)
    for k in sorted(args.rungs):
        measured[k] = measure(library, k, args.sweeps, args.replicate)
        m = measured[k]
        print(f"{k:>4} {m[SUPPORT_ONLY]:>13.4f} {m[ORACLE_ORDER]:>12.4f} "
              f"{m[LEARNED_ORDER]:>13.4f}")

    def cost(arm, k):
        """Returns `(seconds_per_sweep, quality)` with quality in
        {"measured", "interpolated", "EXTRAPOLATED"}.

        Extrapolation past the measured range is reported as such and never silently.
        An earlier version fell back to the nearest measured rung when there was nothing
        above `k`, which made an unmeasured `K = 30` look exactly as cheap as a measured
        `K = 10` and turned a 6x under-projection into a confident "fits budget: yes".
        Under-projecting the most expensive rung is the one direction that matters here.
        """
        if k in measured:
            return measured[k][arm], "measured"
        keys = sorted(measured)
        if len(keys) < 2:
            raise SystemExit("need at least two measured rungs to project the ladder")
        below = [x for x in keys if x < k]
        above = [x for x in keys if x > k]
        if below and above:
            lo, hi, quality = max(below), min(above), "interpolated"
        else:
            # outside the measured range: use the outermost measured pair's exponent
            lo, hi = (keys[-2], keys[-1]) if not above else (keys[0], keys[1])
            quality = "EXTRAPOLATED"
        e = math.log(measured[hi][arm] / measured[lo][arm]) / math.log(hi / lo)
        return measured[lo][arm] * (k / lo) ** e, quality

    for label, sweeps in (("A", args.schedule_a), ("B", args.schedule_b)):
        print(f"\n=== schedule {label}: {sweeps:,} sweeps per chain ===")
        print(f"{'K':>4} {'support h':>10} {'oracle h':>9} {'learned h':>10}")
        per_arm = {a: 0.0 for a, _ in ARMS}
        rows, any_extrapolated = [], False
        for k in sorted(args.ladder):
            row = {"K": k}
            for arm, _ in ARMS:
                sec, quality = cost(arm, k)
                hours = sec * sweeps / 3600
                per_arm[arm] += hours
                row[arm] = hours
                row[f"{arm}_quality"] = quality
            rows.append(row)
            quality = row[f"{SUPPORT_ONLY}_quality"]
            any_extrapolated |= quality == "EXTRAPOLATED"
            mark = {"measured": "", "interpolated": "  (interpolated)",
                    "EXTRAPOLATED": "  (EXTRAPOLATED -- measure this rung)"}[quality]
            print(f"{k:>4} {row[SUPPORT_ONLY]:>10.1f} {row[ORACLE_ORDER]:>9.1f} "
                  f"{row[LEARNED_ORDER]:>10.1f}{mark}")

        n = args.replicates * args.chains
        grand = sum(per_arm.values()) * n
        learned_share = per_arm[LEARNED_ORDER] * n
        print(f"\nper (replicate, chain) over the ladder: "
              f"support {per_arm[SUPPORT_ONLY]:.1f} h, "
              f"oracle {per_arm[ORACLE_ORDER]:.1f} h, "
              f"learned {per_arm[LEARNED_ORDER]:.1f} h")
        print(f"x {args.replicates} replicates x {args.chains} chains = "
              f"{grand:.0f} chain-hours "
              f"({learned_share:.0f} h, {100*learned_share/grand:.0f}%, is learned-order)")
        print(f"\n{'cores':>6} {'wall-clock h':>13}  {'fits budget':>12}")
        budget = args.budget_hours if label == "A" else args.fallback_budget_hours
        for c in args.cores:
            wall = grand / c
            if any_extrapolated:
                verdict = "UNRELIABLE"
            else:
                verdict = "yes" if wall <= budget else "NO"
            print(f"{c:>6} {wall:>13.1f}  {verdict:>12}")
        if any_extrapolated:
            top = max(args.ladder)
            print(f"\n  !! Some rungs were EXTRAPOLATED beyond the measured range, so no "
                  f"budget verdict\n     is given. The projection is least trustworthy "
                  f"exactly where it matters most --\n     the top rung dominates the "
                  f"total. Re-run with --rungs including {top}.")

    print(f"\npeak RSS this probe: {peak_rss_gib():.2f} GiB "
          f"(one process; multiply by concurrent chains)")
    print("\nNOTE: the schedule choice must use timing and memory ONLY, must be made "
          "before\n      any formal chain starts, and must never use recovery or truth.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema": "k-ladder-timing-probe/1.0.0",
        "settings": vars(args) | {"out": str(args.out), "ladder": list(args.ladder)},
        "measured_seconds_per_sweep": {str(k): v for k, v in measured.items()},
        "peak_rss_gib": peak_rss_gib(),
        "platform": sys.platform,
        "note": ("Per-sweep cost depends on core speed, BLAS and memory bandwidth. "
                 "Run this on the machine that will do the work; a projection carried "
                 "from another machine is not evidence about this one."),
    }, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
