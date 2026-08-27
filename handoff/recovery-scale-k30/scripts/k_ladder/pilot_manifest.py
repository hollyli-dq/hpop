#!/usr/bin/env python3
"""Generate the COMPLETE learned-order pilot job manifest, before any result is seen.

The registered pilot is a full factorial:

    K        in {3, 5, 10, 20, 30}
    X        in {50, 100, 166.7}          attempts per role vector (a MEAN, see u_quota)
    u_scale  in {0.25, 0.5, 1.0}
    replicate in {0, 1}
    chain    in {0, 1, 2, 3}

    5 x 3 x 3 x 2 x 4 = 360 chain executions

The manifest is generated **in full and in advance**. No cell is dropped, added or
re-parameterised on the strength of an intermediate result: a pilot that prunes itself as
it goes has selected on its own output, and the frozen pass rule would no longer be
frozen. Every job is emitted even if an earlier one looks hopeless.

## What is and is not divisible

Only a **chain** is indivisible. A job is keyed by

    (replicate, K, X, u_scale, chain)

and may run on any machine, in any order, with no communication -- so the 360 jobs
parallelise across a fleet and the wall-clock floor is one chain, not one replicate and
not one rung.

## Pilot streams are separate from production streams

The pilot draws from its own RNG root. Reusing production roots would make the production
run a continuation of chains whose scale was chosen by looking at them, and the pilot's
selection effect would leak into the production draws.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_cpa.nested_library import K_LADDER, draw_master_library   # noqa: E402
from hpop.mcmc_cpa.u_quota import quota_schedule, update_events          # noqa: E402

#: The pilot's own entropy root. MUST differ from the production root (6_500_000).
PILOT_ROOT_ENTROPY = 6_700_000

TARGETS = (50.0, 100.0, 166.7)
SCALES = (0.25, 0.5, 1.0)
REPLICATES = (0, 1)
CHAINS = (0, 1, 2, 3)


def job_key(replicate, k, target, scale, chain) -> str:
    return (f"rep{int(replicate)}_K{int(k)}_X{float(target):g}"
            f"_s{float(scale):g}_chain{int(chain)}")


def job_hash(payload: dict) -> str:
    canonical = json.dumps({k: payload[k] for k in sorted(payload)
                            if k not in ("job_hash", "output_path")},
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build(args) -> dict:
    library, library_meta = draw_master_library(args.library_seed)
    events_pilot = update_events(args.sweeps, args.u_every).size
    events_prod = update_events(args.production_sweeps, args.u_every).size
    ratio = events_pilot / events_prod if events_prod else 1.0

    jobs = []
    for replicate in REPLICATES:
        for k in K_LADDER:
            corpus_digest = library.library_digest(k)
            for target in TARGETS:
                pilot_target = target * ratio
                schedule = quota_schedule(pilot_target, k, library.n_roles,
                                          args.sweeps, args.warmup, args.u_every)
                for scale in SCALES:
                    for chain in CHAINS:
                        key = job_key(replicate, k, target, scale, chain)
                        payload = {
                            "key": key,
                            "replicate": int(replicate), "K": int(k),
                            "production_target_u_attempts_per_role": float(target),
                            "pilot_target_u_attempts_per_role": float(pilot_target),
                            "pilot_scaling_ratio": float(ratio),
                            "u_scale": float(scale), "chain": int(chain),
                            "sweeps": int(args.sweeps), "warmup": int(args.warmup),
                            "thin": int(args.thin), "u_every": int(args.u_every),
                            "epsilon": float(args.epsilon),
                            "library_seed": int(args.library_seed),
                            "library_digest_at_K": corpus_digest,
                            "M_K": int(schedule["total_quota_M_K"]),
                            "mean_attempts_per_role_total":
                                float(schedule["mean_attempts_per_role_total"]),
                            "crn_root": int(args.pilot_root),
                            "namespace": "PILOT",
                            "arm": "learned-order",
                            "code_tag": args.code_tag,
                            "code_commit": args.code_commit,
                        }
                        payload["job_hash"] = job_hash(payload)
                        payload["output_path"] = str(
                            Path(args.out_root) / f"rep{replicate}" / f"K{k}"
                            / f"X{target:g}" / f"scale{scale:g}"
                            / f"chain{chain}.json")
                        jobs.append(payload)

    keys = [j["key"] for j in jobs]
    if len(set(keys)) != len(keys):
        raise SystemExit("manifest contains duplicate job keys")
    paths = [j["output_path"] for j in jobs]
    if len(set(paths)) != len(paths):
        raise SystemExit("manifest contains duplicate output paths")

    return {
        "schema": "k-ladder-learned-pilot-manifest/1.0.0",
        "namespace": "PILOT",
        "generated_before_any_result": True,
        "factorial": {"K": list(K_LADDER), "X": list(TARGETS), "u_scale": list(SCALES),
                      "replicates": list(REPLICATES), "chains": list(CHAINS)},
        "n_jobs": len(jobs),
        "pilot_root_entropy": int(args.pilot_root),
        "production_root_entropy": 6_500_000,
        "roots_are_separate": int(args.pilot_root) != 6_500_000,
        "library": library_meta,
        "settings": vars(args) | {"out_root": str(args.out_root)},
        "divisibility": ("only a chain is indivisible; every job may run on a different "
                         "machine with no communication"),
        "no_pruning": ("every cell is emitted in advance and must be executed; dropping "
                       "a cell after seeing an intermediate result would select on the "
                       "pilot's own output and unfreeze the pass rule"),
        "jobs": jobs,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweeps", type=int, default=600)
    p.add_argument("--warmup", type=int, default=240)
    p.add_argument("--thin", type=int, default=4)
    p.add_argument("--u-every", type=int, default=10)
    p.add_argument("--production-sweeps", type=int, default=50_000)
    p.add_argument("--epsilon", type=float, default=0.02)
    p.add_argument("--library-seed", type=int, default=0)
    p.add_argument("--pilot-root", type=int, default=PILOT_ROOT_ENTROPY)
    p.add_argument("--code-tag", default="k30-learned-pilot-v1")
    p.add_argument("--code-commit", default="")
    p.add_argument("--out-root", type=Path,
                   default=ROOT / "results" / "k_ladder_pilot" / "factorial")
    p.add_argument("--manifest", type=Path,
                   default=ROOT / "results" / "k_ladder_pilot" / "pilot_manifest.json")
    args = p.parse_args()

    if int(args.pilot_root) == 6_500_000:
        raise SystemExit("the pilot root must differ from the production root")

    manifest = build(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, default=str))

    print(f"jobs: {manifest['n_jobs']}  "
          f"({len(manifest['factorial']['K'])} K x {len(TARGETS)} X x {len(SCALES)} "
          f"scale x {len(REPLICATES)} rep x {len(CHAINS)} chain)")
    print(f"pilot root {manifest['pilot_root_entropy']} "
          f"(production root {manifest['production_root_entropy']}, "
          f"separate: {manifest['roots_are_separate']})")
    print(f"\n{'K':>4} {'X':>7} {'M_K':>7} {'mean/role':>10}   jobs per (K,X)")
    seen = set()
    for j in manifest["jobs"]:
        tag = (j["K"], j["production_target_u_attempts_per_role"])
        if tag in seen:
            continue
        seen.add(tag)
        n = len(SCALES) * len(REPLICATES) * len(CHAINS)
        print(f"{j['K']:>4} {tag[1]:>7.1f} {j['M_K']:>7} "
              f"{j['mean_attempts_per_role_total']:>10.3f} {n:>16}")
    print(f"\nwrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
