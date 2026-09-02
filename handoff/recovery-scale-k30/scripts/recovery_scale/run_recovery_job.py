#!/usr/bin/env python3
"""Run ONE chain-segment of the recovery experiment. The fleet's only worker.

A job = (phase, replicate, K, chain [, u_scale]). The worker loads the frozen dataset
(never regenerating anything), resumes the chain from its last checkpoint (exact resume
is guaranteed by the index-addressed CRN and pinned by tests), advances one registered
segment, and writes the segment draws + new checkpoint atomically. Workers never read
truth/. Between rounds the coordinator decides which chains continue.

    python scripts/recovery_scale/run_recovery_job.py --round-file <jobs.json> --slice i/8
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                                        # noqa: E402

from hpop.mcmc_cpa.recovery_regime import REGIME                          # noqa: E402
from hpop.mcmc_cpa.recovery_runner import chain_crn, init_state, run_segment  # noqa: E402
from hpop.mcmc_original.stage6e_state import Stage6EModel                 # noqa: E402

_PROVENANCE = None


def provenance() -> dict:
    global _PROVENANCE
    if _PROVENANCE is None:
        def git(*args, attempts=5):
            errors = []
            for attempt in range(attempts):
                try:
                    r = subprocess.run(["git", "-C", str(ROOT), *args],
                                       capture_output=True, text=True, timeout=30)
                    if r.returncode == 0:
                        return r.stdout.strip(), errors
                    errors.append(f"{attempt}: exit {r.returncode} "
                                  f"{r.stderr.strip()[:200]}")
                except Exception as exc:
                    errors.append(f"{attempt}: {type(exc).__name__}")
                time.sleep(0.5 * 2 ** attempt)
            return None, errors
        commit, err = git("rev-parse", "HEAD")
        if commit is None:
            raise SystemExit("no runtime git commit after retries:\n" + "\n".join(err))
        describe, err2 = git("describe", "--tags", "--always", "--dirty")
        dirty, err3 = git("status", "--porcelain")
        _PROVENANCE = {"runtime_commit": commit, "runtime_describe": describe,
                       "runtime_tree_dirty": bool(dirty),
                       "provenance_git_errors": err + err2 + err3}
    return _PROVENANCE


def load_model(dataset: Path, replicate: int, k: int):
    from hpop.mcmc_cpa.role_maps import RoleMaps

    payload = json.loads((dataset / "traces" / f"rep{replicate}_K{k}.json").read_text())
    traces = tuple(tuple(t["cpa"]) for t in payload["train"])
    # role maps come from the TRACES side of the dataset? No: the worker needs the role
    # maps to score -- they are part of the observation model (typed supports), not of
    # the sealed order truth. They are stored in the cell manifest by generate_dataset.
    maps_path = dataset / "traces" / f"rep{replicate}_K{k}_rolemaps.json"
    forward = np.asarray(json.loads(maps_path.read_text())["forward"], dtype=int)
    role_maps = RoleMaps(forward, int(json.loads(maps_path.read_text())["n_cpa"]))
    model = Stage6EModel(traces=traces, epsilon=REGIME.EPSILON, delta_b=REGIME.DELTA_B,
                         n_skills=int(k), n_roles=forward.shape[1],
                         min_width=REGIME.MIN_WIDTH, max_width=REGIME.MAX_WIDTH,
                         infer_pi_P=True, eta_initial=1.0, eta_transition=1.0)
    return model, role_maps


def run_job(job: dict, dataset: Path, work: Path) -> dict:
    replicate, k, chain = int(job["replicate"]), int(job["K"]), int(job["chain"])
    u_scale = float(job["u_scale"])
    cell = work / job["phase"] / f"rep{replicate}_K{k}_s{u_scale:g}" / f"chain{chain}"
    cell.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(cell.glob("checkpoint_*.json"))
    model, role_maps = load_model(dataset, replicate, k)
    crn = chain_crn(replicate, k, chain)
    if checkpoints:
        state = json.loads(checkpoints[-1].read_text())["state"]
    else:
        state = init_state(model, crn)
    if state["sweep"] >= int(job.get("cap_sweeps", REGIME.CAP_SWEEPS)):
        return {"status": "at_cap", "sweep": state["sweep"]}

    began = time.perf_counter()
    new_state, draws = run_segment(model, role_maps, state, crn, u_scale,
                                   sweeps=int(job.get("segment_sweeps",
                                                      REGIME.SEGMENT_SWEEPS)))
    seconds = time.perf_counter() - began
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gib = rss / 2**30 if sys.platform == "darwin" else rss / 2**20

    segment_index = new_state["sweep"] // int(job.get("segment_sweeps",
                                                      REGIME.SEGMENT_SWEEPS))
    record = {"schema": "recovery-scale-segment/1.0.0", "job": job,
              "state": new_state, "draws": draws,
              "seconds": seconds, "peak_rss_gib": rss_gib,
              "hostname": os.uname().nodename, **provenance()}
    tmp = cell / f"checkpoint_{segment_index:04d}.json.partial"
    tmp.write_text(json.dumps(record, default=str))
    tmp.replace(cell / f"checkpoint_{segment_index:04d}.json")
    return {"status": "advanced", "sweep": new_state["sweep"], "seconds": seconds}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round-file", type=Path, required=True)
    p.add_argument("--dataset", type=Path,
                   default=ROOT / "dataset" / "recovery_scale_v1")
    p.add_argument("--work", type=Path, default=ROOT / "results" / "recovery_scale")
    p.add_argument("--slice", default="0/1")
    args = p.parse_args()

    i, n = (int(x) for x in args.slice.split("/"))
    jobs = json.loads(args.round_file.read_text())["jobs"]
    mine = [j for idx, j in enumerate(jobs) if idx % n == i]
    done = 0
    for job in mine:
        result = run_job(job, args.dataset, args.work)
        done += result["status"] == "advanced"
        print(f"{result['status']:>9}  {job['phase']} rep{job['replicate']} "
              f"K{job['K']} s{job['u_scale']:g} chain{job['chain']} "
              f"-> sweep {result.get('sweep')}", flush=True)
    print(f"\n{done}/{len(mine)} advanced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
