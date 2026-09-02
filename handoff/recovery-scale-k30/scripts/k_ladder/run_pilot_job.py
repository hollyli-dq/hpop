#!/usr/bin/env python3
"""Execute ONE pilot job from the manifest and write its immutable output.

A job is `(replicate, K, X, u_scale, chain)`. It needs nothing from any other job, so the
whole 360-cell factorial distributes across a fleet by handing each machine a slice of the
manifest. Only the chain is indivisible.

Outputs are immutable: an existing file is never overwritten. A rerun either skips (the
default, so a fleet can be restarted safely) or fails loudly with `--no-skip-existing`.
Every output carries the code tag and commit, the corpus/library digest, the replicate,
`K`, `X`, `u_scale`, chain, the CRN root, the realised quota and per-role attempt summary,
runtime, peak RSS and a completion status -- so a summary built from these files can prove
what produced each number rather than trusting a directory name.

    python scripts/k_ladder/run_pilot_job.py --manifest ... --key rep0_K3_X50_s0.25_chain0
    python scripts/k_ladder/run_pilot_job.py --manifest ... --slice 0/8     # fleet member
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                                       # noqa: E402

from hpop.mcmc_cpa.corpus import generate_ladder_corpus                  # noqa: E402
from hpop.mcmc_cpa.crn import CommonRandomNumbers                        # noqa: E402
from hpop.mcmc_cpa.ladder_runner import LEARNED_ORDER, run_ladder_chain  # noqa: E402
from hpop.mcmc_cpa.nested_library import draw_master_library             # noqa: E402
from hpop.mcmc_cpa.seeds import LadderSeeds                              # noqa: E402
from hpop.mcmc_original.latent_poset import precedence_from_u            # noqa: E402
from hpop.mcmc_original.stage6e_state import Stage6EModel                # noqa: E402


_PROVENANCE_CACHE: dict | None = None


def runtime_code_provenance() -> dict:
    """What THIS process is actually running, read from git at run time.

    Copying the manifest's `code_commit` into the output would make the cross-machine
    consistency check vacuous, so the commit is read from the working tree. The v2 fleet
    then taught the second lesson: under 32 concurrent workers on shared NFS, git fails
    transiently, and the old `except Exception: return None` swallowed the stderr and
    wrote records with a null commit -- which correctly BLOCKED the collector but left
    the cause undiagnosable. So now: captured ONCE per process (it cannot change
    mid-process), retried with backoff, stderr recorded on every failure, and if git
    still cannot answer, the worker REFUSES TO START rather than manufacturing outputs
    that will block the whole fleet's aggregation four hours later.
    """
    global _PROVENANCE_CACHE
    if _PROVENANCE_CACHE is not None:
        return _PROVENANCE_CACHE

    def git(*args, attempts: int = 5) -> tuple:
        errors = []
        for attempt in range(attempts):
            try:
                proc = subprocess.run(["git", "-C", str(ROOT), *args],
                                      capture_output=True, text=True, timeout=30)
                if proc.returncode == 0:
                    return proc.stdout.strip(), errors
                errors.append(f"attempt {attempt}: exit {proc.returncode}: "
                              f"{proc.stderr.strip()[:300]}")
            except Exception as exc:                     # timeout, OSError, ...
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            time.sleep(0.5 * (2 ** attempt))
        return None, errors

    commit, commit_errors = git("rev-parse", "HEAD")
    describe, describe_errors = git("describe", "--tags", "--always", "--dirty")
    dirty, dirty_errors = git("status", "--porcelain")
    if commit is None:
        raise SystemExit(
            "cannot establish the runtime git commit after retries; refusing to run "
            "jobs whose provenance would be null and block the collector. Git said:\n"
            + "\n".join(commit_errors))
    _PROVENANCE_CACHE = {
        "runtime_commit": commit,
        "runtime_describe": describe,
        "runtime_tree_dirty": bool(dirty),
        "provenance_git_errors": commit_errors + describe_errors + dirty_errors,
    }
    return _PROVENANCE_CACHE


def peak_rss_gib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 3) if sys.platform == "darwin" else rss / (1024 ** 2)


def closure_bits(u_by_skill) -> np.ndarray:
    u = np.asarray(u_by_skill, dtype=float)
    off = ~np.eye(u.shape[1], dtype=bool)
    return np.concatenate([np.asarray(precedence_from_u(u[k]))[off]
                           for k in range(u.shape[0])])


def execute(job: dict, library) -> dict:
    k = int(job["K"])
    corpus = generate_ladder_corpus(library, k, int(job["replicate"]))
    u_truth, role_maps = library.prefix(k)
    model = Stage6EModel(traces=corpus.traces("train"), epsilon=float(job["epsilon"]),
                         delta_b=0.15, n_skills=k, n_roles=library.n_roles,
                         min_width=3, max_width=12, infer_pi_P=True,
                         eta_initial=1.0, eta_transition=1.0)
    crn = CommonRandomNumbers(int(job["replicate"]), k, int(job["chain"]),
                              seeds=LadderSeeds(root=int(job["crn_root"])))
    began = time.perf_counter()
    result = run_ladder_chain(
        LEARNED_ORDER, model, role_maps, u_truth, chain=int(job["chain"]),
        sweeps=int(job["sweeps"]), warmup=int(job["warmup"]),
        seed=int(job["crn_root"]), epsilon=float(job["epsilon"]),
        thin=int(job["thin"]), u_every=int(job["u_every"]),
        u_scale=float(job["u_scale"]), replicate=int(job["replicate"]),
        target_u_attempts_per_role=float(job["pilot_target_u_attempts_per_role"]),
        crn=crn)
    seconds = time.perf_counter() - began

    bits = np.array([closure_bits(u) for u in result["draws"]["u"]], dtype=bool)
    return {
        "schema": "k-ladder-learned-pilot-job/1.0.0",
        "namespace": "PILOT",
        "status": "complete",
        "job": job,
        "code_tag": job["code_tag"],
        "code_commit_declared_in_manifest": job["code_commit"],
        **runtime_code_provenance(),
        "library_digest_at_K": job["library_digest_at_K"],
        "crn_root": job["crn_root"], "crn": result["crn"],
        "arm": result["arm"], "u_kernel": result["u_kernel"],
        "N_U": result["N_U"], "N_U_expected": result["N_U_expected"],
        "u_quota_schedule": result["u_quota_schedule"],
        "u_role_attempt_summary": result["u_role_attempt_summary"],
        "u_proposed_burnin": result["u_proposed_burnin"],
        "u_proposed_retained": result["u_proposed_retained"],
        "u_accepted_burnin": result["u_accepted_burnin"],
        "u_accepted_retained": result["u_accepted_retained"],
        "u_acceptance_rate_burnin": result["u_acceptance_rate_burnin"],
        "u_acceptance_rate_retained": result["u_acceptance_rate_retained"],
        "retained_draws": result["retained_draws"],
        "ffbs_states_changed_total": result["ffbs_states_changed_total"],
        "closure_bits": bits.astype(np.uint8).tolist(),
        "seconds": seconds, "peak_rss_gib": peak_rss_gib(),
        "hostname": os.uname().nodename, "platform": sys.platform,
        "truth_used": False,
        "note": ("closure_bits are the induced precedence indicators of the retained U "
                 "draws, indexed by U-update event. The sealed truth is never read."),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path,
                   default=ROOT / "results" / "k_ladder_pilot" / "pilot_manifest.json")
    p.add_argument("--key", default=None, help="run exactly this job key")
    p.add_argument("--slice", default=None,
                   help="I/N: run every job whose index is congruent to I modulo N")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-skip-existing", action="store_true")
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    jobs = manifest["jobs"]
    if args.key:
        jobs = [j for j in jobs if j["key"] == args.key]
        if not jobs:
            raise SystemExit(f"no job with key {args.key!r}")
    if args.slice:
        i, n = (int(x) for x in args.slice.split("/"))
        jobs = [j for idx, j in enumerate(jobs) if idx % n == i]
    if args.limit:
        jobs = jobs[:args.limit]

    library, _ = draw_master_library(manifest["settings"]["library_seed"])
    done = failed = skipped = 0
    for job in jobs:
        out = Path(job["output_path"])
        if out.exists():
            if args.no_skip_existing:
                raise SystemExit(f"refusing to overwrite immutable output {out}")
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            record = execute(job, library)
        except Exception:                       # a failed cell is recorded, not hidden
            record = {"schema": "k-ladder-learned-pilot-job/1.0.0", "namespace": "PILOT",
                      "status": "failed", "job": job,
                      "traceback": traceback.format_exc()[-4000:],
                      "hostname": os.uname().nodename}
            failed += 1
        else:
            done += 1
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(record, default=str))
        tmp.replace(out)                        # atomic: no half-written output
        print(f"{record['status']:>8}  {job['key']}"
              + (f"  {record.get('seconds', 0):.1f}s" if record["status"] == "complete"
                 else ""), flush=True)

    print(f"\ncomplete {done}, failed {failed}, skipped {skipped}, "
          f"of {len(jobs)} selected")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
