"""FULL-LATENT step 1c — PER-SWEEP records, to explain the COND sweep-time bimodality.

    PYTHONPATH=src python scripts/full_latent_persweep_profile.py --cond-sweeps 200 --marg-sweeps 120

Measurement only.  Nothing in `src/` is edited and no formal chain file is opened for
writing.  Throwaway sweeps advance a COPY of a checkpoint state under a private RNG.

`full_latent_sweep_profile.py` reports only summary statistics, so a sweep-time
distribution with mean < median cannot be attributed there.  This script keeps one row
per sweep carrying, side by side,

  * cost      : wall, cpu, and every exclusive phase timer
  * work      : predecessor_terms calls, logsumexp calls, predecessor options summed,
                legal candidate blocks in the table, segments drawn by FFBS
  * structure : whether a structural attempt fired, whether U moved, whether H moved,
                and the exact hash of H = h(U)
  * ambient   : wall-clock offset and 1-minute load average at the sweep boundary

so that "did the work change" and "did the machine change" are separable by regression
rather than by assertion.  The counter wrappers are counting only -- no per-call timing --
but they are Python-level and therefore not free; the observer cost is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "mcmc_original" / "full_latent_perf"

# reuse the audited fixture/timer machinery rather than re-deriving it
_spec = importlib.util.spec_from_file_location(
    "_sweep_profile", ROOT / "scripts" / "full_latent_sweep_profile.py")
_sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sp)

import hpop.mcmc_original.matched_full_latent as mfl
import hpop.mcmc_original.semi_markov_ffbs as rjf


class Counters:
    """Count-only wrappers around the two hot callables inside the forward recursion."""

    def __init__(self) -> None:
        self.reset()
        self._patches = []

    def reset(self) -> None:
        self.pt_calls = 0
        self.pt_options = 0
        self.lse_calls = 0

    def install(self) -> None:
        real_pt = rjf.predecessor_terms
        real_lse = rjf.logsumexp

        def counted_pt(*a, **kw):
            out = real_pt(*a, **kw)
            self.pt_calls += 1
            self.pt_options += out[2].size
            return out

        def counted_lse(*a, **kw):
            self.lse_calls += 1
            return real_lse(*a, **kw)

        rjf.predecessor_terms = counted_pt
        rjf.logsumexp = counted_lse
        self._patches = [("predecessor_terms", real_pt), ("logsumexp", real_lse)]

    def restore(self) -> None:
        for name, original in self._patches:
            setattr(rjf, name, original)
        self._patches = []


def h_hash(u_by_skill) -> str:
    """H = h(U) is one precedence matrix per skill; hash the whole stack."""
    arr = np.asarray(u_by_skill, dtype=float)
    digest = hashlib.blake2s()
    for skill in range(arr.shape[0]):
        precedence = mfl.precedence_from_u(arr[skill])
        digest.update(np.ascontiguousarray(precedence).tobytes())
    return digest.hexdigest()[:16]


def legal_blocks(sampler) -> int:
    """Finite entries in the per-trace candidate block-score tables = legal (block, skill)."""
    tables = getattr(sampler.tables, "_tables", None)
    if not tables:
        return -1
    total = 0
    for value in (tables.values() if isinstance(tables, dict) else tables):
        arr = np.asarray(getattr(value, "log_block_scores", value), dtype=float)
        total += int(np.isfinite(arr).sum())
    return total


def run_arm(arm: str, sweeps: int, warmup: int, checkpoint: Path | None,
            count_work: bool) -> dict:
    setup = _sp.build(arm, checkpoint)
    sampler, state = setup["sampler"], setup["state"]
    rng = np.random.default_rng(_sp.PROFILE_SEED + (0 if arm == mfl.FULL_COND else 1))

    for _ in range(warmup):
        state, _ = mfl.full_latent_sweep_once(state, sampler, rng)

    timer = _sp.ExclusiveTimer()
    _sp.install(timer)
    timer.enabled = True
    counters = Counters()
    if count_work:
        counters.install()

    rows = []
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t_origin = time.perf_counter()
    try:
        for i in range(sweeps):
            timer.reset()
            counters.reset()
            load0 = os.getloadavg()[0]
            offset = time.perf_counter() - t_origin
            wall0, cpu0 = time.perf_counter(), time.process_time()
            state, info = mfl.full_latent_sweep_once(state, sampler, rng)
            wall = time.perf_counter() - wall0
            cpu = time.process_time() - cpu0

            record = info["structural_record"]
            phases = dict(timer.exclusive)
            rows.append({
                "i": i,
                "offset_s": offset,
                "loadavg1": load0,
                "wall_s": wall,
                "cpu_s": cpu,
                "ffbs_forward_s": phases.get("ffbs_forward", 0.0),
                "marg_forward_s": phases.get("marg_forward", 0.0),
                "forward_total_s": (phases.get("ffbs_forward", 0.0)
                                    + phases.get("marg_forward", 0.0)),
                "emission_s": (phases.get("emission_batched_build", 0.0)
                               + phases.get("emission_refresh", 0.0)
                               + phases.get("emission_fast_build", 0.0)),
                "backward_s": phases.get("ffbs_backward", 0.0),
                "piP_s": (phases.get("gibbs_other", 0.0) + phases.get("gibbs_counts", 0.0)
                          + phases.get("gibbs_sample_P", 0.0)),
                "target_s": sum(v for k, v in phases.items() if k.startswith("target_")),
                "pt_calls": counters.pt_calls,
                "pt_options": counters.pt_options,
                "lse_calls": counters.lse_calls,
                "legal_blocks": legal_blocks(sampler),
                "segments": int(sum(len(s) for s in state.segmentations)),
                "structural": bool(info["scheduled_structural"]),
                "u_changed": bool(record["accepted"]) if record else False,
                "h_changed": bool(record["h_changed"]) if record else False,
                "h_hash": h_hash(state.u_by_skill),
                "iteration": int(state.iteration),
            })
    finally:
        timer.enabled = False
        timer.restore()
        counters.restore()
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return {"arm": arm, "state_origin": setup["origin"], "counters_installed": count_work,
            "config": {"structural_cadence": setup["config"].structural_cadence,
                       "structural_scale": setup["config"].structural_scale,
                       "table_source": setup["config"].table_source},
            "rss_mb": {"before": rss0 / 1e6, "after": rss1 / 1e6},
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond-sweeps", type=int, default=200)
    parser.add_argument("--marg-sweeps", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--label", type=str, default="persweep")
    parser.add_argument("--no-counters", action="store_true")
    args = parser.parse_args()

    checkpoints = {}
    if args.checkpoint_dir:
        d = Path(args.checkpoint_dir)
        checkpoints = {mfl.FULL_COND: d / "full_cond_0.npz",
                       mfl.FULL_MARG: d / "full_marg_0.npz"}

    report = {"label": args.label, "source_commit": _sp.source_commit(),
              "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "machine": {"cpu_count": os.cpu_count(),
                          "loadavg_at_start": list(os.getloadavg())},
              "arms": {}}
    for arm, n in ((mfl.FULL_COND, args.cond_sweeps), (mfl.FULL_MARG, args.marg_sweeps)):
        if n <= 0:
            continue
        print(f"[persweep] {arm}: {n} sweeps", flush=True)
        report["arms"][arm] = run_arm(arm, n, args.warmup, checkpoints.get(arm),
                                      not args.no_counters)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"persweep_{args.label}.json"
    path.write_text(json.dumps(_sp.jsonable(report), indent=2, sort_keys=True))
    print(f"[persweep] wrote {path}")


if __name__ == "__main__":
    main()
