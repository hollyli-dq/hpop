"""FULL-LATENT step 1 — where a FULL-COND / FULL-MARG sweep actually spends its time.

    PYTHONPATH=src python scripts/full_latent_sweep_profile.py --sweeps 300

Measurement only.  Nothing in `src/` is edited: the decomposition is obtained by wrapping
the sampler's own callables with an exclusive-time timer for the duration of the profiling
process, so every number below is produced by the *unmodified* kernel running throwaway
sweeps from a real formal checkpoint state.

The decomposition is the one the optimisation plan asks for,

    T_sweep = T_emission + T_forward + T_backward + T_target + T_piP + T_structural
              + T_validate + T_other

with `T_structural` reported both per-attempt (it fires once every `structural_cadence`
sweeps) and amortised per sweep, and with the retained-diagnostic cost reported separately
because it fires once every `thin` sweeps and is not part of the kernel.

Throwaway sweeps advance a COPY of a checkpoint state under a private RNG.  No formal
chain file is opened for writing and no truth file is opened at all.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import math
import os
import platform
import pstats
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# single-threaded BLAS: the formal launch runs 8 chains on 10 cores, so a profiler that
# fans out into threads would measure contention rather than the kernel
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import collapsed_u_kernel as cuk                      # noqa: E402
from hpop.mcmc_original import collapsed_u_likelihood as cul                  # noqa: E402
from hpop.mcmc_original import matched_full_latent as mfl                     # noqa: E402
from hpop.mcmc_original import recurrent_joint_ffbs_mcmc as rjf               # noqa: E402
from hpop.mcmc_original import semi_markov_ffbs as smf                        # noqa: E402
from hpop.mcmc_original import stage6e_block_table as sbt                     # noqa: E402
from hpop.mcmc_original import fast_block_tables as fbt                       # noqa: E402
from hpop.mcmc_original import stage6e_sampler as s6s                         # noqa: E402
from hpop.mcmc_original import stage6e_state as s6t                           # noqa: E402
from hpop.mcmc_original import transitions as trn                             # noqa: E402

CORPUS_DIR = ROOT / "results" / "mcmc_original" / "matched_synthetic_formal_corpus"
OUT = ROOT / "results" / "mcmc_original" / "full_latent_perf"
PROFILE_SEED = 909_112_001


# ------------------------------------------------------------------- exclusive timing
class ExclusiveTimer:
    """Wall seconds attributed to a callable, with nested callees subtracted out."""

    def __init__(self):
        self.exclusive = defaultdict(float)
        self.inclusive = defaultdict(float)
        self.calls = defaultdict(int)
        self._stack: list = []
        self._patches: list = []
        self.enabled = False

    def wrap(self, holder, name: str, bucket: str) -> None:
        original = getattr(holder, name)

        def wrapper(*args, __original=original, __bucket=bucket, **kwargs):
            if not self.enabled:
                return __original(*args, **kwargs)
            self._stack.append(0.0)
            began = time.perf_counter()
            try:
                return __original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - began
                children = self._stack.pop()
                self.exclusive[__bucket] += elapsed - children
                self.inclusive[__bucket] += elapsed
                self.calls[__bucket] += 1
                if self._stack:
                    self._stack[-1] += elapsed

        wrapper.__name__ = getattr(original, "__name__", name)
        setattr(holder, name, wrapper)
        self._patches.append((holder, name, original))

    def restore(self) -> None:
        for holder, name, original in reversed(self._patches):
            setattr(holder, name, original)
        self._patches.clear()

    def reset(self) -> None:
        self.exclusive.clear()
        self.inclusive.clear()
        self.calls.clear()
        self._stack.clear()

    def snapshot(self) -> dict:
        return {"exclusive": dict(self.exclusive), "inclusive": dict(self.inclusive),
                "calls": dict(self.calls)}


def install(timer: ExclusiveTimer) -> None:
    """Wrap every callable the sweep reaches that is worth its own line in the report."""
    # sweep-level structure
    timer.wrap(s6t.Stage6EState, "copy", "state_copy")
    timer.wrap(mfl, "validate_pi_p", "validate_pi_P")
    timer.wrap(mfl, "validate_paths", "validate_paths")
    timer.wrap(mfl.FullLatentFixed, "assert_unchanged", "assert_fixed")
    # the structural attempt
    timer.wrap(mfl, "conditional_structural_mh_step", "structural_COND")
    timer.wrap(mfl, "collapsed_u_mh_step", "structural_MARG")
    timer.wrap(mfl, "propose_row", "propose_row")
    timer.wrap(mfl, "log_structural_prior", "structural_prior")
    timer.wrap(mfl, "precedence_from_u", "precedence_from_u")
    # candidate emission tables
    timer.wrap(rjf.FFBSBlockTables, "refresh", "emission_refresh")
    timer.wrap(sbt.BlockScoreTable, "refresh", "emission_batched_build")
    timer.wrap(fbt.FastBlockScoreTable, "refresh", "emission_fast_build")
    # FFBS
    timer.wrap(mfl, "ffbs_segmentation_draw", "ffbs_draw_other")
    timer.wrap(rjf, "forward", "ffbs_forward")
    timer.wrap(rjf, "backward_sample", "ffbs_backward")
    timer.wrap(rjf, "key_movement", "ffbs_movement")
    timer.wrap(mfl, "segmentation_of", "segmentation_of")
    # the collapsed (marginal) likelihood route
    timer.wrap(cul, "forward", "marg_forward")
    timer.wrap(cul.CollapsedULikelihood, "delta_for_candidate", "marg_delta_other")
    timer.wrap(cul.CollapsedULikelihood, "commit_candidate", "marg_commit")
    # pi/P Gibbs
    timer.wrap(mfl, "gibbs_pi_p", "gibbs_other")
    timer.wrap(mfl, "transition_counts_of", "gibbs_counts")
    timer.wrap(mfl, "initial_counts", "gibbs_counts")
    timer.wrap(mfl, "sample_transition_matrix", "gibbs_sample_P")
    # complete log target
    timer.wrap(mfl, "complete_log_target", "target_other")
    timer.wrap(s6s.SkillBlockLikelihood, "set_blocks", "target_set_blocks")
    timer.wrap(s6s.SkillBlockLikelihood, "full_replay", "target_full_replay")
    timer.wrap(mfl, "_path_prior_components", "target_path_prior")
    for holder in (mfl, rjf, cul):
        timer.wrap(holder, "log_transition_matrix", "log_transition_matrix")


# ------------------------------------------------------------------------ the fixture
def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:                                                # pragma: no cover
        return "unknown"


def jsonable(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if math.isnan(v) else v
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def build(arm: str, checkpoint: Path | None):
    """Model, sampler and a representative state for one arm."""
    corpus = mfl.load_frozen_observed_corpus(CORPUS_DIR)
    fixed = mfl.FullLatentFixed()
    model = mfl.build_full_latent_model(corpus.train, fixed)
    config = mfl.FullLatentConfig(arm=arm, structural_cadence=10, structural_scale=0.5,
                                  table_source="batched")
    sampler = mfl.FullLatentSampler(model=model, fixed=fixed, config=config)
    if checkpoint is not None and Path(checkpoint).exists():
        chain = mfl.FullLatentChain.load(checkpoint, sampler)
        state, origin = chain.state.copy(), f"checkpoint:{Path(checkpoint).name}" \
            f"@sweep{chain.state.iteration}"
        probes = chain.probes
        thin = chain.thin
    else:
        pi, transition = mfl.draw_initial_pi_p(model, PROFILE_SEED)
        u_start = mfl.make_u_start(0, PROFILE_SEED, 1.0, fixed, model)
        state = mfl.initial_full_latent_state(model, u_start, pi, transition)
        origin, thin = "fresh_start", 5
        probes = mfl.select_truth_free_probes(model.traces, corpus.corpus_hash)
    return {"corpus": corpus, "model": model, "sampler": sampler, "state": state,
            "origin": origin, "probes": probes, "thin": thin, "config": config}


def percentiles(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"median": float(np.median(a)), "mean": float(a.mean()),
            "p90": float(np.percentile(a, 90)), "min": float(a.min()),
            "max": float(a.max()), "n": int(a.size)}


# --------------------------------------------------------------------------- the run
def profile_arm(arm: str, sweeps: int, warmup: int, checkpoint: Path | None) -> dict:
    setup = build(arm, checkpoint)
    model, sampler = setup["model"], setup["sampler"]
    state = setup["state"]
    rng = np.random.default_rng(PROFILE_SEED + (0 if arm == mfl.FULL_COND else 1))

    for _ in range(warmup):                       # touch every code path once, warm caches
        state, _ = mfl.full_latent_sweep_once(state, sampler, rng)

    timer = ExclusiveTimer()
    install(timer)
    timer.enabled = True

    per_sweep, per_sweep_cpu, scheduled_flags = [], [], []
    per_sweep_buckets = []
    structural_records = []
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        for _ in range(sweeps):
            timer.reset()
            wall0, cpu0 = time.perf_counter(), time.process_time()
            state, info = mfl.full_latent_sweep_once(state, sampler, rng)
            per_sweep.append(time.perf_counter() - wall0)
            per_sweep_cpu.append(time.process_time() - cpu0)
            scheduled_flags.append(bool(info["scheduled_structural"]))
            per_sweep_buckets.append(dict(timer.exclusive))
            if info["structural_record"] is not None:
                record = info["structural_record"]
                structural_records.append({"accepted": bool(record["accepted"]),
                                           "h_changed": bool(record["h_changed"]),
                                           "invalid": bool(record["invalid"])})
    finally:
        timer.enabled = False
        timer.restore()
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    buckets = sorted({name for row in per_sweep_buckets for name in row})
    total = float(np.sum(per_sweep))
    scheduled = np.array(scheduled_flags)
    breakdown = {}
    for name in buckets:
        series = np.array([row.get(name, 0.0) for row in per_sweep_buckets])
        breakdown[name] = {
            "share_of_total": float(series.sum() / total),
            "per_sweep_mean_ms": float(1000.0 * series.mean()),
            "per_sweep_median_ms": float(1000.0 * np.median(series)),
            "per_sweep_p90_ms": float(1000.0 * np.percentile(series, 90)),
            "on_scheduled_mean_ms": float(1000.0 * series[scheduled].mean())
            if scheduled.any() else 0.0,
            "on_plain_mean_ms": float(1000.0 * series[~scheduled].mean())
            if (~scheduled).any() else 0.0,
            "calls_per_sweep": float(timer.calls.get(name, 0)),   # last sweep only
        }
    accounted = float(sum(np.array([row.get(n, 0.0) for row in per_sweep_buckets]).sum()
                          for n in buckets))

    # the retained-diagnostic step, which fires once every `thin` sweeps outside the kernel
    retain_walls = []
    for _ in range(min(20, sweeps)):
        began = time.perf_counter()
        mfl.invariant_summaries(state, model, setup["probes"])
        mfl._online_path_summaries(state, setup["probes"])
        mfl.relation_indicator_vector(state.u_by_skill)
        retain_walls.append(time.perf_counter() - began)

    return {
        "arm": arm,
        "state_origin": setup["origin"],
        "config": setup["config"].as_dict(),
        "sweeps_measured": int(sweeps),
        "warmup_sweeps": int(warmup),
        "sweep_seconds": percentiles(per_sweep),
        "sweep_cpu_seconds": percentiles(per_sweep_cpu),
        "scheduled_sweep_seconds": percentiles(np.array(per_sweep)[scheduled])
        if scheduled.any() else None,
        "plain_sweep_seconds": percentiles(np.array(per_sweep)[~scheduled])
        if (~scheduled).any() else None,
        "n_scheduled": int(scheduled.sum()),
        "structural_records": {
            "attempts": len(structural_records),
            "accepts": sum(r["accepted"] for r in structural_records),
            "h_accepts": sum(r["accepted"] and r["h_changed"]
                             for r in structural_records),
            "invalid": sum(r["invalid"] for r in structural_records)},
        "breakdown": breakdown,
        "unattributed_share": float((total - accounted) / total),
        "retain_seconds": percentiles(retain_walls),
        "retain_amortised_ms_per_sweep": float(1000.0 * np.median(retain_walls)
                                               / max(1, setup["thin"])),
        "peak_rss_mb": {"before": rss0 / 1e6, "after": rss1 / 1e6},
        "table_stats": {
            "collapsed_forward_evaluations": int(
                sampler.collapsed_likelihood.evaluations),
            "collapsed_cache_hits": int(sampler.collapsed_likelihood.cache_hits),
            "ffbs_table_builds": int(sampler.tables.builds),
            "ffbs_table_build_seconds": float(sampler.tables.build_seconds)},
    }


def function_profile(arm: str, sweeps: int, checkpoint: Path | None,
                     top: int = 30) -> dict:
    """cProfile over unwrapped sweeps: the function-level view of the same work."""
    setup = build(arm, checkpoint)
    sampler, state = setup["sampler"], setup["state"]
    rng = np.random.default_rng(PROFILE_SEED + 17)
    for _ in range(3):
        state, _ = mfl.full_latent_sweep_once(state, sampler, rng)

    holder = {"state": state}

    def run():
        s = holder["state"]
        for _ in range(sweeps):
            s, _ = mfl.full_latent_sweep_once(s, sampler, rng)
        holder["state"] = s

    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("tottime")
    stats.print_stats(top)
    rows = []
    for func, (calls, _, tottime, cumtime, _) in stats.stats.items():
        rows.append({"function": f"{Path(func[0]).name}:{func[1]}({func[2]})",
                     "ncalls": int(calls), "tottime_s": float(tottime),
                     "cumtime_s": float(cumtime),
                     "tottime_per_sweep_ms": float(1000.0 * tottime / sweeps),
                     "calls_per_sweep": float(calls / sweeps)})
    rows.sort(key=lambda r: -r["tottime_s"])
    return {"arm": arm, "sweeps": int(sweeps), "top": rows[:top],
            "text": stream.getvalue()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cprofile-sweeps", type=int, default=20)
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--label", type=str, default="baseline")
    args = parser.parse_args()

    checkpoints = {}
    if args.checkpoint_dir:
        directory = Path(args.checkpoint_dir)
        checkpoints = {mfl.FULL_COND: directory / "full_cond_0.npz",
                       mfl.FULL_MARG: directory / "full_marg_0.npz"}

    report = {"label": args.label, "source_commit": source_commit(),
              "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "corpus_dir": str(CORPUS_DIR),
              "machine": {"platform": platform.platform(),
                          "processor": platform.processor(),
                          "python": sys.version.split()[0],
                          "numpy": np.__version__,
                          "cpu_count": os.cpu_count(),
                          "loadavg": list(os.getloadavg())},
              "arms": {}, "function_profile": {}}

    for arm in (mfl.FULL_COND, mfl.FULL_MARG):
        print(f"[profile] {arm}: {args.sweeps} throwaway sweeps", flush=True)
        report["arms"][arm] = profile_arm(arm, args.sweeps, args.warmup,
                                          checkpoints.get(arm))
        print(f"[profile] {arm}: cProfile over {args.cprofile_sweeps} sweeps", flush=True)
        report["function_profile"][arm] = function_profile(
            arm, args.cprofile_sweeps, checkpoints.get(arm))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"sweep_profile_{args.label}.json"
    path.write_text(json.dumps(jsonable(report), indent=2, sort_keys=True))
    print(f"[profile] wrote {path}")


if __name__ == "__main__":
    main()
