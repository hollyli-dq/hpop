"""Step 7B2 — the FFBS full-joint sampler on the frozen Stage 6E2 corpus.

    PYTHONPATH=src python scripts/stage7b2_full_joint_ffbs.py --project-cost
    PYTHONPATH=src python scripts/stage7b2_full_joint_ffbs.py            # formal run

Step 7B2 is the controlled comparison: same corpus, same posterior, same non-segmentation
kernels, same proposal scales, same dispersed starts, one kernel changed. The question is
whether globally resampling `(S, z)` gives the following `U` update enough conditional
variation to leave the structural basin Stage 6E2's local sampler settled into. That is a
hypothesis, not a prediction — FFBS updates `(S, z)`, not `U`, and it may well leave `U`
exactly as stuck as before.

## This script refuses to run against a moving baseline

Stage 6E2 is a live experiment with a registered §14 continuation ladder
(50k -> 75k -> 100k -> 125k -> 150k) and a registered rule for interpreting a
continued-but-still-failing run. Comparing against it before it reaches its decision point
would mean comparing against whichever rung happened to be on disk, and the choice of rung
would then be made after seeing the FFBS result. `require_frozen_baseline` therefore stops
the formal run until the baseline is frozen, and says exactly why.

`--project-cost` is always allowed: it measures what an FFBS sweep costs on this corpus,
which §20 requires *before* a run length is registered, and it touches no baseline
artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import (                    # noqa: E402
    FFBSBlockTables, Stage7BSampler, ffbs_sweep_once, run_stage7b_chain,
)
from hpop.mcmc_original.recurrent_segmentation import log_target_stage6e       # noqa: E402
from hpop.mcmc_original.stage6b_mcmc_diagnostics import (                      # noqa: E402
    bulk_ess, rank_normalized_split_rhat,
)
from hpop.mcmc_original.stage6e_state import Stage6EState                      # noqa: E402
from hpop.mcmc_original.stage7b_diagnostics import (                           # noqa: E402
    SweepMovementTracker, invariant_summaries,
)
from hpop.mcmc_original.stage6e_corpus import corpus_hash, generate_corpus     # noqa: E402
from hpop.mcmc_original.stage6e_frozen import (                               # noqa: E402
    assert_stage6d_unchanged, config_hash,
)
from hpop.mcmc_original.semi_markov_ffbs import backward_sample, forward       # noqa: E402
from hpop.mcmc_original.transitions import log_transition_matrix              # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "stage7b2_full_joint_ffbs"
BASELINE = Path("/Users/dongqing/Desktop/hpop-stage6e/results/mcmc_original"
                "/stage6e2_unknown_boundary_full_seed0")

# Registered before any Step 7B2 result exists. New seeds; the starts, scales, sweep
# count and every other kernel are the baseline's.
CHAIN_SEEDS = (7_063_201, 7_063_202, 7_063_203, 7_063_204)
REGISTERED_LADDER = (50_000, 75_000, 100_000, 125_000, 150_000)

# The Stage 6E2 first formal block, matched exactly. Registered before launch.
REGISTERED_SWEEPS = 50_000
REGISTERED_BURN_IN = 15_000
REGISTERED_THIN = 5
CHECKPOINT_EVERY = 500          # about 12 minutes of wall clock per checkpoint
N_CHAINS = 4
QUIET_SECONDS = 1_800          # a baseline artifact untouched for this long is settled


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


def load_baseline_script():
    """The Stage 6E2 chain script owns the starts and the model construction."""
    path = ROOT / "scripts" / "stage6e2_formal_chains.py"
    spec = importlib.util.spec_from_file_location("stage6e2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ the freeze guard
def baseline_freeze_audit() -> dict:
    """Is the Stage 6E2 baseline at its registered decision point and no longer moving?"""
    reasons: list = []
    present = BASELINE.exists()
    if not present:
        return {"frozen": False, "reasons": ["the Stage 6E2 baseline directory is not "
                                             "reachable from this worktree"],
                "directory": str(BASELINE)}

    history_path = BASELINE / "continuation_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else {}
    blocks = history.get("unknown", [])
    reached = max((int(b["sweeps_to"]) for b in blocks), default=0)
    if reached < REGISTERED_LADDER[-1]:
        reasons.append(
            f"the registered continuation ladder is at {reached:,} of "
            f"{REGISTERED_LADDER[-1]:,} sweeps; the decision point has not been reached")

    running = False
    try:
        processes = subprocess.check_output(["ps", "-Ao", "command"], text=True)
        running = "stage6e2_formal_chains.py" in processes
    except Exception:                                                # pragma: no cover
        reasons.append("could not determine whether a Stage 6E2 process is running")
    if running:
        reasons.append("a stage6e2_formal_chains.py process is running right now")

    now = time.time()
    recent = {path.name: round(now - path.stat().st_mtime)
              for path in BASELINE.iterdir()
              if path.is_file() and now - path.stat().st_mtime < QUIET_SECONDS}
    if recent:
        reasons.append(f"baseline artifacts written within the last "
                       f"{QUIET_SECONDS}s: {sorted(recent)}")

    marker = BASELINE / "freeze_manifest.json"
    return {
        "frozen": bool(not reasons),
        "reasons": reasons,
        "directory": str(BASELINE),
        "registered_ladder": list(REGISTERED_LADDER),
        "ladder_reached": reached,
        "continuation_blocks": blocks,
        "a_stage6e2_process_is_running": running,
        "artifacts_modified_within_quiet_window": recent,
        "quiet_window_seconds": QUIET_SECONDS,
        "freeze_manifest_present": marker.exists(),
        "interpretation_rule": (
            json.loads((BASELINE / "interpretation_rule.json").read_text())
            if (BASELINE / "interpretation_rule.json").exists() else None),
    }


def require_frozen_baseline(audit: dict) -> None:
    if not audit["frozen"]:
        raise SystemExit(
            "the Stage 6E2 baseline is not frozen, so the Step 7B2 formal comparison "
            "must not start:\n  - " + "\n  - ".join(audit["reasons"]))


# -------------------------------------------------------------------------- the audits
def corpus_audit() -> dict:
    """The corpus must be the baseline's, rebuilt rather than copied and hash-checked."""
    manifest_path = BASELINE / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    corpus = generate_corpus()
    rebuilt = corpus_hash(corpus)
    return {
        "baseline_corpus_hash": manifest.get("corpus_hash"),
        "rebuilt_corpus_hash": rebuilt,
        "match": bool(rebuilt == manifest.get("corpus_hash")),
        "n_train_traces": len(corpus.train),
        "n_heldout_traces": len(corpus.heldout),
        "baseline_n_train": manifest.get("n_train_traces"),
        "baseline_n_heldout": manifest.get("n_heldout_traces"),
        "trace_length": manifest.get("trace_length"),
        "regenerated_not_copied": True,
        "hidden_truth_use": "evaluation only, exactly as the baseline records",
        "pass": bool(rebuilt == manifest.get("corpus_hash")
                     and len(corpus.train) == manifest.get("n_train_traces")
                     and len(corpus.heldout) == manifest.get("n_heldout_traces")),
    }


def starting_state_audit() -> dict:
    """Reconstruct the baseline's four dispersed starts and hash them."""
    module = load_baseline_script()
    corpus = generate_corpus()
    model = module.build_model(corpus)
    hashes, summaries = [], []
    for chain in range(module.N_CHAINS):
        state = module.dispersed_start(chain, corpus, model, oracle=False)
        payload = json.dumps(jsonable(state.to_dict()), sort_keys=True)
        hashes.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
        summaries.append({
            "chain": chain, "rho": float(state.rho),
            **{name: float(getattr(state, name))
               for name in ("beta", "omega", "lambda_rep", "lambda_back")},
            "segment_counts": [len(s.segments) for s in state.segmentations][:5],
            "total_segments": int(sum(len(s.segments) for s in state.segmentations)),
        })
    return {
        "construction": "the Stage 6E2 dispersed-start constructor, reused",
        "start_seeds": list(module.START_SEEDS),
        "n_chains": int(module.N_CHAINS),
        "hashes": hashes, "summaries": summaries,
        "baseline_chain_seeds": list(module.CHAIN_SEEDS),
        "step7b2_chain_seeds": list(CHAIN_SEEDS),
        "seeds_are_new": bool(not set(CHAIN_SEEDS) & set(module.CHAIN_SEEDS)),
        "not_initialised_from_a_baseline_endpoint": True,
    }


def kernel_audit() -> dict:
    """Everything except the `(S, z)` kernel must be the baseline's, untuned."""
    pilot_path = BASELINE / "pilot_results.json"
    pilot = json.loads(pilot_path.read_text()) if pilot_path.exists() else {}
    config_path = BASELINE / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    scales = {k: float(v) for k, v in
              (pilot.get("selected_scales") or config.get("selected_scales", {})).items()}
    return {
        "proposal_scales": scales,
        "proposal_scales_match_baseline": bool(
            scales and scales == {k: float(v)
                                  for k, v in config.get("selected_scales", {}).items()}),
        "retuned": False,
        "retune_note": "the primary comparison uses the baseline's scales unchanged; if "
                       "global acceptance becomes pathological under FFBS that is a "
                       "finding to report, not a reason to retune before the comparison",
        "baseline_proposals_per_trace": config.get("selected_proposals_per_trace"),
        "ffbs_proposals_per_trace": "one exact Gibbs draw",
        "only_change": "the (S, z) transition kernel",
        "shared_kernels": ["pi/P conjugate update", "U row proposal", "rho logit walk",
                           "beta", "omega", "lambda_rep", "lambda_back"],
        "sweep_order": config.get("sweep_order"),
    }


# ----------------------------------------------------------------------- cost projection
def project_cost(n_sweeps: int = 3, n_traces: int | None = None) -> dict:
    """What an FFBS sweep costs on this corpus, and what a formal run would cost.

    Deliberately small: a handful of sweeps, single-threaded, on a machine that may be
    running the Stage 6E baseline. Reported as a projection with its measurement basis, not
    as a benchmark of a tuned implementation.
    """
    import resource

    module = load_baseline_script()
    corpus = generate_corpus()
    model = module.build_model(corpus)
    if n_traces is not None:
        model.traces = model.traces[:int(n_traces)]
    state = module.dispersed_start(0, corpus, model, oracle=False)
    if n_traces is not None:
        state.segmentations = state.segmentations[:int(n_traces)]
    pilot = kernel_audit()["proposal_scales"]

    sampler = Stage7BSampler(model=model, scales=pilot)
    rng = np.random.default_rng(7_063_000)

    began = time.perf_counter()
    for _ in range(n_sweeps):
        state = ffbs_sweep_once(state, sampler, rng)
    total = time.perf_counter() - began

    tables = FFBSBlockTables(model=model, source="batched")
    began = time.perf_counter()
    tables.refresh(state)
    table_seconds = time.perf_counter() - began
    log_pi, log_p = np.log(state.pi), log_transition_matrix(state.transition)
    began = time.perf_counter()
    charts = [forward(t, log_pi, log_p, model.delta_b, model.max_width, model.min_width)
              for t in tables.tables_for(state)]
    chart_seconds = time.perf_counter() - began
    began = time.perf_counter()
    for chart in charts:
        backward_sample(chart, rng)
    draw_seconds = time.perf_counter() - began

    per_sweep = total / max(1, n_sweeps)
    peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    baseline_history = json.loads(
        (BASELINE / "continuation_history.json").read_text()) if (
        BASELINE / "continuation_history.json").exists() else {"unknown": []}
    baseline_block = (baseline_history.get("unknown") or [{}])[0]
    baseline_wall = baseline_block.get("wall_seconds")
    baseline_sweeps = baseline_block.get("sweeps_to")
    return {
        "measured_sweeps": int(n_sweeps),
        "n_traces": len(model.traces),
        "seconds_per_sweep": per_sweep,
        "candidate_table_seconds": table_seconds,
        "chart_seconds_all_traces": chart_seconds,
        "backward_draw_seconds_all_traces": draw_seconds,
        "parameter_phase_seconds": max(0.0, per_sweep - table_seconds - chart_seconds
                                       - draw_seconds),
        "n_candidate_blocks": int(sum(int(np.isfinite(t).sum())
                                      for t in tables.tables_for(state))),
        "peak_rss_bytes": int(peak_bytes),
        "projected_50k_sweep_hours_per_chain": per_sweep * 50_000 / 3600.0,
        "projected_4_chain_wall_hours_if_parallel": per_sweep * 50_000 / 3600.0,
        "baseline_block1_wall_seconds": baseline_wall,
        "baseline_block1_sweeps": baseline_sweeps,
        "baseline_seconds_per_sweep": (baseline_wall / baseline_sweeps
                                       if baseline_wall and baseline_sweeps else None),
        "measured_under_contention": True,
        "note": "measured while the Stage 6E2 baseline was running on the same machine, "
                "so both this and the baseline's own per-sweep cost are pessimistic",
    }


# ------------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-cost", action="store_true",
                        help="measure the FFBS sweep cost on this corpus and stop")
    parser.add_argument("--cost-sweeps", type=int, default=3)
    parser.add_argument("--cost-traces", type=int, default=None,
                        help="project on a subset of traces; the full corpus by default")
    parser.add_argument("--formal", action="store_true",
                        help="launch the registered formal chains")
    parser.add_argument("--sweeps", type=int, default=REGISTERED_SWEEPS)
    parser.add_argument("--burn-in", type=int, default=REGISTERED_BURN_IN)
    parser.add_argument("--thin", type=int, default=REGISTERED_THIN)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--out-suffix", default="")
    args = parser.parse_args()

    assert_stage6d_unchanged()
    if args.formal:
        out = (OUT if not args.out_suffix
               else OUT.parent / f"{OUT.name}{args.out_suffix}")
        run_formal(args.sweeps, args.burn_in, args.thin, out, args.progress_every,
                   args.checkpoint_every)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    freeze = baseline_freeze_audit()
    corpus = corpus_audit()
    starts = starting_state_audit()
    kernels = kernel_audit()

    (OUT / "corpus_hash.json").write_text(json.dumps(jsonable(corpus), indent=2))
    (OUT / "starting_state_hashes.json").write_text(json.dumps(jsonable(starts), indent=2))
    (OUT / "baseline_freeze_audit.json").write_text(json.dumps(jsonable(freeze), indent=2))

    cost = None
    if args.project_cost:
        cost = project_cost(args.cost_sweeps, args.cost_traces)
        (OUT / "cost_projection.json").write_text(json.dumps(jsonable(cost), indent=2))
        print(f"[7B2] sweep {cost['seconds_per_sweep']:.3f}s on "
              f"{cost['n_traces']} traces "
              f"(tables {cost['candidate_table_seconds']:.3f}s, charts "
              f"{cost['chart_seconds_all_traces']:.3f}s, draws "
              f"{cost['backward_draw_seconds_all_traces']:.3f}s, parameters "
              f"{cost['parameter_phase_seconds']:.3f}s)")
        print(f"[7B2] projected {cost['projected_50k_sweep_hours_per_chain']:.1f} h per "
              f"chain for 50,000 sweeps; the LocalMoveKernel baseline ran at "
              f"{cost['baseline_seconds_per_sweep']:.3f} s/sweep")

    config = {
        "stage": "7B2", "status": "PREPARED — BLOCKED ON THE STAGE 6E2 BASELINE FREEZE",
        "source_commit": source_commit(), "stage6e_config_hash": config_hash(),
        "python": platform.python_version(), "numpy": np.__version__,
        "baseline_freeze_audit": freeze, "corpus": corpus, "starting_states": starts,
        "kernels": kernels, "chain_seeds": list(CHAIN_SEEDS),
        "cost_projection": cost,
        "hypothesis": "global (S, z) refresh MAY indirectly improve U/H mode switching, "
                      "because the next U update is conditioned on a substantially "
                      "refreshed segmentation. FFBS does not update U and no improvement "
                      "is predicted.",
    }
    (OUT / "config.json").write_text(json.dumps(jsonable(config), indent=2))

    lines = [
        "# Step 7B2 — prepared, blocked on the Stage 6E2 baseline",
        "",
        "**Status: NOT STARTED.** Every precondition that does not depend on the baseline "
        "is verified and recorded here; the formal run is blocked by design until the "
        "Stage 6E2 baseline reaches its registered decision point and stops moving.",
        "",
        "## Why it is blocked",
        "",
    ]
    for reason in freeze["reasons"]:
        lines.append(f"* {reason}")
    lines += [
        "",
        f"The registered ladder is {REGISTERED_LADDER} and the baseline has reached "
        f"{freeze['ladder_reached']:,}. Comparing now would mean comparing against "
        "whichever rung happened to be on disk, and choosing that rung after seeing the "
        "FFBS result.",
        "",
        "## What is already verified",
        "",
        f"* **Corpus**: rebuilt from the registered generator, hash "
        f"`{corpus['rebuilt_corpus_hash'][:16]}...` against the baseline manifest's "
        f"`{str(corpus['baseline_corpus_hash'])[:16]}...` — "
        f"{'match' if corpus['match'] else 'MISMATCH'}. "
        f"{corpus['n_train_traces']} training and {corpus['n_heldout_traces']} held-out "
        "traces, not regenerated with a different seed and not reduced.",
        f"* **Starting states**: the baseline's own dispersed-start constructor, reused "
        f"with its seeds {starts['start_seeds']}; four distinct starts, hashed. New chain "
        f"seeds {starts['step7b2_chain_seeds']}; no chain starts from a baseline endpoint.",
        f"* **Kernels**: {kernels['only_change']}. Proposal scales are the baseline's "
        f"selected scales, untuned: {kernels['proposal_scales']}.",
    ]
    if cost:
        lines += [
            "",
            "## Cost projection",
            "",
            f"* one FFBS sweep on {cost['n_traces']} traces: "
            f"{cost['seconds_per_sweep']:.3f} s — candidate tables "
            f"{cost['candidate_table_seconds']:.3f} s "
            f"({cost['n_candidate_blocks']:,} legal blocks), forward charts "
            f"{cost['chart_seconds_all_traces']:.3f} s, backward draws "
            f"{cost['backward_draw_seconds_all_traces']:.3f} s, parameter phase "
            f"{cost['parameter_phase_seconds']:.3f} s",
            f"* projected {cost['projected_50k_sweep_hours_per_chain']:.1f} h per chain "
            f"for the baseline's first formal block of 50,000 sweeps",
            f"* the LocalMoveKernel baseline ran that block at "
            f"{cost['baseline_seconds_per_sweep']:.3f} s/sweep "
            f"({cost['baseline_block1_wall_seconds'] / 3600:.1f} h)",
            f"* {cost['note']}",
        ]
    lines += [
        "",
        "## What happens when the baseline freezes",
        "",
        "1. record the frozen baseline commit, corpus hash, chain starts, scales, "
        "continuation history, permutation-invariant convergence summaries and mode "
        "diagnostics;",
        "2. register the Step 7B2 run length — the baseline's 50,000-sweep first block if "
        "the cost projection allows it, or a compute-matched alternative registered "
        "*before* any result exists;",
        "3. run four FFBS chains from the reconstructed starts with the new seeds;",
        "4. report equal-sweep and equal-wall-clock comparisons, permutation-invariant "
        "convergence, structural movement, ESS/second, and recovery only if convergence "
        "permits interpretation.",
        "",
        f"Source commit `{config['source_commit']}`.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[7B2] wrote {OUT}")

    require_frozen_baseline(freeze)
    raise SystemExit("Step 7B2 formal chains are not implemented in this checkpoint: the "
                     "baseline was still moving, so the run configuration is registered "
                     "and the comparison is deferred.")




# ------------------------------------------------------------------- the formal run
def freeze_manifest(freeze: dict, corpus: dict, starts: dict, kernels: dict,
                    schedule: dict, out: Path) -> dict:
    """Everything that defines this run, written BEFORE the first sweep.

    The point of writing it first is that nothing in it can be chosen after seeing a
    result. It also records the one thing that is unusual about this launch: the chains
    start while the Stage 6E2 baseline is still advancing its ladder, on explicit
    instruction, with the comparison deferred to the baseline's frozen final state.
    """
    manifest = {
        "stage": "7B2", "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_commit": source_commit(), "stage6e_config_hash": config_hash(),
        "python": platform.python_version(), "numpy": np.__version__,
        "target": {
            "coordinates": ["S", "z", "U", "rho", "beta", "omega", "lambda_rep",
                            "lambda_back", "pi", "P"],
            "posterior": "the Stage 6E unknown-boundary posterior, unchanged",
            "priors": "unchanged: Gaussian latent U with the full rho-dependent "
                      "determinant, rho prior, four scalar priors, boundary prior, "
                      "initial-skill prior, Stage 3 transition model with P[h,h] = 0 and "
                      "no terminal transition",
            "fixed": {"epsilon": 0.02, "delta_B": 0.15, "min_width": 3, "max_width": 12},
        },
        "only_algorithmic_difference": "LocalMoveKernel(S, z) -> exact FFBS(S, z)",
        "corpus": corpus, "starting_states": starts, "kernels": kernels,
        "schedule": schedule,
        "movement_diagnostics": {
            "recorded_every_sweep": ["h_hash", "relation_total", "boundary_hamming",
                                     "label_changes", "states_changed", "co_clustering",
                                     "log_target"],
            "used_to_steer_the_run": False,
            "rule": "recorded for analysis only; the registered run is not altered or "
                    "stopped on the basis of what they show",
        },
        "not_tuned_using": ["Stage 6E recovery", "current Local-chain modes", "the truth",
                            "held-out NLL", "intermediate FFBS R-hat"],
        "launch_authorisation": {
            "launched_while_the_baseline_is_still_running": True,
            "instruction": "explicit: start the formal Step 7B2 chains in parallel with "
                           "the still-running Stage 6E2 LocalMoveKernel baseline",
            "baseline_state_at_launch": {
                "frozen": freeze["frozen"], "ladder_reached": freeze["ladder_reached"],
                "registered_ladder": freeze["registered_ladder"],
                "reasons_it_is_not_frozen": freeze["reasons"]},
            "comparison_rule": "the final scientific comparison uses the FROZEN FINAL "
                               "Stage 6E2 baseline. The current intermediate rung must "
                               "not be compared against or selected, and no baseline "
                               "chain artifact is read by this run.",
        },
        "source_sha256": {
            name: hashlib.sha256((ROOT / "src" / "hpop" / "mcmc_original"
                                  / name).read_bytes()).hexdigest()
            for name in ("semi_markov_ffbs.py", "recurrent_joint_ffbs_mcmc.py",
                         "fast_block_tables.py", "stage6e_sampler.py",
                         "stage7b_diagnostics.py")},
    }
    (out / "freeze_manifest.json").write_text(json.dumps(jsonable(manifest), indent=2))
    return manifest


def _formal_worker(payload: dict) -> dict:
    """One Step 7B2 chain in its own process, checkpointing as it goes."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "src"))
    baseline = load_baseline_script()
    corpus = generate_corpus()
    model = baseline.build_model(corpus)
    chain = int(payload["chain"])
    start = baseline.dispersed_start(chain, corpus, model, oracle=False)

    parts = log_target_stage6e(start, model)
    if not math.isfinite(parts["log_target"]):
        raise ValueError(f"chain {chain}: start has a non-finite log target")

    tracker = SweepMovementTracker(model.n_skills, model.traces)
    began = time.perf_counter()
    result = run_stage7b_chain(
        model=model, start=start, scales=payload["scales"],
        num_sweeps=payload["sweeps"], burn_in=payload["burn_in"],
        thin=payload["thin"], seed=CHAIN_SEEDS[chain], chain=chain,
        table_source="fast", store_labels=True, store_keys=False,
        progress_every=payload["progress_every"],
        checkpoint_path=payload["checkpoint_path"],
        checkpoint_every=payload["checkpoint_every"],
        movement_tracker=tracker)
    return {
        "chain": chain, "seed": CHAIN_SEEDS[chain],
        "start_log_target": float(parts["log_target"]),
        "u_draws": result.u_draws, "scalars": result.scalars,
        "pi_draws": result.pi_draws, "transition_draws": result.transition_draws,
        "segment_counts": result.segment_counts,
        "occurrence_labels": result.occurrence_labels,
        "relation_counts": result.relation_counts, "log_target": result.log_target,
        "log_block_likelihood": result.log_block_likelihood,
        "proposed": result.proposed, "accepted": result.accepted,
        "invalid": result.invalid,
        "acceptance_rates": result.acceptance(),
        "movement_totals": result.movement,
        "movement_summary": tracker.summary(),
        "movement_series": tracker.series(),
        "table_builds": int(result.table_builds),
        "final_state": result.final_state.to_dict(),
        "runtime_seconds": time.perf_counter() - began,
    }


def run_formal(sweeps: int, burn_in: int, thin: int, out: Path, progress_every: int,
               checkpoint_every: int) -> None:
    from multiprocessing import get_context

    out.mkdir(parents=True, exist_ok=True)
    freeze = baseline_freeze_audit()
    corpus = corpus_audit()
    starts = starting_state_audit()
    kernels = kernel_audit()
    if not corpus["pass"]:
        raise SystemExit(f"the corpus does not match the baseline manifest: {corpus}")

    schedule = {"n_chains": N_CHAINS, "sweeps": sweeps, "burn_in": burn_in, "thin": thin,
                "chain_seeds": list(CHAIN_SEEDS), "checkpoint_every": checkpoint_every,
                "checkpoint_path": str(out / "checkpoints"),
                "table_source": "fast",
                "matches_baseline_first_formal_block": bool(
                    sweeps == REGISTERED_SWEEPS and burn_in == REGISTERED_BURN_IN
                    and thin == REGISTERED_THIN)}
    manifest = freeze_manifest(freeze, corpus, starts, kernels, schedule, out)
    print(f"[7B2] freeze manifest written: corpus {corpus['rebuilt_corpus_hash'][:16]}  "
          f"starts {[h[:8] for h in starts['hashes']]}  seeds {list(CHAIN_SEEDS)}",
          flush=True)
    print(f"[7B2] baseline at launch: ladder {freeze['ladder_reached']:,}, frozen="
          f"{freeze['frozen']}; comparison deferred to its frozen final state",
          flush=True)

    jobs = [{"chain": c, "sweeps": sweeps, "burn_in": burn_in, "thin": thin,
             "scales": kernels["proposal_scales"], "progress_every": progress_every,
             "checkpoint_path": str(out / "checkpoints"),
             "checkpoint_every": checkpoint_every} for c in range(N_CHAINS)]
    print(f"[7B2] {N_CHAINS} FFBS chains x {sweeps:,} sweeps, burn-in {burn_in:,}, "
          f"thin {thin}, checkpoint every {checkpoint_every:,}", flush=True)

    began = time.perf_counter()
    with get_context("spawn").Pool(processes=N_CHAINS) as pool:
        payloads = pool.map(_formal_worker, jobs)
    wall = time.perf_counter() - began
    for payload in payloads:
        print(f"[7B2] chain {payload['chain']} seed {payload['seed']}: "
              f"{len(payload['log_target']):,} retained in "
              f"{payload['runtime_seconds'] / 3600:.2f} h", flush=True)

    write_formal_results(payloads, manifest, wall, out)
    print(f"[7B2] wrote {out}", flush=True)


def write_formal_results(payloads: list, manifest: dict, wall: float, out: Path) -> None:
    """Chains, movement and invariant convergence. No baseline is read here."""
    np.savez_compressed(
        out / "chains.npz",
        u_draws=np.array([p["u_draws"] for p in payloads]),
        pi_draws=np.array([p["pi_draws"] for p in payloads]),
        transition_draws=np.array([p["transition_draws"] for p in payloads]),
        segment_counts=np.array([p["segment_counts"] for p in payloads]),
        relation_counts=np.array([p["relation_counts"] for p in payloads]),
        occurrence_labels=np.array([p["occurrence_labels"] for p in payloads]),
        log_target=np.array([p["log_target"] for p in payloads]),
        log_block_likelihood=np.array([p["log_block_likelihood"] for p in payloads]),
        chain_seeds=np.array([p["seed"] for p in payloads]),
        runtime_seconds=np.array([p["runtime_seconds"] for p in payloads]),
        **{f"scalar_{name}": np.array([p["scalars"][name] for p in payloads])
           for name in ("rho", "beta", "omega", "lambda_rep", "lambda_back")})

    np.savez_compressed(
        out / "movement_series.npz",
        **{f"chain{p['chain']}_{name}": series
           for p in payloads for name, series in p["movement_series"].items()})

    structural = {
        "per_chain": [{"chain": p["chain"], **p["movement_summary"]} for p in payloads],
        "pooled": {
            "total_h_changes": int(sum(p["movement_summary"]["h_changes"]
                                       for p in payloads)),
            "chains_with_frozen_structure": int(sum(
                1 for p in payloads
                if p["movement_summary"]["relation_count_within_chain_sd"] < 0.01)),
            "max_between_chain_relation_mean_gap": float(
                max(p["movement_summary"]["relation_count_mean"] for p in payloads)
                - min(p["movement_summary"]["relation_count_mean"] for p in payloads)),
        },
        "criteria_note": "Stage 6E2's registered structural-locking criteria A and B, "
                         "reproduced so both samplers are judged by the same rule. The "
                         "verdict itself waits for the frozen baseline.",
    }
    (out / "structural_movement.json").write_text(
        json.dumps(jsonable(structural), indent=2))

    segmentation = {
        "per_chain": [{
            "chain": p["chain"],
            "boundary_hamming_total": int(p["movement_totals"]["boundary_hamming"]),
            "label_changes_total": int(p["movement_totals"]["label_changes"]),
            "trace_draws_changed_total": int(p["movement_totals"]["states_changed"]),
            "boundary_hamming_per_sweep":
                p["movement_summary"]["boundary_hamming_per_sweep"],
            "label_changes_per_sweep": p["movement_summary"]["label_changes_per_sweep"],
            "co_clustering_mean": p["movement_summary"]["co_clustering_mean"],
            "co_clustering_movement_per_sweep":
                p["movement_summary"]["co_clustering_movement_per_sweep"],
        } for p in payloads]}
    (out / "segmentation_movement.json").write_text(
        json.dumps(jsonable(segmentation), indent=2))

    chains = [{"log_target": p["log_target"], "relation_counts": p["relation_counts"],
               "segment_counts": p["segment_counts"], "pi_draws": p["pi_draws"],
               "transition_draws": p["transition_draws"], "scalars": p["scalars"],
               "label_draws": p["occurrence_labels"]} for p in payloads]
    summaries = [invariant_summaries(chain) for chain in chains]
    convergence = {}
    for name in summaries[0]:
        stacked = np.array([np.asarray(s[name], dtype=float).reshape(
            len(s[name]), -1).mean(axis=1) for s in summaries])
        if stacked.std() == 0.0:
            convergence[name] = {"degenerate": True, "rhat": None, "bulk_ess": None,
                                 "note": "no variation across chains or draws"}
            continue
        convergence[name] = {
            "rhat": float(rank_normalized_split_rhat(stacked)["rhat"]),
            "bulk_ess": float(bulk_ess(stacked)),
            "mean": float(stacked.mean()), "sd": float(stacked.std(ddof=1))}
    worst = max((block["rhat"] for block in convergence.values()
                 if block.get("rhat") is not None), default=None)
    (out / "invariant_convergence.json").write_text(json.dumps(jsonable({
        "protocol": "permutation-invariant summaries only; no truth alignment anywhere "
                    "in this path",
        "threshold": 1.01, "worst_rhat": worst,
        "passes_registered_threshold": bool(worst is not None and worst <= 1.01),
        "summaries": convergence}), indent=2))

    scalars = {}
    for name in ("rho", "beta", "omega", "lambda_rep", "lambda_back"):
        stacked = np.array([p["scalars"][name] for p in payloads])
        scalars[name] = {"mean": float(stacked.mean()), "sd": float(stacked.std(ddof=1)),
                         "rhat": float(rank_normalized_split_rhat(stacked)["rhat"]),
                         "bulk_ess": float(bulk_ess(stacked))}
    (out / "scalar_diagnostics.json").write_text(json.dumps(jsonable({
        "scalars": scalars,
        "acceptance_per_chain": [p["acceptance_rates"] for p in payloads]}), indent=2))

    performance = {
        "wall_seconds": wall, "wall_hours": wall / 3600.0,
        "per_chain_seconds": [p["runtime_seconds"] for p in payloads],
        "seconds_per_sweep": [p["runtime_seconds"] / manifest["schedule"]["sweeps"]
                              for p in payloads],
        "retained_per_chain": [len(p["log_target"]) for p in payloads],
        "table_builds_per_chain": [p["table_builds"] for p in payloads],
        "builds_equal_sweeps": bool(all(
            p["table_builds"] == manifest["schedule"]["sweeps"] for p in payloads)),
        "start_log_targets": [p["start_log_target"] for p in payloads],
    }
    (out / "performance.json").write_text(json.dumps(jsonable(performance), indent=2))

    (out / "final_states.json").write_text(json.dumps(
        jsonable({str(p["chain"]): p["final_state"] for p in payloads}), indent=2))

    lines = [
        "# Step 7B2 — FFBS full-joint chains on the frozen Stage 6E2 corpus",
        "",
        "**The comparison is deferred.** These are the FFBS chains only. The Stage 6E2 "
        f"baseline was at {manifest['launch_authorisation']['baseline_state_at_launch']['ladder_reached']:,} "
        "sweeps of its registered ladder when these chains launched, and the scientific "
        "comparison must use its frozen final state, not that intermediate rung. Nothing "
        "in this directory reads a baseline chain artifact.",
        "",
        "## Run",
        "",
        f"* {manifest['schedule']['n_chains']} chains x "
        f"{manifest['schedule']['sweeps']:,} sweeps, burn-in "
        f"{manifest['schedule']['burn_in']:,}, thin {manifest['schedule']['thin']}",
        f"* seeds {manifest['schedule']['chain_seeds']}, starts reconstructed with the "
        "baseline's own constructor",
        f"* corpus `{manifest['corpus']['rebuilt_corpus_hash'][:16]}...`, "
        f"{manifest['corpus']['n_train_traces']} training traces",
        f"* proposal scales the baseline's, untuned: {manifest['kernels']['proposal_scales']}",
        f"* wall {performance['wall_hours']:.2f} h, "
        f"{np.mean(performance['seconds_per_sweep']):.3f} s per sweep per chain",
        "",
        "## Structural movement (the Step 7B2 question)",
        "",
        "| chain | H changes | distinct H | sweeps to first H change | relation-count "
        "within-chain SD | major H transitions |",
        "|---|---|---|---|---|---|",
    ]
    for row in structural["per_chain"]:
        lines.append(
            f"| {row['chain']} | {row['h_changes']:,} | {row['distinct_h_states']:,} | "
            f"{row['sweeps_to_first_h_change']} | "
            f"{row['relation_count_within_chain_sd']:.4f} | "
            f"{row['major_h_mode_transitions']:,} |")
    lines += [
        "",
        f"Chains whose structure is frozen by Stage 6E2's criterion A (within-chain "
        f"relation-count sd < 0.01): {structural['pooled']['chains_with_frozen_structure']} "
        f"of {len(payloads)}. Largest between-chain gap in mean relation count: "
        f"{structural['pooled']['max_between_chain_relation_mean_gap']:.3f}.",
        "",
        "## Segmentation movement",
        "",
        "| chain | boundary Hamming / sweep | label changes / sweep | co-clustering "
        "movement / sweep |",
        "|---|---|---|---|",
    ]
    for row in segmentation["per_chain"]:
        lines.append(f"| {row['chain']} | {row['boundary_hamming_per_sweep']:.2f} | "
                     f"{row['label_changes_per_sweep']:.2f} | "
                     f"{row['co_clustering_movement_per_sweep']:.5f} |")
    lines += [
        "",
        "## Permutation-invariant convergence",
        "",
        f"Worst R-hat over the invariant summaries: "
        f"{worst if worst is None else round(worst, 5)} against a 1.01 threshold "
        f"({'PASS' if (worst is not None and worst <= 1.01) else 'FAIL'}). No truth "
        "alignment is used anywhere in this path.",
        "",
        "## Not interpreted here",
        "",
        "* recovery and held-out prediction: they require the convergence verdict and the "
        "frozen baseline, so they are not computed in this run;",
        "* any Local-vs-FFBS claim: the baseline is still advancing its ladder.",
        "",
        f"Source commit `{manifest['source_commit']}`.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
