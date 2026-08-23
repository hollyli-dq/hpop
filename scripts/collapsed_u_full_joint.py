"""Step 8 formal run — the partially-collapsed sampler on the frozen Stage 6E2 corpus.

    PYTHONPATH=src python scripts/collapsed_u_full_joint.py --formal

The registered question, stated before the first sweep: **does the occasional collapsed-U
structural move break the (S, z)-U structural locking** that froze all four Stage 6E2
LocalMoveKernel chains (150k, FAIL/MULTIMODAL) and all four Step 7B2 FFBS chains (50k,
hypothesis falsified) on this exact corpus?

Protocol: Step 7B2's, with ONE further kernel added and nothing else changed —

    same frozen corpus (hash-audited against the baseline manifest)
    same four dispersed starts (hash-audited)
    same proposal scales, UNTUNED (the baseline's selected_scales)
    same 50,000 sweeps / burn-in 15,000 / thin 5 / checkpoints every 500
    Step 7B sweep (exact FFBS (S,z) draw + Stage 6E phases), unchanged
    + one collapsed-U MH move every 10 sweeps, immediately followed by that
      same FFBS refresh (the validated partially-collapsed composition)

New seeds. The audits, the freeze manifest and the results writer are Step 7B2's own
functions, imported and called — not reimplemented. Comparisons against the two frozen
baselines happen in a separate analysis step after the run; nothing here reads them
beyond the freeze audit.

Kernel provenance: exact-correctness evidence and the 600k mixed-reference gates are in
results/mcmc_original/collapsed_u_kernel_validation{,_rep2}. Cadence 10 is the C1
audit's cost-motivated provisional default, registered here as-is, untuned.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.collapsed_u_kernel import (                        # noqa: E402
    MOVE_NAME, CollapsedUConfig, run_collapsed_u_chain,
)
from hpop.mcmc_original.recurrent_segmentation import log_target_stage6e   # noqa: E402
from hpop.mcmc_original.stage6e_corpus import generate_corpus              # noqa: E402
from hpop.mcmc_original.stage7b_diagnostics import SweepMovementTracker    # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "collapsed_u_full_joint"

# Registered before any result exists. New seeds; everything else is Step 7B2's.
CHAIN_SEEDS = (8_160_001, 8_160_002, 8_160_003, 8_160_004)
COLLAPSED_EVERY = 10
N_CHAINS = 4


def load_7b2():
    path = ROOT / "scripts" / "stage7b2_full_joint_ffbs.py"
    spec = importlib.util.spec_from_file_location("stage7b2_full_joint_ffbs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _formal_worker(payload: dict) -> dict:
    """One partially-collapsed chain in its own process, checkpointing as it goes."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "src"))
    b2 = load_7b2()
    baseline = b2.load_baseline_script()
    corpus = generate_corpus()
    model = baseline.build_model(corpus)
    chain = int(payload["chain"])
    start = baseline.dispersed_start(chain, corpus, model, oracle=False)

    parts = log_target_stage6e(start, model)
    if not math.isfinite(parts["log_target"]):
        raise ValueError(f"chain {chain}: start has a non-finite log target")

    tracker = SweepMovementTracker(model.n_skills, model.traces)
    began = time.perf_counter()
    result = run_collapsed_u_chain(
        model=model, start=start, scales=payload["scales"],
        num_sweeps=payload["sweeps"], burn_in=payload["burn_in"],
        thin=payload["thin"], seed=CHAIN_SEEDS[chain],
        collapsed=CollapsedUConfig(every=COLLAPSED_EVERY,
                                   scale=float(payload["scales"]["U"])),
        chain=chain, table_source="fast", store_labels=True, store_keys=False,
        progress_every=payload["progress_every"],
        checkpoint_path=payload["checkpoint_path"],
        checkpoint_every=payload["checkpoint_every"],
        movement_tracker=tracker)

    records = result.collapsed_records
    collapsed_summary = {
        "every": COLLAPSED_EVERY, "attempted": len(records),
        "invalid": sum(1 for r in records if r["invalid"]),
        "h_changed": sum(1 for r in records if r["h_changed"]),
        "accepted": sum(1 for r in records if r["accepted"]),
        "accepted_h_changed": sum(1 for r in records
                                  if r["accepted"] and r["h_changed"]),
        "mean_event_seconds": float(np.mean([r["seconds"] for r in records])
                                    if records else float("nan")),
    }
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
        "collapsed_summary": collapsed_summary,
        "collapsed_records": [{k: v for k, v in r.items()} for r in records],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true",
                        help="launch the registered formal chains")
    parser.add_argument("--sweeps", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--thin", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1_000)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    args = parser.parse_args()
    if not args.formal:
        raise SystemExit("pass --formal to launch the registered run")

    b2 = load_7b2()
    sweeps = args.sweeps if args.sweeps is not None else b2.REGISTERED_SWEEPS
    burn_in = args.burn_in if args.burn_in is not None else b2.REGISTERED_BURN_IN
    thin = args.thin if args.thin is not None else b2.REGISTERED_THIN
    checkpoint_every = (args.checkpoint_every if args.checkpoint_every is not None
                        else b2.CHECKPOINT_EVERY)

    OUT.mkdir(parents=True, exist_ok=True)
    freeze = b2.baseline_freeze_audit()
    corpus_check = b2.corpus_audit()
    starts = b2.starting_state_audit()
    kernels = b2.kernel_audit()
    if not corpus_check["pass"]:
        raise SystemExit(f"corpus does not match the baseline manifest: {corpus_check}")

    registration = {
        "step": "8 — partially-collapsed sampler, formal run on the frozen corpus",
        "source_commit": b2.source_commit(),
        "registered_question": "does one collapsed-U MH move every "
                               f"{COLLAPSED_EVERY} sweeps break the (S,z)-U structural "
                               "locking that held in Stage 6E2 (150k) and Step 7B2 "
                               "(50k) on this corpus?",
        "primary_interpretation_preregistered": {
            "structural_mobility": "accepted cross-H collapsed moves per chain, and "
                                   "relation-count within-chain variability — against "
                                   "7B2's ONE structural change in 28,000 retained "
                                   "draws",
            "locking_conjunction": "evaluate the same A/B/C conjunction the baselines "
                                   "used (chains frozen / relation spread / "
                                   "log-posterior spread) at 50k",
            "convergence": "the same invariant gates the 6E2/7B2 ladder used; "
                           "R-hat threshold 1.01 unchanged",
            "note": "breaking the locking is necessary, not sufficient: gates may "
                    "still fail at 50k for ESS reasons; interpretation follows the "
                    "gates, not enthusiasm"},
        "kernel_under_test": {
            "base": "Step 7B FFBS joint sweep, unchanged",
            "added": f"collapsed-U MH (registered row proposal, scale = baseline U "
                     f"scale) every {COLLAPSED_EVERY} sweeps, immediately followed by "
                     "the sweep's exact FFBS (S,z) refresh",
            "validation": "results/mcmc_original/collapsed_u_kernel_validation and "
                          "_rep2 (600k mixed-reference gates), plus exact tiny-space "
                          "stationarity 2.2e-16"},
        "chain_seeds": list(CHAIN_SEEDS),
        "seeds_are_new": bool(not set(CHAIN_SEEDS)
                              & (set(b2.CHAIN_SEEDS)
                                 | {6_052_001, 6_052_002, 6_052_003, 6_052_004})),
        "collapsed_every_untuned_note": "cadence 10 is the C1 cost-motivated default; "
                                        "deliberately not tuned before this run",
    }
    (OUT / "registration.json").write_text(json.dumps(b2.jsonable(registration),
                                                      indent=2))

    schedule = {"n_chains": N_CHAINS, "sweeps": sweeps, "burn_in": burn_in,
                "thin": thin, "chain_seeds": list(CHAIN_SEEDS),
                "checkpoint_every": checkpoint_every,
                "checkpoint_path": str(OUT / "checkpoints"),
                "table_source": "fast", "collapsed_every": COLLAPSED_EVERY,
                "matches_baseline_first_formal_block": bool(
                    sweeps == b2.REGISTERED_SWEEPS and burn_in == b2.REGISTERED_BURN_IN
                    and thin == b2.REGISTERED_THIN)}
    manifest = b2.freeze_manifest(freeze, corpus_check, starts, kernels, schedule, OUT)
    print(f"[step8] freeze manifest written: corpus "
          f"{corpus_check['rebuilt_corpus_hash'][:16]}  starts "
          f"{[h[:8] for h in starts['hashes']]}  seeds {list(CHAIN_SEEDS)}", flush=True)

    jobs = [{"chain": c, "sweeps": sweeps, "burn_in": burn_in, "thin": thin,
             "scales": kernels["proposal_scales"],
             "progress_every": args.progress_every,
             "checkpoint_path": str(OUT / "checkpoints"),
             "checkpoint_every": checkpoint_every} for c in range(N_CHAINS)]
    print(f"[step8] {N_CHAINS} partially-collapsed chains x {sweeps:,} sweeps, "
          f"burn-in {burn_in:,}, thin {thin}, collapsed every {COLLAPSED_EVERY}, "
          f"checkpoint every {checkpoint_every:,}", flush=True)

    from multiprocessing import get_context
    began = time.perf_counter()
    with get_context("spawn").Pool(processes=N_CHAINS) as pool:
        payloads = pool.map(_formal_worker, jobs)
    wall = time.perf_counter() - began
    for p in payloads:
        c = p["collapsed_summary"]
        print(f"[step8] chain {p['chain']}: {len(p['log_target']):,} retained, "
              f"collapsed {c['attempted']:,} attempts / {c['accepted']:,} accepted "
              f"/ {c['accepted_h_changed']:,} accepted cross-H, "
              f"{p['runtime_seconds'] / 3600:.2f} h", flush=True)

    b2.write_formal_results(payloads, manifest, wall, OUT)
    (OUT / "collapsed_moves.json").write_text(json.dumps(b2.jsonable({
        "per_chain": [{"chain": p["chain"], **p["collapsed_summary"]}
                      for p in payloads],
        "records_per_chain": {str(p["chain"]): p["collapsed_records"]
                              for p in payloads}}), indent=2))
    print(f"[step8] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
