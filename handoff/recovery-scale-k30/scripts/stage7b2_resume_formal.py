"""Step 7B2 — resume the interrupted formal FFBS chains from their sweep-16,000 checkpoints.

    PYTHONPATH=src python scripts/stage7b2_resume_formal.py

The formal run registered in `freeze_manifest.json` (4 chains x 50,000 sweeps, seeds
7,063,201-7,063,204) was interrupted by a power loss with every chain checkpointed at
sweep 16,000. The chain state carries its own RNG (`state.rng_state`, written at the end
of every sweep before the checkpoint), so restoring it and continuing from
`state.iteration` reproduces bit-for-bit the draws an uninterrupted run would have
produced. Nothing in the registered configuration is changed here: same corpus, same
scales, same schedule, same seeds, same sweep order.

What a resume cannot restore, recorded in `resume_manifest.json` rather than papered
over: the in-memory movement tracker and the post-burn-in acceptance counters restart at
the resume sweep, so the movement series and acceptance rates cover sweeps
16,000-50,000 only (the totals in the checkpointed `state.proposed`/`state.accepted`
still span the whole run), and `table_builds` counts only the resumed portion.

Integrity gates before any sweep runs:
  * the rebuilt corpus hash must equal the frozen manifest's;
  * the log target recomputed at each restored state must match the checkpointed value.
The pre-resume checkpoints are preserved in `checkpoints_block0/` before the loop
overwrites them, so a second interruption loses nothing either.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import run_stage7b_chain  # noqa: E402
from hpop.mcmc_original.recurrent_segmentation import log_target_stage6e    # noqa: E402
from hpop.mcmc_original.stage6e_corpus import corpus_hash, generate_corpus  # noqa: E402
from hpop.mcmc_original.stage6e_frozen import assert_stage6d_unchanged      # noqa: E402
from hpop.mcmc_original.stage6e_sampler import SCALAR_ORDER                 # noqa: E402
from hpop.mcmc_original.stage6e_state import Stage6EState                   # noqa: E402
from hpop.mcmc_original.stage7b_diagnostics import SweepMovementTracker     # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "stage7b2_full_joint_ffbs"
CHECKPOINTS = OUT / "checkpoints"
BLOCK0 = OUT / "checkpoints_block0"
PAYLOADS = OUT / "resume_payloads"

DRAW_KEYS = ("u_draws", "pi_draws", "transition_draws", "segment_counts",
             "relation_counts", "log_target", "log_block_likelihood",
             "occurrence_labels")


def load_module(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resume_worker(payload: dict) -> dict:
    """One chain: restore state + RNG from the block-0 checkpoint, run to 50,000."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "src"))
    baseline = load_module("stage6e2_formal_chains")
    stage7b2 = load_module("stage7b2_full_joint_ffbs")
    corpus = generate_corpus()
    model = baseline.build_model(corpus)
    chain = int(payload["chain"])
    seed = stage7b2.CHAIN_SEEDS[chain]

    checkpoint = json.loads((BLOCK0 / f"chain{chain}_checkpoint.json").read_text())
    state = Stage6EState.from_dict(checkpoint["state"])
    rng = np.random.default_rng(seed)
    rng.bit_generator.state = state.rng_state

    # Integrity gate: the target recomputed at the restored state must be the
    # checkpointed one, or the rebuilt corpus/model is not the run's.
    parts = log_target_stage6e(state, model)
    recorded = float(checkpoint["state"]["components"]["log_target"])
    if not math.isfinite(parts["log_target"]):
        raise ValueError(f"chain {chain}: restored state has a non-finite log target")
    if abs(parts["log_target"] - recorded) > 1e-6:
        raise ValueError(
            f"chain {chain}: recomputed log target {parts['log_target']:.10f} does not "
            f"match the checkpointed {recorded:.10f}; the rebuilt model is not the one "
            "the checkpoint was written against")

    # The dispersed start's log target, for performance.json parity with a fresh run.
    fresh_start = baseline.dispersed_start(chain, corpus, model, oracle=False)
    start_parts = log_target_stage6e(fresh_start, model)

    tracker = SweepMovementTracker(model.n_skills, model.traces)
    began = time.perf_counter()
    result = run_stage7b_chain(
        model=model, start=state, state=state, rng=rng,
        scales=payload["scales"],
        num_sweeps=payload["sweeps"], burn_in=payload["burn_in"],
        thin=payload["thin"], seed=seed, chain=chain,
        table_source="fast", store_labels=True, store_keys=False,
        progress_every=payload["progress_every"],
        checkpoint_path=str(CHECKPOINTS),
        checkpoint_every=payload["checkpoint_every"],
        movement_tracker=tracker)
    runtime = time.perf_counter() - began

    # Splice the block-0 retained draws in front of the resumed portion's.
    with np.load(BLOCK0 / f"chain{chain}_checkpoint.npz") as archive:
        block0 = {key: archive[key] for key in archive.files}
    scalars = {name: np.concatenate([block0[f"scalar_{name}"], result.scalars[name]])
               for name in (*SCALAR_ORDER, "rho")}
    new = {"u_draws": result.u_draws, "pi_draws": result.pi_draws,
           "transition_draws": result.transition_draws,
           "segment_counts": result.segment_counts,
           "relation_counts": result.relation_counts,
           "log_target": result.log_target,
           "log_block_likelihood": result.log_block_likelihood,
           "occurrence_labels": result.occurrence_labels}
    draws = {key: np.concatenate([block0[key], new[key]], axis=0) for key in DRAW_KEYS}

    return {
        "chain": chain, "seed": seed,
        "start_log_target": float(start_parts["log_target"]),
        "resume_log_target": float(parts["log_target"]),
        "resumed_from_sweep": int(checkpoint["sweep"]),
        "block0_retained": int(checkpoint["n_retained"]),
        "u_draws": draws["u_draws"], "scalars": scalars,
        "pi_draws": draws["pi_draws"], "transition_draws": draws["transition_draws"],
        "segment_counts": draws["segment_counts"],
        "occurrence_labels": draws["occurrence_labels"],
        "relation_counts": draws["relation_counts"],
        "log_target": draws["log_target"],
        "log_block_likelihood": draws["log_block_likelihood"],
        "proposed": result.proposed, "accepted": result.accepted,
        "invalid": result.invalid,
        "acceptance_rates": result.acceptance(),
        "movement_totals": result.movement,
        "movement_summary": tracker.summary(),
        "movement_series": tracker.series(),
        "table_builds": int(result.table_builds),
        "final_state": result.final_state.to_dict(),
        "runtime_seconds": runtime,
    }


def main() -> None:
    from multiprocessing import get_context

    assert_stage6d_unchanged()
    stage7b2 = load_module("stage7b2_full_joint_ffbs")
    manifest = json.loads((OUT / "freeze_manifest.json").read_text())
    schedule = manifest["schedule"]

    rebuilt = corpus_hash(generate_corpus())
    frozen = manifest["corpus"]["rebuilt_corpus_hash"]
    if rebuilt != frozen:
        raise SystemExit(f"corpus hash {rebuilt} does not match the frozen {frozen}")

    # Preserve the pre-resume checkpoints once; a re-run of this script after a second
    # interruption must keep resuming from the block-0 files, never from its own output.
    if not BLOCK0.exists():
        shutil.copytree(CHECKPOINTS, BLOCK0)
        print(f"[7B2-resume] preserved sweep-16,000 checkpoints in {BLOCK0.name}/",
              flush=True)

    sweeps_done = {json.loads((BLOCK0 / f"chain{c}_checkpoint.json").read_text())["sweep"]
                   for c in range(schedule["n_chains"])}
    print(f"[7B2-resume] resuming {schedule['n_chains']} chains from sweeps "
          f"{sorted(sweeps_done)} to {schedule['sweeps']:,} "
          f"(burn-in {schedule['burn_in']:,}, thin {schedule['thin']}); "
          f"seeds {schedule['chain_seeds']}", flush=True)

    jobs = [{"chain": c, "sweeps": schedule["sweeps"], "burn_in": schedule["burn_in"],
             "thin": schedule["thin"],
             "scales": manifest["kernels"]["proposal_scales"],
             "progress_every": 500,
             "checkpoint_every": schedule["checkpoint_every"]}
            for c in range(schedule["n_chains"])]

    began = time.perf_counter()
    with get_context("spawn").Pool(processes=schedule["n_chains"]) as pool:
        payloads = pool.map(_resume_worker, jobs)
    wall = time.perf_counter() - began

    # Raw payloads first: if the assembly below fails, nothing is lost.
    PAYLOADS.mkdir(exist_ok=True)
    for payload in payloads:
        with open(PAYLOADS / f"chain{payload['chain']}_payload.pkl", "wb") as f:
            pickle.dump(payload, f)

    expected = (schedule["sweeps"] - schedule["burn_in"]) // schedule["thin"]
    for payload in payloads:
        kept = len(payload["log_target"])
        marker = "OK" if kept == expected else f"EXPECTED {expected}"
        print(f"[7B2-resume] chain {payload['chain']}: {kept:,} retained ({marker}), "
              f"resumed portion {payload['runtime_seconds'] / 3600:.2f} h", flush=True)

    resume_manifest = {
        "resumed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reason": "power loss; every chain checkpointed at sweep 16,000",
        "resumed_from_sweep": {p["chain"]: p["resumed_from_sweep"] for p in payloads},
        "block0_retained": {p["chain"]: p["block0_retained"] for p in payloads},
        "bit_identical": "the checkpointed state carries the sweep-end RNG, so the "
                         "resumed draw sequence is the one the uninterrupted run would "
                         "have produced; retained draws are spliced with no gap and no "
                         "overlap (block 0 retains sweeps 15,000-15,995, the resumed "
                         "portion 16,000-49,995, both on the registered thin of 5)",
        "resume_integrity_gate": {
            p["chain"]: {"recomputed_log_target": p["resume_log_target"],
                         "matched_checkpoint": True} for p in payloads},
        "not_restored": {
            "movement_series_and_summary": "cover sweeps 16,000-50,000 only; the "
                                           "tracker for sweeps 0-16,000 died with the "
                                           "interrupted process and was never on disk",
            "acceptance_rates": "post-burn-in counters restart at 16,000, so rates omit "
                                "sweeps 15,000-16,000 (1,000 of 35,000 post-burn-in "
                                "sweeps); whole-run totals in proposed/accepted are "
                                "intact from the checkpointed state",
            "table_builds": "counts the resumed 34,000 sweeps only, so "
                            "performance.json's builds_equal_sweeps reads false",
            "runtime": "per-chain runtime_seconds and the wall clock cover the resumed "
                       "portion; the pre-interruption portion ran 2026-08-14 "
                       "10:40-16:22 BST",
        },
        "configuration_changes": "none; corpus, scales, schedule, seeds and sweep order "
                                 "are the freeze manifest's",
    }
    (OUT / "resume_manifest.json").write_text(json.dumps(resume_manifest, indent=2))

    stage7b2.write_formal_results(payloads, manifest, wall, OUT)
    print(f"[7B2-resume] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
