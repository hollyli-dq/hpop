"""Sequential-stopping efficient final validation of the collapsed-U kernel.

    PYTHONPATH=src python scripts/collapsed_u_efficient_validation.py

Prospectively registered sequential protocol: 4 chains from the registered dispersed
starts, brand-new seeds, evaluated by the FROZEN checkpoint evaluator
(`collapsed_u_efficient_gate.py`, written before the first checkpoint) at
150k / 200k / 250k / 300k / 400k / 500k sweeps. TWO CONSECUTIVE full-PASS checkpoints
stop the run with COLLAPSED-U KERNEL VALIDATED; hard correctness failures stop
immediately; distributional/convergence failures continue to the next checkpoint; the
maximum-length rules of the preregistration apply at 500k.

Between checkpoints only process health and sweep counts are monitored — no posterior
quantity is inspected outside the frozen evaluator. Historical directories are
preserved unchanged; the aborted fixed-length final validation is archived non-primary
and its draws are never pooled here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.collapsed_u_kernel import (                        # noqa: E402
    CollapsedUConfig, run_collapsed_u_chain,
)
from hpop.mcmc_original.fast_segmentation_kernel import key_of             # noqa: E402
from hpop.mcmc_original.stage6d_frozen import REGISTERED_SCALES            # noqa: E402
from hpop.mcmc_original.stage6e_frozen import (                            # noqa: E402
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH, config_hash,
)
from hpop.mcmc_original.stage6e_state import Stage6EModel, Stage6EState    # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "collapsed_u_efficient_final_validation"
SEG_DIR = OUT / "chain_checkpoints"
CHAIN_SEEDS = (8_158_001, 8_158_002, 8_158_003, 8_158_004)
CHECKPOINTS = (150_000, 200_000, 250_000, 300_000, 400_000, 500_000)
THIN, CADENCE = 10, 10
SCALARS = ("rho", "beta", "omega", "lambda_rep", "lambda_back")
FROZEN_SOURCES = [
    "src/hpop/mcmc_original/collapsed_u_kernel.py",
    "src/hpop/mcmc_original/collapsed_u_likelihood.py",
    "src/hpop/mcmc_original/semi_markov_ffbs.py",
    "src/hpop/mcmc_original/sampler_u.py",
    "src/hpop/mcmc_original/fast_block_tables.py",
    "src/hpop/mcmc_original/recurrent_joint_ffbs_mcmc.py",
    "src/hpop/mcmc_original/stage6e_sampler.py",
    "scripts/collapsed_u_efficient_gate.py",
]


def sha(path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def problem():
    e1b = _load("stage6e1b", ROOT / "scripts" / "stage6e1b_mixed_reference.py")
    traces, _ = e1b.generate_corpus()
    mixed = e1b.build_mixed_model(traces)
    model = Stage6EModel(traces=traces, epsilon=e1b.EPSILON, delta_b=DELTA_B,
                         n_skills=e1b.K_SKILLS, n_roles=e1b.M_ROLES,
                         min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
                         infer_pi_P=False)
    return e1b, mixed, model


def _block_worker(payload: dict) -> dict:
    """Run one chain from its previous state (or its dispersed start) to `target`."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "src"))
    e1b, mixed, model = problem()
    chain, target, burn_in = payload["chain"], payload["target"], payload["burn_in"]

    state_path = SEG_DIR / f"chain{chain}_state.json"
    if state_path.exists():
        blob = json.loads(state_path.read_text())
        state = Stage6EState.from_dict(blob["state"])
        rng = np.random.default_rng(CHAIN_SEEDS[chain])
        rng.bit_generator.state = state.rng_state
        kwargs = {"state": state, "rng": rng, "start": state}
    else:
        kwargs = {"start": e1b.dispersed_starts(mixed)[chain]}

    began = time.perf_counter()
    result = run_collapsed_u_chain(
        model=model, scales=REGISTERED_SCALES, num_sweeps=target, burn_in=burn_in,
        thin=THIN, seed=CHAIN_SEEDS[chain],
        collapsed=CollapsedUConfig(every=CADENCE,
                                   scale=float(REGISTERED_SCALES["U"])),
        chain=chain, table_source="batched", store_labels=True, store_keys=True,
        **kwargs)
    seconds = time.perf_counter() - began

    first_retained = payload["first_retained_sweep"]
    records = result.collapsed_records
    np.savez_compressed(
        SEG_DIR / f"chain{chain}_seg{target}.npz",
        u_draws=result.u_draws, segment_counts=result.segment_counts,
        relation_counts=result.relation_counts, log_target=result.log_target,
        occurrence_labels=result.occurrence_labels,
        keys_json=json.dumps([[[list(pair) for pair in trace_key]
                               for trace_key in draw]
                              for draw in result.boundary_keys]),
        first_retained_sweep=first_retained,
        last_retained_sweep=first_retained + (len(result.log_target) - 1) * THIN,
        collapsed_sweep=np.array([r["sweep"] for r in records], dtype=np.int64),
        collapsed_accepted=np.array([r["accepted"] for r in records], dtype=bool),
        collapsed_h_changed=np.array([r["h_changed"] for r in records], dtype=bool),
        collapsed_invalid=np.array([r["invalid"] for r in records], dtype=bool),
        **{f"scalar_{n}": result.scalars[n] for n in SCALARS})
    state_path.write_text(json.dumps({"sweep": target,
                                      "state": result.final_state.to_dict()}))
    return {"chain": chain, "target": target, "seconds": seconds,
            "retained": len(result.log_target),
            "collapsed_event_seconds": float(np.mean([r["seconds"] for r in records])
                                             if records else float("nan"))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SEG_DIR.mkdir(exist_ok=True)

    burnin = json.loads((OUT / "burnin_verification.json").read_text())
    burn_in = int(burnin["registered_burn_in"])

    source_manifest = {p: sha(p) for p in FROZEN_SOURCES}
    (OUT / "source_manifest.json").write_text(json.dumps({
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                          text=True).strip(),
        "stage6e_config_hash": config_hash(),
        "hashes": source_manifest,
        "evaluator_frozen_before_first_checkpoint": True}, indent=2))
    (OUT / "seed_manifest.json").write_text(json.dumps({
        "chain_seeds": list(CHAIN_SEEDS),
        "burnin_diagnostic_seeds": [8_158_901, 8_158_902],
        "evaluator_bootstrap_seed_base": 8_158_500,
        "never_used_previously": True}, indent=2))

    preregistration = {
        "protocol": "sequential-stopping validation of the UNCHANGED collapsed-U "
                    "kernel; supersedes the aborted fixed-length final validation "
                    "(archived non-primary, never pooled)",
        "historical_verdicts_unchanged": ["run 1 FAIL", "rep2 FAIL",
                                          "start[0] probe verdict",
                                          "dependence calibration diagnostic"],
        "chains": 4, "starts": "registered dispersed starts 0-3, unchanged",
        "seeds": list(CHAIN_SEEDS), "burn_in": burn_in,
        "burn_in_verified_by": "burnin_verification.json (criterion frozen before "
                               "the diagnostic ran; diagnostic chains throwaway)",
        "thin": THIN, "collapsed_u_every": CADENCE,
        "checkpoints": list(CHECKPOINTS), "maximum_sweeps": 500_000,
        "stopping_rule": "VALIDATED only after TWO CONSECUTIVE full-PASS "
                         "checkpoints; stop immediately then. Distributional or "
                         "convergence FAILs continue to the next checkpoint. Hard "
                         "correctness failures stop immediately.",
        "checkpoint_requirements": ["A: 17 frozen non-energy gates (verbatim, "
                                    "thresholds unchanged)",
                                    "B: chain-balanced dependence-aware energy gate "
                                    "z <= 2.33 (cutoff frozen by the dependence "
                                    "calibration as the one-sided 99% point)",
                                    "C: block lengths {max(2,l//2), l, 2l} agree, "
                                    "l from the frozen 400-sweep dependence scale "
                                    "converted to the extraction spacing",
                                    "D: max registered R-hat <= 1.01 (log posterior, "
                                    "relation count, co-clustering, 5 scalars)",
                                    "E: scalar bulk ESS >= 1000 and tail ESS >= 500 "
                                    "(EXCEPT lambda_rep: bulk >= 600, tail >= 300 — "
                                    "lowered BY USER DECISION BEFORE LAUNCH because "
                                    "the 1000 floor + 500k cap + two-consecutive-"
                                    "pass rule are arithmetically unsatisfiable at "
                                    "its known mixing rate; R-hat still gates it); "
                                    "relation-count and co-clustering bulk ESS >= "
                                    "1000 (no stricter frozen thresholds exist)",
                                    "F: every chain >= 1 joint induced-H change",
                                    "G: no hard correctness failure"],
        "sensitivity_disagreement": "checkpoint verdict INCONCLUSIVE",
        "max_length_rules": {
            "two_consecutive_pass_before_500k": "STOP, VALIDATED",
            "latest_two_fail_same_distributional_gate": "NOT VALIDATED",
            "alternating_or_sensitivity_inconsistent": "INCONCLUSIVE",
            "ess_only_insufficient": "INCONCLUSIVE — INSUFFICIENT EFFECTIVE SAMPLE"},
        "historical_energy_gate": "computed, DESCRIPTIVE ONLY (dependence audit "
                                  "established its null is miscalibrated for "
                                  "autocorrelated, prefix-pooled draws)",
        "no_peek": "between checkpoints only process health and sweep counts are "
                   "monitored; the frozen evaluator runs once per checkpoint and the "
                   "run is never altered after seeing an evaluation",
        "evaluator": {"path": "scripts/collapsed_u_efficient_gate.py",
                      "sha256": source_manifest["scripts/collapsed_u_efficient_gate"
                                                ".py"]},
        "source_manifest": source_manifest,
    }
    (OUT / "preregistration.json").write_text(json.dumps(preregistration, indent=2))
    print(f"[seq-val] preregistered: burn-in {burn_in:,}, seeds {list(CHAIN_SEEDS)}, "
          f"checkpoints {[c // 1000 for c in CHECKPOINTS]}k", flush=True)

    from multiprocessing import get_context
    history = []
    runtime = {"blocks": [], "checkpoints": {}}
    wall_start = time.perf_counter()
    previous_target = None
    verdict = None
    consecutive = 0
    for target in CHECKPOINTS:
        first_retained = (burn_in if previous_target is None
                          else previous_target)
        jobs = [{"chain": c, "target": target, "burn_in": burn_in,
                 "first_retained_sweep": first_retained} for c in range(4)]
        print(f"[seq-val] running chains to {target:,} sweeps", flush=True)
        block_start = time.perf_counter()
        with get_context("spawn").Pool(processes=4) as pool:
            block = pool.map(_block_worker, jobs)
        runtime["blocks"].append({"target": target,
                                  "wall_seconds": time.perf_counter() - block_start,
                                  "per_chain": block})
        previous_target = target

        print(f"[seq-val] evaluating checkpoint {target // 1000}k", flush=True)
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/collapsed_u_efficient_gate.py"),
             "--sweep", str(target)],
            cwd=ROOT, env={**__import__("os").environ,
                           "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "1"},
            capture_output=True, text=True)
        print(proc.stdout.strip(), flush=True)
        if proc.returncode != 0:
            print(proc.stderr[-2000:], flush=True)
            verdict = "COLLAPSED-U VALIDATION INCONCLUSIVE"
            history.append({"sweep": target, "verdict": "HARD-FAILURE",
                            "stderr_tail": proc.stderr[-2000:]})
            break
        checkpoint = json.loads((OUT / f"checkpoint_{target // 1000}k.json")
                                .read_text())
        cumulative_wall = time.perf_counter() - wall_start
        runtime["checkpoints"][str(target)] = {
            "cumulative_wall_seconds": cumulative_wall,
            "evaluation_seconds": checkpoint["evaluation_seconds"],
            "ess_per_second": {
                n: checkpoint["ess"][n]["bulk"] / cumulative_wall
                for n in ("beta", "lambda_rep", "relation_count", "co_clustering")},
            "accepted_h_changes_per_hour": sum(
                m["accepted_cross_h"]
                for m in checkpoint["structural_movement"].values())
            / (cumulative_wall / 3600)}
        history.append({"sweep": target, "verdict": checkpoint["verdict"],
                        "components": checkpoint["components"]})
        (OUT / "sequential_history.json").write_text(json.dumps(history, indent=2))

        consecutive = consecutive + 1 if checkpoint["verdict"] == "PASS" else 0
        if consecutive >= 2:
            verdict = "COLLAPSED-U KERNEL VALIDATED"
            break
    else:
        verdicts = [h["verdict"] for h in history]
        if verdicts[-2:] == ["FAIL", "FAIL"]:
            last_two = [json.loads((OUT / f"checkpoint_{h['sweep'] // 1000}k.json")
                                   .read_text()) for h in history[-2:]]
            shared_fail = set(
                k for k, g in last_two[0]["frozen_gates"].items() if not g["pass"]
            ) & set(k for k, g in last_two[1]["frozen_gates"].items()
                    if not g["pass"])
            both_primary_fail = all(not c["components"]["B_primary_energy"]
                                    and c["components"]["C_sensitivity_agrees"]
                                    for c in last_two)
            verdict = ("COLLAPSED-U KERNEL NOT VALIDATED"
                       if shared_fail or both_primary_fail
                       else "COLLAPSED-U VALIDATION INCONCLUSIVE")
        elif any(h["verdict"] == "INCONCLUSIVE" for h in history[-2:]):
            verdict = "COLLAPSED-U VALIDATION INCONCLUSIVE"
        elif all(h["components"].get("A_frozen_gates") and
                 h["components"].get("B_primary_energy") for h in history[-2:]):
            verdict = ("COLLAPSED-U VALIDATION INCONCLUSIVE — INSUFFICIENT "
                       "EFFECTIVE SAMPLE")
        else:
            verdict = "COLLAPSED-U VALIDATION INCONCLUSIVE"

    total_wall = time.perf_counter() - wall_start
    (OUT / "sequential_history.json").write_text(json.dumps(history, indent=2))
    (OUT / "runtime.json").write_text(json.dumps(
        {**runtime, "total_wall_seconds": total_wall}, indent=2, default=float))
    (OUT / "final_verdict.json").write_text(json.dumps({
        "verdict": verdict, "history": history,
        "stopped_at_sweeps": history[-1]["sweep"] if history else None,
        "total_wall_seconds": total_wall,
        "next": {"if_validated": "unlock matched-synthetic generator validation -> "
                                 "Conditions A/B/C -> progressive release (NOT "
                                 "launched here)",
                 "if_not_validated": "reduced-target decomposition",
                 "if_inconclusive": "stop for review"}}, indent=2))

    lines = [f"# Sequential collapsed-U validation", "", f"**{verdict}**", "",
             f"Burn-in {burn_in:,} (verified), seeds {list(CHAIN_SEEDS)}, "
             f"cadence {CADENCE}, thin {THIN}.",
             "", "| checkpoint | verdict | components |", "|---|---|---|"]
    for h in history:
        comps = h.get("components", {})
        failed = [k for k, v in comps.items() if not v]
        lines.append(f"| {h['sweep'] // 1000}k | {h['verdict']} | "
                     f"{'all pass' if not failed else ', '.join(failed)} |")
    lines += ["", f"Total wall: {total_wall / 3600:.2f} h."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[seq-val] {verdict} (wall {total_wall / 3600:.2f} h)")


if __name__ == "__main__":
    main()
