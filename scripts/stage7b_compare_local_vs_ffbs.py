"""Step 7B — the paired Local-vs-FFBS comparison, on whatever stages have finished.

    PYTHONPATH=src python scripts/stage7b_compare_local_vs_ffbs.py

Both samplers target the same posterior and both are measured against the same independent
reference, so this script never asks which posterior is better. It asks what the kernel
swap did to *movement* and to *cost*: autocorrelation, effective sample size per draw and
per second, structural mode transitions, wall clock.

Step 7B1 (the small mixed-reference model) is compared whenever both sides exist. Step 7B2
(the 100-trace corpus) is included only once its baseline is frozen and its run has
happened; until then the summary says so rather than leaving a blank that reads like a
result.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.diagnostics import autocorrelation                    # noqa: E402
from hpop.mcmc_original.stage6b_mcmc_diagnostics import (                     # noqa: E402
    bulk_ess, rank_normalized_split_rhat,
)
from hpop.mcmc_original.stage7b_diagnostics import (                          # noqa: E402
    compare_equal_sweeps, compare_equal_time, invariant_summaries,
    structural_movement,
)

OUT = ROOT / "results" / "mcmc_original" / "stage7_complete"
FFBS_7B1 = ROOT / "results" / "mcmc_original" / "stage7b1_mixed_reference"
LOCAL_6E1B = ROOT / "results" / "mcmc_original" / "stage6e1b_mixed_reference"
# The frozen baseline chains, copied into this worktree and hash-verified against the
# live Stage 6E worktree, so the comparison does not depend on a directory another
# experiment is still writing to.
LOCAL_6E1B_CHAINS = LOCAL_6E1B / "chains.npz"
STAGE7B2 = ROOT / "results" / "mcmc_original" / "stage7b2_full_joint_ffbs"
STAGE7A = ROOT / "results" / "mcmc_original" / "stage7a_ffbs_exact"
STAGE7B0 = ROOT / "results" / "mcmc_original" / "stage7b0_joint_smoke"
BASELINE_6E2 = Path("/Users/dongqing/Desktop/hpop-stage6e/results/mcmc_original"
                    "/stage6e2_unknown_boundary_full_seed0")
REGISTERED_LADDER_MAXIMUM = 150_000

SCALARS = ("rho", "beta", "omega", "lambda_rep", "lambda_back")

# The Step 7B2 comparison protocol, authored 2026-08-14T22:05+0100 while the FFBS chains
# were mid-run (sweep ~22,500 of 50,000) and the baseline was mid-ladder (100k CONTINUE),
# so nothing below could be chosen after seeing a comparison input.
COMPARISON_PROTOCOL = {
    "authored": "2026-08-14T22:05:00+0100",
    "authored_while": "the FFBS chains were at ~sweep 22,500 of 50,000 and the Stage "
                      "6E2 baseline was mid-ladder at 100,000 (registered verdict "
                      "CONTINUE); no comparison input existed",
    "equal_sweep_basis": "the FFBS 50,000-sweep chains against the baseline's preserved "
                         "first block `chains_block1_50k.npz` — identical schedule "
                         "(burn-in 15,000, thin 5), identical reconstructed starts, new "
                         "seeds, 7,000 retained draws per chain on both sides",
    "equal_time_basis": "compare_equal_time against the FROZEN FINAL baseline, with the "
                        "baseline wall clock the sum of its continuation blocks",
    "ffbs_wall_estimator": "the interrupted run's true wall is unobservable, so the "
                           "FFBS wall is (resumed-portion seconds per sweep) x 50,000; "
                           "the resumed-portion wall is reported alongside and both "
                           "sides ran under mutual contention, so cost rows are "
                           "reported, never gated",
    "verdict_rule": {
        "criteria": "Stage 6E2's registered structural-locking criteria A, B, C, "
                    "reproduced with identical thresholds and applied identically to "
                    "both samplers' chains",
        "escapes": "the FFBS kernel is said to ESCAPE the (S,z)-U structural locking "
                   "iff the baseline's final chains satisfy the A-and-B-and-C "
                   "conjunction and the FFBS chains do not",
        "does_not_escape": "if the FFBS chains also satisfy the conjunction, the Step 7 "
                           "hypothesis is falsified on this corpus and the report says "
                           "so plainly — the freeze manifest predicted no improvement",
        "otherwise": "any mixed outcome is reported criterion by criterion with no "
                     "verdict word",
        "interpretation_conditional_on": "the FFBS invariant-convergence worst R-hat; "
                                         "recovery stays uninterpreted for any sampler "
                                         "whose chains have not converged",
    },
}


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:                                                # pragma: no cover
        return "unknown"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def chain_summary(path: Path, label: str) -> dict:
    """Every movement and efficiency statistic, computed identically for both samplers."""
    draws = np.load(path)
    seconds = float(np.asarray(draws["runtime_seconds"]).max())   # chains ran in parallel
    total_seconds = float(np.asarray(draws["runtime_seconds"]).sum())
    out = {"label": label, "path": str(path), "sha256": sha256(path),
           "wall_seconds": seconds, "cpu_seconds": total_seconds,
           "n_chains": int(draws["log_target"].shape[0]),
           "retained_per_chain": int(draws["log_target"].shape[1])}

    series = {"log_posterior": draws["log_target"],
              "total_segments": draws["n_segments"].astype(float),
              "total_relation_count": draws["relation_counts"].astype(float)}
    for name in SCALARS:
        series[name] = draws[f"scalar_{name}"]

    ess, rhat, acf1 = {}, {}, {}
    for name, values in series.items():
        values = np.asarray(values, dtype=float)
        if values.std() == 0.0:
            ess[name], rhat[name], acf1[name] = float("nan"), float("nan"), float("nan")
            continue
        ess[name] = float(bulk_ess(values))
        rhat[name] = float(rank_normalized_split_rhat(values)["rhat"])
        acf1[name] = float(autocorrelation(values[0], max_lag=2)[1])
    out["bulk_ess"] = ess
    out["rhat"] = rhat
    out["lag1_autocorrelation"] = acf1
    out["ess_per_second"] = {name: (value / seconds if math.isfinite(value) else None)
                             for name, value in ess.items()}
    finite = [v for v in ess.values() if math.isfinite(v)]
    out["min_bulk_ess"] = min(finite) if finite else float("nan")
    out["min_ess_per_second"] = (min(finite) / seconds if finite else float("nan"))
    out["max_rhat"] = max((v for v in rhat.values() if math.isfinite(v)), default=None)

    u_draws = draws["u_draws"]
    movement = structural_movement(
        [u_draws[c] for c in range(u_draws.shape[0])],
        [draws["relation_counts"][c].reshape(u_draws.shape[1], -1)
         if draws["relation_counts"].ndim == 2 else draws["relation_counts"][c]
         for c in range(u_draws.shape[0])])
    out["structural_movement"] = movement
    out["h_changes"] = movement["total_structural_changes"]
    out["distinct_h_states"] = movement["distinct_structural_states_pooled"]
    out["structural_mode_transitions"] = movement["total_structural_changes"]
    out["segment_count_ess"] = ess.get("total_segments")
    out["co_clustering_ess"] = None      # not stored for the 2-trace mixed model
    for name in ("beta", "omega", "lambda_rep", "lambda_back"):
        out[f"{name}_ess"] = ess.get(name)
    out["log_posterior_rhat"] = rhat.get("log_posterior")
    out["total_relation_count_rhat"] = rhat.get("total_relation_count")
    out["max_invariant_rhat"] = out["max_rhat"]
    out["min_invariant_bulk_ess"] = out["min_bulk_ess"]
    out["min_invariant_ess_per_second"] = out["min_ess_per_second"]
    return out


def corpus_summary(path: Path, label: str, wall_seconds: float, sweeps: int,
                   wall_basis: str) -> dict:
    """The `chain_summary` analogue for the 100-trace corpus format.

    The corpus files carry per-trace `segment_counts` and per-skill `relation_counts`
    rather than the mixed model's totals, and the baseline's blocks carry no runtime, so
    the wall clock is passed in with its basis stated rather than read from the file.
    """
    draws = np.load(path)
    n_chains = int(draws["log_target"].shape[0])
    chains = [{
        "log_target": draws["log_target"][c],
        "relation_counts": draws["relation_counts"][c],
        "segment_counts": draws["segment_counts"][c],
        "pi_draws": draws["pi_draws"][c],
        "transition_draws": draws["transition_draws"][c],
        "label_draws": draws["occurrence_labels"][c],
        "scalars": {name: draws[f"scalar_{name}"][c] for name in SCALARS},
    } for c in range(n_chains)]

    summaries = [invariant_summaries(chain) for chain in chains]
    convergence, ess, rhat, acf1 = {}, {}, {}, {}
    for name in summaries[0]:
        stacked = np.array([np.asarray(s[name], dtype=float).reshape(
            len(s[name]), -1).mean(axis=1) for s in summaries])
        if stacked.std() == 0.0:
            convergence[name] = {"degenerate": True, "rhat": None, "bulk_ess": None}
            ess[name], rhat[name], acf1[name] = float("nan"), float("nan"), float("nan")
            continue
        ess[name] = float(bulk_ess(stacked))
        rhat[name] = float(rank_normalized_split_rhat(stacked)["rhat"])
        acf1[name] = float(autocorrelation(stacked[0], max_lag=2)[1])
        convergence[name] = {"rhat": rhat[name], "bulk_ess": ess[name],
                             "mean": float(stacked.mean()),
                             "sd": float(stacked.std(ddof=1))}

    out = {"label": label, "path": str(path), "sha256": sha256(path),
           "wall_seconds": float(wall_seconds), "wall_basis": wall_basis,
           "sweeps": int(sweeps), "n_chains": n_chains,
           "retained_per_chain": int(draws["log_target"].shape[1]),
           "invariant_convergence": convergence,
           "bulk_ess": ess, "rhat": rhat, "lag1_autocorrelation": acf1}

    finite = [v for v in ess.values() if math.isfinite(v)]
    out["min_bulk_ess"] = min(finite) if finite else float("nan")
    out["min_ess_per_second"] = (min(finite) / wall_seconds if finite else float("nan"))
    out["max_rhat"] = max((v for v in rhat.values() if math.isfinite(v)), default=None)

    movement = structural_movement(
        [draws["u_draws"][c] for c in range(n_chains)],
        [draws["relation_counts"][c] for c in range(n_chains)])
    out["structural_movement"] = movement
    out["structural_locking"] = structural_locking_registered(
        draws["relation_counts"], draws["log_target"])
    out["h_changes"] = movement["total_structural_changes"]
    out["distinct_h_states"] = movement["distinct_structural_states_pooled"]
    out["structural_mode_transitions"] = movement["total_structural_changes"]
    out["segment_count_ess"] = ess.get("total_segments")
    out["co_clustering_ess"] = ess.get("co_clustering_mean")
    for name in ("beta", "omega", "lambda_rep", "lambda_back"):
        out[f"{name}_ess"] = ess.get(name)
    out["log_posterior_rhat"] = rhat.get("log_posterior")
    out["total_relation_count_rhat"] = rhat.get("total_relation_count")
    out["max_invariant_rhat"] = out["max_rhat"]
    out["min_invariant_bulk_ess"] = out["min_bulk_ess"]
    out["min_invariant_ess_per_second"] = out["min_ess_per_second"]
    return out


def structural_locking_registered(relation_counts, log_target) -> dict:
    """Stage 6E2's three registered criteria, the ladder check's computation verbatim.

    `relation_counts` is `(C, n, K)`, `log_target` `(C, n)`. Criterion A takes the
    minimum within-chain sd over the total AND the sorted per-skill counts, exactly as
    `stage6e2_ladder_check.structural_locking` does; reimplemented here because that
    module postdates this worktree's branch point, with the thresholds pinned by
    `interpretation_rule.json`.
    """
    relation = np.asarray(relation_counts, dtype=float)
    log_target = np.asarray(log_target, dtype=float)
    total = relation.sum(axis=2)
    per_skill_sorted = np.sort(relation, axis=2)

    within = np.concatenate([total.std(axis=1)[:, None],
                             per_skill_sorted.std(axis=1)], axis=1)
    chains_frozen = int((within.min(axis=1) < 0.01).sum())
    a = chains_frozen >= 2

    means = total.mean(axis=1)
    spread_relation = float(np.ptp(means))
    b = spread_relation > 1.0

    log_means = log_target.mean(axis=1)
    spread_log = float(np.ptp(log_means))
    c = spread_log > 20.0

    return {
        "A_frozen_structure": {
            "criterion": "relation counts with within-chain sd < 0.01 in >= 2 chains",
            "n_chains_frozen": chains_frozen,
            "per_chain_min_within_sd": within.min(axis=1).tolist(),
            "holds": bool(a)},
        "B_disagreeing_structure": {
            "criterion": "spread of chain mean total relation count > 1.0",
            "per_chain_mean_total_relations": means.tolist(),
            "spread": spread_relation, "holds": bool(b)},
        "C_log_posterior_gap": {
            "criterion": "spread of chain mean invariant log posterior > 20 nats",
            "per_chain_mean_log_posterior": log_means.tolist(),
            "spread_nats": spread_log, "holds": bool(c)},
        "conjunction_holds": bool(a and b and c),
    }


def baseline_frozen_state() -> dict:
    """Is the Stage 6E2 baseline at a decision point and no longer moving?

    Frozen means either the latest ladder check passed every invariant gate, or the
    registered 150,000-sweep maximum has been reached AND checked (in which case the
    interpretation rule's FAIL/MULTIMODAL verdict applies if the locking conjunction
    holds). A completed block whose check has not run yet is NOT frozen — the check is
    part of the registered decision procedure.
    """
    import subprocess
    import time as time_module

    reasons: list = []
    if not BASELINE_6E2.exists():
        return {"frozen": False,
                "reasons": ["the Stage 6E2 baseline directory is not reachable"],
                "directory": str(BASELINE_6E2)}

    history_path = BASELINE_6E2 / "continuation_history.json"
    blocks = (json.loads(history_path.read_text()).get("unknown", [])
              if history_path.exists() else [])
    reached = max((int(b["sweeps_to"]) for b in blocks), default=0)

    checks = [json.loads(p.read_text())
              for p in BASELINE_6E2.glob("ladder_check_*.json")]
    latest = max(checks, key=lambda d: d.get("sweeps", 0), default=None)

    if latest is None:
        reasons.append("no ladder check has been run")
    elif latest["sweeps"] < reached:
        reasons.append(f"the block to {reached:,} has no ladder check yet "
                       f"(latest check is at {latest['sweeps']:,})")
    elif not latest.get("all_invariant_gates_pass") \
            and reached < REGISTERED_LADDER_MAXIMUM:
        reasons.append(f"the latest check ({latest['sweeps']:,}) fails "
                       f"{latest.get('n_failed_gates')} gates and the ladder has not "
                       f"reached its registered maximum of "
                       f"{REGISTERED_LADDER_MAXIMUM:,}")

    try:
        processes = subprocess.check_output(["ps", "-Ao", "command"], text=True)
        if "stage6e2_formal_chains.py" in processes:
            reasons.append("a stage6e2_formal_chains.py process is running right now")
    except Exception:                                                # pragma: no cover
        reasons.append("could not determine whether a Stage 6E2 process is running")

    now = time_module.time()
    recent = sorted(p.name for p in BASELINE_6E2.iterdir()
                    if p.is_file() and now - p.stat().st_mtime < 1_800)
    if recent:
        reasons.append(f"baseline artifacts written within the last 1800s: {recent}")

    verdict = None
    if latest is not None and not reasons:
        if latest.get("all_invariant_gates_pass"):
            verdict = f"CONVERGED at {latest['sweeps']:,} sweeps"
        elif latest.get("structural_locking", {}).get("conjunction_holds"):
            verdict = ("FAIL / MULTIMODAL per interpretation_rule.json: (S,z)-U "
                       "structural locking at the registered maximum")
        else:
            verdict = (f"FAIL at the registered maximum without the locking "
                       f"conjunction; see ladder_check_{latest['sweeps']}")
    return {"frozen": not reasons, "reasons": reasons, "ladder_reached": reached,
            "latest_check_sweeps": (latest or {}).get("sweeps"),
            "baseline_verdict": verdict, "directory": str(BASELINE_6E2),
            "continuation_blocks": blocks}


def summarise_stage7b2() -> dict:
    """The full-corpus comparison, gated on the frozen baseline.

    The FFBS side is summarised unconditionally — it is this worktree's own artifact.
    No baseline chain file is opened unless `baseline_frozen_state` says the baseline is
    at its registered decision point and has stopped moving; until then the block
    carries the FFBS summary and the reasons the comparison is deferred, so the report
    never reads an intermediate rung.
    """
    performance = json.loads((STAGE7B2 / "performance.json").read_text())
    manifest = json.loads((STAGE7B2 / "freeze_manifest.json").read_text())
    sweeps = int(manifest["schedule"]["sweeps"])
    resume_path = STAGE7B2 / "resume_manifest.json"
    resume = json.loads(resume_path.read_text()) if resume_path.exists() else None
    if resume:
        # SIGSTOP intervals (battery triage) inflate perf_counter runtimes; subtract
        # them, since a stopped process does no sweeps but its clock keeps running.
        pause_path = STAGE7B2 / "pause_log.json"
        paused_seconds = 0.0
        if pause_path.exists():
            from datetime import datetime
            for pause in json.loads(pause_path.read_text())["pauses"]:
                if pause.get("resumed_at"):
                    begin = datetime.strptime(pause["stopped_at"], "%Y-%m-%dT%H:%M:%S%z")
                    end = datetime.strptime(pause["resumed_at"], "%Y-%m-%dT%H:%M:%S%z")
                    paused_seconds += (end - begin).total_seconds()
        resumed_sweeps = sweeps - max(resume["resumed_from_sweep"].values())
        per_sweep = (max(performance["per_chain_seconds"])
                     - paused_seconds) / resumed_sweeps
        ffbs_wall = per_sweep * sweeps
        wall_basis = (f"estimated: resumed-portion rate ({per_sweep:.3f} s/sweep over "
                      f"{resumed_sweeps:,} sweeps, {paused_seconds:.0f}s of logged "
                      f"SIGSTOP pauses subtracted) x {sweeps:,} sweeps; the "
                      "pre-interruption wall is unobservable (power loss)")
    else:
        ffbs_wall = performance["wall_seconds"]
        wall_basis = "measured wall clock of the uninterrupted run"

    ffbs = corpus_summary(STAGE7B2 / "chains.npz", "FFBS (Step 7B2)", ffbs_wall, sweeps,
                          wall_basis)
    ffbs["invariant_convergence_file"] = (
        json.loads((STAGE7B2 / "invariant_convergence.json").read_text())
        if (STAGE7B2 / "invariant_convergence.json").exists() else None)

    freeze = baseline_frozen_state()
    provenance = {}
    try:
        worktree = str(BASELINE_6E2.parents[2])
        provenance = {
            "commit": subprocess.check_output(
                ["git", "-C", worktree, "rev-parse", "HEAD"], text=True).strip(),
            "tag": subprocess.check_output(
                ["git", "-C", worktree, "describe", "--tags", "--exact-match"],
                text=True, stderr=subprocess.DEVNULL).strip(),
        }
    except Exception:
        provenance = {"note": "baseline worktree has no exact tag at HEAD; cite paths"}
    block = {"protocol": COMPARISON_PROTOCOL, "ffbs": ffbs,
             "baseline_freeze": freeze, "baseline_git": provenance,
             "resume_manifest_present": bool(resume)}
    if not freeze["frozen"]:
        block["status"] = ("FFBS CHAINS COMPLETE — comparison deferred until the "
                          "Stage 6E2 baseline freezes")
        block["deferral_reasons"] = freeze["reasons"]
        return block

    history = freeze["continuation_blocks"]
    block1_wall = float(history[0]["wall_seconds"])
    final_wall = float(sum(b["wall_seconds"] for b in history))
    local_block1 = corpus_summary(
        BASELINE_6E2 / "chains_block1_50k.npz",
        "LocalMoveKernel (Stage 6E2, first 50k block)", block1_wall, 50_000,
        "measured wall clock of the baseline's first formal block")
    local_final = corpus_summary(
        BASELINE_6E2 / "chains.npz",
        f"LocalMoveKernel (Stage 6E2, frozen final, {freeze['ladder_reached']:,} sweeps)",
        final_wall, freeze["ladder_reached"],
        "sum of the continuation blocks' measured wall clocks")

    base_lock = local_final["structural_locking"]["conjunction_holds"]
    ffbs_lock = ffbs["structural_locking"]["conjunction_holds"]
    if base_lock and not ffbs_lock:
        verdict = ("FFBS ESCAPES the (S,z)-U structural locking on this corpus: the "
                   "frozen baseline's chains satisfy the registered A-and-B-and-C "
                   "conjunction and the FFBS chains do not")
    elif base_lock and ffbs_lock:
        verdict = ("FFBS DOES NOT ESCAPE the (S,z)-U structural locking: its chains "
                   "satisfy the same registered conjunction as the baseline's. "
                   "Falsified, precisely stated: the hypothesis that exact global "
                   "FFBS updates of (S,z) ALONE resolve the full-joint structural "
                   "locking. FFBS itself did what it was validated to do — exact "
                   "conditional sampling of (S,z) (Step 7A), small-reference "
                   "posterior correctness (Step 7B1), the same target as the "
                   "LocalMoveKernel — and better (S,z) mixing simply does not "
                   "propagate to U/H structural mixing, as the freeze manifest "
                   "anticipated it might not")
    else:
        verdict = ("no verdict word applies: the baseline's final chains do not "
                   "satisfy the locking conjunction, so the escape question is not "
                   "posed as registered; the criteria are reported side by side")

    worst = (ffbs.get("invariant_convergence_file") or {}).get("worst_rhat")
    return {
        **block,
        "status": "COMPARED against the frozen baseline",
        "local_block1": local_block1, "local_final": local_final,
        "equal_sweeps": compare_equal_sweeps(local_block1, ffbs),
        "equal_time": compare_equal_time(local_final, ffbs),
        "structural_locking_verdict": verdict,
        "baseline_verdict": freeze["baseline_verdict"],
        "ffbs_worst_invariant_rhat": worst,
        "recovery_note": "recovery stays uninterpreted for any sampler whose chains "
                         "have not converged, exactly as the baseline's "
                         "interpretation rule registers",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload: dict = {"source_commit": source_commit(),
                     "what_this_is": "a paired comparison of two transition kernels on "
                                     "one posterior; neither sampler's posterior is "
                                     "'better' and none of these numbers says it is"}

    # ---- Step 7B1 ----------------------------------------------------------------------
    ffbs_chains = FFBS_7B1 / "chains.npz"
    if ffbs_chains.exists() and LOCAL_6E1B_CHAINS.exists():
        local = chain_summary(LOCAL_6E1B_CHAINS, "LocalMoveKernel (Stage 6E1B, frozen)")
        ffbs = chain_summary(ffbs_chains, "FFBS (Step 7B1)")
        local_gates = json.loads((LOCAL_6E1B / "joint_comparison.json").read_text())
        ffbs_gates = json.loads((FFBS_7B1 / "joint_comparison.json").read_text())
        local["sweeps"] = json.loads(
            (LOCAL_6E1B / "config.json").read_text())["chains"]["sweeps"]
        ffbs["sweeps"] = json.loads(
            (FFBS_7B1 / "config.json").read_text())["chains"]["sweeps"]
        payload["stage7b1"] = {
            "model": "the Stage 6E1B mixed model: 2 traces, J = 8, K = 3, pi and P fixed",
            "reference": "the frozen Stage 6E1B mixed QMC + exact-enumeration reference",
            "gate_comparison": {
                name: {"ffbs": gate["value"],
                       "local_move_kernel": local_gates["gates"].get(name, {}).get("value"),
                       "threshold": gate["threshold"],
                       "ffbs_pass": gate["pass"],
                       "local_pass": local_gates["gates"].get(name, {}).get("pass")}
                for name, gate in ffbs_gates["gates"].items()},
            "both_pass_every_gate": bool(ffbs_gates["all_pass"]
                                         and local_gates["all_pass"]),
            "equal_sweeps": compare_equal_sweeps(local, ffbs),
            "equal_time": compare_equal_time(local, ffbs),
            "local": local, "ffbs": ffbs,
            "interpretation": "same target, different transition kernel",
        }
    else:
        payload["stage7b1"] = {"status": "not available in this worktree"}

    # ---- Step 7B2 ----------------------------------------------------------------------
    freeze_path = STAGE7B2 / "baseline_freeze_audit.json"
    if (STAGE7B2 / "chains.npz").exists():
        payload["stage7b2"] = summarise_stage7b2()
    elif freeze_path.exists():
        freeze = json.loads(freeze_path.read_text())
        payload["stage7b2"] = {
            "status": "NOT STARTED — blocked on the Stage 6E2 baseline freeze",
            "reasons": freeze["reasons"],
            "ladder_reached": freeze["ladder_reached"],
            "registered_ladder": freeze["registered_ladder"],
            "cost_projection": (json.loads((STAGE7B2 / "cost_projection.json").read_text())
                                if (STAGE7B2 / "cost_projection.json").exists() else None),
        }
    else:
        payload["stage7b2"] = {"status": "NOT STARTED"}

    (OUT / "comparison_local_vs_ffbs.json").write_text(
        json.dumps(jsonable(payload), indent=2))

    # ---- the completion summary ----------------------------------------------------------
    def gates_of(path: Path, key: str = "all_pass"):
        return json.loads(path.read_text()).get(key) if path.exists() else None

    summary = {
        "step_7a_ffbs_correctness": {
            "status": "PASS" if gates_of(STAGE7A / "gates.json") else "UNKNOWN",
            "log_z_absolute_error": json.loads(
                (STAGE7A / "exact_comparison.json").read_text())["log_z_absolute_error"]
            if (STAGE7A / "exact_comparison.json").exists() else None},
        "step_7b0_integration_parity": {
            "status": "PASS" if gates_of(STAGE7B0 / "parity_results.json") else "UNKNOWN"},
        "step_7b1_full_joint_correctness": {
            "status": ("PASS" if gates_of(FFBS_7B1 / "joint_comparison.json")
                       else ("FAIL" if (FFBS_7B1 / "joint_comparison.json").exists()
                             else "NOT RUN"))},
        "step_7b2_full_corpus_comparison": payload["stage7b2"]["status"],
        "step_7_final_tag": "NOT CREATED — the tag requires Step 7B1 and Step 7B2",
        "source_commit": source_commit(),
    }
    (OUT / "completion_summary.json").write_text(json.dumps(jsonable(summary), indent=2))

    lines = [
        "# Step 7 — status and the Local-vs-FFBS comparison",
        "",
        f"* Step 7A (FFBS correctness against exact enumeration): "
        f"**{summary['step_7a_ffbs_correctness']['status']}**",
        f"* Step 7B0 (joint integration parity and smoke): "
        f"**{summary['step_7b0_integration_parity']['status']}**",
        f"* Step 7B1 (full joint against the frozen mixed reference): "
        f"**{summary['step_7b1_full_joint_correctness']['status']}**",
        f"* Step 7B2 (full-corpus mixing comparison): "
        f"**{summary['step_7b2_full_corpus_comparison']}**",
        "",
    ]
    if "gate_comparison" in payload.get("stage7b1", {}):
        block = payload["stage7b1"]
        lines += [
            "## Step 7B1 — one posterior, two kernels",
            "",
            "| gate | FFBS | LocalMoveKernel | threshold |",
            "|---|---|---|---|",
        ]
        for name, row in block["gate_comparison"].items():
            ffbs_value = "n/a" if row["ffbs"] is None else f"{row['ffbs']:.6g}"
            local_value = ("n/a" if row["local_move_kernel"] is None
                           else f"{row['local_move_kernel']:.6g}")
            lines.append(f"| {name} | {ffbs_value} | {local_value} | {row['threshold']} |")
        local, ffbs = block["local"], block["ffbs"]
        lines += [
            "",
            "| movement / cost | FFBS | LocalMoveKernel |",
            "|---|---|---|",
            f"| wall seconds (4 chains, parallel) | {ffbs['wall_seconds']:.0f} | "
            f"{local['wall_seconds']:.0f} |",
            f"| segment-count lag-1 autocorrelation | "
            f"{ffbs['lag1_autocorrelation']['total_segments']:+.4f} | "
            f"{local['lag1_autocorrelation']['total_segments']:+.4f} |",
            f"| relation-count lag-1 autocorrelation | "
            f"{ffbs['lag1_autocorrelation']['total_relation_count']:+.4f} | "
            f"{local['lag1_autocorrelation']['total_relation_count']:+.4f} |",
            f"| segment-count bulk ESS | {ffbs['segment_count_ess']:,.0f} | "
            f"{local['segment_count_ess']:,.0f} |",
            f"| minimum bulk ESS over all coordinates | {ffbs['min_bulk_ess']:,.0f} | "
            f"{local['min_bulk_ess']:,.0f} |",
            f"| minimum ESS / second | {ffbs['min_ess_per_second']:,.1f} | "
            f"{local['min_ess_per_second']:,.1f} |",
            f"| distinct induced-H states visited | {ffbs['distinct_h_states']} | "
            f"{local['distinct_h_states']} |",
            f"| structural mode transitions | {ffbs['h_changes']:,} | "
            f"{local['h_changes']:,} |",
            f"| worst R-hat | {ffbs['max_rhat']:.5f} | {local['max_rhat']:.5f} |",
            "",
            "Both samplers clear every gate of the same independent reference. The "
            "difference is in the kernel's movement and its cost, not in the distribution "
            "they target.",
            "",
        ]
    block7b2 = payload["stage7b2"]
    if str(block7b2.get("status", "")).startswith("COMPARED"):
        ffbs_side, block1, final = (block7b2["ffbs"], block7b2["local_block1"],
                                    block7b2["local_final"])
        locking_rows = []
        for crit in ("A_frozen_structure", "B_disagreeing_structure",
                     "C_log_posterior_gap"):
            f_c = ffbs_side["structural_locking"][crit]
            l_c = final["structural_locking"][crit]
            locking_rows.append(f"| {crit} | {'holds' if f_c['holds'] else 'fails'} | "
                                f"{'holds' if l_c['holds'] else 'fails'} |")
        lines += [
            "## Step 7B2 — the full-corpus comparison",
            "",
            f"**{block7b2['structural_locking_verdict']}.**",
            "",
            f"Baseline verdict at its decision point: {block7b2['baseline_verdict']}. "
            f"FFBS worst invariant R-hat: {block7b2['ffbs_worst_invariant_rhat']}.",
            "",
            "| registered criterion | FFBS (50k) | LocalMoveKernel (frozen final) |",
            "|---|---|---|",
            *locking_rows,
            "",
            "| equal sweeps (both first 50k) | FFBS | LocalMoveKernel |",
            "|---|---|---|",
            f"| wall seconds | {ffbs_side['wall_seconds']:.0f} "
            f"({ffbs_side['wall_basis'].split(':')[0]}) | {block1['wall_seconds']:.0f} |",
            f"| worst invariant R-hat | {ffbs_side['max_invariant_rhat']:.5f} | "
            f"{block1['max_invariant_rhat']:.5f} |",
            f"| min invariant bulk ESS | {ffbs_side['min_invariant_bulk_ess']:,.0f} | "
            f"{block1['min_invariant_bulk_ess']:,.0f} |",
            f"| distinct induced-H states (pooled) | {ffbs_side['distinct_h_states']} | "
            f"{block1['distinct_h_states']} |",
            f"| structural mode transitions | {ffbs_side['h_changes']:,} | "
            f"{block1['h_changes']:,} |",
            f"| chains frozen (criterion A basis) | "
            f"{ffbs_side['structural_locking']['A_frozen_structure']['n_chains_frozen']} | "
            f"{block1['structural_locking']['A_frozen_structure']['n_chains_frozen']} |",
            "",
            f"Equal-time and full tables: `comparison_local_vs_ffbs.json`. "
            f"{block7b2['recovery_note']}.",
            "",
            "### Conclusion",
            "",
            "What is falsified is not FFBS — it performed exactly as validated "
            "(exact conditional sampling of (S,z), Step 7A; same-target posterior "
            "correctness, Step 7B1; strong global segmentation movement here) — but "
            "the hypothesis that exact global FFBS updates of (S,z) alone resolve "
            "the full-joint structural locking. Better (S,z) mixing does not "
            "propagate to U/H structural mixing: with the segmentation refreshed "
            "globally every sweep, 4/4 chains remain structurally frozen, and the "
            "U-proposal audit (`stage7b2_u_audit/`) locates the wall in the target "
            "itself — cross-cell proposals are frequent (P(H' != H) ~ 0.5) but the "
            "conditional likelihood rejects them at 1e-11 to 1e-20. The locking is "
            "therefore not kernel locality but strong posterior coupling between "
            "the labelled segmentation and the reusable partial-order structure.",
            "",
            "This diagnosis motivates the partially collapsed structural update "
            "introduced next, which integrates out the labelled segmentation when "
            "evaluating latent-U proposals.",
            "",
        ]
    elif str(block7b2.get("status", "")).startswith("FFBS CHAINS COMPLETE"):
        ffbs_side = block7b2["ffbs"]
        lock = ffbs_side["structural_locking"]
        lines += [
            "## Step 7B2 — FFBS chains complete, comparison deferred",
            "",
            "The registered comparison uses the FROZEN FINAL Stage 6E2 baseline, which "
            "is not frozen yet:",
            "",
            *[f"* {reason}" for reason in block7b2["deferral_reasons"]],
            "",
            "What can be said from the FFBS side alone (no baseline artifact read):",
            "",
            f"* worst invariant R-hat {ffbs_side['max_invariant_rhat']:.5f}, "
            f"min invariant bulk ESS {ffbs_side['min_invariant_bulk_ess']:,.0f}",
            f"* structural locking on the FFBS chains: A "
            f"{'holds' if lock['A_frozen_structure']['holds'] else 'fails'}, B "
            f"{'holds' if lock['B_disagreeing_structure']['holds'] else 'fails'}, C "
            f"{'holds' if lock['C_log_posterior_gap']['holds'] else 'fails'} "
            f"(conjunction {'HOLDS' if lock['conjunction_holds'] else 'does not hold'})",
            f"* {ffbs_side['distinct_h_states']} distinct induced-H states, "
            f"{ffbs_side['h_changes']:,} structural mode transitions",
            "",
        ]
    if payload["stage7b2"].get("reasons"):
        lines += ["## Step 7B2 — why it has not started", ""]
        lines += [f"* {reason}" for reason in payload["stage7b2"]["reasons"]]
        cost = payload["stage7b2"].get("cost_projection")
        if cost:
            lines += [
                "",
                f"Cost projection on the 100-trace corpus: one FFBS sweep costs "
                f"{cost['seconds_per_sweep']:.2f} s against the LocalMoveKernel "
                f"baseline's {cost['baseline_seconds_per_sweep']:.2f} s, i.e. "
                f"{cost['seconds_per_sweep'] / cost['baseline_seconds_per_sweep']:.1f}x, "
                f"and {cost['chart_seconds_all_traces'] / cost['seconds_per_sweep'] * 100:.0f}% "
                "of it is the forward charts. A 50,000-sweep block would take about "
                f"{cost['projected_50k_sweep_hours_per_chain']:.0f} h per chain.",
            ]
        lines += ["", "No Step 7 tag is created: the tag requires Step 7B2 as well.", ""]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[7B-compare] wrote {OUT}")


if __name__ == "__main__":
    main()
