"""Focused start[0] probe — transient/diagnostic artefact (H1) vs basin-specific
kernel interaction (H2) in the partially collapsed-U sampler.

    PYTHONPATH=src python scripts/collapsed_u_start0_probe.py

Four chains of the UNCHANGED kernel, all initialised from the identical registered
dispersed start[0] of the frozen Stage 6E1B mixed-reference problem, differing only in
their RNG seeds. Registration, the start-state manifest, the window definitions, the
extraction rules and the verdict rules are all written to disk BEFORE the first draw.

Nothing is tuned, no threshold moves, the frozen historical gate is reported unchanged
(and is not the probe's decision statistic — its first-4,000-row window is dominated by
chains 0-1 by construction), draws are never pooled with run 1 / rep2 for the primary
verdict, and the two historical formal failures remain failures whatever this finds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.collapsed_u_kernel import (                        # noqa: E402
    MOVE_NAME, CollapsedUConfig, is_collapsed_sweep, run_collapsed_u_chain,
)
from hpop.mcmc_original.latent_poset import precedence_from_u              # noqa: E402
from hpop.mcmc_original.stage6b_joint_diagnostics import (                 # noqa: E402
    calibrate_energy_envelope, energy_distance, standardise,
)
from hpop.mcmc_original.stage6b_mcmc_diagnostics import (                  # noqa: E402
    bulk_ess, rank_normalized_split_rhat, tail_ess,
)
from hpop.mcmc_original.stage6d_frozen import REGISTERED_SCALES            # noqa: E402
from hpop.mcmc_original.stage6e_frozen import (                            # noqa: E402
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH, config_hash,
)
from hpop.mcmc_original.stage6e_state import Stage6EModel, Stage6EState    # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "collapsed_u_start0_probe"
RUN1 = ROOT / "results" / "mcmc_original" / "collapsed_u_kernel_validation"
REP2 = ROOT / "results" / "mcmc_original" / "collapsed_u_kernel_validation_rep2"

# New deterministic seeds. Never used by 6E1B (6052001-4), 7B1 (7053001-4),
# 7B2 (7063201-4), any audit (8150000s/8151500/8152000s), run 1 (8153001-4),
# rep2 (8154001-4), or the reserved unlaunched Step 8 run (8160001-4).
CHAIN_SEEDS = (8_155_001, 8_155_002, 8_155_003, 8_155_004)
PREVIOUSLY_USED = {6052001, 6052002, 6052003, 6052004, 7053001, 7053002, 7053003,
                   7053004, 7063201, 7063202, 7063203, 7063204, 8150000, 8151500,
                   8152000, 8153001, 8153002, 8153003, 8153004, 8154001, 8154002,
                   8154003, 8154004, 8160001, 8160002, 8160003, 8160004}
RESUME_CHECK_SEED = 8_155_900

SWEEPS, BURN_IN, THIN = 600_000, 120_000, 10
COLLAPSED_EVERY = 10
N_CHAINS, N_RETAINED = 4, 48_000
CHECKPOINT_EVERY = 25_000

WINDOWS = {"early": (0, 16_000), "middle": (16_000, 32_000), "late": (32_000, 48_000)}
PER_CHAIN_WINDOW_DRAWS = 2_000        # stride 8, offset 0 — registered, never re-chosen
BALANCED_PER_CHAIN = 1_000            # stride 16, offset 0 — registered
ENVELOPE_SEED, ENVELOPE_REPS = 5, 40  # the frozen diagnostic's own calibration
FROZEN_GATE_ENVELOPE = 0.004522256615497353
SCALARS = ("rho", "beta", "omega", "lambda_rep", "lambda_back")
PRIMARY = ("beta", "lambda_rep")
CONTROLS = ("rho", "omega", "lambda_back")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_state(state: Stage6EState) -> str:
    payload = json.dumps(_jsonable(state.to_dict()), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _jsonable(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, bytes):
        return v.hex()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def problem():
    """The frozen 6E1B problem, its model, and the registered start[0]."""
    e1b = _load("stage6e1b", ROOT / "scripts" / "stage6e1b_mixed_reference.py")
    traces, _ = e1b.generate_corpus()
    mixed = e1b.build_mixed_model(traces)
    model = Stage6EModel(traces=traces, epsilon=e1b.EPSILON, delta_b=DELTA_B,
                         n_skills=e1b.K_SKILLS, n_roles=e1b.M_ROLES,
                         min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
                         infer_pi_P=False)
    start0 = e1b.dispersed_starts(mixed)[0]
    return e1b, mixed, model, start0


# ------------------------------------------------------------------ reference machinery
class Reference:
    """The frozen reference rows in the EXACT frozen-gate coordinates
    [closure indicators (K x m x m), per-trace segment counts, scalars]."""

    def __init__(self, e1b, mixed):
        ref = np.load(RUN1 / "reference_draws.npz", allow_pickle=False)
        closures = ref["retained_closures"].reshape(
            ref["retained_closures"].shape[0], -1).astype(float)
        counts = np.array([[len(mixed.states[j]) for j in row]
                           for row in ref["retained_sampled"]], dtype=float)
        scalars = np.column_stack([ref[f"retained_{n}"] for n in SCALARS])
        rows = np.column_stack([closures, counts, scalars])
        self.spread = rows.std(axis=0, ddof=1)
        self.keep = self.spread > 1e-12
        kept = rows[:, self.keep]
        self.centre, self.scale = kept.mean(0), kept.std(0, ddof=1)
        self.A = standardise(kept, self.centre, self.scale)
        self.scalar_mean = {n: float(ref[f"retained_{n}"].mean()) for n in SCALARS}
        self.scalar_sd = {n: float(ref[f"retained_{n}"].std(ddof=1)) for n in SCALARS}
        self._envelopes: dict = {}

    def rows_of(self, u_draws, segment_counts, scalars) -> np.ndarray:
        n = u_draws.shape[0]
        closures = np.array([[precedence_from_u(u_draws[i, k]).reshape(-1)
                              for k in range(u_draws.shape[1])]
                             for i in range(n)]).reshape(n, -1).astype(float)
        rows = np.column_stack([closures, segment_counts.astype(float),
                                np.column_stack([scalars[k] for k in SCALARS])])
        return standardise(rows[:, self.keep], self.centre, self.scale)

    def energy_z(self, b: np.ndarray) -> dict:
        n_x = min(len(self.A) // 2, len(b))
        n_y = min(len(self.A) - n_x, len(b))
        key = (n_x, n_y)
        if key not in self._envelopes:
            self._envelopes[key] = calibrate_energy_envelope(
                self.A, n_x=n_x, n_y=n_y, n_replicates=ENVELOPE_REPS,
                seed=ENVELOPE_SEED)
        env = self._envelopes[key]
        obs = float(energy_distance(self.A[:n_x * 2:2], b[:n_y]))
        return {"observed": obs, "envelope": float(env["envelope"]),
                "null_mean": float(env["mean"]), "null_sd": float(env["sd"]),
                "z": float((obs - env["mean"]) / max(1e-12, env["sd"])),
                "inside_envelope": bool(obs <= env["envelope"]),
                "n_x": int(n_x), "n_y": int(n_y)}


# ---------------------------------------------------------------------------- workers
def _chain_worker(payload: dict) -> dict:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "src"))
    _, _, model, start0 = problem()
    chain = int(payload["chain"])
    if sha256_state(start0) != payload["start_hash"]:
        raise AssertionError(f"chain {chain}: rebuilt start[0] hash mismatch")

    began = time.perf_counter()
    result = run_collapsed_u_chain(
        model=model, start=start0, scales=REGISTERED_SCALES,
        num_sweeps=SWEEPS, burn_in=BURN_IN, thin=THIN, seed=CHAIN_SEEDS[chain],
        collapsed=CollapsedUConfig(every=COLLAPSED_EVERY,
                                   scale=float(REGISTERED_SCALES["U"])),
        chain=chain, table_source="batched", store_labels=False, store_keys=False,
        progress_every=payload.get("progress_every", 0),
        checkpoint_path=payload["checkpoint_path"],
        checkpoint_every=CHECKPOINT_EVERY)
    return {
        "chain": chain, "seed": CHAIN_SEEDS[chain],
        "start_hash": payload["start_hash"],
        "u_draws": result.u_draws,
        "scalars": {n: result.scalars[n] for n in SCALARS},
        "segment_counts": result.segment_counts,
        "relation_counts": result.relation_counts,
        "log_target": result.log_target,
        "collapsed_records": [dict(r) for r in result.collapsed_records],
        "proposed": result.proposed, "accepted": result.accepted,
        "invalid": result.invalid,
        "final_state_hash": sha256_state(result.final_state),
        "runtime_seconds": time.perf_counter() - began,
    }


def resume_determinism_check(model, start0) -> dict:
    """Uninterrupted 2,000 sweeps == 1,000 + checkpoint + resume, on THIS problem."""
    import tempfile
    kwargs = dict(model=model, scales=REGISTERED_SCALES, burn_in=0, thin=1,
                  seed=RESUME_CHECK_SEED,
                  collapsed=CollapsedUConfig(every=COLLAPSED_EVERY,
                                             scale=float(REGISTERED_SCALES["U"])),
                  store_labels=False)
    full = run_collapsed_u_chain(start=start0.copy(), num_sweeps=2_000, **kwargs)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_collapsed_u_chain(start=start0.copy(), num_sweeps=1_000,
                              checkpoint_path=tmp, checkpoint_every=1_000, **kwargs)
        payload = json.loads((tmp / "chain0_checkpoint.json").read_text())
        restored = Stage6EState.from_dict(payload["state"])
        rng = np.random.default_rng(RESUME_CHECK_SEED)
        rng.bit_generator.state = restored.rng_state
        part2 = run_collapsed_u_chain(start=restored, num_sweeps=2_000, rng=rng,
                                      state=restored, **kwargs)
    a, b = full.final_state.to_dict(), part2.final_state.to_dict()
    a.pop("cache_version"), b.pop("cache_version")
    identical = bool(a == b and np.array_equal(full.u_draws[1000:], part2.u_draws))
    return {"sweeps": 2000, "checkpoint_at": 1000, "seed": RESUME_CHECK_SEED,
            "bit_identical": identical, "pass": identical}


# ------------------------------------------------------------------------ registration
def write_registration(e1b, model, start0, seven_b1) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    reference_audit = seven_b1.verify_frozen_reference()
    if not reference_audit["pass"]:
        raise SystemExit("frozen reference failed verification")

    kernel_now = sha256_file(ROOT / "src/hpop/mcmc_original/collapsed_u_kernel.py")
    manifest1 = json.loads((RUN1 / "implementation_manifest.json").read_text())
    registration = {
        "purpose": "distinguish H1 (start[0] transient amplified by the frozen "
                   "unbalanced energy window) from H2 (basin-specific collapsed-U / "
                   "scalar interaction)",
        "parent_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "stage6e_config_hash": config_hash(),
        "target_script_sha256": sha256_file(ROOT / "scripts"
                                            / "stage6e1b_mixed_reference.py"),
        "kernel_hashes": {
            "collapsed_u_kernel.py": kernel_now,
            "collapsed_u_kernel_at_formal_runs": manifest1["new_files"][
                "src/hpop/mcmc_original/collapsed_u_kernel.py"],
            "collapsed_u_kernel_changed_since_formal_runs": bool(
                kernel_now != manifest1["new_files"][
                    "src/hpop/mcmc_original/collapsed_u_kernel.py"]),
            "change_disclosure": "the only change is an optional movement_tracker "
                                 "parameter added to run_collapsed_u_chain after rep2 "
                                 "launched (None in this probe and in both formal "
                                 "runs' call pattern); collapsed_u_mh_step, "
                                 "collapsed_ffbs_sweep_once and the likelihood are "
                                 "untouched; every=0 bitwise parity and resume "
                                 "determinism tests pass against the current source",
            "collapsed_u_likelihood.py": sha256_file(
                ROOT / "src/hpop/mcmc_original/collapsed_u_likelihood.py"),
            "collapsed_u_likelihood_matches_formal_runs": bool(
                sha256_file(ROOT / "src/hpop/mcmc_original/collapsed_u_likelihood.py")
                == manifest1["new_files"][
                    "src/hpop/mcmc_original/collapsed_u_likelihood.py"]),
            "semi_markov_ffbs.py": sha256_file(
                ROOT / "src/hpop/mcmc_original/semi_markov_ffbs.py"),
            "sampler_u.py": sha256_file(ROOT / "src/hpop/mcmc_original/sampler_u.py"),
        },
        "frozen_reference_audit": {"pass": reference_audit["pass"],
                                   "max_drift": reference_audit[
                                       "max_drift_from_expected"]},
        "prior_run_hashes": {
            "run1_joint_comparison": sha256_file(RUN1 / "joint_comparison.json"),
            "run1_chains": sha256_file(RUN1 / "chains.npz"),
            "rep2_joint_comparison": sha256_file(REP2 / "joint_comparison.json"),
            "rep2_chains": sha256_file(REP2 / "chains.npz")},
        "start0_hash": sha256_state(start0),
        "chain_seeds": list(CHAIN_SEEDS),
        "seeds_never_used_before": bool(not set(CHAIN_SEEDS) & PREVIOUSLY_USED),
        "config": {"n_chains": N_CHAINS, "sweeps": SWEEPS, "burn_in": BURN_IN,
                   "thin": THIN, "retained_per_chain": N_RETAINED,
                   "collapsed_u_every": COLLAPSED_EVERY,
                   "proposal_scales": dict(REGISTERED_SCALES),
                   "checkpoint_every": CHECKPOINT_EVERY,
                   "update_order": "collapsed U MH -> exact FFBS refresh of all "
                                   "(S,z) -> (pi/P fixed) -> conditional U rows -> "
                                   "rho -> beta -> omega -> lambda_rep -> "
                                   "lambda_back"},
        "windows_registered": {k: list(v) for k, v in WINDOWS.items()},
        "extraction_rules_registered": {
            "per_chain_energy": f"{PER_CHAIN_WINDOW_DRAWS} equally spaced draws per "
                                "16,000-draw window: stride 8, offset 0",
            "chain_balanced_energy": f"{BALANCED_PER_CHAIN} equally spaced draws per "
                                     "chain per window (stride 16, offset 0), "
                                     "concatenated to 4,000 with equal chain "
                                     "contribution",
            "coordinates": "the EXACT frozen-gate coordinates [closure indicators, "
                           "per-trace segment counts, scalars]; the "
                           "gate_failure_diagnosis per-chain numbers used a "
                           "total-count approximation and are reported alongside for "
                           "comparability, not for the verdict",
            "envelope": f"calibrate_energy_envelope, seed {ENVELOPE_SEED}, "
                        f"{ENVELOPE_REPS} replicates — the frozen diagnostic's own",
            "scalar_offsets": "offset_sd = (window mean - reference mean) / reference "
                              "SD; offset_se = offset_sd * sqrt(window bulk ESS); "
                              "verdict thresholds of 2 apply to offset_se; "
                              "early-to-late contraction judged on median |offset_sd| "
                              "across chains",
            "tail_ess": "project tail_ess (5%/95% indicator ESS)"},
        "verdict_rules_preregistered": {
            "A_transient": ["late chain-balanced statistic inside its envelope",
                            "at most 1 of 4 late per-chain z > +2",
                            "median late per-chain z <= +1.5",
                            ">=3/4 chains late |beta offset_se| < 2",
                            ">=3/4 chains late |lambda_rep offset_se| < 2",
                            "median |beta offset_sd| decreases early->late",
                            "median |lambda_rep offset_sd| decreases early->late",
                            "no control scalar with >=3/4 same-direction late "
                            "|offset_se| > 2",
                            "late ESS finite and structure still moving"],
            "B_basin_interaction_any_of": [
                ">=3/4 late per-chain z > +2 AND their median > +2",
                "late chain-balanced statistic outside its envelope",
                ">=3/4 chains same-direction late beta offset_se magnitude > 2",
                ">=3/4 chains same-direction late lambda_rep offset_se magnitude > 2",
                "median |offset_sd| does not shrink early->middle->late "
                "(late >= early) for beta or lambda_rep",
                "joint same-direction beta+lambda_rep displacement (both "
                "offset_se magnitude > 2) in >=3/4 chains with clean controls"],
            "C": "neither A nor B -> START-0 FOCUSED PROBE INCONCLUSIVE",
            "note": "the historical frozen gate is reported unchanged and is NOT the "
                    "decision statistic; run 1 and rep2 remain recorded failures "
                    "regardless of this probe"},
    }
    (OUT / "registration.json").write_text(json.dumps(_jsonable(registration),
                                                      indent=2))
    (OUT / "window_definitions.json").write_text(json.dumps({
        "windows_by_retained_index": {k: list(v) for k, v in WINDOWS.items()},
        "per_chain_stride": 8, "balanced_stride": 16, "offsets": 0,
        "registered_before_any_draw": True}, indent=2))
    (OUT / "start_state_manifest.json").write_text(json.dumps(_jsonable({
        "start0_hash": sha256_state(start0),
        "state": start0.to_dict(),
        "identical_for_all_chains": True,
        "chains_differ_only_in": "RNG seed",
        "seeds": list(CHAIN_SEEDS)}), indent=2))
    return registration


# ---------------------------------------------------------------------------- analysis
def window_slice(arr, window, stride):
    lo, hi = WINDOWS[window]
    return arr[lo:hi:stride]


def save_raw(payloads) -> None:
    """Everything the analysis needs, written to disk BEFORE any analysis runs, so a
    late failure can never lose four hours of chains. `--analyze-only` reloads this."""
    arrays = {
        "u_draws": np.array([p["u_draws"] for p in payloads]),
        "segment_counts": np.array([p["segment_counts"] for p in payloads]),
        "relation_counts": np.array([p["relation_counts"] for p in payloads]),
        "log_target": np.array([p["log_target"] for p in payloads]),
        "chain_seeds": np.array([p["seed"] for p in payloads]),
        "runtime_seconds": np.array([p["runtime_seconds"] for p in payloads]),
        **{f"scalar_{n}": np.array([p["scalars"][n] for p in payloads])
           for n in SCALARS}}
    for p in payloads:
        c = p["chain"]
        records = p["collapsed_records"]
        for field, dtype in (("sweep", np.int64), ("accepted", bool),
                             ("h_changed", bool), ("invalid", bool)):
            arrays[f"collapsed_{field}_c{c}"] = np.array(
                [r[field] for r in records], dtype=dtype)
    np.savez_compressed(OUT / "chains.npz", **arrays)
    (OUT / "run_meta.json").write_text(json.dumps(_jsonable({
        "counters": {str(p["chain"]): {
            "proposed": p["proposed"].get(MOVE_NAME, 0),
            "accepted": p["accepted"].get(MOVE_NAME, 0),
            "invalid": p["invalid"].get(MOVE_NAME, 0)} for p in payloads},
        "start_hashes": {str(p["chain"]): p["start_hash"] for p in payloads},
        "final_state_hashes": {str(p["chain"]): p["final_state_hash"]
                               for p in payloads},
        "runtime_seconds": [p["runtime_seconds"] for p in payloads]}), indent=2))
    (OUT / "checkpoint_history.json").write_text(json.dumps({
        "checkpoint_every": CHECKPOINT_EVERY,
        "final_state_hashes": {str(p["chain"]): p["final_state_hash"]
                               for p in payloads},
        "runtime_seconds": [p["runtime_seconds"] for p in payloads]}, indent=2))


def load_raw() -> list:
    z = np.load(OUT / "chains.npz", allow_pickle=False)
    meta = json.loads((OUT / "run_meta.json").read_text())
    payloads = []
    for c in range(N_CHAINS):
        records = [{"sweep": int(s), "accepted": bool(a), "h_changed": bool(h),
                    "invalid": bool(i)}
                   for s, a, h, i in zip(z[f"collapsed_sweep_c{c}"],
                                         z[f"collapsed_accepted_c{c}"],
                                         z[f"collapsed_h_changed_c{c}"],
                                         z[f"collapsed_invalid_c{c}"])]
        payloads.append({
            "chain": c, "seed": int(z["chain_seeds"][c]),
            "u_draws": z["u_draws"][c],
            "scalars": {n: z[f"scalar_{n}"][c] for n in SCALARS},
            "segment_counts": z["segment_counts"][c],
            "relation_counts": z["relation_counts"][c],
            "log_target": z["log_target"][c],
            "collapsed_records": records,
            "proposed": {MOVE_NAME: meta["counters"][str(c)]["proposed"]},
            "accepted": {MOVE_NAME: meta["counters"][str(c)]["accepted"]},
            "invalid": {MOVE_NAME: meta["counters"][str(c)]["invalid"]},
            "start_hash": meta["start_hashes"][str(c)],
            "final_state_hash": meta["final_state_hashes"][str(c)],
            "runtime_seconds": meta["runtime_seconds"][c]})
    return payloads


def analyze(payloads, reference: Reference, registration: dict,
            resume_check: dict) -> None:
    chain_rows = {}
    for p in payloads:
        chain_rows[p["chain"]] = reference.rows_of(
            p["u_draws"], p["segment_counts"], p["scalars"])

    # ---- historical frozen gate, exactly as compare() computes it ------------------
    pooled = np.concatenate([chain_rows[c] for c in range(N_CHAINS)], axis=0)
    thinned = pooled[::max(1, len(pooled) // len(reference.A))]
    hist = reference.energy_z(thinned)     # energy_z takes b[:n_y] = first 4000 rows
    hist["label"] = "historical frozen gate (first-4000-row pooled window; "
    hist["label"] += "dominated by chains 0-1 by construction)"
    hist["frozen_envelope_expected"] = FROZEN_GATE_ENVELOPE
    hist["frozen_envelope_matches"] = bool(
        abs(hist["envelope"] - FROZEN_GATE_ENVELOPE) < 1e-12)
    hist["pass_under_frozen_gate"] = bool(hist["observed"] <= FROZEN_GATE_ENVELOPE)
    (OUT / "historical_gate.json").write_text(json.dumps(_jsonable(hist), indent=2))

    # ---- per-chain, per-window energy ----------------------------------------------
    per_chain = {w: {} for w in WINDOWS}
    for w in WINDOWS:
        for c in range(N_CHAINS):
            b = window_slice(chain_rows[c], w, 8)[:PER_CHAIN_WINDOW_DRAWS]
            per_chain[w][str(c)] = reference.energy_z(b)
    (OUT / "per_chain_energy.json").write_text(json.dumps(_jsonable(per_chain),
                                                          indent=2))

    # ---- chain-balanced, per-window energy -----------------------------------------
    balanced = {}
    for w in WINDOWS:
        b = np.concatenate([window_slice(chain_rows[c], w, 16)[:BALANCED_PER_CHAIN]
                            for c in range(N_CHAINS)], axis=0)
        balanced[w] = reference.energy_z(b)
        balanced[w]["per_chain_contribution"] = BALANCED_PER_CHAIN
    (OUT / "chain_balanced_energy.json").write_text(json.dumps(_jsonable(balanced),
                                                               indent=2))

    # ---- scalar drift ---------------------------------------------------------------
    drift = {}
    for name in SCALARS:
        drift[name] = {}
        for c in range(N_CHAINS):
            series = payloads[c]["scalars"][name]
            rows = {}
            for w in WINDOWS:
                lo, hi = WINDOWS[w]
                window = series[lo:hi]
                ess = float(bulk_ess(window[None, :]))
                offset_sd = (float(window.mean()) - reference.scalar_mean[name]) \
                    / reference.scalar_sd[name]
                rows[w] = {"mean": float(window.mean()),
                           "offset_sd": offset_sd,
                           "offset_se": offset_sd * math.sqrt(max(1.0, ess)),
                           "bulk_ess": ess,
                           "tail_ess": float(tail_ess(window[None, :]))}
            rows["early_to_late_change_sd"] = (rows["late"]["offset_sd"]
                                               - rows["early"]["offset_sd"])
            rows["moves_toward_zero"] = bool(abs(rows["late"]["offset_sd"])
                                             < abs(rows["early"]["offset_sd"]))
            drift[name][str(c)] = rows
        drift[name]["median_abs_offset_sd"] = {
            w: float(np.median([abs(drift[name][str(c)][w]["offset_sd"])
                                for c in range(N_CHAINS)])) for w in WINDOWS}
    (OUT / "scalar_drift.json").write_text(json.dumps(_jsonable(drift), indent=2))

    # ---- structural movement and collapsed acceptance -------------------------------
    movement = {}
    for c in range(N_CHAINS):
        p = payloads[c]
        u = p["u_draws"]
        keys = [tuple(precedence_from_u(u[i, k]).tobytes()
                      for k in range(u.shape[1])) for i in range(u.shape[0])]
        rel = p["relation_counts"].sum(axis=1).astype(float)
        rows = {}
        for w in WINDOWS:
            lo, hi = WINDOWS[w]
            wk = keys[lo:hi]
            changes = sum(1 for a, b in zip(wk[:-1], wk[1:]) if a != b)
            recs = [r for r in p["collapsed_records"]
                    if lo <= (r["sweep"] - BURN_IN) // THIN < hi]
            rows[w] = {
                "distinct_h_states": len(set(wk)),
                "h_changes_between_retained": changes,
                "relation_count_sd": float(rel[lo:hi].std()),
                "collapsed_attempts": len(recs),
                "collapsed_accepted": sum(1 for r in recs if r["accepted"]),
                "collapsed_cross_h": sum(1 for r in recs if r["h_changed"]),
                "collapsed_accepted_cross_h": sum(
                    1 for r in recs if r["accepted"] and r["h_changed"]),
            }
        movement[str(c)] = rows
    (OUT / "structural_movement.json").write_text(json.dumps(_jsonable(movement),
                                                             indent=2))

    # ---- convergence (caveat: same start, so R-hat is not a global proof) -----------
    convergence = {"caveat": "all four chains share start[0]; R-hat here measures "
                             "seed-to-seed agreement, not global posterior coverage"}
    for name in SCALARS:
        stacked = np.stack([payloads[c]["scalars"][name] for c in range(N_CHAINS)])
        convergence[name] = {**rank_normalized_split_rhat(stacked),
                             "bulk_ess": float(bulk_ess(stacked)),
                             "tail_ess": float(tail_ess(stacked))}
    log_t = np.stack([payloads[c]["log_target"] for c in range(N_CHAINS)])
    convergence["log_posterior"] = {**rank_normalized_split_rhat(log_t),
                                    "bulk_ess": float(bulk_ess(log_t)),
                                    "mean": float(log_t.mean()),
                                    "q05": float(np.quantile(log_t, 0.05)),
                                    "q95": float(np.quantile(log_t, 0.95))}
    rel = np.stack([payloads[c]["relation_counts"].sum(axis=1)
                    for c in range(N_CHAINS)]).astype(float)
    convergence["relation_count"] = {**rank_normalized_split_rhat(rel),
                                     "bulk_ess": float(bulk_ess(rel))}
    (OUT / "convergence.json").write_text(json.dumps(_jsonable(convergence), indent=2))

    # ---- the pre-registered verdict -------------------------------------------------
    late_z = [per_chain["late"][str(c)]["z"] for c in range(N_CHAINS)]
    late = {name: [drift[name][str(c)]["late"] for c in range(N_CHAINS)]
            for name in SCALARS}
    med = {name: drift[name]["median_abs_offset_sd"] for name in SCALARS}

    def control_violation(name):
        big = [c for c in range(N_CHAINS) if abs(late[name][c]["offset_se"]) > 2]
        same_sign = (len(big) >= 3 and len({np.sign(late[name][c]["offset_se"])
                                            for c in big}) == 1)
        return bool(same_sign)

    a_conditions = {
        "late_balanced_inside_envelope": bool(balanced["late"]["inside_envelope"]),
        "at_most_one_late_z_above_2": bool(sum(z > 2 for z in late_z) <= 1),
        "median_late_z_leq_1p5": bool(float(np.median(late_z)) <= 1.5),
        "beta_3of4_late_offset_se_below_2": bool(sum(
            abs(late["beta"][c]["offset_se"]) < 2 for c in range(N_CHAINS)) >= 3),
        "lambda_rep_3of4_late_offset_se_below_2": bool(sum(
            abs(late["lambda_rep"][c]["offset_se"]) < 2
            for c in range(N_CHAINS)) >= 3),
        "beta_median_abs_contracts": bool(med["beta"]["late"] < med["beta"]["early"]),
        "lambda_rep_median_abs_contracts": bool(
            med["lambda_rep"]["late"] < med["lambda_rep"]["early"]),
        "controls_clean": bool(not any(control_violation(n) for n in CONTROLS)),
        "late_ess_finite_and_moving": bool(
            all(np.isfinite(late[n][c]["bulk_ess"]) and late[n][c]["bulk_ess"] > 0
                for n in PRIMARY for c in range(N_CHAINS))
            and all(movement[str(c)]["late"]["h_changes_between_retained"] > 0
                    for c in range(N_CHAINS))),
    }

    def same_direction_big(name):
        big = [c for c in range(N_CHAINS) if abs(late[name][c]["offset_se"]) > 2]
        return bool(len(big) >= 3 and len({np.sign(late[name][c]["offset_se"])
                                           for c in big}) == 1)

    joint_cells = [c for c in range(N_CHAINS)
                   if abs(late["beta"][c]["offset_se"]) > 2
                   and abs(late["lambda_rep"][c]["offset_se"]) > 2]
    joint_same = (len(joint_cells) >= 3
                  and len({(np.sign(late["beta"][c]["offset_se"]),
                            np.sign(late["lambda_rep"][c]["offset_se"]))
                           for c in joint_cells}) == 1)
    b_conditions = {
        "three_late_z_above_2_with_median_above_2": bool(
            sum(z > 2 for z in late_z) >= 3 and float(np.median(late_z)) > 2),
        "late_balanced_outside_envelope": bool(
            not balanced["late"]["inside_envelope"]),
        "beta_3of4_same_direction_above_2": same_direction_big("beta"),
        "lambda_rep_3of4_same_direction_above_2": same_direction_big("lambda_rep"),
        "no_shrink_early_to_late": bool(
            med["beta"]["late"] >= med["beta"]["early"]
            or med["lambda_rep"]["late"] >= med["lambda_rep"]["early"]),
        "joint_beta_lambda_rep_displacement_with_clean_controls": bool(
            joint_same and not any(control_violation(n) for n in CONTROLS)),
    }

    a_supported = all(a_conditions.values())
    b_supported = any(b_conditions.values())
    if a_supported and not b_supported:
        verdict = "START-0 TRANSIENT / DIAGNOSTIC-WINDOW EXPLANATION SUPPORTED"
    elif b_supported and not a_supported:
        verdict = "START-0 BASIN-SPECIFIC KERNEL INTERACTION SUPPORTED"
    else:
        verdict = "START-0 FOCUSED PROBE INCONCLUSIVE"

    verdict_payload = {
        "verdict": verdict,
        "a_conditions": a_conditions, "a_supported": bool(a_supported),
        "b_conditions": b_conditions, "b_supported": bool(b_supported),
        "late_per_chain_z": late_z,
        "late_balanced": balanced["late"],
        "historical_gate_pass": hist["pass_under_frozen_gate"],
        "note": "run 1 and rep2 remain recorded failures; this probe does not "
                "convert them into passes",
        "recommended_next_if_A": "one final dispersed-start mixed-reference "
                                 "validation with a prospectively frozen "
                                 "chain-balanced, dependence-aware, full-window "
                                 "multivariate diagnostic (NOT launched here)",
        "recommended_next_if_B": "D0-D4 scalar-release decomposition (collapsed U + "
                                 "FFBS with all scalars fixed; release beta only; "
                                 "lambda_rep only; both; then the rest) to localise "
                                 "the interaction (NOT launched here)",
    }
    (OUT / "verdict.json").write_text(json.dumps(_jsonable(verdict_payload), indent=2))

    correctness = {
        "registration": {k: registration["kernel_hashes"][k]
                         for k in registration["kernel_hashes"]},
        "cadence_check": bool([i for i in range(40) if is_collapsed_sweep(
            i, COLLAPSED_EVERY)] == [9, 19, 29, 39]),
        "all_chains_identical_start": bool(len({p["start_hash"]
                                                for p in payloads}) == 1),
        "chains_differ_only_in_seeds": [p["seed"] for p in payloads],
        "retained_counts": [len(p["log_target"]) for p in payloads],
        "resume_determinism": resume_check,
        "collapsed_counters": {str(p["chain"]): {
            "proposed": p["proposed"].get(MOVE_NAME, 0),
            "accepted": p["accepted"].get(MOVE_NAME, 0),
            "invalid": p["invalid"].get(MOVE_NAME, 0)} for p in payloads},
    }
    (OUT / "correctness.json").write_text(json.dumps(_jsonable(correctness), indent=2))

    write_report(hist, per_chain, balanced, drift, movement, convergence,
                 verdict_payload)
    print(f"[start0-probe] verdict: {verdict}")


def write_report(hist, per_chain, balanced, drift, movement, convergence,
                 verdict_payload) -> None:
    lines = ["# Collapsed-U start[0] focused probe", "",
             f"**{verdict_payload['verdict']}**", "",
             "4 chains, identical registered start[0], seeds "
             f"{list(CHAIN_SEEDS)}, {SWEEPS:,} sweeps each, collapsed move every "
             f"{COLLAPSED_EVERY}. Windows and verdict rules frozen before launch "
             "(registration.json). Run 1 and rep2 remain recorded failures.", "",
             "## Historical frozen gate (unchanged, NOT the decision statistic)",
             f"observed {hist['observed']:.6f} vs frozen envelope "
             f"{FROZEN_GATE_ENVELOPE:.6f} -> "
             f"{'PASS' if hist['pass_under_frozen_gate'] else 'FAIL'}", "",
             "## Per-chain energy z by window",
             "| chain | early | middle | late |", "|---|---|---|---|"]
    for c in range(N_CHAINS):
        lines.append("| %d | %+.2f | %+.2f | %+.2f |" % (
            c, per_chain["early"][str(c)]["z"], per_chain["middle"][str(c)]["z"],
            per_chain["late"][str(c)]["z"]))
    lines += ["", "## Chain-balanced energy (prospective)",
              "| window | observed | envelope | z | inside |", "|---|---|---|---|---|"]
    for w in WINDOWS:
        b = balanced[w]
        lines.append("| %s | %.6f | %.6f | %+.2f | %s |" % (
            w, b["observed"], b["envelope"], b["z"], b["inside_envelope"]))
    lines += ["", "## beta / lambda_rep drift (offset in reference SD; SE-standardised "
              "in scalar_drift.json)",
              "| chain | beta early | beta late | l_rep early | l_rep late |",
              "|---|---|---|---|---|"]
    for c in range(N_CHAINS):
        lines.append("| %d | %+.3f | %+.3f | %+.3f | %+.3f |" % (
            c, drift["beta"][str(c)]["early"]["offset_sd"],
            drift["beta"][str(c)]["late"]["offset_sd"],
            drift["lambda_rep"][str(c)]["early"]["offset_sd"],
            drift["lambda_rep"][str(c)]["late"]["offset_sd"]))
    lines += ["", "## Verdict conditions", "```json",
              json.dumps({"A": verdict_payload["a_conditions"],
                          "B": verdict_payload["b_conditions"]}, indent=2), "```", "",
              f"Recommended next (NOT launched): "
              f"{verdict_payload['recommended_next_if_A'] if verdict_payload['a_supported'] else verdict_payload['recommended_next_if_B']}"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--analyze-only", action="store_true",
                        help="re-run the analysis from the saved chains.npz")
    args = parser.parse_args()

    e1b, mixed, model, start0 = problem()
    seven_b1 = _load("stage7b1", ROOT / "scripts" / "stage7b1_mixed_reference_mcmc.py")

    if args.analyze_only:
        registration = json.loads((OUT / "registration.json").read_text())
        resume_check = json.loads((OUT / "correctness.json").read_text()).get(
            "resume_determinism") if (OUT / "correctness.json").exists() else {
            "pass": True, "note": "recorded at launch; see registration"}
        payloads = load_raw()
        analyze(payloads, Reference(e1b, mixed), registration, resume_check)
        print(f"[start0-probe] re-analysed {OUT}")
        return

    registration = write_registration(e1b, model, start0, seven_b1)
    if not registration["seeds_never_used_before"]:
        raise SystemExit("seed collision with a prior run")
    print(f"[start0-probe] registered: start0 {registration['start0_hash'][:12]}, "
          f"seeds {list(CHAIN_SEEDS)}", flush=True)

    resume_check = resume_determinism_check(model, start0)
    if not resume_check["pass"]:
        raise SystemExit("resume determinism check FAILED — not launching")
    print("[start0-probe] resume determinism: PASS", flush=True)

    from multiprocessing import get_context
    (OUT / "checkpoints").mkdir(exist_ok=True)
    jobs = [{"chain": c, "start_hash": registration["start0_hash"],
             "checkpoint_path": str(OUT / "checkpoints"),
             "progress_every": args.progress_every} for c in range(N_CHAINS)]
    print(f"[start0-probe] {N_CHAINS} chains x {SWEEPS:,} sweeps from the identical "
          "start[0], in parallel", flush=True)
    began = time.perf_counter()
    with get_context("spawn").Pool(processes=N_CHAINS) as pool:
        payloads = pool.map(_chain_worker, jobs)
    payloads.sort(key=lambda p: p["chain"])
    print(f"[start0-probe] chains done in {(time.perf_counter() - began) / 3600:.2f} h",
          flush=True)
    for p in payloads:
        if len(p["log_target"]) != N_RETAINED:
            raise SystemExit(f"chain {p['chain']}: retained "
                             f"{len(p['log_target'])} != {N_RETAINED}")
    save_raw(payloads)
    print("[start0-probe] raw draws persisted; analysing", flush=True)

    analyze(payloads, Reference(e1b, mixed), registration, resume_check)
    print(f"[start0-probe] wrote {OUT}")


if __name__ == "__main__":
    main()
