"""Stage 6D — the joint oracle-block MCMC runs.

    --stage 6d0 --mode smoke   joint smoke on the full corpus (all six coordinates)
    --stage 6d1                small-model chains, compared with the frozen QMC reference
    --stage 6d2                full oracle-block synthetic joint recovery

The frozen reference is read, never rebuilt here, and never adjusted after a chain has
been seen.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_joint_scalar_mcmc import vectorized_state_features
from hpop.mcmc_original.recurrent_latent_poset_mcmc import LatentPosetEvaluator
from hpop.mcmc_original.recurrent_oracle_joint_mcmc import (
    Stage6DState, Stage6DTarget, initial_state, log_target_6d, run_oracle_joint_mcmc,
    sweep_once,
)
from hpop.mcmc_original.recurrent_scalar_posterior import (
    PRIORS, cached_batch_log_likelihood,
)
from hpop.mcmc_original.stage6d_diagnostics import (
    column_permutation_audit, dependence_block, h_labels_from_u, mixed_envelope,
    scalar_block, structural_block,
)
from hpop.mcmc_original.stage6d_frozen import (
    ACTIVE_6D, REGISTERED_SCALES, SCALAR_ORDER, config_hash, load_stage6d_dataset,
)
from hpop.mcmc_original.stage6d_joint_reference import small_model

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results" / "mcmc_original"

# ------------------------------------------------------------- registered run settings
SETTINGS = {
    "6d1": {"chains": 4, "sweeps": 50_000, "burn_in": 10_000, "thin": 4},
    "6d2": {"chains": 4, "sweeps": 30_000, "burn_in": 10_000, "thin": 5},
}
BASE_SEED = 0

# Dispersed scalar starts: prior quantiles arranged so all four chains differ in every
# coordinate, following the Stage 6B2/6B3 Latin-square convention.
QUANTILES = (0.10, 0.35, 0.65, 0.90)
LATIN = {"beta": (0, 1, 2, 3), "omega": (1, 2, 3, 0),
         "lambda_rep": (2, 3, 0, 1), "lambda_back": (3, 0, 1, 2)}
RHO_STARTS = (0.05, 0.30, 0.60, 0.90)


def prior_quantile(name: str, q: float) -> float:
    spec = PRIORS[name]
    if spec["family"] == "gamma":
        return float(stats.gamma.ppf(q, a=spec["shape"], scale=1.0 / spec["rate"]))
    return float(stats.norm.ppf(q, loc=spec["mean"], scale=spec["sd"]))


def u_starts(m: int) -> dict:
    """Four U matrices inducing contrasting H structures. None is the truth."""
    if m == 3:
        return {
            "antichain": np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]),
            "total_order": np.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]]),
            "sparse": np.array([[1.0, 1.0], [0.0, 2.0], [0.5, 0.4]]),
            "dense": np.array([[2.0, 2.5], [1.0, 1.2], [0.0, 0.1]]),
        }
    return {
        "antichain": np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0],
                               [4.0, 0.0]]),
        "total_order": np.array([[5.0, 5.0], [4.0, 4.0], [3.0, 3.0], [2.0, 2.0],
                                 [1.0, 1.0]]),
        "sparse": np.array([[2.0, 2.0], [1.0, 1.0], [0.0, 3.0], [3.0, 0.0],
                            [1.5, 1.6]]),
        "dense": np.array([[4.0, 4.0], [3.0, 3.5], [2.0, 2.0], [1.0, 1.5], [0.0, 0.0]]),
    }


def chain_start(stage: str, chain: int, m: int) -> tuple[np.ndarray, dict, str]:
    starts = u_starts(m)
    name = list(starts)[chain]
    values = {"rho": RHO_STARTS[chain]}
    for scalar in SCALAR_ORDER:
        values[scalar] = prior_quantile(scalar, QUANTILES[LATIN[scalar][chain]])
    return starts[name].copy(), values, name


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:                                            # pragma: no cover
        return "unknown"


def _jsonable(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if math.isnan(v) else v
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def build_problem(stage: str):
    """Return `(roles, u_true, epsilon, truth, heldout)` for the stage's corpus."""
    if stage == "6d1":
        model = small_model()
        return model.roles, model.u_true, model.epsilon, model.truth, None
    frozen = load_stage6d_dataset()
    return frozen.train, frozen.u_true, frozen.epsilon, frozen.truth, frozen.heldout


def make_target(stage: str):
    roles, u_true, epsilon, truth, _ = build_problem(stage)
    evaluator = LatentPosetEvaluator(roles, epsilon=epsilon, omega=truth["omega"])
    return Stage6DTarget(evaluator, active=ACTIVE_6D), roles, u_true, epsilon, truth


def _run_one(payload: dict) -> dict:
    stage, chain = payload["stage"], payload["chain"]
    target, roles, u_true, epsilon, truth = make_target(stage)
    m = u_true.shape[0]
    u_start, values, name = chain_start(stage, chain, m)
    base_seed = payload.get("base_seed", BASE_SEED)
    result = run_oracle_joint_mcmc(
        target, u_start, values, num_sweeps=payload["sweeps"],
        burn_in=payload["burn_in"], thin=payload["thin"], seed=base_seed + chain,
        scales=payload.get("scales"), chain=chain)
    return {
        "chain": chain, "seed": base_seed + chain, "start_name": name,
        "start_relations": int(precedence_from_u(u_start).sum()),
        "start_values": values,
        "u_draws": result.u_draws, "relation_counts": result.relation_counts,
        "scalars": result.scalars, "log_likelihood": result.log_likelihood,
        "log_target": result.log_target,
        "acceptance_total": result.acceptance(post_burn_in=False),
        "acceptance_post": result.acceptance(post_burn_in=True),
        "invalid": result.invalid, "runtime_seconds": result.runtime_seconds,
        "final_u": result.final_state.u,
    }


# --------------------------------------------------------------------------- 6D0 smoke
def run_smoke(out_dir: Path) -> dict:
    frozen = load_stage6d_dataset()
    blocks = frozen.train[:60]
    evaluator = LatentPosetEvaluator(blocks, epsilon=frozen.epsilon,
                                     omega=frozen.truth["omega"])
    target = Stage6DTarget(evaluator, active=ACTIVE_6D)
    u_start, values, _ = chain_start("6d2", 2, 5)

    rng = np.random.default_rng(7)
    state = initial_state(target, u_start, values, rng)
    targets, paths = [state.log_target], {n: [state.values[n]] for n in
                                          ("rho",) + SCALAR_ORDER}
    evaluator.full_replay_calls = 0
    evaluator.cached_calls = 0
    for _ in range(400):
        state = sweep_once(state, target, REGISTERED_SCALES, rng)
        targets.append(state.log_target)
        for n in paths:
            paths[n].append(state.values[n])

    # q_0 reset: every block starts the recurrence at zero
    features = vectorized_state_features(blocks, state.u, state.values["omega"])
    q_zero = bool(np.all(features["q"][:, 0, :] == 0.0))

    # direct (uncached) and cached targets must agree
    evaluator.ensure_cache(state.u, state.values["omega"])
    cached = target.log_target(state.u, state.values, allow_cache=True)
    direct = log_target_6d(target, state.u, state.values)["log_target"]

    # rejection safety: scoring candidates cannot disturb a valid cache
    key_before, builds_before = evaluator.cache_key, evaluator.cache_builds
    for _ in range(10):
        evaluator.full_replay_log_likelihood(
            state.u + rng.normal(size=state.u.shape), state.values["beta"],
            state.values["omega"], state.values["lambda_rep"],
            state.values["lambda_back"])
    rejection_safe = (evaluator.cache_key == key_before
                      and evaluator.cache_builds == builds_before)

    # deterministic serialisation and resume
    restored = Stage6DState.from_dict(json.loads(json.dumps(state.to_dict())))
    serialisation_ok = (np.array_equal(restored.u, state.u)
                        and restored.values == state.values
                        and restored.rng_state == state.rng_state)
    whole = run_oracle_joint_mcmc(Stage6DTarget(
        LatentPosetEvaluator(blocks, epsilon=frozen.epsilon,
                             omega=frozen.truth["omega"]), active=ACTIVE_6D),
        u_start, values, 60, 10, 1, seed=3)
    first = run_oracle_joint_mcmc(Stage6DTarget(
        LatentPosetEvaluator(blocks, epsilon=frozen.epsilon,
                             omega=frozen.truth["omega"]), active=ACTIVE_6D),
        u_start, values, 30, 10, 1, seed=3)
    resume_rng = np.random.default_rng(3)
    resume_rng.bit_generator.state = first.final_state.rng_state
    second = run_oracle_joint_mcmc(Stage6DTarget(
        LatentPosetEvaluator(blocks, epsilon=frozen.epsilon,
                             omega=frozen.truth["omega"]), active=ACTIVE_6D),
        u_start, values, 60, 10, 1, seed=3, state=first.final_state, rng=resume_rng)
    resume_ok = bool(np.array_equal(
        np.concatenate([first.scalars["rho"], second.scalars["rho"]]),
        whole.scalars["rho"]))

    names = ["U", "rho"] + list(SCALAR_ORDER)
    checks = {f"{n}_moves": (state.accepted[n] > 0) for n in names}
    checks.update({f"{n}_accepts_and_rejects":
                   bool(0 < state.accepted[n] < state.proposed[n]) for n in names})
    checks.update({
        "invalid_rho_recorded": "rho" in state.invalid,
        "q_zero_reset": q_zero,
        "no_nans": bool(np.all(np.isfinite(targets))),
        "direct_and_cached_targets_agree": abs(cached - direct) < 1e-9,
        "rejection_safe": bool(rejection_safe),
        "save_load": bool(serialisation_ok),
        "deterministic_resume": resume_ok,
        "diagnostics_ran": True,
    })
    failed = [k for k, v in checks.items() if not v]
    summary = {
        "stage": "6D0", "n_sweeps": 400,
        "checks": {k: bool(v) for k, v in checks.items()},
        "failed_checks": failed, "all_passed": not failed,
        "proposed": state.proposed, "accepted": state.accepted,
        "invalid": state.invalid,
        "acceptance": {n: state.accepted[n] / max(state.proposed[n], 1) for n in names},
        "ranges": {n: [float(np.min(v)), float(np.max(v))] for n, v in paths.items()},
        "replay_calls": evaluator.full_replay_calls,
        "cached_calls": evaluator.cached_calls,
        "expected_replays_per_sweep": "m U rows + 1 omega = %d" % (state.u.shape[0] + 1),
        "final_relation_count": state.relation_count,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    np.savez_compressed(out_dir / "chains.npz", log_target=np.array(targets),
                        **{n: np.array(v) for n, v in paths.items()})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("6d0", "6d1", "6d2"), required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=None)
    parser.add_argument("--out-name", default=None,
                        help="override the result directory name")
    parser.add_argument("--base-seed", type=int, default=None,
                        help="override the per-chain base seed")
    parser.add_argument("--omega-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--scales-json", default=None,
                        help="JSON dict of frozen proposal scales")
    args = parser.parse_args()

    if args.stage == "6d0":
        summary = run_smoke(RESULTS / "stage6d0_joint_smoke")
        print(json.dumps(_jsonable(summary["checks"]), indent=2))
        print("all passed:", summary["all_passed"], summary["failed_checks"])
        if not summary["all_passed"]:
            raise SystemExit(f"smoke failed: {summary['failed_checks']}")
        return

    stage = args.stage
    cfg = dict(SETTINGS[stage])
    if args.sweeps:
        cfg["sweeps"] = args.sweeps
    global BASE_SEED, RUN_SCALES
    if args.base_seed is not None:
        BASE_SEED = args.base_seed
    RUN_SCALES = dict(REGISTERED_SCALES)
    RUN_SCALES["omega"] = REGISTERED_SCALES["omega"] * args.omega_scale_multiplier
    if args.scales_json:
        RUN_SCALES.update({k: float(v) for k, v in
                           json.loads(args.scales_json).items()})
    default_name = ("stage6d1_joint_mcmc" if stage == "6d1"
                    else "stage6d2_oracle_joint_full_seed0")
    out_dir = RESULTS / (args.out_name or default_name)
    roles, u_true, epsilon, truth, heldout = build_problem(stage)
    m = u_true.shape[0]

    payloads = [{"stage": stage, "chain": c, "base_seed": BASE_SEED,
                 "scales": RUN_SCALES, **cfg} for c in range(cfg["chains"])]
    began = time.perf_counter()
    if args.jobs > 1:
        with Pool(min(args.jobs, cfg["chains"])) as pool:
            chains = pool.map(_run_one, payloads)
    else:
        chains = [_run_one(p) for p in payloads]
    wall = time.perf_counter() - began
    chains.sort(key=lambda c: c["chain"])

    u_by_chain = np.array([c["u_draws"] for c in chains])
    scalars = {n: np.array([c["scalars"][n] for c in chains])
               for n in ("rho",) + SCALAR_ORDER}
    log_target = np.array([c["log_target"] for c in chains])
    log_likelihood = np.array([c["log_likelihood"] for c in chains])
    relation_counts = np.array([c["relation_counts"] for c in chains])

    reference = {}
    ref_keys = ref_h = ref_rel = None
    ref_u = ref_scalars = ref_w = None
    if stage == "6d1":
        ref_dir = RESULTS / "stage6d1_joint_reference"
        with np.load(ref_dir / "qmc_replicates.npz") as z:
            ref_h = z["pooled_h_probability"]
            ref_rel = z["pooled_relation_marginal"]
            ref_keys = [bytes(row) for row in z["h_keys"]]
            ref_u = z["pooled_u"].astype(float)
            ref_w = z["pooled_weights"].astype(float)
            ref_scalars = {n: z[f"pooled_{n}"].astype(float)
                           for n in ("rho",) + SCALAR_ORDER}
        summary = json.loads((ref_dir / "reference_summary.json").read_text())
        reference = {n: {"mean": summary["scalars"][n]["mean"],
                         "sd": summary["scalars"][n]["sd"]}
                     for n in ("rho",) + SCALAR_ORDER}

    structural = structural_block(u_by_chain, ref_keys, ref_h, ref_rel)
    truth_for_scalars = {**{n: truth[n] for n in SCALAR_ORDER}, "rho": None}
    acceptance_total = {k: float(np.mean([c["acceptance_total"].get(k, np.nan)
                                          for c in chains]))
                        for k in chains[0]["acceptance_total"]}
    acceptance_post = {k: float(np.mean([c["acceptance_post"].get(k, np.nan)
                                         for c in chains]))
                       for k in chains[0]["acceptance_post"]}
    scalar_diag = scalar_block(scalars, reference, truth_for_scalars,
                               acceptance_total, acceptance_post)
    scalar_diag["log_target"] = {"posterior_mean": float(log_target.mean())}
    from hpop.mcmc_original.stage6c_diagnostics import convergence_block
    scalar_diag["log_target"].update(convergence_block(log_target, "log target"))
    scalar_diag["acceptance"] = {"total": acceptance_total,
                                 "post_burn_in": acceptance_post,
                                 "invalid_rho": sum(c["invalid"].get("rho", 0)
                                                    for c in chains)}

    series = {"relation_count": relation_counts.astype(float),
              "log_likelihood": log_likelihood,
              **{n: scalars[n] for n in ("rho",) + SCALAR_ORDER}}
    joint = dependence_block(series, [
        ("relation_count", "rho"), ("relation_count", "beta"),
        ("relation_count", "omega"), ("rho", "beta"), ("beta", "omega"),
        ("beta", "lambda_rep"), ("omega", "lambda_back"),
        ("log_likelihood", "relation_count")])
    joint["column_permutation"] = column_permutation_audit(u_by_chain)

    gates = {}
    if stage == "6d1":
        mixed = mixed_envelope(u_by_chain.reshape(-1, m, u_by_chain.shape[3]),
                               {n: scalars[n].ravel() for n in ("rho",) + SCALAR_ORDER},
                               ref_u, ref_scalars, ref_w, seed=202)
        gates = {
            "induced_h_total_variation": {
                "value": structural["h_total_variation"], "threshold": 0.01,
                "pass": structural["h_total_variation"] < 0.01},
            "max_relation_marginal_error": {
                "value": structural["max_relation_marginal_error"], "threshold": 0.01,
                "pass": structural["max_relation_marginal_error"] < 0.01},
            "mixed_reference_envelope": {
                "value": mixed["observed"], "threshold": mixed["envelope"],
                "pass": mixed["pass"]},
        }
        for n in ("rho",) + SCALAR_ORDER:
            r = scalar_diag[n].get("rhat")
            gates[f"{n}_rhat"] = {"value": r, "threshold": 1.01,
                                  "pass": r is not None and r <= 1.01}
        lt = scalar_diag["log_target"].get("rhat")
        gates["log_posterior_rhat"] = {"value": lt, "threshold": 1.01,
                                       "pass": lt is not None and lt <= 1.01}
        rc = structural["relation_count_convergence"].get("rhat")
        gates["relation_count_rhat"] = {"value": rc, "threshold": 1.01,
                                        "pass": rc is not None and rc <= 1.01}
        mr = structural.get("max_relation_rhat")
        gates["uncertain_relation_rhat"] = {"value": mr, "threshold": 1.01,
                                            "pass": mr is not None and mr <= 1.01}
        (out_dir).mkdir(parents=True, exist_ok=True)
        (out_dir / "reference_comparison.json").write_text(json.dumps(_jsonable({
            "gates": gates, "mixed": mixed,
            "retained_draws_pooled": int(u_by_chain.shape[0] * u_by_chain.shape[1]),
            "note": "the reference was frozen before these chains ran"}), indent=2))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "chains.npz", u_draws=u_by_chain.astype(np.float32),
        relation_counts=relation_counts, log_target=log_target,
        log_likelihood=log_likelihood,
        **{n: scalars[n] for n in ("rho",) + SCALAR_ORDER})
    (out_dir / "structural_diagnostics.json").write_text(
        json.dumps(_jsonable(structural), indent=2))
    (out_dir / "scalar_diagnostics.json").write_text(
        json.dumps(_jsonable(scalar_diag), indent=2))
    (out_dir / "joint_diagnostics.json").write_text(json.dumps(_jsonable(joint), indent=2))
    (out_dir / "config.json").write_text(json.dumps(_jsonable({
        "source_commit": source_commit(), "stage6d_config_hash": config_hash(),
        "python": platform.python_version(), "numpy": np.__version__,
        "stage": stage, **cfg, "base_seed": BASE_SEED,
        "proposal_scales": RUN_SCALES,
        "omega_scale_multiplier": args.omega_scale_multiplier,
        "chain_starts": [{"chain": c["chain"], "u_start": c["start_name"],
                          "start_relations": c["start_relations"],
                          "start_values": c["start_values"], "seed": c["seed"]}
                         for c in chains],
        "retained_per_chain": int(u_by_chain.shape[1]),
        "retained_pooled": int(u_by_chain.shape[0] * u_by_chain.shape[1]),
        "wall_seconds": wall,
        "chain_runtime_seconds": [c["runtime_seconds"] for c in chains],
    }), indent=2))

    print(f"[{stage}] {cfg['chains']} chains x {cfg['sweeps']:,} sweeps in "
          f"{wall / 60:.1f} min", flush=True)
    for name, g in gates.items():
        v = g["value"]
        print(f"[{stage}] {name}: {v if v is None else f'{v:.6g}'} vs "
              f"{g['threshold']:.4g} -> {'PASS' if g['pass'] else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
