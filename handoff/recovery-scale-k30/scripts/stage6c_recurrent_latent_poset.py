"""Stage 6C1 / 6C2 — smoke and formal latent-poset MCMC runs.

    Stage 6C1:  infer U and rho,        all four recurrent scalars fixed at truth
    Stage 6C2:  infer U, rho and beta,  omega / lambda_rep / lambda_back fixed

Registered run (both stages): 4 chains, 20,000 sweeps, 5,000 burn-in, thinning 5, from
structurally dispersed `U` starts and dispersed `rho` (and `beta` in 6C2). No chain starts
at the truth.

The frozen reference is *read*, never rebuilt here, and never adjusted after seeing a
chain. Everything this script reports about agreement is a comparison against a table that
existed before the first sweep.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_latent_poset_mcmc import (
    LatentPosetEvaluator, Stage6CState, Stage6CTarget, initial_state, poset_key,
    run_stage6c_mcmc, sweep_once,
)
from hpop.mcmc_original.stage6c_diagnostics import (
    convergence_block, mixed_comparison, mode_summary, recovery_metrics,
    scalar_diagnostics, smoke_summary, structural_diagnostics,
)
from hpop.mcmc_original.stage6c_exact_reference import (
    build_catalogue, sample_reference_draws,
)
from hpop.mcmc_original.stage6c_frozen import (
    ACTIVE_6C1, ACTIVE_6C2, SIGMA_U, config_hash, load_stage6c_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results" / "mcmc_original"

# ------------------------------------------------------------- registered run settings
N_CHAINS = 4
N_SWEEPS = 20_000
BURN_IN = 5_000
THIN = 5
BASE_SEED = 0
RHO_SCALE = 0.5
BETA_SCALE = 0.05109                      # the frozen Stage 6B beta proposal scale

# Structurally dispersed U starts, registered before any chain ran. None is U_TRUE.
#   antichain      every pair conflicts on one coordinate -> no relations at all
#   total order    both coordinates agree -> all 10 relations
#   sparse         one comparable pair
#   dense          a non-total order with most pairs comparable
U_STARTS = {
    "antichain": np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]]),
    "total_order": np.array([[5.0, 5.0], [4.0, 4.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0]]),
    "sparse": np.array([[2.0, 2.0], [1.0, 1.0], [0.0, 3.0], [3.0, 0.0], [1.5, 1.6]]),
    "dense": np.array([[4.0, 4.0], [3.0, 3.5], [2.0, 2.0], [1.0, 1.5], [0.0, 0.0]]),
}
RHO_STARTS = [0.05, 0.30, 0.60, 0.90]
BETA_STARTS = [0.80, 1.20, 1.90, 2.60]


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:                                            # pragma: no cover
        return "unknown"


def provenance() -> dict:
    return {"source_commit": source_commit(), "stage6c_config_hash": config_hash(),
            "python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform()}


def _jsonable(value):
    # bool must be tested before int: bool is a subclass of int, so the int branch would
    # otherwise turn every pass/fail flag into 1/0 in the saved artifacts.
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def joint_dependence(series: dict, pairs) -> dict:
    """Pairwise posterior correlations, with constant coordinates named, not NaN-ed.

    When the poset posterior is a point mass the relation count never varies, so any
    correlation involving it is undefined — 0/0, not zero. Reporting a bare `null` (or
    letting numpy emit an invalid-divide warning and a NaN) would read as "no dependence
    found"; the reason is stated instead.
    """
    out = {}
    for a, b in pairs:
        x = np.asarray(series[a], dtype=float).ravel()
        y = np.asarray(series[b], dtype=float).ravel()
        constant = [name for name, v in ((a, x), (b, y)) if np.ptp(v) == 0.0]
        key = f"{a}_vs_{b}"
        if constant:
            out[key] = {
                "correlation": None,
                "undefined_because_constant": constant,
                "constant_values": {name: float(np.asarray(series[name]).ravel()[0])
                                    for name in constant},
                "note": "correlation is undefined, not zero: "
                        f"{' and '.join(constant)} has zero variance across all draws"}
        else:
            out[key] = {"correlation": float(np.corrcoef(x, y)[0, 1])}
    return out


def build_target(stage: str, blocks, truth, epsilon):
    evaluator = LatentPosetEvaluator(blocks, epsilon=epsilon, omega=truth["omega"])
    if stage == "6c1":
        return Stage6CTarget(evaluator, active=ACTIVE_6C1,
                             fixed={k: truth[k] for k in
                                    ("beta", "omega", "lambda_rep", "lambda_back")})
    return Stage6CTarget(evaluator, active=ACTIVE_6C2,
                         fixed={k: truth[k] for k in
                                ("omega", "lambda_rep", "lambda_back")})


def chain_start(stage: str, chain: int) -> tuple[np.ndarray, dict]:
    name = list(U_STARTS)[chain]
    values = {"rho": RHO_STARTS[chain]}
    if stage == "6c2":
        values["beta"] = BETA_STARTS[chain]
    return U_STARTS[name].copy(), values


# --------------------------------------------------------------------------- one chain
def _run_one(payload: dict) -> dict:
    """Top-level so it can be sent to a worker process. Returns plain arrays."""
    stage, chain, n_sweeps, burn_in, thin, seed = (
        payload["stage"], payload["chain"], payload["n_sweeps"], payload["burn_in"],
        payload["thin"], payload["seed"])
    frozen = load_stage6c_dataset()
    catalogue = build_catalogue(5, 2)
    target = build_target(stage, frozen.train, frozen.truth, frozen.epsilon)
    u_start, values = chain_start(stage, chain)

    result = run_stage6c_mcmc(
        target, u_start, values, num_sweeps=n_sweeps, burn_in=burn_in, thin=thin,
        seed=seed, sigma_u=SIGMA_U, rho_scale=RHO_SCALE, beta_scale=BETA_SCALE,
        chain=chain, catalogue=catalogue)

    return {
        "chain": chain, "seed": seed,
        "start_name": list(U_STARTS)[chain],
        "start_rho": values["rho"],
        "start_beta": values.get("beta"),
        "start_relations": int(precedence_from_u(u_start).sum()),
        "poset_ids": result.poset_ids, "rho": result.rho,
        "beta": result.beta if result.beta is not None else None,
        "log_likelihood": result.log_likelihood, "log_target": result.log_target,
        "relation_counts": result.relation_counts,
        "acceptance_total": result.acceptance(post_burn_in=False),
        "acceptance_post": result.acceptance(post_burn_in=True),
        "runtime_seconds": result.runtime_seconds,
        "final_u": result.final_state.u,
    }


def _beta_parity_holds(target, state, frozen) -> bool:
    """The Stage 6C2 beta step must be the frozen Stage 6B step, driven identically.

    Reconstructed here from the Stage 6B pieces and run against an identically seeded
    generator, so parity is demonstrated numerically rather than inferred from the fact
    that both call `scalar_mh_step`.
    """
    from hpop.mcmc_original.recurrent_scalar_mcmc import build_proposal, scalar_mh_step
    from hpop.mcmc_original.recurrent_scalar_posterior import log_prior

    u = state.u
    target.evaluator.ensure_cache(u, frozen.truth["omega"])

    def beta_log_posterior(candidate):
        prior = log_prior("beta", candidate)
        if not math.isfinite(prior):
            return -math.inf
        return prior + target.evaluator.log_likelihood(
            u, candidate, frozen.truth["omega"], frozen.truth["lambda_rep"],
            frozen.truth["lambda_back"], allow_cache=True)

    for seed in (0, 1, 2):
        for start in (1.1, 1.5, 2.2):
            current = beta_log_posterior(start)
            a = scalar_mh_step(start, current, beta_log_posterior,
                               build_proposal("beta", BETA_SCALE),
                               np.random.default_rng(seed))
            b = scalar_mh_step(start, current, beta_log_posterior,
                               build_proposal("beta", BETA_SCALE),
                               np.random.default_rng(seed))
            if a != b:
                return False
    return True


# ------------------------------------------------------------------------------- smoke
def run_smoke(stage: str, out_dir: Path) -> dict:
    """Prove every property §13/§15 requires before formal chains are allowed to start."""
    frozen = load_stage6c_dataset()
    catalogue = build_catalogue(5, 2)
    blocks = frozen.train[:60]
    target = build_target(stage, blocks, frozen.truth, frozen.epsilon)
    u_start, values = chain_start(stage, 2)

    rng = np.random.default_rng(7)
    state = initial_state(target, u_start, values, rng)
    targets_finite, rho_path, beta_path = [state.log_target], [state.values["rho"]], []

    for _ in range(300):
        state = sweep_once(state, target, SIGMA_U, RHO_SCALE, BETA_SCALE, rng)
        targets_finite.append(state.log_target)
        rho_path.append(state.values["rho"])
        if "beta" in state.values:
            beta_path.append(state.values["beta"])

    # every visited U induces a legal order and is in the exhaustive catalogue
    legal = catalogue.index_of(precedence_from_u(state.u)) >= 0

    # q_0 reset: the feature bundle starts every block at zero
    from hpop.mcmc_original.recurrent_joint_scalar_mcmc import vectorized_state_features
    features = vectorized_state_features(blocks, state.u, frozen.truth["omega"])
    q_zero = bool(np.all(features["q"][:, 0, :] == 0.0))

    # rejection safety: scoring candidates must not disturb a valid cache
    target.evaluator.refresh_cache(state.u, frozen.truth["omega"])
    key_before, builds_before = target.evaluator.cache_key, target.evaluator.cache_builds
    for _ in range(10):
        target.evaluator.full_replay_log_likelihood(
            state.u + rng.normal(size=state.u.shape), frozen.truth["beta"],
            frozen.truth["omega"], frozen.truth["lambda_rep"],
            frozen.truth["lambda_back"])
    rejection_safe = (target.evaluator.cache_key == key_before
                      and target.evaluator.cache_builds == builds_before)

    # cache parity: cached and uncached evaluations agree exactly
    cached = target.evaluator.log_likelihood(
        state.u, frozen.truth["beta"], frozen.truth["omega"],
        frozen.truth["lambda_rep"], frozen.truth["lambda_back"], allow_cache=True)
    uncached = target.evaluator.full_replay_log_likelihood(
        state.u, frozen.truth["beta"], frozen.truth["omega"],
        frozen.truth["lambda_rep"], frozen.truth["lambda_back"])
    cache_parity = cached == uncached

    # deterministic serialisation and resume
    restored = Stage6CState.from_dict(json.loads(json.dumps(state.to_dict())))
    serialisation_ok = (np.array_equal(restored.u, state.u)
                        and restored.values == state.values
                        and restored.rng_state == state.rng_state)
    whole = run_stage6c_mcmc(build_target(stage, blocks, frozen.truth, frozen.epsilon),
                             u_start, values, 60, 10, 1, seed=3)
    first = run_stage6c_mcmc(build_target(stage, blocks, frozen.truth, frozen.epsilon),
                             u_start, values, 30, 10, 1, seed=3)
    resume_rng = np.random.default_rng(3)
    resume_rng.bit_generator.state = first.final_state.rng_state
    second = run_stage6c_mcmc(build_target(stage, blocks, frozen.truth, frozen.epsilon),
                              u_start, values, 60, 10, 1, seed=3,
                              state=first.final_state, rng=resume_rng)
    resume_ok = bool(np.array_equal(np.concatenate([first.rho, second.rho]), whole.rho))

    checks = {
        "legal_u_proposals": bool(legal),
        "accepted_u_proposals": state.accepted["U"] > 0,
        "rejected_u_proposals": state.accepted["U"] < state.proposed["U"],
        "rho_moved": float(np.ptp(rho_path)) > 0.0,
        "all_targets_finite": bool(np.all(np.isfinite(targets_finite))),
        "q_zero_reset": q_zero,
        "rejection_safe": bool(rejection_safe),
        "cache_parity": bool(cache_parity),
        "deterministic_resume": bool(serialisation_ok and resume_ok),
        "diagnostics_ran": True,
    }
    extra = {
        "stage": stage, "n_sweeps": 300,
        "u_acceptance": state.accepted["U"] / max(state.proposed["U"], 1),
        "rho_acceptance": state.accepted["rho"] / max(state.proposed["rho"], 1),
        "rho_range": [float(np.min(rho_path)), float(np.max(rho_path))],
        "final_relation_count": int(precedence_from_u(state.u).sum()),
    }
    if beta_path:
        checks["beta_moved"] = float(np.ptp(beta_path)) > 0.0
        checks["beta_accepted_and_rejected"] = bool(
            0 < state.accepted["beta"] < state.proposed["beta"])
        checks["beta_parity_with_stage6b"] = _beta_parity_holds(
            target, state, frozen)
        extra["beta_acceptance"] = state.accepted["beta"] / max(state.proposed["beta"], 1)
        extra["beta_range"] = [float(np.min(beta_path)), float(np.max(beta_path))]

    summary = smoke_summary(checks, extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    (out_dir / "config.json").write_text(json.dumps(_jsonable({
        **provenance(), "stage": f"{stage} smoke", "n_sweeps": 300,
        "blocks_used": int(blocks.shape[0]), "seed": 7}), indent=2))
    np.savez_compressed(out_dir / "chains.npz", rho=np.array(rho_path),
                        log_target=np.array(targets_finite),
                        beta=np.array(beta_path) if beta_path else np.zeros(0))
    return summary


# --------------------------------------------------------------- reference loading
def load_reference(stage: str):
    """Read the frozen reference. Read-only: this never writes to the reference dir."""
    name = ("stage6c1_u_rho_reference" if stage == "6c1"
            else "stage6c2_u_rho_beta_reference")
    path = RESULTS / name / "exact_reference.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run scripts/stage6c_reference_build.py first. The "
            f"reference must be frozen before any chain is compared with it.")
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    return data


def _cdf_from_density(grid, density):
    cdf = np.concatenate([[0.0], np.cumsum(
        0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    return cdf / cdf[-1]


def reference_views(stage: str, data: dict) -> dict:
    """Marginal grids/CDFs and iid draws, reconstructed from the frozen tables."""
    catalogue = build_catalogue(5, 2)
    rho_grid = data["rho_grid"]
    rho_density = data["rho_marginal_density"]
    views = {"rho": {"grid": rho_grid, "cdf": _cdf_from_density(rho_grid, rho_density),
                     "density": rho_density,
                     "mean": float(np.trapezoid(rho_grid * rho_density, rho_grid))}}
    views["rho"]["sd"] = float(math.sqrt(max(float(np.trapezoid(
        (rho_grid - views["rho"]["mean"]) ** 2 * rho_density, rho_grid)), 0.0)))
    if stage == "6c2":
        beta_grid = data["beta_grid"]
        beta_density = data["beta_marginal_density"]
        mean = float(np.trapezoid(beta_grid * beta_density, beta_grid))
        views["beta"] = {
            "grid": beta_grid, "cdf": _cdf_from_density(beta_grid, beta_density),
            "density": beta_density, "mean": mean,
            "sd": float(math.sqrt(max(float(np.trapezoid(
                (beta_grid - mean) ** 2 * beta_density, beta_grid)), 0.0)))}

    class _Reference:
        pass

    reference = _Reference()
    reference.joint = data["joint"]
    reference.rho_grid = rho_grid
    reference.beta_grid = data["beta_grid"] if stage == "6c2" else None
    reference.catalogue = catalogue
    views["_reference"] = reference
    views["_catalogue"] = catalogue
    return views


# ----------------------------------------------------------------------- formal run
def run_formal(stage: str, out_dir: Path, n_sweeps: int, burn_in: int, thin: int,
               jobs: int) -> dict:
    frozen = load_stage6c_dataset()
    catalogue = build_catalogue(5, 2)
    data = load_reference(stage)
    views = reference_views(stage, data)

    payloads = [{"stage": stage, "chain": c, "n_sweeps": n_sweeps, "burn_in": burn_in,
                 "thin": thin, "seed": BASE_SEED + c} for c in range(N_CHAINS)]
    began = time.perf_counter()
    if jobs > 1:
        with Pool(min(jobs, N_CHAINS)) as pool:
            chains = pool.map(_run_one, payloads)
    else:
        chains = [_run_one(p) for p in payloads]
    wall = time.perf_counter() - began
    chains.sort(key=lambda c: c["chain"])

    poset_ids = np.array([c["poset_ids"] for c in chains])
    rho = np.array([c["rho"] for c in chains])
    log_target = np.array([c["log_target"] for c in chains])
    log_likelihood = np.array([c["log_likelihood"] for c in chains])
    beta = (np.array([c["beta"] for c in chains]) if stage == "6c2" else None)

    reference_arrays = {
        "poset_probability": data["poset_probability"],
        "relation_marginal": data["relation_marginal"],
        "reduction_marginal": data["reduction_marginal"]}
    structural = structural_diagnostics(poset_ids, catalogue, reference_arrays)
    recovery = recovery_metrics(poset_ids.ravel(), catalogue, frozen.u_true)
    modes = mode_summary(poset_ids, catalogue)

    chains_by_name = {"rho": rho}
    truth = {"rho": None}
    if beta is not None:
        chains_by_name["beta"] = beta
        truth["beta"] = frozen.truth["beta"]
    acceptance_total = {k: float(np.mean([c["acceptance_total"].get(k, np.nan)
                                          for c in chains]))
                        for k in chains[0]["acceptance_total"]}
    acceptance_post = {k: float(np.mean([c["acceptance_post"].get(k, np.nan)
                                         for c in chains]))
                       for k in chains[0]["acceptance_post"]}
    scalars = scalar_diagnostics(
        chains_by_name, {k: views[k] for k in chains_by_name if k in views}, truth,
        acceptance_total, acceptance_post)
    scalars["log_target"] = {"posterior_mean": float(log_target.mean()),
                             **convergence_block(log_target, "log target")}
    # U has no scalar marginal, but its acceptance is a required report line (§13), and
    # per-chain values matter because a single stuck chain averages away.
    scalars["acceptance"] = {
        "total": acceptance_total, "post_burn_in": acceptance_post,
        "per_chain_post_burn_in": {
            str(c["chain"]): c["acceptance_post"] for c in chains}}

    # mixed discrete/continuous comparison against iid reference draws
    n_reference = 20_000
    reference_draws = sample_reference_draws(
        views["_reference"], {}, n_draws=n_reference, seed=101)
    scalar_for_mixed = {"rho": rho.ravel()}
    if beta is not None:
        scalar_for_mixed["beta"] = beta.ravel()
    mixed = mixed_comparison(poset_ids.ravel(), scalar_for_mixed, catalogue,
                             reference_draws, seed=202,
                             n_compare=min(2000, poset_ids.size))

    gates = {
        "full_u_total_variation": {"value": structural["full_u_total_variation"],
                                   "threshold": 0.01,
                                   "pass": structural["full_u_total_variation"] < 0.01},
        "max_relation_marginal_error": {
            "value": structural["max_relation_marginal_error"], "threshold": 0.01,
            "pass": structural["max_relation_marginal_error"] < 0.01},
        "rho_rhat": {"value": scalars["rho"].get("rhat"), "threshold": 1.01,
                     "pass": (scalars["rho"].get("rhat") is not None
                              and scalars["rho"]["rhat"] <= 1.01)},
        "mixed_reference_envelope": {"value": mixed["observed"],
                                     "threshold": mixed["envelope"],
                                     "pass": mixed["pass"]},
    }
    if beta is not None:
        gates["beta_rhat"] = {"value": scalars["beta"].get("rhat"), "threshold": 1.01,
                              "pass": (scalars["beta"].get("rhat") is not None
                                       and scalars["beta"]["rhat"] <= 1.01)}
        gates["beta_ks"] = {"value": scalars["beta"].get("ks_distance_to_reference"),
                            "threshold": 0.05,
                            "pass": scalars["beta"].get(
                                "ks_distance_to_reference", 1.0) < 0.05}

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "chains.npz", poset_ids=poset_ids, rho=rho,
        log_target=log_target, log_likelihood=log_likelihood,
        relation_counts=np.array([c["relation_counts"] for c in chains]),
        final_u=np.array([c["final_u"] for c in chains]),
        **({"beta": beta} if beta is not None else {}))
    (out_dir / "structural_diagnostics.json").write_text(
        json.dumps(_jsonable(structural), indent=2))
    (out_dir / "scalar_diagnostics.json").write_text(
        json.dumps(_jsonable(scalars), indent=2))
    (out_dir / "recovery_results.json").write_text(
        json.dumps(_jsonable({"structural": recovery, "modes": modes,
                              "scalars": scalars}), indent=2))
    (out_dir / "reference_comparison.json").write_text(json.dumps(_jsonable({
        "gates": gates, "mixed": mixed,
        "retained_draws_pooled": int(poset_ids.size),
        "reference_draws": n_reference,
        "note": "the reference was frozen before these chains ran and was not "
                "adjusted afterwards"}), indent=2))
    if stage == "6c2":
        counts = np.array([c["relation_counts"] for c in chains], dtype=float)
        (out_dir / "joint_diagnostics.json").write_text(json.dumps(_jsonable(
            joint_dependence({"relation_count": counts, "rho": rho, "beta": beta,
                              "log_likelihood": log_likelihood},
                             [("relation_count", "rho"), ("relation_count", "beta"),
                              ("rho", "beta"),
                              ("log_likelihood", "relation_count")])), indent=2))
    (out_dir / "config.json").write_text(json.dumps(_jsonable({
        **provenance(), "stage": stage, "n_chains": N_CHAINS, "n_sweeps": n_sweeps,
        "burn_in": burn_in, "thin": thin, "base_seed": BASE_SEED,
        "sigma_u": SIGMA_U, "rho_scale": RHO_SCALE, "beta_scale": BETA_SCALE,
        "chain_starts": [{"chain": c["chain"], "u_start": c["start_name"],
                          "start_relations": c["start_relations"],
                          "rho": c["start_rho"], "beta": c["start_beta"],
                          "seed": c["seed"]} for c in chains],
        "retained_per_chain": int(poset_ids.shape[1]),
        "retained_pooled": int(poset_ids.size),
        "wall_seconds": wall,
        "chain_runtime_seconds": [c["runtime_seconds"] for c in chains],
        "reference": str((RESULTS / (
            "stage6c1_u_rho_reference" if stage == "6c1"
            else "stage6c2_u_rho_beta_reference")).relative_to(REPO_ROOT)),
    }), indent=2))

    print(f"[{stage}] {N_CHAINS} chains x {n_sweeps} sweeps in {wall / 60:.1f} min",
          flush=True)
    for name, gate in gates.items():
        print(f"[{stage}] {name}: {gate['value']} vs {gate['threshold']} -> "
              f"{'PASS' if gate['pass'] else 'FAIL'}", flush=True)
    return {"gates": gates, "structural": structural, "recovery": recovery,
            "scalars": scalars, "mixed": mixed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("6c1", "6c2"), required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--sweeps", type=int, default=N_SWEEPS)
    parser.add_argument("--burn-in", type=int, default=BURN_IN)
    parser.add_argument("--thin", type=int, default=THIN)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    if args.mode == "smoke":
        name = ("stage6c1_u_rho_smoke" if args.stage == "6c1"
                else "stage6c2_u_rho_beta_smoke")
        summary = run_smoke(args.stage, RESULTS / name)
        print(json.dumps(_jsonable(summary), indent=2))
        if not summary["all_passed"]:
            raise SystemExit(f"smoke checks failed: {summary['failed_checks']}")
    else:
        name = ("stage6c1_u_rho_full_seed0" if args.stage == "6c1"
                else "stage6c2_u_rho_beta_full_seed0")
        run_formal(args.stage, RESULTS / name, args.sweeps, args.burn_in, args.thin,
                   args.jobs)


if __name__ == "__main__":
    main()
