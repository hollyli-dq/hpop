"""Stage 6B2 / 6B3 — the joint scalar sampler, against the frozen joint references.

    PYTHONPATH=src python scripts/stage6b_joint_mcmc.py --stage b2 --mode smoke \
        --output-dir results/mcmc_original/stage6b2_joint3_smoke
    PYTHONPATH=src python scripts/stage6b_joint_mcmc.py --stage b2 --mode full \
        --output-dir results/mcmc_original/stage6b2_joint3_full_seed0

Stage 6B2 infers `beta, omega, lambda_rep` with `lambda_back` held at its registered
value. Stage 6B3 infers all four. Nothing else moves: no U, no rho, no P, no boundaries,
no labels, no epsilon.

The full mode requires the frozen reference produced by
`scripts/stage6b_joint_reference_build.py` to already exist in the output directory. The
reference is read, never written, so the chains cannot move the thing they are judged
against.
"""
from __future__ import annotations

import argparse, json, platform, subprocess, sys, time
from datetime import date
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.recurrent_joint_scalar_mcmc import (                      # noqa: E402
    ACTIVE_B2, ACTIVE_B3, JointScalarTarget, RecurrentJointEvaluator,
    run_joint_scalar_mcmc)
from hpop.mcmc_original.stage6b_frozen import (                                   # noqa: E402
    PARAMETER_ORDER, SWEEP_ORDER, config_hash, frozen_config, load_frozen_dataset)
from hpop.mcmc_original.stage6b_joint_diagnostics import (                        # noqa: E402
    calibrate_correlation_envelope, dependence_diagnostics, marginal_diagnostics,
    multivariate_comparison, recovery_table)
from hpop.mcmc_original.stage6b_joint_reference import (                          # noqa: E402
    ReferenceGrid, normalise_log_density, sample_reference_draws)
from hpop.mcmc_original.stage6b_mcmc_diagnostics import FULL_GATES                # noqa: E402

# Frozen Stage 6B1 proposal scales. Not re-tuned: the joint sampler starts from exactly
# the kernels Stage 6B1 validated, and no pilot is run, so nothing can be tuned against
# the immutable reference.
FROZEN_SCALES = {"beta": 0.05109, "omega": 0.27891,
                 "lambda_rep": 0.07086, "lambda_back": 0.21734}

STAGES = {
    "b2": {"active": ACTIVE_B2, "fixed_from_truth": ("lambda_back",), "seed": 20_250_812},
    "b3": {"active": ACTIVE_B3, "fixed_from_truth": (), "seed": 20_250_813},
}
SETTINGS = {
    "smoke": {"chains": 2, "iterations": 2_000, "burn_in": 500, "thin": 2},
    "full": {"chains": 4, "iterations": 20_000, "burn_in": 4_000, "thin": 2},
}

# Deterministic Latin-hypercube levels: each coordinate visits every prior quantile
# exactly once across the four chains, and no two chains share a level on any coordinate.
START_LEVELS = (0.10, 0.35, 0.65, 0.90)
LATIN_SQUARE = {"beta": (0, 1, 2, 3), "omega": (3, 2, 1, 0),
                "lambda_rep": (1, 3, 0, 2), "lambda_back": (2, 0, 3, 1)}


def git(*a):
    try:
        return subprocess.run(("git",) + a, cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def jsonable(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.floating, np.integer)): return v.item()
    if isinstance(v, (np.bool_, bool)): return bool(v)
    if isinstance(v, dict): return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jsonable(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def prior_quantile(name: str, level: float) -> float:
    """Start values come from the PRIOR, never from a posterior or a reference."""
    spec = frozen_config()["priors"][name]
    if spec["family"] == "gamma":
        return float(stats.gamma(a=spec["shape"], scale=1.0 / spec["rate"]).ppf(level))
    return float(stats.norm(spec["mean"], spec["sd"]).ppf(level))


def dispersed_starts(active, n_chains: int) -> list:
    """Genuinely joint dispersion: every coordinate differs across every chain."""
    starts = []
    for c in range(n_chains):
        starts.append({name: prior_quantile(name, START_LEVELS[LATIN_SQUARE[name][c]])
                       for name in active})
    return starts


def load_reference(directory: Path, active):
    payload = np.load(directory / "joint_reference.npz")
    summary = json.loads((directory / "reference_summary.json").read_text())["summary"]
    axes_z = {n: payload[f"axis_z_{n}"] for n in active}
    log_density = np.asarray(payload["log_density_z"], dtype=float)
    density, log_normaliser = normalise_log_density(tuple(active), axes_z, log_density)
    grid = ReferenceGrid(
        active=tuple(active),
        axes_z=axes_z,
        axes_value={n: payload[f"axis_value_{n}"] for n in active},
        log_density_z=log_density, density_z=density,
        log_normaliser=log_normaliser, n_points=int(summary["n_points"]),
        radius=float(summary["radius"]), fixed={}, map_values={},
        outer_face_mass=float(summary["outer_face_mass"]), config={})
    return grid, summary


def smoke_checks(results, target, active) -> dict:
    """Section 8's smoke criteria, evaluated rather than asserted in prose."""
    checks = {}
    moved = {n: all(np.unique(r.draws[n]).size > 1 for r in results) for n in active}
    checks["all_coordinates_move"] = {"per_coordinate": moved,
                                      "pass": all(moved.values())}
    accepted = {n: all(r.accepted.get(n, 0) > 0 for r in results) for n in active}
    rejected = {n: all(r.proposed[n] - r.accepted.get(n, 0) > 0 for r in results)
                for n in active}
    checks["each_coordinate_accepts"] = {"per_coordinate": accepted,
                                         "pass": all(accepted.values())}
    checks["each_coordinate_rejects"] = {"per_coordinate": rejected,
                                         "pass": all(rejected.values())}
    finite = all(np.isfinite(r.log_posterior).all() and np.isfinite(r.log_likelihood).all()
                 and all(np.isfinite(r.draws[n]).all() for n in active) for r in results)
    checks["no_nans_all_finite"] = {"pass": bool(finite)}

    # q_0 reset: a fresh evaluator must reproduce the final state's likelihood exactly
    final = results[0].final_state
    fresh = RecurrentJointEvaluator(target.evaluator.role_array, target.evaluator.u,
                                    target.evaluator.epsilon)
    complete = target.complete(final.values)
    replayed = fresh.full_replay_log_likelihood(
        complete["beta"], complete["omega"], complete["lambda_rep"], complete["lambda_back"])
    checks["q0_reset_and_state_reproducible"] = {
        "chain_log_likelihood": final.log_likelihood, "fresh_replay": replayed,
        "abs_difference": abs(final.log_likelihood - replayed),
        "pass": bool(abs(final.log_likelihood - replayed) < 1e-9)}

    # state round-trips through its serialised form
    from hpop.mcmc_original.recurrent_joint_scalar_mcmc import JointChainState
    restored = JointChainState.from_dict(final.to_dict())
    checks["state_serialises_and_loads"] = {
        "pass": bool(restored.values == final.values
                     and restored.iteration == final.iteration
                     and restored.rng_state["state"] == final.rng_state["state"])}
    return checks


def resume_check(target, start, scales, seed, iterations, burn_in, thin) -> dict:
    """A run split in two must be bit-identical to the uninterrupted run."""
    target.evaluator.invalidate()
    whole = run_joint_scalar_mcmc(target, start, scales, iterations, burn_in, thin,
                                  seed=seed, chain=0)
    half = iterations // 2
    target.evaluator.invalidate()
    first = run_joint_scalar_mcmc(target, start, scales, half, burn_in, thin,
                                  seed=seed, chain=0)
    rng = np.random.default_rng(seed)
    rng.bit_generator.state = first.final_state.rng_state
    target.evaluator.invalidate()
    second = run_joint_scalar_mcmc(target, start, scales, iterations, burn_in, thin,
                                   seed=seed, chain=0,
                                   initial_state=first.final_state, rng=rng)
    identical = {n: bool(np.array_equal(
        np.concatenate([first.draws[n], second.draws[n]]), whole.draws[n]))
        for n in target.active}
    return {"per_coordinate": identical, "pass": all(identical.values()),
            "n_whole": int(whole.draws[target.active[0]].size),
            "n_split": int(first.draws[target.active[0]].size
                           + second.draws[target.active[0]].size)}


def evaluate_gates(marginals, dependence, multivariate, correlation_envelope,
                   active) -> dict:
    rows = {}
    for name in active:
        d = marginals[name]
        rows[f"{name}:standardized_mean_error"] = (
            d["standardized_mean_error"], FULL_GATES["standardized_mean_error"], "<=")
        rows[f"{name}:standardized_median_error"] = (
            d["standardized_median_error"], FULL_GATES["standardized_median_error"], "<=")
        rows[f"{name}:standardized_q025_error"] = (
            d["standardized_q025_error"], FULL_GATES["standardized_q025_error"], "<=")
        rows[f"{name}:standardized_q975_error"] = (
            d["standardized_q975_error"], FULL_GATES["standardized_q975_error"], "<=")
        rows[f"{name}:ks_distance"] = (d["ks_distance"], FULL_GATES["ks_distance"], "<=")
        rows[f"{name}:rhat"] = (d["rhat"], FULL_GATES["rhat_max"], "<=")
        rows[f"{name}:bulk_ess"] = (d["bulk_ess"], FULL_GATES["bulk_ess_min"], ">=")
        rows[f"{name}:tail_ess"] = (d["tail_ess"], FULL_GATES["tail_ess_min"], ">=")
        lo, hi = FULL_GATES["acceptance_rate_range"]
        rows[f"{name}:acceptance_min"] = (min(d["acceptance_post_burn_in"]), lo, ">=")
        rows[f"{name}:acceptance_max"] = (max(d["acceptance_post_burn_in"]), hi, "<=")
    rows["joint:max_correlation_error"] = (
        dependence["max_correlation_error"], correlation_envelope["envelope"], "<=")
    rows["joint:energy_distance"] = (multivariate["observed"], multivariate["envelope"], "<=")

    out = {}
    for key, (observed, threshold, sense) in rows.items():
        ok = bool(observed <= threshold) if sense == "<=" else bool(observed >= threshold)
        out[key] = {"observed": float(observed), "threshold": float(threshold),
                    "sense": sense, "pass": ok}
    return {"checks": out, "pass": all(v["pass"] for v in out.values())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("b2", "b3"), required=True)
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, default=None)
    ap.add_argument("--diagnostics-only", action="store_true",
                    help="recompute diagnostics and the report from a saved chains.npz")
    a = ap.parse_args()

    spec = STAGES[a.stage]
    settings = SETTINGS[a.mode]
    active = spec["active"]
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)

    frozen = load_frozen_dataset()
    evaluator = RecurrentJointEvaluator(frozen.train, frozen.u_true, frozen.epsilon)
    fixed = {n: frozen.truth[n] for n in spec["fixed_from_truth"]}
    target = JointScalarTarget(evaluator, active, fixed)
    scales = {n: FROZEN_SCALES[n] for n in active}

    print(f"Stage {a.stage.upper()} [{a.mode}] — active {active}, fixed {fixed}")
    print(f"  frozen config hash {config_hash()}")
    print(f"  sweep order {[n for n in SWEEP_ORDER if n in active]}")
    print(f"  frozen Stage 6B1 scales {scales} (no pilot, no adaptation)")

    starts = dispersed_starts(active, settings["chains"])
    for c, start in enumerate(starts):
        evaluator.invalidate()
        parts = target.decompose(start, allow_cache=False)
        if not np.isfinite(parts["log_posterior"]):
            raise SystemExit(f"chain {c} start {start} has non-finite log posterior")
        print(f"  start {c}: " + "  ".join(f"{k} {v:.4f}" for k, v in start.items())
              + f"   log posterior {parts['log_posterior']:.2f}")

    if a.diagnostics_only:
        stored = np.load(out / "chains.npz")
        chains = {n: stored[f"draws_{n}"] for n in active}
        previous = json.loads((out / "config.json").read_text())
        if "acceptance_total" not in previous:
            raise SystemExit(
                f"{out / 'config.json'} predates the acceptance fields, so the diagnostics "
                "cannot be recomputed from it. Re-run without --diagnostics-only.")
        acceptance_total = previous["acceptance_total"]
        acceptance_post = previous["acceptance_post_burn_in"]
        wall = previous.get("wall_seconds", float("nan"))
        results = None
        print(f"  reusing saved chains: {chains[active[0]].shape[0]} chains x "
              f"{chains[active[0]].shape[1]} draws")
    else:
        results = []
        began = time.perf_counter()
        for c, start in enumerate(starts):
            evaluator.invalidate()
            r = run_joint_scalar_mcmc(target, start, scales, settings["iterations"],
                                      settings["burn_in"], settings["thin"],
                                      seed=spec["seed"] + c, chain=c)
            results.append(r)
            acc = r.acceptance(post_burn_in=True)
            print(f"    chain {c}: {r.runtime_seconds:6.1f}s  acceptance "
                  + "  ".join(f"{n} {acc[n]:.3f}" for n in active))
        wall = time.perf_counter() - began

        chains = {n: np.array([r.draws[n] for r in results]) for n in active}
        acceptance_total = {n: [r.acceptance(False)[n] for r in results] for n in active}
        acceptance_post = {n: [r.acceptance(True)[n] for r in results] for n in active}

        np.savez_compressed(out / "chains.npz",
                            **{f"draws_{n}": chains[n] for n in active},
                            log_likelihood=np.array([r.log_likelihood for r in results]),
                            log_posterior=np.array([r.log_posterior for r in results]))

    config = {"stage": a.stage, "mode": a.mode, "active": list(active),
              "fixed": jsonable(fixed), "sweep_order": [n for n in SWEEP_ORDER if n in active],
              "settings": settings, "proposal_scales": scales,
              "proposal_scale_provenance": "frozen Stage 6B1 registered scales; no pilot "
                                           "was run and no adaptation is retained",
              "starts": jsonable(starts),
              "start_provenance": "prior quantiles at levels "
                                  f"{list(START_LEVELS)} arranged by a fixed Latin square; "
                                  "no posterior or reference quantity is used",
              "seed_base": spec["seed"], "chain_seeds": [spec["seed"] + c
                                                         for c in range(settings["chains"])],
              "frozen_config_hash": config_hash(), "frozen_config": frozen_config(),
              "date": date.today().isoformat(),
              "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
              "git_commit": git("rev-parse", "HEAD"),
              "python": platform.python_version(), "numpy": np.__version__,
              "scipy": __import__("scipy").__version__,
              "wall_seconds": wall,
              "acceptance_total": jsonable(acceptance_total),
              "acceptance_post_burn_in": jsonable(acceptance_post),
              "evaluator": {"full_replay_calls": evaluator.full_replay_calls,
                            "cached_calls": evaluator.cached_calls,
                            "cache_builds": evaluator.cache_builds},
              "not_inferred": frozen_config()["never_inferred_in_stage_6b"]}
    (out / "config.json").write_text(json.dumps(jsonable(config), indent=2) + "\n")

    if a.mode == "smoke":
        checks = smoke_checks(results, target, active)
        checks["deterministic_resume"] = resume_check(
            target, starts[0], scales, spec["seed"], 400, 100, 2)
        summary = {"config": config, "checks": checks,
                   "acceptance_total": acceptance_total,
                   "acceptance_post_burn_in": acceptance_post,
                   "draws_per_chain": int(chains[active[0]].shape[1]),
                   "pass": all(v["pass"] for v in checks.values())}
        (out / "summary.json").write_text(json.dumps(jsonable(summary), indent=2) + "\n")
        lines = [f"# Stage {a.stage.upper()} smoke — joint scalar sampler", "",
                 f"Active: `{', '.join(active)}`; fixed: `{fixed}`.", "",
                 "| check | result |", "|---|---|"]
        for name, value in checks.items():
            lines.append(f"| {name} | **{'PASS' if value['pass'] else 'FAIL'}** |")
        lines += ["", "| coordinate | post burn-in acceptance |", "|---|---|"]
        for n in active:
            lines.append(f"| {n} | {', '.join(f'{v:.3f}' for v in acceptance_post[n])} |")
        (out / "report.md").write_text("\n".join(lines) + "\n")
        ok = summary["pass"]
        for name, value in checks.items():
            print(f"  [{'PASS' if value['pass'] else 'FAIL'}] {name}")
        print(f"\nStage {a.stage.upper()} smoke: {'PASS' if ok else 'FAIL'}  -> {out}")
        return 0 if ok else 1

    reference_dir = a.reference_dir or out
    grid, summary = load_reference(reference_dir, active)
    pooled_draws = int(chains[active[0]].size)
    reference_draws = sample_reference_draws(grid, 2 * pooled_draws + 4_000,
                                             seed=spec["seed"] + 777)

    marginals = marginal_diagnostics(chains, summary, frozen.truth,
                                     acceptance_total, acceptance_post)
    dependence = dependence_diagnostics(chains, summary, active)
    multivariate = multivariate_comparison(chains, reference_draws, summary, active,
                                           seed=spec["seed"] + 999)
    # The MCMC correlation estimate carries the noise of its EFFECTIVE sample size, not
    # of its raw draw count, so the null is calibrated at the smallest bulk ESS. Using the
    # raw count would demand of the sampler a precision its own autocorrelation denies it.
    effective = int(min(marginals[n]["bulk_ess"] for n in active))
    correlation_envelope = calibrate_correlation_envelope(
        reference_draws, n_draws=effective, n_replicates=40, seed=spec["seed"] + 555)
    correlation_envelope["calibrated_at"] = "minimum bulk ESS across active coordinates"
    correlation_envelope["pooled_draws"] = pooled_draws
    recovery = recovery_table(chains, frozen.truth)
    gates = evaluate_gates(marginals, dependence, multivariate, correlation_envelope,
                           active)

    (out / "scalar_diagnostics.json").write_text(json.dumps(
        jsonable({"marginals": marginals, "gates": gates}), indent=2) + "\n")
    (out / "joint_diagnostics.json").write_text(json.dumps(
        jsonable({"dependence": dependence, "multivariate": multivariate,
                  "correlation_envelope": correlation_envelope}), indent=2) + "\n")
    (out / "recovery_results.json").write_text(json.dumps(
        jsonable({"recovery": recovery,
                  "note": "synthetic recovery on one finite dataset; distinct from "
                          "sampler correctness, which is the reference comparison"}),
        indent=2) + "\n")

    write_full_report(out, a.stage, active, fixed, config, summary, marginals,
                      dependence, multivariate, correlation_envelope, recovery, gates,
                      reference_dir)

    print(f"\n  {'param':<13}{'ref mean':>10}{'MCMC mean':>11}{'std err':>9}{'KS':>8}"
          f"{'R-hat':>9}{'bulk ESS':>10}{'tail ESS':>10}")
    for n in active:
        d = marginals[n]
        print(f"  {n:<13}{d['reference']['mean']:>10.5f}{d['mcmc']['mean']:>11.5f}"
              f"{d['standardized_mean_error']:>9.4f}{d['ks_distance']:>8.4f}"
              f"{d['rhat']:>9.4f}{d['bulk_ess']:>10.0f}{d['tail_ess']:>10.0f}")
    print(f"\n  max correlation error {dependence['max_correlation_error']:.5f} "
          f"(envelope {correlation_envelope['envelope']:.5f})")
    print(f"  energy distance {multivariate['observed']:.6f} "
          f"(envelope {multivariate['envelope']:.6f}, null mean "
          f"{multivariate['null_mean']:.6f}, z {multivariate['z_score']:+.2f})")
    failed = [k for k, v in gates["checks"].items() if not v["pass"]]
    print(f"\n  gates: {len(gates['checks']) - len(failed)}/{len(gates['checks'])} pass"
          + (f"   FAILED: {failed}" if failed else ""))
    print(f"Stage {a.stage.upper()} full: {'PASS' if gates['pass'] else 'FAIL'}  -> {out}")
    return 0 if gates["pass"] else 1


def write_full_report(out, stage, active, fixed, config, summary, marginals, dependence,
                      multivariate, correlation_envelope, recovery, gates, reference_dir):
    tag = stage.upper()
    L = [f"# Stage 6{tag} — joint scalar MCMC against the independent {len(active)}-D reference",
         "",
         f"Date {config['date']} · branch `{config['git_branch']}` · commit "
         f"`{config['git_commit'][:8]}`",
         f"Frozen model `{config['frozen_config_hash'][:16]}` · Python {config['python']} "
         f"· NumPy {config['numpy']} · SciPy {config['scipy']}", "",
         f"**Target** — `p({', '.join(active)} | observations, U_TRUE, fixed boundaries, "
         f"epsilon = {config['frozen_config']['epsilon']}"
         + (f", lambda_back = {fixed['lambda_back']}" if fixed else "") + ")**",
         "",
         f"proportional to the complete recurrent likelihood times "
         f"{' × '.join('p(' + n + ')' for n in active)}. No interaction priors.", "",
         f"Sweep order: `{' -> '.join(config['sweep_order'])}`, each coordinate seeing the "
         "most recently accepted values of those before it.", "",
         "`U`, `rho`, `P`, segmentation boundaries, skill labels and `epsilon` are not "
         "inferred anywhere in Stage 6B.", "",
         "## Definition of done", "",
         "| parameter | ref mean | MCMC mean | ref 95% | MCMC 95% | R-hat | bulk ESS | "
         "tail ESS | MCSE | KS | result |",
         "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|"]
    for n in active:
        d = marginals[n]
        r, m = d["reference"], d["mcmc"]
        ok = all(v["pass"] for k, v in gates["checks"].items() if k.startswith(f"{n}:"))
        L.append(f"| {n} | {r['mean']:.5f} | {m['mean']:.5f} | "
                 f"[{r['q025']:.5f}, {r['q975']:.5f}] | [{m['q025']:.5f}, {m['q975']:.5f}] | "
                 f"{d['rhat']:.4f} | {d['bulk_ess']:.0f} | {d['tail_ess']:.0f} | "
                 f"{d['mcse_mean']:.5f} | {d['ks_distance']:.4f} | "
                 f"**{'PASS' if ok else 'FAIL'}** |")

    L += ["", "## Registered gates", "",
          "Stage 6B1's scalar gates, applied unchanged — not loosened because sampling is "
          "now joint — plus two calibrated joint gates.", "",
          "| check | observed | threshold | result |", "|---|---:|---:|---|"]
    for key, v in gates["checks"].items():
        L.append(f"| {key} | {v['observed']:.5f} | {v['threshold']:.5f} | "
                 f"{'PASS' if v['pass'] else 'FAIL'} |")

    L += ["", "## Dependence — the part four marginals cannot establish", "",
          "| pair | reference corr | MCMC corr | abs error |", "|---|---:|---:|---:|"]
    for pair, v in dependence["pairs"].items():
        L.append(f"| {pair} | {v['reference_correlation']:+.5f} | "
                 f"{v['mcmc_correlation']:+.5f} | {v['absolute_error']:.5f} |")
    L += ["",
          f"Max pairwise correlation error **{dependence['max_correlation_error']:.5f}** "
          f"against a calibrated envelope of {correlation_envelope['envelope']:.5f} "
          f"(99th percentile of reference-vs-reference at the same sample size; null mean "
          f"{correlation_envelope['mean']:.5f}, sd {correlation_envelope['sd']:.5f}).",
          "",
          f"Max normalised covariance error "
          f"{dependence['max_normalised_covariance_error']:.5f}.", "",
          "## Multivariate comparison", "",
          f"- statistic: {multivariate['statistic']}",
          f"- observed: **{multivariate['observed']:.6f}**",
          f"- calibrated envelope ({multivariate['envelope_quantile']:.0%} of "
          f"{multivariate['n_replicates']} reference-vs-reference replicates): "
          f"**{multivariate['envelope']:.6f}**",
          f"- null mean {multivariate['null_mean']:.6f}, sd {multivariate['null_sd']:.6f}, "
          f"max {multivariate['null_max']:.6f}; observed z = {multivariate['z_score']:+.2f}",
          f"- sample sizes: {multivariate['n_mcmc']} MCMC vs "
          f"{multivariate['n_reference']} reference",
          f"- **{'PASS' if multivariate['pass'] else 'FAIL'}**", "",
          "## Acceptance by coordinate", "",
          "| coordinate | total (per chain) | post burn-in (per chain) |", "|---|---|---|"]
    for n in active:
        d = marginals[n]
        L.append(f"| {n} | {', '.join(f'{v:.3f}' for v in d['acceptance_total'])} | "
                 f"{', '.join(f'{v:.3f}' for v in d['acceptance_post_burn_in'])} |")

    L += ["", "## Synthetic recovery — reported separately from sampler correctness", "",
          "Whether the generating value falls inside the posterior is a statement about "
          "this one finite dataset, not about the sampler. The sampler-correctness claim "
          "is the reference comparison above.", "",
          "| parameter | truth | post. mean | post. median | post. sd | 95% interval | "
          "truth inside | abs error | error in sd |",
          "|---|---:|---:|---:|---:|---|---|---:|---:|"]
    for n in active:
        r = recovery[n]
        L.append(f"| {n} | {r['true_value']:.5f} | {r['posterior_mean']:.5f} | "
                 f"{r['posterior_median']:.5f} | {r['posterior_sd']:.5f} | "
                 f"[{r['q025']:.5f}, {r['q975']:.5f}] | "
                 f"{'yes' if r['truth_in_95_interval'] else '**no**'} | "
                 f"{r['absolute_error']:.5f} | {r['error_in_posterior_sd']:.3f} |")

    L += ["", "## Reference provenance", "",
          f"- built by `scripts/stage6b_joint_reference_build.py` into `{reference_dir}`",
          f"- {summary['grid_points']:,} grid points, n = {summary['n_points']} per axis, "
          f"radius {summary['radius']:.1f} curvature sd",
          f"- integral {summary['integral_check']:.10f}; outer-face mass "
          f"{summary['outer_face_mass']:.3e}",
          "- direct quadrature on the recurrent log posterior; no MCMC kernel, acceptance "
          "ratio or draw is involved", "",
          "## Status", "",
          f"Stage 6{tag} sampler correctness: "
          f"**{'PASS' if gates['pass'] else 'FAIL'}**.",
          f"Stage 6{tag} synthetic recovery: "
          + ", ".join(f"{n} {'PASS' if recovery[n]['truth_in_95_interval'] else 'FAIL'}"
                      for n in active) + ".", ""]
    (out / "report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
