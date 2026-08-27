"""Stage 6B1 — four separate scalar MH samplers, checked against the Stage 6B0 grids.

    PYTHONPATH=src python scripts/stage6b_scalar_mcmc.py --mode smoke \
        --output-dir results/mcmc_original/stage6b1_smoke_seed0
    PYTHONPATH=src python scripts/stage6b_scalar_mcmc.py --mode full \
        --output-dir results/mcmc_original/stage6b1_full_seed0

One parameter at a time. ``U``, ``epsilon`` and the other three scalars are held at their
registered true values, so each run targets a posterior that Stage 6B0 already knows by
numerical integration. There is no U-update, no rho-update, no latent-dimension move and
no dimension-proportional schedule: those belong to Stage 6C, and adding them here would
destroy the only thing this stage can prove.

Stage 5C (full joint S+U+P) remains DEFERRED.
"""
from __future__ import annotations

import argparse, json, math, platform, subprocess, sys, time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import recurrent_synthetic as rs                          # noqa: E402
from hpop.mcmc_original.recurrent_scalar_mcmc import (                            # noqa: E402
    PILOT_SEED_OFFSET, PROPOSAL_KIND, REGISTERED_STARTS, SMOKE_STARTS,
    RecurrentScalarTarget, ScalarMHConfig, blockwise_recurrent_log_likelihood,
    build_proposal, curvature_proposal_scale, run_scalar_mh, tune_proposal_scale,
)
from hpop.mcmc_original.recurrent_scalar_posterior import PRIORS                  # noqa: E402
from hpop.mcmc_original.stage6b_mcmc_diagnostics import (                         # noqa: E402
    compare_to_reference, evaluate_gates, grid_summary, kappa_grid_from_omega,
    load_reference_posteriors,
)

PARAMETERS = ("beta", "omega", "lambda_rep", "lambda_back")
REFERENCE_PATH = ROOT / "results/mcmc_original/stage6b_full_seed0/reference_posteriors.json"

# Registered MCMC settings. `smoke` exists only to catch implementation failures.
SETTINGS = {
    "smoke": {"chains": 2, "iterations": 6_000, "burn_in": 1_000, "thin": 2,
              "starts": SMOKE_STARTS},
    "full":  {"chains": 4, "iterations": 20_000, "burn_in": 4_000, "thin": 2,
              "starts": REGISTERED_STARTS},
}
PILOT_ITERATIONS = 2_000
PREDICTIVE_DRAWS = 200


def jsonable(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.floating, np.integer)): return v.item()
    if isinstance(v, (np.bool_, bool)): return bool(v)
    if isinstance(v, dict): return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)): return [jsonable(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v): return None
    return v


def git(*a):
    try:
        return subprocess.run(("git",) + a, cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------------- held-out (10)
def held_out_diagnostics(parameter, draws, heldout_array, u, truth, epsilon, rng):
    """Secondary reporting only. Never used to choose a proposal scale or a prior.

    Each row substitutes a single value for ``parameter`` into the registered true
    parameter vector, so the numbers isolate that one scalar.
    """
    def nll_per_step(value):
        kw = dict(truth); kw[parameter] = float(value)
        total = float(blockwise_recurrent_log_likelihood(
            heldout_array, u, kw["beta"], epsilon, kw["omega"],
            kw["lambda_rep"], kw["lambda_back"]).sum())
        return -total / heldout_array.size

    pooled = np.asarray(draws).ravel()
    take = np.linspace(0, pooled.size - 1, min(PREDICTIVE_DRAWS, pooled.size)).astype(int)
    subsample = pooled[take]

    per_block = np.empty((subsample.size, heldout_array.shape[0]))
    for i, value in enumerate(subsample):
        kw = dict(truth); kw[parameter] = float(value)
        per_block[i] = blockwise_recurrent_log_likelihood(
            heldout_array, u, kw["beta"], epsilon, kw["omega"],
            kw["lambda_rep"], kw["lambda_back"])
    shift = per_block.max(axis=0)
    predictive_per_block = shift + np.log(np.exp(per_block - shift).mean(axis=0))

    return {
        "n_blocks": int(heldout_array.shape[0]), "n_steps": int(heldout_array.size),
        "nll_at_posterior_mean": nll_per_step(pooled.mean()),
        "nll_at_posterior_median": nll_per_step(float(np.median(pooled))),
        "nll_at_true_value": nll_per_step(truth[parameter]),
        "posterior_predictive_log_score_per_step":
            float(predictive_per_block.sum()) / heldout_array.size,
        "predictive_draws_used": int(subsample.size),
        "note": "diagnostic only; not used to tune proposals or priors",
    }


# ------------------------------------------------------------------------------ one run
def run_parameter(parameter, mode, train, heldout, u, truth, epsilon, reference,
                  base_seed, skip_pilot):
    settings = SETTINGS[mode]
    start_values = settings["starts"][parameter][: settings["chains"]]
    target = RecurrentScalarTarget(parameter, train, u, truth, epsilon)

    seed_root = base_seed + 1000 * PARAMETERS.index(parameter)
    curvature = curvature_proposal_scale(target, truth[parameter])
    if skip_pilot:
        tuning = {"initial_scale": curvature["scale"], "adjusted": False,
                  "final_scale": curvature["scale"], "pilot_acceptance_rate": None,
                  "pilot_iterations": 0, "skipped": True}
    else:
        tuning = tune_proposal_scale(target, curvature["scale"], start_values[0],
                                     seed_root + PILOT_SEED_OFFSET, PILOT_ITERATIONS)
        tuning["skipped"] = False
    tuning["curvature"] = curvature
    scale = tuning["final_scale"]

    chains, records = [], []
    calls_before = target.calls
    wall = time.perf_counter()
    for c, start in enumerate(start_values):
        config = ScalarMHConfig(parameter, scale, float(start), settings["iterations"],
                                settings["burn_in"], settings["thin"], seed_root + c)
        result = run_scalar_mh(config, target, build_proposal(parameter, scale))
        chains.append(result.samples)
        records.append({
            "chain": c, "initial_value": float(start), "seed": config.seed,
            "acceptance_rate": result.acceptance_rate,
            "post_burn_in_acceptance_rate": result.post_burn_in_acceptance_rate,
            "n_kept": int(result.samples.size), "runtime_seconds": result.runtime_seconds,
            "mean": float(result.samples.mean()), "sd": float(result.samples.std(ddof=1)),
        })
        print(f"    chain {c}: start {start:>7.3f}  acc {result.post_burn_in_acceptance_rate:.3f}"
              f"  mean {result.samples.mean():.5f}  {result.runtime_seconds:.1f}s")
    wall = time.perf_counter() - wall

    chains = np.array(chains)
    rates = [r["post_burn_in_acceptance_rate"] for r in records]
    grid = grid_summary(reference["posteriors"][parameter])
    comparison = compare_to_reference(chains, grid)
    gates = evaluate_gates(comparison, rates, mode=mode)

    out = {"parameter": parameter, "proposal_kind": PROPOSAL_KIND[parameter],
           "prior": PRIORS[parameter], "settings": {k: v for k, v in settings.items()
                                                    if k != "starts"},
           "start_values": [float(s) for s in start_values],
           "tuning": tuning, "proposal_scale": float(scale),
           "chains": records, "comparison": comparison, "gates": gates,
           "likelihood_evaluations": int(target.calls - calls_before),
           "wall_seconds": wall,
           "evaluation_path": "cached state features" if target.cached else "full replay"}

    if parameter == "omega":
        kappa_chains = 1.0 / (1.0 + np.exp(-chains))
        kappa_grid = kappa_grid_from_omega(reference["posteriors"]["omega"])
        kappa_comparison = compare_to_reference(kappa_chains, kappa_grid)
        out["kappa"] = {"comparison": kappa_comparison,
                        "gates": evaluate_gates(kappa_comparison, rates, mode=mode)}
        out["kappa_chains"] = kappa_chains

    if heldout is not None and heldout.size:
        out["held_out"] = held_out_diagnostics(parameter, chains, heldout, u, truth,
                                               epsilon, np.random.default_rng(seed_root))
    return out, chains


# --------------------------------------------------------------------------------- report
def write_report(path, mode, results, config, reference):
    L = [f"# Stage 6B1 — scalar MCMC against the Stage 6B0 reference posteriors ({mode})", "",
         f"Date {config['date']} · branch `{config['git_branch']}` · commit `{config['git_commit'][:8]}`",
         f"Python {config['python']} · NumPy {config['numpy']} · SciPy {config['scipy']}", "",
         "Each parameter is inferred **alone**, with `U`, `epsilon` and the other three",
         "scalars held at their registered true values. No `U` / `rho` / latent-dimension",
         "updates, and no dimension-proportional schedule — those are Stage 6C.",
         "**Stage 5C (full joint S+U+P) remains DEFERRED.**", "",
         "The claim being tested is not *truth in interval*. It is",
         "`p_MCMC(theta | D) ~= p_grid(theta | D)`, measured by standardized mean, median",
         "and interval-endpoint errors and a KS distance against the immutable normalized",
         "reference CDF in `stage6b_full_seed0/reference_posteriors.json`.", "",
         "## Definition of done", "",
         "| parameter | grid mean | MCMC mean | grid 95% | MCMC 95% | R-hat | bulk ESS | tail ESS | KS | result |",
         "|---|---:|---:|---|---|---:|---:|---:|---:|---|"]

    def row(name, comparison, gates):
        g, m = comparison["grid"], comparison["mcmc"]
        return (f"| {name} | {g['mean']:.4f} | {m['mean']:.4f} | "
                f"[{g['q025']:.4f}, {g['q975']:.4f}] | [{m['q025']:.4f}, {m['q975']:.4f}] | "
                f"{comparison['rhat']:.4f} | {comparison['bulk_ess']:.0f} | "
                f"{comparison['tail_ess']:.0f} | {comparison['ks_distance']:.4f} | "
                f"**{'PASS' if gates['pass'] else 'FAIL'}** |")

    for name in PARAMETERS:
        r = results[name]
        L.append(row(name, r["comparison"], r["gates"]))
        if name == "omega":
            L.append(row("kappa = sigmoid(omega)", r["kappa"]["comparison"], r["kappa"]["gates"]))
    L += ["", "## Registered gates", ""]
    thresholds = ("standardized mean <= 0.15, median <= 0.15, interval endpoints <= 0.25, "
                  "KS <= 0.03, R-hat <= 1.01, bulk ESS >= 1000, tail ESS >= 500, "
                  "acceptance in [0.15, 0.60]") if mode == "full" else (
                  "R-hat < 1.05, standardized mean <= 0.35 (smoke catches implementation "
                  "failures only)")
    L += [thresholds, "", "| parameter | check | observed | threshold | result |", "|---|---|---:|---:|---|"]
    for name in PARAMETERS:
        for check, r in results[name]["gates"]["checks"].items():
            L.append(f"| {name} | {check} | {r['observed']:.4f} | {r['threshold']:.4f} | "
                     f"{'PASS' if r['pass'] else 'FAIL'} |")

    L += ["", "## Proposals and tuning", "",
          "Proposal scales come from the observed likelihood curvature at the registered",
          "true value, then one 2,000-iteration pilot with at most one adjustment. The",
          "pilot draws are discarded and the reported chains restart from the registered",
          "dispersed starts. The reference grids are never used to tune anything.", "",
          "| parameter | walk | initial scale | pilot acc | adjusted | final scale | evaluations | seconds |",
          "|---|---|---:|---:|---|---:|---:|---:|"]
    for name in PARAMETERS:
        r, t = results[name], results[name]["tuning"]
        acc = "n/a" if t["pilot_acceptance_rate"] is None else f"{t['pilot_acceptance_rate']:.3f}"
        L.append(f"| {name} | {r['proposal_kind']} | {t['initial_scale']:.5f} | {acc} | "
                 f"{t['adjusted']} | {t['final_scale']:.5f} | {r['likelihood_evaluations']:,} | "
                 f"{r['wall_seconds']:.1f} |")

    L += ["", "The `log theta' - log theta` Jacobian on the log-scale walk is verified",
          "analytically in `tests/mcmc_original/test_stage6b1_proposals.py`, and its removal",
          "is shown to move a Gamma(2,2) stationary distribution onto Gamma(1,2) in",
          "`test_stage6b1_scalar_mh.py`. `omega` uses a symmetric walk and carries no",
          "correction; every proposed `omega` replays `q` from zero for all blocks.", "",
          "## Per-chain detail", "",
          "| parameter | chain | start | acceptance (post burn-in) | mean | sd | seconds |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name in PARAMETERS:
        for c in results[name]["chains"]:
            L.append(f"| {name} | {c['chain']} | {c['initial_value']:.3f} | "
                     f"{c['post_burn_in_acceptance_rate']:.3f} | {c['mean']:.5f} | "
                     f"{c['sd']:.5f} | {c['runtime_seconds']:.1f} |")

    if any("held_out" in results[n] for n in PARAMETERS):
        L += ["", "## Held-out diagnostics (secondary)", "",
              "Reported after fitting, and used for nothing else — not for proposal scales,",
              "not for priors. Per-step negative log likelihood on the 200 held-out blocks.", "",
              "| parameter | NLL at post. mean | NLL at post. median | NLL at truth | predictive log score |",
              "|---|---:|---:|---:|---:|"]
        for name in PARAMETERS:
            h = results[name].get("held_out")
            if h:
                L.append(f"| {name} | {h['nll_at_posterior_mean']:.5f} | "
                         f"{h['nll_at_posterior_median']:.5f} | {h['nll_at_true_value']:.5f} | "
                         f"{h['posterior_predictive_log_score_per_step']:.5f} |")

    passed = all(results[n]["gates"]["pass"] for n in PARAMETERS) and \
        results["omega"]["kappa"]["gates"]["pass"]
    L += ["", "## Status", "",
          f"Stage 6B1: **{'PASS' if passed else 'FAIL'}** ({mode}).", ""]
    if mode == "full" and passed:
        L += ["Next: Stage 6B2 — joint `beta`, `omega`, `lambda_rep` with `lambda_back`",
              "fixed at 0.25, which is where posterior *correlation* first appears.", ""]
    Path(path).write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke",
                    help="MCMC settings; the dataset is always the registered 500-block corpus")
    ap.add_argument("--generator-seed", type=int, default=0)
    ap.add_argument("--mcmc-seed", type=int, default=20_250_811)
    ap.add_argument("--parameters", nargs="+", choices=PARAMETERS, default=list(PARAMETERS))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    ap.add_argument("--skip-pilot", action="store_true")
    a = ap.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)

    reference = load_reference_posteriors(a.reference)
    data = rs.generate_recurrent_dataset("full", a.generator_seed)
    train = np.array([b.roles for b in data.train], dtype=int)
    heldout = np.array([b.roles for b in data.heldout], dtype=int)
    truth = dict(reference["truth"])
    epsilon = float(reference["epsilon"])

    if train.shape[0] != reference["n_train_blocks"] or train.shape[1] != reference["T"]:
        raise SystemExit(f"dataset {train.shape} does not match the reference "
                         f"({reference['n_train_blocks']}, {reference['T']})")

    print(f"Stage 6B1 [{a.mode}] — {train.shape[0]} training blocks of T={train.shape[1]}, "
          f"epsilon {epsilon}, U held at U_TRUE")
    results = {}
    for parameter in a.parameters:
        print(f"\n  {parameter}  (prior {PRIORS[parameter]['family']}, "
              f"{PROPOSAL_KIND[parameter]} walk)")
        record, chains = run_parameter(parameter, a.mode, train, heldout, data.u_true,
                                       truth, epsilon, reference, a.mcmc_seed, a.skip_pilot)
        results[parameter] = record
        payload = {"samples": chains}
        if "kappa_chains" in record:
            payload["kappa"] = record.pop("kappa_chains")
        np.savez_compressed(out / f"chains_{parameter}.npz", **payload)
        c = record["comparison"]
        verdict = "PASS" if record["gates"]["pass"] else "FAIL"
        print(f"    -> mean {c['mcmc']['mean']:.5f} vs grid {c['grid']['mean']:.5f}"
              f"   std.err {c['standardized_mean_error']:.3f}   KS {c['ks_distance']:.4f}"
              f"   R-hat {c['rhat']:.4f}   bulk ESS {c['bulk_ess']:.0f}   [{verdict}]")

    config = {"date": date.today().isoformat(),
              "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
              "git_commit": git("rev-parse", "HEAD"),
              "python": platform.python_version(), "numpy": np.__version__,
              "scipy": __import__("scipy").__version__,
              "mode": a.mode, "generator_seed": a.generator_seed, "mcmc_seed": a.mcmc_seed,
              "reference": str(a.reference), "epsilon": epsilon, "truth": truth,
              "u_true": data.u_true.tolist(), "settings": SETTINGS[a.mode]["starts"],
              "stage_5c": "DEFERRED", "u_inference": False, "rho_inference": False,
              "latent_dimension_moves": False,
              "note": "one scalar at a time; the other three held at their true values"}

    (out / "config.json").write_text(json.dumps(jsonable(config), indent=2) + "\n")
    (out / "stage6b1_mcmc.json").write_text(json.dumps(
        jsonable({"config": config, "results": results}), indent=2) + "\n")
    write_report(out / "stage6b1_report.md", a.mode, results, config, reference)

    ok = all(results[p]["gates"]["pass"] for p in a.parameters)
    if "omega" in results:
        ok = ok and results["omega"]["kappa"]["gates"]["pass"]
    print(f"\nOutputs: {out}\nReport : {out / 'stage6b1_report.md'}")
    print(f"Stage 6B1 [{a.mode}]: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
