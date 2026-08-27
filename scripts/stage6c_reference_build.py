"""Stage 6C — build and freeze the independent exact references.

Run this before any formal chain. It touches no MCMC code: the only imports are the
frozen model, the exact-reference module and the Stage 2A structural prior. Nothing here
reads a chain, an acceptance ratio or a transition kernel.

    p(P, rho | Y)        = p(rho) · L(P) · pi_rho(P) / Z          (Stage 6C1)
    p(P, rho, beta | Y)  = p(rho) p(beta) · L(P, beta) · pi_rho(P) / Z   (Stage 6C2)

`P` ranges over the complete catalogue of labelled partial orders reachable by a
2-column product order on 5 elements, `L` is the exact frozen Stage 6B recurrent
likelihood, and

    pi_rho(P) = P_{U ~ N(0, Sigma_rho)}[h(U) = P]

is the prior cell mass — the single Monte Carlo ingredient, computed from the **prior
alone** with common random numbers across the rho grid, and reported with its standard
error everywhere it feeds a gate.

This is a marginal of the continuous target, not a replacement for it: the likelihood is
constant on {U : h(U) = P}, so the integral of p(Y|U) p(U|rho) over that cell factorises
exactly. See results/mcmc_original/stage6c_complete/model_audit.md.

Phases (`--phase`):
    masses      prior cell masses only, cached to NPZ (the long pole, ~45 min)
    reference   both references from a cached mass table
    all         both, in order (default)
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from hpop.mcmc_original.sampler_u import log_u_prior, sigma_rho_matrix
from hpop.mcmc_original.stage6c_exact_reference import (
    build_catalogue, build_6c1_reference, build_6c2_reference, is_partial_order,
    poset_log_likelihoods, poset_log_likelihood_beta_table, prior_cell_masses,
    reference_summary, transitive_reduction,
)
from hpop.mcmc_original.stage6c_frozen import (
    RHO_UPPER, config_hash, load_stage6c_dataset, log_det_sigma_rho, log_rho_prior,
)
from hpop.mcmc_original.latent_poset import precedence_from_u

# ---------------------------------------------------------------- registered settings
# Registered BEFORE any reference was inspected and before any formal chain was run.
N_RHO = 81
RHO_LO = 1e-3
RHO_HI = 0.994                    # inside the truncation point RHO_UPPER = 0.995
N_BETA = 241
BETA_LO = 1.0
BETA_HI = 2.2
N_MASS_DRAWS = 40_000_000
MASS_SEED = 20250811
MASS_CHUNK = 250_000

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results" / "mcmc_original"


def rho_grid() -> np.ndarray:
    return np.linspace(RHO_LO, RHO_HI, N_RHO)


def beta_grid() -> np.ndarray:
    return np.linspace(BETA_LO, BETA_HI, N_BETA)


def source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:                                        # pragma: no cover
        return "unknown"


def provenance() -> dict:
    return {"source_commit": source_commit(),
            "stage6c_config_hash": config_hash(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform()}


# ------------------------------------------------------------------ catalogue artifact
def save_catalogue(catalogue, path: Path) -> None:
    np.savez_compressed(
        path,
        closures=catalogue.closures, reductions=catalogue.reductions,
        relation_counts=catalogue.relation_counts,
        representatives=catalogue.representatives,
        ranking_tuple_counts=catalogue.ranking_tuple_counts,
        keys=catalogue.keys, order=catalogue.order,
        m=np.array(catalogue.m), d=np.array(catalogue.d))


def validate_catalogue(catalogue) -> dict:
    """Independent revalidation of the enumerator — not a restatement of its own logic."""
    keys = catalogue.keys
    duplicates = int(keys.size - np.unique(keys).size)
    all_valid = all(is_partial_order(p) for p in catalogue.closures)
    # closure -> reduction -> closure must round-trip
    round_trip = 0
    for closure, reduction in zip(catalogue.closures, catalogue.reductions):
        reach = reduction.astype(bool).copy()
        for _ in range(catalogue.m):
            reach |= (reach.astype(int) @ reduction.astype(int)) > 0
        if np.array_equal(reach, closure.astype(bool)):
            round_trip += 1
    # every representative really does induce the order it is filed under
    representative_ok = all(
        np.array_equal(precedence_from_u(u), closure)
        for u, closure in zip(catalogue.representatives, catalogue.closures))
    return {
        "size": catalogue.size,
        "expected_labelled_posets_on_5": 4231,
        "matches_expected": catalogue.size == 4231,
        "duplicate_keys": duplicates,
        "all_are_partial_orders": bool(all_valid),
        "closure_reduction_round_trip": int(round_trip),
        "round_trip_complete": round_trip == catalogue.size,
        "representatives_induce_filed_order": bool(representative_ok),
        "ranking_tuples_total": int(catalogue.ranking_tuple_counts.sum()),
        "ranking_tuples_expected": math.factorial(catalogue.m) ** catalogue.d,
    }


# ------------------------------------------------------------- structural prior audit
def structural_prior_audit(m: int, d: int) -> dict:
    """Numbers behind §2.1: the rho-dependent Gaussian normaliser is present and exact."""
    from scipy import stats

    rng = np.random.default_rng(12345)
    grid = np.linspace(0.05, 0.95, 19)

    scipy_max_error = 0.0
    closed_form_max_error = 0.0
    for rho in grid:
        sigma = sigma_rho_matrix(d, float(rho))
        u = rng.normal(size=(m, d)) @ np.linalg.cholesky(sigma).T
        mine = log_u_prior(u, float(rho))
        theirs = float(stats.multivariate_normal(mean=np.zeros(d), cov=sigma)
                       .logpdf(u).sum())
        scipy_max_error = max(scipy_max_error, abs(mine - theirs))
        _, slogdet = np.linalg.slogdet(sigma)
        closed_form_max_error = max(
            closed_form_max_error, abs(slogdet - log_det_sigma_rho(d, float(rho))))

    # Normalisation, exactly. The joint over m rows is the product of the row densities,
    # so it is enough to integrate ONE row by high-accuracy 2-D quadrature and to confirm
    # the factorisation; a 10-dimensional Monte Carlo estimate would be far weaker.
    from scipy import integrate

    integrates_to = {}
    factorisation_max_error = 0.0
    for rho in (0.1, 0.5, 0.9):
        def row_density(x, y, rho=rho):
            return math.exp(log_u_prior(np.array([[x, y]]), rho))

        mass, _ = integrate.dblquad(row_density, -12.0, 12.0, lambda _: -12.0,
                                    lambda _: 12.0, epsabs=1e-11, epsrel=1e-11)
        integrates_to[str(rho)] = float(mass)
        u = rng.normal(size=(m, d))
        factorisation_max_error = max(factorisation_max_error, abs(
            log_u_prior(u, rho) - sum(log_u_prior(u[[i]], rho) for i in range(m))))

    # negative control: drop the m*log_det term and show the rho posterior moves
    def bad_log_u_prior(u, rho):
        sigma = sigma_rho_matrix(d, rho)
        quad = float(np.einsum("ij,jk,ik->", u, np.linalg.inv(sigma), u))
        return -0.5 * (m * d * math.log(2 * math.pi) + quad)

    u_fixed = rng.normal(size=(m, d))
    grid_fine = rho_grid()
    good = np.array([log_u_prior(u_fixed, float(r)) + log_rho_prior(float(r))
                     for r in grid_fine])
    bad = np.array([bad_log_u_prior(u_fixed, float(r)) + log_rho_prior(float(r))
                    for r in grid_fine])

    def posterior_mean(log_density):
        w = np.exp(log_density - log_density.max())
        return float(np.trapezoid(grid_fine * w, grid_fine)
                     / np.trapezoid(w, grid_fine))

    return {
        "scipy_mvn_max_abs_error": scipy_max_error,
        "closed_form_logdet_max_abs_error": closed_form_max_error,
        "single_row_quadrature_mass": integrates_to,
        "row_factorisation_max_abs_error": factorisation_max_error,
        "negative_control": {
            "description": "log_u_prior with the m*log|Sigma_rho| term deleted",
            "rho_posterior_mean_with_normaliser": posterior_mean(good),
            "rho_posterior_mean_without_normaliser": posterior_mean(bad),
            "shift": abs(posterior_mean(good) - posterior_mean(bad)),
        },
        "rho_dependent_normaliser": "-(m/2) log|Sigma_rho|, present in sampler_u.log_u_prior",
        "combinatorial_normaliser_needed": False,
    }


# ----------------------------------------------------------------------- mass phase
def run_masses(cache: Path) -> dict:
    catalogue = build_catalogue(5, 2)
    grid = rho_grid()
    print(f"[masses] {catalogue.size} posets x {grid.size} rho x "
          f"{N_MASS_DRAWS:,} draws", flush=True)
    began = time.perf_counter()
    masses = prior_cell_masses(catalogue, grid, N_MASS_DRAWS, MASS_SEED,
                               chunk=MASS_CHUNK)
    runtime = time.perf_counter() - began
    print(f"[masses] done in {runtime / 60:.1f} min, "
          f"max SE {masses['max_standard_error']:.3e}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, rho_grid=masses["rho_grid"], masses=masses["masses"],
        counts=masses["counts"], standard_error=masses["standard_error"],
        n_draws=np.array(masses["n_draws"]), seed=np.array(masses["seed"]),
        unseen_draws=np.array(masses["unseen_draws"]), runtime=np.array(runtime))
    return masses


def load_masses(cache: Path) -> dict:
    with np.load(cache) as z:
        out = {k: z[k] for k in z.files}
    out["n_draws"] = int(out["n_draws"])
    out["seed"] = int(out["seed"])
    out["unseen_draws"] = int(out["unseen_draws"])
    out["max_standard_error"] = float(out["standard_error"].max())
    out["provenance"] = "prior draws only: no data, no likelihood, no MCMC"
    return out


# ------------------------------------------------------------------ reference phase
def _jsonable(value):
    # bool before int: bool subclasses int, so pass/fail flags would serialise as 1/0.
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def summary_payload(summary: dict, catalogue, true_index: int) -> dict:
    """The reportable part of a reference summary — dense arrays stay in the NPZ."""
    probability = summary["poset_probability"]
    ranking = np.argsort(-probability)
    rank_of_true = int(np.where(ranking == true_index)[0][0]) + 1
    payload = {
        "n_posets": summary["n_posets"],
        "map_poset": summary["map_poset"],
        "map_probability": summary["map_probability"],
        "true_poset_index": int(true_index),
        "true_poset_probability": float(probability[true_index]),
        "true_poset_rank": rank_of_true,
        "map_is_true_poset": int(summary["map_poset"]) == int(true_index),
        "top5": [{"index": int(i), "probability": float(probability[i]),
                  "relations": int(catalogue.relation_counts[i])} for i in ranking[:5]],
        "rho": {k: v for k, v in summary["rho"].items() if k not in ("grid", "cdf")},
        "relation_marginal_min": float(summary["relation_marginal"].min()),
        "relation_marginal_max": float(summary["relation_marginal"].max()),
    }
    if "beta" in summary:
        payload["beta"] = {k: v for k, v in summary["beta"].items()
                           if k not in ("grid", "cdf")}
    return _jsonable(payload)


def coverage_audit(reference, summary) -> dict:
    """§11.1/§11.2: domain coverage, outer-boundary mass, quadrature, refinement."""
    out = {}
    density = summary["rho_marginal_density"]
    grid = reference.rho_grid
    total = float(np.trapezoid(density, grid))
    edge = float(np.trapezoid(density[:2], grid[:2]) + np.trapezoid(
        density[-2:], grid[-2:]))
    out["rho"] = {
        "grid_lo": float(grid[0]), "grid_hi": float(grid[-1]),
        "n_points": int(grid.size), "spacing": float(np.diff(grid).mean()),
        "integrates_to": total,
        "outer_boundary_mass_fraction": edge / total,
        "prior_truncation_point": RHO_UPPER,
    }
    if reference.beta_grid is not None:
        bdensity = summary["beta_marginal_density"]
        bgrid = reference.beta_grid
        btotal = float(np.trapezoid(bdensity, bgrid))
        bedge = float(np.trapezoid(bdensity[:2], bgrid[:2]) + np.trapezoid(
            bdensity[-2:], bgrid[-2:]))
        out["beta"] = {
            "grid_lo": float(bgrid[0]), "grid_hi": float(bgrid[-1]),
            "n_points": int(bgrid.size), "spacing": float(np.diff(bgrid).mean()),
            "integrates_to": btotal,
            "outer_boundary_mass_fraction": bedge / btotal,
        }
    return out


def refinement_audit(build, summary_full, half_kwargs) -> dict:
    """Rebuild on the half-resolution grid and report the movement in every headline."""
    coarse_reference = build(**half_kwargs)
    coarse = reference_summary(coarse_reference, {})
    out = {"rho_mean_fine": summary_full["rho"]["mean"],
           "rho_mean_coarse": coarse["rho"]["mean"],
           "rho_mean_abs_change": abs(summary_full["rho"]["mean"]
                                      - coarse["rho"]["mean"]),
           "rho_sd_abs_change": abs(summary_full["rho"]["sd"] - coarse["rho"]["sd"]),
           "poset_probability_max_abs_change": float(
               np.abs(summary_full["poset_probability"]
                      - coarse["poset_probability"]).max())}
    if "beta" in summary_full and "beta" in coarse:
        out["beta_mean_abs_change"] = abs(summary_full["beta"]["mean"]
                                          - coarse["beta"]["mean"])
        out["beta_sd_abs_change"] = abs(summary_full["beta"]["sd"]
                                        - coarse["beta"]["sd"])
    return out


def build_references(masses: dict) -> None:
    frozen = load_stage6c_dataset()
    catalogue = build_catalogue(5, 2)
    true_index = catalogue.index_of(precedence_from_u(frozen.u_true))
    grid = masses["rho_grid"]
    betas = beta_grid()

    catalogue_report = validate_catalogue(catalogue)
    prior_report = structural_prior_audit(catalogue.m, catalogue.d)
    mass_report = {
        "n_draws": masses["n_draws"], "seed": masses["seed"],
        "max_standard_error": masses["max_standard_error"],
        "unseen_draws": masses["unseen_draws"],
        "common_random_numbers": True,
        "pi_rho_true_poset": _jsonable(masses["masses"][:, true_index]),
        "pi_rho_true_poset_se": _jsonable(masses["standard_error"][:, true_index]),
        "provenance": "prior draws only: no data, no likelihood, no MCMC",
    }

    # -------------------------------------------------------------- Stage 6C1
    out1 = RESULTS / "stage6c1_u_rho_reference"
    out1.mkdir(parents=True, exist_ok=True)
    print("[6c1] exact L(P) for every catalogue entry ...", flush=True)
    began = time.perf_counter()
    reference1 = build_6c1_reference(catalogue, frozen.train, frozen.truth,
                                     frozen.epsilon, grid, masses)
    summary1 = reference_summary(reference1, frozen.truth)
    runtime1 = time.perf_counter() - began
    print(f"[6c1] built in {runtime1:.0f}s", flush=True)

    half = {k: v for k, v in dict(
        catalogue=catalogue, role_array=frozen.train, truth=frozen.truth,
        epsilon=frozen.epsilon, rho_grid=grid[::2],
        cell_masses={**masses, "masses": masses["masses"][::2],
                     "rho_grid": grid[::2]}).items()}
    refine1 = refinement_audit(build_6c1_reference, summary1, half)

    likelihood = reference1.log_joint  # keep the raw table too
    exact_likelihood = poset_log_likelihoods(
        catalogue, frozen.train, frozen.epsilon, frozen.truth["beta"],
        frozen.truth["omega"], frozen.truth["lambda_rep"], frozen.truth["lambda_back"])
    mle_index = int(np.argmax(exact_likelihood))
    gap = float(np.sort(exact_likelihood)[-1] - np.sort(exact_likelihood)[-2])

    save_catalogue(catalogue, out1 / "poset_catalogue.npz")
    np.savez_compressed(
        out1 / "exact_reference.npz",
        rho_grid=grid, log_joint=reference1.log_joint, joint=reference1.joint,
        poset_probability=summary1["poset_probability"],
        relation_marginal=summary1["relation_marginal"],
        reduction_marginal=summary1["reduction_marginal"],
        relation_count_distribution=summary1["relation_count_distribution"],
        rho_marginal_density=summary1["rho_marginal_density"],
        exact_log_likelihood=exact_likelihood,
        cell_masses=masses["masses"], cell_mass_se=masses["standard_error"])
    (out1 / "structural_prior_audit.json").write_text(json.dumps(
        _jsonable({"catalogue": catalogue_report, "structural_prior": prior_report,
                   "prior_cell_masses": mass_report,
                   "exact_likelihood": {
                       "mle_poset_index": mle_index,
                       "true_poset_index": int(true_index),
                       "mle_is_true_poset": mle_index == int(true_index),
                       "nats_clear_of_runner_up": gap}}), indent=2))
    (out1 / "exact_summary.json").write_text(json.dumps(
        {**summary_payload(summary1, catalogue, true_index),
         "coverage": _jsonable(coverage_audit(reference1, summary1)),
         "refinement": _jsonable(refine1)}, indent=2))
    (out1 / "config.json").write_text(json.dumps(_jsonable({
        **provenance(), "stage": "6C1 reference",
        "target": "p(P, rho | Y) proportional to p(rho) L(P) pi_rho(P)",
        "uses_mcmc": False,
        "rho_grid": {"lo": RHO_LO, "hi": RHO_HI, "n": N_RHO},
        "cell_masses": {"n_draws": N_MASS_DRAWS, "seed": MASS_SEED},
        "fixed_scalars": frozen.truth, "epsilon": frozen.epsilon,
        "runtime_seconds": runtime1}), indent=2))
    print(f"[6c1] wrote {out1}", flush=True)

    # -------------------------------------------------------------- Stage 6C2
    out2 = RESULTS / "stage6c2_u_rho_beta_reference"
    out2.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    # The exact L(P, beta) table is ~20 minutes of pure function evaluation and depends
    # only on the frozen corpus, the catalogue and the beta grid — none of which change
    # between rebuilds. Reuse it when it is already on disk for exactly this grid, so
    # regenerating the summaries is cheap and a rebuild is idempotent.
    table_path = out2 / "loglik_table.npz"
    table = None
    if table_path.exists():
        with np.load(table_path) as cached:
            if (cached["beta_grid"].shape == betas.shape
                    and np.allclose(cached["beta_grid"], betas)
                    and cached["log_likelihood"].shape[0] == catalogue.size):
                table = cached["log_likelihood"]
                print(f"[6c2] reusing cached L(P, beta) table from {table_path}",
                      flush=True)
    if table is None:
        print(f"[6c2] exact L(P, beta) table, {catalogue.size} x {betas.size} ...",
              flush=True)
        chunks = []
        for start in range(0, betas.size, 32):
            chunks.append(poset_log_likelihood_beta_table(
                catalogue, frozen.train, frozen.epsilon, betas[start:start + 32],
                frozen.truth["omega"], frozen.truth["lambda_rep"],
                frozen.truth["lambda_back"]))
            print(f"[6c2]   beta {start + chunks[-1].shape[1]}/{betas.size}", flush=True)
        table = np.concatenate(chunks, axis=1)
    reference2 = build_6c2_reference(catalogue, frozen.train, frozen.truth,
                                     frozen.epsilon, grid, betas, masses,
                                     log_likelihood_table=table)
    summary2 = reference_summary(reference2, frozen.truth)
    runtime2 = time.perf_counter() - began
    print(f"[6c2] built in {runtime2 / 60:.1f} min", flush=True)

    half2 = dict(catalogue=catalogue, role_array=frozen.train, truth=frozen.truth,
                 epsilon=frozen.epsilon, rho_grid=grid[::2], beta_grid=betas[::2],
                 cell_masses={**masses, "masses": masses["masses"][::2],
                              "rho_grid": grid[::2]},
                 log_likelihood_table=table[:, ::2])
    refine2 = refinement_audit(build_6c2_reference, summary2, half2)

    save_catalogue(catalogue, out2 / "poset_catalogue.npz")
    np.savez_compressed(out2 / "loglik_table.npz", beta_grid=betas,
                        log_likelihood=table)
    np.savez_compressed(
        out2 / "exact_reference.npz",
        rho_grid=grid, beta_grid=betas, joint=reference2.joint,
        poset_probability=summary2["poset_probability"],
        relation_marginal=summary2["relation_marginal"],
        reduction_marginal=summary2["reduction_marginal"],
        relation_count_distribution=summary2["relation_count_distribution"],
        rho_marginal_density=summary2["rho_marginal_density"],
        beta_marginal_density=summary2["beta_marginal_density"],
        cell_masses=masses["masses"], cell_mass_se=masses["standard_error"])
    (out2 / "exact_summary.json").write_text(json.dumps(
        {**summary_payload(summary2, catalogue, true_index),
         "coverage": _jsonable(coverage_audit(reference2, summary2)),
         "refinement": _jsonable(refine2)}, indent=2))
    (out2 / "config.json").write_text(json.dumps(_jsonable({
        **provenance(), "stage": "6C2 reference",
        "target": "p(P, rho, beta | Y) proportional to p(rho) p(beta) L(P, beta) "
                  "pi_rho(P)",
        "uses_mcmc": False,
        "rho_grid": {"lo": RHO_LO, "hi": RHO_HI, "n": N_RHO},
        "beta_grid": {"lo": BETA_LO, "hi": BETA_HI, "n": N_BETA},
        "beta_prior": "gamma(shape=2, rate=2), the frozen Stage 6B prior",
        "cell_masses": {"n_draws": N_MASS_DRAWS, "seed": MASS_SEED},
        "fixed_scalars": {k: frozen.truth[k]
                          for k in ("omega", "lambda_rep", "lambda_back")},
        "epsilon": frozen.epsilon, "runtime_seconds": runtime2}), indent=2))
    print(f"[6c2] wrote {out2}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("masses", "reference", "all"),
                        default="all")
    parser.add_argument("--mass-cache", type=Path,
                        default=RESULTS / "stage6c1_u_rho_reference" /
                        "prior_cell_masses.npz")
    args = parser.parse_args()

    if args.phase in ("masses", "all"):
        args.mass_cache.parent.mkdir(parents=True, exist_ok=True)
        run_masses(args.mass_cache)
    if args.phase in ("reference", "all"):
        build_references(load_masses(args.mass_cache))


if __name__ == "__main__":
    main()
