"""Stage 6B2 / 6B3 — joint-posterior diagnostics against the independent references.

Four matching one-dimensional marginals do not establish a correct joint posterior: they
are consistent with any copula. So the comparison here runs at three levels —

1. per-coordinate marginals (KS against the reference marginal CDF, means, endpoints);
2. dependence (covariance, correlation, all pairwise marginals);
3. the full joint, by a multivariate two-sample statistic.

The multivariate statistic is an **energy distance** on coordinates standardised by the
reference's own mean and sd. A raw distance means nothing on its own, so its threshold is
calibrated by repeatedly comparing two independent draws *from the reference itself* at
the same sample sizes. That null distribution is what "the sampler agrees with the
reference" is measured against; the envelope is not a number chosen by hand.
"""

from __future__ import annotations

import math

import numpy as np

from hpop.mcmc_original.stage6b_mcmc_diagnostics import (
    bulk_ess, ks_distance_to_grid, mcse_mean, rank_normalized_split_rhat, tail_ess,
)

__all__ = [
    "energy_distance", "calibrate_energy_envelope", "standardise",
    "marginal_diagnostics", "dependence_diagnostics", "multivariate_comparison",
    "calibrate_correlation_envelope",
    "recovery_table",
]


def standardise(sample: np.ndarray, centre, scale) -> np.ndarray:
    return (np.asarray(sample, dtype=float) - np.asarray(centre)) / np.asarray(scale)


def _pairwise_mean_distance(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> float:
    """Mean Euclidean distance between two point sets, chunked to bound memory."""
    total = 0.0
    for start in range(0, a.shape[0], chunk):
        block = a[start:start + chunk]
        total += float(np.sqrt(((block[:, None, :] - b[None, :, :]) ** 2).sum(-1)).sum())
    return total / (a.shape[0] * b.shape[0])


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """``2 E|X-Y| - E|X-X'| - E|Y-Y'|``. Zero exactly when the two laws coincide."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    return float(2.0 * _pairwise_mean_distance(x, y)
                 - _pairwise_mean_distance(x, x) - _pairwise_mean_distance(y, y))


def calibrate_energy_envelope(reference_draws: np.ndarray, n_x: int, n_y: int,
                              n_replicates: int, seed: int,
                              quantile: float = 0.99) -> dict:
    """Null distribution of the energy distance, reference against reference.

    Both sides are drawn from the frozen reference, at exactly the sample sizes used for
    the real comparison, so the envelope carries the same Monte Carlo noise the real
    statistic does.
    """
    rng = np.random.default_rng(seed)
    pool = np.asarray(reference_draws, dtype=float)
    values = np.empty(n_replicates)
    for r in range(n_replicates):
        idx = rng.permutation(pool.shape[0])
        values[r] = energy_distance(pool[idx[:n_x]], pool[idx[n_x:n_x + n_y]])
    return {"replicates": values.tolist(), "n_replicates": n_replicates,
            "n_x": n_x, "n_y": n_y, "quantile": quantile,
            "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "max": float(values.max()),
            "envelope": float(np.quantile(values, quantile))}


# ------------------------------------------------------------------------- marginal level
def marginal_diagnostics(chains: dict, summary: dict, truth: dict,
                         acceptance_total: dict, acceptance_post: dict) -> dict:
    """Per-coordinate convergence and agreement with the reference marginal."""
    out = {}
    for name, draws in chains.items():
        draws = np.atleast_2d(np.asarray(draws, dtype=float))
        pooled = draws.ravel()
        convergence = rank_normalized_split_rhat(draws)
        marginal = summary["marginals"][name]
        grid = np.asarray(marginal["value"], dtype=float)
        cdf = np.asarray(marginal["cdf"], dtype=float)
        reference_sd = summary["sd"][name]
        out[name] = {
            "mcmc": {"mean": float(pooled.mean()), "sd": float(pooled.std(ddof=1)),
                     "median": float(np.median(pooled)),
                     "q025": float(np.quantile(pooled, 0.025)),
                     "q975": float(np.quantile(pooled, 0.975))},
            "reference": {"mean": summary["mean"][name], "sd": reference_sd,
                          "median": summary["median"][name],
                          "q025": summary["q025"][name], "q975": summary["q975"][name]},
            "rhat": convergence["rhat"], "rhat_bulk": convergence["rhat_bulk"],
            "rhat_tail": convergence["rhat_tail"],
            "bulk_ess": bulk_ess(draws), "tail_ess": tail_ess(draws),
            "mcse_mean": mcse_mean(draws),
            "ks_distance": ks_distance_to_grid(pooled, grid, cdf),
            "standardized_mean_error":
                abs(float(pooled.mean()) - summary["mean"][name]) / reference_sd,
            "standardized_median_error":
                abs(float(np.median(pooled)) - summary["median"][name]) / reference_sd,
            "standardized_q025_error":
                abs(float(np.quantile(pooled, 0.025)) - summary["q025"][name]) / reference_sd,
            "standardized_q975_error":
                abs(float(np.quantile(pooled, 0.975)) - summary["q975"][name]) / reference_sd,
            "sd_ratio": float(pooled.std(ddof=1)) / reference_sd,
            "acceptance_total": acceptance_total.get(name),
            "acceptance_post_burn_in": acceptance_post.get(name),
            "true_value": truth[name],
        }
    return out


# ----------------------------------------------------------------------- dependence level
def dependence_diagnostics(chains: dict, summary: dict, active) -> dict:
    """Covariance, correlation and the worst pairwise error against the reference."""
    active = list(active)
    pooled = np.column_stack([np.asarray(chains[n]).ravel() for n in active])
    mcmc_cov = np.cov(pooled, rowvar=False)
    sd = np.sqrt(np.diag(mcmc_cov))
    mcmc_corr = mcmc_cov / np.outer(sd, sd)
    reference_cov = np.array(summary["covariance"], dtype=float)
    reference_corr = np.array(summary["correlation"], dtype=float)

    pairs = {}
    worst_corr = 0.0
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            error = abs(mcmc_corr[i, j] - reference_corr[i, j])
            worst_corr = max(worst_corr, error)
            pairs[f"{active[i]}|{active[j]}"] = {
                "mcmc_correlation": float(mcmc_corr[i, j]),
                "reference_correlation": float(reference_corr[i, j]),
                "absolute_error": float(error),
                "mcmc_covariance": float(mcmc_cov[i, j]),
                "reference_covariance": float(reference_cov[i, j]),
            }
    # covariance compared on the standardised scale so units do not dominate
    scale = np.array([summary["sd"][n] for n in active])
    normalised_cov_error = np.abs(mcmc_cov - reference_cov) / np.outer(scale, scale)
    return {"active": active,
            "mcmc_covariance": mcmc_cov.tolist(),
            "reference_covariance": reference_cov.tolist(),
            "mcmc_correlation": mcmc_corr.tolist(),
            "reference_correlation": reference_corr.tolist(),
            "pairs": pairs,
            "max_correlation_error": float(worst_corr),
            "max_normalised_covariance_error": float(normalised_cov_error.max())}


# ------------------------------------------------------------------------ joint level
def multivariate_comparison(chains: dict, reference_draws: np.ndarray, summary: dict,
                            active, seed: int, n_compare: int = 2000,
                            n_replicates: int = 40, quantile: float = 0.99) -> dict:
    """Energy distance against the reference, judged against its own calibrated null."""
    active = list(active)
    rng = np.random.default_rng(seed)
    pooled = np.column_stack([np.asarray(chains[n]).ravel() for n in active])
    centre = np.array([summary["mean"][n] for n in active])
    scale = np.array([summary["sd"][n] for n in active])

    reference_draws = np.asarray(reference_draws, dtype=float)
    take_mcmc = rng.choice(pooled.shape[0], size=min(n_compare, pooled.shape[0]),
                           replace=False)
    take_ref = rng.choice(reference_draws.shape[0],
                          size=min(n_compare, reference_draws.shape[0]), replace=False)
    x = standardise(pooled[take_mcmc], centre, scale)
    y = standardise(reference_draws[take_ref], centre, scale)

    observed = energy_distance(x, y)
    null = calibrate_energy_envelope(
        standardise(reference_draws, centre, scale), x.shape[0], y.shape[0],
        n_replicates, seed + 1, quantile)
    return {"statistic": "energy distance on reference-standardised coordinates",
            "observed": observed, "envelope": null["envelope"],
            "envelope_quantile": quantile, "null_mean": null["mean"],
            "null_sd": null["sd"], "null_max": null["max"],
            "n_replicates": n_replicates, "n_mcmc": int(x.shape[0]),
            "n_reference": int(y.shape[0]),
            "z_score": (observed - null["mean"]) / null["sd"] if null["sd"] > 0 else float("nan"),
            "pass": bool(observed <= null["envelope"])}


def calibrate_correlation_envelope(reference_draws: np.ndarray, n_draws: int,
                                   n_replicates: int, seed: int,
                                   quantile: float = 0.99) -> dict:
    """Null distribution of the max pairwise correlation error, reference vs reference.

    A correlation estimated from a finite sample carries sampling noise, so "the MCMC
    correlation differs from the reference by 0.01" only means something once you know
    what two independent reference samples of the same size do.
    """
    rng = np.random.default_rng(seed)
    pool = np.asarray(reference_draws, dtype=float)
    d = pool.shape[1]
    if pool.shape[0] < 2 * n_draws:
        raise ValueError(
            f"calibration needs two disjoint samples of {n_draws}, but the reference pool "
            f"holds only {pool.shape[0]} draws; draw at least {2 * n_draws}")
    values = np.empty(n_replicates)
    for r in range(n_replicates):
        idx = rng.permutation(pool.shape[0])
        a = np.corrcoef(pool[idx[:n_draws]], rowvar=False)
        b = np.corrcoef(pool[idx[n_draws:2 * n_draws]], rowvar=False)
        upper = np.triu_indices(d, k=1)
        values[r] = float(np.abs(a[upper] - b[upper]).max())
    return {"n_replicates": n_replicates, "n_draws": n_draws, "quantile": quantile,
            "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "max": float(values.max()),
            "envelope": float(np.quantile(values, quantile))}


# --------------------------------------------------------------------------- recovery
def recovery_table(chains: dict, truth: dict) -> dict:
    """Synthetic recovery, reported separately from sampler correctness.

    Whether the generating value sits inside the posterior is a statement about this
    finite dataset, not about the sampler. A correct sampler on a 500-block draw can put
    the truth outside a 95% interval one time in twenty by construction.
    """
    out = {}
    for name, draws in chains.items():
        pooled = np.asarray(draws).ravel()
        mean = float(pooled.mean())
        sd = float(pooled.std(ddof=1))
        q025 = float(np.quantile(pooled, 0.025))
        q975 = float(np.quantile(pooled, 0.975))
        out[name] = {"true_value": truth[name], "posterior_mean": mean,
                     "posterior_median": float(np.median(pooled)), "posterior_sd": sd,
                     "q025": q025, "q975": q975,
                     "truth_in_95_interval": bool(q025 <= truth[name] <= q975),
                     "absolute_error": abs(mean - truth[name]),
                     "error_in_posterior_sd": abs(mean - truth[name]) / sd}
    return out
