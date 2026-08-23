"""Stage 6B1 — convergence diagnostics, and comparison against the immutable grids.

The decisive Stage 6B1 output is not "the truth is inside the interval" — a wrong
sampler passes that routinely. It is that the MCMC posterior *is* the Stage 6B0 posterior:
standardized mean and median errors, standardized interval endpoints, and a
Kolmogorov-Smirnov distance against the normalized reference CDF.

R-hat and ESS follow Vehtari et al. (2021): rank-normalized, split, and reported as the
maximum of the bulk and folded-tail statistics, with bulk and tail effective sample sizes
computed separately. The plain Gelman-Rubin statistic in ``diagnostics.py`` is kept where
it is already used; it is not rank-normalized and is not the registered Stage 6B1 gate.

The reference grids are read only. Nothing in this module writes to
``reference_posteriors.json``, and the kappa view of the omega grid is a change of
variables on the stored density, not a re-run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import special, stats

__all__ = [
    "autocovariance", "effective_sample_size_core", "rank_normalize", "split_chains",
    "rank_normalized_split_rhat", "bulk_ess", "tail_ess", "mcse_mean",
    "ks_distance_to_grid", "load_reference_posteriors", "grid_summary",
    "kappa_grid_from_omega", "compare_to_reference", "FULL_GATES", "SMOKE_GATES",
    "evaluate_gates",
]

# Registered acceptance criteria. Fixed before any chain was run.
FULL_GATES = {
    "standardized_mean_error": 0.15,
    "standardized_median_error": 0.15,
    "standardized_q025_error": 0.25,
    "standardized_q975_error": 0.25,
    "ks_distance": 0.03,
    "rhat_max": 1.01,
    "bulk_ess_min": 1000.0,
    "tail_ess_min": 500.0,
    "acceptance_rate_range": (0.15, 0.60),
}
SMOKE_GATES = {
    "standardized_mean_error": 0.35,
    "rhat_max": 1.05,
}


# ------------------------------------------------------------------ core autocorrelation
def autocovariance(chains: np.ndarray) -> np.ndarray:
    """Biased autocovariance (denominator ``N``) per chain, by FFT. Shape ``(C, N)``."""
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    n = chains.shape[-1]
    centred = chains - chains.mean(axis=-1, keepdims=True)
    size = int(2 ** math.ceil(math.log2(2 * n)))
    spectrum = np.fft.rfft(centred, n=size, axis=-1)
    spectrum *= np.conjugate(spectrum)
    acov = np.fft.irfft(spectrum, n=size, axis=-1)[..., :n]
    return acov / n


def effective_sample_size_core(chains: np.ndarray) -> float:
    """Geyer initial monotone positive sequence ESS across chains (Vehtari et al. 2021)."""
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    n_chain, n_draw = chains.shape
    if n_draw < 4:
        return float("nan")
    if np.allclose(chains, chains.flat[0]):
        return float("nan")

    acov = autocovariance(chains)
    chain_mean = chains.mean(axis=-1)
    mean_var = float(np.mean(acov[:, 0])) * n_draw / (n_draw - 1.0)
    var_plus = mean_var * (n_draw - 1.0) / n_draw
    if n_chain > 1:
        var_plus += float(np.var(chain_mean, ddof=1))
    if var_plus <= 0:
        return float("nan")

    rho = np.zeros(n_draw)
    rho[0] = 1.0
    rho_even = 1.0
    rho_odd = 1.0 - (mean_var - float(np.mean(acov[:, 1]))) / var_plus
    rho[1] = rho_odd

    t = 1
    while t < (n_draw - 3) and (rho_even + rho_odd) > 0.0:
        rho_even = 1.0 - (mean_var - float(np.mean(acov[:, t + 1]))) / var_plus
        rho_odd = 1.0 - (mean_var - float(np.mean(acov[:, t + 2]))) / var_plus
        if (rho_even + rho_odd) >= 0.0:
            rho[t + 1] = rho_even
            rho[t + 2] = rho_odd
        t += 2

    max_t = t - 2
    if rho_even > 0.0:
        rho[max_t + 1] = rho_even

    # enforce the initial monotone sequence on consecutive pair sums
    k = 1
    while k <= max_t - 2:
        pair = rho[k + 1] + rho[k + 2]
        if pair > (rho[k - 1] + rho[k]):
            rho[k + 1] = (rho[k - 1] + rho[k]) / 2.0
            rho[k + 2] = rho[k + 1]
        k += 2

    total = n_chain * n_draw
    tau = -1.0 + 2.0 * float(np.sum(rho[: max_t + 1])) + float(np.sum(rho[max_t + 1: max_t + 2]))
    tau = max(tau, 1.0 / math.log10(total))
    return float(total / tau)


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Blom rank-normalisation: ``Phi^-1((r - 3/8) / (S - 1/4))`` over all draws jointly."""
    values = np.asarray(values, dtype=float)
    ranks = stats.rankdata(values, method="average").reshape(values.shape)
    size = values.size
    return stats.norm.ppf((ranks - 0.375) / (size - 0.25))


def split_chains(chains: np.ndarray) -> np.ndarray:
    """``(C, N)`` -> ``(2C, N // 2)``, discarding one draw when ``N`` is odd."""
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    half = chains.shape[-1] // 2
    return np.vstack([chains[:, :half], chains[:, -half:]])


def _classic_rhat(chains: np.ndarray) -> float:
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    n_chain, n_draw = chains.shape
    if n_chain < 2 or n_draw < 2:
        return float("nan")
    within = float(chains.var(axis=-1, ddof=1).mean())
    between = float(chains.mean(axis=-1).var(ddof=1))
    if within == 0.0:
        return float("nan") if between > 0 else 1.0
    var_hat = (n_draw - 1) / n_draw * within + between
    return float(math.sqrt(var_hat / within))


def _fold(chains: np.ndarray) -> np.ndarray:
    chains = np.asarray(chains, dtype=float)
    return np.abs(chains - np.median(chains))


def rank_normalized_split_rhat(chains: np.ndarray) -> dict:
    """Bulk and folded-tail split R-hat, and the reported maximum."""
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    bulk = _classic_rhat(rank_normalize(split_chains(chains)))
    tail = _classic_rhat(rank_normalize(split_chains(_fold(chains))))
    return {"rhat_bulk": bulk, "rhat_tail": tail, "rhat": float(max(bulk, tail))}


def bulk_ess(chains: np.ndarray) -> float:
    return effective_sample_size_core(rank_normalize(split_chains(chains)))


def tail_ess(chains: np.ndarray, prob: tuple[float, float] = (0.05, 0.95)) -> float:
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    low, high = np.quantile(chains, prob)
    below_low = (chains <= low).astype(float)
    below_high = (chains <= high).astype(float)
    return float(min(
        effective_sample_size_core(rank_normalize(split_chains(below_low))),
        effective_sample_size_core(rank_normalize(split_chains(below_high)))))


def mcse_mean(chains: np.ndarray) -> float:
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    ess = bulk_ess(chains)
    if not math.isfinite(ess) or ess <= 0:
        return float("nan")
    return float(np.std(chains.ravel(), ddof=1) / math.sqrt(ess))


# --------------------------------------------------------------------- grid comparison
def ks_distance_to_grid(samples, grid, cdf) -> float:
    """Two-sided sup-norm between the empirical CDF of ``samples`` and the reference CDF."""
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    reference = np.interp(x, np.asarray(grid, dtype=float), np.asarray(cdf, dtype=float))
    upper = np.arange(1, n + 1) / n - reference
    lower = reference - np.arange(0, n) / n
    return float(max(upper.max(), lower.max()))


def load_reference_posteriors(path) -> dict:
    """Read the immutable Stage 6B0 grids. Read only — nothing here writes to that file."""
    return json.loads(Path(path).read_text())


def grid_summary(entry: dict) -> dict:
    """Grid, CDF and moments as stored, plus a callable-ready view for the comparison."""
    grid = np.asarray(entry["grid"], dtype=float)
    density = np.asarray(entry["density"], dtype=float)
    cdf = np.asarray(entry["cdf"], dtype=float)
    return {"grid": grid, "density": density, "cdf": cdf,
            "mean": float(entry["mean"]), "median": float(entry["median"]),
            "sd": float(entry["sd"]), "q025": float(entry["q025"]),
            "q975": float(entry["q975"]), "true_value": float(entry["true_value"])}


def kappa_grid_from_omega(entry: dict) -> dict:
    """The kappa view of the omega grid, by change of variables on the stored density.

    ``kappa = sigmoid(omega)`` is strictly increasing, so the CDF carries across
    unchanged and the density picks up the Jacobian ``1 / (kappa (1 - kappa))``. This is
    a re-reading of the immutable grid, not a second grid run: no likelihood is evaluated.
    """
    omega = np.asarray(entry["grid"], dtype=float)
    density_omega = np.asarray(entry["density"], dtype=float)
    cdf = np.asarray(entry["cdf"], dtype=float)
    kappa = special.expit(omega)          # array form of recurrent_rfs.sigmoid
    jacobian = kappa * (1.0 - kappa)
    density_kappa = density_omega / jacobian

    mean = float(np.trapezoid(kappa * density_omega, omega))
    var = float(np.trapezoid((kappa - mean) ** 2 * density_omega, omega))
    quantile = lambda level: float(np.interp(level, cdf, kappa))  # noqa: E731
    return {"grid": kappa, "density": density_kappa, "cdf": cdf,
            "mean": mean, "median": quantile(0.5), "sd": math.sqrt(var),
            "q025": quantile(0.025), "q975": quantile(0.975),
            "true_value": float(special.expit(entry["true_value"]))}


def compare_to_reference(chains: np.ndarray, grid: dict) -> dict:
    """Standardized errors and KS distance against a reference grid.

    Every error is divided by the *grid* standard deviation, so the numbers are
    comparable across parameters on different scales.
    """
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    draws = chains.ravel()
    sd_grid = grid["sd"]
    mcmc = {"mean": float(draws.mean()), "median": float(np.median(draws)),
            "sd": float(draws.std(ddof=1)),
            "q025": float(np.quantile(draws, 0.025)),
            "q975": float(np.quantile(draws, 0.975))}
    convergence = rank_normalized_split_rhat(chains)
    out = {
        "n_draws": int(draws.size), "n_chains": int(chains.shape[0]),
        "mcmc": mcmc, "grid": {k: grid[k] for k in ("mean", "median", "sd", "q025", "q975")},
        "standardized_mean_error": abs(mcmc["mean"] - grid["mean"]) / sd_grid,
        "standardized_median_error": abs(mcmc["median"] - grid["median"]) / sd_grid,
        "standardized_q025_error": abs(mcmc["q025"] - grid["q025"]) / sd_grid,
        "standardized_q975_error": abs(mcmc["q975"] - grid["q975"]) / sd_grid,
        "sd_ratio": mcmc["sd"] / sd_grid,
        "ks_distance": ks_distance_to_grid(draws, grid["grid"], grid["cdf"]),
        "bulk_ess": bulk_ess(chains), "tail_ess": tail_ess(chains),
        "mcse_mean": mcse_mean(chains),
        "true_value": grid["true_value"],
        "truth_in_mcmc_interval": bool(mcmc["q025"] <= grid["true_value"] <= mcmc["q975"]),
        **convergence,
    }
    return out


def evaluate_gates(comparison: dict, acceptance_rates, mode: str = "full") -> dict:
    """Registered PASS/FAIL. ``smoke`` only catches implementation failures."""
    rates = [float(r) for r in acceptance_rates]
    checks = {}
    if mode == "full":
        g = FULL_GATES
        lo, hi = g["acceptance_rate_range"]
        checks = {
            "standardized_mean_error": (comparison["standardized_mean_error"],
                                        g["standardized_mean_error"], "<="),
            "standardized_median_error": (comparison["standardized_median_error"],
                                          g["standardized_median_error"], "<="),
            "standardized_q025_error": (comparison["standardized_q025_error"],
                                        g["standardized_q025_error"], "<="),
            "standardized_q975_error": (comparison["standardized_q975_error"],
                                        g["standardized_q975_error"], "<="),
            "ks_distance": (comparison["ks_distance"], g["ks_distance"], "<="),
            "rhat": (comparison["rhat"], g["rhat_max"], "<="),
            "bulk_ess": (comparison["bulk_ess"], g["bulk_ess_min"], ">="),
            "tail_ess": (comparison["tail_ess"], g["tail_ess_min"], ">="),
            "min_acceptance_rate": (min(rates), lo, ">="),
            "max_acceptance_rate": (max(rates), hi, "<="),
        }
    else:
        g = SMOKE_GATES
        checks = {
            "standardized_mean_error": (comparison["standardized_mean_error"],
                                        g["standardized_mean_error"], "<="),
            "rhat": (comparison["rhat"], g["rhat_max"], "<"),
        }

    rows = {}
    for name, (observed, threshold, sense) in checks.items():
        if sense == "<=":
            ok = bool(observed <= threshold)
        elif sense == "<":
            ok = bool(observed < threshold)
        else:
            ok = bool(observed >= threshold)
        rows[name] = {"observed": float(observed), "threshold": float(threshold),
                      "sense": sense, "pass": ok}
    return {"mode": mode, "checks": rows, "pass": all(r["pass"] for r in rows.values())}
