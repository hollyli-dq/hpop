"""Small MCMC diagnostics and evaluation metrics for the toy stages.

Deliberately minimal — enough to decide PASS/FAIL against the exact answers and to
diagnose a failure, not a general-purpose diagnostics library.
"""

from __future__ import annotations

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u

__all__ = [
    "segmentation_frequencies",
    "boundary_marginals_from_samples",
    "total_variation_distance",
    "relation_posterior",
    "autocorrelation",
    "effective_sample_size",
    "rhat",
    "prf1",
]


def segmentation_frequencies(state_indices, n_states: int) -> np.ndarray:
    """Empirical distribution over the enumerated states."""
    counts = np.bincount(np.asarray(state_indices, dtype=int), minlength=n_states)
    return counts.astype(float) / counts.sum()


def boundary_marginals_from_samples(
    state_indices, cuts, trace_length: int
) -> np.ndarray:
    """``P(cut at t)`` for ``t = 1, ..., T-1``, from sampled state indices."""
    out = np.zeros(trace_length - 1, dtype=float)
    idx = np.asarray(state_indices, dtype=int)
    for state, count in zip(*np.unique(idx, return_counts=True)):
        for t in cuts[state]:
            out[t - 1] += count
    return out / len(idx)


def total_variation_distance(p, q) -> float:
    """``0.5 * sum |p_i - q_i|``."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {q.shape}")
    return 0.5 * float(np.abs(p - q).sum())


def relation_posterior(u_samples) -> np.ndarray:
    """``P(i > j | data)`` estimated as the fraction of draws where ``i > j``."""
    u_samples = np.asarray(u_samples, dtype=float)
    if u_samples.ndim != 3:
        raise ValueError(f"expected (n_samples, m, d), got shape {u_samples.shape}")
    stack = np.array([precedence_from_u(u) for u in u_samples], dtype=float)
    return stack.mean(axis=0)


def autocorrelation(trace, max_lag: int = 50) -> np.ndarray:
    """Autocorrelation of a scalar chain at lags ``0 .. max_lag``."""
    x = np.asarray(trace, dtype=float)
    x = x - x.mean()
    n = len(x)
    max_lag = min(max_lag, n - 1)
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return np.concatenate([[1.0], np.zeros(max_lag)])
    return np.array([float(np.dot(x[: n - k], x[k:])) / denom for k in range(max_lag + 1)])


def effective_sample_size(trace, max_lag: int = 1000) -> float:
    """``N / (1 + 2 sum rho_k)``, truncated at the first non-positive autocorrelation."""
    x = np.asarray(trace, dtype=float)
    n = len(x)
    if n < 2 or np.allclose(x, x[0]):
        return float(n)
    rho = autocorrelation(x, max_lag=min(max_lag, n - 1))
    total = 0.0
    for k in range(1, len(rho)):
        if rho[k] <= 0.0:
            break
        total += rho[k]
    return float(n / (1.0 + 2.0 * total))


def rhat(chains) -> float:
    """Gelman-Rubin potential scale reduction for a set of equal-length chains."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2:
        raise ValueError(f"expected (n_chains, n_draws), got shape {chains.shape}")
    n_chains, n_draws = chains.shape
    if n_chains < 2 or n_draws < 2:
        return float("nan")
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    within = chain_vars.mean()
    between = n_draws * chain_means.var(ddof=1)
    if within == 0.0:
        return float("nan") if between > 0 else 1.0
    var_hat = ((n_draws - 1) / n_draws) * within + between / n_draws
    return float(np.sqrt(var_hat / within))


def prf1(predicted: set, truth: set) -> dict[str, float]:
    """Precision / recall / F1 for two sets of items."""
    predicted, truth = set(predicted), set(truth)
    true_positive = len(predicted & truth)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(truth) if truth else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "n_predicted": len(predicted),
        "n_truth": len(truth),
    }
