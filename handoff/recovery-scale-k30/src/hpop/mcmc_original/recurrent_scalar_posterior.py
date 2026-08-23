"""Stage 6B0 — registered priors, batch likelihood, and reference posterior grids.

The grids here are **numerically normalized reference posteriors**, not analytic ones:
the density is evaluated on a finite grid and integrated by the trapezoid rule on the
*original* parameter scale. Normalizing by a softmax over grid values would ignore grid
spacing and silently treat densities as probabilities, so it is not used.

`omega` is handled by full replay. `kappa = sigmoid(omega)` enters the validity
recursion, so every candidate value recomputes `q` from zero for every block; holding a
true-parameter `q` trajectory fixed would measure only the direct one-step channel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hpop.mcmc_original.recurrent_rfs import recurrent_rfs_log_likelihood, sigmoid

__all__ = [
    "PRIORS", "log_gamma_prior", "log_normal_prior", "log_prior",
    "batch_recurrent_log_likelihood", "log_posterior_grid",
    "normalize_log_density_grid", "reference_posterior", "observed_curvature",
]

# Registered before any posterior was inspected. Gamma is shape-rate, matching the
# convention of the verified static beta sampler.
PRIORS = {
    "beta":        {"family": "gamma",  "shape": 2.0, "rate": 2.0, "support": "(0, inf)"},
    "omega":       {"family": "normal", "mean": 0.0,  "sd": 2.0,   "support": "(-inf, inf)"},
    "lambda_rep":  {"family": "gamma",  "shape": 2.0, "rate": 2.0, "support": "(0, inf)"},
    "lambda_back": {"family": "gamma",  "shape": 2.0, "rate": 2.0, "support": "(0, inf)"},
}
TRUE_VALUES = {"beta": 1.5, "omega": math.log(0.85 / 0.15),
               "lambda_rep": 0.8, "lambda_back": 0.25}


def log_gamma_prior(x: float, shape: float, rate: float) -> float:
    """Unnormalised log Gamma density in the shape-rate convention."""
    if x <= 0:
        return -math.inf
    return (shape - 1.0) * math.log(x) - rate * x


def log_normal_prior(x: float, mean: float, sd: float) -> float:
    return -0.5 * ((x - mean) / sd) ** 2


def log_prior(parameter: str, value: float) -> float:
    spec = PRIORS[parameter]
    if spec["family"] == "gamma":
        return log_gamma_prior(value, spec["shape"], spec["rate"])
    return log_normal_prior(value, spec["mean"], spec["sd"])


def batch_recurrent_log_likelihood(role_sequences, u, beta, epsilon, omega,
                                   lambda_rep, lambda_back) -> float:
    """Total log likelihood over blocks. Restarts ``q`` at zero for every block.

    Exact — no truncation or approximation of the validity recursion.
    """
    return float(sum(
        recurrent_rfs_log_likelihood(seq, u, beta, epsilon, omega, lambda_rep, lambda_back)
        for seq in role_sequences))


def log_posterior_grid(parameter: str, grid, role_sequences, u, truth: dict,
                       epsilon: float) -> np.ndarray:
    """Unnormalised log posterior over a grid, other scalars held at their true values."""
    out = np.empty(len(grid), dtype=float)
    for i, value in enumerate(grid):
        kw = dict(truth)
        kw[parameter] = float(value)
        lp = log_prior(parameter, float(value))
        out[i] = -math.inf if lp == -math.inf else lp + batch_recurrent_log_likelihood(
            role_sequences, u, kw["beta"], epsilon, kw["omega"],
            kw["lambda_rep"], kw["lambda_back"])
    return out


def normalize_log_density_grid(grid, log_density):
    """Trapezoid-normalise on the original scale; returns density, CDF, log Z."""
    grid = np.asarray(grid, dtype=float)
    log_density = np.asarray(log_density, dtype=float)
    shift = float(np.max(log_density[np.isfinite(log_density)]))
    unnormalised = np.exp(np.where(np.isfinite(log_density), log_density - shift, -np.inf))
    area = float(np.trapezoid(unnormalised, grid))
    density = unnormalised / area
    cdf = np.concatenate([[0.0], np.cumsum(
        0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    return density, cdf / cdf[-1], shift + math.log(area)


def _quantile(grid, cdf, level):
    return float(np.interp(level, cdf, grid))


def reference_posterior(parameter, role_sequences, u, truth, epsilon,
                        initial_range, n_points=1001, max_expansions=6,
                        log_drop=25.0, tail_mass=1e-6) -> dict:
    """Reference posterior with automatic range expansion (spec 7.2)."""
    lo, hi = initial_range
    initial = (lo, hi)
    support_low = 0.0 if PRIORS[parameter]["family"] == "gamma" else -np.inf
    expansions = 0
    while True:
        grid = np.linspace(lo, hi, n_points)
        log_density = log_posterior_grid(parameter, grid, role_sequences, u, truth, epsilon)
        density, cdf, log_z = normalize_log_density_grid(grid, log_density)
        peak = float(np.max(log_density[np.isfinite(log_density)]))
        edge = max(1, n_points // 100)
        left_mass = float(np.trapezoid(density[:edge], grid[:edge]))
        right_mass = float(np.trapezoid(density[-edge:], grid[-edge:]))
        need_left = (log_density[0] > peak - log_drop) or (left_mass > tail_mass)
        need_right = (log_density[-1] > peak - log_drop) or (right_mass > tail_mass)
        if (not need_left and not need_right) or expansions >= max_expansions:
            break
        width = hi - lo
        if need_left:
            lo = max(support_low + 1e-6, lo - 0.5 * width) if np.isfinite(support_low) else lo - 0.5 * width
        if need_right:
            hi = hi + 0.5 * width
        expansions += 1

    mean = float(np.trapezoid(grid * density, grid))
    var = float(np.trapezoid((grid - mean) ** 2 * density, grid))
    truth_value = TRUE_VALUES[parameter]
    q025, q975 = _quantile(grid, cdf, 0.025), _quantile(grid, cdf, 0.975)
    return {
        "parameter": parameter, "grid": grid, "log_density": log_density,
        "density": density, "cdf": cdf, "log_normalizer": log_z,
        "initial_range": list(initial), "final_range": [float(lo), float(hi)],
        "n_expansions": expansions, "n_points": n_points,
        "integral_check": float(np.trapezoid(density, grid)),
        "mode": float(grid[int(np.argmax(log_density))]),
        "mean": mean, "median": _quantile(grid, cdf, 0.5), "sd": math.sqrt(var),
        "q025": q025, "q975": q975, "true_value": truth_value,
        "truth_in_interval": bool(q025 <= truth_value <= q975),
        "log_posterior_at_truth": float(np.interp(truth_value, grid, log_density)),
        "log_posterior_at_mode": float(np.max(log_density)),
    }


def observed_curvature(parameter, role_sequences, u, truth, epsilon, centre,
                       delta=None) -> dict:
    """Observed block-likelihood curvature by central differences at ``centre``.

    Full replay: every evaluation recomputes the whole recursion, so ``omega`` is
    handled correctly. Named observed curvature, not Fisher information — no
    expectation under the generative distribution is taken.
    """
    delta = delta if delta is not None else max(0.02, 0.02 * abs(centre))

    def at(value):
        kw = dict(truth); kw[parameter] = float(value)
        return batch_recurrent_log_likelihood(role_sequences, u, kw["beta"], epsilon,
                                              kw["omega"], kw["lambda_rep"], kw["lambda_back"])
    centre_ll, up, down = at(centre), at(centre + delta), at(centre - delta)
    second = (up - 2 * centre_ll + down) / delta ** 2
    curvature = -second
    return {"centre": float(centre), "delta": float(delta),
            "second_derivative": float(second), "observed_curvature": float(curvature),
            "implied_sd": float(1.0 / math.sqrt(curvature)) if curvature > 0 else float("nan")}


def precompute_state_features(role_sequences, u, omega: float):
    """Cache the q-dependent features for a FIXED omega.

    ``q`` depends only on the observed sequence and ``omega``, so for the three
    parameters that do not enter the recursion — ``beta``, ``lambda_rep``,
    ``lambda_back`` — the whole state trajectory can be computed once. This is an exact
    rearrangement, not an approximation, and is verified against full replay in
    `tests/mcmc_original/test_stage6b_batch_likelihood.py`.

    It is **not** valid for ``omega``: changing ``omega`` changes ``q`` and therefore
    every feature below.
    """
    from hpop.mcmc_original.latent_poset import precedence_from_u
    from hpop.mcmc_original.recurrent_rfs import (
        recurrent_back_costs, recurrent_feasibility, recurrent_successor_utilities,
        recurrent_validity_update)

    u = np.asarray(u, dtype=float)
    precedence = precedence_from_u(u)
    m = u.shape[0]
    n, T = len(role_sequences), len(role_sequences[0])
    F = np.empty((n, T, m)); Q = np.empty((n, T, m))
    QV = np.empty((n, T, m)); CB = np.empty((n, T, m))
    obs = np.empty((n, T), dtype=int)
    for b, seq in enumerate(role_sequences):
        q = np.zeros(m)
        for t, y in enumerate(seq):
            F[b, t] = recurrent_feasibility(precedence, q)
            Q[b, t] = recurrent_successor_utilities(precedence, q)
            QV[b, t] = q
            CB[b, t] = recurrent_back_costs(precedence, q, omega)
            obs[b, t] = y
            q = recurrent_validity_update(y, precedence, q, omega)
    return {"F": F, "Q": Q, "q": QV, "C_back": CB, "obs": obs, "m": m, "omega": omega}


def cached_batch_log_likelihood(features, beta, epsilon, lambda_rep, lambda_back) -> float:
    """Exact batch log likelihood from cached features. Valid only at ``features['omega']``."""
    exponent = (beta * features["Q"] - lambda_rep * features["q"]
                - lambda_back * features["C_back"])
    exponent = exponent - exponent.max(axis=-1, keepdims=True)
    weights = features["F"] * np.exp(exponent)
    structural = weights / weights.sum(axis=-1, keepdims=True)
    mixed = (1.0 - epsilon) * structural + epsilon / features["m"]
    n, T = features["obs"].shape
    chosen = mixed[np.arange(n)[:, None], np.arange(T)[None, :], features["obs"]]
    return float(np.log(chosen).sum())


def batch_recurrent_log_likelihood_full_replay(role_array, u, beta, epsilon, omega,
                                               lambda_rep, lambda_back,
                                               return_states: bool = False):
    """Exact full-replay batch likelihood, vectorized across blocks.

    Every call recomputes ``kappa = sigmoid(omega)`` and propagates ``q`` from zero for
    every block through all T steps. **No state trajectory is reused across omega
    values** — that would be the fixed-q shortcut this exists to avoid. No truncation,
    no Taylor expansion, no approximation of any kind: the arithmetic is identical to
    the per-block path, only the loop over blocks is replaced by an array axis.

    Args:
        role_array: integer array ``(n_blocks, T)``; blocks must be equal length.
        return_states: also return the ``q`` trajectory and per-step probabilities,
            for the state-parity test.
    """
    from hpop.mcmc_original.latent_poset import precedence_from_u

    u = np.asarray(u, dtype=float)
    precedence = precedence_from_u(u)
    m = u.shape[0]
    roles = np.asarray(role_array, dtype=int)
    n, T = roles.shape
    kappa = sigmoid(omega)

    pred_mask = precedence.T.astype(bool)          # pred_mask[x, z] = z precedes x
    succ = precedence.astype(float)                # succ[x, z] = x precedes z
    succ_off = succ.copy()
    np.fill_diagonal(succ_off, 0.0)

    q = np.zeros((n, m))
    total = 0.0
    q_trace = np.empty((n, T, m)) if return_states else None
    p_trace = np.empty((n, T, m)) if return_states else None

    for t in range(T):
        if return_states:
            q_trace[:, t, :] = q
        # F[b, x] = prod over predecessors z of x of q[b, z]; empty product = 1
        feasibility = np.prod(np.where(pred_mask[None, :, :], q[:, None, :], 1.0), axis=2)
        stale = (1.0 - q) @ succ.T                 # S[b, x]
        back = kappa * (q @ succ_off.T)            # C_back[b, x]
        exponent = beta * np.log1p(stale) - lambda_rep * q - lambda_back * back
        exponent -= exponent.max(axis=1, keepdims=True)
        weights = feasibility * np.exp(exponent)
        mixed = (1.0 - epsilon) * (weights / weights.sum(axis=1, keepdims=True)) + epsilon / m
        if return_states:
            p_trace[:, t, :] = mixed
        observed = roles[:, t]
        total += float(np.log(mixed[np.arange(n), observed]).sum())
        # q update: refresh the executed role, gate its transitive descendants
        gate = np.where(precedence[observed], kappa, 0.0)   # (n, m)
        q = q * (1.0 - gate)
        q[np.arange(n), observed] = 1.0

    return (total, q_trace, p_trace) if return_states else total
