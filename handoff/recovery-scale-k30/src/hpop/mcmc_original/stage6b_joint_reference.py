"""Stage 6B2 / 6B3 — independent joint reference posteriors by direct quadrature.

Four matching one-dimensional marginals are not evidence of a correct joint posterior, and
the Stage 6B1 grids are conditional: each was built with the other three scalars pinned at
truth. So the joint stages get their own references, built here.

Nothing in this module touches the MCMC transition kernel, the acceptance ratio, or any
formal draw. It evaluates the direct recurrent log posterior on a deterministic grid. The
grid is placed using a MAP found by deterministic optimisation and the curvature there —
both allowed by section 7, and neither derived from a chain.

Coordinates are the ones the Stage 6B1 proposals walk on: ``log beta``, ``omega``,
``log lambda_rep``, ``log lambda_back``. A density in ``lambda`` is *not* a density in
``log lambda``; the change of variables carries ``|d lambda / d log lambda| = lambda``,
so the log density in the transformed coordinate picks up ``+ log lambda``. That term is
included explicitly and tested, because dropping exactly this Jacobian is the failure the
Stage 6B1 negative control demonstrates.

## Why the evaluation is fast enough to be exhaustive

A step contributes to the likelihood only through its ``(F, Q, q, C_back, observed)`` row.
Across the 500x20 corpus only ~1150 distinct rows occur at a given omega, so identical
rows are collapsed and carried with multiplicities. That is an exact regrouping of a sum,
verified against the uncollapsed evaluator, and it buys ~8.7x. Batching the beta axis
buys the rest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from hpop.mcmc_original.recurrent_joint_scalar_mcmc import vectorized_state_features
from hpop.mcmc_original.recurrent_scalar_posterior import log_prior
from hpop.mcmc_original.stage6b_frozen import (
    PARAMETER_ORDER, TRANSFORM, from_unconstrained, log_jacobian_to_unconstrained,
    to_unconstrained,
)

__all__ = [
    "collapse_features", "collapsed_log_likelihood_beta_grid", "transformed_log_posterior",
    "find_map", "curvature_scales", "build_reference", "reference_summary",
    "sample_reference_draws", "ReferenceGrid", "normalise_log_density",
]


# ------------------------------------------------------------------ collapsed evaluation
def collapse_features(features: dict) -> dict:
    """Group identical ``(F, Q, q, C_back, observed)`` steps and carry multiplicities.

    Exact: the log likelihood is a sum over steps, and identical steps contribute
    identical terms. Verified against the uncollapsed evaluator in
    `test_stage6b_joint_reference.py`.
    """
    m = features["m"]
    flat = [features[k].reshape(-1, m) for k in ("F", "Q", "q", "C_back")]
    obs = features["obs"].reshape(-1, 1).astype(float)
    key = np.round(np.concatenate(flat + [obs], axis=1), 12)
    unique, counts = np.unique(key, axis=0, return_counts=True)
    return {"F": unique[:, 0:m], "Q": unique[:, m:2 * m], "q": unique[:, 2 * m:3 * m],
            "C_back": unique[:, 3 * m:4 * m], "obs": unique[:, 4 * m].astype(int),
            "weight": counts.astype(float), "m": m, "omega": features["omega"],
            "n_steps": int(counts.sum()), "n_unique": int(unique.shape[0])}


def collapsed_log_likelihood_beta_grid(collapsed: dict, betas, epsilon: float,
                                       lambda_rep: float, lambda_back: float) -> np.ndarray:
    """Complete log likelihood for a vector of ``beta`` at one fixed ``(omega, lambdas)``.

    Same arithmetic as `cached_batch_log_likelihood`, with the block/step axis collapsed
    and a leading ``beta`` axis added.
    """
    betas = np.atleast_1d(np.asarray(betas, dtype=float))
    m = collapsed["m"]
    base = -lambda_rep * collapsed["q"] - lambda_back * collapsed["C_back"]   # (R, m)
    exponent = betas[:, None, None] * collapsed["Q"][None, :, :] + base[None, :, :]
    exponent -= exponent.max(axis=2, keepdims=True)
    weights = collapsed["F"][None, :, :] * np.exp(exponent)
    mixed = (1.0 - epsilon) * (weights / weights.sum(axis=2, keepdims=True)) + epsilon / m
    chosen = mixed[:, np.arange(collapsed["obs"].size), collapsed["obs"]]        # (nb, R)
    return (np.log(chosen) * collapsed["weight"][None, :]).sum(axis=1)


# --------------------------------------------------------------------- transformed target
def transformed_log_posterior(z, active, evaluator, fixed: dict) -> float:
    """Unnormalised log posterior in the transformed coordinates, Jacobians included."""
    values = dict(fixed)
    total_jacobian = 0.0
    for name, zi in zip(active, np.atleast_1d(z)):
        value = from_unconstrained(name, float(zi))
        values[name] = value
        total_jacobian += log_jacobian_to_unconstrained(name, value)
    prior = sum(log_prior(name, values[name]) for name in active)
    if not math.isfinite(prior):
        return -math.inf
    ll = evaluator.full_replay_log_likelihood(
        values["beta"], values["omega"], values["lambda_rep"], values["lambda_back"])
    return float(prior + total_jacobian + ll)


def find_map(active, evaluator, fixed: dict, start: dict) -> dict:
    """Deterministic MAP in transformed coordinates — Nelder-Mead, no randomness."""
    z0 = np.array([to_unconstrained(n, start[n]) for n in active])
    negative = lambda z: -transformed_log_posterior(z, active, evaluator, fixed)  # noqa: E731
    result = optimize.minimize(negative, z0, method="Nelder-Mead",
                               options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 4000})
    return {"z": result.x, "log_posterior": float(-result.fun), "success": bool(result.success),
            "n_evaluations": int(result.nfev),
            "values": {n: from_unconstrained(n, float(v)) for n, v in zip(active, result.x)}}


def curvature_scales(active, evaluator, fixed: dict, z_mode, step: float = 0.02) -> dict:
    """Marginal sds from the finite-difference Hessian at the mode, in transformed space."""
    z_mode = np.asarray(z_mode, dtype=float)
    d = z_mode.size
    hessian = np.empty((d, d))
    f0 = transformed_log_posterior(z_mode, active, evaluator, fixed)
    for i in range(d):
        for j in range(i, d):
            if i == j:
                zp, zm = z_mode.copy(), z_mode.copy()
                zp[i] += step; zm[i] -= step
                second = (transformed_log_posterior(zp, active, evaluator, fixed)
                          - 2 * f0
                          + transformed_log_posterior(zm, active, evaluator, fixed)) / step ** 2
            else:
                zpp, zpm, zmp, zmm = (z_mode.copy() for _ in range(4))
                zpp[i] += step; zpp[j] += step
                zpm[i] += step; zpm[j] -= step
                zmp[i] -= step; zmp[j] += step
                zmm[i] -= step; zmm[j] -= step
                second = (transformed_log_posterior(zpp, active, evaluator, fixed)
                          - transformed_log_posterior(zpm, active, evaluator, fixed)
                          - transformed_log_posterior(zmp, active, evaluator, fixed)
                          + transformed_log_posterior(zmm, active, evaluator, fixed)) / (4 * step ** 2)
            hessian[i, j] = hessian[j, i] = second
    covariance = np.linalg.inv(-hessian)
    return {"hessian": hessian, "covariance": covariance,
            "sd": np.sqrt(np.diag(covariance))}


# ------------------------------------------------------------------------------ the grid
@dataclass
class ReferenceGrid:
    active: tuple
    axes_z: dict                 # transformed-coordinate axis values
    axes_value: dict             # the same axes mapped back to the parameter scale
    log_density_z: np.ndarray    # unnormalised, shape (n, n, ...) in `active` order
    density_z: np.ndarray        # normalised on the transformed grid
    log_normaliser: float
    n_points: int
    radius: float
    fixed: dict
    map_values: dict
    outer_face_mass: float
    config: dict


def _evaluate_grid(active, axes_z, evaluator, fixed, epsilon) -> np.ndarray:
    """Log posterior on the tensor grid.

    ``omega`` is the outer loop when active, because the state trajectory has to be
    rebuilt for every one of its values and for nothing else; ``beta`` is the inner
    batched axis, because it enters only through a coefficient. Either may be inactive —
    a reference over a subset of the coordinates is a legitimate object, and the
    single-coordinate case is what the Jacobian test compares against Stage 6B0.
    """
    names = list(active)
    shape = tuple(axes_z[n].size for n in names)
    out = np.empty(shape)

    has_omega = "omega" in names
    has_beta = "beta" in names
    other = [n for n in names if n not in ("omega", "beta")]

    # prior + Jacobian terms are separable across coordinates
    log_weight = {}
    for n in names:
        values = np.array([from_unconstrained(n, z) for z in axes_z[n]])
        log_weight[n] = np.array([log_prior(n, v) + log_jacobian_to_unconstrained(n, v)
                                  for v in values])

    betas = (np.array([from_unconstrained("beta", z) for z in axes_z["beta"]])
             if has_beta else np.array([float(fixed["beta"])]))
    omegas = ([from_unconstrained("omega", float(z)) for z in axes_z["omega"]]
              if has_omega else [float(fixed["omega"])])

    for oi, omega in enumerate(omegas):
        collapsed = collapse_features(
            vectorized_state_features(evaluator.role_array, evaluator.u, omega))
        if other:
            grids = np.meshgrid(*[np.arange(axes_z[n].size) for n in other], indexing="ij")
            combos = list(zip(*[g.ravel() for g in grids]))
        else:
            combos = [()]
        for combo in combos:
            lam = dict(fixed)
            for n, idx in zip(other, combo):
                lam[n] = from_unconstrained(n, float(axes_z[n][idx]))
            ll = collapsed_log_likelihood_beta_grid(
                collapsed, betas, epsilon, lam["lambda_rep"], lam["lambda_back"])

            index = [slice(None)] * len(names)
            if has_omega:
                index[names.index("omega")] = oi
            for n, idx in zip(other, combo):
                index[names.index(n)] = idx

            value = ll if has_beta else float(ll[0])
            if has_beta:
                value = value + log_weight["beta"]
            if has_omega:
                value = value + log_weight["omega"][oi]
            for n, idx in zip(other, combo):
                value = value + log_weight[n][idx]
            out[tuple(index)] = value
    return out


def build_reference(active, evaluator, fixed: dict, truth: dict, epsilon: float,
                    n_points: int = 41, radius: float = 6.0,
                    max_expansions: int = 3, face_tolerance: float = 1e-6) -> ReferenceGrid:
    """Deterministic tensor-grid reference, with automatic outer-face expansion."""
    active = tuple(p for p in PARAMETER_ORDER if p in set(active))
    start = {n: truth[n] for n in active}
    mode = find_map(active, evaluator, fixed, start)
    curvature = curvature_scales(active, evaluator, fixed, mode["z"])

    expansions = 0
    while True:
        axes_z = {n: np.linspace(mode["z"][i] - radius * curvature["sd"][i],
                                 mode["z"][i] + radius * curvature["sd"][i], n_points)
                  for i, n in enumerate(active)}
        log_density = _evaluate_grid(active, axes_z, evaluator, fixed, epsilon)
        density, log_z = _normalise(active, axes_z, log_density)
        face = _outer_face_mass(active, axes_z, density)
        if face <= face_tolerance or expansions >= max_expansions:
            break
        radius *= 1.5
        expansions += 1

    return ReferenceGrid(
        active=active, axes_z=axes_z,
        axes_value={n: np.array([from_unconstrained(n, z) for z in axes_z[n]])
                    for n in active},
        log_density_z=log_density, density_z=density, log_normaliser=log_z,
        n_points=n_points, radius=radius, fixed=dict(fixed),
        map_values=mode["values"], outer_face_mass=face,
        config={"method": "deterministic tensor grid on transformed coordinates",
                "transform": {n: TRANSFORM[n] for n in active},
                "jacobian": "log|d value / d z| included per coordinate",
                "map": {k: float(v) for k, v in mode["values"].items()},
                "map_log_posterior": mode["log_posterior"],
                "map_optimiser": "Nelder-Mead (deterministic)",
                "curvature_sd_z": curvature["sd"].tolist(),
                "n_points": n_points, "radius_in_curvature_sd": radius,
                "n_expansions": expansions,
                "grid_points": int(np.prod([axes_z[n].size for n in active])),
                "uses_mcmc": False})


def normalise_log_density(active, axes_z, log_density):
    """Trapezoid-normalise a transformed-coordinate log density; returns (density, log Z).

    Public so that a stored reference can be reconstituted from its log density alone —
    the normalised array is derivable, so only one of the two needs to go to disk.
    """
    return _normalise(active, axes_z, log_density)


def _normalise(active, axes_z, log_density):
    shift = float(np.max(log_density))
    unnormalised = np.exp(log_density - shift)
    # integrate axis by axis; each trapezoid removes axis 0 of the shrinking array
    total = unnormalised
    for n in active:
        total = np.trapezoid(total, axes_z[n], axis=0)
    volume = float(total)
    return unnormalised / volume, shift + math.log(volume)


def _outer_face_mass(active, axes_z, density) -> float:
    """Total probability mass on the outermost slice of every axis — the coverage audit."""
    worst = 0.0
    for i, n in enumerate(active):
        for index in (0, -1):
            face = np.take(density, index, axis=i)
            remaining = [m for j, m in enumerate(active) if j != i]
            total = face
            for other in remaining:
                total = np.trapezoid(total, axes_z[other], axis=0)
            # a slice has no width; weight it by the local spacing to get a mass
            spacing = abs(axes_z[n][1] - axes_z[n][0])
            worst = max(worst, float(total) * spacing)
    return worst


# --------------------------------------------------------------------------- summaries
def reference_summary(grid: ReferenceGrid) -> dict:
    """Moments, quantiles, covariance and correlation on the PARAMETER scale."""
    active = grid.active
    axes_z = [grid.axes_z[n] for n in active]
    values = [grid.axes_value[n] for n in active]
    density = grid.density_z

    def integrate(weighted):
        total = weighted
        for axis in axes_z:
            total = np.trapezoid(total, axis, axis=0)
        return float(total)

    shape = [1] * len(active)
    value_grids = []
    for i, v in enumerate(values):
        s = list(shape); s[i] = v.size
        value_grids.append(v.reshape(s))

    mass = integrate(density)
    means = [integrate(density * vg) / mass for vg in value_grids]
    covariance = np.empty((len(active), len(active)))
    for i in range(len(active)):
        for j in range(i, len(active)):
            cov = integrate(density * (value_grids[i] - means[i]) * (value_grids[j] - means[j])) / mass
            covariance[i, j] = covariance[j, i] = cov
    sd = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(sd, sd)

    marginals, quantiles = {}, {}
    for i, name in enumerate(active):
        marginal = density
        for j in range(len(active) - 1, -1, -1):
            if j != i:
                marginal = np.trapezoid(marginal, axes_z[j], axis=j)
        marginal = marginal / np.trapezoid(marginal, axes_z[i])
        cdf = np.concatenate([[0.0], np.cumsum(
            0.5 * (marginal[1:] + marginal[:-1]) * np.diff(axes_z[i]))])
        cdf = cdf / cdf[-1]
        marginals[name] = {"z": axes_z[i], "value": values[i],
                           "density_z": marginal, "cdf": cdf}
        quantiles[name] = {level: float(np.interp(level, cdf, values[i]))
                           for level in (0.025, 0.5, 0.975)}

    flat_index = int(np.argmax(grid.log_density_z))
    mode_index = np.unravel_index(flat_index, grid.log_density_z.shape)

    return {
        "active": list(active),
        "mean": {n: means[i] for i, n in enumerate(active)},
        "sd": {n: float(sd[i]) for i, n in enumerate(active)},
        "median": {n: quantiles[n][0.5] for n in active},
        "q025": {n: quantiles[n][0.025] for n in active},
        "q975": {n: quantiles[n][0.975] for n in active},
        "covariance": covariance.tolist(), "correlation": correlation.tolist(),
        "mode": {n: float(grid.axes_value[n][mode_index[i]]) for i, n in enumerate(active)},
        "map": grid.map_values,
        "integral_check": mass,
        "outer_face_mass": grid.outer_face_mass,
        "marginals": {n: {"value": marginals[n]["value"].tolist(),
                          "cdf": marginals[n]["cdf"].tolist()} for n in active},
        "n_points": grid.n_points, "radius": grid.radius,
        "grid_points": int(np.prod([grid.axes_z[n].size for n in active])),
    }


def sample_reference_draws(grid: ReferenceGrid, n_draws: int, seed: int) -> np.ndarray:
    """iid draws from the discretised reference, for the multivariate two-sample test.

    Cells are drawn with their trapezoid mass and jittered uniformly inside, then mapped
    back to the parameter scale. This samples the reference *grid*, which is the object
    the sampler is being compared against.
    """
    rng = np.random.default_rng(seed)
    active = grid.active
    axes_z = [grid.axes_z[n] for n in active]
    widths = [np.diff(a) for a in axes_z]

    cell = grid.density_z
    for i in range(len(active)):
        cell = (np.take(cell, np.arange(cell.shape[i] - 1), axis=i)
                + np.take(cell, np.arange(1, cell.shape[i]), axis=i)) / 2.0
    volume = np.ones_like(cell)
    for i, w in enumerate(widths):
        s = [1] * len(active); s[i] = w.size
        volume = volume * w.reshape(s)
    mass = (cell * volume).ravel()
    mass = mass / mass.sum()

    picks = rng.choice(mass.size, size=n_draws, p=mass)
    cells = np.unravel_index(picks, cell.shape)
    out = np.empty((n_draws, len(active)))
    for i, name in enumerate(active):
        lower = axes_z[i][cells[i]]
        out[:, i] = lower + rng.random(n_draws) * widths[i][cells[i]]
        if TRANSFORM[name] == "log":
            out[:, i] = np.exp(out[:, i])
    return out
