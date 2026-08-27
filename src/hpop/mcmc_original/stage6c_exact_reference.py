"""Stage 6C — the exact poset catalogue and the independent reference posteriors.

## What the catalogue is, and what it is not

The Stage 6C target is continuous in `U`. The catalogue below is **not** the state space
and **not** a prior over orders — it is a complete labelling of the orders `h(U)` can
induce, used for two things only: to say which order a draw of `U` fell in, and to build a
reference for comparison.

That reference is a genuine *marginal* of the correct continuous target, not a substitute
for it. Because the likelihood reads `U` only through `h(U)`,

    p(P, rho | Y)  ∝  p(rho) · L(P) · pi_rho(P),
    pi_rho(P) = P_{U ~ N(0, Sigma_rho)}[h(U) = P],

which is exact: the integral of `p(Y|U) p(U|rho)` over `{U : h(U) = P}` factorises because
the likelihood is constant on that set. Nothing is redefined and `U` is not reparameterised
— it is marginalised, which is a legitimate operation on the target actually being sampled.
Conditional on a cell, `U` is exactly the prior truncated to it.

## Why the catalogue is exhaustive rather than sampled

For `d = 2`, `h(U)` depends on `U` only through the two column rankings, so enumerating all
`m!^d` ranking tuples enumerates every reachable order exactly. At `m = 5, d = 2` that is
`14,400` pairs yielding **4231** distinct labelled posets — the total number of labelled
partial orders on 5 elements, because every poset on at most 5 elements has order dimension
at most 2. Nothing is unreachable and nothing is approximated.

`L(P)` is exact for every entry, from the frozen Stage 6B likelihood. The one Monte Carlo
ingredient is `pi_rho(P)`, computed from the **prior alone** — no data, no likelihood, no
chain — with common random numbers across the `rho` grid and a reported standard error.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_joint_scalar_mcmc import vectorized_state_features
from hpop.mcmc_original.recurrent_scalar_posterior import (
    cached_batch_log_likelihood, log_prior,
)
from hpop.mcmc_original.sampler_u import sigma_rho_matrix
from hpop.mcmc_original.stage6c_frozen import RHO_UPPER, log_rho_prior

__all__ = [
    "PosetCatalogue", "build_catalogue", "transitive_reduction", "is_partial_order",
    "prior_cell_masses", "poset_log_likelihoods", "poset_log_likelihood_beta_table",
    "Stage6CReference", "build_6c1_reference", "build_6c2_reference",
    "reference_summary", "sample_reference_draws",
]


# ------------------------------------------------------------------------- poset utils
def is_partial_order(p: np.ndarray) -> bool:
    """Irreflexive, antisymmetric and transitive — checked, not assumed."""
    p = np.asarray(p, dtype=bool)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        return False
    if p.diagonal().any():
        return False
    if (p & p.T).any():
        return False
    return not (((p.astype(int) @ p.astype(int)) > 0) & ~p).any()


def transitive_reduction(p: np.ndarray) -> np.ndarray:
    """Cover relation of a transitively closed order: drop edges with an intermediate."""
    p = np.asarray(p, dtype=bool)
    two_step = (p.astype(int) @ p.astype(int)) > 0
    return p & ~two_step


def _relation_keys(precedence: np.ndarray) -> np.ndarray:
    """Pack an `(n, m, m)` stack of boolean orders into `n` integer keys."""
    flat = np.asarray(precedence, dtype=bool).reshape(precedence.shape[0], -1)
    weights = (1 << np.arange(flat.shape[1], dtype=np.int64))
    return flat.astype(np.int64) @ weights


@dataclass
class PosetCatalogue:
    m: int
    d: int
    closures: np.ndarray             # (R, m, m) bool
    reductions: np.ndarray           # (R, m, m) bool
    relation_counts: np.ndarray      # (R,)
    representatives: np.ndarray      # (R, m, d) a U realising each order
    ranking_tuple_counts: np.ndarray  # (R,) how many ranking tuples induce it
    keys: np.ndarray                 # (R,) integer keys, sorted
    order: np.ndarray                # (R,) permutation restoring catalogue order

    @property
    def size(self) -> int:
        return int(self.closures.shape[0])

    def index_of(self, precedence: np.ndarray) -> int:
        key = int(_relation_keys(np.asarray(precedence, dtype=bool)[None, ...])[0])
        pos = int(np.searchsorted(self.keys, key))
        if pos < self.keys.size and self.keys[pos] == key:
            return int(self.order[pos])
        return -1

    def indices_of(self, precedence_stack: np.ndarray) -> np.ndarray:
        """Vectorised lookup for a whole stack of orders."""
        keys = _relation_keys(np.asarray(precedence_stack, dtype=bool))
        pos = np.searchsorted(self.keys, keys)
        pos = np.clip(pos, 0, self.keys.size - 1)
        hit = self.keys[pos] == keys
        return np.where(hit, self.order[pos], -1)

    def index_of_u(self, u: np.ndarray) -> int:
        return self.index_of(precedence_from_u(u))


def build_catalogue(m: int = 5, d: int = 2) -> PosetCatalogue:
    """Exhaustive enumeration of the orders reachable by a d-column product order."""
    if m < 1 or d < 1:
        raise ValueError(f"need m >= 1 and d >= 1, got m={m}, d={d}")
    perms = list(itertools.permutations(range(m)))
    index: dict[bytes, int] = {}
    closures: list[np.ndarray] = []
    representatives: list[np.ndarray] = []
    counts: list[int] = []

    for ranks in itertools.product(perms, repeat=d):
        u = np.array([[ranks[r][i] for r in range(d)] for i in range(m)], dtype=float)
        precedence = precedence_from_u(u)
        key = precedence.tobytes()
        found = index.get(key)
        if found is None:
            index[key] = len(closures)
            closures.append(precedence)
            representatives.append(u)
            counts.append(1)
        else:
            counts[found] += 1

    closure_stack = np.array(closures)
    keys = _relation_keys(closure_stack)
    order = np.argsort(keys)
    return PosetCatalogue(
        m=m, d=d, closures=closure_stack,
        reductions=np.array([transitive_reduction(p) for p in closure_stack]),
        relation_counts=closure_stack.reshape(len(closures), -1).sum(axis=1),
        representatives=np.array(representatives),
        ranking_tuple_counts=np.array(counts, dtype=int),
        keys=keys[order], order=order.astype(int))


# --------------------------------------------------------------- prior cell masses
def prior_cell_masses(catalogue: PosetCatalogue, rho_grid, n_draws: int, seed: int,
                      chunk: int = 100_000) -> dict:
    """`pi_rho(P)` for every entry and every `rho`, from the PRIOR alone.

    Common random numbers across the `rho` grid: one set of standard normals is mapped
    through each `Sigma_rho`'s Cholesky factor, so the estimates are smooth in `rho` and
    their differences are far better determined than independent draws would be.
    """
    rho_grid = np.asarray(rho_grid, dtype=float)
    m, d = catalogue.m, catalogue.d
    counts = np.zeros((rho_grid.size, catalogue.size))
    unseen = 0
    rng = np.random.default_rng(seed)
    chols = [np.linalg.cholesky(sigma_rho_matrix(d, float(r))) for r in rho_grid]
    diag = np.arange(m)

    remaining = int(n_draws)
    while remaining > 0:
        take = min(chunk, remaining)
        base = rng.normal(size=(take, m, d))
        for gi, chol in enumerate(chols):
            u = base @ chol.T
            precedence = np.all(u[:, :, None, :] > u[:, None, :, :], axis=-1)
            precedence[:, diag, diag] = False
            idx = catalogue.indices_of(precedence)
            unseen += int((idx < 0).sum())
            valid = idx[idx >= 0]
            counts[gi] += np.bincount(valid, minlength=catalogue.size)
        remaining -= take

    total = counts.sum(axis=1, keepdims=True)
    masses = counts / total
    standard_error = np.sqrt(np.maximum(masses * (1.0 - masses), 0.0) / total)
    return {"rho_grid": rho_grid, "masses": masses, "counts": counts,
            "n_draws": int(n_draws), "seed": int(seed),
            "standard_error": standard_error,
            "max_standard_error": float(standard_error.max()),
            "unseen_draws": int(unseen),
            "provenance": "prior draws only: no data, no likelihood, no MCMC"}


# ------------------------------------------------------------------- exact likelihoods
def poset_log_likelihoods(catalogue: PosetCatalogue, role_array, epsilon, beta, omega,
                          lambda_rep, lambda_back) -> np.ndarray:
    """Exact `L(P)` for every catalogue entry, at fixed scalars."""
    out = np.empty(catalogue.size)
    for i, u in enumerate(catalogue.representatives):
        features = vectorized_state_features(role_array, u, omega)
        out[i] = cached_batch_log_likelihood(features, beta, epsilon, lambda_rep,
                                             lambda_back)
    return out


def poset_log_likelihood_beta_table(catalogue: PosetCatalogue, role_array, epsilon, betas,
                                    omega, lambda_rep, lambda_back) -> np.ndarray:
    """Exact `L(P, beta)` table. Features are built once per order, beta is batched."""
    betas = np.asarray(betas, dtype=float)
    out = np.empty((catalogue.size, betas.size))
    for i, u in enumerate(catalogue.representatives):
        f = vectorized_state_features(role_array, u, omega)
        exponent = (betas[:, None, None, None] * f["Q"][None, ...]
                    - lambda_rep * f["q"][None, ...]
                    - lambda_back * f["C_back"][None, ...])
        exponent -= exponent.max(axis=-1, keepdims=True)
        weights = f["F"][None, ...] * np.exp(exponent)
        mixed = ((1.0 - epsilon) * (weights / weights.sum(axis=-1, keepdims=True))
                 + epsilon / f["m"])
        n, T = f["obs"].shape
        chosen = mixed[:, np.arange(n)[:, None], np.arange(T)[None, :], f["obs"]]
        out[i] = np.log(chosen).sum(axis=(1, 2))
    return out


# ------------------------------------------------------------------------- references
@dataclass
class Stage6CReference:
    stage: str
    catalogue: PosetCatalogue
    rho_grid: np.ndarray
    beta_grid: np.ndarray | None
    log_joint: np.ndarray            # (R, n_rho) or (R, n_rho, n_beta), unnormalised
    joint: np.ndarray                # normalised to sum/integrate to 1
    cell_masses: dict
    config: dict


def _normalise(log_joint, rho_grid, beta_grid):
    """Trapezoid in the continuous coordinates, plain sum over the discrete order."""
    shift = float(np.max(log_joint))
    weights = np.exp(log_joint - shift)
    total = weights.sum(axis=0)                       # over posets
    total = np.trapezoid(total, rho_grid, axis=0)
    if beta_grid is not None:
        total = np.trapezoid(total, beta_grid, axis=0)
    return weights / float(total), shift + math.log(float(total))


def build_6c1_reference(catalogue, role_array, truth, epsilon, rho_grid,
                        cell_masses: dict) -> Stage6CReference:
    """`p(P, rho | Y)` with all four recurrent scalars fixed at their registered values."""
    rho_grid = np.asarray(rho_grid, dtype=float)
    log_likelihood = poset_log_likelihoods(
        catalogue, role_array, epsilon, truth["beta"], truth["omega"],
        truth["lambda_rep"], truth["lambda_back"])
    log_mass = np.log(np.maximum(cell_masses["masses"], 1e-300)).T        # (R, n_rho)
    log_rho = np.array([log_rho_prior(float(r)) for r in rho_grid])
    log_joint = log_likelihood[:, None] + log_mass + log_rho[None, :]
    joint, log_z = _normalise(log_joint, rho_grid, None)
    return Stage6CReference(
        stage="6c1", catalogue=catalogue, rho_grid=rho_grid, beta_grid=None,
        log_joint=log_joint, joint=joint, cell_masses=cell_masses,
        config={"method": "exact enumeration over the 4231-poset catalogue x rho grid",
                "likelihood": "exact, frozen Stage 6B recurrent likelihood",
                "cell_masses": "prior Monte Carlo, common random numbers across rho",
                "log_normaliser": log_z, "n_rho": int(rho_grid.size),
                "fixed_scalars": {k: float(v) for k, v in truth.items()},
                "uses_mcmc": False})


def build_6c2_reference(catalogue, role_array, truth, epsilon, rho_grid, beta_grid,
                        cell_masses: dict, log_likelihood_table=None
                        ) -> Stage6CReference:
    """`p(P, rho, beta | Y)` with omega, lambda_rep and lambda_back fixed.

    `log_likelihood_table` is the exact `(R, n_beta)` table of `L(P, beta)`. It is
    recomputed when omitted; passing one already computed for this exact `beta_grid`
    avoids repeating roughly twenty minutes of identical function evaluation, and is
    numerically indistinguishable from recomputing it.
    """
    rho_grid = np.asarray(rho_grid, dtype=float)
    beta_grid = np.asarray(beta_grid, dtype=float)
    if log_likelihood_table is None:
        table = poset_log_likelihood_beta_table(
            catalogue, role_array, epsilon, beta_grid, truth["omega"],
            truth["lambda_rep"], truth["lambda_back"])                    # (R, n_beta)
    else:
        table = np.asarray(log_likelihood_table, dtype=float)
        if table.shape != (catalogue.size, beta_grid.size):
            raise ValueError(
                f"log_likelihood_table has shape {table.shape}, expected "
                f"{(catalogue.size, beta_grid.size)}")
    log_mass = np.log(np.maximum(cell_masses["masses"], 1e-300)).T        # (R, n_rho)
    log_rho = np.array([log_rho_prior(float(r)) for r in rho_grid])
    log_beta = np.array([log_prior("beta", float(b)) for b in beta_grid])
    log_joint = (table[:, None, :] + log_mass[:, :, None]
                 + log_rho[None, :, None] + log_beta[None, None, :])
    joint, log_z = _normalise(log_joint, rho_grid, beta_grid)
    return Stage6CReference(
        stage="6c2", catalogue=catalogue, rho_grid=rho_grid, beta_grid=beta_grid,
        log_joint=log_joint, joint=joint, cell_masses=cell_masses,
        config={"method": "exact enumeration over posets x rho grid x beta grid",
                "likelihood": "exact L(P, beta), features built once per order",
                "cell_masses": "prior Monte Carlo, common random numbers across rho",
                "log_normaliser": log_z, "n_rho": int(rho_grid.size),
                "n_beta": int(beta_grid.size),
                "fixed_scalars": {k: float(truth[k])
                                  for k in ("omega", "lambda_rep", "lambda_back")},
                "uses_mcmc": False})


# --------------------------------------------------------------------------- summaries
def _integrate_continuous(array, rho_grid, beta_grid):
    out = np.trapezoid(array, rho_grid, axis=1)
    if beta_grid is not None:
        out = np.trapezoid(out, beta_grid, axis=1)
    return out


def reference_summary(reference: Stage6CReference, truth: dict) -> dict:
    """Poset probabilities, relation marginals, and the rho / beta marginals."""
    catalogue = reference.catalogue
    joint = reference.joint
    rho_grid, beta_grid = reference.rho_grid, reference.beta_grid

    poset_probability = _integrate_continuous(joint, rho_grid, beta_grid)
    poset_probability = poset_probability / poset_probability.sum()

    closure = catalogue.closures.reshape(catalogue.size, -1).astype(float)
    reduction = catalogue.reductions.reshape(catalogue.size, -1).astype(float)
    relation_marginal = poset_probability @ closure
    reduction_marginal = poset_probability @ reduction

    # rho marginal: sum over posets, then integrate out beta if present
    rho_marginal = joint.sum(axis=0)
    if beta_grid is not None:
        rho_marginal = np.trapezoid(rho_marginal, beta_grid, axis=1)
    rho_marginal = rho_marginal / np.trapezoid(rho_marginal, rho_grid)

    summary = {
        "n_posets": catalogue.size,
        "poset_probability": poset_probability,
        "relation_marginal": relation_marginal,
        "reduction_marginal": reduction_marginal,
        "relation_count_distribution": np.bincount(
            catalogue.relation_counts, weights=poset_probability,
            minlength=catalogue.m * (catalogue.m - 1) // 2 + 1),
        "rho": _continuous_summary(rho_grid, rho_marginal),
        "map_poset": int(np.argmax(poset_probability)),
        "map_probability": float(poset_probability.max()),
    }
    if beta_grid is not None:
        beta_marginal = joint.sum(axis=0)
        beta_marginal = np.trapezoid(beta_marginal, rho_grid, axis=0)
        beta_marginal = beta_marginal / np.trapezoid(beta_marginal, beta_grid)
        summary["beta"] = _continuous_summary(beta_grid, beta_marginal)
        summary["beta_marginal_density"] = beta_marginal
    summary["rho_marginal_density"] = rho_marginal
    return summary


def _continuous_summary(grid, density) -> dict:
    cdf = np.concatenate([[0.0], np.cumsum(
        0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    cdf = cdf / cdf[-1]
    quantile = lambda level: float(np.interp(level, cdf, grid))   # noqa: E731
    mean = float(np.trapezoid(grid * density, grid))
    var = float(np.trapezoid((grid - mean) ** 2 * density, grid))
    return {"mean": mean, "sd": math.sqrt(max(var, 0.0)), "median": quantile(0.5),
            "q025": quantile(0.025), "q975": quantile(0.975),
            "grid": grid, "cdf": cdf}


def sample_reference_draws(reference: Stage6CReference, summary: dict, n_draws: int,
                           seed: int) -> dict:
    """iid draws of `(poset index, rho[, beta])` from the frozen reference."""
    rng = np.random.default_rng(seed)
    joint = reference.joint
    rho_grid, beta_grid = reference.rho_grid, reference.beta_grid

    flat = joint.reshape(joint.shape[0], -1)
    cell = flat / flat.sum()
    picks = rng.choice(cell.size, size=n_draws, p=cell.ravel())
    if beta_grid is None:
        poset_index, rho_index = np.unravel_index(picks, joint.shape)
        beta_index = None
    else:
        poset_index, rho_index, beta_index = np.unravel_index(picks, joint.shape)

    def jitter(grid, index):
        step = np.gradient(grid)
        return np.clip(grid[index] + (rng.random(n_draws) - 0.5) * step[index],
                       grid.min(), grid.max())

    out = {"poset_index": poset_index, "rho": jitter(rho_grid, rho_index)}
    if beta_index is not None:
        out["beta"] = jitter(beta_grid, beta_index)
    out["relations"] = reference.catalogue.closures.reshape(
        reference.catalogue.size, -1)[poset_index].astype(float)
    out["relation_count"] = reference.catalogue.relation_counts[poset_index]
    return out
