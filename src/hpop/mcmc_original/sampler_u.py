"""Row-wise random-walk Metropolis-Hastings over the latent matrices ``U_k``.

Prior. Each role's latent row is an independent draw

    U[k, j, :] ~ Normal(0, Sigma_rho),
    Sigma_rho  = (1 - rho) I_d + rho * 1 1^T

so the ``d`` coordinates of a role are exchangeable and mildly correlated. The log
density is evaluated through a Cholesky factorisation — never by inverting Sigma.

Proposal. One role row at a time,

    U'[j, :] = U[j, :] + sigma_U * eta,   eta ~ Normal(0, I_d)

which is symmetric, so the Hastings ratio is 1 and

    log alpha = [log prior(U') - log prior(U)]
              + [log lik(data | U') - log lik(data | U)].

Note what the latent representation does to this surface: the BPOP likelihood reads
``U`` only through the induced order ``h(U)``, so the likelihood is **piecewise
constant** in ``U``. Inside an order region the chain moves under the prior alone;
the likelihood only ever speaks at region boundaries. That is why the sampler mixes
across orders by jumping rather than diffusing, and why acceptance rates here are
driven mostly by the prior.

``U`` is never compared to the true ``U`` coordinate-wise — that quantity is not
identifiable. Recovery is judged on the induced precedence relations instead.
"""

from __future__ import annotations

import math

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u

__all__ = [
    "sigma_rho_matrix",
    "log_u_prior",
    "propose_row",
    "u_row_sweep",
    "run_u_mcmc",
]


def sigma_rho_matrix(d: int, rho: float) -> np.ndarray:
    """``Sigma_rho = (1 - rho) I_d + rho 1 1^T``, validated positive definite."""
    if d < 1:
        raise ValueError(f"latent dimension must be >= 1, got {d}")
    # Sigma has eigenvalues 1 + (d-1)rho (once) and 1 - rho (d-1 times).
    lower = -1.0 / (d - 1) if d > 1 else -1.0
    if not (lower < rho < 1.0):
        raise ValueError(
            f"rho={rho} does not give a positive-definite Sigma for d={d}; "
            f"need {lower} < rho < 1"
        )
    sigma = (1.0 - rho) * np.eye(d) + rho * np.ones((d, d))
    eigenvalues = np.linalg.eigvalsh(sigma)
    if eigenvalues.min() <= 0.0:
        raise ValueError(f"Sigma_rho is not positive definite; eigenvalues {eigenvalues}")
    return sigma


def log_u_prior(u: np.ndarray, rho: float) -> float:
    """Sum of independent ``Normal(0, Sigma_rho)`` log densities over the rows of U."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 2:
        raise ValueError(f"u must be 2-D, got ndim={u.ndim}")
    if not np.all(np.isfinite(u)):
        return -math.inf

    m, d = u.shape
    sigma = sigma_rho_matrix(d, rho)
    chol = np.linalg.cholesky(sigma)
    log_det = 2.0 * float(np.log(np.diag(chol)).sum())
    solved = np.linalg.solve(chol, u.T)  # d x m
    quadratic = float((solved**2).sum())
    return -0.5 * (m * d * math.log(2.0 * math.pi) + m * log_det + quadratic)


def propose_row(
    u: np.ndarray, row: int, sigma_u: float, rng: np.random.Generator
) -> np.ndarray:
    """Symmetric Gaussian random-walk proposal on a single role row."""
    if sigma_u <= 0.0:
        raise ValueError(f"sigma_u must be positive, got {sigma_u}")
    if not (0 <= row < u.shape[0]):
        raise ValueError(f"row={row} outside range({u.shape[0]})")
    candidate = np.array(u, dtype=float, copy=True)
    candidate[row] = candidate[row] + sigma_u * rng.normal(size=u.shape[1])
    return candidate


def u_row_sweep(
    u: np.ndarray,
    log_likelihood,
    rho: float,
    sigma_u: float,
    rng: np.random.Generator,
    current_log_likelihood: float | None = None,
) -> tuple[np.ndarray, float, int, int]:
    """One MH pass over every role row of a single skill's ``U``.

    Args:
        log_likelihood: callable ``u -> float``, the aggregated corpus log-likelihood.
        current_log_likelihood: cached value for the incoming ``u``, to avoid
            recomputing it.

    Returns:
        ``(u, log_likelihood(u), n_accepted, n_proposed)``.
    """
    u = np.array(u, dtype=float, copy=True)
    current_ll = (
        log_likelihood(u) if current_log_likelihood is None else current_log_likelihood
    )
    current_prior = log_u_prior(u, rho)

    accepted = 0
    for row in range(u.shape[0]):
        candidate = propose_row(u, row, sigma_u, rng)
        candidate_prior = log_u_prior(candidate, rho)
        if candidate_prior == -math.inf:
            continue
        candidate_ll = log_likelihood(candidate)
        log_alpha = (candidate_prior - current_prior) + (candidate_ll - current_ll)
        if log_alpha >= 0.0 or math.log(rng.random()) < log_alpha:
            u = candidate
            current_prior = candidate_prior
            current_ll = candidate_ll
            accepted += 1

    return u, current_ll, accepted, u.shape[0]


def run_u_mcmc(
    evaluator,
    counts: np.ndarray,
    rho: float,
    sigma_u: float,
    n_iterations: int,
    burn_in: int,
    thin: int,
    rng: np.random.Generator,
    init_u: np.ndarray,
) -> dict:
    """Row-wise RW-MH over one skill's ``U``, with the corpus aggregated into counts.

    Returns saved ``U`` draws plus the log-posterior trace and acceptance rate.
    """
    if n_iterations <= burn_in:
        raise ValueError("n_iterations must exceed burn_in")
    if thin < 1:
        raise ValueError("thin must be >= 1")

    def log_likelihood(u: np.ndarray) -> float:
        return evaluator.counts_log_likelihood(u, counts)

    u = np.array(init_u, dtype=float, copy=True)
    current_ll = log_likelihood(u)

    samples: list[np.ndarray] = []
    log_post = np.empty(n_iterations, dtype=float)
    accepted = 0
    proposed = 0

    for i in range(n_iterations):
        u, current_ll, n_acc, n_prop = u_row_sweep(
            u, log_likelihood, rho, sigma_u, rng, current_log_likelihood=current_ll
        )
        accepted += n_acc
        proposed += n_prop
        log_post[i] = log_u_prior(u, rho) + current_ll
        if i >= burn_in and (i - burn_in) % thin == 0:
            samples.append(u.copy())

    return {
        "samples": np.array(samples),
        "log_posterior": log_post,
        "log_posterior_kept": log_post[burn_in:],
        "acceptance_rate": accepted / proposed if proposed else float("nan"),
        "n_iterations": n_iterations,
        "burn_in": burn_in,
        "thin": thin,
        "sigma_u": sigma_u,
        "final_u": u,
    }


def dispersed_initial_u(
    shape: tuple[int, int], rho: float, rng: np.random.Generator, scale: float = 2.0
) -> np.ndarray:
    """An over-dispersed prior draw, for starting independent chains far apart."""
    m, d = shape
    sigma = sigma_rho_matrix(d, rho)
    chol = np.linalg.cholesky(sigma)
    return scale * (rng.normal(size=(m, d)) @ chol.T)


def precedence_of_samples(samples: np.ndarray) -> np.ndarray:
    """Stack of induced precedence matrices, one per saved ``U`` draw."""
    return np.array([precedence_from_u(u) for u in samples])
