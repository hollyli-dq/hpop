"""Stage 6B1 — a generic scalar Metropolis-Hastings engine, and the recurrent targets.

Stage 6B0 established *what* the four one-dimensional posteriors are, by numerically
normalized grids. This module's only job is to draw from those same targets by MCMC, so
that Stage 6B1 can show

    p_MCMC(theta | D)  ~=  p_grid(theta | D)

rather than the much weaker "the truth is inside the interval".

Nothing here calls `mcmc_simulation_po`. That sampler is the verified source of the
proposal mathematics — in particular the ``log(beta') - log(beta)`` Jacobian on the
log-scale random walk — but it cycles among U / rho / noise, calls the *static*
likelihood cache, and assumes non-repeated total orders. The proposal equations are
reused; the likelihood callback is replaced.

Two evaluation paths, and the distinction is not an optimisation detail:

* ``beta``, ``lambda_rep``, ``lambda_back`` do not enter the validity recursion, so the
  whole ``q`` trajectory can be cached once and reused for every proposed value. Exact,
  not approximate.
* ``omega`` *does* enter the recursion through ``kappa = sigmoid(omega)``. Every proposed
  value replays ``q`` from zero for every block. Reusing a ``q`` trajectory across omega
  values would measure only the direct one-step channel and understate the parameter by
  more than an order of magnitude. The full-replay evaluator is vectorized across blocks,
  which is what makes a 20,000-iteration omega chain take about a minute instead of a day.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from hpop.mcmc_original.recurrent_scalar_posterior import (
    PRIORS, TRUE_VALUES, batch_recurrent_log_likelihood_full_replay,
    cached_batch_log_likelihood, log_prior, precompute_state_features,
)
from hpop.mcmc_original.recurrent_rfs import sigmoid

__all__ = [
    "Proposal", "ScalarMHConfig", "ScalarMHResult",
    "gaussian_random_walk", "log_scale_random_walk", "lognormal_log_density",
    "normal_log_density", "build_proposal", "scalar_mh_step", "run_scalar_mh",
    "PROPOSAL_KIND", "REGISTERED_STARTS", "PILOT_SEED_OFFSET",
    "RecurrentScalarTarget", "blockwise_recurrent_log_likelihood",
    "curvature_proposal_scale", "tune_proposal_scale",
]

# Which random walk each parameter uses. Registered, and not tunable: it follows from the
# support, not from anything observed in the data.
PROPOSAL_KIND = {
    "beta": "log", "lambda_rep": "log", "lambda_back": "log", "omega": "identity",
}

# Dispersed starting values. The first and third of each row are the two registered smoke
# starts; the full run needs four, so two interior/exterior values extend the same spread.
REGISTERED_STARTS = {
    "beta":        (0.5, 1.0, 2.5, 4.0),
    "omega":       (-0.5, 0.8, 3.5, 5.0),
    "lambda_rep":  (0.15, 0.5, 1.5, 3.0),
    "lambda_back": (0.05, 0.20, 0.70, 1.50),
}
SMOKE_STARTS = {name: (row[0], row[2]) for name, row in REGISTERED_STARTS.items()}

PILOT_SEED_OFFSET = 900_000        # keeps pilot streams disjoint from registered chains


# --------------------------------------------------------------------------- proposals
@dataclass(frozen=True)
class Proposal:
    """A proposed value together with ``log q(current | proposed) - log q(proposed | current)``."""

    value: float
    log_q_reverse_minus_forward: float


def gaussian_random_walk(scale: float) -> Callable[[float, np.random.Generator], Proposal]:
    """Symmetric walk on the original scale. Used for ``omega``, whose support is all of R."""
    if not scale > 0:
        raise ValueError(f"proposal scale must be positive, got {scale}")

    def propose(current: float, rng: np.random.Generator) -> Proposal:
        return Proposal(float(current + scale * rng.normal()), 0.0)

    return propose


def log_scale_random_walk(scale: float) -> Callable[[float, np.random.Generator], Proposal]:
    """``log theta' = log theta + eta``, i.e. ``theta' = theta * exp(eta)``.

    The proposal is symmetric in ``log theta`` but not in ``theta``, so the MH ratio
    carries ``log theta' - log theta``. This is the same correction the verified static
    beta sampler applies, and dropping it visibly shifts the stationary distribution —
    see the negative control in ``test_stage6b1_scalar_mh.py``.
    """
    if not scale > 0:
        raise ValueError(f"proposal scale must be positive, got {scale}")

    def propose(current: float, rng: np.random.Generator) -> Proposal:
        if not current > 0:
            raise ValueError(f"log-scale walk requires a positive state, got {current}")
        value = float(current * math.exp(scale * rng.normal()))
        return Proposal(value, math.log(value) - math.log(current))

    return propose


def lognormal_log_density(x: float, centre: float, scale: float) -> float:
    """``log q(x | centre)`` for the lognormal proposal, written out in full.

    Only the tests use this: they verify the analytic ratio the samplers rely on instead
    of trusting the algebra.
    """
    if x <= 0 or centre <= 0:
        return -math.inf
    z = (math.log(x) - math.log(centre)) / scale
    return -math.log(x) - math.log(scale) - 0.5 * math.log(2.0 * math.pi) - 0.5 * z * z


def normal_log_density(x: float, centre: float, scale: float) -> float:
    z = (x - centre) / scale
    return -math.log(scale) - 0.5 * math.log(2.0 * math.pi) - 0.5 * z * z


# ----------------------------------------------------------------------- generic engine
@dataclass(frozen=True)
class ScalarMHConfig:
    parameter_name: str
    proposal_scale: float
    initial_value: float
    num_iterations: int
    burn_in: int
    thin: int
    seed: int

    def __post_init__(self) -> None:
        if self.burn_in >= self.num_iterations:
            raise ValueError("burn_in must be smaller than num_iterations")
        if self.thin < 1:
            raise ValueError("thin must be at least 1")


@dataclass
class ScalarMHResult:
    samples: np.ndarray
    log_likelihood: np.ndarray
    log_posterior: np.ndarray
    proposed: int
    accepted: int
    runtime_seconds: float
    config: ScalarMHConfig | None = None
    proposed_after_burn_in: int = 0
    accepted_after_burn_in: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else float("nan")

    @property
    def post_burn_in_acceptance_rate(self) -> float:
        """The rate the registered ``[0.15, 0.60]`` gate is read against.

        Chains start dispersed, so the transient accepts almost every downhill move and
        inflates the overall rate. The stationary-phase rate is the one that says whether
        the proposal scale is right.
        """
        if not self.proposed_after_burn_in:
            return self.acceptance_rate
        return self.accepted_after_burn_in / self.proposed_after_burn_in


def scalar_mh_step(
    current: float,
    current_log_posterior: float,
    log_posterior_fn: Callable[[float], float],
    proposal_fn: Callable[..., Proposal],
    rng: np.random.Generator,
) -> tuple[float, float, bool]:
    """One Metropolis-Hastings step for a single scalar."""
    proposal = proposal_fn(current, rng)

    proposed_log_posterior = log_posterior_fn(proposal.value)

    log_acceptance = (
        proposed_log_posterior
        - current_log_posterior
        + proposal.log_q_reverse_minus_forward
    )

    accepted = bool(np.log(rng.random()) < min(0.0, log_acceptance))

    if accepted:
        return proposal.value, proposed_log_posterior, True

    return current, current_log_posterior, False


def run_scalar_mh(config: ScalarMHConfig, posterior_fn, proposal_fn) -> ScalarMHResult:
    """Run one chain.

    ``posterior_fn`` returns either the log posterior, or ``(log_posterior,
    log_likelihood)`` when the decomposition is available. The current value's log
    posterior is cached: a rejected step costs exactly one likelihood evaluation, never
    two.
    """
    rng = np.random.default_rng(config.seed)
    last = {}

    def wrapped(value: float) -> float:
        out = posterior_fn(value)
        if isinstance(out, tuple):
            last["pair"] = (float(out[0]), float(out[1]))
            return float(out[0])
        last["pair"] = (float(out), float("nan"))
        return float(out)

    current = float(config.initial_value)
    current_lp = wrapped(current)
    current_ll = last["pair"][1]
    if not math.isfinite(current_lp):
        raise ValueError(
            f"{config.parameter_name}: initial value {current} has non-finite log posterior")

    kept_at = np.arange(config.burn_in, config.num_iterations, config.thin)
    keep = np.zeros(config.num_iterations, dtype=bool)
    keep[kept_at] = True
    samples = np.empty(kept_at.size)
    lls = np.empty(kept_at.size)
    lps = np.empty(kept_at.size)

    proposed = accepted = 0
    proposed_post = accepted_post = 0
    k = 0
    start = time.perf_counter()
    for i in range(config.num_iterations):
        new_value, new_lp, was_accepted = scalar_mh_step(
            current, current_lp, wrapped, proposal_fn, rng)
        proposed += 1
        if i >= config.burn_in:
            proposed_post += 1
        if was_accepted:
            accepted += 1
            if i >= config.burn_in:
                accepted_post += 1
            current, current_lp, current_ll = new_value, new_lp, last["pair"][1]
        if keep[i]:
            samples[k], lls[k], lps[k] = current, current_ll, current_lp
            k += 1
    runtime = time.perf_counter() - start

    return ScalarMHResult(samples, lls, lps, proposed, accepted, runtime, config,
                          proposed_post, accepted_post)


# ------------------------------------------------------------------- recurrent targets
def blockwise_recurrent_log_likelihood(role_array, u, beta, epsilon, omega,
                                       lambda_rep, lambda_back) -> np.ndarray:
    """Per-block log likelihoods, full replay, vectorized across blocks.

    Same arithmetic as ``batch_recurrent_log_likelihood_full_replay``; it returns the
    ``(n_blocks,)`` vector instead of the scalar sum, which the posterior predictive
    score needs. Verified block-by-block against ``recurrent_rfs_log_likelihood``.
    """
    from hpop.mcmc_original.latent_poset import precedence_from_u

    u = np.asarray(u, dtype=float)
    precedence = precedence_from_u(u)
    m = u.shape[0]
    roles = np.asarray(role_array, dtype=int)
    n, T = roles.shape
    kappa = sigmoid(omega)

    pred_mask = precedence.T.astype(bool)
    succ = precedence.astype(float)
    succ_off = succ.copy()
    np.fill_diagonal(succ_off, 0.0)

    q = np.zeros((n, m))
    total = np.zeros(n)
    rows = np.arange(n)
    for t in range(T):
        feasibility = np.prod(np.where(pred_mask[None, :, :], q[:, None, :], 1.0), axis=2)
        stale = (1.0 - q) @ succ.T
        back = kappa * (q @ succ_off.T)
        exponent = beta * np.log1p(stale) - lambda_rep * q - lambda_back * back
        exponent -= exponent.max(axis=1, keepdims=True)
        weights = feasibility * np.exp(exponent)
        mixed = (1.0 - epsilon) * (weights / weights.sum(axis=1, keepdims=True)) + epsilon / m
        observed = roles[:, t]
        total += np.log(mixed[rows, observed])
        gate = np.where(precedence[observed], kappa, 0.0)
        q = q * (1.0 - gate)
        q[rows, observed] = 1.0
    return total


class RecurrentScalarTarget:
    """Log posterior of ONE recurrent scalar, the other three held at their true values.

    ``U``, ``epsilon`` and the remaining scalars are fixed. No U-update, no rho-update,
    no latent-dimension move, no dimension-proportional schedule: those belong to Stage
    6C, and including them here would stop this from being a test of the sampler against
    a known one-dimensional target.
    """

    def __init__(self, parameter: str, role_array, u, truth: dict, epsilon: float,
                 use_cache: bool = True):
        if parameter not in PRIORS:
            raise ValueError(f"unknown parameter {parameter!r}")
        self.parameter = parameter
        self.role_array = np.asarray(role_array, dtype=int)
        self.u = np.asarray(u, dtype=float)
        self.truth = dict(truth)
        self.epsilon = float(epsilon)
        self.calls = 0

        # The feature cache holds a q trajectory, so it is valid only at a fixed omega.
        # For omega itself it is never built, and every evaluation replays from zero.
        self.cached = bool(use_cache) and parameter != "omega"
        if self.cached:
            self._features = precompute_state_features(
                [tuple(row) for row in self.role_array], self.u, self.truth["omega"])
        else:
            self._features = None

    def log_likelihood(self, value: float) -> float:
        self.calls += 1
        if self.cached:
            kw = {"beta": self.truth["beta"], "lambda_rep": self.truth["lambda_rep"],
                  "lambda_back": self.truth["lambda_back"]}
            kw[self.parameter] = float(value)
            return cached_batch_log_likelihood(
                self._features, kw["beta"], self.epsilon, kw["lambda_rep"], kw["lambda_back"])
        kw = dict(self.truth)
        kw[self.parameter] = float(value)
        return float(batch_recurrent_log_likelihood_full_replay(
            self.role_array, self.u, kw["beta"], self.epsilon, kw["omega"],
            kw["lambda_rep"], kw["lambda_back"]))

    def log_prior(self, value: float) -> float:
        return log_prior(self.parameter, float(value))

    def __call__(self, value: float) -> tuple[float, float]:
        """Returns ``(log_posterior, log_likelihood)``; ``-inf`` outside the support."""
        lp = self.log_prior(value)
        if not math.isfinite(lp):
            return -math.inf, -math.inf
        ll = self.log_likelihood(value)
        return lp + ll, ll


# -------------------------------------------------------------------------- scale rules
def curvature_proposal_scale(target: RecurrentScalarTarget, centre: float,
                             delta: float | None = None) -> dict:
    """Initial random-walk scale from the observed likelihood curvature at ``centre``.

    A proposal scale cannot bias the stationary distribution, only the efficiency with
    which it is reached, so this is a starting point for the pilot rather than a result.
    It is computed from the data by central differences — the Stage 6B0 reference grids
    are never consulted, so the comparison in section 9 stays honest.

    For a log-scale walk the relevant spread is that of ``log theta``, hence the division
    by ``centre``.
    """
    delta = delta if delta is not None else max(0.02, 0.02 * abs(centre))
    up, mid, down = (target.log_likelihood(centre + delta),
                     target.log_likelihood(centre),
                     target.log_likelihood(centre - delta))
    second = (up - 2.0 * mid + down) / delta ** 2
    curvature = -second
    if not curvature > 0:
        raise ValueError(f"non-positive curvature {curvature} at {centre}")
    sd = 1.0 / math.sqrt(curvature)
    if PROPOSAL_KIND[target.parameter] == "log":
        sd = sd / abs(centre)
    return {"centre": float(centre), "delta": float(delta),
            "observed_curvature": float(curvature), "implied_sd": float(sd),
            "scale": float(2.4 * sd)}


def tune_proposal_scale(target: RecurrentScalarTarget, initial_scale: float,
                        start: float, seed: int, iterations: int = 2000,
                        acceptance_window: tuple[float, float] = (0.20, 0.55)) -> dict:
    """One 2,000-iteration pilot, and at most one adjustment.

    If the pilot's acceptance rate already sits inside the window the initial scale is
    kept. Otherwise the scale is set once from the pilot's own posterior spread, and the
    pilot draws are then discarded — the registered chains restart from the registered
    dispersed starts, so no pilot state leaks into the reported posterior.
    """
    kind = PROPOSAL_KIND[target.parameter]
    build = log_scale_random_walk if kind == "log" else gaussian_random_walk
    config = ScalarMHConfig(target.parameter, initial_scale, start, iterations,
                            iterations // 2, 1, seed)
    pilot = run_scalar_mh(config, target, build(initial_scale))
    rate = pilot.post_burn_in_acceptance_rate

    if acceptance_window[0] <= rate <= acceptance_window[1]:
        return {"initial_scale": float(initial_scale), "pilot_acceptance_rate": float(rate),
                "adjusted": False, "final_scale": float(initial_scale),
                "pilot_iterations": iterations, "pilot_seed": seed,
                "pilot_start": float(start)}

    draws = pilot.samples
    spread = float(np.std(np.log(draws) if kind == "log" else draws, ddof=1))
    if not spread > 0:
        raise ValueError(f"{target.parameter}: pilot chain did not move at scale {initial_scale}")
    final = 2.4 * spread
    return {"initial_scale": float(initial_scale), "pilot_acceptance_rate": float(rate),
            "adjusted": True, "final_scale": float(final),
            "pilot_spread": spread, "pilot_iterations": iterations, "pilot_seed": seed,
            "pilot_start": float(start)}


def build_proposal(parameter: str, scale: float):
    kind = PROPOSAL_KIND[parameter]
    return (log_scale_random_walk if kind == "log" else gaussian_random_walk)(scale)


def default_truth() -> dict:
    return dict(TRUE_VALUES)
