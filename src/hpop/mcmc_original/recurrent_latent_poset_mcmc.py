"""Stage 6C1 / 6C2 — Metropolis-within-Gibbs over the latent `U`, `rho` and `beta`.

The target is continuous in `U`:

    p(U, rho[, beta] | Y)  ∝  p(Y | h(U), fixed scalars) · p(U | rho) · p(rho) [· p(beta)]

so the chain's state is the real matrix `U ∈ R^{m×d}` together with `rho` (and `beta` in
Stage 6C2). A poset identifier is never part of the state; `h(U)` is computed for
*reporting* and for the reference comparison only.

## The three updates, and what each of them carries

**`U`, one row at a time.** The Stage 2A kernel `sampler_u.propose_row` is reused
unchanged: `U'[j,:] = U[j,:] + sigma_U · Normal(0, I_d)`. That is symmetric, so
`log q(U|U') - log q(U'|U) = 0` *exactly* — not by approximation and not by assumption
about neighbourhood sizes, which is a property of this continuous kernel rather than
something that had to be arranged. Legality needs no check either: every real `U` induces
a strict partial order. `Sigma_rho` is shared by numerator and denominator, so the
Gaussian determinant cancels here.

**`rho`.** A random walk on `z = logit(rho)`, carrying two distinct corrections that are
easy to confuse:

* the transform Jacobian `log|d rho / d z| = log(rho(1 - rho))`, from the random walk;
* the Gaussian determinant inside `p(U | rho)`, `-(m/2) log|Sigma_rho|`, which does **not**
  cancel when `U` is fixed and `rho` moves.

The likelihood is *not* evaluated: `rho` acts only through `p(U | rho)`. That is asserted
by a test, and it makes a `rho` update free.

**`beta`** (Stage 6C2 only). The frozen Stage 6B kernel and proposal, reused as objects.

Sweep order is `U -> rho -> beta`, and each update sees the values accepted before it in
the same sweep.

## Replay after a `U` move

Changing `U` can change the induced order, and therefore the frontier, feasibility, the
whole `q` trajectory and every downstream one-step probability. So a proposed `U` is
always scored by a complete replay from `q_0 = 0` for every block. The evaluator caches
only quantities keyed on the *exact* `(precedence, omega)` it was built at, is written
only by an explicit refresh, and is never written by an evaluation — so a rejected
proposal cannot leave anything behind.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_joint_scalar_mcmc import (
    _jsonable_rng_state, _rng_state_from_jsonable, vectorized_state_features,
)
from hpop.mcmc_original.recurrent_scalar_mcmc import build_proposal, scalar_mh_step
from hpop.mcmc_original.recurrent_scalar_posterior import (
    cached_batch_log_likelihood, log_prior,
)
from hpop.mcmc_original.sampler_u import propose_row
from hpop.mcmc_original.stage6c_frozen import (
    ACTIVE_6C1, ACTIVE_6C2, SIGMA_U, SWEEP_ORDER, log_jacobian_rho, log_rho_prior,
    log_structural_prior, rho_from_unconstrained, rho_to_unconstrained,
)

__all__ = [
    "LatentPosetEvaluator", "Stage6CTarget", "Stage6CState", "Stage6CResult",
    "sweep_once", "run_stage6c_mcmc", "poset_key", "ACTIVE_6C1", "ACTIVE_6C2",
]


def poset_key(u: np.ndarray) -> bytes:
    """Canonical label for the order induced by `U` — a report key, not chain state."""
    return precedence_from_u(np.asarray(u, dtype=float)).tobytes()


# ------------------------------------------------------------------------- evaluator
class LatentPosetEvaluator:
    """Complete recurrent log likelihood as a function of `U` and the scalars.

    The state-feature bundle depends on the induced order and on `omega`; it is keyed on
    both and only ever written by `refresh_cache`. An evaluation never writes it, so
    rejection safety is structural rather than defensive.
    """

    def __init__(self, role_array, epsilon: float, omega: float):
        self.role_array = np.asarray(role_array, dtype=int)
        self.epsilon = float(epsilon)
        self.omega = float(omega)
        self.n_blocks, self.T = self.role_array.shape
        self._features = None
        self._key = None
        self.full_replay_calls = 0
        self.cached_calls = 0
        self.cache_builds = 0

    # -- cache lifecycle ---------------------------------------------------------------
    @property
    def cache_key(self):
        return self._key

    def cache_is_valid_for(self, u, omega: float) -> bool:
        return self._key is not None and self._key == (poset_key(u), float(omega))

    def invalidate(self) -> None:
        self._features = None
        self._key = None

    def refresh_cache(self, u, omega: float) -> None:
        self._features = vectorized_state_features(self.role_array, u, float(omega))
        self._key = (poset_key(u), float(omega))
        self.cache_builds += 1

    def ensure_cache(self, u, omega: float) -> None:
        if not self.cache_is_valid_for(u, omega):
            self.refresh_cache(u, omega)

    # -- evaluation --------------------------------------------------------------------
    def full_replay_log_likelihood(self, u, beta, omega, lambda_rep, lambda_back) -> float:
        """Rebuilds the whole `q` trajectory from zero for every block. No state reused."""
        self.full_replay_calls += 1
        features = vectorized_state_features(self.role_array, u, float(omega))
        return float(cached_batch_log_likelihood(
            features, beta, self.epsilon, lambda_rep, lambda_back))

    def log_likelihood(self, u, beta, omega, lambda_rep, lambda_back,
                       allow_cache: bool = True) -> float:
        if allow_cache and self.cache_is_valid_for(u, omega):
            self.cached_calls += 1
            return float(cached_batch_log_likelihood(
                self._features, beta, self.epsilon, lambda_rep, lambda_back))
        return self.full_replay_log_likelihood(u, beta, omega, lambda_rep, lambda_back)


# ---------------------------------------------------------------------------- target
class Stage6CTarget:
    """The direct joint log target, callable entirely outside the transition code."""

    def __init__(self, evaluator: LatentPosetEvaluator, active, fixed: dict):
        active = tuple(a for a in SWEEP_ORDER if a in set(active))
        if "U" not in active or "rho" not in active:
            raise ValueError("Stage 6C always infers U and rho")
        for name in ("beta", "omega", "lambda_rep", "lambda_back"):
            if name not in active and name not in fixed:
                raise ValueError(f"inactive scalar {name} needs a fixed value")
        self.evaluator = evaluator
        self.active = active
        self.fixed = {k: float(v) for k, v in fixed.items() if k not in active}

    def scalars(self, values: dict) -> dict:
        out = dict(self.fixed)
        for name in ("beta", "omega", "lambda_rep", "lambda_back"):
            if name in self.active and name in values:
                out[name] = float(values[name])
        return out

    def log_likelihood(self, u, values: dict, allow_cache: bool = True) -> float:
        s = self.scalars(values)
        return self.evaluator.log_likelihood(
            u, s["beta"], s["omega"], s["lambda_rep"], s["lambda_back"],
            allow_cache=allow_cache)

    def decompose(self, u, values: dict, allow_cache: bool = True) -> dict:
        """Every component of the target, separately — the basis of the decomposition test.

        `log target = log likelihood + log p(U | rho) + log p(rho) [+ log p(beta)]`
        """
        rho = float(values["rho"])
        structural = log_structural_prior(u, rho)
        rho_prior = log_rho_prior(rho)
        scalar_priors = {}
        if "beta" in self.active:
            scalar_priors["beta"] = log_prior("beta", float(values["beta"]))
        finite = (math.isfinite(structural) and math.isfinite(rho_prior)
                  and all(math.isfinite(v) for v in scalar_priors.values()))
        if not finite:
            return {"log_likelihood": -math.inf, "log_structural_prior": structural,
                    "log_rho_prior": rho_prior, "log_scalar_priors": scalar_priors,
                    "log_target": -math.inf}
        ll = self.log_likelihood(u, values, allow_cache=allow_cache)
        total = ll + structural + rho_prior + float(sum(scalar_priors.values()))
        return {"log_likelihood": ll, "log_structural_prior": structural,
                "log_rho_prior": rho_prior, "log_scalar_priors": scalar_priors,
                "log_target": total}

    def log_target(self, u, values: dict, allow_cache: bool = True) -> float:
        return self.decompose(u, values, allow_cache=allow_cache)["log_target"]


# ----------------------------------------------------------------------------- state
@dataclass
class Stage6CState:
    u: np.ndarray
    values: dict                       # rho, and beta when active
    log_likelihood: float
    log_structural_prior: float
    log_rho_prior: float
    log_scalar_priors: dict
    log_target: float
    iteration: int = 0
    proposed: dict = field(default_factory=dict)
    accepted: dict = field(default_factory=dict)
    rng_state: dict | None = None
    chain: int = 0

    def to_dict(self) -> dict:
        return {"u": np.asarray(self.u, dtype=float).tolist(),
                "values": {k: float(v) for k, v in self.values.items()},
                "log_likelihood": float(self.log_likelihood),
                "log_structural_prior": float(self.log_structural_prior),
                "log_rho_prior": float(self.log_rho_prior),
                "log_scalar_priors": {k: float(v)
                                      for k, v in self.log_scalar_priors.items()},
                "log_target": float(self.log_target),
                "iteration": int(self.iteration),
                "proposed": {k: int(v) for k, v in self.proposed.items()},
                "accepted": {k: int(v) for k, v in self.accepted.items()},
                "chain": int(self.chain),
                "rng_state": _jsonable_rng_state(self.rng_state)}

    @classmethod
    def from_dict(cls, payload: dict) -> "Stage6CState":
        return cls(u=np.array(payload["u"], dtype=float),
                   values=dict(payload["values"]),
                   log_likelihood=float(payload["log_likelihood"]),
                   log_structural_prior=float(payload["log_structural_prior"]),
                   log_rho_prior=float(payload["log_rho_prior"]),
                   log_scalar_priors=dict(payload["log_scalar_priors"]),
                   log_target=float(payload["log_target"]),
                   iteration=int(payload["iteration"]),
                   proposed=dict(payload["proposed"]),
                   accepted=dict(payload["accepted"]),
                   rng_state=_rng_state_from_jsonable(payload.get("rng_state")),
                   chain=int(payload.get("chain", 0)))


def initial_state(target: Stage6CTarget, u, values: dict, rng, chain: int = 0
                  ) -> Stage6CState:
    target.evaluator.invalidate()
    parts = target.decompose(u, values, allow_cache=False)
    if not math.isfinite(parts["log_target"]):
        raise ValueError(f"chain {chain}: start has non-finite log target ({values})")
    names = ["U", "rho"] + (["beta"] if "beta" in target.active else [])
    return Stage6CState(u=np.array(u, dtype=float, copy=True), values=dict(values),
                        log_likelihood=parts["log_likelihood"],
                        log_structural_prior=parts["log_structural_prior"],
                        log_rho_prior=parts["log_rho_prior"],
                        log_scalar_priors=parts["log_scalar_priors"],
                        log_target=parts["log_target"], iteration=0,
                        proposed={n: 0 for n in names}, accepted={n: 0 for n in names},
                        rng_state=rng.bit_generator.state, chain=chain)


# ----------------------------------------------------------------------------- sweep
def sweep_once(state: Stage6CState, target: Stage6CTarget, sigma_u: float,
               rho_scale: float, beta_scale: float, rng) -> Stage6CState:
    """One Metropolis-within-Gibbs sweep: every U row, then rho, then beta if active."""
    u = np.array(state.u, dtype=float, copy=True)
    values = dict(state.values)
    proposed = dict(state.proposed)
    accepted = dict(state.accepted)

    scalars = target.scalars(values)
    current_ll = state.log_likelihood
    current_structural = state.log_structural_prior

    # ---- U, one row at a time. Symmetric proposal: no Hastings term. -----------------
    for row in range(u.shape[0]):
        candidate = propose_row(u, row, sigma_u, rng)
        candidate_structural = log_structural_prior(candidate, values["rho"])
        proposed["U"] = proposed.get("U", 0) + 1
        if not math.isfinite(candidate_structural):
            continue
        # a proposed U may induce a different order, so this is a complete replay
        candidate_ll = target.evaluator.full_replay_log_likelihood(
            candidate, scalars["beta"], scalars["omega"], scalars["lambda_rep"],
            scalars["lambda_back"])
        log_alpha = ((candidate_ll - current_ll)
                     + (candidate_structural - current_structural))
        if log_alpha >= 0.0 or math.log(rng.random()) < log_alpha:
            u = candidate
            current_ll = candidate_ll
            current_structural = candidate_structural
            accepted["U"] = accepted.get("U", 0) + 1
            target.evaluator.invalidate()

    # ---- rho. The likelihood is NOT touched: rho acts only through p(U | rho). -------
    # Two corrections ride along: the logit Jacobian, and the Gaussian determinant
    # inside log p(U | rho), which does not cancel when U is held and rho moves.
    current_rho = float(values["rho"])
    z = rho_to_unconstrained(current_rho)
    candidate_rho = rho_from_unconstrained(z + rho_scale * rng.normal())
    proposed["rho"] = proposed.get("rho", 0) + 1
    candidate_structural_rho = log_structural_prior(u, candidate_rho)
    candidate_rho_prior = log_rho_prior(candidate_rho)
    if math.isfinite(candidate_structural_rho) and math.isfinite(candidate_rho_prior):
        log_alpha = ((candidate_structural_rho - current_structural)
                     + (candidate_rho_prior - log_rho_prior(current_rho))
                     + (log_jacobian_rho(candidate_rho) - log_jacobian_rho(current_rho)))
        if log_alpha >= 0.0 or math.log(rng.random()) < log_alpha:
            values["rho"] = candidate_rho
            current_structural = candidate_structural_rho
            accepted["rho"] = accepted.get("rho", 0) + 1

    # ---- beta, reusing the frozen Stage 6B kernel and proposal ------------------------
    if "beta" in target.active:
        target.evaluator.ensure_cache(u, scalars["omega"])

        def beta_log_posterior(candidate):
            trial = dict(values); trial["beta"] = candidate
            s = target.scalars(trial)
            prior = log_prior("beta", candidate)
            if not math.isfinite(prior):
                return -math.inf
            return prior + target.evaluator.log_likelihood(
                u, s["beta"], s["omega"], s["lambda_rep"], s["lambda_back"],
                allow_cache=True)

        current_beta_posterior = log_prior("beta", values["beta"]) + current_ll
        new_beta, new_posterior, was_accepted = scalar_mh_step(
            values["beta"], current_beta_posterior, beta_log_posterior,
            build_proposal("beta", beta_scale), rng)
        proposed["beta"] = proposed.get("beta", 0) + 1
        if was_accepted:
            values["beta"] = new_beta
            current_ll = new_posterior - log_prior("beta", new_beta)
            accepted["beta"] = accepted.get("beta", 0) + 1

    scalar_priors = ({"beta": log_prior("beta", values["beta"])}
                     if "beta" in target.active else {})
    rho_prior = log_rho_prior(values["rho"])
    return Stage6CState(
        u=u, values=values, log_likelihood=current_ll,
        log_structural_prior=current_structural, log_rho_prior=rho_prior,
        log_scalar_priors=scalar_priors,
        log_target=current_ll + current_structural + rho_prior
        + float(sum(scalar_priors.values())),
        iteration=state.iteration + 1, proposed=proposed, accepted=accepted,
        rng_state=rng.bit_generator.state, chain=state.chain)


@dataclass
class Stage6CResult:
    u_draws: np.ndarray                 # (n_kept, m, d)
    poset_ids: np.ndarray               # (n_kept,) index into the catalogue
    rho: np.ndarray
    beta: np.ndarray | None
    log_likelihood: np.ndarray
    log_target: np.ndarray
    relation_counts: np.ndarray
    proposed: dict
    accepted: dict
    proposed_after_burn_in: dict
    accepted_after_burn_in: dict
    final_state: Stage6CState
    runtime_seconds: float
    chain: int
    seed: int

    def acceptance(self, post_burn_in: bool = True) -> dict:
        p = self.proposed_after_burn_in if post_burn_in else self.proposed
        a = self.accepted_after_burn_in if post_burn_in else self.accepted
        return {k: (a.get(k, 0) / p[k] if p.get(k) else float("nan")) for k in p}


def run_stage6c_mcmc(target: Stage6CTarget, u_start, values_start: dict,
                     num_sweeps: int, burn_in: int, thin: int, seed: int,
                     sigma_u: float = SIGMA_U, rho_scale: float = 0.5,
                     beta_scale: float = 0.05109, chain: int = 0,
                     catalogue=None, state: Stage6CState | None = None,
                     rng=None) -> Stage6CResult:
    """Run one chain. `state`/`rng` allow a deterministic, bit-identical resume."""
    if burn_in >= num_sweeps:
        raise ValueError("burn_in must be smaller than num_sweeps")
    if thin < 1:
        raise ValueError("thin must be at least 1")

    rng = np.random.default_rng(seed) if rng is None else rng
    if state is None:
        state = initial_state(target, u_start, values_start, rng, chain)

    start_iteration = state.iteration
    capacity = len(range(max(burn_in, start_iteration), num_sweeps, thin)) + 1
    m, d = np.asarray(u_start if state is None else state.u).shape
    u_draws = np.empty((capacity, m, d))
    poset_ids = np.full(capacity, -1, dtype=int)
    relation_counts = np.empty(capacity, dtype=int)
    rho = np.empty(capacity)
    beta = np.empty(capacity) if "beta" in target.active else None
    lls = np.empty(capacity)
    lts = np.empty(capacity)

    post_proposed = {n: 0 for n in state.proposed}
    post_accepted = {n: 0 for n in state.accepted}

    k = 0
    began = time.perf_counter()
    for i in range(start_iteration, num_sweeps):
        before_p, before_a = dict(state.proposed), dict(state.accepted)
        state = sweep_once(state, target, sigma_u, rho_scale, beta_scale, rng)
        if i >= burn_in:
            for name in post_proposed:
                post_proposed[name] += state.proposed[name] - before_p.get(name, 0)
                post_accepted[name] += state.accepted[name] - before_a.get(name, 0)
            if (i - burn_in) % thin == 0:
                u_draws[k] = state.u
                precedence = precedence_from_u(state.u)
                relation_counts[k] = int(precedence.sum())
                if catalogue is not None:
                    poset_ids[k] = catalogue.index_of(precedence)
                rho[k] = state.values["rho"]
                if beta is not None:
                    beta[k] = state.values["beta"]
                lls[k] = state.log_likelihood
                lts[k] = state.log_target
                k += 1
    runtime = time.perf_counter() - began

    return Stage6CResult(
        u_draws=u_draws[:k], poset_ids=poset_ids[:k], rho=rho[:k],
        beta=None if beta is None else beta[:k], log_likelihood=lls[:k],
        log_target=lts[:k], relation_counts=relation_counts[:k],
        proposed=dict(state.proposed), accepted=dict(state.accepted),
        proposed_after_burn_in=post_proposed, accepted_after_burn_in=post_accepted,
        final_state=state, runtime_seconds=runtime, chain=chain, seed=seed)
