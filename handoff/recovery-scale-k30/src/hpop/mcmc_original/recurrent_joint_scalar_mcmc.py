"""Stage 6B2 / 6B3 — one Metropolis-within-Gibbs orchestrator over an active scalar set.

This module composes the Stage 6B1 kernels; it does not reimplement them. Each coordinate
update calls the validated `scalar_mh_step` with the validated proposal built by
`build_proposal`, so the log-scale Hastings correction and the acceptance rule are the
same objects Stage 6B1 tested, not copies of their algebra. The only new thing here is the
*orchestration*: which coordinate moves, in what order, and what the other coordinates are
holding while it does.

A complete sweep visits `beta -> omega -> lambda_rep -> lambda_back`, and every coordinate
sees the most recently accepted values of those before it.

## The omega rule

`kappa = sigmoid(omega)` is the validity recursion, so an omega proposal changes the whole
`q` trajectory. Every omega likelihood evaluation therefore replays from `q_0 = 0` for
every block, and the state cache is *never* consulted for omega — not as an optimisation
choice but because the cached trajectory belongs to a different omega.

The cache that does exist is keyed on the omega it was built at and is only ever written
by an explicit `refresh_cache` call. An evaluation never writes it. That gives the
rejection semantics for free: a rejected proposal cannot leave anything behind, because a
proposal never stored anything. A changed omega cannot silently read a stale cache either,
since the key check falls through to full replay.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_rfs import sigmoid
from hpop.mcmc_original.recurrent_scalar_mcmc import (
    build_proposal, scalar_mh_step,
)
from hpop.mcmc_original.recurrent_scalar_posterior import (
    batch_recurrent_log_likelihood_full_replay, cached_batch_log_likelihood, log_prior,
)
from hpop.mcmc_original.stage6b_frozen import PARAMETER_ORDER, SWEEP_ORDER

__all__ = [
    "vectorized_state_features", "RecurrentJointEvaluator", "JointScalarTarget",
    "JointChainState", "JointRunResult", "run_joint_scalar_mcmc", "sweep_once",
    "ACTIVE_B2", "ACTIVE_B3",
]

ACTIVE_B2 = ("beta", "omega", "lambda_rep")
ACTIVE_B3 = ("beta", "omega", "lambda_rep", "lambda_back")


# --------------------------------------------------------------------------- state cache
def vectorized_state_features(role_array, u, omega: float) -> dict:
    """The Stage 6B1 feature bundle for a FIXED omega, vectorised across blocks.

    Identical in content to `recurrent_scalar_posterior.precompute_state_features`, which
    it is tested against; the Python loop over blocks is replaced by an array axis. Valid
    only at ``omega`` — the returned ``q`` trajectory is that omega's trajectory.
    """
    u = np.asarray(u, dtype=float)
    precedence = precedence_from_u(u)
    roles = np.asarray(role_array, dtype=int)
    m = u.shape[0]
    n, T = roles.shape
    kappa = sigmoid(omega)

    pred_mask = precedence.T.astype(bool)
    succ = precedence.astype(float)
    succ_off = succ.copy()
    np.fill_diagonal(succ_off, 0.0)

    F = np.empty((n, T, m))
    Q = np.empty((n, T, m))
    QV = np.empty((n, T, m))
    CB = np.empty((n, T, m))

    q = np.zeros((n, m))
    rows = np.arange(n)
    for t in range(T):
        F[:, t, :] = np.prod(np.where(pred_mask[None, :, :], q[:, None, :], 1.0), axis=2)
        Q[:, t, :] = np.log1p((1.0 - q) @ succ.T)
        QV[:, t, :] = q
        CB[:, t, :] = kappa * (q @ succ_off.T)
        observed = roles[:, t]
        gate = np.where(precedence[observed], kappa, 0.0)
        q = q * (1.0 - gate)
        q[rows, observed] = 1.0
    return {"F": F, "Q": Q, "q": QV, "C_back": CB, "obs": roles.copy(), "m": m,
            "omega": float(omega)}


class RecurrentJointEvaluator:
    """Complete recurrent log likelihood, with an explicitly keyed omega cache.

    Data- and U-only quantities (the precedence closure and its masks) are computed once;
    they do not depend on any inferred scalar. The omega-dependent state trajectory is
    cached separately and only ever written by `refresh_cache`.
    """

    def __init__(self, role_array, u, epsilon: float):
        self.role_array = np.asarray(role_array, dtype=int)
        self.u = np.asarray(u, dtype=float)
        self.epsilon = float(epsilon)
        self.n_blocks, self.T = self.role_array.shape
        self.m = int(self.u.shape[0])

        self._features = None
        self._features_omega = None
        self.full_replay_calls = 0
        self.cached_calls = 0
        self.cache_builds = 0

    # -- cache lifecycle ---------------------------------------------------------------
    @property
    def cache_omega(self):
        return self._features_omega

    def cache_is_valid_for(self, omega: float) -> bool:
        return self._features_omega is not None and float(omega) == self._features_omega

    def invalidate(self) -> None:
        """Drop the omega-dependent cache. Called after an accepted omega."""
        self._features = None
        self._features_omega = None

    def refresh_cache(self, omega: float) -> None:
        self._features = vectorized_state_features(self.role_array, self.u, float(omega))
        self._features_omega = float(omega)
        self.cache_builds += 1

    def ensure_cache(self, omega: float) -> None:
        if not self.cache_is_valid_for(omega):
            self.refresh_cache(omega)

    # -- evaluation --------------------------------------------------------------------
    def full_replay_log_likelihood(self, beta, omega, lambda_rep, lambda_back) -> float:
        """Always replays ``q`` from zero for every block. No state is reused or stored."""
        self.full_replay_calls += 1
        return float(batch_recurrent_log_likelihood_full_replay(
            self.role_array, self.u, beta, self.epsilon, omega, lambda_rep, lambda_back))

    def log_likelihood(self, beta, omega, lambda_rep, lambda_back,
                       allow_cache: bool = True) -> float:
        """Complete log likelihood.

        The cache is used only when it was built at exactly this omega. Any other omega —
        including a proposed one — falls through to full replay, so a stale cache cannot
        be read silently. An evaluation never writes the cache.
        """
        if allow_cache and self.cache_is_valid_for(omega):
            self.cached_calls += 1
            return float(cached_batch_log_likelihood(
                self._features, beta, self.epsilon, lambda_rep, lambda_back))
        return self.full_replay_log_likelihood(beta, omega, lambda_rep, lambda_back)


# ------------------------------------------------------------------------- joint target
class JointScalarTarget:
    """Joint log posterior over an active scalar set. Callable outside the MCMC code.

    ``log posterior = complete recurrent log likelihood + sum of ACTIVE scalar log
    priors``. Inactive coordinates are held at their registered fixed values and
    contribute no prior term, so the Stage 6B2 target is exactly the one in section 5 and
    not the Stage 6B3 target with a coordinate pinned.
    """

    def __init__(self, evaluator: RecurrentJointEvaluator, active, fixed: dict):
        unknown = set(active) - set(PARAMETER_ORDER)
        if unknown:
            raise ValueError(f"unknown coordinates {sorted(unknown)}")
        missing = set(PARAMETER_ORDER) - set(active) - set(fixed)
        if missing:
            raise ValueError(f"inactive coordinates need fixed values: {sorted(missing)}")
        self.evaluator = evaluator
        self.active = tuple(p for p in PARAMETER_ORDER if p in set(active))
        self.fixed = {k: float(v) for k, v in fixed.items() if k not in self.active}

    def complete(self, values: dict) -> dict:
        """Fill inactive coordinates from the registered fixed values."""
        out = dict(self.fixed)
        out.update({k: float(values[k]) for k in self.active})
        return out

    def log_priors(self, values: dict) -> dict:
        full = self.complete(values)
        return {name: log_prior(name, full[name]) for name in self.active}

    def log_prior_total(self, values: dict) -> float:
        return float(sum(self.log_priors(values).values()))

    def log_likelihood(self, values: dict, allow_cache: bool = True) -> float:
        f = self.complete(values)
        return self.evaluator.log_likelihood(
            f["beta"], f["omega"], f["lambda_rep"], f["lambda_back"],
            allow_cache=allow_cache)

    def log_posterior(self, values: dict, allow_cache: bool = True) -> float:
        prior = self.log_prior_total(values)
        if not math.isfinite(prior):
            return -math.inf
        return prior + self.log_likelihood(values, allow_cache=allow_cache)

    def decompose(self, values: dict, allow_cache: bool = True) -> dict:
        """Everything the chain state records, in one evaluation."""
        priors = self.log_priors(values)
        total_prior = float(sum(priors.values()))
        if not math.isfinite(total_prior):
            return {"log_prior_components": priors, "log_prior": -math.inf,
                    "log_likelihood": -math.inf, "log_posterior": -math.inf}
        ll = self.log_likelihood(values, allow_cache=allow_cache)
        return {"log_prior_components": priors, "log_prior": total_prior,
                "log_likelihood": ll, "log_posterior": total_prior + ll}


# --------------------------------------------------------------------------- chain state
@dataclass
class JointChainState:
    """Serialisable chain state, sufficient for a deterministic restart."""

    values: dict
    log_likelihood: float
    log_prior_components: dict
    log_posterior: float
    iteration: int = 0
    proposed: dict = field(default_factory=dict)
    accepted: dict = field(default_factory=dict)
    rng_state: dict | None = None
    chain: int = 0

    def to_dict(self) -> dict:
        return {"values": {k: float(v) for k, v in self.values.items()},
                "log_likelihood": float(self.log_likelihood),
                "log_prior_components": {k: float(v)
                                         for k, v in self.log_prior_components.items()},
                "log_posterior": float(self.log_posterior),
                "iteration": int(self.iteration),
                "proposed": {k: int(v) for k, v in self.proposed.items()},
                "accepted": {k: int(v) for k, v in self.accepted.items()},
                "chain": int(self.chain),
                "rng_state": _jsonable_rng_state(self.rng_state)}

    @classmethod
    def from_dict(cls, payload: dict) -> "JointChainState":
        return cls(values=dict(payload["values"]),
                   log_likelihood=float(payload["log_likelihood"]),
                   log_prior_components=dict(payload["log_prior_components"]),
                   log_posterior=float(payload["log_posterior"]),
                   iteration=int(payload["iteration"]),
                   proposed=dict(payload["proposed"]),
                   accepted=dict(payload["accepted"]),
                   rng_state=_rng_state_from_jsonable(payload.get("rng_state")),
                   chain=int(payload.get("chain", 0)))


def _jsonable_rng_state(state):
    if state is None:
        return None
    out = {"bit_generator": state["bit_generator"],
           "state": {k: (int(v) if not isinstance(v, np.ndarray) else v.tolist())
                     for k, v in state["state"].items()}}
    for key in ("has_uint32", "uinteger"):
        if key in state:
            out[key] = int(state[key])
    return out


def _rng_state_from_jsonable(payload):
    if payload is None:
        return None
    state = {"bit_generator": payload["bit_generator"], "state": {}}
    for k, v in payload["state"].items():
        state["state"][k] = np.array(v, dtype=np.uint32) if isinstance(v, list) else int(v)
    for key in ("has_uint32", "uinteger"):
        if key in payload:
            state[key] = int(payload[key])
    return state


# ------------------------------------------------------------------------- the sweep
def sweep_once(state: JointChainState, target: JointScalarTarget, proposals: dict,
               rng: np.random.Generator) -> JointChainState:
    """One complete Metropolis-within-Gibbs sweep in the registered coordinate order.

    Each coordinate is updated by the Stage 6B1 `scalar_mh_step` against a closure that
    substitutes the trial value into the *current* state — so a coordinate later in the
    sweep sees every acceptance that happened earlier in the same sweep.
    """
    values = dict(state.values)
    current_lp = state.log_posterior
    proposed = dict(state.proposed)
    accepted = dict(state.accepted)

    for name in SWEEP_ORDER:
        if name not in target.active:
            continue

        # omega must never read a cached q trajectory; the other three may, but only
        # when the cache was built at the omega currently in the state.
        if name == "omega":
            def coordinate_log_posterior(candidate, _n=name):
                trial = dict(values); trial[_n] = candidate
                return target.log_posterior(trial, allow_cache=False)
        else:
            target.evaluator.ensure_cache(target.complete(values)["omega"])

            def coordinate_log_posterior(candidate, _n=name):
                trial = dict(values); trial[_n] = candidate
                return target.log_posterior(trial, allow_cache=True)

        new_value, new_lp, was_accepted = scalar_mh_step(
            values[name], current_lp, coordinate_log_posterior, proposals[name], rng)

        proposed[name] = proposed.get(name, 0) + 1
        if was_accepted:
            accepted[name] = accepted.get(name, 0) + 1
            values[name] = new_value
            current_lp = new_lp
            if name == "omega":
                target.evaluator.invalidate()      # the cached trajectory is now stale

    parts = target.log_priors(values)
    return JointChainState(values=values,
                           log_likelihood=current_lp - float(sum(parts.values())),
                           log_prior_components=parts, log_posterior=current_lp,
                           iteration=state.iteration + 1, proposed=proposed,
                           accepted=accepted, rng_state=rng.bit_generator.state,
                           chain=state.chain)


@dataclass
class JointRunResult:
    draws: dict                  # name -> (n_kept,) array
    log_likelihood: np.ndarray
    log_posterior: np.ndarray
    proposed: dict
    accepted: dict
    proposed_after_burn_in: dict
    accepted_after_burn_in: dict
    final_state: JointChainState
    runtime_seconds: float
    chain: int
    seed: int

    def acceptance(self, post_burn_in: bool = True) -> dict:
        p = self.proposed_after_burn_in if post_burn_in else self.proposed
        a = self.accepted_after_burn_in if post_burn_in else self.accepted
        return {k: (a.get(k, 0) / p[k] if p.get(k) else float("nan")) for k in p}


def run_joint_scalar_mcmc(target: JointScalarTarget, start: dict, scales: dict,
                          num_iterations: int, burn_in: int, thin: int, seed: int,
                          chain: int = 0, initial_state: JointChainState | None = None,
                          rng: np.random.Generator | None = None) -> JointRunResult:
    """Run one chain of the joint sampler.

    ``rng`` / ``initial_state`` support deterministic resume: hand back the final state
    and a generator restored from its recorded bit-generator state and the continuation is
    bit-identical to an uninterrupted run.
    """
    if burn_in >= num_iterations:
        raise ValueError("burn_in must be smaller than num_iterations")
    if thin < 1:
        raise ValueError("thin must be at least 1")

    rng = np.random.default_rng(seed) if rng is None else rng
    proposals = {name: build_proposal(name, scales[name]) for name in target.active}

    if initial_state is None:
        values = {name: float(start[name]) for name in target.active}
        target.evaluator.invalidate()
        parts = target.decompose(values, allow_cache=False)
        if not math.isfinite(parts["log_posterior"]):
            raise ValueError(f"chain {chain}: start {values} has non-finite log posterior")
        state = JointChainState(values=values, log_likelihood=parts["log_likelihood"],
                                log_prior_components=parts["log_prior_components"],
                                log_posterior=parts["log_posterior"], iteration=0,
                                proposed={n: 0 for n in target.active},
                                accepted={n: 0 for n in target.active},
                                rng_state=rng.bit_generator.state, chain=chain)
    else:
        state = initial_state

    start_iteration = state.iteration
    capacity = len(range(max(burn_in, start_iteration), num_iterations, thin)) + 1
    draws = {name: np.empty(capacity) for name in target.active}
    lls = np.empty(capacity)
    lps = np.empty(capacity)

    # post-burn-in counts are accumulated from per-sweep deltas rather than differenced
    # against a snapshot, so a resumed run accounts correctly for the sweeps it ran.
    post_proposed = {n: 0 for n in target.active}
    post_accepted = {n: 0 for n in target.active}

    k = 0
    began = time.perf_counter()
    for i in range(start_iteration, num_iterations):
        before_p = dict(state.proposed)
        before_a = dict(state.accepted)
        state = sweep_once(state, target, proposals, rng)
        if i >= burn_in:
            for name in target.active:
                post_proposed[name] += state.proposed[name] - before_p.get(name, 0)
                post_accepted[name] += state.accepted[name] - before_a.get(name, 0)
            if (i - burn_in) % thin == 0:
                for name in target.active:
                    draws[name][k] = state.values[name]
                lls[k] = state.log_likelihood
                lps[k] = state.log_posterior
                k += 1
    runtime = time.perf_counter() - began

    return JointRunResult(
        draws={n: v[:k] for n, v in draws.items()}, log_likelihood=lls[:k],
        log_posterior=lps[:k], proposed=dict(state.proposed),
        accepted=dict(state.accepted), proposed_after_burn_in=post_proposed,
        accepted_after_burn_in=post_accepted, final_state=state,
        runtime_seconds=runtime, chain=chain, seed=seed)
