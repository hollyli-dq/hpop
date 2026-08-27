"""Stage 6D — Metropolis-within-Gibbs over `U`, `rho` and all four recurrent scalars.

This module is a **composition layer**. It contains no new acceptance mathematics: every
update calls a kernel that a previous stage validated, as an object rather than as
re-derived algebra.

    U            sampler_u.propose_row                 (Stage 2A, via Stage 6C)
    rho          logit random walk + log(rho(1-rho))   (Stage 6C)
    beta         scalar_mh_step + build_proposal       (Stage 6B)
    omega        scalar_mh_step + build_proposal       (Stage 6B)
    lambda_rep   scalar_mh_step + build_proposal       (Stage 6B)
    lambda_back  scalar_mh_step + build_proposal       (Stage 6B)

Sweep order is `U -> rho -> beta -> omega -> lambda_rep -> lambda_back`, and every update
sees the values accepted before it in the same sweep.

## The two updates that must replay, and the one that must not

`U` and `omega` are the dangerous pair. A `U` move can change the induced order `H = h(U)`
and therefore the frontier, feasibility, the whole `q` trajectory and every downstream
one-step probability. An `omega` move changes `kappa = sigmoid(omega)`, which enters the
`q` recursion itself. Both are therefore scored by a **complete replay from `q_0 = 0` for
every block**, never by editing a cached trajectory.

The evaluator's cache is keyed on the exact `(induced order, omega)` it was built at, and
is written only by an explicit refresh — so a proposal at a different `U` or a different
`omega` cannot read it, and an evaluation cannot leave anything behind. `beta`,
`lambda_rep` and `lambda_back` do not enter the `q` recursion, so they may legitimately be
scored from a cache built at the current `(H, omega)`.

`rho` is the opposite case: it acts only through `p(U | rho)` and touches no likelihood
term at all. A `rho` update therefore consumes **zero** likelihood evaluations, which is
asserted rather than assumed.

## Invalid rho proposals

The registered `rho` support is the open interval `(0, 1 - 5e-3)`. A logit random walk
always lands in `(0, 1)`, but it can land above the truncation point, where the prior is
`-inf`. Such a proposal is rejected immediately and counted as **invalid**, separately
from an ordinary rejection — a state-dependent redraw loop would need its own truncated
normalisation and is deliberately not used.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_joint_scalar_mcmc import (
    _jsonable_rng_state, _rng_state_from_jsonable,
)
from hpop.mcmc_original.recurrent_latent_poset_mcmc import LatentPosetEvaluator, poset_key
from hpop.mcmc_original.recurrent_scalar_mcmc import build_proposal, scalar_mh_step
from hpop.mcmc_original.recurrent_scalar_posterior import log_prior
from hpop.mcmc_original.sampler_u import propose_row
from hpop.mcmc_original.stage6d_frozen import (
    ACTIVE_6D, REGISTERED_SCALES, SCALAR_ORDER, SWEEP_ORDER_6D, log_jacobian_rho,
    log_rho_prior, log_structural_prior, rho_from_unconstrained, rho_is_in_support,
    rho_to_unconstrained,
)

__all__ = [
    "Stage6DTarget", "Stage6DState", "Stage6DResult", "initial_state", "sweep_once",
    "run_oracle_joint_mcmc", "log_target_6d", "ACTIVE_6D", "SWEEP_ORDER_6D",
]

# omega enters the q recursion; the other three scalars do not, so a cache built at the
# current (H, omega) is exact for them.
CACHE_SAFE_SCALARS = ("beta", "lambda_rep", "lambda_back")


# ---------------------------------------------------------------------------- target
class Stage6DTarget:
    """The direct joint log target, callable entirely outside the transition code.

    Nothing here calls an acceptance-ratio helper, so comparing an implemented ratio
    against a difference of these values compares two independent routes.
    """

    def __init__(self, evaluator: LatentPosetEvaluator, active=ACTIVE_6D, fixed=None):
        active = tuple(a for a in SWEEP_ORDER_6D if a in set(active))
        if "U" not in active or "rho" not in active:
            raise ValueError("Stage 6D always infers U and rho")
        fixed = dict(fixed or {})
        for name in SCALAR_ORDER:
            if name not in active and name not in fixed:
                raise ValueError(f"inactive scalar {name} needs a fixed value")
        self.evaluator = evaluator
        self.active = active
        self.fixed = {k: float(v) for k, v in fixed.items() if k not in active}
        self.active_scalars = tuple(s for s in SCALAR_ORDER if s in active)

    def scalars(self, values: dict) -> dict:
        out = dict(self.fixed)
        for name in SCALAR_ORDER:
            if name in self.active and name in values:
                out[name] = float(values[name])
        return out

    def log_likelihood(self, u, values: dict, allow_cache: bool = True) -> float:
        s = self.scalars(values)
        return self.evaluator.log_likelihood(
            u, s["beta"], s["omega"], s["lambda_rep"], s["lambda_back"],
            allow_cache=allow_cache)

    def decompose(self, u, values: dict, allow_cache: bool = True) -> dict:
        """Every component of the target, separately — the basis of §5's decomposition.

        `log target = log likelihood + log p(U|rho) + log p(rho) + sum of scalar priors`
        """
        rho = float(values["rho"])
        structural = log_structural_prior(u, rho)
        rho_prior = log_rho_prior(rho)
        scalar_priors = {name: log_prior(name, float(values[name]))
                         for name in self.active_scalars}
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


def log_target_6d(target: Stage6DTarget, u, values: dict) -> dict:
    """§5's named entry point: the full decomposition, never cached, never via a kernel."""
    return target.decompose(u, values, allow_cache=False)


# ----------------------------------------------------------------------------- state
@dataclass
class Stage6DState:
    u: np.ndarray
    values: dict
    log_likelihood: float
    log_structural_prior: float
    log_rho_prior: float
    log_scalar_priors: dict
    log_target: float
    iteration: int = 0
    proposed: dict = field(default_factory=dict)
    accepted: dict = field(default_factory=dict)
    invalid: dict = field(default_factory=dict)
    rng_state: dict | None = None
    chain: int = 0

    @property
    def relation_count(self) -> int:
        return int(precedence_from_u(self.u).sum())

    def to_dict(self) -> dict:
        return {"u": np.asarray(self.u, dtype=float).tolist(),
                "values": {k: float(v) for k, v in self.values.items()},
                "log_likelihood": float(self.log_likelihood),
                "log_structural_prior": float(self.log_structural_prior),
                "log_rho_prior": float(self.log_rho_prior),
                "log_scalar_priors": {k: float(v)
                                      for k, v in self.log_scalar_priors.items()},
                "log_target": float(self.log_target),
                "relation_count": self.relation_count,
                "poset_key_hex": poset_key(self.u).hex(),
                "iteration": int(self.iteration),
                "proposed": {k: int(v) for k, v in self.proposed.items()},
                "accepted": {k: int(v) for k, v in self.accepted.items()},
                "invalid": {k: int(v) for k, v in self.invalid.items()},
                "chain": int(self.chain),
                "rng_state": _jsonable_rng_state(self.rng_state)}

    @classmethod
    def from_dict(cls, payload: dict) -> "Stage6DState":
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
                   invalid=dict(payload.get("invalid", {})),
                   rng_state=_rng_state_from_jsonable(payload.get("rng_state")),
                   chain=int(payload.get("chain", 0)))


def initial_state(target: Stage6DTarget, u, values: dict, rng, chain: int = 0
                  ) -> Stage6DState:
    target.evaluator.invalidate()
    parts = target.decompose(u, values, allow_cache=False)
    if not math.isfinite(parts["log_target"]):
        raise ValueError(f"chain {chain}: start has non-finite log target ({values})")
    names = ["U", "rho"] + list(target.active_scalars)
    return Stage6DState(
        u=np.array(u, dtype=float, copy=True), values=dict(values),
        log_likelihood=parts["log_likelihood"],
        log_structural_prior=parts["log_structural_prior"],
        log_rho_prior=parts["log_rho_prior"],
        log_scalar_priors=parts["log_scalar_priors"],
        log_target=parts["log_target"], iteration=0,
        proposed={n: 0 for n in names}, accepted={n: 0 for n in names},
        invalid={"rho": 0}, rng_state=rng.bit_generator.state, chain=chain)


# ----------------------------------------------------------------------------- sweep
def sweep_once(state: Stage6DState, target: Stage6DTarget, scales: dict,
               rng) -> Stage6DState:
    """One Metropolis-within-Gibbs sweep: every U row, then rho, then the four scalars."""
    u = np.array(state.u, dtype=float, copy=True)
    values = dict(state.values)
    proposed = dict(state.proposed)
    accepted = dict(state.accepted)
    invalid = dict(state.invalid)

    current_ll = state.log_likelihood
    current_structural = state.log_structural_prior
    evaluator = target.evaluator

    # ---- U, one row at a time. Symmetric proposal, so no Hastings term. --------------
    # Every candidate is scored by a COMPLETE replay: a changed U can change H = h(U),
    # and with it the frontier and the whole q trajectory.
    for row in range(u.shape[0]):
        s = target.scalars(values)
        candidate = propose_row(u, row, scales["U"], rng)
        proposed["U"] = proposed.get("U", 0) + 1
        candidate_structural = log_structural_prior(candidate, values["rho"])
        if not math.isfinite(candidate_structural):
            continue
        candidate_ll = evaluator.full_replay_log_likelihood(
            candidate, s["beta"], s["omega"], s["lambda_rep"], s["lambda_back"])
        log_alpha = ((candidate_ll - current_ll)
                     + (candidate_structural - current_structural))
        if log_alpha >= 0.0 or math.log(rng.random()) < log_alpha:
            u = candidate
            current_ll = candidate_ll
            current_structural = candidate_structural
            accepted["U"] = accepted.get("U", 0) + 1
            evaluator.invalidate()          # H changed: any cache is now stale

    # ---- rho. Touches no likelihood term: it acts only through p(U | rho). -----------
    current_rho = float(values["rho"])
    z = rho_to_unconstrained(current_rho)
    candidate_rho = rho_from_unconstrained(z + scales["rho"] * rng.normal())
    proposed["rho"] = proposed.get("rho", 0) + 1
    if not rho_is_in_support(candidate_rho):
        # Outside the registered support: reject immediately, count separately. No
        # redraw loop, which would need its own state-dependent normalisation.
        invalid["rho"] = invalid.get("rho", 0) + 1
    else:
        candidate_structural_rho = log_structural_prior(u, candidate_rho)
        candidate_rho_prior = log_rho_prior(candidate_rho)
        if math.isfinite(candidate_structural_rho) and math.isfinite(candidate_rho_prior):
            log_alpha = ((candidate_structural_rho - current_structural)
                         + (candidate_rho_prior - log_rho_prior(current_rho))
                         + (log_jacobian_rho(candidate_rho)
                            - log_jacobian_rho(current_rho)))
            if log_alpha >= 0.0 or math.log(rng.random()) < log_alpha:
                values["rho"] = candidate_rho
                current_structural = candidate_structural_rho
                accepted["rho"] = accepted.get("rho", 0) + 1

    # ---- the four recurrent scalars, in the frozen Stage 6B order --------------------
    for name in target.active_scalars:
        s = target.scalars(values)
        # beta / lambda_rep / lambda_back do not enter the q recursion, so a cache built
        # at the current (H, omega) is exact. omega does, so it must never be cached.
        if name in CACHE_SAFE_SCALARS:
            evaluator.ensure_cache(u, s["omega"])
            allow_cache = True
        else:
            allow_cache = False

        def scalar_log_posterior(candidate, _name=name, _s=dict(s)):
            prior = log_prior(_name, candidate)
            if not math.isfinite(prior):
                return -math.inf
            trial = dict(_s)
            trial[_name] = candidate
            return prior + evaluator.log_likelihood(
                u, trial["beta"], trial["omega"], trial["lambda_rep"],
                trial["lambda_back"], allow_cache=allow_cache)

        current_posterior = log_prior(name, values[name]) + current_ll
        new_value, new_posterior, was_accepted = scalar_mh_step(
            values[name], current_posterior, scalar_log_posterior,
            build_proposal(name, scales[name]), rng)
        proposed[name] = proposed.get(name, 0) + 1
        if was_accepted:
            values[name] = new_value
            current_ll = new_posterior - log_prior(name, new_value)
            accepted[name] = accepted.get(name, 0) + 1
            if name == "omega":
                evaluator.invalidate()      # the q trajectory changed

    scalar_priors = {n: log_prior(n, values[n]) for n in target.active_scalars}
    rho_prior = log_rho_prior(values["rho"])
    return Stage6DState(
        u=u, values=values, log_likelihood=current_ll,
        log_structural_prior=current_structural, log_rho_prior=rho_prior,
        log_scalar_priors=scalar_priors,
        log_target=current_ll + current_structural + rho_prior
        + float(sum(scalar_priors.values())),
        iteration=state.iteration + 1, proposed=proposed, accepted=accepted,
        invalid=invalid, rng_state=rng.bit_generator.state, chain=state.chain)


# ------------------------------------------------------------------------------ run
@dataclass
class Stage6DResult:
    u_draws: np.ndarray
    poset_ids: np.ndarray
    relation_counts: np.ndarray
    scalars: dict
    log_likelihood: np.ndarray
    log_target: np.ndarray
    proposed: dict
    accepted: dict
    invalid: dict
    proposed_after_burn_in: dict
    accepted_after_burn_in: dict
    final_state: Stage6DState
    runtime_seconds: float
    chain: int
    seed: int

    def acceptance(self, post_burn_in: bool = True) -> dict:
        p = self.proposed_after_burn_in if post_burn_in else self.proposed
        a = self.accepted_after_burn_in if post_burn_in else self.accepted
        return {k: (a.get(k, 0) / p[k] if p.get(k) else float("nan")) for k in p}


def run_oracle_joint_mcmc(target: Stage6DTarget, u_start, values_start: dict,
                          num_sweeps: int, burn_in: int, thin: int, seed: int,
                          scales: dict | None = None, chain: int = 0, catalogue=None,
                          state: Stage6DState | None = None, rng=None) -> Stage6DResult:
    """Run one chain. `state`/`rng` allow a deterministic, bit-identical resume."""
    if burn_in >= num_sweeps:
        raise ValueError("burn_in must be smaller than num_sweeps")
    if thin < 1:
        raise ValueError("thin must be at least 1")
    scales = dict(REGISTERED_SCALES if scales is None else scales)

    rng = np.random.default_rng(seed) if rng is None else rng
    if state is None:
        state = initial_state(target, u_start, values_start, rng, chain)

    start_iteration = state.iteration
    capacity = len(range(max(burn_in, start_iteration), num_sweeps, thin)) + 1
    m, d = np.asarray(state.u).shape
    u_draws = np.empty((capacity, m, d), dtype=np.float32)   # compact: H needs no more
    poset_ids = np.full(capacity, -1, dtype=np.int32)
    relation_counts = np.empty(capacity, dtype=np.int16)
    names = list(target.active_scalars) + ["rho"]
    scalars = {n: np.empty(capacity) for n in names}
    lls = np.empty(capacity)
    lts = np.empty(capacity)

    post_proposed = {n: 0 for n in state.proposed}
    post_accepted = {n: 0 for n in state.accepted}

    k = 0
    began = time.perf_counter()
    for i in range(start_iteration, num_sweeps):
        before_p, before_a = dict(state.proposed), dict(state.accepted)
        state = sweep_once(state, target, scales, rng)
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
                for n in names:
                    scalars[n][k] = state.values[n]
                lls[k] = state.log_likelihood
                lts[k] = state.log_target
                k += 1
    runtime = time.perf_counter() - began

    return Stage6DResult(
        u_draws=u_draws[:k], poset_ids=poset_ids[:k], relation_counts=relation_counts[:k],
        scalars={n: v[:k] for n, v in scalars.items()},
        log_likelihood=lls[:k], log_target=lts[:k],
        proposed=dict(state.proposed), accepted=dict(state.accepted),
        invalid=dict(state.invalid),
        proposed_after_burn_in=post_proposed, accepted_after_burn_in=post_accepted,
        final_state=state, runtime_seconds=runtime, chain=chain, seed=seed)
