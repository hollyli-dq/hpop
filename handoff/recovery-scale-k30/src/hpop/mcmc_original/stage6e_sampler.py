"""Stage 6E — the production unknown-boundary joint sampler.

This module is a **composition layer**, like `recurrent_oracle_joint_mcmc` before it. It
adds no acceptance mathematics that a previous stage has not already validated:

    (S, z)       FastSegmentationKernel + the Stage 5 Hastings ratio   (Stage 6E0)
    (pi, P)      transitions.sample_transition_matrix                  (Stage 3)
    U            sampler_u.propose_row                                 (Stage 2A / 6C)
    rho          logit random walk + log(rho(1-rho))                   (Stage 6C)
    scalars      scalar_mh_step + build_proposal                       (Stage 6B)

The registered sweep order is

    (S, z) -> (pi, P) -> U -> rho -> beta -> omega -> lambda_rep -> lambda_back

and every update sees the values accepted before it in the same sweep.

## Two evaluators, one arithmetic

The segmentation phase and the parameter phase need the same number — the total recurrent
block log likelihood — but they need it under opposite access patterns, so this module
carries two evaluators and pins them against each other rather than letting them drift.

* `RecurrentBlockScorer` (registered in `recurrent_segmentation`) scores **one candidate
  block at a time**, which is what a boundary proposal asks for: a split touches two
  blocks and nothing else. It is used unchanged.
* `SkillBlockLikelihood` scores **every block of one skill at once**, which is what a `U`
  row proposal asks for: changing `U_k` changes every block labelled `k` and nothing else.
  Blocks are grouped by width so `vectorized_state_features` can replay a whole group
  along an array axis.

Both replay from `q_0 = 0` for every block and neither carries state across a block, so
they compute the same sum by different loop orders. `assert_evaluators_agree` checks that
claim numerically instead of asserting it in prose, and the Stage 6E tests run it.

## What invalidates what

`SkillBlockLikelihood` holds a feature cache keyed on the exact `(U_k, omega, block set)`
it was built at, and — like the Stage 6D evaluator — it is written only by an explicit
refresh, never by an evaluation. `beta`, `lambda_rep` and `lambda_back` do not enter the
`q` recursion, so a cache built at the current `(U_k, omega)` is exact for them. `U` and
`omega` do, so they force a full replay. A `(S, z)` move changes which blocks belong to
which skill, so it invalidates the block set itself.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from hpop.mcmc_original.fast_segmentation_kernel import (
    FastSegmentationKernel, key_of, segmentation_of, spans_of,
)
from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.proposals import MoveType
from hpop.mcmc_original.recurrent_joint_scalar_mcmc import vectorized_state_features
from hpop.mcmc_original.recurrent_scalar_mcmc import build_proposal, scalar_mh_step
from hpop.mcmc_original.recurrent_scalar_posterior import (
    cached_batch_log_likelihood, log_prior,
)
from hpop.mcmc_original.sampler_u import propose_row
from hpop.mcmc_original.stage6c_frozen import (
    log_jacobian_rho, log_rho_prior, log_structural_prior, rho_from_unconstrained,
    rho_to_unconstrained,
)
from hpop.mcmc_original.stage6d_frozen import SCALAR_ORDER, rho_is_in_support
from hpop.mcmc_original.stage6e_block_table import BlockScoreTable
from hpop.mcmc_original.stage6e_frozen import (
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH, log_boundary_prior_6e,
)
from hpop.mcmc_original.stage6e_state import (
    Stage6EModel, Stage6EState, initial_counts, transition_counts_of,
)
from hpop.mcmc_original.transitions import (
    log_transition_matrix, sample_transition_matrix,
)

__all__ = [
    "SkillBlockLikelihood", "TraceSegmentationTarget", "Stage6ESampler",
    "Stage6EChainResult", "assert_evaluators_agree", "segmentation_sweep",
    "sweep_once", "run_stage6e_chain", "CACHE_SAFE_SCALARS",
    "boundary_hamming", "occurrence_label_changes",
]

# omega enters the q recursion; the other three scalars do not.
CACHE_SAFE_SCALARS = ("beta", "lambda_rep", "lambda_back")


# ------------------------------------------------------------------ skill likelihood
@dataclass
class SkillBlockLikelihood:
    """Total recurrent log likelihood of every block carrying one skill.

    The block set is `(trace, start, end)` triples taken from the *currently accepted*
    segmentation. Blocks are bucketed by width, because `vectorized_state_features`
    replays a rectangular `(n_blocks, T)` array; every bucket starts from `q_0 = 0`.
    """

    traces: tuple
    epsilon: float
    _blocks: dict = field(default_factory=dict, repr=False)     # skill -> {width: array}
    _features: dict = field(default_factory=dict, repr=False)   # skill -> features
    _feature_key: dict = field(default_factory=dict, repr=False)

    def set_blocks(self, segmentations, n_skills: int) -> None:
        """Rebuild the per-skill block arrays. Invalidates every feature cache."""
        rows: dict = {k: {} for k in range(int(n_skills))}
        for index, segmentation in enumerate(segmentations):
            for start, end, skill in spans_of(key_of(segmentation)):
                rows[skill].setdefault(end - start, []).append(
                    self.traces[index][start:end])
        self._blocks = {k: {w: np.array(v, dtype=int) for w, v in widths.items()}
                        for k, widths in rows.items()}
        self._features.clear()
        self._feature_key.clear()

    def block_count(self, skill: int) -> int:
        return int(sum(a.shape[0] for a in self._blocks.get(int(skill), {}).values()))

    def refresh(self, skill: int, u_k, omega: float) -> None:
        """The ONLY writer of the feature cache, exactly as the Stage 6D evaluator is."""
        skill = int(skill)
        u_k = np.asarray(u_k, dtype=float)
        self._features[skill] = {w: vectorized_state_features(roles, u_k, float(omega))
                                 for w, roles in self._blocks.get(skill, {}).items()}
        self._feature_key[skill] = (u_k.tobytes(), float(omega))

    def full_replay(self, skill: int, u_k, beta: float, omega: float,
                    lambda_rep: float, lambda_back: float) -> float:
        """Complete replay from `q_0 = 0`. Reads no cache and writes none."""
        total = 0.0
        for roles in self._blocks.get(int(skill), {}).values():
            features = vectorized_state_features(roles, np.asarray(u_k, dtype=float),
                                                 float(omega))
            total += cached_batch_log_likelihood(features, float(beta), self.epsilon,
                                                 float(lambda_rep), float(lambda_back))
        return float(total)

    def cached(self, skill: int, u_k, beta: float, omega: float, lambda_rep: float,
               lambda_back: float) -> float:
        """Cached evaluation, valid only at the exact `(U_k, omega)` of the last refresh."""
        skill = int(skill)
        key = (np.asarray(u_k, dtype=float).tobytes(), float(omega))
        if self._feature_key.get(skill) != key:
            raise AssertionError(
                f"skill {skill}: cache was built at a different (U, omega); a cached read "
                "here would silently score the wrong model")
        return float(sum(
            cached_batch_log_likelihood(features, float(beta), self.epsilon,
                                        float(lambda_rep), float(lambda_back))
            for features in self._features[skill].values()))

    def total_full_replay(self, u_by_skill, beta, omega, lambda_rep, lambda_back) -> float:
        return float(sum(self.full_replay(k, u_by_skill[k], beta, omega, lambda_rep,
                                          lambda_back)
                         for k in range(np.asarray(u_by_skill).shape[0])))


def assert_evaluators_agree(model: Stage6EModel, state: Stage6EState,
                            tolerance: float = 1e-9) -> dict:
    """`sum_blocks scorer.score(...)` must equal `sum_skills SkillBlockLikelihood(...)`.

    The two differ only in the order the same independent per-block terms are summed, so
    they must agree to floating-point noise. Checking it is what licenses using the fast
    grouped evaluator for `U` and the per-block scorer for boundaries.
    """
    scorer = model.scorer_for(state)
    per_block = 0.0
    for index, segmentation in enumerate(state.segmentations):
        for segment in segmentation.segments:
            per_block += scorer.score(index, segment.start, segment.end, segment.skill)

    grouped = SkillBlockLikelihood(traces=model.traces, epsilon=model.epsilon)
    grouped.set_blocks(state.segmentations, model.n_skills)
    per_skill = grouped.total_full_replay(state.u_by_skill, state.beta, state.omega,
                                          state.lambda_rep, state.lambda_back)
    difference = abs(per_block - per_skill)
    return {"per_block_sum": per_block, "per_skill_sum": per_skill,
            "absolute_difference": difference, "tolerance": tolerance,
            "pass": bool(difference < tolerance)}


# --------------------------------------------------------------- segmentation target
@dataclass
class TraceSegmentationTarget:
    """`log w(S_n, z_n)` for one trace, over the fast `((end, skill), ...)` key.

    Identical in content to `recurrent_segmentation.segmentation_log_weight`, which it is
    tested against; only the representation differs. `-inf` is unreachable here because
    the kernel's neighbourhood is already restricted to legal keys, but an illegal key
    passed in from outside still returns `-inf` rather than a wrong number.
    """

    trace_index: int
    trace_length: int
    scorer: object
    delta_b: float = DELTA_B
    log_pi: np.ndarray = None
    log_transition: np.ndarray = None
    min_width: int = MIN_BLOCK_WIDTH
    max_width: int = MAX_BLOCK_WIDTH

    def set_path_prior(self, log_pi, log_transition) -> None:
        self.log_pi = np.asarray(log_pi, dtype=float)
        self.log_transition = np.asarray(log_transition, dtype=float)

    def __call__(self, key) -> float:
        spans = spans_of(key)
        previous = None
        emission = 0.0
        for start, end, skill in spans:
            width = end - start
            if not (self.min_width <= width <= self.max_width):
                return -math.inf
            if skill == previous:
                return -math.inf
            emission += self.scorer.score(self.trace_index, start, end, skill)
            previous = skill
        total = (emission
                 + log_boundary_prior_6e(self.trace_length, len(spans), self.delta_b)
                 + float(self.log_pi[spans[0][2]]))
        for (_, _, left), (_, _, right) in zip(spans[:-1], spans[1:]):
            total += float(self.log_transition[left, right])
        return total


# -------------------------------------------------------------------- movement metrics
def boundary_hamming(key_a, key_b) -> int:
    """Symmetric difference of the internal cut sets — the segmentation movement metric."""
    a = {e for e, _ in key_a[:-1]}
    b = {e for e, _ in key_b[:-1]}
    return len(a ^ b)


def occurrence_label_changes(key_a, key_b) -> int:
    """Number of occurrences whose skill label differs between two segmentations."""
    def labels(key):
        out = []
        for start, end, skill in spans_of(key):
            out.extend([skill] * (end - start))
        return np.array(out, dtype=int)
    la, lb = labels(key_a), labels(key_b)
    return int((la != lb).sum())


# ------------------------------------------------------------------ segmentation sweep
def segmentation_sweep(keys, targets, kernels, n_proposals: int, rng,
                       proposed: dict, accepted: dict, invalid: dict) -> tuple:
    """`n_proposals` local Metropolis steps per trace, in trace order.

    The Hastings ratio is the Stage 5 one, formed from the fast kernel's neighbourhood
    counting; it is never assumed symmetric. A proposal that lands back on the current
    state (an empty neighbourhood for the drawn move type) is counted as proposed and as
    an impossible move, not as a rejection of a different state.
    """
    keys = list(keys)
    movement = {"boundary_hamming": 0, "label_changes": 0}
    if int(n_proposals) <= 0:
        # Exactly equivalent to falling through the loop below: with no proposals every
        # key is returned unchanged, no movement is recorded and no random number is
        # drawn. Returning here skips the per-trace `target(current)` evaluation, which is
        # otherwise pure waste — and it is not small. The oracle-boundary control and the
        # Step 7B sweep both run with zero segmentation proposals, and on the 100-trace
        # corpus that one evaluation per trace costs 510 uncached block replays a sweep.
        return tuple(keys), movement
    for n, (target, kernel) in enumerate(zip(targets, kernels)):
        current = keys[n]
        current_value = target(current)
        for _ in range(int(n_proposals)):
            candidate, move = kernel.sample_proposal(current, rng)
            proposed[move] = proposed.get(move, 0) + 1
            if candidate == current:
                invalid[move] = invalid.get(move, 0) + 1
                continue
            forward = kernel.proposal_prob(current, candidate)
            reverse = kernel.proposal_prob(candidate, current)
            if forward <= 0.0 or reverse <= 0.0:
                invalid[move] = invalid.get(move, 0) + 1
                continue
            candidate_value = target(candidate)
            if not math.isfinite(candidate_value) and candidate_value != -math.inf:
                # NaN, not -inf: an unrepresentable likelihood, rejected explicitly rather
                # than left to a comparison that silently evaluates to False.
                invalid[move] = invalid.get(move, 0) + 1
                continue
            log_alpha = ((candidate_value - current_value)
                         + math.log(reverse) - math.log(forward))
            if log_alpha >= 0.0 or math.log(rng.random()) < log_alpha:
                movement["boundary_hamming"] += boundary_hamming(current, candidate)
                movement["label_changes"] += occurrence_label_changes(current, candidate)
                current, current_value = candidate, candidate_value
                accepted[move] = accepted.get(move, 0) + 1
        keys[n] = current
    return tuple(keys), movement


# -------------------------------------------------------------------------- the sweep
@dataclass
class Stage6ESampler:
    """Holds everything a chain reuses across sweeps: kernels, targets, evaluators."""

    model: Stage6EModel
    scales: dict
    n_proposals_per_trace: int
    max_shift: int | None = None
    use_block_table: bool = False
    _kernels: list = field(default_factory=list, repr=False)
    _targets: list = field(default_factory=list, repr=False)
    _skill: SkillBlockLikelihood | None = field(default=None, repr=False)
    _table: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._kernels = [
            FastSegmentationKernel(trace_length=len(t), n_skills=self.model.n_skills,
                                   min_width=self.model.min_width,
                                   max_width=self.model.max_width,
                                   max_shift=self.max_shift)
            for t in self.model.traces]
        self._skill = SkillBlockLikelihood(traces=self.model.traces,
                                           epsilon=self.model.epsilon)
        if self.use_block_table:
            self._table = BlockScoreTable(
                traces=self.model.traces, epsilon=self.model.epsilon,
                n_skills=self.model.n_skills, min_width=self.model.min_width,
                max_width=self.model.max_width)

    def prepare(self, state: Stage6EState):
        """Return the object the segmentation targets read block scores from.

        With `use_block_table`, every candidate block for the current parameters is
        replayed once in a batch and the phase then reads a table; without it, the
        registered per-block scorer is used directly. Both replay from `q_0 = 0` and are
        pinned against each other by `assert_table_matches_scorer`.
        """
        scorer = self.model.scorer_for(state)
        if self._table is not None:
            self._table.refresh(state.u_by_skill, state.beta, state.omega,
                                state.lambda_rep, state.lambda_back)
            source = self._table
        else:
            source = scorer
        if not self._targets:
            self._targets = [
                TraceSegmentationTarget(trace_index=n, trace_length=len(t), scorer=source,
                                        delta_b=self.model.delta_b,
                                        min_width=self.model.min_width,
                                        max_width=self.model.max_width)
                for n, t in enumerate(self.model.traces)]
        else:
            for target in self._targets:
                target.scorer = source
        return scorer


def sweep_once(state: Stage6EState, sampler: Stage6ESampler, rng) -> Stage6EState:
    """One registered global sweep. Returns a NEW state; the input is not mutated."""
    model = sampler.model
    scales = sampler.scales
    state = state.copy()
    scorer = sampler.prepare(state)
    proposed, accepted, invalid = dict(state.proposed), dict(state.accepted), dict(
        state.invalid)

    # ---- 1. (S, z): one registered local sweep per trace ------------------------------
    log_pi = np.log(state.pi)
    log_transition = log_transition_matrix(state.transition)
    for target in sampler._targets:
        target.set_path_prior(log_pi, log_transition)
    keys = [key_of(s) for s in state.segmentations]
    keys, movement = segmentation_sweep(keys, sampler._targets, sampler._kernels,
                                        sampler.n_proposals_per_trace, rng,
                                        proposed, accepted, invalid)
    state.segmentations = tuple(segmentation_of(k) for k in keys)

    # ---- 2. (pi, P) from the LATEST accepted segmentation, frozen Stage 3 updates -----
    if model.infer_pi_P:
        counts = transition_counts_of(state.segmentations, model.n_skills)
        state.transition = sample_transition_matrix(counts, model.n_skills, rng,
                                                    model.eta_transition)
        state.pi = rng.dirichlet(model.eta_initial
                                 + initial_counts(state.segmentations, model.n_skills))
        proposed["pi_P"] = proposed.get("pi_P", 0) + 1
        accepted["pi_P"] = accepted.get("pi_P", 0) + 1      # Gibbs: always accepted

    # the block set changed, so the grouped evaluator must be rebuilt
    skill_ll = sampler._skill
    skill_ll.set_blocks(state.segmentations, model.n_skills)

    u = np.array(state.u_by_skill, dtype=float, copy=True)
    current_ll = {k: skill_ll.full_replay(k, u[k], state.beta, state.omega,
                                          state.lambda_rep, state.lambda_back)
                  for k in range(model.n_skills)}
    current_structural = {k: log_structural_prior(u[k], state.rho)
                          for k in range(model.n_skills)}

    # ---- 3. U, one row of one skill at a time. Symmetric, so no Hastings term. --------
    for k in range(model.n_skills):
        for row in range(u.shape[1]):
            candidate = propose_row(u[k], row, scales["U"], rng)
            proposed["U"] = proposed.get("U", 0) + 1
            candidate_structural = log_structural_prior(candidate, state.rho)
            if not math.isfinite(candidate_structural):
                invalid["U"] = invalid.get("U", 0) + 1
                continue
            # A changed U can change H = h(U) and with it the whole q trajectory, so this
            # is a COMPLETE replay from q_0 = 0, never an edited cache.
            candidate_ll = skill_ll.full_replay(k, candidate, state.beta, state.omega,
                                                state.lambda_rep, state.lambda_back)
            if not math.isfinite(candidate_ll):
                invalid["U"] = invalid.get("U", 0) + 1
                continue
            log_alpha = ((candidate_ll - current_ll[k])
                         + (candidate_structural - current_structural[k]))
            if log_alpha >= 0.0 or math.log(rng.random()) < log_alpha:
                u[k] = candidate
                current_ll[k] = candidate_ll
                current_structural[k] = candidate_structural
                accepted["U"] = accepted.get("U", 0) + 1
    state.u_by_skill = u

    # ---- 4. rho. Acts only through p(U | rho); consumes no likelihood evaluation. -----
    current_rho = float(state.rho)
    z = rho_to_unconstrained(current_rho)
    candidate_rho = rho_from_unconstrained(z + scales["rho"] * rng.normal())
    proposed["rho"] = proposed.get("rho", 0) + 1
    if not rho_is_in_support(candidate_rho):
        invalid["rho"] = invalid.get("rho", 0) + 1
    else:
        candidate_structural = {k: log_structural_prior(u[k], candidate_rho)
                                for k in range(model.n_skills)}
        candidate_rho_prior = log_rho_prior(candidate_rho)
        if (math.isfinite(candidate_rho_prior)
                and all(math.isfinite(v) for v in candidate_structural.values())):
            log_alpha = ((sum(candidate_structural.values())
                          - sum(current_structural.values()))
                         + (candidate_rho_prior - log_rho_prior(current_rho))
                         + (log_jacobian_rho(candidate_rho)
                            - log_jacobian_rho(current_rho)))
            if log_alpha >= 0.0 or math.log(rng.random()) < log_alpha:
                state.rho = candidate_rho
                current_structural = candidate_structural
                accepted["rho"] = accepted.get("rho", 0) + 1

    # ---- 5-8. the four recurrent scalars, in the frozen Stage 6B order ----------------
    # The feature cache is refreshed ONCE here, at the current (U, omega), and again only
    # if omega is accepted. beta, lambda_rep and lambda_back do not enter the q recursion,
    # so a cache built at (U, omega) is exact for them; omega does, so it never reads one.
    values = {name: float(getattr(state, name)) for name in SCALAR_ORDER}
    total_ll = float(sum(current_ll.values()))
    for k in range(model.n_skills):
        skill_ll.refresh(k, u[k], values["omega"])

    for name in SCALAR_ORDER:
        allow_cache = name in CACHE_SAFE_SCALARS

        def scalar_log_posterior(candidate, _name=name, _v=dict(values),
                                 _cache=allow_cache):
            prior = log_prior(_name, candidate)
            if not math.isfinite(prior):
                return -math.inf
            trial = dict(_v)
            trial[_name] = candidate
            evaluate = skill_ll.cached if _cache else skill_ll.full_replay
            value = prior + float(sum(
                evaluate(k, u[k], trial["beta"], trial["omega"], trial["lambda_rep"],
                         trial["lambda_back"])
                for k in range(model.n_skills)))
            # A NaN must be rejected, and saying so explicitly is not optional. At a large
            # enough `lambda_back` the registered one-step probabilities underflow: every
            # feasible role's exponent goes to -inf while the arg-max role has feasibility
            # 0, so `weights.sum()` is exactly 0 and the frozen likelihood returns NaN.
            # `scalar_mh_step` then forms `min(0.0, NaN)`, which Python evaluates to 0.0 —
            # because every comparison with NaN is False, `min` simply keeps its first
            # argument — so the step would ACCEPT the NaN and the chain would carry a
            # non-finite target from there on. Mapping it to -inf here rejects it, which
            # is also the right answer on the merits: those parameter values give a
            # likelihood too small to represent.
            return value if math.isfinite(value) else -math.inf

        current_posterior = log_prior(name, values[name]) + total_ll
        new_value, new_posterior, was_accepted = scalar_mh_step(
            values[name], current_posterior, scalar_log_posterior,
            build_proposal(name, scales[name]), rng)
        proposed[name] = proposed.get(name, 0) + 1
        if was_accepted:
            values[name] = new_value
            accepted[name] = accepted.get(name, 0) + 1
            total_ll = new_posterior - log_prior(name, new_value)
            if name == "omega":
                # the q trajectory changed: every cached feature bundle is now stale
                for k in range(model.n_skills):
                    skill_ll.refresh(k, u[k], values["omega"])
    for scalar_name, value in values.items():
        setattr(state, scalar_name, value)

    # ---- bookkeeping -----------------------------------------------------------------
    scorer.set_parameters(u_by_skill=state.u_by_skill, beta=state.beta, omega=state.omega,
                          lambda_rep=state.lambda_rep, lambda_back=state.lambda_back,
                          epsilon=model.epsilon)
    blocks = total_ll
    boundary = float(sum(log_boundary_prior_6e(len(model.traces[n]),
                                               len(s.segments), model.delta_b)
                         for n, s in enumerate(state.segmentations)))
    log_pi = np.log(state.pi)
    log_transition = log_transition_matrix(state.transition)
    initial = float(sum(log_pi[s.segments[0].skill] for s in state.segmentations))
    transition = 0.0
    for s in state.segmentations:
        path = [seg.skill for seg in s.segments]
        for a, b in zip(path[:-1], path[1:]):
            transition += float(log_transition[a, b])
    structural = float(sum(current_structural.values()))
    rho_prior = log_rho_prior(state.rho)
    scalar_priors = {n: log_prior(n, values[n]) for n in SCALAR_ORDER}
    pi_prior, p_prior = model.log_pi_P_prior(state)

    state.components = {
        "log_block_likelihood": blocks, "log_boundary_prior": boundary,
        "log_initial": initial, "log_transition": transition,
        "log_structural_prior": structural, "log_rho_prior": rho_prior,
        "log_pi_prior": pi_prior, "log_P_prior": p_prior,
        "log_target": (blocks + boundary + initial + transition + structural + rho_prior
                       + float(sum(scalar_priors.values())) + pi_prior + p_prior),
        "boundary_hamming_moved": movement["boundary_hamming"],
        "label_changes_moved": movement["label_changes"],
    }
    state.proposed, state.accepted, state.invalid = proposed, accepted, invalid
    state.iteration += 1
    state.cache_version = scorer.version
    state.rng_state = rng.bit_generator.state
    return state


# ---------------------------------------------------------------------------- the run
@dataclass
class Stage6EChainResult:
    u_draws: np.ndarray                 # (n_draws, K, m, d) float32
    scalars: dict
    pi_draws: np.ndarray
    transition_draws: np.ndarray
    segment_counts: np.ndarray          # (n_draws, N) int16
    boundary_keys: list                 # per draw, tuple of per-trace keys (small runs)
    # (n_draws, N, max J) int8, padded with -1. Because `P` forbids self-transitions,
    # adjacent segments always carry different labels, so the occurrence-label array
    # determines the segmentation exactly: a cut is precisely a label change. This is the
    # compact representation the 100-trace corpus is stored in.
    occurrence_labels: np.ndarray
    log_target: np.ndarray
    log_block_likelihood: np.ndarray
    relation_counts: np.ndarray         # (n_draws, K)
    proposed: dict
    accepted: dict
    invalid: dict
    proposed_after_burn_in: dict
    accepted_after_burn_in: dict
    movement: dict
    final_state: Stage6EState
    runtime_seconds: float
    chain: int
    seed: int

    def acceptance(self, post_burn_in: bool = True) -> dict:
        p = self.proposed_after_burn_in if post_burn_in else self.proposed
        a = self.accepted_after_burn_in if post_burn_in else self.accepted
        return {k: (a.get(k, 0) / p[k] if p.get(k) else float("nan")) for k in p}


def _write_checkpoint(path, chain, state, sweep, k, u_draws, scalars, pi_draws,
                      transition_draws, segment_counts, relation_counts, log_target,
                      log_blocks, labels) -> None:
    """Atomic-ish checkpoint: draws to `.npz`, resumable state (with RNG) to `.json`."""
    import json
    from pathlib import Path

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "u_draws": u_draws[:k], "pi_draws": pi_draws[:k],
        "transition_draws": transition_draws[:k], "segment_counts": segment_counts[:k],
        "relation_counts": relation_counts[:k], "log_target": log_target[:k],
        "log_block_likelihood": log_blocks[:k],
        **{f"scalar_{n}": v[:k] for n, v in scalars.items()},
    }
    if labels is not None:
        payload["occurrence_labels"] = labels[:k]
    # np.savez_compressed appends ".npz" unless the name already ends in it, so the
    # temporary name carries the suffix and the rename is a plain replace.
    temporary = directory / f"chain{chain}_checkpoint_partial.npz"
    np.savez_compressed(temporary, **payload)
    temporary.replace(directory / f"chain{chain}_checkpoint.npz")
    (directory / f"chain{chain}_checkpoint.json").write_text(json.dumps(
        {"sweep": int(sweep), "n_retained": int(k), "state": state.to_dict()}))


def run_stage6e_chain(model: Stage6EModel, start: Stage6EState, scales: dict,
                      n_proposals_per_trace: int, num_sweeps: int, burn_in: int,
                      thin: int, seed: int, chain: int = 0,
                      max_shift: int | None = None, rng=None,
                      state: Stage6EState | None = None,
                      store_labels: bool = True, store_keys: bool = True,
                      use_block_table: bool = False, progress_every: int = 0,
                      checkpoint_path=None, checkpoint_every: int = 0
                      ) -> Stage6EChainResult:
    """Run one chain. `state`/`rng` allow a deterministic, bit-identical continuation."""
    if burn_in >= num_sweeps:
        raise ValueError("burn_in must be smaller than num_sweeps")
    if thin < 1:
        raise ValueError("thin must be at least 1")

    sampler = Stage6ESampler(model=model, scales=dict(scales),
                             n_proposals_per_trace=int(n_proposals_per_trace),
                             max_shift=max_shift, use_block_table=use_block_table)
    rng = np.random.default_rng(seed) if rng is None else rng
    state = (start.copy() if state is None else state)
    state.chain = chain
    for bucket in (state.proposed, state.accepted, state.invalid):
        for move in MoveType.ALL:
            bucket.setdefault(move, 0)
        for name in ("U", "rho", "pi_P", *SCALAR_ORDER):
            bucket.setdefault(name, 0)

    start_iteration = int(state.iteration)
    capacity = len(range(max(burn_in, start_iteration), num_sweeps, thin)) + 1
    n_traces = len(model.traces)
    K, m, d = np.asarray(state.u_by_skill).shape
    max_length = max(len(t) for t in model.traces)

    u_draws = np.empty((capacity, K, m, d), dtype=np.float32)
    scalars = {n: np.empty(capacity) for n in (*SCALAR_ORDER, "rho")}
    pi_draws = np.empty((capacity, K), dtype=np.float32)
    transition_draws = np.empty((capacity, K, K), dtype=np.float32)
    segment_counts = np.empty((capacity, n_traces), dtype=np.int16)
    relation_counts = np.empty((capacity, K), dtype=np.int16)
    log_target = np.empty(capacity)
    log_blocks = np.empty(capacity)
    boundary_keys: list = []
    labels = (np.full((capacity, n_traces, max_length), -1, dtype=np.int8)
              if store_labels else None)

    post_proposed: dict = {}
    post_accepted: dict = {}
    movement = {"boundary_hamming": 0, "label_changes": 0}

    k = 0
    began = time.perf_counter()
    for i in range(start_iteration, num_sweeps):
        before_p, before_a = dict(state.proposed), dict(state.accepted)
        state = sweep_once(state, sampler, rng)
        movement["boundary_hamming"] += state.components["boundary_hamming_moved"]
        movement["label_changes"] += state.components["label_changes_moved"]
        if i >= burn_in:
            for name in state.proposed:
                post_proposed[name] = (post_proposed.get(name, 0)
                                       + state.proposed[name] - before_p.get(name, 0))
                post_accepted[name] = (post_accepted.get(name, 0)
                                       + state.accepted[name] - before_a.get(name, 0))
            if (i - burn_in) % thin == 0 and k < capacity:
                u_draws[k] = state.u_by_skill
                for name in SCALAR_ORDER:
                    scalars[name][k] = getattr(state, name)
                scalars["rho"][k] = state.rho
                pi_draws[k] = state.pi
                transition_draws[k] = state.transition
                segment_counts[k] = state.segment_counts
                for s in range(K):
                    relation_counts[k, s] = int(precedence_from_u(
                        state.u_by_skill[s]).sum())
                log_target[k] = state.components["log_target"]
                log_blocks[k] = state.components["log_block_likelihood"]
                if store_keys:
                    boundary_keys.append(tuple(key_of(s) for s in state.segmentations))
                if labels is not None:
                    for n, arr in enumerate(state.occurrence_labels()):
                        labels[k, n, :len(arr)] = arr
                k += 1
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    chain {chain}: sweep {i + 1:,}/{num_sweeps:,} "
                  f"({time.perf_counter() - began:.0f}s)", flush=True)
        if checkpoint_path and checkpoint_every and (i + 1) % checkpoint_every == 0:
            # A multi-hour chain that is interrupted should not have to start over. The
            # state carries its own RNG, so `--resume` continues bit-identically from
            # here; the partial draws are written alongside so nothing already computed
            # is lost. This is a crash-recovery device, NOT the section 14 continuation
            # mechanism, which exists for failed convergence gates and is recorded
            # separately in continuation_history.json.
            _write_checkpoint(checkpoint_path, chain, state, i + 1, k, u_draws, scalars,
                              pi_draws, transition_draws, segment_counts,
                              relation_counts, log_target, log_blocks, labels)
    runtime = time.perf_counter() - began

    return Stage6EChainResult(
        u_draws=u_draws[:k], scalars={n: v[:k] for n, v in scalars.items()},
        pi_draws=pi_draws[:k], transition_draws=transition_draws[:k],
        segment_counts=segment_counts[:k], boundary_keys=boundary_keys,
        occurrence_labels=(labels[:k] if labels is not None else np.empty((0,))),
        log_target=log_target[:k], log_block_likelihood=log_blocks[:k],
        relation_counts=relation_counts[:k],
        proposed=dict(state.proposed), accepted=dict(state.accepted),
        invalid=dict(state.invalid), proposed_after_burn_in=post_proposed,
        accepted_after_burn_in=post_accepted, movement=movement, final_state=state,
        runtime_seconds=runtime, chain=chain, seed=seed)
