"""Stage 6E — recurrent candidate-block scoring and the unknown-boundary move kernel.

Three objects live here, and the division between them is the point:

* `RecurrentBlockScorer` — scores one candidate block `[a, b)` under one skill's `U_k`,
  always from `q_0 = 0`, behind a **versioned** cache whose key names every quantity the
  score depends on.
* `Stage6EMoveKernel` — the Stage 5 `LocalMoveKernel` with its legality predicate
  replaced and *nothing else*. The neighbourhood counting and the Hastings ratio are the
  Stage 5 code, inherited rather than re-derived.
* `log_target_stage6e` — the direct target, callable with no MH helper anywhere in its
  call graph, returning a full decomposition.

## Why `q_0 = 0` for every candidate block, without exception

The recurrent likelihood carries a state `q` that accumulates within a block. A candidate
block is a *hypothesis* about where a skill execution starts, so its score must be the
likelihood of that execution starting fresh. Letting `q` leak in from the block to the
left would score a different model, and would make the answer depend on the order in
which candidates happened to be evaluated. Every entry point here therefore rebuilds the
trajectory from zero, and `test_no_recurrent_state_leakage` pins it.

A boundary move changes **two** adjacent blocks. Both are rescored from `q_0 = 0`;
neither inherits anything from the other.

## Why the cache is versioned rather than keyed on floats

The score depends on `(trace, a, b, k, U_k, beta, omega, lambda_rep, lambda_back,
epsilon, config)`. Four of those are floats, and keying a dictionary on float tuples is
both slow and fragile — two bit-patterns that print identically are different keys, and a
scalar that returns to a previous value would silently reuse a stale entry only if the
bits matched exactly, which is the worst kind of intermittent.

Instead the scorer holds a monotone integer `version`. Any change to `U`, to a recurrent
scalar or to `epsilon` bumps the version, which invalidates every cached score at once.
The key is then `(version, trace, a, b, k)` — all integers. Correctness does not depend on
float equality anywhere, and a rejected proposal cannot leave anything behind because the
scorer is never written to by an evaluation, only by an explicit `set_parameters`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from hpop.mcmc_original.proposals import LocalMoveKernel, MoveType
from hpop.mcmc_original.recurrent_joint_scalar_mcmc import vectorized_state_features
from hpop.mcmc_original.recurrent_scalar_posterior import cached_batch_log_likelihood
from hpop.mcmc_original.stage6e_frozen import (
    MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH, log_boundary_prior_6e, log_path_prior_6e,
)
from hpop.mcmc_original.types import Segment, Segmentation

__all__ = [
    "RecurrentBlockScorer", "recurrent_compatible_skills", "Stage6EMoveKernel",
    "log_target_stage6e", "segmentation_log_weight", "path_of",
    "is_legal_segmentation",
]


# ------------------------------------------------------------------- block scoring
@dataclass
class RecurrentBlockScorer:
    """Scores `log p_RFS(x[a:b] | U_k, scalars, epsilon)` from `q_0 = 0`, with a cache.

    `traces` is a tuple of role sequences, one per trace. `u_by_skill` is `(K, m, d)`.
    """

    traces: tuple
    epsilon: float
    u_by_skill: np.ndarray
    beta: float
    omega: float
    lambda_rep: float
    lambda_back: float
    max_width: int = MAX_BLOCK_WIDTH
    min_width: int = MIN_BLOCK_WIDTH
    version: int = 0
    _cache: dict = field(default_factory=dict, repr=False)
    full_replay_calls: int = 0
    cached_calls: int = 0

    def __post_init__(self) -> None:
        self.traces = tuple(tuple(int(v) for v in t) for t in self.traces)
        self.u_by_skill = np.array(self.u_by_skill, dtype=float, copy=True)
        if self.u_by_skill.ndim != 3:
            raise ValueError(f"u_by_skill must be (K, m, d), got {self.u_by_skill.shape}")

    # -- parameter lifecycle ----------------------------------------------------------
    def set_parameters(self, *, u_by_skill=None, beta=None, omega=None,
                       lambda_rep=None, lambda_back=None, epsilon=None) -> None:
        """The ONLY writer. Any change bumps the version, invalidating every score."""
        changed = False
        if u_by_skill is not None:
            new = np.asarray(u_by_skill, dtype=float)
            if new.shape != self.u_by_skill.shape or not np.array_equal(
                    new, self.u_by_skill):
                self.u_by_skill = np.array(new, dtype=float, copy=True)
                changed = True
        for name, value in (("beta", beta), ("omega", omega),
                            ("lambda_rep", lambda_rep), ("lambda_back", lambda_back),
                            ("epsilon", epsilon)):
            if value is not None and float(value) != float(getattr(self, name)):
                setattr(self, name, float(value))
                changed = True
        if changed:
            self.version += 1
            self._cache.clear()

    def set_skill_u(self, skill: int, u) -> None:
        """Update one skill's `U_k`. Bumps the version like any other parameter change."""
        u = np.asarray(u, dtype=float)
        if not np.array_equal(u, self.u_by_skill[skill]):
            self.u_by_skill[skill] = u
            self.version += 1
            self._cache.clear()

    @property
    def n_skills(self) -> int:
        return int(self.u_by_skill.shape[0])

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    # -- scoring ----------------------------------------------------------------------
    def width_is_legal(self, start: int, end: int) -> bool:
        width = end - start
        return self.min_width <= width <= self.max_width

    def replay(self, trace: int, start: int, end: int, skill: int) -> float:
        """Uncached full replay from `q_0 = 0`. No state enters or leaves."""
        roles = np.array([self.traces[trace][start:end]], dtype=int)
        features = vectorized_state_features(roles, self.u_by_skill[skill],
                                             float(self.omega))
        # q_0 is zeros by construction inside vectorized_state_features; asserted by test.
        self.full_replay_calls += 1
        return float(cached_batch_log_likelihood(
            features, float(self.beta), float(self.epsilon),
            float(self.lambda_rep), float(self.lambda_back)))

    def score(self, trace: int, start: int, end: int, skill: int) -> float:
        """Cached score. The key is all-integer: `(version, trace, start, end, skill)`."""
        key = (self.version, int(trace), int(start), int(end), int(skill))
        hit = self._cache.get(key)
        if hit is not None:
            self.cached_calls += 1
            return hit
        value = self.replay(trace, start, end, skill)
        self._cache[key] = value
        return value

    def cache_key_fields(self) -> tuple:
        """Everything the key stands for, for the audit record and the tests."""
        return ("version(-> U_by_skill, beta, omega, lambda_rep, lambda_back, epsilon)",
                "trace", "start", "end", "skill")


# ------------------------------------------------------------- legality / compatibility
def recurrent_compatible_skills(start: int, end: int, n_skills: int,
                                min_width: int = MIN_BLOCK_WIDTH,
                                max_width: int = MAX_BLOCK_WIDTH) -> tuple[int, ...]:
    """Skills a block `[start, end)` may carry.

    The Stage 5 predicate asks whether the block's CPA multiset equals the skill's role
    multiset. That is a BPOP question and it has no recurrent answer: a recurrent block
    repeats roles by design, so it is never a permutation of anything. Here the only
    block-level constraint is the registered width bound, and every skill shares the same
    role alphabet, so a legal-width block may carry any skill. The *sequence* constraint
    that adjacent segments differ is enforced by the target, through `P`'s zero diagonal.
    """
    width = end - start
    if not (min_width <= width <= max_width):
        return ()
    return tuple(range(int(n_skills)))


def is_legal_segmentation(segmentation: Segmentation, n_skills: int,
                          min_width: int = MIN_BLOCK_WIDTH,
                          max_width: int = MAX_BLOCK_WIDTH,
                          forbid_repeats: bool = True) -> bool:
    """Coverage and contiguity are enforced by `Segmentation` itself; this adds the rest."""
    segments = segmentation.segments
    for segment in segments:
        if not (min_width <= segment.length <= max_width):
            return False
        if not (0 <= segment.skill < n_skills):
            return False
    if forbid_repeats:
        for left, right in zip(segments[:-1], segments[1:]):
            if left.skill == right.skill:
                return False
    return True


def path_of(segmentation: Segmentation) -> list[int]:
    return [int(s.skill) for s in segmentation.segments]


# ------------------------------------------------------------------------- move kernel
def _seg(segments) -> Segmentation:
    return Segmentation(segments=tuple(segments))


@dataclass
class Stage6EMoveKernel(LocalMoveKernel):
    """The Stage 5 kernel with the legality predicate swapped and nothing else.

    `proposal_distribution`, `proposal_prob`, `sample_proposal` and the `mh_local_step`
    that consumes them are **inherited unchanged**, so the neighbourhood counting and the
    Hastings ratio remain the objects Stage 4 verified against an exact transition matrix.
    Only `neighbours` is overridden, and only because the Stage 5 compatibility predicate
    has no recurrent meaning (see `stage6e_frozen`, audit finding 4).

    Adjacent segments carrying the same skill are excluded from the neighbourhood rather
    than left to be rejected by a `-inf` target. Both are correct; excluding them keeps
    the proposal supported on the target's support, which is what makes the Hastings
    ratio finite in both directions.
    """

    n_skills: int = 3
    min_width: int = MIN_BLOCK_WIDTH
    max_width: int = MAX_BLOCK_WIDTH
    forbid_repeats: bool = True

    def _ok(self, segments) -> bool:
        try:
            candidate = _seg(segments)
        except ValueError:
            return False
        return is_legal_segmentation(candidate, self.n_skills, self.min_width,
                                     self.max_width, self.forbid_repeats)

    def _labels(self, start: int, end: int) -> tuple[int, ...]:
        return recurrent_compatible_skills(start, end, self.n_skills,
                                           self.min_width, self.max_width)

    def neighbours(self, segmentation: Segmentation, move: str) -> tuple[Segmentation, ...]:
        key = (move, segmentation)
        cached = self._neighbour_cache.get(key)
        if cached is not None:
            return cached
        segments = segmentation.segments
        out: list[Segmentation] = []

        if move == MoveType.RELABEL:
            for index, segment in enumerate(segments):
                for skill in self._labels(segment.start, segment.end):
                    if skill == segment.skill:
                        continue
                    replaced = list(segments)
                    replaced[index] = Segment(segment.start, segment.end, skill)
                    if self._ok(replaced):
                        out.append(_seg(replaced))

        elif move == MoveType.SPLIT:
            for index, segment in enumerate(segments):
                for cut in range(segment.start + 1, segment.end):
                    for left in self._labels(segment.start, cut):
                        for right in self._labels(cut, segment.end):
                            replaced = (list(segments[:index])
                                        + [Segment(segment.start, cut, left),
                                           Segment(cut, segment.end, right)]
                                        + list(segments[index + 1:]))
                            if self._ok(replaced):
                                out.append(_seg(replaced))

        elif move == MoveType.MERGE:
            for index in range(len(segments) - 1):
                left, right = segments[index], segments[index + 1]
                for skill in self._labels(left.start, right.end):
                    replaced = (list(segments[:index])
                                + [Segment(left.start, right.end, skill)]
                                + list(segments[index + 2:]))
                    if self._ok(replaced):
                        out.append(_seg(replaced))

        elif move == MoveType.SHIFT:
            for index in range(len(segments) - 1):
                left, right = segments[index], segments[index + 1]
                for boundary in range(left.start + 1, right.end):
                    if boundary == left.end:
                        continue
                    if self.max_shift is not None and abs(boundary - left.end) > self.max_shift:
                        continue
                    for ls in self._labels(left.start, boundary):
                        for rs in self._labels(boundary, right.end):
                            replaced = (list(segments[:index])
                                        + [Segment(left.start, boundary, ls),
                                           Segment(boundary, right.end, rs)]
                                        + list(segments[index + 2:]))
                            if self._ok(replaced):
                                out.append(_seg(replaced))
        else:
            raise ValueError(f"unknown move type {move!r}")

        result = tuple(out)
        self._neighbour_cache[key] = result
        return result


# ------------------------------------------------------------------------ direct target
def segmentation_log_weight(segmentation: Segmentation, trace_index: int,
                            trace_length: int, scorer: RecurrentBlockScorer,
                            log_pi: np.ndarray, log_transition: np.ndarray,
                            delta_b: float) -> dict:
    """`log w(S_n, z_n)`, decomposed. Returns `-inf` components for an illegal state."""
    if not is_legal_segmentation(segmentation, scorer.n_skills, scorer.min_width,
                                 scorer.max_width, forbid_repeats=True):
        return {"log_block_likelihood": -math.inf, "log_boundary_prior": -math.inf,
                "log_initial": -math.inf, "log_transition": -math.inf,
                "log_weight": -math.inf, "n_segments": len(segmentation.segments)}
    emission = 0.0
    for segment in segmentation.segments:
        emission += scorer.score(trace_index, segment.start, segment.end, segment.skill)
    path = path_of(segmentation)
    boundary = log_boundary_prior_6e(trace_length, len(path), delta_b)
    initial = float(log_pi[path[0]])
    transition = 0.0
    for prev, nxt in zip(path[:-1], path[1:]):
        transition += float(log_transition[prev, nxt])
    return {"log_block_likelihood": emission, "log_boundary_prior": boundary,
            "log_initial": initial, "log_transition": transition,
            "log_weight": emission + boundary + initial + transition,
            "n_segments": len(path)}


def log_target_stage6e(state, model) -> dict:
    """The registered direct target, with no MH helper anywhere in its call graph.

    `state` carries `segmentations`, `u_by_skill`, `rho`, the four scalars and — when
    active — `pi` and `P`. `model` carries the traces, `epsilon`, `delta_B` and the
    priors. Returns every component separately, which is what §3 asks for and what makes
    the parity checks in Stage 6E0 possible.
    """
    from hpop.mcmc_original.recurrent_scalar_posterior import log_prior
    from hpop.mcmc_original.stage6d_frozen import SCALAR_ORDER
    from hpop.mcmc_original.stage6c_frozen import log_rho_prior, log_structural_prior

    scorer = model.scorer_for(state)
    log_pi = np.log(np.asarray(state.pi, dtype=float))
    log_transition = model.log_transition(state)

    blocks = boundary = initial = transition = 0.0
    n_segments = []
    for index, segmentation in enumerate(state.segmentations):
        parts = segmentation_log_weight(segmentation, index, len(model.traces[index]),
                                        scorer, log_pi, log_transition, model.delta_b)
        if not math.isfinite(parts["log_weight"]):
            return {"log_block_likelihood": -math.inf, "log_boundary_prior": -math.inf,
                    "log_initial": -math.inf, "log_transition": -math.inf,
                    "log_structural_prior": -math.inf, "log_rho_prior": -math.inf,
                    "log_scalar_priors": {}, "log_pi_prior": -math.inf,
                    "log_P_prior": -math.inf, "log_target": -math.inf,
                    "n_segments": n_segments, "illegal_trace": index}
        blocks += parts["log_block_likelihood"]
        boundary += parts["log_boundary_prior"]
        initial += parts["log_initial"]
        transition += parts["log_transition"]
        n_segments.append(parts["n_segments"])

    structural = float(sum(log_structural_prior(state.u_by_skill[k], state.rho)
                           for k in range(state.u_by_skill.shape[0])))
    rho_prior = log_rho_prior(state.rho)
    scalar_priors = {name: log_prior(name, float(getattr(state, name)))
                     for name in SCALAR_ORDER}
    pi_prior, p_prior = model.log_pi_P_prior(state)

    total = (blocks + boundary + initial + transition + structural + rho_prior
             + float(sum(scalar_priors.values())) + pi_prior + p_prior)
    return {"log_block_likelihood": blocks, "log_boundary_prior": boundary,
            "log_initial": initial, "log_transition": transition,
            "log_structural_prior": structural, "log_rho_prior": rho_prior,
            "log_scalar_priors": scalar_priors, "log_pi_prior": pi_prior,
            "log_P_prior": p_prior, "log_target": total, "n_segments": n_segments}
