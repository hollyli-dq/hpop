"""Stage 6E — the serialisable joint state and the model container it is scored against.

The split is deliberate. `Stage6EModel` holds everything fixed by the registered
configuration (traces, `epsilon`, `delta_B`, priors, the block scorer); `Stage6EState`
holds everything the chain moves. `log_target_stage6e(state, model)` then needs no global
and no MH helper, which is what makes the Stage 6E0 parity checks meaningful — the direct
target and the transition kernel reach the same number by disjoint routes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from hpop.mcmc_original.recurrent_joint_scalar_mcmc import (
    _jsonable_rng_state, _rng_state_from_jsonable,
)
from hpop.mcmc_original.recurrent_segmentation import RecurrentBlockScorer, path_of
from hpop.mcmc_original.stage6d_frozen import SCALAR_ORDER
from hpop.mcmc_original.stage6e_frozen import (
    DELTA_B, ETA_INITIAL, ETA_TRANSITION, INFER_PI_P, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH,
    N_ROLES, N_SKILLS,
)
from hpop.mcmc_original.transitions import log_transition_matrix
from hpop.mcmc_original.types import Segment, Segmentation

__all__ = ["Stage6EState", "Stage6EModel", "initial_counts", "transition_counts_of"]


def initial_counts(segmentations, n_skills: int) -> np.ndarray:
    """`c[k]` = number of traces whose FIRST segment carries skill `k`."""
    counts = np.zeros(int(n_skills), dtype=float)
    for segmentation in segmentations:
        counts[int(segmentation.segments[0].skill)] += 1.0
    return counts


def transition_counts_of(segmentations, n_skills: int) -> np.ndarray:
    """`C[h, k]` over adjacent segments, pooled across traces. No terminal transition.

    Delegates the counting to the frozen Stage 3 implementation so the two definitions
    cannot drift; this function only assembles the label paths.
    """
    from hpop.mcmc_original.transitions import transition_counts
    return transition_counts([path_of(s) for s in segmentations], int(n_skills))


@dataclass
class Stage6EState:
    """Everything the Stage 6E chain moves. Serialisable and resumable."""

    segmentations: tuple
    u_by_skill: np.ndarray
    rho: float
    beta: float
    omega: float
    lambda_rep: float
    lambda_back: float
    pi: np.ndarray
    transition: np.ndarray
    iteration: int = 0
    proposed: dict = field(default_factory=dict)
    accepted: dict = field(default_factory=dict)
    invalid: dict = field(default_factory=dict)
    cache_version: int = 0
    rng_state: dict | None = None
    chain: int = 0
    components: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.segmentations = tuple(self.segmentations)
        self.u_by_skill = np.array(self.u_by_skill, dtype=float, copy=True)
        self.pi = np.array(self.pi, dtype=float, copy=True)
        self.transition = np.array(self.transition, dtype=float, copy=True)

    @property
    def n_skills(self) -> int:
        return int(self.u_by_skill.shape[0])

    @property
    def segment_counts(self) -> list[int]:
        return [len(s.segments) for s in self.segmentations]

    def boundary_sets(self) -> list[frozenset]:
        """Internal cut positions per trace — the permutation-invariant boundary summary."""
        return [frozenset(seg.end for seg in s.segments[:-1]) for s in self.segmentations]

    def occurrence_labels(self) -> list[np.ndarray]:
        """Per-occurrence skill label, which is what the label marginals are taken over."""
        out = []
        for s in self.segmentations:
            labels = np.empty(s.segments[-1].end, dtype=np.int16)
            for seg in s.segments:
                labels[seg.start:seg.end] = seg.skill
            out.append(labels)
        return out

    def copy(self) -> "Stage6EState":
        return Stage6EState(
            segmentations=self.segmentations, u_by_skill=self.u_by_skill,
            rho=self.rho, beta=self.beta, omega=self.omega,
            lambda_rep=self.lambda_rep, lambda_back=self.lambda_back,
            pi=self.pi, transition=self.transition, iteration=self.iteration,
            proposed=dict(self.proposed), accepted=dict(self.accepted),
            invalid=dict(self.invalid), cache_version=self.cache_version,
            rng_state=self.rng_state, chain=self.chain, components=dict(self.components))

    def to_dict(self) -> dict:
        return {
            "segmentations": [[[s.start, s.end, s.skill] for s in seg.segments]
                              for seg in self.segmentations],
            "u_by_skill": self.u_by_skill.tolist(),
            "rho": float(self.rho),
            **{name: float(getattr(self, name)) for name in SCALAR_ORDER},
            "pi": self.pi.tolist(), "transition": self.transition.tolist(),
            "iteration": int(self.iteration),
            "proposed": {k: int(v) for k, v in self.proposed.items()},
            "accepted": {k: int(v) for k, v in self.accepted.items()},
            "invalid": {k: int(v) for k, v in self.invalid.items()},
            "cache_version": int(self.cache_version), "chain": int(self.chain),
            "components": {k: (float(v) if isinstance(v, (int, float)) else v)
                           for k, v in self.components.items()
                           if not isinstance(v, dict)},
            "rng_state": _jsonable_rng_state(self.rng_state),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Stage6EState":
        segmentations = tuple(
            Segmentation(tuple(Segment(int(a), int(b), int(k)) for a, b, k in seg))
            for seg in payload["segmentations"])
        return cls(
            segmentations=segmentations,
            u_by_skill=np.array(payload["u_by_skill"], dtype=float),
            rho=float(payload["rho"]),
            **{name: float(payload[name]) for name in SCALAR_ORDER},
            pi=np.array(payload["pi"], dtype=float),
            transition=np.array(payload["transition"], dtype=float),
            iteration=int(payload["iteration"]),
            proposed=dict(payload.get("proposed", {})),
            accepted=dict(payload.get("accepted", {})),
            invalid=dict(payload.get("invalid", {})),
            cache_version=int(payload.get("cache_version", 0)),
            rng_state=_rng_state_from_jsonable(payload.get("rng_state")),
            chain=int(payload.get("chain", 0)))


@dataclass
class Stage6EModel:
    """Everything the registered configuration fixes, plus the one block scorer."""

    traces: tuple
    epsilon: float = 0.02
    delta_b: float = DELTA_B
    n_skills: int = N_SKILLS
    n_roles: int = N_ROLES
    min_width: int = MIN_BLOCK_WIDTH
    max_width: int = MAX_BLOCK_WIDTH
    infer_pi_P: bool = INFER_PI_P
    eta_transition: float = ETA_TRANSITION
    eta_initial: float = ETA_INITIAL
    _scorer: RecurrentBlockScorer | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.traces = tuple(tuple(int(v) for v in t) for t in self.traces)

    def scorer_for(self, state: Stage6EState) -> RecurrentBlockScorer:
        """One scorer, kept in step with the state through its versioned setter."""
        if self._scorer is None:
            self._scorer = RecurrentBlockScorer(
                traces=self.traces, epsilon=self.epsilon,
                u_by_skill=state.u_by_skill, beta=state.beta, omega=state.omega,
                lambda_rep=state.lambda_rep, lambda_back=state.lambda_back,
                max_width=self.max_width, min_width=self.min_width)
        else:
            self._scorer.set_parameters(
                u_by_skill=state.u_by_skill, beta=state.beta, omega=state.omega,
                lambda_rep=state.lambda_rep, lambda_back=state.lambda_back,
                epsilon=self.epsilon)
        return self._scorer

    def log_transition(self, state: Stage6EState) -> np.ndarray:
        """Elementwise log with forbidden self-transitions at -inf (frozen Stage 3)."""
        return log_transition_matrix(state.transition)

    def log_pi_P_prior(self, state: Stage6EState) -> tuple[float, float]:
        """Symmetric Dirichlet log densities, up to constants, or (0, 0) when fixed.

        The normalising constants are omitted deliberately and identically on every call,
        so they cancel in every ratio the sampler forms. `eta = 1` makes both flat, and
        the terms are kept in the decomposition anyway so a future non-flat prior has a
        place to go rather than being bolted on.
        """
        if not self.infer_pi_P:
            return 0.0, 0.0
        pi = np.asarray(state.pi, dtype=float)
        if np.any(pi <= 0.0) or not math.isclose(float(pi.sum()), 1.0, abs_tol=1e-9):
            return -math.inf, -math.inf
        pi_prior = float((self.eta_initial - 1.0) * np.log(pi).sum())
        p_prior = 0.0
        for h in range(self.n_skills):
            row = np.asarray(state.transition[h], dtype=float)
            allowed = [k for k in range(self.n_skills) if k != h]
            values = row[allowed]
            if np.any(values <= 0.0) or not math.isclose(float(values.sum()), 1.0,
                                                         abs_tol=1e-9):
                return pi_prior, -math.inf
            if row[h] != 0.0:
                return pi_prior, -math.inf          # the diagonal must be exactly zero
            p_prior += float((self.eta_transition - 1.0) * np.log(values).sum())
        return pi_prior, p_prior
