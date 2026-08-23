"""FULL-LATENT matched synthetic inference, without anchored skill identities.

This module is deliberately separate from :mod:`matched_condition_c`.  Condition C
fixed ``pi`` and ``P`` to their generating values and its marginal arm *added* an
occasional collapsed proposal to an ordinary conditional row sweep.  That is useful
historical infrastructure, but it is not the clean comparison required here.

FULL-LATENT targets, with the recurrent nuisance coordinates held fixed,

    p(S, z, U, pi, P | X, rho_0, beta*, omega*, lambda_rep*, lambda_back*,
                         delta_B*, epsilon*).

The two arms have the same state, start construction, FFBS refresh, pi/P Gibbs update,
single-row proposal family, scale, and cadence.  They differ only in the score of that
one structural proposal:

* ``FULL-COND`` uses the complete-path conditional likelihood at the stored ``(S,z)``;
* ``FULL-MARG`` uses the exact path-marginal forward normalisers and never reads the
  stored paths before the mandatory FFBS refresh.

One structural attempt occurs every ``structural_cadence`` sweeps in *both* arms.  This
uses the formerly registered C path-marginal cadence and the parent Stage-6C generic row
scale, but removes the confounding extra conditional row sweep.  Every sweep then performs exact FFBS and the
conjugate pi/P Gibbs transition.  No swap, transposition, tempering, or other rescue
transition is present.

The formal corpus stores hidden fields beside observations in its NPZ files.  The loader
below intentionally reads only named ``*_cpa`` arrays and validates their frozen file
hashes; this is a procedural truth seal.  Terminal truth recovery belongs in the
separate ``full_latent_recovery`` module and is never imported here.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import gammaln

from hpop.mcmc_original.collapsed_u_kernel import collapsed_u_mh_step, is_collapsed_sweep
from hpop.mcmc_original.collapsed_u_likelihood import CollapsedULikelihood
from hpop.mcmc_original.fast_segmentation_kernel import key_of, segmentation_of
from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.matched_condition_b import relation_indicator_vector
from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import (
    FFBSBlockTables,
    ffbs_segmentation_draw,
)
from hpop.mcmc_original.sampler_u import propose_row, sigma_rho_matrix
from hpop.mcmc_original.full_latent_constants import (
    DELTA_B,
    EPSILON,
    ETA_INITIAL,
    ETA_TRANSITION,
    FIXED_BETA,
    FIXED_LAMBDA_BACK,
    FIXED_LAMBDA_REP,
    FIXED_OMEGA,
    FIXED_RHO_0,
    LATENT_DIM,
    MAX_BLOCK_WIDTH,
    MIN_BLOCK_WIDTH,
    N_ROLES,
    N_SKILLS,
)
from hpop.mcmc_original.stage6c_frozen import log_structural_prior
from hpop.mcmc_original.stage6e_sampler import SkillBlockLikelihood
from hpop.mcmc_original.stage6e_state import (
    Stage6EModel,
    Stage6EState,
    initial_counts,
    transition_counts_of,
)
from hpop.mcmc_original.targets import log_boundary_prior, log_path_prior
from hpop.mcmc_original.transitions import (
    allowed_next,
    log_transition_matrix,
    sample_transition_matrix,
)
from hpop.mcmc_original.types import Segment, Segmentation

__all__ = [
    "FULL_LATENT_MODEL_ID", "ObservedCorpus", "FullLatentFixed",
    "FullLatentConfig", "FullLatentSampler", "FullLatentChain",
    "load_frozen_observed_corpus", "build_full_latent_model",
    "draw_initial_pi_p", "make_u_start", "initial_full_latent_state",
    "validate_pi_p", "validate_paths", "conditional_pi_log_density",
    "conditional_p_log_density", "gibbs_pi_p", "conditional_structural_mh_step",
    "full_latent_sweep_once", "complete_log_target",
    "independent_complete_log_target", "all_label_permutations",
    "permute_state_labels", "select_truth_free_probes", "invariant_summaries",
    "start_state_hash",
]


FULL_LATENT_MODEL_ID = "matched-full-latent-v1"
FULL_COND = "FULL-COND"
FULL_MARG = "FULL-MARG"
_ARMS = frozenset((FULL_COND, FULL_MARG))
_COND_MOVE = "U_full_conditional"
_MARG_MOVE = "U_collapsed"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_hash(payload) -> str:
    return _sha256_bytes(json.dumps(payload, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8"))


@dataclass(frozen=True)
class ObservedCorpus:
    """Frozen observations only; no hidden segmentation or label fields are retained."""

    train: tuple
    heldout: tuple
    corpus_hash: str
    train_hash: str
    heldout_hash: str
    generator_commit: str
    corpus_commit: str
    source_dir: str


def _load_observed_split(path: Path, expected_sha256: str) -> tuple:
    """Read only CPA arrays from a frozen split and verify its raw bytes first."""
    observed_hash = _sha256_bytes(path.read_bytes())
    if observed_hash != str(expected_sha256):
        raise AssertionError(
            f"frozen corpus split hash drift for {path}: {observed_hash} != "
            f"{expected_sha256}")
    with np.load(path, allow_pickle=False) as data:
        n_traces = int(np.asarray(data["n_traces"])[0])
        traces = []
        for index in range(n_traces):
            # Do not enumerate or load hidden NPZ fields.  The observation key is the
            # sole data input permitted to a sealed online sampler.
            key = f"t{index:03d}_cpa"
            if key not in data:
                raise AssertionError(f"missing observed corpus field {key} in {path}")
            traces.append(tuple(int(v) for v in np.asarray(data[key], dtype=np.int16)))
    return tuple(traces)


def load_frozen_observed_corpus(corpus_dir) -> ObservedCorpus:
    """Load the authoritative matched corpus without regenerating or opening truth.

    ``corpus_hash.json`` and ``config.json`` are provenance records, not recovery truth.
    In particular this function never opens ``truth_manifest.json`` and never imports the
    synthetic generator.
    """
    directory = Path(corpus_dir)
    hashes = json.loads((directory / "corpus_hash.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    train = _load_observed_split(directory / "train_traces.npz",
                                 hashes["train_npz_sha256"])
    heldout = _load_observed_split(directory / "heldout_traces.npz",
                                   hashes["heldout_npz_sha256"])
    if len(train) != int(config["n_train_traces"]):
        raise AssertionError("frozen train trace count does not match config")
    if len(heldout) != int(config["n_heldout_traces"]):
        raise AssertionError("frozen held-out trace count does not match config")
    return ObservedCorpus(
        train=train, heldout=heldout,
        corpus_hash=str(hashes["corpus_hash_sha256"]),
        train_hash=str(hashes["train_npz_sha256"]),
        heldout_hash=str(hashes["heldout_npz_sha256"]),
        generator_commit=str(config["generator_commit"]),
        # The frozen corpus report records this commit; storing it explicitly avoids a
        # truth-file read merely to establish provenance.
        corpus_commit="b199374baaf3f795ce5ee6dca16b7478bd07a3b9",
        source_dir=str(directory),
    )


@dataclass(frozen=True)
class FullLatentFixed:
    """The non-(S,z,U,pi,P) coordinates held fixed for the isolation experiment."""

    rho_0: float = float(FIXED_RHO_0)
    beta: float = float(FIXED_BETA)
    omega: float = float(FIXED_OMEGA)
    lambda_rep: float = float(FIXED_LAMBDA_REP)
    lambda_back: float = float(FIXED_LAMBDA_BACK)
    epsilon: float = float(EPSILON)
    delta_b: float = float(DELTA_B)

    def assert_unchanged(self, state: Stage6EState) -> None:
        expected = {
            "rho": self.rho_0, "beta": self.beta, "omega": self.omega,
            "lambda_rep": self.lambda_rep, "lambda_back": self.lambda_back,
        }
        for name, value in expected.items():
            if float(getattr(state, name)) != float(value):
                raise AssertionError(f"FULL-LATENT fixed coordinate {name} moved")

    def as_dict(self) -> dict:
        return {"rho_0": self.rho_0, "beta": self.beta, "omega": self.omega,
                "lambda_rep": self.lambda_rep, "lambda_back": self.lambda_back,
                "epsilon": self.epsilon, "delta_b": self.delta_b}


@dataclass(frozen=True)
class FullLatentConfig:
    """Frozen common structural proposal configuration for both formal arms."""

    arm: str
    structural_cadence: int = 10
    structural_scale: float = 0.5
    table_source: str = "batched"

    def __post_init__(self) -> None:
        if self.arm not in _ARMS:
            raise ValueError(f"unknown FULL-LATENT arm {self.arm!r}")
        if int(self.structural_cadence) < 1:
            raise ValueError("structural_cadence must be at least one")
        if float(self.structural_scale) <= 0.0:
            raise ValueError("structural_scale must be positive")

    @property
    def move_name(self) -> str:
        return _COND_MOVE if self.arm == FULL_COND else _MARG_MOVE

    def as_dict(self) -> dict:
        return {"arm": self.arm, "structural_cadence": int(self.structural_cadence),
                "structural_scale": float(self.structural_scale),
                "table_source": self.table_source}


def build_full_latent_model(traces, fixed: FullLatentFixed | None = None) -> Stage6EModel:
    """Construct the registered model with the existing pi/P prior switched on."""
    fixed = FullLatentFixed() if fixed is None else fixed
    return Stage6EModel(
        traces=tuple(tuple(int(v) for v in trace) for trace in traces),
        epsilon=float(fixed.epsilon), delta_b=float(fixed.delta_b),
        n_skills=N_SKILLS, n_roles=N_ROLES,
        min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
        infer_pi_P=True, eta_initial=ETA_INITIAL, eta_transition=ETA_TRANSITION,
    )


def validate_pi_p(state: Stage6EState, model: Stage6EModel,
                  tolerance: float = 1e-12) -> None:
    """Enforce the entire pi/P support before FFBS or a marginal forward call.

    ``semi_markov_ffbs.forward`` intentionally only consumes a supplied log-transition
    matrix.  The parent model makes the zero diagonal a support restriction, therefore a
    FULL-LATENT wrapper validates it explicitly rather than relying on callers to pass a
    legal matrix.
    """
    pi = np.asarray(state.pi, dtype=float)
    p = np.asarray(state.transition, dtype=float)
    K = int(model.n_skills)
    if pi.shape != (K,) or np.any(~np.isfinite(pi)) or np.any(pi <= 0.0):
        raise AssertionError("pi is not a positive length-K simplex vector")
    if not math.isclose(float(pi.sum()), 1.0, abs_tol=tolerance):
        raise AssertionError(f"pi sums to {pi.sum()}, not one")
    if p.shape != (K, K) or np.any(~np.isfinite(p)) or np.any(p < 0.0):
        raise AssertionError("P is not a finite nonnegative K by K matrix")
    if not np.array_equal(np.diag(p), np.zeros(K)):
        raise AssertionError("P must have an exactly zero diagonal")
    for h in range(K):
        allowed = np.asarray(allowed_next(h, K), dtype=int)
        if np.any(p[h, allowed] <= 0.0):
            raise AssertionError(f"P row {h} has a nonpositive allowed probability")
        if not math.isclose(float(p[h, allowed].sum()), 1.0, abs_tol=tolerance):
            raise AssertionError(f"allowed probabilities in P row {h} do not sum to one")
    pi_prior, p_prior = model.log_pi_P_prior(state)
    if not (math.isfinite(pi_prior) and math.isfinite(p_prior)):
        raise AssertionError("pi/P fails the registered model support")


def validate_paths(state: Stage6EState, model: Stage6EModel) -> None:
    """Check FFBS output is a complete legal path under zero-self-transition support."""
    for trace, segmentation in zip(model.traces, state.segmentations):
        if not segmentation.segments:
            raise AssertionError("empty segmentation")
        cursor = 0
        previous = None
        for segment in segmentation.segments:
            if segment.start != cursor or segment.end <= segment.start:
                raise AssertionError("segmentation is not a contiguous cover")
            width = segment.end - segment.start
            if not model.min_width <= width <= model.max_width:
                raise AssertionError("segmentation has a width outside model support")
            if not 0 <= int(segment.skill) < model.n_skills:
                raise AssertionError("segmentation has a skill outside model support")
            if previous is not None and int(segment.skill) == previous:
                raise AssertionError("a sampled path contains a forbidden self-transition")
            cursor, previous = segment.end, int(segment.skill)
        if cursor != len(trace):
            raise AssertionError("segmentation does not cover its trace")


def draw_initial_pi_p(model: Stage6EModel, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Truth-free, deterministic independent prior draw for a paired start."""
    rng = np.random.default_rng(int(seed))
    pi = rng.dirichlet(np.full(model.n_skills, float(model.eta_initial)))
    p = sample_transition_matrix(np.zeros((model.n_skills, model.n_skills)),
                                 model.n_skills, rng, model.eta_transition)
    probe = Stage6EState(segmentations=(),
                          u_by_skill=np.zeros((model.n_skills, model.n_roles, 2)),
                          rho=0.5, beta=1.0, omega=0.0, lambda_rep=1.0,
                          lambda_back=1.0, pi=pi, transition=p)
    validate_pi_p(probe, model)
    return pi, p


def make_u_start(index: int, seed: int, scale: float, fixed: FullLatentFixed,
                 n_skills: int = N_SKILLS, n_roles: int = N_ROLES,
                 latent_dim: int = LATENT_DIM) -> np.ndarray:
    """The established dispersed wrong-structure start construction, without truth."""
    del index  # index is retained in manifests; the seed supplies the actual randomness.
    rng = np.random.default_rng(int(seed))
    chol = np.linalg.cholesky(sigma_rho_matrix(int(latent_dim), float(fixed.rho_0)))
    return np.array([
        [float(scale) * (chol @ rng.standard_normal(int(latent_dim)))
         for _ in range(int(n_roles))]
        for _ in range(int(n_skills))
    ], dtype=float)


def _deterministic_initial_segmentations(model: Stage6EModel) -> tuple:
    """A legal deterministic tiling; FFBS immediately replaces it on sweep one."""
    out = []
    for trace_index, trace in enumerate(model.traces):
        remaining, widths = len(trace), []
        while remaining > model.max_width:
            step = (model.max_width if remaining - model.max_width >= model.min_width
                    else remaining - model.min_width)
            widths.append(step)
            remaining -= step
        widths.append(remaining)
        if (sum(widths) != len(trace)
                or any(not model.min_width <= width <= model.max_width for width in widths)):
            raise AssertionError(f"cannot construct deterministic legal tiling for J={len(trace)}")
        segments, start = [], 0
        for segment_index, width in enumerate(widths):
            # It is deterministic and truth-free; incrementing labels avoids a forbidden
            # adjacent self-transition even before the first Gibbs refresh.
            skill = (trace_index + segment_index) % model.n_skills
            segments.append(Segment(start, start + width, skill))
            start += width
        out.append(Segmentation(tuple(segments)))
    return tuple(out)


def initial_full_latent_state(model: Stage6EModel, u_start: np.ndarray,
                              pi_start: np.ndarray, p_start: np.ndarray,
                              fixed: FullLatentFixed) -> Stage6EState:
    """Build one paired legal initial state without an oracle or recovered endpoint."""
    state = Stage6EState(
        segmentations=_deterministic_initial_segmentations(model),
        u_by_skill=np.asarray(u_start, dtype=float), rho=fixed.rho_0,
        beta=fixed.beta, omega=fixed.omega, lambda_rep=fixed.lambda_rep,
        lambda_back=fixed.lambda_back, pi=np.asarray(pi_start, dtype=float),
        transition=np.asarray(p_start, dtype=float),
    )
    fixed.assert_unchanged(state)
    validate_pi_p(state, model)
    validate_paths(state, model)
    return state


def start_state_hash(state: Stage6EState) -> str:
    """Stable hash for a paired start, excluding transient counters and RNG state."""
    payload = {
        "u": np.asarray(state.u_by_skill, dtype=float).tolist(),
        "paths": [[[s.start, s.end, s.skill] for s in seg.segments]
                  for seg in state.segmentations],
        "pi": np.asarray(state.pi, dtype=float).tolist(),
        "P": np.asarray(state.transition, dtype=float).tolist(),
    }
    return _json_hash(payload)


def _log_dirichlet_density(value: np.ndarray, alpha: np.ndarray) -> float:
    value, alpha = np.asarray(value, dtype=float), np.asarray(alpha, dtype=float)
    if value.ndim != 1 or alpha.shape != value.shape or np.any(value <= 0.0):
        return -math.inf
    if not math.isclose(float(value.sum()), 1.0, abs_tol=1e-12):
        return -math.inf
    return float(gammaln(alpha.sum()) - gammaln(alpha).sum()
                 + ((alpha - 1.0) * np.log(value)).sum())


def conditional_pi_log_density(pi: np.ndarray, segmentations,
                               model: Stage6EModel) -> float:
    """Independent normalized conditional density for the pi Gibbs test."""
    alpha = (float(model.eta_initial)
             + initial_counts(segmentations, model.n_skills))
    return _log_dirichlet_density(np.asarray(pi, dtype=float), alpha)


def conditional_p_log_density(p: np.ndarray, segmentations,
                              model: Stage6EModel) -> float:
    """Independent normalized conditional density for all restricted P rows."""
    p = np.asarray(p, dtype=float)
    if p.shape != (model.n_skills, model.n_skills):
        return -math.inf
    if not np.array_equal(np.diag(p), np.zeros(model.n_skills)):
        return -math.inf
    counts = transition_counts_of(segmentations, model.n_skills)
    total = 0.0
    for h in range(model.n_skills):
        allowed = np.asarray(allowed_next(h, model.n_skills), dtype=int)
        alpha = float(model.eta_transition) + counts[h, allowed]
        term = _log_dirichlet_density(p[h, allowed], alpha)
        if not math.isfinite(term):
            return -math.inf
        total += term
    return float(total)


def gibbs_pi_p(state: Stage6EState, model: Stage6EModel,
               rng: np.random.Generator) -> dict:
    """The registered conjugate P-then-pi update from the fresh explicit paths."""
    if not model.infer_pi_P:
        raise AssertionError("FULL-LATENT requires infer_pi_P=True")
    counts = transition_counts_of(state.segmentations, model.n_skills)
    state.transition = sample_transition_matrix(counts, model.n_skills, rng,
                                                model.eta_transition)
    state.pi = rng.dirichlet(float(model.eta_initial)
                             + initial_counts(state.segmentations, model.n_skills))
    state.proposed["pi_P"] = state.proposed.get("pi_P", 0) + 1
    state.accepted["pi_P"] = state.accepted.get("pi_P", 0) + 1
    validate_pi_p(state, model)
    return {"transition_counts": counts,
            "initial_counts": initial_counts(state.segmentations, model.n_skills)}


@dataclass
class FullLatentSampler:
    """Reusable exact FFBS tables and both validated U scoring routes for one arm."""

    model: Stage6EModel
    fixed: FullLatentFixed
    config: FullLatentConfig
    _tables: FFBSBlockTables = field(default=None, repr=False)
    _skill: SkillBlockLikelihood = field(default=None, repr=False)
    _collapsed_lik: CollapsedULikelihood = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.model.infer_pi_P:
            raise AssertionError("FULL-LATENT model must infer pi and P")
        self._tables = FFBSBlockTables(model=self.model, source=self.config.table_source)
        self._skill = SkillBlockLikelihood(traces=self.model.traces,
                                           epsilon=self.model.epsilon)
        self._collapsed_lik = CollapsedULikelihood(model=self.model)

    @property
    def tables(self) -> FFBSBlockTables:
        return self._tables

    @property
    def collapsed_likelihood(self) -> CollapsedULikelihood:
        return self._collapsed_lik


def conditional_structural_mh_step(state: Stage6EState, sampler: FullLatentSampler,
                                   rng: np.random.Generator) -> tuple[Stage6EState, dict]:
    """One uniformly selected U row proposal scored against the current explicit path."""
    state = state.copy()
    K, m, _ = np.asarray(state.u_by_skill).shape
    skill, row = int(rng.integers(K)), int(rng.integers(m))
    candidate = propose_row(np.asarray(state.u_by_skill[skill], dtype=float), row,
                            sampler.config.structural_scale, rng)
    state.proposed[_COND_MOVE] = state.proposed.get(_COND_MOVE, 0) + 1
    record = {"skill": skill, "row": row, "accepted": False, "invalid": False,
              "h_changed": False, "log_alpha": None,
              "d_log_lik_conditional": None, "d_log_prior": None}
    old_prior = log_structural_prior(state.u_by_skill[skill], state.rho)
    candidate_prior = log_structural_prior(candidate, state.rho)
    if not math.isfinite(candidate_prior):
        state.invalid[_COND_MOVE] = state.invalid.get(_COND_MOVE, 0) + 1
        record["invalid"] = True
        return state, record

    # The blocks are the currently stored path, and this function is never called for
    # FULL-MARG.  Calling set_blocks here makes that dependency explicit and restart-safe.
    sampler._skill.set_blocks(state.segmentations, sampler.model.n_skills)
    current_ll = sampler._skill.full_replay(
        skill, state.u_by_skill[skill], state.beta, state.omega,
        state.lambda_rep, state.lambda_back)
    candidate_ll = sampler._skill.full_replay(
        skill, candidate, state.beta, state.omega,
        state.lambda_rep, state.lambda_back)
    if not math.isfinite(candidate_ll):
        state.invalid[_COND_MOVE] = state.invalid.get(_COND_MOVE, 0) + 1
        record["invalid"] = True
        return state, record
    d_lik, d_prior = candidate_ll - current_ll, candidate_prior - old_prior
    log_alpha = float(d_lik + d_prior)  # same symmetric Stage-6C row proposal
    h_changed = not np.array_equal(precedence_from_u(candidate),
                                   precedence_from_u(state.u_by_skill[skill]))
    accepted = bool(log_alpha >= 0.0 or math.log(rng.random()) < log_alpha)
    if accepted:
        u = np.array(state.u_by_skill, dtype=float, copy=True)
        u[skill] = candidate
        state.u_by_skill = u
        state.accepted[_COND_MOVE] = state.accepted.get(_COND_MOVE, 0) + 1
    record.update(accepted=accepted, h_changed=bool(h_changed),
                  log_alpha=log_alpha, d_log_lik_conditional=float(d_lik),
                  d_log_prior=float(d_prior))
    return state, record


def _path_prior_components(state: Stage6EState, model: Stage6EModel) -> tuple[float, float]:
    log_pi, log_p = np.log(state.pi), log_transition_matrix(state.transition)
    boundary = path = 0.0
    for trace, segmentation in zip(model.traces, state.segmentations):
        skills = [segment.skill for segment in segmentation.segments]
        boundary += log_boundary_prior(len(trace), len(skills), model.delta_b)
        path += log_path_prior(skills, log_pi, log_p)
    return float(boundary), float(path)


def complete_log_target(state: Stage6EState, model: Stage6EModel,
                        skill_likelihood: SkillBlockLikelihood | None = None) -> dict:
    """Complete-data target decomposition using grouped block replays.

    The returned total is a log density up to the fixed normalizers of the segmentation
    prior and fixed nuisance-coordinate priors.  Those constants cancel in every MCMC
    ratio and are identical across arms.  The pi/P Dirichlet terms are included exactly.
    """
    validate_pi_p(state, model)
    validate_paths(state, model)
    skill_likelihood = (SkillBlockLikelihood(traces=model.traces, epsilon=model.epsilon)
                        if skill_likelihood is None else skill_likelihood)
    skill_likelihood.set_blocks(state.segmentations, model.n_skills)
    blocks = skill_likelihood.total_full_replay(
        state.u_by_skill, state.beta, state.omega, state.lambda_rep,
        state.lambda_back)
    boundary, path = _path_prior_components(state, model)
    structural = float(sum(log_structural_prior(state.u_by_skill[k], state.rho)
                           for k in range(model.n_skills)))
    pi_prior, p_prior = model.log_pi_P_prior(state)
    total = float(blocks + boundary + path + structural + pi_prior + p_prior)
    return {"log_block_likelihood": float(blocks), "log_boundary_prior": boundary,
            "log_path_prior": path, "log_structural_prior": structural,
            "log_pi_prior": float(pi_prior), "log_P_prior": float(p_prior),
            "log_target": total}


def independent_complete_log_target(state: Stage6EState, model: Stage6EModel) -> dict:
    """Independent target recomputation through per-block scorer calls.

    This intentionally does not use ``SkillBlockLikelihood`` and is the numerical
    end-to-end reference for the grouped calculation used in the running chain.
    """
    validate_pi_p(state, model)
    validate_paths(state, model)
    scorer = model.scorer_for(state)
    blocks = 0.0
    for n, segmentation in enumerate(state.segmentations):
        for segment in segmentation.segments:
            blocks += scorer.score(n, segment.start, segment.end, segment.skill)
    boundary, path = _path_prior_components(state, model)
    structural = float(sum(log_structural_prior(state.u_by_skill[k], state.rho)
                           for k in range(model.n_skills)))
    pi_prior, p_prior = model.log_pi_P_prior(state)
    total = float(blocks + boundary + path + structural + pi_prior + p_prior)
    return {"log_block_likelihood": float(blocks), "log_boundary_prior": boundary,
            "log_path_prior": path, "log_structural_prior": structural,
            "log_pi_prior": float(pi_prior), "log_P_prior": float(p_prior),
            "log_target": total}


def full_latent_sweep_once(state: Stage6EState, sampler: FullLatentSampler,
                           rng: np.random.Generator) -> tuple[Stage6EState, dict]:
    """One valid partially-collapsed FULL-LATENT sweep.

    For FULL-MARG the only operation between the marginal U decision and FFBS is local
    bookkeeping: no stored path is read.  Therefore accepted *and rejected* marginal
    attempts are followed by an exact all-trace FFBS refresh before pi/P (or anything
    else) conditions on ``S,z``.  FULL-COND has the matching schedule and differs only
    in its U acceptance score.
    """
    state = state.copy()
    sampler.fixed.assert_unchanged(state)
    validate_pi_p(state, sampler.model)
    validate_paths(state, sampler.model)
    scheduled = is_collapsed_sweep(state.iteration, sampler.config.structural_cadence)
    record = None
    order = []
    if scheduled:
        if sampler.config.arm == FULL_MARG:
            # `collapsed_u_mh_step` uses exactly sum_n log Z_n(U; current pi,P),
            # structural prior, and the symmetric row proposal.  It never reads paths.
            state, record = collapsed_u_mh_step(
                state, sampler.model, sampler.collapsed_likelihood, rng,
                sampler.config.structural_scale)
            order.append("marginal_U")
        else:
            state, record = conditional_structural_mh_step(state, sampler, rng)
            order.append("conditional_U")

    # This is the first path-reading transition after a marginal attempt.
    validate_pi_p(state, sampler.model)
    sampler.tables.refresh(state)
    ffbs = ffbs_segmentation_draw(sampler.model, state, sampler.tables, rng)
    state.segmentations = tuple(segmentation_of(key) for key in ffbs["keys"])
    sampler.tables.mark_stale()
    validate_paths(state, sampler.model)
    order.append("FFBS")

    gibbs = gibbs_pi_p(state, sampler.model, rng)
    order.append("pi_P")
    components = complete_log_target(state, sampler.model, sampler._skill)
    state.components = {
        **components,
        "boundary_hamming_moved": int(ffbs["movement"]["boundary_hamming"]),
        "label_changes_moved": int(ffbs["movement"]["label_changes"]),
        "ffbs_states_changed": int(ffbs["movement"]["states_changed"]),
        "ffbs_log_normalizer_total": float(ffbs["log_normalizers"].sum()),
    }
    state.iteration += 1
    state.rng_state = rng.bit_generator.state
    sampler.fixed.assert_unchanged(state)
    validate_pi_p(state, sampler.model)
    return state, {
        "scheduled_structural": bool(scheduled), "structural_record": record,
        "ffbs": ffbs, "gibbs": gibbs, "kernel_order": tuple(order),
    }


def all_label_permutations(n_skills: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in permutation)
                 for permutation in itertools.permutations(range(int(n_skills))))


def permute_state_labels(state: Stage6EState, permutation) -> Stage6EState:
    """Simultaneously relabel U, pi, rows/columns P, and every z in a state.

    ``permutation[old]`` is the new label of an old label.  This convention makes the
    relabeling explicit and lets tests enumerate all K! symmetry transformations.
    """
    permutation = np.asarray(permutation, dtype=int)
    K = state.n_skills
    if permutation.shape != (K,) or set(permutation.tolist()) != set(range(K)):
        raise ValueError("permutation is not a K-label bijection")
    out = state.copy()
    u = np.empty_like(state.u_by_skill)
    pi = np.empty_like(state.pi)
    p = np.empty_like(state.transition)
    for old in range(K):
        new = int(permutation[old])
        u[new] = state.u_by_skill[old]
        pi[new] = state.pi[old]
        for old_to in range(K):
            p[new, int(permutation[old_to])] = state.transition[old, old_to]
    out.u_by_skill, out.pi, out.transition = u, pi, p
    out.segmentations = tuple(Segmentation(tuple(
        Segment(segment.start, segment.end, int(permutation[segment.skill]))
        for segment in segmentation.segments)) for segmentation in state.segmentations)
    return out


def _stable_rank(corpus_hash: str, kind: str, *parts: int) -> bytes:
    text = "|".join((str(corpus_hash), str(kind), *(str(int(v)) for v in parts)))
    return hashlib.sha256(text.encode("utf-8")).digest()


def select_truth_free_probes(traces, corpus_hash: str, boundary_count: int = 32,
                             coskill_count: int = 64,
                             recovery_coskill_count: int = 256) -> dict:
    """Deterministically select finite truth-free diagnostic/recovery probes."""
    boundary_candidates = [(n, position)
                           for n, trace in enumerate(traces)
                           for position in range(len(trace) - 1)]
    boundary_candidates.sort(key=lambda p: _stable_rank(corpus_hash, "boundary", *p))
    pair_candidates = [(n, left, right)
                       for n, trace in enumerate(traces)
                       for left in range(len(trace))
                       for right in range(left + 1, len(trace))]
    pair_candidates.sort(key=lambda p: _stable_rank(corpus_hash, "coskill", *p))
    recovery_count = min(int(recovery_coskill_count), len(pair_candidates))
    return {
        "boundary": tuple(boundary_candidates[:min(int(boundary_count),
                                                    len(boundary_candidates))]),
        "coskill": tuple(pair_candidates[:min(int(coskill_count),
                                                len(pair_candidates))]),
        "recovery_coskill": tuple(pair_candidates[:recovery_count]),
    }


def _entropy(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    positive = probabilities[probabilities > 0.0]
    return float(-(positive * np.log(positive)).sum())


def _stationary_probabilities(p: np.ndarray) -> np.ndarray:
    """Solve the finite irreducible chain equation deterministically, then sort later."""
    p = np.asarray(p, dtype=float)
    K = p.shape[0]
    system = p.T - np.eye(K)
    system[-1, :] = 1.0
    rhs = np.zeros(K)
    rhs[-1] = 1.0
    stationary = np.linalg.solve(system, rhs)
    if np.any(stationary < -1e-12) or not math.isclose(float(stationary.sum()), 1.0,
                                                       abs_tol=1e-10):
        raise AssertionError("stationary distribution computation failed")
    return stationary / stationary.sum()


def _segment_index_array(segmentation: Segmentation) -> np.ndarray:
    result = np.empty(segmentation.segments[-1].end, dtype=np.int16)
    for index, segment in enumerate(segmentation.segments):
        result[segment.start:segment.end] = index
    return result


def invariant_summaries(state: Stage6EState, model: Stage6EModel, probes: dict,
                        log_target: float | None = None) -> dict:
    """Finite primary convergence summaries, all invariant under global relabeling."""
    validate_pi_p(state, model)
    validate_paths(state, model)
    closures = [precedence_from_u(state.u_by_skill[k]) for k in range(model.n_skills)]
    rel_counts = np.array([int(closure.sum()) for closure in closures], dtype=float)
    segment_lengths = np.array([segment.end - segment.start
                                for seg in state.segmentations
                                for segment in seg.segments], dtype=float)
    labels = state.occurrence_labels()
    segment_indices = [_segment_index_array(segmentation)
                       for segmentation in state.segmentations]
    boundaries = [set(segment.end - 1 for segment in segmentation.segments[:-1])
                  for segmentation in state.segmentations]
    boundary_values = np.array([position in boundaries[n]
                                for n, position in probes["boundary"]], dtype=bool)
    coskill_values = np.array([labels[n][left] == labels[n][right]
                               for n, left, right in probes["coskill"]], dtype=bool)
    same_segment_values = np.array([
        segment_indices[n][left] == segment_indices[n][right]
        for n, left, right in probes["coskill"]], dtype=bool)
    p = np.asarray(state.transition, dtype=float)
    row_entropies = np.sort(np.array([_entropy(p[h, list(allowed_next(h, model.n_skills))])
                                      for h in range(model.n_skills)]))
    stationary = np.sort(_stationary_probabilities(p))
    pi = np.asarray(state.pi, dtype=float)
    if log_target is None:
        log_target = complete_log_target(state, model)["log_target"]
    return {
        "log_target": float(state.components.get("log_target", log_target)),
        "total_relations": float(rel_counts.sum()),
        "sorted_relation_counts": np.sort(rel_counts),
        "total_segments": float(sum(len(seg.segments) for seg in state.segmentations)),
        "mean_segments_per_trace": float(np.mean([len(seg.segments)
                                                   for seg in state.segmentations])),
        "mean_segment_length": float(segment_lengths.mean()),
        "sd_segment_length": float(segment_lengths.std()),
        "boundary_probes": boundary_values,
        "coskill_probes": coskill_values,
        "same_segment_probes": same_segment_values,
        "sorted_pi": np.sort(pi),
        "pi_entropy": _entropy(pi),
        "pi_l2": float(np.linalg.norm(pi)),
        "P_frobenius": float(np.linalg.norm(p)),
        "P_trace2": float(np.trace(p @ p)),
        "P_trace3": float(np.trace(p @ p @ p)),
        "sorted_P_row_entropy": row_entropies,
        "sorted_stationary": stationary,
    }


def _online_path_summaries(state: Stage6EState, probes: dict) -> tuple[list, np.ndarray, np.ndarray]:
    """All-trace boundary sums and preselected label-invariant co-skill indicators."""
    boundary_rows = []
    for segmentation in state.segmentations:
        row = np.zeros(segmentation.segments[-1].end - 1, dtype=float)
        for segment in segmentation.segments[:-1]:
            row[segment.end - 1] = 1.0
        boundary_rows.append(row)
    labels = state.occurrence_labels()
    segments = [_segment_index_array(s) for s in state.segmentations]
    pairs = probes["recovery_coskill"]
    coskill = np.asarray([labels[n][left] == labels[n][right]
                          for n, left, right in pairs], dtype=float)
    same_segment = np.asarray([segments[n][left] == segments[n][right]
                               for n, left, right in pairs], dtype=float)
    return boundary_rows, coskill, same_segment


class FullLatentChain:
    """Resumable FULL-LATENT chain with path-invariant retained diagnostics.

    It checkpoints the complete state, RNG state, retained U/pi/P draws, all diagnostic
    summaries, and online boundary/co-skill accumulators.  It deliberately does *not*
    retain raw paths: recovery uses pre-registered posterior summaries, which avoids
    turning the formal checkpoint ladder into a large truth-accessible path archive.
    """

    _SUMMARY_KEYS = (
        "log_target", "total_relations", "sorted_relation_counts", "total_segments",
        "mean_segments_per_trace", "mean_segment_length", "sd_segment_length",
        "boundary_probes", "coskill_probes", "same_segment_probes", "sorted_pi",
        "pi_entropy", "pi_l2", "P_frobenius", "P_trace2", "P_trace3",
        "sorted_P_row_entropy", "sorted_stationary",
    )

    def __init__(self, sampler: FullLatentSampler, start: Stage6EState, seed: int,
                 burn_in: int, thin: int, probes: dict, start_metadata: dict | None = None):
        if burn_in < 0 or thin < 1:
            raise ValueError("burn_in must be nonnegative and thin must be positive")
        self.sampler = sampler
        self.state = start.copy()
        self.seed, self.burn_in, self.thin = int(seed), int(burn_in), int(thin)
        self.rng = np.random.default_rng(self.seed)
        self.probes = {name: tuple(tuple(int(v) for v in row) for row in values)
                       for name, values in probes.items()}
        self.start_metadata = {} if start_metadata is None else dict(start_metadata)
        self.seconds = 0.0
        self.movement = {"boundary_hamming": 0, "label_changes": 0,
                         "states_changed": 0}
        self.structural = {"attempts": 0, "accepts": 0, "h_accepts": 0,
                           "invalid": 0, "marginal_attempts": 0,
                           "ffbs_after_marginal": 0}
        self.retained = {name: [] for name in self._SUMMARY_KEYS}
        self.u_draws: list = []
        self.pi_draws: list = []
        self.p_draws: list = []
        self.relation_indicators: list = []
        self.boundary_sums = [np.zeros(len(trace) - 1, dtype=float)
                              for trace in sampler.model.traces]
        self.recovery_coskill_sums = np.zeros(len(self.probes["recovery_coskill"]),
                                              dtype=float)
        self.recovery_same_segment_sums = np.zeros_like(self.recovery_coskill_sums)
        self.retained_draws = 0

    # ---------------------------------------------------------------------- diagnostics
    def _retain(self) -> None:
        summary = invariant_summaries(self.state, self.sampler.model, self.probes)
        for name in self._SUMMARY_KEYS:
            value = summary[name]
            self.retained[name].append(np.array(value, copy=True)
                                       if isinstance(value, np.ndarray) else float(value))
        # U determines the discrete closure.  Float32 can merge near-tied coordinates and
        # therefore alter H during terminal recovery, so retained draws must preserve the
        # sampler's float64 state exactly.
        self.u_draws.append(np.asarray(self.state.u_by_skill, dtype=np.float64).copy())
        self.pi_draws.append(np.asarray(self.state.pi, dtype=np.float64).copy())
        self.p_draws.append(np.asarray(self.state.transition, dtype=np.float64).copy())
        self.relation_indicators.append(relation_indicator_vector(self.state.u_by_skill))
        boundary_rows, coskill, same_segment = _online_path_summaries(self.state,
                                                                        self.probes)
        for total, row in zip(self.boundary_sums, boundary_rows):
            total += row
        self.recovery_coskill_sums += coskill
        self.recovery_same_segment_sums += same_segment
        self.retained_draws += 1

    def arrays(self) -> dict:
        return {name: np.asarray(values) for name, values in self.retained.items()}

    # -------------------------------------------------------------------------- running
    def advance(self, upto: int, checkpoint_path=None, checkpoint_every: int = 0,
                progress_every: int = 0) -> None:
        began = last_mark = time.perf_counter()
        while self.state.iteration < int(upto):
            state, info = full_latent_sweep_once(self.state, self.sampler, self.rng)
            record = info["structural_record"]
            if record is not None:
                self.structural["attempts"] += 1
                self.structural["accepts"] += int(record["accepted"])
                self.structural["h_accepts"] += int(record["accepted"]
                                                   and record["h_changed"])
                self.structural["invalid"] += int(record["invalid"])
                if self.sampler.config.arm == FULL_MARG:
                    self.structural["marginal_attempts"] += 1
            self.movement["boundary_hamming"] += int(info["ffbs"]["movement"]
                                                       ["boundary_hamming"])
            self.movement["label_changes"] += int(info["ffbs"]["movement"]
                                                    ["label_changes"])
            self.movement["states_changed"] += int(info["ffbs"]["movement"]
                                                     ["states_changed"])
            if (self.sampler.config.arm == FULL_MARG
                    and info["scheduled_structural"]):
                if info["kernel_order"] != ("marginal_U", "FFBS", "pi_P"):
                    raise AssertionError("marginal attempt was not immediately refreshed")
                self.structural["ffbs_after_marginal"] += 1
            self.state = state
            sweep = self.state.iteration
            if sweep > self.burn_in and (sweep - self.burn_in) % self.thin == 0:
                self._retain()
            now = time.perf_counter()
            if checkpoint_path and checkpoint_every and sweep % int(checkpoint_every) == 0:
                self.seconds += now - last_mark
                last_mark = now
                self.save(checkpoint_path)
            if progress_every and sweep % int(progress_every) == 0:
                print(f"      {self.sampler.config.arm} seed {self.seed}: sweep {sweep:,} "
                      f"({time.perf_counter() - began:.0f}s segment)", flush=True)
        self.seconds += time.perf_counter() - last_mark
        if self.sampler.config.arm == FULL_MARG and (
                self.structural["marginal_attempts"]
                != self.structural["ffbs_after_marginal"]):
            raise AssertionError("marginal U/FFBS ordering counter mismatch")
        self.state.rng_state = self.rng.bit_generator.state
        if checkpoint_path:
            self.save(checkpoint_path)

    # ---------------------------------------------------------------------- persistence
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.state.rng_state = self.rng.bit_generator.state
        meta = {
            "schema": 2, "model_id": FULL_LATENT_MODEL_ID,
            "seed": self.seed, "burn_in": self.burn_in, "thin": self.thin,
            "seconds": self.seconds, "movement": self.movement,
            "structural": self.structural, "retained_draws": self.retained_draws,
            "config": self.sampler.config.as_dict(), "fixed": self.sampler.fixed.as_dict(),
            "probes": {name: [list(row) for row in values]
                       for name, values in self.probes.items()},
            "start_metadata": self.start_metadata,
            "state": self.state.to_dict(),
        }
        payload = {
            "meta": np.array(json.dumps(meta, sort_keys=True)),
            "u_draws": np.asarray(self.u_draws, dtype=np.float64),
            "pi_draws": np.asarray(self.pi_draws, dtype=np.float64),
            "p_draws": np.asarray(self.p_draws, dtype=np.float64),
            "relation_indicators": np.asarray(self.relation_indicators, dtype=bool),
            "recovery_coskill_sums": np.asarray(self.recovery_coskill_sums, dtype=float),
            "recovery_same_segment_sums": np.asarray(self.recovery_same_segment_sums,
                                                      dtype=float),
        }
        for name, values in self.retained.items():
            payload[f"summary__{name}"] = np.asarray(values)
        for index, values in enumerate(self.boundary_sums):
            payload[f"boundary__{index:03d}"] = np.asarray(values, dtype=float)
        temporary = str(path) + ".tmp.npz"
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path, sampler: FullLatentSampler) -> "FullLatentChain":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("schema") != 2 or meta.get("model_id") != FULL_LATENT_MODEL_ID:
                raise ValueError(f"not a {FULL_LATENT_MODEL_ID} checkpoint: {path}")
            if meta["config"] != sampler.config.as_dict():
                raise ValueError("checkpoint config does not match sampler config")
            if meta["fixed"] != sampler.fixed.as_dict():
                raise ValueError("checkpoint fixed coordinates do not match sampler")
            chain = cls.__new__(cls)
            chain.sampler = sampler
            chain.state = Stage6EState.from_dict(meta["state"])
            # Stage6EState.from_dict intentionally omits components because most older
            # chains recompute them on the next sweep.  Here they are also a retained
            # diagnostic, so restore the serialised scalar component block explicitly.
            chain.state.components = dict(meta["state"].get("components", {}))
            chain.seed, chain.burn_in, chain.thin = (int(meta["seed"]),
                                                      int(meta["burn_in"]),
                                                      int(meta["thin"]))
            chain.rng = np.random.default_rng()
            chain.rng.bit_generator.state = chain.state.rng_state
            chain.probes = {name: tuple(tuple(int(v) for v in row) for row in values)
                            for name, values in meta["probes"].items()}
            chain.start_metadata = dict(meta.get("start_metadata", {}))
            chain.seconds = float(meta["seconds"])
            chain.movement = {key: int(value) for key, value in meta["movement"].items()}
            chain.structural = {key: int(value)
                                for key, value in meta["structural"].items()}
            chain.retained_draws = int(meta["retained_draws"])
            chain.retained = {name: list(np.asarray(data[f"summary__{name}"]))
                              for name in cls._SUMMARY_KEYS}
            stored_u_draws = np.asarray(data["u_draws"])
            if stored_u_draws.dtype != np.dtype(np.float64):
                raise ValueError("schema-2 FULL-LATENT checkpoints require exact float64 "
                                 "retained U draws")
            chain.u_draws = list(stored_u_draws)
            chain.pi_draws = list(np.asarray(data["pi_draws"], dtype=np.float64))
            chain.p_draws = list(np.asarray(data["p_draws"], dtype=np.float64))
            chain.relation_indicators = list(np.asarray(data["relation_indicators"],
                                                        dtype=bool))
            chain.boundary_sums = [np.asarray(data[f"boundary__{index:03d}"],
                                              dtype=float)
                                   for index in range(len(sampler.model.traces))]
            chain.recovery_coskill_sums = np.asarray(data["recovery_coskill_sums"],
                                                      dtype=float)
            chain.recovery_same_segment_sums = np.asarray(
                data["recovery_same_segment_sums"], dtype=float)
        sampler.fixed.assert_unchanged(chain.state)
        validate_pi_p(chain.state, sampler.model)
        validate_paths(chain.state, sampler.model)
        if chain.sampler.config.arm == FULL_MARG and (
                chain.structural["marginal_attempts"]
                != chain.structural["ffbs_after_marginal"]):
            raise AssertionError("checkpoint has a marginal U/FFBS ordering mismatch")
        return chain
