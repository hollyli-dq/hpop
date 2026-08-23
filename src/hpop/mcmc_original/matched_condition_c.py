"""Condition C — joint (S, z, U) inference on the formal matched corpus.

Target:

    p(S, z, U | X, vartheta*, pi*, P*, delta_B*, epsilon*, rho_0)

with rho FIXED at rho_0 = 0.5 (the Condition-B registered correction carries
over: the supplied-truth corpus records rho* = null) and the four recurrent
scalars, pi and P fixed to generating truth. Only (S, z) and U move.

## Composition — validated pieces called, never reimplemented

Each sweep is the partially-collapsed composition the Step 8 kernel validated:

    [scheduled collapsed U MH]  ->  exact FFBS refresh of ALL (S, z)
                                ->  conditional U row sweep

* the scheduled move IS `collapsed_u_kernel.collapsed_u_mh_step`, verbatim,
  scored by `collapsed_u_likelihood.CollapsedULikelihood` (Step 7A forward
  normalizers over `FastBlockScoreTable`), at cadence `every` on absolute
  sweep indices (`is_collapsed_sweep`);
* the (S, z) refresh IS `recurrent_joint_ffbs_mcmc.ffbs_segmentation_draw`
  over `FFBSBlockTables` — an exact conditional Gibbs draw, acceptance 1,
  performed IMMEDIATELY after any collapsed move, which is the ordering the
  partially-collapsed invariance argument requires;
* the conditional U phase is the Stage 6E sweep's phase 3 — the registered
  `sampler_u.propose_row` symmetric row walk accepted on
  `log_structural_prior` + `SkillBlockLikelihood.full_replay` (complete
  replays from q_0 = 0) — with the SAME arithmetic, pinned by test.

The two Condition-C arms share this one sweep:

    C-COND:  every = 0   (no collapsed move — conditional-only)
    C-MARG:  every = c   (the partially-collapsed kernel)

Both are invariant for the SAME posterior, which is what the registered
small-reference equality test checks.

## What is fixed, and how

`pi`, `P`, `rho`, `beta`, `omega`, `lambda_rep`, `lambda_back` live in the
`Stage6EState` but no phase of this sweep proposes them; `run_condition_c_chain`
asserts after every sweep that each is bit-identical to its frozen value. The
Stage 6E pi/P Gibbs phase, rho MH phase and scalar MH phases are simply never
called — this module contains no code path that could move them.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from hpop.mcmc_original.collapsed_u_kernel import (
    CollapsedUConfig, collapsed_u_mh_step, is_collapsed_sweep,
)
from hpop.mcmc_original.collapsed_u_likelihood import CollapsedULikelihood
from hpop.mcmc_original.fast_segmentation_kernel import key_of, segmentation_of
from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.matched_condition_b import (
    canonical_h_hash, relation_indicator_vector,
)
from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import (
    FFBSBlockTables, ffbs_segmentation_draw,
)
from hpop.mcmc_original.sampler_u import propose_row
from hpop.mcmc_original.stage6c_frozen import log_structural_prior
from hpop.mcmc_original.stage6e_sampler import SkillBlockLikelihood
from hpop.mcmc_original.stage6e_state import Stage6EModel, Stage6EState
from hpop.mcmc_original.targets import log_boundary_prior, log_path_prior
from hpop.mcmc_original.transitions import log_transition_matrix
from hpop.mcmc_original.types import Segment, Segmentation

__all__ = [
    "ConditionCFixed", "build_condition_c_model", "initial_condition_c_state",
    "ConditionCSampler", "condition_c_sweep_once", "run_condition_c_chain",
    "ConditionCChain",
]


@dataclass(frozen=True)
class ConditionCFixed:
    """The frozen non-(S,z,U) coordinates, asserted bit-identical every sweep."""

    rho_0: float
    beta: float
    omega: float
    lambda_rep: float
    lambda_back: float
    pi: tuple
    transition: tuple

    @classmethod
    def from_truth(cls, truth, rho_0: float) -> "ConditionCFixed":
        return cls(rho_0=float(rho_0), beta=float(truth.beta),
                   omega=float(truth.omega),
                   lambda_rep=float(truth.lambda_rep),
                   lambda_back=float(truth.lambda_back),
                   pi=tuple(float(v) for v in truth.pi),
                   transition=tuple(tuple(float(v) for v in row)
                                    for row in truth.transition))

    def assert_unchanged(self, state: Stage6EState) -> None:
        if (float(state.rho) != self.rho_0
                or float(state.beta) != self.beta
                or float(state.omega) != self.omega
                or float(state.lambda_rep) != self.lambda_rep
                or float(state.lambda_back) != self.lambda_back
                or tuple(float(v) for v in state.pi) != self.pi
                or tuple(tuple(float(v) for v in row)
                         for row in state.transition) != self.transition):
            raise AssertionError("a Condition-C fixed coordinate moved")


def build_condition_c_model(traces) -> Stage6EModel:
    """The Stage 6E model over the given observed traces with pi/P inference OFF."""
    return Stage6EModel(traces=tuple(tuple(int(v) for v in t) for t in traces),
                        infer_pi_P=False)


def initial_condition_c_state(model: Stage6EModel, u_start: np.ndarray,
                              fixed: ConditionCFixed) -> Stage6EState:
    """A legal deterministic starting state; the first FFBS draw replaces (S, z)
    from its exact conditional, so the initial segmentation carries no influence."""
    segmentations = []
    for trace in model.traces:
        J = len(trace)
        if J < model.min_width:
            raise ValueError(f"trace of length {J} below the minimum width")
        widths, remaining = [], J
        while remaining > model.max_width:
            step = (model.max_width
                    if remaining - model.max_width >= model.min_width
                    else remaining - model.min_width)
            widths.append(step)
            remaining -= step
        widths.append(remaining)
        if sum(widths) != J or any(not model.min_width <= w <= model.max_width
                                   for w in widths):
            raise ValueError(f"no deterministic initial tiling for J={J}")
        segments, start = [], 0
        for i, w in enumerate(widths):
            segments.append(Segment(start, start + w, i % 2))
            start += w
        segmentations.append(Segmentation(tuple(segments)))
    return Stage6EState(
        segmentations=tuple(segmentations),
        u_by_skill=np.array(u_start, dtype=float, copy=True),
        rho=fixed.rho_0, beta=fixed.beta, omega=fixed.omega,
        lambda_rep=fixed.lambda_rep, lambda_back=fixed.lambda_back,
        pi=np.array(fixed.pi, dtype=float),
        transition=np.array(fixed.transition, dtype=float))


@dataclass
class ConditionCSampler:
    """Everything one Condition-C chain reuses across sweeps."""

    model: Stage6EModel
    fixed: ConditionCFixed
    u_scale: float
    collapsed: CollapsedUConfig
    table_source: str = "batched"
    _tables: FFBSBlockTables = field(default=None, repr=False)
    _skill: SkillBlockLikelihood = field(default=None, repr=False)
    _collapsed_lik: CollapsedULikelihood = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._tables = FFBSBlockTables(model=self.model,
                                       source=self.table_source)
        self._skill = SkillBlockLikelihood(traces=self.model.traces,
                                           epsilon=self.model.epsilon)
        self._collapsed_lik = CollapsedULikelihood(model=self.model)

    @property
    def collapsed_likelihood(self) -> CollapsedULikelihood:
        return self._collapsed_lik


def _conditional_log_target(state: Stage6EState, model: Stage6EModel,
                            block_ll_by_skill: dict) -> float:
    """Complete-data log target at fixed phases: boundary + path priors + block
    likelihoods + p(U | rho_0). Constant terms in the fixed coordinates are
    included where cheap and harmless (they cancel in every comparison)."""
    log_pi = np.log(state.pi)
    log_p = log_transition_matrix(state.transition)
    total = 0.0
    for segmentation in state.segmentations:
        path = [seg.skill for seg in segmentation.segments]
        length = segmentation.segments[-1].end
        total += log_boundary_prior(length, len(path), model.delta_b)
        total += log_path_prior(path, log_pi, log_p)
    total += float(sum(block_ll_by_skill.values()))
    total += float(sum(log_structural_prior(state.u_by_skill[k], state.rho)
                       for k in range(model.n_skills)))
    return total


def condition_c_sweep_once(state: Stage6EState, sampler: ConditionCSampler,
                           rng) -> tuple[Stage6EState, dict | None, dict]:
    """One Condition-C sweep. Returns (new state, collapsed record or None,
    sweep info). C-COND is `sampler.collapsed.every == 0`; C-MARG schedules the
    verbatim collapsed move and the FFBS refresh runs immediately after it."""
    record = None
    if is_collapsed_sweep(state.iteration, sampler.collapsed.every):
        state, record = collapsed_u_mh_step(
            state, sampler.model, sampler.collapsed_likelihood, rng,
            sampler.collapsed.scale)
    else:
        state = state.copy()

    # exact (S, z) refresh from its full conditional at the (possibly moved) U
    sampler._tables.refresh(state)
    ffbs = ffbs_segmentation_draw(sampler.model, state, sampler._tables, rng)
    state.segmentations = tuple(segmentation_of(k) for k in ffbs["keys"])
    sampler._tables.mark_stale()

    # conditional U rows on the fresh blocks — Stage 6E phase 3 arithmetic
    skill_ll = sampler._skill
    skill_ll.set_blocks(state.segmentations, sampler.model.n_skills)
    u = np.array(state.u_by_skill, dtype=float, copy=True)
    current_ll = {k: skill_ll.full_replay(k, u[k], state.beta, state.omega,
                                          state.lambda_rep, state.lambda_back)
                  for k in range(sampler.model.n_skills)}
    current_structural = {k: log_structural_prior(u[k], state.rho)
                          for k in range(sampler.model.n_skills)}
    proposed = accepted = h_accepted = 0
    for k in range(sampler.model.n_skills):
        for row in range(u.shape[1]):
            candidate = propose_row(u[k], row, sampler.u_scale, rng)
            proposed += 1
            candidate_structural = log_structural_prior(candidate, state.rho)
            if not math.isfinite(candidate_structural):
                continue
            candidate_ll = skill_ll.full_replay(
                k, candidate, state.beta, state.omega, state.lambda_rep,
                state.lambda_back)
            if not math.isfinite(candidate_ll):
                continue
            log_alpha = ((candidate_ll - current_ll[k])
                         + (candidate_structural - current_structural[k]))
            if log_alpha >= 0.0 or math.log(rng.random()) < log_alpha:
                h_accepted += int(not np.array_equal(
                    precedence_from_u(candidate), precedence_from_u(u[k])))
                u[k] = candidate
                current_ll[k] = candidate_ll
                current_structural[k] = candidate_structural
                accepted += 1
    state.u_by_skill = u
    state.iteration += 1

    info = {"ffbs_movement": ffbs["movement"],
            "ffbs_log_normalizer_total": float(ffbs["log_normalizers"].sum()),
            "u_proposed": proposed, "u_accepted": accepted,
            "u_h_accepted": h_accepted,
            "log_target": _conditional_log_target(state, sampler.model,
                                                  current_ll)}
    return state, record, info


def run_condition_c_chain(model: Stage6EModel, fixed: ConditionCFixed,
                          u_start: np.ndarray, u_scale: float,
                          collapsed: CollapsedUConfig, num_sweeps: int,
                          burn_in: int, thin: int, seed: int,
                          store_keys: bool = False,
                          progress_every: int = 0) -> dict:
    """One Condition-C chain with fixed-coordinate assertions every sweep."""
    if burn_in >= num_sweeps:
        raise ValueError("burn_in must be smaller than num_sweeps")
    sampler = ConditionCSampler(model=model, fixed=fixed, u_scale=float(u_scale),
                                collapsed=collapsed)
    state = initial_condition_c_state(model, u_start, fixed)
    rng = np.random.default_rng(seed)

    retained = {"log_target": [], "relation_indicators": [], "h_hashes": [],
                "relation_counts": [], "keys": [], "segment_counts": []}
    movement = {"boundary_hamming": 0, "label_changes": 0, "states_changed": 0,
                "u_proposed": 0, "u_accepted": 0, "u_h_accepted": 0}
    collapsed_records: list = []
    esjd_num = 0.0
    began = time.perf_counter()
    for i in range(int(num_sweeps)):
        u_before = np.array(state.u_by_skill, dtype=float, copy=True)
        state, record, info = condition_c_sweep_once(state, sampler, rng)
        fixed.assert_unchanged(state)
        if record is not None:
            record["sweep"] = i
            collapsed_records.append(record)
        for key in ("boundary_hamming", "label_changes", "states_changed"):
            movement[key] += info["ffbs_movement"][key]
        for key in ("u_proposed", "u_accepted", "u_h_accepted"):
            movement[key] += info[key]
        jump = state.u_by_skill - u_before
        esjd_num += float((jump * jump).sum())
        if i >= burn_in and (i - burn_in) % thin == 0:
            retained["log_target"].append(info["log_target"])
            retained["relation_indicators"].append(
                relation_indicator_vector(state.u_by_skill))
            retained["h_hashes"].append(tuple(
                canonical_h_hash(precedence_from_u(state.u_by_skill[k]))
                for k in range(model.n_skills)))
            retained["relation_counts"].append(int(sum(
                precedence_from_u(state.u_by_skill[k]).sum()
                for k in range(model.n_skills))))
            retained["segment_counts"].append(
                [len(s.segments) for s in state.segmentations])
            if store_keys:
                retained["keys"].append(tuple(key_of(s)
                                              for s in state.segmentations))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    sweep {i + 1:,}/{num_sweeps:,} "
                  f"({time.perf_counter() - began:.0f}s)", flush=True)
    seconds = time.perf_counter() - began
    n_collapsed = len(collapsed_records)
    return {
        "retained": retained, "movement": movement,
        "collapsed_records": collapsed_records,
        "collapsed_proposed": n_collapsed,
        "collapsed_accepted": sum(r["accepted"] for r in collapsed_records),
        "collapsed_h_accepted": sum(r["accepted"] and r["h_changed"]
                                    for r in collapsed_records),
        "esjd_per_sweep": esjd_num / num_sweeps,
        "seconds": seconds, "sweeps": int(num_sweeps),
        "seconds_per_sweep": seconds / num_sweeps,
        "final_state": state, "seed": int(seed),
        "collapsed_config": {"every": int(collapsed.every),
                             "scale": float(collapsed.scale)},
        "u_scale": float(u_scale),
    }


# ===================================================================== formal chain
class ConditionCChain:
    """A resumable Condition-C chain for the formal run.

    The sweep is `condition_c_sweep_once` unchanged; this class adds retained-
    draw storage, ONLINE sufficient statistics for the path marginals (per-trace
    boundary-cut counts and occurrence-label occupancy counts over retained
    draws — no per-draw key storage), and an exact checkpoint/resume: the RNG
    bit-generator state, counters and all retained arrays round-trip through a
    single `.npz`, and a resumed chain is draw-identical to an uninterrupted
    one (pinned by test).
    """

    def __init__(self, sampler: ConditionCSampler, u_start: np.ndarray,
                 seed: int, burn_in: int, thin: int) -> None:
        self.sampler = sampler
        self.burn_in, self.thin = int(burn_in), int(thin)
        self.seed = int(seed)
        self.state = initial_condition_c_state(sampler.model,
                                               np.asarray(u_start, dtype=float),
                                               sampler.fixed)
        self.rng = np.random.default_rng(seed)
        self.movement = {"boundary_hamming": 0, "label_changes": 0,
                         "states_changed": 0, "u_proposed": 0, "u_accepted": 0,
                         "u_h_accepted": 0}
        self.collapsed_proposed = 0
        self.collapsed_accepted = 0
        self.collapsed_h_accepted = 0
        self.esjd_sum = 0.0
        self.seconds = 0.0
        self.retained_log_target: list = []
        self.retained_log_prior: list = []
        self.retained_indicators: list = []
        self.retained_rel_counts: list = []
        self.retained_h_hashes: list = []
        n_traces = len(sampler.model.traces)
        self.boundary_sums = [np.zeros(len(t) - 1) for t in sampler.model.traces]
        self.occupancy_sums = [np.zeros((len(t), sampler.model.n_skills))
                               for t in sampler.model.traces]
        self.marginal_draws = 0

    # ------------------------------------------------------------------ running
    def advance(self, upto: int, checkpoint_path=None,
                checkpoint_every: int = 0, progress_every: int = 0) -> None:
        began = time.perf_counter()
        last_mark = began
        while self.state.iteration < int(upto):
            state, record, info = condition_c_sweep_once(self.state,
                                                         self.sampler, self.rng)
            self.sampler.fixed.assert_unchanged(state)
            u_jump = state.u_by_skill - self.state.u_by_skill
            self.esjd_sum += float((u_jump * u_jump).sum())
            self.state = state
            if record is not None:
                self.collapsed_proposed += 1
                self.collapsed_accepted += int(record["accepted"])
                self.collapsed_h_accepted += int(record["accepted"]
                                                 and record["h_changed"])
            for key in ("boundary_hamming", "label_changes", "states_changed"):
                self.movement[key] += info["ffbs_movement"][key]
            for key in ("u_proposed", "u_accepted", "u_h_accepted"):
                self.movement[key] += info[key]
            sweep = self.state.iteration          # 1-based after the sweep
            if sweep > self.burn_in and (sweep - self.burn_in) % self.thin == 0:
                self.retained_log_target.append(info["log_target"])
                self.retained_log_prior.append(float(sum(
                    log_structural_prior(self.state.u_by_skill[k],
                                         self.state.rho)
                    for k in range(self.sampler.model.n_skills))))
                self.retained_indicators.append(
                    relation_indicator_vector(self.state.u_by_skill))
                self.retained_rel_counts.append(int(sum(
                    precedence_from_u(self.state.u_by_skill[k]).sum()
                    for k in range(self.sampler.model.n_skills))))
                self.retained_h_hashes.append(tuple(
                    canonical_h_hash(precedence_from_u(self.state.u_by_skill[k]))
                    for k in range(self.sampler.model.n_skills)))
                self.marginal_draws += 1
                for n, segmentation in enumerate(self.state.segmentations):
                    occupancy = self.occupancy_sums[n]
                    boundary = self.boundary_sums[n]
                    for seg in segmentation.segments:
                        occupancy[seg.start:seg.end, seg.skill] += 1.0
                        if seg.end < occupancy.shape[0]:
                            boundary[seg.end - 1] += 1.0
            now = time.perf_counter()
            if checkpoint_path and checkpoint_every and \
                    self.state.iteration % int(checkpoint_every) == 0:
                self.seconds += now - last_mark
                last_mark = now
                self.save(checkpoint_path)
            if progress_every and self.state.iteration % progress_every == 0:
                print(f"      chain seed {self.seed}: sweep "
                      f"{self.state.iteration:,} "
                      f"({time.perf_counter() - began:.0f}s segment)",
                      flush=True)
        self.seconds += time.perf_counter() - last_mark
        if checkpoint_path:
            self.save(checkpoint_path)

    # -------------------------------------------------------------- persistence
    def save(self, path) -> None:
        import json as _json
        meta = {
            "seed": self.seed, "burn_in": self.burn_in, "thin": self.thin,
            "iteration": int(self.state.iteration),
            "movement": self.movement,
            "collapsed": [self.collapsed_proposed, self.collapsed_accepted,
                          self.collapsed_h_accepted],
            "esjd_sum": self.esjd_sum, "seconds": self.seconds,
            "marginal_draws": self.marginal_draws,
            "rng_state": self.rng.bit_generator.state,
            "u_scale": self.sampler.u_scale,
            "collapsed_config": {"every": self.sampler.collapsed.every,
                                 "scale": self.sampler.collapsed.scale},
            "segmentations": [[[s.start, s.end, s.skill]
                               for s in seg.segments]
                              for seg in self.state.segmentations],
        }
        payload = {
            "meta": np.array(_json.dumps(meta)),
            "u_by_skill": np.asarray(self.state.u_by_skill, dtype=float),
            "log_target": np.asarray(self.retained_log_target, dtype=float),
            "log_prior": np.asarray(self.retained_log_prior, dtype=float),
            "indicators": np.asarray(self.retained_indicators, dtype=bool),
            "rel_counts": np.asarray(self.retained_rel_counts, dtype=np.int16),
            "h_hashes": np.asarray([list(h) for h in self.retained_h_hashes],
                                   dtype="U16")
            if self.retained_h_hashes else np.zeros((0, 3), dtype="U16"),
        }
        for n in range(len(self.boundary_sums)):
            payload[f"b{n:03d}"] = self.boundary_sums[n]
            payload[f"o{n:03d}"] = self.occupancy_sums[n]
        tmp = str(path) + ".tmp.npz"
        np.savez_compressed(tmp, **payload)
        import os
        os.replace(tmp, str(path))

    @classmethod
    def load(cls, path, sampler: ConditionCSampler) -> "ConditionCChain":
        import json as _json
        data = np.load(str(path))
        meta = _json.loads(str(data["meta"]))
        chain = cls.__new__(cls)
        chain.sampler = sampler
        chain.seed = int(meta["seed"])
        chain.burn_in, chain.thin = int(meta["burn_in"]), int(meta["thin"])
        chain.state = Stage6EState(
            segmentations=tuple(
                Segmentation(tuple(Segment(int(a), int(b), int(k))
                                   for a, b, k in seg))
                for seg in meta["segmentations"]),
            u_by_skill=np.asarray(data["u_by_skill"], dtype=float),
            rho=sampler.fixed.rho_0, beta=sampler.fixed.beta,
            omega=sampler.fixed.omega, lambda_rep=sampler.fixed.lambda_rep,
            lambda_back=sampler.fixed.lambda_back,
            pi=np.array(sampler.fixed.pi, dtype=float),
            transition=np.array(sampler.fixed.transition, dtype=float),
            iteration=int(meta["iteration"]))
        chain.rng = np.random.default_rng()
        chain.rng.bit_generator.state = meta["rng_state"]
        chain.movement = {k: int(v) for k, v in meta["movement"].items()}
        (chain.collapsed_proposed, chain.collapsed_accepted,
         chain.collapsed_h_accepted) = [int(v) for v in meta["collapsed"]]
        chain.esjd_sum = float(meta["esjd_sum"])
        chain.seconds = float(meta["seconds"])
        chain.marginal_draws = int(meta["marginal_draws"])
        chain.retained_log_target = [float(v) for v in data["log_target"]]
        chain.retained_log_prior = [float(v) for v in data["log_prior"]]
        chain.retained_indicators = [np.asarray(row, dtype=bool)
                                     for row in data["indicators"]]
        chain.retained_rel_counts = [int(v) for v in data["rel_counts"]]
        chain.retained_h_hashes = [tuple(str(v) for v in row)
                                   for row in data["h_hashes"]]
        chain.boundary_sums = [np.asarray(data[f"b{n:03d}"], dtype=float)
                               for n in range(len(sampler.model.traces))]
        chain.occupancy_sums = [np.asarray(data[f"o{n:03d}"], dtype=float)
                                for n in range(len(sampler.model.traces))]
        return chain
