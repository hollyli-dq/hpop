"""The collapsed U likelihood — `ell_coll(U) = sum_n log Z_n(U)` with `(S, z)` summed out.

This module owns no new mathematics. Each `log Z_n` is the Step 7A semi-Markov forward
normaliser (`semi_markov_ffbs.forward`, validated against exact enumeration) over the
candidate block tables of `FastBlockScoreTable` (validated entry-for-entry against the
registered per-block scorer, every candidate from `q_0 = 0`). It is the same quantity the
C0/C1 collapsed-U audits scored — `test_collapsed_u_kernel.py` pins this implementation
against the audit scripts' scorer directly — packaged with the two things a *sampler*
needs that a one-shot audit did not:

* a **current-state cache** with exact invalidation, so the expensive baseline
  `ell_coll(U)` is not recomputed when nothing it depends on has moved; and
* a **candidate evaluation** that rebuilds only the proposed skill's score column
  (`FastBlockScoreTable.refresh` is fingerprint-keyed per skill) and then reruns the
  forward recursion on the complete updated table. The normaliser is never reused after
  a score-column change; only the untouched columns are.

## What the collapsed likelihood depends on

    U (through h(U) only)   beta   omega   lambda_rep   lambda_back
    pi   P   delta_B        epsilon   the traces   the legal width support

The cache fingerprint contains every varying item of that list byte-for-byte; the trace
set, `epsilon`, `delta_B` and the width bounds are fixed at construction (they are
`Stage6EModel` constants and the table layout is built from them), so a
`CollapsedULikelihood` must never be shared between models. Any fingerprint mismatch
rebuilds; there is no partial trust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from hpop.mcmc_original.fast_block_tables import FastBlockScoreTable
from hpop.mcmc_original.semi_markov_ffbs import forward
from hpop.mcmc_original.stage6e_state import Stage6EModel, Stage6EState
from hpop.mcmc_original.transitions import log_transition_matrix

__all__ = ["CollapsedULikelihood"]


@dataclass
class CollapsedULikelihood:
    """`ell_coll(U)` and its single-skill candidate deltas, over one fixed model."""

    model: Stage6EModel
    _table: FastBlockScoreTable = field(default=None, repr=False)
    _cache_key: tuple | None = field(default=None, repr=False)
    _cached_log_z: np.ndarray | None = field(default=None, repr=False)
    evaluations: int = 0            # full forward passes (baseline + candidates)
    cache_hits: int = 0

    def __post_init__(self) -> None:
        self._table = FastBlockScoreTable(
            traces=self.model.traces, epsilon=self.model.epsilon,
            n_skills=self.model.n_skills, min_width=self.model.min_width,
            max_width=self.model.max_width, n_roles=self.model.n_roles)

    # ------------------------------------------------------------------ fingerprinting
    def _fingerprint(self, state: Stage6EState) -> tuple:
        """Everything the collapsed likelihood reads that can move between sweeps."""
        return (np.asarray(state.u_by_skill, dtype=float).tobytes(),
                float(state.beta), float(state.omega),
                float(state.lambda_rep), float(state.lambda_back),
                np.asarray(state.pi, dtype=float).tobytes(),
                np.asarray(state.transition, dtype=float).tobytes())

    # ------------------------------------------------------------------ the normaliser
    def _forward_all(self, state: Stage6EState) -> np.ndarray:
        log_pi = np.log(np.asarray(state.pi, dtype=float))
        log_p = log_transition_matrix(state.transition)
        self.evaluations += 1
        return np.array([
            forward(table, log_pi, log_p, self.model.delta_b, self.model.max_width,
                    self.model.min_width).log_normalizer
            for table in self._table.tables], dtype=float)

    def log_z_per_trace(self, state: Stage6EState) -> np.ndarray:
        """`log Z_n(U)` for every trace at the state's parameters, cached exactly."""
        key = self._fingerprint(state)
        if key == self._cache_key:
            self.cache_hits += 1
            return self._cached_log_z
        self._table.refresh(state.u_by_skill, state.beta, state.omega,
                            state.lambda_rep, state.lambda_back)
        self._cached_log_z = self._forward_all(state)
        self._cache_key = key
        return self._cached_log_z

    def log_likelihood(self, state: Stage6EState) -> float:
        return float(self.log_z_per_trace(state).sum())

    # ------------------------------------------------------------------ candidate delta
    def delta_for_candidate(self, state: Stage6EState, skill: int,
                            candidate_row_matrix: np.ndarray) -> tuple[float, np.ndarray]:
        """`Delta ell_coll` for replacing `U_skill` with `candidate_row_matrix`.

        Rebuilds only skill `skill`'s score column, reruns the forward recursion on the
        complete table, and then restores the column to the current `U` — so on return
        the table and the cache once again describe the incoming state, whatever the
        caller decides about the proposal.
        """
        base = self.log_z_per_trace(state)          # also leaves the table at current U
        u_prime = np.array(state.u_by_skill, dtype=float, copy=True)
        u_prime[int(skill)] = np.asarray(candidate_row_matrix, dtype=float)
        info = self._table.refresh(u_prime, state.beta, state.omega,
                                   state.lambda_rep, state.lambda_back)
        if info["rebuilt_skills"] != [int(skill)]:
            raise AssertionError(
                f"candidate refresh rebuilt {info['rebuilt_skills']}, expected "
                f"[{int(skill)}] — the skill-local invalidation contract is broken")
        candidate_log_z = self._forward_all(state)
        # restore the column; the cache entry for the incoming state is still valid
        self._table.refresh(state.u_by_skill, state.beta, state.omega,
                            state.lambda_rep, state.lambda_back)
        delta = float((candidate_log_z - base).sum())
        if not math.isfinite(delta):
            raise ValueError("non-finite collapsed likelihood delta")
        return delta, candidate_log_z

    def commit_candidate(self, new_state: Stage6EState, candidate_log_z: np.ndarray
                         ) -> None:
        """Install an ACCEPTED candidate's normalisers as the cache for the new state.

        Rebuilds the accepted skill's column (bit-identical arithmetic to the candidate
        evaluation) so the table matches the new state; the forward pass is not repeated
        because `candidate_log_z` was computed from exactly this table content.
        """
        self._table.refresh(new_state.u_by_skill, new_state.beta, new_state.omega,
                            new_state.lambda_rep, new_state.lambda_back)
        self._cached_log_z = np.array(candidate_log_z, dtype=float, copy=True)
        self._cache_key = self._fingerprint(new_state)

    # ------------------------------------------------------------------ test support
    def full_rebuild_log_z(self, state: Stage6EState) -> np.ndarray:
        """`log Z_n` from a brand-new table — the parity reference for the incremental
        path. Test-only; never called by the sampler."""
        table = FastBlockScoreTable(
            traces=self.model.traces, epsilon=self.model.epsilon,
            n_skills=self.model.n_skills, min_width=self.model.min_width,
            max_width=self.model.max_width, n_roles=self.model.n_roles)
        table.refresh(state.u_by_skill, state.beta, state.omega,
                      state.lambda_rep, state.lambda_back)
        log_pi = np.log(np.asarray(state.pi, dtype=float))
        log_p = log_transition_matrix(state.transition)
        return np.array([
            forward(t, log_pi, log_p, self.model.delta_b, self.model.max_width,
                    self.model.min_width).log_normalizer for t in table.tables])
