"""The collapsed `U` likelihood over the CPA layer, for the learned-order arm.

`mcmc_original.collapsed_u_likelihood.CollapsedULikelihood` is the registered object, but
it builds `FastBlockScoreTable`, which indexes the vocabulary directly as roles and so
assumes `A = m`. The ladder runs `A = 50` with `m = 10` per skill through injective role
maps, so it needs the same collapsed quantity computed over `CPABlockScoreTable`.

Everything else is deliberately the sealed code: the row proposal, the structural prior,
the forward recursion and the accept rule are imported, not reimplemented. The only thing
that differs from the registered kernel is which table the forward recursion reads, which
is exactly the difference the CPA layer is.

## Skill-local rebuilds are used, and only because they are proven exact

A candidate `U` differs from the current one in a single skill, so only that skill's score
column changes. `CPABlockScoreTable.refresh_changed` rebuilds exactly those columns. That
is a factor of `K` on the dominant cost, which at `K = 30` is the difference between a
usable learned-order arm and an unusable one -- but it is used here **only** because
`scripts/k_ladder/fast_exact_parity_gate.py` establishes it is bitwise identical to
rebuilding everything, including on the accept/reject decisions themselves. An
almost-right column would change a score, then a log ratio, then a decision, and nothing
downstream would say so.
"""

from __future__ import annotations

import math

import numpy as np

from hpop.mcmc_original.semi_markov_ffbs import forward
from hpop.mcmc_original.stage6c_frozen import log_structural_prior
from hpop.mcmc_original.sampler_u import propose_row
from hpop.mcmc_original.stage6e_state import Stage6EState
from hpop.mcmc_original.transitions import log_transition_matrix

from .block_tables import CPABlockScoreTable

__all__ = ["CPACollapsedULikelihood", "collapsed_u_mh_step_cpa"]

MOVE_NAME = "U_collapsed_cpa"


class CPACollapsedULikelihood:
    """`ell_coll(U)` and its single-skill candidate deltas over the CPA candidate table."""

    __slots__ = ("model", "role_maps", "epsilon", "_table", "_cache_key", "_cached_log_z",
                 "evaluations", "cache_hits")

    def __init__(self, model, role_maps, epsilon: float):
        self.model = model
        self.role_maps = role_maps
        self.epsilon = float(epsilon)
        self._table = CPABlockScoreTable(
            traces=model.traces, epsilon=float(epsilon), role_maps=role_maps,
            min_width=model.min_width, max_width=model.max_width)
        self._cache_key = None
        self._cached_log_z = None
        self.evaluations = 0
        self.cache_hits = 0

    # ------------------------------------------------------------------ fingerprinting
    @staticmethod
    def _fingerprint(state: Stage6EState) -> tuple:
        return (np.asarray(state.u_by_skill, dtype=float).tobytes(),
                float(state.beta), float(state.omega),
                float(state.lambda_rep), float(state.lambda_back),
                np.asarray(state.pi, dtype=float).tobytes(),
                np.asarray(state.transition, dtype=float).tobytes())

    def _forward_all(self, state: Stage6EState) -> np.ndarray:
        log_pi = np.log(np.asarray(state.pi, dtype=float))
        log_p = log_transition_matrix(state.transition)
        self.evaluations += 1
        return np.array([
            forward(table, log_pi, log_p, self.model.delta_b, self.model.max_width,
                    self.model.min_width).log_normalizer
            for table in self._table.tables], dtype=float)

    def log_z_per_trace(self, state: Stage6EState) -> np.ndarray:
        key = self._fingerprint(state)
        if key == self._cache_key and self._cached_log_z is not None:
            self.cache_hits += 1
            self._table.refresh_changed(state.u_by_skill, state.beta, state.omega,
                                        state.lambda_rep, state.lambda_back)
            return self._cached_log_z.copy()
        self._table.refresh_changed(state.u_by_skill, state.beta, state.omega,
                                    state.lambda_rep, state.lambda_back)
        log_z = self._forward_all(state)
        self._cache_key, self._cached_log_z = key, log_z.copy()
        return log_z

    @property
    def tables(self) -> list:
        return self._table.tables

    def refresh_to(self, state: Stage6EState) -> None:
        self._table.refresh_changed(state.u_by_skill, state.beta, state.omega,
                                    state.lambda_rep, state.lambda_back)

    # ------------------------------------------------------------------ the delta
    def delta_for_candidate(self, state: Stage6EState, skill: int,
                            candidate_row_matrix) -> tuple:
        """`Delta ell_coll` for replacing `U[skill]`, leaving the table at the CURRENT U.

        Whatever the caller decides about the proposal, on return the table describes the
        incoming state -- so a rejected move cannot leave a candidate column behind.
        """
        base = self.log_z_per_trace(state)
        u_prime = np.array(state.u_by_skill, dtype=float, copy=True)
        u_prime[int(skill)] = np.asarray(candidate_row_matrix, dtype=float)

        info = self._table.refresh_changed(u_prime, state.beta, state.omega,
                                           state.lambda_rep, state.lambda_back)
        if info["rebuilt_skills"] != [int(skill)]:
            raise AssertionError(
                f"candidate refresh rebuilt {info['rebuilt_skills']}, expected "
                f"[{int(skill)}] -- the skill-local invalidation contract is broken")
        candidate_log_z = self._forward_all(state)

        self._table.refresh_changed(state.u_by_skill, state.beta, state.omega,
                                    state.lambda_rep, state.lambda_back)
        delta = float((candidate_log_z - base).sum())
        if not math.isfinite(delta):
            raise ValueError("non-finite collapsed likelihood delta")
        return delta, candidate_log_z

    def commit_candidate(self, new_state: Stage6EState, candidate_log_z) -> None:
        self._table.refresh_changed(new_state.u_by_skill, new_state.beta,
                                    new_state.omega, new_state.lambda_rep,
                                    new_state.lambda_back)
        self._cache_key = self._fingerprint(new_state)
        self._cached_log_z = np.asarray(candidate_log_z, dtype=float).copy()


def collapsed_u_mh_step_cpa(state: Stage6EState, likelihood: CPACollapsedULikelihood,
                            rng, scale: float) -> dict:
    """One collapsed MH update of a uniformly chosen `(skill, row)`, in place on `state`.

    The proposal is the sealed symmetric row random walk and the prior is the sealed
    structural prior, so at fixed `rho` the Hastings term is exactly zero and
    `log alpha = Delta ell_coll + Delta log prior` -- the same rule the registered kernel
    applies, over a different table.
    """
    k = int(np.asarray(state.u_by_skill).shape[0])
    m = int(np.asarray(state.u_by_skill).shape[1])
    skill = int(rng.integers(k))
    row = int(rng.integers(m))
    u_k = np.array(state.u_by_skill[skill], dtype=float)

    candidate = propose_row(u_k, row, float(scale), rng)
    record = {"move": MOVE_NAME, "skill": skill, "row": row, "accepted": False,
              "invalid": False, "log_alpha": None}

    candidate_prior = log_structural_prior(candidate, state.rho)
    if not math.isfinite(candidate_prior):
        record["invalid"] = True
        return record

    d_coll, candidate_log_z = likelihood.delta_for_candidate(state, skill, candidate)
    d_prior = candidate_prior - log_structural_prior(u_k, state.rho)
    log_alpha = d_coll + d_prior

    accepted = bool(log_alpha >= 0.0 or math.log(rng.random()) < log_alpha)
    if accepted:
        u = np.array(state.u_by_skill, dtype=float, copy=True)
        u[skill] = candidate
        state.u_by_skill = u
        likelihood.commit_candidate(state, candidate_log_z)

    record.update(accepted=accepted, log_alpha=float(log_alpha),
                  d_log_lik_collapsed=float(d_coll), d_log_prior=float(d_prior))
    return record
