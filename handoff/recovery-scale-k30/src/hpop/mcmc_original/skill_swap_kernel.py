"""The skill-swap (structure-to-identity reassignment) move — Condition C'.

## Why this move exists

Condition C's formal run showed both arms locking, but the C-MARG lock has a
very specific shape: all four chains recovered the SAME unlabeled structural
library, and differ only in which structure is attached to which anchored skill
identity. That is NOT label switching. With `pi*` and `P*` fixed and asymmetric,
NO non-identity permutation leaves the target invariant — for `sigma = (0 2)`,
`||pi - sigma(pi)||_2 = 0.4243` and `||P - sigma(P)||_F = 0.9798` — so the two
assignments are genuinely different modes, measured at ~125 nats apart on the
formal corpus, with three of four chains stuck in the inferior one.

No single-row `U` proposal can cross that barrier: relocating a whole structure
from one skill index to another requires moving every row of two skills at once,
through a region of near-zero posterior mass. This module supplies exactly that
transition and nothing else.

## The move

Pick a pair `(j, k)` uniformly from the `K(K-1)/2` unordered pairs and transpose
the two skills' complete utility matrices:

    U' = sigma_{jk}(U),   U'_j = U_k,  U'_k = U_j,  U'_i = U_i otherwise.

Three properties make the acceptance ratio exact and cheap:

* **Involution, uniform pair choice** — `q(U -> U') = q(U' -> U) = 2/(K(K-1))`,
  so the Hastings term is exactly zero.
* **Exchangeable prior** — `log p(U | rho) = sum_k log p(U_k | rho)` is a sum
  over skills, so a permutation only reorders the summands and
  `Delta log p(U | rho) = 0` exactly. This module computes the difference
  explicitly anyway and asserts it is zero, so a future non-exchangeable prior
  cannot silently break the ratio.
* **The collapsed likelihood already integrates out `(S, z)`** — with
  `ell_coll(U) = sum_n log Z_n(U)` evaluated at the FIXED `pi*`, `P*`, the
  asymmetry that makes the two assignments inequivalent enters the ratio
  automatically and correctly. Nothing here needs to know how `z` relabels.

Hence

    log alpha = ell_coll(U') - ell_coll(U).

For a chain sitting in the inferior assignment this is the full mode gap, so the
move accepts essentially deterministically.

## Composition

The move is scheduled on absolute sweep indices and runs BEFORE the rest of the
Condition-C sweep, so the unmodified exact FFBS refresh that opens
`condition_c_sweep_once` redraws every `(S, z)` at the post-swap `U`. That is the
same partially-collapsed ordering the validated collapsed-U kernel uses:
marginalise, move, then draw the discarded coordinates from their exact
conditional.

## What this module does not touch

`collapsed_u_kernel.py`, `collapsed_u_likelihood.py`, `semi_markov_ffbs.py`,
`sampler_u.py` and `matched_condition_c.py` are imported and called, never
modified — the Condition-C formal run in flight keeps a byte-identical import
path.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from hpop.mcmc_original.collapsed_u_likelihood import CollapsedULikelihood
from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.matched_condition_b import canonical_h_hash
from hpop.mcmc_original.matched_condition_c import condition_c_sweep_once
from hpop.mcmc_original.stage6c_frozen import log_structural_prior
from hpop.mcmc_original.stage6e_state import Stage6EState

__all__ = [
    "SkillSwapConfig", "is_swap_sweep", "swap_skills", "unordered_pairs",
    "skill_swap_mh_step", "condition_c_swap_sweep_once",
    "permutation_invariance_report",
]

MOVE_NAME = "U_skill_swap"


@dataclass
class SkillSwapConfig:
    """Cadence of the swap move. `every = 0` disables it (recovers Condition C)."""

    every: int = 50

    def __post_init__(self) -> None:
        if int(self.every) < 0:
            raise ValueError(f"swap_every must be >= 0, got {self.every}")


def is_swap_sweep(iteration: int, every: int) -> bool:
    """Scheduled on absolute sweep indices, so a resumed chain keeps its phase."""
    return bool(every) and (int(iteration) + 1) % int(every) == 0


def unordered_pairs(n_skills: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(int(n_skills))
            for j in range(i + 1, int(n_skills))]


def swap_skills(u_by_skill: np.ndarray, j: int, k: int) -> np.ndarray:
    """`sigma_{jk}(U)` — transpose two skills' complete utility matrices."""
    out = np.array(u_by_skill, dtype=float, copy=True)
    out[[int(j), int(k)]] = out[[int(k), int(j)]]
    return out


def permutation_invariance_report(pi, transition) -> dict:
    """Is any non-identity permutation a symmetry of the FIXED `(pi, P)`?

    Reported, not assumed: the whole justification for this move is that the
    answer is NO, so the two assignments are different modes rather than the
    same mode under an arbitrary relabelling.
    """
    from itertools import permutations
    pi = np.asarray(pi, dtype=float)
    transition = np.asarray(transition, dtype=float)
    rows = []
    for perm in permutations(range(len(pi))):
        index = np.array(perm)
        d_pi = float(np.linalg.norm(pi - pi[index]))
        d_p = float(np.linalg.norm(transition
                                   - transition[np.ix_(index, index)]))
        rows.append({"permutation": list(perm), "delta_pi_l2": d_pi,
                     "delta_P_frobenius": d_p,
                     "is_symmetry": bool(d_pi < 1e-12 and d_p < 1e-12)})
    non_identity = [r for r in rows if r["permutation"] != list(range(len(pi)))]
    return {"rows": rows,
            "any_non_identity_symmetry": any(r["is_symmetry"]
                                             for r in non_identity),
            "min_delta_pi_over_non_identity": min(r["delta_pi_l2"]
                                                  for r in non_identity),
            "min_delta_P_over_non_identity": min(r["delta_P_frobenius"]
                                                 for r in non_identity)}


def skill_swap_mh_step(state: Stage6EState, likelihood: CollapsedULikelihood,
                       rng, pairs: list | None = None) -> tuple:
    """One MH transposition of two skills' `U`, scored by the collapsed likelihood.

    Returns a NEW state and the move record. The stored `(S, z)` is neither read
    nor written: the caller's FFBS refresh redraws it at the post-move `U`.
    """
    state = state.copy()
    n_skills = int(np.asarray(state.u_by_skill).shape[0])
    candidates = unordered_pairs(n_skills) if pairs is None else list(pairs)
    j, k = candidates[int(rng.integers(len(candidates)))]

    began = time.perf_counter()
    state.proposed[MOVE_NAME] = state.proposed.get(MOVE_NAME, 0) + 1
    current_ll = float(likelihood.log_z_per_trace(state).sum())

    candidate_u = swap_skills(state.u_by_skill, j, k)
    candidate_state = state.copy()
    candidate_state.u_by_skill = candidate_u
    candidate_ll = float(likelihood.log_z_per_trace(candidate_state).sum())

    # Exactly zero for any exchangeable prior; computed rather than assumed.
    d_prior = float(
        sum(log_structural_prior(candidate_u[i], state.rho)
            for i in range(n_skills))
        - sum(log_structural_prior(state.u_by_skill[i], state.rho)
              for i in range(n_skills)))
    if abs(d_prior) > 1e-8:
        raise AssertionError(
            f"prior is not exchangeable across skills (delta {d_prior}); the "
            "swap acceptance ratio would need a prior term")

    # involution with uniform pair choice: log q(U'->U) - log q(U->U') = 0
    log_alpha = (candidate_ll - current_ll) + d_prior
    h_changed = tuple(canonical_h_hash(precedence_from_u(state.u_by_skill[i]))
                      for i in range(n_skills)) != tuple(
        canonical_h_hash(precedence_from_u(candidate_u[i]))
        for i in range(n_skills))

    accepted = bool(log_alpha >= 0.0 or math.log(rng.random()) < log_alpha)
    if accepted:
        state.u_by_skill = candidate_u
        state.accepted[MOVE_NAME] = state.accepted.get(MOVE_NAME, 0) + 1
        # the likelihood cache now describes the candidate, which IS the state
    else:
        # restore the cache to the retained state, so the next reader is exact
        likelihood.log_z_per_trace(state)

    return state, {"pair": [int(j), int(k)], "accepted": accepted,
                   "assignment_changed": bool(h_changed),
                   "log_alpha": float(log_alpha),
                   "d_log_lik_collapsed": float(candidate_ll - current_ll),
                   "d_log_prior": d_prior,
                   "seconds": time.perf_counter() - began}


def condition_c_swap_sweep_once(state: Stage6EState, sampler,
                                swap: SkillSwapConfig, rng) -> tuple:
    """One Condition-C' sweep: scheduled swap, then the unchanged Condition-C sweep.

    The Condition-C sweep opens with the exact FFBS refresh of every `(S, z)`,
    so an accepted swap is immediately followed by a consistent redraw of the
    paths — the ordering the partially-collapsed argument requires.
    """
    swap_record = None
    if is_swap_sweep(state.iteration, swap.every):
        state, swap_record = skill_swap_mh_step(
            state, sampler.collapsed_likelihood, rng)
    state, collapsed_record, info = condition_c_sweep_once(state, sampler, rng)
    return state, swap_record, collapsed_record, info
