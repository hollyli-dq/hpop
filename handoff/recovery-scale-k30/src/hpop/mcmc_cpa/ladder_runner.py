"""One chain runner, three arms, differing in exactly one thing each.

    support-only    the candidate block score knows only which CPAs a skill can emit
    oracle-order    the full recurrent score, evaluated at the TRUE `U`, held fixed
    learned-order   the full recurrent score, with `U` inferred

The arms must differ in **exactly one place** — what the candidate block score knows — or
the comparison means nothing. So there is one runner, and the arm is a single argument.
The segmentation prior, the transition treatment, the data, the initialisation, the
proposal schedule, the chain count, the sweep count and the RNG stream are shared by
construction rather than by three implementations that were written to match.

## What each contrast means

    oracle-order  -  support-only     how much the AVAILABLE order information is worth
    learned-order -  support-only     the REALISED end-to-end gain
    oracle-order  -  learned-order    the INFERENCE gap

The first is a diagnostic about information, not a bound: an inferred `U` is not obliged
to be less useful than the true one at every finite sweep count, and averaging over the
posterior is not the same operation as plugging in a point truth. The second is the number
a practitioner actually gets. The third says how much of the available information the
sampler is failing to extract, which is the one that says whether to work on inference.

## What the support-only arm is

    log e_support(block, k) = 0     if every CPA in the block lies in skill k's support
                            = -inf  otherwise

A uniform-within-support emission would give `-w log m` for a compatible block, and since
every skill has the same `m` and every segmentation of a trace covers the same `J`
positions, those terms sum to the constant `-J log m` for **every** segmentation and every
labelling. It therefore cancels in every ratio the sampler forms, and `0` is the same model.

## Why structure recovery is not applicable to the baseline


The support-only score does not read `U` at all. A `U` sampler run against it would be
drawing from the prior with the data contributing nothing, so the baseline **does not move
`U`** and reports structure recovery as not applicable. Presenting a prior draw beside a
data-informed posterior as if they were comparable structure estimates would be misleading.
"""

from __future__ import annotations

import time

import numpy as np

from hpop.mcmc_original.matched_full_latent import gibbs_pi_p, validate_paths, validate_pi_p
from hpop.mcmc_original.stage6e_state import Stage6EState
from hpop.mcmc_original.fast_segmentation_kernel import segmentation_of
from hpop.mcmc_original.types import Segment, Segmentation
from hpop.mcmc_optimized.segmentation import ffbs_segmentation_draw

from .block_tables import CPABlockScoreTable
from .collapsed_u import CPACollapsedULikelihood, collapsed_u_mh_step_cpa
from .crn import CommonRandomNumbers
from .u_quota import attempts_per_role_summary, quota_schedule
from .seeds import LadderSeeds
from .support_baseline import SupportOnlyBlockScoreTable

__all__ = ["ORACLE_ORDER", "LEARNED_ORDER", "SUPPORT_ONLY", "ARMS",
           "run_ladder_chain", "ArmTables"]

SUPPORT_ONLY = "support-only"
ORACLE_ORDER = "oracle-order"        # full recurrent score at the TRUE U, held fixed
LEARNED_ORDER = "learned-order"      # full recurrent score, U inferred

ARMS = (SUPPORT_ONLY, ORACLE_ORDER, LEARNED_ORDER)

#: Arms whose candidate score reads `U` at all.
_ORDER_ARMS = (ORACLE_ORDER, LEARNED_ORDER)


class ArmTables:
    """Adapts either score table to the interface `ffbs_segmentation_draw` expects.

    The full arm rebuilds when `U` moves; the support-only arm cannot move because its
    score does not depend on `U`, so it is built once. Both expose the same `tables_for`
    contract, which is what lets one runner drive both.
    """

    def __init__(self, arm: str, model, role_maps, epsilon: float):
        self.arm = arm
        self.model = model
        self.stale = False
        if arm in _ORDER_ARMS:
            self._table = CPABlockScoreTable(
                traces=model.traces, epsilon=float(epsilon), role_maps=role_maps,
                min_width=model.min_width, max_width=model.max_width)
            self._built_at = None
            self._built_key = None
        elif arm == SUPPORT_ONLY:
            self._table = SupportOnlyBlockScoreTable(
                traces=model.traces, role_maps=role_maps,
                min_width=model.min_width, max_width=model.max_width)
        else:
            raise ValueError(f"unknown arm {arm!r}")

    @staticmethod
    def _key(state: Stage6EState) -> tuple:
        """Everything the candidate table is a function of, given a fixed corpus.

        `CPABlockScoreTable.refresh` reads exactly these five; the rest of the table
        (traces, epsilon, role maps, width buckets) is fixed at construction. So two
        states with the same key have the same table, bit for bit.
        """
        return (np.asarray(state.u_by_skill, dtype=float).tobytes(),
                float(state.beta), float(state.omega),
                float(state.lambda_rep), float(state.lambda_back))

    def refresh(self, state: Stage6EState) -> dict:
        if self.arm == SUPPORT_ONLY:
            self.stale = False
            return {"rebuilt": False, "reason": "support-only score does not read U"}
        key = self._key(state)
        if self._built_key is not None and key == self._built_key:
            # A deterministic function of unchanged inputs. Rebuilding would burn the
            # dominant cost of a sweep to reproduce the array already held.
            self.stale = False
            return {"rebuilt": False, "reason": "table inputs unchanged since last build"}
        info = self._table.refresh_changed(state.u_by_skill, state.beta, state.omega,
                                           state.lambda_rep, state.lambda_back)
        self._built_at = np.array(state.u_by_skill, copy=True)
        self._built_key = key
        self.stale = False
        return dict(info) | {"rebuilt": True}

    def mark_stale(self) -> None:
        self.stale = True

    def tables_for(self, state: Stage6EState) -> list:
        if self.stale:
            raise AssertionError("candidate tables are stale")
        if self.arm in _ORDER_ARMS and not np.array_equal(self._built_at,
                                                       state.u_by_skill):
            raise AssertionError("candidate tables were built at a different U")
        return self._table.tables


def _initial_segmentation(length: int, index: int, n_skills: int,
                          min_width: int, max_width: int) -> Segmentation:
    remaining, widths = int(length), []
    while remaining > max_width:
        step = max_width if remaining - max_width >= min_width else remaining - min_width
        widths.append(step)
        remaining -= step
    widths.append(remaining)
    segments, start = [], 0
    for position, width in enumerate(widths):
        segments.append(Segment(start, start + width, (index + position) % n_skills))
        start += width
    return Segmentation(tuple(segments))


def _crn_for(crn, replicate: int, n_skills: int, chain: int, seed: int):
    """The chain's common-random-number source.

    `seed` sets the CRN **root**, not a position in a stream. Arms called with the same
    seed therefore share every indexed generator -- which is the point -- while a
    different seed gives a genuinely different chain. An explicitly supplied `crn` wins:
    that caller has taken responsibility for the sharing.
    """
    if crn is not None:
        return crn
    return CommonRandomNumbers(int(replicate), int(n_skills), int(chain),
                               seeds=LadderSeeds(root=int(seed)))


def _dispersed_u_start(rng, u_truth) -> np.ndarray:
    """A start drawn from the structural prior's scale, deliberately NOT the truth.

    The learned-order arm exists to measure what inference recovers, so it must not be
    handed the answer. Only the SHAPE of the truth is used.
    """
    return rng.standard_normal(size=np.asarray(u_truth).shape)


class _LikelihoodTables:
    """Adapts `CPACollapsedULikelihood` to the `tables_for` contract the FFBS expects.

    The collapsed likelihood already owns the candidate table and keeps it at the current
    `U`, so this neither rebuilds nor caches; it would be wrong for two objects to hold
    opinions about which `U` the table describes.
    """

    __slots__ = ("likelihood", "stale")

    def __init__(self, likelihood):
        self.likelihood = likelihood
        self.stale = False

    def refresh(self, state) -> dict:
        self.likelihood.refresh_to(state)
        self.stale = False
        return {"rebuilt": True, "reason": "learned-order: table follows the U kernel"}

    def mark_stale(self) -> None:
        self.stale = True

    def tables_for(self, state) -> list:
        if self.stale:
            raise AssertionError("candidate tables are stale")
        return self.likelihood.tables


def run_ladder_chain(arm: str, model, role_maps, u_by_skill, chain: int, sweeps: int,
                     warmup: int, seed: int, epsilon: float = 0.02,
                     fixed=None, thin: int = 5, record_every: int = 1,
                     u_every: int = 1, u_moves: int = 1, u_scale: float = 0.5,
                     u_start=None, rho: float | None = None, crn=None,
                     replicate: int = 0,
                     target_u_attempts_per_role: float | None = None) -> dict:
    """One chain. `arm` is the ONLY thing that differs between the conditions.

    `support-only` and `oracle-order` both hold `u_by_skill` fixed -- the first ignores it
    entirely, the second scores at it throughout. That pair isolates what the block score
    contributes to segmentation and skill labelling, with no `U` inference in either.

    `learned-order` starts from `u_start` (a dispersed draw when not supplied, never the
    truth) and moves `U` with the sealed collapsed row kernel. `u_by_skill` is then the
    truth only for scoring recovery, never for initialisation -- passing the truth as the
    start would make the arm an oracle wearing a different name.

    The `U` budget is set one of two ways. `target_u_attempts_per_role` is the registered
    proportional-effort rule: the chain's total quota is `round(target * K * m)`, spread
    over the update events by a cumulative quota, so attempted updates per role vector are
    constant across `K`. `u_moves` is the older fixed-per-event count, kept for pilots and
    for the fixed-`U` arms that make no proposals at all; it leaves effort falling as
    `1/K` and must not be used for a production ladder.
    """
    from hpop.mcmc_original.full_latent_constants import (FIXED_BETA, FIXED_LAMBDA_BACK,
                                                          FIXED_LAMBDA_REP, FIXED_OMEGA,
                                                          FIXED_RHO_0)

    n_skills = int(model.n_skills)
    crn = _crn_for(crn, replicate, n_skills, chain, seed)
    # Initialisation draws from its own indexed stream, so it cannot be shifted by
    # anything an arm does later.
    rng = crn.rng("init", 0)

    pi = rng.dirichlet(np.ones(n_skills))
    from hpop.mcmc_original.transitions import sample_transition_matrix
    transition = sample_transition_matrix(np.zeros((n_skills, n_skills)), n_skills,
                                          rng, float(model.eta_transition))
    state = Stage6EState(
        segmentations=tuple(
            _initial_segmentation(len(t), n + chain, n_skills,
                                  model.min_width, model.max_width)
            for n, t in enumerate(model.traces)),
        u_by_skill=np.asarray(u_by_skill, dtype=float), rho=float(FIXED_RHO_0),
        beta=float(FIXED_BETA), omega=float(FIXED_OMEGA),
        lambda_rep=float(FIXED_LAMBDA_REP), lambda_back=float(FIXED_LAMBDA_BACK),
        pi=pi, transition=transition)

    schedule, moves_at = None, None
    if arm == LEARNED_ORDER:
        if u_start is None:
            u_start = _dispersed_u_start(rng, np.asarray(u_by_skill, dtype=float))
        state.u_by_skill = np.asarray(u_start, dtype=float, copy=True)
        if rho is not None:
            state.rho = float(rho)
        likelihood = CPACollapsedULikelihood(model, role_maps, epsilon)
        likelihood.refresh_to(state)
        tables = _LikelihoodTables(likelihood)
        if target_u_attempts_per_role is not None:
            schedule = quota_schedule(target_u_attempts_per_role, n_skills,
                                      int(model.n_roles), int(sweeps), int(warmup),
                                      int(u_every))
            moves_at = dict(zip(schedule["events"].tolist(),
                                schedule["moves_per_event"].tolist()))

    else:
        likelihood = None
        tables = ArmTables(arm, model, role_maps, epsilon)

    began = time.perf_counter()
    kept = {"labels": [], "n_segments": [], "boundaries": [], "u": []}
    moved = 0
    u_proposed = u_accepted = u_invalid = 0
    u_proposed_burnin = u_accepted_burnin = 0
    u_proposed_retained = u_accepted_retained = 0
    role_attempts = np.zeros((n_skills, int(model.n_roles)), dtype=np.int64)

    for sweep in range(int(sweeps)):
        validate_pi_p(state, model)
        if arm == LEARNED_ORDER and int(u_every) > 0 and sweep % int(u_every) == 0:
            n_here = (moves_at.get(sweep, 0) if moves_at is not None else int(u_moves))
            in_burnin = sweep < int(warmup)
            for proposal in range(int(n_here)):
                record = collapsed_u_mh_step_cpa(state, likelihood,
                                                 crn.rng("u", sweep, proposal),
                                                 float(u_scale))
                u_proposed += 1
                u_accepted += int(record["accepted"])
                u_invalid += int(record["invalid"])
                role_attempts[int(record["skill"]), int(record["row"])] += 1
                if in_burnin:
                    u_proposed_burnin += 1
                    u_accepted_burnin += int(record["accepted"])
                else:
                    u_proposed_retained += 1
                    u_accepted_retained += int(record["accepted"])
        tables.refresh(state)
        draw = ffbs_segmentation_draw(model, state, tables,
                                      crn.rng("ffbs", sweep))
        state.segmentations = tuple(segmentation_of(key) for key in draw["keys"])
        tables.mark_stale()
        validate_paths(state, model)
        gibbs_pi_p(state, model, crn.rng("pi_p", sweep))
        moved += int(draw["movement"]["states_changed"])

        if sweep >= int(warmup) and (sweep - int(warmup)) % int(thin) == 0:
            kept["labels"].append([[int(s.skill) for s in seg.segments]
                                   for seg in state.segmentations])
            kept["n_segments"].append([len(seg.segments)
                                       for seg in state.segmentations])
            kept["boundaries"].append([[int(s.end) for s in seg.segments[:-1]]
                                       for seg in state.segmentations])
            if arm == LEARNED_ORDER:
                kept["u"].append(np.asarray(state.u_by_skill, dtype=float).tolist())
        state.iteration += 1

    return {
        "arm": arm, "chain": int(chain), "seed": int(seed),
        "sweeps": int(sweeps), "warmup": int(warmup), "thin": int(thin),
        "retained_draws": len(kept["labels"]),
        "seconds": time.perf_counter() - began,
        "ffbs_states_changed_total": moved,
        "structure_recovery": ("NOT APPLICABLE" if arm == SUPPORT_ONLY
                               else "fixed at truth" if arm == ORACLE_ORDER
                               else "available"),
        "u_held_fixed": arm != LEARNED_ORDER,
        "crn": crn.provenance(),
        "crn_generators_issued": crn.generators_issued,
        "u_proposed": u_proposed,
        "u_accepted": u_accepted,
        "u_invalid": u_invalid,
        "u_acceptance_rate": (u_accepted / u_proposed) if u_proposed else None,
        "u_proposed_burnin": u_proposed_burnin,
        "u_accepted_burnin": u_accepted_burnin,
        "u_proposed_retained": u_proposed_retained,
        "u_accepted_retained": u_accepted_retained,
        "u_acceptance_rate_burnin": ((u_accepted_burnin / u_proposed_burnin)
                                     if u_proposed_burnin else None),
        "u_acceptance_rate_retained": ((u_accepted_retained / u_proposed_retained)
                                       if u_proposed_retained else None),
        "u_attempts_per_role_burnin": (u_proposed_burnin / role_attempts.size
                                       if role_attempts.size else 0.0),
        "u_attempts_per_role_retained": (u_proposed_retained / role_attempts.size
                                         if role_attempts.size else 0.0),
        "u_role_attempts": role_attempts.tolist(),
        "u_role_attempt_summary": attempts_per_role_summary(role_attempts),
        "u_quota_schedule": (None if schedule is None else
                             {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                              for k, v in schedule.items()
                              if k not in ("events", "moves_per_event")}),
        "u_kernel": ("enabled" if arm == LEARNED_ORDER else "disabled"),
        "N_U": u_proposed,
        "N_U_expected": (None if schedule is None else schedule["total_quota_M_K"]),
        "arm_provenance": {
            SUPPORT_ONLY: "U-kernel disabled, N_U = 0; score does not read U at all",
            ORACLE_ORDER: "U-kernel disabled, N_U = 0; score reads U, held at the truth",
            LEARNED_ORDER: "U-kernel enabled, N_U = M_K from the registered quota",
        }[arm],
        "u_budget_rule": (
            "no U proposals: U is held fixed in this arm" if arm != LEARNED_ORDER else
            "proportional effort: round(target*K*m) spread by cumulative quota"
            if schedule is not None else
            f"fixed {u_moves} per event -- effort per role falls as 1/K; NOT for a "
            f"production ladder"),
        "draws": kept,
        "final_pi": state.pi.tolist(),
        "final_transition": state.transition.tolist(),
    }
