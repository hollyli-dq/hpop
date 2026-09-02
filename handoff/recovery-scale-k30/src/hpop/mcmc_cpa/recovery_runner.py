"""Segment-based learned-order chain for the recovery-at-scale experiment.

A chain advances in registered segments of `REGIME.SEGMENT_SWEEPS` sweeps. Between
segments the full state is serialised, so any machine can resume any chain, and the
coordinator can evaluate the truth-free gates after each segment and stop the chain the
moment they pass -- effort is an outcome, not an input. Exact resume is guaranteed by the
common-random-number design: every generator is derived from
(replicate, K, chain, sweep, move type, proposal index), never from stream position, so a
chain split across ten processes is bit-identical to one that never stopped. That
property is load-bearing and is pinned by `test_recovery_runner.py`.

The U budget is PACED, not capped: `U_RATE_PER_ROLE_PER_SWEEP` proposals per role per
sweep, flat in K by construction (the quota machinery distributes each segment's
`round(rate * K * m * sweeps)` proposals evenly over its update events). The chain stops
when the gates say so or at `CAP_SWEEPS`, which is a resource statement, not a tuning
knob.
"""

from __future__ import annotations

import numpy as np

from hpop.mcmc_original.fast_segmentation_kernel import key_of, segmentation_of
from hpop.mcmc_original.full_latent_constants import (FIXED_LAMBDA_BACK,
                                                     FIXED_LAMBDA_REP, FIXED_OMEGA,
                                                     FIXED_RHO_0)
from hpop.mcmc_original.matched_full_latent import (gibbs_pi_p, validate_paths,
                                                    validate_pi_p)
from hpop.mcmc_original.stage6e_state import Stage6EState
from hpop.mcmc_optimized.segmentation import ffbs_segmentation_draw
from hpop.mcmc_original.transitions import sample_transition_matrix

from .collapsed_u import CPACollapsedULikelihood, collapsed_u_mh_step_cpa
from .crn import CommonRandomNumbers
from .ladder_runner import _LikelihoodTables, _initial_segmentation
from .recovery_regime import REGIME
from .seeds import LadderSeeds
from .u_quota import distribute_quota, update_events

__all__ = ["chain_crn", "init_state", "run_segment", "state_to_json", "state_from_json"]


def chain_crn(replicate: int, k: int, chain: int) -> CommonRandomNumbers:
    return CommonRandomNumbers(int(replicate), int(k), int(chain),
                               seeds=LadderSeeds(root=int(REGIME.ROOT_ENTROPY)))


def init_state(model, crn: CommonRandomNumbers) -> dict:
    """Sweep-0 state: dispersed U start (never the truth), prior pi/P, seeded per chain."""
    rng = crn.rng("init", 0)
    n_skills = int(model.n_skills)
    u_start = rng.standard_normal(size=(n_skills, int(model.n_roles), 2))
    pi = rng.dirichlet(np.ones(n_skills))
    transition = sample_transition_matrix(np.zeros((n_skills, n_skills)), n_skills,
                                          rng, float(model.eta_transition))
    segmentations = tuple(
        _initial_segmentation(len(t), n + crn.chain, n_skills,
                              model.min_width, model.max_width)
        for n, t in enumerate(model.traces))
    return {
        "sweep": 0,
        "u_by_skill": u_start.tolist(),
        "pi": pi.tolist(),
        "transition": transition.tolist(),
        "segmentation_keys": [key_of(s) for s in segmentations],
        "u_proposed": 0, "u_accepted": 0, "u_invalid": 0,
        "h_changing_accepted_per_skill": [0] * n_skills,
        "role_attempts": np.zeros((n_skills, int(model.n_roles)),
                                  dtype=int).tolist(),
    }


def _stage_state(model, state: dict) -> Stage6EState:
    return Stage6EState(
        segmentations=tuple(segmentation_of(tuple(map(tuple, key)))
                            for key in state["segmentation_keys"]),
        u_by_skill=np.asarray(state["u_by_skill"], dtype=float),
        rho=float(FIXED_RHO_0), beta=float(REGIME.BETA), omega=float(FIXED_OMEGA),
        lambda_rep=float(FIXED_LAMBDA_REP), lambda_back=float(FIXED_LAMBDA_BACK),
        pi=np.asarray(state["pi"], dtype=float),
        transition=np.asarray(state["transition"], dtype=float))


def run_segment(model, role_maps, state: dict, crn: CommonRandomNumbers,
                u_scale: float, sweeps: int | None = None) -> tuple:
    """Advance one segment from `state`. Returns `(new_state, draws)`.

    `draws["u"]` is on the U-update-event axis; `draws["labels"]/["boundaries"]` on the
    thin grid. Both carry their sweep indices so the coordinator can window them without
    guessing.
    """
    from hpop.mcmc_cpa.recovery_regime import REGIME as R

    sweeps = int(R.SEGMENT_SWEEPS if sweeps is None else sweeps)
    start = int(state["sweep"])
    stage = _stage_state(model, state)
    n_skills, n_roles = int(model.n_skills), int(model.n_roles)

    likelihood = CPACollapsedULikelihood(model, role_maps, float(R.EPSILON))
    likelihood.refresh_to(stage)
    tables = _LikelihoodTables(likelihood)

    events = update_events(sweeps, int(R.U_EVERY)) + start
    quota = int(round(R.U_RATE_PER_ROLE_PER_SWEEP * n_skills * n_roles * sweeps))
    moves_at = dict(zip(events.tolist(),
                        distribute_quota(quota, events.size).tolist()))

    role_attempts = np.asarray(state["role_attempts"], dtype=int)
    h_changing = list(state["h_changing_accepted_per_skill"])
    u_proposed, u_accepted, u_invalid = (int(state["u_proposed"]),
                                         int(state["u_accepted"]),
                                         int(state["u_invalid"]))
    draws = {"u": [], "u_event_sweep": [], "labels": [], "boundaries": [],
             "draw_sweep": [], "pi_sorted": [], "p_spectrum": []}
    from hpop.mcmc_original.latent_poset import precedence_from_u

    for sweep in range(start, start + sweeps):
        validate_pi_p(stage, model)
        if sweep % int(R.U_EVERY) == 0:
            before = [np.asarray(precedence_from_u(stage.u_by_skill[k]))
                      for k in range(n_skills)] if moves_at.get(sweep, 0) else None
            for proposal in range(int(moves_at.get(sweep, 0))):
                record = collapsed_u_mh_step_cpa(stage, likelihood,
                                                 crn.rng("u", sweep, proposal),
                                                 float(u_scale))
                u_proposed += 1
                u_accepted += int(record["accepted"])
                u_invalid += int(record["invalid"])
                role_attempts[int(record["skill"]), int(record["row"])] += 1
                if record["accepted"]:
                    after = np.asarray(
                        precedence_from_u(stage.u_by_skill[int(record["skill"])]))
                    if not np.array_equal(before[int(record["skill"])], after):
                        h_changing[int(record["skill"])] += 1
                        before[int(record["skill"])] = after
            if moves_at.get(sweep, 0):
                draws["u"].append(np.asarray(stage.u_by_skill, dtype=float).tolist())
                draws["u_event_sweep"].append(int(sweep))

        tables.refresh(stage)
        draw = ffbs_segmentation_draw(model, stage, tables, crn.rng("ffbs", sweep))
        stage.segmentations = tuple(segmentation_of(key) for key in draw["keys"])
        tables.mark_stale()
        validate_paths(stage, model)
        gibbs_pi_p(stage, model, crn.rng("pi_p", sweep))

        if sweep % int(R.THIN) == 0:
            draws["labels"].append([[int(s.skill) for s in seg.segments]
                                    for seg in stage.segmentations])
            draws["boundaries"].append([[int(s.end) for s in seg.segments[:-1]]
                                        for seg in stage.segmentations])
            draws["draw_sweep"].append(int(sweep))
            # permutation-invariant summaries for the gates: sorting removes the label
            draws["pi_sorted"].append(np.sort(np.asarray(stage.pi))[::-1].tolist())
            spectrum = np.sort(np.abs(np.linalg.eigvals(
                np.asarray(stage.transition))))[::-1]
            draws["p_spectrum"].append(spectrum[:5].tolist())
        stage.iteration += 1

    new_state = {
        "sweep": start + sweeps,
        "u_by_skill": np.asarray(stage.u_by_skill, dtype=float).tolist(),
        "pi": np.asarray(stage.pi, dtype=float).tolist(),
        "transition": np.asarray(stage.transition, dtype=float).tolist(),
        "segmentation_keys": [key_of(s) for s in stage.segmentations],
        "u_proposed": u_proposed, "u_accepted": u_accepted, "u_invalid": u_invalid,
        "h_changing_accepted_per_skill": h_changing,
        "role_attempts": role_attempts.tolist(),
    }
    return new_state, draws


def state_to_json(state: dict) -> dict:
    return state


def state_from_json(payload: dict) -> dict:
    return payload
