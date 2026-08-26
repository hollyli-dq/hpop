"""One chain runner, two arms: full recurrent likelihood, and the support-only baseline.

The two arms must differ in **exactly one place** — the candidate block score — or the
comparison means nothing. So there is one runner, and the arm is a single argument. The
segmentation prior, the transition treatment, the data, the initialisation, the proposal
schedule, the chain count, the sweep count and the RNG stream are shared by construction
rather than by two implementations that were written to match.

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
from .support_baseline import SupportOnlyBlockScoreTable

__all__ = ["FULL_RFS", "SUPPORT_ONLY", "run_ladder_chain", "ArmTables"]

FULL_RFS = "full-rfs"
SUPPORT_ONLY = "support-only"


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
        if arm == FULL_RFS:
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
        info = self._table.refresh(state.u_by_skill, state.beta, state.omega,
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
        if self.arm == FULL_RFS and not np.array_equal(self._built_at,
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


def run_ladder_chain(arm: str, model, role_maps, u_by_skill, chain: int, sweeps: int,
                     warmup: int, seed: int, epsilon: float = 0.02,
                     fixed=None, thin: int = 5, record_every: int = 1) -> dict:
    """One chain. `arm` is the ONLY thing that differs between the two conditions.

    `u_by_skill` is held fixed in both arms: this comparison is about what the block score
    contributes to segmentation and skill labelling, so moving `U` in one arm and not the
    other would confound it. The full arm therefore scores at the supplied `U` throughout,
    and the baseline ignores `U` entirely.
    """
    from hpop.mcmc_original.full_latent_constants import (FIXED_BETA, FIXED_LAMBDA_BACK,
                                                          FIXED_LAMBDA_REP, FIXED_OMEGA,
                                                          FIXED_RHO_0)

    rng = np.random.default_rng(int(seed))
    n_skills = int(model.n_skills)

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

    tables = ArmTables(arm, model, role_maps, epsilon)
    began = time.perf_counter()
    kept = {"labels": [], "n_segments": [], "boundaries": []}
    moved = 0

    for sweep in range(int(sweeps)):
        validate_pi_p(state, model)
        tables.refresh(state)
        draw = ffbs_segmentation_draw(model, state, tables, rng)
        state.segmentations = tuple(segmentation_of(key) for key in draw["keys"])
        tables.mark_stale()
        validate_paths(state, model)
        gibbs_pi_p(state, model, rng)
        moved += int(draw["movement"]["states_changed"])

        if sweep >= int(warmup) and (sweep - int(warmup)) % int(thin) == 0:
            kept["labels"].append([[int(s.skill) for s in seg.segments]
                                   for seg in state.segmentations])
            kept["n_segments"].append([len(seg.segments)
                                       for seg in state.segmentations])
            kept["boundaries"].append([[int(s.end) for s in seg.segments[:-1]]
                                       for seg in state.segmentations])
        state.iteration += 1

    return {
        "arm": arm, "chain": int(chain), "seed": int(seed),
        "sweeps": int(sweeps), "warmup": int(warmup), "thin": int(thin),
        "retained_draws": len(kept["labels"]),
        "seconds": time.perf_counter() - began,
        "ffbs_states_changed_total": moved,
        "structure_recovery": "NOT APPLICABLE" if arm == SUPPORT_ONLY else "available",
        "u_held_fixed": True,
        "draws": kept,
        "final_pi": state.pi.tolist(),
        "final_transition": state.transition.tolist(),
    }
