"""The FULL-LATENT sampler and sweep, on the optimized backend.

`matched_full_latent.py` is pinned by `_midrun_vendored_registered.py::SOURCE_DIGESTS`, so
it is not edited. `OptimizedFullLatentSampler` subclasses `FullLatentSampler` and swaps in
the optimized tables and collapsed likelihood; `sweep_once` mirrors
`matched_full_latent.full_latent_sweep_once`, importing every piece it does not optimise so
the two cannot drift on anything but the forward path.

The kernel order is identical, and deliberately so:

    [scheduled] structural U attempt  ->  table refresh  ->  all-trace FFBS
                                      ->  pi/P Gibbs     ->  complete log target

Only two things differ from the reference: the emission table may be served from the H
cache, and the forward pass may be batched. Everything else -- the validators, the
structural kernels, the Gibbs update, the target decomposition, the rng consumption order
-- is the frozen code, called directly.
"""

from __future__ import annotations

from dataclasses import field

import numpy as np

from hpop.mcmc_original.collapsed_u_kernel import collapsed_u_mh_step, is_collapsed_sweep
from hpop.mcmc_original.matched_full_latent import (FULL_MARG, FullLatentSampler,
                                                    complete_log_target,
                                                    conditional_structural_mh_step,
                                                    gibbs_pi_p, segmentation_of,
                                                    validate_paths, validate_pi_p)
from hpop.mcmc_original.stage6e_state import Stage6EState

from .likelihood import BatchedCollapsedULikelihood
from .segmentation import ffbs_segmentation_draw
from .tables import HashCachedFFBSBlockTables


class OptimizedFullLatentSampler(FullLatentSampler):
    """`FullLatentSampler` with the optimized tables and collapsed likelihood installed."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._tables = HashCachedFFBSBlockTables(
            model=self.model, source=self.config.table_source)
        self._collapsed_lik = BatchedCollapsedULikelihood(model=self.model)


def sweep_once(state: Stage6EState, sampler: OptimizedFullLatentSampler,
               rng: np.random.Generator) -> tuple[Stage6EState, dict]:
    """One FULL-LATENT sweep. Same kernel order and same contract as the reference."""
    state = state.copy()
    sampler.fixed.assert_unchanged(state)
    validate_pi_p(state, sampler.model)
    validate_paths(state, sampler.model)
    scheduled = is_collapsed_sweep(state.iteration, sampler.config.structural_cadence)
    record = None
    order = []
    if scheduled:
        if sampler.config.arm == FULL_MARG:
            state, record = collapsed_u_mh_step(
                state, sampler.model, sampler.collapsed_likelihood, rng,
                sampler.config.structural_scale)
            order.append("marginal_U")
        else:
            state, record = conditional_structural_mh_step(state, sampler, rng)
            order.append("conditional_U")

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
