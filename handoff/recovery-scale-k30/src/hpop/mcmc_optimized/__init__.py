"""An exact, faster backend for the FULL-LATENT segmental sampler.

This package exists because `hpop.mcmc_original` is sealed. Three separate gates pin the
bytes of the reference implementation:

  * `test_the_step7a_engine_is_byte_identical_to_the_frozen_checkpoint` -- a SHA256 on
    `semi_markov_ffbs.py`, whose docstring reads "No speed may be bought by editing the
    validated FFBS engine";
  * `SHARED_SOURCES` in `run_matched_condition_c_prime_formal.py`, which requires ten
    files to be unchanged since the Condition C launch commit;
  * `SOURCE_DIGESTS` in `_midrun_vendored_registered.py`.

So nothing here edits `mcmc_original`. This package imports it, subclasses it, and
mirrors it. **The reference implementation remains the numerical oracle**: every routine
below is checked against it, and where they disagree the reference is right by definition.

The interface is deliberately parallel to the reference:

    reference                                     optimized
    ---------                                     ---------
    semi_markov_ffbs.forward                      forward.forward_dispatch
    (none)                                        forward.forward_batched_group
    recurrent_joint_ffbs_mcmc.FFBSBlockTables     tables.HashCachedFFBSBlockTables
    collapsed_u_likelihood.CollapsedULikelihood   likelihood.BatchedCollapsedULikelihood
    recurrent_joint_ffbs_mcmc.ffbs_segmentation_draw
                                                  segmentation.ffbs_segmentation_draw
    matched_full_latent.FullLatentSampler         full_latent.OptimizedFullLatentSampler
    matched_full_latent.full_latent_sweep_once    full_latent.sweep_once

Exactness. Every optimisation targets the same distribution. Only the H-based emission
cache is bit-identical (it skips recomputation and returns the bits a rebuild would have
produced). The inline reduction, the factorised recursion and the batched recursion
re-associate floating-point sums, so alpha moves by ~1e-14 and a categorical draw within
1e-14 of a boundary can fall the other way. The posterior is unchanged; a realised chain
is not reproducible against one drawn with the reference. New launches only.
"""

from .flags import COUNTERS, FLAGS, OptimizationFlags
from .forward import (forward_batched_group, forward_dispatch, forward_factorised,
                      forward_with_inline_reduction, inline_logsumexp)
from .full_latent import OptimizedFullLatentSampler, sweep_once
from .likelihood import BatchedCollapsedULikelihood
from .segmentation import ffbs_segmentation_draw
from .tables import HashCachedFFBSBlockTables

__all__ = ["COUNTERS", "FLAGS", "OptimizationFlags", "inline_logsumexp",
           "forward_with_inline_reduction", "forward_factorised",
           "forward_batched_group", "forward_dispatch", "HashCachedFFBSBlockTables",
           "BatchedCollapsedULikelihood", "ffbs_segmentation_draw",
           "OptimizedFullLatentSampler", "sweep_once"]
