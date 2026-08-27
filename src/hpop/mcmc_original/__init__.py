"""Original latent partial-order HPOP model (clean re-implementation).

This package is deliberately parallel to and independent of ``hpop.inference``.
Nothing here imports from the older implementation, and nothing there imports
from here.

The single statistical object defined at this stage is the map

    U_k -> h(U_k)

where U_k in R^{m_k x d} is the latent embedding of the m_k roles of skill k in
a d-dimensional product order, and h(U_k) is the strict partial order induced by
coordinate-wise dominance:

    i > j   iff   U_k[i, r] > U_k[j, r]  for every r = 1, ..., d.

Order structure is therefore never represented as learnable edges: every U
induces a valid strict partial order by construction.

Given that order, the static BPOP frontier-softmax likelihood scores an observed
execution of the skill's roles:

    p(y | U_k, beta, epsilon)

Modules:
    types          immutable dataclasses (Segment, Segmentation, SkillTemplate)
    latent_poset   the U -> h(U) mapping and precedence queries
    static_bpop    frontier, successor utility, and the static BPOP likelihood
"""

from hpop.mcmc_original.latent_poset import (
    incomparable,
    precedence_from_u,
    predecessors,
    successors,
)
from hpop.mcmc_original.static_bpop import (
    all_permutation_probabilities,
    bpop_likelihood,
    bpop_log_likelihood,
    bpop_step_probabilities,
    frontier,
    remaining_successor_count,
    sample_bpop_sequence,
    successor_utility,
)
from hpop.mcmc_original.types import Segment, Segmentation, SkillTemplate

__all__ = [
    "Segment",
    "Segmentation",
    "SkillTemplate",
    "precedence_from_u",
    "predecessors",
    "successors",
    "incomparable",
    "frontier",
    "remaining_successor_count",
    "successor_utility",
    "bpop_step_probabilities",
    "bpop_log_likelihood",
    "bpop_likelihood",
    "all_permutation_probabilities",
    "sample_bpop_sequence",
]
