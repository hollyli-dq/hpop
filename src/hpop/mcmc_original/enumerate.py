"""Exhaustive enumeration of legal segmentations and the exact posterior.

The toy traces are short enough that every legal boundary configuration and every
skill labelling can be listed outright. That is the whole point: the samplers in
later stages are checked against this, not against plausibility.

A segmentation is *legal* when it tiles ``[0, T)`` with contiguous half-open blocks
(guaranteed by :class:`~hpop.mcmc_original.types.Segmentation`) and every block is
**support-compatible** with its assigned skill — its CPA multiset exactly equals that
skill's label set. Support compatibility depends only on the fixed label sets, never
on ``U``, so a trace's legal state list is computed once and reused across all of
MCMC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hpop.mcmc_original.targets import (
    log_boundary_prior,
    log_path_prior,
    logsumexp,
    normalize_log_weights,
)
from hpop.mcmc_original.types import Segment, Segmentation

__all__ = [
    "enumerate_segmentations",
    "TraceStates",
    "build_trace_states",
    "exact_state_table",
    "boundary_marginals_from_probs",
    "internal_cuts",
    "enumerate_all_labelled_states",
]


def internal_cuts(segmentation: Segmentation) -> frozenset[int]:
    """The internal cut positions of a segmentation (the final end is not a cut)."""
    return frozenset(seg.end for seg in segmentation.segments[:-1])


def enumerate_segmentations(x, skills) -> tuple[Segmentation, ...]:
    """Every support-compatible complete segmentation of ``x``.

    Args:
        x: the observed CPA trace.
        skills: skill templates, indexed by skill id.

    Returns:
        All legal segmentations, ordered deterministically by their segment tuples.
    """
    from hpop.mcmc_original.toy import map_cpa_block_to_roles

    x = tuple(int(v) for v in x)
    total = len(x)
    if total == 0:
        raise ValueError("cannot segment an empty trace")

    out: list[Segmentation] = []
    prefix: list[Segment] = []

    def walk(position: int) -> None:
        if position == total:
            out.append(Segmentation(segments=tuple(prefix)))
            return
        for skill_id, template in enumerate(skills):
            end = position + template.n_roles
            if end > total:
                continue
            if map_cpa_block_to_roles(x[position:end], template.cpa_labels) is None:
                continue
            prefix.append(Segment(position, end, skill_id))
            walk(end)
            prefix.pop()

    walk(0)
    return tuple(
        sorted(out, key=lambda s: tuple((g.start, g.end, g.skill) for g in s.segments))
    )


@dataclass
class TraceStates:
    """One trace plus its complete finite list of legal segmentations.

    ``descriptors`` caches, per state, the ``(skill_id, permutation_index)`` of each
    segment. Evaluating a state's emission term is then a few table lookups, which is
    what keeps the joint sampler affordable.
    """

    x: tuple[int, ...]
    segmentations: tuple[Segmentation, ...]
    descriptors: tuple[tuple[tuple[int, int], ...], ...]
    paths: tuple[tuple[int, ...], ...]
    cuts: tuple[frozenset[int], ...]
    log_boundary: np.ndarray
    true_cut: int | None = None
    true_path: tuple[int, ...] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return len(self.segmentations)

    @property
    def length(self) -> int:
        return len(self.x)

    def log_targets(
        self,
        tables,
        log_pi: np.ndarray,
        log_transition: np.ndarray | None = None,
    ) -> np.ndarray:
        """Unnormalised log target of every legal state, given per-skill log tables."""
        out = np.empty(self.n_states, dtype=float)
        for s, descriptor in enumerate(self.descriptors):
            emission = 0.0
            for skill_id, perm_idx in descriptor:
                emission += float(tables[skill_id][perm_idx])
            out[s] = (
                self.log_boundary[s]
                + log_path_prior(self.paths[s], log_pi, log_transition)
                + emission
            )
        return out


def build_trace_states(
    x,
    skills,
    evaluators,
    delta_b: float,
    true_cut: int | None = None,
    true_path: tuple[int, ...] | None = None,
) -> TraceStates:
    """Enumerate a trace's legal states and precompute everything U-independent."""
    x = tuple(int(v) for v in x)
    segmentations = enumerate_segmentations(x, skills)
    if not segmentations:
        raise ValueError(f"no support-compatible segmentation exists for trace {x}")

    descriptors = []
    paths = []
    cuts = []
    log_boundary = []
    for seg in segmentations:
        descriptor = []
        for g in seg.segments:
            idx = evaluators[g.skill].index_of_block(x[g.start : g.end])
            if idx is None:  # pragma: no cover - enumeration already guarantees this
                raise AssertionError("enumerated an incompatible segment")
            descriptor.append((g.skill, idx))
        descriptors.append(tuple(descriptor))
        paths.append(tuple(g.skill for g in seg.segments))
        cuts.append(internal_cuts(seg))
        log_boundary.append(log_boundary_prior(len(x), len(seg.segments), delta_b))

    return TraceStates(
        x=x,
        segmentations=segmentations,
        descriptors=tuple(descriptors),
        paths=tuple(paths),
        cuts=tuple(cuts),
        log_boundary=np.array(log_boundary, dtype=float),
        true_cut=true_cut,
        true_path=true_path,
    )


def exact_state_table(
    trace_states: TraceStates,
    tables,
    log_pi: np.ndarray,
    log_transition: np.ndarray | None = None,
) -> dict:
    """Exact posterior over a trace's legal segmentations.

    Returns a dict with the unnormalised log targets, the normalised probabilities
    (which sum to 1 by construction of :func:`normalize_log_weights`), and the
    log normaliser.
    """
    log_targets = trace_states.log_targets(tables, log_pi, log_transition)
    probs = normalize_log_weights(log_targets)
    return {
        "segmentations": trace_states.segmentations,
        "paths": trace_states.paths,
        "cuts": trace_states.cuts,
        "log_targets": log_targets,
        "probs": probs,
        "log_normaliser": logsumexp(log_targets),
    }


def boundary_marginals_from_probs(
    trace_states: TraceStates, probs: np.ndarray
) -> np.ndarray:
    """``P(cut at t | x)`` for every internal position ``t = 1, ..., T-1``."""
    out = np.zeros(trace_states.length - 1, dtype=float)
    for state_cuts, p in zip(trace_states.cuts, probs):
        for t in state_cuts:
            out[t - 1] += p
    return out


def enumerate_all_labelled_states(x, skills) -> int:
    """Count every boundary configuration x labelling, ignoring support checks.

    Used only to demonstrate how much the support constraint prunes: there are
    ``2^(T-1)`` boundary configurations and ``K^L`` labellings of each.
    """
    total = len(x)
    n_skills = len(skills)
    count = 0
    for mask in range(1 << (total - 1)):
        n_segments = 1 + bin(mask).count("1")
        count += n_skills**n_segments
    return count
