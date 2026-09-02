"""Invocation exposure: the admissibility quantity that replaced an impossible one.

An early design required every role's FIRST-step emission probability at `q0 = 0` to
exceed `5 eps / m`. That is unsatisfiable for any role with predecessors: feasibility
gives it zero non-uniform weight before its predecessors have appeared, so exactly the
partial orders with real dependency edges would have been rejected. The registered
replacement is **full invocation exposure**,

    eta_kr = E[ #visits to role r during one complete invocation of skill k ]

with the expectation over the registered width law and the skill's own recurrent
dynamics. A role deep in the poset has small first-step probability but perfectly healthy
invocation exposure -- unless the geometry genuinely starves it, which is exactly what
this admissibility criterion exists to exclude at the LIBRARY draw (where rejection with
recorded reasons is the registered mechanism), never on realised corpora.

`eta` is estimated by prior-predictive probes: a registered, fixed number of simulated
invocations per skill through the same segment sampler the corpus generator uses. The
probe count and floor are part of the library's recorded provenance.
"""

from __future__ import annotations

import numpy as np

__all__ = ["invocation_exposure", "exposure_admissible"]


def invocation_exposure(u_skill, params, delta_b: float, min_width: int, max_width: int,
                        trace_length: int, n_probes: int, seed: int) -> np.ndarray:
    """Monte-Carlo `eta_kr` for one skill: mean visits per role per invocation.

    Widths are drawn from the same registered law the generator uses
    (`sample_segmentation_widths` over a probe trace, so the width marginal matches the
    corpus rather than an idealised distribution), and segments from the same
    `sample_recurrent_rfs_sequence`.
    """
    from hpop.mcmc_original.matched_segmentation_prior import (
        sample_segmentation_widths, width_sampling_tables)
    from hpop.mcmc_original.recurrent_rfs import sample_recurrent_rfs_sequence

    def probe_rng(probe_index: int, component: int) -> np.random.Generator:
        # its own namespace; the corpus component_rng only accepts registered splits
        return np.random.default_rng(np.random.SeedSequence(
            entropy=int(seed), spawn_key=(int(probe_index), int(component))))

    u_skill = np.asarray(u_skill, dtype=float)
    m = u_skill.shape[0]
    tables = width_sampling_tables(trace_length, delta_b, min_width, max_width)
    visits = np.zeros(m, dtype=float)
    n_segments = 0
    probe = 0
    while n_segments < int(n_probes):
        widths = sample_segmentation_widths(
            probe_rng(probe, 0), trace_length, delta_b, min_width, max_width, tables)
        for block, width in enumerate(widths):
            if n_segments >= int(n_probes):
                break
            roles = sample_recurrent_rfs_sequence(
                probe_rng(probe, 1 + block), int(width), u_skill, params)
            for r in roles:
                visits[int(r)] += 1.0
            n_segments += 1
        probe += 1
    return visits / float(n_segments)


def exposure_admissible(u, params, delta_b: float, min_width: int, max_width: int,
                        trace_length: int, n_probes: int, floor: float,
                        seed: int) -> tuple:
    """Admissibility over all skills: every role's eta must clear the registered floor.

    Returns `(ok, reasons, eta)` with `eta` shaped `(K, m)` so the draw record can carry
    the measured exposures, accepted and rejected alike.
    """
    u = np.asarray(u, dtype=float)
    reasons, etas = [], []
    for k in range(u.shape[0]):
        eta = invocation_exposure(u[k], params, delta_b, min_width, max_width,
                                  trace_length, n_probes, seed + 1000 * k)
        etas.append(eta)
        starved = np.flatnonzero(eta < float(floor))
        if starved.size:
            reasons.append(
                f"skill {k}: roles {starved.tolist()} have invocation exposure "
                f"{np.round(eta[starved], 3).tolist()} below the floor {floor}")
    return (not reasons), reasons, np.stack(etas)


def pair_evidence_probe(u_skill, params, delta_b: float, min_width: int, max_width: int,
                        trace_length: int, n_probes: int, seed: int) -> dict:
    """Prior-predictive PAIR statistics for one skill: what the eta floor cannot see.

    The evidence a corpus carries about an order is pairwise. For a true edge (a, b) the
    unit of evidence is an invocation containing BOTH roles; for a true incomparable pair
    it is witnessing the pair in BOTH orders across invocations, since one order alone is
    indistinguishable from an edge. Marginal exposure is necessary but nowhere near
    sufficient -- measured directly: skills with healthy eta still failed the corpus
    evidence standard, and skills with poor eta passed it.

    Returns per-pair co-occurrence probabilities and, for each unordered pair, the
    probability of each first-appearance direction.
    """
    from hpop.mcmc_original.matched_segmentation_prior import (
        sample_segmentation_widths, width_sampling_tables)
    from hpop.mcmc_original.recurrent_rfs import sample_recurrent_rfs_sequence
    from hpop.mcmc_original.latent_poset import precedence_from_u

    def probe_rng(probe_index: int, component: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence(
            entropy=int(seed), spawn_key=(int(probe_index), int(component))))

    u_skill = np.asarray(u_skill, dtype=float)
    m = u_skill.shape[0]
    tables = width_sampling_tables(trace_length, delta_b, min_width, max_width)
    cooc = np.zeros((m, m), dtype=float)          # both appear in the invocation
    first = np.zeros((m, m), dtype=float)         # i's first appearance precedes j's
    n_segments = 0
    probe = 0
    while n_segments < int(n_probes):
        widths = sample_segmentation_widths(
            probe_rng(probe, 0), trace_length, delta_b, min_width, max_width, tables)
        for block, width in enumerate(widths):
            if n_segments >= int(n_probes):
                break
            roles = sample_recurrent_rfs_sequence(
                probe_rng(probe, 1 + block), int(width), u_skill, params)
            position = {}
            for t, r in enumerate(roles):
                position.setdefault(int(r), t)
            present = sorted(position)
            for x in range(len(present)):
                for y in range(x + 1, len(present)):
                    i, j = present[x], present[y]
                    cooc[i, j] += 1.0
                    cooc[j, i] += 1.0
                    if position[i] < position[j]:
                        first[i, j] += 1.0
                    else:
                        first[j, i] += 1.0
            n_segments += 1
        probe += 1
    closure = np.asarray(precedence_from_u(u_skill), dtype=bool)
    return {"p_cooc": cooc / n_segments, "p_first": first / n_segments,
            "closure": closure, "n_segments": int(n_segments)}


def evidence_admissible(u, params, delta_b: float, min_width: int, max_width: int,
                        trace_length: int, n_probes: int, seed: int,
                        expected_instances: float, edge_min_expected: float,
                        incomp_min_expected_each_way: float) -> tuple:
    """Identifiability-aware admissibility: every relation expected-identifiable.

    Floors are DERIVED, not tuned: with `expected_instances` invocations of each skill in
    the registered corpus, a true edge (a, b) is required to have

        expected_instances * P(a and b co-occur)   >=  edge_min_expected

    and a true incomparable pair to have, in EACH direction,

        expected_instances * P(that direction witnessed)  >=  incomp_min_expected_each_way

    so admission means "at the registered corpus size, the generating process is expected
    to supply the evidence the recovery standard requires". Applied at the master-library
    draw only, with every rejection reason recorded -- never to realised corpora.
    """
    u = np.asarray(u, dtype=float)
    reasons, profiles = [], []
    for k in range(u.shape[0]):
        stats = pair_evidence_probe(u[k], params, delta_b, min_width, max_width,
                                    trace_length, n_probes, seed + 1000 * k)
        closure, p_cooc, p_first = stats["closure"], stats["p_cooc"], stats["p_first"]
        m = closure.shape[0]
        worst_edge, worst_incomp = np.inf, np.inf
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                if closure[i, j]:
                    worst_edge = min(worst_edge,
                                     expected_instances * p_cooc[i, j])
                elif i < j and not closure[j, i]:
                    worst_incomp = min(worst_incomp,
                                       expected_instances * p_first[i, j],
                                       expected_instances * p_first[j, i])
        profiles.append({"worst_edge_expected": None if np.isinf(worst_edge)
                         else float(worst_edge),
                         "worst_incomp_expected_each_way": None if np.isinf(worst_incomp)
                         else float(worst_incomp)})
        if np.isfinite(worst_edge) and worst_edge < float(edge_min_expected):
            reasons.append(f"skill {k}: weakest true edge expects "
                           f"{worst_edge:.1f} co-occurrences < {edge_min_expected}")
        if np.isfinite(worst_incomp) and \
                worst_incomp < float(incomp_min_expected_each_way):
            reasons.append(f"skill {k}: weakest incomparable pair expects "
                           f"{worst_incomp:.1f} witnesses in one direction < "
                           f"{incomp_min_expected_each_way}")
    return (not reasons), reasons, profiles
