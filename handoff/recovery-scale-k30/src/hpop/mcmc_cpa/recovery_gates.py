"""Truth-free, PERMUTATION-INVARIANT convergence gates for the recovery experiment.

Skill labels are not identified: two chains can hold the same posterior with the skills
relabelled, and any per-skill or per-edge cross-chain statistic then reports disagreement
that is not there. (The v2 pilot's edge-level R-hat had exactly this defect.) Every
statistic below is invariant under relabelling by construction:

    sorted skill-usage profile        occurrences per skill, sorted descending
    sorted relation counts            closure size per skill, sorted descending
    total relation count              a scalar
    sorted pi                          the initial law, sorted
    transition spectrum               sorted |eigenvalues| of P (similarity-invariant)
    co-clustering probes              same-skill indicators for FIXED occurrence pairs
    canonical library digest          sorted per-skill closure digests

R-hat and ESS are computed across the four dispersed chains of ONE (replicate, K) cell,
on the LAST HALF of the draws so far (the registered warm-up rule for run-to-convergence:
the window grows with the chain, no fixed warm-up needs declaring). Replicates are never
pooled. The verdict is frozen per cell BEFORE the sealed truth is opened, and a chain
that exhausts the cap without passing is INFERENCE FAIL at that K -- never a model claim.
"""

from __future__ import annotations

import hashlib

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.stage6b_mcmc_diagnostics import (bulk_ess,
                                                         rank_normalized_split_rhat)

from .recovery_regime import REGIME

__all__ = ["chain_statistics", "cocluster_probe_pairs", "evaluate_cell"]


def cocluster_probe_pairs(trace_lengths, n_pairs: int, seed: int) -> list:
    """Registered probe set: fixed (trace, position) pairs, deterministic from the seed.

    The pairs are chosen once per cell from the diagnostic stream and shared by all
    chains and all checks, so the statistic is comparable across chains and rounds.
    """
    rng = np.random.default_rng(np.random.SeedSequence(entropy=int(seed),
                                                       spawn_key=(424242,)))
    lengths = [int(x) for x in trace_lengths]
    pairs = []
    for _ in range(int(n_pairs)):
        a_trace = int(rng.integers(len(lengths)))
        b_trace = int(rng.integers(len(lengths)))
        pairs.append((a_trace, int(rng.integers(lengths[a_trace])),
                      b_trace, int(rng.integers(lengths[b_trace]))))
    return pairs


def _skill_at(labels, boundaries, trace: int, position: int) -> int:
    ends = list(boundaries[trace]) + [10**9]
    start = 0
    for seg, end in enumerate(ends):
        if position < end:
            return int(labels[trace][seg])
        start = end
    return int(labels[trace][-1])


def chain_statistics(draws: dict, probe_pairs: list) -> dict:
    """Per-draw permutation-invariant series for one chain (its full history)."""
    usage, relation_sorted, relation_total, cocluster = [], [], [], []
    for labels, boundaries in zip(draws["labels"], draws["boundaries"]):
        counts = {}
        for trace_labels, trace_bounds in zip(labels, boundaries):
            widths_end = list(trace_bounds)
            for skill in trace_labels:
                counts[skill] = counts.get(skill, 0) + 1
        usage.append(sorted(counts.values(), reverse=True))
        same = sum(1 for (ta, pa, tb, pb) in probe_pairs
                   if _skill_at(labels, boundaries, ta, pa)
                   == _skill_at(labels, boundaries, tb, pb))
        cocluster.append(same / max(len(probe_pairs), 1))
    for u in draws["u"]:
        u = np.asarray(u, dtype=float)
        sizes = sorted((int(np.asarray(precedence_from_u(u[k])).sum())
                        for k in range(u.shape[0])), reverse=True)
        relation_sorted.append(sizes)
        relation_total.append(sum(sizes))
    digests = []
    for u in draws["u"]:
        u = np.asarray(u, dtype=float)
        per_skill = sorted(
            hashlib.sha256(np.asarray(precedence_from_u(u[k])).tobytes()).hexdigest()
            for k in range(u.shape[0]))
        digests.append(hashlib.sha256("".join(per_skill).encode()).hexdigest()[:16])
    return {"usage": usage, "cocluster": cocluster,
            "relation_sorted": relation_sorted, "relation_total": relation_total,
            "pi_sorted": draws.get("pi_sorted", []),
            "p_spectrum": draws.get("p_spectrum", []),
            "canonical_digests": digests}


def _last_half(series):
    n = len(series)
    return series[n // 2:]


def _stack_last_half(per_chain, key, component=None):
    rows = []
    for c in per_chain:
        vals = _last_half(c[key])
        if component is None:
            rows.append([float(v) for v in vals])
        else:
            rows.append([float(v[component]) if len(v) > component else np.nan
                         for v in vals])
    n = min(len(r) for r in rows)
    return np.array([r[:n] for r in rows], dtype=float)


def evaluate_cell(per_chain_stats: list, acceptance_retained: float | None) -> dict:
    """The frozen gate for one (replicate, K) cell from its four chains' statistics."""
    failures, numbers = [], {}

    def check(name, chains_matrix):
        if chains_matrix.shape[1] < 8 or np.isnan(chains_matrix).any():
            failures.append(f"{name}: insufficient draws")
            return
        if float(chains_matrix.std()) == 0.0:
            numbers[f"rhat_{name}"] = None            # consensus-constant: not a failure
            return
        rhat = rank_normalized_split_rhat(chains_matrix)["rhat"]
        ess = bulk_ess(chains_matrix)
        numbers[f"rhat_{name}"] = float(rhat)
        numbers[f"ess_{name}"] = float(ess)
        if rhat > REGIME.RHAT_MAX:
            failures.append(f"{name}: R-hat {rhat:.3f} > {REGIME.RHAT_MAX}")
        if ess < REGIME.ESS_MIN:
            failures.append(f"{name}: ESS {ess:.1f} < {REGIME.ESS_MIN}")

    check("relation_total", _stack_last_half(per_chain_stats, "relation_total"))
    k = len(per_chain_stats[0]["relation_sorted"][0]) if \
        per_chain_stats[0]["relation_sorted"] else 0
    for component in range(min(k, 5)):                # top-5 sorted relation counts
        check(f"relation_sorted_{component}",
              _stack_last_half(per_chain_stats, "relation_sorted", component))
    check("cocluster", _stack_last_half(per_chain_stats, "cocluster"))
    for component in range(2):
        check(f"usage_{component}",
              _stack_last_half(per_chain_stats, "usage", component))
        check(f"pi_sorted_{component}",
              _stack_last_half(per_chain_stats, "pi_sorted", component))
        check(f"p_spectrum_{component}",
              _stack_last_half(per_chain_stats, "p_spectrum", component))

    modal = []
    for c in per_chain_stats:
        window = _last_half(c["canonical_digests"])
        if window:
            values, counts = np.unique(window, return_counts=True)
            modal.append(str(values[np.argmax(counts)]))
    numbers["modal_digests"] = modal
    if len(set(modal)) > 1:
        failures.append(f"chains' modal canonical libraries differ: {len(set(modal))} "
                        f"distinct")

    if acceptance_retained is not None:
        lo, hi = REGIME.ACCEPT_WINDOW
        numbers["u_acceptance_window"] = float(acceptance_retained)
        if not lo <= acceptance_retained <= hi:
            failures.append(f"U acceptance {acceptance_retained:.3f} outside "
                            f"[{lo}, {hi}]")

    return {"passes": not failures, "failures": failures, "numbers": numbers,
            "window_rule": "last half of all draws so far",
            "permutation_invariant": True}
