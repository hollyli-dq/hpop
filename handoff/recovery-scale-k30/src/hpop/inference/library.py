"""Skill-library prior — the new-skill penalty (CRP / Pitman–Yor).

The global level draws each skill instance's type from a Dirichlet-process library. Collapsing the
DP gives the Chinese-restaurant-process predictive; the log-prior of the type assignment is the
Ewens sampling formula (EPPF). The number of distinct skills K enters with weight log(alpha): each
NEW skill costs ~ -log(alpha), reuse of a size-n_r skill costs ~ log(n_r). One knob alpha controls
the reuse<->novelty trade-off; Pitman–Yor(alpha, d) gives power-law library growth.
"""
from __future__ import annotations

import math


def crp_logprob(counts, alpha):
    """Ewens EPPF: log p(z | alpha) for a partition with cluster sizes `counts` (n_r)."""
    K = len(counts)
    N = sum(counts)
    return (K * math.log(alpha)
            + sum(math.lgamma(n) for n in counts)        # (n_r - 1)!  via lgamma(n_r)
            + math.lgamma(alpha) - math.lgamma(N + alpha))


def crp_predictive(counts, alpha):
    """Next-assignment probabilities: (existing[r], p_new) under CRP(alpha)."""
    N = sum(counts)
    existing = [n / (N + alpha) for n in counts]
    return existing, alpha / (N + alpha)


def pitman_yor_predictive(counts, alpha, d):
    """(existing[r], p_new) under Pitman–Yor(alpha, d), d in [0,1) — power-law libraries."""
    N = sum(counts)
    K = len(counts)
    existing = [(n - d) / (N + alpha) for n in counts]
    return existing, (alpha + d * K) / (N + alpha)


def new_skill_logpenalty(counts, alpha):
    """Δ log-prior of opening a NEW skill vs the cheapest reuse, at the current state."""
    N = sum(counts)
    p_new = alpha / (N + alpha)
    best_reuse = (max(counts) / (N + alpha)) if counts else None
    return math.log(p_new) - (math.log(best_reuse) if best_reuse else 0.0)
