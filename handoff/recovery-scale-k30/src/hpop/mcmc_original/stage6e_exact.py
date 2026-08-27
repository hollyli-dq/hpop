"""Stage 6E1 — exact enumeration of the unknown-boundary segmentation posterior.

Everything here is a *reference*. Nothing in this module is used by the production
sampler, and nothing in it calls the move kernel, the Hastings ratio, or any acceptance
helper — that separation is what makes the Stage 6E1A comparison meaningful.

## Two routes to the same normalising constant

`enumerate_states` walks the legal state space directly: a segmentation of `[0, J)` is a
sequence of widths in `[min_w, max_w]` summing to `J`, and a labelling is any assignment
with no two adjacent segments alike (because `P` has a zero diagonal). Summing
`exp(log w)` over that list gives `log Z_enum`.

`log_evidence_forward` computes the same constant by a **forward recursion over
positions** and never materialises a state:

    f[t, k] = log-sum of the weight of every legal segmentation of x[0:t] whose last
              segment carries skill k

    f[w, k]  =  log pi[k] + score(0, w, k) + (w - 1) log(1 - delta_B)
    f[t, k]  =  logsumexp over (t', k' != k) of
                f[t', k'] + log delta_B + log P[k', k]
                          + score(t', t, k) + (t - t' - 1) log(1 - delta_B)
    log Z    =  logsumexp_k f[J, k]

The boundary prior decomposes because `L - 1` cuts each contribute `log delta_B` and a
segment of width `w` contributes `w - 1` non-cut internal positions, and
`sum_l (w_l - 1) = J - L` — exactly the exponent `targets.log_boundary_prior` uses.

The two routes share no code path: one enumerates and sums, the other recurses over
positions. §8's `|log Z_enum - log Z_forward| < 1e-10` is therefore a real cross-check of
the registered weight rather than a restatement of it.

**The forward recursion is not a sampler and is not FFBS.** It is used only to
cross-check `log Z`, and Stage 6E1B draws its reference `(S, z)` by sampling from the
*enumerated* conditional, as §9 requires. The production segmentation update remains the
registered local move kernel throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from hpop.mcmc_original.fast_segmentation_kernel import spans_of
from hpop.mcmc_original.stage6e_frozen import (
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH,
)

__all__ = [
    "enumerate_states", "state_log_weights", "exact_posterior",
    "log_evidence_forward", "boundary_marginals", "occurrence_label_marginals",
    "segment_count_distribution", "expected_transition_counts", "labelled_segment_marginals",
    "total_variation", "sample_state",
]


# ------------------------------------------------------------------------ enumeration
def _width_sequences(remaining: int, min_w: int, max_w: int) -> list:
    """Every ordered composition of `remaining` into parts in `[min_w, max_w]`."""
    if remaining == 0:
        return [()]
    out = []
    for w in range(min_w, min(max_w, remaining) + 1):
        if remaining - w != 0 and remaining - w < min_w:
            continue                       # the tail could never be completed legally
        for tail in _width_sequences(remaining - w, min_w, max_w):
            out.append((w,) + tail)
    return out


def _labellings(length: int, n_skills: int) -> list:
    """Every label sequence of `length` with no two adjacent alike (zero diagonal in P)."""
    if length == 0:
        return [()]
    out = [(k,) for k in range(n_skills)]
    for _ in range(length - 1):
        nxt = []
        for prefix in out:
            for k in range(n_skills):
                if k != prefix[-1]:
                    nxt.append(prefix + (k,))
        out = nxt
    return out


def enumerate_states(trace_length: int, n_skills: int,
                     min_width: int = MIN_BLOCK_WIDTH,
                     max_width: int = MAX_BLOCK_WIDTH) -> list:
    """Every legal `((end, skill), ...)` key for one trace, built without the kernel.

    Coverage, contiguity and positivity are structural: the key is the running sum of a
    composition of `J`, so gaps and overlaps cannot be represented. Widths are bounded by
    construction and adjacent labels differ by construction.
    """
    states = []
    for widths in _width_sequences(int(trace_length), int(min_width), int(max_width)):
        ends, running = [], 0
        for w in widths:
            running += w
            ends.append(running)
        for labels in _labellings(len(widths), int(n_skills)):
            states.append(tuple(zip(ends, labels)))
    return states


# --------------------------------------------------------------------------- weights
def state_log_weights(states, trace_index: int, trace_length: int, scorer,
                      log_pi, log_transition, delta_b: float = DELTA_B) -> np.ndarray:
    """`log w(S, z)` for every enumerated state, by the registered decomposition."""
    log_pi = np.asarray(log_pi, dtype=float)
    log_transition = np.asarray(log_transition, dtype=float)
    log_db = math.log(delta_b)
    log_1mdb = math.log1p(-delta_b)
    out = np.empty(len(states))
    for i, key in enumerate(states):
        spans = spans_of(key)
        total = float(log_pi[spans[0][2]])
        for start, end, skill in spans:
            total += scorer.score(trace_index, start, end, skill)
        total += (len(spans) - 1) * log_db + (trace_length - len(spans)) * log_1mdb
        for (_, _, left), (_, _, right) in zip(spans[:-1], spans[1:]):
            total += float(log_transition[left, right])
        out[i] = total
    return out


def exact_posterior(states, log_weights) -> dict:
    log_weights = np.asarray(log_weights, dtype=float)
    log_z = float(logsumexp(log_weights))
    probability = np.exp(log_weights - log_z)
    return {"states": list(states), "log_weights": log_weights, "log_evidence": log_z,
            "probability": probability}


def log_evidence_forward(trace_index: int, trace_length: int, n_skills: int, scorer,
                         log_pi, log_transition, delta_b: float = DELTA_B,
                         min_width: int = MIN_BLOCK_WIDTH,
                         max_width: int = MAX_BLOCK_WIDTH) -> float:
    """`log Z` by forward recursion over positions — the independent implementation.

    Never enumerates a state. Used only as §8's cross-check; it is not a sampler.
    """
    log_pi = np.asarray(log_pi, dtype=float)
    log_transition = np.asarray(log_transition, dtype=float)
    log_db = math.log(delta_b)
    log_1mdb = math.log1p(-delta_b)
    J = int(trace_length)
    f = np.full((J + 1, int(n_skills)), -np.inf)

    for w in range(min_width, min(max_width, J) + 1):
        for k in range(n_skills):
            f[w, k] = (log_pi[k] + scorer.score(trace_index, 0, w, k)
                       + (w - 1) * log_1mdb)
    for t in range(min_width + 1, J + 1):
        for k in range(n_skills):
            terms = [f[t, k]] if np.isfinite(f[t, k]) else []
            for w in range(min_width, min(max_width, t) + 1):
                previous = t - w
                if previous < min_width:
                    continue
                block = scorer.score(trace_index, previous, t, k)
                for other in range(n_skills):
                    if other == k or not np.isfinite(f[previous, other]):
                        continue
                    terms.append(f[previous, other] + log_db
                                 + float(log_transition[other, k]) + block
                                 + (w - 1) * log_1mdb)
            f[t, k] = float(logsumexp(terms)) if terms else -np.inf
    return float(logsumexp(f[J]))


# -------------------------------------------------------------------------- marginals
def boundary_marginals(states, probability, trace_length: int) -> np.ndarray:
    """`p(a cut falls at internal position t)` for `t = 1 .. J-1`."""
    out = np.zeros(int(trace_length) - 1)
    for key, p in zip(states, probability):
        for end, _ in key[:-1]:
            out[end - 1] += p
    return out


def occurrence_label_marginals(states, probability, trace_length: int,
                               n_skills: int) -> np.ndarray:
    """`p(z_t = k)` per occurrence `t` — the per-position skill label marginal."""
    out = np.zeros((int(trace_length), int(n_skills)))
    for key, p in zip(states, probability):
        for start, end, skill in spans_of(key):
            out[start:end, skill] += p
    return out


def labelled_segment_marginals(states, probability) -> dict:
    """`p(segment (a, b) carries skill k)` over every labelled segment that appears."""
    out: dict = {}
    for key, p in zip(states, probability):
        for start, end, skill in spans_of(key):
            out[(start, end, skill)] = out.get((start, end, skill), 0.0) + float(p)
    return out


def segment_count_distribution(states, probability, max_segments: int = None) -> np.ndarray:
    counts = [len(k) for k in states]
    size = int(max_segments if max_segments is not None else max(counts)) + 1
    out = np.zeros(size)
    for c, p in zip(counts, probability):
        out[c] += p
    return out


def expected_transition_counts(states, probability, n_skills: int) -> np.ndarray:
    out = np.zeros((int(n_skills), int(n_skills)))
    for key, p in zip(states, probability):
        labels = [k for _, k in key]
        for a, b in zip(labels[:-1], labels[1:]):
            out[a, b] += float(p)
    return out


def total_variation(p, q) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.abs(p - q).sum())


def sample_state(states, probability, rng) -> tuple:
    """Draw one `(S, z)` from the exact enumerated conditional. Used by 6E1B only."""
    index = int(rng.choice(len(states), p=np.asarray(probability, dtype=float)))
    return states[index]
