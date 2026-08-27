"""The semi-Markov forward recursion, three exact ways.

The reference is `hpop.mcmc_original.semi_markov_ffbs.forward`, which is sealed and stays
the oracle. This module never imports it for computation -- only its `ForwardChart`
dataclass and `_validate`, so the charts handed back are the same type the frozen backward
sampler already accepts.

Why each variant exists, from the measured profile of a FULL-LATENT sweep:

O1  `forward_with_inline_reduction`. scipy 1.18's `logsumexp` costs ~134 us per call
    *independent of array size* -- it checks for torch and jax arrays every time -- against
    ~7 us of arithmetic on the ~16-element arrays this recursion produces. 10,300 calls per
    sweep made that ~58% of the entire sweep. Same loop as the reference, inline reduction.

O3  `forward_factorised`. The reference computes, for every (b, k), a term for every (a, h)
    pair: O(J D K^2). Nothing inside the h-sum depends on b, so with

        r[a, k]     = LSE_h ( alpha[a, h] + logP[h, k] )      once per a, O(K^2)
        alpha[b, k] = LSE_a ( r[a, k] + score[a, b, k] + width(a, b) )   O(D) per (b, k)

    the same numbers come out of O(J K^2 + J D K). `r` is the log-domain form of
    r_a = alpha[a-1, :] @ P. `alpha[a, :]` is final for every a < b when b is processed
    (a <= b - min_width), so `r` is always available when read.

O4  `forward_batched_group`. The trace axis vectorises exactly, so a whole length class
    runs at once. This is what collapses ~10,300 tiny reductions per sweep into a few
    hundred large ones, and it is the single largest win.
"""

from __future__ import annotations

import math
import time

import numpy as np

from hpop.mcmc_original.semi_markov_ffbs import ForwardChart, _validate

from .flags import COUNTERS, FLAGS

NEG = -np.inf


# --------------------------------------------------------------------- the reductions
def inline_logsumexp(values: np.ndarray) -> float:
    """`scipy.special.logsumexp` written out, with the same shift for stability."""
    top = values.max()
    if top == NEG:
        return float(NEG)
    return float(top + np.log(np.exp(values - top).sum()))


def reduce_axis(values: np.ndarray, axis: int) -> np.ndarray:
    """The same reduction along one axis; an all -inf slice gives -inf, without warning."""
    top = np.max(values, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(top), top, 0.0)
    total = np.exp(values - safe).sum(axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        out = safe + np.log(total)
    return np.squeeze(np.where(total > 0.0, out, NEG), axis=axis)


def _predecessor_terms(alpha, b, k, scores, log_pi, log_p, log_db, log_1mdb,
                       max_width, min_width):
    """Byte-for-byte the reference recurrence, reproduced so O1 can be measured alone.

    The reference `predecessor_terms` is sealed; this is the same enumeration, and
    `test_optimized_backend_equivalence` pins the two together.
    """
    b, k = int(b), int(k)
    lowest = max(0, b - int(max_width))
    starts, prev, terms = [], [], []
    if lowest == 0 and int(min_width) <= b:
        value = (float(log_pi[k]) + float(scores[0, b, k]) + (b - 1) * log_1mdb)
        if value > NEG:
            starts.append(0)
            prev.append(-1)
            terms.append(value)
    for a in range(max(1, lowest), b - int(min_width) + 1):
        block = float(scores[a, b, k])
        if math.isinf(block) and block < 0:
            continue
        carried = block + log_db + (b - a - 1) * log_1mdb
        for h in range(alpha.shape[1]):
            previous = float(alpha[a, h])
            if previous == NEG:
                continue
            transition = float(log_p[h, k])
            if transition == NEG:
                continue
            starts.append(a)
            prev.append(h)
            terms.append(previous + carried + transition)
    return (np.asarray(starts, dtype=int), np.asarray(prev, dtype=int),
            np.asarray(terms, dtype=float))


# ------------------------------------------------------------------------- O1 and O3
def _chart(alpha, log_z, scores, log_pi, log_p, J, K, boundary_prob, max_width,
           min_width, build_seconds) -> ForwardChart:
    """`ForwardChart` is frozen, so `build_seconds` is supplied at construction."""
    if not np.isfinite(log_z):
        raise ValueError("no finite complete path: every legal segmentation of [0, J) "
                         "has weight zero under the supplied scores and priors")
    return ForwardChart(alpha=alpha, log_normalizer=float(log_z), J=int(J), K=int(K),
                        max_width=int(max_width), min_width=int(min_width),
                        boundary_prob=float(boundary_prob), log_block_scores=scores,
                        log_initial_probs=log_pi, log_transition_matrix=log_p,
                        build_seconds=float(build_seconds))


def forward_with_inline_reduction(log_block_scores, log_initial_probs,
                                  log_transition_matrix, boundary_prob: float,
                                  max_width: int, min_width: int = 1) -> ForwardChart:
    """O1: the reference loop, with the scipy reduction replaced."""
    scores, log_pi, log_p, J, K = _validate(
        log_block_scores, log_initial_probs, log_transition_matrix, boundary_prob,
        max_width, min_width)
    log_db = math.log(float(boundary_prob))
    log_1mdb = math.log1p(-float(boundary_prob))
    began = time.perf_counter()
    COUNTERS.forward_inline_calls += 1

    alpha = np.full((J + 1, K), NEG)
    for b in range(1, J + 1):
        for k in range(K):
            _, _, terms = _predecessor_terms(alpha, b, k, scores, log_pi, log_p,
                                             log_db, log_1mdb, max_width, min_width)
            if terms.size:
                finite = terms[np.isfinite(terms)]
                alpha[b, k] = inline_logsumexp(finite) if finite.size else NEG
    log_z = inline_logsumexp(alpha[J]) if np.isfinite(alpha[J]).any() else NEG
    return _chart(alpha, log_z, scores, log_pi, log_p, J, K, boundary_prob,
                  max_width, min_width, time.perf_counter() - began)


def _factorised_core(scores, log_pi, log_p, log_db, log_1mdb, J, K, max_width, min_width):
    alpha = np.full((J + 1, K), NEG)
    r = np.full((J + 1, K), NEG)
    for b in range(1, J + 1):
        lowest = max(0, b - int(max_width))
        lo, hi = max(1, lowest), b - int(min_width)
        if hi >= lo:
            a_idx = np.arange(lo, hi + 1)
            width_pen = (b - a_idx - 1) * log_1mdb
            terms = (r[lo:hi + 1, :] + scores[lo:hi + 1, b, :]
                     + log_db + width_pen[:, None])
            row = reduce_axis(terms, axis=0)
        else:
            row = np.full(K, NEG)
        if lowest == 0 and int(min_width) <= b:
            row = np.logaddexp(row, log_pi + scores[0, b, :] + (b - 1) * log_1mdb)
        alpha[b, :] = row
        r[b, :] = reduce_axis(alpha[b, :, None] + log_p, axis=0)
    return alpha, float(reduce_axis(alpha[J][None, :], axis=1)[0])


def forward_factorised(log_block_scores, log_initial_probs, log_transition_matrix,
                       boundary_prob: float, max_width: int,
                       min_width: int = 1) -> ForwardChart:
    """O3: O(J K^2 + J D K) instead of O(J D K^2)."""
    scores, log_pi, log_p, J, K = _validate(
        log_block_scores, log_initial_probs, log_transition_matrix, boundary_prob,
        max_width, min_width)
    began = time.perf_counter()
    COUNTERS.forward_factorised_calls += 1
    alpha, log_z = _factorised_core(scores, log_pi, log_p, math.log(float(boundary_prob)),
                                    math.log1p(-float(boundary_prob)), J, K,
                                    max_width, min_width)
    return _chart(alpha, log_z, scores, log_pi, log_p, J, K, boundary_prob,
                  max_width, min_width, time.perf_counter() - began)


# -------------------------------------------------------------------------------- O4
def _batched_core(stack, log_pi, log_p, log_db, log_1mdb, J, K, max_width, min_width):
    B = stack.shape[0]
    alpha = np.full((B, J + 1, K), NEG)
    r = np.full((B, J + 1, K), NEG)
    for b in range(1, J + 1):
        lowest = max(0, b - int(max_width))
        lo, hi = max(1, lowest), b - int(min_width)
        if hi >= lo:
            a_idx = np.arange(lo, hi + 1)
            width_pen = (b - a_idx - 1) * log_1mdb
            terms = (r[:, lo:hi + 1, :] + stack[:, lo:hi + 1, b, :]
                     + log_db + width_pen[None, :, None])
            row = reduce_axis(terms, axis=1)
        else:
            row = np.full((B, K), NEG)
        if lowest == 0 and int(min_width) <= b:
            row = np.logaddexp(row, log_pi[None, :] + stack[:, 0, b, :]
                               + (b - 1) * log_1mdb)
        alpha[:, b, :] = row
        r[:, b, :] = reduce_axis(alpha[:, b, :, None] + log_p[None, :, :], axis=1)
    return alpha, reduce_axis(alpha[:, J, :], axis=1)


def forward_batched_group(tables, log_initial_probs, log_transition_matrix,
                          boundary_prob: float, max_width: int,
                          min_width: int = 1) -> list:
    """O4: one `ForwardChart` per trace, for traces of identical length.

    Each chart keeps its OWN score table -- the same array the sequential path would have
    handed to the frozen `backward_sample` -- so the backward pass is untouched.
    """
    if not tables:
        return []
    shapes = {np.asarray(t).shape for t in tables}
    if len(shapes) != 1:
        raise ValueError(f"forward_batched_group needs one shape, got {sorted(shapes)}")
    stack = np.stack([np.asarray(t, dtype=float) for t in tables])
    log_pi = np.asarray(log_initial_probs, dtype=float)
    log_p = np.asarray(log_transition_matrix, dtype=float)
    B, J, _, K = stack.shape
    began = time.perf_counter()
    alpha, log_z = _batched_core(stack, log_pi, log_p, math.log(float(boundary_prob)),
                                 math.log1p(-float(boundary_prob)), J, K,
                                 max_width, min_width)
    elapsed = time.perf_counter() - began
    COUNTERS.forward_batched_groups += 1
    COUNTERS.forward_batched_traces += int(B)

    return [_chart(alpha[i], log_z[i], np.asarray(table, dtype=float), log_pi, log_p,
                   J, K, boundary_prob, max_width, min_width, elapsed / B)
            for i, table in enumerate(tables)]


# -------------------------------------------------------------------------- dispatch
def forward_dispatch(log_block_scores, log_initial_probs, log_transition_matrix,
                     boundary_prob: float, max_width: int,
                     min_width: int = 1) -> ForwardChart:
    """One trace, whichever single-trace variant the flags select."""
    if FLAGS.factorised_forward:
        return forward_factorised(log_block_scores, log_initial_probs,
                                  log_transition_matrix, boundary_prob, max_width,
                                  min_width)
    if FLAGS.inline_logsumexp:
        return forward_with_inline_reduction(log_block_scores, log_initial_probs,
                                             log_transition_matrix, boundary_prob,
                                             max_width, min_width)
    from hpop.mcmc_original.semi_markov_ffbs import forward as reference_forward
    COUNTERS.forward_reference_calls += 1
    return reference_forward(log_block_scores, log_initial_probs, log_transition_matrix,
                             boundary_prob, max_width, min_width)
