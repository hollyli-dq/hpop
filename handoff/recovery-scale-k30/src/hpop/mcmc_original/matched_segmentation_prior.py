"""The exact matched segmentation prior: normalizer, sampler, and marginals.

For a fixed trace length ``J`` let ``S(J)`` be the contiguous segmentations of
``[0, J)`` whose block widths all satisfy ``D_min <= w <= D_max``. The registered
prior over ``S(J)`` is

    p(S | J, delta_B) = delta_B^(L-1) (1 - delta_B)^(J-L) / C_J(delta_B)

with ``L`` the number of blocks and

    C_J(delta_B) = sum_{S in S(J)} delta_B^(L(S)-1) (1 - delta_B)^(J-L(S)).

The unnormalized weight is exactly what ``targets.log_boundary_prior`` computes
for the production target; this module adds the normalizer over the legal-width
state space and an exact sequential sampler.

## Suffix recursion

Each block of width ``w`` contributes ``(1 - delta_B)^(w-1)``, plus one factor
``delta_B`` if it is not the final block. With ``G(0) = 1`` and, for remaining
length ``r > 0``,

    G(r) = sum_{w in [D_min, min(D_max, r)]}
               (1 - delta_B)^(w-1) * delta_B^[w < r] * G(r - w),

residual lengths with no legal completion get ``G = 0`` automatically (an empty
sum), and ``C_J(delta_B) = G(J)``.

## Exact sequential sampler

At remaining length ``r`` the next width is drawn with probability proportional
to ``(1 - delta_B)^(w-1) * delta_B^[w < r] * G(r - w)``. For a complete draw
``(w_1, ..., w_L)`` with residuals ``r_1 = J``, ``r_{l+1} = r_l - w_l``,

    p(w_1, ..., w_L)
        = prod_l [ (1-dB)^(w_l-1) dB^[w_l < r_l] G(r_l - w_l) / G(r_l) ]
        = dB^(L-1) (1-dB)^(J-L) * G(0) / G(J)
        = dB^(L-1) (1-dB)^(J-L) / C_J(delta_B),

because the ``G`` factors telescope and ``[w_l < r_l]`` holds exactly for the
``L - 1`` non-final blocks. The sampler is therefore *exact*, not approximate.

Everything is computed in log space with the production ``targets.logsumexp``.
The production sampler never enumerates complete segmentations.
"""

from __future__ import annotations

import math

import numpy as np

from hpop.mcmc_original.stage6e_frozen import (
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH,
)
from hpop.mcmc_original.targets import logsumexp

__all__ = [
    "log_suffix_normalizers", "log_normalizer", "log_segmentation_prior",
    "width_sampling_tables", "sample_segmentation_widths", "ends_of_widths",
    "log_prefix_weights", "log_segment_count_weights",
    "segment_count_distribution_dp", "boundary_marginals_dp",
]


def _check_config(delta_b: float, min_width: int, max_width: int) -> tuple[int, int]:
    if not (0.0 < delta_b < 1.0):
        raise ValueError(f"delta_b must be in (0, 1), got {delta_b}")
    min_width, max_width = int(min_width), int(max_width)
    if not (1 <= min_width <= max_width):
        raise ValueError(f"need 1 <= min_width <= max_width, got [{min_width}, {max_width}]")
    return min_width, max_width


# ------------------------------------------------------------------ suffix recursion
def log_suffix_normalizers(trace_length: int, delta_b: float = DELTA_B,
                           min_width: int = MIN_BLOCK_WIDTH,
                           max_width: int = MAX_BLOCK_WIDTH) -> np.ndarray:
    """``log G(r)`` for ``r = 0 .. J``; ``-inf`` where no legal completion exists."""
    min_width, max_width = _check_config(delta_b, min_width, max_width)
    J = int(trace_length)
    if J < 0:
        raise ValueError(f"trace_length must be >= 0, got {J}")
    log_db = math.log(delta_b)
    log_1mdb = math.log1p(-delta_b)
    log_g = np.full(J + 1, -math.inf)
    log_g[0] = 0.0
    for r in range(1, J + 1):
        terms = []
        for w in range(min_width, min(max_width, r) + 1):
            tail = log_g[r - w]
            if tail == -math.inf:
                continue
            terms.append((w - 1) * log_1mdb + (log_db if w < r else 0.0) + tail)
        if terms:
            log_g[r] = logsumexp(terms)
    return log_g


def log_normalizer(trace_length: int, delta_b: float = DELTA_B,
                   min_width: int = MIN_BLOCK_WIDTH,
                   max_width: int = MAX_BLOCK_WIDTH) -> float:
    """``log C_J(delta_B) = log G(J)``."""
    return float(log_suffix_normalizers(trace_length, delta_b,
                                        min_width, max_width)[-1])


def log_segmentation_prior(widths, trace_length: int, delta_b: float = DELTA_B,
                           min_width: int = MIN_BLOCK_WIDTH,
                           max_width: int = MAX_BLOCK_WIDTH,
                           log_c: float | None = None) -> float:
    """Normalized ``log p(S | J, delta_B)`` of one explicit width sequence."""
    min_width, max_width = _check_config(delta_b, min_width, max_width)
    widths = [int(w) for w in widths]
    J = int(trace_length)
    if not widths or sum(widths) != J:
        raise ValueError(f"widths {widths} do not cover a trace of length {J}")
    if any(not (min_width <= w <= max_width) for w in widths):
        return -math.inf
    if log_c is None:
        log_c = log_normalizer(J, delta_b, min_width, max_width)
    L = len(widths)
    return ((L - 1) * math.log(delta_b) + (J - L) * math.log1p(-delta_b) - log_c)


# --------------------------------------------------------------------- exact sampler
def width_sampling_tables(trace_length: int, delta_b: float = DELTA_B,
                          min_width: int = MIN_BLOCK_WIDTH,
                          max_width: int = MAX_BLOCK_WIDTH) -> dict:
    """Per-residual next-width distributions, precomputed once for a fixed ``J``.

    ``tables[r] = (widths, probabilities)`` with probabilities proportional to
    ``(1-dB)^(w-1) * dB^[w < r] * G(r-w)``, normalized in log space.
    """
    min_width, max_width = _check_config(delta_b, min_width, max_width)
    J = int(trace_length)
    log_g = log_suffix_normalizers(J, delta_b, min_width, max_width)
    if log_g[J] == -math.inf:
        raise ValueError(f"no legal segmentation of J={J} with widths in "
                         f"[{min_width}, {max_width}]")
    log_db = math.log(delta_b)
    log_1mdb = math.log1p(-delta_b)
    tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for r in range(1, J + 1):
        if log_g[r] == -math.inf:
            continue
        widths, log_terms = [], []
        for w in range(min_width, min(max_width, r) + 1):
            tail = log_g[r - w]
            if tail == -math.inf:
                continue
            widths.append(w)
            log_terms.append((w - 1) * log_1mdb + (log_db if w < r else 0.0) + tail)
        log_terms = np.asarray(log_terms, dtype=float)
        probabilities = np.exp(log_terms - logsumexp(log_terms))
        probabilities = probabilities / probabilities.sum()   # exact simplex for rng.choice
        tables[r] = (np.asarray(widths, dtype=int), probabilities)
    return {"trace_length": J, "delta_b": float(delta_b), "min_width": min_width,
            "max_width": max_width, "log_g": log_g, "tables": tables}


def sample_segmentation_widths(rng: np.random.Generator, trace_length: int,
                               delta_b: float = DELTA_B,
                               min_width: int = MIN_BLOCK_WIDTH,
                               max_width: int = MAX_BLOCK_WIDTH,
                               tables: dict | None = None) -> tuple[int, ...]:
    """Draw one exact ``S ~ p(S | J, delta_B)`` as a width sequence summing to ``J``."""
    if tables is None:
        tables = width_sampling_tables(trace_length, delta_b, min_width, max_width)
    elif (tables["trace_length"] != int(trace_length)
          or tables["delta_b"] != float(delta_b)
          or tables["min_width"] != int(min_width)
          or tables["max_width"] != int(max_width)):
        raise ValueError("supplied sampling tables were built for a different config")
    remaining = int(trace_length)
    out: list[int] = []
    while remaining > 0:
        widths, probabilities = tables["tables"][remaining]
        w = int(widths[int(rng.choice(len(widths), p=probabilities))])
        out.append(w)
        remaining -= w
    if remaining != 0:
        raise RuntimeError(f"sampler overshot the trace length: residual {remaining}")
    return tuple(out)


def ends_of_widths(widths) -> tuple[int, ...]:
    """Cumulative block ends; the internal boundaries are all but the last entry."""
    ends, running = [], 0
    for w in widths:
        running += int(w)
        ends.append(running)
    return tuple(ends)


# ------------------------------------------- production-side exact marginal routes
def log_prefix_weights(trace_length: int, delta_b: float = DELTA_B,
                       min_width: int = MIN_BLOCK_WIDTH,
                       max_width: int = MAX_BLOCK_WIDTH) -> np.ndarray:
    """``log F(t)``: weight of legal segmentations of ``[0, t)``, cuts attributed
    to the boundary *preceding* each non-first block — the mirror of ``G``."""
    min_width, max_width = _check_config(delta_b, min_width, max_width)
    J = int(trace_length)
    log_db = math.log(delta_b)
    log_1mdb = math.log1p(-delta_b)
    log_f = np.full(J + 1, -math.inf)
    log_f[0] = 0.0
    for t in range(1, J + 1):
        terms = []
        for w in range(min_width, min(max_width, t) + 1):
            head = log_f[t - w]
            if head == -math.inf:
                continue
            terms.append(head + (w - 1) * log_1mdb + (log_db if t - w > 0 else 0.0))
        if terms:
            log_f[t] = logsumexp(terms)
    return log_f


def log_segment_count_weights(trace_length: int, delta_b: float = DELTA_B,
                              min_width: int = MIN_BLOCK_WIDTH,
                              max_width: int = MAX_BLOCK_WIDTH) -> np.ndarray:
    """``log W(J, L)`` for ``L = 0 .. floor(J / min_width)``, where ``W(t, L)`` sums
    ``prod_l (1-dB)^(w_l - 1)`` over compositions of ``t`` into ``L`` legal widths."""
    min_width, max_width = _check_config(delta_b, min_width, max_width)
    J = int(trace_length)
    log_1mdb = math.log1p(-delta_b)
    max_segments = J // min_width if min_width > 0 else J
    log_w = np.full((J + 1, max_segments + 1), -math.inf)
    log_w[0, 0] = 0.0
    for t in range(1, J + 1):
        for L in range(1, max_segments + 1):
            terms = []
            for w in range(min_width, min(max_width, t) + 1):
                head = log_w[t - w, L - 1]
                if head == -math.inf:
                    continue
                terms.append(head + (w - 1) * log_1mdb)
            if terms:
                log_w[t, L] = logsumexp(terms)
    return log_w[J]


def segment_count_distribution_dp(trace_length: int, delta_b: float = DELTA_B,
                                  min_width: int = MIN_BLOCK_WIDTH,
                                  max_width: int = MAX_BLOCK_WIDTH) -> np.ndarray:
    """Exact ``p_model(L | J)`` indexed by ``L``, from the ``(t, L)`` weight DP."""
    log_by_count = log_segment_count_weights(trace_length, delta_b,
                                             min_width, max_width)
    log_db = math.log(delta_b)
    scored = np.full_like(log_by_count, -math.inf)
    for L in range(1, log_by_count.shape[0]):
        if log_by_count[L] != -math.inf:
            scored[L] = (L - 1) * log_db + log_by_count[L]
    finite = scored[np.isfinite(scored)]
    if finite.size == 0:
        raise ValueError(f"no legal segmentation for J={trace_length}")
    log_c = logsumexp(finite)
    out = np.zeros_like(scored)
    mask = np.isfinite(scored)
    out[mask] = np.exp(scored[mask] - log_c)
    return out


def boundary_marginals_dp(trace_length: int, delta_b: float = DELTA_B,
                          min_width: int = MIN_BLOCK_WIDTH,
                          max_width: int = MAX_BLOCK_WIDTH) -> np.ndarray:
    """Exact ``p_model(B_t = 1 | J)`` for ``t = 1 .. J-1`` by prefix-suffix DP:

        p(B_t = 1) = F(t) * delta_B * G(J - t) / G(J).
    """
    J = int(trace_length)
    log_f = log_prefix_weights(J, delta_b, min_width, max_width)
    log_g = log_suffix_normalizers(J, delta_b, min_width, max_width)
    if log_g[J] == -math.inf:
        raise ValueError(f"no legal segmentation for J={J}")
    log_db = math.log(delta_b)
    out = np.zeros(J - 1)
    for t in range(1, J):
        left, right = log_f[t], log_g[J - t]
        if left == -math.inf or right == -math.inf:
            continue
        out[t - 1] = math.exp(left + log_db + right - log_g[J])
    return out
