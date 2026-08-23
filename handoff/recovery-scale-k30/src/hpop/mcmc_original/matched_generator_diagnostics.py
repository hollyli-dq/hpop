"""Independent exact references and empirical summaries for the matched generator.

Everything in this module is a *reference*. Nothing here imports or calls
``matched_segmentation_prior`` or ``matched_synthetic_generator`` — the point of
the parity experiment is that the production sampler and these references reach
the same numbers by disjoint routes.

The reference route is combinatorial: exact integer composition counts
``M(J, L)`` (Python big-ints, so exact for any J used here) combined with the
closed-form weight ``delta_B^(L-1) (1 - delta_B)^(J-L)``:

    C_J(delta_B)      = sum_L delta_B^(L-1) (1-delta_B)^(J-L) M(J, L)
    p_model(L | J)    = delta_B^(L-1) (1-delta_B)^(J-L) M(J, L) / C_J
    p_model(B_t = 1)  = C_t * delta_B * C_{J-t} / C_J

The boundary identity holds because a segmentation with a cut at ``t`` is exactly
a pair (segmentation of ``[0, t)``, segmentation of ``[t, J)``), and its weight
factorizes as ``[dB^(L1-1)(1-dB)^(t-L1)] * dB * [dB^(L2-1)(1-dB)^(J-t-L2)]``.

``enumerate_legal_segmentations`` is exhaustive and intended for tiny ``J`` only.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "enumerate_legal_segmentations", "composition_counts",
    "exact_normalizer", "exact_segment_count_distribution",
    "exact_boundary_marginals", "exact_expected_width_counts",
    "log_normalizer_from_enumeration", "total_variation",
    "empirical_segment_count_distribution", "empirical_boundary_marginals",
    "segmentation_support_violations",
]


def _check(delta_b: float, min_width: int, max_width: int) -> tuple[int, int]:
    if not (0.0 < delta_b < 1.0):
        raise ValueError(f"delta_b must be in (0, 1), got {delta_b}")
    min_width, max_width = int(min_width), int(max_width)
    if not (1 <= min_width <= max_width):
        raise ValueError(f"need 1 <= min_width <= max_width, got [{min_width}, {max_width}]")
    return min_width, max_width


def _logsumexp(values) -> float:
    a = [float(v) for v in values if float(v) != -math.inf]
    if not a:
        return -math.inf
    shift = max(a)
    return shift + math.log(sum(math.exp(v - shift) for v in a))


# --------------------------------------------------------------------- enumeration
def enumerate_legal_segmentations(trace_length: int, min_width: int,
                                  max_width: int) -> list[tuple[int, ...]]:
    """Every ordered width composition of ``J`` with parts in ``[min, max]``.

    Exhaustive; tiny ``J`` only.
    """
    min_width, max_width = int(min_width), int(max_width)
    J = int(trace_length)

    def _extend(remaining: int) -> list[tuple[int, ...]]:
        if remaining == 0:
            return [()]
        out = []
        for w in range(min_width, min(max_width, remaining) + 1):
            for tail in _extend(remaining - w):
                out.append((w,) + tail)
        return out

    return _extend(J)


def log_normalizer_from_enumeration(trace_length: int, delta_b: float,
                                    min_width: int, max_width: int) -> float:
    """``log C_J`` by brute force over the enumerated state space. Tiny ``J`` only."""
    min_width, max_width = _check(delta_b, min_width, max_width)
    J = int(trace_length)
    terms = []
    for widths in enumerate_legal_segmentations(J, min_width, max_width):
        L = len(widths)
        terms.append((L - 1) * math.log(delta_b) + (J - L) * math.log1p(-delta_b))
    return _logsumexp(terms)


# ----------------------------------------------------------- combinatorial references
def composition_counts(trace_length: int, min_width: int,
                       max_width: int) -> list[list[int]]:
    """``M[t][L]``: exact integer count of compositions of ``t`` into ``L`` legal parts."""
    min_width, max_width = int(min_width), int(max_width)
    J = int(trace_length)
    max_segments = J // min_width if min_width > 0 else J
    counts = [[0] * (max_segments + 1) for _ in range(J + 1)]
    counts[0][0] = 1
    for t in range(1, J + 1):
        for L in range(1, max_segments + 1):
            total = 0
            for w in range(min_width, min(max_width, t) + 1):
                total += counts[t - w][L - 1]
            counts[t][L] = total
    return counts


def exact_normalizer(trace_length: int, delta_b: float, min_width: int,
                     max_width: int) -> float:
    """``log C_J(delta_B)`` from exact composition counts."""
    min_width, max_width = _check(delta_b, min_width, max_width)
    J = int(trace_length)
    counts = composition_counts(J, min_width, max_width)[J]
    terms = []
    for L, count in enumerate(counts):
        if L == 0 or count == 0:
            continue
        terms.append((L - 1) * math.log(delta_b) + (J - L) * math.log1p(-delta_b)
                     + math.log(count))
    return _logsumexp(terms)


def exact_segment_count_distribution(trace_length: int, delta_b: float,
                                     min_width: int, max_width: int) -> np.ndarray:
    """``p_model(L | J)`` indexed by ``L`` (index 0 is structurally zero)."""
    min_width, max_width = _check(delta_b, min_width, max_width)
    J = int(trace_length)
    counts = composition_counts(J, min_width, max_width)[J]
    log_terms = np.full(len(counts), -math.inf)
    for L, count in enumerate(counts):
        if L == 0 or count == 0:
            continue
        log_terms[L] = ((L - 1) * math.log(delta_b)
                        + (J - L) * math.log1p(-delta_b) + math.log(count))
    log_c = _logsumexp(log_terms)
    if log_c == -math.inf:
        raise ValueError(f"no legal segmentation for J={J}")
    out = np.zeros(len(counts))
    for L, value in enumerate(log_terms):
        if value != -math.inf:
            out[L] = math.exp(value - log_c)
    return out


def exact_boundary_marginals(trace_length: int, delta_b: float, min_width: int,
                             max_width: int) -> np.ndarray:
    """``p_model(B_t = 1 | J)`` for ``t = 1 .. J-1`` via ``C_t * dB * C_{J-t} / C_J``."""
    min_width, max_width = _check(delta_b, min_width, max_width)
    J = int(trace_length)
    log_c = [-math.inf] * (J + 1)
    log_c[0] = 0.0
    for t in range(1, J + 1):
        log_c[t] = exact_normalizer(t, delta_b, min_width, max_width) \
            if t >= min_width else -math.inf
    if log_c[J] == -math.inf:
        raise ValueError(f"no legal segmentation for J={J}")
    out = np.zeros(J - 1)
    for t in range(1, J):
        left, right = log_c[t], log_c[J - t]
        if left == -math.inf or right == -math.inf:
            continue
        out[t - 1] = math.exp(left + math.log(delta_b) + right - log_c[J])
    return out


def exact_expected_width_counts(trace_length: int, delta_b: float, min_width: int,
                                max_width: int) -> np.ndarray:
    """``E[# blocks of width w]`` per trace, indexed ``w = min_width .. max_width``.

    A block ``[a, a+w)`` appears iff the prefix and suffix segment legally, paying
    ``delta_B`` per adjacent internal cut:

        p(block [a, a+w)) = C_a dB^[a>0] (1-dB)^(w-1) dB^[a+w<J] C_{J-a-w} / C_J.
    """
    min_width, max_width = _check(delta_b, min_width, max_width)
    J = int(trace_length)
    log_c = [0.0] + [exact_normalizer(t, delta_b, min_width, max_width)
                     if t >= min_width else -math.inf for t in range(1, J + 1)]
    log_db = math.log(delta_b)
    log_1mdb = math.log1p(-delta_b)
    out = np.zeros(max_width - min_width + 1)
    for w in range(min_width, max_width + 1):
        for a in range(0, J - w + 1):
            left, right = log_c[a], log_c[J - a - w]
            if left == -math.inf or right == -math.inf:
                continue
            value = (left + (log_db if a > 0 else 0.0) + (w - 1) * log_1mdb
                     + (log_db if a + w < J else 0.0) + right - log_c[J])
            out[w - min_width] += math.exp(value)
    return out


# ------------------------------------------------------------------ empirical side
def total_variation(p, q) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch {p.shape} vs {q.shape}")
    return 0.5 * float(np.abs(p - q).sum())


def empirical_segment_count_distribution(width_samples, max_segments: int) -> np.ndarray:
    out = np.zeros(int(max_segments) + 1)
    n = 0
    for widths in width_samples:
        out[len(widths)] += 1.0
        n += 1
    if n == 0:
        raise ValueError("no samples")
    return out / n


def empirical_boundary_marginals(width_samples, trace_length: int) -> np.ndarray:
    J = int(trace_length)
    out = np.zeros(J - 1)
    n = 0
    for widths in width_samples:
        running = 0
        for w in widths[:-1]:
            running += int(w)
            out[running - 1] += 1.0
        n += 1
    if n == 0:
        raise ValueError("no samples")
    return out / n


def segmentation_support_violations(width_samples, trace_length: int,
                                    min_width: int, max_width: int) -> dict:
    """Counts of every way a sampled width sequence could be illegal.

    Widths are a composition by construction, so 'overlap' and 'gap' both reduce
    to the cover failing to end exactly at ``J``; they are counted separately
    from width violations anyway so the report shows each registered zero.
    """
    J = int(trace_length)
    counts = {"illegal_width": 0, "incomplete_cover": 0, "overlap_or_gap": 0,
              "empty": 0, "n_samples": 0}
    for widths in width_samples:
        counts["n_samples"] += 1
        widths = [int(w) for w in widths]
        if not widths:
            counts["empty"] += 1
            continue
        if any(not (min_width <= w <= max_width) for w in widths):
            counts["illegal_width"] += 1
        total = sum(widths)
        if total != J:
            counts["incomplete_cover"] += 1
            counts["overlap_or_gap"] += 1
    return counts
