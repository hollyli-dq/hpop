"""Step 7A — model-agnostic semi-Markov forward filtering / backward sampling.

This module is an **algorithm**, not a model. It consumes a table of block log scores and
three prior pieces, and it knows nothing whatever about `U`, `rho`, the recurrent RFS
recursion, CPA semantics, Stage 6, or the local move kernel. Nothing here imports from a
Stage-6 module, and the only dependencies are `numpy` and `scipy.special.logsumexp`.

## The distribution

For one sequence of length `J`, a segmentation with `L` blocks is

    S = ((a_1, b_1, z_1), ..., (a_L, b_L, z_L)),   a_1 = 0, b_L = J, a_{l+1} = b_l

with half-open blocks `[a, b)`, and the registered path weight is

    log w(S) = log pi[z_1]
             + sum_l log p_block(x[a_l:b_l] | z_l)
             + (J - L) log(1 - delta_B) + (L - 1) log delta_B
             + sum_{l >= 2} log P[z_{l-1}, z_l].

Every internal position inside a block contributes `log(1 - delta_B)`; every boundary
between two blocks contributes `log delta_B`; the sequence endpoint contributes neither,
and there is no terminal transition. The `(J - L)` count is exactly
`sum_l (b_l - a_l - 1)`, which is why the boundary prior factorises over blocks and the
recursion below is available at all.

## The recursion

    alpha(b, k) = log sum over segmentations of [0, b) whose last block is [a, b)_k
                  of the weight of that prefix

    alpha(b, k) = LSE of
        (initial, a = 0)     log pi[k] + log p_block(0, b, k) + (b - 1) log(1 - delta_B)
        (a >= 1, previous h) alpha(a, h) + log delta_B + log P[h, k]
                             + log p_block(a, b, k) + (b - a - 1) log(1 - delta_B)

    log Z = LSE_k alpha(J, k)

Backward sampling inverts exactly that: draw `z_L ~ softmax_k alpha(J, k)`, then repeatedly
draw one **joint** predecessor option `(a, h)` from the very terms that built `alpha[b, k]`,
never `a` and `h` separately. `predecessor_terms` is the single audited helper that
generates those terms, and the forward recursion and the backward sampler both call it, so
the two cannot silently disagree about what the model is.

## Conventions

`log_block_scores[a, b, k]` is the score of `[a, b)_k`, with shape `(J, J + 1, K)`.
Anything illegal — a width outside the bounds, a block a model forbids — must be `-inf`,
and `-inf` is a supported value in the block table, in `log_initial_probs` and in
`log_transition_matrix`. Under the registered Stage 6E model `P` has a zero diagonal, so
`log_transition_matrix[h, h] = -inf` and adjacent blocks cannot share a skill; that is a
property of the matrix passed in, not of this module.

`min_width` is an optimisation only. Widths below it must already be `-inf` in the table;
passing it merely stops the recursion from visiting terms it would discard anyway, and
`test_min_width_is_an_optimisation_only` pins that the two routes agree bit for bit.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

__all__ = [
    "ForwardChart", "SemiMarkovFFBS", "predecessor_terms", "forward", "backward_sample",
    "backward_sample_many", "posterior_log_marginals",
]


# --------------------------------------------------------------------------- validation
def _validate(log_block_scores, log_initial_probs, log_transition_matrix,
              boundary_prob: float, max_width: int, min_width: int):
    scores = np.asarray(log_block_scores, dtype=float)
    log_pi = np.asarray(log_initial_probs, dtype=float)
    log_p = np.asarray(log_transition_matrix, dtype=float)

    if scores.ndim != 3:
        raise ValueError(f"log_block_scores must be (J, J+1, K); got {scores.shape}")
    J, b_dim, K = scores.shape
    if b_dim != J + 1:
        raise ValueError(f"log_block_scores must be (J, J+1, K); got {scores.shape}")
    if J < 1 or K < 1:
        raise ValueError(f"need J >= 1 and K >= 1; got J={J}, K={K}")
    if log_pi.shape != (K,):
        raise ValueError(f"log_initial_probs must be (K,) = ({K},); got {log_pi.shape}")
    if log_p.shape != (K, K):
        raise ValueError(f"log_transition_matrix must be ({K}, {K}); got {log_p.shape}")
    if not 0.0 < float(boundary_prob) < 1.0:
        raise ValueError(f"boundary_prob must lie in (0, 1); got {boundary_prob}")
    if int(max_width) < 1:
        raise ValueError(f"max_width must be >= 1; got {max_width}")
    if int(min_width) < 1 or int(min_width) > int(max_width):
        raise ValueError(f"min_width must satisfy 1 <= min_width <= max_width; "
                         f"got {min_width} with max_width {max_width}")
    for name, array in (("log_block_scores", scores), ("log_initial_probs", log_pi),
                        ("log_transition_matrix", log_p)):
        if np.isnan(array).any():
            raise ValueError(f"{name} contains NaN")
        if np.isposinf(array).any():
            raise ValueError(f"{name} contains +inf")
    return scores, log_pi, log_p, J, K


# ------------------------------------------------------------------------- the chart
@dataclass(frozen=True)
class ForwardChart:
    """A completed forward pass. Everything backward sampling needs, and nothing else."""

    alpha: np.ndarray                 # (J + 1, K); alpha[0] is -inf and is never read
    log_normalizer: float
    J: int
    K: int
    max_width: int
    min_width: int
    boundary_prob: float
    log_block_scores: np.ndarray
    log_initial_probs: np.ndarray
    log_transition_matrix: np.ndarray
    build_seconds: float = float("nan")

    @property
    def log_evidence(self) -> float:                 # a readable alias
        return float(self.log_normalizer)

    def summary(self) -> dict:
        return {"J": int(self.J), "K": int(self.K), "max_width": int(self.max_width),
                "min_width": int(self.min_width),
                "boundary_prob": float(self.boundary_prob),
                "log_normalizer": float(self.log_normalizer),
                "build_seconds": float(self.build_seconds),
                "alpha_final_row": [float(v) for v in self.alpha[self.J]]}


# --------------------------------------------------- the one audited recurrence helper
def predecessor_terms(alpha, b: int, k: int, log_block_scores, log_initial_probs,
                      log_transition_matrix, log_delta_b: float, log_one_minus_delta_b:
                      float, max_width: int, min_width: int = 1):
    """Every term that contributes to `alpha[b, k]`, with the option that produced it.

    Returns `(starts, previous_skills, log_terms)` as three parallel arrays. A previous
    skill of `-1` marks the *initial* option `a = 0`, whose weight carries `log pi[k]`
    instead of a transition; every other option is a genuine `(a, h)` pair with `a >= 1`.

    The forward recursion sums these terms; the backward sampler draws one of them. That
    is the whole reason this function exists: there is exactly one place in the codebase
    where the semi-Markov weight of a predecessor option is written down.
    """
    b = int(b)
    k = int(k)
    lowest = max(0, b - int(max_width))
    starts, prev, terms = [], [], []

    # the initial option, a = 0: legal only if the whole prefix [0, b) is one block
    if lowest == 0 and int(min_width) <= b:
        value = (float(log_initial_probs[k]) + float(log_block_scores[0, b, k])
                 + (b - 1) * log_one_minus_delta_b)
        if value > -np.inf:                      # an illegal block or skill contributes nothing
            starts.append(0)
            prev.append(-1)
            terms.append(value)

    # the non-initial options (a, h), a >= 1
    for a in range(max(1, lowest), b - int(min_width) + 1):
        block = float(log_block_scores[a, b, k])
        if math.isinf(block) and block < 0:
            continue
        carried = block + log_delta_b + (b - a - 1) * log_one_minus_delta_b
        for h in range(alpha.shape[1]):
            previous = float(alpha[a, h])
            if previous == -np.inf:
                continue
            transition = float(log_transition_matrix[h, k])
            if transition == -np.inf:
                continue
            starts.append(a)
            prev.append(h)
            terms.append(previous + carried + transition)

    return (np.asarray(starts, dtype=int), np.asarray(prev, dtype=int),
            np.asarray(terms, dtype=float))


# ------------------------------------------------------------------ forward filtering
def forward(log_block_scores, log_initial_probs, log_transition_matrix,
            boundary_prob: float, max_width: int, min_width: int = 1) -> ForwardChart:
    """The forward chart. `alpha[b, k]` is the log weight of every prefix of `[0, b)`
    whose last block carries skill `k`, and `log Z = LSE_k alpha[J, k]`."""
    scores, log_pi, log_p, J, K = _validate(
        log_block_scores, log_initial_probs, log_transition_matrix, boundary_prob,
        max_width, min_width)
    log_db = math.log(float(boundary_prob))
    log_1mdb = math.log1p(-float(boundary_prob))
    began = time.perf_counter()

    alpha = np.full((J + 1, K), -np.inf)
    for b in range(1, J + 1):
        for k in range(K):
            _, _, terms = predecessor_terms(
                alpha, b, k, scores, log_pi, log_p, log_db, log_1mdb, max_width,
                min_width)
            if terms.size:
                finite = terms[np.isfinite(terms)]
                alpha[b, k] = float(logsumexp(finite)) if finite.size else -np.inf

    log_z = float(logsumexp(alpha[J])) if np.isfinite(alpha[J]).any() else -np.inf
    if not np.isfinite(log_z):
        raise ValueError("no finite complete path: every legal segmentation of [0, J) "
                         "has weight zero under the supplied scores and priors")
    return ForwardChart(alpha=alpha, log_normalizer=log_z, J=J, K=K,
                        max_width=int(max_width), min_width=int(min_width),
                        boundary_prob=float(boundary_prob), log_block_scores=scores,
                        log_initial_probs=log_pi, log_transition_matrix=log_p,
                        build_seconds=time.perf_counter() - began)


# ------------------------------------------------------------------ backward sampling
def _draw(log_terms: np.ndarray, rng: np.random.Generator) -> int:
    """One categorical draw from unnormalised log weights, by inverse CDF."""
    shifted = log_terms - log_terms.max()
    weights = np.exp(shifted)
    total = weights.sum()
    cumulative = np.cumsum(weights)
    return int(np.searchsorted(cumulative, rng.random() * total, side="right"))


def backward_sample(chart: ForwardChart, rng: np.random.Generator) -> tuple:
    """One exact draw `((a, b, k), ...)` from `p(S | x, Theta)`, chronological order.

    Not a Metropolis step and not an approximation: conditional on the chart the draw is
    exact and independent of every other draw.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator")
    log_db = math.log(chart.boundary_prob)
    log_1mdb = math.log1p(-chart.boundary_prob)

    final = chart.alpha[chart.J]
    k = _draw(final, rng)
    b = int(chart.J)

    blocks = []
    while True:
        starts, prev, terms = predecessor_terms(
            chart.alpha, b, k, chart.log_block_scores, chart.log_initial_probs,
            chart.log_transition_matrix, log_db, log_1mdb, chart.max_width,
            chart.min_width)
        if terms.size == 0:                                       # unreachable by construction
            raise RuntimeError(f"no predecessor option for alpha[{b}, {k}]")
        choice = _draw(terms, rng)                                 # ONE joint (a, h) draw
        a, h = int(starts[choice]), int(prev[choice])
        blocks.append((a, b, k))
        if h < 0:
            break                                                  # the initial option
        b, k = a, h
    blocks.reverse()
    return tuple(blocks)


def backward_sample_many(chart: ForwardChart, n_draws: int,
                         rng: np.random.Generator) -> list:
    """`n_draws` iid draws from the same chart. The chart is read, never written."""
    return [backward_sample(chart, rng) for _ in range(int(n_draws))]


# ------------------------------------------------- exact marginals from the same chart
def posterior_log_marginals(chart: ForwardChart) -> dict:
    """Exact labelled-block marginals `p([a, b)_k is a block of S | x)` by forward-backward.

    The backward pass mirrors the forward one: `beta[a, h]` is the log weight of every
    completion of `[a, J)` given that the block ending at `a` carried skill `h`, with
    `beta[J, h] = 0` because there is no terminal transition. These marginals are a
    *reference* for the sampler — they are exact, so comparing them against the FFBS
    frequencies isolates sampling error from recursion error.
    """
    J, K = chart.J, chart.K
    log_db = math.log(chart.boundary_prob)
    log_1mdb = math.log1p(-chart.boundary_prob)
    scores, log_pi = chart.log_block_scores, chart.log_initial_probs
    log_p = chart.log_transition_matrix

    beta = np.full((J + 1, K), -np.inf)
    beta[J, :] = 0.0
    for a in range(J - 1, 0, -1):
        for h in range(K):
            terms = []
            for b in range(a + chart.min_width, min(J, a + chart.max_width) + 1):
                for k in range(K):
                    piece = (float(log_p[h, k]) + float(scores[a, b, k]) + log_db
                             + (b - a - 1) * log_1mdb + float(beta[b, k]))
                    if np.isfinite(piece):
                        terms.append(piece)
            if terms:
                beta[a, h] = float(logsumexp(terms))

    block_marginals: dict = {}
    for a in range(J):
        for b in range(a + chart.min_width, min(J, a + chart.max_width) + 1):
            for k in range(K):
                score = float(scores[a, b, k])
                if not np.isfinite(score):
                    continue
                if a == 0:
                    prefix = float(log_pi[k]) + score + (b - 1) * log_1mdb
                else:
                    incoming = [float(chart.alpha[a, h]) + float(log_p[h, k])
                                for h in range(K)]
                    incoming = [v for v in incoming if np.isfinite(v)]
                    if not incoming:
                        continue
                    prefix = (float(logsumexp(incoming)) + log_db + score
                              + (b - a - 1) * log_1mdb)
                if not np.isfinite(prefix) or not np.isfinite(beta[b, k]):
                    continue
                block_marginals[(a, b, k)] = math.exp(
                    prefix + float(beta[b, k]) - chart.log_normalizer)

    boundary = np.zeros(max(J - 1, 0))
    labels = np.zeros((J, K))
    for (a, b, k), p in block_marginals.items():
        if a > 0:
            boundary[a - 1] += p
        labels[a:b, k] += p
    return {"beta": beta, "labelled_block_marginals": block_marginals,
            "boundary_marginals": boundary, "occurrence_label_marginals": labels,
            "log_normalizer_from_beta": float(logsumexp(
                [float(log_pi[k]) + float(scores[0, b, k]) + (b - 1) * log_1mdb
                 + float(beta[b, k])
                 for b in range(chart.min_width, min(J, chart.max_width) + 1)
                 for k in range(K)
                 if np.isfinite(scores[0, b, k]) and np.isfinite(beta[b, k])]))}


# ------------------------------------------------------------------------- the facade
@dataclass
class SemiMarkovFFBS:
    """Chart once, draw many. The chart depends only on the parameters, so at fixed
    parameters the construction cost amortises over every draw taken from it."""

    log_block_scores: np.ndarray
    log_initial_probs: np.ndarray
    log_transition_matrix: np.ndarray
    boundary_prob: float
    max_width: int
    min_width: int = 1
    _chart: ForwardChart | None = None

    def chart(self) -> ForwardChart:
        if self._chart is None:
            self._chart = forward(self.log_block_scores, self.log_initial_probs,
                                  self.log_transition_matrix, self.boundary_prob,
                                  self.max_width, self.min_width)
        return self._chart

    @property
    def log_normalizer(self) -> float:
        return self.chart().log_normalizer

    def sample(self, rng: np.random.Generator) -> tuple:
        return backward_sample(self.chart(), rng)

    def sample_many(self, n_draws: int, rng: np.random.Generator) -> list:
        return backward_sample_many(self.chart(), n_draws, rng)
