"""Original static BPOP frontier-softmax likelihood for one skill instance.

Hour 2. Given the latent partial order induced by ``U`` (Hour 1), this module gives
the probability of an observed role execution.

The generative story. A skill has ``m`` roles, ``M = {0, ..., m-1}``. An execution
emits every role exactly once. After ``y_1, ..., y_{t-1}`` have been emitted:

    remaining     R_t = M \\ {y_1, ..., y_{t-1}}

    frontier      F_t = { x in R_t : no z in R_t \\ {x} has z > x }

                  i.e. x is currently executable iff none of its predecessors is
                  still waiting. Note the direction: ``precedence[z, x]`` means z
                  is a PREDECESSOR of x and must come first.

    successors    S_t(x) = #{ z in R_t \\ {x} : x > z }
    utility       Q_t(x) = log(1 + S_t(x))

The agent is Boltzmann-rational on the frontier — it prefers roles that unblock
more of the work that is still outstanding — and trembles with probability
``epsilon`` onto a uniform draw over everything remaining:

    p(y_t = x | y_<t, U, beta, eps)
        = (1 - eps) * 1[x in F_t] * softmax_{F_t}(beta * Q_t)(x)
          + eps / |R_t|                                    for x in R_t

    p(y_t = x | ...) = 0                                   for x not in R_t

The epsilon term is what keeps order-violating executions at positive
probability; with ``eps = 0`` the support is exactly the set of linear extensions
of the partial order.

Everything is driven by the **full precedence relation** induced by U — the
transitive closure that Hour 1 returns. No transitive reduction is taken: ``S_t``
counts all remaining roles that x dominates, not just the ones it covers.

The likelihood over complete executions is a proper sequential distribution, so
it normalises over the ``m!`` permutations by construction. That identity is the
main acceptance criterion for this stage and is tested exhaustively.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u

__all__ = [
    "frontier",
    "remaining_successor_count",
    "successor_utility",
    "bpop_step_probabilities",
    "bpop_log_likelihood",
    "bpop_likelihood",
    "sample_bpop_sequence",
    "all_permutation_probabilities",
]


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------


def _check_precedence(precedence: np.ndarray) -> np.ndarray:
    p = np.asarray(precedence)
    if p.dtype != np.bool_:
        raise ValueError(f"precedence must be boolean, got dtype {p.dtype}")
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError(f"precedence must be a square 2-D matrix, got shape {p.shape}")
    if p.shape[0] < 1:
        raise ValueError("precedence must cover at least one role")
    return p


def _check_remaining(remaining: Sequence[int], m: int) -> tuple[int, ...]:
    """Validate a remaining set and return it as a sorted tuple of ints."""
    if isinstance(remaining, (str, bytes)) or not hasattr(remaining, "__iter__"):
        raise ValueError("remaining must be a sequence of role indices")
    out: list[int] = []
    for value in remaining:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"remaining must contain integer role indices, got {type(value).__name__}"
            )
        value = int(value)
        if not (0 <= value < m):
            raise ValueError(f"remaining contains role {value}, outside range({m})")
        out.append(value)
    if len(set(out)) != len(out):
        raise ValueError(f"remaining must not contain duplicates, got {tuple(out)}")
    return tuple(sorted(out))


def _check_beta_epsilon(beta: float, epsilon: float) -> tuple[float, float]:
    beta = float(beta)
    epsilon = float(epsilon)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError(f"beta must be finite and >= 0, got {beta}")
    if not math.isfinite(epsilon) or not (0.0 <= epsilon < 1.0):
        raise ValueError(f"epsilon must satisfy 0 <= epsilon < 1, got {epsilon}")
    return beta, epsilon


# ---------------------------------------------------------------------------
# frontier and successor utility
# ---------------------------------------------------------------------------


def frontier(remaining: Sequence[int], precedence: np.ndarray) -> tuple[int, ...]:
    """Currently executable roles, in ascending index order.

    ``x`` is in the frontier iff no *other remaining* role precedes it::

        not any(precedence[z, x] for z in remaining if z != x)

    Raises:
        RuntimeError: if a non-empty remaining set has an empty frontier. That
            cannot happen for a strict partial order over a finite set — a
            maximal element always exists — so it would mean ``precedence`` is
            not a valid strict order (e.g. it contains a cycle).
    """
    p = _check_precedence(precedence)
    rem = _check_remaining(remaining, p.shape[0])
    out = tuple(x for x in rem if not any(p[z, x] for z in rem if z != x))
    if rem and not out:
        raise RuntimeError(
            f"empty frontier for non-empty remaining set {rem}; "
            "precedence is not a valid strict partial order"
        )
    return out


def remaining_successor_count(
    x: int, remaining: Sequence[int], precedence: np.ndarray
) -> int:
    """``S_t(x) = #{z in remaining \\ {x} : x > z}`` — full relation, not covers."""
    p = _check_precedence(precedence)
    rem = _check_remaining(remaining, p.shape[0])
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"x must be an integer role index, got {type(x).__name__}")
    x = int(x)
    if x not in rem:
        raise ValueError(f"x={x} must belong to the remaining set {rem}")
    return int(sum(1 for z in rem if z != x and p[x, z]))


def successor_utility(
    x: int, remaining: Sequence[int], precedence: np.ndarray
) -> float:
    """``Q_t(x) = log(1 + S_t(x))``."""
    return float(math.log1p(remaining_successor_count(x, remaining, precedence)))


# ---------------------------------------------------------------------------
# one-step distribution
# ---------------------------------------------------------------------------


def _step_probabilities(
    rem: tuple[int, ...], precedence: np.ndarray, beta: float, epsilon: float
) -> np.ndarray:
    """One-step distribution over all m roles. ``rem`` must already be validated."""
    m = precedence.shape[0]
    if not rem:
        raise ValueError("remaining must be non-empty to take a step")

    probs = np.zeros(m, dtype=float)

    # trembling hand: uniform over everything still remaining
    if epsilon > 0.0:
        for x in rem:
            probs[x] += epsilon / len(rem)

    # Boltzmann-rational choice restricted to the frontier
    front = frontier(rem, precedence)
    q = np.array(
        [math.log1p(sum(1 for z in rem if z != x and precedence[x, z])) for x in front],
        dtype=float,
    )
    logits = beta * q
    logits -= logits.max()  # numerically stable softmax
    weights = np.exp(logits)
    weights /= weights.sum()
    for x, w in zip(front, weights):
        probs[x] += (1.0 - epsilon) * w

    return probs


def bpop_step_probabilities(
    remaining: Sequence[int], u: np.ndarray, beta: float, epsilon: float
) -> np.ndarray:
    """``p(y_t = x | y_<t, U, beta, epsilon)`` for every role ``x`` in ``0..m-1``.

    Roles that have already been executed get probability exactly 0. The entries
    over the remaining roles sum to 1.
    """
    precedence = precedence_from_u(u)
    beta, epsilon = _check_beta_epsilon(beta, epsilon)
    rem = _check_remaining(remaining, precedence.shape[0])
    return _step_probabilities(rem, precedence, beta, epsilon)


# ---------------------------------------------------------------------------
# complete-execution likelihood
# ---------------------------------------------------------------------------


def _check_role_sequence(role_sequence: Sequence[int], m: int) -> tuple[int, ...]:
    seq = []
    for value in role_sequence:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"role_sequence must contain integers, got {type(value).__name__}"
            )
        seq.append(int(value))
    seq = tuple(seq)
    if len(seq) != m:
        raise ValueError(
            f"role_sequence must be a complete execution of {m} roles, got length {len(seq)}"
        )
    if sorted(seq) != list(range(m)):
        raise ValueError(
            f"role_sequence must be a permutation of range({m}), got {seq}"
        )
    return seq


def bpop_log_likelihood(
    role_sequence: Sequence[int], u: np.ndarray, beta: float, epsilon: float
) -> float:
    """``log p(y | U, beta, epsilon)`` for a complete execution ``y``.

    Returns ``-inf`` for an execution the model gives zero probability, which
    happens exactly when ``epsilon = 0`` and ``y`` is not a linear extension of
    ``h(U)``.
    """
    precedence = precedence_from_u(u)
    beta, epsilon = _check_beta_epsilon(beta, epsilon)
    m = precedence.shape[0]
    seq = _check_role_sequence(role_sequence, m)

    remaining = list(range(m))
    total = 0.0
    for y in seq:
        p = float(_step_probabilities(tuple(remaining), precedence, beta, epsilon)[y])
        if p <= 0.0:
            return -math.inf
        total += math.log(p)
        remaining.remove(y)
    return total


def bpop_likelihood(
    role_sequence: Sequence[int], u: np.ndarray, beta: float, epsilon: float
) -> float:
    """``p(y | U, beta, epsilon)``; 0.0 rather than an overflow for zero-probability y."""
    log_p = bpop_log_likelihood(role_sequence, u, beta, epsilon)
    return 0.0 if log_p == -math.inf else math.exp(log_p)


def all_permutation_probabilities(
    u: np.ndarray, beta: float, epsilon: float
) -> dict[tuple[int, ...], float]:
    """Probability of every one of the ``m!`` complete executions.

    Used for the exhaustive normalisation check and the demonstration script. The
    one-step distribution is cached per remaining-set (there are only ``2^m`` of
    them), so this stays cheap for the small ``m`` this stage targets.
    """
    precedence = precedence_from_u(u)
    beta, epsilon = _check_beta_epsilon(beta, epsilon)
    m = precedence.shape[0]

    cache: dict[tuple[int, ...], np.ndarray] = {}

    def step(rem: tuple[int, ...]) -> np.ndarray:
        probs = cache.get(rem)
        if probs is None:
            probs = _step_probabilities(rem, precedence, beta, epsilon)
            cache[rem] = probs
        return probs

    out: dict[tuple[int, ...], float] = {}
    for perm in itertools.permutations(range(m)):
        remaining = list(range(m))
        p = 1.0
        for y in perm:
            p *= float(step(tuple(remaining))[y])
            if p == 0.0:
                break
            remaining.remove(y)
        out[perm] = p
    return out


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def sample_bpop_sequence(
    rng: np.random.Generator, u: np.ndarray, beta: float, epsilon: float
) -> tuple[int, ...]:
    """Draw one complete execution from the static BPOP model."""
    precedence = precedence_from_u(u)
    beta, epsilon = _check_beta_epsilon(beta, epsilon)
    m = precedence.shape[0]

    remaining = list(range(m))
    out: list[int] = []
    for _ in range(m):
        probs = _step_probabilities(tuple(remaining), precedence, beta, epsilon)
        support = np.array(remaining, dtype=int)
        weights = probs[support]
        weights = weights / weights.sum()  # guard against fp drift only
        y = int(rng.choice(support, p=weights))
        out.append(y)
        remaining.remove(y)
    return tuple(out)
