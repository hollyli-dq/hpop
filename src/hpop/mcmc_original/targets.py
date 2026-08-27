"""The exact segmentation log-target, plus stable log-space numerics.

For a legal segmentation ``S`` with ``L`` segments over a trace of length ``T``:

    log p(x, S | U, delta_B, pi)
        = (L-1) log delta_B + (T-L) log(1 - delta_B)      boundary prior
        + log pi[z_1] + sum_l log P[z_{l-1}, z_l]         skill-path prior
        + sum_l log p_BPOP(roles(x[a_l:b_l]) | U_{z_l})   emissions

with no duration term and no composition multinomial. Any support-incompatible
segment sends the whole target to ``-inf``.

At ``delta_B = 0.5`` the boundary prior is ``0.5^(T-1)``, identical for every
segmentation of a given trace, so it cancels in all posterior ratios here. It is
computed explicitly anyway rather than being quietly dropped.

Without a transition matrix (Stages 0-2) every skill label is drawn from the same
uniform ``pi``, so the path prior is ``(1/K)^L``. Stage 3 supplies ``log_transition``
and only the first label uses ``pi``.

``SkillEvaluator`` is the machinery that makes this cheap enough for MCMC. The BPOP
likelihood reads ``U`` only through the induced partial order, and the toy skills have
at most 3 roles, so the entire likelihood is a table over the ``m!`` permutations,
cached keyed by the precedence matrix.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.static_bpop import bpop_log_likelihood
from hpop.mcmc_original.toy import map_cpa_block_to_roles
from hpop.mcmc_original.types import SkillTemplate

__all__ = [
    "logsumexp",
    "normalize_log_weights",
    "sample_categorical_from_log_weights",
    "log_boundary_prior",
    "log_path_prior",
    "SkillEvaluator",
    "log_target_segmentation",
]


# ---------------------------------------------------------------------------
# stable numerics
# ---------------------------------------------------------------------------


def logsumexp(values) -> float:
    """Stable ``log sum exp``; -inf if every entry is -inf."""
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        raise ValueError("logsumexp requires at least one value")
    if np.any(np.isposinf(a)):
        raise ValueError("logsumexp received +inf")
    shift = float(a.max())
    if shift == -math.inf:
        return -math.inf
    return shift + float(np.log(np.exp(a - shift).sum()))


def normalize_log_weights(log_weights) -> np.ndarray:
    """Unnormalised log weights -> probabilities, without exponentiating raw scores."""
    a = np.asarray(log_weights, dtype=float)
    total = logsumexp(a)
    if not math.isfinite(total):
        raise ValueError("cannot normalise log weights that are all -inf")
    return np.exp(a - total)


def sample_categorical_from_log_weights(log_weights, rng: np.random.Generator) -> int:
    """Draw one index with probability proportional to ``exp(log_weights)``."""
    return int(rng.choice(len(log_weights), p=normalize_log_weights(log_weights)))


# ---------------------------------------------------------------------------
# prior terms
# ---------------------------------------------------------------------------


def log_boundary_prior(trace_length: int, n_segments: int, delta_b: float) -> float:
    """``log[ delta_B^(L-1) (1 - delta_B)^(T-L) ]`` over the T-1 internal cuts."""
    if not (0.0 < delta_b < 1.0):
        raise ValueError(f"delta_b must be in (0, 1), got {delta_b}")
    if not (1 <= n_segments <= trace_length):
        raise ValueError(f"n_segments must be in [1, {trace_length}], got {n_segments}")
    return (n_segments - 1) * math.log(delta_b) + (trace_length - n_segments) * math.log(
        1.0 - delta_b
    )


def log_path_prior(
    path, log_pi: np.ndarray, log_transition: np.ndarray | None = None
) -> float:
    """``log pi[z_1] + sum_l log P[z_{l-1}, z_l]``.

    With ``log_transition=None`` every label is uniform under ``log_pi``.
    """
    path = [int(z) for z in path]
    if not path:
        raise ValueError("path must contain at least one skill label")
    total = float(log_pi[path[0]])
    for prev, nxt in zip(path[:-1], path[1:]):
        total += float(log_pi[nxt] if log_transition is None else log_transition[prev, nxt])
    return total


# ---------------------------------------------------------------------------
# cached per-skill likelihood
# ---------------------------------------------------------------------------


class SkillEvaluator:
    """Cached BPOP likelihood for one skill.

    Two independent savings, both exact:

    * identical executions are aggregated into permutation **counts**, so a corpus of
      200 sequences costs one dot product rather than 200 likelihood evaluations;
    * the permutation log-probability table is cached keyed by the induced precedence
      matrix, so a U-proposal that does not change the order costs nothing.
    """

    def __init__(self, template: SkillTemplate) -> None:
        self.template = template
        self.n_roles = template.n_roles
        self.cpa_labels = template.cpa_labels
        self.permutations: tuple[tuple[int, ...], ...] = tuple(
            itertools.permutations(range(self.n_roles))
        )
        self.perm_index: dict[tuple[int, ...], int] = {
            p: i for i, p in enumerate(self.permutations)
        }
        self._cache: dict[bytes, np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def log_table(self, u: np.ndarray) -> np.ndarray:
        """Log-probability of every role permutation under ``h(u)``."""
        key = precedence_from_u(u).tobytes()
        table = self._cache.get(key)
        if table is None:
            self.cache_misses += 1
            table = np.array(
                [
                    bpop_log_likelihood(
                        p, u, self.template.beta, self.template.epsilon
                    )
                    for p in self.permutations
                ],
                dtype=float,
            )
            self._cache[key] = table
        else:
            self.cache_hits += 1
        return table

    def roles_of(self, block) -> tuple[int, ...] | None:
        return map_cpa_block_to_roles(block, self.cpa_labels)

    def index_of_block(self, block) -> int | None:
        roles = self.roles_of(block)
        return None if roles is None else self.perm_index[roles]

    def count_sequences(self, role_sequences) -> np.ndarray:
        """Aggregate role sequences into a count vector over permutations."""
        counts = np.zeros(len(self.permutations), dtype=float)
        for s in role_sequences:
            counts[self.perm_index[tuple(int(v) for v in s)]] += 1.0
        return counts

    def counts_log_likelihood(self, u: np.ndarray, counts: np.ndarray) -> float:
        """Exact total log-likelihood of an aggregated corpus."""
        table = self.log_table(u)
        nz = counts > 0
        if not np.any(nz):
            return 0.0
        return float(np.dot(counts[nz], table[nz]))


def log_target_segmentation(
    x,
    segmentation,
    evaluators,
    u_by_skill,
    delta_b: float,
    log_pi: np.ndarray,
    log_transition: np.ndarray | None = None,
) -> float:
    """Full unnormalised log target of one legal segmentation."""
    path = [seg.skill for seg in segmentation.segments]
    emission = 0.0
    for seg in segmentation.segments:
        ev = evaluators[seg.skill]
        idx = ev.index_of_block(x[seg.start : seg.end])
        if idx is None:
            return -math.inf
        emission += float(ev.log_table(u_by_skill[seg.skill])[idx])
    return (
        log_boundary_prior(len(x), len(path), delta_b)
        + log_path_prior(path, log_pi, log_transition)
        + emission
    )
