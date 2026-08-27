"""Condition B — structure identifiability under oracle paths.

Target (per the registered correction: rho is FIXED, not inferred, because the
formal corpus supplies U* directly and records rho* = null):

    p(U | X, S*, z*, vartheta*, rho_0)
        proportional to
    prod_k p(U_k | rho_0)  x  prod_{oracle blocks (n,l)} p_RFS(block | h(U_{z*_nl}))

Only U moves. S*, z*, all recurrent scalars, pi*, P*, delta_B*, epsilon* and
rho_0 are constants; p(S*|J, delta_B*) and p(z*|pi*, P*) do not enter any MH
ratio and are never computed here.

The likelihood reads U_k only through the induced order H_k = h(U_k)
(`recurrent_step_probabilities` consumes `precedence_from_u(u)` and nothing
else of u), so the oracle-block likelihood is cached per (skill, canonical H):
within-cell row moves are prior-only, and each new cell costs one production
batch replay of that skill's blocks from q_0 = 0.

The MH sweep is the VALIDATED `sampler_u.u_row_sweep` arithmetic — a
symmetric Gaussian row proposal accepted on Delta prior + Delta likelihood —
re-expressed with movement counters; a test drives both with the same RNG and
asserts bit-identical trajectories. Nothing from the collapsed-U or FFBS
modules is imported.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_joint_scalar_mcmc import vectorized_state_features
from hpop.mcmc_original.recurrent_scalar_posterior import cached_batch_log_likelihood
from hpop.mcmc_original.sampler_u import log_u_prior, propose_row

__all__ = [
    "canonical_h_hash", "transitive_reduction", "oracle_blocks_by_skill",
    "OracleBlockLikelihood", "ConditionBTarget", "ConditionBChain",
    "relation_indicator_vector", "closure_metrics", "reduction_metrics",
    "incomparable_metrics",
]


# ------------------------------------------------------------------ H utilities
def canonical_h_hash(precedence: np.ndarray) -> str:
    """Canonical identity of one induced order: sha256 of the closure bytes."""
    p = np.asarray(precedence, dtype=bool)
    return hashlib.sha256(p.astype(np.uint8).tobytes()).hexdigest()[:16]


def transitive_reduction(closure: np.ndarray) -> np.ndarray:
    """Reduction of a transitively closed DAG: drop edges implied by a 2-path."""
    p = np.asarray(closure, dtype=bool)
    implied = (p.astype(int) @ p.astype(int)) > 0
    return p & ~implied


def relation_indicator_vector(u_by_skill: np.ndarray) -> np.ndarray:
    """All K*m*(m-1) off-diagonal closure indicators, fixed (k, i, j) order."""
    out = []
    for k in range(u_by_skill.shape[0]):
        p = precedence_from_u(u_by_skill[k])
        m = p.shape[0]
        out.extend(bool(p[i, j]) for i in range(m) for j in range(m) if i != j)
    return np.asarray(out, dtype=bool)


def relation_pair_names(n_skills: int, n_roles: int) -> list:
    return [f"k{k}:{i}>{j}" for k in range(n_skills)
            for i in range(n_roles) for j in range(n_roles) if i != j]


# ------------------------------------------------------------- oracle likelihood
def oracle_blocks_by_skill(traces, n_skills: int) -> dict:
    """Group oracle blocks by (generating skill, width) for batch replay.

    ``traces`` are MatchedTrace records; role blocks are the observed CPA
    symbols (identity role maps), split at the TRUE boundaries with the TRUE
    labels — the oracle path. Returns {skill: {width: (n_blocks, width) int
    array}}.
    """
    grouped: dict = {k: {} for k in range(n_skills)}
    for trace in traces:
        for skill, block in zip(trace.labels, trace.role_blocks):
            grouped[int(skill)].setdefault(len(block), []).append(
                [int(v) for v in block])
    return {k: {w: np.asarray(blocks, dtype=int)
                for w, blocks in sorted(widths.items())}
            for k, widths in grouped.items()}


def _per_block_log_likelihood(features, beta, epsilon, lambda_rep,
                              lambda_back) -> np.ndarray:
    """Per-block version of the production `cached_batch_log_likelihood`.

    Same arithmetic, summed per block instead of over the whole batch; a test
    pins the sum to the production function to 1e-12.
    """
    exponent = (beta * features["Q"] - lambda_rep * features["q"]
                - lambda_back * features["C_back"])
    exponent = exponent - exponent.max(axis=-1, keepdims=True)
    weights = features["F"] * np.exp(exponent)
    structural = weights / weights.sum(axis=-1, keepdims=True)
    mixed = (1.0 - epsilon) * structural + epsilon / features["m"]
    n, T = features["obs"].shape
    chosen = mixed[np.arange(n)[:, None], np.arange(T)[None, :], features["obs"]]
    return np.log(chosen).sum(axis=1)


class OracleBlockLikelihood:
    """H-cached oracle-block log likelihood, production batch replay per miss.

    Every block is replayed from q_0 = 0 (`vectorized_state_features`
    initialises q to zeros); no state crosses block boundaries because blocks
    are independent rows of the batch.
    """

    def __init__(self, blocks_by_skill: dict, beta: float, epsilon: float,
                 omega: float, lambda_rep: float, lambda_back: float) -> None:
        self.blocks_by_skill = blocks_by_skill
        self.beta, self.epsilon = float(beta), float(epsilon)
        self.omega = float(omega)
        self.lambda_rep, self.lambda_back = float(lambda_rep), float(lambda_back)
        self._sum_cache: dict = {k: {} for k in blocks_by_skill}
        self._block_cache: dict = {k: {} for k in blocks_by_skill}
        self.evaluations = 0

    def _replay(self, skill: int, u_k: np.ndarray) -> np.ndarray:
        self.evaluations += 1
        parts = []
        for width, role_array in self.blocks_by_skill[skill].items():
            features = vectorized_state_features(role_array, u_k, self.omega)
            per_block = _per_block_log_likelihood(
                features, self.beta, self.epsilon, self.lambda_rep,
                self.lambda_back)
            check = cached_batch_log_likelihood(
                features, self.beta, self.epsilon, self.lambda_rep,
                self.lambda_back)
            if abs(float(per_block.sum()) - check) > 1e-9:
                raise RuntimeError("per-block/production batch parity broke")
            parts.append(per_block)
        return np.concatenate(parts) if parts else np.zeros(0)

    def skill_block_log_likelihoods(self, skill: int,
                                    u_k: np.ndarray) -> np.ndarray:
        key = precedence_from_u(u_k).tobytes()
        hit = self._block_cache[skill].get(key)
        if hit is None:
            hit = self._replay(skill, u_k)
            self._block_cache[skill][key] = hit
            self._sum_cache[skill][key] = float(hit.sum())
        return hit

    def skill_log_likelihood(self, skill: int, u_k: np.ndarray) -> float:
        key = precedence_from_u(u_k).tobytes()
        hit = self._sum_cache[skill].get(key)
        if hit is None:
            self.skill_block_log_likelihoods(skill, u_k)
            hit = self._sum_cache[skill][key]
        return hit

    def total(self, u_by_skill: np.ndarray) -> float:
        return float(sum(self.skill_log_likelihood(k, u_by_skill[k])
                         for k in range(u_by_skill.shape[0])))


# ------------------------------------------------------------------------ target
@dataclass
class ConditionBTarget:
    """log p(U | oracle blocks, rho_0), up to the constant (S*, z*) terms."""

    likelihood: OracleBlockLikelihood
    rho_0: float

    def log_prior(self, u_by_skill: np.ndarray) -> float:
        return float(sum(log_u_prior(u_by_skill[k], self.rho_0)
                         for k in range(u_by_skill.shape[0])))

    def log_target(self, u_by_skill: np.ndarray) -> float:
        return self.log_prior(u_by_skill) + self.likelihood.total(u_by_skill)


# ------------------------------------------------------------------------- chain
@dataclass
class ConditionBChain:
    """One Condition-B chain: row-wise U MH at fixed rho_0, with counters.

    The per-row arithmetic is exactly `sampler_u.u_row_sweep` (symmetric
    Gaussian row proposal from `propose_row`, accepted on Delta prior + Delta
    likelihood); a parity test drives both with one RNG stream.
    """

    target: ConditionBTarget
    u_by_skill: np.ndarray
    sigma_u: float
    rng: np.random.Generator
    sweep: int = 0
    proposed: int = 0
    accepted: int = 0
    h_change_proposed: int = 0
    h_change_accepted: int = 0
    esjd_sum: float = 0.0
    first_h_change_sweep: int | None = None
    _skill_ll: list = field(default_factory=list)
    _skill_prior: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.u_by_skill = np.array(self.u_by_skill, dtype=float, copy=True)
        self.n_skills = int(self.u_by_skill.shape[0])
        self._skill_ll = [self.target.likelihood.skill_log_likelihood(
            k, self.u_by_skill[k]) for k in range(self.n_skills)]
        self._skill_prior = [log_u_prior(self.u_by_skill[k], self.target.rho_0)
                             for k in range(self.n_skills)]

    def log_target(self) -> float:
        return float(sum(self._skill_ll) + sum(self._skill_prior))

    def run_sweeps(self, n_sweeps: int) -> None:
        for _ in range(int(n_sweeps)):
            self.sweep += 1
            for k in range(self.n_skills):
                u_k = self.u_by_skill[k]
                current_h = precedence_from_u(u_k)
                for row in range(u_k.shape[0]):
                    candidate = propose_row(u_k, row, self.sigma_u, self.rng)
                    self.proposed += 1
                    candidate_prior = log_u_prior(candidate, self.target.rho_0)
                    if candidate_prior == -math.inf:
                        continue
                    candidate_h = precedence_from_u(candidate)
                    h_differs = not np.array_equal(candidate_h, current_h)
                    if h_differs:
                        self.h_change_proposed += 1
                        candidate_ll = self.target.likelihood \
                            .skill_log_likelihood(k, candidate)
                    else:
                        candidate_ll = self._skill_ll[k]
                    log_alpha = ((candidate_prior - self._skill_prior[k])
                                 + (candidate_ll - self._skill_ll[k]))
                    if log_alpha >= 0.0 or math.log(self.rng.random()) < log_alpha:
                        jump = candidate[row] - u_k[row]
                        self.esjd_sum += float(np.dot(jump, jump))
                        u_k = candidate
                        self.u_by_skill[k] = candidate
                        self._skill_prior[k] = candidate_prior
                        self._skill_ll[k] = candidate_ll
                        self.accepted += 1
                        if h_differs:
                            self.h_change_accepted += 1
                            current_h = candidate_h
                            if self.first_h_change_sweep is None:
                                self.first_h_change_sweep = self.sweep

    # --------------------------------------------------------------- summaries
    def summary_row(self) -> dict:
        indicators = relation_indicator_vector(self.u_by_skill)
        per_skill = [int(precedence_from_u(self.u_by_skill[k]).sum())
                     for k in range(self.n_skills)]
        h_hashes = tuple(canonical_h_hash(precedence_from_u(self.u_by_skill[k]))
                         for k in range(self.n_skills))
        return {
            "log_posterior": self.log_target(),
            "log_prior": float(sum(self._skill_prior)),
            "relation_indicators": indicators,
            "per_skill_relations": per_skill,
            "total_relations": int(sum(per_skill)),
            "h_hashes": h_hashes,
        }

    # --------------------------------------------------------- checkpoint/resume
    def checkpoint(self) -> dict:
        return {
            "u_by_skill": self.u_by_skill.tolist(),
            "sigma_u": self.sigma_u,
            "sweep": self.sweep, "proposed": self.proposed,
            "accepted": self.accepted,
            "h_change_proposed": self.h_change_proposed,
            "h_change_accepted": self.h_change_accepted,
            "esjd_sum": self.esjd_sum,
            "first_h_change_sweep": self.first_h_change_sweep,
            "rng_state": self.rng.bit_generator.state,
        }

    @classmethod
    def resume(cls, payload: dict, target: ConditionBTarget) -> "ConditionBChain":
        rng = np.random.default_rng()
        rng.bit_generator.state = payload["rng_state"]
        chain = cls(target=target,
                    u_by_skill=np.asarray(payload["u_by_skill"], dtype=float),
                    sigma_u=float(payload["sigma_u"]), rng=rng,
                    sweep=int(payload["sweep"]),
                    proposed=int(payload["proposed"]),
                    accepted=int(payload["accepted"]),
                    h_change_proposed=int(payload["h_change_proposed"]),
                    h_change_accepted=int(payload["h_change_accepted"]),
                    esjd_sum=float(payload["esjd_sum"]))
        chain.first_h_change_sweep = payload["first_h_change_sweep"]
        return chain


# ------------------------------------------------------------- recovery metrics
def closure_metrics(relation_marginals: np.ndarray, true_closure: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Ordered-pair closure precision/recall/F1/Hamming at a fixed threshold.

    ``relation_marginals`` is (m, m) with a zero diagonal; both arguments are
    CLOSURES — comparing a reduction with a closure is a category error and is
    handled by `reduction_metrics`.
    """
    r = np.asarray(relation_marginals, dtype=float)
    truth = np.asarray(true_closure, dtype=bool)
    m = truth.shape[0]
    off = ~np.eye(m, dtype=bool)
    predicted = (r >= threshold) & off
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if tp and np.isfinite(precision) and np.isfinite(recall) else
          (1.0 if tp + fp + fn == 0 else 0.0))
    return {"precision": precision, "recall": recall, "f1": f1,
            "hamming": int((predicted != (truth & off)).sum()),
            "exact": bool(np.array_equal(predicted, truth & off)),
            "n_true_pairs": int((truth & off).sum())}


def reduction_metrics(point_closure: np.ndarray,
                      true_closure: np.ndarray) -> dict:
    """Transitive-REDUCTION F1: both sides converted to reductions first."""
    predicted = transitive_reduction(np.asarray(point_closure, dtype=bool))
    truth = transitive_reduction(np.asarray(true_closure, dtype=bool))
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if tp and np.isfinite(precision) and np.isfinite(recall) else
          (1.0 if tp + fp + fn == 0 else 0.0))
    return {"precision": precision, "recall": recall, "f1": f1,
            "exact": bool(np.array_equal(predicted, truth))}


def incomparable_metrics(relation_marginals: np.ndarray,
                         true_closure: np.ndarray,
                         threshold: float = 0.5) -> dict:
    """Unordered incomparable-pair precision/recall/F1 at the fixed threshold."""
    r = np.asarray(relation_marginals, dtype=float)
    truth = np.asarray(true_closure, dtype=bool)
    m = truth.shape[0]
    tp = fp = fn = 0
    for i in range(m):
        for j in range(i + 1, m):
            true_inc = not truth[i, j] and not truth[j, i]
            pred_inc = r[i, j] < threshold and r[j, i] < threshold
            tp += true_inc and pred_inc
            fp += pred_inc and not true_inc
            fn += true_inc and not pred_inc
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if tp and np.isfinite(precision) and np.isfinite(recall) else
          (1.0 if tp + fp + fn == 0 else 0.0))
    return {"precision": precision, "recall": recall, "f1": f1,
            "n_true_incomparable": int(sum(
                1 for i in range(m) for j in range(i + 1, m)
                if not truth[i, j] and not truth[j, i]))}
