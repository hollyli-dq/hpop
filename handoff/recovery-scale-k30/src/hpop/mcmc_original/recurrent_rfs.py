"""Stage 6.0 — recurrent execution under the *same* acyclic ``h(U)``.

A role may be executed more than once. That is represented by an **execution validity
state**, never by adding cycles to the latent order: ``h(U)`` stays a strict acyclic
partial order and is still read off ``U`` by coordinate-wise dominance.

State. Before step ``u`` each role carries a validity ``q_{u-1}(x) in [0,1]``, with
``q_0 = 0``. Executing a role refreshes it to 1 and partially invalidates everything
downstream of it:

    q_u(x) = 1                                    if x == y_u
    q_u(x) = q_{u-1}(x) * (1 - J(y_u, x))         otherwise
    J(z, x) = 1[z > x] * kappa,   kappa = sigmoid(omega)

Invalidation uses the **full transitive closure**, so executing a role can stale a
descendant it does not directly cover.

Choice. A role is feasible to the extent its predecessors are currently valid:

    F_u(x) = prod_{z > x} q_{u-1}(z)          (empty product = 1, so minimal roles are 1)
    S_u(x) = sum_z 1[x > z] (1 - q_{u-1}(z))  stale successors
    Q_u(x) = log(1 + S_u(x))
    W_u(x) = F_u(x) * exp{ beta Q_u(x) - lambda_rep q_{u-1}(x) - lambda_back C_back_u(x) }
    C_back_u(x) = sum_{z != x} J(x, z) q_{u-1}(z)

``lambda_rep`` is a **relative penalty on currently valid candidates** — it is *not* a
monotone cost on any complete sequence containing a repeat, and nothing here claims it is.

Support. Executed roles remain available, so the candidate set is the complete role set
at every step and the trembling hand spreads over all ``m``:

    p_u(x) = (1 - eps) * pi_u(x) + eps / m,    pi_u = W_u / sum W_u

The block length ``T`` is **observed and conditioned upon**. There is no ``p(T | skill)``,
no duration model and no stopping rule, so the likelihood is

    p_RFS(y_1:T | T, U, omega, beta, lambda_rep, lambda_back, eps) = prod_u p_u(y_u)

which normalises over the ``m^T`` repeated-role sequences of that fixed length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u

__all__ = [
    "RecurrentRFSParameters", "sigmoid", "logit",
    "recurrent_feasibility", "recurrent_stale_successor_counts",
    "recurrent_successor_utilities", "recurrent_back_costs",
    "recurrent_structural_weights", "recurrent_step_probabilities",
    "recurrent_validity_update", "recurrent_rfs_log_likelihood",
    "recurrent_rfs_likelihood", "sample_recurrent_rfs_sequence",
    "replay_recurrent_sequence",
]


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x))) if x >= 0 else float(
        math.exp(x) / (1.0 + math.exp(x)))


def logit(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError(f"logit needs p in (0,1), got {p}")
    return float(math.log(p / (1.0 - p)))


@dataclass(frozen=True)
class RecurrentRFSParameters:
    beta: float
    epsilon: float
    shared_omega: float
    lambda_rep: float
    lambda_back: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError(f"beta must be finite and >= 0, got {self.beta}")
        if not math.isfinite(self.epsilon) or not (0.0 <= self.epsilon < 1.0):
            raise ValueError(f"epsilon must satisfy 0 <= eps < 1, got {self.epsilon}")
        if not math.isfinite(self.shared_omega):
            raise ValueError(f"shared_omega must be finite, got {self.shared_omega}")
        if not math.isfinite(self.lambda_rep) or self.lambda_rep < 0:
            raise ValueError(f"lambda_rep must be finite and >= 0, got {self.lambda_rep}")
        if not math.isfinite(self.lambda_back) or self.lambda_back < 0:
            raise ValueError(f"lambda_back must be finite and >= 0, got {self.lambda_back}")

    @property
    def kappa(self) -> float:
        """The shared invalidation gate."""
        return sigmoid(self.shared_omega)


def _check(precedence, q):
    p = np.asarray(precedence)
    if p.dtype != np.bool_ or p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError(f"precedence must be a square boolean matrix, got {p.shape}/{p.dtype}")
    q = np.asarray(q, dtype=float)
    if q.shape != (p.shape[0],):
        raise ValueError(f"q must have shape ({p.shape[0]},), got {q.shape}")
    if not np.all(np.isfinite(q)) or np.any(q < -1e-12) or np.any(q > 1 + 1e-12):
        raise ValueError(f"q must lie in [0,1], got {q}")
    return p, np.clip(q, 0.0, 1.0)


def recurrent_feasibility(precedence, q) -> np.ndarray:
    """``F(x) = prod_{z > x} q(z)``; 1 for roles with no predecessors."""
    p, q = _check(precedence, q)
    m = p.shape[0]
    return np.array([float(np.prod(q[p[:, x]])) if p[:, x].any() else 1.0
                     for x in range(m)])


def recurrent_stale_successor_counts(precedence, q) -> np.ndarray:
    """``S(x) = sum_z 1[x > z] (1 - q(z))``."""
    p, q = _check(precedence, q)
    return p.astype(float) @ (1.0 - q)


def recurrent_successor_utilities(precedence, q) -> np.ndarray:
    """``Q(x) = log(1 + S(x))``."""
    return np.log1p(recurrent_stale_successor_counts(precedence, q))


def recurrent_back_costs(precedence, q, shared_omega: float) -> np.ndarray:
    """``C_back(x) = sum_{z != x} J(x,z) q(z)`` — valid downstream work x would stale."""
    p, q = _check(precedence, q)
    off = p.copy()
    np.fill_diagonal(off, False)
    return sigmoid(shared_omega) * (off.astype(float) @ q)


def recurrent_structural_weights(precedence, q, beta, shared_omega,
                                 lambda_rep, lambda_back) -> np.ndarray:
    """Unnormalised ``W(x)``. Feasibility multiplies; the penalties sit in the exponent."""
    p, q = _check(precedence, q)
    feasibility = recurrent_feasibility(p, q)
    exponent = (beta * recurrent_successor_utilities(p, q)
                - lambda_rep * q
                - lambda_back * recurrent_back_costs(p, q, shared_omega))
    exponent = exponent - exponent.max()          # stable, cancels in the normalisation
    return feasibility * np.exp(exponent)


def recurrent_step_probabilities(u, q, params: RecurrentRFSParameters) -> np.ndarray:
    """``p_u(.)`` over the COMPLETE role set — executed roles stay available."""
    precedence = precedence_from_u(u)
    p, q = _check(precedence, q)
    m = p.shape[0]
    weights = recurrent_structural_weights(p, q, params.beta, params.shared_omega,
                                           params.lambda_rep, params.lambda_back)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError(
            "recurrent structural weights do not form a usable distribution.\n"
            f"  U          = {np.asarray(u).tolist()}\n"
            f"  precedence = {p.astype(int).tolist()}\n"
            f"  q          = {q.tolist()}\n"
            f"  F          = {recurrent_feasibility(p, q).tolist()}\n"
            f"  Q          = {recurrent_successor_utilities(p, q).tolist()}\n"
            f"  C_valid    = {q.tolist()}\n"
            f"  C_back     = {recurrent_back_costs(p, q, params.shared_omega).tolist()}\n"
            f"  W          = {weights.tolist()}\n"
            f"  sum(W)     = {total}")
    structural = weights / total
    return (1.0 - params.epsilon) * structural + params.epsilon / m


def recurrent_validity_update(observed_role: int, precedence, q_previous,
                              shared_omega: float) -> np.ndarray:
    """Refresh the executed role to 1; stale its transitive descendants by ``1 - kappa``."""
    p, q = _check(precedence, q_previous)
    m = p.shape[0]
    if not (0 <= int(observed_role) < m):
        raise ValueError(f"observed_role {observed_role} outside range({m})")
    observed_role = int(observed_role)
    kappa = sigmoid(shared_omega)
    gate = np.where(p[observed_role], kappa, 0.0)      # only descendants are gated
    q_new = q * (1.0 - gate)
    q_new[observed_role] = 1.0
    return q_new


def _validate_sequence(role_sequence, m):
    seq = [int(v) for v in role_sequence]
    if any(v < 0 or v >= m for v in seq):
        raise ValueError(f"role_sequence contains a role outside range({m}): {seq}")
    return seq                                        # repeats are allowed by design


def recurrent_rfs_log_likelihood(role_sequence, u, beta, epsilon, shared_omega,
                                 lambda_rep, lambda_back, return_trace: bool = False):
    """``log p_RFS(y_1:T | T, ...)``. Conditional on T; there is no duration model."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or not np.all(np.isfinite(u)):
        raise ValueError("u must be a finite 2-D array")
    params = RecurrentRFSParameters(beta, epsilon, shared_omega, lambda_rep, lambda_back)
    precedence = precedence_from_u(u)
    m = u.shape[0]
    seq = _validate_sequence(role_sequence, m)

    q = np.zeros(m)
    ever = np.zeros(m, dtype=bool)
    total = 0.0
    trace = []
    for index, y in enumerate(seq):
        feasibility = recurrent_feasibility(precedence, q)
        stale = recurrent_stale_successor_counts(precedence, q)
        utilities = np.log1p(stale)
        back = recurrent_back_costs(precedence, q, shared_omega)
        weights = recurrent_structural_weights(precedence, q, beta, shared_omega,
                                               lambda_rep, lambda_back)
        mixed = recurrent_step_probabilities(u, q, params)
        probability = float(mixed[y])
        if probability <= 0.0:
            return (-math.inf, trace) if return_trace else -math.inf
        total += math.log(probability)
        q_after = recurrent_validity_update(y, precedence, q, shared_omega)
        if return_trace:
            trace.append({
                "step_index": index, "q_before": q.copy(), "feasibility": feasibility.copy(),
                "stale_successor_counts": stale.copy(), "utilities": utilities.copy(),
                "valid_candidate_costs": q.copy(),
                "back_costs": back.copy(), "structural_weights": weights.copy(),
                "structural_probabilities": (weights / weights.sum()).copy(),
                "mixed_probabilities": mixed.copy(), "observed_role": y,
                "observed_probability": probability, "q_after": q_after.copy(),
                "ever_executed_before": bool(ever[y]),
                "observed_was_currently_valid": bool(q[y] > 0.0),
                "observed_back_cost": float(back[y]),
                "selected_via_structural_support": bool(feasibility[y] > 0.0),
            })
        ever[y] = True
        q = q_after
    return (total, trace) if return_trace else total


def recurrent_rfs_likelihood(role_sequence, u, beta, epsilon, shared_omega,
                             lambda_rep, lambda_back) -> float:
    value = recurrent_rfs_log_likelihood(role_sequence, u, beta, epsilon,
                                         shared_omega, lambda_rep, lambda_back)
    return 0.0 if value == -math.inf else math.exp(value)


def sample_recurrent_rfs_sequence(rng, length: int, u, params: RecurrentRFSParameters,
                                  return_trace: bool = False):
    """Draw one fixed-length recurrent block. ``length`` is supplied, never modelled."""
    u = np.asarray(u, dtype=float)
    precedence = precedence_from_u(u)
    m = u.shape[0]
    q = np.zeros(m)
    ever = np.zeros(m, dtype=bool)
    out, trace = [], []
    for index in range(int(length)):
        mixed = recurrent_step_probabilities(u, q, params)
        y = int(rng.choice(m, p=mixed))
        back = recurrent_back_costs(precedence, q, params.shared_omega)
        q_after = recurrent_validity_update(y, precedence, q, params.shared_omega)
        if return_trace:
            trace.append({
                "step_index": index, "q_before": q.copy(),
                "feasibility": recurrent_feasibility(precedence, q).copy(),
                "valid_candidate_costs": q.copy(), "back_costs": back.copy(),
                "mixed_probabilities": mixed.copy(), "observed_role": y,
                "observed_probability": float(mixed[y]), "q_after": q_after.copy(),
                "ever_executed_before": bool(ever[y]),
                "observed_was_currently_valid": bool(q[y] > 0.0),
                "observed_back_cost": float(back[y]),
            })
        out.append(y)
        ever[y] = True
        q = q_after
    sequence = tuple(out)
    return (sequence, trace) if return_trace else sequence


def replay_recurrent_sequence(role_sequence, u, params: RecurrentRFSParameters):
    """Independently re-derive the per-step record of an existing sequence."""
    return recurrent_rfs_log_likelihood(
        role_sequence, u, params.beta, params.epsilon, params.shared_omega,
        params.lambda_rep, params.lambda_back, return_trace=True)
