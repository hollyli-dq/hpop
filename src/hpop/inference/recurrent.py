"""Recurrent relaxed frontier likelihood (HPOP, Sec. "Recurrent Relaxed Frontier Likelihood").

The BPOP frontier likelihood removes an item once it is emitted, so a trace can contain each item
at most once. Software-agent traces re-run tests, re-inspect files, and re-edit code, so the same
CPA role occurs many times inside one skill instance. This module replaces the shrinking
not-yet-emitted set by a **validity state** `q_t in [0,1]^M`: `q_t(x)` is the degree to which the
latest result of role `x` is still valid. Executing `y` sets `q(y)=1` and multiplicatively
invalidates the roles it dominates. The latent graph stays acyclic; only the execution recurs.

Paper equations implemented here (numbers refer to the manuscript):

    J~_omega(z,x) = D_U(z,x) * sigmoid(omega_zx)                       (soft-invalidation-score)
    q~_t(x)       = 1 if x = y_t, else q~_{t-1}(x) * (1 - J~(y_t, x))  (soft-validity-update)
    F~_t(x)       = prod_{z != x} [1 - D_U(z,x) (1 - q~_{t-1}(z))]     (soft-recurrent-frontier)
    S~_t(x)       = sum_{z != x} D_U(x,z) (1 - q~_{t-1}(z))            (recurrent-successor-count)
    Q~_t(x)       = log(1 + S~_t(x))                                   (recurrent-successor-utility)
    C_rep(x)      = q~_{t-1}(x)                                        (recurrent-repeat-cost)
    C_back(x)     = sum_{z != x} J~(x,z) q~_{t-1}(z)                   (recurrent-back-cost)
    W~_t(x)       = F~_t(x) exp{beta Q~ - lam_rep C_rep - lam_back C_back}
    p(y_t = x)    = W~_t(x) / sum_v W~_t(v)                            (recurrent-step-likelihood)

Two deviations from the manuscript, both required for the likelihood to be usable:

1. `eps` trembling-hand floor. With a hard precedence matrix `D in {0,1}` (or any relaxed `D` that
   saturates at 1), `F~_t(x)` is exactly 0 whenever a predecessor of `x` is stale, so an
   order-violating observation gives `log p = -inf` and the objective is undefined. We mix in
   `eps / M` uniform noise exactly as BPOP does. Set `eps=0` to recover the manuscript equation.
2. Everything is computed in the log domain. `F~` is a product over M terms and underflows to 0 in
   float64 for M >~ 40 with confident edges.

`D` is the relaxed pairwise precedence score `D_U(z,x) in [0,1]` ("z must precede x"); the hard
partial order is the special case `D = 1[z > x]` under transitive closure. Diagonals must be 0.
"""
from __future__ import annotations

import numpy as np

_LOG_FLOOR = 1e-300


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def hard_precedence_matrix(n_items, edges):
    """Transitive-closure precedence matrix `D[z, x] = 1` iff z must precede x."""
    D = np.zeros((n_items, n_items), dtype=float)
    for a, b in edges:
        D[a, b] = 1.0
    # Floyd-Warshall reachability
    for k in range(n_items):
        D = np.maximum(D, np.outer(D[:, k], D[k, :]))
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, 1.0)


def invalidation_matrix(D, omega):
    """J~[z, x] = D[z, x] * sigmoid(omega[z, x]) — Eq. (soft-invalidation-score).

    `omega` may be a scalar (shared invalidation strength) or an (M, M) array.
    """
    J = np.asarray(D, dtype=float) * sigmoid(omega)
    np.fill_diagonal(J, 0.0)
    return J


class RecurrentFrontier:
    """Revisitable local partial-order likelihood for one skill template.

    Parameters
    ----------
    D : (M, M) array of relaxed precedence scores, D[z, x] = "z must precede x".
    omega : scalar or (M, M) array, invalidation logits.
    beta : successor-utility temperature.
    lam_rep : cost of re-executing a role whose output is still valid.
    lam_back : cost of a move that invalidates currently-valid downstream roles.
    eps : trembling-hand floor (see module docstring). 0 reproduces the manuscript exactly.
    """

    def __init__(self, D, omega=0.0, beta=1.0, lam_rep=1.0, lam_back=0.5, eps=0.02,
                 theta=None):
        self.D = np.array(D, dtype=float)
        np.fill_diagonal(self.D, 0.0)
        self.M = self.D.shape[0]
        self.omega = omega
        self.J = invalidation_matrix(self.D, omega)
        self.beta = float(beta)
        self.lam_rep = float(lam_rep)
        self.lam_back = float(lam_back)
        self.eps = float(eps)
        # Per-role composition logits. The manuscript's Eq. (recurrent-unnormalized-weight) has no
        # composition term, but Fig. (overview) describes each skill by a "CPA-composition
        # distribution" *and* a local order. theta supplies that composition half; theta = 0
        # reproduces the manuscript weight exactly.
        self.theta = np.zeros(self.M) if theta is None else np.asarray(theta, dtype=float)

    # ---- one step -----------------------------------------------------------------------
    def step_logprobs(self, q):
        """Log p(y_t = x | q_{t-1}) for every x, as a length-M vector."""
        stale = 1.0 - q                                     # (M,)  1 - q_{t-1}(z)
        # F~_t(x) = prod_z [1 - D[z,x] * stale[z]]   -> log-domain column product
        A = 1.0 - self.D * stale[:, None]                   # A[z, x]
        np.fill_diagonal(A, 1.0)
        log_F = np.log(np.clip(A, _LOG_FLOOR, None)).sum(axis=0)
        S = self.D @ stale                                  # S~_t(x) = sum_z D[x,z] stale[z]
        C_back = self.J @ q                                 # C_back(x) = sum_z J[x,z] q[z]
        log_W = (log_F
                 + self.theta
                 + self.beta * np.log1p(S)
                 - self.lam_rep * q
                 - self.lam_back * C_back)
        log_W = log_W - log_W.max()
        W = np.exp(log_W)
        p = W / W.sum()
        if self.eps > 0.0:
            p = (1.0 - self.eps) * p + self.eps / self.M
        return np.log(np.clip(p, _LOG_FLOOR, None))

    def update(self, q, y):
        """Validity update, Eq. (soft-validity-update). Returns a new q; `q` is not mutated."""
        q_new = q * (1.0 - self.J[y, :])
        q_new[y] = 1.0
        return q_new

    # ---- whole sequence -----------------------------------------------------------------
    def logp(self, seq, return_states=False):
        """Log p(y_{1:T} | H) for `seq`, a sequence of role indices in [0, M)."""
        q = np.zeros(self.M)
        total = 0.0
        states = []
        for y in seq:
            lp = self.step_logprobs(q)
            total += lp[y]
            if return_states:
                states.append((q.copy(), lp))
            q = self.update(q, y)
        return (total, states) if return_states else total

    def logp_and_theta_grad(self, seq):
        """(log p(seq), d log p / d theta). theta enters log W additively, so the gradient is the
        familiar softmax form, corrected for the eps mixture."""
        q = np.zeros(self.M)
        total = 0.0
        grad = np.zeros(self.M)
        for y in seq:
            stale = 1.0 - q
            A = 1.0 - self.D * stale[:, None]
            np.fill_diagonal(A, 1.0)
            log_W = (np.log(np.clip(A, _LOG_FLOOR, None)).sum(axis=0)
                     + self.theta
                     + self.beta * np.log1p(self.D @ stale)
                     - self.lam_rep * q
                     - self.lam_back * (self.J @ q))
            log_W = log_W - log_W.max()
            W = np.exp(log_W)
            p_soft = W / W.sum()
            p_mix = (1.0 - self.eps) * p_soft + self.eps / self.M
            total += float(np.log(max(p_mix[y], _LOG_FLOOR)))
            onehot = np.zeros(self.M)
            onehot[y] = 1.0
            grad += (1.0 - self.eps) * p_soft[y] * (onehot - p_soft) / max(p_mix[y], _LOG_FLOOR)
            q = self.update(q, y)
        return total, grad

    def frontier_rank(self, seq):
        """Rank of each observed step under the model (0 = model's top choice). Diagnostic."""
        q = np.zeros(self.M)
        ranks = []
        for y in seq:
            lp = self.step_logprobs(q)
            ranks.append(int((lp > lp[y]).sum()))
            q = self.update(q, y)
        return ranks

    def sample(self, rng, max_steps=64, stop_when_all_valid=True, fail_prob=0.0,
               fail_targets=None, verify_roles=()):
        """Sample an execution trace from the model.

        `fail_prob` / `verify_roles` inject *exogenous* failures: after executing a verify role,
        with probability `fail_prob` a designated upstream role is invalidated (a test failed).
        This is how real repair loops start; the model itself only charges them `lam_rep`.
        """
        q = np.zeros(self.M)
        seq = []
        for _ in range(max_steps):
            lp = self.step_logprobs(q)
            p = np.exp(lp)
            y = int(rng.choice(self.M, p=p / p.sum()))
            seq.append(y)
            q = self.update(q, y)
            if verify_roles and y in verify_roles and rng.random() < fail_prob:
                targets = fail_targets if fail_targets is not None else np.where(self.D[:, y] > 0)[0]
                for z in targets:
                    q[z] = 0.0
                q[y] = 0.0                                  # the failed verification is itself stale
            if stop_when_all_valid and np.all(q > 1.0 - 1e-9):
                break
        return seq


def sequence_logp(D, seq, omega=0.0, beta=1.0, lam_rep=1.0, lam_back=0.5, eps=0.02):
    """Convenience wrapper: log p(seq | D, omega, ...)."""
    return RecurrentFrontier(D, omega, beta, lam_rep, lam_back, eps).logp(seq)
