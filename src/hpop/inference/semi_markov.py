"""Merge-only semi-Markov segmentation over LLM seed segments (HPOP, Sec. 5).

The upstream LLM produces a deliberately fine oversegmentation of a trace into `J` seed segments.
HPOP may only *merge* adjacent seeds, never split one, so every learned skill boundary lies in the
seed boundary set (Eq. merge-only-support). The latent segmentation is therefore a composition of
`J` into parts of width <= D_max, together with a skill label per part, subject to the canonical
adjacent-label convention `z_l != z_{l+1}` (Assumption ass:no-self-transition).

The score array has layout

    scores[i, w - 1, k]  =  score of the block covering 0-based seeds i .. i+w-1 with skill k

and is `-inf` where `i + w > J`. In HPOP this score is the expected generative segment score

    l_n(a, b, k) = E_q[log p~_RFS(x_{a:b} | H_k)] + E_q[log pi_k] - lambda_seg     (Eq. score)

Because both the model and the variational family are semi-Markov over the same seed lattice, the
exact structured posterior is available in closed form: setting the inference-network potential
`eta = l_n` makes `q_phi` the true conditional posterior, so the amortized network of Eq.
(structured-semi-markov-posterior) is not needed here. Everything below is exact
forward-backward/Viterbi on that lattice, cost O(J * D_max * K^2).
"""
from __future__ import annotations

import itertools

import numpy as np

NEG_INF = -np.inf


def _logsumexp(v, axis=None):
    v = np.asarray(v, dtype=float)
    m = np.max(v, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.exp(v - m).sum(axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis) if axis is not None else float(out.reshape(()))


def _leave_one_out_logsumexp(vec, log_trans=None):
    """loo[k] = logsumexp_{h != k} (vec[h] + log_trans[h, k]). Returns -inf when K == 1."""
    K = len(vec)
    if K == 1:
        return np.full(1, NEG_INF)
    mask = ~np.eye(K, dtype=bool)
    tiled = np.where(mask, vec[:, None], NEG_INF)          # tiled[h, k] = vec[h] for h != k
    if log_trans is not None:
        tiled = tiled + np.where(mask, log_trans, 0.0)
    return _logsumexp(tiled, axis=0)


class SemiMarkovLattice:
    """Exact inference over merge-only segmentations with a no-self-transition constraint.

    `log_trans[h, k]` (optional) is an additive log transition weight applied when a k-block follows
    an h-block, and `log_start[k]` when a k-block is first. With both omitted the lattice reduces to
    the label-marginal form used in Run 1. Supplying properly normalized rows turns the segmentation
    prior into a genuine semi-Markov process rather than the unnormalized `-lambda_seg` factor.
    """

    def __init__(self, scores, log_trans=None, log_start=None):
        self.scores = np.asarray(scores, dtype=float)       # (J, D_max, K)
        self.J, self.D_max, self.K = self.scores.shape
        self.log_trans = None if log_trans is None else np.asarray(log_trans, dtype=float)
        self.log_start = (np.zeros(self.K) if log_start is None
                          else np.asarray(log_start, dtype=float))
        # mask out blocks that run past the end of the trace
        for i in range(self.J):
            for w in range(1, self.D_max + 1):
                if i + w > self.J:
                    self.scores[i, w - 1, :] = NEG_INF

    # ---- forward ------------------------------------------------------------------------
    def forward(self):
        """alpha[j, k] = log mass of all segmentations of seeds 1..j ending in a k-block."""
        J, D, K = self.J, self.D_max, self.K
        alpha = np.full((J + 1, K), NEG_INF)
        inmass = np.full((J + 1, K), NEG_INF)
        inmass[0, :] = self.log_start                         # start state, no label constraint
        for j in range(1, J + 1):
            for k in range(K):
                terms = []
                for w in range(1, min(D, j) + 1):
                    s = self.scores[j - w, w - 1, k]
                    if s == NEG_INF or inmass[j - w, k] == NEG_INF:
                        continue
                    terms.append(inmass[j - w, k] + s)
                if terms:
                    alpha[j, k] = _logsumexp(np.array(terms))
            if j < J:
                inmass[j, :] = _leave_one_out_logsumexp(alpha[j, :], self.log_trans)
        self._alpha, self._inmass = alpha, inmass
        return alpha

    def log_partition(self):
        alpha = getattr(self, "_alpha", None)
        if alpha is None:
            alpha = self.forward()
        return _logsumexp(alpha[self.J, :])

    # ---- backward -----------------------------------------------------------------------
    def backward(self):
        """beta[j, k] = log mass of seeds j+1..J given a k-block ended at j."""
        J, D, K = self.J, self.D_max, self.K
        beta = np.full((J + 1, K), NEG_INF)
        beta[J, :] = 0.0
        for j in range(J - 1, -1, -1):
            for k in range(K):
                terms = []
                for w in range(1, min(D, J - j) + 1):
                    for kp in range(K):
                        if kp == k and j > 0:                # start state is unconstrained
                            continue
                        s = self.scores[j, w - 1, kp]
                        if s == NEG_INF or beta[j + w, kp] == NEG_INF:
                            continue
                        edge = (self.log_start[kp] if j == 0 else
                                (0.0 if self.log_trans is None else self.log_trans[k, kp]))
                        terms.append(s + edge + beta[j + w, kp])
                if terms:
                    beta[j, k] = _logsumexp(np.array(terms))
        self._beta = beta
        return beta

    def segment_marginals(self):
        """mu[i, w-1, k] = q(seed block i..i+w-1 is one instance of skill k). Eq. (segment-marginal)."""
        if not hasattr(self, "_alpha"):
            self.forward()
        if not hasattr(self, "_beta"):
            self.backward()
        logZ = self.log_partition()
        mu = np.full_like(self.scores, NEG_INF)
        for i in range(self.J):
            for w in range(1, min(self.D_max, self.J - i) + 1):
                for k in range(self.K):
                    s = self.scores[i, w - 1, k]
                    if s == NEG_INF:
                        continue
                    lp = self._inmass[i, k] + s + self._beta[i + w, k] - logZ
                    mu[i, w - 1, k] = lp
        return np.exp(mu)

    def transition_marginals(self):
        """xi[h, k] = expected number of h-block -> k-block transitions, and start[k].

        Exact: xi(j,h,k) = alpha[j,h] + logT[h,k] + LSE_w[score(j,w,k) + beta[j+w,k]] - logZ.
        """
        if not hasattr(self, "_alpha"):
            self.forward()
        if not hasattr(self, "_beta"):
            self.backward()
        logZ = self.log_partition()
        J, D, K = self.J, self.D_max, self.K
        xi = np.zeros((K, K))
        for j in range(1, J):                                # a transition happens strictly inside
            out = np.full(K, NEG_INF)                        # out[k] = LSE_w score+beta for k at j
            for k in range(K):
                terms = [self.scores[j, w - 1, k] + self._beta[j + w, k]
                         for w in range(1, min(D, J - j) + 1)
                         if np.isfinite(self.scores[j, w - 1, k]) and np.isfinite(self._beta[j + w, k])]
                if terms:
                    out[k] = _logsumexp(np.array(terms))
            for h in range(K):
                if not np.isfinite(self._alpha[j, h]):
                    continue
                for k in range(K):
                    if k == h or not np.isfinite(out[k]):
                        continue
                    t = self._alpha[j, h] + out[k] - logZ
                    if self.log_trans is not None:
                        t += self.log_trans[h, k]
                    xi[h, k] += np.exp(t)
        mu = self.segment_marginals()
        start = mu[0].sum(axis=0)                            # blocks starting at seed 0
        return xi, start

    def expected_num_segments(self):
        return float(self.segment_marginals().sum())

    def entropy(self):
        """H[q] = logZ - E_q[score], exact for an exponential family over segmentations."""
        mu = self.segment_marginals()
        finite = np.isfinite(self.scores)
        expected_score = float((mu[finite] * self.scores[finite]).sum())
        return self.log_partition() - expected_score

    # ---- decoding -----------------------------------------------------------------------
    def viterbi(self):
        """MAP segmentation: list of (start_seed, end_seed_exclusive, skill)."""
        J, D, K = self.J, self.D_max, self.K
        delta = np.full((J + 1, K), NEG_INF)
        inbest = np.full((J + 1, K), NEG_INF)
        inarg = np.full((J + 1, K), -1, dtype=int)
        back = {}
        inbest[0, :] = self.log_start
        for j in range(1, J + 1):
            for k in range(K):
                best, arg = NEG_INF, None
                for w in range(1, min(D, j) + 1):
                    s = self.scores[j - w, w - 1, k]
                    if s == NEG_INF or inbest[j - w, k] == NEG_INF:
                        continue
                    val = inbest[j - w, k] + s
                    if val > best:
                        best, arg = val, w
                delta[j, k], back[(j, k)] = best, arg
            if j < J:
                for k in range(K):
                    mask = np.arange(K) != k
                    cand = np.where(mask, delta[j, :], NEG_INF)
                    if self.log_trans is not None:
                        cand = cand + np.where(mask, self.log_trans[:, k], 0.0)
                    inarg[j, k] = int(np.argmax(cand))
                    inbest[j, k] = cand[inarg[j, k]]
        k = int(np.argmax(delta[J, :]))
        segs, j = [], J
        while j > 0:
            w = back[(j, k)]
            segs.append((j - w, j, k))
            prev = j - w
            k = int(inarg[prev, k]) if prev > 0 else k
            j = prev
        return list(reversed(segs)), float(delta[J, int(np.argmax(delta[J, :]))])


# ---- brute force reference (tests only) -------------------------------------------------
def enumerate_segmentations(J, D_max, K):
    """All (composition, labelling) pairs with parts <= D_max and no adjacent equal labels."""
    def compositions(n):
        if n == 0:
            yield ()
            return
        for w in range(1, min(D_max, n) + 1):
            for rest in compositions(n - w):
                yield (w,) + rest

    for comp in compositions(J):
        for labels in itertools.product(range(K), repeat=len(comp)):
            if any(labels[i] == labels[i + 1] for i in range(len(labels) - 1)):
                continue
            segs, i = [], 0
            for w, k in zip(comp, labels):
                segs.append((i, i + w, k))
                i += w
            yield segs


def _path_score(segs, scores, log_trans=None, log_start=None):
    t = sum(scores[i, (b - i) - 1, k] for i, b, k in segs)
    if log_start is not None:
        t += log_start[segs[0][2]]
    if log_trans is not None:
        t += sum(log_trans[a[2], b_[2]] for a, b_ in zip(segs, segs[1:]))
    return t


def brute_force_log_partition(scores, log_trans=None, log_start=None):
    scores = np.asarray(scores, dtype=float)
    J, D_max, K = scores.shape
    totals = [t for segs in enumerate_segmentations(J, D_max, K)
              if np.isfinite(t := _path_score(segs, scores, log_trans, log_start))]
    return _logsumexp(np.array(totals)) if totals else NEG_INF


def brute_force_marginals(scores, log_trans=None, log_start=None):
    scores = np.asarray(scores, dtype=float)
    J, D_max, K = scores.shape
    logZ = brute_force_log_partition(scores, log_trans, log_start)
    mu = np.zeros_like(scores)
    for segs in enumerate_segmentations(J, D_max, K):
        t = _path_score(segs, scores, log_trans, log_start)
        if not np.isfinite(t):
            continue
        p = np.exp(t - logZ)
        for i, b, k in segs:
            mu[i, (b - i) - 1, k] += p
    return mu


def brute_force_viterbi(scores, log_trans=None, log_start=None):
    scores = np.asarray(scores, dtype=float)
    best, best_segs = NEG_INF, None
    for segs in enumerate_segmentations(*scores.shape):
        t = _path_score(segs, scores, log_trans, log_start)
        if t > best:
            best, best_segs = t, segs
    return best_segs, best
