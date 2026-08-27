"""Condition A — exact semi-Markov posterior over (S, z) under oracle structures.

The target is

    p(S, z | x, U*, vartheta*, pi*, P*, delta_B*, epsilon*)

with EVERYTHING except the per-trace segmentation ``S`` and skill path ``z``
fixed to generating truth. Each trace is conditionally independent, so the
posterior is computed exactly per trace by semi-Markov dynamic programming:

    forward   F[b, k]   = log-weight of all legal prefixes of x[0:b] whose last
                          block carries skill k;
    backward  Bwd[a, k] = log-weight of all legal suffixes of x[a:J] given the
                          previous block carried skill k;
    log Z     = logsumexp_k F[J, k].

The per-state weight is the registered Stage 6E decomposition (unnormalized
boundary prior + path prior + production recurrent block scores from q_0 = 0),
identical to ``stage6e_exact.state_log_weights`` — that module is the
enumeration reference these recursions are gated against on tiny traces.

Everything a metric needs is exact where the DP can make it exact: block,
boundary, occurrence-label and transition marginals, the segment-count
posterior, the Viterbi (max-product) MAP labelled path, the posterior
probability of the true path / true segmentation, and the exact posterior path
entropy via the linear decomposition of log w. FFBS backward sampling from the
frozen forward chart yields iid posterior draws for nonlinear summaries only.

A ``NullScorer`` (every block scores 0) turns the same machinery into the
prior-only p(S, z | J) reference used for the prior-vs-posterior comparison;
its log Z then equals the segmentation-prior normalizer log C_J plus nothing,
which the tests cross-check against the independent combinatorial reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from hpop.mcmc_original.targets import logsumexp

__all__ = [
    "NullScorer", "SemiMarkovPosterior", "boundary_f1", "segmentation_voi",
    "auroc", "average_precision", "calibration_table", "expected_calibration_error",
    "adjusted_rand_index", "normalized_mutual_information",
]

_NEG_INF = -math.inf


class NullScorer:
    """Scores every candidate block 0 — the prior-only reference model."""

    def score(self, trace: int, start: int, end: int, skill: int) -> float:
        return 0.0


def boundary_f1(predicted: set, truth: set) -> float:
    """F1 between two internal-cut sets; 1.0 when both are empty."""
    if not predicted and not truth:
        return 1.0
    tp = len(predicted & truth)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return 2 * precision * recall / (precision + recall)


def segmentation_voi(ends_a, ends_b, length: int) -> float:
    """Variation of information between two block partitions of [0, J)."""
    def blocks(ends):
        out, start = [], 0
        for end in ends:
            out.append((start, end))
            start = end
        return out
    a_blocks, b_blocks = blocks(ends_a), blocks(ends_b)
    n = float(length)
    h_a = -sum(((e - s) / n) * math.log((e - s) / n) for s, e in a_blocks)
    h_b = -sum(((e - s) / n) * math.log((e - s) / n) for s, e in b_blocks)
    mutual = 0.0
    for sa, ea in a_blocks:
        for sb, eb in b_blocks:
            overlap = min(ea, eb) - max(sa, sb)
            if overlap > 0:
                p = overlap / n
                mutual += p * math.log(p / (((ea - sa) / n) * ((eb - sb) / n)))
    return h_a + h_b - 2.0 * mutual


def auroc(scores, outcomes) -> float:
    """Mann-Whitney AUROC with average ranks for ties. NaN if one class absent."""
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    n_pos = int(outcomes.sum())
    n_neg = int(len(outcomes) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_sum_pos = float(ranks[outcomes == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(scores, outcomes) -> float:
    """AUPRC as average precision: sum of precision x recall increments."""
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    n_pos = int(outcomes.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = outcomes[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / n_pos)


def calibration_table(probabilities, outcomes, n_bins: int = 10) -> list:
    """Equal-width reliability table over [0, 1]."""
    probabilities = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    table = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = ((probabilities >= lo) & (probabilities < hi)) if i < n_bins - 1 \
            else ((probabilities >= lo) & (probabilities <= hi))
        count = int(mask.sum())
        table.append({
            "bin": [float(lo), float(hi)], "count": count,
            "mean_predicted": float(probabilities[mask].mean()) if count else None,
            "empirical_rate": float(outcomes[mask].mean()) if count else None,
        })
    return table


def expected_calibration_error(probabilities, outcomes, n_bins: int = 10) -> float:
    table = calibration_table(probabilities, outcomes, n_bins)
    total = sum(row["count"] for row in table)
    return float(sum(row["count"] / total
                     * abs(row["mean_predicted"] - row["empirical_rate"])
                     for row in table if row["count"]))


def _contingency(labels_a, labels_b) -> np.ndarray:
    labels_a = np.asarray(labels_a, dtype=int)
    labels_b = np.asarray(labels_b, dtype=int)
    n_a, n_b = labels_a.max() + 1, labels_b.max() + 1
    table = np.zeros((n_a, n_b))
    for a, b in zip(labels_a, labels_b):
        table[a, b] += 1
    return table


def adjusted_rand_index(labels_true, labels_pred) -> float:
    """ARI on the FIXED label identities — no alignment of any kind."""
    table = _contingency(labels_true, labels_pred)
    n = table.sum()
    if n < 2:
        return float("nan")
    def comb2(x):
        return x * (x - 1) / 2.0
    sum_cells = comb2(table).sum()
    sum_rows = comb2(table.sum(axis=1)).sum()
    sum_cols = comb2(table.sum(axis=0)).sum()
    expected = sum_rows * sum_cols / comb2(n)
    maximum = 0.5 * (sum_rows + sum_cols)
    if maximum == expected:
        return 1.0
    return float((sum_cells - expected) / (maximum - expected))


def normalized_mutual_information(labels_true, labels_pred) -> float:
    """NMI (arithmetic normalization) on fixed identities — no alignment."""
    table = _contingency(labels_true, labels_pred)
    n = table.sum()
    p = table / n
    pa, pb = p.sum(axis=1), p.sum(axis=0)
    mutual = 0.0
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if p[i, j] > 0:
                mutual += p[i, j] * math.log(p[i, j] / (pa[i] * pb[j]))
    h_a = -sum(v * math.log(v) for v in pa if v > 0)
    h_b = -sum(v * math.log(v) for v in pb if v > 0)
    if h_a == 0.0 and h_b == 0.0:
        return 1.0
    denominator = 0.5 * (h_a + h_b)
    return float(mutual / denominator) if denominator > 0 else 0.0


@dataclass
class SemiMarkovPosterior:
    """Exact per-trace posterior over (S, z) with all structures fixed to truth."""

    trace_index: int
    trace_length: int
    scorer: object                     # production RecurrentBlockScorer or NullScorer
    log_pi: np.ndarray
    log_transition: np.ndarray         # -inf diagonal
    delta_b: float
    min_width: int
    max_width: int
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.J = int(self.trace_length)
        self.K = int(len(self.log_pi))
        self.log_db = math.log(self.delta_b)
        self.log_1mdb = math.log1p(-self.delta_b)
        self.log_pi = np.asarray(self.log_pi, dtype=float)
        self.log_transition = np.asarray(self.log_transition, dtype=float)
        self._forward()
        self._backward()

    # ------------------------------------------------------------------ weights
    def block_term(self, a: int, b: int, k: int) -> float:
        """Score + internal non-cut positions of one block: the per-block weight."""
        return (float(self.scorer.score(self.trace_index, a, b, k))
                + (b - a - 1) * self.log_1mdb)

    def _legal_starts(self, b: int):
        for w in range(self.min_width, min(self.max_width, b) + 1):
            a = b - w
            if a == 0 or a >= self.min_width:
                yield a

    def _prefix_entry(self, a: int, k: int) -> float:
        """log weight of everything before a block (a, ., k): pi or F-transitions."""
        if a == 0:
            return float(self.log_pi[k])
        terms = [self.F[a, other] + self.log_db + float(self.log_transition[other, k])
                 for other in range(self.K)
                 if other != k and self.F[a, other] != _NEG_INF]
        return logsumexp(terms) if terms else _NEG_INF

    # ---------------------------------------------------------------- recursions
    def _forward(self) -> None:
        self.F = np.full((self.J + 1, self.K), _NEG_INF)
        for b in range(self.min_width, self.J + 1):
            for k in range(self.K):
                terms = []
                for a in self._legal_starts(b):
                    prefix = self._prefix_entry(a, k)
                    if prefix != _NEG_INF:
                        terms.append(prefix + self.block_term(a, b, k))
                if terms:
                    self.F[b, k] = logsumexp(terms)
        finite = self.F[self.J][np.isfinite(self.F[self.J])]
        if finite.size == 0:
            raise RuntimeError(f"trace {self.trace_index}: no legal segmentation")
        self.log_z = float(logsumexp(self.F[self.J]))

    def _backward(self) -> None:
        self.Bwd = np.full((self.J + 1, self.K), _NEG_INF)
        self.Bwd[self.J, :] = 0.0
        starts = [a for a in range(self.J - self.min_width, self.min_width - 1, -1)]
        for a in starts:
            for k in range(self.K):        # k = skill of the PREVIOUS block
                terms = []
                for w in range(self.min_width, min(self.max_width, self.J - a) + 1):
                    b = a + w
                    if b != self.J and self.J - b < self.min_width:
                        continue
                    for nxt in range(self.K):
                        if nxt == k or self.Bwd[b, nxt] == _NEG_INF:
                            continue
                        terms.append(self.log_db + float(self.log_transition[k, nxt])
                                     + self.block_term(a, b, nxt) + self.Bwd[b, nxt])
                if terms:
                    self.Bwd[a, k] = logsumexp(terms)

    # ------------------------------------------------------------------ marginals
    def block_log_marginal(self, a: int, b: int, k: int) -> float:
        prefix = self._prefix_entry(a, k)
        if prefix == _NEG_INF or self.Bwd[b, k] == _NEG_INF:
            return _NEG_INF
        return prefix + self.block_term(a, b, k) + self.Bwd[b, k] - self.log_z

    def iter_candidate_blocks(self):
        for b in range(self.min_width, self.J + 1):
            if b != self.J and self.J - b < self.min_width:
                continue
            for a in self._legal_starts(b):
                for k in range(self.K):
                    yield a, b, k

    def block_marginals(self) -> dict:
        out = {}
        for a, b, k in self.iter_candidate_blocks():
            value = self.block_log_marginal(a, b, k)
            if value != _NEG_INF:
                out[(a, b, k)] = math.exp(value)
        return out

    def boundary_marginals(self) -> np.ndarray:
        """p(a cut falls at internal position t), t = 1 .. J-1."""
        out = np.zeros(self.J - 1)
        for (a, b, _k), p in self.block_marginals().items():
            if b < self.J:
                out[b - 1] += p
        return out

    def occurrence_label_marginals(self) -> np.ndarray:
        """p(c_t = k) per occurrence t, exact, rows sum to one."""
        out = np.zeros((self.J, self.K))
        for (a, b, k), p in self.block_marginals().items():
            out[a:b, k] += p
        return out

    def expected_transition_counts(self) -> np.ndarray:
        """E[# adjacent (k' -> k) transitions | x]."""
        out = np.zeros((self.K, self.K))
        for t in range(self.min_width, self.J - self.min_width + 1):
            for k in range(self.K):
                inner = []
                for w in range(self.min_width, min(self.max_width, self.J - t) + 1):
                    b = t + w
                    if b != self.J and self.J - b < self.min_width:
                        continue
                    if self.Bwd[b, k] != _NEG_INF:
                        inner.append(self.block_term(t, b, k) + self.Bwd[b, k])
                if not inner:
                    continue
                suffix = logsumexp(inner)
                for prev in range(self.K):
                    if prev == k or self.F[t, prev] == _NEG_INF:
                        continue
                    out[prev, k] += math.exp(
                        self.F[t, prev] + self.log_db
                        + float(self.log_transition[prev, k]) + suffix - self.log_z)
        return out

    def segment_count_posterior(self) -> np.ndarray:
        """Exact p(L | x), by an L-augmented forward recursion."""
        max_l = self.J // self.min_width
        FL = np.full((self.J + 1, self.K, max_l + 1), _NEG_INF)
        for b in range(self.min_width, self.J + 1):
            for k in range(self.K):
                for a in self._legal_starts(b):
                    bt = self.block_term(a, b, k)
                    if a == 0:
                        value = float(self.log_pi[k]) + bt
                        FL[b, k, 1] = np.logaddexp(FL[b, k, 1], value)
                        continue
                    for other in range(self.K):
                        if other == k:
                            continue
                        link = self.log_db + float(self.log_transition[other, k]) + bt
                        for L in range(2, max_l + 1):
                            head = FL[a, other, L - 1]
                            if head != _NEG_INF:
                                FL[b, k, L] = np.logaddexp(FL[b, k, L], head + link)
        by_count = np.full(max_l + 1, _NEG_INF)
        for L in range(1, max_l + 1):
            finite = FL[self.J, :, L][np.isfinite(FL[self.J, :, L])]
            if finite.size:
                by_count[L] = logsumexp(finite)
        finite = by_count[np.isfinite(by_count)]
        log_z_check = float(logsumexp(finite))
        if abs(log_z_check - self.log_z) > 1e-8:
            raise RuntimeError(
                f"trace {self.trace_index}: count-augmented log Z {log_z_check} "
                f"disagrees with forward log Z {self.log_z}")
        out = np.zeros(max_l + 1)
        mask = np.isfinite(by_count)
        out[mask] = np.exp(by_count[mask] - self.log_z)
        return out

    # ------------------------------------------------------------- path evidence
    def path_log_weight(self, ends, labels) -> float:
        """log w(S, z): the registered unnormalized target of one labelled path."""
        ends = [int(e) for e in ends]
        labels = [int(z) for z in labels]
        total, start = float(self.log_pi[labels[0]]), 0
        for i, (end, k) in enumerate(zip(ends, labels)):
            total += self.block_term(start, end, k)
            if i > 0:
                total += self.log_db + float(self.log_transition[labels[i - 1], k])
            start = end
        return total

    def true_path_log_posterior(self, ends, labels) -> float:
        return self.path_log_weight(ends, labels) - self.log_z

    def segmentation_log_posterior(self, ends) -> float:
        """log p(S | x) with labels marginalized: a label-chain DP along fixed S."""
        ends = [int(e) for e in ends]
        widths, start = [], 0
        for end in ends:
            widths.append((start, end))
            start = end
        v = np.array([float(self.log_pi[k]) + self.block_term(*widths[0], k)
                      for k in range(self.K)])
        for a, b in widths[1:]:
            nxt = np.full(self.K, _NEG_INF)
            for k in range(self.K):
                terms = [v[other] + self.log_db
                         + float(self.log_transition[other, k])
                         for other in range(self.K)
                         if other != k and v[other] != _NEG_INF]
                if terms:
                    nxt[k] = logsumexp(terms) + self.block_term(a, b, k)
            v = nxt
        return float(logsumexp(v[np.isfinite(v)])) - self.log_z

    def map_path(self) -> tuple:
        """Exact MAP labelled path by semi-Markov max-product with backtracking."""
        V = np.full((self.J + 1, self.K), _NEG_INF)
        back = {}
        for b in range(self.min_width, self.J + 1):
            for k in range(self.K):
                best, arg = _NEG_INF, None
                for a in self._legal_starts(b):
                    bt = self.block_term(a, b, k)
                    if a == 0:
                        value = float(self.log_pi[k]) + bt
                        if value > best:
                            best, arg = value, (a, None)
                    else:
                        for other in range(self.K):
                            if other == k or V[a, other] == _NEG_INF:
                                continue
                            value = (V[a, other] + self.log_db
                                     + float(self.log_transition[other, k]) + bt)
                            if value > best:
                                best, arg = value, (a, other)
                if arg is not None:
                    V[b, k] = best
                    back[(b, k)] = arg
        k = int(np.argmax(V[self.J]))
        best_score = float(V[self.J, k])
        ends, labels, b = [], [], self.J
        while True:
            ends.append(b)
            labels.append(k)
            a, prev = back[(b, k)]
            if prev is None:
                break
            b, k = a, prev
        ends.reverse()
        labels.reverse()
        return tuple(ends), tuple(labels), best_score - self.log_z

    def expected_log_weight(self) -> float:
        """E[log w(S, z) | x], exact via the linear decomposition of log w."""
        total = 0.0
        first = np.zeros(self.K)
        for (a, b, k), p in self.block_marginals().items():
            total += p * self.block_term(a, b, k)
            if a == 0:
                first[k] += p
        total += float(np.dot(first, self.log_pi))
        expected = self.expected_transition_counts()
        for prev in range(self.K):
            for k in range(self.K):
                if prev != k and expected[prev, k] > 0.0:
                    total += expected[prev, k] * (
                        self.log_db + float(self.log_transition[prev, k]))
        return total

    def path_entropy(self) -> float:
        """Exact posterior path entropy: log Z - E[log w]."""
        return self.log_z - self.expected_log_weight()

    # ----------------------------------------------------------------------- FFBS
    def _sampling_tables(self) -> dict:
        """Per-anchor categorical tables for exact backward sampling."""
        if "ffbs" in self._cache:
            return self._cache["ffbs"]
        last_entries, last_logp = [], []
        for k in range(self.K):
            for w in range(self.min_width, min(self.max_width, self.J) + 1):
                a = self.J - w
                if a != 0 and a < self.min_width:
                    continue
                prefix = self._prefix_entry(a, k)
                if prefix == _NEG_INF:
                    continue
                last_entries.append((a, k))
                last_logp.append(prefix + self.block_term(a, self.J, k))
        last_logp = np.asarray(last_logp)
        last_p = np.exp(last_logp - logsumexp(last_logp))
        tables = {"last": (last_entries, last_p / last_p.sum()), "prev": {}}
        for a in range(self.min_width, self.J - self.min_width + 1):
            for k in range(self.K):                 # k = skill of the block at a
                entries, logs = [], []
                for prev in range(self.K):
                    if prev == k:
                        continue
                    for w in range(self.min_width,
                                   min(self.max_width, a) + 1):
                        start = a - w
                        if start != 0 and start < self.min_width:
                            continue
                        prefix = self._prefix_entry(start, prev)
                        if prefix == _NEG_INF:
                            continue
                        logs.append(prefix + self.block_term(start, a, prev)
                                    + float(self.log_transition[prev, k]))
                        entries.append((start, prev))
                if entries:
                    logs = np.asarray(logs)
                    p = np.exp(logs - logsumexp(logs))
                    tables["prev"][(a, k)] = (entries, p / p.sum())
        self._cache["ffbs"] = tables
        return tables

    def ffbs_draw(self, rng: np.random.Generator) -> tuple:
        """One exact iid draw of (ends, labels) from the posterior."""
        tables = self._sampling_tables()
        entries, p = tables["last"]
        a, k = entries[int(rng.choice(len(entries), p=p))]
        ends, labels = [self.J], [k]
        while a != 0:
            entries, p = tables["prev"][(a, k)]
            start, prev = entries[int(rng.choice(len(entries), p=p))]
            ends.append(a)
            labels.append(prev)
            a, k = start, prev
        ends.reverse()
        labels.reverse()
        return tuple(ends), tuple(labels)
