"""Robust frontier-softmax likelihood (BPOP).

Models an observed action sequence as a stochastic linear extension of a latent partial order `h`.
At step t the agent picks the next action from the current **frontier** F_t (minimal not-yet-emitted
items) via a Boltzmann-rational policy on the successor utility Q(a)=log(1+|successors(a)|), mixed
with an epsilon "trembling-hand" noise term over the remaining items so order-violating traces keep
positive probability. Cost O(T·|A|) — avoids #P-hard linear-extension marginalization.

    p(y_t | h) = (1-eps) * 1[y_t in F_t] * softmax_F(beta*Q)(y_t)  +  eps / |remaining_t|
"""
from __future__ import annotations

import math


def successor_utility(poset):
    return {x: math.log(1.0 + poset.n_successors(x)) for x in poset.items}


def frontier_softmax_logp(poset, seq, beta=1.0, eps=0.05, Q=None):
    """Log p(seq | h, beta, eps). `seq` is an observed ordering of (a subset of) poset.items."""
    if Q is None:
        Q = successor_utility(poset)
    comp = set()
    total = 0.0
    n = len(seq)
    for t, y in enumerate(seq):
        remaining = n - t                      # items still to emit, including y
        noise = eps / remaining
        F = poset.frontier(comp)
        if y in F:
            mx = max(Q[a] for a in F)           # softmax with max-shift for stability
            denom = sum(math.exp(beta * (Q[a] - mx)) for a in F)
            rational = math.exp(beta * (Q[y] - mx)) / denom
            p = (1.0 - eps) * rational + noise
        else:
            p = noise                           # structural violation: rational term is 0
        total += math.log(p)
        comp.add(y)
    return total


def flat_bpop_logp(poset, sequences, beta=1.0, eps=0.05):
    """Occurrence-level flat BPOP log-likelihood: sum of frontier-softmax over independent traces."""
    Q = successor_utility(poset)
    return sum(frontier_softmax_logp(poset, s, beta=beta, eps=eps, Q=Q) for s in sequences)
