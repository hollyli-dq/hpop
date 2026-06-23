"""Partial orders for HPOP/BPOP inference.

A `Poset` is a strict partial order over a finite item set, stored as a transitive-closure
precedence matrix. Build it from explicit must-precede edges or from the BPOP latent-U
representation (a ≻ b iff U_a dominates U_b across all K dimensions — the intersection of K total
orders / poset dimension). Provides frontier/successor queries (for the frontier-softmax likelihood)
and exact linear-extension counting (small posets) for sanity checks.
"""
from __future__ import annotations

import itertools


class Poset:
    def __init__(self, items, edges):
        """items: iterable of hashable ids. edges: (a, b) means a MUST precede b."""
        self.items = list(items)
        self.idx = {x: i for i, x in enumerate(self.items)}
        n = len(self.items)
        prec = [[False] * n for _ in range(n)]
        for a, b in edges:
            prec[self.idx[a]][self.idx[b]] = True
        # transitive closure (Floyd–Warshall on reachability)
        for k in range(n):
            rk = prec[k]
            for i in range(n):
                if prec[i][k]:
                    ri = prec[i]
                    for j in range(n):
                        if rk[j]:
                            ri[j] = True
        self._prec = prec

    @classmethod
    def from_latent_U(cls, U):
        """U: dict item -> sequence of K floats. a precedes b iff U[a] strictly dominates U[b]."""
        items = list(U)
        edges = []
        for a in items:
            ua = U[a]
            for b in items:
                if a is b:
                    continue
                ub = U[b]
                if all(ua[k] > ub[k] for k in range(len(ua))):
                    edges.append((a, b))
        return cls(items, edges)

    def precedes(self, a, b):
        return self._prec[self.idx[a]][self.idx[b]]

    def predecessors(self, x):
        j = self.idx[x]
        return [self.items[i] for i in range(len(self.items)) if self._prec[i][j]]

    def successors(self, x):
        i = self.idx[x]
        return [self.items[j] for j in range(len(self.items)) if self._prec[i][j]]

    def n_successors(self, x):
        return sum(self._prec[self.idx[x]])

    def frontier(self, completed):
        """Minimal not-yet-completed items: those whose predecessors are all completed."""
        comp = set(completed)
        out = []
        for x in self.items:
            if x in comp:
                continue
            if all(p in comp for p in self.predecessors(x)):
                out.append(x)
        return out

    def incomparable_pairs(self):
        return [(a, b) for a, b in itertools.combinations(self.items, 2)
                if not self.precedes(a, b) and not self.precedes(b, a)]

    def is_linear_extension(self, seq):
        if set(seq) != set(self.items):
            return False
        pos = {x: t for t, x in enumerate(seq)}
        return all(pos[a] < pos[b] for a in self.items for b in self.successors(a))

    def num_linear_extensions(self):
        """Exact count via bitmask DP. #P-hard in general; fine for small posets (sanity checks)."""
        n = len(self.items)
        predmask = [0] * n
        for j in range(n):
            for i in range(n):
                if self._prec[i][j]:
                    predmask[j] |= (1 << i)
        dp = [0] * (1 << n)
        dp[0] = 1
        for mask in range(1 << n):
            d = dp[mask]
            if not d:
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                if (predmask[j] & mask) == predmask[j]:
                    dp[mask | (1 << j)] += d
        return dp[(1 << n) - 1]
