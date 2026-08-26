"""Shared-Gamma coupling: one master weight matrix, every rung a restriction of it.

## The problem this fixes

Skills are nested across the ladder -- rung `K` uses the first `K` skills of one master
library -- but until now each rung drew its own transition matrix independently. So "the
same skill at a larger `K`" shared its emission structure and shared nothing about how it
was reached or left. A trend across the ladder then mixes two causes that cannot be
separated after the fact: more skills to confuse, and a completely different transition
environment.

## The construction

For each replicate draw one master directed weight matrix

    G_ij ~ iid Gamma(1, 1),   i != j,   i, j < K_max

apply the master skill permutation to **both** axes, and cut each rung out of it by
restriction and renormalisation:

    P^(K)_ii = 0
    P^(K)_ij = G_ij / sum_{h < K, h != i} G_ih          i != j,  i, j < K

Two properties make this the right coupling rather than merely a coupling.

**The per-rung law is unchanged.** For `G_ij` iid `Gamma(1, 1)`, the normalised vector
`(G_ij / sum_h G_ih)_{h != i}` is exactly `Dirichlet_{K-1}(1, ..., 1)` -- the Gamma
representation of the Dirichlet. So each rung's rows still have precisely the registered
flat-Dirichlet marginal *before* admissibility conditioning; nothing about the marginal
model is traded away for the coupling.

**Relative preferences among old skills survive.** For destinations `j`, `l` both present
at the smaller rung,

    P^(K)_ij / P^(K)_il = G_ij / G_il

which does not depend on `K`. Growing the ladder dilutes every old destination by the same
factor -- the normaliser -- and reorders nothing.

## Joint conditioning, and why it must be joint

The stationary-occupancy band is applied to **all five rungs at once**: one master `G` is
drawn, all rungs are built from it, and the whole master draw is accepted or rejected
together. Rejecting and redrawing a single failing rung would replace that rung's `G` rows
with fresh ones and destroy exactly the coupling this module exists to create.

The consequence must be stated wherever the ladder is described: **the final ladder is
jointly conditioned across rungs.** No rung's transition matrix is a draw from the
unconditional flat-Dirichlet law; each is a draw from that law conditioned on the event
that *every* rung of the same master `G` satisfies the band. That is a stronger and
different conditioning than per-rung acceptance, and it is not interchangeable with it.
"""

from __future__ import annotations

import numpy as np

from .nested_library import K_LADDER
from .seeds import LadderSeeds

__all__ = ["MasterTransitions", "draw_master_gamma", "restrict_and_renormalise",
           "draw_master_transitions", "stationary_band", "K_LADDER"]


def stationary_band(k: int) -> tuple:
    """The registered balance band for a `k`-skill stationary law: `[0.5/k, 1.5/k]`."""
    k = int(k)
    return (0.5 / k, 1.5 / k)


def draw_master_gamma(rng: np.random.Generator, k_max: int) -> np.ndarray:
    """`(K_max, K_max)` iid `Gamma(1, 1)` off the diagonal, exact zero on it.

    Drawn as one `(K_max, K_max)` block with the diagonal overwritten rather than
    element-by-element, so the stream position does not depend on `K_max` in a way that
    would make a larger master library reshuffle a smaller one's weights.
    """
    k_max = int(k_max)
    g = rng.gamma(shape=1.0, scale=1.0, size=(k_max, k_max))
    np.fill_diagonal(g, 0.0)
    return g


def restrict_and_renormalise(g, k: int) -> np.ndarray:
    """Cut rung `k` out of the master weights: `P_ij = G_ij / sum_{h<k, h!=i} G_ih`."""
    g = np.asarray(g, dtype=float)
    k = int(k)
    if not 2 <= k <= g.shape[0]:
        raise ValueError(f"K must be in [2, {g.shape[0]}], got {k}")
    block = g[:k, :k].copy()
    np.fill_diagonal(block, 0.0)
    totals = block.sum(axis=1)
    if not np.all(totals > 0.0):
        raise ValueError(f"K={k}: a master row has zero total weight; "
                         f"rows {np.flatnonzero(totals <= 0).tolist()}")
    return block / totals[:, None]


class MasterTransitions:
    """One master `G` and the whole ladder cut from it, accepted or rejected together."""

    __slots__ = ("g", "g_unpermuted", "permutation", "ladder", "_transitions",
                 "_stationaries", "attempts", "accepted_attempt", "replicate", "k_max")

    def __init__(self, g, g_unpermuted, permutation, ladder, transitions, stationaries,
                 attempts, accepted_attempt, replicate, k_max):
        self.g = np.asarray(g, dtype=float)
        self.g_unpermuted = np.asarray(g_unpermuted, dtype=float)
        self.permutation = np.asarray(permutation, dtype=np.int64)
        self.ladder = tuple(int(k) for k in ladder)
        self._transitions = {int(k): np.asarray(v, dtype=float)
                             for k, v in transitions.items()}
        self._stationaries = {int(k): np.asarray(v, dtype=float)
                              for k, v in stationaries.items()}
        self.attempts = list(attempts)
        self.accepted_attempt = int(accepted_attempt)
        self.replicate = int(replicate)
        self.k_max = int(k_max)

    def transition(self, k: int) -> np.ndarray:
        k = int(k)
        if k not in self._transitions:
            raise KeyError(f"K={k} is not on this ladder {self.ladder}")
        return self._transitions[k].copy()

    def stationary(self, k: int) -> np.ndarray:
        k = int(k)
        if k not in self._stationaries:
            raise KeyError(f"K={k} is not on this ladder {self.ladder}")
        return self._stationaries[k].copy()

    def pi_p(self, k: int) -> tuple:
        """`(pi, P)` for rung `k`, the pair `generate_ladder_corpus` consumes."""
        return self.stationary(k), self.transition(k)

    @property
    def joint_acceptance_rate(self) -> float:
        """Accepted master draws over attempted ones. One accept, so `1 / attempts`."""
        return 1.0 / len(self.attempts) if self.attempts else float("nan")

    def provenance(self) -> dict:
        return {
            "coupling": "shared-Gamma",
            "construction": ("one master G_ij ~ iid Gamma(1,1) off-diagonal per "
                             "replicate, master skill permutation applied to BOTH axes, "
                             "each rung P^(K)_ij = G_ij / sum_{h<K, h!=i} G_ih"),
            "row_marginal_before_conditioning": "Dirichlet_{K-1}(1,...,1), exactly",
            "conditioning": ("JOINT across all rungs: one master G is accepted or "
                             "rejected as a whole against the stationary-occupancy band "
                             "at EVERY rung. The final ladder is jointly conditioned "
                             "across rungs; no rung is an unconditional draw, and this "
                             "is not interchangeable with per-rung acceptance."),
            "pi": "pi^(K) = stationary_distribution(P^(K))",
            "ladder": list(self.ladder),
            "k_max": self.k_max,
            "replicate": self.replicate,
            "bands": {str(k): list(stationary_band(k)) for k in self.ladder},
            "attempts": len(self.attempts),
            "accepted_attempt": self.accepted_attempt,
            "joint_acceptance_rate": self.joint_acceptance_rate,
            "attempt_log": self.attempts,
        }


def draw_master_transitions(replicate: int, permutation, k_max: int = 30,
                            ladder=K_LADDER, seeds: LadderSeeds | None = None,
                            max_attempts: int = 20_000) -> MasterTransitions:
    """One master `G` whose every rung satisfies the stationary band. Jointly conditioned.

    `permutation` is the master skill permutation the library already applied to `U` and
    to the role maps; it is applied here to both axes of `G` so that skill `i` means the
    same skill everywhere.
    """
    from .corpus import stationary_of

    seeds = LadderSeeds() if seeds is None else seeds
    replicate, k_max = int(replicate), int(k_max)
    ladder = tuple(int(k) for k in ladder)
    permutation = np.asarray(permutation, dtype=np.int64)
    if permutation.shape != (k_max,):
        raise ValueError(f"permutation must have shape ({k_max},), got "
                         f"{permutation.shape}")
    if max(ladder) > k_max:
        raise ValueError(f"ladder {ladder} exceeds K_max={k_max}")

    attempts = []
    for attempt in range(int(max_attempts)):
        rng = seeds.generator("master_structural_truth", replicate, 9_001, attempt)
        g_unpermuted = draw_master_gamma(rng, k_max)
        g = g_unpermuted[np.ix_(permutation, permutation)]

        transitions, stationaries, failed = {}, {}, []
        for k in ladder:
            p = restrict_and_renormalise(g, k)
            nu = stationary_of(p, k)
            if nu is None:
                failed.append({"K": k, "reason": "stationary law is not unique"})
                continue
            low, high = stationary_band(k)
            if not np.all((nu >= low) & (nu <= high)):
                failed.append({"K": k, "reason": "outside stationary band",
                               "min": float(nu.min()), "max": float(nu.max()),
                               "band": [low, high]})
                continue
            transitions[k], stationaries[k] = p, nu

        attempts.append({"attempt": attempt, "accepted": not failed,
                         "failed_rungs": failed[:4]})
        if not failed:
            return MasterTransitions(g, g_unpermuted, permutation, ladder, transitions,
                                     stationaries, attempts, attempt, replicate, k_max)

    rate = joint_band_acceptance_rate(permutation, k_max, ladder, seeds, trials=2_000)
    raise RuntimeError(
        f"replicate {replicate}: no master G whose stationary law satisfies the "
        f"registered band at EVERY rung of {ladder} in {max_attempts} attempts "
        f"(measured joint rate {rate:.5f}). Report and stop; do not widen the band and "
        f"do not fall back to per-rung acceptance, which would destroy the coupling.")


def joint_band_acceptance_rate(permutation, k_max: int = 30, ladder=K_LADDER,
                               seeds: LadderSeeds | None = None,
                               trials: int = 2_000) -> float:
    """Measured probability that one master `G` clears the band at every rung at once.

    Reported, never tuned on. It is the quantity that says whether joint conditioning is
    feasible at all, and it is strictly smaller than any single rung's rate.
    """
    from .corpus import stationary_of

    seeds = LadderSeeds() if seeds is None else seeds
    permutation = np.asarray(permutation, dtype=np.int64)
    ladder = tuple(int(k) for k in ladder)
    accepted = 0
    for trial in range(int(trials)):
        rng = seeds.generator("diagnostic", 9_001, int(trial))
        g = draw_master_gamma(rng, int(k_max))[np.ix_(permutation, permutation)]
        ok = True
        for k in ladder:
            nu = stationary_of(restrict_and_renormalise(g, k), k)
            low, high = stationary_band(k)
            if nu is None or not np.all((nu >= low) & (nu <= high)):
                ok = False
                break
        accepted += int(ok)
    return accepted / float(trials)


def per_rung_band_acceptance_rates(permutation, k_max: int = 30, ladder=K_LADDER,
                                   seeds: LadderSeeds | None = None,
                                   trials: int = 2_000) -> dict:
    """Each rung's own band rate under the shared-Gamma construction, for the record."""
    from .corpus import stationary_of

    seeds = LadderSeeds() if seeds is None else seeds
    permutation = np.asarray(permutation, dtype=np.int64)
    ladder = tuple(int(k) for k in ladder)
    hits = {k: 0 for k in ladder}
    for trial in range(int(trials)):
        rng = seeds.generator("diagnostic", 9_001, int(trial))
        g = draw_master_gamma(rng, int(k_max))[np.ix_(permutation, permutation)]
        for k in ladder:
            nu = stationary_of(restrict_and_renormalise(g, k), k)
            low, high = stationary_band(k)
            if nu is not None and np.all((nu >= low) & (nu <= high)):
                hits[k] += 1
    return {k: hits[k] / float(trials) for k in ladder}
