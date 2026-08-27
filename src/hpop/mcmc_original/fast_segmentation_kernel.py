"""Stage 6E — the same segmentation proposal law, on a representation that is cheap.

`Stage6EMoveKernel` is correct and is the reference, but it is not affordable: it
materialises every neighbour as a validated `Segmentation`, and `proposal_distribution`
does that on **both** sides of every Metropolis step. Measured on the registered widths
that is 4.3 ms per attempt at `J = 48` and 10.3 ms at `J = 96` — which puts the §14
configuration (100 traces x 50,000 sweeps x 4 chains) far out of reach.

The cost is representation, not mathematics. A segmentation that covers `[0, J)` without
gaps or overlaps is determined by its cut positions and labels, so

    key = ((end_0, skill_0), (end_1, skill_1), ..., (J, skill_{L-1}))

carries exactly the same information as the `Segmentation` object, with starts implied by
the previous end. Enumerating neighbours over keys is tuple manipulation; enumerating them
over `Segmentation` pays `Segment.__post_init__` and `Segmentation.__post_init__` for every
segment of every neighbour.

**Nothing about the proposal law changes.** The move types, the legality rules, the
neighbourhood contents and the multiplicities — including the case where two different
boundaries reach the same state, which the Stage 5 docstring calls out — are identical by
construction, because this module enumerates the same moves in the same order. That claim
is not left as a comment: `assert_kernels_agree` compares the two distributions state by
state, and the Stage 6E tests run it over random segmentations.

Legality is enforced here rather than inherited from `Segmentation.__post_init__`, so the
invariants are checked explicitly:

* coverage and contiguity are structural — a key is a strictly increasing sequence of ends
  finishing at `J`, so gaps and overlaps cannot be represented at all;
* every width lies in `[min_width, max_width]`;
* every label lies in `[0, K)`;
* no two adjacent segments share a label, because `P` has a zero diagonal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from hpop.mcmc_original.proposals import MoveType
from hpop.mcmc_original.stage6e_frozen import MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH
from hpop.mcmc_original.types import Segment, Segmentation

__all__ = [
    "key_of", "segmentation_of", "spans_of", "FastSegmentationKernel",
    "assert_kernels_agree", "assert_proposal_prob_agrees",
]


# ------------------------------------------------------------------- representation
def key_of(segmentation: Segmentation) -> tuple:
    """`Segmentation` -> `((end, skill), ...)`. Starts are implied by the previous end."""
    return tuple((int(s.end), int(s.skill)) for s in segmentation.segments)


def segmentation_of(key) -> Segmentation:
    """`((end, skill), ...)` -> `Segmentation`, validated on the way out."""
    segments, start = [], 0
    for end, skill in key:
        segments.append(Segment(int(start), int(end), int(skill)))
        start = int(end)
    return Segmentation(tuple(segments))


def spans_of(key) -> list[tuple[int, int, int]]:
    """`[(start, end, skill), ...]`."""
    out, start = [], 0
    for end, skill in key:
        out.append((start, int(end), int(skill)))
        start = int(end)
    return out


# ------------------------------------------------------------------------ the kernel
@dataclass
class FastSegmentationKernel:
    """The Stage 6E proposal law over keys. Same q, same multiplicities, ~50x cheaper."""

    trace_length: int
    n_skills: int
    min_width: int = MIN_BLOCK_WIDTH
    max_width: int = MAX_BLOCK_WIDTH
    max_shift: int | None = None
    move_probabilities: dict | None = None
    # The neighbour cache is bounded and cleared wholesale when it fills. It is a pure
    # function of `(move, key)`, so discarding it changes speed and nothing else.
    #
    # The bound is small on purpose, and the number matters. One cached key costs about
    # 40 KB across the four move types at the registered widths (`shift` alone yields ~84
    # neighbours on a five-block trace), and the Stage 6E2 sampler holds **one kernel per
    # trace** — 100 of them. An unbounded or generously bounded cache is therefore not a
    # minor inefficiency but hundreds of gigabytes over a 50,000-sweep run. The working
    # set a Metropolis step actually needs is tiny: the current key and the candidate,
    # four move types each, so eight entries. 256 leaves room for roughly 32 proposals
    # between clears while costing about 2.6 MB per kernel.
    max_cache_entries: int = 256
    _cache: dict = field(default_factory=dict, repr=False)
    cache_clears: int = 0

    def __post_init__(self) -> None:
        if self.move_probabilities is None:
            self.move_probabilities = {t: 1.0 / len(MoveType.ALL) for t in MoveType.ALL}
        total = sum(self.move_probabilities.values())
        if not math.isclose(total, 1.0, abs_tol=1e-12):
            raise ValueError(f"move probabilities must sum to 1, got {total}")

    # -- legality ---------------------------------------------------------------------
    def _width_ok(self, width: int) -> bool:
        return self.min_width <= width <= self.max_width

    def is_legal(self, key) -> bool:
        if not key or key[-1][0] != self.trace_length:
            return False
        start = 0
        previous_skill = None
        for end, skill in key:
            if end <= start or not self._width_ok(end - start):
                return False
            if not (0 <= skill < self.n_skills):
                return False
            if previous_skill is not None and skill == previous_skill:
                return False
            previous_skill, start = skill, end
        return True

    def _pairs(self, left_forbidden, right_forbidden):
        """Ordered `(ls, rs)` with `ls != rs`, `ls != left_forbidden`, `rs != right_forbidden`."""
        for ls in range(self.n_skills):
            if ls == left_forbidden:
                continue
            for rs in range(self.n_skills):
                if rs == ls or rs == right_forbidden:
                    continue
                yield ls, rs

    # -- neighbours -------------------------------------------------------------------
    def neighbours(self, key, move: str) -> tuple:
        cached = self._cache.get((move, key))
        if cached is not None:
            return cached
        spans = spans_of(key)
        n = len(spans)
        out = []

        if move == MoveType.RELABEL:
            for i, (_, _, skill) in enumerate(spans):
                left = spans[i - 1][2] if i > 0 else None
                right = spans[i + 1][2] if i + 1 < n else None
                for k in range(self.n_skills):
                    if k == skill or k == left or k == right:
                        continue
                    out.append(key[:i] + ((key[i][0], k),) + key[i + 1:])

        elif move == MoveType.SPLIT:
            for i, (start, end, _) in enumerate(spans):
                left_forbidden = spans[i - 1][2] if i > 0 else None
                right_forbidden = spans[i + 1][2] if i + 1 < n else None
                for cut in range(start + 1, end):
                    if not (self._width_ok(cut - start) and self._width_ok(end - cut)):
                        continue
                    for ls, rs in self._pairs(left_forbidden, right_forbidden):
                        out.append(key[:i] + ((cut, ls), (end, rs)) + key[i + 1:])

        elif move == MoveType.MERGE:
            for i in range(n - 1):
                start = spans[i][0]
                end = spans[i + 1][1]
                if not self._width_ok(end - start):
                    continue
                left_forbidden = spans[i - 1][2] if i > 0 else None
                right_forbidden = spans[i + 2][2] if i + 2 < n else None
                for k in range(self.n_skills):
                    if k == left_forbidden or k == right_forbidden:
                        continue
                    out.append(key[:i] + ((end, k),) + key[i + 2:])

        elif move == MoveType.SHIFT:
            for i in range(n - 1):
                start = spans[i][0]
                current = spans[i][1]
                end = spans[i + 1][1]
                left_forbidden = spans[i - 1][2] if i > 0 else None
                right_forbidden = spans[i + 2][2] if i + 2 < n else None
                for boundary in range(start + 1, end):
                    if boundary == current:
                        continue
                    if (self.max_shift is not None
                            and abs(boundary - current) > self.max_shift):
                        continue
                    if not (self._width_ok(boundary - start)
                            and self._width_ok(end - boundary)):
                        continue
                    for ls, rs in self._pairs(left_forbidden, right_forbidden):
                        out.append(key[:i] + ((boundary, ls), (end, rs)) + key[i + 2:])
        else:
            raise ValueError(f"unknown move type {move!r}")

        result = tuple(out)
        if len(self._cache) >= self.max_cache_entries:
            self._cache.clear()
            self.cache_clears += 1
        self._cache[(move, key)] = result
        return result

    # -- proposal law -----------------------------------------------------------------
    def proposal_distribution(self, key) -> dict:
        """`q(key -> .)`, including no-op mass, with multiplicities summed."""
        out: dict = {}
        for move, p_move in self.move_probabilities.items():
            if p_move == 0.0:
                continue
            reachable = self.neighbours(key, move)
            if not reachable:
                out[key] = out.get(key, 0.0) + p_move
                continue
            share = p_move / len(reachable)
            for candidate in reachable:
                out[candidate] = out.get(candidate, 0.0) + share
        return out

    def proposal_prob(self, source, target) -> float:
        """`q(source -> target)`, without materialising the whole distribution.

        `proposal_distribution` hashes every neighbour of every move type into a dict, and
        a Metropolis step needs `proposal_prob` twice — forward and reverse — so that dict
        is built and thrown away twice per proposal. Here the same number is accumulated
        directly, using `tuple.count` so the multiplicity (two different boundaries
        reaching the same state, which the Stage 5 docstring calls out) is still summed
        rather than counted once. `assert_proposal_prob_agrees` pins the two against each
        other, and the Stage 6E tests run it over random states.
        """
        total = 0.0
        for move, p_move in self.move_probabilities.items():
            if p_move == 0.0:
                continue
            reachable = self.neighbours(source, move)
            if not reachable:
                if target == source:
                    total += p_move
                continue
            multiplicity = reachable.count(target)
            if multiplicity:
                total += p_move * multiplicity / len(reachable)
        return total

    def sample_proposal(self, key, rng: np.random.Generator) -> tuple:
        moves = list(self.move_probabilities)
        weights = np.array([self.move_probabilities[m] for m in moves], dtype=float)
        move = moves[int(rng.choice(len(moves), p=weights))]
        reachable = self.neighbours(key, move)
        if not reachable:
            return key, move
        return reachable[int(rng.integers(len(reachable)))], move


# --------------------------------------------------------------------- equivalence
def assert_proposal_prob_agrees(kernel: FastSegmentationKernel, key,
                                tolerance: float = 0.0) -> dict:
    """The direct `proposal_prob` must equal the entry `proposal_distribution` holds.

    Checked over the whole support, including the no-op mass on `key` itself, so the
    shortcut cannot pass by agreeing only where the probability is large.
    """
    law = kernel.proposal_distribution(key)
    worst = 0.0
    for state, probability in law.items():
        worst = max(worst, abs(kernel.proposal_prob(key, state) - probability))
    # states outside the support must return exactly zero
    outside = 0.0
    for move in MoveType.ALL:
        for candidate in kernel.neighbours(key, move):
            for other in kernel.neighbours(candidate, move)[:3]:
                if other not in law:
                    outside = max(outside, kernel.proposal_prob(key, other))
    return {"n_states": len(law), "max_difference_on_support": worst,
            "max_probability_off_support": outside,
            "pass": bool(worst <= tolerance and outside <= tolerance)}


def assert_kernels_agree(fast: FastSegmentationKernel, reference, key,
                         tolerance: float = 1e-15) -> dict:
    """The fast law must equal the enumerated Stage 6E law, state by state.

    `reference` is a `Stage6EMoveKernel` over the same trace and settings. Both
    distributions are put on `Segmentation` keys and compared exactly — support and
    probability, not just support.
    """
    segmentation = segmentation_of(key)
    fast_law = {segmentation_of(k): p for k, p in fast.proposal_distribution(key).items()}
    slow_law = reference.proposal_distribution(segmentation)
    states = set(fast_law) | set(slow_law)
    worst = 0.0
    for state in states:
        worst = max(worst, abs(fast_law.get(state, 0.0) - slow_law.get(state, 0.0)))
    per_move = {}
    for move in MoveType.ALL:
        a = len(fast.neighbours(key, move))
        b = len(reference.neighbours(segmentation, move))
        per_move[move] = {"fast": a, "reference": b, "equal": a == b}
    return {
        "n_states_fast": len(fast_law), "n_states_reference": len(slow_law),
        "support_equal": set(fast_law) == set(slow_law),
        "max_probability_difference": worst,
        "fast_sums_to_one": abs(sum(fast_law.values()) - 1.0) < 1e-12,
        "reference_sums_to_one": abs(sum(slow_law.values()) - 1.0) < 1e-12,
        "neighbour_counts": per_move,
        "pass": bool(set(fast_law) == set(slow_law) and worst <= tolerance
                     and all(v["equal"] for v in per_move.values())),
    }
