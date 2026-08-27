"""Local proposal kernel over segmentations: Split, Merge, Shift, Relabel.

This is the sampler that has to survive contact with real traces. The Stage-1
kernel ("enumerate every legal state, jump to a uniformly chosen other one") is
correct but unusable — it needs the global state list. These moves are *local*:
each one inspects a segment or an adjacent pair and nothing else.

The four moves, on a segmentation ``S = ((a_1,b_1,z_1), ..., (a_L,b_L,z_L))``:

    Relabel   change one segment's skill                    L -> L
    Split     cut one segment in two, label both parts      L -> L+1
    Merge     fuse an adjacent pair into one segment        L -> L-1
    Shift     move one internal boundary, relabel both      L -> L
              adjacent segments

Every proposed state must be support-compatible, so each move enumerates only the
*legal* variants of itself.

Why the proposal probability has to be computed, not assumed
-----------------------------------------------------------
None of these moves is symmetric. Split and Merge change the number of segments,
and the number of available moves differs between ``S`` and ``S'``, so the
Hastings ratio ``q(S'->S) / q(S->S')`` is not 1 and cannot be dropped. Getting
this wrong is the classic way an MCMC sampler silently targets the wrong
distribution — it still runs, still mixes, still looks healthy, and converges to
something that is not the posterior.

So the kernel exposes the proposal law explicitly:

    q(S -> S') = sum_t  p_t * 1[S' in N_t(S)] / |N_t(S)|

where ``N_t(S)`` is the set of states reachable from ``S`` by one move of type
``t`` and ``p_t`` is the probability of choosing that move type. Both
``|N_t(S)|`` and ``|N_t(S')|`` are *local* counts — computed by looking at the
segments of one state, never by enumerating the global space. If a chosen move
type has no legal moves the step is a no-op, which contributes only to
``q(S -> S)`` and so never enters an acceptance ratio.

The reverse of Split is Merge and vice versa; Shift and Relabel are their own
reverses. That pairing is what makes the kernel reversible, and it is tested
rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hpop.mcmc_original.toy import map_cpa_block_to_roles
from hpop.mcmc_original.types import Segment, Segmentation

__all__ = [
    "MoveType",
    "compatible_skills",
    "relabel_moves",
    "split_moves",
    "merge_moves",
    "shift_moves",
    "LocalMoveKernel",
    "mh_local_step",
    "run_local_mcmc",
    "transition_matrix",
]


class MoveType:
    RELABEL = "relabel"
    SPLIT = "split"
    MERGE = "merge"
    SHIFT = "shift"
    ALL = (RELABEL, SPLIT, MERGE, SHIFT)
    #: the move type that undoes each move type
    REVERSE = {RELABEL: RELABEL, SPLIT: MERGE, MERGE: SPLIT, SHIFT: SHIFT}


def compatible_skills(block, skills) -> tuple[int, ...]:
    """Skill ids whose CPA label set exactly matches this block's multiset."""
    return tuple(
        k
        for k, template in enumerate(skills)
        if map_cpa_block_to_roles(block, template.cpa_labels) is not None
    )


def _segmentation(segments) -> Segmentation:
    return Segmentation(segments=tuple(segments))


# ---------------------------------------------------------------------------
# the four move types
# ---------------------------------------------------------------------------


def relabel_moves(x, segmentation: Segmentation, skills) -> tuple[Segmentation, ...]:
    """All states differing from ``S`` by the skill of exactly one segment."""
    out = []
    segments = segmentation.segments
    for index, segment in enumerate(segments):
        for skill in compatible_skills(x[segment.start : segment.end], skills):
            if skill == segment.skill:
                continue
            replaced = list(segments)
            replaced[index] = Segment(segment.start, segment.end, skill)
            out.append(_segmentation(replaced))
    return tuple(out)


def split_moves(x, segmentation: Segmentation, skills) -> tuple[Segmentation, ...]:
    """All states obtained by cutting one segment into two compatible parts."""
    out = []
    segments = segmentation.segments
    for index, segment in enumerate(segments):
        for cut in range(segment.start + 1, segment.end):
            left = compatible_skills(x[segment.start : cut], skills)
            right = compatible_skills(x[cut : segment.end], skills)
            for left_skill in left:
                for right_skill in right:
                    out.append(
                        _segmentation(
                            list(segments[:index])
                            + [
                                Segment(segment.start, cut, left_skill),
                                Segment(cut, segment.end, right_skill),
                            ]
                            + list(segments[index + 1 :])
                        )
                    )
    return tuple(out)


def merge_moves(x, segmentation: Segmentation, skills) -> tuple[Segmentation, ...]:
    """All states obtained by fusing an adjacent pair into one compatible segment."""
    out = []
    segments = segmentation.segments
    for index in range(len(segments) - 1):
        left, right = segments[index], segments[index + 1]
        for skill in compatible_skills(x[left.start : right.end], skills):
            out.append(
                _segmentation(
                    list(segments[:index])
                    + [Segment(left.start, right.end, skill)]
                    + list(segments[index + 2 :])
                )
            )
    return tuple(out)


def shift_moves(
    x, segmentation: Segmentation, skills, max_shift: int | None = None
) -> tuple[Segmentation, ...]:
    """All states obtained by moving one internal boundary and relabelling both sides.

    The boundary between segments ``index`` and ``index+1`` may move anywhere
    strictly inside their combined span (excluding its current position). Both new
    blocks are relabelled with every compatible skill, because changing a block's
    length generally changes which skills accept it.

    Args:
        max_shift: optional cap on ``|new - old|``. ``None`` allows any position in
            the combined span, which is what keeps the kernel irreducible when
            skills have fixed role counts; a cap of 1 gives the classic
            move-by-one-position shift.
    """
    out = []
    segments = segmentation.segments
    for index in range(len(segments) - 1):
        left, right = segments[index], segments[index + 1]
        for boundary in range(left.start + 1, right.end):
            if boundary == left.end:
                continue
            if max_shift is not None and abs(boundary - left.end) > max_shift:
                continue
            left_skills = compatible_skills(x[left.start : boundary], skills)
            right_skills = compatible_skills(x[boundary : right.end], skills)
            for left_skill in left_skills:
                for right_skill in right_skills:
                    out.append(
                        _segmentation(
                            list(segments[:index])
                            + [
                                Segment(left.start, boundary, left_skill),
                                Segment(boundary, right.end, right_skill),
                            ]
                            + list(segments[index + 2 :])
                        )
                    )
    return tuple(out)


# ---------------------------------------------------------------------------
# the kernel
# ---------------------------------------------------------------------------


@dataclass
class LocalMoveKernel:
    """Mixture of the four local moves, with an explicit proposal law.

    Attributes:
        x: the observed CPA trace.
        skills: skill templates indexed by skill id.
        move_probabilities: ``p_t`` for each move type; defaults to uniform.
        max_shift: passed through to :func:`shift_moves`.
    """

    x: tuple[int, ...]
    skills: tuple
    move_probabilities: dict[str, float] | None = None
    max_shift: int | None = None

    def __post_init__(self) -> None:
        self.x = tuple(int(v) for v in self.x)
        if self.move_probabilities is None:
            self.move_probabilities = {t: 1.0 / len(MoveType.ALL) for t in MoveType.ALL}
        total = sum(self.move_probabilities.values())
        if not math.isclose(total, 1.0, abs_tol=1e-12):
            raise ValueError(f"move probabilities must sum to 1, got {total}")
        if any(p < 0 for p in self.move_probabilities.values()):
            raise ValueError("move probabilities must be non-negative")
        self._neighbour_cache: dict[tuple[str, Segmentation], tuple[Segmentation, ...]] = {}

    def neighbours(self, segmentation: Segmentation, move: str) -> tuple[Segmentation, ...]:
        """``N_t(S)`` — the legal states one move of type ``t`` can reach."""
        key = (move, segmentation)
        cached = self._neighbour_cache.get(key)
        if cached is not None:
            return cached
        if move == MoveType.RELABEL:
            out = relabel_moves(self.x, segmentation, self.skills)
        elif move == MoveType.SPLIT:
            out = split_moves(self.x, segmentation, self.skills)
        elif move == MoveType.MERGE:
            out = merge_moves(self.x, segmentation, self.skills)
        elif move == MoveType.SHIFT:
            out = shift_moves(self.x, segmentation, self.skills, self.max_shift)
        else:
            raise ValueError(f"unknown move type {move!r}")
        self._neighbour_cache[key] = out
        return out

    def proposal_distribution(self, segmentation: Segmentation) -> dict[Segmentation, float]:
        """The full law ``q(S -> .)``, including any no-op mass on ``S`` itself.

        A move type with no legal moves contributes its whole probability to the
        no-op. Note a state can be reached by more than one move of the same type
        (for instance two different boundaries that yield the same result); those
        multiplicities are summed, which is exactly what the Hastings ratio needs.
        """
        out: dict[Segmentation, float] = {}
        for move, p_move in self.move_probabilities.items():
            if p_move == 0.0:
                continue
            reachable = self.neighbours(segmentation, move)
            if not reachable:
                out[segmentation] = out.get(segmentation, 0.0) + p_move
                continue
            share = p_move / len(reachable)
            for candidate in reachable:
                out[candidate] = out.get(candidate, 0.0) + share
        return out

    def proposal_prob(self, source: Segmentation, target: Segmentation) -> float:
        """``q(source -> target)``, computed from local counts only."""
        return self.proposal_distribution(source).get(target, 0.0)

    def sample_proposal(
        self, segmentation: Segmentation, rng: np.random.Generator
    ) -> tuple[Segmentation, str]:
        """Draw one proposal: pick a move type, then a uniform move of that type.

        Deliberately does *not* go through :meth:`proposal_distribution` — the two
        are separate code paths so that a test can check they agree.
        """
        moves = list(self.move_probabilities)
        weights = np.array([self.move_probabilities[m] for m in moves], dtype=float)
        move = moves[int(rng.choice(len(moves), p=weights))]
        reachable = self.neighbours(segmentation, move)
        if not reachable:
            return segmentation, move
        return reachable[int(rng.integers(len(reachable)))], move


def mh_local_step(
    segmentation: Segmentation,
    log_target,
    kernel: LocalMoveKernel,
    rng: np.random.Generator,
    current_log_target: float | None = None,
) -> dict:
    """One Metropolis-Hastings step with the full Hastings correction.

        log alpha = [log pi(S') - log pi(S)] + [log q(S'->S) - log q(S->S')]

    The second bracket is not optional: Split and Merge change the number of
    available moves, so dropping it would target the wrong distribution.
    """
    current_value = (
        log_target(segmentation) if current_log_target is None else current_log_target
    )
    candidate, move = kernel.sample_proposal(segmentation, rng)
    if candidate == segmentation:
        return {"state": segmentation, "log_target": current_value, "move": move,
                "proposed": False, "accepted": False}

    candidate_value = log_target(candidate)
    if candidate_value == -math.inf:
        return {"state": segmentation, "log_target": current_value, "move": move,
                "proposed": True, "accepted": False}

    forward = kernel.proposal_prob(segmentation, candidate)
    reverse = kernel.proposal_prob(candidate, segmentation)
    if reverse <= 0.0:
        # The move cannot be undone, so accepting it would break reversibility.
        return {"state": segmentation, "log_target": current_value, "move": move,
                "proposed": True, "accepted": False}

    log_alpha = (candidate_value - current_value) + (math.log(reverse) - math.log(forward))
    if log_alpha >= 0.0 or math.log(rng.random()) < log_alpha:
        return {"state": candidate, "log_target": candidate_value, "move": move,
                "proposed": True, "accepted": True}
    return {"state": segmentation, "log_target": current_value, "move": move,
            "proposed": True, "accepted": False}


def run_local_mcmc(
    initial: Segmentation,
    log_target,
    kernel: LocalMoveKernel,
    n_iterations: int,
    burn_in: int,
    rng: np.random.Generator,
) -> dict:
    """Run the local-move sampler and record per-move-type acceptance statistics."""
    if n_iterations <= burn_in:
        raise ValueError("n_iterations must exceed burn_in")

    state = initial
    value = log_target(state)
    trace: list[Segmentation] = []
    proposed = {m: 0 for m in MoveType.ALL}
    accepted = {m: 0 for m in MoveType.ALL}

    for _ in range(n_iterations):
        step = mh_local_step(state, log_target, kernel, rng, current_log_target=value)
        state, value = step["state"], step["log_target"]
        proposed[step["move"]] += int(step["proposed"])
        accepted[step["move"]] += int(step["accepted"])
        trace.append(state)

    kept = trace[burn_in:]
    return {
        "trace": trace,
        "kept": kept,
        "n_iterations": n_iterations,
        "burn_in": burn_in,
        "proposed_by_move": proposed,
        "accepted_by_move": accepted,
        "acceptance_by_move": {
            m: (accepted[m] / proposed[m] if proposed[m] else float("nan"))
            for m in MoveType.ALL
        },
        "acceptance_rate": (
            sum(accepted.values()) / sum(proposed.values())
            if sum(proposed.values())
            else float("nan")
        ),
    }


def transition_matrix(states, log_targets, kernel: LocalMoveKernel) -> np.ndarray:
    """The exact MH transition matrix ``K`` over an enumerated state space.

    ``K[i, j] = q(i->j) * min(1, pi_j q(j->i) / (pi_i q(i->j)))`` off-diagonal, with
    the diagonal absorbing everything left over (rejections and no-ops). Building
    this explicitly is what lets detailed balance and stationarity be *checked*
    rather than argued.
    """
    states = list(states)
    log_targets = np.asarray(log_targets, dtype=float)
    index = {state: i for i, state in enumerate(states)}
    n = len(states)
    matrix = np.zeros((n, n), dtype=float)

    for i, state in enumerate(states):
        for candidate, forward in kernel.proposal_distribution(state).items():
            if candidate == state or forward <= 0.0:
                continue
            j = index.get(candidate)
            if j is None:
                continue  # proposal outside the enumerated space; always rejected
            reverse = kernel.proposal_prob(candidate, state)
            if reverse <= 0.0:
                continue
            log_alpha = (log_targets[j] - log_targets[i]) + (
                math.log(reverse) - math.log(forward)
            )
            matrix[i, j] = forward * min(1.0, math.exp(min(0.0, log_alpha)))
        matrix[i, i] = 1.0 - matrix[i].sum()
    return matrix
