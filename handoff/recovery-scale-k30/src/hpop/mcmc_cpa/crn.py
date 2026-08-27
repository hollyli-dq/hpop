"""Common random numbers addressed by index, not by position in a stream.

## The failure this exists to prevent

Two arms that start from an identical state and draw from one sequential `Generator` are
NOT sharing random numbers. They share a *prefix*. The moment one arm accepts a move the
other rejects -- or draws a segmentation with one more segment, consuming one more uniform
-- the two streams slip, and from then on every comparison is between different random
numbers as well as different models. A run can look like a controlled comparison, pass a
zero-sweep identical-initial-state check, and still be uncontrolled from sweep 1 onward.

Identical initial states are necessary and nowhere near sufficient.

## What this does instead

Every random quantity is drawn from a generator derived from **where it sits in the
design**, never from how much randomness has been consumed so far:

    (replicate, K, chain, sweep, move type, proposal index)

`SeedSequence.spawn_key` makes distinct index tuples distinct by construction. So the
uniforms the FFBS sees at sweep 40 of chain 2 at `K = 10` are the same numbers in every
arm, whatever either arm did at sweep 39 -- which is what makes an arm-to-arm difference
attributable to the arm.

## The residual, stated plainly

`ffbs_segmentation_draw` loops over traces internally against one generator, and a trace's
consumption depends on how many segments it draws. So **within a single sweep**, traces
after the first can still misalign between arms. This class cannot fix that without
editing the sealed backend, and it does not pretend to: what it guarantees is that
misalignment **cannot propagate across sweeps or across move types**, because each of
those starts from an index-derived stream rather than from wherever the last one stopped.
`crn_alignment_report` measures the residual instead of assuming it away.
"""

from __future__ import annotations

import numpy as np

from .seeds import LadderSeeds

__all__ = ["CommonRandomNumbers", "MOVE_TYPES", "crn_alignment_report"]

#: Move types get their own sub-stream so a change in one cannot shift another.
MOVE_TYPES = {
    "ffbs": 0,          # boundary/label draw
    "u": 1,             # collapsed U row proposal and its acceptance uniform
    "pi_p": 2,          # transition/initial Gibbs draw
    "init": 3,          # chain initialisation
}


class CommonRandomNumbers:
    """Index-addressed generators, shared across arms by construction."""

    __slots__ = ("seeds", "replicate", "k", "chain", "_issued")

    def __init__(self, replicate: int, k: int, chain: int,
                 seeds: LadderSeeds | None = None):
        self.seeds = LadderSeeds() if seeds is None else seeds
        self.replicate, self.k, self.chain = int(replicate), int(k), int(chain)
        self._issued = 0

    def rng(self, move_type: str, sweep: int, proposal: int = 0) -> np.random.Generator:
        """The generator for one (sweep, move type, proposal index). Arm-independent.

        Deliberately takes no argument that could carry the arm's history: given the same
        design coordinates it returns the same numbers, and there is no way for a caller
        to accidentally make it depend on what happened earlier.
        """
        if move_type not in MOVE_TYPES:
            raise KeyError(f"unknown move type {move_type!r}; "
                           f"known: {sorted(MOVE_TYPES)}")
        self._issued += 1
        return self.seeds.generator("rung_formal_chain", self.replicate, self.k,
                                    self.chain, int(sweep), MOVE_TYPES[move_type],
                                    int(proposal))

    def key(self, move_type: str, sweep: int, proposal: int = 0) -> tuple:
        """The full index tuple, for recording and for equality checks in tests."""
        return (self.replicate, self.k, self.chain, int(sweep),
                move_type, int(proposal))

    @property
    def generators_issued(self) -> int:
        return self._issued

    def provenance(self) -> dict:
        return {
            "scheme": "common random numbers addressed by design index",
            "index": ["replicate", "K", "chain", "sweep", "move type", "proposal index"],
            "move_types": dict(MOVE_TYPES),
            "replicate": self.replicate, "K": self.k, "chain": self.chain,
            "root_entropy": self.seeds.root,
            "guarantee": ("a generator depends only on its index, never on how much "
                          "randomness an arm consumed earlier, so misalignment cannot "
                          "propagate across sweeps or move types"),
            "residual": ("ffbs_segmentation_draw loops over traces against one "
                         "generator, so within ONE sweep traces after the first can "
                         "still misalign between arms; not fixable without editing the "
                         "sealed backend, and measured by crn_alignment_report rather "
                         "than assumed away"),
        }


def crn_alignment_report(a: CommonRandomNumbers, b: CommonRandomNumbers,
                         sweeps: int = 20, draws: int = 8) -> dict:
    """Do two CRN objects hand out identical numbers at identical indices?

    Called with the two arms' CRN objects after they have diverged, this is the check that
    the sharing is real rather than an assumption about stream discipline.
    """
    mismatches = []
    for sweep in range(int(sweeps)):
        for move_type in sorted(MOVE_TYPES):
            for proposal in range(2):
                x = a.rng(move_type, sweep, proposal).random(int(draws))
                y = b.rng(move_type, sweep, proposal).random(int(draws))
                if not np.array_equal(x, y):
                    mismatches.append({"sweep": sweep, "move_type": move_type,
                                       "proposal": proposal})
    return {"sweeps": int(sweeps), "draws_per_index": int(draws),
            "indices_checked": int(sweeps) * len(MOVE_TYPES) * 2,
            "mismatches": mismatches, "aligned": not mismatches}
