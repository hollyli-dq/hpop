"""A hierarchical seed namespace that cannot collide, replacing hand-offset integer bands.

The registered scheme derives streams by adding literal offsets to a base:

    master structural truth      6_500_001 + r
    master role-support library  6_500_051 + r
    master skill permutation     6_500_101 + r

Those bands are 50 apart, so at `r >= 50` the structural-truth stream of replicate 50 is
the role-support stream of replicate 0. Nothing warns; two conceptually independent draws
silently share entropy. The same pattern recurs in the per-rung bands, spaced `100 * K`,
which collide once `K` exceeds 100.

The scheme below derives every stream from one root through a **tuple key** using
`SeedSequence.spawn_key`, so distinct roles, replicates, rungs, traces and segments are
distinct by construction rather than by arithmetic that happens not to overlap yet. The
registered integers are kept as the root and as recorded provenance, so the intent of the
original scheme survives while its fragility does not.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LadderSeeds", "ROOT_ENTROPY", "STREAM"]

# The registered root. Every stream below descends from it; the band offsets are no longer
# used to separate streams, only recorded.
ROOT_ENTROPY = 6_500_000

STREAM = {
    "master_structural_truth": 0,
    "master_role_support": 1,
    "master_permutation": 2,
    "rung_pi_p": 3,
    "rung_train_corpus": 4,
    "rung_heldout_corpus": 5,
    "rung_dispersed_start": 6,
    "rung_formal_chain": 7,
    "rung_scale_pilot": 8,
    "diagnostic": 9,
}


class LadderSeeds:
    """Every stream the ladder needs, addressed by name rather than by offset."""

    __slots__ = ("root",)

    def __init__(self, root: int = ROOT_ENTROPY):
        self.root = int(root)

    def _sequence(self, *key: int) -> np.random.SeedSequence:
        return np.random.SeedSequence(entropy=self.root, spawn_key=tuple(int(k)
                                                                        for k in key))

    def generator(self, stream: str, *key: int) -> np.random.Generator:
        if stream not in STREAM:
            raise KeyError(f"unknown stream {stream!r}; known: {sorted(STREAM)}")
        return np.random.default_rng(self._sequence(STREAM[stream], *key))

    def integer(self, stream: str, *key: int) -> int:
        """A 63-bit integer, for the sealed APIs that still take a plain seed."""
        return int(self._sequence(STREAM[stream], *key).generate_state(2,
                                                                       dtype=np.uint64)[0]
                   >> 1)

    # -- named accessors, so a caller never assembles a key by hand -------------------
    def master(self, stream: str, replicate: int) -> int:
        return self.integer(stream, int(replicate))

    def rung(self, stream: str, k: int, replicate: int, *extra: int) -> int:
        return self.integer(stream, int(k), int(replicate), *extra)

    def trace(self, split: str, k: int, replicate: int, index: int,
              component: int = 0) -> int:
        stream = ("rung_train_corpus" if split == "train" else "rung_heldout_corpus")
        return self.integer(stream, int(k), int(replicate), int(index), int(component))

    def as_dict(self) -> dict:
        return {"root_entropy": self.root, "streams": dict(STREAM),
                "scheme": "numpy SeedSequence spawn_key over (stream, ...key); distinct "
                          "by construction, not by non-overlapping integer bands",
                "supersedes": "hand-offset bands 6_500_001+r / 6_500_051+r / "
                              "6_500_101+r, which collide at replicate >= 50, and the "
                              "per-rung 100*K bands, which collide for K > 100"}
