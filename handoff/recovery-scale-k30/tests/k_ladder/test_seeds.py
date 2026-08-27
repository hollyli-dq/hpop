"""The seed namespace: distinct by construction, not by bands that happen not to overlap.

The registered scheme derived streams by adding literal offsets 50 apart, so replicate 50's
structural-truth stream is replicate 0's role-support stream. Nothing warns. These tests
pin that the replacement cannot collide and stays deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_cpa.seeds import ROOT_ENTROPY, STREAM, LadderSeeds   # noqa: E402

K_LADDER = (3, 5, 10, 20, 30)


def test_the_old_hand_offset_bands_really_do_collide():
    """The premise. Without this the replacement has no motivation on record."""
    seen, collisions = set(), 0
    for replicate in range(200):
        for base in (6_500_001, 6_500_051, 6_500_101):
            value = base + replicate
            collisions += value in seen
            seen.add(value)
    assert collisions == 300, collisions


def test_master_streams_never_collide_over_many_replicates():
    seeds = LadderSeeds()
    seen, collisions = set(), 0
    for replicate in range(200):
        for stream in ("master_structural_truth", "master_role_support",
                       "master_permutation"):
            value = seeds.master(stream, replicate)
            collisions += value in seen
            seen.add(value)
    assert collisions == 0


def test_every_rung_stream_is_distinct():
    seeds = LadderSeeds()
    values = set()
    for K in K_LADDER:
        for replicate in (0, 1):
            for stream in ("rung_pi_p", "rung_train_corpus", "rung_heldout_corpus",
                           "rung_dispersed_start", "rung_formal_chain",
                           "rung_scale_pilot"):
                values.add(seeds.rung(stream, K, replicate))
    assert len(values) == len(K_LADDER) * 2 * 6


def test_per_rung_bands_would_have_collided_beyond_K_100():
    """`6_520_000 + 100*K + r` overlaps the next band once K exceeds 100."""
    assert 6_520_000 + 100 * 101 + 0 == 6_530_000 + 100 * 1 + 0


def test_trace_streams_are_distinct_across_split_rung_and_index():
    seeds = LadderSeeds()
    values = set()
    for split in ("train", "heldout"):
        for K in (3, 30):
            for index in range(40):
                for component in range(3):
                    values.add(seeds.trace(split, K, 0, index, component))
    assert len(values) == 2 * 2 * 40 * 3


def test_seeds_are_deterministic_across_instances():
    a, b = LadderSeeds(), LadderSeeds()
    assert a.master("master_permutation", 0) == b.master("master_permutation", 0)
    assert a.rung("rung_pi_p", 30, 1) == b.rung("rung_pi_p", 30, 1)
    first = a.generator("diagnostic", 7).standard_normal(5)
    second = b.generator("diagnostic", 7).standard_normal(5)
    assert np.array_equal(first, second)


def test_a_different_root_gives_a_different_namespace():
    assert (LadderSeeds().master("master_structural_truth", 0)
            != LadderSeeds(ROOT_ENTROPY + 1).master("master_structural_truth", 0))


def test_an_unknown_stream_is_rejected():
    with pytest.raises(KeyError, match="unknown stream"):
        LadderSeeds().generator("not_a_stream", 0)


def test_the_scheme_records_what_it_supersedes():
    payload = LadderSeeds().as_dict()
    assert payload["root_entropy"] == ROOT_ENTROPY
    assert set(payload["streams"]) == set(STREAM)
    assert "collide" in payload["supersedes"]
