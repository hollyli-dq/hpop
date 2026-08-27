"""The nested master library: one draw per replicate, the ladder cut from its prefixes.

Nesting is what stops library *size* being confounded with library *difficulty*. If each
rung were an independent draw, a poor K=30 result could be thirty skills being hard, or
those particular thirty being hard, and nothing in the data would separate the two. With
nesting the K=3 rung is literally three of the K=30 rung.

So the tests below are mostly about that one property holding across every object that
carries structure -- utilities, role maps, closures and the canonical library digest -- and
not merely for the first of them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_cpa.nested_library import (K_LADDER, MasterLibrary,   # noqa: E402
                                          draw_master_library)

REPLICATES = (0, 1)


@pytest.fixture(scope="module")
def libraries():
    return {r: draw_master_library(r)[0] for r in REPLICATES}


@pytest.mark.parametrize("replicate", REPLICATES)
def test_a_master_library_is_admissible_and_recorded(replicate, libraries):
    library = libraries[replicate]
    assert library.k_max == 30 and library.n_roles == 10 and library.n_cpa == 50
    payload = library.as_dict()
    assert payload["replicate"] == replicate
    # every skill's closure is a non-trivial strict partial order
    pairs = 10 * 9 // 2
    for relations in payload["relations_per_skill"]:
        assert 1 <= relations < pairs


@pytest.mark.parametrize("replicate", REPLICATES)
@pytest.mark.parametrize("K", K_LADDER)
def test_every_rung_is_a_prefix_of_the_full_library(K, replicate, libraries):
    library = libraries[replicate]
    u_k, maps_k = library.prefix(K)
    u_max, maps_max = library.prefix(30)
    assert np.array_equal(u_k, u_max[:K]), "latent utilities are not nested"
    assert np.array_equal(maps_k.forward, maps_max.forward[:K]), "role maps are not nested"
    assert np.array_equal(library.closure_bits(K), library.closure_bits(30)[:K]), \
        "closures are not nested"


@pytest.mark.parametrize("replicate", REPLICATES)
def test_the_rungs_are_strictly_increasing_sets_of_the_same_skills(replicate, libraries):
    library = libraries[replicate]
    previous = None
    for K in K_LADDER:
        _, maps = library.prefix(K)
        supports = {tuple(sorted(row)) for row in maps.forward.tolist()}
        assert len(supports) == K, "a rung contains two skills with the same support"
        if previous is not None:
            assert previous < supports, "a smaller rung is not a subset of the larger"
        previous = supports


@pytest.mark.parametrize("replicate", REPLICATES)
def test_the_library_digest_distinguishes_the_rungs(replicate, libraries):
    digests = {K: libraries[replicate].library_digest(K) for K in K_LADDER}
    assert len(set(digests.values())) == len(K_LADDER), \
        "two different rungs hash to the same library"


def test_the_two_replicates_are_genuinely_different_libraries(libraries):
    for K in K_LADDER:
        a = libraries[0].library_digest(K)
        b = libraries[1].library_digest(K)
        assert a != b, f"the two replicates coincide at K={K}"


@pytest.mark.parametrize("replicate", REPLICATES)
def test_the_draw_is_deterministic(replicate):
    first, _ = draw_master_library(replicate)
    second, _ = draw_master_library(replicate)
    assert np.array_equal(first.u, second.u)
    assert np.array_equal(first.role_maps.forward, second.role_maps.forward)
    assert first.library_digest(30) == second.library_digest(30)


@pytest.mark.parametrize("replicate", REPLICATES)
def test_the_attempt_record_names_every_seed_tried(replicate):
    _, attempts = draw_master_library(replicate)
    assert attempts, "no attempt was recorded"
    assert sum(a["accepted"] for a in attempts) == 1
    assert attempts[-1]["accepted"], "the accepted draw is not the last attempt"
    assert all("seed" in a for a in attempts)


def test_the_digest_uses_the_support_and_not_only_the_closure():
    """Two skills can share a closure shape while acting on different CPAs."""
    from hpop.mcmc_cpa.role_maps import RoleMaps
    u = np.random.default_rng(3).standard_normal((2, 10, 2))
    shared = np.stack([np.arange(10), np.arange(10, 20)])
    one = MasterLibrary(u, RoleMaps(shared, 50), np.arange(2), 50, 0, {})
    moved = MasterLibrary(u, RoleMaps(np.stack([np.arange(10), np.arange(20, 30)]), 50),
                          np.arange(2), 50, 0, {})
    assert one.library_digest(2) != moved.library_digest(2), (
        "the digest ignored the CPA support")


def test_prefix_rejects_a_K_outside_the_library():
    library, _ = draw_master_library(0)
    with pytest.raises(ValueError, match=r"K must be in"):
        library.prefix(31)
    with pytest.raises(ValueError, match=r"K must be in"):
        library.prefix(0)
