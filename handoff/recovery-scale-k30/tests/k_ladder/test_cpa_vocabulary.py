"""The CPA-vocabulary layer: A != m, with per-skill role supports.

The property that matters most is the first one. With the identity role map and A = m the
new builder must reduce **bitwise** to the sealed `BlockScoreTable`, because in that case
it is supposed to be the registered model and nothing else. Anything short of bit equality
would mean the translation layer had changed the arithmetic rather than only what feeds it.

Everything else follows: a block containing a CPA outside a skill's support is impossible
for that skill and must be -inf, and an in-support block must equal what the sealed
per-block scorer gives on the relabelled trace.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_cpa import (CPABlockScoreTable, RoleMaps,          # noqa: E402
                           assert_matches_sealed_scorer, sample_role_maps)
from hpop.mcmc_cpa.role_maps import NOT_IN_SUPPORT                # noqa: E402
from hpop.mcmc_original.stage6e_block_table import BlockScoreTable  # noqa: E402

EPS, BETA, OMEGA, LREP, LBACK = 0.02, 1.0, 0.0, 1.0, 1.0
LADDER = (3, 5, 10, 20, 30)


def corpus_from_supports(maps, n_traces, length, seed):
    """Traces emitted from the skills' own supports, as the generator will produce."""
    supports = maps.supports()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_traces):
        trace = []
        while len(trace) < length:
            k = int(rng.integers(maps.n_skills))
            width = int(rng.integers(4, 13))
            trace.extend(int(v) for v in rng.choice(supports[k], size=width))
        out.append(tuple(trace[:length]))
    return tuple(out)


# ------------------------------------------------------- reduction to the sealed model
def test_identity_map_reduces_bitwise_to_the_sealed_builder():
    """The load-bearing test. Identity map, A = m -> must be the registered model exactly."""
    rng = np.random.default_rng(4242)
    K, m = 3, 5
    u = rng.standard_normal((K, m, 2))
    traces = tuple(tuple(int(v) for v in rng.integers(0, m, size=40)) for _ in range(4))

    sealed = BlockScoreTable(traces=traces, epsilon=EPS, n_skills=K,
                             min_width=3, max_width=12)
    sealed.refresh(u, BETA, OMEGA, LREP, LBACK)

    ours = CPABlockScoreTable(traces=traces, epsilon=EPS,
                              role_maps=RoleMaps.identity(K, m),
                              min_width=3, max_width=12)
    info = ours.refresh(u, BETA, OMEGA, LREP, LBACK)

    assert np.array_equal(sealed._table, ours._table), "not bitwise identical"
    assert info["live_fraction"] == 1.0, "with A = m every block is in every support"


@pytest.mark.parametrize("K", (3, 5, 10))
def test_identity_reduction_holds_at_several_K(K):
    rng = np.random.default_rng(100 + K)
    m = 5
    u = rng.standard_normal((K, m, 2))
    traces = tuple(tuple(int(v) for v in rng.integers(0, m, size=30)) for _ in range(3))
    sealed = BlockScoreTable(traces=traces, epsilon=EPS, n_skills=K,
                             min_width=3, max_width=12)
    sealed.refresh(u, BETA, OMEGA, LREP, LBACK)
    ours = CPABlockScoreTable(traces=traces, epsilon=EPS,
                              role_maps=RoleMaps.identity(K, m),
                              min_width=3, max_width=12)
    ours.refresh(u, BETA, OMEGA, LREP, LBACK)
    assert np.array_equal(sealed._table, ours._table)


# --------------------------------------------------------------- the support mask
def test_a_block_outside_a_skills_support_is_impossible_for_that_skill():
    maps = RoleMaps(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), n_cpa=10)
    trace = (0, 1, 2, 3, 4, 0, 1, 2)          # entirely inside skill 0's support
    u = np.random.default_rng(1).standard_normal((2, 5, 2))
    table = CPABlockScoreTable(traces=(trace,), epsilon=EPS, role_maps=maps,
                               min_width=3, max_width=8)
    table.refresh(u, BETA, OMEGA, LREP, LBACK)
    assert np.isfinite(table.score(0, 0, 5, 0)), "skill 0 owns every CPA in this block"
    assert table.score(0, 0, 5, 1) == -np.inf, "skill 1 owns none of them"


def test_one_out_of_support_symbol_kills_the_whole_block():
    maps = RoleMaps(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), n_cpa=10)
    trace = (0, 1, 2, 9, 4, 0)                 # a single symbol 9 belongs to skill 1
    u = np.random.default_rng(2).standard_normal((2, 5, 2))
    table = CPABlockScoreTable(traces=(trace,), epsilon=EPS, role_maps=maps,
                               min_width=3, max_width=6)
    table.refresh(u, BETA, OMEGA, LREP, LBACK)
    assert np.isfinite(table.score(0, 0, 3, 0)), "(0,1,2) is inside skill 0"
    assert table.score(0, 0, 4, 0) == -np.inf, "extending over the 9 must kill it"
    assert table.score(0, 3, 6, 0) == -np.inf


# ------------------------------------------------------------ parity with the oracle
@pytest.mark.parametrize("K,m,A", [(3, 5, 20), (5, 10, 50), (10, 10, 50)])
def test_in_support_scores_match_the_sealed_per_block_scorer(K, m, A):
    maps = sample_role_maps(K, m, A, seed=7 * K + m)
    u = np.random.default_rng(K).standard_normal((K, m, 2))
    traces = corpus_from_supports(maps, 3, 48, seed=K)
    table = CPABlockScoreTable(traces=traces, epsilon=EPS, role_maps=maps,
                               min_width=3, max_width=12)
    info = table.refresh(u, BETA, OMEGA, LREP, LBACK)
    assert info["live_block_skill_pairs"] > 0, "the corpus produced no scorable block"

    report = assert_matches_sealed_scorer(table, u, BETA, OMEGA, LREP, LBACK, limit=300)
    assert report["pass"], report
    assert report["max_absolute_difference"] < 1e-12
    assert report["in_support_blocks_checked"] > 0


# ------------------------------------------------------------------ the role maps
@pytest.mark.parametrize("K", LADDER)
def test_sampled_supports_are_injective_and_pairwise_distinct(K):
    maps = sample_role_maps(K, 10, 50, seed=6_500_051 + K)
    assert maps.forward.shape == (K, 10)
    for k in range(K):
        assert len(set(maps.forward[k].tolist())) == 10
    keys = {tuple(sorted(row)) for row in maps.forward.tolist()}
    assert len(keys) == K, "two skills share a support"
    # inverse is a true inverse
    for k in range(K):
        for r, c in enumerate(maps.forward[k]):
            assert maps.inverse[k, c] == r
    outside = set(range(50)) - set(maps.forward[k].tolist())
    assert all(maps.inverse[k, c] == NOT_IN_SUPPORT for c in outside)


def test_role_maps_reject_a_non_injective_map():
    with pytest.raises(ValueError, match="not injective"):
        RoleMaps(np.array([[0, 1, 1, 2, 3]]), n_cpa=10)
    with pytest.raises(ValueError, match="outside the CPA vocabulary"):
        RoleMaps(np.array([[0, 1, 2, 3, 99]]), n_cpa=10)
    with pytest.raises(ValueError, match="more roles"):
        RoleMaps(np.array([[0, 1, 2]]), n_cpa=2)


def test_u_shaped_over_the_vocabulary_instead_of_the_roles_is_rejected():
    """A likely misuse: passing U as (K, A, d) rather than (K, m, d)."""
    maps = sample_role_maps(3, 10, 50, seed=5)
    traces = corpus_from_supports(maps, 2, 24, seed=5)
    table = CPABlockScoreTable(traces=traces, epsilon=EPS, role_maps=maps,
                               min_width=3, max_width=12)
    with pytest.raises(ValueError, match="per-skill over its OWN roles"):
        table.refresh(np.zeros((3, 50, 2)), BETA, OMEGA, LREP, LBACK)


def test_a_cpa_outside_the_vocabulary_is_rejected_at_construction():
    maps = sample_role_maps(2, 5, 10, seed=3)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        CPABlockScoreTable(traces=((0, 1, 99),), epsilon=EPS, role_maps=maps,
                           min_width=3, max_width=6)


# ------------------------------------------------------------ what the mask buys
def test_sparse_supports_shrink_the_candidate_set():
    """The mechanism that makes thirty skills identifiable, stated as a measurement."""
    K, m, A = 10, 10, 50
    maps = sample_role_maps(K, m, A, seed=99)
    u = np.random.default_rng(9).standard_normal((K, m, 2))
    traces = corpus_from_supports(maps, 4, 60, seed=9)
    table = CPABlockScoreTable(traces=traces, epsilon=EPS, role_maps=maps,
                               min_width=3, max_width=12)
    info = table.refresh(u, BETA, OMEGA, LREP, LBACK)
    assert 0.0 < info["live_fraction"] < 0.5, (
        f"expected most candidates to die, got {info['live_fraction']:.3f}")


def test_dense_tables_carry_the_scores_and_keep_the_frozen_layout():
    maps = sample_role_maps(3, 10, 50, seed=21)
    u = np.random.default_rng(21).standard_normal((3, 10, 2))
    traces = corpus_from_supports(maps, 2, 36, seed=21)
    table = CPABlockScoreTable(traces=traces, epsilon=EPS, role_maps=maps,
                               min_width=3, max_width=12)
    table.refresh(u, BETA, OMEGA, LREP, LBACK)
    for n, trace in enumerate(traces):
        dense = table.tables[n]
        assert dense.shape == (len(trace), len(trace) + 1, 3)
        for (tn, a, w), row in table._row.items():
            if tn != n:
                continue
            for k in range(3):
                assert dense[a, a + w, k] == table._table[k, row]
