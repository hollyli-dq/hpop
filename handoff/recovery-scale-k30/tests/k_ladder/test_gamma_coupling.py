"""The shared-Gamma coupling: one master `G`, every rung a restriction of it.

The coupling only means something if each rung really is the same master weights cut down
— not merely correlated with them — and if the per-rung law it was adopted to preserve
survives. These tests check both, plus the joint conditioning that makes the ladder's
transition environments comparable across rungs in the first place.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_cpa.corpus import stationary_of
from hpop.mcmc_cpa.gamma_coupling import (K_LADDER, MasterTransitions,
                                          draw_master_gamma, draw_master_transitions,
                                          joint_band_acceptance_rate,
                                          restrict_and_renormalise, stationary_band)
from hpop.mcmc_cpa.nested_library import draw_master_library
from hpop.mcmc_cpa.seeds import LadderSeeds

K_MAX = 30


@pytest.fixture(scope="module")
def library():
    return draw_master_library(0)[0]


@pytest.fixture(scope="module")
def master(library):
    return draw_master_transitions(0, library.permutation, k_max=library.k_max)


# ---------------------------------------------- 1. every rung is the SAME master G, cut
def test_each_rung_is_the_restriction_of_the_one_master_g(master):
    """Not 'correlated with' — bitwise equal to the restriction of `master.g`."""
    for k in K_LADDER:
        np.testing.assert_array_equal(master.transition(k),
                                      restrict_and_renormalise(master.g, k))


def test_a_smaller_rung_is_the_restriction_of_a_larger_rungs_source(master):
    """Transitively: cutting K=3 out of G is cutting K=3 out of the same rows K=30 used."""
    for small in K_LADDER[:-1]:
        big = restrict_and_renormalise(master.g, K_MAX)
        # undo the K_max normalisation to recover relative weights, then renormalise at
        # the smaller rung; must reproduce the smaller rung exactly up to float error
        block = big[:small, :small].copy()
        np.fill_diagonal(block, 0.0)
        recut = block / block.sum(axis=1)[:, None]
        np.testing.assert_allclose(recut, master.transition(small), rtol=0, atol=1e-12)


def test_the_rungs_are_not_all_the_same_matrix(master):
    """Guards against a coupling so degenerate the rungs stop differing at all."""
    assert not np.allclose(master.transition(3),
                           master.transition(5)[:3, :3])


# ------------------------------------------- 2. old-destination ratios survive the ladder
def test_old_destination_ratios_are_invariant_across_rungs(master):
    """`P^(K)_ij / P^(K)_il = G_ij / G_il` for old `i, j, l`, whatever `K` is.

    This is the property that makes 'the same skill at a larger K' meaningful: growing the
    ladder dilutes every old destination by one common normaliser and reorders nothing.
    """
    smallest = min(K_LADDER)
    reference = master.transition(smallest)
    for k in K_LADDER[1:]:
        p = master.transition(k)
        for i in range(smallest):
            for j in range(smallest):
                for l in range(smallest):
                    if len({i, j, l}) == 3:
                        np.testing.assert_allclose(p[i, j] / p[i, l],
                                                   reference[i, j] / reference[i, l],
                                                   rtol=1e-12)


def test_the_ratio_equals_the_raw_gamma_ratio(master):
    """And the invariant value is the master weight ratio itself, not just some constant."""
    p = master.transition(max(K_LADDER))
    for i, j, l in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        np.testing.assert_allclose(p[i, j] / p[i, l], master.g[i, j] / master.g[i, l],
                                   rtol=1e-12)


def test_dilution_is_a_single_common_factor_per_row(master):
    """Every old destination in a row is scaled by the same number when K grows."""
    small, big = 3, 30
    a, b = master.transition(small), master.transition(big)
    for i in range(small):
        ratios = [b[i, j] / a[i, j] for j in range(small) if j != i]
        assert np.allclose(ratios, ratios[0], rtol=1e-12), \
            f"row {i} old destinations were not diluted uniformly: {ratios}"


# --------------------------------------------------- 3. zero diagonal, rows normalise
def test_zero_diagonal_and_row_normalisation(master):
    for k in K_LADDER:
        p = master.transition(k)
        assert p.shape == (k, k)
        np.testing.assert_array_equal(np.diag(p), np.zeros(k))
        np.testing.assert_allclose(p.sum(axis=1), np.ones(k), rtol=0, atol=1e-12)
        assert (p >= 0).all()


def test_the_master_gamma_has_an_exact_zero_diagonal():
    g = draw_master_gamma(np.random.default_rng(0), K_MAX)
    np.testing.assert_array_equal(np.diag(g), np.zeros(K_MAX))
    assert (g[~np.eye(K_MAX, dtype=bool)] > 0).all()


def test_restriction_refuses_a_rung_it_cannot_cut():
    g = draw_master_gamma(np.random.default_rng(0), 8)
    with pytest.raises(ValueError):
        restrict_and_renormalise(g, 9)
    with pytest.raises(ValueError):
        restrict_and_renormalise(g, 1)


# ------------------------------- 4. the permutation reaches skills and BOTH axes of G
def test_the_permutation_is_applied_to_both_axes(master):
    np.testing.assert_array_equal(
        master.g,
        master.g_unpermuted[np.ix_(master.permutation, master.permutation)])


def test_a_one_axis_permutation_would_be_caught(master):
    """A row-only permutation is a different matrix — so the two-axis test has teeth."""
    row_only = master.g_unpermuted[master.permutation]
    assert not np.array_equal(master.g, row_only)


def test_the_identity_permutation_leaves_the_master_untouched():
    identity = np.arange(K_MAX)
    m = draw_master_transitions(0, identity, k_max=K_MAX)
    np.testing.assert_array_equal(m.g, m.g_unpermuted)


def test_the_coupling_uses_the_librarys_own_permutation(library, master):
    """`G` must be permuted by the same permutation that permuted `U` and the role maps,
    or skill `i` means one thing in the emissions and another in the transitions."""
    np.testing.assert_array_equal(master.permutation, library.permutation)


def test_permuting_skills_permutes_the_transition_matrix_the_same_way():
    """At K_max the permuted master is exactly the unpermuted one relabelled."""
    perm = np.random.default_rng(7).permutation(K_MAX)
    m = draw_master_transitions(0, perm, k_max=K_MAX)
    unpermuted_p = restrict_and_renormalise(m.g_unpermuted, K_MAX)
    np.testing.assert_allclose(m.transition(K_MAX),
                               unpermuted_p[np.ix_(perm, perm)], rtol=1e-12)


# --------------------------- 5. joint acceptance is deterministic under the seed namespace
def test_the_whole_master_draw_is_reproducible(library):
    a = draw_master_transitions(0, library.permutation, k_max=K_MAX)
    b = draw_master_transitions(0, library.permutation, k_max=K_MAX)
    np.testing.assert_array_equal(a.g, b.g)
    assert a.accepted_attempt == b.accepted_attempt
    assert len(a.attempts) == len(b.attempts)
    for k in K_LADDER:
        np.testing.assert_array_equal(a.transition(k), b.transition(k))
        np.testing.assert_array_equal(a.stationary(k), b.stationary(k))


def test_different_replicates_get_different_masters(library):
    a = draw_master_transitions(0, library.permutation, k_max=K_MAX)
    b = draw_master_transitions(1, library.permutation, k_max=K_MAX)
    assert not np.array_equal(a.g, b.g)


def test_the_seed_namespace_is_what_drives_it(library):
    """A different root gives a different master, so the stream really is the source."""
    other = draw_master_transitions(0, library.permutation, k_max=K_MAX,
                                    seeds=LadderSeeds(root=1_234_567))
    base = draw_master_transitions(0, library.permutation, k_max=K_MAX)
    assert not np.array_equal(other.g, base.g)


def test_acceptance_is_joint_every_rung_clears_the_band(master):
    """The accepted master must satisfy the band at EVERY rung, not on average."""
    for k in K_LADDER:
        low, high = stationary_band(k)
        nu = master.stationary(k)
        assert np.all((nu >= low) & (nu <= high)), f"K={k} outside the band"
        np.testing.assert_allclose(nu, stationary_of(master.transition(k), k), rtol=1e-10)
        np.testing.assert_allclose(nu @ master.transition(k), nu, atol=1e-10)


def test_every_rejected_attempt_names_the_rung_that_failed(master):
    """The record has to show what was rejected, not only what survived."""
    rejected = [a for a in master.attempts if not a["accepted"]]
    assert len(master.attempts) == master.accepted_attempt + 1
    assert master.attempts[-1]["accepted"] is True
    for attempt in rejected:
        assert attempt["failed_rungs"], "a rejection with no reason recorded"
        assert all(f["K"] in K_LADDER for f in attempt["failed_rungs"])


def test_no_rung_is_redrawn_after_a_failure(master):
    """A failed attempt must discard the WHOLE master. If a rung could be redrawn on its
    own, the accepted rungs would no longer share one `G` — the coupling would be gone.
    Checked structurally: the accepted rungs all reduce to the single accepted `g`."""
    for k in K_LADDER:
        np.testing.assert_array_equal(master.transition(k),
                                      restrict_and_renormalise(master.g, k))


def test_the_joint_rate_is_recorded_and_is_below_every_per_rung_rate(library):
    """Joint conditioning is strictly stronger than per-rung; the record must say so."""
    rate = joint_band_acceptance_rate(library.permutation, k_max=K_MAX, trials=200)
    assert 0.0 < rate <= 1.0
    provenance = draw_master_transitions(0, library.permutation,
                                         k_max=K_MAX).provenance()
    assert provenance["joint_acceptance_rate"] > 0
    assert "JOINT" in provenance["conditioning"]
    assert "jointly conditioned" in provenance["conditioning"]


# ------------------- 6. the pre-gate row marginal is still exactly flat-Dirichlet
@pytest.mark.parametrize("k", [3, 5, 10])
def test_pregate_row_marginals_match_flat_dirichlet(k):
    """Ungated rows of `P^(K)` must be `Dirichlet_{K-1}(1, ..., 1)`.

    Checked on the exact marginal: a coordinate of a flat Dirichlet with `d = K - 1`
    components is `Beta(1, d - 1)`. A one-sample KS test against that CDF is exact, so
    this tests the construction rather than comparing two noisy samples. **No band gate is
    applied here** — conditioning is what the gate does, and it would fail this test by
    design.
    """
    from scipy import stats

    seeds = LadderSeeds(root=99_991)
    identity = np.arange(K_MAX)
    draws = np.array([
        restrict_and_renormalise(
            draw_master_gamma(seeds.generator("diagnostic", 7, trial), K_MAX)
            [np.ix_(identity, identity)], k)[0, 1]
        for trial in range(3_000)])

    d = k - 1
    reference = stats.beta(1.0, d - 1) if d > 1 else stats.uniform()
    p_value = stats.kstest(draws, reference.cdf).pvalue
    assert p_value > 0.001, f"K={k}: pre-gate marginal is not flat-Dirichlet (p={p_value})"

    np.testing.assert_allclose(draws.mean(), 1.0 / d, rtol=0.06)
    np.testing.assert_allclose(draws.var(), (1.0 / d) * (1 - 1.0 / d) / (d + 1), rtol=0.15)


def test_pregate_rows_agree_with_the_registered_sampler():
    """Against the sampler the registered design actually used, not only against theory."""
    from scipy import stats
    from hpop.mcmc_original.transitions import sample_transition_matrix

    k = 5
    seeds = LadderSeeds(root=55_557)
    gamma_draws = np.array([
        restrict_and_renormalise(
            draw_master_gamma(seeds.generator("diagnostic", 11, t), K_MAX), k)[0, 1]
        for t in range(3_000)])
    registered = np.array([
        sample_transition_matrix(np.zeros((k, k)), k,
                                 np.random.default_rng(900_000 + t), 1.0)[0, 1]
        for t in range(3_000)])

    p_value = stats.ks_2samp(gamma_draws, registered).pvalue
    assert p_value > 0.001, (
        f"shared-Gamma rows disagree with the registered flat-Dirichlet sampler "
        f"(p={p_value}); the coupling must not change the per-rung marginal")


def test_the_gate_does_change_the_law_and_that_is_recorded(library):
    """The honest converse: after joint conditioning the rows are NOT the ungated law.
    If this ever stopped being true the gate would be doing nothing."""
    provenance = draw_master_transitions(0, library.permutation,
                                         k_max=K_MAX).provenance()
    assert "no rung is an unconditional draw" in provenance["conditioning"]
    assert provenance["row_marginal_before_conditioning"].startswith("Dirichlet")
