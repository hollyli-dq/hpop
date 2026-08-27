"""The support-only baseline, and the finding it produced.

The baseline exists to answer one question: when the ladder reports recovery at K = 30, is
that the partial-order component working, or is it support matching? These tests pin the
answer measured on this design, which is stronger than expected -- the support rule alone
determines the entire candidate set, and the recurrent likelihood eliminates nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_cpa import CPABlockScoreTable                            # noqa: E402
from hpop.mcmc_cpa.corpus import generate_ladder_corpus                 # noqa: E402
from hpop.mcmc_cpa.nested_library import draw_master_library            # noqa: E402
from hpop.mcmc_cpa.support_baseline import (SupportOnlyBlockScoreTable,  # noqa: E402
                                            candidate_survival,
                                            expected_compatible_wrong_skills)
from hpop.mcmc_original.recurrent_scalar_posterior import TRUE_VALUES   # noqa: E402

SCALARS = (float(TRUE_VALUES["beta"]), float(TRUE_VALUES["omega"]),
           float(TRUE_VALUES["lambda_rep"]), float(TRUE_VALUES["lambda_back"]))


@pytest.fixture(scope="module")
def library():
    return draw_master_library(0)[0]


def tables_for(library, K):
    corpus = generate_ladder_corpus(library, K, 0)
    u, _ = library.prefix(K)
    full = CPABlockScoreTable(traces=corpus.traces("train"), epsilon=0.02,
                              role_maps=corpus.role_maps, min_width=3, max_width=12)
    full.refresh(u, *SCALARS)
    support = SupportOnlyBlockScoreTable(traces=corpus.traces("train"),
                                         role_maps=corpus.role_maps,
                                         min_width=3, max_width=12)
    return corpus, full, support


# ------------------------------------------------------- the finding, pinned
@pytest.mark.parametrize("K", (3, 10, 30))
def test_the_support_rule_alone_fixes_the_entire_candidate_set(K, library):
    """An INVARIANT, not an experimental finding.

    Since epsilon > 0, every in-support role sequence has positive probability, so the
    recurrent log score is finite exactly when the block is support-compatible. The two
    tables therefore agree on `isfinite` by construction. Worth pinning as a contract --
    it would break if the contamination component were ever removed -- but it is not
    evidence about how much the partial-order component contributes.

    A support-compatible block always receives a finite recurrent score, and an
    incompatible one is -inf under both models. So candidate survival is decided entirely
    by support membership, and any recovery difference between the two models comes from
    how they *weight* surviving candidates, never from which ones exist.
    """
    corpus, full, support = tables_for(library, K)
    report = candidate_survival(corpus, full, support)
    assert report["pooled_support_only_survival"] == pytest.approx(
        report["pooled_full_model_survival"], abs=1e-12)
    for n in range(len(corpus.train)):
        assert np.array_equal(np.isfinite(full.tables[n]),
                              np.isfinite(support.tables[n]))


@pytest.mark.parametrize("K", (3, 10, 30))
def test_the_true_skill_always_survives_and_wrong_skills_mostly_do_not(K, library):
    corpus, full, support = tables_for(library, K)
    report = candidate_survival(corpus, full, support)
    assert report["true_pair_support_survival"] == pytest.approx(1.0, abs=1e-12), (
        "a block's own skill must always be support-compatible with it")
    assert report["wrong_skill_survival_CONTAINED_BLOCKS_ONLY"] < 0.05


def test_the_two_survival_columns_are_named_apart(library):
    """They average over different populations and must not be confusable.

    `non_true_pair_survival_ALL_BLOCKS` counts every candidate of a boundary-crossing
    block as a non-true pair; the contained-only figure does not. Only the latter
    satisfies E[C_b] = (K-1) * p, and the note has to say which is which.
    """
    corpus, full, support = tables_for(library, 10)
    report = candidate_survival(corpus, full, support)
    assert "non_true_pair_survival_ALL_BLOCKS" in report
    assert "wrong_skill_survival_CONTAINED_BLOCKS_ONLY" in report
    assert "false_pair_support_survival" not in report, "the ambiguous name is gone"
    assert report["non_true_pair_survival_ALL_BLOCKS"] != pytest.approx(
        report["wrong_skill_survival_CONTAINED_BLOCKS_ONLY"], rel=1e-6), (
        "if these ever coincide the test has stopped discriminating")
    assert "does NOT satisfy" in report["NOTE"]


def test_the_accounting_identity_holds_exactly(library):
    """`E[C_b] = (K-1) * p_pair` on one shared population. A mismatch is a bookkeeping bug.

    An earlier report placed a per-pair rate beside a per-block expectation when the two
    had been averaged over different populations: the per-pair figure counted every
    candidate of a boundary-crossing block as a "wrong skill" pair, the per-block one
    counted only blocks contained in a single true segment. The numbers looked like a
    finding and were an artefact. The identity is arithmetic, so this pins it.
    """
    from hpop.mcmc_cpa.support_baseline import accounting_check

    for K in (3, 5, 10, 30):
        corpus, _full, support = tables_for(library, K)
        check = accounting_check(corpus, support)
        assert abs(check["identity_lhs_minus_rhs"]) < 1e-12, (K, check)
        assert check["n_blocks"] > 0


def test_ambiguity_grows_between_the_small_and_large_ends(library):
    """Ambiguity rises with K -- but the empirical curve is not required to be monotone.

    Measured E[false compatible skills per block] is 0.040, 0.036, 0.137, 0.268, 0.419 at
    K = 3, 5, 10, 20, 30: approximately flat over K = 3-5, then increasing substantially.
    Asserting strict monotonicity would be testing the shape of one random realisation
    rather than anything the design guarantees, so the assertion is on the ends.
    """
    from hpop.mcmc_cpa.support_baseline import accounting_check

    small = accounting_check(*tables_for(library, 3)[::2])["expected_wrong_per_block"]
    large = accounting_check(*tables_for(library, 30)[::2])["expected_wrong_per_block"]
    assert large > 4 * small, (small, large)


def test_per_pair_compatibility_does_not_depend_on_K_in_theory(library):
    """`p_d = [C(A-d, m-d) - 1] / [C(A, m) - 1]` contains no K.

    Only the number of competing skills grows. So a measured fall in per-pair survival
    across rungs reflects a change in the block population averaged over -- the mix of
    distinct-CPA counts, or the share of boundary-crossing blocks -- never K making an
    individual pair less compatible.
    """
    from hpop.mcmc_cpa.support_baseline import pairwise_compatibility

    for d in (1, 2, 3, 4):
        at_3 = pairwise_compatibility(3, 10, 50, d)
        at_30 = pairwise_compatibility(30, 10, 50, d)
        assert at_3["p_pair"] == pytest.approx(at_30["p_pair"], rel=1e-15)
        assert at_30["expected_wrong"] == pytest.approx(29 * at_3["p_pair"], rel=1e-12)
        assert 0.0 <= at_30["p_at_least_one_wrong"] <= 1.0


def test_conditional_ambiguity_rises_with_K(library):
    """Once a block is ambiguous it faces more wrong skills, not just more often."""
    from hpop.mcmc_cpa.support_baseline import accounting_check

    conditional = {}
    for K in (10, 20, 30):
        corpus, _full, support = tables_for(library, K)
        conditional[K] = accounting_check(corpus, support)["mean_wrong_GIVEN_at_least_one"]
    assert conditional[10] < conditional[20] < conditional[30], conditional
    assert conditional[30] > 1.5


def test_cross_boundary_shows_no_systematic_improvement(library):
    """Not "flat" -- it varies. The claim is only that it does not improve with K."""
    from hpop.mcmc_cpa.support_baseline import block_ambiguity

    means = {}
    for K in (3, 5, 10, 20, 30):
        corpus, _full, support = tables_for(library, K)
        means[K] = block_ambiguity(corpus, support)[
            "blocks_crossing_a_true_boundary"]["mean_compatible_skills"]
    assert means[30] >= means[3] * 0.8, (
        f"cross-boundary exclusion must not systematically improve with K, got {means}")
    assert all(0.05 < v < 0.2 for v in means.values()), means


# ------------------------------------------------------------ the arithmetic behind it
@pytest.mark.parametrize("d", (1, 2, 3, 4))
def test_expected_compatible_wrong_skills_matches_the_closed_form(d):
    """Conditioned on pairwise-distinct supports, which the master library requires."""
    from math import comb
    A, m, K = 50, 10, 30
    expected = (K - 1) * (comb(A - d, m - d) - 1) / (comb(A, m) - 1)
    assert expected_compatible_wrong_skills(K, m, A, d) == pytest.approx(expected,
                                                                          rel=1e-12)


def test_a_full_support_match_is_impossible_under_distinctness():
    """At d = m the only support containing all of them is the true one.

    The unconditional formula reports a small positive probability for an event that
    distinctness makes exactly impossible.
    """
    assert expected_compatible_wrong_skills(30, 10, 50, 10) == 0.0
    assert expected_compatible_wrong_skills(30, 10, 50, 10,
                                             pairwise_distinct=False) > 0.0


def test_a_block_needing_more_distinct_cpas_than_a_support_holds_is_impossible():
    assert expected_compatible_wrong_skills(30, 10, 50, 11) == 0.0


# ----------------------------------------------------------------- stratification
def test_survival_is_stratified_by_everything_that_drives_it(library):
    corpus, full, support = tables_for(library, 10)
    report = candidate_survival(corpus, full, support)
    assert report["strata"], "no strata produced"
    for row in report["strata"]:
        for field in ("block_length", "distinct_cpas", "true_skill_pair",
                      "crosses_true_boundary", "candidates",
                      "support_only_survival", "full_model_survival"):
            assert field in row, field
        assert 0.0 <= row["support_only_survival"] <= 1.0
    lengths = {row["block_length"] for row in report["strata"]}
    assert lengths == set(range(3, 13))
    assert {row["crosses_true_boundary"] for row in report["strata"]} == {True, False}
    assert report["total_candidate_pairs"] == sum(r["candidates"]
                                                  for r in report["strata"])


def test_more_distinct_cpas_means_fewer_surviving_wrong_skills(library):
    """The mechanism, measured rather than assumed."""
    corpus, full, support = tables_for(library, 30)
    report = candidate_survival(corpus, full, support)
    by_distinct: dict = {}
    for row in report["strata"]:
        if row["true_skill_pair"]:
            continue
        bucket = by_distinct.setdefault(row["distinct_cpas"], [0, 0])
        bucket[0] += row["support_only_survival"] * row["candidates"]
        bucket[1] += row["candidates"]
    rates = {d: total / n for d, (total, n) in sorted(by_distinct.items()) if n}
    low = [rates[d] for d in sorted(rates) if d <= 3]
    high = [rates[d] for d in sorted(rates) if d >= 6]
    assert low and high
    assert min(low) > max(high), (
        f"survival should fall with distinct CPA count, got {rates}")
