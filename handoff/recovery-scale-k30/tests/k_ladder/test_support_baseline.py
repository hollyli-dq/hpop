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
    """The measured result: the recurrent likelihood eliminates no candidate at all.

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
    assert report["false_pair_support_survival"] < 0.05
    # the discrimination sharpens with K, because supports crowd a fixed vocabulary
    assert report["false_pair_support_survival"] < report["pooled_support_only_survival"]


def test_wrong_skill_survival_falls_as_K_grows(library):
    rates = {}
    for K in (3, 10, 30):
        corpus, full, support = tables_for(library, K)
        rates[K] = candidate_survival(corpus, full, support)["false_pair_support_survival"]
    assert rates[3] > rates[10] > rates[30], rates


# ------------------------------------------------------------ the arithmetic behind it
@pytest.mark.parametrize("d,expected", [(1, 5.8), (2, 1.065306), (3, 0.177551), (4, 0.026444)])
def test_expected_compatible_wrong_skills_matches_the_closed_form(d, expected):
    assert expected_compatible_wrong_skills(30, 10, 50, d) == pytest.approx(expected,
                                                                            rel=1e-4)


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
