"""Stage 6B1, step 2b — the diagnostics themselves, on inputs whose answer is known.

A gate is only as trustworthy as the statistic behind it. These tests pin the ESS
estimator to the analytic AR(1) value, show that split R-hat catches a within-chain drift
that the plain multi-chain statistic misses, check the KS routine against SciPy, and —
the important one — run the whole comparison pipeline on draws taken *directly* from the
reference grid by inverse-CDF. That last test says: if a sampler really did target this
grid, the registered gates would pass. Any later failure is then the sampler's, not the
gate's.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from hpop.mcmc_original.recurrent_rfs import sigmoid
from hpop.mcmc_original.stage6b_mcmc_diagnostics import (
    autocovariance, bulk_ess, compare_to_reference, effective_sample_size_core,
    evaluate_gates, grid_summary, kappa_grid_from_omega, ks_distance_to_grid,
    load_reference_posteriors, rank_normalize, rank_normalized_split_rhat,
    split_chains, tail_ess,
)

REFERENCE = (Path(__file__).resolve().parents[2] / "results" / "mcmc_original"
             / "stage6b_full_seed0" / "reference_posteriors.json")


def ar1(rho: float, n: int, rng) -> np.ndarray:
    x = np.empty(n)
    x[0] = rng.normal(0.0, 1.0 / math.sqrt(1.0 - rho ** 2))
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal()
    return x


# ----------------------------------------------------------------------- autocorrelation
def test_autocovariance_matches_the_direct_definition():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    acov = autocovariance(x[None, :])[0]
    centred = x - x.mean()
    for k in (0, 1, 5, 20):
        direct = float(np.dot(centred[: len(x) - k], centred[k:]) / len(x))
        assert acov[k] == pytest.approx(direct, abs=1e-10)


def test_ess_of_independent_draws_is_about_the_number_of_draws():
    rng = np.random.default_rng(1)
    chains = rng.normal(size=(4, 4000))
    assert effective_sample_size_core(chains) == pytest.approx(16_000, rel=0.10)


@pytest.mark.parametrize("rho", [0.5, 0.8, 0.9])
def test_ess_of_an_ar1_chain_matches_the_analytic_value(rho):
    rng = np.random.default_rng(2)
    chains = np.array([ar1(rho, 20_000, rng) for _ in range(4)])
    expected = 4 * 20_000 * (1.0 - rho) / (1.0 + rho)
    assert effective_sample_size_core(chains) == pytest.approx(expected, rel=0.15)


def test_rank_normalisation_is_monotone_and_standardised():
    rng = np.random.default_rng(3)
    x = rng.lognormal(size=(4, 500))
    z = rank_normalize(x)
    assert np.array_equal(np.argsort(x, axis=None), np.argsort(z, axis=None))
    assert abs(float(z.mean())) < 0.02
    assert float(z.std()) == pytest.approx(1.0, rel=0.02)


# -------------------------------------------------------------------------------- R-hat
def test_rhat_is_about_one_for_chains_from_the_same_distribution():
    rng = np.random.default_rng(4)
    chains = rng.normal(size=(4, 5000))
    assert rank_normalized_split_rhat(chains)["rhat"] < 1.01


def test_rhat_detects_chains_stuck_in_different_places():
    rng = np.random.default_rng(5)
    chains = rng.normal(size=(4, 2000)) + np.array([0.0, 1.0, 2.0, 3.0])[:, None]
    assert rank_normalized_split_rhat(chains)["rhat"] > 1.5


def test_split_rhat_catches_a_within_chain_drift_that_plain_rhat_misses():
    """Every chain shares the same drift, so the chain means agree and only splitting sees it."""
    rng = np.random.default_rng(6)
    n = 4000
    drift = np.linspace(-3.0, 3.0, n)
    chains = rng.normal(scale=0.5, size=(4, n)) + drift

    plain_within = chains.var(axis=1, ddof=1).mean()
    plain_between = chains.mean(axis=1).var(ddof=1)
    plain = math.sqrt(((n - 1) / n * plain_within + plain_between) / plain_within)

    assert plain < 1.01, "unsplit R-hat is blind to a drift shared by all chains"
    assert rank_normalized_split_rhat(chains)["rhat"] > 1.20


def test_split_chains_doubles_the_chain_count():
    chains = np.arange(24, dtype=float).reshape(2, 12)
    split = split_chains(chains)
    assert split.shape == (4, 6)
    assert np.array_equal(split[0], chains[0, :6])
    assert np.array_equal(split[1], chains[1, :6])
    assert np.array_equal(split[2], chains[0, 6:])


def test_tail_ess_is_the_minimum_of_the_two_indicator_ess_values():
    """Checked against its definition, recomputed here from the 5% and 95% indicators."""
    rng = np.random.default_rng(7)
    chains = np.array([ar1(0.9, 10_000, rng) for _ in range(4)])
    low, high = np.quantile(chains, (0.05, 0.95))
    expected = min(
        effective_sample_size_core(rank_normalize(split_chains((chains <= low).astype(float)))),
        effective_sample_size_core(rank_normalize(split_chains((chains <= high).astype(float)))))
    assert tail_ess(chains) == pytest.approx(expected, rel=1e-12)


def test_tail_ess_of_independent_draws_is_about_the_number_of_draws():
    rng = np.random.default_rng(17)
    chains = rng.normal(size=(4, 4000))
    assert tail_ess(chains) == pytest.approx(16_000, rel=0.15)


def test_tail_ess_collapses_when_one_chain_under_explores_the_tail():
    """Rank statistics are invariant to monotone transforms, so a heavy tail is not by
    itself a tail-ESS problem. One chain failing to *reach* the tail is, and only the
    tail statistic sees it — the bulk stays at essentially the full draw count."""
    rng = np.random.default_rng(27)
    n = 5000
    fourth = rng.normal(size=n)
    fourth = np.where(fourth > 1.0, 1.0 - (fourth - 1.0) * 0.01, fourth)
    chains = np.vstack([rng.normal(size=(3, n)), fourth[None, :]])
    assert bulk_ess(chains) > 0.9 * 4 * n
    assert tail_ess(chains) < 0.10 * bulk_ess(chains)


# ------------------------------------------------------------------------------- KS
def test_ks_distance_to_grid_matches_scipy():
    rng = np.random.default_rng(8)
    grid = np.linspace(-8.0, 8.0, 20_001)
    cdf = stats.norm.cdf(grid)
    samples = rng.normal(size=5000)
    mine = ks_distance_to_grid(samples, grid, cdf)
    theirs = stats.kstest(samples, stats.norm.cdf).statistic
    assert mine == pytest.approx(theirs, abs=1e-4)


def test_ks_distance_grows_when_the_samples_are_shifted():
    rng = np.random.default_rng(9)
    grid = np.linspace(-8.0, 8.0, 20_001)
    cdf = stats.norm.cdf(grid)
    assert ks_distance_to_grid(rng.normal(size=5000), grid, cdf) < 0.03
    assert ks_distance_to_grid(rng.normal(size=5000) + 0.5, grid, cdf) > 0.15


# ------------------------------------------------------------- the reference grids
def test_kappa_view_reproduces_the_stored_kappa_summaries():
    """The kappa view is a change of variables on the stored density, not a new grid run."""
    reference = load_reference_posteriors(REFERENCE)
    entry = reference["posteriors"]["omega"]
    stored = entry["kappa"]
    derived = kappa_grid_from_omega(entry)
    for key in ("mean", "median", "q025", "q975"):
        assert derived[key] == pytest.approx(stored[key], rel=1e-6)
    assert derived["true_value"] == pytest.approx(sigmoid(entry["true_value"]), rel=1e-12)
    # the Jacobian is applied on the stored omega grid, so the kappa grid is non-uniform
    # and the trapezoid integral carries a small discretisation error of its own
    assert float(np.trapezoid(derived["density"], derived["grid"])) == pytest.approx(1.0, rel=1e-5)
    assert np.all(np.diff(derived["grid"]) > 0)


def test_reading_the_reference_does_not_modify_it():
    before = REFERENCE.read_bytes()
    reference = load_reference_posteriors(REFERENCE)
    for name, entry in reference["posteriors"].items():
        summary = grid_summary(entry)
        assert summary["sd"] > 0
    assert REFERENCE.read_bytes() == before


@pytest.mark.parametrize("parameter", ["beta", "omega", "lambda_rep", "lambda_back"])
def test_draws_taken_from_the_grid_itself_pass_every_registered_gate(parameter):
    """The gates are achievable: inverse-CDF draws from the reference clear all of them.

    This isolates the comparison machinery from the sampler. If a later Stage 6B1 run
    fails a gate, this test says the failure is the chain's, not the threshold's.
    """
    reference = load_reference_posteriors(REFERENCE)
    grid = grid_summary(reference["posteriors"][parameter])
    rng = np.random.default_rng(hash(parameter) % 2**32)
    chains = np.interp(rng.random((4, 5000)), grid["cdf"], grid["grid"])

    comparison = compare_to_reference(chains, grid)
    verdict = evaluate_gates(comparison, [0.35] * 4, mode="full")
    assert verdict["pass"], verdict["checks"]
    assert comparison["ks_distance"] < 0.03
    assert comparison["standardized_mean_error"] < 0.05


def test_gates_fail_on_a_deliberately_shifted_posterior():
    reference = load_reference_posteriors(REFERENCE)
    grid = grid_summary(reference["posteriors"]["beta"])
    rng = np.random.default_rng(21)
    chains = np.interp(rng.random((4, 5000)), grid["cdf"], grid["grid"]) + 0.5 * grid["sd"]

    verdict = evaluate_gates(compare_to_reference(chains, grid), [0.35] * 4, mode="full")
    assert not verdict["pass"]
    assert not verdict["checks"]["standardized_mean_error"]["pass"]
    assert not verdict["checks"]["ks_distance"]["pass"]


def test_acceptance_rate_outside_the_window_fails_the_gate():
    reference = load_reference_posteriors(REFERENCE)
    grid = grid_summary(reference["posteriors"]["beta"])
    rng = np.random.default_rng(22)
    chains = np.interp(rng.random((4, 5000)), grid["cdf"], grid["grid"])
    comparison = compare_to_reference(chains, grid)

    assert evaluate_gates(comparison, [0.20, 0.30, 0.40, 0.55])["pass"]
    assert not evaluate_gates(comparison, [0.20, 0.30, 0.40, 0.90])["pass"]
    assert not evaluate_gates(comparison, [0.02, 0.30, 0.40, 0.55])["pass"]


def test_the_registered_thresholds_are_the_ones_the_spec_states():
    from hpop.mcmc_original.stage6b_mcmc_diagnostics import FULL_GATES, SMOKE_GATES
    assert FULL_GATES["standardized_mean_error"] == 0.15
    assert FULL_GATES["standardized_median_error"] == 0.15
    assert FULL_GATES["standardized_q025_error"] == 0.25
    assert FULL_GATES["standardized_q975_error"] == 0.25
    assert FULL_GATES["ks_distance"] == 0.03
    assert FULL_GATES["rhat_max"] == 1.01
    assert FULL_GATES["bulk_ess_min"] == 1000.0
    assert FULL_GATES["tail_ess_min"] == 500.0
    assert FULL_GATES["acceptance_rate_range"] == (0.15, 0.60)
    assert SMOKE_GATES == {"standardized_mean_error": 0.35, "rhat_max": 1.05}


def test_reference_metadata_is_the_registered_experiment():
    reference = load_reference_posteriors(REFERENCE)
    assert reference["n_train_blocks"] == 500
    assert reference["T"] == 20
    assert reference["epsilon"] == 0.02
    assert reference["truth"]["beta"] == 1.5
    assert reference["truth"]["lambda_rep"] == 0.8
    assert reference["truth"]["lambda_back"] == 0.25
    assert reference["truth"]["omega"] == pytest.approx(math.log(0.85 / 0.15))
    assert json.loads(REFERENCE.read_text())["priors"]["omega"]["sd"] == 2.0
