"""Stage 6B — the joint diagnostics, and the calibration that gives them meaning.

An energy distance of 0.0014 means nothing until you know what two independent samples
*from the same distribution* produce at the same sample sizes. These tests check the
statistic behaves as a two-sample statistic should, that its envelope is calibrated rather
than chosen, and that the calibration refuses to run on a reference pool too small to
supply two disjoint samples — the failure that silently produced a `nan` envelope the
first time the Stage 6B2 diagnostics were run.

They also pin the distinction the whole stage rests on: **sampler correctness** (does the
chain reproduce the independent reference?) is a different claim from **synthetic
recovery** (does this one finite dataset put the generating value inside the interval?).
A correct sampler can fail the second.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hpop.mcmc_original.stage6b_joint_diagnostics import (
    calibrate_correlation_envelope, calibrate_energy_envelope, dependence_diagnostics,
    energy_distance, marginal_diagnostics, multivariate_comparison, recovery_table,
    standardise,
)

RESULTS = Path(__file__).resolve().parents[2] / "results" / "mcmc_original"


def gaussian_sample(n, seed, mean=(0.0, 0.0, 0.0), correlation=0.5):
    rng = np.random.default_rng(seed)
    d = len(mean)
    cov = np.full((d, d), correlation)
    np.fill_diagonal(cov, 1.0)
    return rng.multivariate_normal(np.asarray(mean), cov, size=n)


# ---------------------------------------------------------------- the energy distance
def test_energy_distance_is_zero_for_a_sample_against_itself():
    x = gaussian_sample(400, seed=0)
    assert energy_distance(x, x) == pytest.approx(0.0, abs=1e-10)


def test_energy_distance_is_small_for_two_samples_from_one_law():
    a, b = gaussian_sample(800, seed=1), gaussian_sample(800, seed=2)
    assert 0.0 <= energy_distance(a, b) < 0.05


def test_energy_distance_grows_with_a_mean_shift():
    a = gaussian_sample(800, seed=3)
    near = gaussian_sample(800, seed=4, mean=(0.2, 0.0, 0.0))
    far = gaussian_sample(800, seed=5, mean=(1.0, 0.0, 0.0))
    same = energy_distance(a, gaussian_sample(800, seed=6))
    assert energy_distance(a, near) > same
    assert energy_distance(a, far) > energy_distance(a, near)


def test_energy_distance_detects_a_dependence_change_the_marginals_hide():
    """Identical marginals, different correlation — exactly what a joint test is for."""
    a = gaussian_sample(1200, seed=7, correlation=0.5)
    b = gaussian_sample(1200, seed=8, correlation=-0.5)
    for i in range(3):
        assert abs(a[:, i].mean() - b[:, i].mean()) < 0.15
        assert abs(a[:, i].std() - b[:, i].std()) < 0.15
    null = energy_distance(a, gaussian_sample(1200, seed=9, correlation=0.5))
    assert energy_distance(a, b) > 10 * null


def test_energy_distance_is_symmetric():
    a, b = gaussian_sample(300, seed=10), gaussian_sample(300, seed=11, mean=(0.4, 0, 0))
    assert energy_distance(a, b) == pytest.approx(energy_distance(b, a), rel=1e-10)


# ------------------------------------------------------------------------ calibration
def test_the_energy_envelope_brackets_the_null_it_was_calibrated_on():
    pool = gaussian_sample(4000, seed=12)
    null = calibrate_energy_envelope(pool, 500, 500, n_replicates=30, seed=13)
    replicates = np.array(null["replicates"])
    assert null["envelope"] >= np.quantile(replicates, 0.98)
    assert (replicates <= null["envelope"]).mean() >= 0.95
    assert null["mean"] > 0 and null["sd"] > 0


def test_a_genuinely_different_sample_exceeds_the_calibrated_envelope():
    pool = gaussian_sample(4000, seed=14)
    null = calibrate_energy_envelope(pool, 600, 600, n_replicates=30, seed=15)
    shifted = gaussian_sample(600, seed=16, mean=(0.5, 0.0, 0.0))
    assert energy_distance(shifted, pool[:600]) > null["envelope"]


def test_correlation_envelope_refuses_a_pool_too_small_for_two_disjoint_samples():
    """The regression: asking for 32,000 from a 20,000 pool silently produced `nan`."""
    pool = gaussian_sample(1000, seed=17)
    with pytest.raises(ValueError, match="two disjoint samples"):
        calibrate_correlation_envelope(pool, n_draws=800, n_replicates=5, seed=18)
    envelope = calibrate_correlation_envelope(pool, n_draws=400, n_replicates=5, seed=18)
    assert np.isfinite(envelope["envelope"])


def test_the_correlation_envelope_shrinks_as_the_sample_grows():
    pool = gaussian_sample(20_000, seed=19)
    small = calibrate_correlation_envelope(pool, 500, n_replicates=20, seed=20)
    large = calibrate_correlation_envelope(pool, 5_000, n_replicates=20, seed=21)
    assert large["envelope"] < small["envelope"]
    assert np.isfinite(small["envelope"]) and np.isfinite(large["envelope"])


def test_standardise_centres_and_scales():
    x = gaussian_sample(500, seed=22) * 3.0 + 7.0
    z = standardise(x, x.mean(axis=0), x.std(axis=0, ddof=1))
    assert np.abs(z.mean(axis=0)).max() < 1e-10
    assert np.abs(z.std(axis=0, ddof=1) - 1.0).max() < 1e-10


# ------------------------------------------------------- correctness vs recovery (item 24)
def test_recovery_is_reported_separately_from_agreement_with_the_reference():
    """A chain can match its reference exactly and still miss the generating value."""
    active = ["beta", "omega", "lambda_rep"]
    rng = np.random.default_rng(23)
    # draws from the reference law itself: sampler correctness is perfect by construction
    # omega's spread is deliberately tight here: at the real Stage 6B sd of 0.126 the
    # truth sits 0.95 sd out and IS covered, so a narrower posterior is needed to build
    # the case this test exists to demonstrate
    draws = {"beta": rng.normal(1.49, 0.038, (4, 4000)),
             "omega": rng.normal(1.85, 0.040, (4, 4000)),
             "lambda_rep": rng.normal(0.80, 0.028, (4, 4000))}
    # the truth for omega now sits well outside that posterior
    truth = {"beta": 1.50, "omega": 1.7346, "lambda_rep": 0.80}
    recovery = recovery_table(draws, truth)

    assert recovery["beta"]["truth_in_95_interval"] is True
    assert recovery["omega"]["truth_in_95_interval"] is False
    assert recovery["omega"]["error_in_posterior_sd"] > 2.0
    # the recovery table says nothing about sampler correctness, and vice versa
    assert set(recovery["omega"]) >= {"true_value", "posterior_mean", "posterior_sd",
                                      "q025", "q975", "truth_in_95_interval",
                                      "absolute_error", "error_in_posterior_sd"}
    assert "ks_distance" not in recovery["omega"]
    assert "rhat" not in recovery["omega"]


def test_recovery_table_arithmetic_is_right():
    draws = {"x": np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])}
    recovery = recovery_table(draws, {"x": 2.0})
    assert recovery["x"]["posterior_mean"] == pytest.approx(3.0)
    assert recovery["x"]["posterior_median"] == pytest.approx(3.0)
    assert recovery["x"]["absolute_error"] == pytest.approx(1.0)
    assert recovery["x"]["error_in_posterior_sd"] == pytest.approx(
        1.0 / np.std([1, 2, 3, 4, 5], ddof=1))


# ------------------------------------------------------------------- dependence level
def test_dependence_diagnostics_recover_a_known_correlation():
    active = ["a", "b", "c"]
    sample = gaussian_sample(20_000, seed=24, correlation=0.5)
    chains = {name: sample[:, i].reshape(4, -1) for i, name in enumerate(active)}
    reference_corr = np.full((3, 3), 0.5)
    np.fill_diagonal(reference_corr, 1.0)
    summary = {"covariance": reference_corr.tolist(), "correlation": reference_corr.tolist(),
               "sd": {n: 1.0 for n in active}}
    result = dependence_diagnostics(chains, summary, active)
    assert result["max_correlation_error"] < 0.03
    assert len(result["pairs"]) == 3
    for pair in result["pairs"].values():
        assert pair["reference_correlation"] == pytest.approx(0.5)


def test_dependence_diagnostics_flag_a_wrong_correlation():
    active = ["a", "b", "c"]
    sample = gaussian_sample(20_000, seed=25, correlation=-0.4)
    chains = {name: sample[:, i].reshape(4, -1) for i, name in enumerate(active)}
    reference_corr = np.full((3, 3), 0.5)
    np.fill_diagonal(reference_corr, 1.0)
    summary = {"covariance": reference_corr.tolist(), "correlation": reference_corr.tolist(),
               "sd": {n: 1.0 for n in active}}
    assert dependence_diagnostics(chains, summary, active)["max_correlation_error"] > 0.8


# ------------------------------------------------------------- smoke schemas (items 21/22)
@pytest.mark.parametrize("directory,active", [
    ("stage6b2_joint3_smoke", ("beta", "omega", "lambda_rep")),
    ("stage6b3_joint4_smoke", ("beta", "omega", "lambda_rep", "lambda_back")),
])
def test_smoke_output_has_the_registered_schema(directory, active):
    path = RESULTS / directory
    if not (path / "summary.json").exists():
        pytest.skip(f"{directory} has not been run")

    summary = json.loads((path / "summary.json").read_text())
    for key in ("config", "checks", "acceptance_total", "acceptance_post_burn_in",
                "draws_per_chain", "pass"):
        assert key in summary, key
    for check in ("all_coordinates_move", "each_coordinate_accepts",
                  "each_coordinate_rejects", "no_nans_all_finite",
                  "q0_reset_and_state_reproducible", "state_serialises_and_loads",
                  "deterministic_resume"):
        assert check in summary["checks"], check
        assert summary["checks"][check]["pass"] is True
    assert summary["pass"] is True
    assert set(summary["acceptance_post_burn_in"]) == set(active)

    config = summary["config"]
    assert tuple(config["active"]) == active
    assert config["sweep_order"] == [n for n in
                                     ("beta", "omega", "lambda_rep", "lambda_back")
                                     if n in active]
    assert len(config["frozen_config_hash"]) == 64
    for forbidden in ("U", "rho", "P", "segmentation_boundaries", "skill_labels",
                      "epsilon"):
        assert forbidden in config["not_inferred"]

    stored = np.load(path / "chains.npz")
    for name in active:
        assert f"draws_{name}" in stored
        assert stored[f"draws_{name}"].shape[0] == config["settings"]["chains"]
    assert (path / "report.md").exists()
    assert (path / "config.json").exists()


def test_the_multivariate_comparison_reports_its_own_calibration():
    """The result must carry the evidence for its threshold, not just a verdict."""
    active = ["a", "b", "c"]
    reference = gaussian_sample(6000, seed=26)
    chains = {name: gaussian_sample(4000, seed=27)[:, i].reshape(4, -1)
              for i, name in enumerate(active)}
    summary = {"mean": {n: 0.0 for n in active}, "sd": {n: 1.0 for n in active}}
    result = multivariate_comparison(chains, reference, summary, active, seed=28,
                                     n_compare=500, n_replicates=12)
    for key in ("statistic", "observed", "envelope", "null_mean", "null_sd", "null_max",
                "n_replicates", "n_mcmc", "n_reference", "z_score", "pass"):
        assert key in result, key
    assert result["pass"] is True
    assert result["observed"] >= 0.0
    assert result["envelope"] > 0.0
