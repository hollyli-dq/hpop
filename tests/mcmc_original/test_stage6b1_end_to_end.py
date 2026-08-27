"""Stage 6B1, step 4 — the whole pipeline, on a small corpus, against a grid it computes.

The registered runs live in `results/mcmc_original/stage6b1_full_seed0/`. This file is
the fast version of the same argument: build a reference grid for a 50-block corpus by
numerical integration, run the samplers, and require the same registered gates. It closes
the loop that the earlier files open separately — a correct engine (`test_..._scalar_mh`),
correct proposal ratios (`test_..._proposals`), correct diagnostics
(`test_..._diagnostics`) and a correct callback (`test_..._recurrent_target`) must
together produce a posterior that matches the grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_original.recurrent_scalar_mcmc import (
    PROPOSAL_KIND, REGISTERED_STARTS, RecurrentScalarTarget, ScalarMHConfig,
    build_proposal, curvature_proposal_scale, run_scalar_mh, tune_proposal_scale,
)
from hpop.mcmc_original.recurrent_scalar_posterior import (
    TRUE_VALUES, log_prior, normalize_log_density_grid,
)
from hpop.mcmc_original.recurrent_synthetic import U_TRUE, generate_recurrent_dataset
from hpop.mcmc_original.stage6b_mcmc_diagnostics import (
    compare_to_reference, evaluate_gates,
)

EPSILON = 0.02
RANGES = {"beta": (0.2, 4.0), "omega": (-1.5, 5.0),
          "lambda_rep": (0.01, 3.0), "lambda_back": (0.001, 2.0)}


@pytest.fixture(scope="module")
def corpus():
    data = generate_recurrent_dataset("smoke", seed=0)
    return np.array([b.roles for b in data.train], dtype=int)


def build_grid(target, parameter, n_points=1201):
    """A reference grid for this corpus, by the Stage 6B0 recipe: trapezoid on the
    original scale, never a softmax over grid values."""
    lo, hi = RANGES[parameter]
    grid = np.linspace(lo, hi, n_points)
    log_density = np.array([target(float(x))[0] for x in grid])
    density, cdf, _ = normalize_log_density_grid(grid, log_density)
    quantile = lambda level: float(np.interp(level, cdf, grid))   # noqa: E731
    mean = float(np.trapezoid(grid * density, grid))
    var = float(np.trapezoid((grid - mean) ** 2 * density, grid))
    return {"grid": grid, "density": density, "cdf": cdf, "mean": mean,
            "sd": float(np.sqrt(var)), "median": quantile(0.5),
            "q025": quantile(0.025), "q975": quantile(0.975),
            "true_value": TRUE_VALUES[parameter]}


@pytest.mark.parametrize("parameter", ["beta", "omega"])
def test_the_sampler_reproduces_a_grid_computed_from_the_same_corpus(parameter, corpus):
    target = RecurrentScalarTarget(parameter, corpus, U_TRUE, TRUE_VALUES, EPSILON)
    grid = build_grid(target, parameter)
    assert float(np.trapezoid(grid["density"], grid["grid"])) == pytest.approx(1.0, rel=1e-6)

    starts = REGISTERED_STARTS[parameter]
    scale = tune_proposal_scale(
        target, curvature_proposal_scale(target, TRUE_VALUES[parameter])["scale"],
        starts[0], 4242, iterations=1500)["final_scale"]

    chains, rates = [], []
    for c, start in enumerate(starts):
        config = ScalarMHConfig(parameter, scale, float(start), 4000, 1000, 2, 500 + c)
        result = run_scalar_mh(config, target, build_proposal(parameter, scale))
        chains.append(result.samples)
        rates.append(result.post_burn_in_acceptance_rate)
    comparison = compare_to_reference(np.array(chains), grid)

    verdict = evaluate_gates(comparison, rates, mode="full")
    # bulk/tail ESS gates are written for the 20,000-iteration registered runs; the
    # distributional gates are the ones this short run is asserting.
    for check in ("standardized_mean_error", "standardized_median_error",
                  "standardized_q025_error", "standardized_q975_error", "ks_distance",
                  "rhat", "min_acceptance_rate", "max_acceptance_rate"):
        assert verdict["checks"][check]["pass"], (check, verdict["checks"][check])


def test_chains_from_dispersed_starts_agree_with_each_other(corpus):
    """Every registered start must reach the same place; this is what R-hat encodes."""
    target = RecurrentScalarTarget("lambda_back", corpus, U_TRUE, TRUE_VALUES, EPSILON)
    scale = curvature_proposal_scale(target, TRUE_VALUES["lambda_back"])["scale"]
    means = []
    for c, start in enumerate(REGISTERED_STARTS["lambda_back"]):
        config = ScalarMHConfig("lambda_back", scale, float(start), 4000, 1000, 2, 700 + c)
        means.append(float(run_scalar_mh(config, target, build_proposal("lambda_back", scale))
                           .samples.mean()))
    spread = max(means) - min(means)
    assert spread < 0.02, means


def test_only_the_named_parameter_moves(corpus):
    """The other three scalars are held at truth — verified by the value actually used."""
    target = RecurrentScalarTarget("beta", corpus, U_TRUE, TRUE_VALUES, EPSILON)
    baseline = target.log_likelihood(TRUE_VALUES["beta"])
    from hpop.mcmc_original.recurrent_scalar_posterior import (
        batch_recurrent_log_likelihood_full_replay as replay)
    assert baseline == pytest.approx(replay(
        corpus, U_TRUE, TRUE_VALUES["beta"], EPSILON, TRUE_VALUES["omega"],
        TRUE_VALUES["lambda_rep"], TRUE_VALUES["lambda_back"]), abs=1e-9)
    assert target.log_likelihood(2.5) != pytest.approx(baseline)


def test_registered_starts_bracket_the_truth_and_are_valid():
    for parameter, starts in REGISTERED_STARTS.items():
        assert len(starts) == 4
        assert min(starts) < TRUE_VALUES[parameter] < max(starts)
        if PROPOSAL_KIND[parameter] == "log":
            assert all(s > 0 for s in starts)


def test_pilot_draws_do_not_leak_into_the_reported_chains(corpus):
    """The pilot picks a scale and is then discarded; chains restart from the registered
    starts with their own seeds, so the reported posterior contains no pilot state."""
    target = RecurrentScalarTarget("beta", corpus, U_TRUE, TRUE_VALUES, EPSILON)
    initial = curvature_proposal_scale(target, TRUE_VALUES["beta"])["scale"]
    tuning = tune_proposal_scale(target, initial, REGISTERED_STARTS["beta"][0], 4242,
                                 iterations=1500)
    assert set(tuning) >= {"initial_scale", "pilot_acceptance_rate", "adjusted", "final_scale"}
    assert "samples" not in tuning and "pilot_samples" not in tuning

    config = ScalarMHConfig("beta", tuning["final_scale"], REGISTERED_STARTS["beta"][0],
                            2000, 500, 2, 12345)
    a = run_scalar_mh(config, target, build_proposal("beta", tuning["final_scale"]))
    b = run_scalar_mh(config, target, build_proposal("beta", tuning["final_scale"]))
    assert np.array_equal(a.samples, b.samples), "a chain depends only on its own seed"
