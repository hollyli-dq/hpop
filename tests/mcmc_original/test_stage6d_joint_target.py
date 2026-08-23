"""Stage 6D — the direct joint target and its decomposition
(§18 areas 1, 2, 21, 22, 34, 35, 36, 47).

`log_target_6d` is evaluated entirely outside the transition code and never calls an
acceptance-ratio helper, so comparing an implemented ratio against a difference of these
values compares two genuinely independent routes.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_latent_poset_mcmc import LatentPosetEvaluator
from hpop.mcmc_original.recurrent_oracle_joint_mcmc import (
    Stage6DState, Stage6DTarget, initial_state, log_target_6d, run_oracle_joint_mcmc,
    sweep_once,
)
from hpop.mcmc_original.recurrent_scalar_posterior import log_prior
from hpop.mcmc_original.sampler_u import propose_row
from hpop.mcmc_original.stage6d_frozen import (
    ACTIVE_6D, REGISTERED_SCALES, SCALAR_ORDER, SWEEP_ORDER_6D, config_hash,
    frozen_config, load_stage6d_dataset, log_rho_prior, log_structural_prior,
)


@pytest.fixture(scope="module")
def frozen():
    return load_stage6d_dataset()


@pytest.fixture(scope="module")
def blocks(frozen):
    return frozen.train[:30]


@pytest.fixture()
def target(blocks, frozen):
    evaluator = LatentPosetEvaluator(blocks, epsilon=frozen.epsilon,
                                     omega=frozen.truth["omega"])
    return Stage6DTarget(evaluator, active=ACTIVE_6D)


def start_values(frozen, rho=0.3):
    return {"rho": rho, **{k: float(v) for k, v in frozen.truth.items()}}


# ----------------------------------------------------- area 1: target decomposition
def test_target_is_the_sum_of_its_named_components(target, frozen):
    rng = np.random.default_rng(0)
    for _ in range(30):
        u = frozen.u_true + rng.normal(scale=0.4, size=(5, 2))
        values = {"rho": float(rng.uniform(0.05, 0.9)),
                  "beta": float(rng.uniform(0.5, 3.0)),
                  "omega": float(rng.normal(1.5, 0.5)),
                  "lambda_rep": float(rng.uniform(0.3, 1.5)),
                  "lambda_back": float(rng.uniform(0.05, 0.8))}
        parts = log_target_6d(target, u, values)
        expected = (parts["log_likelihood"] + parts["log_structural_prior"]
                    + parts["log_rho_prior"] + sum(parts["log_scalar_priors"].values()))
        assert parts["log_target"] == pytest.approx(expected, abs=1e-12)
        assert set(parts["log_scalar_priors"]) == set(SCALAR_ORDER)
        assert parts["log_structural_prior"] == pytest.approx(
            log_structural_prior(u, values["rho"]), abs=1e-12)
        assert parts["log_rho_prior"] == pytest.approx(
            log_rho_prior(values["rho"]), abs=1e-12)
        for name in SCALAR_ORDER:
            assert parts["log_scalar_priors"][name] == pytest.approx(
                log_prior(name, values[name]), abs=1e-12)


def test_there_is_no_second_prior_on_the_induced_order(target, frozen):
    """Two U with the SAME induced order but different prior density must differ only in
    `p(U | rho)` — a `p(H | rho)` term would make the difference structural instead."""
    u = frozen.u_true
    rescaled = u * 2.0
    assert np.array_equal(precedence_from_u(u), precedence_from_u(rescaled))
    values = start_values(frozen)
    a = log_target_6d(target, u, values)
    b = log_target_6d(target, rescaled, values)
    assert a["log_likelihood"] == pytest.approx(b["log_likelihood"], abs=1e-9)
    assert (b["log_target"] - a["log_target"]) == pytest.approx(
        b["log_structural_prior"] - a["log_structural_prior"], abs=1e-12)
    assert frozen_config()["second_prior_on_H"] is False


def test_a_non_finite_component_makes_the_target_non_finite(target, frozen):
    assert log_target_6d(target, frozen.u_true,
                         start_values(frozen, rho=1.5))["log_target"] == -math.inf


# ------------------------------------ area 2: U is continuous, H = h(U) is derived
def test_u_is_continuous_and_h_is_derived(target, frozen):
    """The state is a real matrix; H is recomputed from it and is not carried."""
    rng = np.random.default_rng(1)
    state = initial_state(target, frozen.u_true, start_values(frozen), rng)
    assert isinstance(state.u, np.ndarray) and state.u.dtype.kind == "f"
    assert state.u.shape == (5, 2)
    assert state.relation_count == int(precedence_from_u(state.u).sum())
    # nothing in the serialised state is a poset identifier used as state
    payload = state.to_dict()
    assert "u" in payload and isinstance(payload["u"], list)
    assert payload["relation_count"] == state.relation_count


def test_arbitrarily_small_u_changes_are_representable(frozen):
    """A continuous state admits changes a discrete one could not."""
    u = frozen.u_true.copy()
    nudged = u.copy()
    nudged[0, 0] += 1e-9
    assert not np.array_equal(u, nudged)
    assert np.array_equal(precedence_from_u(u), precedence_from_u(nudged))


# --------------------------------- area 21: direct-ratio equality for every update
def test_direct_ratio_identity_for_every_update_type(target, frozen):
    """implemented ratio == direct target difference + reverse - forward, per coordinate."""
    rng = np.random.default_rng(2)
    u = frozen.u_true
    values = start_values(frozen)

    # U: symmetric proposal, so the Hastings term is exactly zero
    for _ in range(20):
        row = int(rng.integers(5))
        candidate = propose_row(u, row, REGISTERED_SCALES["U"], rng)
        implemented = ((target.log_likelihood(candidate, values, allow_cache=False)
                        - target.log_likelihood(u, values, allow_cache=False))
                       + (log_structural_prior(candidate, values["rho"])
                          - log_structural_prior(u, values["rho"])))
        direct = (log_target_6d(target, candidate, values)["log_target"]
                  - log_target_6d(target, u, values)["log_target"])
        assert implemented == pytest.approx(direct, abs=1e-9)

    # the four scalars: the prior + likelihood difference is the whole ratio
    for name in SCALAR_ORDER:
        for candidate in (0.6, 1.1, 1.9):
            trial = dict(values)
            trial[name] = candidate
            implemented = ((log_prior(name, candidate) - log_prior(name, values[name]))
                           + (target.log_likelihood(u, trial, allow_cache=False)
                              - target.log_likelihood(u, values, allow_cache=False)))
            direct = (log_target_6d(target, u, trial)["log_target"]
                      - log_target_6d(target, u, values)["log_target"])
            assert implemented == pytest.approx(direct, abs=1e-9)


def test_rho_update_touches_no_likelihood_term(target, frozen):
    values = start_values(frozen)
    seen = {target.log_likelihood(frozen.u_true, {**values, "rho": r},
                                  allow_cache=False) for r in (0.05, 0.3, 0.6, 0.9)}
    assert len(seen) == 1


def test_a_rho_update_consumes_zero_likelihood_evaluations(target, frozen):
    """A sweep replays exactly `m` times for U plus once for omega — rho adds none."""
    rng = np.random.default_rng(3)
    state = initial_state(target, frozen.u_true, start_values(frozen), rng)
    target.evaluator.full_replay_calls = 0
    n_sweeps = 12
    for _ in range(n_sweeps):
        state = sweep_once(state, target, REGISTERED_SCALES, rng)
    assert target.evaluator.full_replay_calls == n_sweeps * (frozen.n_roles + 1)


# ------------------------------- area 22: later updates see earlier accepted states
def test_every_recorded_component_matches_the_final_accepted_state(target, frozen):
    rng = np.random.default_rng(4)
    state = initial_state(target, frozen.u_true, start_values(frozen), rng)
    for _ in range(40):
        state = sweep_once(state, target, REGISTERED_SCALES, rng)
        recomputed = log_target_6d(target, state.u, state.values)
        assert state.log_structural_prior == pytest.approx(
            recomputed["log_structural_prior"], abs=1e-9)
        assert state.log_target == pytest.approx(recomputed["log_target"], abs=1e-7)


def test_sweep_order_is_the_registered_one():
    assert SWEEP_ORDER_6D == ("U", "rho", "beta", "omega", "lambda_rep", "lambda_back")
    assert frozen_config()["sweep_order"] == list(SWEEP_ORDER_6D)


# --------------------------- areas 34, 35, 36: reproduction, streams, resume
def test_same_seed_reproduces_the_run(blocks, frozen):
    def run():
        evaluator = LatentPosetEvaluator(blocks, epsilon=frozen.epsilon,
                                         omega=frozen.truth["omega"])
        return run_oracle_joint_mcmc(Stage6DTarget(evaluator), frozen.u_true,
                                     start_values(frozen), 40, 5, 2, seed=42)

    np.random.seed(1)
    a = run()
    np.random.seed(999)
    b = run()
    for name in ("rho",) + SCALAR_ORDER:
        assert np.array_equal(a.scalars[name], b.scalars[name])
    assert np.array_equal(a.u_draws, b.u_draws)


def test_different_seeds_give_independent_streams(blocks, frozen):
    results = []
    for seed in (1, 2, 3):
        evaluator = LatentPosetEvaluator(blocks, epsilon=frozen.epsilon,
                                         omega=frozen.truth["omega"])
        results.append(run_oracle_joint_mcmc(Stage6DTarget(evaluator), frozen.u_true,
                                             start_values(frozen), 40, 5, 2, seed=seed))
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            assert not np.array_equal(results[i].scalars["rho"],
                                      results[j].scalars["rho"])


def test_a_split_run_resumes_bit_identically(blocks, frozen):
    def fresh():
        return Stage6DTarget(LatentPosetEvaluator(
            blocks, epsilon=frozen.epsilon, omega=frozen.truth["omega"]))

    whole = run_oracle_joint_mcmc(fresh(), frozen.u_true, start_values(frozen),
                                  60, 10, 1, seed=13)
    first = run_oracle_joint_mcmc(fresh(), frozen.u_true, start_values(frozen),
                                  30, 10, 1, seed=13)
    rng = np.random.default_rng(13)
    rng.bit_generator.state = first.final_state.rng_state
    second = run_oracle_joint_mcmc(fresh(), frozen.u_true, start_values(frozen),
                                   60, 10, 1, seed=13, state=first.final_state, rng=rng)
    for name in ("rho",) + SCALAR_ORDER:
        assert np.array_equal(
            np.concatenate([first.scalars[name], second.scalars[name]]),
            whole.scalars[name])
    assert second.final_state.iteration == 60


def test_state_survives_a_json_round_trip(target, frozen):
    rng = np.random.default_rng(5)
    state = initial_state(target, frozen.u_true, start_values(frozen), rng)
    state = sweep_once(state, target, REGISTERED_SCALES, rng)
    restored = Stage6DState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert np.array_equal(restored.u, state.u)
    assert restored.values == state.values
    assert restored.invalid == state.invalid
    assert restored.rng_state == state.rng_state


def test_serialised_state_carries_everything_section_six_requires(target, frozen):
    rng = np.random.default_rng(6)
    state = sweep_once(initial_state(target, frozen.u_true, start_values(frozen), rng),
                       target, REGISTERED_SCALES, rng)
    payload = state.to_dict()
    for key in ("u", "values", "log_likelihood", "log_structural_prior", "log_rho_prior",
                "log_scalar_priors", "log_target", "relation_count", "poset_key_hex",
                "proposed", "accepted", "invalid", "iteration", "rng_state"):
        assert key in payload, key
    assert set(payload["values"]) == {"rho", *SCALAR_ORDER}


# ------------------------ area 47: n_skills vs rows vs latent columns vs assessors
def test_row_count_latent_dimension_skills_and_assessors_are_distinct(frozen):
    assert frozen.n_roles == 5
    assert frozen.latent_dimension == 2
    assert frozen.n_skills == 1
    assert frozen.n_assessors == 0
    assert len({frozen.n_roles, frozen.latent_dimension, frozen.n_skills}) == 3
    dims = frozen_config()["dimensions"]
    assert dims["m_rows"] == 5 and dims["d_latent_columns"] == 2
    assert dims["n_skills"] == 1 and dims["n_assessors"] == 0


def test_config_hash_is_stable_and_records_stage6c():
    assert config_hash() == config_hash()
    assert frozen_config()["stage6c_config_hash"].startswith("c1545b0b")
