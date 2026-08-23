"""Focused correctness tests for the unanchored FULL-LATENT experiment.

These use a tiny throwaway observed-only model.  They neither load the formal corpus nor
open synthetic truth, so they are safe to run before formal launch.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import dirichlet

from hpop.mcmc_original import matched_full_latent as mfl
from hpop.mcmc_original import full_latent_constants as constants
from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.stage6e_state import initial_counts, transition_counts_of
from hpop.mcmc_original.transitions import allowed_next, sample_transition_matrix


TRACES = (
    (0, 1, 2, 3, 4, 0),
    (1, 0, 2, 4, 3, 1, 0),
    (2, 3, 1, 0, 4, 2, 1, 3, 0, 4),
    (4, 0, 1, 3, 2, 4, 0, 2, 1, 3, 4, 0),
)


def setup_state(seed: int = 41):
    fixed = mfl.FullLatentFixed()
    model = mfl.build_full_latent_model(TRACES, fixed)
    pi, p = mfl.draw_initial_pi_p(model, seed + 100)
    u = mfl.make_u_start(0, seed + 200, 0.5, fixed, model.n_skills,
                         model.n_roles, 2)
    state = mfl.initial_full_latent_state(model, u, pi, p, fixed)
    probes = mfl.select_truth_free_probes(TRACES, "throwaway-corpus-hash",
                                           boundary_count=7, coskill_count=9,
                                           recovery_coskill_count=15)
    return fixed, model, state, probes


def test_pi_and_p_gibbs_are_the_registered_conjugate_conditionals():
    fixed, model, state, _ = setup_state()
    alpha_pi = model.eta_initial + initial_counts(state.segmentations, model.n_skills)
    counts = transition_counts_of(state.segmentations, model.n_skills)

    # Density formula is independently checked against scipy's normalized Dirichlet.
    assert mfl.conditional_pi_log_density(state.pi, state.segmentations, model) == pytest.approx(
        dirichlet.logpdf(state.pi, alpha_pi))
    p_density = 0.0
    for h in range(model.n_skills):
        allowed = np.asarray(allowed_next(h, model.n_skills), dtype=int)
        alpha = model.eta_transition + counts[h, allowed]
        p_density += dirichlet.logpdf(state.transition[h, allowed], alpha)
    assert mfl.conditional_p_log_density(state.transition, state.segmentations, model) == pytest.approx(
        p_density)

    # Same seed, independently expanded equations: P first, then pi, matching Stage 6E.
    seed = 717
    expected_rng = np.random.default_rng(seed)
    expected_p = sample_transition_matrix(counts, model.n_skills, expected_rng,
                                          model.eta_transition)
    expected_pi = expected_rng.dirichlet(alpha_pi)
    observed = state.copy()
    mfl.gibbs_pi_p(observed, model, np.random.default_rng(seed))
    np.testing.assert_array_equal(observed.pi, expected_pi)
    np.testing.assert_array_equal(observed.transition, expected_p)
    mfl.validate_pi_p(observed, model)
    assert np.array_equal(np.diag(observed.transition), np.zeros(model.n_skills))
    np.testing.assert_allclose(observed.transition.sum(axis=1), 1.0, atol=1e-12)
    assert observed.pi.sum() == pytest.approx(1.0, abs=1e-12)


def test_markov_support_validator_rejects_positive_diagonal_before_ffbs():
    _, model, state, _ = setup_state()
    bad = state.copy()
    bad.transition[0, 0] = 0.1
    bad.transition[0, 1] -= 0.1
    with pytest.raises(AssertionError, match="zero diagonal"):
        mfl.validate_pi_p(bad, model)
    sampler = mfl.FullLatentSampler(
        model, mfl.FullLatentFixed(), mfl.FullLatentConfig(mfl.FULL_MARG))
    with pytest.raises(AssertionError, match="zero diagonal"):
        mfl.full_latent_sweep_once(bad, sampler, np.random.default_rng(3))


def test_ffbs_and_marginal_normalizer_use_current_learned_pi_and_p():
    fixed, model, state, _ = setup_state()
    sampler = mfl.FullLatentSampler(
        model, fixed, mfl.FullLatentConfig(mfl.FULL_MARG, structural_cadence=10))
    a = state.copy()
    mfl.validate_pi_p(a, model)
    sampler.tables.refresh(a)
    ffbs_a = mfl.ffbs_segmentation_draw(model, a, sampler.tables, np.random.default_rng(4))
    sampler.tables.mark_stale()
    # `CollapsedULikelihood` supplies a separate fast-table calculation of the same Z.
    marginal_a = sampler.collapsed_likelihood.log_z_per_trace(a)
    np.testing.assert_allclose(ffbs_a["log_normalizers"], marginal_a, atol=1e-9)

    b = state.copy()
    b.pi = np.array([0.98, 0.01, 0.01])
    b.transition = np.array([[0.0, 0.995, 0.005], [0.995, 0.0, 0.005],
                             [0.995, 0.005, 0.0]])
    mfl.validate_pi_p(b, model)
    marginal_b = sampler.collapsed_likelihood.log_z_per_trace(b)
    assert not np.allclose(marginal_a, marginal_b)
    sampler.tables.refresh(b)
    ffbs_b = mfl.ffbs_segmentation_draw(model, b, sampler.tables, np.random.default_rng(4))
    sampler.tables.mark_stale()
    np.testing.assert_allclose(ffbs_b["log_normalizers"], marginal_b, atol=1e-9)


def test_arms_match_proposal_schedule_and_differ_only_in_structural_score():
    fixed, model, state, _ = setup_state()
    cond = mfl.FullLatentSampler(
        model, fixed, mfl.FullLatentConfig(mfl.FULL_COND, structural_cadence=1,
                                            structural_scale=0.5))
    marg = mfl.FullLatentSampler(
        model, fixed, mfl.FullLatentConfig(mfl.FULL_MARG, structural_cadence=1,
                                            structural_scale=0.5))
    c_state, c_info = mfl.full_latent_sweep_once(state, cond, np.random.default_rng(93))
    m_state, m_info = mfl.full_latent_sweep_once(state, marg, np.random.default_rng(93))
    assert c_info["scheduled_structural"] and m_info["scheduled_structural"]
    assert c_info["structural_record"]["skill"] == m_info["structural_record"]["skill"]
    assert c_info["structural_record"]["row"] == m_info["structural_record"]["row"]
    assert cond.config.structural_cadence == marg.config.structural_cadence
    assert cond.config.structural_scale == marg.config.structural_scale
    assert c_info["kernel_order"] == ("conditional_U", "FFBS", "pi_P")
    assert m_info["kernel_order"] == ("marginal_U", "FFBS", "pi_P")
    for new_state in (c_state, m_state):
        fixed.assert_unchanged(new_state)
        mfl.validate_pi_p(new_state, model)


def test_marginal_attempt_is_refreshed_before_any_path_dependent_update():
    fixed, model, state, _ = setup_state()
    sampler = mfl.FullLatentSampler(
        model, fixed, mfl.FullLatentConfig(mfl.FULL_MARG, structural_cadence=1))
    seen_acceptance = set()
    for seed in range(12):
        _, info = mfl.full_latent_sweep_once(state, sampler, np.random.default_rng(seed))
        seen_acceptance.add(bool(info["structural_record"]["accepted"]))
        assert info["kernel_order"] == ("marginal_U", "FFBS", "pi_P")
    # This test is about ordering for both possible MH outcomes.  The small model normally
    # covers both; individual outcome coverage is not a mathematical precondition.
    assert seen_acceptance


def test_independent_target_and_all_permutation_invariant_summaries():
    fixed, model, state, probes = setup_state()
    sampler = mfl.FullLatentSampler(
        model, fixed, mfl.FullLatentConfig(mfl.FULL_COND, structural_cadence=1))
    state, _ = mfl.full_latent_sweep_once(state, sampler, np.random.default_rng(13))
    grouped, direct = mfl.complete_log_target(state, model), mfl.independent_complete_log_target(state, model)
    for name in grouped:
        assert grouped[name] == pytest.approx(direct[name], abs=1e-9)
    baseline = mfl.invariant_summaries(state, model, probes)
    for permutation in mfl.all_label_permutations(model.n_skills):
        permuted = mfl.permute_state_labels(state, permutation)
        mfl.validate_pi_p(permuted, model)
        assert mfl.complete_log_target(permuted, model)["log_target"] == pytest.approx(
            grouped["log_target"], abs=1e-9)
        summary = mfl.invariant_summaries(permuted, model, probes)
        assert summary.keys() == baseline.keys()
        for name in baseline:
            np.testing.assert_allclose(np.asarray(summary[name]), np.asarray(baseline[name]),
                                       atol=1e-10, rtol=1e-10, equal_nan=False)


def test_checkpoint_resume_preserves_pi_p_rng_paths_and_diagnostics(tmp_path):
    fixed, model, state, probes = setup_state()
    config = mfl.FullLatentConfig(mfl.FULL_MARG, structural_cadence=2)
    full_sampler = mfl.FullLatentSampler(model, fixed, config)
    full = mfl.FullLatentChain(full_sampler, state, seed=101, burn_in=2, thin=2,
                               probes=probes)
    full.advance(12)

    split_sampler = mfl.FullLatentSampler(model, fixed, config)
    split = mfl.FullLatentChain(split_sampler, state, seed=101, burn_in=2, thin=2,
                                probes=probes)
    checkpoint = tmp_path / "chain.npz"
    split.advance(6, checkpoint_path=checkpoint, checkpoint_every=3)
    resumed = mfl.FullLatentChain.load(
        checkpoint, mfl.FullLatentSampler(model, fixed, config))
    resumed.advance(12)

    assert full.state.to_dict() == resumed.state.to_dict()
    assert full.structural == resumed.structural
    assert full.movement == resumed.movement
    assert full.retained_draws == resumed.retained_draws
    for name in full.arrays():
        np.testing.assert_array_equal(full.arrays()[name], resumed.arrays()[name])
    np.testing.assert_array_equal(np.asarray(full.u_draws), np.asarray(resumed.u_draws))
    np.testing.assert_array_equal(np.asarray(full.pi_draws), np.asarray(resumed.pi_draws))
    np.testing.assert_array_equal(np.asarray(full.p_draws), np.asarray(resumed.p_draws))
    for a, b in zip(full.boundary_sums, resumed.boundary_sums):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(full.recovery_coskill_sums, resumed.recovery_coskill_sums)
    np.testing.assert_array_equal(full.recovery_same_segment_sums,
                                  resumed.recovery_same_segment_sums)


def test_schema_two_checkpoint_rejects_non_float64_retained_u(tmp_path):
    fixed, model, state, probes = setup_state()
    config = mfl.FullLatentConfig(mfl.FULL_COND, structural_cadence=1)
    chain = mfl.FullLatentChain(mfl.FullLatentSampler(model, fixed, config), state,
                                seed=91, burn_in=0, thin=1, probes=probes)
    checkpoint = tmp_path / "exact_u.npz"
    chain.advance(4, checkpoint_path=checkpoint, checkpoint_every=2)
    with np.load(checkpoint, allow_pickle=False) as data:
        payload = {name: np.array(data[name], copy=True) for name in data.files}
    payload["u_draws"] = payload["u_draws"].astype(np.float32)
    malformed = tmp_path / "lossy_u.npz"
    np.savez_compressed(malformed, **payload)
    with pytest.raises(ValueError, match="exact float64"):
        mfl.FullLatentChain.load(
            malformed, mfl.FullLatentSampler(model, fixed, config))


def test_retained_u_is_float64_and_preserves_near_tie_closures():
    fixed, model, state, probes = setup_state()
    # These two rows have a strict product-order relation in float64 but collapse to an
    # equal pair in float32.  Recovery aligns closures, so retaining float32 would turn a
    # valid posterior draw into a different structural draw.
    state.u_by_skill[0, 0] = [1.0 + 2e-10, 1.0 + 2e-10]
    state.u_by_skill[0, 1] = [1.0 + 1e-10, 1.0 + 1e-10]
    expected = precedence_from_u(state.u_by_skill[0])
    assert expected[0, 1]
    assert not precedence_from_u(state.u_by_skill[0].astype(np.float32))[0, 1]

    chain = mfl.FullLatentChain(
        mfl.FullLatentSampler(model, fixed, mfl.FullLatentConfig(mfl.FULL_COND)),
        state, seed=44, burn_in=0, thin=1, probes=probes,
    )
    chain._retain()
    retained = chain.u_draws[0]
    assert retained.dtype == np.float64
    np.testing.assert_array_equal(retained, state.u_by_skill)
    np.testing.assert_array_equal(precedence_from_u(retained[0]), expected)


def test_formal_import_does_not_load_sealed_generator_modules():
    root = Path(__file__).parents[2]
    code = "\n".join((
        "import importlib.util, json, sys",
        "from hpop.mcmc_original import matched_full_latent",  # noqa: F401
        "spec = importlib.util.spec_from_file_location('full_latent_runner_probe', "
        + repr(str(root / 'scripts' / 'run_matched_full_latent_formal.py')) + ")",
        "module = importlib.util.module_from_spec(spec)",
        "spec.loader.exec_module(module)",
        "forbidden = {'hpop.mcmc_original.recurrent_synthetic', "
        "'hpop.mcmc_original.matched_synthetic_generator', "
        "'hpop.mcmc_original.generate_matched_formal_corpus'}",
        "print(json.dumps(sorted(forbidden.intersection(sys.modules))))",
    ))
    environment = dict(os.environ)
    source_path = str(root / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=environment,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_truth_free_constants_match_the_registered_specialization():
    # The seam is deliberately literal and import-safe; this parity check prevents a
    # future manual edit from changing the finite-Markov model under the same experiment
    # registration.
    from hpop.mcmc_original.recurrent_scalar_posterior import TRUE_VALUES
    from hpop.mcmc_original.stage6b_frozen import EPSILON as legacy_epsilon
    from hpop.mcmc_original.stage6e_frozen import (
        DELTA_B as legacy_delta_b,
        ETA_INITIAL as legacy_eta_initial,
        ETA_TRANSITION as legacy_eta_transition,
        LATENT_DIM as legacy_latent_dim,
        MAX_BLOCK_WIDTH as legacy_max_width,
        MIN_BLOCK_WIDTH as legacy_min_width,
        N_ROLES as legacy_n_roles,
        N_SKILLS as legacy_n_skills,
    )

    assert (constants.EPSILON, constants.DELTA_B, constants.ETA_INITIAL,
            constants.ETA_TRANSITION, constants.MIN_BLOCK_WIDTH,
            constants.MAX_BLOCK_WIDTH, constants.N_ROLES, constants.N_SKILLS,
            constants.LATENT_DIM) == (
                legacy_epsilon, legacy_delta_b, legacy_eta_initial,
                legacy_eta_transition, legacy_min_width, legacy_max_width,
                legacy_n_roles, legacy_n_skills, legacy_latent_dim,
            )
    assert (constants.FIXED_BETA, constants.FIXED_OMEGA,
            constants.FIXED_LAMBDA_REP, constants.FIXED_LAMBDA_BACK) == (
                TRUE_VALUES["beta"], TRUE_VALUES["omega"],
                TRUE_VALUES["lambda_rep"], TRUE_VALUES["lambda_back"],
            )
    assert constants.FIXED_RHO_0 == 0.5  # registered Condition-B/C isolation value


def test_source_has_no_rescue_kernel_import():
    source = (Path(__file__).parents[2] / "src/hpop/mcmc_original/matched_full_latent.py").read_text()
    assert "skill_swap_kernel import" not in source
    assert "SkillSwapConfig" not in source
    assert "tempering" not in source.lower().replace("no swap, transposition, tempering", "")
