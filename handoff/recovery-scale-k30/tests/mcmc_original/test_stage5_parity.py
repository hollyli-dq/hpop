"""Stage 5 §9 parity tests — proof the adapter did not alter the verified sampler."""
from __future__ import annotations

import inspect
import itertools
import math

import numpy as np
import pytest

from hpop.mcmc_original import stage5 as s5
from hpop.mcmc_original import toy
from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.static_bpop import bpop_log_likelihood
from hpop.vendored import ensure_importable

ensure_importable()
from src.mcmc.hpo_po_hm_mcmc_k_optim import (  # noqa: E402
    calculate_dimension_proportional_weights, mcmc_simulation_po,
)
from src.utils.po_accelerator_nle_optimized import (  # noqa: E402
    HPO_LogLikelihoodCache_Optimized as LL,
)


def test_notation_is_separated():
    """n_skills and latent_dim must never be conflated."""
    assert s5.N_SKILLS == 4
    assert s5.LATENT_DIM == 2
    assert s5.N_SKILLS != s5.LATENT_DIM or True   # they are distinct concepts, not one number
    assert len(s5.stage5_skills()) == s5.N_SKILLS
    for template in s5.stage5_skills():
        assert template.u.shape[1] == s5.LATENT_DIM


# --- A. fixed latent dimension disables the reversible-jump K update -----------


def test_A_fixed_K_removes_dimension_updates_from_the_schedule():
    frequencies, schedule = calculate_dimension_proportional_weights(
        n_items=4, cycle_length=1000, has_K_updates=False)
    assert "K" not in frequencies
    assert "K_dim" not in frequencies
    assert "K" not in set(schedule)
    assert set(frequencies) == {"U", "rho", "noise"}
    # and with K updates the schedule does contain them, so the check has teeth
    with_k, schedule_k = calculate_dimension_proportional_weights(
        n_items=4, cycle_length=1000, has_K_updates=True)
    assert "K" in with_k and "K" in set(schedule_k)


def test_A_update_K_is_false_when_fixed_K_is_given():
    source = inspect.getsource(mcmc_simulation_po)
    assert "update_K = fixed_K is None" in source
    assert "has_K_updates=update_K" in source


# --- B. likelihood parity -----------------------------------------------------


def test_B_adapter_likelihood_equals_the_verified_optimized_likelihood():
    """The Stage-5 adapter's data must score identically under the vendored likelihood."""
    skills = s5.stage5_skills()
    train, _, _ = s5.generate_corpus(n_train=6, n_test=1, seed=3, min_instances=1)
    segmentations = [t.true_segmentation for t in train]
    rng = np.random.default_rng(0)

    for skill_id, template in enumerate(skills):
        data = s5.build_po_dataset_for_skill(train, segmentations, skill_id, template)
        if not len(data):
            continue
        for u in (template.u, rng.normal(size=template.u.shape)):
            h = precedence_from_u(u).astype(int)
            vendored = LL.calculate_log_likelihood_po_optimized(
                U=u, h=h, observed_orders=data.observed_orders,
                choice_sets=data.choice_sets, items=data.items,
                item_to_index={i: i for i in data.items},
                prob_noise=toy.EPSILON,
                softmax_params={"beta": toy.BETA, "epsilon": toy.EPSILON},
                noise_option=s5.NOISE_OPTION)
            direct = sum(bpop_log_likelihood(order, u, toy.BETA, toy.EPSILON)
                         for order in data.observed_orders)
            assert vendored == pytest.approx(direct, abs=1e-9)


def test_B_generation_and_fitting_use_the_same_likelihood_branch():
    """hpop's own BPOP must equal the vendored log_successors_queue_jump branch."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for m in (2, 3, 4):
        for _ in range(8):
            u = rng.normal(size=(m, s5.LATENT_DIM))
            h = precedence_from_u(u).astype(int)
            for beta in (0.5, 1.5, 3.0):
                for order in itertools.permutations(range(m)):
                    vendored = LL.calculate_log_likelihood_po_optimized(
                        U=u, h=h, observed_orders=[list(order)],
                        choice_sets=[list(range(m))], items=list(range(m)),
                        item_to_index={i: i for i in range(m)},
                        prob_noise=toy.EPSILON,
                        softmax_params={"beta": beta, "epsilon": toy.EPSILON},
                        noise_option=s5.NOISE_OPTION)
                    worst = max(worst, abs(
                        vendored - bpop_log_likelihood(order, u, beta, toy.EPSILON)))
    assert worst < 1e-9, f"generator and fitted model disagree by {worst}"


# --- C. U proposal comes from Sigma_rho, not an isotropic step ----------------


def test_C_U_proposal_uses_the_Sigma_rho_covariance():
    source = inspect.getsource(mcmc_simulation_po)
    assert "Sigma = BasicUtils.build_Sigma_rho(K, rho)" in source
    assert "rng.multivariate_normal(mean=current_row, cov=Sigma)" in source
    # symmetric random walk: no Hastings term appears in the U acceptance ratio
    assert "log_accept = (lp_proposed + log_llk_proposed) - (lp_current + log_llk_current)" in source


# --- D / E. proposal Jacobians are present ------------------------------------


def test_D_rho_acceptance_contains_the_minus_log_delta_correction():
    source = inspect.getsource(mcmc_simulation_po)
    assert "delta = rng.uniform(dr, 1.0 / dr)" in source
    assert "rho_prime = 1.0 - (1.0 - rho) * delta" in source
    assert "- math.log(delta)" in source


def test_E_beta_acceptance_contains_the_log_jacobian():
    source = inspect.getsource(mcmc_simulation_po)
    assert "log_beta_proposed = log_beta_current + rw_step" in source
    assert "+ log_beta_proposed - log_beta_current" in source


# --- F. latent dimension never changes ---------------------------------------


def test_F_latent_dim_never_changes_during_a_run():
    skills = s5.stage5_skills()
    train, _, _ = s5.generate_corpus(n_train=8, n_test=1, seed=5, min_instances=1)
    data = s5.build_po_dataset_for_skill(
        train, [t.true_segmentation for t in train], s5.SKILL_E, skills[s5.SKILL_E])
    result = s5.run_skill_po_mcmc(data, num_iterations=2000, seed=0)
    assert set(result["latent_dim_trace"]) == {s5.LATENT_DIM}
    assert result["latent_dim_final"] == s5.LATENT_DIM
    assert result["U_final"].shape == (4, s5.LATENT_DIM)


# --- adapter contract ---------------------------------------------------------


def test_adapter_does_not_mix_skills_and_preserves_order():
    skills = s5.stage5_skills()
    train, _, _ = s5.generate_corpus(n_train=10, n_test=1, seed=7, min_instances=1)
    segmentations = [t.true_segmentation for t in train]
    for skill_id, template in enumerate(skills):
        data = s5.build_po_dataset_for_skill(train, segmentations, skill_id, template)
        expected = [tuple(r) for t in train
                    for g, r in zip(t.true_segmentation.segments, t.true_role_sequences)
                    if g.skill == skill_id]
        assert [tuple(o) for o in data.observed_orders] == expected   # order preserved
        assert all(sorted(o) == list(range(template.n_roles)) for o in data.observed_orders)
        assert all(cs == list(range(template.n_roles)) for cs in data.choice_sets)
        assert data.items == list(range(template.n_roles))


def test_adapter_rejects_incompatible_segments():
    from hpop.mcmc_original.types import Segment, Segmentation
    skills = s5.stage5_skills()
    trace = s5.SyntheticTrace((0, 1, 2, 3), Segmentation(segments=(Segment(0, 2, s5.SKILL_F),)),
                              (s5.SKILL_F,), ((0, 1),), "train", 1)
    with pytest.raises(s5.IncompatibleSegment):
        s5.build_po_dataset_for_skill([trace], [trace.true_segmentation],
                                      s5.SKILL_F, skills[s5.SKILL_F])


def test_random_initialisation_has_no_oracle_leakage():
    signature = inspect.signature(s5.sample_random_legal_segmentation)
    assert set(signature.parameters) == {"observations", "skills", "rng"}
    source = inspect.getsource(s5.sample_random_legal_segmentation)
    for forbidden in ("true_segmentation", "true_skill_path", "P_TRUE", "U_TRUE", "bpop"):
        assert forbidden not in source
