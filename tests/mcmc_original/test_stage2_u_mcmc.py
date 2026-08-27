"""Stage 2A — MCMC over the latent U with the segmentation known.

Recovery is judged on the **induced precedence relations**, never on the raw U
coordinates: U is not identifiable, only h(U) is.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_original import toy
from hpop.mcmc_original.diagnostics import relation_posterior, rhat
from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.sampler_u import (
    dispersed_initial_u,
    log_u_prior,
    propose_row,
    run_u_mcmc,
    sigma_rho_matrix,
)
from hpop.mcmc_original.targets import SkillEvaluator

SEED = 20260808
SIGMA_U = 0.8            # calibrated once; see the Stage-2A report section
N_ITERATIONS = 15_000
BURN_IN = 3_000
THIN = 3
N_CHAINS = 4


@pytest.fixture(scope="module")
def stage2a():
    rng = np.random.default_rng(SEED)
    corpus = toy.make_stage2a_corpus(rng)
    skills = toy.stage012_skills()
    out = {}
    for skill_id in (toy.SKILL_A, toy.SKILL_B):
        evaluator = SkillEvaluator(skills[skill_id])
        counts = evaluator.count_sequences(corpus[skill_id])
        chains = []
        for chain in range(N_CHAINS):
            chain_rng = np.random.default_rng(SEED + 100 * (skill_id + 1) + chain)
            init = dispersed_initial_u(
                skills[skill_id].u.shape, toy.RHO_U, chain_rng
            )
            chains.append(
                run_u_mcmc(
                    evaluator, counts, toy.RHO_U, SIGMA_U,
                    N_ITERATIONS, BURN_IN, THIN, chain_rng, init,
                )
            )
        out[skill_id] = {"evaluator": evaluator, "counts": counts, "chains": chains}
    out["corpus"] = corpus
    out["skills"] = skills
    return out


def pooled_relation_posterior(chains):
    return relation_posterior(np.concatenate([c["samples"] for c in chains]))


# ---------------------------------------------------------------------------
# prior
# ---------------------------------------------------------------------------


def test_sigma_rho_is_positive_definite():
    for d in (1, 2, 3, 5):
        sigma = sigma_rho_matrix(d, toy.RHO_U)
        assert sigma.shape == (d, d)
        assert np.linalg.eigvalsh(sigma).min() > 0
        np.testing.assert_allclose(np.diag(sigma), np.ones(d))
    with pytest.raises(ValueError):
        sigma_rho_matrix(2, 1.0)
    with pytest.raises(ValueError):
        sigma_rho_matrix(3, -0.9)


def test_u_prior_is_finite():
    rng = np.random.default_rng(0)
    for shape in ((2, 2), (3, 2), (5, 3)):
        for _ in range(50):
            value = log_u_prior(rng.normal(size=shape) * 3.0, toy.RHO_U)
            assert np.isfinite(value)
    assert log_u_prior(np.array([[np.nan, 0.0], [0.0, 0.0]]), toy.RHO_U) == -np.inf
    assert log_u_prior(np.array([[np.inf, 0.0], [0.0, 0.0]]), toy.RHO_U) == -np.inf


def test_u_prior_matches_scipy_free_reference():
    """Compare against an explicit MVN density built from the inverse and det."""
    rng = np.random.default_rng(1)
    u = rng.normal(size=(3, 2))
    d, rho = 2, toy.RHO_U
    sigma = sigma_rho_matrix(d, rho)
    inverse = np.linalg.inv(sigma)
    _, log_det = np.linalg.slogdet(sigma)
    expected = sum(
        -0.5 * (d * np.log(2 * np.pi) + log_det + row @ inverse @ row) for row in u
    )
    assert log_u_prior(u, rho) == pytest.approx(expected, rel=1e-12)


def test_u_prior_is_maximised_at_the_origin():
    assert log_u_prior(np.zeros((3, 2)), toy.RHO_U) > log_u_prior(
        np.ones((3, 2)), toy.RHO_U
    )


# ---------------------------------------------------------------------------
# proposal
# ---------------------------------------------------------------------------


def test_u_row_proposal_is_symmetric():
    """A Gaussian random walk on one row: the forward and reverse kernels agree."""
    rng = np.random.default_rng(2)
    u = rng.normal(size=(3, 2))
    sigma_u = 0.7
    for row in range(3):
        candidate = propose_row(u, row, sigma_u, rng)
        # only the proposed row moved
        for other in range(3):
            if other != row:
                np.testing.assert_array_equal(candidate[other], u[other])
        forward = candidate[row] - u[row]
        reverse = u[row] - candidate[row]
        # the RW density depends only on the squared step length
        assert np.dot(forward, forward) == pytest.approx(np.dot(reverse, reverse))


def test_proposal_does_not_mutate_the_input():
    rng = np.random.default_rng(3)
    u = np.zeros((2, 2))
    propose_row(u, 0, 0.5, rng)
    np.testing.assert_array_equal(u, np.zeros((2, 2)))


def test_proposal_rejects_bad_arguments():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError):
        propose_row(np.zeros((2, 2)), 0, 0.0, rng)
    with pytest.raises(ValueError):
        propose_row(np.zeros((2, 2)), 5, 0.5, rng)


# ---------------------------------------------------------------------------
# the count aggregation is exact
# ---------------------------------------------------------------------------


def test_permutation_count_aggregation_is_exact():
    """Aggregating identical executions must equal the per-instance sum exactly."""
    from hpop.mcmc_original.static_bpop import bpop_log_likelihood

    rng = np.random.default_rng(5)
    skills = toy.stage012_skills()
    for skill_id in (toy.SKILL_A, toy.SKILL_B):
        template = skills[skill_id]
        evaluator = SkillEvaluator(template)
        sequences = [
            toy.sample_role_execution(template, rng) for _ in range(150)
        ]
        counts = evaluator.count_sequences(sequences)
        assert counts.sum() == len(sequences)

        for u in (template.u, rng.normal(size=template.u.shape)):
            direct = sum(
                bpop_log_likelihood(s, u, template.beta, template.epsilon)
                for s in sequences
            )
            assert evaluator.counts_log_likelihood(u, counts) == pytest.approx(
                direct, rel=1e-12
            )


def test_evaluator_cache_returns_identical_tables():
    """The precedence-keyed cache must not change any value it returns."""
    template = toy.skill_b_total()
    evaluator = SkillEvaluator(template)
    first = evaluator.log_table(template.u).copy()
    jittered = template.u + np.array([[0.01, -0.01], [0.0, 0.02], [-0.01, 0.0]])
    assert np.array_equal(
        precedence_from_u(jittered), precedence_from_u(template.u)
    )
    np.testing.assert_array_equal(evaluator.log_table(jittered), first)
    assert evaluator.cache_hits >= 1


# ---------------------------------------------------------------------------
# recovery of the induced relations
# ---------------------------------------------------------------------------


def test_known_segmentation_u_mcmc_recovers_A_relation(stage2a):
    pooled = pooled_relation_posterior(stage2a[toy.SKILL_A]["chains"])
    assert pooled[0, 1] > 0.90, f"P(0 > 1) = {pooled[0,1]:.4f}"
    assert pooled[1, 0] < 0.10, f"P(1 > 0) = {pooled[1,0]:.4f}"


def test_known_segmentation_u_mcmc_recovers_B_relations(stage2a):
    pooled = pooled_relation_posterior(stage2a[toy.SKILL_B]["chains"])
    for i, j in ((0, 1), (1, 2), (0, 2)):
        assert pooled[i, j] > 0.90, f"P({i} > {j}) = {pooled[i,j]:.4f}"
    for i, j in ((1, 0), (2, 1), (2, 0)):
        assert pooled[i, j] < 0.10, f"P({i} > {j}) = {pooled[i,j]:.4f}"


def test_every_chain_recovers_the_relations_independently(stage2a):
    """Report per chain, so one stuck chain cannot hide inside a pooled average."""
    for skill_id, true_relations in (
        (toy.SKILL_A, [(0, 1)]),
        (toy.SKILL_B, [(0, 1), (1, 2), (0, 2)]),
    ):
        for c, chain in enumerate(stage2a[skill_id]["chains"]):
            posterior = relation_posterior(chain["samples"])
            for i, j in true_relations:
                assert posterior[i, j] > 0.90, (
                    f"skill {skill_id} chain {c}: P({i} > {j}) = {posterior[i,j]:.4f}"
                )


def test_acceptance_rates_are_in_the_target_band(stage2a):
    for skill_id in (toy.SKILL_A, toy.SKILL_B):
        for c, chain in enumerate(stage2a[skill_id]["chains"]):
            rate = chain["acceptance_rate"]
            assert 0.10 < rate < 0.60, f"skill {skill_id} chain {c}: acceptance {rate:.3f}"


def test_log_posterior_rhat_is_close_to_one(stage2a):
    for skill_id in (toy.SKILL_A, toy.SKILL_B):
        chains = np.array(
            [c["log_posterior_kept"] for c in stage2a[skill_id]["chains"]]
        )
        value = rhat(chains)
        assert value < 1.05, f"skill {skill_id}: R-hat(log posterior) = {value:.4f}"


def test_saved_sample_count_matches_the_thinning_plan(stage2a):
    expected = len(range(0, N_ITERATIONS - BURN_IN, THIN))
    for skill_id in (toy.SKILL_A, toy.SKILL_B):
        for chain in stage2a[skill_id]["chains"]:
            assert len(chain["samples"]) == expected


def test_corpus_is_dominated_by_the_true_linear_extension(stage2a):
    """Sanity on the generator itself before trusting recovery."""
    evaluator = stage2a[toy.SKILL_B]["evaluator"]
    counts = stage2a[toy.SKILL_B]["counts"]
    true_index = evaluator.perm_index[(0, 1, 2)]
    assert counts[true_index] / counts.sum() > 0.85
