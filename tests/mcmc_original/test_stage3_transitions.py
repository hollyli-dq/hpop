"""Stage 3 — skill transitions: conjugacy, and resolving an ambiguous boundary."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from hpop.mcmc_original import toy
from hpop.mcmc_original.enumerate import build_trace_states, exact_state_table
from hpop.mcmc_original.latent_poset import incomparable, precedence_from_u
from hpop.mcmc_original.targets import SkillEvaluator
from hpop.mcmc_original.transitions import (
    allowed_next,
    dirichlet_posterior_params,
    log_transition_matrix,
    posterior_mean_transition_matrix,
    sample_transition_matrix,
    transition_counts,
)

SEED = 20260808
K = 3
N_DIRICHLET_DRAWS = 50_000

E_31_34 = 31.0 / 34.0   # 0.9117647058823529
E_3_34 = 3.0 / 34.0     # 0.0882352941176471


def manual_counts() -> np.ndarray:
    counts = np.zeros((K, K), dtype=float)
    counts[toy.SKILL_B, toy.SKILL_A] = 30
    counts[toy.SKILL_B, toy.SKILL_C] = 2
    counts[toy.SKILL_A, toy.SKILL_B] = 2
    counts[toy.SKILL_A, toy.SKILL_C] = 30
    return counts


@pytest.fixture(scope="module")
def stage3_states():
    skills = toy.stage3_skills()
    evaluators = [SkillEvaluator(s) for s in skills]
    trace_states = build_trace_states(toy.PRIMARY_TRACE, skills, evaluators, toy.DELTA_B)
    tables = [evaluators[k].log_table(skills[k].u) for k in range(K)]
    return {
        "skills": skills,
        "evaluators": evaluators,
        "trace_states": trace_states,
        "tables": tables,
        "log_pi": toy.uniform_log_pi(K),
    }


# ---------------------------------------------------------------------------
# the Stage-3 skill templates
# ---------------------------------------------------------------------------


def test_stage3_B_is_antichain():
    precedence = precedence_from_u(toy.U_B_ANTICHAIN)
    assert not precedence.any()
    for i, j in itertools.combinations(range(3), 2):
        assert incomparable(precedence, i, j)
    toy.assert_stage3_b_is_antichain()


def test_stage3_B_gives_every_permutation_one_sixth():
    b = toy.skill_b_antichain()
    for perm in itertools.permutations(range(3)):
        labels = tuple(b.cpa_labels[r] for r in perm)
        assert np.exp(toy.segment_log_likelihood(labels, b)) == pytest.approx(
            1.0 / 6.0, abs=1e-12
        )


def test_the_two_B_templates_are_not_confused():
    """U_B_TOTAL and U_B_ANTICHAIN must stay distinct objects with distinct orders."""
    assert not np.array_equal(toy.U_B_TOTAL, toy.U_B_ANTICHAIN)
    assert precedence_from_u(toy.U_B_TOTAL).sum() == 3
    assert precedence_from_u(toy.U_B_ANTICHAIN).sum() == 0
    assert precedence_from_u(toy.stage012_skills()[toy.SKILL_B].u).sum() == 3
    assert precedence_from_u(toy.stage3_skills()[toy.SKILL_B].u).sum() == 0


def test_stage3_skill_c_order():
    c = toy.skill_c()
    assert c.cpa_labels == (3, 4)
    assert precedence_from_u(c.u)[0, 1]


# ---------------------------------------------------------------------------
# Stage 3A — counts and conjugacy
# ---------------------------------------------------------------------------


def test_allowed_next_excludes_self():
    assert allowed_next(0, 3) == (1, 2)
    assert allowed_next(1, 3) == (0, 2)
    assert allowed_next(2, 3) == (0, 1)
    with pytest.raises(ValueError):
        allowed_next(3, 3)


def test_transition_counts():
    paths = [
        (toy.SKILL_B, toy.SKILL_A),
        (toy.SKILL_B, toy.SKILL_A),
        (toy.SKILL_A, toy.SKILL_C),
        (toy.SKILL_B, toy.SKILL_C),
        (toy.SKILL_A, toy.SKILL_B, toy.SKILL_C),
    ]
    counts = transition_counts(paths, K)
    assert counts[toy.SKILL_B, toy.SKILL_A] == 2
    assert counts[toy.SKILL_A, toy.SKILL_C] == 1
    assert counts[toy.SKILL_A, toy.SKILL_B] == 1
    # the 3-segment path A -> B -> C contributes a second B -> C
    assert counts[toy.SKILL_B, toy.SKILL_C] == 2
    assert counts.sum() == 6
    np.testing.assert_array_equal(np.diag(counts), np.zeros(K))


def test_transition_counts_rejects_self_transitions():
    with pytest.raises(ValueError, match="self-transition"):
        transition_counts([(0, 0)], K)


def test_transition_counts_ignores_single_segment_paths():
    assert transition_counts([(0,), (1,)], K).sum() == 0


def test_transition_dirichlet_parameters():
    params = dirichlet_posterior_params(manual_counts(), K)

    allowed_b, alpha_b = params[toy.SKILL_B]
    assert allowed_b == (toy.SKILL_A, toy.SKILL_C)
    np.testing.assert_array_equal(alpha_b, [31.0, 3.0])

    allowed_a, alpha_a = params[toy.SKILL_A]
    assert allowed_a == (toy.SKILL_B, toy.SKILL_C)
    np.testing.assert_array_equal(alpha_a, [3.0, 31.0])

    allowed_c, alpha_c = params[toy.SKILL_C]
    assert allowed_c == (toy.SKILL_A, toy.SKILL_B)
    np.testing.assert_array_equal(alpha_c, [1.0, 1.0])


def test_analytic_posterior_means():
    means = posterior_mean_transition_matrix(manual_counts(), K)
    assert means[toy.SKILL_B, toy.SKILL_A] == pytest.approx(E_31_34, abs=1e-15)
    assert means[toy.SKILL_B, toy.SKILL_C] == pytest.approx(E_3_34, abs=1e-15)
    assert means[toy.SKILL_A, toy.SKILL_B] == pytest.approx(E_3_34, abs=1e-15)
    assert means[toy.SKILL_A, toy.SKILL_C] == pytest.approx(E_31_34, abs=1e-15)
    assert means[toy.SKILL_C, toy.SKILL_A] == pytest.approx(0.5)
    np.testing.assert_array_equal(np.diag(means), np.zeros(K))
    for h in range(K):
        assert means[h].sum() == pytest.approx(1.0)


def test_transition_gibbs_means_match_analytic():
    rng = np.random.default_rng(SEED)
    counts = manual_counts()
    draws = np.array(
        [sample_transition_matrix(counts, K, rng) for _ in range(N_DIRICHLET_DRAWS)]
    )
    empirical = draws.mean(axis=0)
    analytic = posterior_mean_transition_matrix(counts, K)

    for h in range(K):
        for k in allowed_next(h, K):
            assert abs(empirical[h, k] - analytic[h, k]) < 0.01, (
                f"P[{h},{k}]: empirical {empirical[h,k]:.4f} vs "
                f"analytic {analytic[h,k]:.4f}"
            )
    np.testing.assert_array_equal(draws[:, np.arange(K), np.arange(K)], 0.0)


def test_gibbs_draws_are_valid_distributions():
    rng = np.random.default_rng(SEED + 1)
    for _ in range(200):
        p = sample_transition_matrix(manual_counts(), K, rng)
        assert np.all(p >= 0.0)
        for h in range(K):
            assert p[h].sum() == pytest.approx(1.0)
            assert p[h, h] == 0.0


def test_gibbs_variance_matches_the_dirichlet_variance():
    """Means alone would pass for a wrong-but-centred sampler; check spread too."""
    rng = np.random.default_rng(SEED + 2)
    counts = manual_counts()
    draws = np.array(
        [sample_transition_matrix(counts, K, rng)[toy.SKILL_B, toy.SKILL_A]
         for _ in range(N_DIRICHLET_DRAWS)]
    )
    a, b = 31.0, 3.0
    expected_var = a * b / ((a + b) ** 2 * (a + b + 1))
    assert draws.var() == pytest.approx(expected_var, rel=0.05)


# ---------------------------------------------------------------------------
# Stage 3B — transition context resolves the ambiguity
# ---------------------------------------------------------------------------


def test_ambiguous_trace_has_exactly_two_states(stage3_states):
    trace_states = stage3_states["trace_states"]
    assert trace_states.n_states == 2
    assert set(trace_states.paths) == {
        (toy.SKILL_A, toy.SKILL_B),
        (toy.SKILL_B, toy.SKILL_A),
    }


def test_ambiguous_trace_is_half_half_without_transitions(stage3_states):
    table = exact_state_table(
        stage3_states["trace_states"], stage3_states["tables"], stage3_states["log_pi"]
    )
    np.testing.assert_allclose(table["probs"], [0.5, 0.5], atol=1e-12)


def test_the_two_states_have_identical_emission_terms(stage3_states):
    """Why it is exactly 0.5/0.5: the local likelihoods are equal, not merely close."""
    trace_states, tables = stage3_states["trace_states"], stage3_states["tables"]
    emissions = [
        sum(float(tables[skill][idx]) for skill, idx in descriptor)
        for descriptor in trace_states.descriptors
    ]
    assert emissions[0] == pytest.approx(emissions[1], abs=1e-15)
    assert np.exp(emissions[0]) == pytest.approx((1.0 / 6.0) * 0.975, abs=1e-12)
    assert trace_states.log_boundary[0] == trace_states.log_boundary[1]


def test_transition_context_resolves_ambiguous_trace(stage3_states):
    trace_states = stage3_states["trace_states"]
    means = posterior_mean_transition_matrix(manual_counts(), K)
    table = exact_state_table(
        trace_states,
        stage3_states["tables"],
        stage3_states["log_pi"],
        log_transition_matrix(means),
    )
    by_path = {path: p for path, p in zip(trace_states.paths, table["probs"])}

    assert abs(by_path[(toy.SKILL_B, toy.SKILL_A)] - E_31_34) < 1e-12
    assert abs(by_path[(toy.SKILL_A, toy.SKILL_B)] - E_3_34) < 1e-12
    assert abs(sum(by_path.values()) - 1.0) < 1e-12


def test_transition_context_direction_can_be_reversed(stage3_states):
    """Flipping the counts flips the resolved boundary — it is the context doing it."""
    trace_states = stage3_states["trace_states"]
    counts = np.zeros((K, K), dtype=float)
    counts[toy.SKILL_A, toy.SKILL_B] = 30
    counts[toy.SKILL_A, toy.SKILL_C] = 2
    counts[toy.SKILL_B, toy.SKILL_A] = 2
    counts[toy.SKILL_B, toy.SKILL_C] = 30
    table = exact_state_table(
        trace_states,
        stage3_states["tables"],
        stage3_states["log_pi"],
        log_transition_matrix(posterior_mean_transition_matrix(counts, K)),
    )
    by_path = {path: p for path, p in zip(trace_states.paths, table["probs"])}
    assert by_path[(toy.SKILL_A, toy.SKILL_B)] == pytest.approx(E_31_34, abs=1e-12)


def test_total_order_B_would_not_be_ambiguous():
    """Contrast: with U_B_TOTAL the same trace is 0.983/0.017, not 0.5/0.5."""
    skills = toy.stage012_skills()
    evaluators = [SkillEvaluator(s) for s in skills]
    trace_states = build_trace_states(toy.PRIMARY_TRACE, skills, evaluators, toy.DELTA_B)
    tables = [evaluators[k].log_table(skills[k].u) for k in range(len(skills))]
    table = exact_state_table(trace_states, tables, toy.uniform_log_pi(len(skills)))
    assert max(table["probs"]) == pytest.approx(0.983050847457627, abs=1e-10)
