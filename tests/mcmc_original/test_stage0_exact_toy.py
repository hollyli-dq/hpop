"""Stage 0 — the exact toy posterior with fixed U.

Everything here is deterministic. If these values are wrong, nothing downstream
means anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_original import toy
from hpop.mcmc_original.enumerate import (
    boundary_marginals_from_probs,
    build_trace_states,
    enumerate_segmentations,
    exact_state_table,
)
from hpop.mcmc_original.targets import (
    SkillEvaluator,
    log_boundary_prior,
    log_path_prior,
    logsumexp,
    normalize_log_weights,
    sample_categorical_from_log_weights,
)

P_BA_EXACT = 0.983050847457627
P_AB_EXACT = 0.016949152542373


@pytest.fixture(scope="module")
def stage0():
    skills = toy.stage012_skills()
    evaluators = [SkillEvaluator(s) for s in skills]
    trace_states = build_trace_states(
        toy.PRIMARY_TRACE, skills, evaluators, toy.DELTA_B
    )
    u = {k: skills[k].u for k in range(len(skills))}
    tables = [evaluators[k].log_table(u[k]) for k in range(len(skills))]
    table = exact_state_table(trace_states, tables, toy.uniform_log_pi(len(skills)))
    return {
        "skills": skills,
        "evaluators": evaluators,
        "trace_states": trace_states,
        "table": table,
    }


def index_of_path(trace_states, path):
    matches = [i for i, p in enumerate(trace_states.paths) if p == path]
    assert len(matches) == 1, f"expected exactly one state with path {path}"
    return matches[0]


# ---------------------------------------------------------------------------
# the BPOP values the toy rests on
# ---------------------------------------------------------------------------


def test_reference_bpop_values():
    a, b = toy.stage012_skills()
    assert np.exp(toy.segment_log_likelihood((0, 1), a)) == pytest.approx(
        0.975, abs=1e-12
    )
    assert np.exp(toy.segment_log_likelihood((0, 1, 2), b)) == pytest.approx(
        0.9425, abs=1e-12
    )
    assert np.exp(toy.segment_log_likelihood((2, 0, 1), b)) == pytest.approx(
        0.01625, abs=1e-12
    )


def test_incompatible_block_is_minus_infinity():
    a, _ = toy.stage012_skills()
    assert toy.segment_log_likelihood((0, 2), a) == -np.inf   # wrong labels
    assert toy.segment_log_likelihood((0, 1, 2), a) == -np.inf  # wrong length
    assert toy.segment_log_likelihood((0, 0), a) == -np.inf   # repeated label


def test_map_cpa_block_to_roles():
    assert toy.map_cpa_block_to_roles((0, 1), (0, 1)) == (0, 1)
    assert toy.map_cpa_block_to_roles((1, 0), (0, 1)) == (1, 0)
    assert toy.map_cpa_block_to_roles((2, 0, 1), (0, 1, 2)) == (2, 0, 1)
    assert toy.map_cpa_block_to_roles((3, 4), (0, 1)) is None
    assert toy.map_cpa_block_to_roles((0, 0), (0, 1)) is None


# ---------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------


def test_only_two_support_compatible_states(stage0):
    trace_states = stage0["trace_states"]
    assert trace_states.n_states == 2
    assert set(trace_states.paths) == {(toy.SKILL_A, toy.SKILL_B), (toy.SKILL_B, toy.SKILL_A)}

    spans = {
        tuple((g.start, g.end, g.skill) for g in seg.segments)
        for seg in trace_states.segmentations
    }
    assert spans == {
        ((0, 2, toy.SKILL_A), (2, 5, toy.SKILL_B)),
        ((0, 3, toy.SKILL_B), (3, 5, toy.SKILL_A)),
    }


def test_enumeration_agrees_with_brute_force_over_all_labellings(stage0):
    """Independent check: build every cut set x labelling and filter by support."""
    skills = stage0["skills"]
    x = toy.PRIMARY_TRACE
    total = len(x)
    found = set()
    for mask in range(1 << (total - 1)):
        cuts = [t for t in range(1, total) if mask & (1 << (t - 1))]
        bounds = [0, *cuts, total]
        blocks = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        for labelling in np.ndindex(*([len(skills)] * len(blocks))):
            if all(
                toy.map_cpa_block_to_roles(x[a:b], skills[k].cpa_labels) is not None
                for (a, b), k in zip(blocks, labelling)
            ):
                found.add(tuple((a, b, k) for (a, b), k in zip(blocks, labelling)))
    assert len(found) == 2
    enumerated = {
        tuple((g.start, g.end, g.skill) for g in seg.segments)
        for seg in enumerate_segmentations(x, skills)
    }
    assert enumerated == found


def test_enumeration_is_independent_of_u(stage0):
    """The legal state list depends on label supports only, never on U."""
    skills = stage0["skills"]
    baseline = enumerate_segmentations(toy.PRIMARY_TRACE, skills)
    shuffled = toy.stage3_skills()[:1] + (toy.skill_b_antichain(),)
    assert enumerate_segmentations(toy.PRIMARY_TRACE, shuffled) == baseline


# ---------------------------------------------------------------------------
# the exact posterior
# ---------------------------------------------------------------------------


def test_exact_primary_toy_probabilities(stage0):
    trace_states, table = stage0["trace_states"], stage0["table"]
    i_ba = index_of_path(trace_states, (toy.SKILL_B, toy.SKILL_A))
    i_ab = index_of_path(trace_states, (toy.SKILL_A, toy.SKILL_B))

    assert abs(table["probs"][i_ba] - P_BA_EXACT) < 1e-10
    assert abs(table["probs"][i_ab] - P_AB_EXACT) < 1e-10
    assert abs(table["probs"].sum() - 1.0) < 1e-12


def test_exact_posterior_matches_the_hand_computed_emission_weights(stage0):
    """0.9425*0.975 vs 0.975*0.01625, with the shared prior factors cancelling."""
    w_ba = 0.9425 * 0.975
    w_ab = 0.975 * 0.01625
    assert w_ba == pytest.approx(0.9189375, abs=1e-12)
    assert w_ab == pytest.approx(0.01584375, abs=1e-12)
    assert w_ba / (w_ba + w_ab) == pytest.approx(P_BA_EXACT, abs=1e-12)

    trace_states, table = stage0["trace_states"], stage0["table"]
    i_ba = index_of_path(trace_states, (toy.SKILL_B, toy.SKILL_A))
    i_ab = index_of_path(trace_states, (toy.SKILL_A, toy.SKILL_B))
    ratio = np.exp(table["log_targets"][i_ba] - table["log_targets"][i_ab])
    assert ratio == pytest.approx(w_ba / w_ab, rel=1e-12)


def test_prior_factors_are_identical_for_the_two_states(stage0):
    """Both states have L=2, so boundary and label priors cancel exactly."""
    trace_states = stage0["trace_states"]
    assert trace_states.log_boundary[0] == trace_states.log_boundary[1]
    log_pi = toy.uniform_log_pi(2)
    priors = [log_path_prior(p, log_pi) for p in trace_states.paths]
    assert priors[0] == pytest.approx(priors[1])
    # delta_B = 0.5 makes the boundary prior 0.5^(T-1) regardless of L
    assert trace_states.log_boundary[0] == pytest.approx(4 * np.log(0.5))


def test_exact_primary_toy_boundary_marginals(stage0):
    trace_states, table = stage0["trace_states"], stage0["table"]
    marginals = boundary_marginals_from_probs(trace_states, table["probs"])
    assert marginals.shape == (len(toy.PRIMARY_TRACE) - 1,)
    # cut position t is stored at index t-1
    assert abs(marginals[3 - 1] - P_BA_EXACT) < 1e-10
    assert abs(marginals[2 - 1] - P_AB_EXACT) < 1e-10
    assert marginals[1 - 1] == 0.0
    assert marginals[4 - 1] == 0.0
    assert marginals.sum() == pytest.approx(1.0, abs=1e-12)


def test_boundary_prior_formula():
    assert log_boundary_prior(5, 2, 0.5) == pytest.approx(4 * np.log(0.5))
    assert log_boundary_prior(5, 1, 0.5) == pytest.approx(4 * np.log(0.5))
    assert log_boundary_prior(5, 2, 0.3) == pytest.approx(
        np.log(0.3) + 3 * np.log(0.7)
    )
    with pytest.raises(ValueError):
        log_boundary_prior(5, 2, 0.0)


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------


def test_logsumexp_is_stable_on_large_and_small_scores():
    big = np.array([1000.0, 1000.0])
    assert logsumexp(big) == pytest.approx(1000.0 + np.log(2.0))
    small = np.array([-1000.0, -1000.0])
    assert logsumexp(small) == pytest.approx(-1000.0 + np.log(2.0))
    assert logsumexp([-np.inf, -np.inf]) == -np.inf
    assert logsumexp([-np.inf, 0.0]) == pytest.approx(0.0)


def test_normalize_log_weights_sums_to_one():
    for scale in (0.0, 500.0, -500.0):
        p = normalize_log_weights(np.array([0.0, 1.0, -2.0]) + scale)
        assert p.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(p >= 0.0)
    with pytest.raises(ValueError):
        normalize_log_weights([-np.inf, -np.inf])


def test_categorical_sampler_respects_log_weights():
    rng = np.random.default_rng(0)
    log_w = np.log([0.2, 0.8])
    draws = [sample_categorical_from_log_weights(log_w, rng) for _ in range(20_000)]
    assert np.mean(draws) == pytest.approx(0.8, abs=0.02)


def test_impossible_state_is_excluded_not_crashing():
    """A -inf state normalises to probability 0 rather than raising."""
    p = normalize_log_weights([0.0, -np.inf])
    assert p[0] == pytest.approx(1.0)
    assert p[1] == 0.0
