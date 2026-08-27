"""Stage 4 — algorithm correctness of the local proposal kernel.

Stage 1 proved the *math* with a proposal that needs the global state list. These
tests prove the *algorithm* for the moves a real sampler would use: Split, Merge,
Shift and Relabel.

Three things are checked, in increasing strength:

1. the sampled proposal law matches the computed one (``q`` is what it claims);
2. the MH kernel satisfies detailed balance exactly, and has the exact posterior
   as its stationary distribution;
3. a 200,000-step run reproduces the exact posterior to TV < 0.02.

Plus the control that gives all of this teeth: dropping the Hastings correction
must visibly break stationarity.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from hpop.mcmc_original import toy
from hpop.mcmc_original.diagnostics import total_variation_distance
from hpop.mcmc_original.enumerate import build_trace_states, exact_state_table
from hpop.mcmc_original.proposals import (
    LocalMoveKernel,
    MoveType,
    compatible_skills,
    merge_moves,
    relabel_moves,
    run_local_mcmc,
    shift_moves,
    split_moves,
    transition_matrix,
)
from hpop.mcmc_original.targets import SkillEvaluator, log_target_segmentation
from hpop.mcmc_original.types import Segment, Segmentation

SEED = 20260808
N_ITERATIONS = 200_000
BURN_IN = 10_000
TV_TOLERANCE = 0.02


@pytest.fixture(scope="module")
def kernel_toy():
    skills = toy.stage4_skills()
    x = toy.S4_TRACE
    evaluators = [SkillEvaluator(s) for s in skills]
    trace_states = build_trace_states(x, skills, evaluators, toy.S4_DELTA_B)
    tables = [evaluators[k].log_table(skills[k].u) for k in range(len(skills))]
    log_pi = toy.uniform_log_pi(len(skills))
    exact = exact_state_table(trace_states, tables, log_pi)
    kernel = LocalMoveKernel(x=x, skills=skills)
    u_by_skill = {k: skills[k].u for k in range(len(skills))}

    def log_target(segmentation):
        return log_target_segmentation(
            x, segmentation, evaluators, u_by_skill, toy.S4_DELTA_B, log_pi
        )

    return {
        "x": x,
        "skills": skills,
        "states": list(trace_states.segmentations),
        "trace_states": trace_states,
        "exact": exact,
        "kernel": kernel,
        "log_target": log_target,
    }


# ---------------------------------------------------------------------------
# the toy itself is fit for purpose
# ---------------------------------------------------------------------------


def test_kernel_toy_has_the_expected_state_space(kernel_toy):
    assert len(kernel_toy["states"]) == 11
    lengths = {len(s.segments) for s in kernel_toy["states"]}
    assert lengths == {2, 3, 4}, "L must vary or Split/Merge are untested"


def test_kernel_toy_posterior_is_well_spread(kernel_toy):
    """A near-degenerate posterior would make the TV test pass for free."""
    probs = kernel_toy["exact"]["probs"]
    assert probs.sum() == pytest.approx(1.0, abs=1e-12)
    assert probs.min() > 0.05
    assert probs.max() < 0.30


def test_skills_a_and_d_share_a_support(kernel_toy):
    """Relabel is only a real move because two skills accept the same block."""
    skills = kernel_toy["skills"]
    assert skills[toy.S4_SKILL_A].cpa_labels == skills[toy.S4_SKILL_D].cpa_labels
    assert not np.array_equal(skills[toy.S4_SKILL_A].u, skills[toy.S4_SKILL_D].u)
    assert compatible_skills((0, 1), skills) == (toy.S4_SKILL_A, toy.S4_SKILL_D)


def test_skill_e_support_is_the_union_of_a_and_f(kernel_toy):
    """Split/Merge are only real moves because E's block splits into A|F."""
    skills = kernel_toy["skills"]
    assert set(skills[toy.S4_SKILL_E].cpa_labels) == set(
        skills[toy.S4_SKILL_A].cpa_labels
    ) | set(skills[toy.S4_SKILL_F].cpa_labels)
    assert compatible_skills((0, 1, 2, 3), skills) == (toy.S4_SKILL_E,)


def test_every_move_type_is_live_somewhere(kernel_toy):
    """A dead move would silently make its correctness untested."""
    kernel = kernel_toy["kernel"]
    for move in MoveType.ALL:
        total = sum(len(kernel.neighbours(s, move)) for s in kernel_toy["states"])
        assert total > 0, f"move type {move} is never available"


# ---------------------------------------------------------------------------
# the individual moves
# ---------------------------------------------------------------------------


def test_relabel_changes_only_the_skill(kernel_toy):
    x, skills = kernel_toy["x"], kernel_toy["skills"]
    for state in kernel_toy["states"]:
        for candidate in relabel_moves(x, state, skills):
            assert len(candidate.segments) == len(state.segments)
            assert [(g.start, g.end) for g in candidate.segments] == [
                (g.start, g.end) for g in state.segments
            ]
            differing = [
                i
                for i, (a, b) in enumerate(zip(state.segments, candidate.segments))
                if a.skill != b.skill
            ]
            assert len(differing) == 1


def test_split_increases_and_merge_decreases_segment_count(kernel_toy):
    x, skills = kernel_toy["x"], kernel_toy["skills"]
    for state in kernel_toy["states"]:
        for candidate in split_moves(x, state, skills):
            assert len(candidate.segments) == len(state.segments) + 1
        for candidate in merge_moves(x, state, skills):
            assert len(candidate.segments) == len(state.segments) - 1


def test_shift_preserves_segment_count_and_moves_a_boundary(kernel_toy):
    x, skills = kernel_toy["x"], kernel_toy["skills"]
    for state in kernel_toy["states"]:
        for candidate in shift_moves(x, state, skills):
            assert len(candidate.segments) == len(state.segments)
            before = [g.end for g in state.segments]
            after = [g.end for g in candidate.segments]
            assert before != after


def test_every_move_lands_on_a_legal_state(kernel_toy):
    """No move may produce a support-incompatible segmentation."""
    x, skills = kernel_toy["x"], kernel_toy["skills"]
    legal = set(kernel_toy["states"])
    for state in kernel_toy["states"]:
        for move in MoveType.ALL:
            for candidate in kernel_toy["kernel"].neighbours(state, move):
                assert candidate in legal, f"{move} produced an illegal state"
                assert candidate.length == len(x)


def test_moves_never_return_the_current_state(kernel_toy):
    for state in kernel_toy["states"]:
        for move in MoveType.ALL:
            assert state not in kernel_toy["kernel"].neighbours(state, move)


def test_split_and_merge_are_exact_inverses(kernel_toy):
    """S' in N_split(S) must imply S in N_merge(S'), and vice versa."""
    kernel = kernel_toy["kernel"]
    for state in kernel_toy["states"]:
        for candidate in kernel.neighbours(state, MoveType.SPLIT):
            assert state in kernel.neighbours(candidate, MoveType.MERGE)
        for candidate in kernel.neighbours(state, MoveType.MERGE):
            assert state in kernel.neighbours(candidate, MoveType.SPLIT)


def test_shift_and_relabel_are_their_own_inverses(kernel_toy):
    kernel = kernel_toy["kernel"]
    for state in kernel_toy["states"]:
        for move in (MoveType.SHIFT, MoveType.RELABEL):
            for candidate in kernel.neighbours(state, move):
                assert state in kernel.neighbours(candidate, move)


def test_move_types_produce_disjoint_neighbourhoods(kernel_toy):
    """Relabel/Shift keep L, Split/Merge change it, so the sets cannot overlap."""
    kernel = kernel_toy["kernel"]
    for state in kernel_toy["states"]:
        seen: Counter = Counter()
        for move in MoveType.ALL:
            seen.update(set(kernel.neighbours(state, move)))
        assert all(count == 1 for count in seen.values())


def test_max_shift_restricts_the_boundary_displacement(kernel_toy):
    x, skills = kernel_toy["x"], kernel_toy["skills"]
    for state in kernel_toy["states"]:
        restricted = shift_moves(x, state, skills, max_shift=1)
        unrestricted = shift_moves(x, state, skills, max_shift=None)
        assert set(restricted) <= set(unrestricted)
    # with role counts of 2 and 4, a one-position shift is never legal here
    assert all(
        not shift_moves(x, state, skills, max_shift=1) for state in kernel_toy["states"]
    )


# ---------------------------------------------------------------------------
# the proposal law
# ---------------------------------------------------------------------------


def test_proposal_distribution_is_a_distribution(kernel_toy):
    for state in kernel_toy["states"]:
        law = kernel_toy["kernel"].proposal_distribution(state)
        assert sum(law.values()) == pytest.approx(1.0, abs=1e-12)
        assert all(p >= 0.0 for p in law.values())


def test_sampled_proposals_match_the_computed_law(kernel_toy):
    """The classic MCMC bug: the q used in the ratio is not the q being sampled."""
    kernel = kernel_toy["kernel"]
    rng = np.random.default_rng(SEED)
    n_draws = 60_000
    for state in kernel_toy["states"]:
        law = kernel.proposal_distribution(state)
        counts: Counter = Counter()
        for _ in range(n_draws):
            candidate, _ = kernel.sample_proposal(state, rng)
            counts[candidate] += 1
        assert set(counts) <= set(law)
        for candidate, expected in law.items():
            observed = counts[candidate] / n_draws
            assert abs(observed - expected) < 0.01, (
                f"q mismatch: expected {expected:.4f}, sampled {observed:.4f}"
            )


def test_the_proposal_is_genuinely_asymmetric(kernel_toy):
    """If q were symmetric the Hastings correction would be untested."""
    kernel = kernel_toy["kernel"]
    worst = 0.0
    for state in kernel_toy["states"]:
        for candidate, forward in kernel.proposal_distribution(state).items():
            if candidate == state:
                continue
            worst = max(worst, abs(forward - kernel.proposal_prob(candidate, state)))
    assert worst > 0.1, f"proposal is nearly symmetric (max gap {worst:.4f})"


def test_proposal_support_is_reversible(kernel_toy):
    """q(S->S') > 0 must imply q(S'->S) > 0, or the move can never be undone."""
    kernel = kernel_toy["kernel"]
    for state in kernel_toy["states"]:
        for candidate, forward in kernel.proposal_distribution(state).items():
            if candidate == state or forward <= 0.0:
                continue
            assert kernel.proposal_prob(candidate, state) > 0.0


def test_move_probabilities_are_validated(kernel_toy):
    with pytest.raises(ValueError, match="sum to 1"):
        LocalMoveKernel(
            x=kernel_toy["x"],
            skills=kernel_toy["skills"],
            move_probabilities={m: 0.1 for m in MoveType.ALL},
        )


# ---------------------------------------------------------------------------
# detailed balance and stationarity — the core of Stage 4
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kernel_matrix(kernel_toy):
    return transition_matrix(
        kernel_toy["states"], kernel_toy["exact"]["log_targets"], kernel_toy["kernel"]
    )


def test_transition_matrix_is_stochastic(kernel_matrix):
    assert np.all(kernel_matrix >= -1e-15)
    np.testing.assert_allclose(kernel_matrix.sum(axis=1), 1.0, atol=1e-12)


def test_detailed_balance(kernel_toy, kernel_matrix):
    """pi(S) K(S,S') == pi(S') K(S',S) for every ordered pair."""
    pi = kernel_toy["exact"]["probs"]
    flow = pi[:, None] * kernel_matrix
    worst = np.abs(flow - flow.T).max()
    assert worst < 1e-12, f"detailed balance violated by {worst:.3e}"


def test_exact_posterior_is_stationary(kernel_toy, kernel_matrix):
    pi = kernel_toy["exact"]["probs"]
    assert np.abs(pi @ kernel_matrix - pi).max() < 1e-12


def test_stationary_distribution_equals_the_posterior(kernel_toy, kernel_matrix):
    """Independent check: the leading left eigenvector of K is the posterior."""
    values, vectors = np.linalg.eig(kernel_matrix.T)
    leading = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    leading = leading / leading.sum()
    assert total_variation_distance(leading, kernel_toy["exact"]["probs"]) < 1e-10


def test_kernel_is_irreducible(kernel_toy, kernel_matrix):
    """Local moves must connect the whole space, or the chain cannot target pi."""
    n = len(kernel_toy["states"])
    reachable = np.linalg.matrix_power(kernel_matrix + np.eye(n), n)
    assert np.all(reachable > 0)


def test_dropping_the_hastings_correction_breaks_stationarity(kernel_toy):
    """The control: this is the bug the whole stage exists to catch.

    A kernel that assumes q is symmetric still runs and still mixes, but its
    stationary distribution is not the posterior.
    """
    kernel = kernel_toy["kernel"]
    states = kernel_toy["states"]
    log_targets = kernel_toy["exact"]["log_targets"]
    index = {s: i for i, s in enumerate(states)}

    n = len(states)
    naive = np.zeros((n, n))
    for i, state in enumerate(states):
        for candidate, forward in kernel.proposal_distribution(state).items():
            if candidate == state or forward <= 0.0:
                continue
            j = index[candidate]
            log_alpha = log_targets[j] - log_targets[i]   # no q ratio
            naive[i, j] = forward * min(1.0, math.exp(min(0.0, log_alpha)))
        naive[i, i] = 1.0 - naive[i].sum()

    pi = kernel_toy["exact"]["probs"]
    flow = pi[:, None] * naive
    assert np.abs(flow - flow.T).max() > 1e-3, "control failed: naive kernel is balanced"

    values, vectors = np.linalg.eig(naive.T)
    leading = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    leading = leading / leading.sum()
    assert total_variation_distance(leading, pi) > 0.05, (
        "control failed: the naive kernel happens to target the right posterior, "
        "so this toy cannot detect a missing Hastings correction"
    )


# ---------------------------------------------------------------------------
# the long run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def long_run(kernel_toy):
    rng = np.random.default_rng(SEED)
    return run_local_mcmc(
        kernel_toy["states"][0],
        kernel_toy["log_target"],
        kernel_toy["kernel"],
        N_ITERATIONS,
        BURN_IN,
        rng,
    )


def test_local_kernel_reproduces_the_exact_posterior(kernel_toy, long_run):
    index = {s: i for i, s in enumerate(kernel_toy["states"])}
    counts = np.bincount(
        [index[s] for s in long_run["kept"]], minlength=len(kernel_toy["states"])
    )
    empirical = counts / counts.sum()
    exact = kernel_toy["exact"]["probs"]
    tv = total_variation_distance(empirical, exact)
    assert tv < TV_TOLERANCE, f"TV = {tv:.5f}"
    for i in range(len(exact)):
        assert abs(empirical[i] - exact[i]) < 0.02, (
            f"state {i}: empirical {empirical[i]:.5f} vs exact {exact[i]:.5f}"
        )


def test_all_states_are_visited(kernel_toy, long_run):
    assert len(set(long_run["kept"])) == len(kernel_toy["states"])


def test_every_move_type_is_proposed_and_accepted(long_run):
    for move in MoveType.ALL:
        assert long_run["proposed_by_move"][move] > 0, f"{move} never proposed"
        assert long_run["accepted_by_move"][move] > 0, f"{move} never accepted"


def test_run_is_reproducible(kernel_toy):
    a = run_local_mcmc(
        kernel_toy["states"][0], kernel_toy["log_target"], kernel_toy["kernel"],
        20_000, 1_000, np.random.default_rng(7),
    )
    b = run_local_mcmc(
        kernel_toy["states"][0], kernel_toy["log_target"], kernel_toy["kernel"],
        20_000, 1_000, np.random.default_rng(7),
    )
    assert a["kept"] == b["kept"]


def test_chain_started_anywhere_finds_the_same_posterior(kernel_toy):
    """Convergence must not depend on the starting state."""
    index = {s: i for i, s in enumerate(kernel_toy["states"])}
    exact = kernel_toy["exact"]["probs"]
    for start in (len(kernel_toy["states"]) - 1, 1):
        result = run_local_mcmc(
            kernel_toy["states"][start], kernel_toy["log_target"],
            kernel_toy["kernel"], 60_000, 5_000, np.random.default_rng(SEED + start),
        )
        counts = np.bincount(
            [index[s] for s in result["kept"]], minlength=len(exact)
        )
        assert total_variation_distance(counts / counts.sum(), exact) < TV_TOLERANCE


def test_kernel_matches_the_stage1_uniform_kernel_on_the_primary_toy():
    """Cross-check: on the Stage-0 trace the local kernel must agree with Stage 1."""
    skills = toy.stage012_skills()
    x = toy.PRIMARY_TRACE
    evaluators = [SkillEvaluator(s) for s in skills]
    trace_states = build_trace_states(x, skills, evaluators, toy.DELTA_B)
    tables = [evaluators[k].log_table(skills[k].u) for k in range(len(skills))]
    log_pi = toy.uniform_log_pi(len(skills))
    exact = exact_state_table(trace_states, tables, log_pi)
    u_by_skill = {k: skills[k].u for k in range(len(skills))}
    kernel = LocalMoveKernel(x=x, skills=skills)

    def log_target(segmentation):
        return log_target_segmentation(
            x, segmentation, evaluators, u_by_skill, toy.DELTA_B, log_pi
        )

    result = run_local_mcmc(
        trace_states.segmentations[0], log_target, kernel,
        100_000, 5_000, np.random.default_rng(SEED),
    )
    index = {s: i for i, s in enumerate(trace_states.segmentations)}
    counts = np.bincount([index[s] for s in result["kept"]], minlength=2)
    empirical = counts / counts.sum()
    assert total_variation_distance(empirical, exact["probs"]) < 0.01
    # on this trace the two states differ by a Shift, and nothing else applies
    assert result["proposed_by_move"][MoveType.SHIFT] > 0
    assert result["accepted_by_move"][MoveType.SHIFT] > 0
