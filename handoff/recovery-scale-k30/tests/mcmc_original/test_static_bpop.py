"""Correctness tests for the static BPOP frontier-softmax likelihood.

The central claim: for any U, beta >= 0 and 0 <= epsilon < 1, the likelihood over
complete executions is a proper distribution — the m! permutation probabilities
sum to exactly 1.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from hpop.mcmc_original.latent_poset import incomparable, precedence_from_u
from hpop.mcmc_original.static_bpop import (
    all_permutation_probabilities,
    bpop_likelihood,
    bpop_log_likelihood,
    bpop_step_probabilities,
    frontier,
    remaining_successor_count,
    sample_bpop_sequence,
    successor_utility,
)

SEED = 20260808

# 0 > 2, 1 > 2, and 0 incomparable to 1
U_FORK = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
# 0 > 1 > 2 > 3
U_CHAIN = np.array([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
# pairwise incomparable
U_ANTICHAIN = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

BETAS = (0.0, 0.5, 1.5, 5.0)
EPSILONS = (0.0, 0.01, 0.1)
M_GRID = (2, 3, 4, 5)
D_GRID = (1, 2, 3)
DRAWS = 20


# ---------------------------------------------------------------------------
# the reference posets are what we think they are
# ---------------------------------------------------------------------------


def test_fork_poset_structure():
    p = precedence_from_u(U_FORK)
    assert p[0, 2] and p[1, 2]
    assert incomparable(p, 0, 1)
    assert not p[2, 0] and not p[2, 1]


def test_chain_poset_structure():
    p = precedence_from_u(U_CHAIN)
    for i, j in itertools.combinations(range(4), 2):
        assert p[i, j], f"expected {i} > {j}"
        assert not p[j, i]


def test_antichain_poset_structure():
    """Verify the dominance structure explicitly before relying on it."""
    p = precedence_from_u(U_ANTICHAIN)
    assert not p.any(), "U_ANTICHAIN must induce no precedence at all"
    for i, j in itertools.combinations(range(3), 2):
        assert incomparable(p, i, j)


# ---------------------------------------------------------------------------
# Task 5 — frontier
# ---------------------------------------------------------------------------


def test_frontier_fork():
    p = precedence_from_u(U_FORK)
    assert frontier([0, 1, 2], p) == (0, 1)
    assert frontier([1, 2], p) == (1,)
    assert frontier([0, 2], p) == (0,)
    assert frontier([2], p) == (2,)


def test_frontier_chain_unrolls_in_order():
    p = precedence_from_u(U_CHAIN)
    assert frontier([0, 1, 2, 3], p) == (0,)
    assert frontier([1, 2, 3], p) == (1,)
    assert frontier([2, 3], p) == (2,)
    assert frontier([3], p) == (3,)


def test_frontier_antichain_contains_everything():
    p = precedence_from_u(U_ANTICHAIN)
    assert frontier([0, 1, 2], p) == (0, 1, 2)
    assert frontier([1, 2], p) == (1, 2)


def test_frontier_is_sorted_regardless_of_input_order():
    p = precedence_from_u(U_ANTICHAIN)
    assert frontier([2, 0, 1], p) == (0, 1, 2)


def test_frontier_of_empty_remaining_is_empty():
    assert frontier([], precedence_from_u(U_FORK)) == ()


def test_frontier_is_never_empty_on_random_posets():
    rng = np.random.default_rng(SEED)
    for m in M_GRID:
        for d in D_GRID:
            for _ in range(DRAWS):
                p = precedence_from_u(rng.normal(size=(m, d)))
                for size in range(1, m + 1):
                    for rem in itertools.combinations(range(m), size):
                        assert frontier(rem, p), f"empty frontier for {rem}"


def test_frontier_raises_on_a_cyclic_relation():
    """A hand-made cyclic matrix is not a valid strict order and must be caught."""
    bad = np.array([[False, True], [True, False]])
    with pytest.raises(RuntimeError, match="empty frontier"):
        frontier([0, 1], bad)


@pytest.mark.parametrize(
    "bad_remaining", [[0, 0], [3], [-1], [1.5]]
)
def test_frontier_rejects_malformed_remaining(bad_remaining):
    with pytest.raises(ValueError):
        frontier(bad_remaining, precedence_from_u(U_FORK))


# ---------------------------------------------------------------------------
# Task 6 — successor count and utility
# ---------------------------------------------------------------------------


def test_fork_successor_counts_initially():
    p = precedence_from_u(U_FORK)
    assert remaining_successor_count(0, [0, 1, 2], p) == 1
    assert remaining_successor_count(1, [0, 1, 2], p) == 1
    assert remaining_successor_count(2, [0, 1, 2], p) == 0
    assert successor_utility(0, [0, 1, 2], p) == pytest.approx(math.log(2))
    assert successor_utility(1, [0, 1, 2], p) == pytest.approx(math.log(2))
    assert successor_utility(2, [0, 1, 2], p) == pytest.approx(0.0)


def test_fork_successor_counts_after_executing_zero():
    p = precedence_from_u(U_FORK)
    assert remaining_successor_count(1, [1, 2], p) == 1
    assert successor_utility(1, [1, 2], p) == pytest.approx(math.log(2))
    assert remaining_successor_count(2, [1, 2], p) == 0
    assert successor_utility(2, [1, 2], p) == pytest.approx(0.0)


def test_chain_successor_counts_use_the_full_closure():
    """S(0)=3 counts every dominated role, not just the covered one."""
    p = precedence_from_u(U_CHAIN)
    rem = [0, 1, 2, 3]
    assert [remaining_successor_count(x, rem, p) for x in rem] == [3, 2, 1, 0]
    assert successor_utility(0, rem, p) == pytest.approx(math.log(4))


def test_successor_count_is_restricted_to_remaining():
    p = precedence_from_u(U_CHAIN)
    assert remaining_successor_count(0, [0, 1, 2, 3], p) == 3
    assert remaining_successor_count(0, [0, 3], p) == 1
    assert remaining_successor_count(0, [0], p) == 0


def test_successor_count_requires_x_to_be_remaining():
    p = precedence_from_u(U_CHAIN)
    with pytest.raises(ValueError, match="must belong to the remaining set"):
        remaining_successor_count(0, [1, 2], p)


# ---------------------------------------------------------------------------
# Task 7 — one-step normalisation
# ---------------------------------------------------------------------------


def test_step_probabilities_normalise_over_random_models():
    rng = np.random.default_rng(SEED + 1)
    checked = 0
    for m in M_GRID:
        for d in D_GRID:
            for _ in range(DRAWS):
                u = rng.normal(size=(m, d))
                prefixes = [tuple(range(m))]
                order = list(rng.permutation(m))
                for cut in range(1, m):
                    prefixes.append(tuple(sorted(order[cut:])))
                for beta in BETAS:
                    for eps in EPSILONS:
                        for rem in prefixes:
                            if not rem:
                                continue
                            probs = bpop_step_probabilities(rem, u, beta, eps)
                            assert probs.shape == (m,)
                            assert probs.sum() == pytest.approx(1.0, abs=1e-12)
                            executed = set(range(m)) - set(rem)
                            for x in executed:
                                assert probs[x] == 0.0, "executed role must be exactly 0"
                            for x in rem:
                                assert probs[x] >= 0.0
                                if eps > 0.0:
                                    assert probs[x] > 0.0
                            checked += 1
    assert checked > 1000


def test_step_probabilities_reject_bad_parameters():
    with pytest.raises(ValueError, match="beta"):
        bpop_step_probabilities([0, 1, 2], U_FORK, -1.0, 0.05)
    with pytest.raises(ValueError, match="epsilon"):
        bpop_step_probabilities([0, 1, 2], U_FORK, 1.5, 1.0)
    with pytest.raises(ValueError, match="non-empty"):
        bpop_step_probabilities([], U_FORK, 1.5, 0.05)


# ---------------------------------------------------------------------------
# Task 8 — special cases
# ---------------------------------------------------------------------------


def test_a_epsilon_zero_gives_illegal_move_probability_zero():
    probs = bpop_step_probabilities([0, 1, 2], U_FORK, beta=1.5, epsilon=0.0)
    assert probs[2] == 0.0
    assert probs[0] + probs[1] == pytest.approx(1.0)


def test_b_epsilon_positive_gives_illegal_move_epsilon_over_three():
    eps = 0.09
    probs = bpop_step_probabilities([0, 1, 2], U_FORK, beta=1.5, epsilon=eps)
    assert probs[2] == pytest.approx(eps / 3.0, abs=1e-15)


def test_c_beta_zero_makes_the_frontier_uniform():
    p = precedence_from_u(U_ANTICHAIN)
    probs = bpop_step_probabilities([0, 1, 2], U_ANTICHAIN, beta=0.0, epsilon=0.0)
    assert frontier([0, 1, 2], p) == (0, 1, 2)
    np.testing.assert_allclose(probs, np.full(3, 1.0 / 3.0))

    # a frontier of size 2 inside a larger model
    u = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    probs = bpop_step_probabilities([0, 1, 2], u, beta=0.0, epsilon=0.0)
    assert probs[0] == pytest.approx(0.5)
    assert probs[1] == pytest.approx(0.5)


def test_d_total_order_with_epsilon_zero_has_one_legal_execution():
    probs = all_permutation_probabilities(U_CHAIN, beta=1.5, epsilon=0.0)
    assert probs[(0, 1, 2, 3)] == pytest.approx(1.0, abs=1e-12)
    for perm, p in probs.items():
        if perm != (0, 1, 2, 3):
            assert p == 0.0


def test_e_fork_with_epsilon_zero_splits_evenly_for_any_beta():
    for beta in BETAS:
        probs = all_permutation_probabilities(U_FORK, beta=beta, epsilon=0.0)
        assert probs[(0, 1, 2)] == pytest.approx(0.5, abs=1e-12)
        assert probs[(1, 0, 2)] == pytest.approx(0.5, abs=1e-12)
        for perm, p in probs.items():
            if perm not in {(0, 1, 2), (1, 0, 2)}:
                assert p == 0.0


def test_antichain_is_uniform_over_all_executions_for_any_beta():
    for beta in BETAS:
        for eps in EPSILONS:
            probs = all_permutation_probabilities(U_ANTICHAIN, beta=beta, epsilon=eps)
            for p in probs.values():
                assert p == pytest.approx(1.0 / 6.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Task 9 — THE acceptance criterion: the likelihood normalises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("eps", EPSILONS)
@pytest.mark.parametrize(
    "name,u", [("chain", U_CHAIN), ("fork", U_FORK), ("antichain", U_ANTICHAIN)]
)
def test_known_models_normalise(name, u, beta, eps):
    probs = all_permutation_probabilities(u, beta, eps)
    assert len(probs) == math.factorial(u.shape[0])
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-10)


def test_random_models_normalise():
    """20 random U per (m, d), across the full beta x epsilon grid."""
    rng = np.random.default_rng(SEED + 2)
    checked = 0
    for m in M_GRID:
        for d in D_GRID:
            for _ in range(DRAWS):
                u = rng.normal(size=(m, d))
                for beta in BETAS:
                    for eps in EPSILONS:
                        probs = all_permutation_probabilities(u, beta, eps)
                        assert len(probs) == math.factorial(m)
                        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-10)
                        checked += 1
    assert checked == len(M_GRID) * len(D_GRID) * DRAWS * len(BETAS) * len(EPSILONS)


def test_all_permutation_probabilities_matches_the_step_by_step_likelihood():
    """The cached enumerator and the reference per-step path agree exactly."""
    rng = np.random.default_rng(SEED + 3)
    for m in (2, 3, 4):
        for _ in range(5):
            u = rng.normal(size=(m, 2))
            for beta, eps in ((0.0, 0.0), (1.5, 0.05), (5.0, 0.1)):
                table = all_permutation_probabilities(u, beta, eps)
                for perm, p in table.items():
                    assert bpop_likelihood(perm, u, beta, eps) == pytest.approx(
                        p, abs=1e-15
                    )


def test_ties_between_u_rows_still_normalise():
    """Repeated rows make many roles mutually incomparable — still a distribution."""
    u = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    for beta in BETAS:
        for eps in EPSILONS:
            probs = all_permutation_probabilities(u, beta, eps)
            assert sum(probs.values()) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# likelihood contract
# ---------------------------------------------------------------------------


def test_log_likelihood_matches_manual_product_on_the_fork():
    beta, eps = 1.5, 0.05
    u = U_FORK
    expected = 1.0
    remaining = [0, 1, 2]
    for y in (0, 1, 2):
        expected *= bpop_step_probabilities(remaining, u, beta, eps)[y]
        remaining.remove(y)
    assert bpop_likelihood((0, 1, 2), u, beta, eps) == pytest.approx(expected, abs=1e-15)
    assert bpop_log_likelihood((0, 1, 2), u, beta, eps) == pytest.approx(
        math.log(expected)
    )


def test_zero_probability_execution_gives_minus_inf_and_zero():
    assert bpop_log_likelihood((3, 2, 1, 0), U_CHAIN, 1.5, 0.0) == -math.inf
    assert bpop_likelihood((3, 2, 1, 0), U_CHAIN, 1.5, 0.0) == 0.0


@pytest.mark.parametrize(
    "bad_sequence", [(0, 1), (0, 1, 2, 3), (0, 0, 1), (0, 1, 3), (0, 1, -1)]
)
def test_log_likelihood_rejects_non_permutations(bad_sequence):
    with pytest.raises(ValueError):
        bpop_log_likelihood(bad_sequence, U_FORK, 1.5, 0.05)


def test_spec_reference_values():
    """The three values the Stage-0 toy spec pins down, at beta=1.5, eps=0.05."""
    beta, eps = 1.5, 0.05
    u_a = np.array([[1.0, 1.0], [0.0, 0.0]])
    u_b = np.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
    assert bpop_likelihood((0, 1), u_a, beta, eps) == pytest.approx(0.975, abs=1e-12)
    assert bpop_likelihood((0, 1, 2), u_b, beta, eps) == pytest.approx(
        0.9425, abs=1e-12
    )
    assert bpop_likelihood((2, 0, 1), u_b, beta, eps) == pytest.approx(
        0.01625, abs=1e-12
    )


# ---------------------------------------------------------------------------
# Task 10 — sampling
# ---------------------------------------------------------------------------


def test_sampling_always_returns_a_permutation():
    rng = np.random.default_rng(SEED + 4)
    for m in M_GRID:
        u = rng.normal(size=(m, 2))
        for _ in range(50):
            seq = sample_bpop_sequence(rng, u, beta=1.5, epsilon=0.05)
            assert sorted(seq) == list(range(m))


def test_sampling_respects_the_order_when_epsilon_is_zero():
    rng = np.random.default_rng(SEED + 5)
    for _ in range(200):
        assert sample_bpop_sequence(rng, U_CHAIN, beta=1.5, epsilon=0.0) == (0, 1, 2, 3)


def test_sampling_frequencies_match_analytic_probabilities_on_the_fork():
    rng = np.random.default_rng(SEED + 6)
    n = 20_000
    counts: dict[tuple[int, ...], int] = {}
    for _ in range(n):
        seq = sample_bpop_sequence(rng, U_FORK, beta=1.5, epsilon=0.0)
        counts[seq] = counts.get(seq, 0) + 1
    assert set(counts) == {(0, 1, 2), (1, 0, 2)}
    assert counts[(0, 1, 2)] / n == pytest.approx(0.5, abs=0.02)
    assert counts[(1, 0, 2)] / n == pytest.approx(0.5, abs=0.02)


def test_sampling_frequencies_match_analytic_probabilities_with_noise():
    rng = np.random.default_rng(SEED + 7)
    beta, eps, n = 1.5, 0.1, 40_000
    exact = all_permutation_probabilities(U_FORK, beta, eps)
    counts: dict[tuple[int, ...], int] = {p: 0 for p in exact}
    for _ in range(n):
        counts[sample_bpop_sequence(rng, U_FORK, beta, eps)] += 1
    for perm, p in exact.items():
        assert counts[perm] / n == pytest.approx(p, abs=0.02)
