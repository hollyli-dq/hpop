"""Correctness tests for the ``U -> h(U)`` latent partial-order mapping.

The central claim under test: for *any* real matrix U, coordinate-wise dominance

    i > j  iff  U[i, r] > U[j, r] for all r

is a strict partial order — irreflexive, asymmetric and transitive. These
properties are checked on hand-built examples and on many random draws.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from hpop.mcmc_original.latent_poset import (
    incomparable,
    precedence_from_u,
    predecessors,
    successors,
)

SEED = 20260808
CONFIGURATIONS = [(m, d) for m in (2, 3, 4, 5, 8) for d in (1, 2, 3, 4)]
DRAWS_PER_CONFIGURATION = 20


def assert_irreflexive(p: np.ndarray) -> None:
    for i in range(p.shape[0]):
        assert not p[i, i], f"P[{i},{i}] must be False (irreflexivity)"


def assert_asymmetric(p: np.ndarray) -> None:
    m = p.shape[0]
    for i in range(m):
        for j in range(m):
            if p[i, j]:
                assert not p[j, i], f"P[{i},{j}] and P[{j},{i}] both True (asymmetry)"


def assert_transitive(p: np.ndarray) -> None:
    m = p.shape[0]
    for i in range(m):
        for j in range(m):
            for k in range(m):
                if p[i, j] and p[j, k]:
                    assert p[i, k], (
                        f"P[{i},{j}] and P[{j},{k}] hold but P[{i},{k}] is False "
                        "(transitivity)"
                    )


def assert_strict_partial_order(p: np.ndarray) -> None:
    assert_irreflexive(p)
    assert_asymmetric(p)
    assert_transitive(p)


def random_u_matrices() -> list[tuple[int, int, int, np.ndarray]]:
    """Deterministic pool of random U matrices across the configuration grid."""
    rng = np.random.default_rng(SEED)
    cases: list[tuple[int, int, int, np.ndarray]] = []
    for m, d in CONFIGURATIONS:
        for draw in range(DRAWS_PER_CONFIGURATION):
            cases.append((m, d, draw, rng.normal(size=(m, d))))
    return cases


RANDOM_CASES = random_u_matrices()


# --------------------------------------------------------------------------
# Basic contract
# --------------------------------------------------------------------------


def test_returns_bool_matrix_of_right_shape():
    p = precedence_from_u(np.array([[1.0, 2.0], [0.0, 0.0], [3.0, 3.0]]))
    assert p.dtype == np.bool_
    assert p.shape == (3, 3)


def test_accepts_array_like_input():
    p = precedence_from_u([[1.0, 1.0], [0.0, 0.0]])
    assert p.dtype == np.bool_
    assert p[0, 1] and not p[1, 0]


def test_integer_input_is_accepted():
    p = precedence_from_u(np.array([[2, 2], [1, 1]], dtype=int))
    assert p[0, 1] and not p[1, 0]


def test_input_array_is_not_mutated():
    u = np.array([[1.0, 0.0], [0.0, 1.0]])
    original = u.copy()
    precedence_from_u(u)
    np.testing.assert_array_equal(u, original)


def test_single_role_has_empty_order():
    p = precedence_from_u(np.array([[0.5]]))
    assert p.shape == (1, 1)
    assert not p[0, 0]


def test_one_dimensional_latent_space_is_allowed():
    p = precedence_from_u(np.array([[2.0], [1.0], [0.0]]))
    assert p[0, 1] and p[1, 2] and p[0, 2]
    assert_strict_partial_order(p)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        np.array([1.0, 2.0, 3.0]),  # 1-D
        np.zeros((2, 2, 2)),  # 3-D
    ],
)
def test_rejects_wrong_ndim(bad):
    with pytest.raises(ValueError, match="2-D"):
        precedence_from_u(bad)


def test_rejects_zero_roles():
    with pytest.raises(ValueError, match="at least one role"):
        precedence_from_u(np.zeros((0, 3)))


def test_rejects_zero_latent_dimension():
    with pytest.raises(ValueError, match="latent dimension"):
        precedence_from_u(np.zeros((3, 0)))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_values(bad_value):
    u = np.array([[1.0, 0.0], [0.0, bad_value]])
    with pytest.raises(ValueError, match="NaN or inf"):
        precedence_from_u(u)


def test_rejects_non_numeric_input():
    with pytest.raises(ValueError):
        precedence_from_u([["a", "b"], ["c", "d"]])


# --------------------------------------------------------------------------
# Known examples
# --------------------------------------------------------------------------


def test_known_v_shaped_example():
    """Two incomparable roles both preceding a third."""
    u = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
        ]
    )
    p = precedence_from_u(u)

    assert p[0, 2]
    assert p[1, 2]
    assert not p[0, 1]
    assert not p[1, 0]

    assert incomparable(p, 0, 1)
    assert not incomparable(p, 0, 2)
    assert not incomparable(p, 2, 0)

    assert predecessors(p, 2) == (0, 1)
    assert predecessors(p, 0) == ()
    assert predecessors(p, 1) == ()
    assert successors(p, 0) == (2,)
    assert successors(p, 1) == (2,)
    assert successors(p, 2) == ()

    assert_strict_partial_order(p)


def test_known_total_order_example_is_transitively_closed():
    u = np.array(
        [
            [3.0, 3.0],
            [2.0, 2.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )
    p = precedence_from_u(u)

    # Cover relations.
    assert p[0, 1]
    assert p[1, 2]
    assert p[2, 3]
    # The returned matrix is the transitive closure, not a Hasse diagram.
    assert p[0, 2]
    assert p[0, 3]
    assert p[1, 3]

    # Nothing points backwards.
    for i, j in itertools.combinations(range(4), 2):
        assert not p[j, i]

    assert p.sum() == 6  # all ordered pairs of a 4-element chain
    assert_strict_partial_order(p)


def test_identical_rows_are_incomparable():
    u = np.array([[1.0, 1.0], [1.0, 1.0]])
    p = precedence_from_u(u)
    assert not p.any()
    assert incomparable(p, 0, 1)


def test_antichain_is_empty_order():
    """Every role wins on exactly one coordinate: no dominance anywhere."""
    u = np.eye(4)
    p = precedence_from_u(u)
    assert not p.any()
    for i, j in itertools.combinations(range(4), 2):
        assert incomparable(p, i, j)


def test_dominance_requires_all_coordinates():
    """Strictly greater on coordinate 0 but tied on coordinate 1 is not >."""
    u = np.array([[2.0, 1.0], [1.0, 1.0]])
    p = precedence_from_u(u)
    assert not p[0, 1]
    assert not p[1, 0]
    assert incomparable(p, 0, 1)


# --------------------------------------------------------------------------
# Randomised proof of the partial-order axioms
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m,d,draw,u",
    RANDOM_CASES,
    ids=[f"m{m}-d{d}-draw{draw}" for m, d, draw, _ in RANDOM_CASES],
)
def test_random_u_induces_strict_partial_order(m, d, draw, u):
    p = precedence_from_u(u)
    assert p.shape == (m, m)
    assert p.dtype == np.bool_
    assert_strict_partial_order(p)


def test_random_ties_still_induce_strict_partial_order():
    """Coarsely quantised U produces many ties; the axioms must still hold."""
    rng = np.random.default_rng(SEED + 1)
    for m, d in CONFIGURATIONS:
        for _ in range(DRAWS_PER_CONFIGURATION):
            u = rng.integers(low=0, high=3, size=(m, d)).astype(float)
            assert_strict_partial_order(precedence_from_u(u))


def test_random_matches_naive_definition():
    """Vectorised implementation agrees with the literal elementwise loop."""
    rng = np.random.default_rng(SEED + 2)
    for m, d in CONFIGURATIONS:
        for _ in range(DRAWS_PER_CONFIGURATION):
            u = rng.normal(size=(m, d))
            p = precedence_from_u(u)
            for i in range(m):
                for j in range(m):
                    expected = i != j and all(u[i, r] > u[j, r] for r in range(d))
                    assert bool(p[i, j]) == expected, (i, j)


def test_order_is_invariant_to_coordinatewise_monotone_rescaling():
    """h(U) depends only on per-coordinate ordering, not on the scale."""
    rng = np.random.default_rng(SEED + 3)
    for _ in range(20):
        u = rng.normal(size=(6, 3))
        rescaled = np.column_stack(
            [3.0 * u[:, 0] + 5.0, np.exp(u[:, 1]), np.arctan(u[:, 2])]
        )
        np.testing.assert_array_equal(
            precedence_from_u(u), precedence_from_u(rescaled)
        )


def test_permuting_roles_permutes_the_order():
    rng = np.random.default_rng(SEED + 4)
    for _ in range(20):
        u = rng.normal(size=(5, 2))
        perm = rng.permutation(5)
        np.testing.assert_array_equal(
            precedence_from_u(u[perm]), precedence_from_u(u)[np.ix_(perm, perm)]
        )


# --------------------------------------------------------------------------
# predecessors / successors / incomparable
# --------------------------------------------------------------------------


def test_predecessors_and_successors_are_consistent_on_random_orders():
    rng = np.random.default_rng(SEED + 5)
    for m, d in CONFIGURATIONS:
        for _ in range(DRAWS_PER_CONFIGURATION):
            p = precedence_from_u(rng.normal(size=(m, d)))
            for x in range(m):
                preds = predecessors(p, x)
                succs = successors(p, x)
                assert preds == tuple(sorted(preds))
                assert succs == tuple(sorted(succs))
                assert x not in preds
                assert x not in succs
                for z in preds:
                    assert p[z, x]
                    assert x in successors(p, z)
                for z in succs:
                    assert p[x, z]
                    assert x in predecessors(p, z)


def test_incomparable_partitions_every_distinct_pair():
    rng = np.random.default_rng(SEED + 6)
    for m, d in CONFIGURATIONS:
        for _ in range(DRAWS_PER_CONFIGURATION):
            p = precedence_from_u(rng.normal(size=(m, d)))
            for i, j in itertools.combinations(range(m), 2):
                exactly_one = int(p[i, j]) + int(p[j, i]) + int(incomparable(p, i, j))
                assert exactly_one == 1, (i, j)


def test_incomparable_is_false_for_a_role_with_itself():
    p = precedence_from_u(np.array([[1.0], [0.0]]))
    assert not incomparable(p, 0, 0)
    assert not incomparable(p, 1, 1)


def test_incomparable_is_symmetric():
    rng = np.random.default_rng(SEED + 7)
    for _ in range(20):
        p = precedence_from_u(rng.normal(size=(6, 2)))
        for i, j in itertools.combinations(range(6), 2):
            assert incomparable(p, i, j) == incomparable(p, j, i)


@pytest.mark.parametrize("bad_index", [-1, 3, 100])
def test_role_queries_reject_out_of_range_indices(bad_index):
    p = precedence_from_u(np.array([[1.0], [0.0], [-1.0]]))
    with pytest.raises(ValueError, match="out of range"):
        predecessors(p, bad_index)
    with pytest.raises(ValueError, match="out of range"):
        successors(p, bad_index)
    with pytest.raises(ValueError, match="out of range"):
        incomparable(p, 0, bad_index)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 3), dtype=bool),  # not square
        np.zeros((2, 2), dtype=float),  # not boolean
        np.zeros((2,), dtype=bool),  # not 2-D
    ],
)
def test_role_queries_reject_malformed_precedence_matrices(bad):
    with pytest.raises(ValueError):
        predecessors(bad, 0)
    with pytest.raises(ValueError):
        successors(bad, 0)
    with pytest.raises(ValueError):
        incomparable(bad, 0, 1)
