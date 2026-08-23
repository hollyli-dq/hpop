"""Stage 6C — the poset catalogue (§17 areas 1, 2, 3, 26, 37).

The catalogue is a *label set for reporting*, not the chain's state space. The Stage 6C
state is the continuous matrix `U`; `h(U)` is derived. These tests check that the label
set is exhaustive, unambiguous and internally consistent, and that nothing in the frozen
configuration confuses the three integers that are all easy to conflate: the role count
`m = 5`, the latent dimension `d = 2`, and the skill count `K = 1`.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.stage6c_exact_reference import (
    build_catalogue, is_partial_order, transitive_reduction,
)
from hpop.mcmc_original.stage6c_frozen import frozen_config, load_stage6c_dataset


@pytest.fixture(scope="module")
def catalogue():
    return build_catalogue(5, 2)


# --------------------------------------------------------------- area 26: exhaustive
def test_catalogue_is_every_labelled_poset_on_five_elements(catalogue):
    """4231 is the number of labelled partial orders on 5 elements (OEIS A001035).

    It is asserted here as an independently known constant, not read back out of the
    enumerator, so a bug in the enumerator cannot define its own success criterion.
    """
    assert catalogue.size == 4231


def test_every_ranking_tuple_is_accounted_for(catalogue):
    assert int(catalogue.ranking_tuple_counts.sum()) == math.factorial(5) ** 2 == 14_400


def test_enumerator_agrees_with_an_independent_brute_force_on_a_small_case():
    """m = 3: rebuild the reachable set with a completely separate loop and compare."""
    catalogue = build_catalogue(3, 2)
    seen = set()
    for a in itertools.permutations(range(3)):
        for b in itertools.permutations(range(3)):
            u = np.array([[a[i], b[i]] for i in range(3)], dtype=float)
            seen.add(precedence_from_u(u).tobytes())
    assert catalogue.size == len(seen)
    assert {c.tobytes() for c in catalogue.closures} == seen


# ------------------------------------------------------------- areas 1, 2: canonical
def test_no_duplicate_states(catalogue):
    keys = catalogue.keys
    assert keys.size == np.unique(keys).size


def test_every_entry_is_a_partial_order(catalogue):
    assert all(is_partial_order(p) for p in catalogue.closures)


def test_lookup_is_exact_and_total(catalogue):
    """Every stored closure resolves to its own index, and lookup is order-independent."""
    indices = catalogue.indices_of(catalogue.closures)
    assert np.array_equal(indices, np.arange(catalogue.size))
    for i in (0, 17, 2000, catalogue.size - 1):
        assert catalogue.index_of(catalogue.closures[i]) == i


def test_representative_u_induces_the_order_it_is_filed_under(catalogue):
    for u, closure in zip(catalogue.representatives, catalogue.closures):
        assert np.array_equal(precedence_from_u(u), closure)


def test_unknown_relation_returns_minus_one(catalogue):
    """A relation outside the catalogue must be reported, never silently mapped."""
    bogus = np.zeros((5, 5), dtype=bool)
    bogus[0, 1] = bogus[1, 0] = True          # not antisymmetric, so not a partial order
    assert catalogue.index_of(bogus) == -1


# --------------------------------------------------------- area 3: closure/reduction
def test_reduction_round_trips_to_the_closure(catalogue):
    """Reduction then transitive re-closure must return the original closure, for all."""
    for closure, reduction in zip(catalogue.closures, catalogue.reductions):
        reach = reduction.astype(bool).copy()
        for _ in range(catalogue.m):
            reach = reach | ((reach.astype(int) @ reduction.astype(int)) > 0)
        assert np.array_equal(reach, closure.astype(bool))


def test_reduction_is_contained_in_the_closure(catalogue):
    assert np.all(catalogue.reductions <= catalogue.closures)


def test_reduction_of_a_reduction_is_idempotent_on_a_sample(catalogue):
    for i in (0, 5, 500, 4002, catalogue.size - 1):
        reduction = catalogue.reductions[i]
        assert np.array_equal(transitive_reduction(reduction), reduction)


def test_closure_is_transitively_closed(catalogue):
    """The stored representation is the closure, not the cover relation."""
    for i in (0, 100, 4002, catalogue.size - 1):
        closure = catalogue.closures[i].astype(int)
        assert not (((closure @ closure) > 0) & ~closure.astype(bool)).any()


# ------------------------------------------------------ the registered true structure
def test_true_poset_is_in_the_catalogue_with_its_registered_relations():
    frozen = load_stage6c_dataset()
    catalogue = build_catalogue(5, 2)
    closure = precedence_from_u(frozen.u_true)
    index = catalogue.index_of(closure)
    assert index >= 0
    expected = {(0, 2), (0, 3), (0, 4), (2, 3), (2, 4), (3, 4)}
    observed = {(int(i), int(j)) for i, j in zip(*np.where(closure))}
    assert observed == expected
    # role 1 is incomparable to everything, which is the point of this fixture
    assert not closure[1].any() and not closure[:, 1].any()


# ------------------------------------------ area 37: m vs d vs K must not be confused
def test_role_count_latent_dimension_and_skill_count_are_distinct():
    frozen = load_stage6c_dataset()
    assert frozen.n_roles == 5
    assert frozen.latent_dimension == 2
    assert frozen.n_skills == 1
    assert frozen.n_roles != frozen.latent_dimension
    assert frozen.n_skills != frozen.n_roles
    assert frozen.n_skills != frozen.latent_dimension


def test_frozen_config_records_a_single_skill_not_the_role_count():
    config = frozen_config()
    assert config["n_skills"] == 1
    assert "R^{m x d}" in config["structural_model"]["latent"]


def test_catalogue_dimensions_are_the_role_count_not_the_latent_dimension(catalogue):
    assert catalogue.closures.shape == (catalogue.size, 5, 5)
    assert catalogue.representatives.shape == (catalogue.size, 5, 2)
