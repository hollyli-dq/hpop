"""The terminal gate must work at every K on the ladder, not only at K = 3.

The gate used to reshape relation indicators with a literal 3. That is the worst kind of
bug: `reshape(n, 3, width // 3)` succeeds whenever the width happens to divide by three,
so at K = 30 with m = 10 (width 30*10*9 = 2700, divisible by 3) it would have produced a
confident, wrong library identifier rather than an error.

These tests pin the two properties that matter: the identifier is computed over the right
number of skills, and it is invariant to relabelling them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "terminal_gate", ROOT / "scripts" / "confirmatory_terminal_gate.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

LADDER = (3, 5, 10, 20, 30)
M_ROLES = 10                       # the registered role-support size for the K ladder


def indicators(n_draws: int, n_skills: int, m: int, seed: int) -> np.ndarray:
    """Relation-indicator bits laid out as the chains write them: K blocks of m(m-1)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n_draws, n_skills * m * (m - 1))).astype(bool)


@pytest.mark.parametrize("K", LADDER)
def test_library_ids_uses_the_K_it_is_given(K):
    bits = indicators(40, K, M_ROLES, seed=K)
    ids, table = gate.library_ids(bits, K)
    assert ids.shape == (40,)
    assert len(table) == len(set(ids.tolist()))
    assert all(len(h) == 16 for h in table.values())


@pytest.mark.parametrize("K", LADDER)
def test_identifier_is_invariant_to_relabelling_the_skills(K):
    """The whole point: 30! relabellings, and none of them may change the identifier."""
    per_skill = M_ROLES * (M_ROLES - 1)
    bits = indicators(12, K, M_ROLES, seed=100 + K)
    blocks = bits.reshape(12, K, per_skill)

    rng = np.random.default_rng(7)
    permuted = np.empty_like(blocks)
    for i in range(12):
        permuted[i] = blocks[i][rng.permutation(K)]

    original, _ = gate.library_ids(bits, K)
    relabelled, _ = gate.library_ids(permuted.reshape(12, -1), K)
    assert np.array_equal(original, relabelled)


@pytest.mark.parametrize("K", LADDER)
def test_two_draws_with_the_same_multiset_of_closures_get_the_same_id(K):
    per_skill = M_ROLES * (M_ROLES - 1)
    rng = np.random.default_rng(11)
    one = rng.integers(0, 2, size=(K, per_skill)).astype(bool)
    two = one[rng.permutation(K)]
    ids, _ = gate.library_ids(np.stack([one.ravel(), two.ravel()]), K)
    assert ids[0] == ids[1]


@pytest.mark.parametrize("K", LADDER)
def test_a_genuinely_different_library_gets_a_different_id(K):
    """Invariance must not have collapsed into indifference."""
    per_skill = M_ROLES * (M_ROLES - 1)
    rng = np.random.default_rng(13)
    one = rng.integers(0, 2, size=(K, per_skill)).astype(bool)
    two = one.copy()
    two[0, 0] = ~two[0, 0]
    ids, _ = gate.library_ids(np.stack([one.ravel(), two.ravel()]), K)
    assert ids[0] != ids[1]


def test_a_mismatched_width_is_an_error_not_a_silent_reshape():
    """The old failure mode: 2700 bits divides by 3, so K=3 would have 'worked'."""
    bits = indicators(5, 30, M_ROLES, seed=1)          # width 2700, really 30 skills
    assert bits.shape[1] % 3 == 0, "the premise of this test"
    wrong_ids, wrong_table = gate.library_ids(bits, 3)   # divides, so no error is raised
    right_ids, right_table = gate.library_ids(bits, 30)

    # The integer ids are enumeration indices assigned in order of first appearance, so
    # five distinct draws give [0,1,2,3,4] under BOTH readings. The content lives in the
    # hash table, and that is what must differ.
    assert np.array_equal(wrong_ids, right_ids), "premise: ids alone cannot tell them apart"
    assert set(wrong_table.values()) != set(right_table.values()), (
        "reading 30 skills as 3 must not produce the same library hashes")

    with pytest.raises(ValueError, match="not divisible"):
        gate.library_ids(bits, 7)                       # 2700 % 7 != 0 -> loud
    with pytest.raises(ValueError):
        gate.library_ids(bits, 0)


@pytest.mark.parametrize("K", LADDER)
def test_skills_in_reads_K_from_the_chain(K):
    chain = {"u_draws": np.zeros((17, K, M_ROLES, 2))}
    assert gate.skills_in(chain) == K


def test_skills_in_rejects_a_malformed_u_draws():
    with pytest.raises(ValueError, match=r"\(draws, K, m, d\)"):
        gate.skills_in({"u_draws": np.zeros((17, 3))})
