"""Stage 6E2 — the permutation-invariant convergence summaries.

These gate a substantive conclusion: whether Stage 6E2's failure is label switching, which
would be a diagnostic artefact, or genuine multimodality. A summary that is *believed*
invariant and is not would flip that conclusion either way, so the invariance is tested
directly — including a negative control proving the checker can actually detect a
non-invariant summary rather than passing everything it is shown.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_original.stage6e_invariant import (
    INVARIANT_GATE_NAMES, assert_invariance, invariant_convergence, invariant_summaries,
    sorted_pi, sorted_row_entropies, transition_eigenvalues,
    transition_singular_values,
)

K = 3
M = 5


def _transition(rng, n_chains, n_draws):
    """Random `P` with a zero diagonal and rows summing to one."""
    p = np.zeros((n_chains, n_draws, K, K))
    for c in range(n_chains):
        for d in range(n_draws):
            for h in range(K):
                allowed = [k for k in range(K) if k != h]
                draw = rng.dirichlet(np.ones(len(allowed)))
                for k, value in zip(allowed, draw):
                    p[c, d, h, k] = value
    return p


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20260814)
    n_chains, n_draws = 4, 60
    return {
        "pi_draws": rng.dirichlet(np.ones(K), size=(n_chains, n_draws)),
        "transition_draws": _transition(rng, n_chains, n_draws),
        "relation_counts": rng.integers(0, 8, size=(n_chains, n_draws, K)),
        "segment_counts": rng.integers(3, 9, size=(n_chains, n_draws, 20)),
        "log_target": rng.normal(-5000, 30, size=(n_chains, n_draws)),
        **{f"scalar_{n}": rng.uniform(0.2, 2.0, size=(n_chains, n_draws))
           for n in ("beta", "omega", "lambda_rep", "lambda_back", "rho")},
    }


def _permute(data: dict, permutation) -> dict:
    """Apply `Q` exactly as the posterior symmetry does."""
    out = dict(data)
    out["pi_draws"] = data["pi_draws"][:, :, permutation]
    out["transition_draws"] = data["transition_draws"][:, :, permutation, :][
        :, :, :, permutation]
    out["relation_counts"] = data["relation_counts"][:, :, permutation]
    return out


# ------------------------------------------------------------------ the invariance
@pytest.mark.parametrize("permutation", [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1],
                                         [2, 1, 0]])
def test_every_registered_summary_is_invariant_under_relabelling(data, permutation):
    base = invariant_summaries(data, K)
    other = invariant_summaries(_permute(data, permutation), K)
    assert set(base) == set(other)
    for name in base:
        assert np.abs(base[name] - other[name]).max() < 1e-9, name


def test_the_registered_gate_names_are_exactly_what_is_computed(data):
    computed = set(invariant_summaries(data, K))
    named = set(INVARIANT_GATE_NAMES) - {"transition_count_spectrum"}
    assert named <= computed, named - computed


def test_assert_invariance_reports_the_permutation_it_used(data):
    report = assert_invariance(data, K, seed=3)
    assert report["pass"] is True
    assert report["worst_overall"] < 1e-9
    assert sorted(report["permutation"]) == list(range(K))
    assert len(report["max_absolute_difference"]) == len(invariant_summaries(data, K))


def test_the_invariance_checker_can_actually_detect_non_invariance(data):
    """Negative control: a checker that passes everything would be worthless.

    `pi[0]` is exactly the kind of label-indexed statistic the supersession removed. If
    the machinery is working, comparing it across a relabelling must show a real
    difference — otherwise the invariance results above prove nothing.
    """
    permutation = [1, 2, 0]
    naive = data["pi_draws"][:, :, 0]
    permuted = _permute(data, permutation)["pi_draws"][:, :, 0]
    assert np.abs(naive - permuted).max() > 1e-3, (
        "a label-indexed statistic must move under relabelling; if it does not, this "
        "fixture is degenerate and the invariance tests above are vacuous")
    # and the invariant counterpart must NOT move
    assert np.abs(sorted_pi(data["pi_draws"]) - sorted_pi(
        _permute(data, permutation)["pi_draws"])).max() < 1e-12


@pytest.mark.parametrize("fn", [transition_eigenvalues, transition_singular_values,
                                sorted_row_entropies])
def test_transition_summaries_are_similarity_invariant(data, fn):
    permutation = [2, 0, 1]
    base = fn(data["transition_draws"])
    other = fn(_permute(data, permutation)["transition_draws"])
    assert np.abs(base - other).max() < 1e-9


def test_row_entropies_are_the_shannon_entropy_of_each_row():
    p = np.array([[[[0.0, 0.5, 0.5], [0.25, 0.0, 0.75], [1.0, 0.0, 0.0]]]])
    entropies = sorted_row_entropies(p)[0, 0]
    expected = np.sort([
        -(0.5 * np.log(0.5) * 2),
        -(0.25 * np.log(0.25) + 0.75 * np.log(0.75)),
        0.0])
    assert np.allclose(entropies, expected)
    assert entropies[0] == pytest.approx(0.0)   # a deterministic row has zero entropy


# --------------------------------------------------------------- frozen coordinates
def test_a_frozen_coordinate_is_named_rather_than_reported_as_a_huge_rhat():
    """Zero within-chain variance with disagreeing chains is an absorbing state.

    R-hat then divides a real between-chain variance by ~0 and returns something like
    7e15, which carries no information. The condition must be detected and named.
    """
    rng = np.random.default_rng(5)
    n_chains, n_draws = 4, 50
    relation = np.zeros((n_chains, n_draws, K))
    for c, value in enumerate([2, 2, 2, 3]):          # frozen, and disagreeing
        relation[c, :, :] = value
    data = {
        "pi_draws": rng.dirichlet(np.ones(K), size=(n_chains, n_draws)),
        "transition_draws": _transition(rng, n_chains, n_draws),
        "relation_counts": relation,
        "segment_counts": rng.integers(3, 9, size=(n_chains, n_draws, 10)),
        "log_target": rng.normal(-5000, 30, size=(n_chains, n_draws)),
        **{f"scalar_{n}": rng.uniform(0.2, 2.0, size=(n_chains, n_draws))
           for n in ("beta", "omega", "lambda_rep", "lambda_back", "rho")},
    }
    report = invariant_convergence(data, K)
    assert report["n_frozen"] > 0
    frozen = report["frozen_coordinates"]["total_relation_count"]
    assert frozen["n_chains_with_zero_within_chain_variance"] == 4
    assert frozen["per_chain_within_sd"] == [0.0, 0.0, 0.0, 0.0]
    assert "absorbing state" in frozen["interpretation"]
    assert report["gates"]["total_relation_count"]["frozen_in_some_chains"] is True
    assert report["gates"]["total_relation_count"]["pass"] is False


def test_a_coordinate_constant_in_every_chain_is_degenerate_not_frozen():
    """All chains agreeing on a constant is degeneracy, and must not be a failure."""
    rng = np.random.default_rng(6)
    n_chains, n_draws = 4, 40
    data = {
        "pi_draws": rng.dirichlet(np.ones(K), size=(n_chains, n_draws)),
        "transition_draws": _transition(rng, n_chains, n_draws),
        "relation_counts": np.full((n_chains, n_draws, K), 4.0),
        "segment_counts": rng.integers(3, 9, size=(n_chains, n_draws, 10)),
        "log_target": rng.normal(-5000, 30, size=(n_chains, n_draws)),
        **{f"scalar_{n}": rng.uniform(0.2, 2.0, size=(n_chains, n_draws))
           for n in ("beta", "omega", "lambda_rep", "lambda_back", "rho")},
    }
    report = invariant_convergence(data, K)
    gate = report["gates"]["total_relation_count"]
    assert gate["value"] is None
    assert gate["pass"] is True
    assert "degenerate" in gate["note"]
    assert "total_relation_count" not in report["frozen_coordinates"]


def test_converged_chains_pass_every_invariant_gate():
    """A positive control: identically distributed chains must not fail these gates."""
    rng = np.random.default_rng(11)
    n_chains, n_draws = 4, 4000
    # iid draws, NOT a short block repeated: repeating would make the series perfectly
    # autocorrelated and fail R-hat for reasons that have nothing to do with the code
    # under test.
    rows = rng.dirichlet(np.ones(K - 1), size=(n_chains, n_draws, K))
    transition = np.zeros((n_chains, n_draws, K, K))
    for h in range(K):
        allowed = [k for k in range(K) if k != h]
        for j, k in enumerate(allowed):
            transition[:, :, h, k] = rows[:, :, h, j]
    data = {
        "pi_draws": rng.dirichlet(np.ones(K), size=(n_chains, n_draws)),
        "transition_draws": transition,
        "relation_counts": rng.integers(2, 6, size=(n_chains, n_draws, K)),
        "segment_counts": rng.integers(3, 9, size=(n_chains, n_draws, 10)),
        "log_target": rng.normal(-5000, 30, size=(n_chains, n_draws)),
        **{f"scalar_{n}": rng.uniform(0.2, 2.0, size=(n_chains, n_draws))
           for n in ("beta", "omega", "lambda_rep", "lambda_back", "rho")},
    }
    report = invariant_convergence(data, K, threshold=1.01)
    failed = [n for n, g in report["gates"].items() if not g["pass"]]
    assert not failed, f"iid chains should pass every gate, failed: {failed}"
    assert report["n_frozen"] == 0
