"""Step 7B2 — the machinery for the full-corpus Local-vs-FFBS comparison.

Step 7B2 itself cannot run until the Stage 6E2 baseline is frozen at its registered
decision point, and these tests do not pretend otherwise: they exercise the diagnostics and
the comparison schemas on synthetic draws, and they check the guards that stop the run from
starting against a moving baseline or a different corpus.

The one substantive property tested here is that the convergence path never touches the
hidden truth. Aligning labels to the truth before judging cross-chain agreement would turn
`3! = 6` label exchangeability into manufactured convergence; alignment belongs after the
verdict, in recovery.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hpop.mcmc_original import stage7b_diagnostics as diagnostics
from hpop.mcmc_original.stage7b_diagnostics import (
    INVARIANT_SUMMARY_NAMES, assert_no_truth_alignment, co_clustering_series,
    compare_equal_sweeps, compare_equal_time, compare_heldout, h_label, h_label_series,
    invariant_summaries, mode_occupancy, segmentation_movement, structural_movement,
)

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path("/Users/dongqing/Desktop/hpop-stage6e/results/mcmc_original"
                "/stage6e2_unknown_boundary_full_seed0")


def fake_chain(n_draws, K=3, m=5, d=2, frozen=True, seed=0):
    """Draws whose structure either never moves or moves every few draws."""
    rng = np.random.default_rng(seed)
    if frozen:
        base = rng.normal(size=(K, m, d))
        u = np.repeat(base[None], n_draws, axis=0) + rng.normal(scale=1e-6,
                                                                size=(n_draws, K, m, d))
    else:
        u = rng.normal(size=(n_draws, K, m, d)) * 2.0
    relation = np.array([[int(np.sum(np.all(u[i, k][:, None, :] > u[i, k][None, :, :],
                                            axis=2)))
                          for k in range(K)] for i in range(n_draws)])
    return {"u_draws": u, "relation_counts": relation,
            "log_target": rng.normal(size=n_draws) - 100.0,
            "segment_counts": rng.integers(2, 6, size=(n_draws, 4)),
            "pi_draws": rng.dirichlet(np.ones(K), size=n_draws),
            "transition_draws": np.array([
                _random_transition(rng, K) for _ in range(n_draws)]),
            "scalars": {name: rng.normal(size=n_draws)
                        for name in ("rho", "beta", "omega", "lambda_rep",
                                     "lambda_back")}}


def _random_transition(rng, K):
    matrix = np.zeros((K, K))
    for h in range(K):
        allowed = [k for k in range(K) if k != h]
        matrix[h, allowed] = rng.dirichlet(np.ones(len(allowed)))
    return matrix


# ------------------------------------------------------------------ structural movement
def test_a_frozen_chain_is_reported_as_frozen():
    chains = [fake_chain(200, frozen=True, seed=s) for s in range(4)]
    movement = structural_movement([c["u_draws"] for c in chains],
                                   [c["relation_counts"] for c in chains])
    assert movement["chains_with_frozen_structure"] == 4
    assert movement["criterion_A_frozen_structure"] is True
    assert movement["total_structural_changes"] == 0
    for row in movement["per_chain"]:
        assert row["distinct_states"] == 1
        assert row["draws_to_first_change"] is None
        assert row["modal_occupancy"] == 1.0


def test_a_moving_chain_is_reported_as_moving():
    chains = [fake_chain(200, frozen=False, seed=10 + s) for s in range(4)]
    movement = structural_movement([c["u_draws"] for c in chains],
                                   [c["relation_counts"] for c in chains])
    assert movement["chains_with_frozen_structure"] == 0
    assert movement["criterion_A_frozen_structure"] is False
    assert movement["total_structural_changes"] > 100
    assert movement["distinct_structural_states_pooled"] > 1
    for row in movement["per_chain"]:
        assert row["draws_to_first_change"] is not None
        assert row["changes_per_1000_draws"] > 0


def test_the_h_label_is_the_induced_order_and_joins_every_skill():
    u = np.zeros((1, 3, 3, 2))
    u[0, 0] = [[2.0, 2.0], [1.0, 0.0], [0.0, 1.0]]
    u[0, 1] = [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0]]
    u[0, 2] = [[3.0, 3.0], [2.0, 2.0], [0.0, 0.0]]
    labels = h_label_series(u)
    assert len(labels) == 1
    assert labels[0] == b"|".join(h_label(u[0, k]) for k in range(3))
    # a strictly increasing reparameterisation of one column leaves the order alone
    moved = u.copy()
    moved[0, 2] = moved[0, 2] * 3.0 + 1.0
    assert h_label_series(moved)[0] == labels[0]


def test_mode_occupancy_counts_changes_and_first_exit():
    labels = [b"a", b"a", b"a", b"b", b"b", b"a"]
    occupancy = mode_occupancy(labels)
    assert occupancy["distinct_states"] == 2
    assert occupancy["changes"] == 2
    assert occupancy["draws_to_first_change"] == 3
    assert occupancy["draws_to_leave_the_initial_state"] == 3
    assert occupancy["modal_occupancy"] == pytest.approx(4 / 6)


def test_segmentation_movement_reads_keys_and_per_sweep_totals():
    keys = [((( 8, 0),), ((4, 1), (8, 2))), ((( 8, 0),), ((4, 1), (8, 2))),
            (((3, 2), (8, 0)), ((8, 1),))]
    movement = segmentation_movement(
        boundary_keys_by_chain=[keys],
        per_sweep_movement=[{"boundary_hamming": 12, "label_changes": 40,
                             "states_changed": 7}])
    row = movement["per_chain"][0]
    assert row["retained_draws"] == 3
    assert row["retained_draws_with_a_change"] == 1
    assert row["distinct_per_trace_segmentations_visited"] == 4
    assert movement["per_sweep_totals"][0]["states_changed"] == 7


# ------------------------------------------------------------- permutation invariance
def test_every_invariant_summary_is_unchanged_by_relabelling_the_skills():
    chain = fake_chain(50, frozen=False, seed=3)
    summaries = invariant_summaries(chain)
    permutation = np.array([2, 0, 1])
    permuted = {
        "u_draws": chain["u_draws"][:, permutation],
        "relation_counts": chain["relation_counts"][:, permutation],
        "log_target": chain["log_target"], "segment_counts": chain["segment_counts"],
        "pi_draws": chain["pi_draws"][:, permutation],
        "transition_draws": chain["transition_draws"][:, permutation][:, :, permutation],
        "scalars": chain["scalars"]}
    other = invariant_summaries(permuted)
    for name in summaries:
        assert np.allclose(summaries[name], other[name], atol=1e-9), name


def test_the_invariant_summary_set_is_the_registered_one():
    chain = fake_chain(20, frozen=False, seed=4)
    summaries = invariant_summaries(chain)
    assert set(summaries) <= set(INVARIANT_SUMMARY_NAMES)
    # co-clustering needs the occurrence labels, which this chain does not carry
    assert set(summaries) == set(INVARIANT_SUMMARY_NAMES) - {"co_clustering_mean"}
    rng = np.random.default_rng(41)
    chain["label_draws"] = rng.integers(0, 3, size=(20, 4, 12))
    assert set(invariant_summaries(chain)) == set(INVARIANT_SUMMARY_NAMES)


def test_co_clustering_is_invariant_to_relabelling_and_reads_the_same_pairs():
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 3, size=(30, 4, 12))
    series = co_clustering_series(labels)
    permutation = np.array([2, 0, 1])
    assert np.array_equal(co_clustering_series(permutation[labels]), series)
    assert series.shape == (30,)
    assert ((series >= 0.0) & (series <= 1.0)).all()


def test_the_heldout_schema_refuses_to_interpret_unconverged_chains():
    unconverged = compare_heldout({"converged": False, "heldout_nll_per_step": 1.9},
                                  {"converged": True, "heldout_nll_per_step": 1.8})
    assert unconverged["interpretable"] is False
    assert unconverged["status"].startswith("NOT INTERPRETED")

    converged = compare_heldout(
        {"converged": True, "heldout_nll_per_step": 1.90},
        {"converged": True, "heldout_nll_per_step": 1.91},
        oracle={"converged": True, "heldout_nll_per_step": 1.85},
        true_parameters={"converged": True, "heldout_nll_per_step": 1.83})
    assert converged["interpretable"] is True
    assert set(converged["rows"]) == {"local_move_kernel", "ffbs",
                                      "oracle_boundary_control", "true_parameter_oracle"}
    assert converged["local_minus_ffbs_absolute_gap"] == pytest.approx(0.01)
    assert converged["flag"] is None

    divergent = compare_heldout({"converged": True, "heldout_nll_per_step": 1.90},
                                {"converged": True, "heldout_nll_per_step": 2.30})
    assert divergent["flag"] is not None


def test_the_convergence_module_never_aligns_to_truth():
    """Identifiers only: prose may discuss truth alignment, code may not perform it.

    The scan is over the module's actual names — `ast` drops comments, and docstrings are
    string constants rather than identifiers — with the audit function itself excluded,
    since it necessarily names the symbols it forbids.
    """
    import ast

    tree = ast.parse(Path(diagnostics.__file__).read_text())
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.FunctionDef)
                         and node.name == "assert_no_truth_alignment")]
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    identifiers |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    identifiers |= {node.name for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    forbidden = {"skill_alignment", "hidden_true_labels", "hidden_true_boundaries",
                 "u_true", "true_keys", "align_to_truth"}
    assert not (identifiers & forbidden)


def test_the_truth_alignment_guard_catches_a_violation():
    audit = assert_no_truth_alignment("labels = skill_alignment(draw, truth, K)")
    assert audit["pass"] is False
    assert "skill_alignment" in audit["found"]


# ---------------------------------------------------------------- comparison schemas
def test_the_equal_sweep_schema_reports_missing_entries_rather_than_filling_them():
    local = {"wall_seconds": 100.0, "sweeps": 50_000, "beta_ess": 400.0}
    ffbs = {"wall_seconds": 150.0, "sweeps": 50_000, "beta_ess": 900.0}
    table = compare_equal_sweeps(local, ffbs)
    assert table["rows"]["beta_ess"] == {"local_move_kernel": 400.0, "ffbs": 900.0}
    assert table["rows"]["h_changes"]["ffbs"] is None
    assert "h_changes" in table["missing"]


def test_the_equal_time_schema_rescales_to_the_smaller_budget():
    local = {"wall_seconds": 100.0, "beta_ess": 400.0}
    ffbs = {"wall_seconds": 200.0, "beta_ess": 900.0}
    table = compare_equal_time(local, ffbs)
    assert table["budget_seconds"] == 100.0
    assert table["rows"]["beta_ess"]["local_move_kernel"] == pytest.approx(400.0)
    assert table["rows"]["beta_ess"]["ffbs"] == pytest.approx(450.0)
    assert "assumption" in table


# ------------------------------------------------------------------- the 7B2 guards
def load_stage7b2():
    path = ROOT / "scripts" / "stage7b2_full_joint_ffbs.py"
    spec = importlib.util.spec_from_file_location("stage7b2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_run_refuses_to_start_against_an_unfrozen_baseline():
    module = load_stage7b2()
    audit = module.baseline_freeze_audit()
    assert "frozen" in audit
    if not audit["frozen"]:
        with pytest.raises(SystemExit, match="not frozen"):
            module.require_frozen_baseline(audit)
    assert isinstance(audit["reasons"], list)


@pytest.mark.skipif(not (BASELINE / "corpus_manifest.json").exists(),
                    reason="the Stage 6E2 baseline corpus is not reachable")
def test_the_corpus_hash_is_checked_against_the_baseline_manifest():
    module = load_stage7b2()
    audit = module.corpus_audit()
    manifest = json.loads((BASELINE / "corpus_manifest.json").read_text())
    assert audit["baseline_corpus_hash"] == manifest["corpus_hash"]
    assert audit["n_train_traces"] == manifest["n_train_traces"] == 100
    assert audit["n_heldout_traces"] == manifest["n_heldout_traces"] == 45
    assert audit["rebuilt_corpus_hash"] == audit["baseline_corpus_hash"]
    assert audit["pass"] is True


@pytest.mark.skipif(not (BASELINE / "config.json").exists(),
                    reason="the Stage 6E2 baseline is not reachable")
def test_the_starting_states_are_constructed_the_same_way_as_the_baseline():
    module = load_stage7b2()
    audit = module.starting_state_audit()
    assert audit["n_chains"] == 4
    assert len(audit["hashes"]) == audit["n_chains"]
    assert len(set(audit["hashes"])) == audit["n_chains"], "starts must be distinct"
    assert audit["construction"] == "the Stage 6E2 dispersed-start constructor, reused"


def test_the_scales_and_kernels_are_the_baselines():
    module = load_stage7b2()
    audit = module.kernel_audit()
    assert audit["proposal_scales_match_baseline"] is True
    assert audit["retuned"] is False
    assert audit["only_change"] == "the (S, z) transition kernel"
