"""The pilot's frozen pass rule, and the edge classification it depends on.

The rule was written and amended while blind to any pilot output. These tests exist so
that it cannot quietly change afterwards, and so that the one distinction most easily got
wrong -- two opposite situations both producing an undefined R-hat -- is pinned.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "u_budget_pilot", ROOT / "scripts" / "k_ladder" / "u_budget_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def chains_from(edge_series) -> list:
    """Build the per-chain dicts `pooled_diagnostics` expects from edge indicators."""
    arr = np.asarray(edge_series, dtype=float)          # (chains, draws, edges)
    return [{"draws": arr.shape[1], "relation_count": arr[c].sum(axis=1),
             "edge_indicators": arr[c], "insufficient": False}
            for c in range(arr.shape[0])]


# ------------------------------------- the two undefined-R-hat cases are opposites
def test_a_consensus_fixed_edge_is_not_a_failure():
    """All four chains hold the same value: they agree. R-hat is undefined, but nothing
    is wrong -- the posterior is simply certain about that relation."""
    e = np.ones((4, 20, 1))
    out = pilot.pooled_diagnostics(chains_from(e))
    assert out["edges_consensus_fixed"] == 1
    assert out["edges_chain_disagreeing_frozen"] == 0
    verdict = pilot.evaluate_pass_rule({**out, "u_acceptance_retained": 0.3,
                                        "closure_changing_fraction": 0.5})
    assert verdict["passes"], verdict["failures"]


def test_a_chain_disagreeing_frozen_edge_is_an_automatic_failure():
    """Two chains stuck at 0, two stuck at 1. Also undefined R-hat -- and a sampler that
    never escaped its starting mode."""
    e = np.concatenate([np.zeros((2, 20, 1)), np.ones((2, 20, 1))])
    out = pilot.pooled_diagnostics(chains_from(e))
    assert out["edges_chain_disagreeing_frozen"] == 1
    assert out["edges_consensus_fixed"] == 0
    verdict = pilot.evaluate_pass_rule({**out, "u_acceptance_retained": 0.3,
                                        "closure_changing_fraction": 0.5})
    assert not verdict["passes"]
    assert any("disagreeing" in f for f in verdict["failures"])


def test_the_two_cases_are_distinguished_when_both_are_present():
    consensus = np.ones((4, 20, 1))
    split = np.concatenate([np.zeros((2, 20, 1)), np.ones((2, 20, 1))])
    out = pilot.pooled_diagnostics(chains_from(np.concatenate([consensus, split], axis=2)))
    assert out["edges_consensus_fixed"] == 1
    assert out["edges_chain_disagreeing_frozen"] == 1


def test_a_partially_frozen_edge_is_counted_but_not_pooled():
    """Constant in some chains, moving in others. The v2 run showed why pooling these is
    wrong: split-chain within-variance is zero for the frozen chains, so rank-normalised
    split R-hat degenerates to ~1e16 and ESS to a ~4.3 floor -- numbers that carried no
    information and drowned every real edge. Counted as their own category, never pooled,
    and not a failure by themselves."""
    rng = np.random.default_rng(0)
    e = np.concatenate([np.zeros((2, 40, 1)),
                        rng.integers(0, 2, size=(2, 40, 1)).astype(float)])
    out = pilot.pooled_diagnostics(chains_from(e))
    assert out["edges_partially_frozen"] == 1
    assert out["edges_diagnosed"] == 0
    assert out["rhat_edge_max"] is None
    verdict = pilot.evaluate_pass_rule({**out, "u_acceptance_retained": 0.3,
                                        "closure_changing_fraction": 0.5})
    # The cell may still fail other gates (here, relation-count R-hat -- two chains
    # genuinely disagree with the other two, and that gate is right to say so). What the
    # fix guarantees is that no EDGE-level failure is manufactured from the frozen edge.
    assert not any("rhat_edge" in f or "ess" in f.lower()
                   for f in verdict["failures"]), verdict["failures"]


def test_the_v2_degeneracy_cannot_recur():
    """Regression built from the v2 run's structure: one partially-frozen edge next to
    one genuinely moving edge. In v2 the frozen one produced R-hat 2.3e16 and an ESS
    floor of 4.3 that masked the moving edge entirely. Now the pooled numbers must be
    the MOVING edge's numbers -- finite, and far below 1e15."""
    rng = np.random.default_rng(7)
    moving = rng.integers(0, 2, size=(4, 40, 1)).astype(float)
    partial = np.concatenate([np.zeros((3, 40, 1)),
                              rng.integers(0, 2, size=(1, 40, 1)).astype(float)])
    out = pilot.pooled_diagnostics(chains_from(np.concatenate([moving, partial], axis=2)))
    assert out["edges_partially_frozen"] == 1
    assert out["edges_diagnosed"] == 1
    assert out["rhat_edge_max"] is not None and out["rhat_edge_max"] < 1e3
    assert out["relation_level_ess_min"] is not None and out["relation_level_ess_min"] > 10


def test_no_finite_rhat_is_reported_for_an_all_frozen_set():
    out = pilot.pooled_diagnostics(chains_from(np.ones((4, 20, 3))))
    assert out["rhat_edge_max"] is None
    assert out["relation_level_ess_min"] is None
    assert out["edges_diagnosed"] == 0


# ------------------------------------------------- the failure thresholds themselves
@pytest.mark.parametrize("field,value,expected_pass", [
    ("rhat_relation_count", 1.04, True), ("rhat_relation_count", 1.06, False),
    ("rhat_edge_max", 1.05, True), ("rhat_edge_max", 1.20, False),
    ("relation_level_ess_min", 100.0, True), ("relation_level_ess_min", 99.0, False),
    ("u_acceptance_retained", 0.15, True), ("u_acceptance_retained", 0.14, False),
    ("u_acceptance_retained", 0.60, True), ("u_acceptance_retained", 0.61, False),
    ("closure_changing_fraction", 0.02, True), ("closure_changing_fraction", 0.019, False),
])
def test_each_threshold_bites_where_it_is_declared(field, value, expected_pass):
    cell = {"rhat_relation_count": 1.0, "rhat_edge_max": 1.0,
            "relation_level_ess_min": 500.0, "u_acceptance_retained": 0.3,
            "closure_changing_fraction": 0.5, "edges_chain_disagreeing_frozen": 0}
    cell[field] = value
    assert pilot.evaluate_pass_rule(cell)["passes"] is expected_pass


def test_missing_diagnostics_do_not_silently_pass_a_frozen_chain_set():
    """A cell with nothing computable must still fail on disagreeing edges."""
    cell = {"edges_chain_disagreeing_frozen": 3}
    assert not pilot.evaluate_pass_rule(cell)["passes"]


# ---------------------------------------------------- the frozen scale-selection rule
def _cell(scale, ess_min, ess_median=None, passes=True, replicate=0):
    return {"u_scale": scale, "replicate": replicate,
            "relation_level_ess_min": ess_min,
            "relation_level_ess_median": ess_median if ess_median is not None
            else ess_min * 2,
            "pass_rule_verdict": {"passes": passes, "failures": []}}


def _by_scale(*groups):
    """groups: (scale, [cells]) pairs -> the dict select_scale expects."""
    return {scale: cells for scale, cells in groups}


def test_the_scale_with_the_best_worst_case_ess_is_selected():
    chosen = pilot.select_scale(_by_scale(
        (0.25, [_cell(0.25, 120.0, replicate=0), _cell(0.25, 130.0, replicate=1)]),
        (0.5, [_cell(0.5, 400.0, replicate=0), _cell(0.5, 380.0, replicate=1)]),
        (1.0, [_cell(1.0, 200.0, replicate=0), _cell(1.0, 210.0, replicate=1)])))
    assert chosen["selected_u_scale"] == 0.5
    assert chosen["worst_case_relation_ess"] == 380.0


def test_the_score_is_worst_cased_over_replicates_not_averaged():
    """A scale that mixes beautifully in one replicate and badly in the other must not
    be rescued by the good one."""
    chosen = pilot.select_scale(_by_scale(
        (0.25, [_cell(0.25, 300.0, replicate=0), _cell(0.25, 290.0, replicate=1)]),
        (0.5, [_cell(0.5, 900.0, replicate=0), _cell(0.5, 110.0, replicate=1)])))
    assert chosen["selected_u_scale"] == 0.25


def test_a_scale_passing_in_only_one_replicate_is_ineligible():
    chosen = pilot.select_scale(_by_scale(
        (0.25, [_cell(0.25, 999.0, replicate=0),
                _cell(0.25, 999.0, replicate=1, passes=False)]),
        (0.5, [_cell(0.5, 150.0, replicate=0), _cell(0.5, 150.0, replicate=1)])))
    assert chosen["selected_u_scale"] == 0.5
    assert chosen["eligible_scales"] == [0.5]


def test_a_scale_present_in_only_one_replicate_is_ineligible():
    """Missing the second replicate is not the same as passing it."""
    chosen = pilot.select_scale(_by_scale(
        (0.25, [_cell(0.25, 999.0, replicate=0)]),
        (0.5, [_cell(0.5, 150.0, replicate=0), _cell(0.5, 150.0, replicate=1)])))
    assert chosen["selected_u_scale"] == 0.5


def test_ties_within_tolerance_break_by_median_then_ascending_scale():
    chosen = pilot.select_scale(_by_scale(
        (0.25, [_cell(0.25, 396.0, 800.0, replicate=0),
                _cell(0.25, 396.0, 800.0, replicate=1)]),
        (0.5, [_cell(0.5, 400.0, 800.0, replicate=0),
               _cell(0.5, 400.0, 800.0, replicate=1)])))
    assert chosen["selected_u_scale"] == 0.25       # within 10%, smaller wins
    assert chosen["tied_within_tolerance"] == [0.25, 0.5]


def test_a_scale_outside_tolerance_wins_on_merit_not_size():
    chosen = pilot.select_scale(_by_scale(
        (0.25, [_cell(0.25, 200.0, replicate=0), _cell(0.25, 200.0, replicate=1)]),
        (0.5, [_cell(0.5, 400.0, replicate=0), _cell(0.5, 400.0, replicate=1)])))
    assert chosen["selected_u_scale"] == 0.5


def test_no_eligible_scale_selects_nothing():
    chosen = pilot.select_scale(_by_scale(
        (0.5, [_cell(0.5, 100.0, passes=False), _cell(0.5, 100.0, passes=False)])))
    assert chosen["selected_u_scale"] is None


def test_the_selection_score_is_hardware_independent():
    """An earlier draft maximised order-changing moves per SECOND, which would let a
    faster machine choose a different kernel. Nothing timing-related may appear."""
    import inspect

    src = inspect.getsource(pilot.select_scale)
    for forbidden in ("second", "seconds", "per_second", "time", "runtime", "rss"):
        assert forbidden not in src.lower().replace("hardware-independent", ""), \
            f"{forbidden!r} appeared in the scale selection rule"


# ------------------------------------------------------- the rule's own commitments
def test_a_failed_candidate_set_terminates_the_pilot():
    text = pilot.PASS_RULE["if_none_pass"]
    assert "TERMINATES" in text
    assert "fresh CRN streams" in text and "new pilot version" in text


def test_no_adaptive_rerun_on_the_same_streams():
    """The full factorial is already executed, so the completed grid is evaluated in
    registered order rather than re-run when a rung fails."""
    text = pilot.PASS_RULE["global_X"]
    assert "nothing is re-run" in text
    assert "50 -> 100 -> 166.7" in text
    assert "terminates unsuccessfully" in text


def test_replicates_are_never_pooled_for_diagnostics():
    text = pilot.PASS_RULE["replicates"]
    assert "NEVER pooled across replicates" in text
    assert "BOTH replicates pass" in text
    assert "eight-chain R-hat" in text


def test_u_scale_is_tuned_at_the_rung_never_carried_over():
    assert "never carried over from a cheaper one" in pilot.PASS_RULE["per_rung_u_scale"]


def test_wall_clock_is_excluded_from_the_statistical_choice():
    assert "never a reason to prefer a smaller X" in pilot.PASS_RULE["never"]
    assert "recovery against the sealed U is not an input" in pilot.PASS_RULE["never"]
