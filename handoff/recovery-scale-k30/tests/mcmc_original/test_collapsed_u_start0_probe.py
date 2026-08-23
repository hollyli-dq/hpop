"""Focused tests for the start[0] probe: frozen definitions, and end-to-end wiring.

The wiring test runs the probe's own worker, persistence, reload and analysis code on a
deliberately tiny configuration so that a failure surfaces in seconds here rather than
after four hours of chains. It changes lengths and windows ONLY — every statistic,
extraction rule and verdict rule is executed exactly as registered.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "collapsed_u_start0_probe", ROOT / "scripts" / "collapsed_u_start0_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_definitions():
    probe = _load_probe()
    assert probe.WINDOWS == {"early": (0, 16_000), "middle": (16_000, 32_000),
                             "late": (32_000, 48_000)}
    assert probe.PER_CHAIN_WINDOW_DRAWS == 2_000
    assert probe.BALANCED_PER_CHAIN == 1_000
    assert probe.CHAIN_SEEDS == (8_155_001, 8_155_002, 8_155_003, 8_155_004)
    assert not set(probe.CHAIN_SEEDS) & probe.PREVIOUSLY_USED
    assert probe.RESUME_CHECK_SEED not in probe.PREVIOUSLY_USED
    assert probe.COLLAPSED_EVERY == 10
    assert (probe.SWEEPS, probe.BURN_IN, probe.THIN) == (600_000, 120_000, 10)
    assert (probe.SWEEPS - probe.BURN_IN) // probe.THIN == probe.N_RETAINED


def test_window_slice_is_deterministic():
    probe = _load_probe()
    arr = np.arange(48_000)
    early = probe.window_slice(arr, "early", 8)
    assert early[0] == 0 and early[1] == 8 and len(early) == 2_000
    late = probe.window_slice(arr, "late", 16)
    assert late[0] == 32_000 and len(late) == 1_000


def test_end_to_end_wiring_on_a_tiny_configuration(tmp_path):
    probe = _load_probe()
    probe.OUT = tmp_path                    # never touch the real results directory
    probe.SWEEPS, probe.BURN_IN, probe.THIN = 300, 60, 2
    probe.N_RETAINED = 120
    probe.WINDOWS = {"early": (0, 40), "middle": (40, 80), "late": (80, 120)}
    probe.CHECKPOINT_EVERY = 0

    e1b, mixed, model, start0 = probe.problem()
    seven_b1 = probe._load("stage7b1",
                           ROOT / "scripts" / "stage7b1_mixed_reference_mcmc.py")
    registration = probe.write_registration(e1b, model, start0, seven_b1)
    assert registration["seeds_never_used_before"]
    assert (tmp_path / "registration.json").exists()
    assert (tmp_path / "start_state_manifest.json").exists()
    assert (tmp_path / "window_definitions.json").exists()
    manifest = json.loads((tmp_path / "start_state_manifest.json").read_text())
    assert manifest["start0_hash"] == registration["start0_hash"]

    (tmp_path / "checkpoints").mkdir(exist_ok=True)
    payloads = []
    for chain in range(probe.N_CHAINS):
        payloads.append(probe._chain_worker({
            "chain": chain, "start_hash": registration["start0_hash"],
            "checkpoint_path": str(tmp_path / "checkpoints"), "progress_every": 0}))
    assert all(len(p["log_target"]) == probe.N_RETAINED for p in payloads)
    assert len({p["start_hash"] for p in payloads}) == 1
    # different seeds must give different trajectories
    assert not np.array_equal(payloads[0]["u_draws"], payloads[1]["u_draws"])

    probe.save_raw(payloads)
    reloaded = probe.load_raw()
    assert np.array_equal(reloaded[2]["u_draws"], payloads[2]["u_draws"])
    assert reloaded[1]["collapsed_records"][0]["sweep"] == \
        payloads[1]["collapsed_records"][0]["sweep"]

    probe.analyze(reloaded, probe.Reference(e1b, mixed), registration,
                  {"pass": True, "note": "tiny wiring test"})
    for name in ("historical_gate.json", "per_chain_energy.json",
                 "chain_balanced_energy.json", "scalar_drift.json",
                 "structural_movement.json", "convergence.json", "verdict.json",
                 "correctness.json", "report.md"):
        assert (tmp_path / name).exists(), name
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["verdict"] in (
        "START-0 TRANSIENT / DIAGNOSTIC-WINDOW EXPLANATION SUPPORTED",
        "START-0 BASIN-SPECIFIC KERNEL INTERACTION SUPPORTED",
        "START-0 FOCUSED PROBE INCONCLUSIVE")
    assert set(verdict["a_conditions"]) == {
        "late_balanced_inside_envelope", "at_most_one_late_z_above_2",
        "median_late_z_leq_1p5", "beta_3of4_late_offset_se_below_2",
        "lambda_rep_3of4_late_offset_se_below_2", "beta_median_abs_contracts",
        "lambda_rep_median_abs_contracts", "controls_clean",
        "late_ess_finite_and_moving"}
    assert len(verdict["b_conditions"]) == 6
