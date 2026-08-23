"""Protocol tests for the sealed FULL-LATENT formal-runner orchestration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _runner_module():
    root = Path(__file__).parents[2]
    path = root / "scripts" / "run_matched_full_latent_formal.py"
    spec = importlib.util.spec_from_file_location("full_latent_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resume_never_recomputes_an_existing_gate(tmp_path, monkeypatch):
    """A persisted 30k PASS/FAIL is one observation, never a second gate vote."""
    runner = _runner_module()
    runner.OUT = tmp_path
    runner.CHAIN_DIR = tmp_path / "formal_chains"
    runner.CHECKPOINTS = (30, 50, 75)
    runner.ARMS = {"TEST-ARM": (1, 2, 3, 4)}
    runner._assert_launch_manifest = lambda: {}  # launcher integrity is tested separately

    runner._write_json(runner._gate_path("TEST-ARM", 30), {
        "arm": "TEST-ARM", "checkpoint": 30, "chain_iterations": [30] * 4,
        "pass": False,
    })
    runner.CHAIN_DIR.mkdir(exist_ok=True)
    for index in range(4):
        runner._chain_path("TEST-ARM", index).touch()
    iterations = {"TEST-ARM": [30] * 4}
    calls = []

    def advance(arms, upto, workers):
        calls.append((tuple(arms), int(upto), int(workers)))
        for arm in arms:
            iterations[arm] = [int(upto)] * 4

    def gate(arm, checkpoint):
        return {"arm": arm, "checkpoint": int(checkpoint),
                "chain_iterations": [int(checkpoint)] * 4, "pass": False,
                "formal_truth_free_verdict": "FAIL"}

    runner.advance_formal = advance
    runner._chain_iterations = lambda arm: list(iterations[arm])
    runner.arm_gate = gate

    runner.run_formal(workers=1)

    assert calls == [(('TEST-ARM',), 50, 1), (('TEST-ARM',), 75, 1)]
    assert runner._gate_history("TEST-ARM") == [False, False, False]
    # The ceiling terminal record means a second launcher invocation is a no-op too.
    runner.run_formal(workers=1)
    assert calls == [(('TEST-ARM',), 50, 1), (('TEST-ARM',), 75, 1)]


def test_resume_refuses_to_backfill_a_gate_from_a_later_chain_state(tmp_path, monkeypatch):
    runner = _runner_module()
    runner.OUT = tmp_path
    runner.CHAIN_DIR = tmp_path / "formal_chains"
    runner.CHECKPOINTS = (30, 50)
    runner.ARMS = {"TEST-ARM": (1, 2, 3, 4)}
    runner._assert_launch_manifest = lambda: {}
    runner._chain_iterations = lambda arm: [50] * 4

    with pytest.raises(RuntimeError, match="advanced beyond its missing 30 gate"):
        runner.run_formal(workers=1)


def test_resume_requires_checkpoint_lineage_for_an_existing_gate(tmp_path):
    """A gate cannot be continued from a missing or differently-aged checkpoint."""
    runner = _runner_module()
    runner.OUT = tmp_path
    runner.CHAIN_DIR = tmp_path / "formal_chains"
    runner.CHECKPOINTS = (30, 50)
    runner.ARMS = {"TEST-ARM": (1, 2, 3, 4)}
    runner._assert_launch_manifest = lambda: {}
    runner._write_json(runner._gate_path("TEST-ARM", 30), {
        "arm": "TEST-ARM", "checkpoint": 30, "chain_iterations": [30] * 4,
        "pass": False,
    })

    with pytest.raises(RuntimeError, match="missing its checkpoint"):
        runner.run_formal(workers=1)

    runner.CHAIN_DIR.mkdir(exist_ok=True)
    for index in range(4):
        runner._chain_path("TEST-ARM", index).touch()
    runner._chain_iterations = lambda arm: [30, 30, 20, 30]

    with pytest.raises(RuntimeError, match="does not match its last recorded 30 gate"):
        runner.run_formal(workers=1)
