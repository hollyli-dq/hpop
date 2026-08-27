"""Seal tests for the terminal-only FULL-LATENT recovery entry point.

The driver must refuse before it imports model/truth code or follows any recovery
path.  These tests use only tiny JSON gate artifacts; they deliberately never create
a corpus or truth manifest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import scipy


ROOT = Path(__file__).resolve().parents[2]
DRIVER_PATH = ROOT / "scripts" / "run_matched_full_latent_recovery.py"


@pytest.fixture()
def driver_module():
    spec = importlib.util.spec_from_file_location("full_latent_recovery_driver_test",
                                                  DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def _gate(out: Path, arm: str, checkpoint: int, passed: bool) -> None:
    _write_json(out / f"formal_gate_{arm.lower().replace('-', '_')}_{checkpoint}.json", {
        "arm": arm,
        "checkpoint": checkpoint,
        "chain_iterations": [checkpoint] * 4,
        "pass": passed,
    })


def _terminal(out: Path, arm: str, checkpoint: int, history: list[bool]) -> None:
    _write_json(out / f"terminal_{arm.lower().replace('-', '_')}.json", {
        "arm": arm,
        "terminal_checkpoint": checkpoint,
        "reason": "two consecutive PASS checkpoints",
        "gate_history": history,
        "truth_unsealed": False,
    })


def _valid_terminal_artifacts(out: Path) -> None:
    for arm in ("FULL-COND", "FULL-MARG"):
        _gate(out, arm, 30_000, False)
        _gate(out, arm, 50_000, True)
        _gate(out, arm, 75_000, True)
        _terminal(out, arm, 75_000, [False, True, True])
    _write_json(out / "formal_status.json", {
        "terminal": {"FULL-COND": True, "FULL-MARG": True},
        "truth_unsealed": False,
    })


def test_recovery_refuses_before_any_truth_loader_can_run(tmp_path, driver_module,
                                                           monkeypatch):
    called = False

    def truth_loader(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("truth loader must not run before terminal verification")

    monkeypatch.setattr(driver_module, "_load_truth_after_terminal", truth_loader)
    with pytest.raises(RuntimeError, match="terminal_full_cond"):
        driver_module.run_terminal_recovery(out=tmp_path, corpus_dir=tmp_path,
                                            workers=1)
    assert not called


def test_terminal_certificate_requires_two_sealed_arm_records(tmp_path, driver_module):
    _valid_terminal_artifacts(tmp_path)
    certificate = driver_module.assert_both_arms_terminal(tmp_path)
    assert certificate["truth_unsealed_before_recovery"] is False
    assert certificate["arms"]["FULL-COND"]["terminal_checkpoint"] == 75_000
    assert certificate["arms"]["FULL-MARG"]["gate_history"] == [False, True, True]

    (tmp_path / "terminal_full_marg.json").unlink()
    with pytest.raises(RuntimeError, match="terminal_full_marg"):
        driver_module.assert_both_arms_terminal(tmp_path)


def test_terminal_certificate_rejects_nonterminal_early_gate(tmp_path, driver_module):
    for arm in ("FULL-COND", "FULL-MARG"):
        _gate(tmp_path, arm, 30_000, False)
        _gate(tmp_path, arm, 50_000, False)
        _terminal(tmp_path, arm, 50_000, [False, False])
    with pytest.raises(RuntimeError, match="two consecutive PASS"):
        driver_module.assert_both_arms_terminal(tmp_path)


def test_unseal_started_event_is_durable_and_binds_terminal_certificate(tmp_path,
                                                                        driver_module):
    _valid_terminal_artifacts(tmp_path)
    certificate = driver_module.assert_both_arms_terminal(tmp_path)
    event = driver_module.record_terminal_recovery_unseal_started(tmp_path, certificate)
    assert event["truth_unsealed"] is True
    assert event["truth_opened_at_record_time"] is False
    assert (tmp_path / "terminal_recovery_unseal_started.json").exists()
    assert driver_module.record_terminal_recovery_unseal_started(tmp_path, certificate) == event
    with pytest.raises(RuntimeError, match="different terminal artifacts"):
        driver_module.record_terminal_recovery_unseal_started(
            tmp_path, {**certificate, "arms": {"different": {}}})


def test_unseal_event_and_frozen_launch_check_precede_any_model_import(tmp_path,
                                                                       driver_module,
                                                                       monkeypatch):
    _valid_terminal_artifacts(tmp_path)
    imported = False

    def post_terminal_imports():
        nonlocal imported
        imported = True
        raise AssertionError("model import must occur after frozen-launch validation")

    monkeypatch.setattr(driver_module, "_post_terminal_imports", post_terminal_imports)
    with pytest.raises(RuntimeError, match="launch_manifest"):
        driver_module.run_terminal_recovery(out=tmp_path, corpus_dir=tmp_path,
                                            workers=1)
    assert not imported
    assert (tmp_path / "terminal_recovery_unseal_started.json").exists()


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE)


def _frozen_launch_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    out = repo / "results" / "mcmc_original" / "matched_full_latent"
    driver = repo / "scripts" / "run_matched_full_latent_recovery.py"
    auxiliary = repo / "src" / "frozen_dependency.py"
    out.mkdir(parents=True)
    driver.parent.mkdir(parents=True)
    auxiliary.parent.mkdir(parents=True)
    driver.write_text("# frozen terminal driver\n")
    auxiliary.write_text("# frozen dependency\n")
    source_hashes = {
        "scripts/run_matched_full_latent_recovery.py": hashlib.sha256(
            driver.read_bytes()).hexdigest(),
        "src/frozen_dependency.py": hashlib.sha256(auxiliary.read_bytes()).hexdigest(),
    }
    pairing = {str(index): {"FULL-COND": "a" * 64, "FULL-MARG": "b" * 64}
               for index in range(4)}
    _write_json(out / "launch_manifest.json", {
        "source_commit": "placeholder",
        "source_hashes": source_hashes,
        "runtime": {"python": sys.version, "numpy": np.__version__,
                    "scipy": scipy.__version__},
        "starts": {"pairing": pairing},
        "chain_seeds": {
            "FULL-COND": [1, 2, 3, 4],
            "FULL-MARG": [5, 6, 7, 8],
        },
    })
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "FULL-LATENT test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "freeze launch")
    return repo, out, driver


def test_frozen_launch_integrity_checks_head_worktree_and_recovery_driver(tmp_path,
                                                                            driver_module):
    repo, out, recovery_driver = _frozen_launch_repo(tmp_path)
    verified = driver_module.assert_frozen_launch_integrity(out, root=repo)
    assert verified["verified_source_paths"] == [
        "scripts/run_matched_full_latent_recovery.py", "src/frozen_dependency.py",
    ]
    assert verified["chain_seeds"]["FULL-MARG"] == [5, 6, 7, 8]

    recovery_driver.write_text("# local post-launch edit\n")
    with pytest.raises(RuntimeError, match="disk hash differs"):
        driver_module.assert_frozen_launch_integrity(out, root=repo)

    recovery_driver.write_text("# frozen terminal driver\n")
    manifest = out / "launch_manifest.json"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(RuntimeError, match="differs from its committed HEAD blob"):
        driver_module.assert_frozen_launch_integrity(out, root=repo)


def test_frozen_launch_integrity_refuses_runtime_drift(tmp_path, driver_module):
    repo, out, _ = _frozen_launch_repo(tmp_path)
    manifest = out / "launch_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["runtime"]["scipy"] = "not-the-frozen-runtime"
    _write_json(manifest, payload)
    _git(repo, "add", "results/mcmc_original/matched_full_latent/launch_manifest.json")
    _git(repo, "commit", "-qm", "freeze mismatched runtime")
    with pytest.raises(RuntimeError, match="Python/NumPy/SciPy runtime differs"):
        driver_module.assert_frozen_launch_integrity(out, root=repo)


def test_heldout_worker_scores_exact_retained_u_pi_p_draws_without_truth(driver_module):
    """The worker only needs held-out observations and retained posterior draws."""
    from hpop.mcmc_original import matched_full_latent as mfl

    fixed = mfl.FullLatentFixed()
    traces = ((0, 1, 2, 3, 4, 0),)
    model = mfl.build_full_latent_model(traces, fixed)
    pi, transition = mfl.draw_initial_pi_p(model, 131)
    u = mfl.make_u_start(0, 132, 0.5, fixed)
    result = driver_module._heldout_log_z_worker(
        (0, np.stack([u]), np.stack([pi]), np.stack([transition]), traces,
         fixed.as_dict())
    )
    assert result["log_z"].shape == (1, 1)
    assert np.isfinite(result["log_z"]).all()


def test_runtime_summary_reports_per_chain_and_arm_wall_clock_rates(driver_module):
    class State:
        iteration = 20

    class Chain:
        def __init__(self, seconds):
            self.seconds = seconds
            self.state = State()

    summary = driver_module._runtime_summary([
        {"index": 0, "chain": Chain(10.0)},
        {"index": 1, "chain": Chain(20.0)},
    ])
    assert summary["per_chain"][0]["seconds_per_sweep"] == pytest.approx(0.5)
    assert summary["per_chain"][1]["sweeps_per_wall_clock_hour"] == pytest.approx(3600.0)
    assert summary["wall_clock_proxy_seconds"] == 20.0
    assert summary["arm_sweeps_per_wall_clock_hour"] == pytest.approx(3600.0)
