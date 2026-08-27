"""Step 7B1 — the frozen Stage 6E1B reference, and the comparison built on it.

Step 7B1 builds no reference. It reads the one Stage 6E1B froze, and the risk that
introduces is a quiet substitution: a reference whose quality gates were never met, a
reference for a different target, or one that has drifted since it was frozen. These tests
check the artifact rather than the handoff.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "results" / "mcmc_original" / "stage6e1b_mixed_reference"
STAGE7B1 = ROOT / "results" / "mcmc_original" / "stage7b1_mixed_reference"

# The values the Step 7B1 script requires of the frozen reference, restated here so a
# silent replacement of the artifact fails a test rather than only a run.
EXPECTED = {
    "max_rqmc_standard_error": 5.667929455901159e-04,
    "max_half_width_95": 1.2080905663045393e-03,
    "min_relative_ess": 4.386141741129308e-02,
    "max_normalised_weight": 3.295848775089024e-04,
    "log_evidence_sd": 3.3823943089510656e-03,
}

pytestmark = pytest.mark.skipif(
    not (FROZEN / "reference_registration.json").exists(),
    reason="the frozen Stage 6E1B reference is not present in this worktree")


def registration() -> dict:
    return json.loads((FROZEN / "reference_registration.json").read_text())


def test_the_reference_passed_its_own_active_gates():
    payload = registration()
    assert payload["primary_pass"] is True
    assert payload["all_active_pass"] is True
    for name, check in payload["checks"].items():
        assert check["pass"] is True, f"{name} did not pass in the frozen reference"


def test_the_reference_quality_statistics_are_the_expected_ones():
    checks = registration()["checks"]
    for name, expected in EXPECTED.items():
        assert abs(checks[name]["value"] - expected) < 1e-12, name


def test_the_two_superseded_diagnostics_are_still_reported_as_failing():
    """They were superseded in Stage 6D1. They must not be revived as gates, and their
    failure must not be quietly relabelled."""
    superseded = registration()["superseded_checks"]
    assert set(superseded) == {"max_replicate_h_total_variation",
                               "max_replicate_relation_departure"}
    for name, check in superseded.items():
        assert check["pass"] is False, f"{name} is recorded as passing"
        assert "SUPERSEDED" in check["status"]


def test_the_reference_problem_is_nondegenerate():
    nondegeneracy = registration()["nondegeneracy"]
    assert all(check["pass"] for check in nondegeneracy.values())
    assert nondegeneracy["segmentation_max_probability"]["value"] < 0.90
    assert nondegeneracy["induced_h_states_above_0.01"]["value"] >= 3


def test_no_relabelling_is_a_symmetry_of_the_reference_target():
    conclusion = registration()["label_permutation_audit"]["conclusion"]
    assert "no nontrivial relabelling is a symmetry" in conclusion


def test_the_reference_target_fixes_pi_and_P():
    """Step 7B1 must fix them too; inferring them would be a different posterior."""
    config = json.loads((FROZEN / "config.json").read_text())
    assert "pi" in config["fixed"] and "P" in config["fixed"]
    assert set(config["latent"]) == {"S", "z", "U", "rho", "beta", "omega", "lambda_rep",
                                     "lambda_back"}
    assert config["problem"]["pi_fixed"] == [0.6, 0.3, 0.1]


def test_the_reference_draws_carry_every_array_the_comparison_reads():
    draws = np.load(FROZEN / "reference_draws.npz")
    for name in ("segmentation_conditional", "segmentation_sampled", "boundary", "labels",
                 "segment_counts", "relation", "state_ends", "state_labels",
                 "retained_closures", "retained_sampled"):
        assert name in draws, name
    for k in range(3):
        assert f"h_probability_skill{k}" in draws
        assert f"h_keys_skill{k}" in draws
    conditional = draws["segmentation_conditional"]
    assert conditional.shape == (2, 21)
    assert np.allclose(conditional.sum(axis=1), 1.0)


def test_the_loader_used_by_the_script_agrees_with_these_files():
    import importlib.util
    path = ROOT / "scripts" / "stage7b1_mixed_reference_mcmc.py"
    spec = importlib.util.spec_from_file_location("stage7b1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audit = module.verify_frozen_reference()
    assert audit["pass"] is True
    assert audit["max_drift_from_expected"] == 0.0
    assert set(audit["sha256"]) >= {"reference_draws.npz", "reference_registration.json",
                                    "qmc_summary.json"}


@pytest.mark.skipif(not (STAGE7B1 / "joint_comparison.json").exists(),
                    reason="Step 7B1 has not been run in this worktree")
def test_the_step7b1_comparison_uses_the_stage6e1b_gate_set():
    """The comparison must be the frozen one — same gate names, same thresholds."""
    ours = json.loads((STAGE7B1 / "joint_comparison.json").read_text())
    baseline = json.loads((FROZEN / "joint_comparison.json").read_text())
    assert set(ours["gates"]) == set(baseline["gates"])
    for name, gate in ours["gates"].items():
        if name == "mixed_multivariate_reference_statistic":
            continue                      # its threshold is a calibrated envelope
        assert gate["threshold"] == baseline["gates"][name]["threshold"], name


@pytest.mark.skipif(not (STAGE7B1 / "config.json").exists(),
                    reason="Step 7B1 has not been run in this worktree")
def test_step7b1_changed_only_the_segmentation_kernel():
    config = json.loads((STAGE7B1 / "config.json").read_text())
    baseline = json.loads((FROZEN / "config.json").read_text())
    assert config["chains"]["sweeps"] == baseline["chains"]["sweeps"]
    assert config["chains"]["burn_in"] == baseline["chains"]["burn_in"]
    assert config["chains"]["thin"] == baseline["chains"]["thin"]
    assert config["chains"]["scales"] == baseline["chains"]["scales"]
    assert config["chains"]["n_chains"] == baseline["chains"]["n_chains"]
    assert config["target"]["fixed"][:2] == ["pi", "P"]
    # new seeds, and none of the baseline's
    assert not set(config["chains"]["seeds"]) & set(baseline["chains"]["seeds"])
    assert config["segmentation_update"]["acceptance_probability"] == 1.0
    assert config["segmentation_update"]["hastings_correction"] == "none"
