"""Condition C terminalization: quarantine, seal/unseal, artifact integrity.

Run:  PYTHONPATH=src .venv/bin/python -m unittest tests.mcmc_original.test_condition_c_terminal -v
"""

from __future__ import annotations

import json
import pathlib
import unittest

import numpy as np

from hpop.mcmc_original.matched_condition_c_prime import (
    SealedTruth, SealedTruthError,
)
from hpop.mcmc_original import matched_synthetic_generator as msg

ROOT = pathlib.Path(__file__).resolve().parents[2]
TERM = ROOT / "results/mcmc_original/condition_c_terminal_75k"
C_DIR = ROOT / "results/mcmc_original/matched_condition_c"
CHAINS = C_DIR / "formal_chains"

TERMINAL_SWEEP, BURN_IN, THIN = 75_000, 10_000, 5
EXPECTED_DRAWS = (TERMINAL_SWEEP - BURN_IN) // THIN     # 13,000


class TestQuarantine(unittest.TestCase):
    def setUp(self):
        self.q = json.loads((TERM / "quarantine_manifest.json").read_text())

    def test_primary_draw_count_matches_the_indexing_rule(self):
        self.assertEqual(self.q["draws_used_per_chain"], EXPECTED_DRAWS)
        self.assertEqual(self.q["terminal_sweep_for_primary_analysis"],
                         TERMINAL_SWEEP)

    def test_every_chain_has_its_excess_recorded_and_nothing_deleted(self):
        self.assertFalse(self.q["deleted"])
        self.assertEqual(len(self.q["per_chain"]), 8)
        for name, row in self.q["per_chain"].items():
            path = CHAINS / f"{name}.npz"
            self.assertTrue(path.exists(), f"{name} checkpoint must be kept")
            n = len(np.load(str(path))["log_target"])
            self.assertEqual(row["retained_draws_total"], n)
            self.assertEqual(row["post_75k_draws_quarantined"],
                             n - EXPECTED_DRAWS)
            self.assertGreaterEqual(row["post_75k_draws_quarantined"], 0)

    def test_the_quarantine_label_is_explicit(self):
        self.assertIn("NOT USED IN PRIMARY", self.q["label"])

    def test_last_primary_draw_is_at_or_before_the_terminal_sweep(self):
        """Draw i maps to sweep 10000+(i+1)*5; draw 12,999 is exactly 75,000."""
        self.assertEqual(BURN_IN + EXPECTED_DRAWS * THIN, TERMINAL_SWEEP)
        self.assertGreater(BURN_IN + (EXPECTED_DRAWS + 1) * THIN,
                           TERMINAL_SWEEP)


class TestRegisteredArtifactsUnchanged(unittest.TestCase):
    def test_all_six_registered_gates_exist_and_failed(self):
        for rung in (30_000, 50_000, 75_000):
            for arm in ("cond", "marg"):
                path = C_DIR / f"formal_gate_{arm}_{rung}.json"
                self.assertTrue(path.exists())
                self.assertFalse(json.loads(path.read_text())["pass"],
                                 f"{arm}@{rung} must be recorded as FAIL")

    def test_recorded_hashes_match_the_files_on_disk(self):
        hashes = json.loads((TERM / "artifact_hashes.json").read_text())
        import hashlib
        checked = 0
        for rel, digest in hashes["hashes"].items():
            path = ROOT / rel
            if not path.exists() or "condition_c_terminal_75k" in rel:
                continue
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), digest,
                f"{rel} changed after the terminal freeze")
            checked += 1
        self.assertGreater(checked, 10)

    def test_registration_records_the_launch_state(self):
        reg = json.loads((C_DIR / "formal_registration.json").read_text())
        self.assertEqual(reg["parent_commit"], "50eee50")
        self.assertEqual(reg["checkpoints"], [30000, 50000, 75000, 100000])


class TestTerminalVerdict(unittest.TestCase):
    def setUp(self):
        self.v = json.loads((TERM / "terminal_verdict.json").read_text())

    def test_formal_verdict_is_not_converged_for_both_arms(self):
        a = self.v["A_REGISTERED_FORMAL_VERDICT"]
        self.assertEqual(a["C-COND"], "NOT CONVERGED")
        self.assertEqual(a["C-MARG"], "NOT CONVERGED")

    def test_deviation_is_disclosed_with_the_registered_reason(self):
        text = self.v["early_termination_deviation"]
        self.assertIn("before the preregistered 100k ceiling", text)
        self.assertIn("two consecutive passing checkpoints", text)
        self.assertIn("no longer attainable", text)

    def test_diagnosis_is_separated_from_the_formal_verdict(self):
        self.assertIn("A_REGISTERED_FORMAL_VERDICT", self.v)
        self.assertIn("B_SCIENTIFIC_DIAGNOSIS", self.v)
        self.assertEqual(self.v["B_SCIENTIFIC_DIAGNOSIS"]["phenomenon"],
                         "anchored structure-to-skill assignment multimodality")


class TestForbiddenLanguage(unittest.TestCase):
    FORBIDDEN = ("label switching", "converged up to permutation",
                 "solved joint inference", "cannot cross the barrier")

    NEGATORS = ("not ", "never ", "no ", "rather than", "avoid", "forbidden",
                "do not")

    def test_terminal_documents_avoid_forbidden_wording(self):
        """A forbidden phrase may appear ONLY inside an explicit disavowal.

        The reports are required to say the phenomenon is *not* label
        switching, so a bare substring check would forbid the very
        clarification the protocol demands. Each occurrence must therefore be
        immediately preceded by a negation.
        """
        for path in sorted(TERM.glob("*.md")):
            if path.name == "CONDITION_C_PAPER_LEDGER.md":
                continue              # the ledger lists them as prohibitions
            text = path.read_text().lower()
            for phrase in self.FORBIDDEN:
                start = 0
                while (i := text.find(phrase, start)) != -1:
                    window = text[max(0, i - 40):i]
                    self.assertTrue(
                        any(n in window for n in self.NEGATORS),
                        f"{path.name}: '{phrase}' used affirmatively "
                        f"near: ...{text[max(0, i-60):i+len(phrase)]}")
                    start = i + len(phrase)

    def test_the_ledger_lists_them_as_prohibitions(self):
        text = (TERM / "CONDITION_C_PAPER_LEDGER.md").read_text()
        self.assertIn("Forbidden wording", text)
        for phrase in self.FORBIDDEN:
            self.assertIn(phrase, text.lower())


class TestRecoveryDiscipline(unittest.TestCase):
    def setUp(self):
        self.r = json.loads((TERM / "recovery_75k.json").read_text())

    def test_recovery_carries_the_non_converged_warning(self):
        self.assertIn("not interpreted as posterior estimates",
                      self.r["NON_CONVERGED_WARNING"])
        for arm in ("cond", "marg"):
            self.assertIn("WARNING", self.r["pooled_per_arm"][arm])

    def test_recovery_is_reported_per_chain_and_per_arm(self):
        self.assertEqual(len(self.r["per_chain"]), 8)
        for arm in ("cond", "marg"):
            self.assertEqual(
                len(self.r["per_arm"][arm]["per_chain_mean_closure_f1"]), 4)

    def test_recovery_used_only_terminal_draws(self):
        self.assertIn(str(EXPECTED_DRAWS), self.r["draws_used"])
        self.assertIn(str(TERMINAL_SWEEP), self.r["draws_used"])

    def test_path_marginal_recovery_is_flagged_supplementary(self):
        for arm in ("cond", "marg"):
            self.assertIn("SUPPLEMENTARY",
                          self.r["supplementary_path_marginal"][arm]["STATUS"])


class TestTruthSeal(unittest.TestCase):
    def test_sealed_truth_still_blocks_before_unsealing(self):
        sealed = SealedTruth(msg.supplied_truth())
        with self.assertRaises(SealedTruthError):
            sealed.u_by_skill
        self.assertAlmostEqual(float(sealed.beta), 1.5)

    def test_unsealing_is_explicit(self):
        sealed = SealedTruth(msg.supplied_truth())
        sealed.unseal("Condition C terminated")
        self.assertEqual(np.asarray(sealed.u_by_skill).shape, (3, 5, 2))


class TestDiagnosticReproducibility(unittest.TestCase):
    def test_assignment_gap_trajectory_is_consistent_across_artifacts(self):
        geom = json.loads((TERM / "condition_c_failure_geometry.json").read_text())
        table = json.loads((TERM / "condition_c_checkpoint_table.json").read_text())
        from_geom = {g["rung"]: g["assignment_gap_nats"]
                     for g in geom["assignment_gap_trajectory"]}
        for row in table:
            if row["arm"] == "marg":
                self.assertAlmostEqual(from_geom[row["rung"]],
                                       row["assignment_gap_nats"], places=2)

    def test_gap_is_stable_in_the_registered_band(self):
        geom = json.loads((TERM / "condition_c_failure_geometry.json").read_text())
        gaps = [g["assignment_gap_nats"]
                for g in geom["assignment_gap_trajectory"]]
        self.assertEqual(len(gaps), 3)
        for g in gaps:
            self.assertGreater(g, 124.0)
            self.assertLess(g, 125.0)

    def test_marg_shares_one_library_and_cond_does_not(self):
        geom = json.loads((TERM / "condition_c_failure_geometry.json").read_text())
        self.assertEqual(
            geom["arms"]["marg"]["structure"]["distinct_unordered_libraries"], 1)
        self.assertGreater(
            geom["arms"]["cond"]["structure"]["distinct_unordered_libraries"], 1)

    def test_path_marginal_accepts_are_not_assignment_crossings(self):
        geom = json.loads((TERM / "condition_c_failure_geometry.json").read_text())
        marg = geom["arms"]["marg"]
        self.assertGreater(marg["path_marginal_accepts_total"], 100)
        self.assertEqual(
            marg["anchored_assignment_changes_after_burn_in_total"], 0)


class TestChronologyAndDisclosure(unittest.TestCase):
    def test_cprime_is_labelled_as_a_followup_not_a_preregistration(self):
        chron = json.loads(
            (TERM / "condition_c_cprime_chronology.json").read_text())
        label = chron["cprime_labelling"]
        self.assertIn("prospectively frozen follow-up", label)
        self.assertIn("NOT preregistered before Condition C", label)

    def test_blinding_disclosure_present_and_specific(self):
        text = (TERM / "condition_c_blinding_disclosure.md").read_text()
        self.assertIn("incidentally recognised", text)
        self.assertIn("No recovery metric was computed before termination",
                      text)

    def test_resume_continuity_records_both_interruptions(self):
        cont = json.loads(
            (TERM / "condition_c_resume_continuity.json").read_text())
        self.assertEqual(len(cont["interruptions"]), 2)
        for row in cont["interruptions"]:
            self.assertTrue(row["rng_state_restored"])
            self.assertTrue(row["gate_history_unchanged"])
            self.assertEqual(row["retained_draws_lost"], 0)
            self.assertFalse(row["scientific_state_changed"])


class TestConditionCPrimeUntouchedAtFreeze(unittest.TestCase):
    """C' had never been launched AT THE MOMENT Condition C was frozen.

    This is a historical fact about the terminal commit, and it is checked
    against the committed artifact. It deliberately does NOT read the working
    copy of the dry-run file: Condition C' was subsequently launched under
    explicit authorisation, which rewrites that file, and a live-state check
    would then fail for a reason that has nothing to do with the integrity of
    the Condition C freeze.
    """

    TERMINAL_COMMIT = "c693d85"

    def test_cprime_was_unlaunched_in_the_terminal_commit(self):
        import subprocess
        blob = subprocess.run(
            ["git", "show",
             f"{self.TERMINAL_COMMIT}:results/mcmc_original/"
             "matched_condition_c_prime/condition_c_prime_runner_dry_run.json"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(blob.returncode, 0, "terminal commit must be readable")
        dry = json.loads(blob.stdout)
        self.assertFalse(dry["would_launch"])
        self.assertEqual(dry["chains_started"], 0)

    def test_chronology_records_cprime_as_unlaunched_at_the_freeze(self):
        chron = json.loads(
            (TERM / "condition_c_cprime_chronology.json").read_text())
        status = [e for e in chron["events"]
                  if e["event"] == "Condition C' status"]
        self.assertEqual(len(status), 1)
        self.assertIn("UNLAUNCHED", status[0]["detail"])


if __name__ == "__main__":
    unittest.main()
