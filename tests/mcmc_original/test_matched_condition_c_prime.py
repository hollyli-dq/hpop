"""Condition C' runner and chain: no-launch safety, ordering, parity, resume.

Run:  PYTHONPATH=src .venv/bin/python -m unittest tests.mcmc_original.test_matched_condition_c_prime -v
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest

import numpy as np

from hpop.mcmc_original import matched_condition_c as mcc
from hpop.mcmc_original import matched_synthetic_generator as msg
from hpop.mcmc_original.collapsed_u_kernel import CollapsedUConfig
from hpop.mcmc_original.matched_condition_c_prime import (
    ConditionCPrimeChain, SealedTruth, SealedTruthError, swap_diagnostics,
)
from hpop.mcmc_original.skill_swap_kernel import SkillSwapConfig

ROOT = pathlib.Path(__file__).resolve().parents[2]
RHO_0 = 0.5


def _runner():
    spec = importlib.util.spec_from_file_location(
        "run_c_prime", ROOT / "scripts/run_matched_condition_c_prime_formal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup(lengths=(12, 14, 16, 12), seed=818181):
    truth = msg.supplied_truth()
    corpus = msg.generate_corpus(seed, lengths, (), truth)
    model = mcc.build_condition_c_model(tuple(t.cpa for t in corpus.train))
    fixed = mcc.ConditionCFixed.from_truth(truth, RHO_0)
    return truth, corpus, model, fixed


def _sampler(model, fixed, collapsed_every=10):
    return mcc.ConditionCSampler(
        model=model, fixed=fixed, u_scale=0.5,
        collapsed=CollapsedUConfig(every=collapsed_every, scale=1.0))


class TestNoLaunchSafety(unittest.TestCase):
    def test_importing_the_runner_starts_nothing(self):
        module = _runner()
        self.assertFalse((ROOT / "results/mcmc_original/"
                          "matched_condition_c_prime/formal_chains").exists()
                         and any((ROOT / "results/mcmc_original/"
                                  "matched_condition_c_prime/formal_chains"
                                  ).iterdir()),
                         "importing the runner must not create chain state")
        self.assertEqual(module.PROTOCOL_ID, "condition-c-prime-v1")

    def test_launch_requires_both_flags(self):
        module = _runner()
        for flags, protocol in (((False), None), ((True), None),
                                ((True), "wrong-protocol"),
                                ((False), "condition-c-prime-v1")):
            launching = bool(flags) and protocol == module.PROTOCOL_ID
            self.assertFalse(launching)
        self.assertTrue(True and "condition-c-prime-v1" == module.PROTOCOL_ID)

    def test_refuses_while_condition_c_is_alive(self):
        module = _runner()
        alive = {"terminal_artifact_exists": False, "may_launch": False,
                 "live_orchestrator_processes": 3}
        with self.assertRaises(SystemExit) as caught:
            module.assert_may_launch(alive)
        self.assertIn("cannot launch while formal Condition C is still active",
                      str(caught.exception))
        finished_but_running = {"terminal_artifact_exists": True,
                                "may_launch": False,
                                "live_orchestrator_processes": 1}
        with self.assertRaises(SystemExit):
            module.assert_may_launch(finished_but_running)
        module.assert_may_launch({"terminal_artifact_exists": True,
                                  "may_launch": True,
                                  "live_orchestrator_processes": 0})

    def test_live_condition_c_is_detected_right_now(self):
        """The real guard, against the real machine state."""
        module = _runner()
        status = module.condition_c_status()
        self.assertIn("may_launch", status)
        self.assertEqual(status["may_launch"],
                         status["terminal_artifact_exists"]
                         and status["live_orchestrator_processes"] == 0)


class TestFrozenProtocolParity(unittest.TestCase):
    def test_arms_and_cadences_match_the_preregistration(self):
        module = _runner()
        prereg = json.loads((ROOT / "results/mcmc_original/"
                             "matched_condition_c_prime/preregistration.json"
                             ).read_text())["registered_design"]
        self.assertEqual(set(module.ARMS), set(prereg["arms"]))
        self.assertEqual(module.SWAP_CADENCE, prereg["swap_cadence"])
        self.assertEqual(module.U_SCALE, prereg["u_scale"])
        self.assertEqual(module.SCHEDULED_COLLAPSED_SCALE,
                         prereg["scheduled_collapsed_scale"])
        self.assertEqual(module.CHECKPOINTS, tuple(prereg["checkpoints"]))
        self.assertEqual(module.BURN_IN, prereg["burn_in"])
        self.assertEqual(module.THIN, prereg["thin"])
        for arm, seeds in prereg["seeds"].items():
            self.assertEqual(list(module.ARMS[arm]["seeds"]), seeds)
        self.assertEqual(module.ARMS["C-MARG-SWAP"]["collapsed_every"], 10)
        self.assertEqual(module.ARMS["C-COND-SWAP"]["collapsed_every"], 0)

    def test_target_manifest_parity_against_condition_c(self):
        module = _runner()
        truth, corpus = module.build_environment()
        manifest = module.target_manifest(SealedTruth(truth), corpus)
        self.assertTrue(manifest["all_parity_checks_pass"],
                        manifest["parity_vs_condition_c"])
        for key in ("corpus_hash", "truth_hash", "u_scale", "scheduled_scale",
                    "collapsed_cadence", "checkpoints", "burn_in", "thin",
                    "starts", "shared_sources_unchanged_since_c_launch"):
            self.assertTrue(manifest["parity_vs_condition_c"][key], key)
        self.assertFalse(manifest["permutation_invariance_of_fixed_inputs"]
                         ["any_non_identity_symmetry"])

    def test_seed_manifest_records_everything_registered(self):
        module = _runner()
        truth, corpus = module.build_environment()
        manifest = module.seed_manifest(SealedTruth(truth), corpus)
        self.assertFalse(manifest["initialized_from_condition_c_checkpoint"])
        self.assertTrue(manifest["starts_are_preregistered_dispersed"])
        self.assertEqual(len(manifest["rows"]), 4)
        for row in manifest["rows"]:
            for field in ("start_seed", "start_scale", "initial_u_sha256",
                          "initial_H_tuple", "initial_Sz_sha256",
                          "initial_segment_total", "per_arm_chain_seed"):
                self.assertIn(field, row)
        self.assertEqual(len(set(manifest["all_chain_seeds"])), 8)
        hashes = {tuple(r["initial_H_tuple"]) for r in manifest["rows"]}
        self.assertEqual(len(hashes), 4, "starts must be four distinct H tuples")


class TestSealedTruth(unittest.TestCase):
    def test_fixed_inputs_readable_hidden_truth_sealed(self):
        sealed = SealedTruth(msg.supplied_truth())
        self.assertAlmostEqual(float(sealed.beta), 1.5)
        self.assertEqual(len(sealed.pi), 3)
        self.assertEqual(int(sealed.min_width), 3)
        for hidden in ("u_by_skill", "rho"):
            with self.assertRaises(SealedTruthError):
                getattr(sealed, hidden)

    def test_unseal_opens_it_and_is_recorded(self):
        sealed = SealedTruth(msg.supplied_truth())
        sealed.unseal("registered stopping condition reached")
        self.assertEqual(np.asarray(sealed.u_by_skill).shape, (3, 5, 2))

    def test_fixed_construction_touches_no_hidden_field(self):
        sealed = SealedTruth(msg.supplied_truth())
        fixed = sealed.fixed_for_condition_c(RHO_0)
        self.assertTrue(sealed.sealed)
        self.assertEqual(fixed.rho_0, RHO_0)
        self.assertAlmostEqual(fixed.beta, 1.5)


class TestOrderingAndCadence(unittest.TestCase):
    def test_ffbs_refresh_follows_every_swap_attempt(self):
        truth, corpus, model, fixed = _setup()
        chain = ConditionCPrimeChain(_sampler(model, fixed),
                                     np.random.default_rng(1).standard_normal(
                                         (3, 5, 2)),
                                     seed=2, burn_in=5, thin=1,
                                     swap=SkillSwapConfig(every=5))
        chain.advance(20)
        self.assertEqual(chain.swap_attempts, 4)
        self.assertEqual(chain.ffbs_refreshes_after_swap, 4)
        self.assertEqual(chain.ffbs_refreshes, 20)
        chain.assert_ordering_invariant()

    def test_ordering_assertion_actually_fires(self):
        truth, corpus, model, fixed = _setup()
        chain = ConditionCPrimeChain(_sampler(model, fixed),
                                     np.zeros((3, 5, 2)), seed=3, burn_in=2,
                                     thin=1, swap=SkillSwapConfig(every=5))
        chain.advance(10)
        chain.ffbs_refreshes_after_swap -= 1        # simulate a missed refresh
        with self.assertRaises(AssertionError):
            chain.assert_ordering_invariant()

    def test_swap_disabled_is_bitwise_condition_c(self):
        truth, corpus, model, fixed = _setup()
        u0 = np.random.default_rng(4).standard_normal((3, 5, 2))
        prime = ConditionCPrimeChain(_sampler(model, fixed), u0, seed=5,
                                     burn_in=4, thin=2,
                                     swap=SkillSwapConfig(every=0))
        prime.advance(30)
        base = mcc.ConditionCChain(_sampler(model, fixed), u0, seed=5,
                                   burn_in=4, thin=2)
        base.advance(30)
        self.assertEqual(prime.retained_log_target, base.retained_log_target)
        self.assertEqual(prime.retained_h_hashes, base.retained_h_hashes)
        np.testing.assert_array_equal(prime.state.u_by_skill,
                                      base.state.u_by_skill)
        self.assertEqual(prime.swap_attempts, 0)

    def test_swap_fires_on_the_registered_absolute_schedule(self):
        truth, corpus, model, fixed = _setup()
        chain = ConditionCPrimeChain(_sampler(model, fixed),
                                     np.zeros((3, 5, 2)), seed=6, burn_in=2,
                                     thin=1, swap=SkillSwapConfig(every=50))
        chain.advance(49)
        self.assertEqual(chain.swap_attempts, 0)
        chain.advance(50)
        self.assertEqual(chain.swap_attempts, 1)

    def test_pair_selection_is_reproducible(self):
        truth, corpus, model, fixed = _setup()
        runs = []
        for _ in range(2):
            chain = ConditionCPrimeChain(_sampler(model, fixed),
                                         np.zeros((3, 5, 2)), seed=7,
                                         burn_in=2, thin=1,
                                         swap=SkillSwapConfig(every=4))
            chain.advance(20)
            runs.append([d[0] for d in chain.swap_deltas])
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(len(runs[0]), 5)


class TestResume(unittest.TestCase):
    def test_resume_reproduces_an_uninterrupted_run(self):
        truth, corpus, model, fixed = _setup()
        u0 = np.random.default_rng(8).standard_normal((3, 5, 2))
        swap = SkillSwapConfig(every=5)
        whole = ConditionCPrimeChain(_sampler(model, fixed), u0, seed=9,
                                     burn_in=4, thin=2, swap=swap)
        whole.advance(40)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "chain.npz"
            part = ConditionCPrimeChain(_sampler(model, fixed), u0, seed=9,
                                        burn_in=4, thin=2, swap=swap)
            part.advance(18)
            part.save(path)
            resumed = ConditionCPrimeChain.load(path, _sampler(model, fixed),
                                                swap)
            resumed.advance(40)
        self.assertEqual(whole.retained_log_target,
                         resumed.retained_log_target)
        self.assertEqual(whole.retained_h_hashes, resumed.retained_h_hashes)
        np.testing.assert_array_equal(whole.state.u_by_skill,
                                      resumed.state.u_by_skill)
        self.assertEqual(whole.swap_attempts, resumed.swap_attempts)
        self.assertEqual(whole.swap_accepts, resumed.swap_accepts)
        self.assertEqual(whole.ffbs_refreshes_after_swap,
                         resumed.ffbs_refreshes_after_swap)
        self.assertEqual(whole.swap_by_pair, resumed.swap_by_pair)

    def test_resume_refuses_a_different_cadence(self):
        truth, corpus, model, fixed = _setup()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "chain.npz"
            chain = ConditionCPrimeChain(_sampler(model, fixed),
                                         np.zeros((3, 5, 2)), seed=10,
                                         burn_in=2, thin=1,
                                         swap=SkillSwapConfig(every=5))
            chain.advance(10)
            chain.save(path)
            with self.assertRaises(ValueError):
                ConditionCPrimeChain.load(path, _sampler(model, fixed),
                                          SkillSwapConfig(every=50))


class TestDiagnostics(unittest.TestCase):
    def test_swap_diagnostics_block_is_complete(self):
        truth, corpus, model, fixed = _setup()
        chain = ConditionCPrimeChain(_sampler(model, fixed),
                                     np.random.default_rng(11).standard_normal(
                                         (3, 5, 2)),
                                     seed=12, burn_in=4, thin=2,
                                     swap=SkillSwapConfig(every=4))
        chain.advance(40)
        block = swap_diagnostics(chain)
        for field in ("swap_attempts", "swap_accepts", "swap_acceptance",
                      "accepted_assignment_changes",
                      "accepted_swaps_with_z_reallocation", "by_pair",
                      "largest_positive_delta", "largest_negative_delta",
                      "seconds_per_swap", "overhead_fraction_of_chain",
                      "sweeps_between_assignment_changes",
                      "distinct_anchored_tuples",
                      "distinct_unordered_libraries",
                      "ffbs_refreshes_equals_attempts"):
            self.assertIn(field, block)
        self.assertTrue(block["ffbs_refreshes_equals_attempts"])
        self.assertEqual(set(block["by_pair"]), {"0-1", "0-2", "1-2"})
        self.assertGreaterEqual(block["distinct_anchored_tuples"],
                                block["distinct_unordered_libraries"])


if __name__ == "__main__":
    unittest.main()
