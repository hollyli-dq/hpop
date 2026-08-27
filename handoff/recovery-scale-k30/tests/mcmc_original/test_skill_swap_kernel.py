"""Condition C' skill-swap move: exactness of the ratio and safety of the reuse.

Run:  PYTHONPATH=src .venv/bin/python -m unittest tests.mcmc_original.test_skill_swap_kernel -v
"""

from __future__ import annotations

import math
import pathlib
import unittest

import numpy as np

from hpop.mcmc_original import matched_condition_c as mcc
from hpop.mcmc_original import matched_synthetic_generator as msg
from hpop.mcmc_original import skill_swap_kernel as ssk
from hpop.mcmc_original.collapsed_u_kernel import CollapsedUConfig
from hpop.mcmc_original.stage6c_frozen import log_structural_prior

ROOT = pathlib.Path(__file__).resolve().parents[2]
RHO_0 = 0.5


def _setup(lengths=(12, 14, 16, 12), seed=717171):
    truth = msg.supplied_truth()
    corpus = msg.generate_corpus(seed, lengths, (), truth)
    model = mcc.build_condition_c_model(tuple(t.cpa for t in corpus.train))
    fixed = mcc.ConditionCFixed.from_truth(truth, RHO_0)
    sampler = mcc.ConditionCSampler(
        model=model, fixed=fixed, u_scale=0.5,
        collapsed=CollapsedUConfig(every=0, scale=1.0))
    return truth, corpus, model, fixed, sampler


class TestProposalAlgebra(unittest.TestCase):
    def test_swap_is_an_involution(self):
        rng = np.random.default_rng(1)
        u = rng.standard_normal((3, 5, 2))
        for j, k in ssk.unordered_pairs(3):
            once = ssk.swap_skills(u, j, k)
            twice = ssk.swap_skills(once, j, k)
            np.testing.assert_array_equal(twice, u)
            self.assertFalse(np.array_equal(once, u))
            np.testing.assert_array_equal(once[j], u[k])
            np.testing.assert_array_equal(once[k], u[j])

    def test_prior_difference_is_exactly_zero(self):
        rng = np.random.default_rng(2)
        for _ in range(5):
            u = rng.standard_normal((3, 5, 2))
            base = sum(log_structural_prior(u[i], RHO_0) for i in range(3))
            for j, k in ssk.unordered_pairs(3):
                swapped = ssk.swap_skills(u, j, k)
                after = sum(log_structural_prior(swapped[i], RHO_0)
                            for i in range(3))
                self.assertLess(abs(after - base), 1e-12)

    def test_uniform_pair_choice_makes_hastings_zero(self):
        """Symmetric proposal: every pair equally likely, and the map is its own
        inverse, so q(U->U') = q(U'->U) structurally."""
        rng = np.random.default_rng(3)
        pairs = ssk.unordered_pairs(3)
        counts = {p: 0 for p in pairs}
        for _ in range(6000):
            counts[pairs[int(rng.integers(len(pairs)))]] += 1
        for value in counts.values():
            self.assertLess(abs(value / 6000 - 1 / 3), 0.02)


class TestAcceptanceRatio(unittest.TestCase):
    def setUp(self):
        (self.truth, self.corpus, self.model, self.fixed,
         self.sampler) = _setup()

    def test_log_alpha_is_antisymmetric_under_the_involution(self):
        """log alpha(U->U') = -log alpha(U'->U): the detailed-balance identity
        for a symmetric involution with an exchangeable prior."""
        likelihood = self.sampler.collapsed_likelihood
        u = np.random.default_rng(4).standard_normal((3, 5, 2))
        state = mcc.initial_condition_c_state(self.model, u, self.fixed)
        forward, _ = ssk.skill_swap_mh_step(
            state, likelihood, np.random.default_rng(5), pairs=[(0, 2)])
        _, record_f = ssk.skill_swap_mh_step(
            state, likelihood, np.random.default_rng(5), pairs=[(0, 2)])
        swapped = state.copy()
        swapped.u_by_skill = ssk.swap_skills(state.u_by_skill, 0, 2)
        _, record_b = ssk.skill_swap_mh_step(
            swapped, likelihood, np.random.default_rng(5), pairs=[(0, 2)])
        self.assertLess(abs(record_f["log_alpha"] + record_b["log_alpha"]),
                        1e-8)

    def test_delta_matches_an_independent_full_rebuild(self):
        likelihood = self.sampler.collapsed_likelihood
        u = np.random.default_rng(6).standard_normal((3, 5, 2))
        state = mcc.initial_condition_c_state(self.model, u, self.fixed)
        _, record = ssk.skill_swap_mh_step(
            state, likelihood, np.random.default_rng(7), pairs=[(1, 2)])
        swapped = state.copy()
        swapped.u_by_skill = ssk.swap_skills(state.u_by_skill, 1, 2)
        independent = (float(likelihood.full_rebuild_log_z(swapped).sum())
                       - float(likelihood.full_rebuild_log_z(state).sum()))
        self.assertLess(abs(record["d_log_lik_collapsed"] - independent), 1e-8)

    def test_cache_is_exact_after_accept_and_after_reject(self):
        likelihood = self.sampler.collapsed_likelihood
        u = np.random.default_rng(8).standard_normal((3, 5, 2))
        state = mcc.initial_condition_c_state(self.model, u, self.fixed)
        for seed in range(6):
            new_state, record = ssk.skill_swap_mh_step(
                state, likelihood, np.random.default_rng(seed))
            cached = likelihood.log_z_per_trace(new_state)
            fresh = likelihood.full_rebuild_log_z(new_state)
            self.assertLess(float(np.abs(cached - fresh).max()), 1e-9,
                            f"cache stale after {'accept' if record['accepted'] else 'reject'}")

    def test_accepts_decisively_from_an_inferior_assignment(self):
        """A state holding the registered structures under a transposed
        assignment must be repaired in one accepted move.

        Needs enough data for the assignment to be identified: the mode gap
        grows with the corpus (about -1 nat at 4 traces, +12 at 16, ~125 on the
        100-trace formal corpus), so this uses 16 traces rather than the small
        fixture the algebraic tests share.
        """
        truth, corpus, model, fixed, sampler = _setup(
            lengths=tuple((24, 32, 40, 48)[i % 4] for i in range(16)))
        likelihood = sampler.collapsed_likelihood
        good = np.array(truth.u_by_skill, dtype=float, copy=True)
        bad = ssk.swap_skills(good, 0, 2)
        state = mcc.initial_condition_c_state(model, bad, fixed)
        _, record = ssk.skill_swap_mh_step(
            state, likelihood, np.random.default_rng(9), pairs=[(0, 2)])
        self.assertGreater(record["log_alpha"], 5.0)
        self.assertTrue(record["accepted"])
        self.assertTrue(record["assignment_changed"])
        # and the reverse move out of the good assignment is strongly rejected
        good_state = mcc.initial_condition_c_state(model, good, fixed)
        _, back = ssk.skill_swap_mh_step(
            good_state, likelihood, np.random.default_rng(9), pairs=[(0, 2)])
        self.assertLess(back["log_alpha"], -5.0)
        self.assertLess(abs(record["log_alpha"] + back["log_alpha"]), 1e-8)


class TestFixedInputsAreNotSymmetric(unittest.TestCase):
    def test_no_non_identity_permutation_is_a_symmetry(self):
        truth = msg.supplied_truth()
        report = ssk.permutation_invariance_report(truth.pi, truth.transition)
        self.assertFalse(report["any_non_identity_symmetry"])
        self.assertGreater(report["min_delta_pi_over_non_identity"], 0.1)
        self.assertGreater(report["min_delta_P_over_non_identity"], 0.1)
        transposition = next(r for r in report["rows"]
                             if r["permutation"] == [2, 1, 0])
        self.assertAlmostEqual(transposition["delta_pi_l2"], 0.4243, places=3)
        self.assertAlmostEqual(transposition["delta_P_frobenius"], 0.9798,
                               places=3)

    def test_a_symmetric_pi_P_would_be_detected(self):
        """Negative control: on an exchangeable (pi, P) the report says so."""
        pi = np.full(3, 1 / 3)
        transition = (np.ones((3, 3)) - np.eye(3)) / 2.0
        report = ssk.permutation_invariance_report(pi, transition)
        self.assertTrue(report["any_non_identity_symmetry"])


class TestComposition(unittest.TestCase):
    def test_every_zero_is_bitwise_condition_c(self):
        truth, corpus, model, fixed, sampler = _setup()
        u = np.random.default_rng(10).standard_normal((3, 5, 2))
        state = mcc.initial_condition_c_state(model, u, fixed)
        a, _, _, info_a = ssk.condition_c_swap_sweep_once(
            state, sampler, ssk.SkillSwapConfig(every=0),
            np.random.default_rng(11))
        sampler2 = mcc.ConditionCSampler(
            model=model, fixed=fixed, u_scale=0.5,
            collapsed=CollapsedUConfig(every=0, scale=1.0))
        b, _, info_b = mcc.condition_c_sweep_once(state, sampler2,
                                                  np.random.default_rng(11))
        np.testing.assert_array_equal(a.u_by_skill, b.u_by_skill)
        self.assertEqual(info_a["log_target"], info_b["log_target"])

    def test_swap_fires_on_schedule_and_ffbs_follows_it(self):
        self.assertEqual([i for i in range(120) if ssk.is_swap_sweep(i, 50)],
                         [49, 99])
        source = (ROOT / "src/hpop/mcmc_original/skill_swap_kernel.py"
                  ).read_text()
        sweep = source[source.index("def condition_c_swap_sweep_once"):]
        self.assertLess(sweep.index("skill_swap_mh_step("),
                        sweep.index("condition_c_sweep_once("))

    def test_validated_modules_are_not_modified(self):
        """The move composes by CALL; no validated module is edited or copied."""
        source = (ROOT / "src/hpop/mcmc_original/skill_swap_kernel.py"
                  ).read_text()
        for token in ("def forward(", "def backward_sample(",
                      "class CollapsedULikelihood", "def collapsed_u_mh_step",
                      "def condition_c_sweep_once("):
            self.assertNotIn(token, source)
        for expected in ("from hpop.mcmc_original.collapsed_u_likelihood import",
                         "from hpop.mcmc_original.matched_condition_c import"):
            self.assertIn(expected, source)


if __name__ == "__main__":
    unittest.main()
