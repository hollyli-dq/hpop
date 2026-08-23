"""Matched generator: truth validation, generation order, label parity, and the
source-level ban on the old Stage 6E2 block-count mechanism.

Run:  PYTHONPATH=src .venv/bin/python -m unittest tests.mcmc_original.test_matched_synthetic_generator -v
"""

from __future__ import annotations

import math
import pathlib
import unittest

import numpy as np

from hpop.mcmc_original import matched_synthetic_generator as msg
from hpop.mcmc_original.matched_segmentation_prior import width_sampling_tables
from hpop.mcmc_original.recurrent_rfs import sample_recurrent_rfs_sequence

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "hpop" / "mcmc_original"


class TestSuppliedTruth(unittest.TestCase):
    def test_matches_registered_stage6e2_truth(self):
        """The restated supplied truth must equal the registered Stage 6E2 truth.

        The old corpus module is imported HERE ONLY, to prove provenance of the
        values; the production generator modules never touch it (see the
        source-level regression test below).
        """
        from hpop.mcmc_original import stage6e_corpus as old
        truth = msg.supplied_truth()
        np.testing.assert_array_equal(truth.u_by_skill, old.U_TRUE_BY_SKILL)
        np.testing.assert_array_equal(truth.pi, old.PI_TRUE)
        np.testing.assert_array_equal(truth.transition, old.P_TRUE)
        self.assertEqual(truth.scalars, old.SCALAR_TRUTH)
        self.assertEqual(truth.epsilon, old.EPSILON_TRUE)

    def test_validate_supplied_truth(self):
        report = msg.validate_truth(msg.supplied_truth())
        self.assertTrue(report["orders_are_strict_partial_orders"])
        self.assertTrue(report["transition_diagonal_exactly_zero"])
        self.assertTrue(report["role_maps_injective"])

    def test_negative_control_self_transitions_detected(self):
        """Negative control 4: nonzero P[k, k] must fail truth validation."""
        base = msg.supplied_truth()
        bad_p = base.transition.copy()
        bad_p[0, 0] = 0.10
        bad_p[0] = bad_p[0] / bad_p[0].sum()
        bad = msg.MatchedTruth(
            u_by_skill=base.u_by_skill, pi=base.pi, transition=bad_p,
            beta=base.beta, omega=base.omega, lambda_rep=base.lambda_rep,
            lambda_back=base.lambda_back, epsilon=base.epsilon,
            delta_b=base.delta_b, min_width=base.min_width,
            max_width=base.max_width, role_maps=base.role_maps)
        with self.assertRaises(AssertionError):
            msg.validate_truth(bad)

    def test_rejects_non_injective_role_map(self):
        base = msg.supplied_truth()
        bad = msg.MatchedTruth(
            u_by_skill=base.u_by_skill, pi=base.pi, transition=base.transition,
            beta=base.beta, omega=base.omega, lambda_rep=base.lambda_rep,
            lambda_back=base.lambda_back, epsilon=base.epsilon,
            delta_b=base.delta_b, min_width=base.min_width,
            max_width=base.max_width,
            role_maps=((0, 1, 2, 3, 3),) + base.role_maps[1:])
        with self.assertRaises(AssertionError):
            msg.validate_truth(bad)

    def test_rejects_unnormalized_pi_and_p_rows(self):
        base = msg.supplied_truth()
        bad_pi = msg.MatchedTruth(
            u_by_skill=base.u_by_skill, pi=np.array([0.5, 0.3, 0.3]),
            transition=base.transition, beta=base.beta, omega=base.omega,
            lambda_rep=base.lambda_rep, lambda_back=base.lambda_back,
            epsilon=base.epsilon, delta_b=base.delta_b,
            min_width=base.min_width, max_width=base.max_width,
            role_maps=base.role_maps)
        with self.assertRaises(AssertionError):
            msg.validate_truth(bad_pi)
        bad_row = base.transition.copy()
        bad_row[1, 0] += 0.05
        bad_p = msg.MatchedTruth(
            u_by_skill=base.u_by_skill, pi=base.pi, transition=bad_row,
            beta=base.beta, omega=base.omega, lambda_rep=base.lambda_rep,
            lambda_back=base.lambda_back, epsilon=base.epsilon,
            delta_b=base.delta_b, min_width=base.min_width,
            max_width=base.max_width, role_maps=base.role_maps)
        with self.assertRaises(AssertionError):
            msg.validate_truth(bad_p)


class TestPriorDrawMode(unittest.TestCase):
    def test_prior_draw_produces_valid_truth(self):
        rng = np.random.default_rng(21)
        truth = msg.sample_prior_truth(rng)
        self.assertEqual(truth.mode, "prior_draw")
        self.assertIsNotNone(truth.rho)
        self.assertTrue(0.0 < truth.rho < 0.995)
        msg.validate_truth(truth)          # H legality, pi, P, scalars, ell

    def test_prior_draws_differ_across_seeds(self):
        a = msg.sample_prior_truth(np.random.default_rng(1))
        b = msg.sample_prior_truth(np.random.default_rng(2))
        self.assertFalse(np.array_equal(a.u_by_skill, b.u_by_skill))


class TestGenerationOrder(unittest.TestCase):
    def test_trace_invariants(self):
        truth = msg.supplied_truth()
        corpus = msg.generate_corpus(2026, [24, 32, 40, 48], [24, 32], truth)
        for trace in corpus.train + corpus.heldout:
            self.assertEqual(sum(trace.widths), trace.length)
            self.assertEqual(len(trace.cpa), trace.length)
            self.assertEqual(len(trace.widths), len(trace.labels))
            self.assertEqual(len(trace.widths), len(trace.role_blocks))
            for width, block in zip(trace.widths, trace.role_blocks):
                self.assertEqual(len(block), width)
                self.assertTrue(truth.min_width <= width <= truth.max_width)
            for prev, nxt in zip(trace.labels[:-1], trace.labels[1:]):
                self.assertNotEqual(prev, nxt)     # zero-diagonal P
            # identity ell: observed CPA is the concatenated role sequence
            flat = [r for block in trace.role_blocks for r in block]
            self.assertEqual(list(trace.cpa), flat)
            self.assertEqual(trace.boundaries,
                             tuple(np.cumsum(trace.widths)[:-1].tolist()))

    def test_blocks_are_independent_of_other_blocks(self):
        """Same block-specific stream => same block, whatever came before it.

        Regenerating any block alone (no other block generated first) must give
        byte-identical roles: proof that no recurrent state and no RNG state
        leaks across block boundaries.
        """
        truth = msg.supplied_truth()
        corpus = msg.generate_corpus(77, [32, 40], [24], truth)
        params = truth.rfs_parameters()
        for trace in corpus.train + corpus.heldout:
            blocks = list(zip(trace.widths, trace.labels, trace.role_blocks))
            for order in (range(len(blocks)), reversed(range(len(blocks)))):
                for l in order:
                    width, skill, recorded = blocks[l]
                    rng = msg.block_rng(corpus.master_seed, trace.split,
                                        trace.trace_index, l)
                    alone = sample_recurrent_rfs_sequence(
                        rng, width, truth.u_by_skill[skill], params)
                    self.assertEqual(tuple(alone), recorded,
                                     f"block {l} of trace {trace.trace_index} "
                                     "depends on generation order")

    def test_q0_reset_state_leak_detector(self):
        """Every generated block replays exactly from q_0 = 0 under the production
        log-likelihood; a block continuing another block's state does not."""
        from hpop.mcmc_original.recurrent_rfs import recurrent_rfs_log_likelihood
        truth = msg.supplied_truth()
        corpus = msg.generate_corpus(78, [40], [], truth)
        trace = corpus.train[0]
        for block, skill, recorded_ll in zip(trace.role_blocks, trace.labels,
                                             trace.block_log_likelihoods):
            replay = recurrent_rfs_log_likelihood(
                block, truth.u_by_skill[skill], truth.beta, truth.epsilon,
                truth.omega, truth.lambda_rep, truth.lambda_back)
            self.assertLess(abs(replay - recorded_ll), 1e-10)


class TestSkillLabelParity(unittest.TestCase):
    N = 60_000

    def test_initial_distribution(self):
        truth = msg.supplied_truth()
        rng = np.random.default_rng(31)
        counts = np.zeros(truth.n_skills)
        for _ in range(self.N):
            counts[msg.sample_initial_skill(rng, truth.pi)] += 1
        self.assertLess(float(np.abs(counts / self.N - truth.pi).max()), 0.01)

    def test_transition_rows_and_zero_self_transitions(self):
        truth = msg.supplied_truth()
        rng = np.random.default_rng(32)
        for h in range(truth.n_skills):
            counts = np.zeros(truth.n_skills)
            for _ in range(self.N):
                counts[msg.sample_next_skill(rng, truth.transition, h)] += 1
            self.assertEqual(counts[h], 0.0)          # exactly zero self-transitions
            self.assertLess(
                float(np.abs(counts / self.N - truth.transition[h]).max()), 0.01)


class TestOldMechanismBanned(unittest.TestCase):
    """Source-level regression: the new generator cannot contain, call, or
    indirectly reuse the old Stage 6E2 block-count mechanism."""

    PRODUCTION_SOURCES = (
        "matched_segmentation_prior.py",
        "matched_synthetic_generator.py",
        "matched_generator_diagnostics.py",
    )
    FORBIDDEN = (
        "BLOCKS_PER_TRACE",              # the old registered constant
        "stage6e_corpus",                # any import of the old generator module
        "(4, 5, 6)", "[4, 5, 6]",        # the old block-count support
        "(4,5,6)", "[4,5,6]",
        "n_blocks",                      # a block count sampled before segmentation
    )

    def test_forbidden_tokens_absent(self):
        for name in self.PRODUCTION_SOURCES:
            source = (SRC / name).read_text()
            for token in self.FORBIDDEN:
                self.assertNotIn(token, source, f"{name} contains '{token}'")

    def test_production_reuse_is_present(self):
        source = (SRC / "matched_synthetic_generator.py").read_text()
        self.assertIn("sample_recurrent_rfs_sequence", source)
        self.assertIn("sample_segmentation_widths", source)

    def test_segment_count_support_is_not_restricted(self):
        """L must range over everything the exact prior allows, not {4, 5, 6}."""
        truth = msg.supplied_truth()
        rng = np.random.default_rng(41)
        J = 48
        tables = width_sampling_tables(J, truth.delta_b, truth.min_width,
                                       truth.max_width)
        observed = {len(msg.generate_trace(int(rng.integers(1 << 30)), "train", 0,
                                           J, truth, tables).widths)
                    for _ in range(400)}
        self.assertTrue(observed - {4, 5, 6},
                        "every sampled L fell in the old {4,5,6} support")


if __name__ == "__main__":
    unittest.main()
