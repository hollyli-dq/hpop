"""Exact segmentation prior: DP vs independent references vs enumeration.

Run:  PYTHONPATH=src .venv/bin/python -m unittest tests.mcmc_original.test_matched_segmentation_prior -v
"""

from __future__ import annotations

import math
import unittest
from collections import Counter

import numpy as np

from hpop.mcmc_original import matched_generator_diagnostics as mgd
from hpop.mcmc_original import matched_segmentation_prior as msp
from hpop.mcmc_original.stage6e_frozen import (
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH,
)

TINY_J = (6, 7, 9, 10, 12)
DESIGN_J = (24, 32, 40, 48)


class TestExactNormalizer(unittest.TestCase):
    def test_dp_matches_enumeration_on_tiny_j(self):
        for J in TINY_J:
            dp = msp.log_normalizer(J, DELTA_B, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
            enum = mgd.log_normalizer_from_enumeration(
                J, DELTA_B, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
            self.assertLess(abs(dp - enum), 1e-12, f"J={J}")

    def test_dp_matches_combinatorial_reference_on_design_j(self):
        for J in TINY_J + DESIGN_J:
            dp = msp.log_normalizer(J, DELTA_B, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
            ref = mgd.exact_normalizer(J, DELTA_B, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
            self.assertLess(abs(dp - ref), 1e-12, f"J={J}")

    def test_impossible_lengths_have_zero_mass(self):
        log_g = msp.log_suffix_normalizers(10, DELTA_B, MIN_BLOCK_WIDTH,
                                           MAX_BLOCK_WIDTH)
        self.assertEqual(log_g[0], 0.0)
        for r in (1, 2):                      # below MIN_BLOCK_WIDTH: no completion
            self.assertEqual(log_g[r], -math.inf)

    def test_prior_sums_to_one_over_enumeration(self):
        for J in TINY_J:
            log_c = msp.log_normalizer(J, DELTA_B, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
            total = sum(
                math.exp(msp.log_segmentation_prior(
                    widths, J, DELTA_B, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH,
                    log_c=log_c))
                for widths in mgd.enumerate_legal_segmentations(
                    J, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH))
            self.assertLess(abs(total - 1.0), 1e-12, f"J={J}")


class TestExactMarginals(unittest.TestCase):
    def test_segment_count_distribution_two_routes(self):
        for J in TINY_J + DESIGN_J:
            a = msp.segment_count_distribution_dp(J, DELTA_B, MIN_BLOCK_WIDTH,
                                                  MAX_BLOCK_WIDTH)
            b = mgd.exact_segment_count_distribution(J, DELTA_B, MIN_BLOCK_WIDTH,
                                                     MAX_BLOCK_WIDTH)
            n = max(len(a), len(b))
            a = np.pad(a, (0, n - len(a)))
            b = np.pad(b, (0, n - len(b)))
            self.assertLess(float(np.abs(a - b).max()), 1e-12, f"J={J}")
            self.assertLess(abs(float(a.sum()) - 1.0), 1e-12, f"J={J}")

    def test_segment_count_distribution_vs_enumeration(self):
        for J in TINY_J:
            exact = mgd.exact_segment_count_distribution(J, DELTA_B,
                                                         MIN_BLOCK_WIDTH,
                                                         MAX_BLOCK_WIDTH)
            log_c = mgd.exact_normalizer(J, DELTA_B, MIN_BLOCK_WIDTH,
                                         MAX_BLOCK_WIDTH)
            brute = np.zeros_like(exact)
            for widths in mgd.enumerate_legal_segmentations(
                    J, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH):
                L = len(widths)
                brute[L] += math.exp((L - 1) * math.log(DELTA_B)
                                     + (J - L) * math.log1p(-DELTA_B) - log_c)
            self.assertLess(float(np.abs(exact - brute).max()), 1e-12, f"J={J}")

    def test_boundary_marginals_two_routes(self):
        for J in TINY_J + DESIGN_J:
            a = msp.boundary_marginals_dp(J, DELTA_B, MIN_BLOCK_WIDTH,
                                          MAX_BLOCK_WIDTH)
            b = mgd.exact_boundary_marginals(J, DELTA_B, MIN_BLOCK_WIDTH,
                                             MAX_BLOCK_WIDTH)
            self.assertLess(float(np.abs(a - b).max()), 1e-12, f"J={J}")

    def test_boundary_marginals_vs_enumeration(self):
        for J in TINY_J:
            exact = mgd.exact_boundary_marginals(J, DELTA_B, MIN_BLOCK_WIDTH,
                                                 MAX_BLOCK_WIDTH)
            log_c = mgd.exact_normalizer(J, DELTA_B, MIN_BLOCK_WIDTH,
                                         MAX_BLOCK_WIDTH)
            brute = np.zeros(J - 1)
            for widths in mgd.enumerate_legal_segmentations(
                    J, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH):
                L = len(widths)
                p = math.exp((L - 1) * math.log(DELTA_B)
                             + (J - L) * math.log1p(-DELTA_B) - log_c)
                running = 0
                for w in widths[:-1]:
                    running += w
                    brute[running - 1] += p
            self.assertLess(float(np.abs(exact - brute).max()), 1e-12, f"J={J}")

    def test_boundary_sum_equals_expected_cuts(self):
        for J in TINY_J + DESIGN_J:
            boundaries = mgd.exact_boundary_marginals(J, DELTA_B, MIN_BLOCK_WIDTH,
                                                      MAX_BLOCK_WIDTH)
            counts = mgd.exact_segment_count_distribution(J, DELTA_B,
                                                          MIN_BLOCK_WIDTH,
                                                          MAX_BLOCK_WIDTH)
            expected_cuts = float(np.dot(np.arange(len(counts)), counts)) - 1.0
            self.assertLess(abs(float(boundaries.sum()) - expected_cuts), 1e-10,
                            f"J={J}")

    def test_expected_width_counts_vs_enumeration(self):
        for J in TINY_J:
            exact = mgd.exact_expected_width_counts(J, DELTA_B, MIN_BLOCK_WIDTH,
                                                    MAX_BLOCK_WIDTH)
            log_c = mgd.exact_normalizer(J, DELTA_B, MIN_BLOCK_WIDTH,
                                         MAX_BLOCK_WIDTH)
            brute = np.zeros_like(exact)
            for widths in mgd.enumerate_legal_segmentations(
                    J, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH):
                L = len(widths)
                p = math.exp((L - 1) * math.log(DELTA_B)
                             + (J - L) * math.log1p(-DELTA_B) - log_c)
                for w in widths:
                    brute[w - MIN_BLOCK_WIDTH] += p
            self.assertLess(float(np.abs(exact - brute).max()), 1e-12, f"J={J}")


class TestExactSampler(unittest.TestCase):
    def test_samples_are_always_legal(self):
        rng = np.random.default_rng(11)
        for J in (24, 32):
            tables = msp.width_sampling_tables(J, DELTA_B, MIN_BLOCK_WIDTH,
                                               MAX_BLOCK_WIDTH)
            for _ in range(500):
                widths = msp.sample_segmentation_widths(
                    rng, J, DELTA_B, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH,
                    tables=tables)
                self.assertEqual(sum(widths), J)
                self.assertTrue(all(MIN_BLOCK_WIDTH <= w <= MAX_BLOCK_WIDTH
                                    for w in widths))
                ends = msp.ends_of_widths(widths)
                self.assertEqual(ends[-1], J)
                self.assertEqual(list(ends), sorted(set(ends)))

    def test_tiny_full_state_distribution(self):
        """Complete-state parity on a tiny J — more than the L and B_t marginals."""
        rng = np.random.default_rng(12)
        J, n = 10, 100_000
        tables = msp.width_sampling_tables(J, DELTA_B, MIN_BLOCK_WIDTH,
                                           MAX_BLOCK_WIDTH)
        counts = Counter(
            msp.sample_segmentation_widths(rng, J, DELTA_B, MIN_BLOCK_WIDTH,
                                           MAX_BLOCK_WIDTH, tables=tables)
            for _ in range(n))
        log_c = mgd.exact_normalizer(J, DELTA_B, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
        states = mgd.enumerate_legal_segmentations(J, MIN_BLOCK_WIDTH,
                                                   MAX_BLOCK_WIDTH)
        self.assertEqual(set(counts), set(states) & set(counts))  # support subset
        tv, max_err = 0.0, 0.0
        for state in states:
            L = len(state)
            exact = math.exp((L - 1) * math.log(DELTA_B)
                             + (J - L) * math.log1p(-DELTA_B) - log_c)
            err = abs(counts.get(state, 0) / n - exact)
            tv += 0.5 * err
            max_err = max(max_err, err)
            if exact * n > 50:
                self.assertGreater(counts.get(state, 0), 0,
                                   f"state {state} never visited")
        self.assertLess(tv, 0.01)
        self.assertLess(max_err, 0.01)

    def test_moderate_j_marginal_parity(self):
        rng = np.random.default_rng(13)
        J, n = 32, 20_000
        tables = msp.width_sampling_tables(J, DELTA_B, MIN_BLOCK_WIDTH,
                                           MAX_BLOCK_WIDTH)
        samples = [msp.sample_segmentation_widths(rng, J, DELTA_B,
                                                  MIN_BLOCK_WIDTH,
                                                  MAX_BLOCK_WIDTH, tables=tables)
                   for _ in range(n)]
        exact_l = mgd.exact_segment_count_distribution(J, DELTA_B,
                                                       MIN_BLOCK_WIDTH,
                                                       MAX_BLOCK_WIDTH)
        emp_l = mgd.empirical_segment_count_distribution(samples,
                                                         len(exact_l) - 1)
        self.assertLess(mgd.total_variation(emp_l, exact_l), 0.02)
        exact_b = mgd.exact_boundary_marginals(J, DELTA_B, MIN_BLOCK_WIDTH,
                                               MAX_BLOCK_WIDTH)
        emp_b = mgd.empirical_boundary_marginals(samples, J)
        self.assertLess(float(np.abs(emp_b - exact_b).max()), 0.02)
        violations = mgd.segmentation_support_violations(samples, J,
                                                         MIN_BLOCK_WIDTH,
                                                         MAX_BLOCK_WIDTH)
        self.assertEqual(violations["illegal_width"], 0)
        self.assertEqual(violations["incomplete_cover"], 0)
        self.assertEqual(violations["overlap_or_gap"], 0)

    def test_rejects_mismatched_tables(self):
        tables = msp.width_sampling_tables(24, DELTA_B, MIN_BLOCK_WIDTH,
                                           MAX_BLOCK_WIDTH)
        with self.assertRaises(ValueError):
            msp.sample_segmentation_widths(np.random.default_rng(0), 32, DELTA_B,
                                           MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH,
                                           tables=tables)

    def test_rejects_bad_configuration(self):
        with self.assertRaises(ValueError):
            msp.log_normalizer(24, 0.0, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
        with self.assertRaises(ValueError):
            msp.log_normalizer(24, DELTA_B, 5, 4)
        with self.assertRaises(ValueError):
            msp.width_sampling_tables(2, DELTA_B, MIN_BLOCK_WIDTH,
                                      MAX_BLOCK_WIDTH)   # no legal segmentation


if __name__ == "__main__":
    unittest.main()
