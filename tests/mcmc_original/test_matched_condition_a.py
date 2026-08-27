"""Condition A machinery: formal-corpus freeze, exact DP, MAP, FFBS, metrics.

Run:  PYTHONPATH=src .venv/bin/python -m unittest tests.mcmc_original.test_matched_condition_a -v
"""

from __future__ import annotations

import math
import pathlib
import unittest
from collections import Counter

import numpy as np

from hpop.mcmc_original import matched_generator_diagnostics as mgd
from hpop.mcmc_original import matched_synthetic_generator as msg
from hpop.mcmc_original.fast_segmentation_kernel import spans_of
from hpop.mcmc_original.matched_condition_a import (
    NullScorer, SemiMarkovPosterior, adjusted_rand_index, auroc,
    average_precision, boundary_f1, calibration_table,
    expected_calibration_error, normalized_mutual_information,
    segmentation_voi,
)
from hpop.mcmc_original.recurrent_rfs import recurrent_rfs_log_likelihood
from hpop.mcmc_original.recurrent_segmentation import RecurrentBlockScorer
from hpop.mcmc_original.stage6e_exact import enumerate_states, state_log_weights
from hpop.mcmc_original.targets import logsumexp

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Frozen formal-corpus identity (registered in
# results/mcmc_original/matched_synthetic_formal_corpus/ before inference).
FORMAL_SEED = 6_200_001
FORMAL_CORPUS_HASH = \
    "dd280a4a09896154e167f388edd401a9119ba398167c09404aba5f7743e58ec2"
FORMAL_TRUTH_HASH = \
    "fc41538fd44d170df8d0a6401f0c6e6b49d52418c487e22f9e4f45ee047f903e"


def _tiny_setup(seed=424242, lengths=(6, 7, 10, 13)):
    truth = msg.supplied_truth()
    corpus = msg.generate_corpus(seed, lengths, (), truth)
    scorer = RecurrentBlockScorer(
        traces=tuple(t.cpa for t in corpus.train), epsilon=truth.epsilon,
        u_by_skill=truth.u_by_skill, beta=truth.beta, omega=truth.omega,
        lambda_rep=truth.lambda_rep, lambda_back=truth.lambda_back,
        max_width=truth.max_width, min_width=truth.min_width)
    with np.errstate(divide="ignore"):
        log_pi = np.log(truth.pi)
        log_p = np.log(truth.transition)
    return truth, corpus, scorer, log_pi, log_p


class TestFormalCorpusFreeze(unittest.TestCase):
    def test_reproducibility_and_pinned_hashes(self):
        truth = msg.supplied_truth()
        corpus = msg.generate_corpus(
            FORMAL_SEED,
            tuple((24, 32, 40, 48)[i % 4] for i in range(100)),
            tuple((24, 32, 40, 48)[i % 4] for i in range(45)), truth)
        self.assertEqual(msg.corpus_hash(corpus), FORMAL_CORPUS_HASH)
        self.assertEqual(
            msg.sha256_hex(msg.canonical_json(msg.truth_to_jsonable(truth))),
            FORMAL_TRUTH_HASH)
        lengths = [t.length for t in corpus.train]
        self.assertEqual(Counter(lengths), Counter({24: 25, 32: 25, 40: 25,
                                                    48: 25}))
        self.assertEqual(Counter(t.length for t in corpus.heldout),
                         Counter({24: 12, 32: 11, 40: 11, 48: 11}))

    def test_candidate_block_q0_reset(self):
        """The scorer used for candidate blocks replays from q_0 = 0 exactly."""
        truth, corpus, scorer, _, _ = _tiny_setup(lengths=(24,))
        trace = corpus.train[0]
        for start, end, skill in ((0, 5, 0), (5, 12, 1), (12, 24, 2),
                                  (3, 9, 1), (0, 24, 0)):
            direct = recurrent_rfs_log_likelihood(
                trace.cpa[start:end], truth.u_by_skill[skill], truth.beta,
                truth.epsilon, truth.omega, truth.lambda_rep, truth.lambda_back)
            self.assertLess(abs(scorer.score(0, start, end, skill) - direct),
                            1e-10)


class TestExactDPAgainstEnumeration(unittest.TestCase):
    def setUp(self):
        (self.truth, self.corpus, self.scorer,
         self.log_pi, self.log_p) = _tiny_setup()

    def _posterior_and_enum(self, i):
        trace = self.corpus.train[i]
        post = SemiMarkovPosterior(i, trace.length, self.scorer, self.log_pi,
                                   self.log_p, self.truth.delta_b,
                                   self.truth.min_width, self.truth.max_width)
        states = enumerate_states(trace.length, self.truth.n_skills,
                                  self.truth.min_width, self.truth.max_width)
        weights = state_log_weights(states, i, trace.length, self.scorer,
                                    self.log_pi, self.log_p, self.truth.delta_b)
        return trace, post, states, np.asarray(weights)

    def test_log_z_boundary_occurrence_parity(self):
        for i in range(len(self.corpus.train)):
            trace, post, states, weights = self._posterior_and_enum(i)
            log_z = float(logsumexp(weights))
            probs = np.exp(weights - log_z)
            self.assertLess(abs(post.log_z - log_z), 1e-10)
            bm = np.zeros(trace.length - 1)
            occ = np.zeros((trace.length, self.truth.n_skills))
            for key, p in zip(states, probs):
                for end, _ in key[:-1]:
                    bm[end - 1] += p
                for a, b, k in spans_of(key):
                    occ[a:b, k] += p
            self.assertLess(float(np.abs(post.boundary_marginals() - bm).max()),
                            1e-10)
            self.assertLess(float(np.abs(post.occurrence_label_marginals()
                                         - occ).max()), 1e-10)

    def test_map_viterbi_parity(self):
        for i in range(len(self.corpus.train)):
            _, post, states, weights = self._posterior_and_enum(i)
            best = states[int(np.argmax(weights))]
            ends, labels, log_post = post.map_path()
            self.assertEqual(tuple(zip(ends, labels)), best)
            self.assertLess(abs(log_post - (float(weights.max())
                                            - float(logsumexp(weights)))), 1e-10)

    def test_true_path_and_segmentation_posterior(self):
        for i in range(len(self.corpus.train)):
            trace, post, states, weights = self._posterior_and_enum(i)
            log_z = float(logsumexp(weights))
            true_ends = list(trace.boundaries) + [trace.length]
            key = tuple(zip(true_ends, trace.labels))
            idx = states.index(key)
            self.assertLess(abs(post.true_path_log_posterior(true_ends,
                                                             trace.labels)
                                - (float(weights[idx]) - log_z)), 1e-10)
            same_seg = [j for j, s in enumerate(states)
                        if tuple(e for e, _ in s) == tuple(true_ends)]
            expected = float(logsumexp(weights[same_seg])) - log_z
            self.assertLess(abs(post.segmentation_log_posterior(true_ends)
                                - expected), 1e-10)

    def test_ffbs_deterministic_child_seeds_and_legality(self):
        _, post, states, weights = self._posterior_and_enum(2)
        seed = np.random.SeedSequence(entropy=123, spawn_key=(2,))
        draws_a = [post.ffbs_draw(np.random.default_rng(seed))
                   for seed in [np.random.SeedSequence(entropy=123,
                                                       spawn_key=(2, d))
                                for d in range(50)]]
        draws_b = [post.ffbs_draw(np.random.default_rng(seed))
                   for seed in [np.random.SeedSequence(entropy=123,
                                                       spawn_key=(2, d))
                                for d in range(50)]]
        self.assertEqual(draws_a, draws_b)         # same child stream, same draw
        other = [post.ffbs_draw(np.random.default_rng(
            np.random.SeedSequence(entropy=123, spawn_key=(3, d))))
            for d in range(50)]
        self.assertNotEqual(draws_a, other)        # different trace stream
        legal = set(states)
        for ends, labels in draws_a:
            self.assertIn(tuple(zip(ends, labels)), legal)

    def test_ffbs_matches_exact_posterior(self):
        _, post, states, weights = self._posterior_and_enum(2)
        probs = np.exp(weights - logsumexp(weights))
        rng = np.random.default_rng(7)
        n = 40_000
        counts = Counter(tuple(zip(*post.ffbs_draw(rng))) for _ in range(n))
        tv = 0.5 * sum(abs(counts.get(s, 0) / n - p)
                       for s, p in zip(states, probs))
        self.assertLess(tv, 0.02)

    def test_prior_mode_matches_independent_references(self):
        for J in (10, 24):
            prior = SemiMarkovPosterior(0, J, NullScorer(), self.log_pi,
                                        self.log_p, self.truth.delta_b,
                                        self.truth.min_width,
                                        self.truth.max_width)
            self.assertLess(abs(prior.log_z - mgd.exact_normalizer(
                J, self.truth.delta_b, 3, 12)), 1e-10)
            self.assertLess(float(np.abs(
                prior.boundary_marginals()
                - mgd.exact_boundary_marginals(J, self.truth.delta_b, 3,
                                               12)).max()), 1e-10)
            counts = prior.segment_count_posterior()
            reference = mgd.exact_segment_count_distribution(
                J, self.truth.delta_b, 3, 12)
            n = max(len(counts), len(reference))
            self.assertLess(float(np.abs(
                np.pad(counts, (0, n - len(counts)))
                - np.pad(reference, (0, n - len(reference)))).max()), 1e-10)

    def test_entropy_and_consistency(self):
        for i in range(len(self.corpus.train)):
            trace, post, states, weights = self._posterior_and_enum(i)
            probs = np.exp(weights - logsumexp(weights))
            enum_entropy = -float(np.sum(
                probs * (weights - logsumexp(weights))))
            self.assertLess(abs(post.path_entropy() - enum_entropy), 1e-10)
            occ = post.occurrence_label_marginals()
            self.assertLess(float(np.abs(occ.sum(axis=1) - 1.0).max()), 1e-10)
            counts = post.segment_count_posterior()
            self.assertLess(abs(float(counts.sum()) - 1.0), 1e-10)


class TestMetricFunctions(unittest.TestCase):
    def test_auroc_and_auprc_hand_cases(self):
        self.assertAlmostEqual(auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)
        self.assertAlmostEqual(auroc([0.5, 0.5], [0, 1]), 0.5)   # ties
        self.assertAlmostEqual(
            average_precision([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]),
            (1.0 + 2.0 / 3.0) / 2.0)
        self.assertTrue(math.isnan(auroc([0.2, 0.3], [1, 1])))

    def test_calibration(self):
        p = [0.05] * 100 + [0.95] * 100
        y = [0] * 95 + [1] * 5 + [1] * 95 + [0] * 5
        self.assertLess(expected_calibration_error(p, y), 1e-12)
        table = calibration_table(p, y)
        self.assertEqual(sum(row["count"] for row in table), 200)

    def test_ari_nmi_no_alignment(self):
        """A label permutation must NOT score as perfect accuracy-like metrics —
        ARI/NMI are permutation-invariant by definition (reported as such), but
        the confusion matrix and accuracy use raw identities."""
        truth = [0, 0, 1, 1, 2, 2]
        permuted = [1, 1, 2, 2, 0, 0]
        self.assertAlmostEqual(adjusted_rand_index(truth, truth), 1.0)
        self.assertAlmostEqual(normalized_mutual_information(truth, truth), 1.0)
        self.assertAlmostEqual(adjusted_rand_index(truth, permuted), 1.0)
        accuracy = float(np.mean(np.asarray(truth) == np.asarray(permuted)))
        self.assertEqual(accuracy, 0.0)            # identity-based, not aligned

    def test_no_alignment_in_source(self):
        for name in ("src/hpop/mcmc_original/matched_condition_a.py",
                     "scripts/run_matched_condition_a.py"):
            source = (ROOT / name).read_text().lower()
            # actual alignment machinery, not documentation mentioning its absence
            for token in ("linear_sum_assignment", "munkres",
                          "scipy.optimize", "permutations(range"):
                self.assertNotIn(token, source, f"{name} contains '{token}'")

    def test_boundary_f1_and_voi(self):
        self.assertEqual(boundary_f1(set(), set()), 1.0)
        self.assertEqual(boundary_f1({3, 7}, {3, 7}), 1.0)
        self.assertAlmostEqual(boundary_f1({3}, {3, 7}), 2 / 3)
        self.assertAlmostEqual(segmentation_voi((5, 10), (5, 10), 10), 0.0)
        self.assertGreater(segmentation_voi((5, 10), (3, 10), 10), 0.0)


if __name__ == "__main__":
    unittest.main()
