"""Tests for the merge-only semi-Markov lattice: exact inference against brute-force enumeration.

Run:  PYTHONPATH=src .venv/bin/python -m unittest tests.test_semi_markov -v
"""
import unittest

import numpy as np

from hpop.inference.semi_markov import (SemiMarkovLattice, brute_force_log_partition,
                                        brute_force_marginals, brute_force_viterbi,
                                        enumerate_segmentations)

SHAPES = [(5, 3, 3), (6, 2, 2), (4, 4, 4), (7, 3, 2), (8, 4, 3), (3, 3, 1)]


def random_scores(shape, seed=0):
    return np.random.default_rng(seed).normal(size=shape)


class TestExactInference(unittest.TestCase):
    def test_log_partition_matches_enumeration(self):
        for i, shape in enumerate(SHAPES):
            s = random_scores(shape, i)
            with self.subTest(shape=shape):
                self.assertAlmostEqual(SemiMarkovLattice(s.copy()).log_partition(),
                                       brute_force_log_partition(s.copy()), places=9)

    def test_segment_marginals_match_enumeration(self):
        for i, shape in enumerate(SHAPES):
            s = random_scores(shape, 100 + i)
            with self.subTest(shape=shape):
                got = SemiMarkovLattice(s.copy()).segment_marginals()
                want = brute_force_marginals(s.copy())
                self.assertLess(float(np.max(np.abs(got - want))), 1e-9)

    def test_viterbi_matches_enumeration(self):
        for i, shape in enumerate(SHAPES):
            s = random_scores(shape, 200 + i)
            with self.subTest(shape=shape):
                segs, val = SemiMarkovLattice(s.copy()).viterbi()
                bsegs, bval = brute_force_viterbi(s.copy())
                self.assertAlmostEqual(val, bval, places=9)
                self.assertEqual(segs, bsegs)

    def test_entropy_matches_enumeration(self):
        s = random_scores((6, 3, 3), 7)
        logZ = brute_force_log_partition(s.copy())
        ent = 0.0
        for segs in enumerate_segmentations(6, 3, 3):
            t = sum(s[i, (b - i) - 1, k] for i, b, k in segs)
            p = np.exp(t - logZ)
            if p > 0:
                ent -= p * (t - logZ)
        self.assertAlmostEqual(SemiMarkovLattice(s.copy()).entropy(), ent, places=9)


class TestPosteriorProperties(unittest.TestCase):
    def test_each_seed_position_is_covered_exactly_once(self):
        for i, shape in enumerate(SHAPES):
            J, D, K = shape
            s = random_scores(shape, 300 + i)
            mu = SemiMarkovLattice(s.copy()).segment_marginals()
            with self.subTest(shape=shape):
                for j in range(J):
                    total = sum(mu[a, w - 1, k]
                                for a in range(J) for w in range(1, D + 1) for k in range(K)
                                if a <= j < a + w)
                    self.assertAlmostEqual(total, 1.0, places=9)

    def test_expected_segment_count_is_between_one_and_J(self):
        J, D, K = 7, 3, 3
        lat = SemiMarkovLattice(random_scores((J, D, K), 11))
        n = lat.expected_num_segments()
        self.assertGreaterEqual(n, J / D - 1e-9)
        self.assertLessEqual(n, J + 1e-9)

    def test_entropy_is_non_negative(self):
        for i, shape in enumerate(SHAPES):
            lat = SemiMarkovLattice(random_scores(shape, 400 + i))
            self.assertGreaterEqual(lat.entropy(), -1e-9)


class TestTransitionLattice(unittest.TestCase):
    """Run 2 correction E.1: a real semi-Markov prior (duration + transition + start) instead of
    the unnormalized -lambda_seg factor. Exactness must survive the extra terms."""

    def _rand(self, shape, seed):
        rng = np.random.default_rng(seed)
        J, D, K = shape
        return (rng.normal(size=shape), rng.normal(size=(K, K)), rng.normal(size=K))

    def test_log_partition_matches_enumeration_with_transitions(self):
        for i, shape in enumerate(SHAPES[:4]):
            s, T, st = self._rand(shape, 500 + i)
            with self.subTest(shape=shape):
                got = SemiMarkovLattice(s.copy(), log_trans=T, log_start=st).log_partition()
                self.assertAlmostEqual(got, brute_force_log_partition(s.copy(), T, st), places=9)

    def test_marginals_and_viterbi_match_enumeration_with_transitions(self):
        for i, shape in enumerate(SHAPES[:4]):
            s, T, st = self._rand(shape, 600 + i)
            lat = SemiMarkovLattice(s.copy(), log_trans=T, log_start=st)
            with self.subTest(shape=shape):
                self.assertLess(float(np.max(np.abs(lat.segment_marginals()
                                                    - brute_force_marginals(s.copy(), T, st)))), 1e-9)
                segs, val = lat.viterbi()
                bsegs, bval = brute_force_viterbi(s.copy(), T, st)
                self.assertAlmostEqual(val, bval, places=9)
                self.assertEqual(segs, bsegs)

    def test_transition_marginals_match_enumeration(self):
        from hpop.inference.semi_markov import _path_score
        for i, shape in enumerate(SHAPES[:4]):
            J, D, K = shape
            s, T, st = self._rand(shape, 700 + i)
            xi, start = SemiMarkovLattice(s.copy(), log_trans=T, log_start=st).transition_marginals()
            logZ = brute_force_log_partition(s.copy(), T, st)
            bxi, bst = np.zeros((K, K)), np.zeros(K)
            for segs in enumerate_segmentations(J, D, K):
                t = _path_score(segs, s, T, st)
                if not np.isfinite(t):
                    continue
                p = np.exp(t - logZ)
                bst[segs[0][2]] += p
                for a, b in zip(segs, segs[1:]):
                    bxi[a[2], b[2]] += p
            with self.subTest(shape=shape):
                self.assertLess(float(np.max(np.abs(xi - bxi))), 1e-9)
                self.assertLess(float(np.max(np.abs(start - bst))), 1e-9)

    def test_zero_transitions_reduce_to_the_plain_lattice(self):
        s = random_scores((6, 3, 3), 42)
        plain = SemiMarkovLattice(s.copy()).log_partition()
        zeroed = SemiMarkovLattice(s.copy(), log_trans=np.zeros((3, 3)),
                                   log_start=np.zeros(3)).log_partition()
        self.assertAlmostEqual(plain, zeroed, places=12)


class TestConstraints(unittest.TestCase):
    def test_no_adjacent_segments_share_a_label(self):
        """Assumption (ass:no-self-transition): consecutive instances get different skills."""
        segs, _ = SemiMarkovLattice(random_scores((8, 3, 3), 5)).viterbi()
        labels = [k for _, _, k in segs]
        self.assertTrue(all(labels[i] != labels[i + 1] for i in range(len(labels) - 1)))

    def test_segments_tile_the_seed_sequence(self):
        J = 8
        segs, _ = SemiMarkovLattice(random_scores((J, 3, 3), 6)).viterbi()
        self.assertEqual(segs[0][0], 0)
        self.assertEqual(segs[-1][1], J)
        for (_, b, _), (a2, _, _) in zip(segs, segs[1:]):
            self.assertEqual(b, a2)

    def test_merge_width_cap_is_respected(self):
        D = 3
        segs, _ = SemiMarkovLattice(random_scores((9, D, 3), 8)).viterbi()
        self.assertTrue(all(b - a <= D for a, b, _ in segs))

    def test_single_skill_cannot_tile_more_than_D_max_seeds(self):
        """With K = 1 the no-self-transition rule makes any multi-segment tiling illegal."""
        lat = SemiMarkovLattice(np.zeros((5, 2, 1)))
        self.assertEqual(lat.log_partition(), -np.inf)

    def test_single_skill_covering_the_whole_trace_is_legal(self):
        lat = SemiMarkovLattice(np.zeros((3, 3, 1)))
        self.assertAlmostEqual(lat.log_partition(), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
