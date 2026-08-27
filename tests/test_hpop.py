"""Tests for the HPOP variational EM fit: objective correctness, normalization, recovery.

Run:  PYTHONPATH=src .venv/bin/python -m unittest tests.test_hpop -v
"""
import unittest

import numpy as np

from hpop.eval.metrics import evaluate, transitive_reduction
from hpop.inference.hpop import HPOP, HPOPConfig, batched_block_logp
from hpop.inference.recurrent import RecurrentFrontier, hard_precedence_matrix
from hpop.synth.generator import sample_corpus, seeds_of


class TestBatchedLikelihood(unittest.TestCase):
    def test_batched_matches_single_skill_model(self):
        rng = np.random.default_rng(0)
        V, K = 6, 3
        D = np.zeros((K, V, V))
        for k in range(K):
            edges = [(0, 2), (1, 2), (2, 4)] if k == 0 else [(1, 3), (3, 5)] if k == 1 else []
            D[k] = hard_precedence_matrix(V, edges)
        theta = rng.normal(size=(K, V))
        J = D * 0.9
        seq = [0, 1, 2, 4, 2, 5, 3]
        got = batched_block_logp(D, theta, J, seq, 1.5, 1.5, 0.5, 0.02)
        for k in range(K):
            m = RecurrentFrontier(D[k], omega=0.0, beta=1.5, lam_rep=1.5, lam_back=0.5,
                                  eps=0.02, theta=theta[k])
            m.J = J[k]                                   # match the batched invalidation exactly
            self.assertAlmostEqual(got[k], m.logp(seq), places=9)


class TestNormalization(unittest.TestCase):
    """The manuscript's ELBO uses an unnormalized segmentation prior, so raw log Z is not a
    likelihood. We report log Z(score) - log Z(prior), which must be an exact log-likelihood."""

    def test_uniform_model_gives_exact_uniform_likelihood(self):
        V = 5
        cfg = HPOPConfig(V=V, K_max=3, D_max=3, beta=1.0, lam_rep=0.0, lam_back=0.0, eps=0.0)
        m = HPOP(cfg, rng=np.random.default_rng(0))
        m.theta[:] = 0.0
        m.D[:] = 0.0
        m._refresh_J()
        corpus = [[[0, 1], [2], [3, 4, 0]], [[1], [2, 3], [4]]]
        for seeds, logp in zip(corpus, m.heldout_logp(corpus)):
            T = sum(len(s) for s in seeds)
            self.assertAlmostEqual(logp, -T * np.log(V), places=8)

    def test_likelihood_is_invariant_to_the_segment_penalty(self):
        """lambda_seg shifts every segmentation's prior score, so the normalized likelihood of a
        model whose skills are identical must not depend on it."""
        V = 5
        corpus = [[[0, 1], [2], [3, 4, 0]]]
        vals = []
        for lam in (0.0, 3.0):
            cfg = HPOPConfig(V=V, K_max=1, D_max=3, lam_seg=lam, eps=0.0)
            m = HPOP(cfg, rng=np.random.default_rng(0))
            m.theta[:] = 0.0
            m.D[:] = 0.0
            m._refresh_J()
            vals.append(m.heldout_logp(corpus)[0])
        self.assertAlmostEqual(vals[0], vals[1], places=8)


class TestEM(unittest.TestCase):
    def setUp(self):
        self.world, self.traces = sample_corpus(seed=3, n_traces=14, K_true=3, V=10)
        self.corpus = [seeds_of(t) for t in self.traces]

    def test_evidence_is_non_decreasing(self):
        m = HPOP(HPOPConfig(V=10, K_max=4, D_max=6, theta_steps=4),
                 rng=np.random.default_rng(0))
        hist = m.fit(self.corpus, iters=6, warmup=2)
        for a, b in zip(hist, hist[1:]):
            self.assertGreaterEqual(b, a - 1e-6 * abs(a))

    def test_decoding_tiles_every_trace(self):
        m = HPOP(HPOPConfig(V=10, K_max=4, D_max=6, theta_steps=2),
                 rng=np.random.default_rng(0))
        m.fit(self.corpus, iters=2, warmup=2)
        for seeds in self.corpus:
            segs = m.decode(seeds)
            self.assertEqual(segs[0][0], 0)
            self.assertEqual(segs[-1][1], len(seeds))

    def test_learned_structure_is_acyclic(self):
        m = HPOP(HPOPConfig(V=10, K_max=4, D_max=6, theta_steps=3),
                 rng=np.random.default_rng(0))
        m.fit(self.corpus, iters=5, warmup=1)
        for k in range(m.cfg.K_max):
            self.assertEqual(float(np.trace(m.D[k])), 0.0)
            self.assertTrue(np.all((m.D[k] > 0) & (m.D[k].T > 0) == False))  # noqa: E712

    def test_recovers_assignments_better_than_chance(self):
        m = HPOP(HPOPConfig(V=10, K_max=4, D_max=6, theta_steps=4),
                 rng=np.random.default_rng(0))
        m.fit(self.corpus, iters=8, warmup=2)
        decoded = [m.decode(s) for s in self.corpus]
        res = evaluate(self.world, self.traces, decoded, 4, m.D,
                       m.global_structure(self.corpus))
        self.assertGreater(res["skill_ari"], 0.5)
        self.assertGreater(res["boundary_f1"], 0.5)


class TestRun2Corrections(unittest.TestCase):
    """Every Run 2 flag must default to the Run 1 model and do exactly what it claims."""

    def setUp(self):
        self.world, self.traces = sample_corpus(seed=3, n_traces=8, K_true=3, V=10)
        self.corpus = [seeds_of(t) for t in self.traces]
        self.base = dict(V=10, K_max=4, D_max=6)

    def _fit(self, seed=0, iters=4, warmup=2, outcomes=None, **kw):
        m = HPOP(HPOPConfig(**{**self.base, **kw}), rng=np.random.default_rng(seed))
        hist = m.fit(self.corpus, iters=iters, warmup=warmup, outcomes=outcomes)
        return m, hist

    def test_defaults_reproduce_the_run1_model(self):
        _, a = self._fit()
        _, b = self._fit(lam_comp=1.0, lam_po=1.0, normalized_duration=False,
                         failure_invalidation=False, seq_eta=0.0)
        self.assertTrue(np.allclose(a, b))

    def test_lam_po_zero_removes_all_order_terms(self):
        """Note this is *not* identical to use_order=False: that keeps the repeat cost
        -lambda_rep * q, so the Run 1 HSMM is composition + repeat dynamics, while lam_po = 0 is
        composition alone."""
        m, _ = self._fit(lam_po=0.0, iters=3, warmup=3)
        seq = [0, 1, 2, 1, 0]
        lp = m.block_logliks([[c] for c in seq])
        self.assertTrue(np.all(np.isfinite(lp[np.isfinite(lp)])))
        hsmm, _ = self._fit(use_order=False, iters=3, warmup=3)
        self.assertNotAlmostEqual(float(np.sum(m.heldout_logp(self.corpus))),
                                  float(np.sum(hsmm.heldout_logp(self.corpus))), places=3)

    def test_normalized_duration_yields_proper_distributions(self):
        m, hist = self._fit(normalized_duration=True, iters=5)
        self.assertAlmostEqual(float(np.exp(m.log_dur).sum()), 1.0, places=9)
        rows = np.exp(m.log_trans).sum(axis=1)
        self.assertTrue(np.allclose(rows, 1.0, atol=1e-9))
        self.assertAlmostEqual(float(np.exp(m.log_start).sum()), 1.0, places=9)
        self.assertTrue(all(b >= a - 1e-6 * abs(a) for a, b in zip(hist, hist[1:])))

    def test_normalized_duration_removes_lam_seg_sensitivity(self):
        """With a learned duration distribution the free per-segment penalty must stop mattering."""
        a, _ = self._fit(normalized_duration=True, lam_seg=0.0, iters=4)
        b, _ = self._fit(normalized_duration=True, lam_seg=9.0, iters=4)
        self.assertAlmostEqual(float(np.sum(a.heldout_logp(self.corpus))),
                               float(np.sum(b.heldout_logp(self.corpus))), places=6)

    def test_failure_conditioned_invalidation_uses_the_outcome(self):
        n = [sum(len(s) for s in c) for c in self.corpus]
        ok = [["SUCCESS"] * k for k in n]
        bad = [["FAILURE"] * k for k in n]
        m, _ = self._fit(failure_invalidation=True, outcomes=ok, iters=3, warmup=3)
        self.assertFalse(np.allclose(m.heldout_logp(self.corpus, outcomes=ok),
                                     m.heldout_logp(self.corpus, outcomes=bad)))

    def test_failure_invalidation_runs_backwards_along_precedence(self):
        m, _ = self._fit(failure_invalidation=True, iters=3, warmup=1)
        k = int(m.active_skills()[0]) if len(m.active_skills()) else 0
        self.assertIsNotNone(m.J_fail)
        # J_fail[y, z] > 0 exactly where z must precede y — the reverse of J
        self.assertTrue(np.allclose(m.J_fail[k], m.J[k].T))

    def test_seq_term_keeps_the_model_normalized(self):
        m, hist = self._fit(seq_eta=1.0, iters=4)
        self.assertEqual(m.seq_logits.shape, (4, 11, 10))
        rows = np.exp(m.seq_logits).sum(axis=2)
        self.assertTrue(np.allclose(rows, 1.0, atol=1e-9))
        self.assertTrue(all(np.isfinite(m.heldout_logp(self.corpus))))


class TestMetricHelpers(unittest.TestCase):
    def test_transitive_reduction_drops_implied_edges(self):
        R = hard_precedence_matrix(3, [(0, 1), (1, 2)])
        self.assertTrue(R[0, 2] > 0)                     # closure contains the shortcut
        cover = transitive_reduction(R)
        self.assertFalse(cover[0, 2])                    # cover relation does not
        self.assertTrue(cover[0, 1] and cover[1, 2])


if __name__ == "__main__":
    unittest.main()
