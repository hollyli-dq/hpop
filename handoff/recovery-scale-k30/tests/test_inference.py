"""Sanity checks for the BPOP frontier-softmax likelihood, posets, and the CRP new-skill penalty.

Run:  PYTHONPATH=src .venv/bin/python -m unittest tests.test_inference -v
"""
import math
import unittest

from hpop.inference.poset import Poset
from hpop.inference.likelihood import successor_utility, frontier_softmax_logp, flat_bpop_logp
from hpop.inference.library import crp_logprob, crp_predictive, pitman_yor_predictive, new_skill_logpenalty


class TestPoset(unittest.TestCase):
    def setUp(self):
        # a->c, b->c, c->d ; a || b (incomparable)
        self.h = Poset(["a", "b", "c", "d"], [("a", "c"), ("b", "c"), ("c", "d")])

    def test_relations(self):
        self.assertTrue(self.h.precedes("a", "c"))
        self.assertTrue(self.h.precedes("a", "d"))      # transitive
        self.assertFalse(self.h.precedes("a", "b"))
        self.assertEqual(set(self.h.successors("a")), {"c", "d"})
        self.assertEqual(self.h.n_successors("a"), 2)
        self.assertEqual(self.h.n_successors("d"), 0)
        self.assertEqual(self.h.incomparable_pairs(), [("a", "b")])

    def test_frontier(self):
        self.assertEqual(set(self.h.frontier([])), {"a", "b"})
        self.assertEqual(set(self.h.frontier(["a"])), {"b"})
        self.assertEqual(set(self.h.frontier(["a", "b"])), {"c"})

    def test_linear_extensions(self):
        # only valid orders: (a,b,c,d) and (b,a,c,d)  -> 2
        self.assertEqual(self.h.num_linear_extensions(), 2)
        self.assertTrue(self.h.is_linear_extension(["a", "b", "c", "d"]))
        self.assertTrue(self.h.is_linear_extension(["b", "a", "c", "d"]))
        self.assertFalse(self.h.is_linear_extension(["c", "a", "b", "d"]))

    def test_latent_U_dominance(self):
        # U_a dominates U_c in both dims -> a precedes c; a,b non-dominating -> incomparable
        U = {"a": (2.0, 2.0), "b": (2.1, 1.9), "c": (1.0, 1.0), "d": (0.0, 0.0)}
        g = Poset.from_latent_U(U)
        self.assertTrue(g.precedes("a", "c") and g.precedes("a", "d"))
        self.assertFalse(g.precedes("a", "b") or g.precedes("b", "a"))  # incomparable


class TestLikelihood(unittest.TestCase):
    def setUp(self):
        self.h = Poset(["a", "b", "c", "d"], [("a", "c"), ("b", "c"), ("c", "d")])

    def test_valid_beats_violation(self):
        valid = frontier_softmax_logp(self.h, ["a", "b", "c", "d"], beta=2.0, eps=0.05)
        violation = frontier_softmax_logp(self.h, ["c", "a", "b", "d"], beta=2.0, eps=0.05)
        self.assertGreater(valid, violation)              # respecting the order is more likely
        self.assertLess(valid, 0.0)                       # log-prob

    def test_noise_floor_on_violation(self):
        # the violating step (c first) contributes only the eps/|R| noise term
        eps, n = 0.1, 4
        lp = frontier_softmax_logp(self.h, ["c", "a", "b", "d"], beta=2.0, eps=eps)
        self.assertGreater(lp, -50.0)                     # finite, not -inf (robustness)

    def test_successor_utility(self):
        Q = successor_utility(self.h)
        self.assertAlmostEqual(Q["a"], math.log(3))       # 2 successors -> log(1+2)
        self.assertAlmostEqual(Q["d"], math.log(1))

    def test_flat_bpop_sums(self):
        seqs = [["a", "b", "c", "d"], ["b", "a", "c", "d"]]
        total = flat_bpop_logp(self.h, seqs, beta=1.0, eps=0.05)
        parts = sum(frontier_softmax_logp(self.h, s, beta=1.0, eps=0.05) for s in seqs)
        self.assertAlmostEqual(total, parts)


class TestLibraryPrior(unittest.TestCase):
    def test_crp_predictive_normalizes(self):
        ex, pnew = crp_predictive([5, 3, 1], alpha=1.0)
        self.assertAlmostEqual(sum(ex) + pnew, 1.0)
        self.assertAlmostEqual(pnew, 1.0 / 10.0)          # alpha/(N+alpha)=1/10

    def test_new_skill_is_penalized(self):
        # opening a new skill is less likely than reusing the most-used one
        ex, pnew = crp_predictive([8, 2], alpha=0.5)
        self.assertLess(pnew, max(ex))
        self.assertLess(new_skill_logpenalty([8, 2], 0.5), 0.0)   # negative => a cost

    def test_smaller_alpha_more_reuse(self):
        _, pnew_small = crp_predictive([5], alpha=0.2)
        _, pnew_big = crp_predictive([5], alpha=2.0)
        self.assertLess(pnew_small, pnew_big)             # smaller alpha -> fewer new skills

    def test_pitman_yor_grows_with_K(self):
        # with discount d>0, p_new increases as the number of distinct skills K grows
        _, pny_K1 = pitman_yor_predictive([10], alpha=1.0, d=0.5)
        _, pny_K3 = pitman_yor_predictive([4, 3, 3], alpha=1.0, d=0.5)
        self.assertLess(pny_K1, pny_K3)

    def test_ewens_eppf_finite(self):
        self.assertLess(crp_logprob([5, 3, 1], alpha=1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
