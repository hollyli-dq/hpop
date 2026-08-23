"""Tests for the recurrent relaxed frontier likelihood.

Run:  PYTHONPATH=src .venv/bin/python -m unittest tests.test_recurrent -v
"""
import math
import unittest

import numpy as np

from hpop.inference.poset import Poset
from hpop.inference.recurrent import (RecurrentFrontier, hard_precedence_matrix,
                                      invalidation_matrix, sigmoid)

# a -> c, b -> c, c -> d ; a || b
ITEMS = ["a", "b", "c", "d"]
EDGES = [(0, 2), (1, 2), (2, 3)]


def hard_model(**kw):
    kw.setdefault("omega", 8.0)          # sigmoid(8) ~ 1: executing z fully invalidates its succ.
    return RecurrentFrontier(hard_precedence_matrix(4, EDGES), **kw)


class TestValidityState(unittest.TestCase):
    def test_initial_state_is_all_invalid(self):
        m = hard_model()
        q = np.zeros(m.M)
        # only the minimal elements a, b are frontier-feasible at t=1
        lp = m.step_logprobs(q)
        self.assertGreater(lp[0], lp[2])
        self.assertGreater(lp[1], lp[2])

    def test_execution_sets_validity_to_one(self):
        m = hard_model()
        q = m.update(np.zeros(m.M), 0)
        self.assertAlmostEqual(q[0], 1.0)

    def test_invalidation_only_flows_along_precedence(self):
        m = hard_model(omega=0.0)                       # sigmoid(0) = 0.5
        J = invalidation_matrix(m.D, 0.0)
        self.assertAlmostEqual(J[0, 2], 0.5)            # a precedes c -> invalidates it
        self.assertAlmostEqual(J[2, 0], 0.0)            # c does not precede a -> no effect
        # ... which is exactly why a failed test cannot invalidate the edit that preceded it.
        q = np.ones(m.M)
        q2 = m.update(q, 0)
        self.assertAlmostEqual(q2[2], 0.5)
        self.assertAlmostEqual(q2[3], 0.5)

    def test_full_invalidation_when_omega_large(self):
        m = hard_model(omega=20.0)
        q = m.update(np.ones(m.M), 0)
        self.assertLess(q[2], 1e-6)

    def test_no_invalidation_when_omega_very_negative(self):
        m = hard_model(omega=-20.0)
        q = m.update(np.ones(m.M), 0)
        self.assertAlmostEqual(q[2], 1.0, places=6)


class TestFrontierEquivalence(unittest.TestCase):
    """Eq. (frontier-equivalence): F^RFS(x) = 1 iff every predecessor is currently valid."""

    def test_matches_bpop_frontier_on_a_non_repeating_trace(self):
        m = hard_model(omega=-30.0)                     # nothing is ever invalidated
        poset = Poset(ITEMS, [(ITEMS[a], ITEMS[b]) for a, b in EDGES])
        q = np.zeros(m.M)
        completed = []
        for y in [0, 1, 2, 3]:
            stale = 1.0 - q
            A = 1.0 - m.D * stale[:, None]
            np.fill_diagonal(A, 1.0)
            F = np.prod(A, axis=0)
            feasible = {ITEMS[i] for i in range(m.M) if F[i] > 1 - 1e-9 and q[i] < 0.5}
            self.assertEqual(feasible, set(poset.frontier(completed)))
            q = m.update(q, y)
            completed.append(ITEMS[y])

    def test_hard_precedence_gives_binary_frontier(self):
        m = hard_model()
        q = np.zeros(m.M)
        A = 1.0 - m.D * (1 - q)[:, None]
        np.fill_diagonal(A, 1.0)
        F = np.prod(A, axis=0)
        self.assertTrue(np.all((F < 1e-12) | (F > 1 - 1e-12)))

    def test_relaxed_frontier_is_interior(self):
        D = 0.6 * hard_precedence_matrix(4, EDGES)
        m = RecurrentFrontier(D, omega=1.0)
        q = np.zeros(m.M)
        A = 1.0 - m.D * (1 - q)[:, None]
        np.fill_diagonal(A, 1.0)
        F = np.prod(A, axis=0)
        self.assertTrue(np.all(F > 0.0))
        self.assertLess(F[2], 1.0)


class TestLikelihood(unittest.TestCase):
    def test_valid_order_beats_violation(self):
        m = hard_model()
        good = m.logp([0, 1, 2, 3])
        bad = m.logp([2, 0, 1, 3])
        self.assertGreater(good, bad)

    def test_incomparable_orders_are_equally_likely(self):
        m = hard_model()
        self.assertAlmostEqual(m.logp([0, 1, 2, 3]), m.logp([1, 0, 2, 3]), places=9)

    def test_eps_zero_gives_violations_zero_probability(self):
        """The manuscript's Eq. (recurrent-step-likelihood) has no noise floor: with a hard
        precedence matrix an order violation has probability zero, so the objective is -inf (here
        only saved from -inf by the numerical floor) and gradients are undefined."""
        strict = hard_model(eps=0.0)
        violation = strict.logp([2, 0, 1, 3])
        self.assertLess(violation, -600.0)              # numerically zero probability
        robust = hard_model(eps=0.02)
        self.assertTrue(math.isfinite(robust.logp([2, 0, 1, 3])))
        self.assertGreater(robust.logp([2, 0, 1, 3]), -30.0)

    def test_repeat_cost_lowers_the_probability_of_the_repeating_step(self):
        """lambda_rep is a per-step cost on items whose output is already valid."""
        cheap, dear = hard_model(lam_rep=0.0, omega=-30.0), hard_model(lam_rep=4.0, omega=-30.0)
        q = np.zeros(4)
        for y in [0, 1, 2]:
            q = cheap.update(q, y)
        self.assertGreater(cheap.step_logprobs(q)[2], dear.step_logprobs(q)[2])

    def test_repeat_cost_does_not_lower_whole_sequence_likelihood(self):
        """But lambda_rep also suppresses every *other* currently-valid competitor, so raising it
        can *raise* the likelihood of a trace that contains repeats. The repeat penalty is a global
        reshaping of the step distribution, not a local cost that can be read off a trace score."""
        cheap, dear = hard_model(lam_rep=0.0, omega=-30.0), hard_model(lam_rep=4.0, omega=-30.0)
        seq = [0, 1, 2, 2, 3]                       # c executed twice, nothing invalidated it
        self.assertGreater(dear.logp(seq), cheap.logp(seq))

    def test_reexecuting_a_stale_item_is_cheaper_than_reexecuting_a_valid_one(self):
        m = hard_model(lam_rep=3.0, omega=20.0)
        # after re-running `a`, `c` is stale, so re-running `c` is cheap
        stale_rerun = m.logp([0, 1, 2, 3, 0, 2])
        # `b` is never invalidated by `a`, so re-running `b` pays the full repeat cost
        valid_rerun = m.logp([0, 1, 2, 3, 0, 1])
        self.assertGreater(stale_rerun, valid_rerun)

    def test_backward_jump_cost_penalizes_invalidating_moves(self):
        cheap = hard_model(lam_back=0.0, omega=20.0)
        dear = hard_model(lam_back=3.0, omega=20.0)
        seq = [0, 1, 2, 3, 0]                       # the last step invalidates valid c and d
        self.assertGreater(cheap.logp(seq), dear.logp(seq))

    def test_repair_loop_stays_acyclic(self):
        """An edit-test-repair cycle is representable without any cycle in the latent order."""
        m = hard_model(omega=20.0)
        loop = m.logp([0, 1, 2, 3, 0, 2, 3])
        self.assertTrue(math.isfinite(loop))
        self.assertEqual(int(np.sum(np.diag(m.D))), 0)


class TestThetaGradient(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        theta = rng.normal(size=4) * 0.5
        m = hard_model(theta=theta)
        seq = [0, 1, 2, 3, 0, 2]
        lp, grad = m.logp_and_theta_grad(seq)
        self.assertAlmostEqual(lp, m.logp(seq), places=9)
        h = 1e-6
        for v in range(4):
            t2 = theta.copy()
            t2[v] += h
            num = (hard_model(theta=t2).logp(seq) - lp) / h
            self.assertAlmostEqual(num, grad[v], places=4)


class TestCompositionTerm(unittest.TestCase):
    def test_theta_shifts_mass_towards_preferred_roles(self):
        base = hard_model()
        biased = hard_model(theta=np.array([3.0, -3.0, 0.0, 0.0]))
        p_base = np.exp(base.step_logprobs(np.zeros(4)))
        p_bias = np.exp(biased.step_logprobs(np.zeros(4)))
        self.assertGreater(p_bias[0], p_base[0])
        self.assertLess(p_bias[1], p_base[1])


if __name__ == "__main__":
    unittest.main()
