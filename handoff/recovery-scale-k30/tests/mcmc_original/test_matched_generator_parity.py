"""Generator/inference parity and the negative controls that prove the parity
checks have teeth.

Run:  PYTHONPATH=src .venv/bin/python -m unittest tests.mcmc_original.test_matched_generator_parity -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from hpop.mcmc_original import matched_generator_diagnostics as mgd
from hpop.mcmc_original import matched_synthetic_generator as msg
from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_rfs import (
    recurrent_step_probabilities, recurrent_validity_update,
)
from hpop.mcmc_original.recurrent_segmentation import RecurrentBlockScorer
from hpop.mcmc_original.stage6e_frozen import (
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH,
)


def _scorer_for(traces, truth):
    return RecurrentBlockScorer(
        traces=tuple(t.cpa for t in traces), epsilon=truth.epsilon,
        u_by_skill=truth.u_by_skill, beta=truth.beta, omega=truth.omega,
        lambda_rep=truth.lambda_rep, lambda_back=truth.lambda_back,
        max_width=truth.max_width, min_width=truth.min_width)


class TestCompleteDataLogProbParity(unittest.TestCase):
    def test_generator_vs_inference_decomposition(self):
        truth = msg.supplied_truth()
        corpus = msg.generate_corpus(90, [24, 32, 40, 48] * 4, [24, 32], truth)
        traces = corpus.train + corpus.heldout
        scorer = _scorer_for(traces, truth)
        log_c = {J: mgd.exact_normalizer(J, truth.delta_b, truth.min_width,
                                         truth.max_width)
                 for J in {t.length for t in traces}}
        for i, trace in enumerate(traces):
            generator_side = msg.generator_complete_data_log_prob(trace)
            inference_side = msg.inference_complete_data_log_prob(
                trace, truth, scorer, i, log_c[trace.length])
            self.assertTrue(math.isfinite(generator_side))
            self.assertTrue(math.isfinite(inference_side))
            self.assertLess(abs(generator_side - inference_side), 1e-10,
                            f"trace {i}")
            for value in trace.block_log_likelihoods:
                self.assertTrue(math.isfinite(value))

    def test_tiny_trace_log_evidence_forward_vs_enumeration(self):
        """Semi-Markov forward log Z vs direct enumeration on tiny traces."""
        from hpop.mcmc_original.stage6e_exact import (
            enumerate_states, log_evidence_forward, state_log_weights,
        )
        from hpop.mcmc_original.targets import logsumexp
        truth = msg.supplied_truth()
        corpus = msg.generate_corpus(91, [6, 7, 10], [], truth)
        scorer = _scorer_for(corpus.train, truth)
        with np.errstate(divide="ignore"):
            log_pi = np.log(truth.pi)
            log_transition = np.log(truth.transition)
        for i, trace in enumerate(corpus.train):
            states = enumerate_states(trace.length, truth.n_skills,
                                      truth.min_width, truth.max_width)
            weights = state_log_weights(states, i, trace.length, scorer,
                                        log_pi, log_transition, truth.delta_b)
            log_z_enum = float(logsumexp(weights))
            log_z_forward = log_evidence_forward(
                i, trace.length, truth.n_skills, scorer, log_pi, log_transition,
                truth.delta_b, truth.min_width, truth.max_width)
            self.assertLess(abs(log_z_forward - log_z_enum), 1e-10,
                            f"J={trace.length}")


class TestNegativeControlOldBlockCount(unittest.TestCase):
    """Negative control 1: the old mechanism (L ~ Uniform{4,5,6}, then iid widths)
    must FAIL the registered segment-count parity gate."""

    def test_exact_conditional_distribution_fails_tv_gate(self):
        J = 32
        widths = np.arange(MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH + 1)
        weights = (1.0 - DELTA_B) ** (widths - 1)
        width_p = weights / weights.sum()
        # exact L-fold convolutions: P(sum of L iid old-mechanism widths = J)
        old = np.zeros(J // MIN_BLOCK_WIDTH + 1)
        for L in (4, 5, 6):
            pmf = np.zeros(1)
            pmf[0] = 1.0
            for _ in range(L):
                nxt = np.zeros(len(pmf) + MAX_BLOCK_WIDTH)
                for w, p in zip(widths, width_p):
                    nxt[w:w + len(pmf)] += p * pmf
                pmf = nxt
            if J < len(pmf):
                old[L] = pmf[J] / 3.0            # L is uniform on {4, 5, 6}
        self.assertGreater(old.sum(), 0.0)
        old = old / old.sum()                    # condition on total length J
        exact = mgd.exact_segment_count_distribution(J, DELTA_B, MIN_BLOCK_WIDTH,
                                                     MAX_BLOCK_WIDTH)
        n = max(len(old), len(exact))
        old = np.pad(old, (0, n - len(old)))
        exact = np.pad(exact, (0, n - len(exact)))
        tv = mgd.total_variation(old, exact)
        self.assertGreater(tv, 0.01, "the parity gate failed to detect the old "
                                     f"block-count mechanism (TV={tv})")

    def test_sampled_old_mechanism_fails_tv_gate(self):
        rng = np.random.default_rng(51)
        J, target = 32, 4000
        widths = np.arange(MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH + 1)
        weights = (1.0 - DELTA_B) ** (widths - 1)
        width_p = weights / weights.sum()
        accepted = []
        while len(accepted) < target:
            L = int(rng.choice((4, 5, 6)))
            draw = rng.choice(widths, size=L, p=width_p)
            if int(draw.sum()) == J:
                accepted.append(tuple(int(w) for w in draw))
        exact = mgd.exact_segment_count_distribution(J, DELTA_B, MIN_BLOCK_WIDTH,
                                                     MAX_BLOCK_WIDTH)
        empirical = mgd.empirical_segment_count_distribution(accepted,
                                                             len(exact) - 1)
        self.assertGreater(mgd.total_variation(empirical, exact), 0.01)


class TestNegativeControlStateLeak(unittest.TestCase):
    """Negative control 2: carrying q across block boundaries must break the
    generator/inference log-probability parity."""

    @staticmethod
    def _leaky_trace(rng, widths, labels, truth):
        params = truth.rfs_parameters()
        q = np.zeros(truth.n_roles)              # deliberately NOT reset per block
        blocks, log_liks, q_starts = [], [], []
        for width, skill in zip(widths, labels):
            u = truth.u_by_skill[skill]
            precedence = precedence_from_u(u)
            q_starts.append(q.copy())
            roles, total = [], 0.0
            for _ in range(width):
                mixed = recurrent_step_probabilities(u, q, params)
                y = int(rng.choice(truth.n_roles, p=mixed))
                roles.append(y)
                total += math.log(float(mixed[y]))
                q = recurrent_validity_update(y, precedence, q, params.shared_omega)
            blocks.append(tuple(roles))
            log_liks.append(total)
        return blocks, log_liks, q_starts

    def test_leak_detected_by_parity_and_reset_check(self):
        truth = msg.supplied_truth()
        rng = np.random.default_rng(61)
        widths, labels = (4, 5, 4, 5), (0, 1, 2, 0)
        worst = 0.0
        leak_seen = False
        for _ in range(20):
            blocks, log_liks, q_starts = self._leaky_trace(rng, widths, labels,
                                                           truth)
            leak_seen = leak_seen or any(np.any(qs != 0.0) for qs in q_starts[1:])
            flat = tuple(r for b in blocks for r in b)
            scorer = _scorer_for(
                [msg.MatchedTrace(0, "train", len(flat), widths,
                                  tuple(np.cumsum(widths)[:-1].tolist()), labels,
                                  tuple(blocks), flat, 0.0, 0.0,
                                  tuple(log_liks))], truth)
            start = 0
            production = []
            for width, skill in zip(widths, labels):
                production.append(scorer.score(0, start, start + width, skill))
                start += width
            worst = max(worst, max(abs(a - b)
                                   for a, b in zip(log_liks, production)))
        self.assertTrue(leak_seen, "the leaky control never actually leaked state")
        self.assertGreater(worst, 1e-6,
                           "the parity check failed to detect recurrent-state "
                           "leakage across block boundaries")


class TestNegativeControlTerminalBoundary(unittest.TestCase):
    """Negative control 3: charging the final block a delta_B factor must break
    normalizer parity and log-probability parity.

    Note the normalized state distribution is INVARIANT to this fault (the extra
    factor is the same for every state), which is exactly why the registered
    checks include the normalizer and the cross-side log-probability, not just
    state-distribution TV.
    """

    def test_faulty_normalizer_detected(self):
        J = 10
        correct = mgd.exact_normalizer(J, DELTA_B, MIN_BLOCK_WIDTH,
                                       MAX_BLOCK_WIDTH)
        faulty = correct + math.log(DELTA_B)     # every state pays one extra delta_B
        enum = mgd.log_normalizer_from_enumeration(J, DELTA_B, MIN_BLOCK_WIDTH,
                                                   MAX_BLOCK_WIDTH)
        self.assertLess(abs(correct - enum), 1e-12)
        self.assertGreater(abs(faulty - enum), 1e-12)

    def test_faulty_convention_breaks_log_prob_parity(self):
        truth = msg.supplied_truth()
        corpus = msg.generate_corpus(71, [24], [], truth)
        trace = corpus.train[0]
        scorer = _scorer_for(corpus.train, truth)
        log_c = mgd.exact_normalizer(trace.length, truth.delta_b,
                                     truth.min_width, truth.max_width)
        inference_side = msg.inference_complete_data_log_prob(
            trace, truth, scorer, 0, log_c)
        faulty_generator_side = (msg.generator_complete_data_log_prob(trace)
                                 + math.log(truth.delta_b))
        self.assertGreater(abs(faulty_generator_side - inference_side), 1e-10)


if __name__ == "__main__":
    unittest.main()
