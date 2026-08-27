"""Condition B machinery: target parity, kernel reuse, invariances, resume.

Run:  PYTHONPATH=src .venv/bin/python -m unittest tests.mcmc_original.test_matched_condition_b -v
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import unittest

import numpy as np

from hpop.mcmc_original import matched_condition_b as mcb
from hpop.mcmc_original import matched_synthetic_generator as msg
from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_joint_scalar_mcmc import (
    vectorized_state_features,
)
from hpop.mcmc_original.recurrent_rfs import recurrent_rfs_log_likelihood
from hpop.mcmc_original.sampler_u import log_u_prior, u_row_sweep

ROOT = pathlib.Path(__file__).resolve().parents[2]
RHO_0 = 0.5


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_matched_condition_b", ROOT / "scripts/run_matched_condition_b.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small_setup(seed=515151, train=(24, 32, 40), heldout=(24,)):
    truth = msg.supplied_truth()
    corpus = msg.generate_corpus(seed, train, heldout, truth)
    blocks = mcb.oracle_blocks_by_skill(corpus.train, truth.n_skills)
    likelihood = mcb.OracleBlockLikelihood(
        blocks, truth.beta, truth.epsilon, truth.omega, truth.lambda_rep,
        truth.lambda_back)
    target = mcb.ConditionBTarget(likelihood=likelihood, rho_0=RHO_0)
    return truth, corpus, target


class TestTargetParity(unittest.TestCase):
    def setUp(self):
        self.truth, self.corpus, self.target = _small_setup()

    def _direct_ll(self, u):
        return sum(recurrent_rfs_log_likelihood(
            block, u[k], self.truth.beta, self.truth.epsilon, self.truth.omega,
            self.truth.lambda_rep, self.truth.lambda_back)
            for t in self.corpus.train
            for k, block in zip(t.labels, t.role_blocks))

    def test_likelihood_and_target_parity(self):
        rng = np.random.default_rng(0)
        for scale in (0.0, 0.3, 1.0):
            u = self.truth.u_by_skill + scale * rng.standard_normal(
                self.truth.u_by_skill.shape)
            self.assertLess(abs(self.target.likelihood.total(u)
                                - self._direct_ll(u)), 1e-10)
            expected = self._direct_ll(u) + sum(
                log_u_prior(u[k], RHO_0) for k in range(3))
            self.assertLess(abs(self.target.log_target(u) - expected), 1e-10)

    def test_prior_includes_gaussian_determinant(self):
        """p(U | rho) must carry -(1/2) log|Sigma_rho| per row (scipy pin)."""
        from scipy.stats import multivariate_normal
        from hpop.mcmc_original.sampler_u import sigma_rho_matrix
        rng = np.random.default_rng(1)
        u = rng.standard_normal((3, 5, 2))
        for rho in (0.2, RHO_0, 0.8):
            sigma = sigma_rho_matrix(2, rho)
            expected = sum(float(multivariate_normal(np.zeros(2), sigma)
                                 .logpdf(u[k, j]))
                           for k in range(3) for j in range(5))
            actual = sum(log_u_prior(u[k], rho) for k in range(3))
            self.assertLess(abs(actual - expected), 1e-9)
        # dropping the determinant is detected: rho=0.2 vs 0.8 differ even for
        # U = 0 where the quadratic form vanishes
        zero = np.zeros((5, 2))
        self.assertGreater(abs(log_u_prior(zero, 0.2)
                               - log_u_prior(zero, 0.8)), 0.1)

    def test_oracle_block_q0_reset(self):
        for k in range(3):
            per_block = self.target.likelihood.skill_block_log_likelihoods(
                k, self.truth.u_by_skill[k])
            direct = [recurrent_rfs_log_likelihood(
                row, self.truth.u_by_skill[k], self.truth.beta,
                self.truth.epsilon, self.truth.omega, self.truth.lambda_rep,
                self.truth.lambda_back)
                for arr in self.target.likelihood.blocks_by_skill[k].values()
                for row in arr]
            self.assertLess(float(np.abs(per_block
                                         - np.asarray(direct)).max()), 1e-10)


class TestKernelReuseAndFixedVariables(unittest.TestCase):
    def setUp(self):
        self.truth, self.corpus, self.target = _small_setup()

    def test_u_proposal_parity_with_validated_sampler_u(self):
        """One Condition-B sweep is bit-identical to sampler_u.u_row_sweep
        driven with the same RNG stream, skill by skill."""
        start = np.random.default_rng(7).standard_normal((3, 5, 2))
        chain = mcb.ConditionBChain(
            target=self.target, u_by_skill=start.copy(), sigma_u=0.5,
            rng=np.random.default_rng(99))
        chain.run_sweeps(3)

        reference = start.copy()
        rng = np.random.default_rng(99)
        ref_lik = mcb.OracleBlockLikelihood(
            self.target.likelihood.blocks_by_skill, self.truth.beta,
            self.truth.epsilon, self.truth.omega, self.truth.lambda_rep,
            self.truth.lambda_back)
        for _ in range(3):
            for k in range(3):
                updated, _, _, _ = u_row_sweep(
                    reference[k],
                    lambda u, kk=k: ref_lik.skill_log_likelihood(kk, u),
                    RHO_0, 0.5, rng)
                reference[k] = updated
        np.testing.assert_array_equal(chain.u_by_skill, reference)

    def test_fixed_s_z_never_change_and_no_scalar_state(self):
        blocks_before = {
            k: {w: arr.copy() for w, arr in groups.items()}
            for k, groups in self.target.likelihood.blocks_by_skill.items()}
        chain = mcb.ConditionBChain(
            target=self.target,
            u_by_skill=np.random.default_rng(3).standard_normal((3, 5, 2)),
            sigma_u=0.5, rng=np.random.default_rng(4))
        chain.run_sweeps(50)
        for k, groups in blocks_before.items():
            for w, arr in groups.items():
                np.testing.assert_array_equal(
                    self.target.likelihood.blocks_by_skill[k][w], arr)
        for name in ("segmentations", "boundaries", "labels", "beta", "omega",
                     "lambda_rep", "lambda_back", "pi", "transition", "rho"):
            self.assertFalse(hasattr(chain, name),
                             f"chain unexpectedly carries '{name}'")
        self.assertEqual(self.target.rho_0, RHO_0)

    def test_no_ffbs_no_collapsed_u_no_pi_p_no_scalar_calls(self):
        for name in ("src/hpop/mcmc_original/matched_condition_b.py",
                     "scripts/run_matched_condition_b.py"):
            source = (ROOT / name).read_text()
            # import/call forms only — the source manifest legitimately NAMES
            # the untouched modules in documentation strings
            for token in ("import collapsed_u", "from hpop.mcmc_original."
                          "collapsed", "import semi_markov_ffbs",
                          "ffbs_draw(", "sample_transition_matrix(",
                          "dirichlet_posterior_params(",
                          "SemiMarkovPosterior", "scalar_sweep"):
                self.assertNotIn(token, source, f"{name} contains '{token}'")


class TestInvariancesAndMetrics(unittest.TestCase):
    def test_summaries_invariant_to_latent_column_permutation(self):
        rng = np.random.default_rng(11)
        u = rng.standard_normal((3, 5, 2))
        permuted = u[:, :, ::-1].copy()
        np.testing.assert_array_equal(mcb.relation_indicator_vector(u),
                                      mcb.relation_indicator_vector(permuted))
        for k in range(3):
            self.assertEqual(
                mcb.canonical_h_hash(precedence_from_u(u[k])),
                mcb.canonical_h_hash(precedence_from_u(permuted[k])))

    def test_transitive_reduction_and_metric_separation(self):
        # chain 0 > 1 > 2 with roles 3, 4 isolated
        closure = np.zeros((5, 5), dtype=bool)
        closure[0, 1] = closure[1, 2] = closure[0, 2] = True
        reduction = mcb.transitive_reduction(closure)
        expected = np.zeros((5, 5), dtype=bool)
        expected[0, 1] = expected[1, 2] = True
        np.testing.assert_array_equal(reduction, expected)
        # a marginal matrix equal to the closure recovers it exactly at 0.5
        marginals = closure.astype(float)
        cm = mcb.closure_metrics(marginals, closure)
        self.assertEqual(cm["f1"], 1.0)
        self.assertTrue(cm["exact"])
        rm = mcb.reduction_metrics(closure, closure)
        self.assertEqual(rm["f1"], 1.0)
        # comparing the closure against the reduction as if both were closures
        # is the category error the split exists for: F1 drops below 1
        wrong = mcb.closure_metrics(reduction.astype(float), closure)
        self.assertLess(wrong["f1"], 1.0)
        inc = mcb.incomparable_metrics(marginals, closure)
        self.assertEqual(inc["f1"], 1.0)
        self.assertEqual(inc["n_true_incomparable"], 7)

    def test_features_from_closure_matches_production(self):
        runner = _load_runner()
        rng = np.random.default_rng(21)
        u = rng.standard_normal((5, 2))
        arr = rng.integers(0, 5, size=(4, 6))
        production = vectorized_state_features(arr, u, 1.7346)
        by_closure = runner.features_from_closure(
            arr, precedence_from_u(u), 1.7346)
        for key in ("F", "Q", "q", "C_back"):
            np.testing.assert_array_equal(production[key], by_closure[key])

    def test_heldout_scoring_parity(self):
        runner = _load_runner()
        truth, corpus, target = _small_setup(train=(24,), heldout=(24, 32))
        heldout_blocks = mcb.oracle_blocks_by_skill(corpus.heldout, 3)
        heldout_lik = mcb.OracleBlockLikelihood(
            heldout_blocks, truth.beta, truth.epsilon, truth.omega,
            truth.lambda_rep, truth.lambda_back)
        for k in range(3):
            if not heldout_blocks[k]:
                continue
            closure = precedence_from_u(truth.u_by_skill[k])
            via_closure = runner._per_block_for_closure(
                heldout_blocks[k], closure, truth)
            via_lik = heldout_lik.skill_block_log_likelihoods(
                k, truth.u_by_skill[k])
            self.assertLess(float(np.abs(via_closure - via_lik).max()), 1e-10)


class TestDeterminism(unittest.TestCase):
    def test_checkpoint_resume_bit_identical(self):
        truth, corpus, target = _small_setup(train=(24, 32))
        chain = mcb.ConditionBChain(
            target=target,
            u_by_skill=np.random.default_rng(5).standard_normal((3, 5, 2)),
            sigma_u=0.5, rng=np.random.default_rng(6))
        chain.run_sweeps(120)
        payload = chain.checkpoint()
        chain.run_sweeps(80)
        resumed = mcb.ConditionBChain.resume(payload, target)
        resumed.run_sweeps(80)
        np.testing.assert_array_equal(chain.u_by_skill, resumed.u_by_skill)
        self.assertEqual(chain.accepted, resumed.accepted)
        self.assertEqual(chain.h_change_accepted, resumed.h_change_accepted)
        self.assertEqual(chain.summary_row()["h_hashes"],
                         resumed.summary_row()["h_hashes"])

    def test_deterministic_starts_and_seed_manifest(self):
        runner = _load_runner()
        truth = msg.supplied_truth()
        starts_a = [runner.make_start(truth, c) for c in range(4)]
        starts_b = [runner.make_start(truth, c) for c in range(4)]
        for a, b in zip(starts_a, starts_b):
            np.testing.assert_array_equal(a, b)
        hashes = [runner.h_tuple_hashes(s) for s in starts_a]
        self.assertEqual(len(set(hashes)), 4)
        self.assertNotIn(runner.h_tuple_hashes(truth.u_by_skill), hashes)
        previously_used = {6_053_000, 6_063_000, 6_100_001, 6_200_001,
                           6_200_777, 6_201_001}
        new_seeds = (set(runner.FORMAL_SEEDS) | set(runner.START_SEEDS)
                     | set(runner.PILOT_SEEDS)
                     | set(runner.PRIOR_CHECK_MCMC_SEEDS)
                     | {runner.PRIOR_CHECK_IID_SEED, runner.PARITY_SEED})
        self.assertFalse(new_seeds & previously_used)
        self.assertEqual(len(runner.FORMAL_SEEDS), 4)
        self.assertEqual(len(set(runner.FORMAL_SEEDS)), 4)


if __name__ == "__main__":
    unittest.main()
