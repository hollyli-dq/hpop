"""Condition C composition: fixed coordinates, verbatim kernel reuse, parity.

Run:  PYTHONPATH=src .venv/bin/python -m unittest tests.mcmc_original.test_matched_condition_c -v
"""

from __future__ import annotations

import math
import pathlib
import unittest

import numpy as np

from hpop.mcmc_original import matched_condition_c as mcc
from hpop.mcmc_original import matched_synthetic_generator as msg
from hpop.mcmc_original.collapsed_u_kernel import (
    CollapsedUConfig, is_collapsed_sweep,
)
from hpop.mcmc_original.fast_segmentation_kernel import key_of
from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.sampler_u import propose_row
from hpop.mcmc_original.stage6c_frozen import log_structural_prior
from hpop.mcmc_original.stage6e_sampler import SkillBlockLikelihood

ROOT = pathlib.Path(__file__).resolve().parents[2]
RHO_0 = 0.5


def _setup(lengths=(6, 7, 10, 12), seed=616161):
    truth = msg.supplied_truth()
    corpus = msg.generate_corpus(seed, lengths, (), truth)
    model = mcc.build_condition_c_model(tuple(t.cpa for t in corpus.train))
    fixed = mcc.ConditionCFixed.from_truth(truth, RHO_0)
    return truth, corpus, model, fixed


class TestFixedCoordinates(unittest.TestCase):
    def test_fixed_variables_never_move(self):
        truth, corpus, model, fixed = _setup()
        u0 = np.random.default_rng(1).standard_normal((3, 5, 2))
        result = mcc.run_condition_c_chain(
            model, fixed, u0, 0.5, CollapsedUConfig(every=5, scale=0.5),
            num_sweeps=60, burn_in=10, thin=2, seed=2)
        state = result["final_state"]
        self.assertEqual(float(state.rho), RHO_0)
        self.assertEqual(float(state.beta), truth.beta)
        self.assertEqual(float(state.omega), truth.omega)
        self.assertEqual(float(state.lambda_rep), truth.lambda_rep)
        self.assertEqual(float(state.lambda_back), truth.lambda_back)
        np.testing.assert_array_equal(state.pi, np.asarray(truth.pi))
        np.testing.assert_array_equal(state.transition,
                                      np.asarray(truth.transition))

    def test_assert_unchanged_detects_a_moved_coordinate(self):
        truth, corpus, model, fixed = _setup()
        u0 = np.zeros((3, 5, 2))
        state = mcc.initial_condition_c_state(model, u0, fixed)
        fixed.assert_unchanged(state)
        state.rho = 0.51
        with self.assertRaises(AssertionError):
            fixed.assert_unchanged(state)

    def test_initial_state_is_legal_and_deterministic(self):
        truth, corpus, model, fixed = _setup(lengths=(6, 7, 10, 13, 14, 24,
                                                      32, 40, 48))
        u0 = np.zeros((3, 5, 2))
        a = mcc.initial_condition_c_state(model, u0, fixed)
        b = mcc.initial_condition_c_state(model, u0, fixed)
        self.assertEqual([key_of(s) for s in a.segmentations],
                         [key_of(s) for s in b.segmentations])
        for segmentation, trace in zip(a.segmentations, model.traces):
            self.assertEqual(segmentation.segments[-1].end, len(trace))
            for seg in segmentation.segments:
                self.assertTrue(model.min_width <= seg.length
                                <= model.max_width)
            for left, right in zip(segmentation.segments[:-1],
                                   segmentation.segments[1:]):
                self.assertNotEqual(left.skill, right.skill)


class TestComposition(unittest.TestCase):
    def test_cadence_semantics(self):
        fires = [i for i in range(40) if is_collapsed_sweep(i, 10)]
        self.assertEqual(fires, [9, 19, 29, 39])
        self.assertEqual([i for i in range(40) if is_collapsed_sweep(i, 0)],
                         [])

    def test_every_zero_runs_no_collapsed_move(self):
        truth, corpus, model, fixed = _setup()
        u0 = np.random.default_rng(3).standard_normal((3, 5, 2))
        result = mcc.run_condition_c_chain(
            model, fixed, u0, 0.5, CollapsedUConfig(every=0, scale=0.5),
            num_sweeps=40, burn_in=5, thin=1, seed=4)
        self.assertEqual(result["collapsed_proposed"], 0)
        self.assertEqual(result["collapsed_records"], [])
        result10 = mcc.run_condition_c_chain(
            model, fixed, u0, 0.5, CollapsedUConfig(every=10, scale=0.5),
            num_sweeps=40, burn_in=5, thin=1, seed=4)
        self.assertEqual(result10["collapsed_proposed"], 4)
        self.assertEqual([r["sweep"] for r in result10["collapsed_records"]],
                         [9, 19, 29, 39])

    def test_collapsed_move_is_the_verbatim_kernel(self):
        source = (ROOT / "src/hpop/mcmc_original/matched_condition_c.py"
                  ).read_text()
        self.assertIn("from hpop.mcmc_original.collapsed_u_kernel import",
                      source)
        self.assertIn("collapsed_u_mh_step(", source)
        self.assertIn("ffbs_segmentation_draw(", source)
        # no local reimplementation of the collapsed delta or the FFBS draw
        for token in ("delta_for_candidate", "backward_sample(",
                      "log_normalizer ="):
            self.assertNotIn(token, source)

    def test_conditional_u_phase_parity(self):
        """The U phase is the Stage 6E phase-3 arithmetic: a manual replica
        driven by the same RNG reproduces the sweep's U bit for bit."""
        truth, corpus, model, fixed = _setup()
        sampler = mcc.ConditionCSampler(
            model=model, fixed=fixed, u_scale=0.5,
            collapsed=CollapsedUConfig(every=0, scale=0.5))
        state = mcc.initial_condition_c_state(
            model, np.random.default_rng(5).standard_normal((3, 5, 2)), fixed)
        rng = np.random.default_rng(6)
        new_state, record, info = mcc.condition_c_sweep_once(state, sampler,
                                                             rng)

        # replay: consume the FFBS randomness identically, then run the manual
        # row loop with the SAME stream
        from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import (
            FFBSBlockTables, ffbs_segmentation_draw,
        )
        from hpop.mcmc_original.fast_segmentation_kernel import segmentation_of
        rng2 = np.random.default_rng(6)
        replica = state.copy()
        tables = FFBSBlockTables(model=model, source="batched")
        tables.refresh(replica)
        ffbs = ffbs_segmentation_draw(model, replica, tables, rng2)
        replica.segmentations = tuple(segmentation_of(k)
                                      for k in ffbs["keys"])
        skill_ll = SkillBlockLikelihood(traces=model.traces,
                                        epsilon=model.epsilon)
        skill_ll.set_blocks(replica.segmentations, model.n_skills)
        u = np.array(replica.u_by_skill, dtype=float, copy=True)
        current_ll = {k: skill_ll.full_replay(k, u[k], replica.beta,
                                              replica.omega,
                                              replica.lambda_rep,
                                              replica.lambda_back)
                      for k in range(3)}
        current_prior = {k: log_structural_prior(u[k], replica.rho)
                         for k in range(3)}
        for k in range(3):
            for row in range(5):
                candidate = propose_row(u[k], row, 0.5, rng2)
                cand_prior = log_structural_prior(candidate, replica.rho)
                if not math.isfinite(cand_prior):
                    continue
                cand_ll = skill_ll.full_replay(k, candidate, replica.beta,
                                               replica.omega,
                                               replica.lambda_rep,
                                               replica.lambda_back)
                if not math.isfinite(cand_ll):
                    continue
                log_alpha = ((cand_ll - current_ll[k])
                             + (cand_prior - current_prior[k]))
                if log_alpha >= 0.0 or math.log(rng2.random()) < log_alpha:
                    u[k] = candidate
                    current_ll[k] = cand_ll
                    current_prior[k] = cand_prior
        np.testing.assert_array_equal(new_state.u_by_skill, u)
        self.assertEqual([key_of(s) for s in new_state.segmentations],
                         [key_of(s) for s in replica.segmentations])

    def test_deterministic_and_seed_sensitive(self):
        truth, corpus, model, fixed = _setup()
        u0 = np.random.default_rng(8).standard_normal((3, 5, 2))
        config = CollapsedUConfig(every=10, scale=0.5)
        a = mcc.run_condition_c_chain(model, fixed, u0, 0.5, config,
                                      num_sweeps=50, burn_in=10, thin=2,
                                      seed=9, store_keys=True)
        b = mcc.run_condition_c_chain(model, fixed, u0, 0.5, config,
                                      num_sweeps=50, burn_in=10, thin=2,
                                      seed=9, store_keys=True)
        c = mcc.run_condition_c_chain(model, fixed, u0, 0.5, config,
                                      num_sweeps=50, burn_in=10, thin=2,
                                      seed=10, store_keys=True)
        self.assertEqual(a["retained"]["log_target"],
                         b["retained"]["log_target"])
        self.assertEqual(a["retained"]["keys"], b["retained"]["keys"])
        self.assertNotEqual(a["retained"]["log_target"],
                            c["retained"]["log_target"])

    def test_ffbs_refresh_runs_immediately_after_collapsed_move(self):
        """Source-level ordering pin: the collapsed branch precedes the FFBS
        refresh which precedes the conditional U loop, within the sweep."""
        source = (ROOT / "src/hpop/mcmc_original/matched_condition_c.py"
                  ).read_text()
        sweep = source[source.index("def condition_c_sweep_once"):]
        sweep = sweep[:sweep.index("def run_condition_c_chain")]
        i_collapsed = sweep.index("collapsed_u_mh_step(")
        i_ffbs = sweep.index("ffbs_segmentation_draw(")
        i_rows = sweep.index("propose_row(")
        self.assertLess(i_collapsed, i_ffbs)
        self.assertLess(i_ffbs, i_rows)


if __name__ == "__main__":
    unittest.main()
