"""The partially-collapsed runner on the ACTUAL Step 7B1 reference problem (smoke).

The full gate evaluation is `scripts/collapsed_u_mixed_reference.py` at the registered
600k-sweep length; these tests only pin that on the reference problem itself (a) the
runner targets the same posterior object — with the collapsed move disabled it is
bit-identical to the validated Step 7B chain — and (b) the scheduled collapsed move
runs, scores, and moves structure without touching pi/P (which the reference fixes).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from hpop.mcmc_original.collapsed_u_kernel import (
    MOVE_NAME, CollapsedUConfig, run_collapsed_u_chain,
)
from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import run_stage7b_chain
from hpop.mcmc_original.stage6d_frozen import REGISTERED_SCALES
from hpop.mcmc_original.stage6e_frozen import DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH
from hpop.mcmc_original.stage6e_state import Stage6EModel

ROOT = Path(__file__).resolve().parent.parent.parent


def load_6e1b():
    path = ROOT / "scripts" / "stage6e1b_mixed_reference.py"
    spec = importlib.util.spec_from_file_location("stage6e1b", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_problem():
    module = load_6e1b()
    traces, _ = module.generate_corpus()
    mixed = module.build_mixed_model(traces)
    model = Stage6EModel(traces=traces, epsilon=module.EPSILON, delta_b=DELTA_B,
                         n_skills=module.K_SKILLS, n_roles=module.M_ROLES,
                         min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
                         infer_pi_P=False)
    start = module.dispersed_starts(mixed)[0]
    return model, start


def test_disabled_collapsed_move_is_bitwise_step7b_on_the_reference():
    model, start = reference_problem()
    baseline = run_stage7b_chain(model=model, start=start.copy(),
                                 scales=REGISTERED_SCALES, num_sweeps=60, burn_in=10,
                                 thin=2, seed=8_153_900, store_labels=False)
    model2, start2 = reference_problem()
    ours = run_collapsed_u_chain(model=model2, start=start2.copy(),
                                 scales=REGISTERED_SCALES, num_sweeps=60, burn_in=10,
                                 thin=2, seed=8_153_900,
                                 collapsed=CollapsedUConfig(every=0),
                                 store_labels=False)
    assert np.array_equal(baseline.u_draws, ours.u_draws)
    assert np.array_equal(baseline.log_target, ours.log_target)
    assert baseline.boundary_keys == ours.boundary_keys


def test_scheduled_moves_run_and_pi_p_stay_fixed_on_the_reference():
    model, start = reference_problem()
    pi_before = np.array(start.pi, copy=True)
    p_before = np.array(start.transition, copy=True)
    result = run_collapsed_u_chain(model=model, start=start,
                                   scales=REGISTERED_SCALES, num_sweeps=50, burn_in=5,
                                   thin=1, seed=8_153_901,
                                   collapsed=CollapsedUConfig(every=10),
                                   store_labels=False)
    assert result.proposed[MOVE_NAME] == 5                # sweeps 9, 19, 29, 39, 49
    assert [r["sweep"] for r in result.collapsed_records] == [9, 19, 29, 39, 49]
    final = result.final_state
    assert np.array_equal(final.pi, pi_before)            # the reference fixes pi/P
    assert np.array_equal(final.transition, p_before)
    assert all(np.isfinite(r["log_alpha"]) for r in result.collapsed_records
               if not r["invalid"])
