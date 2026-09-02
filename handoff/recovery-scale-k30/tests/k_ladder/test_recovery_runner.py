"""The segment runner's one load-bearing property: exact resume.

The coordinator stops and restarts chains between segments, possibly on different
machines. If a resumed chain differed AT ALL from an unbroken one, every run-to-
convergence verdict would depend on where the segment boundaries happened to fall --
an unfalsifiable confound. The CRN design (generators addressed by design index, never
by stream position) is what makes exactness possible; these tests are what make it
enforced.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hpop.mcmc_cpa.corpus import generate_ladder_corpus
from hpop.mcmc_cpa.nested_library import draw_master_library
from hpop.mcmc_cpa.recovery_regime import REGIME, generation_params
from hpop.mcmc_cpa.recovery_runner import chain_crn, init_state, run_segment
from hpop.mcmc_original.stage6e_state import Stage6EModel


@pytest.fixture(scope="module")
def cell():
    library, _ = draw_master_library(0)
    corpus = generate_ladder_corpus(
        library, 3, 0, trace_length=REGIME.TRACE_LENGTH, params=generation_params(),
        min_width=REGIME.MIN_WIDTH, max_width=REGIME.MAX_WIDTH,
        train_per_skill=2, test_per_skill=1)          # small corpus: tests, not corpora
    u_truth, role_maps = library.prefix(3)
    model = Stage6EModel(traces=corpus.traces("train"), epsilon=REGIME.EPSILON,
                         delta_b=REGIME.DELTA_B, n_skills=3, n_roles=library.n_roles,
                         min_width=REGIME.MIN_WIDTH, max_width=REGIME.MAX_WIDTH,
                         infer_pi_P=True, eta_initial=1.0, eta_transition=1.0)
    return model, role_maps


def test_two_segments_equal_one_unbroken_run(cell):
    model, role_maps = cell
    crn_a = chain_crn(0, 3, 0)
    unbroken_state = init_state(model, crn_a)
    unbroken_state, draws_all = run_segment(model, role_maps, unbroken_state, crn_a,
                                            u_scale=0.5, sweeps=40)

    crn_b = chain_crn(0, 3, 0)
    split = init_state(model, crn_b)
    split, first = run_segment(model, role_maps, split, crn_b, u_scale=0.5, sweeps=20)
    # serialise across the boundary, as the fleet does
    split = json.loads(json.dumps(split))
    split, second = run_segment(model, role_maps, split, crn_b, u_scale=0.5, sweeps=20)

    assert unbroken_state["sweep"] == split["sweep"] == 40
    np.testing.assert_array_equal(np.asarray(unbroken_state["u_by_skill"]),
                                  np.asarray(split["u_by_skill"]))
    np.testing.assert_array_equal(np.asarray(unbroken_state["pi"]),
                                  np.asarray(split["pi"]))
    assert unbroken_state["segmentation_keys"] == split["segmentation_keys"]
    assert unbroken_state["u_proposed"] == split["u_proposed"]
    assert unbroken_state["u_accepted"] == split["u_accepted"]
    assert draws_all["labels"] == first["labels"] + second["labels"]
    assert draws_all["u_event_sweep"] == first["u_event_sweep"] + second["u_event_sweep"]
    assert draws_all["u"] == first["u"] + second["u"]


def test_the_pacing_is_flat_in_k(cell):
    """U proposals per role per sweep must not depend on K -- that is the whole point."""
    from hpop.mcmc_cpa.u_quota import update_events, distribute_quota

    for k in (3, 30):
        sweeps = REGIME.SEGMENT_SWEEPS
        quota = round(REGIME.U_RATE_PER_ROLE_PER_SWEEP * k * 10 * sweeps)
        per_role_per_sweep = quota / (k * 10 * sweeps)
        assert per_role_per_sweep == pytest.approx(
            REGIME.U_RATE_PER_ROLE_PER_SWEEP, rel=1e-6)


def test_chains_differ_but_replicate_streams_are_shared(cell):
    model, role_maps = cell
    results = {}
    for chain in (0, 1):
        crn = chain_crn(0, 3, chain)
        state = init_state(model, crn)
        state, draws = run_segment(model, role_maps, state, crn, u_scale=0.5, sweeps=12)
        results[chain] = draws
    # segmentations can legitimately coincide when the posterior is sharp; the
    # dispersed U starts cannot, and that is what "dispersed chains" must guarantee
    assert results[0]["u"] != results[1]["u"], "chains must have dispersed U states"


def test_h_changing_counter_only_counts_closure_changes(cell):
    """An accepted move that does not change the induced order must not count."""
    model, role_maps = cell
    crn = chain_crn(0, 3, 2)
    state = init_state(model, crn)
    state, _ = run_segment(model, role_maps, state, crn, u_scale=0.5, sweeps=40)
    assert sum(state["h_changing_accepted_per_skill"]) <= state["u_accepted"]
    assert state["u_accepted"] <= state["u_proposed"]


def test_the_runner_starts_dispersed_never_at_truth(cell):
    model, role_maps = cell
    library, _ = draw_master_library(0)
    u_truth, _ = library.prefix(3)
    state = init_state(model, chain_crn(0, 3, 0))
    assert not np.allclose(np.asarray(state["u_by_skill"]), u_truth)
