"""Step 7B — the sweep: exact conditionals, untouched kernels, and reproducibility.

The parity tests here are regressions rather than the primary argument. `ffbs_sweep_once`
*calls* `stage6e_sampler.sweep_once` for every phase after `(S, z)`, so the Stage 6E
kernels are not reimplemented and cannot drift; what these tests guard is that the wrapper
never perturbs them — not by an extra random draw, not by a mutated state, not by a
reordered phase.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_original.fast_segmentation_kernel import key_of, segmentation_of
from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import (
    FFBSBlockTables, Stage7BSampler, ffbs_segmentation_draw, ffbs_sweep_once,
    run_stage7b_chain,
)
from hpop.mcmc_original.semi_markov_ffbs import forward
from hpop.mcmc_original.stage6d_frozen import REGISTERED_SCALES, SCALAR_ORDER
from hpop.mcmc_original.stage6e_exact import (
    enumerate_states, exact_posterior, state_log_weights, total_variation,
)
from hpop.mcmc_original.stage6e_frozen import MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH
from hpop.mcmc_original.stage6e_sampler import Stage6ESampler, sweep_once
from hpop.mcmc_original.stage6e_state import Stage6EState
from hpop.mcmc_original.transitions import log_transition_matrix
from tests.mcmc_original.test_stage7b_joint_target import (
    K, TRACES, make_model, make_state,
)


# ---------------------------------------------------------------- the exact conditional
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_ffbs_conditional_is_the_enumerated_conditional(seed):
    """Per trace, the chart normalises exactly `p(S_n, z_n | x_n, Theta)`."""
    model = make_model()
    rng = np.random.default_rng(seed)
    state = make_state(rng)
    tables = FFBSBlockTables(model=model, source="batched")
    tables.refresh(state)
    scorer = model.scorer_for(state)
    log_pi = np.log(state.pi)
    log_p = log_transition_matrix(state.transition)
    states = enumerate_states(len(TRACES[0]), K, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
    for n, table in enumerate(tables.tables_for(state)):
        chart = forward(table, log_pi, log_p, model.delta_b, model.max_width,
                        model.min_width)
        weights = state_log_weights(states, n, len(TRACES[n]), scorer, log_pi, log_p,
                                    model.delta_b)
        exact = exact_posterior(states, weights)
        assert abs(chart.log_normalizer - exact["log_evidence"]) < 1e-10


def test_ffbs_draw_frequencies_match_the_exact_conditional():
    model = make_model()
    rng = np.random.default_rng(4)
    state = make_state(rng)
    tables = FFBSBlockTables(model=model, source="batched")
    tables.refresh(state)
    scorer = model.scorer_for(state)
    log_pi, log_p = np.log(state.pi), log_transition_matrix(state.transition)
    states = enumerate_states(len(TRACES[0]), K, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
    index = {s: i for i, s in enumerate(states)}

    counts = np.zeros((len(TRACES), len(states)))
    draw_rng = np.random.default_rng(99)
    for _ in range(20_000):
        drawn = ffbs_segmentation_draw(model, state, tables, draw_rng)
        for n, key in enumerate(drawn["keys"]):
            counts[n, index[key]] += 1.0
    empirical = counts / counts.sum(axis=1, keepdims=True)
    for n in range(len(TRACES)):
        weights = state_log_weights(states, n, len(TRACES[n]), scorer, log_pi, log_p,
                                    model.delta_b)
        exact = exact_posterior(states, weights)["probability"]
        assert total_variation(empirical[n], exact) < 0.02


def test_one_draw_per_trace_per_segmentation_sweep():
    model = make_model()
    rng = np.random.default_rng(5)
    state = make_state(rng)
    tables = FFBSBlockTables(model=model, source="batched")
    tables.refresh(state)
    drawn = ffbs_segmentation_draw(model, state, tables, rng)
    assert len(drawn["keys"]) == len(TRACES)
    assert len(drawn["log_normalizers"]) == len(TRACES)
    assert tables.reads == 1                      # one pass over the trace tables

    sampler = Stage7BSampler(model=model, scales=REGISTERED_SCALES)
    after = ffbs_sweep_once(state, sampler, rng)
    assert after.proposed["S_z_ffbs"] == len(TRACES)


def test_every_candidate_block_starts_from_q0_zero():
    """The tables are built by the audited Step 7A path; the invariant is rechecked here."""
    from hpop.mcmc_original.block_score_adapters import assert_no_recurrent_state_leak
    model = make_model()
    state = make_state(np.random.default_rng(6))
    scorer = model.scorer_for(state)
    for trace in range(len(TRACES)):
        audit = assert_no_recurrent_state_leak(scorer, trace, (0, 3, 0), (3, 8, 1))
        assert audit["bit_identical"]


# ------------------------------------------------------------------------ kernel parity
def test_every_phase_after_segmentation_is_stage6es_own(monkeypatch):
    """Segmentation draw off: Step 7B and Stage 6E return bit-identical states."""
    model = make_model()
    rng = np.random.default_rng(7)
    for _ in range(5):
        state = make_state(rng)
        a = ffbs_sweep_once(
            state, Stage7BSampler(model=model, scales=REGISTERED_SCALES,
                                  draw_segmentation=False), np.random.default_rng(808))
        b = sweep_once(
            state, Stage6ESampler(model=model, scales=dict(REGISTERED_SCALES),
                                  n_proposals_per_trace=0), np.random.default_rng(808))
        assert np.array_equal(a.u_by_skill, b.u_by_skill)             # U
        assert a.rho == b.rho                                          # rho
        for name in SCALAR_ORDER:                                      # the four scalars
            assert getattr(a, name) == getattr(b, name)
        assert np.array_equal(a.pi, b.pi)                              # pi
        assert np.array_equal(a.transition, b.transition)              # P
        assert a.components["log_target"] == b.components["log_target"]
        assert a.segmentations == b.segmentations


def test_the_wrapper_consumes_no_extra_randomness_when_the_draw_is_off():
    model = make_model()
    state = make_state(np.random.default_rng(8))
    rng_a, rng_b = np.random.default_rng(11), np.random.default_rng(11)
    ffbs_sweep_once(state, Stage7BSampler(model=model, scales=REGISTERED_SCALES,
                                          draw_segmentation=False), rng_a)
    sweep_once(state, Stage6ESampler(model=model, scales=dict(REGISTERED_SCALES),
                                     n_proposals_per_trace=0), rng_b)
    assert rng_a.bit_generator.state == rng_b.bit_generator.state


def test_the_sweep_does_not_mutate_the_state_it_was_given():
    model = make_model()
    state = make_state(np.random.default_rng(9))
    before_u = state.u_by_skill.copy()
    before_segmentations = state.segmentations
    before_beta = state.beta
    ffbs_sweep_once(state, Stage7BSampler(model=model, scales=REGISTERED_SCALES),
                    np.random.default_rng(12))
    assert np.array_equal(state.u_by_skill, before_u)
    assert state.segmentations == before_segmentations
    assert state.beta == before_beta


# ----------------------------------------------------------------------- reproducibility
def test_the_same_seed_gives_the_same_sweep():
    model = make_model()
    state = make_state(np.random.default_rng(10))
    outputs = []
    for _ in range(2):
        sampler = Stage7BSampler(model=model, scales=REGISTERED_SCALES)
        after = ffbs_sweep_once(state, sampler, np.random.default_rng(4242))
        outputs.append(after)
    a, b = outputs
    assert a.segmentations == b.segmentations
    assert np.array_equal(a.u_by_skill, b.u_by_skill)
    assert a.components["log_target"] == b.components["log_target"]


def test_a_chain_is_deterministic_and_resumes_bit_identically():
    model = make_model()
    start = make_state(np.random.default_rng(13))
    first = run_stage7b_chain(model=model, start=start.copy(), scales=REGISTERED_SCALES,
                              num_sweeps=120, burn_in=20, thin=5, seed=77)
    again = run_stage7b_chain(model=model, start=start.copy(), scales=REGISTERED_SCALES,
                              num_sweeps=120, burn_in=20, thin=5, seed=77)
    assert np.array_equal(first.u_draws, again.u_draws)
    assert first.boundary_keys == again.boundary_keys

    rng = np.random.default_rng(78)
    part = run_stage7b_chain(model=model, start=start.copy(), scales=REGISTERED_SCALES,
                             num_sweeps=60, burn_in=20, thin=5, seed=0, rng=rng)
    rest = run_stage7b_chain(model=model, start=start.copy(), scales=REGISTERED_SCALES,
                             num_sweeps=120, burn_in=20, thin=5, seed=0, rng=rng,
                             state=part.final_state)
    straight = run_stage7b_chain(model=model, start=start.copy(),
                                 scales=REGISTERED_SCALES, num_sweeps=120, burn_in=20,
                                 thin=5, seed=0, rng=np.random.default_rng(78))
    assert np.array_equal(np.concatenate([part.u_draws, rest.u_draws]),
                          straight.u_draws)


def test_the_state_survives_a_json_round_trip():
    model = make_model()
    start = make_state(np.random.default_rng(14))
    result = run_stage7b_chain(model=model, start=start, scales=REGISTERED_SCALES,
                               num_sweeps=40, burn_in=10, thin=5, seed=79)
    restored = Stage6EState.from_dict(result.final_state.to_dict())
    assert restored.segmentations == result.final_state.segmentations
    assert np.array_equal(restored.u_by_skill, result.final_state.u_by_skill)
    for name in SCALAR_ORDER:
        assert getattr(restored, name) == getattr(result.final_state, name)


def test_the_chain_records_movement_and_visits_more_than_one_segmentation():
    model = make_model()
    start = make_state(np.random.default_rng(15))
    result = run_stage7b_chain(model=model, start=start, scales=REGISTERED_SCALES,
                               num_sweeps=400, burn_in=50, thin=5, seed=80)
    assert result.movement["states_changed"] > 0
    assert result.movement["boundary_hamming"] > 0
    visited = {key for draw in result.boundary_keys for key in draw}
    assert len(visited) > 1
    assert all(np.isfinite(result.log_target))


def test_drawn_segmentations_are_always_legal():
    model = make_model()
    start = make_state(np.random.default_rng(16))
    result = run_stage7b_chain(model=model, start=start, scales=REGISTERED_SCALES,
                               num_sweeps=200, burn_in=20, thin=2, seed=81)
    for draw in result.boundary_keys:
        for n, key in enumerate(draw):
            spans = segmentation_of(key).segments
            assert spans[-1].end == len(TRACES[n])
            assert all(MIN_BLOCK_WIDTH <= s.length <= MAX_BLOCK_WIDTH for s in spans)
            labels = [s.skill for s in spans]
            assert all(a != b for a, b in zip(labels[:-1], labels[1:]))
            assert key_of(segmentation_of(key)) == key


def test_a_checkpointed_chain_resumes_bit_identically_through_json(tmp_path):
    """A 50,000-sweep Step 7B2 chain is many hours, so resume must survive serialisation.

    The in-memory resume test above keeps the state object and the generator alive. This
    one goes through the checkpoint the way a restart actually would: read the JSON,
    rebuild the state, restore the RNG from it, and continue.
    """
    import json

    from hpop.mcmc_original.stage6e_state import Stage6EState

    model = make_model()
    start = make_state(np.random.default_rng(17))
    part = run_stage7b_chain(model=model, start=start.copy(), scales=REGISTERED_SCALES,
                             num_sweeps=60, burn_in=10, thin=5, seed=99, chain=0,
                             checkpoint_path=str(tmp_path), checkpoint_every=20)
    payload = json.loads((tmp_path / "chain0_checkpoint.json").read_text())
    assert payload["sweep"] == 60
    assert (tmp_path / "chain0_checkpoint.npz").exists()

    restored = Stage6EState.from_dict(payload["state"])
    rng = np.random.default_rng(99)
    rng.bit_generator.state = restored.rng_state
    resumed = run_stage7b_chain(model=model, start=start.copy(),
                                scales=REGISTERED_SCALES, num_sweeps=120, burn_in=10,
                                thin=5, seed=99, chain=0, rng=rng, state=restored)
    straight = run_stage7b_chain(model=model, start=start.copy(),
                                 scales=REGISTERED_SCALES, num_sweeps=120, burn_in=10,
                                 thin=5, seed=99, chain=0)
    joined = np.concatenate([part.u_draws[:payload["n_retained"]], resumed.u_draws])
    assert np.array_equal(joined[:len(straight.u_draws)], straight.u_draws)
    joined_target = np.concatenate(
        [part.log_target[:payload["n_retained"]], resumed.log_target])
    assert np.array_equal(joined_target[:len(straight.log_target)], straight.log_target)


def test_no_checkpoint_is_written_when_it_is_not_asked_for(tmp_path):
    model = make_model()
    start = make_state(np.random.default_rng(18))
    run_stage7b_chain(model=model, start=start, scales=REGISTERED_SCALES, num_sweeps=30,
                      burn_in=5, thin=5, seed=100, chain=0, checkpoint_path=str(tmp_path))
    assert list(tmp_path.iterdir()) == []
