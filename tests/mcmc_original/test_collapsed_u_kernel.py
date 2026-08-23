"""The collapsed-U likelihood and MH move: parity, invariances, cadence, cache.

Everything here runs on a deliberately tiny two-trace model so the whole file stays in
seconds; the full-size behaviour is pinned by the C0/C1 audit artifacts and by the mixed
reference run, not by unit tests.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from hpop.mcmc_original.block_score_adapters import build_log_block_scores
from hpop.mcmc_original.collapsed_u_kernel import (
    MOVE_NAME, CollapsedUConfig, collapsed_u_mh_step, is_collapsed_sweep,
    run_collapsed_u_chain,
)
from hpop.mcmc_original.collapsed_u_likelihood import CollapsedULikelihood
from hpop.mcmc_original.fast_block_tables import FastBlockScoreTable
from hpop.mcmc_original.fast_segmentation_kernel import segmentation_of
from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import run_stage7b_chain
from hpop.mcmc_original.sampler_u import propose_row
from hpop.mcmc_original.stage6c_frozen import log_structural_prior
from hpop.mcmc_original.stage6e_exact import (
    enumerate_states, exact_posterior, state_log_weights,
)
from hpop.mcmc_original.stage6e_state import Stage6EModel, Stage6EState
from hpop.mcmc_original.transitions import log_transition_matrix

ROOT = Path(__file__).resolve().parent.parent.parent

TRACES = ((0, 1, 2, 0, 1, 2), (2, 0, 1, 0, 1, 2, 0))
SCALES = {"U": 0.5, "rho": 0.5, "beta": 0.05, "omega": 0.18, "lambda_rep": 0.04,
          "lambda_back": 0.09}


def tiny_model() -> Stage6EModel:
    return Stage6EModel(traces=TRACES, epsilon=0.02, delta_b=0.15, n_skills=2,
                        n_roles=3, min_width=3, max_width=6, infer_pi_P=False)


def tiny_state(seed: int = 0) -> Stage6EState:
    rng = np.random.default_rng(seed)
    return Stage6EState(
        segmentations=(segmentation_of(((6, 0),)),
                       segmentation_of(((3, 0), (7, 1)))),
        u_by_skill=rng.normal(size=(2, 3, 2)), rho=0.2, beta=1.0, omega=0.5,
        lambda_rep=0.3, lambda_back=0.2, pi=np.array([0.6, 0.4]),
        transition=np.array([[0.0, 1.0], [1.0, 0.0]]))


def load_fast_audit():
    path = ROOT / "scripts" / "collapsed_u_fast_audit.py"
    spec = importlib.util.spec_from_file_location("collapsed_u_fast_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------- parity
def test_collapsed_likelihood_matches_the_validated_audit_scorer():
    """Bit-identical to the C0/C1 audit code path: same tables, same forward call."""
    fast = load_fast_audit()
    model, state = tiny_model(), tiny_state()
    likelihood = CollapsedULikelihood(model=model)
    ours = likelihood.log_z_per_trace(state)

    table = FastBlockScoreTable(traces=model.traces, epsilon=model.epsilon,
                                n_skills=model.n_skills, min_width=model.min_width,
                                max_width=model.max_width, n_roles=model.n_roles)
    table.refresh(state.u_by_skill, state.beta, state.omega, state.lambda_rep,
                  state.lambda_back)
    audit = fast.collapsed_log_z(table, model, np.log(state.pi),
                                 log_transition_matrix(state.transition))
    assert np.array_equal(ours, audit)


def test_forward_normaliser_matches_state_enumeration():
    """`log Z_n` against the registered per-state enumeration — an independent route."""
    model, state = tiny_model(), tiny_state()
    likelihood = CollapsedULikelihood(model=model)
    log_z = likelihood.log_z_per_trace(state)
    log_pi = np.log(state.pi)
    log_p = log_transition_matrix(state.transition)
    for n, trace in enumerate(model.traces):
        states = enumerate_states(len(trace), model.n_skills, model.min_width,
                                  model.max_width)
        weights = state_log_weights(states, n, len(trace), likelihood._table, log_pi,
                                    log_p, model.delta_b)
        assert abs(exact_posterior(states, weights)["log_evidence"]
                   - log_z[n]) < 1e-10


def test_incremental_delta_matches_full_rebuild():
    model, state = tiny_model(), tiny_state()
    likelihood = CollapsedULikelihood(model=model)
    rng = np.random.default_rng(7)
    candidate = propose_row(np.array(state.u_by_skill[1]), 2, 0.5, rng)
    delta, candidate_log_z = likelihood.delta_for_candidate(state, 1, candidate)

    prime = state.copy()
    u = np.array(prime.u_by_skill, copy=True)
    u[1] = candidate
    prime.u_by_skill = u
    full_prime = likelihood.full_rebuild_log_z(prime)
    full_base = likelihood.full_rebuild_log_z(state)
    assert abs(delta - float((full_prime - full_base).sum())) <= 1e-10
    assert float(np.abs(candidate_log_z - full_prime).max()) <= 1e-10

    # block-table parity for the incremental column rebuild itself
    likelihood._table.refresh(u, state.beta, state.omega, state.lambda_rep,
                              state.lambda_back)
    fresh = FastBlockScoreTable(traces=model.traces, epsilon=model.epsilon,
                                n_skills=model.n_skills, min_width=model.min_width,
                                max_width=model.max_width, n_roles=model.n_roles)
    fresh.refresh(u, state.beta, state.omega, state.lambda_rep, state.lambda_back)
    for a, b in zip(likelihood._table.tables, fresh.tables):
        finite = np.isfinite(a)
        assert (finite == np.isfinite(b)).all()
        assert float(np.abs(a[finite] - b[finite]).max()) <= 1e-10


def test_q0_reset_every_candidate_block():
    """The dense tables must equal the adapter's per-block replay, which rebuilds every
    candidate from `q_0 = 0` independently — any recurrent-state leak would differ."""
    model, state = tiny_model(), tiny_state()
    likelihood = CollapsedULikelihood(model=model)
    likelihood.log_z_per_trace(state)
    scorer = model.scorer_for(state)
    for n, trace in enumerate(model.traces):
        adapter = build_log_block_scores(scorer, n, len(trace), model.n_skills,
                                         model.min_width, model.max_width)
        fast = likelihood._table.tables[n]
        finite = np.isfinite(adapter)
        assert (finite == np.isfinite(fast)).all()
        assert float(np.abs(adapter[finite] - fast[finite]).max()) <= 1e-10


# ------------------------------------------------------------------------- the move
def test_hastings_term_is_zero_for_the_registered_row_proposal():
    rng = np.random.default_rng(3)
    u = rng.normal(size=(3, 2))
    for _ in range(50):
        row = int(rng.integers(3))
        candidate = propose_row(u, row, 0.5, rng)
        step = candidate[row] - u[row]

        def density(s):
            return (-0.5 * s.size * math.log(2 * math.pi * 0.25)
                    - 0.5 * float(s @ s) / 0.25)

        assert density(step) == density(-step)


def test_same_h_candidate_has_exactly_zero_collapsed_delta():
    model, state = tiny_model(), tiny_state()
    likelihood = CollapsedULikelihood(model=model)
    candidate = np.array(state.u_by_skill[0]) * 1.7        # order-preserving rescale
    assert np.array_equal(precedence_from_u(candidate),
                          precedence_from_u(np.array(state.u_by_skill[0])))
    delta, _ = likelihood.delta_for_candidate(state, 0, candidate)
    assert delta == 0.0


def test_mh_acceptance_calculation_is_the_registered_ratio():
    """Replay the move's RNG stream and rebuild `log_alpha` from its parts."""
    model, state = tiny_model(), tiny_state()
    likelihood = CollapsedULikelihood(model=model)
    seed = 11
    new_state, record = collapsed_u_mh_step(state, model, likelihood,
                                            np.random.default_rng(seed), 0.5)
    replay = np.random.default_rng(seed)
    skill = int(replay.integers(2))
    row = int(replay.integers(3))
    candidate = propose_row(np.array(state.u_by_skill[skill]), row, 0.5, replay)
    assert record["skill"] == skill and record["row"] == row

    check = CollapsedULikelihood(model=model)
    delta, _ = check.delta_for_candidate(state, skill, candidate)
    d_prior = (log_structural_prior(candidate, state.rho)
               - log_structural_prior(np.array(state.u_by_skill[skill]), state.rho))
    assert record["log_alpha"] == pytest.approx(delta + d_prior, abs=1e-12)
    assert record["d_log_lik_collapsed"] == pytest.approx(delta, abs=1e-12)
    if record["accepted"]:
        assert np.array_equal(new_state.u_by_skill[skill], candidate)
    else:
        assert np.array_equal(new_state.u_by_skill, state.u_by_skill)


def test_move_counters_live_in_the_state():
    model, state = tiny_model(), tiny_state()
    likelihood = CollapsedULikelihood(model=model)
    new_state, _ = collapsed_u_mh_step(state, model, likelihood,
                                       np.random.default_rng(0), 0.5)
    assert new_state.proposed.get(MOVE_NAME, 0) == 1
    assert state.proposed.get(MOVE_NAME, 0) == 0          # input state untouched


# ------------------------------------------------------------------ cadence and cache
def test_cadence_schedules_exactly_as_configured():
    assert [i for i in range(20) if is_collapsed_sweep(i, 10)] == [9, 19]
    assert [i for i in range(9) if is_collapsed_sweep(i, 3)] == [2, 5, 8]
    assert not any(is_collapsed_sweep(i, 0) for i in range(50))

    model, state = tiny_model(), tiny_state()
    result = run_collapsed_u_chain(model=model, start=state, scales=SCALES,
                                   num_sweeps=9, burn_in=1, thin=1, seed=5,
                                   collapsed=CollapsedUConfig(every=3),
                                   store_labels=False)
    assert [r["sweep"] for r in result.collapsed_records] == [2, 5, 8]
    assert result.proposed[MOVE_NAME] == 3


def test_cache_invalidation_is_exact():
    model, state = tiny_model(), tiny_state()
    likelihood = CollapsedULikelihood(model=model)
    first = likelihood.log_z_per_trace(state)
    evaluations = likelihood.evaluations
    assert np.array_equal(likelihood.log_z_per_trace(state), first)
    assert likelihood.evaluations == evaluations          # a pure cache hit

    for mutate in (lambda s: setattr(s, "beta", s.beta + 0.01),
                   lambda s: setattr(s, "omega", s.omega + 0.01),
                   lambda s: setattr(s, "lambda_rep", s.lambda_rep + 0.01),
                   lambda s: setattr(s, "lambda_back", s.lambda_back + 0.01),
                   lambda s: setattr(s, "pi", np.array([0.5, 0.5])),
                   lambda s: setattr(s, "transition",
                                     np.array([[0.0, 1.0], [1.0, 0.0]]) * 1.0),
                   lambda s: setattr(s, "u_by_skill", s.u_by_skill + 0.05)):
        moved = state.copy()
        mutate(moved)
        before = likelihood.evaluations
        likelihood.log_z_per_trace(moved)
        if np.array_equal(moved.pi, state.pi) and np.array_equal(
                moved.transition, state.transition) and np.array_equal(
                moved.u_by_skill, state.u_by_skill) and (
                moved.beta, moved.omega, moved.lambda_rep, moved.lambda_back) == (
                state.beta, state.omega, state.lambda_rep, state.lambda_back):
            assert likelihood.evaluations == before      # nothing moved: cache hit
        else:
            assert likelihood.evaluations == before + 1  # something moved: recompute
        likelihood.log_z_per_trace(state)                # restore the cached state


# ------------------------------------------------------- ordinary sweeps are untouched
def test_every_zero_reproduces_run_stage7b_chain_bitwise():
    """With the collapsed move disabled the new runner IS Step 7B, draw for draw."""
    model = tiny_model()
    state = tiny_state()
    baseline = run_stage7b_chain(model=tiny_model(), start=tiny_state(), scales=SCALES,
                                 num_sweeps=40, burn_in=5, thin=2, seed=99,
                                 store_labels=True)
    ours = run_collapsed_u_chain(model=model, start=state, scales=SCALES,
                                 num_sweeps=40, burn_in=5, thin=2, seed=99,
                                 collapsed=CollapsedUConfig(every=0),
                                 store_labels=True)
    assert np.array_equal(baseline.u_draws, ours.u_draws)
    assert np.array_equal(baseline.log_target, ours.log_target)
    assert np.array_equal(baseline.segment_counts, ours.segment_counts)
    for name in baseline.scalars:
        assert np.array_equal(baseline.scalars[name], ours.scalars[name])
    assert baseline.boundary_keys == ours.boundary_keys
    assert ours.collapsed_records == []
