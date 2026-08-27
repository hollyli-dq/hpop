"""The partially-collapsed ordering: exact stationarity, and why the refresh must be
immediate.

The load-bearing test discretises `U` to a small grid so the joint state space
`(U_grid, S, z)` is finite and the FULL transition matrix of the scheduled block

    collapsed U MH  ->  exact (S, z) conditional draw

can be written down and multiplied against the exact joint target. Stationarity to
machine precision is the partially-collapsed Gibbs argument made numerical, using the
production `CollapsedULikelihood` for every acceptance ratio.

The negative control builds the DELIBERATELY WRONG ordering — collapsed U, then a
conditional update that consumes the stale `(S, z)`, then the refresh — and shows its
stationary distribution is NOT the target. That ordering must never ship; this test
exists to lock in why.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_original.collapsed_u_kernel import (
    CollapsedUConfig, collapsed_ffbs_sweep_once, collapsed_u_mh_step,
)
from hpop.mcmc_original.collapsed_u_likelihood import CollapsedULikelihood
from hpop.mcmc_original.fast_segmentation_kernel import segmentation_of
from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import Stage7BSampler
from hpop.mcmc_original.sampler_u import log_u_prior
from hpop.mcmc_original.stage6e_exact import (
    enumerate_states, exact_posterior, state_log_weights,
)
from hpop.mcmc_original.stage6e_state import Stage6EModel, Stage6EState
from hpop.mcmc_original.transitions import log_transition_matrix

TRACE = (0, 1, 2, 0, 1, 2)                     # J = 6: paths [6] and [3, 3]
RHO = 0.2


def grid_model() -> Stage6EModel:
    return Stage6EModel(traces=(TRACE,), epsilon=0.02, delta_b=0.15, n_skills=2,
                        n_roles=3, min_width=3, max_width=6, infer_pi_P=False)


def state_at(u: np.ndarray, key=((6, 0),)) -> Stage6EState:
    return Stage6EState(segmentations=(segmentation_of(key),),
                        u_by_skill=np.array(u, dtype=float), rho=RHO, beta=1.0,
                        omega=0.5, lambda_rep=0.3, lambda_back=0.2,
                        pi=np.array([0.6, 0.4]),
                        transition=np.array([[0.0, 1.0], [1.0, 0.0]]))


def build_grid(model: Stage6EModel, n_grid: int = 4):
    """The finite joint problem: grid U's, enumerated paths, exact per-grid weights."""
    rng = np.random.default_rng(42)
    grid = [rng.normal(size=(2, 3, 2)) for _ in range(n_grid)]
    paths = enumerate_states(len(TRACE), model.n_skills, model.min_width,
                             model.max_width)
    log_pi = np.log(np.array([0.6, 0.4]))
    log_p = log_transition_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))

    likelihood = CollapsedULikelihood(model=model)
    prior = np.empty(n_grid)                    # log p(U_g | rho)
    log_w = np.empty((n_grid, len(paths)))      # log w(path | U_g)
    conditional = np.empty((n_grid, len(paths)))
    for g, u in enumerate(grid):
        state = state_at(u)
        prior[g] = sum(log_u_prior(u[k], RHO) for k in range(2))
        likelihood.log_z_per_trace(state)       # leaves the table at U_g
        weights = state_log_weights(paths, 0, len(TRACE), likelihood._table, log_pi,
                                    log_p, model.delta_b)
        log_w[g] = weights
        conditional[g] = exact_posterior(paths, weights)["probability"]

    # the exact joint target on the finite space, pi(g, path)
    joint = np.exp(prior[:, None] + log_w)
    joint /= joint.sum()

    # ell_coll(U_g) from the production scorer, for the MH ratios
    ell = np.array([
        CollapsedULikelihood(model=model).log_likelihood(state_at(u)) for u in grid])
    return grid, paths, prior, joint, conditional, ell


def collapsed_mh_matrix(prior: np.ndarray, ell: np.ndarray) -> np.ndarray:
    """Uniform grid proposal, accepted with the collapsed ratio — invariant for the
    marginal exp(prior + ell) if the ratio is the registered one."""
    n = len(prior)
    m = np.zeros((n, n))
    for g in range(n):
        for h in range(n):
            if h == g:
                continue
            log_alpha = (ell[h] - ell[g]) + (prior[h] - prior[g])
            m[g, h] = (1.0 / (n - 1)) * min(1.0, float(np.exp(min(0.0, log_alpha))))
        m[g, g] = 1.0 - m[g].sum()
    return m


def test_collapsed_then_exact_refresh_is_stationary_for_the_joint():
    model = grid_model()
    grid, paths, prior, joint, conditional, ell = build_grid(model)
    n_grid, n_paths = len(grid), len(paths)
    m = collapsed_mh_matrix(prior, ell)

    # K[(g,p) -> (g',p')] = M[g,g'] * p(p' | U_g'): the scheduled block, exactly
    kernel = np.zeros((n_grid * n_paths, n_grid * n_paths))
    for g in range(n_grid):
        for p in range(n_paths):
            for h in range(n_grid):
                kernel[g * n_paths + p, h * n_paths:(h + 1) * n_paths] = (
                    m[g, h] * conditional[h])
    flat = joint.reshape(-1)
    assert np.abs(kernel.sum(axis=1) - 1.0).max() < 1e-12
    assert np.abs(flat @ kernel - flat).max() < 1e-12


def test_stale_path_ordering_is_not_stationary():
    """collapsed U -> conditional move consuming the STALE path -> refresh: broken."""
    model = grid_model()
    grid, paths, prior, joint, conditional, ell = build_grid(model)
    n_grid, n_paths = len(grid), len(paths)
    m_coll = collapsed_mh_matrix(prior, ell)

    _, _, _, _, _, _ = grid, paths, prior, joint, conditional, ell
    log_w = np.log(conditional) + 0.0           # conditional shape reference only
    # conditional MH on the grid given a FIXED (stale) path p, from the exact weights
    grid_state_weights = np.empty((n_grid, n_paths))
    likelihood = CollapsedULikelihood(model=model)
    log_pi = np.log(np.array([0.6, 0.4]))
    log_p = log_transition_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    for g, u in enumerate(grid):
        likelihood.log_z_per_trace(state_at(u))
        grid_state_weights[g] = state_log_weights(paths, 0, len(TRACE),
                                                  likelihood._table, log_pi, log_p,
                                                  model.delta_b)

    def conditional_mh_matrix(path_index: int) -> np.ndarray:
        m = np.zeros((n_grid, n_grid))
        for g in range(n_grid):
            for h in range(n_grid):
                if h == g:
                    continue
                log_alpha = ((grid_state_weights[h, path_index]
                              - grid_state_weights[g, path_index])
                             + (prior[h] - prior[g]))
                m[g, h] = (1.0 / (n_grid - 1)) * min(1.0, float(
                    np.exp(min(0.0, log_alpha))))
            m[g, g] = 1.0 - m[g].sum()
        return m

    kernel = np.zeros((n_grid * n_paths, n_grid * n_paths))
    for g in range(n_grid):
        for p in range(n_paths):
            m_stale = conditional_mh_matrix(p)          # consumes the STALE path p
            for h1 in range(n_grid):
                for h2 in range(n_grid):
                    kernel[g * n_paths + p, h2 * n_paths:(h2 + 1) * n_paths] += (
                        m_coll[g, h1] * m_stale[h1, h2] * conditional[h2])
    flat = joint.reshape(-1)
    assert np.abs(kernel.sum(axis=1) - 1.0).max() < 1e-12
    deviation = np.abs(flat @ kernel - flat).max()
    assert deviation > 1e-4, (
        "the stale-path ordering unexpectedly looks stationary on this problem; "
        "the negative control needs a more asymmetric configuration")
    del log_w


def test_collapsed_move_never_reads_the_stored_segmentation():
    """Byte-identical decision under a scrambled stored (S, z): the acceptance ratio
    contains no conditional term and consumes no segmentation-dependent randomness."""
    model = grid_model()
    rng_u = np.random.default_rng(1)
    u = rng_u.normal(size=(2, 3, 2))
    a = state_at(u, key=((6, 0),))
    b = state_at(u, key=((3, 0), (6, 1)))       # a completely different stored (S, z)
    _, record_a = collapsed_u_mh_step(a, model, CollapsedULikelihood(model=model),
                                      np.random.default_rng(21), 0.5)
    _, record_b = collapsed_u_mh_step(b, model, CollapsedULikelihood(model=model),
                                      np.random.default_rng(21), 0.5)
    drop_timing = lambda r: {k: v for k, v in r.items() if k != "seconds"}  # noqa: E731
    assert drop_timing(record_a) == drop_timing(record_b)


def test_ffbs_refresh_runs_immediately_after_the_move_at_the_post_move_u():
    """The first table refresh of the sweep — the one FFBS draws from — must see the
    post-move U, before any parameter update has run."""
    model = grid_model()
    state = state_at(np.random.default_rng(2).normal(size=(2, 3, 2)))
    scales = {"U": 0.5, "rho": 0.5, "beta": 0.05, "omega": 0.18,
              "lambda_rep": 0.04, "lambda_back": 0.09}
    likelihood = CollapsedULikelihood(model=model)

    for seed in range(6):                        # cover accepted and rejected moves
        sampler = Stage7BSampler(model=model, scales=scales, table_source="batched")
        seen: list = []
        original = sampler.tables.refresh

        def spy(refresh_state, _original=original, _seen=seen):
            _seen.append(np.array(refresh_state.u_by_skill, copy=True))
            return _original(refresh_state)

        sampler.tables.refresh = spy
        rng = np.random.default_rng(seed)
        state9 = state.copy()
        state9.iteration = 9                     # scheduled under every = 10
        replay = np.random.default_rng(seed)
        moved, record = collapsed_u_mh_step(state9.copy(), model,
                                            CollapsedULikelihood(model=model), replay,
                                            0.5)
        result, rec = collapsed_ffbs_sweep_once(state9, sampler, likelihood,
                                                CollapsedUConfig(every=10), rng)
        assert rec is not None and rec["accepted"] == record["accepted"]
        assert len(seen) == 1                    # one refresh: the FFBS draw's
        assert np.array_equal(seen[0], moved.u_by_skill)


def test_unscheduled_sweep_never_calls_the_collapsed_move():
    model = grid_model()
    state = state_at(np.random.default_rng(3).normal(size=(2, 3, 2)))
    scales = {"U": 0.5, "rho": 0.5, "beta": 0.05, "omega": 0.18,
              "lambda_rep": 0.04, "lambda_back": 0.09}
    sampler = Stage7BSampler(model=model, scales=scales, table_source="batched")
    likelihood = CollapsedULikelihood(model=model)
    state.iteration = 3                          # not scheduled under every = 10
    _, record = collapsed_ffbs_sweep_once(state, sampler, likelihood,
                                          CollapsedUConfig(every=10),
                                          np.random.default_rng(0))
    assert record is None
    assert likelihood.evaluations == 0           # the collapsed scorer never ran
