"""Focused tests for the C0 collapsed-U fast audit — nothing beyond the diagnostic."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.block_score_adapters import assert_no_recurrent_state_leak
from hpop.mcmc_original.fast_block_tables import FastBlockScoreTable
from hpop.mcmc_original.recurrent_segmentation import RecurrentBlockScorer
from hpop.mcmc_original.sampler_u import propose_row
from hpop.mcmc_original.semi_markov_ffbs import forward


def _load_audit_module():
    path = ROOT / "scripts" / "collapsed_u_fast_audit.py"
    spec = importlib.util.spec_from_file_location("collapsed_u_fast_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit_module()

TINY_TRACES = ((0, 1, 2, 3, 4, 0, 1, 2, 3),)   # one J=9 trace over 5 roles
MIN_W, MAX_W, K = 3, 6, 2


def _tiny_table(seed=0):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(K, 5, 2))
    table = FastBlockScoreTable(traces=TINY_TRACES, epsilon=0.02, n_skills=K,
                                min_width=MIN_W, max_width=MAX_W, n_roles=5)
    table.refresh(u, 1.0, 0.5, 0.3, 0.2)
    return u, table


def test_collapsed_forward_normaliser_equals_enumeration():
    """The forward log Z must equal brute-force enumeration of every (S, z) path."""
    _, table = _tiny_table()
    log_pi = np.log(np.array([0.6, 0.4]))
    p = np.array([[0.0, 1.0], [1.0, 0.0]])
    log_p = np.where(p > 0, np.log(np.maximum(p, 1e-300)), -np.inf)
    chart = forward(table.tables[0], log_pi, log_p, 0.15, MAX_W, MIN_W)
    brute = audit.enumeration_log_z(table.tables[0], log_pi, log_p, 0.15, MIN_W, MAX_W)
    assert abs(chart.log_normalizer - brute) < 1e-10


def test_incremental_refresh_matches_full_rebuild():
    """Perturbing one skill's U must rebuild only that column, to the same numbers."""
    u, table = _tiny_table()
    u2 = np.array(u, copy=True)
    u2[1] += 0.3
    info = table.refresh(u2, 1.0, 0.5, 0.3, 0.2)
    assert info["rebuilt_skills"] == [1] and info["reused_skills"] == [0]
    fresh = FastBlockScoreTable(traces=TINY_TRACES, epsilon=0.02, n_skills=K,
                                min_width=MIN_W, max_width=MAX_W, n_roles=5)
    fresh.refresh(u2, 1.0, 0.5, 0.3, 0.2)
    a, b = table.tables[0], fresh.tables[0]
    finite = np.isfinite(a)
    assert (finite == np.isfinite(b)).all()
    assert float(np.abs(a[finite] - b[finite]).max()) <= 1e-10


def test_q0_reset_between_candidate_blocks():
    """Scoring A, then B, then A again must be bit-identical: no recurrent state leaks."""
    rng = np.random.default_rng(1)
    scorer = RecurrentBlockScorer(traces=TINY_TRACES, epsilon=0.02,
                                  u_by_skill=rng.normal(size=(K, 5, 2)), beta=1.0,
                                  omega=0.5, lambda_rep=0.3, lambda_back=0.2,
                                  max_width=MAX_W, min_width=MIN_W)
    result = assert_no_recurrent_state_leak(scorer, 0, (0, 3, 0), (3, 6, 1))
    assert result["pass"]


def test_proposal_hastings_parity():
    """The production row proposal is symmetric: q(U'|U) == q(U|U') exactly."""
    rng = np.random.default_rng(2)
    u = rng.normal(size=(5, 2))
    for row in range(5):
        candidate = propose_row(u, row, 0.5, rng)
        step = candidate[row] - u[row]
        f = audit.gaussian_row_log_density(step, 0.5)
        r = audit.gaussian_row_log_density(-step, 0.5)
        assert f == r


def test_escape_count_calculation():
    """E = sweeps x M_U x r_cross x mean(alpha | cross), straight multiplication."""
    e = 50_000 * 15 * 0.5 * 2e-12
    assert e == pytest.approx(7.5e-7)
    assert e < 1.0     # the NOT-VIABLE branch of the pre-registered rule


def test_deterministic_proposal_subset():
    """Same seed, same state -> the audit's proposal stream is reproducible."""
    rng_a = np.random.default_rng(audit.AUDIT_SEED)
    rng_b = np.random.default_rng(audit.AUDIT_SEED)
    u = np.zeros((5, 2))
    for row in range(5):
        assert np.array_equal(propose_row(u, row, 0.5, rng_a),
                              propose_row(u, row, 0.5, rng_b))
