"""Focused tests for the C1 expanded collapsed-U audit helpers — nothing beyond them."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


expanded = _load("collapsed_u_expanded_audit",
                 ROOT / "scripts" / "collapsed_u_expanded_audit.py")


def _random_h_list(seed=0, K=3, m=5):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(K):
        h = np.triu(rng.random((m, m)) < 0.4, k=1)   # a DAG-ish strict relation
        out.append(h)
    return out


def test_canonical_hash_invariant_under_skill_relabelling():
    h_list = _random_h_list()
    base = expanded.canonical_h_hash(h_list)
    for perm in ([1, 0, 2], [2, 1, 0], [1, 2, 0]):
        assert expanded.canonical_h_hash([h_list[i] for i in perm]) == base


def test_exact_hash_is_order_sensitive():
    h_list = _random_h_list(seed=1)
    assert (expanded.state_h_hash(h_list)
            != expanded.state_h_hash([h_list[1], h_list[0], h_list[2]]))


def test_canonical_hash_distinguishes_different_orders():
    a = _random_h_list(seed=2)
    b = _random_h_list(seed=3)
    assert expanded.canonical_h_hash(a) != expanded.canonical_h_hash(b)


def test_wilson_interval():
    lo, hi = expanded.wilson(0, 300)
    assert lo == 0.0 and 0.0 < hi < 0.02          # zero successes: upper ~1.27%
    lo, hi = expanded.wilson(150, 300)
    assert lo < 0.5 < hi and hi - lo < 0.12
    lo, hi = expanded.wilson(300, 300)
    assert hi == pytest.approx(1.0) and lo > 0.98


def test_escape_rate_and_sweeps_per_escape_are_reciprocal():
    r_cross, mean_alpha = 0.5, 0.03
    rate = expanded.M_U_PER_SWEEP * r_cross * mean_alpha
    escapes_50k = expanded.SWEEPS * rate
    sweeps_per_escape = 1.0 / rate
    assert escapes_50k * sweeps_per_escape == pytest.approx(expanded.SWEEPS)
