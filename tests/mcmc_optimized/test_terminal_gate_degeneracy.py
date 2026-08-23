"""The terminal gate's degeneracy branches, pinned.

The confirmatory gate turns on classifying a summary before diagnosing it. Each branch is
tested here on constructed series, because on real draws a branch may simply not occur --
and the branch that did not occur is exactly the one a later change would break silently.

The fourth case is the one this experiment actually hit: a Bernoulli probe whose 5% and 95%
quantile indicators are both constant, so tail ESS is 0/0. That is an UNDEFINED statistic,
not a small one, and the gate must not silently convert it into a pass.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "_gate", Path(__file__).parents[2] / "scripts" / "confirmatory_terminal_gate.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

N = 4000


def _chains(rng, mean=0.0, sd=1.0, m=4):
    return np.array([rng.normal(mean, sd, N) for _ in range(m)])


# ------------------------------------------------------- (a) constant and equal
def test_constant_and_equal_is_degenerate_with_no_ess_floor():
    series = np.full((4, N), 7.0)
    d = gate.diagnose(series)
    assert d["branch"] == "constant_and_equal"
    assert d["rhat"] is None and d["bulk_ess"] is None and d["tail_ess"] is None
    ok, why = gate.gate_for("total_relations", d)
    assert ok, why
    assert "degenerate" in why


def test_constant_and_equal_passes_even_for_log_target_which_has_a_higher_floor():
    d = gate.diagnose(np.full((4, N), -123.0))
    ok, _ = gate.gate_for("log_target", d)
    assert ok, "a point mass has no variance to estimate; no floor can apply"


# ------------------------------------ (b) constant within chains, unequal across
def test_constant_within_unequal_across_is_an_automatic_fail():
    series = np.stack([np.full(N, v) for v in (1.0, 2.0, 2.0, 3.0)])
    d = gate.diagnose(series)
    assert d["branch"] == "constant_within_unequal_across"
    assert d["rhat"] == float("inf")
    assert d["bulk_ess"] == 0.0
    ok, why = gate.gate_for("canonical_library", d)
    assert not ok
    assert "automatic FAIL" in why


def test_two_chains_agreeing_does_not_rescue_constant_but_unequal():
    """Three chains at one value and one elsewhere is still disagreement."""
    series = np.stack([np.full(N, 5.0), np.full(N, 5.0), np.full(N, 5.0),
                       np.full(N, 6.0)])
    ok, _ = gate.gate_for("canonical_library", gate.diagnose(series))
    assert not ok


# ------------------------------------------------------------ (c) non-degenerate
def test_well_mixed_continuous_summary_passes():
    d = gate.diagnose(_chains(np.random.default_rng(1)))
    assert d["branch"] == "non_degenerate"
    assert d["rhat"] <= 1.01
    assert d["bulk_ess"] >= 400 and d["tail_ess"] >= 400
    assert gate.gate_for("pi_entropy", d)[0]


def test_offset_chains_fail_on_rhat():
    rng = np.random.default_rng(2)
    series = np.stack([_chains(rng, mean=o, m=1)[0] for o in (0.0, 0.0, 0.0, 6.0)])
    d = gate.diagnose(series)
    assert d["branch"] == "non_degenerate"
    ok, why = gate.gate_for("log_target", d)
    assert not ok and "rhat" in why


def test_log_target_floor_is_stricter_than_the_others():
    """A summary with ESS between 400 and 1000 passes as a probe and fails as log_target."""
    rng = np.random.default_rng(3)
    short = np.array([rng.normal(size=150) for _ in range(4)])   # ~600 draws total
    d = gate.diagnose(short)
    if not (400 <= d["bulk_ess"] < 1000):
        pytest.skip(f"constructed series landed at bulk ESS {d['bulk_ess']:.0f}")
    assert gate.gate_for("boundary_probes", d)[0]
    assert not gate.gate_for("log_target", d)[0]


# ------------------- (d) near-degenerate Bernoulli: undefined tail ESS
def _bernoulli(rng, p, m=4):
    return np.array([(rng.random(N) < p).astype(float) for _ in range(m)])


def test_near_degenerate_bernoulli_yields_undefined_tail_ess_and_does_not_pass():
    """p ~ 0.998: q05 = q95 = 1, both tail indicators constant, tail ESS is 0/0.

    This is the case FULL-MARG hit on 11 probes. The gate must report it as a failure
    rather than silently treating an undefined statistic as satisfied -- amending that
    rule is a post-hoc decision, not something the implementation may take on itself.
    """
    series = _bernoulli(np.random.default_rng(4), 0.998)
    d = gate.diagnose(series)
    assert d["branch"] == "non_degenerate", "not every chain is constant"
    assert np.isnan(d["tail_ess"]), "both quantile indicators are constant"
    assert d["rhat"] <= 1.01, "the probe itself mixes perfectly"
    assert d["bulk_ess"] > 400, "and has ample bulk ESS"
    ok, why = gate.gate_for("coskill_probes", d)
    assert not ok, "an undefined tail ESS must not silently pass"
    assert "tail ESS" in why


def test_moderately_extreme_bernoulli_still_has_a_defined_tail_ess():
    """p ~ 0.5: q05 = 0 varies, so the informative side is usable."""
    d = gate.diagnose(_bernoulli(np.random.default_rng(5), 0.5))
    assert d["branch"] == "non_degenerate"
    assert not np.isnan(d["tail_ess"])
    assert gate.gate_for("boundary_probes", d)[0]


def test_nan_tail_ess_is_not_treated_as_zero_or_as_infinity():
    d = gate.diagnose(_bernoulli(np.random.default_rng(6), 0.999))
    assert np.isnan(d["tail_ess"])
    assert not (d["tail_ess"] >= 400)
    assert not (d["tail_ess"] < 400)     # nan compares false both ways, by design


# ------------------------------------------------------- canonical library ids
def test_library_ids_are_invariant_to_skill_relabelling():
    rng = np.random.default_rng(7)
    base = rng.random((10, 3, 20)) < 0.3
    permuted = base[:, [2, 0, 1], :]
    a, _ = gate.library_ids(base.reshape(10, 60))
    b, _ = gate.library_ids(permuted.reshape(10, 60))
    assert np.array_equal(a, b), "the library must be invariant to skill relabelling"


def test_library_ids_separate_genuinely_different_structures():
    draws = np.zeros((2, 60), dtype=bool)
    draws[0, 0] = True
    draws[1, 1] = True
    ids, _ = gate.library_ids(draws)
    assert ids[0] != ids[1]
