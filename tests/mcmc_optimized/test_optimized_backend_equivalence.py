"""`hpop.mcmc_optimized` must reproduce `hpop.mcmc_original`, which is the oracle.

The optimized backend exists only because the reference is sealed. That makes two things
testable here, and both are:

  1. the reference really is untouched -- asserted against the same SHA the Step 7A gate
     uses, from this side, so a future edit fails here too rather than only there;
  2. every optimized routine returns the reference's numbers.

Tolerance is 1e-10, far tighter than the measured ~4e-14 at |alpha| ~ 1e2. A regression to
1e-6 would still "look small" without a gate, so the gate is deliberately strict.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from hpop.mcmc_original import semi_markov_ffbs
from hpop.mcmc_original.semi_markov_ffbs import backward_sample
from hpop.mcmc_original.semi_markov_ffbs import forward as reference_forward
from hpop.mcmc_optimized import (COUNTERS, FLAGS, BatchedCollapsedULikelihood,
                                 HashCachedFFBSBlockTables, forward_batched_group,
                                 forward_dispatch, forward_factorised,
                                 forward_with_inline_reduction, inline_logsumexp)

TOLERANCE = 1e-10
BOUNDARY = 0.15
FROZEN_ENGINE_SHA256 = "8150bb8235eb159d5e2f08ada7c698c383f4c7fd31f3b882e218b207dd135486"


@pytest.fixture(autouse=True)
def _reset_flags():
    FLAGS.reset()
    COUNTERS.reset()
    yield
    FLAGS.reset()
    COUNTERS.reset()


def _problem(J, K, seed, min_width=3, max_width=12):
    rng = np.random.default_rng(seed)
    scores = rng.normal(-2.0, 1.0, size=(J, J + 1, K))
    for a in range(J):
        for b in range(J + 1):
            if not (min_width <= b - a <= max_width):
                scores[a, b, :] = -np.inf
    return (scores, np.log(rng.dirichlet(np.ones(K))),
            np.log(rng.dirichlet(np.ones(K), size=K)), min_width, max_width)


def _same_chart(want, have):
    finite = np.isfinite(want.alpha)
    assert np.array_equal(finite, np.isfinite(have.alpha)), "the -inf support differs"
    assert np.allclose(want.alpha[finite], have.alpha[finite], atol=TOLERANCE, rtol=0)
    assert have.log_normalizer == pytest.approx(want.log_normalizer, abs=TOLERANCE)


# ------------------------------------------------------------------ the seal, restated
def test_the_reference_engine_is_still_byte_identical():
    """If this fails, the optimized backend has stopped being a separate backend."""
    digest = hashlib.sha256(Path(semi_markov_ffbs.__file__).read_bytes()).hexdigest()
    assert digest == FROZEN_ENGINE_SHA256


# ------------------------------------------------------------------------- the flags
def test_flags_default_to_all_optimisations_on():
    assert FLAGS.snapshot() == {"inline_logsumexp": True, "emission_hash_cache": True,
                                "factorised_forward": True, "batched_forward": True}


def test_all_off_falls_back_to_the_reference_implementation():
    scores, log_pi, log_p, lo, hi = _problem(20, 3, 40)
    FLAGS.all_off()
    assert FLAGS.label() == "reference_algorithm"
    got = forward_dispatch(scores, log_pi, log_p, BOUNDARY, hi, lo)
    assert COUNTERS.forward_reference_calls == 1
    _same_chart(reference_forward(scores, log_pi, log_p, BOUNDARY, hi, lo), got)


# -------------------------------------------------------------------------------- O1
@pytest.mark.parametrize("values", [
    np.array([0.0]),
    np.array([-1.0, -2.0, -3.0]),
    np.array([-np.inf, -1.0, -np.inf]),
    np.array([-np.inf, -np.inf]),
    np.array([1e3, 1e3 - 1.0, 1e3 - 2.0]),
    np.array([-1e300, -1e300 - 1.0]),
])
def test_inline_logsumexp_matches_scipy(values):
    got = inline_logsumexp(values)
    if not np.isfinite(values).any():
        assert got == -np.inf
        return
    assert got == pytest.approx(float(logsumexp(values)), abs=1e-12, rel=1e-13)


@pytest.mark.parametrize("J,K,seed", [(14, 2, 1), (26, 3, 2), (32, 4, 3)])
def test_inline_forward_matches_reference(J, K, seed):
    scores, log_pi, log_p, lo, hi = _problem(J, K, seed)
    want = reference_forward(scores, log_pi, log_p, BOUNDARY, hi, lo)
    have = forward_with_inline_reduction(scores, log_pi, log_p, BOUNDARY, hi, lo)
    assert COUNTERS.forward_inline_calls == 1
    _same_chart(want, have)


# -------------------------------------------------------------------------------- O3
@pytest.mark.parametrize("J,K,seed", [(14, 2, 4), (26, 3, 5), (32, 4, 6)])
def test_factorised_forward_matches_reference(J, K, seed):
    scores, log_pi, log_p, lo, hi = _problem(J, K, seed)
    want = reference_forward(scores, log_pi, log_p, BOUNDARY, hi, lo)
    have = forward_factorised(scores, log_pi, log_p, BOUNDARY, hi, lo)
    assert COUNTERS.forward_factorised_calls == 1
    _same_chart(want, have)


# -------------------------------------------------------------------------------- O4
@pytest.mark.parametrize("J,K,seed", [(14, 2, 7), (26, 3, 8)])
def test_batched_forward_matches_reference(J, K, seed):
    group = [_problem(J, K, seed + i)[0] for i in range(4)]
    _, log_pi, log_p, lo, hi = _problem(J, K, seed)
    want = [reference_forward(t, log_pi, log_p, BOUNDARY, hi, lo) for t in group]
    have = forward_batched_group(group, log_pi, log_p, BOUNDARY, hi, lo)
    assert COUNTERS.forward_batched_groups == 1
    assert COUNTERS.forward_batched_traces == len(group)
    for a, b in zip(want, have):
        _same_chart(a, b)


def test_batched_group_rejects_mixed_lengths():
    """Silently padding traces of different length would be a correctness bug."""
    a, log_pi, log_p, lo, hi = _problem(14, 3, 9)
    b = _problem(20, 3, 10)[0]
    with pytest.raises(ValueError, match="one shape"):
        forward_batched_group([a, b], log_pi, log_p, BOUNDARY, hi, lo)


def test_batched_charts_drive_identical_backward_draws():
    """Same chart, same rng, same segmentation: this is what protects the draw order."""
    group = [_problem(26, 3, 20 + i)[0] for i in range(3)]
    _, log_pi, log_p, lo, hi = _problem(26, 3, 20)
    want = [backward_sample(reference_forward(t, log_pi, log_p, BOUNDARY, hi, lo),
                            np.random.default_rng(99)) for t in group]
    have = [backward_sample(c, np.random.default_rng(99))
            for c in forward_batched_group(group, log_pi, log_p, BOUNDARY, hi, lo)]
    assert want == have


# -------------------------------------------------------------------------------- O2
def _model_and_state():
    path = (Path(__file__).parents[1] / "mcmc_original"
            / "test_stage7b2_optimisation.py")
    spec = importlib.util.spec_from_file_location("_s7b2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._model_and_state()


def test_emission_hash_cache_is_bitwise_and_reports_honestly():
    model, state = _model_and_state()
    tables = HashCachedFFBSBlockTables(model=model, source="fast")
    tables.refresh(state)
    assert COUNTERS.emission_cache_hits == 0, "nothing to hit on the first build"
    first = [np.array(t, copy=True) for t in tables.tables_for(state)]

    tables.refresh(state)
    assert COUNTERS.emission_cache_hits == 1
    for want, have in zip(first, tables.tables_for(state)):
        assert np.array_equal(want, have), "a cache hit must return the same bits"
    assert tables.last_refresh["rebuilt_skills"] == []
    assert tables.last_refresh["reused_skills"] == list(range(model.n_skills))


def test_emission_cache_misses_when_H_moves():
    from hpop.mcmc_original.latent_poset import precedence_from_u
    model, state = _model_and_state()
    tables = HashCachedFFBSBlockTables(model=model, source="fast")
    tables.refresh(state)
    hits = COUNTERS.emission_cache_hits

    moved = state.copy()
    u = np.array(state.u_by_skill, dtype=float, copy=True)
    u[0] = u[0][::-1]
    if np.array_equal(precedence_from_u(u[0]), precedence_from_u(state.u_by_skill[0])):
        pytest.skip("this fixture's U_0 is symmetric under row reversal")
    moved.u_by_skill = u
    tables.refresh(moved)
    assert COUNTERS.emission_cache_hits == hits, "H moved; this must not be a hit"
    assert tables.last_refresh["rebuilt_skills"] != []


def test_cache_disabled_always_rebuilds():
    model, state = _model_and_state()
    FLAGS.apply(emission_hash_cache=False)
    tables = HashCachedFFBSBlockTables(model=model, source="fast")
    tables.refresh(state)
    tables.refresh(state)
    assert COUNTERS.emission_cache_hits == 0
    assert COUNTERS.emission_rebuilds == 2


def test_optimized_collapsed_likelihood_matches_reference():
    from hpop.mcmc_original.collapsed_u_likelihood import CollapsedULikelihood
    model, state = _model_and_state()
    want = CollapsedULikelihood(model=model).log_z_per_trace(state)
    have = BatchedCollapsedULikelihood(model=model).log_z_per_trace(state)
    assert np.allclose(want, have, atol=TOLERANCE, rtol=0)
