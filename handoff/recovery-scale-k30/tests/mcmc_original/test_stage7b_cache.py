"""Step 7B — the candidate-table lifecycle.

The table of every candidate block score is what makes FFBS possible and what makes it
expensive. Two mistakes would be easy and both would be silent: reading a table built at
parameters the chain has since left, and rebuilding the whole table for every scalar
proposal. The first corrupts the target; the second multiplies the cost of a sweep by the
number of proposals in it. These tests pin both.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_original.block_score_adapters import build_log_block_scores
from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import (
    FFBSBlockTables, Stage7BSampler, assert_sources_agree, ffbs_sweep_once,
    run_stage7b_chain,
)
from hpop.mcmc_original.stage6d_frozen import REGISTERED_SCALES, SCALAR_ORDER
from hpop.mcmc_original.stage6e_frozen import MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH
from tests.mcmc_original.test_stage7b_joint_target import (
    K, TRACES, make_model, make_state,
)


# --------------------------------------------------------------------- staleness
def test_a_stale_table_refuses_to_be_read():
    model = make_model()
    state = make_state(np.random.default_rng(20))
    tables = FFBSBlockTables(model=model, source="batched")
    with pytest.raises(AssertionError, match="stale"):
        tables.tables_for(state)                       # never built
    tables.refresh(state)
    tables.tables_for(state)                           # fine
    tables.mark_stale()
    with pytest.raises(AssertionError, match="stale"):
        tables.tables_for(state)


def test_a_table_built_at_other_parameters_refuses_to_be_read():
    model = make_model()
    state = make_state(np.random.default_rng(21))
    tables = FFBSBlockTables(model=model, source="batched")
    tables.refresh(state)
    for name in ("beta", "omega", "lambda_rep", "lambda_back"):
        moved = state.copy()
        setattr(moved, name, getattr(state, name) + 0.25)
        with pytest.raises(AssertionError, match="different parameters"):
            tables.tables_for(moved)
    moved_u = state.copy()
    moved_u.u_by_skill = state.u_by_skill + 0.1
    with pytest.raises(AssertionError, match="different parameters"):
        tables.tables_for(moved_u)


def test_rho_and_pi_P_do_not_invalidate_the_table():
    """`rho` acts only through `p(U | rho)` and `pi`/`P` only through the path prior, so
    neither enters a block score. The fingerprint must not depend on them."""
    model = make_model()
    state = make_state(np.random.default_rng(22))
    tables = FFBSBlockTables(model=model, source="batched")
    tables.refresh(state)
    moved = state.copy()
    moved.rho = min(0.95, state.rho + 0.2)
    moved.pi = np.array([0.2, 0.5, 0.3])
    tables.tables_for(moved)                           # must not raise


def test_the_table_is_built_once_per_segmentation_sweep():
    model = make_model()
    state = make_state(np.random.default_rng(23))
    sampler = Stage7BSampler(model=model, scales=REGISTERED_SCALES)
    rng = np.random.default_rng(24)
    for i in range(1, 16):
        state = ffbs_sweep_once(state, sampler, rng)
        assert sampler.tables.builds == i
        assert sampler.tables.stale                    # invalidated before the parameters moved


def test_no_rebuild_for_any_scalar_proposal():
    """A sweep proposes `U` rows, `rho` and four scalars; none may touch the table."""
    model = make_model()
    start = make_state(np.random.default_rng(25))
    sweeps = 30
    result = run_stage7b_chain(model=model, start=start, scales=REGISTERED_SCALES,
                               num_sweeps=sweeps, burn_in=5, thin=5, seed=26)
    assert result.table_builds == sweeps
    proposals = sum(result.proposed[name] for name in (*SCALAR_ORDER, "U", "rho"))
    assert proposals > sweeps * (len(SCALAR_ORDER) + 1)
    # one build per sweep, not one per proposal
    assert result.table_builds < proposals / len(SCALAR_ORDER)


def test_a_disabled_segmentation_draw_builds_nothing():
    model = make_model()
    start = make_state(np.random.default_rng(27))
    result = run_stage7b_chain(model=model, start=start, scales=REGISTERED_SCALES,
                               num_sweeps=20, burn_in=5, thin=5, seed=28,
                               draw_segmentation=False)
    assert result.table_builds == 0


# ------------------------------------------------------------------ eager vs batched
def test_eager_and_batched_tables_are_the_same_numbers():
    model = make_model()
    rng = np.random.default_rng(29)
    for _ in range(5):
        audit = assert_sources_agree(model, make_state(rng))
        assert audit["pass"]
        assert audit["same_support"]
        assert audit["max_absolute_difference"] < 1e-9


def test_the_batched_table_matches_the_step7a_adapter_entry_for_entry():
    model = make_model()
    state = make_state(np.random.default_rng(30))
    tables = FFBSBlockTables(model=model, source="batched")
    tables.refresh(state)
    scorer = model.scorer_for(state)
    for n, table in enumerate(tables.tables_for(state)):
        eager = build_log_block_scores(scorer, n, len(TRACES[n]), K, MIN_BLOCK_WIDTH,
                                       MAX_BLOCK_WIDTH)
        finite = np.isfinite(table)
        assert np.array_equal(finite, np.isfinite(eager))
        assert np.abs(table[finite] - eager[finite]).max() < 1e-9


def test_an_unknown_table_source_is_rejected():
    with pytest.raises(ValueError, match="unknown table source"):
        FFBSBlockTables(model=make_model(), source="magic")


def test_illegal_widths_are_minus_infinity_in_the_table():
    model = make_model()
    state = make_state(np.random.default_rng(31))
    tables = FFBSBlockTables(model=model, source="batched")
    tables.refresh(state)
    for n, table in enumerate(tables.tables_for(state)):
        J = len(TRACES[n])
        for a in range(J):
            for b in range(J + 1):
                legal = MIN_BLOCK_WIDTH <= b - a <= MAX_BLOCK_WIDTH
                assert np.isfinite(table[a, b]).all() == legal
