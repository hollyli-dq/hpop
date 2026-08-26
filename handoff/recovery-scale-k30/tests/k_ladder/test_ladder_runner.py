"""The two arms must differ in exactly one place: the candidate block score.

If anything else differs — the initialisation, the RNG stream, the segmentation prior, the
transition treatment, the data — then a gap between the arms is not evidence about the
recurrent likelihood, and the baseline is worthless. These tests pin that down directly
rather than trusting that one runner was written to match another.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_cpa.corpus import generate_ladder_corpus
from hpop.mcmc_cpa.ladder_runner import (FULL_RFS, SUPPORT_ONLY, ArmTables,
                                         run_ladder_chain)
from hpop.mcmc_cpa.nested_library import draw_master_library
from hpop.mcmc_original.stage6e_state import Stage6EModel

ARMS = (FULL_RFS, SUPPORT_ONLY)


class _FakeState:
    """Only the fields `ArmTables.refresh` reads."""

    def __init__(self, u_by_skill, beta=1.0, omega=1.0, lambda_rep=0.0, lambda_back=0.0):
        self.u_by_skill = np.asarray(u_by_skill, dtype=float)
        self.beta, self.omega = beta, omega
        self.lambda_rep, self.lambda_back = lambda_rep, lambda_back


@pytest.fixture(scope="module")
def rung():
    """A small K=3 rung, built once — the corpus generator is the slow part."""
    library, _ = draw_master_library(0)
    corpus = generate_ladder_corpus(library, 3, 0)
    u_by_skill, role_maps = library.prefix(3)
    model = Stage6EModel(traces=corpus.traces("train"), epsilon=0.02, delta_b=0.15,
                         n_skills=3, n_roles=library.n_roles,
                         min_width=3, max_width=12, infer_pi_P=True,
                         eta_initial=1.0, eta_transition=1.0)
    return corpus, model, role_maps, u_by_skill


def _run(rung, arm, **kw):
    _, model, role_maps, u_by_skill = rung
    kw.setdefault("chain", 0)
    kw.setdefault("sweeps", 6)
    kw.setdefault("warmup", 2)
    kw.setdefault("seed", 4242)
    kw.setdefault("thin", 2)
    return run_ladder_chain(arm, model, role_maps, u_by_skill, **kw)


# ------------------------------------------------------------- the arms share everything
def test_both_arms_share_their_initial_state(rung):
    """Zero sweeps exposes the initialisation alone. It must not depend on the arm."""
    runs = {arm: _run(rung, arm, sweeps=0, warmup=0) for arm in ARMS}
    assert runs[FULL_RFS]["final_pi"] == runs[SUPPORT_ONLY]["final_pi"]
    assert runs[FULL_RFS]["final_transition"] == runs[SUPPORT_ONLY]["final_transition"]


def test_both_arms_retain_the_same_number_of_draws(rung):
    """Same sweep/warmup/thin schedule, so the recording schedule cannot differ."""
    runs = {arm: _run(rung, arm) for arm in ARMS}
    assert runs[FULL_RFS]["retained_draws"] == runs[SUPPORT_ONLY]["retained_draws"] == 2


def test_each_arm_is_deterministic_given_its_seed(rung):
    for arm in ARMS:
        first, second = _run(rung, arm), _run(rung, arm)
        assert first["draws"]["labels"] == second["draws"]["labels"]
        assert first["draws"]["boundaries"] == second["draws"]["boundaries"]


def test_the_seed_actually_moves_the_chain(rung):
    """Guards against a runner that ignores its seed and would make the arms agree
    for the wrong reason."""
    assert (_run(rung, FULL_RFS, seed=1)["draws"]["labels"]
            != _run(rung, FULL_RFS, seed=999)["draws"]["labels"])


def test_u_is_held_fixed_in_both_arms(rung):
    """`U` moving in one arm and not the other would confound the comparison."""
    _, _, _, u_by_skill = rung
    before = np.array(u_by_skill, dtype=float, copy=True)
    for arm in ARMS:
        result = _run(rung, arm)
        assert result["u_held_fixed"] is True
        np.testing.assert_array_equal(np.asarray(u_by_skill, dtype=float), before)


# ------------------------------------------------------------ the arms differ in one place
def test_the_arms_do_not_produce_the_same_chain(rung):
    """Identical everywhere else, so if the block score did nothing the arms would be
    indistinguishable and the baseline would be measuring nothing."""
    assert (_run(rung, FULL_RFS)["draws"]["labels"]
            != _run(rung, SUPPORT_ONLY)["draws"]["labels"])


def test_the_support_only_table_never_rebuilds(rung):
    """Its score does not depend on `U`, so a rebuild would be either wasted work or a
    sign that `U` had leaked into the baseline."""
    _, model, role_maps, u_by_skill = rung
    adapter = ArmTables(SUPPORT_ONLY, model, role_maps, 0.02)
    state = _FakeState(u_by_skill)
    assert adapter.refresh(state)["rebuilt"] is False
    adapter.mark_stale()
    assert adapter.refresh(state)["rebuilt"] is False
    # and it does not read U at all, so a different U is not an error for it
    assert adapter.refresh(_FakeState(np.zeros_like(np.asarray(u_by_skill,
                                                               dtype=float))))


def test_support_only_never_claims_structure_recovery(rung):
    """Its score does not read `U`, so any `U` estimate it reported would be a prior draw
    dressed up as a posterior."""
    assert _run(rung, SUPPORT_ONLY)["structure_recovery"] == "NOT APPLICABLE"
    assert _run(rung, FULL_RFS)["structure_recovery"] == "available"


def test_support_only_labels_are_always_support_feasible(rung):
    """The baseline scores an incompatible block at -inf, so no retained draw may label a
    block with a skill whose support misses one of its CPAs."""
    corpus, model, role_maps, _ = rung
    supports = [set(int(c) for c in row) for row in role_maps.supports()]
    result = _run(rung, SUPPORT_ONLY, sweeps=10, warmup=4)
    assert result["retained_draws"] > 0
    for labels, boundaries in zip(result["draws"]["labels"],
                                  result["draws"]["boundaries"]):
        for trace, trace_labels, trace_bounds in zip(model.traces, labels, boundaries):
            starts = [0] + list(trace_bounds)
            ends = list(trace_bounds) + [len(trace)]
            for (a, b), skill in zip(zip(starts, ends), trace_labels):
                assert set(int(c) for c in trace[a:b]) <= supports[skill]


def test_stale_tables_are_refused_before_they_can_be_used(rung):
    """The full arm scores at a particular `U`; serving a table built at another one
    would silently break the likelihood."""
    _, model, role_maps, u_by_skill = rung
    adapter = ArmTables(FULL_RFS, model, role_maps, 0.02)
    state = _FakeState(u_by_skill)
    adapter.refresh(state)
    adapter.tables_for(state)
    adapter.mark_stale()
    with pytest.raises(AssertionError):
        adapter.tables_for(state)


def test_the_full_arm_refuses_a_table_built_at_a_different_u(rung):
    _, model, role_maps, u_by_skill = rung
    adapter = ArmTables(FULL_RFS, model, role_maps, 0.02)
    adapter.refresh(_FakeState(u_by_skill))
    moved = np.asarray(u_by_skill, dtype=float).copy()
    moved[0] = -moved[0]
    with pytest.raises(AssertionError):
        adapter.tables_for(_FakeState(moved))


def test_an_unknown_arm_is_refused(rung):
    with pytest.raises((ValueError, KeyError)):
        _run(rung, "whatever-arm")


# ------------------------------------------- the rebuild memo must be a pure optimisation
def test_the_memo_returns_a_bit_identical_table(rung):
    """The candidate table is a deterministic function of the five inputs the memo keys
    on. If reusing it ever differed from rebuilding it, every chain that hit the memo
    would be sampling from a different model."""
    _, model, role_maps, u_by_skill = rung
    state = _FakeState(u_by_skill)

    memoised = ArmTables(FULL_RFS, model, role_maps, 0.02)
    assert memoised.refresh(state)["rebuilt"] is True
    assert memoised.refresh(state)["rebuilt"] is False, "second refresh should hit the memo"
    reused = [np.array(t, copy=True) for t in memoised.tables_for(state)]

    forced = ArmTables(FULL_RFS, model, role_maps, 0.02)
    forced.refresh(state)
    forced._built_key = None                      # defeat the memo
    assert forced.refresh(state)["rebuilt"] is True
    rebuilt = forced.tables_for(state)

    assert len(reused) == len(rebuilt)
    for a, b in zip(reused, rebuilt):
        np.testing.assert_array_equal(a, b)


def test_the_memo_misses_when_any_of_its_inputs_moves(rung):
    """A memo that held on through a parameter change would silently score at the wrong
    `U`, which is exactly the failure `tables_for` exists to catch."""
    _, model, role_maps, u_by_skill = rung
    adapter = ArmTables(FULL_RFS, model, role_maps, 0.02)
    adapter.refresh(_FakeState(u_by_skill))

    moved_u = np.asarray(u_by_skill, dtype=float).copy()
    moved_u[0, 0, 0] += 0.5
    assert adapter.refresh(_FakeState(moved_u))["rebuilt"] is True

    for field in ("beta", "omega", "lambda_rep", "lambda_back"):
        adapter.refresh(_FakeState(moved_u))
        assert adapter.refresh(_FakeState(moved_u, **{field: 0.37}))["rebuilt"] is True, \
            f"{field} moved but the memo held"


def test_the_memo_does_not_change_the_chain(rung):
    """End-to-end: the draws must not depend on whether the table was rebuilt."""
    _, model, role_maps, u_by_skill = rung
    with_memo = run_ladder_chain(FULL_RFS, model, role_maps, u_by_skill, chain=0,
                                 sweeps=6, warmup=2, seed=4242, thin=2)

    import hpop.mcmc_cpa.ladder_runner as runner_module
    original = ArmTables.refresh

    def always_rebuild(self, state):
        self._built_key = None
        return original(self, state)

    ArmTables.refresh = always_rebuild
    try:
        without_memo = run_ladder_chain(FULL_RFS, model, role_maps, u_by_skill, chain=0,
                                        sweeps=6, warmup=2, seed=4242, thin=2)
    finally:
        ArmTables.refresh = original

    assert with_memo["draws"]["labels"] == without_memo["draws"]["labels"]
    assert with_memo["draws"]["boundaries"] == without_memo["draws"]["boundaries"]
    assert with_memo["ffbs_states_changed_total"] == \
        without_memo["ffbs_states_changed_total"]
