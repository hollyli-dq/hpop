"""The skill-local table refresh must be EXACT, and the gate must be able to say it isn't.

`table_source="fast"` rebuilds only the skill whose `U` moved. That is a factor of `K` on
the dominant cost of the learned-order arm, and it is exactly the kind of optimisation
that can be *almost* right: a stale column shifts a score, then a log ratio, then an
accept/reject decision, and nothing downstream reports it.

So two things are tested here. That fast equals exact — bitwise, on masks, scores,
collapsed deltas and decisions. And that these checks would **fail** if it did not, which
is the part that makes the first one worth anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_cpa.block_tables import CPABlockScoreTable
from hpop.mcmc_cpa.corpus import generate_ladder_corpus
from hpop.mcmc_cpa.nested_library import draw_master_library
from hpop.mcmc_original.stage6e_state import Stage6EModel

PARAMS = dict(beta=1.0, omega=1.0, lambda_rep=0.1, lambda_back=0.1)


@pytest.fixture(scope="module")
def rung():
    library, _ = draw_master_library(0)
    corpus = generate_ladder_corpus(library, 3, 0)
    u_by_skill, role_maps = library.prefix(3)
    model = Stage6EModel(traces=corpus.traces("train"), epsilon=0.02, delta_b=0.15,
                         n_skills=3, n_roles=library.n_roles, min_width=3, max_width=12,
                         infer_pi_P=True, eta_initial=1.0, eta_transition=1.0)
    return corpus, model, role_maps, u_by_skill


def _table(model, role_maps):
    return CPABlockScoreTable(traces=model.traces, epsilon=0.02, role_maps=role_maps,
                              min_width=model.min_width, max_width=model.max_width)


def _move_one_skill(u, skill, rng, scale=0.4):
    candidate = np.array(u, dtype=float, copy=True)
    row = int(rng.integers(candidate.shape[1]))
    candidate[skill, row] += scale * rng.standard_normal(candidate.shape[2])
    return candidate


def _reorder_one_skill(u, skill):
    """A change that is guaranteed to move the skill's column.

    The candidate score reads `U` **only** through the induced precedence relation --
    every quantity in the builder derives from `all(u[:,None,:] > u[None,:,:], axis=2)`
    and none from the coordinates themselves. So a random nudge often leaves the column
    bit-identical, and a mutation test built on one can pass while testing nothing.
    Negating a skill's `U` reverses its order and is certain to bite.
    """
    candidate = np.array(u, dtype=float, copy=True)
    candidate[skill] = -candidate[skill]
    return candidate


# ------------------------------------------------------------------- fast equals exact
def test_skill_local_refresh_is_bitwise_identical_to_a_full_rebuild(rung):
    _, model, role_maps, u = rung
    exact, fast = _table(model, role_maps), _table(model, role_maps)
    exact.refresh(u, **PARAMS)
    fast.refresh(u, **PARAMS)

    rng = np.random.default_rng(4)
    for move in range(6):
        skill = int(rng.integers(model.n_skills))
        candidate = _move_one_skill(u, skill, rng)
        exact.refresh(candidate, **PARAMS)
        info = fast.refresh_changed(candidate, **PARAMS)
        assert info["rebuilt_skills"] == [skill], info["rebuilt_skills"]
        for n in range(len(model.traces)):
            a, b = exact.tables[n], fast.tables[n]
            np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
            np.testing.assert_array_equal(a, b)
        u = candidate


def test_a_shared_parameter_move_rebuilds_every_skill(rung):
    """`beta` and friends are shared, so a move in any of them staleness every column.
    Rebuilding one would be the classic silent-staleness bug."""
    _, model, role_maps, u = rung
    fast = _table(model, role_maps)
    fast.refresh(u, **PARAMS)
    for field in ("beta", "omega", "lambda_rep", "lambda_back"):
        fast.refresh(u, **PARAMS)
        info = fast.refresh_changed(u, **(PARAMS | {field: 0.37}))
        assert info["rebuilt_skills"] == list(range(model.n_skills)), \
            f"{field} moved but only {info['rebuilt_skills']} were rebuilt"


def test_no_move_rebuilds_nothing(rung):
    _, model, role_maps, u = rung
    fast = _table(model, role_maps)
    fast.refresh(u, **PARAMS)
    assert fast.refresh_changed(u, **PARAMS)["rebuilt_skills"] == []


def test_two_skills_moving_rebuilds_exactly_those_two(rung):
    _, model, role_maps, u = rung
    fast = _table(model, role_maps)
    fast.refresh(u, **PARAMS)
    rng = np.random.default_rng(9)
    candidate = _move_one_skill(_move_one_skill(u, 0, rng), 2, rng)
    assert fast.refresh_changed(candidate, **PARAMS)["rebuilt_skills"] == [0, 2]


def test_the_collapsed_delta_is_identical_under_both_paths(rung):
    """The number a `U` move is actually accepted or rejected on."""
    from hpop.mcmc_original.semi_markov_ffbs import forward
    from hpop.mcmc_original.transitions import log_transition_matrix

    corpus, model, role_maps, u = rung
    log_pi = np.log(np.asarray(corpus.pi, dtype=float))
    log_p = log_transition_matrix(np.asarray(corpus.transition, dtype=float))

    def log_z(backend):
        return np.array([forward(t, log_pi, log_p, model.delta_b, model.max_width,
                                 model.min_width).log_normalizer
                         for t in backend.tables], dtype=float)

    exact, fast = _table(model, role_maps), _table(model, role_maps)
    exact.refresh(u, **PARAMS)
    fast.refresh(u, **PARAMS)
    base_exact, base_fast = log_z(exact), log_z(fast)

    rng = np.random.default_rng(17)
    for _ in range(4):
        skill = int(rng.integers(model.n_skills))
        candidate = _move_one_skill(u, skill, rng)
        exact.refresh(candidate, **PARAMS)
        fast.refresh_changed(candidate, **PARAMS)
        d_exact = float((log_z(exact) - base_exact).sum())
        d_fast = float((log_z(fast) - base_fast).sum())
        assert d_exact == d_fast, f"collapsed delta differs: {d_exact} vs {d_fast}"


# ---------------------------------------- the checks would catch a broken fast path
def test_a_stale_column_is_detected(rung):
    """Mutation: skip the rebuild entirely and confirm the comparison fails.

    Without this, "fast equals exact" could be passing because both paths are wrong in
    the same way, or because the comparison has no teeth.
    """
    _, model, role_maps, u = rung
    exact, broken = _table(model, role_maps), _table(model, role_maps)
    exact.refresh(u, **PARAMS)
    broken.refresh(u, **PARAMS)

    candidate = _reorder_one_skill(u, 1)
    exact.refresh(candidate, **PARAMS)
    broken.refresh(candidate, **PARAMS, skills=[])                     # rebuild nothing

    differs = any(not np.array_equal(exact.tables[n], broken.tables[n])
                  for n in range(len(model.traces)))
    assert differs, "a completely skipped rebuild went undetected — the check is inert"


def test_rebuilding_the_wrong_skill_is_detected(rung):
    _, model, role_maps, u = rung
    exact, broken = _table(model, role_maps), _table(model, role_maps)
    exact.refresh(u, **PARAMS)
    broken.refresh(u, **PARAMS)

    candidate = _reorder_one_skill(u, 1)
    exact.refresh(candidate, **PARAMS)
    broken.refresh(candidate, **PARAMS, skills=[0])        # the wrong column

    differs = any(not np.array_equal(exact.tables[n], broken.tables[n])
                  for n in range(len(model.traces)))
    assert differs, "rebuilding the wrong skill went undetected"


def test_the_score_reads_u_only_through_the_induced_order(rung):
    """Recorded because it is what makes a naive mutation test inert -- and because it is
    a real property of the model: a monotone rescale of `U` is the same latent poset and
    must give the same candidate scores, bit for bit."""
    from hpop.mcmc_original.latent_poset import precedence_from_u

    _, model, role_maps, u = rung
    rescaled = u * 3.0 + 1.0                        # strictly monotone per coordinate
    for k in range(model.n_skills):
        np.testing.assert_array_equal(precedence_from_u(u[k]),
                                      precedence_from_u(rescaled[k]))
    a, b = _table(model, role_maps), _table(model, role_maps)
    a.refresh(u, **PARAMS)
    b.refresh(rescaled, **PARAMS)
    for n in range(len(model.traces)):
        np.testing.assert_array_equal(a.tables[n], b.tables[n])

    reordered = _reorder_one_skill(u, 1)
    c = _table(model, role_maps)
    c.refresh(reordered, **PARAMS)
    assert any(not np.array_equal(a.tables[n], c.tables[n])
               for n in range(len(model.traces))), \
        "reversing a skill's order left every score unchanged"


def test_out_of_range_skills_are_refused(rung):
    _, model, role_maps, u = rung
    table = _table(model, role_maps)
    with pytest.raises(ValueError):
        table.refresh(u, **PARAMS, skills=[model.n_skills])
    with pytest.raises(ValueError):
        table.refresh(u, **PARAMS, skills=[-1])
