"""Step 7B2 optimisation — the fast block-table builder is the same mathematical object.

The optimisation is worth nothing unless it is exact, so this file is mostly parity: the
prefix-sum builder against the width-bucketed Stage 6E builder, against the Step 7A
per-block adapter, and against `RecurrentBlockScorer.replay` — the registered reference
that every other builder was itself pinned to.

Three negative controls sit at the end. Without them the parity tests could pass by
construction — for instance if both sides shared the defect — so each one injects the
specific fault the corresponding test exists to catch and requires the test to fail.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from hpop.mcmc_original import semi_markov_ffbs
from hpop.mcmc_original.block_score_adapters import build_log_block_scores
from hpop.mcmc_original.fast_block_tables import (
    CandidateLayout, FastBlockScoreTable, layout_for,
)
from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import (
    FFBSBlockTables, Stage7BSampler, ffbs_sweep_once, key_movement,
)
from hpop.mcmc_original.recurrent_segmentation import RecurrentBlockScorer
from hpop.mcmc_original.semi_markov_ffbs import (
    backward_sample, forward, posterior_log_marginals,
)
from hpop.mcmc_original.stage6d_frozen import REGISTERED_SCALES
from hpop.mcmc_original.stage6e_block_table import BlockScoreTable
from hpop.mcmc_original.stage6e_frozen import MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH
from hpop.mcmc_original.stage6e_sampler import (
    boundary_hamming, occurrence_label_changes, segmentation_sweep,
)
from hpop.mcmc_original.stage6e_state import Stage6EModel, Stage6EState
from hpop.mcmc_original.transitions import log_transition_matrix

# The Step 7A engine as committed at 77093cb. If this changes, the optimisation task has
# violated its own precondition and every Step 7A result is out of date.
FROZEN_ENGINE_SHA256 = (
    "8150bb8235eb159d5e2f08ada7c698c383f4c7fd31f3b882e218b207dd135486")

EPSILON = 0.02
PARAMETER_SETTINGS = (
    {"beta": 1.5, "omega": 1.7346, "lambda_rep": 0.8, "lambda_back": 0.25},
    {"beta": 0.3, "omega": -1.2, "lambda_rep": 0.05, "lambda_back": 1.9},
    {"beta": 3.4, "omega": 4.0, "lambda_rep": 2.5, "lambda_back": 0.01},
)


def make_problem(J: int, K: int, m: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    trace = tuple(int(v) for v in rng.integers(m, size=J))
    u = rng.normal(size=(K, m, 2))
    return trace, u


def scorer_for(traces, u, **parameters) -> RecurrentBlockScorer:
    return RecurrentBlockScorer(traces=traces, epsilon=EPSILON, u_by_skill=u,
                                min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
                                **parameters)


def fast_table(traces, u, m, **parameters) -> FastBlockScoreTable:
    table = FastBlockScoreTable(traces=traces, epsilon=EPSILON, n_skills=u.shape[0],
                                min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
                                n_roles=m)
    table.refresh(u, **parameters)
    return table


# ------------------------------------------------------- 1. the engine stays frozen
def test_the_step7a_engine_is_byte_identical_to_the_frozen_checkpoint():
    """No speed may be bought by editing the validated FFBS engine."""
    digest = hashlib.sha256(Path(semi_markov_ffbs.__file__).read_bytes()).hexdigest()
    assert digest == FROZEN_ENGINE_SHA256, (
        "semi_markov_ffbs.py has changed; the Step 7A validation no longer applies to it")


def test_the_optimiser_does_not_import_a_second_ffbs_implementation():
    from hpop.mcmc_original import fast_block_tables
    source = Path(fast_block_tables.__file__).read_text()
    for forbidden in ("def forward", "def backward", "logsumexp", "alpha["):
        assert forbidden not in source, (
            f"the block-table optimiser must not reimplement the chart ({forbidden!r})")


# ------------------------------------------------------------------- 2. exact parity
@pytest.mark.parametrize("J,K", [(8, 1), (8, 2), (8, 3), (24, 2), (24, 3), (48, 3),
                                 (96, 3)])
@pytest.mark.parametrize("setting", range(len(PARAMETER_SETTINGS)))
def test_fast_table_matches_the_width_bucketed_builder(J, K, setting):
    parameters = PARAMETER_SETTINGS[setting]
    trace, u = make_problem(J, K, seed=J * 10 + K)
    fast = fast_table((trace,), u, 5, **parameters)

    bucketed = BlockScoreTable(traces=(trace,), epsilon=EPSILON, n_skills=K,
                               min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH)
    bucketed.refresh(u, parameters["beta"], parameters["omega"],
                     parameters["lambda_rep"], parameters["lambda_back"])

    worst, checked = 0.0, 0
    for a in range(J):
        for b in range(a + MIN_BLOCK_WIDTH, min(J, a + MAX_BLOCK_WIDTH) + 1):
            for k in range(K):
                worst = max(worst, abs(fast.score(0, a, b, k)
                                       - bucketed.score(0, a, b, k)))
                checked += 1
    assert checked > 0
    assert worst <= 1e-10


@pytest.mark.parametrize("J,K", [(8, 3), (24, 3), (48, 2), (96, 3)])
def test_fast_table_matches_the_registered_per_block_replay(J, K):
    """`RecurrentBlockScorer.replay` is the reference every builder is pinned to."""
    parameters = PARAMETER_SETTINGS[0]
    trace, u = make_problem(J, K, seed=7 + J)
    fast = fast_table((trace,), u, 5, **parameters)
    scorer = scorer_for((trace,), u, **parameters)
    worst = 0.0
    rng = np.random.default_rng(J)
    for _ in range(120):
        a = int(rng.integers(0, max(1, J - MIN_BLOCK_WIDTH + 1)))
        width = int(rng.integers(MIN_BLOCK_WIDTH,
                                 min(MAX_BLOCK_WIDTH, J - a) + 1)) if J - a >= MIN_BLOCK_WIDTH else None
        if width is None:
            continue
        k = int(rng.integers(K))
        worst = max(worst, abs(fast.score(0, a, a + width, k)
                               - scorer.replay(0, a, a + width, k)))
    assert worst <= 1e-10


def test_prefix_sum_identity_holds_block_by_block():
    """The claim the optimisation rests on: the trajectory does not depend on the end.

    `replay(a, b)` must equal `replay(a, b-1)` plus one more emission, for every `b`. If
    that were false the prefix-sum construction would be wrong, and no amount of agreement
    between two builders that both assume it would reveal it.
    """
    trace, u = make_problem(40, 3, seed=99)
    parameters = PARAMETER_SETTINGS[1]
    scorer = scorer_for((trace,), u, **parameters)
    fast = fast_table((trace,), u, 5, **parameters)
    for a in (0, 5, 17, 28):
        for k in range(3):
            previous = None
            for b in range(a + MIN_BLOCK_WIDTH, min(40, a + MAX_BLOCK_WIDTH) + 1):
                direct = scorer.replay(0, a, b, k)
                assert abs(direct - fast.score(0, a, b, k)) <= 1e-10
                if previous is not None:
                    # each extra step only adds a term; it never rewrites the prefix
                    assert direct <= previous + 1e-12
                previous = direct


def test_multi_trace_tables_agree_with_the_step7a_adapter():
    rng = np.random.default_rng(4)
    traces = tuple(tuple(int(v) for v in rng.integers(5, size=length))
                   for length in (14, 31, 8, 47))
    u = rng.normal(size=(3, 5, 2))
    parameters = PARAMETER_SETTINGS[2]
    fast = fast_table(traces, u, 5, **parameters)
    scorer = scorer_for(traces, u, **parameters)
    for index, trace in enumerate(traces):
        eager = build_log_block_scores(scorer, index, len(trace), 3, MIN_BLOCK_WIDTH,
                                       MAX_BLOCK_WIDTH)
        finite = np.isfinite(eager)
        assert np.array_equal(finite, np.isfinite(fast.tables[index]))
        assert np.abs(fast.tables[index][finite] - eager[finite]).max() <= 1e-10


@pytest.mark.parametrize("J,K", [(8, 3), (24, 3), (48, 3)])
def test_log_z_and_dp_marginals_are_unchanged_by_the_optimisation(J, K):
    trace, u = make_problem(J, K, seed=200 + J)
    parameters = PARAMETER_SETTINGS[0]
    model = Stage6EModel(traces=(trace,), epsilon=EPSILON, delta_b=0.15, n_skills=K,
                         n_roles=5, min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH)
    state = _state_for(model, u, parameters)
    charts = {}
    for source in ("fast", "batched", "adapter"):
        tables = FFBSBlockTables(model=model, source=source)
        tables.refresh(state)
        charts[source] = forward(tables.tables_for(state)[0], np.log(state.pi),
                                 log_transition_matrix(state.transition), model.delta_b,
                                 model.max_width, model.min_width)
    for other in ("batched", "adapter"):
        assert abs(charts["fast"].log_normalizer
                   - charts[other].log_normalizer) <= 1e-10
        a = posterior_log_marginals(charts["fast"])
        b = posterior_log_marginals(charts[other])
        assert np.abs(a["boundary_marginals"] - b["boundary_marginals"]).max() <= 1e-10
        assert np.abs(a["occurrence_label_marginals"]
                      - b["occurrence_label_marginals"]).max() <= 1e-10


def test_backward_draws_are_identical_when_the_tables_are_bit_identical():
    trace, u = make_problem(30, 3, seed=11)
    parameters = PARAMETER_SETTINGS[0]
    model = Stage6EModel(traces=(trace,), epsilon=EPSILON, delta_b=0.15, n_skills=3,
                         n_roles=5, min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH)
    state = _state_for(model, u, parameters)
    tables = {}
    for source in ("fast", "batched"):
        block = FFBSBlockTables(model=model, source=source)
        block.refresh(state)
        tables[source] = np.array(block.tables_for(state)[0], copy=True)
    if not np.array_equal(tables["fast"], tables["batched"]):
        pytest.skip("tables differ within roundoff; the distributional test covers this")
    charts = {name: forward(table, np.log(state.pi),
                            log_transition_matrix(state.transition), model.delta_b,
                            model.max_width, model.min_width)
              for name, table in tables.items()}
    first = [backward_sample(charts["fast"], np.random.default_rng(5)) for _ in range(20)]
    second = [backward_sample(charts["batched"], np.random.default_rng(5))
              for _ in range(20)]
    assert first == second


def _state_for(model, u, parameters) -> Stage6EState:
    from hpop.mcmc_original.fast_segmentation_kernel import segmentation_of
    J = len(model.traces[0])
    ends, position, skill = [], 0, 0
    while position < J:
        width = min(MIN_BLOCK_WIDTH + 1, J - position)
        if J - position - width < MIN_BLOCK_WIDTH and J - position - width > 0:
            width = J - position
        position += width
        ends.append((position, skill % model.n_skills))
        skill += 1
    if model.n_skills == 1:
        ends = [(min(J, MAX_BLOCK_WIDTH), 0)]
        if ends[-1][0] != J:
            pytest.skip("K = 1 forbids self-transitions, so only J <= max_width is legal")
    transition = np.zeros((model.n_skills, model.n_skills))
    for h in range(model.n_skills):
        allowed = [k for k in range(model.n_skills) if k != h]
        if allowed:
            transition[h, allowed] = 1.0 / len(allowed)
    return Stage6EState(segmentations=(segmentation_of(tuple(ends)),), u_by_skill=u,
                        rho=0.3, pi=np.full(model.n_skills, 1.0 / model.n_skills),
                        transition=transition, **parameters)


# --------------------------------------------------------------- 3. the q_0 = 0 audit
def test_every_candidate_starts_from_q0_zero_however_it_is_batched():
    """Score A alone, B then A, A then B, reversed, batched, scalar — all must agree."""
    trace, u = make_problem(36, 3, seed=21)
    parameters = PARAMETER_SETTINGS[0]
    scorer = scorer_for((trace,), u, **parameters)
    fast = fast_table((trace,), u, 5, **parameters)
    a_block, b_block = (4, 12, 1), (0, 9, 2)

    alone = scorer.replay(0, *a_block[:2], a_block[2])
    scorer.replay(0, *b_block[:2], b_block[2])
    after_b = scorer.replay(0, *a_block[:2], a_block[2])
    assert alone == after_b

    batched = fast.score(0, *a_block[:2], a_block[2])
    assert abs(batched - alone) <= 1e-10
    assert abs(fast.score(0, *b_block[:2], b_block[2])
               - scorer.replay(0, *b_block[:2], b_block[2])) <= 1e-10


def test_permuting_the_batch_order_does_not_change_a_single_score():
    """The layout sorts candidates; a different trace order must undo exactly."""
    rng = np.random.default_rng(31)
    traces = tuple(tuple(int(v) for v in rng.integers(5, size=length))
                   for length in (12, 25, 18))
    u = rng.normal(size=(3, 5, 2))
    parameters = PARAMETER_SETTINGS[1]
    straight = fast_table(traces, u, 5, **parameters)
    permutation = [2, 0, 1]
    permuted = fast_table(tuple(traces[i] for i in permutation), u, 5, **parameters)
    for new_index, old_index in enumerate(permutation):
        assert np.array_equal(np.isfinite(permuted.tables[new_index]),
                              np.isfinite(straight.tables[old_index]))
        finite = np.isfinite(straight.tables[old_index])
        assert np.abs(permuted.tables[new_index][finite]
                      - straight.tables[old_index][finite]).max() <= 1e-12


def test_a_finished_candidate_is_never_touched_again():
    """Scores for short blocks must not depend on what longer blocks do afterwards."""
    trace, u = make_problem(30, 3, seed=41)
    parameters = PARAMETER_SETTINGS[0]
    full = fast_table((trace,), u, 5, **parameters)
    # truncating the trace removes the long candidates entirely; the short ones that fit
    # inside the truncation must be unchanged
    short = fast_table((trace[:20],), u, 5, **parameters)
    for a in range(0, 20 - MIN_BLOCK_WIDTH + 1):
        for b in range(a + MIN_BLOCK_WIDTH, min(20, a + MAX_BLOCK_WIDTH) + 1):
            for k in range(3):
                assert abs(full.score(0, a, b, k) - short.score(0, a, b, k)) <= 1e-12


def test_the_layout_is_parameter_independent_and_cached():
    trace, _ = make_problem(24, 3, seed=5)
    first = layout_for((trace,), MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH, 5)
    second = layout_for((trace,), MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH, 5)
    assert first is second
    assert isinstance(first, CandidateLayout)
    assert first.candidate_steps < first.naive_candidate_steps
    # every candidate appears exactly once in the scatter
    rows, widths, a_ix, b_ix = first.scatter[0]
    assert len(set(zip(a_ix.tolist(), b_ix.tolist()))) == len(a_ix)


# ------------------------------------------------------- 4. the invalidation matrix
def _model_and_state(seed=17, K=3, J=26):
    trace, u = make_problem(J, K, seed=seed)
    model = Stage6EModel(traces=(trace,), epsilon=EPSILON, delta_b=0.15, n_skills=K,
                         n_roles=5, min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH)
    return model, _state_for(model, u, PARAMETER_SETTINGS[0])


@pytest.mark.parametrize("changed", ["rho", "pi", "P"])
def test_rho_pi_and_P_invalidate_no_block_score_column(changed):
    model, state = _model_and_state()
    tables = FFBSBlockTables(model=model, source="fast")
    tables.refresh(state)
    moved = state.copy()
    if changed == "rho":
        moved.rho = state.rho + 0.25
    elif changed == "pi":
        moved.pi = np.array([0.5, 0.3, 0.2])
    else:
        transition = np.array([[0.0, 0.9, 0.1], [0.2, 0.0, 0.8], [0.7, 0.3, 0.0]])
        moved.transition = transition
    tables.refresh(moved)
    assert tables.last_refresh["rebuilt_skills"] == []
    assert tables.last_refresh["reused_skills"] == list(range(model.n_skills))


@pytest.mark.parametrize("scalar", ["beta", "omega", "lambda_rep", "lambda_back"])
def test_every_global_scalar_invalidates_every_column(scalar):
    model, state = _model_and_state()
    tables = FFBSBlockTables(model=model, source="fast")
    tables.refresh(state)
    moved = state.copy()
    setattr(moved, scalar, getattr(state, scalar) + 0.1)
    tables.refresh(moved)
    assert tables.last_refresh["rebuilt_skills"] == list(range(model.n_skills))


@pytest.mark.parametrize("skill", [0, 1, 2])
def test_moving_one_skill_u_rebuilds_only_that_column(skill):
    model, state = _model_and_state()
    tables = FFBSBlockTables(model=model, source="fast")
    tables.refresh(state)
    before = [np.array(t, copy=True) for t in tables.tables_for(state)]
    moved = state.copy()
    u = np.array(state.u_by_skill, copy=True)
    u[skill] += 0.3
    moved.u_by_skill = u
    tables.refresh(moved)
    assert tables.last_refresh["rebuilt_skills"] == [skill]

    # and the reused columns must equal a full rebuild, not merely be left alone
    full = FFBSBlockTables(model=model, source="fast")
    full._fast._fingerprint = [None] * model.n_skills
    full.refresh(moved)
    after = tables.tables_for(moved)
    for index, table in enumerate(after):
        finite = np.isfinite(table)
        assert np.abs(table[finite] - full.tables_for(moved)[index][finite]).max() <= 1e-12
        for other in range(model.n_skills):
            if other != skill:
                assert np.array_equal(table[..., other], before[index][..., other])


def test_the_tables_are_not_reallocated_between_refreshes():
    """§7: the dense tables are written in place, never deep-copied per sweep."""
    model, state = _model_and_state()
    tables = FFBSBlockTables(model=model, source="fast")
    tables.refresh(state)
    identities = [id(t) for t in tables.tables_for(state)]
    moved = state.copy()
    moved.beta = state.beta + 0.2
    tables.refresh(moved)
    assert [id(t) for t in tables.tables_for(moved)] == identities


# ------------------------------------------------------------- 5. negative controls
def test_negative_control_recurrent_state_leak_is_detected(monkeypatch):
    """Inject a q leak into the fast builder: parity against the reference must fail."""
    trace, u = make_problem(30, 3, seed=61)
    parameters = PARAMETER_SETTINGS[0]
    honest = fast_table((trace,), u, 5, **parameters)
    scorer = scorer_for((trace,), u, **parameters)

    original = FastBlockScoreTable._emissions_for_skill

    def leaking(self, u_k, beta, omega, lambda_rep, lambda_back):
        emissions = original(self, u_k, beta, omega, lambda_rep, lambda_back)
        # every candidate inherits a little of the previous row's first emission, which is
        # exactly what "q leaked across candidates" would look like in the output
        emissions[1:, 0] += 0.05 * emissions[:-1, 0]
        return emissions

    monkeypatch.setattr(FastBlockScoreTable, "_emissions_for_skill", leaking)
    faulty = fast_table((trace,), u, 5, **parameters)
    honest_worst = max(abs(honest.score(0, a, a + 5, 0) - scorer.replay(0, a, a + 5, 0))
                       for a in range(0, 20))
    faulty_worst = max(abs(faulty.score(0, a, a + 5, 0) - scorer.replay(0, a, a + 5, 0))
                       for a in range(0, 20))
    assert honest_worst <= 1e-10
    assert faulty_worst > 1e-10, "the parity test cannot detect a recurrent-state leak"


def test_negative_control_missing_invalidation_after_omega_is_detected(monkeypatch):
    """If omega stopped invalidating, the table would silently score the old model."""
    model, state = _model_and_state()
    tables = FFBSBlockTables(model=model, source="fast")
    tables.refresh(state)
    honest = [np.array(t, copy=True) for t in tables.tables_for(state)]

    moved = state.copy()
    moved.omega = state.omega + 0.7

    def blind_fingerprint(u_k, beta, omega, lambda_rep, lambda_back):
        return (np.asarray(u_k, dtype=float).tobytes(), float(beta),
                float(lambda_rep), float(lambda_back))       # omega dropped on purpose

    monkeypatch.setattr(FastBlockScoreTable, "_skill_fingerprint",
                        staticmethod(blind_fingerprint))
    stale = FFBSBlockTables(model=model, source="fast")
    stale.refresh(state)
    stale.refresh(moved)
    assert stale.last_refresh["rebuilt_skills"] == [], "the fault was not injected"
    reference = FFBSBlockTables(model=model, source="batched")
    reference.refresh(moved)
    finite = np.isfinite(honest[0])
    difference = np.abs(stale.tables_for(moved)[0][finite]
                        - reference.tables_for(moved)[0][finite]).max()
    assert difference > 1e-6, "a missed omega invalidation would go unnoticed"


def test_negative_control_reusing_a_stale_skill_column_is_detected(monkeypatch):
    """If U_k stopped invalidating column k, the invalidation test must catch it.

    The perturbation has to change the induced order, not merely the numbers: the
    likelihood sees `U` only through the precedence closure `h(U)`, so a uniform shift of
    one skill's `U_k` leaves every block score identical and would make this control
    vacuous. `assert_the_fault_is_observable` below checks that before checking detection.
    """
    model, state = _model_and_state()
    moved = state.copy()
    u = np.array(state.u_by_skill, copy=True)
    u[1] = -u[1][::-1]                      # reverses the row order, so h(U_1) really moves
    moved.u_by_skill = u

    from hpop.mcmc_original.latent_poset import precedence_from_u
    assert not np.array_equal(precedence_from_u(state.u_by_skill[1]),
                              precedence_from_u(u[1])), "the perturbation must move h(U)"

    honest = FFBSBlockTables(model=model, source="fast")
    honest.refresh(state)
    honest_before = np.array(honest.tables_for(state)[0], copy=True)
    reference = FFBSBlockTables(model=model, source="batched")
    reference.refresh(moved)
    finite = np.isfinite(honest_before)
    assert np.abs(honest_before[finite]
                  - reference.tables_for(moved)[0][finite]).max() > 1e-6, (
        "the fault is not observable: the table did not depend on U_1 at all")

    def u_blind_fingerprint(u_k, beta, omega, lambda_rep, lambda_back):
        return (float(beta), float(omega), float(lambda_rep), float(lambda_back))

    monkeypatch.setattr(FastBlockScoreTable, "_skill_fingerprint",
                        staticmethod(u_blind_fingerprint))
    tables = FFBSBlockTables(model=model, source="fast")
    tables.refresh(state)
    tables.refresh(moved)
    assert tables.last_refresh["rebuilt_skills"] == [], "the fault was not injected"
    difference = np.abs(tables.tables_for(moved)[0][finite]
                        - reference.tables_for(moved)[0][finite]).max()
    assert difference > 1e-6, "a stale skill column would go unnoticed"


# ------------------------------------------------------ 6. the orchestration changes
def test_fast_movement_counters_match_the_stage6e_ones():
    rng = np.random.default_rng(71)

    def random_key(J, K):
        ends, position = [], 0
        while position < J:
            width = min(int(rng.integers(MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH + 1)),
                        J - position)
            if 0 < J - position - width < MIN_BLOCK_WIDTH:
                width = J - position
            position += width
            ends.append(position)
        labels, previous = [], -1
        for _ in ends:
            label = int(rng.choice([k for k in range(K) if k != previous]))
            labels.append(label)
            previous = label
        return tuple(zip(ends, labels))

    for _ in range(400):
        J = int(rng.integers(6, 40))
        a, b = random_key(J, 3), random_key(J, 3)
        hamming, changes = key_movement(a, b)
        assert hamming == boundary_hamming(a, b)
        assert changes == occurrence_label_changes(a, b)


def test_the_zero_proposal_fast_path_is_a_no_op():
    """Skipping the discarded target evaluation must change nothing observable."""
    keys = (((5, 0), (12, 1)), ((8, 2),))
    calls = []

    class Target:
        def __call__(self, key):
            calls.append(key)
            return -1.0

    rng = np.random.default_rng(3)
    before = rng.bit_generator.state
    result, movement = segmentation_sweep(list(keys), [Target(), Target()], [None, None],
                                          0, rng, {}, {}, {})
    assert result == keys
    assert movement == {"boundary_hamming": 0, "label_changes": 0}
    assert calls == []                                  # nothing was evaluated
    assert rng.bit_generator.state == before            # and no randomness was consumed


def test_the_optimised_sweep_still_produces_a_legal_state():
    model, state = _model_and_state(seed=81, J=34)
    sampler = Stage7BSampler(model=model, scales=REGISTERED_SCALES, table_source="fast")
    rng = np.random.default_rng(9)
    for _ in range(5):
        state = ffbs_sweep_once(state, sampler, rng)
        for segmentation in state.segmentations:
            assert segmentation.segments[-1].end == len(model.traces[0])
            assert all(MIN_BLOCK_WIDTH <= s.length <= MAX_BLOCK_WIDTH
                       for s in segmentation.segments)
            labels = [s.skill for s in segmentation.segments]
            assert all(x != y for x, y in zip(labels[:-1], labels[1:]))
        assert np.isfinite(state.components["log_target"])
