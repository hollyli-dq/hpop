"""Stage 6E — the direct target, the move kernel, and the recurrent block scorer.

Covers §18 areas 1-16. Everything here is small and exact: no chain is run, and every
assertion is against a quantity computed a second way rather than against a recorded
number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hpop.mcmc_original.fast_segmentation_kernel import (
    FastSegmentationKernel, assert_kernels_agree, assert_proposal_prob_agrees, key_of,
    segmentation_of, spans_of,
)
from hpop.mcmc_original.proposals import MoveType
from hpop.mcmc_original.recurrent_joint_scalar_mcmc import vectorized_state_features
from hpop.mcmc_original.recurrent_latent_poset_mcmc import LatentPosetEvaluator
from hpop.mcmc_original.recurrent_oracle_joint_mcmc import Stage6DTarget
from hpop.mcmc_original.recurrent_scalar_posterior import log_prior
from hpop.mcmc_original.recurrent_segmentation import (
    RecurrentBlockScorer, Stage6EMoveKernel, is_legal_segmentation, log_target_stage6e,
    recurrent_compatible_skills, segmentation_log_weight,
)
from hpop.mcmc_original.stage6c_frozen import log_rho_prior
from hpop.mcmc_original.stage6d_frozen import (
    ACTIVE_6D, REGISTERED_SCALES as REGISTERED_SCALES_FOR_TEST, SCALAR_ORDER,
)
from hpop.mcmc_original.stage6e_block_table import (
    BlockScoreTable, assert_table_matches_scorer,
)
from hpop.mcmc_original.stage6e_frozen import (
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH, N_SKILLS, TERMINAL_TRANSITION,
    assert_stage6d_unchanged, frozen_config, log_boundary_prior_6e,
)
from hpop.mcmc_original.stage6e_sampler import assert_evaluators_agree
from hpop.mcmc_original.stage6e_state import (
    Stage6EModel, Stage6EState, initial_counts, transition_counts_of,
)
from hpop.mcmc_original.transitions import (
    allowed_next, log_transition_matrix, transition_counts,
)
from hpop.mcmc_original.types import Segment, Segmentation

BLOCK = 6
N_TRACES = 3
SEGMENTS = 3
M_ROLES = 5


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(4242)
    length = SEGMENTS * BLOCK
    traces = tuple(tuple(int(v) for v in rng.integers(0, M_ROLES, size=length))
                   for _ in range(N_TRACES))
    u = rng.normal(size=(N_SKILLS, M_ROLES, 2))
    segmentations = tuple(
        Segmentation(tuple(Segment(i * BLOCK, (i + 1) * BLOCK, (i + n) % N_SKILLS)
                           for i in range(SEGMENTS)))
        for n in range(N_TRACES))
    pi = np.full(N_SKILLS, 1.0 / N_SKILLS)
    transition = np.zeros((N_SKILLS, N_SKILLS))
    for h in range(N_SKILLS):
        for k in allowed_next(h, N_SKILLS):
            transition[h, k] = 1.0 / (N_SKILLS - 1)
    model = Stage6EModel(traces=traces)
    state = Stage6EState(segmentations=segmentations, u_by_skill=u, rho=0.35,
                         beta=1.5, omega=1.7346, lambda_rep=0.8, lambda_back=0.25,
                         pi=pi, transition=transition)
    return model, state


# --------------------------------------------------------------- 1. target decomposition
def test_direct_target_decomposition_sums_to_the_total(problem):
    model, state = problem
    parts = log_target_stage6e(state, model)
    required = ("log_block_likelihood", "log_boundary_prior", "log_initial",
                "log_transition", "log_structural_prior", "log_rho_prior",
                "log_scalar_priors", "log_pi_prior", "log_P_prior", "log_target")
    for name in required:
        assert name in parts, name
    total = (parts["log_block_likelihood"] + parts["log_boundary_prior"]
             + parts["log_initial"] + parts["log_transition"]
             + parts["log_structural_prior"] + parts["log_rho_prior"]
             + sum(parts["log_scalar_priors"].values())
             + parts["log_pi_prior"] + parts["log_P_prior"])
    assert parts["log_target"] == pytest.approx(total, abs=1e-9)
    assert set(parts["log_scalar_priors"]) == set(SCALAR_ORDER)


def test_direct_target_calls_no_acceptance_helper(problem):
    """§3: the direct target must not reach an MH helper. Checked on the call graph."""
    import inspect

    import hpop.mcmc_original.recurrent_segmentation as module
    source = inspect.getsource(module.log_target_stage6e)
    for forbidden in ("mh_local_step", "scalar_mh_step", "mh_segmentation_step",
                      "acceptance", "log_alpha"):
        assert forbidden not in source, forbidden


# ------------------------------------------------- 2. oracle parity with Stage 6D
def test_oracle_boundary_parity_with_stage6d_is_one_constant(problem):
    """With (S, z) pinned, 6E - 6D must be the SAME constant at every theta."""
    model, state = problem
    assert_stage6d_unchanged()
    rng = np.random.default_rng(7)

    blocks_by_skill = {k: [] for k in range(model.n_skills)}
    for index, segmentation in enumerate(state.segmentations):
        for segment in segmentation.segments:
            blocks_by_skill[segment.skill].append(
                model.traces[index][segment.start:segment.end])

    log_pi = np.log(state.pi)
    log_transition = log_transition_matrix(state.transition)
    constant = 0.0
    for index, segmentation in enumerate(state.segmentations):
        path = [s.skill for s in segmentation.segments]
        constant += log_boundary_prior_6e(len(model.traces[index]), len(path),
                                          model.delta_b)
        constant += float(log_pi[path[0]])
        for a, b in zip(path[:-1], path[1:]):
            constant += float(log_transition[a, b])

    differences = []
    for trial in range(4):
        trial_state = state.copy()
        if trial:
            trial_state.u_by_skill = state.u_by_skill + rng.normal(
                scale=0.5, size=state.u_by_skill.shape)
            trial_state.rho = float(rng.uniform(0.05, 0.9))
            trial_state.beta = float(rng.uniform(0.6, 2.4))
            trial_state.omega = float(rng.uniform(-1.5, 3.5))
            trial_state.lambda_rep = float(rng.uniform(0.2, 1.8))
            trial_state.lambda_back = float(rng.uniform(0.05, 1.2))
        six_e = log_target_stage6e(trial_state, model)["log_target"]
        values = {n: float(getattr(trial_state, n)) for n in SCALAR_ORDER}
        six_d = 0.0
        for k in range(model.n_skills):
            roles = np.array(blocks_by_skill[k], dtype=int)
            evaluator = LatentPosetEvaluator(roles, epsilon=model.epsilon,
                                             omega=values["omega"])
            target = Stage6DTarget(evaluator, active=ACTIVE_6D)
            parts = target.decompose(trial_state.u_by_skill[k],
                                     {"rho": trial_state.rho, **values},
                                     allow_cache=False)
            six_d += parts["log_likelihood"] + parts["log_structural_prior"]
        six_d += float(sum(log_prior(n, values[n]) for n in SCALAR_ORDER))
        six_d += log_rho_prior(trial_state.rho)
        differences.append(six_e - six_d)

    for value in differences:
        assert value == pytest.approx(constant, abs=1e-9)
    assert max(differences) - min(differences) < 1e-9


# ------------------------------------------- 3-6. kernel parity and reverse probabilities
def test_fast_kernel_matches_the_reference_kernel_exactly(problem):
    model, _ = problem
    trace = model.traces[0]
    fast = FastSegmentationKernel(trace_length=len(trace), n_skills=N_SKILLS)
    reference = Stage6EMoveKernel(x=trace, skills=(), n_skills=N_SKILLS,
                                  min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH)
    from hpop.mcmc_original.stage6e_exact import enumerate_states
    states = enumerate_states(len(trace), N_SKILLS)
    rng = np.random.default_rng(0)
    for index in rng.choice(len(states), size=min(25, len(states)), replace=False):
        report = assert_kernels_agree(fast, reference, states[index])
        assert report["pass"], report


def test_direct_proposal_prob_equals_the_full_distribution(problem):
    model, _ = problem
    fast = FastSegmentationKernel(trace_length=len(model.traces[0]), n_skills=N_SKILLS)
    from hpop.mcmc_original.stage6e_exact import enumerate_states
    states = enumerate_states(len(model.traces[0]), N_SKILLS)
    rng = np.random.default_rng(1)
    for index in rng.choice(len(states), size=min(20, len(states)), replace=False):
        report = assert_proposal_prob_agrees(fast, states[index])
        assert report["pass"], report


@pytest.mark.parametrize("move", [MoveType.SPLIT, MoveType.MERGE, MoveType.SHIFT,
                                  MoveType.RELABEL])
def test_reverse_probabilities_are_strictly_positive_and_not_assumed_symmetric(problem,
                                                                               move):
    """Every legal move must be reversible, and split/merge must NOT be symmetric."""
    model, state = problem
    fast = FastSegmentationKernel(trace_length=len(model.traces[0]), n_skills=N_SKILLS)
    current = key_of(state.segmentations[0])
    neighbours = fast.neighbours(current, move)
    assert neighbours, f"{move} has an empty neighbourhood on the fixture"
    asymmetric = 0
    for candidate in neighbours[:12]:
        forward = fast.proposal_prob(current, candidate)
        reverse = fast.proposal_prob(candidate, current)
        assert forward > 0.0, (move, candidate)
        assert reverse > 0.0, f"{move} to {candidate} is not reversible"
        if abs(math.log(reverse) - math.log(forward)) > 1e-12:
            asymmetric += 1
    if move in (MoveType.SPLIT, MoveType.MERGE):
        assert asymmetric > 0, (f"{move} changes the neighbourhood size, so at least one "
                                "candidate must carry a non-zero Hastings term")


def test_acceptance_ratio_matches_the_registered_construction(problem):
    """log R = [log p(S') - log p(S)] + [log q(S|S') - log q(S'|S)], both routes."""
    model, state = problem
    scorer = model.scorer_for(state)
    log_pi = np.log(state.pi)
    log_transition = log_transition_matrix(state.transition)
    trace = model.traces[0]
    fast = FastSegmentationKernel(trace_length=len(trace), n_skills=N_SKILLS)

    def target(key):
        return segmentation_log_weight(segmentation_of(key), 0, len(trace), scorer,
                                       log_pi, log_transition, DELTA_B)["log_weight"]

    current = key_of(state.segmentations[0])
    checked = 0
    for move in MoveType.ALL:
        for candidate in fast.neighbours(current, move)[:5]:
            forward = fast.proposal_prob(current, candidate)
            reverse = fast.proposal_prob(candidate, current)
            direct = ((target(candidate) - target(current))
                      + math.log(reverse) - math.log(forward))
            law_forward = fast.proposal_distribution(current)[candidate]
            law_reverse = fast.proposal_distribution(candidate)[current]
            second = ((target(candidate) - target(current))
                      + math.log(law_reverse) - math.log(law_forward))
            assert direct == pytest.approx(second, abs=1e-12)
            checked += 1
    assert checked >= 8


# ------------------------------------------------- 7-8. legality: coverage and widths
def test_every_neighbour_covers_the_trace_without_gaps_or_overlaps(problem):
    model, state = problem
    trace_length = len(model.traces[0])
    fast = FastSegmentationKernel(trace_length=trace_length, n_skills=N_SKILLS)
    current = key_of(state.segmentations[0])
    seen = 0
    for move in MoveType.ALL:
        for candidate in fast.neighbours(current, move):
            spans = spans_of(candidate)
            assert spans[0][0] == 0
            assert spans[-1][1] == trace_length
            for left, right in zip(spans[:-1], spans[1:]):
                assert left[1] == right[0], "gap or overlap"
            assert all(b > a for a, b, _ in spans), "non-positive length"
            assert fast.is_legal(candidate)
            # and the Segmentation constructor must accept it too
            assert is_legal_segmentation(segmentation_of(candidate), N_SKILLS,
                                         MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
            seen += 1
    assert seen > 0


def test_maximum_and_minimum_width_are_enforced(problem):
    model, state = problem
    fast = FastSegmentationKernel(trace_length=len(model.traces[0]), n_skills=N_SKILLS)
    current = key_of(state.segmentations[0])
    for move in MoveType.ALL:
        for candidate in fast.neighbours(current, move):
            for a, b, _ in spans_of(candidate):
                assert MIN_BLOCK_WIDTH <= b - a <= MAX_BLOCK_WIDTH
    assert recurrent_compatible_skills(0, MAX_BLOCK_WIDTH + 1, N_SKILLS) == ()
    assert recurrent_compatible_skills(0, MIN_BLOCK_WIDTH - 1, N_SKILLS) == ()
    assert recurrent_compatible_skills(0, MIN_BLOCK_WIDTH, N_SKILLS) == tuple(
        range(N_SKILLS))


def test_no_neighbour_repeats_a_label_across_a_boundary(problem):
    model, state = problem
    fast = FastSegmentationKernel(trace_length=len(model.traces[0]), n_skills=N_SKILLS)
    current = key_of(state.segmentations[0])
    for move in MoveType.ALL:
        for candidate in fast.neighbours(current, move):
            labels = [k for _, k in candidate]
            assert all(a != b for a, b in zip(labels[:-1], labels[1:]))


# --------------------------------------------------------- 9-10. transitions and prior
def test_there_is_no_terminal_transition(problem):
    _, state = problem
    assert TERMINAL_TRANSITION is False
    assert frozen_config()["pi_P"]["terminal_transition"] is False
    paths = [[s.skill for s in seg.segments] for seg in state.segmentations]
    counts = transition_counts_of(state.segmentations, N_SKILLS)
    assert counts.sum() == sum(len(p) - 1 for p in paths)
    assert counts.sum() != sum(len(p) for p in paths)
    assert np.array_equal(counts, transition_counts(paths, N_SKILLS))
    assert all(counts[h, h] == 0.0 for h in range(N_SKILLS))
    assert initial_counts(state.segmentations, N_SKILLS).sum() == len(paths)


@pytest.mark.parametrize("length,segments", [(20, 1), (20, 4), (33, 5), (12, 2)])
def test_boundary_prior_counts_cuts_and_non_cuts(length, segments):
    value = log_boundary_prior_6e(length, segments, DELTA_B)
    expected = ((segments - 1) * math.log(DELTA_B)
                + (length - segments) * math.log(1.0 - DELTA_B))
    assert value == pytest.approx(expected, abs=1e-12)
    # the exponents must add to the J-1 internal positions
    assert (segments - 1) + (length - segments) == length - 1


# ------------------------------------------------------- 11-13. q0, leakage, rescoring
def test_q_zero_resets_for_every_candidate_block(problem):
    model, state = problem
    for start, end in ((0, 6), (3, 9), (6, 18), (7, 12)):
        roles = np.array([model.traces[0][start:end]], dtype=int)
        features = vectorized_state_features(roles, state.u_by_skill[0], state.omega)
        assert np.all(features["q"][:, 0, :] == 0.0)


def test_no_recurrent_state_leaks_between_blocks_traces_or_skills(problem):
    """A block's score must not depend on what was scored before it, ever."""
    model, state = problem
    scorer = model.scorer_for(state)
    target = scorer.replay(0, 6, 12, 1)
    for other in ((0, 0, 6, 0), (1, 3, 9, 2), (2, 0, 12, 0), (0, 12, 18, 2)):
        scorer.replay(*other)
        assert scorer.replay(0, 6, 12, 1) == target
    # a block scored inside a longer trace equals the same block scored alone
    isolated = RecurrentBlockScorer(
        traces=(model.traces[0][6:12],), epsilon=model.epsilon,
        u_by_skill=state.u_by_skill, beta=state.beta, omega=state.omega,
        lambda_rep=state.lambda_rep, lambda_back=state.lambda_back)
    assert isolated.replay(0, 0, 6, 1) == pytest.approx(target, abs=1e-12)


def test_a_boundary_move_rescores_both_changed_blocks_from_zero(problem):
    model, state = problem
    scorer = model.scorer_for(state)
    # shift the first boundary of trace 0 from 6 to 7: blocks [0,7) and [7,12) are new
    before = key_of(state.segmentations[0])
    after = ((7, before[0][1]), (12, before[1][1]), before[2])
    changed = [(0, 7, before[0][1]), (7, 12, before[1][1])]
    for start, end, skill in changed:
        cached = scorer.score(0, start, end, skill)
        fresh = scorer.replay(0, start, end, skill)
        assert cached == pytest.approx(fresh, abs=1e-12)
    # the unchanged third block keeps its score
    assert scorer.score(0, 12, 18, before[2][1]) == pytest.approx(
        scorer.replay(0, 12, 18, before[2][1]), abs=1e-12)
    assert after != before


# --------------------------------------------------- 14-16. cache invalidation and safety
def test_a_u_change_bumps_the_version_and_clears_the_cache(problem):
    model, state = problem
    scorer = model.scorer_for(state)
    scorer.score(0, 0, 6, 0)
    version, size = scorer.version, scorer.cache_size
    assert size > 0
    scorer.set_skill_u(0, state.u_by_skill[0] + 0.1)
    assert scorer.version == version + 1
    assert scorer.cache_size == 0


@pytest.mark.parametrize("name", ["beta", "omega", "lambda_rep", "lambda_back"])
def test_every_scalar_change_invalidates_every_cached_score(problem, name):
    model, state = problem
    scorer = model.scorer_for(state)
    scorer.score(0, 0, 6, 0)
    scorer.score(1, 6, 12, 1)
    version = scorer.version
    assert scorer.cache_size == 2
    scorer.set_parameters(**{name: float(getattr(scorer, name)) + 0.05})
    assert scorer.version == version + 1
    assert scorer.cache_size == 0


def test_a_rejected_proposal_leaves_the_accepted_cache_untouched(problem):
    model, state = problem
    scorer = model.scorer_for(state)
    accepted = {(0, 0, 6, 0): scorer.score(0, 0, 6, 0),
                (0, 6, 12, 1): scorer.score(0, 6, 12, 1)}
    version = scorer.version
    # score a pile of candidates as a rejected proposal would
    for start in range(0, 12):
        for skill in range(N_SKILLS):
            if scorer.width_is_legal(start, start + 4):
                scorer.score(0, start, start + 4, skill)
    assert scorer.version == version, "an evaluation must never bump the version"
    for (trace, start, end, skill), value in accepted.items():
        assert scorer.score(trace, start, end, skill) == value


def test_the_block_table_equals_the_registered_scorer(problem):
    model, state = problem
    scorer = model.scorer_for(state)
    table = BlockScoreTable(traces=model.traces, epsilon=model.epsilon,
                            n_skills=N_SKILLS, min_width=MIN_BLOCK_WIDTH,
                            max_width=MAX_BLOCK_WIDTH)
    table.refresh(state.u_by_skill, state.beta, state.omega, state.lambda_rep,
                  state.lambda_back)
    report = assert_table_matches_scorer(table, scorer, limit=900)
    assert report["pass"], report


def test_the_grouped_and_per_block_evaluators_agree(problem):
    model, state = problem
    report = assert_evaluators_agree(model, state)
    assert report["pass"], report


def test_min_of_zero_and_nan_is_zero_which_is_why_nan_must_be_mapped_to_minus_inf():
    """The trap this guard exists for, stated as an executable fact.

    `scalar_mh_step` accepts when `log(u) < min(0.0, log_alpha)`. Every comparison with
    NaN is False, so `min` keeps its first argument and `min(0.0, NaN)` is `0.0` — an
    automatic acceptance. A NaN log posterior must therefore be turned into `-inf`
    *before* it reaches the step, which is what `sweep_once` does.
    """
    assert min(0.0, float("nan")) == 0.0
    assert not (float("nan") < 0.0)
    assert not (float("nan") >= 0.0)


@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_a_non_finite_likelihood_is_rejected_rather_than_accepted():
    """At a large enough lambda_back the frozen likelihood underflows to NaN.

    The registered one-step probabilities divide by `weights.sum()`; when every feasible
    role's exponent underflows and the arg-max role has zero feasibility, that sum is
    exactly zero. The chain must reject such a proposal and keep a finite target, not
    accept it and carry NaN forever.
    """
    from hpop.mcmc_original.recurrent_scalar_posterior import cached_batch_log_likelihood
    from hpop.mcmc_original.stage6e_sampler import (
        SkillBlockLikelihood, Stage6ESampler, sweep_once,
    )

    rng = np.random.default_rng(11)
    traces = tuple(tuple(int(v) for v in rng.integers(0, M_ROLES, size=18))
                   for _ in range(3))
    model = Stage6EModel(traces=traces)
    state = Stage6EState(
        segmentations=tuple(
            Segmentation((Segment(0, 6, 0), Segment(6, 12, 1), Segment(12, 18, 2)))
            for _ in traces),
        u_by_skill=rng.normal(size=(N_SKILLS, M_ROLES, 2)), rho=0.4,
        beta=1.5, omega=1.7346, lambda_rep=0.8, lambda_back=0.25,
        pi=np.full(N_SKILLS, 1.0 / N_SKILLS),
        transition=np.where(np.eye(N_SKILLS, dtype=bool), 0.0, 1.0 / (N_SKILLS - 1)))

    # the hazard is real: a large enough lambda_back makes the frozen likelihood NaN
    grouped = SkillBlockLikelihood(traces=traces, epsilon=model.epsilon)
    grouped.set_blocks(state.segmentations, N_SKILLS)
    extreme = grouped.full_replay(0, state.u_by_skill[0], 1.5, 1.7346, 0.8, 1e6)
    assert math.isnan(extreme) or extreme == -math.inf, (
        "this test's premise is that an extreme lambda_back is not representable; if the "
        "frozen likelihood has become robust to it, the guard may be revisited")
    del cached_batch_log_likelihood

    # with enormous scalar scales the chain still keeps a finite target throughout
    scales = dict(REGISTERED_SCALES_FOR_TEST)
    scales.update({"beta": 8.0, "omega": 12.0, "lambda_rep": 8.0, "lambda_back": 12.0})
    sampler = Stage6ESampler(model=Stage6EModel(traces=traces), scales=scales,
                             n_proposals_per_trace=2, use_block_table=True)
    current = state.copy()
    walker = np.random.default_rng(4)
    for _ in range(40):
        current = sweep_once(current, sampler, walker)
        assert math.isfinite(current.components["log_target"]), (
            "a non-finite proposal was accepted")
        for name in SCALAR_ORDER:
            assert math.isfinite(getattr(current, name))


def test_the_block_table_is_only_valid_at_the_parameters_it_was_refreshed_at(problem):
    """The table is a snapshot, and the sweep moves past it. Both halves are pinned.

    Within a sweep the registered order updates `(S, z)` first, so the table built at the
    start of the sweep is exact for the whole segmentation phase. After the sweep's `U`
    and scalar updates it is stale — by design — and reading it then would score the
    previous sweep's model. The staleness is not a defect to be papered over; it is why
    `Stage6ESampler.prepare` refreshes on every sweep.
    """
    from hpop.mcmc_original.stage6e_sampler import Stage6ESampler, sweep_once
    model, state = problem
    fresh = Stage6EModel(traces=model.traces)
    sampler = Stage6ESampler(model=fresh, scales=dict(REGISTERED_SCALES_FOR_TEST),
                             n_proposals_per_trace=2, use_block_table=True)
    rng = np.random.default_rng(23)
    moved = sweep_once(state.copy(), sampler, rng)
    scorer = fresh.scorer_for(moved)

    # refreshed at the same parameters: exact agreement
    sampler._table.refresh(moved.u_by_skill, moved.beta, moved.omega, moved.lambda_rep,
                           moved.lambda_back)
    assert assert_table_matches_scorer(sampler._table, scorer, limit=600)["pass"]

    # a parameter change without a refresh must make the table stale, not silently right
    scorer.set_parameters(beta=float(moved.beta) + 0.5)
    stale = assert_table_matches_scorer(sampler._table, scorer, limit=600)
    assert not stale["pass"], (
        "the table appeared to agree with a scorer at different parameters, which would "
        "mean the refresh is not actually load-bearing")


@pytest.mark.parametrize("use_block_table", [False, True])
def test_the_sweeps_tracked_log_target_equals_the_direct_target(problem,
                                                                 use_block_table):
    """The sampler carries the target incrementally; it must not drift from the truth.

    `sweep_once` accumulates the likelihood through accepted moves rather than
    recomputing it, which is what makes it affordable. If that bookkeeping were wrong the
    chain would still *run* — it would simply target something else — so the accumulated
    value is checked against the independently computed `log_target_stage6e` after every
    sweep, with the block table both on and off.
    """
    from hpop.mcmc_original.stage6e_sampler import Stage6ESampler, sweep_once
    model, state = problem
    fresh = Stage6EModel(traces=model.traces)
    sampler = Stage6ESampler(model=fresh, scales=dict(REGISTERED_SCALES_FOR_TEST),
                             n_proposals_per_trace=2, use_block_table=use_block_table)
    rng = np.random.default_rng(17)
    current = state.copy()
    for _ in range(6):
        current = sweep_once(current, sampler, rng)
        direct = log_target_stage6e(current, fresh)["log_target"]
        assert current.components["log_target"] == pytest.approx(direct, abs=1e-7), (
            "the incrementally tracked target has drifted from the direct target")


def test_cached_score_equals_a_fresh_uncached_replay(problem):
    model, state = problem
    scorer = model.scorer_for(state)
    for index, segmentation in enumerate(state.segmentations):
        for segment in segmentation.segments:
            assert scorer.score(index, segment.start, segment.end,
                                segment.skill) == pytest.approx(
                scorer.replay(index, segment.start, segment.end, segment.skill),
                abs=1e-12)


def test_block_scores_are_order_invariant(problem):
    model, state = problem
    scorer = model.scorer_for(state)
    first = scorer.score(0, 0, 6, 0)
    scorer.score(1, 6, 12, 1)
    scorer.score(2, 0, 12, 2)
    assert scorer.score(0, 0, 6, 0) == first


def test_an_illegal_state_has_minus_infinite_weight(problem):
    model, state = problem
    scorer = model.scorer_for(state)
    log_pi = np.log(state.pi)
    log_transition = log_transition_matrix(state.transition)
    repeated = Segmentation((Segment(0, 6, 0), Segment(6, 12, 0), Segment(12, 18, 1)))
    parts = segmentation_log_weight(repeated, 0, 18, scorer, log_pi, log_transition,
                                    DELTA_B)
    assert parts["log_weight"] == -math.inf
    too_wide = Segmentation((Segment(0, 18, 0),))
    assert segmentation_log_weight(too_wide, 0, 18, scorer, log_pi, log_transition,
                                   DELTA_B)["log_weight"] == -math.inf


def test_state_serialises_and_resumes_deterministically(problem):
    import json
    model, state = problem
    rng = np.random.default_rng(5)
    state = state.copy()
    state.rng_state = rng.bit_generator.state
    restored = Stage6EState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert restored.segmentations == state.segmentations
    assert np.array_equal(restored.u_by_skill, state.u_by_skill)
    assert np.array_equal(restored.pi, state.pi)
    assert np.array_equal(restored.transition, state.transition)
    assert all(getattr(restored, n) == getattr(state, n) for n in SCALAR_ORDER)
    a = np.random.default_rng(0); a.bit_generator.state = state.rng_state
    b = np.random.default_rng(0); b.bit_generator.state = restored.rng_state
    assert np.array_equal(a.random(32), b.random(32))
    assert model.n_skills == N_SKILLS
