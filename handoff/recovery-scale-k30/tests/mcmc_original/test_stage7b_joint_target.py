"""Step 7B — the target is Stage 6E's, and the segmentation update is a Gibbs step.

Two claims are pinned here, and they are the ones that would invalidate everything
downstream if they were false: that swapping the kernel did not move the posterior, and
that the swapped-in kernel is a genuine conditional draw rather than a Metropolis move
missing its correction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hpop.mcmc_original import recurrent_joint_ffbs_mcmc as ffbs_module
from hpop.mcmc_original.fast_segmentation_kernel import key_of, segmentation_of
from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import (
    Stage7BSampler, ffbs_segmentation_draw, ffbs_sweep_once,
)
from hpop.mcmc_original.recurrent_segmentation import log_target_stage6e
from hpop.mcmc_original.stage6d_frozen import REGISTERED_SCALES, SCALAR_ORDER
from hpop.mcmc_original.stage6e_frozen import (
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH,
)
from hpop.mcmc_original.stage6e_state import (
    Stage6EModel, Stage6EState, initial_counts, transition_counts_of,
)
from hpop.mcmc_original.transitions import log_transition_matrix

K = 3
M_ROLES = 3
D_LATENT = 2
EPSILON = 0.02
TRACES = ((2, 0, 2, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 2, 0, 0))
PI_FIXED = np.array([0.60, 0.30, 0.10])
P_FIXED = np.array([[0.00, 0.70, 0.30], [0.25, 0.00, 0.75], [0.80, 0.20, 0.00]])


def make_model(infer_pi_P: bool = False) -> Stage6EModel:
    return Stage6EModel(traces=TRACES, epsilon=EPSILON, delta_b=DELTA_B, n_skills=K,
                        n_roles=M_ROLES, min_width=MIN_BLOCK_WIDTH,
                        max_width=MAX_BLOCK_WIDTH, infer_pi_P=infer_pi_P)


def make_state(rng, infer_pi_P: bool = False) -> Stage6EState:
    shapes = [((8, 0),), ((3, 0), (8, 1)), ((4, 2), (8, 0)), ((5, 1), (8, 2))]
    if infer_pi_P:
        pi = rng.dirichlet(np.ones(K))
        transition = np.zeros((K, K))
        for h in range(K):
            allowed = [k for k in range(K) if k != h]
            transition[h, allowed] = rng.dirichlet(np.ones(len(allowed)))
    else:
        pi, transition = PI_FIXED, P_FIXED
    return Stage6EState(
        segmentations=tuple(segmentation_of(shapes[int(rng.integers(len(shapes)))])
                            for _ in TRACES),
        u_by_skill=rng.normal(scale=1.5, size=(K, M_ROLES, D_LATENT)),
        rho=float(rng.uniform(0.05, 0.9)), beta=float(rng.uniform(0.5, 2.5)),
        omega=float(rng.uniform(-1.0, 3.0)), lambda_rep=float(rng.uniform(0.2, 1.5)),
        lambda_back=float(rng.uniform(0.05, 1.0)), pi=pi, transition=transition)


# ------------------------------------------------------------------------ target parity
@pytest.mark.parametrize("infer_pi_P", [False, True])
def test_swept_states_are_scored_by_the_stage6e_target(infer_pi_P):
    model = make_model(infer_pi_P)
    sampler = Stage7BSampler(model=model, scales=REGISTERED_SCALES)
    rng = np.random.default_rng(31)
    worst = 0.0
    for _ in range(12):
        state = make_state(rng, infer_pi_P)
        after = ffbs_sweep_once(state, sampler, rng)
        direct = log_target_stage6e(after, model)
        worst = max(worst, abs(float(direct["log_target"])
                               - float(after.components["log_target"])))
    assert worst < 1e-9


def test_the_target_decomposition_has_every_registered_term():
    model = make_model(infer_pi_P=True)
    sampler = Stage7BSampler(model=model, scales=REGISTERED_SCALES)
    rng = np.random.default_rng(32)
    after = ffbs_sweep_once(make_state(rng, True), sampler, rng)
    for term in ("log_block_likelihood", "log_boundary_prior", "log_initial",
                 "log_transition", "log_structural_prior", "log_rho_prior",
                 "log_pi_prior", "log_P_prior", "log_target"):
        assert term in after.components


# ------------------------------------------------------------- the Gibbs step has no MH
def test_no_hastings_correction_in_the_segmentation_update():
    """The FFBS draw must contain no proposal ratio and no acceptance test."""
    source = Path(ffbs_module.__file__).read_text()
    body = source.split('"""', 2)[-1]                    # drop the module docstring
    draw = body.split("def ffbs_segmentation_draw", 1)[1].split("\ndef ", 1)[0]
    # prose may discuss acceptance; the executable lines may not perform it
    pieces = draw.split('"""')
    code = "".join(pieces[0:1] + pieces[2:])             # drop the function docstring
    code = "\n".join(line.split("#", 1)[0] for line in code.splitlines())
    for forbidden in ("proposal_prob", "log_alpha", "accept", "hastings", "rng.random()",
                      "reverse", "mh_"):
        assert forbidden not in code, (
            f"the segmentation Gibbs draw must not contain {forbidden!r}")
    assert "backward_sample" in code


def test_the_segmentation_update_is_accepted_by_construction():
    model = make_model()
    sampler = Stage7BSampler(model=model, scales=REGISTERED_SCALES)
    rng = np.random.default_rng(33)
    state = make_state(rng)
    after = ffbs_sweep_once(state, sampler, rng)
    assert after.proposed["S_z_ffbs"] == len(TRACES)
    assert after.accepted["S_z_ffbs"] == after.proposed["S_z_ffbs"]


# ----------------------------------------------------------------------- pi and P rules
def test_pi_and_P_updates_see_the_labels_ffbs_has_just_drawn():
    """The conjugate update must be fed the new segmentation, not the previous sweep's."""
    model = make_model(infer_pi_P=True)
    sampler = Stage7BSampler(model=model, scales=REGISTERED_SCALES)
    rng = np.random.default_rng(34)
    state = make_state(rng, True)

    # take the FFBS draw alone, then check the counts the sweep would use
    sampler.tables.refresh(state)
    drawn = ffbs_segmentation_draw(model, state, sampler.tables, np.random.default_rng(5))
    sampler.tables.mark_stale()
    after_segmentations = tuple(segmentation_of(k) for k in drawn["keys"])
    old_counts = transition_counts_of(state.segmentations, K)
    new_counts = transition_counts_of(after_segmentations, K)
    assert not np.array_equal(old_counts, new_counts) or drawn["keys"] == tuple(
        key_of(s) for s in state.segmentations)

    # and the full sweep's sampled P must be a legal draw from the new counts' support
    after = ffbs_sweep_once(state, sampler, np.random.default_rng(6))
    counts = transition_counts_of(after.segmentations, K)
    assert counts.sum() == sum(len(s.segments) - 1 for s in after.segmentations)
    assert np.all(np.diag(after.transition) == 0.0)
    assert np.allclose(after.transition.sum(axis=1), 1.0)
    assert initial_counts(after.segmentations, K).sum() == len(TRACES)


def test_transition_counts_have_no_terminal_transition():
    """`L` segments contribute `L - 1` transitions, never `L`."""
    model = make_model(infer_pi_P=True)
    sampler = Stage7BSampler(model=model, scales=REGISTERED_SCALES)
    rng = np.random.default_rng(35)
    state = make_state(rng, True)
    for _ in range(10):
        state = ffbs_sweep_once(state, sampler, rng)
        counts = transition_counts_of(state.segmentations, K)
        expected = sum(len(s.segments) - 1 for s in state.segmentations)
        assert counts.sum() == expected


def test_the_transition_diagonal_stays_exactly_zero():
    model = make_model(infer_pi_P=True)
    sampler = Stage7BSampler(model=model, scales=REGISTERED_SCALES)
    rng = np.random.default_rng(36)
    state = make_state(rng, True)
    for _ in range(20):
        state = ffbs_sweep_once(state, sampler, rng)
        assert np.all(np.diag(state.transition) == 0.0)
        # and no drawn segmentation may repeat a label across a boundary
        for segmentation in state.segmentations:
            labels = [s.skill for s in segmentation.segments]
            assert all(a != b for a, b in zip(labels[:-1], labels[1:]))


def test_a_forbidden_transition_is_never_drawn():
    """With `P[0, 1] = 0` the sampler must never produce the pair `0 -> 1`."""
    model = make_model()
    transition = np.array([[0.0, 0.0, 1.0], [0.25, 0.0, 0.75], [0.8, 0.2, 0.0]])
    rng = np.random.default_rng(37)
    state = make_state(rng)
    state.transition = transition
    sampler = Stage7BSampler(model=model, scales=REGISTERED_SCALES)
    log_p = log_transition_matrix(transition)
    assert log_p[0, 1] == -np.inf
    for _ in range(30):
        sampler.tables.refresh(state)
        drawn = ffbs_segmentation_draw(model, state, sampler.tables, rng)
        sampler.tables.mark_stale()
        for key in drawn["keys"]:
            labels = [k for _, k in key]
            assert not any(a == 0 and b == 1 for a, b in zip(labels[:-1], labels[1:]))
        state.segmentations = tuple(segmentation_of(k) for k in drawn["keys"])


def test_scalar_order_is_the_frozen_one():
    assert SCALAR_ORDER == ("beta", "omega", "lambda_rep", "lambda_back")
