"""Common random numbers must survive divergence, not merely precede it.

Two arms that share a sequential generator share a prefix. Once one arm accepts a move the
other rejects, or draws one more segment and consumes one more uniform, the streams slip
and every later comparison is confounded by different randomness as well as a different
model. A zero-sweep identical-initial-state check passes happily through that failure.

These tests therefore start *after* the arms have diverged.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_cpa.corpus import generate_ladder_corpus
from hpop.mcmc_cpa.crn import (MOVE_TYPES, CommonRandomNumbers, crn_alignment_report)
from hpop.mcmc_cpa.ladder_runner import (ARMS, LEARNED_ORDER, ORACLE_ORDER,
                                         SUPPORT_ONLY, run_ladder_chain)
from hpop.mcmc_cpa.nested_library import draw_master_library
from hpop.mcmc_original.stage6e_state import Stage6EModel


@pytest.fixture(scope="module")
def rung():
    library, _ = draw_master_library(0)
    corpus = generate_ladder_corpus(library, 3, 0)
    u_by_skill, role_maps = library.prefix(3)
    model = Stage6EModel(traces=corpus.traces("train"), epsilon=0.02, delta_b=0.15,
                         n_skills=3, n_roles=library.n_roles, min_width=3, max_width=12,
                         infer_pi_P=True, eta_initial=1.0, eta_transition=1.0)
    return corpus, model, role_maps, u_by_skill


# ------------------------------------------------- the index is the ONLY thing that matters
def test_a_generator_depends_only_on_its_index():
    a = CommonRandomNumbers(replicate=0, k=10, chain=1)
    b = CommonRandomNumbers(replicate=0, k=10, chain=1)
    # drive `a` hard first: if consumption mattered, `a` would have drifted
    for sweep in range(50):
        a.rng("ffbs", sweep).random(1000)
        a.rng("u", sweep, 3).random(7)
    np.testing.assert_array_equal(a.rng("ffbs", 77).random(16),
                                  b.rng("ffbs", 77).random(16))


def test_alignment_holds_across_every_index_after_heavy_divergent_use():
    a, b = (CommonRandomNumbers(0, 10, 1) for _ in range(2))
    for sweep in range(30):                        # only `a` consumes, and unevenly
        a.rng("ffbs", sweep).random(int(1 + 97 * (sweep % 5)))
    report = crn_alignment_report(a, b, sweeps=20, draws=8)
    assert report["aligned"], report["mismatches"][:5]
    assert report["indices_checked"] == 20 * len(MOVE_TYPES) * 2


def test_distinct_indices_give_distinct_numbers():
    """Sharing is worthless if every index returns the same stream."""
    crn = CommonRandomNumbers(0, 10, 1)
    base = crn.rng("ffbs", 5, 0).random(8)
    for other in (crn.rng("ffbs", 6, 0), crn.rng("ffbs", 5, 1),
                  crn.rng("u", 5, 0), crn.rng("pi_p", 5, 0)):
        assert not np.array_equal(base, other.random(8))


def test_replicate_k_and_chain_all_separate_streams():
    base = CommonRandomNumbers(0, 10, 1).rng("ffbs", 3).random(8)
    for other in (CommonRandomNumbers(1, 10, 1), CommonRandomNumbers(0, 20, 1),
                  CommonRandomNumbers(0, 10, 2)):
        assert not np.array_equal(base, other.rng("ffbs", 3).random(8))


def test_an_unknown_move_type_is_refused():
    with pytest.raises(KeyError):
        CommonRandomNumbers(0, 3, 0).rng("not-a-move", 0)


# --------------------------------------------- the arms really do diverge, and still share
def test_the_arms_diverge_within_the_recorded_sweeps(rung):
    """The premise of every test below. If they did not diverge there would be nothing
    to protect against."""
    _, model, role_maps, u_by_skill = rung
    runs = {arm: run_ladder_chain(arm, model, role_maps, u_by_skill, chain=0, sweeps=10,
                                  warmup=0, seed=11, thin=1, replicate=0)
            for arm in (SUPPORT_ONLY, ORACLE_ORDER)}
    assert runs[SUPPORT_ONLY]["draws"]["labels"] != runs[ORACLE_ORDER]["draws"]["labels"]


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_draws_the_same_numbers_at_the_same_indices(rung, arm):
    """After a full chain, the arm's own CRN must still hand out the shared numbers.

    Run the arm, then ask a fresh CRN with the same coordinates for the numbers at a
    handful of indices; they must match what any other arm would get. This is the check
    that a chain cannot privately advance the shared streams.
    """
    _, model, role_maps, u_by_skill = rung
    used = CommonRandomNumbers(0, model.n_skills, 0)
    run_ladder_chain(arm, model, role_maps, u_by_skill, chain=0, sweeps=8, warmup=0,
                     seed=11, thin=1, crn=used, replicate=0)
    fresh = CommonRandomNumbers(0, model.n_skills, 0)
    report = crn_alignment_report(used, fresh, sweeps=10, draws=6)
    assert report["aligned"], report["mismatches"][:5]


def test_two_arms_share_the_ffbs_stream_at_every_sweep_after_divergence(rung):
    """The headline: sweep by sweep, both arms' FFBS uniforms are the same numbers even
    though by then they are sampling different segmentations."""
    _, model, role_maps, u_by_skill = rung
    crns = {}
    for arm in (SUPPORT_ONLY, ORACLE_ORDER):
        crns[arm] = CommonRandomNumbers(0, model.n_skills, 0)
        run_ladder_chain(arm, model, role_maps, u_by_skill, chain=0, sweeps=12, warmup=0,
                         seed=11, thin=1, crn=crns[arm], replicate=0)
    for sweep in range(12):
        np.testing.assert_array_equal(
            crns[SUPPORT_ONLY].rng("ffbs", sweep).random(12),
            crns[ORACLE_ORDER].rng("ffbs", sweep).random(12),
            err_msg=f"FFBS uniforms differ between arms at sweep {sweep}")


def test_the_learned_arm_moves_u_without_shifting_the_shared_streams(rung):
    """`learned-order` proposes `U` moves the other arms never make. Those extra draws
    must come from their own indexed streams, not from the FFBS stream."""
    _, model, role_maps, u_by_skill = rung
    learned = CommonRandomNumbers(0, model.n_skills, 0)
    result = run_ladder_chain(LEARNED_ORDER, model, role_maps, u_by_skill, chain=0,
                              sweeps=8, warmup=0, seed=11, thin=1, u_moves=2,
                              crn=learned, replicate=0)
    assert result["u_proposed"] > 0, "the learned arm proposed no U moves"
    plain = CommonRandomNumbers(0, model.n_skills, 0)
    run_ladder_chain(SUPPORT_ONLY, model, role_maps, u_by_skill, chain=0, sweeps=8,
                     warmup=0, seed=11, thin=1, crn=plain, replicate=0)
    for sweep in range(8):
        np.testing.assert_array_equal(learned.rng("ffbs", sweep).random(10),
                                      plain.rng("ffbs", sweep).random(10))


# ------------------------------------------------ the residual is measured, not assumed
def test_the_residual_within_sweep_misalignment_is_declared(rung):
    """`ffbs_segmentation_draw` loops over traces against one generator, so traces after
    the first can misalign inside a sweep. That cannot be fixed without editing the sealed
    backend, so it must at least be written down rather than quietly assumed away."""
    provenance = CommonRandomNumbers(0, 3, 0).provenance()
    assert "within ONE sweep" in provenance["residual"]
    assert "sealed backend" in provenance["residual"]
    assert "cannot propagate across sweeps" in provenance["guarantee"]


def test_a_sequential_generator_would_fail_this(rung):
    """The control: a single shared `Generator` slips as soon as consumption differs.
    If this ever stopped failing, the CRN machinery would be unnecessary."""
    a = np.random.default_rng(12345)
    b = np.random.default_rng(12345)
    a.random(3)                     # one arm consumed three extra uniforms
    assert not np.array_equal(a.random(8), b.random(8))
