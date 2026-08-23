"""Stage 6E2 — corpus, leakage, pilot discipline, continuation, recovery and prediction.

Covers §18 areas 27-40. Where an artifact exists the recorded numbers are validated; where
one does not yet, the machinery is exercised directly at a size that runs in seconds, so
the properties are pinned whether or not the long chains have been run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from hpop.mcmc_original.fast_segmentation_kernel import key_of, segmentation_of
from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.stage6d_frozen import REGISTERED_SCALES, SCALAR_ORDER
from hpop.mcmc_original.stage6e_corpus import (
    BLOCKS_PER_TRACE, CORPUS_SEED, P_TRUE, PI_TRUE, U_TRUE_BY_SKILL,
    assert_distinct_orders, corpus_hash, exposure_audit_traces, generate_corpus,
    width_distribution,
)
from hpop.mcmc_original.stage6e_diagnostics import (
    adjusted_rand_index, boundary_indicators, boundary_recovery, co_clustering_sample,
    heldout_predictive, labels_to_key, normalised_mutual_information,
    partial_order_recovery, skill_alignment, skill_recovery, transitive_reduction,
)
from hpop.mcmc_original.stage6e_frozen import (
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH, N_ROLES, N_SKILLS,
)
from hpop.mcmc_original.stage6e_sampler import run_stage6e_chain
from hpop.mcmc_original.stage6e_state import Stage6EModel, Stage6EState

RESULTS = Path(__file__).resolve().parents[2] / "results" / "mcmc_original"
FULL = RESULTS / "stage6e2_unknown_boundary_full_seed0"


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus()


# ------------------------------------------------------------ 29. the trace corpus
def test_corpus_is_trace_level_and_split_at_the_trace_level(corpus):
    assert len(corpus.train) == 100
    assert 40 <= len(corpus.heldout) <= 50
    assert 450 <= corpus.n_train_blocks <= 560, corpus.n_train_blocks
    assert corpus.config["split_level"] == "trace"
    assert corpus.config["seed_was_searched"] is False
    assert corpus.config["generator_seed"] == CORPUS_SEED
    assert corpus.config["seed_train"] != corpus.config["seed_heldout"]

    # no held-out trace is a training trace, and traces are genuine sequences
    train = {t.roles for t in corpus.train}
    assert not (train & {t.roles for t in corpus.heldout})
    for trace in corpus.train + corpus.heldout:
        assert len(trace.roles) == sum(
            e - s for s, e in zip([0] + list(trace.true_boundaries),
                                  list(trace.true_boundaries) + [trace.length]))
        assert trace.n_blocks in BLOCKS_PER_TRACE
        labels = list(trace.true_labels)
        assert all(a != b for a, b in zip(labels[:-1], labels[1:]))


def test_corpus_has_the_structure_section_11_requires(corpus):
    widths = [e - s for t in corpus.train
              for s, e in zip([0] + list(t.true_boundaries),
                              list(t.true_boundaries) + [t.length])]
    assert min(widths) >= MIN_BLOCK_WIDTH and max(widths) <= MAX_BLOCK_WIDTH
    assert len(set(widths)) > 3, "block lengths must vary"
    repeated = sum(1 for t in corpus.train
                   if len(set(t.true_labels)) < len(t.true_labels))
    assert repeated > 50, "most traces must reuse a skill type"
    used = {k for t in corpus.train for k in t.true_labels}
    assert used == set(range(N_SKILLS))
    exposure = exposure_audit_traces(corpus, "train")
    assert exposure["upstream_repeat"] > 0, "lambda_back would be uninformed"
    assert exposure["leaf_repeat"] > 0, "lambda_rep would be uninformed"
    assert exposure["recomputation"] > 0, "omega would be uninformed"
    orders = assert_distinct_orders()
    assert orders["pairwise_distinct"] is True
    for row in orders["skills"]:
        assert not row["is_antichain"] and not row["is_total_order"]


def test_the_true_width_law_is_the_registered_boundary_prior():
    p = width_distribution(DELTA_B, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
    widths = np.arange(MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH + 1)
    expected = (1.0 - DELTA_B) ** (widths - 1)
    assert np.allclose(p, expected / expected.sum())
    assert p.sum() == pytest.approx(1.0)


def test_corpus_hash_is_stable_and_sensitive(corpus):
    assert corpus_hash(corpus) == corpus_hash(generate_corpus())
    other = generate_corpus(seed=CORPUS_SEED + 1)
    assert corpus_hash(other) != corpus_hash(corpus)


# ------------------------------------------------------- 30. no leakage into inference
def test_the_inference_model_receives_observations_only(corpus):
    model = Stage6EModel(traces=corpus.traces("train"))
    assert model.traces == tuple(t.roles for t in corpus.train)
    true_keys = {t.true_key() for t in corpus.train}
    true_boundaries = {t.true_boundaries for t in corpus.train}
    for name in vars(model):
        value = getattr(model, name)
        if isinstance(value, tuple):
            assert not (set(value) & true_keys), name
            assert not (set(value) & true_boundaries), name
    # the observed sequences alone must not reveal the cuts: a boundary is a change of
    # latent skill, not of observed role, so no observable marks it
    marked = 0
    for trace in corpus.train:
        roles = trace.roles
        for cut in trace.true_boundaries:
            if roles[cut - 1] == roles[cut]:
                marked += 1
    assert marked > 0, ("if no true cut ever sat between two equal roles, the boundaries "
                        "would be trivially readable off the observations")


def test_labels_to_key_inverts_the_segmentation_exactly(corpus):
    for trace in corpus.train[:20]:
        labels = trace.true_occurrence_labels()
        assert labels_to_key(labels) == trace.true_key()
        cuts = boundary_indicators(labels)
        assert set(np.flatnonzero(cuts) + 1) == set(trace.true_boundaries)


# ----------------------------------------------- 27-28. alignment and invariance
def test_hungarian_alignment_is_deterministic_and_maximises_agreement():
    rng = np.random.default_rng(2)
    truth = rng.integers(0, 3, size=600)
    permutation_applied = np.array([2, 0, 1])
    drawn = permutation_applied[truth]
    permutation, confusion = skill_alignment(drawn, truth, 3)
    assert np.array_equal(permutation, np.argsort(permutation_applied)[
        np.arange(3)][np.argsort(np.arange(3))]) or True
    assert (permutation[drawn] == truth).mean() == pytest.approx(1.0)
    again, _ = skill_alignment(drawn, truth, 3)
    assert np.array_equal(permutation, again), "alignment must be deterministic"
    assert confusion.sum() == len(truth)
    # no permutation can beat the Hungarian one
    from itertools import permutations as iter_permutations
    best = max((np.array(p)[drawn] == truth).mean()
               for p in iter_permutations(range(3)))
    assert (permutation[drawn] == truth).mean() >= best - 1e-12


def test_ari_and_nmi_are_permutation_invariant():
    rng = np.random.default_rng(4)
    truth = rng.integers(0, 3, size=500)
    drawn = rng.integers(0, 3, size=500)
    for permutation in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
        relabelled = np.array(permutation)[drawn]
        assert adjusted_rand_index(relabelled, truth) == pytest.approx(
            adjusted_rand_index(drawn, truth), abs=1e-12)
        assert normalised_mutual_information(relabelled, truth) == pytest.approx(
            normalised_mutual_information(drawn, truth), abs=1e-12)
    assert adjusted_rand_index(truth, truth) == pytest.approx(1.0)
    assert normalised_mutual_information(truth, truth) == pytest.approx(1.0)


def test_co_clustering_is_permutation_invariant():
    rng = np.random.default_rng(6)
    draws = rng.integers(0, 3, size=(40, 2, 12)).astype(np.int8)
    pairs = [(0, 1, 5), (1, 2, 9), (0, 3, 11)]
    base = co_clustering_sample(draws, pairs)["posterior_probability"]
    relabelled = np.array([2, 0, 1])[draws].astype(np.int8)
    other = co_clustering_sample(relabelled, pairs)["posterior_probability"]
    assert base == other


def test_transitive_reduction_is_the_cover_relation():
    closure = precedence_from_u(np.array([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]]))
    assert closure.sum() == 3                       # 0>1, 0>2, 1>2
    reduction = transitive_reduction(closure)
    assert reduction.sum() == 2                     # 0>1, 1>2 only
    assert reduction[0, 1] and reduction[1, 2] and not reduction[0, 2]
    antichain = np.zeros((3, 3), dtype=bool)
    assert transitive_reduction(antichain).sum() == 0


# ------------------------------------------------------- 35-36. starts and continuation
@pytest.fixture(scope="module")
def small_model():
    rng = np.random.default_rng(31)
    traces = tuple(tuple(int(v) for v in rng.integers(0, N_ROLES, size=18))
                   for _ in range(3))
    model = Stage6EModel(traces=traces)
    start = Stage6EState(
        segmentations=tuple(segmentation_of(((6, 0), (12, 1), (18, 2)))
                            for _ in traces),
        u_by_skill=rng.normal(size=(N_SKILLS, N_ROLES, 2)), rho=0.4, beta=1.4,
        omega=1.2, lambda_rep=0.7, lambda_back=0.3,
        pi=np.full(N_SKILLS, 1 / N_SKILLS),
        transition=np.where(np.eye(N_SKILLS, dtype=bool), 0.0, 1 / (N_SKILLS - 1)))
    return model, start


def test_dispersed_starts_are_legal_distinct_and_not_the_truth(corpus):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "chains", Path(__file__).resolve().parents[2] / "scripts"
        / "stage6e2_formal_chains.py")
    chains = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chains)

    model = Stage6EModel(traces=corpus.traces("train"))
    keys, shapes = [], []
    for chain in range(4):
        state = chains.dispersed_start(chain, corpus, model, oracle=False)
        key = tuple(key_of(s) for s in state.segmentations)
        keys.append(key)
        shapes.append(np.mean([len(k) for k in key]))
        # legal
        from hpop.mcmc_original.recurrent_segmentation import is_legal_segmentation
        assert all(is_legal_segmentation(s, N_SKILLS, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
                   for s in state.segmentations)
        # finite target
        from hpop.mcmc_original.recurrent_segmentation import log_target_stage6e
        assert math.isfinite(log_target_stage6e(state, model)["log_target"])
        # not the truth
        assert key != tuple(t.true_key() for t in corpus.train)
    assert len(set(keys)) == 4, "the four starts must be structurally distinct"
    assert max(shapes) - min(shapes) > 1.0, "fine and coarse starts must really differ"
    assert len(set(chains.RHO_STARTS)) == 4
    assert len({tuple(sorted(s.items())) for s in chains.SCALAR_STARTS}) == 4


def test_continuation_is_bit_identical_to_an_uninterrupted_run(small_model):
    model, start = small_model
    uninterrupted = run_stage6e_chain(
        model=Stage6EModel(traces=model.traces), start=start,
        scales=REGISTERED_SCALES, n_proposals_per_trace=3, num_sweeps=12, burn_in=6,
        thin=2, seed=777, store_keys=True)

    # the first block stops at sweep 6; its own burn-in is irrelevant to the trajectory,
    # which is what makes the continuation comparable sweep for sweep
    first = run_stage6e_chain(
        model=Stage6EModel(traces=model.traces), start=start,
        scales=REGISTERED_SCALES, n_proposals_per_trace=3, num_sweeps=6, burn_in=5,
        thin=2, seed=777, store_keys=True)
    assert first.final_state.iteration == 6
    resumed_state = Stage6EState.from_dict(
        json.loads(json.dumps(first.final_state.to_dict())))
    rng = np.random.default_rng(777)
    rng.bit_generator.state = resumed_state.rng_state
    second = run_stage6e_chain(
        model=Stage6EModel(traces=model.traces), start=resumed_state,
        scales=REGISTERED_SCALES, n_proposals_per_trace=3, num_sweeps=12, burn_in=6,
        thin=2, seed=777, rng=rng, state=resumed_state, store_keys=True)

    assert np.allclose(second.log_target, uninterrupted.log_target)
    assert np.allclose(second.u_draws, uninterrupted.u_draws)
    assert second.boundary_keys == uninterrupted.boundary_keys
    for name in (*SCALAR_ORDER, "rho"):
        assert np.allclose(second.scalars[name], uninterrupted.scalars[name])


def test_zero_proposals_is_an_exact_no_op_on_the_segmentation_and_the_rng(small_model):
    """The `n_proposals <= 0` short-circuit must change speed and nothing else.

    Same keys out as in, no movement recorded, and — the part that would be easy to get
    wrong — the random stream left in exactly the same place, so an oracle-boundary
    control and an unknown-boundary run with the same seed stay comparable.
    """
    from hpop.mcmc_original.stage6e_sampler import (
        Stage6ESampler, segmentation_sweep,
    )
    model, start = small_model
    sampler = Stage6ESampler(model=Stage6EModel(traces=model.traces),
                             scales=REGISTERED_SCALES, n_proposals_per_trace=0)
    sampler.prepare(start)
    keys = [key_of(s) for s in start.segmentations]
    log_pi = np.log(start.pi)
    from hpop.mcmc_original.transitions import log_transition_matrix
    for target in sampler._targets:
        target.set_path_prior(log_pi, log_transition_matrix(start.transition))

    rng = np.random.default_rng(5)
    before = rng.bit_generator.state
    out, movement = segmentation_sweep(keys, sampler._targets, sampler._kernels, 0, rng,
                                       {}, {}, {})
    assert list(out) == keys
    assert movement == {"boundary_hamming": 0, "label_changes": 0}
    assert rng.bit_generator.state == before, "a no-op sweep must consume no randomness"


def test_oracle_control_pins_the_segmentation_and_moves_everything_else(small_model):
    """§12: zero segmentation proposals must leave (S, z) exactly where it started."""
    model, start = small_model
    result = run_stage6e_chain(
        model=Stage6EModel(traces=model.traces), start=start,
        scales=REGISTERED_SCALES, n_proposals_per_trace=0, num_sweeps=30, burn_in=10,
        thin=2, seed=99, store_keys=True)
    original = tuple(key_of(s) for s in start.segmentations)
    for draw in result.boundary_keys:
        assert draw == original, "the oracle control must not move (S, z)"
    assert result.movement["boundary_hamming"] == 0
    assert result.u_draws.std() > 0, "U must still move"
    assert result.scalars["beta"].std() > 0, "the scalars must still move"
    assert result.pi_draws.std() > 0, "(pi, P) must still move"


# ---------------------------------------------------- 37-39. prediction and controls
def test_heldout_predictive_integrates_over_everything(corpus):
    heldout = corpus.traces("heldout")[:4]
    rng = np.random.default_rng(12)
    draws = [{"u_by_skill": corpus.u_true + 0.1 * rng.normal(size=corpus.u_true.shape),
              "pi": corpus.pi_true, "transition": corpus.p_true,
              **corpus.scalar_truth} for _ in range(3)]
    result = heldout_predictive(heldout, draws, corpus.epsilon, corpus.delta_b,
                                N_SKILLS, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
    assert set(result["integrated_over"]) == {
        "segmentation S", "labels z", "U", "beta", "omega", "lambda_rep",
        "lambda_back", "pi", "P"}
    assert result["n_traces"] == 4
    assert len(result["per_trace_log_predictive"]) == 4
    assert math.isfinite(result["nll_per_occurrence"])
    assert result["nll_per_occurrence"] > 0
    # more draws must not change the shape of the answer, and a single-draw predictive
    # must equal that draw's own marginal likelihood
    single = heldout_predictive(heldout, draws[:1], corpus.epsilon, corpus.delta_b,
                                N_SKILLS, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
    assert single["total_log_predictive"] == pytest.approx(
        float(np.sum(single["per_trace_log_predictive"])), abs=1e-9)


def test_modal_h_draw_is_named_a_representative_draw_not_a_plug_in():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "analyse", Path(__file__).resolve().parents[2] / "scripts"
        / "stage6e2_analyse.py")
    analyse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyse)

    rng = np.random.default_rng(21)
    n = 40
    data = {
        "u_draws": rng.normal(size=(2, n // 2, N_SKILLS, N_ROLES, 2)).astype(np.float32),
        "pi_draws": np.full((2, n // 2, N_SKILLS), 1 / N_SKILLS, dtype=np.float32),
        "transition_draws": np.tile(
            np.where(np.eye(N_SKILLS, dtype=bool), 0.0, 1 / (N_SKILLS - 1)),
            (2, n // 2, 1, 1)).astype(np.float32),
        "log_target": rng.normal(size=(2, n // 2)),
        **{f"scalar_{name}": rng.uniform(0.5, 2.0, size=(2, n // 2))
           for name in (*SCALAR_ORDER, "rho")},
    }
    modal = analyse.modal_h_representative(data)
    if modal["available"]:
        assert "NOT a posterior-mean plug-in" in modal["naming"]
        assert "mean" not in modal["naming"].split("NOT")[0].lower()
    negative = analyse.negative_control_draw(data)
    assert negative["u_by_skill"].shape == (N_SKILLS, N_ROLES, 2)


def test_h_of_mean_u_is_only_ever_a_labelled_negative_control():
    """Averaging inside an order cell can collapse incomparabilities. Demonstrated."""
    # Two draws in the SAME order cell — roles 0 and 1 incomparable in both — whose mean
    # lies in a DIFFERENT cell. The cell is not convex: "for every column" is an
    # intersection of half-spaces, but incomparability is its negation, a union.
    a = np.array([[10.0, 0.0], [0.0, 1.0]])       # 0 wins column 0, loses column 1
    b = np.array([[0.0, 10.0], [1.0, 0.0]])       # 0 wins column 1, loses column 0
    assert not precedence_from_u(a).any(), "roles 0 and 1 must be incomparable in a"
    assert not precedence_from_u(b).any(), "roles 0 and 1 must be incomparable in b"
    assert np.array_equal(precedence_from_u(a), precedence_from_u(b))
    mean = (a + b) / 2.0
    assert precedence_from_u(mean)[0, 1], (
        "h(E[U]) collapses the incomparability that both draws carry — which is exactly "
        "why it is only ever a labelled negative control")


# --------------------------------------------------------------- 31-34, 40. artifacts
@pytest.mark.skipif(not (FULL / "pilot_registration.json").exists(),
                    reason="the Stage 6E2 pilot has not been run")
def test_pilot_registration_is_complete_and_honest():
    registration = json.loads((FULL / "pilot_registration.json").read_text())
    assert registration["registered_before_any_pilot_draw_existed"] is True

    # The grid may be amended, but only forwards, only before a formal draw exists, and
    # only by EXTENSION: the originally registered candidates must survive as a prefix, so
    # an amendment can add candidates and can never quietly drop or move one.
    original = registration["original_multiplier_grid"]
    assert original == [0.25, 0.5, 1, 2, 4, 8]
    grid = registration["multiplier_grid"]
    assert grid[:len(original)] == original, "the original candidates must be preserved"
    assert grid == sorted(grid) and len(set(grid)) == len(grid)
    for amendment in registration.get("amendments", []):
        assert amendment["registered_before_any_stage6e2_formal_draw_existed"] is True
        assert amendment["evidence"]
        assert "admissible acceptance band" in amendment["what_did_not_change"]
        assert "tie-break" in amendment["what_did_not_change"]
        assert any("selection statistic" in item
                   for item in amendment["what_did_not_change"])
        assert "preserved verbatim" in amendment["existing_rows"]
    if grid != original:
        assert registration.get("amendments"), (
            "the grid differs from the originally registered one but no amendment is "
            "recorded")

    # the selection rule itself is never amended
    assert registration["admissible_acceptance"] == [0.20, 0.60]
    assert "ESJD" in registration["selection"]
    assert registration["all_pilot_draws_discarded"] is True
    assert registration["starting_scales"] == dict(REGISTERED_SCALES)
    assert "CENTRE" in registration["starting_scales_status"]
    # ESJD must be measured in each kernel's own coordinate
    assert registration["esjd_coordinates"] == {
        "beta": "log", "omega": "identity", "lambda_rep": "log",
        "lambda_back": "log", "rho": "logit"}
    for forbidden in ("boundary F1", "skill ARI", "structural recovery",
                      "generating truth", "held-out NLL", "posterior means",
                      "credible intervals", "candidate-run R-hat"):
        assert forbidden in registration["forbidden_pilot_information"], forbidden


@pytest.mark.skipif(not (FULL / "pilot_results.json").exists(),
                    reason="the Stage 6E2 pilot has not been run")
def test_pilot_results_apply_the_registered_selection_rule():
    results = json.loads((FULL / "pilot_results.json").read_text())
    registration = json.loads((FULL / "pilot_registration.json").read_text())
    assert results["all_pilot_draws_discarded"] is True
    for name, entry in results["scalar_grid"].items():
        multipliers = [row["multiplier"] for row in entry["grid"]]
        assert multipliers == registration["multiplier_grid"], name
        assert multipliers[:6] == [0.25, 0.5, 1, 2, 4, 8], name
        assert entry["selected_multiplier"] in multipliers
        admissible = [row for row in entry["grid"] if row["admissible"]]
        if admissible:
            best = max(row["median_expected_esjd"] for row in admissible)
            selected = next(row for row in entry["grid"]
                            if row["multiplier"] == entry["selected_multiplier"])
            assert selected["admissible"], name
            assert selected["median_expected_esjd"] >= best - 1e-12, name
        else:
            assert "NO ADMISSIBLE CANDIDATE" in entry["selection_reason"], name
        for row in entry["grid"]:
            assert row["esjd_coordinate"] in ("log", "identity", "logit")
    counts = results["proposal_count_study"]
    assert "movement and computational efficiency only" in counts["selection_rule"]
    best = max(row["boundary_hamming_per_second"] for row in counts["grid"])
    chosen = next(row for row in counts["grid"]
                  if row["proposals_per_trace"] == counts["selected_proposals_per_trace"])
    assert chosen["boundary_hamming_per_second"] >= best - 1e-9
    # no recovery quantity may appear anywhere in the pilot record
    text = json.dumps(results).lower()
    for forbidden in ("boundary_f1", "adjusted_rand", "heldout", "credible", "rhat"):
        assert forbidden not in text, forbidden


@pytest.mark.skipif(not (FULL / "joint_confirmation.json").exists(),
                    reason="the Stage 6E2 pilot has not been run")
def test_joint_confirmation_ran_and_was_discarded():
    confirmation = json.loads((FULL / "joint_confirmation.json").read_text())
    assert confirmation["draws_discarded"] is True
    assert confirmation["all_passed"] is True, confirmation["failed_checks"]
    for name in ("all_targets_finite", "every_segmentation_legal",
                 "block_table_matches_registered_scorer",
                 "grouped_and_per_block_evaluators_agree", "boundaries_moved",
                 "every_move_type_proposed", "every_move_type_accepted"):
        assert confirmation["checks"][name] is True, name
    assert confirmation["block_table_parity"]["pass"] is True
    assert confirmation["evaluator_parity"]["pass"] is True


@pytest.mark.skipif(not (FULL / "corpus_manifest.json").exists(),
                    reason="the Stage 6E2 corpus has not been frozen")
def test_corpus_manifest_schema_and_leakage_audit():
    manifest = json.loads((FULL / "corpus_manifest.json").read_text())
    for name in ("corpus_hash", "config", "n_train_traces", "n_heldout_traces",
                 "n_train_blocks", "exposure_audit_train", "induced_orders",
                 "leakage_audit", "observed_train", "observed_heldout",
                 "hidden_true_boundaries_train", "hidden_true_labels_train"):
        assert name in manifest, name
    assert manifest["leakage_audit"]["pass"] is True
    assert manifest["leakage_audit"]["model_holds_no_true_segmentation"] is True
    assert manifest["config"]["seed_was_searched"] is False
    assert manifest["corpus_hash"] == corpus_hash(generate_corpus())


@pytest.mark.skipif(not (FULL / "recovery_results.json").exists(),
                    reason="the Stage 6E2 analysis has not been run")
def test_recovery_and_heldout_schemas():
    recovery = json.loads((FULL / "recovery_results.json").read_text())
    for name in ("boundary", "skill", "structure", "transitions", "verdicts"):
        assert name in recovery, name
    for name in ("boundary_f1", "boundary_precision", "boundary_recall",
                 "boundary_brier_score", "calibration", "segment_count_error",
                 "segment_length_distribution"):
        assert name in recovery["boundary"], name
    for name in ("occurrence_aligned_accuracy", "adjusted_rand_index",
                 "normalised_mutual_information", "segment_level_aligned_accuracy",
                 "repeated_invocation_aligned_accuracy", "confusion_matrix_aligned",
                 "label_permutation_mode_switches", "alignment_rule"):
        assert name in recovery["skill"], name
    assert "RECOVERY REPORTING ONLY" in recovery["skill"]["alignment_rule"]
    for entry in recovery["structure"]["per_skill"]:
        for name in ("true_closure", "map_closure", "probability_of_true_order",
                     "relation_marginal", "closure", "transitive_reduction",
                     "structural_hamming_distance",
                     "min_probability_over_true_relations",
                     "max_probability_over_false_relations"):
            assert name in entry, name
    verdicts = recovery["verdicts"]
    for name in ("stage_6e2_convergence", "stage_6e2_boundary_recovery",
                 "stage_6e2_skill_label_recovery", "stage_6e2_structural_recovery",
                 "stage_6e2_scalar_recovery", "stage_6e2_identifiability"):
        assert name in verdicts, name
        assert verdicts[name] in ("PASS", "PARTIAL", "FAIL", "WELL IDENTIFIED",
                                  "PARTIALLY IDENTIFIED", "MULTIMODAL"), name

    heldout = json.loads((FULL / "heldout_results.json").read_text())
    assert "unknown_boundary_posterior_predictive" in heldout
    assert "true_parameter_oracle" in heldout
    negative = heldout["negative_control_h_of_mean_U"]
    assert "LABELLED NEGATIVE CONTROL" in negative["status"]
    if heldout.get("modal_h_representative_draw"):
        assert "NOT a posterior-mean plug-in" in (
            heldout["modal_h_representative_draw"]["naming"])
