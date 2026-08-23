"""Stage 2B — joint segmentation + U MCMC on a small synthetic corpus."""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_original import toy
from hpop.mcmc_original.diagnostics import prf1, relation_posterior
from hpop.mcmc_original.enumerate import build_trace_states
from hpop.mcmc_original.sampler_segmentation import run_joint_mcmc
from hpop.mcmc_original.sampler_u import dispersed_initial_u
from hpop.mcmc_original.targets import SkillEvaluator

SEED = 20260808
SIGMA_U = 0.8
N_ITERATIONS = 15_000
BURN_IN = 3_000
THIN = 3
N_CHAINS = 4

TRUE_RELATIONS = {
    toy.SKILL_A: {(0, 1)},
    toy.SKILL_B: {(0, 1), (1, 2), (0, 2)},
}


@pytest.fixture(scope="module")
def stage2b():
    rng = np.random.default_rng(SEED)
    corpus = toy.make_stage2b_corpus(rng)
    skills = toy.stage012_skills()
    evaluators = [SkillEvaluator(s) for s in skills]
    trace_states = [
        build_trace_states(
            c["x"], skills, evaluators, toy.DELTA_B, c["true_cut"], c["true_path"]
        )
        for c in corpus
    ]
    log_pi = toy.uniform_log_pi(len(skills))

    chains = []
    for chain in range(N_CHAINS):
        chain_rng = np.random.default_rng(SEED + 900 + chain)
        init_u = {
            k: dispersed_initial_u(skills[k].u.shape, toy.RHO_U, chain_rng)
            for k in range(len(skills))
        }
        chains.append(
            run_joint_mcmc(
                trace_states, evaluators, init_u, toy.RHO_U, SIGMA_U,
                N_ITERATIONS, BURN_IN, THIN, chain_rng, log_pi,
            )
        )
    return {
        "corpus": corpus,
        "skills": skills,
        "trace_states": trace_states,
        "chains": chains,
    }


def boundary_sets(trace_states, states):
    """Posterior-majority cuts (> 0.5) and the true cuts, as comparable sets."""
    predicted, truth = set(), set()
    n_draws = states.shape[0]
    for t, ts in enumerate(trace_states):
        truth.add((t, ts.true_cut))
        frequencies = np.bincount(states[:, t], minlength=ts.n_states) / n_draws
        marginal: dict[int, float] = {}
        for s, p in enumerate(frequencies):
            for cut in ts.cuts[s]:
                marginal[cut] = marginal.get(cut, 0.0) + p
        for cut, p in marginal.items():
            if p > 0.5:
                predicted.add((t, cut))
    return predicted, truth


def relation_sets(chains):
    predicted, truth = set(), set()
    for skill_id, relations in TRUE_RELATIONS.items():
        pooled = relation_posterior(
            np.concatenate([c["u_samples"][skill_id] for c in chains])
        )
        for i, j in relations:
            truth.add((skill_id, i, j))
        m = pooled.shape[0]
        for i in range(m):
            for j in range(m):
                if i != j and pooled[i, j] > 0.5:
                    predicted.add((skill_id, i, j))
    return predicted, truth


# ---------------------------------------------------------------------------
# the corpus and its state lists
# ---------------------------------------------------------------------------


def test_corpus_shape_and_truth_are_consistent(stage2b):
    corpus = stage2b["corpus"]
    assert len(corpus) == 40
    assert sum(1 for c in corpus if c["true_path"] == (toy.SKILL_B, toy.SKILL_A)) == 20
    assert sum(1 for c in corpus if c["true_path"] == (toy.SKILL_A, toy.SKILL_B)) == 20
    for c in corpus:
        assert len(c["x"]) == 5
        assert 0 < c["true_cut"] < 5


def test_true_segmentation_is_always_among_the_enumerated_states(stage2b):
    """If the truth were not enumerable, recovery would be impossible by construction."""
    for ts in stage2b["trace_states"]:
        assert any(
            ts.true_cut in cuts and path == ts.true_path
            for cuts, path in zip(ts.cuts, ts.paths)
        ), f"truth missing for trace {ts.x}"


def test_state_lists_are_small_and_nonempty(stage2b):
    sizes = [ts.n_states for ts in stage2b["trace_states"]]
    assert min(sizes) >= 1
    assert max(sizes) <= 2
    assert any(s == 2 for s in sizes), "no ambiguous trace: the test would be vacuous"


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------


def test_joint_mcmc_recovers_boundaries(stage2b):
    for c, chain in enumerate(stage2b["chains"]):
        predicted, truth = boundary_sets(stage2b["trace_states"], chain["states"])
        scores = prf1(predicted, truth)
        assert scores["f1"] >= 0.85, f"chain {c}: boundary F1 = {scores['f1']:.4f}"


def test_joint_mcmc_recovers_skill_paths(stage2b):
    for c, chain in enumerate(stage2b["chains"]):
        correct = 0
        for t, ts in enumerate(stage2b["trace_states"]):
            frequencies = np.bincount(
                chain["states"][:, t], minlength=ts.n_states
            ) / chain["states"].shape[0]
            if ts.paths[int(np.argmax(frequencies))] == ts.true_path:
                correct += 1
        accuracy = correct / len(stage2b["trace_states"])
        assert accuracy >= 0.85, f"chain {c}: skill-path accuracy = {accuracy:.4f}"


def test_joint_mcmc_recovers_precedence_relations(stage2b):
    predicted, truth = relation_sets(stage2b["chains"])
    scores = prf1(predicted, truth)
    assert scores["f1"] >= 0.85, f"relation F1 = {scores['f1']:.4f}"
    assert predicted == truth, f"predicted {sorted(predicted)} vs true {sorted(truth)}"


def test_joint_mcmc_does_not_collapse_or_fragment(stage2b):
    """Guard against the two documented failure modes rather than only checking F1."""
    for c, chain in enumerate(stage2b["chains"]):
        n_segments = []
        for t, ts in enumerate(stage2b["trace_states"]):
            for state in np.unique(chain["states"][:, t]):
                n_segments.append(len(ts.paths[state]))
        assert set(n_segments) == {2}, f"chain {c}: unexpected segment counts"
        assert 0.0 <= chain["segmentation_acceptance_rate"] <= 1.0
        assert 0.10 < chain["u_acceptance_rate"] < 0.60


def test_chains_agree_with_each_other(stage2b):
    """Independent chains should not disagree about the boundaries."""
    predictions = [
        boundary_sets(stage2b["trace_states"], chain["states"])[0]
        for chain in stage2b["chains"]
    ]
    for other in predictions[1:]:
        assert other == predictions[0]
