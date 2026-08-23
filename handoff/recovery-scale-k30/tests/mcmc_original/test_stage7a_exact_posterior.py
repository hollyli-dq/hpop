"""Step 7A — FFBS against the frozen Stage 6E1A exact posterior.

This is the primary correctness comparison in miniature: the same frozen problem, the same
exact answer, a smaller draw count so the test stays fast. The full 100,000-draw result
lives in `results/mcmc_original/stage7a_ffbs_exact/`.

Also here: the independence property the whole comparison rests on. If `semi_markov_ffbs`
ever imported the Stage 6E reference recursion, agreement between them would stop being
evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hpop.mcmc_original import semi_markov_ffbs
from hpop.mcmc_original.block_score_adapters import build_log_block_scores
from hpop.mcmc_original.recurrent_segmentation import RecurrentBlockScorer
from hpop.mcmc_original.semi_markov_ffbs import (
    backward_sample_many, forward, posterior_log_marginals,
)
from hpop.mcmc_original.stage6e_exact import (
    boundary_marginals, enumerate_states, exact_posterior, log_evidence_forward,
    occurrence_label_marginals, state_log_weights, total_variation,
)
from hpop.mcmc_original.transitions import allowed_next, log_transition_matrix
from tests.mcmc_original.test_stage7a_recurrent_blocks import (
    EPSILON, J, K, MAX_WIDTH, MIN_WIDTH, SCALARS, TRACE, U_BY_SKILL, make_scorer,
)

DELTA_B = 0.15
FROZEN = (Path(__file__).resolve().parents[2] / "results" / "mcmc_original"
          / "stage6e1a_exact_segmentation")


def path_prior():
    log_pi = np.log(np.full(K, 1.0 / K))
    transition = np.zeros((K, K))
    for h in range(K):
        for k in allowed_next(h, K):
            transition[h, k] = 1.0 / (K - 1)
    return log_pi, log_transition_matrix(transition)


def exact_reference():
    log_pi, log_p = path_prior()
    scorer = make_scorer()
    states = enumerate_states(J, K, MIN_WIDTH, MAX_WIDTH)
    weights = state_log_weights(states, 0, J, scorer, log_pi, log_p, DELTA_B)
    return states, exact_posterior(states, weights), scorer, log_pi, log_p


def test_the_trace_is_the_frozen_stage6e1a_trace():
    config = json.loads((FROZEN / "config.json").read_text())
    assert list(TRACE) == list(config["problem"]["observed_trace"])
    assert config["problem"]["trace_length_J"] == J
    assert config["problem"]["n_skills"] == K
    assert config["problem"]["m_roles"] == 3
    assert config["problem"]["delta_B"] == DELTA_B
    assert config["problem"]["min_width"] == MIN_WIDTH
    assert config["problem"]["max_width"] == MAX_WIDTH
    assert config["problem"]["epsilon"] == EPSILON
    assert config["problem"]["scalars_fixed"] == SCALARS
    assert np.array_equal(np.array(config["problem"]["U_by_skill"]), U_BY_SKILL)


def test_log_z_matches_the_frozen_enumeration_to_machine_precision():
    states, posterior, scorer, log_pi, log_p = exact_reference()
    frozen = np.load(FROZEN / "exact_reference.npz")
    assert abs(posterior["log_evidence"] - float(frozen["log_evidence"][0])) < 1e-12

    table = build_log_block_scores(scorer, 0, J, K, MIN_WIDTH, MAX_WIDTH)
    chart = forward(table, log_pi, log_p, DELTA_B, MAX_WIDTH, MIN_WIDTH)
    assert abs(chart.log_normalizer - float(frozen["log_evidence"][0])) < 1e-10
    assert abs(chart.log_normalizer - posterior["log_evidence"]) < 1e-10


def test_ffbs_and_the_stage6e1a_forward_recursion_agree_but_are_separate_code():
    """Two recursions, written independently, over the same problem."""
    states, posterior, scorer, log_pi, log_p = exact_reference()
    reference = log_evidence_forward(0, J, K, scorer, log_pi, log_p, DELTA_B, MIN_WIDTH,
                                     MAX_WIDTH)
    table = build_log_block_scores(scorer, 0, J, K, MIN_WIDTH, MAX_WIDTH)
    chart = forward(table, log_pi, log_p, DELTA_B, MAX_WIDTH, MIN_WIDTH)
    assert abs(chart.log_normalizer - reference) < 1e-10


def test_the_engine_is_model_agnostic_by_construction():
    source = Path(semi_markov_ffbs.__file__).read_text()
    body = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith(("*", "#")))
    for forbidden in ("hpop.", "stage6e", "LocalMoveKernel", "recurrent_rfs",
                      "latent_poset", "RecurrentBlockScorer"):
        assert forbidden not in body, f"the FFBS engine must not mention {forbidden!r}"
    assert "import numpy" in source and "logsumexp" in source


def test_dynamic_programming_marginals_match_the_enumerated_marginals():
    states, posterior, scorer, log_pi, log_p = exact_reference()
    table = build_log_block_scores(scorer, 0, J, K, MIN_WIDTH, MAX_WIDTH)
    chart = forward(table, log_pi, log_p, DELTA_B, MAX_WIDTH, MIN_WIDTH)
    marginals = posterior_log_marginals(chart)
    exact_p = posterior["probability"]
    assert np.abs(marginals["boundary_marginals"]
                  - boundary_marginals(states, exact_p, J)).max() < 1e-12
    assert np.abs(marginals["occurrence_label_marginals"]
                  - occurrence_label_marginals(states, exact_p, J, K)).max() < 1e-12
    assert abs(marginals["log_normalizer_from_beta"] - chart.log_normalizer) < 1e-12


@pytest.mark.parametrize("seed,n_draws", [(7_071_001, 40_000)])
def test_ffbs_draws_reproduce_the_exact_path_distribution(seed, n_draws):
    states, posterior, scorer, log_pi, log_p = exact_reference()
    exact_p = posterior["probability"]
    table = build_log_block_scores(scorer, 0, J, K, MIN_WIDTH, MAX_WIDTH)
    chart = forward(table, log_pi, log_p, DELTA_B, MAX_WIDTH, MIN_WIDTH)
    draws = backward_sample_many(chart, n_draws, np.random.default_rng(seed))

    index = {s: i for i, s in enumerate(states)}
    counts = np.zeros(len(states))
    for draw in draws:
        counts[index[tuple((b, k) for _, b, k in draw)]] += 1.0
    empirical = counts / counts.sum()

    assert total_variation(empirical, exact_p) < 0.01
    assert np.abs(boundary_marginals(states, empirical, J)
                  - boundary_marginals(states, exact_p, J)).max() < 0.01
    assert np.abs(occurrence_label_marginals(states, empirical, J, K)
                  - occurrence_label_marginals(states, exact_p, J, K)).max() < 0.01


def test_ffbs_support_is_exactly_the_enumerated_support():
    """Enough draws to visit every state with non-negligible mass, and never one outside."""
    states, posterior, scorer, log_pi, log_p = exact_reference()
    table = build_log_block_scores(scorer, 0, J, K, MIN_WIDTH, MAX_WIDTH)
    chart = forward(table, log_pi, log_p, DELTA_B, MAX_WIDTH, MIN_WIDTH)
    draws = backward_sample_many(chart, 20_000, np.random.default_rng(11))
    visited = {tuple((b, k) for _, b, k in draw) for draw in draws}
    assert visited <= set(states)
    heavy = {s for s, p in zip(states, posterior["probability"]) if p > 0.005}
    assert heavy <= visited


def test_a_different_scorer_parameterisation_moves_both_routes_together():
    """A change of parameters must move the enumerated posterior and the chart alike."""
    log_pi, log_p = path_prior()
    scorer = RecurrentBlockScorer(traces=(TRACE,), epsilon=EPSILON,
                                  u_by_skill=U_BY_SKILL, beta=2.4,
                                  omega=SCALARS["omega"], lambda_rep=0.3,
                                  lambda_back=0.9, min_width=MIN_WIDTH,
                                  max_width=MAX_WIDTH)
    states = enumerate_states(J, K, MIN_WIDTH, MAX_WIDTH)
    weights = state_log_weights(states, 0, J, scorer, log_pi, log_p, DELTA_B)
    posterior = exact_posterior(states, weights)
    table = build_log_block_scores(scorer, 0, J, K, MIN_WIDTH, MAX_WIDTH)
    chart = forward(table, log_pi, log_p, DELTA_B, MAX_WIDTH, MIN_WIDTH)
    assert abs(chart.log_normalizer - posterior["log_evidence"]) < 1e-10
