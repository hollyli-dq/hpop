"""Stage 6 — recurrent state, step probabilities, likelihood, generator, exposure.

Includes an INDEPENDENT slow reference implementation (spec section 11). It is written
with explicit loops and never calls the vectorized production functions, so a shared
indexing or orientation bug cannot hide behind agreement.
"""
from __future__ import annotations
import itertools, math
import numpy as np
import pytest
from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.static_bpop import bpop_step_probabilities
from hpop.mcmc_original import recurrent_synthetic as rs
from hpop.mcmc_original import stage6_diagnostics as sd
from hpop.mcmc_original.recurrent_rfs import (
    RecurrentRFSParameters, logit, sigmoid, recurrent_back_costs, recurrent_feasibility,
    recurrent_rfs_likelihood, recurrent_rfs_log_likelihood, recurrent_stale_successor_counts,
    recurrent_step_probabilities, recurrent_structural_weights, recurrent_successor_utilities,
    recurrent_validity_update, sample_recurrent_rfs_sequence,
)

KAPPA = 0.85
OMEGA = logit(KAPPA)
PARAMS = RecurrentRFSParameters(1.5, 0.02, OMEGA, 0.8, 0.25)


# ---------------------------------------------------------------- reference (sec 11)
def ref_feasibility(P, q):
    m = len(q); out = []
    for x in range(m):
        v = 1.0
        for z in range(m):
            if P[z][x]: v *= q[z]
        out.append(v)
    return out

def ref_stale(P, q):
    m = len(q); out = []
    for x in range(m):
        s = 0.0
        for z in range(m):
            if z != x and P[x][z]: s += 1.0 - q[z]
        out.append(s)
    return out

def ref_back(P, q, omega):
    m = len(q); k = 1.0/(1.0+math.exp(-omega)); out = []
    for x in range(m):
        s = 0.0
        for z in range(m):
            if z != x and P[x][z]: s += k * q[z]
        out.append(s)
    return out

def ref_update(y, P, q, omega):
    m = len(q); k = 1.0/(1.0+math.exp(-omega)); out = []
    for x in range(m):
        if x == y: out.append(1.0)
        elif P[y][x]: out.append(q[x]*(1.0-k))
        else: out.append(q[x])
    return out

def ref_step(P, q, p):
    m = len(q)
    F = ref_feasibility(P, q); S = ref_stale(P, q); B = ref_back(P, q, p.shared_omega)
    W = [F[x]*math.exp(p.beta*math.log(1.0+S[x]) - p.lambda_rep*q[x] - p.lambda_back*B[x])
         for x in range(m)]
    total = sum(W)
    return [(1.0-p.epsilon)*(W[x]/total) + p.epsilon/m for x in range(m)]

def ref_loglik(seq, U, p):
    P = [[bool(v) for v in row] for row in precedence_from_u(U)]
    m = U.shape[0]; q = [0.0]*m; total = 0.0
    for y in seq:
        prob = ref_step(P, q, p)[y]
        if prob <= 0.0:            # eps = 0 makes an infeasible role impossible
            return -math.inf
        total += math.log(prob)
        q = ref_update(y, P, q, p.shared_omega)
    return total


def test_stage6_u_order_is_correct():
    P = precedence_from_u(rs.U_TRUE)
    assert sorted((i, j) for i in range(5) for j in range(5) if P[i, j]) == \
        [(0, 2), (0, 3), (0, 4), (2, 3), (2, 4), (3, 4)]
    for j in (0, 2, 3, 4):
        assert not P[1, j] and not P[j, 1]
    rs.assert_stage6_library()


def test_stage6_existing_order_mapping_parity():
    """The recurrent module must read the SAME orientation as the static code."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        U = rng.normal(size=(5, 2))
        P = precedence_from_u(U)
        for z in range(5):
            for x in range(5):
                if P[z, x]:                       # z precedes x
                    assert all(U[z, r] > U[x, r] for r in range(2))
        # a role with a remaining predecessor has zero feasibility at q=0
        F = recurrent_feasibility(P, np.zeros(5))
        for x in range(5):
            assert (F[x] == 0.0) == bool(P[:, x].any())


def test_stage6_initial_feasibility():
    assert recurrent_feasibility(precedence_from_u(rs.U_TRUE), np.zeros(5)).tolist() == \
        [1.0, 1.0, 0.0, 0.0, 0.0]


def test_stage6_initial_successor_utilities():
    P = precedence_from_u(rs.U_TRUE); q = np.zeros(5)
    assert recurrent_stale_successor_counts(P, q).astype(int).tolist() == [3, 0, 2, 1, 0]
    Q = recurrent_successor_utilities(P, q)
    assert Q[0] == pytest.approx(math.log(4)); assert Q[1] == pytest.approx(0.0)
    w = recurrent_structural_weights(P, q, 1.5, OMEGA, 0.0, 0.0)
    assert w[0] != pytest.approx(w[1])          # beta is active


def test_stage6_validity_state_bounds():
    P = precedence_from_u(rs.U_TRUE); rng = np.random.default_rng(1)
    for _ in range(200):
        q = rng.random(5)
        for y in range(5):
            new = recurrent_validity_update(y, P, q, OMEGA)
            assert np.all(new >= 0.0) and np.all(new <= 1.0)


def test_stage6_refreshes_observed_role():
    P = precedence_from_u(rs.U_TRUE); rng = np.random.default_rng(2)
    for _ in range(50):
        q = rng.random(5)
        for y in range(5):
            assert recurrent_validity_update(y, P, q, OMEGA)[y] == 1.0


def test_stage6_only_descendants_are_invalidated():
    P = precedence_from_u(rs.U_TRUE); rng = np.random.default_rng(3)
    for _ in range(50):
        q = rng.random(5)
        for y in range(5):
            new = recurrent_validity_update(y, P, q, OMEGA)
            for x in range(5):
                if x != y and not P[y, x]:
                    assert new[x] == q[x]
                elif x != y:
                    assert new[x] == pytest.approx(q[x]*(1-KAPPA))


def test_stage6_transitive_descendants_are_invalidated():
    P = precedence_from_u(rs.U_TRUE)
    new = recurrent_validity_update(0, P, np.ones(5), OMEGA)
    for x in (2, 3, 4):                      # not just the cover successor 2
        assert new[x] == pytest.approx(1 - KAPPA)
    assert new[1] == 1.0


def test_stage6_repeated_invalidation_compounds():
    P = precedence_from_u(rs.U_TRUE)
    q = recurrent_validity_update(0, P, np.ones(5), OMEGA)
    q = recurrent_validity_update(0, P, q, OMEGA)
    assert q[4] == pytest.approx((1 - KAPPA) ** 2)


def test_stage6_step_probabilities_normalize():
    rng = np.random.default_rng(4)
    for m in (2, 3, 5):
        for _ in range(30):
            U = rng.normal(size=(m, 2)); q = rng.random(m)
            p = recurrent_step_probabilities(U, q, PARAMS)
            assert np.all(np.isfinite(p)) and np.all(p >= 0)
            assert p.sum() == pytest.approx(1.0, abs=1e-12)


def test_stage6_noise_gives_full_support():
    rng = np.random.default_rng(5)
    for _ in range(30):
        U = rng.normal(size=(5, 2)); q = rng.random(5)
        assert np.all(recurrent_step_probabilities(U, q, PARAMS) > 0.0)


def test_stage6_epsilon_zero_respects_feasibility():
    zero = RecurrentRFSParameters(1.5, 0.0, OMEGA, 0.8, 0.25)
    P = precedence_from_u(rs.U_TRUE)
    p = recurrent_step_probabilities(rs.U_TRUE, np.zeros(5), zero)
    F = recurrent_feasibility(P, np.zeros(5))
    for x in range(5):
        if F[x] == 0.0:
            assert p[x] == 0.0
    assert p.sum() == pytest.approx(1.0, abs=1e-12)


def test_stage6_first_step_matches_static_bpop():
    flat = RecurrentRFSParameters(1.5, 0.02, OMEGA, 0.0, 0.0)
    rec = recurrent_step_probabilities(rs.U_TRUE, np.zeros(5), flat)
    sta = bpop_step_probabilities(list(range(5)), rs.U_TRUE, 1.5, 0.02)
    np.testing.assert_allclose(rec, sta, atol=1e-12)


def test_stage6_reference_matches_production():
    """Independent loop-based reference vs the vectorized production code."""
    rng = np.random.default_rng(6)
    for m in (2, 3, 4):
        for _ in range(15):
            U = rng.normal(size=(m, 2)); q = rng.random(m)
            P = precedence_from_u(U); Pl = [[bool(v) for v in r] for r in P]
            for pr in (PARAMS, RecurrentRFSParameters(0.0, 0.0, logit(0.3), 0.0, 0.0),
                       RecurrentRFSParameters(3.0, 0.1, logit(0.99), 2.0, 1.5)):
                np.testing.assert_allclose(recurrent_feasibility(P, q), ref_feasibility(Pl, q.tolist()), atol=1e-12)
                np.testing.assert_allclose(recurrent_stale_successor_counts(P, q), ref_stale(Pl, q.tolist()), atol=1e-12)
                np.testing.assert_allclose(recurrent_back_costs(P, q, pr.shared_omega),
                                           ref_back(Pl, q.tolist(), pr.shared_omega), atol=1e-12)
                np.testing.assert_allclose(recurrent_step_probabilities(U, q, pr),
                                           ref_step(Pl, q.tolist(), pr), atol=1e-12)
                for y in range(m):
                    np.testing.assert_allclose(recurrent_validity_update(y, P, q, pr.shared_omega),
                                               ref_update(y, Pl, q.tolist(), pr.shared_omega), atol=1e-12)
                seq = [int(v) for v in rng.integers(0, m, size=5)]
                produced = recurrent_rfs_log_likelihood(
                    seq, U, pr.beta, pr.epsilon, pr.shared_omega, pr.lambda_rep, pr.lambda_back)
                expected = ref_loglik(seq, U, pr)
                if expected == -math.inf:
                    assert produced == -math.inf   # both agree the sequence is impossible
                else:
                    assert produced == pytest.approx(expected, abs=1e-12)


def _normalizes(m, T):
    rng = np.random.default_rng(7)
    for (b, e, k, lr, lb) in [(1.5, .02, .85, .8, .25), (0., 0., .5, 0., 0.),
                              (3., .1, .99, 2., 1.5), (1., .05, .01, 0., 1.)]:
        U = rng.normal(size=(m, 2))
        total = sum(recurrent_rfs_likelihood(s, U, b, e, logit(k), lr, lb)
                    for s in itertools.product(range(m), repeat=T))
        assert total == pytest.approx(1.0, abs=1e-10), f"m={m} T={T} sum={total}"


def test_stage6_fixed_length_likelihood_normalizes_m2_t4(): _normalizes(2, 4)
def test_stage6_fixed_length_likelihood_normalizes_m3_t3(): _normalizes(3, 3)
def test_stage6_fixed_length_likelihood_normalizes_m3_t4(): _normalizes(3, 4)


def test_stage6_likelihood_allows_repeats_and_validates_input():
    assert recurrent_rfs_likelihood([0, 0, 0], rs.U_TRUE, 1.5, .02, OMEGA, .8, .25) > 0
    with pytest.raises(ValueError): recurrent_rfs_likelihood([0, 9], rs.U_TRUE, 1.5, .02, OMEGA, .8, .25)
    with pytest.raises(ValueError): recurrent_rfs_likelihood([0], rs.U_TRUE, -1., .02, OMEGA, .8, .25)
    with pytest.raises(ValueError): recurrent_rfs_likelihood([0], rs.U_TRUE, 1.5, 1.0, OMEGA, .8, .25)
    with pytest.raises(ValueError): recurrent_rfs_likelihood([0], rs.U_TRUE, 1.5, .02, OMEGA, -1., .25)


def test_stage6_generator_likelihood_parity():
    data = rs.generate_recurrent_dataset("smoke", 0)
    p = sd.generator_likelihood_parity(data.train, data.u_true, data.parameters)
    assert p["max_abs_log_likelihood_difference"] < 1e-12
    assert p["max_abs_q_difference"] == 0.0
    assert p["max_abs_step_logp_difference"] == 0.0


def test_stage6_generator_is_reproducible():
    a = rs.generate_recurrent_dataset("smoke", 0)
    b = rs.generate_recurrent_dataset("smoke", 0)
    assert [x.roles for x in a.train] == [x.roles for x in b.train]
    assert [x.roles for x in rs.generate_recurrent_dataset("smoke", 1).train] != [x.roles for x in a.train]


def test_stage6_empirical_frequencies_match_exact():
    U = np.array([[1.0, 1.0], [0.0, 0.0]])
    pr = RecurrentRFSParameters(1.5, .02, OMEGA, .8, .25)
    exact = {s: recurrent_rfs_likelihood(s, U, pr.beta, pr.epsilon, pr.shared_omega,
                                         pr.lambda_rep, pr.lambda_back)
             for s in itertools.product(range(2), repeat=4)}
    assert sum(exact.values()) == pytest.approx(1.0, abs=1e-12)
    rng = np.random.default_rng(0); n = 100_000
    counts = {s: 0 for s in exact}
    for _ in range(n):
        counts[sample_recurrent_rfs_sequence(rng, 4, U, pr)] += 1
    tv = 0.5 * sum(abs(counts[s]/n - exact[s]) for s in exact)
    assert tv < 0.01, f"TV = {tv}"


def test_stage6_exposure_audit_counts_hand_example():
    data = rs.generate_recurrent_dataset("smoke", 0)
    audit = rs.exposure_audit(data.train)
    assert audit["total_steps"] == 50 * 12
    assert audit["valid_repeat"] == audit["leaf_repeat"] + audit["upstream_repeat"]
    assert audit["leaf_repeat"] > 0 and audit["upstream_repeat"] > 0
    gate = np.array(audit["gate_exposure"])
    P = precedence_from_u(rs.U_TRUE)
    for z in range(5):
        for x in range(5):
            if not P[z, x]:
                assert gate[z, x] == 0, "gate exposure must only accumulate on true ordered pairs"


def test_stage6_true_model_beats_wrong_antichain():
    data = rs.generate_recurrent_dataset("smoke", 0)
    diag = sd.held_out_diagnostics(data.train, data.u_true, data.parameters)
    assert diag["hard_criterion_true_beats_antichain"]
    assert diag["variants"]["wrong_antichain_U"]["paired_mean_difference"] > 0
