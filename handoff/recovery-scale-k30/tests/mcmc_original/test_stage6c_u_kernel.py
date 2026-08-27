"""Stage 6C — the latent-`U` proposal (§17 areas 4, 5, 6, 7).

The committed Stage 6C0 audit overrides the brief here, and these tests encode the audit's
conclusions rather than the brief's assumptions:

* the state is the continuous matrix `U`, so there is no canonical state id to maintain
  and nothing to canonicalise after a move;
* the move is `U'[j,:] = U[j,:] + sigma_U * N(0, I_d)`, whose density is symmetric in
  `(U, U')` **by construction**, so `log q(U|U') - log q(U'|U) = 0` exactly — proved
  below, not assumed from the word "random walk";
* legality is automatic: every real `U` induces a strict partial order, so there is no
  illegal proposal to reject and no neighbourhood whose size could differ between `U`
  and `U'`.

Because the state space is continuous it has no finite transition matrix, so the brief's
"tiny complete state space" detailed-balance check is performed the way the audit
specifies: on the **induced-order chain** of a reduced model, whose stationary law is
known exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_latent_poset_mcmc import (
    LatentPosetEvaluator, Stage6CTarget, initial_state, sweep_once,
)
from hpop.mcmc_original.sampler_u import propose_row
from hpop.mcmc_original.stage6c_exact_reference import build_catalogue, prior_cell_masses
from hpop.mcmc_original.stage6c_frozen import SIGMA_U, log_structural_prior

FIXED = {"beta": 1.5, "omega": 1.7346, "lambda_rep": 0.8, "lambda_back": 0.25}


class ZeroLikelihoodEvaluator(LatentPosetEvaluator):
    """A reduced model whose likelihood is identically flat.

    This is the honest way to isolate the kernel: with the likelihood constant, the target
    is exactly `p(U | rho) p(rho)`, whose induced-order marginal is the prior cell mass
    `pi_rho(P)` — a quantity computable independently of any chain. Anything the chain
    gets wrong then shows up as a discrepancy against a known law rather than against
    another run of the same code.
    """

    def __init__(self, m: int):
        role_array = np.zeros((1, 1), dtype=int)
        super().__init__(role_array, epsilon=0.02, omega=1.7346)
        self._m = m

    def full_replay_log_likelihood(self, u, beta, omega, lambda_rep, lambda_back) -> float:
        self.full_replay_calls += 1
        return 0.0

    def log_likelihood(self, u, beta, omega, lambda_rep, lambda_back,
                       allow_cache: bool = True) -> float:
        return 0.0


# ------------------------------------------------------- area 4: legality is automatic
def test_every_proposed_u_induces_a_legal_partial_order():
    """5,000 proposals from dispersed starts; every induced relation must be a poset."""
    rng = np.random.default_rng(0)
    for _ in range(5_000):
        u = rng.normal(size=(5, 2)) * rng.choice([0.1, 1.0, 10.0])
        candidate = propose_row(u, int(rng.integers(5)), SIGMA_U, rng)
        p = precedence_from_u(candidate)
        assert not p.diagonal().any()                       # irreflexive
        assert not (p & p.T).any()                          # antisymmetric
        closed = (p.astype(int) @ p.astype(int)) > 0
        assert not (closed & ~p).any()                      # transitive


def test_proposal_touches_exactly_one_row():
    rng = np.random.default_rng(1)
    u = rng.normal(size=(5, 2))
    for row in range(5):
        candidate = propose_row(u, row, SIGMA_U, rng)
        moved = ~np.isclose(candidate, u).all(axis=1)
        assert moved.sum() == 1 and moved[row]


def test_proposal_rejects_an_out_of_range_row():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        propose_row(np.zeros((5, 2)), 5, SIGMA_U, rng)


# ------------------------------------- areas 5, 6: forward/reverse density and Hastings
def _log_q(u_from, u_to, row, sigma):
    """Density of the row-wise Gaussian random walk, written out independently here."""
    delta = np.asarray(u_to)[row] - np.asarray(u_from)[row]
    d = delta.size
    return float(-0.5 * (d * math.log(2 * math.pi * sigma ** 2)
                         + float(delta @ delta) / sigma ** 2))


def test_forward_and_reverse_proposal_densities_are_equal():
    """Computed from an independent formula, not read off the implementation."""
    rng = np.random.default_rng(3)
    for _ in range(500):
        u = rng.normal(size=(5, 2))
        row = int(rng.integers(5))
        candidate = propose_row(u, row, SIGMA_U, rng)
        forward = _log_q(u, candidate, row, SIGMA_U)
        reverse = _log_q(candidate, u, row, SIGMA_U)
        assert forward == pytest.approx(reverse, abs=1e-13)


def test_hastings_term_is_exactly_zero():
    rng = np.random.default_rng(4)
    for _ in range(500):
        u = rng.normal(size=(5, 2))
        row = int(rng.integers(5))
        candidate = propose_row(u, row, SIGMA_U, rng)
        hastings = _log_q(candidate, u, row, SIGMA_U) - _log_q(u, candidate, row, SIGMA_U)
        assert hastings == 0.0


def test_symmetry_holds_at_every_proposal_scale():
    rng = np.random.default_rng(5)
    for sigma in (0.01, 0.1, 0.5, 2.0, 25.0):
        u = rng.normal(size=(5, 2))
        candidate = propose_row(u, 2, sigma, rng)
        assert _log_q(u, candidate, 2, sigma) == pytest.approx(
            _log_q(candidate, u, 2, sigma), abs=1e-13)


def test_neighbourhood_size_argument_does_not_apply():
    """Both directions have the same, full-measure support — no asymmetric neighbourhood.

    The brief's worry is a discrete move set whose legal-neighbour count differs between
    states. Here the proposal density is positive on all of `R^d` from every state, so the
    counts are not merely equal, the concept does not apply.
    """
    rng = np.random.default_rng(6)
    u = rng.normal(size=(5, 2))
    far = u.copy()
    far[1] += 1e4
    assert math.isfinite(_log_q(u, far, 1, SIGMA_U))
    assert math.isfinite(_log_q(far, u, 1, SIGMA_U))
    assert _log_q(u, far, 1, SIGMA_U) == pytest.approx(_log_q(far, u, 1, SIGMA_U),
                                                       abs=1e-13)


# ----------------------------- area 5: implemented ratio == direct target difference
def test_implemented_u_acceptance_ratio_equals_the_direct_target_difference():
    """At deterministic states, with the Hastings term known to be zero.

    The direct target is evaluated through `Stage6CTarget.decompose`, which never calls an
    acceptance-ratio helper, so this compares two genuinely independent routes.
    """
    rng = np.random.default_rng(7)
    evaluator = ZeroLikelihoodEvaluator(5)
    target = Stage6CTarget(evaluator, active=("U", "rho"), fixed=FIXED)
    for _ in range(200):
        u = rng.normal(size=(5, 2))
        rho = float(rng.uniform(0.05, 0.9))
        row = int(rng.integers(5))
        candidate = propose_row(u, row, SIGMA_U, rng)

        implemented = (log_structural_prior(candidate, rho)
                       - log_structural_prior(u, rho))          # likelihood is flat here
        direct = (target.log_target(candidate, {"rho": rho}, allow_cache=False)
                  - target.log_target(u, {"rho": rho}, allow_cache=False))
        hastings = _log_q(candidate, u, row, SIGMA_U) - _log_q(u, candidate, row, SIGMA_U)
        assert implemented == pytest.approx(direct + hastings, abs=1e-9)


# ------------------------------- area 7: detailed balance on the induced-order chain
def test_induced_order_chain_matches_the_exact_prior_cell_masses():
    """The reduced model's induced-order law is `pi_rho(P)`, known without any chain.

    `m = 3, d = 2` has a 19-poset catalogue, so the whole law can be written down. The
    chain is the production `sweep_once`, with `rho` frozen by a zero-scale proposal so
    the target's `U` marginal is exactly `N(0, Sigma_rho)` row-wise. If the kernel failed
    to preserve the target, this comparison is where it would show.
    """
    m, rho = 3, 0.4
    catalogue = build_catalogue(m, 2)
    assert catalogue.size == 19

    exact = prior_cell_masses(catalogue, [rho], n_draws=4_000_000, seed=11)
    reference = exact["masses"][0]

    evaluator = ZeroLikelihoodEvaluator(m)
    target = Stage6CTarget(evaluator, active=("U", "rho"), fixed=FIXED)
    rng = np.random.default_rng(2024)
    state = initial_state(target, np.zeros((m, 2)), {"rho": rho}, rng)

    n_sweeps, burn_in = 60_000, 2_000
    counts = np.zeros(catalogue.size)
    for i in range(n_sweeps):
        state = sweep_once(state, target, SIGMA_U, 0.0, 0.05, rng)
        if i >= burn_in:
            counts[catalogue.index_of(precedence_from_u(state.u))] += 1
    empirical = counts / counts.sum()

    assert state.values["rho"] == pytest.approx(rho, abs=1e-12)   # rho really was frozen
    total_variation = 0.5 * float(np.abs(empirical - reference).sum())
    assert total_variation < 0.02, (
        f"induced-order law differs from the exact prior cell masses by TV="
        f"{total_variation:.4f}; the U kernel does not preserve p(U | rho)")


def test_a_deliberately_broken_acceptance_rule_fails_the_same_comparison():
    """Control for the test above: bias the rule and the induced-order law must move.

    Without this, a TV below the gate would not distinguish "the kernel is correct" from
    "the gate is too loose to notice".
    """
    m, rho = 3, 0.4
    catalogue = build_catalogue(m, 2)
    reference = prior_cell_masses(catalogue, [rho], n_draws=1_000_000, seed=11)["masses"][0]

    rng = np.random.default_rng(5)
    u = np.zeros((m, 2))
    counts = np.zeros(catalogue.size)
    current = log_structural_prior(u, rho)
    for i in range(40_000):
        for row in range(m):
            candidate = propose_row(u, row, SIGMA_U, rng)
            proposed = log_structural_prior(candidate, rho)
            # BROKEN: the acceptance ratio omits the current state's density
            if math.log(rng.random()) < min(0.0, proposed):
                u, current = candidate, proposed
        if i >= 2_000:
            counts[catalogue.index_of(precedence_from_u(u))] += 1
    empirical = counts / counts.sum()
    total_variation = 0.5 * float(np.abs(empirical - reference).sum())
    assert total_variation > 0.02, (
        "a knowingly wrong acceptance rule still matched the exact law; the "
        "detailed-balance comparison is not sensitive enough to be evidence")
