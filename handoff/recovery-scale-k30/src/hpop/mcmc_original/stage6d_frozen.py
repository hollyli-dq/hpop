"""Stage 6D — the frozen model, extending Stage 6C by freeing the last three scalars.

Stage 6D changes exactly one thing relative to Stage 6C2: `omega`, `lambda_rep` and
`lambda_back` stop being held at their registered values and join `U`, `rho` and `beta` as
inferred coordinates. The corpus, the likelihood branch, `epsilon`, the oracle boundaries,
the oracle skill labels, `q_0` and every prior are the Stage 6B/6C ones, unchanged and
re-verified here by hash.

    p(U, rho, beta, omega, lambda_rep, lambda_back | x, S*, z*, epsilon)
        proportional to
    p_RFS(x | S*, z*, H=h(U), beta, omega, lambda_rep, lambda_back, epsilon)
        x p(U | rho) p(rho) p(beta) p(omega) p(lambda_rep) p(lambda_back)

## Notation, stated once because three integers are easy to conflate

    m = 5    rows of U          role occurrences
    d = 2    columns of U       latent utility coordinates (the Stage 6D brief calls
                                this K; the repository has always called it d, and `K`
                                in this codebase means the number of skills)
    K = 1    skills             one U matrix, oracle-labelled blocks
    assessors                   none: the model has no assessor level
    tau                         does not exist in this model

`U in R^{m x d}` is continuous and is the MCMC state. `H = h(U)` is the *derived* strict
partial order, `i > j` iff `U[i,r] > U[j,r]` for every column r — the intersection of the
d column orderings, returned as a transitive closure. H is never state, never given a
second prior, and never enumerated as though it were the state space.

`rho` is the equicorrelation between the d COLUMNS within one row; rows are iid:

    U[j,:] | rho ~ N_d(0, Sigma_rho),  Sigma_rho = (1-rho) I_d + rho 1 1^T

## Where this file deliberately diverges from the Stage 6D brief

The brief was written against assumptions that the frozen Stage 6C model does not
satisfy. Each divergence below follows the brief's own precedence rule ("use the actual
repository state", "reuse the frozen Stage 6C proposal/update") rather than its
illustrative text, and each is recorded in `frozen_config()["spec_divergences"]` so the
report cannot quietly paper over it.

1. **No assessor hierarchy and no `tau`.** The brief's §2 allows for `U^(0)`, `U^(a)` and
   `tau`. This model has a single `U`; there is no assessor level, so the assessor-residual
   density of §3 has no referent and the hierarchical QMC construction of §10.1 is not
   built. `tau_in_model` is False.

2. **The `rho` prior is Beta(1, 1) truncated, not Beta(1, 1/6).** §4 names Beta(1, 1/6)
   "unless the final Stage 6C configuration establishes another prior" — it does. The
   registered prior comes from the vendored `StatisticalUtils.dRprior` with
   `stage5.RHO_PRIOR = 1.0` as the `fac` argument, i.e. Beta(1, 1) truncated at 1 - 5e-3
   and renormalised, which is Uniform(0, 0.995).

3. **The `rho` proposal is a logit random walk, not the scaling proposal.** §4 describes
   `delta ~ U(d_r, 1/d_r)`, `rho' = 1 - (1-rho) delta`, with ratio `-log delta`. The frozen
   Stage 6C kernel is a random walk on `z = logit(rho)` carrying `log(rho(1-rho))`. §6
   requires reusing the frozen Stage 6C rho update and §7.2 requires *numerical equality*
   with it, so adopting §4's proposal would fail the brief's own parity gate. The frozen
   kernel is therefore production. The scaling proposal's density ratio is nevertheless
   implemented and tested below (`scaling_proposal_log_ratio`) so that the identity §4
   asks about is pinned, clearly marked as **not used in production**.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from hpop.mcmc_original.recurrent_scalar_posterior import PRIORS, TRUE_VALUES
from hpop.mcmc_original.sampler_u import log_u_prior, sigma_rho_matrix
from hpop.mcmc_original.stage6c_frozen import (
    RHO_LOWER, RHO_PRIOR_FAC, RHO_TOL, RHO_UPPER, SIGMA_U, STAGE6B_CONFIG_HASH,
    STAGE6C_MODEL_ID, config_hash as stage6c_hash, frozen_config as stage6c_frozen_config,
    load_stage6c_dataset, log_det_sigma_rho, log_jacobian_rho, log_rho_prior,
    log_structural_prior, rho_from_unconstrained, rho_to_unconstrained,
)

__all__ = [
    "STAGE6D_MODEL_ID", "ACTIVE_6D", "SWEEP_ORDER_6D", "SCALAR_ORDER",
    "STAGE6C_CONFIG_HASH", "sigma_rho_inverse", "log_mvn_equicorrelated",
    "scaling_proposal", "scaling_proposal_log_ratio", "rho_is_in_support",
    "frozen_config", "config_hash", "assert_stage6c_unchanged", "load_stage6d_dataset",
    "REGISTERED_SCALES", "N_ROWS", "N_LATENT_COLUMNS", "N_SKILLS",
]

STAGE6D_MODEL_ID = "recurrent-rfs-oracle-joint-latent-v1"

# The Stage 6C freeze this stage extends. Verified by hash at import-adjacent call sites.
STAGE6C_CONFIG_HASH = "c1545b0b24acad470aca96954963ef619e4ee55d7094bb597041b6e4e98a8dad"

SCALAR_ORDER = ("beta", "omega", "lambda_rep", "lambda_back")
ACTIVE_6D = ("U", "rho") + SCALAR_ORDER
SWEEP_ORDER_6D = ACTIVE_6D

N_ROWS = 5              # m: role occurrences
N_LATENT_COLUMNS = 2    # d: latent utility columns (the brief's K)
N_SKILLS = 1            # one U matrix; NOT the row count and NOT the latent dimension

# Proposal scales, reused unchanged from the parents: sigma_U and the rho logit scale from
# Stage 6C, the four scalar scales from the frozen Stage 6B tuning.
REGISTERED_SCALES = {
    "U": SIGMA_U,
    "rho": 0.5,
    "beta": 0.05109,
    "omega": 0.18213,
    "lambda_rep": 0.04155,
    "lambda_back": 0.09472,
}


# ----------------------------------------------------------- analytic equicorrelation
def sigma_rho_inverse(d: int, rho: float) -> np.ndarray:
    """``Sigma_rho^{-1} = 1/(1-rho) [ I - rho/(1+(d-1)rho) 1 1^T ]``.

    The analytic form the brief's §3 asks for. `sampler_u.log_u_prior` evaluates the
    density through a Cholesky solve instead, which is better conditioned; this closed
    form exists so a test can confirm the two agree rather than so the sampler can use it.
    """
    if d < 1:
        raise ValueError(f"latent dimension must be >= 1, got {d}")
    lower = -1.0 / (d - 1) if d > 1 else -1.0
    if not (lower < rho < 1.0):
        raise ValueError(f"rho={rho} gives no positive-definite Sigma for d={d}")
    scale = 1.0 / (1.0 - rho)
    return scale * (np.eye(d) - (rho / (1.0 + (d - 1) * rho)) * np.ones((d, d)))


def log_mvn_equicorrelated(u: np.ndarray, rho: float) -> float:
    """`sum_j log N_d(U[j,:]; 0, Sigma_rho)`, built only from the analytic det and inverse.

    An independent route to `sampler_u.log_u_prior`: this one uses the closed-form
    determinant and inverse, that one uses a Cholesky factorisation. Agreement between
    them is evidence that the rho-dependent normaliser is present and correct, since the
    determinant enters here explicitly rather than implicitly through `log diag(chol)`.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 2:
        raise ValueError(f"u must be 2-D, got ndim={u.ndim}")
    m, d = u.shape
    quadratic = float(np.einsum("ij,jk,ik->", u, sigma_rho_inverse(d, rho), u))
    return -0.5 * (m * d * math.log(2.0 * math.pi)
                   + m * log_det_sigma_rho(d, rho)
                   + quadratic)


# ------------------------------------------------------------------- rho support/proposal
def rho_is_in_support(rho: float) -> bool:
    """The registered support is the open interval `(0, 1 - tol)`, tol = 5e-3."""
    return bool(RHO_LOWER < float(rho) < RHO_UPPER)


def scaling_proposal(rho: float, d_r: float, rng) -> tuple[float, float, float]:
    """**NOT USED IN PRODUCTION.** The brief's §4 scaling proposal, for its identity only.

    `delta ~ Uniform(d_r, 1/d_r)`, `rho' = 1 - (1-rho) delta`. Returns
    `(rho_proposed, delta, log q(rho|rho') - log q(rho'|rho))`.

    The production `rho` kernel is the frozen Stage 6C logit random walk; see this
    module's docstring for why. This function exists so the `-log(delta)` identity is
    pinned by a test rather than asserted in prose, and so a future stage that adopts the
    scaling proposal inherits a verified implementation instead of re-deriving it.
    """
    if not (0.0 < d_r < 1.0):
        raise ValueError(f"d_r must lie in (0, 1), got {d_r}")
    delta = float(rng.uniform(d_r, 1.0 / d_r))
    proposed = 1.0 - (1.0 - float(rho)) * delta
    return proposed, delta, scaling_proposal_log_ratio(delta)


def scaling_proposal_log_ratio(delta: float) -> float:
    """`log q(rho | rho') - log q(rho' | rho) = -log(delta)` for the scaling proposal.

    The forward move multiplies `1-rho` by `delta`; the reverse multiplies `1-rho'` by
    `1/delta`. The Uniform density on `delta` is flat, so the whole ratio is the Jacobian
    of the map, namely `-log delta`.
    """
    if delta <= 0.0:
        return -math.inf
    return -math.log(float(delta))


# ------------------------------------------------------------------------ frozen config
@dataclass(frozen=True)
class FrozenStage6D:
    train: np.ndarray
    heldout: np.ndarray
    u_true: np.ndarray
    epsilon: float
    truth: dict
    config: dict

    @property
    def n_roles(self) -> int:
        return int(self.u_true.shape[0])

    @property
    def latent_dimension(self) -> int:
        """d — the number of latent COLUMNS. Not the role count, not the skill count."""
        return int(self.u_true.shape[1])

    @property
    def n_skills(self) -> int:
        return N_SKILLS

    @property
    def n_assessors(self) -> int:
        """Zero: this model has no assessor level and no tau."""
        return 0


def frozen_config() -> dict:
    base = stage6c_frozen_config()
    return {
        "stage": "6D",
        "model_id": STAGE6D_MODEL_ID,
        "stage6c_model_id": STAGE6C_MODEL_ID,
        "stage6c_config_hash": stage6c_hash(),
        "stage6b_config_hash": STAGE6B_CONFIG_HASH,
        "target": "p(U, rho, beta, omega, lambda_rep, lambda_back | x, S*, z*, epsilon) "
                  "proportional to p_RFS(x | S*, z*, h(U), scalars, epsilon) p(U|rho) "
                  "p(rho) p(beta) p(omega) p(lambda_rep) p(lambda_back)",
        "inherits_from_stage6c": {
            "likelihood_branch": base["inherits_from_stage6b"]["likelihood_branch"],
            "dataset": base["inherits_from_stage6b"]["dataset"],
            "epsilon": base["inherits_from_stage6b"]["epsilon"],
            "boundaries": base["inherits_from_stage6b"]["boundaries"],
            "numerics": base["inherits_from_stage6b"]["numerics"],
            "structural_model": base["structural_model"],
            "rho_prior": base["rho_prior"],
            "rho_transform": base["rho_transform"],
            "u_proposal": base["u_proposal"],
        },
        "dimensions": {
            "m_rows": N_ROWS, "d_latent_columns": N_LATENT_COLUMNS,
            "n_skills": N_SKILLS, "n_assessors": 0,
            "note": "m, d and K are three different integers; the brief writes the "
                    "latent column count as K, the repository writes it as d and "
                    "reserves K for the skill count",
        },
        "active": list(ACTIVE_6D),
        "sweep_order": list(SWEEP_ORDER_6D),
        "scalar_order": list(SCALAR_ORDER),
        "proposal_scales": dict(REGISTERED_SCALES),
        "scalar_priors": {name: dict(spec) for name, spec in PRIORS.items()},
        "truth": {name: float(value) for name, value in TRUE_VALUES.items()},
        "fixed_in_stage_6d": ["segmentation_boundaries (oracle)", "skill_labels (oracle)",
                              "epsilon", "q_0 = zeros(m)"],
        "tau_in_model": False,
        "assessor_level_in_model": False,
        "h_is_derived_not_state": True,
        "second_prior_on_H": False,
        "spec_divergences": [
            {"section": "2",
             "brief_assumes": "U^(0)/U^(a) assessor hierarchy with a tau dependence",
             "frozen_reality": "a single U in R^{5x2}; no assessor level, no tau",
             "resolution": "use the frozen state, as the brief's own fallback clause "
                           "directs; the assessor-residual density and the hierarchical "
                           "QMC construction have no referent and are not built"},
            {"section": "4",
             "brief_assumes": "rho ~ Beta(1, 1/6)",
             "frozen_reality": "rho ~ Beta(1, 1) truncated at 1 - 5e-3, i.e. "
                               "Uniform(0, 0.995), from StatisticalUtils.dRprior with "
                               "stage5.RHO_PRIOR = 1.0",
             "resolution": "the brief defers to the final Stage 6C configuration, which "
                           "establishes this prior"},
            {"section": "4",
             "brief_assumes": "rho scaling proposal delta ~ U(d_r, 1/d_r) with ratio "
                              "-log(delta)",
             "frozen_reality": "logit random walk carrying log(rho(1-rho))",
             "resolution": "sections 6 and 7.2 require reusing the frozen Stage 6C rho "
                           "update and proving numerical equality with it; adopting the "
                           "scaling proposal would fail that gate. The scaling "
                           "proposal's density ratio is implemented and tested as a "
                           "non-production utility so the identity is still pinned"},
        ],
    }


def config_hash() -> str:
    payload = json.dumps(frozen_config(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_stage6c_unchanged() -> dict:
    """The Stage 6C freeze must still hash to its registered value."""
    observed = stage6c_hash()
    if observed != STAGE6C_CONFIG_HASH:
        raise AssertionError(
            f"Stage 6C configuration has drifted: expected {STAGE6C_CONFIG_HASH}, "
            f"observed {observed}")
    return {"stage6c_config_hash": observed, "stage6b_config_hash": STAGE6B_CONFIG_HASH}


def load_stage6d_dataset() -> FrozenStage6D:
    assert_stage6c_unchanged()
    frozen = load_stage6c_dataset()
    return FrozenStage6D(frozen.train, frozen.heldout, frozen.u_true, frozen.epsilon,
                         dict(frozen.truth), frozen_config())
