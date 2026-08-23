"""Stage 6C — the frozen model, extending the Stage 6B freeze with `U` and `rho`.

Stage 6C changes exactly one thing relative to Stage 6B: `U` and `rho` stop being held at
their registered values and become inferred. The corpus, the likelihood branch, `epsilon`,
the oracle boundaries, `q_0` and the scalar priors are the Stage 6B ones, unchanged and
re-verified here by hash.

## The target is continuous in U

    p(U, rho | Y, fixed scalars)  ∝  p(Y | h(U), fixed scalars) · p(U | rho) · p(rho)

`U ∈ R^{m×d}` is a continuous latent utility matrix and `h(U)` is the *derived* discrete
partial order — the intersection of the `d` column orderings,

    i > j   iff   U[i, r] > U[j, r]  for every coordinate r.

The chain's state is `U` itself, never a poset identifier. Nothing here integrates `U` out
or replaces the target with a prior over orders.

## The normalisation that matters is the Gaussian determinant

`p(U | rho) = prod_j N_d(U[j,:]; 0, Sigma_rho)` with
`Sigma_rho = (1-rho) I_d + rho 1 1^T`, so `rho` is the correlation **between the d latent
columns within a row**; rows are iid. (A correlation between item rows would need an
`(m, m)` matrix or a matrix-normal prior, and the registered model has neither.)

There is no sum over legal partial orders anywhere in this model, so there is no
combinatorial `Z_m(rho)` to compute. The `rho`-dependent normaliser is the Gaussian one:

    log|Sigma_rho| = (d-1) log(1-rho) + log(1 + (d-1) rho)

contributing `-(m/2) log|Sigma_rho|`. It **cancels** in a `U` update at fixed `rho`
(numerator and denominator share `Sigma_rho`) and **does not cancel** in a `rho` update at
fixed `U`. `sampler_u.log_u_prior` includes it — verified against `scipy`'s MVN logpdf to
1e-9, shown to integrate to 1, and guarded by a negative control in
`test_stage6c_structural_prior.py` in which a quadratic-form-only variant is shown to
shift the `rho` posterior.

`rho` does not enter the likelihood at all; it acts only through `p(U | rho)`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from hpop.mcmc_original.sampler_u import log_u_prior, sigma_rho_matrix
from hpop.mcmc_original.stage6b_frozen import (
    EPSILON, STAGE6B_MODEL_ID, assert_likelihood_branch, config_hash as stage6b_hash,
    frozen_config as stage6b_frozen_config, load_frozen_dataset,
)

__all__ = [
    "STAGE6C_MODEL_ID", "RHO_PRIOR_FAC", "RHO_TOL", "RHO_LOWER", "RHO_UPPER",
    "STAGE6B_CONFIG_HASH", "log_rho_prior", "rho_to_unconstrained",
    "rho_from_unconstrained", "log_jacobian_rho", "log_structural_prior",
    "log_det_sigma_rho", "frozen_config", "config_hash", "assert_stage6b_unchanged",
    "load_stage6c_dataset", "FrozenStage6C", "ACTIVE_6C1", "ACTIVE_6C2", "SWEEP_ORDER",
    "SIGMA_U",
]

STAGE6C_MODEL_ID = "recurrent-rfs-latent-product-order-v1"

# Registered rho prior: Beta(1, fac) truncated at 1 - tol, per the vendored
# StatisticalUtils.dRprior with stage5.RHO_PRIOR = 1.0 as the `fac` argument. Read from
# the code, not assumed: with fac = 1 this is Uniform(0, 0.995).
RHO_PRIOR_FAC = 1.0
RHO_TOL = 5e-3
RHO_LOWER = 0.0
RHO_UPPER = 1.0 - RHO_TOL

# Stage 2A row-proposal scale, reused unchanged.
SIGMA_U = 0.5

STAGE6B_CONFIG_HASH = "9ad850f22065d85f6cfd855443395d06b8a566e8c8dcdd4f9d85b1f031e911cc"

ACTIVE_6C1 = ("U", "rho")
ACTIVE_6C2 = ("U", "rho", "beta")
SWEEP_ORDER = ("U", "rho", "beta")


# --------------------------------------------------------------------------- rho prior
def log_rho_prior(rho: float) -> float:
    """Registered `rho` prior: Beta(1, fac) truncated at `1 - tol`, renormalised."""
    if not (RHO_LOWER < rho < RHO_UPPER):
        return -math.inf
    density = stats.beta(1.0, RHO_PRIOR_FAC)
    return float(density.logpdf(rho) - density.logcdf(RHO_UPPER))


def rho_to_unconstrained(rho: float) -> float:
    """`z = logit(rho)`. The random walk lives here; the prior lives on `rho`."""
    if not (0.0 < rho < 1.0):
        raise ValueError(f"rho must lie in (0, 1) to transform, got {rho}")
    return math.log(rho) - math.log1p(-rho)


def rho_from_unconstrained(z: float) -> float:
    return float(1.0 / (1.0 + math.exp(-z)))


def log_jacobian_rho(rho: float) -> float:
    """``log |d rho / d z|`` for ``z = logit(rho)``, namely ``log(rho (1 - rho))``.

    Carried explicitly in the rho acceptance ratio, alongside — and distinct from — the
    Gaussian determinant `-(m/2) log|Sigma_rho|` that lives inside `p(U | rho)`. The two
    are different corrections and both are needed; each has its own negative control.
    """
    if not (0.0 < rho < 1.0):
        return -math.inf
    return math.log(rho) + math.log1p(-rho)


def log_det_sigma_rho(d: int, rho: float) -> float:
    """``log det Sigma_rho = (d-1) log(1-rho) + log(1 + (d-1) rho)``.

    The rho-dependent normaliser. Already inside `sampler_u.log_u_prior`; this closed form
    exists so a test can confirm the two agree and so the term can be isolated.
    """
    return (d - 1) * math.log1p(-rho) + math.log1p((d - 1) * rho)


def log_structural_prior(u: np.ndarray, rho: float) -> float:
    """`log p(U | rho)` — the registered Stage 2A prior, reused, not reimplemented."""
    u = np.asarray(u, dtype=float)
    d = u.shape[1]
    lower = -1.0 / (d - 1) if d > 1 else -1.0
    if not (lower < rho < 1.0):
        return -math.inf
    return log_u_prior(u, rho)


# ------------------------------------------------------------------------ frozen config
@dataclass(frozen=True)
class FrozenStage6C:
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
        """The embedding dimension d — NOT the role count and NOT the number of skills."""
        return int(self.u_true.shape[1])

    @property
    def n_skills(self) -> int:
        """Stage 6 has a single skill: one U, oracle blocks, oracle labels."""
        return 1


def frozen_config() -> dict:
    base = stage6b_frozen_config()
    return {
        "stage": "6C",
        "model_id": STAGE6C_MODEL_ID,
        "stage6b_model_id": STAGE6B_MODEL_ID,
        "stage6b_config_hash": stage6b_hash(),
        "target": "p(U, rho[, beta] | Y) proportional to p(Y | h(U), fixed) p(U | rho) "
                  "p(rho) [p(beta)] -- continuous in U; U is never integrated out and the "
                  "chain state is never a poset identifier",
        "inherits_from_stage6b": {
            "likelihood_branch": base["likelihood_branch"],
            "dataset": base["dataset"], "epsilon": base["epsilon"],
            "boundaries": base["boundaries"], "numerics": base["numerics"],
            "scalar_priors": base["priors"], "truth": base["truth"],
        },
        "structural_model": {
            "latent": "U in R^{m x d} continuous latent utility matrix",
            "induced_order": "h(U): i > j iff U[i,r] > U[j,r] for every r; the "
                             "intersection of the d column orderings",
            "h_depends_on": "U only -- not on beta",
            "legality": "every real U induces a legal strict partial order; no acyclicity "
                        "check exists or is needed",
            "representation": "precedence_from_u returns the TRANSITIVE CLOSURE",
            "mcmc_state": "U itself (continuous); the poset is a derived label",
            "prior": "rows iid Normal(0, Sigma_rho), Sigma_rho = (1-rho) I_d + rho 11^T",
            "rho_is": "correlation between the d latent COLUMNS within a row; rows are "
                      "iid. Not a row covariance -- that would need a matrix-normal prior",
            "prior_normalised": True,
            "rho_dependent_normaliser": "-(m/2) log|Sigma_rho|, present in log_u_prior; "
                                        "cancels in a U update at fixed rho, does NOT "
                                        "cancel in a rho update at fixed U",
            "combinatorial_normaliser_needed": False,
            "combinatorial_normaliser_reason": "the prior is a density on R^{m x d}, not a "
                                               "score over a set of partial orders, so no "
                                               "sum over legal posets arises",
            "rho_enters_likelihood": False,
            "tau_in_model": False,
        },
        "rho_prior": {"family": "beta", "a": 1.0, "b": RHO_PRIOR_FAC,
                      "truncated_at": RHO_UPPER, "tol": RHO_TOL,
                      "support": f"({RHO_LOWER}, {RHO_UPPER})",
                      "renormalised": True,
                      "source": "vendored StatisticalUtils.dRprior with stage5.RHO_PRIOR"},
        "rho_transform": {"name": "logit", "jacobian": "log(rho (1 - rho))"},
        "u_proposal": {"kernel": "sampler_u.propose_row (Stage 2A), reused unchanged",
                       "form": "U'[j,:] = U[j,:] + sigma_U * Normal(0, I_d)",
                       "sigma_u": SIGMA_U, "symmetric": True, "log_hastings": 0.0},
        "n_skills": 1,
        "rho_true_exists": False,
        "rho_true_reason": "U_TRUE is hand specified in recurrent_synthetic.py, not drawn "
                           "from p(U | rho); no RHO_TRUE constant exists",
        "active_6c1": list(ACTIVE_6C1), "active_6c2": list(ACTIVE_6C2),
        "sweep_order": list(SWEEP_ORDER),
        "never_inferred_in_stage_6c": ["segmentation_boundaries", "skill_labels",
                                       "epsilon", "omega", "lambda_rep", "lambda_back"],
    }


def config_hash() -> str:
    payload = json.dumps(frozen_config(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_stage6b_unchanged() -> dict:
    """The Stage 6B freeze must still hash to its registered value."""
    observed = stage6b_hash()
    if observed != STAGE6B_CONFIG_HASH:
        raise AssertionError(
            f"Stage 6B configuration has drifted: expected {STAGE6B_CONFIG_HASH}, "
            f"observed {observed}")
    branch = assert_likelihood_branch()
    if branch["model_id"] != STAGE6B_MODEL_ID:
        raise AssertionError(f"likelihood branch changed: {branch['model_id']}")
    return {"stage6b_config_hash": observed, "likelihood_branch": branch,
            "epsilon": EPSILON}


def load_stage6c_dataset() -> FrozenStage6C:
    assert_stage6b_unchanged()
    frozen = load_frozen_dataset()
    return FrozenStage6C(frozen.train, frozen.heldout, frozen.u_true, frozen.epsilon,
                         dict(frozen.truth), frozen_config())
