"""Stage 6B — the frozen registered model, shared unchanged by 6B1, 6B2 and 6B3.

Stages 6B2 and 6B3 must differ from 6B1 in exactly one respect: which scalar
coordinates are inferred. Everything else — the corpus, ``U_TRUE``, the boundary
convention, ``epsilon``, the priors, the likelihood branch, the initial recurrent state,
the numerical stabilisation and the seed conventions — is registered here and hashed, so
a silent drift between stages becomes a test failure rather than a puzzling posterior.

The likelihood branch is the **utility-weighted frontier**: feasibility multiplies, and
``beta * log(1 + S)`` sits in the exponent alongside the two penalties. It is *not* the
uniform-frontier variant. `assert_likelihood_branch` pins that with hand-computed values
so the two cannot be interchanged by accident.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_rfs import (
    recurrent_back_costs, recurrent_feasibility, recurrent_stale_successor_counts,
    recurrent_structural_weights, recurrent_successor_utilities, sigmoid,
)
from hpop.mcmc_original.recurrent_scalar_posterior import PRIORS, TRUE_VALUES

__all__ = [
    "STAGE6B_MODEL_ID", "EPSILON", "PARAMETER_ORDER", "SWEEP_ORDER",
    "TRANSFORM", "SEED_CONVENTIONS", "FrozenStage6B",
    "frozen_config", "config_hash", "assert_likelihood_branch", "load_frozen_dataset",
    "to_unconstrained", "from_unconstrained", "log_jacobian_to_unconstrained",
]

STAGE6B_MODEL_ID = "recurrent-rfs-utility-weighted-frontier-v1"
EPSILON = 0.02

# The registered coordinate order. A complete sweep visits them in exactly this order.
PARAMETER_ORDER = ("beta", "omega", "lambda_rep", "lambda_back")
SWEEP_ORDER = PARAMETER_ORDER

# Coordinate transforms, chosen to match the Stage 6B1 proposal parameterisations rather
# than invented here: a log-scale random walk is a symmetric walk on ``log theta``.
TRANSFORM = {"beta": "log", "omega": "identity",
             "lambda_rep": "log", "lambda_back": "log"}

SEED_CONVENTIONS = {
    "generator_seed_train": 0,
    "generator_seed_heldout": 10_000,
    "stage6b1_mcmc_seed": 20_250_811,
    "stage6b1_parameter_offset": 1_000,
    "stage6b1_pilot_offset": 900_000,
    "stage6b2_mcmc_seed": 20_250_812,
    "stage6b3_mcmc_seed": 20_250_813,
    "rule": "one independent numpy Generator per chain; no global RNG state is read",
}

BOUNDARY_CONVENTION = {
    "segmentation": "none — boundaries are known and fixed",
    "blocks": "every block is one independent trace of fixed length T",
    "skills_per_block": 1,
    "q_reset": "q_0 = 0 at the start of every block; q never crosses a block boundary",
    "conditions_on_T": True,
    "duration_model": None,
}

NUMERICAL_CONVENTIONS = {
    "exponent_stabilisation": "row-wise max subtraction before exp; cancels in the "
                              "normalisation",
    "mixture_floor": "p = (1 - epsilon) * structural + epsilon / m",
    "initial_state": "q_0 = zeros(m)",
    "feasibility_empty_product": 1.0,
    "dtype": "float64",
}


@dataclass(frozen=True)
class FrozenStage6B:
    """The registered corpus and constants, as arrays ready for a likelihood call."""

    train: np.ndarray            # (n_blocks, T) integer role indices
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
        """The dimension of the latent embedding — NOT the number of roles."""
        return int(self.u_true.shape[1])


def frozen_config() -> dict:
    """The registered configuration, in a canonical form suitable for hashing."""
    # Keep the historical generator/truth dependency out of a live inference import.
    # This function is a legacy corpus-registration audit, not part of the sealed
    # FULL-LATENT runtime path.
    from hpop.mcmc_original.recurrent_synthetic import U_TRUE

    return {
        "model_id": STAGE6B_MODEL_ID,
        "likelihood_branch": {
            "name": "utility-weighted frontier",
            "feasibility": "F(x) = prod_{z precedes x} q(z), empty product 1",
            "utility": "Q(x) = log(1 + S(x)),  S(x) = sum_z 1[x precedes z] (1 - q(z))",
            "back_cost": "C_back(x) = sigmoid(omega) * sum_{z != x} 1[x precedes z] q(z)",
            "weights": "W(x) = F(x) * exp(beta*Q(x) - lambda_rep*q(x) "
                       "- lambda_back*C_back(x))",
            "emission": "p(x) = (1 - epsilon) * W(x)/sum(W) + epsilon/m",
            "state_update": "q(y)=1; q(x) *= (1 - sigmoid(omega)) for x downstream of y",
            "not": "uniform-frontier (beta would not multiply a utility term)",
        },
        "dataset": {"mode": "full", "generator_seed": 0, "n_train": 500,
                    "n_heldout": 200, "T": 20},
        "u_true": np.asarray(U_TRUE).tolist(),
        "n_roles": int(np.asarray(U_TRUE).shape[0]),
        "latent_dimension": int(np.asarray(U_TRUE).shape[1]),
        "epsilon": EPSILON,
        "truth": {k: float(v) for k, v in sorted(TRUE_VALUES.items())},
        "priors": {k: PRIORS[k] for k in sorted(PRIORS)},
        "parameter_order": list(PARAMETER_ORDER),
        "sweep_order": list(SWEEP_ORDER),
        "transform": dict(sorted(TRANSFORM.items())),
        "boundaries": BOUNDARY_CONVENTION,
        "numerics": NUMERICAL_CONVENTIONS,
        "seeds": SEED_CONVENTIONS,
        "inferred_in_stage_6b": ["beta", "omega", "lambda_rep", "lambda_back"],
        "never_inferred_in_stage_6b": ["U", "rho", "P", "segmentation_boundaries",
                                       "skill_labels", "epsilon"],
    }


def config_hash() -> str:
    """SHA-256 over the canonical JSON of `frozen_config`."""
    payload = json.dumps(frozen_config(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_likelihood_branch() -> dict:
    """Pin the utility-weighted frontier with values computed by hand.

    A switch to a uniform-frontier model would leave ``F`` unchanged but make the
    structural weights independent of ``beta`` given feasibility, so both the ``Q``
    identity and the beta-sensitivity check below would fail.
    """
    # `U_TRUE` belongs exclusively to this historical registration audit.  Delaying this
    # import means production inference can reuse the likelihood declaration without
    # loading synthetic-generator truth into its process.
    from hpop.mcmc_original.recurrent_synthetic import U_TRUE

    precedence = precedence_from_u(U_TRUE)
    q0 = np.zeros(5)

    feasibility = recurrent_feasibility(precedence, q0)
    if feasibility.tolist() != [1.0, 1.0, 0.0, 0.0, 0.0]:
        raise AssertionError(f"F(q0) must be [1,1,0,0,0], got {feasibility.tolist()}")

    stale = recurrent_stale_successor_counts(precedence, q0)
    if stale.astype(int).tolist() != [3, 0, 2, 1, 0]:
        raise AssertionError(f"S(q0) must be [3,0,2,1,0], got {stale.tolist()}")

    utility = recurrent_successor_utilities(precedence, q0)
    if not np.allclose(utility, np.log1p(stale)):
        raise AssertionError("Q must equal log(1 + S) — utility-weighted frontier")

    # back cost carries the sigmoid of omega, not omega itself
    omega = TRUE_VALUES["omega"]
    q_mid = np.array([1.0, 1.0, 0.5, 0.25, 0.0])
    back = recurrent_back_costs(precedence, q_mid, omega)
    expected_back0 = sigmoid(omega) * (q_mid[2] + q_mid[3] + q_mid[4])
    if abs(float(back[0]) - expected_back0) > 1e-12:
        raise AssertionError(f"C_back(0) must be {expected_back0}, got {float(back[0])}")

    # beta must actually move the structural weights: uniform-frontier would not
    low = recurrent_structural_weights(precedence, q_mid, 0.5, omega, 0.8, 0.25)
    high = recurrent_structural_weights(precedence, q_mid, 3.0, omega, 0.8, 0.25)
    ratio_low = float(low[0] / low[1])
    ratio_high = float(high[0] / high[1])
    if not ratio_high > 1.5 * ratio_low:
        raise AssertionError(
            "beta does not reweight the frontier — this is not the utility-weighted "
            f"branch (ratio {ratio_low:.4f} -> {ratio_high:.4f})")

    return {"feasibility_q0": feasibility.tolist(),
            "stale_counts_q0": stale.astype(int).tolist(),
            "back_cost_check": float(back[0]),
            "beta_weight_ratio_low": ratio_low, "beta_weight_ratio_high": ratio_high,
            "model_id": STAGE6B_MODEL_ID}


def load_frozen_dataset() -> FrozenStage6B:
    """The registered 500-block training corpus and its 200 held-out blocks."""
    from hpop.mcmc_original.recurrent_synthetic import (
        U_TRUE,
        generate_recurrent_dataset,
    )

    assert_likelihood_branch()
    data = generate_recurrent_dataset("full", SEED_CONVENTIONS["generator_seed_train"])
    train = np.array([b.roles for b in data.train], dtype=int)
    heldout = np.array([b.roles for b in data.heldout], dtype=int)
    config = frozen_config()
    if train.shape != (config["dataset"]["n_train"], config["dataset"]["T"]):
        raise AssertionError(f"frozen corpus shape drifted: {train.shape}")
    return FrozenStage6B(train, heldout, np.asarray(data.u_true, dtype=float),
                         EPSILON, dict(TRUE_VALUES), config)


# ------------------------------------------------------------------ coordinate transforms
def to_unconstrained(name: str, value: float) -> float:
    """Map to the scale the Stage 6B1 proposal actually walks on."""
    return math.log(value) if TRANSFORM[name] == "log" else float(value)


def from_unconstrained(name: str, value: float) -> float:
    return math.exp(value) if TRANSFORM[name] == "log" else float(value)


def log_jacobian_to_unconstrained(name: str, value: float) -> float:
    """``log |d(value) / d(unconstrained)|``.

    For a positive parameter with ``z = log theta`` this is ``log theta``: a density in
    ``theta`` becomes ``p(theta) * theta`` as a density in ``z``. Getting this wrong is
    exactly the error the Stage 6B1 negative control demonstrates, so the joint
    references carry it explicitly rather than implicitly.
    """
    return math.log(value) if TRANSFORM[name] == "log" else 0.0
