"""Stage 6E — the frozen unknown-boundary model.

Stage 6E removes the two oracles Stage 6D still had: the block boundaries `S` and the
skill labels `z`. Everything else is the Stage 6D model, unchanged and re-verified here
by hash.

    p(S_{1:N}, z_{1:N}, U_{1:K}, rho, beta, omega, lambda_rep, lambda_back, pi, P | x)
        proportional to
    prod_n w(S_n, z_n) x p(U | rho) p(rho) p(beta) p(omega) p(lambda_rep) p(lambda_back)
                        x p(pi) p(P)

with the per-trace segmentation weight

    log w(S_n, z_n) = log pi_{z_n1}
                    + sum_l log p_RFS(x^(n)_{a_l:b_l} | U_{z_l}, scalars, epsilon)
                    + (J_n - L_n) log(1 - delta_B) + (L_n - 1) log delta_B
                    + sum_{l>=2} log P[z_{l-1}, z_l]

## Four audit findings that determine everything downstream

Each was established by reading the registered code before any Stage 6E inference was
written, and each is recorded here rather than left implicit.

**1. `pi` and `P` are INFERRED, using the frozen Stage 3 updates.** They are part of the
registered final target: `targets.log_target_segmentation` takes `log_pi` and
`log_transition`, and `transitions.py` carries the validated conjugate updates that
Stage 3 verified. They were absent from Stage 6D only because that stage had `K = 1`
skill and oracle labels, so no path prior existed to infer. Stage 6E restores them rather
than inventing them.

**2. `P` forbids self-transitions.** `transitions.allowed_next(h, K)` excludes `h`, and
`sample_transition_matrix` leaves the diagonal at exactly zero. So consecutive segments
must carry *different* skills, and a segmentation with an adjacent repeated label has
log target `-inf`. This is a registered constraint of the parent model, not a Stage 6E
choice, and it is what keeps the segmentation identifiable: without it, every segment
could be split arbitrarily at no cost in the path prior.

**3. There is no terminal transition.** `log_path_prior` sums `log pi[z_1]` and then one
`log P[z_{l-1}, z_l]` per adjacent pair. Nothing is emitted for ending the path, and none
is added here.

**4. The Stage 5 compatibility predicate does NOT carry over, and this is the one place
Stage 6E cannot reuse a Stage 5 component verbatim.** `proposals.compatible_skills` calls
`toy.map_cpa_block_to_roles`, which requires the block's CPA multiset to equal the skill's
role multiset exactly — so a legal Stage 5 block is a *permutation* of its skill's roles,
visiting each exactly once. The recurrent model is defined by repeats: `lambda_rep` and
`lambda_back` exist precisely because a role can recur, and the Stage 6D corpus is blocks
of `T = 20` over `m = 5` roles. Under the Stage 5 predicate **no recurrent block is ever
compatible with any skill**, so the Stage 5 move generators would return an empty
neighbourhood for every state.

The registered escape clause applies ("unless the existing kernel cannot represent the
registered state space"), and it is taken as narrowly as possible: Stage 6E replaces the
*legality predicate* and nothing else. The move semantics, the neighbourhood counting and
the Hastings ratio are the Stage 5 objects, reused rather than re-derived — see
`recurrent_segmentation.Stage6EMoveKernel`, which subclasses `LocalMoveKernel` and
overrides `neighbours` alone.

## Registered constants
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from hpop.mcmc_original.recurrent_scalar_posterior import PRIORS, TRUE_VALUES
from hpop.mcmc_original.stage6d_frozen import (
    ACTIVE_6D, N_LATENT_COLUMNS, REGISTERED_SCALES as STAGE6D_SCALES, SCALAR_ORDER,
    config_hash as stage6d_hash, frozen_config as stage6d_frozen_config,
)

__all__ = [
    "STAGE6E_MODEL_ID", "STAGE6D_CONFIG_HASH", "SWEEP_ORDER_6E", "ACTIVE_6E",
    "DELTA_B", "MAX_BLOCK_WIDTH", "MIN_BLOCK_WIDTH", "N_SKILLS", "N_ROLES",
    "LATENT_DIM", "ETA_TRANSITION", "ETA_INITIAL", "INFER_PI_P",
    "SELF_TRANSITIONS_ALLOWED", "TERMINAL_TRANSITION", "REGISTERED_SCALES",
    "frozen_config", "config_hash", "assert_stage6d_unchanged",
    "log_boundary_prior_6e", "log_path_prior_6e",
]

STAGE6E_MODEL_ID = "recurrent-rfs-unknown-boundary-joint-v1"

# The Stage 6D freeze this stage extends. Verified by hash at call sites.
STAGE6D_CONFIG_HASH = "ebd5effd5b32e68aeb61df044bfbdcbef998c01955673ae1c077531be3d2ac73"

# ------------------------------------------------------------------ dimensions / model
N_ROLES = 5                 # m: role occurrences inside one skill's partial order
LATENT_DIM = N_LATENT_COLUMNS   # d = 2 latent utility columns
N_SKILLS = 3                # K: reusable skill types. Stage 6D had K = 1.

# Boundary prior. delta_B is the per-position probability of a cut, over the J-1 internal
# positions, exactly as `targets.log_boundary_prior` counts them.
DELTA_B = 0.15

# Block-width bounds. A block must be long enough for the recurrent state to mean
# anything and short enough that the candidate set stays finite.
MIN_BLOCK_WIDTH = 3
MAX_BLOCK_WIDTH = 12

# Conjugate hyperparameters for the frozen Stage 3 updates.
ETA_TRANSITION = 1.0
ETA_INITIAL = 1.0

# ---------------------------------------------------------------- audit conclusions
INFER_PI_P = True
SELF_TRANSITIONS_ALLOWED = False        # transitions.allowed_next excludes h
TERMINAL_TRANSITION = False             # log_path_prior emits none

SWEEP_ORDER_6E = ("S_z", "pi_P") + ACTIVE_6D          # (S,z) -> (pi,P) -> U -> rho -> ...
ACTIVE_6E = SWEEP_ORDER_6E

# Stage 6D2's frozen scales are the STARTING POINT for the Stage 6E pilot, never its
# conclusion: a target with latent (S, z) is a different target, and Stage 6D established
# that a proposal scale is a property of the target rather than of the kernel.
REGISTERED_SCALES = dict(STAGE6D_SCALES)


# --------------------------------------------------------------------- prior pieces
def log_boundary_prior_6e(trace_length: int, n_segments: int,
                          delta_b: float = DELTA_B) -> float:
    """`(L-1) log delta_B + (J-L) log(1-delta_B)`, over the `J-1` internal positions.

    Delegates to the Stage 5 implementation so the two cannot drift apart.
    """
    from hpop.mcmc_original.targets import log_boundary_prior
    return log_boundary_prior(trace_length, n_segments, delta_b)


def log_path_prior_6e(path, log_pi: np.ndarray,
                      log_transition: np.ndarray | None) -> float:
    """`log pi[z_1] + sum_{l>=2} log P[z_{l-1}, z_l]`. No terminal transition."""
    from hpop.mcmc_original.targets import log_path_prior
    return log_path_prior(path, log_pi, log_transition)


# ------------------------------------------------------------------------ frozen config
def frozen_config() -> dict:
    base = stage6d_frozen_config()
    return {
        "stage": "6E",
        "model_id": STAGE6E_MODEL_ID,
        "stage6d_model_id": base["model_id"],
        "stage6d_config_hash": stage6d_hash(),
        "target": "p(S, z, U, rho, beta, omega, lambda_rep, lambda_back, pi, P | x) "
                  "proportional to prod_n w(S_n, z_n) p(U|rho) p(rho) p(beta) p(omega) "
                  "p(lambda_rep) p(lambda_back) p(pi) p(P)",
        "segmentation_weight": "log pi[z_1] + sum_l log p_RFS(x[a_l:b_l] | U_{z_l}, "
                               "scalars, epsilon) + (J-L) log(1-delta_B) "
                               "+ (L-1) log delta_B + sum_{l>=2} log P[z_{l-1}, z_l]",
        "inherits_from_stage6d": {
            "likelihood_branch": base["inherits_from_stage6c"]["likelihood_branch"],
            "epsilon": base["inherits_from_stage6c"]["epsilon"],
            "numerics": base["inherits_from_stage6c"]["numerics"],
            "rho_prior": base["inherits_from_stage6c"]["rho_prior"],
            "rho_transform": base["inherits_from_stage6c"]["rho_transform"],
            "u_proposal": base["inherits_from_stage6c"]["u_proposal"],
            "scalar_priors": base["scalar_priors"],
        },
        "dimensions": {
            "m_roles": N_ROLES, "d_latent_columns": LATENT_DIM, "K_skills": N_SKILLS,
            "note": "Stage 6D had K = 1 skill and oracle labels; Stage 6E has K = "
                    f"{N_SKILLS} reusable skills whose labels are latent",
        },
        "newly_latent_in_stage_6e": ["segmentation S_{1:N}", "skill labels z_{1:N}"],
        "still_inferred": list(ACTIVE_6D),
        "sweep_order": list(SWEEP_ORDER_6E),
        "fixed_in_stage_6e": {
            "epsilon": base["inherits_from_stage6c"]["epsilon"],
            "n_skills": N_SKILLS,
            "delta_B": DELTA_B,
            "min_block_width": MIN_BLOCK_WIDTH,
            "max_block_width": MAX_BLOCK_WIDTH,
            "q_0": "zeros(m), reset at the start of EVERY candidate block",
            "tau": "does not exist in this model (inherited from Stage 6D)",
        },
        "pi_P": {
            "inferred": INFER_PI_P,
            "why": "they belong to the registered final target: "
                   "targets.log_target_segmentation takes log_pi and log_transition, and "
                   "transitions.py carries the conjugate updates Stage 3 validated. They "
                   "were absent from Stage 6D only because K = 1 left no path prior to "
                   "infer.",
            "update": "frozen Stage 3 conjugate Dirichlet updates "
                      "(transitions.dirichlet_posterior_params / "
                      "sample_transition_matrix)",
            "eta_transition": ETA_TRANSITION, "eta_initial": ETA_INITIAL,
            "self_transitions_allowed": SELF_TRANSITIONS_ALLOWED,
            "self_transition_note": "transitions.allowed_next(h, K) excludes h and the "
                                    "sampled diagonal is exactly 0, so adjacent segments "
                                    "must carry different skills and a segmentation that "
                                    "repeats a label across a boundary has log target "
                                    "-inf. Registered in the parent, not chosen here.",
            "terminal_transition": TERMINAL_TRANSITION,
        },
        "segmentation_kernel": {
            "reused": "proposals.LocalMoveKernel — proposal_distribution, proposal_prob, "
                      "sample_proposal and mh_local_step are used unchanged",
            "move_types": ["relabel", "split", "merge", "shift"],
            "what_changed": "the legality predicate only",
            "why_it_had_to_change":
                "proposals.compatible_skills requires the block's CPA multiset to equal "
                "the skill's role multiset, so a legal Stage 5 block is a permutation of "
                "its skill's roles visiting each exactly once. The recurrent model is "
                "defined by repeats — lambda_rep and lambda_back exist because roles "
                "recur — so under that predicate NO recurrent block is compatible with "
                "any skill and every neighbourhood would be empty.",
            "hastings_ratio": "unchanged: log q(S'->S) - log q(S->S') from the Stage 5 "
                              "neighbourhood counting, never assumed symmetric",
        },
        "structural_model": base["inherits_from_stage6c"]["structural_model"],
        "h_is_derived_not_state": True,
        "second_prior_on_H": False,
        "scalar_priors": {name: dict(spec) for name, spec in PRIORS.items()},
        "scalar_truth": {name: float(value) for name, value in TRUE_VALUES.items()},
        "starting_proposal_scales": dict(REGISTERED_SCALES),
        "starting_scales_note": "the frozen Stage 6D2 scales, used ONLY as the pilot's "
                                "centre. A target with latent (S, z) is a different "
                                "target and carries no efficiency guarantee.",
    }


def config_hash() -> str:
    payload = json.dumps(frozen_config(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_stage6d_unchanged() -> dict:
    observed = stage6d_hash()
    if observed != STAGE6D_CONFIG_HASH:
        raise AssertionError(
            f"Stage 6D configuration has drifted: expected {STAGE6D_CONFIG_HASH}, "
            f"observed {observed}")
    return {"stage6d_config_hash": observed}
