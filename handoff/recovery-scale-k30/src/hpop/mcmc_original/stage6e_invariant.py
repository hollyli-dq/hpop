"""Stage 6E2 — permutation-invariant convergence summaries, and why they are required.

## The Stage 6E2 posterior is exactly label-exchangeable

For any permutation `Q` of the `K` skills,

    pi' = Q pi,    P' = Q P Q^T,    U'_k = U_{Q^{-1}(k)},    z' = Q(z)

leaves the posterior **unchanged**. This is not an approximation and not a near-symmetry:
`pi` and `P` are inferred under *symmetric* Dirichlet(eta = 1) priors, the `U_k` are iid
under an exchangeable Gaussian prior, and the likelihood reads a block only through the
`U` of whichever skill the block carries. So the posterior has `K!` exactly equivalent
modes, and

    Rhat(pi_1),    Rhat(P_12),    Rhat(relation count of skill 2)

are statistics of a quantity the model does not identify. Four chains sitting in four
different label permutations would make them fail no matter how perfectly the sampler had
converged, and — worse in the other direction — they could pass by accident if chains
happened to agree on a labelling while disagreeing about everything that matters.

This differs from Stage 6E1B on purpose. There `pi` and `P` were *fixed* and deliberately
asymmetric, precisely so that no relabelling was a symmetry and per-skill summaries were
identified; `label_permutation_audit` checked it before any comparison was made. Here they
are inferred, so the symmetry is back and the diagnostics must change with it.

## What replaces them

Every quantity below is invariant under `Q`, by construction rather than by hope:

| summary | why it is invariant |
|---|---|
| `sorted(pi)` | sorting discards the labelling |
| eigenvalues of `P` | `Q P Q^T` is a similarity transform |
| singular values of `P` | likewise |
| sorted row entropies of `P` | permuting rows permutes the multiset of entropies |
| transition-count spectrum | `Q C Q^T` is a similarity transform |
| total number of segments | no skill index appears |
| total relation count | sum over skills |
| sorted per-skill relation counts | sorting discards the labelling |
| posterior co-clustering | "same skill" is a relation between occurrences, not a name |
| log posterior | invariant by the symmetry above |
| `beta, omega, lambda_rep, lambda_back, rho` | carry no skill index |

`assert_invariance` verifies the claim numerically on a random permutation instead of
asserting it, because a summary that is *believed* invariant and is not would hide exactly
the failure these statistics exist to expose.

## The division of labour with recovery

Alignment to the true labels is a **recovery** device and never a convergence one:

    alignment may be used for recovery;
    it may never be used to conceal a convergence problem.

Hungarian matching against the generating truth can make four chains in four different
permutations *look* identical, which would turn a genuine multimodality into an apparent
pass. Convergence therefore uses only the table above, and `stage6e_diagnostics`
alignment is applied afterwards, to recovery quantities alone.
"""

from __future__ import annotations

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.stage6c_diagnostics import convergence_block

__all__ = [
    "sorted_pi", "transition_eigenvalues", "transition_singular_values",
    "sorted_row_entropies", "invariant_summaries", "assert_invariance",
    "INVARIANT_GATE_NAMES",
]

INVARIANT_GATE_NAMES = (
    "sorted_pi_0", "sorted_pi_1", "sorted_pi_2",
    "P_eigenvalue_0", "P_eigenvalue_1", "P_eigenvalue_2",
    "P_singular_value_0", "P_singular_value_1", "P_singular_value_2",
    "P_row_entropy_0", "P_row_entropy_1", "P_row_entropy_2",
    "transition_count_spectrum", "total_segments", "total_relation_count",
    "sorted_relation_count_0", "sorted_relation_count_1", "sorted_relation_count_2",
    "log_posterior", "beta", "omega", "lambda_rep", "lambda_back", "rho",
)


def sorted_pi(pi_draws: np.ndarray) -> np.ndarray:
    """`(chains, draws, K)` -> sorted ascending along K."""
    return np.sort(np.asarray(pi_draws, dtype=float), axis=-1)


def transition_eigenvalues(transition_draws: np.ndarray) -> np.ndarray:
    """Eigenvalues of `P`, sorted by magnitude. `Q P Q^T` is a similarity transform."""
    p = np.asarray(transition_draws, dtype=float)
    values = np.linalg.eigvals(p)
    order = np.argsort(np.abs(values), axis=-1)
    return np.take_along_axis(np.real(values), order, axis=-1)


def transition_singular_values(transition_draws: np.ndarray) -> np.ndarray:
    return np.linalg.svd(np.asarray(transition_draws, dtype=float),
                         compute_uv=False)


def sorted_row_entropies(transition_draws: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row of `P`, sorted. Permuting rows permutes the multiset."""
    p = np.asarray(transition_draws, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, -p * np.log(p), 0.0)
    return np.sort(terms.sum(axis=-1), axis=-1)


def invariant_summaries(data: dict, n_skills: int) -> dict:
    """Every registered permutation-invariant series, shaped `(chains, draws)`."""
    pi = sorted_pi(data["pi_draws"])
    eigenvalues = transition_eigenvalues(data["transition_draws"])
    singular = transition_singular_values(data["transition_draws"])
    entropies = sorted_row_entropies(data["transition_draws"])
    relation = np.sort(np.asarray(data["relation_counts"], dtype=float), axis=-1)

    out = {}
    for k in range(n_skills):
        out[f"sorted_pi_{k}"] = pi[:, :, k]
        out[f"P_eigenvalue_{k}"] = eigenvalues[:, :, k]
        out[f"P_singular_value_{k}"] = singular[:, :, k]
        out[f"P_row_entropy_{k}"] = entropies[:, :, k]
        out[f"sorted_relation_count_{k}"] = relation[:, :, k]
    out["total_segments"] = np.asarray(data["segment_counts"], dtype=float).sum(axis=2)
    out["total_relation_count"] = np.asarray(data["relation_counts"],
                                             dtype=float).sum(axis=2)
    out["log_posterior"] = np.asarray(data["log_target"], dtype=float)
    for name in ("beta", "omega", "lambda_rep", "lambda_back", "rho"):
        out[name] = np.asarray(data[f"scalar_{name}"], dtype=float)
    return out


def assert_invariance(data: dict, n_skills: int, seed: int = 0) -> dict:
    """Apply a random relabelling and check every summary is unchanged.

    Verified, not assumed. A summary that is believed invariant and is not would let four
    chains in four different label permutations look like a convergence failure, or let a
    genuine failure look like a pass.
    """
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_skills)
    inverse = np.argsort(permutation)

    permuted = dict(data)
    permuted["pi_draws"] = np.asarray(data["pi_draws"])[:, :, permutation]
    transition = np.asarray(data["transition_draws"])
    permuted["transition_draws"] = transition[:, :, permutation, :][:, :, :, permutation]
    permuted["relation_counts"] = np.asarray(data["relation_counts"])[:, :, permutation]

    base = invariant_summaries(data, n_skills)
    other = invariant_summaries(permuted, n_skills)
    worst = {name: float(np.abs(base[name] - other[name]).max()) for name in base}
    return {"permutation": permutation.tolist(), "inverse": inverse.tolist(),
            "max_absolute_difference": worst,
            "worst_overall": float(max(worst.values())),
            "pass": bool(max(worst.values()) < 1e-9)}


def invariant_convergence(data: dict, n_skills: int, threshold: float = 1.01) -> dict:
    """R-hat / ESS on every registered invariant summary, with the gate verdict."""
    summaries = invariant_summaries(data, n_skills)
    blocks, gates, frozen = {}, {}, {}
    for name, series in summaries.items():
        series = np.asarray(series, dtype=float)
        block = convergence_block(series, name)
        blocks[name] = block

        # A coordinate whose WITHIN-chain variance is zero in one or more chains, while
        # the chains disagree, is not "badly mixing" — it is frozen. R-hat then divides a
        # non-zero between-chain variance by ~0 and returns an astronomical number that
        # says nothing beyond "these are different". Naming the condition is far more
        # useful than reporting 7e15, so it is detected and reported explicitly.
        within = series.std(axis=1)
        stuck = int((within == 0.0).sum())
        if stuck and float(np.ptp(series.mean(axis=1))) > 0:
            frozen[name] = {
                "n_chains_with_zero_within_chain_variance": stuck,
                "per_chain_mean": series.mean(axis=1).tolist(),
                "per_chain_within_sd": within.tolist(),
                "interpretation": f"{stuck} of {series.shape[0]} chains never moved this "
                                  "coordinate at all, and the chains disagree about its "
                                  "value. This is an absorbing state, not slow mixing.",
            }

        # `frozen_in_some_chains` is set on EVERY branch. A frozen coordinate is precisely
        # the case whose R-hat comes back non-finite, so attaching the flag only where
        # R-hat is finite would omit it exactly where it matters most.
        value = None if block.get("degenerate") else block.get("rhat")
        if value is None:
            gates[name] = {"value": None, "threshold": threshold, "pass": True,
                           "frozen_in_some_chains": name in frozen,
                           "note": "degenerate (constant in EVERY chain); recorded as "
                                   "such, never as 1.0"}
        elif not np.isfinite(value):
            gates[name] = {"value": None, "threshold": threshold, "pass": False,
                           "frozen_in_some_chains": name in frozen,
                           "note": "R-hat undefined (NaN/inf), which is what a frozen "
                                   "coordinate produces: a real between-chain variance "
                                   "divided by ~0. Reported as a failure."}
        else:
            gates[name] = {"value": float(value), "threshold": threshold,
                           "pass": bool(value <= threshold),
                           "frozen_in_some_chains": name in frozen}
    return {"blocks": blocks, "gates": gates, "frozen_coordinates": frozen,
            "all_pass": all(g["pass"] for g in gates.values()),
            "n_gates": len(gates), "n_frozen": len(frozen)}
