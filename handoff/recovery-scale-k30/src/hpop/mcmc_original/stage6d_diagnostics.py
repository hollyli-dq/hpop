"""Stage 6D — diagnostics for the joint latent-`U` + four-scalar chains.

Thin by design: rank-normalised split R-hat, bulk/tail ESS, MCSE, the energy-distance
envelope and total variation all come from the Stage 6B/6C modules that validated them.
What is new here is the handling of `H = h(U)` when there is **no fixed catalogue** — the
Stage 6D1 small model and the Stage 6D2 full corpus have different induced-order spaces,
so labels are built from the observed closures and aligned between MCMC and reference by
their canonical bytes rather than by an index into a shared table.

Two conventions are carried over from Stage 6C because they prevent confidently wrong
reports:

* a **constant** trace is reported as `degenerate`, never as a converged R-hat of 1.0;
* a correlation involving a constant coordinate is reported as **undefined with a reason**,
  never as `null` alone and never as 0.
"""

from __future__ import annotations

import math

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.stage6b_joint_diagnostics import (
    calibrate_energy_envelope, energy_distance, standardise,
)
from hpop.mcmc_original.stage6b_mcmc_diagnostics import (
    bulk_ess, ks_distance_to_grid, mcse_mean, rank_normalized_split_rhat, tail_ess,
)
from hpop.mcmc_original.stage6c_diagnostics import convergence_block

__all__ = [
    "h_labels_from_u", "align_h_distributions", "h_total_variation",
    "relation_marginal_from_u", "structural_block", "scalar_block", "dependence_block",
    "mixed_envelope", "column_permutation_audit", "weighted_quantile",
]


# ------------------------------------------------------------------- induced H labels
def h_labels_from_u(u_draws: np.ndarray) -> tuple[np.ndarray, list[bytes], np.ndarray]:
    """Canonical `H` key per draw, the distinct keys, and their closure stack."""
    u_draws = np.asarray(u_draws, dtype=float)
    closures = np.array([precedence_from_u(u) for u in u_draws])
    keys = [c.tobytes() for c in closures]
    index: dict[bytes, int] = {}
    order: list[bytes] = []
    labels = np.empty(len(keys), dtype=np.int32)
    for i, key in enumerate(keys):
        if key not in index:
            index[key] = len(order)
            order.append(key)
        labels[i] = index[key]
    first = {k: keys.index(k) for k in order}
    unique = np.array([closures[first[k]] for k in order])
    return labels, order, unique


def align_h_distributions(mcmc_keys, mcmc_probability, reference_keys,
                          reference_probability) -> tuple[np.ndarray, np.ndarray, list]:
    """Put two `H` distributions on the union of their canonical keys."""
    union = list(mcmc_keys)
    for k in reference_keys:
        if k not in union:
            union.append(k)
    a = np.zeros(len(union))
    b = np.zeros(len(union))
    for k, p in zip(mcmc_keys, mcmc_probability):
        a[union.index(k)] = p
    for k, p in zip(reference_keys, reference_probability):
        b[union.index(k)] = p
    return a, b, union


def h_total_variation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())


def relation_marginal_from_u(u_draws, weights=None) -> np.ndarray:
    """`P(i > j)` from continuous draws, optionally importance-weighted."""
    closures = np.array([precedence_from_u(u) for u in np.asarray(u_draws, dtype=float)])
    flat = closures.reshape(closures.shape[0], -1).astype(float)
    if weights is None:
        return flat.mean(axis=0)
    w = np.asarray(weights, dtype=float)
    return (w[:, None] * flat).sum(axis=0) / w.sum()


def weighted_quantile(values, weights, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) / weights.sum()
    return float(np.interp(q, cdf, values[order]))


# ------------------------------------------------------------------------- structural
def structural_block(u_by_chain: np.ndarray, reference_keys=None,
                     reference_h_probability=None,
                     reference_relation_marginal=None) -> dict:
    """Induced-H distribution, TV against the reference, relation and incomparability."""
    u_by_chain = np.asarray(u_by_chain, dtype=float)
    n_chains, n_draws = u_by_chain.shape[0], u_by_chain.shape[1]
    m = u_by_chain.shape[2]
    pooled = u_by_chain.reshape(-1, m, u_by_chain.shape[3])

    labels, keys, closures = h_labels_from_u(pooled)
    probability = np.bincount(labels, minlength=len(keys)).astype(float)
    probability /= probability.sum()

    flat = closures.reshape(len(keys), -1).astype(float)
    relation_marginal = probability @ flat
    relation_counts = flat.sum(axis=1)
    counts_per_draw = relation_counts[labels].reshape(n_chains, n_draws)

    # incomparability: neither i>j nor j>i, off the diagonal
    closure_sq = closures.astype(bool)
    incomparable = np.zeros((len(keys), m, m), dtype=float)
    for k in range(len(keys)):
        c = closure_sq[k]
        inc = ~(c | c.T)
        np.fill_diagonal(inc, False)
        incomparable[k] = inc
    incomparability_marginal = np.tensordot(probability, incomparable, axes=(0, 0))

    max_pairs = m * (m - 1) // 2
    count_distribution = np.zeros(max_pairs + 1)
    np.add.at(count_distribution, relation_counts.astype(int), probability)

    out = {
        "n_h_states_visited": int(len(keys)),
        "h_keys_hex": [k.hex() for k in keys],
        "h_probability": probability.tolist(),
        "relation_marginal": relation_marginal.tolist(),
        "incomparability_marginal": incomparability_marginal.reshape(-1).tolist(),
        "relation_count_distribution": count_distribution.tolist(),
        "relation_count_convergence": convergence_block(counts_per_draw,
                                                        "relation count"),
        "per_chain_h_occupancy": [
            {int(l): int(c) for l, c in zip(*np.unique(
                labels.reshape(n_chains, n_draws)[i], return_counts=True))}
            for i in range(n_chains)],
    }

    if reference_keys is not None:
        a, b, _ = align_h_distributions(keys, probability, reference_keys,
                                        reference_h_probability)
        out["h_total_variation"] = h_total_variation(a, b)
        out["h_probability_aligned_mcmc"] = a.tolist()
        out["h_probability_aligned_reference"] = b.tolist()
    if reference_relation_marginal is not None:
        error = np.abs(relation_marginal - np.asarray(reference_relation_marginal))
        out["max_relation_marginal_error"] = float(error.max())
        out["mean_relation_marginal_error"] = float(error.mean())

    # per-relation R-hat over the uncertain relations only: a relation that never varies
    # has no R-hat, and averaging it in would flatter the report.
    indicators = np.array([[precedence_from_u(u).reshape(-1)
                            for u in chain] for chain in u_by_chain], dtype=float)
    per_relation = []
    for j in range(indicators.shape[2]):
        trace = indicators[:, :, j]
        if np.ptp(trace) == 0.0:
            continue
        block = rank_normalized_split_rhat(trace)
        per_relation.append({"relation": int(j), "rhat": block["rhat"],
                             "bulk_ess": float(bulk_ess(trace))})
    out["uncertain_relations"] = per_relation
    out["max_relation_rhat"] = (max(p["rhat"] for p in per_relation)
                                if per_relation else None)
    out["min_relation_bulk_ess"] = (min(p["bulk_ess"] for p in per_relation)
                                    if per_relation else None)
    out["n_uncertain_relations"] = len(per_relation)
    return out


# ----------------------------------------------------------------------------- scalars
def scalar_block(chains_by_name: dict, reference: dict, truth: dict,
                 acceptance_total: dict, acceptance_post: dict) -> dict:
    """Per-scalar convergence and agreement with the frozen reference marginal."""
    out = {}
    for name, chains in chains_by_name.items():
        chains = np.atleast_2d(np.asarray(chains, dtype=float))
        pooled = chains.ravel()
        entry = {"posterior_mean": float(pooled.mean()),
                 "posterior_sd": float(pooled.std(ddof=1)),
                 "median": float(np.median(pooled)),
                 "q025": float(np.quantile(pooled, 0.025)),
                 "q975": float(np.quantile(pooled, 0.975)),
                 "acceptance_total": acceptance_total.get(name),
                 "acceptance_post_burn_in": acceptance_post.get(name),
                 "n_draws": int(pooled.size)}
        entry.update(convergence_block(chains, name))
        ref = reference.get(name)
        if ref is not None:
            entry["reference_mean"] = float(ref["mean"])
            entry["reference_sd"] = float(ref["sd"])
            entry["mean_error"] = entry["posterior_mean"] - entry["reference_mean"]
            entry["mean_error_in_reference_sd"] = (
                entry["mean_error"] / entry["reference_sd"]
                if entry["reference_sd"] > 0 else float("nan"))
            if "grid" in ref and "cdf" in ref:
                entry["ks_distance_to_reference"] = ks_distance_to_grid(
                    pooled, ref["grid"], ref["cdf"])
        if name in truth and truth[name] is not None:
            t = float(truth[name])
            entry["true_value"] = t
            entry["truth_in_95_interval"] = bool(entry["q025"] <= t <= entry["q975"])
            entry["posterior_mean_error"] = entry["posterior_mean"] - t
            entry["error_in_posterior_sd"] = (
                entry["posterior_mean_error"] / entry["posterior_sd"]
                if entry["posterior_sd"] > 0 else float("nan"))
        else:
            entry["true_value"] = None
            entry["recovery"] = "NOT APPLICABLE — no generating value exists"
        out[name] = entry
    return out


def dependence_block(series: dict, pairs) -> dict:
    """Pairwise correlations, naming any constant coordinate instead of emitting NaN."""
    out = {}
    for a, b in pairs:
        x = np.asarray(series[a], dtype=float).ravel()
        y = np.asarray(series[b], dtype=float).ravel()
        constant = [n for n, v in ((a, x), (b, y)) if np.ptp(v) == 0.0]
        key = f"{a}_vs_{b}"
        if constant:
            out[key] = {"correlation": None, "undefined_because_constant": constant,
                        "note": "correlation is undefined, not zero: "
                                f"{' and '.join(constant)} has zero variance"}
        else:
            out[key] = {"correlation": float(np.corrcoef(x, y)[0, 1])}
    return out


# ----------------------------------------------------- mixed discrete/continuous gate
def mixed_envelope(mcmc_u, mcmc_scalars: dict, reference_u, reference_scalars: dict,
                   reference_weights=None, seed: int = 0, n_compare: int = 2000,
                   n_replicates: int = 40, quantile: float = 0.99) -> dict:
    """Energy distance on [closure indicators, standardised scalars, invariant U summaries].

    The reference cloud is importance-weighted, so it is resampled to an iid-equivalent
    sample first — comparing a weighted cloud against an unweighted one with an unweighted
    statistic would silently mis-state the null.
    """
    rng = np.random.default_rng(seed)
    reference_u = np.asarray(reference_u, dtype=float)
    if reference_weights is not None:
        w = np.asarray(reference_weights, dtype=float)
        w = w / w.sum()
        take = rng.choice(reference_u.shape[0], size=min(20000, reference_u.shape[0]),
                          replace=True, p=w)
    else:
        take = np.arange(reference_u.shape[0])
    reference_u = reference_u[take]
    reference_scalars = {k: np.asarray(v, dtype=float)[take]
                         for k, v in reference_scalars.items()}

    names = [n for n in ("rho", "beta", "omega", "lambda_rep", "lambda_back")
             if n in mcmc_scalars and n in reference_scalars]

    def featurise(u, scalars):
        u = np.asarray(u, dtype=float)
        closures = np.array([precedence_from_u(x).reshape(-1) for x in u], dtype=float)
        # permutation-invariant U summaries: row norms sorted, and the Gram trace
        gram = np.einsum("nij,nkj->nik", u, u)
        invariants = np.column_stack([
            np.sort(np.linalg.norm(u, axis=2), axis=1).reshape(u.shape[0], -1),
            np.trace(gram, axis1=1, axis2=2)[:, None]])
        return np.hstack([closures, invariants]
                         + [np.asarray(scalars[n]).reshape(-1, 1) for n in names])

    x_all = featurise(mcmc_u, mcmc_scalars)
    y_all = featurise(reference_u, reference_scalars)

    centre = y_all.mean(axis=0)
    scale = y_all.std(axis=0, ddof=1)
    keep = scale > 1e-12
    dropped = int((~keep).sum())
    if not keep.any():
        return {"observed": 0.0, "envelope": 0.0, "pass": True, "n_coordinates": 0,
                "dropped_constant_coordinates": dropped}

    ix = rng.choice(x_all.shape[0], size=min(n_compare, x_all.shape[0]), replace=False)
    iy = rng.choice(y_all.shape[0], size=min(n_compare, y_all.shape[0]), replace=False)
    x = standardise(x_all[ix][:, keep], centre[keep], scale[keep])
    y = standardise(y_all[iy][:, keep], centre[keep], scale[keep])

    observed = energy_distance(x, y)
    null = calibrate_energy_envelope(
        standardise(y_all[:, keep], centre[keep], scale[keep]),
        x.shape[0], y.shape[0], n_replicates, seed + 1, quantile)
    return {"statistic": "energy distance on [closure indicators, invariant U "
                         "summaries, standardised scalars]",
            "observed": float(observed), "envelope": float(null["envelope"]),
            "envelope_quantile": quantile, "null_mean": null["mean"],
            "null_sd": null["sd"], "n_replicates": n_replicates,
            "n_mcmc": int(x.shape[0]), "n_reference": int(y.shape[0]),
            "n_coordinates": int(keep.sum()),
            "dropped_constant_coordinates": dropped,
            "z_score": (float((observed - null["mean"]) / null["sd"])
                        if null["sd"] > 0 else float("nan")),
            "pass": bool(observed <= null["envelope"])}


# --------------------------------------------------------------- column-permutation
def column_permutation_audit(u_by_chain: np.ndarray) -> dict:
    """§13: is the target column-exchangeable, and do chains sit in different labellings?

    `h(U)` is the intersection of the column orderings and `Sigma_rho` is exchangeable in
    the columns, so the target is invariant under permuting them. Entrywise `U` traces are
    therefore not a convergence criterion; this reports the symmetry's footprint instead
    of pretending it is absent.
    """
    u = np.asarray(u_by_chain, dtype=float)
    n_chains = u.shape[0]
    # A column-order-sensitive statistic: mean of column 0 minus column 1.
    contrast = (u[..., 0] - u[..., 1]).mean(axis=(1, 2))
    # A column-invariant statistic: the same thing symmetrised.
    invariant = np.abs(u[..., 0] - u[..., 1]).mean(axis=(1, 2))
    return {
        "target_is_column_exchangeable": True,
        "per_chain_signed_column_contrast": contrast.tolist(),
        "per_chain_absolute_column_contrast": invariant.tolist(),
        "chains_in_opposite_labellings": bool(
            np.sign(contrast).min() != np.sign(contrast).max()),
        "signed_contrast_rhat": rank_normalized_split_rhat(
            (u[..., 0] - u[..., 1]).mean(axis=2))["rhat"] if n_chains > 1 else None,
        "note": "a large signed-contrast R-hat with a small invariant-summary R-hat is "
                "label switching, not non-convergence; H and relation probabilities are "
                "the reported quantities and are invariant",
    }
