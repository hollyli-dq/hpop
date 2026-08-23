"""Stage 6C — structural and scalar diagnostics for the latent-`U` chains.

Thin by design. Everything that already exists is reused rather than reimplemented:

* rank-normalised split R-hat, bulk/tail ESS and MCSE come from `stage6b_mcmc_diagnostics`;
* the energy-distance comparison and its reference-against-reference envelope come from
  `stage6b_joint_diagnostics`;
* total variation and the relation posterior come from `mcmc_original.diagnostics`.

What is genuinely new here is the *structural* side: the chain's state is a continuous
`U`, so every structural summary is a function of the induced order `h(U)` — a derived
label, never the state. Two conventions are held throughout:

* **closure vs reduction are reported separately.** `precedence_from_u` returns the
  transitive closure; the reduction is the cover relation. Recovery scored on the closure
  and on the reduction answer different questions and are not interchangeable.
* **degenerate coordinates are reported, not silently NaN-ed.** When the poset posterior
  is effectively a point mass the relation-count trace is constant, and R-hat/ESS are
  undefined rather than good. Those cases return `"degenerate": True` with the constant
  value attached, so a report cannot quietly present them as convergence.
"""

from __future__ import annotations

import math

import numpy as np

from hpop.mcmc_original.diagnostics import total_variation_distance
from hpop.mcmc_original.stage6b_joint_diagnostics import (
    calibrate_energy_envelope, energy_distance, standardise,
)
from hpop.mcmc_original.stage6b_mcmc_diagnostics import (
    bulk_ess, ks_distance_to_grid, mcse_mean, rank_normalized_split_rhat, tail_ess,
)

__all__ = [
    "poset_distribution", "full_u_total_variation", "relation_marginals",
    "structural_diagnostics", "recovery_metrics", "scalar_diagnostics",
    "mixed_comparison", "mode_summary", "convergence_block", "smoke_summary",
    "SMOKE_REQUIRED_KEYS",
]

# Every property a Stage 6C smoke run must demonstrate before formal chains are allowed
# to start (spec §13 for 6C1, §15 for 6C2). Kept here so the script that writes the
# summary and the test that checks it cannot drift apart.
SMOKE_REQUIRED_KEYS = (
    "legal_u_proposals", "accepted_u_proposals", "rejected_u_proposals",
    "rho_moved", "all_targets_finite", "q_zero_reset", "rejection_safe",
    "cache_parity", "deterministic_resume", "diagnostics_ran",
)


# ------------------------------------------------------------------- small utilities
def _is_degenerate(chains: np.ndarray) -> bool:
    values = np.asarray(chains, dtype=float)
    return bool(np.ptp(values) == 0.0)


def convergence_block(chains: np.ndarray, label: str = "") -> dict:
    """R-hat / ESS / MCSE, or an explicit degenerate marker when the trace is constant.

    A constant trace is not evidence of convergence; it is the absence of variation, and
    rank-normalisation of a constant vector is undefined. Saying so is the honest report.
    """
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    if _is_degenerate(chains):
        return {"degenerate": True, "constant_value": float(chains.ravel()[0]),
                "note": f"{label} is constant across all draws and chains; R-hat and ESS "
                        f"are undefined, not passing",
                "rhat": None, "bulk_ess": None, "tail_ess": None, "mcse": None}
    out = dict(rank_normalized_split_rhat(chains))
    out.update({"degenerate": False,
                "bulk_ess": float(bulk_ess(chains)),
                "tail_ess": float(tail_ess(chains)),
                "mcse": float(mcse_mean(chains)),
                "n_draws": int(chains.size), "n_chains": int(chains.shape[0])})
    return out


# ------------------------------------------------------------------------------ smoke
def smoke_summary(checks: dict, extra: dict | None = None) -> dict:
    """Assemble a smoke summary and say plainly whether every required property held.

    `checks` must cover `SMOKE_REQUIRED_KEYS` with booleans. A missing key is an error
    rather than a default, so a check that was never run cannot be mistaken for one that
    passed.
    """
    missing = [k for k in SMOKE_REQUIRED_KEYS if k not in checks]
    if missing:
        raise ValueError(f"smoke summary is missing required checks: {missing}")
    non_boolean = [k for k in SMOKE_REQUIRED_KEYS if not isinstance(checks[k], bool)]
    if non_boolean:
        raise ValueError(f"smoke checks must be booleans, got non-boolean: {non_boolean}")

    # Stage-specific checks beyond the required set (Stage 6C2 adds beta ones) are kept
    # and gated too — dropping them would let a stage-specific failure go unreported.
    extra_checks = {k: v for k, v in checks.items()
                    if k not in SMOKE_REQUIRED_KEYS and isinstance(v, bool)}
    all_checks = {**{k: bool(checks[k]) for k in SMOKE_REQUIRED_KEYS}, **extra_checks}
    failed = [k for k, v in all_checks.items() if not v]
    out = {"checks": all_checks,
           "failed_checks": failed,
           "all_passed": not failed}
    if extra:
        out.update(extra)
    return out


# ------------------------------------------------------------------------- structural
def poset_distribution(poset_ids, n_posets: int) -> np.ndarray:
    """Empirical distribution over catalogue entries. Unmatched ids (-1) are an error."""
    ids = np.asarray(poset_ids, dtype=int).ravel()
    if (ids < 0).any():
        raise ValueError(f"{int((ids < 0).sum())} draws did not match the catalogue; the "
                         f"catalogue is supposed to be exhaustive")
    counts = np.bincount(ids, minlength=n_posets).astype(float)
    return counts / counts.sum()


def full_u_total_variation(poset_ids, reference_probability, n_posets: int) -> dict:
    """Full-state TV between the MCMC poset distribution and the reference."""
    empirical = poset_distribution(poset_ids, n_posets)
    reference = np.asarray(reference_probability, dtype=float)
    reference = reference / reference.sum()
    tv = total_variation_distance(empirical, reference)
    worst = int(np.argmax(np.abs(empirical - reference)))
    return {"total_variation": float(tv),
            "worst_state": worst,
            "worst_state_mcmc": float(empirical[worst]),
            "worst_state_reference": float(reference[worst]),
            "n_states_visited": int((empirical > 0).sum()),
            "n_states_in_reference_support": int((reference > 1e-12).sum())}


def relation_marginals(poset_ids, catalogue) -> dict:
    """Closure and reduction marginals implied by the visited orders."""
    ids = np.asarray(poset_ids, dtype=int).ravel()
    closure = catalogue.closures[ids].reshape(ids.size, -1).astype(float)
    reduction = catalogue.reductions[ids].reshape(ids.size, -1).astype(float)
    return {"relation_marginal": closure.mean(axis=0),
            "reduction_marginal": reduction.mean(axis=0),
            "relation_counts": catalogue.relation_counts[ids]}


def structural_diagnostics(poset_ids_by_chain, catalogue,
                           reference_summary_arrays: dict) -> dict:
    """Every structural gate in one place: TV, marginal errors, count distribution.

    `poset_ids_by_chain` is `(n_chains, n_draws)` so per-chain agreement can be reported
    without pooling away a chain that disagrees.
    """
    ids = np.asarray(poset_ids_by_chain, dtype=int)
    if ids.ndim == 1:
        ids = ids[None, :]
    pooled = ids.ravel()

    tv = full_u_total_variation(pooled, reference_summary_arrays["poset_probability"],
                                catalogue.size)
    marginals = relation_marginals(pooled, catalogue)

    reference_relation = np.asarray(
        reference_summary_arrays["relation_marginal"], dtype=float)
    reference_reduction = np.asarray(
        reference_summary_arrays["reduction_marginal"], dtype=float)
    relation_error = np.abs(marginals["relation_marginal"] - reference_relation)
    reduction_error = np.abs(marginals["reduction_marginal"] - reference_reduction)

    # per-chain: each chain compared with the reference on its own
    per_chain = []
    for c in range(ids.shape[0]):
        chain_marginal = relation_marginals(ids[c], catalogue)["relation_marginal"]
        per_chain.append({
            "chain": c,
            "max_relation_marginal_error": float(
                np.abs(chain_marginal - reference_relation).max()),
            "total_variation": full_u_total_variation(
                ids[c], reference_summary_arrays["poset_probability"],
                catalogue.size)["total_variation"]})

    max_pairs = catalogue.m * (catalogue.m - 1) // 2
    count_distribution = np.bincount(marginals["relation_counts"],
                                     minlength=max_pairs + 1).astype(float)
    count_distribution /= count_distribution.sum()

    return {
        "full_u_total_variation": tv["total_variation"],
        "full_u_detail": tv,
        "max_relation_marginal_error": float(relation_error.max()),
        "mean_relation_marginal_error": float(relation_error.mean()),
        "max_reduction_marginal_error": float(reduction_error.max()),
        "relation_marginal": marginals["relation_marginal"].tolist(),
        "reduction_marginal": marginals["reduction_marginal"].tolist(),
        "relation_count_distribution": count_distribution.tolist(),
        "relation_count_convergence": convergence_block(
            np.asarray([catalogue.relation_counts[row] for row in ids], dtype=float),
            "relation count"),
        "per_chain": per_chain,
        "worst_chain_relation_error": float(
            max(p["max_relation_marginal_error"] for p in per_chain)),
    }


def recovery_metrics(poset_ids, catalogue, u_true) -> dict:
    """Structural recovery against the generating order. Closure and reduction, separately.

    This answers "did we recover `U_TRUE`", which is a different question from "is the
    sampler correct". A wrong sampler can recover the truth and a correct one can fail to.
    """
    from hpop.mcmc_original.latent_poset import precedence_from_u
    from hpop.mcmc_original.stage6c_exact_reference import transitive_reduction

    ids = np.asarray(poset_ids, dtype=int).ravel()
    true_closure = precedence_from_u(np.asarray(u_true, dtype=float))
    true_index = catalogue.index_of(true_closure)
    true_reduction = transitive_reduction(true_closure)

    empirical = poset_distribution(ids, catalogue.size)
    ranking = np.argsort(-empirical)
    rank_of_true = int(np.where(ranking == true_index)[0][0]) + 1

    marginals = relation_marginals(ids, catalogue)
    relation = marginals["relation_marginal"].reshape(catalogue.m, catalogue.m)
    reduction = marginals["reduction_marginal"].reshape(catalogue.m, catalogue.m)

    def prf(marginal, truth_matrix, threshold=0.5):
        predicted = marginal >= threshold
        truth_matrix = np.asarray(truth_matrix, dtype=bool)
        tp = int((predicted & truth_matrix).sum())
        fp = int((predicted & ~truth_matrix).sum())
        fn = int((~predicted & truth_matrix).sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if tp + fp and tp + fn and (precision + recall) > 0 else float("nan"))
        return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
                "precision": precision, "recall": recall, "f1": f1,
                "structural_hamming": fp + fn}

    true_mask = np.asarray(true_closure, dtype=bool)
    off_diagonal = ~np.eye(catalogue.m, dtype=bool)
    true_relation_probs = relation[true_mask]
    false_relation_probs = relation[(~true_mask) & off_diagonal]

    return {
        "true_poset_index": int(true_index),
        "map_poset_index": int(ranking[0]),
        "map_is_true": bool(int(ranking[0]) == int(true_index)),
        "posterior_probability_of_true": float(empirical[true_index]),
        "posterior_rank_of_true": rank_of_true,
        "map_probability": float(empirical[ranking[0]]),
        "n_unique_states_visited": int((empirical > 0).sum()),
        "closure": prf(relation, true_closure),
        "reduction": prf(reduction, true_reduction),
        "min_true_relation_probability": (float(true_relation_probs.min())
                                          if true_relation_probs.size else float("nan")),
        "max_false_relation_probability": (float(false_relation_probs.max())
                                           if false_relation_probs.size else float("nan")),
        "posterior_mean_relation_matrix": relation.tolist(),
        "true_relation_count": int(true_mask.sum()),
    }


def mode_summary(poset_ids_by_chain, catalogue, top: int = 5) -> dict:
    """Major modes, per-chain visits, and transitions between them."""
    ids = np.asarray(poset_ids_by_chain, dtype=int)
    if ids.ndim == 1:
        ids = ids[None, :]
    pooled = ids.ravel()
    empirical = poset_distribution(pooled, catalogue.size)
    modes = np.argsort(-empirical)[:top]
    modes = [int(m) for m in modes if empirical[m] > 0]

    visits = {str(m): [int((row == m).sum()) for row in ids] for m in modes}
    transitions = 0
    for row in ids:
        in_mode = np.isin(row, modes)
        labelled = np.where(in_mode, row, -1)
        changes = labelled[1:] != labelled[:-1]
        transitions += int((changes & (labelled[1:] != -1) & (labelled[:-1] != -1)).sum())
    return {"major_modes": modes,
            "major_mode_probability": [float(empirical[m]) for m in modes],
            "visits_by_chain": visits,
            "transitions_between_major_modes": transitions,
            "n_unique_states_visited": int((empirical > 0).sum())}


# ----------------------------------------------------------------------------- scalar
def scalar_diagnostics(chains_by_name: dict, reference: dict, truth: dict,
                       acceptance_total: dict, acceptance_post: dict) -> dict:
    """Per-scalar convergence and agreement with the frozen reference marginal.

    `reference[name]` must carry `grid`, `cdf` and the reference moments. `truth[name]`
    may be absent — that is the correct state for `rho`, which has no generating value.
    """
    out = {}
    for name, chains in chains_by_name.items():
        chains = np.atleast_2d(np.asarray(chains, dtype=float))
        pooled = chains.ravel()
        entry = {
            "posterior_mean": float(pooled.mean()),
            "posterior_sd": float(pooled.std(ddof=1)),
            "median": float(np.median(pooled)),
            "q025": float(np.quantile(pooled, 0.025)),
            "q975": float(np.quantile(pooled, 0.975)),
            "acceptance_total": acceptance_total.get(name),
            "acceptance_post_burn_in": acceptance_post.get(name),
            "n_draws": int(pooled.size),
        }
        entry.update(convergence_block(chains, name))
        if name in reference:
            grid = np.asarray(reference[name]["grid"], dtype=float)
            cdf = np.asarray(reference[name]["cdf"], dtype=float)
            entry["ks_distance_to_reference"] = ks_distance_to_grid(pooled, grid, cdf)
            entry["reference_mean"] = float(reference[name]["mean"])
            entry["reference_sd"] = float(reference[name]["sd"])
            entry["mean_error"] = entry["posterior_mean"] - entry["reference_mean"]
            entry["mean_error_in_reference_sd"] = (
                entry["mean_error"] / entry["reference_sd"]
                if entry["reference_sd"] > 0 else float("nan"))
        if name in truth and truth[name] is not None:
            true_value = float(truth[name])
            entry["true_value"] = true_value
            entry["truth_in_95_interval"] = bool(
                entry["q025"] <= true_value <= entry["q975"])
            entry["posterior_mean_error"] = entry["posterior_mean"] - true_value
            entry["error_in_posterior_sd"] = (
                entry["posterior_mean_error"] / entry["posterior_sd"]
                if entry["posterior_sd"] > 0 else float("nan"))
        else:
            entry["true_value"] = None
            entry["recovery"] = "NOT APPLICABLE — no generating value exists"
        out[name] = entry
    return out


# ------------------------------------------------------- mixed discrete + continuous
def mixed_comparison(poset_ids, scalars_by_name: dict, catalogue, reference_draws: dict,
                     seed: int, n_compare: int = 2000, n_replicates: int = 40,
                     quantile: float = 0.99) -> dict:
    """Energy distance on [closure indicators, standardised scalars], vs its own null.

    The discrete part enters as the canonical closure relation indicators, exactly as
    §12 requires, so the comparison is genuinely joint rather than a stack of marginals.
    Constant coordinates (a relation that never varies) are dropped before standardising
    — dividing by a zero scale would manufacture infinities out of perfect agreement.
    """
    ids = np.asarray(poset_ids, dtype=int).ravel()
    mcmc_relations = catalogue.closures[ids].reshape(ids.size, -1).astype(float)
    reference_relations = np.asarray(reference_draws["relations"], dtype=float)

    names = list(scalars_by_name)
    mcmc_scalars = np.column_stack(
        [np.asarray(scalars_by_name[n]).ravel() for n in names]) if names else None
    reference_scalars = np.column_stack(
        [np.asarray(reference_draws[n]).ravel() for n in names]) if names else None

    mcmc = (np.hstack([mcmc_relations, mcmc_scalars])
            if names else mcmc_relations)
    ref = (np.hstack([reference_relations, reference_scalars])
           if names else reference_relations)

    centre = ref.mean(axis=0)
    scale = ref.std(axis=0, ddof=1)
    keep = scale > 1e-12
    dropped = int((~keep).sum())
    if not keep.any():
        return {"statistic": "energy distance", "observed": 0.0, "envelope": 0.0,
                "pass": True, "n_coordinates": 0, "dropped_constant_coordinates": dropped,
                "note": "every coordinate is constant in the reference; the joint "
                        "comparison is vacuous and the marginal gates carry the result"}

    rng = np.random.default_rng(seed)
    take_mcmc = rng.choice(mcmc.shape[0], size=min(n_compare, mcmc.shape[0]),
                           replace=False)
    take_ref = rng.choice(ref.shape[0], size=min(n_compare, ref.shape[0]), replace=False)
    x = standardise(mcmc[take_mcmc][:, keep], centre[keep], scale[keep])
    y = standardise(ref[take_ref][:, keep], centre[keep], scale[keep])

    observed = energy_distance(x, y)
    null = calibrate_energy_envelope(
        standardise(ref[:, keep], centre[keep], scale[keep]),
        x.shape[0], y.shape[0], n_replicates, seed + 1, quantile)
    return {"statistic": "energy distance on reference-standardised "
                         "[closure indicators, scalars]",
            "observed": float(observed), "envelope": float(null["envelope"]),
            "envelope_quantile": quantile, "null_mean": null["mean"],
            "null_sd": null["sd"], "null_max": null["max"],
            "n_replicates": n_replicates, "n_mcmc": int(x.shape[0]),
            "n_reference": int(y.shape[0]),
            "n_coordinates": int(keep.sum()),
            "dropped_constant_coordinates": dropped,
            "z_score": (float((observed - null["mean"]) / null["sd"])
                        if null["sd"] > 0 else float("nan")),
            "pass": bool(observed <= null["envelope"])}
