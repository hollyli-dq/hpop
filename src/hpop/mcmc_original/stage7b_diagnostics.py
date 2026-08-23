"""Step 7B — movement diagnostics for the Local-vs-FFBS comparison.

Step 7B2 asks a question no gate in Stage 6E answers: *does the chain move through
structure at all*. Stage 6E2's diagnosis was that the induced orders `H = h(U)` are
effectively frozen within a chain — the segmentation is stuck given `U` and `U` is stuck
given the segmentation — so the statistics that matter are the ones that count movement
rather than the ones that summarise a posterior.

Everything here is therefore a **descriptive movement measure**, computed identically for
both samplers so the two can be put in one table. Nothing here is a convergence gate; the
convergence protocol is Stage 6E2's frozen permutation-invariant one and is applied to its
own summaries, which `invariant_summaries` assembles from the draws.

## Label exchangeability

`pi` and `P` are inferred on the Stage 6E2 corpus and the target has `3! = 6` equivalent
skill relabellings, so per-skill quantities indexed by an arbitrary label are not
comparable across chains. Every summary here is either invariant to relabelling by
construction (totals, sorted vectors, spectra, co-clustering) or is explicitly labelled
per-chain-only. `assert_no_truth_alignment` exists because the tempting fix — aligning
labels to the truth — would manufacture agreement between chains that the sampler never
achieved; alignment belongs to recovery, after convergence has been judged.
"""

from __future__ import annotations

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u

__all__ = [
    "SweepMovementTracker", "h_label", "h_label_series", "structural_movement", "segmentation_movement",
    "invariant_summaries", "mode_occupancy", "compare_equal_sweeps", "compare_equal_time",
    "compare_heldout", "co_clustering_series", "assert_no_truth_alignment",
    "INVARIANT_SUMMARY_NAMES", "CO_CLUSTERING_PAIRS",
]

# The Stage 6E2 permutation-invariant summary set. Each is unchanged by relabelling the
# skills, which is what makes cross-chain R-hat meaningful under 3! = 6 exchangeability.
INVARIANT_SUMMARY_NAMES = (
    "log_posterior", "total_relation_count", "sorted_relation_counts",
    "total_segments", "sorted_pi", "transition_eigenvalue_moduli",
    "transition_singular_values", "sorted_row_entropies", "co_clustering_mean",
    "rho", "beta", "omega", "lambda_rep", "lambda_back",
)

# Co-clustering asks whether two occurrences carry the *same* skill, never which skill, so
# it is invariant under relabelling. It needs the occurrence-label draws, which the small
# Stage 6E1B model does not store; `invariant_summaries` therefore emits it only when they
# are supplied, and every consumer treats the summary set as a subset of the names above.
CO_CLUSTERING_PAIRS = 64


class SweepMovementTracker:
    """Per-sweep structural and segmentation movement, recorded while a chain runs.

    Thinned draws cannot answer "how many sweeps until the first `H` change" or "when did
    the chain leave its initial structural basin" — by construction those questions live at
    sweep resolution, and a thin of 5 hides four sweeps in five. So the chain records the
    raw series here and every summary is computed afterwards from it. Nothing in this class
    is read by the sampler: it observes, and a registered run must not be steered by what
    it sees.

    The per-sweep cost is a precedence closure per skill plus a few hundred label lookups,
    which is a fraction of a percent of a Stage 6E2 sweep.
    """

    def __init__(self, n_skills: int, traces, n_pairs: int = 256, seed: int = 7_063_900):
        self.n_skills = int(n_skills)
        rng = np.random.default_rng(seed)
        lengths = [len(t) for t in traces]
        usable = [n for n, length in enumerate(lengths) if length >= 2]
        self.pairs = []
        for _ in range(int(n_pairs)):
            trace = int(rng.choice(usable))
            i, j = rng.choice(lengths[trace], size=2, replace=False)
            self.pairs.append((trace, int(i), int(j)))
        self.traces_used = sorted({p[0] for p in self.pairs})
        self.h_hash: list = []
        self.relation_total: list = []
        self.boundary_hamming: list = []
        self.label_changes: list = []
        self.states_changed: list = []
        self.co_clustering: list = []
        self.log_target: list = []

    @staticmethod
    def _label_at(key, position: int) -> int:
        start = 0
        for end, skill in key:
            if start <= position < end:
                return int(skill)
            start = end
        return -1

    def record(self, state, keys) -> None:
        u = np.asarray(state.u_by_skill, dtype=float)
        closures = [precedence_from_u(u[k]) for k in range(self.n_skills)]
        self.h_hash.append(hash(b"|".join(c.tobytes() for c in closures)))
        self.relation_total.append(int(sum(int(c.sum()) for c in closures)))
        self.boundary_hamming.append(
            int(state.components.get("boundary_hamming_moved", 0)))
        self.label_changes.append(int(state.components.get("label_changes_moved", 0)))
        self.states_changed.append(int(state.components.get("ffbs_states_changed", 0)))
        self.log_target.append(float(state.components.get("log_target", float("nan"))))
        same = 0
        for trace, i, j in self.pairs:
            key = keys[trace]
            same += int(self._label_at(key, i) == self._label_at(key, j))
        self.co_clustering.append(same / max(1, len(self.pairs)))

    def series(self) -> dict:
        return {"h_hash": np.array(self.h_hash, dtype=np.int64),
                "relation_total": np.array(self.relation_total, dtype=np.int16),
                "boundary_hamming": np.array(self.boundary_hamming, dtype=np.int32),
                "label_changes": np.array(self.label_changes, dtype=np.int32),
                "states_changed": np.array(self.states_changed, dtype=np.int16),
                "co_clustering": np.array(self.co_clustering, dtype=np.float32),
                "log_target": np.array(self.log_target, dtype=float)}

    def summary(self) -> dict:
        """Every §19 movement measure, computed from the recorded series."""
        series = self.series()
        labels = [bytes(str(value), "ascii") for value in series["h_hash"].tolist()]
        occupancy = mode_occupancy(labels)
        relation = series["relation_total"].astype(float)
        co = series["co_clustering"].astype(float)
        major = [state for state, count in
                 _counts(labels).items() if count >= 0.01 * max(1, len(labels))]
        major_set = set(major)
        major_transitions = sum(
            1 for a, b in zip(labels[:-1], labels[1:])
            if a != b and a in major_set and b in major_set)
        return {
            "sweeps": len(labels),
            "h_changes": occupancy["changes"],
            "distinct_h_states": occupancy["distinct_states"],
            "sweeps_to_first_h_change": occupancy["draws_to_first_change"],
            "sweeps_to_leave_the_initial_h_basin":
                occupancy["draws_to_leave_the_initial_state"],
            "modal_h_occupancy": occupancy["modal_occupancy"],
            "h_occupancy_top20": occupancy["occupancy_top20"],
            "major_h_states": len(major),
            "major_h_mode_transitions": int(major_transitions),
            "relation_count_within_chain_sd": float(relation.std(ddof=1))
            if relation.size > 1 else 0.0,
            "relation_count_mean": float(relation.mean()) if relation.size else float("nan"),
            "distinct_relation_totals": int(len(np.unique(relation))),
            "boundary_hamming_per_sweep": float(series["boundary_hamming"].mean()),
            "label_changes_per_sweep": float(series["label_changes"].mean()),
            "trace_draws_changed_per_sweep": float(series["states_changed"].mean()),
            "co_clustering_mean": float(co.mean()) if co.size else float("nan"),
            "co_clustering_movement_per_sweep": float(np.abs(np.diff(co)).mean())
            if co.size > 1 else 0.0,
        }


def _counts(labels) -> dict:
    out: dict = {}
    for label in labels:
        out[label] = out.get(label, 0) + 1
    return out


def h_label(u_k) -> bytes:
    """The induced order of one skill, as a hashable label."""
    return precedence_from_u(np.asarray(u_k, dtype=float)).tobytes()


def h_label_series(u_draws) -> list:
    """`(n_draws, K, m, d)` -> one label per draw, joining every skill's induced order.

    The joint label is the object Stage 6E2 found frozen: a chain can shuffle which skill
    holds which order without the *set* of orders moving, and the per-skill labels alone
    would not show that.
    """
    u_draws = np.asarray(u_draws, dtype=float)
    return [b"|".join(h_label(u_draws[i, k]) for k in range(u_draws.shape[1]))
            for i in range(u_draws.shape[0])]


def mode_occupancy(labels) -> dict:
    """How the draws distribute over distinct structural states, and how often they move."""
    labels = list(labels)
    counts: dict = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    changes = sum(1 for a, b in zip(labels[:-1], labels[1:]) if a != b)
    first_change = next((i + 1 for i, (a, b) in enumerate(zip(labels[:-1], labels[1:]))
                         if a != b), None)
    left_initial = next((i + 1 for i, label in enumerate(labels[1:])
                         if label != labels[0]), None) if labels else None
    ordered = sorted(counts.values(), reverse=True)
    total = max(1, len(labels))
    # only the head of the occupancy vector is reported: with tens of thousands of draws
    # the tail is one draw per cell and carries no information the count does not.
    head = [c / total for c in ordered[:20]]
    return {
        "n_draws": len(labels), "distinct_states": len(counts), "changes": int(changes),
        "changes_per_1000_draws": 1000.0 * changes / total,
        "draws_to_first_change": first_change,
        "draws_to_leave_the_initial_state": left_initial,
        "occupancy_top20": head,
        "occupancy_tail_mass": float(1.0 - sum(head)),
        "modal_occupancy": ordered[0] / total if ordered else float("nan"),
    }


def structural_movement(u_draws_by_chain, relation_counts_by_chain) -> dict:
    """§21's structural movement measures, per chain and pooled.

    `u_draws_by_chain` is `(C, n_draws, K, m, d)`; `relation_counts_by_chain` is
    `(C, n_draws, K)`. Both are read as retained draws in chain order.
    """
    per_chain = []
    all_labels = set()
    for chain, u_draws in enumerate(u_draws_by_chain):
        labels = h_label_series(u_draws)
        all_labels.update(labels)
        counts = np.asarray(relation_counts_by_chain[chain], dtype=float)
        totals = counts.sum(axis=1)
        occupancy = mode_occupancy(labels)
        per_chain.append({
            "chain": int(chain),
            **occupancy,
            "total_relation_count_mean": float(totals.mean()),
            "total_relation_count_within_chain_sd": float(totals.std(ddof=1))
            if len(totals) > 1 else 0.0,
            "per_skill_relation_within_chain_sd": [
                float(counts[:, k].std(ddof=1)) if len(counts) > 1 else 0.0
                for k in range(counts.shape[1])],
            "distinct_total_relation_counts": int(len(np.unique(totals))),
        })
    frozen_chains = sum(1 for row in per_chain
                        if row["total_relation_count_within_chain_sd"] < 0.01)
    means = [row["total_relation_count_mean"] for row in per_chain]
    return {
        "per_chain": per_chain,
        "distinct_structural_states_pooled": len(all_labels),
        "total_structural_changes": int(sum(r["changes"] for r in per_chain)),
        "chains_with_frozen_structure": int(frozen_chains),
        "max_between_chain_relation_count_gap": float(max(means) - min(means))
        if means else float("nan"),
        "criterion_A_frozen_structure": bool(frozen_chains >= 2),
        "criterion_B_disagreeing_structure": bool(
            means and (max(means) - min(means)) > 1.0),
        "criteria_note": "A and B are Stage 6E2's registered structural-locking criteria, "
                         "reproduced here so the two samplers are judged by the same rule; "
                         "criterion C is the invariant log-posterior spread and is "
                         "computed by `invariant_summaries`",
    }


def segmentation_movement(label_draws_by_chain=None, boundary_keys_by_chain=None,
                          per_sweep_movement=None) -> dict:
    """Movement of `(S, z)` itself: boundary Hamming, label changes, segment counts.

    `per_sweep_movement` is the sampler's own running total (it sees every sweep, not only
    the retained ones) and is reported alongside the retained-draw measures, because a
    thinned chain understates movement by construction.
    """
    out: dict = {}
    if per_sweep_movement is not None:
        out["per_sweep_totals"] = [dict(m) for m in per_sweep_movement]
    if boundary_keys_by_chain is not None:
        rows = []
        for chain, draws in enumerate(boundary_keys_by_chain):
            keys = [tuple(tuple(k) for k in draw) for draw in draws]
            changes = sum(1 for a, b in zip(keys[:-1], keys[1:]) if a != b)
            distinct = len({k for draw in keys for k in draw})
            counts = np.array([[len(k) for k in draw] for draw in keys], dtype=float)
            rows.append({
                "chain": int(chain), "retained_draws": len(keys),
                "retained_draws_with_a_change": int(changes),
                "fraction_of_retained_draws_that_moved": changes / max(1, len(keys) - 1),
                "distinct_per_trace_segmentations_visited": int(distinct),
                "segment_count_mean": float(counts.mean()),
                "segment_count_within_chain_sd": float(counts.std(ddof=1)),
            })
        out["per_chain"] = rows
    if label_draws_by_chain is not None:
        hamming = []
        for draws in label_draws_by_chain:
            draws = np.asarray(draws)
            if draws.ndim < 2 or len(draws) < 2:
                continue
            differences = (draws[1:] != draws[:-1]) & (draws[1:] >= 0)
            hamming.append(float(differences.sum(axis=tuple(range(1, draws.ndim))).mean()))
        out["mean_occurrence_label_hamming_between_retained_draws"] = hamming
    return out


def _row_entropy(row) -> float:
    row = np.asarray(row, dtype=float)
    positive = row[row > 0]
    return float(-(positive * np.log(positive)).sum())


def invariant_summaries(chain: dict) -> dict:
    """Permutation-invariant series for one chain, keyed by `INVARIANT_SUMMARY_NAMES`.

    `chain` carries `log_target`, `relation_counts` `(n, K)`, `segment_counts` `(n, N)`,
    `pi_draws` `(n, K)`, `transition_draws` `(n, K, K)` and the scalar series. Every output
    is invariant under relabelling the skills: totals and sorted vectors by construction,
    and the spectra because a relabelling is a simultaneous row/column permutation, i.e. a
    similarity transform by a permutation matrix.
    """
    out: dict = {}
    log_target = np.asarray(chain["log_target"], dtype=float)
    relation = np.asarray(chain["relation_counts"], dtype=float)
    out["log_posterior"] = log_target
    out["total_relation_count"] = relation.sum(axis=1)
    out["sorted_relation_counts"] = np.sort(relation, axis=1)
    if chain.get("segment_counts") is not None:
        out["total_segments"] = np.asarray(chain["segment_counts"], dtype=float).sum(axis=1)
    if chain.get("pi_draws") is not None:
        pi = np.asarray(chain["pi_draws"], dtype=float)
        out["sorted_pi"] = np.sort(pi, axis=1)
    if chain.get("transition_draws") is not None:
        transition = np.asarray(chain["transition_draws"], dtype=float)
        moduli, singular, entropies = [], [], []
        for matrix in transition:
            moduli.append(np.sort(np.abs(np.linalg.eigvals(matrix)))[::-1])
            singular.append(np.linalg.svd(matrix, compute_uv=False))
            entropies.append(np.sort([_row_entropy(row) for row in matrix]))
        out["transition_eigenvalue_moduli"] = np.array(moduli)
        out["transition_singular_values"] = np.array(singular)
        out["sorted_row_entropies"] = np.array(entropies)
    if chain.get("label_draws") is not None:
        out["co_clustering_mean"] = co_clustering_series(chain["label_draws"])
    for name in ("rho", "beta", "omega", "lambda_rep", "lambda_back"):
        if chain.get("scalars") is not None and name in chain["scalars"]:
            out[name] = np.asarray(chain["scalars"][name], dtype=float)
    return out


def co_clustering_series(label_draws, n_pairs: int = CO_CLUSTERING_PAIRS,
                         seed: int = 6_053_900) -> np.ndarray:
    """Per draw, the fraction of a fixed occurrence-pair sample that shares a skill.

    The pair sample is drawn once from a fixed seed and reused for every draw and every
    chain, so the series are comparable; the statistic itself never names a skill, which is
    what makes it invariant under the `K!` relabellings.
    """
    draws = np.asarray(label_draws)
    n_draws = draws.shape[0]
    flat = draws.reshape(n_draws, -1)
    valid = np.flatnonzero(flat[0] >= 0)
    rng = np.random.default_rng(seed)
    if valid.size < 2:
        return np.zeros(n_draws)
    left = rng.choice(valid, size=n_pairs)
    right = rng.choice(valid, size=n_pairs)
    keep = left != right
    left, right = left[keep], right[keep]
    if left.size == 0:
        return np.zeros(n_draws)
    return (flat[:, left] == flat[:, right]).mean(axis=1).astype(float)


def compare_heldout(local: dict, ffbs: dict, oracle: dict | None = None,
                    true_parameters: dict | None = None) -> dict:
    """The §26 held-out table: four predictive rows, reported only when interpretable.

    Local and FFBS target the same posterior, so once both have converged their predictive
    distributions should agree; a large gap after both are called converged is a diagnostic
    problem, not a result, and `flag` says so rather than ranking them.
    """
    rows = {"local_move_kernel": local, "ffbs": ffbs}
    if oracle is not None:
        rows["oracle_boundary_control"] = oracle
    if true_parameters is not None:
        rows["true_parameter_oracle"] = true_parameters
    interpretable = bool(local.get("converged") and ffbs.get("converged"))
    values = {name: block.get("heldout_nll_per_step") for name, block in rows.items()}
    gap = None
    if values.get("local_move_kernel") is not None and values.get("ffbs") is not None:
        gap = abs(values["local_move_kernel"] - values["ffbs"])
    return {
        "rows": rows, "heldout_nll_per_step": values,
        "local_minus_ffbs_absolute_gap": gap,
        "interpretable": interpretable,
        "status": ("interpreted" if interpretable else
                   "NOT INTERPRETED — FORMAL CHAINS DID NOT CONVERGE"),
        "flag": ("a large predictive gap between two samplers of the same posterior "
                 "indicates a diagnostic problem, not a better sampler"
                 if gap is not None and interpretable and gap > 0.05 else None),
        "never_tuned_on": "held-out NLL is reported, never used to choose a scale, a "
                          "prior, a run length or a kernel",
    }


def assert_no_truth_alignment(module_source: str) -> dict:
    """Convergence code must not touch the hidden truth.

    Aligning skill labels to the truth before measuring cross-chain agreement would turn a
    label-exchangeability artefact into apparent convergence. The check is a source scan
    rather than a promise.
    """
    forbidden = ("skill_alignment", "hidden_true_labels", "hidden_true_boundaries",
                 "u_true", "true_keys", "align_to_truth")
    found = [name for name in forbidden if name in module_source]
    return {"forbidden_symbols": list(forbidden), "found": found,
            "pass": bool(not found)}


# ------------------------------------------------------------------ comparison schemas
def compare_equal_sweeps(local: dict, ffbs: dict) -> dict:
    """The §23 table at equal sweep counts. Values are passed through, never recomputed.

    Both inputs carry the same keys; anything missing is reported as `None` rather than
    filled in, because a blank in this table is information about what was not measured.
    """
    rows = ("wall_seconds", "sweeps", "log_posterior_rhat", "total_relation_count_rhat",
            "max_invariant_rhat", "min_invariant_bulk_ess", "min_invariant_ess_per_second",
            "h_changes", "distinct_h_states", "structural_mode_transitions",
            "segment_count_ess", "co_clustering_ess", "beta_ess", "omega_ess",
            "lambda_rep_ess", "lambda_back_ess")
    return {
        "basis": "equal sweeps",
        "rows": {name: {"local_move_kernel": local.get(name), "ffbs": ffbs.get(name)}
                 for name in rows},
        "missing": [name for name in rows
                    if local.get(name) is None or ffbs.get(name) is None],
    }


def compare_equal_time(local: dict, ffbs: dict) -> dict:
    """The same table on a wall-clock basis, with the ESS rescaled to a common budget.

    Rescaling assumes ESS grows linearly in draws, which is right for a chain past burn-in
    and wrong near it; the assumption is recorded in the output rather than buried.
    """
    budget = min(float(local.get("wall_seconds", float("inf"))),
                 float(ffbs.get("wall_seconds", float("inf"))))
    out = {"basis": "equal wall clock", "budget_seconds": budget,
           "assumption": "ESS scales linearly with wall time within a run; valid past "
                         "burn-in, optimistic near it",
           "rows": {}}
    for name in ("min_invariant_bulk_ess", "segment_count_ess", "beta_ess", "omega_ess",
                 "lambda_rep_ess", "lambda_back_ess", "co_clustering_ess"):
        row = {}
        for label, block in (("local_move_kernel", local), ("ffbs", ffbs)):
            value, seconds = block.get(name), block.get("wall_seconds")
            row[label] = (None if value is None or not seconds
                          else float(value) * budget / float(seconds))
        out["rows"][name] = row
    return out
