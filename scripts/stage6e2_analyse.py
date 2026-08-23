"""Stage 6E2 — convergence (§15), recovery (§16) and held-out prediction (§17).

    PYTHONPATH=src python scripts/stage6e2_analyse.py

Correctness, convergence and recovery are three separate verdicts and are reported as
three separate verdicts. Nothing here can change the sampler, the target or the proposal
scales: by the time this script runs, the chains are finished and the scales are frozen in
`pilot_results.json`.

The generating truth enters only from this point onward, and only for recovery reporting.
The Hungarian alignment of §10 is applied per retained draw, on a frozen cost, and its
result is never fed back into a convergence statistic — those use permutation-invariant
summaries (co-clustering on a fixed pair sample, segment counts, relation counts, the
transition-count spectrum) precisely because a raw per-skill trace would be meaningless if
the target were label-exchangeable and misleading if it were only nearly so.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.latent_poset import precedence_from_u                  # noqa: E402
from hpop.mcmc_original.recurrent_scalar_posterior import TRUE_VALUES          # noqa: E402
from hpop.mcmc_original.stage6c_diagnostics import convergence_block           # noqa: E402
from hpop.mcmc_original.stage6d_frozen import SCALAR_ORDER                     # noqa: E402
from hpop.mcmc_original.stage6e_corpus import generate_corpus                  # noqa: E402
from hpop.mcmc_original.stage6e_diagnostics import (                           # noqa: E402
    boundary_indicators, boundary_recovery, co_clustering_sample, heldout_predictive,
    labels_to_key, partial_order_recovery, skill_alignment, skill_recovery,
    transitive_reduction,
)
from hpop.mcmc_original.stage6e_frozen import (                                # noqa: E402
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH, N_ROLES, N_SKILLS,
)
from hpop.mcmc_original.transitions import transition_counts                   # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "stage6e2_unknown_boundary_full_seed0"

RHAT_GATE = 1.01
MIN_BULK_ESS = 400
MAX_MCSE_OVER_SD = 0.05
ACCEPTANCE_BAND = (0.10, 0.70)
N_PREDICTIVE_DRAWS = 200
N_COCLUSTER_PAIRS = 300
PAIR_SEED = 6_053_401


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:                                                # pragma: no cover
        return "unknown"


def jsonable(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def rhat_of(block):
    """The block's R-hat, or `None` when it is genuinely undefined.

    A *constant* trace is degenerate: rank-normalisation of a constant vector has no
    value, so `None` is the honest answer and the coordinate is recorded as degenerate
    rather than as an R-hat of 1.0. A **NaN** is different — it means the statistic was
    attempted and did not produce a number — and is returned as NaN so the gate below can
    fail it rather than quietly treat it as absent.
    """
    if block is None or block.get("degenerate"):
        return None
    return block.get("rhat")


# --------------------------------------------------------------------------- loading
def load(tag: str) -> dict:
    path = OUT / ("chains.npz" if tag == "unknown" else "oracle_control_chains.npz")
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ------------------------------------------------------------------ scalar diagnostics
def scalar_diagnostics(data: dict, summary: dict) -> dict:
    out = {}
    for name in (*SCALAR_ORDER, "rho"):
        chains = data[f"scalar_{name}"]
        block = convergence_block(chains, name)
        flat = chains.ravel()
        truth = TRUE_VALUES.get(name)
        q025, q975 = float(np.quantile(flat, 0.025)), float(np.quantile(flat, 0.975))
        out[name] = {
            "posterior_mean": float(flat.mean()), "posterior_sd": float(flat.std(ddof=1)),
            "median": float(np.median(flat)), "q025": q025, "q975": q975,
            "true_value": truth,
            "truth_in_95_credible_interval": (None if truth is None
                                              else bool(q025 <= truth <= q975)),
            **block,
            "mcse_over_sd": (block["mcse"] / max(1e-12, float(flat.std(ddof=1)))
                             if not block.get("degenerate") else None),
        }
    out["rho"]["true_value"] = None
    out["rho"]["truth_in_95_credible_interval"] = None
    out["rho"]["rho_true_status"] = (
        "NOT APPLICABLE — U_TRUE_BY_SKILL is hand-specified in stage6e_corpus.py, not "
        "drawn from p(U | rho), so no rho_true exists to recover. Inherited from the "
        "Stage 6C freeze unchanged.")
    out["log_target"] = convergence_block(data["log_target"], "log_target")
    out["acceptance"] = summary.get("acceptance_post_burn_in")
    return out


# ------------------------------------------------------- segmentation diagnostics
def segmentation_diagnostics(data: dict, corpus, summary: dict) -> dict:
    labels = data["occurrence_labels"]                      # (chains, draws, N, Jmax)
    n_chains, n_draws, n_traces, _ = labels.shape
    lengths = [t.length for t in corpus.train]

    segment_counts = data["segment_counts"].astype(float)   # (chains, draws, N)
    total_segments = segment_counts.sum(axis=2)
    count_block = convergence_block(total_segments, "total segment count")

    # a selected boundary indicator per trace: the internal position whose posterior
    # probability is closest to 0.5, i.e. the least determined and hardest to mix
    pooled = labels.reshape(n_chains * n_draws, n_traces, -1)
    indicator_blocks = []
    for n in range(min(n_traces, 20)):
        probability = np.array([boundary_indicators(pooled[d, n])
                                for d in range(pooled.shape[0])]).mean(axis=0)
        if probability.size == 0:
            continue
        position = int(np.argmin(np.abs(probability - 0.5)))
        series = np.array([[boundary_indicators(labels[c, d, n])[position]
                            for d in range(n_draws)] for c in range(n_chains)],
                          dtype=float)
        indicator_blocks.append({
            "trace": n, "position": position + 1,
            "posterior_probability": float(probability[position]),
            **convergence_block(series, f"boundary trace {n} position {position + 1}")})
    varying = [b for b in indicator_blocks if not b.get("degenerate")]

    # per-chain occupancy of each trace's segmentation, and mode transitions
    occupancy, mode_transitions, unique_states = [], [], []
    for n in range(min(n_traces, 20)):
        per_chain = []
        for c in range(n_chains):
            keys = [labels_to_key(labels[c, d, n]) for d in range(n_draws)]
            counts: dict = {}
            for k in keys:
                counts[k] = counts.get(k, 0) + 1
            per_chain.append(counts)
            if c == 0:
                mode_transitions.append(
                    sum(1 for a, b in zip(keys[:-1], keys[1:]) if a != b))
        union = sorted({k for counts in per_chain for k in counts},
                       key=lambda k: -sum(c.get(k, 0) for c in per_chain))
        unique_states.append(len(union))
        occupancy.append({
            "trace": n, "n_distinct_segmentations": len(union),
            "top_states": [{"key": [list(p) for p in k],
                            "per_chain_probability": [c.get(k, 0) / n_draws
                                                      for c in per_chain]}
                           for k in union[:3]],
            "max_per_chain_probability_gap": float(max(
                max(c.get(k, 0) / n_draws for c in per_chain)
                - min(c.get(k, 0) / n_draws for c in per_chain)
                for k in union[:3])) if union else 0.0,
        })

    return {
        "n_chains": n_chains, "n_draws_per_chain": n_draws, "n_traces": n_traces,
        "total_segment_count": {
            "posterior_mean": float(total_segments.mean()),
            "posterior_sd": float(total_segments.std(ddof=1)),
            "true_total": int(sum(t.n_blocks for t in corpus.train)),
            **count_block},
        "per_trace_segment_count_mean": segment_counts.mean(axis=(0, 1)).tolist(),
        "boundary_count_convergence": count_block,
        "selected_boundary_indicators": indicator_blocks,
        "max_boundary_indicator_rhat": (
            float(max(b["rhat"] for b in varying)) if varying else None),
        "min_boundary_indicator_bulk_ess": (
            float(min(b["bulk_ess"] for b in varying)) if varying else None),
        "n_degenerate_boundary_indicators": len(indicator_blocks) - len(varying),
        "unique_segmentations_per_trace": {
            "mean": float(np.mean(unique_states)), "min": int(min(unique_states)),
            "max": int(max(unique_states))},
        "segmentation_mode_transitions_chain0": mode_transitions,
        "per_chain_occupancy": occupancy[:5],
        "move_acceptance_by_type": summary.get("acceptance_post_burn_in"),
        "boundary_hamming_movement": [p["movement"] for p in summary["per_chain"]],
        "trace_lengths": lengths,
    }


# ------------------------------------------------------------- skill diagnostics
def skill_diagnostics(data: dict, corpus, pair_sample) -> dict:
    labels = data["occurrence_labels"]
    n_chains, n_draws, n_traces, _ = labels.shape
    pooled = labels.reshape(n_chains * n_draws, n_traces, -1)
    chain_sizes = [n_draws] * n_chains

    co_cluster = co_clustering_sample(pooled, pair_sample, chain_sizes)

    # transition-count spectrum: permutation invariant (sorted eigenvalue magnitudes)
    spectra = np.empty((n_chains, n_draws))
    counts_mean = np.zeros((N_SKILLS, N_SKILLS))
    for c in range(n_chains):
        for d in range(n_draws):
            paths = [[k for _, k in labels_to_key(labels[c, d, n])]
                     for n in range(n_traces)]
            counts = transition_counts(paths, N_SKILLS)
            counts_mean += counts
            spectra[c, d] = float(np.abs(np.linalg.eigvals(counts)).sum())
    counts_mean /= (n_chains * n_draws)

    pi_blocks = {f"pi[{k}]": convergence_block(data["pi_draws"][:, :, k], f"pi[{k}]")
                 for k in range(N_SKILLS)}
    p_blocks = {}
    for h in range(N_SKILLS):
        for k in range(N_SKILLS):
            if h == k:
                continue
            p_blocks[f"P[{h},{k}]"] = convergence_block(
                data["transition_draws"][:, :, h, k], f"P[{h},{k}]")

    return {
        "co_clustering": co_cluster,
        "co_clustering_note": "permutation invariant, so it is a legitimate convergence "
                              "summary whether or not the labels are exchangeable",
        "transition_count_spectrum": convergence_block(spectra,
                                                       "transition-count spectrum"),
        "mean_transition_counts": counts_mean.tolist(),
        "true_transition_counts": transition_counts(
            [list(t.true_labels) for t in corpus.train], N_SKILLS).tolist(),
        "pi_convergence": pi_blocks,
        "P_convergence": p_blocks,
        "pi_posterior_mean": data["pi_draws"].reshape(-1, N_SKILLS).mean(axis=0).tolist(),
        "pi_true": corpus.pi_true.tolist(),
        "P_posterior_mean": data["transition_draws"].reshape(
            -1, N_SKILLS, N_SKILLS).mean(axis=0).tolist(),
        "P_true": corpus.p_true.tolist(),
        "unaligned_label_frequency": [
            float((pooled == k).sum() / (pooled >= 0).sum()) for k in range(N_SKILLS)],
        "unaligned_note": "raw per-skill label frequencies, reported WITHOUT alignment so "
                          "the unaligned posterior behaviour is visible; the aligned "
                          "summaries live in recovery_results.json",
    }


# ------------------------------------------------------- structural diagnostics
def structural_diagnostics(data: dict, corpus) -> dict:
    u = data["u_draws"]                                  # (chains, draws, K, m, d)
    n_chains, n_draws = u.shape[0], u.shape[1]
    relation_counts = data["relation_counts"].astype(float)

    per_skill = []
    for k in range(N_SKILLS):
        keys: dict = {}
        per_chain = []
        for c in range(n_chains):
            chain_keys: dict = {}
            for d in range(n_draws):
                key = precedence_from_u(u[c, d, k]).tobytes()
                keys[key] = keys.get(key, 0) + 1
                chain_keys[key] = chain_keys.get(key, 0) + 1
            per_chain.append(chain_keys)
        total = n_chains * n_draws
        top = sorted(keys, key=keys.get, reverse=True)[:3]
        per_skill.append({
            "skill_index_unaligned": k,
            "n_distinct_induced_orders": len(keys),
            "top_orders": [{"probability": keys[t] / total,
                            "per_chain_probability": [c.get(t, 0) / n_draws
                                                      for c in per_chain],
                            "closure": np.frombuffer(t, dtype=bool).reshape(
                                N_ROLES, N_ROLES).tolist()} for t in top],
            "max_per_chain_occupancy_gap": float(max(
                max(c.get(t, 0) / n_draws for c in per_chain)
                - min(c.get(t, 0) / n_draws for c in per_chain) for t in top)),
            "relation_count_convergence": convergence_block(
                relation_counts[:, :, k], f"relation count skill {k}"),
        })

    total_relation = relation_counts.sum(axis=2)
    # a selected relation indicator: the one whose posterior probability is nearest 0.5
    indicator_blocks = []
    for k in range(N_SKILLS):
        marginal = np.array([[precedence_from_u(u[c, d, k]) for d in range(n_draws)]
                             for c in range(n_chains)]).reshape(-1, N_ROLES, N_ROLES)
        probability = marginal.mean(axis=0)
        off = ~np.eye(N_ROLES, dtype=bool)
        candidates = np.where(off, np.abs(probability - 0.5), 10.0)
        i, j = np.unravel_index(int(np.argmin(candidates)), candidates.shape)
        series = np.array([[float(precedence_from_u(u[c, d, k])[i, j])
                            for d in range(n_draws)] for c in range(n_chains)])
        indicator_blocks.append({
            "skill_index_unaligned": k, "relation": [int(i), int(j)],
            "posterior_probability": float(probability[i, j]),
            **convergence_block(series, f"relation {k} {i}->{j}")})
    varying = [b for b in indicator_blocks if not b.get("degenerate")]

    # latent-column symmetry: h(U) is invariant to permuting the d columns, so a column
    # swap in one chain and not another is a labelling artefact, not a disagreement
    column_gap = []
    for k in range(N_SKILLS):
        means = u[:, :, k, :, :].mean(axis=1)            # (chains, m, d)
        swapped = means[:, :, ::-1]
        column_gap.append(float(min(np.abs(means - means[0]).max(),
                                    np.abs(swapped - means[0]).max())))

    return {
        "per_skill": per_skill,
        "total_relation_count_convergence": convergence_block(
            total_relation, "total relation count"),
        "selected_relation_indicators": indicator_blocks,
        "max_relation_indicator_rhat": (
            float(max(b["rhat"] for b in varying)) if varying else None),
        "min_relation_indicator_bulk_ess": (
            float(min(b["bulk_ess"] for b in varying)) if varying else None),
        "n_degenerate_relation_indicators": len(indicator_blocks) - len(varying),
        "latent_column_symmetry": {
            "per_skill_min_gap_over_column_permutations": column_gap,
            "note": "h(U) is invariant to permuting the d latent columns, so a per-chain "
                    "U mean is compared under both column orders; the minimum gap is the "
                    "meaningful one",
        },
        "true_relation_counts": [int(precedence_from_u(corpus.u_true[k]).sum())
                                 for k in range(N_SKILLS)],
    }


# ------------------------------------------------------------------------- recovery
def recovery(data: dict, corpus) -> dict:
    labels = data["occurrence_labels"]
    n_chains, n_draws, n_traces, _ = labels.shape
    pooled = labels.reshape(n_chains * n_draws, n_traces, -1)
    u_pooled = data["u_draws"].reshape(-1, N_SKILLS, N_ROLES, 2)

    lengths = [t.length for t in corpus.train]
    true_keys = [t.true_key() for t in corpus.train]
    true_label_arrays = [t.true_occurrence_labels() for t in corpus.train]

    boundary = boundary_recovery(pooled, true_keys, lengths)
    skills = skill_recovery(pooled, true_label_arrays, N_SKILLS, true_keys)

    flat_true = np.concatenate(true_label_arrays)
    permutations = []
    for d in range(pooled.shape[0]):
        drawn = np.concatenate([pooled[d, n][:lengths[n]] for n in range(n_traces)])
        permutation, _ = skill_alignment(drawn, flat_true, N_SKILLS)
        permutations.append(permutation)
    structure = partial_order_recovery(u_pooled, permutations, corpus.u_true, N_SKILLS)

    # transitions and pi/P, aligned by the same per-draw permutation
    pi_aligned = np.zeros(N_SKILLS)
    p_aligned = np.zeros((N_SKILLS, N_SKILLS))
    pi_draws = data["pi_draws"].reshape(-1, N_SKILLS)
    p_draws = data["transition_draws"].reshape(-1, N_SKILLS, N_SKILLS)
    for d, permutation in enumerate(permutations):
        order = np.argsort(permutation)                 # inferred index playing true k
        pi_aligned += pi_draws[d][order]
        p_aligned += p_draws[d][np.ix_(order, order)]
    pi_aligned /= len(permutations)
    p_aligned /= len(permutations)

    return {
        "boundary": boundary,
        "skill": skills,
        "structure": structure,
        "transitions": {
            "pi_posterior_mean_aligned": pi_aligned.tolist(),
            "pi_true": corpus.pi_true.tolist(),
            "pi_max_absolute_error": float(np.abs(pi_aligned - corpus.pi_true).max()),
            "P_posterior_mean_aligned": p_aligned.tolist(),
            "P_true": corpus.p_true.tolist(),
            "P_max_absolute_error": float(np.abs(p_aligned - corpus.p_true).max()),
            "alignment": "each draw's pi and P permuted by that draw's Hungarian "
                         "assignment, so they are comparable with the truth",
        },
    }


# -------------------------------------------------------------- held-out prediction
def draw_set(data: dict, n: int, seed: int = 11) -> list:
    total = data["log_target"].size
    flat = {k: data[k].reshape((total,) + data[k].shape[2:])
            for k in ("u_draws", "pi_draws", "transition_draws")}
    scalars = {name: data[f"scalar_{name}"].reshape(total)
               for name in (*SCALAR_ORDER, "rho")}
    rng = np.random.default_rng(seed)
    index = rng.choice(total, size=min(n, total), replace=False)
    index.sort()
    return [{"u_by_skill": flat["u_draws"][i].astype(float),
             "pi": flat["pi_draws"][i].astype(float),
             "transition": flat["transition_draws"][i].astype(float),
             **{name: float(scalars[name][i]) for name in (*SCALAR_ORDER, "rho")}}
            for i in index]


def modal_h_representative(data: dict) -> dict:
    """A single retained draw whose induced orders are the modal ones, for every skill.

    This is a *representative draw in the modal induced-order cell*. It is deliberately
    NOT called a posterior-mean plug-in, and it is not one: no averaging happens anywhere
    in its construction.
    """
    total = data["log_target"].size
    u = data["u_draws"].reshape(total, N_SKILLS, N_ROLES, 2)
    keys = np.array([[precedence_from_u(u[i, k]).tobytes() for k in range(N_SKILLS)]
                     for i in range(total)], dtype=object)
    modal = []
    for k in range(N_SKILLS):
        counts: dict = {}
        for key in keys[:, k]:
            counts[key] = counts.get(key, 0) + 1
        modal.append(max(counts, key=counts.get))
    in_cell = np.array([all(keys[i, k] == modal[k] for k in range(N_SKILLS))
                        for i in range(total)])
    if not in_cell.any():
        return {"available": False,
                "why": "no retained draw sits in the modal cell for every skill "
                       "simultaneously"}
    candidates = np.where(in_cell)[0]
    # the candidate closest to the cell's own centre in the standardised scalars
    scalars = np.column_stack([data[f"scalar_{n}"].reshape(total)[candidates]
                               for n in (*SCALAR_ORDER, "rho")])
    centre = scalars.mean(axis=0)
    spread = scalars.std(axis=0, ddof=1)
    spread[spread == 0] = 1.0
    chosen = candidates[int(np.argmin(
        np.abs((scalars - centre) / spread).max(axis=1)))]
    return {
        "available": True, "draw_index": int(chosen),
        "cell_probability": float(in_cell.mean()),
        "n_draws_in_cell": int(in_cell.sum()),
        "naming": "representative posterior draw in the modal induced-order cell — NOT a "
                  "posterior-mean plug-in, and not described as one",
        "draw": {"u_by_skill": u[chosen].astype(float),
                 "pi": data["pi_draws"].reshape(total, N_SKILLS)[chosen].astype(float),
                 "transition": data["transition_draws"].reshape(
                     total, N_SKILLS, N_SKILLS)[chosen].astype(float),
                 **{name: float(data[f"scalar_{name}"].reshape(total)[chosen])
                    for name in (*SCALAR_ORDER, "rho")}},
    }


def negative_control_draw(data: dict) -> dict:
    """`h(E[U])`. A LABELLED NEGATIVE CONTROL, never the principal plug-in.

    Averaging continuous `U` across an order cell can collapse incomparabilities, so the
    induced order of the mean is not the mean of the induced orders and this quantity has
    no posterior interpretation. It is computed only so the size of that failure is
    visible rather than assumed.
    """
    total = data["log_target"].size
    u_mean = data["u_draws"].reshape(total, N_SKILLS, N_ROLES, 2).mean(axis=0)
    return {
        "u_by_skill": u_mean.astype(float),
        "pi": data["pi_draws"].reshape(total, N_SKILLS).mean(axis=0).astype(float),
        "transition": data["transition_draws"].reshape(
            total, N_SKILLS, N_SKILLS).mean(axis=0).astype(float),
        **{name: float(data[f"scalar_{name}"].reshape(total).mean())
           for name in (*SCALAR_ORDER, "rho")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictive-draws", type=int, default=N_PREDICTIVE_DRAWS)
    parser.add_argument("--skip-oracle", action="store_true")
    args = parser.parse_args()

    corpus = generate_corpus()
    unknown = load("unknown")
    unknown_summary = json.loads((OUT / "unknown_run_summary.json").read_text())
    oracle = None if args.skip_oracle else load("oracle")
    oracle_summary = (None if args.skip_oracle else
                      json.loads((OUT / "oracle_run_summary.json").read_text()))

    rng = np.random.default_rng(PAIR_SEED)
    lengths = [t.length for t in corpus.train]
    pair_sample = []
    while len(pair_sample) < N_COCLUSTER_PAIRS:
        n = int(rng.integers(len(lengths)))
        i, j = sorted(rng.choice(lengths[n], size=2, replace=False).tolist())
        if i != j:
            pair_sample.append((n, int(i), int(j)))

    print("[6E2] scalar diagnostics ...", flush=True)
    scalars = scalar_diagnostics(unknown, unknown_summary)
    print("[6E2] segmentation diagnostics ...", flush=True)
    segmentation = segmentation_diagnostics(unknown, corpus, unknown_summary)
    print("[6E2] skill diagnostics ...", flush=True)
    skills = skill_diagnostics(unknown, corpus, pair_sample)
    print("[6E2] structural diagnostics ...", flush=True)
    structure = structural_diagnostics(unknown, corpus)
    print("[6E2] recovery ...", flush=True)
    recovered = recovery(unknown, corpus)

    # ---- convergence gates --------------------------------------------------------------
    rhats = {f"{n}_rhat": rhat_of(scalars[n]) for n in (*SCALAR_ORDER, "rho")}
    rhats["log_target_rhat"] = rhat_of(scalars["log_target"])
    rhats["segment_count_rhat"] = rhat_of(segmentation["boundary_count_convergence"])
    rhats["boundary_indicator_rhat"] = segmentation["max_boundary_indicator_rhat"]
    rhats["relation_count_rhat"] = rhat_of(
        structure["total_relation_count_convergence"])
    rhats["relation_indicator_rhat"] = structure["max_relation_indicator_rhat"]
    rhats["transition_count_spectrum_rhat"] = rhat_of(skills["transition_count_spectrum"])
    rhats["co_clustering_rhat"] = skills["co_clustering"].get("max_rhat")
    for name, block in skills["pi_convergence"].items():
        rhats[f"{name}_rhat"] = rhat_of(block)
    for name, block in skills["P_convergence"].items():
        rhats[f"{name}_rhat"] = rhat_of(block)

    gates = {}
    for name, value in rhats.items():
        if value is None:
            gates[name] = {"value": None, "threshold": RHAT_GATE, "pass": True,
                           "note": "degenerate (a constant trace); recorded as such, "
                                   "never as an R-hat of 1.0"}
        elif math.isnan(value):
            gates[name] = {"value": None, "threshold": RHAT_GATE, "pass": False,
                           "note": "R-hat is UNDEFINED (NaN) — the statistic was "
                                   "attempted and produced no number. Reported as a "
                                   "failure, not as a degenerate coordinate and not as "
                                   "a pass."}
        else:
            gates[name] = {"value": value, "threshold": RHAT_GATE,
                           "pass": bool(value <= RHAT_GATE), "note": None}
    ess_values = {n: scalars[n]["bulk_ess"] for n in (*SCALAR_ORDER, "rho")
                  if not scalars[n].get("degenerate")}
    gates["min_scalar_bulk_ess"] = {
        "value": float(min(ess_values.values())), "threshold": MIN_BULK_ESS,
        "pass": bool(min(ess_values.values()) >= MIN_BULK_ESS), "comparison": ">="}
    mcse = {n: scalars[n]["mcse_over_sd"] for n in (*SCALAR_ORDER, "rho")
            if scalars[n].get("mcse_over_sd") is not None}
    gates["max_mcse_over_sd"] = {
        "value": float(max(mcse.values())), "threshold": MAX_MCSE_OVER_SD,
        "pass": bool(max(mcse.values()) <= MAX_MCSE_OVER_SD)}
    acceptance = unknown_summary["acceptance_post_burn_in"]
    tracked = [n for n in ("U", "rho", *SCALAR_ORDER)]
    observed = {n: float(np.mean([a[n] for a in acceptance if a.get(n) is not None]))
                for n in tracked}
    gates["acceptance_band"] = {
        "value": observed, "threshold": list(ACCEPTANCE_BAND),
        "pass": bool(all(ACCEPTANCE_BAND[0] <= v <= ACCEPTANCE_BAND[1]
                         for v in observed.values()))}
    convergence_pass = all(g["pass"] for g in gates.values())

    # ---- held-out prediction ------------------------------------------------------------
    heldout = corpus.traces("heldout")
    print(f"[6E2] held-out prediction over {len(heldout)} traces "
          f"({args.predictive_draws} draws) ...", flush=True)
    began = time.perf_counter()
    unknown_draws = draw_set(unknown, args.predictive_draws)
    unknown_predictive = heldout_predictive(
        heldout, unknown_draws, corpus.epsilon, corpus.delta_b, N_SKILLS,
        MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH, progress=max(1, args.predictive_draws // 4))
    unknown_predictive["runtime_seconds"] = time.perf_counter() - began

    truth_draw = [{"u_by_skill": corpus.u_true, "pi": corpus.pi_true,
                   "transition": corpus.p_true, **corpus.scalar_truth, "rho": 0.5}]
    truth_predictive = heldout_predictive(
        heldout, truth_draw, corpus.epsilon, corpus.delta_b, N_SKILLS,
        MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)

    modal = modal_h_representative(unknown)
    modal_predictive = None
    if modal.get("available"):
        modal_predictive = heldout_predictive(
            heldout, [modal["draw"]], corpus.epsilon, corpus.delta_b, N_SKILLS,
            MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)

    negative = negative_control_draw(unknown)
    negative_predictive = heldout_predictive(
        heldout, [negative], corpus.epsilon, corpus.delta_b, N_SKILLS,
        MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)

    oracle_predictive = None
    oracle_recovery = None
    if oracle is not None:
        print("[6E2] oracle-boundary control prediction ...", flush=True)
        oracle_predictive = heldout_predictive(
            heldout, draw_set(oracle, args.predictive_draws), corpus.epsilon,
            corpus.delta_b, N_SKILLS, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
        oracle_u = oracle["u_draws"].reshape(-1, N_SKILLS, N_ROLES, 2)
        identity = [np.arange(N_SKILLS)] * oracle_u.shape[0]
        oracle_recovery = {
            "structure": partial_order_recovery(oracle_u, identity, corpus.u_true,
                                                N_SKILLS),
            "scalars": {n: {"posterior_mean": float(oracle[f"scalar_{n}"].mean()),
                            "posterior_sd": float(oracle[f"scalar_{n}"].std(ddof=1)),
                            "q025": float(np.quantile(oracle[f"scalar_{n}"], 0.025)),
                            "q975": float(np.quantile(oracle[f"scalar_{n}"], 0.975)),
                            "true_value": TRUE_VALUES.get(n)}
                        for n in (*SCALAR_ORDER, "rho")},
            "alignment_note": "the control's labels are the true labels by construction, "
                              "so no Hungarian alignment is applied or needed",
        }

    per_trace_unknown = np.array(unknown_predictive["per_trace_log_predictive"])
    comparison = {
        "unknown_boundary_posterior_predictive": {
            k: unknown_predictive[k] for k in
            ("nll_per_trace", "nll_per_occurrence", "total_log_predictive",
             "predictive_interval_per_trace", "n_draws", "runtime_seconds")},
        "true_parameter_oracle": {
            k: truth_predictive[k] for k in
            ("nll_per_trace", "nll_per_occurrence", "total_log_predictive")},
        "modal_h_representative_draw": (
            None if modal_predictive is None else
            {**{k: modal_predictive[k] for k in
                ("nll_per_trace", "nll_per_occurrence", "total_log_predictive")},
             "naming": modal["naming"], "cell_probability": modal["cell_probability"]}),
        "negative_control_h_of_mean_U": {
            **{k: negative_predictive[k] for k in
               ("nll_per_trace", "nll_per_occurrence", "total_log_predictive")},
            "status": "LABELLED NEGATIVE CONTROL. h(E[U]) is not a valid plug-in: "
                      "averaging U inside an order cell can collapse incomparabilities, "
                      "so the induced order of the mean is not the mean of the induced "
                      "orders. Reported to size that failure, never as a result.",
            "induced_orders_of_mean_U": [
                precedence_from_u(negative["u_by_skill"][k]).tolist()
                for k in range(N_SKILLS)],
        },
    }
    if oracle_predictive is not None:
        per_trace_oracle = np.array(oracle_predictive["per_trace_log_predictive"])
        comparison["oracle_boundary_control"] = {
            k: oracle_predictive[k] for k in
            ("nll_per_trace", "nll_per_occurrence", "total_log_predictive",
             "predictive_interval_per_trace", "n_draws")}
        comparison["gap_from_oracle_boundary_control"] = {
            "nll_per_occurrence": (unknown_predictive["nll_per_occurrence"]
                                   - oracle_predictive["nll_per_occurrence"]),
            "nll_per_trace": (unknown_predictive["nll_per_trace"]
                              - oracle_predictive["nll_per_trace"]),
            "fraction_of_traces_favouring_unknown_boundary": float(
                (per_trace_unknown > per_trace_oracle).mean()),
            "note": "both marginalise (S, z) on the held-out traces, because held-out "
                    "boundaries are unknown to both. The control's advantage is entirely "
                    "in the TRAINING posterior it carries.",
        }

    # ---- verdicts -------------------------------------------------------------------------
    boundary_f1 = recovered["boundary"]["boundary_f1"]
    ari = recovered["skill"]["adjusted_rand_index"]["mean"]
    closure_f1 = recovered["structure"]["closure_f1_min"]
    scalar_hits = [scalars[n]["truth_in_95_credible_interval"] for n in SCALAR_ORDER]
    verdicts = {
        "stage_6e_sampler_correctness": "see stage6e0/6e1a/6e1b — not restated here",
        "stage_6e2_convergence": "PASS" if convergence_pass else "FAIL",
        "stage_6e2_boundary_recovery": (
            "PASS" if boundary_f1 >= 0.80 else
            "PARTIAL" if boundary_f1 >= 0.50 else "FAIL"),
        "stage_6e2_skill_label_recovery": (
            "PASS" if ari >= 0.70 else "PARTIAL" if ari >= 0.30 else "FAIL"),
        "stage_6e2_structural_recovery": (
            "PASS" if closure_f1 >= 0.95 else
            "PARTIAL" if closure_f1 >= 0.60 else "FAIL"),
        "stage_6e2_scalar_recovery": (
            "PASS" if all(scalar_hits) else
            "PARTIAL" if any(scalar_hits) else "FAIL"),
        "verdict_thresholds": {
            "boundary_f1": {"PASS": ">= 0.80", "PARTIAL": ">= 0.50"},
            "skill_ari": {"PASS": ">= 0.70", "PARTIAL": ">= 0.30"},
            "closure_f1_min": {"PASS": ">= 0.95", "PARTIAL": ">= 0.60"},
            "scalars": {"PASS": "truth inside every 95% interval",
                        "PARTIAL": "truth inside at least one"},
            "registered": "before the chains were analysed",
        },
    }
    occupancy_gap = max(o["max_per_chain_probability_gap"]
                        for o in segmentation["per_chain_occupancy"])
    verdicts["stage_6e2_identifiability"] = (
        "WELL IDENTIFIED" if (boundary_f1 >= 0.80 and ari >= 0.70
                              and recovered["skill"][
                                  "n_distinct_alignment_permutations"] == 1)
        else "MULTIMODAL" if occupancy_gap > 0.3
        else "PARTIALLY IDENTIFIED")

    # ---- write ---------------------------------------------------------------------------
    (OUT / "scalar_diagnostics.json").write_text(json.dumps(jsonable(scalars), indent=2))
    (OUT / "segmentation_diagnostics.json").write_text(
        json.dumps(jsonable(segmentation), indent=2))
    (OUT / "skill_diagnostics.json").write_text(json.dumps(jsonable(skills), indent=2))
    (OUT / "structural_diagnostics.json").write_text(
        json.dumps(jsonable(structure), indent=2))
    (OUT / "transition_diagnostics.json").write_text(json.dumps(jsonable({
        "pi_convergence": skills["pi_convergence"],
        "P_convergence": skills["P_convergence"],
        "mean_transition_counts": skills["mean_transition_counts"],
        "true_transition_counts": skills["true_transition_counts"],
        "transition_count_spectrum": skills["transition_count_spectrum"],
        "recovery": recovered["transitions"],
        "no_terminal_transition": True,
    }), indent=2))
    (OUT / "recovery_results.json").write_text(json.dumps(jsonable({
        **recovered, "oracle_boundary_control": oracle_recovery,
        "verdicts": verdicts}), indent=2))
    (OUT / "heldout_results.json").write_text(json.dumps(jsonable(comparison), indent=2))
    (OUT / "convergence_gates.json").write_text(json.dumps(jsonable({
        "gates": gates, "all_pass": convergence_pass,
        "source_commit": source_commit(),
        "registered_thresholds": {"rhat": RHAT_GATE, "min_bulk_ess": MIN_BULK_ESS,
                                  "max_mcse_over_sd": MAX_MCSE_OVER_SD,
                                  "acceptance_band": list(ACCEPTANCE_BAND)},
    }), indent=2))

    print("\n--- convergence gates ---")
    for name, gate in sorted(gates.items()):
        value = gate["value"]
        shown = ("n/a" if value is None else
                 f"{value:.5f}" if isinstance(value, float) else str(value))
        print(f"  {name:36s} {shown:>28s} -> {'PASS' if gate['pass'] else 'FAIL'}")
    print(f"  convergence: {'PASS' if convergence_pass else 'FAIL'}")
    print("\n--- recovery ---")
    print(f"  boundary F1 {boundary_f1:.4f}  precision "
          f"{recovered['boundary']['boundary_precision']:.4f}  recall "
          f"{recovered['boundary']['boundary_recall']:.4f}  Brier "
          f"{recovered['boundary']['boundary_brier_score']:.4f}")
    print(f"  skill ARI {ari:.4f}  NMI "
          f"{recovered['skill']['normalised_mutual_information']['mean']:.4f}  "
          f"occurrence accuracy "
          f"{recovered['skill']['occurrence_aligned_accuracy']['mean']:.4f}")
    print(f"  closure F1 (min over skills) {closure_f1:.4f}  reduction F1 "
          f"{recovered['structure']['reduction_f1_min']:.4f}")
    for name in SCALAR_ORDER:
        s = scalars[name]
        print(f"  {name:12s} {s['posterior_mean']:.4f} "
              f"[{s['q025']:.4f}, {s['q975']:.4f}] true {s['true_value']} "
              f"-> {'in' if s['truth_in_95_credible_interval'] else 'OUT'}")
    print("\n--- held-out ---")
    print(f"  unknown-boundary NLL/occurrence "
          f"{unknown_predictive['nll_per_occurrence']:.5f}")
    if oracle_predictive:
        print(f"  oracle-control  NLL/occurrence "
              f"{oracle_predictive['nll_per_occurrence']:.5f}")
    print(f"  true-parameter  NLL/occurrence "
          f"{truth_predictive['nll_per_occurrence']:.5f}")
    print(f"  negative control h(E[U]) NLL/occurrence "
          f"{negative_predictive['nll_per_occurrence']:.5f}")
    print("\n--- verdicts ---")
    for k, v in verdicts.items():
        if isinstance(v, str):
            print(f"  {k:36s} {v}")
    print(f"\n[6E2] wrote {OUT}")


if __name__ == "__main__":
    main()
