"""Stage 6E2 — evaluate one rung of the registered §14 continuation ladder.

    PYTHONPATH=src python scripts/stage6e2_ladder_check.py

Reads whatever `chains.npz` currently holds, applies the **pre-registered**
permutation-invariant gates and the **pre-registered** structural-locking criteria, and
says what the registered rules require next. It decides nothing on its own: both the gate
list (`gate_supersession.json`) and the stopping rule (`interpretation_rule.json`) were
frozen before any continuation result existed, and this script only evaluates them.

The ladder is `50k -> 75k -> 100k -> 125k -> 150k`, same chains, same seeds, resuming from
saved state and RNG. Nothing else may change between rungs: not the target, not a proposal
scale, not a gate, not a seed.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.latent_poset import precedence_from_u                  # noqa: E402
from hpop.mcmc_original.stage6e_corpus import generate_corpus                  # noqa: E402
from hpop.mcmc_original.stage6e_diagnostics import (                           # noqa: E402
    boundary_recovery, labels_to_key,
)
from hpop.mcmc_original.stage6e_frozen import N_SKILLS                         # noqa: E402
from hpop.mcmc_original.stage6e_invariant import (                             # noqa: E402
    assert_invariance, invariant_convergence, invariant_summaries,
    sorted_row_entropies, sorted_pi, transition_eigenvalues,
)

OUT = ROOT / "results" / "mcmc_original" / "stage6e2_unknown_boundary_full_seed0"
LADDER = (50_000, 75_000, 100_000, 125_000, 150_000)
RHAT_GATE = 1.01


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


def structural_locking(data: dict) -> dict:
    """The three pre-registered criteria, evaluated exactly as registered."""
    summaries = invariant_summaries(data, N_SKILLS)
    relation = np.asarray(data["relation_counts"], dtype=float)
    total = relation.sum(axis=2)
    per_skill_sorted = np.sort(relation, axis=2)

    within = np.concatenate([total.std(axis=1)[:, None],
                             per_skill_sorted.std(axis=1)], axis=1)   # (chains, 1+K)
    chains_frozen = int((within.min(axis=1) < 0.01).sum())
    a = chains_frozen >= 2

    means = total.mean(axis=1)
    spread_relation = float(np.ptp(means))
    b = spread_relation > 1.0

    log_means = summaries["log_posterior"].mean(axis=1)
    spread_log = float(np.ptp(log_means))
    c = spread_log > 20.0

    return {
        "A_frozen_structure": {
            "criterion": "relation counts with within-chain sd < 0.01 in >= 2 chains",
            "n_chains_frozen": chains_frozen,
            "per_chain_min_within_sd": within.min(axis=1).tolist(),
            "holds": bool(a)},
        "B_disagreeing_structure": {
            "criterion": "spread of chain mean total relation count > 1.0",
            "per_chain_mean_total_relations": means.tolist(),
            "spread": spread_relation, "holds": bool(b)},
        "C_log_posterior_gap": {
            "criterion": "spread of chain mean invariant log posterior > 20 nats",
            "per_chain_mean_log_posterior": log_means.tolist(),
            "spread_nats": spread_log, "holds": bool(c)},
        "conjunction_holds": bool(a and b and c),
    }


def mode_table(data: dict, corpus) -> list:
    n_chains, n_draws = data["log_target"].shape
    lengths = [t.length for t in corpus.train]
    true_keys = [t.true_key() for t in corpus.train]
    rows = []
    for c in range(n_chains):
        labels = data["occurrence_labels"][c]
        unique_h = {tuple(precedence_from_u(data["u_draws"][c, d, k]).tobytes()
                          for k in range(N_SKILLS)) for d in range(n_draws)}
        unique_segmentations = [
            len({labels_to_key(labels[d, n]) for d in range(n_draws)})
            for n in range(min(len(lengths), 20))]
        rows.append({
            "chain": c,
            "mean_log_posterior": float(data["log_target"][c].mean()),
            "max_log_posterior_visited": float(data["log_target"][c].max()),
            "total_relation_count_mean": float(
                data["relation_counts"][c].sum(axis=1).mean()),
            "total_relation_count_within_sd": float(
                data["relation_counts"][c].sum(axis=1).std()),
            "sorted_per_skill_relation_counts": np.sort(
                data["relation_counts"][c], axis=1).mean(axis=0).tolist(),
            "total_segments_mean": float(data["segment_counts"][c].sum(axis=1).mean()),
            "sorted_pi_mean": sorted_pi(
                data["pi_draws"][c][None])[0].mean(axis=0).tolist(),
            "P_eigenvalues_mean": transition_eigenvalues(
                data["transition_draws"][c][None])[0].mean(axis=0).tolist(),
            "P_sorted_row_entropies_mean": sorted_row_entropies(
                data["transition_draws"][c][None])[0].mean(axis=0).tolist(),
            "lambda_rep_mean": float(data["scalar_lambda_rep"][c].mean()),
            "beta_mean": float(data["scalar_beta"][c].mean()),
            "unique_induced_H_joint_states": len(unique_h),
            "unique_segmentations_first20_traces_mean": float(
                np.mean(unique_segmentations)),
            "RECOVERY_DIAGNOSTIC_ONLY_boundary_f1":
                boundary_recovery(labels, true_keys, lengths)["boundary_f1"],
        })
    return rows


def main() -> None:
    corpus = generate_corpus()
    data = {k: v for k, v in np.load(OUT / "chains.npz").items()}
    history = json.loads((OUT / "continuation_history.json").read_text())
    sweeps = max(b["sweeps_to"] for b in history["unknown"])
    n_chains, n_draws = data["log_target"].shape

    invariance = assert_invariance(data, N_SKILLS)
    convergence = invariant_convergence(data, N_SKILLS, RHAT_GATE)
    locking = structural_locking(data)

    failed = [n for n, g in convergence["gates"].items() if not g["pass"]]
    at_max = sweeps >= LADDER[-1]
    if convergence["all_pass"]:
        action = "STOP: every pre-registered invariant gate passes. Proceed to recovery."
    elif at_max and locking["conjunction_holds"]:
        action = ("STOP: the registered maximum of 150,000 sweeps is reached and all "
                  "three structural-locking criteria hold. The pre-registered verdict is "
                  "FAIL / MULTIMODAL with diagnosis '(S, z)--U structural locking'. "
                  "Describing this as insufficient run length is ruled out by the rule "
                  "registered before any continuation result existed.")
    elif at_max:
        action = ("STOP: the registered maximum is reached and gates still fail, but the "
                  "structural-locking conjunction does NOT hold. Report FAIL and describe "
                  "the actual pattern; do not assert structural locking.")
    else:
        nxt = next(s for s in LADDER if s > sweeps)
        action = (f"CONTINUE to {nxt:,} per section 14: "
                  f"scripts/stage6e2_formal_chains.py --run unknown --resume")

    payload = {
        "sweeps": sweeps, "n_chains": n_chains, "n_draws_per_chain": n_draws,
        "ladder": list(LADDER), "at_registered_maximum": at_max,
        "invariance_check": {"worst_departure": invariance["worst_overall"],
                             "pass": invariance["pass"]},
        "gates": convergence["gates"],
        "all_invariant_gates_pass": convergence["all_pass"],
        "n_failed_gates": len(failed), "failed_gates": failed,
        "frozen_coordinates": convergence["frozen_coordinates"],
        "structural_locking": locking,
        "required_next_action": action,
        "recovery_status": "computed but NOT interpreted while the chains have not "
                           "converged",
    }
    (OUT / f"ladder_check_{sweeps // 1000}k.json").write_text(
        json.dumps(jsonable(payload), indent=2))
    (OUT / f"mode_diagnostics_{sweeps // 1000}k.json").write_text(
        json.dumps(jsonable({"sweeps": sweeps, "chains": mode_table(data, corpus)}),
                   indent=2))

    print(f"[6E2 ladder] {sweeps:,} sweeps, {n_draws:,} draws/chain "
          f"({n_chains} chains)")
    print(f"  invariance check worst departure {invariance['worst_overall']:.2e} -> "
          f"{'PASS' if invariance['pass'] else 'FAIL'}")
    print(f"  invariant gates: {len(convergence['gates']) - len(failed)}"
          f"/{len(convergence['gates'])} pass")
    if failed:
        print(f"  failing: {', '.join(failed[:8])}"
              f"{' ...' if len(failed) > 8 else ''}")
    print(f"  structural locking  A={locking['A_frozen_structure']['holds']} "
          f"({locking['A_frozen_structure']['n_chains_frozen']}/{n_chains} chains frozen) "
          f" B={locking['B_disagreeing_structure']['holds']} "
          f"(relation spread {locking['B_disagreeing_structure']['spread']:.2f}) "
          f" C={locking['C_log_posterior_gap']['holds']} "
          f"(log-posterior spread {locking['C_log_posterior_gap']['spread_nats']:.1f} nats)"
          f"  -> conjunction {locking['conjunction_holds']}")
    print(f"\n  {action}")


if __name__ == "__main__":
    main()
