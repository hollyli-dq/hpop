"""Stage 6E2 — supersede the label-indexed convergence gates, and freeze the 50k modes.

    PYTHONPATH=src python scripts/stage6e2_register_invariant_gates.py

Two things happen here, both of which must happen **before** any continuation result is
examined.

## 1. The supersession

The originally registered convergence gate list included `pi[k]` and `P[h,k]` R-hat, and
per-skill relation-count and relation-indicator R-hat. Those are statistics of quantities
the Stage 6E2 posterior does not identify: `pi` and `P` are inferred under symmetric
Dirichlet priors and the `U_k` are exchangeable, so for any skill permutation `Q`

    pi' = Q pi,   P' = Q P Q^T,   U'_k = U_{Q^{-1}(k)}

leaves the posterior exactly invariant, and the posterior has `K! = 6` equivalent modes.
Four chains in four different labellings would fail those gates however well the sampler
had converged.

This is a **correction of a statistic that measures the wrong thing**, which is the only
kind of gate change permitted, and it is recorded rather than done quietly. It is *not* a
relaxation: the threshold stays at `Rhat <= 1.01`, the replacement list is longer than the
one it replaces, and — as the accompanying run shows — the superseded gates and their
replacements both fail. Nothing is rescued by the change.

## 2. The 50k mode table

The four chains at 50,000 sweeps are frozen into a fixed table so that the later blocks
can be compared against it rather than against a memory. Recovery quantities appear in it
**labelled as recovery diagnostics only**, and are not interpreted while the chains have
not converged.
"""

from __future__ import annotations

import json
import math
import platform
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
from hpop.mcmc_original.stage6e_frozen import N_ROLES, N_SKILLS                # noqa: E402
from hpop.mcmc_original.stage6e_invariant import (                             # noqa: E402
    INVARIANT_GATE_NAMES, assert_invariance, invariant_convergence,
    sorted_row_entropies, sorted_pi, transition_eigenvalues,
)

OUT = ROOT / "results" / "mcmc_original" / "stage6e2_unknown_boundary_full_seed0"
RHAT_GATE = 1.01

SUPERSEDED = {
    "pi[0]_rhat": "pi is only identified up to a skill permutation",
    "pi[1]_rhat": "pi is only identified up to a skill permutation",
    "pi[2]_rhat": "pi is only identified up to a skill permutation",
    "P[0,1]_rhat": "P is only identified up to Q P Q^T",
    "P[0,2]_rhat": "P is only identified up to Q P Q^T",
    "P[1,0]_rhat": "P is only identified up to Q P Q^T",
    "P[1,2]_rhat": "P is only identified up to Q P Q^T",
    "P[2,0]_rhat": "P is only identified up to Q P Q^T",
    "P[2,1]_rhat": "P is only identified up to Q P Q^T",
    "relation_indicator_rhat": "a relation indicator is indexed by an unaligned skill, so "
                               "it is not identified; it also rank-normalises to a "
                               "meaningless value when nearly constant in some chains",
}


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


def mode_table(data: dict, corpus) -> list:
    """One row per chain: the fixed table later blocks are compared against."""
    n_chains, n_draws = data["log_target"].shape
    lengths = [t.length for t in corpus.train]
    true_keys = [t.true_key() for t in corpus.train]
    rows = []
    for c in range(n_chains):
        labels = data["occurrence_labels"][c]
        unique_segmentations, unique_h = [], set()
        for n in range(min(len(lengths), 20)):
            unique_segmentations.append(
                len({labels_to_key(labels[d, n]) for d in range(n_draws)}))
        for d in range(n_draws):
            unique_h.add(tuple(precedence_from_u(data["u_draws"][c, d, k]).tobytes()
                               for k in range(N_SKILLS)))
        recovery = boundary_recovery(labels, true_keys, lengths)
        rows.append({
            "chain": c,
            "mean_log_posterior": float(data["log_target"][c].mean()),
            "max_log_posterior_visited": float(data["log_target"][c].max()),
            "min_log_posterior_visited": float(data["log_target"][c].min()),
            "total_relation_count_mean": float(
                data["relation_counts"][c].sum(axis=1).mean()),
            "sorted_per_skill_relation_counts": np.sort(
                data["relation_counts"][c], axis=1).mean(axis=0).tolist(),
            "total_segments_mean": float(data["segment_counts"][c].sum(axis=1).mean()),
            "sorted_pi_mean": sorted_pi(data["pi_draws"][c][None])[0].mean(axis=0).tolist(),
            "P_eigenvalues_mean": transition_eigenvalues(
                data["transition_draws"][c][None])[0].mean(axis=0).tolist(),
            "P_sorted_row_entropies_mean": sorted_row_entropies(
                data["transition_draws"][c][None])[0].mean(axis=0).tolist(),
            "lambda_rep_mean": float(data["scalar_lambda_rep"][c].mean()),
            "beta_mean": float(data["scalar_beta"][c].mean()),
            "omega_mean": float(data["scalar_omega"][c].mean()),
            "lambda_back_mean": float(data["scalar_lambda_back"][c].mean()),
            "rho_mean": float(data["scalar_rho"][c].mean()),
            "unique_induced_H_joint_states": len(unique_h),
            "unique_segmentations_first20_traces": {
                "mean": float(np.mean(unique_segmentations)),
                "min": int(min(unique_segmentations)),
                "max": int(max(unique_segmentations))},
            "RECOVERY_DIAGNOSTIC_ONLY_boundary_f1": recovery["boundary_f1"],
            "RECOVERY_DIAGNOSTIC_ONLY_note":
                "a recovery quantity, recorded per chain to characterise the modes. NOT "
                "interpreted while the chains have not converged, and never used as a "
                "convergence statistic.",
        })
    return rows


def main() -> None:
    corpus = generate_corpus()
    path = OUT / "chains.npz"
    preserved = OUT / "chains_block1_50k.npz"
    data = {k: v for k, v in np.load(preserved if preserved.exists() else path).items()}
    n_chains, n_draws = data["log_target"].shape
    if n_draws != 7000:
        print(f"[6E2] WARNING: expected the 50,000-sweep block (7,000 draws per chain), "
              f"found {n_draws}. Using it anyway; check which block this is.")

    invariance = assert_invariance(data, N_SKILLS)
    convergence = invariant_convergence(data, N_SKILLS, RHAT_GATE)

    registration = {
        "registered_before_any_continuation_result_was_examined": True,
        "source_commit": source_commit(),
        "python": platform.python_version(), "numpy": np.__version__,
        "why": "the Stage 6E2 posterior is EXACTLY label-exchangeable: pi and P are "
               "inferred under symmetric Dirichlet(eta=1) priors and the U_k are "
               "exchangeable, so for any skill permutation Q, (pi, P, U, z) -> "
               "(Q pi, Q P Q^T, U o Q^-1, Q z) leaves the posterior unchanged and there "
               "are K! = 6 equivalent modes.",
        "contrast_with_stage6e1b": "Stage 6E1B FIXED pi and P and chose them asymmetric "
                                   "precisely so that no relabelling was a symmetry, and "
                                   "verified it with label_permutation_audit before "
                                   "comparing. There, per-skill R-hat was meaningful. "
                                   "Here pi and P are inferred, so the symmetry returns.",
        "superseded_gates": SUPERSEDED,
        "superseded_status": "These measured a quantity the model does not identify. This "
                             "is a correction of a mis-specified statistic, not a "
                             "relaxation: the threshold is unchanged at Rhat <= 1.01, the "
                             "replacement list is LONGER, and both the superseded gates "
                             "and their replacements fail on the 50,000-sweep block, so "
                             "nothing is rescued by the change.",
        "replacement_gates": list(INVARIANT_GATE_NAMES),
        "threshold": RHAT_GATE,
        "invariance_check": invariance,
        "alignment_policy": "Hungarian alignment to the true labels is a RECOVERY device "
                            "only. It may never be applied to a convergence statistic, "
                            "because aligning each draw to the truth would make four "
                            "chains in four different label permutations look identical "
                            "and would turn a genuine multimodality into an apparent pass.",
        "not_doing": ["no scalar proposal retuning — the pilot's diagnosis was a joint "
                      "(S, z, U, theta) multimodality, not a tiny-step pathology, and "
                      "enlarging a random-walk scale cannot close a 257-nat gap between "
                      "chains",
                      "no FFBS in Stage 6E2 — the registered algorithm is the local move "
                      "kernel and it stays; poor mixing here is the MOTIVATION for Step 7, "
                      "not something to paper over by importing Step 7 early",
                      "no restart-and-select, no reinitialisation, no seed change"],
        "continuation_plan": "registered section 14 blocks only: 50k -> 75k -> 100k -> "
                             "125k -> 150k, same chains, same seeds, resuming from saved "
                             "state and RNG.",
    }
    (OUT / "gate_supersession.json").write_text(json.dumps(jsonable(registration),
                                                           indent=2))
    (OUT / "invariant_convergence_50k.json").write_text(json.dumps(jsonable({
        "block": "50,000 sweeps, 7,000 retained draws per chain",
        "gates": convergence["gates"], "all_pass": convergence["all_pass"],
        "n_gates": convergence["n_gates"],
        "frozen_coordinates": convergence["frozen_coordinates"],
        "n_frozen": convergence["n_frozen"],
        "frozen_note": "a coordinate with ZERO within-chain variance in one or more "
                       "chains, while the chains disagree, is frozen rather than slowly "
                       "mixing. R-hat then divides a real between-chain variance by ~0 "
                       "and returns an astronomical number; the condition is named here "
                       "instead.",
    }), indent=2))
    (OUT / "mode_diagnostics_50k.json").write_text(json.dumps(jsonable({
        "block": "50,000 sweeps", "n_chains": n_chains, "n_draws_per_chain": n_draws,
        "purpose": "a fixed table the later continuation blocks are compared against, so "
                   "mode structure is tracked rather than remembered",
        "chains": mode_table(data, corpus),
    }), indent=2))

    print(f"[6E2] invariance check: worst departure "
          f"{invariance['worst_overall']:.2e} -> "
          f"{'PASS' if invariance['pass'] else 'FAIL'} "
          f"(permutation {invariance['permutation']})")
    print(f"[6E2] permutation-invariant convergence on the 50k block:")
    for name, gate in convergence["gates"].items():
        value = gate["value"]
        shown = "n/a" if value is None else f"{value:.5f}"
        print(f"       {name:28s} {shown:>10s} -> "
              f"{'PASS' if gate['pass'] else 'FAIL'}")
    print(f"[6E2] all invariant gates pass: {convergence['all_pass']}")
    print(f"[6E2] wrote gate_supersession.json, invariant_convergence_50k.json, "
          f"mode_diagnostics_50k.json")


if __name__ == "__main__":
    main()
