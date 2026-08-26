#!/usr/bin/env python3
"""Short three-arm smoke comparison at selected rungs.

This is the pre-production check the review asked for: before any sweep is launched, show
end-to-end that the recurrent likelihood actually buys something over knowing only which
CPAs each skill can emit. It is a **smoke test, not a result** — one chain, one replicate,
one corpus seed per rung, far too few sweeps for a posterior. Its job is to catch a
baseline that is broken, trivially equal, or accidentally better.

Both arms share their data, seeds, initialisation, sweep schedule and transition
treatment; `hpop.mcmc_cpa.ladder_runner` is the single runner and the arm is one argument.
Three arms, and three contrasts that mean different things:

    oracle-order  -  support-only     how much the AVAILABLE order information is worth
    learned-order -  support-only     the REALISED end-to-end gain
    oracle-order  -  learned-order    the INFERENCE gap

`support-only` and `oracle-order` hold `U` fixed; `learned-order` starts away from the
truth and infers it. Structure recovery is reported only for `learned-order`: the
support-only score never reads `U`, and the oracle arm is handed it.

**The oracle-order arm scores at the true `U`.** Each arm gets ground-truth side
information of its own kind — the baseline the true supports, the oracle arm the true
supports *and* the true within-skill order. So `oracle - support` is an **oracle
information diagnostic**: what the order is worth to a sampler that is handed it. It is
*not* a strict upper bound on `learned - support`. An inferred `U` is not obliged to be
less useful than the true one at every finite sweep count, and averaging over a posterior
is a different operation from plugging in a point truth. What the learned arm achieves is
an empirical question, which is why it is an arm and not an argument.

    python scripts/k_ladder/smoke_three_arms.py --rungs 3 10 30 --sweeps 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_cpa.corpus import generate_ladder_corpus              # noqa: E402
from hpop.mcmc_cpa.ladder_runner import (ARMS, LEARNED_ORDER,           # noqa: E402
                                         ORACLE_ORDER, SUPPORT_ONLY,
                                         run_ladder_chain)
from hpop.mcmc_cpa.nested_library import draw_master_library         # noqa: E402
from hpop.mcmc_original.stage6e_state import Stage6EModel            # noqa: E402

ARM_ORDER = (SUPPORT_ONLY, ORACLE_ORDER, LEARNED_ORDER)


def truth_of(corpus, split="train"):
    """Per-position true skill and the set of internal boundaries, per trace."""
    labels, boundaries = [], []
    for record in getattr(corpus, split):
        per_position = np.empty(len(record.cpa), dtype=int)
        cursor, ends = 0, []
        for width, skill in zip(record.widths, record.labels):
            per_position[cursor:cursor + width] = skill
            cursor += width
            ends.append(cursor)
        labels.append(per_position)
        boundaries.append(set(ends[:-1]))
    return labels, boundaries


def score(result, truth_labels, truth_boundaries, n_skills):
    """Boundary F1 and permutation-aligned per-occurrence skill accuracy.

    Skill identity is not identified by the likelihood, so labels are aligned by the
    assignment that maximises agreement (Hungarian) before accuracy is read off. Pooling
    over retained draws is deliberate: it reports the posterior's typical labelling, not
    a single draw's.
    """
    from scipy.optimize import linear_sum_assignment

    draws = result["draws"]
    hit = false_pos = missed = 0
    confusion = np.zeros((n_skills, n_skills))
    for index in range(len(draws["boundaries"])):
        for trace, (found, labels) in enumerate(zip(draws["boundaries"][index],
                                                    draws["labels"][index])):
            found_set = set(found)
            truth_set = truth_boundaries[trace]
            hit += len(found_set & truth_set)
            false_pos += len(found_set - truth_set)
            missed += len(truth_set - found_set)
            starts = [0] + list(found)
            ends = list(found) + [len(truth_labels[trace])]
            for (a, b), label in zip(zip(starts, ends), labels):
                for position in range(a, b):
                    confusion[truth_labels[trace][position], label] += 1
    precision = hit / max(hit + false_pos, 1)
    recall = hit / max(hit + missed, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    rows, cols = linear_sum_assignment(-confusion)
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "skill_accuracy": float(confusion[rows, cols].sum() / max(confusion.sum(), 1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rungs", type=int, nargs="+", default=[3, 30])
    parser.add_argument("--sweeps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--library-seed", type=int, default=0)
    parser.add_argument("--corpus-replicate", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "k_ladder" / "smoke_three_arms.json")
    args = parser.parse_args()

    library, library_meta = draw_master_library(args.library_seed)
    rows, records = [], []

    header = (f"{'K':>4} {'arm':>13} {'sec':>8} {'draws':>6} "
              f"{'bnd F1':>8} {'skill acc':>10} {'moved':>8}")
    print(header)
    print("-" * len(header))

    for n_skills in args.rungs:
        corpus = generate_ladder_corpus(library, n_skills, args.corpus_replicate)
        u_by_skill, role_maps = library.prefix(n_skills)
        truth_labels, truth_boundaries = truth_of(corpus)
        model = Stage6EModel(
            traces=corpus.traces("train"), epsilon=args.epsilon, delta_b=0.15,
            n_skills=n_skills, n_roles=library.n_roles, min_width=3, max_width=12,
            infer_pi_P=True, eta_initial=1.0, eta_transition=1.0)

        for arm in ARM_ORDER:
            began = time.perf_counter()
            result = run_ladder_chain(arm, model, role_maps, u_by_skill, chain=0,
                                      sweeps=args.sweeps, warmup=args.warmup,
                                      seed=90_000 + n_skills, epsilon=args.epsilon,
                                      thin=args.thin)
            metrics = score(result, truth_labels, truth_boundaries, n_skills)
            record = {
                "K": n_skills, "arm": arm,
                "seconds": result["seconds"],
                "wall_seconds": time.perf_counter() - began,
                "retained_draws": result["retained_draws"],
                "ffbs_states_changed_total": result["ffbs_states_changed_total"],
                "structure_recovery": result["structure_recovery"],
                "u_held_fixed": result["u_held_fixed"],
                **metrics,
            }
            records.append(record)
            print(f"{n_skills:>4} {arm:>13} {result['seconds']:>8.1f} "
                  f"{result['retained_draws']:>6} {metrics['boundary_f1']:>8.4f} "
                  f"{metrics['skill_accuracy']:>10.4f} "
                  f"{result['ffbs_states_changed_total']:>8}")

    CONTRASTS = (
        ("available order information", ORACLE_ORDER, SUPPORT_ONLY),
        ("realised end-to-end gain", LEARNED_ORDER, SUPPORT_ONLY),
        ("inference gap", ORACLE_ORDER, LEARNED_ORDER),
    )
    print()
    print(f"{'contrast':>28} {'K':>4} {'d boundary F1':>15} {'d skill acc':>13}")
    print("-" * 64)
    verdict = []
    for label, left, right in CONTRASTS:
        for n_skills in args.rungs:
            a = next(r for r in records if r["K"] == n_skills and r["arm"] == left)
            b = next(r for r in records if r["K"] == n_skills and r["arm"] == right)
            delta_f1 = a["boundary_f1"] - b["boundary_f1"]
            delta_acc = a["skill_accuracy"] - b["skill_accuracy"]
            rows.append({"contrast": label, "left": left, "right": right,
                         "K": n_skills, "delta_boundary_f1": delta_f1,
                         "delta_skill_accuracy": delta_acc})
            if left == ORACLE_ORDER and right == SUPPORT_ONLY:
                verdict.append(delta_f1 > 0 and delta_acc > 0)
            print(f"{label:>28} {n_skills:>4} {delta_f1:>15.4f} {delta_acc:>13.4f}")

    print()
    if all(verdict):
        print("SMOKE: oracle-order beats support-only at every rung tested.")
    else:
        print("SMOKE: NOT every rung favours oracle-order over support-only — "
              "investigate before\n       launching production sweeps.")
    print("NOTE: one chain, one replicate, few sweeps. Not a result; a sanity check.")
    print("NOTE: oracle-order scores at the TRUE U. `oracle - support` is an ORACLE "
          "INFORMATION\n      DIAGNOSTIC, not a bound on `learned - support`.")
    print("NOTE: no trend in K is claimed. Report the largest observed gap and its rung.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema": "k-ladder-smoke/1.0.0",
        "settings": vars(args) | {"out": str(args.out)},
        "library": library_meta,
        "runs": records,
        "deltas": rows,
        "all_rungs_favour_full": bool(all(verdict)),
        "caveat": ("Smoke test only: one chain, one corpus replicate, few sweeps. "
                   "oracle-order scores at the TRUE U, so oracle-minus-support is an "
                   "ORACLE INFORMATION DIAGNOSTIC -- what the available order "
                   "information is worth to a sampler handed it -- and NOT a strict "
                   "upper bound on learned-minus-support. Structure recovery is not "
                   "applicable to support-only because its score does not read U. No "
                   "trend in K is claimed; report the largest observed gap and its "
                   "rung."),
    }, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
