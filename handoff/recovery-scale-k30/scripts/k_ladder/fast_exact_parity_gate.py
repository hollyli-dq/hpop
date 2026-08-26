#!/usr/bin/env python3
"""Is the skill-local ("fast") table refresh EXACTLY the all-skills ("exact") one?

`table_source="fast"` rebuilds only the skill whose `U` moved and reuses the other `K-1`
columns. At `K = 30` that is the difference between a usable learned-order arm and an
unusable one. It is also exactly the kind of optimisation that can be *almost* right: a
column that is stale by one ulp changes a score, which changes a log ratio, which changes
an accept/reject decision, which changes the chain -- and nothing downstream would say so.

So this gate does not ask whether fast is close. It asks whether it is identical, at five
levels, and then whether the decisions built on it are identical too:

1. **finite masks** -- the same candidate blocks are feasible under both;
2. **entrywise scores** -- every finite entry equal, checked as an exact bit comparison
   and reported as a max absolute difference;
3. **one-skill `U`-update deltas** -- the collapsed likelihood delta a `U` move is
   accepted or rejected on;
4. **total MH acceptance ratios** -- `log alpha` for real proposed moves;
5. **short-chain accept/reject trajectories** -- the actual sequence of decisions.

Run at `K = 3, 10, 20, 30`. **A single mismatch anywhere is a production blocker**: if
fast is approximate and changes MH decisions, it must not be used without an exact
correction, and this script says `RESULT: BLOCKED` rather than reporting a small number.

    python scripts/k_ladder/fast_exact_parity_gate.py --rungs 3 10 20 30
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_cpa.block_tables import CPABlockScoreTable            # noqa: E402
from hpop.mcmc_cpa.corpus import generate_ladder_corpus              # noqa: E402
from hpop.mcmc_cpa.nested_library import draw_master_library         # noqa: E402
from hpop.mcmc_original.semi_markov_ffbs import forward               # noqa: E402
from hpop.mcmc_original.stage6e_state import Stage6EModel            # noqa: E402
from hpop.mcmc_original.transitions import log_transition_matrix      # noqa: E402
from hpop.mcmc_cpa.gamma_coupling import draw_master_transitions      # noqa: E402


def _tables(model, role_maps, epsilon):
    return CPABlockScoreTable(traces=model.traces, epsilon=float(epsilon),
                              role_maps=role_maps, min_width=model.min_width,
                              max_width=model.max_width)


def log_z_per_trace(table_backend, model, log_pi, log_p) -> np.ndarray:
    """The collapsed log normaliser per trace, through the sealed forward recursion.

    This is the quantity a `U` move is actually accepted or rejected on, so comparing it
    is what makes checks 3-5 about decisions rather than about arrays.
    """
    return np.array([
        forward(table, log_pi, log_p, model.delta_b, model.max_width,
                model.min_width).log_normalizer
        for table in table_backend.tables], dtype=float)


def _propose(rng, u, skill, scale):
    """One row proposal on one skill, the shape the registered U kernel makes.

    The scale matters more than it looks. The candidate score reads `U` **only** through
    the induced precedence relation, so a small nudge frequently leaves the skill's whole
    score column bit-identical -- and a parity check made of such moves compares two
    unchanged arrays and passes while testing nothing. The gate therefore counts how many
    moves actually moved a column and refuses to return PASS on too few.
    """
    candidate = np.array(u, dtype=float, copy=True)
    row = int(rng.integers(candidate.shape[1]))
    candidate[skill, row] += scale * rng.standard_normal(candidate.shape[2])
    return candidate, row


def compare_arrays(a, b) -> dict:
    """Exact-first comparison. `max_abs` is reported for context, never as a pass rule."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    mask_identical = bool(np.array_equal(finite_a, finite_b))
    both = finite_a & finite_b
    if both.any():
        diff = np.abs(a[both] - b[both])
        max_abs = float(diff.max())
        n_differing = int((diff > 0).sum())
    else:
        max_abs, n_differing = 0.0, 0
    bitwise = bool(np.array_equal(np.nan_to_num(a, nan=0.0, posinf=1e308, neginf=-1e308),
                                  np.nan_to_num(b, nan=0.0, posinf=1e308, neginf=-1e308)))
    return {"finite_mask_identical": mask_identical, "bitwise_identical": bitwise,
            "max_abs_difference": max_abs, "entries_differing": n_differing,
            "n_finite": int(both.sum())}


def gate_one_rung(k: int, library, replicate: int, moves: int, sweeps: int,
                  scale: float, epsilon: float, seed: int) -> dict:
    corpus = generate_ladder_corpus(library, k, replicate)
    u_true, role_maps = library.prefix(k)
    model = Stage6EModel(traces=corpus.traces("train"), epsilon=epsilon, delta_b=0.15,
                         n_skills=k, n_roles=library.n_roles, min_width=3, max_width=12,
                         infer_pi_P=True, eta_initial=1.0, eta_transition=1.0)
    rng = np.random.default_rng(seed)
    beta, omega, l_rep, l_back = 1.0, 1.0, 0.1, 0.1

    log_pi = np.log(np.asarray(corpus.pi, dtype=float))
    log_p = log_transition_matrix(np.asarray(corpus.transition, dtype=float))

    exact = _tables(model, role_maps, epsilon)
    fast = _tables(model, role_maps, epsilon)
    exact.refresh(u_true, beta, omega, l_rep, l_back)
    fast.refresh(u_true, beta, omega, l_rep, l_back)
    base_exact = log_z_per_trace(exact, model, log_pi, log_p)
    base_fast = log_z_per_trace(fast, model, log_pi, log_p)

    checks: list = []

    # ---- 1 & 2: masks and entrywise scores, after a sequence of one-skill U moves
    u = np.array(u_true, dtype=float, copy=True)
    delta_rows, alpha_rows, trajectory = [], [], []
    rebuilt_counts = []
    order_changing = 0
    for move in range(moves):
        skill = int(rng.integers(k))
        candidate, row = _propose(rng, u, skill, scale)

        before = np.array(exact._table[skill], copy=True)
        exact_info = exact.refresh(candidate, beta, omega, l_rep, l_back)   # all skills
        fast_info = fast.refresh_changed(candidate, beta, omega, l_rep, l_back)
        if not np.array_equal(before, exact._table[skill]):
            order_changing += 1
        rebuilt_counts.append(len(fast_info["rebuilt_skills"]))
        if fast_info["rebuilt_skills"] != [skill]:
            checks.append({"check": "skill_local_invalidation", "move": move,
                           "passed": False,
                           "detail": f"fast rebuilt {fast_info['rebuilt_skills']}, "
                                     f"expected [{skill}]"})

        for n in range(len(model.traces)):
            comparison = compare_arrays(exact.tables[n], fast.tables[n])
            if not (comparison["finite_mask_identical"]
                    and comparison["bitwise_identical"]):
                checks.append({"check": "entrywise_scores", "move": move, "trace": n,
                               "passed": False, **comparison})

        # ---- 3: the one-skill U-update delta, through the sealed forward recursion
        cand_exact = log_z_per_trace(exact, model, log_pi, log_p)
        cand_fast = log_z_per_trace(fast, model, log_pi, log_p)
        d_exact = float((cand_exact - base_exact).sum())
        d_fast = float((cand_fast - base_fast).sum())
        delta_rows.append({"move": move, "skill": skill, "row": row,
                           "exact": d_exact, "fast": d_fast,
                           "abs_difference": abs(d_exact - d_fast),
                           "identical": d_exact == d_fast})

        # ---- 4: the MH acceptance ratio built on that delta
        log_alpha_exact = min(0.0, d_exact)
        log_alpha_fast = min(0.0, d_fast)
        alpha_rows.append({"move": move, "exact": log_alpha_exact,
                           "fast": log_alpha_fast,
                           "abs_difference": abs(log_alpha_exact - log_alpha_fast),
                           "identical": log_alpha_exact == log_alpha_fast})

        # ---- 5: the accept/reject decision, taken from a SHARED uniform
        uniform = float(rng.random())
        accept_exact = bool(math.log(uniform) < log_alpha_exact)
        accept_fast = bool(math.log(uniform) < log_alpha_fast)
        trajectory.append({"move": move, "exact": accept_exact, "fast": accept_fast,
                           "agree": accept_exact == accept_fast})
        if accept_exact:
            u = candidate
            base_exact, base_fast = cand_exact, cand_fast
        else:
            exact.refresh(u, beta, omega, l_rep, l_back)
            fast.refresh_changed(u, beta, omega, l_rep, l_back)

    checks.append({"check": "u_update_deltas", "passed":
                   all(r["identical"] for r in delta_rows), "n": len(delta_rows)})
    checks.append({"check": "mh_acceptance_ratios", "passed":
                   all(r["identical"] for r in alpha_rows), "n": len(alpha_rows)})
    checks.append({"check": "accept_reject_trajectory", "passed":
                   all(r["agree"] for r in trajectory), "n": len(trajectory),
                   "disagreements": [r["move"] for r in trajectory if not r["agree"]]})
    if not any(c["check"] == "entrywise_scores" for c in checks):
        checks.append({"check": "entrywise_scores", "passed": True,
                       "detail": f"{moves} moves x {len(model.traces)} traces, "
                                 f"bitwise identical"})
    if not any(c["check"] == "skill_local_invalidation" for c in checks):
        checks.append({"check": "skill_local_invalidation", "passed": True,
                       "detail": f"fast rebuilt exactly one skill on every one of "
                                 f"{moves} moves"})

    min_effective = max(3, moves // 5)
    checks.append({
        "check": "moves_actually_changed_a_column",
        "passed": order_changing >= min_effective,
        "detail": (f"{order_changing}/{moves} proposals changed the moved skill's score "
                   f"column (need >= {min_effective}); the rest left `U`'s induced order "
                   f"unchanged and so compared unchanged arrays")})

    return {
        "K": k,
        "n_traces": len(model.traces),
        "moves": moves,
        "order_changing_moves": order_changing,
        "mean_skills_rebuilt_by_fast": float(np.mean(rebuilt_counts)),
        "skills_rebuilt_by_exact": k,
        "checks": checks,
        "passed": all(c.get("passed", False) for c in checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rungs", type=int, nargs="+", default=[3, 10, 20, 30])
    parser.add_argument("--moves", type=int, default=25)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--library-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=770_001)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "k_ladder" / "fast_exact_parity.json")
    args = parser.parse_args()

    library, _ = draw_master_library(args.library_seed)
    began = time.perf_counter()
    rungs = []
    print(f"{'K':>4} {'moves':>6} {'effective':>10} {'fast/move':>10} {'exact':>6} "
          f"{'verdict':>9}")
    print("-" * 60)
    for k in args.rungs:
        result = gate_one_rung(k, library, args.replicate, args.moves, args.sweeps,
                               args.scale, args.epsilon, args.seed + k)
        rungs.append(result)
        print(f"{k:>4} {result['moves']:>6} {result['order_changing_moves']:>10} "
              f"{result['mean_skills_rebuilt_by_fast']:>10.2f} "
              f"{result['skills_rebuilt_by_exact']:>6} "
              f"{'PASS' if result['passed'] else 'BLOCKED':>9}")
        for check in result["checks"]:
            flag = "ok  " if check.get("passed") else "FAIL"
            print(f"       {flag} {check['check']}"
                  + (f"  -- {check.get('detail', '')}" if check.get("detail") else ""))

    all_passed = all(r["passed"] for r in rungs)
    print()
    if all_passed:
        print("RESULT: PASS -- the fast refresh is bitwise identical to the exact one at "
              "every rung tested,\n        and no MH decision differs. It is an exact "
              "optimisation, not an approximation.")
    else:
        print("RESULT: BLOCKED -- the fast refresh is NOT exact. Do not use "
              "table_source='fast'\n        without an exact correction. Production is "
              "blocked on this.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema": "k-ladder-fast-exact-parity/1.0.0",
        "settings": vars(args) | {"out": str(args.out)},
        "rungs": rungs,
        "all_passed": all_passed,
        "verdict": "PASS" if all_passed else "BLOCKED",
        "seconds": time.perf_counter() - began,
        "note": ("Exactness is the pass rule: finite masks identical, scores bitwise "
                 "identical, U-update deltas identical, MH log-ratios identical, and "
                 "accept/reject trajectories identical under a shared uniform. "
                 "max_abs_difference is reported for context and is never the criterion."),
    }, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
