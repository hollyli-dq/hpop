"""Stage 00 — demonstrate the static BPOP likelihood on the known fork poset.

Run with:

    PYTHONPATH=src python scripts/stage00_check_bpop.py

The fork is

    0 > 2,  1 > 2,  0 incomparable to 1

so the two legal linear extensions are (0,1,2) and (1,0,2). With epsilon > 0 the
remaining four executions keep small positive probability. The total over all 3! = 6
executions must be exactly 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hpop.mcmc_original.latent_poset import incomparable, precedence_from_u  # noqa: E402
from hpop.mcmc_original.static_bpop import (  # noqa: E402
    all_permutation_probabilities,
    bpop_step_probabilities,
    frontier,
    remaining_successor_count,
    sample_bpop_sequence,
    successor_utility,
)

BETA = 1.5
EPSILON = 0.05
SEED = 20260808

U_FORK = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, -1.0],
    ]
)


def rule(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> int:
    print("Stage 00 — static BPOP frontier-softmax likelihood on the fork poset")
    print(f"beta = {BETA}   epsilon = {EPSILON}")

    rule("1. latent U")
    for i, row in enumerate(U_FORK):
        print(f"  u_{i} = {np.array2string(row, precision=2)}")

    rule("2. induced precedence matrix  (P[i,j] = 1 means i > j)")
    p = precedence_from_u(U_FORK)
    print("       " + " ".join(f"{j:>3}" for j in range(3)))
    for i in range(3):
        print(f"    {i}  " + " ".join("  1" if p[i, j] else "  ." for j in range(3)))
    print(f"\n  0 > 2 : {bool(p[0, 2])}")
    print(f"  1 > 2 : {bool(p[1, 2])}")
    print(f"  0 || 1: {incomparable(p, 0, 1)}")

    remaining = (0, 1, 2)

    rule("3. initial frontier")
    print(f"  remaining = {remaining}")
    print(f"  frontier  = {frontier(remaining, p)}")

    rule("4. successor counts and utilities at the initial step")
    for x in remaining:
        s = remaining_successor_count(x, remaining, p)
        q = successor_utility(x, remaining, p)
        print(f"  role {x}:  S(x) = {s}   Q(x) = log(1+{s}) = {q:.6f}")

    rule(f"5. one-step probabilities  (beta = {BETA}, epsilon = {EPSILON})")
    probs = bpop_step_probabilities(remaining, U_FORK, BETA, EPSILON)
    for x in range(3):
        print(f"  p(y_1 = {x}) = {probs[x]:.10f}")
    print(f"  sum         = {probs.sum():.16f}")

    rule("6-8. all 3! = 6 complete executions")
    table = all_permutation_probabilities(U_FORK, BETA, EPSILON)
    legal = {(0, 1, 2), (1, 0, 2)}
    for perm, value in sorted(table.items()):
        tag = "linear extension" if perm in legal else "order violation"
        print(f"  p({perm}) = {value:.12f}   {tag}")
    total = sum(table.values())
    print(f"\n  total = {total:.16f}")
    print(f"  |total - 1| = {abs(total - 1.0):.3e}")

    rule("9. ten sampled executions")
    rng = np.random.default_rng(SEED)
    for i in range(10):
        print(f"  sample {i + 1:>2}: {sample_bpop_sequence(rng, U_FORK, BETA, EPSILON)}")

    ok = abs(total - 1.0) < 1e-12
    print()
    print(f"[{'PASS' if ok else 'FAIL'}] complete BPOP likelihood sums to 1")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
