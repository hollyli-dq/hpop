"""Step 7A — how the FFBS cost splits between the chart and the draws.

    PYTHONPATH=src python scripts/stage7a_ffbs_benchmark.py [--lengths 8 24 48 96]

Correctness lives in `stage7a_ffbs_exact.py`; this script measures only cost, and it exists
because the cost of FFBS has two halves that behave completely differently:

* the **forward chart** is a function of the parameters, so at fixed parameters it is paid
  once no matter how many segmentations are drawn;
* a **backward draw** is `O(L x max_width x K)` and is paid every time.

A sweep of the joint sampler changes the parameters and therefore rebuilds the chart, so
the ratio between the two is what decides whether blocked sampling pays for itself inside
Step 7B. Reporting draws-per-second alone would hide exactly that.

This is a benchmark, not a gate. It is single-threaded on purpose and touches nothing in
the Stage 6E result tree.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.block_score_adapters import build_log_block_scores    # noqa: E402
from hpop.mcmc_original.recurrent_segmentation import RecurrentBlockScorer    # noqa: E402
from hpop.mcmc_original.semi_markov_ffbs import backward_sample, forward      # noqa: E402
from hpop.mcmc_original.stage6e_frozen import (                               # noqa: E402
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH,
)
from hpop.mcmc_original.transitions import allowed_next, log_transition_matrix  # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "stage7a_ffbs_exact"

K_SKILLS = 3
M_ROLES = 3
EPSILON = 0.02
SCALARS = {"beta": 1.5, "omega": 1.7346, "lambda_rep": 0.8, "lambda_back": 0.25}
U_BY_SKILL = np.array([
    [[2.0, 2.0], [1.0, 0.0], [0.0, 1.0]],
    [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0]],
    [[3.0, 3.0], [2.0, 2.0], [0.0, 0.0]],
], dtype=float)
SEED = 7_072_101
TRACES_PER_LENGTH = 3
DRAWS_PER_TRACE = 200


def path_prior():
    log_pi = np.log(np.full(K_SKILLS, 1.0 / K_SKILLS))
    transition = np.zeros((K_SKILLS, K_SKILLS))
    for h in range(K_SKILLS):
        for k in allowed_next(h, K_SKILLS):
            transition[h, k] = 1.0 / (K_SKILLS - 1)
    return log_pi, log_transition_matrix(transition)


def one_trace(length: int, rng) -> dict:
    trace = tuple(int(v) for v in rng.integers(M_ROLES, size=length))
    scorer = RecurrentBlockScorer(
        traces=(trace,), epsilon=EPSILON, u_by_skill=U_BY_SKILL, beta=SCALARS["beta"],
        omega=SCALARS["omega"], lambda_rep=SCALARS["lambda_rep"],
        lambda_back=SCALARS["lambda_back"], min_width=MIN_BLOCK_WIDTH,
        max_width=MAX_BLOCK_WIDTH)
    log_pi, log_p = path_prior()

    began = time.perf_counter()
    table = build_log_block_scores(scorer, 0, length, K_SKILLS, MIN_BLOCK_WIDTH,
                                   MAX_BLOCK_WIDTH)
    table_seconds = time.perf_counter() - began

    began = time.perf_counter()
    chart = forward(table, log_pi, log_p, DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH)
    chart_seconds = time.perf_counter() - began

    rng_draw = np.random.default_rng(SEED + length)
    began = time.perf_counter()
    lengths = [len(backward_sample(chart, rng_draw)) for _ in range(DRAWS_PER_TRACE)]
    draw_seconds = (time.perf_counter() - began) / DRAWS_PER_TRACE

    fixed = table_seconds + chart_seconds
    return {"J": length, "n_legal_blocks": int(np.isfinite(table).sum()),
            "block_table_seconds": table_seconds, "chart_seconds": chart_seconds,
            "fixed_cost_seconds": fixed,
            "one_backward_draw_seconds": draw_seconds,
            "draws_amortising_the_fixed_cost": fixed / draw_seconds,
            "mean_blocks_per_draw": float(np.mean(lengths)),
            "log_normalizer": chart.log_normalizer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8, 24, 48, 96])
    parser.add_argument("--traces", type=int, default=TRACES_PER_LENGTH)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for length in args.lengths:
        per_trace = [one_trace(length, rng) for _ in range(args.traces)]
        summary = {
            "J": length, "traces": per_trace,
            "median_block_table_seconds": float(np.median(
                [r["block_table_seconds"] for r in per_trace])),
            "median_chart_seconds": float(np.median(
                [r["chart_seconds"] for r in per_trace])),
            "median_one_backward_draw_seconds": float(np.median(
                [r["one_backward_draw_seconds"] for r in per_trace])),
            "median_draws_amortising_the_fixed_cost": float(np.median(
                [r["draws_amortising_the_fixed_cost"] for r in per_trace])),
            "median_blocks_per_draw": float(np.median(
                [r["mean_blocks_per_draw"] for r in per_trace])),
            "n_legal_blocks": per_trace[0]["n_legal_blocks"],
        }
        rows.append(summary)
        print(f"[7A-bench] J={length:>3}  blocks {summary['n_legal_blocks']:>5}  "
              f"table {summary['median_block_table_seconds']*1e3:7.1f} ms  "
              f"chart {summary['median_chart_seconds']*1e3:6.1f} ms  "
              f"draw {summary['median_one_backward_draw_seconds']*1e6:7.0f} us  "
              f"break-even {summary['median_draws_amortising_the_fixed_cost']:7.0f} draws")

    payload = {
        "what": "FFBS cost decomposition: fixed chart construction vs per-draw cost",
        "python": platform.python_version(), "numpy": np.__version__,
        "single_threaded": True,
        "model": {"K": K_SKILLS, "m_roles": M_ROLES, "epsilon": EPSILON,
                  "delta_B": DELTA_B, "min_width": MIN_BLOCK_WIDTH,
                  "max_width": MAX_BLOCK_WIDTH, "scalars": SCALARS},
        "seed": SEED, "traces_per_length": args.traces,
        "draws_per_trace": DRAWS_PER_TRACE,
        "rows": rows,
        "caveat": "synthetic traces drawn uniformly over roles, not the Stage 6E2 corpus; "
                  "these are cost measurements only and no gate depends on them",
    }
    (OUT / "benchmark.json").write_text(json.dumps(payload, indent=2))
    print(f"[7A-bench] wrote {OUT / 'benchmark.json'}")


if __name__ == "__main__":
    main()
