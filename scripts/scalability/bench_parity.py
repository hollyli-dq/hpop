"""Parity gate: the optimized backend against the frozen reference, before any scaling.

    python scripts/scalability/bench_parity.py [--out results/.../parity_results.json]

The reference engine is `hpop.mcmc_original` and it is the oracle. Nothing here writes to
it, imports it for anything but computation, or relaxes a tolerance to get a pass.

Grid, as registered by the benchmark plan:

    J in {24, 48}   K in {3, 5}   A in {5, 10}   D_max in {6, 12}

sixteen points, each checked in both role-support regimes, on:

    alpha            max absolute error <= 1e-10
    log Z            max absolute error <= 1e-10
    -inf pattern     identical
    legal blocks     counted from the tables, identical, and equal to the geometry
    backward draw    a legal complete cover with no forbidden self-transition
    numerics         no NaN anywhere, every mixed emission probability in (0, 1]
    P                zero diagonal, allowed rows summing to one, after a full sweep
    full sweep       the optimized sweep against `full_latent_sweep_once`, same rng

If any point exceeds 1e-10 the gate fails and no scaling point may run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                     # noqa: E402
import bench_common as bc                                              # noqa: E402

TOLERANCE = 1e-10
GRID_J = (24, 48)
GRID_K = (3, 5)
GRID_A = (5, 10)
GRID_D = (6, 12)


def check_point(J: int, K: int, A: int, D: int, regime: str) -> dict:
    from hpop.mcmc_original.matched_full_latent import (FullLatentConfig,
                                                        FullLatentFixed,
                                                        FullLatentSampler,
                                                        full_latent_sweep_once)
    from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import FFBSBlockTables
    from hpop.mcmc_original.semi_markov_ffbs import backward_sample
    from hpop.mcmc_original.semi_markov_ffbs import forward as reference_forward
    from hpop.mcmc_original.transitions import log_transition_matrix
    from hpop.mcmc_optimized import (COUNTERS, FLAGS, HashCachedFFBSBlockTables,
                                     OptimizedFullLatentSampler)
    from hpop.mcmc_optimized import sweep_once as optimized_sweep_once
    from hpop.mcmc_optimized.forward import forward_batched_group

    FLAGS.reset()
    COUNTERS.reset()

    cfg = bc.BenchConfig(axis="parity", label=f"parity_J{J}_K{K}_A{A}_D{D}_{regime}",
                         N=6, J=J, K=K, A=A, D_max=D, regime=regime)
    model = bc.build_model(cfg)
    state = bc.build_state(cfg, model)
    log_pi = np.log(state.pi)
    log_p = log_transition_matrix(state.transition)

    # --- tables: the frozen builder against the H-cached subclass -----------------------
    reference_tables = FFBSBlockTables(model=model, source="batched")
    reference_tables.refresh(state)
    ref_tables = [np.array(t, copy=True) for t in reference_tables.tables_for(state)]

    optimized_tables = HashCachedFFBSBlockTables(model=model, source="batched")
    optimized_tables.refresh(state)
    optimized_tables.refresh(state)                     # exercise the H-keyed short circuit
    opt_tables = list(optimized_tables.tables_for(state))

    tables_bitwise = all(np.array_equal(a, b) and a.dtype == b.dtype
                         for a, b in zip(ref_tables, opt_tables))
    ref_legal = int(sum(int(np.isfinite(t).sum()) for t in ref_tables))
    opt_legal = int(sum(int(np.isfinite(t).sum()) for t in opt_tables))
    geometry = bc.legal_block_count(cfg, model)
    legal_counts_identical = (ref_legal == opt_legal
                              == geometry["legal_blocks_times_skills"])

    # --- forward: the frozen recursion against the batched one --------------------------
    reference_charts = [reference_forward(t, log_pi, log_p, model.delta_b,
                                          model.max_width, model.min_width)
                        for t in ref_tables]
    classes: dict = {}
    for n, table in enumerate(opt_tables):
        classes.setdefault(np.asarray(table).shape[0], []).append(n)
    optimized_charts = [None] * len(opt_tables)
    for _length, members in sorted(classes.items()):
        for n, chart in zip(members, forward_batched_group(
                [opt_tables[n] for n in members], log_pi, log_p, model.delta_b,
                model.max_width, model.min_width)):
            optimized_charts[n] = chart

    worst_alpha, worst_z, pattern_ok, nan_free = 0.0, 0.0, True, True
    for reference, optimized in zip(reference_charts, optimized_charts):
        a_ref = np.asarray(reference.alpha, dtype=float)
        a_opt = np.asarray(optimized.alpha, dtype=float)
        finite = np.isfinite(a_ref)
        if not np.array_equal(finite, np.isfinite(a_opt)):
            pattern_ok = False
        if np.isnan(a_ref).any() or np.isnan(a_opt).any():
            nan_free = False
        if finite.any():
            worst_alpha = max(worst_alpha,
                              float(np.max(np.abs(a_ref[finite] - a_opt[finite]))))
        worst_z = max(worst_z, abs(float(reference.log_normalizer)
                                   - float(optimized.log_normalizer)))

    # --- backward draw: the frozen sampler on the optimized charts ----------------------
    rng = np.random.default_rng(4242)
    draws = [backward_sample(chart, rng) for chart in optimized_charts]
    draw_ok = True
    for trace, blocks in zip(model.traces, draws):
        cursor, previous = 0, None
        for a, b, k in blocks:
            if a != cursor or not model.min_width <= b - a <= model.max_width:
                draw_ok = False
            if not 0 <= int(k) < model.n_skills:
                draw_ok = False
            if previous is not None and int(k) == previous:
                draw_ok = False
            cursor, previous = b, int(k)
        if cursor != len(trace):
            draw_ok = False

    # --- emission probabilities stay probabilities -------------------------------------
    probabilities_ok = True
    for table in opt_tables:
        finite = table[np.isfinite(table)]
        if finite.size and float(finite.max()) > 1e-9:
            probabilities_ok = False        # a log probability must not exceed zero

    # --- a full sweep, both engines, identically seeded ---------------------------------
    sweep_rows = {}
    for arm in ("FULL-COND", "FULL-MARG"):
        fixed = FullLatentFixed()
        config = FullLatentConfig(arm=arm, structural_cadence=1, structural_scale=0.5,
                                  table_source="batched")
        reference_sampler = FullLatentSampler(model=model, fixed=fixed, config=config)
        optimized_sampler = OptimizedFullLatentSampler(model=model, fixed=fixed,
                                                       config=config)
        base = state.copy()
        base.iteration = 0
        reference_state, _ = full_latent_sweep_once(
            base.copy(), reference_sampler, np.random.default_rng(99))
        optimized_state, _ = optimized_sweep_once(
            base.copy(), optimized_sampler, np.random.default_rng(99))
        keys_match = all(
            tuple((s.start, s.end, s.skill) for s in a.segments)
            == tuple((s.start, s.end, s.skill) for s in b.segments)
            for a, b in zip(reference_state.segmentations,
                            optimized_state.segmentations))
        target_error = abs(float(reference_state.components["log_target"])
                           - float(optimized_state.components["log_target"]))
        p = np.asarray(optimized_state.transition, dtype=float)
        allowed = [[k for k in range(model.n_skills) if k != h]
                   for h in range(model.n_skills)]
        sweep_rows[arm] = {
            "segmentations_identical": bool(keys_match),
            "log_target_abs_error": target_error,
            "log_target_within_tolerance": bool(target_error <= TOLERANCE),
            "P_diagonal_exactly_zero": bool(np.array_equal(np.diag(p), np.zeros(len(p)))),
            "P_allowed_rows_sum_to_one": bool(all(
                abs(float(p[h, allowed[h]].sum()) - 1.0) <= 1e-12
                for h in range(model.n_skills))),
            "pi_sums_to_one": bool(abs(float(optimized_state.pi.sum()) - 1.0) <= 1e-12),
            "no_nan_in_components": bool(all(
                np.isfinite(float(v)) for v in optimized_state.components.values()
                if isinstance(v, (int, float)))),
        }

    passed = (worst_alpha <= TOLERANCE and worst_z <= TOLERANCE and pattern_ok
              and nan_free and tables_bitwise and legal_counts_identical and draw_ok
              and probabilities_ok
              and all(row["segmentations_identical"] and row["log_target_within_tolerance"]
                      and row["P_diagonal_exactly_zero"] and row["P_allowed_rows_sum_to_one"]
                      and row["pi_sums_to_one"] and row["no_nan_in_components"]
                      for row in sweep_rows.values()))

    return {
        "J": J, "K": K, "A": A, "D_max": D, "D_min": cfg.D_min, "N": cfg.N,
        "regime": regime,
        "max_abs_alpha_error": worst_alpha,
        "max_abs_logZ_error": worst_z,
        "inf_pattern_identical": bool(pattern_ok),
        "no_nan": bool(nan_free),
        "emission_tables_bitwise_identical": bool(tables_bitwise),
        "legal_blocks_reference": ref_legal,
        "legal_blocks_optimized": opt_legal,
        "legal_blocks_geometry": geometry["legal_blocks_times_skills"],
        "legal_block_counts_identical": bool(legal_counts_identical),
        "backward_draw_valid": bool(draw_ok),
        "log_probabilities_not_positive": bool(probabilities_ok),
        "role_graph": bc.role_graph_summary(state.u_by_skill),
        "sweep_parity": sweep_rows,
        "tolerance": TOLERANCE,
        "PASS": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(bc.RESULTS / "parity_results.json"))
    args = parser.parse_args()

    began = time.time()
    rows = []
    for J, K, A, D, regime in product(GRID_J, GRID_K, GRID_A, GRID_D,
                                      (bc.FULL_SUPPORT, bc.SPARSE_SUPPORT)):
        row = check_point(J, K, A, D, regime)
        rows.append(row)
        print(f"J={J:3d} K={K} A={A:2d} D={D:2d} {regime:6s} "
              f"alpha={row['max_abs_alpha_error']:.3e} "
              f"logZ={row['max_abs_logZ_error']:.3e} "
              f"blocks={row['legal_blocks_optimized']:>7d} "
              f"{'PASS' if row['PASS'] else 'FAIL'}", flush=True)

    worst_alpha = max(r["max_abs_alpha_error"] for r in rows)
    worst_z = max(r["max_abs_logZ_error"] for r in rows)
    report = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": time.time() - began,
        "tolerance": TOLERANCE,
        "grid": {"J": list(GRID_J), "K": list(GRID_K), "A": list(GRID_A),
                 "D_max": list(GRID_D),
                 "regimes": [bc.FULL_SUPPORT, bc.SPARSE_SUPPORT]},
        "points": rows,
        "n_points": len(rows),
        "n_passed": sum(1 for r in rows if r["PASS"]),
        "worst_max_abs_alpha_error": worst_alpha,
        "worst_max_abs_logZ_error": worst_z,
        "software": bc.software_manifest(),
        "ALL_PASS": all(r["PASS"] for r in rows),
    }
    bc.atomic_write(Path(args.out), json.dumps(report, indent=2, sort_keys=True,
                                               default=float))
    print(f"\nworst alpha {worst_alpha:.3e}   worst logZ {worst_z:.3e}   "
          f"tolerance {TOLERANCE:.0e}")
    print(f"ALL_PASS = {report['ALL_PASS']}  ({report['n_passed']}/{len(rows)})")
    sys.exit(0 if report["ALL_PASS"] else 1)


if __name__ == "__main__":
    main()
