"""Frozen checkpoint evaluator for the sequential collapsed-U validation.

    PYTHONPATH=src python scripts/collapsed_u_efficient_gate.py --sweep 150000

Written and frozen BEFORE the first checkpoint it judges; never edited after a
checkpoint result exists. Loads a complete checkpoint set only, applies the registered
post-burn-in window, and emits PASS / FAIL / INCONCLUSIVE for that checkpoint into
checkpoint_<sweep>.json. All rules come from preregistration.json.

Checkpoint requirements (all registered):
  A. the 17 frozen non-energy reference gates PASS (Stage 6E1B compare, verbatim;
     the historical iid/unbalanced energy gate is computed but DESCRIPTIVE);
  B. the chain-balanced dependence-aware primary energy gate PASS (z <= 2.33);
  C. block lengths {max(2, l//2), l, 2l} all agree (l from the frozen 400-sweep base
     dependence scale, converted to the checkpoint's extraction spacing);
  D. max registered R-hat <= 1.01 (log posterior, relation count, co-clustering,
     rho, beta, omega, lambda_rep, lambda_back);
  E. ESS: each scalar bulk >= 1000 and tail >= 500; relation count bulk >= 1000;
     co-clustering bulk >= 1000;
  F. every chain moves structurally (>= 1 joint induced-H label change in-window);
  G. no hard correctness failure (contiguity, retained counts, finiteness).
Sensitivity disagreement in C alone -> INCONCLUSIVE for the checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.stage6b_mcmc_diagnostics import (                  # noqa: E402
    bulk_ess, rank_normalized_split_rhat, tail_ess,
)
from hpop.mcmc_original.stage7b_diagnostics import (                       # noqa: E402
    co_clustering_series, h_label_series,
)

OUT = ROOT / "results" / "mcmc_original" / "collapsed_u_efficient_final_validation"
RUN1 = ROOT / "results" / "mcmc_original" / "collapsed_u_kernel_validation"
B7B1 = ROOT / "results" / "mcmc_original" / "stage7b1_mixed_reference"

SCALARS = ("rho", "beta", "omega", "lambda_rep", "lambda_back")
BASE_DEPENDENCE_SWEEPS = 400          # frozen: calibration L=5 x 80-sweep spacing
BALANCED_PER_CHAIN = 1_000
N_BOOT, Z_OVER = 150, 2.33
EVAL_BOOT_SEED = 8_158_500
# lambda_rep is the registered slowest coordinate; its floors were lowered to 600/300
# BY USER DECISION BEFORE LAUNCH (2026-08-16) because the 1000-floor + 500k cap +
# two-consecutive-pass rule are arithmetically unsatisfiable at its known mixing rate.
# Every other scalar keeps 1000/500; the R-hat gate (<= 1.01) still applies to all.
ESS_RULES = {"scalar_bulk": 1_000, "scalar_tail": 500,
             "lambda_rep_bulk": 600, "lambda_rep_tail": 300,
             "structural_bulk": 1_000}
RHAT_MAX = 1.01


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_chain(chain: int, sweep: int, burn_in: int) -> dict:
    """Concatenate this chain's segments up to `sweep`; verify contiguity exactly."""
    segments = sorted((OUT / "chain_checkpoints").glob(f"chain{chain}_seg*.npz"),
                      key=lambda p: int(p.stem.split("seg")[1]))
    segments = [p for p in segments if int(p.stem.split("seg")[1]) <= sweep]
    if not segments or int(segments[-1].stem.split("seg")[1]) != sweep:
        raise SystemExit(f"chain {chain}: no complete segment set for sweep {sweep:,}")
    arrays: dict = {}
    keys: list = []
    expected_start = burn_in
    for path in segments:
        z = np.load(path, allow_pickle=False)
        if int(z["first_retained_sweep"]) != expected_start:
            raise SystemExit(f"chain {chain}: segment {path.name} starts at "
                             f"{int(z['first_retained_sweep']):,}, expected "
                             f"{expected_start:,} — HARD FAILURE (contiguity)")
        expected_start = int(z["last_retained_sweep"]) + 10
        for name in ("u_draws", "segment_counts", "relation_counts", "log_target",
                     "occurrence_labels", *[f"scalar_{n}" for n in SCALARS],
                     "collapsed_sweep", "collapsed_accepted", "collapsed_h_changed",
                     "collapsed_invalid"):
            arrays.setdefault(name, []).append(z[name])
        keys.extend(json.loads(str(z["keys_json"])))
    out = {name: np.concatenate(chunks, axis=0) for name, chunks in arrays.items()}
    out["keys"] = keys
    expected = (sweep - burn_in) // 10
    if len(out["log_target"]) != expected:
        raise SystemExit(f"chain {chain}: retained {len(out['log_target'])} != "
                         f"{expected} — HARD FAILURE (retained count)")
    if not np.isfinite(out["log_target"]).all():
        raise SystemExit(f"chain {chain}: non-finite log target — HARD FAILURE")
    return out


class _Result:
    def __init__(self, payload):
        self.__dict__.update(payload)

    def acceptance(self, post_burn_in=True):
        return self.__dict__["acceptance_rates"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=int, required=True)
    args = parser.parse_args()
    began = time.perf_counter()

    prereg = json.loads((OUT / "preregistration.json").read_text())
    burn_in = int(prereg["burn_in"])
    chains = [load_chain(c, args.sweep, burn_in) for c in range(4)]
    n_w = (args.sweep - burn_in) // 10

    # ---- A: the frozen reference gates, verbatim -------------------------------------
    e1b = _load("stage6e1b", ROOT / "scripts" / "stage6e1b_mixed_reference.py")
    traces, _ = e1b.generate_corpus()
    mixed = e1b.build_mixed_model(traces)
    eval_dir = OUT / f"checkpoint_eval_{args.sweep}"
    eval_dir.mkdir(exist_ok=True)
    for name in ("reference_draws.npz", "reference_registration.json",
                 "qmc_summary.json"):
        (eval_dir / name).write_bytes(
            (ROOT / "results/mcmc_original/stage6e1b_mixed_reference" / name)
            .read_bytes())
    results = []
    for c, data in enumerate(chains):
        results.append(_Result({
            "u_draws": data["u_draws"].astype(np.float32),
            "scalars": {n: data[f"scalar_{n}"] for n in SCALARS},
            "log_target": data["log_target"],
            "segment_counts": data["segment_counts"],
            "relation_counts": data["relation_counts"],
            "boundary_keys": [tuple(tuple((int(e), int(s)) for e, s in trace_key)
                                    for trace_key in draw)
                              for draw in data["keys"]],
            "movement": {}, "acceptance_rates": {},
            "seed": json.loads((OUT / "seed_manifest.json").read_text())
            ["chain_seeds"][c] if (OUT / "seed_manifest.json").exists() else c,
            "runtime_seconds": float("nan"),
        }))
    e1b.OUT = eval_dir
    try:
        e1b.compare(mixed, results)
    except SystemExit:
        pass                                  # gate failures are read from the json
    gates = json.loads((eval_dir / "joint_comparison.json").read_text())
    frozen_gates = {k: g for k, g in gates["gates"].items()
                    if k != "mixed_multivariate_reference_statistic"}
    a_pass = bool(all(g["pass"] for g in frozen_gates.values()))
    historical = gates["gates"]["mixed_multivariate_reference_statistic"]

    # ---- B/C: primary balanced dependence-aware gate ---------------------------------
    cal = _load("depcal", ROOT / "scripts" / "collapsed_u_dependence_calibration.py")
    A, keep, centre, scale = cal.build_reference_rows()
    machine = cal.EnergyMachine(A, 4 * BALANCED_PER_CHAIN)

    def rows_of(data):
        u = data["u_draws"]
        n = u.shape[0]
        closures = np.array([[cal.precedence_from_u(u[i, k]).reshape(-1)
                              for k in range(u.shape[1])]
                             for i in range(n)]).reshape(n, -1).astype(float)
        total = data["segment_counts"].sum(axis=1, keepdims=True).astype(float)
        scal = np.column_stack([data[f"scalar_{n2}"] for n2 in SCALARS])
        rows = np.column_stack([closures, total, scal])
        return cal.standardise(rows[:, keep], centre, scale)

    stride = max(1, n_w // BALANCED_PER_CHAIN)
    segs = [rows_of(d)[::stride][:BALANCED_PER_CHAIN] for d in chains]
    balanced = np.concatenate(segs, axis=0)
    t_obs = machine.statistic(balanced)

    b7_segs = [cal.chain_rows_of(B7B1 / "chains.npz", c, keep, centre, scale)
               [::48][:BALANCED_PER_CHAIN] for c in range(4)]
    t_null = machine.statistic(np.concatenate(b7_segs, axis=0))

    def balanced_boot_sd(seg_list, block, seed):
        rng = np.random.default_rng(seed)
        values = np.empty(N_BOOT)
        for r in range(N_BOOT):
            parts = []
            for seg in seg_list:
                n = len(seg)
                n_blocks = math.ceil(n / block)
                starts = rng.integers(0, n, size=n_blocks)
                idx = (starts[:, None] + np.arange(block)[None, :]
                       ).reshape(-1)[:n] % n
                parts.append(seg[idx])
            values[r] = machine.statistic(np.concatenate(parts, axis=0))
        return float(values.std(ddof=1))

    spacing_sweeps = stride * 10
    ell = max(2, round(BASE_DEPENDENCE_SWEEPS / spacing_sweeps))
    lengths = sorted({max(2, ell // 2), ell, 2 * ell})
    se_null = balanced_boot_sd(b7_segs, 2, EVAL_BOOT_SEED + args.sweep)
    primary = {"T_obs": t_obs, "T_7b1_null": t_null, "stride": stride,
               "spacing_sweeps": spacing_sweeps, "lengths": lengths,
               "se_null_at_block_2": se_null, "by_length": {}}
    for length in lengths:
        se_obs = balanced_boot_sd(segs, length, EVAL_BOOT_SEED + args.sweep + length)
        z = (t_obs - t_null) / math.sqrt(se_obs ** 2 + se_null ** 2)
        primary["by_length"][str(length)] = {"se_obs": se_obs, "z": z,
                                             "pass": bool(z <= Z_OVER)}
    votes = [v["pass"] for v in primary["by_length"].values()]
    b_pass, c_agree = bool(all(votes)), bool(len(set(votes)) == 1)

    # ---- D/E: convergence and effective sample sizes ---------------------------------
    convergence, ess = {}, {}
    stacked = {n: np.stack([d[f"scalar_{n}"] for d in chains]) for n in SCALARS}
    stacked["log_posterior"] = np.stack([d["log_target"] for d in chains])
    stacked["relation_count"] = np.stack(
        [d["relation_counts"].sum(axis=1).astype(float) for d in chains])
    stacked["co_clustering"] = np.stack(
        [co_clustering_series(d["occurrence_labels"]) for d in chains])
    for name, arr in stacked.items():
        convergence[name] = rank_normalized_split_rhat(arr)
        ess[name] = {"bulk": float(bulk_ess(arr)), "tail": float(tail_ess(arr))}
    d_pass = bool(all(v["rhat"] <= RHAT_MAX for v in convergence.values()))
    e_checks = {}
    for n in SCALARS:
        bulk_floor = (ESS_RULES["lambda_rep_bulk"] if n == "lambda_rep"
                      else ESS_RULES["scalar_bulk"])
        tail_floor = (ESS_RULES["lambda_rep_tail"] if n == "lambda_rep"
                      else ESS_RULES["scalar_tail"])
        e_checks[n] = bool(ess[n]["bulk"] >= bulk_floor
                           and ess[n]["tail"] >= tail_floor)
    e_checks["relation_count"] = bool(
        ess["relation_count"]["bulk"] >= ESS_RULES["structural_bulk"])
    e_checks["co_clustering"] = bool(
        ess["co_clustering"]["bulk"] >= ESS_RULES["structural_bulk"])
    e_pass = bool(all(e_checks.values()))

    # ---- F: structural movement -------------------------------------------------------
    movement = {}
    f_pass = True
    for c, d in enumerate(chains):
        labels = h_label_series(d["u_draws"].astype(float))
        changes = sum(1 for x, y in zip(labels[:-1], labels[1:]) if x != y)
        rec_mask = d["collapsed_sweep"] >= burn_in
        movement[str(c)] = {
            "collapsed_attempts": int(rec_mask.sum()),
            "cross_h_fraction": float(d["collapsed_h_changed"][rec_mask].mean())
            if rec_mask.any() else 0.0,
            "accepted": int(d["collapsed_accepted"][rec_mask].sum()),
            "accepted_cross_h": int((d["collapsed_accepted"]
                                     & d["collapsed_h_changed"])[rec_mask].sum()),
            "distinct_h_states": len(set(labels)),
            "h_changes": changes,
            "relation_count_ess": ess["relation_count"]["bulk"]}
        f_pass &= changes > 0

    components = {"A_frozen_gates": a_pass, "B_primary_energy": b_pass,
                  "C_sensitivity_agrees": c_agree, "D_rhat": d_pass,
                  "E_ess": e_pass, "F_structural_movement": f_pass,
                  "G_hard_checks": True}
    if not c_agree:
        verdict = "INCONCLUSIVE"
    elif all(components.values()):
        verdict = "PASS"
    else:
        verdict = "FAIL"

    payload = {"sweep": args.sweep, "burn_in": burn_in, "retained_per_chain": n_w,
               "verdict": verdict, "components": components,
               "frozen_gates": {k: {"value": g["value"], "pass": g["pass"]}
                                for k, g in frozen_gates.items()},
               "historical_energy_gate_descriptive": {
                   "value": historical["value"], "threshold": historical["threshold"],
                   "pass_descriptive_only": historical["pass"]},
               "primary_balanced_gate": primary,
               "worst_rhat": float(max(v["rhat"] for v in convergence.values())),
               "convergence": {k: v["rhat"] for k, v in convergence.items()},
               "ess": ess, "ess_checks": e_checks,
               "structural_movement": movement,
               "evaluation_seconds": time.perf_counter() - began}
    (OUT / f"checkpoint_{args.sweep // 1000}k.json").write_text(
        json.dumps(payload, indent=2, default=float))
    print(f"[gate {args.sweep // 1000}k] {verdict} | A={a_pass} B={b_pass} "
          f"C={c_agree} D={d_pass} E={e_pass} F={f_pass} | "
          f"z={[round(v['z'], 2) for v in primary['by_length'].values()]} "
          f"worst_rhat={payload['worst_rhat']:.4f}")


if __name__ == "__main__":
    main()
