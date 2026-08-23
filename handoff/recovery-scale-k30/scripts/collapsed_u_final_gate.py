"""Final-validation primary gate — evaluate the prospectively frozen chain-balanced,
dependence-aware, full-window multivariate diagnostic on the finished run.

    PYTHONPATH=src python scripts/collapsed_u_final_gate.py

Written and committed to disk BEFORE the run it judges finished; every rule it applies
is the one frozen in collapsed_u_final_validation/preregistration.json. Reads only
finished artifacts; produces final_gate.json, final_verdict.json and report.md.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "mcmc_original" / "collapsed_u_final_validation"
B7B1 = ROOT / "results" / "mcmc_original" / "stage7b1_mixed_reference"

BALANCED_PER_CHAIN, BALANCED_STRIDE = 1_000, 48
FULLWIN_ROWS, FULLWIN_STRIDE = 2_000, 24
N_BOOT, BOOT_SEED, Z_OVER = 150, 8_157_900, 2.33


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    cal = _load("depcal", ROOT / "scripts" / "collapsed_u_dependence_calibration.py")
    A, keep, centre, scale = cal.build_reference_rows()
    machine = cal.EnergyMachine(A, 4_000)

    def balanced_rows(npz_path):
        segs = [cal.chain_rows_of(npz_path, c, keep, centre, scale)
                [::BALANCED_STRIDE][:BALANCED_PER_CHAIN] for c in range(4)]
        return segs, np.concatenate(segs, axis=0)

    def balanced_boot_sd(segs, block, seed):
        """Within-chain circular moving-block bootstrap of the balanced sample."""
        rng = np.random.default_rng(seed)
        values = np.empty(N_BOOT)
        for r in range(N_BOOT):
            parts = []
            for seg in segs:
                n = len(seg)
                n_blocks = math.ceil(n / block)
                starts = rng.integers(0, n, size=n_blocks)
                idx = (starts[:, None] + np.arange(block)[None, :]
                       ).reshape(-1)[:n] % n
                parts.append(seg[idx])
            values[r] = machine.statistic(np.concatenate(parts, axis=0))
        return float(values.std(ddof=1))

    run_segs, run_bal = balanced_rows(OUT / "chains.npz")
    b7_segs, b7_bal = balanced_rows(B7B1 / "chains.npz")
    t_obs = machine.statistic(run_bal)
    t_null = machine.statistic(b7_bal)

    psi_iacts = [cal.iact(machine.contribution_series(seg)) for seg in run_segs]
    L = int(math.ceil(max(psi_iacts)))
    lengths = sorted({max(2, L // 2), max(2, L), 2 * L})

    primary = {"T_obs": t_obs, "T_7b1_null": t_null, "psi_iacts": psi_iacts,
               "L": L, "lengths": lengths, "by_length": {}}
    for length in lengths:
        se_obs = balanced_boot_sd(run_segs, length, BOOT_SEED + length)
        se_null = balanced_boot_sd(b7_segs, length, BOOT_SEED + 50 + length)
        z = (t_obs - t_null) / math.sqrt(se_obs ** 2 + se_null ** 2)
        primary["by_length"][str(length)] = {"se_obs": se_obs, "se_null": se_null,
                                             "z_bal": z, "pass": bool(z <= Z_OVER)}
    primary["pass_all_lengths"] = bool(all(v["pass"]
                                           for v in primary["by_length"].values()))

    # supporting per-chain full-window check, machinery identical to the calibration
    machine_fw = cal.EnergyMachine(A, FULLWIN_ROWS)
    b7_fw = [cal.chain_rows_of(B7B1 / "chains.npz", c, keep, centre, scale)
             [::FULLWIN_STRIDE][:FULLWIN_ROWS] for c in range(4)]
    mu0_fw = float(np.mean([machine_fw.statistic(b) for b in b7_fw]))
    supporting = {"mu0_full_window": mu0_fw, "chains": {}}
    over_by_length = {str(length): 0 for length in lengths}
    for c in range(4):
        b = cal.chain_rows_of(OUT / "chains.npz", c, keep, centre, scale)
        b = b[::FULLWIN_STRIDE][:FULLWIN_ROWS]
        t = machine_fw.statistic(b)
        entry = {"T": t, "z_by_length": {}}
        for length in lengths:
            boot = cal.block_bootstrap_sd(machine_fw, b, length, N_BOOT,
                                          BOOT_SEED + 100 + 10 * c + length)
            z = (t - mu0_fw) / max(1e-12, boot["sd"])
            entry["z_by_length"][str(length)] = z
            over_by_length[str(length)] += int(z > Z_OVER)
        supporting["chains"][str(c)] = entry
    supporting["over_by_length"] = over_by_length
    supporting["at_most_one_over_everywhere"] = bool(
        all(v <= 1 for v in over_by_length.values()))
    supporting["two_or_more_over_somewhere"] = bool(
        any(v >= 2 for v in over_by_length.values()))

    frozen = json.loads((OUT / "joint_comparison.json").read_text())
    frozen_gates = {k: g for k, g in frozen["gates"].items()
                    if k != "mixed_multivariate_reference_statistic"}
    frozen_pass = bool(all(g["pass"] for g in frozen_gates.values()))
    historical_mixed = frozen["gates"]["mixed_multivariate_reference_statistic"]

    if frozen_pass and primary["pass_all_lengths"] and \
            supporting["at_most_one_over_everywhere"]:
        verdict = "COLLAPSED-U KERNEL VALIDATED — MATCHED SYNTHETIC UNBLOCKED"
    elif (not frozen_pass) or all(not v["pass"]
                                  for v in primary["by_length"].values()):
        verdict = "COLLAPSED-U KERNEL NOT VALIDATED"
    else:
        verdict = "FINAL VALIDATION INCONCLUSIVE — REVIEW"

    payload = {"verdict": verdict,
               "seventeen_frozen_gates_all_pass": frozen_pass,
               "worst_rhat": frozen["worst_rhat"],
               "primary_balanced_gate": primary,
               "supporting_per_chain": supporting,
               "historical_unbalanced_mixed_gate_descriptive": {
                   "value": historical_mixed["value"],
                   "threshold": historical_mixed["threshold"],
                   "pass": historical_mixed["pass"],
                   "status": "historical/descriptive only, per preregistration"}}
    (OUT / "final_gate.json").write_text(json.dumps(payload, indent=2, default=float))

    lines = ["# Collapsed-U final validation — primary gate", "",
             f"**{verdict}**", "",
             f"17 frozen gates: {'ALL PASS' if frozen_pass else 'FAILURE'} "
             f"(worst R-hat {frozen['worst_rhat']:.5f}).",
             f"Historical unbalanced mixed gate (descriptive): "
             f"{historical_mixed['value']:.6f} vs {historical_mixed['threshold']:.6f} "
             f"({'pass' if historical_mixed['pass'] else 'fail'}).", "",
             f"Primary balanced gate: T_obs {t_obs:.6f} vs 7B1 null {t_null:.6f}; "
             f"z by length: " + ", ".join(
                 f"l={k}: {v['z_bal']:+.2f}" for k, v in primary['by_length'].items())
             + f" (threshold {Z_OVER}).",
             f"Supporting per-chain: over-threshold counts {over_by_length} "
             f"(mu0 {mu0_fw:.6f})."]
    (OUT / "final_verdict.md").write_text("\n".join(lines) + "\n")
    print(f"[final-gate] {verdict}")
    for k, v in primary["by_length"].items():
        print(f"[final-gate] balanced z (l={k}): {v['z_bal']:+.3f}")
    print(f"[final-gate] supporting over-counts: {over_by_length}")


if __name__ == "__main__":
    main()
