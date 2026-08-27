"""Dependence-aware calibration of the per-chain energy statistic — no new MCMC chains.

    PYTHONPATH=src python scripts/collapsed_u_dependence_calibration.py

The question: are the uniformly positive per-chain energy z-scores of the start[0]
probe still anomalous once serial dependence is accounted for?

Estimator audit (from source, recorded in estimator_audit.json): `energy_distance` is a
full all-pairs V-statistic; serial dependence enters through the Y-Y term (close-in-time
draws are close in space, so E|Y-Y'| is under-estimated and the statistic inflated for
ANY correct dependent chain), and `calibrate_energy_envelope` draws BOTH null sides iid,
so both the null mean and null sd understate a correct chain's sampling distribution.

Registered procedure (all rules frozen in registration.json before any dependence-aware
number is computed):

* samples: the probe's late/early/middle windows (n = 2,000, stride 8, offset 0) in the
  cross-run coordinates [closures, TOTAL segment count, scalars] (run1/rep2/7B1 store no
  per-draw per-trace counts, so the total-count variant — whose group-level immateriality
  was established in the C-run forensics — is used consistently for every sample);
* block lengths: L = ceil(max IACT of the per-draw contribution series psi_j over the
  four start[0] late windows), sensitivity {max(2, L//2), L, 2L}; the conclusion must be
  stable across all three;
* dependence-aware SE: circular moving-block bootstrap of the time-ordered window rows,
  150 replicates, recomputing the FULL statistic each time (no linearisation);
* null centre mu0: the mean late-window statistic of the four VALIDATED 7B1 chains at
  the identical extraction — a correct kernel with matched dependence (Control A);
* z_dep(chain) = (T_chain - mu0) / SE_boot(chain); "clearly over" = z_dep > 2.33
  (one-sided 99%);
* Control A (internal): leave-one-out z of each 7B1 chain; pass if >= 3/4 within +-2.33
  at every block length;
* Control B (finite-state): the tiny exactly-solved collapsed-U grid chain (stationarity
  2.2e-16); 4 test chains simulated from its exact transition kernel, mu0 from 8 further
  independent chains, iid reference from the exact joint; pass if >= 3/4 within +-2.33
  at every block length. If Control B fails the method is untrustworthy.

Registered verdicts:
  ARTIFACT SUPPORTED   : at ALL three lengths, <= 1 of 4 start[0] late z_dep > 2.33,
                         and both controls pass;
  REAL INTERACTION     : at ALL three lengths, >= 3 of 4 start[0] late z_dep > 2.33,
                         and both controls pass;
  INCONCLUSIVE         : anything else (instability across lengths, 2/4 over, or a
                         control failure).

No historical verdict is overwritten; no kernel change; no new production chains;
matched-synthetic stays paused.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.latent_poset import precedence_from_u              # noqa: E402
from hpop.mcmc_original.sampler_u import log_u_prior                       # noqa: E402
from hpop.mcmc_original.stage6b_joint_diagnostics import standardise       # noqa: E402
from hpop.mcmc_original.stage6e_exact import (                             # noqa: E402
    enumerate_states, exact_posterior, state_log_weights,
)
from hpop.mcmc_original.transitions import log_transition_matrix           # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "collapsed_u_dependence_calibration"
PROBE = ROOT / "results" / "mcmc_original" / "collapsed_u_start0_probe"
RUN1 = ROOT / "results" / "mcmc_original" / "collapsed_u_kernel_validation"
REP2 = ROOT / "results" / "mcmc_original" / "collapsed_u_kernel_validation_rep2"
B7B1 = ROOT / "results" / "mcmc_original" / "stage7b1_mixed_reference"

WINDOWS = {"early": (0, 16_000), "middle": (16_000, 32_000), "late": (32_000, 48_000)}
STRIDE, N_ROWS = 8, 2_000
N_BOOT = 150
BOOT_SEED = 8_156_000
Z_OVER = 2.33                 # one-sided 99%
SCALARS = ("rho", "beta", "omega", "lambda_rep", "lambda_back")

CONTROL_B_SEED = 8_156_500
CONTROL_B_STEPS = 2_500       # per chain, after burn-in 500
CONTROL_B_BURN = 500
CONTROL_B_REF = 8_000
CONTROL_B_TEST_CHAINS = 4
CONTROL_B_NULL_CHAINS = 8


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- rows and statistic
def build_reference_rows():
    ref = np.load(RUN1 / "reference_draws.npz", allow_pickle=False)
    e1b = _load("stage6e1b", ROOT / "scripts" / "stage6e1b_mixed_reference.py")
    traces, _ = e1b.generate_corpus()
    mixed = e1b.build_mixed_model(traces)
    closures = ref["retained_closures"].reshape(
        ref["retained_closures"].shape[0], -1).astype(float)
    total = np.array([[len(mixed.states[j]) for j in row]
                      for row in ref["retained_sampled"]], dtype=float).sum(
        axis=1, keepdims=True)
    scal = np.column_stack([ref[f"retained_{n}"] for n in SCALARS])
    rows = np.column_stack([closures, total, scal])
    spread = rows.std(axis=0, ddof=1)
    keep = spread > 1e-12
    kept = rows[:, keep]
    centre, scale = kept.mean(0), kept.std(0, ddof=1)
    return standardise(kept, centre, scale), keep, centre, scale


def chain_rows_of(npz_path: Path, chain: int, keep, centre, scale) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=False)
    u = z["u_draws"][chain]
    n = u.shape[0]
    closures = np.array([[precedence_from_u(u[i, k]).reshape(-1)
                          for k in range(u.shape[1])]
                         for i in range(n)]).reshape(n, -1).astype(float)
    if "n_segments" in z.files:
        total = z["n_segments"][chain].reshape(-1, 1).astype(float)
    else:
        total = z["segment_counts"][chain].sum(axis=1, keepdims=True).astype(float)
    scal = np.column_stack([z[f"scalar_{n2}"][chain] for n2 in SCALARS])
    rows = np.column_stack([closures, total, scal])
    return standardise(rows[:, keep], centre, scale)


class EnergyMachine:
    """The frozen all-pairs V-statistic against the fixed reference side, with the
    A-side terms precomputed once (they never change across bootstrap replicates)."""

    def __init__(self, A: np.ndarray, n_rows: int):
        n_x = min(len(A) // 2, n_rows)
        self.a = A[:n_x * 2:2]
        self.e_aa = float(cdist(self.a, self.a).mean())

    def statistic(self, b: np.ndarray) -> float:
        return float(2.0 * cdist(self.a, b).mean() - self.e_aa
                     - cdist(b, b).mean())

    def contribution_series(self, b: np.ndarray) -> np.ndarray:
        """psi_j = 2 mean_i|a_i - b_j| - 2 mean_j'|b_j - b_j'| — the V-statistic's
        per-draw projection, used ONLY to estimate the IACT for block lengths."""
        cross = cdist(self.a, b).mean(axis=0)
        within = cdist(b, b).mean(axis=0)
        return 2.0 * cross - 2.0 * within


def iact(series: np.ndarray, max_lag: int | None = None) -> float:
    """Initial-positive-sequence integrated autocorrelation time (in sample steps)."""
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    n = len(x)
    if x.std() == 0:
        return 1.0
    max_lag = max_lag or n // 4
    acf = np.correlate(x, x, mode="full")[n - 1:n - 1 + max_lag + 1]
    acf = acf / acf[0]
    total = 1.0
    for lag in range(1, max_lag):
        if acf[lag] <= 0:
            break
        total += 2.0 * acf[lag]
    return float(max(1.0, total))


def block_bootstrap_sd(machine: EnergyMachine, b: np.ndarray, block: int,
                       n_boot: int, seed: int) -> dict:
    """Circular moving-block bootstrap of the time-ordered rows; full recompute."""
    rng = np.random.default_rng(seed)
    n = len(b)
    n_blocks = math.ceil(n / block)
    values = np.empty(n_boot)
    for r in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).reshape(-1)[:n] % n
        values[r] = machine.statistic(b[idx])
    return {"sd": float(values.std(ddof=1)), "mean": float(values.mean()),
            "n_boot": n_boot, "block": int(block)}


def window_rows(rows: np.ndarray, window: str) -> np.ndarray:
    lo, hi = WINDOWS[window]
    return rows[lo:hi:STRIDE][:N_ROWS]


# ------------------------------------------------------------------ control B (exact)
def control_b(block_lengths_rule: str) -> dict:
    """The finite-state collapsed-U grid chain: exact target, exact kernel matrix."""
    ordering = _load("ordering", ROOT / "tests" / "mcmc_original"
                     / "test_collapsed_u_ordering.py")
    model = ordering.grid_model()
    grid, paths, prior, joint, conditional, ell = ordering.build_grid(model)
    n_g, n_p = len(grid), len(paths)
    m = ordering.collapsed_mh_matrix(prior, ell)
    kernel = np.zeros((n_g * n_p, n_g * n_p))
    for g in range(n_g):
        for p in range(n_p):
            for h in range(n_g):
                kernel[g * n_p + p, h * n_p:(h + 1) * n_p] = m[g, h] * conditional[h]
    flat = joint.reshape(-1)
    features = np.zeros((n_g * n_p, n_g + n_p))
    for g in range(n_g):
        for p in range(n_p):
            features[g * n_p + p, g] = 1.0
            features[g * n_p + p, n_g + p] = 1.0

    rng = np.random.default_rng(CONTROL_B_SEED)
    ref_rows = features[rng.choice(len(flat), size=CONTROL_B_REF, p=flat)]
    spread = ref_rows.std(0, ddof=1)
    keep = spread > 1e-12
    centre, scale = ref_rows[:, keep].mean(0), ref_rows[:, keep].std(0, ddof=1)
    A = standardise(ref_rows[:, keep], centre, scale)
    machine = EnergyMachine(A, CONTROL_B_STEPS)

    def simulate(seed):
        r = np.random.default_rng(seed)
        state = int(r.choice(len(flat), p=flat))
        out = np.empty(CONTROL_B_STEPS, dtype=int)
        for t in range(CONTROL_B_BURN + CONTROL_B_STEPS):
            state = int(r.choice(len(flat), p=kernel[state]))
            if t >= CONTROL_B_BURN:
                out[t - CONTROL_B_BURN] = state
        return standardise(features[out][:, keep], centre, scale)

    null_ts = [machine.statistic(simulate(CONTROL_B_SEED + 100 + i))
               for i in range(CONTROL_B_NULL_CHAINS)]
    mu0 = float(np.mean(null_ts))

    tests = []
    psi_iacts = []
    test_rows = []
    for i in range(CONTROL_B_TEST_CHAINS):
        b = simulate(CONTROL_B_SEED + 500 + i)
        test_rows.append(b)
        psi_iacts.append(iact(machine.contribution_series(b)))
    L_b = int(math.ceil(max(psi_iacts)))
    lengths = sorted({max(2, L_b // 2), max(2, L_b), max(2, 2 * L_b)})
    for i, b in enumerate(test_rows):
        t_obs = machine.statistic(b)
        entry = {"chain": i, "T": t_obs, "z_by_length": {}}
        for length in lengths:
            boot = block_bootstrap_sd(machine, b, length, N_BOOT,
                                      CONTROL_B_SEED + 900 + i * 10 + length)
            entry["z_by_length"][str(length)] = (t_obs - mu0) / max(1e-12, boot["sd"])
        tests.append(entry)
    passes = {str(length): sum(1 for t in tests
                               if abs(t["z_by_length"][str(length)]) <= Z_OVER)
              for length in lengths}
    return {"L": L_b, "lengths": lengths, "mu0": mu0, "null_ts": null_ts,
            "psi_iacts": psi_iacts, "tests": tests,
            "within_2p33_by_length": passes,
            "pass": bool(all(v >= 3 for v in passes.values())),
            "note": "target and kernel exact (stationarity 2.2e-16); a failure here "
                    "would indict the calibration method, not the chain",
            "block_length_rule": block_lengths_rule}


# --------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    estimator_audit = {
        "function": "stage6b_joint_diagnostics.energy_distance",
        "form": "full all-pairs V-statistic: 2 E|X-Y| - E|X-X'| - E|Y-Y'|; chunked "
                "exact summation over every pair (self-pairs included at distance 0, "
                "an O(1/n) term identical on both sides of the calibration)",
        "pairing": "no subsampling, no adjacent/fixed-lag pairing, no linear-time "
                   "approximation; row ORDER does not enter the statistic — order "
                   "matters only through upstream sample selection (the pooled "
                   "first-4000 prefix in the frozen gate; strides here)",
        "dependence_entry_point": "the Y-Y term: temporally close draws are spatially "
                                  "close, E|Y-Y'| is under-estimated, so the statistic "
                                  "is inflated for any CORRECT dependent chain; the "
                                  "Y-side of the cross term also carries the "
                                  "dependence into the sampling variance",
        "frozen_envelope": "calibrate_energy_envelope permutes the reference pool and "
                           "draws BOTH sides iid — the null carries no serial "
                           "dependence, understating both mean and sd for MCMC input",
    }
    (OUT / "estimator_audit.json").write_text(json.dumps(estimator_audit, indent=2))

    registration = {
        "purpose": "dependence-aware recalibration of the per-chain energy z, "
                   "existing draws only; no kernel change; no new production chains; "
                   "historical verdicts unchanged; matched-synthetic stays paused",
        "samples": {"windows": {k: list(v) for k, v in WINDOWS.items()},
                    "stride": STRIDE, "n_rows": N_ROWS,
                    "coordinates": "[closures, TOTAL segment count, scalars] for "
                                   "cross-run comparability (7B1/run1/rep2 store no "
                                   "per-draw per-trace counts)"},
        "block_length_rule": "L = ceil(max IACT of the contribution series psi_j over "
                             "the four start[0] LATE windows); sensitivity "
                             "{max(2, L//2), L, 2L}; conclusions must be stable "
                             "across all three",
        "bootstrap": {"kind": "circular moving-block, full statistic recompute",
                      "replicates": N_BOOT, "seed": BOOT_SEED},
        "null_centre": "mu0 = mean LATE-window statistic of the four validated 7B1 "
                       "chains at the identical extraction (correct kernel, matched "
                       "dependence); its own SE ~ SE_dep/2 is noted, not corrected",
        "threshold": {"clearly_over": Z_OVER, "meaning": "one-sided 99%"},
        "controls": {
            "A": "7B1 leave-one-out z per chain; pass if >= 3/4 within +-2.33 at "
                 "every block length",
            "B": "finite-state grid chain from the exact composed kernel; 4 test "
                 "chains vs mu0 from 8 independent null chains; pass if >= 3/4 "
                 "within +-2.33 at every length"},
        "verdicts_preregistered": {
            "ARTIFACT SUPPORTED": "at ALL lengths <= 1 of 4 start[0] late z_dep > "
                                  "2.33, both controls pass",
            "REAL INTERACTION SUPPORTED": "at ALL lengths >= 3 of 4 start[0] late "
                                          "z_dep > 2.33, both controls pass",
            "INCONCLUSIVE": "anything else — instability across lengths, 2/4 over, "
                            "or any control failure; then review the dependent-sample "
                            "two-sample test choice (spectral/HAC, batch means, "
                            "dependent wild bootstrap) before anything else"},
        "context_only": "run1/rep2 chain-0 full-window z_dep and early/middle probe "
                        "windows are reported for context at length L, never for the "
                        "verdict",
    }
    (OUT / "registration.json").write_text(json.dumps(registration, indent=2))
    print("[dep-cal] registration and estimator audit written", flush=True)

    A, keep, centre, scale = build_reference_rows()
    machine = EnergyMachine(A, N_ROWS)

    # ---- start[0] probe rows and the block-length rule ------------------------------
    began = time.perf_counter()
    probe_rows = {c: chain_rows_of(PROBE / "chains.npz", c, keep, centre, scale)
                  for c in range(4)}
    psi_iacts = {}
    for c in range(4):
        psi = machine.contribution_series(window_rows(probe_rows[c], "late"))
        psi_iacts[c] = iact(psi)
    L = int(math.ceil(max(psi_iacts.values())))
    lengths = sorted({max(2, L // 2), L, 2 * L})
    (OUT / "iact.json").write_text(json.dumps({
        "psi_iact_late_by_chain": {str(c): psi_iacts[c] for c in range(4)},
        "L": L, "sensitivity_lengths": lengths,
        "units": "stride-8 retained-draw index (1 unit = 80 sweeps)"}, indent=2))
    print(f"[dep-cal] IACT by chain {psi_iacts} -> L={L}, lengths {lengths}",
          flush=True)

    # ---- Control A: 7B1 late windows -------------------------------------------------
    b7_rows = {c: chain_rows_of(B7B1 / "chains.npz", c, keep, centre, scale)
               for c in range(4)}
    b7_T = {c: machine.statistic(window_rows(b7_rows[c], "late")) for c in range(4)}
    mu0 = float(np.mean(list(b7_T.values())))
    control_a = {"late_T_by_chain": {str(c): b7_T[c] for c in range(4)}, "mu0": mu0,
                 "loo": {}}
    for length in lengths:
        control_a["loo"][str(length)] = {}
        for c in range(4):
            others = [b7_T[j] for j in range(4) if j != c]
            boot = block_bootstrap_sd(machine, window_rows(b7_rows[c], "late"),
                                      length, N_BOOT, BOOT_SEED + 10 * c + length)
            control_a["loo"][str(length)][str(c)] = {
                "z": (b7_T[c] - float(np.mean(others))) / max(1e-12, boot["sd"]),
                "se_dep": boot["sd"]}
    control_a["within_2p33_by_length"] = {
        str(length): sum(1 for c in range(4)
                         if abs(control_a["loo"][str(length)][str(c)]["z"]) <= Z_OVER)
        for length in lengths}
    control_a["pass"] = bool(all(v >= 3
                                 for v in control_a["within_2p33_by_length"].values()))
    print(f"[dep-cal] control A (7B1): mu0 {mu0:.6f}, within +-2.33: "
          f"{control_a['within_2p33_by_length']}, pass={control_a['pass']}", flush=True)

    # ---- primary: start[0] late windows, all three lengths ---------------------------
    primary = {}
    for c in range(4):
        b = window_rows(probe_rows[c], "late")
        t_obs = machine.statistic(b)
        entry = {"T": t_obs, "iid_frame_z_recorded_in_probe": True, "by_length": {}}
        for length in lengths:
            boot = block_bootstrap_sd(machine, b, length, N_BOOT,
                                      BOOT_SEED + 1000 + 10 * c + length)
            entry["by_length"][str(length)] = {
                "se_dep": boot["sd"],
                "z_dep": (t_obs - mu0) / max(1e-12, boot["sd"])}
        primary[str(c)] = entry
        print(f"[dep-cal] start0 late chain {c}: T {t_obs:.6f}, z_dep "
              f"{[round(entry['by_length'][str(length)]['z_dep'], 2) for length in lengths]}",
              flush=True)

    over_by_length = {str(length): sum(
        1 for c in range(4)
        if primary[str(c)]["by_length"][str(length)]["z_dep"] > Z_OVER)
        for length in lengths}

    # ---- context: early/middle at L, and run1/rep2 chain 0 full-window ---------------
    context = {"early_middle_z_at_L": {}, "chain0_full_window": {}}
    for w in ("early", "middle"):
        context["early_middle_z_at_L"][w] = {}
        for c in range(4):
            b = window_rows(probe_rows[c], w)
            t = machine.statistic(b)
            boot = block_bootstrap_sd(machine, b, L, N_BOOT,
                                      BOOT_SEED + 2000 + 10 * c + (0 if w == "early"
                                                                   else 1))
            context["early_middle_z_at_L"][w][str(c)] = (t - mu0) / max(1e-12,
                                                                        boot["sd"])
    for name, path in (("run1", RUN1), ("rep2", REP2)):
        rows = chain_rows_of(path / "chains.npz", 0, keep, centre, scale)
        b = rows[::max(1, len(rows) // (3 * N_ROWS))][:N_ROWS]   # full-window, n=2000
        t = machine.statistic(b)
        boot = block_bootstrap_sd(machine, b, L, N_BOOT, BOOT_SEED + 3000
                                  + (0 if name == "run1" else 1))
        context["chain0_full_window"][name] = {
            "T": t, "z_dep_at_L": (t - mu0) / max(1e-12, boot["sd"])}

    # ---- Control B ------------------------------------------------------------------
    ctrl_b = control_b(registration["block_length_rule"])
    print(f"[dep-cal] control B (finite-state): within +-2.33 "
          f"{ctrl_b['within_2p33_by_length']}, pass={ctrl_b['pass']}", flush=True)

    (OUT / "dependence_z.json").write_text(json.dumps({
        "mu0_from_7b1_late": mu0, "lengths": lengths,
        "start0_late_primary": primary, "over_2p33_by_length": over_by_length,
        "context": context}, indent=2, default=float))
    (OUT / "controls.json").write_text(json.dumps(
        {"control_a_7b1": control_a, "control_b_finite_state": ctrl_b},
        indent=2, default=float))

    # ---- the pre-registered verdict --------------------------------------------------
    controls_pass = bool(control_a["pass"] and ctrl_b["pass"])
    stable_artifact = all(v <= 1 for v in over_by_length.values())
    stable_interaction = all(v >= 3 for v in over_by_length.values())
    if controls_pass and stable_artifact:
        verdict = "ESTIMATOR ARTIFACT / SERIAL DEPENDENCE SUPPORTED"
    elif controls_pass and stable_interaction:
        verdict = "REAL BASIN INTERACTION SUPPORTED"
    else:
        verdict = "DEPENDENCE CALIBRATION INCONCLUSIVE"
    verdict_payload = {
        "verdict": verdict,
        "over_2p33_by_length": over_by_length,
        "controls_pass": controls_pass,
        "control_a_pass": control_a["pass"], "control_b_pass": ctrl_b["pass"],
        "recommended_next": {
            "if_artifact": "ONE final dispersed-start formal validation with a "
                           "prospectively frozen chain-balanced, dependence-aware, "
                           "full-window gate (all four dispersed starts, kernel and "
                           "cadence unchanged); matched synthetic only after it "
                           "passes",
            "if_interaction": "D0-D_full reduced-target decomposition, each level "
                              "judged against its own independent reference (never "
                              "truth-hit)",
            "if_inconclusive": "review the dependent-sample two-sample test with "
                               "Geoff (spectral/HAC variance, batch means, dependent "
                               "wild bootstrap, block-bootstrap energy test) before "
                               "any further compute"},
        "runtime_seconds": time.perf_counter() - began,
    }
    (OUT / "verdict.json").write_text(json.dumps(verdict_payload, indent=2,
                                                 default=float))

    lines = ["# Dependence-aware calibration of the per-chain energy statistic", "",
             f"**{verdict}**", "",
             f"mu0 (7B1 late mean) = {mu0:.6f}; block lengths {lengths} "
             f"(L from max psi IACT = {max(psi_iacts.values()):.1f}); "
             f"{N_BOOT} circular moving-block replicates; threshold z > {Z_OVER}.", "",
             "| start[0] chain | T_late |" + "".join(
                 f" z_dep (l={length}) |" for length in lengths),
             "|---|---|" + "---|" * len(lengths)]
    for c in range(4):
        entry = primary[str(c)]
        lines.append(f"| {c} | {entry['T']:.6f} |" + "".join(
            f" {entry['by_length'][str(length)]['z_dep']:+.2f} |"
            for length in lengths))
    lines += ["",
              f"Chains over +{Z_OVER} by length: {over_by_length}",
              f"Control A (7B1 leave-one-out): within +-2.33 "
              f"{control_a['within_2p33_by_length']} -> "
              f"{'PASS' if control_a['pass'] else 'FAIL'}",
              f"Control B (finite-state exact): within +-2.33 "
              f"{ctrl_b['within_2p33_by_length']} -> "
              f"{'PASS' if ctrl_b['pass'] else 'FAIL'}", "",
              "Historical verdicts unchanged; kernel unchanged; no new production "
              "chains; matched-synthetic paused."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[dep-cal] {verdict}")
    print(f"[dep-cal] wrote {OUT}")


if __name__ == "__main__":
    main()
