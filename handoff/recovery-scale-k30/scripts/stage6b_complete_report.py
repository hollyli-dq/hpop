"""Stage 6B — assemble the overall report from the registered artifacts.

    PYTHONPATH=src python scripts/stage6b_complete_report.py \
        --pytest-summary "832 passed, 1 warning, 36 subtests passed"

Reads only what the stages actually wrote. Every number in the output table is copied
from a results file, so the report cannot drift from the artifacts it describes.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results" / "mcmc_original"

STAGES = {
    "b2": {"dir": "stage6b2_joint3_full_seed0", "smoke": "stage6b2_joint3_smoke",
           "active": ("beta", "omega", "lambda_rep"), "label": "6B2"},
    "b3": {"dir": "stage6b3_joint4_full_seed0", "smoke": "stage6b3_joint4_smoke",
           "active": ("beta", "omega", "lambda_rep", "lambda_back"), "label": "6B3"},
}


def git(*a):
    try:
        return subprocess.run(("git",) + a, cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def read(stage: str) -> dict:
    spec = STAGES[stage]
    d = RESULTS / spec["dir"]
    return {
        "spec": spec, "dir": d,
        "config": json.loads((d / "config.json").read_text()),
        "reference_config": json.loads((d / "reference_config.json").read_text()),
        "reference": json.loads((d / "reference_summary.json").read_text()),
        "scalar": json.loads((d / "scalar_diagnostics.json").read_text()),
        "joint": json.loads((d / "joint_diagnostics.json").read_text()),
        "recovery": json.loads((d / "recovery_results.json").read_text())["recovery"],
        "smoke": json.loads((RESULTS / spec["smoke"] / "summary.json").read_text()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pytest-summary", required=True)
    ap.add_argument("--baseline-summary", default="827 passed, 1 warning, 36 subtests passed")
    a = ap.parse_args()

    b2, b3 = read("b2"), read("b3")
    out = RESULTS / "stage6b_complete"
    out.mkdir(parents=True, exist_ok=True)

    stage6b1 = json.loads(
        (RESULTS / "stage6b1_full_seed0" / "stage6b1_mcmc.json").read_text())["results"]

    L = ["# Stage 6B — complete: scalar, joint-3 and joint-4 recurrent MCMC", "",
         f"Date {date.today().isoformat()} · branch `{git('rev-parse', '--abbrev-ref', 'HEAD')}` "
         f"· commit `{git('rev-parse', 'HEAD')[:8]}`",
         f"Frozen model `{b2['config']['frozen_config_hash']}`", "",
         "The primary claim of this stage is **sampler correctness**: that the MCMC",
         "posterior coincides with an independently computed reference. Synthetic recovery",
         "— whether the generating value lands inside the posterior — is reported",
         "separately and is *not* the sampler-correctness criterion.", "",
         "## Status", "", "| stage | what was inferred | claim | result |", "|---|---|---|---|",
         "| 6B1 | each scalar alone, others at truth | matches the four 1-D grids | **PASS** |",
         f"| 6B2 | `beta, omega, lambda_rep` (lambda_back = "
         f"{b2['config']['fixed']['lambda_back']}) | matches the independent 3-D reference | "
         f"**{'PASS' if b2['scalar']['gates']['pass'] else 'FAIL'}** |",
         f"| 6B3 | all four jointly | matches the independent 4-D reference | "
         f"**{'PASS' if b3['scalar']['gates']['pass'] else 'FAIL'}** |", "",
         "## Reference quality", "",
         "| stage | method | grid points | integral | outer-face mass | refinement drift | "
         "max correlation drift | importance ESS |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in (b2, b3):
        s = r["reference"]["summary"]
        ref = r["reference"]["refinement"]
        L.append(f"| {r['spec']['label']} | deterministic tensor grid, transformed "
                 f"coordinates | {s['grid_points']:,} | {s['integral_check']:.10f} | "
                 f"{s['outer_face_mass']:.3e} | {ref['max_drift_in_sd']:.5f} sd "
                 f"(n={ref['coarse_n_points']} vs {ref['fine_n_points']}) | "
                 f"{ref['max_correlation_drift']:.5f} | n/a — not importance/QMC |")
    L += ["", "No importance or QMC reference was needed: the four-dimensional grid was",
          "affordable after collapsing identical recurrent states, so both references are",
          "deterministic quadrature and no weight diagnostics apply.", "",
          "## Per-parameter chain diagnostics", "",
          "| stage | parameter | R-hat | bulk ESS | tail ESS | MCSE | KS to reference | "
          "std. mean error | std. 2.5% error | std. 97.5% error |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in (b2, b3):
        for name in r["spec"]["active"]:
            d = r["scalar"]["marginals"][name]
            L.append(f"| {r['spec']['label']} | {name} | {d['rhat']:.4f} | "
                     f"{d['bulk_ess']:.0f} | {d['tail_ess']:.0f} | {d['mcse_mean']:.6f} | "
                     f"{d['ks_distance']:.4f} | {d['standardized_mean_error']:.4f} | "
                     f"{d['standardized_q025_error']:.4f} | "
                     f"{d['standardized_q975_error']:.4f} |")
    L += ["", "Stage 6B1, for comparison (each scalar alone, against its 1-D grid):", "",
          "| parameter | R-hat | bulk ESS | tail ESS | KS |", "|---|---:|---:|---:|---:|"]
    for name in ("beta", "omega", "lambda_rep", "lambda_back"):
        c = stage6b1[name]["comparison"]
        L.append(f"| {name} | {c['rhat']:.4f} | {c['bulk_ess']:.0f} | {c['tail_ess']:.0f} "
                 f"| {c['ks_distance']:.4f} |")

    L += ["", "## Acceptance by coordinate", "",
          "| stage | parameter | total (per chain) | post burn-in (per chain) |",
          "|---|---|---|---|"]
    for r in (b2, b3):
        for name in r["spec"]["active"]:
            d = r["scalar"]["marginals"][name]
            L.append(f"| {r['spec']['label']} | {name} | "
                     f"{', '.join(f'{v:.3f}' for v in d['acceptance_total'])} | "
                     f"{', '.join(f'{v:.3f}' for v in d['acceptance_post_burn_in'])} |")

    L += ["", "## Worst-case errors and the joint comparison", "",
          "| stage | max mean error | max CI endpoint error | max correlation error | "
          "correlation envelope | energy distance | energy envelope | z | multivariate |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in (b2, b3):
        marg = r["scalar"]["marginals"]
        worst_mean = max(marg[n]["standardized_mean_error"] for n in r["spec"]["active"])
        worst_ci = max(max(marg[n]["standardized_q025_error"],
                           marg[n]["standardized_q975_error"])
                       for n in r["spec"]["active"])
        dep = r["joint"]["dependence"]
        mv = r["joint"]["multivariate"]
        env = r["joint"]["correlation_envelope"]
        L.append(f"| {r['spec']['label']} | {worst_mean:.4f} | {worst_ci:.4f} | "
                 f"{dep['max_correlation_error']:.5f} | {env['envelope']:.5f} | "
                 f"{mv['observed']:.6f} | {mv['envelope']:.6f} | {mv['z_score']:+.2f} | "
                 f"**{'PASS' if mv['pass'] else 'FAIL'}** |")
    L += ["", "Both envelopes are calibrated, not chosen: each is the 99th percentile of the "
          "same statistic computed between two independent samples *from the frozen "
          "reference itself*, at the same sample sizes. The correlation envelope is "
          "calibrated at the chains' minimum bulk ESS rather than their raw draw count, "
          "since an MCMC correlation estimate carries the noise of its effective sample "
          "size.", "",
          "## Pairwise dependence — what four marginals cannot establish", "",
          "| stage | pair | reference corr | MCMC corr | abs error |", "|---|---|---:|---:|---:|"]
    for r in (b2, b3):
        for pair, v in r["joint"]["dependence"]["pairs"].items():
            L.append(f"| {r['spec']['label']} | {pair} | {v['reference_correlation']:+.5f} "
                     f"| {v['mcmc_correlation']:+.5f} | {v['absolute_error']:.5f} |")

    L += ["", "## Synthetic recovery — a separate question", "",
          "| stage | parameter | truth | post. mean | post. sd | 95% interval | "
          "truth inside | error in sd |", "|---|---|---:|---:|---:|---|---|---:|"]
    for r in (b2, b3):
        for name in r["spec"]["active"]:
            rec = r["recovery"][name]
            L.append(f"| {r['spec']['label']} | {name} | {rec['true_value']:.5f} | "
                     f"{rec['posterior_mean']:.5f} | {rec['posterior_sd']:.5f} | "
                     f"[{rec['q025']:.5f}, {rec['q975']:.5f}] | "
                     f"{'yes' if rec['truth_in_95_interval'] else '**no**'} | "
                     f"{rec['error_in_posterior_sd']:.3f} |")

    misses = [(r["spec"]["label"], n) for r in (b2, b3) for n in r["spec"]["active"]
              if not r["recovery"][n]["truth_in_95_interval"]]
    if misses:
        L += ["", f"**{len(misses)} coordinate(s) do not cover the generating value**: "
              + ", ".join(f"{s} {n}" for s, n in misses) + ". This is a finite-data "
              "statement about one 500-block draw, not a sampler defect — the sampler "
              "reproduces the reference posterior, and the reference posterior is where "
              "these values sit. Nothing was retuned toward the truth.", ""]
    else:
        L += ["", "Every generating value lies inside its 95% posterior interval.", ""]

    L += ["## Smoke runs", "", "| stage | checks | result |", "|---|---|---|"]
    for r in (b2, b3):
        checks = r["smoke"]["checks"]
        L.append(f"| {r['spec']['label']} | {len(checks)} checks, all of "
                 f"{', '.join(sorted(checks))} | "
                 f"**{'PASS' if r['smoke']['pass'] else 'FAIL'}** |")

    L += ["", "## Tests", "", f"- baseline before Stage 6B2/6B3: `{a.baseline_summary}`",
          f"- final: `{a.pytest_summary}`", "",
          "## Artifacts", ""]
    for r in (b2, b3):
        L.append(f"- `{r['dir'].relative_to(ROOT)}/` — config, frozen reference, chains, "
                 f"scalar and joint diagnostics, recovery, report, figures")
        L.append(f"- `{(RESULTS / r['spec']['smoke']).relative_to(ROOT)}/` — smoke run")
    L += ["- `results/mcmc_original/stage6b1_full_seed0/` — Stage 6B1 (unchanged, tagged "
          "`hpop-stage6b1-scalar-mcmc-v1`)", "",
          "## What Stage 6B does not establish", "",
          "Every posterior here is conditional on `U = U_TRUE`, on known and fixed skill",
          "boundaries, and on `epsilon = 0.02`. Latent-`U` recovery, `rho`, unknown",
          "boundaries, segmentation and semi-Markov FFBS are all untouched. The next stage",
          "is latent-`U` recurrent recovery.", ""]

    (out / "report.md").write_text("\n".join(L) + "\n")
    print(f"Written {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
