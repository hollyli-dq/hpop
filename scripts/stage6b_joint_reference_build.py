"""Stage 6B2 / 6B3 — build and freeze the independent joint reference posteriors.

    PYTHONPATH=src python scripts/stage6b_joint_reference_build.py --stage b2 \
        --output-dir results/mcmc_original/stage6b2_joint3_full_seed0
    PYTHONPATH=src python scripts/stage6b_joint_reference_build.py --stage b3 \
        --output-dir results/mcmc_original/stage6b3_joint4_full_seed0

These references are built by direct quadrature on the recurrent log posterior. No MCMC
transition kernel, acceptance ratio, or draw is involved — see
`stage6b_joint_reference.py`. Each reference is written once and then treated as
immutable: the chains are compared against it, never the other way round.

A coarser grid is also built and reported as a refinement check, so the quoted summaries
come with evidence that they are resolved rather than an assertion that they are.
"""
from __future__ import annotations

import argparse, json, platform, subprocess, sys, time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.recurrent_joint_scalar_mcmc import (                      # noqa: E402
    ACTIVE_B2, ACTIVE_B3, RecurrentJointEvaluator)
from hpop.mcmc_original.stage6b_frozen import (                                   # noqa: E402
    config_hash, frozen_config, load_frozen_dataset)
from hpop.mcmc_original.stage6b_joint_reference import (                          # noqa: E402
    build_reference, reference_summary)

STAGES = {
    "b2": {"active": ACTIVE_B2, "n_points": 61, "refine_n": 41,
           "fixed_from_truth": ("lambda_back",)},
    "b3": {"active": ACTIVE_B3, "n_points": 51, "refine_n": 35,
           "fixed_from_truth": ()},
}


def git(*a):
    try:
        return subprocess.run(("git",) + a, cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def jsonable(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.floating, np.integer)): return v.item()
    if isinstance(v, (np.bool_, bool)): return bool(v)
    if isinstance(v, dict): return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jsonable(x) for x in v]
    return v


def refinement_report(fine: dict, coarse: dict, active) -> dict:
    """Max absolute drift between the two resolutions, in units of the fine sd."""
    rows = {}
    worst = 0.0
    for name in active:
        sd = fine["sd"][name]
        entry = {}
        for key in ("mean", "sd", "median", "q025", "q975"):
            drift = abs(fine[key][name] - coarse[key][name])
            entry[key] = {"fine": fine[key][name], "coarse": coarse[key][name],
                          "abs_drift": drift, "drift_in_sd": drift / sd}
            worst = max(worst, drift / sd)
        rows[name] = entry
    fine_corr = np.array(fine["correlation"])
    coarse_corr = np.array(coarse["correlation"])
    corr_drift = float(np.abs(fine_corr - coarse_corr).max())
    return {"per_parameter": rows, "max_drift_in_sd": worst,
            "max_correlation_drift": corr_drift,
            "fine_n_points": fine["n_points"], "coarse_n_points": coarse["n_points"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("b2", "b3"), required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-points", type=int, default=None)
    ap.add_argument("--refine-n", type=int, default=None)
    ap.add_argument("--radius", type=float, default=6.0)
    a = ap.parse_args()

    spec = STAGES[a.stage]
    active = spec["active"]
    n_points = a.n_points or spec["n_points"]
    refine_n = a.refine_n or spec["refine_n"]
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)

    frozen = load_frozen_dataset()
    evaluator = RecurrentJointEvaluator(frozen.train, frozen.u_true, frozen.epsilon)
    fixed = {name: frozen.truth[name] for name in spec["fixed_from_truth"]}

    print(f"Stage {a.stage.upper()} reference — active {active}, fixed {fixed}")
    print(f"  frozen config hash {config_hash()}")

    began = time.perf_counter()
    fine = build_reference(active, evaluator, fixed, frozen.truth, frozen.epsilon,
                           n_points=n_points, radius=a.radius)
    fine_summary = reference_summary(fine)
    fine_seconds = time.perf_counter() - began
    print(f"  fine grid   n={n_points:<3} {fine_summary['grid_points']:>10,} points "
          f"({fine_seconds:6.1f}s)  integral {fine_summary['integral_check']:.10f}  "
          f"outer-face {fine_summary['outer_face_mass']:.3e}")

    began = time.perf_counter()
    coarse = build_reference(active, evaluator, fixed, frozen.truth, frozen.epsilon,
                             n_points=refine_n, radius=a.radius)
    coarse_summary = reference_summary(coarse)
    print(f"  coarse grid n={refine_n:<3} {coarse_summary['grid_points']:>10,} points "
          f"({time.perf_counter() - began:6.1f}s)")

    refinement = refinement_report(fine_summary, coarse_summary, active)
    print(f"  refinement: max drift {refinement['max_drift_in_sd']:.5f} sd, "
          f"max correlation drift {refinement['max_correlation_drift']:.5f}")

    # Only the log density is stored, as float32: the normalised density is derivable
    # from it exactly, and a 4-D grid in float64 would be a ~100 MB artifact for no gain.
    # float32 holds ~7 significant digits of a log density whose summaries are already
    # written to reference_summary.json in full precision.
    np.savez_compressed(
        out / "joint_reference.npz",
        log_density_z=fine.log_density_z.astype(np.float32),
        **{f"axis_z_{n}": fine.axes_z[n] for n in active},
        **{f"axis_value_{n}": fine.axes_value[n] for n in active})

    reference_config = {
        "stage": a.stage, "active": list(active), "fixed": jsonable(fixed),
        "date": date.today().isoformat(),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": git("rev-parse", "HEAD"),
        "python": platform.python_version(), "numpy": np.__version__,
        "frozen_config_hash": config_hash(), "frozen_config": frozen_config(),
        "grid": fine.config, "refine_n_points": refine_n,
        "build_seconds": fine_seconds,
        "provenance": "direct quadrature on the recurrent log posterior; no MCMC "
                      "kernel, acceptance ratio, or draw is used at any point",
    }
    (out / "reference_config.json").write_text(
        json.dumps(jsonable(reference_config), indent=2) + "\n")
    (out / "reference_summary.json").write_text(json.dumps(jsonable({
        "summary": fine_summary, "coarse_summary": coarse_summary,
        "refinement": refinement}), indent=2) + "\n")

    print(f"\n  {'param':<13}{'mean':>10}{'sd':>10}{'median':>10}{'2.5%':>10}{'97.5%':>10}"
          f"{'truth':>10}")
    for name in active:
        print(f"  {name:<13}{fine_summary['mean'][name]:>10.5f}{fine_summary['sd'][name]:>10.5f}"
              f"{fine_summary['median'][name]:>10.5f}{fine_summary['q025'][name]:>10.5f}"
              f"{fine_summary['q975'][name]:>10.5f}{frozen.truth[name]:>10.5f}")
    print(f"\n  correlation:\n{np.round(np.array(fine_summary['correlation']), 4)}")
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
