"""Turn the raw benchmark results into tables, fits and figures.

    python scripts/scalability/analyze.py [--out results/scalability/optimized_segmental_v1]

Everything downstream of the measurements lives here, and everything here is derived from
`raw/*.json` alone. No number in a table, a figure or a report is typed in by hand; the
report generator reads the same CSV rows a reader can open.

## The fits

`time = c * x^b` is fitted by ordinary least squares on `log10(time)` against `log10(x)`,
over points where the operation banked at least five timed repetitions. Intervals come
from a residual bootstrap:
resampling the fitted residuals rather than the points, because five to seven points are
too few for a pair bootstrap to be stable. The analytic OLS standard error is reported
beside it, and where the two disagree the wider one should be believed.

A fitted exponent is a description of the tested range. It is not extrapolated, and
`extrapolation_caveat` records the factor beyond the largest measured value at which any
statement based on the fit stops being supportable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                     # noqa: E402
import bench_common as bc                                              # noqa: E402
from bench_common import SPARSE_SUPPORT_SIZE                            # noqa: E402
import bench_plan as bp                                                # noqa: E402

import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

PRIMARY_OPS = ("emission_build", "emission_cache_hit", "forward_batched",
               "backward_sample", "ffbs_complete", "cond_plain", "cond_structural",
               "marg_plain", "marg_structural")

STYLE = {
    "figure.dpi": 140, "savefig.dpi": 300, "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11.5, "legend.fontsize": 9.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
}

COLOURS = {
    "emission_build": "#1b6ca8", "emission_cache_hit": "#7fb3d5",
    "forward_batched": "#c0392b", "backward_sample": "#e67e22",
    "ffbs_complete": "#8e44ad", "cond_plain": "#16794a", "cond_structural": "#5fa87c",
    "marg_plain": "#b8860b", "marg_structural": "#d4a843",
}
MARKERS = {"emission_build": "o", "emission_cache_hit": "s", "forward_batched": "^",
           "backward_sample": "v", "ffbs_complete": "D", "cond_plain": "o",
           "cond_structural": "s", "marg_plain": "^", "marg_structural": "v"}


# ------------------------------------------------------------------------- loading
PHASE_DIRS = (("main", "raw"), ("quiet", "raw_quiet"), ("optional", "raw_optional"))
# Preference order for which pass the headline tables and figures are drawn from. The
# controlled pass wins when it exists; when it does not -- a run that only ever took one
# pass -- the analysis must still produce a complete set of outputs rather than empty
# figures, so the preference is resolved against what is actually on disk.
PHASE_PREFERENCE = ("quiet", "main", "optional")


def resolve_primary(records) -> str:
    present = {r.get("_phase", "main") for r in records
               if r.get("status") in ("ok", "partial")}
    for phase in PHASE_PREFERENCE:
        if phase in present:
            return phase
    return "main"


def load(out_dir: Path) -> dict:
    """Every worker record from every phase, each tagged with the phase that took it.

    The passes are kept apart rather than pooled. They were taken under different machine
    conditions, and pooling them would average a contaminated measurement with a
    controlled one and report the result as if it were a single number.
    """
    records = []
    for phase, directory in PHASE_DIRS:
        raw = out_dir / directory
        if not raw.exists():
            continue
        for path in sorted(raw.glob("*__*.json")):
            try:
                record = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            record["_phase"] = phase
            records.append(record)
    # Both phases write into the same `raw/` directory but keep separate state files.
    # The censored table has to cover both, so the task maps are merged with the phase
    # recorded on every row.
    state = {"tasks": {}, "decisions": [], "phases": {}}
    for name, phase in (("state.json", "main"), ("state_quiet.json", "quiet"),
                        ("state_optional.json", "optional")):
        path = out_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        state["phases"][phase] = {
            "started_at_utc": payload.get("started_at_utc"),
            "finished_at_utc": payload.get("finished_at_utc"),
            "runs": payload.get("runs"),
            "n_tasks": len(payload.get("tasks", {})),
        }
        for task_id, record in payload.get("tasks", {}).items():
            state["tasks"][task_id] = {**record, "phase": phase}
        state["decisions"].extend(payload.get("decisions", []))
        if phase == "main":
            for key in ("started_at_utc", "finished_at_utc", "runs", "plan_digest"):
                state[key] = payload.get(key)
    return {"records": records, "state": state}


def row_key(record) -> dict:
    cfg = record["config"]
    return {"axis": cfg["axis"], "label": cfg["label"], "N": cfg["N"], "J": cfg["J"],
            "K": cfg["K"], "A": cfg["A"], "D_min": cfg["D_min"], "D_max": cfg["D_max"],
            "regime": cfg["regime"], "group": record["group"],
            "phase": record.get("_phase", "main"), "seed": cfg.get("seed"),
            "config_id": record.get("config_id", cfg["label"])}


def measured(records) -> list:
    """Records carrying usable timings.

    `partial` is the snapshot a worker flushes after every repetition, so that a
    configuration killed by the driver's hard timeout still leaves the repetitions it had
    already taken. Those repetitions are real measurements and are kept; the record is
    marked censored so nothing downstream reports it as a completed point.
    """
    out = []
    for record in records:
        status = record.get("status")
        if status == "ok":
            out.append(record)
        elif status == "partial":
            record["censored"] = True
            out.append(record)
    return out


# --------------------------------------------------------------------------- tables
def write_raw_timings(records, path: Path) -> int:
    fields = ["config_id", "axis", "label", "N", "J", "K", "A", "D_min", "D_max",
              "regime", "group", "phase", "seed", "operation", "rep_index", "wall_seconds",
              "cpu_seconds"]
    rows = 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in measured(records):
            base = row_key(record)
            for operation, payload in sorted(record["operations"].items()):
                for index, (wall, cpu) in enumerate(zip(payload["wall_raw"],
                                                        payload["cpu_raw"])):
                    writer.writerow({**base, "operation": operation,
                                     "rep_index": index, "wall_seconds": f"{wall:.9f}",
                                     "cpu_seconds": f"{cpu:.9f}"})
                    rows += 1
    return rows


def reference_probe_seconds(records) -> float:
    """The median machine-speed probe across every record that carries one."""
    values = [r.get("speed_probe_median_seconds") for r in records]
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else float("nan")


def write_timing_summary(records, path: Path) -> int:
    fields = ["config_id", "axis", "label", "N", "J", "K", "A", "D_min", "D_max",
              "regime", "group", "phase", "seed", "operation", "n_reps",
              "reduced_repetitions",
              "censored", "budget_limited", "wall_median_s", "wall_iqr_s", "wall_p90_s", "wall_ci_lo_s",
              "wall_ci_hi_s", "wall_ci_rel_half_width", "cpu_median_s", "cpu_iqr_s",
              "cpu_p90_s", "peak_rss_bytes", "legal_blocks_total",
              "legal_blocks_times_skills", "trace_occurrences", "forward_states",
              "forward_total_reductions", "role_relation_density",
              "role_mean_predecessors", "loadavg_before_1m", "loadavg_after_1m",
              "speed_probe_median_s", "speed_normalised_wall_median_s"]
    rows = 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        reference_probe = reference_probe_seconds(measured(records))
        for record in measured(records):
            base = row_key(record)
            geometry, work = record["geometry"], record["forward_work"]
            graph = record["role_graph"]
            probe = record.get("speed_probe_median_seconds")
            probe = float(probe) if probe is not None else float("nan")
            for operation, payload in sorted(record["operations"].items()):
                wall, cpu = payload["wall"], payload["cpu"]
                writer.writerow({
                    **base, "operation": operation, "n_reps": wall.get("n", 0),
                    "reduced_repetitions": int(wall.get("n", 0) < 15),
                    "censored": int(bool(record.get("censored"))),
                    "budget_limited": int(operation in
                                        (record.get("budget_limited_operations") or [])),
                    "wall_median_s": f"{wall.get('median', float('nan')):.9f}",
                    "wall_iqr_s": f"{wall.get('iqr', float('nan')):.9f}",
                    "wall_p90_s": f"{wall.get('p90', float('nan')):.9f}",
                    "wall_ci_lo_s": f"{wall.get('ci_lo', float('nan')):.9f}",
                    "wall_ci_hi_s": f"{wall.get('ci_hi', float('nan')):.9f}",
                    "wall_ci_rel_half_width":
                        f"{wall.get('ci_relative_half_width', float('nan')):.6f}",
                    "cpu_median_s": f"{cpu.get('median', float('nan')):.9f}",
                    "cpu_iqr_s": f"{cpu.get('iqr', float('nan')):.9f}",
                    "cpu_p90_s": f"{cpu.get('p90', float('nan')):.9f}",
                    "peak_rss_bytes": record.get("peak_rss_bytes", 0),
                    "legal_blocks_total": geometry["legal_blocks_total"],
                    "legal_blocks_times_skills": geometry["legal_blocks_times_skills"],
                    "trace_occurrences": geometry["trace_occurrences"],
                    "forward_states": work["forward_states"],
                    "forward_total_reductions": work["forward_total_reductions"],
                    "role_relation_density":
                        f"{graph['mean_relation_density']:.6f}",
                    "role_mean_predecessors":
                        f"{graph['mean_predecessors_per_role']:.4f}",
                    "loadavg_before_1m":
                        f"{(record.get('loadavg_before') or [float('nan')])[0]:.2f}",
                    "loadavg_after_1m":
                        f"{(record.get('loadavg_after') or [float('nan')])[0]:.2f}",
                    "speed_probe_median_s":
                        f"{record.get('speed_probe_median_seconds', float('nan')):.6f}",
                    "speed_normalised_wall_median_s": (
                        f"{wall.get('median', float('nan')) * reference_probe / probe:.9f}"
                        if probe and np.isfinite(probe) else ""),
                })
                rows += 1
    return rows


def write_memory_summary(records, path: Path) -> int:
    fields = ["config_id", "axis", "label", "N", "J", "K", "A", "D_min", "D_max",
              "regime", "group", "phase", "seed", "peak_rss_bytes", "peak_rss_mib",
              "dense_block_table_bytes", "dense_copies_live_in_this_group",
              "batched_stack_bytes_worst_class", "alpha_chart_bytes",
              "batched_alpha_r_bytes", "fast_cumulative_bytes",
              "predicted_arrays_total_bytes", "predicted_process_rss_bytes",
              "projected_banded_bytes_NOT_IMPLEMENTED",
              "projected_banded_saving_ratio_NOT_IMPLEMENTED", "swap_used_mb"]
    rows = 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in measured(records):
            memory = record["predicted_memory"]
            writer.writerow({
                **row_key(record),
                "peak_rss_bytes": record.get("peak_rss_bytes", 0),
                "peak_rss_mib": round(record.get("peak_rss_bytes", 0) / 1024 ** 2, 2),
                **{k: memory.get(k) for k in fields
                   if k in memory},
                "swap_used_mb": (record.get("swap") or {}).get("used_mb", ""),
            })
            rows += 1
    return rows


def write_censored(records, state, path: Path) -> int:
    fields = ["label", "group", "phase", "status", "reason", "seconds", "attempts",
              "predicted_rss_bytes", "partial_result_preserved"]
    rows = 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task_id, record in sorted((state.get("tasks") or {}).items()):
            status = record.get("status", "")
            if status in ("ok", "pending", "running"):
                continue
            label, _, group = task_id.partition("::")
            writer.writerow({
                "label": label, "group": group, "status": status,
                "phase": record.get("phase", "main"),
                "reason": (record.get("reason") or "")[:400],
                "seconds": round(float(record.get("seconds", 0.0)), 2),
                "attempts": record.get("attempts", 0),
                "predicted_rss_bytes": record.get("predicted_rss_bytes", ""),
                "partial_result_preserved": int(bool(record.get("partial_result"))),
            })
            rows += 1
    return rows


# ----------------------------------------------------------------------------- fits
def loglog_fit(x, y, resamples: int = 5000, seed: int = 987) -> dict:
    """`y = c x^b` by OLS on logs, with a residual bootstrap interval on `b`."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[keep], y[keep]
    if x.size < 3 or np.allclose(x, x[0]):
        return {"n_points": int(x.size), "exponent": None,
                "note": "fewer than three distinct measured points; no fit attempted"}
    lx, ly = np.log10(x), np.log10(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    fitted = slope * lx + intercept
    residuals = ly - fitted
    dof = max(1, x.size - 2)
    sigma = float(np.sqrt((residuals ** 2).sum() / dof))
    sxx = float(((lx - lx.mean()) ** 2).sum())
    stderr = sigma / math.sqrt(sxx) if sxx > 0 else float("nan")
    ss_total = float(((ly - ly.mean()) ** 2).sum())
    r_squared = 1.0 - float((residuals ** 2).sum()) / ss_total if ss_total > 0 else 1.0

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, residuals.size, size=(resamples, residuals.size))
    slopes = np.empty(resamples)
    for i in range(resamples):
        slopes[i] = np.polyfit(lx, fitted + residuals[draws[i]], 1)[0]
    lo, hi = np.quantile(slopes, [0.025, 0.975])
    return {
        "n_points": int(x.size),
        "exponent": float(slope),
        "prefactor_log10": float(intercept),
        "bootstrap_ci_95": [float(lo), float(hi)],
        "ols_stderr": float(stderr),
        "ols_ci_95": [float(slope - 1.96 * stderr), float(slope + 1.96 * stderr)],
        "r_squared": float(r_squared),
        "x_min": float(x.min()), "x_max": float(x.max()),
        "y_min_s": float(y.min()), "y_max_s": float(y.max()),
        "extrapolation_caveat": (
            f"fitted over x in [{x.min():g}, {x.max():g}]; no statement beyond "
            f"x = {2 * x.max():g} is supported by this fit"),
    }


MIN_REPS_FOR_FIT = 5


def series(records, axis: str, operation: str, x_field: str, regime=None,
           phase: str | None = None) -> tuple:
    """Measured points for one operation on one axis, in ascending x.

    A point is admitted on the strength of how many timed repetitions its operation
    actually got, not on whether the driver flagged the surrounding group. A group that
    hit the wall while a cheap operation had already banked fifty clean repetitions still
    measured that operation; dropping it would discard good data. Fewer than
    `MIN_REPS_FOR_FIT` repetitions is too thin a median to fit through, and is dropped.
    """
    points = []
    for record in measured(records):
        cfg = record["config"]
        if cfg["axis"] != axis or operation not in record["operations"]:
            continue
        if regime is not None and cfg["regime"] != regime:
            continue
        if phase is not None and record.get("_phase", "main") != phase:
            continue
        if record["operations"][operation]["wall"].get("n", 0) < MIN_REPS_FOR_FIT:
            continue
        if x_field in cfg:
            x = float(cfg[x_field])
        else:
            x = float(record["geometry"].get(x_field, record["forward_work"].get(
                x_field, float("nan"))))
        payload = record["operations"][operation]["wall"]
        points.append((x, payload["median"], payload.get("ci_lo"),
                       payload.get("ci_hi"), record))
    points.sort(key=lambda row: row[0])
    return ([p[0] for p in points], [p[1] for p in points],
            [p[2] for p in points], [p[3] for p in points], [p[4] for p in points])


AXIS_FITS = [
    ("J", "J", "trace length J"),
    ("K", "K", "skill library size K"),
    ("N", "N", "corpus size N"),
    ("D", "D_max", "maximum segment width D"),
    ("D", "legal_blocks_total", "number of legal candidate blocks"),
    ("A_full", "A", "role inventory A, full support"),
    ("A_sparse", "A", "role inventory A, sparse support"),
]


def phases_present(records) -> list:
    seen = []
    for record in measured(records):
        phase = record.get("_phase", "main")
        if phase not in seen:
            seen.append(phase)
    return seen


def complexity_fits(records) -> dict:
    available = [p for p in phases_present(records) if p in ("main", "quiet")]
    primary = resolve_primary(records)
    if primary not in available and available:
        primary = available[0]
    out = {
        "method": "ordinary least squares on log10(median wall seconds) against "
                  "log10(x), over points with at least five timed repetitions; 95% interval from a "
                  "residual bootstrap with 5000 resamples, analytic OLS interval "
                  "reported beside it",
        "primary_phase": primary,
        "phases_fitted": available,
        "why_phases_are_separate":
            "the two passes ran under different machine conditions. On this hardware a "
            "process moved between performance and efficiency cores keeps a whole core "
            "-- cpu/wall stays at one -- while running at roughly half the speed, so "
            "neither wall time nor CPU time reveals the shift. The passes are therefore "
            "fitted separately and the controlled one is primary; pooling them would "
            "average a contaminated measurement with a clean one.",
        "by_phase": {},
        "statistic": "median wall-clock seconds per operation",
        "inclusion_rule": f"an operation is fitted at a point when it banked at least {MIN_REPS_FOR_FIT} timed repetitions there",
        "axes": {},
        "cpu_time_cross_check": {},
    }
    for phase in available:
        rows = {}
        for axis, x_field, description in AXIS_FITS:
            key = f"{axis}::{x_field}"
            rows[key] = {"axis": axis, "x": x_field, "description": description,
                         "operations": {}}
            for operation in PRIMARY_OPS:
                xs, ys, _lo, _hi, _rec = series(records, axis, operation, x_field,
                                                phase=phase)
                if len(xs) >= 3:
                    rows[key]["operations"][operation] = loglog_fit(xs, ys)
        out["by_phase"][phase] = rows
    out["axes"] = out["by_phase"].get(primary, {})
    # the same fit on process CPU time, as the stability cross-check Section 2 asks for
    for axis, x_field, _ in AXIS_FITS:
        key = f"{axis}::{x_field}"
        rows = {}
        for operation in PRIMARY_OPS:
            points = []
            for record in measured(records):
                cfg = record["config"]
                if cfg["axis"] != axis or operation not in record["operations"]:
                    continue
                if record.get("_phase", "main") != primary:
                    continue
                if record["operations"][operation]["cpu"].get("n", 0) < MIN_REPS_FOR_FIT:
                    continue
                x = float(cfg[x_field]) if x_field in cfg else float(
                    record["geometry"].get(x_field, float("nan")))
                points.append((x, record["operations"][operation]["cpu"]["median"]))
            if len(points) >= 3:
                rows[operation] = loglog_fit([p[0] for p in points],
                                             [p[1] for p in points])
        out["cpu_time_cross_check"][key] = rows
    out["machine_speed"] = machine_speed_summary(records)
    return out


def machine_speed_summary(records) -> dict:
    """What the fixed-work probe says about the machine during each pass."""
    out = {}
    for phase in phases_present(records):
        probes, loads = [], []
        for record in measured(records):
            if record.get("_phase", "main") != phase:
                continue
            value = record.get("speed_probe_median_seconds")
            if value is not None and np.isfinite(float(value)):
                probes.append(float(value))
            load = (record.get("loadavg_before") or [None])[0]
            if load is not None:
                loads.append(float(load))
        if not probes:
            out[phase] = {"probe_available": False,
                          "loadavg_median": float(np.median(loads)) if loads else None,
                          "note": "this pass predates the machine-speed probe; its "
                                  "timings can only be judged by load average"}
            continue
        out[phase] = {
            "probe_available": True,
            "probe_median_s": float(np.median(probes)),
            "probe_min_s": float(np.min(probes)),
            "probe_max_s": float(np.max(probes)),
            "probe_spread_ratio": float(np.max(probes) / np.min(probes)),
            "n_records": len(probes),
            "loadavg_median": float(np.median(loads)) if loads else None,
        }
    return out


def is_optional(record) -> bool:
    return str(record["config"]["axis"]).startswith("optional")


def marginalisation_overhead(records, phase: str | None = None) -> dict:
    """FULL-MARG against FULL-COND, per configuration and pooled per axis."""
    phase = phase or resolve_primary(records)
    by_label: dict = {}
    for record in measured(records):
        if is_optional(record) or record.get("_phase", "main") != phase:
            continue
        by_label.setdefault(record["config"]["label"], {})[record["group"]] = record
    rows = {}
    for label, groups in sorted(by_label.items()):
        cond, marg = groups.get("cond"), groups.get("marg")
        if not cond or not marg:
            continue
        cfg = cond["config"]
        def med(record, name):
            payload = record["operations"].get(name)
            return payload["wall"]["median"] if payload else float("nan")
        cond_plain, marg_plain = med(cond, "cond_plain"), med(marg, "marg_plain")
        cond_structural = med(cond, "cond_structural")
        marg_structural = med(marg, "marg_structural")
        cadence = 10
        rows[label] = {
            "axis": cfg["axis"], "N": cfg["N"], "J": cfg["J"], "K": cfg["K"],
            "A": cfg["A"], "D_max": cfg["D_max"], "regime": cfg["regime"],
            "cond_plain_s": cond_plain, "marg_plain_s": marg_plain,
            "cond_structural_s": cond_structural, "marg_structural_s": marg_structural,
            "plain_ratio_marg_over_cond": marg_plain / cond_plain if cond_plain else None,
            "structural_ratio_marg_over_cond": (marg_structural / cond_structural
                                                if cond_structural else None),
            "amortized_cond_s_at_cadence_10":
                (9 * cond_plain + cond_structural) / cadence,
            "amortized_marg_s_at_cadence_10":
                (9 * marg_plain + marg_structural) / cadence,
            "amortized_ratio_at_cadence_10": (
                (9 * marg_plain + marg_structural)
                / (9 * cond_plain + cond_structural)
                if (9 * cond_plain + cond_structural) else None),
            "structural_extra_seconds": marg_structural - cond_structural,
        }
    return {
        "cadence": 10,
        "definition": "the registered structural cadence is one structural sweep in "
                      "ten; the amortized figures weight nine plain sweeps against one "
                      "structural sweep accordingly",
        "structural_measurement_note":
            "every structural repetition is measured with the H-keyed emission cache "
            "forced to miss, so both arms pay a full candidate-table rebuild. That is "
            "the H-moved case and an upper bound; the rebuild component is measured "
            "separately as `emission_build`.",
        "per_configuration": rows,
    }


CORROBORATION_PAIRS = {
    "optional_target_seed2": "target_operating_point",
    "optional_target_full_support_retry": "target_operating_point_full_support",
    "optional_baseline_quiet": "baseline_matched_scale",
    "optional_J_384_seed2": "J_384",
    "optional_K_40_seed2": "K_40",
    "optional_N_128_seed2": "N_128",
    "optional_D_48_seed2": "D_48",
    "optional_A_full_50_seed2": "A_full_50",
    "optional_A_sparse_50_seed2": "A_sparse_50",
}


def pass_comparison(records) -> dict:
    """The same configuration measured in both passes, side by side.

    This is the load-contamination evidence, and it is reported rather than quietly
    corrected. Where a point is much slower in the first pass than the second, the
    machine was slower, not the algorithm.
    """
    by_phase: dict = {"main": {}, "quiet": {}}
    for record in measured(records):
        phase = record.get("_phase", "main")
        if phase not in by_phase:
            continue
        label = record["config"]["label"]
        entry = by_phase[phase].setdefault(label, {"operations": {}})
        entry["loadavg"] = (record.get("loadavg_before") or [None])[0]
        entry["probe"] = record.get("speed_probe_median_seconds")
        for name, payload in record["operations"].items():
            entry["operations"][name] = payload["wall"]["median"]

    rows = {}
    for label in sorted(set(by_phase["main"]) & set(by_phase["quiet"])):
        first, second = by_phase["main"][label], by_phase["quiet"][label]
        shared = sorted(set(first["operations"]) & set(second["operations"]))
        if not shared:
            continue
        ratios = {}
        for name in shared:
            a, b = first["operations"][name], second["operations"][name]
            ratios[name] = {"first_pass_s": a, "quiet_pass_s": b,
                            "first_over_quiet": (a / b) if b else None}
        finite = [v["first_over_quiet"] for v in ratios.values()
                  if v["first_over_quiet"]]
        rows[label] = {
            "first_pass_loadavg": first.get("loadavg"),
            "quiet_pass_loadavg": second.get("loadavg"),
            "quiet_pass_probe_s": second.get("probe"),
            "operations": ratios,
            "median_first_over_quiet": float(np.median(finite)) if finite else None,
        }
    slowdowns = [r["median_first_over_quiet"] for r in rows.values()
                 if r["median_first_over_quiet"]]
    return {
        "purpose": "the same registered configuration measured twice: once while the "
                   "machine was busy, once while it was idle. A ratio above one means "
                   "the first pass was slower, and the load average and speed probe "
                   "beside it say why.",
        "n_configurations_compared": len(rows),
        "median_first_over_quiet_across_configurations":
            float(np.median(slowdowns)) if slowdowns else None,
        "max_first_over_quiet": float(np.max(slowdowns)) if slowdowns else None,
        "per_configuration": rows,
    }


def corroboration(records) -> dict:
    """Optional-phase repeats against the primary measurement of the same point.

    Only a consistency check. A repeat that lands within a few per cent of the primary
    says the primary is not an artifact of one corpus draw or one moment of machine
    load; a repeat that does not says so plainly. Nothing here is a new finding, and
    nothing here replaces a primary number.
    """
    by_label: dict = {}
    for record in measured(records):
        by_label.setdefault(record["config"]["label"], {}).update(
            {name: payload["wall"]["median"]
             for name, payload in record["operations"].items()})
    rows = {}
    for repeat, primary in sorted(CORROBORATION_PAIRS.items()):
        if repeat not in by_label or primary not in by_label:
            continue
        operations = {}
        for name in sorted(set(by_label[repeat]) & set(by_label[primary])):
            a, b = by_label[primary][name], by_label[repeat][name]
            operations[name] = {
                "primary_s": a, "repeat_s": b,
                "ratio_repeat_over_primary": (b / a) if a else None,
                "relative_difference": ((b - a) / a) if a else None,
            }
        if operations:
            worst = max(abs(v["relative_difference"]) for v in operations.values()
                        if v["relative_difference"] is not None)
            rows[repeat] = {"primary_label": primary, "operations": operations,
                            "worst_relative_difference": worst}
    return {
        "purpose": "Section 17 corroboration: a second deterministic data seed, a "
                   "quieter machine, and a retry of the censored points. Consistency "
                   "check only -- never a replacement for a primary measurement, and "
                   "never averaged with one.",
        "pairs": rows,
    }


def regime_comparison(records, phase: str | None = None) -> dict:
    """Full against sparse support at each `A`, which is the honest form of that axis.

    A single power law across the whole `A` range would mislead, because the sparse
    regime is not active below `A = 10`: a support of `min(10, A)` roles IS every role
    when `A` is five or ten, so the two regimes are the same corpus and the same `U` by
    construction, and the curves coincide there for a reason that has nothing to do with
    scaling. The per-`A` ratio shows where the regimes actually separate.
    """
    phase = phase or resolve_primary(records)
    rows: dict = {}
    for record in measured(records):
        cfg = record["config"]
        if cfg["axis"] not in ("A_full", "A_sparse"):
            continue
        if record.get("_phase", "main") != phase:
            continue
        bucket = rows.setdefault(int(cfg["A"]), {})
        for operation, payload in record["operations"].items():
            bucket.setdefault(operation, {})[cfg["regime"]] = payload["wall"]["median"]
        bucket.setdefault("_graph", {})[cfg["regime"]] = {
            "relation_density": record["role_graph"]["mean_relation_density"],
            "mean_predecessors": record["role_graph"]["mean_predecessors_per_role"],
            "max_predecessors": record["role_graph"]["max_predecessors_per_role"],
        }
    out = {}
    for A in sorted(rows):
        entry = {"A": A,
                 "regimes_coincide_by_construction": A <= SPARSE_SUPPORT_SIZE,
                 "role_graph": rows[A].get("_graph", {}), "operations": {}}
        for operation, values in sorted(rows[A].items()):
            if operation == "_graph":
                continue
            full, sparse = values.get("full"), values.get("sparse")
            entry["operations"][operation] = {
                "full_support_s": full, "sparse_support_s": sparse,
                "full_over_sparse": (full / sparse
                                     if full and sparse else None)}
        out[str(A)] = entry
    return {
        "note": "the two regimes are the same construction at A <= "
                f"{SPARSE_SUPPORT_SIZE}, because a support of min(10, A) roles is every "
                "role there; they are never averaged, and the ratio is only "
                "interpretable above that point",
        "sparse_support_size": SPARSE_SUPPORT_SIZE,
        "per_A": out,
    }


def bottleneck_breakdown(records, phase: str | None = None) -> dict:
    """Where the time goes, per configuration, from the measured components."""
    phase = phase or resolve_primary(records)
    by_label: dict = {}
    for record in measured(records):
        if is_optional(record) or record.get("_phase", "main") != phase:
            continue
        by_label.setdefault(record["config"]["label"], {})[record["group"]] = record
    out = {}
    for label, groups in sorted(by_label.items()):
        build, primitives = groups.get("build"), groups.get("primitives")
        cond = groups.get("cond")
        if not (build and primitives and cond):
            continue
        def med(record, name):
            payload = record["operations"].get(name)
            return payload["wall"]["median"] if payload else float("nan")
        emission = med(build, "emission_build")
        forward = med(primitives, "forward_batched")
        backward = med(primitives, "backward_sample")
        ffbs = med(primitives, "ffbs_complete")
        plain = med(cond, "cond_plain")
        structural = med(cond, "cond_structural")
        remainder = plain - ffbs
        components = {"emission_rebuild": emission, "forward": forward,
                      "backward": backward,
                      "gibbs_target_and_validation": remainder}
        finite = {k: v for k, v in components.items() if np.isfinite(v)}
        dominant = max(finite, key=finite.get) if finite else None
        cfg = build["config"]
        out[label] = {
            "axis": cfg["axis"], "N": cfg["N"], "J": cfg["J"], "K": cfg["K"],
            "A": cfg["A"], "D_max": cfg["D_max"], "regime": cfg["regime"],
            "emission_rebuild_s": emission, "forward_s": forward,
            "backward_s": backward, "ffbs_complete_s": ffbs,
            "plain_sweep_s": plain, "structural_sweep_s": structural,
            "gibbs_target_and_validation_s": remainder,
            "forward_share_of_plain_sweep": forward / plain if plain else None,
            "backward_share_of_plain_sweep": backward / plain if plain else None,
            "ffbs_share_of_plain_sweep": ffbs / plain if plain else None,
            "emission_share_of_structural_sweep":
                emission / structural if structural else None,
            "dominant_component_of_a_structural_sweep": (
                "emission_rebuild" if np.isfinite(emission) and np.isfinite(structural)
                and emission > 0.5 * structural else dominant),
            "dominant_component_of_a_plain_sweep": max(
                {"forward": forward, "backward": backward,
                 "gibbs_target_and_validation": remainder}.items(),
                key=lambda kv: (kv[1] if np.isfinite(kv[1]) else -1))[0],
        }
    return out


# --------------------------------------------------------------------------- figures
def _save(fig, out_dir: Path, name: str) -> list:
    paths = []
    for suffix in ("png", "pdf"):
        path = out_dir / "figures" / f"{name}.{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _plot_axis(records, out_dir, name, axis, x_field, xlabel, title, operations,
               regime=None, fits=None, annotate_key=None, phase=None):
    phase = phase or resolve_primary(records)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    plotted = 0
    for operation in operations:
        xs, ys, lo, hi, _ = series(records, axis, operation, x_field, regime,
                                   phase=phase)
        if len(xs) < 2:
            continue
        plotted += 1
        colour = COLOURS.get(operation, "#444444")
        yerr = None
        if all(v is not None and np.isfinite(v) for v in lo + hi):
            yerr = np.vstack([np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)])
            yerr = np.clip(yerr, 0, None)
        label = operation.replace("_", " ")
        exponent = None
        if fits is not None:
            entry = fits.get("axes", {}).get(f"{axis}::{x_field}", {}) \
                        .get("operations", {}).get(operation)
            if entry and entry.get("exponent") is not None:
                exponent = entry["exponent"]
                label = f"{label}  (slope {exponent:.2f})"
        ax.errorbar(xs, ys, yerr=yerr, marker=MARKERS.get(operation, "o"),
                    color=colour, markersize=5.5, linewidth=1.6, capsize=2.5,
                    elinewidth=1.0, label=label)
    if plotted == 0:
        plt.close(fig)
        return []
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("median wall-clock seconds")
    ax.set_title(title, loc="left", pad=8)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.2)
    return _save(fig, out_dir, name)


def figure_memory(records, out_dir) -> list:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, axis, x_field, xlabel in (
            (axes[0], "J", "J", "trace length J"),
            (axes[1], "K", "K", "skill library size K")):
        rows = [r for r in measured(records)
                if r["config"]["axis"] == axis and r["group"] == "marg"
                and r.get("_phase", "main") == resolve_primary(records)]
        rows.sort(key=lambda r: r["config"][x_field])
        if not rows:
            continue
        xs = [r["config"][x_field] for r in rows]
        rss = [r["peak_rss_bytes"] / 2 ** 20 for r in rows]
        dense = [r["predicted_memory"]["dense_block_table_bytes"] / 2 ** 20
                 for r in rows]
        banded = [r["predicted_memory"]["projected_banded_bytes_NOT_IMPLEMENTED"]
                  / 2 ** 20 for r in rows]
        ax.plot(xs, rss, "o-", color="#1b6ca8", label="measured peak RSS", linewidth=1.8)
        ax.plot(xs, dense, "s--", color="#c0392b",
                label="dense score table (one copy)", linewidth=1.6)
        ax.plot(xs, banded, "^:", color="#7f8c8d",
                label="projected banded — NOT IMPLEMENTED", linewidth=1.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mebibytes")
        ax.set_title(f"memory against {xlabel}", loc="left")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        ax.grid(True, which="both", alpha=0.2)
    fig.suptitle("Measured resident memory and the counterfactual banded layout",
                 x=0.01, ha="left", fontsize=13)
    return _save(fig, out_dir, "fig_memory_JK")


def figure_marg_overhead(records, out_dir, overhead) -> list:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    panels = (("J", "J", "trace length J"), ("K", "K", "skill library size K"),
              ("N", "N", "corpus size N"))
    for ax, (axis, field, xlabel) in zip(axes, panels):
        rows = [(v[field], v) for v in overhead["per_configuration"].values()
                if v["axis"] == axis]
        rows.sort()
        if not rows:
            continue
        xs = [r[0] for r in rows]
        for key, colour, marker, label in (
                ("plain_ratio_marg_over_cond", "#16794a", "o", "plain sweep"),
                ("structural_ratio_marg_over_cond", "#b8860b", "s", "structural sweep"),
                ("amortized_ratio_at_cadence_10", "#8e44ad", "^",
                 "amortized at cadence 1/10")):
            ys = [r[1][key] for r in rows]
            if not any(y is not None and np.isfinite(y) for y in ys):
                continue
            ax.plot(xs, ys, marker=marker, color=colour, linewidth=1.7,
                    markersize=5.5, label=label)
        ax.axhline(1.0, color="#666666", linewidth=1.0, linestyle=":")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("FULL-MARG / FULL-COND wall time")
        ax.set_title(f"against {xlabel}", loc="left")
        ax.legend(frameon=False, fontsize=8.5)
        ax.grid(True, which="both", alpha=0.2)
    fig.suptitle("Marginalisation overhead after optimisation", x=0.01, ha="left",
                 fontsize=13)
    return _save(fig, out_dir, "fig_marg_overhead")


def figure_target(records, out_dir) -> list:
    rows = [r for r in measured(records)
            if r["config"]["axis"] in ("target", "target_long")
            and r.get("_phase", "main") == resolve_primary(records)]
    if not rows:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    labels, values, colours = [], [], []
    for record in rows:
        for operation, payload in sorted(record["operations"].items()):
            labels.append(f"{record['config']['label'].replace('target_', '')}\n"
                          f"{operation.replace('_', ' ')}")
            values.append(payload["wall"]["median"])
            colours.append(COLOURS.get(operation, "#555555"))
    order = np.argsort(values)
    ax.barh([labels[i] for i in order], [values[i] for i in order],
            color=[colours[i] for i in order], height=0.65)
    ax.set_xscale("log")
    ax.set_xlabel("median wall-clock seconds")
    ax.set_title("target operating point — per-operation cost", loc="left")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(True, axis="x", which="both", alpha=0.25)

    ax = axes[1]
    names, rss = [], []
    for record in rows:
        names.append(f"{record['config']['label'].replace('target_', '')}\n"
                     f"{record['group']}")
        rss.append(record["peak_rss_bytes"] / 2 ** 30)
    ax.bar(range(len(names)), rss, color="#1b6ca8", width=0.62)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7.5, rotation=30, ha="right")
    ax.set_ylabel("peak RSS (GiB)")
    ax.axhline(bc.memory_cap_bytes() / 2 ** 30, color="#c0392b", linestyle="--",
               linewidth=1.2, label="memory gate")
    ax.set_title("peak resident memory", loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Anticipated real-data operating point (N=100, J=200, K=20, A=50)",
                 x=0.01, ha="left", fontsize=13)
    return _save(fig, out_dir, "fig_target_operating_point")


def figure_breakdown(records, out_dir, breakdown) -> list:
    if not breakdown:
        return []
    picks = [label for label in ("baseline_matched_scale", "J_192", "K_20",
                                 "A_full_50", "A_sparse_50",
                                 "target_operating_point")
             if label in breakdown]
    if not picks:
        picks = sorted(breakdown)[:6]
    components = ("forward_s", "backward_s", "gibbs_target_and_validation_s")
    pretty = {"forward_s": "optimized forward", "backward_s": "FFBS backward draw",
              "gibbs_target_and_validation_s": "pi/P Gibbs, target, validators"}
    colours = ("#c0392b", "#e67e22", "#16794a")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

    ax = axes[0]
    bottom = np.zeros(len(picks))
    for component, colour in zip(components, colours):
        values = np.array([max(0.0, breakdown[p].get(component) or 0.0) for p in picks])
        ax.bar(range(len(picks)), values, bottom=bottom, color=colour,
               label=pretty[component], width=0.62)
        bottom += values
    ax.set_xticks(range(len(picks)))
    ax.set_xticklabels([p.replace("_", "\n") for p in picks], fontsize=8)
    ax.set_ylabel("seconds per plain sweep")
    ax.set_title("plain sweep — where the time goes", loc="left")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    emission = [breakdown[p].get("emission_rebuild_s") or 0.0 for p in picks]
    rest = [max(0.0, (breakdown[p].get("structural_sweep_s") or 0.0) - e)
            for p, e in zip(picks, emission)]
    ax.bar(range(len(picks)), emission, color="#1b6ca8",
           label="candidate table rebuild", width=0.62)
    ax.bar(range(len(picks)), rest, bottom=emission, color="#8e44ad",
           label="structural move, FFBS, Gibbs, target", width=0.62)
    ax.set_xticks(range(len(picks)))
    ax.set_xticklabels([p.replace("_", "\n") for p in picks], fontsize=8)
    ax.set_ylabel("seconds per structural sweep")
    ax.set_title("structural sweep — where the time goes", loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Runtime breakdown by measured component", x=0.01, ha="left",
                 fontsize=13)
    return _save(fig, out_dir, "fig_runtime_breakdown")


def figure_A(records, out_dir, fits) -> list:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharey=True)
    for ax, axis, title in ((axes[0], "A_full", "A. full support — every skill on all A"),
                            (axes[1], "A_sparse",
                             "B. sparse support — ten roles per skill")):
        for operation in ("emission_build", "forward_batched", "cond_plain",
                          "marg_structural"):
            xs, ys, lo, hi, _ = series(records, axis, operation, "A",
                                       phase=resolve_primary(records))
            if len(xs) < 2:
                continue
            entry = fits.get("axes", {}).get(f"{axis}::A", {}).get(
                "operations", {}).get(operation)
            label = operation.replace("_", " ")
            if entry and entry.get("exponent") is not None:
                label = f"{label}  (slope {entry['exponent']:.2f})"
            ax.plot(xs, ys, marker=MARKERS.get(operation, "o"),
                    color=COLOURS.get(operation, "#444"), linewidth=1.7,
                    markersize=5.5, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("role inventory size A")
        ax.set_title(title, loc="left", fontsize=11.5)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        ax.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("median wall-clock seconds")
    fig.suptitle("Role-inventory scaling, reported separately by support regime "
                 "(never averaged)", x=0.01, ha="left", fontsize=13)
    return _save(fig, out_dir, "fig_scaling_A")


def make_figures(records, out_dir, fits, overhead, breakdown) -> dict:
    plt.rcParams.update(STYLE)
    figures = {}
    figures["fig_scaling_J"] = _plot_axis(
        records, out_dir, "fig_scaling_J", "J", "J", "trace length J",
        "Scaling in trace length (N=16, K=10, A=20, D in [3,12])",
        ("emission_build", "forward_batched", "backward_sample", "ffbs_complete",
         "cond_plain", "marg_structural"), fits=fits)
    figures["fig_scaling_K"] = _plot_axis(
        records, out_dir, "fig_scaling_K", "K", "K", "skill library size K",
        "Scaling in the skill library (N=32, J=128, A=20, D in [3,12])",
        ("emission_build", "forward_batched", "backward_sample", "ffbs_complete",
         "cond_plain", "marg_structural"), fits=fits)
    figures["fig_scaling_N"] = _plot_axis(
        records, out_dir, "fig_scaling_N", "N", "N", "corpus size N",
        "Scaling in corpus size (J=128, K=10, A=20, D in [3,12])",
        ("emission_build", "forward_batched", "backward_sample", "ffbs_complete",
         "cond_plain", "marg_structural"), fits=fits)
    figures["fig_scaling_D"] = _plot_axis(
        records, out_dir, "fig_scaling_D", "D", "legal_blocks_total",
        "number of legal candidate blocks",
        "Scaling in segment width, against legal blocks (N=16, J=192, K=10, A=20)",
        ("emission_build", "forward_batched", "backward_sample", "ffbs_complete",
         "cond_plain", "marg_structural"), fits=fits)
    figures["fig_scaling_A"] = figure_A(records, out_dir, fits)
    figures["fig_memory_JK"] = figure_memory(records, out_dir)
    figures["fig_marg_overhead"] = figure_marg_overhead(records, out_dir, overhead)
    figures["fig_target_operating_point"] = figure_target(records, out_dir)
    figures["fig_runtime_breakdown"] = figure_breakdown(records, out_dir, breakdown)
    return {k: v for k, v in figures.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(bc.RESULTS))
    args = parser.parse_args()
    out_dir = Path(args.out)

    loaded = load(out_dir)
    records, state = loaded["records"], loaded["state"]
    counts = {
        "raw_timings_rows": write_raw_timings(records, out_dir / "raw_timings.csv"),
        "timing_summary_rows": write_timing_summary(records,
                                                    out_dir / "timing_summary.csv"),
        "memory_summary_rows": write_memory_summary(records,
                                                    out_dir / "memory_summary.csv"),
        "censored_rows": write_censored(records, state, out_dir / "censored_points.csv"),
    }
    fits = complexity_fits(records)
    overhead = marginalisation_overhead(records)
    breakdown = bottleneck_breakdown(records)
    bc.atomic_write(out_dir / "complexity_fits.json",
                    json.dumps(fits, indent=2, sort_keys=True, default=float))
    bc.atomic_write(out_dir / "marginalisation_overhead.json",
                    json.dumps(overhead, indent=2, sort_keys=True, default=float))
    bc.atomic_write(out_dir / "pass_comparison.json",
                    json.dumps(pass_comparison(records), indent=2, sort_keys=True,
                               default=float))
    bc.atomic_write(out_dir / "corroboration.json",
                    json.dumps(corroboration(records), indent=2, sort_keys=True,
                               default=float))
    bc.atomic_write(out_dir / "regime_comparison.json",
                    json.dumps(regime_comparison(records), indent=2, sort_keys=True,
                               default=float))
    bc.atomic_write(out_dir / "runtime_breakdown.json",
                    json.dumps(breakdown, indent=2, sort_keys=True, default=float))
    figures = make_figures(records, out_dir, fits, overhead, breakdown)
    bc.atomic_write(out_dir / "analysis_index.json",
                    json.dumps({"counts": counts, "figures": figures,
                                "n_records": len(records),
                                "n_measured": len(measured(records))},
                               indent=2, sort_keys=True))
    print(json.dumps({"counts": counts, "figures": sorted(figures)}, indent=2))


if __name__ == "__main__":
    main()
