#!/usr/bin/env python3
"""Truth-free, read-only interim diagnostics for the live FULL-LATENT run.

This script is deliberately outside the frozen formal source set.  It only opens the
atomic ``formal_chains/*.npz`` snapshots with ``allow_pickle=False`` and writes a
separate exploratory audit.  It never imports the sampler, corpus loader, recovery
code, synthetic truth, or held-out data; it cannot alter a process or checkpoint.

The formal 30k gate remains authoritative.  These diagnostics use the same registered
rank-normalized split-R-hat / ESS functions and registered permutation-invariant summary
arrays, but do not write a gate or a stopping decision.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


AUDIT_DIR = Path(__file__).resolve().parent
RUN_DIR = AUDIT_DIR.parent
ROOT = RUN_DIR.parents[2]
CHAIN_DIR = RUN_DIR / "formal_chains"
LAUNCH_MANIFEST = RUN_DIR / "launch_manifest.json"
MIDRUN_UNSEAL_RECORD = RUN_DIR / "TRUTH_UNSEAL_midrun.json"
ARMS = ("FULL-COND", "FULL-MARG")
PREFIX = {"FULL-COND": "full_cond", "FULL-MARG": "full_marg"}
BURN_IN = 10_000
THIN = 5
RECENT_SWEEPS = 2_000
RHAT_THRESHOLD = 1.01
FORBIDDEN = {
    "hpop.mcmc_original.recurrent_synthetic",
    "hpop.mcmc_original.matched_synthetic_generator",
    "hpop.mcmc_original.generate_matched_formal_corpus",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _load_registered_diagnostic_functions():
    """Import only the frozen diagnostic implementation, then prove no truth module loaded."""
    sys.path.insert(0, str(ROOT / "src"))
    from hpop.mcmc_original.stage6b_mcmc_diagnostics import (  # pylint: disable=import-outside-toplevel
        bulk_ess,
        rank_normalized_split_rhat,
        tail_ess,
    )
    loaded = sorted(name for name in FORBIDDEN if name in sys.modules)
    if loaded:
        raise RuntimeError(f"truth-bearing module imported by read-only diagnostic: {loaded}")
    return rank_normalized_split_rhat, bulk_ess, tail_ess


RANK_RHAT, BULK_ESS, TAIL_ESS = _load_registered_diagnostic_functions()


def _checkpoint_path(arm: str, index: int) -> Path:
    return CHAIN_DIR / f"{PREFIX[arm]}_{index}.npz"


def _read_chain(arm: str, index: int) -> dict:
    path = _checkpoint_path(arm, index)
    if not path.is_file():
        raise FileNotFoundError(path)
    # os.replace makes a complete old or complete new file visible; no checkpoint is written.
    before = path.stat()
    digest = _sha256(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        summaries = {
            key[len("summary__"):]: np.asarray(data[key])
            for key in data.files if key.startswith("summary__")
        }
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"checkpoint changed while being read; rerun on a stable atomic snapshot: {path}")
    return {"path": str(path.relative_to(ROOT)), "sha256": digest, "meta": meta,
            "summaries": summaries}


def _summary_labels(summaries: dict[str, np.ndarray]) -> list[tuple[str, tuple[int, ...]]]:
    labels = []
    for name in sorted(summaries):
        shape = summaries[name].shape[1:]
        if not shape:
            labels.append((name, ()))
        else:
            labels.extend((name, tuple(int(v) for v in index))
                          for index in itertools.product(*(range(int(n)) for n in shape)))
    return labels


def _label(name: str, index: tuple[int, ...]) -> str:
    return name if not index else f"{name}[{','.join(str(v) for v in index)}]"


def _series(chain: dict, name: str, index: tuple[int, ...]) -> np.ndarray:
    value = chain["summaries"][name]
    if index:
        value = value[(slice(None),) + index]
    return np.asarray(value, dtype=float)


def _registered_diag(chains: list[np.ndarray]) -> dict:
    values = np.asarray(chains, dtype=float)
    if values.ndim != 2 or values.shape[0] != 4 or values.shape[1] < 4:
        raise ValueError(f"expected four equal-length chains, got {values.shape}")
    constant = [bool(np.all(chain == chain[0])) for chain in values]
    if all(constant):
        unique = {float(chain[0]) for chain in values}
        if len(unique) == 1:
            return {"rhat": 1.0, "bulk_ess": float(values.size),
                    "tail_ess": float(values.size), "degenerate": "constant"}
        return {"rhat": float("inf"), "bulk_ess": 0.0, "tail_ess": 0.0,
                "degenerate": "constant-but-unequal"}
    return {"rhat": float(RANK_RHAT(values)["rhat"]),
            "bulk_ess": float(BULK_ESS(values)), "tail_ess": float(TAIL_ESS(values)),
            "degenerate": None}


def _recent_size(n_draws: int) -> int:
    return min(int(n_draws), int(RECENT_SWEEPS // THIN))


def _shape_of_disagreement(chains: list[np.ndarray]) -> str:
    values = [np.asarray(row, dtype=float) for row in chains]
    all_constant = all(np.all(row == row[0]) for row in values)
    means = np.array([row.mean() for row in values])
    if all_constant and len(set(float(row[0]) for row in values)) > 1:
        return "constant-but-different across chains"
    pooled = np.concatenate(values)
    binary = bool(np.all(np.isin(np.unique(pooled), (0.0, 1.0))))
    if binary:
        rates = np.array([row.mean() for row in values])
        if rates.min() < 0.05 or rates.max() > 0.95:
            return "rare/near-deterministic binary disagreement"
        return "binary co-occurrence disagreement"
    recent = np.array([row[-_recent_size(row.size):].mean() for row in values])
    within = np.mean([max(float(row.std()), 1e-12) for row in values])
    if float(np.ptp(means)) > 2.0 * within:
        if float(np.ptp(recent)) < float(np.ptp(means)):
            return "slowly drifting toward agreement"
        return "broad continuous/discrete separation"
    if any(np.unique(row).size > 2 for row in values):
        return "within-chain multi-state movement"
    return "limited cross-chain separation"


def _probe_location(label: str, probes: dict) -> dict | None:
    if "[" not in label:
        return None
    base, tail = label.split("[", 1)
    index = int(tail.rstrip("]").split(",")[0])
    if base == "boundary_probes":
        return {"kind": "boundary", "trace_position": probes["boundary"][index]}
    if base in {"coskill_probes", "same_segment_probes"}:
        return {"kind": base, "trace_left_right": probes["coskill"][index]}
    return None


def _metric_record(label: str, chains: list[np.ndarray], probes: dict) -> dict:
    diag = _registered_diag(chains)
    recent_n = _recent_size(chains[0].size)
    chain_rows = []
    for series in chains:
        recent = series[-recent_n:]
        chain_rows.append({
            "posterior_mean": float(series.mean()),
            "posterior_sd": float(series.std()),
            "recent_mean": float(recent.mean()),
            "recent_sd": float(recent.std()),
            "last": float(series[-1]),
            "unique_values": int(np.unique(series).size),
            "recent_unique_values": int(np.unique(recent).size),
        })
    return {"rhat": diag["rhat"], "bulk_ess": diag["bulk_ess"],
            "tail_ess": diag["tail_ess"], "degenerate": diag["degenerate"],
            "disagreement": _shape_of_disagreement(chains),
            "recent_draws": recent_n, "per_chain": chain_rows,
            "probe": _probe_location(label, probes)}


def _rhat_rank(item: tuple[str, dict]) -> tuple[int, float]:
    value = float(item[1]["rhat"])
    return (0 if math.isinf(value) else 1, -value if math.isfinite(value) else 0.0)


def _probe_summary(metrics: dict, prefix: str) -> dict:
    selected = [(label, value) for label, value in metrics.items()
                if label.startswith(prefix + "[")]
    finite = [float(value["rhat"]) for _, value in selected if math.isfinite(float(value["rhat"]))]
    constant_different = sum(value["degenerate"] == "constant-but-unequal"
                             for _, value in selected)
    threshold_count = sum(float(value["rhat"]) <= RHAT_THRESHOLD for _, value in selected)
    ordered = sorted(selected, key=_rhat_rank)[:5]
    return {
        "n_probes": len(selected), "median_rhat": float(np.median(finite)) if finite else float("inf"),
        "max_rhat": max((float(value["rhat"]) for _, value in selected), default=float("nan")),
        "at_registered_rhat_threshold": threshold_count,
        "fraction_at_registered_rhat_threshold": threshold_count / max(1, len(selected)),
        "constant_but_different": constant_different,
        "top5": [{"label": label, **value} for label, value in ordered],
    }


def _counter_distribution(values: np.ndarray) -> list[dict]:
    counts = Counter(int(v) for v in np.asarray(values, dtype=int))
    return [{"value": key, "draws": int(counts[key]), "fraction": counts[key] / len(values)}
            for key in sorted(counts)]


def _tuple_distribution(values: np.ndarray) -> list[dict]:
    counts = Counter(tuple(int(v) for v in row) for row in np.asarray(values, dtype=int))
    return [{"tuple": list(key), "draws": int(counts[key]), "fraction": counts[key] / len(values)}
            for key in sorted(counts)]


def _basin_summary(chain: dict, baseline_chain: dict | None) -> dict:
    summaries = chain["summaries"]
    n = int(summaries["total_relations"].shape[0])
    recent_n = _recent_size(n)
    total = np.asarray(summaries["total_relations"][-recent_n:], dtype=int)
    relation_tuple = np.asarray(summaries["sorted_relation_counts"][-recent_n:], dtype=int)
    tuple_changes = int(np.count_nonzero(np.any(relation_tuple[1:] != relation_tuple[:-1], axis=1)))
    structural = chain["meta"]["structural"]
    result = {
        "recent_sweep_window": [BURN_IN + THIN * (n - recent_n + 1), BURN_IN + THIN * n],
        "recent_total_relation_distribution": _counter_distribution(total),
        "recent_sorted_relation_count_distribution": _tuple_distribution(relation_tuple),
        "distinct_recent_relation_count_tuples": int(np.unique(relation_tuple, axis=0).shape[0]),
        "recent_tuple_changes": tuple_changes,
        "basin_description": (
            "stationary in one coarse, permutation-invariant relation-count basin"
            if np.unique(relation_tuple, axis=0).shape[0] == 1
            else "still relocating among coarse, permutation-invariant relation-count basins"
        ),
        "cumulative_structural_attempts": int(structural["attempts"]),
        "cumulative_structural_accepts": int(structural["accepts"]),
        "cumulative_H_changes": int(structural["h_accepts"]),
    }
    if baseline_chain:
        old = baseline_chain["meta"]["structural"]
        result["since_baseline"] = {
            "sweeps": int(chain["meta"]["state"]["iteration"] - baseline_chain["meta"]["state"]["iteration"]),
            "accepted_structural_moves": int(structural["accepts"] - old["accepts"]),
            "accepted_H_changes": int(structural["h_accepts"] - old["h_accepts"]),
        }
    return result


def _read_arm(arm: str) -> dict:
    chains = [_read_chain(arm, index) for index in range(4)]
    iterations = [int(chain["meta"]["state"]["iteration"]) for chain in chains]
    retained = [int(chain["meta"]["retained_draws"]) for chain in chains]
    if len(set(iterations)) != 1 or len(set(retained)) != 1:
        raise RuntimeError(f"{arm} is between checkpoint waves: iterations={iterations}, retained={retained}")
    names = set(chains[0]["summaries"])
    if any(set(chain["summaries"]) != names for chain in chains):
        raise RuntimeError(f"{arm} checkpoint summaries have different schemas")
    probes = chains[0]["meta"]["probes"]
    metrics = {}
    for name, index in _summary_labels(chains[0]["summaries"]):
        label = _label(name, index)
        metrics[label] = _metric_record(label, [_series(chain, name, index) for chain in chains],
                                        probes)
    worst_label, worst = sorted(metrics.items(), key=_rhat_rank)[0]
    return {
        "arm": arm, "iteration": iterations[0], "retained_draws_per_chain": retained,
        "checkpoints": [{"path": chain["path"], "sha256": chain["sha256"],
                         "seed": int(chain["meta"]["seed"]),
                         "seconds": float(chain["meta"]["seconds"])} for chain in chains],
        "chains": chains, "metrics": metrics,
        "worst_invariant": {"label": worst_label, **worst},
        "probe_summaries": {
            "boundary": _probe_summary(metrics, "boundary_probes"),
            "coskill": _probe_summary(metrics, "coskill_probes"),
            "same_segment": _probe_summary(metrics, "same_segment_probes"),
        },
    }


def _compact_arm(arm: dict, baseline: dict | None = None) -> dict:
    metrics = arm["metrics"]
    out = {
        "iteration": arm["iteration"], "retained_draws_per_chain": arm["retained_draws_per_chain"],
        "checkpoints": arm["checkpoints"],
        "log_target": metrics["log_target"],
        "total_relations": metrics["total_relations"],
        "sorted_relation_counts": {key: value for key, value in metrics.items()
                                    if key.startswith("sorted_relation_counts[")},
        "segmentation": {key: value for key, value in metrics.items()
                         if key in {"total_segments", "mean_segments_per_trace",
                                    "mean_segment_length", "sd_segment_length"}},
        "pi": {key: value for key, value in metrics.items()
               if key.startswith("sorted_pi[") or key in {"pi_entropy", "pi_l2"}},
        "P": {key: value for key, value in metrics.items()
              if key in {"P_frobenius", "P_trace2", "P_trace3"}
              or key.startswith("sorted_P_row_entropy[") or key.startswith("sorted_stationary[")},
        "worst_invariant": arm["worst_invariant"], "probe_summaries": arm["probe_summaries"],
    }
    base_chains = baseline.get("chains") if baseline else [None] * 4
    out["basins"] = [_basin_summary(chain, base_chains[index])
                     for index, chain in enumerate(arm["chains"])]
    return out


def _trend_value(previous: float, current: float) -> dict:
    return {"previous": previous, "current": current,
            "difference": current - previous if math.isfinite(previous) and math.isfinite(current) else None,
            "factor": current / previous if previous not in (0.0, float("inf")) and math.isfinite(current) else None}


def _metric_comparison(previous: dict, current: dict) -> dict:
    """Exact registered-diagnostic comparison, retaining current chain means."""
    return {
        "rhat": _trend_value(float(previous["rhat"]), float(current["rhat"])),
        "bulk_ess": _trend_value(float(previous["bulk_ess"]), float(current["bulk_ess"])),
        "tail_ess": _trend_value(float(previous["tail_ess"]), float(current["tail_ess"])),
        "previous_per_chain_posterior_mean": [
            float(row["posterior_mean"]) for row in previous["per_chain"]
        ],
        "current_per_chain_posterior_mean": [
            float(row["posterior_mean"]) for row in current["per_chain"]
        ],
        "current_per_chain_recent_mean": [
            float(row["recent_mean"]) for row in current["per_chain"]
        ],
    }


def _family_comparison(previous: dict, current: dict, labels: list[str]) -> dict:
    return {label: _metric_comparison(previous["metrics"][label], current["metrics"][label])
            for label in labels}


def _probe_summary_comparison(previous: dict, current: dict) -> dict:
    fields = ("median_rhat", "max_rhat", "fraction_at_registered_rhat_threshold",
              "constant_but_different")
    return {
        field: _trend_value(float(previous[field]), float(current[field]))
        for field in fields
    } | {
        "previous_at_registered_rhat_threshold": previous["at_registered_rhat_threshold"],
        "current_at_registered_rhat_threshold": current["at_registered_rhat_threshold"],
        "current_top5_labels": [row["label"] for row in current["top5"]],
    }


def _trend(current: dict, previous: dict | None) -> dict | None:
    if previous is None:
        return None
    output = {}
    for arm in ARMS:
        old, now = previous[arm], current[arm]
        old_worst, now_worst = old["worst_invariant"], now["worst_invariant"]
        output[arm] = {
            "log_target_rhat": _trend_value(float(old["metrics"]["log_target"]["rhat"]),
                                              float(now["metrics"]["log_target"]["rhat"])),
            "total_relations_rhat": _trend_value(float(old["metrics"]["total_relations"]["rhat"]),
                                                   float(now["metrics"]["total_relations"]["rhat"])),
            "max_invariant_rhat": _trend_value(float(old_worst["rhat"]), float(now_worst["rhat"])),
            "previous_worst_label": old_worst["label"],
            "current_worst_label": now_worst["label"],
            "worst_identity_changed": old_worst["label"] != now_worst["label"],
            "structural_library": _family_comparison(
                old, now, ["total_relations"] + sorted(
                    key for key in now["metrics"] if key.startswith("sorted_relation_counts[")
                )),
            "segmentation": _family_comparison(
                old, now, ["total_segments", "mean_segments_per_trace",
                           "mean_segment_length", "sd_segment_length"]),
            "pi": _family_comparison(
                old, now, sorted(key for key in now["metrics"]
                                 if key.startswith("sorted_pi[")
                                 or key in {"pi_entropy", "pi_l2"})),
            "P": _family_comparison(
                old, now, sorted(key for key in now["metrics"]
                                 if key in {"P_frobenius", "P_trace2", "P_trace3"}
                                 or key.startswith("sorted_P_row_entropy[")
                                 or key.startswith("sorted_stationary["))),
            "probe_summaries": {
                key: _probe_summary_comparison(old["probe_summaries"][key],
                                               now["probe_summaries"][key])
                for key in ("boundary", "coskill", "same_segment")
            },
        }
    return output


def _validate_baseline(raw: dict, current: dict, launch: dict) -> dict:
    """Reject a baseline from a different registered run or summary definition."""
    if raw.get("event") != "read_only_interim_trend_diagnostic":
        raise ValueError("baseline is not an interim diagnostic artifact")
    if raw.get("launch_source_commit") != launch["source_commit"]:
        raise ValueError("baseline launch source commit differs from current formal launch")
    previous = raw.get("_internal_arms")
    if not isinstance(previous, dict) or set(previous) != set(ARMS):
        raise ValueError("baseline does not contain both internal arm snapshots")
    for arm in ARMS:
        old, now = previous[arm], current[arm]
        if set(old["metrics"]) != set(now["metrics"]):
            raise ValueError(f"baseline/current registered-summary schema differs for {arm}")
        for index, (old_chain, now_chain) in enumerate(zip(old["chains"], now["chains"], strict=True)):
            old_meta, now_meta = old_chain["meta"], now_chain["meta"]
            for key in ("seed", "burn_in", "thin", "probes"):
                if old_meta.get(key) != now_meta.get(key):
                    raise ValueError(f"baseline/current {key} differs for {arm} chain {index}")
    return {
        "validated": True,
        "baseline_source_commit": raw["launch_source_commit"],
        "baseline_iterations": {arm: previous[arm]["iteration"] for arm in ARMS},
        "checks": ["source_commit", "summary_schema", "chain_seeds", "burn_in", "thin", "probes"],
    }


def _markdown(report: dict) -> str:
    lines = [
        "# Exploratory FULL-LATENT interim trend diagnostic",
        "",
        "**Nonformal / truth-free / read-only.** The registered 30k gate remains authoritative; "
        "this artifact neither writes a gate nor changes the running experiment.",
        "",
        f"Snapshot UTC: `{report['timestamp_utc']}`.",
        "",
        "## Current registered-summary diagnostics",
        "",
        "| Arm | Sweep | Retained/chain | log-target R-hat | total-relations R-hat | max invariant R-hat | worst invariant |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for arm in ARMS:
        value = report["arms"][arm]
        worst = value["worst_invariant"]
        lines.append("| {arm} | {sweep} | {draws} | {log:.4g} | {rel:.4g} | {maxr} | `{worst}` |".format(
            arm=arm, sweep=value["iteration"], draws=value["retained_draws_per_chain"][0],
            log=float(value["log_target"]["rhat"]), rel=float(value["total_relations"]["rhat"]),
            maxr=value["worst_invariant"]["rhat"], worst=worst["label"]))
        lines += ["", f"### {arm}: worst invariant", "",
                  f"`{worst['label']}`; type: **{worst['disagreement']}**.", "",
                  "| Chain | posterior mean | recent 2k-sweep mean | recent SD | unique values |",
                  "|---:|---:|---:|---:|---:|"]
        for index, row in enumerate(worst["per_chain"]):
            lines.append("| {i} | {mean:.6g} | {recent:.6g} | {sd:.6g} | {unique} |".format(
                i=index, mean=row["posterior_mean"], recent=row["recent_mean"],
                sd=row["recent_sd"], unique=row["unique_values"]))
        if worst.get("probe"):
            lines.append("")
            lines.append(f"Probe location: `{worst['probe']}`.")
    if report.get("trend"):
        lines += ["", "## Previous vs current", "",
                  "| Arm | metric | previous | current | factor |",
                  "|---|---|---:|---:|---:|"]
        for arm, values in report["trend"].items():
            for label in ("log_target_rhat", "total_relations_rhat", "max_invariant_rhat"):
                row = values[label]
                lines.append(f"| {arm} | {label} | {row['previous']} | {row['current']} | {row['factor']} |")
            lines.append(f"| {arm} | worst invariant | `{values['previous_worst_label']}` | "
                         f"`{values['current_worst_label']}` | changed={values['worst_identity_changed']} |")
    lines += ["", "## Scope", "",
              "Only registered, permutation-invariant checkpoint summaries were read.  No raw skill-indexed "
              "trace is interpreted, and this diagnostic opened no synthetic truth or held-out recovery.  "
              "No formal source, running process, checkpoint, threshold, seed, scale, cadence, or datum was modified.",
              "",
              "This audit attests only to its own truth-free scope.  It does not make a global experiment "
              "truth-seal claim; a separately recorded mid-run unseal event exists and was not opened by this audit."
              if MIDRUN_UNSEAL_RECORD.exists() else
              "No separately recorded mid-run unseal artifact was present when this audit was written.", ""]
    return "\n".join(lines)


def _output_payload(arms: dict, baseline: dict | None, baseline_validation: dict | None) -> dict:
    launch = json.loads(LAUNCH_MANIFEST.read_text())
    compact = {arm: _compact_arm(arms[arm], baseline[arm] if baseline else None) for arm in ARMS}
    return {
        "event": "read_only_interim_trend_diagnostic",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "formal_status": "EXPLORATORY pre-30k only; no formal PASS/FAIL is claimed",
        "registered_gate_unchanged": True,
        "launch_source_commit": launch["source_commit"],
        "script_sha256": _sha256(Path(__file__)),
        "diagnostic_truth_scope": (
            "PASS: this diagnostic imported no known truth/generator module and did not inspect "
            "truth or held-out recovery artifacts"
        ),
        "global_experiment_truth_seal": (
            "NOT ATTESTED: a separately recorded mid-run unseal artifact exists; its contents were not opened "
            "by this diagnostic"
            if MIDRUN_UNSEAL_RECORD.exists() else
            "No separately recorded mid-run unseal artifact was present; this is not an independent global seal proof"
        ),
        "run_state_modified": False,
        "formal_source_modified": False,
        "checkpoint_write": False,
        "arms": compact,
        "trend": _trend(arms, baseline),
        "baseline_validation": baseline_validation,
        "registered_rhat_threshold": RHAT_THRESHOLD,
        "recent_window_sweeps": RECENT_SWEEPS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="previous JSON snapshot made by this tool")
    parser.add_argument("--write-baseline", action="store_true",
                        help="save the current snapshot as a baseline only")
    parser.add_argument("--require-marg-sweep", type=int, default=0,
                        help="refuse a full trend report below this common MARG checkpoint")
    args = parser.parse_args()
    arms = {arm: _read_arm(arm) for arm in ARMS}
    if args.require_marg_sweep and arms["FULL-MARG"]["iteration"] < args.require_marg_sweep:
        raise RuntimeError("FULL-MARG durable checkpoint is below requested threshold: "
                           f"{arms['FULL-MARG']['iteration']} < {args.require_marg_sweep}")
    baseline = None
    baseline_validation = None
    if args.baseline:
        raw = json.loads(args.baseline.read_text())
        # The complete internal snapshot is intentionally retained only in baseline files.
        baseline = raw["_internal_arms"]
        baseline_validation = _validate_baseline(
            raw, arms, json.loads(LAUNCH_MANIFEST.read_text()))
    payload = _output_payload(arms, baseline, baseline_validation)
    if args.write_baseline:
        payload["kind"] = "baseline_snapshot_for_future_read_only_trend_comparison"
        payload["_internal_arms"] = arms
        name = "baseline_pre30k_cond{c}_marg{m}.json".format(
            c=arms["FULL-COND"]["iteration"], m=arms["FULL-MARG"]["iteration"])
        target = AUDIT_DIR / name
        target.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n")
        print(target)
        return 0
    payload["kind"] = "interim_trend_report"
    marg_iteration = arms["FULL-MARG"]["iteration"]
    json_path = AUDIT_DIR / f"interim_trend_pre30k_marg{marg_iteration}.json"
    md_path = AUDIT_DIR / f"interim_trend_pre30k_marg{marg_iteration}.md"
    json_path.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n")
    md_path.write_text(_markdown(payload))
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
