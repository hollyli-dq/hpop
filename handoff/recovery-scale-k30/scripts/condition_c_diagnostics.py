"""Condition C truth-free explanatory diagnostics. READ-ONLY, RECOVERY-SEALED.

Run:  PYTHONPATH=src .venv/bin/python scripts/condition_c_diagnostics.py [--snapshot LABEL]

Reads only the formal-chain checkpoint files with numpy/json. It imports no
generator, no truth object and no recovery code, so it cannot compute an oracle
quantity even by accident: every comparison is chain-versus-chain.

It does NOT replace the registered convergence verdict. The registered gate
(scripts/run_matched_condition_c_formal.py::arm_gate) is authoritative; this
module only explains a gate outcome in terms of

    structural-library movement   (which structures a chain holds)

versus

    anchored reassignment         (which identity each structure is attached to)

`--snapshot LABEL` writes the current cumulative counters so a later report can
difference a well-defined window rather than guessing one.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C_DIR = ROOT / "results" / "mcmc_original" / "matched_condition_c"
CHAINS = C_DIR / "formal_chains"
BASELINES = C_DIR / "diagnostic_baselines"

SEALED = ("recovery quantities (H*, S*, z*, relation/closure/boundary F1, ARI, "
          "occurrence accuracy, Hamming to truth) are NOT computed here and "
          "remain sealed until Condition C reaches its registered terminal "
          "condition")


def _read(path: Path) -> dict:
    data = np.load(str(path))
    meta = json.loads(str(data["meta"]))
    h = [tuple(str(v) for v in row) for row in data["h_hashes"]] \
        if data["h_hashes"].size else []
    n_traces = sum(1 for k in data.files if k.startswith("b"))
    return {
        "name": path.stem,
        "arm": "cond" if path.stem.startswith("cond") else "marg",
        "iteration": int(meta["iteration"]),
        "burn_in": int(meta["burn_in"]), "thin": int(meta["thin"]),
        "movement": meta["movement"],
        "collapsed": [int(v) for v in meta["collapsed"]],
        "seconds": float(meta["seconds"]),
        "log_target": np.asarray(data["log_target"], dtype=float),
        "rel_counts": np.asarray(data["rel_counts"], dtype=int),
        "h_hashes": h,
        "segments": [len(seg) for seg in meta["segmentations"]],
        "boundary_sums": [np.asarray(data[f"b{n:03d}"], dtype=float)
                          for n in range(n_traces)],
        "occupancy_sums": [np.asarray(data[f"o{n:03d}"], dtype=float)
                           for n in range(n_traces)],
        "marginal_draws": int(meta["marginal_draws"]),
    }


def _sweep_of_draw(chain: dict, index: int) -> int:
    return chain["burn_in"] + (index + 1) * chain["thin"]


def _last_change(chain: dict, key) -> dict:
    """Last retained draw at which `key(h_tuple)` changed, as a sweep index."""
    series = [key(h) for h in chain["h_hashes"]]
    last = None
    for i in range(1, len(series)):
        if series[i] != series[i - 1]:
            last = i
    return {"last_change_sweep": (_sweep_of_draw(chain, last)
                                  if last is not None else None),
            "sweeps_since_change": (chain["iteration"]
                                    - _sweep_of_draw(chain, last)
                                    if last is not None else None),
            "n_changes_in_retained": sum(
                1 for i in range(1, len(series)) if series[i] != series[i - 1]),
            "distinct_values": len(set(series)),
            "current": list(series[-1]) if series else None}


def chain_report(chain: dict, baseline: dict | None) -> dict:
    anchored = _last_change(chain, lambda h: tuple(h))
    library = _last_change(chain, lambda h: tuple(sorted(h)))
    out = {
        "chain": chain["name"], "arm": chain["arm"],
        "sweeps": chain["iteration"],
        "retained_draws": len(chain["log_target"]),
        "log_target_mean": float(chain["log_target"].mean())
        if chain["log_target"].size else None,
        "log_target_last": float(chain["log_target"][-1])
        if chain["log_target"].size else None,
        "relation_count_current": int(chain["rel_counts"][-1])
        if chain["rel_counts"].size else None,
        "segment_total_current": int(sum(chain["segments"])),
        "anchored_H_tuple": anchored,
        "unordered_H_library": library,
        "cumulative": {
            "u_proposed": chain["movement"]["u_proposed"],
            "u_accepted": chain["movement"]["u_accepted"],
            "u_accepted_H_changes": chain["movement"]["u_h_accepted"],
            "collapsed_proposed": chain["collapsed"][0],
            "collapsed_accepted": chain["collapsed"][1],
            "collapsed_accepted_cross_H": chain["collapsed"][2],
            "ffbs_boundary_hamming": chain["movement"]["boundary_hamming"],
            "ffbs_label_changes": chain["movement"]["label_changes"],
            "ffbs_states_changed": chain["movement"]["states_changed"],
        },
    }
    if baseline is not None:
        base = baseline.get(chain["name"])
        if base is not None:
            window = chain["iteration"] - base["iteration"]
            out["window"] = {
                "from_sweep": base["iteration"], "to_sweep": chain["iteration"],
                "sweeps": window,
                **{k: out["cumulative"][k] - base["cumulative"][k]
                   for k in out["cumulative"]},
            }
    return out


def arm_report(chains: list) -> dict:
    """Chain-versus-chain structure of one arm. No truth, no recovery."""
    anchored = [tuple(c["h_hashes"][-1]) for c in chains if c["h_hashes"]]
    libraries = [tuple(sorted(t)) for t in anchored]
    groups: dict = {}
    for c, t in zip(chains, anchored):
        groups.setdefault(t, []).append(c["name"])
    lib_groups: dict = {}
    for c, lib in zip(chains, libraries):
        lib_groups.setdefault(lib, []).append(c["name"])

    by_assignment = []
    for tup, names in groups.items():
        values = np.concatenate([c["log_target"] for c in chains
                                 if c["name"] in names])
        by_assignment.append({
            "anchored_tuple": [h[:6] for h in tup], "chains": names,
            "log_target_mean": float(values.mean()),
            "log_target_sd": float(values.std(ddof=1)) if values.size > 1
            else 0.0})
    by_assignment.sort(key=lambda r: -r["log_target_mean"])

    # cross-chain disagreement in the path marginals (co-clustering surrogate)
    occ_max = bnd_max = 0.0
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            for a, b in zip(chains[i]["occupancy_sums"],
                            chains[j]["occupancy_sums"]):
                occ_max = max(occ_max, float(np.abs(
                    a / chains[i]["marginal_draws"]
                    - b / chains[j]["marginal_draws"]).max()))
            for a, b in zip(chains[i]["boundary_sums"],
                            chains[j]["boundary_sums"]):
                bnd_max = max(bnd_max, float(np.abs(
                    a / chains[i]["marginal_draws"]
                    - b / chains[j]["marginal_draws"]).max()))
    segments = [int(sum(c["segments"])) for c in chains]
    return {
        "n_chains": len(chains),
        "distinct_anchored_assignments": len(groups),
        "distinct_unordered_libraries": len(lib_groups),
        "shares_one_unordered_library": len(lib_groups) == 1,
        "assignment_split": sorted((len(v) for v in groups.values()),
                                   reverse=True),
        "library_split": sorted((len(v) for v in lib_groups.values()),
                                reverse=True),
        "groups_by_anchored_assignment": by_assignment,
        "log_target_gap_between_top_two_assignments":
            (by_assignment[0]["log_target_mean"]
             - by_assignment[1]["log_target_mean"])
            if len(by_assignment) > 1 else None,
        "max_pairwise_occurrence_marginal_difference": occ_max,
        "max_pairwise_boundary_marginal_difference": bnd_max,
        "segment_totals": segments,
        "segment_total_spread": max(segments) - min(segments),
        "separation_statement": (
            "shares one unordered structural library; separated in the "
            "assignment of that library to anchored skill identities"
            if len(lib_groups) == 1 and len(groups) > 1 else
            "separated at the structural-library level"
            if len(lib_groups) > 1 else
            "chains agree on both the unordered library and its anchored "
            "assignment"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=None,
                        help="write the current counters as a baseline")
    parser.add_argument("--baseline", default=None,
                        help="difference against a stored baseline label")
    args = parser.parse_args()

    chains = [_read(Path(p)) for p in sorted(glob.glob(str(CHAINS / "*.npz")))]
    if not chains:
        raise SystemExit("no formal chain checkpoints found")

    if args.snapshot:
        BASELINES.mkdir(parents=True, exist_ok=True)
        payload = {c["name"]: {"iteration": c["iteration"],
                               "cumulative": chain_report(c, None)["cumulative"]}
                   for c in chains}
        path = BASELINES / f"{args.snapshot}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"baseline written: {path.relative_to(ROOT)} "
              f"(sweeps {[c['iteration'] for c in chains]})")
        return 0

    baseline = None
    if args.baseline:
        baseline = json.loads(
            (BASELINES / f"{args.baseline}.json").read_text())

    report = {"sealed_note": SEALED,
              "registered_gate_is_authoritative":
                  "scripts/run_matched_condition_c_formal.py::arm_gate",
              "per_chain": [chain_report(c, baseline) for c in chains],
              "per_arm": {arm: arm_report([c for c in chains
                                           if c["arm"] == arm])
                          for arm in ("cond", "marg")}}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
