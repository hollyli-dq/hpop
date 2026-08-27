#!/usr/bin/env python3
"""Aggregate the 360 pilot jobs and apply the frozen rule, in the registered order.

    1. verify the manifest is complete: no missing job, no duplicate, no failure
    2. pool the four chains of each (replicate, K, X, u_scale) cell and compute the
       registered truth-free mixing diagnostics
    3. evaluate X in the registered order 50 -> 100 -> 166.7 and take the FIRST for
       which every rung has a u_scale passing in BOTH replicates independently
    4. within that X, choose each rung's u_scale from those scales by the frozen
       hardware-independent tie-break

Diagnostics are computed within `(replicate, K, X, u_scale)` over that cell's four
chains and are **never** pooled across replicates: the two replicates have different
master truths and corpora, so an eight-chain R-hat would be comparing two posteriors and
measuring nothing. Both replicates must pass for a scale to count.

The order matters. `X` is chosen first and globally, so no rung can buy itself a larger
budget; `u_scale` is chosen second and per rung, because it is proposal efficiency matched
to that rung's geometry rather than a grant of extra compute.

Runtime is reported but is **not** an input. Hardware feasibility is a separate gate
applied after the statistical choice: preferring a cheaper `X` because the machine is
small would let the hardware pick the experiment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location(
    "u_budget_pilot", Path(__file__).parent / "u_budget_pilot.py")
_pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pilot)

from hpop.mcmc_original.stage6b_mcmc_diagnostics import (                # noqa: E402
    bulk_ess, rank_normalized_split_rhat)


def integrity(manifest: dict) -> dict:
    """Every blocking condition, checked before a single number is interpreted."""
    problems, records, missing, failed = [], {}, [], []
    seen_hashes = defaultdict(list)
    for job in manifest["jobs"]:
        path = Path(job["output_path"])
        if not path.exists():
            missing.append(job["key"])
            continue
        record = json.loads(path.read_text())
        if record.get("status") != "complete":
            failed.append(job["key"])
            continue
        if record["job"]["job_hash"] != job["job_hash"]:
            problems.append(f"{job['key']}: job hash mismatch (stale output)")
        if record["job"]["library_digest_at_K"] != job["library_digest_at_K"]:
            problems.append(f"{job['key']}: corpus/library digest mismatch")
        if record.get("truth_used"):
            problems.append(f"{job['key']}: output claims the truth was used")
        if record["N_U"] != record["N_U_expected"]:
            problems.append(f"{job['key']}: N_U {record['N_U']} != quota "
                            f"{record['N_U_expected']}")
        seen_hashes[job["job_hash"]].append(job["key"])
        records[job["key"]] = record
    for h, keys in seen_hashes.items():
        if len(keys) > 1:
            problems.append(f"duplicate job hash {h}: {keys}")

    # Provenance must be homogeneous. A summary assembled from outputs produced by two
    # code versions, or from two RNG roots, is not one experiment -- and the mismatch is
    # invisible in the numbers themselves, which is exactly why it is checked here.
    for field, label in (("code_tag", "code tag"), ("crn_root", "pilot RNG root")):
        values = {r["job"].get(field) for r in records.values()}
        if len(values) > 1:
            problems.append(f"mixed {label} across completed jobs: {sorted(map(str, values))}")
    # The commit is taken from the working tree that produced each output, not from the
    # manifest, so this catches an operator who checked out the wrong thing.
    commits = {r.get("runtime_commit") for r in records.values()}
    if len(commits) > 1:
        problems.append(f"mixed RUNTIME code commit across jobs: "
                        f"{sorted(map(str, commits))}")
    dirty = sorted(k for k, r in records.items() if r.get("runtime_tree_dirty"))
    if dirty:
        problems.append(f"{len(dirty)} job(s) ran from a DIRTY working tree, e.g. "
                        f"{dirty[:3]}")
    roots = {r["job"].get("crn_root") for r in records.values()}
    if 6_500_000 in roots:
        problems.append("a pilot job used the PRODUCTION RNG root 6500000")

    rungs = {j["K"] for j in manifest["jobs"]}
    have = {records[k]["job"]["K"] for k in records}
    for k in sorted(rungs - have):
        problems.append(f"rung K={k} has no completed job")
    return {"records": records, "missing": missing, "failed": failed,
            "problems": problems,
            "blocking": bool(missing or failed or problems)}


def cell_diagnostics(chain_records: list) -> dict:
    per_chain = []
    for r in chain_records:
        bits = np.array(r["closure_bits"], dtype=float)
        if bits.shape[0] < 2:
            per_chain.append({"insufficient": True})
            continue
        changes = (bits[1:] != bits[:-1]).sum(axis=1)
        per_chain.append({
            "draws": int(bits.shape[0]), "relation_count": bits.sum(axis=1),
            "edge_indicators": bits,
            "closure_changed_between_draws": int((changes > 0).sum()),
            "relation_changes_per_change": (float(changes[changes > 0].mean())
                                            if (changes > 0).any() else 0.0),
            "unique_closures_visited": len({b.tobytes() for b in bits.astype(bool)}),
            "insufficient": False})
    pooled = _pilot.pooled_diagnostics(per_chain)

    accepted = sum(r["u_accepted_retained"] for r in chain_records)
    proposed = sum(r["u_proposed_retained"] for r in chain_records)
    seconds = sum(r["seconds"] for r in chain_records)
    changing = sum(c["closure_changed_between_draws"] for c in per_chain
                   if not c.get("insufficient"))
    usable = [c for c in per_chain if not c.get("insufficient")]
    return {
        **pooled,
        "u_acceptance_retained": (accepted / proposed) if proposed else None,
        "u_acceptance_burnin": (
            sum(r["u_accepted_burnin"] for r in chain_records)
            / max(sum(r["u_proposed_burnin"] for r in chain_records), 1)),
        "closure_changing_transitions": changing,
        "closure_changing_fraction": (changing / accepted) if accepted else None,
        "relation_changes_per_change": float(np.mean(
            [c["relation_changes_per_change"] for c in usable] or [0.0])),
        "unique_closures_visited": int(sum(c["unique_closures_visited"]
                                           for c in usable)),
        "effective_order_changing_moves_per_second": (changing / seconds
                                                      if seconds else 0.0),
        "seconds_total": seconds,
        "peak_rss_gib_max": max(r["peak_rss_gib"] for r in chain_records),
        "chains": len(chain_records),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path,
                   default=ROOT / "results" / "k_ladder_pilot" / "pilot_manifest.json")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "k_ladder_pilot" / "pilot_summary.json")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="report progress only; never used to choose X")
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    check = integrity(manifest)
    print(f"jobs in manifest: {manifest['n_jobs']}   completed: {len(check['records'])}"
          f"   missing: {len(check['missing'])}   failed: {len(check['failed'])}")
    for problem in check["problems"][:20]:
        print(f"  BLOCKING  {problem}")

    if check["blocking"] and not args.allow_incomplete:
        print("\nRESULT: BLOCKED -- the factorial is incomplete or inconsistent. "
              "No X may be\n        chosen from a partial pilot: the missing cells are "
              "not missing at random\n        once any of them was allowed to influence "
              "the decision to stop.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"status": "BLOCKED", **{
            k: v for k, v in check.items() if k != "records"}}, indent=2, default=str))
        return 1

    cells = defaultdict(list)
    for record in check["records"].values():
        job = record["job"]
        cells[(job["replicate"], job["K"],
               job["production_target_u_attempts_per_role"],
               job["u_scale"])].append(record)

    evaluated = []
    for (rep, k, target, scale), chain_records in sorted(cells.items()):
        diag = cell_diagnostics(chain_records)
        cell = {"replicate": rep, "K": k, "X": target, "u_scale": scale, **diag}
        cell["pass_rule_verdict"] = _pilot.evaluate_pass_rule(cell)
        evaluated.append(cell)

    print(f"\n{'rep':>4} {'K':>4} {'X':>7} {'scale':>6} {'acc':>6} {'Rhat|rel|':>10} "
          f"{'Rhat edge':>10} {'ESS min':>8} {'consens':>8} {'split':>6} "
          f"{'ordmv/s':>8} {'pass':>5}")
    print("-" * 100)
    for c in evaluated:
        f = lambda v, w, p=3: (f"{v:>{w}.{p}f}" if isinstance(v, float)
                               and np.isfinite(v) else f"{'-':>{w}}")
        print(f"{c['replicate']:>4} {c['K']:>4} {c['X']:>7.1f} {c['u_scale']:>6.2f} "
              f"{f(c['u_acceptance_retained'],6)} {f(c['rhat_relation_count'],10)} "
              f"{f(c.get('rhat_edge_max'),10)} {f(c.get('relation_level_ess_min'),8,1)} "
              f"{c.get('edges_consensus_fixed','-'):>8} "
              f"{c.get('edges_chain_disagreeing_frozen','-'):>6} "
              f"{c['effective_order_changing_moves_per_second']:>8.3f} "
              f"{'yes' if c['pass_rule_verdict']['passes'] else 'NO':>5}")

    decision = {"global_X": None, "per_rung_u_scale": {}, "reason": None,
                "evaluation_order": sorted({c["X"] for c in evaluated}),
                "replicates_pooled_for_diagnostics": False}
    if not check["blocking"]:
        rungs = sorted({c["K"] for c in evaluated})
        replicates = sorted({c["replicate"] for c in evaluated})
        trace = []
        for target in sorted({c["X"] for c in evaluated}):    # 50 -> 100 -> 166.7
            failing = []
            for k in rungs:
                by_scale = defaultdict(list)
                for c in evaluated:
                    if c["K"] == k and c["X"] == target:
                        by_scale[c["u_scale"]].append(c)
                # a scale counts only if EVERY replicate passes it independently
                ok = [sc for sc, cells in by_scale.items()
                      if len(cells) == len(replicates)
                      and all(x["pass_rule_verdict"]["passes"] for x in cells)]
                if not ok:
                    failing.append(k)
            trace.append({"X": target, "rungs_without_a_scale_passing_both_replicates":
                          failing})
            if not failing:
                decision["global_X"] = target
                break
        decision["evaluation_trace"] = trace
        if decision["global_X"] is None:
            decision["reason"] = (
                "no candidate X had a u_scale passing in BOTH replicates at every rung. "
                "By the frozen rule this pilot terminates unsuccessfully; nothing is "
                "re-run on these streams, and any kernel revision requires a new "
                "preregistered pilot on fresh streams.")
            decision["terminates_unsuccessfully"] = True
        else:
            decision["reason"] = ("first X, in registered order, with a u_scale passing "
                                  "in both replicates at every rung")
            for k in rungs:
                by_scale = defaultdict(list)
                for c in evaluated:
                    if c["K"] == k and c["X"] == decision["global_X"]:
                        by_scale[c["u_scale"]].append(c)
                decision["per_rung_u_scale"][str(k)] = _pilot.select_scale(
                    dict(by_scale), n_replicates=len(replicates))
        print(f"\nevaluation order: {decision['evaluation_order']}")
        for step in decision.get("evaluation_trace", []):
            miss = step["rungs_without_a_scale_passing_both_replicates"]
            print(f"  X={step['X']:>6.1f}  "
                  + ("SELECTED" if not miss else f"rejected; rungs failing: {miss}"))
        print(f"\nGLOBAL X: {decision['global_X']}   ({decision['reason']})")
        for k, sel in decision["per_rung_u_scale"].items():
            print(f"  K={k:>3}  u_scale = {sel['selected_u_scale']}  ({sel['reason']})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema": "k-ladder-learned-pilot-summary/1.0.0", "namespace": "PILOT",
        "status": "BLOCKED" if check["blocking"] else "complete",
        "manifest_jobs": manifest["n_jobs"], "completed": len(check["records"]),
        "missing": check["missing"], "failed": check["failed"],
        "problems": check["problems"], "cells": evaluated,
        "pass_rule": _pilot.PASS_RULE, "ess_tolerance": _pilot.ESS_TOLERANCE,
        "decision": decision,
        "runtime_excluded_from_decision": True,
        "replicates_pooled_for_diagnostics": False,
        "diagnostic_grouping": "(replicate, K, X, u_scale) -> 4 chains; never 8",
    }, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
