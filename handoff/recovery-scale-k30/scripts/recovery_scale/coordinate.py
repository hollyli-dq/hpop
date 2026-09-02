#!/usr/bin/env python3
"""The coordinator: evaluates gates between rounds and emits the next round's jobs.

Runs on ONE machine (any of them) after every round. It never touches truth/.

    Phase A (sigma pilot): every rung x 3 scales x 2 replicates x 4 chains, ONE segment.
        After the round, sigma*_K is selected per rung by the frozen truth-free rule:
        among scales whose cells pass the sanity window in BOTH replicates, the scale
        maximising the worst-case relation-total ESS (worst-cased over replicates), ties
        toward the smaller scale. Selection is written to sigma_selection.json.

    Phase B (production): every rung x sigma*_K x 2 replicates x 4 chains. After each
        round the permutation-invariant gates are evaluated per (replicate, K) cell on
        the last half of all draws so far: PASS freezes the cell's verdict; a cell at
        CAP_SWEEPS without passing is INFERENCE FAIL; anything else continues.

    All verdicts, numbers and decisions are appended to an audit log. When every cell is
    PASS or FAIL the coordinator writes ALL_DONE and the fleet stops. Truth is opened
    only after that, by evaluate_recovery.py, on PASS cells alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_cpa.recovery_gates import (chain_statistics, cocluster_probe_pairs,  # noqa: E402
                                          evaluate_cell)
from hpop.mcmc_cpa.recovery_regime import REGIME                          # noqa: E402

SIGMAS = (0.25, 0.5, 1.0)


def cell_dir(work: Path, phase: str, replicate: int, k: int, sigma: float) -> Path:
    return work / phase / f"rep{replicate}_K{k}_s{sigma:g}"


def load_chain(cell: Path, chain: int) -> dict | None:
    files = sorted((cell / f"chain{chain}").glob("checkpoint_*.json"))
    if not files:
        return None
    draws = {"labels": [], "boundaries": [], "u": [], "u_event_sweep": [],
             "pi_sorted": [], "p_spectrum": []}
    state = None
    for f in files:
        payload = json.loads(f.read_text())
        for key in draws:
            draws[key].extend(payload["draws"].get(key, []))
        state = payload["state"]
    return {"draws": draws, "state": state}


def acceptance(chains: list) -> float | None:
    proposed = sum(c["state"]["u_proposed"] for c in chains)
    accepted = sum(c["state"]["u_accepted"] for c in chains)
    return (accepted / proposed) if proposed else None


def gate_cell(work, phase, replicate, k, sigma, trace_lengths) -> dict | None:
    cell = cell_dir(work, phase, replicate, k, sigma)
    chains = [load_chain(cell, c) for c in range(REGIME.CHAINS)]
    if any(c is None for c in chains):
        return None
    pairs = cocluster_probe_pairs(trace_lengths, 200,
                                  seed=REGIME.ROOT_ENTROPY + 17 * k + replicate)
    stats = [chain_statistics(c["draws"], pairs) for c in chains]
    verdict = evaluate_cell(stats, acceptance(chains))
    verdict["sweep"] = chains[0]["state"]["sweep"]
    verdict["u_proposed_total"] = sum(c["state"]["u_proposed"] for c in chains)
    verdict["proposals_per_row"] = verdict["u_proposed_total"] / (
        REGIME.CHAINS * k * 10)
    verdict["h_changing_per_skill"] = [
        sum(c["state"]["h_changing_accepted_per_skill"][s] for c in chains)
        for s in range(k)]
    return verdict


def trace_lengths_for(dataset: Path, replicate: int, k: int) -> list:
    payload = json.loads((dataset / "traces" / f"rep{replicate}_K{k}.json").read_text())
    return [len(t["cpa"]) for t in payload["train"]]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path,
                   default=ROOT / "dataset" / "recovery_scale_v1")
    p.add_argument("--work", type=Path, default=ROOT / "results" / "recovery_scale")
    args = p.parse_args()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    audit = work / "coordinator_audit.jsonl"

    def log(entry: dict) -> None:
        with audit.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    sigma_file = work / "sigma_selection.json"
    if not sigma_file.exists():
        # ---- phase A
        complete = all(
            load_chain(cell_dir(work, "phaseA", r, k, s), c) is not None
            for r in REGIME.REPLICATES for k in REGIME.K_LADDER
            for s in SIGMAS for c in range(REGIME.CHAINS))
        if not complete:
            jobs = [{"phase": "phaseA", "replicate": r, "K": k, "u_scale": s,
                     "chain": c, "segment_sweeps": REGIME.SEGMENT_SWEEPS}
                    for r in REGIME.REPLICATES for k in REGIME.K_LADDER
                    for s in SIGMAS for c in range(REGIME.CHAINS)]
            (work / "round_jobs.json").write_text(json.dumps(
                {"round": "phaseA", "jobs": jobs}, indent=1))
            print(f"PHASE A: {len(jobs)} pilot jobs -> {work/'round_jobs.json'}")
            return 0
        selection = {}
        for k in REGIME.K_LADDER:
            lengths = trace_lengths_for(args.dataset, 0, k)
            scores = {}
            for s in SIGMAS:
                cells = [gate_cell(work, "phaseA", r, k, s, lengths)
                         for r in REGIME.REPLICATES]
                accs = [c["numbers"].get("u_acceptance_window") for c in cells]
                lo, hi = REGIME.ACCEPT_WINDOW
                sane = all(a is not None and lo <= a <= hi for a in accs)
                ess = [c["numbers"].get("ess_relation_total") for c in cells]
                if sane and all(e is not None for e in ess):
                    scores[s] = min(ess)
            if not scores:
                selection[str(k)] = {"sigma": None,
                                     "reason": "no scale passed sanity in both replicates"}
                continue
            peak = max(scores.values())
            near = [s for s, v in scores.items() if v >= 0.9 * peak]
            selection[str(k)] = {"sigma": min(near), "scores": scores,
                                 "rule": "max worst-case relation-total ESS over "
                                         "replicates; ties within 10% to smaller sigma"}
        sigma_file.write_text(json.dumps(selection, indent=1))
        log({"event": "sigma_selected", "selection": selection})
        if any(v["sigma"] is None for v in selection.values()):
            print("SIGMA SELECTION FAILED at some rung -- experiment terminates; "
                  "see sigma_selection.json")
            return 1
        print("sigma selected per rung:",
              {k: v["sigma"] for k, v in selection.items()})
        # fall through to emit phase B round 1

    selection = json.loads(sigma_file.read_text())
    verdict_file = work / "verdicts.json"
    verdicts = json.loads(verdict_file.read_text()) if verdict_file.exists() else {}
    jobs, states = [], {"PASS": 0, "FAIL": 0, "RUN": 0, "NEW": 0}
    for r in REGIME.REPLICATES:
        for k in REGIME.K_LADDER:
            sigma = float(selection[str(k)]["sigma"])
            key = f"rep{r}_K{k}"
            if verdicts.get(key, {}).get("final") in ("PASS", "FAIL"):
                states[verdicts[key]["final"]] += 1
                continue
            lengths = trace_lengths_for(args.dataset, r, k)
            verdict = gate_cell(work, "phaseB", r, k, sigma, lengths)
            if verdict is None:
                states["NEW"] += 1
            elif verdict["passes"]:
                verdicts[key] = {"final": "PASS", **verdict}
                log({"event": "cell_pass", "cell": key, **verdict})
                states["PASS"] += 1
                continue
            elif verdict["sweep"] >= REGIME.CAP_SWEEPS:
                verdicts[key] = {"final": "FAIL",
                                 "meaning": "INFERENCE FAIL at this K -- not a model "
                                            "scaling claim", **verdict}
                log({"event": "cell_fail_at_cap", "cell": key, **verdict})
                states["FAIL"] += 1
                continue
            else:
                log({"event": "cell_continue", "cell": key,
                     "sweep": verdict["sweep"], "failures": verdict["failures"][:4]})
                states["RUN"] += 1
            for c in range(REGIME.CHAINS):
                jobs.append({"phase": "phaseB", "replicate": r, "K": k,
                             "u_scale": sigma, "chain": c,
                             "segment_sweeps": REGIME.SEGMENT_SWEEPS,
                             "cap_sweeps": REGIME.CAP_SWEEPS})
    verdict_file.write_text(json.dumps(verdicts, indent=1, default=str))
    print(f"cells: {states}")
    if not jobs:
        (work / "ALL_DONE").write_text(json.dumps(states))
        print("ALL_DONE -- every cell PASS or FAIL. Truth may now be opened with "
              "evaluate_recovery.py on PASS cells.")
        return 0
    (work / "round_jobs.json").write_text(json.dumps(
        {"round": "phaseB", "jobs": jobs}, indent=1))
    print(f"next round: {len(jobs)} jobs -> {work/'round_jobs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
