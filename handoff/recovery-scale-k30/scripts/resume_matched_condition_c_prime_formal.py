"""Resume the formal Condition C' run after an orchestrator interruption.

Run:  PYTHONPATH=src .venv/bin/python scripts/resume_matched_condition_c_prime_formal.py

Changes NO element of the frozen protocol — same arms, seeds, starts, scales,
cadences, gate function, burn-in, thinning, ceiling and stopping rule, all the
frozen runner's own objects. It restores only the orchestration state that died
with the parent process, exactly as the validated Condition-C resume did:

* every chain resumes from its own checkpoint (RNG bit-state included), so the
  continuation is draw-identical to an uninterrupted run;
* the ladder restarts at the first rung NOT already evaluated — re-entering at
  30k would recompute that gate over all draws retained since and overwrite the
  registered artifact;
* the two-consecutive-pass counters are reconstructed from the recorded gate
  files.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

D = ROOT / "results" / "mcmc_original" / "matched_condition_c_prime"
CHAINS = D / "formal_chains"
PROTOCOL_ID = "condition-c-prime-v1"


def main() -> int:
    listing = subprocess.run(["ps", "ax", "-o", "command="],
                             capture_output=True, text=True).stdout
    if any("run_matched_condition_c_prime_formal" in l and "bash -c" not in l
           for l in listing.splitlines()):
        raise SystemExit("a C' orchestrator is already alive — not resuming")

    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_matched_condition_c_prime_formal as runner   # by NAME: spawn
                                                            # children unpickle
                                                            # _advance through it

    sweeps = {}
    for path in sorted(CHAINS.glob("*.npz")):
        meta = json.loads(str(np.load(str(path))["meta"]))
        sweeps[path.stem] = int(meta["iteration"])
    if len(sweeps) != 8:
        raise SystemExit(f"expected 8 chain checkpoints, found {len(sweeps)}")

    history, evaluated = [], set()
    for rung in runner.CHECKPOINTS:
        for arm in runner.ARMS:
            gate = D / f"formal_gate_{arm}_{rung}.json"
            if gate.exists():
                history.append({"rung": rung, "arm": arm,
                                "pass": bool(json.loads(
                                    gate.read_text())["pass"])})
                evaluated.add(rung)
    if not evaluated:
        raise SystemExit("no gate evaluated yet — use the launch command, "
                         "not this resume")
    last = max(evaluated)
    behind = {k: v for k, v in sweeps.items() if v < last}
    if behind:
        raise SystemExit(f"chains behind the last evaluated rung {last:,}: "
                         f"{behind}")
    remaining = tuple(c for c in runner.CHECKPOINTS if c > last)
    if not remaining:
        raise SystemExit("ladder already complete")

    consecutive = {arm: 0 for arm in runner.ARMS}
    for row in history:
        consecutive[row["arm"]] = (consecutive[row["arm"]] + 1
                                   if row["pass"] else 0)
    stopped = [a for a, n in consecutive.items() if n >= 2]
    if stopped:
        raise SystemExit(f"arm(s) {stopped} already satisfied the stopping "
                         "rule; the run should have terminated")

    manifest_path = D / "resume_manifest.json"
    prior = (json.loads(manifest_path.read_text())["interruptions"]
             if manifest_path.exists() else [])
    prior.append({
        "n": len(prior) + 1,
        "chain_sweeps_at_resume": sweeps,
        "gates_already_evaluated": sorted(evaluated),
        "remaining_rungs": list(remaining),
        "reconstructed_consecutive_pass_counters": consecutive,
        "observed_cause": "orchestrator and workers died overnight with no "
                          "Python traceback while the machine (on battery) "
                          "cycled maintenance sleeps; chains checkpointed "
                          "cleanly mid-segment, no partial writes",
        "protocol_elements_changed": "none",
        "compute_lost": "at most 2,000 sweeps per chain since the last "
                        "within-segment checkpoint",
    })
    manifest_path.write_text(json.dumps(
        {"interruptions": prior,
         "continuation_fidelity": "chains resume from stored RNG bit-state; "
                                  "draw-identical to uninterrupted execution",
         "history_preservation": "already-evaluated rungs are skipped so no "
                                 "registered gate is recomputed"},
        indent=2, sort_keys=True) + "\n")

    print(f"resuming C': chains at {sorted(set(sweeps.values()))}, evaluated "
          f"{sorted(evaluated)}, remaining {list(remaining)}, counters "
          f"{consecutive}", flush=True)

    # Guards and parity run with the FULL frozen ladder — restricting the
    # ladder first made the target manifest disagree with Condition C's
    # registration, and the frozen parity guard correctly refused to launch
    # (first resume attempt). Only after every check passes is the ladder
    # narrowed to the un-evaluated rungs for run_formal().
    status = runner.condition_c_status()
    runner.assert_may_launch(status)
    truth, corpus = runner.build_environment()
    sealed = runner.SealedTruth(truth)
    targets = runner.target_manifest(sealed, corpus)
    if not targets["all_parity_checks_pass"]:
        raise SystemExit(f"target parity failed on resume: "
                         f"{targets['parity_vs_condition_c']}")
    print("guards and full-ladder target parity PASS; narrowing ladder to "
          f"{list(remaining)}", flush=True)
    runner.CHECKPOINTS = remaining
    return runner.run_formal()


if __name__ == "__main__":
    raise SystemExit(main())
