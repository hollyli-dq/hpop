"""Resume the formal Condition C run after an orchestrator interruption.

Run:  PYTHONPATH=src .venv/bin/python scripts/resume_matched_condition_c_formal.py

This changes NO element of the frozen protocol. It imports the frozen runner and
uses its objects unmodified — same arms, seeds, starts, scales, cadence, gate
function, burn-in, thinning, ceiling and stopping rule — and only restores the
orchestration state that died with the parent process:

* every chain resumes from its own checkpoint, whose RNG bit-generator state is
  stored, so the continuation is draw-identical to an uninterrupted run
  (pinned by tests.mcmc_original.test_matched_condition_c);
* the ladder restarts at the first rung that has NOT already been evaluated.
  Re-entering at 30k would re-run the gate over draws retained up to the
  interruption and overwrite the recorded 30k and 50k gate files — the
  registered history must not be rewritten, so already-evaluated rungs are
  skipped;
* the two-consecutive-pass counters are reconstructed from those recorded gate
  files rather than reset, so the stopping rule sees the true history.

A resume manifest records the interruption and this reconstruction for the
protocol-deviation section of the terminal report.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

C_DIR = ROOT / "results" / "mcmc_original" / "matched_condition_c"
CHAINS = C_DIR / "formal_chains"


def load_runner():
    """Import the frozen runner as a normal module.

    It must be importable BY NAME, not loaded from a file spec: the worker
    processes are spawned, and unpickling `_advance_chain` in a child resolves
    it through `import run_matched_condition_c_formal`. A spec-loaded module is
    a different object and the identity check fails. The runner guards its own
    entry point with `if __name__ == "__main__"`, so importing it runs nothing.
    """
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_matched_condition_c_formal as module   # noqa: E402
    return module


def recorded_history(runner) -> list:
    """The gate outcomes already on disk, in ladder order."""
    history = []
    for checkpoint in runner.CHECKPOINTS:
        for arm in runner.ARMS:
            path = C_DIR / f"formal_gate_{arm}_{checkpoint}.json"
            if path.exists():
                payload = json.loads(path.read_text())
                history.append({"checkpoint": checkpoint, "arm": arm,
                                "pass": bool(payload["pass"]),
                                "checks": payload["checks"]})
    return history


def main() -> int:
    runner = load_runner()

    sweeps = {}
    for path in sorted(CHAINS.glob("*.npz")):
        meta = json.loads(str(np.load(str(path))["meta"]))
        sweeps[path.stem] = int(meta["iteration"])
    if len(sweeps) != 8:
        raise SystemExit(f"expected 8 chain checkpoints, found {len(sweeps)}")

    history = recorded_history(runner)
    evaluated = sorted({row["checkpoint"] for row in history})
    if not evaluated:
        raise SystemExit("no gate has been evaluated yet — start the run with "
                         "run_matched_condition_c_formal.py, not this resume")
    last_evaluated = evaluated[-1]

    # every chain must have reached the last evaluated rung, or the recorded
    # gate would not describe the state we are resuming from
    behind = {k: v for k, v in sweeps.items() if v < last_evaluated}
    if behind:
        raise SystemExit(f"chains behind the last evaluated rung "
                         f"{last_evaluated:,}: {behind}")

    remaining = tuple(c for c in runner.CHECKPOINTS if c > last_evaluated)
    if not remaining:
        raise SystemExit("the ladder is already complete; nothing to resume")

    # reconstruct the stopping-rule counters from the recorded history
    consecutive = {arm: 0 for arm in runner.ARMS}
    for row in history:
        consecutive[row["arm"]] = (consecutive[row["arm"]] + 1
                                   if row["pass"] else 0)
    already_stopped = [arm for arm, n in consecutive.items() if n >= 2]
    if already_stopped:
        raise SystemExit(f"arm(s) {already_stopped} already satisfied the "
                         "stopping rule; the run should have terminated")

    manifest = {
        "event": "orchestrator process interrupted; chains and gates intact",
        "chain_sweeps_at_resume": sweeps,
        "gates_already_evaluated": evaluated,
        "last_evaluated_rung": last_evaluated,
        "remaining_rungs": list(remaining),
        "reconstructed_consecutive_pass_counters": consecutive,
        "protocol_elements_changed": "none — arms, seeds, starts, scales, "
                                     "cadence, gate function, burn-in, "
                                     "thinning, ceiling and stopping rule are "
                                     "the frozen runner's own objects",
        "why_the_ladder_is_not_re_entered_at_the_first_rung":
            "re-running an already-evaluated rung would recompute its gate "
            "over the draws retained since, and overwrite the recorded result; "
            "the registered history is preserved instead",
        "continuation_fidelity": "each chain resumes from its stored RNG "
                                 "bit-generator state, so the continuation is "
                                 "draw-identical to an uninterrupted run",
        "compute_lost": "at most the sweeps since each chain's last checkpoint; "
                        "all eight were at exactly 50,000, a registered rung, "
                        "so no retained draw was lost",
    }
    (C_DIR / "resume_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"resuming: chains at {sorted(set(sweeps.values()))}, "
          f"rungs already evaluated {evaluated}, remaining {list(remaining)}, "
          f"consecutive passes {consecutive}", flush=True)

    # restrict the ladder to the un-evaluated rungs; every other protocol
    # element is the frozen runner's own. `main()` re-verifies the corpus hash
    # and the frozen scales on the way in, which is a useful integrity check on
    # resume; it also rewrites formal_registration.json, whose parent_commit
    # must keep describing the LAUNCH state, so that file is restored after.
    registration = C_DIR / "formal_registration.json"
    original = registration.read_text() if registration.exists() else None
    runner.CHECKPOINTS = remaining
    try:
        status = runner.main()
    finally:
        if original is not None:
            registration.write_text(original)

    # restore the complete ladder history in the convergence artifact
    path = C_DIR / "formal_convergence.json"
    if path.exists():
        payload = json.loads(path.read_text())
        payload["checkpoint_log"] = history + payload.get("checkpoint_log", [])
        payload["resumed"] = True
        payload["resume_manifest"] = "resume_manifest.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
