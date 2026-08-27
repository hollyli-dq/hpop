"""Detach the autopilot from this shell and keep the machine awake while it runs.

    python scripts/scalability/launch.py --deadline-epoch <unix-seconds>

`caffeinate -dimsu` holds off display, idle, disk and system sleep for the lifetime of the
child it wraps, so a ten-hour unattended run is not cut in half by the lid closing.

`start_new_session=True` puts the child in its own session and process group. It is
therefore not in this shell's foreground process group, so a SIGHUP or SIGINT delivered to
the terminal never reaches it, and it is reparented to launchd when this process exits.
Its stdout and stderr go to files, not to a pipe, so nothing blocks when no one is reading.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--phase", default="main", choices=("main", "optional", "quiet"))
    parser.add_argument("--out", default=str(ROOT / "results" / "scalability"
                                             / "optimized_segmental_v1"))
    args = parser.parse_args()

    out = Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    stdout_path = out / "logs" / f"autopilot_{args.phase}_{stamp}.out"
    stderr_path = out / "logs" / f"autopilot_{args.phase}_{stamp}.err"

    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    })

    command = ["caffeinate", "-dimsu", sys.executable,
               str(HERE / "run_autopilot.py"),
               "--deadline-epoch", f"{args.deadline_epoch:.0f}",
               "--out", str(out), "--phase", args.phase]

    with stdout_path.open("wb") as so, stderr_path.open("wb") as se:
        child = subprocess.Popen(command, cwd=str(ROOT), env=environment,
                                 stdout=so, stderr=se, stdin=subprocess.DEVNULL,
                                 start_new_session=True)

    record = {
        "launched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "caffeinate_pid": child.pid,
        "command": command,
        "deadline_epoch": args.deadline_epoch,
        "phase": args.phase,
        "deadline_local": time.strftime("%Y-%m-%d %H:%M:%S",
                                        time.localtime(args.deadline_epoch)),
        "stdout": str(stdout_path), "stderr": str(stderr_path),
        "session_leader": True,
        "cwd": str(ROOT),
    }
    suffix = "" if args.phase == "main" else f"_{args.phase}"
    (out / f"launch{suffix}.json").write_text(json.dumps(record, indent=2,
                                                        sort_keys=True))
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
