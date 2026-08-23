"""The unattended scalability benchmark driver: a resumable state machine.

    python scripts/scalability/run_autopilot.py --deadline-epoch <unix-seconds>

One task is one `(configuration, group)` pair and one subprocess. Tasks run strictly
sequentially -- never two at once -- because the study measures single-process timing on a
machine whose ambient load is already the largest source of variance.

## State

`state.json` is rewritten atomically after every task transition, so an interruption at any
instant leaves either the previous complete state or the next one, never a torn file. A
resume reads it, verifies the plan digest matches, and skips every task already recorded as
`ok`, `skipped_*` or `failed` with its retries exhausted.

`events.jsonl` is append-only and flushed per line: it is the audit trail, and it is what
survives even if the process is killed between state writes.

`progress.md` is rewritten after every task as a human-readable heartbeat.

## Refusals, and why each one is a refusal rather than an attempt

*Memory preflight* -- the worker predicts its own array bytes from exact shapes before
allocating, and refuses above `min(6 GB, 50% of physical RAM)` or above 80% of what the
kernel currently reports reclaimable. The driver records the refusal as a censored point.

*Monotone axis skip* -- when a point on an ordered axis is refused for memory or is
censored by timeout, every larger point on that same axis is skipped without being
attempted. Section 15 requires this, and it is also the only way one bad point cannot eat
the remaining budget.

*Conditional points* -- `J = 1024` and `K = 80` run only when their registered predecessor
finished inside the timeout and below 40% of physical RAM.

*Global deadline* -- benchmarking stops at the deadline the caller passes, whatever is left
in the queue. Everything unstarted is recorded as `skipped_deadline`.

Retries are bounded at two per task, and a task that fails twice is recorded and left.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_common as bc                                              # noqa: E402
import bench_plan as bp                                                # noqa: E402

MAX_ATTEMPTS = 2
WORKER = Path(__file__).resolve().parent / "bench_worker.py"

# Axes whose points are ordered by cost, so that refusing one refuses everything above it.
MONOTONE_AXES = {"J": lambda c: c.J, "K": lambda c: c.K, "N": lambda c: c.N,
                 "D": lambda c: c.D_max, "A_full": lambda c: c.A,
                 "A_sparse": lambda c: c.A}


class Autopilot:
    def __init__(self, out_dir: Path, deadline: float, resume: bool = True,
                 phase: str = "main"):
        self.out = Path(out_dir)
        # Each phase writes into its own raw directory. Re-measuring a configuration
                # must never overwrite the pass that measured it before: the whole point of
        # a second pass is to be able to compare the two.
        self.raw = self.out / ("raw" if phase == "main" else f"raw_{phase}")
        self.deadline = float(deadline)
        self.phase = phase
        self.raw = self.out / ("raw" if phase == "main" else f"raw_{phase}")
        self.raw.mkdir(parents=True, exist_ok=True)
        suffix = "" if phase == "main" else f"_{phase}"
        # A separate phase gets its own state, events and heartbeat, so running it can
        # never disturb a completed run's record.
        self.state_path = self.out / f"state{suffix}.json"
        self.events_path = self.out / f"events{suffix}.jsonl"
        self.progress_path = self.out / f"progress{suffix}.md"

        if phase == "optional":
            self.configs = bp.optional_configs()
            self.tasks = bp.optional_tasks()
        elif phase == "quiet":
            # the registered plan again, on an idle machine, with the speed probe
            self.configs = bp.full_plan()
            self.tasks = bp.tasks_for(self.configs)
        else:
            self.configs = bp.full_plan()
            self.tasks = bp.tasks_for(self.configs)
        self.by_label = {c.label: c for c in self.configs}
        self.digest = bp.plan_digest(self.configs)
        self.state = self._load(resume)

    # -------------------------------------------------------------------------- state
    def _fresh(self) -> dict:
        return {
            "plan_digest": self.digest,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "started_at_epoch": time.time(),
            "deadline_epoch": self.deadline,
            "worker": str(WORKER),
            "bench_seed": bc.BENCH_SEED,
            "tasks": {t["task_id"]: {"status": "pending", "attempts": 0}
                      for t in self.tasks},
            "decisions": [],
            "runs": 0,
        }

    def _load(self, resume: bool) -> dict:
        if resume and self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text())
            except json.JSONDecodeError:
                state = None
            if state and state.get("plan_digest") == self.digest:
                state["deadline_epoch"] = self.deadline
                state["runs"] = int(state.get("runs", 0)) + 1
                state["resumed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                        time.gmtime())
                # any task caught mid-flight by the interruption goes back to pending
                for record in state["tasks"].values():
                    if record.get("status") == "running":
                        record["status"] = "pending"
                for task in self.tasks:
                    state["tasks"].setdefault(task["task_id"],
                                              {"status": "pending", "attempts": 0})
                return state
            if state:
                self.event("plan_digest_mismatch", stored=state.get("plan_digest"),
                           current=self.digest,
                           action="starting a fresh state file; the previous one is "
                                  "left in place as state.superseded.json")
                bc.atomic_write(self.state_path.with_suffix(".superseded.json"),
                                json.dumps(state, indent=2, sort_keys=True))
        return self._fresh()

    def save(self) -> None:
        self.state["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        bc.atomic_write(self.state_path,
                        json.dumps(self.state, indent=2, sort_keys=True, default=float))

    def event(self, kind: str, **payload) -> None:
        line = json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "epoch": time.time(), "kind": kind, **payload},
                          sort_keys=True, default=float)
        with self.events_path.open("a") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def decide(self, what: str, why: str, **extra) -> None:
        """Record a non-fatal judgement call rather than asking about it."""
        self.state["decisions"].append({"what": what, "why": why,
                                        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                            time.gmtime()), **extra})
        self.event("decision", what=what, why=why, **extra)

    # ------------------------------------------------------------------------- gating
    def _refused_threshold(self, axis: str):
        """The smallest refused size on a monotone axis, or None."""
        getter = MONOTONE_AXES.get(axis)
        if getter is None:
            return None
        sizes = []
        for task in self.tasks:
            cfg = task["config"]
            if cfg.axis != axis:
                continue
            record = self.state["tasks"].get(task["task_id"], {})
            if record.get("status") in ("skipped_memory", "censored_timeout"):
                sizes.append(getter(cfg))
        return min(sizes) if sizes else None

    def _conditional_ok(self, cfg) -> tuple:
        rule = bp.CONDITIONAL.get(cfg.label)
        if rule is None:
            return True, ""
        required = rule["requires"]
        records = [r for tid, r in self.state["tasks"].items()
                   if tid.startswith(required + "::")]
        if not records or any(r.get("status") != "ok" for r in records):
            return False, (f"{required} did not complete cleanly on every group; "
                           f"{rule['why']}")
        peak = max(int(r.get("peak_rss_bytes", 0)) for r in records)
        physical = bc.physical_memory_bytes()
        if physical and peak > rule["max_rss_fraction_of_physical"] * physical:
            return False, (f"{required} peaked at {peak / 2**30:.2f} GiB, above "
                           f"{rule['max_rss_fraction_of_physical']:.0%} of physical RAM")
        if any(bool(r.get("censored")) for r in records):
            return False, f"{required} was censored by its timeout"
        swap = bc.swapping_now()
        if swap.get("available") and swap.get("used_mb", 0.0) > 8192.0:
            return False, "the machine is under heavy swap pressure"
        return True, ""

    # --------------------------------------------------------------------------- run
    def run_task(self, task) -> dict:
        cfg, group, task_id = task["config"], task["group"], task["task_id"]
        record = self.state["tasks"][task_id]

        allowed, why = self._conditional_ok(cfg)
        if not allowed:
            record.update(status="skipped_conditional", reason=why)
            self.event("skipped_conditional", task=task_id, reason=why)
            return record

        threshold = self._refused_threshold(cfg.axis)
        getter = MONOTONE_AXES.get(cfg.axis)
        if threshold is not None and getter is not None and getter(cfg) > threshold:
            why = (f"a smaller point on axis {cfg.axis} at size {threshold} was already "
                   f"refused, so every larger point on that axis is skipped unattempted")
            record.update(status="skipped_monotone", reason=why)
            self.event("skipped_monotone", task=task_id, reason=why)
            return record

        config_path = self.raw / f"{cfg.label}.config.json"
        bc.atomic_write(config_path, json.dumps(cfg.as_dict(), indent=2, sort_keys=True))
        out_path = self.raw / f"{cfg.label}__{group}.json"

        remaining = self.deadline - time.time()
        timeout = max(30.0, min(float(cfg.timeout_s), remaining))
        # the worker stops adding repetitions a little before the driver's hard kill, so
        # a slow point returns the repetitions it did manage instead of being lost
        worker_deadline = max(20.0, timeout - 25.0)

        record["status"] = "running"
        record["attempts"] = int(record.get("attempts", 0)) + 1
        record["loadavg_before"] = bc.load_average()
        self.save()
        self.event("task_start", task=task_id, attempt=record["attempts"],
                   timeout_s=timeout, loadavg=record["loadavg_before"],
                   config=cfg.as_dict())

        environment = dict(os.environ)
        environment.update({
            "PYTHONPATH": str(bc.ROOT / "src"),
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        })
        began = time.time()
        try:
            completed = subprocess.run(
                [sys.executable, str(WORKER), "--config", str(config_path),
                 "--group", group, "--out", str(out_path),
                 "--deadline-s", f"{worker_deadline:.1f}"],
                cwd=str(bc.ROOT), env=environment, capture_output=True, text=True,
                timeout=timeout)
            elapsed = time.time() - began
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip()[-2000:] or "nonzero exit")
            payload = json.loads(out_path.read_text())
            status = payload.get("status", "ok")
            if status == "skipped_memory_preflight":
                record.update(status="skipped_memory", seconds=elapsed,
                              reason="; ".join(payload["preflight"]["reasons"]),
                              result=str(out_path),
                              predicted_rss_bytes=payload["predicted_memory"][
                                  "predicted_process_rss_bytes"])
                self.event("skipped_memory", task=task_id, reason=record["reason"])
            else:
                record.update(status="ok", seconds=elapsed, result=str(out_path),
                              reps=payload.get("reps_completed", 0),
                              censored=bool(payload.get("censored")),
                              peak_rss_bytes=payload.get("peak_rss_bytes", 0),
                              loadavg_after=payload.get("loadavg_after"))
                self.event("task_ok", task=task_id, seconds=elapsed,
                           reps=record["reps"], censored=record["censored"],
                           peak_rss_mb=round(record["peak_rss_bytes"] / 1024 ** 2, 1))
        except subprocess.TimeoutExpired:
            elapsed = time.time() - began
            partial = out_path.exists()
            reps = 0
            if partial:
                try:
                    snapshot = json.loads(out_path.read_text())
                    reps = int(snapshot.get("reps_completed", 0))
                except (json.JSONDecodeError, OSError):
                    reps = 0
            record.update(status="censored_timeout", seconds=elapsed,
                          reason=f"exceeded the {timeout:.0f}s ceiling"
                                 + (f"; {reps} repetitions flushed before the kill"
                                    if reps else "; nothing was flushed before the kill"),
                          partial_result=bool(partial), reps=reps,
                          result=str(out_path) if partial else None)
            self.event("censored_timeout", task=task_id, seconds=elapsed,
                       partial_preserved=bool(partial), reps_flushed=reps)
        except Exception as error:                          # noqa: BLE001
            elapsed = time.time() - began
            if record["attempts"] >= MAX_ATTEMPTS:
                record.update(status="failed", seconds=elapsed, reason=str(error)[:2000])
                self.event("task_failed", task=task_id, error=str(error)[:2000],
                           attempts=record["attempts"])
            else:
                record.update(status="pending", last_error=str(error)[:2000])
                self.event("task_retry", task=task_id, error=str(error)[:2000],
                           attempts=record["attempts"])
        record["loadavg_after_driver"] = bc.load_average()
        return record

    # ---------------------------------------------------------------------- heartbeat
    def write_progress(self, current: str = "") -> None:
        counts: dict = {}
        for record in self.state["tasks"].values():
            counts[record["status"]] = counts.get(record["status"], 0) + 1
        total = len(self.state["tasks"])
        settled = sum(v for k, v in counts.items() if k not in ("pending", "running"))
        remaining = max(0.0, self.deadline - time.time())
        lines = [
            "# Scalability autopilot — progress",
            "",
            f"- backend under test: `optimized_segmental_v1` "
            f"(commit `564995efd056d7d33984f0ca1532386e6140ea0c`)",
            f"- updated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            f"- started: {self.state['started_at_utc']} (UTC)",
            f"- benchmarking deadline: "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.deadline))} "
            f"({remaining / 3600:.2f} h remaining)",
            f"- driver runs (including resumes): {self.state.get('runs', 0)}",
            f"- tasks settled: **{settled} / {total}**",
            f"- load average now: "
            f"{[round(v, 2) for v in bc.load_average()]}",
            "",
            "## Task status",
            "",
            "| status | count |",
            "| --- | --- |",
        ]
        for status in sorted(counts):
            lines.append(f"| {status} | {counts[status]} |")
        lines += ["", f"Currently running: `{current or '(between tasks)'}`", "",
                  "## Per-axis", "", "| axis | ok | skipped | censored | failed | pending |",
                  "| --- | --- | --- | --- | --- | --- |"]
        axes: dict = {}
        for task in self.tasks:
            axis = task["config"].axis
            status = self.state["tasks"][task["task_id"]]["status"]
            bucket = axes.setdefault(axis, {"ok": 0, "skipped": 0, "censored": 0,
                                            "failed": 0, "pending": 0})
            if status == "ok":
                bucket["ok"] += 1
            elif status.startswith("skipped"):
                bucket["skipped"] += 1
            elif status.startswith("censored"):
                bucket["censored"] += 1
            elif status == "failed":
                bucket["failed"] += 1
            else:
                bucket["pending"] += 1
        for axis in ("baseline", "J", "K", "N", "D", "A_full", "A_sparse", "target",
                     "target_long"):
            if axis in axes:
                b = axes[axis]
                lines.append(f"| {axis} | {b['ok']} | {b['skipped']} | {b['censored']} "
                             f"| {b['failed']} | {b['pending']} |")
        if self.state.get("decisions"):
            lines += ["", "## Recorded decisions", ""]
            for decision in self.state["decisions"][-30:]:
                lines.append(f"- **{decision['what']}** — {decision['why']}")
        bc.atomic_write(self.progress_path, "\n".join(lines) + "\n")

    # --------------------------------------------------------------------------- main
    def run(self) -> None:
        self.event("driver_start", deadline_epoch=self.deadline, pid=os.getpid(),
                   plan_digest=self.digest, n_tasks=len(self.tasks))
        suffix = "" if self.phase == "main" else f"_{self.phase}"
        bc.atomic_write(self.out / f"hardware_manifest{suffix}.json",
                        json.dumps(bc.hardware_manifest(), indent=2, sort_keys=True))
        bc.atomic_write(self.out / f"software_manifest{suffix}.json",
                        json.dumps(bc.software_manifest(), indent=2, sort_keys=True))
        self.save()
        self.write_progress()

        while True:
            pending = [t for t in self.tasks
                       if self.state["tasks"][t["task_id"]]["status"] == "pending"]
            if not pending:
                break
            if time.time() >= self.deadline:
                for task in pending:
                    self.state["tasks"][task["task_id"]].update(
                        status="skipped_deadline",
                        reason="the benchmarking budget ended before this task started")
                self.event("deadline_reached", skipped=len(pending))
                self.save()
                break
            task = pending[0]
            self.write_progress(current=task["task_id"])
            self.run_task(task)
            self.save()
            self.write_progress()

        self.state["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.state["finished_at_epoch"] = time.time()
        self.save()
        self.write_progress()
        self.event("driver_finished",
                   settled=sum(1 for r in self.state["tasks"].values()
                               if r["status"] not in ("pending", "running")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(bc.RESULTS))
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--phase", default="main",
                        choices=("main", "optional", "quiet"))
    args = parser.parse_args()
    Autopilot(Path(args.out), args.deadline_epoch, resume=not args.no_resume,
              phase=args.phase).run()


if __name__ == "__main__":
    main()
