# Recovery-at-scale — runbook for the 8-machine fleet

This is the redesigned experiment: **end-to-end recovery of the partial-order library at
K = 3…30, run to convergence, with the cost reported.** It replaces every previous
K-ladder pilot. Read this whole file before launching; the design doc is
`RECOVERY_AT_SCALE_DESIGN.md` and the corpus calibration is
`CORPUS_CALIBRATION_MEMO.md`.

## What is different from the pilot you ran before

1. **The dataset is FROZEN and shipped.** `dataset/recovery_scale_v1/` contains every
   trace, hashed. Machines never generate data. `dataset/recovery_scale_v1/truth/` is
   SEALED — no worker reads it, and nobody opens it until the coordinator writes
   ALL_DONE.
2. **No fixed sweep count.** Chains run in 2,000-sweep segments until the truth-free
   convergence gates pass, or until the 100,000-sweep cap (cap hit = inference FAIL at
   that K — a result, not a problem to fix). Effort per K is an *outcome*.
3. **Rounds, not one long run.** Workers advance every live chain by one segment; then
   the coordinator (one machine) evaluates gates and emits the next round. Repeat.

## Preflight (every machine)

```bash
git clone https://github.com/hollyli-dq/hpop.git && cd hpop
git checkout <RECOVERY_TAG>          # Holly supplies the exact tag
cd handoff/recovery-scale-k30
pip install -r requirements.txt
python verify_environment.py         # must print RESULT: READY
git status --porcelain               # must print NOTHING (a dirty tree blocks analysis)
python - <<'EOF'                     # dataset integrity
import json, hashlib, pathlib
root = pathlib.Path("dataset/recovery_scale_v1")
m = json.loads((root/"dataset_manifest.json").read_text())
bad = [p for p, h in m["files_sha256"].items()
       if hashlib.sha256((root/p).read_bytes()).hexdigest() != h]
print("dataset:", "OK" if not bad else f"CORRUPT: {bad[:3]}")
EOF
```

## The round loop

One machine (pick one; call it the coordinator machine) runs:

```bash
python scripts/recovery_scale/coordinate.py
```

This writes `results/recovery_scale/round_jobs.json`. Then EVERY machine i = 0..7 runs
its slice, with several workers to use its cores (each worker is single-threaded):

```bash
# machine i, with W workers: sub-slice i, i+8, i+16, ... of 8*W
for s in $(seq $i 8 $((8*W-1))); do
  OMP_NUM_THREADS=1 PYTHONPATH=src nohup \
    python3 scripts/recovery_scale/run_recovery_job.py \
      --round-file results/recovery_scale/round_jobs.json --slice $s/$((8*W)) \
      > logs/round_slice$s.log 2>&1 &
done; wait
```

When all machines finish the round, the coordinator machine runs `coordinate.py` again.
Repeat until it prints **ALL_DONE**. If results/ is on shared NFS (as in the pilot), the
loop is just: coordinate → everyone runs → coordinate → …

Phase A (the first rounds) is the σ pilot — short, fixed length, selects the proposal
scale per rung by a frozen truth-free rule. Phase B is the run-to-convergence production.

## Safe operations

- **Interrupting is safe.** Checkpoints are atomic; re-running a round resumes exactly
  (bit-identical — this is tested, not hoped).
- **A crashed machine is safe.** Its slice simply hasn't advanced; re-run the same slice.
- **Do not**: edit any parameter, skip cells, rerun a segment "for a better draw" (it
  cannot give one — draws are a deterministic function of the design indices), or read
  anything under `dataset/*/truth/`.

## After ALL_DONE

Send back `results/recovery_scale/` (checkpoints, verdicts.json, coordinator_audit.jsonl,
sigma_selection.json) plus your logs. Holly runs:

```bash
python scripts/recovery_scale/evaluate_recovery.py     # opens truth, PASS cells only
```

## What every output records

Runtime git commit (read from the worktree, retried, refuses to run if unavailable),
hostname, seconds, peak RSS, U proposals/acceptances (burn-in irrelevant — the gate
windows the last half), per-skill H-changing accepted moves, per-role attempt counts.
The coordinator's audit log records every verdict and every continue decision with its
failing diagnostics, so the run's history is replayable from the archive alone.
