# Learned-order U-budget pilot — instructions for the four operators

You are running a **pilot**, not the production experiment. Its only purpose is to choose
two things: one global `U` proposal budget `X`, and one proposal scale `u_scale` per rung.
Nothing you produce is a scientific result, and nothing here should be quoted as one.

The whole job is **360 independent chains**, split four ways. Each machine runs 90 of them
and takes roughly **3 hours** on hardware comparable to an Apple M4. There is no
communication between machines.

---

## What each operator does

### 0. Get the code — the same code

```bash
git clone https://github.com/hollyli-dq/hpop.git
cd hpop
git checkout <PILOT_TAG>          # Holly will give you the exact tag
cd handoff/recovery-scale-k30
pip install -r requirements.txt
```

**All four machines must be on the same tag.** The collection step refuses to merge output
produced by different commits, and it is right to: two code versions is not one
experiment. Do not `git pull` mid-run.

### 1. Preflight — stop if any of this fails

```bash
python verify_environment.py            # must print: RESULT: READY
pytest tests -q                         # expect 1721+ passed, 63 skipped, 0 failed
```

`verify_environment.py` checks file integrity, the pinned Python/NumPy/SciPy versions, and
a numerical parity gate. The 63 skips are expected — they audit historical records this
package deliberately does not ship. **Any failure is a real failure.** Report it and stop;
do not "try it anyway".

### 2. Measure your machine, don't assume it

```bash
python scripts/k_ladder/worker_throughput.py --workers 1 2 4 8 --K 10
```

This spawns real concurrent processes and reports how many **effective** workers your
machine actually delivers. On our reference machine, 10 logical CPUs gave **4.22** — core
count overstated capacity by more than half. Send us this output; it tells us how many
processes you should run in parallel.

### 3. Run your slice

Operator `i` runs slice `i` of 4 (`i` = 0, 1, 2, or 3 — Holly assigns these):

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src \
  python3 scripts/k_ladder/run_pilot_job.py --slice i/4
```

To use your cores, launch several of these with a finer split. If your machine sustains 4
effective workers, operator 0 would run:

```bash
for s in 0 4 8 12; do
  OMP_NUM_THREADS=1 PYTHONPATH=src nohup \
    python3 scripts/k_ladder/run_pilot_job.py --slice $s/16 > slice$s.log 2>&1 &
done
```

Slices `0, 4, 8, 12` of 16 are exactly slice `0` of 4, split four ways.

**`OMP_NUM_THREADS=1` is not optional.** Without it each worker's BLAS opens its own
threads, the machine oversubscribes, and everything runs slower.

Safe to interrupt: outputs are written atomically and an existing one is never
overwritten, so re-running the same command resumes where it stopped. It will not redo
finished work and it will not corrupt anything.

Progress:

```bash
find results/k_ladder_pilot/factorial -name 'chain*.json' | wc -l    # your share is 90
```

### 4. Send back

The whole directory `results/k_ladder_pilot/factorial/`, plus your
`worker_throughput.py` output and your slice logs. Each file records its own hostname,
commit, RNG root and runtime, so we can verify the merge.

---

## What to expect

- `K = 3` chains finish in seconds; `K = 30, X = 166.7` chains take about **40 minutes**.
  A single chain cannot be split, so 40 minutes is the floor no amount of parallelism
  beats.
- Peak memory is about **0.2 GB per worker**. Memory will not be your limit; cores will.
- A `failed` line in the log is recorded, not hidden. Keep going and tell us — one failed
  cell blocks the analysis, so we need to see it rather than have it silently missing.

## Please do not

- **Skip cells that look unpromising.** Every one of the 360 must run. A pilot that prunes
  itself has selected on its own results.
- **Change any parameter** — sweeps, warm-up, `u_every`, `u_scale`, `X`, chains,
  replicates. They come from the manifest, which was frozen before any result existed.
- **Rerun a cell hoping for a better number.** If something looks wrong, tell us.
- **Mix machines' output into one directory yourself.** Send four separate trees; the
  collection script checks them against each other.

## Why the fuss about not changing things

The decision rule — which `X` and which `u_scale` win — was written down and tested
**before** any of these numbers existed, and it never looks at the ground truth. That is
what makes the choice defensible rather than a story fitted afterwards. Every deviation,
however sensible it seems at 2am, weakens it. If something needs to change, it should
change deliberately, in the manifest, with a new tag.

---

## What Holly does afterwards

```bash
python scripts/k_ladder/collect_pilot.py --from m0/ m1/ m2/ m3/ --dry-run
python scripts/k_ladder/collect_pilot.py --from m0/ m1/ m2/ m3/
python scripts/k_ladder/aggregate_pilot.py
```

The aggregator refuses to choose anything from an incomplete or inconsistent grid: any
missing cell, duplicate, hash mismatch, mixed commit, mixed RNG root, or `N_U` not equal
to its quota is a blocking failure.
