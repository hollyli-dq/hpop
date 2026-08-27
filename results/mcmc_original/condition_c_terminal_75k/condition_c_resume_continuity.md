# Condition C — resume and interruption continuity

execution/provenance deviation only; the scientific target, kernel, seeds, scales, cadence, gates and stopping rule were unchanged throughout.

| # | checkpoint sweep | resumed sweep | RNG restored | gate history unchanged | draws lost | compute lost | scientific state changed |
|---|---|---|---|---|---|---|---|
| 1 | 50,000 | 50,000 | yes | yes | 0 | ~20 | no |
| 2 | 52,000 | 52,000 | yes | yes | 0 | ~30 | no |

**Cause.** signals delivered to the launching shell's process group; resolved by relaunching the orchestrator in its own session (start_new_session=True, verified PPID 1).

**Resume mechanism.** each chain resumed from its stored chain state and RNG bit-generator state; the ladder restarted at the first UN-evaluated rung so no registered checkpoint result was recomputed; the two-consecutive-pass counters were reconstructed from the recorded gate files.

**Verified.** All registered gate artifacts (30k, 50k, 75k for both arms)
remained byte-identical across both resumes. Compute time was lost; no retained
scientific state was lost.

**Transient metadata rewrite.** the frozen runner rewrites formal_registration.json on entry, so during each resumed segment two fields transiently carried resume-time values: parent_commit and checkpoints. Both were restored to the launch values (50eee50; the full four-rung ladder). Every other field regenerated identically.

**Final stop.** SIGTERM to the orchestrator, then SIGTERM to its eight orphaned workers after verifying their identity by elapsed time; peer-session processes untouched. Partial writes found:
none. All checkpoints preserved.
