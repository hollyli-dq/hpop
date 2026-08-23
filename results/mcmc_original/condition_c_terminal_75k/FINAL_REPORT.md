# CONDITION C TERMINATED AT 75k — FORMAL VERDICT: NOT CONVERGED

Terminal commit: `8024bf653c16c0ee5279af0d5834d9c3cce1784c` · launch commit `50eee50` ·
corpus `dd280a4a09896154…` · truth `fc41538fd44d170d…`

## 1. Formal registered target

`p(S, z, U | X, vartheta*, pi*, P*, delta_B*, epsilon*, rho_0)` with rho fixed
at 0.5. Only `(S, z)` and `U` are latent; the four
recurrent scalars, `pi*`, `P*`, `delta_B` and `epsilon` are fixed to the
generating values.

## 2. Configuration

Frozen formal corpus (100 train + 45 held-out traces, K=3, m=5, d=2),
u_scale 0.5, scheduled path-marginal scale
1.0 at cadence 10.

## 3. Starts and seeds

Four paired dispersed starts (seeds
6204101, 6204102, 6204103, 6204104), shared across arms;
chain seeds [6204001, 6204002, 6204003, 6204004] (C-COND) and
[6204011, 6204012, 6204013, 6204014] (C-MARG). No start coincided with the truth
`H` tuple.

## 4. Proposal kernels

C-COND: conditional row moves on `U` plus an exact FFBS refresh of every
`(S, z)` each sweep. C-MARG: the same, plus a path-marginal structural move
every 10 sweeps scored against `ell_coll(U) = sum_n log Z_n(U)`.

## 5. Registered stopping protocol

Ladder [30000, 50000, 75000, 100000], burn-in 10,000, thin 5,
per arm PASS at two consecutive checkpoints, ceiling never extended.

## 6. Early-termination deviation

Condition C was terminated after the third consecutive failed registered checkpoint at 75k sweeps, before the preregistered 100k ceiling. Because the protocol required two consecutive passing checkpoints and only one checkpoint remained, the registered convergence criterion was no longer attainable.

## 7. Checkpoint results

All six registered gates FAILED. See `condition_c_checkpoint_table.md`.

## 8. Truth-free convergence diagnosis

Both arms fail through the degenerate case in which structural summaries are
constant within each chain but unequal across chains. Within-cell mixing is
healthy throughout: the `U` log-prior has R-hat ~1.00 in both arms at every
rung.

## 9. Structural-library diagnosis

C-COND: separated at the structural-library level
(3 libraries, split
2-1-1).
C-MARG: 1 library shared by
all four chains.

## 10. Anchored-assignment diagnosis

C-MARG remains split 3-1
across anchored assignments, gap
30k 124.39 nats; 50k 124.44 nats; 75k 124.30 nats. The phenomenon is
**anchored structure-to-skill assignment multimodality**, not label switching:
no non-identity permutation leaves the fixed `pi*`/`P*` invariant.

## 11. Movement diagnostics

C-MARG: 3,949 path-marginal accepts,
32 changed the anchored
tuple, 0 after burn-in.
C-COND: 7 assignment
changes after burn-in. Full detail in `figure_data/condition_c_movement.csv`.

## 12. Post-termination predefined recovery

Opened only after termination; predefined metrics only; draws through 75,000
only. Recovery summaries are descriptive diagnostics conditional on non-converged chains and are not interpreted as posterior estimates. See `recovery_75k.json`. Headline: C-MARG recovers the true
unordered library in
4/4 chains against
1/4 for C-COND;
exactly one chain per arm recovers the true anchored assignment.

## 13. Runtime

75,000 sweeps per chain in the primary
analysis (300,000 chain-sweeps per arm),
463 chain-hours including
post-decision compute. See `runtime_accounting.json`.

## 14. Resume and interruption provenance

Two orchestrator interruptions, both resumed from stored chain and RNG state
with all registered gate artifacts unchanged. See
`condition_c_resume_continuity.md`.

## 15. Blinding disclosure

See `condition_c_blinding_disclosure.md`.

## 16. Paper-safe claims

12 claims with exact wording, numbers, sources and caveats in
`CONDITION_C_PAPER_LEDGER.md`; drafted text in `CONDITION_C_PAPER_DRAFT.md`.

## 17. Limitations

Terminated before the preregistered ceiling; recovery is descriptive and
conditional on non-converged chains; the absence of anchored-assignment
crossings is an observation within the registered budget, not an impossibility
claim; basin volumes are unknown, so the nat gap is not a posterior odds ratio.

## 18. Condition C' motivation

The residual barrier is a coordinated reassignment of whole structures between
anchored identities. Condition C' adds exactly one global transposition move,
scored by the same path-marginal likelihood. It is a **prospectively frozen
follow-up motivated by Condition C diagnostics**, not preregistered before
Condition C data were observed, and it remains **unlaunched**.

## 19. Artifact inventory

`pre_stop_state.json`, `quarantine_manifest.json`, `artifact_hashes.json`,
`condition_c_checkpoint_table.{csv,json,md}`,
`condition_c_failure_geometry.{json,md}`, `recovery_75k.json`,
`runtime_accounting.json`, `terminal_verdict.json`,
`condition_c_resume_continuity.{json,md}`,
`condition_c_blinding_disclosure.md`,
`condition_c_cprime_chronology.{json,md}`, `CONDITION_C_PAPER_LEDGER.md`,
`CONDITION_C_PAPER_DRAFT.md`, `figure_data/*.csv`, plus the untouched
registered artifacts under `matched_condition_c/`.

## 20. Source and test hashes

`artifact_hashes.json`. Launch commit `50eee50`; C' preregistration `9b8e590`;
C' runner `ed63b55`.

## 21. Final verdict

**CONDITION C TERMINATED AT 75k — FORMAL VERDICT: NOT CONVERGED**

C-COND: NOT CONVERGED. C-MARG: NOT CONVERGED. Neither arm obtained two
consecutive passing checkpoints.
