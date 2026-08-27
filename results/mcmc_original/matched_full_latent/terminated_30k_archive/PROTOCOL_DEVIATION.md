# Protocol deviation — FULL-LATENT formal ladder stopped after the 30,000 gate

**Condition:** FULL-LATENT (`matched-full-latent-v1`), arms FULL-COND and FULL-MARG.
**Authorised by:** principal investigator, explicit instruction in session.
**Nature of deviation:** the registered checkpoint ladder (30k → 50k → 75k → 100k, two
consecutive PASS to stop, ceiling 100k never extended) was **halted after the 30,000 gate**
instead of being run to a terminal state.

## What was NOT changed

- The registered 30,000 verdicts stand exactly as computed and written by the runner:
  **FULL-COND = FAIL**, **FULL-MARG = FAIL**. Neither was altered or reclassified.
- No threshold, gate function, ESS floor, R-hat gate, burn-in, thin, cadence, proposal
  scale, seed, or any other sampler setting was modified.
- No posterior result was modified. No checkpoint was edited.
- No truth-dependent recovery was opened or recomputed. The diagnosis used only
  permutation-invariant, truth-free quantities.
- The optimised backend was **not** used to advance or resume these chains.

## Why the ladder was stopped

### 1. FULL-COND shows persistent disagreement across four exact unordered libraries

Each chain's canonical **unordered** structural library (the three skills' 20-bit relation
vectors, sorted so the quantity is invariant to skill relabelling) is distinct and stable:

| chain | current library | transitions after burn-in | last transition |
|---|---|---|---|
| 0 | `ea3c2619` | 0 | — |
| 1 | `475b84a2` | 2 | sweep 11,270 |
| 2 | `ee323379` | 3 | sweep 14,900 |
| 3 | `bd386c35` | 5 | sweep 24,560 |

126 of 183 registered summaries violate R-hat ≤ 1.01 or bulk ESS ≥ 500.
`sorted_relation_counts[1]` is constant-but-unequal across chains (4, 5, 5, 5), giving
R-hat = infinity and bulk ESS = 0 at every prefix from 14k to 30k. `log_target` chain means
span roughly 825 nats and are as far apart in the last 2,000 sweeps as over the full window.
No registered invariant trends toward 1; `P_trace2`/`P_trace3` worsen.

This is *stronger* than the Condition C signature: there the chains shared one unlabeled
library and differed only in assignment to anchored identities. Here the libraries differ.

### 2. FULL-MARG has occupied one common exact unordered library since sweep 12,490

All four chains are in library `2d0f2bd5`. Consensus has been unbroken since **sweep 12,490**
— 3,503 of 4,000 retained draws, **87.6% of the retained window**. Chains 0, 1 and 2 have
made zero library transitions; chain 3 made three, the last at sweep 12,490.

Segmentation is well mixed in FULL-MARG and has essentially stabilised — this is not a
statement about a transient "currently mixed" state but about a settled one. Every
non-structural axis is under threshold: boundary probes max R-hat 1.003 (100% under),
same-segment probes 98% under, `total_segments` R-hat 1.000 with bulk ESS 15,108,
`pi_entropy`/`pi_l2` R-hat 1.0001 with bulk ESS ≈ 16,000, and 8 of 9 P summaries under 1.01.

### 3. MARG's formal failure is driven by the fixed retained window and ESS floors on
### discrete summaries that became constant and equal

After sweep 12,490 the failing structural invariants are a **point mass across all four
chains** — verified all-chains-constant-and-equal at `total_relations` = 17,
`sorted_relation_counts[1]` = 6, `sorted_relation_counts[2]` = 6.

Restricted to the post-consensus window, R-hat is exactly **1.0000** and bulk ESS is 14,012
for each of them. Over the full window they read R-hat 1.15 and bulk ESS ≈ 18.

The mechanism is an interaction inside the registered gate:

- `_diag` returns `bulk_ess = n_draws` (a trivial PASS) when all four chains are constant
  **and equal over the whole window**.
- The retained window begins at the fixed burn-in and therefore permanently contains chain
  3's pre-12,490 excursion, so the non-degenerate branch is taken instead.
- The resulting ESS measures **one historical relocation**, not ongoing exploration, and a
  quantity that has converged to a point cannot accumulate ESS against a floor of 1000/500.

MARG's failure is therefore not evidence of present between-chain disagreement.

### 4. Measured counterfactual extension predicts neither arm can pass by the 100k ceiling

Counterfactual assumption, stated explicitly and applied only to the discrete structural
invariants that have in fact been frozen for 3,503 consecutive draws: each chain holds its
current value. This is an assumption, not observed data, and no new sampling is implied.

| FULL-MARG invariant | metric | 30k | 50k | 75k | 100k | floor |
|---|---|---|---|---|---|---|
| `total_relations` | R-hat | 1.1516 | 1.0680 | 1.0403 | **1.0286** | 1.01 |
| `total_relations` | bulk ESS | 17.9 | 34.3 | 54.5 | **74.8** | 1000 |
| `sorted_relation_counts[1]` | R-hat | 1.1533 | 1.0685 | 1.0405 | **1.0288** | 1.01 |
| `sorted_relation_counts[1]` | bulk ESS | 17.8 | 34.1 | 54.3 | **74.5** | 500 |

The transient's share of the window falls 12.4% → 6.2% → 3.8% → 2.8%, so **R-hat improves
under the counterfactual extension but does not reach 1.01 by the 100k ceiling** — it
reaches only 1.029, still above the registered gate — and bulk ESS remains one to two
orders of magnitude below its floor. Measured ESS accumulation
for `total_relations` is ~0.87 per 1,000 sweeps; the floor of 1000 would require on the
order of 1.1M sweeps against a ceiling of 100k.

FULL-COND has no invariant trending toward its gate at any prefix, and its
constant-but-unequal degeneracy cannot be removed by additional sweeps at the observed
structural mobility (12–22 accepted H changes per chain in 30,000 sweeps, chain 0 zero).

### 5. The run was stopped to avoid approximately 45 hours of computation that could not
### change the registered verdict

At the measured contended rates (2.09–2.12 s/sweep COND, 2.31–2.35 s/sweep MARG), advancing
both arms from 30,000 to the 100,000 ceiling is ~70,000 further sweeps per chain, about
**45 hours of wall-clock** on this machine. On the evidence above that computation would
have terminated in FAIL at every remaining rung.

## Where the run was stopped

Stopped at a **normal durable checkpoint at sweep 32,000**, uniform across all eight
chains (COND written 22:15, MARG 22:5x on 2026-08-22). Checkpoints are written every 2,000
sweeps via `os.replace`, i.e. atomically, so a stop between writes cannot corrupt one.

Sequence: SIGTERM to the orchestrator (pid 3682), which exited gracefully; the eight worker
processes and the multiprocessing resource tracker then terminated. Final process count:
orchestrator 0, workers 0, tracker 0. No stray `.tmp.npz` files were left in
`formal_chains/`. No sampler setting was changed to reach or to hasten this checkpoint.

## Verification performed after stopping

| check | result |
|---|---|
| every worker terminated | orchestrator 0, workers 0, tracker 0 |
| checkpoints load and are self-consistent | 16/16 (8 at 30k, 8 at 32k), **0 corrupted**; retained-draw counts match metadata; all `log_target` finite |
| stray atomic-write temp files | none |
| 30k gate files byte-identical to the archived copies | yes, both (`70c9cc11…`, `b720ebe2…`) |
| verdicts still read FAIL | FULL-COND FAIL/`pass=false`, FULL-MARG FAIL/`pass=false`, both at checkpoint 30000, `chain_iterations = [30000]*4` |
| thresholds unchanged | `git diff HEAD` empty for the runner and all of `src/hpop/mcmc_original/`; `RHAT_GATE = 1.01`, `BURN_IN = 10_000`, `THIN = 5`, `CHECKPOINTS = (30_000, 50_000, 75_000, 100_000)`, `ESS_FLOORS = {log_target_bulk 1000, log_target_tail 500, total_relations_bulk 1000, remaining_invariant_bulk 500}` |

**Neither formal FAIL verdict was reclassified.** No threshold, gate, or posterior result
was modified at any point.

## Provenance of record

- Launch source commit: `bed564b5ce7ee780d37360c3bb14cb8b6ccac724`
- Corpus hash `dd280a4a…`; train `717b77a4…`; held-out `4d586fc5…`; truth `fc41538f…`
- Arm seeds: FULL-COND 6,206,201–204; FULL-MARG 6,206,211–214
- Paired starts: U-start seeds 6,204,101–104 at scales 0.5 / 1.0 / 2.0 / 3.0;
  pi/P start seeds 6,206,101–104. Four paired starts shared across both arms.
- Burn-in 10,000; thin 5; checkpoint every 2,000; structural cadence 10, scale 0.5.

**Truth status.** This corpus's terminal recovery truth was unsealed mid-run on 2026-08-22
(PI-authorised, recorded in `TRUTH_UNSEAL_midrun.json` and commit `3244ba4`). That unseal is
irrelevant to this gate diagnosis, which used only permutation-invariant truth-free
quantities, but it does mean **this run can no longer serve as a sealed confirmatory
result**. A fresh, fully sealed launch is required for that purpose.

## Consequence for resuming (stated, not acted on)

The chains now sit at 32,000 while a gate is on record at 30,000.
`_assert_gate_checkpoint_lineage` requires the four checkpoints of an arm with gate history
to sit at *exactly* the last gated checkpoint, so a future `--launch-formal` will refuse
with a lineage error rather than silently splice diagnostics from different chain states.
This is the anti-splice rail behaving as designed. The exact 30,000 states are preserved and
hashed in `checkpoints_30k/`, so the pre-gate lineage can be restored deliberately if that is
ever wanted. **No such restoration was performed.**

## Contents of this archive

| path | contents |
|---|---|
| `checkpoints_30k/` | the eight gated 30,000 checkpoints |
| `checkpoints_32k_interrupt/` | the eight 32,000 interrupt checkpoints |
| `gate_30k/` | the two registered 30,000 gate artifacts |
| `diagnosis/gate_diagnosis_30k.json` | truth-free diagnosis output |
| `diagnosis/rerun_diagnosis.py` | self-contained script reproducing it from `checkpoints_30k/` |
| `manifests/` | launch manifest, prelaunch audit/validation, smoke, discarded-prefix record, preregistration (json + md) |
| `SHA256SUMS.txt` | hashes of the 30k checkpoints and gate artifacts |
| `SHA256SUMS_interrupt_and_manifests.txt` | hashes of the 32k interrupt checkpoints and manifests |
| `formal_runner.log` | orchestrator log for the whole run |

The optimised backend was **not** used to advance or resume these chains at any point.


## Correction log

| date | change | reason |
|---|---|---|
| 2026-08-22 | §2 wording: "Every other axis is mixed" → "Segmentation is well mixed in FULL-MARG and has essentially stabilised…" | PI correction. The original phrasing could be read as describing a transient state; FULL-MARG's segmentation is settled, not merely mixing at the moment of measurement. |
| 2026-08-22 | §4 wording: "R-hat does dilute and trend toward 1 — but reaches only 1.029" → "R-hat improves under the counterfactual extension but does not reach 1.01 by the 100k ceiling" | PI correction. States the gate outcome explicitly rather than leaving the reader to compare 1.029 against 1.01. |

Wording only. **No number, verdict, threshold or hash was altered**, and this file is not
covered by either `SHA256SUMS` manifest, so no recorded hash is invalidated.
