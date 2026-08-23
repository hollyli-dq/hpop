# Truth-free terminal verdict — optimized FULL-LATENT confirmatory experiment

## FORMAL VERDICTS

| arm | verdict |
|---|---|
| **FULL-COND** | **FAIL** |
| **FULL-MARG** | **FAIL** |

Neither verdict is amended or reclassified. Both were produced by the single terminal
gate defined in `PREREG_CONFIRMATORY.md` (commit `8f99dd58`), evaluated once on
production draws only.

## Provenance and timing

| item | value |
|---|---|
| backend commit | `564995efd056d7d33984f0ca1532386e6140ea0c` |
| preregistration commit | `8f99dd582e172b5dd253cae1bd74d65c96972f93` |
| runner commit | `fe47d9141b2be87a1fb01d22eff93627716bdfb3` |
| corpus hash | `3e3aa6533bd7951f9b2ed1dfa050e9d07f1b2e96b0e3914e044010130e9acdfa` |
| truth hash | `effccc91114d9647f87859aaa7ee219e3bc664bb41d2c6389844ea64346a5e8e` (SEALED) |
| sampling completed | 2026-08-23T02:12:54Z |
| chains | 8 × 150,000 sweeps (50,000 warm-up discarded + 100,000 production, thin 5) |
| production draws | 20,000 per chain, 80,000 per arm |

**Gate implementation timing, stated explicitly.** The terminal gate was implemented
AFTER sampling terminated and BEFORE truth was unsealed. Its decision rules were fixed in
the preregistration commit made before launch; only the code realising them was written
afterwards. Truth remained sealed throughout gate development, execution and audit, and
no gate quantity is a function of truth.

## Independent reproduction

The gate imports the repository's registered `stage6b_mcmc_diagnostics`. The audit
(`scripts/confirmatory_gate_audit.py`) implements rank-normalized split R-hat, bulk ESS
and tail ESS from scratch after Vehtari et al. (2021) and compares.

| arm | branch agreement | per-summary pass/fail | gate verdict | audit verdict |
|---|---|---|---|---|
| FULL-COND | 179/179 | 173/179 | FAIL | FAIL |
| FULL-MARG | 179/179 | 179/179 | FAIL | FAIL |

R-hat reproduces to 1.9e-16. Six FULL-COND probes straddle the ESS 400 floor because two
legitimate Geyer variants differ by up to 70% on slowly-mixing binary series; the audit is
strictly the more conservative of the two, so it fails more summaries, not fewer. **Both
arm verdicts reproduce.**

The audit also found and fixed two defects in its own implementation before agreeing:
an ESS floor that permitted ESS above the sample size, and a tail-ESS rule that discarded
the informative quantile side for binary probes. Both were audit bugs; the gate's numbers
were correct in each case.

## FULL-COND — genuine non-convergence

**130 of 179 registered summaries fail.**

| summary | R-hat | bulk ESS | tail ESS | MCSE | gate |
|---|---|---|---|---|---|
| `log_target` | 2.4589 | 4.8 | 23.4 | 3.68e+01 | **FAIL** |
| `total_segments` | 1.0995 | 24.1 | 52.4 | 1.93e+00 | **FAIL** |
| `mean_segment_length` | 1.0995 | 24.1 | 47.0 | 2.33e-02 | **FAIL** |
| `sd_segment_length` | 1.2341 | 11.8 | 27.2 | 1.82e-02 | **FAIL** |
| `sorted_pi[0]` | 1.4989 | 7.2 | 74.7 | 1.00e-02 | **FAIL** |
| `sorted_pi[1]` | 2.0489 | 5.3 | 24.5 | 5.94e-02 | **FAIL** |
| `sorted_pi[2]` | 2.0721 | 5.2 | 24.3 | 5.71e-02 | **FAIL** |
| `pi_entropy` | 1.7423 | 6.0 | 30.4 | 6.22e-02 | **FAIL** |
| `pi_l2` | 1.9053 | 5.5 | 38.7 | 3.19e-02 | **FAIL** |
| `P_frobenius` | 1.2606 | 11.2 | 36.2 | 1.94e-02 | **FAIL** |
| `P_trace2` | 1.6926 | 6.2 | 34.6 | 5.70e-02 | **FAIL** |
| `P_trace3` | 1.6926 | 6.2 | 34.6 | 8.54e-02 | **FAIL** |

**Canonical closure library: branch (c), non-degenerate — FAIL.**

| quantity | value |
|---|---|
| R-hat | **3.1140** |
| bulk ESS | 4.5 |
| tail ESS | 5.0 |
| distinct libraries per chain | [1, 5, 3, 2] |

The four chains hold disjoint library sets. They are exploring different structural
regions and have not met.

## FULL-MARG — 12 failures, 11 of them an undefined statistic

| summary | R-hat | bulk ESS | tail ESS | MCSE | gate |
|---|---|---|---|---|---|
| `log_target` | 1.0039 | 2144.1 | 34841.5 | 4.61e-01 | PASS |
| `total_segments` | 1.0000 | 79726.1 | 78070.8 | 2.49e-02 | PASS |
| `mean_segment_length` | 1.0000 | 79727.4 | 78218.2 | 3.12e-04 | PASS |
| `sd_segment_length` | 1.0000 | 80134.6 | 78118.7 | 1.53e-04 | PASS |
| `sorted_pi[0]` | 1.0000 | 80491.6 | 78357.1 | 1.06e-04 | PASS |
| `sorted_pi[1]` | 1.0000 | 79351.1 | 79953.4 | 1.74e-04 | PASS |
| `sorted_pi[2]` | 1.0000 | 78813.8 | 79680.2 | 1.86e-04 | PASS |
| `pi_entropy` | 1.0000 | 79693.6 | 78096.6 | 2.20e-04 | PASS |
| `pi_l2` | 1.0000 | 79069.9 | 78903.5 | 1.01e-04 | PASS |
| `P_frobenius` | 1.0000 | 69645.0 | 75864.3 | 1.32e-04 | PASS |
| `P_trace2` | 1.0000 | 69786.7 | 76507.7 | 3.81e-04 | PASS |
| `P_trace3` | 1.0000 | 69786.5 | 76507.7 | 5.72e-04 | PASS |

**Canonical closure library: branch (a), constant and equal — but FAIL on precondition 4.**

| precondition | status |
|---|---|
| 1. starts structurally dispersed | PASS — four distinct start libraries |
| 2. every chain accepted an H-changing move in warm-up | PASS — 34 / 49 / 59 / 48 |
| 3. library constant and equal in production | PASS — `d72975bc6f6cbaf5` in all 80,000 draws |
| 4. all other registered diagnostics pass | **FAIL** |

Precondition 4 fails because of the 11 probes below, so the library result is recorded as
FAIL. It is not independent evidence of structural disagreement.

### The 11 undefined-tail-ESS probes

| probe | R-hat | bulk ESS | tail ESS | posterior mean |
|---|---|---|---|---|
| `coskill_probes[2]` | 1.0000 | 80376.2 | undefined (0/0) | 0.99775 |
| `coskill_probes[3]` | 1.0000 | 80095.6 | undefined (0/0) | 0.99950 |
| `coskill_probes[11]` | 1.0000 | 80047.2 | undefined (0/0) | 0.99979 |
| `coskill_probes[17]` | 1.0000 | 79399.0 | undefined (0/0) | 0.99057 |
| `coskill_probes[37]` | 1.0000 | 79818.9 | undefined (0/0) | 0.95381 |
| `coskill_probes[40]` | 1.0000 | 80278.3 | undefined (0/0) | 0.96548 |
| `coskill_probes[47]` | 1.0000 | 78965.5 | undefined (0/0) | 0.98615 |
| `same_segment_probes[2]` | 1.0000 | 80041.8 | undefined (0/0) | 0.99638 |
| `same_segment_probes[3]` | 1.0000 | 80095.6 | undefined (0/0) | 0.99950 |
| `same_segment_probes[11]` | 1.0000 | 80047.2 | undefined (0/0) | 0.99979 |
| `same_segment_probes[17]` | 1.0000 | 79399.0 | undefined (0/0) | 0.99057 |

Every one has R-hat 1.0000 and bulk ESS near 80,000 — the maximum attainable from 80,000
draws. Their posterior means lie between 0.953 and 0.9999, so both the 5% and the 95%
quantile indicators are constant and tail ESS is 0/0.

**These are not convergence failures.** They are a gap in the preregistered gate: §6.2b
branched binary probes on *constancy* and did not anticipate *near*-constancy leaving tail
ESS undefined while the probe mixes perfectly. The gap was written into the
preregistration before any draw existed, and the gate is applied as written.

## Substantive truth-free finding

On a fresh sealed corpus with four structurally dispersed starts, FULL-MARG's four chains
converged to one exact canonical partial-order library and held it for all 80,000
production draws, having demonstrably moved H during warm-up. FULL-COND's four chains did
not meet at all. That separation is truth-free and is unaffected by the gate defect.

Recovery against truth is deliberately not reported here; truth is sealed at the time of
this document.
