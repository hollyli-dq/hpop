# POST-HOC DIAGNOSTIC SENSITIVITY — NOT THE FORMAL VERDICT

> The registered verdicts are **FULL-COND = FAIL** and **FULL-MARG = FAIL**.
> Nothing in this document amends, reclassifies or supersedes them. The rule examined
> below was chosen **after** seeing which summaries failed, which is precisely why it
> cannot carry a verdict.

## The alternative rule

> For a Bernoulli probe whose 5% and 95% quantile indicators are both constant, tail ESS is undefined; treat it as not applicable and require R-hat <= 1.01, bulk ESS >= 400, and MCSE of the posterior probability <= 0.01.

## What it would conclude

| arm | registered verdict | reclassified as N/A | still failing | verdict under alternative |
|---|---|---|---|---|
| FULL-COND | **FAIL** | 0 | 129 | FAIL *(sensitivity only)* |
| FULL-MARG | **FAIL** | 11 | 0 | PASS *(sensitivity only)* |

**FULL-COND is unaffected.** Zero probes qualify for reclassification and 129 summaries
still fail, including the canonical library at R-hat 3.1140. Its failure is not a
diagnostic-definition artifact under any reading.

**FULL-MARG turns on exactly the 11 probes.** All 11 reclassify, no summary then fails, and
the canonical library's precondition 4 becomes satisfied, so the library records branch (a)
degenerate agreement.

### The 11 probes under the alternative rule

| probe | R-hat | bulk ESS | posterior probability | MCSE(p) |
|---|---|---|---|---|
| `coskill_probes[11]` | 1.0000 | 80047.2 | 0.99979 | 5.15e-05 |
| `coskill_probes[17]` | 1.0000 | 79399.0 | 0.99057 | 3.43e-04 |
| `coskill_probes[2]` | 1.0000 | 80376.2 | 0.99775 | 1.67e-04 |
| `coskill_probes[37]` | 1.0000 | 79818.9 | 0.95381 | 7.43e-04 |
| `coskill_probes[3]` | 1.0000 | 80095.6 | 0.99950 | 7.90e-05 |
| `coskill_probes[40]` | 1.0000 | 80278.3 | 0.96547 | 6.44e-04 |
| `coskill_probes[47]` | 1.0000 | 78965.5 | 0.98615 | 4.16e-04 |
| `same_segment_probes[11]` | 1.0000 | 80047.2 | 0.99979 | 5.15e-05 |
| `same_segment_probes[17]` | 1.0000 | 79399.0 | 0.99057 | 3.43e-04 |
| `same_segment_probes[2]` | 1.0000 | 80041.8 | 0.99638 | 2.12e-04 |
| `same_segment_probes[3]` | 1.0000 | 80095.6 | 0.99950 | 7.90e-05 |

Every one has R-hat 1.0000, bulk ESS near the attainable maximum of 80,000, and an MCSE on
its posterior probability below 1e-4 — four orders of magnitude inside the 0.01 threshold.

## How to read this

The sensitivity says the FULL-MARG failure is driven entirely by an undefined statistic
rather than by any evidence of non-convergence. It does **not** say FULL-MARG passed. The
gate defect was written into the preregistration before any draw existed; the honest
record is a FAIL under the registered rule, with this analysis attached.

If the rule is to be adopted, it must be preregistered for a **future** experiment, not
applied to this one.
