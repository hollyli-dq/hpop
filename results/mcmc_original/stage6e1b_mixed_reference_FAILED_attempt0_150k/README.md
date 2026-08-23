# Stage 6E1B, attempt 0 — FAILED one gate. Preserved, not deleted.

4 chains x 150,000 sweeps, 30,000 burn-in, thin 4 (120,000 pooled draws).

    induced_h_total_variation   0.010500   threshold 0.01   FAIL

Every other gate passed: TV(S, z) 0.00976, boundary 0.00447, occurrence-label 0.00448,
relation marginal 0.00805, segment-count TV 0.00713, the mixed multivariate statistic
(0.003969 against a 0.004522 envelope, z = 0.78), and all ten registered R-hats
(worst 1.00418).

## Why this is Monte Carlo error and not bias

Every statistic that would move under a wrong invariant distribution is clean:

| scalar | MCMC mean | reference mean | gap in reference SD | sd ratio | bulk ESS |
|---|---:|---:|---:|---:|---:|
| rho | +0.4734 | +0.4746 | 0.004 | 1.001 | 4549 |
| beta | +1.7547 | +1.7321 | 0.024 | 0.992 | 594 |
| omega | -0.2306 | -0.2536 | 0.012 | 0.999 | 1020 |
| lambda_rep | +0.5368 | +0.5390 | 0.006 | 0.993 | **301** |
| lambda_back | +0.8925 | +0.9056 | 0.020 | 0.993 | 1390 |

A sampler targeting the wrong distribution does not reproduce five posterior means to
within 0.024 reference standard deviations and five standard deviations to within 0.8%.
What it does show is a **low effective sample size** — 301 for `lambda_rep` out of 120,000
retained draws, an integrated autocorrelation time near 400.

`TV(H)` is a sum of 19 cell errors per skill, so it accumulates Monte Carlo noise faster
than any single marginal: at an effective size of order 4 x 10^4 the expected TV is
approximately 0.5 * sum_i sqrt(p_i / n) ~ 0.01 — exactly what was observed. The per-skill
values, 0.01037 / 0.01050 / 0.00667, are consistent with noise of that size rather than
with a systematic displacement.

## What was changed for attempt 1

**Nothing about the model, the target, the kernel, the proposal scales, the reference or
the gate.** The gate stays at 0.01 and is not widened. The only change is the number of
draws — 150,000 sweeps to 600,000 — which is the one intervention that reduces Monte Carlo
error without touching what is being measured. If the discrepancy were bias it would not
move; if it is noise it falls as 1/sqrt(n), to roughly 0.005.

The chains were also switched from sequential to parallel execution, which changes wall
time and nothing else: each chain has its own registered seed and its own RNG stream.
