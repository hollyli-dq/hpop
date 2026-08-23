# Synthetic held-out predictive evaluation — VERIFIED

> Verified against the stored artifacts; **not recomputed** — every check passed.
> The registered convergence verdicts are **FULL-COND = FAIL** and **FULL-MARG = FAIL**.
> This predictive calculation describes the retained draws and does not alter them.

## Result

| Method | Total NLL | Per-trace mean | NLL / occurrence | Better traces |
|---|---|---|---|---|
| **Ours (path-marginal inference)** | **2481.612** | **55.1469** | **1.5433** | **41/45** |
| w/o path marginalisation | 2618.691 | 58.1931 | 1.6285 | 4/45 |
| *Truth plug-in (reference)* | *2475.973* | *55.0216* | *1.5398* | *—* |

**Paired comparison on the same 45 traces:** difference (w/o − ours) =
**+3.0462 nats per trace**, paired bootstrap 95% CI
**[2.4751, 3.5726]**, total **+137.08 nats**,
ours better on **41/45** traces.

The marginal per-arm bootstrap intervals overlap — ours [50.78,
59.56], w/o [53.71,
62.70] — because trace difficulty is shared and cancels
only in the paired comparison. The paired interval is the informative one.

## Verification performed

| item | value |
|---|---|
| held-out trace IDs | 0–44, 45 unique, never used for inference |
| trace length distribution | 24×12, 32×11, 40×11, 48×11 |
| total CPA occurrences | 1608 |
| draw selection | systematic, stride 20 through the 20,000 retained production draws, offset 0 |
| draws per arm | 4000 (1,000 per chain × 4) |
| estimator | log-domain `-(logsumexp_m log Z_n - log M)` |
| log Z source | frozen semi-Markov forward recursion |
| bootstrap | 10000 resamples, seed 6300010, paired over the 45 traces |
| optimized vs reference cross-check | PASS, tolerance 1e-10, worst abs difference **4.26e-14** |
| CSV totals vs JSON | match exactly |
| freeze manifest | 34 files, no hash drift |

Every displayed value reproduces the stored artifacts. No recomputation was required.

## Reading the truth plug-in

It evaluates the same held-out marginal likelihood at the generating parameters. It is a
single point, not a posterior average, so it is **a scale reference and neither an upper
nor a lower bound** — a posterior average can beat it. It is kept out of the main
two-row method table and shown only as a separated reference row.
