# Held-out negative log likelihood — optimized FULL-LATENT confirmatory experiment

Section 10 of `PREREG_CONFIRMATORY.md`, the last quantity the preregistration promised.
The estimator, the subsample size and the subsample spacing were all fixed before the
seal was opened, so none of them could be chosen after seeing a result.

## Estimator

For each of the 45 held-out traces, using production draws only:

    NLL_n = -log( (1/M) * sum_m Z_n(U^(m), pi^(m), P^(m)) )

computed in the log domain as `-(logsumexp_m log Z_n^(m) - log M)`. `Z_n` is the exact
all-segmentation marginal likelihood from the **frozen** semi-Markov forward recursion in
`hpop.mcmc_original`. `M` = 1,000 draws per chain, **4,000 per arm**, systematic at
stride 20 through the 20,000 retained production draws (warm-up 50,000 discarded,
production 100,000, thinning 5).

The held-out traces were never used for inference. Both arms were computed by identical
code on identical traces.

## Registered result

| | FULL-COND | FULL-MARG |
|---|---|---|
| total NLL over 45 traces | **2618.691** | **2481.612** |
| bootstrap 95% interval (total) | [2417.2, 2821.7] | [2285.1, 2680.1] |
| per-trace mean | **58.1931** | **55.1469** |
| bootstrap 95% interval (per trace) | [53.7146, 62.7046] | [50.7790, 59.5575] |
| per CPA occurrence (1608 occurrences) | 1.6285 | 1.5433 |
| draws used | 4000 | 4000 |

Lower is better.

## How to read the two intervals

**The marginal intervals overlap and that is not evidence of no difference.** They are
bootstrap intervals over traces, and the between-trace spread is large — held-out traces
differ in length and in content, so some are simply harder than others for both arms. That
variance is common to the two arms and cancels.

Both arms are evaluated on the **same** 45 traces, so the comparison is
paired:

| | value |
|---|---|
| FULL-COND minus FULL-MARG, per trace | **+3.0462** |
| paired bootstrap 95% interval | **[+2.4751, +3.5726]** |
| total difference | +137.079 nats |
| traces on which FULL-MARG predicts better | **41 of 45** |

The paired interval excludes zero. **FULL-MARG predicts held-out traces better than
FULL-COND**, by about 3.0 nats per trace.

The preregistration fixes the estimator and both arms; it does not separately register
this difference. The difference is the contrast implied by computing both, not a
statistic chosen after the fact, and it is reported here as such.

## Supplementary reference — NOT preregistered

A predictive score is hard to read without a scale, so the same estimator was evaluated at
the **unsealed truth parameters**. This is descriptive context only. It is not part of
Section 10, it is not a gate, and it does not enter any verdict.

| | total NLL | per occurrence |
|---|---|---|
| truth parameters (single point) | 2475.973 | 1.5398 |
| FULL-MARG posterior | 2481.612 | 1.5433 |
| FULL-COND posterior | 2618.691 | 1.6285 |

FULL-MARG is **+5.64 nats** from the truth-parameter
value across all 45 traces; FULL-COND is
**+142.72 nats**.

Two cautions on that comparison. The truth row is a **plug-in point prediction**, the arm
rows are **posterior averages** — different objects, and a posterior average is not
required to be worse than a plug-in, so the truth row is a reference scale and not an
upper bound. And the truth was drawn from the prior the sampler assumes, which is the most
favourable case a well-specified model can face.

## The convergence caveat, stated plainly

**Both arms FAIL the single terminal gate** — FULL-COND on 130 criteria, FULL-MARG on 12 — and neither verdict is amended by anything in this document. These predictive numbers are computed from draws
whose convergence was **not** established. They describe what the retained draws predict;
they are not evidence that either chain sampled its posterior correctly.

## Numerical verification

The primary computation uses the frozen recursion because Section 10 names it. The
optimized backend was run as a cross-check on every 40th draw:

| | value |
|---|---|
| worst absolute disagreement | 4.263e-14 |
| draws cross-checked per arm | 100 |
| tolerance | 1e-10 |
| result | **PASS** |

Every `log Z` was finite. Threading was pinned to one thread so the numbers reproduce.

## Files

- `heldout_nll.json` — every quantity above, machine-readable
- `heldout_nll_per_trace.csv` — per-trace NLL for both arms and their difference
- `run.log` — the run transcript (38.2 minutes)

Bootstrap: 10,000 resamples, seed 6300010, over the
45 held-out traces.
