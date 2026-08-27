# Stage 6B — complete: scalar, joint-3 and joint-4 recurrent MCMC

Date 2026-08-11 · branch `mcmc-original-latent-poset` · commit `8c6e0dc9`
Frozen model `9ad850f22065d85f6cfd855443395d06b8a566e8c8dcdd4f9d85b1f031e911cc`

The primary claim of this stage is **sampler correctness**: that the MCMC
posterior coincides with an independently computed reference. Synthetic recovery
— whether the generating value lands inside the posterior — is reported
separately and is *not* the sampler-correctness criterion.

## Status

| stage | what was inferred | claim | result |
|---|---|---|---|
| 6B1 | each scalar alone, others at truth | matches the four 1-D grids | **PASS** |
| 6B2 | `beta, omega, lambda_rep` (lambda_back = 0.25) | matches the independent 3-D reference | **PASS** |
| 6B3 | all four jointly | matches the independent 4-D reference | **PASS** |

## Reference quality

| stage | method | grid points | integral | outer-face mass | refinement drift | max correlation drift | importance ESS |
|---|---|---:|---:|---:|---:|---:|---|
| 6B2 | deterministic tensor grid, transformed coordinates | 226,981 | 1.0000000000 | 1.190e-07 | 0.02155 sd (n=41 vs 61) | 0.00000 | n/a — not importance/QMC |
| 6B3 | deterministic tensor grid, transformed coordinates | 6,765,201 | 1.0000000000 | 2.775e-09 | 0.10246 sd (n=35 vs 51) | 0.00000 | n/a — not importance/QMC |

No importance or QMC reference was needed: the four-dimensional grid was
affordable after collapsing identical recurrent states, so both references are
deterministic quadrature and no weight diagnostics apply.

## Per-parameter chain diagnostics

| stage | parameter | R-hat | bulk ESS | tail ESS | MCSE | KS to reference | std. mean error | std. 2.5% error | std. 97.5% error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6B2 | beta | 1.0004 | 7685 | 11357 | 0.000440 | 0.0064 | 0.0075 | 0.0129 | 0.0555 |
| 6B2 | omega | 1.0002 | 13264 | 14759 | 0.001085 | 0.0094 | 0.0019 | 0.0325 | 0.0268 |
| 6B2 | lambda_rep | 1.0010 | 7514 | 11432 | 0.000324 | 0.0080 | 0.0071 | 0.0079 | 0.0103 |
| 6B3 | beta | 1.0007 | 7413 | 10830 | 0.000454 | 0.0156 | 0.0275 | 0.0217 | 0.0921 |
| 6B3 | omega | 1.0003 | 12350 | 14560 | 0.001165 | 0.0106 | 0.0068 | 0.0437 | 0.0415 |
| 6B3 | lambda_rep | 1.0003 | 8055 | 11732 | 0.000317 | 0.0159 | 0.0256 | 0.0062 | 0.0772 |
| 6B3 | lambda_back | 1.0001 | 10146 | 12732 | 0.000238 | 0.0104 | 0.0077 | 0.0284 | 0.0311 |

Stage 6B1, for comparison (each scalar alone, against its 1-D grid):

| parameter | R-hat | bulk ESS | tail ESS | KS |
|---|---:|---:|---:|---:|
| beta | 1.0003 | 14165 | 15126 | 0.0118 |
| omega | 1.0004 | 14498 | 15892 | 0.0074 |
| lambda_rep | 1.0002 | 13935 | 15444 | 0.0049 |
| lambda_back | 1.0002 | 13739 | 14231 | 0.0082 |

## Acceptance by coordinate

| stage | parameter | total (per chain) | post burn-in (per chain) |
|---|---|---|---|
| 6B2 | beta | 0.444, 0.446, 0.444, 0.446 | 0.444, 0.447, 0.444, 0.448 |
| 6B2 | omega | 0.465, 0.463, 0.466, 0.458 | 0.462, 0.462, 0.467, 0.460 |
| 6B2 | lambda_rep | 0.436, 0.447, 0.443, 0.445 | 0.431, 0.447, 0.444, 0.442 |
| 6B3 | beta | 0.444, 0.447, 0.447, 0.443 | 0.441, 0.445, 0.447, 0.447 |
| 6B3 | omega | 0.472, 0.459, 0.469, 0.471 | 0.472, 0.462, 0.472, 0.472 |
| 6B3 | lambda_rep | 0.436, 0.440, 0.440, 0.436 | 0.436, 0.442, 0.437, 0.435 |
| 6B3 | lambda_back | 0.478, 0.476, 0.474, 0.474 | 0.479, 0.477, 0.476, 0.472 |

## Worst-case errors and the joint comparison

| stage | max mean error | max CI endpoint error | max correlation error | correlation envelope | energy distance | energy envelope | z | multivariate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 6B2 | 0.0075 | 0.0555 | 0.00839 | 0.04152 | 0.002126 | 0.004948 | +0.00 | **PASS** |
| 6B3 | 0.0275 | 0.0921 | 0.01539 | 0.05623 | 0.003142 | 0.003990 | +0.94 | **PASS** |

Both envelopes are calibrated, not chosen: each is the 99th percentile of the same statistic computed between two independent samples *from the frozen reference itself*, at the same sample sizes. The correlation envelope is calibrated at the chains' minimum bulk ESS rather than their raw draw count, since an MCMC correlation estimate carries the noise of its effective sample size.

## Pairwise dependence — what four marginals cannot establish

| stage | pair | reference corr | MCMC corr | abs error |
|---|---|---:|---:|---:|
| 6B2 | beta|omega | -0.15793 | -0.15849 | 0.00056 |
| 6B2 | beta|lambda_rep | +0.54948 | +0.55787 | 0.00839 |
| 6B2 | omega|lambda_rep | -0.08306 | -0.08896 | 0.00590 |
| 6B3 | beta|omega | -0.18815 | -0.17276 | 0.01539 |
| 6B3 | beta|lambda_rep | +0.49434 | +0.49821 | 0.00387 |
| 6B3 | beta|lambda_back | +0.20181 | +0.19846 | 0.00335 |
| 6B3 | omega|lambda_rep | -0.04799 | -0.04062 | 0.00737 |
| 6B3 | omega|lambda_back | -0.18471 | -0.17539 | 0.00931 |
| 6B3 | lambda_rep|lambda_back | -0.17197 | -0.17109 | 0.00088 |

## Synthetic recovery — a separate question

| stage | parameter | truth | post. mean | post. sd | 95% interval | truth inside | error in sd |
|---|---|---:|---:|---:|---|---|---:|
| 6B2 | beta | 1.50000 | 1.49207 | 0.03861 | [1.41675, 1.56705] | yes | 0.205 |
| 6B2 | omega | 1.73460 | 1.85424 | 0.12492 | [1.61795, 2.10839] | yes | 0.958 |
| 6B2 | lambda_rep | 0.80000 | 0.80009 | 0.02811 | [0.74462, 0.85582] | yes | 0.003 |
| 6B3 | beta | 1.50000 | 1.48097 | 0.03913 | [1.40521, 1.55809] | yes | 0.486 |
| 6B3 | omega | 1.73460 | 1.88731 | 0.12951 | [1.64116, 2.14781] | yes | 1.179 |
| 6B3 | lambda_rep | 0.80000 | 0.80587 | 0.02847 | [0.74988, 0.86205] | yes | 0.206 |
| 6B3 | lambda_back | 0.25000 | 0.21881 | 0.02402 | [0.17242, 0.26665] | yes | 1.298 |

Every generating value lies inside its 95% posterior interval.

## Smoke runs

| stage | checks | result |
|---|---|---|
| 6B2 | 7 checks, all of all_coordinates_move, deterministic_resume, each_coordinate_accepts, each_coordinate_rejects, no_nans_all_finite, q0_reset_and_state_reproducible, state_serialises_and_loads | **PASS** |
| 6B3 | 7 checks, all of all_coordinates_move, deterministic_resume, each_coordinate_accepts, each_coordinate_rejects, no_nans_all_finite, q0_reset_and_state_reproducible, state_serialises_and_loads | **PASS** |

## Tests

- baseline before Stage 6B2/6B3: `827 passed, 1 warning, 36 subtests passed`
- final: `910 passed, 1 warning, 36 subtests passed`

## Artifacts

- `results/mcmc_original/stage6b2_joint3_full_seed0/` — config, frozen reference, chains, scalar and joint diagnostics, recovery, report, figures
- `results/mcmc_original/stage6b2_joint3_smoke/` — smoke run
- `results/mcmc_original/stage6b3_joint4_full_seed0/` — config, frozen reference, chains, scalar and joint diagnostics, recovery, report, figures
- `results/mcmc_original/stage6b3_joint4_smoke/` — smoke run
- `results/mcmc_original/stage6b1_full_seed0/` — Stage 6B1 (unchanged, tagged `hpop-stage6b1-scalar-mcmc-v1`)

## What Stage 6B does not establish

Every posterior here is conditional on `U = U_TRUE`, on known and fixed skill
boundaries, and on `epsilon = 0.02`. Latent-`U` recovery, `rho`, unknown
boundaries, segmentation and semi-Markov FFBS are all untouched. The next stage
is latent-`U` recurrent recovery.

