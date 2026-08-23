# Stage 6B2 — joint scalar MCMC against the independent 3-D reference

Date 2026-08-11 · branch `mcmc-original-latent-poset` · commit `8c6e0dc9`
Frozen model `9ad850f22065d85f` · Python 3.13.2 · NumPy 2.4.6 · SciPy 1.18.0

**Target** — `p(beta, omega, lambda_rep | observations, U_TRUE, fixed boundaries, epsilon = 0.02, lambda_back = 0.25)**

proportional to the complete recurrent likelihood times p(beta) × p(omega) × p(lambda_rep). No interaction priors.

Sweep order: `beta -> omega -> lambda_rep`, each coordinate seeing the most recently accepted values of those before it.

`U`, `rho`, `P`, segmentation boundaries, skill labels and `epsilon` are not inferred anywhere in Stage 6B.

## Definition of done

| parameter | ref mean | MCMC mean | ref 95% | MCMC 95% | R-hat | bulk ESS | tail ESS | MCSE | KS | result |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| beta | 1.49236 | 1.49207 | [1.41724, 1.56919] | [1.41675, 1.56705] | 1.0004 | 7685 | 11357 | 0.00044 | 0.0064 | **PASS** |
| omega | 1.85401 | 1.85424 | [1.61385, 2.11178] | [1.61795, 2.10839] | 1.0002 | 13264 | 14759 | 0.00108 | 0.0094 | **PASS** |
| lambda_rep | 0.80028 | 0.80009 | [0.74485, 0.85611] | [0.74462, 0.85582] | 1.0010 | 7514 | 11432 | 0.00032 | 0.0080 | **PASS** |

## Registered gates

Stage 6B1's scalar gates, applied unchanged — not loosened because sampling is now joint — plus two calibrated joint gates.

| check | observed | threshold | result |
|---|---:|---:|---|
| beta:standardized_mean_error | 0.00753 | 0.15000 | PASS |
| beta:standardized_median_error | 0.00424 | 0.15000 | PASS |
| beta:standardized_q025_error | 0.01288 | 0.25000 | PASS |
| beta:standardized_q975_error | 0.05551 | 0.25000 | PASS |
| beta:ks_distance | 0.00638 | 0.03000 | PASS |
| beta:rhat | 1.00042 | 1.01000 | PASS |
| beta:bulk_ess | 7685.17959 | 1000.00000 | PASS |
| beta:tail_ess | 11356.73850 | 500.00000 | PASS |
| beta:acceptance_min | 0.44419 | 0.15000 | PASS |
| beta:acceptance_max | 0.44819 | 0.60000 | PASS |
| omega:standardized_mean_error | 0.00187 | 0.15000 | PASS |
| omega:standardized_median_error | 0.00635 | 0.15000 | PASS |
| omega:standardized_q025_error | 0.03254 | 0.25000 | PASS |
| omega:standardized_q975_error | 0.02685 | 0.25000 | PASS |
| omega:ks_distance | 0.00942 | 0.03000 | PASS |
| omega:rhat | 1.00022 | 1.01000 | PASS |
| omega:bulk_ess | 13264.03835 | 1000.00000 | PASS |
| omega:tail_ess | 14758.57009 | 500.00000 | PASS |
| omega:acceptance_min | 0.45981 | 0.15000 | PASS |
| omega:acceptance_max | 0.46725 | 0.60000 | PASS |
| lambda_rep:standardized_mean_error | 0.00707 | 0.15000 | PASS |
| lambda_rep:standardized_median_error | 0.00954 | 0.15000 | PASS |
| lambda_rep:standardized_q025_error | 0.00791 | 0.25000 | PASS |
| lambda_rep:standardized_q975_error | 0.01031 | 0.25000 | PASS |
| lambda_rep:ks_distance | 0.00803 | 0.03000 | PASS |
| lambda_rep:rhat | 1.00097 | 1.01000 | PASS |
| lambda_rep:bulk_ess | 7514.11333 | 1000.00000 | PASS |
| lambda_rep:tail_ess | 11431.58367 | 500.00000 | PASS |
| lambda_rep:acceptance_min | 0.43119 | 0.15000 | PASS |
| lambda_rep:acceptance_max | 0.44712 | 0.60000 | PASS |
| joint:max_correlation_error | 0.00839 | 0.04152 | PASS |
| joint:energy_distance | 0.00213 | 0.00495 | PASS |

## Dependence — the part four marginals cannot establish

| pair | reference corr | MCMC corr | abs error |
|---|---:|---:|---:|
| beta|omega | -0.15793 | -0.15849 | 0.00056 |
| beta|lambda_rep | +0.54948 | +0.55787 | 0.00839 |
| omega|lambda_rep | -0.08306 | -0.08896 | 0.00590 |

Max pairwise correlation error **0.00839** against a calibrated envelope of 0.04152 (99th percentile of reference-vs-reference at the same sample size; null mean 0.02000, sd 0.00837).

Max normalised covariance error 0.01869.

## Multivariate comparison

- statistic: energy distance on reference-standardised coordinates
- observed: **0.002126**
- calibrated envelope (99% of 40 reference-vs-reference replicates): **0.004948**
- null mean 0.002125, sd 0.000896, max 0.005426; observed z = +0.00
- sample sizes: 2000 MCMC vs 2000 reference
- **PASS**

## Acceptance by coordinate

| coordinate | total (per chain) | post burn-in (per chain) |
|---|---|---|
| beta | 0.444, 0.446, 0.444, 0.446 | 0.444, 0.447, 0.444, 0.448 |
| omega | 0.465, 0.463, 0.466, 0.458 | 0.462, 0.462, 0.467, 0.460 |
| lambda_rep | 0.436, 0.447, 0.443, 0.445 | 0.431, 0.447, 0.444, 0.442 |

## Synthetic recovery — reported separately from sampler correctness

Whether the generating value falls inside the posterior is a statement about this one finite dataset, not about the sampler. The sampler-correctness claim is the reference comparison above.

| parameter | truth | post. mean | post. median | post. sd | 95% interval | truth inside | abs error | error in sd |
|---|---:|---:|---:|---:|---|---|---:|---:|
| beta | 1.50000 | 1.49207 | 1.49194 | 0.03861 | [1.41675, 1.56705] | yes | 0.00793 | 0.205 |
| omega | 1.73460 | 1.85424 | 1.85178 | 0.12492 | [1.61795, 2.10839] | yes | 0.11964 | 0.958 |
| lambda_rep | 0.80000 | 0.80009 | 0.79996 | 0.02811 | [0.74462, 0.85582] | yes | 0.00009 | 0.003 |

## Reference provenance

- built by `scripts/stage6b_joint_reference_build.py` into `results/mcmc_original/stage6b2_joint3_full_seed0`
- 226,981 grid points, n = 61 per axis, radius 6.0 curvature sd
- integral 1.0000000000; outer-face mass 1.190e-07
- direct quadrature on the recurrent log posterior; no MCMC kernel, acceptance ratio or draw is involved

## Status

Stage 6B2 sampler correctness: **PASS**.
Stage 6B2 synthetic recovery: beta PASS, omega PASS, lambda_rep PASS.

