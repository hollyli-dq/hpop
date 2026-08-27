# Stage 6B3 — joint scalar MCMC against the independent 4-D reference

Date 2026-08-11 · branch `mcmc-original-latent-poset` · commit `8c6e0dc9`
Frozen model `9ad850f22065d85f` · Python 3.13.2 · NumPy 2.4.6 · SciPy 1.18.0

**Target** — `p(beta, omega, lambda_rep, lambda_back | observations, U_TRUE, fixed boundaries, epsilon = 0.02)**

proportional to the complete recurrent likelihood times p(beta) × p(omega) × p(lambda_rep) × p(lambda_back). No interaction priors.

Sweep order: `beta -> omega -> lambda_rep -> lambda_back`, each coordinate seeing the most recently accepted values of those before it.

`U`, `rho`, `P`, segmentation boundaries, skill labels and `epsilon` are not inferred anywhere in Stage 6B.

## Definition of done

| parameter | ref mean | MCMC mean | ref 95% | MCMC 95% | R-hat | bulk ESS | tail ESS | MCSE | KS | result |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| beta | 1.48205 | 1.48097 | [1.40437, 1.56170] | [1.40521, 1.55809] | 1.0007 | 7413 | 10830 | 0.00045 | 0.0156 | **PASS** |
| omega | 1.88642 | 1.88731 | [1.63548, 2.15320] | [1.64116, 2.14781] | 1.0003 | 12350 | 14560 | 0.00117 | 0.0106 | **PASS** |
| lambda_rep | 0.80660 | 0.80587 | [0.74971, 0.86425] | [0.74988, 0.86205] | 1.0003 | 8055 | 11732 | 0.00032 | 0.0159 | **PASS** |
| lambda_back | 0.21900 | 0.21881 | [0.17174, 0.26739] | [0.17242, 0.26665] | 1.0001 | 10146 | 12732 | 0.00024 | 0.0104 | **PASS** |

## Registered gates

Stage 6B1's scalar gates, applied unchanged — not loosened because sampling is now joint — plus two calibrated joint gates.

| check | observed | threshold | result |
|---|---:|---:|---|
| beta:standardized_mean_error | 0.02751 | 0.15000 | PASS |
| beta:standardized_median_error | 0.02129 | 0.15000 | PASS |
| beta:standardized_q025_error | 0.02166 | 0.25000 | PASS |
| beta:standardized_q975_error | 0.09213 | 0.25000 | PASS |
| beta:ks_distance | 0.01558 | 0.03000 | PASS |
| beta:rhat | 1.00069 | 1.01000 | PASS |
| beta:bulk_ess | 7412.83437 | 1000.00000 | PASS |
| beta:tail_ess | 10830.07647 | 500.00000 | PASS |
| beta:acceptance_min | 0.44081 | 0.15000 | PASS |
| beta:acceptance_max | 0.44669 | 0.60000 | PASS |
| omega:standardized_mean_error | 0.00681 | 0.15000 | PASS |
| omega:standardized_median_error | 0.00495 | 0.15000 | PASS |
| omega:standardized_q025_error | 0.04369 | 0.25000 | PASS |
| omega:standardized_q975_error | 0.04146 | 0.25000 | PASS |
| omega:ks_distance | 0.01056 | 0.03000 | PASS |
| omega:rhat | 1.00028 | 1.01000 | PASS |
| omega:bulk_ess | 12350.32195 | 1000.00000 | PASS |
| omega:tail_ess | 14560.25054 | 500.00000 | PASS |
| omega:acceptance_min | 0.46237 | 0.15000 | PASS |
| omega:acceptance_max | 0.47231 | 0.60000 | PASS |
| lambda_rep:standardized_mean_error | 0.02564 | 0.15000 | PASS |
| lambda_rep:standardized_median_error | 0.02431 | 0.15000 | PASS |
| lambda_rep:standardized_q025_error | 0.00618 | 0.25000 | PASS |
| lambda_rep:standardized_q975_error | 0.07721 | 0.25000 | PASS |
| lambda_rep:ks_distance | 0.01591 | 0.03000 | PASS |
| lambda_rep:rhat | 1.00034 | 1.01000 | PASS |
| lambda_rep:bulk_ess | 8055.33107 | 1000.00000 | PASS |
| lambda_rep:tail_ess | 11731.60310 | 500.00000 | PASS |
| lambda_rep:acceptance_min | 0.43462 | 0.15000 | PASS |
| lambda_rep:acceptance_max | 0.44169 | 0.60000 | PASS |
| lambda_back:standardized_mean_error | 0.00765 | 0.15000 | PASS |
| lambda_back:standardized_median_error | 0.00734 | 0.15000 | PASS |
| lambda_back:standardized_q025_error | 0.02843 | 0.25000 | PASS |
| lambda_back:standardized_q975_error | 0.03107 | 0.25000 | PASS |
| lambda_back:ks_distance | 0.01040 | 0.03000 | PASS |
| lambda_back:rhat | 1.00006 | 1.01000 | PASS |
| lambda_back:bulk_ess | 10146.03704 | 1000.00000 | PASS |
| lambda_back:tail_ess | 12732.49323 | 500.00000 | PASS |
| lambda_back:acceptance_min | 0.47194 | 0.15000 | PASS |
| lambda_back:acceptance_max | 0.47906 | 0.60000 | PASS |
| joint:max_correlation_error | 0.01539 | 0.05623 | PASS |
| joint:energy_distance | 0.00314 | 0.00399 | PASS |

## Dependence — the part four marginals cannot establish

| pair | reference corr | MCMC corr | abs error |
|---|---:|---:|---:|
| beta|omega | -0.18815 | -0.17276 | 0.01539 |
| beta|lambda_rep | +0.49434 | +0.49821 | 0.00387 |
| beta|lambda_back | +0.20181 | +0.19846 | 0.00335 |
| omega|lambda_rep | -0.04799 | -0.04062 | 0.00737 |
| omega|lambda_back | -0.18471 | -0.17539 | 0.00931 |
| lambda_rep|lambda_back | -0.17197 | -0.17109 | 0.00088 |

Max pairwise correlation error **0.01539** against a calibrated envelope of 0.05623 (99th percentile of reference-vs-reference at the same sample size; null mean 0.02830, sd 0.01148).

Max normalised covariance error 0.01620.

## Multivariate comparison

- statistic: energy distance on reference-standardised coordinates
- observed: **0.003142**
- calibrated envelope (99% of 40 reference-vs-reference replicates): **0.003990**
- null mean 0.002427, sd 0.000760, max 0.004055; observed z = +0.94
- sample sizes: 2000 MCMC vs 2000 reference
- **PASS**

## Acceptance by coordinate

| coordinate | total (per chain) | post burn-in (per chain) |
|---|---|---|
| beta | 0.444, 0.447, 0.447, 0.443 | 0.441, 0.445, 0.447, 0.447 |
| omega | 0.472, 0.459, 0.469, 0.471 | 0.472, 0.462, 0.472, 0.472 |
| lambda_rep | 0.436, 0.440, 0.440, 0.436 | 0.436, 0.442, 0.437, 0.435 |
| lambda_back | 0.478, 0.476, 0.474, 0.474 | 0.479, 0.477, 0.476, 0.472 |

## Synthetic recovery — reported separately from sampler correctness

Whether the generating value falls inside the posterior is a statement about this one finite dataset, not about the sampler. The sampler-correctness claim is the reference comparison above.

| parameter | truth | post. mean | post. median | post. sd | 95% interval | truth inside | abs error | error in sd |
|---|---:|---:|---:|---:|---|---|---:|---:|
| beta | 1.50000 | 1.48097 | 1.48093 | 0.03913 | [1.40521, 1.55809] | yes | 0.01903 | 0.486 |
| omega | 1.73460 | 1.88731 | 1.88411 | 0.12951 | [1.64116, 2.14781] | yes | 0.15270 | 1.179 |
| lambda_rep | 0.80000 | 0.80587 | 0.80584 | 0.02847 | [0.74988, 0.86205] | yes | 0.00587 | 0.206 |
| lambda_back | 0.25000 | 0.21881 | 0.21865 | 0.02402 | [0.17242, 0.26665] | yes | 0.03119 | 1.298 |

## Reference provenance

- built by `scripts/stage6b_joint_reference_build.py` into `results/mcmc_original/stage6b3_joint4_full_seed0`
- 6,765,201 grid points, n = 51 per axis, radius 9.0 curvature sd
- integral 1.0000000000; outer-face mass 2.775e-09
- direct quadrature on the recurrent log posterior; no MCMC kernel, acceptance ratio or draw is involved

## Status

Stage 6B3 sampler correctness: **PASS**.
Stage 6B3 synthetic recovery: beta PASS, omega PASS, lambda_rep PASS, lambda_back PASS.

