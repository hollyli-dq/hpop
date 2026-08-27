# Stage 6B1 — scalar MCMC against the Stage 6B0 reference posteriors (full)

Date 2026-08-11 · branch `mcmc-original-latent-poset` · commit `f5497308`
Python 3.13.2 · NumPy 2.4.6 · SciPy 1.18.0

Each parameter is inferred **alone**, with `U`, `epsilon` and the other three
scalars held at their registered true values. No `U` / `rho` / latent-dimension
updates, and no dimension-proportional schedule — those are Stage 6C.
**Stage 5C (full joint S+U+P) remains DEFERRED.**

The claim being tested is not *truth in interval*. It is
`p_MCMC(theta | D) ~= p_grid(theta | D)`, measured by standardized mean, median
and interval-endpoint errors and a KS distance against the immutable normalized
reference CDF in `stage6b_full_seed0/reference_posteriors.json`.

## Definition of done

| parameter | grid mean | MCMC mean | grid 95% | MCMC 95% | R-hat | bulk ESS | tail ESS | KS | result |
|---|---:|---:|---|---|---:|---:|---:|---:|---|
| beta | 1.4961 | 1.4964 | [1.4338, 1.5590] | [1.4338, 1.5584] | 1.0003 | 14165 | 15126 | 0.0118 | **PASS** |
| omega | 1.8506 | 1.8494 | [1.6153, 2.1036] | [1.6176, 2.1035] | 1.0004 | 14498 | 15892 | 0.0074 | **PASS** |
| kappa = sigmoid(omega) | 0.8635 | 0.8634 | [0.8341, 0.8913] | [0.8345, 0.8912] | 1.0004 | 14498 | 15892 | 0.0074 | **PASS** |
| lambda_rep | 0.8032 | 0.8034 | [0.7569, 0.8495] | [0.7571, 0.8497] | 1.0002 | 13935 | 15444 | 0.0049 | **PASS** |
| lambda_back | 0.2288 | 0.2288 | [0.1851, 0.2729] | [0.1851, 0.2729] | 1.0002 | 13739 | 14231 | 0.0082 | **PASS** |

## Registered gates

standardized mean <= 0.15, median <= 0.15, interval endpoints <= 0.25, KS <= 0.03, R-hat <= 1.01, bulk ESS >= 1000, tail ESS >= 500, acceptance in [0.15, 0.60]

| parameter | check | observed | threshold | result |
|---|---|---:|---:|---|
| beta | standardized_mean_error | 0.0093 | 0.1500 | PASS |
| beta | standardized_median_error | 0.0214 | 0.1500 | PASS |
| beta | standardized_q025_error | 0.0004 | 0.2500 | PASS |
| beta | standardized_q975_error | 0.0175 | 0.2500 | PASS |
| beta | ks_distance | 0.0118 | 0.0300 | PASS |
| beta | rhat | 1.0003 | 1.0100 | PASS |
| beta | bulk_ess | 14165.3098 | 1000.0000 | PASS |
| beta | tail_ess | 15126.0743 | 500.0000 | PASS |
| beta | min_acceptance_rate | 0.4358 | 0.1500 | PASS |
| beta | max_acceptance_rate | 0.4446 | 0.6000 | PASS |
| omega | standardized_mean_error | 0.0090 | 0.1500 | PASS |
| omega | standardized_median_error | 0.0123 | 0.1500 | PASS |
| omega | standardized_q025_error | 0.0185 | 0.2500 | PASS |
| omega | standardized_q975_error | 0.0005 | 0.2500 | PASS |
| omega | ks_distance | 0.0074 | 0.0300 | PASS |
| omega | rhat | 1.0004 | 1.0100 | PASS |
| omega | bulk_ess | 14498.2504 | 1000.0000 | PASS |
| omega | tail_ess | 15892.4219 | 500.0000 | PASS |
| omega | min_acceptance_rate | 0.4608 | 0.1500 | PASS |
| omega | max_acceptance_rate | 0.4674 | 0.6000 | PASS |
| lambda_rep | standardized_mean_error | 0.0055 | 0.1500 | PASS |
| lambda_rep | standardized_median_error | 0.0060 | 0.1500 | PASS |
| lambda_rep | standardized_q025_error | 0.0104 | 0.2500 | PASS |
| lambda_rep | standardized_q975_error | 0.0065 | 0.2500 | PASS |
| lambda_rep | ks_distance | 0.0049 | 0.0300 | PASS |
| lambda_rep | rhat | 1.0002 | 1.0100 | PASS |
| lambda_rep | bulk_ess | 13935.4607 | 1000.0000 | PASS |
| lambda_rep | tail_ess | 15443.9976 | 500.0000 | PASS |
| lambda_rep | min_acceptance_rate | 0.4372 | 0.1500 | PASS |
| lambda_rep | max_acceptance_rate | 0.4514 | 0.6000 | PASS |
| lambda_back | standardized_mean_error | 0.0024 | 0.1500 | PASS |
| lambda_back | standardized_median_error | 0.0095 | 0.1500 | PASS |
| lambda_back | standardized_q025_error | 0.0014 | 0.2500 | PASS |
| lambda_back | standardized_q975_error | 0.0040 | 0.2500 | PASS |
| lambda_back | ks_distance | 0.0082 | 0.0300 | PASS |
| lambda_back | rhat | 1.0002 | 1.0100 | PASS |
| lambda_back | bulk_ess | 13738.5031 | 1000.0000 | PASS |
| lambda_back | tail_ess | 14230.7796 | 500.0000 | PASS |
| lambda_back | min_acceptance_rate | 0.4609 | 0.1500 | PASS |
| lambda_back | max_acceptance_rate | 0.4721 | 0.6000 | PASS |

## Proposals and tuning

Proposal scales come from the observed likelihood curvature at the registered
true value, then one 2,000-iteration pilot with at most one adjustment. The
pilot draws are discarded and the reported chains restart from the registered
dispersed starts. The reference grids are never used to tune anything.

| parameter | walk | initial scale | pilot acc | adjusted | final scale | evaluations | seconds |
|---|---|---:|---:|---|---:|---:|---:|
| beta | log | 0.05109 | 0.449 | False | 0.05109 | 80,004 | 63.6 |
| omega | identity | 0.27891 | 0.471 | False | 0.27891 | 80,004 | 274.7 |
| lambda_rep | log | 0.07086 | 0.424 | False | 0.07086 | 80,004 | 63.5 |
| lambda_back | log | 0.21734 | 0.482 | False | 0.21734 | 80,004 | 61.1 |

The `log theta' - log theta` Jacobian on the log-scale walk is verified
analytically in `tests/mcmc_original/test_stage6b1_proposals.py`, and its removal
is shown to move a Gamma(2,2) stationary distribution onto Gamma(1,2) in
`test_stage6b1_scalar_mh.py`. `omega` uses a symmetric walk and carries no
correction; every proposed `omega` replays `q` from zero for all blocks.

## Per-chain detail

| parameter | chain | start | acceptance (post burn-in) | mean | sd | seconds |
|---|---:|---:|---:|---:|---:|---:|
| beta | 0 | 0.500 | 0.442 | 1.49621 | 0.03118 | 15.8 |
| beta | 1 | 1.000 | 0.445 | 1.49661 | 0.03194 | 15.9 |
| beta | 2 | 2.500 | 0.436 | 1.49601 | 0.03176 | 16.2 |
| beta | 3 | 4.000 | 0.436 | 1.49666 | 0.03173 | 15.7 |
| omega | 0 | -0.500 | 0.461 | 1.85163 | 0.12525 | 68.7 |
| omega | 1 | 0.800 | 0.467 | 1.84868 | 0.12401 | 70.4 |
| omega | 2 | 3.500 | 0.462 | 1.84756 | 0.12175 | 70.4 |
| omega | 3 | 5.000 | 0.467 | 1.84988 | 0.12506 | 65.1 |
| lambda_rep | 0 | 0.150 | 0.451 | 0.80317 | 0.02368 | 15.5 |
| lambda_rep | 1 | 0.500 | 0.442 | 0.80326 | 0.02408 | 15.6 |
| lambda_rep | 2 | 1.500 | 0.437 | 0.80323 | 0.02327 | 16.2 |
| lambda_rep | 3 | 3.000 | 0.438 | 0.80380 | 0.02393 | 16.2 |
| lambda_back | 0 | 0.050 | 0.465 | 0.22873 | 0.02256 | 15.2 |
| lambda_back | 1 | 0.200 | 0.466 | 0.22848 | 0.02188 | 15.4 |
| lambda_back | 2 | 0.700 | 0.472 | 0.22890 | 0.02216 | 15.2 |
| lambda_back | 3 | 1.500 | 0.461 | 0.22893 | 0.02242 | 15.2 |

## Held-out diagnostics (secondary)

Reported after fitting, and used for nothing else — not for proposal scales,
not for priors. Per-step negative log likelihood on the 200 held-out blocks.

| parameter | NLL at post. mean | NLL at post. median | NLL at truth | predictive log score |
|---|---:|---:|---:|---:|
| beta | 1.24315 | 1.24315 | 1.24316 | -1.24316 |
| omega | 1.24306 | 1.24306 | 1.24316 | -1.24308 |
| lambda_rep | 1.24314 | 1.24314 | 1.24316 | -1.24315 |
| lambda_back | 1.24341 | 1.24342 | 1.24316 | -1.24342 |

## Status

Stage 6B1: **PASS** (full).

Next: Stage 6B2 — joint `beta`, `omega`, `lambda_rep` with `lambda_back`
fixed at 0.25, which is where posterior *correlation* first appears.

