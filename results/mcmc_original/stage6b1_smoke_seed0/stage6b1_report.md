# Stage 6B1 — scalar MCMC against the Stage 6B0 reference posteriors (smoke)

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
| beta | 1.4961 | 1.4956 | [1.4338, 1.5590] | [1.4343, 1.5577] | 1.0021 | 2258 | 2103 | 0.0118 | **PASS** |
| omega | 1.8506 | 1.8481 | [1.6153, 2.1036] | [1.6136, 2.1037] | 1.0010 | 2095 | 2097 | 0.0178 | **PASS** |
| kappa = sigmoid(omega) | 0.8635 | 0.8632 | [0.8341, 0.8913] | [0.8339, 0.8913] | 1.0008 | 2095 | 2097 | 0.0178 | **PASS** |
| lambda_rep | 0.8032 | 0.8029 | [0.7569, 0.8495] | [0.7586, 0.8490] | 0.9998 | 2117 | 2402 | 0.0194 | **PASS** |
| lambda_back | 0.2288 | 0.2284 | [0.1851, 0.2729] | [0.1866, 0.2718] | 1.0004 | 2443 | 2446 | 0.0163 | **PASS** |

## Registered gates

R-hat < 1.05, standardized mean <= 0.35 (smoke catches implementation failures only)

| parameter | check | observed | threshold | result |
|---|---|---:|---:|---|
| beta | standardized_mean_error | 0.0154 | 0.3500 | PASS |
| beta | rhat | 1.0021 | 1.0500 | PASS |
| omega | standardized_mean_error | 0.0200 | 0.3500 | PASS |
| omega | rhat | 1.0010 | 1.0500 | PASS |
| lambda_rep | standardized_mean_error | 0.0145 | 0.3500 | PASS |
| lambda_rep | rhat | 0.9998 | 1.0500 | PASS |
| lambda_back | standardized_mean_error | 0.0173 | 0.3500 | PASS |
| lambda_back | rhat | 1.0004 | 1.0500 | PASS |

## Proposals and tuning

Proposal scales come from the observed likelihood curvature at the registered
true value, then one 2,000-iteration pilot with at most one adjustment. The
pilot draws are discarded and the reported chains restart from the registered
dispersed starts. The reference grids are never used to tune anything.

| parameter | walk | initial scale | pilot acc | adjusted | final scale | evaluations | seconds |
|---|---|---:|---:|---|---:|---:|---:|
| beta | log | 0.05109 | 0.449 | False | 0.05109 | 12,002 | 9.4 |
| omega | identity | 0.27891 | 0.471 | False | 0.27891 | 12,002 | 37.9 |
| lambda_rep | log | 0.07086 | 0.424 | False | 0.07086 | 12,002 | 9.4 |
| lambda_back | log | 0.21734 | 0.482 | False | 0.21734 | 12,002 | 9.3 |

The `log theta' - log theta` Jacobian on the log-scale walk is verified
analytically in `tests/mcmc_original/test_stage6b1_proposals.py`, and its removal
is shown to move a Gamma(2,2) stationary distribution onto Gamma(1,2) in
`test_stage6b1_scalar_mh.py`. `omega` uses a symmetric walk and carries no
correction; every proposed `omega` replays `q` from zero for all blocks.

## Per-chain detail

| parameter | chain | start | acceptance (post burn-in) | mean | sd | seconds |
|---|---:|---:|---:|---:|---:|---:|
| beta | 0 | 0.500 | 0.452 | 1.49509 | 0.03136 | 4.7 |
| beta | 1 | 2.500 | 0.443 | 1.49608 | 0.03158 | 4.7 |
| omega | 0 | -0.500 | 0.454 | 1.84450 | 0.12208 | 18.9 |
| omega | 1 | 3.500 | 0.470 | 1.85164 | 0.12599 | 19.0 |
| lambda_rep | 0 | 0.150 | 0.449 | 0.80320 | 0.02288 | 4.8 |
| lambda_rep | 1 | 1.500 | 0.439 | 0.80259 | 0.02366 | 4.6 |
| lambda_back | 0 | 0.050 | 0.458 | 0.22850 | 0.02197 | 4.7 |
| lambda_back | 1 | 0.700 | 0.474 | 0.22835 | 0.02168 | 4.6 |

## Held-out diagnostics (secondary)

Reported after fitting, and used for nothing else — not for proposal scales,
not for priors. Per-step negative log likelihood on the 200 held-out blocks.

| parameter | NLL at post. mean | NLL at post. median | NLL at truth | predictive log score |
|---|---:|---:|---:|---:|
| beta | 1.24315 | 1.24315 | 1.24316 | -1.24316 |
| omega | 1.24306 | 1.24306 | 1.24316 | -1.24306 |
| lambda_rep | 1.24314 | 1.24314 | 1.24316 | -1.24315 |
| lambda_back | 1.24342 | 1.24341 | 1.24316 | -1.24344 |

## Status

Stage 6B1: **PASS** (smoke).

