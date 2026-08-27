# Stage 6.0 / 6A — recurrent likelihood correctness (full, seed 0)

Date 2026-08-10 · branch `mcmc-original-latent-poset` · commit `08e2cb6d`
Python 3.13.2 · NumPy 2.4.6

**Stage 5C (full joint S+U+P) remains DEFERRED** — not required here, since this task
holds the boundaries and the latent poset fixed. **No recurrent-parameter MCMC has been
implemented**: this validates the likelihood and the generator only.

## PASS / FAIL

| check | result |
|---|---|
| Stage 6.0 deterministic recurrent equations | **PASS** |
| Stage 6.0 fixed-length normalization | **PASS** |
| Stage 6A generator-likelihood parity | **PASS** |
| Stage 6.0 empirical frequency check | **PASS** |
| Stage 6A exposure audit | **PASS** |
| Stage 6A held-out: true beats wrong antichain U | **PASS** |

## Model and seeds

- `U_TRUE` closure: [[0, 2], [0, 3], [0, 4], [2, 3], [2, 4], [3, 4]]
- role 1 incomparable with all others; `U_ANTICHAIN` induces no ordered pairs
- fixed `T = 20` — **the likelihood conditions on T**; there is no
  `p(T | skill)`, no duration and no stopping model
- beta 1.5, epsilon 0.02, kappa 0.85,
  lambda_rep 0.8, lambda_back 0.25
- seeds: train 0, held-out 10000

## Deterministic recurrent equations

- `F(q0)` = [1.0, 1.0, 0.0, 0.0, 0.0] (expected [1,1,0,0,0])
- `S(q0)` = [3, 0, 2, 1, 0] (expected [3,0,2,1,0])
- first-step parity with static BPOP: max diff **0.000e+00**

## Fixed-length normalization (the gate)

| m | T | sequences | beta | eps | kappa | l_rep | l_back | \|sum-1\| |
|---|---|---|---|---|---|---|---|---|
| 2 | 4 | 16 | 1.5 | 0.02 | 0.85 | 0.8 | 0.25 | 2.22e-16 |
| 2 | 4 | 16 | 0.0 | 0.0 | 0.5 | 0.0 | 0.0 | 2.22e-16 |
| 2 | 4 | 16 | 3.0 | 0.1 | 0.99 | 2.0 | 1.5 | 2.22e-16 |
| 2 | 4 | 16 | 1.0 | 0.05 | 0.01 | 0.0 | 1.0 | 0.00e+00 |
| 2 | 4 | 16 | 0.0 | 0.2 | 0.999 | 0.0 | 0.0 | 2.22e-16 |
| 3 | 3 | 27 | 1.5 | 0.02 | 0.85 | 0.8 | 0.25 | 1.11e-16 |
| 3 | 3 | 27 | 0.0 | 0.0 | 0.5 | 0.0 | 0.0 | 2.22e-16 |
| 3 | 3 | 27 | 3.0 | 0.1 | 0.99 | 2.0 | 1.5 | 1.11e-16 |
| 3 | 3 | 27 | 1.0 | 0.05 | 0.01 | 0.0 | 1.0 | 0.00e+00 |
| 3 | 3 | 27 | 0.0 | 0.2 | 0.999 | 0.0 | 0.0 | 0.00e+00 |
| 3 | 4 | 81 | 1.5 | 0.02 | 0.85 | 0.8 | 0.25 | 1.11e-16 |
| 3 | 4 | 81 | 0.0 | 0.0 | 0.5 | 0.0 | 0.0 | 0.00e+00 |
| 3 | 4 | 81 | 3.0 | 0.1 | 0.99 | 2.0 | 1.5 | 2.22e-16 |
| 3 | 4 | 81 | 1.0 | 0.05 | 0.01 | 0.0 | 1.0 | 1.11e-16 |
| 3 | 4 | 81 | 0.0 | 0.2 | 0.999 | 0.0 | 0.0 | 2.22e-16 |

**worst |sum − 1| = 2.220e-16** (criterion < 1e-10)

## Generator–likelihood parity

- blocks replayed: 500
- max \|log-likelihood difference\|: **1.066e-14**
- max \|q difference\|: **0.000e+00**
- max \|step log-p difference\|: **0.000e+00**

## Empirical vs analytic frequencies (m=2, T=4)

- 100,000 samples; **TV = 0.00360**, max abs error 0.00188

## Exposure audit (training blocks)

- steps 10,000 over 500 blocks
- valid-role repeats 7,701; **leaf repeats 2,809**; **upstream repeats 4,892**; **recomputations 2,167**

Gate exposure `E_zx` for the true ordered pairs:

| pair | count |
|---|---|
| 0->2 | 2113 |
| 0->3 | 1573 |
| 0->4 | 646 |
| 2->3 | 2069 |
| 2->4 | 877 |
| 3->4 | 609 |

## Held-out diagnostics

NLL per step under true parameters: **1.24316** over 200 blocks.

| perturbation | NLL | paired mean diff | s.e. | median | frac favouring true |
|---|---|---|---|---|---|
| wrong_antichain_U | 1.70554 | +0.46238 | 0.01254 | +0.45483 | 0.995 |
| beta_0 | 1.38283 | +0.13967 | 0.00837 | +0.13910 | 0.885 |
| kappa_0.05 | 1.37320 | +0.13004 | 0.00440 | +0.14032 | 0.960 |
| lambda_rep_0 | 1.30154 | +0.05838 | 0.00465 | +0.06471 | 0.815 |
| lambda_back_0 | 1.25237 | +0.00921 | 0.00198 | +0.00279 | 0.570 |
| lambda_rep_and_back_0 | 1.32148 | +0.07832 | 0.00559 | +0.08086 | 0.810 |

Only the antichain row is a hard criterion. The others are **diagnostics**:
a small mean difference is reported as small, and no scalar-parameter
identifiability is claimed — that is Stage 6B.

## Identifiability — observed block-likelihood curvature

**These are local curvature diagnostics on observed data, not recovery results.**
The quantity below is a sample average of the observed information over the
generated blocks, not an exact expectation under the generative law.

It is computed at **block** level, not per step: `kappa = sigmoid(omega)` enters the
validity-state recursion, so perturbing `omega` changes every later `q` and hence
every later action probability. Each block is therefore re-scored from `q_0 = 0`
for each candidate value; a curvature at fixed `q` measures only the direct
one-step channel and understates `omega` substantially.

| parameter | true | curvature / block | implied sd (500 blocks) | true/sd |
|---|---|---|---|---|
| beta | 1.500 | 1.9621 | 0.0319 | 47.0 |
| lambda_rep | 0.800 | 3.5835 | 0.0236 | 33.9 |
| lambda_back | 0.250 | 3.9005 | 0.0226 | 11.0 |
| omega | 1.735 | 0.1482 | 0.1162 | 14.9 |

Curvature is **not invariant under reparameterisation**, so these columns must not
be compared across parameters as if they were a ranking. For the gate, uncertainty
is reported on both scales:

- `omega` = 1.735 +/- 0.1162 (unconstrained scale)
- `kappa` = sigmoid(omega) = 0.8500, implied 95% [0.8186, 0.8768]
- `dkappa/domega` at the truth = kappa(1-kappa) = 0.1275

### How to read the small `lambda_back` ablation gap

Setting `lambda_back` from 0.25 to 0 raises held-out NLL by only +0.0092. That is
**not** evidence of weak identifiability. A zero-ablation gap depends on both the
curvature and the distance from the true value to zero, and here the true value is
itself small. Under a quadratic approximation the predicted gap is
`0.5 * I_step * delta^2 ~ 0.5 * 0.201 * 0.25^2 ~ 0.0063`, the same order as the
0.0092 observed. The two are consistent.

The defensible conclusion is the negative one: **there is no present evidence that**
**`lambda_back` or `omega` is intrinsically unidentifiable.** Whether all four
parameters are recoverable is a Stage-6B question, to be settled by exact
one-dimensional posterior profiles, then one-parameter MCMC, then MCMC-vs-grid
agreement — not by these diagnostics.

## Deviations and notes

- `lambda_rep` is a **relative penalty on currently valid candidates**, not a monotone
  sequence-level repeat cost; no monotonicity is asserted anywhere.
- The latent graph stays **acyclic**; repetition is carried by the validity state.
- Repeated sequences are never passed to the static optimized likelihood.
- No U / rho / beta / omega / lambda inference was performed.

