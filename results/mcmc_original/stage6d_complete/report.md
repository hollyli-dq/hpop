# Stage 6D — the joint oracle-block sampler, complete report

Model `recurrent-rfs-oracle-joint-latent-v1`, configuration hash `ebd5effd5b32e68aeb61df044bfbdcbef998c01955673ae1c077531be3d2ac73`, source commit `0905a16b5bfdf1b7e52dc9a99954bc23453b10e4`.
Assembled 2026-08-12T15:11:59+00:00.

Stage 6D infers `U`, `rho`, `beta`, `omega`, `lambda_rep` and `lambda_back` jointly, on oracle block boundaries and oracle skill labels. The state is the real matrix `U in R^{5x2}`; `H = h(U)` is derived, never state, and carries no second prior. Read [`model_audit.md`](model_audit.md) first — it settles the model and overrides any brief that assumes an assessor hierarchy, a `tau`, or a discrete-poset state.

## 1. The three integers that are easy to conflate

| symbol | value | meaning |
|---|---:|---|
| `m` | 5 | rows of `U` — role occurrences |
| `d` | 2 | latent columns of `U` — what the Stage 6D brief writes as `K` |
| `K` | 1 | skills — one `U` matrix; the repository's meaning of `K` |
| assessors | 0 | there is no assessor level and no `tau` |

## 2. Divergences from the brief, and the clause that resolves each

| § | brief assumes | frozen reality | resolution |
|---|---|---|---|
| 2 | U^(0)/U^(a) assessor hierarchy with a tau dependence | a single U in R^{5x2}; no assessor level, no tau | use the frozen state, as the brief's own fallback clause directs; the assessor-residual density and the hierarchical QMC construction have no referent and are not built |
| 4 | rho ~ Beta(1, 1/6) | rho ~ Beta(1, 1) truncated at 1 - 5e-3, i.e. Uniform(0, 0.995), from StatisticalUtils.dRprior with stage5.RHO_PRIOR = 1.0 | the brief defers to the final Stage 6C configuration, which establishes this prior |
| 4 | rho scaling proposal delta ~ U(d_r, 1/d_r) with ratio -log(delta) | logit random walk carrying log(rho(1-rho)) | sections 6 and 7.2 require reusing the frozen Stage 6C rho update and proving numerical equality with it; adopting the scaling proposal would fail that gate. The scaling proposal's density ratio is implemented and tested as a non-production utility so the identity is still pinned |

The scaling proposal is implemented and tested as a **non-production utility** (`stage6d_frozen.scaling_proposal_log_ratio`, the exact `-log(delta)` identity), so the mathematics §4 asks about is pinned without displacing the kernel §6 and §7.2 require parity with.

## 3. Stage 6D0 — the joint smoke, and kernel parity

20 checks on 60 blocks over 400 sweeps, all passing: every coordinate moves and also rejects, `q_0` is reset at the start of every block, the direct and cached targets agree, a rejected proposal cannot disturb a valid cache, and the chain serialises and resumes bit-identically.

A sweep replays exactly `m + 1` times (m U rows + 1 omega = 6): one complete replay per `U` row and one for `omega`. `rho` consumes **zero** likelihood evaluations, because it acts only through `p(U | rho)`; `beta`, `lambda_rep` and `lambda_back` are scored from the `(H, omega)`-keyed cache.

Kernel parity with both parents, computed by reconstructing each parent's acceptance ratio from that parent's own objects:

| parent | coordinate | maximum discrepancy | tolerance |
|---|---|---:|---:|
| Stage 6B | the four scalars | 4.55e-13 | 7.3e-12 |
| Stage 6C | `U` | exactly 0.0 | 1e-9 |
| Stage 6C | `rho` | 2.13e-13 | 1e-9 |

## 4. The Stage 6D1 reference — independent, and frozen before any chain ran

The reference shares no code path with the transition kernel. It evaluates the same direct target by scrambled-Sobol importance sampling in **prior coordinates**, building `U = Z L(rho)^T` non-centred, so the unnormalised weight collapses to the likelihood alone. That removes the Gaussian determinant from the weight entirely — a determinant error in the sampler cannot hide behind the same error in the reference. The centred density is checked against the construction separately.

- 524,288 points x 32 independent scrambles
- log evidence -15.247029 (sd across replicates 1.85e-03)
- relative ESS 0.0812, maximum normalised weight 8.45e-05

| primary gate | value | threshold | verdict |
|---|---:|---:|---|
| max_rqmc_standard_error | 7.913e-04 | 0.001000 | PASS |
| max_structural_half_width_95 | 5.326e-04 | 0.002500 | PASS |

| secondary diagnostic (**not a gate**) | value | threshold | verdict |
|---|---:|---:|---|
| max_replicate_h_total_variation | 0.003529 | 0.003000 | **FAIL** |
| max_replicate_relation_departure | 0.002293 | 0.003000 | PASS |
| min_relative_ess | 0.080878 | 0.020000 | PASS |
| max_normalised_weight | 9.212e-05 | 0.001000 | PASS |
| log_evidence_sd | 0.001854 | 0.050000 | PASS |

**A superseded statistic, kept visible and still failing.** The reference was first registered on the maximum departure of any single replicate from the replicate mean. That statistic estimates the dispersion of *one* replicate, so it does not shrink as `R` grows — it samples further into the tail — and it is not an uncertainty for the quantity the comparison actually consumes, which is the replicate *mean*. Doubling `N` from `2^18` to `2^19` left it essentially unchanged (1.704e-3 to 1.727e-3) while the log-evidence standard deviation fell as expected: the gate was measuring the wrong quantity, not detecting an inadequate reference. The registered gate is now `rqmc_se = sd/sqrt(R)`, superseded **before any MCMC comparison existed**.

The superseded statistics are still computed on this run and still fail their old thresholds. That is reported as a failure of a retired statistic, not relabelled as a pass:

| retired statistic | value | old threshold | old verdict |
|---|---:|---:|---|
| max_h_probability_spread | 0.003462 | 0.001000 | **FAIL** |
| max_relation_marginal_spread | 0.002293 | 0.001000 | **FAIL** |

`all_pass` is no and `primary_pass` is yes. The reference was frozen on `primary_pass`, and the distinction is kept visible rather than collapsed.

## 5. §G — the three Stage 6D1 attempts, side by side

Stage 6D1 did not pass on the first run, and the two failures are the substantive finding of this stage: **all four scalar proposal scales were 16-32x too small**, because the registered Stage 6B scales had been tuned on the 500-block corpus and Stage 6D1's reference model is three blocks of `T = 5`, where the posterior is far broader. No gate was relaxed at any point.

| attempt | scales (`beta`/`omega`/`lambda_rep`/`lambda_back`) | sweeps | outcome | failed gates |
|---|---|---:|---|---|
| attempt 1 — original scales, 50,000 sweeps | 0.05109 / 0.18213 / 0.04155 / 0.09472 | 50,000 | FAILED | beta_rhat, lambda_rep_rhat |
| attempt 1 — original scales, 100,000 ceiling | 0.05109 / 0.18213 / 0.04155 / 0.09472 | 100,000 | FAILED | omega_rhat |
| attempt 2 — omega x32, 50,000 sweeps *(re-execution — see §5.1)* | 0.05109 / 5.8282 / 0.04155 / 0.09472 | 50,000 | FAILED | induced_h_total_variation, beta_rhat, lambda_rep_rhat, lambda_back_rhat *(of the re-execution; the original's recorded result is beta_rhat 1.03094)* |
| attempt 3 — beta/lambda retuned, 50,000 sweeps | 1.6349 / 5.8282 / 1.3296 / 1.5155 | 50,000 | PASSED | none |

### 5.1 Every §G statistic, per attempt

**attempt 1 — original scales, 50,000 sweeps** — FAILED

`results/mcmc_original/stage6d1_joint_mcmc_FAILED_attempt0_50k`, 40,000 retained pooled draws, base seed 0.

| coordinate | scale | acceptance | R-hat | bulk ESS | tail ESS | MCSE |
|---|---:|---:|---:|---:|---:|---:|
| `rho` | 0.50000 | 0.863 | 1.00068 | 2405 | 3907 | 0.005488 |
| `beta` | 0.05109 | 0.977 | 1.01184 | 181 | 192 | 0.042931 |
| `omega` | 0.18213 | 0.969 | 1.00858 | 308 | 740 | 0.113469 |
| `lambda_rep` | 0.04155 | 0.972 | 1.01457 | 212 | 356 | 0.043986 |
| `lambda_back` | 0.09472 | 0.959 | 1.00424 | 512 | 581 | 0.026826 |

log-posterior R-hat 1.00146 · induced-`H` TV 0.00800 · max relation-marginal error 0.00645 · mixed statistic 0.01022 (envelope 0.01083) · `H` states visited 10

**attempt 1 — original scales, 100,000 ceiling** — FAILED

`results/mcmc_original/stage6d1_joint_mcmc_FAILED_attempt1`, 90,000 retained pooled draws, base seed 0.

| coordinate | scale | acceptance | R-hat | bulk ESS | tail ESS | MCSE |
|---|---:|---:|---:|---:|---:|---:|
| `rho` | 0.50000 | 0.865 | 1.00078 | 5396 | 7642 | 0.003672 |
| `beta` | 0.05109 | 0.975 | 1.00207 | 413 | 526 | 0.029606 |
| `omega` | 0.18213 | 0.969 | 1.01205 | 650 | 1513 | 0.077561 |
| `lambda_rep` | 0.04155 | 0.972 | 1.00337 | 366 | 376 | 0.034882 |
| `lambda_back` | 0.09472 | 0.959 | 1.00443 | 1322 | 1668 | 0.016647 |

log-posterior R-hat 1.00097 · induced-`H` TV 0.00665 · max relation-marginal error 0.00628 · mixed statistic 0.00699 (envelope 0.01083) · `H` states visited 10

**attempt 2 — omega x32, 50,000 sweeps** — FAILED

`results/mcmc_original/stage6d1_joint_mcmc_FAILED_attempt2_omega_retuned_REEXECUTED`, 40,000 retained pooled draws, base seed 0.

**Re-execution, not the original chain.** The omega-retuned attempt's directory was overwritten by the third attempt's rerun, and its base seed was never written down. Re-running the recorded configuration (omega x32 = 5.82816, the other three scalars at their registered values, 50,000 sweeps) at each seed that appears anywhere in the Stage 6D1 record — [6020000, 0] — reproduces the attempt's *finding* but not its exact numbers, so no seed can be claimed as the original. These artifacts are the configuration re-run at base seed 0; the numbers recorded at the time are listed beside them and are what the report cites as the attempt's own result. Nothing here is presented as the original chain.

| gate | recorded at the time (the attempt's own result) | this re-execution |
|---|---:|---:|
| omega_rhat | 1.00006 | 1.00007 |
| beta_rhat | 1.03094 | 1.02910 |

| coordinate | scale | acceptance | R-hat | bulk ESS | tail ESS | MCSE |
|---|---:|---:|---:|---:|---:|---:|
| `rho` | 0.50000 | 0.865 | 1.00096 | 2365 | 3607 | 0.005556 |
| `beta` | 0.05109 | 0.974 | 1.02910 | 141 | 179 | 0.049813 |
| `omega` | 5.82816 | 0.370 | 1.00007 | 17134 | 27991 | 0.015141 |
| `lambda_rep` | 0.04155 | 0.971 | 1.01697 | 198 | 148 | 0.044563 |
| `lambda_back` | 0.09472 | 0.958 | 1.01973 | 468 | 458 | 0.028185 |

log-posterior R-hat 1.00262 · induced-`H` TV 0.01124 · max relation-marginal error 0.00799 · mixed statistic 0.01003 (envelope 0.01083) · `H` states visited 9

**attempt 3 — beta/lambda retuned, 50,000 sweeps** — PASSED

`results/mcmc_original/stage6d1_joint_mcmc`, 40,000 retained pooled draws, base seed 6020000.

| coordinate | scale | acceptance | R-hat | bulk ESS | tail ESS | MCSE |
|---|---:|---:|---:|---:|---:|---:|
| `rho` | 0.50000 | 0.865 | 1.00193 | 2638 | 3820 | 0.005235 |
| `beta` | 1.63488 | 0.438 | 0.99994 | 21217 | 26010 | 0.004037 |
| `omega` | 5.82816 | 0.371 | 1.00001 | 25840 | 29988 | 0.012401 |
| `lambda_rep` | 1.32960 | 0.407 | 1.00003 | 22331 | 23543 | 0.004259 |
| `lambda_back` | 1.51552 | 0.488 | 1.00004 | 25302 | 24882 | 0.003920 |

log-posterior R-hat 1.00160 · induced-`H` TV 0.00873 · max relation-marginal error 0.00324 · mixed statistic 0.00565 (envelope 0.01083) · `H` states visited 9

### 5.2 What each retuning bought

Bulk ESS by attempt, read from each attempt's own artifacts rather than from a hand-copied summary:

| coordinate | attempt 1 (original scales, 50,000 sweeps) | attempt 1 (original scales, 100,000 ceiling) | attempt 2 (omega x32, 50,000 sweeps) | attempt 3 (beta/lambda retuned, 50,000 sweeps) |
|---|---|---|---|---|
| `rho` | 2405 | 5396 | 2365 | 2638 |
| `beta` | 181 | 413 | 141 | 21217 |
| `omega` | 308 | 650 | 17134 | 25840 |
| `lambda_rep` | 212 | 366 | 198 | 22331 |
| `lambda_back` | 512 | 1322 | 468 | 25302 |

The summary recorded at the time, whose 'before' column is the worst observed value for each coordinate across the failing attempts:

| coordinate | bulk ESS, registered scales | bulk ESS, pilot scales | factor |
|---|---:|---:|---:|
| `beta` | 349 | 21,217 | 61x |
| `omega` | 650 | 25,840 | 40x |
| `lambda_rep` | 528 | 22,331 | 42x |
| `lambda_back` | 1,143 | 25,302 | 22x |

All four scalars were 16-32x under-scaled. The registered Stage 6B scales were tuned on the 500-block corpus, where the posterior is far tighter than on this deliberately small 3-block reference model.

Both pilots were **efficiency-only**: acceptance, ESJD, finite-target checks, invalid-proposal counts and replay/cache consistency, and nothing else. Neither loaded the reference, the truth, or any recovery or R-hat statistic, and every pilot draw was discarded. ESJD for `beta`, `lambda_rep` and `lambda_back` is measured in **log** space, because `PROPOSAL_KIND` registers them as log random walks; measuring in raw parameter space rewards large absolute moves at large parameter values and systematically selects scales that are too big.

## 6. Stage 6D1 — sampler correctness: PASS

| gate | value | threshold | verdict |
|---|---:|---:|---|
| induced_h_total_variation | 0.00873 | 0.0100 | PASS |
| max_relation_marginal_error | 0.00324 | 0.0100 | PASS |
| mixed_reference_envelope | 0.00565 | 0.0108 | PASS |
| rho_rhat | 1.00193 | 1.0100 | PASS |
| beta_rhat | 0.99994 | 1.0100 | PASS |
| omega_rhat | 1.00001 | 1.0100 | PASS |
| lambda_rep_rhat | 1.00003 | 1.0100 | PASS |
| lambda_back_rhat | 1.00004 | 1.0100 | PASS |
| log_posterior_rhat | 1.00160 | 1.0100 | PASS |
| relation_count_rhat | 1.00045 | 1.0100 | PASS |
| uncertain_relation_rhat | 1.00050 | 1.0100 | PASS |

All eleven registered gates pass simultaneously at the initial 50,000 sweeps; no continuation was needed and the 100,000 ceiling was not approached.

## 7. The Stage 6D2 pilot — scales are a property of the corpus

The Stage 6D1 scales were **not** carried forward. They were selected on a three-block model whose posterior is deliberately broad; the Stage 6D2 corpus is 500 blocks of `T = 20` and its posterior is much tighter, so those multipliers were expected to be far too large. The registered Stage 6B scales were equally unverified here, because they were tuned with `U` pinned at the truth and the other scalars fixed. A separate pilot was run over a multiplier grid symmetric about 1, covering all six coordinates in production sweep order.

| coordinate | base scale | selected multiplier | selected scale | median acceptance | ESJD space |
|---|---:|---:|---:|---:|---|
| `U` | 0.50000 | x1 | 0.500000 | 0.322 | identity |
| `rho` | 0.50000 | x8 | 4.000000 | 0.329 | logit |
| `beta` | 0.05109 | x1 | 0.051090 | 0.446 | log |
| `omega` | 0.18213 | x2 | 0.364260 | 0.379 | identity |
| `lambda_rep` | 0.04155 | x2 | 0.083100 | 0.396 | log |
| `lambda_back` | 0.09472 | x2 | 0.189440 | 0.518 | log |

Joint confirmation over all six tuned coordinates: median acceptance `U` 0.320, `rho` 0.328, `beta` 0.439, `omega` 0.383, `lambda_rep` 0.407, `lambda_back` 0.518 — all inside the registered band [0.2, 0.65]. PASS.

## 8. Stage 6D2 — the full oracle-block synthetic run

4 chains x 30,000 sweeps, 10,000 burn-in, thin 5, 16,000 retained pooled draws, 20.7 minutes wall. Starts are dispersed in every coordinate: four contrasting `H` structures, `rho` across its support, and the four scalars at prior quantiles arranged by a fixed Latin square.

| chain | start `U` | start relations | start `rho` | seed |
|---|---|---:|---:|---:|
| 0 | antichain | 0 | 0.05 | 0 |
| 1 | total_order | 10 | 0.30 | 1 |
| 2 | sparse | 3 | 0.60 | 2 |
| 3 | dense | 10 | 0.90 | 3 |

### 8.1 §15 — convergence

| coordinate | posterior mean | sd | acceptance | R-hat | bulk ESS | tail ESS | MCSE / sd |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rho` | 0.31865 | 0.21769 | 0.328 | 1.00091 | 4874 | 5527 | 0.0143 |
| `beta` | 1.48337 | 0.03926 | 0.444 | 1.00069 | 8344 | 11124 | 0.0109 |
| `omega` | 1.88510 | 0.12924 | 0.386 | 1.00024 | 11822 | 12719 | 0.0092 |
| `lambda_rep` | 0.80747 | 0.02845 | 0.389 | 1.00090 | 8365 | 10940 | 0.0109 |
| `lambda_back` | 0.21906 | 0.02398 | 0.519 | 1.00023 | 10122 | 11233 | 0.0099 |
| log target | -12459.095 | — | — | 1.00290 | 2022 | 3790 | — |

**The relation-count trace is constant at 6.0000, and is reported as `degenerate`, not as an R-hat of 1.0.** This is the expected finding, not a defect: Stage 6C established that on this corpus the induced-order posterior is a point mass at the true order, and Stage 6D2 confirms it survives freeing `omega`, `lambda_rep` and `lambda_back`. 1 induced order(s) were visited across all chains, and no relation varies, so there is no per-relation R-hat to compute.

### 8.2 §16 — parent consistency: is anything confounded?

The recurrent likelihood reads `U` only through `h(U)`. If the induced order is the point mass Stage 6C found, then the likelihood the four scalars see is *identical* to the one Stage 6B3 saw with `U` pinned at `U_TRUE`. The marginals must therefore agree, and a disagreement would be `U`-`beta`, `H`-`omega` or `rho`-`U` confounding.

| coordinate | Stage 6D2 | parent | parent stage | difference in parent sd |
|---|---|---|---|---:|
| `beta` | 1.48337 ± 0.03926 | 1.48097 ± 0.03913 | 6B3 (`U` at truth) | +0.0613 |
| `omega` | 1.88510 ± 0.12924 | 1.88731 ± 0.12951 | 6B3 (`U` at truth) | -0.0171 |
| `lambda_rep` | 0.80747 ± 0.02845 | 0.80587 ± 0.02847 | 6B3 (`U` at truth) | +0.0562 |
| `lambda_back` | 0.21906 ± 0.02398 | 0.21881 ± 0.02402 | 6B3 (`U` at truth) | +0.0105 |
| `beta` | 1.48337 ± 0.03926 | 1.49648 ± 0.03170 | 6C2 — *reported, not a gate* | -0.4137 |
| `rho` | 0.31865 ± 0.21769 | 0.32308 ± 0.22159 | 6C2 | -0.0200 |

**Why `beta` against Stage 6C2 is reported rather than gated.** Stage 6C2 held `omega`, `lambda_rep` and `lambda_back` at their registered values while Stage 6B3 and Stage 6D2 marginalise over them, and `beta` is correlated with those three. The two parents therefore already disagree with *each other* by +0.3964 Stage 6B3 sd, so no single Stage 6D2 value could satisfy a 0.25 sd gate against both, and requiring agreement with a differently conditioned posterior would be a gate on a quantity that is not supposed to be equal. This was decided and recorded in the gate registration **before any Stage 6D2 draw existed**. The contrast is kept because it measures something real: the gap between the two parents is itself the size of the effect of conditioning on omega and the lambdas rather than marginalising over them; it is not a defect in either.

Structure: Stage 6C2 placed probability 1.0000 on order #4002 with `omega` fixed; Stage 6D2 places 1.0000 on the same order with `omega` free. Freeing the three remaining scalars does not move the structure.

- **U-beta**: beta agreeing with Stage 6B3, which pinned U at the truth while marginalising over the same three other scalars, means marginalising over U does not move beta: no U-beta confounding.
- **H-omega**: the induced order agreeing with Stage 6C2's, where omega was fixed, means freeing omega does not move the structure: no H-omega confounding.
- **rho-U**: rho agreeing with Stage 6C2's means the extra three free scalars do not reach rho, which is expected because rho enters only p(U | rho) and the order posterior is unchanged.

### 8.3 §16 — recovery

| quantity | value |
|---|---:|
| posterior probability of the generating order | 1.00000 |
| MAP order is the generating one | yes |
| closure precision / recall / F1 | 1.000 / 1.000 / 1.000 |
| closure structural Hamming | 0 |
| reduction F1 / Hamming | 1.000 / 0 |
| distinct orders visited | 1 |

| scalar | posterior mean ± sd | 95% interval | truth | inside | error in sd |
|---|---|---|---:|---|---:|
| `beta` | 1.48337 ± 0.03926 | [1.40688, 1.56108] | 1.50000 | PASS | -0.424 |
| `omega` | 1.88510 ± 0.12924 | [1.63904, 2.14420] | 1.73460 | PASS | +1.164 |
| `lambda_rep` | 0.80747 ± 0.02845 | [0.75201, 0.86242] | 0.80000 | PASS | +0.262 |
| `lambda_back` | 0.21906 ± 0.02398 | [0.17257, 0.26623] | 0.25000 | PASS | -1.290 |

Every generating value is inside its 95% interval. The largest standardised error is `lambda_back` at -1.290 posterior sd, which is a property of this corpus rather than of the sampler: Stage 6B3 obtained the same offset with `U` pinned at the truth, and §8.2 shows the two posteriors agree to 0.0105 parent sd. A finite-data posterior is not obliged to centre on the generating value; it is obliged to contain it, and to match the posterior an independent route obtains from the same likelihood.

**`rho` recovery is NOT APPLICABLE, permanently.** U_TRUE is hand specified in recurrent_synthetic.py rather than drawn from p(U | rho); no RHO_TRUE exists and none may be manufactured by regenerating a more favourable dataset. The posterior is 0.3186 ± 0.2177; that is a statement about the prior cell mass of one order, not a recovery.

**Entrywise `U` recovery is not claimed and cannot be.** The likelihood is piecewise constant in `U` — it speaks only at order boundaries — and the target is invariant under permuting the `d` columns and under any strictly increasing reparameterisation within a column. Structure is the recoverable object; the matrix is not.

### 8.4 §13 — the column-permutation audit

`h(U)` is the intersection of the `d` column orderings and `Sigma_rho` is exchangeable in the columns, so the target is column-exchangeable. Raw entrywise `U` traces may therefore swap labels between chains with no convergence failure at all, which is why they are not a convergence criterion here.

| chain | signed column contrast | absolute (invariant) contrast |
|---|---:|---:|
| 0 | +0.02789 | 1.00471 |
| 1 | +0.01051 | 0.98288 |
| 2 | +0.02141 | 0.99478 |
| 3 | -0.00580 | 0.96431 |

Chains sit in opposite labellings: yes. Signed-contrast R-hat 1.00228. a large signed-contrast R-hat with a small invariant-summary R-hat is label switching, not non-convergence; H and relation probabilities are the reported quantities and are invariant

### 8.5 §17 — held-out prediction

200 held-out blocks, 4,000 steps, 400 posterior draws. **Reported, not gated**: following the Stage 6B convention, held-out numbers never drive a decision and were not used to choose a scale, a prior or a threshold.

| quantity | log score per step |
|---|---:|
| posterior predictive | -1.243315 |
| at the generating truth | -1.243161 |
| at the posterior-mean scalars, `U` from a modal-order draw | -1.243312 |
| at the prior mean, `U` at truth (a floor, not a competitor) | -1.325226 |

**A negative control worth stating.** Plugging in the *entrywise* posterior mean of `U` scores -2.028830 per step — worse than the prior. averaging U entrywise across the posterior collapses the incomparabilities: every draw realises the same partial order, but their mean lands in a region where all coordinates are strictly ordered, so h(mean U) is a denser order than h(U) for any draw. The likelihood reads U only through h(U), so this plug-in scores a different model. It is reported as a negative control — a concrete demonstration of why entrywise U recovery is not claimed — and not as a competitor. Concretely, every retained draw induces an order with 6 relations, while the entrywise mean induces one with 10. This is the clearest single demonstration of why Stage 6D reports structure rather than the matrix.

### 8.6 §20 — the registered gates

Every threshold below was written to `results/mcmc_original/stage6d2_gate_registration.json` **before the formal chains started**. None was moved after a value was seen.

| gate | value | threshold | verdict |
|---|---:|---:|---|
| rho_rhat | 1.00091 | <= 1.0100 | PASS |
| beta_rhat | 1.00069 | <= 1.0100 | PASS |
| omega_rhat | 1.00024 | <= 1.0100 | PASS |
| lambda_rep_rhat | 1.00090 | <= 1.0100 | PASS |
| lambda_back_rhat | 1.00023 | <= 1.0100 | PASS |
| log_posterior_rhat | 1.00290 | <= 1.0100 | PASS |
| relation_count_rhat | n/a | <= 1.0100 | PASS |
| uncertain_relation_rhat | n/a | <= 1.0100 | PASS |
| min_bulk_ess | 4873.97669 | >= 1,000 | PASS |
| max_mcse_over_posterior_sd | 0.01432 | <= 0.0500 | PASS |
| post_burn_in_acceptance_band | U 0.318, rho 0.328, beta 0.444, omega 0.386, lambda_rep 0.389, lambda_back 0.519 | inside [0.2, 0.65] | PASS |
| beta_matches_stage6b3 | 0.06127 | <= 0.2500 | PASS |
| omega_matches_stage6b3 | 0.01706 | <= 0.2500 | PASS |
| lambda_rep_matches_stage6b3 | 0.05617 | <= 0.2500 | PASS |
| lambda_back_matches_stage6b3 | 0.01045 | <= 0.2500 | PASS |
| rho_matches_stage6c2 | 0.01999 | <= 0.2500 | PASS |
| structure_matches_stage6c2 | 1.00000 | >= 0.9900 | PASS |
| structural_recovery_closure_f1 | 1.00000 | == 1.0000 | PASS |
| structural_recovery_hamming | 0 | == 0 | PASS |
| probability_of_the_true_order | 1.00000 | >= 0.9900 | PASS |
| beta_truth_in_95_interval | yes | == yes | PASS |
| omega_truth_in_95_interval | yes | == yes | PASS |
| lambda_rep_truth_in_95_interval | yes | == yes | PASS |
| lambda_back_truth_in_95_interval | yes | == yes | PASS |

**24 gates, all pass.**

## 9. Verdicts, kept apart on purpose

```
    Stage 6D0 smoke and kernel parity          PASS
    Stage 6D1 sampler correctness              PASS
    Stage 6D2 convergence                      PASS
    Stage 6D2 structural (U) recovery          PASS
    Stage 6D2 scalar recovery                  PASS
    Stage 6D rho recovery                      NOT APPLICABLE — no generating value exists
    Stage 6D entrywise U recovery              NOT CLAIMED — the target is invariant
```

## 10. §19 — the result directories

| directory | present | size |
|---|---|---:|
| `results/mcmc_original/stage6d0_joint_smoke` | PASS | 0.0 MB |
| `results/mcmc_original/stage6d1_joint_reference` | PASS | 4.1 MB |
| `results/mcmc_original/stage6d1_joint_mcmc` | PASS | 2.8 MB |
| `results/mcmc_original/stage6d2_oracle_joint_full_seed0` | PASS | 1.3 MB |
| `results/mcmc_original/stage6d_complete` | PASS | 2.6 MB |

Also preserved, unmodified, and never relabelled as pilots or as passes:

- `results/mcmc_original/stage6d1_joint_mcmc_FAILED_attempt0_50k` — attempt 1 — original scales, 50,000 sweeps
- `results/mcmc_original/stage6d1_joint_mcmc_FAILED_attempt1` — attempt 1 — original scales, 100,000 ceiling
- `results/mcmc_original/stage6d1_joint_mcmc_FAILED_attempt2_omega_retuned_REEXECUTED` — attempt 2 — omega x32, 50,000 sweeps
- `stage6d1_omega_pilot`, `stage6d1_scalar_pilot`, `stage6d2_pilot` — the efficiency-only pilots, with their registrations
- `stage6d2_pipeline_smoke_DISCARDED` — **disclosed and discarded.** 600 sweeps, 100 burn-in, thin 5, four chains, at the REGISTERED (un-piloted) Stage 6B/6C scales. Run only to exercise the §15/§16/§13/§17 analysis code end to end so that the formal run's analysis would not fail on a code path bug after the chains had been paid for. It is kept rather than deleted because deleting it would hide that Stage 6D2 numbers were computed, on registered scales, before the pilot finished. It set no threshold, no scale and no start: the gate registration predates it, and the pilot was already running under a selection rule executed in code, which was neither consulted nor adjusted in response to it.

## 11. Known shortfalls, recorded rather than papered over

- **No independent reference exists for the Stage 6D2 corpus, and none can be built by the Stage 6D1 route.** Prior importance sampling degrades as the likelihood sharpens; at 30 blocks of `T = 8` the relative ESS was already 0.005 with one point holding 10% of the weight. Stage 6D2's correctness claim is therefore *inherited* from Stage 6D1 and supported by parent consistency, not established afresh.
- **`rho` remains weakly identified**, exactly as Stage 6C found. The order posterior is a point mass and `rho` never enters the likelihood, so `p(rho | Y)` is driven entirely by how one order's prior cell mass varies with `rho`, on five rows of a two-dimensional Gaussian.
- **The Stage 6D1 pilot's `replay_per_sweep_ok` flag was off by one** — it compared against `sweeps x (m + 1)` and omitted the single replay that builds the initial state, so it recorded `false` on every row. The flag is reported, never read by the selection rule, so no scale was affected. The Stage 6D2 pilot compares against `1 + sweeps x (m + 1)` and records both the observed and the expected count.
- **Unknown boundaries, skill-label inference and semi-Markov FFBS are not started.** Stage 6D is the last stage that runs on oracle segmentation and oracle labels.

