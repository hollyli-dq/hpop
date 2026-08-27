# Stage 5 — static synthetic recovery (smoke, generator seed 0)

Date: 2026-08-10
Branch: `mcmc-original-latent-poset`  Commit: `08e2cb6deeec1634d6c91667368b736a5427f2e0`  (dirty)
Python 3.13.2, NumPy 2.4.6

Stage 5 is deliberately split. **5A** hides `U` and `P` but keeps the true
segmentation; **5B** hides the segmentation but keeps the true `U` and `P`. The joint
S+U+P sampler is *not* run here — the point is that a failure can be attributed.

## PASS / FAIL

| stage | result | headline |
|---|---|---|
| prerequisites | **PASS** | library + Stage-4 kernel available |
| Stage-5 generator | **PASS** | 8 train / 4 test, 67% boundary-ambiguous |
| Stage 5A oracle-S recovery | **PASS** | macro ordered F1 1.0000, transition MAE 0.1273 |
| Stage 5B oracle-U/P segmentation | **PASS** | boundary F1 1.0000, ARI 0.9082 |

## 1. Fixed model constants and library

`beta = 1.5`, `epsilon = 0.05`, `delta_B = 0.5`, `rho_U = 0.25`, `d = 2`, `sigma_U = 0.8`, uniform `pi_k = 1/4`.

| skill | CPA support | true induced order (transitive closure) |
|---|---|---|
| A | (0, 1) | 0>1 |
| D | (0, 1) | antichain |
| F | (2, 3) | 0>1 |
| E | (0, 1, 2, 3) | 0>2, 0>3 |

`U_A` = `[[1.0, 1.0], [0.0, 0.0]]`

`U_D` = `[[1.0, 0.0], [0.0, 1.0]]`

`U_F` = `[[1.0, 1.0], [0.0, 0.0]]`

`U_E` = `[[2.0, 2.0], [3.0, -1.0], [1.0, 0.0], [0.0, 1.0]]`

`P_TRUE` (rows/cols A, D, F, E):

```
           A       D       F       E
  A    0.00    0.10    0.75    0.15
  D    0.10    0.00    0.70    0.20
  F    0.55    0.15    0.00    0.30
  E    0.35    0.35    0.30    0.00
```

## 2. Dataset

- `n_train`: 8
- `n_train_extra_for_coverage`: 0
- `n_test`: 4
- `n_test_natural`: 2
- `n_test_ambiguous`: 2
- `generation_attempts`: 12
- `rejected_traces`: 0
- `instance_counts`: {'A': 10, 'D': 9, 'F': 12, 'E': 8}
- `outgoing_per_row`: {'A': 8, 'D': 8, 'F': 8, 'E': 7}
- `tilings_min`: 3
- `tilings_max`: 26
- `tilings_mean`: 9.583333333333334
- `pct_ambiguous_overall`: 100.0
- `cut_patterns_min`: 1
- `cut_patterns_max`: 6
- `cut_patterns_mean`: 2.8333333333333335
- `pct_boundary_ambiguous_overall`: 66.66666666666666
- `trace_lengths`: {'min': 10, 'max': 16}

Training transition counts (rows A, D, F, E):

```
  A     0     1     5     2
  D     1     0     4     3
  F     3     3     0     2
  E     2     2     3     0
```

## 3. Stage 5A — oracle S, infer U and P

2 chains, 5,000 iterations, 1,000 burn-in, thin 2, `sigma_U = 0.8` (pilot adjustment applied: False).

### 3.1 Latent orders

| skill | executions | acceptance by chain | R-hat(log post) | true ordered pairs | inferred | ordered F1 | incomparable F1 |
|---|---|---|---|---|---|---|---|
| A | 10 | [0.467, 0.469] | 0.9999 | [(0, 1)] | [(0, 1)] | 1.0000 | 1.0000* |
| D | 9 | [0.421, 0.434] | 1.0011 | (none) | (none) | 1.0000* | 1.0000 |
| F | 12 | [0.476, 0.447] | 1.0116 | [(0, 1)] | [(0, 1)] | 1.0000 | 1.0000* |
| E | 8 | [0.3, 0.298] | 0.9999 | [(0, 2), (0, 3)] | [(0, 2), (0, 3)] | 1.0000 | 1.0000 |

`*` marks a **vacuous** F1: both the true and the predicted set were empty, which is correct recovery but undefined for F1 and scored 1.0 by convention. 3 of 4 skills have at least one. The micro (pooled) scores below are not vacuous and are the ones to trust.

- **macro ordered F1 = 1.0000**, macro incomparable F1 = 1.0000
- micro ordered F1 = 1.0000 (4/4 true pairs), micro incomparable F1 = 1.0000

Posterior probability of each true ordered pair, and of its reverse:

| skill | pair | P(true direction) | P(reverse) |
|---|---|---|---|
| A | 0>1 | 0.9383 | 0.0000 |
| F | 0>1 | 0.9998 | 0.0000 |
| E | 0>2 | 0.8865 | 0.0000 |
| E | 0>3 | 0.9145 | 0.0000 |

### 3.2 Transition matrix

| row | next | count | alpha | posterior mean | 95% CI | true |
|---|---|---|---|---|---|---|
| A | D | 1 | 2 | 0.1818 | [0.025, 0.451] | 0.10 |
| A | F | 5 | 6 | 0.5455 | [0.267, 0.815] | 0.75 |
| A | E | 2 | 3 | 0.2727 | [0.066, 0.553] | 0.15 |
| D | A | 1 | 2 | 0.1818 | [0.025, 0.447] | 0.10 |
| D | F | 4 | 5 | 0.4545 | [0.190, 0.736] | 0.70 |
| D | E | 3 | 4 | 0.3636 | [0.121, 0.655] | 0.20 |
| F | A | 3 | 4 | 0.3636 | [0.119, 0.651] | 0.55 |
| F | D | 3 | 4 | 0.3636 | [0.121, 0.654] | 0.15 |
| F | E | 2 | 3 | 0.2727 | [0.067, 0.556] | 0.30 |
| E | A | 2 | 3 | 0.3000 | [0.075, 0.601] | 0.35 |
| E | D | 2 | 3 | 0.3000 | [0.074, 0.603] | 0.35 |
| E | F | 3 | 4 | 0.4000 | [0.136, 0.701] | 0.30 |

- **MAE over allowed entries = 0.1273** (max 0.2455)
- row-wise KL(P_TRUE || posterior mean): [0.0894, 0.1229, 0.1233, 0.0216], mean 0.0893

## 4. Stage 5B — oracle U and P, infer S

2 chains, 5,000 sweeps, 1,000 burn-in, thin 2. Chains start from `sample_random_legal_segmentation`, which reads only the CPA sequence and the skill supports — never the truth.

- **Boundary F1 = 1.0000** (P 1.0000, R 1.0000, 18/18 true cuts)
- **Skill ARI = 0.9082**, occurrence accuracy 0.8929, exact path accuracy 0.5000, best-retained path accuracy 0.5000

| subset | traces | boundary P | R | F1 |
|---|---|---|---|---|
| natural | 2 | 1.0000 | 1.0000 | 1.0000 |
| ambiguous | 2 | 1.0000 | 1.0000 | 1.0000 |

- mean posterior on **true** cuts: 0.9772
- mean posterior on **false** candidate cuts: 0.0075
- mean posterior mass on the true occurrence skill: 0.8385

### 4.1 Moves

| move | proposed | accepted | acceptance |
|---|---|---|---|
| relabel | 9,988 | 4,912 | 0.4918 |
| split | 2,823 | 872 | 0.3089 |
| merge | 9,797 | 869 | 0.0887 |
| shift | 2,893 | 0 | 0.0000 |

Every available move's exact MH log-acceptance ratio, enumerated from states the chains actually visited. This is what distinguishes a move the posterior refuses from a move that is silently broken:

| move | moves examined | max log alpha | with log alpha >= 0 | median | target impossible |
|---|---|---|---|---|---|
| relabel | 58 | 2.0361 | 29 | 0.0 | 0.0% |
| split | 20 | 2.1172 | 9 | -0.4579 | 0.0% |
| merge | 68 | 4.9688 | 11 | -1.0991 | 58.8% |
| shift | 20 | -5.8538 | 0 | -9.3743 | 40.0% |

Log-target ESS by chain: [252, 217].

### 4.2 Is the target even pointing at the truth?

On **2 of 4** test traces the posterior *strictly prefers* a segmentation other than the true one, by up to 2.416 in log target (mean 1.577). On those traces no sampler can recover the truth: the target itself points elsewhere. Any shortfall in skill metrics below is therefore a property of the model, not of the algorithm.

### 4.3 Per trace

| # | group | T | tilings | cut patterns | true cuts | predicted cuts | P(true cut) | P(false cut) | segment-count ESS |
|---|---|---|---|---|---|---|---|---|---|
| 0 | natural | 16 | 8 | 1 | [4, 6, 8, 10, 14] | [4, 6, 8, 10, 14] | 1.000 | 0.000 | 4000 |
| 1 | natural | 12 | 6 | 2 | [2, 4, 6, 10] | [2, 4, 6, 10] | 0.975 | 0.000 | 1048 |
| 2 | ambiguous | 14 | 10 | 4 | [4, 6, 10, 12] | [4, 6, 10, 12] | 0.994 | 0.030 | 316 |
| 3 | ambiguous | 14 | 10 | 4 | [4, 6, 8, 10, 12] | [4, 6, 8, 10, 12] | 0.940 | 0.000 | 797 |

## 5. Deviations, warnings and notes

- The '>= 2 support-compatible tilings' criterion for ambiguity enrichment is NON-SELECTIVE for this library: skills A and D share the CPA support {0,1}, so any trace containing a 2-block admits an A<->D relabel and trivially has >= 2 tilings. 100% of traces qualify and 0 were rejected. `num_distinct_cut_patterns` is reported alongside as the boundary-ambiguity measure that actually stresses Stage 5B.
- sigma_U = 0.8 is Stage-2A's validated scale (toy_stage0_to_stage3_report.md), used as the initial value per spec. The spec's fallback of 0.25 was measured at acceptance 0.65-0.81, outside the [0.10, 0.60] band, so the fallback was not used and no pilot adjustment was needed.
- F1 for a skill whose true relation set is empty (skill D's ordered pairs, skill A/F's incomparable pairs) is undefined; correct recovery predicts the empty set too. Those are scored 1.0 and flagged 'vacuous'. Micro (pooled) scores are reported alongside and are not vacuous.
- IDENTIFIABILITY LIMIT (the cause of any Stage-5B skill-metric shortfall): skills A and D share the CPA support {0,1}. D is an antichain, so it emits the ordered permutation (0,1) about half the time; for such a block the likelihood ratio is p_A/p_D = 0.975/0.500 = 1.95, i.e. the CORRECT posterior favours A over the true D by ~2:1. Roughly half of all D instances are therefore intrinsically mislabelled by any method that reports the true posterior. This caps achievable skill ARI and is a property of the fixed library, not a sampler defect. Boundary recovery is unaffected.
- Stage 5B uses CachedLocalMoveKernel, a subclass that memoises proposal_distribution. LocalMoveKernel's mathematics is untouched; the cache is verified to return identical proposal probabilities in tests/mcmc_original/test_stage5_oracle_up.py.

