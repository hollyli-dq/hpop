# Stage 5 — static synthetic recovery (full, generator seed 0)

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
| Stage-5 generator | **PASS** | 40 train / 20 test, 70% boundary-ambiguous |
| Stage 5A oracle-S recovery | **PASS** | macro ordered F1 1.0000, transition MAE 0.0654 |
| Stage 5B oracle-U/P segmentation | **FAIL** | expected >= 0.85, observed 0.6670 |

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

- `n_train`: 40
- `n_train_extra_for_coverage`: 0
- `n_test`: 20
- `n_test_natural`: 10
- `n_test_ambiguous`: 10
- `generation_attempts`: 60
- `rejected_traces`: 0
- `instance_counts`: {'A': 43, 'D': 37, 'F': 60, 'E': 41}
- `outgoing_per_row`: {'A': 31, 'D': 31, 'F': 42, 'E': 37}
- `tilings_min`: 1
- `tilings_max`: 88
- `tilings_mean`: 17.15
- `pct_ambiguous_overall`: 96.66666666666667
- `cut_patterns_min`: 1
- `cut_patterns_max`: 15
- `cut_patterns_mean`: 4.116666666666666
- `pct_boundary_ambiguous_overall`: 70.0
- `trace_lengths`: {'min': 6, 'max': 16}

Training transition counts (rows A, D, F, E):

```
  A     0     4    22     5
  D     1     0    21     9
  F    17    12     0    13
  E    16    13     8     0
```

## 3. Stage 5A — oracle S, infer U and P

4 chains, 15,000 iterations, 3,000 burn-in, thin 3, `sigma_U = 0.8` (pilot adjustment applied: False).

### 3.1 Latent orders

| skill | executions | acceptance by chain | R-hat(log post) | true ordered pairs | inferred | ordered F1 | incomparable F1 |
|---|---|---|---|---|---|---|---|
| A | 43 | [0.456, 0.46, 0.467, 0.458] | 1.0009 | [(0, 1)] | [(0, 1)] | 1.0000 | 1.0000* |
| D | 37 | [0.426, 0.425, 0.428, 0.43] | 1.0002 | (none) | (none) | 1.0000* | 1.0000 |
| F | 60 | [0.464, 0.458, 0.461, 0.46] | 1.0015 | [(0, 1)] | [(0, 1)] | 1.0000 | 1.0000* |
| E | 41 | [0.249, 0.252, 0.252, 0.246] | 1.0007 | [(0, 2), (0, 3)] | [(0, 2), (0, 3)] | 1.0000 | 1.0000 |

`*` marks a **vacuous** F1: both the true and the predicted set were empty, which is correct recovery but undefined for F1 and scored 1.0 by convention. 3 of 4 skills have at least one. The micro (pooled) scores below are not vacuous and are the ones to trust.

- **macro ordered F1 = 1.0000**, macro incomparable F1 = 1.0000
- micro ordered F1 = 1.0000 (4/4 true pairs), micro incomparable F1 = 1.0000

Posterior probability of each true ordered pair, and of its reverse:

| skill | pair | P(true direction) | P(reverse) |
|---|---|---|---|
| A | 0>1 | 1.0000 | 0.0000 |
| F | 0>1 | 1.0000 | 0.0000 |
| E | 0>2 | 1.0000 | 0.0000 |
| E | 0>3 | 0.9656 | 0.0000 |

### 3.2 Transition matrix

| row | next | count | alpha | posterior mean | 95% CI | true |
|---|---|---|---|---|---|---|
| A | D | 4 | 5 | 0.1471 | [0.051, 0.284] | 0.10 |
| A | F | 22 | 23 | 0.6765 | [0.514, 0.821] | 0.75 |
| A | E | 5 | 6 | 0.1765 | [0.069, 0.318] | 0.15 |
| D | A | 1 | 2 | 0.0588 | [0.007, 0.159] | 0.10 |
| D | F | 21 | 22 | 0.6471 | [0.483, 0.796] | 0.70 |
| D | E | 9 | 10 | 0.2941 | [0.157, 0.454] | 0.20 |
| F | A | 17 | 18 | 0.4000 | [0.265, 0.547] | 0.55 |
| F | D | 12 | 13 | 0.2889 | [0.167, 0.428] | 0.15 |
| F | E | 13 | 14 | 0.3111 | [0.187, 0.452] | 0.30 |
| E | A | 16 | 17 | 0.4250 | [0.277, 0.579] | 0.35 |
| E | D | 13 | 14 | 0.3500 | [0.211, 0.502] | 0.35 |
| E | F | 8 | 9 | 0.2250 | [0.111, 0.367] | 0.30 |

- **MAE over allowed entries = 0.0654** (max 0.1500)
- row-wise KL(P_TRUE || posterior mean): [0.0144, 0.031, 0.0659, 0.0184], mean 0.0324

## 4. Stage 5B — oracle U and P, infer S

4 chains, 20,000 sweeps, 5,000 burn-in, thin 5. Chains start from `sample_random_legal_segmentation`, which reads only the CPA sequence and the skill supports — never the truth.

- **Boundary F1 = 0.9655** (P 0.9459, R 0.9859, 70/71 true cuts)
- **Skill ARI = 0.6670**, occurrence accuracy 0.8333, exact path accuracy 0.5000, best-retained path accuracy 0.5000

| subset | traces | boundary P | R | F1 |
|---|---|---|---|---|
| natural | 10 | 0.9730 | 0.9730 | 0.9730 |
| ambiguous | 10 | 0.9189 | 1.0000 | 0.9577 |

- mean posterior on **true** cuts: 0.9707
- mean posterior on **false** candidate cuts: 0.0248
- mean posterior mass on the true occurrence skill: 0.7981

### 4.1 Moves

| move | proposed | accepted | acceptance |
|---|---|---|---|
| relabel | 377,792 | 154,963 | 0.4102 |
| split | 169,109 | 30,674 | 0.1814 |
| merge | 326,194 | 30,679 | 0.0941 |
| shift | 155,193 | 85 | 0.0005 |

Every available move's exact MH log-acceptance ratio, enumerated from states the chains actually visited. This is what distinguishes a move the posterior refuses from a move that is silently broken:

| move | moves examined | max log alpha | with log alpha >= 0 | median | target impossible |
|---|---|---|---|---|---|
| relabel | 812 | 3.2834 | 278 | -0.3801 | 26.8% |
| split | 514 | 9.7738 | 172 | -0.2291 | 28.2% |
| merge | 803 | 7.0875 | 169 | -2.4048 | 25.0% |
| shift | 446 | 9.5974 | 31 | -8.087 | 34.1% |

Log-target ESS by chain: [637, 921, 769, 894].

### 4.2 Is the target even pointing at the truth?

On **10 of 20** test traces the posterior *strictly prefers* a segmentation other than the true one, by up to 3.008 in log target (mean 1.411). On those traces no sampler can recover the truth: the target itself points elsewhere. Any shortfall in skill metrics below is therefore a property of the model, not of the algorithm.

### 4.3 Per trace

| # | group | T | tilings | cut patterns | true cuts | predicted cuts | P(true cut) | P(false cut) | segment-count ESS |
|---|---|---|---|---|---|---|---|---|---|
| 0 | natural | 14 | 88 | 13 | [4, 6, 8, 10, 12] | [4, 6, 8, 10, 12] | 0.983 | 0.016 | 2047 |
| 1 | natural | 12 | 34 | 9 | [2, 4, 6, 8] | [2, 4, 6, 8, 10] | 0.963 | 0.132 | 6559 |
| 2 | natural | 10 | 12 | 4 | [2, 4, 6] | [2, 4, 6] | 0.971 | 0.048 | 3887 |
| 3 | natural | 12 | 4 | 1 | [4, 6, 10] | [4, 6, 10] | 1.000 | 0.000 | 12000 |
| 4 | natural | 12 | 1 | 1 | [2, 6, 8] | [2, 6, 8] | 1.000 | 0.000 | 12000 |
| 5 | natural | 16 | 84 | 13 | [2, 4, 6, 10, 12] | [2, 4, 6, 10, 12] | 0.958 | 0.011 | 3082 |
| 6 | natural | 14 | 8 | 3 | [2, 4, 6, 8, 12] | [4, 6, 8, 12] | 0.838 | 0.000 | 3220 |
| 7 | natural | 12 | 34 | 9 | [2, 4, 6, 8, 10] | [2, 4, 6, 8, 10] | 0.958 | 0.000 | 5895 |
| 8 | natural | 8 | 2 | 1 | [2, 6] | [2, 6] | 1.000 | 0.000 | 12000 |
| 9 | natural | 8 | 2 | 1 | [4, 6] | [4, 6] | 1.000 | 0.000 | 12000 |
| 10 | ambiguous | 16 | 64 | 11 | [2, 6, 8, 12, 14] | [2, 4, 6, 8, 12, 14] | 0.995 | 0.084 | 707 |
| 11 | ambiguous | 8 | 10 | 4 | [2, 4, 6] | [2, 4, 6] | 0.945 | 0.000 | 7392 |
| 12 | ambiguous | 14 | 52 | 11 | [2, 6, 8, 12] | [2, 4, 6, 8, 12] | 1.000 | 0.112 | 2174 |
| 13 | ambiguous | 8 | 2 | 1 | [4, 6] | [4, 6] | 1.000 | 0.000 | 12000 |
| 14 | ambiguous | 16 | 68 | 15 | [2, 6, 8, 10, 12] | [2, 4, 6, 8, 10, 12] | 0.956 | 0.093 | 3015 |
| 15 | ambiguous | 12 | 8 | 3 | [4, 6, 8, 10] | [4, 6, 8, 10] | 0.939 | 0.000 | 5534 |
| 16 | ambiguous | 6 | 8 | 3 | [2, 4] | [2, 4] | 0.951 | 0.000 | 6864 |
| 17 | ambiguous | 6 | 4 | 3 | [2, 4] | [2, 4] | 0.956 | 0.000 | 7016 |
| 18 | ambiguous | 12 | 2 | 1 | [4, 6, 8] | [4, 6, 8] | 1.000 | 0.000 | 12000 |
| 19 | ambiguous | 12 | 12 | 2 | [2, 6, 8, 10] | [2, 6, 8, 10] | 1.000 | 0.000 | 4802 |

## 5. Deviations, warnings and notes

- The '>= 2 support-compatible tilings' criterion for ambiguity enrichment is NON-SELECTIVE for this library: skills A and D share the CPA support {0,1}, so any trace containing a 2-block admits an A<->D relabel and trivially has >= 2 tilings. 100% of traces qualify and 0 were rejected. `num_distinct_cut_patterns` is reported alongside as the boundary-ambiguity measure that actually stresses Stage 5B.
- sigma_U = 0.8 is Stage-2A's validated scale (toy_stage0_to_stage3_report.md), used as the initial value per spec. The spec's fallback of 0.25 was measured at acceptance 0.65-0.81, outside the [0.10, 0.60] band, so the fallback was not used and no pilot adjustment was needed.
- F1 for a skill whose true relation set is empty (skill D's ordered pairs, skill A/F's incomparable pairs) is undefined; correct recovery predicts the empty set too. Those are scored 1.0 and flagged 'vacuous'. Micro (pooled) scores are reported alongside and are not vacuous.
- IDENTIFIABILITY LIMIT (the cause of any Stage-5B skill-metric shortfall): skills A and D share the CPA support {0,1}. D is an antichain, so it emits the ordered permutation (0,1) about half the time; for such a block the likelihood ratio is p_A/p_D = 0.975/0.500 = 1.95, i.e. the CORRECT posterior favours A over the true D by ~2:1. Roughly half of all D instances are therefore intrinsically mislabelled by any method that reports the true posterior. This caps achievable skill ARI and is a property of the fixed library, not a sampler defect. Boundary recovery is unaffected.
- Stage 5B uses CachedLocalMoveKernel, a subclass that memoises proposal_distribution. LocalMoveKernel's mathematics is untouched; the cache is verified to return identical proposal probabilities in tests/mcmc_original/test_stage5_oracle_up.py.

