# Stage 6E2 — unknown-boundary joint inference on a trace corpus

## The corpus (§11)

The Stage 6D corpus is 500 **independent blocks** with `K = 1`, no trace
structure and no `pi`/`P`. Concatenating them would manufacture a skill-transition
structure the generator never sampled, so §11's fallback applies and a
trace-level corpus was generated and frozen.

| | |
|---|---|
| training traces / blocks | 100 / 510 |
| held-out traces / blocks | 45 / 223 |
| trace length `J` | mean 32.0, range 16–52 |
| block width | mean 6.27, range 3–12 |
| traces reusing a skill type | 100 of 100 |
| corpus hash | `02be246edf9bd4f4148efa3a3e269afa…` |
| generation seed searched? | **no** |

Block widths are drawn from the **registered boundary prior** — `p(w) ∝ (1-delta_B)^(w-1)` truncated to `[3, 12]` — rather than from a convenient
uniform, so the generated truth follows the law the target assumes.

### Exposure audit

Reported, never used to select the corpus. A dataset with no upstream repeats
cannot inform `lambda_back`, and saying so is more useful than a seed search that
hides it.

| event | training count |
|---|---:|
| total steps | 3,199 |
| valid repeat | 1,617 |
| leaf repeat | 931 |
| upstream repeat | 686 |
| recomputation | 243 |

### Leakage audit (§11, §30)

- model traces equal the observed sequences: **True**
- model holds no true segmentation: **True**
- verdict: **PASS**

The hidden boundaries and labels live only in the frozen manifest and are
read by the recovery evaluation and the oracle-boundary control — never by
the unknown-boundary sampler.

## The pilot (§13), and AMENDMENT 1

**Amendment 1**, registered before any Stage 6E2 formal draw existed: the candidate multiplier grid only, from [0.25, 0.5, 1.0, 2.0, 4.0, 8.0] to [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0].

*Evidence.* all four scalar coordinates selected the grid's upper boundary x8, and lambda_rep and lambda_back were still inadmissible there (expected acceptance 0.653 and 0.661 against a 0.60 ceiling). Four coordinates pinned to the same edge is evidence that the search RANGE was truncated, not that the rule failed.

*Unchanged:* admissible acceptance band; selection statistic (largest median expected ESJD); tie-break; ESJD coordinates; permitted and forbidden information; proposal-count study and its selection; U and rho policy and their grid.

*Existing rows:* preserved verbatim — pass 1 reruns the originally registered grid with its original seed, and the new candidates are measured in a second pass with their own registered seed, so adding candidates cannot move a number that was already recorded.

| coordinate | selected | scale | expected acceptance | median ESJD (own coordinate) |
|---|---:|---:|---:|---:|
| `beta` | x8 | 0.40872 | 0.5972 | 1.9669e-02 |
| `omega` | x8 | 1.45704 | 0.5618 | 1.7551e-01 |
| `lambda_rep` | x16 | 0.66480 | 0.3442 | 5.8270e-04 |
| `lambda_back` | x16 | 1.51552 | 0.3011 | 1.8887e-02 |
| `rho` | frozen Stage 6D | 0.50000 | 0.8643 | — |
| `U` | frozen Stage 6D | 0.50000 | 0.3371 | — |

Segmentation proposals per trace per sweep: **32**, chosen on boundary-Hamming movement per *second* — movement and computational efficiency only, as §13 requires.

| proposals/trace | ms/sweep | Hamming/sweep | Hamming/second | distinct segmentations per trace |
|---:|---:|---:|---:|---:|
| 16 | 301.3 | 174.4 | 578.9 | 43.3 |
| 32 | 469.6 | 280.6 | 597.5 | 49.9 |
| 64 | 803.8 | 358.0 | 445.4 | 51.1 |
| 128 | 1461.5 | 446.3 | 305.4 | 51.6 |

All pilot draws were discarded. The pilot saw only acceptance, ESJD, invalid-proposal rates, finite-target checks, replay checks, cache consistency, movement and wall time.

### Discarded joint confirmation

| check | verdict |
|---|---|
| all targets finite | PASS |
| every segmentation legal | PASS |
| block table matches registered scorer | PASS |
| grouped and per block evaluators agree | PASS |
| boundaries moved | PASS |
| every trace visited more than one segmentation | PASS |
| every move type proposed | PASS |
| every move type accepted | PASS |

## Convergence (§15)

| gate | value | threshold | verdict |
|---|---:|---:|---|
| `P[0,1]_rhat` | 1.93655 | 1.01 | **FAIL** |
| `P[0,2]_rhat` | 1.93658 | 1.01 | **FAIL** |
| `P[1,0]_rhat` | 2.18379 | 1.01 | **FAIL** |
| `P[1,2]_rhat` | 2.18373 | 1.01 | **FAIL** |
| `P[2,0]_rhat` | 1.79391 | 1.01 | **FAIL** |
| `P[2,1]_rhat` | 1.79392 | 1.01 | **FAIL** |
| `acceptance_band` | {'U': 0.33962733333333334, 'rho': 0.84934, 'beta': 0.10593000000000001, 'omega': 0.5212100000000001, 'lambda_rep': 0.36218, 'lambda_back': 0.19561999999999996} | [0.1, 0.7] | **FAIL** |
| `beta_rhat` | 1.55889 | 1.01 | **FAIL** |
| `boundary_indicator_rhat` | 1.55102 | 1.01 | **FAIL** |
| `co_clustering_rhat` | 2.24744 | 1.01 | **FAIL** |
| `lambda_back_rhat` | 1.14061 | 1.01 | **FAIL** |
| `lambda_rep_rhat` | 1.72069 | 1.01 | **FAIL** |
| `log_target_rhat` | 2.48425 | 1.01 | **FAIL** |
| `max_mcse_over_sd` | 0.40555 | 0.05 | **FAIL** |
| `min_scalar_bulk_ess` | 6.08020 | 400 | **FAIL** |
| `omega_rhat` | 1.03648 | 1.01 | **FAIL** |
| `pi[0]_rhat` | 1.93360 | 1.01 | **FAIL** |
| `pi[1]_rhat` | 2.16742 | 1.01 | **FAIL** |
| `pi[2]_rhat` | 1.88660 | 1.01 | **FAIL** |
| `relation_count_rhat` | 4.28943 | 1.01 | **FAIL** |
| `relation_indicator_rhat` | 6124797779770170.00000 | 1.01 | **FAIL** |
| `rho_rhat` | 1.01015 | 1.01 | **FAIL** |
| `segment_count_rhat` | 1.07919 | 1.01 | **FAIL** |
| `transition_count_spectrum_rhat` | 1.58561 | 1.01 | **FAIL** |

**Convergence: FAIL**

### Scalars

| scalar | mean | SD | 95% CI | truth | in CI | R-hat | bulk ESS | tail ESS | MCSE |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|
| `beta` | 1.4923 | 0.1360 | [1.2621, 1.7803] | 1.5000 | yes | 1.55889 | 7 | 25 | 0.05191 |
| `omega` | 2.1268 | 1.2485 | [-0.1179, 4.8562] | 1.7346 | yes | 1.03648 | 65 | 84 | 0.15428 |
| `lambda_rep` | 0.2296 | 0.1252 | [0.0185, 0.4427] | 0.8000 | **no** | 1.72069 | 6 | 25 | 0.05079 |
| `lambda_back` | 0.4211 | 0.1678 | [0.1080, 0.7708] | 0.2500 | yes | 1.14061 | 18 | 45 | 0.03930 |
| `rho` | 0.1472 | 0.1255 | [0.0040, 0.4640] | — | — | 1.01015 | 3139 | 10991 | 0.00224 |

`rho` has **NOT APPLICABLE** status for recovery: `U_TRUE_BY_SKILL` is hand-specified, not drawn from `p(U | rho)`, so no `rho_true` exists. Inherited from the Stage 6C freeze unchanged.

## Recovery (§16)

Correctness, convergence and recovery are separate verdicts. A recovery failure is not evidence that the sampler is wrong.

### Boundaries

| statistic | value |
|---|---:|
| Boundary F1 | 0.2896 |
| precision | 0.7327 |
| recall | 0.1805 |
| Brier score | 0.0844 |
| expected calibration error | 0.0298 |
| mean posterior probability at true cuts | 0.3168 |
| mean posterior probability elsewhere | 0.0996 |
| segment-count MAE per trace | 0.5493 |
| segment-length TV | 0.0330 |

### Skill labels

| statistic | value |
|---|---:|
| occurrence-level aligned accuracy | 0.6687 |
| adjusted Rand index | 0.2798 |
| NMI | 0.2436 |
| segment-level aligned accuracy | 0.7411 |
| repeated-invocation aligned accuracy | 0.7190 |
| distinct alignment permutations | 3 |
| label-permutation mode switches | 3 (0.0000 per draw) |
| worst-confused pair | inferred 2 vs true 1, 0.2214 |

deterministic Hungarian assignment per retained draw on the frozen cost -confusion[inferred, true]; used for RECOVERY REPORTING ONLY and never by the target, the proposals or any convergence statistic

### Partial orders, per aligned skill

| skill | P(true H) | MAP = truth | closure F1 | reduction F1 | Hamming | min P(true relation) | max P(false relation) |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0.2135 | **no** | 0.5000 | 0.4000 | 4 | 0.2135 | 0.2500 |
| 1 | 0.0000 | **no** | 0.2500 | 0.0000 | 6 | 0.0000 | 0.2500 |
| 2 | 0.2500 | **no** | 0.8889 | 0.7500 | 1 | 0.2500 | 0.0000 |

### Transitions

- `pi` max absolute error (aligned): 0.0848
- `P` max absolute error (aligned): 0.0445

## Held-out prediction (§17)

Posterior-predictive NLL, integrating **analytically** over every legal
`(S, z)` on each held-out trace by the forward recursion, then averaging
over posterior draws. Segmentation, labels, `U`, the four scalars and
`(pi, P)` are all integrated over.

| representation | NLL per occurrence | NLL per trace |
|---|---:|---:|
| unknown-boundary posterior predictive | 1.55842 | 48.2071 |
| like-for-like oracle-boundary control | 1.49263 | 46.1721 |
| true-parameter oracle | 1.49059 | 46.1089 |
| `h(E[U])` — **LABELLED NEGATIVE CONTROL** | 1.97864 | 61.2061 |

Gap from the oracle-boundary control: +0.06579 per occurrence. Fraction of held-out traces favouring the unknown-boundary posterior: 0.133.

both marginalise (S, z) on the held-out traces, because held-out boundaries are unknown to both. The control's advantage is entirely in the TRAINING posterior it carries.


LABELLED NEGATIVE CONTROL. h(E[U]) is not a valid plug-in: averaging U inside an order cell can collapse incomparabilities, so the induced order of the mean is not the mean of the induced orders. Reported to size that failure, never as a result.

## Verdicts

```
stage_6e_sampler_correctness           see stage6e0/6e1a/6e1b — not restated here
stage_6e2_convergence                  FAIL
stage_6e2_boundary_recovery            FAIL
stage_6e2_skill_label_recovery         FAIL
stage_6e2_structural_recovery          FAIL
stage_6e2_scalar_recovery              PARTIAL
stage_6e2_identifiability              PARTIALLY IDENTIFIED
```

## Continuation history (§14)

```
{
  "unknown": [
    {
      "block": 1,
      "sweeps_to": 50000,
      "burn_in": 15000,
      "thin": 5,
      "resumed": true,
      "kind": "crash recovery (same registered sweep count)",
      "wall_seconds": 34566.951150000095,
      "retained_pooled": 28000,
      "scales": {
        "U": 0.5,
        "rho": 0.5,
        "beta": 0.40872,
        "omega": 1.45704,
        "lambda_rep": 0.6648,
        "lambda_back": 1.51552
      },
      "n_proposals_per_trace": 32,
      "why": "crash recovery: the chains were interrupted and resumed from their last checkpoint to the SAME registered sweep count, with each chain's own RNG state restored. Not a continuation and not an extension."
    },
    {
      "block": 2,
      "sweeps_to": 75000,
      "burn_in": 15000,
      "thin": 5,
      "resumed": true,
      "kind": "section 14 continuation",
      "wall_seconds": 26416.102708791965,
      "retained_pooled": 20000,
      "scales": {
        "U": 0.5,
        "rho": 0.5,
        "beta": 0.40872,
        "omega": 1.45704,
        "lambda_rep": 0.6648,
        "lambda_back": 1.51552
      },
      "n_proposals_per_trace": 32,
      "why": "registered 25,000-sweep continuation after a convergence gate failed"
    },
    {
      "block": 3,
      "sweeps_to": 100000,
      "burn_in": 15000,
      "thin": 5,
      "resumed": true,
      "kind": "section 14 continuation",
      "wall_seconds": 22541.17177870893,
      "retained_pooled": 20000,
      "scales": {
        "U": 0.5,
        "rho": 0.5,
        "beta": 0.40872,
        "omega": 1.45704,
        "lambda_rep": 0.6648,
        "lambda_back": 1.51552
      },
      "n_proposals_per_trace": 32,
      "why": "registered 25,000-sweep continuation after a convergence gate failed"
    },
    {
      "block": 4,
      "sweeps_to": 125000,
      "burn_in": 15000,
      "thin": 5,
      "resumed": true,
      "kind": "section 14 continuation",
      "wall_seconds": 26125.417116959,
      "retained_pooled": 20000,
      
```

