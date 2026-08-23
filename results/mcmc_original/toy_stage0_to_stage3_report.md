# Toy validation of the original latent partial-order model — Stages 0-3

Date: 2026-08-08
Branch: `mcmc-original-latent-poset`  Commit: `08e2cb6deeec1634d6c91667368b736a5427f2e0`  (working tree dirty)
Python 3.13.2, NumPy 2.4.6

## PASS / FAIL summary

| stage | result | headline |
|---|---|---|
| Stage 0 exact posterior | **PASS** | P(S_BA) = 0.983050847458, sum = 1.000000000000 |
| Stage 1 segmentation MCMC | **PASS** | TV = 0.00041, acceptance 0.035 |
| Stage 2A latent-U MCMC | **PASS** | all true relations recovered; min pooled P = 1.0000 |
| Stage 2B joint segmentation + U | **PASS** | boundary F1 = 1.0000, relation F1 = 1.0000 |
| Stage 3A transition Gibbs | **PASS** | worst abs error vs analytic = 0.00105 |
| Stage 3B transition ambiguity resolution | **PASS** | 0.5/0.5 -> 0.911765/0.088235 |
| Stage 3C (optional) joint S+P Gibbs | **PASS** | P(B->A) = 0.9117, P(A->C) = 0.9286 |

## 1. Model configuration

- `beta = 1.5`, `epsilon = 0.05`, `delta_B = 0.5`
- `rho_U = 0.25`, latent dimension `d = 2`
- `sigma_U = 0.8` (calibrated once; see Stage 2A)
- uniform first-skill prior `pi_k = 1/K`

### RNG seeds

| stage | seed |
|---|---|
| Stage 1 | `20260808` |
| Stage 2A | `20260808` |
| Stage 2B | `20260808` |
| Stage 3A | `20260808` |
| Stage 3C | `20260808` |

Chains derive their seeds deterministically from the stage seed, so every
number in this report reproduces exactly.

### Latent matrices and induced orders

**`U_A`** — roles (0,1) -> CPA (0,1); 0 > 1

```
[[1. 1.]
 [0. 0.]]

induced precedence h(U)  (1 means row > column)
            0       1
    0       .       1
    1       .       .
```

**`U_B_TOTAL`** — roles (0,1,2) -> CPA (0,1,2); 0 > 1 > 2. Stages 0-2B

```
[[2. 2.]
 [1. 1.]
 [0. 0.]]

induced precedence h(U)  (1 means row > column)
            0       1       2
    0       .       1       1
    1       .       .       1
    2       .       .       .
```

**`U_B_ANTICHAIN`** — roles (0,1,2) -> CPA (0,1,2); no order at all. Stage 3 only

```
[[1.  0. ]
 [0.  1. ]
 [0.5 0.5]]

induced precedence h(U)  (1 means row > column)
            0       1       2
    0       .       .       .
    1       .       .       .
    2       .       .       .
```

**`U_C`** — roles (0,1) -> CPA (3,4); 3 > 4

```
[[1. 1.]
 [0. 0.]]

induced precedence h(U)  (1 means row > column)
            0       1
    0       .       1
    1       .       .
```

## 2. Stage 0 — exact posterior with fixed U

Trace `x = (0, 1, 2, 0, 1)`. Support-compatible complete states: **2** (asserted; no third state exists).

| state | blocks | log target | exact P(S \| x) |
|---|---|---|---|
| A->B | [0, 1]_A + [2, 0, 1]_B | -8.303863 | 0.016949152542373 |
| B->A | [0, 1, 2]_B + [0, 1]_A | -4.243420 | 0.983050847457627 |

- `P_exact(S_BA) = 0.983050847457627`  (target 0.983050847457627)
- `P_exact(S_AB) = 0.016949152542373`  (target 0.016949152542373)
- `sum_S P(S) = 1.000000000000000`

Boundary marginals (half-open cut positions):

- `P(B_3 = 1 | x) = 0.983050847457627`
- `P(B_2 = 1 | x) = 0.016949152542373`

Both states have `L = 2`, so the boundary prior and the uniform label prior
cancel exactly and the posterior ratio is the BPOP emission ratio:
`0.9425 * 0.975 = 0.9189375` against `0.975 * 0.01625 = 0.01584375`.

## 3. Stage 1 — segmentation MCMC against the exact posterior

100,000 iterations, 5,000 burn-in, 95,000 kept, seed `20260808`, acceptance rate **0.0347**.

| state | exact | MCMC | abs error |
|---|---|---|---|
| S_BA | 0.983051 | 0.982642 | 0.000409 |
| S_AB | 0.016949 | 0.017358 | 0.000409 |

**Total variation distance = 0.000409** (criterion < 0.01)

| cut | exact | MCMC | abs error |
|---|---|---|---|
| B_2 | 0.016949 | 0.017358 | 0.000409 |
| B_3 | 0.983051 | 0.982642 | 0.000409 |

Indicator `1[S = S_BA]`: autocorrelation lag-1 = -0.0177, lag-5 = 0.0002, ESS = 95,000.

## 4. Stage 2A — MCMC over latent U, segmentation known

4 chains, 15,000 iterations, 3,000 burn-in, thin 3, `sigma_U = 0.8`, `rho_U = 0.25`, `d = 2`.

Recovery is judged on **induced precedence relations**, never on raw U
coordinates — U is not identifiable, only h(U) is.

### Skill A — 100 executions

| permutation | count |
|---|---|
| (0, 1) | 97 |
| (1, 0) | 3 |

| chain | acceptance | P(0>1) |
|---|---|---|
| 0 | 0.4632 | 1.0000 |
| 1 | 0.4617 | 1.0000 |
| 2 | 0.4635 | 1.0000 |
| 3 | 0.4649 | 1.0000 |
| **pooled** | | 1.0000 |

Pooled relation posterior (row > column):

```
            0       1
    0  0.0000  1.0000
    1  0.0000  0.0000
```

R-hat(log posterior) = 1.0010; ESS(log posterior) by chain = [884, 862, 772, 878].
Likelihood-table cache: 3 distinct orders visited, 120,001 hits.

### Skill B — 200 executions

| permutation | count |
|---|---|
| (0, 1, 2) | 191 |
| (0, 2, 1) | 2 |
| (1, 0, 2) | 5 |
| (2, 0, 1) | 2 |

| chain | acceptance | P(0>1) | P(1>2) | P(0>2) |
|---|---|---|---|---|
| 0 | 0.3549 | 1.0000 | 1.0000 | 1.0000 |
| 1 | 0.3543 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 0.3603 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 0.3569 | 1.0000 | 1.0000 | 1.0000 |
| **pooled** | | 1.0000 | 1.0000 | 1.0000 |

Pooled relation posterior (row > column):

```
            0       1       2
    0  0.0000  1.0000  1.0000
    1  0.0000  0.0000  1.0000
    2  0.0000  0.0000  0.0000
```

R-hat(log posterior) = 1.0017; ESS(log posterior) by chain = [397, 613, 567, 778].
Likelihood-table cache: 17 distinct orders visited, 179,987 hits.

## 5. Stage 2B — joint segmentation + U MCMC

40 traces (20 true `B -> A`, 20 true `A -> B`), of which **20 are genuinely ambiguous** (2 legal states); the rest have a single legal state.
4 chains, 15,000 iterations, 3,000 burn-in, thin 3.

| chain | boundary P | boundary R | boundary F1 | skill-path acc | seg accept | U accept |
|---|---|---|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0342 | 0.4030 |
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0341 | 0.4020 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0339 | 0.4011 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0350 | 0.4051 |

**Boundary F1 (worst chain) = 1.0000** (criterion >= 0.85)
**Skill-path accuracy (worst chain) = 1.0000**

**Ordered-pair (precedence) F1 = 1.0000** (precision 1.0000, recall 1.0000, 4/4 true relations recovered)

Pooled relation posterior, skill A:

```
            0       1
    0  0.0000  1.0000
    1  0.0000  0.0000
```

Pooled relation posterior, skill B:

```
            0       1       2
    0  0.0000  1.0000  1.0000
    1  0.0000  0.0000  1.0000
    2  0.0000  0.0000  0.0000
```

## 6. Stage 3A — Dirichlet transition Gibbs

Manual transition counts:

```
            A       B       C
    A  0.0000  2.0000 30.0000
    B 30.0000  0.0000  2.0000
    C  0.0000  0.0000  0.0000
```

| row | allowed next | Dirichlet alpha |
|---|---|---|
| A | B, C | (3, 31) |
| B | A, C | (31, 3) |
| C | A, B | (1, 1) |

50,000 Gibbs draws, seed `20260808`.

| transition | analytic mean | empirical mean | abs error |
|---|---|---|---|
| A -> B | 0.0882352941 | 0.0882477618 | 0.000012 |
| A -> C | 0.9117647059 | 0.9117522382 | 0.000012 |
| B -> A | 0.9117647059 | 0.9115264027 | 0.000238 |
| B -> C | 0.0882352941 | 0.0884735973 | 0.000238 |
| C -> A | 0.5000000000 | 0.5010482011 | 0.001048 |
| C -> B | 0.5000000000 | 0.4989517989 | 0.001048 |

Worst absolute error = 0.001048 (criterion < 0.01).

## 7. Stage 3B — transition context resolves an ambiguous boundary

Trace `x = (0, 1, 2, 0, 1)`, using **`U_B_ANTICHAIN`** so that
every B permutation has probability exactly 1/6 and the two states have
identical emission terms.

| state | without transitions | with transition context | expected |
|---|---|---|---|
| S_BA | 0.500000000000000 | 0.911764705882353 | 0.911764705882353 |
| S_AB | 0.500000000000000 | 0.088235294117647 | 0.088235294117647 |

This is the headline result: a segmentation that the local partial-order
likelihood cannot distinguish at all (exactly 0.5 / 0.5) is resolved to
0.9118 / 0.0882 purely by skill-transition context.

## 8. Stage 3C (optional) — joint segmentation + transition Gibbs

64 traces, 9 ambiguous, U held fixed. 4,000 iterations, 1,000 burn-in.

True transition counts:

```
            A       B       C
    A  0.0000  2.0000 30.0000
    B 30.0000  0.0000  2.0000
    C  0.0000  0.0000  0.0000
```

Posterior mean transition matrix:

```
            A       B       C
    A  0.0000  0.0714  0.9286
    B  0.9117  0.0000  0.0883
    C  0.5014  0.4986  0.0000
```

- B prefers A over C: **True**
- A prefers C over B: **True**

## Deviations, warnings and notes

- sigma_U was calibrated once to 0.8 (the spec suggested 0.25, which gave acceptance ~0.80, above the [0.10, 0.60] band). No adaptation happens during saved sampling.
- Stage 1's acceptance rate is low by construction: the posterior is 0.983/0.017, so most proposals to the minority state are correctly rejected.
- Stage 2B's segmentation acceptance rate is likewise low for the same reason, and about half the traces have a single legal state where the update is a no-op.
- The BPOP likelihood depends on U only through h(U), so the likelihood surface is piecewise constant and the U chains mix by jumping between order regions.
- Stage 3C is a weaker test than it looks: a B->A trace is only ambiguous when its sampled B permutation happens to end in role 2 (probability 1/3 under the antichain), so most of its 64 traces have a single legal state and the transition posterior is largely pinned by unambiguous data. Stage 3B remains the sharp, deterministic test of ambiguity resolution.

