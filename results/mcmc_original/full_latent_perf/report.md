# FULL-LATENT performance diagnosis (profiling only)

Status: **measurement complete, nothing optimised**. No file in `src/` was edited and the
live 8-worker formal chain was not touched. Every benchmark ran on copies of checkpoint
state under private RNGs.

Source commit `3f99926`. Corpus `matched_synthetic_formal_corpus`: N=100 traces,
J in {24,32,40,48} (25 each), K=3 skills, 5 roles, min_width=3, max_width=12, delta_b=0.15.

## Baseline caveat, which governs every absolute number below

There is **no valid uncontended baseline**. The item-1 run started 16:25:18; the formal
chain relaunched 16:26:00 with 8 workers on a 10-core box, mid-COND-arm. Item-1 COND is
therefore part-contended (median 1.876 s), the Part-A rerun is fully contended
(median 2.582 s), and the only uncontended observation is item-1's `min 0.736 s`.

**Percentages and ratios are within-process and robust; milliseconds are not.** Absolute
timings should be re-measured on an idle box before any speedup is quoted externally.

## A. The COND bimodality is ambient, not model-driven

200 per-sweep records, COND. Work counters are flat:

| counter | min | max | cv |
|---|---|---|---|
| `lse_calls` | 10,300 | 10,300 | 0.000% |
| `legal_blocks` | 88,500 | 88,500 | 0.000% |
| `pt_options` | 170,696 | 171,556 | 0.082% |
| `segments` | 537 | 582 | 1.505% |

Wall time vs `pt_options`: r = 0.047, **R^2 = 0.002**. One distinct H hash across all 200
sweeps. Under uniform contention the distribution is unimodal (median 2.582, mean 2.606).
The item-1 bimodality was the formal-chain launch, not any quantity in the sampler.

**FULL-MARG is the opposite**: `pt_options` cv = 48%, and wall vs `pt_options`
**R^2 = 0.9907**. `lse_calls` takes exactly two values, 10,300 and 30,900 = 3x, on 12 of
120 sweeps (cadence 10). This is an independent counter-level confirmation of D.

Incidental (mixing, not performance): across both arms **0 of 22 H-changing proposals were
accepted**; H is frozen at `structural_scale = 0.5`.

## B. `predecessor_terms` is exactly O(J D K^2), with nothing shared

Census over one real forward pass, all 100 traces:

| quantity | measured | predicted | match |
|---|---|---|---|
| calls | 10,800 | J*K = 10,800 | exact |
| inner (a,h) pairs | 256,500 | J*D*K^2 = 256,500 | exact |
| block-score reads | 85,500 | legal (a,b,k) minus a=0 | exact |
| options returned | 162,000 | - | 36.8% pruned |
| options per call | mean 15.0, median 20, max 20 | - | |

`log_transition_matrix[h,k]` and `alpha[a,h]` are both fetched inside the innermost loop
(`semi_markov_ffbs.py:172-180`), so the h-sum is recomputed for every b. Nothing is shared.

Empirical exponent in K (J=40, D=12, K in {3,5,10,20}): **1.372**, not 2.0, because at
small K the fixed per-call cost dominates the K^2 term.

## C. 94.6% of logsumexp time is scipy dispatch, not arithmetic

On the real size distribution (6,000 recorded arrays, mean size 15.8):

| component | us/call | % of scipy call |
|---|---|---|
| python call overhead | 0.039 | 0.0% |
| reduction arithmetic | 8.207 | 5.7% |
| allocation/copy | -0.466 | -0.3% |
| **scipy array-API shim** | **135.233** | **94.6%** |
| scipy total | 143.013 | 100% |

Independently reproduced with `timeit` (scipy 1.18.0, numpy 2.4.6): scipy `logsumexp` costs
~134 us at n=16 and ~133 us at n=100 -- **size-independent**, i.e. pure per-call overhead --
against 7.1 us inline. Ratio **18.9x**, agreeing to 4.4e-16.

The shim is visible in the profile: `is_torch_array` 61,800 calls/pass, `is_jax_array`
30,900, `_is_jax_zero_gradient_array` 30,900.

## D. MARG runs 3 all-trace forward passes per structural sweep; 2 are necessary

`CollapsedULikelihood._fingerprint` includes `state.pi` and `state.transition`, and
`gibbs_pi_p` runs *after* FFBS in `full_latent_sweep_once`. So the fingerprint always misses
at the top of a structural sweep -- hence measured `cache_hits = 0`, `evaluations = 2`.

```
(U, pi, P) --> pass 1  base log Z          [delta_for_candidate -> log_z_per_trace]
(U', pi, P) --> pass 2  candidate log Z    [delta_for_candidate -> _forward_all]
   accept/reject (pi, P UNCHANGED)
(U_acc, pi, P) --> pass 3  FFBS chart      [ffbs_segmentation_draw]
   backward sample --> pi/P Gibbs (pi, P change only HERE)
```

pi and P are frozen across the structural step and FFBS, so **pass 3 is mathematically
identical to pass 2 on accept and pass 1 on reject**. `_forward_all` keeps only
`.log_normalizer` and discards alpha; retaining alpha costs 100 x 49 x 3 x 8 = 118 kB.

| | value |
|---|---|
| current all-trace passes / structural sweep | 3 (measured: `lse_calls` 30,900 = 3 x 10,300) |
| mathematically minimum | 2 |
| per collapsed forward | 2,117 ms; one `ffbs_forward` = 2,125 ms |
| saved per structural sweep | ~2,117 ms |
| amortised at cadence 1/10 | ~212 ms/sweep = **7.3% of the MARG sweep** |

**Caveat:** the two table builders are not bit-identical. `BlockScoreTable` (batched, used by
FFBS) and `FastBlockScoreTable` (used by the collapsed likelihood) agree to **3.55e-15** with
an identical -inf pattern, but not bitwise. Reusing a chart across them perturbs the chain at
~16 ULP. Bit-exact reuse requires both paths to share one table object.

## E. The factorisation is worth little at K=3 and a great deal at K=20

With `r[a,k] = LSE_h(alpha[a,h] + logP[h,k])` computed once per a (the log-domain form of
`r_a = alpha[a-1,:] @ P`), the recursion becomes O(J K^2 + J D K).

Real corpus, K=3: 1.951 s -> 1.613 s = **1.209x**, max disagreement **4.26e-14**.

Synthetic, J=40, D=12:

| K | current ms | factorised ms | speedup | op ratio | lse calls | disagreement |
|---|---|---|---|---|---|---|
| 3 | 21.08 | 12.54 | 1.68x | 2.40 | 120 -> 80 | 3.6e-15 |
| 5 | 34.92 | 15.07 | 2.32x | 3.53 | 200 -> 80 | 1.8e-15 |
| 10 | 97.17 | 14.49 | 6.70x | 5.45 | 400 -> 80 | 1.8e-15 |
| 20 | 273.46 | 18.03 | **15.16x** | 7.50 | 800 -> 80 | 1.8e-15 |

Empirical exponent in K: current **1.372**, factorised **0.162** (essentially flat).

## F. The forward pass is 4% arithmetic

Composition of one forward pass (2.046 s, best of 3, uninstrumented), from C:

| | share of forward |
|---|---|
| scipy array-API dispatch | **68.1%** |
| our Python (`predecessor_terms`, `forward`, `list.append`) | 28.0% |
| logsumexp reduction arithmetic | 4.2% |

cProfile self-time bucketing over the top 30 functions agrees: dispatch 61.5%, our Python
25.3%, numpy arithmetic (C) 5.6%, allocation 2.0%, other 5.6%.

**The bottleneck is implementation overhead, not mathematical work.**

## G. Same-length batching: 31.2x on the forward pass, exact

The four length classes of 25 traces vectorise exactly on the trace axis.

| J | n | sequential current | batched | speedup | disagreement in log Z | stack |
|---|---|---|---|---|---|---|
| 24 | 25 | 304.6 ms | 10.3 ms | 29.6x | 1.4e-14 | 0.36 MB |
| 32 | 25 | 422.9 ms | 15.9 ms | 26.6x | 1.4e-14 | 0.63 MB |
| 40 | 25 | 553.5 ms | 18.1 ms | 30.5x | 2.8e-14 | 0.98 MB |
| 48 | 25 | 681.8 ms | 18.5 ms | 36.8x | 2.8e-14 | 1.41 MB |

All traces: **1.963 s -> 0.063 s = 31.2x**. Batching collapses ~10,300 logsumexp calls into
~288, so it defeats the C bottleneck and the B bottleneck at once. It changes the dominant
bottleneck: after batching, emission construction becomes the largest single term.

## H. Memory is driven by the dense J^2 layout, not by N or K

Measured (N=100, K=3): emission tables 3.389 MB, all sampler table arrays 8.633 MB, forward
alpha across all traces 0.089 MB, transient peak per forward 0.022 MB, per backward 0.003 MB.
Against ~193 MB RSS, model data is <5%: the rest is interpreter plus numpy/scipy.

The block-score table is stored densely as (J, J+1, K) per trace = O(N J^2 K), but only the
band `min_width <= b-a <= max_width` can ever be finite = O(N J D K).

Extrapolation, N=100 (arithmetic, nothing allocated):

| | dense | banded | ratio | alpha |
|---|---|---|---|---|
| J=50, K=3 | 6.1 MB | 1.0 MB | 5.9x | 0.12 MB |
| J=100, K=20 | 161.6 MB | 15.0 MB | 10.8x | 1.62 MB |
| J=200, K=20 | 643.2 MB | 31.0 MB | 20.8x | 3.22 MB |
| J=500, K=20 | **4,008 MB** | **79 MB** | 50.8x | 8.02 MB |
| J=500, K=3 | 601.2 MB | 11.8 MB | 50.8x | 1.20 MB |

**J drives memory quadratically only because of the dense layout.** The information content
is O(N J D K) and linear in J. N and K are both linear and harmless. Nothing here supports a
"memory-hungry" characterisation at the current scale.

## I. Consolidated table

Shares are robust; ms are contended and drift (see caveat).

### FULL-COND (item-1 mean 1,554 ms/sweep)

| Component | ms/sweep | % | theoretically removable | evidence |
|---|---|---|---|---|
| forward - scipy dispatch | 904 | **58.2%** | ~100%, 18.9x, agrees 4.4e-16 | C, F |
| forward - our Python | 372 | 23.9% | most, via batching (31.2x) | B, F, G |
| emission build | 179 | 11.5% | 100%, H never changed | item 1, A |
| forward - lse arithmetic | 55 | 3.5% | no - irreducible work | C |
| backward/FFBS | 32 | 2.1% | partly | item 1 |
| complete log-target | 12 | 0.7% | ~100% | item 1 |
| piP + structural + other | 2 | 0.2% | negligible | item 1 |

### FULL-MARG (item-1 mean 2,895 ms/sweep)

| Component | ms/sweep | % | theoretically removable | evidence |
|---|---|---|---|---|
| forward - scipy dispatch | 1,735 | **59.9%** | ~100%, 18.9x | C, F |
| forward - our Python | 713 | 24.6% | most, via batching | B, F, G |
| emission build | 276 | 9.5% | 100%, H never changed | item 1, A |
| forward - lse arithmetic | 106 | 3.7% | no | C |
| backward/FFBS | 50 | 1.7% | partly | item 1 |
| complete log-target | 17 | 0.6% | ~100% | item 1 |
| piP + structural + other | 4 | 0.1% | negligible | item 1 |
| (of the forward total, redundant 3rd pass) | 212 | 7.3% | 100% | D |

### Ranking by measured potential saving

1. **scipy logsumexp dispatch** - 58-60% of the sweep, ~100% removable, one function
2. **Python-level forward recursion** - 24-25%, removable by batching
3. **Emission rebuild** - 9.5-11.5%, 100% removable by H-hash cache
4. **MARG redundant 3rd forward pass** - 7.3% of MARG only
5. everything else - <1% combined

### Amdahl bounds

| | COND | MARG |
|---|---|---|
| perfect emission optimisation | 1.130x | 1.105x |
| perfect forward optimisation | 6.88x | 8.34x |
| perfect non-forward optimisation | 1.17x | 1.14x |
| **measured** inline logsumexp only | **2.39x** | **2.50x** |
| **measured** batched DP (31.2x forward) | **5.79x** | **6.75x** |
| **measured** batched + emission cache | **17.47x** | **18.96x** |

## Answers

1. **What dominates?** The forward semi-Markov recursion: 85.5% COND, 88.0% MARG. But
   within it the cost is not arithmetic -- scipy's `logsumexp` per-call dispatch alone is
   ~58-60% of the entire sweep.
2. **Why?** `forward` calls `predecessor_terms` J*K times and `logsumexp` 10,300 times per
   sweep on arrays averaging 15.8 elements. scipy 1.18's `logsumexp` carries ~134 us of
   size-independent array-API overhead (torch/jax checks) against ~7 us of arithmetic, so
   the per-call overhead is 18.9x the work being done.
3. **Three mechanisms with the largest measured potential:** (a) inline logsumexp, 18.9x on
   the call, 2.4-2.5x on the sweep; (b) same-length-class batching of the DP, 31.2x on the
   forward pass, 5.8-6.8x on the sweep; (c) H-hash emission cache, 1.10-1.13x, with a
   measured 100% hit rate. The O(JK^2+JDK) factorisation is a distant fourth at K=3 (1.21x)
   but first-rank at K=20 (15.2x).
4. **Evidence-based ceiling before GPU or approximation:** ~17-19x, all transformations
   verified exact to <=4.3e-14.
5. **What remains uncertain:** no uncontended baseline exists; the integration cost of
   batching is unmeasured (31.2x is the DP in isolation); after batching the backward
   sampler and emission build become dominant and were not optimised in this analysis;
   the K=3 conclusions do not transfer to K=20, where the factorisation reverses the
   ranking; and H being frozen is what makes the emission cache hit 100% -- fixing the
   mixing problem would lower that hit rate.

## Artifacts

| file | contents |
|---|---|
| `sweep_profile_item1_baseline_300.json` | item 1, 300 sweeps/arm, 7-bucket decomposition |
| `persweep_A_contended.json` | A, per-sweep records with work counters |
| `forward_anatomy.json` | B census + K-scaling, C decomposition, F attribution |
| `dp_factorisation_bench.json` | E factorisation across K, G batching |
| `memory_decomposition.json` | H measured bytes and extrapolation |

Scripts (all measurement-only, none modify `src/`): `scripts/full_latent_sweep_profile.py`,
`full_latent_persweep_profile.py`, `full_latent_forward_anatomy.py`,
`full_latent_dp_factorisation_bench.py`, `full_latent_memory_decomposition.py`.
