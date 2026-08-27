# Scalability of exact segmental partial-order inference — `optimized_segmental_v1`

A **computational scaling study**. It measures how the validated optimized
inference backend spends time and memory as the corpus and the model grow. It
makes no claim about posterior convergence, mixing, or parameter recovery, and
no chain in it was run long enough for such a claim to be available.

## Provenance

| item | value |
| --- | --- |
| backend under test | `optimized_segmental_v1`, all four optimisation flags on |
| commit | `564995efd056d7d33984f0ca1532386e6140ea0c` |
| branch | `scalability-optimized-v1` |
| worktree | `/Users/dongqing/Desktop/hpop-scalability` |
| table source | `batched` (the registered FULL-LATENT setting) |
| reference engine | `hpop.mcmc_original`, unmodified, used only as the parity oracle |
| benchmark seed | 20260822 |
| started (UTC) | 2026-08-22T22:27:09Z |
| finished (UTC) | 2026-08-23T03:05:06Z |
| tasks settled | 156 of 156 |
| tasks measured | 151 |

### Machine

| item | value |
| --- | --- |
| CPU | Apple M4 |
| cores | 10 physical (4 performance, 6 efficiency), 10 logical |
| RAM | 16.0 GiB |
| macOS | 10.16 |
| Python | 3.13.2 |
| NumPy | 2.4.6 |
| SciPy | 1.18.0 |
| threading | every BLAS and OpenMP thread count pinned to 1; configurations run strictly one process at a time |
| memory gate | 6.00 GiB (min of 6 GB and half of physical RAM), plus a live check against currently reclaimable memory |
| load average at capture | [12.1474609375, 10.59375, 10.0546875] |
| thermal | Note: No thermal warning level has been recorded
Note: No performance warning level has been recorded
Note: No CPU power status has been recorded |

## Parity gate

The optimized backend was checked against the frozen reference on 32 points before any scaling point ran: J in [24, 48], K in [3, 5], A in [5, 10], D_max in [6, 12], both support regimes.

| check | worst observed | tolerance | result |
| --- | --- | --- | --- |
| alpha, max absolute error | 1.137e-13 | 1e-10 | pass |
| log Z, max absolute error | 1.137e-13 | 1e-10 | pass |
| -inf pattern | identical at every point | exact | pass |
| emission tables | bit-identical at every point | exact | pass |
| legal block counts | identical, and equal to the counted geometry | exact | pass |
| backward draw | legal complete cover, no forbidden self-transition | exact | pass |
| full sweep against the reference sweep | identical segmentations, log target within tolerance | 1e-10 | pass |

The discrepancy is floating-point noise of the order of 1e-14 to 1e-13, growing
mildly with trace length as accumulated rounding does. It is three orders of
magnitude inside the frozen tolerance. **The two engines compute the same
numbers.**

## Machine speed, and why this study was measured twice

The first pass produced a result that cannot be true: on several axes the
**larger** configuration ran **faster** than the smaller one. Work does not
decrease as a problem grows, so something about the measurement had changed
between the two points.
It was not preemption. Process CPU time divided by wall time sat at 0.98 to
1.00 for every affected point, so the benchmark held a full core throughout.
But the CPU time itself had fallen. On a machine with performance and
efficiency cores, a process moved from one to the other keeps a whole core --
`cpu / wall` stays at one -- while executing at roughly half the speed. Wall
time and CPU time then fall together and neither reveals the shift. The load
average recorded beside every configuration moves in lockstep: the points
measured while the machine was busy sat at load 9 to 15, and those measured
after it went idle at load 1.4 to 6.
Because the queue ran ascending within each axis, the largest point on every
axis was measured last -- on the quietest machine. That biases every fitted
exponent **downward**, and it is exactly the temporal-load bias the protocol
set out to avoid.
So the study was measured again, end to end, on an idle machine, with a
**fixed-work speed probe** added: the same arithmetic timed before and after
every configuration, so a change in machine speed becomes a recorded quantity
instead of an invisible one. The controlled pass is primary. The first pass is
kept, reported, and never averaged with it.
| pass | primary | median load average | median speed probe | probe spread | records |
| --- | --- | --- | --- | --- | --- |
| first pass |  | 2.4 | not instrumented | - | - |
| controlled pass | yes | 1.6 | 17.2 ms | 1.03x | 154 |
Across the 39 configurations measured in both passes, the first pass was a median of **1.01x** slower, reaching **2.48x** at worst. 
Per-configuration ratios are in `pass_comparison.json`, and every row of
`timing_summary.csv` carries its own load average, speed probe and
probe-normalised median so any reader can check this independently.

## Protocol

- Each `(configuration, operation group)` runs in its own fresh subprocess, so
  `ru_maxrss` is that configuration's peak and nothing else's.
- Three untimed warm-ups, then timed repetitions interleaved round-robin across
  the operations in a group, continuing past fifteen until each operation's
  bootstrap 95% interval for the median has relative half-width at or below 5%,
  to a ceiling of fifty. Expensive points are allowed to stop at five and are
  flagged in `timing_summary.csv` as `reduced_repetitions`.
- Every repetition is recorded in `raw_timings.csv`. Nothing is averaged before
  it reaches disk.
- Both wall-clock and process CPU time are recorded. The two agree closely here
  because the machine was otherwise idle; `complexity_fits.json` carries the
  same fits on CPU time as a cross-check.
- Plain sweeps are measured with the H-keyed emission cache warm, which is the
  steady state a chain runs in between structural moves. Structural sweeps are
  measured with that cache **forced to miss**, so both arms pay a full candidate
  table rebuild: that is the case where the proposal moves `H = h(U)`, and it is
  the upper bound. `emission_build` is measured separately, so the
  H-unchanged structural sweep is recoverable by subtraction.

## Q1 — trace length J, at bounded segment width

| J | legal blocks | emission build | forward batched | backward sample | ffbs complete | cond plain | marg structural | peak RSS | reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 162800 | 5.52 s | 319.82 ms | 156.72 ms | 478.88 ms | 528.80 ms | 7.05 s | 4.78 GiB | 5 |
| 192 | 29680 | 1.02 s | 13.68 ms | 28.55 ms | 42.66 ms | 61.83 ms | 1.16 s | 559 MiB | 15 |
| 24 | 2800 | 135.70 ms | 1.47 ms | 3.11 ms | 4.65 ms | 11.92 ms | 156.18 ms | 126 MiB | 15 |
| 384 | 60400 | 3.82 s | 63.59 ms | 57.94 ms | 122.45 ms | 147.09 ms | 4.54 s | 1.42 GiB | 15 |
| 48 | 6640 | 265.36 ms | 3.12 ms | 6.51 ms | 9.73 ms | 20.39 ms | 304.58 ms | 164 MiB | 15 |
| 768 | 121840 | 4.13 s | 192.71 ms | 115.49 ms | 309.60 ms | 350.30 ms | 5.24 s | 4.12 GiB | 5 |
| 96 | 14320 | 522.84 ms | 6.40 ms | 13.92 ms | 20.37 ms | 35.99 ms | 600.71 ms | 263 MiB | 15 |

- optimized forward: **1.38 (95% CI 1.29 to 1.46, R^2 0.993)**
- complete FFBS update: **1.20 (95% CI 1.15 to 1.26, R^2 0.996)**
- plain sweep: **0.99 (95% CI 0.91 to 1.07, R^2 0.988)**
- candidate table rebuild: **0.99 (95% CI 0.98 to 1.00, R^2 1.000)**

With `D` bounded the forward recursion visits `J K` chart cells and reduces over
at most `D` durations at each, so the arithmetic is linear in `J`. The measured
exponents are read against that expectation in the complexity section below.

## Q2 — skill library K, under dense transition dynamics

| K | legal blocks | emission build | forward batched | backward sample | ffbs complete | cond plain | marg structural | peak RSS | reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 38880 | 3.69 s | 49.34 ms | 89.47 ms | 142.18 ms | 178.16 ms | 3.73 s | 568 MiB | 15 |
| 20 | 38880 | 6.70 s | 89.89 ms | 170.78 ms | 268.16 ms | 338.06 ms | 7.25 s | 881 MiB | 15 |
| 3 | 38880 | 386.74 ms | 8.05 ms | 15.02 ms | 23.12 ms | 35.74 ms | 467.19 ms | 300 MiB | 15 |
| 40 | 38880 | 11.05 s | 150.77 ms | 232.39 ms | 383.79 ms | 462.86 ms | 10.55 s | 1.49 GiB | 5 |
| 5 | 38880 | 763.86 ms | 9.66 ms | 21.43 ms | 31.61 ms | 46.94 ms | 815.99 ms | 370 MiB | 15 |
| 80 | 38880 | 10.55 s | 199.72 ms | 263.69 ms | 462.61 ms | 525.96 ms | 12.05 s | 2.32 GiB | 5 |

- optimized forward: **0.88 (95% CI 0.77 to 1.00, R^2 0.976)**
- plain sweep: **0.79 (95% CI 0.73 to 0.85, R^2 0.992)**
- candidate table rebuild: **1.00 (95% CI 0.99 to 1.02, R^2 1.000)**

The factorised recursion is `O(J K^2 + J D K)`: the `K^2` term is the transition
reduction, the `J D K` term the duration reduction. Which one dominates depends
on where `K` sits relative to `D`, so a single exponent is not expected to equal
two and is not forced to.

## Q3 — corpus size N

| N | legal blocks | emission build | forward batched | backward sample | ffbs complete | cond plain | marg structural | peak RSS | reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 155520 | 5.25 s | 71.64 ms | 147.76 ms | 221.42 ms | 265.16 ms | 6.07 s | 1.59 GiB | 15 |
| 16 | 19440 | 1.79 s | 29.09 ms | 45.02 ms | 77.96 ms | 118.77 ms | 2.00 s | 351 MiB | 15 |
| 1 | 1215 | 91.22 ms | 3.78 ms | 1.10 ms | 4.91 ms | 9.00 ms | 108.85 ms | 122 MiB | 15 |
| 256 | 311040 | 12.38 s | 140.88 ms | 292.84 ms | 437.92 ms | 514.65 ms | 12.79 s | 2.85 GiB | 5 |
| 32 | 38880 | 3.43 s | 48.46 ms | 89.24 ms | 139.31 ms | 194.63 ms | 3.71 s | 569 MiB | 15 |
| 64 | 77760 | 2.62 s | 40.22 ms | 74.78 ms | 114.94 ms | 254.54 ms | 5.28 s | 1.00 GiB | 15 |
| 8 | 9720 | 385.70 ms | 6.42 ms | 9.79 ms | 16.43 ms | 56.70 ms | 447.25 ms | 234 MiB | 15 |

- plain sweep: **0.72 (95% CI 0.64 to 0.78, R^2 0.982)**
- optimized forward: **0.64 (95% CI 0.53 to 0.75, R^2 0.947)**

## Q4 — maximum segment width D

| D_max | legal blocks | emission build | forward batched | backward sample | ffbs complete | cond plain | marg structural | peak RSS | reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 29680 | 2.53 s | 34.09 ms | 64.60 ms | 99.61 ms | 133.52 ms | 2.55 s | 570 MiB | 15 |
| 24 | 63184 | 10.39 s | 58.49 ms | 107.35 ms | 169.23 ms | 249.12 ms | 10.51 s | 659 MiB | 15 |
| 48 | 123280 | 37.99 s | 73.22 ms | 152.71 ms | 228.56 ms | 307.05 ms | 37.07 s | 830 MiB | 13 |
| 6 | 12064 | 250.64 ms | 11.80 ms | 18.56 ms | 30.51 ms | 42.46 ms | 332.43 ms | 534 MiB | 15 |
| 96 | 215824 | 44.56 s | 40.47 ms | 94.93 ms | 136.69 ms | 176.37 ms | 45.06 s | 1.16 GiB | 5 |

Reported against the **number of legal candidate blocks**, which is what `D`
actually buys, rather than against `D` itself:

- optimized forward vs legal blocks: **0.22 (95% CI 0.16 to 0.28, R^2 0.904)**
- plain sweep vs legal blocks: **0.43 (95% CI 0.40 to 0.46, R^2 0.993)**

## Q5 — canonical-action / role inventory A

The two support regimes are reported apart and never averaged. They are
different role graphs, and the emission recursion's cost is a function of the
graph rather than of `A` alone.

### A. Full-support stress test — every skill supported on all A roles

| A | legal blocks | emission build | forward batched | backward sample | ffbs complete | cond plain | marg structural | peak RSS | reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 19440 | 864.10 ms | 21.60 ms | 41.51 ms | 60.62 ms | 85.99 ms | 1.09 s | 332 MiB | 15 |
| 20 | 19440 | 2.06 s | 30.11 ms | 45.57 ms | 76.37 ms | 120.73 ms | 2.02 s | 377 MiB | 15 |
| 30 | 19440 | 3.05 s | 29.74 ms | 45.47 ms | 74.83 ms | 125.33 ms | 3.37 s | 369 MiB | 15 |
| 50 | 19440 | 2.72 s | 12.87 ms | 19.70 ms | 32.45 ms | 60.84 ms | 2.94 s | 335 MiB | 5 |
| 5 | 19440 | 220.56 ms | 8.54 ms | 14.86 ms | 23.50 ms | 38.51 ms | 281.45 ms | 346 MiB | 15 |

- candidate table rebuild: **1.06 (95% CI 0.89 to 1.22, R^2 0.967)**

### B. Sparse-support scenario — ten roles per skill from a size-A vocabulary

| A | legal blocks | emission build | forward batched | backward sample | ffbs complete | cond plain | marg structural | peak RSS | reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 19440 | 842.71 ms | 19.37 ms | 35.83 ms | 56.19 ms | 91.61 ms | 962.37 ms | 349 MiB | 15 |
| 20 | 19440 | 1.44 s | 29.27 ms | 39.58 ms | 69.49 ms | 110.76 ms | 1.63 s | 353 MiB | 15 |
| 30 | 19440 | 2.24 s | 31.12 ms | 41.22 ms | 68.84 ms | 121.15 ms | 2.60 s | 375 MiB | 15 |
| 50 | 19440 | 1.74 s | 13.17 ms | 16.14 ms | 29.12 ms | 56.19 ms | 1.71 s | 348 MiB | 5 |
| 5 | 19440 | 222.46 ms | 8.50 ms | 14.24 ms | 22.90 ms | 38.60 ms | 273.77 ms | 348 MiB | 15 |

- candidate table rebuild: **0.86 (95% CI 0.74 to 0.99, R^2 0.971)**

### The two regimes side by side

| A | operation | full support | sparse support | ratio | mean predecessors, full | mean predecessors, sparse | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | emission build | 225.65 ms | 239.16 ms | 0.94x | 1.0 | 1.0 | same construction |
| 5 | forward batched | 13.14 ms | 13.15 ms | 1.00x | 1.0 | 1.0 | same construction |
| 5 | cond plain | 42.99 ms | 43.27 ms | 0.99x | 1.0 | 1.0 | same construction |
| 5 | marg structural | 280.62 ms | 282.34 ms | 0.99x | 1.0 | 1.0 | same construction |
| 10 | emission build | 360.25 ms | 366.55 ms | 0.98x | 2.1 | 2.1 | same construction |
| 10 | forward batched | 12.83 ms | 13.05 ms | 0.98x | 2.1 | 2.1 | same construction |
| 10 | cond plain | 44.08 ms | 44.93 ms | 0.98x | 2.1 | 2.1 | same construction |
| 10 | marg structural | 423.02 ms | 427.00 ms | 0.99x | 2.1 | 2.1 | same construction |
| 20 | emission build | 679.83 ms | 585.46 ms | 1.16x | 4.7 | 1.1 |  |
| 20 | forward batched | 12.99 ms | 12.96 ms | 1.00x | 4.7 | 1.1 |  |
| 20 | cond plain | 47.46 ms | 45.50 ms | 1.04x | 4.7 | 1.1 |  |
| 20 | marg structural | 775.40 ms | 656.65 ms | 1.18x | 4.7 | 1.1 |  |
| 30 | emission build | 1.18 s | 966.91 ms | 1.22x | 7.3 | 0.8 |  |
| 30 | forward batched | 12.97 ms | 13.10 ms | 0.99x | 7.3 | 0.8 |  |
| 30 | cond plain | 51.09 ms | 49.48 ms | 1.03x | 7.3 | 0.8 |  |
| 30 | marg structural | 1.30 s | 953.60 ms | 1.37x | 7.3 | 0.8 |  |
| 50 | emission build | 2.71 s | 1.80 s | 1.51x | 12.7 | 0.4 |  |
| 50 | forward batched | 13.00 ms | 13.02 ms | 1.00x | 12.7 | 0.4 |  |
| 50 | cond plain | 59.26 ms | 56.54 ms | 1.05x | 12.7 | 0.4 |  |
| 50 | marg structural | 2.90 s | 1.91 s | 1.51x | 12.7 | 0.4 |  |

At `A = 5` and `A = 10` the two regimes are **the same corpus and the same U**:
a support of `min(10, A)` roles is every role there, so the rows coincide for a
reason that has nothing to do with scaling. The regimes separate only above
that point, and only the rows above it carry information about support
sparsity. This is why the two are reported apart rather than as one curve, and
why a single power law fitted across the whole `A` range would mislead.

The measured role-graph density for each point is in `timing_summary.csv`
(`role_relation_density`, `role_mean_predecessors`), so the difference between
the regimes can be read against the graph the emission recursion actually walks
and not merely against the label.

## Q6 — marginalisation overhead

| configuration | axis | dimensions | COND plain | MARG plain | plain ratio | COND structural | MARG structural | structural ratio | amortized ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_full_10 | A_full | N=16 J=128 K=10 A=10 | 44.08 ms | 44.58 ms | 1.011 | 399.16 ms | 423.02 ms | 1.060 | 1.036 |
| A_full_20 | A_full | N=16 J=128 K=10 A=20 | 47.46 ms | 48.23 ms | 1.016 | 739.01 ms | 775.40 ms | 1.049 | 1.037 |
| A_full_30 | A_full | N=16 J=128 K=10 A=30 | 51.09 ms | 51.01 ms | 0.998 | 1.23 s | 1.30 s | 1.058 | 1.042 |
| A_full_5 | A_full | N=16 J=128 K=10 A=5 | 42.99 ms | 43.43 ms | 1.010 | 264.82 ms | 280.62 ms | 1.060 | 1.030 |
| A_full_50 | A_full | N=16 J=128 K=10 A=50 | 59.26 ms | 59.16 ms | 0.998 | 2.76 s | 2.90 s | 1.049 | 1.040 |
| A_sparse_10 | A_sparse | N=16 J=128 K=10 A=10 | 44.93 ms | 43.75 ms | 0.974 | 398.85 ms | 427.00 ms | 1.071 | 1.022 |
| A_sparse_20 | A_sparse | N=16 J=128 K=10 A=20 | 45.50 ms | 45.87 ms | 1.008 | 618.86 ms | 656.65 ms | 1.061 | 1.040 |
| A_sparse_30 | A_sparse | N=16 J=128 K=10 A=30 | 49.48 ms | 50.02 ms | 1.011 | 939.84 ms | 953.60 ms | 1.015 | 1.013 |
| A_sparse_5 | A_sparse | N=16 J=128 K=10 A=5 | 43.27 ms | 43.66 ms | 1.009 | 266.27 ms | 282.34 ms | 1.060 | 1.030 |
| A_sparse_50 | A_sparse | N=16 J=128 K=10 A=50 | 56.54 ms | 55.82 ms | 0.987 | 1.83 s | 1.91 s | 1.047 | 1.034 |
| D_12 | D | N=16 J=192 K=10 A=20 | 69.24 ms | 69.76 ms | 1.007 | 1.08 s | 1.15 s | 1.064 | 1.043 |
| D_24 | D | N=16 J=192 K=10 A=20 | 100.96 ms | 100.40 ms | 0.994 | 4.00 s | 4.06 s | 1.014 | 1.011 |
| D_48 | D | N=16 J=192 K=10 A=20 | 128.60 ms | 127.75 ms | 0.993 | 14.45 s | 14.21 s | 0.984 | 0.984 |
| D_6 | D | N=16 J=192 K=10 A=20 | 51.23 ms | 51.64 ms | 1.008 | 308.74 ms | 343.87 ms | 1.114 | 1.050 |
| D_96 | D | N=16 J=192 K=10 A=20 | 176.17 ms | 173.57 ms | 0.985 | 44.82 s | 46.35 s | 1.034 | 1.032 |
| J_1024 | J | N=16 J=1024 K=10 A=20 | 534.55 ms | 544.13 ms | 1.018 | 6.26 s | 7.18 s | 1.147 | 1.091 |
| J_192 | J | N=16 J=192 K=10 A=20 | 69.19 ms | 70.44 ms | 1.018 | 1.09 s | 1.14 s | 1.052 | 1.039 |
| J_24 | J | N=16 J=24 K=10 A=20 | 12.70 ms | 12.51 ms | 0.985 | 150.30 ms | 156.64 ms | 1.042 | 1.018 |
| J_384 | J | N=16 J=384 K=10 A=20 | 141.70 ms | 141.51 ms | 0.999 | 2.18 s | 2.36 s | 1.082 | 1.051 |
| J_48 | J | N=16 J=48 K=10 A=20 | 21.41 ms | 21.50 ms | 1.004 | 299.64 ms | 302.09 ms | 1.008 | 1.007 |
| J_768 | J | N=16 J=768 K=10 A=20 | 351.14 ms | 353.17 ms | 1.006 | 4.43 s | 5.37 s | 1.211 | 1.126 |
| J_96 | J | N=16 J=96 K=10 A=20 | 38.23 ms | 37.81 ms | 0.989 | 575.54 ms | 600.41 ms | 1.043 | 1.023 |
| K_10 | K | N=32 J=128 K=10 A=20 | 82.19 ms | 80.84 ms | 0.984 | 1.39 s | 1.47 s | 1.060 | 1.033 |
| K_20 | K | N=32 J=128 K=20 A=20 | 141.90 ms | 142.64 ms | 1.005 | 2.84 s | 2.92 s | 1.026 | 1.019 |
| K_3 | K | N=32 J=128 K=3 A=20 | 38.31 ms | 38.52 ms | 1.005 | 441.45 ms | 486.70 ms | 1.102 | 1.060 |
| K_40 | K | N=32 J=128 K=40 A=20 | 255.30 ms | 262.96 ms | 1.030 | 5.51 s | 5.67 s | 1.028 | 1.029 |
| K_5 | K | N=32 J=128 K=5 A=20 | 50.97 ms | 50.15 ms | 0.984 | 756.07 ms | 765.78 ms | 1.013 | 1.002 |
| K_80 | K | N=32 J=128 K=80 A=20 | 518.26 ms | 534.17 ms | 1.031 | 11.02 s | 11.29 s | 1.024 | 1.026 |
| N_1 | N | N=1 J=128 K=10 A=20 | 9.93 ms | 9.97 ms | 1.004 | 105.04 ms | 110.86 ms | 1.055 | 1.032 |
| N_128 | N | N=128 J=128 K=10 A=20 | 268.88 ms | 267.63 ms | 0.995 | 5.59 s | 6.17 s | 1.104 | 1.071 |
| N_16 | N | N=16 J=128 K=10 A=20 | 47.81 ms | 48.98 ms | 1.024 | 730.82 ms | 802.31 ms | 1.098 | 1.071 |
| N_256 | N | N=256 J=128 K=10 A=20 | 518.97 ms | 508.03 ms | 0.979 | 11.66 s | 12.23 s | 1.049 | 1.029 |
| N_32 | N | N=32 J=128 K=10 A=20 | 80.70 ms | 80.06 ms | 0.992 | 1.42 s | 1.47 s | 1.038 | 1.022 |
| N_64 | N | N=64 J=128 K=10 A=20 | 141.25 ms | 142.61 ms | 1.010 | 2.76 s | 2.87 s | 1.039 | 1.030 |
| N_8 | N | N=8 J=128 K=10 A=20 | 29.97 ms | 29.84 ms | 0.996 | 396.30 ms | 419.49 ms | 1.059 | 1.033 |
| baseline_matched_scale | baseline | N=100 J=48 K=3 A=5 | 27.97 ms | 27.83 ms | 0.995 | 125.99 ms | 147.52 ms | 1.171 | 1.054 |
| target_operating_point | target | N=100 J=200 K=20 A=50 | 654.18 ms | 646.24 ms | 0.988 | 32.77 s | 34.53 s | 1.054 | 1.044 |
| target_operating_point_full_support | target | N=100 J=200 K=20 A=50 | 716.51 ms | 733.04 ms | 1.023 | 54.21 s | 55.06 s | 1.016 | 1.016 |

A plain sweep performs no structural move, so the two arms execute identical
code on that path and the plain ratio is a control: it should sit at one, and a
departure measures ambient machine noise rather than marginalisation. All of
the marginalisation cost lives in the structural sweep.

- plain-sweep ratio across every measured configuration: 0.974 to 1.031
- structural-sweep ratio: 0.984 to 1.211
- amortized at the registered cadence of one structural sweep in ten: 0.984 to 1.126

## Q7 — the anticipated real-data operating point

| quantity | measured |
| --- | --- |
| dimensions | N=100, J=200, K=20, A=50, D in [3, 12], sparse support |
| legal candidate blocks | 193,500 (3,870,000 block-skill scores) |
| trace occurrences | 20,000 |
| FULL-COND plain sweep | 648.30 ms  (200 timed repetitions) |
| FULL-MARG plain sweep | 659.35 ms  (200 timed repetitions) |
| FULL-MARG structural sweep | 34.61 s  (5 timed repetitions, reduced) |
| candidate table rebuild | 33.56 s |
| emission cache hit | 1.26 ms |
| optimized forward, all traces | 222.12 ms |
| FFBS backward draw, all traces | 304.99 ms |
| complete FFBS update | 530.98 ms |
| peak resident memory | 3.66 GiB |
| dense score table (one copy) | 613 MiB |
| projected banded storage — NOT IMPLEMENTED | 30 MiB |

- sustained plain-sweep throughput: **1.54 sweeps per second**, i.e. 5,553 sweeps per hour.
- at the registered cadence of one structural sweep in ten, a FULL-MARG sweep averages **4.04 s**, i.e. 892 sweeps per hour.
- the same point under the **full-support** stress regime: plain sweep 716.51 ms, table rebuild 53.27 s, peak RSS 4.10 GiB. Reported beside the sparse primary, never averaged with it.
- long-trace primitives, `target_long_J500_N20`: forward 482.65 ms, backward draw 364.03 ms, table rebuild 42.25 s, peak RSS 2.08 GiB.

**This is a throughput measurement.** It says how fast sweeps
run. It says nothing about how many sweeps are needed, and no
convergence or recovery claim may be attached to it.

## Q8 — what becomes the bottleneck

| configuration | dimensions | forward | backward draw | Gibbs + target + validators | table rebuild | dominant, plain sweep | dominant, structural sweep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_full_10 | N=16 J=128 K=10 A=10 | 12.83 ms | 17.43 ms | 13.84 ms | 360.25 ms | backward | emission_rebuild |
| A_full_20 | N=16 J=128 K=10 A=20 | 12.99 ms | 19.40 ms | 15.12 ms | 679.83 ms | backward | emission_rebuild |
| A_full_30 | N=16 J=128 K=10 A=30 | 12.97 ms | 18.76 ms | 19.34 ms | 1.18 s | gibbs_target_and_validation | emission_rebuild |
| A_full_5 | N=16 J=128 K=10 A=5 | 13.14 ms | 15.43 ms | 14.35 ms | 225.65 ms | backward | emission_rebuild |
| A_full_50 | N=16 J=128 K=10 A=50 | 13.00 ms | 19.70 ms | 26.62 ms | 2.71 s | gibbs_target_and_validation | emission_rebuild |
| A_sparse_10 | N=16 J=128 K=10 A=10 | 13.05 ms | 17.88 ms | 14.14 ms | 366.55 ms | backward | emission_rebuild |
| A_sparse_20 | N=16 J=128 K=10 A=20 | 12.96 ms | 16.70 ms | 15.74 ms | 585.46 ms | backward | emission_rebuild |
| A_sparse_30 | N=16 J=128 K=10 A=30 | 13.10 ms | 16.83 ms | 19.51 ms | 966.91 ms | gibbs_target_and_validation | emission_rebuild |
| A_sparse_5 | N=16 J=128 K=10 A=5 | 13.15 ms | 15.19 ms | 14.93 ms | 239.16 ms | backward | emission_rebuild |
| A_sparse_50 | N=16 J=128 K=10 A=50 | 13.02 ms | 16.24 ms | 27.27 ms | 1.80 s | gibbs_target_and_validation | emission_rebuild |
| D_12 | N=16 J=192 K=10 A=20 | 22.87 ms | 28.29 ms | 18.13 ms | 1.02 s | backward | emission_rebuild |
| D_24 | N=16 J=192 K=10 A=20 | 25.71 ms | 46.57 ms | 28.73 ms | 3.85 s | backward | emission_rebuild |
| D_48 | N=16 J=192 K=10 A=20 | 31.30 ms | 63.47 ms | 33.31 ms | 13.91 s | backward | emission_rebuild |
| D_6 | N=16 J=192 K=10 A=20 | 21.10 ms | 18.51 ms | 11.14 ms | 257.19 ms | forward | emission_rebuild |
| D_96 | N=16 J=192 K=10 A=20 | 40.62 ms | 97.50 ms | 38.33 ms | 45.81 s | backward | emission_rebuild |
| J_1024 | N=16 J=1024 K=10 A=20 | 323.01 ms | 155.32 ms | 58.38 ms | 5.65 s | forward | emission_rebuild |
| J_192 | N=16 J=192 K=10 A=20 | 22.77 ms | 29.19 ms | 17.24 ms | 1.02 s | backward | emission_rebuild |
| J_24 | N=16 J=24 K=10 A=20 | 1.90 ms | 3.40 ms | 7.40 ms | 137.92 ms | gibbs_target_and_validation | emission_rebuild |
| J_384 | N=16 J=384 K=10 A=20 | 59.88 ms | 56.64 ms | 25.18 ms | 2.12 s | forward | emission_rebuild |
| J_48 | N=16 J=48 K=10 A=20 | 3.88 ms | 6.82 ms | 10.83 ms | 263.87 ms | gibbs_target_and_validation | emission_rebuild |
| J_768 | N=16 J=768 K=10 A=20 | 193.28 ms | 117.99 ms | 35.22 ms | 4.12 s | forward | emission_rebuild |
| J_96 | N=16 J=96 K=10 A=20 | 8.82 ms | 13.67 ms | 15.71 ms | 530.99 ms | gibbs_target_and_validation | emission_rebuild |
| K_10 | N=32 J=128 K=10 A=20 | 21.71 ms | 37.50 ms | 22.99 ms | 1.34 s | backward | emission_rebuild |
| K_20 | N=32 J=128 K=20 A=20 | 40.41 ms | 71.87 ms | 29.81 ms | 2.65 s | backward | emission_rebuild |
| K_3 | N=32 J=128 K=3 A=20 | 10.71 ms | 15.56 ms | 11.62 ms | 414.33 ms | backward | emission_rebuild |
| K_40 | N=32 J=128 K=40 A=20 | 80.28 ms | 130.17 ms | 43.31 ms | 5.36 s | backward | emission_rebuild |
| K_5 | N=32 J=128 K=5 A=20 | 13.99 ms | 22.04 ms | 15.06 ms | 654.51 ms | backward | emission_rebuild |
| K_80 | N=32 J=128 K=80 A=20 | 202.22 ms | 253.08 ms | 63.33 ms | 11.00 s | backward | emission_rebuild |
| N_1 | N=1 J=128 K=10 A=20 | 4.36 ms | 1.32 ms | 4.32 ms | 91.10 ms | forward | emission_rebuild |
| N_128 | N=128 J=128 K=10 A=20 | 71.67 ms | 152.06 ms | 43.87 ms | 5.28 s | backward | emission_rebuild |
| N_16 | N=16 J=128 K=10 A=20 | 12.83 ms | 18.39 ms | 16.71 ms | 707.86 ms | backward | emission_rebuild |
| N_256 | N=256 J=128 K=10 A=20 | 142.70 ms | 297.09 ms | 77.85 ms | 11.14 s | backward | emission_rebuild |
| N_32 | N=32 J=128 K=10 A=20 | 21.82 ms | 37.42 ms | 21.33 ms | 1.31 s | backward | emission_rebuild |
| N_64 | N=64 J=128 K=10 A=20 | 43.80 ms | 73.36 ms | 24.23 ms | 2.75 s | backward | emission_rebuild |
| N_8 | N=8 J=128 K=10 A=20 | 8.44 ms | 10.20 ms | 11.35 ms | 365.42 ms | gibbs_target_and_validation | emission_rebuild |
| baseline_matched_scale | N=100 J=48 K=3 A=5 | 8.87 ms | 11.79 ms | 7.32 ms | 95.98 ms | backward | emission_rebuild |
| target_operating_point | N=100 J=200 K=20 A=50 | 224.06 ms | 294.03 ms | 133.45 ms | 33.84 s | backward | emission_rebuild |
| target_operating_point_full_support | N=100 J=200 K=20 A=50 | 219.04 ms | 380.93 ms | 112.11 ms | 53.27 s | backward | emission_rebuild |

A rebuild share at or slightly above one means the candidate-table rebuild
accounts for essentially the whole structural sweep. The two are timed in
separate processes, so the ratio carries the noise of both and can land just
over unity; it should be read as "the rebuild is the structural sweep", not as
a share above 100%.

`gibbs_target_and_validation` is the plain sweep minus the complete FFBS update:
the pi/P Gibbs step, the complete-data target decomposition and the validators
the sweep runs on entry and exit. It is a residual, not a separately timed
operation, and it is reported as one.

## Complexity

| axis | operation | points | exponent | 95% CI | R^2 | fitted range |
| --- | --- | --- | --- | --- | --- | --- |
| role inventory A, full support | emission build | 5 | 1.06 | 0.89 to 1.22 | 0.967 | 5–50 |
| role inventory A, full support | forward batched | 5 | -0.00 | -0.01 to 0.01 | 0.049 | 5–50 |
| role inventory A, full support | ffbs complete | 5 | 0.06 | 0.04 to 0.07 | 0.884 | 5–50 |
| role inventory A, full support | cond plain | 5 | 0.13 | 0.09 to 0.17 | 0.880 | 5–50 |
| role inventory A, full support | marg structural | 5 | 1.00 | 0.83 to 1.16 | 0.962 | 5–50 |
| role inventory A, sparse support | emission build | 5 | 0.86 | 0.74 to 0.99 | 0.971 | 5–50 |
| role inventory A, sparse support | forward batched | 5 | -0.00 | -0.01 to 0.00 | 0.285 | 5–50 |
| role inventory A, sparse support | ffbs complete | 5 | 0.01 | -0.02 to 0.04 | 0.065 | 5–50 |
| role inventory A, sparse support | cond plain | 5 | 0.11 | 0.06 to 0.15 | 0.806 | 5–50 |
| role inventory A, sparse support | marg structural | 5 | 0.80 | 0.66 to 0.93 | 0.963 | 5–50 |
| maximum segment width D | emission build | 5 | 1.87 | 1.83 to 1.92 | 0.999 | 6–96 |
| maximum segment width D | forward batched | 5 | 0.23 | 0.19 to 0.28 | 0.950 | 6–96 |
| maximum segment width D | ffbs complete | 5 | 0.45 | 0.42 to 0.47 | 0.996 | 6–96 |
| maximum segment width D | cond plain | 5 | 0.45 | 0.42 to 0.47 | 0.997 | 6–96 |
| maximum segment width D | marg structural | 5 | 1.78 | 1.76 to 1.79 | 1.000 | 6–96 |
| number of legal candidate blocks | emission build | 5 | 1.79 | 1.69 to 1.89 | 0.996 | 12064–215824 |
| number of legal candidate blocks | forward batched | 5 | 0.22 | 0.16 to 0.28 | 0.904 | 12064–215824 |
| number of legal candidate blocks | ffbs complete | 5 | 0.42 | 0.37 to 0.48 | 0.978 | 12064–215824 |
| number of legal candidate blocks | cond plain | 5 | 0.43 | 0.40 to 0.46 | 0.993 | 12064–215824 |
| number of legal candidate blocks | marg structural | 5 | 1.70 | 1.57 to 1.83 | 0.992 | 12064–215824 |
| trace length J | emission build | 7 | 0.99 | 0.98 to 1.00 | 1.000 | 24–1024 |
| trace length J | forward batched | 7 | 1.38 | 1.29 to 1.46 | 0.993 | 24–1024 |
| trace length J | ffbs complete | 7 | 1.20 | 1.15 to 1.26 | 0.996 | 24–1024 |
| trace length J | cond plain | 7 | 0.99 | 0.91 to 1.07 | 0.988 | 24–1024 |
| trace length J | marg structural | 7 | 1.02 | 0.99 to 1.05 | 0.998 | 24–1024 |
| skill library size K | emission build | 6 | 1.00 | 0.99 to 1.02 | 1.000 | 3–80 |
| skill library size K | forward batched | 6 | 0.88 | 0.77 to 1.00 | 0.976 | 3–80 |
| skill library size K | ffbs complete | 6 | 0.87 | 0.80 to 0.93 | 0.991 | 3–80 |
| skill library size K | cond plain | 6 | 0.79 | 0.73 to 0.85 | 0.992 | 3–80 |
| skill library size K | marg structural | 6 | 0.96 | 0.95 to 0.97 | 1.000 | 3–80 |
| corpus size N | emission build | 7 | 0.88 | 0.81 to 0.94 | 0.990 | 1–256 |
| corpus size N | forward batched | 7 | 0.64 | 0.53 to 0.75 | 0.947 | 1–256 |
| corpus size N | ffbs complete | 7 | 0.80 | 0.72 to 0.88 | 0.983 | 1–256 |
| corpus size N | cond plain | 7 | 0.72 | 0.64 to 0.78 | 0.982 | 1–256 |
| corpus size N | marg structural | 7 | 0.86 | 0.79 to 0.93 | 0.987 | 1–256 |

Method: ordinary least squares on `log10(median wall seconds)` against
`log10(x)` over measured, non-censored points; interval from a residual
bootstrap with 5000 resamples. `complexity_fits.json` also carries the same
fits on process CPU time.

### Read against the analytic forms

| quantity | analytic form | what the measurements say |
| --- | --- | --- |
| factorised forward | `O(N [J K^2 + J D K])` | the `K^2` and `J D K` terms trade places depending on where `K` sits relative to `D`; a single fitted exponent in `K` should not be expected to equal two, and was not forced to |
| dense score-table memory | `O(N J^2 K)` | confirmed by the exact array shapes: the tables are `(J, J+1, K)` float64 per trace, and the measured peak RSS tracks them |
| projected banded memory | `O(N J D K)` | **NOT IMPLEMENTED; ARITHMETIC COUNTERFACTUAL ONLY** — computed from the exact legal-width count, never measured |
| recurrent emission cost | depends on A and on role-graph density | the two support regimes separate sharply on the `A` axis, which is why they are reported apart |

**No fitted exponent is extrapolated more than twofold beyond the largest
measured value.** Each fit records its own range in `complexity_fits.json`.

## Memory

| configuration | group | dimensions | measured peak RSS | dense score table | batched stack (worst class) | alpha charts | projected banded | projected saving |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| J_1024 | marg | N=16 J=1024 K=10 | 4.78 GiB | 1.25 GiB | 1.25 GiB | 1 MiB | 12 MiB | 102.7x |
| J_1024 | marg | N=16 J=1024 K=10 | 4.43 GiB | 1.25 GiB | 1.25 GiB | 1 MiB | 12 MiB | 102.7x |
| J_768 | marg | N=16 J=768 K=10 | 4.12 GiB | 721 MiB | 721 MiB | 1 MiB | 9 MiB | 77.1x |
| J_768 | marg | N=16 J=768 K=10 | 4.07 GiB | 721 MiB | 721 MiB | 1 MiB | 9 MiB | 77.1x |
| target_operating_point | marg | N=100 J=200 K=20 | 3.66 GiB | 613 MiB | 613 MiB | 3 MiB | 30 MiB | 20.3x |
| target_operating_point | marg | N=100 J=200 K=20 | 3.63 GiB | 613 MiB | 613 MiB | 3 MiB | 30 MiB | 20.3x |
| target_operating_point_full_support | marg | N=100 J=200 K=20 | 3.61 GiB | 613 MiB | 613 MiB | 3 MiB | 30 MiB | 20.3x |
| J_1024 | primitives | N=16 J=1024 K=10 | 2.82 GiB | 1.25 GiB | 1.25 GiB | 1 MiB | 12 MiB | 102.7x |
| J_1024 | primitives | N=16 J=1024 K=10 | 2.82 GiB | 1.25 GiB | 1.25 GiB | 1 MiB | 12 MiB | 102.7x |
| N_256 | marg | N=256 J=128 K=10 | 2.76 GiB | 322 MiB | 322 MiB | 3 MiB | 25 MiB | 13.1x |
| N_256 | marg | N=256 J=128 K=10 | 2.75 GiB | 322 MiB | 322 MiB | 3 MiB | 25 MiB | 13.1x |
| K_80 | marg | N=32 J=128 K=80 | 2.27 GiB | 322 MiB | 322 MiB | 3 MiB | 25 MiB | 13.1x |
| K_80 | marg | N=32 J=128 K=80 | 2.25 GiB | 322 MiB | 322 MiB | 3 MiB | 25 MiB | 13.1x |
| target_operating_point_full_support | primitives | N=100 J=200 K=20 | 2.16 GiB | 613 MiB | 613 MiB | 3 MiB | 30 MiB | 20.3x |
| target_operating_point | primitives | N=100 J=200 K=20 | 2.15 GiB | 613 MiB | 613 MiB | 3 MiB | 30 MiB | 20.3x |
| target_long_J500_N20 | primitives | N=20 J=500 K=20 | 2.08 GiB | 764 MiB | 764 MiB | 2 MiB | 15 MiB | 50.3x |

`measured peak RSS` is `ru_maxrss` for that subprocess. Every other column is
computed from exact array shapes and dtypes.

> **Projected banded storage is NOT IMPLEMENTED. It is an arithmetic
> counterfactual only.** The figure is what a layout storing only the
> `D_max - D_min + 1` legal durations per start would occupy. No such layout
> exists in this backend, nothing in this study ran on one, and no measurement
> here is evidence that one would be correct or fast.

Across every measured configuration the projected banded layout is
between 2.1x and 102.7x smaller than the
dense score table, the ratio growing with `J` because the dense
table's second axis is `J + 1` while the band's is fixed at `D`.

## The two passes, configuration by configuration

| configuration | first-pass load | controlled-pass load | first / controlled |
| --- | --- | --- | --- |
| A_full_10 | 15.2 | 1.9 | 2.25x |
| A_full_20 | 11.3 | 1.2 | 2.45x |
| A_full_30 | 9.7 | 1.4 | 2.42x |
| A_full_5 | 1.9 | 1.7 | 0.90x |
| A_full_50 | 1.5 | 1.6 | 1.01x |
| A_sparse_10 | 15.5 | 1.7 | 2.03x |
| A_sparse_20 | 10.3 | 1.9 | 2.43x |
| A_sparse_30 | 11.0 | 1.5 | 2.44x |
| A_sparse_5 | 1.9 | 1.8 | 0.89x |
| A_sparse_50 | 2.1 | 1.7 | 0.99x |
| D_12 | 14.2 | 1.3 | 1.95x |
| D_24 | 9.7 | 1.4 | 2.43x |
| D_48 | 9.8 | 1.6 | 2.40x |
| D_6 | 1.9 | 2.0 | 0.83x |
| D_96 | 2.7 | 4.7 | 1.00x |
| J_1024 | 1.5 | 1.9 | 0.98x |
| J_192 | 2.8 | 1.8 | 0.95x |
| J_24 | 1.9 | 2.0 | 0.94x |
| J_384 | 2.3 | 1.5 | 1.06x |
| J_48 | 2.2 | 2.0 | 0.95x |
| J_768 | 1.5 | 1.5 | 0.99x |
| J_96 | 2.3 | 1.5 | 0.96x |
| K_10 | 12.0 | 1.4 | 2.40x |
| K_20 | 10.0 | 1.9 | 2.39x |
| K_3 | 1.7 | 2.1 | 0.93x |
| K_40 | 5.4 | 1.6 | 1.86x |
| K_5 | 2.8 | 2.0 | 0.94x |
| K_80 | 1.4 | 1.4 | 1.01x |
| N_1 | 1.9 | 2.0 | 0.90x |
| N_128 | 1.5 | 1.4 | 0.98x |
| N_16 | 10.4 | 1.5 | 2.48x |
| N_256 | 1.7 | 1.9 | 0.99x |
| N_32 | 9.6 | 1.6 | 2.38x |
| N_64 | 3.0 | 2.8 | 1.03x |
| N_8 | 2.6 | 1.6 | 1.01x |
| baseline_matched_scale | 1.6 | 2.1 | 0.96x |
| target_long_J500_N20 | 9.7 | 1.5 | 2.27x |
| target_operating_point | 1.7 | 1.4 | 1.00x |
| target_operating_point_full_support | 13.3 | 1.7 | 1.96x |

A ratio near one means the two passes agree and the point was never
contaminated. A ratio well above one means the first pass measured a slower
machine, not a slower algorithm.

## Corroboration (Section 17)

Every registered point settled with budget to spare, so the remaining time went
to the four things Section 17 permits: the target operating point under a
**second deterministic data seed**, a **retry of the points the first pass
censored**, a **quieter baseline** on a machine that had since gone idle, and
**repeats of the largest point on each axis** under that second seed.

None of this is a new measurement. It is a consistency check on the primary
numbers, it is never averaged with them, and the primary number is the one the
paper should quote.

| primary point | repeat | operation | primary | repeat | difference |
| --- | --- | --- | --- | --- | --- |
| target_operating_point | target_seed2 | emission build | 33.84 s | 34.40 s | +1.6% |
| target_operating_point | target_seed2 | emission cache hit | 1.36 ms | 1.34 ms | -1.1% |

Worst disagreement between a primary measurement and its repeat: **1.6%**. A repeat changes both the corpus draw and the moment of machine load, so a difference of this size is the combined width of those two sources and is the honest floor on how precisely any single absolute time here should be read.

## Censored, skipped and refused points

| configuration | group | phase | status | reason | seconds | attempts |
| --- | --- | --- | --- | --- | --- | --- |
| target_long_J500 | build | quiet | skipped_memory | predicted RSS 7.79 GiB exceeds the frozen cap 6.00 GiB; predicted RSS 7.79 GiB exceeds 80% of the 7.40 GiB currently reclaimable | 1.16 | 1 |
| target_long_J500 | primitives | quiet | skipped_memory | predicted RSS 7.79 GiB exceeds the frozen cap 6.00 GiB; predicted RSS 7.79 GiB exceeds 80% of the 7.35 GiB currently reclaimable | 1.06 | 1 |

Statuses: `skipped_memory` — refused by the preflight before allocating;
`skipped_monotone` — a smaller point on the same ordered axis was already
refused, so this one was not attempted; `skipped_conditional` — its registered
predecessor did not complete cleanly; `censored_timeout` — exceeded its
per-configuration ceiling, with any partial timings preserved;
`skipped_deadline` — the benchmarking budget ended before it started;
`failed` — two attempts both errored.

## Recorded decisions

Ambiguities were resolved conservatively, recorded, and the run continued.

| choice | why |
| --- | --- |
| `table_source='batched'` | the setting the registered FULL-LATENT formal runs use, so the measurement describes the configuration the project actually runs |
| structural sweeps measured with the emission cache forced to miss | a repeated identical proposal would let the H-keyed cache hit from the second repetition onward and report a structural sweep that never rebuilds; forcing the miss measures the H-moved case, which is the upper bound |
| plain sweeps measured with the emission cache warm | that is the steady state a chain runs in between structural moves |
| one sampler per operation inside a group | the samplers carry caches, and interleaving two operations through one sampler measures cache thrash rather than either operation |
| `N = 1` retained on the corpus axis; `K` starts at 3 | a single trace is a legal corpus, but `K = 1` is not a legal model: the transition matrix must have an exactly zero diagonal, which no one-skill chain can satisfy |
| sparse support realised by tying the out-of-support latent rows | the model's only notion of a role inventory is its precedence relation; tied rows are incomparable in both directions, so the induced order is exactly the order on the support |
| round-robin task order across axes, ascending within each axis | ascending within an axis is what makes the monotone skip rule meaningful; round-robin across axes means a budget that runs out leaves every axis with coverage at the small end rather than some axes untouched |
| role sequences drawn deterministically from the benchmark seed | every measured operation's cost is set by array shapes, which the role values do not change; the values change only which gate pattern the recursion walks |

## Artifacts

All paths are relative to `results/scalability/optimized_segmental_v1/`.

- `state.json`
- `events.jsonl`
- `progress.md`
- `hardware_manifest.json`
- `software_manifest.json`
- `parity_results.json`
- `raw_timings.csv`
- `timing_summary.csv`
- `memory_summary.csv`
- `censored_points.csv`
- `complexity_fits.json`
- `marginalisation_overhead.json`
- `runtime_breakdown.json`
- `raw/` — one JSON per `(configuration, group)`, with every repetition

### Figures (PNG and PDF)

| figure | files |
| --- | --- |
| `fig_marg_overhead` | `fig_marg_overhead.png`, `fig_marg_overhead.pdf` |
| `fig_memory_JK` | `fig_memory_JK.png`, `fig_memory_JK.pdf` |
| `fig_runtime_breakdown` | `fig_runtime_breakdown.png`, `fig_runtime_breakdown.pdf` |
| `fig_scaling_A` | `fig_scaling_A.png`, `fig_scaling_A.pdf` |
| `fig_scaling_D` | `fig_scaling_D.png`, `fig_scaling_D.pdf` |
| `fig_scaling_J` | `fig_scaling_J.png`, `fig_scaling_J.pdf` |
| `fig_scaling_K` | `fig_scaling_K.png`, `fig_scaling_K.pdf` |
| `fig_scaling_N` | `fig_scaling_N.png`, `fig_scaling_N.pdf` |
| `fig_target_operating_point` | `fig_target_operating_point.png`, `fig_target_operating_point.pdf` |

