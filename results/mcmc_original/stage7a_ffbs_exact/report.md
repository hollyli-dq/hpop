# Step 7A — semi-Markov FFBS against the exact segmentation posterior

Status: **PASS**. Step 7B (full-joint FFBS integration): NOT STARTED.

Step 7A replaces the local Metropolis update of `(S, z)` with an exact blocked
draw from the same conditional. The model is untouched: same trace, same widths,
same `delta_B`, same `pi`, `P`, `U`, scalars and `epsilon` as Stage 6E1A, and the
same registered path weight.

## The target

```text
log pi[z_1] + sum_l log p_block(x[a_l:b_l] | z_l) + (J - L) log(1 - delta_B) + (L - 1) log delta_B + sum_{l>=2} log P[z_{l-1}, z_l]
```

* self-transitions: forbidden (P has a zero diagonal)
* terminal transition: none
* duration model: none beyond delta_B and the width bounds
* recurrent state: every candidate block is scored from q_0 = 0

## Exact comparison

| quantity | value |
|---|---|
| exact paths enumerated | 21 |
| log Z (exact enumeration) | -7.939698706439894 |
| log Z (FFBS forward chart) | -7.939698706439893 |
| absolute log-Z error | 8.882e-16 |
| log Z (Stage 6E1A forward recursion, independent) | -7.939698706439893 |
| DP boundary marginal error vs enumeration | 4.857e-17 |
| DP label marginal error vs enumeration | 5.551e-16 |
| DP labelled-segment marginal error vs enumeration | 4.441e-16 |

## Sampling, exact vs FFBS vs LocalMoveKernel

| statistic | FFBS | LocalMoveKernel |
|---|---|---|
| retained draws | 100,000 | 380,000 |
| full-path TV to exact | 0.002388 | 0.001470 |
| max boundary marginal error | 0.000639 | 0.000284 |
| max occurrence-label error | 0.002388 | 0.001066 |
| max labelled-segment error | 0.002388 | 0.000993 |
| max unlabelled-segment error | 0.001450 | 0.000284 |
| segment-count distribution TV | 0.001450 | 0.000068 |
| max expected transition-count error | 0.000732 | 0.000111 |

FFBS-to-LocalMoveKernel TV is 0.003030, against the sum of their individual errors 0.003859. Both samplers agree with the same exact posterior; neither produces a *better*
posterior, and the difference between them is Monte Carlo error, not target drift.

The raw TVs are not comparable as they stand — the LocalMoveKernel has 3.8x the draws. Scaled by the square root of the draw count they are 0.755 for FFBS and 0.906 for the LocalMoveKernel, i.e. the same order, as two correct samplers of one distribution should be.

## Recurrent block scoring

* `q_0 = 0` reset per candidate block: PASS (score A, score B, score A again — bit-identical in 4 pairs)
* evaluation-order invariance over 4 legal orders: PASS (tables identical, not merely close)
* cached vs uncached table: identical
* table [8, 9, 3]: 63 legal blocks, 153 at -inf

## Efficiency

| quantity | FFBS | LocalMoveKernel |
|---|---|---|
| wall seconds | 3.50 | 31.16 |
| seconds per retained draw | 3.505e-05 | 8.201e-05 |
| segment-count lag-1 autocorrelation | -0.0119 | +0.3986 |
| segment-count bulk ESS | 99,987 | 166,012 |
| segment-count ESS per retained draw | 1.000 | 0.437 |
| segment-count ESS / second | 28,529 | 5,327 |
| worst boundary-indicator ESS | 97,158 | 189,014 |
| worst boundary-indicator ESS / second | 27,722 | 6,065 |
| unique complete paths visited | 9 | 9 |

Both samplers visit the same 9 of the 21 enumerated states, and those are exactly the 9 states carrying more than 1e-6 of the exact mass. The remaining 12 have exact probability at most 2.6e-07, so neither sampler was expected to reach them.

Both wall clocks were measured while four Stage 6E worker processes were saturating four of this machine's ten cores, so both are pessimistic. The LocalMoveKernel re-run took 31.2s against the frozen Stage 6E1A run's 23.1s for the identical chains; scoring its ESS against that faster frozen runtime is also reported, and FFBS still leads on ESS/second.

Scored against the frozen Stage 6E1A runtime rather than this contended re-run, the LocalMoveKernel reaches 7,188 segment-count ESS/second, still below FFBS's 28,529. The durable difference is not the wall clock but the 2.3x in ESS per retained draw: FFBS draws are independent by construction, and the local kernel's are not.

Forward chart: block table 8.5 ms + recursion 1.6 ms (median of 20); one backward draw 35 us. The chart pays for itself after 290 draws at fixed parameters.

| J | block table (ms) | chart (ms) | one draw (us) |
|---|---|---|---|
| 48 | 229.7 | 13.9 | 242 |
| 96 | 499.0 | 30.0 | 481 |

## Gates

| gate | value | threshold | verdict |
|---|---|---|---|
| log_z_vs_independent_enumeration | 8.882e-16 | 1e-10 | PASS |
| dp_marginals_vs_enumeration | 5.551e-16 | 1e-10 | PASS |
| ffbs_draws_completed | 100000 | 100000 | PASS |
| ffbs_full_path_total_variation | 2.388e-03 | 0.01 | PASS |
| ffbs_max_boundary_marginal_error | 6.389e-04 | 0.01 | PASS |
| ffbs_max_occurrence_label_marginal_error | 2.388e-03 | 0.01 | PASS |
| recurrent_candidate_q0_reset | 0.000e+00 | 0.0 | PASS |
| evaluation_order_invariance | 0.000e+00 | 0.0 | PASS |
| cached_equals_uncached_block_table | 0.000e+00 | 0.0 | PASS |
| local_move_kernel_reproduces_frozen_frequencies | 0.000e+00 | 1e-12 | PASS |
| local_move_kernel_and_ffbs_agree_with_the_same_exact_posterior | 2.388e-03 | 0.01 | PASS |

## Complete path table

| state | exact | FFBS | LocalMoveKernel |
|---|---|---|---|
| (8,2) | 0.474758 | 0.472370 | 0.475400 |
| (8,0) | 0.420640 | 0.420940 | 0.419647 |
| (8,1) | 0.027373 | 0.028010 | 0.027655 |
| (4,2) (8,1) | 0.022162 | 0.022280 | 0.021758 |
| (3,0) (8,1) | 0.013737 | 0.013880 | 0.013800 |
| (4,0) (8,1) | 0.013400 | 0.013750 | 0.013521 |
| (5,2) (8,1) | 0.010107 | 0.010520 | 0.010395 |
| (3,2) (8,1) | 0.009729 | 0.009930 | 0.009803 |
| (5,0) (8,1) | 0.008094 | 0.008320 | 0.008021 |
| (5,2) (8,0) | 0.000000 | 0.000000 | 0.000000 |
| (5,0) (8,2) | 0.000000 | 0.000000 | 0.000000 |
| (5,1) (8,0) | 0.000000 | 0.000000 | 0.000000 |
| (5,1) (8,2) | 0.000000 | 0.000000 | 0.000000 |
| (4,2) (8,0) | 0.000000 | 0.000000 | 0.000000 |
| (4,0) (8,2) | 0.000000 | 0.000000 | 0.000000 |
| (4,1) (8,0) | 0.000000 | 0.000000 | 0.000000 |
| (4,1) (8,2) | 0.000000 | 0.000000 | 0.000000 |
| (3,0) (8,2) | 0.000000 | 0.000000 | 0.000000 |
| (3,2) (8,0) | 0.000000 | 0.000000 | 0.000000 |
| (3,1) (8,2) | 0.000000 | 0.000000 | 0.000000 |
| (3,1) (8,0) | 0.000000 | 0.000000 | 0.000000 |

## What this does and does not establish

**Established.** A model-agnostic semi-Markov FFBS engine reproduces the exact
fixed-parameter segmentation posterior: its normaliser matches an independent
enumeration to machine precision, its dynamic-programming marginals match the
enumerated marginals to machine precision, and 100,000 iid backward draws match
the exact path distribution well inside every registered gate. The engine consumes
only block scores, `pi`, `P`, `delta_B` and the width bounds — it imports nothing
from Stage 6 and never sees `U`, `rho` or the recurrent recursion.

**Not established, and not claimed.**

* Nothing about the *joint* sampler. `(S, z)` is drawn at fixed parameters here;
  composing FFBS with the `U`, `rho`, `pi`, `P` and scalar kernels is Step 7B and
  has not been started.
* No claim that FFBS gives a better posterior. It gives the same posterior with
  independent draws; the LocalMoveKernel result stands unchanged.
* The efficiency numbers are for one `J = 8` problem with 21 legal states, plus a
  chart-construction benchmark at `J = 48` and `J = 96`. They do not extrapolate to
  the Stage 6E2 corpus, and no Stage 6E chain was disturbed to measure them.

Source commit `155dd5cc48828815d7e3b953e6a8d98e9644e89a`.
