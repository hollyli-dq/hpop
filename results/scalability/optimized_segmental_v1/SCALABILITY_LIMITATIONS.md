# Limitations of the scalability study

This document exists so that nothing in `SCALABILITY_REPORT.md` is read for more than it
says.

## What this study is not

1. **Not a convergence study.** No chain here was run to stationarity, and none was
   diagnosed. Sweep timings are throughput measurements. A sweep count per second says
   nothing about how many sweeps are needed.
2. **Not a recovery study.** No truth was compared to any posterior. The synthetic
   corpora were built to have the right *shapes*, not to be recoverable.
3. **Not a Condition D.** No formal arm was launched, registered, resumed or modified.
4. **Not a model change.** The backend measured is `optimized_segmental_v1` exactly as committed at
   `564995efd056d7d33984f0ca1532386e6140ea0c`. No optimisation was added: no banded storage, no optimized backward
   sampling, no third-forward-pass reuse, no sparse `P`, no pruning, no beam search, no
   GPU kernel, no alternative initializer, no approximate DP, and no new sampler move.

## What the measurements are conditional on

- **Core type, and why this study was measured twice.** The first pass produced
  larger configurations that ran *faster* than smaller ones. Process CPU time over wall
  time stayed at 0.98--1.00 throughout, so nothing was preempted; the CPU time itself had
  fallen, which on a machine with performance and efficiency cores means the process had
  moved between them. A process that changes core type keeps a whole core and runs at
  roughly half the speed, and **neither wall time nor CPU time reveals it**. Because the
  queue ran ascending within each axis, the largest point on every axis was measured last,
  on the quietest machine, biasing every exponent downward. The study was therefore
  re-measured end to end on an idle machine with a fixed-work speed probe, and the
  controlled pass is the primary one. The first pass is kept and reported, never averaged
  in. Anyone repeating this work on Apple silicon should instrument machine speed
  directly; load average and CPU time are both insufficient on their own.
- **One machine, one thread.** Apple M4,
  every BLAS and OpenMP thread count pinned to one, configurations run strictly
  sequentially. Absolute times do not transfer to other hardware; the *exponents* are
  more portable than the constants, and even they are a property of this implementation
  rather than of the algorithm.
- **Ambient load.** Load average is recorded before and after every configuration in
  `timing_summary.csv`, together with the speed probe and a probe-normalised median, so
  the machine's condition at each point is auditable. Process CPU time is recorded
  alongside wall time and the fits are reproduced on it in `complexity_fits.json` -- but
  see the core-type note above for why CPU time alone is not a sufficient control here.
- **Synthetic corpora.** Role sequences are deterministic draws from the benchmark seed.
  Every measured operation's cost is set by array shapes, which do not depend on the role
  values, so this is a sound basis for a *computational* study and not for any inferential
  one.
- **A fixed structural cadence.** Amortized figures assume one structural sweep in ten,
  the registered cadence. A different cadence gives a different amortized cost, and the
  plain and structural figures are reported separately so it can be recomputed.

## Where the numbers are weakest

- **Fits whose interval contains zero.** Several operations show no detectable dependence
  on `A`, which is the correct answer -- the forward and backward recursions never touch
  the role inventory, only candidate-table construction does. The report names these "no
  detectable dependence" rather than quoting a slope, because a fitted exponent with an
  interval straddling zero and an R^2 near zero is noise, not a slow growth rate.
- **Fits with low R^2.** Where a single power law does not describe the points -- the
  forward pass against segment width is one -- the report says so instead of quoting the
  slope as if it were a scaling law.

- **Reduced-repetition points.** Expensive configurations are allowed to stop at five
  timed repetitions instead of fifteen. Every such point is flagged
  `reduced_repetitions = 1` in `timing_summary.csv`, and its interval is correspondingly
  wide.
- **Censored, refused and skipped points: 2 in this run.** They are listed in
  `censored_points.csv` with the reason. A refused point is *absent evidence*, not
  evidence of a limit: the memory preflight refuses on a prediction, and a prediction that
  refuses is not a measurement that failed.
- **Fits from five to seven points.** Every exponent is an OLS slope over a handful of
  points. The residual bootstrap interval is honest about the sampling noise but cannot
  repair a fit whose points do not span enough range.
- **The structural-sweep timing is an upper bound.** It forces a candidate-table rebuild
  every repetition. A structural proposal that does not move `H = h(U)` is served from the
  cache and is cheaper by roughly the measured `emission_build`.

## The memory limitation, stated plainly

The current implementation stores a dense `(J, J+1, K)` float64 candidate score table per
trace. That is **quadratic in trace length**, and it is the binding constraint on `J`, not
runtime. A layout storing only the `D_max - D_min + 1` legal durations per start would
need `O(N J D K)` instead.

> **PROJECTED BANDED STORAGE: NOT IMPLEMENTED; ARITHMETIC COUNTERFACTUAL ONLY.**

Every banded figure in this study is arithmetic on array shapes. It is labelled
`NOT_IMPLEMENTED` in every artifact that carries it, and it must never be reported as
measured memory, as a result, or as a completed optimisation. No banded layout exists in
this backend and nothing in this study ran on one.

## Extrapolation

No claim in this study extends more than a factor of two beyond the largest measured value
on the relevant axis. `complexity_fits.json` records the fitted range for every exponent.
Beyond that factor, the fits are not evidence.
