# Paper-ready claims — what may and may not be written

Sorted by what supports them. A claim in section D must not appear in the paper in any
wording.

---

## A. Directly measured claims

These are readings from `timing_summary.csv` and `memory_summary.csv` on this machine, at
this commit, under the stated protocol. Quote them with the configuration attached.

- The optimized backend is numerically equivalent to the frozen reference engine:
  across the parity grid the worst absolute error in `alpha` and in `log Z` was
  1.1e-13, against a
  pre-registered tolerance of 1e-10, with identical `-inf` patterns, bit-identical
  emission tables and identical legal-block counts.
- "With bounded segment width, the optimized forward computation exhibits approximately 1.38 scaling in trace length over the tested range 24 to 1024 (95% CI 1.29 to 1.46)."
- "With bounded segment width, the optimized forward computation exhibits approximately 0.99 scaling in trace length, for a complete plain sweep over the tested range 24 to 1024 (95% CI 0.91 to 1.07)."
- "With bounded segment width, the optimized forward computation exhibits approximately 0.79 scaling in the number of skills, for a complete plain sweep over the tested range 3 to 80 (95% CI 0.73 to 0.85)."
- "With bounded segment width, the optimized forward computation exhibits approximately 0.72 scaling in corpus size, for a complete plain sweep over the tested range 1 to 256 (95% CI 0.64 to 0.78)."
- At the anticipated real-data operating point (N=100, J=200, K=20, A=50, D in [3,12],
  sparse support), a plain FULL-COND sweep took **648.30 ms** and the benchmark
  process peaked at **3.66 GiB** resident.
- Path marginalisation costs nothing on a plain sweep — the arms execute the same code —
  and its entire cost falls on the structural sweep. See the ratio table in the report.
- The role-support regime, not `A` alone, sets the emission cost: at equal `A` the two
  regimes differ by a factor recorded in the report's Q5 tables.

**Always attach:** the machine, the single-thread pinning, the commit, and the fact that
these are throughput measurements.

**Measured on the controlled pass.** An earlier pass of the same plan ran while the
machine was busy and is contaminated by performance/efficiency-core migration; it is
reported in full but must not be quoted. If a number in a draft cannot be traced to a
`phase = quiet` row of `timing_summary.csv`, it is the wrong number.

---

## B. Complexity-derived claims

Supported by the algorithm's form and *consistent with* the measurements, but not
themselves measurements.

- The factorised forward recursion is `O(N [J K^2 + J D K])` per all-trace pass, against
  the reference recursion's `O(N J D K^2)`.
- With `D` bounded, the number of chart cells is `J K` and the number of duration terms
  reduced over is at most `D` per cell, so the forward pass is linear in `J` at fixed
  `K`, `D`.
- The current dense candidate score table is `O(N J^2 K)` in memory. This follows from
  the array shape `(J, J+1, K)` per trace and is confirmed by the recorded shapes.

**Phrase these as complexity statements, never as measurements.**

---

## C. Counterfactual banded-memory projections

- A layout storing only the `D_max - D_min + 1` legal durations per start would require
  `O(N J D K)` for the candidate table instead of `O(N J^2 K)`.

Required wording, or something that says the same thing:

> "The current dense score-table implementation remains quadratic in J in memory;
> storing only legal duration bands would reduce the corresponding table requirement to
> O(NJDK), but this layout is not implemented here."

**Every mention must carry `not implemented`.** These figures are arithmetic on array
shapes. Nothing in this study ran on a banded layout.

---

## D. Claims that must NOT appear, in any wording

- ❌ "The method scales to arbitrary K." — `K` was measured to
  80 and no further.
- ❌ "Memory is linear in J." — it is **quadratic** in `J` in this implementation.
- ❌ "The J=500 posterior converges." — no posterior was assessed at any `J`.
- ❌ "Path marginalization is free." — it is free *on a plain sweep only*; the structural
  sweep pays for it, and the amortized ratio is in the report.
- ❌ "Banded storage is implemented." — it is not.
- ❌ Any statement that the sampler mixes, recovers truth, or has converged.
- ❌ Any extrapolation beyond twice the largest measured value on an axis
  (J was measured to 1024).
- ❌ "The optimized backend is faster than the reference by the product of its four
  optimisations." — the optimisations attack overlapping costs and are not multiplicative.
- ❌ Any per-second figure quoted without its configuration.
- ❌ Any exponent taken from the first pass, or any figure that pools the two passes.
- ❌ "Runtime is independent of A." — the *forward and backward recursions* are; candidate
  table construction is not, and that is where the A cost lives.
- ❌ Quoting a slope for a fit the report marks "no detectable dependence" or "weak fit".

---

## Ready-to-paste sentences

> With bounded segment width, exact inference in the segmental partial-order model scales
> approximately linearly in trace length over the tested range, and approximately linearly
> in corpus size.

> The current dense score-table implementation remains quadratic in J in memory; storing
> only legal duration bands would reduce the corresponding table requirement to O(NJDK),
> but this layout is not implemented here.

> Marginalising the partial order costs nothing on a sweep that performs no structural
> move; the entire overhead falls on the scheduled structural update, and at the
> registered cadence of one in ten it is amortised to the ratio reported in Table N.
