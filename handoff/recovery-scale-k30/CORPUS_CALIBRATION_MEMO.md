# Corpus calibration for order recovery — measured response surface and decision

Goal set by Holly: regenerate the ladder corpora so the partial orders are recoverable —
IP-Cov ≥ 0.9 — with computation cost reported. All measurements below are on
**calibration replicates (100–102)**, never the production replicates; metrics use the
vendored BPOP machinery (`critical_pair_coverage`, `nle`, `transitive_reduction`).

## 1. The three evidence metrics (per skill)

- **edge witnessing** — a true edge a≺b is informative only in instances containing both
  roles; feasibility permits one order, so co-occurrence count is the evidence.
- **IP-Cov** — a true incomparable pair is identified only when witnessed in BOTH orders
  (one direction is indistinguishable from an edge). Prior BPOP work targeted 0.95.
- **all-pairs resolved fraction** — combined: (edge with ≥5 co-occurrences) or
  (incomparable pair witnessed both ways), over all 45 pairs. This is the proposed
  registered criterion: it catches both failure modes and weights them by prevalence.

## 2. What the sweeps established

1. **Brute force saturates.** Width/length/count alone: 4.8× compute bought IP-Cov
   0.59 → 0.75 (K=30 median). Extrapolation to 0.95 is ~hundreds of ×. Cause: the RFS
   utilities (β = 1.5) make order selection nearly deterministic, so incomparable pairs
   are realised in one order only, regardless of sample size.
2. **β is the missing lever.** Lowering β toward uniform-over-feasible (the classic
   linear-extension regime of the original BPOP generators) diversifies realised orders
   while the U-identification channel (feasibility gating, governed by ε = 0.02) stays
   sharp. β = 0.25 beats both 1.5 and 0.0 — some utility pressure is needed to penetrate
   deep posets, so 0.25 is a sweet spot, not a limit.
3. **Leading cell: min_width 6, max_width 16, J = 128, 2× traces, β = 0.25.**
   K=3: IP-Cov 1.00/1.00. K=30: IP-Cov median 0.98, min 0.75.
4. **The K=30 tail is structural.** Only skills 2, 20, 10 miss IP-Cov 0.85; they are the
   densest posets (33–36 of 45 pairs comparable — near-chains), where the few
   incomparable pairs are hard to witness both ways. corr(IP-Cov, edges) = −0.50;
   corr with instance count = −0.03 → more data cannot fix them.
5. **By the all-pairs criterion the tail is wider** (14/30 skills < 0.90 at 2×), driven
   by edge witnessing — which DOES scale linearly with instances (edge≥5 median went
   0.70 → 0.90 from 1× to 2×). The 3× measurement decides how much of the tail closes
   with data alone. **Result: 3× closes K=3 completely (min 0.93, none below 0.90) and
   moves K=30 from 14 to 6 failing skills (median 0.97) — but the six that remain
   (0, 2, 6, 16, 18, 20) barely improved from 2× to 3×. They are the structural core:
   data cannot resolve them at any affordable multiple.**

## 3. Costs (frozen-exponent pricing; Xeon ≈ 2× Mac)

| corpus | compute × | K=30 learned chain, X=166.7 | at X=100 |
| --- | --- | --- | --- |
| current (W3–12, J96, 1×) | 1.0 | ~111 h | ~68 h |
| leading cell (W6–16, J128, 2×) | ~2.8 | **~310 h (13 days)** | ~190 h (8 days) |
| leading cell, 3× | ~4.2 | ~470 h | ~285 h |

**The corpus and X decisions are one decision.** The evidence-rich corpus at X=166.7 is
wall-clock infeasible (a chain is indivisible). Feasible combinations need either
X ≤ 100, shorter registered schedules, or acceptance that K=30 chains run for ~1–2 weeks.
Note the counterweight: a corpus at IP-Cov ~0.95+ has a far sharper posterior, which is
precisely the regime where fewer sweeps and a modest X should suffice — but that must be
demonstrated by the gates, not assumed.

## 4. Options for the residual tail (after the 3× result)

A. **Registered library admissibility**: extend `_admissible` with an
   evidence-feasibility predicate (e.g., incomparable pairs ≥ 14/45, capping near-chain
   posets). Honest scoping — those posets are unidentifiable from traces of this shape —
   and it lifts the floor at every rung. Requires redrawing the master library
   (rejections recorded, as the machinery already does).
B. **Stratified endpoint for the tail**: primary recovery claim on skills meeting the
   registered evidence bar; per-skill evidence profile published alongside.
C. **Pay for 3×** where it closes the gap, combined with A or B for what remains.

## 5. Required registered changes once the operating point is chosen

1. Generation scalars: β = 0.25 (was TRUE_VALUES 1.5) — a chosen truth regime, stated in
   the paper; ε, ω, λ unchanged.
2. Inference must score at the SAME β: `run_ladder_chain` currently hardcodes
   FIXED_BETA — needs a registered params override used by both sides.
3. Candidate width range in the model = generation range (6–16, was 3–12).
4. Every previously generated ladder corpus and derived number becomes void; manifest
   and provenance regenerate under the new registered parameters.
5. Per-skill evidence profile (all three metrics + nLE) becomes part of every corpus's
   coverage report — measured and reported, never enforced by rejection of realised
   corpora (the selection-bias rule stands; admissibility applies to the LIBRARY draw,
   which is already rejection-sampled with recorded reasons).
