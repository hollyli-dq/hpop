# Recovery at scale — consolidated design v2 (draft for sign-off; nothing frozen, nothing run)

Incorporates the coauthor's eleven-point specification, the corpus calibration
(CORPUS_CALIBRATION_MEMO.md), and Holly's directives: end-to-end recovery demonstrated at
large K, computation cost reported, IP-Cov ≥ 0.9, β ∈ {0, 1} → **resolved to β = 0 by
measurement** (§2).

## 1. Question and claim structure

Does the model still work when K is large? "Works" = jointly, per the spec:
**structure + segmentation + reuse assignment + prediction**, each vs K, with the cost to
obtain a *converged* recovery result reported alongside.

## 2. Generation regime (calibrated, to be registered)

- **β = 0** (feasibility + repetition/backtrack penalties active; utilities off). Measured
  head-to-head at identical corpus tiers: β = 0 reaches IP-Cov 1.00 median / resolved
  1.00 med, 0.80 min at K = 30 where β = 1 leaves 12+ skills under the bar. At β = 0 the
  traces are constrained linear extensions of the latent order — the classic BPOP data
  regime; the recurrent-utility mechanism is exercised by the confirmatory experiments
  (β = 1.5), not re-tested here. Inference scores at the same β = 0 (registered override
  of FIXED_BETA); candidate width range equals the generation range.
- **Corpus: min_width 8, max_width 20, J = 128, 3× traces (15K per rung)** — the cheapest
  measured cell meeting IP-Cov ≥ 0.9 and all-pairs-resolved ≥ 0.9 at median with the
  smallest tail (2 skills at K = 30).
- **Evidence metrics in every corpus report** (measured, never enforced on realised
  draws): per-skill edge witnessing, IP-Cov, all-pairs resolved fraction, nLE.
- Compute multiplier vs the old corpus: ≈ 4× — priced into §9 before launch.

## 3. Library admissibility: the invocation-exposure gate (spec §4)

Verified: the impossible first-step 5ε/m condition is already absent (withdrawn in the
earlier audit). What replaces it, per the spec, is **full invocation exposure**:

    eta_kr = E[ sum_u 1(y_u = r) | H_k, params, delta_B ]

estimated by a registered, fixed number of prior-predictive invocation probes at master-
truth draw time, requiring every role's expected exposure per invocation to clear a
preregistered floor. This is a LIBRARY admissibility criterion — rejection at the master
draw, with every rejection recorded, which is the already-registered mechanism — never a
filter on realised corpora. It is also what removes the calibration tail: the residual
under-evidenced skills at K = 30 are exactly low-exposure geometries. Additional registered
floors: minimum instances per skill and minimum per-role occurrence count in the training
corpus — measured and reported per corpus.

## 4. One engine, all rungs (spec §5)

Same inference engine, priors, path-marginal cadence, convergence rules, initialization at
every K. The registered engine is the sealed FFBS (S, z) sampler + the path-marginal
collapsed-U row kernel at cadence 1/10 (the combination validated in the confirmatory
line). If replica exchange is adopted, it is adopted for ALL K under one preregistered
ladder-construction rule with all replica costs counted — never bolted onto a failing K.

## 5. Effort is an outcome, not an input (supersedes any fixed E_U)

No registered proposal count exists. Holly's directive: a fixed budget constant (200,
166.7, any number) is arbitrary in the way beta = 0.25 was; what is registered instead:

- **pacing**: U proposals arrive at `U_RATE_PER_ROLE_PER_SWEEP` per role per sweep,
  flat in K by construction (the quota machinery distributes each segment's proposals
  evenly over its update events) -- so at any moment every rung has spent the same
  effort per role, preserving the fairness the structural-epoch idea was for;
- **stopping**: a chain runs until its cell's truth-free gates pass (checked every
  segment, window = last half of all draws so far);
- **cap**: `CAP_SWEEPS` = 100,000, a resource statement -- hitting it is INFERENCE FAIL
  at that K, reported as such;
- **reported**: sweeps, U-row proposals, realised proposals per row (mean + min/median/
  max -- the pacing fixes the MEAN; selection is a uniform random scan), accepted
  H-changing proposals per skill, wall time, and the headline: effort-to-verdict per K.

## 6. Convergence verdict before truth (spec §7)

Four dispersed chains per (replicate, K). **All convergence diagnostics permutation-
invariant**: sorted skill-usage profiles, co-clustering of occurrences, total and sorted
relation counts, transition spectra, held-out marginal likelihood, canonical (sorted-
digest) library summaries. This corrects a defect in the earlier gate design: per-skill
edge indicators are not label-invariant, so chains converged to relabelled posteriors
would have shown false disagreement. The truth-free verdict is frozen per (replicate, K)
BEFORE the sealed truth is opened; a failed gate is reported as **inference FAIL at that
K**, never converted into a model-scaling claim. Escalation (one registered sweep
extension) is fixed in advance.

## 7. Recovery endpoints (spec §8, truth opened only after §6)

Deterministic Hungarian alignment, then:
- structure: macro closure F1, macro incomparability F1, exact-skill recovery fraction
- segmentation: boundary AUROC
- reuse: occurrence ARI
- prediction: held-out NLL and its gap to the truth plug-in
Stratified by the per-skill evidence profile, so unidentifiable relations are reported as
such rather than as model failure.

## 8. Support-overlap stress control (spec §10)

Premise verified and stronger than stated: candidate block-skill survival is 13.1% at
K = 3 and **1.6% at K = 30** — support typing alone eliminates 98.4% of candidates, so
typed supports plausibly carry much of the discrimination. Stress condition: at K = 30
only, all skills share ONE 10-CPA support (identity role maps), distinguished solely by
order, recurrent execution, allocation and transitions. Not a second grid — one endpoint.
Compute note: shared support makes every block live for every skill (~60× table work at
K = 30); the stress runs at reduced replicate count and its cost is reported. If it fails,
the main ladder stands and the paper's claim is scoped to the typed-support setting.

## 9. Cost reporting (spec §9)

Per (K, arm): seconds/sweep, seconds/structural epoch, component times (block-table,
forward, FFBS, path-marginal update), total wall, CPU-hours, peak RSS, block-table memory,
relation ESS/hour, accepted H-changing moves/hour, and the headline: **time to the frozen
convergence verdict** — cost of a *converged* result, not cost of a sweep. Pricing before
launch uses the frozen scalability exponents; all previous estimates are void pending the
new corpus (≈ 4×) and the run-to-convergence stopping rule (effort unknown until measured -- that is the point).

## 10. Figure (spec §11)

Four panels vs K: (a) structure (macro closure F1, exact-skill fraction);
(b) path (boundary AUROC, occurrence ARI); (c) prediction (held-out NLL, gap to plug-in);
(d) cost (total wall, peak RSS; right axis hours-to-convergence).

## 11. Sequencing

1. implement invocation-exposure admissibility + redraw master library (recorded)
2. regenerate calibration evidence profile on the new library; confirm tail cleared
3. register: β = 0, corpus cell, pacing rate + cap, gates, endpoints, stress design
4. σ pilot (per-rung, sanity + ranking, fresh streams)
5. production ladder + stress endpoint; gates; unseal truth; figure + cost table

## Open for sign-off

- FULL-MARG for the path draws as well? (current registered engine: conditional FFBS
  paths + path-marginal U; switching is a uniform engine change, amortised cost ≈ 1.044)
- exposure-floor value for eta_kr and the probe count
- stress-endpoint replicate count (cost-driven)
