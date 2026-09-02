# Learned-order pilot v3 — design for review (DRAFT: not yet tagged, no data exists)

Status: **draft for Holly and coauthor review.** Nothing here is tagged, no v3 stream has
been drawn, and the rules below are proposed-frozen: they become frozen the moment the v3
tag is sealed, which happens only after this document is approved.

## 1. Why v2 terminated, in one paragraph

v2 preserved production's per-*event* proposal rate, which cut each chain's total `U`
effort to ~1% of production — 0.6–2.0 mean attempts per role — and then applied absolute
convergence gates (R-hat ≤ 1.05, ESS ≥ 100, zero disagreeing frozen edges) that only
production-length chains could meet. At 2 attempts/role, P(a role untouched in ≥1 of 4
chains) ≈ 0.43, so frozen disagreement was guaranteed; with 36 retained events × 4 chains
= 144 total samples, ESS ≥ 100 was infeasible even under perfect mixing. All 90 cells
failed identically: **the gate had zero discriminating power**, which is a property of the
pilot design, not of any candidate. Per the frozen rule, v2 terminated; v3 runs on fresh
streams.

## 2. The category error, named

A short pilot **cannot certify convergence**, and must not pretend to. What it can do is
**rank** proposal scales under a fair, fixed budget. What certifies an `X` is a
production-length chain judged by production-length gates. v3 therefore splits into two
stages with different epistemic claims:

    Stage A   ranks u_scale per rung          (pilot length, sanity gates + ranking)
    Stage B   certifies the global X          (production length, absolute gates)

## 3. Code changes already made (all regression-tested)

1. **Event axis.** `ladder_runner` now records one `U` state per retained U-update event
   (`kept["u_event_sweep"]` alongside), independent of `thin`. v2 recorded 90 thin-grid
   snapshots where 36 events existed, against its own spec.
2. **Edge pool.** Only indicators that move within *every* chain enter the R-hat/ESS
   pool. Partially-frozen indicators (the source of v2's 2.3e16 R-hats and 4.3 ESS
   floor) are counted as their own category — reported, never pooled, not a failure by
   themselves. Chain-disagreeing *frozen* edges remain the automatic failure.
3. **Provenance.** The runtime commit is captured once per process, retried with backoff,
   with git's stderr recorded on every failure; a worker that cannot establish its commit
   **refuses to start** instead of emitting null-provenance outputs that block the
   collector hours later (v2: 4/360, cause undiagnosable).

## 4. Stage A — per-rung `u_scale` ranking

**Grid.** K ∈ {3, 5, 10, 20, 30} × σ ∈ {0.25, 0.5, 1.0} × 2 replicates × 4 chains =
**120 chains**, at **fixed X_pilot = 20 attempts per role** (absolute, NOT event-scaled:
M_K = round(20·K·m), the same per-role effort at every rung; this replaces v2's ~1%
scaling, and at 20 attempts/role P(role untouched in a chain) ≈ e⁻²⁰ ≈ 0).
Schedule: 600 sweeps, warm-up 240, u_every 10 → 36 retained U events. Fresh CRN root
**6,800,000**. σ is frozen after warm-up.

**Why X is fixed in Stage A.** σ is a proposal-efficiency parameter; ranking it requires
identical budgets across candidates. X_pilot = 20 is a *pilot* budget chosen for signal,
and carries no claim about production adequacy — that is Stage B's question.

**Sanity gates (hard; both replicates independently):**
- retained acceptance ∈ [0.15, 0.60]
- N_U == quota, exactly
- ≥ 1 closure-changing accepted move per chain in the retained window
  (a σ at which the kernel never changes the order is dead at that rung)

**No absolute R-hat/ESS gate at pilot length** — that is the v2 error, not repeated.

**Ranking (among σ passing sanity in both replicates; frozen, truth-free,
hardware-independent):**
1. maximise min-over-replicates of relation-count bulk ESS on the event axis;
2. ties within 10%: maximise min-over-replicates of the minimum eligible-edge ESS
   (a cell with no eligible edge ranks below any cell with one);
3. remaining ties: the smaller σ.

**Termination.** A rung with no σ passing sanity in both replicates terminates v3
(kernel revision → preregistered v4 on fresh streams). v2's acceptance data suggest
σ = 0.25 will fail the ceiling and 0.5/1.0 will pass — recorded here as an expectation,
not a criterion.

## 5. Stage B — production-length X certification

**Grid.** K ∈ {3, 30} × X ∈ {50, 100, 166.7} × σ*_K (from Stage A) × 2 replicates ×
4 chains = **48 chains** at the production schedule: 50,000 sweeps, warm-up 20,000,
u_every 10 → 3,000 retained U events per chain. Same fresh root, disjoint design indices.

**Gates (the v2 absolute gates, now feasible at this length; both replicates, both
rungs):** relation-count R-hat ≤ 1.05; eligible-edge R-hat ≤ 1.05; eligible-edge ESS ≥
100; chain-disagreeing frozen edges = 0; retained acceptance ∈ [0.15, 0.60];
closure-changing fraction ≥ 0.02.

**Selection.** Evaluate X in the registered order 50 → 100 → 166.7; the first passing at
BOTH rungs in BOTH replicates is the production X. If none passes, v3 terminates and the
production ladder is reported with the oracle-order and support-only arms only.

**The bracketing assumption, stated.** Per-role effort is flat in K by the quota, and
K = 3 and K = 30 are the ladder's endpoints; middle rungs inherit X (each keeps its own
σ*_K). This is an assumption, not a theorem — certifying all five rungs at production
length multiplies Stage B's cost by ~1.75 and is listed as the conservative alternative.

**Storage.** 3,000 events × up to 2,700 indicators does not fit sensibly in JSON;
Stage B chains write closure bits as packed-bool `.npz` beside the JSON record. Same
immutability rules.

## 6. Cost on the measured fleet (22.5 avg timed workers; Xeon ≈ 1.6× Mac per core)

| stage | chains | est. compute | wall-clock floor | est. wall |
| --- | --- | --- | --- | --- |
| A | 120 | ≈ 300 Xeon-h | K=30 chain ≈ 9 h | **≈ 14 h** |
| B | 48 | ≈ 1,500 Xeon-h | K=30, X=166.7 chain ≈ **89 h** | **≈ 3.7 days** |

Stage B's wall-clock is floored by its longest single chain, not by worker count. Total
v3 ≈ 4.5 days on the 8-server fleet. (All-five-rungs Stage B: ≈ 7–8 days.)

## 7. What v3 does not change

The quota mechanism, the shared-Gamma corpora, the three-arm runner, the CRN design-index
scheme, the collector/aggregator integrity gates, immutability, the no-pruning rule, and
the two-replicates-never-pooled principle are all carried unchanged. Recovery against the
sealed truth remains excluded from every decision. Runtime remains excluded from every
statistical choice.

## 8. Open items for review (decide before tagging)

1. Stage B scope: endpoints K ∈ {3,30} (assumption stated) vs all five rungs (+~1.75×).
2. X_pilot = 20: any reason to prefer another absolute pilot budget?
3. Stage A ranking primary = relation-count ESS: acceptable, or prefer eligible-edge ESS
   as primary with relation-count as fallback?
4. Whether Stage B failure should also block the oracle/support production ladder or
   (as drafted) leave it to run without the learned arm.
