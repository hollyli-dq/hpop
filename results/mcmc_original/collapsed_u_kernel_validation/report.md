# Collapsed-U kernel validation

Status: **COLLAPSED-U KERNEL NOT YET VALIDATED — STOP** (pre-registered rule: any
reference gate fails → stop). 17 of 18 frozen Stage 6E1B gates PASS; the single failure
is marginal, localized, and characterized below — but the rule is the rule.

## What was built

The occasional partially-collapsed latent-U structural update, as two new modules
composing existing validated code by call (no validated module edited — all diffs empty):

* `src/hpop/mcmc_original/collapsed_u_likelihood.py` — `ell_coll(U) = sum_n log Z_n(U)`
  via the Step 7A forward recursion over `FastBlockScoreTable` candidate tables
  (bit-identical to the C0/C1 audit scorer), with an exactly-fingerprinted cache and
  skill-local candidate deltas (normaliser always recomputed after a column change).
* `src/hpop/mcmc_original/collapsed_u_kernel.py` — `collapsed_u_mh_step` (the registered
  Stage 6C row proposal, scale 0.5, symmetric ⇒ zero Hastings, accepted on
  `Δell_coll + Δlog p(U|rho)`; never reads the stored `(S,z)`),
  `collapsed_ffbs_sweep_once` (scheduled sweeps prepend the move; EVERY sweep then runs
  the unmodified `ffbs_sweep_once`, whose first action is the exact FFBS refresh of all
  `(S,z)` at the post-move U), and `run_collapsed_u_chain` (storage/checkpoint format of
  `run_stage7b_chain`; cadence `collapsed_u_every` configurable, provisional default 10,
  scheduled on the absolute sweep index so resume keeps phase).

Sweep ordering on a scheduled sweep:

    collapsed U MH  →  exact FFBS draw of all (S,z) at the new U  →  (pi,P if inferred)
    → conditional U rows → rho → beta → omega → lambda_rep → lambda_back

On unscheduled sweeps the kernel IS Step 7B, bit for bit (test-pinned). Ordinary U moves
are unchanged and still run every sweep. Why the composition preserves the target: the
collapsed MH is invariant for the (S,z)-marginalised posterior of U; drawing (S,z) from
its exact conditional at the accepted U reconstructs the joint (standard partially
collapsed Gibbs); nothing between the two consumes the stale (S,z).

## Correctness evidence (all PASS)

* **Tiny exact reference** (finite U-grid × enumerated paths, production scorer in every
  ratio): stationarity deviation of the composed kernel **2.2e-16**; the deliberately
  wrong ordering (a conditional update consuming the stale path between move and refresh)
  deviates by **2.9e-3** — the immediate refresh is load-bearing
  (`tiny_exact_reference.json`).
* **Parity**: collapsed likelihood vs the validated audit scorer, forward vs enumeration,
  incremental vs full rebuild, same-H invariance, Hastings term — all **exactly 0.0**
  (`correctness.json`).
* **Resume**: uninterrupted == checkpoint+resume bit-for-bit (draws, final state, RNG,
  counters, collapsed schedule phase); only the non-mathematical `cache_version` counter
  restarts (`resume_check.json`).
* **Tests**: 19 new tests in four files; full project suite under the registered
  invocation `PYTHONPATH=src pytest tests`: **1384 passed, 5 skipped (pre-existing), 0
  failed**. Step 7A and Step 7B test files untouched.

## The mixed-reference run (registered length)

4 chains × 600,000 sweeps, burn-in 120,000, thin 10, seeds 8,153,001–004, cadence 10,
pi/P fixed as the reference fixes them; frozen reference verified (drift 0.0). Collapsed
move on this problem: 60,000 attempts/chain, 34% crossed H, **63.7% accepted**.

| gate | value | threshold | verdict |
|---|---|---|---|
| segmentation TV | 0.00473 | 0.01 | PASS |
| segmentation TV vs sampled estimator | 0.00456 | 0.01 | PASS |
| max boundary marginal error | 0.00126 | 0.01 | PASS |
| max occurrence-label marginal error | 0.00255 | 0.01 | PASS |
| induced-H TV | 0.00521 | 0.01 | PASS |
| max relation marginal error | 0.00276 | 0.01 | PASS |
| segment-count TV | 0.00136 | 0.01 | PASS |
| **mixed multivariate energy statistic** | **0.00514** | **0.00452** | **FAIL** |
| all 11 R-hat gates (worst 1.0035) | ≤1.0035 | 1.01 | PASS |

## Diagnosis of the one failure (`gate_failure_diagnosis.json`)

z = +2.86 on a null calibrated with BOTH samples iid from the reference (99% envelope).
Everything a bias would normally touch is clean: scalar means within 0.014 reference SD
(better than both passing baselines), closure-cell marginals equal to 7B1's, dependence
structure as close to the reference as 7B1's, ESS profile identical to 7B1. The
exceedance concentrates in the scalar block (group z +2.06, 0.7% over its own envelope)
and is persistent across subsample offsets (5/6 over). The envelope is known-conservative
for autocorrelated chains (scalar ESS 1,300–3,400 vs n=4,000), so the nominal 1%
false-alarm rate is understated for ANY correct sampler here — but with ESS identical to
7B1 (z −1.15), realization luck and a sub-resolution residual discrepancy cannot be
separated **without new draws**, which the registered stop condition forbids in this
task. The natural next probe (user's call): an independent-seed replication of this same
registered comparison, before any tuning or any larger experiment.

## Cost (runtime.json)

Reference problem: ordinary sweep 26.8 ms; partially-collapsed sweep at cadence 10:
27.8 ms (**+3.7%**); collapsed event 17.7 ms mean. Full corpus (read-only 7B2
checkpoint): collapsed proposal 1.79 s; at cadence 10 ≈ **+13%** of the observed 1.38
s/sweep. 600k×4 validation wall: 3.4 h compute.

## Verdict

**COLLAPSED-U KERNEL NOT YET VALIDATED — STOP.** No matched-synthetic experiment, no
generator work, no cadence tuning, no production run. The implementation, its exact-
correctness evidence, and the single marginal gate failure are handed back for review.
