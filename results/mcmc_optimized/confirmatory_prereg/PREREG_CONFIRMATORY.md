# Preregistration — optimized FULL-LATENT confirmatory experiment

**Status: NOT LAUNCHED. Awaiting PI approval.**
Written before any inference was run on this corpus. No draw from this corpus exists.

Backend commit `564995efd056d7d33984f0ca1532386e6140ea0c` (`hpop.mcmc_optimized`).
Reference oracle `hpop.mcmc_original`, unmodified, `semi_markov_ffbs.py` =
`8150bb8235eb159d5e2f08ada7c698c383f4c7fd31f3b882e218b207dd135486`.

This experiment replaces the FULL-LATENT programme terminated at 30,000 sweeps
(both arms FAIL, archived in `terminated_30k_archive/`). **No state, start, truth or
tuning from that programme is reused.** Its truth was unsealed on 2026-08-22, which is
why a new corpus is mandatory rather than convenient.

---

## 1. Target distribution (frozen)

Unchanged from the terminated programme, and this is the point: only the corpus, the
starts and the schedule differ.

Complete-data target over `(S, z, U, pi, P)` with the recurrent partial-order emission:

    p(S, z, U, pi, P | X)  ∝  p(X | S, z, U)  p(S | delta_b)  p(z | pi, P)  p(U | rho)  p(pi) p(P)

* **Emission.** For a block `[a, b)` with skill `k`, the recurrent RFS likelihood over CPA
  roles, with invocation-local validity `q` reset to `q_0 = 0` at every block boundary.
  Parameters `(beta, omega, lambda_rep, lambda_back, epsilon)`.
* **Segmentation prior** `p(S | J, delta_b)`: boundary probability `delta_b = 0.15`,
  widths restricted to `3 <= w <= 12`.
* **Label prior** `p(z | pi, P)`: initial `pi`, transition `P` with an exactly zero
  diagonal (no self-transitions).
* **Structural prior** `p(U_k | rho)`: rows iid `N(0, Sigma_rho)`, `rho = 0.5`.
  `H_k = h(U_k)` is the induced strict partial order.
* **Conjugate priors** `pi ~ Dir(eta_initial = 1)`, rows of `P ~ Dir(eta_transition = 1)`
  over allowed successors.

**Inferred:** `S, z, U, pi, P`.
**Held fixed** (`FullLatentFixed`, asserted every sweep): `rho = 0.5`, `beta = 1.5`,
`omega = 1.7346010553881064`, `lambda_rep = 0.8`, `lambda_back = 0.25`, `epsilon = 0.02`,
`delta_b = 0.15`. `K = 3` skills, `m = 5` roles, `d = 2` latent dimensions.

## 2. Corpus and truth

| item | value |
|---|---|
| corpus dir | `results/mcmc_optimized/confirmatory_corpus/` |
| corpus hash | `3e3aa6533bd7951f9b2ed1dfa050e9d07f1b2e96b0e3914e044010130e9acdfa` |
| train npz | `e05a46013b398072614d11c40a8891e9afedb780b76db2e07e6200cdf69ad04e` |
| held-out npz | `2698bce19236ca9ebc4590cc0a9852eb7c61aa97487b4dddd08672601d28f0c7` |
| **truth hash** | `effccc91114d9647f87859aaa7ee219e3bc664bb41d2c6389844ea64346a5e8e` |
| master seed | `6_300_001` |
| truth seed | `6_300_002` (accepted on attempt 1 of 100) |
| design | 100 train + 45 held-out, `J` cycling `(24, 32, 40, 48)` |
| validation | Section-4 checks all 0; q0 reset worst `7.11e-15`; generator↔inference parity `1.42e-14` |

**Three declared departures from the terminated generator**, each recorded in
`corpus_config.json`:

1. New master seed. The `6_2xx_xxx` band is exhausted; `6_300_001` / `6_300_002` were
   verified unused across code, results manifests, both worktrees and project memory.
2. **The truth is a fresh prior draw under the admissibility event, not the supplied
   truth.** Candidates are proposed from

       U* ~ p(U | rho = 0.5),   pi* ~ Dir(1),   rows of P* ~ Dir(1)

   at `rho = 0.5` because `rho` is fixed at 0.5 in **both generation and inference**
   (`FIXED_RHO_0 = 0.5`), so the generative prior for `U` is exactly the prior the sampler
   assumes. Scalars are the registered `TRUE_VALUES`.

   **The accepted truth is not an unrestricted prior draw.** Rejection sampling against the
   preregistered admissibility event `A` (criteria 1–12 below) means the accepted draw is
   distributed as the model prior **conditional on `A`**:

       truth ~ p(truth | rho = 0.5, A)

   This is stated precisely because the conditioning is not innocuous: `A` excludes
   antichains, total orders and any pair of coincident skill orders, so the accepted truth
   is drawn from a strict subset of the prior support. Any statement about recovery is a
   statement under `p(truth | rho = 0.5, A)`, not under the unrestricted prior.
3. **`supplied_truth()` is not used.** It is a hardcoded configuration and is the truth of
   the terminated run, unsealed on 2026-08-22. Reusing it would make this experiment
   sealed in name only.

### Truth admissibility criteria (all automated, all preregistered)

A candidate draw is admitted iff **all twelve** hold:

| # | criterion |
|---|---|
| 1 | `K >= 2` |
| 2 | each induced relation is a strict partial order (irreflexive, transitively closed) |
| 3 | `pi` is a length-`K` probability vector |
| 4 | `P` nonnegative, exactly zero diagonal, rows sum to 1 |
| 5 | all scalars inside their registered support |
| 6 | `0 < delta_b < 1` |
| 7 | `1 <= min_width <= max_width` |
| 8 | `rho` strictly inside `(RHO_LOWER, RHO_UPPER)` |
| 9 | every role map injective over `m` roles |
| 10 | the three induced closures are **pairwise distinct** |
| 11 | each closure has **at least one** relation (not empty) |
| 12 | each closure has **fewer than `m(m-1)/2`** relations (not a total order) |

Criteria 1–9 are `validate_truth`; 10–12 are the structural conditions preserving the
controlled configuration class of the terminated programme.

**Correction to criterion 12.** An earlier draft wrote the upper bound as `m(m-1)`, the
number of *ordered* pairs. A strict partial order is antisymmetric, so its closure can hold
at most one of `(i, j)` / `(j, i)` for each unordered pair: the maximum is **`m(m-1)/2`**,
attained exactly by a total order. With `m = 5` the correct bound is **10**, not 20, and the
earlier bound was **vacuous** — no strict partial order can reach 20, so criterion 12
excluded nothing. The corrected condition is

    1 <= relation_count(H_k) < m(m-1)/2      for every true skill k

which excludes both antichains (count 0) and total orders (count `m(m-1)/2`), as intended.
The sealed attempt-1 truth was re-tested against the corrected condition and **passes**;
criteria 10 and 11 also still pass. No regeneration was required and the corpus and truth
hashes are unchanged. The re-test printed no truth value and no per-skill relation count.

**Assertion.** Every criterion is a function of the truth parameters alone. **None
evaluates a likelihood, a recovery metric, a convergence diagnostic, a held-out
prediction, or any comparison between FULL-COND and FULL-MARG.** No corpus is drawn until
a truth is admitted, so no criterion can depend on realised data. On rejection the truth
seed increments by 1; cap 100 attempts.

**Attempt record** (`truth_SEAL.json`):

| attempt | truth seed | accepted | rejection reason |
|---|---|---|---|
| 1 | 6300002 | yes | — |

Rejections: **none**. **The first prior draw satisfied the preregistered admissibility
conditions.**

## 3. Truth seal

Truth is written to `truth_SEALED.json` and **not opened until formal termination**.
`truth_SEAL.json` records only its hash and the acceptance metadata.

The sampler cannot see it even by accident: `load_frozen_observed_corpus` never opens the
truth file and `_load_observed_split` reads only the `t{i}_cpa` key, ignoring the stored
labels, widths and boundaries. All monitoring during the run is **truth-free**: every
diagnostic below is a function of the draws alone.

**Unsealing is permitted exactly once**, at formal termination (Section 8), and is recorded
in `TRUTH_UNSEAL.json` with a timestamp and the hash of the terminal draw set. Any earlier
open, for any reason, converts this run to exploratory and it may not be reported as
confirmatory.

## 4. Arms, starts and seeds

Only two arms, differing **solely** in the U acceptance score:

* **FULL-COND** — `conditional_structural_mh_step`: `U` scored against the currently
  stored explicit path `(S, z)`.
* **FULL-MARG** — `collapsed_u_mh_step`: `U` scored against `sum_n log Z_n(U; pi, P)`,
  the path-marginal likelihood.

Everything else is matched: same corpus, same starts, same fixed coordinates, same
`structural_cadence = 10`, same `structural_scale = 0.5`, same kernel order
(structural attempt → table refresh → all-trace FFBS → pi/P Gibbs → target), same
`table_source = "batched"`, same backend, same schedule.

Starts manifest `43e2b63e4054df60312087e49a72eff140f5e33d048c7b36752c330c2714e84a`:

| i | U seed | scale | pi/P seed | U sha256 | pi sha256 | P sha256 |
|---|---|---|---|---|---|---|
| 0 | 6304101 | 0.5 | 6306101 | `881223525ed4541e…` | `22a49bd6c038e7cb…` | `86edbbe42a603e9c…` |
| 1 | 6304102 | 1.0 | 6306102 | `0ef14686b77e821b…` | `f87d2c0b9808728b…` | `410adaa18063a745…` |
| 2 | 6304103 | 2.0 | 6306103 | `3204f89b770c6446…` | `7afbf898cc946f86…` | `6e1e97cc8c64dd65…` |
| 3 | 6304104 | 3.0 | 6306104 | `2738ec68d267b9e0…` | `86586a438fe04544…` | `71da5a5260836942…` |

Chain seeds: FULL-COND `6306201–6306204`, FULL-MARG `6306211–6306214`.
The four paired starts are **shared across arms**, so a between-arm difference cannot be a
start artifact. Starts are dispersed wrong-structure draws at scales 0.5–3.0 and are
truth-free by construction. **No old checkpoint and no truth-informed initialisation.**

## 5. Schedule: warm-up discarded, production separate

| phase | sweeps | use |
|---|---|---|
| warm-up | **50,000** | **discarded entirely**; no diagnostic is computed from it |
| production | **100,000** | the only source of every formal diagnostic |
| thinning | 5 | 20,000 retained draws per chain, **80,000 per arm** across 4 chains |
| checkpoint every | 2,000 sweeps | durable, atomic (`.tmp` + `os.replace`) |

This is the substantive protocol change. The terminated run used a fixed burn-in of 10,000
inside a single ladder, so the retained window permanently contained a pre-consensus
transient; that is precisely what made FULL-MARG fail on a quantity that had converged to a
point. Separating a discarded warm-up from a clean production phase removes that failure
mode by construction rather than by argument.

**No adaptation, tuning or rescue at the warm-up/production boundary.** The kernel,
scales and cadence are identical in both phases; the boundary only marks which draws count.

### Runtime, measured on an idle machine

Measured on this box with the formal run stopped (load 1.89), median of 12 interleaved
rounds, per chain, single-threaded:

| arm | reference | optimized | speedup |
|---|---|---|---|
| COND plain | 0.7489 s | 0.0275 s | 27.21x |
| COND structural | 0.7487 s | 0.0314 s | 23.88x |
| MARG plain | 0.7439 s | 0.0271 s | 27.48x |
| MARG structural | 1.3860 s | 0.0477 s | 29.06x |

Mean at cadence 1/10: COND **27.9 ms/sweep**, MARG **29.1 ms/sweep**.

| phase | COND | MARG |
|---|---|---|
| warm-up 50,000 | 23.3 min | 24.3 min |
| production 100,000 | 46.5 min | 48.6 min |
| **total 150,000 per chain** | **69.8 min** | **72.8 min** |

Eight chains run concurrently on 10 cores. The measured 8-worker contention factor for the
optimized backend is ~1.19x, so the expected **wall-clock for the whole experiment is
~1.4–2.0 hours**. On the reference backend the same design would take **31–34 hours per
chain**; it is the backend that makes this schedule feasible at all.

## 6. Convergence summaries and thresholds

All computed **only on the 80,000 production draws per arm**, split-chain,
rank-normalized R-hat and bulk/tail ESS (Vehtari et al. 2021).

### 6.1 The gate table

All diagnostics are split-chain, rank-normalized, **pooled across the four chains**, and
computed only on production draws.

| summary class | members | R-hat | bulk ESS | tail ESS |
|---|---|---|---|---|
| log target | `log_target` | **≤ 1.01** | **≥ 1000** | **≥ 500** |
| segmentation | `total_segments`, `mean_segments_per_trace`, `mean_segment_length`, `sd_segment_length` | ≤ 1.01 | ≥ 400 | ≥ 400 |
| boundary | 32 truth-free boundary probes | ≤ 1.01 | ≥ 400 | ≥ 400 |
| co-skill | 64 truth-free co-skill probes | ≤ 1.01 | ≥ 400 | ≥ 400 |
| pi | `pi_entropy`, `pi_l2`, **`sorted_pi[0]`, `sorted_pi[1]`, `sorted_pi[2]`** | ≤ 1.01 | ≥ 400 | ≥ 400 |
| P | the 9 registered `P` summaries | ≤ 1.01 | ≥ 400 | ≥ 400 |
| discrete structural | Section 6.2 | branch-dependent | branch-dependent | branch-dependent |

**R-hat ≤ 1.01 applies to every non-degenerate registered summary without exception.**

#### ESS 400 is a diagnostic floor, not a precision claim

400 is the Vehtari et al. (2021) threshold below which rank-normalized R-hat and quantile
estimates are themselves unreliable — i.e. the point below which the diagnostic stops
being able to detect non-convergence. **It is a minimum diagnostic threshold, not a
guarantee of estimation precision.** An ESS of 400 corresponds to a Monte Carlo standard
error of roughly 5% of a posterior standard deviation, which is adequate to certify mixing
and inadequate for a tight interval.

**Terminal MCSE reporting is therefore mandatory.** Every posterior quantity that appears
in the paper must be reported with its Monte Carlo standard error,
`MCSE = posterior_sd / sqrt(ESS)`, computed from production draws. A quantity whose MCSE is
too large for the claim being made must not carry that claim, regardless of having passed
the gate. Passing the gate licenses "the chains mixed", never "this number is precise".

#### Legacy-threshold reporting, descriptive only

At termination the report must also state, **for every summary**, whether it would have
satisfied the terminated programme's legacy floors (bulk ESS ≥ 1000, tail ESS ≥ 500). This
is recorded so the two protocols can be compared directly and so the effect of the change
is visible rather than buried.

**This comparison is descriptive and carries no authority.** The formal verdict is
determined solely by the gate table above. A summary that passes the new gate and fails the
legacy one is a PASS, and is reported as such with the legacy shortfall stated.

### 6.2 Discrete structural invariants — the exact canonical library

**The diagnosed object is the exact canonical closure library, not relation counts.**

For each skill `k`, take the transitive-closure precedence matrix `H_k = h(U_k)` and
serialise its `m(m-1) = 20` off-diagonal bits. The **canonical library** is the multiset
`{H_0, H_1, H_2}` with its three bit-vectors **sorted**, making it invariant to skill
relabelling, then hashed to a stable identifier. Two draws share a library identifier iff
their partial orders are *exactly* equal up to relabelling.

This replaces relation counts as the primary structural diagnostic. Counts
(`total_relations`, `sorted_relation_counts`) are coarse: two genuinely different orders
can share them, so a count-based diagnostic can report agreement where the structures
differ. They remain **reported** as secondary descriptive summaries, but the branch below
is decided on the exact library identifier alone.

Evaluated over all production draws of all four chains:

**(a) Constant and equal** — one library identifier across every production draw of every
chain. Record **degenerate cross-chain agreement**; **no ESS floor and no R-hat gate**.
A point mass has no sampling variability to estimate; demanding ESS of it is a category
error, and that error is what turned a converged FULL-MARG into a FAIL. **Branch (a) is
admissible only under the preconditions in 6.3.**

**(b) Constant within each chain but unequal across chains** — **automatic FAIL**. This is
the FULL-COND signature: four chains locked in four different libraries. A genuine
multimodality finding, never to be reported as agreement.

**(c) Non-degenerate** — the library identifier varies within at least one chain. Ordinary
gates: R-hat ≤ 1.01 and pooled bulk ESS ≥ 400, computed on an integer encoding of the
library identifier.

### 6.2b Truth-free probe selection, and degeneracy for binary probes

**Selection.** The 32 boundary probes, 64 co-skill probes and 256 recovery co-skill probes
are chosen by `select_truth_free_probes(observed_train_traces, corpus_hash)`: a
deterministic stable rank of candidate indices keyed by the corpus hash. It opens no truth
file and reads only observed CPA sequences. Verified deterministic on repeated calls.
The selected IDs are frozen in `probes_manifest.json`:

| probe set | n | sha256 |
|---|---|---|
| boundary | 32 | `08c390f9b3cbe5cbe8b23b0e1edb75d7da4217936c8eae85a0f8d81a553c78e9` |
| coskill | 64 | `48568cd70a735e86c33395d7149aab9bcb3a1d05c025f7f1a46b54245255842c` |
| recovery_coskill | 256 | `c4af3cce0dabc27cb57e7a4290211aca79a8fde2ab841c9b81c53eb6ece2ce9f` |
| **combined** | 352 | **`3787cde68996257ecb739fc56df7fba59820cc2cc4fc2775f6301d3a5a3206b0`** |

**Degeneracy for binary probes.** Boundary and co-skill probes are Bernoulli indicators, so
the same three-branch logic that governs the canonical library governs them, per probe:

* **constant and equal** across every production chain and draw → **degenerate agreement**;
  no ESS floor and no R-hat gate for that probe. A probe that is 0 in every draw of every
  chain carries no information about mixing and cannot be given an ESS.
* **constant within each chain but unequal across chains** → **automatic FAIL**. This is a
  probe-level lock: each chain is certain, and they disagree.
* **non-degenerate** → ordinary gates, R-hat ≤ 1.01 and pooled bulk/tail ESS ≥ 400.

The count of probes falling in each branch is reported. Degenerate probes are listed
explicitly rather than silently dropped, so a run that passes largely because most probes
were constant is visible as such.

### 6.3 Preconditions on branch (a)

Branch (a) may be claimed **only if all four hold**. If any fails, the summary is treated
as **FAIL**, not as degenerate agreement.

1. **Starts are structurally dispersed.** The four paired starts must occupy at least two
   distinct canonical libraries at sweep 0. Verified and recorded before warm-up begins.
2. **Every chain accepts at least one H-changing move during warm-up.** Per-chain counts of
   accepted structural moves with `h_changed = True` are recorded during warm-up and all
   four must be `>= 1`. A chain that never moved H has not demonstrated it *could*, so its
   agreement is not evidence.
3. **The exact canonical library is constant and equal across all production draws.**
4. **All remaining registered diagnostics pass** their Section 6.1 gates.

Together these separate "the chains explored and agreed" from "the chains never moved".
Only the former is evidence.

**Note on the discarded warm-up.** Warm-up *draws* are discarded entirely and enter no
diagnostic. Precondition 2 retains only per-chain **counters** of accepted H-changing
moves — integers about kernel behaviour, not samples — and precondition 1 records the
libraries of the starts. Neither contributes a draw to any posterior summary, so the
"discarded completely" rule in Section 5 is intact.

## 7. Justification of every ESS floor

Required rather than copied. Two principles are used throughout.

*MCSE.* For a scalar with posterior SD `sigma`, `MCSE = sigma / sqrt(ESS)`. ESS 100 gives
10% of a posterior SD; 400 gives 5%; 1000 gives 3.2%.
*Reliability of the diagnostic itself.* Below ESS ≈ 400 (100 per chain × 4 chains),
rank-normalized R-hat and quantile estimates are themselves unreliable, so a floor beneath
400 would certify convergence with a statistic that cannot be trusted.

| floor | value | justification |
|---|---|---|
| `log_target` bulk | **1000** | The most sensitive global mixing summary: an unvisited mode shows here first. MCSE ≤ 3.2% of a posterior SD. Attainability checked, not assumed: the terminated MARG run reached bulk ESS 66.7 on 4,000 draws and this design retains **20x** that (80,000), projecting ~1,300 — demanding but reachable, which is what a gate should be. Retained from the old protocol and re-derived, not inherited. |
| `log_target` tail | **500** | Tail ESS governs the 5%/95% quantiles that expose a heavy-tailed or bimodal target. Held above the 400 reliability floor because the log target is the summary most likely to reveal a missed mode. |
| all other non-degenerate registered summaries, pooled bulk and tail | **400** | The Vehtari et al. (2021) reliability threshold, 100 draws per chain, below which R-hat and quantile estimates are themselves untrustworthy. **Lowered from the old protocol's 1000, declared in advance.** In the terminated run these summaries mixed freely (boundary probes max R-hat 1.003, `total_segments` bulk ESS 15,108), so 1000 was never the binding constraint and carried no diagnostic value. See the precision caveat in 6.1: this is a diagnostic floor, not a precision guarantee. |
| discrete library, branch (c) bulk | **400** | A discrete summary over few attainable values has intrinsically lower ESS per draw than a continuous one; 1000 demands more effective independence than the state space can supply. 400 preserves the reliability floor. Lowered from 1000, declared in advance, and applied only to this experiment — the terminated verdicts are untouched. |
| discrete library, branch (a) | **none** | A quantity constant and equal across all chains has zero variance; ESS is undefined, not small. Applying a floor here is the exact defect diagnosed in the terminated run. Permitted only under the 6.3 preconditions. |

**Honest pre-launch prediction.** If a discrete invariant lands in branch (c) at the
structural mobility observed in the terminated programme (~0.87 ESS per 1,000 sweeps),
100,000 production sweeps yield ESS ≈ 87 — well below 400, and the gate would FAIL.
That is intended: it means the structure neither locks nor mixes fast enough to certify,
and the honest verdict is that the experiment cannot resolve it at this length. This is
recorded now so that a later failure cannot be reinterpreted as a surprise.

## 8. Stopping rule

Single-shot, no ladder, no extension.

1. Run 50,000 warm-up + 100,000 production per chain. **No interim gate. No peeking that
   could motivate a stop.** Truth-free monitoring may be inspected for operational health
   (crashes, non-finite values) but not to decide stopping.
2. At exactly 150,000 sweeps, compute all Section 6 diagnostics on production draws.
3. **PASS** iff every gate in the 6.1 table passes AND the canonical library is either
   in branch (a) **with all four Section 6.3 preconditions satisfied**, or in branch (c)
   passing R-hat ≤ 1.01 and pooled bulk ESS ≥ 400.
   **FAIL** if any 6.1 gate fails, or the library is in branch (b), or it is in branch (c)
   and misses its floor, or it is constant-and-equal but any 6.3 precondition fails.
   There is **no adaptive stopping**: this single terminal gate is evaluated once.
4. The verdict is terminal for both arms. **There is no extension to a higher ceiling**,
   because the terminated programme demonstrated that extension does not rescue a
   degenerate discrete gate — R-hat improved under counterfactual extension but did not
   reach 1.01 by the 100k ceiling.
5. Only after the verdict is recorded is the truth unsealed (Section 3).

## 9. Sealed recovery protocol

Recovery is computed **after** the verdict is recorded and the truth unsealed, and cannot
change it.

* **Alignment.** Skill labels are exchangeable, so recovery is evaluated over all `3! = 6`
  label permutations and the best-matching permutation is reported, with the full set of
  six also recorded. Alignment is by posterior co-skill structure, never by truth.
* **Structural recovery.** Posterior probability of each true precedence relation; exact
  posterior probability of the true unordered library; Hamming distance between the
  posterior-modal library and truth.
* **Permutation-invariant summaries** (`total_relations`, `sorted_relation_counts`) are
  reported both before and after alignment, since they need none.
* **Segmentation recovery.** Boundary F1 and per-position skill accuracy against the true
  `(S, z)`, at the posterior mode and averaged over draws.
* Recovery is **descriptive**. A PASS/FAIL from Section 8 is never revised by it.

## 10. Held-out negative log-likelihood

Defined now to prevent a post-hoc choice.

For each of the 45 held-out traces, using **only production draws** and the observed CPA
sequence:

    NLL_n  =  -log ( (1/M) * sum_{m=1..M} Z_n(U^(m), pi^(m), P^(m)) )

where `Z_n` is the exact all-segmentation marginal likelihood of trace `n` computed by the
frozen semi-Markov forward recursion — the same `log_normalizer` the sampler already
produces — and `M` is a fixed systematic subsample of **1,000 draws per chain (4,000 per
arm)**, taken at equal spacing through the production phase. Reported as the total over
the 45 held-out traces and the per-trace mean, with a bootstrap interval over traces.

This is a **log pointwise predictive density under the posterior**, computed identically
for both arms, on traces never used for inference. Subsample size and spacing are fixed
here so neither can be tuned after seeing the result.

## 11. Frozen register

Nothing below may change after approval. Any change voids the preregistration and requires
a new one.

| item | frozen value |
|---|---|
| target, priors, fixed coordinates | Section 1 |
| backend commit | `564995efd056d7d33984f0ca1532386e6140ea0c` |
| reference engine sha256 | `8150bb8235eb159d5e2f08ada7c698c383f4c7fd31f3b882e218b207dd135486` |
| corpus / truth hash | `3e3aa653…` / `effccc91…` |
| starts manifest | `43e2b63e…` |
| chain seeds | COND `6306201–04`, MARG `6306211–14` |
| proposal scale / cadence | `structural_scale = 0.5`, `structural_cadence = 10` |
| warm-up / production | 50,000 discarded / 100,000 retained |
| thinning | 5 |
| convergence summaries | Section 6 |
| thresholds | the gate table in 6.1 |
| discrete structural representation | exact canonical closure library (6.2), not relation counts |
| branch (a) preconditions | Section 6.3, all four required |
| probe selection | frozen, combined sha256 `3787cde6…` (6.2b) |
| binary-probe degeneracy | same three branches as the library (6.2b) |
| pi summaries | `pi_entropy`, `pi_l2`, `sorted_pi[0..2]` |
| terminal MCSE reporting | mandatory for every paper-facing posterior quantity (6.1) |
| legacy 1000/500 reporting | descriptive only, no effect on the verdict (6.1) |
| stopping rule | Section 8, single terminal gate, no adaptive stopping, no extension |
| recovery alignment | Section 9, all 6 permutations, post-verdict |
| held-out NLL | Section 10 |

**Not permitted after launch:** tempering, global swap moves, rescues, restarts from a
better state, changes to data/model/K/fixed nuisances/scales/thresholds, truth-informed
starts, held-out tuning, ladder extension, or reclassification of a FAIL.

## 12. Relationship to the terminated programme

This does not resume, extend or reinterpret the terminated run. Those verdicts stand:
**FULL-COND = FAIL, FULL-MARG = FAIL at 30,000**, archived and hashed. This is a new
experiment on a new corpus with a new sealed truth, and its result will be reported
separately with its own provenance.
