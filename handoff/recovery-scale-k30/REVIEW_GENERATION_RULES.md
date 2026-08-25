# Data generation and admissibility — for review

Written so you can check the rules against what the code does, and mark up the two
criteria that need amending. Every constant below is read from the frozen model, not
chosen here.

---

## Part 1 — How the data is generated

### 1.1 The master library, once per replicate

Drawn at `K_max = 30`, then permuted once; each rung `K` is the first `K` skills of the
permuted order, so `K=3 ⊂ 5 ⊂ 10 ⊂ 20 ⊂ 30`.

| object | how it is drawn | shape |
| --- | --- | --- |
| latent utilities `U` | each role's 2-vector ~ `N(0, Σ)`, `Σ = [[1, .5], [.5, 1]]` (ρ = 0.5) | `(30, 10, 2)` |
| partial order | **derived**, not drawn: role *a* precedes *b* iff `U[a]` beats `U[b]` in **both** coordinates | — |
| role maps `ℓ_k` | uniform injective draw of 10 CPAs from 50, resampled until all 30 supports are pairwise distinct | `(30, 10)` |
| permutation | one uniform draw over the 30 skills | `(30,)` |

Seeds: structural truth `6_500_001 + r`, role supports `6_500_051 + r`, permutation
`6_500_101 + r`.

### 1.2 Per rung: `π` and `P`

Drawn per rung from the registered flat Dirichlet (`η = 1`), **not** truncated from the
K = 30 matrix — truncating and renormalising would change the generative model between
rungs. `P` has an exactly zero diagonal, so consecutive segments always differ in skill.
Seed `6_510_000 + 100K + r`.

### 1.3 Per trace

For each of `N_train = 5K` and `N_test = 2K` traces of length `J = 96`:

1. **Segment widths** — one exact draw from the registered boundary prior
   `p(S | J, δ_B = 0.15)`, widths in `[3, 12]`, summing to 96.
2. **Skill path** — first label from `π`, each next from `P[previous]`. Self-transitions
   are impossible by construction.
3. **Roles within a segment** — for a segment of width `w` under skill `k`, run the
   recurrent RFS forward `w` steps from `q = 0`, drawing a role at each step from

   `p ∝ F · exp(β·Q − λ_rep·q − λ_back·C_back)`, then mixed with `ε/m` uniform noise

   with `β = 1.5`, `ω = 1.7346`, `λ_rep = 0.8`, `λ_back = 0.25`, `ε = 0.02`, `m = 10`.
   `F` is the feasibility term: a role is available only once all its predecessors in the
   partial order have fired.
4. **Roles → CPAs** — `cpa = ℓ_k(role)`. Injective, so a CPA identifies its role uniquely
   and the likelihood needs no correction term.

Seeds: train `6_520_000 + 100K + r`, held-out `6_530_000 + 100K + r`. Every trace and every
segment gets its own derived stream, so a trace is reproducible in isolation.

### 1.4 What this means for the scorer

A block whose CPAs all lie inside skill `k`'s support is translated back to role indices
and scored by the ordinary recurrent likelihood. A block containing **any** CPA outside
that support scores `−inf`: skill `k` cannot have produced it. Measured effect — about
**19%** of candidate block-skill pairs survive.

---

## Part 2 — The criteria, as registered

### Section 7, master truth — criteria 1–7 (per skill)

| # | criterion | status |
| --- | --- | --- |
| 1 | role map is injective | **holds** by construction |
| 2 | induced closure is a valid strict partial order | **holds** |
| 3 | relation count ≥ 1 | **holds** |
| 4 | relation count < `m(m−1)/2` = 45 | **holds** |
| 5 | all 30 role-labelled closures pairwise distinct | **holds** |
| 6 | all 30 CPA supports distinct | **holds** by construction |
| 7 | every support holds exactly 10 roles | **holds** by construction |

Measured on replicate 0: relations per skill range 16–36, accepted on the first attempt.

### Section 7, per-rung `π`/`P` — criteria 8–12

| # | criterion | status |
| --- | --- | --- |
| 8 | `π` valid | holds |
| 9 | `P` non-negative, row-normalised, zero diagonal | holds |
| 10 | unique stationary occupancy | holds |
| 11 | each **stationary** probability in `[0.5/K, 1.5/K]` | **reachable** — 50–84% |
| 12 | each **`π`** component in `[0.5/K, 1.5/K]` | **UNREACHABLE at K ≥ 10** |

Exact acceptance for criterion 12, `p_K = Σ_j (−1)^j C(K,j) (1/2 − j/K)₊^{K−1}`, verified
numerically:

| K | exact `p_K` | 1 / `p_K` |
| --- | --- | --- |
| 3 | 1.667e−01 | 6 |
| 5 | 2.300e−02 | 43 |
| 10 | 1.562e−04 | 6,402 |
| 20 | 7.116e−09 | 1.4e8 |
| 30 | 3.234e−13 | 3.1e12 |

An earlier draft quoted ~1/2,000 at K = 10 from a 4,000-draw estimate that contained two
successes. The closed form above supersedes it.

### Section 8, corpus coverage (per skill)

| criterion | status |
| --- | --- |
| training ≥ 30 true instances | holds |
| training 240–720 CPA occurrences | holds |
| training: every role appears ≥ 5 times | **CANNOT BE MET BY RESAMPLING** |
| held-out ≥ 8 instances, ≥ 60 occurrences | holds |

---

## Part 3 — Measured feasibility

### Criterion 12

Acceptance of a flat-Dirichlet draw, against the registered cap of 100 attempts:

| K | π band | stationary band | both | expected attempts |
| --- | --- | --- | --- | --- |
| 3 | 0.178 | 0.781 | 0.137 | 7 |
| 5 | 0.020 | 0.548 | 0.011 | 95 |
| 10 | 0.0005 | 0.502 | not observed | ~2,000+ |
| 20 | ~0 | 0.665 | not observed | — |
| 30 | ~0 | 0.836 | not observed | — |

The stationary band is not the binding constraint. Requiring **every one** of K independent
components to sit inside a band of width `1/K` becomes vanishingly unlikely as K grows.

### The per-role band

Role frequency is fixed by `U`, not by the corpus draw. One truth, four corpus seeds:

    skill 0 per-role counts:  [29, 53, 12,  4,  2, 222, 158, 41, 40, 32]
                              [39, 36,  4,  5,  1, 191, 130, 39, 39, 36]
                              [38, 48,  6,  2,  3, 201, 154, 39, 31, 39]
                              [37, 34,  5,  8,  2, 208, 134, 48, 30, 33]

First-step emission probabilities for that skill:

    [0.002, 0.095, 0.002, 0.002, 0.002, 0.889, 0.002, 0.002, 0.002, 0.002]

A 445-fold spread, with the floor at exactly `ε/m = 0.002` — those roles fire **only**
through the noise term. **All 30 master skills** have at least one role below 0.01.

The mechanism: the partial order gates emission. A role deep in the order, or dominated on
both latent coordinates, is structurally suppressed. Resampling the corpus cannot change
that; only redrawing `U` can.

---

## Part 4 — Decisions after two rounds of external review

Both of my original amendments were wrong and both audits were right. What follows is what
is **implemented**, with the withdrawn reasoning kept visible.

### Implemented

| decision | what it is |
| --- | --- |
| **D — soft feasibility, retained** | The frozen confirmatory RFS semantics are kept unchanged: `(1−ε)·F·exp(·)/Σ + ε/m`, full-support contamination, `q₀ = 0` per block. |
| **`π = ν(P)`** | Draw `P` from the registered flat-Dirichlet row model, require `ν(P)` inside `[0.5/K, 1.5/K]`, then set `π = ν(P)`. |
| **No corpus acceptance, at all** | Train and held-out are each generated exactly once. No realised-count rejection of any kind. |
| **`role_exposure`** | Reports expected count per segment *and* probability of at least one occurrence, both with Monte-Carlo standard error. |

### D — the correct statement of the model

Not "soft partial order". The latent structure is a **hard acyclic partial order**; the
*observation* model has full-support contamination:

> The structured component assigns zero probability to a role until all its predecessors
> have been observed at least once, and thereafter grades compatibility through the
> recurrent state `q ∈ [0,1]` — so `F` is graded, not binary. A fixed full-support
> contamination component permits off-frontier observations. **Every observed role,
> including an off-frontier one, drives the same recurrent-state update.**

Measured, K = 10, 150 traces of J = 96:

| off-frontier events per trace | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| traces | 163 | 124 | 72 | 35 | 4 | 2 |

Mean 1.00, **but 41% of traces have none** — an earlier draft wrote "almost every trace has
one", which does not follow from a mean. Cascade effect (structured emissions made feasible
only by an earlier contamination): **9 of 38,001 = 0.02%**, against a direct off-frontier
rate of 1.04%.

### Why `π` could not be left free

Each of the `5 × K` traces contributes a fresh first segment, so skill `k` receives about
`5·K·π_k` instances from first segments alone — linear in `K` — while a balanced skill's
total is about `5·E[L] ≈ 71` and does not grow with `K`. Measured max/min ratio of expected
per-skill instances under an unconstrained flat Dirichlet:

| K | 3 | 5 | 10 | 20 | 30 |
| --- | --- | --- | --- | --- | --- |
| ratio | 2.32 | 2.65 | 2.48 | 2.32 | 2.09 |
| first-segment instances | 9.1 | 11.6 | 15.2 | 18.0 | 20.2 |

My earlier claim that `π` "only affects one segment in thirteen" counted segments within a
single trace and ignored that every trace contributes one. `E[L] = 14.18` at `J = 96`.

### Why the corpus loop had to go entirely

Rejecting until a coverage event `A` holds samples `p(D | θ, A) = p(D | θ)·1{A}/P_θ(A)`
while the sampler scores `p(D | θ)` with no `−log P_θ(A)` term. `P_θ(A)` depends on `U` and
the recurrent dynamics, so the omission biases each rung differently. Searching for a truth
seed that yields agreeable realised counts is the same bias by another route, and is
equally forbidden. A structural test asserts no acceptance loop can reappear by edit.

**What deleting it revealed.** Coverage that the rejection loop had been hiding:

| K | instances/skill | occurrences/skill | max/min instances | roles never seen |
| --- | --- | --- | --- | --- |
| 3 | 64–75 | 462–505 | 1.17 | 1 |
| 5 | 43–95 | 264–647 | 2.21 | 2 |
| 10 | 47–104 | 327–715 | 2.21 | 7 |
| 20 | 43–98 | 283–646 | 2.28 | 15 |
| 30 | 42–100 | 290–675 | 2.38 | 18 |

Occurrences sit inside the old 240–720 reference band at every rung, but instance counts
still vary by a factor of ~2.3 and 18 roles are never observed at K = 30. **That is the
real generative difficulty**, previously masked. It is reported, not corrected.

### Withdrawn claims

* **"Require every role's first-step probability above `5ε/m`."** Impossible: a role with
  any predecessor has `F = 0` at `q = 0`, so its first-step probability is exactly `ε/m`.
  Verified on 1,395 such roles, zero exceptions. The requirement forces an antichain, which
  criterion 3 forbids. "All 30 skills have a role below 0.01" was a tautology, not a finding.
* **"Truncating `P` changes the per-rung law."** False — a symmetric Dirichlet is
  projective; truncation changes rung-to-rung coupling only.
* **"Role coverage is an upper bound on recovery."** Too strong. Absence is itself
  likelihood information, transitivity constrains unobserved edges, and the prior
  contributes. Report as a **data-supported observability diagnostic**.
* **"Directly comparable with the confirmatory experiment."** Sharing
  `sample_recurrent_rfs_sequence` establishes a shared likelihood family and reusable
  parity tests — not statistical comparability. The ladder additionally changes `K`,
  `N_train`, `N_test`, support construction, truth admissibility, occupancy conditioning
  and CPA ambiguity. The `K = 3` rung may serve as a sanity bridge; recovery numbers must
  not be pooled.

### E — support-only baseline: implemented, and the result is stronger than expected

The baseline replaces the recurrent score with `0` for a support-compatible block and
`−inf` otherwise, leaving the segmentation prior and transition matrix untouched. Measured
on training corpora from replicate 0:

| K | pooled survival, support-only | pooled survival, full model | true-skill pairs | wrong-skill pairs | cross-boundary |
| --- | --- | --- | --- | --- | --- |
| 3 | 0.1297 | 0.1297 | 1.0000 | 0.0368 | 0.0413 |
| 10 | 0.0391 | 0.0391 | 1.0000 | 0.0112 | 0.0098 |
| 30 | 0.0164 | 0.0164 | 1.0000 | 0.0071 | 0.0043 |

**The two survival columns are identical to machine precision, at every rung.** A
support-compatible block always receives a finite recurrent score and an incompatible one
is `−inf` under both models, so `isfinite` agrees element-for-element across the whole dense
table. The candidate set is decided **entirely** by support membership; the recurrent
likelihood eliminates nothing.

Three consequences for how the ladder may be read:

* Whatever the partial-order component contributes, it is in **how surviving candidates are
  weighted**, never in which ones exist. Any claim that partial-order inference "identifies
  skills" has to be made against the support-only baseline, not against a null model.
* A block's own skill is always support-compatible (survival exactly 1.0), while wrong
  skills survive at 0.0071 by `K = 30`, falling monotonically with `K`. The design gets
  *easier* to discriminate as `K` grows, in this one respect — the opposite of the
  degradation the ladder is looking for, and it must not be mistaken for it.
* Cross-boundary candidates survive at 0.0043 at `K = 30`, so the support rule alone
  forbids most of them. Segmentation is being shaped before any likelihood is consulted.

Survival is reported stratified by block length, distinct-CPA count, true/false skill pair
and boundary crossing; the closed form `(K−1)·C(A−d, m−d)/C(A, m)` is verified against the
measured strata, and survival falls monotonically with distinct-CPA count as it predicts.

### Monte-Carlo diagnostics validated by exhaustive enumeration

`role_exposure` is checked at `m = 3`, `w ∈ {2, 3}` against a full enumeration of all `m^w`
role paths: the enumerated distribution sums to 1, expected counts sum to the width,
`P(at least one) ≤ E[count]` with equality only when repeats are impossible, and the
estimator lands within four of its own standard errors of the exact value for every role.
Width integration over a requested range is checked to be real rather than collapsing to a
single width.

### Still outstanding — not implemented

* **Partial pairing.** Skill structures are nested across rungs; transition environments are
  not, since `P` is redrawn per rung. Gamma coupling would pair them: `G_j ~ iid Gamma(1,1)`
  shared across rungs, each rung's `π`, `P` rows formed from the first `K` weights — the
  marginal stays flat Dirichlet.
* **Naming.** This is **constant-evidence-per-skill scaling**, or joint library-and-corpus
  scaling — not pure `K`-scaling. `N_train = 5 × K`, and a CPA belongs to `K/5` skills on
  average, so 6 skills at `K = 30`. Report normalised cost (per trace, per CPA occurrence,
  per surviving block-skill pair) and MCMC efficiency alongside raw runtime.
* **`ω = 1.7346`** is listed but its entry point into `Q` is not written out.
* **Seed bands** `6_500_001 + r` and `6_500_051 + r` collide at `r ≥ 50`; use
  `SeedSequence.spawn` or tuple keys.
