# Scalability generation specification — FROZEN

| | |
| --- | --- |
| schema version | `1.0.0` |
| frozen at commit | `4c56eebb2bb371a57a495ed7056f887ea796f413` |
| date | 2026-08-25 |
| sealed backend | `564995efd056d7d33984f0ca1532386e6140ea0c` |
| supersedes | `REVIEW_GENERATION_RULES.md`, retained as audit history only |

`REVIEW_GENERATION_RULES.md` is **SUPERSEDED**. It still contains withdrawn rules — the
`pi` band, realised-corpus coverage rejection, the first-step starvation explanation, and
Amendments A and B — kept so the errors stay visible. **Do not implement from it.** This
file is the specification; the changelog at the end records every withdrawal.

---

## 1. Model

### 1.1 Emission within one segment

A segment of width `w` under skill `k` is generated from `q = 0`, one role per step.
At each step, with `P` the transitive closure of skill `k`'s partial order and
`q ∈ [0,1]^m` the recurrent state:

    F(x)      = prod over z of q[z], for z a predecessor of x        (1 if x has none)
    Q(x)      = log1p( number of stale successors of x under q )
    C_back(x) = sigmoid(omega) * sum over z != x of P[x,z] * q[z]

    w~(x)     = F(x) * exp( beta*Q(x) - lambda_rep*q[x] - lambda_back*C_back(x) )
    p~(x)     = w~(x) / sum_y w~(y)
    p(x)      = (1 - epsilon) * p~(x) + epsilon / m

**`omega` enters only through `C_back`, as `kappa = sigmoid(omega)`.** It does not
appear in `Q`, in `F`, or anywhere else. An earlier draft listed it as a parameter
without saying where it acted, which made the specification unreproducible.

Registered values: `beta = 1.5`, `omega = 1.7346010553881064` so
`kappa = 0.8500`, `lambda_rep = 0.8`, `lambda_back = 0.25`, `epsilon = 0.02`,
`m = 10`.

### 1.2 State update — including after an off-frontier observation

After observing role `x`, for every role `z`:

    q[z] <- q[z] * (1 - kappa)   if x precedes z
    q[x] <- 1

**This runs identically whether or not `F(x) = 0`.** Generator and scorer share it.

### 1.3 Feasibility is hard in the structure, soft in the observation

The latent poset is a **hard acyclic partial order**. The structured component gives a
role zero weight until all its predecessors have been observed, and thereafter grades
compatibility through `q ∈ [0,1]` — so `F` is **graded, not binary**. The fixed
full-support contamination `epsilon/m` permits off-frontier observations.

Consequence, and it is a theorem rather than a finding: a role with any predecessor has
`F = 0` at `q = 0`, so its first-step probability is **exactly** `epsilon/m`. Any
admissibility rule phrased on first-step probability is unsatisfiable whenever the poset
has one relation. Measured off-frontier rate 0.93% of steps; **41% of traces contain
none**; downstream cascade 0.02%.

### 1.4 CPA vocabulary

Skill `k` owns an injective `ell_k : {0..m-1} -> {0..A-1}`. A block is scored under
`k` by translating each CPA to its role index; a block containing any CPA outside
`k`'s support scores `-inf`. `A = 50`, `m = 10`.

Because `epsilon > 0`, `log p(block | k) > -inf` **iff** the block is
support-compatible. This is a **generator–scorer consistency invariant**, not evidence
about the partial-order component's contribution.

---

## 2. Truth

### 2.1 Master library, one per replicate

`K_max = 30`, permuted once; rung `K` is the first `K` skills, so
`L_3 ⊂ L_5 ⊂ L_10 ⊂ L_20 ⊂ L_30`. Nesting covers utilities, role maps and closures.

Admissibility, all functions of the truth alone: role map injective; closure a valid
strict partial order; relation count in `[1, m(m-1)/2)`; closures pairwise distinct;
supports pairwise distinct; support size exactly `m`.

### 2.2 Per rung: `pi = nu(P)`

Draw `P` row-wise from the registered flat Dirichlet with exactly zero diagonal, require
its stationary law `nu(P)` inside `[0.5/K, 1.5/K]`, then **set `pi = nu(P)`**.

Every segment index then has marginal `nu`, with no reliance on mixing. This is a
**ladder-specific controlled truth design in the same likelihood family as the
confirmatory experiment** — not an identical prior-predictive draw. Given the truth the
data likelihood is exactly right; no prior-predictive calibration claim may be made.

### 2.3 Seeds

Hierarchical, from `hpop.mcmc_cpa.seeds`: one root (`6_500_000`) and a
`SeedSequence` spawn key per `(stream, replicate, rung, index, component)`. The
registered hand-offset bands are **not** used to separate streams — they collide 300 times
over replicates 0–199, and the per-rung `100*K` bands collide for `K > 100`.

---

## 3. Data

`N_train = 5 x K`, `N_test = 2 x K`, `J = 96`, widths in `[3, 12]`,
`delta_B = 0.15`. Per trace: widths from the registered boundary prior; skill path from
`pi` then `P`; roles per segment from §1.1; CPAs by `ell_k`.

**Corpora are generated exactly once.** There is no acceptance loop on realised counts,
and no search over truth seeds for agreeable realised counts. Rejecting until an event
`A` holds samples `p(D | theta, A)` while the scorer uses `p(D | theta)` with no
`-log P_theta(A)` term, and `P_theta(A)` depends on `U` — a likelihood mismatch, not
an inefficiency. A structural test asserts no loop can reappear.

Instance, occurrence, role-exposure and held-out coverage are **recorded as diagnostics**.

---

## 4. Diagnostics and their denominators

| quantity | definition | population |
| --- | --- | --- |
| `p_pair` | wrong-skill support compatibility | blocks contained in one true segment |
| `E[C_b]` | wrong compatible skills per block | same, and `E[C_b] = (K-1) p_pair` exactly |
| `non_true_pair_survival_ALL_BLOCKS` | any non-own-skill candidate | **includes boundary-crossing blocks**; does *not* satisfy the identity |
| `role_exposure` | expected count per segment **and** `P(at least one)` | pilot streams, MC standard error reported |
| `R_k` | realised instances / `nu_k * sum_i L_i` | finite-horizon expectation under `pi = nu` |

Theory: `p_d = [C(A-d, m-d) - 1] / [C(A, m) - 1]` under pairwise-distinct supports.
**`p_d` contains no `K`.** Only the count of competing skills grows, so a measured
change in per-pair survival across rungs is a change in the block population averaged
over, never `K` altering an individual pair.

Measured on replicate 0, contained blocks:

| K | `p_pair` | `E[C_b]` | `P(C_b>=1)` | `E[C_b \| C_b>=1]` |
| --- | --- | --- | --- | --- |
| 3 | 0.0202 | 0.0404 | 0.040 | 1.00 |
| 5 | 0.0089 | 0.0357 | 0.036 | 1.00 |
| 10 | 0.0152 | 0.1366 | 0.115 | 1.19 |
| 20 | 0.0141 | 0.2683 | 0.187 | 1.43 |
| 30 | 0.0144 | 0.4187 | 0.249 | 1.68 |

Block-level ambiguity is **approximately flat over K = 3–5, then increases substantially
through K = 30**; the empirical curve is not required to be monotone. Cross-boundary
compatibility stays in ~0.09–0.13 and shows **no systematic improvement** with `K`.

Zero-exposure roles: report **both count and proportion**, across all five rungs and both
replicates. Proportion is roughly stable from `K = 10` (0.05–0.09); only the absolute
count grows. Because `epsilon > 0` such a role is never structurally impossible — say
**no direct role-specific evidence in this realised corpus**, never "cannot be recovered".

---

## 5. Naming

**constant-evidence-per-skill scaling**, or joint library-and-corpus scaling. Not pure
`K`-scaling: `N_train = 5 x K`, and a CPA belongs to `K/5` skills on average — six at
`K = 30`. Report normalised cost (per trace, per CPA occurrence, per surviving
block-skill pair) and MCMC efficiency alongside raw runtime.

Sharing `sample_recurrent_rfs_sequence` with the confirmatory experiment establishes a
shared likelihood family and reusable parity tests — **not** statistical comparability.

---

## 6. Changelog of withdrawn rules

| withdrawn | why |
| --- | --- |
| `pi` component band `[0.5/K, 1.5/K]` | exact acceptance `3.2e-13` at `K = 30` against a cap of 100 |
| first-step emission floor `5 eps/m` | impossible: `F = 0` at `q = 0` forces exactly `eps/m`; verified on 1,395 roles, zero exceptions |
| realised-corpus acceptance loop | samples `p(D \| theta, A)` while the scorer uses `p(D \| theta)` |
| "truncating `P` changes the per-rung law" | false; symmetric Dirichlet is projective |
| "role coverage bounds recovery" | absence is likelihood information; transitivity constrains unobserved edges |
| "larger `K` is easier to discriminate" | per-pair fell but competitors grew as `K-1`; `E[C_b]` rises 0.040 → 0.419 |
| "directly comparable with confirmatory" | shared likelihood family, not statistical comparability |
| hand-offset seed bands | 300 collisions over replicates 0–199 |

---

## 7. Outstanding before production

1. ~~**Support-only inference baseline**~~ — **done**. `hpop.mcmc_cpa.ladder_runner` is a
   single runner whose `arm` argument is the only difference between the two conditions,
   so the segmentation prior, transition treatment, data, initialisation, proposal
   schedule, sweep schedule and RNG stream are shared by construction rather than by two
   implementations written to match. Structure recovery is reported as
   `"NOT APPLICABLE"` for the baseline: its score never reads `U`, so any `U` it produced
   would be a prior draw.
2. ~~A short `K = 3` / `K = 30` end-to-end smoke run~~ — **done**, extended to
   `K ∈ {3, 10, 30}`; see §7.1.
3. ~~**Gamma coupling**~~ — **adopted and frozen**; see §8.

### 7.1 Smoke result (not a result — a sanity check)

`scripts/k_ladder/smoke_three_arms.py`, one chain, one corpus replicate, 100 sweeps, 50
warm-up, thin 5, library seed 0, **on shared-Gamma coupled corpora** (§8). `U` is fixed in
`support-only` and `oracle-order`; `learned-order` starts dispersed and infers it.

| `K` | arm | s | boundary F1 | skill acc | FFBS states changed |
| --- | --- | --- | --- | --- | --- |
| 3 | support-only | 0.8 | 0.7919 | 0.9404 | 1442 |
| 3 | oracle-order | 0.8 | 0.9020 | 0.9619 | 1152 |
| 3 | learned-order | 21.2 | 0.8197 | 0.9307 | 809 |
| 10 | support-only | 2.8 | 0.8020 | 0.9339 | 4771 |
| 10 | oracle-order | 3.2 | 0.9523 | 0.9889 | 2379 |
| 10 | learned-order | 92.9 | 0.7557 | 0.9029 | 4228 |
| 30 | support-only | 23.6 | 0.7937 | 0.8947 | 14610 |
| 30 | oracle-order | 28.1 | 0.9502 | 0.9871 | 7879 |
| 30 | learned-order | 533.2 | 0.6780 | 0.7796 | 13383 |

**available order information** (`oracle − support`)

| `K` | Δ boundary F1 | Δ skill accuracy |
| --- | --- | --- |
| 3 | +0.1102 | +0.0215 |
| 10 | +0.1503 | +0.0550 |
| 30 | +0.1565 | +0.0924 |

**realised end-to-end gain** (`learned − support`)

| `K` | Δ boundary F1 | Δ skill accuracy |
| --- | --- | --- |
| 3 | +0.0278 | -0.0097 |
| 10 | -0.0464 | -0.0311 |
| 30 | -0.1157 | -0.1151 |

**inference gap** (`oracle − learned`)

| `K` | Δ boundary F1 | Δ skill accuracy |
| --- | --- | --- |
| 3 | +0.0824 | +0.0312 |
| 10 | +0.1966 | +0.0861 |
| 30 | +0.2722 | +0.2075 |

Statistics are exactly reproducible from the recorded seeds; only the `s` column moves.

### 7.1.1 Which of these numbers survives a change of random stream

This smoke was run twice under two independent common-random-number roots, identical in
every other respect. Comparing them is a crude `n = 2` stability check, and it separates
the columns sharply (boundary F1):

| contrast | `K` | run A | run B | \|diff\| | sign |
| --- | --- | --- | --- | --- | --- |
| oracle − support | 3 | +0.0934 | +0.1102 | 0.0168 | same |
| oracle − support | 10 | +0.1577 | +0.1503 | 0.0074 | same |
| oracle − support | 30 | +0.1588 | +0.1565 | 0.0023 | same |
| learned − support | 3 | −0.0331 | +0.0278 | 0.0609 | **flips** |
| learned − support | 10 | +0.0283 | −0.0464 | 0.0747 | **flips** |
| learned − support | 30 | −0.1365 | −0.1157 | 0.0208 | same |
| oracle − learned | 3 | +0.1265 | +0.0824 | 0.0441 | same |
| oracle − learned | 10 | +0.1295 | +0.1966 | 0.0671 | same |
| oracle − learned | 30 | +0.2953 | +0.2722 | 0.0231 | same |

`oracle − support` moves by at most 0.017 and never changes sign — the one contrast this
smoke can support. **`learned − support` changes sign at two of three rungs.** It is not a
small effect measured imprecisely; at this budget it is noise, demonstrated by measurement
rather than argued from first principles.

### 7.1.2 Why the learned arm is noise here, and must not be quoted

The arm runs and moves `U`, but its `U` budget is far too small for it to have learned
anything. At 100 sweeps with one proposal per sweep:

| `K` | `U` rows (`K·m`) | `U` proposals | proposals per row |
| --- | --- | --- | --- |
| 3 | 30 | 100 | 3.33 |
| 10 | 100 | 100 | 1.00 |
| 30 | 300 | 100 | 0.33 |

At `K = 30` that is one proposal per three rows, so the arm reports a barely-moved
dispersed draw. A wrong `U` actively misleads segmentation where agnostic support
membership does not, which is why the contrast can go **negative** at all.

This is the concrete reason the preregistration must not call `oracle − support` an upper
bound on `learned − support`. Under a bound framing a negative realised gain is
unsayable; here it is simply what happened, in both realisations at `K = 30`.

**Production consequence.** `learned-order` needs a `U` burn-in budget scaled to `K·m`
rather than a fixed sweep count, and `u_scale` (currently 0.5, untuned) needs its own
efficiency-only pilot per rung — proposal scales do not transfer between corpora. Neither
is set. Until both are, the production ladder should report `oracle − support` and treat
the learned arm as **unvalidated**.

### 7.1.3 Caveats that apply to the whole table

- One chain, one replicate, 100 sweeps. Nowhere near a posterior.
- `oracle-order` scores at the **true** `U`. Each arm gets ground-truth side information
  of its own kind — the baseline the true supports, the oracle arm the true supports *and*
  the true within-skill order. `oracle − support` is therefore an **oracle information
  diagnostic**: what the available order information is worth to a sampler handed it. It
  is **not** a strict upper bound on `learned − support`; §7.1.1 is the demonstration.
- Structure recovery is reported for `learned-order` only: `support-only` never reads `U`
  and `oracle-order` is handed it.
- These corpora are **jointly conditioned across rungs** (§8.3), so the numbers are not
  comparable with any pre-coupling run.

### 7.2 The gap is consistent with the predicted ambiguity

`expected_compatible_wrong_skills` is closed-form support combinatorics and never sees a
chain. It gives the expected number of *wrong* skills that can accommodate a block of `d`
distinct CPAs (`m = 10`, `A = 50`):

| `K` | `d = 2` | `d = 3` | `d = 4` | `d = 5` |
| --- | --- | --- | --- | --- |
| 3 | 0.0735 | 0.0122 | 0.0018 | 0.0002 |
| 10 | 0.3306 | 0.0551 | 0.0082 | 0.0011 |
| 30 | 1.0653 | 0.1776 | 0.0264 | 0.0034 |

It grows roughly as `K − 1`. The measured arms are consistent with that direction: the
oracle arm's skill accuracy is 0.9619, 0.9889, 0.9871 across the ladder while the
baseline's is 0.9404, 0.9339, 0.8947, the substantial degradation appearing at `K = 30`.

Three cautions. Neither series is monotone, and no monotonicity in `K` is claimed here or
anywhere else. The largest observed gap is at `K = 30`; that is a statement about three
points, not a trend. And this is a consistency check between a prediction and one chain per
rung, not a validation of either — what it was run to exclude is a baseline that *improved*
with `K`, and it would have caught that.

---

## 8. Shared-Gamma coupling (adopted)

Skills are nested across the ladder; until this was adopted, transition environments were
not. Each rung drew its own `P` independently, so "the same skill at a larger `K`" shared
its emissions and shared nothing about how it was reached or left — and a trend across the
ladder mixed two causes that could not be separated afterwards: more skills to confuse,
and a different transition environment.

### 8.1 The construction

For each replicate draw one master directed weight matrix

    G_ij ~ iid Gamma(1, 1),   i != j,   i, j < K_max

apply the master skill permutation to **both** axes, and cut each rung out by restriction
and renormalisation:

    P^(K)_ii = 0
    P^(K)_ij = G_ij / sum_{h < K, h != i} G_ih          i != j,  i, j < K
    pi^(K)   = nu(P^(K))

### 8.2 Why this construction

**The per-rung law is untouched.** For `G_ij` iid `Gamma(1,1)` the normalised vector
`(G_ij / sum_h G_ih)_{h != i}` is exactly `Dirichlet_{K-1}(1, ..., 1)` — the Gamma
representation of the Dirichlet. Each rung's rows keep precisely the registered
flat-Dirichlet marginal *before* admissibility conditioning; the coupling costs nothing in
the marginal model. Tested two ways: a one-sample KS test against the exact `Beta(1, K-2)`
coordinate marginal, and a two-sample KS test against the registered
`sample_transition_matrix` itself. Mutation-checked — the same test rejects `Gamma(2,1)`
at `p = 3e-43` and `Gamma(0.5,1)` at `p = 1e-52`.

**Relative preferences among old skills survive.** For destinations `j, l` both present at
the smaller rung,

    P^(K)_ij / P^(K)_il = G_ij / G_il

independent of `K`. Growing the ladder dilutes every old destination by one common
normaliser — the row total — and reorders nothing. Tested over all old triples at every
rung, and separately that the dilution factor is common within a row.

### 8.3 The conditioning is joint, and that must be stated

The stationary-occupancy band is applied to **all five rungs at once**: one master `G` is
drawn, every rung is built from it, and the whole draw is accepted or rejected together.
Rejecting and redrawing a single failing rung would replace that rung's `G` rows with
fresh ones and destroy exactly the coupling this exists to create, so it is not done, and
a test asserts every accepted rung reduces to the one accepted `g`.

**The final ladder is jointly conditioned across rungs.** No rung's transition matrix is a
draw from the unconditional flat-Dirichlet law; each is a draw from that law conditioned on
the event that *every* rung of the same master satisfies the band. This is stronger and
different conditioning than per-rung acceptance, and the two are not interchangeable. Any
statement about the ladder's transition law must carry it.

### 8.4 Measured joint acceptance

Replicate 0, 1000 trials, `K_LADDER = (3, 5, 10, 20, 30)`:

| rung | own band rate |
| --- | --- |
| 3 | 0.772 |
| 5 | 0.592 |
| 10 | 0.498 |
| 20 | 0.699 |
| 30 | 0.865 |

Product if independent: 0.138. **Measured joint rate: 0.200** — about five attempts. The
band events are positively correlated under a shared `G`, so joint conditioning costs less
than independence would suggest. The rate, the attempt count and every rejection's failing
rungs are recorded in each corpus's `coverage["transition_coupling"]`; nothing is tuned on
them.

---

## 9. Three arms

    support-only     the block score knows only which CPAs a skill can emit
    oracle-order     the full recurrent score at the TRUE U, held fixed
    learned-order    the full recurrent score, U inferred from a dispersed start

| contrast | reads as |
| --- | --- |
| `oracle − support` | how much the **available order information** is worth |
| `learned − support` | the **realised end-to-end gain** |
| `oracle − learned` | the **inference gap** |

`oracle − support` is an **oracle information diagnostic, not a bound**. An inferred `U` is
not obliged to be less useful than the true one at every finite sweep count, and averaging
over a posterior is not the same operation as plugging in a point truth. `learned − support`
is the number a practitioner gets. `oracle − learned` is the one that says whether effort
belongs in inference or in the model.

Structure recovery is reported for `learned-order` only: `support-only` never reads `U`,
and `oracle-order` is handed it.

---

## 10. `table_source="fast"` — exact, and gated

`CPABlockScoreTable.refresh_changed` rebuilds only the skills whose `U` moved, a factor of
`K` on the dominant cost. `scripts/k_ladder/fast_exact_parity_gate.py` compares it against
the all-skills rebuild at `K = 3, 10, 20, 30` on finite masks, entrywise scores, one-skill
`U`-update deltas, MH log-ratios, and accept/reject trajectories under a shared uniform.
**Exactness is the pass rule**; `max_abs_difference` is reported for context and is never
the criterion. A single mismatch is a production blocker.

**A trap the gate had to be hardened against.** The candidate score reads `U` *only*
through the induced precedence relation. A small random nudge therefore often leaves a
skill's entire score column bit-identical, so a parity check made of such moves compares
unchanged arrays and passes while testing nothing. The gate now counts how many proposals
actually moved a column and refuses to pass on too few; the mutation tests reverse a
skill's order rather than nudging it. Stated as a model invariant: a monotone rescale of
`U` is the same latent poset and gives bitwise-identical candidate scores.

---

## 11. Common random numbers

Identical initial states are necessary and nowhere near sufficient. Two arms drawing from
one sequential generator share only a prefix; once one accepts a move the other rejects,
the streams slip and every later difference confounds randomness with model.

Every generator is derived from `(replicate, K, chain, sweep, move type, proposal index)`
through `SeedSequence.spawn_key`, so it depends on where it sits in the design and never on
consumption. Tests drive one arm's CRN hard and check it still matches an untouched one at
every index, and check both arms share the FFBS uniforms sweep by sweep *after* they have
demonstrably diverged.

**Residual.** `ffbs_segmentation_draw` loops over traces against one generator and a
trace's consumption depends on how many segments it draws, so within a single sweep traces
after the first can still misalign. Fixing it means editing the sealed backend. The
guarantee is that misalignment **cannot propagate across sweeps or move types**;
`crn_alignment_report` measures it rather than assuming it.
