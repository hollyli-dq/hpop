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
3. **Gamma coupling** — skills are nested across rungs, transition environments are not.
   Shared `Gamma(1,1)` weights would pair them with the marginal unchanged. A design
   choice, **still open**, to be made before production replicates.

### 7.1 Smoke result (not a result — a sanity check)

`scripts/k_ladder/smoke_full_vs_support.py`, one chain, one corpus replicate, 100 sweeps,
50 warm-up, thin 5, library seed 0. `U` held fixed in both arms.

| `K` | arm | s | boundary F1 | skill acc | FFBS states changed |
| --- | --- | --- | --- | --- | --- |
| 3 | full-RFS | 0.9 | 0.9059 | 0.9748 | 1200 |
| 3 | support-only | 0.8 | 0.7867 | 0.9426 | 1481 |
| 10 | full-RFS | 3.5 | 0.9508 | 0.9852 | 2274 |
| 10 | support-only | 3.0 | 0.8333 | 0.9449 | 4540 |
| 30 | full-RFS | 31.5 | 0.9436 | 0.9875 | 8446 |
| 30 | support-only | 26.6 | 0.7838 | 0.8875 | 14721 |

| `K` | Δ boundary F1 | Δ skill accuracy |
| --- | --- | --- |
| 3 | +0.1191 | +0.0322 |
| 10 | +0.1175 | +0.0402 |
| 30 | +0.1598 | +0.0999 |

Statistics are exactly reproducible from the recorded seeds; only the `s` column moves between runs.

The full likelihood wins at every rung, and its margin **grows with `K`** — which is what
the corrected ambiguity table in §6 predicts: `E[C_b]` rises 0.040 → 0.419 across the
ladder, so support membership alone discriminates progressively worse while the recurrent
score does not degrade. The baseline's FFBS movement is roughly double the full arm's at
every rung, consistent with a flatter score wandering among support-feasible labellings.

**Read this as a floor, not as the headline.** Three caveats, all recorded in the run's
JSON:

- One chain, one replicate, 100 sweeps. Nowhere near a posterior.
- The full arm scores at the **true** `U`. Both arms get ground-truth side information of
  their own kind — the baseline the true supports, the full arm the true supports *and*
  the true within-skill order — so the gap is an **upper bound** on what the recurrent
  likelihood contributes once `U` must be inferred. A gap here is necessary for the full
  model to be worth its cost, not sufficient.
- Structure recovery is absent by construction, not by omission.

### 7.2 The gap tracks the predicted ambiguity

`expected_compatible_wrong_skills` is a closed form derived from the support combinatorics
alone — it never sees a chain. It says how many *wrong* skills can accommodate a block of
`d` distinct CPAs (`m = 10`, `A = 50`):

| `K` | `d = 2` | `d = 3` | `d = 4` | `d = 5` |
| --- | --- | --- | --- | --- |
| 3 | 0.0735 | 0.0122 | 0.0018 | 0.0002 |
| 10 | 0.3306 | 0.0551 | 0.0082 | 0.0011 |
| 30 | 1.0653 | 0.1776 | 0.0264 | 0.0034 |

It grows roughly as `K - 1`. The measured arms move the way that predicts: the full arm's
skill accuracy is flat to slightly rising across the ladder (0.9748, 0.9852, 0.9875) while
the baseline's falls away at `K = 30` (0.9426, 0.9449, 0.8875), so the gap widens.

Two cautions on reading it. The baseline is **not monotone** — `K = 3` sits 0.002 below
`K = 10`, a difference far inside single-chain noise, and no monotonicity is claimed here
or anywhere else. And this is a consistency check between a prediction and one chain per
rung, not a validation of either: it would have caught a baseline that improved with `K`,
which is the failure it was run to exclude.
