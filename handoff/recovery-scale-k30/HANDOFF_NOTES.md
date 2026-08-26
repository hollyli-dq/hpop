# Handoff notes — read before running the prompt

This package contains the code, not the history. Everything below was checked against the
source repository rather than assumed. **All four blockers are resolved and the Section 8 corpus generator is written.** What
remains are two preregistration criteria that cannot be met as written, and one design
issue that is not a blocker but will change how the result
must be read. Each resolution is recorded below with its verification, so the claim can
be checked rather than taken.

---

## Blocker 1 — RESOLVED: Section 1 amended for a code-only package

**No action needed. Recorded here so the amendment is auditable.**

The original Section 1 required:

    git cat-file -e 564995efd056d7d33984f0ca1532386e6140ea0c
    git merge-base --is-ancestor 564995efd056d7d33984f0ca1532386e6140ea0c HEAD

and said to stop if the commit was missing. This package ships code without the source
repository's history, so no such commit exists here.

That is deliberate and correct for this study: **no historical experiment is needed.** The
ladder generates its own sealed truths and corpora as its first step, and shipping any
historical corpus would risk truth leakage into a design that is truth-free until terminal
unsealing.

Shipping the history was also not practical. The source history carries a **176.5 MB
obsolete version** of `results/mcmc_original/stage6d1_joint_reference/qmc_replicates.npz`,
introduced at `ac44b0c` (2026-08-12), above GitHub's 100 MB hard limit. The live version of
that file is only 3.8 MB and is regenerable by `scripts/stage6d_joint_reference_build.py` —
but that is beside the point. `ac44b0c` is an **ancestor of `564995ef`**, so stripping the
blob rewrites 90 commits and changes `564995ef` itself, along with `07b474fe` and the
preregistration commit `8f99dd58`. Those three hashes are cited in 35 committed artifacts
(29 for `564995ef` alone), including freeze manifests. Rewriting them to satisfy a
transport limit trades the integrity of the record for a convenience.

**Section 1 is therefore amended in place**, marked inline in the prompt, to verify the
backend by content:

    python verify_environment.py     # must print RESULT: READY

`SOURCE_INTEGRITY.json` records that `src/hpop/mcmc_original/` and
`src/hpop/mcmc_optimized/` are byte-identical to the trees of `564995ef`, checked with
`git hash-object` against that commit at package build time: 199 files, zero drift. This is
a stronger claim than ancestry — ancestry says "that commit is reachable", content says
"these are exactly those bytes".

If the full history is ever wanted on the target machine anyway, `git bundle create
hpop.bundle --all` transferred by disk or scp carries it with every hash intact and no
GitHub limit involved. That option remains available and changes nothing here.

## Blocker 2 — RESOLVED: the CPA-vocabulary layer is implemented

`src/hpop/mcmc_cpa/` supplies what the registered model could not express. A skill `k` owns
an **injective** role map `ell_k : {0..m-1} -> {0..A-1}`; an observed CPA is translated to
that skill's role index before the recurrent arithmetic runs, and a block containing any
CPA outside the skill's support scores `-inf`, because that skill cannot have produced it.

`hpop.mcmc_original` is untouched. The translation happens before the sealed arithmetic.

| file | what it does |
| --- | --- |
| `role_maps.py` | injective per-skill maps, inverse lookup, deterministic sampling with pairwise-distinct supports, overlap matrix |
| `block_tables.py` | `CPABlockScoreTable`, the dense `(J, J+1, K)` builder with translation and the support mask |
| `nested_library.py` | the `K_max = 30` master library, its permutation, and the nested ladder prefixes |

### Why the likelihood needs no correction

`ell_k` is injective, so an observed CPA determines its role uniquely and the density over
CPAs is the density over roles — no Jacobian, no renormalisation. The `epsilon / m` uniform
component stays over `m` rather than `A`: it is a slip among the roles the skill *has*, not
among every action in the world.

### Verification

The load-bearing check is that with the identity map at `A = m` the new builder reduces
**bitwise** to the sealed `BlockScoreTable` — `max |diff| = 0.000e+00`, `array_equal` true.
Anything less would mean the layer had changed the arithmetic rather than only what feeds
it. Under non-identity maps every in-support score matches the sealed per-block scorer on
the relabelled trace to `7.1e-15`, and every out-of-support block is verified `-inf`.

`assert_matches_sealed_scorer` performs that comparison and should be run at the start of
the study, as its own parity gate, alongside the forward-recursion gate.

### What this changes about cost

With `m = 10` of `A = 50`, a block survives for a skill only if all its CPAs fall in that
skill's ten. On a corpus emitted from the supports the measured live fraction is about
**0.19** — roughly a fivefold reduction in the candidate set. Good for identifiability and
for speed, but it means **runtime measured under `A = m` does not transfer**. Recalibrate
per Section 11 rather than extrapolating from the scalability study.

## Blocker 3 — RESOLVED: the terminal gate reads K from the chain

`library_ids` took the number of skills as a literal `3`. That was the dangerous kind of
bug: `reshape(n, 3, width // 3)` *succeeds* whenever the width divides by three, and at
`K = 30, m = 10` the width is `30*10*9 = 2700`, which does. It would have produced a
confident, wrong library identifier rather than an error.

It now takes `n_skills` explicitly, read from the chain's own `u_draws` of shape
`(draws, K, m, d)` via `skills_in()`, and raises when the width is not divisible. Two
existing tests that relied on the implicit `3` were updated to pass it, and a new test
pins that there is no silent fallback.

`confirmatory_heldout_nll.py` had the same defect through its `N_SKILLS` / `N_ROLES`
imports and reads both from the chain now.

## Blocker 4 — RESOLVED: the nested master library

`nested_library.draw_master_library(replicate)` draws one admissible `K_max = 30` library
per replicate under Section 7's criteria 1–7, applies the registered permutation once, and
exposes each rung as a prefix. Nesting is verified across utilities, role maps and closures
together, and the rungs are checked to be strictly increasing sets.

Nesting is what stops library *size* being confounded with library *difficulty*: with
independent draws, a poor `K = 30` result could be thirty skills being hard or *those*
thirty being hard. The K=3 rung is literally three of the K=30 rung.

`pi` and `P` are deliberately **not** nested — they are drawn per rung from the registered
prior with their own seeds, because a `K = 30` transition matrix truncated to its first
three rows and columns is not a draw from the `K = 3` prior.

The consequence for reading the ladder: the rungs are not independent, so it is a
within-replicate comparison and the two replicates carry the between-truth variation. With
exactly two, report both points and their range, never a Gaussian interval through two
numbers.

### Still to do before launch

The corpus generator (Section 8) is not written: emitting traces from each skill's support
under `pi`/`P`, and the per-skill coverage bands. The pieces it needs — role maps, nested
library, the scoring layer — are all in place and tested.

## Not a blocker: the permutation-invariance you might expect to be a problem is already solved

`K = 30` means `30! ≈ 2.65e32` label permutations, so anything that enumerates them is
hopeless. It does not need to. `scripts/confirmatory_recovery.py:37` builds the canonical
library identifier as

    sha256(b"".join(sorted(precedence_from_u(u[k]).tobytes() for k in range(K))))

— a multiset hash over skills, invariant to relabelling by construction, `O(K log K)`.
`all_label_permutations` is exported in `__all__` but **called nowhere**. This metric goes
to `K = 30` unchanged. Do not rebuild it.

Section 17's *matching* (assign learned skills to true skills by closure Hamming distance)
is a separate thing and does need an assignment algorithm — use Hungarian / `scipy.optimize.
linear_sum_assignment`, `O(K^3)`, which is trivial at `K = 30`. Do not enumerate.

---

## Two preregistration criteria cannot be met as written — decide before launch

Both were found by running the generator, not by reading the prompt. Either would have
stopped the study on the target machine after it had been booked for a week. Neither is a
code defect, and neither is mine to fix: they are preregistration decisions.

### 1. Criterion 12 — the `pi` band is unreachable at large K

Section 7 asks every component of `pi` to lie in `[0.5/K, 1.5/K]`, drawn from the
registered flat Dirichlet, with an attempt cap of 100. Measured acceptance:

| K | P(pi in band) | P(stationary in band) | P(both) | expected attempts |
| --- | --- | --- | --- | --- |
| 3 | 0.178 | 0.781 | 0.137 | 7 |
| 5 | 0.020 | 0.548 | 0.011 | 95 |
| 10 | 0.0005 | 0.502 | not observed | ~2,000+ |
| 20 | ~0 | 0.665 | not observed | — |
| 30 | ~0 | 0.836 | not observed | — |

Criterion 11, the stationary band, is **not** the problem. Criterion 12 is, and it becomes
impossible somewhere between K = 5 and K = 10.

`draw_pi_p` fails loudly with the measured rate rather than working around it. Three
repairs, all preregistration changes:

* widen the band so the event stays reachable — simplest, but changes what "balanced" means;
* declare a concentrated Dirichlet as the generator — reachable, but the sampler assumes a
  flat prior, so generator and inference prior would no longer match, and that must be
  stated rather than absorbed;
* sample the constrained conditional directly — exactly what the prompt says, no prior
  change, but needs a sampler for the uniform distribution on a box-constrained simplex.

The library deliberately does **not** pick one. Substituting a concentrated Dirichlet
silently would change the generative prior invisibly, which is the worst of the three.

### 2. The per-role coverage band is checked in the wrong place

Section 8 asks that every one of a skill's ten roles appear at least five times in
training, and puts that check in the corpus loop with 100 resampling attempts. But role
frequency is set by `U`, not by the corpus draw. Under one fixed truth, across four corpus
seeds:

    skill 0, per-role counts:  [29, 53, 12,  4,  2, 222, 158, 41, 40, 32]
                               [39, 36,  4,  5,  1, 191, 130, 39, 39, 36]
                               [38, 48,  6,  2,  3, 201, 154, 39, 31, 39]
                               [37, 34,  5,  8,  2, 208, 134, 48, 30, 33]

The same roles starve every time. The first-step emission probabilities explain it:

    [0.002, 0.095, 0.002, 0.002, 0.002, 0.889, 0.002, 0.002, 0.002, 0.002]

a 445-fold spread, with the floor sitting exactly at `epsilon / m = 0.002`. Those roles fire
**only** through the noise term — the partial order suppresses them structurally.
**All 30 master skills** have at least one role below 0.01.

So resampling corpora cannot satisfy this band, and raising the attempt cap cannot either.
The criterion is a function of the truth, so it belongs in Section 7 master-truth
admissibility, where a rejected draw is replaced. `generate_ladder_corpus` fails with that
diagnosis attached rather than exhausting 100 futile attempts in silence.

Whether to move the criterion, weaken it, or accept starved roles as a property of the
generative model is a scientific decision. Note that accepting it has a consequence worth
stating in the paper: a role that never fires cannot be recovered, so per-skill recovery is
bounded above by role coverage before inference begins.

## Design issue: the structural-proposal budget falls as 1/K

You equalised evidence on the **data** side — `5K` traces of `J = 96` gives ≈480 training
occurrences per skill at every `K`, which is the core of the design and is right. The
**sampler** side is not equalised.

A structural sweep proposes one `(skill, row)` of `U`, chosen uniformly. `U` has `K × m_k`
= `K × 10` rows. At cadence 1/10, schedule A (50,000 sweeps) gives 5,000 structural attempts
per chain:

| K | U rows | structural attempts per row, per chain | schedule B |
|---|---|---|---|
| 3 | 30 | **167** | 133 |
| 5 | 50 | 100 | 80 |
| 10 | 100 | 50 | 40 |
| 20 | 200 | 25 | 20 |
| 30 | 300 | **16.7** | 13.3 |

**Each row gets ten times fewer proposals at K=30 than at K=3.** With acceptance near the
pilot target of 0.40, that is roughly 7 accepted moves per row over the entire K=30 run.

**This was confirmed empirically by the three-arm smoke, at a much smaller budget.** The
`learned-order` arm at 100 sweeps with one proposal per sweep gets 3.33 / 1.00 / 0.33
attempts per `U` row at `K = 3 / 10 / 30` — the same `1/K` collapse, two orders of
magnitude further down. Its recovery duly falls apart at `K = 30`, and its advantage over
the baseline changes sign between two random-stream realisations at `K = 3` and `K = 10`.
That is this design issue showing up as a measurement, and it is the reason
`SCALABILITY_GENERATION_SPEC.md` §7.1.2 marks the learned column unquotable rather than
reporting it as a finding about inference. Whatever is decided for the production
schedules below applies equally to the learned arm's `U` budget.

So "recovery degrades with K" and "each row received 10× less structural mixing effort" are
confounded, and the figures cannot separate them. This is exactly the confound the data-side
design was built to avoid, reappearing on the sampler side.

Three ways to handle it, in decreasing cost:

1. scale sweeps with `K` so attempts-per-row is constant — restores the comparison, but
   `K=30` becomes ten times more expensive and will not fit one week;
2. scale the cadence with `K` (more structural sweeps per FFBS sweep at large `K`) — cheaper,
   but changes the kernel across conditions, which needs registering;
3. accept it, and state in the preregistration that the ladder measures **recovery at a fixed
   compute budget**, not recovery at fixed mixing effort.

Option 3 is legitimate and may well be what you want — a fixed-budget result is what a
practitioner faces. But it must be *chosen and written down*, because the natural reading of
`fig_recovery_vs_K` is the other one. If the answer is 3, add the attempts-per-row column to
the figure or the caption so no reader can misattribute the trend.

---

## Smaller things

- **No environment spec existed** in the source repo. `requirements.txt` here is pinned from
  the source machine's manifests. The parity tolerance is 1e-10 and the observed discrepancy
  is ~1e-13; a different BLAS could move that, so run the gate first.
- **Section 2 asks for "the complete project suite"** — the shipped `tests/` collect 1594
  tests. Two files from the source repo were removed because they read
  `results/mcmc_original/**` and raise `FileNotFoundError` without those artifacts:
  `test_collapsed_u_expanded_audit.py` and `test_collapsed_u_fast_audit.py`. Treat the
  phrase as meaning the shipped suite, and see `TEST_BASELINE.md` for its known state.
- **Section 11's speed probe**: if the target machine has heterogeneous cores (Apple silicon,
  Intel P/E), note that a process migrating between core types keeps a whole core —
  `cpu/wall` stays at 1.0 — while running at roughly half speed. Neither wall time nor CPU
  time reveals it. `scripts/harness_reference/bench_common.py::speed_probe` is a fixed-work
  ruler that does; the source study needed it after a first pass produced larger
  configurations that ran *faster* than smaller ones.
- **`table_source`**: the confirmatory runs used `"batched"`, which rebuilds **all** `K`
  skill columns when a structural proposal moves one. `"fast"` rebuilds only the changed
  skill. On the source machine at `K=20` that was 33.2 s versus 0.30 s for a one-skill `U`
  change. **The numerics of that substitution were never verified** — `assert_sources_agree`
  exists for exactly this check. At `K=30` the potential saving is larger still, so it is
  worth an hour before committing to a week.

---

## The support-only baseline is now in the package

`src/hpop/mcmc_cpa/ladder_runner.py` runs all three conditions. It is **one** runner, and
the arm is a single argument — `SUPPORT_ONLY`, `ORACLE_ORDER` or `LEARNED_ORDER`. That is
deliberate: runners written to match each other drift, and a comparison whose arms differ
anywhere except what the candidate block score knows is not evidence about the block
score. Data, initialisation, RNG stream, segmentation prior, transition treatment and
sweep schedule are shared by construction. `tests/k_ladder/test_ladder_runner.py` pins
this down, including that zero sweeps gives the arms an identical initial state — though
see the common-random-numbers section below for why that check is nowhere near
sufficient on its own.

The baseline's score is `0` for a block every one of whose CPAs lies in the candidate
skill's support and `-inf` otherwise. A uniform-within-support emission would add
`-w log m` per block, which sums to the constant `-J log m` over every segmentation of a
trace and therefore cancels in every ratio the sampler forms — so `0` *is* the
uniform-within-support model, not an approximation to it.

**It reports `structure_recovery = "NOT APPLICABLE"`.** Its score never reads `U`, so a
`U` sampler run against it would draw from the prior with the data contributing nothing.
Putting that beside a data-informed posterior as if the two were comparable structure
estimates would be misleading, so the baseline declines to produce one. Compare the arms
on **segmentation and skill labelling only**.

### One optimisation, verified not to change anything

`ArmTables.refresh` memoises the full arm's candidate table on exactly the five inputs
`CPABlockScoreTable.refresh` reads (`u_by_skill`, `beta`, `omega`, `lambda_rep`,
`lambda_back`). With `U` held fixed the table is constant, and rebuilding it each sweep
was spending the dominant cost of a sweep to reproduce an array already in hand: `K=10`
went 58.5 s → 3.6 s, `K=30` from over ten minutes to 31.4 s. Three tests guard it — the
memoised and force-rebuilt tables are asserted bit-identical, the memo is asserted to miss
when any one of the five moves, and a full chain is asserted to produce identical draws
with the memo disabled. The pre-memo run's numbers were reproduced to the last digit.

This does not help the production ladder as much as it helps the smoke test, because there
`U` moves and the rebuild is real work. It matters wherever a chain holds `U` fixed.

### What the smoke run says, and what it does not

See `SCALABILITY_GENERATION_SPEC.md` §7.1 for the table, **re-run on the shared-Gamma
coupled corpora** — the coupling changes every corpus, so any pre-coupling number is void.
The oracle-order arm beats the baseline at `K = 3, 10, 30`. **The largest observed gap is
at `K = 30`** (+0.1565 boundary F1, +0.0924 skill accuracy); no trend in `K` is claimed.
At `K = 30` the baseline shows a substantial, approximately nine-percentage-point
degradation in skill accuracy against the oracle arm.

**Do not quote the `learned-order` column.** The smoke was run under two independent
common-random-number roots, identical otherwise. `oracle − support` moved by at most 0.017
and never changed sign; `learned − support` **changed sign at two of three rungs**. At 100
sweeps with one `U` proposal per sweep the arm gets 3.33 / 1.00 / 0.33 proposals per `U`
row at `K = 3 / 10 / 30`, so at `K = 30` it is reporting a barely-moved dispersed draw. It
needs a `U` burn-in budget scaled to `K·m` and a per-rung pilot for `u_scale` before it
means anything. It is also the concrete demonstration that `oracle − support` is not an
upper bound on `learned − support`: under a bound framing a negative realised gain would
be unsayable.

---

## Shared-Gamma coupling (adopted; the ladder is jointly conditioned)

One master `G_ij ~ iid Gamma(1,1)` off-diagonal per replicate, the master skill
permutation applied to **both** axes, and every rung cut out by restriction and
renormalisation:

    P^(K)_ii = 0,   P^(K)_ij = G_ij / sum_{h < K, h != i} G_ih,   pi^(K) = nu(P^(K))

Two properties are why this construction and not another. The Gamma representation of the
Dirichlet means each rung's rows are still exactly `Dirichlet_{K-1}(1,...,1)` before
conditioning, so the registered per-rung marginal is not traded away for the coupling. And
`P^(K)_ij / P^(K)_il = G_ij / G_il` for old `i, j, l` whatever `K` is, so growing the
ladder dilutes every old destination by one common normaliser and reorders nothing. Both
are tested, the second exhaustively over all old triples.

**The band is applied jointly.** All five rungs are built from one master `G` and the whole
draw is accepted or rejected together. Redrawing a single failing rung would replace its
`G` rows and destroy the coupling, so it is not done, and a test asserts every accepted
rung reduces to the one accepted `g`. State this wherever the ladder is described: **the
final ladder is jointly conditioned across rungs.** No rung is an unconditional
flat-Dirichlet draw; each is conditioned on *every* rung of the same master clearing the
band. That is stronger than per-rung acceptance and not interchangeable with it.

Measured on replicate 0: joint acceptance **0.20** (about five attempts), against per-rung
rates 0.772, 0.592, 0.498, 0.699, 0.865 whose product would be 0.138 — the rungs' band
events are positively correlated under a shared `G`. Both the rate and the attempt count
are recorded in every corpus's `coverage["transition_coupling"]`.

## Three arms, and what each contrast means

    support-only     the block score knows only which CPAs a skill can emit
    oracle-order     the full recurrent score at the TRUE U, held fixed
    learned-order    the full recurrent score, U inferred from a dispersed start

    oracle  - support     how much the AVAILABLE order information is worth
    learned - support     the REALISED end-to-end gain
    oracle  - learned     the INFERENCE gap

The first is a diagnostic about information, **not a bound**. The third is the one that
says whether to spend effort on inference rather than on the model.

`learned-order` uses the sealed row proposal, the sealed structural prior and the sealed
forward recursion; the only new part is `CPACollapsedULikelihood`, which computes the same
collapsed quantity over the CPA candidate table because the registered
`CollapsedULikelihood` builds `FastBlockScoreTable` and so assumes `A = m`.

## `table_source="fast"` is now settled, and it is exact

`CPABlockScoreTable.refresh_changed` rebuilds only the skills whose `U` actually moved.
`scripts/k_ladder/fast_exact_parity_gate.py` compares it against the all-skills rebuild at
`K = 3, 10, 20, 30` on finite masks, entrywise scores, one-skill `U`-update deltas, MH
log-ratios and accept/reject trajectories under a shared uniform. Exactness is the pass
rule; `max_abs_difference` is reported for context and is never the criterion.

**A trap worth knowing.** The candidate score reads `U` *only* through the induced
precedence relation — every quantity in the builder derives from
`all(u[:,None,:] > u[None,:,:], axis=2)` and none from the coordinates. So a small random
nudge often leaves a skill's whole score column bit-identical, and a parity check built
from such moves compares unchanged arrays and passes while testing nothing. The gate
counts how many proposals actually moved a column and refuses to return PASS on too few;
the mutation tests negate a skill's `U` rather than nudging it. A monotone rescale of `U`
is asserted to give bitwise-identical scores, which is the same property stated as a
model invariant.

## Common random numbers are addressed by index, not by stream position

Identical initial states are necessary and nowhere near sufficient. Two arms drawing from
one sequential `Generator` share only a *prefix*: the moment one accepts a move the other
rejects, or draws one more segment and consumes one more uniform, the streams slip and
every later comparison is confounded by different randomness as well as a different model.

`hpop.mcmc_cpa.crn.CommonRandomNumbers` derives every generator from
`(replicate, K, chain, sweep, move type, proposal index)` through `SeedSequence.spawn_key`,
so a generator depends only on where it sits in the design and never on how much
randomness an arm consumed earlier. Tests drive one arm's CRN hard, then check it still
hands out the same numbers as an untouched one at every index, and check the two arms
share the FFBS uniforms sweep by sweep *after* they have demonstrably diverged.

**The residual, stated rather than assumed.** `ffbs_segmentation_draw` loops over traces
against one generator and a trace's consumption depends on how many segments it draws, so
**within a single sweep** traces after the first can still misalign between arms. Fixing
that would mean editing the sealed backend. What the index scheme does guarantee is that
misalignment **cannot propagate across sweeps or across move types**, because each starts
from an index-derived stream. `crn_alignment_report` measures alignment rather than
assuming it.
