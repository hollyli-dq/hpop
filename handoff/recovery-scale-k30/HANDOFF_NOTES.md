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

`src/hpop/mcmc_cpa/ladder_runner.py` runs both conditions. It is **one** runner, and the
arm is a single argument — `FULL_RFS` or `SUPPORT_ONLY`. That is deliberate: two runners
written to match each other drift, and a comparison whose arms differ anywhere except the
candidate block score is not evidence about the block score. Data, initialisation, RNG
stream, segmentation prior, transition treatment and sweep schedule are shared by
construction. `tests/k_ladder/test_ladder_runner.py` pins this down, including that zero
sweeps gives both arms the identical initial state.

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

See `SCALABILITY_GENERATION_SPEC.md` §7.1 for the table. The full likelihood beats the
baseline at `K = 3, 10, 30`, and its margin grows with `K`, which is the direction the
corrected ambiguity table predicts.

**The full arm scores at the true `U`.** The gap is therefore an upper bound on what the
recurrent likelihood contributes in the real setting where `U` is inferred — necessary for
the model to be worth its cost, not sufficient. One chain, one replicate, 100 sweeps: a
sanity check, not a result.

## Still open before production

**Gamma coupling.** Skills are nested across rungs; transition environments are not. Shared
`Gamma(1,1)` weights would pair them with the marginal unchanged. This is a design choice
and it has not been made. It should be settled before production replicates, because it
changes what "the same skill at a larger `K`" means.
