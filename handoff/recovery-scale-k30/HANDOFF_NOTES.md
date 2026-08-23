# Handoff notes — read before running the prompt

This package contains the code, not the history. Everything below was checked against the
source repository rather than assumed. **Three blockers** remain between this package and a run of
`prompt/PROMPT_RECOVERY_SCALE_K30.md`, plus one design issue that is not a blocker but
will change how the result must be read. A fourth — the Section 1 git check — is already
resolved by an amendment to the prompt, recorded below for audit.

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

## Blocker 2 — the model does not support A != m

This is the largest item and it is modelling work, not configuration.

Section 3 asks for `A = 50` with `role support size m_k = 10`, and Section 7 asks for
injective role maps with distinct CPA supports. **The shipped inference path cannot express
that.** The evidence, not inference:

- `recurrent_segmentation.py:135` — the scorer reads observed trace symbols **directly** as
  row indices into `u_by_skill[skill]`, which is `(K, m, d)`. There is no per-skill map.
- `matched_synthetic_generator.py:433` raises
  `AssertionError("the production scorer requires identity role maps")`.
- `matched_synthetic_generator.py:165,203` construct `role_maps` as the identity, always.
- The confirmatory corpus that this machinery produced used `A = m = 5`.

`role_maps` exists in the truth dataclass and is serialised into `truth_SEALED.json`, but it
is a placeholder: nothing consumes it at scoring time.

**What implementing it involves.** A CPA symbol in `0..A-1` must be translated through skill
`k`'s inverse role map before indexing `u_by_skill[k]`, and any candidate block containing a
CPA outside skill `k`'s support must score `-inf`. The clean way is a **new block-table
builder** that does the translation while filling the table — `mcmc_original` stays sealed
and untouched. The generator needs the matching change: each segment emits CPAs drawn from
its skill's support.

Two consequences to think about before committing to the design:

1. With `m_k = 10` of `A = 50` and random supports, most candidate blocks will be `-inf` for
   most skills. That is the mechanism that makes skills identifiable — it is the point of the
   design — but it changes the candidate geometry completely from anything measured so far,
   so the runtime projections in `RUNTIME_NOTES.md` do not transfer.
2. A new table builder needs its **own** parity gate against a reference implementation
   before it can be trusted. The existing gate covers the existing builder only.

---

## Blocker 3 — three hard-coded `3`s in the terminal gate

`scripts/confirmatory_terminal_gate.py`, `library_ids()`, lines 86–91:

    per_skill = width // 3
    blocks = relation_indicators.reshape(n, 3, per_skill)
    key = b"".join(sorted(np.packbits(blocks[i, k]).tobytes() for k in range(3)))

`K` must be read from the chain metadata. Small and mechanical, but the gate will silently
mis-reshape at any `K != 3` rather than fail loudly, so fix it before the first run.

`scripts/confirmatory_heldout_nll.py` similarly imports `N_SKILLS` / `N_ROLES` from the
frozen constants (lines 58, 102, 109, 288) and needs the same treatment.

---

## Blocker 4 — the nested master library is new code

Section 5 (one `K_max = 30` master library per replicate, one fixed permutation, nested
prefixes `K=3 ⊂ 5 ⊂ 10 ⊂ 20 ⊂ 30`) has no implementation here. Sections 7 and 8
(admissibility over 30 skills, per-skill coverage bands) are likewise new.

---

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
