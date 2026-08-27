# Resume point — original latent-partial-order model

Branch `stage6e-unknown-boundaries` (worktree `/Users/dongqing/Desktop/hpop-stage6e`),
branched from `53914ff` = Stage 6D (`fd2d178`, tag `hpop-stage6d-oracle-joint-v1`) plus the
Part X walkthrough commit. Do **not** re-branch from `fd2d178`: that would drop Part X.

**Immediate next task: Step 7 — replace the local segmentation update with model-agnostic
semi-Markov FFBS and verify that it targets the same posterior.** FFBS is **not**
implemented in Stage 6E; Stage 6E deliberately uses the registered local move kernel
throughout, and the only forward recursion in the codebase
(`stage6e_exact.log_evidence_forward`) exists to cross-check `log Z` and to marginalise
`(S, z)` for held-out prediction. It is not a sampler and has no backward pass.

**The interpreter is `/Users/dongqing/Desktop/hpop/.venv/bin/python`.** Earlier notes said
`env/bin/python`; that path does not exist.

**Test counts differ between trees, and the difference is not a regression.** In the
*committed* tree: **1110 passed, 0 failures, 0 warnings, 12:05** — 978 through Stage 6C
plus 132 new Stage 6D tests across six files. (Stage 6C's own figure was 978: 860 before
it, plus 118 new.) The previously recorded **910** counts the *main* worktree, where
`tests/test_hpop.py`, `tests/test_recurrent.py` and `tests/test_semi_markov.py` are still
**untracked** — they are absent from any commit, so a fresh clone or worktree will never
see them. The old "1 pre-existing warning" is a `SyntaxWarning` raised while byte-compiling
`src/hpop/vendored/po_inference_agent/.../po_accelerator_nle_optimized.py`; it appears only
on a cold `__pycache__` and is not a test failure.

| commit | contents |
|---|---|
| *(this)* | Stage 6D — joint oracle-block sampler over `U, rho, beta, omega, lambda_rep, lambda_back` (tag `hpop-stage6d-oracle-joint-v1`) |
| `0c936b8` | Stage 6C — latent-U/rho/beta sampler, exact references, diagnostics, 4 scripts, 7 test files (tag `hpop-stage6c-latent-u-v1`) |
| `05ef2c2` | Stage 6C0 — model audit + corrected rho update (Stage 6C then incomplete) |
| `683225c` | Stage 6B2/6B3 — joint sampler, frozen model, joint references, 5 test files |
| `f20cdf6` | Stage 6B1 — scalar MH samplers, diagnostics, runner, 5 test files (tag `hpop-stage6b1-scalar-mcmc-v1`) |
| `3ca30eb` | Stage 6.0 / 6A — recurrent likelihood + generator |
| `05e0664` | Stage 6B0 — reference posteriors + vectorized omega replay |
| `7e76ed7` | Stages 0–5, vendored PO sampler, walkthrough notebook |

Read `notebooks/mcmc_original_walkthrough.ipynb` first — 66 cells, executed, it is the
single narrative of everything up to Stage 6B0.

## Stage 6B1 — DONE, all gates pass

`results/mcmc_original/stage6b1_full_seed0/` (4 chains, 20,000 iterations, 4,000 burn-in,
thin 2, dispersed starts). Every registered gate passes with one to two orders of
magnitude of margin, and the claim established is the strong one —
`p_MCMC(theta | D) ~= p_grid(theta | D)`, not merely truth-in-interval.

| parameter | grid mean | MCMC mean | std. mean err | KS | R-hat | bulk ESS |
|---|---:|---:|---:|---:|---:|---:|
| `beta` | 1.4961 | 1.4964 | 0.009 | 0.0118 | 1.0003 | 14165 |
| `omega` | 1.8506 | 1.8494 | 0.009 | 0.0074 | 1.0004 | 14498 |
| `kappa` | 0.8635 | 0.8634 | 0.005 | 0.0074 | 1.0004 | 14498 |
| `lambda_rep` | 0.8032 | 0.8034 | 0.006 | 0.0049 | 1.0002 | 13935 |
| `lambda_back` | 0.2288 | 0.2288 | 0.002 | 0.0082 | 1.0002 | 13739 |

Written: `src/hpop/mcmc_original/recurrent_scalar_mcmc.py`,
`src/hpop/mcmc_original/stage6b_mcmc_diagnostics.py`, `scripts/stage6b_scalar_mcmc.py`,
and `tests/mcmc_original/test_stage6b1_{scalar_mh,proposals,diagnostics,recurrent_target,
end_to_end}.py`. Smoke run in `stage6b1_smoke_seed0/`.

Registered in code, do not silently re-derive:

- **Starts** (`REGISTERED_STARTS`): the spec supplied two per parameter; four were needed,
  so each row was extended symmetrically keeping the original pair —
  `beta (0.5, 1.0, 2.5, 4.0)`, `omega (-0.5, 0.8, 3.5, 5.0)`,
  `lambda_rep (0.15, 0.5, 1.5, 3.0)`, `lambda_back (0.05, 0.20, 0.70, 1.50)`.
- **Proposal scales** come from the observed likelihood curvature at the true value —
  computed from the data by central differences, *never* from the reference grids. The
  2,000-iteration pilot then confirmed acceptance 0.42–0.48 and needed **no** adjustment
  for any parameter. Final scales: `beta 0.05109`, `omega 0.27891`, `lambda_rep 0.07086`,
  `lambda_back 0.21734`.
- **The acceptance gate is read on the post-burn-in rate.** Chains start dispersed, so the
  transient accepts nearly every downhill move and inflates the overall rate.
- R-hat and ESS are **rank-normalized and split** (Vehtari et al. 2021), implemented in
  `stage6b_mcmc_diagnostics.py`. The plain Gelman-Rubin in `diagnostics.py` is not the
  Stage 6B1 gate and is left alone where it is already used.

## Stage 6B2 and 6B3 — DONE, all gates pass

Full report: `results/mcmc_original/stage6b_complete/report.md`.

**6B2** (`beta, omega, lambda_rep`; `lambda_back` fixed at 0.25) — 32/32 gates.
**6B3** (all four jointly) — 42/42 gates. Both compared against **independent joint
references** built by direct quadrature, never against the Stage 6B1 1-D grids, which are
conditional and cannot validate joint sampling.

| stage | reference | grid points | integral | outer-face | refinement drift | energy distance (envelope) | max corr error (envelope) |
|---|---|---:|---:|---:|---:|---|---|
| 6B2 | 3-D tensor grid | 226,981 | 1.0000000000 | 1.19e-07 | 0.0216 sd | 0.002126 (0.004948), z +0.00 | 0.00839 (0.04152) |
| 6B3 | 4-D tensor grid | 6,765,201 | 1.0000000000 | 2.78e-09 | 0.1025 sd | 0.003142 (0.003990), z +0.94 | 0.01539 (0.05623) |

Worst standardized mean error 0.0275, worst interval endpoint 0.0921, all R-hat <= 1.0007,
bulk ESS >= 7413. Every generating value is inside its 95% interval, but that is reported
**separately** — the sampler-correctness claim is the reference comparison.

Registered facts a fresh session must not change:

- **The joint sampler composes the Stage 6B1 kernels, it does not reimplement them.** Each
  coordinate update calls `scalar_mh_step` with `build_proposal`. Coordinate acceptance-ratio
  parity against Stage 6B1 is **<= 7.3e-12** and is enforced by test.
- **Proposal scales are the frozen Stage 6B1 ones**, used unchanged. No pilot was run for
  6B2 or 6B3, so nothing could be tuned against the immutable references. Acceptance still
  landed at 0.43-0.48 for every coordinate.
- **Starts are prior quantiles** at levels (0.10, 0.35, 0.65, 0.90) arranged by a fixed
  Latin square, so all four chains differ in *every* coordinate.
- The omega state cache is keyed on the omega it was built at and is written **only** by an
  explicit `refresh_cache`. An evaluation never writes it, so a rejected proposal cannot
  leave anything behind; a changed omega falls through to full replay.
- The correlation envelope is calibrated at the chains' **minimum bulk ESS**, not their raw
  draw count — an MCMC correlation estimate carries the noise of its effective sample size.
- Frozen model hash `9ad850f22065d85f6cfd855443395d06b8a566e8c8dcdd4f9d85b1f031e911cc`,
  model id `recurrent-rfs-utility-weighted-frontier-v1`. `stage6b_frozen.py` pins the
  likelihood branch with hand-computed values so a silent switch to a uniform-frontier
  variant fails a test rather than changing a posterior.

## Stage 6C — DONE, all gates pass (tag `hpop-stage6c-latent-u-v1`)

`U` and `rho` are inferred in 6C1; `beta` joins them in 6C2. **The target is continuous
in `U`** — the chain's state is the real matrix `U ∈ R^{5×2}`, never a poset id, and
`h(U)` (the coordinate-wise dominance order) is a *derived label* used for reporting and
for the reference comparison only. Read
`results/mcmc_original/stage6c_complete/model_audit.md` before touching Stage 6D: it
settles the model and overrides any brief that assumes a discrete-poset state.

| gate | 6C1 | 6C2 | threshold |
|---|---:|---:|---:|
| full-U total variation | 1.22e-118 | 3.63e-115 | < 0.01 |
| max relation-marginal error | 2.45e-118 | 7.25e-115 | < 0.01 |
| `rho` R-hat | 1.0066 | 1.0029 | ≤ 1.01 |
| mixed energy distance vs envelope | 0.0013 / 0.0030 | 0.0026 / 0.0034 | inside |
| `beta` R-hat | — | 1.000004 | ≤ 1.01 |
| `beta` KS vs reference | — | 0.0112 | < 0.05 |

Verdicts, kept separate on purpose:

    Stage 6C sampler correctness : PASS  (both stages)
    Stage 6C U recovery          : PASS  (closure and reduction F1 = 1.0, Hamming 0)
    Stage 6C rho recovery        : NOT APPLICABLE
    Stage 6C beta recovery       : PASS  (1.4965 ± 0.0317, truth 1.5 inside 95%)

- **`rho` recovery is NOT APPLICABLE, permanently.** `U_TRUE` is hand specified in
  `recurrent_synthetic.py`, not drawn from `p(U | rho)`. No `RHO_TRUE` exists. Do not
  generate a more favourable dataset to manufacture one.
- **`rho` is weakly identified even though the sampler is exactly right.** The poset
  posterior is a point mass and `rho` never enters the likelihood, so
  `p(rho | Y) ∝ p(rho) · pi_rho(P_true)` — the entire signal is how one poset's prior
  cell mass varies with `rho`, on 5 rows of a 2-D Gaussian. Reference `rho` 0.3253 ±
  0.2198; MCMC 0.3132 (6C1) and 0.3231 (6C2). See `figures/rho_identifiability.png`.
- **No U/beta confounding.** Freeing `beta` changes nothing structural (probability 1.0
  on the true order in both stages, one order visited), and the 6C2 `beta` posterior
  1.4965 ± 0.0317 matches the Stage 6B1 posterior 1.4961 ± 0.0319 obtained with `U`
  fixed at the truth. The true order beats its nearest competitor by **271.5 nats**.
- **Relation-count R-hat/ESS are `null`, not 1.0.** The trace is constant at 6, so they
  are undefined; the diagnostics say `degenerate` rather than reporting a flattering
  number. Correlations involving it are `null` with the reason attached.
- **The §2.1 normalisation gate passed for a different reason than the brief expects.**
  There is no combinatorial `Z_m(rho)`: the prior is a density on `R^{m×d}`, and the
  rho-dependent normaliser is the Gaussian `-(m/2) log|Sigma_rho|`, already present in
  `sampler_u.log_u_prior` (matches scipy to 3.6e-15; single-row quadrature mass 1.0 to
  1e-14). Deleting it shifts the `rho` posterior mean measurably — that negative control
  is a test, not a comment.
- Reference is exact by enumeration: all **4231** labelled posets on 5 elements from
  14,400 ranking tuples. The single Monte Carlo ingredient is `pi_rho(P)`: 40,000,000
  prior draws, common random numbers across the rho grid, max SE **1.44e-05**.

Written: `stage6c_frozen.py`, `recurrent_latent_poset_mcmc.py`,
`stage6c_exact_reference.py`, `stage6c_diagnostics.py`, four `scripts/stage6c_*.py`, and
seven `tests/mcmc_original/test_stage6c_*.py` (118 tests).

**Known shortfalls, recorded not papered over** (detail in the complete report): the
registered protocol retains 12,000 pooled draws against the spec's 100,000 request, and
no `rho` KS threshold was pre-registered (observed 0.0249 / 0.0124, reported against the
distribution-free 1.36/√ESS ≈ 0.042 rather than a fitted gate).

## Stage 6D — DONE, all gates pass (tag `hpop-stage6d-oracle-joint-v1`)

Full report: `results/mcmc_original/stage6d_complete/report.md`. Read
`stage6d_complete/model_audit.md` **first** — it settles the model and overrides any brief
that assumes an assessor hierarchy, a `tau`, or a discrete-poset state.

All six coordinates are inferred jointly on oracle boundaries and oracle skill labels.
Three verdicts are kept apart on purpose:

    Stage 6D0 smoke and kernel parity   PASS
    Stage 6D1 sampler correctness       PASS  (11/11 gates against an independent reference)
    Stage 6D2 convergence               PASS  (24/24 registered gates)
    Stage 6D2 structural (U) recovery   PASS  (closure and reduction F1 = 1.0, Hamming 0)
    Stage 6D2 scalar recovery           PASS  (all four truths inside 95%)
    Stage 6D rho recovery               NOT APPLICABLE — no generating value exists
    Stage 6D entrywise U recovery       NOT CLAIMED — the target is invariant

**Stage 6D2** (`stage6d2_oracle_joint_full_seed0`, 4 chains x 30,000 sweeps, 10,000
burn-in, thin 5, 16,000 pooled draws, 20.7 min). Every gate passed at the initial sweep
count; the 20,000-sweep continuation blocks and the 100,000 ceiling were never needed.

| coordinate | posterior | acceptance | R-hat | bulk ESS | vs Stage 6B3 (`U` at truth) |
|---|---|---:|---:|---:|---:|
| `rho` | 0.3186 ± 0.2177 | 0.328 | 1.00091 | 4874 | −0.0200 sd (vs 6C2) |
| `beta` | 1.4834 ± 0.0393 | 0.444 | 1.00069 | 8344 | +0.0613 sd |
| `omega` | 1.8851 ± 0.1292 | 0.386 | 1.00024 | 11822 | −0.0171 sd |
| `lambda_rep` | 0.8075 ± 0.0285 | 0.389 | 1.00090 | 8365 | +0.0562 sd |
| `lambda_back` | 0.2191 ± 0.0240 | 0.519 | 1.00023 | 10122 | +0.0105 sd |

Registered facts a fresh session must not re-derive or change:

- **Proposal scales are a property of the CORPUS, not of the kernel.** This is the
  headline lesson of Stage 6D and it cost three attempts to learn. On the 3-block Stage
  6D1 reference model all four registered Stage 6B scalar scales were **16–32x too
  small** (acceptance 0.96–0.98, bulk ESS 349–1143); correcting them lifted ESS **22–61x**
  and collapsed every R-hat to ~1.000. On the 500-block Stage 6D2 corpus the *same*
  registered scales are right to within x1–x2 — because that is the corpus they were
  tuned on. **Never carry a scale between corpora; run a separate pilot.**
- **Frozen Stage 6D2 scales** (`stage6d2_pilot/selected_scales.json`):
  `U 0.5` (x1) · `rho 4.0` (**x8**) · `beta 0.05109` (x1) · `omega 0.36426` (x2) ·
  `lambda_rep 0.0831` (x2) · `lambda_back 0.18944` (x2).
- **`rho` was the one coordinate no stage had ever tuned.** Stage 6C inherited its scale
  and Stage 6D1 froze it by instruction, and it was the weakest-mixing coordinate in
  both. At x8 its acceptance falls 0.857 → 0.329 and bulk ESS rises from 897 (6C2) to
  **4874**.
- **The induced-order posterior on this corpus is a point mass**, so the relation-count
  trace is constant at 6 and is reported as `degenerate` — never as an R-hat of 1.0. That
  point mass is gated by structural recovery instead.
- **No confounding.** Because the likelihood reads `U` only through `h(U)` and `h(U)` is
  that point mass, the likelihood the scalars see is *identical* to the one Stage 6B3 saw
  with `U` pinned at `U_TRUE`. All four marginals agree to ≤ 0.062 parent sd, `rho`
  agrees with Stage 6C2 to 0.020 sd, and the induced order is unchanged by freeing
  `omega` — no `U`-`beta`, `H`-`omega` or `rho`-`U` confounding.
- **`beta` is deliberately NOT gated against Stage 6C2.** Stage 6C2 fixes `omega` and the
  two lambdas while 6B3 and 6D2 marginalise over them, so the two parents already
  disagree with each other by 0.40 Stage 6B3 sd. Gating against both was impossible; the
  contrast is reported instead, and this was recorded before any Stage 6D2 draw existed.
- **The entrywise posterior mean of `U` is not a valid plug-in**, and the report carries
  the negative control: it scores −2.029 per held-out step against the posterior
  predictive's −1.2433, because averaging inside the order cell collapses the
  incomparabilities — `h(mean U)` has 10 relations where every draw has 6.
- Held-out (§17, **reported, never gated**): posterior predictive −1.24331 per step
  against −1.24316 at the generating truth, on 200 blocks / 4,000 steps.
- **All failed attempts are preserved and never relabelled**:
  `stage6d1_joint_mcmc_FAILED_attempt0_50k`, `..._FAILED_attempt1`,
  `..._FAILED_attempt2_omega_retuned_REEXECUTED`. The third is an explicitly labelled
  **re-execution**: the original omega-retuned run's directory was overwritten and its
  base seed was never recorded, so neither candidate seed reproduces its exact numbers.
  The values recorded at the time are cited alongside it.
- The Stage 6D1 reference's `max_replicate_h_total_variation` **still fails** its old
  threshold and is still reported as a failure of a **retired** statistic. It was
  superseded by `rqmc_se = sd/sqrt(R)` before any MCMC comparison existed, because a
  maximum over replicates estimates the dispersion of a *single* replicate and does not
  shrink with `R`. Do not relabel its failures as passes.

Written: `stage6d_frozen.py`, `recurrent_oracle_joint_mcmc.py`,
`stage6d_joint_reference.py`, `stage6d_diagnostics.py`, seven `scripts/stage6d*.py`, and
six `tests/mcmc_original/test_stage6d_*.py`.

## Stage 6E — unknown boundaries

Stage 6E frees the two oracles Stage 6D still had: the block boundaries `S` and the skill
labels `z`. `K = 3` reusable skills over `m = 5` roles, `d = 2`.

### The four audit findings that determined the stage

Read `results/mcmc_original/stage6e_complete/model_audit.md` before anything else.

1. **`pi` and `P` are INFERRED**, through the frozen Stage 3 conjugate updates. They belong
   to the registered target; Stage 6D omitted them only because `K = 1` left no path prior.
2. **`P` forbids self-transitions** (`allowed_next` excludes `h`; the sampled diagonal is
   exactly 0), so adjacent segments must carry different skills. Load-bearing: without it a
   segment could be cut anywhere at no cost in the path prior. It also makes the
   occurrence-label array an *exact* encoding of the segmentation — a cut is precisely a
   label change — which is how the 100-trace chains are stored.
3. **There is no terminal transition.** Do not add one.
4. **The Stage 5 legality predicate cannot represent the recurrent state space.**
   `compatible_skills` requires the block's multiset to equal the skill's role multiset (a
   permutation visiting each role once); the recurrent model is defined by repeats. Under
   it every neighbourhood is empty. `Stage6EMoveKernel` therefore overrides `neighbours`
   **alone** — `proposal_distribution`, `proposal_prob`, `sample_proposal` and
   `mh_local_step` are the Stage 5 objects, inherited.

### What was built

| module | role |
|---|---|
| `stage6e_frozen.py` | frozen config, `delta_B`, widths, `eta`, prior pieces |
| `recurrent_segmentation.py` | block scorer, `Stage6EMoveKernel`, `log_target_stage6e` |
| `stage6e_state.py` | `Stage6EState` / `Stage6EModel`, counts |
| `fast_segmentation_kernel.py` | the same proposal law over `((end, skill), ...)` keys |
| `stage6e_block_table.py` | every candidate block score for the current parameters, batched |
| `stage6e_sampler.py` | the production sweep, `SkillBlockLikelihood`, chain runner |
| `stage6e_exact.py` | enumeration + the independent forward `log Z` recursion |
| `stage6e_mixed_reference.py` | the streamed QMC + enumeration mixed reference |
| `stage6e_corpus.py` | the frozen trace-level corpus |
| `stage6e_diagnostics.py` | boundary / label / structure recovery, held-out prediction |

### Results

* **6E0** — parity and smoke, all PASS. Oracle parity with Stage 6D is the strong form:
  `6E - 6D` is one explicitly computed constant at every theta, worst residual 7.1e-14.
  Kernel parity 1.8e-15 over 18 comparisons, all with a non-zero Hastings term.
* **6E1A** — exact segmentation-only reference, **13/13 gates PASS**. 21 enumerated states;
  `log Z` by enumeration and by an independent forward recursion agree to **8.9e-16**;
  380,000 retained draws; TV(path) **1.47e-3**, boundary **2.8e-4**, occurrence-label
  **1.07e-3**.
* **6E1B** — mixed reference, 16 scrambles x 2^20 = 16.8M draws. Primary quality gates
  PASS (`rqmc_se` 5.67e-4 <= 1e-3; half-width 1.21e-3 <= 2.5e-3). The two
  maximum-over-replicate statistics FAIL and are reported as failures — they are
  **superseded descriptive diagnostics**, superseded in Stage 6D1 because a maximum over
  replicates estimates the dispersion of a single replicate and does not shrink with `R`.
  Comparison: **18/18 gates PASS** at 4 chains x 600,000 sweeps. TV(H) 0.00531, TV(S, z)
  0.00461, relation 0.00388, boundary 0.00152, label 0.00236, segment-count 0.00091, the
  mixed multivariate energy statistic 0.003989 against a 0.004523 envelope, and every
  R-hat <= 1.00256.

  **One failed attempt is preserved, not deleted.** Attempt 0 (150,000 sweeps) cleared
  everything except `induced_h_total_variation`, at 0.01050 against 0.01. Diagnosis:
  bulk ESS 301 on `lambda_rep`, while five posterior means sat within 0.024 reference SD
  and every R-hat within 1.0042 — Monte Carlo error, not bias. The response was **more
  draws and nothing else**; the gate was not widened. With 4x the draws all five
  distributional statistics fell by ~`sqrt(4) = 2`, which a bias could not do. See
  `stage6e1b_mixed_reference_FAILED_attempt0_150k/README.md`.
* **6E2** — corpus frozen (100 train traces / 510 blocks, 45 held-out / 223 blocks, `J`
  mean 32, widths 3-12 drawn from the registered boundary prior, all three skills reused
  within traces, upstream repeats and recomputations present). Leakage audit PASS.
  Pilot and discarded joint confirmation PASS (8/8).

  **Formal chains: RUNNING at the time of writing, not finished.** 4 chains x 50,000
  sweeps, 15,000 burn-in, thin 5, plus the like-for-like oracle-boundary control. Nothing
  in this repository claims a Stage 6E2 convergence, recovery or held-out result, and no
  Stage 6E tag exists. To finish:

  ```
  PYTHONPATH=src python scripts/stage6e2_formal_chains.py --run unknown   # if not done
  PYTHONPATH=src python scripts/stage6e2_formal_chains.py --run oracle    # if not done
  PYTHONPATH=src python scripts/stage6e2_analyse.py
  PYTHONPATH=src python scripts/stage6e_complete_report.py
  ```

  Chains checkpoint every 10% to `{unknown,oracle}_checkpoints/` and resume
  bit-identically from the saved state and RNG. §20 gates 1-11 and 16 (sampler
  correctness) all PASS; 12-14 are the outstanding ones.

### Three traps this stage paid for, in code

1. **The neighbour cache had to be bounded.** One cached key costs ~40 KB across the four
   move types at the registered widths, and the sampler holds one kernel per trace — 100 of
   them. At the original 200,000-entry cap a 50,000-sweep run would have needed **209 GB**.
   It is capped at 256, which is ~32 proposals of reuse for ~2.6 MB per kernel.
2. **`min(0.0, NaN)` is `0.0` in Python.** Every comparison with NaN is False, so `min`
   keeps its first argument. At a large enough `lambda_back` the registered likelihood
   underflows to NaN (`weights.sum()` is exactly 0), and `scalar_mh_step` would then
   **accept** it and carry a non-finite target forever. `sweep_once` maps a non-finite
   candidate posterior to `-inf` before it reaches the step.
3. **The block table is a snapshot, and the sweep moves past it.** It is refreshed at the
   start of a sweep for the segmentation phase; the `U` and scalar updates that follow make
   it stale by design. Comparing it against the scorer at the *end* of a sweep compares two
   different models — that produced a spurious parity failure of 13.6 before the check was
   put at the parameters the table was actually built at.

### The Stage 6E2 pilot, and AMENDMENT 1

The Stage 6D2 scales are the pilot's centre, never its conclusion. The first pass over the
registered grid `[0.25, 0.5, 1, 2, 4, 8]` put **all four** scalars on the upper boundary,
and `lambda_rep`/`lambda_back` were still inadmissible there (0.653 / 0.661 against a 0.60
ceiling). That is evidence of a truncated search **range**, not of a failed rule, so before
any formal draw existed the grid was extended to `[..., 16, 32, 64]` with the band, the
statistic, the tie-break and the ESJD coordinates all unchanged, and the original rows
preserved verbatim in their own pass.

Outcome: `beta` and `omega` have genuine **interior** optima at x8 (x16 is worse — beta's
acceptance collapses to 0.072), so the boundary selection was not an artefact;
`lambda_rep` and `lambda_back` gained admissible candidates at x16.

```
beta 0.40872 (x8)   omega 1.45704 (x8)   lambda_rep 0.66480 (x16)
lambda_back 1.51552 (x16)   U 0.5   rho 0.5   32 segmentation proposals per trace per sweep
```

`U` (0.337) and `rho` (0.864) sit inside the registered pathology band, so their frozen
Stage 6D definitions are kept unchanged. Joint confirmation 8/8 PASS. All pilot draws
discarded.

**Measured cost, not guessed.** ~580 ms per sweep single-process at 32 proposals per trace
on the 100-trace corpus, so the registered §14 configuration (4 chains x 50,000 sweeps) is
an overnight run. The chains checkpoint every 10% and resume bit-identically from the saved
state and RNG.

## Registered facts a fresh session must not re-derive or change

- Priors, fixed before any posterior was inspected: `Gamma(shape 2, rate 2)` for
  `beta`/`lambda_rep`/`lambda_back`; `Normal(0, 2^2)` **directly on omega**, not on kappa.
- Truths: `beta 1.5`, `omega logit(0.85) = 1.7346`, `lambda_rep 0.8`, `lambda_back 0.25`,
  `epsilon 0.02` (fixed), `U_TRUE` 5 roles.
- Reference posteriors (all contain the truth at 95%):
  `beta` 1.4961 ± 0.0319 · `omega` 1.8506 ± 0.1245 · `kappa` 0.8636, 95% [0.8343, 0.8911] ·
  `lambda_rep` 0.8032 ± 0.0236 · `lambda_back` 0.2288 ± 0.0224.
- The feature cache is exact but valid **only** for the three parameters that do not enter
  the `q` recursion. Never reuse a `q` trajectory across omega values. This is now pinned
  by `test_a_fixed_q_shortcut_for_omega_is_wrong`, not just by a comment.

## Open items, none blocking Stage 6B1

1. **Skill E's `(0,3)`** is unrecovered under joint beta inference. Diagnosed, not fixed:
   the candidate-structure ranking flips between beta 2.0 and 2.5, and inferred chains roam
   beta in [0.59, 4.80]. Pinning beta at truth gives P(0>3) = 0.981; inferring it gives
   0.438. Identified *conditional on beta*; marginalising over beta dissolves it.
2. **The unseen linear-extension experiment** is still skipped, and has now slipped past
   three stages. It sits ahead of recurrence in the original roadmap, is the result that
   would show a partial order is not a bigram, and depends only on already-validated
   machinery. Decide it before starting 6B2, or it will keep sliding.
3. **Stage 5C** (joint S+U+P) remains **DEFERRED** — never represent it as passed.
4. Stage 5B's registered ARI gate is unmet on the full split (0.667 vs 0.85). This is *not*
   a sampler defect: conditional on correct boundaries the sampler sits exactly on the
   oracle ceiling. The A/D pair is support-matched by design, which caps the ceiling at
   ~0.89, and 5 of 20 traces have boundary errors.

## Traps this project has already fallen into

- **A proposal scale tuned on the unknown-boundary target is wrong for the
  oracle-boundary control, and the gap is large.** The Stage 6E2 pilot measured `beta`
  acceptance 0.597 at scale 0.40872; the like-for-like control, which differs *only* in
  pinning `(S, z)` at the truth, ran the same scale at **0.085**. Nothing is broken —
  oracle labels are information, the scalar posterior is correspondingly sharper, and a
  fixed step size then accepts less often. But it is a sharper illustration of "scales are
  a property of the target" than Stage 6D's, because here the two targets share a corpus,
  a kernel and a parameterisation and differ in one thing only. The control still
  converged (R-hat <= 1.00241, bulk ESS 2453-18378) and recovered `beta` 1.5123,
  `lambda_rep` 0.7875, `lambda_back` 0.2474 against truths 1.5, 0.8, 0.25 — so this cost
  efficiency, not validity. Any future control of this kind should get its own pilot.

- **A zero-ablation NLL gap is not an identifiability measure.** It depends on both the
  curvature *and* the distance from the truth to zero. This produced a wrong warning about
  `lambda_back` that took two corrections to undo.
- **Curvature for omega must be block-level.** `kappa = sigmoid(omega)` enters the state
  recursion, so a per-step curvature at fixed `q` misses the principal channel and
  understates omega by ~25x.
- **Information is not reparameterisation-invariant.** Never rank parameters by raw Fisher
  or curvature values; report omega on both the omega and kappa scales.
- **The exposure audit counts opportunities, not effect size.** Passing it does not predict
  recoverability.
- `pytest -q` from the repo root collects 8 pre-existing `archive/tests/` errors. Use
  `tests/mcmc_original` or `tests`, never bare.
