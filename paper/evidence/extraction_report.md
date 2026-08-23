# Evidence extraction report

**Task:** build a paper-writing evidence pack for the HPOP ICLR submission from
frozen, committed artifacts only.
**Date:** 2026-08-17.
**Repository HEAD at extraction:** `50eee50` on branch `condition-c-integration`.
**Mode:** read-only extraction and writing. No experiment was launched, stopped,
resumed, or inspected. No inference kernel was modified. No main-paper file was
touched (none exists in this repository — see §6).

---

## 1. What was read, and what was deliberately not

### Read (committed, frozen)

| Area | Location | Commit |
|---|---|---|
| Condition A | `results/mcmc_original/matched_condition_a/` (19 files) | `b199374` |
| Condition B | `results/mcmc_original/matched_condition_b/` (25 files) | `34873d8` |
| Matched generator validation | `results/mcmc_original/matched_generator_validation/` (12 files) | `8ca8281` |
| Formal corpus | `results/mcmc_original/matched_synthetic_formal_corpus/` | `b199374` |
| Smoke corpus | `results/mcmc_original/matched_generator_smoke_corpus/` | `8ca8281` |
| Collapsed-U run 1 (FAIL) | `results/mcmc_original/collapsed_u_kernel_validation/` | `58f005e` |
| Collapsed-U rep2 (FAIL) | `results/mcmc_original/collapsed_u_kernel_validation_rep2/` | `58f005e` |
| Dependence calibration | `results/mcmc_original/collapsed_u_dependence_calibration/` | `58f005e` |
| start[0] probe | `results/mcmc_original/collapsed_u_start0_probe/` | `738fe00` |
| Collapsed-U final validation (VALIDATED) | `results/mcmc_original/collapsed_u_efficient_final_validation/` | `58f005e` |
| C0 / C1 proposal audits | `collapsed_u_fast_audit/`, `collapsed_u_expanded_audit/` | `58f005e` |
| Step 7B2 FFBS-only (negative) | `results/mcmc_original/stage7b2_full_joint_ffbs/` | `061e1f6` |
| TaskBench audit + benchmark | `results/external/taskbench_multimedia_*/` | `e6afb17` |
| TaskBench static fit | `results/external/taskbench_static_poset/` | `10d4bf4` |
| τ³ audit + pilot dataset | `results/external/tau3_retail_*/` | `8100fc2` |
| Condition C **design only** | `matched_condition_c/prelaunch_report.md`, `prelaunch_registration.json`, `small_reference_equality.json` | `a136654`, `50eee50` |

TaskBench and τ³ artifacts were read from their own worktrees
(`/Users/dongqing/Desktop/hpop-taskbench` @ `10d4bf4`,
`/Users/dongqing/Desktop/hpop-tau3` @ `8100fc2`), both verified clean under
`results/`.

### Deliberately not read

- `results/mcmc_original/matched_condition_c/formal_chains/` and
  `formal_registration.json` — **untracked**, produced by a live run.
- Any file under `/Users/dongqing/Desktop/hpop-smoke` beyond `git ls-files`
  (the τ³ smoke fit is live).
- `results/external/tau3_retail_hpop_pilot/test_trajectories.jsonl` — sealed
  test split.
- Live checkpoints, terminal summaries, conversation memory, and every
  uncommitted result directory.

Two processes were confirmed running by a read-only `ps` at the start of
extraction and then left alone: `scripts/run_matched_condition_c_formal.py`
(PID 51950, 8 workers, started 12:04) and `scripts/tau3_smoke_fit.py`
(PID 58240, 4 workers, started 20:49).

At the end of extraction, a second read-only `ps` showed PID 51950 still
running (9 h 21 m elapsed) and PID 58240 no longer present. **Nothing was done
about that and nothing was read as a result of it.** A vanished process is not
a result: it may have completed, been interrupted, or been restarted, and the
smoke-fit branch still contains no committed artifact. Claim C10 remains
`pending` until a frozen artifact is committed, which is the only evidence this
pack will accept.

---

## 2. Completed experiments discovered

| # | Experiment | Frozen classification | Commit |
|---|---|---|---|
| 1 | Matched synthetic generator validation | ALL 15 GATES PASS | `8ca8281` |
| 2 | Matched generator smoke corpus | frozen, parity verified | `8ca8281` |
| 3 | Matched synthetic formal corpus | frozen, §4 validation all PASS | `b199374` |
| 4 | Collapsed-U kernel validation, run 1 | **NOT YET VALIDATED — STOP** (17/18 gates PASS) | `58f005e` |
| 5 | Collapsed-U validation, rep2 (independent seeds) | **FAIL**, same single gate | `58f005e` |
| 6 | Collapsed-U dependence calibration | ESTIMATOR ARTIFACT / SERIAL DEPENDENCE SUPPORTED | `58f005e` |
| 7 | Collapsed-U start[0] focused probe | START-0 BASIN-SPECIFIC KERNEL INTERACTION SUPPORTED | `738fe00` |
| 8 | Collapsed-U sequential final validation | **COLLAPSED-U KERNEL VALIDATED** | `58f005e` |
| 9 | Collapsed-U C0 fast audit | MECHANISM VIABLE — EXPAND AUDIT | `58f005e` |
| 10 | Collapsed-U C1 expanded audit | barrier reduction quantified | `58f005e` |
| 11 | Step 7B2 FFBS full-joint chains | structural locking NOT resolved (negative) | `061e1f6` |
| 12 | Stage 6E2 unknown-boundary joint | **TERMINAL FAIL / MULTIMODAL** at registered 150k | `6eb43b3` |
| 13 | **Condition A** | **PATH STRONGLY IDENTIFIABLE** | `b199374` |
| 14 | **Condition B** | **STRUCTURE STRONGLY IDENTIFIABLE UNDER ORACLE PATHS** | `34873d8` |
| 15 | TaskBench benchmark construction | 18/18 validation gates PASS | `e6afb17` |
| 16 | **TaskBench static partial-order fit** | **EXTERNAL PARTIAL-ORDER GENERALIZATION SUPPORTED; BAYESIAN ADVANTAGE IS UNCERTAINTY AND PREDICTION** | `10d4bf4` |
| 17 | τ³-Retail source audit | VERSION MATCH VERIFIED | `8100fc2` |
| 18 | **τ³-Retail pilot dataset** | 24/24 construction gates PASS | `8100fc2` |

Items 4, 5, 11 and 12 are **failures or negative results** and are recorded as
such throughout the pack. None has been restated as a pass.

## 3. Expected experiments that are missing

| Expected | Status |
|---|---|
| Condition C (C-COND and C-MARG arms) | **RUNNING.** No frozen result artifact. Only pre-launch design artifacts are committed. Claim C7 marked `pending`. |
| τ³-Retail development smoke fit | **RUNNING.** No committed artifact of any kind on the smoke-fit branch. Claim C10 marked `pending`. |
| τ³ smoke-fit preregistration | **DOES NOT EXIST.** `git ls-files` on `tau3-retail-hpop-smoke-fit` returns no smoke-fit file. `tool_vocabulary.json` explicitly defers `K` to the fitting task. Consequently the pack does **not** state "infer $S, z, U, \pi, P$ with fixed recurrent scalars" as a preregistered target — the brief conditioned that statement on preregistration, and the condition is not met. |
| Condition D | Not run, not designed beyond being named in stopping rules. |
| ρ prior-draw calibration | Explicitly out of scope in the Condition B registration; a future experiment. |
| Vector (PDF/TikZ) TaskBench figures | Do not exist; PNG only. Non-blocking for submission, blocking for camera-ready. |

---

## 4. Exact source artifacts used

### Condition A (commit `b199374`, corpus `dd280a4a…`, truth `fc41538f…`)
`final_verdict.json` · `preregistration.json` · `corpus_manifest.json` ·
`correctness.json` · `boundary_metrics.json` · `label_metrics.json` ·
`joint_path_metrics.json` · `prior_vs_posterior.json` ·
`segment_count_posteriors_summary.json` · `true_path_probabilities.json` ·
`runtime.json` · `report.md`
(all under `results/mcmc_original/matched_condition_a/`).
NPZ payloads (`boundary_marginals.npz`, `occurrence_label_marginals.npz`,
`map_paths.npz`, `segment_count_posteriors.npz`, `exact_forward_results.npz`,
`ffbs_sample_summaries.json`, `metric_definitions.json`) exist and were not
needed: every requested scalar was available in the JSON summaries.

### Condition B (commit `34873d8`, same corpus and truth hashes)
`final_verdict.json` · `preregistration.json` · `target_manifest.json` ·
`corpus_manifest.json` · `source_manifest.json` · `correctness.json` ·
`convergence.json` · `structure_recovery.json` ·
`heldout_oracle_path_nll.json` · `structural_movement.json` ·
`start_manifest.json` · `selected_scales.json` · `pilot_results.json` ·
`rho_posterior.json` · `runtime.json` · `report.md`
(all under `results/mcmc_original/matched_condition_b/`).

### TaskBench
Construction (commit `e6afb17`): `taskbench_multimedia_audit/source_manifest.json`,
`parsing_summary.json`; `taskbench_multimedia_poset_benchmark/report.md`,
`benchmark_hash.json`, `graph_statistics.json`, `preregistration.json`,
`filter_config.json`, `test_split.json`, `development_split.json`,
`validation.json`.
Fit (commit `10d4bf4`): `taskbench_static_poset/final_verdict.json`,
`main_table.json`, `test/paired_bootstrap.json`, `test/condition_curves.json`,
`convergence_summary.json`, `runtime.json`, `correctness.json`,
`model_config.json`, `selected_scale.json`, `benchmark_manifest.json`,
`source_manifest.json`, `report.md`,
`figure_data/{figure_a_condition_curves,figure_b_margins}.json`,
`figure_data/figure_c_manifest.json`.

### τ³-Retail setup
Commit `8100fc2`: `tau3_retail_audit/source_manifest.json`,
`version_alignment.json`, `extraction_summary.json`;
`tau3_retail_hpop_pilot/corpus_hash.json`, `corpus_summary.json`,
`split_manifest.json`, `tool_vocabulary.json`, `selected_task_ids.json`,
`filter_config.json`, `preregistration.json`, `validation.json`, `report.md`.
`test_trajectories.jsonl` was **not** opened.

---

## 5. Numerical discrepancies found

Six. None changes any conclusion; all are recorded rather than silently
resolved.

### D1 — Generator validation report cites a different commit from the one that froze it
`matched_generator_validation/report.md` line 3 records
`Source commit: 6eb43b355333d07d72d7dffd836b654413702b04`, and
`matched_generator_smoke_corpus/report.md` says the same.
The artifacts were actually **committed at `8ca8281`** ("Implement exact matched
HPOP synthetic generator"), which is also the `generator_commit` recorded inside
`matched_synthetic_formal_corpus/config.json`, `corpus_manifest.json` and
Condition A's own manifest.
**Reading:** `6eb43b3` is the tree state the validation *ran against* (the
generator was uncommitted at that moment); `8ca8281` is the commit that froze
both the code and the validation record, and it supersedes.
**Resolution:** cite `8ca8281`. Both values are retained in the manifest.

### D2 — rep2's z-score is reported two ways within one file
`collapsed_u_kernel_validation_rep2/mixed_reference_comparison.json`:
- `gates.mixed_multivariate_reference_statistic.z_score = 5.759871639422952`
- `verdict` (same file) = "…0.006777 vs envelope 0.004522 (z +5.88)"
- `two_run_conclusion` (same file) = "…at z +2.86 and +5.88"

**Reading:** a ~2% difference, consistent with the two z's being computed
against slightly different null standard deviations (the structured field
against the run's own null, the prose against the historical envelope's).
**Resolution:** neither is preferred; the ledger records the structured field
(`5.7599`) as the machine value and this report records the disagreement. Both
are far above the 2.33 cutoff, so the FAIL verdict is unaffected. If the paper
quotes a z for rep2, quote `+5.76` from the structured field and do not
reproduce `+5.88` without this note.

### D3 — Burn-in described as "verified" while its artifact records `pass = false`
`collapsed_u_efficient_final_validation/report.md` line 5: "Burn-in 100,000
(verified)". `burnin_verification.json`: `pass = false`, because chain 3's
λ_rep block means were not inside the 7B1 reference band for every block in
[25k, 50k].
**Reading:** the criterion was evaluated on **throwaway diagnostic chains**
(seeds 8158901/8158902) that are never pooled into any validation, and the
registered burn-in was set to 100,000 — four times beyond the window where the
criterion failed.
**Resolution:** do not quote "(verified)" bare. State the diagnostic outcome and
the 4× margin. Recorded as caveat under claim C3 and as risk CR12.

### D4 — TaskBench run report prints the wrong AUROC in one cell
`taskbench_static_poset/report.md` main table, TWO / First total order row,
prints AUROC `0.818`. `main_table.json` records
`TWO.first_total_order.auroc.mean = 0.8547818651985318` and
`TWO.first_total_order.pairwise_acc.mean = 0.8177552552552552`.
**Reading:** the pairwise-accuracy value was transcribed into the AUROC column
of the human-readable report. The two columns are identical for the Bayesian and
antichain rows (which is why it is easy to miss) but not for this baseline.
**Resolution:** `main_table.json` is authoritative. The pack uses `0.855`, and
`table_taskbench_appendix.tex` carries an errata note. This affects a baseline
row only; no comparison, CI or verdict changes.

### D5 — τ³ repeated-CPA rate: 0.924, not 0.87
The task brief specified "87% repeated-CPA rate". Every frozen artifact
disagrees and agrees with itself: `corpus_summary.json`
`repeated_cpa_fraction = 0.9244`; `validation.json`
`repeated_cpa_rate_ge_25pct.detail = "0.924"`; `report.md` "repeated-CPA
fraction 92%".
**Resolution:** the artifact value `0.9244` is used everywhere in this pack.
`0.87` appears in no frozen file and was not adopted.

### D6 — TaskBench DAG count: README 565, released data 550
`taskbench_multimedia_audit/parsing_summary.json` documents this itself: the
upstream README groups counts over 5,584 human-verified samples (565 DAG), while
`data.json` at the pinned commit holds 5,555 records (550 DAG). Twenty-nine
records (14 chain, 15 DAG) have `user_requests` entries but no `data.json`
record at that commit.
**Resolution:** already handled correctly upstream — filters were not altered to
force agreement. Cite 550 marked / 529 independently verified acyclic. Listed
here so the paper does not accidentally quote the README's 565.

### Not a discrepancy, but a trap worth flagging
TaskBench antichain relation Brier is `0.2496` in `main_table.json` at ONE/TWO
and `0.2206` at FOUR, while `condition_curves.json` reports `0.2206` for the
antichain in *all three* conditions. These are consistent: `main_table.json`
aggregates over each condition's own test graphs (37/37/26) and
`condition_curves.json` aggregates over the 26 graphs common to all three. Do
not mix the two sources within one table.

### Quantities requested but not present in any frozen artifact
- The **full** Stage 6E2 trace-corpus SHA-256 (`report.md` truncates it to
  `02be246edf9bd4f4148efa3a3e269afa…`). Marked `NOT REPORTED IN FROZEN ARTIFACT`
  in the manifest; no claim in this pack depends on it.
- A τ³ smoke-fit preregistration. Does not exist; see §3.

---

## 6. Main paper untouched

No `main.tex`, no ICLR section files, no `.bib` and no figure-inclusion order
exist in this repository — `papers/` holds two reference PDFs only, and the
selected `HPOP_development_note_updated_20260816.tex` is a development note
outside the repository. The instruction to leave the main paper unchanged is
therefore satisfied trivially: **every file created by this task is new**, under
`paper/` and `scripts/validate_paper_evidence.py`. Nothing was overwritten and
no backup was needed.

---

## 7. Deliverables

```
paper/evidence/
    evidence_manifest.json          12 claims (C1-C12), scoped and status-marked
    result_ledger.csv               291 unrounded rows, each with artifact+field+commit
    result_ledger.md                same content, grouped and human-readable
    taskbench_figure_manifest.json  3 figures, selection rules, captions, vector TODO
    paper_todo.md                   blocking / non-blocking / sequencing
    risk_register.md                12 claim risks, 5 compute risks
    extraction_report.md            this file
paper/drafts/
    results_matched_synthetic_ab.tex
    results_taskbench.tex
    tau3_retail_setup.tex
    result_transition_paragraphs.tex
paper/tables/
    table_matched_synthetic.tex
    table_taskbench_main.tex
    table_taskbench_appendix.tex
    table_claim_scope.tex
scripts/
    validate_paper_evidence.py
```

## 8. Claim status summary

| Claim | Scope | Status |
|---|---|---|
| C1 exact FFBS | inference kernel | supported |
| C2 path-marginal + immediate FFBS preserves the target | inference kernel | supported |
| C3 collapsed-U kernel validated | inference kernel | supported (with two historical FAILs reported) |
| C4 matched generator | matched synthetic | supported |
| C5 path identifiable under oracle structures | matched synthetic | supported |
| C6 structure identifiable under oracle paths | matched synthetic | supported |
| C7 joint recoverability | pending | **pending** |
| C8 path marginalization improves joint mixing | inference kernel | **partially supported** |
| C9 static component generalizes externally | static partial-order component | supported |
| C10 full HPOP predictive gains on τ³-Retail | pending | **pending** |
| C11 τ³-Retail corpus frozen and validated | external data setup | supported |
| C12 TaskBench benchmark constructed reproducibly | external data setup | supported |

C8 is marked *partially supported* rather than pending because frozen artifacts
do establish the mechanism (FFBS-only produces 0–1 structural changes in 34,000
sweeps; the collapsed move cuts the cross-structure barrier by ~132 nats and
raises acceptance from ~1e-14 to ~7e-2; the validated kernel visits >3,500
distinct structures per chain). What remains unestablished — that this yields
correct joint posterior recovery — is exactly C7, and is pending.
