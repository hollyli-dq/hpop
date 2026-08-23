# CPA Library Sampling Strategy

How to sample the SWE-rebench corpus to induce the CPA (Coding-agent Process Action) library,
and how the trace-sampling strata were chosen. Written 2026-06-29.

## 1. Problem

We want to build a CPA library (codebook) by open-coding agent trajectories, then scale to a larger
sample. Two constraints shape the design:

- **Diminishing returns.** New-CPA discovery saturates fast, so most of a large random sample is
  wasted re-confirming common CPAs.
- **Coverage.** Rare CPAs (REPAIR, BACKTRACK, RE-PLAN) are concentrated in particular kinds of task,
  so a flat random draw under-discovers them.

The goal is a sample that maximises CPA coverage per annotated trace, with a principled stopping rule.

## 2. Dataset facts

| | |
|---|---|
| Trajectories dataset | `nebius/SWE-rebench-openhands-trajectories` — **67,074** trajectories |
| Task dataset | `nebius/SWE-rebench` — **21,336** unique task instances (the "questions") |
| Relationship | the 67k trajectories are repeated agent runs over the 21k instances |
| Annotated so far | 500 trajectories (CPA-annotated via rule-apply seed) |
| Repo spread (3k issue sample) | 510 distinct repos; top repo only 3.3% → repo is a high-cardinality nuisance axis, not a stratum |

Practical note: the trajectories dataset's `/rows` endpoint is ~14 s/row (it drags the full trajectory
column), so issue text for clustering/labelling is pulled from the **light task dataset** instead
(`problem_statement` column; ~100 rows / 30 s). The real issue text lives inside the
`<issue_description>` block of the first user message.

## 3. Diminishing-returns evidence (from the 500 annotated traces)

Randomised rarefaction (200 permutations) of CPA-type discovery vs traces annotated:

| Traces | E[CPA types] | Marginal new types / trace |
|---|---|---|
| 50 | 22.1 | 0.078 |
| 100 | 24.0 | 0.038 |
| 200 | 25.9 | 0.014 |
| 300 | 26.8 | 0.010 |
| 500 | 28.0 | 0.005 |

Marginal discovery drops below 0.01 new types/trace at ~300 traces; Good-Turing coverage 0.996.
**Implication:** 5% of 67k (≈3,350 traces) is a *ceiling*, not a target. Codebook discovery likely
saturates in **~300–800 stratified traces**; the surplus budget should go to a rare-tail/completeness
pass and to occurrence statistics for the partial-order model, **not** to redundant discovery.
(Caveat: this curve is from a fixed-codebook rule annotator, so it shows the *shape* and understates
true open-coding discovery — inflate the estimate with a safety factor.)

## 4. Choosing the stratification axis

The user's idea was to cluster/segment the questions first, then sample traces across the clusters.
Tested directly:

- **TF-IDF + KMeans on issue text → fails.** Silhouette ≈ 0.01 across all k; clusters collapse to
  *library/domain identity* (sympy, pandas, sqlfluff…) plus surface artifacts (URLs, tracebacks,
  screenshots). That is redundant with stratifying on `repo`, so it adds nothing.

The axis that *does* matter is **issue INTENT** (the kind of change requested), which is orthogonal to
repo and is what drives the CPA repertoire:

| Intent | CPA repertoire it tends to drive |
|---|---|
| crash_traceback | REPRODUCE / REPAIR / BACKTRACK (the rare CPAs) |
| bug_incorrect | locate → reproduce → fix → verify |
| feature_request | DESIGN / DECOMPOSE / IMPLEMENT |
| api_design | broad-edit, compat / deprecation |
| typing_docs | narrow, low CPA diversity |
| perf | profile → optimise (rare) |
| other | residual / ambiguous |

## 5. How intent was extracted — three methods tried

| Method | Tool | `other` rate | Verdict |
|---|---|---|---|
| Regex keyword cascade | `scripts/tag_issue_intent.py` | 32.5% | brittle, first-match bias, English-bound |
| Static embeddings, label-anchored + repo de-confounding | `scripts/embed_issue_intent.py` (model2vec) | 30.3% | de-confounding balanced it, but margins weak (~0.08); ~30% ambiguous |
| **LLM-anchored** | `scripts/llm_issue_intent.py` (Haiku + structured outputs); this run executed via 12 Claude subagents | **5.8%** | accurate; reads each issue and commits to a real intent |

Why embeddings alone also fail: domain/library is the dominant axis of variance in both TF-IDF and
neural embeddings, so naive clustering recovers domain, not intent. The fixes that work are
**label-anchored classification** (assign to nearest intent prototype, not to raw geometry) plus
**per-repo de-confounding** (subtract each issue's repo-mean embedding). The LLM does this implicitly
and far better.

## 6. Result — LLM-anchored intent distribution (2,992 issues, all labeled, 0 missing)

| Intent | Regex | Embed+deconf | **LLM** |
|---|---|---|---|
| bug_incorrect | 20.2% | 11.3% | **40.3%** |
| feature_request | 9.7% | 13.7% | **29.3%** |
| crash_traceback | 19.6% | 11.0% | **16.0%** |
| api_design | 8.9% | 11.3% | **5.0%** |
| typing_docs | 7.1% | 12.4% | **2.4%** |
| perf | 2.0% | 10.1% | **1.1%** |
| **other (unclassifiable)** | 32.5% | 30.3% | **5.8%** |

Headline: `other` collapses from ~30% → 6%. The picture is **repair-dominated** —
40% wrong-behavior + 16% crashes = **56% repair work** — with feature requests the other large class
(29%). That split is what you'd expect from a SWE-bench-style corpus (built from real bug-fix PRs), so
it doubles as a sanity check on the labels.

## 7. Recommended sampling plan

**A. Discovery sample → freeze the library (~600–800 traces).** Stratify on intent with the
allocation below; cap ≤4 traces/repo (spread across the 510 repos); force ~50/50 resolved within each
intent; soft-oversample long trajectories. Open-code in batches; **stop** when 2 consecutive batches add
< 2 new CPA types AND Good-Turing coverage > 0.95.

Allocation for an 800-trace discovery budget (oversample rare-CPA-bearing intents):

| Intent | Share | Alloc |
|---|---|---|
| feature_request | 29.3% | 261 |
| bug_incorrect | 40.3% | 239 |
| crash_traceback | 16.0% | 152 |
| api_design | 5.0% | 38 |
| typing_docs | 2.4% | 36 (floor) |
| perf | 1.1% | 36 (floor) |
| other | 5.8% | 36 |

**B. Tail / completeness pass (~200–300 traces).** Embed all trajectories, sample clusters with no
annotated representative yet — the backstop random sampling misses.

**C. Spend the rest on occurrence statistics, not discovery.** Once the codebook is frozen, use the
remaining budget toward reliable CPA frequencies + co-occurrence for the partial-order model.

**New CPA found mid-annotation →** candidate→promote→back-annotate loop: emit a `NEW_CANDIDATE`, park in
`rules/cpa_dictionary_proposed.json` (status=proposed), triage against neighbours' `distinguish` notes,
promote to canonical only on recurrence (≥k traces, ≥2 repos), then re-scan already-annotated traces for
the new code. Count canonical *promotions* (not raw candidates) for the saturation stop rule.

## 8. Open caveats / what to validate next

1. **Stated intent ≠ enacted work.** These labels classify the *issue text* (what the human asked
   for), not the *trajectory* (what the agent did). A feature_request can still involve heavy
   debugging. Intent is a **prior over the CPA distribution, not a measurement of it.**
2. **The oversampling weights are an untested assumption.** They assume crash/feature traces carry
   rarer CPAs. **Testable now:** join these intent labels to the 500 already-CPA-annotated trajectories
   by `instance_id` and cross-tab intent × CPA inventory. If intent predicts CPA-mix, the weights are
   justified; if intent and CPA-mix are roughly independent, intent is a weak stratifier and we should
   switch to a trajectory-derived axis (length, failure flag, tool-family entropy). **This is the
   highest-value check before committing the sampler.**
3. **Single-annotator, no agreement check.** Each issue was labeled once. For a defensible number, do a
   second pass on a ~200 subsample and report inter-annotator agreement; tighten the rubric on the
   boundary cases (bug vs crash, bug vs api_design) if low.

## 9. Artifacts

| Path | What |
|---|---|
| `scripts/pull_problem_statements.py` | pull clean issue text from the light task dataset |
| `scripts/cluster_issues.py` | TF-IDF/KMeans (kept as negative evidence) |
| `scripts/tag_issue_intent.py` | regex intent tagger (baseline) |
| `scripts/embed_issue_intent.py` | static-embedding label-anchored classifier (+ de-confounding) |
| `scripts/llm_issue_intent.py` | LLM-anchored classifier (Haiku + structured outputs) |
| `data/interim/swe_rebench/issues_3k.jsonl` | 3k issues sampled across the 21k question space |
| `data/interim/swe_rebench/issues_llm_tagged.jsonl` | all 2,992 issues with LLM intent labels (keyed by `instance_id`) |
