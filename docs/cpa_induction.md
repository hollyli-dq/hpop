# CPA Induction — LLM-assisted inductive coding (methodology)

**The CPA library is an OUTPUT of a cleaning study, not an input.** We do not begin from a
predefined CPA list. Canonical Procedural Actions are *induced* from the trajectories by
LLM-assisted open (inductive) coding with selective human adjudication.

```
raw trajectories → LLM-assisted open coding → candidate CPA library → human consolidation → cleaned annotated dataset
```

The 13 WebLINX CPAs (`rules/cpas.yaml`) are therefore a **pilot library induced from the WebLINX
domain (v0)** — illustrative seed examples, NOT a universal/closed ontology. They must not be
described as a fixed action vocabulary.

## Two libraries — keep separate
- **CPA library** — canonical procedural **actions** discovered during cleaning (this document).
- **Skill library** — reusable **local partial orders over CPA instances**, learned *later* by HPOP
  (`docs/MODEL.md`). A skill is NOT an LLM-proposed CPA label.

```
Raw events → induced CPAs → local partial-order skills → global partial order
            └── LLM open coding ──┘ └────────── HPOP discovers ──────────┘
```
HPOP later discovers skills such as `H_k: c1 ≺ c3, c2 ≺ c3` over CPA *instances* c1,c2,c3.

## Role of the LLM — occurrence-level CPA extraction, DATA PREPARATION ONLY
The annotator converts one trajectory into an ordered sequence of **occurrence-level CPAs**. It must
**NOT infer** skill boundaries, reusable skills, local/global partial orders, phases, or hidden
reasoning — those are learned downstream by BPOP/HPOP. (The finalized annotator spec is implemented
verbatim as the system prompt in `src/hpop/annotate/opencode.py`.)

A **CPA** = a temporally contiguous span of ≥1 observable events that (1) performs one coherent
procedural function, (2) operates on / produces an identifiable artifact or state, (3) is more
abstract than a raw tool name, (4) could recur across tasks/repos, (5) is supported by observable
evidence. A tool call is *not* automatically a CPA (the same shell tool may search / read / test /
install / execute / inspect / repair — infer the function from command+args+observation+context).

**Two modes** (input field `mode`):
- **INDUCE** — library not predefined; `decision = PROPOSE_NEW` unless genuinely uncertain (`ABSTAIN`).
- **APPLY** — match a *frozen* library by **definition** (not label wording); `decision =
  MATCH_EXISTING` when a definition fits, else `PROPOSE_NEW`, else `ABSTAIN`.

Per occurrence the model returns: contiguous `source_event_ids` (boundary; preserve retries / repeats
as separate occurrences), `candidate_label` (UPPER_SNAKE_CASE, verb+object, tool-agnostic) /
`canonical_label`, `definition`, `procedural_function`, `input_artifacts` / `output_artifacts` (with
evidence event ids), `state_before` / `state_after`, `outcome ∈ {SUCCESS,FAILURE,PARTIAL,UNKNOWN}`,
`boundary_confidence`, `label_confidence` (0.90–1.00 demonstrated / 0.70–0.89 supported / 0.50–0.69
review / <0.50 ABSTAIN), `evidence` (short statements, no chain-of-thought), `ambiguity`,
`review_required`. Plus per trajectory: `excluded_events`, `candidate_library_updates`,
`review_queue`, and `quality_checks` (incl. `skill_labels_inferred: false`).

`C^(0) = ∅` (or a tiny illustrated set) — not a closed label set. **Dependencies / partial orders are
NOT annotated here** (earlier drafts emitted them; the finalized spec leaves them to BPOP/HPOP).

## Consolidation (human-in-the-loop)
`C^(t+1) = MergeSplitReview( C^(t) ∪ { ĉ_i : decision_i = PROPOSE_NEW } )`.
- **Merge** labels only when their procedural function is equivalent
  (e.g. `inspect source file / read implementation / examine code / open relevant module` → one CPA).
- **Split** a broad label that covers procedurally different actions
  (e.g. `run command` → `RUN_TEST`, `INSTALL_DEPENDENCY`, `SEARCH_CODE`, `EXECUTE_PROGRAM`).

A proposed CPA enters the library only when it (1) recurs across trajectories, (2) has a stable
procedural definition, (3) is distinguishable from existing CPAs, (4) has consistent I/O behaviour,
(5) passes human or strong multi-model adjudication. Acceptance gate:
```
Accept(c) = 1[ N(c) ≥ m ] · 1[ R(c) ≥ r ] · 1[ A(c) ≥ τ ]
  N(c) = #occurrences,  R(c) = #distinct repos/tasks,  A(c) = adjudication agreement.
```
This stops one unusual command sequence from creating a permanent CPA.

## Humans are not removed
Human role shifts from *label every trajectory* → *define the annotation principle, review the
emerging library, and adjudicate uncertain/structurally-important cases*: proposed-new CPAs,
merge/split decisions, low-confidence boundaries, LLM-annotator disagreements, rare CPAs, and cases
that substantially alter dependency graphs (uncertainty-directed, selective review).

## Dataset: `nebius/SWE-rebench-openhands-trajectories`
OpenHands, role-separated: `assistant` tool_calls = actions, `tool` messages = observations, plus
`repo`, `model_patch`, `exit_status`, binary `resolved`, and `gen_tests_correct` /
`pred_passes_gen_tests` verification metadata. 67,074 trajectories / 1,823 repos. **Tool-call
arguments are JSON stored as strings — deserialize first** (`ingest/swe_rebench.py` does this).
Per-event tuple: `e_t = (tool name, arguments, observation, order, repository state)`.

## The cleaning experiment — 100-trajectory pilot
1. Sample **100 trajectories = 50 resolved + 50 unresolved**, across ≥15–20 repos, spanning short /
   medium / long, multi-edit, test-failure-then-edit, generated-tests, and unsubmitted runs.
   (Done: `data/interim/swe_rebench/pilot100.jsonl` — 100 traces, 87 repos, mean 66 actions.)
2. Two strong LLM configs perform open procedural coding **independently**.
3. Consolidate candidate actions → **CPA Library v0.1** (`consolidate.py`, `Accept` gate).
4. Human-review all proposed-new labels and disagreements.
5. Re-annotate a second sample with v0.1 (keep `PROPOSE_NEW` / `ABSTAIN`). 6. Update → **v0.2**.
7. **Freeze** the library before annotating the main review dataset.

## Output — one annotation object per trajectory (the finalized schema)
`opencode.py` writes the full spec object per trajectory to `<out>.jsonl` and flattens occurrences to
`<out>.cpa_instances.jsonl`. The object has: `trajectory_id`, `mode`, `library_version`,
`cpa_instances[]`, `excluded_events[]`, `candidate_library_updates[]`, `review_queue[]`,
`quality_checks{}`. Each occurrence:
```json
{
  "occurrence_id": "trajectory_id::CPA_0007",
  "source_event_ids": ["e0018", "e0019", "e0020"],
  "start_event_id": "e0018", "end_event_id": "e0020",
  "decision": "PROPOSE_NEW",
  "canonical_label": null,
  "candidate_label": "INSPECT_TEST_FAILURE",
  "definition": "Read failing test output to identify the failure cause.",
  "procedural_function": "examine pytest output and locate the failing assertion",
  "input_artifacts": [{"artifact": "pytest output", "evidence_event_ids": ["e0018"]}],
  "output_artifacts": [{"artifact": "candidate failure cause", "evidence_event_ids": ["e0020"]}],
  "state_before": "tests failing", "state_after": "failure cause hypothesized",
  "outcome": "SUCCESS", "boundary_confidence": 0.9, "label_confidence": 0.88,
  "evidence": ["pytest exit code 1 followed by traceback inspection"],
  "ambiguity": null, "review_required": false
}
```
No dependency/skill/partial-order records here — those are downstream (BPOP/HPOP).

## Code map
- `src/hpop/ingest/swe_rebench.py` — SWE-rebench-openhands → normalized trajectories `X` (50/50 pilot).
- `src/hpop/annotate/opencode_schema.py` — the finalized occurrence-level CPA schema (no fixed enum).
- `src/hpop/annotate/opencode.py` — the annotator (verbatim spec prompt; INDUCE/APPLY; writes the object).
- `src/hpop/annotate/consolidate.py` — `MergeSplitReview` scaffolding: aggregate `candidate_label`s,
  compute N/R/A, apply `Accept`, emit candidate library + human-review queue.

> Revised research target: **use LLM-assisted inductive coding, supported by selective human
> adjudication, to derive a domain-grounded CPA library and produce a provenance-preserving cleaned
> SWE-agent trajectory dataset for review.** ([inductive coding w/ LLMs](https://arxiv.org/abs/2512.00046);
> [selective human review](https://arxiv.org/abs/2511.09833))
