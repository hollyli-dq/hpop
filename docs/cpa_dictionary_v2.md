# The CPA Dictionary (v2, phase-derived) — paper reference

**Canonical sources.** `rules/cpas_v2.yaml` (source of truth, carries `trigger` +
`proximate_goal`) → `rules/cpa_dictionary_v2.json` (32 entries, consumed by
`src/hpop/annotate/rule_apply.py` at stage 3 of `scripts/run_all.sh`).
Loader: `src/hpop/annotate/cpa_v2.py`. Phase source: `rules/phases.yaml`.

**Corpus.** 500 SWE-rebench-openhands trajectories over 334 repos (5 pilot batches of
100; 50 resolved + 50 unresolved each), drawn from `nebius/SWE-rebench-openhands-trajectories`
(67,074 trajectories / 1,823 repos over 21,336 task instances). 32,664 raw events →
30,548 action tokens (2,116 merged by Layer B-1a normalization) → **23,929 labeled
occurrences**. 72% of tokens carry an `artifact_id`.

---

## 1. What a CPA is, and how v2 differs from v1

A **CPA** (Cognitive-Procedural Action) is the unit of the action layer: the
*proximate goal* an agent pursues with one observable action, in Leontiev's Action
tier sense.

The v2 dictionary is **top-down**. Each CPA is *created by* the cognitive rule:

> every CPA realizes **exactly one** of the 9 phases (`rules/phases.yaml`), and CPAs
> within a phase are individuated **only** by their proximate goal.

This inverts v1, which was bottom-up ("mine commands, then bucket them"). The
distinction matters for the paper's claim: top-down **declares** cognitively-valid
actions the sample never surfaced, and its phase assignment is a definition rather
than a post-hoc grouping.

**32 CPAs declared across 9 phases; 29 realized in the 500-trajectory sample.** No
v1 CPA was removed.

> **Numerical coincidence to state explicitly** — v1 also had 29 CPAs (of which 28
> fired). The v1 count and the v2 *realized* count coincide by accident over
> different vocabularies. Reviewers will otherwise read them as the same set.

## 2. The nine phases (Layer A)

Grounded in PDAF v3 (coauthor Report 3, §2.2); citations preserved verbatim from the
source table. Phases are a **recurrence vocabulary, not a linear order** — and they
are an *optional coarse procedural descriptor*, not the model's skill types. HPOP
receives CPA occurrence sequences and learns the poset itself; a learned skill may
span several phases.

| Level | Phase | Cognitive state | Reference | Occ. (500 traj) |
|---|---|---|---|---|
| L1 goal-setting | `PLAN` | Goal decomposition — PFC sets the goal hierarchy | Miller & Cohen (2001); Grosz & Sidner (1986) | 0 |
| L2 execution | `RETRIEVE` | Directed subgoal pursuit — externally-directed acquisition | Botvinick et al. (2009) | 4,298 |
| L2 execution | `INSPECT` | Metacognitive monitoring — passive scanning, no proposition tested | Nelson & Narens (1990) | 4,079 |
| L2 execution | `EXTRACT` | Focused subgoal pursuit — central executive locks on target | Baddeley (2000) | 1,681 |
| L2 execution | `VERIFY` | Metacognitive control — explicit proposition; output is a truth value | Nelson & Narens (1990); Wason (1960) | 7,931 |
| L2 execution | `WRITE` | Generative production — externalization mode | Flower & Hayes (1981) | 5,824 |
| L3 integration | `SYNTHESIZE` | Multi-source integration — assembles a representation no single unit possessed | Hutchins (1995) | 0 |
| L4 interrupt | `REPAIR` | Error monitoring — SAS override after a failure signal | Norman & Shallice (1986) | 116 |
| L4 interrupt | `HANDOFF` | Distributed cognitive load transfer (agent identity change) | Hutchins (1995); Monsell (2003) | 0 |

Phase grammar: `PLAN` opens an episode; L2 states freely recur and interleave;
`SYNTHESIZE` follows acquisition; `REPAIR`/`HANDOFF` may interrupt at any point and
return.

**Empty phases are a finding, not a gap.** `PLAN` is empty because our pipeline
excludes `think` tokens — planning is latent, not absent. `SYNTHESIZE` is a declared
theory slot (candidate: `INTEGRATE_FINDINGS`, a fix coordinating edits across several
files) that the rule layer cannot detect; deferred to the LLM pass. `HANDOFF` fires
only on an `agent_id` change and is structurally impossible in single-agent traces.

## 3. The codebook

Definitions and `distinguish` notes are **verbatim** from `rules/cpa_dictionary_v2.json`.
Occurrence counts are from `data/modelling/swe_rebench/sequences.vocab.json` (the v2
rule-apply run over 500 trajectories); `traj` = number of distinct trajectories in
which the CPA appears.

### L1 — PLAN

| ID | CPA | Definition (proximate goal) | Distinguish from | Occ. | Traj. |
|---|---|---|---|---|---|
| V2_01 | `PLAN_APPROACH` | decompose the task into an ordered set of sub-goals before acting | setting the goal hierarchy (here) vs pursuing a sub-goal (any L2 phase) | 0 | 0 |

*Trigger:* explicit plan/TODO/numbered-steps written by the agent (mostly latent in `think`).

### L2 — RETRIEVE

| ID | CPA | Definition | Distinguish from | Occ. | Traj. |
|---|---|---|---|---|---|
| V2_02 | `EXPLORE_REPOSITORY` | build a map of repo structure | going to a directory (here) vs opening a known file to read (INSPECT/READ_SOURCE) | 1,027 | 491 |
| V2_03 | `LOCATE_CODE` | find where a symbol/file/definition lives | searching to acquire a target (here) vs reading the target once found (INSPECT) | 3,271 | 500 |

*Triggers:* `view(<dir>)`, `ls`, `tree` — `grep`, `rg`, `find`, `find_file`, `search_dir`.

### L2 — INSPECT

| ID | CPA | Definition | Distinguish from | Occ. | Traj. |
|---|---|---|---|---|---|
| V2_04 | `READ_SOURCE` | understand an implementation by reading it | reading to understand, no truth test (here) vs reading to pull a specific value (EXTRACT) vs to test a proposition (VERIFY) | 3,207 | 494 |
| V2_05 | `READ_DOCUMENTATION` | understand intended behaviour from docs | reading docs (here) vs reading code (READ_SOURCE) | 491 | 437 |
| V2_06 | `INSPECT_CHANGES` | see one's own uncommitted edits | scanning your own diff (here) vs project source (READ_SOURCE) vs history (INSPECT_HISTORY) | 224 | 194 |
| V2_07 | `INSPECT_HISTORY` | understand how code reached its current state | scanning past history (here) vs your own diff (INSPECT_CHANGES) | 139 | 104 |
| V2_08 | `INSPECT_ENVIRONMENT` | see what is installed / the interpreter state | scanning env state (here) vs reading repo source (READ_SOURCE) | 18 | 17 |

*Triggers:* `view(<file>)`/`cat`/`head`/`tail`/`goto`/`scroll` — `view README|CHANGELOG|docs/*.md|.rst` — `git diff`/`git status`/`git show` (working tree) — `git log`/`git blame`/`git show <sha>` — `pip list|freeze|show`, `python --version`, `uname`, `printenv`.

### L2 — EXTRACT

| ID | CPA | Definition | Distinguish from | Occ. | Traj. |
|---|---|---|---|---|---|
| V2_09 | `DIAGNOSE_FAILURE` | lock onto a failure and pull out its root-cause value (the offending line/assertion) | extracting THE cause from a failure (here) vs general reading-to-understand (INSPECT) | 1,681 | 458 |

*Trigger:* reading a traceback / a view right after a failed run.
**Open question (flagged in-file):** `[coauthor: EXTRACT vs INSPECT — confirm]`. This
is the single unresolved phase assignment in the dictionary.

### L2 — VERIFY

Every CPA here has a **proposition** whose output is a truth value.

| ID | CPA | Definition (the proposition) | Distinguish from | Occ. | Traj. |
|---|---|---|---|---|---|
| V2_12 | `VERIFY_FIX` | did my change resolve it? | post-edit confirmation (here) vs an initial run (RUN_TEST_SUITE/REPRODUCE_ISSUE) | 4,247 | 500 |
| V2_10 | `RUN_TEST_SUITE` | is the behaviour correct? (first/observational test run) | proposition = tests pass? Initial run (here) vs post-edit confirmation (VERIFY_FIX) | 1,946 | 488 |
| V2_11 | `REPRODUCE_ISSUE` | does the reported bug actually occur? | proposition = bug reproduces? (here) vs the test suite (RUN_TEST_SUITE) | 1,689 | 461 |
| V2_14 | `RUN_LINTER` | is the style/lint clean? | style proposition (here) vs type soundness (CHECK_TYPES) | 35 | 6 |
| V2_15 | `COMPARE_OUTPUT` | do these two outputs/files match? | equality proposition over two artifacts (here) vs reviewing one's own git diff (INSPECT_CHANGES) | 7 | 6 |
| V2_16 | `MEASURE_PERFORMANCE` | is it fast/efficient enough? | performance proposition (here) vs correctness (RUN_TEST_SUITE) | 4 | 1 |
| V2_13 | `CHECK_TYPES` | are the types/static constraints sound? | static-analysis proposition (here) vs dynamic tests (RUN_TEST_SUITE) | 3 | 2 |

*Triggers:* re-run tests/repro after an edit — `pytest`/`tox`/`unittest` — run a repro script (pre-fix) — `flake8`/`pylint`/`ruff`/`pycodestyle`/`eslint` — `diff a b`/`cmp`/`difflib` — `timeit`/`cProfile`/`--durations` — `mypy`/`pyright`/`pyre`/`tsc`.

The `RUN_TEST_SUITE` / `REPRODUCE_ISSUE` / `VERIFY_FIX` triad is the clearest
demonstration that CPAs are not commands: **all three can be the same `pytest`
invocation**, individuated only by proximate goal and position relative to the edit.

### L2 — WRITE

| ID | CPA | Definition | Distinguish from | Occ. | Traj. |
|---|---|---|---|---|---|
| V2_18 | `WRITE_TEST` | externalize a kept regression test | kept test (here) vs throwaway repro (WRITE_REPRODUCTION_SCRIPT) | 1,988 | 491 |
| V2_17 | `EDIT_SOURCE` | change behaviour toward the fix | behavioural fix (here) vs temp prints (ADD_DEBUG_INSTRUMENTATION) vs no-op restructure (REFACTOR_CODE) vs test file (WRITE_TEST) | 1,478 | 482 |
| V2_19 | `WRITE_REPRODUCTION_SCRIPT` | externalize a scratch script that triggers the bug | scratch repro (here) vs kept test (WRITE_TEST) | 868 | 396 |
| V2_30 | `SUBMIT_SOLUTION` | externalize the final patch as the solution | terminal submission (here) vs intermediate write (EDIT_SOURCE) | 454 | 454 |
| V2_20 | `ADD_DEBUG_INSTRUMENTATION` | externalize temporary prints/logging to investigate | temp diagnostics (here) vs the actual fix (EDIT_SOURCE) | 384 | 181 |
| V2_28 | `CLEANUP_ARTIFACTS` | remove scratch before finalizing | removing scratch (here) vs reverting a source edit (REVERT_CHANGE) | 368 | 284 |
| V2_25 | `INSTALL_DEPENDENCY` | provide packages needed to run | installing deps (here) vs building (BUILD_PROJECT) | 168 | 123 |
| V2_26 | `BUILD_PROJECT` | compile/build so it can run | building (here) vs installing deps (INSTALL_DEPENDENCY) | 40 | 29 |
| V2_24 | `MANAGE_FILESYSTEM` | arrange files/dirs in the workspace | filesystem op (here) vs editing contents (EDIT_SOURCE) vs deleting scratch (CLEANUP_ARTIFACTS) | 34 | 16 |
| V2_21 | `FORMAT_CODE` | mechanically reformat code | mechanical reformat (here) vs behavioural change (EDIT_SOURCE) | 30 | 4 |
| V2_23 | `CONFIGURE_ENVIRONMENT` | make the code/tests runnable (env vars, venv) | preparing the runtime (here) vs installing packages (INSTALL_DEPENDENCY) | 7 | 4 |
| V2_29 | `COMMIT_CHANGES` | record changes in version control | committing (here) vs undoing (REVERT_CHANGE) | 5 | 3 |
| V2_22 | `REFACTOR_CODE` | restructure without changing behaviour | no behaviour change (here) vs a fix (EDIT_SOURCE) | 0 | 0 |
| V2_27 | `APPLY_PATCH` | externalize a prepared patch onto the tree | applying a patch (here) vs hand-editing (EDIT_SOURCE) | 0 | 0 |

*Triggers:* `create(test_*.py)`/test-file edits — `str_replace`/`insert` on source — `create(repro.py|debug_*.py)` — `finish`/`submit` — `str_replace` inserting `print`/`logging`/`pdb` — `rm <temp/scratch>` — `pip|conda|poetry install`, `apt-get` — `make`/`cmake`/`setup.py build`/`cargo|go build`/`mvn|gradle` — `mkdir`/`mv`/`cp`/`touch`/`chmod`/`ln -s` — `black`/`isort`/`autopep8`/`prettier`/`gofmt` — `export VAR=`/`source activate`/`venv`/`conda activate` — `git add|commit|branch|tag` — edits preserving observable behaviour — `git apply`/`patch -p`/`git am`.

Note on `ADD_DEBUG_INSTRUMENTATION`: when motivated by a failure the **control state
is `REPAIR`** while the **realized CPA is a `WRITE`**. This is the cleanest worked
example of phase ≠ CPA in the dictionary.

### L3 — SYNTHESIZE

Declared empty. Candidate `INTEGRATE_FINDINGS` (a fix coordinating edits across
multiple files/sources into one coherent change) is not rule-detectable; deferred to
the LLM pass.

### L4 — REPAIR

| ID | CPA | Definition | Distinguish from | Occ. | Traj. |
|---|---|---|---|---|---|
| V2_31 | `REVERT_CHANGE` | undo an edit that failed or regressed | restoring prior state (here) vs a new change (EDIT_SOURCE) | 102 | 37 |
| V2_32 | `INTERACTIVE_DEBUG` | drop into a debugger/REPL to inspect runtime state after a failure | interactive runtime override (here) vs adding print statements (ADD_DEBUG_INSTRUMENTATION, a WRITE) | 14 | 1 |

*Triggers:* `git checkout --`/`git restore`/`git stash`/`undo_edit`/`git reset` — `pdb`/`breakpoint()`/`ipython`/`python -i`.

### L4 — HANDOFF

Declared empty. Fires only on an `agent_id` change (Layer B field); no CPA in
single-agent SWE traces. Kept for multi-agent generality.

## 4. Declared-but-unrealized CPAs

The three CPAs the top-down lens declares that never fire in 500 trajectories —
**`PLAN_APPROACH`, `APPLY_PATCH`, `REFACTOR_CODE`** — are the concrete payload of the
bottom-up/top-down difference. Each is cognitively valid and phase-licensed; the
sample simply does not contain it. Bottom-up induction *cannot* produce them by
construction.

Their zero counts have distinct causes, which is worth saying in the paper:

- `PLAN_APPROACH` — measurement artifact: the pipeline excludes `think` tokens, so
  planning is latent rather than absent. A genuine theory gap.
- `APPLY_PATCH` — real behavioural absence: OpenHands agents hand-edit via
  `str_replace`; they do not `git apply`.
- `REFACTOR_CODE` — an operationalization limit, not an absence: "edits preserving
  observable behaviour" is not decidable by a rule layer, so no trigger can fire. Its
  occurrences are almost certainly absorbed into `EDIT_SOURCE`.

32 declared − 3 unrealized = **29 realized**, which is the `|V| = 29` used throughout
the evaluation.

## 5. Evaluation

### 5.1 Structure ladder (`data/modelling/swe_rebench/ladder.json`)

Transition mutual information `I(L_t; L_{t+1})` in bits. Null = in-place
per-trajectory shuffle (preserves length and composition, destroys order).
`excess = I_real − I_rand`; `eff = excess / |V|`.

| Scheme | \|V\| | I_real | I_rand | Excess | Ratio | Eff |
|---|---|---|---|---|---|---|
| `cmdword` (no instruction) | 62 | 0.6227 | 0.0406 | 0.5821 | 15.33× | 0.0978 |
| `actobj` (no instruction) | 24 | 0.9413 | 0.0231 | 0.9182 | 40.74× | 0.2003 |
| v1 CPA (induced, bottom-up) | 28 | 0.9676 | 0.0274 | 0.9402 | 35.36× | 0.1956 |
| **v2 CPA (phase-derived)** | **29** | **1.0086** | **0.0299** | **0.9787** | **33.71×** | **0.2015** |

The no-instruction baselines (`random`, `cmdword`, `actobj`) are built as first-class
annotation files by `scripts/build_baselines.py` and scored by
`scripts/eval_dictionary.py`.

### 5.2 Three-layer head-to-head (`docs/cpa_study.html`)

| Layer | \|V\| | Predictability (NP) | Info on next action | Coverage |
|---|---|---|---|---|
| physical (`action_type`) | 5 | 0.097 | 0.159 b | 100% |
| action (CPA) | 29 | 0.281 | 0.355 b | 100% |
| cognitive (phase) | 6 | 0.186 | 0.253 b | 100% |

**The key result.** At near-equal vocabulary size (5 vs 6), the phase layer is ~2×
more self-predictable than the physical layer, and knowing the current *phase* reveals
more about the next physical action (0.253 b) than knowing the current *physical
action* does (0.159 b). The cognitive layer is not a relabeling of the physical one.

### 5.3 Saturation / rarefaction (200 permutations over the 500 annotated)

| Traces | E[CPA types] | Marginal new/trace |
|---|---|---|
| 50 | 22.1 | 0.078 |
| 100 | 24.0 | 0.038 |
| 200 | 25.9 | 0.014 |
| 300 | 26.8 | 0.010 |
| 500 | 28.0 | 0.005 |

Good-Turing coverage 0.996; saturation estimated at ~300–800 stratified traces.

### 5.4 Sampling strategy (`docs/cpa_sampling_strategy.md`)

Issue-intent stratification over 2,992 issues / 510 repos. Three taggers compared;
LLM anchoring (`claude-haiku-4-5`, structured outputs, ~25 issues/batch) wins —
regex leaves 32.5% `other`, TF-IDF/KMeans clustering fails outright (retained as
negative evidence in `scripts/cluster_issues.py`).

| Intent | Regex | Embed+deconf | **LLM** |
|---|---|---|---|
| bug_incorrect | 20.2% | 11.3% | **40.3%** |
| feature_request | 9.7% | 13.7% | **29.3%** |
| crash_traceback | 19.6% | 11.0% | **16.0%** |
| api_design | 8.9% | 11.3% | **5.0%** |
| typing_docs | 7.1% | 12.4% | **2.4%** |
| perf | 2.0% | 10.1% | **1.1%** |
| other | 32.5% | 30.3% | **5.8%** |

The corpus is **repair-dominated**: 40.3% + 16.0% = 56% repair work. This motivates
intent-stratified rather than uniform sampling for the CPA library.

## 6. Caveats to carry into the paper

Reviewers will find these; state them first.

1. **The guidance matters only modestly for raw structure** (`actobj` 0.918 vs guided
   v2 0.979). The instruction's decisive win is **vocabulary control** — the
   interpretable, phase-licensed 29 rather than 62 command-words. Do not overclaim MI.
2. **Task success is not a usable utility metric.** Predicting `resolved` from label
   profiles gives AUC ≈ 0.60–0.66 for *all* layers; procedural labeling is only
   loosely tied to outcome. We do not rely on it.
3. **Single-trajectory artifacts.** `INTERACTIVE_DEBUG` (14 occ / **1 traj**),
   `MEASURE_PERFORMANCE` (4 occ / 1 traj), `CHECK_TYPES` (3 occ / 2 traj) fall under
   the appearance threshold. They are not established categories on this evidence.
4. **Confounded triggers.** `RUN_LINTER` (35 occ / 6 traj) and `FORMAT_CODE` (30 occ /
   4 traj) fire partly on repos where the linter/formatter *is the system under test*.
   Extremely concentrated.
5. **No gold standard.** `data/gold/` is empty; intent tagging is single-annotator
   with no inter-annotator agreement check. The rarefaction curve comes from a
   **fixed-codebook rule annotator**, so it shows shape and *understates* true
   open-coding discovery.
6. **`EXTRACT` vs `INSPECT` for `DIAGNOSE_FAILURE`** is unresolved pending coauthor
   sign-off. It carries 1,681 occurrences — the entire `EXTRACT` phase — so the
   decision moves a whole phase.

## 7. Errata in the source files (fix before submission)

Three statements in the repo contradict the data. The **data is right**; the metadata
is stale, carried over from the v1 era.

1. **`READ_DOCUMENTATION` is mismarked `declared_unseen`** in both `cpas_v2.yaml:53`
   and `cpa_dictionary_v2.json:53` ("0 occurrences in 500-traj sample"). It in fact
   fires **491 times across 437 trajectories** — it is one of the more common CPAs.
2. **`REFACTOR_CODE` is mismarked `active`** but has **0 occurrences**. It, not
   `READ_DOCUMENTATION`, is the third declared-unseen CPA.
3. **The `cpas_v2.yaml` header narrative is wrong**: it names the "+3 declared" set as
   `PLAN_APPROACH` / `READ_DOCUMENTATION` / `APPLY_PATCH`. The realized set is
   `PLAN_APPROACH` / `APPLY_PATCH` / `REFACTOR_CODE`. `docs/cpa_study.html` already
   states this correctly.

Also note:

- **`docs/cpa_dictionary.md` is the v1 codebook** (29 CPAs, 4 ad-hoc phases) and is
  *not* current. This file supersedes it for v2.
- **The `docs/baselines_theory.tex` pytest example is not dictionary-conformant**: it
  uses `REPRODUCE_FAILURE`, `RUN_TEST_TO_VALIDATE_PATCH`, `CHECK_REPAIR_SIGNAL`. The
  real names are `REPRODUCE_ISSUE` and `VERIFY_FIX`; there is no repair-signal CPA.
  Either rename to conform or mark the example explicitly illustrative.
- **The 95%-incomparable artifact-grounding result** (80% coverage, 28 labels → 7,997
  grounded tokens) is cited in the DAG argument, but its outputs were deleted in
  `9acd960`. Recover from git at `146fa46` before citing.
- `scripts/mini_partial_order_test.py:28` hardcodes `FULL_CPA_VOCAB = 29` sourced from
  `docs/cpa_view.html`. Correct by coincidence under v2 — worth pinning to the vocab
  file.
- `docs/cpa_sampling_strategy.md` and the entire issue-intent pipeline (7 scripts,
  5 data files, `chunks/`) are **untracked**. `run_all.sh` stage 5 recovers v1
  sequences via `git show 146fa46:...`, so the ladder is only reproducible with full
  git history present.

## 8. Generative model (for §sec:generative)

Phase-first coarse-to-fine, per span `x_i` with local context `ξ_i`:

```
h_i ~ p(h_i | ξ_i)                  (phase)
c_i ~ p(c_i | h_i, A_{h_i})         (CPA within phase)
x_i ~ p(x_i | c_i, h_i)             (realisation)

p(H,C,X) = Π_i p(h_i|ξ_i) · p(c_i|h_i) · p(x_i|c_i,h_i)
```

**Phase-specific dictionary complexity penalty.** With `K_h = |A_h|`, regularize by
`−λ_CPA · Σ_h K_h`, admitting a candidate CPA into phase `h` only when

```
Δ log p(x_i | c_new, h) > λ_CPA + ΔComplexity
```

which prevents synonym explosion *within* a phase.

**Three latents, kept distinct:** `h_i` (phase — annotation prior, fixed in
preprocessing) ≠ `c_i` (CPA label) ≠ `z_ℓ` (HPOP skill — learned poset). In general
`h_i ≠ z_ℓ`: a learned skill may span several phases.

Worked example — the same `pytest tests/test_parser.py` command is
`(INSPECT, REPRODUCE_ISSUE)` pre-edit, `(VERIFY, VERIFY_FIX)` post-edit, and a
`REPAIR`-state action after a failure. Command identity is constant; the CPA is not.
