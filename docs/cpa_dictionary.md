# CPA Dictionary — Canonical Procedural Actions for coding-agent traces

A **codebook**: the controlled vocabulary an annotator (LLM or human) looks up to label a
software-engineering agent trajectory. Each entry has a **definition**, what it **includes**
(triggers), how to **distinguish** it from near neighbours (the part that drives consistency), and
its **input → output**. Grounded in the SWE-rebench/OpenHands pilot (100 trajectories); the
frequency note per phase reflects what actually occurs.

> A CPA is the procedural *function*, not the tool. The same `execute_bash` is `RUN_TEST_SUITE`,
> `LOCATE_CODE`, `INSTALL_DEPENDENCY`, or `REPRODUCE_ISSUE` depending on the command + context.
> Machine-readable form: `rules/cpa_dictionary.json`. This is a draft for coauthor review; entries
> are revisable from data (LLM open coding may PROPOSE_NEW beyond this set).

---

## A. Orient  *(view 1705 · grep/find/ls 1054 · cat/head 314)*

**EXPLORE_REPOSITORY** — Survey repository structure to build a mental map before acting.
*Includes:* `view(<dir>)`, `ls`, `tree`, `cd`/`pwd`. *Distinguish:* viewing a **directory** (explore)
vs viewing a **file** (`READ_SOURCE`). *In→Out:* task/issue → repo layout.

**LOCATE_CODE** — Search the codebase to find the file, symbol, or definition relevant to the task.
*Includes:* `grep`, `rg`, `find`, `find_file`, `search_dir`. *Distinguish:* *searching for* something
(locate) vs *reading* a known file (`READ_SOURCE`). *In→Out:* symbol/error → file paths / line numbers.

**READ_SOURCE** — Open and read a source, test, or doc file to understand its implementation/behaviour.
*Includes:* `view(<file>)`, `cat`, `head`, `tail`, `goto`, `scroll`. *Distinguish:* reading to
*understand* (here) vs reading a **failure/traceback to find a cause** (`DIAGNOSE_FAILURE`). *In→Out:*
file path → code understanding.

**INSPECT_HISTORY** — Examine the repository's **commit history** (`git log`, `git blame`,
`git show <sha>`, `git diff <sha>`) to understand how/why the code reached its current state.
*Distinguish:* reading **past history/blame** (here) vs reviewing your **own uncommitted edits**
(`INSPECT_CHANGES`) vs reading **current source** (`READ_SOURCE`). *In→Out:* symbol/regression → origin
of code/regression. *(NEW — induced from batch B; 34 repos across both pilots were forcing this into
`REPRODUCE_ISSUE`.)*

## B. Reproduce & diagnose  *(pytest 683 · python-run 933 · git diff 65)*

**RUN_TEST_SUITE** — Execute the project's tests (suite or a test file) to observe pass/fail status.
*Includes:* `pytest`, `tox`, `unittest`, `python -m pytest`. *Distinguish:* a *first/observational*
test run (here) vs re-running **after an edit to confirm a fix** (`VERIFY_FIX`); vs running a
**reproduction script** rather than tests (`REPRODUCE_ISSUE`). *In→Out:* test paths → test report.

**REPRODUCE_ISSUE** — Run a command/script to trigger and observe the reported bug.
*Includes:* `execute_bash <repro cmd>`, `python repro.py`. *Distinguish:* runs a **repro/program**
(here) vs the **test suite** (`RUN_TEST_SUITE`); pre-fix observation vs post-fix check (`VERIFY_FIX`).
*In→Out:* issue recipe → observed failure.

**WRITE_REPRODUCTION_SCRIPT** — Create a script/file that reproduces the bug for repeatable checking.
*Includes:* `create(repro.py)`, `create(debug_*.py)`. *Distinguish:* a **repro/scratch** script (here)
vs a **regression test** kept in the suite (`WRITE_TEST`). *In→Out:* hypothesis → reproduction script.

**DIAGNOSE_FAILURE** — Inspect failing output (traceback, diff, assertion) to identify the cause;
follows a failure. *Includes:* reading a traceback / a `view` after a failed run. *Distinguish:* reading
*because something failed* (here) vs ordinary `READ_SOURCE`. *In→Out:* failure output → root-cause hypothesis.

**INSPECT_CHANGES** — Review the agent's own modifications so far. *Includes:* `git diff`, `git status`,
`git show`. *Distinguish:* viewing **your edits/diff** (here) vs reading project source (`READ_SOURCE`);
vs examining **past commit history** (`INSPECT_HISTORY`). *In→Out:* working tree → change summary.

**CHECK_TYPES** — Run a static type checker / static analyzer (`mypy`, `pyright`, `pyre`, `tsc`) to
surface type/static errors **without executing** the program. *Distinguish:* static analysis (here) vs
running **dynamic tests** (`RUN_TEST_SUITE`) vs executing a **repro** (`REPRODUCE_ISSUE`). *In→Out:*
sources → type/static error report. *(NEW — induced; recurs in both pilots, distinct from test runs.)*

## C. Modify  *(str_replace 533 · create 566 · install 42 · build 10)*

**EDIT_SOURCE** — Modify source code to change behaviour toward fixing the issue.
*Includes:* `str_replace`/`insert` on a source file. *Distinguish:* a real behavioural change (here) vs
inserting **temporary prints** (`ADD_DEBUG_INSTRUMENTATION`) vs **behaviour-preserving** restructuring
(`REFACTOR_CODE`) vs editing a **test** file (`WRITE_TEST`). *In→Out:* root cause → patched source.

**WRITE_TEST** — Add or modify a test that guards intended behaviour. *Includes:* `create(test_*.py)`,
edits to test files. *Distinguish:* a **kept regression test** (here) vs a throwaway repro
(`WRITE_REPRODUCTION_SCRIPT`). *In→Out:* intended behaviour → new test.

**ADD_DEBUG_INSTRUMENTATION** — Insert temporary prints/logging/asserts to investigate (not the fix).
*Distinguish:* temporary diagnostic edits (here) vs the actual fix (`EDIT_SOURCE`). *In→Out:* hypothesis → debug trace.

**REFACTOR_CODE** — Restructure code without changing externally observable behaviour.
*Distinguish:* no behaviour change (here) vs a fixing change (`EDIT_SOURCE`). *In→Out:* working code → restructured code.

**REVERT_CHANGE** — Undo a previous edit that failed or regressed. *Includes:* `undo_edit`,
`git checkout --`, `git restore`, restoring prior content. *In→Out:* failed change → prior state.

**INSTALL_DEPENDENCY** — Install/configure packages needed to run the code or tests. *Includes:*
`pip install`, `conda install`, `poetry add`, `apt-get`. *In→Out:* missing-dep error → prepared env.

**BUILD_PROJECT** — Compile/build the project so it can run. *Includes:* `make`, `cmake`, `setup.py
build`, `cargo build`, `go build`, `mvn`/`gradle`. *Distinguish:* building (here) vs installing deps
(`INSTALL_DEPENDENCY`). *In→Out:* sources → build artifacts. *(new — rare but real, 10 occurrences.)*

## D. Verify & finalize  *(rm 65 · finish/submit)*

**VERIFY_FIX** — Re-run the reproduction or relevant tests *after* a change to confirm it resolves the
issue. *Distinguish:* a post-edit confirmation (here) vs an initial `RUN_TEST_SUITE`/`REPRODUCE_ISSUE`.
*In→Out:* patched source + test/repro → pass confirmation.

**CLEANUP_ARTIFACTS** — Remove temporary/scratch files (repro scripts, debug output) before finalizing.
*Includes:* `rm <temp>`. *In→Out:* temp files → clean working tree.

**SUBMIT_SOLUTION** — Finalize and submit the patch as the solution. *Includes:* `finish`, `submit`.
*In→Out:* verified fix → submitted patch.

---

### Notes on granularity
- **Not too broad** — never `EXECUTE_COMMAND`/`USE_TOOL`; the procedural function is the unit.
- **Not too narrow** — never `READ_PARSER_PY_LINE_84`; CPAs are portable across repos/languages.
- **Excluded as non-action:** pure planning/`think` messages (no observable procedural effect).
- **29 CPAs** in 4 phases (orient / reproduce-diagnose / modify / verify-finalize). The phase grouping
  is a reading aid, not a label the model uses. The 9 phase-2 induced CPAs are listed below.

### Phase-2 induced CPAs (appearance threshold)
- **RUN_LINTER** *(reproduce_diagnose)* — style/lint scan (flake8, pylint, ruff, pycodestyle, eslint); ≠ CHECK_TYPES, ≠ RUN_TEST_SUITE.
- **FORMAT_CODE** *(modify)* — auto-formatter (black, isort, autopep8, prettier, gofmt); mechanical, ≠ EDIT_SOURCE/REFACTOR_CODE.
- **CONFIGURE_ENVIRONMENT** *(modify)* — set env vars / activate-create venv/conda (export, PYTHONPATH, source activate); ≠ INSTALL_DEPENDENCY/BUILD_PROJECT.
- **MANAGE_FILESYSTEM** *(modify)* — mkdir/mv/cp/touch/chmod/symlink; not a source edit, not scratch cleanup.
- **COMMIT_CHANGES** *(verify_finalize)* — record in VCS (git add/commit/branch/tag); ≠ REVERT_CHANGE/INSPECT_CHANGES. (rare — agents seldom commit.)
- **MEASURE_PERFORMANCE** *(reproduce_diagnose)* — profile/benchmark (timeit, cProfile, --durations).
- **COMPARE_OUTPUT** *(reproduce_diagnose)* — diff two outputs/files (diff a b, cmp, difflib); ≠ INSPECT_CHANGES (own git diff).
- **INTERACTIVE_DEBUG** *(reproduce_diagnose)* — interactive debugger/REPL (pdb, ipython, python -i); ≠ ADD_DEBUG_INSTRUMENTATION (print/log edits).
- **INSPECT_ENVIRONMENT** *(orient)* — inspect installed packages / interpreter (pip list/freeze, python --version, uname).

### Induction log (how the set grew)
**Induction policy (coauthor, 2026-06-25): appearance threshold.** A distinct procedural function becomes
a CPA as soon as it **appears** in the data (≥1 occurrence) — no repo-count gate — provided it still obeys
the label rules (portable verb+object, not a one-off entity). The dictionary grows **monotonically** as we
iterate over batches of 100. Method: `scripts/iter_induct.py` (offline stand-in for the LLM INDUCE pass).

**Phase 1 (gap analysis, ≥4-repo gate) added 2:** `INSPECT_HISTORY` (git log/blame; was REPRODUCE_ISSUE)
and `CHECK_TYPES` (mypy/pyright; ≠ dynamic tests) → 20 CPAs.

**Phase 2 (appearance threshold) — iterative growth over 5 pilots (500 traj, 334 repos):**

| after batch | corpus | dictionary | newly induced |
|---|---|---|---|
| 1 (A) | 100 | 27 | RUN_LINTER, FORMAT_CODE, CONFIGURE_ENVIRONMENT, MANAGE_FILESYSTEM, INSPECT_ENVIRONMENT, MEASURE_PERFORMANCE, COMPARE_OUTPUT |
| 2 (B) | 200 | 27 | — |
| 3 (C) | 300 | 28 | COMMIT_CHANGES |
| 4 (D) | 400 | 29 | INTERACTIVE_DEBUG |
| 5 (E) | 500 | 29 | — |

The curve **flattens**: batches 2 and 5 add nothing, and two probed functions (`READ_DOCUMENTATION`,
`APPLY_PATCH`) never appeared, so they were not added — faithful to "as long as it appears." **29 CPAs**
total. Confounds are kept but flagged: `RUN_LINTER`/`FORMAT_CODE` fire partly on repos where the
linter/formatter *is* the system under test; `COMMIT_CHANGES` is rare (agents seldom commit — the patch is
the deliverable). The per-occurrence rule counts for induced CPAs are conservative (primary-function
gating) and lower than their raw appearance counts; the LLM APPLY pass will label more precisely.
