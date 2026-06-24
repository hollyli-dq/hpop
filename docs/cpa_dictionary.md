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
`git show`. *Distinguish:* viewing **your edits/diff** (here) vs reading project source (`READ_SOURCE`).
*In→Out:* working tree → change summary. *(new — 65 occurrences were previously unlabeled.)*

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
- **18 CPAs** in 4 phases (orient / reproduce-diagnose / modify / verify-finalize). The phase grouping
  is a reading aid, not a label the model uses.
