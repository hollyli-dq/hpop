"""Iterative CPA induction — appearance-threshold (>=1) open coding over batches of 100.

Policy (coauthor, 2026-06-25): drop the >=4-repo gate. Process trajectory batches in order; ANY
distinct procedural function that APPEARS becomes a CPA. The dictionary grows monotonically; each batch
can surface more. Still subject to the CPA label rules (portable verb+object, not a one-off entity).

Mechanism (offline stand-in for the LLM INDUCE pass): a CPA "appears in batch b" if the rule classifier
emits it OR a new-function probe matches an event. We record the FIRST batch each CPA appears in, so the
output is a growth curve plus the newly-induced entries to merge into rules/cpa_dictionary.json.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/iter_induct.py \
        --library rules/cpa_dictionary.json \
        --batches data/interim/swe_rebench/pilot100.jsonl data/interim/swe_rebench/pilot100_b.jsonl ... \
        --emit rules/cpa_dictionary_proposed.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, OrderedDict

from hpop.annotate.rule_apply import classify

# New procedural functions beyond the base set. (name, phase, regex over full command+edit text,
# definition, distinguish). Appearance (>=1 match anywhere) is sufficient to induct.
NEW_CPAS = [
    ("RUN_LINTER", "reproduce_diagnose", r"\bflake8\b|\bpylint\b|\bruff\b(?!\s+format)|\bpycodestyle\b|\bpyflakes\b|\beslint\b",
     "Run a style/lint checker (flake8, pylint, ruff, pycodestyle, eslint) to surface style/lint violations.",
     "style/lint scan (here) vs static type checking (CHECK_TYPES) vs running tests (RUN_TEST_SUITE)"),
    ("FORMAT_CODE", "modify", r"\bblack\b|\bisort\b|\bautopep8\b|\byapf\b|\bprettier\b|ruff\s+format|\bgofmt\b|\brustfmt\b",
     "Run an auto-formatter (black, isort, autopep8, prettier, gofmt) to reformat code mechanically.",
     "mechanical reformat (here) vs a behavioural fix (EDIT_SOURCE) vs restructuring (REFACTOR_CODE)"),
    ("CONFIGURE_ENVIRONMENT", "modify", r"\bexport\s+\w+=|conda\s+(activate|create)|source\s+\S*activate|virtualenv\b|python\s+-m\s+venv|\bpyenv\b|PYTHONPATH=",
     "Set environment variables / activate or create a virtualenv/conda env to make code or tests runnable.",
     "preparing the runtime env (here) vs installing packages (INSTALL_DEPENDENCY) vs building (BUILD_PROJECT)"),
    ("MANAGE_FILESYSTEM", "modify", r"\bmkdir\b|\btouch\b|\bmv\s+\S|\bcp\s+\S|\bchmod\b|\bln\s+-s",
     "Manipulate files/dirs in the workspace (mkdir, mv, cp, touch, chmod, symlink) — not a source edit.",
     "filesystem op (here) vs editing file contents (EDIT_SOURCE) vs deleting scratch (CLEANUP_ARTIFACTS)"),
    ("COMMIT_CHANGES", "verify_finalize", r"git\s+(add|commit|branch|tag)\b",
     "Record changes in version control (git add/commit/branch/tag).",
     "committing/branching (here) vs undoing (REVERT_CHANGE) vs viewing the diff (INSPECT_CHANGES)"),
    ("MEASURE_PERFORMANCE", "reproduce_diagnose", r"\btimeit\b|cProfile|\bpstats\b|--durations|\bhotshot\b|perf_counter",
     "Profile or benchmark code to measure timing/performance.",
     "measuring speed (here) vs checking correctness via tests (RUN_TEST_SUITE)"),
    ("COMPARE_OUTPUT", "reproduce_diagnose", r"\bdiff\s+\S+\s+\S+|\bcmp\s+\S+\s+\S|difflib|deepdiff",
     "Compare two outputs/files to find differences (diff, cmp, difflib).",
     "comparing two artifacts (here) vs reviewing one's own git diff (INSPECT_CHANGES)"),
    ("INTERACTIVE_DEBUG", "reproduce_diagnose", r"\bpdb\b|breakpoint\(\)|\bipython\b|python\s+-i\b|import\s+pdb",
     "Drop into an interactive debugger/REPL (pdb, ipython, python -i) to inspect runtime state.",
     "interactive runtime inspection (here) vs adding print/log statements (ADD_DEBUG_INSTRUMENTATION)"),
    ("READ_DOCUMENTATION", "orient", r"view\([^)]*(?:readme|changelog|contributing|license|/docs?/|/doc/)[^)]*\)|view\([^)]*\.(?:md|rst)\)",
     "Read project documentation (README, CHANGELOG, docs/, .md/.rst) rather than source code.",
     "reading docs (here) vs reading source/tests (READ_SOURCE)"),
    ("APPLY_PATCH", "modify", r"git\s+apply\b|\bpatch\s+-p|\bgit\s+am\b|apply_patch",
     "Apply a patch/diff to the working tree (git apply, patch -p, git am).",
     "applying a prepared patch (here) vs hand-editing source (EDIT_SOURCE)"),
    ("INSPECT_ENVIRONMENT", "orient", r"pip\s+(list|freeze|show)|conda\s+list|python\s+--version|python\s+-V\b|printenv|conda\s+info|\buname\b",
     "Inspect the environment / installed packages / interpreter version (pip list/freeze, python --version, uname).",
     "checking what's installed/configured (here) vs reading repo source (READ_SOURCE)"),
]
NEW_RX = [(n, p, re.compile(rx, re.I), d, dist) for (n, p, rx, d, dist) in NEW_CPAS]


def full_text(ev):
    a = ev.get("args") or {}
    parts = [str(a.get("command") or ev.get("command") or "")]
    for k in ("new_str", "insert_str", "file_text"):
        if a.get(k):
            parts.append(str(a[k])[:200])
    return " \n ".join(parts)


def labels_in_batch(path):
    """Return {cpa_name: (count, set(repos))} appearing in this batch (base rules + new probes)."""
    present = {}
    for l in open(path):
        if not l.strip():
            continue
        t = json.loads(l)
        repo = t.get("repo")
        edited = False
        for ev in t.get("action_tokens", []):
            cur, _ = classify(ev, edited)
            if cur in ("EDIT_SOURCE", "WRITE_TEST", "WRITE_REPRODUCTION_SCRIPT", "ADD_DEBUG_INSTRUMENTATION"):
                edited = True
            elif cur in ("RUN_TEST_SUITE", "VERIFY_FIX", "REPRODUCE_ISSUE"):
                edited = False
            if cur:
                c, rs = present.get(cur, (0, set())); present[cur] = (c + 1, rs | {repo})
            text = full_text(ev)
            for name, _ph, rx, _d, _dist in NEW_RX:
                if rx.search(text):
                    c, rs = present.get(name, (0, set())); present[name] = (c + 1, rs | {repo})
    return present


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--batches", nargs="+", required=True)
    ap.add_argument("--emit", default=None)
    args = ap.parse_args(argv)

    base = json.load(open(args.library))
    known = OrderedDict((c["name"], c) for c in base)
    new_def = {n: (ph, d, dist) for (n, ph, _rx, d, dist) in NEW_CPAS}

    first_seen = OrderedDict()      # cpa -> batch index of first appearance
    agg = {}                        # cpa -> (count, repos)
    print("ITERATIVE INDUCTION  (threshold = appears >=1)  starting dictionary = {} CPAs".format(len(known)))
    print("-" * 78)
    for bi, path in enumerate(args.batches, 1):
        present = labels_in_batch(path)
        for name, (cnt, repos) in present.items():
            if name not in agg:
                agg[name] = [0, set()]
            agg[name][0] += cnt; agg[name][1] |= repos
            if name not in first_seen:
                first_seen[name] = bi
        newly = [n for n in present if first_seen[n] == bi]
        induced = [n for n in newly if n not in known]
        total_dict = len(set(known) | set(first_seen))
        tag = "  +NEW: " + ", ".join(induced) if induced else ""
        print("after batch {} ({:<34}): {:>3} distinct seen | dictionary = {}{}".format(
            bi, path.split('/')[-1], len([n for n in first_seen if first_seen[n] <= bi]), total_dict, tag))

    induced_all = [n for n in first_seen if n not in known]
    print("\nINDUCED (new beyond the base {}):".format(len(known)))
    for n in sorted(induced_all, key=lambda x: -agg[x][0]):
        ph, d, dist = new_def.get(n, ("?", "", ""))
        print("  {:<24} first@batch{}  {:>4} occ / {:>2} repos   [{}]".format(
            n, first_seen[n], agg[n][0], len(agg[n][1]), ph))

    if args.emit and induced_all:
        entries = []
        for i, n in enumerate(induced_all):
            ph, d, dist = new_def[n]
            entries.append({"id": "CPA{:02d}".format(21 + i), "name": n, "phase": ph,
                            "definition": d, "distinguish": dist,
                            "input_artifacts": [], "output_artifacts": []})
        json.dump(entries, open(args.emit, "w"), indent=2)
        print("\nwrote {} proposed entries -> {}".format(len(entries), args.emit))


if __name__ == "__main__":
    main()
