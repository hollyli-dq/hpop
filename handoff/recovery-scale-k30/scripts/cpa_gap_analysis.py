"""CPA gap analysis — find procedural functions the current dictionary does NOT capture.

Offline stand-in for an LLM INDUCE pass over a fresh batch. For every event we (a) ask the existing
rule classifier what CPA it would assign, then (b) test the command/edit against a set of CANDIDATE
new-CPA probes (linting, type-checking, formatting, version control, env config, filesystem ops,
interactive runtime, patch application, doc reading). When a probe fires on an event that the current
rules either EXCLUDE or force into a generic bucket (REPRODUCE_ISSUE / READ_SOURCE), that's evidence
the 17-CPA dictionary is too coarse there.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/cpa_gap_analysis.py \
        --input data/interim/swe_rebench/pilot100_b.jsonl \
        --library rules/cpa_dictionary.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

from hpop.annotate.rule_apply import classify

# Candidate NEW CPAs, each a probe over the full command (or edit content). Ordered: first match wins.
CANDIDATES = [
    ("RUN_LINTER", re.compile(r"\bflake8\b|\bpylint\b|\bruff\b(?!.*format)|\bpycodestyle\b|\bpyflakes\b|\beslint\b", re.I)),
    ("CHECK_TYPES", re.compile(r"\bmypy\b|\bpyright\b|\bpyre\b|\btsc\b|type[\s_-]?check", re.I)),
    ("FORMAT_CODE", re.compile(r"\bblack\b|\bisort\b|\bautopep8\b|\byapf\b|\bprettier\b|ruff\s+format|gofmt|rustfmt", re.I)),
    ("COMMIT_CHANGES", re.compile(r"git\s+(add|commit|branch|checkout\s+-b|stash\s+pop|tag)\b", re.I)),
    ("APPLY_PATCH", re.compile(r"git\s+apply\b|\bpatch\s+-|\bgit\s+am\b|apply_patch", re.I)),
    ("CONFIGURE_ENVIRONMENT", re.compile(r"\bexport\s+\w+=|conda\s+(activate|create)|source\s+.*activate|virtualenv|python\s+-m\s+venv|pyenv\b|\bsource\s+\S+\.sh", re.I)),
    ("MANAGE_FILESYSTEM", re.compile(r"\bmkdir\b|\btouch\b|\b(mv|cp)\s+\S|chmod\b|\bln\s+-s", re.I)),
    ("INSPECT_HISTORY", re.compile(r"git\s+(log|blame|show\s+[0-9a-f]{6,}|diff\s+HEAD~)", re.I)),
    ("INTERACTIVE_RUNTIME", re.compile(r"\bpdb\b|breakpoint\(\)|ipython\b|python\s+-i\b|import\s+pdb", re.I)),
    ("READ_DOCUMENTATION", re.compile(r"view\((?:[^)]*/)?(?:readme|changelog|contributing|docs?/|.*\.(?:md|rst))", re.I)),
    ("MEASURE_PERFORMANCE", re.compile(r"\btimeit\b|\bbenchmark\b|\bcProfile\b|\bperf\s|--durations", re.I)),
    ("COMPARE_OUTPUT", re.compile(r"\bdiff\s+\S+\s+\S+|\bcmp\b|difflib", re.I)),
]

GENERIC_OR_EXCLUDED = {None, "REPRODUCE_ISSUE", "READ_SOURCE", "EXPLORE_REPOSITORY"}


def full_text(ev):
    a = ev.get("args") or {}
    parts = [str(a.get("command") or ev.get("command") or "")]
    for k in ("new_str", "insert_str", "file_text"):
        if a.get(k):
            parts.append(str(a[k])[:200])
    return " \n ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Find CPAs missing from the current dictionary.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--library", required=True)
    args = ap.parse_args(argv)

    lib = json.load(open(args.library))
    known = {c["name"] for c in lib}

    traces = [json.loads(l) for l in open(args.input) if l.strip()]
    fired = Counter()             # candidate -> total events that match
    fired_in_gap = Counter()      # candidate -> events the current rules miss (generic/excluded)
    current_bucket = defaultdict(Counter)   # candidate -> Counter of what rule_apply currently calls it
    examples = defaultdict(list)
    repos_for = defaultdict(set)

    n_ev = 0
    for t in traces:
        edited = False
        for ev in t.get("action_tokens", []):
            n_ev += 1
            cur, _ = classify(ev, edited)
            if cur in ("EDIT_SOURCE", "WRITE_TEST", "WRITE_REPRODUCTION_SCRIPT", "ADD_DEBUG_INSTRUMENTATION"):
                edited = True
            elif cur in ("RUN_TEST_SUITE", "VERIFY_FIX", "REPRODUCE_ISSUE"):
                edited = False
            text = full_text(ev)
            for name, rx in CANDIDATES:
                if rx.search(text):
                    fired[name] += 1
                    current_bucket[name][cur] += 1
                    repos_for[name].add(t.get("repo"))
                    if cur in GENERIC_OR_EXCLUDED:
                        fired_in_gap[name] += 1
                        if len(examples[name]) < 3:
                            examples[name].append((cur, text.replace("\n", " ")[:90]))
                    break

    print("scanned {} trajectories / {} events  (current dictionary = {} CPAs)\n".format(
        len(traces), n_ev, len(known)))
    print("CANDIDATE NEW CPA           fires  in-gap  repos   currently-labeled-as")
    print("-" * 92)
    for name, _ in CANDIDATES:
        if not fired[name]:
            continue
        cur = ", ".join("{}={}".format(k or "EXCLUDED", v) for k, v in current_bucket[name].most_common(3))
        print("{:<26} {:>5}  {:>6}  {:>5}   {}".format(
            name, fired[name], fired_in_gap[name], len(repos_for[name]), cur))
    print("\nEVIDENCE (events the current 17 CPAs mislabel or drop):")
    for name, _ in CANDIDATES:
        if fired_in_gap[name] >= 2:
            print("\n  {}  ({} gap-events, {} repos)".format(name, fired_in_gap[name], len(repos_for[name])))
            for cur, ex in examples[name]:
                print("    [{}]  {}".format(cur or "EXCLUDED", ex))


if __name__ == "__main__":
    main()
