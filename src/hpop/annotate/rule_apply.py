"""Rule-based APPLY annotator — context-aware silver CPA labels over a full corpus (no API).

A transparent, deterministic stand-in for the LLM APPLY pass: maps each normalized event to a seed
CPA (rules/cpa_library_seed.json) using tool + command keywords + failure context + a little state
(was there an edit since the last run? -> VERIFY_FIX vs RUN_TEST_SUITE). Groups maximal contiguous
runs of the same CPA into one occurrence. Emits the SAME finalized schema as `opencode` so all
downstream tools (extract, render_from_opencode, consolidate) work unchanged.

NOT the LLM semantic open-coding (that needs a key + does finer contextual disambiguation). This is a
silver baseline to unblock modelling on all 100 trajectories now.

Usage:
    PYTHONPATH=src .venv/bin/python -m hpop.annotate.rule_apply \
        --input data/interim/swe_rebench/pilot100.jsonl \
        --library rules/cpa_library_seed.json \
        --output data/annotated/swe_rebench/cpa_rule
"""
from __future__ import annotations

import argparse
import json
import os
import re

THINK = {"think"}
_CODE_EXT = (".py", ".js", ".ts", ".java", ".go", ".rb", ".c", ".cpp", ".h", ".rs", ".md", ".txt",
            ".cfg", ".toml", ".yaml", ".yml", ".json", ".ini", ".rst")


def _has_ext(path):
    return any(path.endswith(e) for e in _CODE_EXT)


_DEBUG_RE = re.compile(r"print\(|logging\.|logger\.|console\.log|\bpdb\b|traceback\.print|sys\.stderr|warnings\.warn", re.I)


def classify(ev, edited_since_run):
    """Return (cpa_name or None, outcome). None => excluded non-action."""
    tn = ev.get("tool_name", ""); fam = ev.get("tool_family", "")
    args = ev.get("args") or {}
    cmd = ((args.get("command") or ev.get("command") or "")).lower()   # full bash command if present
    obs = (ev.get("observation") or "")
    fail = bool(ev.get("after_fail"))
    if tn in THINK or fam == "think":
        return None, None
    if tn in ("finish",) or fam == "submit":
        return "SUBMIT_SOLUTION", "SUCCESS"

    if tn == "str_replace_editor":
        m = re.search(r"\((.*)\)", ev.get("command") or "")
        path = (m.group(1) if m else "").strip()
        sub = cmd.split("(")[0]
        if sub == "view":
            if fail:
                return "DIAGNOSE_FAILURE", "UNKNOWN"
            return ("READ_SOURCE", "SUCCESS") if _has_ext(path) else ("EXPLORE_REPOSITORY", "SUCCESS")
        if sub == "create":
            if re.search(r"test", path):
                return "WRITE_TEST", "SUCCESS"
            if re.search(r"repro|reproduce|debug|bug", path):
                return "WRITE_REPRODUCTION_SCRIPT", "SUCCESS"
            return "EDIT_SOURCE", "SUCCESS"
        # str_replace / insert / edit — use the edit content to disambiguate
        new = str(args.get("new_str") or args.get("insert_str") or args.get("file_text") or "")
        if _DEBUG_RE.search(new) and not re.search(r"test", path):
            return "ADD_DEBUG_INSTRUMENTATION", "SUCCESS"
        if re.search(r"test", path):
            return "WRITE_TEST", "SUCCESS"
        return "EDIT_SOURCE", "SUCCESS"

    if fam in ("execute", "test", "search", "install") or tn == "execute_bash":
        if re.search(r"\bpip install|conda install|poetry add|apt-get|apt install|npm install|pip3 install", cmd):
            return "INSTALL_DEPENDENCY", ("FAILURE" if fail else "SUCCESS")
        # --- induced CPAs (appearance-threshold open coding; iter_induct.py) ---
        if re.search(r"pip\s+(list|freeze|show)|conda\s+(list|info)|python\s+--version|python\s+-v\b|printenv|\buname\b", cmd):
            return "INSPECT_ENVIRONMENT", "SUCCESS"
        if re.search(r"\bflake8\b|\bpylint\b|\bruff\b(?!\s+format)|\bpycodestyle\b|\bpyflakes\b|\beslint\b", cmd):
            return "RUN_LINTER", ("FAILURE" if fail else "SUCCESS")
        if re.search(r"\bblack\b|\bisort\b|\bautopep8\b|\byapf\b|\bprettier\b|ruff\s+format|\bgofmt\b|\brustfmt\b", cmd):
            return "FORMAT_CODE", "SUCCESS"
        if re.search(r"\btimeit\b|cprofile|\bpstats\b|--durations|\bhotshot\b|perf_counter", cmd):
            return "MEASURE_PERFORMANCE", "SUCCESS"
        if re.search(r"\bpdb\b|breakpoint\(\)|\bipython\b|python\s+-i\b|import\s+pdb", cmd):
            return "INTERACTIVE_DEBUG", "SUCCESS"
        if re.search(r"\bdiff\s+\S+\s+\S+|\bcmp\s+\S+\s+\S|difflib|deepdiff", cmd) and not re.search(r"git\s+diff", cmd):
            return "COMPARE_OUTPUT", "SUCCESS"
        if re.search(r"git\s+(add|commit|branch|tag)\b", cmd):
            return "COMMIT_CHANGES", "SUCCESS"
        if re.search(r"\bmkdir\b|\btouch\b|\bmv\s+\S|\bcp\s+\S|\bchmod\b|\bln\s+-s", cmd):
            return "MANAGE_FILESYSTEM", "SUCCESS"
        if re.search(r"\bexport\s+\w+=|source\s+\S*activate|conda\s+activate|virtualenv\b|python\s+-m\s+venv|\bpyenv\b|pythonpath=", cmd) \
                and not re.search(r"pytest|python\s+\S*\.py|python\s+-c|python\s+-m\s+(?!venv)", cmd):
            return "CONFIGURE_ENVIRONMENT", "SUCCESS"
        # --- end induced ---
        if re.search(r"\bgrep\b|\bfind\b|\brg\b|\bls\b|find_file|search_dir", cmd):
            return "LOCATE_CODE", "SUCCESS"
        if re.search(r"\bcat\b|\bhead\b|\btail\b|\bless\b", cmd):
            return ("DIAGNOSE_FAILURE", "UNKNOWN") if fail else ("READ_SOURCE", "SUCCESS")
        if re.search(r"\brm\b|rm -|del ", cmd):
            return "CLEANUP_ARTIFACTS", "SUCCESS"
        if re.search(r"git checkout|git restore|git stash|undo_edit|git reset", cmd):
            return "REVERT_CHANGE", "SUCCESS"
        if re.search(r"\bgit log\b|\bgit blame\b|git show\s+[0-9a-f]{6,}|git diff\s+[0-9a-f]{6,}|git diff\s+head[~^]", cmd):
            return "INSPECT_HISTORY", "SUCCESS"
        if re.search(r"git diff|git status|git show", cmd):
            return "INSPECT_CHANGES", "SUCCESS"
        if re.search(r"\bmypy\b|\bpyright\b|\bpyre\s+check\b|\btsc\b", cmd):
            return "CHECK_TYPES", ("FAILURE" if fail else "SUCCESS")
        if re.search(r"\bmake\b|cmake|cargo build|go build|\bmvn\b|gradle|setup\.py build|python setup\.py", cmd):
            return "BUILD_PROJECT", ("FAILURE" if fail else "SUCCESS")
        is_test = bool(re.search(r"\bpytest\b|tox|unittest|nosetests|python -m pytest", cmd))
        outcome = "FAILURE" if fail else "SUCCESS"
        if is_test:
            return ("VERIFY_FIX" if edited_since_run else "RUN_TEST_SUITE"), outcome
        # generic program / script execution
        return ("VERIFY_FIX" if edited_since_run else "REPRODUCE_ISSUE"), outcome
    return None, None  # unknown -> exclude


def annotate(trace, libdefs):
    eid = lambda i: "e{:04d}".format(i)
    raw = []
    edited_since_run = False
    for ev in trace.get("action_tokens", []):
        cpa, outcome = classify(ev, edited_since_run)
        raw.append((ev, cpa, outcome))
        if cpa in ("EDIT_SOURCE", "WRITE_TEST", "WRITE_REPRODUCTION_SCRIPT", "ADD_DEBUG_INSTRUMENTATION"):
            edited_since_run = True
        elif cpa in ("RUN_TEST_SUITE", "VERIFY_FIX", "REPRODUCE_ISSUE"):
            edited_since_run = False

    occ, excluded = [], []
    i = 0
    while i < len(raw):
        ev, cpa, outcome = raw[i]
        if cpa is None:
            excluded.append({"event_id": eid(ev["i"]), "reason": "NON_ACTION_METADATA"})
            i += 1
            continue
        # merge maximal contiguous run of the same CPA into one occurrence
        j = i
        ids = []
        outc = outcome
        while j < len(raw) and raw[j][1] == cpa:
            ids.append(raw[j][0]["i"])
            if raw[j][2] == "FAILURE":
                outc = "FAILURE"
            j += 1
        in_lib = cpa in libdefs
        occ.append({
            "occurrence_id": "{}::CPA_{:04d}".format(trace["trace_id"], len(occ) + 1),
            "source_event_ids": [eid(k) for k in ids], "start_event_id": eid(ids[0]), "end_event_id": eid(ids[-1]),
            "decision": "MATCH_EXISTING" if in_lib else "PROPOSE_NEW",
            "canonical_label": cpa if in_lib else None, "candidate_label": cpa,
            "definition": libdefs.get(cpa, ""), "procedural_function": libdefs.get(cpa, cpa.lower()),
            "input_artifacts": [], "output_artifacts": [],
            "state_before": None, "state_after": None, "outcome": outc,
            "boundary_confidence": 0.7, "label_confidence": 0.75,
            "evidence": [(raw[i][0].get("command") or "")[:80]], "ambiguity": None,
            "review_required": (not in_lib),
        })
        i = j
    return {
        "trajectory_id": trace.get("trajectory_id"), "mode": "APPLY", "library_version": "seed-v0-rule",
        "cpa_instances": occ, "excluded_events": excluded,
        "candidate_library_updates": [], "review_queue": [],
        "quality_checks": {"action_event_coverage_complete": True, "spans_sorted": True,
                           "spans_non_overlapping": True, "repeated_occurrences_preserved": True,
                           "skill_labels_inferred": False, "unassigned_action_event_ids": []},
        "_instance_id": trace.get("trace_id"), "_repo": trace.get("repo"),
        "_resolved": trace.get("resolved"), "_annotator": "rule_apply_seed-v0",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rule-based APPLY annotator (silver CPA labels).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--output", required=True, help="prefix (.jsonl + .cpa_instances.jsonl)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    lib = json.load(open(args.library))
    libdefs = {c["name"]: c.get("definition", "") for c in lib}
    traces = [json.loads(l) for l in open(args.input) if l.strip()]
    if args.limit:
        traces = traces[: args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    ff = open(args.output + ".jsonl", "w", encoding="utf-8")
    fc = open(args.output + ".cpa_instances.jsonl", "w", encoding="utf-8")
    from collections import Counter
    freq = Counter(); n_occ = 0; n_new = 0; n_excl = 0
    for t in traces:
        ann = annotate(t, libdefs)
        ff.write(json.dumps(ann, ensure_ascii=False) + "\n")
        for c in ann["cpa_instances"]:
            fc.write(json.dumps(dict(c, trajectory_id=ann["trajectory_id"], instance_id=ann["_instance_id"],
                                     repo=ann["_repo"], annotator=ann["_annotator"]), ensure_ascii=False) + "\n")
            freq[c["candidate_label"]] += 1
            n_occ += 1; n_new += (c["decision"] == "PROPOSE_NEW")
        n_excl += len(ann["excluded_events"])
    ff.close(); fc.close()
    print("annotated {} trajectories -> {}.jsonl".format(len(traces), args.output))
    print("CPA occurrences: {}  | PROPOSE_NEW: {}  | excluded non-action: {}".format(n_occ, n_new, n_excl))
    print("CPA frequency:")
    for k, v in freq.most_common():
        print("   {:>5}  {}".format(v, k))


if __name__ == "__main__":
    main()
