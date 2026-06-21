"""Ingest `nebius/SWE-rebench-openhands-trajectories` into normalized action-token traces.

OpenHands, role-separated. Each row: trajectory_id, instance_id, repo, trajectory (JSON-string of
messages), tools (JSON-string), model_patch (JSON-string), exit_status, resolved(0/1),
gen_tests_correct, pred_passes_gen_tests.

Events (per the collaborator spec): each `assistant` message's tool_call is an ACTION; the following
`tool` message is the OBSERVATION. Tool-call `arguments` are JSON stored as strings -> deserialized
first (the required first cleaning step). We keep: tool name, parsed args, observation, order, and
repo (repository state), plus a failure flag (observation looks like an error) — the REPAIR/VERIFY seed.

Pilot sampler: collect 50 resolved + 50 unresolved across many repos (diverse lengths).

Usage:
    PYTHONPATH=src .venv/bin/python -m hpop.ingest.swe_rebench \
        --resolved 50 --unresolved 50 --output data/interim/swe_rebench/pilot100.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.parse
import urllib.request

DATASET = "nebius/SWE-rebench-openhands-trajectories"
_FAIL = re.compile(r"traceback|error:|errno|no such|not found|failed|assertionerror|exit code [1-9]|exception", re.I)


def _family(tool_name, args):
    """Map an OpenHands tool call to a coarse tool_family (CONTEXT ONLY — CPAs are induced, not this)."""
    if tool_name == "think":
        return "think"
    if tool_name in ("finish",):
        return "submit"
    if tool_name == "str_replace_editor":
        cmd = (args or {}).get("command", "")
        return "read" if cmd == "view" else "edit"
    if tool_name in ("execute_bash", "run_ipython", "execute_ipython_cell"):
        c = ((args or {}).get("command") or (args or {}).get("code") or "").lower()
        if re.search(r"\bpytest\b|tox|unittest|python -m pytest|nosetests", c):
            return "test"
        if re.search(r"\bpip install|conda install|apt-get|poetry add|npm install", c):
            return "install"
        if re.search(r"\bgrep\b|\bfind\b|\bls\b|\brg\b|\bcat\b|\bhead\b|\btail\b", c):
            return "search"
        return "execute"
    if "browse" in tool_name or "web" in tool_name:
        return "browse"
    return "other"


def _short_cmd(tool_name, args):
    a = args or {}
    if tool_name == "str_replace_editor":
        return "{}({})".format(a.get("command", ""), a.get("path", ""))[:160]
    if tool_name in ("execute_bash", "run_ipython", "execute_ipython_cell"):
        return (a.get("command") or a.get("code") or "")[:160]
    if tool_name == "think":
        return "think"
    return json.dumps(a)[:160]


def parse_trajectory(row):
    msgs = json.loads(row["trajectory"]) if isinstance(row["trajectory"], str) else row["trajectory"]
    issue = next((m.get("content", "") for m in msgs if m.get("role") == "user" and m.get("content")), "")
    goal = re.sub(r"\s+", " ", issue).strip()[:160]

    # index tool observations by tool_call_id for pairing
    obs_by_id = {m.get("tool_call_id"): (m.get("content") or "")
                 for m in msgs if m.get("role") == "tool"}
    # also keep order of tool messages as a fallback
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]

    tokens, ti = [], 0
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {})
            name = fn.get("name", "?")
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                args = {"_raw": fn.get("arguments", "")}
            obs = obs_by_id.get(tc.get("id"))
            if obs is None and ti < len(tool_msgs):
                obs = tool_msgs[ti].get("content", "")
            ti += 1
            tokens.append({
                "i": len(tokens), "tool_name": name, "tool_family": _family(name, args),
                "command": _short_cmd(name, args), "args": args,
                "observation": re.sub(r"\s+", " ", (obs or ""))[:140],
                "after_fail": bool(_FAIL.search(obs or "")),
            })
    return goal, tokens


def row_to_trace(row):
    goal, tokens = parse_trajectory(row)
    patch = row.get("model_patch")
    if isinstance(patch, str):
        try:
            patch = json.loads(patch)
        except Exception:
            pass
    return {
        "trace_id": row.get("instance_id"), "trajectory_id": row.get("trajectory_id"),
        "source": "swe-rebench-openhands", "repo": row.get("repo"),
        "resolved": bool(int(row.get("resolved", 0))), "exit_status": row.get("exit_status"),
        "gen_tests_correct": row.get("gen_tests_correct"),
        "pred_passes_gen_tests": row.get("pred_passes_gen_tests"),
        "model_patch_chars": len(patch) if isinstance(patch, str) else 0,
        "instructor_goal": goal, "num_action_tokens": len(tokens), "action_tokens": tokens,
    }


def _get(url, tries=5):
    import time
    delay = 2.0
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and k < tries - 1:
                time.sleep(delay); delay *= 2; continue
            raise
    return None


def sample(n_res, n_unres, page=2, max_pages=400):
    import time
    enc = urllib.parse.quote(DATASET, safe="")
    base = "https://datasets-server.huggingface.co/rows"
    res, unres, seen = [], [], set()
    offset = 0
    for _ in range(max_pages):
        if len(res) >= n_res and len(unres) >= n_unres:
            break
        url = "{}?dataset={}&config=default&split=train&offset={}&length={}".format(base, enc, offset, page)
        try:
            data = _get(url)
        except Exception as e:
            print("  fetch error @{}: {}".format(offset, str(e)[:50])); offset += page; time.sleep(1.0); continue
        rows = (data or {}).get("rows", [])
        if not rows:
            break
        for rw in rows:
            try:
                tr = row_to_trace(rw["row"])
            except Exception as e:
                continue
            if tr["trace_id"] in seen or tr["num_action_tokens"] < 4:
                continue
            seen.add(tr["trace_id"])
            if tr["resolved"] and len(res) < n_res:
                res.append(tr)
            elif not tr["resolved"] and len(unres) < n_unres:
                unres.append(tr)
        offset += page
        time.sleep(0.5)
        if (offset // page) % 10 == 0:
            print("  ...scanned offset {} | resolved {}/{} unresolved {}/{} | repos {}".format(
                offset, len(res), n_res, len(unres), n_unres, len({t['repo'] for t in res + unres})))
    return res + unres


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ingest SWE-rebench-openhands trajectories (pilot sample).")
    ap.add_argument("--resolved", type=int, default=50)
    ap.add_argument("--unresolved", type=int, default=50)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    print("sampling {} resolved + {} unresolved from {} ...".format(args.resolved, args.unresolved, DATASET))
    traces = sample(args.resolved, args.unresolved)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    import statistics as st
    from collections import Counter
    r = sum(1 for t in traces if t["resolved"]); fam = Counter()
    for t in traces:
        for x in t["action_tokens"]:
            fam[x["tool_family"]] += 1
    acts = [t["num_action_tokens"] for t in traces]
    print("collected : {} ({} resolved / {} unresolved) across {} repos -> {}".format(
        len(traces), r, len(traces) - r, len({t["repo"] for t in traces}), args.output))
    if traces:
        print("actions/trace: mean {:.1f} median {} (min {} max {})".format(
            st.mean(acts), st.median(acts), min(acts), max(acts)))
        print("tool_family : " + ", ".join("{}={}".format(k, v) for k, v in fam.most_common()))


if __name__ == "__main__":
    main()
