"""Ingest SWE-agent trajectories into normalized action-token traces.

Source: HuggingFace `nebius/SWE-agent-trajectories` (and similar). Each row has:
  instance_id, model_name, target(bool), trajectory(list of {role, text, ...}),
  exit_status, generated_patch, eval_logs.

An `ai` turn carries a thought + an action command (in a fenced block); a `user` turn carries the
tool observation (traceback / file content / shell output). We parse each `ai` action into an
action token (verb + tool_family + cmd), flag tokens that follow a FAILED observation (the seed of
REPAIR / VERIFY structure), and keep the issue text as the goal.

Fetched lazily via the HF datasets-server `/rows` API (no full download / no `datasets` lib).
Use --resolved to keep only target=True (successful) trajectories.

Usage:
    PYTHONPATH=src .venv/bin/python -m hpop.ingest.swe \
        --dataset nebius/SWE-agent-trajectories --limit 30 --resolved \
        --output data/interim/swe/trajectories.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.parse
import urllib.request

CMD_FAMILY = {
    "open": "read", "goto": "read", "cat": "read", "scroll_down": "read", "scroll_up": "read",
    "search_dir": "search", "search_file": "search", "find_file": "search", "grep": "search",
    "ls": "search", "find": "search",
    "edit": "edit", "create": "edit", "insert": "edit",
    "python": "execute", "python3": "execute", "pytest": "execute", "bash": "execute",
    "make": "execute", "tox": "execute",
    "submit": "submit",
}
_FENCE = re.compile(r"```(?:bash)?\s*(.*?)```", re.S)
_FAIL = re.compile(r"traceback|error|no such|not found|failed|exception", re.I)


def parse_action(ai_text):
    blocks = _FENCE.findall(ai_text or "")
    cmd = blocks[-1].strip() if blocks else ""
    if not cmd:
        for ln in reversed((ai_text or "").splitlines()):
            if ln.strip():
                cmd = ln.strip()
                break
    verb = (cmd.split() or ["?"])[0]
    fam = CMD_FAMILY.get(verb, "execute" if verb.endswith(".py") or "/" in verb else "other")
    return verb, fam, cmd


def trajectory_to_trace(row):
    traj = row.get("trajectory", [])
    issue = next((m.get("text", "") for m in traj if m.get("role") == "user" and m.get("text")), "")
    goal = issue.split("ISSUE:")[-1].strip().splitlines()[0][:140] if "ISSUE" in issue else issue[:140]
    tokens, last_fail = [], False
    for m in traj:
        if m.get("role") == "ai":
            verb, fam, cmd = parse_action(m.get("text", ""))
            tokens.append({
                "i": len(tokens), "action_type": verb, "tool_family": fam,
                "agent_id": "agent", "artifact_id": cmd[:50], "cmd": cmd[:160],
                "after_fail": last_fail,
            })
        elif m.get("role") == "user":
            last_fail = bool(_FAIL.search(m.get("text", "") or ""))
    return {
        "trace_id": row.get("instance_id"), "source": "swe-agent",
        "model_name": row.get("model_name"), "target": bool(row.get("target")),
        "exit_status": row.get("exit_status"), "instructor_goal": goal,
        "num_action_tokens": len(tokens), "action_tokens": tokens,
        "patch_chars": len(row.get("generated_patch") or ""),
    }


def _get(url, tries=5):
    """GET with exponential backoff on 429/5xx (in-process sleep, not the shell `sleep`)."""
    import time
    delay = 2.0
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and k < tries - 1:
                time.sleep(delay); delay *= 2; continue
            raise
    return None


def fetch_rows(dataset, want, resolved, page=5, max_pages=120, split="train"):
    """Page the datasets-server /rows API; dedup by instance_id; until `want` unique collected."""
    import time
    base = "https://datasets-server.huggingface.co/rows"
    enc = urllib.parse.quote(dataset, safe="")
    got, seen, offset = [], set(), 0
    for _ in range(max_pages):
        url = "{}?dataset={}&config=default&split={}&offset={}&length={}".format(base, enc, split, offset, page)
        try:
            data = _get(url)
        except Exception as e:
            print("  fetch error at offset {}: {}".format(offset, str(e)[:60]))
            offset += page
            time.sleep(1.0)
            continue
        rows = (data or {}).get("rows", [])
        if not rows:
            break
        for rw in rows:
            tr = trajectory_to_trace(rw["row"])
            if tr["trace_id"] in seen:
                continue
            if resolved and not tr["target"]:
                continue
            if tr["num_action_tokens"] < 4:
                continue
            seen.add(tr["trace_id"])
            got.append(tr)
            if len(got) >= want:
                return got
        offset += page
        time.sleep(0.6)  # be polite to the datasets-server
    return got


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ingest SWE-agent trajectories -> normalized traces.")
    ap.add_argument("--dataset", default="nebius/SWE-agent-trajectories")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--resolved", action="store_true", help="keep only target=True trajectories")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    print("fetching up to {} {}trajectories from {} ...".format(
        args.limit, "resolved " if args.resolved else "", args.dataset))
    traces = fetch_rows(args.dataset, args.limit, args.resolved)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    import statistics as st
    from collections import Counter
    fam = Counter()
    for t in traces:
        for tok in t["action_tokens"]:
            fam[tok["tool_family"]] += 1
    acts = [t["num_action_tokens"] for t in traces]
    repairs = sum(1 for t in traces for tok in t["action_tokens"] if tok["after_fail"])
    print("collected     : {} traces -> {}".format(len(traces), args.output))
    if traces:
        print("actions/trace : mean {:.1f} median {} (min {}, max {})".format(
            st.mean(acts), st.median(acts), min(acts), max(acts)))
        print("tool_family   : " + ", ".join("{}={}".format(k, v) for k, v in fam.most_common()))
        print("after_fail (REPAIR/VERIFY seed) tokens: {}".format(repairs))
        print("sample ids    : " + ", ".join(t["trace_id"] for t in traces[:6]))


if __name__ == "__main__":
    main()
