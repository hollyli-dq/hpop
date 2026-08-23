"""Layer B · Step 1a — Physical Normalization (action segmentation).

Canonicalise each raw trace event into an ACTION TOKEN with four fields, collapse repeated identical
actions (the "5-second window" dedup), and normalise tool names. This step applies NO Layer-A (phase) or
Layer-B (CPA/dependency) rule — it is pure data cleaning + format unification, and it is the substrate
that CPA Abstraction (Step 1b) consumes.

Theory / standards (this step is NOT part of DISRPT; it is our agent-trace adaptation):
  • ISO 24612:2012 — Linguistic Annotation Framework (LAF): our data model is a standoff multi-layer
    annotation over a normalized base segmentation — conformant to the international standard.
  • Bird & Liberman (1999) — multi-layer annotation is well-founded as a graph over a common base.

Action token (the four canonical fields):
  action_type  — the physical act           (view / edit / create / run / submit / think / navigate)
  tool_family  — the tool class             (editor / shell / notebook / browser / meta)
  agent_id     — which agent executed it     (single-agent OpenHands ⇒ constant "A0")
  artifact_id  — the object operated on      (normalized <role:id>; reused by ground.py)

Adaptation note: OpenHands SWE-rebench events are ORDERED but carry no wall-clock timestamps, so the
"merge identical actions within 5 s" rule becomes "merge ADJACENT identical action tokens" (same
action_type · tool_family · agent_id · artifact_id · command). Flagged for the coauthor.

The raw signal needed by Step 1b (command, observation, args, after_fail, original tool_name) is retained
on each token as attached evidence (LAF: annotation layers reference the base) — re-indexed contiguously.

Usage:
    PYTHONPATH=src .venv/bin/python -m hpop.annotate.normalize \
        --input data/interim/swe_rebench/pilot500.jsonl \
        --output data/interim/swe_rebench/tokens500.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

from hpop.extract.ground import _artifact_of

# raw OpenHands tool name -> canonical tool_family
_TOOL_FAMILY = {
    "str_replace_editor": "editor",
    "execute_bash": "shell",
    "run_ipython": "notebook",
    "execute_ipython_cell": "notebook",
    "finish": "meta",
    "think": "meta",
}

# str_replace_editor sub-command -> physical action_type
_EDITOR_ACT = {"view": "view", "create": "create", "str_replace": "edit", "insert": "edit",
               "undo_edit": "undo"}


def _action_type(tool, args):
    a = args or {}
    if tool == "str_replace_editor":
        return _EDITOR_ACT.get(a.get("command", ""), "edit")
    if tool in ("execute_bash", "run_ipython", "execute_ipython_cell"):
        return "run"
    if tool == "finish":
        return "submit"
    if tool == "think":
        return "think"
    if "browse" in (tool or ""):
        return "navigate"
    return "act"


def _tool_family(tool):
    if tool in _TOOL_FAMILY:
        return _TOOL_FAMILY[tool]
    if "browse" in (tool or ""):
        return "browser"
    return "other"


def normalize_trajectory(trace):
    """Raw trace -> list of canonical action tokens (deduped, re-indexed)."""
    agent_default = trace.get("agent_id") or "A0"
    toks = []
    for ev in trace.get("action_tokens", []):
        tool = ev.get("tool_name") or ev.get("action_type") or ""
        art = _artifact_of(ev)
        artifact_id = "{}:{}".format(*art) if art else None
        toks.append({
            "action_type": _action_type(tool, ev.get("args")),
            "tool_family": _tool_family(tool),
            "agent_id": ev.get("agent_id") or agent_default,
            "artifact_id": artifact_id,
            # retained raw signal for Step 1b (CPA abstraction)
            "tool_name": tool, "args": ev.get("args"), "command": ev.get("command"),
            "observation": ev.get("observation"), "after_fail": bool(ev.get("after_fail")),
            "_src": [ev.get("i")],
        })

    # collapse ADJACENT identical action tokens (the 5s-window dedup, adjacency-based)
    merged = []
    for t in toks:
        if merged:
            p = merged[-1]
            same = (p["action_type"] == t["action_type"] and p["tool_family"] == t["tool_family"]
                    and p["agent_id"] == t["agent_id"] and p["artifact_id"] == t["artifact_id"]
                    and (p["command"] or "") == (t["command"] or ""))
            if same:
                p["_src"].extend(t["_src"]); p["after_fail"] = p["after_fail"] or t["after_fail"]
                continue
        merged.append(t)

    for i, t in enumerate(merged):
        t["i"] = i
        t["source_event_ids"] = ["e{:04d}".format(j) for j in t.pop("_src")]
    out = dict(trace)
    out["action_tokens"] = merged
    out["num_action_tokens"] = len(merged)
    out["num_raw_events"] = len(toks)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Layer B Step 1a — physical normalization to action tokens.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    from collections import Counter
    raw_tot = norm_tot = n = 0
    fam = Counter(); act = Counter(); art = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for l in open(args.input):
            if not l.strip():
                continue
            t = normalize_trajectory(json.loads(l))
            fout.write(json.dumps(t, ensure_ascii=False) + "\n")
            n += 1; raw_tot += t["num_raw_events"]; norm_tot += t["num_action_tokens"]
            for tok in t["action_tokens"]:
                fam[tok["tool_family"]] += 1; act[tok["action_type"]] += 1
                art += bool(tok["artifact_id"])
    print("normalized {} trajectories -> {}".format(n, args.output))
    print("events {} raw -> {} action tokens  ({} merged, {:.0f}% kept)".format(
        raw_tot, norm_tot, raw_tot - norm_tot, 100.0 * norm_tot / max(raw_tot, 1)))
    print("artifact_id present on {:.0f}% of tokens".format(100.0 * art / max(norm_tot, 1)))
    print("action_type : " + ", ".join("{}={}".format(k, v) for k, v in act.most_common()))
    print("tool_family : " + ", ".join("{}={}".format(k, v) for k, v in fam.most_common()))


if __name__ == "__main__":
    main()
