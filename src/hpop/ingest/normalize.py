"""PDAF Step 1a — Physical Normalization.

Raw events (from `hpop.ingest.weblinx`) -> normalized **action tokens**:
``(action_type, tool_family, agent_id, artifact_id)`` plus a kept reference to the raw action.
Repeated actions are collapsed within a short window. WebLINX `chat` events carry no
timestamps, so we approximate the report's "5s window" by collapsing *consecutive identical*
``(action_type, artifact_id)`` tokens (turn adjacency).

These action tokens are the input to Step 1b (CPA Abstraction). Deterministic; no API needed.

Usage:
    PYTHONPATH=src python3 -m hpop.ingest.normalize \
        --input data/interim/weblinx/valid.jsonl \
        --output data/interim/weblinx/valid.normalized.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

# action_type (WebLINX verb) -> tool_family
TOOL_FAMILY = {
    "click": "pointer", "hover": "pointer", "scroll": "pointer",
    "type": "keyboard", "text_input": "keyboard", "fill_field": "keyboard", "change": "keyboard",
    "load": "navigation", "tabcreate": "navigation", "tabswitch": "navigation",
    "tabremove": "navigation", "go_back": "navigation",
    "copy": "clipboard", "paste": "clipboard",
    "submit": "form",
    "say": "dialogue",
}


def _artifact_id(verb, args):
    """Best-effort short artifact handle from the raw action args."""
    args = args or {}
    for k in ("uid", "url", "text", "value"):
        if args.get(k):
            v = str(args[k])
            return v[:80]
    return ""


def normalize_events(events):
    """events (raw, from weblinx ingest) -> list of normalized action tokens."""
    tokens = []
    for e in events:
        verb = e.get("verb", "unknown")
        args = e.get("args", {})
        tok = {
            "i": len(tokens),
            "raw_i": e.get("i"),
            "action_type": verb,
            "tool_family": TOOL_FAMILY.get(verb, "other"),
            "agent_id": e.get("speaker") if verb == "say" else "navigator",
            "artifact_id": _artifact_id(verb, args),
            "raw_action": e.get("action", ""),
        }
        if verb == "say":
            tok["utterance"] = e.get("utterance")
        # collapse: drop if identical (action_type, artifact_id) to the previous token
        # (and not a dialogue turn, which we always keep for context)
        if (tokens and verb != "say"
                and tokens[-1]["action_type"] == tok["action_type"]
                and tokens[-1]["artifact_id"] == tok["artifact_id"]):
            tokens[-1]["collapsed"] = tokens[-1].get("collapsed", 1) + 1
            continue
        tokens.append(tok)
    # reindex after collapse
    for idx, t in enumerate(tokens):
        t["i"] = idx
    return tokens


def normalize_trace(trace):
    out = dict(trace)
    out["action_tokens"] = normalize_events(trace.get("events", []))
    out["num_action_tokens"] = len(out["action_tokens"])
    out.pop("events", None)
    return out


def iter_traces(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description="PDAF Step 1a: normalize raw events into action tokens.")
    ap.add_argument("--input", required=True, help="interim per-trace jsonl (from hpop.ingest.weblinx)")
    ap.add_argument("--output", required=True, help="normalized output jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    traces = [normalize_trace(t) for t in iter_traces(args.input)]
    if args.limit is not None:
        traces = traces[: args.limit]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    raw = sum(len(t.get("events", [])) for t in iter_traces(args.input))
    norm = sum(t["num_action_tokens"] for t in traces)
    fams = {}
    for t in traces:
        for tok in t["action_tokens"]:
            fams[tok["tool_family"]] = fams.get(tok["tool_family"], 0) + 1
    print("traces           : {}  ->  {}".format(len(traces), args.output))
    print("raw events       : {}".format(raw))
    print("action tokens    : {}  (after collapse)".format(norm))
    print("tool_family mix  : " + ", ".join("{}={}".format(k, v) for k, v in sorted(fams.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
