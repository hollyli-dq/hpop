"""Ingest WebLINX `chat` data into per-trace raw-event sequences.

WebLINX `chat` files are gzipped JSON-Lines: one record per *turn*, with fields
``demo, turn, action, action_history, utterances, candidates, clean_html, viewport``.

A demonstration (all turns sharing a ``demo`` id, ordered by ``turn``) is one **workflow**
in the HPOP taxonomy (DESIGN.md). Each turn's ``action`` is a **raw trace event** (level 0).
This module groups turns into traces and parses each action into ``(verb, args)`` so the
downstream annotator has a clean, structured event stream to label.

Output: one JSON object per trace, written as JSON-Lines to ``data/interim/``.

Usage:
    PYTHONPATH=src python3 -m hpop.ingest.weblinx \
        --input data/raw/weblinx/chat/valid.json.gz --split valid \
        --output data/interim/weblinx/valid.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections import OrderedDict

# action looks like:  verb(arg="...", arg2="...")   e.g.  click(uid="abc")
_VERB_RE = re.compile(r"^\s*([a-zA-Z_]\w*)\s*\(")
_KV_RE = re.compile(r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"')
# a top-level `verb(` opener anywhere in an action_history string
_CALL_OPEN_RE = re.compile(r"([a-zA-Z_]\w*)\s*\(")


def parse_action(action):
    """Parse a WebLINX action string into ``{"verb": str, "args": {..}, "raw": str}``."""
    raw = (action or "").strip()
    verb_m = _VERB_RE.match(raw)
    verb = verb_m.group(1) if verb_m else (raw.split("(", 1)[0].strip() or "unknown")
    args = OrderedDict()
    for k, v in _KV_RE.findall(raw):
        args[k] = v
    return {"verb": verb, "args": args, "raw": raw}


def parse_action_sequence(history):
    """Parse a WebLINX ``action_history`` string into an ordered list of parsed actions.

    The history concatenates calls like ``say(...) load(...) click(...)`` separated by
    model tokens (``</s><s>[INST]``). Utterances may contain parentheses, so we scan for
    each ``verb(`` opener and consume to its matching ``)`` while respecting double quotes.
    """
    if not history:
        return []
    actions = []
    i, n = 0, len(history)
    while i < n:
        m = _CALL_OPEN_RE.search(history, i)
        if not m:
            break
        verb = m.group(1)
        j = m.end()  # just after "("
        depth, in_str, esc = 1, False, False
        while j < n and depth > 0:
            c = history[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
            j += 1
        raw = history[m.start():j]
        actions.append(parse_action(raw))
        i = j
    return actions


def iter_chat_records(path):
    """Yield turn records from a (optionally gzipped) WebLINX chat JSONL file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_traces(records, split=None):
    """Group turn records by ``demo`` into ordered traces (workflows).

    Returns a list of trace dicts. Heavy per-turn fields (``clean_html``, ``candidates``,
    ``action_history``) are dropped here — they are page context for later stages, not part
    of the raw-event stream we annotate. ``viewport`` is kept once per trace.
    """
    by_demo = OrderedDict()
    for r in records:
        by_demo.setdefault(r["demo"], []).append(r)

    traces = []
    for demo, turns in by_demo.items():
        turns = sorted(turns, key=lambda r: r.get("turn", 0))
        # Reconstruct the complete trace. `action_history` accumulates across turns and
        # recovers the instructor's goal-setting utterances and all dialogue, which the
        # per-turn `action` fields (navigator predictions only) omit. The last turn can have
        # a null history, so use the turn with the LONGEST history as the base, then append
        # the `action` of every turn at or after it (those are not yet in that base history).
        base = max(turns, key=lambda r: len(r.get("action_history") or ""))
        base_turn = base.get("turn", 0)
        parsed_actions = parse_action_sequence(base.get("action_history") or "")
        for r in turns:
            if r.get("turn", 0) >= base_turn and r.get("action"):
                parsed_actions.append(parse_action(r.get("action", "")))

        events = []
        for parsed in parsed_actions:
            ev = {
                "i": len(events),
                "verb": parsed["verb"],
                "action": parsed["raw"],
            }
            if parsed["args"]:
                ev["args"] = parsed["args"]
            # surface dialogue for readability when the action is a `say`
            if parsed["verb"] == "say":
                ev["speaker"] = parsed["args"].get("speaker")
                ev["utterance"] = parsed["args"].get("utterance")
            events.append(ev)

        # first instructor utterance = the task goal, when present
        goal = next(
            (e.get("utterance") for e in events
             if e.get("verb") == "say" and e.get("speaker") == "instructor"),
            None,
        )
        trace = {
            "trace_id": demo,
            "source": "weblinx",
            "num_events": len(events),
            "viewport": turns[0].get("viewport"),
            "instructor_goal": goal,
            "events": events,
        }
        if split:
            trace["split"] = split
        traces.append(trace)
    return traces


def write_jsonl(traces, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ingest WebLINX chat data into per-trace events.")
    ap.add_argument("--input", required=True, help="WebLINX chat .json.gz (or .jsonl) file")
    ap.add_argument("--split", default=None, help="split name to record on each trace")
    ap.add_argument("--output", required=True, help="output .jsonl path (one trace per line)")
    ap.add_argument("--limit", type=int, default=None, help="only keep the first N traces")
    args = ap.parse_args(argv)

    records = list(iter_chat_records(args.input))
    traces = build_traces(records, split=args.split)
    if args.limit is not None:
        traces = traces[: args.limit]
    write_jsonl(traces, args.output)

    n_events = sum(t["num_events"] for t in traces)
    verbs = {}
    for t in traces:
        for e in t["events"]:
            verbs[e["verb"]] = verbs.get(e["verb"], 0) + 1
    top = sorted(verbs.items(), key=lambda kv: -kv[1])[:12]
    print("input records : {}".format(len(records)))
    print("traces written: {}  ->  {}".format(len(traces), args.output))
    print("total events  : {}  (avg {:.1f}/trace)".format(n_events, n_events / max(1, len(traces))))
    print("verb counts   : " + ", ".join("{}={}".format(k, v) for k, v in top))


if __name__ == "__main__":
    main()
