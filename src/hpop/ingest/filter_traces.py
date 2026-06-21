"""Filter normalized traces to the PROCEDURAL / agentic subset.

WebLINX `chat` is dominated by conversational Q&A traces (read -> say the answer), which are
near-linear with shallow dependencies and a single skill — they under-exercise the HPOP
hierarchical partial-order / skill model. This step keeps traces that look like multi-step agent
*procedures* (state-changing actions, tool diversity, enough non-dialogue work) and drops the rest.

A trace is KEPT iff all of:
  - dialogue fraction        <  --max-dialogue-frac     (default 0.5)
  - non-dialogue actions     >= --min-actions           (default 4)
  - distinct non-dialogue verbs >= --min-distinct-verbs (default 2)
  - has >=1 state-changing action (--require-state-change, default on)
        STATE_CHANGE verbs = submit, text_input, fill_field, change, load,
                             tabcreate, tabswitch, tabremove, paste

Writes the kept subset + a per-trace report (features, keep, reasons).

Usage:
    PYTHONPATH=src python3 -m hpop.ingest.filter_traces \
        --input  data/interim/weblinx/valid.normalized.jsonl \
        --output data/interim/weblinx/valid.filtered.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

STATE_CHANGE = {"submit", "text_input", "fill_field", "change", "load",
                "tabcreate", "tabswitch", "tabremove", "paste"}


def features(trace):
    toks = trace.get("action_tokens", [])
    n = len(toks) or 1
    say = sum(1 for t in toks if t["action_type"] == "say")
    nondlg = [t for t in toks if t["action_type"] != "say"]
    verbs = sorted({t["action_type"] for t in nondlg})
    return {
        "trace_id": trace.get("trace_id"),
        "n_tokens": len(toks),
        "n_actions": len(nondlg),
        "dialogue_frac": round(say / n, 3),
        "distinct_verbs": len(verbs),
        "verbs": verbs,
        "has_state_change": bool(set(verbs) & STATE_CHANGE),
    }


def decide(f, max_dialogue_frac, min_actions, min_distinct_verbs, require_state_change):
    reasons = []
    if f["dialogue_frac"] >= max_dialogue_frac:
        reasons.append("dialogue>={:.0%}".format(max_dialogue_frac))
    if f["n_actions"] < min_actions:
        reasons.append("actions<{}".format(min_actions))
    if f["distinct_verbs"] < min_distinct_verbs:
        reasons.append("verbs<{}".format(min_distinct_verbs))
    if require_state_change and not f["has_state_change"]:
        reasons.append("no_state_change")
    return (len(reasons) == 0), reasons


def iter_traces(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Filter normalized traces to the procedural subset.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default=None, help="per-trace report jsonl (default: <output>.filter_report.jsonl)")
    ap.add_argument("--max-dialogue-frac", type=float, default=0.5)
    ap.add_argument("--min-actions", type=int, default=4)
    ap.add_argument("--min-distinct-verbs", type=int, default=2)
    ap.add_argument("--no-require-state-change", action="store_true")
    args = ap.parse_args(argv)

    report_path = args.report or (os.path.splitext(args.output)[0] + ".filter_report.jsonl")
    require_sc = not args.no_require_state_change

    kept, dropped, reason_counts = [], 0, {}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as rep:
        for t in iter_traces(args.input):
            f = features(t)
            keep, reasons = decide(f, args.max_dialogue_frac, args.min_actions,
                                   args.min_distinct_verbs, require_sc)
            f["keep"] = keep
            f["reasons"] = reasons
            rep.write(json.dumps(f, ensure_ascii=False) + "\n")
            if keep:
                kept.append(t)
            else:
                dropped += 1
                for r in reasons:
                    reason_counts[r] = reason_counts.get(r, 0) + 1

    with open(args.output, "w", encoding="utf-8") as out:
        for t in kept:
            out.write(json.dumps(t, ensure_ascii=False) + "\n")

    total = len(kept) + dropped
    print("input traces : {}".format(total))
    print("KEPT         : {}  ({:.0%})  -> {}".format(len(kept), len(kept) / max(1, total), args.output))
    print("dropped      : {}".format(dropped))
    print("drop reasons : " + (", ".join("{}={}".format(k, v) for k, v in
          sorted(reason_counts.items(), key=lambda kv: -kv[1])) or "none"))
    print("report       : {}".format(report_path))
    if kept:
        import statistics as st
        acts = [features(t)["n_actions"] for t in kept]
        dv = [features(t)["distinct_verbs"] for t in kept]
        print("kept actions/trace : mean {:.1f} median {}".format(st.mean(acts), st.median(acts)))
        print("kept distinct verbs: mean {:.1f} median {}".format(st.mean(dv), st.median(dv)))
        print("kept sample        : " + ", ".join(t["trace_id"] for t in kept[:8]))


if __name__ == "__main__":
    main()
