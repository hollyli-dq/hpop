"""PDAF pipeline driver (TWO-LEVEL) — Steps 1b/2/3a/3b (LLM) + 4 (confidence/monitor) + 5 (flags).

Reads normalized traces (from `hpop.ingest.normalize`), annotates each via `annotator.py`,
assigns canonical ids, and writes:

    <out>.skills.jsonl         one line per trace  (skill instances + their CPA instances)
    <out>.local_orders.jsonl   one line per CPA-level edge   (HPOP local poset priors)
    <out>.global_orders.jsonl  one line per skill-level edge  (HPOP global poset priors)
    <out>.review.jsonl         one line per flagged item (Step 5 queue)

--dry-run prints the system + user prompt for the first traces; no API call / key needed.

Usage:
    PYTHONPATH=src .venv/bin/python -m hpop.annotate.run \
        --input data/interim/weblinx/valid.normalized.jsonl \
        --output data/annotated/weblinx/valid --limit 5
"""
from __future__ import annotations

import argparse
import json
import os

from hpop.annotate.annotator import build_system_prompt, build_user_message, annotate_trace
from hpop.annotate.schema import CPA_TO_PHASE

HIGH, MED = 0.8, 0.5
ADJ_ONLY_WARN = 0.15

# Mirror of rules/skill_library.yaml (keep in sync). name -> (skill_id, skill_file).
SKILL_INDEX = {
    "search_and_summarize": ("S001", "S1_search_and_summarize.md"),
    "compare_products": ("S002", "S2_compare_products.md"),
    "fill_and_submit_form": ("S003", "S3_fill_and_submit_form.md"),
    "search_and_retrieve": ("S004", "S4_search_and_retrieve.md"),
    "verify_fact": ("S005", "S5_verify_fact.md"),
}


def _tier(c):
    return "HIGH" if c >= HIGH else ("MEDIUM" if c >= MED else "ABSTAIN")


def _adj_monitor(trace_id, edges, scope):
    n = len(edges)
    if not n:
        return []
    adj = sum(1 for e in edges if e["label"] == "ADJACENT_ONLY")
    if adj / n > ADJ_ONLY_WARN:
        return [{"trace_id": trace_id, "kind": "distribution", "tier": "advisory",
                 "scope": scope, "metric": "adjacent_only_fraction",
                 "value": round(adj / n, 3), "threshold": ADJ_ONLY_WARN}]
    return []


def postprocess(trace_id, ann):
    """Two-level PDAF annotation -> (skills_record, local_edges, global_edges, review_flags)."""
    flags = []
    cpas = sorted(ann.get("cpa_instances", []), key=lambda c: c["index"])
    cid = {c["index"]: "{}-CPA{}".format(trace_id, c["index"] + 1) for c in cpas}
    skill_idx_of_cpa = {c["index"]: c.get("skill_index", 0) for c in cpas}

    # skill instances
    sins = sorted(ann.get("skill_instances", []), key=lambda s: s["index"])
    sid = {s["index"]: "{}-S{}".format(trace_id, s["index"] + 1) for s in sins}
    sname = {s["index"]: s.get("skill_name", "unknown") for s in sins}

    # group CPA instances under their skill
    by_skill = {}
    for c in cpas:
        phase = c.get("phase") or CPA_TO_PHASE.get(c["cpa"], "")
        conf = float(c.get("confidence", 0.0))
        rec = {
            "cpa_instance_id": cid[c["index"]], "cpa": c["cpa"], "phase": phase,
            "turn_start": c.get("turn_start"), "turn_end": c.get("turn_end"),
            "event_indices": c.get("event_indices", []),
            "artifact_id": c.get("artifact_id", ""), "confidence": conf,
        }
        by_skill.setdefault(c.get("skill_index", 0), []).append(rec)
        if _tier(conf) == "ABSTAIN":
            flags.append({"trace_id": trace_id, "kind": "cpa", "tier": "mandatory",
                          "id": cid[c["index"]], "cpa": c["cpa"], "confidence": conf})
        elif _tier(conf) == "MEDIUM":
            flags.append({"trace_id": trace_id, "kind": "cpa", "tier": "optional",
                          "id": cid[c["index"]], "cpa": c["cpa"], "confidence": conf})

    skill_instances = []
    for s in sins:
        name = sname[s["index"]]
        skill_id, skill_file = SKILL_INDEX.get(name, ("SNEW", "{}.md".format(name)))
        members = by_skill.get(s["index"], [])
        is_new = bool(s.get("skill_is_new", skill_id == "SNEW"))
        skill_instances.append({
            "skill_instance_id": sid[s["index"]], "skill_id": skill_id,
            "skill_name": name, "skill_is_new": is_new,
            "skill_confidence": float(s.get("skill_confidence", 0.0)),
            "cpa_instances": members,
            "local_cpa_sequence": [m["cpa"] for m in members],
            "local_phase_sequence": [m["phase"] for m in members],
        })
        if is_new:
            flags.append({"trace_id": trace_id, "kind": "skill", "tier": "advisory",
                          "id": sid[s["index"]], "skill_name": name,
                          "rationale": "skill not in SKILL_LIBRARY"})

    skills_record = {
        "trace_id": trace_id,
        "num_skill_instances": len(skill_instances),
        "global_skill_sequence": [si["skill_name"] for si in skill_instances],
        "skill_instances": skill_instances,
    }

    # local edges (CPA-level, within a skill)
    local_out = []
    for n, e in enumerate(ann.get("local_edges", [])):
        si, ti = e.get("source_index"), e.get("target_index")
        if si not in cid or ti not in cid:
            continue
        sk = skill_idx_of_cpa.get(si, 0)
        src = next((c for c in cpas if c["index"] == si), {})
        tgt = next((c for c in cpas if c["index"] == ti), {})
        conf = float(e.get("confidence", 0.0))
        local_out.append({
            "edge_id": "{}-LE{}".format(trace_id, n + 1), "trace_id": trace_id,
            "skill_instance_id": sid.get(sk), "skill_name": sname.get(sk),
            "source_cpa_id": cid[si], "target_cpa_id": cid[ti],
            "source_cpa": src.get("cpa"), "target_cpa": tgt.get("cpa"),
            "label": e.get("label", "ADJACENT_ONLY"), "rationale": e.get("rationale", ""),
            "confidence": conf, "artifact_transferred": e.get("artifact_transferred", ""),
        })
        if _tier(conf) == "ABSTAIN":
            flags.append({"trace_id": trace_id, "kind": "local_edge", "tier": "mandatory",
                          "id": local_out[-1]["edge_id"], "confidence": conf})
    flags += _adj_monitor(trace_id, local_out, "local")

    # global edges (skill-level)
    global_out = []
    for n, e in enumerate(ann.get("global_edges", [])):
        si, ti = e.get("source_skill_index"), e.get("target_skill_index")
        if si not in sid or ti not in sid:
            continue
        conf = float(e.get("confidence", 0.0))
        global_out.append({
            "edge_id": "{}-GE{}".format(trace_id, n + 1), "trace_id": trace_id,
            "source_skill_id": sid[si], "target_skill_id": sid[ti],
            "source_skill_name": sname.get(si), "target_skill_name": sname.get(ti),
            "label": e.get("label", "ADJACENT_ONLY"), "rationale": e.get("rationale", ""),
            "confidence": conf, "artifact_transferred": e.get("artifact_transferred", ""),
        })
        if _tier(conf) == "ABSTAIN":
            flags.append({"trace_id": trace_id, "kind": "global_edge", "tier": "mandatory",
                          "id": global_out[-1]["edge_id"], "confidence": conf})
    flags += _adj_monitor(trace_id, global_out, "global")

    return skills_record, local_out, global_out, flags


def iter_traces(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description="PDAF two-level pipeline driver (Steps 1b-5).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True, help="prefix (.skills/.local_orders/.global_orders/.review .jsonl)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print prompts; no API call")
    args = ap.parse_args(argv)

    traces = list(iter_traces(args.input))
    if args.limit is not None:
        traces = traces[: args.limit]
    system_prompt = build_system_prompt()

    if args.dry_run:
        print("=" * 80, "\nSYSTEM PROMPT ({} chars):\n".format(len(system_prompt)) + "=" * 80)
        print(system_prompt)
        for t in traces[:1]:
            print("\n" + "=" * 80 + "\nUSER MESSAGE for trace {}:\n".format(t.get("trace_id")) + "=" * 80)
            print(build_user_message(t))
        print("\n[dry-run] {} trace(s) would be annotated with claude-opus-4-8.".format(len(traces)))
        return

    import anthropic
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise SystemExit("ANTHROPIC_API_KEY not set. Export it, or use --dry-run.")
    client = anthropic.Anthropic()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fs = open(args.output + ".skills.jsonl", "w", encoding="utf-8")
    fl = open(args.output + ".local_orders.jsonl", "w", encoding="utf-8")
    fg = open(args.output + ".global_orders.jsonl", "w", encoding="utf-8")
    fr = open(args.output + ".review.jsonl", "w", encoding="utf-8")
    n_tr = n_le = n_ge = n_fl = 0
    try:
        for i, t in enumerate(traces, 1):
            tid = t.get("trace_id")
            try:
                ann = annotate_trace(client, t, system_prompt=system_prompt)
            except Exception as exc:
                fr.write(json.dumps({"trace_id": tid, "kind": "error", "error": str(exc)}) + "\n")
                print("  [{}/{}] {} ERROR: {}".format(i, len(traces), tid, exc))
                continue
            srec, le, ge, flags = postprocess(tid, ann)
            fs.write(json.dumps(srec, ensure_ascii=False) + "\n")
            for e in le:
                fl.write(json.dumps(e, ensure_ascii=False) + "\n")
            for e in ge:
                fg.write(json.dumps(e, ensure_ascii=False) + "\n")
            for f in flags:
                fr.write(json.dumps(f, ensure_ascii=False) + "\n")
            n_tr += 1; n_le += len(le); n_ge += len(ge); n_fl += len(flags)
            print("  [{}/{}] {}  skills={} ({})  CPAs={}  local={}  global={}  flags={}".format(
                i, len(traces), tid, srec["num_skill_instances"],
                "/".join(srec["global_skill_sequence"]),
                sum(len(s["cpa_instances"]) for s in srec["skill_instances"]),
                len(le), len(ge), len(flags)))
    finally:
        fs.close(); fl.close(); fg.close(); fr.close()
    print("\nwrote {} traces, {} local edges, {} global edges, {} flags -> {}.*.jsonl".format(
        n_tr, n_le, n_ge, n_fl, args.output))


if __name__ == "__main__":
    main()
