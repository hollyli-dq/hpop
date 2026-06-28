"""Occurrence-level CPA annotator (collaborator spec) — DATA PREPARATION ONLY.

Converts one raw agent trajectory into an ordered sequence of occurrence-level Canonical Procedural
Actions (CPAs) for downstream partial-order learning. Does NOT infer skills, partial orders, phases,
or hidden reasoning. Modes: INDUCE (propose candidate CPAs) / APPLY (match a frozen library).

Writes one full annotation object per trajectory to `<out>.jsonl` (the exact spec schema) and a
flattened `<out>.cpa_instances.jsonl` for `consolidate.py`.

Usage:
    PYTHONPATH=src .venv/bin/python -m hpop.annotate.opencode \
        --input data/interim/swe_rebench/pilot100.jsonl \
        --output data/annotated/swe_rebench/cpa_A --mode INDUCE --annotator llm_config_A --limit 100
    # APPLY a frozen library:  --mode APPLY --library rules/cpa_library_v0.1.json --library-version v0.1
    # --dry-run prints the prompt without an API call (no key needed)
"""
from __future__ import annotations

import argparse
import json
import os

from hpop.annotate.opencode_schema import CPA_ANNOTATION_SCHEMA

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """\
You are a high-precision procedural-trace annotator for software-engineering agent trajectories.

PURPOSE
Convert one raw agent trajectory into an ordered sequence of occurrence-level Canonical Procedural
Actions (CPAs) for downstream partial-order learning. You are performing DATA PREPARATION only.

You must NOT infer: skill boundaries; reusable skills; local partial orders; a global partial order;
latent reasoning or hidden intentions; phase labels. The downstream structure-learning model will
infer those objects later.

DEFINITION OF A CPA
A Canonical Procedural Action is a temporally contiguous span of one or more observable trajectory
events that: (1) performs one coherent procedural function; (2) operates on or produces an
identifiable artifact or environment state; (3) is more abstract than a raw tool name; (4) could
recur across different tasks or repositories; (5) is supported by observable evidence in the
trajectory. A tool call is not automatically a CPA. The same shell tool can be used for searching,
reading, testing, installation, execution, inspection, or repair — determine the procedural function
from the command, arguments, observations, artifacts, task context, and surrounding events.

ANNOTATION MODE
The input contains a field named "mode".
When mode = "INDUCE": the CPA library is not predefined; propose concise candidate CPA labels from
the data; use decision = "PROPOSE_NEW" unless the action is genuinely uncertain; use decision =
"ABSTAIN" when the procedural function cannot be determined; do not force candidates into an assumed
ontology.
When mode = "APPLY": use the supplied frozen CPA library; match by the library definition, not merely
by label wording; use decision = "MATCH_EXISTING" when a definition fits; use decision =
"PROPOSE_NEW" only when no existing definition fits; use decision = "ABSTAIN" when evidence is
insufficient.

BOUNDARY RULES
1. CPA spans must be temporally contiguous. 2. Usually combine a tool call with its resulting
observation. 3. Merge adjacent events only when they jointly perform one indivisible procedural
function. 4. Split the span when the procedural objective, artifact, state, or causal role changes.
5. Do not merge events merely because they use the same tool. 6. Preserve repeated actions as
separate occurrences. 7. Preserve retries, repeated tests, repeated reads, and repeated edits. 8. A
failed attempt and a later retry must be separate CPA occurrences when the failure changes the state
or motivates another action. 9. Remove only genuine logging duplicates, not semantically repeated
actions. 10. Messages containing no observable procedural action may be excluded as metadata,
planning text, or conversational text. 11. Every action-bearing tool event must belong to exactly one
CPA span. 12. Do not collapse two CPA occurrences because they receive the same label.

LABEL RULES
Candidate labels must: describe procedural function rather than the interface or tool; use
UPPER_SNAKE_CASE; normally contain a verb and an object; be portable across repositories and
programming languages; avoid file names, issue-specific entities, repository names, and tool brands;
avoid labels that are too broad (e.g. EXECUTE_COMMAND, USE_TOOL); avoid labels that are too narrow
(e.g. READ_PARSER_PY_LINE_84). Within one trajectory, reuse the same candidate label for procedurally
equivalent occurrences. Do not create synonyms merely because the actions occur at different times.

EVIDENCE AND UNCERTAINTY
Use only observable evidence: task statement; tool name; tool arguments; command; file or artifact
names; tool observations; exit codes; test outcomes; repository-state changes; surrounding observable
events. Do not invent private reasoning. Do not output chain-of-thought. Output only short evidence
statements. Use calibrated confidence: 0.90-1.00 directly demonstrated; 0.70-0.89 strongly supported
but some interpretation; 0.50-0.69 ambiguous and requires review; below 0.50 ABSTAIN.

OUTPUT REQUIREMENTS
Return valid JSON only. Do not return Markdown. Do not add text outside the JSON. Use the exact input
event identifiers. CPA spans must be sorted and non-overlapping. Occurrence identifiers must be
unique. Repeated CPA labels are explicitly allowed. Conform exactly to the response schema.
"""


def _frozen_library_block(library, version):
    if not library:
        return ""
    lines = ["FROZEN CPA LIBRARY (version {}):".format(version)]
    for c in library:
        lines.append("  - {} :: {}".format(c.get("name") or c.get("canonical_label") or c.get("id"),
                                            c.get("definition", "")))
    return "\n".join(lines)


def build_user_message(trace, mode, library, library_version):
    eid = lambda i: "e{:04d}".format(i)
    events = []
    for t in trace.get("action_tokens", []):
        a = t.get("args") or {}
        tool = t.get("tool_name") or t.get("action_type")
        cmd = t.get("command") or t.get("cmd") or t.get("artifact_id")
        ev = {
            "event_id": eid(t["i"]),
            "role": "assistant_tool_call",
            "tool": tool,
            "tool_family_hint": t.get("tool_family"),
            "command": (a.get("command")[:400] if a.get("command") else cmd),  # full bash command
            "observation": t.get("observation"),
            "follows_failed_observation": bool(t.get("after_fail")),
        }
        # expose edit content so the annotator can tell a real fix from a debug print / a test
        if tool == "str_replace_editor":
            if a.get("file_text"):
                ev["created_content"] = a["file_text"][:350]
            elif a.get("new_str") is not None:
                ev["edit_old"] = (a.get("old_str") or "")[:160]
                ev["edit_new"] = (a.get("new_str") or "")[:300]
        events.append(ev)
    payload = {
        "trajectory_id": trace.get("trajectory_id") or trace.get("trace_id"),
        "instance_id": trace.get("trace_id"),
        "repo": trace.get("repo"),
        "resolved": trace.get("resolved"),
        "mode": mode,
        "library_version": library_version,
        "task_statement": (trace.get("instructor_goal") or "")[:600],
        "events": events,
    }
    head = "INPUT (annotate per the rules; return ONLY the schema JSON):\n"
    if mode == "APPLY":
        head += _frozen_library_block(library, library_version) + "\n\n"
    return head + json.dumps(payload, ensure_ascii=False, indent=1)


def annotate_trace(client, trace, mode, library, library_version, max_tokens=20000):
    resp = client.messages.create(
        model=MODEL, max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": CPA_ANNOTATION_SCHEMA}},
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": build_user_message(trace, mode, library, library_version)}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("refused: {}".format(getattr(resp, "stop_details", None)))
    return json.loads(next((b.text for b in resp.content if b.type == "text"), ""))


def iter_traces(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Occurrence-level CPA annotator (INDUCE/APPLY).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True, help="prefix (.jsonl full + .cpa_instances.jsonl flat)")
    ap.add_argument("--mode", choices=["INDUCE", "APPLY"], default="INDUCE")
    ap.add_argument("--library", default=None, help="frozen CPA library json (APPLY mode)")
    ap.add_argument("--library-version", default=None)
    ap.add_argument("--annotator", default="llm_config_A")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    library = json.load(open(args.library)) if args.library and os.path.exists(args.library) else []
    traces = list(iter_traces(args.input))
    if args.limit is not None:
        traces = traces[: args.limit]

    if args.dry_run:
        print("=" * 80 + "\nMODE: {} | library size {} | version {}\n".format(args.mode, len(library), args.library_version) + "=" * 80)
        print("SYSTEM PROMPT ({} chars)".format(len(SYSTEM_PROMPT)))
        if traces:
            print("\n" + "=" * 80 + "\nUSER MESSAGE for {} (head):\n".format(traces[0].get("trace_id")) + "=" * 80)
            print("\n".join(build_user_message(traces[0], args.mode, library, args.library_version).splitlines()[:34]))
        print("\n[dry-run] {} trajectories would be annotated with {}.".format(len(traces), MODEL))
        return

    import anthropic
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise SystemExit("ANTHROPIC_API_KEY not set. Export it, or use --dry-run.")
    client = anthropic.Anthropic()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    ff = open(args.output + ".jsonl", "w", encoding="utf-8")
    fc = open(args.output + ".cpa_instances.jsonl", "w", encoding="utf-8")
    n_t = n_i = n_new = n_rev = 0
    try:
        for k, t in enumerate(traces, 1):
            tid = t.get("trace_id")
            try:
                ann = annotate_trace(client, t, args.mode, library, args.library_version)
            except Exception as exc:
                print("  [{}/{}] {} ERROR: {}".format(k, len(traces), tid, exc)); continue
            ann["_instance_id"] = tid; ann["_repo"] = t.get("repo"); ann["_annotator"] = args.annotator
            ff.write(json.dumps(ann, ensure_ascii=False) + "\n")
            for c in ann.get("cpa_instances", []):
                rec = dict(c, trajectory_id=ann.get("trajectory_id"), instance_id=tid,
                           repo=t.get("repo"), annotator=args.annotator)
                fc.write(json.dumps(rec, ensure_ascii=False) + "\n")
            inst = ann.get("cpa_instances", [])
            nn = sum(1 for c in inst if c.get("decision") == "PROPOSE_NEW")
            nr = sum(1 for c in inst if c.get("review_required"))
            n_t += 1; n_i += len(inst); n_new += nn; n_rev += nr
            qc = ann.get("quality_checks", {})
            print("  [{}/{}] {}  cpa={} new={} review={} skill_inferred={}".format(
                k, len(traces), tid, len(inst), nn, nr, qc.get("skill_labels_inferred")))
    finally:
        ff.close(); fc.close()
    print("\nwrote {} trajectories, {} CPA occurrences ({} new, {} review) -> {}.jsonl + .cpa_instances.jsonl".format(
        n_t, n_i, n_new, n_rev, args.output))


if __name__ == "__main__":
    main()
