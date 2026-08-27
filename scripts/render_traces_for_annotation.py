"""Render normalized traces into compact per-trajectory files for in-session Claude annotation.

The annotator is Claude reading these files directly (no API): `tokens500.jsonl` is 32MB because
`args` is stored untruncated, so a subagent cannot hold many raw traces at once. This selects exactly
the fields the API path's `build_user_message()` selects — event id, tool, command, observation,
after_fail, and edit content — and drops the rest.

Edit content matters and is NOT optional: it is the only thing separating EDIT_SOURCE from
ADD_DEBUG_INSTRUMENTATION from WRITE_TEST. Without it those three CPAs are indistinguishable and
PROPOSE_NEW inflates for reasons that have nothing to do with the library.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/render_traces_for_annotation.py \
        --input data/interim/swe_rebench/tokens_heldout.jsonl \
        --outdir data/interim/swe_rebench/annot/heldout \
        --manifest data/interim/swe_rebench/annot/heldout.manifest.json
"""
import argparse
import json
import os


def render(trace):
    eid = lambda i: "e{:04d}".format(i)
    L = []
    L.append("TRAJECTORY: {}".format(trace.get("trace_id")))
    L.append("REPO:       {}".format(trace.get("repo")))
    L.append("TASK:       {}".format((trace.get("instructor_goal") or "")[:600]))
    L.append("EVENTS:     {} action tokens".format(len(trace.get("action_tokens", []))))
    L.append("")
    L.append("-" * 78)
    for t in trace.get("action_tokens", []):
        a = t.get("args") or {}
        tool = t.get("tool_name") or t.get("action_type")
        cmd = a.get("command") or t.get("command") or t.get("artifact_id") or ""
        L.append("[{}] tool={}{}".format(
            eid(t["i"]), tool, "  (FOLLOWS A FAILED OBSERVATION)" if t.get("after_fail") else ""))
        if cmd:
            L.append("     cmd: {}".format(str(cmd)[:400].replace("\n", " ")))
        # expose edit content — the ONLY signal separating a fix from a debug print from a test
        if tool == "str_replace_editor":
            # `args.command` is only the editor sub-command ("view"/"create"/"str_replace") — the PATH
            # lives in args.path and is what separates READ_SOURCE / READ_DOCUMENTATION /
            # EXPLORE_REPOSITORY. Emit it explicitly or those three are undecidable.
            if a.get("path"):
                L.append("     path: {}".format(str(a["path"])[:200]))
            if a.get("view_range"):
                L.append("     view_range: {}".format(a["view_range"]))
            if a.get("file_text"):
                L.append("     created_content: {}".format(str(a["file_text"])[:350].replace("\n", " ⏎ ")))
            elif a.get("new_str") is not None:
                L.append("     edit_old: {}".format(str(a.get("old_str") or "")[:160].replace("\n", " ⏎ ")))
                L.append("     edit_new: {}".format(str(a.get("new_str") or "")[:300].replace("\n", " ⏎ ")))
        obs = t.get("observation")
        if obs:
            L.append("     obs: {}".format(str(obs)[:140].replace("\n", " ")))
    L.append("-" * 78)
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render traces to compact per-trajectory annotation files.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    traces = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if args.limit:
        traces = traces[: args.limit]

    manifest, total_chars = [], 0
    for t in traces:
        tid = t.get("trace_id")
        safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(tid))
        path = os.path.join(args.outdir, safe + ".txt")
        body = render(t)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        total_chars += len(body)
        manifest.append({"trace_id": tid, "repo": t.get("repo"), "resolved": t.get("resolved"),
                         "n_events": len(t.get("action_tokens", [])), "path": path,
                         "chars": len(body)})

    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    n = len(manifest) or 1
    print("rendered {} traces -> {}".format(len(manifest), args.outdir))
    print("manifest -> {}".format(args.manifest))
    print("size: {:.1f} KB total, {:.1f} KB/trace mean, {:.1f} KB max (~{:.0f} tokens/trace)".format(
        total_chars / 1024, total_chars / n / 1024,
        max(m["chars"] for m in manifest) / 1024 if manifest else 0,
        total_chars / n / 3.6))
    print("repos: {}  | resolved: {}/{}".format(
        len({m["repo"] for m in manifest}), sum(1 for m in manifest if m["resolved"]), len(manifest)))


if __name__ == "__main__":
    main()
