"""LLM-anchored issue-intent classification — the accurate replacement for regex / static embeddings.

Mirrors the project's annotator client (anthropic.Anthropic(); ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN
in env). Uses claude-haiku-4-5 — classification is the documented Haiku use case, far cheaper than the
opus-4-8 the deep annotator uses. Batches ~25 issues per request and forces a valid label set with
structured outputs (output_config.format), so every issue gets exactly one of the 7 taxonomy labels.

Cost: ~3k issues / 25 per call = ~120 Haiku calls (cents). Full 21k = ~850 calls.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/llm_issue_intent.py \
        --input data/interim/swe_rebench/issues_3k.jsonl --batch 25 --budget 800 \
        --output data/interim/swe_rebench/issues_llm_tagged.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, collections

MODEL = "claude-haiku-4-5"
LABELS = ["crash_traceback", "bug_incorrect", "feature_request",
          "api_design", "perf", "typing_docs", "other"]

SYSTEM = """You classify software issue reports by the author's INTENT — the kind of change being requested. \
This drives which problem-solving steps an agent will take, so classify by what the task fundamentally is, \
not by surface keywords. Exactly one label per issue:

- crash_traceback: the program crashes / raises an exception / prints a traceback; fix the failure
- bug_incorrect: existing behavior is wrong or unexpected (no crash); make it produce the right result
- feature_request: add a new capability / option / function that does not exist yet
- api_design: change a public API, signature, naming, deprecation, or backward compatibility
- perf: make something faster or use less memory
- typing_docs: type hints / type checking / docstrings / documentation only
- other: none of the above, or genuinely ambiguous

Pick the single best fit. A crash caused by a bug is crash_traceback (the traceback dominates the work)."""

WEIGHT = {"crash_traceback": 1.6, "feature_request": 1.5, "api_design": 1.3,
          "bug_incorrect": 1.0, "perf": 1.4, "typing_docs": 0.7, "other": 0.6}
FLOOR = 40

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "id": {"type": "integer"},
                    "intent": {"type": "string", "enum": LABELS},
                },
                "required": ["id", "intent"],
            },
        }
    },
    "required": ["labels"],
}


def classify_batch(client, batch):
    listing = "\n".join("[{}] {}".format(i, r["issue"][:1200]) for i, r in enumerate(batch))
    resp = client.messages.create(
        model=MODEL, max_tokens=4000,
        system=SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content":
                   "Classify each issue. Return one label per id (0..{}).\n\n{}".format(len(batch) - 1, listing)}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    by_id = {d["id"]: d["intent"] for d in json.loads(text)["labels"]}
    return [by_id.get(i, "other") for i in range(len(batch))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0, help="classify only first N (smoke test)")
    args = ap.parse_args()

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("Set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) in the environment first.")
    import anthropic
    client = anthropic.Anthropic()

    rows = [json.loads(l) for l in open(args.input)]
    if args.limit:
        rows = rows[:args.limit]

    for s in range(0, len(rows), args.batch):
        batch = rows[s:s + args.batch]
        try:
            intents = classify_batch(client, batch)
        except Exception as e:
            print("  batch @{} failed: {}".format(s, str(e)[:80])); intents = ["other"] * len(batch)
        for r, it in zip(batch, intents):
            r["intent"] = it
        print("  {}/{}".format(min(s + args.batch, len(rows)), len(rows)), flush=True)

    with open(args.output, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    dist = collections.Counter(r["intent"] for r in rows)
    tot = len(rows)
    raw = {k: dist[k] / tot * WEIGHT[k] for k in dist}
    s = sum(raw.values())
    alloc = {k: max(FLOOR, round(args.budget * raw[k] / s)) for k in raw}
    s2 = sum(alloc.values())
    alloc = {k: round(v * args.budget / s2) for k, v in alloc.items()}

    print("\n{} issues classified by {} | budget {}\n".format(tot, MODEL, args.budget))
    print("{:16s} {:>7s} {:>6s}".format("intent", "share", "alloc"))
    for k, _ in dist.most_common():
        print("{:16s} {:6.1%} {:6d}".format(k, dist[k] / tot, alloc.get(k, 0)))
    print("wrote -> {}".format(args.output))


if __name__ == "__main__":
    main()
