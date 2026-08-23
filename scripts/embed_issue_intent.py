"""Embedding-based issue-intent labelling — replaces the brittle regex tagger.

Method (label-anchored / zero-shot, NOT clustering):
  TF-IDF and KMeans fail here because domain/library is the dominant axis of variance, so clusters
  collapse to "pandas issues", "sympy issues", etc. We instead embed each issue AND a short prototype
  sentence per intent, then assign each issue to its nearest prototype by cosine similarity. This
  forces the intent axis (similarity to intent concepts) instead of raw geometry.

Optional --deconfound subtracts each issue's per-repo mean embedding before scoring, projecting out
the library nuisance direction so paraphrase-level intent signal dominates.

Static embeddings via model2vec (no torch, no API key). Outputs per-issue intent + margin (top1-top2
cosine, a confidence proxy) and the allocation table.

Usage:
    .venv/bin/python scripts/embed_issue_intent.py --input data/interim/swe_rebench/issues_3k.jsonl \
        --model data/interim/swe_rebench/potion-8M --budget 800 \
        --output data/interim/swe_rebench/issues_embed_tagged.jsonl
"""
from __future__ import annotations
import argparse, json, collections
import numpy as np
from model2vec import StaticModel

# one short, intent-defining prototype per label. Multiple phrasings averaged -> a robust anchor.
PROTOTYPES = {
    "crash_traceback": [
        "The program crashes and raises an exception with a traceback.",
        "Running the code throws an error and aborts unexpectedly.",
    ],
    "bug_incorrect": [
        "Existing behavior is wrong: it returns an incorrect or unexpected result.",
        "A bug where the output differs from what is expected.",
    ],
    "feature_request": [
        "A request to add a new feature or capability that does not exist yet.",
        "It would be nice to support a new option or functionality.",
    ],
    "api_design": [
        "A change to the public API, function signature, naming, or backward compatibility.",
        "Rename or redesign an interface, parameter, or method.",
    ],
    "perf": [
        "A performance problem: the code is too slow or uses too much memory.",
        "Optimize runtime or reduce memory usage.",
    ],
    "typing_docs": [
        "An issue about type hints, type checking, or documentation.",
        "Improve docstrings, docs, or type annotations.",
    ],
}
WEIGHT = {"crash_traceback": 1.6, "feature_request": 1.5, "api_design": 1.3,
          "bug_incorrect": 1.0, "perf": 1.4, "typing_docs": 0.7, "other": 0.6}
FLOOR = 40
OTHER_MARGIN = 0.02  # if top1-top2 cosine margin below this -> 'other' (ambiguous)


def _norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--deconfound", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input)]
    m = StaticModel.from_pretrained(args.model)

    labels = list(PROTOTYPES)
    proto = _norm(np.array([m.encode(PROTOTYPES[k]).mean(0) for k in labels]))

    E = _norm(np.array(m.encode([r["issue"][:2000] for r in rows])))

    if args.deconfound:
        repo_idx = collections.defaultdict(list)
        for i, r in enumerate(rows):
            repo_idx[r["repo"]].append(i)
        for idx in repo_idx.values():
            E[idx] -= E[idx].mean(0, keepdims=True)
        E = _norm(E)

    sims = E @ proto.T                      # (N, n_labels) cosine
    top2 = np.sort(sims, axis=1)[:, ::-1][:, :2]
    margin = top2[:, 0] - top2[:, 1]
    arg = sims.argmax(1)

    for i, r in enumerate(rows):
        r["intent"] = "other" if margin[i] < OTHER_MARGIN else labels[arg[i]]
        r["intent_margin"] = round(float(margin[i]), 4)
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

    print(f"{tot} issues | deconfound={args.deconfound} | budget {args.budget}\n")
    print(f"{'intent':16s} {'share':>7s} {'avg_margin':>11s} {'alloc':>6s}")
    for k, _ in dist.most_common():
        mm = np.mean([r["intent_margin"] for r in rows if r["intent"] == k])
        print(f"{k:16s} {dist[k]/tot:6.1%} {mm:11.3f} {alloc.get(k,0):6d}")
    print("\n--- 2 examples per intent ---")
    for k in dist:
        exs = [r for r in rows if r["intent"] == k][:2]
        print(f"[{k}]")
        for r in exs:
            print("   ", r["issue"][:110])


if __name__ == "__main__":
    main()
