"""Tag SWE-rebench issues by INTENT (task type) — the stratum for trace sampling.

Why not KMeans on TF-IDF: silhouette ~0.01 and clusters collapse to library/domain identity
(redundant with `repo`) plus surface artifacts (URLs, tracebacks, screenshots). Intent is the
orthogonal, low-dimensional axis that drives which CPAs appear, so we stratify on it instead.

Emits a per-issue intent and an allocation table for a given discovery budget: floor per intent,
oversample the failure-prone / high-CPA-diversity intents (crash, feature, api), cap per repo.

Usage:
    .venv/bin/python scripts/tag_issue_intent.py --input data/interim/swe_rebench/issues_3k.jsonl \
        --budget 800 --output data/interim/swe_rebench/issues_tagged.jsonl
"""
from __future__ import annotations
import argparse, json, re, collections

# ordered: first match wins (most specific / rarest signals first)
PAT = [
    ("crash_traceback", r"traceback|stack trace|segfault|core dump|raise[sd]? \w*error|exception"),
    ("perf",            r"\bperformance\b|slow(er|ness)?|speed ?up|memory (usage|leak)|optimi[sz]e|too slow|faster"),
    ("feature_request", r"feature request|would be (nice|great|good)|please add|add support|propose|enhancement|ability to"),
    ("api_design",      r"\bapi\b|signature|deprecat|backward[- ]compat|\brename\b|interface|keyword argument"),
    ("typing_docs",     r"type hint|typing|mypy|docstring|documentation|\bdocs?\b|annotation"),
    ("bug_incorrect",   r"\bbug\b|incorrect|wrong|unexpected|should (not |n't )?(return|be|raise)|expected .* but|broken|fails?\b|does(n't| not) work|regression"),
]
# oversample weight: rare + CPA-diverse intents pull more than their natural share
WEIGHT = {"crash_traceback": 1.6, "feature_request": 1.5, "api_design": 1.3,
          "bug_incorrect": 1.0, "perf": 1.4, "typing_docs": 0.7, "other": 0.6}
FLOOR = 40  # min traces per intent so rare types are still discoverable


def tag(t):
    t = t.lower()
    for n, p in PAT:
        if re.search(p, t):
            return n
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input)]
    for r in rows:
        r["intent"] = tag(r["issue"])
    with open(args.output, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    dist = collections.Counter(r["intent"] for r in rows)
    tot = len(rows)
    # weighted allocation with floor, renormalized to the budget
    raw = {k: dist[k] / tot * WEIGHT[k] for k in dist}
    s = sum(raw.values())
    alloc = {k: max(FLOOR, round(args.budget * raw[k] / s)) for k in raw}
    # rescale to hit budget after flooring
    s2 = sum(alloc.values())
    alloc = {k: round(v * args.budget / s2) for k, v in alloc.items()}

    print(f"{tot} issues | discovery budget {args.budget} (ceiling 3350 = 5% of 67k)\n")
    print(f"{'intent':16s} {'natural%':>9s} {'weight':>7s} {'alloc':>6s}")
    for k, _ in dist.most_common():
        print(f"{k:16s} {dist[k]/tot:8.1%} {WEIGHT[k]:7.1f} {alloc[k]:6d}")
    print(f"{'TOTAL':16s} {'':9s} {'':7s} {sum(alloc.values()):6d}")
    print("\nrules: cap <=4 traces/repo (spread), force ~50/50 resolved within each intent,")
    print("       soft-oversample long trajectories; stop on saturation per the new-CPA rate.")


if __name__ == "__main__":
    main()
