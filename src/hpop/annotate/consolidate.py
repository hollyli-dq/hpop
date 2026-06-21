"""Consolidate open-coding proposals into a candidate CPA library (MergeSplitReview + Accept gate).

Reads one or more `*.opencode.jsonl` provenance files (ideally from >=2 independent annotators),
groups proposed labels, computes N(c)/R(c)/A(c), and applies the acceptance gate
    Accept(c) = 1[N(c) >= m] * 1[R(c) >= r] * 1[A(c) >= tau]
to emit a candidate CPA library + a human-review queue. Label grouping here is a simple normalized
exact-match proxy; true semantic MERGE/SPLIT is the human/multi-model step (docs/cpa_induction.md).

Usage:
    PYTHONPATH=src python3 -m hpop.annotate.consolidate \
        --inputs data/annotated/swe/opencode_A.opencode.jsonl data/annotated/swe/opencode_B.opencode.jsonl \
        --out-library rules/cpa_library_candidate.json \
        --out-review data/annotated/swe/cpa_review_queue.jsonl --m 3 --r 2 --tau 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import re


def _norm(label):
    return re.sub(r"[^a-z0-9 ]", "", (label or "").lower()).strip()


def _label(r):
    return (r.get("candidate_label") or r.get("canonical_label")
            or r.get("proposed_cpa") or r.get("proposed_label"))


def _conf(r):
    v = r.get("label_confidence")
    return float(v) if v is not None else float(r.get("confidence", 0.0))


def _read(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if _label(r):
                        yield r


def main(argv=None):
    ap = argparse.ArgumentParser(description="Consolidate open-coding proposals -> candidate CPA library.")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-library", required=True)
    ap.add_argument("--out-review", required=True)
    ap.add_argument("--m", type=int, default=3, help="min occurrences N(c)")
    ap.add_argument("--r", type=int, default=2, help="min distinct traces/repos R(c)")
    ap.add_argument("--tau", type=float, default=0.5, help="min adjudication agreement A(c)")
    args = ap.parse_args(argv)

    recs = list(_read(args.inputs))
    annotators = sorted({r.get("annotator", "?") for r in recs})
    n_ann = len(annotators) or 1

    groups = {}  # normalized label -> aggregate
    for r in recs:
        key = _norm(r.get("canonical_label") or r.get("canonical_cpa") or _label(r))
        if not key:
            continue
        g = groups.setdefault(key, {"label": _label(r), "n": 0, "traces": set(),
                                    "annotators": set(), "confs": [], "decisions": set(),
                                    "definitions": set(), "examples": []})
        g["n"] += 1
        g["traces"].add(r.get("repo") or r.get("trajectory_id") or r.get("trace_id"))
        g["annotators"].add(r.get("annotator"))
        g["confs"].append(_conf(r))
        g["decisions"].add(r.get("decision") or r.get("status"))
        if r.get("definition"):
            g["definitions"].add(r["definition"][:120])
        if len(g["examples"]) < 3:
            g["examples"].append({"trace_id": r.get("trace_id"), "evidence": r.get("evidence", "")[:100]})

    library, review = [], []
    cid = 1
    for key, g in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
        N = g["n"]; R = len(g["traces"]); A = len(g["annotators"]) / n_ann
        accept = (N >= args.m) and (R >= args.r) and (A >= args.tau)
        entry = {
            "candidate_label": g["label"], "N": N, "R": R, "A": round(A, 2),
            "mean_confidence": round(sum(g["confs"]) / len(g["confs"]), 2),
            "definitions": sorted(g["definitions"])[:3], "examples": g["examples"],
        }
        if accept:
            library.append(dict(entry, id="CPA{:03d}".format(cid),
                                name=re.sub(r"\s+", "_", g["label"].strip()).upper()[:40]))
            cid += 1
        else:
            reasons = []
            if N < args.m: reasons.append("N<{}".format(args.m))
            if R < args.r: reasons.append("R<{}".format(args.r))
            if A < args.tau: reasons.append("A<{}".format(args.tau))
            review.append(dict(entry, review_reasons=reasons, tier="proposed_new_pending"))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_library)) or ".", exist_ok=True)
    json.dump(library, open(args.out_library, "w"), indent=2)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_review)) or ".", exist_ok=True)
    with open(args.out_review, "w", encoding="utf-8") as f:
        for r in review:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("annotators        : {} ({})".format(n_ann, ", ".join(annotators)))
    print("CPA instances read : {}".format(len(recs)))
    print("distinct labels    : {}".format(len(groups)))
    print("ACCEPTED -> library: {}  (m>={}, r>={}, tau>={})  -> {}".format(
        len(library), args.m, args.r, args.tau, args.out_library))
    print("review queue       : {}  -> {}".format(len(review), args.out_review))
    for c in library[:12]:
        print("   ✓ {:<26} N={} R={} A={}".format(c["name"], c["N"], c["R"], c["A"]))


if __name__ == "__main__":
    main()
