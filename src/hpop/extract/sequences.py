"""Turn CPA annotations into modelling-ready traces (the BPOP/HPOP input).

The model consumes, per trajectory, an ordered sequence of CPA occurrences (the "linearized trace"
BPOP de-linearizes) over a corpus CPA vocabulary. This reads `opencode` output (the finalized
`<prefix>.jsonl`, one annotation object per trajectory), orders each trajectory's occurrences by
their event span, and emits:

    <out>.sequences.jsonl   one line/trajectory: {trajectory_id, instance_id, repo, resolved,
                            cpa_sequence:[label...], occurrences:[{label,decision,confidence,outcome,span}]}
    <out>.vocab.json        corpus CPA vocabulary with occurrence + trajectory frequencies

Usage:
    PYTHONPATH=src .venv/bin/python -m hpop.extract.sequences \
        --opencode data/annotated/swe_rebench/cpa_demo --out data/modelling/swe_rebench/pilot
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter


def _label(c):
    return c.get("canonical_label") or c.get("candidate_label")


def _start(c):
    ev = c.get("source_event_ids") or []
    def k(e):
        try:
            return int(str(e).lstrip("e"))
        except Exception:
            return 0
    return min((k(e) for e in ev), default=0)


def iter_annotations(prefix):
    path = prefix + ".jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def to_sequence(ann):
    occ = [c for c in ann.get("cpa_instances", []) if c.get("decision") != "ABSTAIN" and _label(c)]
    occ.sort(key=_start)
    seq = [_label(c) for c in occ]
    occs = [{"label": _label(c), "decision": c.get("decision"),
             "confidence": c.get("label_confidence", c.get("confidence")),
             "outcome": c.get("outcome"), "span": c.get("source_event_ids", [])} for c in occ]
    return {
        "trajectory_id": ann.get("trajectory_id"),
        "instance_id": ann.get("_instance_id"),
        "repo": ann.get("_repo"),
        "resolved": ann.get("_resolved"),
        "cpa_sequence": seq,
        "occurrences": occs,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="CPA annotations -> modelling-ready CPA sequences.")
    ap.add_argument("--opencode", required=True, help="prefix of <prefix>.jsonl")
    ap.add_argument("--out", required=True, help="output prefix (.sequences.jsonl + .vocab.json)")
    args = ap.parse_args(argv)

    anns = list(iter_annotations(args.opencode))
    seqs = [to_sequence(a) for a in anns]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out + ".sequences.jsonl", "w", encoding="utf-8") as f:
        for s in seqs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    occ_freq = Counter(lab for s in seqs for lab in s["cpa_sequence"])
    traj_freq = Counter(lab for s in seqs for lab in set(s["cpa_sequence"]))
    vocab = [{"cpa": lab, "occurrences": occ_freq[lab], "trajectories": traj_freq[lab]}
             for lab in sorted(occ_freq, key=lambda k: -occ_freq[k])]
    json.dump(vocab, open(args.out + ".vocab.json", "w"), indent=2)

    lens = [len(s["cpa_sequence"]) for s in seqs]
    print("trajectories       : {}  -> {}.sequences.jsonl".format(len(seqs), args.out))
    print("CPA vocabulary     : {} distinct  -> {}.vocab.json".format(len(vocab), args.out))
    print("seq length         : {}".format("mean {:.1f}".format(sum(lens) / len(lens)) if lens else "n/a"))
    print("top CPAs (occ/traj): " + ", ".join("{}={}/{}".format(v["cpa"], v["occurrences"], v["trajectories"]) for v in vocab[:8]))


if __name__ == "__main__":
    main()
