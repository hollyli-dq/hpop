"""Construct the NO-INSTRUCTION baseline annotations and the structure-ladder table.

For the same 500-trajectory occurrence segmentation, produce four labelings that receive NO procedural
taxonomy, save each as a first-class annotation file (parallel to the guided v1/v2 sequences), then
score every scheme against the shuffle null.

Constructed baselines (saved to data/modelling/swe_rebench/baseline_<name>.sequences.jsonl):
  random   — uniform-random label id (no theory floor)
  cmdword  — naive: literal leading command word (pytest/grep/view...), natural vocabulary
  actobj   — smart: action_type + artifact role (run:test, edit:source...), natural vocabulary

Guided references (already on disk): v1 CPA, v2 CPA, v2 phase.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/build_baselines.py \
        --annot data/annotated/swe_rebench/cpa_rule.jsonl \
        --tokens data/interim/swe_rebench/tokens500.jsonl \
        --v1 /tmp/seq_v1.jsonl --outdir data/modelling/swe_rebench
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "src")
from hpop.extract.ground import _artifact_of


def _I(seqs):
    pairs = [(s[i], s[i + 1]) for s in seqs for i in range(len(s) - 1)]
    n = len(pairs)
    if not n:
        return 0.0
    cur = Counter(x for x, _ in pairs); nxt = Counter(y for _, y in pairs); jt = Counter(pairs)
    return sum((c / n) * math.log2((c / n) / ((cur[x] / n) * (nxt[y] / n))) for (x, y), c in jt.items())


def _score(seqs, trials=150, seed=0):
    rng = np.random.default_rng(seed)
    I = _I(seqs)
    vals = []
    for _ in range(trials):
        sh = [list(s) for s in seqs]
        for a in sh:
            rng.shuffle(a)
        vals.append(_I(sh))
    mu, sd = float(np.mean(vals)), float(np.std(vals))
    V = len({x for s in seqs for x in s})
    return {"V": V, "I": I, "rand": mu, "excess": I - mu, "ratio": I / mu if mu else float("nan"),
            "eff": (I - mu) / math.log2(V) if V > 1 else 0.0}


def _lead(cmd):
    c = re.sub(r"^(cd\s+\S+\s*&&\s*)+", "", (cmd or "").strip())
    m = re.match(r"([A-Za-z_][\w.-]*)", c)
    w = m.group(1).lower() if m else "other"
    return {"python3": "python", "pip3": "pip"}.get(w, w)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--k", type=int, default=29)
    args = ap.parse_args(argv)

    tok = {}
    for l in open(args.tokens):
        if l.strip():
            t = json.loads(l)
            tok[t["trace_id"]] = {ev["i"]: ev for ev in t["action_tokens"]}

    meta, cmd_s, act_s, cpa_s, phase_s = [], [], [], [], []
    for l in open(args.annot):
        if not l.strip():
            continue
        ann = json.loads(l)
        ia = tok.get(ann["_instance_id"], {})
        occ = sorted(ann.get("cpa_instances", []), key=lambda c: int(str(c["start_event_id"])[1:]))
        cm, ac, cp, ph = [], [], [], []
        for c in occ:
            ev = ia.get(int(str(c["start_event_id"])[1:]), {})
            a = ev.get("args") or {}
            sub = a.get("command", "") if ev.get("tool_name") == "str_replace_editor" else None
            cm.append("edit:" + sub if sub else _lead(a.get("command") or ev.get("command")))
            art = _artifact_of(ev)
            ac.append("{}:{}".format(ev.get("action_type", "?"), art[0] if art else "none"))
            cp.append(c.get("candidate_label")); ph.append(c.get("phase"))
        meta.append({"instance_id": ann["_instance_id"], "repo": ann.get("_repo"), "resolved": ann.get("_resolved")})
        cmd_s.append(cm); act_s.append(ac); cpa_s.append(cp); phase_s.append(ph)

    # SAVE the constructed no-instruction annotations as first-class artifacts
    os.makedirs(args.outdir, exist_ok=True)
    saved = {"cmdword": cmd_s, "actobj": act_s}
    for name, seqs in saved.items():
        path = os.path.join(args.outdir, "baseline_{}.sequences.jsonl".format(name))
        with open(path, "w", encoding="utf-8") as f:
            for m, s in zip(meta, seqs):
                f.write(json.dumps({**m, "label_sequence": s}, ensure_ascii=False) + "\n")
        print("wrote {}  ({} traj, |V|={})".format(path, len(seqs), len({x for ss in seqs for x in ss})))

    v1_s = [json.loads(l)["cpa_sequence"] for l in open(args.v1) if l.strip()]

    ladder = [
        ("cmdword (no instruction)", cmd_s),
        ("actobj  (no instruction)", act_s),
        ("v1 CPA  (induced)", v1_s),
        ("v2 CPA  (cognitive)", cpa_s),
    ]
    print("\nSTRUCTURE LADDER  (excess = bits beyond shuffle null; eff = excess per bit of vocab)\n")
    print("{:<26} {:>4} {:>8} {:>8} {:>9} {:>7} {:>7}".format("scheme", "|V|", "I_real", "I_rand", "excess", "ratio", "eff"))
    print("-" * 74)
    out = []
    for name, seqs in ladder:
        r = _score(seqs)
        out.append({"scheme": name, **r})
        print("{:<26} {:>4} {:>8.4f} {:>8.4f} {:>9.4f} {:>6.1f}x {:>7.4f}".format(
            name, r["V"], r["I"], r["rand"], r["excess"], r["ratio"], r["eff"]))
    json.dump(out, open(os.path.join(args.outdir, "ladder.json"), "w"), indent=2)
    print("\nwrote {}/ladder.json".format(args.outdir))


if __name__ == "__main__":
    main()
