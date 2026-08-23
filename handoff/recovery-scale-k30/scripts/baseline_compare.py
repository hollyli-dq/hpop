"""Does theoretical guidance beat NO theory? — ladder from random floor to cognitive ceiling.

Re-labels the SAME 500-trajectory occurrence segmentation under four schemes and measures how much
sequential structure each carries beyond a shuffled null (excess transition mutual-information, bits):

  RANDOM   uniform-random label per occurrence            — no theory at all (floor)
  SURFACE  raw leading command word (pytest/grep/view/git)— surface form, no procedural meaning
  v2 CPA   29-label procedural action vocabulary           — rule guidance
  v2 phase 9 cognitive control states (Miller & Cohen)     — cognitive guidance
  (v1 CPA  shown for reference — its own segmentation/substrate)

If RANDOM ~ 0 and SURFACE < CPA/phase, the guidance demonstrably adds structure a no-theory labeling
cannot — that is the argument that the CPA/phase layers are not arbitrary.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/baseline_compare.py \
        --annot data/annotated/swe_rebench/cpa_rule.jsonl \
        --tokens data/interim/swe_rebench/tokens500.jsonl \
        --v1 /tmp/seq_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter

import numpy as np


def _I(seqs):
    pairs = [(s[i], s[i + 1]) for s in seqs for i in range(len(s) - 1)]
    n = len(pairs)
    if not n:
        return 0.0
    cur = Counter(x for x, _ in pairs); nxt = Counter(y for _, y in pairs); joint = Counter(pairs)
    return sum((c / n) * math.log2((c / n) / ((cur[x] / n) * (nxt[y] / n))) for (x, y), c in joint.items())


def _excess(seqs, trials=150, seed=0):
    rng = np.random.default_rng(seed)
    I = _I(seqs)
    vals = []
    for _ in range(trials):
        shuf = [list(s) for s in seqs]
        for a in shuf:
            rng.shuffle(a)
        vals.append(_I(shuf))
    mu, sd = float(np.mean(vals)), float(np.std(vals))
    return I, mu, (I - mu), (I / mu if mu else float("nan"))


def _lead_word(cmd):
    """Raw leading command token, atheoretical (strip `cd <dir> &&` prefixes)."""
    c = (cmd or "").strip()
    c = re.sub(r"^(cd\s+\S+\s*&&\s*)+", "", c)        # drop navigation prefix
    m = re.match(r"([A-Za-z_][\w.-]*)", c)
    w = m.group(1).lower() if m else "other"
    return {"python3": "python", "pip3": "pip"}.get(w, w)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--k", type=int, default=29, help="vocab size for RANDOM/SURFACE (match v2 CPA)")
    args = ap.parse_args(argv)

    tok = {}
    for l in open(args.tokens):
        if l.strip():
            t = json.loads(l)
            tok[t["trace_id"]] = {ev["i"]: ev for ev in t["action_tokens"]}

    # build aligned per-trajectory occurrence streams (same segmentation for all schemes)
    cpa_seqs, phase_seqs, surf_raw = [], [], []
    for l in open(args.annot):
        if not l.strip():
            continue
        ann = json.loads(l)
        ia = tok.get(ann.get("_instance_id"), {})
        occ = sorted(ann.get("cpa_instances", []), key=lambda c: int(str(c["start_event_id"])[1:]))
        cs, ps, ss = [], [], []
        for c in occ:
            cs.append(c.get("candidate_label")); ps.append(c.get("phase"))
            ev = ia.get(int(str(c["start_event_id"])[1:]), {})
            cmd = (ev.get("args") or {}).get("command") or ev.get("command")
            sub = (ev.get("args") or {}).get("command", "") if ev.get("tool_name") == "str_replace_editor" else None
            ss.append(("edit:" + sub) if sub else _lead_word(cmd))
        cpa_seqs.append(cs); phase_seqs.append(ps); surf_raw.append(ss)

    # SURFACE: cap to top-(k-1) leading words + OTHER
    freq = Counter(w for s in surf_raw for w in s)
    keep = {w for w, _ in freq.most_common(args.k - 1)}
    surf_seqs = [[w if w in keep else "OTHER" for w in s] for s in surf_raw]

    # RANDOM: uniform label in {0..k-1}, same lengths/positions
    rng = np.random.default_rng(0)
    rand_seqs = [[int(x) for x in rng.integers(0, args.k, size=len(s))] for s in surf_raw]

    v1_seqs = [json.loads(l)["cpa_sequence"] for l in open(args.v1) if l.strip()]

    schemes = [("RANDOM (no theory)", rand_seqs), ("SURFACE (command word)", surf_seqs),
               ("v2 CPA (rule)", cpa_seqs), ("v2 phase (cognitive)", phase_seqs),
               ("v1 CPA (ref)", v1_seqs)]
    print("LADDER: structure beyond a shuffled null (higher excess = less random / more meaningful)\n")
    print("{:<24} {:>4} {:>9} {:>9} {:>9} {:>8}".format("scheme", "|V|", "I_real", "I_rand", "excess", "ratio"))
    print("-" * 68)
    for name, seqs in schemes:
        V = len({x for s in seqs for x in s})
        I, mu, ex, ratio = _excess(seqs)
        print("{:<24} {:>4} {:>9.4f} {:>9.4f} {:>9.4f} {:>7.1f}x".format(name, V, I, mu, ex, ratio))


if __name__ == "__main__":
    main()
