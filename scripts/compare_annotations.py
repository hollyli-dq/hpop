"""Which annotation scheme is LESS random — v1 (29 CPA, induced) vs v2 (32 CPA, phase-derived)?

For each scheme we measure how much SEQUENTIAL STRUCTURE its label sequences carry beyond chance.
Null = shuffle each trajectory's labels in place (preserves length + label composition, destroys only
ORDER). A better annotation produces orderings a random permutation cannot mimic.

Metric: I(L_t ; L_t+1) in bits (transition mutual information).
  excess = I_real - mean(I_shuffled)        absolute structure beyond chance (bits)
  z      = (I_real - mean) / std(I_shuffled)  how many sigma above random  (higher = less random)
  ratio  = I_real / mean(I_shuffled)

Usage:
    PYTHONPATH=src .venv/bin/python scripts/compare_annotations.py \
        --v1 /tmp/seq_v1.jsonl --v2 data/modelling/swe_rebench/sequences.sequences.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict

import numpy as np


def _I(seqs):
    pairs = [(s[i], s[i + 1]) for s in seqs for i in range(len(s) - 1)]
    n = len(pairs)
    if not n:
        return 0.0
    cur = Counter(x for x, _ in pairs); nxt = Counter(y for _, y in pairs)
    joint = Counter(pairs)
    I = 0.0
    for (x, y), c in joint.items():
        pxy = c / n
        I += pxy * math.log2(pxy / ((cur[x] / n) * (nxt[y] / n)))
    return I


def _null(seqs, trials=200, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(trials):
        shuf = []
        for s in seqs:
            a = list(s); rng.shuffle(a); shuf.append(a)
        vals.append(_I(shuf))
    return float(np.mean(vals)), float(np.std(vals))


def evaluate(name, seqs):
    V = len({x for s in seqs for x in s})
    I = _I(seqs)
    mu, sd = _null(seqs)
    z = (I - mu) / sd if sd else float("nan")
    print("{:<18} |V|={:<3} I_real={:.4f}  I_rand={:.4f}  excess={:.4f}b  z={:>6.1f}  ratio={:.2f}x".format(
        name, V, I, mu, I - mu, z, I / mu if mu else float("nan")))
    return {"name": name, "V": V, "I": I, "excess": I - mu, "z": z}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2", required=True)
    args = ap.parse_args(argv)

    v1 = [json.loads(l)["cpa_sequence"] for l in open(args.v1) if l.strip()]
    v2rows = [json.loads(l) for l in open(args.v2) if l.strip()]
    v2_cpa = [r["cpa_sequence"] for r in v2rows]
    v2_phase = [r.get("phase_sequence") or [] for r in v2rows]

    print("STRUCTURE vs RANDOM NULL (shuffle within trajectory) — higher excess/z = LESS random\n")
    r1 = evaluate("v1 CPA (29)", v1)
    r2 = evaluate("v2 CPA (32)", v2_cpa)
    r3 = evaluate("v2 phase (9)", v2_phase)

    print("\nVERDICT")
    better = max([r1, r2], key=lambda r: r["excess"])
    print("  v1 vs v2 CPA — larger structure-beyond-chance: {}  (excess {:.4f}b vs {:.4f}b)".format(
        better["name"], r2["excess"] if better is r2 else r1["excess"],
        r1["excess"] if better is r2 else r2["excess"]))
    print("  per-symbol efficiency (excess / log2|V|):")
    for r in (r1, r2, r3):
        print("    {:<14} {:.4f} bits-excess per bit-of-vocab".format(r["name"], r["excess"] / math.log2(r["V"])))


if __name__ == "__main__":
    main()
