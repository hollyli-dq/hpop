"""Is the cognitive (9-phase) layer a BETTER vocabulary? — head-to-head metrics.

Compares three label layers over the SAME 500-trajectory data, at the occurrence unit:
  physical  = action_type  (view/edit/run/...)   — the raw layer
  action    = CPA           (29 labels)            — the induced action layer
  cognitive = phase         (9 Miller&Cohen)       — the derived cognitive layer

A label layer earns its place iff it is (1) INFORMATIVE beyond the physical layer and (2) USEFUL for
downstream prediction. We report:

  |V|            vocabulary size
  NP             sequential predictability = I(L_t; L_t+1) / H(L_t)   (higher = more structured)
  IG_next_act    info gained about the NEXT physical action by knowing the current label:
                 H(act_t+1) - H(act_t+1 | L_t)  [bits]  (if a CPA/phase beats action_type, the
                 abstraction captures procedural structure the surface does not)
  AUC_resolved   5-fold CV AUC predicting task success (resolved) from the trajectory's label profile
  coverage       % occurrences carrying a non-generic label (excluded/none = uncovered)
  cohesion       within-label artifact-role purity (from grounded.jsonl)

Usage:
    PYTHONPATH=src .venv/bin/python scripts/eval_dictionary.py \
        --annot data/annotated/swe_rebench/cpa_rule.jsonl \
        --tokens data/interim/swe_rebench/tokens500.jsonl \
        --grounded data/modelling/swe_rebench/grounded.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict

import numpy as np


def _H(counter):
    n = sum(counter.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counter.values() if c)


def _cond_H(pairs):
    """H(Y|X) from (x,y) pairs, bits."""
    by_x = defaultdict(Counter)
    for x, y in pairs:
        by_x[x][y] += 1
    n = len(pairs)
    return sum((sum(yc.values()) / n) * _H(yc) for yc in by_x.values())


def _mi_seq(seq_of_seqs):
    """I(L_t; L_t+1) and H(L_t) over consecutive pairs, bits."""
    pairs = [(s[i], s[i + 1]) for s in seq_of_seqs for i in range(len(s) - 1)]
    if not pairs:
        return 0.0, 0.0
    cur = Counter(x for x, _ in pairs)
    nxt = Counter(y for _, y in pairs)
    Hc, Hn = _H(cur), _H(nxt)
    return Hn - _cond_H(pairs), Hc      # I(cur;next) = H(next)-H(next|cur); plus H(cur)


def _auc_cv(X, y, folds=5, l2=1.0, iters=400, lr=0.3):
    """Mean k-fold CV AUC of L2 logistic regression (numpy)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.hstack([X, np.ones((len(X), 1))])
    n = len(y)
    rng = np.arange(n)              # deterministic folds (no RNG — keeps runs reproducible)
    aucs = []
    for f in range(folds):
        te = rng[rng % folds == f]
        tr = rng[rng % folds != f]
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        if len(set(yte.tolist())) < 2 or len(set(ytr.tolist())) < 2:
            continue
        w = np.zeros(X.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-Xtr @ w))
            g = Xtr.T @ (p - ytr) / len(ytr) + l2 * w / len(ytr)
            w -= lr * g
        s = Xte @ w
        pos = s[yte == 1]; neg = s[yte == 0]
        # Mann-Whitney AUC
        wins = sum((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum() for _ in [0])
        aucs.append(wins / (len(pos) * len(neg)))
    return float(np.mean(aucs)) if aucs else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--grounded", default=None, help="optional grounded.jsonl for the cohesion metric")
    args = ap.parse_args(argv)

    # token i -> action_type, per trace
    tok_act = {}
    for l in open(args.tokens):
        if not l.strip():
            continue
        t = json.loads(l)
        tok_act[t["trace_id"]] = {ev["i"]: ev["action_type"] for ev in t["action_tokens"]}

    # occurrence-level aligned (action_type, cpa, phase) per trajectory + resolved
    act_seqs, cpa_seqs, phase_seqs = [], [], []
    rows = []   # (resolved, act_bag, cpa_bag, phase_bag)
    n_occ = n_uncov = 0
    for l in open(args.annot):
        if not l.strip():
            continue
        ann = json.loads(l)
        tid = ann.get("_instance_id"); resolved = 1 if ann.get("_resolved") else 0
        ia = tok_act.get(tid, {})
        occ = sorted(ann.get("cpa_instances", []), key=lambda c: int(str(c["start_event_id"])[1:]))
        a_s, c_s, p_s = [], [], []
        for c in occ:
            n_occ += 1
            cpa = c.get("candidate_label"); ph = c.get("phase")
            i0 = int(str(c["start_event_id"])[1:])
            a = ia.get(i0, "act")
            if cpa is None:
                n_uncov += 1
            a_s.append(a); c_s.append(cpa); p_s.append(ph)
        if a_s:
            act_seqs.append(a_s); cpa_seqs.append(c_s); phase_seqs.append(p_s)
            rows.append((resolved, Counter(a_s), Counter(c_s), Counter(p_s)))

    # next-physical-action info: pairs (label_t, act_t+1)
    def next_act_pairs(label_seqs):
        out = []
        for ls, as_ in zip(label_seqs, act_seqs):
            for i in range(len(ls) - 1):
                out.append((ls[i], as_[i + 1]))
        return out
    H_actnext = _H(Counter(p[1] for p in next_act_pairs(act_seqs)))
    layers = {"physical(action_type)": act_seqs, "action(CPA)": cpa_seqs, "cognitive(phase)": phase_seqs}

    # build bag matrices (shared vocab per layer)
    def bags(idx):
        vocab = sorted({k for r in rows for k in r[idx]})
        X = np.array([[r[idx].get(v, 0) for v in vocab] for r in rows], float)
        X = X / (X.sum(1, keepdims=True) + 1e-9)
        return X
    y = [r[0] for r in rows]

    print("HEAD-TO-HEAD over {} trajectories / {} occurrences  (resolved rate {:.2f})".format(
        len(rows), n_occ, np.mean(y)))
    print("{:<22} {:>4} {:>7} {:>12} {:>13}".format("layer", "|V|", "NP", "IG_next_act", "AUC_resolved"))
    print("-" * 62)
    for name, seqs in layers.items():
        V = len({x for s in seqs for x in s})
        mi, Hc = _mi_seq(seqs)
        NP = mi / Hc if Hc else 0.0
        ig = H_actnext - _cond_H(next_act_pairs(seqs))
        idx = {"physical(action_type)": 1, "action(CPA)": 2, "cognitive(phase)": 3}[name]
        auc = _auc_cv(bags(idx), y)
        print("{:<22} {:>4} {:>7.3f} {:>10.3f}b {:>13.3f}".format(name, V, NP, ig, auc))

    print("\ncoverage : {:.1f}% of occurrences carry a CPA label ({} uncovered)".format(
        100 * (1 - n_uncov / max(n_occ, 1)), n_uncov))

    # cohesion (optional): within-CPA artifact-role purity from grounded
    if args.grounded:
        role_by_cpa = defaultdict(Counter)
        for l in open(args.grounded):
            if not l.strip():
                continue
            g = json.loads(l)
            for o in g["occurrences"]:
                if o.get("role"):
                    role_by_cpa[o["label"]][o["role"]] += 1
        num = den = 0
        for cpa, rc in role_by_cpa.items():
            tot = sum(rc.values()); num += max(rc.values()); den += tot
        print("cohesion : {:.1f}% within-CPA artifact-role purity (modal role share, weighted)".format(
            100 * num / max(den, 1)))


if __name__ == "__main__":
    main()
