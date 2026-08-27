"""HPOP on the real SWE-rebench OpenHands CPA pilot (100 annotated trajectories).

There is no ground-truth skill library for real traces, so this script measures what the manuscript
proposes for real trajectories: held-out predictive fit under a **repository-disjoint** split
(cross-repository transfer with the library frozen), compression against sequential baselines, the
size and content of the learned library, and repair-loop statistics.

Seed segmentation. The pilot has no LLM phase segmentation attached, and consecutive CPA labels are
almost never repeated (mean run length 1.01), so we use the *maximal* oversegmentation: one seed per
CPA occurrence. Every position is then an admissible boundary, which makes the merge-only
restriction (Eq. merge-only-support) vacuous -- a deliberately conservative choice that isolates the
model from LLM seeding error.

Run:
    PYTHONPATH=src .venv/bin/python scripts/exp_real_pilot.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from hpop.inference.hpop import HPOP, HPOPConfig, FlatPoset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQ = ROOT / "data" / "modelling" / "swe_rebench" / "pilot100.sequences.jsonl"


def load(path, min_count=1):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    counts = Counter(c for r in rows for c in r["cpa_sequence"])
    vocab = [c for c, n in counts.most_common() if n >= min_count]
    idx = {c: i for i, c in enumerate(vocab)}
    for r in rows:
        r["seq"] = [idx[c] for c in r["cpa_sequence"] if c in idx]
    return rows, vocab


def repo_disjoint_split(rows, frac=0.7, seed=0):
    repos = sorted({r["repo"] for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(repos)
    cut = int(frac * len(repos))
    train_repos = set(repos[:cut])
    tr = [r for r in rows if r["repo"] in train_repos]
    te = [r for r in rows if r["repo"] not in train_repos]
    return tr, te, len(train_repos), len(repos) - len(train_repos)


# ---- sequential baselines -----------------------------------------------------------------
def ngram_nll(train, test, V, order=1, alpha=0.5):
    """Per-occurrence NLL of an add-alpha smoothed unigram (order=0) or bigram (order=1) model."""
    if order == 0:
        c = np.full(V, alpha)
        for r in train:
            for x in r["seq"]:
                c[x] += 1
        logp = np.log(c / c.sum())
        tot = sum(logp[x] for r in test for x in r["seq"])
    else:
        c = np.full((V + 1, V), alpha)                 # row V = start state
        for r in train:
            prev = V
            for x in r["seq"]:
                c[prev, x] += 1
                prev = x
        logp = np.log(c / c.sum(axis=1, keepdims=True))
        tot = 0.0
        for r in test:
            prev = V
            for x in r["seq"]:
                tot += logp[prev, x]
                prev = x
    n = sum(len(r["seq"]) for r in test)
    return -tot / n


# ---- reporting ----------------------------------------------------------------------------
def describe_skill(model, k, vocab, top=5):
    order = np.argsort(-model.theta[k])[:top]
    comp = [vocab[i] for i in order]
    edges = [(vocab[a], vocab[b]) for a in range(len(vocab)) for b in range(len(vocab))
             if model.D[k, a, b] > 0]
    return comp, edges


def repair_stats(model, corpus):
    """Share of decoded skill instances that re-execute a CPA role (an edit-test-repair cycle)."""
    total = repeat = 0
    lens = []
    for seeds in corpus:
        for a, b, _ in model.decode(seeds):
            block = sum(seeds[a:b], [])
            total += 1
            lens.append(len(block))
            if len(set(block)) < len(block):
                repeat += 1
    return repeat / max(total, 1), float(np.mean(lens)), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default=str(DEFAULT_SEQ))
    ap.add_argument("--K-max", type=int, default=8)
    ap.add_argument("--D-max", type=int, default=10)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--lam-seg", type=float, default=1.0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=0,
                    help="subsample the training set to this many trajectories (0 = all); "
                         "used for the data-efficiency sweep")
    ap.add_argument("--out", default=str(ROOT / "data" / "experiments" / "real_pilot.json"))
    args = ap.parse_args()

    rows, vocab = load(args.sequences)
    V = len(vocab)
    tr, te, n_tr_repo, n_te_repo = repo_disjoint_split(rows, seed=args.split_seed)
    if args.n_train and args.n_train < len(tr):
        idx = np.random.default_rng(args.split_seed).permutation(len(tr))[:args.n_train]
        tr = [tr[i] for i in sorted(idx)]
        n_tr_repo = len({r["repo"] for r in tr})
    tr_corpus = [[[c] for c in r["seq"]] for r in tr]        # one seed per CPA occurrence
    te_corpus = [[[c] for c in r["seq"]] for r in te]
    n_test_occ = sum(len(r["seq"]) for r in te)

    print(f"SWE-rebench OpenHands CPA pilot: {len(rows)} trajectories, {V} CPA types")
    print(f"repository-disjoint split: {len(tr)} train traj / {n_tr_repo} repos, "
          f"{len(te)} test traj / {n_te_repo} repos ({n_test_occ} held-out occurrences)")
    frac_repeat = np.mean([1 - len(set(r["seq"])) / len(r["seq"]) for r in rows])
    print(f"repeated-occurrence fraction per trajectory: {frac_repeat:.2f}\n")

    results = {"uniform": float(np.log(V)),
               "unigram": ngram_nll(tr, te, V, order=0),
               "bigram": ngram_nll(tr, te, V, order=1)}

    base = dict(V=V, K_max=args.K_max, D_max=args.D_max, lam_seg=args.lam_seg)
    models = {}
    for name, cfg in [("HSMM (composition only)", HPOPConfig(use_order=False, **base)),
                      ("HPOP, no recurrence", HPOPConfig(use_recurrence=False, **base)),
                      ("HPOP", HPOPConfig(**base))]:
        m = HPOP(cfg, rng=np.random.default_rng(args.split_seed))
        m.fit(tr_corpus, iters=args.iters, warmup=max(1, args.iters // 4))
        results[name] = -float(np.sum(m.heldout_logp(te_corpus)) / n_test_occ)
        models[name] = m
        print(f"fitted {name}: K+ = {len(m.active_skills())}", flush=True)

    flat = FlatPoset(HPOPConfig(**base)).fit(tr_corpus)
    results["Flat poset (K=1)"] = -float(np.sum(flat.logp(te_corpus)) / n_test_occ)

    print("\nHeld-out predictive fit, repository-disjoint (nats per CPA occurrence, lower better)")
    print(f"{'model':<26}{'NLL/occ':>10}{'vs uniform':>12}{'vs bigram':>12}")
    print("-" * 60)
    order = ["uniform", "unigram", "bigram", "Flat poset (K=1)", "HSMM (composition only)",
             "HPOP, no recurrence", "HPOP"]
    for name in order:
        v = results[name]
        print(f"{name:<26}{v:>10.3f}{results['uniform'] - v:>12.3f}"
              f"{results['bigram'] - v:>12.3f}")

    m = models["HPOP"]
    act = m.active_skills()
    print(f"\nLearned library: K_max = {args.K_max}, active K+ = {len(act)}")
    for k in act:
        comp, edges = describe_skill(m, k, vocab)
        share = (m.counts[k] - m.cfg.alpha / m.cfg.K_max) / max(
            (m.counts - m.cfg.alpha / m.cfg.K_max).sum(), 1e-9)
        print(f"  skill {k}  usage {share:5.1%}  composition: {', '.join(comp)}")
        for a, b in edges[:6]:
            print(f"            {a} -> {b}")

    rep, mean_len, n_inst = repair_stats(m, te_corpus)
    print(f"\nDecoded held-out instances: {n_inst}, mean length {mean_len:.1f} CPAs, "
          f"{rep:.1%} contain a re-executed role")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": vars(args), "vocab": vocab, "nll": results,
        "n_train_traj": len(tr), "n_test_traj": len(te),
        "n_train_repos": n_tr_repo, "n_test_repos": n_te_repo,
        "K_active": int(len(act)),
        "skills": {int(k): {"composition": describe_skill(m, k, vocab)[0],
                            "edges": describe_skill(m, k, vocab)[1],
                            "usage": float(m.counts[k])} for k in act},
        "repair_instance_fraction": rep,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
