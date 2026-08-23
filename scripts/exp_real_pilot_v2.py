"""Run 2 on the real SWE-rebench pilot: replication, then corrected-model ablations.

Two things Run 1 got wrong that this script fixes:

1. **The split seed and the model seed were the same variable.** `exp_real_pilot.py` built the model
   with `default_rng(split_seed)`, so repository partition and EM/k-means initialization moved
   together and their effects could not be separated. They are now `--split-seed` and `--model-seed`.
2. **Train/test only.** A repository-disjoint *development* partition is now available
   (`--dev-frac`) so hyperparameters can be chosen without touching test repositories.

Modes
-----
  replicate  Run 2.1 — frozen Run-1 hyperparameters, sweep splits x inits, no tuning. Answers
             "is the 0.35-nat bigram advantage stable, and how much of the variance is split vs
             initialization?"
  ablate     Run 2.3 — original vs corrected HPOP with one component removed at a time.

Every condition sees the same trajectories and the same held-out representation, so NLL/occ is
comparable within a run.

Run:
    PYTHONPATH=src .venv/bin/python scripts/exp_real_pilot_v2.py --mode replicate \
        --split-seeds 0 1 2 3 4 --model-seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from hpop.inference.hpop import HPOP, HPOPConfig, FlatPoset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQ = ROOT / "data" / "modelling" / "swe_rebench" / "pilot100.sequences.jsonl"


# ---- data ---------------------------------------------------------------------------------
def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    counts = Counter(c for r in rows for c in r["cpa_sequence"])
    vocab = [c for c, _ in counts.most_common()]
    idx = {c: i for i, c in enumerate(vocab)}
    for r in rows:
        r["seq"] = [idx[c] for c in r["cpa_sequence"]]
        # per-occurrence outcome, aligned with seq; used by failure-conditioned invalidation
        r["outcome"] = [o.get("outcome", "UNKNOWN") for o in r.get("occurrences", [])]
        if len(r["outcome"]) != len(r["seq"]):
            r["outcome"] = ["UNKNOWN"] * len(r["seq"])
    return rows, vocab


def repo_split(rows, split_seed, dev_frac=0.0, train_frac=0.7):
    """Repository-disjoint train / dev / test. dev_frac = 0 reproduces the Run-1 two-way split."""
    repos = sorted({r["repo"] for r in rows})
    rng = np.random.default_rng(split_seed)
    rng.shuffle(repos)
    n = len(repos)
    n_tr = int(train_frac * n)
    n_dev = int(dev_frac * n)
    tr_repos = set(repos[:n_tr])
    dev_repos = set(repos[n_tr:n_tr + n_dev])
    sel = lambda s: [r for r in rows if r["repo"] in s]  # noqa: E731
    test = [r for r in rows if r["repo"] not in tr_repos | dev_repos]
    return sel(tr_repos), sel(dev_repos), test


def as_corpus(rows):
    return [[[c] for c in r["seq"]] for r in rows]


def occ(rows):
    return sum(len(r["seq"]) for r in rows)


# ---- sequential baselines -----------------------------------------------------------------
def outcome_bigram_nll(train, test, V, alpha=0.5):
    """p(c_t | c_{t-1}, o_{t-1}) — the matched baseline for failure-conditioned HPOP variants.

    Those variants read the observed outcome, so their score is a *conditional* likelihood
    p(sequence | outcomes) and is not comparable to a plain bigram. This baseline gets the same
    side information.
    """
    O = {"SUCCESS": 0, "FAILURE": 1, "UNKNOWN": 2}
    c = np.full((V + 1, 3, V), alpha)
    for r in train:
        prev, po = V, 2
        for x, o in zip(r["seq"], r["outcome"]):
            c[prev, po, x] += 1
            prev, po = x, O.get(o, 2)
    lp = np.log(c / c.sum(axis=2, keepdims=True))
    tot = 0.0
    for r in test:
        prev, po = V, 2
        for x, o in zip(r["seq"], r["outcome"]):
            tot += lp[prev, po, x]
            prev, po = x, O.get(o, 2)
    return -tot / occ(test)


def ngram_nll(train, test, V, order=1, alpha=0.5):
    if order == 0:
        c = np.full(V, alpha)
        for r in train:
            for x in r["seq"]:
                c[x] += 1
        lp = np.log(c / c.sum())
        tot = sum(lp[x] for r in test for x in r["seq"])
    else:
        c = np.full((V + 1, V), alpha)
        for r in train:
            prev = V
            for x in r["seq"]:
                c[prev, x] += 1
                prev = x
        lp = np.log(c / c.sum(axis=1, keepdims=True))
        tot = 0.0
        for r in test:
            prev = V
            for x in r["seq"]:
                tot += lp[prev, x]
                prev = x
    return -tot / occ(test)


# ---- model variants -----------------------------------------------------------------------
def variant_config(name, V, args):
    """Every variant differs from `original` by exactly the flags named after it."""
    base = dict(V=V, K_max=args.K_max, D_max=args.D_max, lam_seg=args.lam_seg)
    v = {
        "HSMM":                      dict(use_order=False),
        "HPOP (original)":           dict(),
        "HPOP, no recurrence":       dict(use_recurrence=False),
        # --- corrected model, Run 2.3 -------------------------------------------------------
        "HPOP (corrected)":          dict(normalized_duration=True, failure_invalidation=True,
                                          lam_comp=1.0, lam_po=1.0),
        "corrected, no composition": dict(normalized_duration=True, failure_invalidation=True,
                                          lam_comp=0.0, lam_po=1.0),
        "corrected, no recurrence":  dict(normalized_duration=True, failure_invalidation=True,
                                          use_recurrence=False),
        "corrected, no failure-cond": dict(normalized_duration=True, failure_invalidation=False),
        "corrected, unnormalized dur": dict(normalized_duration=False, failure_invalidation=True),
        "HPOP+Seq":                  dict(normalized_duration=True, failure_invalidation=True,
                                          seq_eta=args.seq_eta),
    }[name]
    return HPOPConfig(**{**base, **v})


def fit_and_score(name, V, args, train, test, model_seed):
    cfg = variant_config(name, V, args)
    m = HPOP(cfg, rng=np.random.default_rng(model_seed))
    tr_corpus, te_corpus = as_corpus(train), as_corpus(test)
    tr_out = [r["outcome"] for r in train] if getattr(cfg, "failure_invalidation", False) else None
    te_out = [r["outcome"] for r in test] if getattr(cfg, "failure_invalidation", False) else None
    m.fit(tr_corpus, iters=args.iters, warmup=max(1, args.iters // 4), outcomes=tr_out)
    nll = -float(np.sum(m.heldout_logp(te_corpus, outcomes=te_out)) / occ(test))
    return nll, m


# ---- modes --------------------------------------------------------------------------------
def run_cell(job):
    (mode, seq_path, split_seed, model_seed, variants, argsd) = job
    args = argparse.Namespace(**argsd)
    rows, vocab = load(seq_path)
    V = len(vocab)
    train, dev, test = repo_split(rows, split_seed, dev_frac=args.dev_frac)
    t0 = time.time()
    out = {"split_seed": split_seed, "model_seed": model_seed,
           "n_train_traj": len(train), "n_dev_traj": len(dev), "n_test_traj": len(test),
           "n_train_repo": len({r['repo'] for r in train}),
           "n_test_repo": len({r['repo'] for r in test}),
           "n_test_occ": occ(test), "nll": {}}
    # baselines depend only on the split
    out["nll"]["uniform"] = float(np.log(V))
    out["nll"]["unigram"] = ngram_nll(train, test, V, order=0)
    out["nll"]["bigram"] = ngram_nll(train, test, V, order=1)
    out["nll"]["bigram+outcome"] = outcome_bigram_nll(train, test, V)
    for name in variants:
        nll, m = fit_and_score(name, V, args, train, test, model_seed)
        out["nll"][name] = nll
        out.setdefault("K_active", {})[name] = int(len(m.active_skills()))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(rows, variants, args):
    names = [n for n in ["uniform", "unigram", "bigram", "bigram+outcome"]
             if n in rows[0]["nll"]] + variants
    print(f"\n{'model':<30}{'NLL/occ':>10}{'95% CI (splits)':>20}{'sd across inits':>18}"
          f"{'vs bigram':>12}")
    print("-" * 92)
    # collect every model's per-split values FIRST; the reference baseline must exist before any
    # gap is computed
    per_split = {}
    for name in names:
        by_split = {}
        for r in rows:
            by_split.setdefault(r["split_seed"], []).append(r["nll"][name])
        per_split[name] = by_split

    ref = "bigram" if "bigram" in per_split else names[0]
    ref_means = {s: float(np.mean(v)) for s, v in per_split[ref].items()}

    summary = {}
    for name in names:
        by_split = per_split[name]
        means = [float(np.mean(v)) for v in by_split.values()]
        init_sd = float(np.mean([np.std(v, ddof=1) if len(v) > 1 else 0.0
                                 for v in by_split.values()]))
        m = float(np.mean(means))
        ci = 1.96 * float(np.std(means, ddof=1) / np.sqrt(len(means))) if len(means) > 1 else 0.0
        summary[name] = {"mean": m, "ci95_across_splits": ci, "sd_across_inits": init_sd,
                         "per_split_mean": [float(np.mean(by_split[s])) for s in sorted(by_split)]}
        gap = m - float(np.mean(list(ref_means.values())))
        print(f"{name:<30}{m:>10.3f}{'±' + format(ci, '.3f'):>20}{init_sd:>18.4f}{gap:>+12.3f}")

    # paired per-split comparison — the statistic that actually answers "is the gap stable"
    print(f"\nPaired per-split gap vs bigram (positive = worse than bigram)")
    for name in variants:
        d = [float(np.mean(per_split[name][s]) - ref_means[s]) for s in sorted(per_split[name])]
        m = float(np.mean(d))
        ci = 1.96 * float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        wins = sum(x < 0 for x in d)
        print(f"  {name:<30}{m:>+8.3f} ± {ci:.3f}   per-split {[round(x, 3) for x in d]}"
              f"   beats bigram on {wins}/{len(d)} splits")
        summary[name]["gap_vs_bigram"] = {"mean": m, "ci95": ci, "per_split": d,
                                          "splits_beating_bigram": wins}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["replicate", "ablate"], default="replicate")
    ap.add_argument("--sequences", default=str(DEFAULT_SEQ))
    ap.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dev-frac", type=float, default=0.0,
                    help="repo fraction held out for tuning; 0 reproduces the Run-1 two-way split")
    ap.add_argument("--K-max", type=int, default=10)
    ap.add_argument("--D-max", type=int, default=12)
    ap.add_argument("--lam-seg", type=float, default=3.0)
    ap.add_argument("--seq-eta", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=18)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.variants is None:
        args.variants = (["HSMM", "HPOP (original)", "HPOP, no recurrence"]
                         if args.mode == "replicate" else
                         ["HPOP (original)", "HPOP (corrected)", "corrected, no composition",
                          "corrected, no recurrence", "corrected, no failure-cond",
                          "corrected, unnormalized dur", "HPOP+Seq"])
    out_path = Path(args.out or (ROOT / "data" / "experiments" / f"run2_{args.mode}.json"))

    jobs = [(args.mode, args.sequences, s, i, args.variants, vars(args))
            for s in args.split_seeds for i in args.model_seeds]
    print(f"Run 2 [{args.mode}] — {len(args.split_seeds)} splits x {len(args.model_seeds)} inits "
          f"x {len(args.variants)} variants = {len(jobs) * len(args.variants)} fits", flush=True)

    if args.workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
            rows = pool.map(run_cell, jobs)
    else:
        rows = [run_cell(j) for j in jobs]

    # persist raw cells immediately — summarizing must never be able to lose hours of fits
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_name(out_path.stem + "_cells.json")
    raw_path.write_text(json.dumps({"config": vars(args), "cells": rows}, indent=2))
    print(f"raw cells -> {raw_path}", flush=True)

    for r in rows:
        print(f"  split {r['split_seed']} init {r['model_seed']}: "
              f"{r['n_train_traj']}/{r['n_test_traj']} traj, {r['n_test_occ']} occ, "
              f"{r['seconds']}s", flush=True)

    summary = summarize(rows, args.variants, args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"config": vars(args), "cells": rows, "summary": summary},
                                   indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
