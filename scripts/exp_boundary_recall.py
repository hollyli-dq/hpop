"""Initialization audit: how far can merge-only HPOP recover when the LLM seeds miss boundaries?

The manuscript requires this check ("Required initialization audit", Sec. 5): because HPOP may only
merge seed segments, a true skill boundary that is absent from the seed set is unrecoverable *by
construction*. This script sweeps the seed-boundary recall of the simulated LLM oversegmentation and
measures how the learned segmentation degrades, alongside the analytic ceiling.

Run:
    PYTHONPATH=src .venv/bin/python scripts/exp_boundary_recall.py --seeds 3 --traces 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hpop.eval.metrics import evaluate
from hpop.inference.hpop import HPOP, HPOPConfig
from hpop.synth.generator import sample_corpus, seeds_of

ROOT = Path(__file__).resolve().parents[1]


def seed_boundary_recall(traces):
    """Ceiling on any merge-only model: share of true boundaries present in the seed set."""
    hit = tot = 0
    for t in traces:
        s = set(t.seed_boundaries)
        hit += sum(b in s for b in t.true_boundaries)
        tot += len(t.true_boundaries)
    return hit / tot if tot else 1.0


def oversegmentation_ratio(traces):
    """Seed segments per true skill instance."""
    return float(np.mean([(len(t.seed_boundaries) + 1) / len(t.skill_labels) for t in traces]))


def run_point(job):
    recall, seed, n_traces, K_true, V, K_max, D_max, iters = job
    world, traces = sample_corpus(seed=seed, n_traces=n_traces, K_true=K_true, V=V,
                                  boundary_recall=recall)
    split = int(0.7 * len(traces))
    tr_traces, te_traces = traces[:split], traces[split:]
    tr_corpus = [seeds_of(t) for t in tr_traces]
    te_corpus = [seeds_of(t) for t in te_traces]

    m = HPOP(HPOPConfig(V=V, K_max=K_max, D_max=D_max), rng=np.random.default_rng(seed))
    m.fit(tr_corpus, iters=iters, warmup=max(1, iters // 4))
    decoded = [m.decode(s) for s in te_corpus]
    res = evaluate(world, te_traces, decoded, K_max, m.D, m.global_structure(te_corpus))
    n_occ = sum(len(t.cpas) for t in te_traces)
    res["heldout_nll_per_occ"] = -float(np.sum(m.heldout_logp(te_corpus)) / n_occ)
    res["seed_boundary_recall"] = seed_boundary_recall(te_traces)
    res["oversegmentation_ratio"] = oversegmentation_ratio(te_traces)
    res["target_recall"] = recall
    res["seed"] = seed
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--traces", type=int, default=30)
    ap.add_argument("--K-true", type=int, default=4)
    ap.add_argument("--V", type=int, default=12)
    ap.add_argument("--K-max", type=int, default=6)
    ap.add_argument("--D-max", type=int, default=8)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--recalls", type=float, nargs="+", default=[1.0, 0.9, 0.75, 0.5, 0.25])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "data" / "experiments" / "boundary_recall.json"))
    args = ap.parse_args()

    jobs = [(r, s, args.traces, args.K_true, args.V, args.K_max, args.D_max, args.iters)
            for r in args.recalls for s in range(args.seeds)]
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers) as pool:
            rows = pool.map(run_point, jobs)
    else:
        rows = [run_point(j) for j in jobs]

    header = (f"{'target':>7}{'seed bnd recall':>17}{'overseg':>9}"
              f"{'model bnd recall':>18}{'bnd F1':>9}{'skill ARI':>11}"
              f"{'local edge F1':>15}{'NLL/occ':>10}")
    print("\nMerge-only initialization audit (mean over seeds)")
    print(header)
    print("-" * len(header))
    summary = []
    for r in args.recalls:
        sel = [x for x in rows if x["target_recall"] == r]
        row = {
            "target_recall": r,
            "seed_boundary_recall": float(np.mean([x["seed_boundary_recall"] for x in sel])),
            "oversegmentation_ratio": float(np.mean([x["oversegmentation_ratio"] for x in sel])),
            "boundary_recall": float(np.mean([x["boundary_recall"] for x in sel])),
            "boundary_f1": float(np.mean([x["boundary_f1"] for x in sel])),
            "skill_ari": float(np.mean([x["skill_ari"] for x in sel])),
            "local_rel_f1": float(np.mean([x["local_rel_f1"] for x in sel])),
            "heldout_nll_per_occ": float(np.mean([x["heldout_nll_per_occ"] for x in sel])),
        }
        summary.append(row)
        print(f"{r:>7.2f}{row['seed_boundary_recall']:>17.3f}"
              f"{row['oversegmentation_ratio']:>9.2f}{row['boundary_recall']:>18.3f}"
              f"{row['boundary_f1']:>9.3f}{row['skill_ari']:>11.3f}"
              f"{row['local_rel_f1']:>15.3f}{row['heldout_nll_per_occ']:>10.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "rows": rows, "summary": summary}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
