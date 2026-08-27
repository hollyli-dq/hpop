"""Run 2.3 on SYNTHETIC data: does each correction improve recovery of the *true* structure?

Why synthetic rather than the real pilot: the real CPA layer is rule-based silver, where the
outcome field carries no signal (re-execution follows FAILURE 77.2% of the time vs 77.8% otherwise),
so the failure-conditioned correction cannot be evaluated there at all. The generator records which
verification actually failed, and that signal is real (91.1% vs 74.2%), so every correction can be
tested against known ground truth.

Primary metrics are structural — skill ARI, boundary F1, local/global edge F1, library size — not
next-action NLL. Note that the failure-conditioned variants score a *conditional* likelihood
p(sequence | outcomes), so their NLL column is not comparable with the rest; the structural columns
are unaffected.

Run:
    PYTHONPATH=src .venv/bin/python scripts/exp_synthetic_ablate.py --seeds 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hpop.eval.metrics import evaluate
from hpop.inference.hpop import HPOP, HPOPConfig
from hpop.synth.generator import sample_corpus, seeds_of

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    "HPOP (original)":            dict(),
    "HPOP (corrected)":           dict(normalized_duration=True, failure_invalidation=True),
    "  – composition":            dict(normalized_duration=True, failure_invalidation=True,
                                       lam_comp=0.0),
    "  – recurrence":             dict(normalized_duration=True, failure_invalidation=True,
                                       use_recurrence=False),
    "  – failure-conditioning":   dict(normalized_duration=True, failure_invalidation=False),
    "  – normalized duration":    dict(normalized_duration=False, failure_invalidation=True),
    "HPOP+Seq":                   dict(normalized_duration=True, failure_invalidation=True,
                                       seq_eta=1.0),
}
NEEDS_OUTCOME = {"HPOP (corrected)", "  – composition", "  – recurrence",
                 "  – normalized duration", "HPOP+Seq"}


def run_seed(job):
    seed, n_traces, K_true, V, K_max, D_max, iters = job
    world, traces = sample_corpus(seed=seed, n_traces=n_traces, K_true=K_true, V=V)
    split = int(0.7 * len(traces))
    tr_traces, te_traces = traces[:split], traces[split:]
    tr_corpus = [seeds_of(t) for t in tr_traces]
    te_corpus = [seeds_of(t) for t in te_traces]
    tr_out = [t.outcomes for t in tr_traces]
    te_out = [t.outcomes for t in te_traces]
    n_occ = sum(len(t.cpas) for t in te_traces)

    out = {"seed": seed, "failure_rate": float(np.mean([t.failure_rate() for t in traces])),
           "results": {}}
    for name, kw in VARIANTS.items():
        t0 = time.time()
        cfg = HPOPConfig(V=V, K_max=K_max, D_max=D_max, **kw)
        m = HPOP(cfg, rng=np.random.default_rng(seed))
        use_out = name in NEEDS_OUTCOME and cfg.failure_invalidation
        m.fit(tr_corpus, iters=iters, warmup=max(1, iters // 4),
              outcomes=tr_out if use_out else None)
        decoded = [m.decode(s) for s in te_corpus]
        res = evaluate(world, te_traces, decoded, K_max, m.D, m.global_structure(te_corpus))
        res["nll_per_occ"] = -float(np.sum(m.heldout_logp(
            te_corpus, outcomes=te_out if use_out else None)) / n_occ)
        res["K_active"] = int(len(m.active_skills()))
        res["K_error"] = abs(res["K_active"] - K_true)
        res["conditional_nll"] = bool(use_out)
        res["seconds"] = round(time.time() - t0, 1)
        out["results"][name] = res
    return out


COLUMNS = [("skill_ari", "Skill ARI", 1), ("boundary_f1", "Bnd F1", 1),
           ("local_rel_f1", "Local edge F1", 1), ("global_rel_f1", "Global edge F1", 1),
           ("nll_per_occ", "NLL/occ", -1), ("K_error", "|K+-K|", -1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--traces", type=int, default=40)
    ap.add_argument("--K-true", type=int, default=4)
    ap.add_argument("--V", type=int, default=12)
    ap.add_argument("--K-max", type=int, default=6)
    ap.add_argument("--D-max", type=int, default=8)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "data" / "experiments" / "synthetic_ablate.json"))
    args = ap.parse_args()

    jobs = [(s, args.traces, args.K_true, args.V, args.K_max, args.D_max, args.iters)
            for s in range(args.seeds)]
    print(f"synthetic ablation — {args.seeds} seeds x {len(VARIANTS)} variants = "
          f"{args.seeds * len(VARIANTS)} fits", flush=True)
    if args.workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
            rows = pool.map(run_seed, jobs)
    else:
        rows = [run_seed(j) for j in jobs]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.with_name(out_path.stem + "_cells.json").write_text(
        json.dumps({"config": vars(args), "cells": rows}, indent=2))

    print(f"\nmean failure rate: {np.mean([r['failure_rate'] for r in rows]):.3f}")
    head = f"{'variant':<26}" + "".join(f"{lab:>17}" for _, lab, _ in COLUMNS)
    print("\n" + head + "\n" + "-" * len(head))
    summary = {}
    base = None
    for name in VARIANTS:
        cells = [r["results"][name] for r in rows]
        row, line = {}, f"{name:<26}"
        for key, _, _ in COLUMNS:
            v = [c[key] for c in cells]
            m = float(np.mean(v))
            ci = 1.96 * float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            row[key] = [m, ci]
            mark = "*" if key == "nll_per_occ" and cells[0]["conditional_nll"] else " "
            line += (f"{m:>10.3f}±{ci:.3f}" if key != "K_error" else f"{m:>16.2f}") + mark
        summary[name] = row
        print(line)
        if name == "HPOP (corrected)":
            base = row
    print("\n* conditional NLL — p(sequence | outcomes); not comparable with unstarred rows.")

    if base:
        print("\nEffect of removing each component from the corrected model "
              "(negative = that component was helping):")
        for name in VARIANTS:
            if not name.startswith("  – "):
                continue
            d = {k: summary[name][k][0] - base[k][0] for k, _, _ in COLUMNS}
            print(f"  {name.strip():<26} ARI {d['skill_ari']:+.3f}   "
                  f"local edge F1 {d['local_rel_f1']:+.3f}   bnd F1 {d['boundary_f1']:+.3f}")

    out_path.write_text(json.dumps({"config": vars(args), "cells": rows,
                                    "summary": summary}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
