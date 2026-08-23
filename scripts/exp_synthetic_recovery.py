"""Synthetic-recovery experiment for HPOP (fills the manuscript's `tab:synthetic`).

Question: when the ground truth is known, does the merge-only semi-Markov segmentation with a
library of *revisitable local partial orders* recover boundaries, skill assignments, local order,
and global order better than (a) segmentation without order, (b) order without segmentation, and
(c) the same model without the recurrent validity state?

Methods compared
----------------
  HSMM                  merge-only segmentation, composition only (D = 0). No partial order.
  Flat poset (K=1)      one revisitable poset over the whole CPA vocabulary. No segmentation.
  HPOP, fixed bounds    true instance boundaries supplied; only assignment + structure learned.
  HPOP, joint           LLM-seed lattice, merge-only segmentation + assignment + structure.
  HPOP, no recurrence   ablation: sigmoid(omega) = 0, so nothing is ever invalidated.

Every method is trained on the train split and scored on the *same* held-out representation (the
seed lattice), so held-out NLL is comparable; `HPOP, fixed bounds` gets oracle boundaries only at
training time.

Run:
    PYTHONPATH=src .venv/bin/python scripts/exp_synthetic_recovery.py --seeds 5 --traces 40
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hpop.eval.metrics import evaluate, merge_only_feasible
from hpop.inference.hpop import HPOP, HPOPConfig, FlatPoset
from hpop.synth.generator import sample_corpus, seeds_of

ROOT = Path(__file__).resolve().parents[1]


def true_instance_seeds(trace):
    """Oracle 'seed' segmentation: exactly one seed per true skill instance."""
    return [trace.cpas[a:b] for a, b in trace.instance_spans]


def per_occurrence(logps, traces):
    n = sum(len(t.cpas) for t in traces)
    return float(np.sum(logps) / n)


def run_seed(seed, n_traces, K_true, V, K_max, D_max, iters, boundary_recall, verbose=False):
    world, traces = sample_corpus(seed=seed, n_traces=n_traces, K_true=K_true, V=V,
                                  boundary_recall=boundary_recall)
    split = int(0.7 * len(traces))
    tr_traces, te_traces = traces[:split], traces[split:]
    tr_corpus = [seeds_of(t) for t in tr_traces]
    te_corpus = [seeds_of(t) for t in te_traces]

    base = dict(V=V, K_max=K_max, D_max=D_max, theta_steps=6)
    results = {}

    def finish(name, model, decode_corpus, traces_eval):
        decoded = [model.decode(s) for s in decode_corpus]
        glob = model.global_structure(decode_corpus)
        m = evaluate(world, traces_eval, decoded, K_max, model.D, glob)
        m["heldout_logp_per_occ"] = per_occurrence(model.heldout_logp(te_corpus), te_traces)
        m["K_active"] = int(len(model.active_skills()))
        m["K_error"] = abs(m["K_active"] - K_true)
        results[name] = m

    t0 = time.time()

    # --- HSMM: segmentation + composition, no partial order --------------------------------
    hsmm = HPOP(HPOPConfig(use_order=False, **base), rng=np.random.default_rng(seed))
    hsmm.fit(tr_corpus, iters=iters, warmup=iters, verbose=verbose)
    finish("HSMM", hsmm, te_corpus, te_traces)

    # --- Flat poset (K=1), no segmentation --------------------------------------------------
    flat = FlatPoset(HPOPConfig(**base)).fit(tr_corpus)
    flat_lp = flat.logp(te_corpus)
    # a single skill covering each trace end to end
    flat_decoded = [[(0, len(s), 0)] for s in te_corpus]
    flat_metrics = evaluate(world, te_traces, flat_decoded, 1, [flat.D], [])
    flat_metrics["heldout_logp_per_occ"] = per_occurrence(flat_lp, te_traces)
    flat_metrics["K_active"] = 1
    flat_metrics["K_error"] = abs(1 - K_true)
    results["Flat poset (K=1)"] = flat_metrics

    # --- HPOP with oracle boundaries --------------------------------------------------------
    fixed_corpus = [true_instance_seeds(t) for t in tr_traces]
    fixed = HPOP(HPOPConfig(**{**base, "D_max": 1}), rng=np.random.default_rng(seed))
    fixed.fit(fixed_corpus, iters=iters, warmup=max(1, iters // 4), verbose=verbose)
    fixed.cfg.D_max = D_max                       # score held-out on the same seed lattice
    finish("HPOP, fixed bounds", fixed, te_corpus, te_traces)

    # --- HPOP joint (the model) --------------------------------------------------------------
    joint = HPOP(HPOPConfig(**base), rng=np.random.default_rng(seed))
    joint.fit(tr_corpus, iters=iters, warmup=max(1, iters // 4), verbose=verbose)
    finish("HPOP, joint", joint, te_corpus, te_traces)

    # --- ablation: no recurrent validity state ------------------------------------------------
    norec = HPOP(HPOPConfig(**{**base, "use_recurrence": False}), rng=np.random.default_rng(seed))
    norec.fit(tr_corpus, iters=iters, warmup=max(1, iters // 4), verbose=verbose)
    finish("HPOP, no recurrence", norec, te_corpus, te_traces)

    meta = {
        "seed": seed,
        "seconds": round(time.time() - t0, 1),
        "merge_only_feasible_train": merge_only_feasible(tr_traces),
        "mean_trace_len": float(np.mean([len(t.cpas) for t in traces])),
        "mean_seeds": float(np.mean([len(s) for s in tr_corpus])),
        "repeat_fraction": float(np.mean([
            1 - len(set(t.cpas[a:b])) / (b - a) for t in traces for a, b in t.instance_spans])),
    }
    return results, meta


METHOD_ORDER = ["HSMM", "Flat poset (K=1)", "HPOP, fixed bounds", "HPOP, joint",
                "HPOP, no recurrence"]
COLUMNS = [("skill_ari", "Skill ARI", 1), ("boundary_f1", "Bnd F1", 1),
           ("local_rel_f1", "Local edge F1", 1), ("global_rel_f1", "Global edge F1", 1),
           ("heldout_logp_per_occ", "NLL/occ", -1), ("K_error", "|K+ - K|", -1)]


def aggregate(all_results):
    agg = {}
    for method in METHOD_ORDER:
        rows = [r[method] for r in all_results if method in r]
        agg[method] = {}
        for key, _, _ in COLUMNS:
            vals = [row.get(key, float("nan")) for row in rows]
            vals = [v for v in vals if v == v]
            if not vals:
                continue
            mean = float(np.mean(vals))
            if key == "heldout_logp_per_occ":
                mean = -mean                        # report NLL
                vals = [-v for v in vals]
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            agg[method][key] = (mean, 1.96 * sem)
    return agg


def to_latex(agg):
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lcccccc}",
             r"\toprule",
             "Method & Skill ARI $\\uparrow$ & Boundary F1 $\\uparrow$ & Local edge F1 $\\uparrow$ "
             "& Global edge F1 $\\uparrow$ & NLL/occ $\\downarrow$ & $|\\widehat K_+-K_+|$ "
             "$\\downarrow$\\\\", r"\midrule"]
    for method in METHOD_ORDER:
        cells = []
        for key, _, _ in COLUMNS:
            if key not in agg[method]:
                cells.append("--")
                continue
            m, ci = agg[method][key]
            cells.append(f"{m:.3f}\\,$\\pm$\\,{ci:.3f}" if key != "K_error" else f"{m:.2f}")
        name = method.replace("&", r"\&")
        lines.append(f"{name} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Synthetic recovery, mean $\pm$ 95\% CI over seeds. Higher is better "
              r"except NLL and library-size error.}", r"\label{tab:synthetic}", r"\end{table}"]
    return "\n".join(lines)


def to_text(agg):
    head = f"{'Method':<22}" + "".join(f"{lab:>18}" for _, lab, _ in COLUMNS)
    out = [head, "-" * len(head)]
    for method in METHOD_ORDER:
        row = f"{method:<22}"
        for key, _, _ in COLUMNS:
            if key not in agg[method]:
                row += f"{'--':>18}"
            else:
                m, ci = agg[method][key]
                row += f"{m:>11.3f}±{ci:.3f}" if key != "K_error" else f"{m:>18.2f}"
        out.append(row)
    return "\n".join(out)


def _run_job(job):
    seed, n_traces, K_true, V, K_max, D_max, iters, brecall, verbose = job
    return run_seed(seed, n_traces, K_true, V, K_max, D_max, iters, brecall, verbose=verbose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--traces", type=int, default=40)
    ap.add_argument("--K-true", type=int, default=4)
    ap.add_argument("--V", type=int, default=12)
    ap.add_argument("--K-max", type=int, default=6)
    ap.add_argument("--D-max", type=int, default=8)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--boundary-recall", type=float, default=1.0)
    ap.add_argument("--out", default=str(ROOT / "data" / "experiments" / "synthetic_recovery.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    jobs = [(s, args.traces, args.K_true, args.V, args.K_max, args.D_max, args.iters,
             args.boundary_recall, args.verbose) for s in range(args.seeds)]
    if args.workers > 1 and args.seeds > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(args.workers, args.seeds)) as pool:
            outputs = pool.map(_run_job, jobs)
    else:
        outputs = [_run_job(j) for j in jobs]

    all_results, metas = [], []
    for res, meta in outputs:
        all_results.append(res)
        metas.append(meta)
        print(f"seed {meta['seed']} done in {meta['seconds']}s  "
              f"(merge-only feasible {meta['merge_only_feasible_train']:.2f}, "
              f"repeat fraction {meta['repeat_fraction']:.2f})", flush=True)

    agg = aggregate(all_results)
    print()
    print(to_text(agg))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": vars(args),
        "per_seed": all_results,
        "meta": metas,
        "aggregate": {m: {k: list(v) for k, v in d.items()} for m, d in agg.items()},
    }, indent=2))
    (out.with_suffix(".tex")).write_text(to_latex(agg))
    print(f"\nwrote {out}\nwrote {out.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
