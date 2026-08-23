"""Does HPOP generalize across *valid* reorderings where a sequence model cannot?

Held-out next-occurrence likelihood rewards memorizing the serialization order the agent happened to
use. The claim that motivates partial orders is different: two executions that differ only in the
order of *incomparable* actions are the same program, and a model of the program should score them
alike -- while still rejecting an order that breaks a genuine dependency.

This script measures exactly that, non-circularly, on synthetic data where incomparability is known
from the ground-truth local posets (never from the fitted model):

  valid swap    -- exchange two adjacent occurrences that are incomparable in the true local poset
  invalid swap  -- exchange two adjacent occurrences where the true poset requires the first

For each model we report the mean change in per-occurrence NLL. A model of the *program* should show
delta_valid ~ 0 and delta_invalid > 0; a model of the *serialization* penalizes both. The gap
delta_invalid - delta_valid is the discrimination score.

Run:
    PYTHONPATH=src .venv/bin/python scripts/exp_order_invariance.py --seeds 3 --traces 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hpop.inference.hpop import HPOP, HPOPConfig
from hpop.synth.generator import sample_corpus, seeds_of

ROOT = Path(__file__).resolve().parents[1]


# ---- baselines ----------------------------------------------------------------------------
class Bigram:
    def __init__(self, V, alpha=0.5):
        self.V, self.alpha = V, alpha

    def fit(self, seqs):
        c = np.full((self.V + 1, self.V), self.alpha)
        for s in seqs:
            prev = self.V
            for x in s:
                c[prev, x] += 1
                prev = x
        self.logp = np.log(c / c.sum(axis=1, keepdims=True))
        return self

    def sequence_logp(self, s):
        tot, prev = 0.0, self.V
        for x in s:
            tot += self.logp[prev, x]
            prev = x
        return float(tot)


# ---- swap construction --------------------------------------------------------------------
def find_swaps(trace, world):
    """(valid, invalid) lists of positions t meaning 'swap occurrences t and t+1'."""
    valid, invalid = [], []
    locals_ = world.local_matrices()
    for (a, b), k in zip(trace.instance_spans, trace.skill_labels):
        D = locals_[k]
        for t in range(a, b - 1):
            u, v = trace.cpas[t], trace.cpas[t + 1]
            if u == v:
                continue
            if D[u, v] > 0:
                invalid.append(t)                 # u must precede v; swapping breaks it
            elif D[v, u] == 0:
                valid.append(t)                   # genuinely incomparable
    return valid, invalid


def apply_swap(trace, t):
    seq = list(trace.cpas)
    seq[t], seq[t + 1] = seq[t + 1], seq[t]
    cuts = [0] + list(trace.seed_boundaries) + [len(seq)]
    return [seq[i:j] for i, j in zip(cuts, cuts[1:]) if j > i], seq


def run_seed(job):
    seed, n_traces, K_true, V, K_max, D_max, iters, n_swaps = job
    rng = np.random.default_rng(1000 + seed)
    world, traces = sample_corpus(seed=seed, n_traces=n_traces, K_true=K_true, V=V)
    split = int(0.7 * len(traces))
    tr_traces, te_traces = traces[:split], traces[split:]
    tr_corpus = [seeds_of(t) for t in tr_traces]

    hpop = HPOP(HPOPConfig(V=V, K_max=K_max, D_max=D_max), rng=np.random.default_rng(seed))
    hpop.fit(tr_corpus, iters=iters, warmup=max(1, iters // 4))
    hsmm = HPOP(HPOPConfig(V=V, K_max=K_max, D_max=D_max, use_order=False),
                rng=np.random.default_rng(seed))
    hsmm.fit(tr_corpus, iters=iters, warmup=iters)
    bigram = Bigram(V).fit([t.cpas for t in tr_traces])

    models = {"HPOP": hpop, "HSMM": hsmm}
    deltas = {m: {"valid": [], "invalid": []} for m in list(models) + ["Bigram"]}

    for tr in te_traces:
        base_seeds = seeds_of(tr)
        base = {name: m.heldout_logp([base_seeds])[0] for name, m in models.items()}
        base["Bigram"] = bigram.sequence_logp(tr.cpas)
        n_occ = len(tr.cpas)
        valid, invalid = find_swaps(tr, world)
        for kind, positions in (("valid", valid), ("invalid", invalid)):
            if not positions:
                continue
            pick = rng.choice(positions, size=min(n_swaps, len(positions)), replace=False)
            for t in np.atleast_1d(pick):
                sw_seeds, sw_seq = apply_swap(tr, int(t))
                for name, m in models.items():
                    lp = m.heldout_logp([sw_seeds])[0]
                    deltas[name][kind].append((base[name] - lp) / n_occ)
                deltas["Bigram"][kind].append(
                    (base["Bigram"] - bigram.sequence_logp(sw_seq)) / n_occ)

    out = {}
    for name, d in deltas.items():
        dv = float(np.mean(d["valid"])) if d["valid"] else float("nan")
        di = float(np.mean(d["invalid"])) if d["invalid"] else float("nan")
        out[name] = {"delta_valid": dv, "delta_invalid": di, "discrimination": di - dv,
                     "n_valid": len(d["valid"]), "n_invalid": len(d["invalid"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--traces", type=int, default=40)
    ap.add_argument("--K-true", type=int, default=4)
    ap.add_argument("--V", type=int, default=12)
    ap.add_argument("--K-max", type=int, default=6)
    ap.add_argument("--D-max", type=int, default=8)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--swaps-per-trace", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "data" / "experiments" / "order_invariance.json"))
    args = ap.parse_args()

    jobs = [(s, args.traces, args.K_true, args.V, args.K_max, args.D_max, args.iters,
             args.swaps_per_trace) for s in range(args.seeds)]
    if args.workers > 1 and args.seeds > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(args.workers, args.seeds)) as pool:
            rows = pool.map(run_seed, jobs)
    else:
        rows = [run_seed(j) for j in jobs]

    print("\nCost of reordering held-out traces (delta NLL per occurrence, mean +/- 95% CI)")
    print(f"{'model':<10}{'valid swap':>22}{'invalid swap':>22}{'discrimination':>18}")
    print("-" * 72)
    summary = {}
    for name in ["Bigram", "HSMM", "HPOP"]:
        def stat(key):
            v = [r[name][key] for r in rows]
            m = float(np.mean(v))
            ci = 1.96 * float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            return m, ci
        dv, dvc = stat("delta_valid")
        di, dic = stat("delta_invalid")
        dd, ddc = stat("discrimination")
        summary[name] = {"delta_valid": [dv, dvc], "delta_invalid": [di, dic],
                         "discrimination": [dd, ddc]}
        print(f"{name:<10}{dv:>15.4f}±{dvc:.4f}{di:>15.4f}±{dic:.4f}{dd:>11.4f}±{ddc:.4f}")

    # paired per-seed comparison against the sequential baseline (more powerful than the marginals)
    paired = {}
    for name in ["HSMM", "HPOP"]:
        for key in ["delta_valid", "discrimination"]:
            d = [r[name][key] - r["Bigram"][key] for r in rows]
            m = float(np.mean(d))
            ci = 1.96 * float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
            paired[f"{name}-Bigram {key}"] = [m, ci]
    print("\nPaired per-seed difference vs Bigram (negative valid-swap cost = more order-invariant)")
    for k, (m, ci) in paired.items():
        print(f"  {k:<32}{m:>9.4f} ± {ci:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "per_seed": rows, "summary": summary,
                               "paired_vs_bigram": paired}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
