"""Burn-in verification for the sequential validation — registered criterion first.

    PYTHONPATH=src python scripts/collapsed_u_burnin_verification.py

Question (registered in the criterion block below BEFORE the diagnostic runs): on this
mixed-reference problem, do log target, all five scalars and the total relation count
leave the initialization transient well before 50,000 sweeps, starting from the
registered dispersed starts?

Criterion (frozen): run TWO throwaway diagnostic chains (starts 0 and 3 — the two most
separated dispersed starts; seeds 8158901/8158902, never used elsewhere, never pooled
into any validation) for 50,000 sweeps with burn-in 0, thin 10. For every monitored
series, block means over 2,500-sweep blocks (250 retained draws) must lie inside the
7B1 pooled 2.5-97.5% band for EVERY block from sweep 25,000 to 50,000, in both chains.
All pass -> burn-in 50,000 is registered. Any failure -> burn-in 100,000 is registered.
The decision is made by this criterion alone, before any formal chain exists.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.collapsed_u_kernel import (                        # noqa: E402
    CollapsedUConfig, run_collapsed_u_chain,
)
from hpop.mcmc_original.stage6d_frozen import REGISTERED_SCALES            # noqa: E402
from hpop.mcmc_original.stage6e_frozen import (                            # noqa: E402
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH,
)
from hpop.mcmc_original.stage6e_state import Stage6EModel                  # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "collapsed_u_efficient_final_validation"
B7B1 = ROOT / "results" / "mcmc_original" / "stage7b1_mixed_reference"
DIAG_SEEDS = {0: 8_158_901, 3: 8_158_902}
SWEEPS, BLOCK = 50_000, 2_500
SCALARS = ("rho", "beta", "omega", "lambda_rep", "lambda_back")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    e1b = _load("stage6e1b", ROOT / "scripts" / "stage6e1b_mixed_reference.py")
    traces, _ = e1b.generate_corpus()
    mixed = e1b.build_mixed_model(traces)
    model = Stage6EModel(traces=traces, epsilon=e1b.EPSILON, delta_b=DELTA_B,
                         n_skills=e1b.K_SKILLS, n_roles=e1b.M_ROLES,
                         min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
                         infer_pi_P=False)

    ref = np.load(B7B1 / "chains.npz", allow_pickle=False)
    bands = {}
    for name in SCALARS:
        pooled = ref[f"scalar_{name}"].ravel()
        bands[name] = (float(np.quantile(pooled, 0.025)),
                       float(np.quantile(pooled, 0.975)))
    bands["log_target"] = (float(np.quantile(ref["log_target"].ravel(), 0.025)),
                           float(np.quantile(ref["log_target"].ravel(), 0.975)))
    rel_raw = ref["relation_counts"]
    rel = (rel_raw.sum(axis=2) if rel_raw.ndim == 3 else rel_raw).ravel().astype(float)
    bands["relation_total"] = (float(np.quantile(rel, 0.025)),
                               float(np.quantile(rel, 0.975)))

    results = {"criterion": "block means (2,500-sweep blocks) inside the 7B1 pooled "
                            "2.5-97.5% band for EVERY block in sweeps [25k, 50k], "
                            "both chains, all monitored series",
               "bands": bands, "chains": {}}
    all_pass = True
    for chain, seed in DIAG_SEEDS.items():
        start = e1b.dispersed_starts(mixed)[chain]
        run = run_collapsed_u_chain(model=model, start=start,
                                    scales=REGISTERED_SCALES, num_sweeps=SWEEPS,
                                    burn_in=0, thin=10, seed=seed,
                                    collapsed=CollapsedUConfig(every=10),
                                    store_labels=False, store_keys=False)
        n_blocks = SWEEPS // BLOCK
        per_block = BLOCK // 10
        series = {**{n: run.scalars[n] for n in SCALARS},
                  "log_target": run.log_target,
                  "relation_total": run.relation_counts.sum(axis=1).astype(float)}
        chain_report = {}
        for name, values in series.items():
            blocks = values[:n_blocks * per_block].reshape(n_blocks, per_block).mean(1)
            lo, hi = bands[name]
            settled = [bool(lo <= b <= hi) for b in blocks]
            tail_ok = all(settled[SWEEPS // (2 * BLOCK):])   # blocks from 25k on
            first_inside = next((i for i, s in enumerate(settled) if s), None)
            chain_report[name] = {
                "block_means": [float(b) for b in blocks],
                "first_block_inside": first_inside,
                "all_blocks_inside_from_25k": tail_ok}
            all_pass &= tail_ok
        results["chains"][str(chain)] = {"seed": seed, "series": chain_report}

    results["pass"] = bool(all_pass)
    results["registered_burn_in"] = 50_000 if all_pass else 100_000
    results["note"] = "diagnostic chains are throwaway (seeds 8158901/8158902) and are "
    results["note"] += "never pooled into any validation"
    (OUT / "burnin_verification.json").write_text(json.dumps(results, indent=2))
    print(f"[burnin] pass={all_pass} -> registered burn-in "
          f"{results['registered_burn_in']:,}")


if __name__ == "__main__":
    main()
