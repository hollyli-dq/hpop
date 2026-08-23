"""Collapsed-U kernel validation — the partially-collapsed sampler against the frozen
Stage 6E1B mixed reference.

    PYTHONPATH=src python scripts/collapsed_u_mixed_reference.py [--sweeps N]

Step 7B1 already established that the FFBS joint sampler targets the Stage 6E1B
posterior: all 18 frozen gates PASS. This run asks the SAME question of the
partially-collapsed sampler — Step 7B plus one occasional collapsed-U structural move
every `COLLAPSED_EVERY` sweeps, composed as

    collapsed U MH  ->  exact FFBS refresh of all (S, z)  ->  the Stage 6E phases

— and answers it the same way: the frozen reference, the frozen gates, Stage 6E1B's own
`compare` function called verbatim, the registered 600,000-sweep length, new seeds. No
new reference is built, no gate or threshold is changed.

Everything identical to Step 7B1's comparison except the kernel under test:

    corpus, priors, pi and P (FIXED, as the reference fixes them), dispersed starts,
    proposal scales, sweeps / burn-in / thin, every gate, the comparison code itself.

Writes into results/mcmc_original/collapsed_u_kernel_validation/ (the C0/C1 audit
directories and all Stage 6E2 / Step 7B2 results are untouched).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.collapsed_u_kernel import (                        # noqa: E402
    MOVE_NAME, CollapsedUConfig, run_collapsed_u_chain,
)
from hpop.mcmc_original.latent_poset import precedence_from_u              # noqa: E402
from hpop.mcmc_original.stage6d_frozen import REGISTERED_SCALES, SCALAR_ORDER  # noqa: E402
from hpop.mcmc_original.stage6e_frozen import (                            # noqa: E402
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH, assert_stage6d_unchanged, config_hash,
)
from hpop.mcmc_original.stage6e_state import Stage6EModel                  # noqa: E402

# Seeds and the output directory are env-overridable so an independent-seed
# REPLICATION of the identical registered comparison can run without touching the
# first run's artifacts. Defaults are the first run's registered values. Spawn workers
# re-import this module and re-read the same environment, so parent and children agree.
OUT = (ROOT / "results" / "mcmc_original"
       / os.environ.get("COLLAPSED_U_OUT", "collapsed_u_kernel_validation"))
FROZEN_6E1B = ROOT / "results" / "mcmc_original" / "stage6e1b_mixed_reference"
BASELINE_7B1 = ROOT / "results" / "mcmc_original" / "stage7b1_mixed_reference"

# Registered before the run. New seeds; everything else is Stage 6E1B / Step 7B1's.
CHAIN_SEEDS = tuple(int(s) for s in os.environ.get(
    "COLLAPSED_U_SEEDS", "8153001,8153002,8153003,8153004").split(","))
TABLE_SOURCE = "batched"
COLLAPSED_EVERY = 10        # provisional default from the C1 audit; NOT tuned here
COLLAPSED_SCALE = 0.5       # the registered production row scale, unchanged


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:                                                # pragma: no cover
        return "unknown"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if math.isnan(v) else v
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def load_6e1b():
    path = ROOT / "scripts" / "stage6e1b_mixed_reference.py"
    spec = importlib.util.spec_from_file_location("stage6e1b", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_7b1():
    path = ROOT / "scripts" / "stage7b1_mixed_reference_mcmc.py"
    spec = importlib.util.spec_from_file_location("stage7b1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _chain_worker(payload: dict) -> dict:
    """One partially-collapsed chain in its own process."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "src"))
    module = load_6e1b()
    chain = payload["chain"]
    traces, _ = module.generate_corpus()
    mixed = module.build_mixed_model(traces)
    start = module.dispersed_starts(mixed)[chain]
    model = Stage6EModel(traces=traces, epsilon=module.EPSILON, delta_b=DELTA_B,
                         n_skills=module.K_SKILLS, n_roles=module.M_ROLES,
                         min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
                         infer_pi_P=False)
    began = time.perf_counter()
    result = run_collapsed_u_chain(
        model=model, start=start, scales=REGISTERED_SCALES,
        num_sweeps=payload["num_sweeps"], burn_in=payload["burn_in"],
        thin=payload["thin"], seed=CHAIN_SEEDS[chain],
        collapsed=CollapsedUConfig(every=payload["collapsed_every"],
                                   scale=COLLAPSED_SCALE),
        chain=chain, table_source=TABLE_SOURCE, store_keys=True, store_labels=False,
        progress_every=payload.get("progress_every", 0))

    records = result.collapsed_records
    n_attempted = len(records)
    n_valid = sum(1 for r in records if not r["invalid"])
    n_cross = sum(1 for r in records if r["h_changed"])
    n_accepted = sum(1 for r in records if r["accepted"])
    n_accepted_cross = sum(1 for r in records if r["accepted"] and r["h_changed"])
    return {"chain": chain, "seed": CHAIN_SEEDS[chain], "u_draws": result.u_draws,
            "scalars": result.scalars, "log_target": result.log_target,
            "segment_counts": result.segment_counts,
            "relation_counts": result.relation_counts,
            "boundary_keys": result.boundary_keys, "movement": result.movement,
            "acceptance_rates": result.acceptance(),
            "table_builds": int(result.table_builds),
            "table_build_seconds": float(result.table_build_seconds),
            "runtime_seconds": time.perf_counter() - began,
            "collapsed": {
                "every": payload["collapsed_every"], "attempted": n_attempted,
                "valid": n_valid, "h_changed": n_cross, "accepted": n_accepted,
                "accepted_h_changed": n_accepted_cross,
                "fraction_h_changed": n_cross / max(1, n_valid),
                "fraction_accepted": n_accepted / max(1, n_valid),
                "mean_eval_seconds": float(np.mean([r["seconds"] for r in records])
                                           if records else float("nan")),
                "p95_eval_seconds": float(np.quantile(
                    [r["seconds"] for r in records], 0.95) if records
                    else float("nan"))},
            "likelihood_stats": result.collapsed_likelihood_stats}


def movement_diagnostics(payloads, baseline_dir: Path) -> dict:
    """Structural-movement summary for the collapsed chains, next to the 7B1 kernel's."""
    out = {"per_chain": [], "baseline_7b1": {}}
    for p in payloads:
        relations = p["relation_counts"].sum(axis=1).astype(float)
        centred = relations - relations.mean()
        denominator = float((centred ** 2).sum())
        lag1 = (float((centred[:-1] * centred[1:]).sum()) / denominator
                if denominator > 0 else float("nan"))
        # distinct induced-H states over the retained draws (exact per-skill tuple)
        seen = set()
        for draw in p["u_draws"]:
            seen.add(tuple(precedence_from_u(np.asarray(draw[k], dtype=float))
                           .tobytes() for k in range(draw.shape[0])))
        # accepted H changes from one retained draw to the next (any kernel's doing)
        changes = 0
        previous = None
        for draw in p["u_draws"]:
            key = tuple(precedence_from_u(np.asarray(draw[k], dtype=float)).tobytes()
                        for k in range(draw.shape[0]))
            if previous is not None and key != previous:
                changes += 1
            previous = key
        ess = (len(relations) * (1.0 - lag1) / (1.0 + lag1)
               if math.isfinite(lag1) and lag1 < 1.0 else float("nan"))
        out["per_chain"].append({
            "chain": p["chain"], **p["collapsed"],
            "distinct_h_states_retained": len(seen),
            "h_changes_between_retained_draws": changes,
            "relation_count_lag1_autocorrelation": lag1,
            "relation_count_ess_lag1_estimate": ess})
    baseline = json.loads((baseline_dir / "joint_comparison.json").read_text())
    out["baseline_7b1"] = {"movement": baseline.get("movement"),
                           "worst_rhat": baseline.get("worst_rhat"),
                           "note": "descriptive only; correctness is judged by the "
                                   "gates, not by movement"}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps", type=int, default=None,
                        help="override ONLY for a smoke run; the registered Stage 6E1B "
                             "length is the default")
    parser.add_argument("--progress-every", type=int, default=50_000)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    assert_stage6d_unchanged()
    module = load_6e1b()
    b7b1 = load_7b1()

    reference_audit = b7b1.verify_frozen_reference()
    if not reference_audit["pass"]:
        raise SystemExit(f"the frozen Stage 6E1B reference did not verify: "
                         f"{reference_audit['drift_by_statistic']}")
    print(f"[coll-u val] frozen reference verified "
          f"(drift {reference_audit['max_drift_from_expected']:.1e})")

    n_sweeps = args.sweeps if args.sweeps is not None else module.N_SWEEPS
    burn_in = module.BURN_IN if args.sweeps is None else max(1, n_sweeps // 5)
    thin = module.THIN
    registered_length = n_sweeps == module.N_SWEEPS and burn_in == module.BURN_IN

    config = {
        "task": "collapsed-U kernel validation against the frozen Stage 6E1B mixed "
                "reference",
        "source_commit": source_commit(), "stage6e_config_hash": config_hash(),
        "python": platform.python_version(), "numpy": np.__version__,
        "kernel_under_test": {
            "composition": "collapsed U MH -> exact FFBS refresh of all (S, z) -> "
                           "Stage 6E parameter phase (unchanged, by call)",
            "collapsed_u_every": COLLAPSED_EVERY,
            "collapsed_u_every_note": "provisional default from the C1 audit; "
                                      "deliberately NOT tuned in this task",
            "proposal": "sampler_u.propose_row, one uniformly chosen (skill, row), "
                        f"scale {COLLAPSED_SCALE} — the registered production proposal",
            "hastings": "zero (symmetric row proposal at fixed rho; test-verified)",
            "acceptance": "Delta ell_coll + Delta log p(U | rho)",
            "ordinary_u_moves": "UNCHANGED — the Stage 6C conditional row sweep still "
                                "runs inside sweep_once on every sweep"},
        "what_is_identical_to_7b1": [
            "corpus and trace generation", "pi and P (FIXED, as the reference fixes "
            "them)", "priors", "proposal scales", "dispersed starting states",
            "sweeps / burn-in / thin", "every gate and threshold",
            "the comparison function itself (Stage 6E1B's `compare`, verbatim)"],
        "chains": {"n_chains": module.N_CHAINS, "sweeps": n_sweeps, "burn_in": burn_in,
                   "thin": thin, "seeds": list(CHAIN_SEEDS),
                   "scales": dict(REGISTERED_SCALES), "table_source": TABLE_SOURCE,
                   "registered_length_used": registered_length},
        "frozen_reference": reference_audit,
        "go_criteria_preregistered": [
            "tiny exact correctness PASS (test suite + validation artifacts)",
            "all Stage 6E1B / Step 7B1 gates PASS, thresholds unchanged",
            "worst R-hat <= 1.01 on the registered summaries",
            "resume determinism PASS", "all project tests PASS"],
    }
    (OUT / "config.json").write_text(json.dumps(jsonable(config), indent=2))

    if not registered_length:
        print(f"[coll-u val] WARNING: smoke length {n_sweeps:,}/{burn_in:,}/{thin} — "
              "gates from this run are NOT a validation result", flush=True)

    from multiprocessing import get_context
    began = time.perf_counter()
    jobs = [{"chain": c, "num_sweeps": n_sweeps, "burn_in": burn_in, "thin": thin,
             "collapsed_every": COLLAPSED_EVERY,
             "progress_every": args.progress_every}
            for c in range(module.N_CHAINS)]
    print(f"[coll-u val] {module.N_CHAINS} partially-collapsed chains x "
          f"{n_sweeps:,} sweeps (collapsed move every {COLLAPSED_EVERY}), in parallel",
          flush=True)
    with get_context("spawn").Pool(processes=module.N_CHAINS) as pool:
        payloads = pool.map(_chain_worker, jobs)
    wall = time.perf_counter() - began
    for p in payloads:
        c = p["collapsed"]
        print(f"[coll-u val] chain {p['chain']}: {len(p['log_target']):,} retained, "
              f"{c['attempted']:,} collapsed attempts, "
              f"{c['fraction_h_changed']:.3f} crossed H, "
              f"{c['fraction_accepted']:.3f} accepted, "
              f"{p['runtime_seconds']:.0f}s", flush=True)

    results = [module._ChainResult(p) for p in payloads]

    mixed = module.build_mixed_model(module.generate_corpus()[0])
    for name in ("reference_draws.npz", "reference_registration.json",
                 "qmc_summary.json"):
        (OUT / name).write_bytes((FROZEN_6E1B / name).read_bytes())
    module.OUT = OUT
    try:
        module.compare(mixed, results)
    except SystemExit as failure:
        print(f"[coll-u val] gate failure reported by the comparison: {failure}",
              flush=True)

    gates = json.loads((OUT / "joint_comparison.json").read_text())
    (OUT / "mixed_reference_comparison.json").write_text(json.dumps(jsonable({
        "all_pass": gates["all_pass"], "worst_rhat": gates["worst_rhat"],
        "gates": gates["gates"],
        "baseline_7b1": {name: json.loads((BASELINE_7B1
                                           / "joint_comparison.json").read_text())
                         .get(name) for name in ("all_pass", "worst_rhat")},
        "registered_length_used": registered_length}), indent=2))

    movement = movement_diagnostics(payloads, BASELINE_7B1)
    (OUT / "movement.json").write_text(json.dumps(jsonable(movement), indent=2))

    performance = {
        "wall_seconds": wall,
        "per_chain_seconds": [p["runtime_seconds"] for p in payloads],
        "seconds_per_sweep": [p["runtime_seconds"] / n_sweeps for p in payloads],
        "collapsed_event_seconds": [p["collapsed"]["mean_eval_seconds"]
                                    for p in payloads],
        "baseline_7b1_seconds_per_sweep": json.loads(
            (BASELINE_7B1 / "performance.json").read_text())["seconds_per_sweep"],
        "likelihood_stats": [p["likelihood_stats"] for p in payloads],
    }
    (OUT / "reference_performance.json").write_text(json.dumps(jsonable(performance),
                                                               indent=2))

    verdict = bool(gates["all_pass"]) and registered_length
    print(f"[coll-u val] gates all_pass={gates['all_pass']} "
          f"worst_rhat={gates['worst_rhat']} registered_length={registered_length}")
    print(f"[coll-u val] wrote {OUT}")
    if not gates["all_pass"]:
        raise SystemExit("collapsed-U mixed-reference validation FAILED: "
                         f"{[k for k, g in gates['gates'].items() if not g['pass']]}")
    del verdict


if __name__ == "__main__":
    main()
