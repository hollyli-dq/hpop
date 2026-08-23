"""Step 7B1 — the FFBS joint sampler against the frozen Stage 6E1B mixed reference.

    PYTHONPATH=src python scripts/stage7b1_mixed_reference_mcmc.py [--sweeps N]

The question is whether replacing the local segmentation Metropolis update with an exact
FFBS Gibbs draw leaves the *full joint* invariant distribution alone. The Stage 6E1B
reference already answers "what is that distribution" independently of any sampler —
scrambled Sobol in prior coordinates, with the segmentation marginalised by exact
enumeration inside each QMC point — and its quality gates are frozen and passed. So this
script builds no new reference, changes no gate, and reuses Stage 6E1B's own comparison
code verbatim.

## What is held identical to the Stage 6E1B production comparison

    corpus, priors, pi and P (both FIXED, exactly as the reference fixes them),
    dispersed starting states, proposal scales, 600,000 sweeps, 120,000 burn-in,
    thin 10, four chains, every gate and every threshold, the comparison function itself.

## What differs, and it is the only thing

    Stage 6E1B:  8 local Metropolis proposals per trace per sweep (Hastings ratio)
    Step 7B1:    1 exact FFBS draw per trace per sweep  (Gibbs, acceptance 1)

and new RNG seeds, because reusing the baseline's seeds would confound the kernel
comparison with a shared random stream. No chain is initialised from a LocalMoveKernel
endpoint or from the reference posterior.

## Run length

The registered Stage 6E1B length is used unchanged. That matters: Stage 6E1B's own first
attempt failed `induced_h_total_variation` at 150,000 sweeps for lack of effective sample
size, and the registered response was more draws. Shortening the run here — even though
FFBS draws `(S, z)` exactly — would risk a Monte Carlo failure that says nothing about the
kernel, and choosing the length after seeing the answer would be worse still.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import (                   # noqa: E402
    run_stage7b_chain, sweep_cost_breakdown,
)
from hpop.mcmc_original.stage6d_frozen import REGISTERED_SCALES, SCALAR_ORDER  # noqa: E402
from hpop.mcmc_original.stage6e_frozen import (                              # noqa: E402
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH, assert_stage6d_unchanged, config_hash,
)
from hpop.mcmc_original.stage6e_state import Stage6EModel                    # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "stage7b1_mixed_reference"
FROZEN_6E1B = ROOT / "results" / "mcmc_original" / "stage6e1b_mixed_reference"
LIVE_6E1B = Path("/Users/dongqing/Desktop/hpop-stage6e/results/mcmc_original"
                 "/stage6e1b_mixed_reference")

# Registered before the run. New seeds; everything else is Stage 6E1B's.
CHAIN_SEEDS = (7_053_001, 7_053_002, 7_053_003, 7_053_004)
TABLE_SOURCE = "batched"

# The reference-quality numbers this script requires of the frozen reference, checked
# against the artifact rather than trusted from the handoff.
EXPECTED_REFERENCE = {
    "max_rqmc_standard_error": 5.667929455901159e-04,
    "max_half_width_95": 1.2080905663045393e-03,
    "min_relative_ess": 4.386141741129308e-02,
    "max_normalised_weight": 3.295848775089024e-04,
    "log_evidence_sd": 3.3823943089510656e-03,
}


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
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def load_6e1b():
    """The Stage 6E1B script defines the problem, the starts and the comparison. Import
    it rather than restate any of it."""
    path = ROOT / "scripts" / "stage6e1b_mixed_reference.py"
    spec = importlib.util.spec_from_file_location("stage6e1b", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_frozen_reference() -> dict:
    """The reference must be the frozen one, and it must have passed its own gates."""
    registration = json.loads((FROZEN_6E1B / "reference_registration.json").read_text())
    observed = {name: check["value"] for name, check in registration["checks"].items()}
    drift = {name: abs(observed[name] - expected)
             for name, expected in EXPECTED_REFERENCE.items()}
    hashes = {path.name: sha256(path) for path in sorted(FROZEN_6E1B.iterdir())
              if path.is_file()}
    live_hashes = ({path.name: sha256(path) for path in sorted(LIVE_6E1B.iterdir())
                    if path.is_file()} if LIVE_6E1B.exists() else {})
    matches_live = {name: (live_hashes.get(name) == digest)
                    for name, digest in hashes.items() if name in live_hashes}
    return {
        "directory": str(FROZEN_6E1B.relative_to(ROOT)),
        "live_directory": str(LIVE_6E1B),
        "sha256": hashes,
        "matches_live_stage6e_worktree": matches_live,
        "all_copies_match_live": bool(all(matches_live.values())) if matches_live else None,
        "checks": registration["checks"],
        "superseded_checks": registration["superseded_checks"],
        "superseded_note": "these two maximum-over-replicate diagnostics were superseded "
                           "in Stage 6D1 and still fail; they are reported as failing "
                           "descriptive diagnostics and are NOT gates here either",
        "nondegeneracy": registration["nondegeneracy"],
        "primary_pass": registration["primary_pass"],
        "all_active_pass": registration["all_active_pass"],
        "label_permutation_audit": registration["label_permutation_audit"]["conclusion"],
        "max_drift_from_expected": max(drift.values()),
        "drift_by_statistic": drift,
        "pass": bool(registration["primary_pass"] and registration["all_active_pass"]
                     and max(drift.values()) < 1e-12),
    }


def _chain_worker(payload: dict) -> dict:
    """One FFBS chain in its own process. Rebuilds the model rather than unpickling it."""
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
    result = run_stage7b_chain(
        model=model, start=start, scales=REGISTERED_SCALES,
        num_sweeps=payload["num_sweeps"], burn_in=payload["burn_in"],
        thin=payload["thin"], seed=CHAIN_SEEDS[chain], chain=chain,
        table_source=TABLE_SOURCE, store_keys=True, store_labels=False,
        progress_every=payload.get("progress_every", 0))
    return {"chain": chain, "seed": CHAIN_SEEDS[chain], "u_draws": result.u_draws,
            "scalars": result.scalars, "log_target": result.log_target,
            "segment_counts": result.segment_counts,
            "relation_counts": result.relation_counts,
            "boundary_keys": result.boundary_keys, "movement": result.movement,
            "acceptance_rates": result.acceptance(),
            "table_builds": int(result.table_builds),
            "table_build_seconds": float(result.table_build_seconds),
            "runtime_seconds": time.perf_counter() - began}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps", type=int, default=None,
                        help="override ONLY for a smoke run; the registered length is "
                             "Stage 6E1B's and is used by default")
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--thin", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25_000)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    assert_stage6d_unchanged()
    module = load_6e1b()

    reference_audit = verify_frozen_reference()
    if not reference_audit["pass"]:
        raise SystemExit(f"the frozen Stage 6E1B reference did not verify: "
                         f"{reference_audit['drift_by_statistic']}")
    print(f"[7B1] frozen reference verified: primary_pass="
          f"{reference_audit['primary_pass']} all_active_pass="
          f"{reference_audit['all_active_pass']} "
          f"drift={reference_audit['max_drift_from_expected']:.1e}")

    n_sweeps = args.sweeps if args.sweeps is not None else module.N_SWEEPS
    burn_in = (args.burn_in if args.burn_in is not None
               else (module.BURN_IN if args.sweeps is None else max(1, n_sweeps // 5)))
    thin = args.thin if args.thin is not None else module.THIN
    registered_length = (n_sweeps == module.N_SWEEPS and burn_in == module.BURN_IN
                         and thin == module.THIN)

    traces, truths = module.generate_corpus()
    mixed = module.build_mixed_model(traces)
    starts = module.dispersed_starts(mixed)
    model = Stage6EModel(traces=traces, epsilon=module.EPSILON, delta_b=DELTA_B,
                         n_skills=module.K_SKILLS, n_roles=module.M_ROLES,
                         min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
                         infer_pi_P=False)

    cost = sweep_cost_breakdown(model, starts[0], REGISTERED_SCALES, n_sweeps=100,
                                table_source=TABLE_SOURCE)
    projected = cost["seconds_per_sweep"] * n_sweeps
    print(f"[7B1] sweep cost {cost['seconds_per_sweep'] * 1e3:.1f} ms "
          f"-> {projected / 3600:.2f} h per chain, {module.N_CHAINS} chains in parallel")

    config = {
        "stage": "7B1", "source_commit": source_commit(),
        "stage6e_config_hash": config_hash(),
        "python": platform.python_version(), "numpy": np.__version__,
        "what_changed": "the (S, z) update only: 8 local Metropolis proposals per trace "
                        "per sweep -> one exact FFBS Gibbs draw per trace per sweep",
        "what_is_identical": [
            "corpus and trace generation", "pi and P (FIXED, as the reference fixes them)",
            "priors", "proposal scales", "dispersed starting states",
            "sweeps / burn-in / thin", "every gate and threshold",
            "the comparison function itself (Stage 6E1B's `compare`, called verbatim)"],
        "segmentation_update": {
            "kind": "Gibbs — exact draw from p(S_n, z_n | x_n, Theta)",
            "acceptance_probability": 1.0, "hastings_correction": "none",
            "draws_per_trace_per_sweep": 1,
            "rng": "sequential from the single chain generator, in trace order"},
        "target": {
            "coordinates": ["S", "z", "U", "rho", *SCALAR_ORDER],
            "fixed": ["pi", "P", "delta_B", "epsilon", "K", "m", "d"],
            "why_pi_P_fixed": "the frozen Stage 6E1B reference fixes them, so Step 7B1 "
                              "must fix them too or it would be comparing a different "
                              "posterior. pi/P inference is exercised in Step 7B2."},
        "chains": {"n_chains": module.N_CHAINS, "sweeps": n_sweeps, "burn_in": burn_in,
                   "thin": thin, "seeds": list(CHAIN_SEEDS),
                   "seeds_note": "new seeds; the Stage 6E1B chains used "
                                 f"{list(module.CHAIN_SEEDS)}",
                   "scales": dict(REGISTERED_SCALES),
                   "starts": "module.dispersed_starts, the Stage 6E1B starts unchanged",
                   "registered_length_used": registered_length,
                   "table_source": TABLE_SOURCE},
        "frozen_reference": reference_audit,
        "baseline_local_move_kernel": {
            "directory": str(FROZEN_6E1B.relative_to(ROOT)),
            "chains_npz_left_in_place": str(LIVE_6E1B / "chains.npz"),
            "chains_npz_sha256": (sha256(LIVE_6E1B / "chains.npz")
                                  if (LIVE_6E1B / "chains.npz").exists() else None)},
        "projected_cost": cost,
    }
    (OUT / "config.json").write_text(json.dumps(jsonable(config), indent=2))

    if not registered_length:
        print(f"[7B1] WARNING: not the registered length "
              f"({n_sweeps:,}/{burn_in:,}/{thin} against "
              f"{module.N_SWEEPS:,}/{module.BURN_IN:,}/{module.THIN}); this is a smoke "
              f"run and its gates are not a Step 7B1 result", flush=True)

    # ---- the chains --------------------------------------------------------------------
    from multiprocessing import get_context
    began = time.perf_counter()
    jobs = [{"chain": c, "num_sweeps": n_sweeps, "burn_in": burn_in, "thin": thin,
             "progress_every": args.progress_every} for c in range(module.N_CHAINS)]
    print(f"[7B1] {module.N_CHAINS} FFBS chains x {n_sweeps:,} sweeps, burn-in "
          f"{burn_in:,}, thin {thin}, in parallel", flush=True)
    with get_context("spawn").Pool(processes=module.N_CHAINS) as pool:
        payloads = pool.map(_chain_worker, jobs)
    wall = time.perf_counter() - began
    for payload in payloads:
        print(f"[7B1] chain {payload['chain']} seed {payload['seed']}: "
              f"{len(payload['log_target']):,} retained in "
              f"{payload['runtime_seconds']:.0f}s", flush=True)
    print(f"[7B1] {module.N_CHAINS} chains in {wall:.0f}s wall", flush=True)

    results = [module._ChainResult(p) for p in payloads]

    # ---- the comparison, run by Stage 6E1B's own code, into this stage's directory -----
    for name in ("reference_draws.npz", "reference_registration.json",
                 "qmc_summary.json"):
        (OUT / name).write_bytes((FROZEN_6E1B / name).read_bytes())
    module.OUT = OUT
    try:
        module.compare(mixed, results)
    except SystemExit as failure:
        # `compare` raises when a gate fails. The artifacts it writes before raising are
        # the result and must be preserved and reported, not lost to an early exit.
        print(f"[7B1] gate failure reported by the Stage 6E1B comparison: {failure}",
              flush=True)

    performance = {
        "wall_seconds": wall,
        "per_chain_seconds": [p["runtime_seconds"] for p in payloads],
        "sweeps": n_sweeps, "retained_per_chain": [len(p["log_target"]) for p in payloads],
        "seconds_per_sweep": [p["runtime_seconds"] / n_sweeps for p in payloads],
        "table_builds_per_chain": [p["table_builds"] for p in payloads],
        "table_build_seconds_per_chain": [p["table_build_seconds"] for p in payloads],
        "table_build_fraction": [p["table_build_seconds"] / p["runtime_seconds"]
                                 for p in payloads],
        "builds_equal_sweeps": bool(all(p["table_builds"] == n_sweeps for p in payloads)),
        "movement_per_chain": [p["movement"] for p in payloads],
        "acceptance_per_chain": [p["acceptance_rates"] for p in payloads],
        "cost_breakdown": cost,
        "baseline_runtime_seconds": json.loads(
            (FROZEN_6E1B / "joint_comparison.json").read_text())["runtime_seconds"],
        "baseline_movement": json.loads(
            (FROZEN_6E1B / "joint_comparison.json").read_text())["movement"],
    }
    (OUT / "performance.json").write_text(json.dumps(jsonable(performance), indent=2))

    gates = json.loads((OUT / "joint_comparison.json").read_text())
    write_report(module, config, gates, performance, reference_audit, n_sweeps)
    print(f"[7B1] all_pass={gates['all_pass']} worst_rhat={gates['worst_rhat']}")
    print(f"[7B1] wrote {OUT}")
    if not gates["all_pass"]:
        raise SystemExit("Step 7B1 FAILED: "
                         f"{[k for k, g in gates['gates'].items() if not g['pass']]}")


def write_report(module, config, gates, performance, reference_audit, n_sweeps) -> None:
    baseline = json.loads((FROZEN_6E1B / "joint_comparison.json").read_text())
    lines = [
        "# Step 7B1 — FFBS inside the full recurrent joint sampler",
        "",
        f"Status: **{'PASS' if gates['all_pass'] else 'FAIL'}**. "
        "Step 7B2 (full-corpus mixing comparison): see its own directory.",
        "",
        "The `(S, z)` update is an exact FFBS Gibbs draw; every other kernel, the target, "
        "the corpus, the starts, the scales, the run length, the gates and the comparison "
        "code are Stage 6E1B's, unchanged. The reference is Stage 6E1B's frozen mixed "
        "QMC + exact-enumeration reference; no new reference was built.",
        "",
        "## The frozen reference, verified from the artifact",
        "",
        "| statistic | value | threshold | verdict |",
        "|---|---|---|---|",
    ]
    for name, check in reference_audit["checks"].items():
        comparison = "&ge;" if check.get("comparison") == ">=" else "&le;"
        lines.append(f"| {name} | {check['value']:.4e} | {comparison} "
                     f"{check['threshold']} | {'PASS' if check['pass'] else 'FAIL'} |")
    for name, check in reference_audit["superseded_checks"].items():
        lines.append(f"| {name} (superseded, descriptive) | {check['value']:.4e} | "
                     f"{check['threshold']} | "
                     f"{'pass' if check['pass'] else 'FAIL — reported as such'} |")
    lines += [
        "",
        f"Label permutation audit: {reference_audit['label_permutation_audit']}",
        "",
        "## Gates — the Stage 6E1B gates, unchanged",
        "",
        "| gate | FFBS | LocalMoveKernel (frozen) | threshold | verdict |",
        "|---|---|---|---|---|",
    ]
    for name, gate in gates["gates"].items():
        value = gate["value"]
        base = baseline["gates"].get(name, {}).get("value")
        shown = "n/a" if value is None else f"{value:.6g}"
        base_shown = "n/a" if base is None else f"{base:.6g}"
        lines.append(f"| {name} | {shown} | {base_shown} | {gate['threshold']} | "
                     f"{'PASS' if gate['pass'] else 'FAIL'} |")
    lines += [
        "",
        f"Worst R-hat: FFBS {gates['worst_rhat']:.6f}, LocalMoveKernel "
        f"{baseline['worst_rhat']:.6f}, threshold 1.01.",
        "",
        "## Same target, different kernel",
        "",
        "| quantity | FFBS | LocalMoveKernel |",
        "|---|---|---|",
        f"| sweeps | {n_sweeps:,} | {module.N_SWEEPS:,} |",
        f"| retained (pooled) | {gates['retained_pooled']:,} | "
        f"{baseline['retained_pooled']:,} |",
        f"| wall seconds (4 chains, parallel) | {performance['wall_seconds']:.0f} | "
        f"{max(baseline['runtime_seconds']):.0f} |",
        f"| seconds per sweep (per chain) | "
        f"{np.mean(performance['seconds_per_sweep']):.4f} | "
        f"{np.mean(baseline['runtime_seconds']) / module.N_SWEEPS:.4f} |",
        f"| segmentation acceptance | 1.000 (Gibbs, by construction) | "
        f"{baseline['acceptance_by_chain'][0]['relabel']:.3f} relabel / "
        f"{baseline['acceptance_by_chain'][0]['split']:.3f} split / "
        f"{baseline['acceptance_by_chain'][0]['shift']:.3f} shift |",
        f"| boundary changes (chain 0) | "
        f"{performance['movement_per_chain'][0]['boundary_hamming']:,} | "
        f"{baseline['movement'][0]['boundary_hamming']:,} |",
        f"| occurrence-label changes (chain 0) | "
        f"{performance['movement_per_chain'][0]['label_changes']:,} | "
        f"{baseline['movement'][0]['label_changes']:,} |",
        "",
        "Neither sampler produces a better posterior: both are compared against the same "
        "independent reference and both must clear the same gates. What differs is the "
        "transition kernel and its cost.",
        "",
        "## Scalar ESS and convergence",
        "",
        "| coordinate | FFBS bulk ESS | LocalMoveKernel bulk ESS | FFBS R-hat | "
        "LocalMoveKernel R-hat |",
        "|---|---|---|---|---|",
    ]
    for name, block in gates["convergence"].items():
        base_block = baseline["convergence"].get(name, {})
        if not isinstance(block, dict) or block.get("degenerate"):
            continue
        lines.append(
            f"| {name} | {block.get('bulk_ess', float('nan')):,.0f} | "
            f"{base_block.get('bulk_ess', float('nan')):,.0f} | "
            f"{block.get('rhat', float('nan')):.5f} | "
            f"{base_block.get('rhat', float('nan')):.5f} |")
    lines += [
        "",
        "## Cost",
        "",
        f"* FFBS sweep: {performance['cost_breakdown']['seconds_per_sweep'] * 1e3:.1f} ms "
        f"— candidate tables "
        f"{performance['cost_breakdown']['block_table_seconds_per_sweep'] * 1e3:.1f} ms, "
        f"charts {performance['cost_breakdown']['chart_seconds_per_sweep'] * 1e3:.1f} ms, "
        f"backward draws "
        f"{performance['cost_breakdown']['backward_draw_seconds_per_sweep'] * 1e3:.2f} ms, "
        f"parameter phase the remainder",
        f"* candidate tables built once per sweep: "
        f"{performance['builds_equal_sweeps']}; "
        f"{np.mean(performance['table_build_fraction']) * 100:.0f}% of chain wall time",
        f"* LocalMoveKernel baseline: "
        f"{np.mean(baseline['runtime_seconds']) / module.N_SWEEPS * 1e3:.1f} ms per sweep",
        "",
        f"Source commit `{config['source_commit']}`.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
