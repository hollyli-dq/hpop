"""Stage 6D2 — the efficiency-only proposal-scale pilot for the full oracle-block corpus.

    PYTHONPATH=src python scripts/stage6d2_pilot.py

## Why a *separate* pilot, and why the Stage 6D1 scales must not be carried over

Stage 6D1 established that all four registered Stage 6B scalar scales were 16-32x too
small **on the three-block reference model**, and the corrected scales
(`beta 1.63488`, `omega 5.82816`, `lambda_rep 1.3296`, `lambda_back 1.51552`) lifted bulk
ESS by 22-61x there. Those numbers are a property of that deliberately tiny corpus, not of
the kernel: the Stage 6D1 model has 3 blocks of T = 5, so its posterior is broad and wants
large steps. Stage 6D2 runs on the frozen 500-block corpus of T = 20, where the posterior
is *tighter than either* — tighter than 6D1 by construction, and the registered Stage 6B
scales were themselves tuned on this corpus but with `U` held at the truth and the other
scalars fixed. Neither set of scales can be assumed correct here, in either direction, so
this pilot re-derives all of them from scratch and the registered multiplier grid is
symmetric about 1 rather than one-sided.

## Scope: every coordinate the sampler moves

Stage 6D1 tuned only the scalars because `U` and `rho` were explicitly frozen by
instruction. No such instruction covers Stage 6D2 and the corpus is ~167x larger, which
changes the `U` acceptance regime specifically — a `U` row move is scored against 500
blocks of likelihood rather than 3. All six coordinates are therefore tuned, **in
production sweep order**, each frozen at its selection before the next is tuned:

    U -> rho -> beta -> omega -> lambda_rep -> lambda_back

## ESJD is measured in the coordinate each kernel actually walks in

    U             identity   symmetric row random walk on U; squared Frobenius movement
    rho           logit      the frozen Stage 6C kernel walks on z = logit(rho)
    beta          log        PROPOSAL_KIND registers a log random walk
    omega         identity   PROPOSAL_KIND registers an identity random walk
    lambda_rep    log
    lambda_back   log

Measuring a log-scale kernel in raw parameter space rewards large absolute moves at large
parameter values and systematically selects scales that are too big; this cost was already
paid once in Stage 6D1 and is not paid again.

## Permitted information

Acceptance, ESJD, finite-target checks, invalid-proposal counts, recurrent replay counts
and rejection/cache consistency. This script never loads the reference, never reads the
generating truth or `U_TRUE`, never computes a recovery statistic, a held-out likelihood
or an R-hat. The evaluator is constructed at `omega = 0.0` so that not even the true omega
reaches the pilot through a cache seed. **Every pilot draw is discarded**, including the
rejected ones: nothing but the six selected scales survives this script.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hpop.mcmc_original.recurrent_latent_poset_mcmc import (      # noqa: E402
    LatentPosetEvaluator,
)
from hpop.mcmc_original.recurrent_oracle_joint_mcmc import (      # noqa: E402
    Stage6DTarget, initial_state, sweep_once,
)
from hpop.mcmc_original.recurrent_scalar_mcmc import PROPOSAL_KIND    # noqa: E402
from hpop.mcmc_original.stage6d_frozen import (                   # noqa: E402
    ACTIVE_6D, REGISTERED_SCALES, SWEEP_ORDER_6D, config_hash, load_stage6d_dataset,
    rho_to_unconstrained,
)

# The formal runner owns the registered chain starts; importing them rather than
# re-typing them is what guarantees the pilot explores the same starting configurations
# the formal run will use. `chain_start` reads prior quantiles and four hand-written U
# structures only — it touches no truth and no reference.
from stage6d_oracle_joint_mcmc import chain_start                 # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "stage6d2_pilot"

# ------------------------------------------------- registered BEFORE the pilot is run
COORDINATES = tuple(SWEEP_ORDER_6D)                 # U, rho, beta, omega, l_rep, l_back
MULTIPLIERS = (Fraction(1, 32), Fraction(1, 16), Fraction(1, 8), Fraction(1, 4),
               Fraction(1, 2), Fraction(1), Fraction(2), Fraction(4), Fraction(8),
               Fraction(16))
ADMISSIBLE = (0.25, 0.60)
PREFERRED = (0.40, 0.50)
JOINT_CONFIRMATION_BAND = (0.20, 0.65)

# ESJD space per coordinate. The four scalars defer to PROPOSAL_KIND, which is the same
# registry the kernel itself reads, so these cannot drift apart.
ESJD_SPACE = {"U": "identity", "rho": "logit",
              **{name: PROPOSAL_KIND[name]
                 for name in ("beta", "omega", "lambda_rep", "lambda_back")}}

PILOT_CHAINS = 4
PILOT_SWEEPS = 4_000
PILOT_WARM_UP = 1_000
PILOT_BASE_SEED = 90_200_001
SEED_STRIDE = {"U": 0, "rho": 1_000_000, "beta": 2_000_000, "omega": 3_000_000,
               "lambda_rep": 4_000_000, "lambda_back": 5_000_000}
CONFIRMATION_SEED_STRIDE = 9_000_000

# If the registered grid contains no admissible multiplier for some coordinate, the grid
# is extended ONCE, geometrically, in the direction the observed acceptances indicate
# (too-high acceptance -> larger scales; too-low -> smaller). Registered here so that an
# extension is a pre-declared contingency rather than a reaction to a disappointing table.
EXTENSION_FACTOR = 8
EXTENSION_ALLOWED_ONCE = True


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:                                            # pragma: no cover
        return "unknown"


# ------------------------------------------------------------------------ one pilot run
_CORPUS = None


def _corpus():
    """The frozen 500-block training corpus and epsilon. No truth, no U_TRUE, no heldout."""
    global _CORPUS
    if _CORPUS is None:
        frozen = load_stage6d_dataset()
        _CORPUS = (frozen.train, float(frozen.epsilon), int(frozen.u_true.shape[0]))
    return _CORPUS


def expected_replays(m: int) -> int:
    """`1 + sweeps * (m + 1)`: m U rows and one omega per sweep, plus the initial state.

    The Stage 6D1 pilot compared against `sweeps * (m + 1)` and therefore recorded
    `replay_per_sweep_ok = False` on every row. That was an off-by-one in a *reported*
    health flag, not in the sampler and not in the selection rule — `select` never reads
    it — but it is corrected here rather than reproduced, and the correction is recorded
    in the Stage 6D report.
    """
    return 1 + PILOT_SWEEPS * (m + 1)


def _movement(coordinate: str, before, after) -> float:
    """Squared movement of one sweep, in the coordinate the kernel walks in."""
    space = ESJD_SPACE[coordinate]
    if coordinate == "U":
        return float(np.sum((np.asarray(after) - np.asarray(before)) ** 2))
    if space == "logit":
        return float((rho_to_unconstrained(after) - rho_to_unconstrained(before)) ** 2)
    if space == "log":
        return float((math.log(after) - math.log(before)) ** 2)
    return float((after - before) ** 2)


def run_pilot_chain(payload: dict) -> dict:
    """One discarded pilot chain. Returns acceptance / ESJD / health only."""
    coordinate, chain, scales, seed = (payload["coordinate"], payload["chain"],
                                       payload["scales"], payload["seed"])
    roles, epsilon, m = _corpus()
    # omega = 0.0: a neutral cache seed, so the generating omega never enters the pilot.
    evaluator = LatentPosetEvaluator(roles, epsilon=epsilon, omega=0.0)
    target = Stage6DTarget(evaluator, active=ACTIVE_6D)
    u_start, values, _ = chain_start("6d2", chain, m)
    rng = np.random.default_rng(seed)
    state = initial_state(target, u_start, values, rng)

    def snapshot(s):
        return np.array(s.u, copy=True) if coordinate == "U" else s.values[coordinate]

    previous = snapshot(state)
    jumps, non_finite = [], 0
    accepted_before, proposed_before = dict(state.accepted), dict(state.proposed)

    began = time.perf_counter()
    for sweep in range(PILOT_SWEEPS):
        state = sweep_once(state, target, scales, rng)
        if not math.isfinite(state.log_target):
            non_finite += 1
        current = snapshot(state)
        if sweep == PILOT_WARM_UP - 1:
            # discard the transient entirely: counters and ESJD both restart here
            accepted_before, proposed_before = dict(state.accepted), dict(state.proposed)
            jumps, previous = [], current
            continue
        if sweep >= PILOT_WARM_UP:
            jumps.append(_movement(coordinate, previous, current))
            previous = current

    proposed = state.proposed[coordinate] - proposed_before[coordinate]
    accepted = state.accepted[coordinate] - accepted_before[coordinate]
    return {
        "coordinate": coordinate, "chain": chain, "seed": seed,
        "multiplier": payload["multiplier"],
        "acceptance": accepted / proposed if proposed else float("nan"),
        "esjd": float(np.mean(jumps)) if jumps else float("nan"),
        "esjd_space": ESJD_SPACE[coordinate],
        "non_finite_targets": non_finite,
        "invalid_rho": state.invalid.get("rho", 0),
        "full_replay_calls": evaluator.full_replay_calls,
        "expected_full_replay_calls": expected_replays(m),
        "cached_calls": evaluator.cached_calls,
        "replay_per_sweep_ok": evaluator.full_replay_calls == expected_replays(m),
        "runtime_seconds": time.perf_counter() - began,
    }


# ------------------------------------------------------------------------- selection
def summarise(coordinate: str, rows: list[dict], multipliers) -> list[dict]:
    table = []
    for multiplier in multipliers:
        group = [r for r in rows if r["multiplier"] == float(multiplier)]
        if not group:
            continue
        acceptance = float(np.median([r["acceptance"] for r in group]))
        esjd = float(np.median([r["esjd"] for r in group]))
        table.append({
            "multiplier": float(multiplier),
            "multiplier_label": str(multiplier),
            "scale": REGISTERED_SCALES[coordinate] * float(multiplier),
            "median_acceptance": acceptance, "median_esjd": esjd,
            "admissible": ADMISSIBLE[0] <= acceptance <= ADMISSIBLE[1],
            "in_preferred_region": PREFERRED[0] <= acceptance <= PREFERRED[1],
            "non_finite_targets": sum(r["non_finite_targets"] for r in group),
            "replay_invariant_held": all(r["replay_per_sweep_ok"] for r in group),
        })
    return table


def select(coordinate: str, table: list[dict]) -> dict:
    """Largest median ESJD among admissible; ties to the smaller multiplier."""
    admissible = [t for t in table if t["admissible"]]
    if not admissible:
        return {"candidate_table": table, "selected": None,
                "admissible_candidates": [], "tie_broken": False}
    best = max(t["median_esjd"] for t in admissible)
    winners = [t for t in admissible if t["median_esjd"] >= best - 1e-12]
    chosen = min(winners, key=lambda t: t["multiplier"])
    return {"candidate_table": table, "selected": chosen,
            "admissible_candidates": [t["multiplier_label"] for t in admissible],
            "tie_broken": len(winners) > 1}


def extension_direction(table: list[dict]) -> str:
    """Which way an empty admissible set points. Acceptance falls as the scale grows."""
    if min(t["median_acceptance"] for t in table) > ADMISSIBLE[1]:
        return "larger"          # every scale still accepts too often
    if max(t["median_acceptance"] for t in table) < ADMISSIBLE[0]:
        return "smaller"         # every scale already rejects too often
    return "none"                # acceptance straddles the band without landing in it


# ---------------------------------------------------------------------- confirmation
def confirmation_chain(payload: dict) -> dict:
    """§D: confirm, do not optimise. Every coordinate must move, accept and reject."""
    chain, scales, seed = payload["chain"], payload["scales"], payload["seed"]
    roles, epsilon, m = _corpus()
    evaluator = LatentPosetEvaluator(roles, epsilon=epsilon, omega=0.0)
    target = Stage6DTarget(evaluator, active=ACTIVE_6D)
    u_start, values, _ = chain_start("6d2", chain, m)
    rng = np.random.default_rng(seed)
    state = initial_state(target, u_start, values, rng)
    before_a, before_p = dict(state.accepted), dict(state.proposed)
    for _ in range(PILOT_SWEEPS):
        state = sweep_once(state, target, scales, rng)
    row = {"chain": chain, "seed": seed}
    for name in COORDINATES:
        p = state.proposed[name] - before_p[name]
        a = state.accepted[name] - before_a[name]
        row[name] = {"acceptance": a / p if p else float("nan"),
                     "moved": bool(a > 0), "rejected_some": bool(a < p)}
    row["full_replay_calls"] = int(evaluator.full_replay_calls)
    row["expected_full_replay_calls"] = expected_replays(m)
    row["replay_invariant_held"] = bool(
        evaluator.full_replay_calls == expected_replays(m))
    row["all_targets_finite"] = bool(math.isfinite(state.log_target))
    row["invalid_rho"] = int(state.invalid.get("rho", 0))
    return row


def joint_confirmation(scales: dict, jobs: int) -> dict:
    payloads = [{"chain": c, "scales": scales,
                 "seed": PILOT_BASE_SEED + CONFIRMATION_SEED_STRIDE + c}
                for c in range(PILOT_CHAINS)]
    with Pool(min(jobs, len(payloads))) as pool:
        results = pool.map(confirmation_chain, payloads)
    results.sort(key=lambda r: r["chain"])
    acceptance = {n: float(np.median([r[n]["acceptance"] for r in results]))
                  for n in COORDINATES}
    outside = {n: v for n, v in acceptance.items()
               if not (JOINT_CONFIRMATION_BAND[0] <= v <= JOINT_CONFIRMATION_BAND[1])}
    return {
        "per_chain": results, "median_acceptance": acceptance,
        "band": list(JOINT_CONFIRMATION_BAND), "outside_band": outside,
        "all_move": all(r[n]["moved"] for r in results for n in COORDINATES),
        "all_reject_some": all(r[n]["rejected_some"] for r in results
                               for n in COORDINATES),
        "replay_invariant_held": all(r["replay_invariant_held"] for r in results),
        "all_targets_finite": all(r["all_targets_finite"] for r in results),
        "pass": not outside,
        "note": "confirmation only, not another optimisation round",
    }


# ------------------------------------------------------------------------------- main
def registration() -> dict:
    return {
        "registered_before_the_pilot_ran": True,
        "purpose": "choose proposal scales for the Stage 6D2 formal run on the frozen "
                   "500-block corpus, using efficiency information only",
        "why_stage6d1_scales_are_not_reused":
            "the Stage 6D1 corrections (16-32x) were tuned on a 3-block T=5 reference "
            "model whose posterior is deliberately broad; the Stage 6D2 posterior is "
            "much tighter, so those multipliers are expected to be far too large. The "
            "registered Stage 6B scales are equally unverified here because they were "
            "tuned with U fixed at the truth and the other scalars held fixed.",
        "coordinates_tuned_in_sweep_order": list(COORDINATES),
        "coordinates_frozen_untouched": [],
        "why_U_and_rho_are_in_scope":
            "Stage 6D1 froze U and rho by explicit instruction. No such instruction "
            "covers Stage 6D2, and the corpus is ~167x larger, which changes the U "
            "acceptance regime directly: a U row move is scored against 500 blocks of "
            "likelihood rather than 3.",
        "multipliers": [str(m) for m in MULTIPLIERS],
        "multiplier_grid_is_symmetric_about_1": True,
        "base_scales": dict(REGISTERED_SCALES),
        "admissible_acceptance": list(ADMISSIBLE),
        "preferred_acceptance": list(PREFERRED),
        "selection_rule": "largest median ESJD among admissible; ties to the smaller "
                          "multiplier",
        "esjd_space": dict(ESJD_SPACE),
        "esjd_note": "U uses squared Frobenius movement of the whole matrix over a "
                     "sweep; rho uses squared movement of logit(rho); the log-scale "
                     "scalars use squared movement of log(x). Raw-space ESJD on a "
                     "log-scale kernel biases selection toward oversized scales.",
        "sequential_rule": "each coordinate is frozen at its selection before the next "
                           "is tuned; later coordinates keep their base scales until "
                           "their turn",
        "grid_extension_rule": {
            "allowed_once": EXTENSION_ALLOWED_ONCE, "factor": EXTENSION_FACTOR,
            "direction": "toward larger scales if every median acceptance exceeds the "
                         "admissible band, toward smaller scales if every median "
                         "acceptance falls below it",
            "registered_before_running": True},
        "chains": PILOT_CHAINS, "sweeps": PILOT_SWEEPS,
        "discarded_warm_up": PILOT_WARM_UP, "base_seed": PILOT_BASE_SEED,
        "seed_rule": "PILOT_BASE_SEED + SEED_STRIDE[coordinate] + 1000 * grid_index "
                     "+ chain",
        "joint_confirmation_band": list(JOINT_CONFIRMATION_BAND),
        "starts": "the registered Stage 6D2 formal starts, imported from "
                  "scripts/stage6d_oracle_joint_mcmc.chain_start",
        "evaluator_omega_seed": 0.0,
        "permitted_statistics": ["acceptance", "ESJD", "finite-target checks",
                                 "invalid-proposal counts", "recurrent replay counts",
                                 "rejection/cache consistency"],
        "forbidden_and_not_computed": [
            "the generating truth and U_TRUE", "held-out likelihood",
            "any reference or reference summary", "recovery metrics",
            "induced-H total variation", "relation-marginal errors",
            "R-hat or ESS from candidate pilot runs"],
        "all_pilot_draws_discarded": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    reg = registration()
    (OUT / "pilot_registration.json").write_text(json.dumps(reg, indent=2))

    roles, epsilon, m = _corpus()
    print(f"[6d2 pilot] corpus {roles.shape[0]} blocks x T={roles.shape[1]}, m={m}, "
          f"{len(COORDINATES)} coordinates x {len(MULTIPLIERS)} multipliers x "
          f"{PILOT_CHAINS} chains", flush=True)

    scales = dict(REGISTERED_SCALES)
    began = time.perf_counter()
    decisions, all_rows, extensions = {}, [], {}

    for coordinate in COORDINATES:
        print(f"\n[6d2 pilot] {coordinate}: base scale "
              f"{REGISTERED_SCALES[coordinate]:.6g}, ESJD in "
              f"{ESJD_SPACE[coordinate]} space", flush=True)
        grid = list(MULTIPLIERS)
        rows: list[dict] = []
        decision = None
        for attempt in range(2):
            payloads = []
            for index, multiplier in enumerate(grid):
                if any(r["multiplier"] == float(multiplier) for r in rows):
                    continue
                trial = dict(scales)
                trial[coordinate] = REGISTERED_SCALES[coordinate] * float(multiplier)
                for chain in range(PILOT_CHAINS):
                    payloads.append({
                        "coordinate": coordinate, "chain": chain, "scales": trial,
                        "multiplier": float(multiplier),
                        "seed": (PILOT_BASE_SEED + SEED_STRIDE[coordinate]
                                 + 1000 * index + chain)})
            if payloads:
                with Pool(min(args.jobs, len(payloads))) as pool:
                    rows.extend(pool.map(run_pilot_chain, payloads))
            table = summarise(coordinate, rows, grid)
            for t in table:
                print(f"[6d2 pilot]   x{t['multiplier_label']:<5} scale "
                      f"{t['scale']:.6g}  median acceptance "
                      f"{t['median_acceptance']:.3f}  median ESJD "
                      f"{t['median_esjd']:.6g}"
                      f"{'  <- admissible' if t['admissible'] else ''}", flush=True)
            decision = select(coordinate, table)
            if decision["selected"] is not None:
                break
            direction = extension_direction(table)
            if attempt == 1 or not EXTENSION_ALLOWED_ONCE or direction == "none":
                break
            factor = (Fraction(EXTENSION_FACTOR) if direction == "larger"
                      else Fraction(1, EXTENSION_FACTOR))
            grid = sorted(set(grid) | {mult * factor for mult in grid})
            extensions[coordinate] = {
                "direction": direction, "factor": EXTENSION_FACTOR,
                "extended_grid": [str(g) for g in grid],
                "reason": "no multiplier in the registered grid was admissible; the "
                          "pre-registered single extension was applied"}
            print(f"[6d2 pilot]   no admissible multiplier; applying the registered "
                  f"single extension toward {direction} scales", flush=True)

        if decision["selected"] is None:
            (OUT / "pilot_results.json").write_text(json.dumps({
                "registration": reg, "decisions": decisions, "per_chain_rows": all_rows,
                "failed_coordinate": coordinate,
                "failed_candidate_table": decision["candidate_table"],
                "extensions": extensions,
                "outcome": "no admissible multiplier after the registered extension; "
                           "reporting the failure rather than relaxing the band"},
                indent=2))
            raise SystemExit(
                f"{coordinate}: no multiplier gives acceptance inside {ADMISSIBLE}")

        decisions[coordinate] = decision
        all_rows.extend(rows)
        scales[coordinate] = decision["selected"]["scale"]     # freeze before the next
        print(f"[6d2 pilot]   -> selected x{decision['selected']['multiplier_label']}  "
              f"scale {decision['selected']['scale']:.6g}  acceptance "
              f"{decision['selected']['median_acceptance']:.3f}", flush=True)

    print("\n[6d2 pilot] joint confirmation with all six tuned coordinates ...",
          flush=True)
    confirmation = joint_confirmation(scales, args.jobs)
    for name in COORDINATES:
        print(f"[6d2 pilot]   {name:<12} median acceptance "
              f"{confirmation['median_acceptance'][name]:.3f}", flush=True)

    payload = {
        "registration": reg,
        "decisions": {c: decisions[c] for c in COORDINATES},
        "extensions": extensions,
        "per_chain_rows": all_rows,
        "selected_scales": scales,
        "selected_multipliers": {c: decisions[c]["selected"]["multiplier_label"]
                                 for c in COORDINATES},
        "registered_base_scales": dict(REGISTERED_SCALES),
        "stage6d1_scales_for_contrast": {
            "U": 0.5, "rho": 0.5, "beta": 1.63488, "omega": 5.82816,
            "lambda_rep": 1.3296, "lambda_back": 1.51552,
            "note": "recorded for the report only; never used as a starting point here"},
        "joint_confirmation": confirmation,
        "runtime_seconds": time.perf_counter() - began,
        "source_commit": source_commit(), "stage6d_config_hash": config_hash(),
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(),
        "all_pilot_draws_discarded": True,
    }
    (OUT / "pilot_results.json").write_text(json.dumps(payload, indent=2))
    (OUT / "selected_scales.json").write_text(json.dumps(scales, indent=2))

    if not confirmation["pass"]:
        print(f"\n[6d2 pilot] JOINT CONFIRMATION FAILED: "
              f"{confirmation['outside_band']}", flush=True)
        raise SystemExit("a coordinate's acceptance fell outside the joint confirmation "
                         "band; reporting the interaction rather than selecting another "
                         "scale")
    print(f"\n[6d2 pilot] confirmed. frozen scales: "
          f"{ {k: float(f'{v:.6g}') for k, v in scales.items()} }", flush=True)
    print(f"[6d2 pilot] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
