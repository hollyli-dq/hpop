"""Stage 6D1 — the authorised efficiency-only pilot for beta, lambda_rep, lambda_back.

Authorised **only after** the following efficiency diagnostics were observed in the
corrected formal run, which remained FAIL on beta R-hat 1.01334:

    beta         acceptance 0.974   bulk ESS  349
    lambda_rep   acceptance 0.972   bulk ESS  528
    lambda_back  acceptance 0.958   bulk ESS 1143

They exhibit the same severely under-scaled random-walk pathology already confirmed and
corrected for omega, where cutting acceptance 0.969 -> 0.373 raised bulk ESS 650 -> 45,433.

## Scope

Tuned here: beta, lambda_rep, lambda_back, sequentially in production sweep order, each
one frozen before the next is tuned.

Untouched, by instruction: the U proposal, the rho proposal, the omega scale (frozen at
5.8282, multiplier 32), the target, the priors, the reference, every convergence gate and
the 100,000-sweep ceiling.

## ESJD is measured in the kernel's own coordinate

`PROPOSAL_KIND` registers beta, lambda_rep and lambda_back as **log**-scale random walks
and omega as identity. Squared movement is therefore accumulated in log space for these
three — `(log x' - log x)^2` — because that is the space the proposal actually walks in.
Measuring in raw parameter space would reward large absolute moves at large parameter
values and systematically favour too-large scales.

## Permitted information

Acceptance, ESJD, finite-target checks, invalid-proposal counts, recurrent replay checks
and rejection/cache consistency. Nothing else: this script never loads the reference, the
truth, or any recovery or R-hat statistic. Every pilot draw is discarded.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from hpop.mcmc_original.recurrent_latent_poset_mcmc import LatentPosetEvaluator
from hpop.mcmc_original.recurrent_oracle_joint_mcmc import (
    Stage6DTarget, initial_state, sweep_once,
)
from hpop.mcmc_original.recurrent_scalar_mcmc import PROPOSAL_KIND
from hpop.mcmc_original.stage6d_frozen import ACTIVE_6D, REGISTERED_SCALES, config_hash
from hpop.mcmc_original.stage6d_joint_reference import small_model

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "mcmc_original" / "stage6d1_scalar_pilot"

# ------------------------------------------------- registered BEFORE the pilot is run
COORDINATES = ("beta", "lambda_rep", "lambda_back")
MULTIPLIERS = (2, 4, 8, 16, 32, 64)
ADMISSIBLE = (0.25, 0.60)
PREFERRED = (0.40, 0.50)
JOINT_CONFIRMATION_BAND = (0.20, 0.65)
OMEGA_MULTIPLIER = 32                     # frozen from the successful omega pilot

# identical to the successful omega pilot
PILOT_CHAINS = 4
PILOT_SWEEPS = 4_000
PILOT_WARM_UP = 1_000
PILOT_BASE_SEED = 90_000_001
SEED_STRIDE = {"beta": 1_000_000, "lambda_rep": 2_000_000, "lambda_back": 3_000_000}

AUTHORISATION = {
    "authorised_after_observing": {
        "beta": {"acceptance": 0.974, "bulk_ess": 349},
        "lambda_rep": {"acceptance": 0.972, "bulk_ess": 528},
        "lambda_back": {"acceptance": 0.958, "bulk_ess": 1143},
    },
    "context": "The corrected Stage 6D1 formal run remained FAIL on beta R-hat 1.01334 "
               "at the 100,000-sweep ceiling. These acceptance and ESS diagnostics are "
               "the separately recorded efficiency evidence that authorised this pilot.",
    "previous_failed_attempts_preserved_unmodified": [
        "stage6d1_joint_mcmc_FAILED_attempt0_50k",
        "stage6d1_joint_mcmc_FAILED_attempt1",
        "stage6d1_joint_mcmc (omega-retuned attempt, superseded by the final rerun)",
    ],
    "not_relabelled_as_pilots": True,
}


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:                                            # pragma: no cover
        return "unknown"


def pilot_starts(chain: int) -> tuple[np.ndarray, dict]:
    """The same start construction the omega pilot used."""
    starts = {
        0: np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]),
        1: np.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]]),
        2: np.array([[1.0, 1.0], [0.0, 2.0], [0.5, 0.4]]),
        3: np.array([[2.0, 2.5], [1.0, 1.2], [0.0, 0.1]]),
    }
    rho = (0.05, 0.30, 0.60, 0.90)[chain]
    scalars = ({"beta": 0.35, "omega": -2.56, "lambda_rep": 1.35, "lambda_back": 1.95},
               {"beta": 0.66, "omega": 1.35, "lambda_rep": 1.95, "lambda_back": 0.35},
               {"beta": 1.35, "omega": 1.95, "lambda_rep": 0.35, "lambda_back": 0.66},
               {"beta": 1.95, "omega": 0.35, "lambda_rep": 0.66,
                "lambda_back": 1.35})[chain]
    return starts[chain], {"rho": rho, **scalars}


def run_pilot_chain(coordinate: str, scales: dict, chain: int, model, seed: int) -> dict:
    """One discarded pilot chain. Returns acceptance/ESJD/health only."""
    evaluator = LatentPosetEvaluator(model.roles, epsilon=model.epsilon, omega=0.0)
    target = Stage6DTarget(evaluator, active=ACTIVE_6D)
    u_start, values = pilot_starts(chain)
    rng = np.random.default_rng(seed)
    state = initial_state(target, u_start, values, rng)

    log_scale = PROPOSAL_KIND[coordinate] == "log"
    previous = state.values[coordinate]
    jumps, non_finite = [], 0
    accepted_before, proposed_before = dict(state.accepted), dict(state.proposed)

    for sweep in range(PILOT_SWEEPS):
        state = sweep_once(state, target, scales, rng)
        if not math.isfinite(state.log_target):
            non_finite += 1
        current = state.values[coordinate]
        if sweep == PILOT_WARM_UP - 1:
            accepted_before, proposed_before = dict(state.accepted), dict(state.proposed)
            jumps, previous = [], current
            continue
        if sweep >= PILOT_WARM_UP:
            # ESJD in the coordinate the kernel walks in
            if log_scale:
                jumps.append((math.log(current) - math.log(previous)) ** 2)
            else:
                jumps.append((current - previous) ** 2)
            previous = current

    proposed = state.proposed[coordinate] - proposed_before[coordinate]
    accepted = state.accepted[coordinate] - accepted_before[coordinate]
    return {
        "coordinate": coordinate, "chain": chain, "seed": seed,
        "acceptance": accepted / proposed if proposed else float("nan"),
        "esjd": float(np.mean(jumps)) if jumps else float("nan"),
        "esjd_space": "log" if log_scale else "identity",
        "non_finite_targets": non_finite,
        "invalid_rho": state.invalid.get("rho", 0),
        "full_replay_calls": evaluator.full_replay_calls,
        "cached_calls": evaluator.cached_calls,
        "replay_per_sweep_ok": evaluator.full_replay_calls == PILOT_SWEEPS * (
            model.m + 1),
    }


def select(coordinate: str, rows: list[dict]) -> dict:
    table = []
    for multiplier in MULTIPLIERS:
        group = [r for r in rows if r["multiplier"] == multiplier]
        acceptance = float(np.median([r["acceptance"] for r in group]))
        esjd = float(np.median([r["esjd"] for r in group]))
        table.append({
            "multiplier": float(multiplier),
            "scale": REGISTERED_SCALES[coordinate] * multiplier,
            "median_acceptance": acceptance, "median_esjd": esjd,
            "admissible": ADMISSIBLE[0] <= acceptance <= ADMISSIBLE[1],
            "in_preferred_region": PREFERRED[0] <= acceptance <= PREFERRED[1],
            "non_finite_targets": sum(r["non_finite_targets"] for r in group),
            "replay_invariant_held": all(r["replay_per_sweep_ok"] for r in group),
        })
    admissible = [t for t in table if t["admissible"]]
    if not admissible:
        raise SystemExit(f"{coordinate}: no multiplier inside {ADMISSIBLE}")
    best = max(t["median_esjd"] for t in admissible)
    winners = [t for t in admissible if t["median_esjd"] >= best - 1e-12]
    chosen = min(winners, key=lambda t: t["multiplier"])
    return {"candidate_table": table, "selected": chosen,
            "admissible_candidates": [t["multiplier"] for t in admissible],
            "tie_broken": len(winners) > 1}


def joint_confirmation(scales: dict, model) -> dict:
    """§D: confirm, do not optimise. All four scalars must move, accept and reject."""
    results, acceptance = [], {}
    for chain in range(PILOT_CHAINS):
        evaluator = LatentPosetEvaluator(model.roles, epsilon=model.epsilon, omega=0.0)
        target = Stage6DTarget(evaluator, active=ACTIVE_6D)
        u_start, values = pilot_starts(chain)
        rng = np.random.default_rng(PILOT_BASE_SEED + 7_000_000 + chain)
        state = initial_state(target, u_start, values, rng)
        before_a, before_p = dict(state.accepted), dict(state.proposed)
        for _ in range(PILOT_SWEEPS):
            state = sweep_once(state, target, scales, rng)
        row = {"chain": chain}
        for name in ("beta", "omega", "lambda_rep", "lambda_back"):
            p = state.proposed[name] - before_p[name]
            a = state.accepted[name] - before_a[name]
            row[name] = {"acceptance": a / p if p else float("nan"),
                         "moved": a > 0, "rejected_some": a < p}
        row["replay_invariant_held"] = evaluator.full_replay_calls == PILOT_SWEEPS * (
            model.m + 1)
        row["all_targets_finite"] = math.isfinite(state.log_target)
        results.append(row)
    for name in ("beta", "omega", "lambda_rep", "lambda_back"):
        acceptance[name] = float(np.median([r[name]["acceptance"] for r in results]))
    outside = {n: v for n, v in acceptance.items()
               if not (JOINT_CONFIRMATION_BAND[0] <= v <= JOINT_CONFIRMATION_BAND[1])}
    return {
        "per_chain": results, "median_acceptance": acceptance,
        "band": list(JOINT_CONFIRMATION_BAND),
        "outside_band": outside,
        "all_move": all(r[n]["moved"] for r in results
                        for n in ("beta", "omega", "lambda_rep", "lambda_back")),
        "all_reject_some": all(r[n]["rejected_some"] for r in results
                               for n in ("beta", "omega", "lambda_rep", "lambda_back")),
        "replay_invariant_held": all(r["replay_invariant_held"] for r in results),
        "all_targets_finite": all(r["all_targets_finite"] for r in results),
        "pass": not outside,
        "note": "confirmation only, not another optimisation round",
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    model = small_model()
    OUT.mkdir(parents=True, exist_ok=True)

    registration = {
        "registered_before_the_pilot_ran": True,
        "authorisation": AUTHORISATION,
        "coordinates_tuned_in_sweep_order": list(COORDINATES),
        "coordinates_frozen_untouched": {
            "U": REGISTERED_SCALES["U"], "rho": REGISTERED_SCALES["rho"],
            "omega": REGISTERED_SCALES["omega"] * OMEGA_MULTIPLIER,
            "omega_multiplier": OMEGA_MULTIPLIER},
        "multipliers": list(MULTIPLIERS),
        "admissible_acceptance": list(ADMISSIBLE),
        "preferred_acceptance": list(PREFERRED),
        "selection_rule": "largest median ESJD among admissible; ties to the smaller "
                          "multiplier",
        "esjd_space": {c: PROPOSAL_KIND[c] for c in COORDINATES},
        "esjd_note": "log-scale kernels use squared movement in log parameter space",
        "sequential_rule": "each coordinate is frozen at its selection before the next "
                           "is tuned; later coordinates keep their old scales until "
                           "their turn",
        "chains": PILOT_CHAINS, "sweeps": PILOT_SWEEPS,
        "discarded_warm_up": PILOT_WARM_UP, "base_seed": PILOT_BASE_SEED,
        "joint_confirmation_band": list(JOINT_CONFIRMATION_BAND),
        "permitted_statistics": ["acceptance", "ESJD", "finite-target checks",
                                 "invalid-proposal counts", "recurrent replay checks",
                                 "rejection/cache consistency"],
        "forbidden_and_not_computed": [
            "reference means", "reference quantiles", "KS distances", "energy distance",
            "induced-H TV", "relation errors", "generating truth", "recovery metrics",
            "R-hat from candidate pilot runs"],
        "all_pilot_draws_discarded": True,
    }
    (OUT / "pilot_registration.json").write_text(json.dumps(registration, indent=2))

    scales = dict(REGISTERED_SCALES)
    scales["omega"] = REGISTERED_SCALES["omega"] * OMEGA_MULTIPLIER
    began = time.perf_counter()
    decisions, all_rows = {}, []

    for coordinate in COORDINATES:
        print(f"[pilot] {coordinate}: multipliers {list(MULTIPLIERS)} "
              f"(ESJD in {PROPOSAL_KIND[coordinate]} space)", flush=True)
        rows = []
        for multiplier in MULTIPLIERS:
            trial = dict(scales)
            trial[coordinate] = REGISTERED_SCALES[coordinate] * multiplier
            for chain in range(PILOT_CHAINS):
                seed = (PILOT_BASE_SEED + SEED_STRIDE[coordinate]
                        + 1000 * int(multiplier) + chain)
                row = run_pilot_chain(coordinate, trial, chain, model, seed)
                row["multiplier"] = float(multiplier)
                rows.append(row)
            group = [r for r in rows if r["multiplier"] == multiplier]
            print(f"[pilot]   x{multiplier:<3g} scale "
                  f"{REGISTERED_SCALES[coordinate] * multiplier:.5f}"
                  f"  median acceptance "
                  f"{np.median([r['acceptance'] for r in group]):.3f}"
                  f"  median ESJD {np.median([r['esjd'] for r in group]):.5f}",
                  flush=True)
        decision = select(coordinate, rows)
        decisions[coordinate] = decision
        all_rows.extend(rows)
        scales[coordinate] = decision["selected"]["scale"]      # freeze before the next
        print(f"[pilot]   -> selected x{decision['selected']['multiplier']:g}  "
              f"scale {decision['selected']['scale']:.5f}  "
              f"acceptance {decision['selected']['median_acceptance']:.3f}\n", flush=True)

    print("[pilot] joint confirmation with all four tuned scalars ...", flush=True)
    confirmation = joint_confirmation(scales, model)
    for name, value in confirmation["median_acceptance"].items():
        print(f"[pilot]   {name:<12} median acceptance {value:.3f}", flush=True)

    payload = {
        "registration": registration,
        "decisions": {c: decisions[c] for c in COORDINATES},
        "per_chain_rows": all_rows,
        "selected_scales": scales,
        "joint_confirmation": confirmation,
        "runtime_seconds": time.perf_counter() - began,
        "source_commit": source_commit(), "stage6d_config_hash": config_hash(),
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(),
        "all_pilot_draws_discarded": True,
    }
    (OUT / "pilot_results.json").write_text(json.dumps(payload, indent=2))

    if not confirmation["pass"]:
        print(f"\n[pilot] JOINT CONFIRMATION FAILED: {confirmation['outside_band']}",
              flush=True)
        raise SystemExit("a scalar acceptance fell outside the joint confirmation band; "
                         "reporting the interaction rather than selecting another scale")
    print(f"\n[pilot] confirmed. frozen scales: "
          f"{ {k: round(v, 5) for k, v in scales.items()} }", flush=True)


if __name__ == "__main__":
    main()
