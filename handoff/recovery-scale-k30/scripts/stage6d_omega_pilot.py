"""Stage 6D1 — the permitted, fully discarded omega proposal-scale pilot.

The formal Stage 6D1 attempt failed its registered omega convergence gate
(R-hat 1.01205 > 1.01) at the 100,000-sweep ceiling, with scalar acceptance at
0.96-0.98 — the signature of a random-walk scale far too small for this target. Section 11
permits a pilot to set proposal scales provided every pilot draw is discarded, the formal
chains restart, and the scales are frozen before the formal run.

## What this pilot is allowed to look at, and what it is not

Permitted, and all that is computed here:

    acceptance rate
    expected squared jumping distance (ESJD)
    finite-target checks
    rejection and cache diagnostics

Forbidden, and never loaded by this script — it does not even import the reference
module or the truth:

    reference means, reference KS distances, the generating truth,
    posterior recovery, the MCMC/reference energy distance, or the resulting formal R-hat

**omega only.** No other coordinate is retuned. beta, lambda_rep, lambda_back, rho and U
keep their registered scales, because no registered diagnostic has shown their efficiency
to be unacceptable.

Every pilot draw is discarded. Nothing produced here is used as a posterior sample.
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
from hpop.mcmc_original.stage6d_frozen import ACTIVE_6D, REGISTERED_SCALES, config_hash
from hpop.mcmc_original.stage6d_joint_reference import small_model

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "mcmc_original" / "stage6d1_omega_pilot"

# ------------------------------------------------- registered BEFORE the pilot is run
MULTIPLIERS = (2, 4, 8, 16, 32, 64)
ADMISSIBLE = (0.25, 0.60)
PREFERRED = (0.40, 0.50)
SELECTION_RULE = ("Among multipliers whose median across-chain acceptance lies inside "
                  "the admissible interval [0.25, 0.60], select the one with the largest "
                  "median expected squared jumping distance across the pilot chains. "
                  "Ties (within 1e-12) are broken towards the SMALLER multiplier. The "
                  "preferred acceptance region [0.40, 0.50] is reported for context and "
                  "is not itself a filter.")
PILOT_CHAINS = 4
PILOT_SWEEPS = 4_000
PILOT_WARM_UP = 1_000          # discarded before any statistic is accumulated
PILOT_BASE_SEED = 90_000_001


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:                                            # pragma: no cover
        return "unknown"


def pilot_starts(chain: int) -> tuple[np.ndarray, dict]:
    """The registered dispersed Stage 6D1 starts, reused so the pilot explores the same
    region the formal chains will. No truth and no reference is consulted."""
    starts = {
        0: np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]),      # antichain
        1: np.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]]),      # total order
        2: np.array([[1.0, 1.0], [0.0, 2.0], [0.5, 0.4]]),      # sparse
        3: np.array([[2.0, 2.5], [1.0, 1.2], [0.0, 0.1]]),      # dense
    }
    rho = (0.05, 0.30, 0.60, 0.90)[chain]
    scalars = ({"beta": 0.35, "omega": -2.56, "lambda_rep": 1.35, "lambda_back": 1.95},
               {"beta": 0.66, "omega": 1.35, "lambda_rep": 1.95, "lambda_back": 0.35},
               {"beta": 1.35, "omega": 1.95, "lambda_rep": 0.35, "lambda_back": 0.66},
               {"beta": 1.95, "omega": 0.35, "lambda_rep": 0.66, "lambda_back": 1.35})[chain]
    return starts[chain], {"rho": rho, **scalars}


def run_pilot_chain(multiplier: float, chain: int, model) -> dict:
    """One discarded pilot chain. Returns only acceptance/ESJD/health diagnostics."""
    evaluator = LatentPosetEvaluator(model.roles, epsilon=model.epsilon,
                                     omega=0.0)
    target = Stage6DTarget(evaluator, active=ACTIVE_6D)
    scales = dict(REGISTERED_SCALES)
    scales["omega"] = REGISTERED_SCALES["omega"] * multiplier

    u_start, values = pilot_starts(chain)
    rng = np.random.default_rng(PILOT_BASE_SEED + 1000 * int(multiplier) + chain)
    state = initial_state(target, u_start, values, rng)

    previous_omega = state.values["omega"]
    squared_jumps = []
    non_finite = 0
    accepted_before = dict(state.accepted)
    proposed_before = dict(state.proposed)

    for sweep in range(PILOT_SWEEPS):
        state = sweep_once(state, target, scales, rng)
        if not math.isfinite(state.log_target):
            non_finite += 1
        if sweep == PILOT_WARM_UP - 1:
            # discard the warm-up entirely: reset the counters the statistics use
            accepted_before = dict(state.accepted)
            proposed_before = dict(state.proposed)
            squared_jumps = []
            previous_omega = state.values["omega"]
            continue
        if sweep >= PILOT_WARM_UP:
            squared_jumps.append((state.values["omega"] - previous_omega) ** 2)
            previous_omega = state.values["omega"]

    proposed = state.proposed["omega"] - proposed_before["omega"]
    accepted = state.accepted["omega"] - accepted_before["omega"]
    return {
        "multiplier": float(multiplier), "chain": chain,
        "omega_acceptance": accepted / proposed if proposed else float("nan"),
        "omega_esjd": float(np.mean(squared_jumps)) if squared_jumps else float("nan"),
        "non_finite_targets": non_finite,
        "full_replay_calls": evaluator.full_replay_calls,
        "cached_calls": evaluator.cached_calls,
        "invalid_rho": state.invalid.get("rho", 0),
        "u_acceptance": ((state.accepted["U"] - accepted_before["U"])
                         / max(state.proposed["U"] - proposed_before["U"], 1)),
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    model = small_model()
    OUT.mkdir(parents=True, exist_ok=True)

    registration = {
        "registered_before_the_pilot_ran": True,
        "coordinate_tuned": "omega only",
        "coordinates_left_at_registered_scales": ["U", "rho", "beta", "lambda_rep",
                                                  "lambda_back"],
        "why_omega_only": "The failed formal attempt showed omega acceptance at 0.969 "
                          "with bulk ESS 650. No registered diagnostic has shown the "
                          "other coordinates' efficiency to be unacceptable, so they are "
                          "not retuned.",
        "multipliers": list(MULTIPLIERS),
        "base_omega_scale": REGISTERED_SCALES["omega"],
        "admissible_acceptance": list(ADMISSIBLE),
        "preferred_acceptance": list(PREFERRED),
        "selection_rule": SELECTION_RULE,
        "tie_rule": "choose the smaller multiplier",
        "chains": PILOT_CHAINS, "sweeps": PILOT_SWEEPS,
        "discarded_warm_up": PILOT_WARM_UP,
        "base_seed": PILOT_BASE_SEED,
        "permitted_statistics": ["acceptance rate", "expected squared jumping distance",
                                 "finite-target checks",
                                 "rejection and cache diagnostics"],
        "forbidden_and_not_computed": [
            "reference means", "reference KS distances", "generating truth",
            "posterior recovery", "MCMC/reference energy distance", "formal R-hat"],
        "all_pilot_draws_discarded": True,
    }
    (OUT / "pilot_registration.json").write_text(json.dumps(registration, indent=2))

    print(f"[pilot] omega only, multipliers {list(MULTIPLIERS)}, "
          f"{PILOT_CHAINS} chains x {PILOT_SWEEPS} sweeps "
          f"({PILOT_WARM_UP} discarded)", flush=True)
    began = time.perf_counter()
    rows = []
    for multiplier in MULTIPLIERS:
        for chain in range(PILOT_CHAINS):
            rows.append(run_pilot_chain(multiplier, chain, model))
        acceptance = [r["omega_acceptance"] for r in rows if r["multiplier"] == multiplier]
        esjd = [r["omega_esjd"] for r in rows if r["multiplier"] == multiplier]
        print(f"[pilot]   x{multiplier:<3g} scale {REGISTERED_SCALES['omega'] * multiplier:.4f}"
              f"  median acceptance {np.median(acceptance):.3f}"
              f"  median ESJD {np.median(esjd):.4f}", flush=True)

    summary = []
    for multiplier in MULTIPLIERS:
        group = [r for r in rows if r["multiplier"] == multiplier]
        acceptance = float(np.median([r["omega_acceptance"] for r in group]))
        esjd = float(np.median([r["omega_esjd"] for r in group]))
        summary.append({
            "multiplier": float(multiplier),
            "omega_scale": REGISTERED_SCALES["omega"] * multiplier,
            "median_acceptance": acceptance, "median_esjd": esjd,
            "admissible": ADMISSIBLE[0] <= acceptance <= ADMISSIBLE[1],
            "in_preferred_region": PREFERRED[0] <= acceptance <= PREFERRED[1],
            "non_finite_targets": sum(r["non_finite_targets"] for r in group),
        })

    admissible = [s for s in summary if s["admissible"]]
    if not admissible:
        raise SystemExit("no multiplier landed inside the admissible acceptance interval")
    best = max(s["median_esjd"] for s in admissible)
    winners = [s for s in admissible if s["median_esjd"] >= best - 1e-12]
    selected = min(winners, key=lambda s: s["multiplier"])       # tie -> smaller

    decision = {
        "selected_multiplier": selected["multiplier"],
        "selected_omega_scale": selected["omega_scale"],
        "selected_median_acceptance": selected["median_acceptance"],
        "selected_median_esjd": selected["median_esjd"],
        "in_preferred_region": selected["in_preferred_region"],
        "admissible_candidates": [s["multiplier"] for s in admissible],
        "rule_applied": SELECTION_RULE,
        "tie_broken": len(winners) > 1,
        "all_pilot_draws_discarded": True,
        "no_reference_or_truth_consulted": True,
    }
    (OUT / "pilot_results.json").write_text(json.dumps({
        "registration": registration, "per_multiplier": summary,
        "per_chain": rows, "decision": decision,
        "runtime_seconds": time.perf_counter() - began,
        "source_commit": source_commit(), "stage6d_config_hash": config_hash(),
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform()}, indent=2))

    print(f"\n[pilot] selected multiplier x{selected['multiplier']:g}  "
          f"omega scale {selected['omega_scale']:.4f}  "
          f"acceptance {selected['median_acceptance']:.3f}  "
          f"ESJD {selected['median_esjd']:.4f}", flush=True)
    print("[pilot] every pilot draw discarded; formal chains restart from the "
          "registered dispersed starts with this scale frozen", flush=True)


if __name__ == "__main__":
    main()
