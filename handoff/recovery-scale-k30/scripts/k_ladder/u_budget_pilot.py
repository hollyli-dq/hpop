#!/usr/bin/env python3
"""Pilot the `U` effort budget and `u_scale` using TRUTH-FREE diagnostics only.

The proportional-effort rule is registered: attempted `U` updates per role vector are held
constant across `K` at `target_u_attempts_per_role`. What that constant should *be* is not
registered, and 166.7 is inherited from whatever `K = 3` happened to get under the old
fixed cadence -- a number with no claim on adequacy. This pilot compares candidate budgets
and tunes `u_scale`.

**No diagnostic here may touch the truth.** Recovery against the sealed `U` is exactly the
quantity the study exists to measure; selecting a budget or a scale on it would tune the
experiment on its own answer. So the diagnostics below are all functions of the chains
alone:

    U acceptance rate                       -- burn-in and retained, reported separately
    closure-changing acceptance             -- accepted moves that alter the induced
                                               precedence closure; a move that shifts `U`
                                               without changing the order does no
                                               structural work
    relation changes per accepted proposal  -- how far each accepted move travels
    relation-count R-hat                    -- between-chain agreement on |relations|
    selected-edge R-hat                     -- the same on individual edge indicators
    relation-level ESS                      -- per-edge effective sample size
    unique closures visited                 -- raw exploration
    effective order-changing moves / second -- the throughput that actually matters

`u_scale` is tuned during warm-up and **frozen** afterwards; warm-up and retained
proposals are never pooled, because they were made under different kernels.

## Superseded by the full factorial

An earlier version of this plan ran a staged `K = 30` grid to save compute. That shortcut
is **withdrawn**: the compute constraint was relaxed, and the registered pilot is now the
complete factorial over every rung, budget and scale, generated as a manifest before any
result is seen. See `pilot_manifest.py`, `run_pilot_job.py` and `aggregate_pilot.py`.

This script remains useful for interactive exploration of a few cells. It is **not** the
registered pilot and its output must not be merged into the pilot summary.

    python scripts/k_ladder/u_budget_pilot.py --rungs 3 10 --targets 50 100 166.7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_cpa.corpus import generate_ladder_corpus                 # noqa: E402
from hpop.mcmc_cpa.crn import CommonRandomNumbers                       # noqa: E402
from hpop.mcmc_cpa.ladder_runner import LEARNED_ORDER, run_ladder_chain  # noqa: E402
from hpop.mcmc_cpa.nested_library import draw_master_library            # noqa: E402
from hpop.mcmc_cpa.u_quota import quota_schedule, update_events          # noqa: E402
from hpop.mcmc_original.latent_poset import precedence_from_u           # noqa: E402
from hpop.mcmc_original.stage6b_mcmc_diagnostics import (               # noqa: E402
    bulk_ess, rank_normalized_split_rhat)
from hpop.mcmc_original.stage6e_state import Stage6EModel               # noqa: E402


def closure_bits(u_by_skill) -> np.ndarray:
    """Flattened off-diagonal precedence bits for every skill: the induced closure."""
    u = np.asarray(u_by_skill, dtype=float)
    m = u.shape[1]
    off = ~np.eye(m, dtype=bool)
    return np.concatenate([np.asarray(precedence_from_u(u[k]))[off]
                           for k in range(u.shape[0])])


def chain_diagnostics(u_draws) -> dict:
    """Everything derivable from one chain's retained `U` draws. Truth never enters.

    `u_draws` is indexed by retained `U` state, i.e. by `U`-update event, not by FFBS
    sweep. Mixing of the order is a property of the `U` kernel, and measuring it on a
    sweep axis would dilute it with sweeps during which `U` cannot have moved at all.
    """
    bits = np.array([closure_bits(u) for u in u_draws], dtype=bool)
    if bits.shape[0] < 2:
        return {"draws": int(bits.shape[0]), "insufficient": True}
    changes = (bits[1:] != bits[:-1]).sum(axis=1)
    unique = len({b.tobytes() for b in bits})
    return {
        "draws": int(bits.shape[0]),
        "relation_count": bits.sum(axis=1).astype(float),
        "edge_indicators": bits.astype(float),
        "closure_changed_between_draws": int((changes > 0).sum()),
        "relation_changes_per_change": (float(changes[changes > 0].mean())
                                        if (changes > 0).any() else 0.0),
        "unique_closures_visited": unique,
        "insufficient": False,
    }


def pooled_diagnostics(per_chain: list) -> dict:
    """R-hat and ESS across chains, on relation count and on EVERY role-pair indicator.

    With `m = 10` a skill has `m(m-1) = 90` directed role pairs, so the full relation
    space is `90K` indicators -- 2,700 at `K = 30`. That is small enough to diagnose
    exhaustively, and exhaustive is the right choice: any rule for picking a subset is a
    place where the true `U` could leak in through the back door, and a subset chosen
    before seeing the chains still risks missing exactly the edges that fail to mix.

    The time axis is the **`U`-update event**, not the FFBS sweep. Between two `U` updates
    the closure cannot change, so indexing by sweep would pad every series with runs of
    identical values and report an autocorrelation that is an artefact of the recording
    cadence rather than of the `U` kernel.
    """
    usable = [c for c in per_chain if not c.get("insufficient")]
    if len(usable) < 2:
        return {"rhat_relation_count": None, "rhat_selected_edge_max": None,
                "relation_level_ess_min": None, "chains_usable": len(usable)}
    n = min(c["draws"] for c in usable)
    counts = np.stack([c["relation_count"][:n] for c in usable])
    out = {"chains_usable": len(usable), "draws_per_chain": int(n),
           "rhat_relation_count": rank_normalized_split_rhat(counts)["rhat"],
           "ess_relation_count": bulk_ess(counts)}

    edges = np.stack([c["edge_indicators"][:n] for c in usable])     # (chains, n, E)

    # Classify every indicator before computing anything. R-hat is undefined when the
    # within-chain variance is zero, but "undefined" covers two opposite situations and
    # collapsing them would be a serious misreading: chains that AGREE on a fixed edge are
    # showing consensus, chains that are each frozen at DIFFERENT values are showing a
    # multimodality the sampler never escaped.
    constant_in_chain = edges.std(axis=1) == 0                       # (chains, E)
    all_constant = constant_in_chain.all(axis=0)
    first_value = edges[:, 0, :]
    same_everywhere = (first_value == first_value[0]).all(axis=0)

    consensus_fixed = all_constant & same_everywhere
    chain_disagreeing = all_constant & ~same_everywhere
    partially_frozen = constant_in_chain.any(axis=0) & ~all_constant
    eligible = np.flatnonzero(~all_constant)

    out.update(
        edges_total=int(edges.shape[2]),
        edges_consensus_fixed=int(consensus_fixed.sum()),
        edges_chain_disagreeing_frozen=int(chain_disagreeing.sum()),
        edges_partially_frozen=int(partially_frozen.sum()),
        edges_diagnosed=int(eligible.size),
        edge_selection=("exhaustive over all role-pair indicators; consensus-fixed "
                        "indicators are exempt (agreement, not failure), "
                        "chain-disagreeing frozen indicators are an automatic failure"))
    if chain_disagreeing.any():
        out["chain_disagreeing_warning"] = (
            f"{int(chain_disagreeing.sum())} indicator(s) frozen within every chain at "
            f"DIFFERENT values -- chains stuck in different orders")
    if eligible.size == 0:
        out.update(rhat_edge_max=None, relation_level_ess_min=None,
                   note=("every indicator was constant within all chains; no R-hat is "
                         "defined. This is a failure only if any of them disagree "
                         "between chains."))
        return out
    variable = eligible
    rhats, esss = [], []
    for e in variable:                       # every eligible indicator, no subset
        rhats.append(rank_normalized_split_rhat(edges[:, :, e])["rhat"])
        esss.append(bulk_ess(edges[:, :, e]))
    out.update(rhat_edge_max=float(np.nanmax(rhats)),
               rhat_edge_median=float(np.nanmedian(rhats)),
               relation_level_ess_min=float(np.nanmin(esss)),
               relation_level_ess_median=float(np.nanmedian(esss)))
    return out


#: The pass rule, frozen BEFORE any pilot output is inspected, and amended only while
#: still blind to it. A budget is selected by applying this mechanically -- never by
#: looking at which cell produced the prettiest table, and never by consulting recovery
#: against the sealed `U`.
PASS_RULE = {
    "select": ("the SMALLEST candidate X, applied GLOBALLY to every rung, for which no "
               "rung triggers any failure below"),
    "global_X": ("X must not vary with K. Letting each rung pick its own X would "
                 "reintroduce the compute-budget confound this rule exists to remove. "
                 "The full factorial is already executed, so nothing is re-run "
                 "adaptively: the completed grid is evaluated in the registered order "
                 "50 -> 100 -> 166.7 and the FIRST X satisfying the rule at every rung "
                 "is selected. If none does, this pilot terminates unsuccessfully."),
    "per_rung_u_scale": ("u_scale MAY be tuned and frozen per rung: it is a proposal "
                         "efficiency parameter matched to that rung's posterior "
                         "geometry, not a grant of extra compute. It is tuned AT that "
                         "rung and never carried over from a cheaper one."),
    "degenerate_edges": {
        "consensus_fixed": ("an indicator constant within every chain AND equal across "
                            "chains. R-hat and ESS are undefined; this is agreement, "
                            "not a mixing failure. Reported and counted separately, "
                            "EXEMPT from the R-hat and ESS requirements."),
        "chain_disagreeing_frozen": ("an indicator constant within every chain but NOT "
                                     "equal across chains: the chains are stuck in "
                                     "different orders. Automatic FAILURE."),
        "partially_frozen": ("constant in some chains and not others: retained in the "
                             "R-hat and ESS pools and counted, since within-chain "
                             "variance is defined."),
    },
    "failures": {
        "rhat_relation_count": "> 1.05",
        "rhat_edge_max": "> 1.05 over NON-DEGENERATE indicators",
        "relation_level_ess_min": "< 100 over NON-DEGENERATE indicators",
        "u_acceptance_retained": "outside [0.15, 0.60]",
        "closure_changing_fraction": "< 0.02 of accepted moves",
        "chain_disagreeing_frozen_edges": "> 0",
    },
    "replicates": ("the two replicates have DIFFERENT master truths and corpora, hence "
                   "different posterior targets. R-hat and ESS are computed within a "
                   "replicate over its four chains and NEVER pooled across replicates -- "
                   "an eight-chain R-hat across two posteriors measures nothing. A "
                   "(K, X, u_scale) cell counts as passing only if BOTH replicates pass "
                   "independently."),
    "scale_selection": ("among the u_scale values passing in BOTH replicates at the "
                        "selected X, take the one maximising the WORST-CASE relation "
                        "ESS over non-degenerate indicators, worst-cased again over the "
                        "two replicates; ties within ESS_TOLERANCE break by median "
                        "relation ESS, then by the fixed ascending scale ordering. "
                        "Deliberately hardware-INDEPENDENT: an earlier draft maximised "
                        "order-changing moves per second, which would have let a faster "
                        "machine choose a different kernel."),
    "if_none_pass": ("a failed candidate set TERMINATES this pilot. It is not an "
                     "invitation to keep trying scales or budgets on the same streams. "
                     "Any revision of the U kernel requires a separately preregistered "
                     "pilot on fresh CRN streams under a new pilot version."),
    "never": ("recovery against the sealed U is not an input to this decision, and "
              "neither is wall-clock: compute feasibility is a separate gate applied "
              "AFTER the statistical choice, never a reason to prefer a smaller X."),
}

#: Relative tolerance within which two scales count as tied on worst-case relation ESS.
ESS_TOLERANCE = 0.10


def evaluate_pass_rule(cell: dict) -> dict:
    """Apply the frozen rule to one cell. Mechanical, so it cannot drift on inspection."""
    fails = []
    rhat_count = cell.get("rhat_relation_count")
    rhat_edge = cell.get("rhat_edge_max")
    ess = cell.get("relation_level_ess_min")
    acc = cell.get("u_acceptance_retained")
    if rhat_count is not None and rhat_count > 1.05:
        fails.append(f"rhat_relation_count={rhat_count:.3f}>1.05")
    if rhat_edge is not None and rhat_edge > 1.05:
        fails.append(f"rhat_edge_max={rhat_edge:.3f}>1.05")
    if ess is not None and ess < 100:
        fails.append(f"relation_level_ess_min={ess:.1f}<100")
    if acc is not None and not (0.15 <= acc <= 0.60):
        fails.append(f"u_acceptance_retained={acc:.3f} outside [0.15,0.60]")
    if cell.get("edges_chain_disagreeing_frozen", 0) > 0:
        fails.append(f"{cell['edges_chain_disagreeing_frozen']} chain-disagreeing "
                     f"frozen edge(s): chains stuck in different orders")
    changing = cell.get("closure_changing_fraction")
    if changing is not None and changing < 0.02:
        fails.append(f"closure_changing_fraction={changing:.4f}<0.02")
    return {"passes": not fails, "failures": fails}


def select_scale(cells_by_scale: dict, n_replicates: int = 2) -> dict:
    """The frozen truth-free, hardware-independent scale rule for one rung at one X.

    `cells_by_scale` maps `u_scale` to the list of that scale's per-replicate cells. A
    scale is eligible only if **every** replicate passes: the replicates are separate
    posteriors and a scale that mixes in one but not the other has not been shown to work.

    Score is the worst-case relation ESS -- the minimum over non-degenerate indicators,
    minimised again over replicates -- because the binding constraint on a poset sampler
    is the relation it explores least, not its average. Nothing in the score depends on
    the machine.
    """
    eligible, incomplete = {}, []
    for scale, cells in cells_by_scale.items():
        if len({c.get("replicate") for c in cells}) < int(n_replicates):
            # a missing replicate is not a passing one
            incomplete.append(scale)
            continue
        if not cells or not all(c["pass_rule_verdict"]["passes"] for c in cells):
            continue
        worst = [c.get("relation_level_ess_min") for c in cells]
        median = [c.get("relation_level_ess_median") for c in cells]
        if any(v is None for v in worst):
            continue
        eligible[scale] = {
            "worst_case_ess": float(min(worst)),
            "median_ess": float(min(v for v in median if v is not None))
            if any(v is not None for v in median) else float("nan"),
            "replicates": len(cells)}
    if not eligible:
        return {"selected_u_scale": None, "scales_missing_a_replicate": sorted(incomplete),
                "reason": "no u_scale passed in every replicate at this rung"}

    peak = max(v["worst_case_ess"] for v in eligible.values())
    tied = {s: v for s, v in eligible.items()
            if v["worst_case_ess"] >= peak * (1 - ESS_TOLERANCE)}
    best_median = max(v["median_ess"] for v in tied.values())
    tied2 = {s: v for s, v in tied.items()
             if not np.isfinite(v["median_ess"])
             or v["median_ess"] >= best_median * (1 - ESS_TOLERANCE)}
    chosen = min(tied2)                       # fixed ascending scale ordering
    return {"selected_u_scale": chosen,
            "worst_case_relation_ess": eligible[chosen]["worst_case_ess"],
            "peak_worst_case_ess": peak,
            "tied_within_tolerance": sorted(tied),
            "eligible_scales": sorted(eligible),
            "scales_missing_a_replicate": sorted(incomplete),
            "reason": (f"max worst-case relation ESS over non-degenerate indicators, "
                       f"worst-cased over replicates; ties within {ESS_TOLERANCE:.0%} "
                       f"broken by median ESS then by ascending scale")}


def scaled_target(production_target: float, args) -> tuple:
    """The pilot target that reproduces production's moves-per-event rate.

    `u_scale` tuning and the per-move mixing diagnostics depend on how many proposals each
    update event makes, not on the chain's grand total. Scaling by the event ratio keeps
    the kernel identical to production while making the pilot affordable; the grand total
    (and therefore R-hat and ESS at the production budget) is deliberately NOT reproduced,
    and no convergence claim may be read off a pilot chain.
    """
    events_pilot = update_events(args.sweeps, args.u_every).size
    events_prod = update_events(args.production_sweeps, args.u_every).size
    if events_prod == 0 or events_pilot == 0:
        return production_target, 1.0
    ratio = events_pilot / events_prod
    return production_target * ratio, ratio


def run_cell(library, k, target, scale, args) -> dict:
    corpus = generate_ladder_corpus(library, k, args.replicate)
    u_truth, role_maps = library.prefix(k)
    model = Stage6EModel(traces=corpus.traces("train"), epsilon=args.epsilon,
                         delta_b=0.15, n_skills=k, n_roles=library.n_roles,
                         min_width=3, max_width=12, infer_pi_P=True,
                         eta_initial=1.0, eta_transition=1.0)
    pilot_target, ratio = scaled_target(target, args)
    schedule = quota_schedule(pilot_target, k, library.n_roles, args.sweeps,
                              args.warmup, args.u_every)
    began = time.perf_counter()
    per_chain, accepted, proposed, closure_changing = [], 0, 0, 0
    for chain in range(args.chains):
        result = run_ladder_chain(
            LEARNED_ORDER, model, role_maps, u_truth, chain=chain, sweeps=args.sweeps,
            warmup=args.warmup, seed=args.seed, epsilon=args.epsilon, thin=args.thin,
            u_every=args.u_every, u_scale=scale, replicate=args.replicate,
            target_u_attempts_per_role=pilot_target,
            crn=CommonRandomNumbers(args.replicate, k, chain))
        diag = chain_diagnostics(result["draws"]["u"])
        per_chain.append(diag)
        accepted += result["u_accepted_retained"]
        proposed += result["u_proposed_retained"]
        if not diag.get("insufficient"):
            closure_changing += diag["closure_changed_between_draws"]
        last = result
    seconds = time.perf_counter() - began
    pooled = pooled_diagnostics(per_chain)

    order_changing_rate = closure_changing / seconds if seconds else 0.0
    return {
        "K": k,
        "production_target_u_attempts_per_role": target,
        "pilot_target_u_attempts_per_role": pilot_target,
        "pilot_scaling_ratio": ratio,
        "moves_per_event": (schedule["total_quota_M_K"] / schedule["n_events"]
                            if schedule["n_events"] else 0.0),
        "u_scale": scale,
        "seconds": seconds,
        "M_K": schedule["total_quota_M_K"],
        "mean_attempts_per_role_total": schedule["mean_attempts_per_role_total"],
        "mean_attempts_per_role_burnin": schedule["mean_attempts_per_role_burnin"],
        "mean_attempts_per_role_retained": schedule["mean_attempts_per_role_retained"],
        "u_acceptance_burnin": last["u_acceptance_rate_burnin"],
        "u_acceptance_retained": (accepted / proposed) if proposed else None,
        "u_role_attempt_summary": last["u_role_attempt_summary"],
        "closure_changing_transitions": closure_changing,
        "closure_changing_fraction": (closure_changing / accepted) if accepted else None,
        "u_accepted_retained_total": accepted,
        "relation_changes_per_change": float(np.mean(
            [c["relation_changes_per_change"] for c in per_chain
             if not c.get("insufficient")] or [0.0])),
        "unique_closures_visited": int(np.sum(
            [c["unique_closures_visited"] for c in per_chain
             if not c.get("insufficient")])),
        "effective_order_changing_moves_per_second": order_changing_rate,
        **pooled,
        "truth_free": True,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rungs", type=int, nargs="+", default=[3, 30])
    p.add_argument("--targets", type=float, nargs="+", default=[50.0, 100.0, 166.7])
    p.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--sweeps", type=int, default=400)
    p.add_argument("--warmup", type=int, default=160)
    p.add_argument("--u-every", type=int, default=10)
    p.add_argument("--production-sweeps", type=int, default=50_000,
                   help=("chain length the candidate targets are stated for. The quota "
                         "is a per-CHAIN total, so a short pilot chain must scale it "
                         "down or it would pack the entire production budget into a "
                         "handful of events and cost more than production itself."))
    p.add_argument("--thin", type=int, default=4)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--replicate", type=int, default=0)
    p.add_argument("--seed", type=int, default=880_001)
    p.add_argument("--epsilon", type=float, default=0.02)
    p.add_argument("--library-seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "k_ladder_pilot" / "u_budget_pilot.json")
    args = p.parse_args()

    library, _ = draw_master_library(args.library_seed)
    events_pilot = update_events(args.sweeps, args.u_every).size
    events_prod = update_events(args.production_sweeps, args.u_every).size
    print("TRUTH-FREE pilot. No diagnostic below reads the sealed U.")
    print(f"Targets are stated for a {args.production_sweeps:,}-sweep production chain "
          f"and scaled by\n{events_pilot}/{events_prod} events so the pilot makes the "
          f"same proposals per event. R-hat and ESS\nhere describe the PILOT chain "
          f"length, not the production budget.\n")
    header = (f"{'K':>4} {'target':>7} {'scale':>6} {'acc(ret)':>9} {'closure':>8} "
              f"{'rel/chg':>8} {'uniq':>6} {'Rhat|rel|':>10} {'Rhat edge':>10} "
              f"{'ESS min':>8} {'ordmv/s':>8}")
    print(header)
    print("-" * len(header))
    rows = []
    for k in args.rungs:
        for target in args.targets:
            for scale in args.scales:
                r = run_cell(library, k, target, scale, args)
                r["pass_rule_verdict"] = evaluate_pass_rule(r)
                rows.append(r)
                def fmt(v, w, p=3):
                    if not isinstance(v, float) or not np.isfinite(v):
                        return f"{'-':>{w}}"
                    return f"{v:>{w}.{p}f}"
                print(f"{k:>4} {target:>7.1f} {scale:>6.2f} "
                      f"{fmt(r['u_acceptance_retained'],9)} "
                      f"{r['closure_changing_transitions']:>8} "
                      f"{r['relation_changes_per_change']:>8.2f} "
                      f"{r['unique_closures_visited']:>6} "
                      f"{fmt(r['rhat_relation_count'],10)} "
                      f"{fmt(r.get('rhat_edge_max'),10,2)} "
                      f"{fmt(r.get('relation_level_ess_min'),8,1)} "
                      f"{r['effective_order_changing_moves_per_second']:>8.2f}")

    print("\nSelection rule: prefer the (target, scale) with the highest effective "
          "order-changing\nmoves per second among cells whose R-hat is acceptable. "
          "Do NOT select on recovery.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema": "k-ladder-u-budget-pilot/1.0.0",
        "namespace": "PILOT",
        "settings": vars(args) | {"out": str(args.out)},
        "cells": rows,
        "pass_rule": PASS_RULE,
        "pass_rule_frozen_before_results": True,
        "truth_free": True,
        "note": ("Budget and u_scale are selected on chain-only diagnostics. Recovery "
                 "against the sealed U is never consulted here: it is the quantity the "
                 "study measures, and tuning on it would tune the experiment on its own "
                 "answer. u_scale is frozen after warm-up; burn-in and retained "
                 "proposals are recorded separately and never pooled."),
    }, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
