"""Condition C FORMAL run — C-COND vs C-MARG on the frozen formal corpus.

Run:  PYTHONPATH=src .venv/bin/python scripts/run_matched_condition_c_formal.py

Primary question (registered): does path-marginal inference solve the joint
(S, z)-U coupling problem? Two arms, one code path (`condition_c_sweep_once`):
C-COND every = 0, C-MARG every = 10 with the frozen scheduled scale. 4 paired
dispersed starts shared across arms, 8 chain seeds, checkpoints 30k/50k/75k/
100k, burn-in 10k, thin 5, Condition-B convergence gates, per-arm stop at two
consecutive checkpoint passes, ceiling 100k never extended. Recovery is
analysed ONLY after both arms stop. Everything below is frozen before the
first sweep.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_condition_b as mcb                      # noqa: E402
from hpop.mcmc_original import matched_condition_c as mcc                      # noqa: E402
from hpop.mcmc_original import matched_synthetic_generator as msg              # noqa: E402
from hpop.mcmc_original.collapsed_u_kernel import CollapsedUConfig             # noqa: E402
from hpop.mcmc_original.latent_poset import precedence_from_u                  # noqa: E402
from hpop.mcmc_original.sampler_u import sigma_rho_matrix                      # noqa: E402
from hpop.mcmc_original.stage6b_mcmc_diagnostics import (                      # noqa: E402
    bulk_ess, rank_normalized_split_rhat, tail_ess,
)

OUT = ROOT / "results" / "mcmc_original" / "matched_condition_c"
CHAINS = OUT / "formal_chains"
CORPUS_DIR = ROOT / "results" / "mcmc_original" / "matched_synthetic_formal_corpus"

# ------------------------------------------------------------- frozen protocol
RHO_0 = 0.5
GENERATION_SEED = 6_200_001
U_SCALE = 0.5                       # frozen by the pre-launch pilot
SCHEDULED_SCALE = 1.0               # frozen by the pre-launch pilot
CADENCE = 10                        # fixed by the pre-launch amendment
ARMS = {"cond": {"every": 0, "seeds": (6_204_001, 6_204_002, 6_204_003,
                                       6_204_004)},
        "marg": {"every": CADENCE, "seeds": (6_204_011, 6_204_012, 6_204_013,
                                             6_204_014)}}
START_SEEDS = (6_204_101, 6_204_102, 6_204_103, 6_204_104)
START_SCALES = (0.5, 1.0, 2.0, 3.0)
CHECKPOINTS = (30_000, 50_000, 75_000, 100_000)
BURN_IN = 10_000
THIN = 5
CHECKPOINT_EVERY = 2_000
RHAT_GATE = 1.01
ESS_FLOORS = {"log_target_bulk": 1000.0, "log_target_tail": 500.0,
              "total_relations_bulk": 1000.0, "uncertain_relation_bulk": 500.0}
UNCERTAIN_BAND = (0.05, 0.95)
THRESHOLD = 0.5

VERDICT_RULE = {
    "marg_not_converged": "'PATH-MARGINAL ARM NOT CONVERGED — NO CLAIM'",
    "marg_recovers_means": "mean closure F1 >= 0.90 AND heldout posterior-"
                           "mean NLL/occ - truth NLL/occ <= 0.05",
    "if_marg_converged_and_recovers": {
        "cond_not_converged": "'JOINT COUPLING SOLVED BY PATH-MARGINAL "
                              "INFERENCE — C-COND LOCKS, C-MARG CONVERGES'",
        "cond_converged_comparable": "closure F1 within 0.05 AND heldout NLL "
                                     "within 0.02 of C-MARG -> 'NO COUPLING "
                                     "BARRIER AT THIS SCALE — BOTH ARMS "
                                     "CONVERGE AND RECOVER'",
        "cond_converged_worse": "'PATH-MARGINAL INFERENCE SUPERIOR — C-COND "
                                "CONVERGES TO INFERIOR STRUCTURE'"},
    "marg_converged_no_recovery": "mean closure F1 < 0.60 -> 'STRUCTURE NOT "
                                  "IDENTIFIABLE UNDER JOINT INFERENCE'",
    "otherwise": "'PARTIAL — see report'",
}

_WORKER: dict = {}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _dump(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_environment():
    truth = msg.supplied_truth()
    corpus = msg.generate_corpus(
        GENERATION_SEED, tuple((24, 32, 40, 48)[i % 4] for i in range(100)),
        tuple((24, 32, 40, 48)[i % 4] for i in range(45)), truth)
    return truth, corpus


def make_start(index: int) -> np.ndarray:
    rng = np.random.default_rng(START_SEEDS[index])
    chol = np.linalg.cholesky(sigma_rho_matrix(2, RHO_0))
    return np.array([[START_SCALES[index] * (chol @ rng.standard_normal(2))
                      for _ in range(5)] for _ in range(3)])


# ================================================================ chain worker
def _advance_chain(args):
    arm, index, upto = args
    if "env" not in _WORKER:
        truth, corpus = build_environment()
        _WORKER["env"] = (truth, corpus)
    truth, corpus = _WORKER["env"]
    key = f"sampler_{arm}"
    if key not in _WORKER:
        model = mcc.build_condition_c_model(tuple(t.cpa for t in corpus.train))
        fixed = mcc.ConditionCFixed.from_truth(truth, RHO_0)
        _WORKER[key] = mcc.ConditionCSampler(
            model=model, fixed=fixed, u_scale=U_SCALE,
            collapsed=CollapsedUConfig(every=ARMS[arm]["every"],
                                       scale=SCHEDULED_SCALE))
    sampler = _WORKER[key]
    path = CHAINS / f"{arm}{index}.npz"
    if path.exists():
        chain = mcc.ConditionCChain.load(path, sampler)
    else:
        chain = mcc.ConditionCChain(sampler, make_start(index),
                                    seed=ARMS[arm]["seeds"][index],
                                    burn_in=BURN_IN, thin=THIN)
    chain.advance(upto, checkpoint_path=path,
                  checkpoint_every=CHECKPOINT_EVERY, progress_every=10_000)
    return arm, index, {
        "log_target": list(chain.retained_log_target),
        "log_prior": list(chain.retained_log_prior),
        "rel_counts": list(chain.retained_rel_counts),
        "indicators": np.asarray(chain.retained_indicators, dtype=bool),
        "movement": dict(chain.movement),
        "collapsed": [chain.collapsed_proposed, chain.collapsed_accepted,
                      chain.collapsed_h_accepted],
        "seconds": chain.seconds, "iteration": int(chain.state.iteration),
    }


# ============================================================ convergence gates
def _diag(series_by_chain) -> dict:
    chains = np.asarray(series_by_chain, dtype=float)
    per_chain_const = [np.all(c == c[0]) for c in chains]
    if all(per_chain_const):
        values = {float(c[0]) for c in chains}
        if len(values) == 1:
            return {"rhat": 1.0, "bulk_ess": float(chains.size),
                    "tail_ess": float(chains.size), "degenerate": "constant"}
        return {"rhat": float("inf"), "bulk_ess": 0.0, "tail_ess": 0.0,
                "degenerate": "constant-but-unequal"}
    return {"rhat": rank_normalized_split_rhat(chains)["rhat"],
            "bulk_ess": bulk_ess(chains), "tail_ess": tail_ess(chains),
            "degenerate": None}


def arm_gate(chain_rows: list) -> dict:
    """Condition-B gate set over one arm's 4 chains."""
    K = 3
    summaries = {
        "log_target": _diag([r["log_target"] for r in chain_rows]),
        "log_prior": _diag([r["log_prior"] for r in chain_rows]),
        "total_relations": _diag([r["rel_counts"] for r in chain_rows]),
    }
    per_skill = []
    for r in chain_rows:
        ind = np.asarray(r["indicators"], dtype=float)
        counts = ind.reshape(ind.shape[0], K, 20).sum(axis=2)
        per_skill.append(counts)
    for k in range(K):
        summaries[f"relations_skill{k}"] = _diag(
            [c[:, k] for c in per_skill])
        summaries[f"sorted_relations_rank{k}"] = _diag(
            [np.sort(c, axis=1)[:, k] for c in per_skill])
    indicators = [np.asarray(r["indicators"], dtype=float)
                  for r in chain_rows]
    pooled = np.concatenate(indicators, axis=0).mean(axis=0)
    uncertain = [int(i) for i in np.flatnonzero(
        (pooled >= UNCERTAIN_BAND[0]) & (pooled <= UNCERTAIN_BAND[1]))]
    uncertain_diags = {str(i): _diag([ind[:, i] for ind in indicators])
                       for i in uncertain}
    h_changes = [r["movement"]["u_h_accepted"] + r["collapsed"][2]
                 for r in chain_rows]
    checks = {
        "max_rhat": max(v["rhat"] for v in summaries.values()),
        "uncertain_max_rhat": max([v["rhat"]
                                   for v in uncertain_diags.values()],
                                  default=1.0),
        "log_target_bulk_ess": summaries["log_target"]["bulk_ess"],
        "log_target_tail_ess": summaries["log_target"]["tail_ess"],
        "total_relations_bulk_ess": summaries["total_relations"]["bulk_ess"],
        "uncertain_min_bulk_ess": min([v["bulk_ess"]
                                       for v in uncertain_diags.values()],
                                      default=float("inf")),
        "n_uncertain_relations": len(uncertain),
        "chains_with_zero_accepted_h_changes": sum(1 for h in h_changes
                                                   if h == 0),
        "accepted_h_changes_per_chain": h_changes,
    }
    passed = bool(
        checks["max_rhat"] <= RHAT_GATE
        and checks["uncertain_max_rhat"] <= RHAT_GATE
        and checks["log_target_bulk_ess"] >= ESS_FLOORS["log_target_bulk"]
        and checks["log_target_tail_ess"] >= ESS_FLOORS["log_target_tail"]
        and checks["total_relations_bulk_ess"]
        >= ESS_FLOORS["total_relations_bulk"]
        and checks["uncertain_min_bulk_ess"]
        >= ESS_FLOORS["uncertain_relation_bulk"]
        and checks["chains_with_zero_accepted_h_changes"] == 0)
    return {"summaries": summaries, "uncertain": uncertain_diags,
            "checks": checks, "pass": passed}


# ==================================================================== analysis
def _load_runner_b_helpers():
    spec = importlib.util.spec_from_file_location(
        "run_matched_condition_b", ROOT / "scripts/run_matched_condition_b.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyse_arm(arm: str, truth, corpus, helpers) -> dict:
    model_traces = tuple(t.cpa for t in corpus.train)
    sampler = mcc.ConditionCSampler(
        model=mcc.build_condition_c_model(model_traces),
        fixed=mcc.ConditionCFixed.from_truth(truth, RHO_0), u_scale=U_SCALE,
        collapsed=CollapsedUConfig(every=ARMS[arm]["every"],
                                   scale=SCHEDULED_SCALE))
    chains = [mcc.ConditionCChain.load(CHAINS / f"{arm}{c}.npz", sampler)
              for c in range(4)]
    K, m = 3, 5
    indicators = np.concatenate([np.asarray(c.retained_indicators, dtype=bool)
                                 for c in chains]).astype(float)
    pooled = indicators.mean(axis=0)
    per_skill_marginals = pooled.reshape(K, 20)
    true_closures = [precedence_from_u(truth.u_by_skill[k]) for k in range(K)]
    true_hashes = [mcb.canonical_h_hash(c) for c in true_closures]

    def marginal_matrix(k):
        matrix = np.zeros((m, m))
        pos = 0
        for i in range(m):
            for j in range(m):
                if i != j:
                    matrix[i, j] = per_skill_marginals[k, pos]
                    pos += 1
        return matrix

    structure = []
    for k in range(K):
        matrix = marginal_matrix(k)
        structure.append({
            "skill": k,
            "closure": mcb.closure_metrics(matrix, true_closures[k],
                                           THRESHOLD),
            "incomparable": mcb.incomparable_metrics(matrix, true_closures[k],
                                                     THRESHOLD),
            "reduction": mcb.reduction_metrics(
                (matrix >= THRESHOLD), true_closures[k]),
        })
    h_draws = [h for c in chains for h in c.retained_h_hashes]
    n_draws = len(h_draws)
    per_skill_h = [Counter(h[k] for h in h_draws) for k in range(K)]
    for k in range(K):
        modal_hash, modal_count = per_skill_h[k].most_common(1)[0]
        structure[k]["modal_h_equals_truth"] = modal_hash == true_hashes[k]
        structure[k]["true_h_posterior"] = per_skill_h[k].get(
            true_hashes[k], 0) / n_draws
        structure[k]["n_distinct_h"] = len(per_skill_h[k])

    # path marginals from the online accumulators
    boundary_f1_stats, ari, nmi, modal_acc = [], [], [], []
    tp = fp = fn = 0
    for n, trace in enumerate(corpus.train):
        draws = chains[0].marginal_draws
        boundary = sum(c.boundary_sums[n] for c in chains) \
            / sum(c.marginal_draws for c in chains)
        occupancy = sum(c.occupancy_sums[n] for c in chains) \
            / sum(c.marginal_draws for c in chains)
        true_cuts = set(trace.boundaries)
        predicted = {t + 1 for t in np.flatnonzero(boundary >= THRESHOLD)}
        tp += len(predicted & true_cuts)
        fp += len(predicted - true_cuts)
        fn += len(true_cuts - predicted)
        modal = np.argmax(occupancy, axis=1)
        true_occ = np.repeat(np.asarray(trace.labels),
                             np.asarray(trace.widths))
        modal_acc.append(float((modal == true_occ).mean()))
        ari.append(mcb.adjusted_rand_index(true_occ, modal))
        nmi.append(mcb.normalized_mutual_information(true_occ, modal))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    boundary_f1 = (2 * precision * recall / (precision + recall)
                   if tp and (precision + recall) > 0 else 0.0)

    # held-out oracle-path NLL via per-hash closure scoring
    heldout_blocks = mcb.oracle_blocks_by_skill(corpus.heldout, K)
    n_occ = sum(t.length for t in corpus.heldout)
    indicator_rows = np.concatenate(
        [np.asarray(c.retained_indicators, dtype=bool) for c in chains])
    hash_ll: list = [{} for _ in range(K)]
    for d, h in enumerate(h_draws):
        for k in range(K):
            if h[k] not in hash_ll[k]:
                closure = helpers._closure_from_indicators(indicator_rows[d],
                                                           k, m)
                hash_ll[k][h[k]] = float(helpers._per_block_for_closure(
                    heldout_blocks[k], closure, truth).sum())
    per_draw_nll = np.array([-sum(hash_ll[k][h[k]] for k in range(K)) / n_occ
                             for h in h_draws])
    heldout_lik = mcb.OracleBlockLikelihood(
        heldout_blocks, truth.beta, truth.epsilon, truth.omega,
        truth.lambda_rep, truth.lambda_back)
    truth_nll = -heldout_lik.total(truth.u_by_skill) / n_occ
    antichain_nll = -heldout_lik.total(np.zeros_like(truth.u_by_skill)) / n_occ

    total_seconds = sum(c.seconds for c in chains)
    h_changes = [c.movement["u_h_accepted"] + c.collapsed_h_accepted
                 for c in chains]
    rel_ess = _diag([c.retained_rel_counts for c in chains])["bulk_ess"]
    return {
        "structure": structure,
        "mean_closure_f1": float(np.mean([s["closure"]["f1"]
                                          for s in structure])),
        "true_h_tuple_posterior": Counter(h_draws).get(tuple(true_hashes), 0)
        / n_draws,
        "boundary_f1_at_0.5": boundary_f1,
        "boundary_precision": precision, "boundary_recall": recall,
        "occurrence_modal_accuracy": float(np.mean(modal_acc)),
        "occurrence_ari_mean": float(np.mean(ari)),
        "occurrence_nmi_mean": float(np.mean(nmi)),
        "heldout_posterior_mean_nll": float(per_draw_nll.mean()),
        "heldout_truth_nll": float(truth_nll),
        "heldout_antichain_nll": float(antichain_nll),
        "accepted_cross_h_total": int(sum(h_changes)),
        "accepted_cross_h_per_hour": float(sum(h_changes)
                                           / (total_seconds / 3600.0)),
        "structural_ess": float(rel_ess) if math.isfinite(rel_ess) else None,
        "structural_ess_per_second": (float(rel_ess / total_seconds)
                                      if math.isfinite(rel_ess) else None),
        "wall_clock_hours_sum": total_seconds / 3600.0,
        "seconds_per_sweep": total_seconds / sum(
            c.state.iteration for c in chains),
        "n_pooled_draws": n_draws,
        "u_acceptance": float(np.mean(
            [c.movement["u_accepted"] / c.movement["u_proposed"]
             for c in chains])),
        "collapsed_acceptance": (
            float(np.mean([c.collapsed_accepted / max(c.collapsed_proposed, 1)
                           for c in chains])) if arm == "marg" else None),
    }


def classify(converged: dict, analysis: dict) -> str:
    marg, cond = analysis["marg"], analysis["cond"]
    if not converged["marg"]:
        return "PATH-MARGINAL ARM NOT CONVERGED — NO CLAIM"
    recovers = (marg["mean_closure_f1"] >= 0.90
                and marg["heldout_posterior_mean_nll"]
                - marg["heldout_truth_nll"] <= 0.05)
    if recovers:
        if not converged["cond"]:
            return ("JOINT COUPLING SOLVED BY PATH-MARGINAL INFERENCE — "
                    "C-COND LOCKS, C-MARG CONVERGES")
        comparable = (abs(cond["mean_closure_f1"] - marg["mean_closure_f1"])
                      <= 0.05
                      and abs(cond["heldout_posterior_mean_nll"]
                              - marg["heldout_posterior_mean_nll"]) <= 0.02)
        if comparable:
            return ("NO COUPLING BARRIER AT THIS SCALE — BOTH ARMS CONVERGE "
                    "AND RECOVER")
        return ("PATH-MARGINAL INFERENCE SUPERIOR — C-COND CONVERGES TO "
                "INFERIOR STRUCTURE")
    if marg["mean_closure_f1"] < 0.60:
        return "STRUCTURE NOT IDENTIFIABLE UNDER JOINT INFERENCE"
    return "PARTIAL — see report"


def main() -> int:
    wall0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    CHAINS.mkdir(exist_ok=True)

    truth, corpus = build_environment()
    recorded = json.loads((CORPUS_DIR / "corpus_hash.json").read_text())
    if msg.corpus_hash(corpus) != recorded["corpus_hash_sha256"]:
        raise SystemExit("formal corpus hash changed — refusing to run")
    selected = json.loads((OUT / "selected_scales.json").read_text())
    if (selected["u_scale"] != U_SCALE
            or selected["scheduled_collapsed_scale"] != SCHEDULED_SCALE
            or selected["cadence"] != CADENCE):
        raise SystemExit("frozen scales drifted from selected_scales.json")

    starts = [make_start(c) for c in range(4)]
    start_hashes = [tuple(mcb.canonical_h_hash(precedence_from_u(s[k]))
                          for k in range(3)) for s in starts]
    truth_hashes = tuple(mcb.canonical_h_hash(
        precedence_from_u(truth.u_by_skill[k])) for k in range(3))
    if len(set(start_hashes)) != 4 or truth_hashes in start_hashes:
        raise SystemExit("starts are not four distinct non-truth H tuples")
    _dump("formal_registration.json", {
        "primary_question": "Does path-marginal inference solve the joint "
                            "(S,z)-U coupling problem?",
        "arms": {arm: {"every": cfg["every"], "seeds": list(cfg["seeds"])}
                 for arm, cfg in ARMS.items()},
        "paired_starts": {"seeds": list(START_SEEDS),
                          "scales": list(START_SCALES),
                          "h_hashes": [list(h) for h in start_hashes],
                          "truth_h_hashes": list(truth_hashes),
                          "pairing": "chain c of each arm starts from start c"},
        "u_scale": U_SCALE, "scheduled_scale": SCHEDULED_SCALE,
        "cadence": CADENCE,
        "checkpoints": list(CHECKPOINTS), "burn_in": BURN_IN, "thin": THIN,
        "within_segment_checkpoint_every": CHECKPOINT_EVERY,
        "stopping_rule": "per arm: PASS at two consecutive checkpoints; "
                         "ceiling 100k never extended",
        "convergence_gates": "Condition B set: max R-hat <= 1.01 over "
                             "{log_target, log_prior, total relations, "
                             "per-skill and sorted counts, uncertain "
                             "relations}; ESS floors 1000/500/1000/500; "
                             "degenerate-summary rule; movement rule (zero "
                             "accepted H changes cannot PASS)",
        "recovery_metrics": "as the Condition-C prompt: closure/incomparable/"
                            "reduction F1, boundary F1 at 0.5, occurrence "
                            "ARI/NMI/accuracy, held-out oracle-path NLL, "
                            "structural movement; primary 7-row table",
        "primary_table_rows": ["Boundary F1", "Occurrence ARI", "Closure F1",
                               "Held-out NLL", "Accepted cross-H/hour",
                               "Structural ESS/sec", "Wall clock"],
        "verdict_rule": VERDICT_RULE,
        "corpus_hash_sha256": recorded["corpus_hash_sha256"],
        "truth_hash_sha256": recorded["truth_hash_sha256"],
        "parent_commit": _git("rev-parse", "HEAD"),
        "no_recovery_inspection_between_checkpoints": True,
    })

    active = {"cond": True, "marg": True}
    consecutive = {"cond": 0, "marg": 0}
    stopped_at = {}
    checkpoint_log = []
    latest_rows: dict = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for checkpoint in CHECKPOINTS:
            jobs = [(arm, c, checkpoint) for arm in ARMS if active[arm]
                    for c in range(4)]
            if not jobs:
                break
            print(f"== advancing to {checkpoint:,} "
                  f"({sum(active.values())} arm(s) active) ==", flush=True)
            for arm, index, rows in pool.map(_advance_chain, jobs):
                latest_rows.setdefault(arm, [None] * 4)[index] = rows
            for arm in ARMS:
                if not active[arm]:
                    continue
                gate = arm_gate(latest_rows[arm])
                checkpoint_log.append({"checkpoint": checkpoint, "arm": arm,
                                       "pass": gate["pass"],
                                       "checks": gate["checks"]})
                _dump(f"formal_gate_{arm}_{checkpoint}.json", {
                    "checks": gate["checks"],
                    "summaries": gate["summaries"],
                    "uncertain": gate["uncertain"], "pass": gate["pass"]})
                consecutive[arm] = consecutive[arm] + 1 if gate["pass"] else 0
                print(f"  [{arm}] checkpoint {checkpoint:,}: "
                      f"{'PASS' if gate['pass'] else 'FAIL'} "
                      f"(max R-hat {gate['checks']['max_rhat']:.4f}, "
                      f"logT ESS "
                      f"{gate['checks']['log_target_bulk_ess']:.0f})",
                      flush=True)
                if consecutive[arm] >= 2:
                    active[arm] = False
                    stopped_at[arm] = {"checkpoint": checkpoint,
                                       "converged": True}
            if not any(active.values()):
                break
    for arm in ARMS:
        if arm not in stopped_at:
            stopped_at[arm] = {"checkpoint": CHECKPOINTS[-1],
                               "converged": False}
    converged = {arm: stopped_at[arm]["converged"] for arm in ARMS}
    _dump("formal_convergence.json", {"checkpoint_log": checkpoint_log,
                                      "stopped_at": stopped_at,
                                      "converged": converged})

    print("== recovery analysis (both arms stopped) ==", flush=True)
    helpers = _load_runner_b_helpers()
    analysis = {arm: analyse_arm(arm, truth, corpus, helpers) for arm in ARMS}
    _dump("formal_recovery.json", analysis)

    verdict = classify(converged, analysis)
    _dump("formal_verdict.json", {"classification": verdict,
                                  "converged": converged,
                                  "rule": VERDICT_RULE})

    def fmt(value, digits=4):
        return ("—" if value is None
                else f"{value:.{digits}f}" if isinstance(value, float)
                else str(value))
    cond, marg = analysis["cond"], analysis["marg"]
    table = [
        ("Boundary F1", fmt(cond["boundary_f1_at_0.5"], 3),
         fmt(marg["boundary_f1_at_0.5"], 3)),
        ("Occurrence ARI", fmt(cond["occurrence_ari_mean"], 3),
         fmt(marg["occurrence_ari_mean"], 3)),
        ("Closure F1", fmt(cond["mean_closure_f1"], 3),
         fmt(marg["mean_closure_f1"], 3)),
        ("Held-out NLL/occ", fmt(cond["heldout_posterior_mean_nll"]),
         fmt(marg["heldout_posterior_mean_nll"])),
        ("Accepted cross-H/hour", fmt(cond["accepted_cross_h_per_hour"], 1),
         fmt(marg["accepted_cross_h_per_hour"], 1)),
        ("Structural ESS/sec", fmt(cond["structural_ess_per_second"], 3),
         fmt(marg["structural_ess_per_second"], 3)),
        ("Wall clock (chain-hours)", fmt(cond["wall_clock_hours_sum"], 2),
         fmt(marg["wall_clock_hours_sum"], 2)),
    ]
    _dump("primary_table.json", {"rows": table,
                                 "truth_heldout_nll":
                                     marg["heldout_truth_nll"],
                                 "converged": converged})
    lines = [
        "# Condition C — formal result",
        "",
        f"Commit `{_git('rev-parse', 'HEAD')}` &middot; corpus "
        f"`{recorded['corpus_hash_sha256'][:16]}…` &middot; "
        f"stopped: C-COND at {stopped_at['cond']['checkpoint']:,} "
        f"({'converged' if converged['cond'] else 'NOT converged'}), "
        f"C-MARG at {stopped_at['marg']['checkpoint']:,} "
        f"({'converged' if converged['marg'] else 'NOT converged'})",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "| Metric | C-COND | C-MARG |",
        "|---|---|---|",
    ] + [f"| {name} | {a} | {b} |" for name, a, b in table] + [
        "",
        f"(truth held-out NLL/occ {marg['heldout_truth_nll']:.4f}; antichain "
        f"{marg['heldout_antichain_nll']:.4f}; everything else in "
        "formal_recovery.json — the appendix.)",
    ]
    (OUT / "formal_report.md").write_text("\n".join(lines) + "\n")
    _dump("formal_runtime.json", {
        "wall_seconds_orchestrator": time.perf_counter() - wall0,
        "per_arm_chain_hours": {arm: analysis[arm]["wall_clock_hours_sum"]
                                for arm in ARMS},
        "seconds_per_sweep": {arm: analysis[arm]["seconds_per_sweep"]
                              for arm in ARMS}})
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
