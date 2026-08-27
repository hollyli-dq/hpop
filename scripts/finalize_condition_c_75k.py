"""Freeze Condition C at its registered 75k terminal checkpoint.

Run:  PYTHONPATH=src .venv/bin/python scripts/finalize_condition_c_75k.py

This is NOT a new experiment. It stops nothing, samples nothing and changes no
sampler, target, gate or stopping rule. It reads the frozen artifacts, restricts
every primary quantity to draws retained through sweep 75,000, and writes the
terminal record.

Draw indexing: retained draw `i` corresponds to sweep `burn_in + (i+1)*thin`
= `10000 + (i+1)*5`, so draws through 75,000 are exactly the first 13,000.
Anything after that is post-decision continuation and is quarantined, not used
and not deleted.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.matched_condition_a import (                          # noqa: E402
    adjusted_rand_index, normalized_mutual_information,
)
from hpop.mcmc_original import matched_condition_b as mcb                      # noqa: E402
from hpop.mcmc_original import matched_synthetic_generator as msg              # noqa: E402
from hpop.mcmc_original.latent_poset import precedence_from_u                  # noqa: E402
from hpop.mcmc_original.stage6b_mcmc_diagnostics import (                      # noqa: E402
    bulk_ess, rank_normalized_split_rhat, tail_ess,
)

C_DIR = ROOT / "results" / "mcmc_original" / "matched_condition_c"
CHAINS = C_DIR / "formal_chains"
OUT = ROOT / "results" / "mcmc_original" / "condition_c_terminal_75k"
FIG = OUT / "figure_data"

TERMINAL_SWEEP = 75_000
BURN_IN, THIN = 10_000, 5
DRAWS_THROUGH_TERMINAL = (TERMINAL_SWEEP - BURN_IN) // THIN     # 13,000
RUNGS = (30_000, 50_000, 75_000)
ARMS = ("cond", "marg")


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def runner_helpers():
    spec = importlib.util.spec_from_file_location(
        "run_matched_condition_b", ROOT / "scripts/run_matched_condition_b.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ load + truncate
def load_chain(path: Path) -> dict:
    data = np.load(str(path))
    meta = json.loads(str(data["meta"]))
    h = [tuple(str(v) for v in row) for row in data["h_hashes"]]
    n_traces = sum(1 for k in data.files if k.startswith("b"))
    return {
        "name": path.stem, "arm": path.stem[:4],
        "sweeps_at_stop": int(meta["iteration"]),
        "seconds": float(meta["seconds"]), "movement": meta["movement"],
        "collapsed": [int(v) for v in meta["collapsed"]],
        "segments_at_stop": [len(s) for s in meta["segmentations"]],
        "log_target": np.asarray(data["log_target"], dtype=float),
        "log_prior": np.asarray(data["log_prior"], dtype=float),
        "rel_counts": np.asarray(data["rel_counts"], dtype=int),
        "indicators": np.asarray(data["indicators"], dtype=bool),
        "h_hashes": h,
        "boundary_sums": [np.asarray(data[f"b{n:03d}"], dtype=float)
                          for n in range(n_traces)],
        "occupancy_sums": [np.asarray(data[f"o{n:03d}"], dtype=float)
                           for n in range(n_traces)],
        "marginal_draws": int(meta["marginal_draws"]),
    }


def truncate(chain: dict, n: int) -> dict:
    """Primary view: the first `n` retained draws = state through sweep 75,000."""
    out = dict(chain)
    for key in ("log_target", "log_prior", "rel_counts", "indicators"):
        out[key] = chain[key][:n]
    out["h_hashes"] = chain["h_hashes"][:n]
    return out


def sweep_of(i: int) -> int:
    return BURN_IN + (i + 1) * THIN


# ------------------------------------------------------------------- diagnostics
def diag(series) -> dict:
    c = np.asarray(series, dtype=float)
    if all(np.all(r == r[0]) for r in c):
        vals = {float(r[0]) for r in c}
        if len(vals) == 1:
            return {"rhat": 1.0, "bulk_ess": float(c.size),
                    "degenerate": "constant"}
        return {"rhat": float("inf"), "bulk_ess": 0.0,
                "degenerate": "constant-but-unequal"}
    return {"rhat": rank_normalized_split_rhat(c)["rhat"],
            "bulk_ess": bulk_ess(c), "tail_ess": tail_ess(c),
            "degenerate": None}


def last_change(seq) -> dict:
    last = None
    n = 0
    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            last, n = i, n + 1
    return {"last_change_sweep": sweep_of(last) if last is not None else None,
            "n_changes_after_burn_in": n,
            "distinct_values": len(set(seq))}


def arm_structure(chains: list) -> dict:
    anchored = [tuple(c["h_hashes"][-1]) for c in chains]
    libs = [tuple(sorted(t)) for t in anchored]
    groups: dict = {}
    for c, t in zip(chains, anchored):
        groups.setdefault(t, []).append(c["name"])
    lib_groups: dict = {}
    for c, l in zip(chains, libs):
        lib_groups.setdefault(l, []).append(c["name"])
    by_mode = []
    for t, names in groups.items():
        v = np.concatenate([c["log_target"] for c in chains
                            if c["name"] in names])
        by_mode.append({"anchored_tuple": list(t), "chains": names,
                        "log_target_mean": float(v.mean()),
                        "log_target_sd": float(v.std(ddof=1))})
    by_mode.sort(key=lambda r: -r["log_target_mean"])
    return {
        "distinct_unordered_libraries": len(lib_groups),
        "library_split": sorted((len(v) for v in lib_groups.values()),
                                reverse=True),
        "distinct_anchored_assignments": len(groups),
        "assignment_split": sorted((len(v) for v in groups.values()),
                                   reverse=True),
        "modes": by_mode,
        "assignment_gap_nats": (by_mode[0]["log_target_mean"]
                                - by_mode[1]["log_target_mean"])
        if len(by_mode) > 1 else None,
        "statement": ("shares one unordered structural library; separated in "
                      "the assignment of that library to anchored skill "
                      "identities" if len(lib_groups) == 1 and len(groups) > 1
                      else "separated at the structural-library level"
                      if len(lib_groups) > 1 else "chains agree"),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    helpers = runner_helpers()

    raw = {p.stem: load_chain(p) for p in sorted(CHAINS.glob("*.npz"))}
    prim = {k: truncate(v, DRAWS_THROUGH_TERMINAL) for k, v in raw.items()}

    # ---------------------------------------------------------------- quarantine
    quarantine = {
        "terminal_sweep_for_primary_analysis": TERMINAL_SWEEP,
        "draw_indexing": "retained draw i corresponds to sweep 10000+(i+1)*5",
        "draws_used_per_chain": DRAWS_THROUGH_TERMINAL,
        "per_chain": {},
        "label": "POST-DECISION / POST-75k CONTINUATION — NOT USED IN PRIMARY "
                 "CONDITION-C ANALYSIS",
        "deleted": False,
        "note": "the continuation draws remain inside the chain .npz files; "
                "they are excluded by index, not removed",
    }
    for name, c in raw.items():
        excess = len(c["log_target"]) - DRAWS_THROUGH_TERMINAL
        quarantine["per_chain"][name] = {
            "max_sweep_reached": c["sweeps_at_stop"],
            "retained_draws_total": len(c["log_target"]),
            "draws_used_primary": DRAWS_THROUGH_TERMINAL,
            "post_75k_draws_quarantined": excess,
            "post_75k_sweeps": c["sweeps_at_stop"] - TERMINAL_SWEEP,
        }
    quarantine["accumulator_caveat"] = (
        "the per-trace boundary and occupancy accumulators are cumulative sums "
        "kept without per-draw storage, so unlike the per-draw arrays they "
        "cannot be truncated to 75,000. The checkpoint written at exactly "
        "75,000 was overwritten by the continuation before the termination "
        "decision. Path-marginal recovery quantities derived from them are "
        "therefore reported as SUPPLEMENTARY, computed over each chain's full "
        "retained set; the maximum possible shift in any marginal from the "
        "extra draws is bounded by excess/total, i.e. "
        + ", ".join(f"{n} <= {v['post_75k_draws_quarantined']}/"
                    f"{v['retained_draws_total']} = "
                    f"{v['post_75k_draws_quarantined']/v['retained_draws_total']:.3%}"
                    for n, v in quarantine["per_chain"].items()))
    dump("quarantine_manifest.json", quarantine)

    # ------------------------------------------------------------------- hashes
    files = {}
    for p in sorted(C_DIR.glob("*.json")) + sorted(C_DIR.glob("*.md")):
        files[str(p.relative_to(ROOT))] = sha(p)
    for p in sorted(CHAINS.glob("*.npz")):
        files[str(p.relative_to(ROOT))] = sha(p)
    for rel in ("src/hpop/mcmc_original/matched_condition_c.py",
                "src/hpop/mcmc_original/skill_swap_kernel.py",
                "src/hpop/mcmc_original/matched_condition_c_prime.py",
                "scripts/run_matched_condition_c_formal.py",
                "scripts/resume_matched_condition_c_formal.py",
                "scripts/run_matched_condition_c_prime_formal.py",
                "scripts/condition_c_diagnostics.py",
                "scripts/finalize_condition_c_75k.py"):
        if (ROOT / rel).exists():
            files[rel] = sha(ROOT / rel)
    dump("artifact_hashes.json", {
        "generated_at_commit": git("rev-parse", "HEAD"),
        "condition_c_launch_commit": "50eee50",
        "cprime_preregistration_commit": "9b8e590",
        "cprime_runner_commit": "ed63b55",
        "gate_files_are_registered_and_were_not_recomputed": True,
        "hashes": files})

    # ------------------------------------------------- checkpoint trajectory table
    rows = []
    for rung in RUNGS:
        for arm in ARMS:
            g = json.loads((C_DIR / f"formal_gate_{arm}_{rung}.json").read_text())
            ch = g["checks"]
            n_draws = (rung - BURN_IN) // THIN
            sub = [truncate(raw[f"{arm}{i}"], n_draws) for i in range(4)]
            st = arm_structure(sub)
            seg = [sum(c["segments_at_stop"]) for c in sub] if rung == RUNGS[-1] \
                else None
            # cross-chain path-marginal disagreement is only available from the
            # (untruncatable) accumulators, so report it at the terminal rung only
            bnd = occ = None
            if rung == TERMINAL_SWEEP:
                bnd = occ = 0.0
                for i in range(4):
                    for j in range(i + 1, 4):
                        a, b = sub[i], sub[j]
                        for x, y in zip(raw[a["name"]]["occupancy_sums"],
                                        raw[b["name"]]["occupancy_sums"]):
                            occ = max(occ, float(np.abs(
                                x / raw[a["name"]]["marginal_draws"]
                                - y / raw[b["name"]]["marginal_draws"]).max()))
                        for x, y in zip(raw[a["name"]]["boundary_sums"],
                                        raw[b["name"]]["boundary_sums"]):
                            bnd = max(bnd, float(np.abs(
                                x / raw[a["name"]]["marginal_draws"]
                                - y / raw[b["name"]]["marginal_draws"]).max()))
            rows.append({
                "rung": rung, "arm": arm,
                "registered_verdict": "PASS" if g["pass"] else "FAIL",
                "max_anchored_rhat": ch["max_rhat"],
                "log_target_bulk_ess": round(ch["log_target_bulk_ess"], 1),
                "log_target_tail_ess": round(ch["log_target_tail_ess"], 1),
                "total_relations_rhat": (
                    ch.get("max_rhat") if False else
                    round(g["summaries"]["total_relations"]["rhat"], 3)
                    if math.isfinite(g["summaries"]["total_relations"]["rhat"])
                    else "inf"),
                "total_relations_bulk_ess":
                    round(ch["total_relations_bulk_ess"], 1),
                "uncertain_relation_indicators": ch["n_uncertain_relations"],
                "distinct_unordered_libraries":
                    st["distinct_unordered_libraries"],
                "library_split": "-".join(map(str, st["library_split"])),
                "anchored_assignment_split":
                    "-".join(map(str, st["assignment_split"])),
                "top_mode_log_target": round(st["modes"][0]["log_target_mean"], 2),
                "second_mode_log_target":
                    round(st["modes"][1]["log_target_mean"], 2)
                    if len(st["modes"]) > 1 else None,
                "assignment_gap_nats": round(st["assignment_gap_nats"], 2)
                    if st["assignment_gap_nats"] is not None else None,
                "accepted_cross_h_per_chain":
                    "/".join(map(str, ch["accepted_h_changes_per_chain"])),
                "segment_total_spread": (max(seg) - min(seg)) if seg else None,
                "max_pairwise_boundary_marginal_diff":
                    round(bnd, 3) if bnd is not None else None,
                "max_pairwise_occurrence_marginal_diff":
                    round(occ, 3) if occ is not None else None,
            })
    with (OUT / "condition_c_checkpoint_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    dump("condition_c_checkpoint_table.json", rows)

    # ------------------------------------------------------------ failure geometry
    geom = {"terminal_sweep": TERMINAL_SWEEP, "arms": {}}
    for arm in ARMS:
        sub = [prim[f"{arm}{i}"] for i in range(4)]
        st = arm_structure(sub)
        per_chain = {}
        for c in sub:
            r = raw[c["name"]]
            per_chain[c["name"]] = {
                "anchored_tuple": list(c["h_hashes"][-1]),
                "unordered_library": sorted(set(c["h_hashes"][-1])),
                "anchored": last_change(c["h_hashes"]),
                "library": last_change([tuple(sorted(h))
                                        for h in c["h_hashes"]]),
                "log_target_mean": float(c["log_target"].mean()),
                "ordinary_u_proposed": r["movement"]["u_proposed"],
                "ordinary_u_accepted": r["movement"]["u_accepted"],
                "ordinary_u_accepted_h_changes": r["movement"]["u_h_accepted"],
                "path_marginal_proposed": r["collapsed"][0],
                "path_marginal_accepted": r["collapsed"][1],
                "path_marginal_accepted_that_changed_h": r["collapsed"][2],
            }
        geom["arms"][arm] = {
            "structure": st, "per_chain": per_chain,
            "anchored_assignment_changes_after_burn_in_total":
                sum(v["anchored"]["n_changes_after_burn_in"]
                    for v in per_chain.values()),
            "path_marginal_accepts_total":
                sum(v["path_marginal_accepted"] for v in per_chain.values()),
            "path_marginal_accepts_that_changed_h_total":
                sum(v["path_marginal_accepted_that_changed_h"]
                    for v in per_chain.values()),
        }
    geom["assignment_gap_trajectory"] = [
        {"rung": r["rung"], "arm": r["arm"],
         "assignment_gap_nats": r["assignment_gap_nats"]}
        for r in rows if r["arm"] == "marg"]
    geom["distinction"] = ("accepted path-marginal proposals are NOT anchored-"
                           "assignment crossings: the counts above separate "
                           "within-cell accepts from accepts that changed the "
                           "anchored H tuple")
    geom["language_note"] = ("no claim of mathematical impossibility is made; "
                             "the observation is that no retained C-MARG chain "
                             "traversed the anchored-assignment barrier within "
                             "the observed run through 75,000 sweeps")
    dump("condition_c_failure_geometry.json", geom)

    # -------------------------------------------------------- predefined recovery
    truth = msg.supplied_truth()
    corpus = msg.generate_corpus(
        6_200_001, tuple((24, 32, 40, 48)[i % 4] for i in range(100)),
        tuple((24, 32, 40, 48)[i % 4] for i in range(45)), truth)
    K, m = truth.n_skills, truth.n_roles
    true_cl = [precedence_from_u(truth.u_by_skill[k]) for k in range(K)]
    true_hash = [mcb.canonical_h_hash(c) for c in true_cl]
    heldout_blocks = mcb.oracle_blocks_by_skill(corpus.heldout, K)
    n_occ = sum(t.length for t in corpus.heldout)
    hl = mcb.OracleBlockLikelihood(heldout_blocks, truth.beta, truth.epsilon,
                                   truth.omega, truth.lambda_rep,
                                   truth.lambda_back)
    truth_nll = -hl.total(truth.u_by_skill) / n_occ
    anti_nll = -hl.total(np.zeros_like(truth.u_by_skill)) / n_occ

    def marginals_from(indicators):
        pooled = np.asarray(indicators, dtype=float).mean(axis=0)
        per = np.zeros((K, m, m))
        idx = 0
        for k in range(K):
            for i in range(m):
                for j in range(m):
                    if i != j:
                        per[k, i, j] = pooled[idx]
                        idx += 1
        return per

    def structural_recovery(indicators, h_list):
        per = marginals_from(indicators)
        out = []
        for k in range(K):
            out.append({
                "skill": k,
                "closure": mcb.closure_metrics(per[k], true_cl[k], 0.5),
                "incomparable": mcb.incomparable_metrics(per[k], true_cl[k], 0.5),
                "reduction": mcb.reduction_metrics(per[k] >= 0.5, true_cl[k]),
            })
        cnt = Counter(h_list)
        return {
            "per_skill": out,
            "mean_closure_f1": float(np.mean([s["closure"]["f1"] for s in out])),
            "true_anchored_tuple_posterior": cnt.get(tuple(true_hash), 0)
            / max(len(h_list), 1),
            "modal_anchored_tuple_equals_truth":
                cnt.most_common(1)[0][0] == tuple(true_hash),
            "true_library_posterior": sum(
                v for kk, v in cnt.items()
                if tuple(sorted(kk)) == tuple(sorted(true_hash)))
            / max(len(h_list), 1),
        }

    hash_blocks: dict = {k: {} for k in range(K)}

    def heldout_nll(h_list, indicator_rows):
        for d, h in enumerate(h_list):
            for k in range(K):
                if h[k] not in hash_blocks[k]:
                    cl = helpers._closure_from_indicators(indicator_rows[d], k, m)
                    hash_blocks[k][h[k]] = helpers._per_block_for_closure(
                        heldout_blocks[k], cl, truth)
        per_draw = np.array([
            -sum(float(hash_blocks[k][h[k]].sum()) for k in range(K)) / n_occ
            for h in h_list])
        return {"mean": float(per_draw.mean()), "sd": float(per_draw.std(ddof=1))
                if per_draw.size > 1 else 0.0}

    recovery = {
        "SEALING": "recovery was opened only after Condition C terminated",
        "draws_used": f"first {DRAWS_THROUGH_TERMINAL} retained draws per chain "
                      f"= sweeps through {TERMINAL_SWEEP}",
        "NON_CONVERGED_WARNING":
            "Recovery summaries are descriptive diagnostics conditional on "
            "non-converged chains and are not interpreted as posterior "
            "estimates.",
        "reference_levels": {"generating_truth_heldout_nll_per_occ": truth_nll,
                             "antichain_baseline_nll_per_occ": anti_nll},
        "per_chain": {}, "per_arm": {}, "pooled_per_arm": {},
        "supplementary_path_marginal": {},
    }
    for name, c in prim.items():
        rec = structural_recovery(c["indicators"], c["h_hashes"])
        rec["heldout_oracle_path_nll_per_occ"] = heldout_nll(
            c["h_hashes"], np.asarray(c["indicators"], dtype=bool))
        rec["anchored_tuple"] = list(c["h_hashes"][-1])
        recovery["per_chain"][name] = rec
    for arm in ARMS:
        sub = [prim[f"{arm}{i}"] for i in range(4)]
        recovery["per_arm"][arm] = {
            "per_chain_mean_closure_f1": [
                recovery["per_chain"][c["name"]]["mean_closure_f1"]
                for c in sub],
            "per_chain_heldout_nll": [
                recovery["per_chain"][c["name"]]
                ["heldout_oracle_path_nll_per_occ"]["mean"] for c in sub],
            "chains_whose_modal_tuple_equals_truth": sum(
                recovery["per_chain"][c["name"]]
                ["modal_anchored_tuple_equals_truth"] for c in sub),
            "chains_recovering_the_true_library": sum(
                1 for c in sub
                if recovery["per_chain"][c["name"]]["true_library_posterior"]
                > 0.5),
        }
        ind = np.concatenate([np.asarray(c["indicators"], dtype=bool)
                              for c in sub])
        hs = [h for c in sub for h in c["h_hashes"]]
        pooled = structural_recovery(ind, hs)
        pooled["WARNING"] = ("pooled across NON-CONVERGED chains; not a "
                             "posterior estimate")
        recovery["pooled_per_arm"][arm] = pooled

        # supplementary path marginals (accumulators; see quarantine caveat)
        tp = fp = fn = 0
        acc, ari, nmi = [], [], []
        for n, trace in enumerate(corpus.train):
            bnd = sum(raw[c["name"]]["boundary_sums"][n] for c in sub) / sum(
                raw[c["name"]]["marginal_draws"] for c in sub)
            occ = sum(raw[c["name"]]["occupancy_sums"][n] for c in sub) / sum(
                raw[c["name"]]["marginal_draws"] for c in sub)
            true_cuts = set(trace.boundaries)
            pred = {t + 1 for t in np.flatnonzero(bnd >= 0.5)}
            tp += len(pred & true_cuts); fp += len(pred - true_cuts)
            fn += len(true_cuts - pred)
            modal = np.argmax(occ, axis=1)
            true_occ = np.repeat(np.asarray(trace.labels),
                                 np.asarray(trace.widths))
            acc.append(float((modal == true_occ).mean()))
            ari.append(adjusted_rand_index(true_occ, modal))
            nmi.append(normalized_mutual_information(true_occ, modal))
        pr = tp / (tp + fp) if tp + fp else float("nan")
        rc = tp / (tp + fn) if tp + fn else float("nan")
        recovery["supplementary_path_marginal"][arm] = {
            "STATUS": "SUPPLEMENTARY — computed over each chain's full retained "
                      "set because the accumulators cannot be truncated to "
                      "75,000; see quarantine_manifest.json",
            "boundary_f1_at_0.5": (2 * pr * rc / (pr + rc)
                                   if tp and (pr + rc) > 0 else 0.0),
            "boundary_precision": pr, "boundary_recall": rc,
            "occurrence_modal_accuracy": float(np.mean(acc)),
            "occurrence_ari_mean": float(np.mean(ari)),
            "occurrence_nmi_mean": float(np.mean(nmi)),
        }
    dump("recovery_75k.json", recovery)

    # ------------------------------------------------------------- runtime + figs
    runtime = {
        "primary_analysis_sweeps_per_chain": TERMINAL_SWEEP,
        "primary_total_chain_sweeps_per_arm": TERMINAL_SWEEP * 4,
        "retained_draws_per_chain_primary": DRAWS_THROUGH_TERMINAL,
        "per_chain": {n: {"sweeps_at_stop": c["sweeps_at_stop"],
                          "compute_seconds_total": c["seconds"],
                          "seconds_per_sweep": c["seconds"] / c["sweeps_at_stop"]}
                      for n, c in raw.items()},
        "post_decision_compute": {
            "sweeps_beyond_75k": {n: c["sweeps_at_stop"] - TERMINAL_SWEEP
                                  for n, c in raw.items()},
            "note": "produced by the automatic advance toward the 100k ceiling "
                    "before the termination decision; excluded from primary "
                    "analysis"},
        "chain_hours_total_including_post_decision":
            sum(c["seconds"] for c in raw.values()) / 3600,
    }
    dump("runtime_accounting.json", runtime)

    with (FIG / "condition_c_convergence.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rung", "arm", "verdict", "log_target_bulk_ess",
                    "total_relations_bulk_ess", "uncertain_relation_indicators",
                    "distinct_unordered_libraries", "anchored_assignment_split"])
        for r in rows:
            w.writerow([r["rung"], r["arm"], r["registered_verdict"],
                        r["log_target_bulk_ess"], r["total_relations_bulk_ess"],
                        r["uncertain_relation_indicators"],
                        r["distinct_unordered_libraries"],
                        r["anchored_assignment_split"]])
    with (FIG / "condition_c_assignment_gap.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rung", "arm", "assignment_gap_nats"])
        for r in rows:
            if r["assignment_gap_nats"] is not None:
                w.writerow([r["rung"], r["arm"], r["assignment_gap_nats"]])
    with (FIG / "condition_c_library_assignment.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "chain", "anchored_H_tuple", "unordered_library",
                    "log_target_mean"])
        for arm in ARMS:
            for i in range(4):
                c = prim[f"{arm}{i}"]
                w.writerow([arm, c["name"], "|".join(c["h_hashes"][-1]),
                            "|".join(sorted(set(c["h_hashes"][-1]))),
                            round(float(c["log_target"].mean()), 2)])
    with (FIG / "condition_c_movement.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "chain", "ordinary_u_proposed", "ordinary_u_accepted",
                    "ordinary_u_accepted_h_changes", "path_marginal_proposed",
                    "path_marginal_accepted",
                    "path_marginal_accepted_that_changed_h",
                    "anchored_assignment_changes_after_burn_in",
                    "last_anchored_change_sweep"])
        for arm in ARMS:
            for i in range(4):
                n = f"{arm}{i}"
                g = geom["arms"][arm]["per_chain"][n]
                w.writerow([arm, n, g["ordinary_u_proposed"],
                            g["ordinary_u_accepted"],
                            g["ordinary_u_accepted_h_changes"],
                            g["path_marginal_proposed"],
                            g["path_marginal_accepted"],
                            g["path_marginal_accepted_that_changed_h"],
                            g["anchored"]["n_changes_after_burn_in"],
                            g["anchored"]["last_change_sweep"]])

    # -------------------------------------------------------------- verdict
    dump("terminal_verdict.json", {
        "headline": "CONDITION C TERMINATED AT 75k — FORMAL VERDICT: NOT CONVERGED",
        "A_REGISTERED_FORMAL_VERDICT": {
            "C-COND": "NOT CONVERGED", "C-MARG": "NOT CONVERGED",
            "basis": "neither arm obtained two consecutive passing checkpoints; "
                     "30k, 50k and 75k all FAILED",
            "registered_history": {str(r): {a: ("PASS" if json.loads(
                (C_DIR / f"formal_gate_{a}_{r}.json").read_text())["pass"]
                else "FAIL") for a in ARMS} for r in RUNGS}},
        "B_SCIENTIFIC_DIAGNOSIS": {
            "C-COND": geom["arms"]["cond"]["structure"]["statement"],
            "C-MARG": geom["arms"]["marg"]["structure"]["statement"],
            "phenomenon": "anchored structure-to-skill assignment multimodality",
            "not_label_switching": "fixed pi* and P* are not invariant under any "
                                   "non-identity permutation, so the competing "
                                   "assignments are distinct posterior states"},
        "early_termination_deviation":
            "Condition C was terminated after the third consecutive failed "
            "registered checkpoint at 75k sweeps, before the preregistered 100k "
            "ceiling. Because the protocol required two consecutive passing "
            "checkpoints and only one checkpoint remained, the registered "
            "convergence criterion was no longer attainable.",
        "terminal_commit": "recorded at commit time"})

    print("checkpoint table rows:", len(rows))
    for arm in ARMS:
        s = geom["arms"][arm]["structure"]
        print(f"  {arm}: libs={s['distinct_unordered_libraries']} "
              f"split={s['library_split']} assign={s['assignment_split']} "
              f"gap={s['assignment_gap_nats']}")
    print("gap trajectory:", [(g["rung"], g["assignment_gap_nats"])
                              for g in geom["assignment_gap_trajectory"]])
    print("quarantined post-75k draws:",
          {n: v["post_75k_draws_quarantined"]
           for n, v in quarantine["per_chain"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
