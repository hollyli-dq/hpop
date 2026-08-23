"""Condition C PRE-LAUNCH: reconciliation record, equality test, pilot, freeze.

Run:  PYTHONPATH=src .venv/bin/python scripts/condition_c_prelaunch.py

Implements the registered pre-launch amendment: source manifest over the
reconciled implementation, the C-COND vs C-MARG small-reference equality test
(the two arms must agree on the same posterior), a separately registered
efficiency-only proposal pilot on the formal corpus, and the frozen common
scheduled-proposal scale with cadence c = 10. STOPS before any formal
Condition-C chain.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_condition_c as mcc                      # noqa: E402
from hpop.mcmc_original import matched_synthetic_generator as msg              # noqa: E402
from hpop.mcmc_original.collapsed_u_kernel import CollapsedUConfig             # noqa: E402
from hpop.mcmc_original.sampler_u import sigma_rho_matrix                      # noqa: E402
from hpop.mcmc_original.stage6b_mcmc_diagnostics import bulk_ess               # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "matched_condition_c"
CORPUS_DIR = ROOT / "results" / "mcmc_original" / "matched_synthetic_formal_corpus"

RHO_0 = 0.5
GENERATION_SEED = 6_200_001
CADENCE = 10                        # fixed by the pre-launch amendment; not tuned
COLLAPSED_SCALE_DEFAULT = 0.5       # the registered production row scale

EQ = {"corpus_seed": 6_203_100,
      "trace_lengths": (6, 7, 10, 12, 14, 6, 7, 10, 12, 14),
      "start_seeds": (6_203_021, 6_203_022), "start_scale": 1.5,
      "cond_seeds": (6_203_001, 6_203_002),
      "marg_seeds": (6_203_011, 6_203_012),
      "sweeps": 8_000, "burn_in": 2_000, "thin": 4,
      "gates": {"relation_marginal": "abs diff <= max(0.04, 4 * joint MCSE)",
                "boundary_marginal_abs": 0.05,
                "occurrence_marginal_abs": 0.05,
                "scalar_summaries": "abs diff <= 4 * joint MCSE (log target, "
                                    "relation-count mean)"}}

EQ_V2 = {
    "revision_rationale":
        "v1 FAILED by design defect, preserved in "
        "small_reference_equality_v1_FAILED.json: on the 10-trace tiny "
        "reference the posterior has near-equal-mass label-permutation modes; "
        "diagnostics showed the two C-MARG chains disagreed with EACH OTHER "
        "(max relation-marginal diff 0.98 — each occupying a different "
        "permutation mode) while the C-COND chains agreed (0.18) because "
        "conditional dynamics cannot cross modes. Pooled-marginal equality "
        "therefore measured inter-mode MIXING, which C-COND structurally "
        "lacks — the very phenomenon Condition C studies — rather than "
        "stationary-distribution equality, which is what the amendment "
        "requires. v2 tests invariance-equality: a more informative "
        "reference and one COMMON registered warm start inside a single "
        "basin, where two correct kernels must produce the same marginals. "
        "Registered before any v2 draw; gates unchanged in form.",
    "corpus_seed": 6_203_150,
    "trace_lengths": (24, 32) * 6,
    "warm_start": {"u_seed": 6_203_051, "u_scale": 1.5, "chain_seed": 6_203_050,
                   "sweeps": 500, "kernel": "C-MARG (every=10, scale=0.5)",
                   "note": "the warm phase only locates a basin; its draws are "
                           "discarded and no recovery quantity is inspected"},
    "cond_seeds": (6_203_061, 6_203_062),
    "marg_seeds": (6_203_071, 6_203_072),
    "sweeps": 6_000, "burn_in": 1_500, "thin": 3,
    "gates": {"relation_marginal": "abs diff <= max(0.04, 4 * joint MCSE)",
              "boundary_marginal_abs": 0.05,
              "occurrence_marginal_abs": 0.05,
              "scalar_summaries": "abs diff <= 4 * joint MCSE"},
}

PILOT = {"start_seed": 6_203_211, "start_scale": 1.5,
         "stage1": {"u_scale_grid": (0.25, 0.5, 1.0), "central": 0.5,
                    "seeds": (6_203_201, 6_203_202, 6_203_203),
                    "sweeps": 300, "every": 0},
         "stage2": {"collapsed_scale_grid": (0.25, 0.5, 1.0),
                    "seeds": (6_203_204, 6_203_205, 6_203_206),
                    "sweeps": 600, "every": CADENCE},
         "acceptance_band": (0.20, 0.60),
         "stage1_rule": "acceptance in band -> max per-proposal ESJD; else "
                        "closest to band, band never widened",
         "stage2_rule": "collapsed acceptance in band -> LARGEST admissible "
                        "scale (monotone ESJD proxy for a symmetric row walk); "
                        "else closest to band, band never widened",
         "may_inspect": ["acceptance", "ESJD", "H-change counts", "runtime",
                         "numerical health"],
         "must_not_inspect": ["truth recovery", "relation F1", "Hamming to H*",
                              "posterior truth coverage", "held-out "
                              "likelihood", "formal R-hat"]}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _dump(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def make_u_start(seed: int, scale: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(sigma_rho_matrix(2, RHO_0))
    return np.array([[scale * (chol @ rng.standard_normal(2))
                      for _ in range(5)] for _ in range(3)])


# ===================================================================== phase 1
def freeze_and_manifest() -> dict:
    truth = msg.supplied_truth()
    corpus = msg.generate_corpus(
        GENERATION_SEED, tuple((24, 32, 40, 48)[i % 4] for i in range(100)),
        tuple((24, 32, 40, 48)[i % 4] for i in range(45)), truth)
    recorded = json.loads((CORPUS_DIR / "corpus_hash.json").read_text())
    if msg.corpus_hash(corpus) != recorded["corpus_hash_sha256"]:
        raise SystemExit("formal corpus hash changed — refusing to run")
    truth_hash = msg.sha256_hex(msg.canonical_json(msg.truth_to_jsonable(truth)))
    if truth_hash != recorded["truth_hash_sha256"]:
        raise SystemExit("truth hash changed — refusing to run")
    for verdict_path, expected in (
            ("matched_condition_a/final_verdict.json",
             "PATH STRONGLY IDENTIFIABLE"),
            ("matched_condition_b/final_verdict.json",
             "STRUCTURE STRONGLY IDENTIFIABLE UNDER ORACLE PATHS")):
        payload = json.loads((ROOT / "results/mcmc_original"
                              / verdict_path).read_text())
        observed = payload.get("verdict", payload.get("classification"))
        if expected not in observed:
            raise SystemExit(f"{verdict_path}: expected '{expected}', "
                             f"found '{observed}'")

    _dump("prelaunch_registration.json", {
        "phase": "CONDITION C PRE-LAUNCH ONLY — no formal chain is launched",
        "target": "p(S, z, U | X, vartheta*, pi*, P*, delta_B*, epsilon*, "
                  "rho_0) with rho_0 = 0.5 fixed (Condition-B correction "
                  "carries over: rho* = null)",
        "inferred": ["S", "z", "U"],
        "fixed": {"beta": 1.5, "omega": 1.7346, "lambda_rep": 0.8,
                  "lambda_back": 0.25, "pi": "pi*", "P": "P*",
                  "delta_B": 0.15, "epsilon": 0.02, "rho": RHO_0},
        "arms": {"C-COND": "conditional-only: exact FFBS (S,z) refresh + "
                           "conditional U rows (collapsed cadence every = 0)",
                 "C-MARG": "partially collapsed: scheduled verbatim "
                           "collapsed_u_mh_step -> immediate exact FFBS "
                           "refresh -> conditional U rows (every = "
                           f"{CADENCE})"},
        "cadence": CADENCE,
        "cadence_provenance": "fixed by the pre-launch amendment; the "
                              "validated Step 8 kernel default, not tuned",
        "corpus_hash_sha256": recorded["corpus_hash_sha256"],
        "truth_hash_sha256": recorded["truth_hash_sha256"],
        "parent_commits": {"condition_a": "b199374", "condition_b": "34873d8",
                           "collapsed_u_validation": "58f005e",
                           "integration_merge": _git("rev-parse", "HEAD")},
        "equality_test_v1": {k: list(v) if isinstance(v, tuple) else v
                             for k, v in EQ.items()},
        "equality_test_v2": {k: ({kk: (list(vv) if isinstance(vv, tuple)
                                       else vv) for kk, vv in v.items()}
                                 if isinstance(v, dict)
                                 else (list(v) if isinstance(v, tuple) else v))
                             for k, v in EQ_V2.items()},
        "pilot": {k: ({kk: (list(vv) if isinstance(vv, tuple) else vv)
                       for kk, vv in v.items()} if isinstance(v, dict)
                      else (list(v) if isinstance(v, tuple) else v))
                  for k, v in PILOT.items()},
        "stop_condition": "STOP after equality test, pilot and scale freeze; "
                          "formal Condition-C chains are NOT launched in this "
                          "task",
    })
    _dump("integration_manifest.json", {
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "merge_commit": _git("rev-parse", "HEAD"),
        "parents": _git("log", "-1", "--format=%P").split(),
        "condition_b_parent": "34873d8c1aa157978d4292f01b45642976b61ff3",
        "collapsed_u_validation_commit": "58f005e (Step 8: partially-collapsed "
                                         "U kernel — implemented, diagnosed, "
                                         "VALIDATED)",
        "conflict_resolutions": {
            "src/hpop/mcmc_original/stage6e_sampler.py":
                "taken from 58f005e — the byte version the collapsed-U kernel "
                "was validated against; the two sides differ only in one "
                "comment paragraph",
            "tests/mcmc_original/test_stage6e_pipeline.py":
                "taken from 34873d8 — a strict superset (adds the zero-"
                "proposal no-op test); dropping a test would weaken the suite",
            "scripts/stage6e2_formal_chains.py":
                "taken from 34873d8 — the Stage 6E2 TERMINAL ladder version "
                "with continuation support; not imported by any collapsed-U "
                "module"},
        "collapsed_u_reimplemented": False,
    })
    _dump("source_manifest.json", {
        "commit": _git("rev-parse", "HEAD"),
        "hashes": {
            "matched_generator":
                _sha("src/hpop/mcmc_original/matched_synthetic_generator.py"),
            "formal_corpus_loader":
                _sha("scripts/generate_matched_formal_corpus.py"),
            "semi_markov_ffbs":
                _sha("src/hpop/mcmc_original/semi_markov_ffbs.py"),
            "ordinary_u_row_kernel":
                _sha("src/hpop/mcmc_original/sampler_u.py"),
            "path_marginal_likelihood":
                _sha("src/hpop/mcmc_original/collapsed_u_likelihood.py"),
            "collapsed_u_kernel":
                _sha("src/hpop/mcmc_original/collapsed_u_kernel.py"),
            "recurrent_block_scorer":
                _sha("src/hpop/mcmc_original/recurrent_segmentation.py"),
            "ffbs_joint_sampler":
                _sha("src/hpop/mcmc_original/recurrent_joint_ffbs_mcmc.py"),
            "fast_block_tables":
                _sha("src/hpop/mcmc_original/fast_block_tables.py"),
            "condition_c_composition":
                _sha("src/hpop/mcmc_original/matched_condition_c.py"),
        }})
    return {"truth": truth}


# =============================================================== equality test
def _pooled_marginals(runs, model, n_skills: int):
    indicators = np.concatenate(
        [np.asarray(r["retained"]["relation_indicators"], dtype=float)
         for r in runs])
    relation_marginals = indicators.mean(axis=0)
    boundary, occupancy, totals = {}, {}, 0
    for r in runs:
        for keys in r["retained"]["keys"]:
            totals += 1
            for n, key in enumerate(keys):
                J = len(model.traces[n])
                b = boundary.setdefault(n, np.zeros(J - 1))
                o = occupancy.setdefault(n, np.zeros((J, n_skills)))
                start = 0
                for end, skill in key:
                    if end < J:
                        b[end - 1] += 1
                    o[start:end, skill] += 1
                    start = end
    for n in boundary:
        boundary[n] /= totals
        occupancy[n] /= totals
    return relation_marginals, boundary, occupancy, indicators


def run_equality_test_v2(truth) -> dict:
    """v2: invariance-equality from one COMMON registered warm start."""
    corpus = msg.generate_corpus(EQ_V2["corpus_seed"], EQ_V2["trace_lengths"],
                                 (), truth)
    model = mcc.build_condition_c_model(tuple(t.cpa for t in corpus.train))
    fixed = mcc.ConditionCFixed.from_truth(truth, RHO_0)

    warm_cfg = EQ_V2["warm_start"]
    print("  warm phase (draws discarded)...")
    warm = mcc.run_condition_c_chain(
        model, fixed, make_u_start(warm_cfg["u_seed"], warm_cfg["u_scale"]),
        COLLAPSED_SCALE_DEFAULT,
        CollapsedUConfig(every=CADENCE, scale=COLLAPSED_SCALE_DEFAULT),
        num_sweeps=warm_cfg["sweeps"], burn_in=warm_cfg["sweeps"] - 1, thin=1,
        seed=warm_cfg["chain_seed"])
    shared_u = np.array(warm["final_state"].u_by_skill, dtype=float, copy=True)

    def run_arm(every: int, seeds) -> list:
        return [mcc.run_condition_c_chain(
            model, fixed, shared_u, COLLAPSED_SCALE_DEFAULT,
            CollapsedUConfig(every=every, scale=COLLAPSED_SCALE_DEFAULT),
            num_sweeps=EQ_V2["sweeps"], burn_in=EQ_V2["burn_in"],
            thin=EQ_V2["thin"], seed=seeds[c], store_keys=True)
            for c in range(2)]

    print("  running C-COND arm (every=0)...")
    cond = run_arm(0, EQ_V2["cond_seeds"])
    print("  running C-MARG arm (every=10)...")
    marg = run_arm(CADENCE, EQ_V2["marg_seeds"])
    out = _compare_arms(cond, marg, model, EQ_V2["gates"])
    out["version"] = "v2 (common warm start; see revision_rationale in "
    out["version"] += "prelaunch registration)"
    out["corpus_seed"] = EQ_V2["corpus_seed"]
    out["n_traces"] = len(corpus.train)
    return out


def _compare_arms(cond, marg, model, gates) -> dict:
    rel_c, bnd_c, occ_c, ind_c = _pooled_marginals(cond, model, 3)
    rel_m, bnd_m, occ_m, ind_m = _pooled_marginals(marg, model, 3)

    def scalar_mcse(runs, name):
        chains = np.asarray([r["retained"][name] for r in runs], dtype=float)
        ess = bulk_ess(chains)
        return float(chains.mean()), float(chains.std(ddof=1)
                                           / np.sqrt(max(ess, 1.0))), ess

    def indicator_joint_gate(i):
        halves = []
        for ind, runs in ((ind_c, cond), (ind_m, marg)):
            per_chain = np.asarray(
                [np.asarray(r["retained"]["relation_indicators"],
                            dtype=float)[:, i] for r in runs])
            if np.all(per_chain == per_chain.flat[0]):
                halves.append(0.0)
                continue
            ess = max(bulk_ess(per_chain), 8.0)
            p = float(per_chain.mean())
            halves.append(np.sqrt(max(p * (1 - p), 1e-12) / ess))
        return max(0.04, 4.0 * float(np.hypot(*halves)))

    rel_diffs = np.abs(rel_c - rel_m)
    rel_gates = np.array([indicator_joint_gate(i)
                          for i in range(len(rel_diffs))])
    rel_pass = bool(np.all(rel_diffs <= rel_gates))

    bnd_worst = max(float(np.abs(bnd_c[n] - bnd_m[n]).max())
                    for n in bnd_c)
    occ_worst = max(float(np.abs(occ_c[n] - occ_m[n]).max())
                    for n in occ_c)

    lt_c, se_c, ess_c = scalar_mcse(cond, "log_target")
    lt_m, se_m, ess_m = scalar_mcse(marg, "log_target")
    lt_gate = 4.0 * float(np.hypot(se_c, se_m))
    rc_c, rse_c, _ = scalar_mcse(cond, "relation_counts")
    rc_m, rse_m, _ = scalar_mcse(marg, "relation_counts")
    rc_gate = 4.0 * float(np.hypot(rse_c, rse_m))

    out = {
        "retained_per_arm": int(ind_c.shape[0]),
        "relation_marginals": {
            "max_abs_diff": float(rel_diffs.max()),
            "max_gate": float(rel_gates.max()),
            "n_exceeding_gate": int(np.sum(rel_diffs > rel_gates)),
            "pass": rel_pass},
        "boundary_marginals": {"max_abs_diff": bnd_worst,
                               "gate": gates["boundary_marginal_abs"],
                               "pass": bool(bnd_worst
                                            <= gates["boundary_marginal_abs"])},
        "occurrence_marginals": {"max_abs_diff": occ_worst,
                                 "gate": gates["occurrence_marginal_abs"],
                                 "pass": bool(occ_worst
                                              <= gates
                                              ["occurrence_marginal_abs"])},
        "log_target": {"cond_mean": lt_c, "marg_mean": lt_m,
                       "abs_diff": abs(lt_c - lt_m), "gate": lt_gate,
                       "ess": [ess_c, ess_m],
                       "pass": bool(abs(lt_c - lt_m) <= lt_gate)},
        "relation_count_mean": {"cond": rc_c, "marg": rc_m,
                                "abs_diff": abs(rc_c - rc_m), "gate": rc_gate,
                                "pass": bool(abs(rc_c - rc_m) <= rc_gate)},
        "collapsed_moves_in_marg_arm": {
            "proposed": sum(r["collapsed_proposed"] for r in marg),
            "accepted": sum(r["collapsed_accepted"] for r in marg),
            "h_accepted": sum(r["collapsed_h_accepted"] for r in marg)},
        "seconds": {"cond": sum(r["seconds"] for r in cond),
                    "marg": sum(r["seconds"] for r in marg)},
    }
    out["pass"] = bool(rel_pass and out["boundary_marginals"]["pass"]
                       and out["occurrence_marginals"]["pass"]
                       and out["log_target"]["pass"]
                       and out["relation_count_mean"]["pass"])
    print(f"  equality: rel diff {out['relation_marginals']['max_abs_diff']:.4f}"
          f" bnd {bnd_worst:.4f} occ {occ_worst:.4f} "
          f"dLT {abs(lt_c - lt_m):.3f} (gate {lt_gate:.3f}) -> "
          f"{'PASS' if out['pass'] else 'FAIL'}")
    return out


# ======================================================================= pilot
def run_pilot(truth) -> tuple:
    corpus = msg.generate_corpus(
        GENERATION_SEED, tuple((24, 32, 40, 48)[i % 4] for i in range(100)),
        tuple((24, 32, 40, 48)[i % 4] for i in range(45)), truth)
    model = mcc.build_condition_c_model(tuple(t.cpa for t in corpus.train))
    fixed = mcc.ConditionCFixed.from_truth(truth, RHO_0)
    start = make_u_start(PILOT["start_seed"], PILOT["start_scale"])
    lo, hi = PILOT["acceptance_band"]

    stage1 = []
    for scale, seed in zip(PILOT["stage1"]["u_scale_grid"],
                           PILOT["stage1"]["seeds"]):
        r = mcc.run_condition_c_chain(
            model, fixed, start, scale, CollapsedUConfig(every=0, scale=0.5),
            num_sweeps=PILOT["stage1"]["sweeps"], burn_in=1, thin=1, seed=seed)
        acceptance = r["movement"]["u_accepted"] / r["movement"]["u_proposed"]
        stage1.append({"u_scale": scale, "acceptance": acceptance,
                       "esjd_per_proposal": r["esjd_per_sweep"] / 15.0,
                       "u_h_accepted": r["movement"]["u_h_accepted"],
                       "seconds_per_sweep": r["seconds_per_sweep"],
                       "finite": bool(np.isfinite(
                           r["retained"]["log_target"]).all())})
        print(f"  stage1 u_scale={scale}: acc={acceptance:.3f} "
              f"ESJD={stage1[-1]['esjd_per_proposal']:.4f} "
              f"{r['seconds_per_sweep']:.2f}s/sweep")
    admissible = [r for r in stage1 if lo <= r["acceptance"] <= hi
                  and r["finite"]]
    if admissible:
        chosen1 = max(admissible, key=lambda r: r["esjd_per_proposal"])
        rule1 = "acceptance in band, max ESJD"
    else:
        chosen1 = min(stage1, key=lambda r: min(abs(r["acceptance"] - lo),
                                                abs(r["acceptance"] - hi)))
        rule1 = "closest to band"

    stage2 = []
    for scale, seed in zip(PILOT["stage2"]["collapsed_scale_grid"],
                           PILOT["stage2"]["seeds"]):
        r = mcc.run_condition_c_chain(
            model, fixed, start, chosen1["u_scale"],
            CollapsedUConfig(every=CADENCE, scale=scale),
            num_sweeps=PILOT["stage2"]["sweeps"], burn_in=1, thin=1, seed=seed)
        n_prop = max(r["collapsed_proposed"], 1)
        acceptance = r["collapsed_accepted"] / n_prop
        stage2.append({"collapsed_scale": scale,
                       "collapsed_proposed": r["collapsed_proposed"],
                       "collapsed_acceptance": acceptance,
                       "collapsed_h_accepted": r["collapsed_h_accepted"],
                       "seconds_per_sweep": r["seconds_per_sweep"],
                       "finite": bool(np.isfinite(
                           r["retained"]["log_target"]).all())})
        print(f"  stage2 collapsed_scale={scale}: acc={acceptance:.3f} "
              f"H-acc={r['collapsed_h_accepted']}/{r['collapsed_proposed']} "
              f"{r['seconds_per_sweep']:.2f}s/sweep")
    admissible2 = [r for r in stage2
                   if lo <= r["collapsed_acceptance"] <= hi and r["finite"]]
    if admissible2:
        chosen2 = max(admissible2, key=lambda r: r["collapsed_scale"])
        rule2 = "collapsed acceptance in band, largest admissible scale"
    else:
        chosen2 = min(stage2, key=lambda r: min(
            abs(r["collapsed_acceptance"] - lo),
            abs(r["collapsed_acceptance"] - hi)))
        rule2 = "closest to band"

    selected = {
        "u_scale": chosen1["u_scale"],
        "u_scale_rule": rule1,
        "u_scale_acceptance": chosen1["acceptance"],
        "u_scale_esjd": chosen1["esjd_per_proposal"],
        "scheduled_collapsed_scale": chosen2["collapsed_scale"],
        "scheduled_rule": rule2,
        "scheduled_acceptance": chosen2["collapsed_acceptance"],
        "cadence": CADENCE,
        "common_to_both_arms": "u_scale is shared; the scheduled proposal "
                               "(scale + cadence 10) is registered once and "
                               "applies to C-MARG (C-COND runs every = 0)",
        "all_pilot_draws_discarded": True,
    }
    return {"stage1": stage1, "stage2": stage2}, selected


def main() -> int:
    wall0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    print("== phase 1: freeze, manifests, hash verification ==")
    env = freeze_and_manifest()
    _dump("pilot_registration.json", json.loads(
        (OUT / "prelaunch_registration.json").read_text())["pilot"])

    # preserve the v1 failure record before the registered v2 runs
    v1_path = OUT / "small_reference_equality.json"
    if v1_path.exists():
        payload = json.loads(v1_path.read_text())
        if not payload.get("pass", False) and "version" not in payload:
            v1_path.rename(OUT / "small_reference_equality_v1_FAILED.json")

    print("== phase 2: small-reference equality test v2 (C-COND vs C-MARG) ==")
    equality = run_equality_test_v2(env["truth"])
    _dump("small_reference_equality.json", equality)
    if not equality["pass"]:
        print("EQUALITY TEST FAILED — STOP (no pilot, no formal chains)")
        return 1

    print("== phase 3: efficiency-only pilot on the formal corpus ==")
    pilot_results, selected = run_pilot(env["truth"])
    _dump("pilot_results.json", {**pilot_results,
                                 "recovery_inspected": False})
    _dump("selected_scales.json", selected)

    (OUT / "prelaunch_report.md").write_text("\n".join([
        "# Condition C — pre-launch record",
        "",
        f"Integration commit `{_git('rev-parse', 'HEAD')}` on branch "
        f"`{_git('rev-parse', '--abbrev-ref', 'HEAD')}` (merge of Condition B "
        "`34873d8` and the collapsed-U validation `58f005e`).",
        "",
        "## Small-reference equality (C-COND vs C-MARG, same posterior)",
        f"- relation-marginal max |diff| "
        f"{equality['relation_marginals']['max_abs_diff']:.4f} "
        f"(0 of 60 exceed their MCSE gates)"
        if equality['relation_marginals']['n_exceeding_gate'] == 0 else
        f"- relation marginals: "
        f"{equality['relation_marginals']['n_exceeding_gate']} exceed gates",
        f"- boundary max |diff| "
        f"{equality['boundary_marginals']['max_abs_diff']:.4f} (gate 0.05); "
        f"occurrence max |diff| "
        f"{equality['occurrence_marginals']['max_abs_diff']:.4f} (gate 0.05)",
        f"- mean log-target diff {equality['log_target']['abs_diff']:.3f} "
        f"(gate {equality['log_target']['gate']:.3f})",
        f"- verdict: {'PASS' if equality['pass'] else 'FAIL'}",
        "",
        "## Frozen scales (efficiency-only pilot; recovery never inspected)",
        f"- conditional U row scale: {selected['u_scale']} "
        f"(acceptance {selected['u_scale_acceptance']:.3f})",
        f"- scheduled collapsed scale: "
        f"{selected['scheduled_collapsed_scale']} "
        f"(acceptance {selected['scheduled_acceptance']:.3f})",
        f"- cadence: c = {CADENCE} (fixed by amendment, not tuned)",
        "",
        "Conditions A and B, the formal corpus, the smoke corpus and the "
        "generator validation are unchanged. NO formal Condition-C chain was "
        "launched.",
    ]) + "\n")
    print(f"\nPRE-LAUNCH COMPLETE in {time.perf_counter() - wall0:.0f}s — "
          "formal chains NOT launched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
