"""C1 expanded collapsed-U audit — does EVERY frozen structural mode have an escape tail?

    cd hpop-step7 && PYTHONPATH=src python scripts/collapsed_u_expanded_audit.py [--probe]

The C0 fast audit (results/mcmc_original/collapsed_u_fast_audit/) established on 2 chains
that FFBS-collapsing (S, z) shrinks the cross-H barrier (median +215 nats) and that the
collapsed acceptance mass is TAIL-driven. This expansion scores 300 cross-H proposals from
the exact production U kernel at ALL EIGHT frozen chain states (4 Stage 6E2 Local, 4 Step
7B2 FFBS) and asks, per chain, whether a non-negligible escape tail exists.

Pre-registered per-chain tail criteria (written to config.json before scoring):

    P(alpha_coll > 0.01 | H' != H) >= 0.01   AND   E_escape_50k >= 10

Pre-registered overall classification by the number of chains satisfying BOTH:

    8       -> COLLAPSED-U ROBUST ACROSS MODES
    6-7     -> COLLAPSED-U MOSTLY ROBUST
    3-5     -> COLLAPSED-U MODE-DEPENDENT
    <=2     -> COLLAPSED-U NOT ROBUST

Checkpoint selection rule, registered before any score is inspected: for every chain, the
latest complete checkpoint file present at audit start (within each experiment all four
chains sit at the same ladder position: 6E2 final 150k, 7B2 latest in-flight block).
Checkpoints are COPIED before being read; a file caught mid-write fails its JSON parse
and is re-copied. No formal chain is written, evolved, or steered. Diagnostic only:
no production sampler, no new MCMC, no corpus regeneration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fast = _load("collapsed_u_fast_audit", ROOT / "scripts" / "collapsed_u_fast_audit.py")

from hpop.mcmc_original.block_score_adapters import (            # noqa: E402
    assert_no_recurrent_state_leak, build_log_block_scores,
)
from hpop.mcmc_original.fast_block_tables import FastBlockScoreTable   # noqa: E402
from hpop.mcmc_original.latent_poset import precedence_from_u          # noqa: E402
from hpop.mcmc_original.recurrent_segmentation import log_target_stage6e  # noqa: E402
from hpop.mcmc_original.sampler_u import propose_row                   # noqa: E402
from hpop.mcmc_original.semi_markov_ffbs import forward                # noqa: E402
from hpop.mcmc_original.stage6c_frozen import log_structural_prior     # noqa: E402
from hpop.mcmc_original.stage6e_corpus import corpus_hash, generate_corpus  # noqa: E402
from hpop.mcmc_original.stage6e_sampler import Stage6ESampler          # noqa: E402
from hpop.mcmc_original.stage6e_state import Stage6EState              # noqa: E402
from hpop.mcmc_original.transitions import log_transition_matrix       # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "collapsed_u_expanded_audit"
U_SCALE = 0.5
AUDIT_SEED = 8_152_000
N_CROSS = 300
N_SAME_H = 8
N_REBUILD_PARITY = 2
N_ADAPTER_PARITY_TRACES = 5        # full-corpus parity was already done in C0
N_RERUN = 5                        # deterministic-rerun subset per chain
M_U_PER_SWEEP = 15                 # verified: sweep_once step 3 loops K=3 skills x m=5 rows
SWEEPS = 50_000
PARITY_TOL = 1e-10
TAIL_P_THRESHOLD = 0.01            # pre-registered: P(alpha > 0.01 | cross) >= 0.01
TAIL_E_THRESHOLD = 10.0            # pre-registered: E_escape_50k >= 10
FORMAL_SWEEP_SECONDS_7B2 = 60_500 / 44_000   # observed: resume_run.log, 44k sweeps

STEP7_CKPT = ROOT / "results" / "mcmc_original" / "stage7b2_full_joint_ffbs" / "checkpoints"
E2_CKPT = Path("/Users/dongqing/Desktop/hpop-stage6e/results/mcmc_original"
               "/stage6e2_unknown_boundary_full_seed0/unknown_checkpoints")

SOURCES = [(exp, chain,
            (STEP7_CKPT if exp == "7b2_ffbs" else E2_CKPT)
            / f"chain{chain}_checkpoint.json")
           for exp in ("7b2_ffbs", "6e2_local") for chain in range(4)]


# ------------------------------------------------------------------ hashing helpers
def h_matrix_bytes(h: np.ndarray) -> bytes:
    return np.packbits(np.asarray(h, dtype=bool).astype(np.uint8)).tobytes()


def state_h_hash(h_list) -> str:
    """Exact skill-indexed induced-order hash (NOT permutation invariant)."""
    return hashlib.sha256(b"".join(h_matrix_bytes(h) for h in h_list)).hexdigest()[:16]


def canonical_h_hash(h_list) -> str:
    """Skill-relabelling-invariant hash: the sorted multiset of per-skill orders.

    Exact skill-indexed matching is NOT meaningful across chains because the target is
    invariant under joint relabelling of (skills, pi, P, U); the registered convergence
    summaries are permutation invariant for the same reason. This canonicalisation
    compares the multiset {H_k} — invariant under any skill permutation.
    """
    return hashlib.sha256(b"".join(sorted(h_matrix_bytes(h)
                                          for h in h_list))).hexdigest()[:16]


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ------------------------------------------------------------------ proposal stream
def score_stream(state: Stage6EState, model, table: FastBlockScoreTable,
                 skill_ll, cond_ll, prior_now, h_now, log_pi, log_p, seed: int,
                 n_cross: int, n_same_h: int, do_parity: bool,
                 shortest: int) -> dict:
    """Score proposals from the frozen state until both quotas fill. Never updates state.

    The RNG is consumed once per raw draw regardless of quota state, so two calls with the
    same seed produce identical scored prefixes — which is what the deterministic-rerun
    control checks.
    """
    K, m, _ = np.asarray(state.u_by_skill).shape
    u = np.array(state.u_by_skill, dtype=float)
    rng = np.random.default_rng(seed)

    cross, same_h, rebuild_parity = [], [], []
    enum_error_proposed = None
    hastings_max = 0.0
    n_raw = n_invalid = n_cross_seen = n_seen = 0

    while len(cross) < n_cross or len(same_h) < n_same_h:
        k = (n_seen // m) % K
        row = n_seen % m
        n_seen += 1
        n_raw += 1
        candidate_k = propose_row(u[k], row, U_SCALE, rng)
        cand_prior = log_structural_prior(candidate_k, state.rho)
        if not math.isfinite(cand_prior):
            n_invalid += 1
            continue
        h_new = precedence_from_u(candidate_k)
        h_changed = not np.array_equal(h_new, h_now[k])
        n_cross_seen += int(h_changed)
        need_cross = h_changed and len(cross) < n_cross
        need_same = (not h_changed) and len(same_h) < n_same_h
        if not (need_cross or need_same):
            continue

        d_prior = cand_prior - prior_now[k]
        cand_cond = skill_ll.full_replay(k, candidate_k, state.beta, state.omega,
                                         state.lambda_rep, state.lambda_back)
        d_ll_cond = cand_cond - cond_ll[k]

        step = candidate_k[row] - u[k][row]
        hastings_max = max(hastings_max, abs(
            fast.gaussian_row_log_density(-step, U_SCALE)
            - fast.gaussian_row_log_density(step, U_SCALE)))

        u_prime = np.array(u, copy=True)
        u_prime[k] = candidate_k
        began = time.perf_counter()
        info = table.refresh(u_prime, state.beta, state.omega, state.lambda_rep,
                             state.lambda_back)
        if info["rebuilt_skills"] != [k]:
            raise AssertionError(f"incremental refresh rebuilt {info['rebuilt_skills']}")
        prop_log_z = fast.collapsed_log_z(table, model, log_pi, log_p)
        eval_seconds = time.perf_counter() - began

        if do_parity and need_cross and len(rebuild_parity) < N_REBUILD_PARITY:
            fresh = FastBlockScoreTable(traces=model.traces, epsilon=model.epsilon,
                                        n_skills=K, min_width=model.min_width,
                                        max_width=model.max_width,
                                        n_roles=model.n_roles)
            fresh.refresh(u_prime, state.beta, state.omega, state.lambda_rep,
                          state.lambda_back)
            worst_block = max(
                float(np.abs(a[np.isfinite(a)] - b[np.isfinite(b)]).max())
                for a, b in zip(fresh.tables, table.tables))
            worst_z = float(np.abs(fast.collapsed_log_z(fresh, model, log_pi, log_p)
                                   - prop_log_z).max())
            if worst_block > PARITY_TOL or worst_z > PARITY_TOL:
                raise AssertionError(
                    f"rebuild parity failed: {worst_block:.2e}/{worst_z:.2e}")
            rebuild_parity.append({"skill": int(k), "row": int(row),
                                   "max_block_score_diff": worst_block,
                                   "max_log_z_diff": worst_z})
            if enum_error_proposed is None:
                enum = fast.enumeration_log_z(table.tables[shortest], log_pi, log_p,
                                              model.delta_b, model.min_width,
                                              model.max_width)
                enum_error_proposed = abs(enum - prop_log_z[shortest])
                if enum_error_proposed > PARITY_TOL:
                    raise AssertionError(f"enum parity at U' failed: "
                                         f"{enum_error_proposed:.2e}")

        d_ll_coll = float((prop_log_z - score_stream.base_log_z).sum())
        table.refresh(u, state.beta, state.omega, state.lambda_rep, state.lambda_back)

        proposed_h = [h_new if j == k else h_now[j] for j in range(K)]
        record = {
            "skill": int(k), "row": int(row), "h_changed": bool(h_changed),
            "d_h": int((h_new != h_now[k]).sum()),
            "relations_current": int(h_now[k].sum()),
            "relations_proposed": int(h_new.sum()),
            "proposed_state_hash": state_h_hash(proposed_h),
            "proposed_canonical_hash": canonical_h_hash(proposed_h),
            "d_log_prior": float(d_prior),
            "d_log_lik_conditional": float(d_ll_cond),
            "d_log_lik_collapsed": float(d_ll_coll),
            "log_alpha_conditional": float(d_ll_cond + d_prior),
            "log_alpha_collapsed": float(d_ll_coll + d_prior),
            "p_accept_conditional": min(1.0, math.exp(min(0.0, d_ll_cond + d_prior))),
            "p_accept_collapsed": min(1.0, math.exp(min(0.0, d_ll_coll + d_prior))),
            "eval_seconds": float(eval_seconds),
        }
        (cross if h_changed else same_h).append(record)

    return {"cross": cross, "same_h": same_h, "n_raw": n_raw, "n_invalid": n_invalid,
            "n_cross_seen": n_cross_seen, "hastings_max": hastings_max,
            "rebuild_parity": rebuild_parity,
            "enum_error_proposed": enum_error_proposed}


def audit_chain(exp: str, chain: int, state: Stage6EState, model, seed: int,
                n_cross: int, checkpoint_log_target: float | None) -> dict:
    K, m, _ = np.asarray(state.u_by_skill).shape
    u = np.array(state.u_by_skill, dtype=float)
    log_pi = np.log(np.asarray(state.pi, dtype=float))
    log_p = log_transition_matrix(state.transition)

    # integrity gate: the recomputed target must match the checkpointed value
    parts = log_target_stage6e(state, model)
    integrity = {"recomputed_log_target": float(parts["log_target"]),
                 "checkpointed_log_target": checkpoint_log_target}
    if checkpoint_log_target is not None:
        drift = abs(parts["log_target"] - checkpoint_log_target)
        integrity["abs_diff"] = float(drift)
        if drift > 1e-6:
            raise AssertionError(f"{exp} chain {chain}: log target integrity "
                                 f"failed ({drift:.3e})")

    sampler = Stage6ESampler(model=model, scales={"U": U_SCALE},
                             n_proposals_per_trace=0)
    sampler.prepare(state)
    skill_ll = sampler._skill
    skill_ll.set_blocks(state.segmentations, model.n_skills)
    cond_ll = {k: skill_ll.full_replay(k, u[k], state.beta, state.omega,
                                       state.lambda_rep, state.lambda_back)
               for k in range(K)}
    prior_now = {k: log_structural_prior(u[k], state.rho) for k in range(K)}
    h_now = {k: precedence_from_u(u[k]) for k in range(K)}

    table = FastBlockScoreTable(traces=model.traces, epsilon=model.epsilon,
                                n_skills=K, min_width=model.min_width,
                                max_width=model.max_width, n_roles=model.n_roles)
    table.refresh(u, state.beta, state.omega, state.lambda_rep, state.lambda_back)
    base_log_z = fast.collapsed_log_z(table, model, log_pi, log_p)
    score_stream.base_log_z = base_log_z

    # adapter parity on a few traces (full-corpus parity was pinned in C0)
    scorer = model.scorer_for(state)
    q0 = assert_no_recurrent_state_leak(
        scorer, 0, (0, model.min_width, 0), (model.min_width, 2 * model.min_width, 1))
    adapter_worst = 0.0
    for n in range(N_ADAPTER_PARITY_TRACES):
        adapter = build_log_block_scores(scorer, n, len(model.traces[n]), K,
                                         model.min_width, model.max_width)
        finite = np.isfinite(adapter)
        if (finite != np.isfinite(table.tables[n])).any():
            raise AssertionError("adapter/fast support mismatch")
        adapter_worst = max(adapter_worst, float(
            np.abs(adapter[finite] - table.tables[n][finite]).max()))
        z = forward(adapter, log_pi, log_p, model.delta_b, model.max_width,
                    model.min_width).log_normalizer
        adapter_worst = max(adapter_worst, abs(z - base_log_z[n]))
    if adapter_worst > PARITY_TOL:
        raise AssertionError(f"adapter parity failed: {adapter_worst:.2e}")

    shortest = int(np.argmin([len(t) for t in model.traces]))
    enum_at_u = abs(fast.enumeration_log_z(table.tables[shortest], log_pi, log_p,
                                           model.delta_b, model.min_width,
                                           model.max_width) - base_log_z[shortest])
    if enum_at_u > PARITY_TOL:
        raise AssertionError(f"enum parity at U failed: {enum_at_u:.2e}")

    stream = score_stream(state, model, table, skill_ll, cond_ll, prior_now, h_now,
                          log_pi, log_p, seed, n_cross, N_SAME_H, do_parity=True,
                          shortest=shortest)

    # same-H negative controls
    same_cond = np.array([r["d_log_lik_conditional"] for r in stream["same_h"]])
    same_coll = np.array([r["d_log_lik_collapsed"] for r in stream["same_h"]])
    if same_cond.size and float(np.abs(same_cond).max()) > 1e-9:
        raise AssertionError("same-H conditional leak")
    if same_coll.size and float(np.abs(same_coll).max()) > 1e-9:
        raise AssertionError("same-H collapsed leak")

    # deterministic rerun of a small scored prefix, from a fresh table and fresh RNG
    fresh_table = FastBlockScoreTable(traces=model.traces, epsilon=model.epsilon,
                                      n_skills=K, min_width=model.min_width,
                                      max_width=model.max_width, n_roles=model.n_roles)
    fresh_table.refresh(u, state.beta, state.omega, state.lambda_rep, state.lambda_back)
    rerun = score_stream(state, model, fresh_table, skill_ll, cond_ll, prior_now, h_now,
                         log_pi, log_p, seed, N_RERUN - 1, 1, do_parity=False,
                         shortest=shortest)
    compare_fields = ("skill", "row", "d_h", "d_log_prior", "d_log_lik_conditional",
                      "d_log_lik_collapsed", "log_alpha_collapsed")
    rerun_identical = all(
        first[f] == again[f]
        for first, again in list(zip(stream["cross"], rerun["cross"]))
        + list(zip(stream["same_h"], rerun["same_h"]))
        for f in compare_fields)
    if not rerun_identical:
        raise AssertionError(f"{exp} chain {chain}: deterministic rerun differs")

    return {
        "experiment": exp, "chain": chain,
        "current_state_hash": state_h_hash([h_now[k] for k in range(K)]),
        "current_canonical_hash": canonical_h_hash([h_now[k] for k in range(K)]),
        "current_relations_per_skill": [int(h_now[k].sum()) for k in range(K)],
        "integrity": integrity,
        "cross": stream["cross"], "same_h": stream["same_h"],
        "n_raw": stream["n_raw"], "n_invalid": stream["n_invalid"],
        "n_cross_seen": stream["n_cross_seen"],
        "r_cross": stream["n_cross_seen"] / max(1, stream["n_raw"]
                                                - stream["n_invalid"]),
        "checks": {
            "adapter_parity_max_diff": adapter_worst,
            "adapter_parity_traces": N_ADAPTER_PARITY_TRACES,
            "enum_error_at_U": float(enum_at_u),
            "enum_error_at_U_prime": stream["enum_error_proposed"],
            "q0_reset_bit_identical": bool(q0["pass"]),
            "hastings_max_abs": float(stream["hastings_max"]),
            "rebuild_parity": stream["rebuild_parity"],
            "same_h_max_abs_d_ll_conditional": float(np.abs(same_cond).max()),
            "same_h_max_abs_d_ll_collapsed": float(np.abs(same_coll).max()),
            "deterministic_rerun_identical": bool(rerun_identical),
            "rerun_prefix_length": N_RERUN,
        },
    }


# ------------------------------------------------------------------------ summaries
def chain_summary(result: dict) -> dict:
    cross = result["cross"]
    n = len(cross)
    cond = np.array([r["d_log_lik_conditional"] for r in cross])
    coll = np.array([r["d_log_lik_collapsed"] for r in cross])
    alpha = np.array([r["p_accept_collapsed"] for r in cross])
    log_alpha = np.array([r["log_alpha_collapsed"] for r in cross])
    n_1pct = int((alpha > 0.01).sum())
    n_10pct = int((alpha > 0.10).sum())
    n_pos = int((log_alpha >= 0.0).sum())
    mean_alpha = float(alpha.mean())
    r_cross = result["r_cross"]
    rate = M_U_PER_SWEEP * r_cross * mean_alpha
    summary = {
        "experiment": result["experiment"], "chain": result["chain"],
        "current_state_hash": result["current_state_hash"],
        "current_canonical_hash": result["current_canonical_hash"],
        "n_cross": n, "n_raw": result["n_raw"], "n_invalid": result["n_invalid"],
        "r_cross": r_cross,
        "median_d_ll_conditional": float(np.median(cond)),
        "median_d_ll_collapsed": float(np.median(coll)),
        "median_barrier_reduction": float(np.median(coll - cond)),
        "d_ll_conditional": fast.quantile_summary(cond),
        "d_ll_collapsed": fast.quantile_summary(coll),
        "barrier_reduction": fast.quantile_summary(coll - cond),
        "p_alpha_gt_1pct": n_1pct / n, "p_alpha_gt_1pct_wilson95": wilson(n_1pct, n),
        "p_alpha_gt_10pct": n_10pct / n, "p_alpha_gt_10pct_wilson95": wilson(n_10pct, n),
        "p_log_alpha_ge_0": n_pos / n, "p_log_alpha_ge_0_wilson95": wilson(n_pos, n),
        "mean_alpha_collapsed": mean_alpha,
        "median_alpha_collapsed": float(np.median(alpha)),
        "max_log_alpha_collapsed": float(log_alpha.max()),
        "expected_escapes_50k": SWEEPS * rate,
        "expected_sweeps_per_escape": (1.0 / rate) if rate > 0 else float("inf"),
        "tail_criterion_p": (n_1pct / n) >= TAIL_P_THRESHOLD,
        "tail_criterion_e": (SWEEPS * rate) >= TAIL_E_THRESHOLD,
    }
    summary["useful_tail"] = bool(summary["tail_criterion_p"]
                                  and summary["tail_criterion_e"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true",
                        help="2 chains, 10 cross each, to validate wiring and cost")
    args = parser.parse_args()
    n_cross = 10 if args.probe else N_CROSS
    sources = SOURCES[:1] + SOURCES[4:5] if args.probe else SOURCES

    OUT.mkdir(parents=True, exist_ok=True)
    copies = OUT / "checkpoint_copies"
    copies.mkdir(exist_ok=True)

    config = {
        "question": "does every frozen structural mode have a non-negligible "
                    "collapsed-U escape tail, or was C0 driven by favourable states?",
        "u_scale": U_SCALE, "audit_seed": AUDIT_SEED, "n_cross_per_chain": N_CROSS,
        "n_same_h_controls": N_SAME_H, "m_u_per_sweep": M_U_PER_SWEEP,
        "sweeps": SWEEPS,
        "checkpoint_rule_preregistered": "latest complete checkpoint per chain at audit "
                                        "start; 6E2 chains all at the final 150k rung, "
                                        "7B2 chains at the latest in-flight block; "
                                        "copies parsed before use, re-copied on a "
                                        "failed parse",
        "r_cross_source": "measured from this audit's raw proposal stream per chain",
        "tail_criteria_preregistered": {
            "per_chain": "P(alpha_coll > 0.01 | cross) >= 0.01 AND "
                         "E_escape_50k >= 10",
            "classification": {"8": "COLLAPSED-U ROBUST ACROSS MODES",
                               "6-7": "COLLAPSED-U MOSTLY ROBUST",
                               "3-5": "COLLAPSED-U MODE-DEPENDENT",
                               "<=2": "COLLAPSED-U NOT ROBUST"}},
        "parity_tolerance": PARITY_TOL,
        "proposal_selection": "first N cross-H proposals in deterministic (skill,row) "
                              "cycle order under the registered per-chain seed; never "
                              "filtered on acceptance, distance, or proposed order",
        "seeds_per_chain": {f"{e}:{c}": AUDIT_SEED + 1000 * i
                            for i, (e, c, _) in enumerate(SOURCES)},
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2))

    corpus = generate_corpus()
    module = fast.load_baseline_script()
    expected_hash = corpus_hash(corpus)

    manifest = {"corpus_hash": expected_hash, "checkpoints": {}}
    states: dict = {}
    for exp, chain, path in sources:
        name = f"{exp}:{chain}"
        copied, payload = fast.copy_checkpoint(path, copies, f"{exp}_chain{chain}")
        state = Stage6EState.from_dict(payload["state"])
        states[name] = (state, payload)
        h_list = [precedence_from_u(state.u_by_skill[k])
                  for k in range(state.n_skills)]
        manifest["checkpoints"][name] = {
            "experiment": exp, "chain": chain, "source": str(path),
            "copied_to": str(copied), "sha256": fast.sha256_of(copied),
            "sweep": int(payload["sweep"]),
            "induced_h_hash": state_h_hash(h_list),
            "induced_h_canonical_hash": canonical_h_hash(h_list),
            "total_relation_count": int(sum(h.sum() for h in h_list)),
            "relations_per_skill": [int(h.sum()) for h in h_list],
            "segmentation_count": int(sum(len(s.segments)
                                          for s in state.segmentations)),
            "checkpoint_log_target": payload["state"].get(
                "components", {}).get("log_target"),
            "rho": float(state.rho), "beta": float(state.beta),
            "omega": float(state.omega), "lambda_rep": float(state.lambda_rep),
            "lambda_back": float(state.lambda_back),
        }
    (OUT / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2))

    began_all = time.perf_counter()
    results = []
    for i, (exp, chain, _) in enumerate(sources):
        name = f"{exp}:{chain}"
        state, payload = states[name]
        model = module.build_model(corpus)
        seed = config["seeds_per_chain"][name]
        print(f"[expanded] {name} (sweep {payload['sweep']:,}) ...", flush=True)
        result = audit_chain(exp, chain, state, model, seed, n_cross,
                             manifest["checkpoints"][name]["checkpoint_log_target"])
        results.append(result)
        s = chain_summary(result)
        print(f"[expanded] {name}: r_cross {s['r_cross']:.3f}, "
              f"median dLL cond {s['median_d_ll_conditional']:.1f} / "
              f"coll {s['median_d_ll_collapsed']:.1f}, "
              f"P(a>1%) {s['p_alpha_gt_1pct']:.3f}, "
              f"E_50k {s['expected_escapes_50k']:.2e}, "
              f"tail={'YES' if s['useful_tail'] else 'NO'}", flush=True)
        if args.probe and i == len(sources) - 1:
            per = float(np.mean([r["eval_seconds"] for r in result["cross"]]))
            print(f"[expanded] probe: {per:.2f}s per collapsed eval -> "
                  f"~{per * (N_CROSS + N_SAME_H) * 8 / 60:.0f} min for the full pass")
            return
    wall_seconds = time.perf_counter() - began_all

    # ------------------------------------------------------------------ raw arrays
    all_records = [dict(r, chain_name=f"{res['experiment']}:{res['chain']}")
                   for res in results for r in res["cross"] + res["same_h"]]
    arrays: dict = {
        "chain_name": np.array([r["chain_name"] for r in all_records]),
        "proposed_state_hash": np.array([r["proposed_state_hash"]
                                         for r in all_records]),
        "proposed_canonical_hash": np.array([r["proposed_canonical_hash"]
                                             for r in all_records]),
        "h_changed": np.array([r["h_changed"] for r in all_records]),
    }
    for f in ("skill", "row", "d_h", "relations_current", "relations_proposed",
              "d_log_prior", "d_log_lik_conditional", "d_log_lik_collapsed",
              "log_alpha_conditional", "log_alpha_collapsed", "p_accept_conditional",
              "p_accept_collapsed", "eval_seconds"):
        arrays[f] = np.array([r[f] for r in all_records], dtype=float)
    np.savez_compressed(OUT / "proposals.npz", **arrays)

    # ------------------------------------------------------------------ summaries
    summaries = [chain_summary(res) for res in results]
    (OUT / "per_chain_summary.json").write_text(json.dumps(summaries, indent=2))

    def pool(subset):
        cross = [r for res in subset for r in res["cross"]]
        cond = np.array([r["d_log_lik_conditional"] for r in cross])
        coll = np.array([r["d_log_lik_collapsed"] for r in cross])
        alpha = np.array([r["p_accept_collapsed"] for r in cross])
        la = np.array([r["log_alpha_collapsed"] for r in cross])
        subs = [chain_summary(res) for res in subset]
        return {
            "n_cross": len(cross),
            "r_cross_mean": float(np.mean([s["r_cross"] for s in subs])),
            "median_d_ll_conditional": float(np.median(cond)),
            "median_d_ll_collapsed": float(np.median(coll)),
            "median_barrier_reduction": float(np.median(coll - cond)),
            "p_alpha_gt_1pct": float((alpha > 0.01).mean()),
            "p_alpha_gt_10pct": float((alpha > 0.10).mean()),
            "p_log_alpha_ge_0": float((la >= 0).mean()),
            "mean_alpha_collapsed": float(alpha.mean()),
            "expected_escapes_50k_mean": float(np.mean(
                [s["expected_escapes_50k"] for s in subs])),
            "expected_escapes_50k_min": float(np.min(
                [s["expected_escapes_50k"] for s in subs])),
        }

    pooled = {"all": pool(results),
              "ffbs": pool([r for r in results if r["experiment"] == "7b2_ffbs"]),
              "local": pool([r for r in results if r["experiment"] == "6e2_local"])}
    (OUT / "pooled_summary.json").write_text(json.dumps(pooled, indent=2))

    # ------------------------------------------------------ structural distance strata
    cross_all = [r for res in results for r in res["cross"]]
    strata = {}
    for label, keep in (("d_h=1", lambda d: d == 1), ("d_h=2", lambda d: d == 2),
                        ("d_h>=3", lambda d: d >= 3)):
        rows = [r for r in cross_all if keep(r["d_h"])]
        if not rows:
            strata[label] = {"count": 0}
            continue
        cond = np.array([r["d_log_lik_conditional"] for r in rows])
        coll = np.array([r["d_log_lik_collapsed"] for r in rows])
        alpha = np.array([r["p_accept_collapsed"] for r in rows])
        la = np.array([r["log_alpha_collapsed"] for r in rows])
        strata[label] = {
            "count": len(rows),
            "median_d_ll_conditional": float(np.median(cond)),
            "median_d_ll_collapsed": float(np.median(coll)),
            "median_barrier_reduction": float(np.median(coll - cond)),
            "mean_alpha_collapsed": float(alpha.mean()),
            "p_alpha_gt_1pct": float((alpha > 0.01).mean()),
            "p_log_alpha_ge_0": float((la >= 0).mean()),
        }
    (OUT / "structural_distance.json").write_text(json.dumps(strata, indent=2))

    # ------------------------------------------------------------ proposed-H diversity
    diversity = {}
    for res in results:
        name = f"{res['experiment']}:{res['chain']}"
        cross = res["cross"]
        tail1 = [r for r in cross if r["p_accept_collapsed"] > 0.01]
        tail0 = [r for r in cross if r["log_alpha_collapsed"] >= 0.0]
        diversity[name] = {
            "distinct_proposed_h": len({r["proposed_canonical_hash"] for r in cross}),
            "distinct_proposed_h_alpha_gt_1pct": len(
                {r["proposed_canonical_hash"] for r in tail1}),
            "distinct_proposed_h_log_alpha_ge_0": len(
                {r["proposed_canonical_hash"] for r in tail0}),
            "n_alpha_gt_1pct": len(tail1), "n_log_alpha_ge_0": len(tail0),
        }
    (OUT / "proposed_order_diversity.json").write_text(json.dumps(diversity, indent=2))

    # ------------------------------------------------------------------ connectivity
    current_by_chain = {f"{res['experiment']}:{res['chain']}":
                        {"exact": res["current_state_hash"],
                         "canonical": res["current_canonical_hash"]}
                        for res in results}
    connectivity = {
        "note": "exact skill-indexed hashes are NOT comparable across chains (the "
                "target is invariant under joint skill relabelling); matches are "
                "therefore reported under the canonical multiset-of-orders hash, with "
                "exact matches shown for completeness. Absence of matches does not "
                "imply lack of connectivity.",
        "current_hashes": current_by_chain, "matches": []}
    for res in results:
        name = f"{res['experiment']}:{res['chain']}"
        for r in res["cross"]:
            if r["p_accept_collapsed"] <= 0.01 and r["log_alpha_collapsed"] < 0.0:
                continue
            for other, hashes in current_by_chain.items():
                if other == name:
                    continue
                canon = r["proposed_canonical_hash"] == hashes["canonical"]
                exact = r["proposed_state_hash"] == hashes["exact"]
                if canon or exact:
                    connectivity["matches"].append({
                        "from_chain": name, "to_chain": other,
                        "match_type": ("canonical" if canon else "exact"),
                        "skill": r["skill"], "row": r["row"], "d_h": r["d_h"],
                        "p_accept_collapsed": r["p_accept_collapsed"],
                        "log_alpha_collapsed": r["log_alpha_collapsed"]})
    connectivity["n_matches"] = len(connectivity["matches"])
    (OUT / "connectivity.json").write_text(json.dumps(connectivity, indent=2))

    # ------------------------------------------------------------------ runtime
    eval_times = np.array([r["eval_seconds"] for r in cross_all])
    median_eval = float(np.median(eval_times))
    runtime = {
        "n_collapsed_evaluations": int(sum(len(res["cross"]) + len(res["same_h"])
                                           for res in results)),
        "median_eval_seconds": median_eval,
        "p95_eval_seconds": float(np.quantile(eval_times, 0.95)),
        "total_wall_seconds": float(wall_seconds),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "observed_formal_sweep_seconds_7b2": FORMAL_SWEEP_SECONDS_7B2,
        "naive_replacement_estimate": {
            "description": "replace all 15 production U proposals per sweep with "
                           "collapsed scoring (descriptive only, NOT implemented)",
            "added_seconds_per_sweep": 15 * median_eval,
            "added_hours_per_50k_sweeps": 15 * median_eval * SWEEPS / 3600,
            "slowdown_factor_vs_observed_7b2_sweep": 15 * median_eval
            / FORMAL_SWEEP_SECONDS_7B2},
        "occasional_collapsed_proposal_estimate": {
            f"every_{c}_sweeps": {
                "added_seconds_per_sweep": median_eval / c,
                "added_hours_per_50k_sweeps": median_eval * (SWEEPS / c) / 3600,
                "relative_overhead_vs_observed_7b2_sweep": (median_eval / c)
                / FORMAL_SWEEP_SECONDS_7B2}
            for c in (1, 5, 10)},
    }
    (OUT / "runtime.json").write_text(json.dumps(runtime, indent=2))

    correctness = {f"{res['experiment']}:{res['chain']}":
                   dict(res["checks"], integrity=res["integrity"]) for res in results}
    (OUT / "correctness.json").write_text(json.dumps(correctness, indent=2))

    # ------------------------------------------------------------------ verdict
    n_tail = sum(1 for s in summaries if s["useful_tail"])
    if n_tail == 8:
        verdict = "COLLAPSED-U ROBUST ACROSS MODES"
    elif n_tail >= 6:
        verdict = "COLLAPSED-U MOSTLY ROBUST"
    elif n_tail >= 3:
        verdict = "COLLAPSED-U MODE-DEPENDENT"
    else:
        verdict = "COLLAPSED-U NOT ROBUST"

    # ------------------------------------------------------------------ report
    lines = ["# Collapsed-U expanded audit (C1)", "",
             f"8 frozen chains, {N_CROSS} cross-H proposals each from the exact "
             f"production U kernel (scale {U_SCALE}); seeds registered in config.json; "
             "r_cross measured from each chain's raw stream.", "",
             "| Experiment | Chain | H hash | r_cross | med dLL_cond | med dLL_coll | "
             "med reduction | P(a>1%) | P(a>10%) | P(loga>=0) | mean a | E/50k | "
             "tail |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        lines.append(
            f"| {s['experiment']} | {s['chain']} | {s['current_canonical_hash'][:8]} "
            f"| {s['r_cross']:.3f} | {s['median_d_ll_conditional']:.1f} "
            f"| {s['median_d_ll_collapsed']:.1f} "
            f"| {s['median_barrier_reduction']:+.1f} "
            f"| {s['p_alpha_gt_1pct']:.3f} | {s['p_alpha_gt_10pct']:.3f} "
            f"| {s['p_log_alpha_ge_0']:.3f} | {s['mean_alpha_collapsed']:.2e} "
            f"| {s['expected_escapes_50k']:.2e} "
            f"| {'YES' if s['useful_tail'] else 'NO'} |")
    lines += ["", "## Pooled (after the per-chain table)", "```json",
              json.dumps(pooled, indent=2), "```", "",
              "## Structural distance", "```json", json.dumps(strata, indent=2),
              "```", "", f"Chains with a useful tail (pre-registered criteria): "
              f"{n_tail}/8.", "", f"**{verdict}**", ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print(f"[expanded] {verdict} ({n_tail}/8 chains with a useful tail)")
    print(f"[expanded] wall time {wall_seconds / 60:.0f} min; wrote {OUT}")


if __name__ == "__main__":
    main()
