"""C0 fast diagnostic — does FFBS-collapsing (S, z) shrink the cross-H barrier for U?

    cd hpop-step7 && PYTHONPATH=src python scripts/collapsed_u_fast_audit.py [--probe]

The question, and ONLY the question: for the exact production U proposal (row-wise
Gaussian, scale 0.5, same rho, same structural prior, same h(U)), is

    delta_ell_collapsed = sum_n [ log Z_n(U') - log Z_n(U) ]

substantially less negative than the conditional delta log likelihood that the
structural-locking audit showed kills essentially every cross-H proposal?

Each log Z_n(U) = log sum_{S_n, z_n} p(S_n, z_n, X_n | U, Theta, pi, P) is computed by
the validated Step 7A semi-Markov forward recursion (`semi_markov_ffbs.forward`) over the
validated candidate block tables (`FastBlockScoreTable`, every candidate from q_0 = 0).
No backward sampling, no state update, no formal chain touched: the checkpoints are
COPIED into this audit's own directory before being read, and nothing is ever written
back. The two live experiments continue unobserved.

Pre-registered decision rule (written to config.json before any proposal is scored):

    E_escapes = 50,000 sweeps x M_U x r_cross(frozen audit) x E[alpha_coll | cross]
    E_escapes  < 1   -> COLLAPSED-U NOT VIABLE — STOP
    1 <= E   < 10    -> COLLAPSED-U MARGINAL — REVIEW
    E_escapes >= 10  -> COLLAPSED-U MECHANISM VIABLE — EXPAND AUDIT
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.block_score_adapters import (            # noqa: E402
    assert_no_recurrent_state_leak, build_log_block_scores,
)
from hpop.mcmc_original.fast_block_tables import FastBlockScoreTable   # noqa: E402
from hpop.mcmc_original.latent_poset import precedence_from_u          # noqa: E402
from hpop.mcmc_original.sampler_u import propose_row                   # noqa: E402
from hpop.mcmc_original.semi_markov_ffbs import forward                # noqa: E402
from hpop.mcmc_original.stage6c_frozen import log_structural_prior     # noqa: E402
from hpop.mcmc_original.stage6e_corpus import corpus_hash, generate_corpus  # noqa: E402
from hpop.mcmc_original.stage6e_sampler import Stage6ESampler          # noqa: E402
from hpop.mcmc_original.stage6e_state import Stage6EState              # noqa: E402
from hpop.mcmc_original.transitions import log_transition_matrix       # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "collapsed_u_fast_audit"
U_SCALE = 0.5                    # the registered production scale, identical in both runs
AUDIT_SEED = 8_151_500           # this audit's own seed; the formal RNGs are never read
N_CROSS_PER_CHAIN = 50
N_SAME_H_CONTROLS = 8
N_FULL_REBUILD_PARITY = 3        # incremental-vs-full-rebuild checks per chain
M_U_PER_SWEEP = 15               # K x m single-row U proposals per production sweep
SWEEPS = 50_000
PARITY_TOL = 1e-10

FROZEN_AUDIT = ROOT / "results" / "mcmc_original" / "stage7b2_u_audit" / "audit.json"

SOURCES = {
    "7b2_ffbs": ("7b2_ffbs", 0,
                 ROOT / "results" / "mcmc_original" / "stage7b2_full_joint_ffbs"
                 / "checkpoints" / "chain0_checkpoint.json"),
    "6e2_local": ("6e2_local", 0,
                  Path("/Users/dongqing/Desktop/hpop-stage6e/results/mcmc_original"
                       "/stage6e2_unknown_boundary_full_seed0/unknown_checkpoints"
                       "/chain0_checkpoint.json")),
}


def load_baseline_script():
    import importlib.util
    path = ROOT / "scripts" / "stage6e2_formal_chains.py"
    spec = importlib.util.spec_from_file_location("stage6e2_formal_chains", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_checkpoint(source: Path, destination_dir: Path, name: str) -> tuple[Path, dict]:
    """Copy, then parse. A file caught mid-write fails the parse; retry once."""
    destination = destination_dir / f"{name}_chain_checkpoint.json"
    for attempt in range(2):
        shutil.copyfile(source, destination)
        try:
            payload = json.loads(destination.read_text())
            return destination, payload
        except json.JSONDecodeError:
            if attempt == 1:
                raise
            time.sleep(5.0)
    raise RuntimeError("unreachable")


# ------------------------------------------------------------------ collapsed scoring
def collapsed_log_z(table: FastBlockScoreTable, model, log_pi, log_p) -> np.ndarray:
    """Per-trace log Z_n from the validated Step 7A forward recursion."""
    return np.array([
        forward(t, log_pi, log_p, model.delta_b, model.max_width,
                model.min_width).log_normalizer
        for t in table.tables], dtype=float)


def enumeration_log_z(table: np.ndarray, log_pi, log_p, delta_b: float,
                      min_width: int, max_width: int) -> float:
    """Brute-force log Z of one trace by explicit enumeration of every (S, z) path,
    straight from the registered weight formula — no recursion shared with forward()."""
    J = table.shape[0]
    K = table.shape[2]
    log_db = math.log(delta_b)
    log_1mdb = math.log1p(-delta_b)

    def compositions(remaining):
        if remaining == 0:
            yield ()
            return
        for width in range(min_width, min(max_width, remaining) + 1):
            for rest in compositions(remaining - width):
                yield (width,) + rest

    terms = []
    for widths in compositions(J):
        L = len(widths)
        bounds = np.concatenate([[0], np.cumsum(widths)])
        for labels in itertools.product(range(K), repeat=L):
            if any(labels[i] == labels[i + 1] for i in range(L - 1)):
                continue
            value = (float(log_pi[labels[0]])
                     + (J - L) * log_1mdb + (L - 1) * log_db)
            for i in range(L):
                value += float(table[bounds[i], bounds[i + 1], labels[i]])
                if i > 0:
                    value += float(log_p[labels[i - 1], labels[i]])
            if math.isfinite(value):
                terms.append(value)
    return float(logsumexp(terms))


def gaussian_row_log_density(step: np.ndarray, sigma: float) -> float:
    d = step.size
    return float(-0.5 * d * math.log(2.0 * math.pi * sigma * sigma)
                 - 0.5 * float(step @ step) / (sigma * sigma))


def quantile_summary(values: np.ndarray) -> dict:
    qs = np.quantile(values, [0.025, 0.25, 0.50, 0.75, 0.975])
    return {"n": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "min": float(values.min()),
            "max": float(values.max()),
            "q2.5": float(qs[0]), "q25": float(qs[1]), "q50": float(qs[2]),
            "q75": float(qs[3]), "q97.5": float(qs[4])}


# ------------------------------------------------------------------------ the audit
def audit_chain(name: str, state: Stage6EState, model, seed: int,
                n_cross: int, probe: bool) -> dict:
    K, m, d = np.asarray(state.u_by_skill).shape
    u = np.array(state.u_by_skill, dtype=float)
    log_pi = np.log(np.asarray(state.pi, dtype=float))
    log_p = log_transition_matrix(state.transition)

    # -- conditional baseline: exactly the structural-locking audit's machinery --------
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

    # -- collapsed baseline via the fast tables ----------------------------------------
    table = FastBlockScoreTable(traces=model.traces, epsilon=model.epsilon,
                                n_skills=K, min_width=model.min_width,
                                max_width=model.max_width, n_roles=model.n_roles)
    table.refresh(u, state.beta, state.omega, state.lambda_rep, state.lambda_back)
    began = time.perf_counter()
    base_log_z = collapsed_log_z(table, model, log_pi, log_p)
    baseline_seconds = time.perf_counter() - began

    # -- Check 1: current-state parity through an independently constructed table -----
    scorer = model.scorer_for(state)
    q0_leak = assert_no_recurrent_state_leak(
        scorer, 0, (0, model.min_width, 0), (model.min_width, 2 * model.min_width, 1))
    adapter_max_diff = 0.0
    adapter_log_z = []
    n_parity_traces = 3 if probe else len(model.traces)
    for n in range(n_parity_traces):
        adapter_table = build_log_block_scores(
            scorer, n, len(model.traces[n]), K, model.min_width, model.max_width)
        fast = table.tables[n]
        finite = np.isfinite(adapter_table) & np.isfinite(fast)
        if (np.isfinite(adapter_table) != np.isfinite(fast)).any():
            raise AssertionError(f"trace {n}: fast/adapter table support differs")
        adapter_max_diff = max(adapter_max_diff, float(
            np.abs(adapter_table[finite] - fast[finite]).max()))
        adapter_log_z.append(forward(adapter_table, log_pi, log_p, model.delta_b,
                                     model.max_width, model.min_width).log_normalizer)
    log_z_parity = float(np.abs(np.asarray(adapter_log_z)
                                - base_log_z[:n_parity_traces]).max())
    if adapter_max_diff > PARITY_TOL or log_z_parity > PARITY_TOL:
        raise AssertionError(
            f"{name}: current-state parity failed — block diff {adapter_max_diff:.3e}, "
            f"log Z diff {log_z_parity:.3e}")

    # -- Check 3: exact enumeration on the shortest trace ------------------------------
    shortest = int(np.argmin([len(t) for t in model.traces]))
    J_short = len(model.traces[shortest])
    enum_current = enumeration_log_z(table.tables[shortest], log_pi, log_p,
                                     model.delta_b, model.min_width, model.max_width)
    enum_error_current = abs(enum_current - base_log_z[shortest])
    if enum_error_current > PARITY_TOL:
        raise AssertionError(f"{name}: enumeration parity at U failed: "
                             f"{enum_error_current:.3e}")

    # -- the proposal loop: production kernel, frozen state, state never updated -------
    rng = np.random.default_rng(seed)
    cross_records: list[dict] = []
    same_h_records: list[dict] = []
    rebuild_parity: list[dict] = []
    enum_error_proposed = None
    hastings_max_abs = 0.0
    n_seen = 0

    while len(cross_records) < n_cross or len(same_h_records) < N_SAME_H_CONTROLS:
        k = (n_seen // m) % K                # cycle skill 0 rows, skill 1 rows, ...
        row = n_seen % m
        n_seen += 1
        candidate_k = propose_row(u[k], row, U_SCALE, rng)
        cand_prior = log_structural_prior(candidate_k, state.rho)
        if not math.isfinite(cand_prior):
            continue
        h_new = precedence_from_u(candidate_k)
        h_changed = not np.array_equal(h_new, h_now[k])
        need_cross = h_changed and len(cross_records) < n_cross
        need_same = (not h_changed) and len(same_h_records) < N_SAME_H_CONTROLS
        if not (need_cross or need_same):
            continue

        d_prior = cand_prior - prior_now[k]
        cand_cond_ll = skill_ll.full_replay(k, candidate_k, state.beta, state.omega,
                                            state.lambda_rep, state.lambda_back)
        d_ll_cond = cand_cond_ll - cond_ll[k]

        step = candidate_k[row] - u[k][row]
        forward_q = gaussian_row_log_density(step, U_SCALE)
        reverse_q = gaussian_row_log_density(-step, U_SCALE)
        hastings_max_abs = max(hastings_max_abs, abs(reverse_q - forward_q))

        u_prime = np.array(u, copy=True)
        u_prime[k] = candidate_k
        began = time.perf_counter()
        refresh_info = table.refresh(u_prime, state.beta, state.omega,
                                     state.lambda_rep, state.lambda_back)
        if refresh_info["rebuilt_skills"] != [k]:
            raise AssertionError(f"incremental refresh rebuilt "
                                 f"{refresh_info['rebuilt_skills']}, expected [{k}]")
        prop_log_z = collapsed_log_z(table, model, log_pi, log_p)
        eval_seconds = time.perf_counter() - began

        if need_cross and len(rebuild_parity) < N_FULL_REBUILD_PARITY:
            fresh = FastBlockScoreTable(
                traces=model.traces, epsilon=model.epsilon, n_skills=K,
                min_width=model.min_width, max_width=model.max_width,
                n_roles=model.n_roles)
            fresh.refresh(u_prime, state.beta, state.omega, state.lambda_rep,
                          state.lambda_back)
            worst_block = max(
                float(np.abs(a[np.isfinite(a)] - b[np.isfinite(b)]).max())
                for a, b in zip(fresh.tables, table.tables))
            fresh_log_z = collapsed_log_z(fresh, model, log_pi, log_p)
            worst_z = float(np.abs(fresh_log_z - prop_log_z).max())
            rebuild_parity.append({"skill": int(k), "row": int(row),
                                   "max_block_score_diff": worst_block,
                                   "max_log_z_diff": worst_z})
            if worst_block > PARITY_TOL or worst_z > PARITY_TOL:
                raise AssertionError(f"incremental/full rebuild parity failed: "
                                     f"{worst_block:.3e} / {worst_z:.3e}")
            if enum_error_proposed is None:
                enum_prime = enumeration_log_z(
                    table.tables[shortest], log_pi, log_p, model.delta_b,
                    model.min_width, model.max_width)
                enum_error_proposed = abs(enum_prime - prop_log_z[shortest])
                if enum_error_proposed > PARITY_TOL:
                    raise AssertionError(f"enumeration parity at U' failed: "
                                         f"{enum_error_proposed:.3e}")

        d_ll_coll = float((prop_log_z - base_log_z).sum())
        # restore skill k's column to the frozen U before the next proposal
        table.refresh(u, state.beta, state.omega, state.lambda_rep, state.lambda_back)

        record = {
            "chain": name, "skill": int(k), "row": int(row),
            "h_current_relations": int(h_now[k].sum()),
            "h_proposed_relations": int(h_new.sum()),
            "d_h_hamming": int((h_new != h_now[k]).sum()),
            "h_changed": bool(h_changed),
            "d_log_prior": float(d_prior),
            "d_log_lik_conditional": float(d_ll_cond),
            "d_log_lik_collapsed": float(d_ll_coll),
            "log_alpha_conditional": float(d_ll_cond + d_prior),
            "log_alpha_collapsed": float(d_ll_coll + d_prior),
            "p_accept_conditional": min(1.0, math.exp(min(0.0, d_ll_cond + d_prior))),
            "p_accept_collapsed": min(1.0, math.exp(min(0.0, d_ll_coll + d_prior))),
            "collapsed_eval_seconds": float(eval_seconds),
        }
        (cross_records if h_changed else same_h_records).append(record)
        if probe and len(cross_records) >= 3 and len(same_h_records) >= 2:
            break

    # -- Check 2: same-H negative control -----------------------------------------------
    same_cond = np.array([r["d_log_lik_conditional"] for r in same_h_records])
    same_coll = np.array([r["d_log_lik_collapsed"] for r in same_h_records])
    if same_cond.size and float(np.abs(same_cond).max()) > 1e-9:
        raise AssertionError(f"{name}: same-H conditional dLL leak "
                             f"{float(np.abs(same_cond).max()):.3e}")

    return {
        "chain": name, "K": int(K), "m": int(m), "d": int(d),
        "current_relations_per_skill": [int(h_now[k].sum()) for k in range(K)],
        "baseline_collapsed_log_z_total": float(base_log_z.sum()),
        "baseline_collapsed_seconds": float(baseline_seconds),
        "checks": {
            "check1_current_state_parity": {
                "n_traces_compared": n_parity_traces,
                "max_block_score_diff_fast_vs_adapter": adapter_max_diff,
                "max_log_z_diff_fast_vs_adapter": log_z_parity,
                "tolerance": PARITY_TOL, "pass": True},
            "check2_same_h_negative_control": {
                "n_same_h": int(same_cond.size),
                "max_abs_d_ll_conditional": float(np.abs(same_cond).max())
                if same_cond.size else None,
                "max_abs_d_ll_collapsed": float(np.abs(same_coll).max())
                if same_coll.size else None},
            "check3_enumeration": {
                "trace": shortest, "trace_length": J_short,
                "abs_error_at_U": float(enum_error_current),
                "abs_error_at_U_prime": (float(enum_error_proposed)
                                         if enum_error_proposed is not None else None),
                "tolerance": PARITY_TOL, "pass": True},
            "q0_reset_bit_identical": bool(q0_leak["pass"]),
            "hastings_max_abs_diff": float(hastings_max_abs),
            "incremental_vs_full_rebuild": rebuild_parity,
        },
        "cross_records": cross_records,
        "same_h_records": same_h_records,
        "n_proposals_drawn": int(n_seen),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true",
                        help="one chain, ~3 cross proposals, to measure cost and stop")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    copies = OUT / "checkpoint_copies"
    copies.mkdir(exist_ok=True)

    # ---- pre-registration, written before any proposal is scored ----------------------
    config = {
        "question": "does FFBS-collapsing (S,z) shrink the cross-H likelihood barrier "
                    "for the unchanged production U proposal?",
        "u_scale": U_SCALE, "audit_seed": AUDIT_SEED,
        "n_cross_per_chain": N_CROSS_PER_CHAIN,
        "n_same_h_controls": N_SAME_H_CONTROLS,
        "chains": {name: str(path) for name, (_, _, path) in SOURCES.items()},
        "m_u_per_sweep": M_U_PER_SWEEP, "sweeps_for_escape_estimate": SWEEPS,
        "r_cross_source": "frozen structural-locking audit (stage7b2_u_audit)",
        "escape_rule_preregistered": {
            "formula": "E = 50000 * M_U * r_cross * mean(alpha_coll | cross)",
            "E < 1": "COLLAPSED-U NOT VIABLE — STOP",
            "1 <= E < 10": "COLLAPSED-U MARGINAL — REVIEW",
            "E >= 10": "COLLAPSED-U MECHANISM VIABLE — EXPAND AUDIT"},
        "parity_tolerance": PARITY_TOL,
        "proposal_selection": "first N cross-H proposals in deterministic (skill,row) "
                              "cycle order under the registered seed; never filtered "
                              "on acceptance probability",
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2))

    corpus = generate_corpus()
    module = load_baseline_script()

    frozen = json.loads(FROZEN_AUDIT.read_text())
    if frozen["corpus_hash"] != corpus_hash(corpus):
        raise AssertionError("corpus hash mismatch against the frozen audit")

    manifest = {"corpus_hash": corpus_hash(corpus), "checkpoints": {}}
    states = {}
    for name, (audit_key, chain_index, path) in SOURCES.items():
        copied, payload = copy_checkpoint(path, copies, name)
        states[name] = Stage6EState.from_dict(payload["state"])
        manifest["checkpoints"][name] = {
            "source": str(path), "copied_to": str(copied),
            "sha256": sha256_of(copied), "sweep": int(payload["sweep"]),
            "chain_index": chain_index,
            "r_cross_frozen_audit": frozen["experiments"][audit_key][
                str(chain_index)]["r_propose"],
        }
    (OUT / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2))

    results = {}
    order = ["7b2_ffbs", "6e2_local"]
    for i, name in enumerate(order):
        model = module.build_model(corpus)
        print(f"[collapsed-u] {name}: sweep "
              f"{manifest['checkpoints'][name]['sweep']:,} ...", flush=True)
        results[name] = audit_chain(name, states[name], model,
                                    AUDIT_SEED + 1000 * i,
                                    3 if args.probe else N_CROSS_PER_CHAIN, args.probe)
        n_done = len(results[name]["cross_records"])
        per = float(np.mean([r["collapsed_eval_seconds"]
                             for r in results[name]["cross_records"]]))
        print(f"[collapsed-u] {name}: {n_done} cross-H proposals scored, "
              f"{per:.2f}s per collapsed evaluation", flush=True)
        if args.probe:
            print("[collapsed-u] probe complete — stopping before the full pass")
            return

    # ------------------------------------------------------------------ raw arrays
    all_cross = [r for name in order for r in results[name]["cross_records"]]
    all_same = [r for name in order for r in results[name]["same_h_records"]]
    fields = ["skill", "row", "h_current_relations", "h_proposed_relations",
              "d_h_hamming", "d_log_prior", "d_log_lik_conditional",
              "d_log_lik_collapsed", "log_alpha_conditional", "log_alpha_collapsed",
              "p_accept_conditional", "p_accept_collapsed", "collapsed_eval_seconds"]
    arrays = {"chain": np.array([r["chain"] for r in all_cross + all_same]),
              "h_changed": np.array([r["h_changed"] for r in all_cross + all_same])}
    for f in fields:
        arrays[f] = np.array([r[f] for r in all_cross + all_same], dtype=float)
    np.savez_compressed(OUT / "proposals.npz", **arrays)

    # ------------------------------------------------------------------ statistics
    def stats_for(records):
        cond_dll = np.array([r["d_log_lik_conditional"] for r in records])
        coll_dll = np.array([r["d_log_lik_collapsed"] for r in records])
        return {
            "conditional": {
                "d_log_lik": quantile_summary(cond_dll),
                "log_alpha": quantile_summary(np.array(
                    [r["log_alpha_conditional"] for r in records])),
                "p_accept_mean": float(np.mean(
                    [r["p_accept_conditional"] for r in records])),
                "p_accept_median": float(np.median(
                    [r["p_accept_conditional"] for r in records]))},
            "collapsed": {
                "d_log_lik": quantile_summary(coll_dll),
                "log_alpha": quantile_summary(np.array(
                    [r["log_alpha_collapsed"] for r in records])),
                "p_accept_mean": float(np.mean(
                    [r["p_accept_collapsed"] for r in records])),
                "p_accept_median": float(np.median(
                    [r["p_accept_collapsed"] for r in records]))},
            "barrier_reduction": quantile_summary(coll_dll - cond_dll),
        }

    comparison = {"pooled_cross_h": stats_for(all_cross),
                  "per_chain": {name: stats_for(results[name]["cross_records"])
                                for name in order},
                  "checks": {name: results[name]["checks"] for name in order},
                  "runtime_per_collapsed_eval_seconds": quantile_summary(np.array(
                      [r["collapsed_eval_seconds"] for r in all_cross + all_same]))}
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2))

    # ------------------------------------------------------------------ viability
    viability = {"per_chain": {}, "rule": config["escape_rule_preregistered"]}
    for name in order:
        records = results[name]["cross_records"]
        r_cross = manifest["checkpoints"][name]["r_cross_frozen_audit"]
        mean_alpha_coll = float(np.mean([r["p_accept_collapsed"] for r in records]))
        mean_alpha_cond = float(np.mean([r["p_accept_conditional"] for r in records]))
        viability["per_chain"][name] = {
            "r_cross_frozen": r_cross,
            "mean_alpha_conditional_given_cross": mean_alpha_cond,
            "mean_alpha_collapsed_given_cross": mean_alpha_coll,
            "expected_escapes_conditional": SWEEPS * M_U_PER_SWEEP * r_cross
            * mean_alpha_cond,
            "expected_escapes_collapsed": SWEEPS * M_U_PER_SWEEP * r_cross
            * mean_alpha_coll,
        }
    worst_e = min(v["expected_escapes_collapsed"]
                  for v in viability["per_chain"].values())
    best_e = max(v["expected_escapes_collapsed"]
                 for v in viability["per_chain"].values())
    pooled_e = float(np.mean([v["expected_escapes_collapsed"]
                              for v in viability["per_chain"].values()]))
    if pooled_e < 1.0:
        verdict = "COLLAPSED-U NOT VIABLE — STOP"
    elif pooled_e < 10.0:
        verdict = "COLLAPSED-U MARGINAL — REVIEW"
    else:
        verdict = "COLLAPSED-U MECHANISM VIABLE — EXPAND AUDIT"
    viability.update({"expected_escapes_min": worst_e, "expected_escapes_max": best_e,
                      "expected_escapes_mean": pooled_e, "verdict": verdict})
    (OUT / "viability.json").write_text(json.dumps(viability, indent=2))

    # ------------------------------------------------------------------ report
    pooled = comparison["pooled_cross_h"]
    e_cond = float(np.mean([v["expected_escapes_conditional"]
                            for v in viability["per_chain"].values()]))
    lines = [
        "# Collapsed-U fast audit (C0)", "",
        f"Chains: 7B2 FFBS chain 0 (sweep "
        f"{manifest['checkpoints']['7b2_ffbs']['sweep']:,}), 6E2 Local chain 0 (sweep "
        f"{manifest['checkpoints']['6e2_local']['sweep']:,}); "
        f"{len(all_cross)} cross-H proposals from the exact production U kernel "
        f"(scale {U_SCALE}), seed {AUDIT_SEED}. Corpus hash matches the frozen audit.",
        "", "```text",
        "                              Conditional U       Collapsed U",
        "--------------------------------------------------------------",
        f"median cross-H delta log L    {pooled['conditional']['d_log_lik']['median']:>12.1f}"
        f"        {pooled['collapsed']['d_log_lik']['median']:>12.1f}",
        f"max cross-H log alpha         {pooled['conditional']['log_alpha']['max']:>12.1f}"
        f"        {pooled['collapsed']['log_alpha']['max']:>12.1f}",
        f"mean cross-H acceptance       {pooled['conditional']['p_accept_mean']:>12.2e}"
        f"        {pooled['collapsed']['p_accept_mean']:>12.2e}",
        f"expected escapes / 50k        {e_cond:>12.2e}"
        f"        {pooled_e:>12.2e}",
        "```", "",
        f"Barrier reduction (coll - cond): median "
        f"{pooled['barrier_reduction']['median']:+.1f} nats, "
        f"q2.5 {pooled['barrier_reduction']['q2.5']:+.1f}, "
        f"q97.5 {pooled['barrier_reduction']['q97.5']:+.1f}.",
        "", f"**{verdict}**", "",
        "Checks: current-state parity (fast vs adapter tables + log Z) <= 1e-10; "
        "enumeration parity on the shortest trace at U and U' <= 1e-10; incremental "
        "vs full rebuild <= 1e-10; q0 reset bit-identical; same-H conditional dLL = 0; "
        "Hastings term numerically 0. Details in comparison.json.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[collapsed-u] {verdict}")
    print(f"[collapsed-u] wrote {OUT}")


if __name__ == "__main__":
    main()
