"""Condition B — structure identifiability under oracle paths.

Run:  PYTHONPATH=src .venv/bin/python scripts/run_matched_condition_b.py

Target (registered correction applied): p(U | X, S*, z*, vartheta*, rho_0)
with rho FIXED at the preregistered rho_0 — the formal corpus supplies U*
directly and records rho* = null, so no rho-recovery claim is made or gated.
Only U moves. Protocol, pilot, starts, seeds, gates and the verdict rule are
frozen before any formal draw; recovery is analysed only after the sequential
checkpoint rule stops the chains.
"""

from __future__ import annotations

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
from hpop.mcmc_original import matched_synthetic_generator as msg              # noqa: E402
from hpop.mcmc_original.latent_poset import precedence_from_u                  # noqa: E402
from hpop.mcmc_original.recurrent_rfs import (                                 # noqa: E402
    RecurrentRFSParameters, recurrent_rfs_log_likelihood,
    recurrent_step_probabilities, recurrent_validity_update,
)
from hpop.mcmc_original.sampler_u import log_u_prior, sigma_rho_matrix         # noqa: E402
from hpop.mcmc_original.stage6b_mcmc_diagnostics import (                      # noqa: E402
    bulk_ess, rank_normalized_split_rhat, tail_ess,
)
from hpop.mcmc_original.stage6c_frozen import SIGMA_U                          # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "matched_condition_b"
CORPUS_DIR = ROOT / "results" / "mcmc_original" / "matched_synthetic_formal_corpus"

# ------------------------------------------------------------- frozen protocol
RHO_0 = 0.5                     # preregistered: near the registered prior median
                                # (Uniform(0, 0.995) -> 0.4975); NOT inferred.
GENERATION_SEED = 6_200_001
FORMAL_SEEDS = (6_202_001, 6_202_002, 6_202_003, 6_202_004)
START_SEEDS = (6_202_101, 6_202_102, 6_202_103, 6_202_104)
START_SCALES = (0.5, 1.0, 2.0, 3.0)
PILOT_SEEDS = (6_202_201, 6_202_202)
PILOT_START_SEEDS = (6_202_211, 6_202_212)
PRIOR_CHECK_MCMC_SEEDS = (6_202_301, 6_202_302)
PRIOR_CHECK_IID_SEED = 6_202_310
PARITY_SEED = 6_202_401

PILOT_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
PILOT_SWEEPS = 3_000
ACCEPTANCE_BAND = (0.20, 0.60)

CHECKPOINTS = (30_000, 50_000, 75_000, 100_000)
BURN_IN = 10_000
THIN = 5
RHAT_GATE = 1.01
ESS_FLOORS = {"log_posterior_bulk": 1000.0, "log_posterior_tail": 500.0,
              "total_relations_bulk": 1000.0, "uncertain_relation_bulk": 500.0}
UNCERTAIN_BAND = (0.05, 0.95)
THRESHOLD = 0.5

PRIOR_CHECK = {"mcmc_sweeps": 60_000, "mcmc_burn": 5_000, "mcmc_thin": 5,
               "iid_draws": 200_000, "relation_marginal_gate": 0.02,
               "relation_count_tv_gate": 0.02}

VERDICT_RULE = {
    "not_converged": "any convergence gate unmet after the 100k ceiling -> "
                     "'CONDITION B INFERENCE NOT CONVERGED — NO RECOVERY "
                     "CLAIM'",
    "strong_requires_all": {
        "convergence": "two consecutive checkpoint passes",
        "per_skill_closure_f1": ">= 0.90 for every skill (threshold 0.5)",
        "per_skill_incomparable_f1": ">= 0.80 for every skill",
        "modal_h_equals_truth": "for >= 2 of 3 skills",
        "min_true_relation_marginal": ">= 0.5",
        "max_false_relation_marginal": "< 0.5",
        "heldout": "posterior-mean NLL/occ <= antichain and <= total-order "
                   "baselines AND within 0.05 nats/occ of generating truth",
    },
    "not_identifiable_if_any": {
        "mean_closure_f1": "< 0.60",
        "heldout": "posterior-mean NLL/occ worse than the antichain baseline",
    },
    "otherwise": "STRUCTURE PARTIALLY IDENTIFIABLE UNDER ORACLE PATHS",
}

_WORKER_CACHE: dict = {}


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


def build_target(truth, corpus) -> mcb.ConditionBTarget:
    blocks = mcb.oracle_blocks_by_skill(corpus.train, truth.n_skills)
    likelihood = mcb.OracleBlockLikelihood(
        blocks, truth.beta, truth.epsilon, truth.omega, truth.lambda_rep,
        truth.lambda_back)
    return mcb.ConditionBTarget(likelihood=likelihood, rho_0=RHO_0)


def make_start(truth, start_index: int) -> np.ndarray:
    rng = np.random.default_rng(START_SEEDS[start_index])
    chol = np.linalg.cholesky(sigma_rho_matrix(truth.latent_dim, RHO_0))
    return np.array([
        [START_SCALES[start_index] * (chol @ rng.standard_normal(
            truth.latent_dim)) for _ in range(truth.n_roles)]
        for _ in range(truth.n_skills)])


def h_tuple_hashes(u_by_skill) -> tuple:
    return tuple(mcb.canonical_h_hash(precedence_from_u(u_by_skill[k]))
                 for k in range(u_by_skill.shape[0]))


# ================================================================ phase 0/1
def verify_preconditions() -> dict:
    log = _git("log", "--format=%H", "-20")
    if "8ca828153e8e263bf4ea4823e45a53fa454037ad" not in log:
        raise SystemExit("matched generator commit not in history")
    gen_report = (ROOT / "results/mcmc_original/matched_generator_validation/"
                  "report.md").read_text()
    if "ALL GATES PASS" not in gen_report:
        raise SystemExit("generator validation did not pass")
    verdict_a = json.loads((ROOT / "results/mcmc_original/matched_condition_a/"
                            "final_verdict.json").read_text())
    if verdict_a["verdict"] == "PATH NOT IDENTIFIABLE UNDER ORACLE STRUCTURES":
        raise SystemExit("Condition A failed its ladder gate: user "
                         "authorization required before Condition B")
    recorded = json.loads((CORPUS_DIR / "corpus_hash.json").read_text())
    truth, corpus = build_environment()
    if msg.corpus_hash(corpus) != recorded["corpus_hash_sha256"]:
        raise SystemExit("formal corpus hash changed — refusing to run")
    truth_hash = msg.sha256_hex(msg.canonical_json(msg.truth_to_jsonable(truth)))
    if truth_hash != recorded["truth_hash_sha256"]:
        raise SystemExit("truth manifest hash changed — refusing to run")
    return {"condition_a_verdict": verdict_a["verdict"],
            "corpus_hash_sha256": recorded["corpus_hash_sha256"],
            "truth_hash_sha256": recorded["truth_hash_sha256"]}


def freeze_protocol(pre: dict) -> None:
    _dump("preregistration.json", {
        "condition": "B — structure identifiability under oracle paths",
        "correction_applied": "rho FIXED at rho_0 (formal corpus records "
                              "rho* = null); no rho recovery or coverage gate; "
                              "a later prior-draw calibration experiment may "
                              "evaluate rho and is NOT part of this task",
        "rho_0": RHO_0,
        "rho_0_rationale": "near the registered rho prior median (Uniform(0, "
                           "0.995) -> 0.4975); positive-definite Sigma with "
                           "moderate column coupling; frozen before any draw",
        **pre,
        "parent_commit": _git("rev-parse", "HEAD"),
        "active_variables": ["U_1", "U_2", "U_3"],
        "fixed_variables": {"S": "S* (oracle)", "z": "z* (oracle)",
                            "beta": 1.5, "omega": math.log(0.85 / 0.15),
                            "lambda_rep": 0.8, "lambda_back": 0.25,
                            "pi": "pi*", "P": "P*", "delta_B": 0.15,
                            "epsilon": 0.02, "rho": RHO_0},
        "no_moves": ["FFBS", "local segmentation", "collapsed-U", "pi/P",
                     "recurrent scalars", "rho"],
        "proposal_family": "sampler_u.propose_row symmetric Gaussian row walk; "
                           "MH on Delta log p(U_k|rho_0) + Delta oracle-block "
                           "log likelihood (u_row_sweep arithmetic)",
        "pilot": {"grid_multipliers": list(PILOT_MULTIPLIERS),
                  "base_scale": SIGMA_U, "sweeps_per_scale": PILOT_SWEEPS,
                  "chains": 2, "seeds": list(PILOT_SEEDS),
                  "start_seeds": list(PILOT_START_SEEDS),
                  "acceptance_band": list(ACCEPTANCE_BAND),
                  "selection_rule": "acceptance in band -> max ESJD; else "
                                    "closest to band, band never widened",
                  "may_inspect": ["acceptance", "ESJD", "runtime",
                                  "H-change frequency", "numerical health"],
                  "must_not_inspect": ["truth recovery", "relation F1",
                                       "Hamming to H*", "coverage",
                                       "held-out likelihood", "formal R-hat"]},
        "formal_seeds": list(FORMAL_SEEDS),
        "start_seeds": list(START_SEEDS), "start_scales": list(START_SCALES),
        "checkpoints": list(CHECKPOINTS), "burn_in": BURN_IN, "thin": THIN,
        "ceiling": CHECKPOINTS[-1],
        "stopping_rule": "two consecutive checkpoint passes; never extend "
                         "beyond the ceiling automatically",
        "rhat_gate": RHAT_GATE, "ess_floors": ESS_FLOORS,
        "uncertain_relation_band": list(UNCERTAIN_BAND),
        "degenerate_summary_rule": "a summary constant within every chain AND "
                                   "equal across chains passes trivially "
                                   "(R-hat := 1, ESS := n_retained); constant "
                                   "within chains but unequal across chains "
                                   "is an automatic R-hat failure",
        "movement_rule": "a chain with zero accepted H changes cannot receive "
                         "a convergence PASS",
        "point_estimate_rule": "posterior modal canonical H per skill "
                               "(pre-registered; the 0.5-thresholded relation "
                               "matrix is reported secondarily with a "
                               "validity check)",
        "threshold": THRESHOLD,
        "verdict_rule": VERDICT_RULE,
        "u_recovery_note": "raw entrywise U error is NOT a target; latent "
                           "coordinates are exchangeable and the likelihood "
                           "reads U only through h(U)",
    })
    _dump("corpus_manifest.json", {
        "corpus_dir": str(CORPUS_DIR.relative_to(ROOT)),
        "corpus_hash_sha256": pre["corpus_hash_sha256"],
        "truth_hash_sha256": pre["truth_hash_sha256"],
        "generation_seed": GENERATION_SEED,
        "reused_exactly": True, "observations_regenerated": False,
        "oracle_blocks": "TRAIN split: 541 blocks; heldout 244 blocks used "
                         "only for oracle-path prediction"})
    _dump("target_manifest.json", {
        "target": "p(U | X, S*, z*, vartheta*, rho_0) proportional to "
                  "prod_k p(U_k | rho_0) prod_(n,l) e_{n,z*_nl}(a*_nl, b*_nl; "
                  "h(U), vartheta*)",
        "constant_terms_never_computed": ["p(S* | J, delta_B*)",
                                         "p(z* | S*, pi*, P*)"],
        "likelihood_reads_u_through": "H_k = h(U_k) only (verified: "
                                      "recurrent_step_probabilities consumes "
                                      "precedence_from_u(u) alone), so the "
                                      "block likelihood is cached per "
                                      "(skill, canonical H)",
        "q0": "every oracle block replayed from q_0 = 0 (production "
              "vectorized_state_features initialises q = 0)",
        "rho_0": RHO_0})
    sources = ["src/hpop/mcmc_original/matched_condition_b.py",
               "scripts/run_matched_condition_b.py",
               "src/hpop/mcmc_original/sampler_u.py",
               "src/hpop/mcmc_original/recurrent_joint_scalar_mcmc.py",
               "src/hpop/mcmc_original/recurrent_scalar_posterior.py",
               "src/hpop/mcmc_original/stage6b_mcmc_diagnostics.py",
               "src/hpop/mcmc_original/latent_poset.py"]
    import hashlib
    _dump("source_manifest.json", {
        "parent_commit": _git("rev-parse", "HEAD"),
        "file_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                        for p in sources if (ROOT / p).exists()},
        "collapsed_u_and_ffbs": "collapsed_u_kernel.py, "
                                "collapsed_u_likelihood.py and "
                                "semi_markov_ffbs.py do not exist in this "
                                "worktree and are not imported anywhere in "
                                "the Condition-B modules (source-level test "
                                "enforces this); their diffs are empty by "
                                "absence",
        "reused_validated_components": [
            "sampler_u.propose_row / log_u_prior / sigma_rho_matrix",
            "latent_poset.precedence_from_u",
            "recurrent_joint_scalar_mcmc.vectorized_state_features",
            "recurrent_scalar_posterior.cached_batch_log_likelihood",
            "stage6b_mcmc_diagnostics rank-normalized split R-hat / ESS"]})


# ============================================================ phase 2: parity
def run_parity_checks(truth, corpus, target) -> dict:
    params = RecurrentRFSParameters(
        beta=truth.beta, epsilon=truth.epsilon, shared_omega=truth.omega,
        lambda_rep=truth.lambda_rep, lambda_back=truth.lambda_back)

    def direct_ll(u):
        return sum(recurrent_rfs_log_likelihood(
            block, u[k], truth.beta, truth.epsilon, truth.omega,
            truth.lambda_rep, truth.lambda_back)
            for t in corpus.train for k, block in zip(t.labels, t.role_blocks))

    def direct_prior(u):
        total = 0.0
        sigma = sigma_rho_matrix(truth.latent_dim, RHO_0)
        inv = np.linalg.inv(sigma)
        _, logdet = np.linalg.slogdet(sigma)
        for k in range(u.shape[0]):
            for j in range(u.shape[1]):
                row = u[k, j]
                total += -0.5 * (truth.latent_dim * math.log(2 * math.pi)
                                 + logdet + float(row @ inv @ row))
        return total

    rng = np.random.default_rng(PARITY_SEED)
    points = [truth.u_by_skill] + [
        truth.u_by_skill + s * rng.standard_normal(truth.u_by_skill.shape)
        for s in (0.1, 0.3, 0.5, 0.8, 1.2) for _ in range(4)]
    worst_ll = worst_prior = worst_target = 0.0
    for u in points:
        fresh = build_target(truth, corpus)     # cold caches: honest replay
        worst_ll = max(worst_ll, abs(fresh.likelihood.total(u) - direct_ll(u)))
        worst_prior = max(worst_prior, abs(fresh.log_prior(u)
                                           - direct_prior(u)))
        worst_target = max(worst_target, abs(
            fresh.log_target(u) - (direct_ll(u) + direct_prior(u))))

    # q_0 reset on every oracle block: per-block replay parity, per skill
    q0_worst = 0.0
    for k in range(truth.n_skills):
        per_block = target.likelihood.skill_block_log_likelihoods(
            k, truth.u_by_skill[k])
        direct_blocks = []
        for width, arr in target.likelihood.blocks_by_skill[k].items():
            for row in arr:
                direct_blocks.append(recurrent_rfs_log_likelihood(
                    row, truth.u_by_skill[k], truth.beta, truth.epsilon,
                    truth.omega, truth.lambda_rep, truth.lambda_back))
        q0_worst = max(q0_worst, float(np.abs(
            per_block - np.asarray(direct_blocks)).max()))

    # negative control: leak q across a trace's oracle blocks
    leak_rng = np.random.default_rng(PARITY_SEED + 1)
    trace = corpus.train[0]
    q = np.zeros(truth.n_roles)
    leaky_total = 0.0
    for k, block in zip(trace.labels, trace.role_blocks):
        u = truth.u_by_skill[int(k)]
        precedence = precedence_from_u(u)
        for y in block:
            mixed = recurrent_step_probabilities(u, q, params)
            leaky_total += math.log(float(mixed[int(y)]))
            q = recurrent_validity_update(int(y), precedence, q,
                                          params.shared_omega)
    honest_total = sum(recurrent_rfs_log_likelihood(
        block, truth.u_by_skill[int(k)], truth.beta, truth.epsilon,
        truth.omega, truth.lambda_rep, truth.lambda_back)
        for k, block in zip(trace.labels, trace.role_blocks))
    leak_gap = abs(leaky_total - honest_total)

    out = {
        "n_points": len(points),
        "max_likelihood_discrepancy": worst_ll,
        "max_prior_discrepancy": worst_prior,
        "max_complete_target_discrepancy": worst_target,
        "gate": 1e-10,
        "q0_reset_per_block_max_error": q0_worst,
        "negative_control_state_leak": {
            "abs_log_likelihood_gap": leak_gap,
            "detected": bool(leak_gap > 1e-6)},
        "pass": bool(worst_ll < 1e-10 and worst_prior < 1e-10
                     and worst_target < 1e-10 and q0_worst < 1e-10
                     and leak_gap > 1e-6),
    }
    print(f"  parity: LL {worst_ll:.2e}, prior {worst_prior:.2e}, target "
          f"{worst_target:.2e}, q0 {q0_worst:.2e}, leak detected "
          f"{out['negative_control_state_leak']['detected']}")
    return out


# =================================================== phase 3: prior reference
def run_prior_reference_check(truth) -> dict:
    empty = {k: {} for k in range(truth.n_skills)}
    prior_lik = mcb.OracleBlockLikelihood(
        empty, truth.beta, truth.epsilon, truth.omega, truth.lambda_rep,
        truth.lambda_back)
    prior_target = mcb.ConditionBTarget(likelihood=prior_lik, rho_0=RHO_0)

    retained_indicators, retained_counts = [], []
    log_prior_parity = 0.0
    for c, seed in enumerate(PRIOR_CHECK_MCMC_SEEDS):
        start = make_start(truth, c % 4)
        chain = mcb.ConditionBChain(
            target=prior_target, u_by_skill=start, sigma_u=SIGMA_U,
            rng=np.random.default_rng(seed))
        for sweep in range(1, PRIOR_CHECK["mcmc_sweeps"] + 1):
            chain.run_sweeps(1)
            if (sweep > PRIOR_CHECK["mcmc_burn"]
                    and (sweep - PRIOR_CHECK["mcmc_burn"])
                    % PRIOR_CHECK["mcmc_thin"] == 0):
                row = chain.summary_row()
                retained_indicators.append(row["relation_indicators"])
                retained_counts.append(row["total_relations"])
        recomputed = prior_target.log_prior(chain.u_by_skill)
        log_prior_parity = max(log_prior_parity,
                               abs(chain.log_target() - recomputed))
    mcmc_marginals = np.mean(retained_indicators, axis=0)

    rng = np.random.default_rng(PRIOR_CHECK_IID_SEED)
    chol = np.linalg.cholesky(sigma_rho_matrix(truth.latent_dim, RHO_0))
    n = PRIOR_CHECK["iid_draws"]
    z = rng.standard_normal((n, truth.n_skills, truth.n_roles,
                             truth.latent_dim))
    u_draws = z @ chol.T
    iid_ind = np.empty((n, truth.n_skills * truth.n_roles
                        * (truth.n_roles - 1)), dtype=bool)
    iid_counts = np.empty(n, dtype=int)
    for d in range(n):
        iid_ind[d] = mcb.relation_indicator_vector(u_draws[d])
        iid_counts[d] = int(iid_ind[d].sum() // 1)
    iid_marginals = iid_ind.mean(axis=0)

    max_count = max(int(iid_counts.max()), max(retained_counts))
    mcmc_hist = np.bincount(retained_counts, minlength=max_count + 1) \
        / len(retained_counts)
    iid_hist = np.bincount(iid_counts, minlength=max_count + 1) / n
    count_tv = 0.5 * float(np.abs(mcmc_hist - iid_hist).sum())
    marginal_diff = float(np.abs(mcmc_marginals - iid_marginals).max())
    out = {
        "reference": "prior-only target (empty oracle-block subset): the "
                     "row-MH sampler must reproduce iid draws from "
                     "p(U | rho_0); rho marginal is N/A (rho fixed). The "
                     "validated Stage 6C/6D oracle-block U kernel is the "
                     "cited precedent for the with-data kernel.",
        "mcmc_retained": len(retained_counts), "iid_draws": n,
        "relation_marginal_max_abs_diff": marginal_diff,
        "relation_marginal_gate": PRIOR_CHECK["relation_marginal_gate"],
        "relation_count_tv": count_tv,
        "relation_count_tv_gate": PRIOR_CHECK["relation_count_tv_gate"],
        "log_target_parity": log_prior_parity,
        "pass": bool(marginal_diff < PRIOR_CHECK["relation_marginal_gate"]
                     and count_tv < PRIOR_CHECK["relation_count_tv_gate"]
                     and log_prior_parity < 1e-10),
    }
    print(f"  prior reference: marginal diff {marginal_diff:.4f}, count TV "
          f"{count_tv:.4f}, parity {log_prior_parity:.2e}")
    return out


# ====================================================== phase 4: pilot
def run_pilot(truth, corpus) -> tuple:
    results = []
    for multiplier in PILOT_MULTIPLIERS:
        scale = SIGMA_U * multiplier
        acc, esjd, h_freq, sweeps_per_s = [], [], [], []
        healthy = True
        for c in range(2):
            target = build_target(truth, corpus)
            start_rng = np.random.default_rng(PILOT_START_SEEDS[c])
            chol = np.linalg.cholesky(sigma_rho_matrix(truth.latent_dim,
                                                       RHO_0))
            start = np.array([[1.5 * (chol @ start_rng.standard_normal(
                truth.latent_dim)) for _ in range(truth.n_roles)]
                for _ in range(truth.n_skills)])
            chain = mcb.ConditionBChain(
                target=target, u_by_skill=start, sigma_u=scale,
                rng=np.random.default_rng(PILOT_SEEDS[c]))
            t0 = time.perf_counter()
            chain.run_sweeps(PILOT_SWEEPS)
            dt = time.perf_counter() - t0
            healthy &= math.isfinite(chain.log_target())
            acc.append(chain.accepted / chain.proposed)
            esjd.append(chain.esjd_sum / chain.proposed)
            h_freq.append(chain.h_change_accepted / chain.proposed)
            sweeps_per_s.append(PILOT_SWEEPS / dt)
        results.append({
            "multiplier": multiplier, "sigma_u": scale,
            "acceptance": float(np.mean(acc)),
            "esjd_per_proposal": float(np.mean(esjd)),
            "h_change_accept_rate": float(np.mean(h_freq)),
            "sweeps_per_second": float(np.mean(sweeps_per_s)),
            "finite": healthy})
        print(f"  pilot sigma_u={scale:.3f}: acc={np.mean(acc):.3f} "
              f"ESJD={np.mean(esjd):.4f} H-rate={np.mean(h_freq):.4f}")
    lo, hi = ACCEPTANCE_BAND
    admissible = [r for r in results if lo <= r["acceptance"] <= hi
                  and r["finite"]]
    if admissible:
        chosen = max(admissible, key=lambda r: r["esjd_per_proposal"])
        rule = "acceptance in band, max ESJD"
    else:
        chosen = min(results, key=lambda r: min(abs(r["acceptance"] - lo),
                                                abs(r["acceptance"] - hi)))
        rule = "no admissible scale: closest to band, band not widened"
    return results, {"sigma_u": chosen["sigma_u"],
                     "multiplier": chosen["multiplier"],
                     "acceptance": chosen["acceptance"],
                     "esjd_per_proposal": chosen["esjd_per_proposal"],
                     "selection_rule_applied": rule,
                     "rho_scale": "N/A — rho fixed at rho_0",
                     "all_pilot_draws_discarded": True}


# ============================================== phase 5/6: formal chains
def _formal_segment(args):
    """Worker: advance one chain to the next checkpoint, recording summaries."""
    chain_index, payload, sigma_u, upto = args
    key = "env"
    if key not in _WORKER_CACHE:
        truth, corpus = build_environment()
        _WORKER_CACHE[key] = (truth, corpus)
    truth, corpus = _WORKER_CACHE[key]
    target_key = f"target{chain_index}"
    if target_key not in _WORKER_CACHE:
        _WORKER_CACHE[target_key] = build_target(truth, corpus)
    target = _WORKER_CACHE[target_key]
    if payload is None:
        start = make_start(truth, chain_index)
        chain = mcb.ConditionBChain(
            target=target, u_by_skill=start, sigma_u=sigma_u,
            rng=np.random.default_rng(FORMAL_SEEDS[chain_index]))
    else:
        chain = mcb.ConditionBChain.resume(payload, target)
    rows = {"log_posterior": [], "log_prior": [], "total_relations": [],
            "per_skill": [], "indicators": [], "h_hashes": []}
    t0 = time.perf_counter()
    while chain.sweep < upto:
        chain.run_sweeps(1)
        if (chain.sweep > BURN_IN and (chain.sweep - BURN_IN) % THIN == 0):
            row = chain.summary_row()
            rows["log_posterior"].append(row["log_posterior"])
            rows["log_prior"].append(row["log_prior"])
            rows["total_relations"].append(row["total_relations"])
            rows["per_skill"].append(row["per_skill_relations"])
            rows["indicators"].append(row["relation_indicators"])
            rows["h_hashes"].append(row["h_hashes"])
    seconds = time.perf_counter() - t0
    movement = {
        "proposed": chain.proposed, "accepted": chain.accepted,
        "h_change_proposed": chain.h_change_proposed,
        "h_change_accepted": chain.h_change_accepted,
        "first_h_change_sweep": chain.first_h_change_sweep,
        "esjd_sum": chain.esjd_sum,
        "likelihood_evaluations": target.likelihood.evaluations,
        "segment_seconds": seconds,
    }
    return chain_index, chain.checkpoint(), rows, movement


def _diag(series_by_chain: list) -> dict:
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


def checkpoint_gates(all_rows: list, movement: list) -> dict:
    n_chains = len(all_rows)
    summaries = {}
    summaries["log_posterior"] = _diag([r["log_posterior"] for r in all_rows])
    summaries["log_prior"] = _diag([r["log_prior"] for r in all_rows])
    summaries["total_relations"] = _diag([r["total_relations"]
                                          for r in all_rows])
    for k in range(3):
        summaries[f"relations_skill{k}"] = _diag(
            [[row[k] for row in r["per_skill"]] for r in all_rows])
        summaries[f"sorted_relations_rank{k}"] = _diag(
            [[sorted(row)[k] for row in r["per_skill"]] for r in all_rows])
    indicators = [np.asarray(r["indicators"], dtype=float) for r in all_rows]
    pooled = np.concatenate(indicators, axis=0).mean(axis=0)
    uncertain = [int(i) for i in np.flatnonzero(
        (pooled >= UNCERTAIN_BAND[0]) & (pooled <= UNCERTAIN_BAND[1]))]
    uncertain_diags = {}
    for i in uncertain:
        uncertain_diags[str(i)] = _diag([ind[:, i] for ind in indicators])
    distinct_states = len({h for r in all_rows for h in r["h_hashes"]})

    checks = {
        "max_rhat": max(v["rhat"] for v in summaries.values()),
        "uncertain_max_rhat": max([v["rhat"] for v in uncertain_diags.values()],
                                  default=1.0),
        "log_posterior_bulk_ess": summaries["log_posterior"]["bulk_ess"],
        "log_posterior_tail_ess": summaries["log_posterior"]["tail_ess"],
        "total_relations_bulk_ess": summaries["total_relations"]["bulk_ess"],
        "uncertain_min_bulk_ess": min([v["bulk_ess"]
                                       for v in uncertain_diags.values()],
                                      default=float("inf")),
        "chains_with_zero_accepted_h_changes": sum(
            1 for m in movement if m["h_change_accepted"] == 0),
        "n_uncertain_relations": len(uncertain),
        "distinct_canonical_h_states_pooled": distinct_states,
    }
    passed = bool(
        checks["max_rhat"] <= RHAT_GATE
        and checks["uncertain_max_rhat"] <= RHAT_GATE
        and checks["log_posterior_bulk_ess"]
        >= ESS_FLOORS["log_posterior_bulk"]
        and checks["log_posterior_tail_ess"]
        >= ESS_FLOORS["log_posterior_tail"]
        and checks["total_relations_bulk_ess"]
        >= ESS_FLOORS["total_relations_bulk"]
        and checks["uncertain_min_bulk_ess"]
        >= ESS_FLOORS["uncertain_relation_bulk"]
        and checks["chains_with_zero_accepted_h_changes"] == 0)
    return {"summaries": summaries, "uncertain_relations": uncertain_diags,
            "checks": checks, "pass": passed}


def run_formal(truth, corpus, sigma_u: float) -> dict:
    (OUT / "chain_checkpoints").mkdir(exist_ok=True)
    payloads = [None] * 4
    rows_accum = [
        {"log_posterior": [], "log_prior": [], "total_relations": [],
         "per_skill": [], "indicators": [], "h_hashes": []}
        for _ in range(4)]
    movement_accum = [None] * 4
    checkpoint_log, consecutive, stopped_at = [], 0, None
    with ProcessPoolExecutor(max_workers=4) as pool:
        for checkpoint in CHECKPOINTS:
            jobs = [(c, payloads[c], sigma_u, checkpoint) for c in range(4)]
            for c, payload, rows, movement in pool.map(_formal_segment, jobs):
                payloads[c] = payload
                for key in rows_accum[c]:
                    rows_accum[c][key].extend(rows[key])
                if movement_accum[c] is None:
                    movement_accum[c] = movement
                else:
                    prev_seconds = movement_accum[c]["segment_seconds"]
                    movement_accum[c] = movement
                    movement_accum[c]["segment_seconds"] += prev_seconds
                (OUT / "chain_checkpoints" /
                 f"chain{c}_at{checkpoint}.json").write_text(
                    json.dumps(payload, indent=None, sort_keys=True))
            gate = checkpoint_gates(rows_accum, movement_accum)
            checkpoint_log.append({"checkpoint": checkpoint,
                                   "pass": gate["pass"],
                                   "checks": gate["checks"]})
            print(f"  checkpoint {checkpoint}: "
                  f"{'PASS' if gate['pass'] else 'FAIL'} "
                  f"(max R-hat {gate['checks']['max_rhat']:.4f}, "
                  f"logpost ESS {gate['checks']['log_posterior_bulk_ess']:.0f})")
            consecutive = consecutive + 1 if gate["pass"] else 0
            if consecutive >= 2:
                stopped_at = checkpoint
                final_gate = gate
                break
            final_gate = gate
    converged = consecutive >= 2
    return {"rows": rows_accum, "movement": movement_accum,
            "checkpoint_log": checkpoint_log, "final_gate": final_gate,
            "converged": converged,
            "stopped_at": stopped_at if stopped_at else CHECKPOINTS[-1],
            "stop_reason": ("two consecutive checkpoint passes" if converged
                            else "ceiling reached without two consecutive "
                                 "passes")}


# ================================================= phase 7/8: recovery analysis
def analyse(truth, corpus, formal: dict) -> dict:
    rows = formal["rows"]
    indicators = np.concatenate(
        [np.asarray(r["indicators"], dtype=float) for r in rows], axis=0)
    pooled_marginals = indicators.mean(axis=0)
    K, m = truth.n_skills, truth.n_roles
    per_skill_marginals = np.zeros((K, m, m))
    idx = 0
    for k in range(K):
        for i in range(m):
            for j in range(m):
                if i != j:
                    per_skill_marginals[k, i, j] = pooled_marginals[idx]
                    idx += 1

    true_closures = [precedence_from_u(truth.u_by_skill[k]) for k in range(K)]
    true_hashes = [mcb.canonical_h_hash(c) for c in true_closures]

    h_draws = [h for r in rows for h in r["h_hashes"]]
    n_draws = len(h_draws)
    per_skill_h = [Counter(h[k] for h in h_draws) for k in range(K)]
    joint_h = Counter(h_draws)

    recovery = {"per_skill": [], "threshold": THRESHOLD}
    min_true, max_false = 1.0, 0.0
    for k in range(K):
        marginals_k = per_skill_marginals[k]
        closure = mcb.closure_metrics(marginals_k, true_closures[k], THRESHOLD)
        incomparable = mcb.incomparable_metrics(marginals_k, true_closures[k],
                                               THRESHOLD)
        modal_hash, modal_count = per_skill_h[k].most_common(1)[0]
        # rebuild the modal closure from any draw is not possible from hash
        # alone; store thresholded matrix and compare hashes instead
        thresholded = (marginals_k >= THRESHOLD)
        np.fill_diagonal(thresholded, False)
        thresh_valid = (not np.any(thresholded & thresholded.T)
                        and not np.any(((thresholded.astype(int)
                                         @ thresholded.astype(int)) > 0)
                                       & ~thresholded))
        reduction = mcb.reduction_metrics(thresholded, true_closures[k])
        off = ~np.eye(m, dtype=bool)
        true_vals = marginals_k[true_closures[k] & off]
        false_vals = marginals_k[~true_closures[k] & off]
        min_true = min(min_true, float(true_vals.min()) if true_vals.size
                       else 1.0)
        max_false = max(max_false, float(false_vals.max()) if false_vals.size
                        else 0.0)
        probs = np.array([c / n_draws for c in per_skill_h[k].values()])
        recovery["per_skill"].append({
            "skill": k, "true_h_hash": true_hashes[k],
            "closure": closure, "incomparable": incomparable,
            "transitive_reduction": reduction,
            "thresholded_matrix_is_valid_poset": bool(thresh_valid),
            "modal_h_hash": modal_hash,
            "modal_h_posterior": modal_count / n_draws,
            "modal_h_equals_truth": modal_hash == true_hashes[k],
            "true_h_posterior": per_skill_h[k].get(true_hashes[k], 0)
            / n_draws,
            "true_h_rank": 1 + sum(
                1 for h, c in per_skill_h[k].items()
                if c > per_skill_h[k].get(true_hashes[k], 0)),
            "h_entropy_nats": float(-(probs * np.log(probs)).sum()),
            "n_states_above_1pct": int((probs > 0.01).sum()),
            "n_distinct_states": len(per_skill_h[k]),
            "min_true_relation_marginal": float(true_vals.min())
            if true_vals.size else None,
            "max_false_relation_marginal": float(false_vals.max())
            if false_vals.size else None,
        })
    joint_true = tuple(true_hashes)
    recovery["joint"] = {
        "true_tuple_posterior": joint_h.get(joint_true, 0) / n_draws,
        "true_tuple_rank": 1 + sum(1 for h, c in joint_h.items()
                                   if c > joint_h.get(joint_true, 0)),
        "n_distinct_tuples": len(joint_h),
        "min_true_relation_marginal_overall": min_true,
        "max_false_relation_marginal_overall": max_false,
        "n_pooled_draws": n_draws,
    }

    # ---------------------------------------------------- held-out oracle NLL
    heldout_blocks = mcb.oracle_blocks_by_skill(corpus.heldout, K)
    heldout_lik = mcb.OracleBlockLikelihood(
        heldout_blocks, truth.beta, truth.epsilon, truth.omega,
        truth.lambda_rep, truth.lambda_back)
    n_occ = sum(t.length for t in corpus.heldout)

    def nll_of(u_by_skill):
        return -heldout_lik.total(u_by_skill) / n_occ

    # every visited canonical H gets one closure-driven held-out replay; the
    # closure is reconstructed from the retained indicator draws.
    indicator_rows = np.concatenate(
        [np.asarray(r["indicators"], dtype=bool) for r in rows], axis=0)
    hash_to_blocks: list = [{} for _ in range(K)]
    for d, h in enumerate(h_draws):
        for k in range(K):
            if h[k] not in hash_to_blocks[k]:
                closure = _closure_from_indicators(indicator_rows[d], k, m)
                hash_to_blocks[k][h[k]] = _per_block_for_closure(
                    heldout_blocks[k], closure, truth)
    hash_to_ll = [{h: float(blocks.sum())
                   for h, blocks in hash_to_blocks[k].items()}
                  for k in range(K)]
    per_draw_nll = np.array([
        -sum(hash_to_ll[k][h[k]] for k in range(K)) / n_occ for h in h_draws])
    # posterior predictive: per block, mixture over per-skill H frequencies
    predictive_nll_total = 0.0
    for k in range(K):
        freqs = {h: c / n_draws for h, c in per_skill_h[k].items()}
        n_blocks = len(next(iter(hash_to_blocks[k].values()))) \
            if hash_to_blocks[k] else 0
        for b in range(n_blocks):
            mix = sum(freqs[h] * math.exp(hash_to_blocks[k][h][b])
                      for h in freqs)
            predictive_nll_total += -math.log(mix)
    predictive_nll = predictive_nll_total / n_occ

    truth_nll = nll_of(truth.u_by_skill)
    total_order_u = np.array([[[5.0 - r] * truth.latent_dim
                               for r in range(m)]] * K)
    antichain_u = np.zeros_like(truth.u_by_skill)
    heldout = {
        "n_heldout_traces": len(corpus.heldout), "n_occurrences": n_occ,
        "scored_with": "held-out oracle S*, z* (no boundary or label "
                       "inference)",
        "generating_truth_nll_per_occ": truth_nll,
        "posterior_mean_of_nll_per_occ": float(per_draw_nll.mean()),
        "posterior_nll_sd": float(per_draw_nll.std(ddof=1)),
        "posterior_predictive_nll_per_occ": predictive_nll,
        "modal_h_nll_per_occ": -sum(
            hash_to_ll[k][per_skill_h[k].most_common(1)[0][0]]
            for k in range(K)) / n_occ,
        "total_order_baseline_nll_per_occ": nll_of(total_order_u),
        "antichain_baseline_nll_per_occ": nll_of(antichain_u),
    }
    return {"recovery": recovery, "heldout": heldout,
            "per_skill_marginals": per_skill_marginals,
            "true_hashes": true_hashes}


def features_from_closure(role_array, closure, omega):
    """The production `vectorized_state_features` arithmetic, driven by the
    closure directly. The production function reads its ``u`` argument only
    through ``precedence_from_u(u)``, so this is the same computation with the
    closure passed in; a test pins the two to exact equality on random U."""
    from hpop.mcmc_original.recurrent_joint_scalar_mcmc import sigmoid
    precedence = np.asarray(closure, dtype=bool)
    roles = np.asarray(role_array, dtype=int)
    m = precedence.shape[0]
    n, T = roles.shape
    kappa = sigmoid(omega)
    pred_mask = precedence.T.astype(bool)
    succ = precedence.astype(float)
    succ_off = succ.copy()
    np.fill_diagonal(succ_off, 0.0)
    F = np.empty((n, T, m))
    Q = np.empty((n, T, m))
    QV = np.empty((n, T, m))
    CB = np.empty((n, T, m))
    q = np.zeros((n, m))
    rows = np.arange(n)
    for t in range(T):
        F[:, t, :] = np.prod(np.where(pred_mask[None, :, :], q[:, None, :],
                                      1.0), axis=2)
        Q[:, t, :] = np.log1p((1.0 - q) @ succ.T)
        QV[:, t, :] = q
        CB[:, t, :] = kappa * (q @ succ_off.T)
        observed = roles[:, t]
        gate = np.where(precedence[observed], kappa, 0.0)
        q = q * (1.0 - gate)
        q[rows, observed] = 1.0
    return {"F": F, "Q": Q, "q": QV, "C_back": CB, "obs": roles.copy(),
            "m": m, "omega": float(omega)}


def _closure_from_indicators(indicator_row, k, m):
    closure = np.zeros((m, m), dtype=bool)
    base = k * m * (m - 1)
    pos = 0
    for i in range(m):
        for j in range(m):
            if i != j:
                closure[i, j] = indicator_row[base + pos]
                pos += 1
    return closure


def _per_block_for_closure(width_groups, closure, truth):
    parts = []
    for arr in width_groups.values():
        parts.append(mcb._per_block_log_likelihood(
            features_from_closure(arr, closure, truth.omega), truth.beta,
            truth.epsilon, truth.lambda_rep, truth.lambda_back))
    return np.concatenate(parts) if parts else np.zeros(0)


# ===================================================================== verdict
def classify(formal: dict, analysis: dict) -> dict:
    if not formal["converged"]:
        return {"classification": "CONDITION B INFERENCE NOT CONVERGED — NO "
                                  "RECOVERY CLAIM",
                "rule": VERDICT_RULE, "converged": False}
    per_skill = analysis["recovery"]["per_skill"]
    joint = analysis["recovery"]["joint"]
    heldout = analysis["heldout"]
    closure_f1 = [s["closure"]["f1"] for s in per_skill]
    incomparable_f1 = [s["incomparable"]["f1"] for s in per_skill]
    modal_hits = sum(s["modal_h_equals_truth"] for s in per_skill)
    observed = {
        "per_skill_closure_f1": closure_f1,
        "per_skill_incomparable_f1": incomparable_f1,
        "modal_h_equals_truth_count": modal_hits,
        "min_true_relation_marginal": joint["min_true_relation_marginal_overall"],
        "max_false_relation_marginal": joint["max_false_relation_marginal_overall"],
        "heldout_posterior_mean_nll": heldout["posterior_mean_of_nll_per_occ"],
        "heldout_truth_nll": heldout["generating_truth_nll_per_occ"],
        "heldout_antichain_nll": heldout["antichain_baseline_nll_per_occ"],
        "heldout_total_order_nll": heldout["total_order_baseline_nll_per_occ"],
    }
    post_nll = observed["heldout_posterior_mean_nll"]
    strong = (all(f >= 0.90 for f in closure_f1)
              and all(f >= 0.80 for f in incomparable_f1)
              and modal_hits >= 2
              and observed["min_true_relation_marginal"] >= 0.5
              and observed["max_false_relation_marginal"] < 0.5
              and post_nll <= observed["heldout_antichain_nll"]
              and post_nll <= observed["heldout_total_order_nll"]
              and post_nll - observed["heldout_truth_nll"] <= 0.05)
    not_identifiable = (float(np.mean(closure_f1)) < 0.60
                        or post_nll > observed["heldout_antichain_nll"])
    if strong:
        label = "STRUCTURE STRONGLY IDENTIFIABLE UNDER ORACLE PATHS"
    elif not_identifiable:
        label = "STRUCTURE NOT IDENTIFIABLE UNDER ORACLE PATHS"
    else:
        label = "STRUCTURE PARTIALLY IDENTIFIABLE UNDER ORACLE PATHS"
    return {"classification": label, "converged": True,
            "levels": {
                "A_kernel_mixing": "PASS",
                "B_structural_identifiability": label,
                "C_hyperparameter_identifiability":
                    "NOT EVALUATED — rho fixed at rho_0 by registered "
                    "correction (rho* = null in the supplied-truth corpus)"},
            "observed": observed, "rule": VERDICT_RULE}


def main() -> int:
    wall0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    print("== phase 0/1: preconditions and protocol freeze ==")
    pre = verify_preconditions()
    freeze_protocol(pre)
    _dump("pilot_registration.json", json.loads(
        (OUT / "preregistration.json").read_text())["pilot"])
    truth, corpus = build_environment()
    target = build_target(truth, corpus)

    print("== phase 2: target/scorer parity ==")
    parity = run_parity_checks(truth, corpus, target)
    print("== phase 3: prior-only reference check ==")
    prior_check = run_prior_reference_check(truth)
    _dump("correctness.json", {"target_parity": parity,
                               "prior_reference_check": prior_check})
    if not (parity["pass"] and prior_check["pass"]):
        print("CORRECTNESS FAILURE — STOP")
        return 1

    print("== phase 4: efficiency-only pilot ==")
    pilot_results, selected = run_pilot(truth, corpus)
    _dump("pilot_results.json", {"grid": pilot_results,
                                 "inspected": ["acceptance", "ESJD", "H-change "
                                               "frequency", "sweeps/s"],
                                 "recovery_inspected": False})
    _dump("selected_scales.json", selected)
    print(f"  selected sigma_u = {selected['sigma_u']}")

    print("== phase 5: dispersed starts ==")
    starts = [make_start(truth, c) for c in range(4)]
    start_hashes = [h_tuple_hashes(s) for s in starts]
    truth_hashes = h_tuple_hashes(truth.u_by_skill)
    if len(set(start_hashes)) != 4:
        raise SystemExit("start H tuples are not distinct")
    if any(h == truth_hashes for h in start_hashes):
        raise SystemExit("a start coincides with the truth H tuple")
    _dump("start_manifest.json", {
        "start_seeds": list(START_SEEDS), "start_scales": list(START_SCALES),
        "start_h_hashes": [list(h) for h in start_hashes],
        "truth_h_hashes": list(truth_hashes),
        "distinct": True, "none_equals_truth": True,
        "relation_counts": [[int(precedence_from_u(s[k]).sum())
                             for k in range(3)] for s in starts],
        "initialized_at_truth": False,
        "initialized_from_prior_chain_endpoints": False})

    print("== phase 6: formal chains (4 parallel) ==")
    formal = run_formal(truth, corpus, selected["sigma_u"])
    _dump("convergence.json", {
        "checkpoint_log": formal["checkpoint_log"],
        "final_gate_checks": formal["final_gate"]["checks"],
        "final_summary_diagnostics": {
            k: v for k, v in formal["final_gate"]["summaries"].items()},
        "uncertain_relation_diagnostics":
            formal["final_gate"]["uncertain_relations"],
        "converged": formal["converged"],
        "stopped_at": formal["stopped_at"],
        "stop_reason": formal["stop_reason"]})
    _dump("structural_movement.json", {
        f"chain{c}": {
            **{k: v for k, v in formal["movement"][c].items()},
            "acceptance": formal["movement"][c]["accepted"]
            / formal["movement"][c]["proposed"],
            "esjd_per_proposal": formal["movement"][c]["esjd_sum"]
            / formal["movement"][c]["proposed"],
            "retained_h_changes": int(sum(
                1 for a, b in zip(formal["rows"][c]["h_hashes"][:-1],
                                  formal["rows"][c]["h_hashes"][1:])
                if a != b)),
            "distinct_canonical_h_tuples": len(set(
                formal["rows"][c]["h_hashes"])),
        } for c in range(4)})

    print("== phase 7/8: recovery analysis and held-out prediction ==")
    analysis = analyse(truth, corpus, formal)
    np.savez_compressed(OUT / "relation_marginals.npz",
                        per_skill_marginals=analysis["per_skill_marginals"],
                        true_closures=np.array([
                            precedence_from_u(truth.u_by_skill[k])
                            for k in range(truth.n_skills)]))
    _dump("structure_recovery.json", analysis["recovery"])
    _dump("rho_posterior.json", {
        "rho_inferred": False, "rho_fixed_at": RHO_0,
        "note": "registered correction: the supplied-truth corpus records "
                "rho* = null, so rho is fixed and no rho-recovery claim is "
                "made; a separate prior-draw calibration experiment (rho* ~ "
                "p(rho), U* ~ p(U|rho*)) may later evaluate rho and is "
                "explicitly out of scope here"})
    _dump("heldout_oracle_path_nll.json", analysis["heldout"])

    verdict = classify(formal, analysis)
    _dump("final_verdict.json", verdict)
    total_secs = sum(m["segment_seconds"] for m in formal["movement"])
    retained_total = sum(len(r["log_posterior"]) for r in formal["rows"])
    lp = np.concatenate([np.asarray(r["log_posterior"])
                         for r in formal["rows"]])
    _dump("runtime.json", {
        "wall_seconds_total": time.perf_counter() - wall0,
        "chain_seconds_sum": total_secs,
        "sweeps_per_chain": formal["stopped_at"],
        "retained_draws_total": retained_total,
        "structural_ess_per_second": (
            formal["final_gate"]["checks"]["total_relations_bulk_ess"]
            / total_secs if math.isfinite(
                formal["final_gate"]["checks"]["total_relations_bulk_ess"])
            else None),
        "mean_log_posterior": float(lp.mean())})

    r = analysis["recovery"]
    h = analysis["heldout"]
    (OUT / "report.md").write_text("\n".join([
        "# Condition B — structure identifiability under oracle paths",
        "",
        f"Parent commit `{_git('rev-parse', 'HEAD')}` &middot; corpus "
        f"`{pre['corpus_hash_sha256'][:16]}…` &middot; rho fixed at "
        f"{RHO_0} (registered correction; rho* = null)",
        "",
        f"## Classification: **{verdict['classification']}**",
        "",
        f"- stopped at {formal['stopped_at']} sweeps ({formal['stop_reason']})",
        f"- max R-hat {formal['final_gate']['checks']['max_rhat']:.4f}, "
        f"log-posterior bulk ESS "
        f"{formal['final_gate']['checks']['log_posterior_bulk_ess']:.0f}",
        f"- selected sigma_u {selected['sigma_u']} "
        f"(acceptance {selected['acceptance']:.3f})",
        "",
        "| skill | closure F1 | incomparable F1 | reduction F1 | p(H*) | "
        "modal = H* |",
        "|---|---|---|---|---|---|"] + [
        f"| {s['skill']} | {s['closure']['f1']:.3f} | "
        f"{s['incomparable']['f1']:.3f} | "
        f"{s['transitive_reduction']['f1']:.3f} | "
        f"{s['true_h_posterior']:.3f} | {s['modal_h_equals_truth']} |"
        for s in r["per_skill"]] + [
        "",
        f"- min true-relation marginal "
        f"{r['joint']['min_true_relation_marginal_overall']:.3f}; max false "
        f"{r['joint']['max_false_relation_marginal_overall']:.3f}",
        f"- held-out oracle-path NLL/occ: posterior mean "
        f"{h['posterior_mean_of_nll_per_occ']:.4f}, predictive "
        f"{h['posterior_predictive_nll_per_occ']:.4f}, truth "
        f"{h['generating_truth_nll_per_occ']:.4f}, total-order "
        f"{h['total_order_baseline_nll_per_occ']:.4f}, antichain "
        f"{h['antichain_baseline_nll_per_occ']:.4f}",
        "",
        "Raw entrywise U error is not evaluated: the latent product-order "
        "coordinates are exchangeable and the likelihood reads U only through "
        "h(U). Level C (rho) is not evaluated by registered correction.",
        "",
        "STOPPED as registered: no Condition C/D, no FFBS, no collapsed-U, "
        "no scalar inference.",
    ]) + "\n")
    print(f"\nCLASSIFICATION: {verdict['classification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
