"""End-to-end runner for the Stage 0-3 toy validation of the original model.

    PYTHONPATH=src python scripts/toy_stage0_to_stage3.py
    PYTHONPATH=src python scripts/toy_stage0_to_stage3.py --run-optional-stage3c
    PYTHONPATH=src python scripts/toy_stage0_to_stage3.py --continue-on-failure

Stages run in order and the script stops at the first failure unless
``--continue-on-failure`` is passed (debug only). Results are written to
``results/mcmc_original/toy_stage0_to_stage3.json`` and a markdown report.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import toy  # noqa: E402
from hpop.mcmc_original.diagnostics import (  # noqa: E402
    autocorrelation,
    boundary_marginals_from_samples,
    effective_sample_size,
    prf1,
    relation_posterior,
    rhat,
    segmentation_frequencies,
    total_variation_distance,
)
from hpop.mcmc_original.enumerate import (  # noqa: E402
    boundary_marginals_from_probs,
    build_trace_states,
    exact_state_table,
)
from hpop.mcmc_original.latent_poset import precedence_from_u  # noqa: E402
from hpop.mcmc_original.sampler_segmentation import (  # noqa: E402
    run_joint_mcmc,
    run_segmentation_mcmc,
)
from hpop.mcmc_original.sampler_u import (  # noqa: E402
    dispersed_initial_u,
    run_u_mcmc,
)
from hpop.mcmc_original.targets import SkillEvaluator  # noqa: E402
from hpop.mcmc_original.transitions import (  # noqa: E402
    allowed_next,
    dirichlet_posterior_params,
    log_transition_matrix,
    posterior_mean_transition_matrix,
    run_segmentation_transition_gibbs,
    sample_transition_matrix,
    transition_counts,
)

# ---------------------------------------------------------------------------
# configuration — every seed and knob in one place
# ---------------------------------------------------------------------------

SEED_STAGE1 = 20260808
SEED_STAGE2A = 20260808
SEED_STAGE2B = 20260808
SEED_STAGE3A = 20260808
SEED_STAGE3C = 20260808

SIGMA_U = 0.8            # calibrated once, see the report
N_CHAINS = 4
STAGE1_ITERATIONS = 100_000
STAGE1_BURN_IN = 5_000
STAGE2_ITERATIONS = 15_000
STAGE2_BURN_IN = 3_000
STAGE2_THIN = 3
N_DIRICHLET_DRAWS = 50_000
STAGE3C_ITERATIONS = 4_000
STAGE3C_BURN_IN = 1_000
STAGE3C_THIN = 2

P_BA_EXACT = 0.983050847457627
P_AB_EXACT = 0.016949152542373
E_31_34 = 31.0 / 34.0
E_3_34 = 3.0 / 34.0

K2 = 2
K3 = 3


class StageFailure(Exception):
    """Raised when a stage's acceptance criterion is not met."""

    def __init__(self, stage, expected, observed, config):
        super().__init__(f"{stage}: expected {expected}, observed {observed}")
        self.stage = stage
        self.expected = expected
        self.observed = observed
        self.config = config


def check(condition, stage, expected, observed, config):
    if not condition:
        raise StageFailure(stage, expected, observed, config)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage0() -> dict:
    skills = toy.stage012_skills()
    evaluators = [SkillEvaluator(s) for s in skills]
    trace_states = build_trace_states(toy.PRIMARY_TRACE, skills, evaluators, toy.DELTA_B)
    tables = [evaluators[k].log_table(skills[k].u) for k in range(K2)]
    table = exact_state_table(trace_states, tables, toy.uniform_log_pi(K2))

    by_path = {p: float(v) for p, v in zip(trace_states.paths, table["probs"])}
    marginals = boundary_marginals_from_probs(trace_states, table["probs"])
    config = {"trace": toy.PRIMARY_TRACE, "delta_B": toy.DELTA_B, "beta": toy.BETA,
              "epsilon": toy.EPSILON}

    check(trace_states.n_states == 2, "Stage 0", "exactly 2 support-compatible states",
          trace_states.n_states, config)
    p_ba = by_path[(toy.SKILL_B, toy.SKILL_A)]
    p_ab = by_path[(toy.SKILL_A, toy.SKILL_B)]
    check(abs(p_ba - P_BA_EXACT) < 1e-10, "Stage 0", P_BA_EXACT, p_ba, config)
    check(abs(p_ab - P_AB_EXACT) < 1e-10, "Stage 0", P_AB_EXACT, p_ab, config)
    check(abs(table["probs"].sum() - 1.0) < 1e-12, "Stage 0", 1.0,
          float(table["probs"].sum()), config)
    check(abs(marginals[2] - P_BA_EXACT) < 1e-10, "Stage 0 boundary B_3",
          P_BA_EXACT, float(marginals[2]), config)
    check(abs(marginals[1] - P_AB_EXACT) < 1e-10, "Stage 0 boundary B_2",
          P_AB_EXACT, float(marginals[1]), config)

    return {
        "trace": list(toy.PRIMARY_TRACE),
        "n_states": trace_states.n_states,
        "states": [
            {
                "path": ["ABC"[g.skill] for g in seg.segments],
                "spans": [[g.start, g.end] for g in seg.segments],
                "blocks": [list(toy.PRIMARY_TRACE[g.start:g.end]) for g in seg.segments],
                "log_target": float(lt),
                "probability": float(p),
            }
            for seg, lt, p in zip(
                trace_states.segmentations, table["log_targets"], table["probs"]
            )
        ],
        "P_exact_S_BA": p_ba,
        "P_exact_S_AB": p_ab,
        "sum_probabilities": float(table["probs"].sum()),
        "boundary_marginals": marginals.tolist(),
        "P_B3": float(marginals[2]),
        "P_B2": float(marginals[1]),
        "_internals": {"trace_states": trace_states, "tables": tables,
                       "evaluators": evaluators, "skills": skills,
                       "log_targets": table["log_targets"], "probs": table["probs"]},
    }


def stage1(stage0_result: dict) -> dict:
    internals = stage0_result["_internals"]
    trace_states = internals["trace_states"]
    exact = internals["probs"]

    rng = np.random.default_rng(SEED_STAGE1)
    result = run_segmentation_mcmc(
        internals["log_targets"], STAGE1_ITERATIONS, STAGE1_BURN_IN, rng, init=0
    )
    empirical = segmentation_frequencies(result["kept"], trace_states.n_states)
    tv = total_variation_distance(empirical, exact)
    exact_marginals = boundary_marginals_from_probs(trace_states, exact)
    empirical_marginals = boundary_marginals_from_samples(
        result["kept"], trace_states.cuts, trace_states.length
    )

    by_path_exact = dict(zip(trace_states.paths, exact))
    by_path_mcmc = dict(zip(trace_states.paths, empirical))
    indicator = (result["kept"] == list(trace_states.paths).index(
        (toy.SKILL_B, toy.SKILL_A))).astype(float)
    rho = autocorrelation(indicator, max_lag=20)
    ess = effective_sample_size(indicator)

    config = {"iterations": STAGE1_ITERATIONS, "burn_in": STAGE1_BURN_IN,
              "seed": SEED_STAGE1}
    for path, label in (((toy.SKILL_B, toy.SKILL_A), "S_BA"),
                        ((toy.SKILL_A, toy.SKILL_B), "S_AB")):
        delta = abs(by_path_mcmc[path] - by_path_exact[path])
        check(delta < 0.01, f"Stage 1 {label}", f"|MCMC - exact| < 0.01",
              f"{delta:.5f}", config)
    check(tv < 0.01, "Stage 1 TV", "TV < 0.01", f"{tv:.5f}", config)
    for t, exact_value in ((3, P_BA_EXACT), (2, P_AB_EXACT)):
        delta = abs(empirical_marginals[t - 1] - exact_value)
        check(delta < 0.01, f"Stage 1 boundary B_{t}", "|MCMC - exact| < 0.01",
              f"{delta:.5f}", config)

    return {
        "iterations": STAGE1_ITERATIONS,
        "burn_in": STAGE1_BURN_IN,
        "seed": SEED_STAGE1,
        "acceptance_rate": result["acceptance_rate"],
        "exact": {"S_BA": float(by_path_exact[(toy.SKILL_B, toy.SKILL_A)]),
                  "S_AB": float(by_path_exact[(toy.SKILL_A, toy.SKILL_B)])},
        "mcmc": {"S_BA": float(by_path_mcmc[(toy.SKILL_B, toy.SKILL_A)]),
                 "S_AB": float(by_path_mcmc[(toy.SKILL_A, toy.SKILL_B)])},
        "total_variation": tv,
        "boundary_exact": exact_marginals.tolist(),
        "boundary_mcmc": empirical_marginals.tolist(),
        "autocorrelation_lag1": float(rho[1]),
        "autocorrelation_lag5": float(rho[5]),
        "effective_sample_size": ess,
        "n_kept": int(len(result["kept"])),
    }


def stage2a() -> dict:
    rng = np.random.default_rng(SEED_STAGE2A)
    corpus = toy.make_stage2a_corpus(rng)
    skills = toy.stage012_skills()

    out = {"sigma_U": SIGMA_U, "iterations": STAGE2_ITERATIONS,
           "burn_in": STAGE2_BURN_IN, "thin": STAGE2_THIN, "n_chains": N_CHAINS,
           "seed": SEED_STAGE2A, "rho_U": toy.RHO_U, "latent_dim": toy.LATENT_DIM,
           "skills": {}}
    config = {k: out[k] for k in ("sigma_U", "iterations", "burn_in", "thin", "seed")}

    true_relations = {toy.SKILL_A: [(0, 1)], toy.SKILL_B: [(0, 1), (1, 2), (0, 2)]}
    for skill_id, name in ((toy.SKILL_A, "A"), (toy.SKILL_B, "B")):
        evaluator = SkillEvaluator(skills[skill_id])
        counts = evaluator.count_sequences(corpus[skill_id])
        chains = []
        for chain in range(N_CHAINS):
            chain_rng = np.random.default_rng(SEED_STAGE2A + 100 * (skill_id + 1) + chain)
            init = dispersed_initial_u(skills[skill_id].u.shape, toy.RHO_U, chain_rng)
            chains.append(run_u_mcmc(evaluator, counts, toy.RHO_U, SIGMA_U,
                                     STAGE2_ITERATIONS, STAGE2_BURN_IN, STAGE2_THIN,
                                     chain_rng, init))

        pooled = relation_posterior(np.concatenate([c["samples"] for c in chains]))
        per_chain = [relation_posterior(c["samples"]) for c in chains]
        log_post_rhat = rhat(np.array([c["log_posterior_kept"] for c in chains]))

        for i, j in true_relations[skill_id]:
            check(pooled[i, j] > 0.90, f"Stage 2A skill {name} relation {i}>{j}",
                  "P > 0.90", f"{pooled[i, j]:.4f}", config)
        for chain_index, chain in enumerate(chains):
            rate = chain["acceptance_rate"]
            check(0.10 < rate < 0.60, f"Stage 2A skill {name} chain {chain_index}",
                  "0.10 < acceptance < 0.60", f"{rate:.4f}", config)

        out["skills"][name] = {
            "n_sequences": int(counts.sum()),
            "permutation_counts": counts.astype(int).tolist(),
            "permutations": [list(p) for p in evaluator.permutations],
            "true_relations": [list(r) for r in true_relations[skill_id]],
            "acceptance_by_chain": [c["acceptance_rate"] for c in chains],
            "relation_posterior_pooled": pooled.tolist(),
            "relation_posterior_by_chain": [p.tolist() for p in per_chain],
            "log_posterior_rhat": log_post_rhat,
            "log_posterior_ess_by_chain": [
                effective_sample_size(c["log_posterior_kept"]) for c in chains
            ],
            "n_saved_per_chain": int(len(chains[0]["samples"])),
            "cache_hits": evaluator.cache_hits,
            "cache_misses": evaluator.cache_misses,
        }
    return out


def stage2b() -> dict:
    rng = np.random.default_rng(SEED_STAGE2B)
    corpus = toy.make_stage2b_corpus(rng)
    skills = toy.stage012_skills()
    evaluators = [SkillEvaluator(s) for s in skills]
    trace_states = [
        build_trace_states(c["x"], skills, evaluators, toy.DELTA_B,
                           c["true_cut"], c["true_path"])
        for c in corpus
    ]
    log_pi = toy.uniform_log_pi(K2)
    true_relations = {toy.SKILL_A: {(0, 1)}, toy.SKILL_B: {(0, 1), (1, 2), (0, 2)}}

    config = {"iterations": STAGE2_ITERATIONS, "burn_in": STAGE2_BURN_IN,
              "thin": STAGE2_THIN, "sigma_U": SIGMA_U, "seed": SEED_STAGE2B,
              "n_traces": len(corpus)}

    chains = []
    for chain in range(N_CHAINS):
        chain_rng = np.random.default_rng(SEED_STAGE2B + 900 + chain)
        init_u = {k: dispersed_initial_u(skills[k].u.shape, toy.RHO_U, chain_rng)
                  for k in range(K2)}
        chains.append(run_joint_mcmc(trace_states, evaluators, init_u, toy.RHO_U,
                                     SIGMA_U, STAGE2_ITERATIONS, STAGE2_BURN_IN,
                                     STAGE2_THIN, chain_rng, log_pi))

    per_chain = []
    for chain in chains:
        states = chain["states"]
        n_draws = states.shape[0]
        predicted, truth = set(), set()
        correct_paths = 0
        for t, ts in enumerate(trace_states):
            truth.add((t, ts.true_cut))
            frequencies = np.bincount(states[:, t], minlength=ts.n_states) / n_draws
            marginal: dict[int, float] = {}
            for s, p in enumerate(frequencies):
                for cut in ts.cuts[s]:
                    marginal[cut] = marginal.get(cut, 0.0) + p
            for cut, p in marginal.items():
                if p > 0.5:
                    predicted.add((t, cut))
            if ts.paths[int(np.argmax(frequencies))] == ts.true_path:
                correct_paths += 1
        per_chain.append({
            "boundary": prf1(predicted, truth),
            "skill_path_accuracy": correct_paths / len(trace_states),
            "segmentation_acceptance_rate": chain["segmentation_acceptance_rate"],
            "u_acceptance_rate": chain["u_acceptance_rate"],
        })

    pooled_relations, relation_truth = set(), set()
    relation_posteriors = {}
    for skill_id, name in ((toy.SKILL_A, "A"), (toy.SKILL_B, "B")):
        pooled = relation_posterior(
            np.concatenate([c["u_samples"][skill_id] for c in chains])
        )
        relation_posteriors[name] = pooled.tolist()
        for i, j in true_relations[skill_id]:
            relation_truth.add((skill_id, i, j))
        for i in range(pooled.shape[0]):
            for j in range(pooled.shape[0]):
                if i != j and pooled[i, j] > 0.5:
                    pooled_relations.add((skill_id, i, j))
    relation_scores = prf1(pooled_relations, relation_truth)

    for c, summary in enumerate(per_chain):
        check(summary["boundary"]["f1"] >= 0.85, f"Stage 2B chain {c} boundary F1",
              ">= 0.85", f"{summary['boundary']['f1']:.4f}", config)
    check(relation_scores["f1"] >= 0.85, "Stage 2B relation F1", ">= 0.85",
          f"{relation_scores['f1']:.4f}", config)

    return {
        "n_traces": len(corpus),
        "n_traces_ba": sum(1 for c in corpus if c["true_path"] == (toy.SKILL_B, toy.SKILL_A)),
        "n_traces_ab": sum(1 for c in corpus if c["true_path"] == (toy.SKILL_A, toy.SKILL_B)),
        "state_count_histogram": np.bincount([t.n_states for t in trace_states]).tolist(),
        "n_ambiguous_traces": sum(1 for t in trace_states if t.n_states > 1),
        "sigma_U": SIGMA_U,
        "iterations": STAGE2_ITERATIONS,
        "burn_in": STAGE2_BURN_IN,
        "thin": STAGE2_THIN,
        "n_chains": N_CHAINS,
        "seed": SEED_STAGE2B,
        "by_chain": per_chain,
        "boundary_f1_min": min(s["boundary"]["f1"] for s in per_chain),
        "skill_path_accuracy_min": min(s["skill_path_accuracy"] for s in per_chain),
        "relation_posterior": relation_posteriors,
        "relation_scores": relation_scores,
    }


def stage3a() -> dict:
    counts = np.zeros((K3, K3), dtype=float)
    counts[toy.SKILL_B, toy.SKILL_A] = 30
    counts[toy.SKILL_B, toy.SKILL_C] = 2
    counts[toy.SKILL_A, toy.SKILL_B] = 2
    counts[toy.SKILL_A, toy.SKILL_C] = 30

    params = dirichlet_posterior_params(counts, K3)
    analytic = posterior_mean_transition_matrix(counts, K3)
    rng = np.random.default_rng(SEED_STAGE3A)
    draws = np.array([sample_transition_matrix(counts, K3, rng)
                      for _ in range(N_DIRICHLET_DRAWS)])
    empirical = draws.mean(axis=0)

    config = {"draws": N_DIRICHLET_DRAWS, "seed": SEED_STAGE3A, "eta": 1.0}
    names = "ABC"
    for h in range(K3):
        for k in allowed_next(h, K3):
            delta = abs(empirical[h, k] - analytic[h, k])
            check(delta < 0.01, f"Stage 3A P[{names[h]},{names[k]}]",
                  f"{analytic[h, k]:.6f} (+/- 0.01)", f"{empirical[h, k]:.6f}", config)
    check(abs(analytic[toy.SKILL_B, toy.SKILL_A] - E_31_34) < 1e-12,
          "Stage 3A analytic mean", E_31_34,
          float(analytic[toy.SKILL_B, toy.SKILL_A]), config)

    return {
        "counts": counts.astype(int).tolist(),
        "n_draws": N_DIRICHLET_DRAWS,
        "seed": SEED_STAGE3A,
        "dirichlet_parameters": {
            names[h]: {"allowed_next": [names[k] for k in allowed],
                       "alpha": alpha.tolist()}
            for h, (allowed, alpha) in params.items()
        },
        "analytic_means": analytic.tolist(),
        "empirical_means": empirical.tolist(),
        "max_abs_error": float(
            max(abs(empirical[h, k] - analytic[h, k])
                for h in range(K3) for k in allowed_next(h, K3))
        ),
        "_counts": counts,
    }


def stage3b(stage3a_result: dict) -> dict:
    skills = toy.stage3_skills()
    toy.assert_stage3_b_is_antichain()
    evaluators = [SkillEvaluator(s) for s in skills]
    trace_states = build_trace_states(toy.PRIMARY_TRACE, skills, evaluators, toy.DELTA_B)
    tables = [evaluators[k].log_table(skills[k].u) for k in range(K3)]
    log_pi = toy.uniform_log_pi(K3)
    config = {"trace": toy.PRIMARY_TRACE, "U_B": "U_B_ANTICHAIN"}

    before = exact_state_table(trace_states, tables, log_pi)
    before_by_path = {p: float(v) for p, v in zip(trace_states.paths, before["probs"])}
    for path, label in (((toy.SKILL_B, toy.SKILL_A), "S_BA"),
                        ((toy.SKILL_A, toy.SKILL_B), "S_AB")):
        check(abs(before_by_path[path] - 0.5) < 1e-12, f"Stage 3B before {label}",
              0.5, before_by_path[path], config)

    means = posterior_mean_transition_matrix(stage3a_result["_counts"], K3)
    after = exact_state_table(trace_states, tables, log_pi, log_transition_matrix(means))
    after_by_path = {p: float(v) for p, v in zip(trace_states.paths, after["probs"])}
    check(abs(after_by_path[(toy.SKILL_B, toy.SKILL_A)] - E_31_34) < 1e-12,
          "Stage 3B after S_BA", E_31_34,
          after_by_path[(toy.SKILL_B, toy.SKILL_A)], config)
    check(abs(after_by_path[(toy.SKILL_A, toy.SKILL_B)] - E_3_34) < 1e-12,
          "Stage 3B after S_AB", E_3_34,
          after_by_path[(toy.SKILL_A, toy.SKILL_B)], config)

    return {
        "trace": list(toy.PRIMARY_TRACE),
        "U_B_used": "U_B_ANTICHAIN",
        "before": {"S_BA": before_by_path[(toy.SKILL_B, toy.SKILL_A)],
                   "S_AB": before_by_path[(toy.SKILL_A, toy.SKILL_B)]},
        "after": {"S_BA": after_by_path[(toy.SKILL_B, toy.SKILL_A)],
                  "S_AB": after_by_path[(toy.SKILL_A, toy.SKILL_B)]},
        "expected_after": {"S_BA": E_31_34, "S_AB": E_3_34},
        "transition_means_used": means.tolist(),
    }


def stage3c() -> dict:
    rng = np.random.default_rng(SEED_STAGE3C)
    corpus = toy.make_stage3c_corpus(rng)
    skills = toy.stage3_skills()
    evaluators = [SkillEvaluator(s) for s in skills]
    trace_states = [
        build_trace_states(c["x"], skills, evaluators, toy.DELTA_B,
                           c["true_cut"], c["true_path"])
        for c in corpus
    ]
    u_by_skill = {k: skills[k].u for k in range(K3)}
    log_pi = toy.uniform_log_pi(K3)

    chain_rng = np.random.default_rng(SEED_STAGE3C + 1)
    result = run_segmentation_transition_gibbs(
        trace_states, evaluators, u_by_skill, K3, log_pi,
        STAGE3C_ITERATIONS, STAGE3C_BURN_IN, STAGE3C_THIN, chain_rng
    )
    p_mean = result["p_mean"]
    true_counts = transition_counts([c["true_path"] for c in corpus], K3)

    prefers_a = bool(p_mean[toy.SKILL_B, toy.SKILL_A] > p_mean[toy.SKILL_B, toy.SKILL_C])
    prefers_c = bool(p_mean[toy.SKILL_A, toy.SKILL_C] > p_mean[toy.SKILL_A, toy.SKILL_B])

    return {
        "n_traces": len(corpus),
        "true_counts": true_counts.astype(int).tolist(),
        "iterations": STAGE3C_ITERATIONS,
        "burn_in": STAGE3C_BURN_IN,
        "thin": STAGE3C_THIN,
        "seed": SEED_STAGE3C,
        "acceptance_rate": result["acceptance_rate"],
        "posterior_mean_P": p_mean.tolist(),
        "posterior_mean_counts": result["count_mean"].tolist(),
        "n_ambiguous_traces": sum(1 for t in trace_states if t.n_states > 1),
        "B_prefers_A": prefers_a,
        "A_prefers_C": prefers_c,
        "passed": prefers_a and prefers_c,
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:  # pragma: no cover - git may be unavailable
            return "unknown"
    return {"branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "commit": run("git", "rev-parse", "HEAD"),
            "dirty": run("git", "status", "--porcelain") != ""}


def matrix_block(matrix, labels=None) -> str:
    matrix = np.asarray(matrix)
    n = matrix.shape[0]
    labels = labels or [str(i) for i in range(n)]
    head = "      " + " ".join(f"{c:>7}" for c in labels)
    rows = [f"  {labels[i]:>3} " + " ".join(
        ("      ." if not matrix[i, j] else "      1") if matrix.dtype == bool
        else f"{matrix[i, j]:>7.4f}" for j in range(n)) for i in range(n)]
    return "\n".join([head, *rows])


def write_report(results: dict, decisions: list, path: Path) -> None:
    info = results["environment"]
    lines: list[str] = []
    add = lines.append

    add("# Toy validation of the original latent partial-order model — Stages 0-3")
    add("")
    add(f"Date: {date.today().isoformat()}")
    add(f"Branch: `{info['git']['branch']}`  Commit: `{info['git']['commit']}`"
        f"{'  (working tree dirty)' if info['git']['dirty'] else ''}")
    add(f"Python {info['python']}, NumPy {info['numpy']}")
    add("")

    add("## PASS / FAIL summary")
    add("")
    add("| stage | result | headline |")
    add("|---|---|---|")
    for decision in decisions:
        add(f"| {decision['stage']} | **{decision['status']}** | {decision['headline']} |")
    add("")

    add("## 1. Model configuration")
    add("")
    add(f"- `beta = {toy.BETA}`, `epsilon = {toy.EPSILON}`, `delta_B = {toy.DELTA_B}`")
    add(f"- `rho_U = {toy.RHO_U}`, latent dimension `d = {toy.LATENT_DIM}`")
    add(f"- `sigma_U = {SIGMA_U}` (calibrated once; see Stage 2A)")
    add(f"- uniform first-skill prior `pi_k = 1/K`")
    add("")
    add("### RNG seeds")
    add("")
    add("| stage | seed |")
    add("|---|---|")
    for name, seed in (("Stage 1", SEED_STAGE1), ("Stage 2A", SEED_STAGE2A),
                       ("Stage 2B", SEED_STAGE2B), ("Stage 3A", SEED_STAGE3A),
                       ("Stage 3C", SEED_STAGE3C)):
        add(f"| {name} | `{seed}` |")
    add("")
    add("Chains derive their seeds deterministically from the stage seed, so every")
    add("number in this report reproduces exactly.")
    add("")

    add("### Latent matrices and induced orders")
    add("")
    for name, u, note in (
        ("U_A", toy.U_A, "roles (0,1) -> CPA (0,1); 0 > 1"),
        ("U_B_TOTAL", toy.U_B_TOTAL, "roles (0,1,2) -> CPA (0,1,2); 0 > 1 > 2. Stages 0-2B"),
        ("U_B_ANTICHAIN", toy.U_B_ANTICHAIN, "roles (0,1,2) -> CPA (0,1,2); no order at all. Stage 3 only"),
        ("U_C", toy.U_C, "roles (0,1) -> CPA (3,4); 3 > 4"),
    ):
        add(f"**`{name}`** — {note}")
        add("")
        add("```")
        add(np.array2string(np.asarray(u), precision=2))
        add("")
        add("induced precedence h(U)  (1 means row > column)")
        add(matrix_block(precedence_from_u(np.asarray(u))))
        add("```")
        add("")

    if results.get("stage0"):
        s0 = results["stage0"]
        add("## 2. Stage 0 — exact posterior with fixed U")
        add("")
        add(f"Trace `x = {tuple(s0['trace'])}`. Support-compatible complete states: "
            f"**{s0['n_states']}** (asserted; no third state exists).")
        add("")
        add("| state | blocks | log target | exact P(S \\| x) |")
        add("|---|---|---|---|")
        for state in s0["states"]:
            blocks = " + ".join(
                f"{block}_{skill}" for block, skill in zip(state["blocks"], state["path"])
            )
            add(f"| {'->'.join(state['path'])} | {blocks} | {state['log_target']:.6f} "
                f"| {state['probability']:.15f} |")
        add("")
        add(f"- `P_exact(S_BA) = {s0['P_exact_S_BA']:.15f}`  (target 0.983050847457627)")
        add(f"- `P_exact(S_AB) = {s0['P_exact_S_AB']:.15f}`  (target 0.016949152542373)")
        add(f"- `sum_S P(S) = {s0['sum_probabilities']:.15f}`")
        add("")
        add("Boundary marginals (half-open cut positions):")
        add("")
        add(f"- `P(B_3 = 1 | x) = {s0['P_B3']:.15f}`")
        add(f"- `P(B_2 = 1 | x) = {s0['P_B2']:.15f}`")
        add("")
        add("Both states have `L = 2`, so the boundary prior and the uniform label prior")
        add("cancel exactly and the posterior ratio is the BPOP emission ratio:")
        add("`0.9425 * 0.975 = 0.9189375` against `0.975 * 0.01625 = 0.01584375`.")
        add("")

    if results.get("stage1"):
        s1 = results["stage1"]
        add("## 3. Stage 1 — segmentation MCMC against the exact posterior")
        add("")
        add(f"{s1['iterations']:,} iterations, {s1['burn_in']:,} burn-in, "
            f"{s1['n_kept']:,} kept, seed `{s1['seed']}`, "
            f"acceptance rate **{s1['acceptance_rate']:.4f}**.")
        add("")
        add("| state | exact | MCMC | abs error |")
        add("|---|---|---|---|")
        for label in ("S_BA", "S_AB"):
            e, m = s1["exact"][label], s1["mcmc"][label]
            add(f"| {label} | {e:.6f} | {m:.6f} | {abs(m - e):.6f} |")
        add("")
        add(f"**Total variation distance = {s1['total_variation']:.6f}** (criterion < 0.01)")
        add("")
        add("| cut | exact | MCMC | abs error |")
        add("|---|---|---|---|")
        for t in (2, 3):
            e, m = s1["boundary_exact"][t - 1], s1["boundary_mcmc"][t - 1]
            add(f"| B_{t} | {e:.6f} | {m:.6f} | {abs(m - e):.6f} |")
        add("")
        add(f"Indicator `1[S = S_BA]`: autocorrelation lag-1 = {s1['autocorrelation_lag1']:.4f}, "
            f"lag-5 = {s1['autocorrelation_lag5']:.4f}, ESS = {s1['effective_sample_size']:,.0f}.")
        add("")

    if results.get("stage2a"):
        s2a = results["stage2a"]
        add("## 4. Stage 2A — MCMC over latent U, segmentation known")
        add("")
        add(f"{N_CHAINS} chains, {s2a['iterations']:,} iterations, {s2a['burn_in']:,} burn-in, "
            f"thin {s2a['thin']}, `sigma_U = {s2a['sigma_U']}`, `rho_U = {s2a['rho_U']}`, "
            f"`d = {s2a['latent_dim']}`.")
        add("")
        add("Recovery is judged on **induced precedence relations**, never on raw U")
        add("coordinates — U is not identifiable, only h(U) is.")
        add("")
        for name, block in s2a["skills"].items():
            add(f"### Skill {name} — {block['n_sequences']} executions")
            add("")
            add("| permutation | count |")
            add("|---|---|")
            for perm, count in zip(block["permutations"], block["permutation_counts"]):
                if count:
                    add(f"| {tuple(perm)} | {count} |")
            add("")
            add("| chain | acceptance | " + " | ".join(
                f"P({i}>{j})" for i, j in block["true_relations"]) + " |")
            add("|---|---|" + "---|" * len(block["true_relations"]))
            for c, (rate, posterior) in enumerate(
                zip(block["acceptance_by_chain"], block["relation_posterior_by_chain"])
            ):
                cells = " | ".join(f"{posterior[i][j]:.4f}" for i, j in block["true_relations"])
                add(f"| {c} | {rate:.4f} | {cells} |")
            pooled = block["relation_posterior_pooled"]
            cells = " | ".join(f"{pooled[i][j]:.4f}" for i, j in block["true_relations"])
            add(f"| **pooled** | | {cells} |")
            add("")
            add("Pooled relation posterior (row > column):")
            add("")
            add("```")
            add(matrix_block(np.array(pooled)))
            add("```")
            add("")
            add(f"R-hat(log posterior) = {block['log_posterior_rhat']:.4f}; "
                f"ESS(log posterior) by chain = "
                f"{[round(v) for v in block['log_posterior_ess_by_chain']]}.")
            add(f"Likelihood-table cache: {block['cache_misses']} distinct orders visited, "
                f"{block['cache_hits']:,} hits.")
            add("")

    if results.get("stage2b"):
        s2b = results["stage2b"]
        add("## 5. Stage 2B — joint segmentation + U MCMC")
        add("")
        add(f"{s2b['n_traces']} traces ({s2b['n_traces_ba']} true `B -> A`, "
            f"{s2b['n_traces_ab']} true `A -> B`), of which "
            f"**{s2b['n_ambiguous_traces']} are genuinely ambiguous** (2 legal states); "
            f"the rest have a single legal state.")
        add(f"{N_CHAINS} chains, {s2b['iterations']:,} iterations, {s2b['burn_in']:,} burn-in, "
            f"thin {s2b['thin']}.")
        add("")
        add("| chain | boundary P | boundary R | boundary F1 | skill-path acc | seg accept | U accept |")
        add("|---|---|---|---|---|---|---|")
        for c, summary in enumerate(s2b["by_chain"]):
            b = summary["boundary"]
            add(f"| {c} | {b['precision']:.4f} | {b['recall']:.4f} | {b['f1']:.4f} "
                f"| {summary['skill_path_accuracy']:.4f} "
                f"| {summary['segmentation_acceptance_rate']:.4f} "
                f"| {summary['u_acceptance_rate']:.4f} |")
        add("")
        add(f"**Boundary F1 (worst chain) = {s2b['boundary_f1_min']:.4f}** (criterion >= 0.85)")
        add(f"**Skill-path accuracy (worst chain) = {s2b['skill_path_accuracy_min']:.4f}**")
        add("")
        rs = s2b["relation_scores"]
        add(f"**Ordered-pair (precedence) F1 = {rs['f1']:.4f}** "
            f"(precision {rs['precision']:.4f}, recall {rs['recall']:.4f}, "
            f"{rs['true_positive']}/{rs['n_truth']} true relations recovered)")
        add("")
        for name, posterior in s2b["relation_posterior"].items():
            add(f"Pooled relation posterior, skill {name}:")
            add("")
            add("```")
            add(matrix_block(np.array(posterior)))
            add("```")
            add("")

    if results.get("stage3a"):
        s3a = results["stage3a"]
        add("## 6. Stage 3A — Dirichlet transition Gibbs")
        add("")
        add("Manual transition counts:")
        add("")
        add("```")
        add(matrix_block(np.array(s3a["counts"], dtype=float), ["A", "B", "C"]))
        add("```")
        add("")
        add("| row | allowed next | Dirichlet alpha |")
        add("|---|---|---|")
        for h, block in s3a["dirichlet_parameters"].items():
            add(f"| {h} | {', '.join(block['allowed_next'])} | "
                f"({', '.join(f'{a:g}' for a in block['alpha'])}) |")
        add("")
        add(f"{s3a['n_draws']:,} Gibbs draws, seed `{s3a['seed']}`.")
        add("")
        add("| transition | analytic mean | empirical mean | abs error |")
        add("|---|---|---|---|")
        names = "ABC"
        for h in range(K3):
            for k in allowed_next(h, K3):
                a = s3a["analytic_means"][h][k]
                e = s3a["empirical_means"][h][k]
                add(f"| {names[h]} -> {names[k]} | {a:.10f} | {e:.10f} | {abs(e - a):.6f} |")
        add("")
        add(f"Worst absolute error = {s3a['max_abs_error']:.6f} (criterion < 0.01).")
        add("")

    if results.get("stage3b"):
        s3b = results["stage3b"]
        add("## 7. Stage 3B — transition context resolves an ambiguous boundary")
        add("")
        add(f"Trace `x = {tuple(s3b['trace'])}`, using **`{s3b['U_B_used']}`** so that")
        add("every B permutation has probability exactly 1/6 and the two states have")
        add("identical emission terms.")
        add("")
        add("| state | without transitions | with transition context | expected |")
        add("|---|---|---|---|")
        for label in ("S_BA", "S_AB"):
            add(f"| {label} | {s3b['before'][label]:.15f} | {s3b['after'][label]:.15f} "
                f"| {s3b['expected_after'][label]:.15f} |")
        add("")
        add("This is the headline result: a segmentation that the local partial-order")
        add("likelihood cannot distinguish at all (exactly 0.5 / 0.5) is resolved to")
        add("0.9118 / 0.0882 purely by skill-transition context.")
        add("")

    if results.get("stage3c"):
        s3c = results["stage3c"]
        add("## 8. Stage 3C (optional) — joint segmentation + transition Gibbs")
        add("")
        add(f"{s3c['n_traces']} traces, {s3c['n_ambiguous_traces']} ambiguous, "
            f"U held fixed. {s3c['iterations']:,} iterations, {s3c['burn_in']:,} burn-in.")
        add("")
        add("True transition counts:")
        add("")
        add("```")
        add(matrix_block(np.array(s3c["true_counts"], dtype=float), ["A", "B", "C"]))
        add("```")
        add("")
        add("Posterior mean transition matrix:")
        add("")
        add("```")
        add(matrix_block(np.array(s3c["posterior_mean_P"]), ["A", "B", "C"]))
        add("```")
        add("")
        add(f"- B prefers A over C: **{s3c['B_prefers_A']}**")
        add(f"- A prefers C over B: **{s3c['A_prefers_C']}**")
        add("")

    add("## Deviations, warnings and notes")
    add("")
    for note in results["notes"]:
        add(f"- {note}")
    add("")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-optional-stage3c", action="store_true",
                        help="also run the optional Stage 3C joint S+P Gibbs test")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="debug only: keep going after a stage fails")
    args = parser.parse_args()

    out_dir = ROOT / "results" / "mcmc_original"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "toy_stage0_to_stage3.json"
    report_path = out_dir / "toy_stage0_to_stage3_report.md"

    results: dict = {
        "environment": {
            "git": git_info(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "notes": [],
    }
    decisions: list[dict] = []
    failures: list[StageFailure] = []

    def run_stage(name, headline_fn, fn, *fn_args):
        try:
            value = fn(*fn_args)
        except StageFailure as failure:
            decisions.append({"stage": name, "status": "FAIL",
                              "headline": f"expected {failure.expected}, "
                                          f"observed {failure.observed}"})
            print(f"[FAIL] {name}")
            print(f"       stage    : {failure.stage}")
            print(f"       expected : {failure.expected}")
            print(f"       observed : {failure.observed}")
            print(f"       config   : {failure.config}")
            print(f"       report   : {report_path}")
            failures.append(failure)
            return None
        decisions.append({"stage": name, "status": "PASS",
                          "headline": headline_fn(value)})
        print(f"[PASS] {name}")
        return value

    s0 = run_stage("Stage 0 exact posterior",
                   lambda v: f"P(S_BA) = {v['P_exact_S_BA']:.12f}, sum = "
                             f"{v['sum_probabilities']:.12f}",
                   stage0)
    if s0 is None and not args.continue_on_failure:
        return finish(results, decisions, json_path, report_path, failures)
    results["stage0"] = {k: v for k, v in s0.items() if k != "_internals"} if s0 else {}

    s1 = None
    if s0 is not None:
        s1 = run_stage("Stage 1 segmentation MCMC",
                       lambda v: f"TV = {v['total_variation']:.5f}, "
                                 f"acceptance {v['acceptance_rate']:.3f}",
                       stage1, s0)
        if s1 is not None:
            results["stage1"] = s1
    if s1 is None and not args.continue_on_failure:
        return finish(results, decisions, json_path, report_path, failures)

    s2a = run_stage("Stage 2A latent-U MCMC",
                    lambda v: "all true relations recovered; min pooled P = "
                              + f"{min(min(v['skills'][n]['relation_posterior_pooled'][i][j] for i, j in v['skills'][n]['true_relations']) for n in v['skills']):.4f}",
                    stage2a)
    if s2a is not None:
        results["stage2a"] = s2a
    if s2a is None and not args.continue_on_failure:
        return finish(results, decisions, json_path, report_path, failures)

    s2b = run_stage("Stage 2B joint segmentation + U",
                    lambda v: f"boundary F1 = {v['boundary_f1_min']:.4f}, "
                              f"relation F1 = {v['relation_scores']['f1']:.4f}",
                    stage2b)
    if s2b is not None:
        results["stage2b"] = s2b
    if s2b is None and not args.continue_on_failure:
        return finish(results, decisions, json_path, report_path, failures)

    s3a = run_stage("Stage 3A transition Gibbs",
                    lambda v: f"worst abs error vs analytic = {v['max_abs_error']:.5f}",
                    stage3a)
    if s3a is not None:
        results["stage3a"] = {k: v for k, v in s3a.items() if k != "_counts"}
    if s3a is None and not args.continue_on_failure:
        return finish(results, decisions, json_path, report_path, failures)

    if s3a is not None:
        s3b = run_stage("Stage 3B transition ambiguity resolution",
                        lambda v: f"0.5/0.5 -> {v['after']['S_BA']:.6f}/"
                                  f"{v['after']['S_AB']:.6f}",
                        stage3b, s3a)
        if s3b is not None:
            results["stage3b"] = s3b
        if s3b is None and not args.continue_on_failure:
            return finish(results, decisions, json_path, report_path, failures)

    if args.run_optional_stage3c and not failures:
        value = stage3c()
        results["stage3c"] = value
        status = "PASS" if value["passed"] else "FAIL"
        decisions.append({
            "stage": "Stage 3C (optional) joint S+P Gibbs",
            "status": status,
            "headline": f"P(B->A) = {value['posterior_mean_P'][toy.SKILL_B][toy.SKILL_A]:.4f}, "
                        f"P(A->C) = {value['posterior_mean_P'][toy.SKILL_A][toy.SKILL_C]:.4f}",
        })
        print(f"[{status}] Stage 3C (optional) joint S+P Gibbs")
        if not value["passed"]:
            results["notes"].append(
                "Stage 3C (optional) did not recover the intended preferences; "
                "reported as-is, required stages are unaffected."
            )

    return finish(results, decisions, json_path, report_path, failures)


def finish(results, decisions, json_path, report_path, failures) -> int:
    results["notes"].extend([
        "sigma_U was calibrated once to 0.8 (the spec suggested 0.25, which gave "
        "acceptance ~0.80, above the [0.10, 0.60] band). No adaptation happens during "
        "saved sampling.",
        "Stage 1's acceptance rate is low by construction: the posterior is 0.983/0.017, "
        "so most proposals to the minority state are correctly rejected.",
        "Stage 2B's segmentation acceptance rate is likewise low for the same reason, "
        "and about half the traces have a single legal state where the update is a no-op.",
        "The BPOP likelihood depends on U only through h(U), so the likelihood surface is "
        "piecewise constant and the U chains mix by jumping between order regions.",
        "Stage 3C is a weaker test than it looks: a B->A trace is only ambiguous when its "
        "sampled B permutation happens to end in role 2 (probability 1/3 under the "
        "antichain), so most of its 64 traces have a single legal state and the transition "
        "posterior is largely pinned by unambiguous data. Stage 3B remains the sharp, "
        "deterministic test of ambiguity resolution.",
    ])
    results["decisions"] = decisions
    json_path.write_text(json.dumps(jsonable(results), indent=2) + "\n")
    write_report(results, decisions, report_path)

    print()
    print(f"JSON   : {json_path}")
    print(f"Report : {report_path}")
    if failures:
        print(f"\n{len(failures)} stage(s) FAILED")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
