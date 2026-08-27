"""Stage 6E1A — the exact segmentation-only reference, and the kernel's agreement with it.

    PYTHONPATH=src python scripts/stage6e1a_exact_segmentation.py

Every continuous coordinate is **fixed**. The only latent quantities are the segmentation
`S` and the skill labels `z`, so the posterior is a distribution over a finite, fully
enumerable set and there is an exact answer to compare against — no reference sampler, no
Monte Carlo error on the reference side.

## What is being tested, and what is not

This is a **sampler-correctness** result. It asks whether the registered local move kernel,
driven by the registered Hastings ratio, has the registered posterior as its invariant
distribution. It says nothing about recovery: the generating segmentation is used to build
the observed sequence and then never referred to again.

## Three routes, two of which must agree exactly

1. `enumerate_states` lists every legal `(S, z)` and `state_log_weights` scores each by the
   registered decomposition. Summing gives `log Z_enum`.
2. `log_evidence_forward` reaches the same constant by a forward recursion over positions,
   materialising no state. §8's gate is `|log Z_enum - log Z_forward| < 1e-10`.
3. The Stage 6E move kernel samples. Its empirical frequencies are compared with route 1.

Routes 1 and 2 share no code path. Route 3 shares only the block scorer, which both parity
checks in Stage 6E0 already pinned.

## Problem selection is registered, not searched for a favourable answer

§8 asks for a *nondegenerate* problem: `max p(S, z | x) < 0.90` and at least three states
above 0.01. Those are reference-quality criteria — a posterior concentrated on one state
would let a broken kernel pass — so the selection rule is registered here before any
sampling: scan the candidate seeds in ascending order and take the **first** that satisfies
both. No recovery quantity enters the choice, and the chosen problem is frozen into
`config.json` before a single MCMC draw is taken.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.fast_segmentation_kernel import (                    # noqa: E402
    FastSegmentationKernel, assert_kernels_agree, key_of, segmentation_of, spans_of,
)
from hpop.mcmc_original.proposals import MoveType                            # noqa: E402
from hpop.mcmc_original.recurrent_rfs import (                               # noqa: E402
    recurrent_step_probabilities, recurrent_validity_update, RecurrentRFSParameters,
)
from hpop.mcmc_original.latent_poset import precedence_from_u                # noqa: E402
from hpop.mcmc_original.recurrent_segmentation import (                      # noqa: E402
    RecurrentBlockScorer, Stage6EMoveKernel, segmentation_log_weight,
)
from hpop.mcmc_original.stage6e_exact import (                               # noqa: E402
    boundary_marginals, enumerate_states, exact_posterior, expected_transition_counts,
    labelled_segment_marginals, log_evidence_forward, occurrence_label_marginals,
    segment_count_distribution, state_log_weights, total_variation,
)
from hpop.mcmc_original.stage6e_frozen import (                              # noqa: E402
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH, assert_stage6d_unchanged, config_hash,
)
from hpop.mcmc_original.stage6e_sampler import TraceSegmentationTarget       # noqa: E402
from hpop.mcmc_original.transitions import allowed_next, log_transition_matrix  # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "stage6e1a_exact_segmentation"

# ------------------------------------------------------------------ registered problem
J = 8                      # trace length; §8 requires J <= 8
K_SKILLS = 3               # §8 requires K <= 3
M_ROLES = 3                # reference-quality choice, as in the Stage 6D1 small model
D_LATENT = 2
EPSILON = 0.02
RHO_FIXED = 0.3
SCALARS_FIXED = {"beta": 1.5, "omega": 1.7346, "lambda_rep": 0.8, "lambda_back": 0.25}

# Three distinct, nondegenerate induced orders, fixed before any sampling.
U_BY_SKILL = np.array([
    [[2.0, 2.0], [1.0, 0.0], [0.0, 1.0]],     # role 0 dominates; 1 and 2 incomparable
    [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0]],     # a pure antichain
    [[3.0, 3.0], [2.0, 2.0], [0.0, 0.0]],     # a total order 0 > 1 > 2
], dtype=float)

CANDIDATE_SEEDS = tuple(range(50))            # scanned in ascending order
NONDEGENERACY = {"max_probability_below": 0.90, "min_states_above_0.01": 3}

# Chains
N_CHAINS = 4
N_SWEEPS = 400_000
BURN_IN = 20_000
THIN = 4
CHAIN_SEEDS = (6_051_001, 6_051_002, 6_051_003, 6_051_004)


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:                                                # pragma: no cover
        return "unknown"


def jsonable(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if math.isnan(v) else v
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


# ------------------------------------------------------------------ problem generation
def generate_trace(seed: int) -> tuple:
    """Sample one length-J trace from the registered generative model.

    A skill path is drawn from `(pi, P)`, widths from the legal set, and each block is
    generated by the same recurrent equations the likelihood evaluates, from `q_0 = 0`.
    The generating segmentation is returned for the record and is never used again.
    """
    rng = np.random.default_rng(seed)
    # a two-block truth: the only multi-segment shape J = 8 admits under width >= 3
    widths = (4, 4)
    labels = [int(rng.integers(K_SKILLS))]
    labels.append(int(rng.choice(allowed_next(labels[0], K_SKILLS))))
    roles = []
    for width, skill in zip(widths, labels):
        params = RecurrentRFSParameters(
            beta=SCALARS_FIXED["beta"], epsilon=EPSILON,
            shared_omega=SCALARS_FIXED["omega"],
            lambda_rep=SCALARS_FIXED["lambda_rep"],
            lambda_back=SCALARS_FIXED["lambda_back"])
        u = U_BY_SKILL[skill]
        precedence = precedence_from_u(u)
        q = np.zeros(M_ROLES)
        for _ in range(width):
            mixed = recurrent_step_probabilities(u, q, params)
            y = int(rng.choice(M_ROLES, p=mixed))
            roles.append(y)
            q = recurrent_validity_update(y, precedence, q, params.shared_omega)
    ends, running = [], 0
    for w in widths:
        running += w
        ends.append(running)
    return tuple(roles), tuple(zip(ends, labels))


def build_scorer(trace) -> RecurrentBlockScorer:
    return RecurrentBlockScorer(
        traces=(trace,), epsilon=EPSILON, u_by_skill=U_BY_SKILL,
        beta=SCALARS_FIXED["beta"], omega=SCALARS_FIXED["omega"],
        lambda_rep=SCALARS_FIXED["lambda_rep"], lambda_back=SCALARS_FIXED["lambda_back"],
        min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH)


def fixed_path_prior():
    log_pi = np.log(np.full(K_SKILLS, 1.0 / K_SKILLS))
    transition = np.zeros((K_SKILLS, K_SKILLS))
    for h in range(K_SKILLS):
        for k in allowed_next(h, K_SKILLS):
            transition[h, k] = 1.0 / (K_SKILLS - 1)
    return log_pi, transition, log_transition_matrix(transition)


def select_problem() -> dict:
    """Registered scan: first candidate seed meeting both nondegeneracy criteria."""
    log_pi, transition, log_transition = fixed_path_prior()
    states = enumerate_states(J, K_SKILLS, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
    scanned = []
    for seed in CANDIDATE_SEEDS:
        trace, truth = generate_trace(seed)
        scorer = build_scorer(trace)
        weights = state_log_weights(states, 0, J, scorer, log_pi, log_transition, DELTA_B)
        posterior = exact_posterior(states, weights)
        p = posterior["probability"]
        row = {"seed": seed, "max_probability": float(p.max()),
               "n_states_above_0.01": int((p > 0.01).sum())}
        scanned.append(row)
        if (row["max_probability"] < NONDEGENERACY["max_probability_below"]
                and row["n_states_above_0.01"] >= NONDEGENERACY["min_states_above_0.01"]):
            return {"selected_seed": seed, "trace": trace, "generating_segmentation": truth,
                    "scan": scanned, "states": states, "posterior": posterior,
                    "log_pi": log_pi, "transition": transition,
                    "log_transition": log_transition}
    raise SystemExit(f"no candidate seed met the registered nondegeneracy criteria; "
                     f"scanned {len(scanned)}")


# --------------------------------------------------------------------------- the chain
def run_chain(trace, scorer, log_pi, log_transition, seed: int) -> dict:
    """One segmentation-only chain: the registered local move kernel and nothing else."""
    kernel = FastSegmentationKernel(trace_length=J, n_skills=K_SKILLS,
                                    min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH)
    target = TraceSegmentationTarget(trace_index=0, trace_length=J, scorer=scorer,
                                     delta_b=DELTA_B, min_width=MIN_BLOCK_WIDTH,
                                     max_width=MAX_BLOCK_WIDTH)
    target.set_path_prior(log_pi, log_transition)

    rng = np.random.default_rng(seed)
    # dispersed start: a random legal state, not the generating one
    legal = enumerate_states(J, K_SKILLS, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
    current = legal[int(rng.integers(len(legal)))]
    current_value = target(current)

    proposed = {m: 0 for m in MoveType.ALL}
    accepted = {m: 0 for m in MoveType.ALL}
    impossible = {m: 0 for m in MoveType.ALL}
    retained = []
    began = time.perf_counter()
    for i in range(N_SWEEPS):
        candidate, move = kernel.sample_proposal(current, rng)
        proposed[move] += 1
        if candidate == current:
            impossible[move] += 1
        else:
            forward = kernel.proposal_prob(current, candidate)
            reverse = kernel.proposal_prob(candidate, current)
            candidate_value = target(candidate)
            log_alpha = ((candidate_value - current_value)
                         + math.log(reverse) - math.log(forward))
            if log_alpha >= 0.0 or math.log(rng.random()) < log_alpha:
                current, current_value = candidate, candidate_value
                accepted[move] += 1
        if i >= BURN_IN and (i - BURN_IN) % THIN == 0:
            retained.append(current)
    return {"seed": seed, "retained": retained, "proposed": proposed,
            "accepted": accepted, "impossible": impossible,
            "runtime_seconds": time.perf_counter() - began}


def empirical(states, retained) -> np.ndarray:
    index = {s: i for i, s in enumerate(states)}
    counts = np.zeros(len(states))
    for key in retained:
        counts[index[key]] += 1.0
    return counts / counts.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps", type=int, default=N_SWEEPS)
    args = parser.parse_args()
    globals()["N_SWEEPS"] = args.sweeps

    OUT.mkdir(parents=True, exist_ok=True)
    assert_stage6d_unchanged()

    # ---- registered problem selection, before any sampling ---------------------------
    problem = select_problem()
    trace = problem["trace"]
    states = problem["states"]
    posterior = problem["posterior"]
    exact_p = posterior["probability"]
    scorer = build_scorer(trace)
    log_pi, transition, log_transition = (problem["log_pi"], problem["transition"],
                                          problem["log_transition"])

    # ---- §8 gate 1: two independent routes to log Z ------------------------------------
    log_z_forward = log_evidence_forward(0, J, K_SKILLS, scorer, log_pi, log_transition,
                                         DELTA_B, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
    log_z_gap = abs(posterior["log_evidence"] - log_z_forward)

    # ---- the enumerated weight must equal the registered direct target -----------------
    direct_gap = 0.0
    for key, log_w in zip(states, posterior["log_weights"]):
        direct = segmentation_log_weight(segmentation_of(key), 0, J, scorer, log_pi,
                                         log_transition, DELTA_B)["log_weight"]
        direct_gap = max(direct_gap, abs(direct - float(log_w)))

    # ---- the fast kernel must reproduce the reference kernel's law ---------------------
    reference_kernel = Stage6EMoveKernel(x=trace, skills=(), n_skills=K_SKILLS,
                                         min_width=MIN_BLOCK_WIDTH,
                                         max_width=MAX_BLOCK_WIDTH)
    fast_kernel = FastSegmentationKernel(trace_length=J, n_skills=K_SKILLS,
                                         min_width=MIN_BLOCK_WIDTH,
                                         max_width=MAX_BLOCK_WIDTH)
    agreement = [assert_kernels_agree(fast_kernel, reference_kernel, key)
                 for key in states]
    kernel_worst = max(a["max_probability_difference"] for a in agreement)
    kernel_all_pass = all(a["pass"] for a in agreement)

    # the enumerated support must be exactly the kernel's reachable support
    reachable = set()
    frontier = [states[0]]
    seen = {states[0]}
    while frontier:
        key = frontier.pop()
        reachable.add(key)
        for move in MoveType.ALL:
            for nxt in fast_kernel.neighbours(key, move):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
    support_matches = reachable == set(states)

    print(f"[6E1A] seed {problem['selected_seed']}  |states| = {len(states)}  "
          f"max p = {exact_p.max():.4f}  #(p>0.01) = {int((exact_p > 0.01).sum())}")
    print(f"[6E1A] log Z enum {posterior['log_evidence']:.12f}  forward {log_z_forward:.12f}"
          f"  gap {log_z_gap:.3e}")

    # ---- chains ------------------------------------------------------------------------
    chains = [run_chain(trace, scorer, log_pi, log_transition, s)
              for s in CHAIN_SEEDS[:N_CHAINS]]
    for c in chains:
        print(f"[6E1A] chain seed {c['seed']}: {len(c['retained']):,} retained in "
              f"{c['runtime_seconds']:.1f}s")

    pooled = [k for c in chains for k in c["retained"]]
    empirical_p = empirical(states, pooled)
    per_chain = [empirical(states, c["retained"]) for c in chains]

    # ---- §8 comparisons -----------------------------------------------------------------
    tv_path = total_variation(empirical_p, exact_p)
    tv_per_chain = [total_variation(p, exact_p) for p in per_chain]

    exact_boundary = boundary_marginals(states, exact_p, J)
    mcmc_boundary = boundary_marginals(states, empirical_p, J)
    boundary_error = float(np.abs(exact_boundary - mcmc_boundary).max())

    exact_labels = occurrence_label_marginals(states, exact_p, J, K_SKILLS)
    mcmc_labels = occurrence_label_marginals(states, empirical_p, J, K_SKILLS)
    label_error = float(np.abs(exact_labels - mcmc_labels).max())

    exact_segments = labelled_segment_marginals(states, exact_p)
    mcmc_segments = labelled_segment_marginals(states, empirical_p)
    segment_keys = sorted(set(exact_segments) | set(mcmc_segments))
    segment_error = max(abs(exact_segments.get(k, 0.0) - mcmc_segments.get(k, 0.0))
                        for k in segment_keys)

    exact_counts = segment_count_distribution(states, exact_p, max_segments=J)
    mcmc_counts = segment_count_distribution(states, empirical_p, max_segments=J)
    count_tv = total_variation(mcmc_counts, exact_counts)

    exact_transitions = expected_transition_counts(states, exact_p, K_SKILLS)
    mcmc_transitions = expected_transition_counts(states, empirical_p, K_SKILLS)
    transition_error = float(np.abs(exact_transitions - mcmc_transitions).max())

    gates = {
        "log_evidence_independent_agreement": {
            "value": log_z_gap, "threshold": 1e-10, "pass": bool(log_z_gap < 1e-10)},
        "enumerated_weight_equals_direct_target": {
            "value": direct_gap, "threshold": 1e-12, "pass": bool(direct_gap < 1e-12)},
        "fast_kernel_matches_reference_kernel": {
            "value": kernel_worst, "threshold": 1e-15, "pass": bool(kernel_all_pass)},
        "kernel_support_equals_enumerated_support": {
            "value": len(reachable), "threshold": len(states),
            "pass": bool(support_matches)},
        "nondegenerate_max_probability": {
            "value": float(exact_p.max()), "threshold": 0.90,
            "pass": bool(exact_p.max() < 0.90)},
        "nondegenerate_state_count": {
            "value": int((exact_p > 0.01).sum()), "threshold": 3,
            "pass": bool((exact_p > 0.01).sum() >= 3)},
        "retained_draws": {
            "value": len(pooled), "threshold": 100_000,
            "pass": bool(len(pooled) >= 100_000)},
        "path_total_variation": {
            "value": tv_path, "threshold": 0.01, "pass": bool(tv_path < 0.01)},
        "max_boundary_marginal_error": {
            "value": boundary_error, "threshold": 0.01,
            "pass": bool(boundary_error < 0.01)},
        "max_occurrence_label_marginal_error": {
            "value": label_error, "threshold": 0.01, "pass": bool(label_error < 0.01)},
        "max_labelled_segment_marginal_error": {
            "value": segment_error, "threshold": 0.01,
            "pass": bool(segment_error < 0.01)},
        "segment_count_total_variation": {
            "value": count_tv, "threshold": 0.01, "pass": bool(count_tv < 0.01)},
        "max_expected_transition_count_error": {
            "value": transition_error, "threshold": 0.01,
            "pass": bool(transition_error < 0.01)},
    }
    all_pass = all(g["pass"] for g in gates.values())

    # ---- artifacts ----------------------------------------------------------------------
    config = {
        "stage": "6E1A", "source_commit": source_commit(),
        "stage6e_config_hash": config_hash(),
        "python": platform.python_version(), "numpy": np.__version__,
        "problem": {
            "trace_length_J": J, "n_skills": K_SKILLS, "m_roles": M_ROLES,
            "d_latent": D_LATENT, "epsilon": EPSILON, "delta_B": DELTA_B,
            "min_width": MIN_BLOCK_WIDTH, "max_width": MAX_BLOCK_WIDTH,
            "rho_fixed": RHO_FIXED, "scalars_fixed": SCALARS_FIXED,
            "U_by_skill": U_BY_SKILL.tolist(),
            "induced_orders": [precedence_from_u(U_BY_SKILL[k]).tolist()
                               for k in range(K_SKILLS)],
            "pi": np.exp(log_pi).tolist(), "P": transition.tolist(),
            "observed_trace": list(trace),
            "generating_segmentation": [list(p) for p in problem["generating_segmentation"]],
            "generating_segmentation_note": "recorded only; never used in any comparison",
        },
        "selection_rule": {
            "candidate_seeds": list(CANDIDATE_SEEDS),
            "criteria": NONDEGENERACY,
            "rule": "first seed in ascending order meeting both criteria; registered "
                    "before any MCMC draw",
            "selected_seed": problem["selected_seed"],
            "scan": problem["scan"][:problem["scan"].index(
                next(r for r in problem["scan"]
                     if r["seed"] == problem["selected_seed"])) + 1],
        },
        "chains": {"n_chains": N_CHAINS, "sweeps": N_SWEEPS, "burn_in": BURN_IN,
                   "thin": THIN, "seeds": list(CHAIN_SEEDS[:N_CHAINS]),
                   "proposals_per_sweep": 1,
                   "kernel": "FastSegmentationKernel (proven equal to Stage6EMoveKernel "
                             "at every enumerated state, see gates)"},
        "latent": ["S", "z"], "fixed": ["U", "rho", "beta", "omega", "lambda_rep",
                                        "lambda_back", "pi", "P", "delta_B", "epsilon"],
    }

    np.savez_compressed(
        OUT / "exact_reference.npz",
        state_ends=np.array([[e for e, _ in k] + [-1] * (J - len(k)) for k in states],
                            dtype=np.int16),
        state_labels=np.array([[s for _, s in k] + [-1] * (J - len(k)) for k in states],
                              dtype=np.int8),
        state_n_segments=np.array([len(k) for k in states], dtype=np.int8),
        log_weights=posterior["log_weights"], probability=exact_p,
        log_evidence=np.array([posterior["log_evidence"]]),
        log_evidence_forward=np.array([log_z_forward]),
        boundary_marginals=exact_boundary, label_marginals=exact_labels,
        segment_count_distribution=exact_counts,
        expected_transition_counts=exact_transitions,
        observed_trace=np.array(trace, dtype=np.int8))

    np.savez_compressed(
        OUT / "chains.npz",
        empirical_probability=empirical_p,
        per_chain_probability=np.array(per_chain),
        chain_seeds=np.array([c["seed"] for c in chains]),
        mcmc_boundary_marginals=mcmc_boundary, mcmc_label_marginals=mcmc_labels,
        mcmc_segment_count_distribution=mcmc_counts,
        mcmc_expected_transition_counts=mcmc_transitions,
        n_retained=np.array([len(c["retained"]) for c in chains]),
        runtime_seconds=np.array([c["runtime_seconds"] for c in chains]))

    comparison = {
        "gates": gates, "all_pass": all_pass,
        "log_evidence_enumerated": posterior["log_evidence"],
        "log_evidence_forward_recursion": log_z_forward,
        "n_enumerated_states": len(states),
        "n_retained_total": len(pooled),
        "n_retained_per_chain": [len(c["retained"]) for c in chains],
        "path_total_variation_pooled": tv_path,
        "path_total_variation_per_chain": tv_per_chain,
        "max_boundary_marginal_error": boundary_error,
        "max_occurrence_label_marginal_error": label_error,
        "max_labelled_segment_marginal_error": segment_error,
        "segment_count_total_variation": count_tv,
        "max_expected_transition_count_error": transition_error,
        "exact_boundary_marginals": exact_boundary.tolist(),
        "mcmc_boundary_marginals": mcmc_boundary.tolist(),
        "exact_segment_count_distribution": exact_counts.tolist(),
        "mcmc_segment_count_distribution": mcmc_counts.tolist(),
        "exact_expected_transition_counts": exact_transitions.tolist(),
        "mcmc_expected_transition_counts": mcmc_transitions.tolist(),
        "top_states": [
            {"key": [list(p) for p in states[i]],
             "exact": float(exact_p[i]), "mcmc": float(empirical_p[i]),
             "absolute_error": float(abs(exact_p[i] - empirical_p[i]))}
            for i in np.argsort(-exact_p)[:10]],
        "acceptance_by_move": {
            m: {"proposed": sum(c["proposed"][m] for c in chains),
                "accepted": sum(c["accepted"][m] for c in chains),
                "impossible": sum(c["impossible"][m] for c in chains),
                "acceptance_rate": (sum(c["accepted"][m] for c in chains)
                                    / max(1, sum(c["proposed"][m] for c in chains)))}
            for m in MoveType.ALL},
        "runtime_seconds": sum(c["runtime_seconds"] for c in chains),
    }

    (OUT / "config.json").write_text(json.dumps(jsonable(config), indent=2))
    (OUT / "comparison.json").write_text(json.dumps(jsonable(comparison), indent=2))

    for name, gate in gates.items():
        print(f"[6E1A] {name:48s} {gate['value']!s:>12.12s} "
              f"(<= {gate['threshold']}) -> {'PASS' if gate['pass'] else 'FAIL'}")
    print(f"[6E1A] wrote {OUT}")
    if not all_pass:
        raise SystemExit("Stage 6E1A FAILED: "
                         f"{[k for k, g in gates.items() if not g['pass']]}")


if __name__ == "__main__":
    main()
