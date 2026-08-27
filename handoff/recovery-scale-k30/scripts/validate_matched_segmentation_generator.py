"""Registered validation of the exact matched HPOP synthetic generator.

Run:  PYTHONPATH=src .venv/bin/python scripts/validate_matched_segmentation_generator.py

Writes results/mcmc_original/matched_generator_validation/ and exits nonzero if
any registered gate fails. Runs NO MCMC and NO inference experiment: everything
here is exact dynamic programming, exhaustive enumeration on tiny traces, and
forward sampling from the generator being validated.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_generator_diagnostics as mgd            # noqa: E402
from hpop.mcmc_original import matched_segmentation_prior as msp               # noqa: E402
from hpop.mcmc_original import matched_synthetic_generator as msg              # noqa: E402
from hpop.mcmc_original.latent_poset import precedence_from_u                  # noqa: E402
from hpop.mcmc_original.recurrent_rfs import (                                 # noqa: E402
    recurrent_rfs_log_likelihood, recurrent_step_probabilities,
    recurrent_validity_update, sample_recurrent_rfs_sequence,
)
from hpop.mcmc_original.recurrent_segmentation import RecurrentBlockScorer     # noqa: E402
from hpop.mcmc_original.stage6e_exact import (                                 # noqa: E402
    enumerate_states, log_evidence_forward, state_log_weights,
)
from hpop.mcmc_original.targets import logsumexp                               # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "matched_generator_validation"

# ----------------------------------------------------------------- registered design
PREREGISTRATION = {
    "task": "matched synthetic generator validation",
    "generator_version": msg.GENERATOR_VERSION,
    "trace_length_design_J": [24, 32, 40, 48],
    "trace_length_design_note": "no registered trace-length support exists in the "
                                "current model (the likelihood conditions on J); "
                                "the suggested design {24, 32, 40, 48} is adopted "
                                "and registered here before sampling",
    "tiny_J": [6, 7, 10],
    "n_segmentation_samples_per_J": 100_000,
    "n_tiny_state_samples_per_J": 200_000,
    "n_label_samples": 200_000,
    "n_parity_traces": 50,
    "seeds": {
        "segmentation_parity": {"24": 700_024, "32": 700_032, "40": 700_040,
                                "48": 700_048},
        "tiny_state_parity": {"6": 710_006, "7": 710_007, "10": 710_010},
        "skill_label_initial": 720_001,
        "skill_label_transitions": 720_002,
        "corpus_label_frequencies": 720_003,
        "log_prob_parity_corpus": 730_001,
        "tiny_logz_corpus": 730_002,
        "reset_parity_corpus": 730_003,
        "negative_control_old_mechanism": 740_001,
        "negative_control_state_leak": 740_002,
        "reproducibility_master_a": 750_001,
        "reproducibility_master_b": 750_002,
    },
    "gates": {
        "dp_normalizer_vs_enumeration_abs_log_error": 1e-12,
        "exact_segment_count_two_route_max_error": 1e-12,
        "exact_boundary_marginal_two_route_max_error": 1e-12,
        "empirical_segment_count_tv": 0.01,
        "empirical_boundary_marginal_max_error": 0.01,
        "tiny_full_state_tv": 0.01,
        "tiny_full_state_max_probability_error": 0.01,
        "pi_empirical_max_error": 0.01,
        "transition_row_empirical_max_error": 0.01,
        "self_transition_count": 0,
        "illegal_segmentation_count": 0,
        "complete_data_log_prob_parity": 1e-10,
        "tiny_logz_parity": 1e-10,
    },
    "truth_configuration": "supplied mode: the Stage 6E2 registered truth "
                           "(U, pi, P, scalars, epsilon, delta_B, widths), "
                           "recorded before any observation was generated",
}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _gate(gates: list, name: str, value, threshold, passed: bool) -> bool:
    gates.append({"gate": name, "value": value, "threshold": threshold,
                  "pass": bool(passed)})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {value} "
          f"(threshold {threshold})")
    return bool(passed)


# ------------------------------------------------------------------ source manifest
def write_source_manifest() -> None:
    sources = [
        "src/hpop/mcmc_original/matched_segmentation_prior.py",
        "src/hpop/mcmc_original/matched_synthetic_generator.py",
        "src/hpop/mcmc_original/matched_generator_diagnostics.py",
        "src/hpop/mcmc_original/recurrent_rfs.py",
        "src/hpop/mcmc_original/latent_poset.py",
        "src/hpop/mcmc_original/targets.py",
        "src/hpop/mcmc_original/transitions.py",
        "src/hpop/mcmc_original/recurrent_segmentation.py",
        "src/hpop/mcmc_original/stage6e_frozen.py",
        "src/hpop/mcmc_original/stage6e_corpus.py",
        "src/hpop/mcmc_original/stage6c_frozen.py",
        "src/hpop/mcmc_original/sampler_u.py",
        "scripts/validate_matched_segmentation_generator.py",
        "scripts/generate_matched_smoke_corpus.py",
    ]
    truth = msg.supplied_truth()
    manifest = {
        "source_commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        "generator_version": msg.GENERATOR_VERSION,
        "file_hashes": {p: _sha256_file(ROOT / p) for p in sources
                        if (ROOT / p).exists()},
        "model_facts_verified_against_source": {
            "K_skills": {"value": truth.n_skills,
                         "source": "stage6e_frozen.N_SKILLS"},
            "m_roles": {"value": truth.n_roles, "source": "stage6e_frozen.N_ROLES"},
            "d_product_order": {"value": truth.latent_dim,
                                "source": "stage6e_frozen.LATENT_DIM "
                                          "(= stage6d_frozen.N_LATENT_COLUMNS)"},
            "delta_B": {"value": truth.delta_b, "source": "stage6e_frozen.DELTA_B"},
            "epsilon": {"value": truth.epsilon,
                        "source": "stage6b_frozen.EPSILON, inherited through the "
                                  "6C/6D/6E freezes"},
            "D_min": {"value": truth.min_width,
                      "source": "stage6e_frozen.MIN_BLOCK_WIDTH"},
            "D_max": {"value": truth.max_width,
                      "source": "stage6e_frozen.MAX_BLOCK_WIDTH"},
            "P_diagonal_zero": {"value": True,
                                "source": "transitions.allowed_next excludes h; "
                                          "stage6e_frozen.SELF_TRANSITIONS_ALLOWED "
                                          "= False"},
            "terminal_transition": {"value": False,
                                    "source": "targets.log_path_prior emits none; "
                                              "stage6e_frozen.TERMINAL_TRANSITION "
                                              "= False"},
            "duration_model": {"value": None,
                               "source": "recurrent_rfs: block length observed and "
                                         "conditioned upon; no p(T | skill)"},
            "background_skill": {"value": False, "source": "no such component in "
                                                           "stage6e_frozen/state"},
            "generalized_EM": {"value": False, "source": "not present"},
            "unknown_K_inference": {"value": False,
                                    "source": "stage6e_frozen fixes N_SKILLS"},
            "role_to_cpa_maps": {"value": "identity, injective",
                                 "source": "Stage6EModel feeds observed symbols "
                                           "directly to RecurrentBlockScorer as "
                                           "role indices"},
            "segmentation_prior_weight": {
                "value": "delta_B^(L-1) (1-delta_B)^(J-L) on widths in [D_min, D_max]",
                "source": "targets.log_boundary_prior via "
                          "stage6e_frozen.log_boundary_prior_6e; width bounds "
                          "enforced by recurrent_segmentation.is_legal_segmentation"},
            "recurrent_execution": {
                "value": "p_RFS from q_0 = 0 per block",
                "source": "recurrent_rfs.recurrent_step_probabilities / "
                          "recurrent_validity_update / "
                          "sample_recurrent_rfs_sequence"},
            "U_rho_prior": {
                "value": "U rows iid N(0, Sigma_rho); rho ~ Beta(1,1) truncated "
                         "to (0, 0.995)",
                "source": "sampler_u.log_u_prior / sigma_rho_matrix; "
                          "stage6c_frozen.log_rho_prior"},
        },
        "old_generator_preserved_not_reused": {
            "file": "src/hpop/mcmc_original/stage6e_corpus.py",
            "mechanism": "BLOCKS_PER_TRACE = (4, 5, 6) drawn uniformly, then iid "
                         "widths not conditioned on total length",
            "status": "unmodified; source-level regression test "
                      "tests/mcmc_original/test_matched_synthetic_generator.py::"
                      "TestOldMechanismBanned proves the new generator does not "
                      "contain, import, or call it",
        },
        "production_functions_reused_by_generator": [
            "matched_segmentation_prior.sample_segmentation_widths (exact DP)",
            "recurrent_rfs.sample_recurrent_rfs_sequence",
            "recurrent_rfs.recurrent_step_probabilities",
            "recurrent_rfs.recurrent_validity_update",
            "transitions.allowed_next",
            "targets.logsumexp",
        ],
        "inference_side_scorers_used_for_parity": [
            "targets.log_boundary_prior", "targets.log_path_prior",
            "recurrent_segmentation.RecurrentBlockScorer.score",
            "stage6e_exact.log_evidence_forward (tiny-trace log Z only)",
        ],
    }
    _dump("source_manifest.json", manifest)


# ------------------------------------------------------------------- exact references
def run_exact_reference_checks(gates: list) -> None:
    print("== exact normalizers and marginals ==")
    tol = PREREGISTRATION["gates"]["dp_normalizer_vs_enumeration_abs_log_error"]
    normalizers = {}
    ok = True
    for J in PREREGISTRATION["tiny_J"] + PREREGISTRATION["trace_length_design_J"]:
        dp = msp.log_normalizer(J)
        ref = mgd.exact_normalizer(J, 0.15, 3, 12)
        entry = {"log_C_dp_suffix": dp, "log_C_reference_combinatorial": ref,
                 "abs_error_dp_vs_reference": abs(dp - ref)}
        if J <= 12:
            enum = mgd.log_normalizer_from_enumeration(J, 0.15, 3, 12)
            entry["log_C_enumeration"] = enum
            entry["abs_error_dp_vs_enumeration"] = abs(dp - enum)
            ok &= abs(dp - enum) < tol
        ok &= abs(dp - ref) < tol
        normalizers[str(J)] = entry
    _dump("exact_normalizers.json", {"delta_B": 0.15, "D_min": 3, "D_max": 12,
                                     "normalizers": normalizers})
    _gate(gates, "dp_normalizer_vs_enumeration_and_reference",
          max(max(v.get("abs_error_dp_vs_enumeration", 0.0),
                  v["abs_error_dp_vs_reference"]) for v in normalizers.values()),
          tol, ok)

    tol2 = PREREGISTRATION["gates"]["exact_segment_count_two_route_max_error"]
    tol3 = PREREGISTRATION["gates"]["exact_boundary_marginal_two_route_max_error"]
    worst_l, worst_b = 0.0, 0.0
    for J in PREREGISTRATION["tiny_J"] + PREREGISTRATION["trace_length_design_J"]:
        a = msp.segment_count_distribution_dp(J)
        b = mgd.exact_segment_count_distribution(J, 0.15, 3, 12)
        n = max(len(a), len(b))
        worst_l = max(worst_l, float(np.abs(np.pad(a, (0, n - len(a)))
                                            - np.pad(b, (0, n - len(b)))).max()))
        worst_b = max(worst_b, float(np.abs(
            msp.boundary_marginals_dp(J)
            - mgd.exact_boundary_marginals(J, 0.15, 3, 12)).max()))
    _gate(gates, "exact_segment_count_two_route_max_error", worst_l, tol2,
          worst_l < tol2)
    _gate(gates, "exact_boundary_marginal_two_route_max_error", worst_b, tol3,
          worst_b < tol3)


# ------------------------------------------------------- segmentation sampler parity
def run_segmentation_parity(gates: list) -> None:
    print("== segmentation sampler parity (100k per J) ==")
    n = PREREGISTRATION["n_segmentation_samples_per_J"]
    tv_gate = PREREGISTRATION["gates"]["empirical_segment_count_tv"]
    b_gate = PREREGISTRATION["gates"]["empirical_boundary_marginal_max_error"]
    count_report, boundary_report = {}, {}
    ok_tv = ok_b = ok_support = True
    for J in PREREGISTRATION["trace_length_design_J"]:
        seed = PREREGISTRATION["seeds"]["segmentation_parity"][str(J)]
        rng = np.random.default_rng(seed)
        tables = msp.width_sampling_tables(J)
        samples = [msp.sample_segmentation_widths(rng, J, tables=tables)
                   for _ in range(n)]
        exact_l = mgd.exact_segment_count_distribution(J, 0.15, 3, 12)
        emp_l = mgd.empirical_segment_count_distribution(samples, len(exact_l) - 1)
        tv = mgd.total_variation(emp_l, exact_l)
        exact_b = mgd.exact_boundary_marginals(J, 0.15, 3, 12)
        emp_b = mgd.empirical_boundary_marginals(samples, J)
        b_err = float(np.abs(emp_b - exact_b).max())
        violations = mgd.segmentation_support_violations(samples, J, 3, 12)
        widths_emp = np.zeros(12 - 3 + 1)
        for widths in samples:
            for w in widths:
                widths_emp[w - 3] += 1
        widths_emp /= n
        widths_exact = mgd.exact_expected_width_counts(J, 0.15, 3, 12)
        count_report[str(J)] = {
            "n_samples": n, "seed": seed,
            "empirical_p_L": emp_l.tolist(), "exact_p_L": exact_l.tolist(),
            "tv": tv, "tv_gate": tv_gate,
            "empirical_mean_blocks": float(np.dot(np.arange(len(emp_l)), emp_l)),
            "exact_mean_blocks": float(np.dot(np.arange(len(exact_l)), exact_l)),
            "empirical_expected_width_counts": widths_emp.tolist(),
            "exact_expected_width_counts": widths_exact.tolist(),
            "max_width_count_error": float(np.abs(widths_emp - widths_exact).max()),
            "support_violations": violations,
            "monte_carlo_note": "multinomial; per-category binomial s.e. <= "
                                f"{0.5 / math.sqrt(n):.2e} at p = 0.5",
        }
        boundary_report[str(J)] = {
            "n_samples": n, "seed": seed,
            "empirical": emp_b.tolist(), "exact": exact_b.tolist(),
            "max_abs_error": b_err, "gate": b_gate,
            "max_binomial_se": float(np.sqrt(exact_b * (1 - exact_b) / n).max()),
        }
        ok_tv &= tv < tv_gate
        ok_b &= b_err < b_gate
        ok_support &= (violations["illegal_width"] == 0
                       and violations["incomplete_cover"] == 0
                       and violations["overlap_or_gap"] == 0
                       and violations["empty"] == 0)
        print(f"  J={J}: TV(L)={tv:.5f}, max boundary err={b_err:.5f}, "
              f"violations={violations}")
    _dump("segment_count_parity.json", count_report)
    _dump("boundary_marginal_parity.json", boundary_report)
    _gate(gates, "empirical_segment_count_tv_all_J",
          max(v["tv"] for v in count_report.values()), tv_gate, ok_tv)
    _gate(gates, "empirical_boundary_marginal_max_error_all_J",
          max(v["max_abs_error"] for v in boundary_report.values()), b_gate, ok_b)
    _gate(gates, "illegal_segmentation_count",
          sum(sum(v["support_violations"][k] for k in
                  ("illegal_width", "incomplete_cover", "overlap_or_gap", "empty"))
              for v in count_report.values()), 0, ok_support)


def run_tiny_state_parity(gates: list) -> None:
    print("== tiny full-state parity (200k per J) ==")
    n = PREREGISTRATION["n_tiny_state_samples_per_J"]
    tv_gate = PREREGISTRATION["gates"]["tiny_full_state_tv"]
    p_gate = PREREGISTRATION["gates"]["tiny_full_state_max_probability_error"]
    report = {}
    ok = True
    for J in PREREGISTRATION["tiny_J"]:
        seed = PREREGISTRATION["seeds"]["tiny_state_parity"][str(J)]
        rng = np.random.default_rng(seed)
        tables = msp.width_sampling_tables(J)
        counts = Counter(msp.sample_segmentation_widths(rng, J, tables=tables)
                         for _ in range(n))
        states = mgd.enumerate_legal_segmentations(J, 3, 12)
        log_c = mgd.exact_normalizer(J, 0.15, 3, 12)
        rows, tv, max_err, all_visited = [], 0.0, 0.0, True
        for state in states:
            L = len(state)
            exact = math.exp((L - 1) * math.log(0.15)
                             + (J - L) * math.log1p(-0.15) - log_c)
            observed = counts.get(state, 0) / n
            tv += 0.5 * abs(observed - exact)
            max_err = max(max_err, abs(observed - exact))
            if exact * n > 50 and counts.get(state, 0) == 0:
                all_visited = False
            rows.append({"state": list(state), "exact": exact,
                         "empirical": observed, "count": counts.get(state, 0)})
        outside = sum(c for s, c in counts.items() if s not in set(states))
        report[str(J)] = {"n_samples": n, "seed": seed, "n_states": len(states),
                          "tv": tv, "max_state_probability_error": max_err,
                          "samples_outside_legal_support": outside,
                          "all_expected_states_visited": all_visited,
                          "states": rows}
        ok &= tv < tv_gate and max_err < p_gate and outside == 0 and all_visited
        print(f"  J={J}: {len(states)} states, TV={tv:.5f}, max err={max_err:.5f}")
    _dump("tiny_state_parity.json", report)
    _gate(gates, "tiny_full_state_tv", max(v["tv"] for v in report.values()),
          tv_gate, ok)


# ------------------------------------------------------------- skill-label parity
def run_skill_label_parity(gates: list) -> None:
    print("== skill-label parity (200k draws per distribution) ==")
    truth = msg.supplied_truth()
    n = PREREGISTRATION["n_label_samples"]
    pi_gate = PREREGISTRATION["gates"]["pi_empirical_max_error"]
    row_gate = PREREGISTRATION["gates"]["transition_row_empirical_max_error"]

    rng = np.random.default_rng(PREREGISTRATION["seeds"]["skill_label_initial"])
    counts = np.zeros(truth.n_skills)
    for _ in range(n):
        counts[msg.sample_initial_skill(rng, truth.pi)] += 1
    pi_err = float(np.abs(counts / n - truth.pi).max())

    rng = np.random.default_rng(PREREGISTRATION["seeds"]["skill_label_transitions"])
    row_errors, self_transitions, row_report = [], 0, {}
    for h in range(truth.n_skills):
        row_counts = np.zeros(truth.n_skills)
        for _ in range(n):
            row_counts[msg.sample_next_skill(rng, truth.transition, h)] += 1
        self_transitions += int(row_counts[h])
        err = float(np.abs(row_counts / n - truth.transition[h]).max())
        row_errors.append(err)
        row_report[str(h)] = {"empirical": (row_counts / n).tolist(),
                              "exact": truth.transition[h].tolist(),
                              "max_error": err,
                              "self_transition_count": int(row_counts[h])}

    # secondary end-to-end check: corpus-level path frequencies
    corpus = msg.generate_corpus(
        PREREGISTRATION["seeds"]["corpus_label_frequencies"], [32] * 400, [],
        truth)
    first = np.zeros(truth.n_skills)
    pair_counts = np.zeros((truth.n_skills, truth.n_skills))
    for trace in corpus.train:
        first[trace.labels[0]] += 1
        for a, b in zip(trace.labels[:-1], trace.labels[1:]):
            pair_counts[a, b] += 1
    _dump("skill_label_parity.json", {
        "n_samples": n,
        "pi": {"empirical": (counts / n).tolist(), "exact": truth.pi.tolist(),
               "max_error": pi_err, "gate": pi_gate},
        "transition_rows": row_report,
        "self_transition_count_total": self_transitions,
        "corpus_level_secondary_check": {
            "n_traces": len(corpus.train),
            "initial_frequencies": (first / first.sum()).tolist(),
            "transition_frequencies_row_normalized":
                (pair_counts / np.maximum(pair_counts.sum(axis=1, keepdims=True),
                                          1.0)).tolist(),
            "corpus_self_transition_count": int(np.trace(pair_counts)),
        },
    })
    _gate(gates, "pi_empirical_max_error", pi_err, pi_gate, pi_err < pi_gate)
    _gate(gates, "transition_row_empirical_max_error", max(row_errors), row_gate,
          max(row_errors) < row_gate)
    _gate(gates, "self_transition_count",
          self_transitions + int(np.trace(pair_counts)), 0,
          self_transitions == 0 and int(np.trace(pair_counts)) == 0)


# ------------------------------------------------------- recurrent reset + parity
def run_recurrent_and_log_prob_parity(gates: list) -> None:
    print("== recurrent q0 reset and complete-data log-probability parity ==")
    truth = msg.supplied_truth()
    seed = PREREGISTRATION["seeds"]["reset_parity_corpus"]
    corpus = msg.generate_corpus(seed, [24, 32, 40, 48] * 3, [24, 32], truth)
    params = truth.rfs_parameters()

    order_independent = True
    replay_worst = 0.0
    for trace in corpus.train + corpus.heldout:
        blocks = list(zip(trace.widths, trace.labels, trace.role_blocks,
                          trace.block_log_likelihoods))
        for order in (range(len(blocks)), reversed(range(len(blocks)))):
            for l in order:
                width, skill, recorded, recorded_ll = blocks[l]
                rng = msg.block_rng(seed, trace.split, trace.trace_index, l)
                alone = sample_recurrent_rfs_sequence(
                    rng, width, truth.u_by_skill[skill], params)
                order_independent &= tuple(alone) == recorded
                replay = recurrent_rfs_log_likelihood(
                    recorded, truth.u_by_skill[skill], truth.beta, truth.epsilon,
                    truth.omega, truth.lambda_rep, truth.lambda_back)
                replay_worst = max(replay_worst, abs(replay - recorded_ll))
    _dump("recurrent_reset_parity.json", {
        "corpus_seed": seed,
        "n_traces": len(corpus.train) + len(corpus.heldout),
        "block_regenerated_alone_equals_recorded_forward_and_reverse":
            order_independent,
        "q0_reset_replay_max_abs_error": replay_worst,
        "state_leak_detector": "every block regenerated in isolation (its own "
                               "SeedSequence stream, q_0 = 0 inside the production "
                               "sampler) is byte-identical to the block generated "
                               "inside the full trace, in forward AND reverse "
                               "order; a leaked q or shared stream would break "
                               "this equality",
        "batch_vs_scalar": "generation is scalar-only; the vectorized batch path "
                           "(fast_segmentation_kernel via RecurrentBlockScorer) is "
                           "compared against the scalar generator accumulation in "
                           "log_probability_parity.json",
    })
    _gate(gates, "recurrent_q0_reset_and_block_independence", replay_worst, 1e-10,
          order_independent and replay_worst < 1e-10)

    # complete-data log-probability parity, generator vs production inference side
    seed2 = PREREGISTRATION["seeds"]["log_prob_parity_corpus"]
    n_traces = PREREGISTRATION["n_parity_traces"]
    lengths = (PREREGISTRATION["trace_length_design_J"]
               * (n_traces // 4 + 1))[:n_traces]
    parity_corpus = msg.generate_corpus(seed2, lengths, [24, 32], truth)
    traces = parity_corpus.train + parity_corpus.heldout
    scorer = RecurrentBlockScorer(
        traces=tuple(t.cpa for t in traces), epsilon=truth.epsilon,
        u_by_skill=truth.u_by_skill, beta=truth.beta, omega=truth.omega,
        lambda_rep=truth.lambda_rep, lambda_back=truth.lambda_back,
        max_width=truth.max_width, min_width=truth.min_width)
    log_c = {J: mgd.exact_normalizer(J, truth.delta_b, truth.min_width,
                                     truth.max_width)
             for J in {t.length for t in traces}}
    diffs, all_finite = [], True
    for i, trace in enumerate(traces):
        g = msg.generator_complete_data_log_prob(trace)
        s = msg.inference_complete_data_log_prob(trace, truth, scorer, i,
                                                 log_c[trace.length])
        diffs.append(abs(g - s))
        all_finite &= math.isfinite(g) and math.isfinite(s)
        all_finite &= all(math.isfinite(v) for v in trace.block_log_likelihoods)
    parity_gate = PREREGISTRATION["gates"]["complete_data_log_prob_parity"]

    # tiny-trace marginal likelihood: forward recursion vs enumeration
    seed3 = PREREGISTRATION["seeds"]["tiny_logz_corpus"]
    tiny_corpus = msg.generate_corpus(seed3, [6, 7, 10, 6, 7, 10], [], truth)
    tiny_scorer = RecurrentBlockScorer(
        traces=tuple(t.cpa for t in tiny_corpus.train), epsilon=truth.epsilon,
        u_by_skill=truth.u_by_skill, beta=truth.beta, omega=truth.omega,
        lambda_rep=truth.lambda_rep, lambda_back=truth.lambda_back,
        max_width=truth.max_width, min_width=truth.min_width)
    with np.errstate(divide="ignore"):
        log_pi = np.log(truth.pi)
        log_transition = np.log(truth.transition)
    logz_rows, logz_worst = [], 0.0
    for i, trace in enumerate(tiny_corpus.train):
        states = enumerate_states(trace.length, truth.n_skills, truth.min_width,
                                  truth.max_width)
        weights = state_log_weights(states, i, trace.length, tiny_scorer, log_pi,
                                    log_transition, truth.delta_b)
        z_enum = float(logsumexp(weights))
        z_forward = log_evidence_forward(i, trace.length, truth.n_skills,
                                         tiny_scorer, log_pi, log_transition,
                                         truth.delta_b, truth.min_width,
                                         truth.max_width)
        logz_worst = max(logz_worst, abs(z_forward - z_enum))
        logz_rows.append({"J": trace.length, "log_Z_enumeration": z_enum,
                          "log_Z_forward": z_forward,
                          "abs_error": abs(z_forward - z_enum)})
    logz_gate = PREREGISTRATION["gates"]["tiny_logz_parity"]
    _dump("log_probability_parity.json", {
        "corpus_seed": seed2, "n_traces": len(traces),
        "max_abs_complete_data_log_prob_difference": max(diffs),
        "gate": parity_gate, "all_likelihoods_finite": all_finite,
        "generator_decomposition": "log p(S | J, delta_B) [normalized, exact DP] "
                                   "+ log p(z | pi, P) + sum_l scalar-accumulated "
                                   "log p_RFS",
        "inference_decomposition": "targets.log_boundary_prior - log C_J "
                                   "[independent combinatorial reference] + "
                                   "targets.log_path_prior + "
                                   "RecurrentBlockScorer.score (vectorized batch "
                                   "path)",
        "tiny_logz": {"corpus_seed": seed3, "rows": logz_rows,
                      "max_abs_error": logz_worst, "gate": logz_gate},
    })
    _gate(gates, "complete_data_log_prob_parity", max(diffs), parity_gate,
          max(diffs) < parity_gate and all_finite)
    _gate(gates, "tiny_logz_parity", logz_worst, logz_gate,
          logz_worst < logz_gate)


# ---------------------------------------------------------------- negative controls
def run_negative_controls(gates: list) -> None:
    print("== negative controls ==")
    truth = msg.supplied_truth()
    report = {}

    # 1. old block-count mechanism, exact conditional distribution at J = 32
    J = 32
    widths = np.arange(3, 13)
    width_p = (0.85 ** (widths - 1)) / (0.85 ** (widths - 1)).sum()
    old = np.zeros(J // 3 + 1)
    for L in (4, 5, 6):
        pmf = np.array([1.0])
        for _ in range(L):
            nxt = np.zeros(len(pmf) + 12)
            for w, p in zip(widths, width_p):
                nxt[w:w + len(pmf)] += p * pmf
            pmf = nxt
        if J < len(pmf):
            old[L] = pmf[J] / 3.0
    old = old / old.sum()
    exact = mgd.exact_segment_count_distribution(J, 0.15, 3, 12)
    n = max(len(old), len(exact))
    tv_old = mgd.total_variation(np.pad(old, (0, n - len(old))),
                                 np.pad(exact, (0, n - len(exact))))
    report["old_block_count_mechanism"] = {
        "J": J, "tv_vs_model": tv_old, "registered_gate": 0.01,
        "detected": tv_old > 0.01,
        "note": "exact conditional distribution of the old L~Uniform{4,5,6} + iid "
                "widths mechanism given total length J, vs the exact matched "
                "prior p(L | J)",
    }

    # 2. recurrent-state leakage across block boundaries
    rng = np.random.default_rng(
        PREREGISTRATION["seeds"]["negative_control_state_leak"])
    params = truth.rfs_parameters()
    worst = 0.0
    for _ in range(20):
        q = np.zeros(truth.n_roles)              # deliberately carried across blocks
        leaky_blocks, leaky_lls = [], []
        for width, skill in zip((4, 5, 4, 5), (0, 1, 2, 0)):
            u = truth.u_by_skill[skill]
            precedence = precedence_from_u(u)
            roles, total = [], 0.0
            for _ in range(width):
                mixed = recurrent_step_probabilities(u, q, params)
                y = int(rng.choice(truth.n_roles, p=mixed))
                roles.append(y)
                total += math.log(float(mixed[y]))
                q = recurrent_validity_update(y, precedence, q,
                                              params.shared_omega)
            leaky_blocks.append(roles)
            leaky_lls.append(total)
        for roles, skill, leaky_ll in zip(leaky_blocks, (0, 1, 2, 0), leaky_lls):
            production = recurrent_rfs_log_likelihood(
                roles, truth.u_by_skill[skill], truth.beta, truth.epsilon,
                truth.omega, truth.lambda_rep, truth.lambda_back)
            worst = max(worst, abs(production - leaky_ll))
    report["recurrent_state_leakage"] = {
        "max_abs_log_prob_discrepancy": worst, "registered_gate": 1e-10,
        "detected": worst > 1e-10,
        "note": "q carried across block boundaries; the production q_0 = 0 replay "
                "disagrees with the leaky generator's accumulated log-probability",
    }

    # 3. terminal-block boundary convention
    correct = mgd.exact_normalizer(10, 0.15, 3, 12)
    faulty = correct + math.log(0.15)
    enum = mgd.log_normalizer_from_enumeration(10, 0.15, 3, 12)
    report["terminal_block_delta_factor"] = {
        "J": 10, "abs_error_correct_vs_enumeration": abs(correct - enum),
        "abs_error_faulty_vs_enumeration": abs(faulty - enum),
        "registered_gate": 1e-12,
        "detected": abs(faulty - enum) > 1e-12 and abs(correct - enum) < 1e-12,
        "note": "charging the final block a delta_B factor shifts every state by "
                "the same constant, so the normalizer (and cross-side log-prob "
                "parity) detects it; the normalized state distribution alone "
                "cannot, and the registered checks therefore include both",
    }

    # 4. self-transitions injected into P
    bad_p = truth.transition.copy()
    bad_p[0, 0] = 0.10
    bad_p[0] = bad_p[0] / bad_p[0].sum()
    try:
        msg.validate_truth(msg.MatchedTruth(
            u_by_skill=truth.u_by_skill, pi=truth.pi, transition=bad_p,
            beta=truth.beta, omega=truth.omega, lambda_rep=truth.lambda_rep,
            lambda_back=truth.lambda_back, epsilon=truth.epsilon,
            delta_b=truth.delta_b, min_width=truth.min_width,
            max_width=truth.max_width, role_maps=truth.role_maps))
        detected_self = False
    except AssertionError:
        detected_self = True
    report["self_transitions"] = {
        "detected": detected_self,
        "note": "nonzero P[0, 0] injected; validate_truth must refuse it",
    }

    _dump("negative_controls.json", report)
    all_detected = all(v["detected"] for v in report.values())
    _gate(gates, "all_negative_controls_detected",
          {k: v["detected"] for k, v in report.items()}, "all True", all_detected)


# ------------------------------------------------------------------- reproducibility
def run_reproducibility(gates: list) -> None:
    print("== reproducibility ==")
    seed_a = PREREGISTRATION["seeds"]["reproducibility_master_a"]
    seed_b = PREREGISTRATION["seeds"]["reproducibility_master_b"]
    lengths = ([24, 32, 40, 48], [24, 32])
    a1 = msg.generate_corpus(seed_a, *lengths)
    a2 = msg.generate_corpus(seed_a, *lengths)
    b = msg.generate_corpus(seed_b, *lengths)
    text_a1 = msg.canonical_json(msg.corpus_to_jsonable(a1))
    text_a2 = msg.canonical_json(msg.corpus_to_jsonable(a2))
    hash_a1, hash_a2 = msg.sha256_hex(text_a1), msg.sha256_hex(text_a2)
    hash_b = msg.corpus_hash(b)
    reloaded = json.loads(text_a1)
    roundtrip = msg.canonical_json(reloaded) == text_a1
    ok = (text_a1 == text_a2 and hash_a1 == hash_a2 and hash_a1 != hash_b
          and roundtrip)
    _dump("reproducibility.json", {
        "seed_a": seed_a, "seed_b": seed_b,
        "same_seed_byte_identical": text_a1 == text_a2,
        "same_seed_hash": hash_a1,
        "different_seed_hash": hash_b,
        "different_seed_hash_differs": hash_a1 != hash_b,
        "save_load_byte_identical": roundtrip,
        "hash_algorithm": "sha256 over canonical JSON (sorted keys, no "
                          "whitespace, no timestamps, no paths)",
    })
    _gate(gates, "reproducibility_byte_identical_and_hashes", ok, True, ok)


# --------------------------------------------------------------------------- report
def write_report(gates: list) -> bool:
    all_pass = all(g["pass"] for g in gates)
    lines = [
        "# Matched synthetic generator — validation report",
        "",
        f"Source commit: `{_git('rev-parse', 'HEAD')}` "
        f"(branch `{_git('rev-parse', '--abbrev-ref', 'HEAD')}`)",
        f"Generator version: {msg.GENERATOR_VERSION}",
        "",
        "The generator implements the registered factorization "
        "`p(U, rho) p(S, z | delta_B, pi, P) p(X | S, z, h(U), vartheta)` with the "
        "exact normalized segmentation prior "
        "`p(S | J, delta_B) = delta_B^(L-1) (1-delta_B)^(J-L) / C_J(delta_B)` over "
        "contiguous segmentations with widths in [3, 12], sampled by the "
        "suffix-DP `G(r)` recursion (exact by telescoping; see "
        "`matched_segmentation_prior.py`). The old Stage 6E2 block-count "
        "mechanism (`L ~ Uniform{4,5,6}` then iid widths) is preserved untouched "
        "in `stage6e_corpus.py` and banned from the new generator by a "
        "source-level regression test.",
        "",
        "## Registered gates",
        "",
        "| gate | value | threshold | pass |",
        "|---|---|---|---|",
    ]
    for g in gates:
        value = g["value"]
        if isinstance(value, float):
            value = f"{value:.3e}"
        lines.append(f"| {g['gate']} | {value} | {g['threshold']} | "
                     f"{'PASS' if g['pass'] else 'FAIL'} |")
    lines += [
        "",
        f"## Verdict: {'ALL GATES PASS' if all_pass else 'GATE FAILURE — STOP'}",
        "",
        "No MCMC, no FFBS, no conditional- or collapsed-U inference, and no "
        "Condition A/B/C/D experiment was run. Artifacts in this directory are "
        "the complete registered validation record.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    return all_pass


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _dump("preregistration.json", PREREGISTRATION)
    write_source_manifest()
    gates: list = []
    run_exact_reference_checks(gates)
    run_segmentation_parity(gates)
    run_tiny_state_parity(gates)
    run_skill_label_parity(gates)
    run_recurrent_and_log_prob_parity(gates)
    run_negative_controls(gates)
    run_reproducibility(gates)
    all_pass = write_report(gates)
    print(f"\n{'ALL GATES PASS' if all_pass else 'GATE FAILURE — STOP'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
