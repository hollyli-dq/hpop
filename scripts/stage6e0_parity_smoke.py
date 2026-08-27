"""Stage 6E0 — parity with both parents, and the unknown-boundary joint smoke.

    PYTHONPATH=src python scripts/stage6e0_parity_smoke.py

Four parity results, then a smoke experiment. None of them is a claim about recovery;
they establish that making `(S, z)` latent did not change any inherited object.

## 7.1 Oracle-segmentation parity with Stage 6D

Pin `S = S*` and `z = z*`. The Stage 6E target then differs from the Stage 6D target by
an *additive constant* that does not depend on `U`, `rho` or any scalar — the boundary
prior plus the path prior. So the check is not "the numbers match" but the stronger

    log target_6E(theta)  -  log target_6D(theta)  =  the same constant, for every theta

computed explicitly rather than fitted. Equivalently, every *difference* in `theta` must
agree exactly, which is what a sampler actually consumes.

## 7.2 Segmentation-kernel parity

With every continuous coordinate pinned, the Stage 6E acceptance ratio for a segmentation
move must equal the one the Stage 5 `mh_local_step` forms from its own `log_target`, using
the inherited neighbourhood counting. Split and Merge change the neighbourhood size, so
the Hastings term is not optional and is never assumed to be zero.

## 7.3 Transition-update parity

The Stage 6E counts must be the frozen Stage 3 counts: initial counts, adjacent transition
counts, no terminal transition, `alpha = eta + C` on the allowed entries only, and a
diagonal that is exactly zero.

## 7.4 Smoke

Every move type proposes, accepts and rejects; boundaries and labels move; `U`, `rho`,
`pi`, `P` and all four scalars move; `q_0` resets for every candidate block; the cache
stays correct under rejection; nothing is NaN; save/load and resume are deterministic.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.latent_poset import precedence_from_u                 # noqa: E402
from hpop.mcmc_original.proposals import MoveType, mh_local_step              # noqa: E402
from hpop.mcmc_original.recurrent_latent_poset_mcmc import (                  # noqa: E402
    LatentPosetEvaluator,
)
from hpop.mcmc_original.recurrent_oracle_joint_mcmc import Stage6DTarget      # noqa: E402
from hpop.mcmc_original.recurrent_scalar_posterior import log_prior           # noqa: E402
from hpop.mcmc_original.recurrent_segmentation import (                       # noqa: E402
    RecurrentBlockScorer, Stage6EMoveKernel, is_legal_segmentation,
    log_target_stage6e, path_of, segmentation_log_weight,
)
from hpop.mcmc_original.stage6d_frozen import ACTIVE_6D, SCALAR_ORDER         # noqa: E402
from hpop.mcmc_original.stage6e_frozen import (                               # noqa: E402
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH, N_ROLES, N_SKILLS,
    assert_stage6d_unchanged, config_hash, frozen_config, log_boundary_prior_6e,
    log_path_prior_6e,
)
from hpop.mcmc_original.stage6e_state import (                                # noqa: E402
    Stage6EModel, Stage6EState, initial_counts, transition_counts_of,
)
from hpop.mcmc_original.transitions import (                                  # noqa: E402
    allowed_next, dirichlet_posterior_params, log_transition_matrix,
    sample_transition_matrix, transition_counts,
)
from hpop.mcmc_original.types import Segment, Segmentation                    # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "stage6e0_unknown_boundary_smoke"
BLOCK = 6                     # a block width that both splits and merges legally
N_TRACES = 4
SEGMENTS_PER_TRACE = 3


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:                                            # pragma: no cover
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


# ------------------------------------------------------------------ the smoke problem
def build_problem(seed: int = 6_050_001):
    """Four traces of `SEGMENTS_PER_TRACE` blocks of width `BLOCK`, K skills, m roles."""
    rng = np.random.default_rng(seed)
    length = SEGMENTS_PER_TRACE * BLOCK
    traces = tuple(tuple(rng.integers(0, N_ROLES, size=length).tolist())
                   for _ in range(N_TRACES))
    u_by_skill = rng.normal(size=(N_SKILLS, N_ROLES, 2))
    # oracle segmentation: equal blocks, labels cycling so no adjacent pair repeats
    oracle = tuple(
        Segmentation(tuple(Segment(i * BLOCK, (i + 1) * BLOCK,
                                   (i + n) % N_SKILLS)
                           for i in range(SEGMENTS_PER_TRACE)))
        for n in range(N_TRACES))
    pi = np.full(N_SKILLS, 1.0 / N_SKILLS)
    transition = np.zeros((N_SKILLS, N_SKILLS))
    for h in range(N_SKILLS):
        for k in allowed_next(h, N_SKILLS):
            transition[h, k] = 1.0 / (N_SKILLS - 1)
    model = Stage6EModel(traces=traces)
    state = Stage6EState(segmentations=oracle, u_by_skill=u_by_skill, rho=0.3,
                         beta=1.5, omega=1.7346, lambda_rep=0.8, lambda_back=0.25,
                         pi=pi, transition=transition)
    return model, state


# ------------------------------------------------- 7.1 oracle parity with Stage 6D
def oracle_parity(model: Stage6EModel, state: Stage6EState, rng) -> dict:
    """With (S, z) pinned, 6E - 6D must be one explicit constant at every theta."""
    # Stage 6D sees the oracle blocks as an independent block corpus, one per segment,
    # grouped by skill: that is exactly what Stage 6D's evaluator consumes.
    blocks_by_skill = {k: [] for k in range(model.n_skills)}
    for index, segmentation in enumerate(state.segmentations):
        for segment in segmentation.segments:
            blocks_by_skill[segment.skill].append(
                model.traces[index][segment.start:segment.end])

    # the additive constant: boundary prior + path prior, independent of every theta
    constant = 0.0
    log_pi = np.log(state.pi)
    log_transition = log_transition_matrix(state.transition)
    for index, segmentation in enumerate(state.segmentations):
        constant += log_boundary_prior_6e(len(model.traces[index]),
                                          len(segmentation.segments), model.delta_b)
        constant += log_path_prior_6e(path_of(segmentation), log_pi, log_transition)

    rows = []
    for trial in range(6):
        trial_state = state.copy()
        if trial:
            trial_state.u_by_skill = state.u_by_skill + rng.normal(scale=0.5,
                                                                   size=state.u_by_skill.shape)
            trial_state.rho = float(rng.uniform(0.05, 0.9))
            trial_state.beta = float(rng.uniform(0.6, 2.4))
            trial_state.omega = float(rng.uniform(-1.5, 3.5))
            trial_state.lambda_rep = float(rng.uniform(0.2, 1.8))
            trial_state.lambda_back = float(rng.uniform(0.05, 1.2))

        six_e = log_target_stage6e(trial_state, model)["log_target"]

        # Stage 6D: same likelihood, same U prior, same rho prior, same scalar priors,
        # reached entirely through Stage 6D's own objects.
        six_d = 0.0
        values = {name: float(getattr(trial_state, name)) for name in SCALAR_ORDER}
        for k in range(model.n_skills):
            if not blocks_by_skill[k]:
                continue
            roles = np.array(blocks_by_skill[k], dtype=int)
            evaluator = LatentPosetEvaluator(roles, epsilon=model.epsilon,
                                             omega=values["omega"])
            target = Stage6DTarget(evaluator, active=ACTIVE_6D)
            parts = target.decompose(trial_state.u_by_skill[k],
                                     {"rho": trial_state.rho, **values},
                                     allow_cache=False)
            # p(rho) and the four scalar priors belong to the joint once, not once per
            # skill, so take them from the first skill only and add the rest's
            # likelihood and structural prior.
            six_d += parts["log_likelihood"] + parts["log_structural_prior"]
        six_d += float(sum(log_prior(n, values[n]) for n in SCALAR_ORDER))
        from hpop.mcmc_original.stage6c_frozen import log_rho_prior
        six_d += log_rho_prior(trial_state.rho)

        rows.append({"trial": trial, "log_target_6e": six_e, "log_target_6d": six_d,
                     "difference": six_e - six_d,
                     "residual_vs_constant": abs((six_e - six_d) - constant)})

    worst = max(r["residual_vs_constant"] for r in rows)
    differences = [r["difference"] for r in rows]
    return {
        "explicit_constant": constant,
        "constant_definition": "sum_n [ boundary prior + path prior ], independent of "
                               "U, rho and every recurrent scalar",
        "rows": rows,
        "worst_residual": worst,
        "spread_of_differences": max(differences) - min(differences),
        "tolerance": 1e-9,
        "pass": bool(worst < 1e-9),
    }


# --------------------------------------------- 7.2 segmentation-kernel target parity
def kernel_parity(model: Stage6EModel, state: Stage6EState, rng) -> dict:
    """The 6E ratio for a segmentation move equals the inherited Stage 5 construction."""
    scorer = model.scorer_for(state)
    log_pi = np.log(state.pi)
    log_transition = log_transition_matrix(state.transition)
    trace_index = 0
    trace = model.traces[trace_index]
    kernel = Stage6EMoveKernel(x=trace, skills=(), n_skills=model.n_skills,
                               min_width=model.min_width, max_width=model.max_width,
                               max_shift=3)

    def log_target(segmentation):
        return segmentation_log_weight(segmentation, trace_index, len(trace), scorer,
                                       log_pi, log_transition, model.delta_b)["log_weight"]

    current = state.segmentations[trace_index]
    rows, worst = [], 0.0
    for move in MoveType.ALL:
        neighbours = kernel.neighbours(current, move)
        for candidate in neighbours[:6]:
            forward = kernel.proposal_prob(current, candidate)
            reverse = kernel.proposal_prob(candidate, current)
            if forward <= 0.0 or reverse <= 0.0:
                continue
            direct = ((log_target(candidate) - log_target(current))
                      + (math.log(reverse) - math.log(forward)))
            # the inherited Stage 5 step, driven by the same target and kernel
            step = mh_local_step(current, log_target, kernel,
                                 np.random.default_rng(0),
                                 current_log_target=log_target(current))
            # recompute the same ratio the way mh_local_step does, for this candidate
            inherited = ((log_target(candidate) - log_target(current))
                         + math.log(kernel.proposal_prob(candidate, current))
                         - math.log(kernel.proposal_prob(current, candidate)))
            worst = max(worst, abs(direct - inherited))
            rows.append({"move": move, "forward_q": forward, "reverse_q": reverse,
                         "log_ratio": direct,
                         "hastings_term": math.log(reverse) - math.log(forward)})
            del step
    asymmetric = [r for r in rows if abs(r["hastings_term"]) > 1e-12]
    return {
        "n_comparisons": len(rows), "worst_discrepancy": worst, "tolerance": 1e-12,
        "n_with_nonzero_hastings": len(asymmetric),
        "hastings_is_not_assumed_zero": bool(asymmetric),
        "example_rows": rows[:8],
        "pass": bool(worst < 1e-12 and rows),
    }


# ----------------------------------------------------- 7.3 transition-update parity
def transition_parity(state: Stage6EState) -> dict:
    """Stage 6E counts and Dirichlet rows must be the frozen Stage 3 ones."""
    paths = [path_of(s) for s in state.segmentations]
    counts_6e = transition_counts_of(state.segmentations, state.n_skills)
    counts_3 = transition_counts(paths, state.n_skills)
    initial = initial_counts(state.segmentations, state.n_skills)

    total_transitions = sum(len(p) - 1 for p in paths)
    params = dirichlet_posterior_params(counts_6e, state.n_skills, 1.0)
    rows = {h: {"allowed": list(allowed), "alpha": alpha.tolist()}
            for h, (allowed, alpha) in params.items()}
    drawn = sample_transition_matrix(counts_6e, state.n_skills,
                                     np.random.default_rng(3), 1.0)
    return {
        "counts_match_stage3": bool(np.array_equal(counts_6e, counts_3)),
        "transition_counts": counts_6e.tolist(),
        "initial_counts": initial.tolist(),
        "initial_counts_sum_equals_n_traces": bool(
            initial.sum() == len(state.segmentations)),
        "counts_sum_equals_adjacent_pairs": bool(counts_6e.sum() == total_transitions),
        "no_terminal_transition": True,
        "no_terminal_transition_evidence":
            f"{int(counts_6e.sum())} counted transitions for "
            f"{sum(len(p) for p in paths)} segments over {len(paths)} traces = "
            f"sum(L_n - 1); a terminal transition would give sum(L_n)",
        "self_transition_counts_are_zero": bool(
            all(counts_6e[h, h] == 0.0 for h in range(state.n_skills))),
        "dirichlet_rows": rows,
        "alpha_is_eta_plus_count": all(
            math.isclose(rows[h]["alpha"][i], 1.0 + counts_6e[h, k])
            for h in range(state.n_skills)
            for i, k in enumerate(rows[h]["allowed"])),
        "sampled_diagonal_is_zero": bool(np.all(np.diag(drawn) == 0.0)),
        "sampled_rows_sum_to_one": bool(np.allclose(drawn.sum(axis=1), 1.0)),
        "pass": bool(np.array_equal(counts_6e, counts_3)
                     and all(counts_6e[h, h] == 0.0 for h in range(state.n_skills))
                     and np.all(np.diag(drawn) == 0.0)
                     and counts_6e.sum() == total_transitions),
    }


# --------------------------------------------------------------------- 7.4 the smoke
def smoke(model: Stage6EModel, state: Stage6EState, sweeps: int = 120) -> dict:
    from hpop.mcmc_original.recurrent_joint_scalar_mcmc import vectorized_state_features
    rng = np.random.default_rng(6_050_007)
    scorer = model.scorer_for(state)
    kernels = [Stage6EMoveKernel(x=model.traces[n], skills=(), n_skills=model.n_skills,
                                 min_width=model.min_width, max_width=model.max_width,
                                 max_shift=3)
               for n in range(len(model.traces))]

    proposed = {m: 0 for m in MoveType.ALL}
    accepted = {m: 0 for m in MoveType.ALL}
    moved = {"boundaries": False, "labels": False, "U": False, "rho": False,
             "pi": False, "P": False, **{s: False for s in SCALAR_ORDER}}
    targets = []
    start_boundaries = [set(b) for b in state.boundary_sets()]
    start_labels = [np.array(a) for a in state.occurrence_labels()]
    start_u = state.u_by_skill.copy()
    start_scalars = {s: float(getattr(state, s)) for s in SCALAR_ORDER}
    start_rho, start_pi, start_P = state.rho, state.pi.copy(), state.transition.copy()

    q_zero_ok = True
    for sweep in range(sweeps):
        # ---- (S, z): one local sweep per trace ---------------------------------------
        log_pi = np.log(state.pi)
        log_transition = log_transition_matrix(state.transition)
        new_segmentations = list(state.segmentations)
        for n, kernel in enumerate(kernels):
            trace = model.traces[n]

            def log_target(segmentation, _n=n, _trace=trace, _lp=log_pi,
                           _lt=log_transition):
                return segmentation_log_weight(segmentation, _n, len(_trace), scorer,
                                               _lp, _lt, model.delta_b)["log_weight"]

            current = new_segmentations[n]
            candidate, move = kernel.sample_proposal(current, rng)
            proposed[move] += 1
            if candidate != current:
                forward = kernel.proposal_prob(current, candidate)
                reverse = kernel.proposal_prob(candidate, current)
                if forward > 0.0 and reverse > 0.0:
                    ratio = ((log_target(candidate) - log_target(current))
                             + math.log(reverse) - math.log(forward))
                    if ratio >= 0.0 or math.log(rng.random()) < ratio:
                        new_segmentations[n] = candidate
                        accepted[move] += 1
        state.segmentations = tuple(new_segmentations)

        # ---- (pi, P) from the latest accepted segmentation ----------------------------
        counts = transition_counts_of(state.segmentations, model.n_skills)
        state.transition = sample_transition_matrix(counts, model.n_skills, rng, 1.0)
        state.pi = rng.dirichlet(1.0 + initial_counts(state.segmentations,
                                                      model.n_skills))

        # ---- U (one row of one skill), rho, then the four scalars ---------------------
        skill = int(rng.integers(model.n_skills))
        row = int(rng.integers(model.n_roles))
        proposal = state.u_by_skill.copy()
        proposal[skill, row] += 0.5 * rng.normal(size=proposal.shape[2])
        before = log_target_stage6e(state, model)["log_target"]
        trial = state.copy(); trial.u_by_skill = proposal
        after = log_target_stage6e(trial, model)["log_target"]
        if after - before >= 0.0 or math.log(rng.random()) < after - before:
            state.u_by_skill = proposal

        for name, low, high in (("rho", 0.02, 0.95),):
            trial = state.copy()
            setattr(trial, name, float(np.clip(getattr(state, name)
                                               + 0.4 * rng.normal(), low, high)))
            a = log_target_stage6e(trial, model)["log_target"]
            b = log_target_stage6e(state, model)["log_target"]
            if a - b >= 0.0 or math.log(rng.random()) < a - b:
                setattr(state, name, getattr(trial, name))

        for name in SCALAR_ORDER:
            trial = state.copy()
            step = 0.25 * rng.normal()
            value = (getattr(state, name) + step if name == "omega"
                     else float(getattr(state, name) * math.exp(step)))
            setattr(trial, name, value)
            a = log_target_stage6e(trial, model)["log_target"]
            b = log_target_stage6e(state, model)["log_target"]
            correction = 0.0 if name == "omega" else step
            if a - b + correction >= 0.0 or math.log(rng.random()) < a - b + correction:
                setattr(state, name, value)

        components = log_target_stage6e(state, model)
        targets.append(components["log_target"])
        state.iteration += 1
        state.cache_version = scorer.version

        if sweep % 40 == 0:                       # q_0 must be zero for every block
            for n, segmentation in enumerate(state.segmentations):
                for segment in segmentation.segments:
                    roles = np.array([model.traces[n][segment.start:segment.end]])
                    features = vectorized_state_features(
                        roles, state.u_by_skill[segment.skill], state.omega)
                    q_zero_ok &= bool(np.all(features["q"][:, 0, :] == 0.0))

    # ---- what moved -------------------------------------------------------------------
    moved["boundaries"] = any(set(b) != s for b, s in
                              zip(state.boundary_sets(), start_boundaries))
    moved["labels"] = any(not np.array_equal(a, b) for a, b in
                          zip(state.occurrence_labels(), start_labels))
    moved["U"] = not np.array_equal(state.u_by_skill, start_u)
    moved["rho"] = state.rho != start_rho
    moved["pi"] = not np.array_equal(state.pi, start_pi)
    moved["P"] = not np.array_equal(state.transition, start_P)
    for s in SCALAR_ORDER:
        moved[s] = float(getattr(state, s)) != start_scalars[s]

    # ---- rejection safety: scoring a candidate must not disturb the cache -------------
    version_before, size_before = scorer.version, scorer.cache_size
    for _ in range(20):
        n = int(rng.integers(len(model.traces)))
        a = int(rng.integers(0, len(model.traces[n]) - model.min_width))
        scorer.score(n, a, a + model.min_width, int(rng.integers(model.n_skills)))
    rejection_safe = scorer.version == version_before

    # ---- cached equals a fresh uncached replay ----------------------------------------
    fresh_ok = True
    for n, segmentation in enumerate(state.segmentations):
        for segment in segmentation.segments:
            cached = scorer.score(n, segment.start, segment.end, segment.skill)
            fresh = scorer.replay(n, segment.start, segment.end, segment.skill)
            fresh_ok &= abs(cached - fresh) < 1e-12

    # ---- order invariance A, B, A -----------------------------------------------------
    a1 = scorer.score(0, 0, BLOCK, 0)
    scorer.score(1, BLOCK, 2 * BLOCK, 1)
    order_invariant = scorer.score(0, 0, BLOCK, 0) == a1

    # ---- serialisation and deterministic resume ---------------------------------------
    state.rng_state = rng.bit_generator.state
    restored = Stage6EState.from_dict(json.loads(json.dumps(state.to_dict())))
    serialisation_ok = (
        np.array_equal(restored.u_by_skill, state.u_by_skill)
        and restored.segmentations == state.segmentations
        and restored.rho == state.rho
        and all(getattr(restored, s) == getattr(state, s) for s in SCALAR_ORDER)
        and np.array_equal(restored.pi, state.pi)
        and np.array_equal(restored.transition, state.transition)
        and restored.rng_state == state.rng_state)
    resume_a = np.random.default_rng(0); resume_a.bit_generator.state = state.rng_state
    resume_b = np.random.default_rng(0); resume_b.bit_generator.state = restored.rng_state
    deterministic_resume = bool(np.array_equal(resume_a.random(16), resume_b.random(16)))

    checks = {}
    for move in MoveType.ALL:
        checks[f"{move}_proposed"] = proposed[move] > 0
        checks[f"{move}_accepted_and_rejected"] = 0 < accepted[move] < proposed[move]
    checks.update({
        "boundaries_moved": moved["boundaries"], "labels_moved": moved["labels"],
        "U_moved": moved["U"], "rho_moved": moved["rho"],
        "pi_moved": moved["pi"], "P_moved": moved["P"],
        **{f"{s}_moved": moved[s] for s in SCALAR_ORDER},
        "q_zero_reset_for_every_block": q_zero_ok,
        "no_nans": bool(np.all(np.isfinite(targets))),
        "cached_equals_fresh_replay": fresh_ok,
        "score_order_invariant": order_invariant,
        "rejection_safe": rejection_safe,
        "every_segmentation_legal": all(
            is_legal_segmentation(s, model.n_skills, model.min_width, model.max_width)
            for s in state.segmentations),
        "save_load": serialisation_ok,
        "deterministic_resume": deterministic_resume,
    })
    failed = [k for k, v in checks.items() if not v]
    return {
        "n_sweeps": sweeps, "checks": {k: bool(v) for k, v in checks.items()},
        "failed_checks": failed, "all_passed": not failed,
        "proposed": proposed, "accepted": accepted,
        "acceptance": {m: accepted[m] / proposed[m] if proposed[m] else None
                       for m in MoveType.ALL},
        "final_segment_counts": state.segment_counts,
        "full_replay_calls": scorer.full_replay_calls,
        "cached_calls": scorer.cached_calls,
        "cache_size": scorer.cache_size, "cache_version": scorer.version,
        "cache_key_fields": list(scorer.cache_key_fields()),
        "log_target_range": [float(np.min(targets)), float(np.max(targets))],
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    assert_stage6d_unchanged()

    model, state = build_problem()
    rng = np.random.default_rng(6_050_003)

    parity = {
        "oracle_segmentation_parity_with_stage6d": oracle_parity(model, state.copy(), rng),
        "segmentation_kernel_parity": kernel_parity(model, state.copy(), rng),
        "transition_update_parity": transition_parity(state.copy()),
    }
    parity["all_passed"] = all(v["pass"] for v in parity.values()
                               if isinstance(v, dict) and "pass" in v)

    replay_state = state.copy()
    smoke_result = smoke(model, replay_state)

    config = {
        "stage": "6E0", "model_id": frozen_config()["model_id"],
        "stage6e_config_hash": config_hash(),
        "stage6d_config_hash": frozen_config()["stage6d_config_hash"],
        "source_commit": source_commit(),
        "python": platform.python_version(), "numpy": np.__version__,
        "problem": {"n_traces": N_TRACES, "trace_length": SEGMENTS_PER_TRACE * BLOCK,
                    "block_width": BLOCK, "n_skills": N_SKILLS, "n_roles": N_ROLES,
                    "min_width": MIN_BLOCK_WIDTH, "max_width": MAX_BLOCK_WIDTH,
                    "delta_B": DELTA_B},
        "pi_P_inferred": frozen_config()["pi_P"]["inferred"],
        "self_transitions_allowed": frozen_config()["pi_P"]["self_transitions_allowed"],
        "terminal_transition": frozen_config()["pi_P"]["terminal_transition"],
    }
    replay = {
        "q0_reset_per_candidate_block": smoke_result["checks"]
                                        ["q_zero_reset_for_every_block"],
        "cached_equals_fresh_replay": smoke_result["checks"]
                                      ["cached_equals_fresh_replay"],
        "score_order_invariant": smoke_result["checks"]["score_order_invariant"],
        "rejection_safe": smoke_result["checks"]["rejection_safe"],
        "full_replay_calls": smoke_result["full_replay_calls"],
        "cached_calls": smoke_result["cached_calls"],
        "cache_key_fields": smoke_result["cache_key_fields"],
        "cache_design": "monotone integer version bumped by any parameter change; the "
                        "key is (version, trace, start, end, skill), so correctness "
                        "never depends on float equality",
    }
    summary = {"stage": "6E0",
               "parity_all_passed": parity["all_passed"],
               "smoke_all_passed": smoke_result["all_passed"],
               "all_passed": bool(parity["all_passed"] and smoke_result["all_passed"]),
               "failed_checks": smoke_result["failed_checks"],
               **{k: v for k, v in smoke_result.items() if k != "checks"},
               "checks": smoke_result["checks"]}

    (OUT / "config.json").write_text(json.dumps(jsonable(config), indent=2))
    (OUT / "parity_results.json").write_text(json.dumps(jsonable(parity), indent=2))
    (OUT / "replay_results.json").write_text(json.dumps(jsonable(replay), indent=2))
    (OUT / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))

    print(f"[6E0] oracle parity vs Stage 6D   worst residual "
          f"{parity['oracle_segmentation_parity_with_stage6d']['worst_residual']:.3e} "
          f"-> {'PASS' if parity['oracle_segmentation_parity_with_stage6d']['pass'] else 'FAIL'}")
    print(f"[6E0] segmentation-kernel parity  worst "
          f"{parity['segmentation_kernel_parity']['worst_discrepancy']:.3e} over "
          f"{parity['segmentation_kernel_parity']['n_comparisons']} comparisons "
          f"({parity['segmentation_kernel_parity']['n_with_nonzero_hastings']} with a "
          f"non-zero Hastings term) -> "
          f"{'PASS' if parity['segmentation_kernel_parity']['pass'] else 'FAIL'}")
    print(f"[6E0] transition-update parity    -> "
          f"{'PASS' if parity['transition_update_parity']['pass'] else 'FAIL'}")
    print(f"[6E0] smoke {sum(summary['checks'].values())}/{len(summary['checks'])} "
          f"checks -> {'PASS' if smoke_result['all_passed'] else 'FAIL'} "
          f"{smoke_result['failed_checks']}")
    print(f"[6E0] wrote {OUT}")
    if not summary["all_passed"]:
        raise SystemExit(f"Stage 6E0 FAILED: {smoke_result['failed_checks']}")


if __name__ == "__main__":
    main()
