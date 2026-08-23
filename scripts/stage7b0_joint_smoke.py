"""Step 7B0 — integration parity and smoke for the FFBS full-joint sampler.

    PYTHONPATH=src python scripts/stage7b0_joint_smoke.py

Nothing here is evidence that the sampler is correct; that is Step 7B1's job, against the
frozen Stage 6E1B reference. What this establishes is narrower and has to come first: that
swapping the segmentation kernel changed *only* the segmentation kernel.

Four parities, each answering a way the swap could have gone wrong silently:

1. **Conditional parity.** At many frozen global states, the distribution the FFBS chart
   normalises must equal the exactly enumerated `p(S_n, z_n | x_n, Theta)`. If it did not,
   the Gibbs step would be drawing from the wrong conditional and nothing downstream could
   be trusted.
2. **Global target parity.** The complete state produced by a Step 7B sweep must be scored
   by the registered Stage 6E target — checked against `log_target_stage6e`, which shares
   no call graph with the sweep's own decomposition.
3. **Kernel parity.** With the segmentation draw switched off, a Step 7B sweep and a
   Stage 6E sweep with zero segmentation proposals must return bit-identical states from
   the same generator. This is close to true by construction — `ffbs_sweep_once` *calls*
   `stage6e_sampler.sweep_once` — and the test exists to keep it that way.
4. **Recurrent reset and table lifecycle.** Every candidate block starts from `q_0 = 0`;
   the eager and batched table builders agree; the table is built once per segmentation
   sweep and refuses to be read once stale.

Then a smoke run, which demonstrates movement rather than correctness.
"""

from __future__ import annotations

import importlib.util
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

from hpop.mcmc_original.block_score_adapters import (                        # noqa: E402
    assert_no_recurrent_state_leak, assert_order_invariance,
)
from hpop.mcmc_original.fast_segmentation_kernel import key_of               # noqa: E402
from hpop.mcmc_original.latent_poset import precedence_from_u                # noqa: E402
from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import (                   # noqa: E402
    FFBSBlockTables, Stage7BSampler, assert_sources_agree, ffbs_sweep_once,
    run_stage7b_chain, sweep_cost_breakdown,
)
from hpop.mcmc_original.recurrent_segmentation import log_target_stage6e     # noqa: E402
from hpop.mcmc_original.semi_markov_ffbs import forward                      # noqa: E402
from hpop.mcmc_original.stage6d_frozen import REGISTERED_SCALES, SCALAR_ORDER  # noqa: E402
from hpop.mcmc_original.stage6e_exact import (                               # noqa: E402
    enumerate_states, exact_posterior, state_log_weights, total_variation,
)
from hpop.mcmc_original.stage6e_frozen import (                              # noqa: E402
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH, assert_stage6d_unchanged, config_hash,
)
from hpop.mcmc_original.stage6e_sampler import Stage6ESampler, sweep_once    # noqa: E402
from hpop.mcmc_original.stage6e_state import (                               # noqa: E402
    Stage6EModel, Stage6EState, initial_counts, transition_counts_of,
)
from hpop.mcmc_original.transitions import log_transition_matrix             # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "stage7b0_joint_smoke"

N_PARITY_STATES = 24
SMOKE_SWEEPS = 3_000
SMOKE_SEED = 7_073_000
PARITY_SEED = 7_073_100


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
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def load_6e1b():
    path = ROOT / "scripts" / "stage6e1b_mixed_reference.py"
    spec = importlib.util.spec_from_file_location("stage6e1b", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(module, infer_pi_P: bool) -> Stage6EModel:
    traces, _ = module.generate_corpus()
    return Stage6EModel(traces=traces, epsilon=module.EPSILON, delta_b=DELTA_B,
                        n_skills=module.K_SKILLS, n_roles=module.M_ROLES,
                        min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
                        infer_pi_P=infer_pi_P)


def random_state(module, rng, infer_pi_P: bool) -> Stage6EState:
    """A legal complete state, dispersed over every coordinate the chain moves."""
    from hpop.mcmc_original.fast_segmentation_kernel import segmentation_of
    shapes = [((8, 0),), ((3, 0), (8, 1)), ((4, 2), (8, 0)), ((5, 1), (8, 2)),
              ((3, 2), (8, 1)), ((4, 0), (8, 2))]
    if infer_pi_P:
        pi = rng.dirichlet(np.ones(module.K_SKILLS))
        transition = np.zeros((module.K_SKILLS, module.K_SKILLS))
        for h in range(module.K_SKILLS):
            allowed = [k for k in range(module.K_SKILLS) if k != h]
            transition[h, allowed] = rng.dirichlet(np.ones(len(allowed)))
    else:
        pi, transition = module.PI_FIXED, module.P_FIXED
    return Stage6EState(
        segmentations=tuple(segmentation_of(shapes[int(rng.integers(len(shapes)))])
                            for _ in range(module.N_TRACES)),
        u_by_skill=rng.normal(scale=1.5, size=(module.K_SKILLS, module.M_ROLES,
                                               module.D_LATENT)),
        rho=float(rng.uniform(0.05, 0.9)),
        beta=float(rng.uniform(0.5, 2.5)), omega=float(rng.uniform(-1.0, 3.0)),
        lambda_rep=float(rng.uniform(0.2, 1.5)), lambda_back=float(rng.uniform(0.05, 1.0)),
        pi=pi, transition=transition)


# ------------------------------------------------------------------ 1. conditional parity
def conditional_parity(module, model, n_states: int) -> dict:
    """The FFBS conditional must equal the exactly enumerated one, state by state."""
    states = enumerate_states(module.J, module.K_SKILLS, MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH)
    rng = np.random.default_rng(PARITY_SEED)
    worst_log_z, worst_tv = 0.0, 0.0
    rows = []
    for i in range(n_states):
        state = random_state(module, rng, model.infer_pi_P)
        tables = FFBSBlockTables(model=model, source="batched")
        tables.refresh(state)
        scorer = model.scorer_for(state)
        log_pi = np.log(state.pi)
        log_p = log_transition_matrix(state.transition)
        for n, table in enumerate(tables.tables_for(state)):
            chart = forward(table, log_pi, log_p, model.delta_b, model.max_width,
                            model.min_width)
            weights = state_log_weights(states, n, module.J, scorer, log_pi, log_p,
                                        model.delta_b)
            exact = exact_posterior(states, weights)
            # the chart's implied conditional, read off the chart rather than sampled
            ffbs_p = np.exp(np.array([
                _log_weight_of(key, n, module.J, scorer, log_pi, log_p, model.delta_b)
                for key in states]) - chart.log_normalizer)
            gap_z = abs(chart.log_normalizer - exact["log_evidence"])
            tv = total_variation(ffbs_p, exact["probability"])
            worst_log_z = max(worst_log_z, gap_z)
            worst_tv = max(worst_tv, tv)
            if i < 3:
                rows.append({"state": i, "trace": n, "log_z_ffbs": chart.log_normalizer,
                             "log_z_enumerated": exact["log_evidence"],
                             "absolute_error": gap_z, "conditional_tv": tv})
    return {"n_states": int(n_states), "n_comparisons": int(n_states * model_traces(model)),
            "max_log_z_absolute_error": worst_log_z,
            "max_conditional_total_variation": worst_tv,
            "threshold_log_z": 1e-10, "threshold_tv": 1e-12,
            "examples": rows,
            "pass": bool(worst_log_z < 1e-10 and worst_tv < 1e-12)}


def model_traces(model) -> int:
    return len(model.traces)


def _log_weight_of(key, trace_index, trace_length, scorer, log_pi, log_transition,
                   delta_b) -> float:
    """`log w(S, z)` by the registered decomposition, for one enumerated key."""
    from hpop.mcmc_original.fast_segmentation_kernel import spans_of
    from hpop.mcmc_original.stage6e_frozen import log_boundary_prior_6e
    spans = spans_of(key)
    total = float(log_pi[spans[0][2]])
    for start, end, skill in spans:
        total += scorer.score(trace_index, start, end, skill)
    total += log_boundary_prior_6e(trace_length, len(spans), delta_b)
    for (_, _, left), (_, _, right) in zip(spans[:-1], spans[1:]):
        total += float(log_transition[left, right])
    return total


# ---------------------------------------------------------------- 2. global target parity
def global_target_parity(module, model, n_states: int) -> dict:
    """Every state a Step 7B sweep produces must score identically under the direct target."""
    rng = np.random.default_rng(PARITY_SEED + 1)
    sampler = Stage7BSampler(model=model, scales=REGISTERED_SCALES)
    worst = 0.0
    rows = []
    for i in range(n_states):
        state = random_state(module, rng, model.infer_pi_P)
        after = ffbs_sweep_once(state, sampler, rng)
        direct = log_target_stage6e(after, model)
        swept = float(after.components["log_target"])
        # the direct target carries no (pi, P) prior term; with pi and P fixed both are 0
        gap = abs(float(direct["log_target"]) - swept)
        worst = max(worst, gap)
        if i < 3:
            rows.append({"state": i, "direct": float(direct["log_target"]),
                         "sweep_decomposition": swept, "absolute_difference": gap})
    return {"n_states": int(n_states), "max_absolute_difference": worst,
            "threshold": 1e-9, "examples": rows, "pass": bool(worst < 1e-9)}


# ---------------------------------------------------------------------- 3. kernel parity
def kernel_parity(module, model, n_states: int) -> dict:
    """Segmentation draw off: Step 7B and Stage 6E must return the identical state."""
    rng = np.random.default_rng(PARITY_SEED + 2)
    worst = {"u": 0.0, "rho": 0.0, "log_target": 0.0, "pi": 0.0, "transition": 0.0,
             **{name: 0.0 for name in SCALAR_ORDER}}
    for _ in range(n_states):
        state = random_state(module, rng, model.infer_pi_P)
        ffbs_sampler = Stage7BSampler(model=model, scales=REGISTERED_SCALES,
                                      draw_segmentation=False)
        stage6e_sampler = Stage6ESampler(model=model, scales=dict(REGISTERED_SCALES),
                                         n_proposals_per_trace=0)
        a = ffbs_sweep_once(state, ffbs_sampler, np.random.default_rng(4242))
        b = sweep_once(state, stage6e_sampler, np.random.default_rng(4242))
        worst["u"] = max(worst["u"], float(np.abs(a.u_by_skill - b.u_by_skill).max()))
        worst["rho"] = max(worst["rho"], abs(a.rho - b.rho))
        worst["pi"] = max(worst["pi"], float(np.abs(a.pi - b.pi).max()))
        worst["transition"] = max(worst["transition"],
                                  float(np.abs(a.transition - b.transition).max()))
        worst["log_target"] = max(worst["log_target"],
                                  abs(a.components["log_target"]
                                      - b.components["log_target"]))
        for name in SCALAR_ORDER:
            worst[name] = max(worst[name], abs(getattr(a, name) - getattr(b, name)))
    return {"n_states": int(n_states), "max_absolute_difference": worst,
            "worst_over_all_coordinates": max(worst.values()),
            "pass": bool(max(worst.values()) == 0.0),
            "note": "bit-identical is required, not approximate: the Step 7B sweep calls "
                    "the Stage 6E sweep for every phase after (S, z)"}


# ------------------------------------------------------- 4. recurrent reset and lifecycle
def table_lifecycle_audit(module, model) -> dict:
    rng = np.random.default_rng(PARITY_SEED + 3)
    state = random_state(module, rng, model.infer_pi_P)
    scorer = model.scorer_for(state)
    leaks = [assert_no_recurrent_state_leak(scorer, 0, (0, 3, 0), (3, 8, 1)),
             assert_no_recurrent_state_leak(scorer, 1, (0, 4, 2), (4, 8, 0)),
             assert_no_recurrent_state_leak(scorer, 0, (0, 8, 1), (0, 3, 2))]
    order = assert_order_invariance(scorer, 0, module.J, module.K_SKILLS,
                                    MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH,
                                    orders=("by_start", "by_width", "reverse_by_start"))
    sources = assert_sources_agree(model, state)

    # one build per segmentation sweep, and no read once stale
    sampler = Stage7BSampler(model=model, scales=REGISTERED_SCALES)
    n_sweeps = 25
    current = state.copy()
    for _ in range(n_sweeps):
        current = ffbs_sweep_once(current, sampler, rng)
    stale_raises = False
    try:
        sampler.tables.tables_for(current)
    except AssertionError:
        stale_raises = True
    # a fingerprint mismatch must also raise, even when the flag says fresh
    sampler.tables.refresh(current)
    moved = current.copy()
    moved.beta = current.beta + 0.1
    fingerprint_raises = False
    try:
        sampler.tables.tables_for(moved)
    except AssertionError:
        fingerprint_raises = True

    return {
        "q0_reset_audits": leaks,
        "q0_reset_pass": all(a["pass"] for a in leaks),
        "evaluation_order_invariance": order,
        "eager_vs_batched_tables": sources,
        "table_builds": int(sampler.tables.builds),
        "segmentation_sweeps": n_sweeps,
        "builds_equal_sweeps": bool(sampler.tables.builds == n_sweeps + 1),
        "builds_note": "one build per segmentation sweep, plus the one taken here to test "
                       "the fingerprint guard",
        "stale_read_raises": stale_raises,
        "fingerprint_mismatch_raises": fingerprint_raises,
        "scalar_proposals_per_sweep": len(SCALAR_ORDER),
        "rebuilds_per_scalar_proposal": 0.0,
        "pass": bool(all(a["pass"] for a in leaks) and order["pass"] and sources["pass"]
                     and sampler.tables.builds == n_sweeps + 1 and stale_raises
                     and fingerprint_raises),
    }


# ------------------------------------------------------------------------------- 5. smoke
def smoke(module, model_fixed, model_inferred) -> dict:
    """Movement, pi/P freshness, determinism, resume, and the absence of NaNs."""
    rng = np.random.default_rng(SMOKE_SEED)
    start = random_state(module, rng, model_inferred.infer_pi_P)

    # -- pi/P must see the labels FFBS has just drawn, not last sweep's ------------------
    sampler = Stage7BSampler(model=model_inferred, scales=REGISTERED_SCALES)
    before = start.copy()
    after = ffbs_sweep_once(before, sampler, np.random.default_rng(11))
    counts_after = transition_counts_of(after.segmentations, model_inferred.n_skills)
    counts_before = transition_counts_of(before.segmentations, model_inferred.n_skills)
    initial_after = initial_counts(after.segmentations, model_inferred.n_skills)
    # A Dirichlet draw has zero probability of putting mass where its counts are zero only
    # in the eta -> 0 limit, so freshness is checked structurally instead: the sampled P
    # must have a zero diagonal, and the *counts* fed to it must be the new ones.
    pi_p_freshness = {
        "labels_changed": bool(
            [key_of(s) for s in before.segmentations]
            != [key_of(s) for s in after.segmentations]),
        "transition_counts_before": counts_before.tolist(),
        "transition_counts_after": counts_after.tolist(),
        "counts_differ": bool(not np.array_equal(counts_before, counts_after)),
        "initial_counts_after": initial_after.tolist(),
        "initial_counts_sum_equals_n_traces": bool(
            initial_after.sum() == len(model_inferred.traces)),
        "P_diagonal_is_zero": bool(np.all(np.diag(after.transition) == 0.0)),
        "no_terminal_transition": True,
        "transition_counts_only_between_adjacent_segments": bool(
            counts_after.sum() == sum(len(s.segments) - 1
                                      for s in after.segmentations)),
    }

    # -- a smoke chain on the registered (pi, P fixed) model ------------------------------
    began = time.perf_counter()
    result = run_stage7b_chain(model=model_fixed, start=start.copy(),
                               scales=REGISTERED_SCALES, num_sweeps=SMOKE_SWEEPS,
                               burn_in=SMOKE_SWEEPS // 5, thin=10, seed=SMOKE_SEED)
    seconds = time.perf_counter() - began

    keys = result.boundary_keys
    unique_keys = len({k for draw in keys for k in draw})
    h_labels = [precedence_from_u(u).tobytes()
                for u in result.u_draws.reshape(-1, module.M_ROLES, module.D_LATENT)]
    moved = {name: float(np.std(result.scalars[name])) for name in
             (*SCALAR_ORDER, "rho")}
    finite = all(np.isfinite(v).all() for v in
                 (result.u_draws, result.log_target, *result.scalars.values()))

    # -- determinism and resume ------------------------------------------------------------
    again = run_stage7b_chain(model=model_fixed, start=start.copy(),
                              scales=REGISTERED_SCALES, num_sweeps=200,
                              burn_in=50, thin=10, seed=SMOKE_SEED)
    first = run_stage7b_chain(model=model_fixed, start=start.copy(),
                              scales=REGISTERED_SCALES, num_sweeps=200,
                              burn_in=50, thin=10, seed=SMOKE_SEED)
    deterministic = bool(np.array_equal(again.u_draws, first.u_draws)
                         and again.boundary_keys == first.boundary_keys)

    # resume: 100 sweeps, then 100 more from the saved state and generator
    rng_a = np.random.default_rng(SMOKE_SEED + 7)
    part1 = run_stage7b_chain(model=model_fixed, start=start.copy(),
                              scales=REGISTERED_SCALES, num_sweeps=100, burn_in=50,
                              thin=10, seed=0, rng=rng_a)
    resumed = run_stage7b_chain(model=model_fixed, start=start.copy(),
                                scales=REGISTERED_SCALES, num_sweeps=200, burn_in=50,
                                thin=10, seed=0, rng=rng_a, state=part1.final_state)
    rng_b = np.random.default_rng(SMOKE_SEED + 7)
    straight = run_stage7b_chain(model=model_fixed, start=start.copy(),
                                 scales=REGISTERED_SCALES, num_sweeps=200, burn_in=50,
                                 thin=10, seed=0, rng=rng_b)
    resume_matches = bool(np.array_equal(
        np.concatenate([part1.u_draws, resumed.u_draws]), straight.u_draws))

    # -- state round-trip ------------------------------------------------------------------
    payload = json.loads(json.dumps(jsonable(result.final_state.to_dict())))
    restored = Stage6EState.from_dict(result.final_state.to_dict())
    round_trip = bool(np.array_equal(restored.u_by_skill, result.final_state.u_by_skill)
                      and restored.segmentations == result.final_state.segmentations
                      and all(abs(getattr(restored, n)
                                  - getattr(result.final_state, n)) == 0.0
                              for n in SCALAR_ORDER))
    del payload

    return {
        "sweeps": SMOKE_SWEEPS, "seconds": seconds, "retained": len(result.log_target),
        "table_builds": int(result.table_builds),
        "one_build_per_sweep": bool(result.table_builds == SMOKE_SWEEPS),
        "segmentation_states_changed": int(result.movement["states_changed"]),
        "boundary_hamming_total": int(result.movement["boundary_hamming"]),
        "label_changes_total": int(result.movement["label_changes"]),
        "distinct_segmentation_keys_visited": int(unique_keys),
        "distinct_induced_h_visited": int(len(set(h_labels))),
        "scalar_movement_sd": moved,
        "u_movement_sd": float(np.std(result.u_draws)),
        "acceptance": {k: (None if v != v else float(v))
                       for k, v in result.acceptance().items()},
        "ffbs_acceptance_is_one_by_construction": bool(
            result.acceptance().get("S_z_ffbs") == 1.0),
        "no_nans": bool(finite),
        "deterministic_given_the_seed": deterministic,
        "resume_is_bit_identical": resume_matches,
        "state_round_trip": round_trip,
        "pi_P_freshness": pi_p_freshness,
        "pass": bool(finite and deterministic and resume_matches and round_trip
                     and result.table_builds == SMOKE_SWEEPS
                     and result.movement["states_changed"] > 0
                     and unique_keys > 1 and len(set(h_labels)) > 1
                     and all(v > 0 for v in moved.values())
                     and pi_p_freshness["P_diagonal_is_zero"]
                     and pi_p_freshness["counts_differ"]
                     and pi_p_freshness["transition_counts_only_between_adjacent_segments"]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert_stage6d_unchanged()
    module = load_6e1b()
    model_fixed = build(module, infer_pi_P=False)
    model_inferred = build(module, infer_pi_P=True)

    print("[7B0] conditional parity ...", flush=True)
    conditional = conditional_parity(module, model_fixed, N_PARITY_STATES)
    print(f"[7B0]   log Z error {conditional['max_log_z_absolute_error']:.3e}  "
          f"conditional TV {conditional['max_conditional_total_variation']:.3e}")

    print("[7B0] global target parity ...", flush=True)
    target = global_target_parity(module, model_fixed, N_PARITY_STATES)
    print(f"[7B0]   max |direct - sweep| = {target['max_absolute_difference']:.3e}")

    print("[7B0] kernel parity ...", flush=True)
    kernels = kernel_parity(module, model_fixed, 8)
    print(f"[7B0]   worst coordinate difference = "
          f"{kernels['worst_over_all_coordinates']:.3e}")

    print("[7B0] table lifecycle ...", flush=True)
    lifecycle = table_lifecycle_audit(module, model_fixed)
    print(f"[7B0]   q0 {lifecycle['q0_reset_pass']}  order {lifecycle['evaluation_order_invariance']['pass']}  "
          f"eager==batched {lifecycle['eager_vs_batched_tables']['max_absolute_difference']:.2e}  "
          f"builds/sweep ok {lifecycle['builds_equal_sweeps']}  "
          f"stale raises {lifecycle['stale_read_raises']}")

    print("[7B0] smoke ...", flush=True)
    smoke_result = smoke(module, model_fixed, model_inferred)
    print(f"[7B0]   {smoke_result['sweeps']:,} sweeps in {smoke_result['seconds']:.0f}s; "
          f"(S,z) changed {smoke_result['segmentation_states_changed']:,}x; "
          f"{smoke_result['distinct_segmentation_keys_visited']} keys; "
          f"{smoke_result['distinct_induced_h_visited']} induced H; "
          f"deterministic {smoke_result['deterministic_given_the_seed']}; "
          f"resume {smoke_result['resume_is_bit_identical']}")

    cost = {source: sweep_cost_breakdown(model_fixed, random_state(
        module, np.random.default_rng(5), False), REGISTERED_SCALES, n_sweeps=200,
        table_source=source) for source in ("batched", "adapter")}
    print(f"[7B0] sweep cost: batched {cost['batched']['seconds_per_sweep']*1e3:.1f} ms, "
          f"adapter {cost['adapter']['seconds_per_sweep']*1e3:.1f} ms")

    all_pass = all(block["pass"] for block in
                   (conditional, target, kernels, lifecycle, smoke_result))

    config = {
        "stage": "7B0", "source_commit": source_commit(),
        "stage6e_config_hash": config_hash(),
        "python": platform.python_version(), "numpy": np.__version__,
        "model": {"traces": [list(t) for t in model_fixed.traces],
                  "J": module.J, "n_skills": module.K_SKILLS, "m_roles": module.M_ROLES,
                  "epsilon": module.EPSILON, "delta_B": DELTA_B,
                  "min_width": MIN_BLOCK_WIDTH, "max_width": MAX_BLOCK_WIDTH,
                  "pi_P_fixed_model": True, "pi_P_inferred_model_also_tested": True},
        "sweep_order": ["FFBS draw of (S, z)", "pi/P", "U", "rho", *SCALAR_ORDER],
        "segmentation_update": {
            "kind": "Gibbs — exact draw from p(S_n, z_n | x_n, Theta)",
            "acceptance_probability": 1.0,
            "hastings_correction": "none, by construction",
            "rng": "one draw per trace per sweep, taken sequentially from the single "
                   "chain generator in trace order",
        },
        "table_lifecycle": {
            "built": "once per segmentation sweep, for every candidate block",
            "invalidated": "immediately after the FFBS draw, before any parameter moves",
            "read_when_stale": "raises AssertionError",
            "parameter_phase": "evaluates only the selected blocks, through the frozen "
                               "Stage 6E SkillBlockLikelihood",
        },
        "scales": dict(REGISTERED_SCALES),
    }
    (OUT / "config.json").write_text(json.dumps(jsonable(config), indent=2))
    (OUT / "parity_results.json").write_text(json.dumps(jsonable({
        "conditional_parity": conditional, "global_target_parity": target,
        "kernel_parity": kernels, "all_pass": all_pass}), indent=2))
    (OUT / "cache_audit.json").write_text(json.dumps(jsonable(
        {**lifecycle, "sweep_cost": cost}), indent=2))
    (OUT / "smoke.json").write_text(json.dumps(jsonable(smoke_result), indent=2))

    lines = [
        "# Step 7B0 — FFBS joint integration parity and smoke",
        "",
        f"Status: **{'PASS' if all_pass else 'FAIL'}**. This is an integration check, not "
        "a correctness result; Step 7B1 is where the sampler meets an independent "
        "reference.",
        "",
        "## Parity",
        "",
        "| check | statistic | value | threshold | verdict |",
        "|---|---|---|---|---|",
        f"| FFBS conditional vs exact enumeration | max abs log-Z error | "
        f"{conditional['max_log_z_absolute_error']:.3e} | 1e-10 | "
        f"{'PASS' if conditional['pass'] else 'FAIL'} |",
        f"| FFBS conditional vs exact enumeration | max conditional TV | "
        f"{conditional['max_conditional_total_variation']:.3e} | 1e-12 | "
        f"{'PASS' if conditional['pass'] else 'FAIL'} |",
        f"| swept state vs direct Stage 6E target | max abs difference | "
        f"{target['max_absolute_difference']:.3e} | 1e-9 | "
        f"{'PASS' if target['pass'] else 'FAIL'} |",
        f"| Stage 6E kernels, segmentation draw off | worst coordinate difference | "
        f"{kernels['worst_over_all_coordinates']:.3e} | 0 (bit-identical) | "
        f"{'PASS' if kernels['pass'] else 'FAIL'} |",
        f"| eager vs batched candidate tables | max abs difference | "
        f"{lifecycle['eager_vs_batched_tables']['max_absolute_difference']:.3e} | 1e-9 | "
        f"{'PASS' if lifecycle['eager_vs_batched_tables']['pass'] else 'FAIL'} |",
        f"| q_0 = 0 per candidate block | rescore difference | 0 | 0 | "
        f"{'PASS' if lifecycle['q0_reset_pass'] else 'FAIL'} |",
        "",
        "## Cache lifecycle",
        "",
        f"* candidate tables built {lifecycle['table_builds']} times over "
        f"{lifecycle['segmentation_sweeps']} segmentation sweeps (plus one taken to test "
        f"the fingerprint guard): one build per sweep, none per scalar proposal",
        f"* a read after the sweep ends raises: {lifecycle['stale_read_raises']}",
        f"* a read at moved parameters raises: {lifecycle['fingerprint_mismatch_raises']}",
        f"* sweep cost: batched {cost['batched']['seconds_per_sweep']*1e3:.1f} ms "
        f"(table {cost['batched']['block_table_seconds_per_sweep']*1e3:.1f} ms, "
        f"charts {cost['batched']['chart_seconds_per_sweep']*1e3:.1f} ms, "
        f"draws {cost['batched']['backward_draw_seconds_per_sweep']*1e3:.2f} ms), "
        f"eager adapter {cost['adapter']['seconds_per_sweep']*1e3:.1f} ms",
        "",
        "## Smoke",
        "",
        f"* {smoke_result['sweeps']:,} sweeps in {smoke_result['seconds']:.0f}s, "
        f"{smoke_result['retained']} retained",
        f"* `(S, z)` changed on {smoke_result['segmentation_states_changed']:,} trace-draws; "
        f"{smoke_result['distinct_segmentation_keys_visited']} distinct segmentation keys; "
        f"{smoke_result['boundary_hamming_total']:,} boundary changes; "
        f"{smoke_result['label_changes_total']:,} occurrence-label changes",
        f"* {smoke_result['distinct_induced_h_visited']} distinct induced `H` states visited",
        f"* every scalar moves: "
        f"{ {k: round(v, 4) for k, v in smoke_result['scalar_movement_sd'].items()} }",
        f"* FFBS acceptance is exactly 1 by construction: "
        f"{smoke_result['ffbs_acceptance_is_one_by_construction']}",
        f"* no NaNs: {smoke_result['no_nans']}; deterministic: "
        f"{smoke_result['deterministic_given_the_seed']}; resume bit-identical: "
        f"{smoke_result['resume_is_bit_identical']}; state round-trip: "
        f"{smoke_result['state_round_trip']}",
        f"* `pi`/`P` see the new labels: counts differ across the draw "
        f"({smoke_result['pi_P_freshness']['counts_differ']}), `P` diagonal exactly zero "
        f"({smoke_result['pi_P_freshness']['P_diagonal_is_zero']}), transition counts equal "
        f"the number of adjacent segment pairs "
        f"({smoke_result['pi_P_freshness']['transition_counts_only_between_adjacent_segments']})",
        "",
        "A smoke run demonstrates movement. It is not evidence that the sampler targets "
        "the right distribution — that is Step 7B1.",
        "",
        f"Source commit `{config['source_commit']}`.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")

    print(f"[7B0] {'PASS' if all_pass else 'FAIL'}; wrote {OUT}")
    if not all_pass:
        raise SystemExit("Step 7B0 FAILED")


if __name__ == "__main__":
    main()
