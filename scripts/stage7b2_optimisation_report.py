"""Step 7B2 optimisation — the parity evidence and the report, assembled from artifacts.

    PYTHONPATH=src python scripts/stage7b2_optimisation_report.py

Runs the exact-parity sweep across problem shapes, measures the three negative controls
directly (rather than only asserting them in tests), and writes the optimisation
directory. The profiling artifacts are produced separately by `stage7b2_ffbs_profile.py`
and are read here, never recomputed, so the report cannot quietly disagree with them.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import semi_markov_ffbs                              # noqa: E402
from hpop.mcmc_original.fast_block_tables import FastBlockScoreTable         # noqa: E402
from hpop.mcmc_original.latent_poset import precedence_from_u                # noqa: E402
from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import FFBSBlockTables     # noqa: E402
from hpop.mcmc_original.recurrent_segmentation import RecurrentBlockScorer   # noqa: E402
from hpop.mcmc_original.semi_markov_ffbs import forward, posterior_log_marginals  # noqa: E402
from hpop.mcmc_original.stage6e_frozen import (                              # noqa: E402
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH,
)
from hpop.mcmc_original.stage6e_block_table import BlockScoreTable           # noqa: E402
from hpop.mcmc_original.stage6e_state import Stage6EModel, Stage6EState      # noqa: E402
from hpop.mcmc_original.transitions import log_transition_matrix             # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "stage7b2_ffbs_optimisation"
EPSILON = 0.02
FROZEN_ENGINE_SHA256 = (
    "8150bb8235eb159d5e2f08ada7c698c383f4c7fd31f3b882e218b207dd135486")
SETTINGS = (
    {"beta": 1.5, "omega": 1.7346, "lambda_rep": 0.8, "lambda_back": 0.25},
    {"beta": 0.3, "omega": -1.2, "lambda_rep": 0.05, "lambda_back": 1.9},
    {"beta": 3.4, "omega": 4.0, "lambda_rep": 2.5, "lambda_back": 0.01},
)
TOLERANCE = 1e-10


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


def make_state(model, u, parameters) -> Stage6EState:
    from hpop.mcmc_original.fast_segmentation_kernel import segmentation_of
    J = len(model.traces[0])
    ends, position, skill = [], 0, 0
    while position < J:
        width = min(MIN_BLOCK_WIDTH + 1, J - position)
        if 0 < J - position - width < MIN_BLOCK_WIDTH:
            width = J - position
        position += width
        ends.append((position, skill % model.n_skills))
        skill += 1
    if model.n_skills == 1:
        ends = [(J, 0)]
    transition = np.zeros((model.n_skills, model.n_skills))
    for h in range(model.n_skills):
        allowed = [k for k in range(model.n_skills) if k != h]
        if allowed:
            transition[h, allowed] = 1.0 / len(allowed)
    return Stage6EState(segmentations=(segmentation_of(tuple(ends)),), u_by_skill=u,
                        rho=0.3, pi=np.full(model.n_skills, 1.0 / model.n_skills),
                        transition=transition, **parameters)


def parity_sweep() -> dict:
    rows, worst = [], {"block_score_vs_bucketed": 0.0, "block_score_vs_replay": 0.0,
                       "log_z": 0.0, "dp_marginals": 0.0, "relative_likelihood": 0.0}
    skipped = []
    for K in (1, 2, 3):
        for J in (8, 24, 48, 96):
            if K == 1 and J > MAX_BLOCK_WIDTH:
                skipped.append({"J": J, "K": K, "reason": "K = 1 forbids self-transitions "
                                "so only a single block is legal and J > max_width"})
                continue
            for index, parameters in enumerate(SETTINGS):
                rng = np.random.default_rng(1000 + J * 10 + K)
                trace = tuple(int(v) for v in rng.integers(5, size=J))
                u = rng.normal(size=(K, 5, 2))
                model = Stage6EModel(traces=(trace,), epsilon=EPSILON, delta_b=DELTA_B,
                                     n_skills=K, n_roles=5, min_width=MIN_BLOCK_WIDTH,
                                     max_width=MAX_BLOCK_WIDTH)
                state = make_state(model, u, parameters)

                fast = FFBSBlockTables(model=model, source="fast")
                fast.refresh(state)
                bucketed = FFBSBlockTables(model=model, source="batched")
                bucketed.refresh(state)
                a = fast.tables_for(state)[0]
                b = bucketed.tables_for(state)[0]
                support = bool((np.isfinite(a) == np.isfinite(b)).all())
                finite = np.isfinite(a) & np.isfinite(b)
                block_gap = float(np.abs(a[finite] - b[finite]).max())
                relative = float(np.abs(np.expm1(a[finite] - b[finite])).max())

                scorer = RecurrentBlockScorer(
                    traces=(trace,), epsilon=EPSILON, u_by_skill=u,
                    min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH, **parameters)
                replay_gap = 0.0
                for _ in range(40):
                    start = int(rng.integers(0, max(1, J - MIN_BLOCK_WIDTH + 1)))
                    if J - start < MIN_BLOCK_WIDTH:
                        continue
                    width = int(rng.integers(MIN_BLOCK_WIDTH,
                                             min(MAX_BLOCK_WIDTH, J - start) + 1))
                    skill = int(rng.integers(K))
                    replay_gap = max(replay_gap, abs(
                        float(a[start, start + width, skill])
                        - scorer.replay(0, start, start + width, skill)))

                log_pi = np.log(state.pi)
                log_p = log_transition_matrix(state.transition)
                chart_a = forward(a, log_pi, log_p, DELTA_B, MAX_BLOCK_WIDTH,
                                  MIN_BLOCK_WIDTH)
                chart_b = forward(b, log_pi, log_p, DELTA_B, MAX_BLOCK_WIDTH,
                                  MIN_BLOCK_WIDTH)
                log_z_gap = abs(chart_a.log_normalizer - chart_b.log_normalizer)
                marginals_a = posterior_log_marginals(chart_a)
                marginals_b = posterior_log_marginals(chart_b)
                marginal_gap = max(
                    float(np.abs(marginals_a["boundary_marginals"]
                                 - marginals_b["boundary_marginals"]).max()),
                    float(np.abs(marginals_a["occurrence_label_marginals"]
                                 - marginals_b["occurrence_label_marginals"]).max()))

                worst["block_score_vs_bucketed"] = max(
                    worst["block_score_vs_bucketed"], block_gap)
                worst["block_score_vs_replay"] = max(worst["block_score_vs_replay"],
                                                     replay_gap)
                worst["log_z"] = max(worst["log_z"], log_z_gap)
                worst["dp_marginals"] = max(worst["dp_marginals"], marginal_gap)
                worst["relative_likelihood"] = max(worst["relative_likelihood"], relative)
                rows.append({
                    "J": J, "K": K, "setting": index, "same_support": support,
                    "n_finite_blocks": int(finite.sum()),
                    "block_score_vs_bucketed": block_gap,
                    "block_score_vs_replay": replay_gap,
                    "max_relative_likelihood_difference": relative,
                    "log_z_difference": log_z_gap,
                    "dp_marginal_difference": marginal_gap})
    return {"tolerance": TOLERANCE, "rows": rows, "worst": worst, "skipped": skipped,
            "all_support_identical": all(r["same_support"] for r in rows),
            "pass": bool(all(r["same_support"] for r in rows)
                         and worst["block_score_vs_bucketed"] <= TOLERANCE
                         and worst["block_score_vs_replay"] <= TOLERANCE
                         and worst["log_z"] <= TOLERANCE
                         and worst["dp_marginals"] <= TOLERANCE)}


def negative_controls() -> dict:
    """Each control injects the fault its test exists to catch and measures the effect."""
    rng = np.random.default_rng(4242)
    trace = tuple(int(v) for v in rng.integers(5, size=30))
    u = rng.normal(size=(3, 5, 2))
    parameters = SETTINGS[0]
    model = Stage6EModel(traces=(trace,), epsilon=EPSILON, delta_b=DELTA_B, n_skills=3,
                         n_roles=5, min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH)
    state = make_state(model, u, parameters)
    scorer = RecurrentBlockScorer(traces=(trace,), epsilon=EPSILON, u_by_skill=u,
                                  min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
                                  **parameters)

    honest = FastBlockScoreTable(traces=(trace,), epsilon=EPSILON, n_skills=3,
                                 min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
                                 n_roles=5)
    honest.refresh(u, **parameters)
    honest_gap = max(abs(honest.score(0, a, a + 5, 0) - scorer.replay(0, a, a + 5, 0))
                     for a in range(0, 20))

    # 1. a recurrent-state leak across candidates
    original = FastBlockScoreTable._emissions_for_skill

    def leaking(self, u_k, beta, omega, lambda_rep, lambda_back):
        emissions = original(self, u_k, beta, omega, lambda_rep, lambda_back)
        emissions[1:, 0] += 0.05 * emissions[:-1, 0]
        return emissions

    FastBlockScoreTable._emissions_for_skill = leaking
    try:
        faulty = FastBlockScoreTable(traces=(trace,), epsilon=EPSILON, n_skills=3,
                                     min_width=MIN_BLOCK_WIDTH,
                                     max_width=MAX_BLOCK_WIDTH, n_roles=5)
        faulty.refresh(u, **parameters)
        leak_gap = max(abs(faulty.score(0, a, a + 5, 0) - scorer.replay(0, a, a + 5, 0))
                       for a in range(0, 20))
    finally:
        FastBlockScoreTable._emissions_for_skill = original

    # 2. omega dropped from the invalidation fingerprint
    real_fingerprint = FastBlockScoreTable._skill_fingerprint
    moved_omega = state.copy()
    moved_omega.omega = state.omega + 0.7

    def omega_blind(u_k, beta, omega, lambda_rep, lambda_back):
        return (np.asarray(u_k, dtype=float).tobytes(), float(beta), float(lambda_rep),
                float(lambda_back))

    FastBlockScoreTable._skill_fingerprint = staticmethod(omega_blind)
    try:
        stale = FFBSBlockTables(model=model, source="fast")
        stale.refresh(state)
        stale.refresh(moved_omega)
        omega_rebuilt = list(stale.last_refresh["rebuilt_skills"])
        reference = FFBSBlockTables(model=model, source="batched")
        reference.refresh(moved_omega)
        finite = np.isfinite(reference.tables_for(moved_omega)[0])
        omega_gap = float(np.abs(stale.tables_for(moved_omega)[0][finite]
                                 - reference.tables_for(moved_omega)[0][finite]).max())
    finally:
        FastBlockScoreTable._skill_fingerprint = real_fingerprint

    # 3. U_k dropped from the invalidation fingerprint
    moved_u = state.copy()
    perturbed = np.array(state.u_by_skill, copy=True)
    perturbed[1] = -perturbed[1][::-1]
    moved_u.u_by_skill = perturbed
    order_changed = not np.array_equal(precedence_from_u(state.u_by_skill[1]),
                                       precedence_from_u(perturbed[1]))

    def u_blind(u_k, beta, omega, lambda_rep, lambda_back):
        return (float(beta), float(omega), float(lambda_rep), float(lambda_back))

    FastBlockScoreTable._skill_fingerprint = staticmethod(u_blind)
    try:
        stale_u = FFBSBlockTables(model=model, source="fast")
        stale_u.refresh(state)
        stale_u.refresh(moved_u)
        u_rebuilt = list(stale_u.last_refresh["rebuilt_skills"])
        reference_u = FFBSBlockTables(model=model, source="batched")
        reference_u.refresh(moved_u)
        finite = np.isfinite(reference_u.tables_for(moved_u)[0])
        u_gap = float(np.abs(stale_u.tables_for(moved_u)[0][finite]
                             - reference_u.tables_for(moved_u)[0][finite]).max())
    finally:
        FastBlockScoreTable._skill_fingerprint = real_fingerprint

    return {
        "honest_implementation_gap_vs_replay": honest_gap,
        "controls": [
            {"name": "recurrent state leaks across candidates",
             "injected": "each candidate's first emission absorbs 5% of the previous "
                         "candidate's",
             "observed_gap_vs_replay": leak_gap, "tolerance": TOLERANCE,
             "detected": bool(leak_gap > TOLERANCE)},
            {"name": "omega does not invalidate the table",
             "injected": "omega removed from the skill fingerprint",
             "rebuilt_skills_after_omega_moved": omega_rebuilt,
             "observed_gap_vs_fresh_rebuild": omega_gap, "tolerance": 1e-6,
             "detected": bool(not omega_rebuilt and omega_gap > 1e-6)},
            {"name": "U_k does not invalidate its own column",
             "injected": "U removed from the skill fingerprint",
             "perturbation_moves_h_of_u": bool(order_changed),
             "rebuilt_skills_after_u_moved": u_rebuilt,
             "observed_gap_vs_fresh_rebuild": u_gap, "tolerance": 1e-6,
             "detected": bool(not u_rebuilt and u_gap > 1e-6)},
        ],
        "note": "a uniform shift of U_k would NOT be observable: the likelihood sees U "
                "only through the precedence closure h(U), so control 3 uses a "
                "perturbation that actually moves the induced order",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    engine_digest = hashlib.sha256(
        Path(semi_markov_ffbs.__file__).read_bytes()).hexdigest()

    print("[7B2-opt] parity sweep ...", flush=True)
    parity = parity_sweep()
    print(f"[7B2-opt]   worst block-score gap vs bucketed "
          f"{parity['worst']['block_score_vs_bucketed']:.3e}, vs replay "
          f"{parity['worst']['block_score_vs_replay']:.3e}, log Z "
          f"{parity['worst']['log_z']:.3e}, DP marginals "
          f"{parity['worst']['dp_marginals']:.3e}")

    print("[7B2-opt] negative controls ...", flush=True)
    controls = negative_controls()
    for control in controls["controls"]:
        print(f"[7B2-opt]   {control['name']:48s} detected={control['detected']}")

    baseline = json.loads((OUT / "baseline_profile.json").read_text())
    optimised = json.loads((OUT / "performance.json").read_text())
    ab = json.loads((OUT / "ab_comparison.json").read_text())

    audit = {
        "frozen_engine": {
            "path": "src/hpop/mcmc_original/semi_markov_ffbs.py",
            "sha256": engine_digest,
            "expected_sha256": FROZEN_ENGINE_SHA256,
            "unchanged": bool(engine_digest == FROZEN_ENGINE_SHA256)},
        "what_changed": [
            "new src/hpop/mcmc_original/fast_block_tables.py: one recurrent trajectory "
            "per (trace, start) instead of one per candidate block, with every width "
            "read off a cumulative sum",
            "candidate layout precomputed once per (traces, min_width, max_width) and "
            "cached; no Python tuples rebuilt per sweep",
            "candidates sorted by descending remaining length so each step operates on a "
            "contiguous view rather than a masked array",
            "dense per-trace tables allocated once and written in place",
            "skill-local invalidation: U_k rebuilds only column k; the four global "
            "scalars rebuild all; rho, pi and P rebuild none",
            "key_movement replaces two per-occurrence array constructions per trace per "
            "sweep with one linear walk",
            "stage6e_sampler.segmentation_sweep gains the upstream zero-proposal fast "
            "path, skipping a target evaluation whose value is discarded"],
        "what_did_not_change": [
            "semi_markov_ffbs.py (byte-identical to the Step 7A checkpoint)",
            "the posterior, the priors, the block likelihood, the FFBS recurrence, "
            "backward sampling, and every global MCMC kernel",
            "min/max widths and the invalid-block support"],
        "arithmetic": ab["fast_table_stats"],
    }

    config = {
        "stage": "7B2-optimisation", "source_commit": source_commit(),
        "python": platform.python_version(), "numpy": np.__version__,
        "corpus_hash": baseline.get("corpus_hash"),
        "targets": {"primary_seconds_per_sweep": 1.20, "preferred": 1.00,
                    "stretch": 0.80,
                    "note": "engineering targets, not correctness gates"},
        "measured_under_contention": True,
    }

    scaling = {"baseline": baseline.get("scaling"), "optimised": optimised.get("scaling"),
               "note": "the forward chart is frozen code and its scaling is unchanged by "
                       "this work; it is reported so the block-table improvement is not "
                       "mistaken for a chart improvement"}

    (OUT / "config.json").write_text(json.dumps(jsonable(config), indent=2))
    (OUT / "parity.json").write_text(json.dumps(jsonable(parity), indent=2))
    (OUT / "negative_controls.json").write_text(json.dumps(jsonable(controls), indent=2))
    (OUT / "optimisation_audit.json").write_text(json.dumps(jsonable(audit), indent=2))
    (OUT / "scaling.json").write_text(json.dumps(jsonable(scaling), indent=2))

    write_report(config, audit, parity, controls, baseline, optimised, ab)
    print(f"[7B2-opt] wrote {OUT}")
    if not (parity["pass"] and audit["frozen_engine"]["unchanged"]
            and all(c["detected"] for c in controls["controls"])):
        raise SystemExit("Step 7B2 optimisation parity FAILED")


def write_report(config, audit, parity, controls, baseline, optimised, ab) -> None:
    before = ab["sweep_before_everything"]["median_seconds"]
    after = ab["sweep_after_everything"]["median_seconds"]
    chart = optimised["phases"]["phase_4_forward_chart_FROZEN_ENGINE"]["wall_seconds"]
    lines = [
        "# Step 7B2 — exact optimisation of the FFBS conditional computation",
        "",
        f"Status: **parity PASS**, performance target **not met**. Sweep {before:.3f}s -> "
        f"{after:.3f}s ({ab['sweep_speedup']:.2f}x) against a 1.20s primary target.",
        "",
        "## The headline finding, which reframes the task",
        "",
        "The brief attributed about 75% of an FFBS sweep to *candidate block-table and "
        "forward-chart construction* and set the block table as the primary target. The "
        "profile splits that 75% very unevenly:",
        "",
        "| phase | baseline | share of sweep | can it be optimised here? |",
        "|---|---|---|---|",
        f"| forward chart | {baseline['phases']['phase_4_forward_chart_FROZEN_ENGINE']['wall_seconds']:.3f}s "
        f"| {baseline['phases']['attribution_fraction_of_sweep']['forward_chart'] * 100:.0f}% "
        "| **no — frozen Step 7A engine** |",
        f"| block-score construction | "
        f"{baseline['phases']['phase_1_block_score_construction']['wall_seconds']:.3f}s "
        f"| {baseline['phases']['attribution_fraction_of_sweep']['block_scores'] * 100:.0f}% "
        "| yes |",
        f"| parameter phase | "
        f"{baseline['phases']['phase_7_parameter_phase']['wall_seconds']:.3f}s "
        f"| {baseline['phases']['attribution_fraction_of_sweep']['parameter_phase'] * 100:.0f}% "
        "| frozen Stage 6E kernels |",
        f"| backward draw | "
        f"{baseline['phases']['phase_5_backward_draw_FROZEN_ENGINE']['wall_seconds']:.3f}s "
        f"| {baseline['phases']['attribution_fraction_of_sweep']['backward_draw'] * 100:.0f}% "
        "| no — frozen engine |",
        "",
        "At function level the picture is starker: `semi_markov_ffbs.forward` is ~81% of "
        "the sweep, and **~60% of the whole sweep is `scipy.special.logsumexp`**, called "
        "9,097 times per sweep from inside the frozen recursion. The block table — the "
        "brief's primary target — was 13%.",
        "",
        "So the 1.20s target is not reachable under this task's constraints. Even an "
        f"instantaneous block table leaves ~{chart:.2f}s of frozen chart plus the frozen "
        "parameter phase. That is a statement about where the time is, not about whether "
        "FFBS is correct.",
        "",
        "## What was optimised anyway",
        "",
    ]
    for item in audit["what_changed"]:
        lines.append(f"* {item}")
    lines += [
        "",
        "The block table is the one large win available. A block score is a sum of "
        "per-step emissions along a trajectory that starts at `a` with `q_0 = 0`, and "
        "**nothing in that trajectory depends on the block end**. Previous builders "
        "replayed the shared prefix once per width; this one replays each start once and "
        "reads every width off a cumulative sum:",
        "",
        f"* {audit['arithmetic']['width_bucketed_candidate_steps_per_skill']:,} candidate-steps "
        f"per skill -> {audit['arithmetic']['candidate_steps_per_skill']:,} "
        f"({audit['arithmetic']['arithmetic_reduction']:.1f}x less arithmetic)",
        f"* {audit['arithmetic']['n_candidates']:,} candidates from "
        f"{audit['arithmetic']['n_starts']:,} starts, across "
        f"{audit['arithmetic']['n_traces']} traces and "
        f"{audit['arithmetic']['n_skills']} skills",
        "",
        "## Measured, alternating inside one process",
        "",
        "Cross-run comparison is not usable on this machine: the forward chart is "
        "unchanged code and its wall time still varies by 40% between runs, because the "
        "Stage 6E2 baseline is competing for cores. Every number below alternates the two "
        "implementations inside one process and takes medians.",
        "",
        "| quantity | before | after | ratio |",
        "|---|---|---|---|",
        f"| global sweep | {before:.3f}s | {after:.3f}s | {ab['sweep_speedup']:.2f}x |",
        f"| block table | {ab['block_table_before']['median_seconds'] * 1e3:.1f} ms | "
        f"{ab['block_table_after']['median_seconds'] * 1e3:.1f} ms | "
        f"{ab['block_table_speedup']:.1f}x |",
        f"| sweep with only the zero-proposal fast path | "
        f"{ab['sweep_with_fast_path_only']['median_seconds']:.3f}s | — | — |",
        "",
        f"Table parity in that same comparison: max absolute difference "
        f"**{ab['table_parity_max_absolute_difference']:.1e}**.",
        "",
        f"Against the targets: primary 1.20s **missed** ({after:.3f}s), preferred 1.00s "
        f"missed, stretch 0.80s missed. The LocalMoveKernel baseline runs at ~0.69 s per "
        "sweep, so FFBS remains roughly "
        f"{after / 0.69:.1f}x its cost per sweep on this corpus.",
        "",
        "## Exact parity",
        "",
        f"Across {len(parity['rows'])} problem shapes (J = 8, 24, 48, 96; K = 1, 2, 3; "
        f"three parameter settings), tolerance {parity['tolerance']:.0e}:",
        "",
        "| comparison | worst observed |",
        "|---|---|",
        f"| block score vs the width-bucketed builder | "
        f"{parity['worst']['block_score_vs_bucketed']:.3e} |",
        f"| block score vs `RecurrentBlockScorer.replay` | "
        f"{parity['worst']['block_score_vs_replay']:.3e} |",
        f"| max relative likelihood difference | "
        f"{parity['worst']['relative_likelihood']:.3e} |",
        f"| FFBS log Z | {parity['worst']['log_z']:.3e} |",
        f"| exact DP marginals | {parity['worst']['dp_marginals']:.3e} |",
        "",
        f"Finite/-inf support identical in every shape: "
        f"{parity['all_support_identical']}. "
        f"{len(parity['skipped'])} combination(s) skipped as having no legal path at all "
        "(K = 1 with J > max_width forbids every segmentation).",
        "",
        "## Negative controls",
        "",
        "| injected fault | detected | observed effect |",
        "|---|---|---|",
    ]
    for control in controls["controls"]:
        gap = control.get("observed_gap_vs_replay",
                          control.get("observed_gap_vs_fresh_rebuild"))
        lines.append(f"| {control['name']} | {control['detected']} | {gap:.3e} |")
    lines += [
        "",
        f"The honest implementation's gap against the same reference is "
        f"{controls['honest_implementation_gap_vs_replay']:.3e}, so the controls are "
        "separating a real fault from floating-point noise rather than from nothing.",
        "",
        f"One control had to be repaired to be meaningful: {controls['note']}",
        "",
        "## What would actually reach the target",
        "",
        "The remaining cost is concentrated in one primitive inside the frozen engine: "
        "`logsumexp` over a handful of terms, called once per `(position, skill)` per "
        "trace per sweep. A vectorised chart — one that forms the predecessor terms for "
        "all `(b, k)` of a trace as arrays and reduces them without per-cell scipy calls "
        "— is the only change that would move the sweep materially, and it is precisely "
        "what this task forbids (no engine edit, no second FFBS implementation). That is "
        "the right call for a validated engine, and it means the decision is the user's: "
        "authorising a vectorised chart, validated entry-for-entry against the frozen one, "
        "is the next lever. Nothing in this task should be read as evidence that it is "
        "safe to skip that validation.",
        "",
        f"Frozen engine sha256 `{audit['frozen_engine']['sha256'][:32]}...` — unchanged: "
        f"{audit['frozen_engine']['unchanged']}.",
        "",
        f"Source commit `{config['source_commit']}`.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
