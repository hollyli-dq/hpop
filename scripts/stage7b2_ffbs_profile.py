"""Step 7B2 optimisation — where an FFBS global sweep actually spends its time.

    PYTHONPATH=src python scripts/stage7b2_ffbs_profile.py --label baseline
    PYTHONPATH=src python scripts/stage7b2_ffbs_profile.py --label optimised

Profiling comes before optimising, and it is kept as an artifact so the "before" is not
reconstructed from memory afterwards. The decomposition below is deliberately finer than
"table vs chart": the two are built by different code with different scaling, and only one
of them is inside the frozen Step 7A engine.

Phases measured, on the frozen Stage 6E2 corpus at the registered Step 7B2 starting state:

    1  recurrent candidate block-score construction   (the batched builder)
    2  candidate index/scatter into the dense table    (Python-level layout work)
    3  allocation and zeroing of log_block_scores
    4  FFBS forward chart                              (FROZEN ENGINE)
    5  FFBS backward draw                              (FROZEN ENGINE)
    6  pi/P conjugate update
    7  U / rho / scalar parameter phase
    8  one complete global sweep

Phase 7 is measured as the whole Stage 6E parameter phase, because that is exactly the
object Step 7B reuses: `ffbs_sweep_once` calls `stage6e_sampler.sweep_once` with the local
segmentation phase switched off. Phase 6 is timed separately by calling the frozen Stage 3
conjugate updates on the same state, so it can be attributed inside phase 7 rather than
guessed at.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import math
import platform
import pstats
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import (                    # noqa: E402
    FFBSBlockTables, Stage7BSampler, ffbs_sweep_once,
)
from hpop.mcmc_original.recurrent_segmentation import RecurrentBlockScorer     # noqa: E402
from hpop.mcmc_original.semi_markov_ffbs import backward_sample, forward       # noqa: E402
from hpop.mcmc_original.stage6e_corpus import corpus_hash, generate_corpus     # noqa: E402
from hpop.mcmc_original.stage6e_frozen import (                               # noqa: E402
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH, assert_stage6d_unchanged,
)
from hpop.mcmc_original.stage6e_sampler import Stage6ESampler, sweep_once      # noqa: E402
from hpop.mcmc_original.stage6e_state import (                                # noqa: E402
    Stage6EModel, initial_counts, transition_counts_of,
)
from hpop.mcmc_original.transitions import (                                   # noqa: E402
    log_transition_matrix, sample_transition_matrix,
)

OUT = ROOT / "results" / "mcmc_original" / "stage7b2_ffbs_optimisation"
SYNTHETIC_LENGTHS = (8, 24, 48, 96)
SYNTHETIC_SEED = 7_064_000


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


def load_stage7b2():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stage7b2", ROOT / "scripts" / "stage7b2_full_joint_ffbs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def timed(function, repeats: int = 1):
    """Wall and CPU seconds per call, as a median over `repeats`."""
    walls, cpus = [], []
    for _ in range(repeats):
        wall0, cpu0 = time.perf_counter(), time.process_time()
        result = function()
        walls.append(time.perf_counter() - wall0)
        cpus.append(time.process_time() - cpu0)
    return {"wall_seconds": float(np.median(walls)),
            "cpu_seconds": float(np.median(cpus)),
            "repeats": int(repeats)}, result


# --------------------------------------------------------------------------- the corpus
def corpus_setup(table_source: str):
    module = load_stage7b2()
    baseline = module.load_baseline_script()
    corpus = generate_corpus()
    model = baseline.build_model(corpus)
    state = baseline.dispersed_start(0, corpus, model, oracle=False)
    scales = module.kernel_audit()["proposal_scales"]
    sampler = Stage7BSampler(model=model, scales=scales, table_source=table_source)
    return {"corpus": corpus, "model": model, "state": state, "scales": scales,
            "sampler": sampler, "corpus_hash": corpus_hash(corpus)}


def phase_profile(setup: dict, table_source: str, sweeps: int = 3) -> dict:
    """The eight-phase decomposition on the real corpus."""
    model, state, scales = setup["model"], setup["state"], setup["scales"]
    sampler = setup["sampler"]
    rng = np.random.default_rng(SYNTHETIC_SEED)

    tables = FFBSBlockTables(model=model, source=table_source)
    allocation, _ = timed(lambda: FFBSBlockTables(model=model, source=table_source),
                          repeats=3)

    # Every repeat must move a parameter, or the skill-local cache would legitimately
    # return an unchanged table in ~0 ms and the artifact would report a speedup that a
    # real sweep never sees: in a real sweep the scalars move almost every time.
    perturbed = []
    for i in range(3):
        variant = state.copy()
        variant.beta = float(state.beta) + 1e-3 * (i + 1)
        perturbed.append(variant)
    counter = {"i": 0}

    def rebuild():
        variant = perturbed[counter["i"] % len(perturbed)]
        counter["i"] += 1
        return tables.refresh(variant)

    build, _ = timed(rebuild, repeats=3)
    tables.refresh(state)

    dense = tables.tables_for(state)
    n_blocks = int(sum(int(np.isfinite(t).sum()) for t in dense))
    log_pi = np.log(state.pi)
    log_p = log_transition_matrix(state.transition)

    chart_timing, charts = timed(
        lambda: [forward(t, log_pi, log_p, model.delta_b, model.max_width,
                         model.min_width) for t in dense], repeats=3)
    draw_timing, _ = timed(
        lambda: [backward_sample(chart, rng) for chart in charts], repeats=3)

    counts = transition_counts_of(state.segmentations, model.n_skills)
    pi_p_timing, _ = timed(
        lambda: (sample_transition_matrix(counts, model.n_skills, rng,
                                          model.eta_transition),
                 rng.dirichlet(model.eta_initial
                               + initial_counts(state.segmentations, model.n_skills))),
        repeats=20)

    # the Stage 6E parameter phase exactly as Step 7B calls it
    parameter_sampler = Stage6ESampler(model=model, scales=dict(scales),
                                       n_proposals_per_trace=0)
    parameter_timing, _ = timed(
        lambda: sweep_once(state, parameter_sampler, np.random.default_rng(1)),
        repeats=3)

    # the no-op segmentation phase inside that call: with zero proposals the Stage 6E
    # sweep still evaluates every trace's target once, which costs uncached block replays
    scorer = model.scorer_for(state)
    before = int(scorer.full_replay_calls)
    noop_timing, _ = timed(
        lambda: [target(key) for target, key in zip(
            parameter_sampler._targets or [], [])], repeats=1)
    del noop_timing
    parameter_sampler.prepare(state)
    from hpop.mcmc_original.fast_segmentation_kernel import key_of
    keys = [key_of(s) for s in state.segmentations]
    for target in parameter_sampler._targets:
        target.set_path_prior(log_pi, log_p)
    zero_proposal_timing, _ = timed(
        lambda: [target(key) for target, key in zip(parameter_sampler._targets, keys)],
        repeats=3)
    replays_for_zero_proposals = int(scorer.full_replay_calls) - before

    sweep_timing, _ = timed(
        lambda: ffbs_sweep_once(state, sampler, np.random.default_rng(2)),
        repeats=sweeps)

    return {
        "table_source": table_source,
        "n_traces": len(model.traces),
        "trace_length": {"mean": float(np.mean([len(t) for t in model.traces])),
                         "min": int(min(len(t) for t in model.traces)),
                         "max": int(max(len(t) for t in model.traces))},
        "n_candidate_blocks": n_blocks,
        "phase_1_block_score_construction": build,
        "phase_2_and_3_allocation_and_layout": allocation,
        "phase_4_forward_chart_FROZEN_ENGINE": chart_timing,
        "phase_5_backward_draw_FROZEN_ENGINE": draw_timing,
        "phase_6_pi_P_update": pi_p_timing,
        "phase_7_parameter_phase": parameter_timing,
        "phase_7a_zero_proposal_segmentation_target_evaluations": {
            **zero_proposal_timing,
            "uncached_block_replays": replays_for_zero_proposals,
            "note": "the Stage 6E segmentation phase evaluates each trace's target once "
                    "even at zero proposals; Step 7B does not need that value"},
        "phase_8_total_sweep": sweep_timing,
        "attribution_fraction_of_sweep": {
            "block_scores": build["wall_seconds"] / sweep_timing["wall_seconds"],
            "forward_chart": chart_timing["wall_seconds"] / sweep_timing["wall_seconds"],
            "backward_draw": draw_timing["wall_seconds"] / sweep_timing["wall_seconds"],
            "parameter_phase": (parameter_timing["wall_seconds"]
                                / sweep_timing["wall_seconds"]),
        },
        "scorer_calls": {"full_replays": int(scorer.full_replay_calls),
                         "cached": int(scorer.cached_calls)},
        "peak_rss_bytes": peak_rss_bytes(),
    }


def synthetic_scaling(table_source: str, lengths=SYNTHETIC_LENGTHS,
                      n_skills=(1, 2, 3)) -> list:
    """Block table, chart and draw cost against `J` and `K`, on synthetic traces."""
    rng = np.random.default_rng(SYNTHETIC_SEED)
    u_full = rng.normal(size=(3, 5, 2))
    rows = []
    skipped = []
    for K in n_skills:
        for J in lengths:
            if K == 1 and J > MAX_BLOCK_WIDTH:
                # with one skill and a zero transition diagonal the only legal
                # segmentation is a single block, which cannot cover J > max_width. The
                # engine is right to refuse; the combination has no legal path at all.
                skipped.append({"J": J, "K": K,
                                "reason": "no legal path: K = 1 forbids self-transitions "
                                          "so only a single block is legal, and "
                                          f"J > max_width = {MAX_BLOCK_WIDTH}"})
                continue
            trace = tuple(int(v) for v in rng.integers(5, size=J))
            model = Stage6EModel(traces=(trace,), epsilon=0.02, delta_b=DELTA_B,
                                 n_skills=K, n_roles=5, min_width=MIN_BLOCK_WIDTH,
                                 max_width=MAX_BLOCK_WIDTH, infer_pi_P=True)
            state = _synthetic_state(model, u_full[:K], rng)
            tables = FFBSBlockTables(model=model, source=table_source)
            variants = []
            for i in range(5):
                variant = state.copy()
                variant.beta = float(state.beta) + 1e-3 * (i + 1)
                variants.append(variant)
            counter = {"i": 0}

            def rebuild(_tables=tables, _variants=variants, _counter=counter):
                variant = _variants[_counter["i"] % len(_variants)]
                _counter["i"] += 1
                return _tables.refresh(variant)

            build, _ = timed(rebuild, repeats=5)
            tables.refresh(state)
            dense = tables.tables_for(state)
            log_pi = np.log(state.pi)
            log_p = log_transition_matrix(state.transition)
            chart, charts = timed(
                lambda: forward(dense[0], log_pi, log_p, model.delta_b, model.max_width,
                                model.min_width), repeats=5)
            draw, _ = timed(lambda: backward_sample(charts, rng), repeats=20)
            rows.append({
                "J": J, "K": K,
                "n_candidate_blocks": int(np.isfinite(dense[0]).sum()),
                "block_table": build, "forward_chart": chart, "backward_draw": draw,
                "log_normalizer": float(charts.log_normalizer)})
    if skipped:
        rows.append({"skipped_combinations": skipped})
    return rows


def _synthetic_state(model, u, rng):
    from hpop.mcmc_original.fast_segmentation_kernel import segmentation_of
    from hpop.mcmc_original.stage6e_state import Stage6EState
    J = len(model.traces[0])
    ends, running, skill = [], 0, 0
    while running < J:
        width = min(MIN_BLOCK_WIDTH + int(rng.integers(0, 4)), J - running)
        if J - running - width < MIN_BLOCK_WIDTH and J - running - width > 0:
            width = J - running
        running += width
        ends.append((running, skill % model.n_skills))
        skill += 1
    transition = np.zeros((model.n_skills, model.n_skills))
    for h in range(model.n_skills):
        allowed = [k for k in range(model.n_skills) if k != h]
        if allowed:
            transition[h, allowed] = 1.0 / len(allowed)
    if model.n_skills == 1:
        ends = [(J, 0)]
    return Stage6EState(
        segmentations=(segmentation_of(tuple(ends)),), u_by_skill=u,
        rho=0.3, beta=1.5, omega=1.7346, lambda_rep=0.8, lambda_back=0.25,
        pi=np.full(model.n_skills, 1.0 / model.n_skills), transition=transition)


def hot_functions(setup: dict, table_source: str, n_sweeps: int = 2) -> dict:
    """cProfile over complete sweeps: which functions, and how many calls."""
    sampler = Stage7BSampler(model=setup["model"], scales=setup["scales"],
                             table_source=table_source)
    state = setup["state"]
    rng = np.random.default_rng(3)
    profiler = cProfile.Profile()
    profiler.enable()
    current = state
    for _ in range(n_sweeps):
        current = ffbs_sweep_once(current, sampler, rng)
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(18)
    rows = []
    for (path, line, name), (calls, _, total, cumulative, _) in stats.stats.items():
        rows.append({"function": f"{Path(path).name}:{line}({name})",
                     "calls_per_sweep": calls / n_sweeps,
                     "total_seconds": total, "cumulative_seconds": cumulative})
    rows.sort(key=lambda row: -row["cumulative_seconds"])
    return {"n_sweeps": n_sweeps, "top": rows[:18], "text": stream.getvalue()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--table-source", default=None,
                        help="default: 'batched' for the baseline label, 'fast' otherwise")
    parser.add_argument("--sweeps", type=int, default=3)
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--ab", action="store_true",
                        help="alternate before/after in one process and stop")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    assert_stage6d_unchanged()
    table_source = args.table_source or ("batched" if args.label == "baseline" else "fast")

    setup = corpus_setup(table_source)
    if args.ab:
        comparison = ab_comparison(setup)
        (OUT / "ab_comparison.json").write_text(json.dumps(jsonable(comparison), indent=2))
        print(f"[A/B] sweep before {comparison['sweep_before_everything']['median_seconds']:.3f}s"
              f"  -> after {comparison['sweep_after_everything']['median_seconds']:.3f}s"
              f"  ({comparison['sweep_speedup']:.2f}x)")
        print(f"[A/B] block table {comparison['block_table_before']['median_seconds']*1e3:.1f} ms"
              f" -> {comparison['block_table_after']['median_seconds']*1e3:.1f} ms"
              f"  ({comparison['block_table_speedup']:.1f}x), parity "
              f"{comparison['table_parity_max_absolute_difference']:.1e}")
        print(f"[A/B] wrote {OUT / 'ab_comparison.json'}")
        return
    print(f"[profile:{args.label}] corpus {setup['corpus_hash'][:16]}  "
          f"{len(setup['model'].traces)} traces  source={table_source}", flush=True)

    phases = phase_profile(setup, table_source, sweeps=args.sweeps)
    print(f"[profile:{args.label}] sweep {phases['phase_8_total_sweep']['wall_seconds']:.3f}s"
          f"  tables {phases['phase_1_block_score_construction']['wall_seconds']:.3f}s"
          f"  charts {phases['phase_4_forward_chart_FROZEN_ENGINE']['wall_seconds']:.3f}s"
          f"  draws {phases['phase_5_backward_draw_FROZEN_ENGINE']['wall_seconds']:.3f}s"
          f"  parameters {phases['phase_7_parameter_phase']['wall_seconds']:.3f}s",
          flush=True)
    print(f"[profile:{args.label}] zero-proposal target evaluations "
          f"{phases['phase_7a_zero_proposal_segmentation_target_evaluations']['wall_seconds']:.3f}s "
          f"({phases['phase_7a_zero_proposal_segmentation_target_evaluations']['uncached_block_replays']} "
          f"uncached replays)", flush=True)

    scaling = synthetic_scaling(table_source)
    for row in scaling:
        if row.get("K") == 3:
            print(f"[profile:{args.label}] J={row['J']:>3} K={row['K']}  "
                  f"blocks {row['n_candidate_blocks']:>5}  "
                  f"table {row['block_table']['wall_seconds'] * 1e3:7.2f} ms  "
                  f"chart {row['forward_chart']['wall_seconds'] * 1e3:7.2f} ms  "
                  f"draw {row['backward_draw']['wall_seconds'] * 1e6:7.0f} us", flush=True)

    payload = {
        "label": args.label, "table_source": table_source,
        "source_commit": source_commit(),
        "python": platform.python_version(), "numpy": np.__version__,
        "corpus_hash": setup["corpus_hash"],
        "measured_under_contention": True,
        "contention_note": "the Stage 6E2 LocalMoveKernel baseline was running on the "
                           "same machine; every wall clock here is pessimistic and the "
                           "before/after comparison is made under the same conditions",
        "phases": phases, "scaling": scaling,
    }
    if not args.skip_profile:
        payload["hot_functions"] = hot_functions(setup, table_source)
        print(f"[profile:{args.label}] top by cumulative time:")
        for row in payload["hot_functions"]["top"][:8]:
            print(f"    {row['cumulative_seconds']:8.3f}s  "
                  f"{row['calls_per_sweep']:9.0f} calls/sweep  {row['function']}")

    name = "baseline_profile.json" if args.label == "baseline" else "performance.json"
    (OUT / name).write_text(json.dumps(jsonable(payload), indent=2))
    print(f"[profile:{args.label}] wrote {OUT / name}")




# --------------------------------------------------------------------- the honest A/B
def ab_comparison(setup: dict, repeats: int = 6) -> dict:
    """Before and after, alternating inside one process.

    Cross-run comparison is not usable here: the forward chart is unchanged code and its
    wall time still varies by 40% between runs, because the Stage 6E2 baseline is
    competing for cores. Alternating the two implementations inside one process and taking
    medians is the only way to attribute a difference to the code rather than to the load.

    The `stage6e_sampler` zero-proposal fast path is a source change, so "before" for that
    one is reconstructed by a measurement-only monkeypatch that restores the discarded
    target evaluation. That patch is never shipped; it exists so the saving can be
    attributed rather than asserted.
    """
    import hpop.mcmc_original.stage6e_sampler as stage6e

    model, state, scales = setup["model"], setup["state"], setup["scales"]
    original_sweep = stage6e.segmentation_sweep

    def without_fast_path(keys, targets, kernels, n_proposals, rng, proposed, accepted,
                          invalid):
        keys = list(keys)
        movement = {"boundary_hamming": 0, "label_changes": 0}
        for target, key in zip(targets, keys):          # the discarded evaluation
            target(key)
        return tuple(keys), movement

    samplers = {name: Stage7BSampler(model=model, scales=scales, table_source=source)
                for name, source in (("before", "batched"), ("after", "fast"))}
    states = {name: state for name in samplers}
    rngs = {name: np.random.default_rng(11) for name in samplers}
    for name, sampler in samplers.items():              # warm
        states[name] = ffbs_sweep_once(states[name], sampler, rngs[name])

    sweep_times = {"before": [], "after": [], "before_without_fast_path": []}
    for _ in range(repeats):
        for name in ("before", "after"):
            began = time.perf_counter()
            states[name] = ffbs_sweep_once(states[name], samplers[name], rngs[name])
            sweep_times[name].append(time.perf_counter() - began)
        stage6e.segmentation_sweep = without_fast_path
        try:
            began = time.perf_counter()
            states["before"] = ffbs_sweep_once(states["before"], samplers["before"],
                                               rngs["before"])
            sweep_times["before_without_fast_path"].append(time.perf_counter() - began)
        finally:
            stage6e.segmentation_sweep = original_sweep

    build_times = {"before": [], "after": []}
    tables = {"before": FFBSBlockTables(model=model, source="batched"),
              "after": FFBSBlockTables(model=model, source="fast")}
    for index in range(repeats):
        variant = state.copy()
        variant.beta = float(state.beta) + 1e-3 * (index + 1)
        for name, table in tables.items():
            began = time.perf_counter()
            table.refresh(variant)
            build_times[name].append(time.perf_counter() - began)

    worst = 0.0
    for a, b in zip(tables["before"].tables_for(variant),
                    tables["after"].tables_for(variant)):
        finite = np.isfinite(a) & np.isfinite(b)
        worst = max(worst, float(np.abs(a[finite] - b[finite]).max()))

    def summary(values):
        return {"median_seconds": float(np.median(values)),
                "min_seconds": float(np.min(values)), "n": len(values)}

    before = float(np.median(sweep_times["before_without_fast_path"]))
    after = float(np.median(sweep_times["after"]))
    return {
        "protocol": "alternating inside one process; medians over "
                    f"{repeats} interleaved repeats",
        "sweep_before_everything": summary(sweep_times["before_without_fast_path"]),
        "sweep_with_fast_path_only": summary(sweep_times["before"]),
        "sweep_after_everything": summary(sweep_times["after"]),
        "block_table_before": summary(build_times["before"]),
        "block_table_after": summary(build_times["after"]),
        "block_table_speedup": (float(np.median(build_times["before"]))
                                / max(1e-12, float(np.median(build_times["after"])))),
        "sweep_speedup": before / max(1e-12, after),
        "table_parity_max_absolute_difference": worst,
        "fast_table_stats": tables["after"]._fast.stats(),
    }

if __name__ == "__main__":
    main()
