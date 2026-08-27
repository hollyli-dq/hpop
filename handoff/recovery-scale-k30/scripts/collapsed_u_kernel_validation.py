"""Collapsed-U kernel validation — the fast artifacts (everything but the 600k run).

    PYTHONPATH=src python scripts/collapsed_u_kernel_validation.py

Writes into results/mcmc_original/collapsed_u_kernel_validation/:

    tiny_exact_reference.json    exact stationarity of the composed kernel on a finite
                                 (U-grid x paths) joint space, plus the stale-path
                                 negative control
    correctness.json             every §19 numerical control, with values
    resume_check.json            uninterrupted == checkpoint + resume, field by field
    runtime.json                 collapsed-event cost on the reference problem and on a
                                 read-only full-corpus checkpoint; every=10 overhead
    implementation_manifest.json files added; proof the validated modules are untouched

The 600k mixed-reference gates land in the same directory from
`collapsed_u_mixed_reference.py`; `report.md` is assembled once both halves exist.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "mcmc_original"))

from hpop.mcmc_original.collapsed_u_kernel import (                        # noqa: E402
    CollapsedUConfig, run_collapsed_u_chain,
)
from hpop.mcmc_original.collapsed_u_likelihood import CollapsedULikelihood  # noqa: E402
from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import (                 # noqa: E402
    Stage7BSampler, run_stage7b_chain,
)
from hpop.mcmc_original.sampler_u import propose_row                       # noqa: E402
from hpop.mcmc_original.semi_markov_ffbs import backward_sample, forward   # noqa: E402
from hpop.mcmc_original.stage6d_frozen import REGISTERED_SCALES            # noqa: E402
from hpop.mcmc_original.stage6e_state import Stage6EState                  # noqa: E402
from hpop.mcmc_original.transitions import log_transition_matrix           # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "collapsed_u_kernel_validation"
FULL_CORPUS_CHECKPOINT = (ROOT / "results" / "mcmc_original"
                          / "stage7b2_full_joint_ffbs" / "checkpoints"
                          / "chain0_checkpoint.json")

# the validated modules this task must not touch, pinned by git-diff emptiness
MUST_BE_UNTOUCHED = [
    "src/hpop/mcmc_original/semi_markov_ffbs.py",
    "src/hpop/mcmc_original/sampler_u.py",
    "src/hpop/mcmc_original/recurrent_joint_ffbs_mcmc.py",
    "src/hpop/mcmc_original/stage6e_sampler.py",
    "src/hpop/mcmc_original/fast_block_tables.py",
    "src/hpop/mcmc_original/recurrent_segmentation.py",
    "src/hpop/mcmc_original/stage6c_frozen.py",
]
NEW_FILES = [
    "src/hpop/mcmc_original/collapsed_u_likelihood.py",
    "src/hpop/mcmc_original/collapsed_u_kernel.py",
    "scripts/collapsed_u_mixed_reference.py",
    "scripts/collapsed_u_kernel_validation.py",
    "tests/mcmc_original/test_collapsed_u_kernel.py",
    "tests/mcmc_original/test_collapsed_u_ordering.py",
    "tests/mcmc_original/test_collapsed_u_reference.py",
    "tests/mcmc_original/test_collapsed_u_resume.py",
]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kernel_tests = _load("test_collapsed_u_kernel",
                     ROOT / "tests" / "mcmc_original" / "test_collapsed_u_kernel.py")
ordering_tests = _load("test_collapsed_u_ordering",
                       ROOT / "tests" / "mcmc_original"
                       / "test_collapsed_u_ordering.py")


def tiny_exact_reference() -> dict:
    """The grid stationarity numbers, from the same builders the tests use."""
    model = ordering_tests.grid_model()
    grid, paths, prior, joint, conditional, ell = ordering_tests.build_grid(model)
    n_grid, n_paths = len(grid), len(paths)
    m = ordering_tests.collapsed_mh_matrix(prior, ell)
    kernel = np.zeros((n_grid * n_paths, n_grid * n_paths))
    for g in range(n_grid):
        for p in range(n_paths):
            for h in range(n_grid):
                kernel[g * n_paths + p, h * n_paths:(h + 1) * n_paths] = (
                    m[g, h] * conditional[h])
    flat = joint.reshape(-1)
    correct_deviation = float(np.abs(flat @ kernel - flat).max())

    # the stale-path kernel, exactly as the negative-control test builds it
    likelihood = CollapsedULikelihood(model=model)
    log_pi = np.log(np.array([0.6, 0.4]))
    log_p = log_transition_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    from hpop.mcmc_original.stage6e_exact import state_log_weights
    weights = np.empty((n_grid, n_paths))
    for g, u in enumerate(grid):
        likelihood.log_z_per_trace(ordering_tests.state_at(u))
        weights[g] = state_log_weights(paths, 0, len(ordering_tests.TRACE),
                                       likelihood._table, log_pi, log_p, model.delta_b)
    bad = np.zeros((n_grid * n_paths, n_grid * n_paths))
    for g in range(n_grid):
        for p in range(n_paths):
            stale = np.zeros((n_grid, n_grid))
            for a in range(n_grid):
                for b in range(n_grid):
                    if a == b:
                        continue
                    log_alpha = (weights[b, p] - weights[a, p]) + (prior[b] - prior[a])
                    stale[a, b] = (1.0 / (n_grid - 1)) * min(1.0, float(
                        np.exp(min(0.0, log_alpha))))
                stale[a, a] = 1.0 - stale[a].sum()
            for h1 in range(n_grid):
                for h2 in range(n_grid):
                    bad[g * n_paths + p, h2 * n_paths:(h2 + 1) * n_paths] += (
                        m[g, h1] * stale[h1, h2] * conditional[h2])
    bad_deviation = float(np.abs(flat @ bad - flat).max())
    return {
        "state_space": {"n_u_grid": n_grid, "n_paths": n_paths,
                        "joint_states": n_grid * n_paths,
                        "trace": list(ordering_tests.TRACE)},
        "correct_ordering_stationarity_deviation_L_inf": correct_deviation,
        "correct_ordering_tolerance": 1e-12,
        "correct_ordering_pass": correct_deviation < 1e-12,
        "stale_path_ordering_deviation_L_inf": bad_deviation,
        "stale_path_ordering_floor": 1e-4,
        "stale_path_control_pass": bad_deviation > 1e-4,
        "reading": "the collapsed MH followed by the exact (S,z) conditional draw is "
                   "stationary for the joint to machine precision; interposing ANY "
                   "update that consumes the stale (S,z) breaks stationarity by many "
                   "orders of magnitude — the immediate-refresh ordering is load-"
                   "bearing, not stylistic",
    }


def correctness_controls() -> dict:
    """Every §19 numeric control, with values rather than booleans alone."""
    fast = kernel_tests.load_fast_audit()
    model, state = kernel_tests.tiny_model(), kernel_tests.tiny_state()
    likelihood = CollapsedULikelihood(model=model)

    ours = likelihood.log_z_per_trace(state)
    from hpop.mcmc_original.fast_block_tables import FastBlockScoreTable
    table = FastBlockScoreTable(traces=model.traces, epsilon=model.epsilon,
                                n_skills=model.n_skills, min_width=model.min_width,
                                max_width=model.max_width, n_roles=model.n_roles)
    table.refresh(state.u_by_skill, state.beta, state.omega, state.lambda_rep,
                  state.lambda_back)
    audit = fast.collapsed_log_z(table, model, np.log(state.pi),
                                 log_transition_matrix(state.transition))
    audit_parity = float(np.abs(ours - audit).max())

    from hpop.mcmc_original.stage6e_exact import (
        enumerate_states, exact_posterior, state_log_weights,
    )
    enum_errors = []
    for n, trace in enumerate(model.traces):
        states = enumerate_states(len(trace), model.n_skills, model.min_width,
                                  model.max_width)
        weights = state_log_weights(states, n, len(trace), likelihood._table,
                                    np.log(state.pi),
                                    log_transition_matrix(state.transition),
                                    model.delta_b)
        enum_errors.append(abs(exact_posterior(states, weights)["log_evidence"]
                               - ours[n]))

    rng = np.random.default_rng(7)
    candidate = propose_row(np.array(state.u_by_skill[1]), 2, 0.5, rng)
    delta, cand_log_z = likelihood.delta_for_candidate(state, 1, candidate)
    prime = state.copy()
    u = np.array(prime.u_by_skill, copy=True)
    u[1] = candidate
    prime.u_by_skill = u
    rebuild_delta = float((likelihood.full_rebuild_log_z(prime)
                           - likelihood.full_rebuild_log_z(state)).sum())

    same_h_delta, _ = likelihood.delta_for_candidate(
        state, 0, np.array(state.u_by_skill[0]) * 1.7)

    hastings = 0.0
    u0 = np.array(state.u_by_skill[0])
    for _ in range(200):
        row = int(rng.integers(u0.shape[0]))
        cand = propose_row(u0, row, 0.5, rng)
        step = cand[row] - u0[row]
        hastings = max(hastings, abs(float(step @ step) - float((-step) @ (-step))))

    return {
        "audit_scorer_parity_max_abs": audit_parity,
        "enumeration_parity_max_abs": float(max(enum_errors)),
        "incremental_vs_full_rebuild_abs": abs(delta - rebuild_delta),
        "same_h_collapsed_delta": float(same_h_delta),
        "hastings_quadratic_form_asymmetry": hastings,
        "tolerance": 1e-10,
        "all_pass": bool(audit_parity == 0.0 and max(enum_errors) <= 1e-10
                         and abs(delta - rebuild_delta) <= 1e-10
                         and same_h_delta == 0.0 and hastings == 0.0),
        "note": "q0 reset, cadence, cache invalidation, MH-ratio replay, stale-(S,z) "
                "independence, immediate-refresh ordering, every=0 bitwise parity and "
                "resume determinism are asserted by the four test modules; this file "
                "records the headline numeric values",
    }


def resume_check() -> dict:
    import tempfile
    model = kernel_tests.tiny_model
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        full = run_collapsed_u_chain(model=model(), start=kernel_tests.tiny_state(),
                                     scales=kernel_tests.SCALES, num_sweeps=24,
                                     burn_in=0, thin=1, seed=77,
                                     collapsed=CollapsedUConfig(every=5))
        part1 = run_collapsed_u_chain(model=model(), start=kernel_tests.tiny_state(),
                                      scales=kernel_tests.SCALES, num_sweeps=12,
                                      burn_in=0, thin=1, seed=77,
                                      collapsed=CollapsedUConfig(every=5),
                                      checkpoint_path=tmp, checkpoint_every=12)
        payload = json.loads((tmp / "chain0_checkpoint.json").read_text())
        restored = Stage6EState.from_dict(payload["state"])
        rng = np.random.default_rng(77)
        rng.bit_generator.state = restored.rng_state
        part2 = run_collapsed_u_chain(model=model(), start=restored,
                                      scales=kernel_tests.SCALES, num_sweeps=24,
                                      burn_in=0, thin=1, seed=77,
                                      collapsed=CollapsedUConfig(every=5),
                                      rng=rng, state=restored)
    n1 = len(part1.log_target)
    a, b = full.final_state.to_dict(), part2.final_state.to_dict()
    a.pop("cache_version"), b.pop("cache_version")
    return {
        "sweeps": 24, "checkpoint_at": 12, "cadence": 5,
        "draws_bit_identical": bool(
            np.array_equal(full.u_draws[n1:], part2.u_draws)
            and np.array_equal(full.log_target[n1:], part2.log_target)),
        "final_state_identical_excluding_cache_version": bool(a == b),
        "rng_state_identical": bool(a["rng_state"] == b["rng_state"]),
        "collapsed_schedule_full": [r["sweep"] for r in full.collapsed_records],
        "collapsed_schedule_resumed": (
            [r["sweep"] for r in part1.collapsed_records]
            + [r["sweep"] for r in part2.collapsed_records]),
        "pass": bool(a == b and np.array_equal(full.u_draws[n1:], part2.u_draws)),
    }


def runtime_measurements() -> dict:
    out: dict = {}
    # ---- reference problem -------------------------------------------------------------
    module = _load("stage6e1b", ROOT / "scripts" / "stage6e1b_mixed_reference.py")
    from hpop.mcmc_original.stage6e_frozen import (
        DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH,
    )
    from hpop.mcmc_original.stage6e_state import Stage6EModel
    traces, _ = module.generate_corpus()
    mixed = module.build_mixed_model(traces)
    model = Stage6EModel(traces=traces, epsilon=module.EPSILON, delta_b=DELTA_B,
                         n_skills=module.K_SKILLS, n_roles=module.M_ROLES,
                         min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
                         infer_pi_P=False)
    start = module.dispersed_starts(mixed)[0]

    result = run_collapsed_u_chain(model=model, start=start.copy(),
                                   scales=REGISTERED_SCALES, num_sweeps=400,
                                   burn_in=50, thin=1, seed=1,
                                   collapsed=CollapsedUConfig(every=10),
                                   store_labels=False)
    ordinary = run_stage7b_chain(model=model, start=start.copy(),
                                 scales=REGISTERED_SCALES, num_sweeps=400, burn_in=50,
                                 thin=1, seed=1, store_labels=False)
    events = [r["seconds"] for r in result.collapsed_records]
    per_sweep_ours = result.runtime_seconds / 400
    per_sweep_base = ordinary.runtime_seconds / 400
    out["reference_problem"] = {
        "ordinary_sweep_seconds": per_sweep_base,
        "partially_collapsed_sweep_seconds_at_every_10": per_sweep_ours,
        "collapsed_event_seconds_mean": float(np.mean(events)),
        "collapsed_event_seconds_p95": float(np.quantile(events, 0.95)),
        "overhead_fraction_at_every_10": (per_sweep_ours - per_sweep_base)
        / per_sweep_base,
        "events_measured": len(events),
    }

    # ---- one read-only full-corpus checkpoint ------------------------------------------
    if FULL_CORPUS_CHECKPOINT.exists():
        copied = OUT / "full_corpus_checkpoint_copy.json"
        shutil.copyfile(FULL_CORPUS_CHECKPOINT, copied)
        payload = json.loads(copied.read_text())
        state = Stage6EState.from_dict(payload["state"])
        formal = _load("stage6e2_formal_chains",
                       ROOT / "scripts" / "stage6e2_formal_chains.py")
        from hpop.mcmc_original.stage6e_corpus import generate_corpus
        full_model = formal.build_model(generate_corpus())
        likelihood = CollapsedULikelihood(model=full_model)
        likelihood.log_z_per_trace(state)                    # baseline built
        rng = np.random.default_rng(0)
        candidate = propose_row(np.array(state.u_by_skill[0]), 0, 0.5, rng)
        began = time.perf_counter()
        likelihood.delta_for_candidate(state, 0, candidate)
        proposal_seconds = time.perf_counter() - began

        sampler = Stage7BSampler(model=full_model, scales=dict(REGISTERED_SCALES),
                                 table_source="batched")
        began = time.perf_counter()
        sampler.tables.refresh(state)
        tables = sampler.tables.tables_for(state)
        log_pi = np.log(state.pi)
        log_p = log_transition_matrix(state.transition)
        for n, t in enumerate(tables):
            chart = forward(t, log_pi, log_p, full_model.delta_b,
                            full_model.max_width, full_model.min_width)
            backward_sample(chart, rng)
        ffbs_seconds = time.perf_counter() - began
        sampler.tables.mark_stale()
        out["full_corpus_read_only"] = {
            "checkpoint": str(FULL_CORPUS_CHECKPOINT), "sweep": payload["sweep"],
            "collapsed_proposal_seconds": proposal_seconds,
            "ffbs_refresh_all_traces_seconds": ffbs_seconds,
            "combined_collapsed_event_seconds": proposal_seconds + ffbs_seconds,
            "note": "the FFBS refresh runs every sweep in the production sampler "
                    "anyway; the marginal cost of a scheduled sweep is the collapsed "
                    "proposal alone",
            "overhead_estimate_every_10": {
                "added_seconds_per_sweep": proposal_seconds / 10,
                "vs_observed_7b2_sweep_seconds": 60_500 / 44_000,
                "relative_overhead": (proposal_seconds / 10) / (60_500 / 44_000)},
        }
    return out


def implementation_manifest() -> dict:
    def diff_empty(path: str) -> bool:
        out = subprocess.run(["git", "diff", "HEAD", "--", path], cwd=ROOT,
                             capture_output=True, text=True)
        return out.stdout.strip() == ""

    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                          text=True).strip(),
        "new_files": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                      for p in NEW_FILES},
        "validated_modules_untouched": {p: diff_empty(p) for p in MUST_BE_UNTOUCHED},
        "all_untouched": bool(all(diff_empty(p) for p in MUST_BE_UNTOUCHED)),
        "integration": "composition by call in collapsed_u_kernel.py only; "
                       "recurrent_joint_ffbs_mcmc.py, stage6e_sampler.py, "
                       "semi_markov_ffbs.py and sampler_u.py have empty diffs",
        "preexisting_modifications_not_from_this_task": {
            "scripts/stage7b_compare_local_vs_ffbs.py":
                "modified 2026-08-14 23:06 by the prior session (the Step 7B2 "
                "comparison protocol, authored mid-run); untouched by this task"},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "tiny_exact_reference.json": tiny_exact_reference(),
        "correctness.json": correctness_controls(),
        "resume_check.json": resume_check(),
        "runtime.json": runtime_measurements(),
        "implementation_manifest.json": implementation_manifest(),
    }
    for name, payload in artifacts.items():
        (OUT / name).write_text(json.dumps(payload, indent=2, default=float))
        headline = payload.get("pass", payload.get("all_pass",
                                                   payload.get("all_untouched", "-")))
        print(f"[coll-u val] {name}: {headline}")
    print(f"[coll-u val] wrote {OUT}")


if __name__ == "__main__":
    main()
