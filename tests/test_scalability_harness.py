"""Tests for the scalability benchmark harness. Nothing here tests inference.

The harness is new code that decides what gets measured, what gets refused and what gets
written down, so it needs its own gates. The inference engines it calls are already
covered by the project's existing suite, and none of these tests touch them beyond
asserting that the harness has not modified them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCALABILITY = ROOT / "scripts" / "scalability"
sys.path.insert(0, str(SCALABILITY))
sys.path.insert(0, str(ROOT / "src"))

import bench_common as bc                                              # noqa: E402
import bench_plan as bp                                                # noqa: E402

VALIDATED_COMMIT = "564995efd056d7d33984f0ca1532386e6140ea0c"


# ------------------------------------------------------- deterministic configuration
def test_corpus_is_a_pure_function_of_the_configuration():
    cfg = bc.BenchConfig(axis="t", label="t", N=6, J=40, K=4, A=12, D_max=12)
    first = bc.make_traces(cfg.N, cfg.J, cfg.K, cfg.A, cfg.regime, cfg.seed)
    second = bc.make_traces(cfg.N, cfg.J, cfg.K, cfg.A, cfg.regime, cfg.seed)
    assert first == second
    assert [len(t) for t in first] == [40] * 6
    assert max(max(t) for t in first) < 12


def test_a_different_seed_gives_a_different_corpus():
    common = dict(N=6, J=40, K=4, A=12, regime=bc.FULL_SUPPORT)
    assert (bc.make_traces(seed=1, **common) != bc.make_traces(seed=2, **common))


def test_latent_u_and_pi_p_are_reproducible():
    for regime in (bc.FULL_SUPPORT, bc.SPARSE_SUPPORT):
        a = bc.make_u(5, 20, regime, bc.BENCH_SEED)
        b = bc.make_u(5, 20, regime, bc.BENCH_SEED)
        assert np.array_equal(a, b)
    pi_a, p_a = bc.initial_pi_p(6, bc.BENCH_SEED)
    pi_b, p_b = bc.initial_pi_p(6, bc.BENCH_SEED)
    assert np.array_equal(pi_a, pi_b) and np.array_equal(p_a, p_b)
    assert np.array_equal(np.diag(p_a), np.zeros(6))


def test_length_mix_is_honoured_exactly():
    mix = tuple(J for J in (24, 32, 40, 48) for _ in range(25))
    traces = bc.make_traces(100, 48, 3, 5, bc.FULL_SUPPORT, bc.BENCH_SEED, mix)
    assert [len(t) for t in traces] == list(mix)


def test_sparse_regime_holds_the_role_graph_flat_while_the_vocabulary_grows():
    """The sparse regime's defining property: support size, not `A`, sets the graph."""
    densities = {}
    for A in (20, 30, 50):
        u = bc.make_u(6, A, bc.SPARSE_SUPPORT, bc.BENCH_SEED)
        summary = bc.role_graph_summary(u)
        densities[A] = summary["mean_relation_density"]
        assert summary["max_predecessors_per_role"] <= bc.SPARSE_SUPPORT_SIZE
    assert densities[50] < densities[30] < densities[20]

    full = {A: bc.role_graph_summary(
        bc.make_u(6, A, bc.FULL_SUPPORT, bc.BENCH_SEED))["mean_predecessors_per_role"]
        for A in (20, 30, 50)}
    assert full[50] > full[30] > full[20]


def test_the_two_regimes_coincide_when_every_role_is_in_support():
    for A in (5, 10):
        assert np.array_equal(bc.make_u(4, A, bc.FULL_SUPPORT, bc.BENCH_SEED),
                              bc.make_u(4, A, bc.SPARSE_SUPPORT, bc.BENCH_SEED))


def test_plan_is_deterministic_and_digest_is_stable():
    first, second = bp.full_plan(), bp.full_plan()
    assert [c.as_dict() for c in first] == [c.as_dict() for c in second]
    assert bp.plan_digest(first) == bp.plan_digest(second)
    labels = [c.label for c in first]
    assert len(labels) == len(set(labels))


def test_plan_covers_every_axis_the_study_registers():
    axes = {c.axis for c in bp.full_plan()}
    assert axes == {"baseline", "J", "K", "N", "D", "A_full", "A_sparse", "target",
                    "target_long"}


def test_tasks_cover_every_configuration_group_pair_exactly_once():
    configs = bp.full_plan()
    tasks = bp.tasks_for(configs)
    assert len(tasks) == sum(len(c.groups) for c in configs)
    assert len({t["task_id"] for t in tasks}) == len(tasks)


# ------------------------------------------------------------ counted geometry / work
def test_legal_block_count_matches_a_direct_enumeration():
    cfg = bc.BenchConfig(axis="t", label="t", N=3, J=20, K=4, A=6, D_max=7, D_min=3)
    model = bc.build_model(cfg)
    counted = bc.legal_block_count(cfg, model)
    brute = sum(1 for t in model.traces for a in range(len(t))
                for b in range(a + 1, len(t) + 1) if 3 <= b - a <= 7)
    assert counted["legal_blocks_total"] == brute
    assert counted["legal_blocks_times_skills"] == brute * 4


def test_forward_work_counts_match_the_recursion_index_ranges():
    cfg = bc.BenchConfig(axis="t", label="t", N=2, J=15, K=3, A=6, D_max=6, D_min=3)
    model = bc.build_model(cfg)
    work = bc.forward_work_counts(cfg, model)
    expected = 0
    for trace in model.traces:
        for b in range(1, len(trace) + 1):
            expected += max(0, (b - 3) - max(1, b - 6) + 1) * 3
    assert work["forward_duration_reductions"] == expected
    assert work["forward_states"] == 2 * 15 * 3


# ------------------------------------------------------------------ memory preflight
def test_memory_prediction_uses_the_exact_dense_shape():
    cfg = bc.BenchConfig(axis="t", label="t", N=4, J=50, K=6, A=10, D_max=12)
    model = bc.build_model(cfg)
    memory = bc.predict_memory(cfg, model, "primitives")
    assert memory["dense_block_table_bytes"] == 4 * 50 * 51 * 6 * 8
    assert memory["dense_copies_live_in_this_group"] == 1
    assert bc.predict_memory(cfg, model, "marg")["dense_copies_live_in_this_group"] == 2


def test_projected_banded_memory_is_labelled_and_smaller_than_dense():
    cfg = bc.BenchConfig(axis="t", label="t", N=4, J=200, K=6, A=10, D_max=12, D_min=3)
    model = bc.build_model(cfg)
    memory = bc.predict_memory(cfg, model)
    key = "projected_banded_bytes_NOT_IMPLEMENTED"
    assert key in memory
    assert memory[key] < memory["dense_block_table_bytes"]
    assert memory["projected_banded_saving_ratio_NOT_IMPLEMENTED"] > 1.0


def test_preflight_refuses_above_the_frozen_cap_and_allows_a_small_point():
    assert bc.memory_preflight(64 * 1024 ** 2)["allowed"]
    refused = bc.memory_preflight(64 * 1024 ** 3)
    assert not refused["allowed"] and refused["reasons"]
    assert bc.memory_cap_bytes() <= 6 * 1024 ** 3


def test_worker_refuses_an_oversized_configuration_without_allocating(tmp_path):
    """The refusal must come from the prediction, before the arrays exist."""
    cfg = bc.BenchConfig(axis="t", label="oversized", N=400, J=2000, K=40, A=20,
                         D_max=12, min_reps=1, max_reps=1, warmups=0)
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg.as_dict()))
    out = tmp_path / "out.json"
    completed = subprocess.run(
        [sys.executable, str(SCALABILITY / "bench_worker.py"), "--config",
         str(config_path), "--group", "primitives", "--out", str(out),
         "--deadline-s", "60"],
        cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out.read_text())
    assert payload["status"] == "skipped_memory_preflight"
    assert payload["preflight"]["reasons"]
    assert payload["peak_rss_bytes"] if "peak_rss_bytes" in payload else True


# ------------------------------------------------------------------- raw timing schema
@pytest.mark.parametrize("group", ["build", "primitives", "cond", "marg"])
def test_worker_emits_the_documented_schema_for_every_group(tmp_path, group):
    cfg = bc.BenchConfig(axis="t", label=f"schema_{group}", N=3, J=24, K=3, A=5,
                         D_max=6, min_reps=2, max_reps=2, warmups=1)
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg.as_dict()))
    out = tmp_path / "out.json"
    completed = subprocess.run(
        [sys.executable, str(SCALABILITY / "bench_worker.py"), "--config",
         str(config_path), "--group", group, "--out", str(out), "--deadline-s", "300"],
        cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out.read_text())

    for field in ("status", "config", "group", "operations", "invariants", "geometry",
                  "forward_work", "predicted_memory", "role_graph", "peak_rss_bytes",
                  "loadavg_before", "loadavg_after", "counters", "flags",
                  "reps_completed_per_operation"):
        assert field in payload, field
    assert payload["flags"] == {"inline_logsumexp": True, "emission_hash_cache": True,
                                "factorised_forward": True, "batched_forward": True}
    assert payload["operations"]
    for name, record in payload["operations"].items():
        assert len(record["wall_raw"]) == len(record["cpu_raw"]) >= 2
        assert all(v > 0 for v in record["wall_raw"]), name
        for statistic in ("median", "iqr", "p90", "ci_lo", "ci_hi",
                          "ci_relative_half_width", "n"):
            assert statistic in record["wall"], (name, statistic)
        # every raw repetition is kept, never only the summary
        assert record["wall"]["n"] == len(record["wall_raw"])
    assert payload["peak_rss_bytes"] > 0


def test_worker_invariants_are_actually_checked_not_merely_reported(tmp_path):
    cfg = bc.BenchConfig(axis="t", label="inv", N=3, J=24, K=3, A=5, D_max=6,
                         min_reps=2, max_reps=2, warmups=1)
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg.as_dict()))
    results = {}
    for group in ("primitives", "cond"):
        out = tmp_path / f"{group}.json"
        subprocess.run(
            [sys.executable, str(SCALABILITY / "bench_worker.py"), "--config",
             str(config_path), "--group", group, "--out", str(out),
             "--deadline-s", "300"],
            cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            check=True, capture_output=True, text=True, timeout=600)
        results[group] = json.loads(out.read_text())["invariants"]
    assert results["primitives"]["backward_draw_covers_trace"]
    assert results["primitives"]["backward_draw_no_self_transition"]
    assert results["primitives"]["no_nan_in_alpha"]
    assert results["cond"]["all_invariants_pass"]
    assert results["cond"]["plain_P_diagonal_exactly_zero"]
    assert results["cond"]["structural_scheduled_structural"] is True
    assert results["cond"]["plain_scheduled_structural"] is False


# ----------------------------------------------------------------- timeout handling
def test_worker_stops_adding_repetitions_at_its_deadline(tmp_path):
    """A deadline must yield a censored-but-recorded point, never a lost one."""
    cfg = bc.BenchConfig(axis="t", label="deadline", N=8, J=96, K=6, A=20, D_max=12,
                         min_reps=10_000, max_reps=10_000, warmups=0)
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg.as_dict()))
    out = tmp_path / "out.json"
    began = time.time()
    subprocess.run(
        [sys.executable, str(SCALABILITY / "bench_worker.py"), "--config",
         str(config_path), "--group", "build", "--out", str(out), "--deadline-s", "8"],
        cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=True, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - began
    payload = json.loads(out.read_text())
    assert elapsed < 120
    assert payload["censored"] is True
    assert payload["reps_completed"] >= 1
    assert payload["operations"]["emission_build"]["wall_raw"]


# ------------------------------------------------------------------- atomic and resume
def test_atomic_write_leaves_no_partial_file(tmp_path):
    target = tmp_path / "state.json"
    bc.atomic_write(target, json.dumps({"a": 1}))
    assert json.loads(target.read_text()) == {"a": 1}
    bc.atomic_write(target, json.dumps({"a": 2, "b": [1, 2, 3]}))
    assert json.loads(target.read_text())["a"] == 2
    assert not list(tmp_path.glob("*.tmp*"))


def test_driver_resumes_without_repeating_settled_tasks(tmp_path):
    import run_autopilot as ra
    first = ra.Autopilot(tmp_path, deadline=time.time() + 3600)
    task_id = first.tasks[0]["task_id"]
    first.state["tasks"][task_id].update(status="ok", seconds=1.0, reps=15)
    first.save()

    second = ra.Autopilot(tmp_path, deadline=time.time() + 3600)
    assert second.state["tasks"][task_id]["status"] == "ok"
    assert second.state["runs"] == first.state["runs"] + 1
    pending = [t for t in second.tasks
               if second.state["tasks"][t["task_id"]]["status"] == "pending"]
    assert task_id not in {t["task_id"] for t in pending}


def test_a_task_interrupted_mid_flight_returns_to_pending(tmp_path):
    import run_autopilot as ra
    first = ra.Autopilot(tmp_path, deadline=time.time() + 3600)
    task_id = first.tasks[3]["task_id"]
    first.state["tasks"][task_id]["status"] = "running"
    first.save()
    second = ra.Autopilot(tmp_path, deadline=time.time() + 3600)
    assert second.state["tasks"][task_id]["status"] == "pending"


def test_a_changed_plan_never_resumes_onto_the_old_state(tmp_path):
    import run_autopilot as ra
    first = ra.Autopilot(tmp_path, deadline=time.time() + 3600)
    first.save()
    state = json.loads((tmp_path / "state.json").read_text())
    state["plan_digest"] = "0" * 64
    (tmp_path / "state.json").write_text(json.dumps(state))
    second = ra.Autopilot(tmp_path, deadline=time.time() + 3600)
    assert second.state["plan_digest"] == second.digest
    assert (tmp_path / "state.superseded.json").exists()


def test_monotone_axis_skip_refuses_every_larger_point(tmp_path):
    import run_autopilot as ra
    pilot = ra.Autopilot(tmp_path, deadline=time.time() + 3600)
    for task in pilot.tasks:
        if task["config"].label == "J_192":
            pilot.state["tasks"][task["task_id"]]["status"] = "skipped_memory"
    assert pilot._refused_threshold("J") == 192
    bigger = next(t for t in pilot.tasks if t["config"].label == "J_384")
    record = pilot.run_task(bigger)
    assert record["status"] == "skipped_monotone"
    smaller = next(t for t in pilot.tasks if t["config"].label == "J_96")
    assert pilot._refused_threshold("K") is None
    assert smaller["config"].J < 192


def test_conditional_points_refuse_until_their_predecessor_is_clean(tmp_path):
    import run_autopilot as ra
    pilot = ra.Autopilot(tmp_path, deadline=time.time() + 3600)
    cfg_1024 = pilot.by_label["J_1024"]
    allowed, why = pilot._conditional_ok(cfg_1024)
    assert not allowed and "J_768" in why
    for task_id in list(pilot.state["tasks"]):
        if task_id.startswith("J_768::"):
            pilot.state["tasks"][task_id].update(status="ok", peak_rss_bytes=10 ** 8,
                                                 censored=False)
    allowed, _ = pilot._conditional_ok(cfg_1024)
    assert allowed
    for task_id in list(pilot.state["tasks"]):
        if task_id.startswith("J_768::"):
            pilot.state["tasks"][task_id]["peak_rss_bytes"] = 10 ** 12
    allowed, why = pilot._conditional_ok(cfg_1024)
    assert not allowed and "physical RAM" in why


def test_progress_heartbeat_is_written_and_readable(tmp_path):
    import run_autopilot as ra
    pilot = ra.Autopilot(tmp_path, deadline=time.time() + 3600)
    pilot.write_progress(current="demo::build")
    text = (tmp_path / "progress.md").read_text()
    assert "Scalability autopilot" in text
    assert "demo::build" in text
    assert "564995efd056d7d33984f0ca1532386e6140ea0c" in text


def test_events_are_append_only_and_one_json_object_per_line(tmp_path):
    import run_autopilot as ra
    pilot = ra.Autopilot(tmp_path, deadline=time.time() + 3600)
    pilot.event("a", x=1)
    pilot.event("b", y=2)
    lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["kind"] for line in lines] == ["a", "b"]


# --------------------------------------------------------------------- statistics
def test_bootstrap_interval_is_deterministic_and_brackets_the_median():
    samples = [1.0, 1.1, 1.05, 0.98, 1.02, 1.2, 0.95, 1.08, 1.01, 1.03]
    first = bc.bootstrap_median_ci(samples)
    assert first == bc.bootstrap_median_ci(samples)
    assert first["lo"] <= first["median"] <= first["hi"]
    assert first["relative_half_width"] >= 0.0
    tight = bc.bootstrap_median_ci([1.0] * 30)
    assert tight["relative_half_width"] == 0.0


def test_summary_reports_every_statistic_the_plan_requires():
    summary = bc.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    for key in ("n", "median", "iqr", "p90", "ci_lo", "ci_hi",
                "ci_relative_half_width"):
        assert key in summary
    assert summary["median"] == 3.0


# ------------------------------------------------------- the harness touches nothing
def _tracked_digests(prefix: str) -> dict:
    listing = subprocess.run(
        ["git", "-C", str(ROOT), "ls-tree", "-r", VALIDATED_COMMIT, "--", prefix],
        capture_output=True, text=True, check=True).stdout
    return {line.split("\t")[1]: line.split()[2] for line in listing.splitlines()}


def test_no_sealed_or_reference_source_has_been_modified():
    """Every inference source must still be the byte content of the validated commit."""
    changed = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", "src/"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert changed == "", f"the benchmark modified files under src/:\n{changed}"

    for prefix in ("src/hpop/mcmc_original", "src/hpop/mcmc_optimized"):
        for path, blob in _tracked_digests(prefix).items():
            current = subprocess.run(
                ["git", "-C", str(ROOT), "hash-object", path],
                capture_output=True, text=True, check=True).stdout.strip()
            assert current == blob, f"{path} differs from the validated commit"


def test_the_benchmark_adds_only_benchmark_paths():
    """Nothing outside a known set of additive paths may appear in the working tree.

    `paper/figures/` and `scripts/paper/` were added when the frozen benchmark results
    were turned into paper-facing figures. That step reads the benchmark artifacts and
    writes only figures, so it is additive in the same sense the benchmark is. The
    stricter guarantee -- that no sealed or reference source changed -- is asserted
    separately by `test_no_sealed_or_reference_source_has_been_modified`, which is
    untouched.
    """
    listing = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True, text=True, check=True).stdout.splitlines()
    allowed = ("scripts/scalability/", "scripts/paper/", "tests/test_scalability_",
               "tests/test_paper_figures.py", "results/scalability/",
               "paper/figures/", "paper/", "docs/scalability")
    for line in listing:
        path = line[3:].strip().strip('"')
        assert path.startswith(allowed), f"unexpected working-tree change: {path}"


def test_the_worktree_is_based_on_the_validated_commit():
    merge_base = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", VALIDATED_COMMIT, "HEAD"],
        capture_output=True, text=True)
    assert merge_base.returncode == 0
    assert bc.software_manifest()["git_describe_base"] == VALIDATED_COMMIT


def test_optimized_backend_runs_with_every_optimisation_on():
    from hpop.mcmc_optimized import FLAGS
    FLAGS.reset()
    assert FLAGS.snapshot() == {"inline_logsumexp": True, "emission_hash_cache": True,
                                "factorised_forward": True, "batched_forward": True}


# -------------------------------------------------------------------- the parity gate
def test_parity_gate_passes_on_a_representative_point():
    import bench_parity as parity
    row = parity.check_point(J=48, K=5, A=10, D=12, regime=bc.FULL_SUPPORT)
    assert row["max_abs_alpha_error"] <= parity.TOLERANCE
    assert row["max_abs_logZ_error"] <= parity.TOLERANCE
    assert row["inf_pattern_identical"]
    assert row["legal_block_counts_identical"]
    assert row["backward_draw_valid"]
    assert row["emission_tables_bitwise_identical"]
    assert row["PASS"]
    # the discrepancy is floating-point noise, not a different algorithm
    assert row["max_abs_alpha_error"] < 1e-11


def test_parity_gate_detects_a_genuinely_different_forward():
    """A tolerance that nothing can fail is not a gate; perturb and watch it fail."""
    from hpop.mcmc_original.semi_markov_ffbs import forward as reference_forward
    from hpop.mcmc_optimized.forward import forward_factorised
    cfg = bc.BenchConfig(axis="t", label="t", N=2, J=24, K=3, A=5, D_max=6)
    model = bc.build_model(cfg)
    state = bc.build_state(cfg, model)
    from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import FFBSBlockTables
    from hpop.mcmc_original.transitions import log_transition_matrix
    tables = FFBSBlockTables(model=model, source="batched")
    tables.refresh(state)
    table = list(tables.tables_for(state))[0]
    log_pi, log_p = np.log(state.pi), log_transition_matrix(state.transition)
    good = forward_factorised(table, log_pi, log_p, model.delta_b, model.max_width,
                              model.min_width)
    reference = reference_forward(table, log_pi, log_p, model.delta_b, model.max_width,
                                  model.min_width)
    assert abs(good.log_normalizer - reference.log_normalizer) <= 1e-10

    perturbed = np.array(table, copy=True)
    finite = np.isfinite(perturbed)
    perturbed[finite] += 1e-6
    bad = forward_factorised(perturbed, log_pi, log_p, model.delta_b, model.max_width,
                             model.min_width)
    assert abs(bad.log_normalizer - reference.log_normalizer) > 1e-10
