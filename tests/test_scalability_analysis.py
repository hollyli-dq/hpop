"""Tests for the analysis and reporting layer.

These run against a synthetic fixture of worker records rather than the real run, so they
test the arithmetic and the provenance rules without depending on whatever the benchmark
happened to measure. The one thing they assert about the real run, when it exists, is that
every number quoted in a document can be found in a CSV or JSON artifact.
"""

from __future__ import annotations

import csv
import os
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCALABILITY = ROOT / "scripts" / "scalability"
sys.path.insert(0, str(SCALABILITY))
sys.path.insert(0, str(ROOT / "src"))

import analyze                                                          # noqa: E402
import bench_common as bc                                               # noqa: E402
import report as rp                                                     # noqa: E402


# ------------------------------------------------------------------------- fixture
def _record(axis, label, group, N, J, K, A, D, regime, operations, seed=0):
    rng = np.random.default_rng(seed)
    ops = {}
    for name, base in operations.items():
        raw = [float(base * (1.0 + 0.01 * rng.standard_normal())) for _ in range(15)]
        ops[name] = {"wall_raw": raw, "cpu_raw": [v * 0.99 for v in raw],
                     "wall": bc.summarize(raw),
                     "cpu": bc.summarize([v * 0.99 for v in raw])}
    widths = range(3, D + 1)
    blocks = N * sum(max(0, J - w + 1) for w in widths)
    return {
        "status": "ok", "censored": False, "group": group,
        "config_id": f"{axis}/{label}", "reps_completed": 15,
        "reps_completed_per_operation": {k: 15 for k in ops},
        "config": {"axis": axis, "label": label, "N": N, "J": J, "K": K, "A": A,
                   "D_min": 3, "D_max": D, "regime": regime, "seed": 1,
                   "length_mix": [], "groups": ["build", "primitives", "cond", "marg"],
                   "min_reps": 15, "max_reps": 50, "warmups": 3, "timeout_s": 720,
                   "note": "", "op_reps": []},
        "operations": ops,
        "invariants": {"all_invariants_pass": True},
        "geometry": {"legal_blocks_total": blocks,
                     "legal_blocks_times_skills": blocks * K,
                     "legal_blocks_per_trace_mean": blocks / max(N, 1),
                     "n_legal_widths": len(list(widths)),
                     "trace_occurrences": N * J},
        "forward_work": {"forward_states": N * J * K,
                         "forward_duration_reductions": blocks * K,
                         "forward_transition_reductions": N * J * K * K,
                         "forward_total_reductions": blocks * K + N * J * K * K},
        "predicted_memory": bc.predict_memory(
            bc.BenchConfig(axis=axis, label=label, N=N, J=J, K=K, A=A, D_max=D),
            _FakeModel(N, J), group),
        "role_graph": {"n_roles": A, "n_skills": K, "relations_per_skill": [A * A // 4],
                       "relation_density_per_skill": [0.25],
                       "mean_relation_density": 0.25,
                       "mean_predecessors_per_role": A / 4.0,
                       "max_predecessors_per_role": A // 2},
        "preflight": {"allowed": True, "reasons": []},
        "peak_rss_bytes": int(2e8 + N * J * J * K * 8),
        "loadavg_before": [0.5, 0.5, 0.5], "loadavg_after": [0.5, 0.5, 0.5],
        "swap": {"used_mb": 100.0},
        "counters": {}, "flags": {"inline_logsumexp": True, "emission_hash_cache": True,
                                  "factorised_forward": True, "batched_forward": True},
    }


class _FakeModel:
    def __init__(self, N, J):
        self.traces = tuple(tuple(range(J)) for _ in range(N))


@pytest.fixture
def fixture_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    tasks = {}
    # a clean power law: forward time exactly proportional to J
    for J in (24, 48, 96, 192, 384):
        scale = J / 24.0
        for group, ops in (
                ("build", {"emission_build": 0.02 * scale,
                           "emission_cache_hit": 5e-5}),
                ("primitives", {"forward_batched": 0.001 * scale,
                                "backward_sample": 0.0008 * scale,
                                "ffbs_complete": 0.002 * scale}),
                ("cond", {"cond_plain": 0.005 * scale,
                          "cond_structural": 0.025 * scale}),
                ("marg", {"marg_plain": 0.005 * scale,
                          "marg_structural": 0.030 * scale})):
            record = _record("J", f"J_{J}", group, 16, J, 10, 20, 12, "full", ops,
                             seed=J)
            (raw / f"J_{J}__{group}.json").write_text(json.dumps(record))
            tasks[f"J_{J}::{group}"] = {"status": "ok", "attempts": 1, "seconds": 1.0,
                                        "reps": 15,
                                        "peak_rss_bytes": record["peak_rss_bytes"]}
    tasks["J_768::build"] = {"status": "skipped_memory", "attempts": 1, "seconds": 0.4,
                             "reason": "predicted RSS exceeds the frozen cap",
                             "predicted_rss_bytes": 9 * 1024 ** 3}
    (tmp_path / "state.json").write_text(json.dumps({
        "plan_digest": "x", "started_at_utc": "2026-08-22T22:00:00Z",
        "finished_at_utc": "2026-08-23T06:00:00Z", "tasks": tasks,
        "decisions": [{"what": "a recorded choice", "why": "a recorded reason"}],
        "runs": 1}))
    (tmp_path / "hardware_manifest.json").write_text(
        json.dumps(bc.hardware_manifest(), default=float))
    (tmp_path / "software_manifest.json").write_text(
        json.dumps(bc.software_manifest(), default=float))
    (tmp_path / "parity_results.json").write_text(json.dumps({
        "ALL_PASS": True, "n_points": 32, "n_passed": 32,
        "worst_max_abs_alpha_error": 1.1e-13, "worst_max_abs_logZ_error": 1.1e-13,
        "grid": {"J": [24, 48], "K": [3, 5], "A": [5, 10], "D_max": [6, 12]},
        "points": []}))
    return tmp_path


def _run(module, out_dir):
    completed = subprocess.run(
        [sys.executable, str(SCALABILITY / f"{module}.py"), "--out", str(out_dir)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
             "HOME": str(Path.home()), "MPLBACKEND": "Agg"})
    assert completed.returncode == 0, completed.stderr[-3000:]
    return completed


# ------------------------------------------------------------------ complexity fits
def test_a_known_power_law_is_recovered_exactly():
    xs = [1.0, 2.0, 4.0, 8.0, 16.0]
    fit = analyze.loglog_fit(xs, [3.0 * x ** 1.5 for x in xs])
    assert fit["exponent"] == pytest.approx(1.5, abs=1e-9)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)
    assert fit["prefactor_log10"] == pytest.approx(math.log10(3.0), abs=1e-9)


def test_fits_are_reproducible_bit_for_bit():
    rng = np.random.default_rng(0)
    xs = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    ys = [2.0 * x ** 1.2 * (1 + 0.05 * rng.standard_normal()) for x in xs]
    first, second = analyze.loglog_fit(xs, ys), analyze.loglog_fit(xs, ys)
    assert first == second
    assert first["bootstrap_ci_95"][0] < first["exponent"] < first["bootstrap_ci_95"][1]


def test_a_fit_is_refused_when_there_are_too_few_points():
    fit = analyze.loglog_fit([1.0, 2.0], [1.0, 2.0])
    assert fit["exponent"] is None and "fewer than three" in fit["note"]


def test_every_fit_records_its_own_range_and_an_extrapolation_caveat():
    fit = analyze.loglog_fit([10.0, 20.0, 40.0, 80.0], [1.0, 2.0, 4.0, 8.0])
    assert fit["x_min"] == 10.0 and fit["x_max"] == 80.0
    assert "160" in fit["extrapolation_caveat"]


def test_fits_admit_a_point_on_its_own_repetition_count():
    """A group flag must not discard an operation that banked clean repetitions.

    A configuration whose expensive operation ran out of budget can still have measured
    its cheap operation fifty times. The inclusion rule therefore looks at how many timed
    repetitions *this operation* got at *this point*, not at the surrounding group.
    """
    good = _record("J", "J_24", "primitives", 16, 24, 10, 20, 12, "full",
                   {"forward_batched": 0.001})
    flagged = _record("J", "J_48", "primitives", 16, 48, 10, 20, 12, "full",
                      {"forward_batched": 0.002})
    flagged["censored"] = True                  # the group hit the wall...
    flagged["budget_limited_operations"] = ["forward_batched"]
    thin = _record("J", "J_96", "primitives", 16, 96, 10, 20, 12, "full",
                   {"forward_batched": 0.004})
    for payload in thin["operations"].values():  # ...but this one is genuinely too thin
        payload["wall"]["n"] = 2
        payload["cpu"]["n"] = 2
    not_ok = _record("J", "J_192", "primitives", 16, 192, 10, 20, 12, "full",
                     {"forward_batched": 0.008})
    not_ok["status"] = "skipped_memory_preflight"

    xs, _, _, _, _ = analyze.series([good, flagged, thin, not_ok], "J",
                                    "forward_batched", "J")
    assert xs == [24.0, 48.0], "the flagged-but-well-sampled point must survive"


# ------------------------------------------------------------------------- pipeline
def test_analysis_writes_every_required_artifact(fixture_dir):
    _run("analyze", fixture_dir)
    for name in ("raw_timings.csv", "timing_summary.csv", "memory_summary.csv",
                 "censored_points.csv", "complexity_fits.json",
                 "marginalisation_overhead.json", "runtime_breakdown.json"):
        assert (fixture_dir / name).exists(), name
    figures = fixture_dir / "figures"
    for stem in ("fig_scaling_J", "fig_memory_JK", "fig_marg_overhead",
                 "fig_runtime_breakdown"):
        assert (figures / f"{stem}.png").exists(), stem
        assert (figures / f"{stem}.pdf").exists(), stem


def test_raw_timings_keeps_every_repetition(fixture_dir):
    _run("analyze", fixture_dir)
    with (fixture_dir / "raw_timings.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    # five J points, four groups, fifteen repetitions of each operation
    per_config = {}
    for row in rows:
        per_config.setdefault((row["label"], row["operation"]), 0)
        per_config[(row["label"], row["operation"])] += 1
    assert per_config
    assert set(per_config.values()) == {15}
    assert all(float(row["wall_seconds"]) > 0 for row in rows)


def test_the_recovered_exponent_matches_the_planted_one(fixture_dir):
    _run("analyze", fixture_dir)
    fits = json.loads((fixture_dir / "complexity_fits.json").read_text())
    forward = fits["axes"]["J::J"]["operations"]["forward_batched"]
    assert forward["exponent"] == pytest.approx(1.0, abs=0.05)
    lo, hi = forward["bootstrap_ci_95"]
    assert lo < 1.0 < hi


def test_censored_points_are_recorded_with_their_reason(fixture_dir):
    _run("analyze", fixture_dir)
    with (fixture_dir / "censored_points.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert any(r["status"] == "skipped_memory" and "cap" in r["reason"] for r in rows)


def test_banded_projection_is_labelled_in_every_artifact_that_carries_it(fixture_dir):
    _run("analyze", fixture_dir)
    header = (fixture_dir / "memory_summary.csv").read_text().splitlines()[0]
    assert "projected_banded_bytes_NOT_IMPLEMENTED" in header
    memory = json.loads(
        next((fixture_dir / "raw").glob("*__marg.json")).read_text()
    )["predicted_memory"]
    assert any("NOT_IMPLEMENTED" in key for key in memory)
    assert not any(key.startswith("measured_banded") for key in memory)


def test_marginalisation_overhead_is_computed_from_the_two_arms(fixture_dir):
    _run("analyze", fixture_dir)
    overhead = json.loads((fixture_dir / "marginalisation_overhead.json").read_text())
    row = overhead["per_configuration"]["J_96"]
    assert row["plain_ratio_marg_over_cond"] == pytest.approx(1.0, abs=0.05)
    assert row["structural_ratio_marg_over_cond"] > 1.0
    expected = (9 * row["marg_plain_s"] + row["marg_structural_s"]) / 10
    assert row["amortized_marg_s_at_cadence_10"] == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------- the documents
def test_reports_are_written_and_carry_the_required_discipline(fixture_dir):
    _run("analyze", fixture_dir)
    _run("report", fixture_dir)
    for name in ("SCALABILITY_REPORT.md", "SCALABILITY_LIMITATIONS.md",
                 "SAFE_PAPER_CLAIMS.md", "TODO_FOR_HOLLY.md",
                 "SCALABILITY_SECTION_DRAFT.tex"):
        assert (fixture_dir / name).exists(), name

    claims = (fixture_dir / "SAFE_PAPER_CLAIMS.md").read_text()
    for section in ("A. Directly measured claims", "B. Complexity-derived claims",
                    "C. Counterfactual banded-memory projections",
                    "D. Claims that must NOT appear"):
        assert section in claims, section
    for banned in ("scales to arbitrary K", "Memory is linear in J",
                   "posterior converges", "marginalization is free",
                   "Banded storage is implemented"):
        assert banned in claims, banned

    limits = (fixture_dir / "SCALABILITY_LIMITATIONS.md").read_text()
    assert "Not a convergence study" in limits
    assert "Not a recovery study" in limits
    assert "NOT IMPLEMENTED" in limits

    tex = (fixture_dir / "SCALABILITY_SECTION_DRAFT.tex").read_text()
    assert "not implemented" in tex.lower()
    assert "O(NJ^2K)" in tex or "NJ^2K" in tex


def test_every_banded_mention_carries_the_not_implemented_label(fixture_dir):
    _run("analyze", fixture_dir)
    _run("report", fixture_dir)
    for name in ("SCALABILITY_REPORT.md", "SCALABILITY_LIMITATIONS.md",
                 "SAFE_PAPER_CLAIMS.md", "SCALABILITY_SECTION_DRAFT.tex"):
        text = (fixture_dir / name).read_text()
        for line_number, line in enumerate(text.splitlines(), 1):
            lowered = line.lower()
            if "banded" not in lowered:
                continue
            if lowered.lstrip().startswith(("- \u274c", "\u274c")):
                continue        # a banned-claim bullet is itself the prohibition
            if "no banded" in lowered or "not implemented" in lowered:
                continue        # a line saying it was NOT done needs no further label
            window = "\n".join(text.splitlines()[max(0, line_number - 4):
                                                 line_number + 4]).lower()
            assert ("not implemented" in window or "not_implemented" in window
                    or "counterfactual" in window or "projected" in window), \
                f"{name}:{line_number} mentions banded storage without the label"


def test_paper_tables_are_derived_exactly_from_the_artifacts(fixture_dir):
    """Every timing the report quotes must be reproducible from timing_summary.csv."""
    _run("analyze", fixture_dir)
    _run("report", fixture_dir)
    art = rp.Artifacts(fixture_dir)
    with (fixture_dir / "timing_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(r["label"], r["operation"]): float(r["wall_median_s"]) for r in rows}

    text = (fixture_dir / "SCALABILITY_REPORT.md").read_text()
    checked = 0
    for (label, operation), value in lookup.items():
        rendered = rp.sec(value)
        if art.value(label, operation) is None:
            continue
        assert art.value(label, operation) == pytest.approx(value, rel=1e-9)
        if operation in ("forward_batched", "cond_plain", "emission_build"):
            assert rendered in text, f"{label}/{operation} rendered as {rendered}"
            checked += 1
    assert checked >= 5

    fits = json.loads((fixture_dir / "complexity_fits.json").read_text())
    quoted = fits["axes"]["J::J"]["operations"]["forward_batched"]["exponent"]
    assert f"{quoted:.2f}" in text


def test_report_survives_a_run_with_no_measurements(tmp_path):
    (tmp_path / "raw").mkdir()
    _run("analyze", tmp_path)
    _run("report", tmp_path)
    text = (tmp_path / "SCALABILITY_REPORT.md").read_text()
    assert "Scalability of exact segmental partial-order inference" in text
    assert (tmp_path / "SAFE_PAPER_CLAIMS.md").exists()


def test_no_convergence_or_recovery_language_leaks_into_the_documents(fixture_dir):
    _run("analyze", fixture_dir)
    _run("report", fixture_dir)
    banned = ("converged", "mixing well", "recovers the truth", "posterior is correct",
              "well-mixed")
    for name in ("SCALABILITY_SECTION_DRAFT.tex",):
        text = (fixture_dir / name).read_text().lower()
        for phrase in banned:
            assert phrase not in text, f"{name} contains {phrase!r}"


# ------------------------------------------------------ machine speed and two passes
def test_speed_probe_is_stable_and_deterministic_in_shape():
    """The probe is a ruler: same work every call, so a change in it is the machine."""
    bc.speed_probe()                                  # warm the allocator
    samples = [bc.speed_probe() for _ in range(7)]
    assert all(s > 0 for s in samples)
    assert max(samples) / min(samples) < 3.0, samples
    # it must do real work, not be optimised away
    assert min(samples) > 1e-4


def test_speed_probe_scales_with_the_work_it_is_asked_to_do():
    bc.speed_probe()
    small = min(bc.speed_probe(repeats=10) for _ in range(3))
    large = min(bc.speed_probe(repeats=40) for _ in range(3))
    assert large > 2.0 * small


def test_worker_records_the_probe(tmp_path):
    cfg = bc.BenchConfig(axis="t", label="probe", N=3, J=24, K=3, A=5, D_max=6,
                         min_reps=2, max_reps=2, warmups=1)
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg.as_dict()))
    out = tmp_path / "out.json"
    subprocess.run(
        [sys.executable, str(SCALABILITY / "bench_worker.py"), "--config",
         str(config_path), "--group", "primitives", "--out", str(out),
         "--deadline-s", "300"],
        cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=True, capture_output=True, text=True, timeout=600)
    payload = json.loads(out.read_text())
    assert len(payload["speed_probe_seconds"]) >= 3
    assert payload["speed_probe_median_seconds"] > 0


def test_an_exponent_is_refused_when_the_interval_contains_zero():
    """Noise must be reported as noise, never rendered as a small scaling exponent."""
    noisy = {"exponent": 0.25, "bootstrap_ci_95": [-0.22, 0.73], "r_squared": 0.169,
             "x_min": 5, "x_max": 50}
    rendered = rp.exp_str(noisy)
    assert "no detectable dependence" in rendered
    assert "interval contains zero" in rendered

    weak = {"exponent": 0.47, "bootstrap_ci_95": [0.08, 0.85], "r_squared": 0.30,
            "x_min": 6, "x_max": 96}
    assert "weak fit" in rp.exp_str(weak)

    clean = {"exponent": 1.01, "bootstrap_ci_95": [0.92, 1.10], "r_squared": 0.986,
             "x_min": 24, "x_max": 1024}
    rendered = rp.exp_str(clean)
    assert "no detectable" not in rendered and "weak fit" not in rendered
    assert "1.01" in rendered


def test_passes_are_loaded_separately_and_never_pooled(tmp_path):
    for directory, phase in (("raw", "main"), ("raw_quiet", "quiet")):
        target = tmp_path / directory
        target.mkdir(parents=True)
        # the same configuration, twice as slow in the first pass
        scale = 2.0 if phase == "main" else 1.0
        record = _record("J", "J_96", "primitives", 16, 96, 10, 20, 12, "full",
                         {"forward_batched": 0.004 * scale}, seed=1)
        record["speed_probe_median_seconds"] = 0.030 if phase == "main" else 0.017
        (target / "J_96__primitives.json").write_text(json.dumps(record))

    loaded = analyze.load(tmp_path)
    phases = {r["_phase"] for r in loaded["records"]}
    assert phases == {"main", "quiet"}
    assert len(loaded["records"]) == 2

    main_only = analyze.series(loaded["records"], "J", "forward_batched", "J",
                               phase="main")
    quiet_only = analyze.series(loaded["records"], "J", "forward_batched", "J",
                                phase="quiet")
    assert main_only[1][0] == pytest.approx(2 * quiet_only[1][0], rel=0.05)

    comparison = analyze.pass_comparison(loaded["records"])
    row = comparison["per_configuration"]["J_96"]
    assert row["operations"]["forward_batched"]["first_over_quiet"] == pytest.approx(
        2.0, rel=0.05)


def test_the_controlled_pass_is_the_primary_one(tmp_path):
    for directory, phase in (("raw", "main"), ("raw_quiet", "quiet")):
        target = tmp_path / directory
        target.mkdir(parents=True)
        for J in (24, 48, 96, 192):
            scale = (2.0 if phase == "main" else 1.0) * (J / 24.0)
            record = _record("J", f"J_{J}", "primitives", 16, J, 10, 20, 12, "full",
                             {"forward_batched": 0.001 * scale}, seed=J)
            record["speed_probe_median_seconds"] = 0.03 if phase == "main" else 0.017
            (target / f"J_{J}__primitives.json").write_text(json.dumps(record))
    loaded = analyze.load(tmp_path)
    fits = analyze.complexity_fits(loaded["records"])
    assert fits["primary_phase"] == "quiet"
    assert set(fits["by_phase"]) == {"main", "quiet"}
    # `axes` -- what the report reads -- must be the controlled pass
    assert fits["axes"] == fits["by_phase"]["quiet"]
    speed = fits["machine_speed"]
    assert speed["main"]["probe_median_s"] > speed["quiet"]["probe_median_s"]
