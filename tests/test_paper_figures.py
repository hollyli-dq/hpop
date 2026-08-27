"""Tests for the paper-facing scalability figures.

These check provenance and claim discipline, not aesthetics. The figure is a claim about
measurements someone else will have to trust, so what is tested is that every plotted mark
can be traced to a benchmark artifact, that the contaminated first pass contributes
nothing, and that the projected banded-memory series is labelled as unimplemented wherever
it appears.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "paper" / "make_scalability_figures.py"
FIGURES = ROOT / "paper" / "figures"
BENCH = ROOT / "results" / "scalability" / "optimized_segmental_v1"

SCALABILITY_COMMIT = "07b474fe8bb961b9664c83d4152a11f648d07930"
BACKEND_COMMIT = "564995efd056d7d33984f0ca1532386e6140ea0c"


@pytest.fixture(scope="module")
def built():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)], cwd=str(ROOT), capture_output=True, text=True,
        timeout=900, env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
                          "MPLBACKEND": "Agg"})
    assert completed.returncode == 0, completed.stderr[-3000:]
    return json.loads((FIGURES / "fig_scalability_main_provenance.json").read_text())


def data_rows() -> list:
    with (FIGURES / "fig_scalability_main_data.csv").open() as handle:
        return list(csv.DictReader(handle))


def flat(text: str) -> str:
    """Caption text with its line wrapping removed, for substring checks."""
    return " ".join(text.split())


def timing_rows() -> list:
    with (BENCH / "timing_summary.csv").open() as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- outputs
def test_every_required_output_exists(built):
    for name in ("fig_scalability_main.pdf", "fig_scalability_main.png",
                 "fig_scalability_appendix.pdf", "fig_scalability_appendix.png",
                 "fig_scalability_main_caption.tex",
                 "fig_scalability_main_include.tex",
                 "fig_scalability_main_data.csv",
                 "fig_scalability_main_provenance.json"):
        assert (FIGURES / name).exists(), name


def test_figures_are_true_vector_pdfs(built):
    for name in ("fig_scalability_main.pdf", "fig_scalability_appendix.pdf"):
        raw = (FIGURES / name).read_bytes()
        assert b"/Subtype /Image" not in raw and b"/Subtype/Image" not in raw, name
        assert re.search(rb"/BaseFont", raw), f"{name} embeds no font"


def test_main_figure_fits_a_full_width_two_column_slot(built):
    raw = (FIGURES / "fig_scalability_main.pdf").read_bytes()
    box = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", raw)
    x0, y0, x1, y1 = (float(v) for v in box.groups())
    width, height = (x1 - x0) / 72.0, (y1 - y0) / 72.0
    assert 6.8 <= width <= 7.4, width
    assert 4.5 <= height <= 5.4, height


# ------------------------------------------------------------------------ provenance
def test_provenance_names_both_commits_and_the_controlled_pass(built):
    assert built["scalability_commit"] == SCALABILITY_COMMIT
    assert built["optimized_backend_commit"] == BACKEND_COMMIT
    assert built["controlled_pass_filter"]["value"] == "quiet"
    assert built["contaminated_first_pass_rows_used"] == 0
    assert built["controlled_pass_filter"]["rows_plotted"] > 0
    assert built["controlled_pass_filter"]["timing_rows_verified"] > 0


def test_provenance_hashes_match_the_files_on_disk(built):
    import hashlib
    for relative, digest in built["outputs"].items():
        path = ROOT / relative
        assert path.exists(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative
    for name, digest in built["source_artifacts"].items():
        assert hashlib.sha256((BENCH / name).read_bytes()).hexdigest() == digest, name


def test_provenance_records_the_fits_that_were_quoted(built):
    for key in ("J::J::cond_plain", "K::K::cond_plain"):
        entry = built["fits_used"][key]
        assert entry["exponent"] is not None
        assert len(entry["bootstrap_ci_95"]) == 2


# ----------------------------------------------------------- the contaminated pass
def test_no_plotted_row_comes_from_the_first_pass(built):
    rows = data_rows()
    assert rows
    assert {r["phase"] for r in rows} == {"quiet"}


def test_the_first_pass_exists_but_is_excluded(built):
    """The exclusion has to be a real filter, not an artifact of there being one pass."""
    phases = {r["phase"] for r in timing_rows()}
    assert "main" in phases, "the contaminated pass is missing; the filter is untested"
    assert "quiet" in phases


# ----------------------------------------------------- every mark traces to a source
def test_every_plotted_timing_matches_timing_summary(built):
    lookup = {(r["label"], r["operation"]): r for r in timing_rows()
              if r["phase"] == "quiet"}
    checked = 0
    for row in data_rows():
        if row["y_name"] != "wall_median_s":
            continue
        source = lookup[(row["label"], row["series"])]
        assert float(source["wall_median_s"]) == pytest.approx(float(row["y"]),
                                                               abs=1e-12)
        assert float(source["wall_ci_lo_s"]) == pytest.approx(float(row["y_ci_lo"]),
                                                              abs=1e-12)
        assert float(source["wall_ci_hi_s"]) == pytest.approx(float(row["y_ci_hi"]),
                                                              abs=1e-12)
        checked += 1
    assert checked >= 19


def test_every_plotted_memory_value_matches_memory_summary(built):
    with (BENCH / "memory_summary.csv").open() as handle:
        memory = [r for r in csv.DictReader(handle) if r["phase"] == "quiet"]
    dense = {r["label"]: float(r["dense_block_table_bytes"]) for r in memory}
    banded = {r["label"]: float(r["projected_banded_bytes_NOT_IMPLEMENTED"])
              for r in memory}
    checked = 0
    for row in data_rows():
        if row["series"] == "dense_score_table_gib":
            assert dense[row["label"]] / (1 << 30) == pytest.approx(float(row["y"]),
                                                                    rel=1e-9)
            checked += 1
        elif row["series"] == "projected_banded_gib":
            assert banded[row["label"]] / (1 << 30) == pytest.approx(float(row["y"]),
                                                                     rel=1e-9)
            checked += 1
    assert checked >= 14


def test_the_refused_point_is_marked_as_predicted_not_measured(built):
    rows = [r for r in data_rows() if r["series"] == "refused_predicted_gib"]
    assert len(rows) == 1
    row = rows[0]
    assert "PREDICTED" in row["note"] or "predicted" in row["note"]
    assert "never allocated" in row["note"]
    assert float(row["y"]) == pytest.approx(7.79, abs=0.02)


# ---------------------------------------------------------------- claim discipline
def test_projected_banded_storage_is_labelled_everywhere_it_appears(built):
    for row in data_rows():
        if row["series"] == "projected_banded_gib":
            assert "NOT IMPLEMENTED" in row["note"]
    assert "NOT IMPLEMENTED" in built["projected_banded_storage"]

    caption = flat((FIGURES / "fig_scalability_main_caption.tex").read_text()).lower()
    assert "banded" in caption
    assert "not implemented" in caption

    script = SCRIPT.read_text()
    assert "NOT IMPLEMENTED" in script


def test_caption_numbers_match_the_source_artifacts(built):
    caption = (FIGURES / "fig_scalability_main_caption.tex").read_text()
    fits = json.loads((BENCH / "complexity_fits.json").read_text())["by_phase"]["quiet"]
    j = fits["J::J"]["operations"]["cond_plain"]
    k = fits["K::K"]["operations"]["cond_plain"]
    assert f"{j['exponent']:.2f}" in caption
    assert f"{j['bootstrap_ci_95'][0]:.2f}" in caption
    assert f"{k['exponent']:.2f}" in caption
    assert f"{k['bootstrap_ci_95'][1]:.2f}" in caption

    lookup = {(r["label"], r["operation"]): r for r in timing_rows()
              if r["phase"] == "quiet"}
    plain = float(lookup[("target_operating_point", "cond_plain")]["wall_median_s"])
    rebuild = float(lookup[("target_operating_point",
                            "emission_build")]["wall_median_s"])
    assert f"{plain * 1000:.0f}" in caption
    assert f"{rebuild:.1f}" in caption


def test_caption_does_not_overclaim(built):
    caption = flat((FIGURES / "fig_scalability_main_caption.tex").read_text()).lower()
    for banned in ("posterior converges", "recovers the truth", "sublinear in $k$",
                   "scales to arbitrary", "memory is linear",
                   "banded storage is implemented", "marginalisation is free"):
        assert banned not in caption, banned
    # the K panel must say the fit is finite-range and that K^2 survives
    assert "finite-range" in caption
    assert "k^2" in caption or "k^{2}" in caption
    # not sold as a forward-pass exponent
    assert "plain sweep" in caption
    # and it must carry the disclaimer explicitly, not merely avoid the words
    assert "no claim about posterior convergence" in caption
    assert "throughput measurements" in caption


def test_k_panel_states_the_finite_range_caveat_on_the_figure_itself(built):
    script = SCRIPT.read_text()
    assert "finite-range fit; dense transitions retain a $K^2$ term" in script


def test_include_snippet_points_at_the_generated_files(built):
    include = (FIGURES / "fig_scalability_main_include.tex").read_text()
    assert "figures/fig_scalability_main.pdf" in include
    assert "figures/fig_scalability_main_caption.tex" in include
    assert "\\label{fig:scalability}" in include
    assert "figure*" in include


# ------------------------------------------------------- nothing else was touched
def test_no_benchmark_artifact_was_modified(built):
    changed = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--",
         "results/scalability"], capture_output=True, text=True, check=True).stdout
    assert changed.strip() == "", f"benchmark artifacts changed:\n{changed}"


def test_no_inference_source_was_modified(built):
    changed = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", "src/"],
        capture_output=True, text=True, check=True).stdout
    assert changed.strip() == ""


def test_the_figure_script_imports_no_inference_code():
    """A plotting step must not be able to run the model by accident."""
    source = SCRIPT.read_text()
    assert "hpop.mcmc" not in source
    assert "import hpop" not in source


def test_no_main_paper_tex_file_was_created_or_edited(built):
    """Only files under paper/figures/ may appear."""
    # `git status --porcelain` collapses a wholly untracked directory to one entry, so
    # the check walks the tree instead of trusting that summary.
    present = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "paper").rglob("*")
                     if p.is_file())
    assert present, "no paper files were written"
    for path in present:
        assert path.startswith("paper/figures/"), path
    assert not any(p.suffix == ".tex" and "figures" not in str(p)
                   for p in (ROOT / "paper").rglob("*.tex"))
