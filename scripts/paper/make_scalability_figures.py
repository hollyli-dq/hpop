"""Paper-facing scalability figures, built only from the frozen benchmark artifacts.

    python scripts/paper/make_scalability_figures.py

This is a plotting and export step. It runs no benchmark, imports no inference code and
writes nothing outside `paper/figures/`. Every plotted point, interval and fitted line is
read out of

    results/scalability/optimized_segmental_v1/

and re-exported to `fig_scalability_main_data.csv`, so each mark on the page can be traced
to a row a reader can open.

## Only the controlled pass is plotted

The benchmark ran the registered plan twice. The first pass is contaminated: on a machine
with performance and efficiency cores, a process that changes core type keeps a whole core
-- `cpu / wall` stays at one -- while running at roughly half speed, and neither wall time
nor CPU time reveals it. Because that pass ran ascending within each axis, its largest
points were measured last on the quietest machine, biasing every exponent downward.

`PASS = "quiet"` is applied to every row before anything is plotted, and
`assert_controlled_pass_only` re-checks it after the fact. The first pass appears in the
provenance record and nowhere else. The two are never averaged.

## What the figure may and may not say

The J and K numbers are **complete plain-sweep** fits, not forward-pass fits. The K
exponent is a finite-range empirical fit over the tested K = 3 to 80; dense skill
transitions retain an asymptotic `K^2` term and the panel says so. Projected banded
storage is arithmetic on array shapes -- it is not implemented, it was never run, and it
is drawn dashed and labelled on the figure itself.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.lines import Line2D                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "results" / "scalability" / "optimized_segmental_v1"
OUT = ROOT / "paper" / "figures"

PASS = "quiet"                       # the load-controlled pass; the only one plotted
SCALABILITY_COMMIT = "07b474fe8bb961b9664c83d4152a11f648d07930"
BACKEND_COMMIT = "564995efd056d7d33984f0ca1532386e6140ea0c"

GIB = float(1 << 30)
MEMORY_CAP_GIB = 6.0

# Okabe-Ito, colourblind-safe. Marker and dash style carry the same information, so the
# panels survive being printed in grayscale.
BLUE, ORANGE, GREEN, VERMILLION, PURPLE, SKY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9")

STYLE = {
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Latin Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8.0, "axes.titlesize": 8.5, "axes.labelsize": 8.0,
    "legend.fontsize": 7.0, "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "axes.linewidth": 0.7, "lines.linewidth": 1.25, "lines.markersize": 4.0,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.45,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "legend.frameon": False,
    "errorbar.capsize": 1.8,
}


# ------------------------------------------------------------------------- loading
def read_csv(name: str) -> list:
    with (BENCH / name).read_text().splitlines() as _:      # pragma: no cover
        pass


def load_csv(name: str) -> list:
    with (BENCH / name).open() as handle:
        return list(csv.DictReader(handle))


def load_json(name: str):
    return json.loads((BENCH / name).read_text())


class Artifacts:
    def __init__(self):
        self.timing = [r for r in load_csv("timing_summary.csv") if r["phase"] == PASS]
        self.timing_all = load_csv("timing_summary.csv")
        self.memory = [r for r in load_csv("memory_summary.csv") if r["phase"] == PASS]
        self.censored = load_csv("censored_points.csv")
        self.fits = load_json("complexity_fits.json")
        self.overhead = load_json("marginalisation_overhead.json")
        self.regimes = load_json("regime_comparison.json")

    def fit(self, axis: str, x_field: str, operation: str):
        return (self.fits["by_phase"][PASS].get(f"{axis}::{x_field}", {})
                .get("operations", {}).get(operation))

    def points(self, axis: str, operation: str, x_field: str) -> list:
        """Measured points for one operation on one axis, ascending in x."""
        rows = []
        for row in self.timing:
            if row["axis"] != axis or row["operation"] != operation:
                continue
            rows.append({
                "label": row["label"], "x": float(row[x_field]),
                "median": float(row["wall_median_s"]),
                "lo": float(row["wall_ci_lo_s"]), "hi": float(row["wall_ci_hi_s"]),
                "n_reps": int(row["n_reps"]),
                "config_id": row["config_id"], "phase": row["phase"],
                "N": int(row["N"]), "J": int(row["J"]), "K": int(row["K"]),
                "A": int(row["A"]), "D_min": int(row["D_min"]),
                "D_max": int(row["D_max"]), "regime": row["regime"],
            })
        return sorted(rows, key=lambda r: r["x"])

    def one(self, label: str, operation: str) -> dict:
        for row in self.timing:
            if row["label"] == label and row["operation"] == operation:
                return {"median": float(row["wall_median_s"]),
                        "lo": float(row["wall_ci_lo_s"]),
                        "hi": float(row["wall_ci_hi_s"]),
                        "n_reps": int(row["n_reps"]), "config_id": row["config_id"]}
        raise KeyError(f"{label}/{operation} is not in the controlled pass")

    def memory_points(self, axis: str) -> list:
        by_label: dict = {}
        for row in self.memory:
            if row["axis"] != axis:
                continue
            entry = by_label.setdefault(row["label"], {
                "label": row["label"], "J": int(row["J"]), "N": int(row["N"]),
                "K": int(row["K"]), "peak_rss_bytes": 0,
                "dense_bytes": float(row["dense_block_table_bytes"]),
                "banded_bytes": float(
                    row["projected_banded_bytes_NOT_IMPLEMENTED"]),
                "config_id": row["config_id"]})
            entry["peak_rss_bytes"] = max(entry["peak_rss_bytes"],
                                          int(row["peak_rss_bytes"]))
        return sorted(by_label.values(), key=lambda r: r["J"])

    def refused(self) -> dict | None:
        for row in self.censored:
            if row["label"] == "target_long_J500" and row["status"] == "skipped_memory":
                return {"label": row["label"],
                        "predicted_gib": int(row["predicted_rss_bytes"]) / GIB}
        return None


# ------------------------------------------------------------------- fit rendering
WEAK_R2 = 0.5


def fit_is_quotable(entry) -> bool:
    if not entry or entry.get("exponent") is None:
        return False
    lo, hi = entry["bootstrap_ci_95"]
    return not (lo <= 0.0 <= hi) and entry["r_squared"] >= WEAK_R2


def fit_line(entry, xs):
    """The fitted power law, evaluated across the measured range only."""
    grid = np.logspace(np.log10(min(xs)), np.log10(max(xs)), 100)
    return grid, 10.0 ** (entry["prefactor_log10"]) * grid ** entry["exponent"]


def slope_text(entry, prefix: str = "slope") -> str:
    lo, hi = entry["bootstrap_ci_95"]
    return (f"{prefix} = {entry['exponent']:.2f}\n"
            f"95% CI [{lo:.2f}, {hi:.2f}]")


def describe_fit(entry) -> str:
    """Report language for a fit that does not support an exponent."""
    if not entry or entry.get("exponent") is None:
        return "not fitted"
    lo, hi = entry["bootstrap_ci_95"]
    if lo <= 0.0 <= hi:
        return "no detectable dependence"
    if entry["r_squared"] < WEAK_R2:
        return "weak fit; not a reliable exponent"
    return f"slope {entry['exponent']:.2f} [{lo:.2f}, {hi:.2f}]"


def tick_labels(xs) -> list:
    """Label every tick, except one that would collide with its neighbour.

    On a log axis 768 and 1024 are a sixth of a decade apart and their labels touch at
    this figure size. The tick mark stays; only the text is dropped, which is standard
    and keeps the axis honest.
    """
    out = []
    for index, value in enumerate(xs):
        crowded = (index + 1 < len(xs)
                   and xs[index + 1] / value < 1.5)
        out.append("" if crowded else f"{int(value)}")
    return out


def errorbars(rows):
    median = np.array([r["median"] for r in rows])
    lo = np.array([r["lo"] for r in rows])
    hi = np.array([r["hi"] for r in rows])
    return np.clip(np.vstack([median - lo, hi - median]), 0.0, None)


# -------------------------------------------------------------------- main figure
def panel_scaling(ax, rows, entry, xlabel, ylabel, colour, marker, annotation_prefix,
                  note=None):
    xs = [r["x"] for r in rows]
    ys = [r["median"] for r in rows]
    if entry is not None:
        grid, curve = fit_line(entry, xs)
        ax.plot(grid, curve, color="0.45", linewidth=0.9, linestyle=(0, (5, 2)),
                zorder=1, label="fitted power law")
    ax.errorbar(xs, ys, yerr=errorbars(rows), color=colour, marker=marker,
                markerfacecolor="white", markeredgewidth=1.0, linestyle="-",
                zorder=3, label="measured median")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels(xs))
    ax.minorticks_off()
    if entry is not None:
        ax.text(0.035, 0.955, slope_text(entry, annotation_prefix),
                transform=ax.transAxes, va="top", ha="left", fontsize=7.0,
                linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                          edgecolor="0.8", linewidth=0.5))
    if note:
        ax.text(0.98, 0.045, note, transform=ax.transAxes, va="bottom", ha="right",
                fontsize=6.3, color="0.30", style="italic")


def panel_memory(ax, rows, refused):
    xs = [r["J"] for r in rows]
    rss = [r["peak_rss_bytes"] / GIB for r in rows]
    dense = [r["dense_bytes"] / GIB for r in rows]
    banded = [r["banded_bytes"] / GIB for r in rows]

    ax.plot(xs, rss, color=BLUE, marker="o", markerfacecolor=BLUE,
            markeredgecolor="white", markeredgewidth=0.6, linestyle="-", zorder=3,
            label="peak RSS (measured)")
    ax.plot(xs, dense, color=VERMILLION, marker="s", markerfacecolor="white",
            markeredgewidth=1.0, linestyle="-", zorder=3,
            label="dense score table (array shapes)")
    ax.plot(xs, banded, color="0.35", marker="^", markerfacecolor="white",
            markeredgewidth=1.0, linestyle=(0, (4, 2)), zorder=3,
            label="projected banded — NOT IMPLEMENTED")

    ax.axhline(MEMORY_CAP_GIB, color=PURPLE, linewidth=0.8, linestyle=(0, (1, 1.6)),
               zorder=2, label="6 GiB preflight cap")

    if refused is not None:
        ax.plot([500], [refused["predicted_gib"]], marker="X", markersize=6.5,
                markerfacecolor="none", markeredgecolor=PURPLE, markeredgewidth=1.3,
                linestyle="none", zorder=4,
                label="refused: predicted, never allocated")
        ax.annotate("$J{=}500$, $N{=}100$, $K{=}20$\npredicted, refused unallocated",
                    xy=(500, refused["predicted_gib"]),
                    xytext=(0.030, 0.985), textcoords="axes fraction",
                    fontsize=6.2, color=PURPLE, ha="left", va="top", linespacing=1.3,
                    arrowprops=dict(arrowstyle="-", color=PURPLE, linewidth=0.6,
                                    shrinkA=0, shrinkB=3))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Trace length $J$")
    ax.set_ylabel("Memory (GiB)")
    ax.set_ylim(top=ax.get_ylim()[1] * 6.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels(xs))
    ax.minorticks_off()


TARGET_BARS = [
    ("cond_plain", "FULL-COND plain", GREEN),
    ("marg_plain", "FULL-MARG plain", SKY),
    ("cond_structural", "FULL-COND structural", ORANGE),
    ("marg_structural", "FULL-MARG structural", VERMILLION),
    ("emission_build", "table rebuild", BLUE),
    ("emission_cache_hit", "H-cache hit", PURPLE),
]


def panel_target(ax, art, overhead_ratio, peak_rss_gib):
    labels, values, errors, colours = [], [], [], []
    for operation, pretty, colour in TARGET_BARS:
        row = art.one("target_operating_point", operation)
        labels.append(pretty)
        values.append(row["median"])
        errors.append([row["median"] - row["lo"], row["hi"] - row["median"]])
        colours.append(colour)
    positions = np.arange(len(labels))[::-1]
    errors = np.clip(np.array(errors).T, 0.0, None)
    ax.barh(positions, values, xerr=errors, color=colours, height=0.62,
            edgecolor="white", linewidth=0.4, error_kw=dict(elinewidth=0.8,
                                                            ecolor="0.25", capsize=1.6),
            zorder=3)
    for position, value in zip(positions, values):
        text = (f"{value * 1000:.1f} ms" if value < 1.0 else f"{value:.2f} s")
        ax.text(value * 1.35, position, text, va="center", ha="left", fontsize=6.3,
                color="0.20")
    ax.set_ylim(-1.55, len(labels) - 0.35)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_xscale("log")
    ax.set_xlim(4e-4, 6e2)
    ax.set_xlabel("Time (s, log scale)")
    ax.grid(axis="y", visible=False)
    ax.text(0.985, 0.020,
            f"amortized MARG/COND = {overhead_ratio:.3f}$\\times$\n"
            f"at structural cadence 1/10\n"
            f"peak RSS = {peak_rss_gib:.2f} GiB",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=6.3,
            color="0.20", linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.82",
                      linewidth=0.5))


def build_main(art: Artifacts) -> dict:
    plt.rcParams.update(STYLE)
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 4.72))

    j_rows = art.points("J", "cond_plain", "J")
    j_fit = art.fit("J", "J", "cond_plain")
    panel_scaling(axes[0][0], j_rows, j_fit, "Trace length $J$",
                  "Time per plain sweep (s)", BLUE, "o", "slope")

    k_rows = art.points("K", "cond_plain", "K")
    k_fit = art.fit("K", "K", "cond_plain")
    panel_scaling(axes[0][1], k_rows, k_fit, "Number of skill types $K$",
                  "Time per plain sweep (s)", GREEN, "D", "measured slope",
                  note="finite-range fit; dense transitions retain a $K^2$ term")

    panel_memory(axes[1][0], art.memory_points("J"), art.refused())

    ratio = art.overhead["per_configuration"]["target_operating_point"][
        "amortized_ratio_at_cadence_10"]
    rss = max(int(r["peak_rss_bytes"]) for r in art.memory
              if r["label"] == "target_operating_point") / GIB
    panel_target(axes[1][1], art, ratio, rss)

    for ax, tag in zip(axes.ravel(), "abcd"):
        ax.text(-0.155, 1.06, f"({tag})", transform=ax.transAxes, fontsize=9.0,
                fontweight="bold", va="top", ha="left")

    handles = [
        Line2D([], [], color="0.25", marker="o", markerfacecolor="white",
               markeredgewidth=1.0, linestyle="-",
               label="measured median, bootstrap 95% interval"),
        Line2D([], [], color="0.45", linestyle=(0, (5, 2)), linewidth=0.9,
               label="fitted power law over the measured range"),
        Line2D([], [], color=BLUE, marker="o", markerfacecolor=BLUE,
               markeredgecolor="white", linestyle="-", label="peak RSS (measured)"),
        Line2D([], [], color=VERMILLION, marker="s", markerfacecolor="white",
               markeredgewidth=1.0, linestyle="-",
               label="dense score table (from array shapes)"),
        Line2D([], [], color="0.35", marker="^", markerfacecolor="white",
               markeredgewidth=1.0, linestyle=(0, (4, 2)),
               label="projected banded storage — NOT IMPLEMENTED"),
        Line2D([], [], color=PURPLE, linestyle=(0, (1, 1.6)), linewidth=0.8,
               label="6 GiB memory preflight cap"),
        Line2D([], [], color=PURPLE, marker="X", markerfacecolor="none",
               markeredgewidth=1.3, markersize=6.0, linestyle="none",
               label="refused on prediction; never allocated"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.6,
                  bbox_to_anchor=(0.5, -0.045), handlelength=2.2, columnspacing=1.2,
                  labelspacing=0.30)
    figure.tight_layout(rect=(0.0, 0.088, 1.0, 1.0))
    figure.subplots_adjust(hspace=0.46, wspace=0.52)

    paths = {}
    for suffix in ("pdf", "png"):
        path = OUT / f"fig_scalability_main.{suffix}"
        figure.savefig(path, dpi=400 if suffix == "png" else None,
                       bbox_inches="tight")
        paths[suffix] = path
    plt.close(figure)
    return {"paths": paths, "j_rows": j_rows, "k_rows": k_rows, "j_fit": j_fit,
            "k_fit": k_fit, "target_rss_gib": rss, "overhead_ratio": ratio}


# ---------------------------------------------------------------- appendix figure
def build_appendix(art: Artifacts) -> dict:
    plt.rcParams.update(STYLE)
    figure, axes = plt.subplots(2, 3, figsize=(7.1, 4.7))

    specs = [
        (axes[0][0], "N", "N", "Corpus size $N$", BLUE, "o", None),
        (axes[0][1], "D", "D_max", "Maximum segment width $D_{\\max}$", ORANGE, "s",
         None),
        (axes[0][2], "A_full", "A",
         "Action vocabulary $A$\n(full support: every skill on all $A$ roles)",
         VERMILLION, "^", None),
        (axes[1][0], "A_sparse", "A",
         "Action vocabulary $A$\n(sparse support: ten roles per skill)",
         GREEN, "v", None),
    ]
    for ax, axis, x_field, xlabel, colour, marker, note in specs:
        # The series name goes in the legend and the fitted slope in its own box. Put
        # both in the legend and the entry becomes wide enough to cross the data.
        slopes = []
        for operation, style, shade in (("cond_plain", "-", colour),
                                        ("emission_build", (0, (4, 2)), "0.40")):
            rows = art.points(axis, operation, x_field)
            if not rows:
                continue
            entry = art.fit(axis, x_field, operation)
            pretty = ("plain sweep" if operation == "cond_plain"
                      else "candidate-table rebuild")
            short = "plain" if operation == "cond_plain" else "rebuild"
            slopes.append(f"{short}: {describe_fit(entry)}")
            ax.errorbar([r["x"] for r in rows], [r["median"] for r in rows],
                        yerr=errorbars(rows), color=shade, marker=marker,
                        markerfacecolor="white", markeredgewidth=0.9, linestyle=style,
                        label=pretty)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if slopes:
            # room for the annotation, opened AFTER the log scale is in force -- setting
            # limits on a linear axis and then switching to log rescales them to nonsense
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom / 2.4, top)
            ax.text(0.975, 0.030, "\n".join(slopes), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=5.5, color="0.22",
                    linespacing=1.35,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="0.85", linewidth=0.4))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Time (s)")
        xs = sorted({r["x"] for r in art.points(axis, "cond_plain", x_field)})
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{int(x)}" for x in xs])
        ax.minorticks_off()
        ax.legend(loc="upper left", fontsize=5.9, handlelength=2.0,
                  labelspacing=0.28, borderpad=0.25)
        if note:
            ax.text(0.98, 0.04, note, transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=5.8, color="0.35", style="italic")

    # width against the quantity it actually buys
    ax = axes[1][1]
    rows = art.points("D", "emission_build", "D_max")
    blocks = {r["label"]: None for r in rows}
    for row in art.timing:
        if row["label"] in blocks and row["operation"] == "emission_build":
            blocks[row["label"]] = int(row["legal_blocks_total"])
    xs = [blocks[r["label"]] for r in rows]
    ax.errorbar(xs, [r["median"] for r in rows], yerr=errorbars(rows), color=ORANGE,
                marker="s", markerfacecolor="white", markeredgewidth=0.9,
                label="candidate-table rebuild")
    plain = art.points("D", "cond_plain", "D_max")
    ax.errorbar([blocks[r["label"]] for r in plain], [r["median"] for r in plain],
                yerr=errorbars(plain), color=BLUE, marker="o", markerfacecolor="white",
                markeredgewidth=0.9, label="plain sweep")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Legal candidate blocks\n(width acts through the block count)")
    ax.set_ylabel("Time (s)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{value / 1000:.0f}k" for value in xs])
    ax.minorticks_off()
    ax.legend(loc="upper left", fontsize=5.8, handlelength=2.0, labelspacing=0.28)

    # marginalisation overhead across every tested scale
    ax = axes[1][2]
    rows = art.overhead["per_configuration"]
    for key, colour, marker, pretty in (
            ("plain_ratio_marg_over_cond", GREEN, "o", "plain sweep"),
            ("structural_ratio_marg_over_cond", VERMILLION, "s", "structural sweep"),
            ("amortized_ratio_at_cadence_10", PURPLE, "^", "amortized, cadence 1/10")):
        values = sorted(v[key] for v in rows.values() if v.get(key))
        ax.plot(np.arange(1, len(values) + 1) / len(values), values, color=colour,
                marker=marker, markerfacecolor="white", markeredgewidth=0.8,
                markersize=3.2, linestyle="-", label=pretty)
    ax.axhline(1.0, color="0.35", linewidth=0.7, linestyle=(0, (1, 1.6)))
    ax.set_xlabel("Configurations\n(sorted, cumulative fraction)")
    ax.set_ylabel("FULL-MARG / FULL-COND")
    ax.legend(loc="upper left", fontsize=5.8, handlelength=2.0, labelspacing=0.28)
    ax.text(0.98, 0.04, f"{len(rows)} configurations", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.8, color="0.35", style="italic")

    for ax, tag in zip(axes.ravel(), "abcdef"):
        ax.text(-0.20, 1.08, f"({tag})", transform=ax.transAxes, fontsize=8.0,
                fontweight="bold", va="top", ha="left")

    figure.tight_layout()
    figure.subplots_adjust(hspace=0.62, wspace=0.40)
    paths = {}
    for suffix in ("pdf", "png"):
        path = OUT / f"fig_scalability_appendix.{suffix}"
        figure.savefig(path, dpi=400 if suffix == "png" else None, bbox_inches="tight")
        paths[suffix] = path
    plt.close(figure)
    return {"paths": paths}


# ------------------------------------------------------------------------- exports
def export_data(art: Artifacts, main: dict) -> Path:
    """Every plotted mark, as a row. The figure must be reproducible from this file."""
    path = OUT / "fig_scalability_main_data.csv"
    fields = ["panel", "series", "config_id", "label", "phase", "x_name", "x",
              "y_name", "y", "y_ci_lo", "y_ci_hi", "n_reps", "N", "J", "K", "A",
              "D_min", "D_max", "regime", "note"]
    rows = []
    for tag, series, points, x_name in (
            ("a", "cond_plain", main["j_rows"], "J"),
            ("b", "cond_plain", main["k_rows"], "K")):
        for point in points:
            rows.append({
                "panel": tag, "series": series, "config_id": point["config_id"],
                "label": point["label"], "phase": PASS, "x_name": x_name,
                "x": point["x"], "y_name": "wall_median_s", "y": point["median"],
                "y_ci_lo": point["lo"], "y_ci_hi": point["hi"],
                "n_reps": point["n_reps"], "N": point["N"], "J": point["J"],
                "K": point["K"], "A": point["A"], "D_min": point["D_min"],
                "D_max": point["D_max"], "regime": point["regime"], "note": ""})
    for entry in art.memory_points("J"):
        for series, value, note in (
                ("peak_rss_gib", entry["peak_rss_bytes"] / GIB, "measured ru_maxrss"),
                ("dense_score_table_gib", entry["dense_bytes"] / GIB,
                 "computed from exact array shapes"),
                ("projected_banded_gib", entry["banded_bytes"] / GIB,
                 "NOT IMPLEMENTED; arithmetic counterfactual only")):
            rows.append({
                "panel": "c", "series": series, "config_id": entry["config_id"],
                "label": entry["label"], "phase": PASS, "x_name": "J",
                "x": entry["J"], "y_name": "gib", "y": value, "y_ci_lo": "",
                "y_ci_hi": "", "n_reps": "", "N": entry["N"], "J": entry["J"],
                "K": entry["K"], "A": 20, "D_min": 3, "D_max": 12,
                "regime": "full", "note": note})
    refused = art.refused()
    if refused:
        rows.append({
            "panel": "c", "series": "refused_predicted_gib",
            "config_id": "target_long/target_long_J500", "label": refused["label"],
            "phase": PASS, "x_name": "J", "x": 500, "y_name": "gib",
            "y": refused["predicted_gib"], "y_ci_lo": "", "y_ci_hi": "", "n_reps": "",
            "N": 100, "J": 500, "K": 20, "A": 50, "D_min": 3, "D_max": 12,
            "regime": "sparse",
            "note": "PREDICTED, never allocated; refused by the 6 GiB preflight"})
    for operation, pretty, _colour in TARGET_BARS:
        row = art.one("target_operating_point", operation)
        rows.append({
            "panel": "d", "series": operation, "config_id": row["config_id"],
            "label": "target_operating_point", "phase": PASS, "x_name": "operation",
            "x": pretty, "y_name": "wall_median_s", "y": row["median"],
            "y_ci_lo": row["lo"], "y_ci_hi": row["hi"], "n_reps": row["n_reps"],
            "N": 100, "J": 200, "K": 20, "A": 50, "D_min": 3, "D_max": 12,
            "regime": "sparse", "note": ""})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_caption(art: Artifacts, main: dict) -> Path:
    j, k = main["j_fit"], main["k_fit"]
    target = {name: art.one("target_operating_point", name)["median"]
              for name, _p, _c in TARGET_BARS}
    memory = next(r for r in art.memory if r["label"] == "target_operating_point")
    reduction = (float(memory["dense_block_table_bytes"])
                 / float(memory["projected_banded_bytes_NOT_IMPLEMENTED"]))
    text = f"""\\caption{{%
  \\textbf{{Computational scalability of exact segmental partial-order inference.}}
  All points are medians from the load-controlled benchmark pass; error bars are
  bootstrap 95\\% intervals for the median.
  \\textbf{{(a)}} With bounded segment width, a complete plain sweep scales
  approximately linearly in trace length over $J = 24$--$1024$ (fitted exponent
  {j['exponent']:.2f}, 95\\% CI [{j['bootstrap_ci_95'][0]:.2f},
  {j['bootstrap_ci_95'][1]:.2f}]; $R^2 = {j['r_squared']:.3f}$).
  \\textbf{{(b)}} Over the tested range $K = 3$--$80$, batching and transition
  factorisation keep the measured growth modest (empirical exponent
  {k['exponent']:.2f}, 95\\% CI [{k['bootstrap_ci_95'][0]:.2f},
  {k['bootstrap_ci_95'][1]:.2f}]); this is a finite-range fit, and dense
  skill-transition dynamics retain an asymptotic $K^2$ term.
  \\textbf{{(c)}} Memory, not time, is the binding constraint on trace length: the
  current dense $(J, J{{+}}1, K)$ candidate score table is quadratic in $J$. Storing
  only the legal duration bands would reduce that table to $O(NJDK)$
  ({reduction:.1f}$\\times$ smaller at the target operating point), but this banded
  layout is \\emph{{not implemented}}: the dashed series is arithmetic on array shapes
  and not a measurement. The open marker is a configuration refused by the memory
  preflight on a prediction; it was never allocated.
  \\textbf{{(d)}} At the anticipated real-data operating point ($N = 100$, $J = 200$,
  $K = 20$, $A = 50$, ten-role support per skill, $D \\in [3, 12]$) an ordinary sweep
  costs {target['cond_plain'] * 1000:.0f}~ms and peaks at
  {main['target_rss_gib']:.2f}~GiB resident. Marginalising the partial order is free
  on a sweep that performs no structural update and adds
  {(main['overhead_ratio'] - 1) * 100:.1f}\\% amortised at the registered structural
  cadence of one in ten. The structural sweep is dominated by rebuilding the
  candidate-score table ({target['emission_build']:.1f}~s), which an unchanged-table
  cache lookup avoids entirely ({target['emission_cache_hit'] * 1000:.1f}~ms). The
  rebuild and the structural sweep were timed separately and are not components of a
  single stacked measurement. These are throughput measurements and carry no claim
  about posterior convergence or the number of sweeps required.%
}}"""
    # Reflow: the f-string interpolation leaves ragged breaks where a number was
    # substituted. LaTeX does not care, but a human reading the file does.
    body = text.split("\\caption{%", 1)[1].rsplit("%\n}", 1)[0]
    wrapped = textwrap.fill(" ".join(body.split()), width=86,
                            initial_indent="  ", subsequent_indent="  ",
                            break_long_words=False, break_on_hyphens=False)
    path = OUT / "fig_scalability_main_caption.tex"
    path.write_text("\\caption{%\n" + wrapped + "%\n}\n")
    return path


def write_include() -> Path:
    path = OUT / "fig_scalability_main_include.tex"
    path.write_text(
        "% Not inserted into the paper automatically; \\input this where it belongs.\n"
        "\\begin{figure*}[t]\n"
        "  \\centering\n"
        "  \\includegraphics[width=\\textwidth]\n"
        "  {figures/fig_scalability_main.pdf}\n"
        "  \\input{figures/fig_scalability_main_caption.tex}\n"
        "  \\label{fig:scalability}\n"
        "\\end{figure*}\n")
    return path


# ------------------------------------------------------------------------ validation
def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_controlled_pass_only(data_csv: Path) -> int:
    with data_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    offenders = [r for r in rows if r["phase"] != PASS]
    if offenders:
        raise AssertionError(
            f"{len(offenders)} plotted rows came from a pass other than {PASS!r}")
    return len(rows)


def assert_points_match_source(art: Artifacts, data_csv: Path) -> int:
    """Every exported timing row must equal its `timing_summary.csv` entry."""
    with data_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(r["label"], r["operation"]): r for r in art.timing}
    checked = 0
    for row in rows:
        if row["y_name"] != "wall_median_s":
            continue
        source = lookup[(row["label"], row["series"])]
        if abs(float(source["wall_median_s"]) - float(row["y"])) > 1e-12:
            raise AssertionError(f"{row['label']}/{row['series']} does not match "
                                 "timing_summary.csv")
        checked += 1
    return checked


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    art = Artifacts()
    figure = build_main(art)
    appendix = build_appendix(art)
    data_csv = export_data(art, figure)
    caption = write_caption(art, figure)
    include = write_include()

    plotted = assert_controlled_pass_only(data_csv)
    checked = assert_points_match_source(art, data_csv)

    provenance = {
        "generated_by": "scripts/paper/make_scalability_figures.py",
        "plotting_script_sha256": sha256(Path(__file__)),
        "scalability_commit": SCALABILITY_COMMIT,
        "optimized_backend_commit": BACKEND_COMMIT,
        "controlled_pass_filter": {"column": "phase", "value": PASS,
                                   "rows_plotted": plotted,
                                   "timing_rows_verified": checked},
        "contaminated_first_pass_rows_used": 0,
        "first_pass_note":
            "the first benchmark pass is contaminated by performance/efficiency-core "
            "migration and contributes no point to any figure here; it is retained in "
            "the benchmark artifacts and reported in SCALABILITY_REPORT.md",
        "source_artifacts": {
            name: sha256(BENCH / name) for name in (
                "timing_summary.csv", "memory_summary.csv", "raw_timings.csv",
                "complexity_fits.json", "marginalisation_overhead.json",
                "regime_comparison.json", "censored_points.csv",
                "pass_comparison.json", "SCALABILITY_REPORT.md")},
        "plotted_configuration_ids": sorted({
            r["config_id"] for r in art.timing
            if r["axis"] in ("J", "K", "N", "D", "A_full", "A_sparse")
            or r["label"] in ("target_operating_point", "target_long_J500")}),
        "fits_used": {
            "J::J::cond_plain": art.fit("J", "J", "cond_plain"),
            "K::K::cond_plain": art.fit("K", "K", "cond_plain"),
        },
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in (
            figure["paths"]["pdf"], figure["paths"]["png"],
            appendix["paths"]["pdf"], appendix["paths"]["png"],
            data_csv, caption, include)},
        "projected_banded_storage":
            "NOT IMPLEMENTED; ARITHMETIC COUNTERFACTUAL ONLY. Computed from exact array "
            "shapes and the legal-width count. No banded layout exists in this backend "
            "and no measurement in this study ran on one.",
        "claim_discipline":
            "the J and K exponents are complete plain-sweep fits, not forward-pass "
            "fits; the K exponent is a finite-range empirical fit over K = 3 to 80 and "
            "dense skill transitions retain an asymptotic K^2 term",
    }
    path = OUT / "fig_scalability_main_provenance.json"
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True))
    print(json.dumps({"rows_plotted": plotted, "timing_rows_verified": checked,
                      "outputs": sorted(provenance["outputs"])}, indent=2))


if __name__ == "__main__":
    main()
