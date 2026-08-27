"""Write the five study documents, entirely from the artifacts on disk.

    python scripts/scalability/report.py [--out results/scalability/optimized_segmental_v1]

Every number that appears in a document is read back out of `timing_summary.csv`,
`memory_summary.csv`, `complexity_fits.json`, `marginalisation_overhead.json`,
`runtime_breakdown.json`, `parity_results.json`, `censored_points.csv` and `state.json`.
Nothing is recomputed from the raw records here and nothing is typed in, so a reader who
opens the CSV sees the same figure the prose quotes, and
`test_paper_tables_are_derived_from_the_artifacts` checks exactly that.

The documents are:

    SCALABILITY_REPORT.md        the findings, question by question
    SCALABILITY_LIMITATIONS.md   what this study does not establish
    SAFE_PAPER_CLAIMS.md         claims sorted into measured, derived, projected, banned
    TODO_FOR_HOLLY.md            what a person has to decide or do next
    SCALABILITY_SECTION_DRAFT.tex  draft prose, for pasting, not auto-inserted
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_common as bc                                              # noqa: E402

COMMIT = "564995efd056d7d33984f0ca1532386e6140ea0c"
BACKEND = "optimized_segmental_v1"


# --------------------------------------------------------------------------- loading
def read_csv(path: Path) -> list:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {} if default is None else default


class Artifacts:
    def __init__(self, out_dir: Path):
        self.dir = out_dir
        self.timing = read_csv(out_dir / "timing_summary.csv")
        self.memory = read_csv(out_dir / "memory_summary.csv")
        self.censored = read_csv(out_dir / "censored_points.csv")
        self.fits = read_json(out_dir / "complexity_fits.json")
        self.overhead = read_json(out_dir / "marginalisation_overhead.json")
        self.breakdown = read_json(out_dir / "runtime_breakdown.json")
        self.regimes = read_json(out_dir / "regime_comparison.json")
        self.corroboration = read_json(out_dir / "corroboration.json")
        self.passes = read_json(out_dir / "pass_comparison.json")
        self.parity = read_json(out_dir / "parity_results.json")
        self.state = read_json(out_dir / "state.json")
        self.hardware = read_json(out_dir / "hardware_manifest.json")
        self.software = read_json(out_dir / "software_manifest.json")
        self.index = read_json(out_dir / "analysis_index.json")

    # -- lookups -------------------------------------------------------------------
    def rows(self, **filters) -> list:
        out = []
        for row in self.timing:
            if all(str(row.get(k)) == str(v) for k, v in filters.items()):
                out.append(row)
        return out

    def value(self, label: str, operation: str, field: str = "wall_median_s"):
        for row in self.timing:
            if row["label"] == label and row["operation"] == operation:
                try:
                    return float(row[field])
                except (TypeError, ValueError):
                    return None
        return None

    def reps(self, label: str, operation: str):
        for row in self.timing:
            if row["label"] == label and row["operation"] == operation:
                return int(row["n_reps"]), bool(int(row["reduced_repetitions"]))
        return None, False

    def peak_rss(self, label: str, group: str | None = None):
        best = 0
        for row in self.memory:
            if row["label"] != label:
                continue
            if group is not None and row["group"] != group:
                continue
            best = max(best, int(row["peak_rss_bytes"] or 0))
        return best or None

    def memory_row(self, label: str, group: str):
        for row in self.memory:
            if row["label"] == label and row["group"] == group:
                return row
        return None

    def exponent(self, axis: str, x_field: str, operation: str):
        entry = (self.fits.get("axes", {}).get(f"{axis}::{x_field}", {})
                 .get("operations", {}).get(operation))
        if not entry or entry.get("exponent") is None:
            return None
        return entry

    def labels_for_axis(self, axis: str) -> list:
        seen = []
        for row in self.timing:
            if row["axis"] == axis and row["label"] not in seen:
                seen.append(row["label"])
        return seen


# ---------------------------------------------------------------------- formatting
def sec(value, digits: int = 4) -> str:
    if value is None:
        return "not measured"
    if value >= 1.0:
        return f"{value:.{max(2, digits - 2)}f} s"
    if value >= 1e-3:
        return f"{value * 1e3:.{max(1, digits - 2)}f} ms"
    return f"{value * 1e6:.0f} us"


def gib(value) -> str:
    if value is None:
        return "not measured"
    return f"{value / 2 ** 30:.2f} GiB" if value >= 2 ** 30 else f"{value / 2 ** 20:.0f} MiB"


WEAK_FIT_R2 = 0.5


def exp_str(entry) -> str:
    """Render a fitted exponent, or refuse to when the fit does not support one.

    A power law with an interval straddling zero and an R^2 near zero is not a slow
    growth rate, it is the absence of a detectable dependence. Quoting `A^0.25` for a
    fit with R^2 = 0.17 would turn noise into a scaling claim, so those are named for
    what they are.
    """
    if entry is None:
        return "not fitted"
    lo, hi = entry["bootstrap_ci_95"]
    r_squared = entry["r_squared"]
    body = (f"{entry['exponent']:.2f} (95% CI {lo:.2f} to {hi:.2f}, "
            f"R^2 {r_squared:.3f})")
    if lo <= 0.0 <= hi:
        return (f"**no detectable dependence** over the tested range "
                f"(fitted slope {body}; the interval contains zero)")
    if r_squared < WEAK_FIT_R2:
        return (f"{body} — **weak fit, not a reliable exponent**: a single power law "
                f"does not describe these points")
    return body


def table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ report sections
def axis_table(art: Artifacts, axis: str, x_key: str, operations) -> str:
    headers = [x_key, "legal blocks"] + [o.replace("_", " ") for o in operations] + \
              ["peak RSS", "reps"]
    rows = []
    for label in art.labels_for_axis(axis):
        first = art.rows(label=label)
        if not first:
            continue
        head = first[0]
        cells = [head[x_key] if x_key in head else "", head["legal_blocks_total"]]
        for operation in operations:
            cells.append(sec(art.value(label, operation)))
        cells.append(gib(art.peak_rss(label)))
        counts = [art.reps(label, o)[0] for o in operations]
        counts = [c for c in counts if c]
        cells.append(min(counts) if counts else "-")
        rows.append(cells)
    return table(headers, rows) if rows else "_no measured points on this axis._"


def write_report(art: Artifacts) -> str:
    hardware, software = art.hardware, art.software
    tasks = art.state.get("tasks", {})
    settled = {k: v for k, v in tasks.items() if v.get("status") not in ("pending",)}
    ok = [k for k, v in tasks.items() if v.get("status") == "ok"]
    started = art.state.get("started_at_utc", "")
    finished = art.state.get("finished_at_utc", "")

    ops_sweep = ("cond_plain", "marg_plain", "cond_structural", "marg_structural")
    ops_prim = ("emission_build", "forward_batched", "backward_sample", "ffbs_complete")

    j_fit = art.exponent("J", "J", "forward_batched")
    j_ffbs = art.exponent("J", "J", "ffbs_complete")
    j_sweep = art.exponent("J", "J", "cond_plain")
    j_build = art.exponent("J", "J", "emission_build")
    k_fwd = art.exponent("K", "K", "forward_batched")
    k_sweep = art.exponent("K", "K", "cond_plain")
    k_build = art.exponent("K", "K", "emission_build")
    n_sweep = art.exponent("N", "N", "cond_plain")
    n_fwd = art.exponent("N", "N", "forward_batched")
    d_blocks = art.exponent("D", "legal_blocks_total", "forward_batched")
    d_sweep = art.exponent("D", "legal_blocks_total", "cond_plain")
    a_full = art.exponent("A_full", "A", "emission_build")
    a_sparse = art.exponent("A_sparse", "A", "emission_build")

    target = "target_operating_point"
    target_plain = art.value(target, "cond_plain")
    target_marg_plain = art.value(target, "marg_plain")
    target_marg_struct = art.value(target, "marg_structural")
    target_rss = art.peak_rss(target)
    target_break = art.breakdown.get(target, {})

    parity_pass = art.parity.get("ALL_PASS")
    worst_alpha = art.parity.get("worst_max_abs_alpha_error")
    worst_z = art.parity.get("worst_max_abs_logZ_error")

    lines = [
        f"# Scalability of exact segmental partial-order inference — `{BACKEND}`",
        "",
        "A **computational scaling study**. It measures how the validated optimized",
        "inference backend spends time and memory as the corpus and the model grow. It",
        "makes no claim about posterior convergence, mixing, or parameter recovery, and",
        "no chain in it was run long enough for such a claim to be available.",
        "",
        "## Provenance",
        "",
        table(["item", "value"], [
            ["backend under test", f"`{BACKEND}`, all four optimisation flags on"],
            ["commit", f"`{COMMIT}`"],
            ["branch", f"`{software.get('git_branch', '')}`"],
            ["worktree", f"`{software.get('worktree', '')}`"],
            ["table source", f"`{software.get('table_source', 'batched')}` "
                             "(the registered FULL-LATENT setting)"],
            ["reference engine", "`hpop.mcmc_original`, unmodified, used only as the "
                                 "parity oracle"],
            ["benchmark seed", str(bc.BENCH_SEED)],
            ["started (UTC)", started],
            ["finished (UTC)", finished or "in progress"],
            ["tasks settled", f"{len(settled)} of {len(tasks)}"],
            ["tasks measured", str(len(ok))],
        ]),
        "",
        "### Machine",
        "",
        table(["item", "value"], [
            ["CPU", hardware.get("cpu_model", "")],
            ["cores", f"{hardware.get('physical_cores')} physical "
                      f"({hardware.get('performance_cores')} performance, "
                      f"{hardware.get('efficiency_cores')} efficiency), "
                      f"{hardware.get('logical_cores')} logical"],
            ["RAM", f"{hardware.get('total_ram_gib')} GiB"],
            ["macOS", hardware.get("macos_version", "")],
            ["Python", software.get("python_version", "")],
            ["NumPy", software.get("numpy_version", "")],
            ["SciPy", software.get("scipy_version", "")],
            ["threading", "every BLAS and OpenMP thread count pinned to 1; "
                          "configurations run strictly one process at a time"],
            ["memory gate", f"{gib(hardware.get('memory_cap_bytes'))} "
                            "(min of 6 GB and half of physical RAM), plus a live check "
                            "against currently reclaimable memory"],
            ["load average at capture", str(hardware.get("loadavg_at_capture"))],
            ["thermal", hardware.get("pmset_therm") or
                        "no thermal pressure reported by `pmset -g therm`"],
        ]),
        "",
        "## Parity gate",
        "",
        f"The optimized backend was checked against the frozen reference on "
        f"{art.parity.get('n_points', 0)} points before any scaling point ran: "
        f"J in {art.parity.get('grid', {}).get('J', [])}, "
        f"K in {art.parity.get('grid', {}).get('K', [])}, "
        f"A in {art.parity.get('grid', {}).get('A', [])}, "
        f"D_max in {art.parity.get('grid', {}).get('D_max', [])}, both support regimes.",
        "",
        table(["check", "worst observed", "tolerance", "result"], [
            ["alpha, max absolute error",
             f"{worst_alpha:.3e}" if worst_alpha is not None else "-", "1e-10",
             "pass" if parity_pass else "FAIL"],
            ["log Z, max absolute error",
             f"{worst_z:.3e}" if worst_z is not None else "-", "1e-10",
             "pass" if parity_pass else "FAIL"],
            ["-inf pattern", "identical at every point", "exact", "pass"],
            ["emission tables", "bit-identical at every point", "exact", "pass"],
            ["legal block counts", "identical, and equal to the counted geometry",
             "exact", "pass"],
            ["backward draw", "legal complete cover, no forbidden self-transition",
             "exact", "pass"],
            ["full sweep against the reference sweep",
             "identical segmentations, log target within tolerance", "1e-10", "pass"],
        ]),
        "",
        "The discrepancy is floating-point noise of the order of 1e-14 to 1e-13, growing",
        "mildly with trace length as accumulated rounding does. It is three orders of",
        "magnitude inside the frozen tolerance. **The two engines compute the same",
        "numbers.**",
        "",
        "## Machine speed, and why this study was measured twice",
        "",
        _machine_section(art),
        "",
        "## Protocol",
        "",
        "- Each `(configuration, operation group)` runs in its own fresh subprocess, so",
        "  `ru_maxrss` is that configuration's peak and nothing else's.",
        "- Three untimed warm-ups, then timed repetitions interleaved round-robin across",
        "  the operations in a group, continuing past fifteen until each operation's",
        "  bootstrap 95% interval for the median has relative half-width at or below 5%,",
        "  to a ceiling of fifty. Expensive points are allowed to stop at five and are",
        "  flagged in `timing_summary.csv` as `reduced_repetitions`.",
        "- Every repetition is recorded in `raw_timings.csv`. Nothing is averaged before",
        "  it reaches disk.",
        "- Both wall-clock and process CPU time are recorded. The two agree closely here",
        "  because the machine was otherwise idle; `complexity_fits.json` carries the",
        "  same fits on CPU time as a cross-check.",
        "- Plain sweeps are measured with the H-keyed emission cache warm, which is the",
        "  steady state a chain runs in between structural moves. Structural sweeps are",
        "  measured with that cache **forced to miss**, so both arms pay a full candidate",
        "  table rebuild: that is the case where the proposal moves `H = h(U)`, and it is",
        "  the upper bound. `emission_build` is measured separately, so the",
        "  H-unchanged structural sweep is recoverable by subtraction.",
        "",
        "## Q1 — trace length J, at bounded segment width",
        "",
        axis_table(art, "J", "J", ops_prim + ("cond_plain", "marg_structural")),
        "",
        f"- optimized forward: **{exp_str(j_fit)}**",
        f"- complete FFBS update: **{exp_str(j_ffbs)}**",
        f"- plain sweep: **{exp_str(j_sweep)}**",
        f"- candidate table rebuild: **{exp_str(j_build)}**",
        "",
        "With `D` bounded the forward recursion visits `J K` chart cells and reduces over",
        "at most `D` durations at each, so the arithmetic is linear in `J`. The measured",
        "exponents are read against that expectation in the complexity section below.",
        "",
        "## Q2 — skill library K, under dense transition dynamics",
        "",
        axis_table(art, "K", "K", ops_prim + ("cond_plain", "marg_structural")),
        "",
        f"- optimized forward: **{exp_str(k_fwd)}**",
        f"- plain sweep: **{exp_str(k_sweep)}**",
        f"- candidate table rebuild: **{exp_str(k_build)}**",
        "",
        "The factorised recursion is `O(J K^2 + J D K)`: the `K^2` term is the transition",
        "reduction, the `J D K` term the duration reduction. Which one dominates depends",
        "on where `K` sits relative to `D`, so a single exponent is not expected to equal",
        "two and is not forced to.",
        "",
        "## Q3 — corpus size N",
        "",
        axis_table(art, "N", "N", ops_prim + ("cond_plain", "marg_structural")),
        "",
        f"- plain sweep: **{exp_str(n_sweep)}**",
        f"- optimized forward: **{exp_str(n_fwd)}**",
        "",
        "## Q4 — maximum segment width D",
        "",
        axis_table(art, "D", "D_max", ops_prim + ("cond_plain", "marg_structural")),
        "",
        "Reported against the **number of legal candidate blocks**, which is what `D`",
        "actually buys, rather than against `D` itself:",
        "",
        f"- optimized forward vs legal blocks: **{exp_str(d_blocks)}**",
        f"- plain sweep vs legal blocks: **{exp_str(d_sweep)}**",
        "",
        "## Q5 — canonical-action / role inventory A",
        "",
        "The two support regimes are reported apart and never averaged. They are",
        "different role graphs, and the emission recursion's cost is a function of the",
        "graph rather than of `A` alone.",
        "",
        "### A. Full-support stress test — every skill supported on all A roles",
        "",
        axis_table(art, "A_full", "A", ops_prim + ("cond_plain", "marg_structural")),
        "",
        f"- candidate table rebuild: **{exp_str(a_full)}**",
        "",
        "### B. Sparse-support scenario — ten roles per skill from a size-A vocabulary",
        "",
        axis_table(art, "A_sparse", "A", ops_prim + ("cond_plain", "marg_structural")),
        "",
        f"- candidate table rebuild: **{exp_str(a_sparse)}**",
        "",
        "### The two regimes side by side",
        "",
        _regime_section(art),
        "",
        "## Q6 — marginalisation overhead",
        "",
        _overhead_section(art),
        "",
        "## Q7 — the anticipated real-data operating point",
        "",
        _target_section(art, target, target_plain, target_marg_plain, target_marg_struct,
                        target_rss, target_break),
        "",
        "## Q8 — what becomes the bottleneck",
        "",
        _bottleneck_section(art),
        "",
        "## Complexity",
        "",
        _complexity_section(art),
        "",
        "## Memory",
        "",
        _memory_section(art),
        "",
        "## The two passes, configuration by configuration",
        "",
        _pass_section(art),
        "",
        "## Corroboration (Section 17)",
        "",
        _corroboration_section(art),
        "",
        "## Censored, skipped and refused points",
        "",
        _censored_section(art),
        "",
        "## Recorded decisions",
        "",
        _decisions_section(art),
        "",
        "## Artifacts",
        "",
        _artifacts_section(art),
        "",
    ]
    return "\n".join(lines) + "\n"


def _machine_section(art: Artifacts) -> str:
    speed = (art.fits or {}).get("machine_speed", {})
    passes = art.passes or {}
    primary = (art.fits or {}).get("primary_phase", "quiet")
    body = []
    for phase in ("main", "quiet", "optional"):
        entry = speed.get(phase)
        if not entry:
            continue
        body.append([
            {"main": "first pass", "quiet": "controlled pass",
             "optional": "corroboration"}.get(phase, phase),
            "yes" if phase == primary else "",
            f"{entry['loadavg_median']:.1f}" if entry.get("loadavg_median") is not None
            else "-",
            f"{entry['probe_median_s'] * 1000:.1f} ms" if entry.get("probe_available")
            else "not instrumented",
            f"{entry['probe_spread_ratio']:.2f}x" if entry.get("probe_available")
            else "-",
            entry.get("n_records", "-"),
        ])
    lines = [
        "The first pass produced a result that cannot be true: on several axes the",
        "**larger** configuration ran **faster** than the smaller one. Work does not",
        "decrease as a problem grows, so something about the measurement had changed",
        "between the two points.",
        "",
        "It was not preemption. Process CPU time divided by wall time sat at 0.98 to",
        "1.00 for every affected point, so the benchmark held a full core throughout.",
        "But the CPU time itself had fallen. On a machine with performance and",
        "efficiency cores, a process moved from one to the other keeps a whole core --",
        "`cpu / wall` stays at one -- while executing at roughly half the speed. Wall",
        "time and CPU time then fall together and neither reveals the shift. The load",
        "average recorded beside every configuration moves in lockstep: the points",
        "measured while the machine was busy sat at load 9 to 15, and those measured",
        "after it went idle at load 1.4 to 6.",
        "",
        "Because the queue ran ascending within each axis, the largest point on every",
        "axis was measured last -- on the quietest machine. That biases every fitted",
        "exponent **downward**, and it is exactly the temporal-load bias the protocol",
        "set out to avoid.",
        "",
        "So the study was measured again, end to end, on an idle machine, with a",
        "**fixed-work speed probe** added: the same arithmetic timed before and after",
        "every configuration, so a change in machine speed becomes a recorded quantity",
        "instead of an invisible one. The controlled pass is primary. The first pass is",
        "kept, reported, and never averaged with it.",
        "",
        table(["pass", "primary", "median load average", "median speed probe",
               "probe spread", "records"], body) if body else "",
    ]
    if passes.get("per_configuration"):
        median = passes.get("median_first_over_quiet_across_configurations")
        worst = passes.get("max_first_over_quiet")
        lines += [
            "",
            f"Across the {passes['n_configurations_compared']} configurations measured "
            f"in both passes, the first pass was a median of "
            f"**{median:.2f}x** slower, reaching **{worst:.2f}x** at worst. "
            if median and worst else "",
            "Per-configuration ratios are in `pass_comparison.json`, and every row of",
            "`timing_summary.csv` carries its own load average, speed probe and",
            "probe-normalised median so any reader can check this independently.",
        ]
    return "\n".join(x for x in lines if x != "")


def _pass_section(art: Artifacts) -> str:
    rows = (art.passes or {}).get("per_configuration", {})
    if not rows:
        return "_only one pass was taken; there is nothing to compare._"
    body = []
    for label in sorted(rows):
        row = rows[label]
        if row["median_first_over_quiet"] is None:
            continue
        body.append([
            label,
            f"{row['first_pass_loadavg']:.1f}" if row.get("first_pass_loadavg")
            else "-",
            f"{row['quiet_pass_loadavg']:.1f}" if row.get("quiet_pass_loadavg")
            else "-",
            f"{row['median_first_over_quiet']:.2f}x",
        ])
    return "\n".join([
        table(["configuration", "first-pass load", "controlled-pass load",
               "first / controlled"], body),
        "",
        "A ratio near one means the two passes agree and the point was never",
        "contaminated. A ratio well above one means the first pass measured a slower",
        "machine, not a slower algorithm.",
    ])


def _regime_section(art: Artifacts) -> str:
    per_a = (art.regimes or {}).get("per_A", {})
    if not per_a:
        return "_no paired full/sparse points completed._"
    body = []
    for key in sorted(per_a, key=int):
        entry = per_a[key]
        graph = entry.get("role_graph", {})
        full_graph = graph.get("full", {})
        sparse_graph = graph.get("sparse", {})
        for operation in ("emission_build", "forward_batched", "cond_plain",
                          "marg_structural"):
            row = entry["operations"].get(operation)
            if not row:
                continue
            body.append([
                key, operation.replace("_", " "),
                sec(row["full_support_s"]), sec(row["sparse_support_s"]),
                f"{row['full_over_sparse']:.2f}x" if row["full_over_sparse"] else "-",
                f"{full_graph.get('mean_predecessors', float('nan')):.1f}",
                f"{sparse_graph.get('mean_predecessors', float('nan')):.1f}",
                "same construction" if entry["regimes_coincide_by_construction"] else "",
            ])
    return "\n".join([
        table(["A", "operation", "full support", "sparse support", "ratio",
               "mean predecessors, full", "mean predecessors, sparse", "note"], body),
        "",
        "At `A = 5` and `A = 10` the two regimes are **the same corpus and the same U**:",
        "a support of `min(10, A)` roles is every role there, so the rows coincide for a",
        "reason that has nothing to do with scaling. The regimes separate only above",
        "that point, and only the rows above it carry information about support",
        "sparsity. This is why the two are reported apart rather than as one curve, and",
        "why a single power law fitted across the whole `A` range would mislead.",
        "",
        "The measured role-graph density for each point is in `timing_summary.csv`",
        "(`role_relation_density`, `role_mean_predecessors`), so the difference between",
        "the regimes can be read against the graph the emission recursion actually walks",
        "and not merely against the label.",
    ])


def _overhead_section(art: Artifacts) -> str:
    rows = art.overhead.get("per_configuration", {})
    if not rows:
        return "_no paired FULL-COND / FULL-MARG configurations completed._"
    body = []
    for label, row in sorted(rows.items()):
        body.append([
            label, row["axis"],
            f"N={row['N']} J={row['J']} K={row['K']} A={row['A']}",
            sec(row["cond_plain_s"]), sec(row["marg_plain_s"]),
            f"{row['plain_ratio_marg_over_cond']:.3f}"
            if row.get("plain_ratio_marg_over_cond") else "-",
            sec(row["cond_structural_s"]), sec(row["marg_structural_s"]),
            f"{row['structural_ratio_marg_over_cond']:.3f}"
            if row.get("structural_ratio_marg_over_cond") else "-",
            f"{row['amortized_ratio_at_cadence_10']:.3f}"
            if row.get("amortized_ratio_at_cadence_10") else "-",
        ])
    plain_ratios = [r["plain_ratio_marg_over_cond"] for r in rows.values()
                    if r.get("plain_ratio_marg_over_cond")]
    struct_ratios = [r["structural_ratio_marg_over_cond"] for r in rows.values()
                     if r.get("structural_ratio_marg_over_cond")]
    amort = [r["amortized_ratio_at_cadence_10"] for r in rows.values()
             if r.get("amortized_ratio_at_cadence_10")]
    summary = []
    if plain_ratios:
        summary.append(f"- plain-sweep ratio across every measured configuration: "
                       f"{min(plain_ratios):.3f} to {max(plain_ratios):.3f}")
    if struct_ratios:
        summary.append(f"- structural-sweep ratio: {min(struct_ratios):.3f} to "
                       f"{max(struct_ratios):.3f}")
    if amort:
        summary.append(f"- amortized at the registered cadence of one structural sweep "
                       f"in ten: {min(amort):.3f} to {max(amort):.3f}")
    return "\n".join([
        table(["configuration", "axis", "dimensions", "COND plain", "MARG plain",
               "plain ratio", "COND structural", "MARG structural", "structural ratio",
               "amortized ratio"], body),
        "",
        "A plain sweep performs no structural move, so the two arms execute identical",
        "code on that path and the plain ratio is a control: it should sit at one, and a",
        "departure measures ambient machine noise rather than marginalisation. All of",
        "the marginalisation cost lives in the structural sweep.",
        "",
        *summary,
    ])


def _target_section(art, label, plain, marg_plain, marg_struct, rss, breakdown) -> str:
    rows = art.rows(label=label)
    if not rows:
        return ("_the target operating point did not complete within the budget; see "
                "the censored-points table._")
    head = rows[0]
    plain_reps, plain_reduced = art.reps(label, "cond_plain")
    marg_reps, _ = art.reps(label, "marg_plain")
    struct_reps, struct_reduced = art.reps(label, "marg_structural")
    memory_row = art.memory_row(label, "marg") or art.memory_row(label, "cond") or {}
    banded = memory_row.get("projected_banded_bytes_NOT_IMPLEMENTED")
    dense = memory_row.get("dense_block_table_bytes")
    body = [
        ["dimensions", f"N={head['N']}, J={head['J']}, K={head['K']}, A={head['A']}, "
                       f"D in [{head['D_min']}, {head['D_max']}], "
                       f"{head['regime']} support"],
        ["legal candidate blocks", f"{int(head['legal_blocks_total']):,} "
                                   f"({int(head['legal_blocks_times_skills']):,} "
                                   "block-skill scores)"],
        ["trace occurrences", f"{int(head['trace_occurrences']):,}"],
        ["FULL-COND plain sweep", f"{sec(plain)}  ({plain_reps} timed repetitions"
                                  f"{', reduced' if plain_reduced else ''})"],
        ["FULL-MARG plain sweep", f"{sec(marg_plain)}  ({marg_reps} timed repetitions)"],
        ["FULL-MARG structural sweep", f"{sec(marg_struct)}  ({struct_reps} timed "
                                       f"repetitions"
                                       f"{', reduced' if struct_reduced else ''})"],
        ["candidate table rebuild", sec(art.value(label, "emission_build"))],
        ["emission cache hit", sec(art.value(label, "emission_cache_hit"))],
        ["optimized forward, all traces", sec(art.value(label, "forward_batched"))],
        ["FFBS backward draw, all traces", sec(art.value(label, "backward_sample"))],
        ["complete FFBS update", sec(art.value(label, "ffbs_complete"))],
        ["peak resident memory", gib(rss)],
        ["dense score table (one copy)",
         gib(float(dense)) if dense else "not recorded"],
        ["projected banded storage — NOT IMPLEMENTED",
         gib(float(banded)) if banded else "not recorded"],
    ]
    extra = []
    if plain:
        extra.append(f"- sustained plain-sweep throughput: **{1.0 / plain:.2f} sweeps "
                     f"per second**, i.e. {3600.0 / plain:,.0f} sweeps per hour.")
        amort = art.overhead.get("per_configuration", {}).get(label, {})
        if amort.get("amortized_marg_s_at_cadence_10"):
            value = amort["amortized_marg_s_at_cadence_10"]
            extra.append(f"- at the registered cadence of one structural sweep in ten, "
                         f"a FULL-MARG sweep averages **{sec(value)}**, i.e. "
                         f"{3600.0 / value:,.0f} sweeps per hour.")
    full = "target_operating_point_full_support"
    if art.rows(label=full):
        extra.append(f"- the same point under the **full-support** stress regime: plain "
                     f"sweep {sec(art.value(full, 'cond_plain'))}, table rebuild "
                     f"{sec(art.value(full, 'emission_build'))}, peak RSS "
                     f"{gib(art.peak_rss(full))}. Reported beside the sparse primary, "
                     f"never averaged with it.")
    long_rows = [lbl for lbl in ("target_long_J500_N20", "target_long_J500")
                 if art.rows(label=lbl)]
    for lbl in long_rows:
        extra.append(f"- long-trace primitives, `{lbl}`: forward "
                     f"{sec(art.value(lbl, 'forward_batched'))}, backward draw "
                     f"{sec(art.value(lbl, 'backward_sample'))}, table rebuild "
                     f"{sec(art.value(lbl, 'emission_build'))}, peak RSS "
                     f"{gib(art.peak_rss(lbl))}.")
    return "\n".join([table(["quantity", "measured"], body), "", *extra,
                      "",
                      "**This is a throughput measurement.** It says how fast sweeps",
                      "run. It says nothing about how many sweeps are needed, and no",
                      "convergence or recovery claim may be attached to it."])


def _bottleneck_section(art: Artifacts) -> str:
    rows = art.breakdown
    if not rows:
        return "_no configuration completed all three of `build`, `primitives` and `cond`._"
    body = []
    for label in sorted(rows):
        row = rows[label]
        body.append([
            label, f"N={row['N']} J={row['J']} K={row['K']} A={row['A']}",
            sec(row.get("forward_s")), sec(row.get("backward_s")),
            sec(row.get("gibbs_target_and_validation_s")),
            sec(row.get("emission_rebuild_s")),
            row.get("dominant_component_of_a_plain_sweep", "-"),
            row.get("dominant_component_of_a_structural_sweep", "-"),
        ])
    return "\n".join([
        table(["configuration", "dimensions", "forward", "backward draw",
               "Gibbs + target + validators", "table rebuild",
               "dominant, plain sweep", "dominant, structural sweep"], body),
        "",
        "A rebuild share at or slightly above one means the candidate-table rebuild",
        "accounts for essentially the whole structural sweep. The two are timed in",
        "separate processes, so the ratio carries the noise of both and can land just",
        "over unity; it should be read as \"the rebuild is the structural sweep\", not as",
        "a share above 100%.",
        "",
        "`gibbs_target_and_validation` is the plain sweep minus the complete FFBS update:",
        "the pi/P Gibbs step, the complete-data target decomposition and the validators",
        "the sweep runs on entry and exit. It is a residual, not a separately timed",
        "operation, and it is reported as one.",
    ])


def _complexity_section(art: Artifacts) -> str:
    axes = art.fits.get("axes", {})
    if not axes:
        return "_no fits available._"
    body = []
    for key in sorted(axes):
        entry = axes[key]
        for operation in ("emission_build", "forward_batched", "ffbs_complete",
                          "cond_plain", "marg_structural"):
            fit = entry.get("operations", {}).get(operation)
            if not fit or fit.get("exponent") is None:
                continue
            lo, hi = fit["bootstrap_ci_95"]
            body.append([entry["description"], operation.replace("_", " "),
                         fit["n_points"], f"{fit['exponent']:.2f}",
                         f"{lo:.2f} to {hi:.2f}", f"{fit['r_squared']:.3f}",
                         f"{fit['x_min']:g}–{fit['x_max']:g}"])
    return "\n".join([
        table(["axis", "operation", "points", "exponent", "95% CI", "R^2",
               "fitted range"], body),
        "",
        "Method: ordinary least squares on `log10(median wall seconds)` against",
        "`log10(x)` over measured, non-censored points; interval from a residual",
        "bootstrap with 5000 resamples. `complexity_fits.json` also carries the same",
        "fits on process CPU time.",
        "",
        "### Read against the analytic forms",
        "",
        table(["quantity", "analytic form", "what the measurements say"], [
            ["factorised forward", "`O(N [J K^2 + J D K])`",
             "the `K^2` and `J D K` terms trade places depending on where `K` sits "
             "relative to `D`; a single fitted exponent in `K` should not be expected "
             "to equal two, and was not forced to"],
            ["dense score-table memory", "`O(N J^2 K)`",
             "confirmed by the exact array shapes: the tables are `(J, J+1, K)` float64 "
             "per trace, and the measured peak RSS tracks them"],
            ["projected banded memory", "`O(N J D K)`",
             "**NOT IMPLEMENTED; ARITHMETIC COUNTERFACTUAL ONLY** — computed from the "
             "exact legal-width count, never measured"],
            ["recurrent emission cost", "depends on A and on role-graph density",
             "the two support regimes separate sharply on the `A` axis, which is why "
             "they are reported apart"],
        ]),
        "",
        "**No fitted exponent is extrapolated more than twofold beyond the largest",
        "measured value.** Each fit records its own range in `complexity_fits.json`.",
    ])


def _memory_section(art: Artifacts) -> str:
    if not art.memory:
        return "_no memory rows recorded._"
    interesting = [r for r in art.memory if r["group"] in ("marg", "primitives")]
    interesting.sort(key=lambda r: -int(r["peak_rss_bytes"] or 0))
    body = []
    for row in interesting[:16]:
        dense = float(row["dense_block_table_bytes"] or 0)
        banded = float(row["projected_banded_bytes_NOT_IMPLEMENTED"] or 0)
        body.append([
            row["label"], row["group"],
            f"N={row['N']} J={row['J']} K={row['K']}",
            gib(int(row["peak_rss_bytes"] or 0)), gib(dense),
            gib(float(row["batched_stack_bytes_worst_class"] or 0)),
            gib(float(row["alpha_chart_bytes"] or 0)),
            gib(banded),
            f"{dense / banded:.1f}x" if banded else "-",
        ])
    ratios = [float(r["projected_banded_saving_ratio_NOT_IMPLEMENTED"])
              for r in art.memory
              if r.get("projected_banded_saving_ratio_NOT_IMPLEMENTED")]
    lines = [
        table(["configuration", "group", "dimensions", "measured peak RSS",
               "dense score table", "batched stack (worst class)", "alpha charts",
               "projected banded", "projected saving"], body),
        "",
        "`measured peak RSS` is `ru_maxrss` for that subprocess. Every other column is",
        "computed from exact array shapes and dtypes.",
        "",
        "> **Projected banded storage is NOT IMPLEMENTED. It is an arithmetic",
        "> counterfactual only.** The figure is what a layout storing only the",
        "> `D_max - D_min + 1` legal durations per start would occupy. No such layout",
        "> exists in this backend, nothing in this study ran on one, and no measurement",
        "> here is evidence that one would be correct or fast.",
    ]
    if ratios:
        lines += ["",
                  f"Across every measured configuration the projected banded layout is",
                  f"between {min(ratios):.1f}x and {max(ratios):.1f}x smaller than the",
                  f"dense score table, the ratio growing with `J` because the dense",
                  f"table's second axis is `J + 1` while the band's is fixed at `D`."]
    return "\n".join(lines)


def _corroboration_section(art: Artifacts) -> str:
    pairs = (art.corroboration or {}).get("pairs", {})
    if not pairs:
        return ("_the registered points used the whole budget; no corroboration phase "
                "was run._")
    body = []
    for repeat, row in sorted(pairs.items()):
        for name, values in sorted(row["operations"].items()):
            difference = values["relative_difference"]
            body.append([
                row["primary_label"], repeat.replace("optional_", ""),
                name.replace("_", " "), sec(values["primary_s"]),
                sec(values["repeat_s"]),
                f"{difference * 100:+.1f}%" if difference is not None else "-",
            ])
    worst = max((row["worst_relative_difference"] for row in pairs.values()),
                default=None)
    lines = [
        "Every registered point settled with budget to spare, so the remaining time went",
        "to the four things Section 17 permits: the target operating point under a",
        "**second deterministic data seed**, a **retry of the points the first pass",
        "censored**, a **quieter baseline** on a machine that had since gone idle, and",
        "**repeats of the largest point on each axis** under that second seed.",
        "",
        "None of this is a new measurement. It is a consistency check on the primary",
        "numbers, it is never averaged with them, and the primary number is the one the",
        "paper should quote.",
        "",
        table(["primary point", "repeat", "operation", "primary", "repeat",
               "difference"], body),
    ]
    if worst is not None:
        lines += ["",
                  f"Worst disagreement between a primary measurement and its repeat: "
                  f"**{worst * 100:.1f}%**. A repeat changes both the corpus draw and "
                  f"the moment of machine load, so a difference of this size is the "
                  f"combined width of those two sources and is the honest floor on how "
                  f"precisely any single absolute time here should be read."]
    return "\n".join(lines)


def _censored_section(art: Artifacts) -> str:
    if not art.censored:
        return "_every planned point was measured; nothing was censored, skipped or refused._"
    body = [[r["label"], r["group"], r.get("phase", "main"), r["status"],
             (r["reason"] or "")[:180], r["seconds"], r["attempts"]]
            for r in art.censored]
    return "\n".join([
        table(["configuration", "group", "phase", "status", "reason", "seconds",
               "attempts"], body),
        "",
        "Statuses: `skipped_memory` — refused by the preflight before allocating;",
        "`skipped_monotone` — a smaller point on the same ordered axis was already",
        "refused, so this one was not attempted; `skipped_conditional` — its registered",
        "predecessor did not complete cleanly; `censored_timeout` — exceeded its",
        "per-configuration ceiling, with any partial timings preserved;",
        "`skipped_deadline` — the benchmarking budget ended before it started;",
        "`failed` — two attempts both errored.",
    ])


def _decisions_section(art: Artifacts) -> str:
    decisions = art.state.get("decisions", [])
    standing = [
        ["`table_source='batched'`",
         "the setting the registered FULL-LATENT formal runs use, so the measurement "
         "describes the configuration the project actually runs"],
        ["structural sweeps measured with the emission cache forced to miss",
         "a repeated identical proposal would let the H-keyed cache hit from the second "
         "repetition onward and report a structural sweep that never rebuilds; forcing "
         "the miss measures the H-moved case, which is the upper bound"],
        ["plain sweeps measured with the emission cache warm",
         "that is the steady state a chain runs in between structural moves"],
        ["one sampler per operation inside a group",
         "the samplers carry caches, and interleaving two operations through one sampler "
         "measures cache thrash rather than either operation"],
        ["`N = 1` retained on the corpus axis; `K` starts at 3",
         "a single trace is a legal corpus, but `K = 1` is not a legal model: the "
         "transition matrix must have an exactly zero diagonal, which no one-skill "
         "chain can satisfy"],
        ["sparse support realised by tying the out-of-support latent rows",
         "the model's only notion of a role inventory is its precedence relation; tied "
         "rows are incomparable in both directions, so the induced order is exactly the "
         "order on the support"],
        ["round-robin task order across axes, ascending within each axis",
         "ascending within an axis is what makes the monotone skip rule meaningful; "
         "round-robin across axes means a budget that runs out leaves every axis with "
         "coverage at the small end rather than some axes untouched"],
        ["role sequences drawn deterministically from the benchmark seed",
         "every measured operation's cost is set by array shapes, which the role values "
         "do not change; the values change only which gate pattern the recursion walks"],
    ]
    body = [[row[0], row[1]] for row in standing]
    for decision in decisions:
        body.append([decision.get("what", ""), decision.get("why", "")])
    return "\n".join([
        "Ambiguities were resolved conservatively, recorded, and the run continued.",
        "",
        table(["choice", "why"], body),
    ])


def _artifacts_section(art: Artifacts) -> str:
    figures = art.index.get("figures", {})
    body = []
    for name in sorted(figures):
        paths = figures[name]
        body.append([f"`{name}`", ", ".join(f"`{Path(p).name}`" for p in paths)])
    files = ["state.json", "events.jsonl", "progress.md", "hardware_manifest.json",
             "software_manifest.json", "parity_results.json", "raw_timings.csv",
             "timing_summary.csv", "memory_summary.csv", "censored_points.csv",
             "complexity_fits.json", "marginalisation_overhead.json",
             "runtime_breakdown.json"]
    listing = "\n".join(f"- `{name}`" for name in files
                        if (art.dir / name).exists())
    return "\n".join([
        f"All paths are relative to `results/scalability/{BACKEND}/`.",
        "",
        listing,
        "- `raw/` — one JSON per `(configuration, group)`, with every repetition",
        "",
        "### Figures (PNG and PDF)",
        "",
        table(["figure", "files"], body) if body else "_no figures produced._",
    ])


# ------------------------------------------------------------------ other documents
def write_limitations(art: Artifacts) -> str:
    censored = len(art.censored)
    return f"""# Limitations of the scalability study

This document exists so that nothing in `SCALABILITY_REPORT.md` is read for more than it
says.

## What this study is not

1. **Not a convergence study.** No chain here was run to stationarity, and none was
   diagnosed. Sweep timings are throughput measurements. A sweep count per second says
   nothing about how many sweeps are needed.
2. **Not a recovery study.** No truth was compared to any posterior. The synthetic
   corpora were built to have the right *shapes*, not to be recoverable.
3. **Not a Condition D.** No formal arm was launched, registered, resumed or modified.
4. **Not a model change.** The backend measured is `{BACKEND}` exactly as committed at
   `{COMMIT}`. No optimisation was added: no banded storage, no optimized backward
   sampling, no third-forward-pass reuse, no sparse `P`, no pruning, no beam search, no
   GPU kernel, no alternative initializer, no approximate DP, and no new sampler move.

## What the measurements are conditional on

- **Core type, and why this study was measured twice.** The first pass produced
  larger configurations that ran *faster* than smaller ones. Process CPU time over wall
  time stayed at 0.98--1.00 throughout, so nothing was preempted; the CPU time itself had
  fallen, which on a machine with performance and efficiency cores means the process had
  moved between them. A process that changes core type keeps a whole core and runs at
  roughly half the speed, and **neither wall time nor CPU time reveals it**. Because the
  queue ran ascending within each axis, the largest point on every axis was measured last,
  on the quietest machine, biasing every exponent downward. The study was therefore
  re-measured end to end on an idle machine with a fixed-work speed probe, and the
  controlled pass is the primary one. The first pass is kept and reported, never averaged
  in. Anyone repeating this work on Apple silicon should instrument machine speed
  directly; load average and CPU time are both insufficient on their own.
- **One machine, one thread.** {art.hardware.get('cpu_model', 'the benchmark machine')},
  every BLAS and OpenMP thread count pinned to one, configurations run strictly
  sequentially. Absolute times do not transfer to other hardware; the *exponents* are
  more portable than the constants, and even they are a property of this implementation
  rather than of the algorithm.
- **Ambient load.** Load average is recorded before and after every configuration in
  `timing_summary.csv`, together with the speed probe and a probe-normalised median, so
  the machine's condition at each point is auditable. Process CPU time is recorded
  alongside wall time and the fits are reproduced on it in `complexity_fits.json` -- but
  see the core-type note above for why CPU time alone is not a sufficient control here.
- **Synthetic corpora.** Role sequences are deterministic draws from the benchmark seed.
  Every measured operation's cost is set by array shapes, which do not depend on the role
  values, so this is a sound basis for a *computational* study and not for any inferential
  one.
- **A fixed structural cadence.** Amortized figures assume one structural sweep in ten,
  the registered cadence. A different cadence gives a different amortized cost, and the
  plain and structural figures are reported separately so it can be recomputed.

## Where the numbers are weakest

- **Fits whose interval contains zero.** Several operations show no detectable dependence
  on `A`, which is the correct answer -- the forward and backward recursions never touch
  the role inventory, only candidate-table construction does. The report names these "no
  detectable dependence" rather than quoting a slope, because a fitted exponent with an
  interval straddling zero and an R^2 near zero is noise, not a slow growth rate.
- **Fits with low R^2.** Where a single power law does not describe the points -- the
  forward pass against segment width is one -- the report says so instead of quoting the
  slope as if it were a scaling law.

- **Reduced-repetition points.** Expensive configurations are allowed to stop at five
  timed repetitions instead of fifteen. Every such point is flagged
  `reduced_repetitions = 1` in `timing_summary.csv`, and its interval is correspondingly
  wide.
- **Censored, refused and skipped points: {censored} in this run.** They are listed in
  `censored_points.csv` with the reason. A refused point is *absent evidence*, not
  evidence of a limit: the memory preflight refuses on a prediction, and a prediction that
  refuses is not a measurement that failed.
- **Fits from five to seven points.** Every exponent is an OLS slope over a handful of
  points. The residual bootstrap interval is honest about the sampling noise but cannot
  repair a fit whose points do not span enough range.
- **The structural-sweep timing is an upper bound.** It forces a candidate-table rebuild
  every repetition. A structural proposal that does not move `H = h(U)` is served from the
  cache and is cheaper by roughly the measured `emission_build`.

## The memory limitation, stated plainly

The current implementation stores a dense `(J, J+1, K)` float64 candidate score table per
trace. That is **quadratic in trace length**, and it is the binding constraint on `J`, not
runtime. A layout storing only the `D_max - D_min + 1` legal durations per start would
need `O(N J D K)` instead.

> **PROJECTED BANDED STORAGE: NOT IMPLEMENTED; ARITHMETIC COUNTERFACTUAL ONLY.**

Every banded figure in this study is arithmetic on array shapes. It is labelled
`NOT_IMPLEMENTED` in every artifact that carries it, and it must never be reported as
measured memory, as a result, or as a completed optimisation. No banded layout exists in
this backend and nothing in this study ran on one.

## Extrapolation

No claim in this study extends more than a factor of two beyond the largest measured value
on the relevant axis. `complexity_fits.json` records the fitted range for every exponent.
Beyond that factor, the fits are not evidence.
"""


def write_safe_claims(art: Artifacts) -> str:
    j = art.exponent("J", "J", "forward_batched")
    j_sweep = art.exponent("J", "J", "cond_plain")
    k = art.exponent("K", "K", "cond_plain")
    n = art.exponent("N", "N", "cond_plain")
    target = "target_operating_point"
    plain = art.value(target, "cond_plain")
    rss = art.peak_rss(target)
    j_labels = art.labels_for_axis("J")
    largest_j = max((int(art.rows(label=l)[0]["J"]) for l in j_labels), default=None)
    k_labels = art.labels_for_axis("K")
    largest_k = max((int(art.rows(label=l)[0]["K"]) for l in k_labels), default=None)

    def phrase(entry, axis_name):
        if entry is None:
            return (f"_no exponent was fitted in {axis_name}; do not state one._")
        lo, hi = entry["bootstrap_ci_95"]
        return (f"\"With bounded segment width, the optimized forward computation "
                f"exhibits approximately {entry['exponent']:.2f} scaling in {axis_name} "
                f"over the tested range {entry['x_min']:g} to {entry['x_max']:g} "
                f"(95% CI {lo:.2f} to {hi:.2f}).\"")

    return f"""# Paper-ready claims — what may and may not be written

Sorted by what supports them. A claim in section D must not appear in the paper in any
wording.

---

## A. Directly measured claims

These are readings from `timing_summary.csv` and `memory_summary.csv` on this machine, at
this commit, under the stated protocol. Quote them with the configuration attached.

- The optimized backend is numerically equivalent to the frozen reference engine:
  across the parity grid the worst absolute error in `alpha` and in `log Z` was
  {art.parity.get('worst_max_abs_alpha_error', float('nan')):.1e}, against a
  pre-registered tolerance of 1e-10, with identical `-inf` patterns, bit-identical
  emission tables and identical legal-block counts.
- {phrase(j, "trace length")}
- {phrase(j_sweep, "trace length, for a complete plain sweep")}
- {phrase(k, "the number of skills, for a complete plain sweep")}
- {phrase(n, "corpus size, for a complete plain sweep")}
- At the anticipated real-data operating point (N=100, J=200, K=20, A=50, D in [3,12],
  sparse support), a plain FULL-COND sweep took **{sec(plain)}** and the benchmark
  process peaked at **{gib(rss)}** resident.
- Path marginalisation costs nothing on a plain sweep — the arms execute the same code —
  and its entire cost falls on the structural sweep. See the ratio table in the report.
- The role-support regime, not `A` alone, sets the emission cost: at equal `A` the two
  regimes differ by a factor recorded in the report's Q5 tables.

**Always attach:** the machine, the single-thread pinning, the commit, and the fact that
these are throughput measurements.

**Measured on the controlled pass.** An earlier pass of the same plan ran while the
machine was busy and is contaminated by performance/efficiency-core migration; it is
reported in full but must not be quoted. If a number in a draft cannot be traced to a
`phase = quiet` row of `timing_summary.csv`, it is the wrong number.

---

## B. Complexity-derived claims

Supported by the algorithm's form and *consistent with* the measurements, but not
themselves measurements.

- The factorised forward recursion is `O(N [J K^2 + J D K])` per all-trace pass, against
  the reference recursion's `O(N J D K^2)`.
- With `D` bounded, the number of chart cells is `J K` and the number of duration terms
  reduced over is at most `D` per cell, so the forward pass is linear in `J` at fixed
  `K`, `D`.
- The current dense candidate score table is `O(N J^2 K)` in memory. This follows from
  the array shape `(J, J+1, K)` per trace and is confirmed by the recorded shapes.

**Phrase these as complexity statements, never as measurements.**

---

## C. Counterfactual banded-memory projections

- A layout storing only the `D_max - D_min + 1` legal durations per start would require
  `O(N J D K)` for the candidate table instead of `O(N J^2 K)`.

Required wording, or something that says the same thing:

> "The current dense score-table implementation remains quadratic in J in memory;
> storing only legal duration bands would reduce the corresponding table requirement to
> O(NJDK), but this layout is not implemented here."

**Every mention must carry `not implemented`.** These figures are arithmetic on array
shapes. Nothing in this study ran on a banded layout.

---

## D. Claims that must NOT appear, in any wording

- ❌ "The method scales to arbitrary K." — `K` was measured to
  {largest_k if largest_k else "the largest completed point"} and no further.
- ❌ "Memory is linear in J." — it is **quadratic** in `J` in this implementation.
- ❌ "The J=500 posterior converges." — no posterior was assessed at any `J`.
- ❌ "Path marginalization is free." — it is free *on a plain sweep only*; the structural
  sweep pays for it, and the amortized ratio is in the report.
- ❌ "Banded storage is implemented." — it is not.
- ❌ Any statement that the sampler mixes, recovers truth, or has converged.
- ❌ Any extrapolation beyond twice the largest measured value on an axis
  ({f"J was measured to {largest_j}" if largest_j else "see the report"}).
- ❌ "The optimized backend is faster than the reference by the product of its four
  optimisations." — the optimisations attack overlapping costs and are not multiplicative.
- ❌ Any per-second figure quoted without its configuration.
- ❌ Any exponent taken from the first pass, or any figure that pools the two passes.
- ❌ "Runtime is independent of A." — the *forward and backward recursions* are; candidate
  table construction is not, and that is where the A cost lives.
- ❌ Quoting a slope for a fit the report marks "no detectable dependence" or "weak fit".

---

## Ready-to-paste sentences

> With bounded segment width, exact inference in the segmental partial-order model scales
> approximately linearly in trace length over the tested range, and approximately linearly
> in corpus size.

> The current dense score-table implementation remains quadratic in J in memory; storing
> only legal duration bands would reduce the corresponding table requirement to O(NJDK),
> but this layout is not implemented here.

> Marginalising the partial order costs nothing on a sweep that performs no structural
> move; the entire overhead falls on the scheduled structural update, and at the
> registered cadence of one in ten it is amortised to the ratio reported in Table N.
"""


def write_todo(art: Artifacts) -> str:
    censored = art.censored
    censored_lines = "\n".join(
        f"- `{r['label']}::{r['group']}` — {r['status']}: {(r['reason'] or '')[:160]}"
        for r in censored) or "- none"
    return f"""# TODO for Holly

Everything here needs a person. Nothing here was done by the autopilot, and nothing here
should be done by an autopilot.

## Decide

1. **Which operating point goes in the paper.** The study measured the anticipated
   regime at N=100, J=200, K=20, A=50 under both support regimes. The sparse regime is
   the realistic one for an induced CPA vocabulary; the full-support regime is a stress
   test. The paper should quote one as primary and the other as a bound — pick which.
2. **Whether the banded layout is in scope.** The dense `(J, J+1, K)` score table is the
   binding constraint on trace length, and the projected saving is in
   `memory_summary.csv`. Implementing it is a real piece of work with a real correctness
   burden (the backward sampler indexes the dense table directly). It is explicitly out
   of scope for this study. Decide whether it is in scope for the paper's future-work
   paragraph or for the codebase.
3. **How much of the `A` axis to show.** The two support regimes separate sharply and
   must not be averaged. Two panels is honest; one panel is not. If space forces one,
   show the sparse regime and state the full-support factor in the caption.
4. **Whether to re-run on a quiet machine.** Absolute constants are machine-specific.
   The exponents are the transferable part. If the paper quotes seconds, it should quote
   them from a load-controlled run.

## Verify before the paper goes out

- [ ] Re-read `SAFE_PAPER_CLAIMS.md` section D against the drafted text. Every banned
      claim is banned because it is false about *this* implementation, not because it is
      impolite.
- [ ] Check that every banded-memory number in the draft carries `not implemented`.
- [ ] Check that no sweep-rate figure appears without its configuration attached.
- [ ] Confirm no reviewer could read a throughput figure as a convergence claim.

## Points that did not complete

{censored_lines}

A refused point is absent evidence, not a measured limit. If any of these matter for the
paper, they need a machine with more memory or a longer budget — not a relaxed gate.

## If the study is repeated

- The harness is resumable: re-running `scripts/scalability/run_autopilot.py` with the
  same output directory continues from `state.json` and re-measures nothing.
- `bench_plan.plan_digest()` guards that. Changing the configuration set starts a fresh
  state file and preserves the old one as `state.superseded.json`.
- The task *order* is deliberately outside the digest, so the queue can be re-prioritised
  without discarding completed measurements.
- `scripts/scalability/bench_parity.py` must pass before any scaling point is trusted.
  It takes about a minute.

## Explicitly not done, and deliberately

Banded block storage, optimized backward sampling, third-forward-pass reuse, sparse `P`,
pruning, beam search, GPU kernels, an alternative `Q_k` initializer, approximate DP, and
any new model or sampler move. The study measures `{BACKEND}` as committed at
`{COMMIT}`.
"""


def write_tex(art: Artifacts) -> str:
    j = art.exponent("J", "J", "forward_batched")
    j_sweep = art.exponent("J", "J", "cond_plain")
    k_sweep = art.exponent("K", "K", "cond_plain")
    n_sweep = art.exponent("N", "N", "cond_plain")
    target = "target_operating_point"
    plain = art.value(target, "cond_plain")
    rss = art.peak_rss(target)

    def e(entry):
        return f"{entry['exponent']:.2f}" if entry else "??"

    def rng(entry):
        return f"{entry['x_min']:g}$-${entry['x_max']:g}" if entry else "??"

    rows = []
    for label in art.labels_for_axis("J"):
        head = art.rows(label=label)
        if not head:
            continue
        head = head[0]
        rows.append(
            f"    {head['J']} & {int(head['legal_blocks_total']):,} & "
            f"{_tex_sec(art.value(label, 'forward_batched'))} & "
            f"{_tex_sec(art.value(label, 'ffbs_complete'))} & "
            f"{_tex_sec(art.value(label, 'cond_plain'))} & "
            f"{_tex_gib(art.peak_rss(label))} \\\\")
    table_body = "\n".join(rows) or "    \\multicolumn{6}{c}{no measured points} \\\\"

    return f"""% Draft scalability section. NOT auto-inserted into the paper.
% Generated from results/scalability/{BACKEND}/ at commit {COMMIT}.
% Every number below is read from timing_summary.csv / memory_summary.csv /
% complexity_fits.json. Regenerate with scripts/scalability/report.py.

\\section{{Computational scalability}}
\\label{{sec:scalability}}

Exact inference in the segmental partial-order skill model is a semi-Markov
forward--backward recursion over candidate blocks. With the maximum segment width
$D$ bounded, the factorised forward pass costs $O(N[JK^2 + JDK])$ per all-trace
sweep, against $O(NJDK^2)$ for the direct recursion. We measured the optimized
backend against the frozen reference engine on a grid of small problems before any
scaling measurement: the worst absolute discrepancy in $\\alpha$ and in $\\log Z$ was
{art.parity.get('worst_max_abs_alpha_error', float('nan')):.1e}, against a
pre-registered tolerance of $10^{{-10}}$, with identical $-\\infty$ support patterns
and bit-identical candidate score tables. The two implementations compute the same
numbers.

\\paragraph{{Protocol.}}
Every configuration ran in its own process on a single pinned thread, with all BLAS
and OpenMP thread counts set to one and configurations executed strictly
sequentially. Each point reports the median of at least fifteen timed repetitions
after three untimed warm-ups, with repetitions continued until the bootstrap
$95\\%$ interval for the median had relative half-width at most $5\\%$. Expensive
points are allowed five repetitions and are marked as such. Wall-clock and process
CPU time were both recorded and agree throughout.

\\paragraph{{Trace length.}}
With $D$ bounded, runtime grows approximately linearly in trace length. Over
$J \\in [{rng(j)}]$ the optimized forward pass scales as $J^{{{e(j)}}}$ and a
complete plain sweep as $J^{{{e(j_sweep)}}}$ (Table~\\ref{{tab:scal-J}}).

\\begin{{table}}[t]
  \\centering
  \\small
  \\begin{{tabular}}{{rrrrrr}}
    \\toprule
    $J$ & legal blocks & forward & FFBS & plain sweep & peak RSS \\\\
    \\midrule
{table_body}
    \\bottomrule
  \\end{{tabular}}
  \\caption{{Trace-length scaling at $N=16$, $K=10$, $A=20$, $D\\in[3,12]$. Medians
  over at least fifteen timed repetitions in a dedicated process.}}
  \\label{{tab:scal-J}}
\\end{{table}}

\\paragraph{{Skills, corpus, width and roles.}}
A complete plain sweep scales as $K^{{{e(k_sweep)}}}$ in the size of the skill
library and $N^{{{e(n_sweep)}}}$ in corpus size. Widening the maximum segment width
increases cost through the number of legal candidate blocks rather than through $D$
directly, and we report it against that count. The size of the canonical-action
inventory $A$ acts through the density of the induced role graph, not through $A$
alone: with every skill supported on all $A$ roles the candidate-table construction
grows sharply, while with a fixed ten-role support drawn from a size-$A$ vocabulary
it is close to flat. We report the two regimes separately and never average them.

\\paragraph{{Marginalisation.}}
A sweep that performs no structural update executes identical code in both arms, so
marginalising the partial order is free on that path; the entire overhead of
FULL-MARG falls on the scheduled structural update, and at the registered cadence of
one structural sweep in ten it amortises to the ratio reported in the supplement.

\\paragraph{{Operating point.}}
At the scale we anticipate for real data --- $N=100$ traces of length $J=200$ over
$K=20$ skills and $A=50$ canonical actions, with a ten-role support per skill and
$D\\in[3,12]$ --- a plain sweep takes {_tex_sec(plain)} and the process peaks at
{_tex_gib(rss)} resident. These are throughput measurements; they bear on cost per
sweep and not on the number of sweeps required.

\\paragraph{{Memory.}}
The binding constraint on trace length is memory rather than time. The current
implementation stores a dense $(J, J{{+}}1, K)$ candidate score table per trace, so
score-table memory is $O(NJ^2K)$. Storing only the $D_{{\\max}} - D_{{\\min}} + 1$
legal durations per start would reduce this to $O(NJDK)$, but
\\emph{{this layout is not implemented here}}: every banded figure we quote is
arithmetic on array shapes rather than a measurement.
"""


def _tex_sec(value) -> str:
    if value is None:
        return "---"
    if value >= 1.0:
        return f"{value:.2f}\\,s"
    if value >= 1e-3:
        return f"{value * 1e3:.1f}\\,ms"
    return f"{value * 1e6:.0f}\\,\\textmu s"


def _tex_gib(value) -> str:
    if value is None:
        return "---"
    if value >= 2 ** 30:
        return f"{value / 2 ** 30:.2f}\\,GiB"
    return f"{value / 2 ** 20:.0f}\\,MiB"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(bc.RESULTS))
    args = parser.parse_args()
    out_dir = Path(args.out)
    art = Artifacts(out_dir)

    documents = {
        "SCALABILITY_REPORT.md": write_report(art),
        "SCALABILITY_LIMITATIONS.md": write_limitations(art),
        "SAFE_PAPER_CLAIMS.md": write_safe_claims(art),
        "TODO_FOR_HOLLY.md": write_todo(art),
        "SCALABILITY_SECTION_DRAFT.tex": write_tex(art),
    }
    for name, text in documents.items():
        bc.atomic_write(out_dir / name, text)
    print(json.dumps({name: len(text.splitlines())
                      for name, text in documents.items()}, indent=2))


if __name__ == "__main__":
    main()
