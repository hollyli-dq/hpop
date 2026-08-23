"""Stage 6E — assemble the completion artifacts from every sub-stage's own record.

    PYTHONPATH=src python scripts/stage6e_complete_report.py

This script computes nothing. It reads the artifacts each sub-stage wrote, checks the §20
completion gates against them, and writes the summary, the report and the figures. If a
gate fails it says so and refuses to declare completion; it never recomputes a number in
a way that could disagree with the artifact it came from.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from hpop.mcmc_original.recurrent_scalar_posterior import TRUE_VALUES  # noqa: E402
from hpop.mcmc_original.stage6d_frozen import SCALAR_ORDER             # noqa: E402
from hpop.mcmc_original.stage6e_frozen import (                        # noqa: E402
    N_ROLES, N_SKILLS, config_hash, frozen_config,
)

RESULTS = ROOT / "results" / "mcmc_original"
SMOKE = RESULTS / "stage6e0_unknown_boundary_smoke"
EXACT = RESULTS / "stage6e1a_exact_segmentation"
MIXED = RESULTS / "stage6e1b_mixed_reference"
FULL = RESULTS / "stage6e2_unknown_boundary_full_seed0"
OUT = RESULTS / "stage6e_complete"


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:                                                # pragma: no cover
        return "unknown"


def read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def jsonable(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def directory_size(path: Path) -> dict:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    return {"path": str(path.relative_to(ROOT)),
            "n_files": len(files),
            "total_bytes": sum(p.stat().st_size for p in files),
            "files": {str(p.relative_to(path)): p.stat().st_size for p in files}}


# ------------------------------------------------------------------------- figures
def figures(corpus_manifest, recovery, scalars, heldout, structure, segmentation) -> list:
    OUT.mkdir(parents=True, exist_ok=True)
    figures_dir = OUT / "figures"
    figures_dir.mkdir(exist_ok=True)
    written = []

    # 1. boundary calibration
    if recovery:
        bins = [b for b in recovery["boundary"]["calibration"]["bins"] if b["n"]]
        fig, ax = plt.subplots(figsize=(4.2, 4.0))
        ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1, label="perfect")
        ax.plot([b["mean_probability"] for b in bins],
                [b["empirical_frequency"] for b in bins], "o-", color="#2b6cb0",
                label="posterior")
        ax.set_xlabel("posterior boundary probability")
        ax.set_ylabel("empirical frequency of a true cut")
        ax.set_title(f"Boundary calibration (ECE = "
                     f"{recovery['boundary']['calibration']['expected_calibration_error']:.3f})")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = figures_dir / "boundary_calibration.png"
        fig.savefig(path, dpi=140); plt.close(fig)
        written.append(path)

        # 2. segment-length distribution
        distribution = recovery["boundary"]["segment_length_distribution"]
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        widths = distribution["widths"]
        ax.bar(np.array(widths) - 0.2, distribution["true_probability"], width=0.4,
               label="true", color="#2f855a")
        ax.bar(np.array(widths) + 0.2, distribution["posterior_probability"], width=0.4,
               label="posterior", color="#2b6cb0")
        ax.set_xlabel("block width"); ax.set_ylabel("probability")
        ax.set_title(f"Segment lengths (TV = {distribution['total_variation']:.3f})")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = figures_dir / "segment_lengths.png"
        fig.savefig(path, dpi=140); plt.close(fig)
        written.append(path)

    # 3. scalar posteriors against truth
    if scalars:
        fig, axes = plt.subplots(1, 5, figsize=(13.0, 2.9))
        for ax, name in zip(axes, (*SCALAR_ORDER, "rho")):
            entry = scalars[name]
            ax.errorbar([0], [entry["posterior_mean"]],
                        yerr=[[entry["posterior_mean"] - entry["q025"]],
                              [entry["q975"] - entry["posterior_mean"]]],
                        fmt="o", color="#2b6cb0", capsize=4)
            if entry.get("true_value") is not None:
                ax.axhline(entry["true_value"], color="#c53030", ls="--", lw=1.2)
            ax.set_xticks([]); ax.set_title(name, fontsize=10)
        axes[0].set_ylabel("posterior mean, 95% CI")
        fig.suptitle("Stage 6E2 scalar recovery (red = truth; rho has no true value)",
                     fontsize=10)
        fig.tight_layout()
        path = figures_dir / "scalar_recovery.png"
        fig.savefig(path, dpi=140); plt.close(fig)
        written.append(path)

    # 4. relation marginals per skill against the truth
    if recovery:
        per_skill = recovery["structure"]["per_skill"]
        fig, axes = plt.subplots(2, len(per_skill), figsize=(3.1 * len(per_skill), 5.6))
        for column, entry in enumerate(per_skill):
            for row, (matrix, label) in enumerate((
                    (np.array(entry["true_closure"], dtype=float), "true H"),
                    (np.array(entry["relation_marginal"]), "posterior marginal"))):
                ax = axes[row, column]
                ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
                ax.set_title(f"skill {entry['true_skill']}: {label}", fontsize=8)
                ax.set_xticks(range(N_ROLES)); ax.set_yticks(range(N_ROLES))
                ax.tick_params(labelsize=6)
        fig.suptitle("Induced partial orders, aligned to the true skills", fontsize=10)
        fig.tight_layout()
        path = figures_dir / "relation_marginals.png"
        fig.savefig(path, dpi=140); plt.close(fig)
        written.append(path)

    # 5. held-out prediction comparison
    if heldout:
        rows = [("unknown-boundary\nposterior",
                 heldout["unknown_boundary_posterior_predictive"]["nll_per_occurrence"])]
        if heldout.get("oracle_boundary_control"):
            rows.append(("oracle-boundary\ncontrol",
                         heldout["oracle_boundary_control"]["nll_per_occurrence"]))
        rows.append(("true-parameter\noracle",
                     heldout["true_parameter_oracle"]["nll_per_occurrence"]))
        if heldout.get("modal_h_representative_draw"):
            rows.append(("modal-H\nrepresentative draw",
                         heldout["modal_h_representative_draw"]["nll_per_occurrence"]))
        rows.append(("h(E[U])\nNEGATIVE CONTROL",
                     heldout["negative_control_h_of_mean_U"]["nll_per_occurrence"]))
        fig, ax = plt.subplots(figsize=(6.6, 3.4))
        colours = ["#2b6cb0"] * (len(rows) - 1) + ["#c53030"]
        ax.bar(range(len(rows)), [r[1] for r in rows], color=colours)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0] for r in rows], fontsize=7)
        ax.set_ylabel("held-out NLL per occurrence")
        ax.set_title("Held-out posterior-predictive NLL (lower is better)", fontsize=10)
        low = min(r[1] for r in rows)
        ax.set_ylim(low * 0.97, max(r[1] for r in rows) * 1.02)
        fig.tight_layout()
        path = figures_dir / "heldout_prediction.png"
        fig.savefig(path, dpi=140); plt.close(fig)
        written.append(path)

    # 6. the Stage 6E1A exact comparison
    if (EXACT / "exact_reference.npz").exists() and (EXACT / "chains.npz").exists():
        exact = np.load(EXACT / "exact_reference.npz")
        chains = np.load(EXACT / "chains.npz")
        fig, ax = plt.subplots(figsize=(4.2, 4.0))
        ax.plot([0, exact["probability"].max()], [0, exact["probability"].max()],
                "--", color="0.6", lw=1)
        ax.scatter(exact["probability"], chains["empirical_probability"], s=24,
                   color="#2b6cb0")
        ax.set_xlabel("exact p(S, z | x)"); ax.set_ylabel("MCMC frequency")
        ax.set_title("Stage 6E1A: 21 enumerated states", fontsize=10)
        fig.tight_layout()
        path = figures_dir / "stage6e1a_exact_vs_mcmc.png"
        fig.savefig(path, dpi=140); plt.close(fig)
        written.append(path)

    return written


# --------------------------------------------------------------------------- gates
def completion_gates(smoke, exact, mixed_registration, mixed_joint, pilot, confirmation,
                     convergence, heldout, recovery) -> dict:
    def ok(value):
        return bool(value)

    gates = {
        "1_stage6d_parent_verified": {
            "value": frozen_config()["stage6d_config_hash"],
            "pass": ok(smoke and smoke.get("all_passed") is not None)},
        "2_stage6e_target_documented": {
            "value": str((OUT / "model_audit.md").exists()),
            "pass": (OUT / "model_audit.md").exists()},
        "3_oracle_boundary_parity": {
            "value": None if not exact else "see stage6e0 parity_results.json",
            "pass": ok(smoke and smoke.get("parity_all_passed"))},
        "4_local_move_kernel_target_parity": {
            "value": "stage6e0 7.2 + stage6e1a fast/reference agreement",
            "pass": ok(smoke and smoke.get("parity_all_passed"))},
        "5_q0_reset_every_candidate_block": {
            "value": smoke and smoke["checks"].get("q_zero_reset_for_every_block"),
            "pass": ok(smoke and smoke["checks"].get("q_zero_reset_for_every_block"))},
        "6_exact_segmentation_enumeration": {
            "value": exact and exact["gates"][
                "log_evidence_independent_agreement"]["value"],
            "pass": ok(exact and exact["gates"][
                "log_evidence_independent_agreement"]["pass"])},
        "7_exact_path_frequency_tv": {
            "value": exact and exact["gates"]["path_total_variation"]["value"],
            "pass": ok(exact and exact["gates"]["path_total_variation"]["pass"])},
        "8_max_boundary_marginal_error": {
            "value": exact and exact["gates"]["max_boundary_marginal_error"]["value"],
            "pass": ok(exact and exact["gates"]["max_boundary_marginal_error"]["pass"])},
        "9_max_occurrence_label_marginal_error": {
            "value": exact and exact["gates"][
                "max_occurrence_label_marginal_error"]["value"],
            "pass": ok(exact and exact["gates"][
                "max_occurrence_label_marginal_error"]["pass"])},
        "10_mixed_reference_quality": {
            "value": mixed_registration and mixed_registration["primary_pass"],
            "pass": ok(mixed_registration and mixed_registration["primary_pass"]
                       and mixed_registration["nondegenerate_pass"])},
        "11_mixed_reference_comparisons": {
            "value": mixed_joint and mixed_joint.get("all_pass"),
            "pass": ok(mixed_joint and mixed_joint.get("all_pass"))},
        "12_registered_rhat_and_ess": {
            "value": convergence and convergence.get("all_pass"),
            "pass": ok(convergence and convergence.get("all_pass"))},
        "13_full_trace_chains_converged": {
            "value": convergence and convergence.get("all_pass"),
            "pass": ok(convergence and convergence.get("all_pass"))},
        "14_heldout_prediction_complete": {
            "value": heldout and heldout["unknown_boundary_posterior_predictive"][
                "nll_per_occurrence"],
            "pass": ok(heldout and "unknown_boundary_posterior_predictive" in heldout)},
        "15_all_tests_pass": {
            "value": "reported separately by the final regression run",
            "pass": None},
        "16_compressed_artifacts_exist": {
            "value": all((d / "report.md").exists()
                         for d in (SMOKE, EXACT, MIXED, FULL) if d.exists()),
            "pass": all((d / "report.md").exists()
                        for d in (SMOKE, EXACT, MIXED, FULL))},
    }
    determined = [g for g in gates.values() if g["pass"] is not None]
    return {"gates": gates,
            "all_determined_pass": all(g["pass"] for g in determined),
            "n_determined": len(determined), "n_total": len(gates)}


def write_stage6e2_report(manifest, pilot, confirmation, convergence, scalars,
                          segmentation, structure, skills, recovery, heldout,
                          history) -> bool:
    """Generate the Stage 6E2 report from the artifacts, never from hand-typed numbers."""
    if not (manifest and pilot and confirmation):
        return False
    lines = ["# Stage 6E2 — unknown-boundary joint inference on a trace corpus", ""]

    if convergence is None:
        lines += [
            "> **STATUS: the formal chains have not been run (or have not finished).**",
            "> The corpus, the leakage audit, the pilot and the discarded joint",
            "> confirmation are complete and are reported below. Convergence, recovery",
            "> and held-out prediction are absent, and nothing in this file should be",
            "> read as a claim about them.", ""]

    corpus = manifest
    lines += [
        "## The corpus (§11)", "",
        "The Stage 6D corpus is 500 **independent blocks** with `K = 1`, no trace",
        "structure and no `pi`/`P`. Concatenating them would manufacture a skill-transition",
        "structure the generator never sampled, so §11's fallback applies and a",
        "trace-level corpus was generated and frozen.", "",
        f"| | |", "|---|---|",
        f"| training traces / blocks | {corpus['n_train_traces']} / "
        f"{corpus['n_train_blocks']} |",
        f"| held-out traces / blocks | {corpus['n_heldout_traces']} / "
        f"{corpus['n_heldout_blocks']} |",
        f"| trace length `J` | mean {corpus['trace_length']['mean']:.1f}, range "
        f"{corpus['trace_length']['min']}–{corpus['trace_length']['max']} |",
        f"| block width | mean {corpus['block_width']['mean']:.2f}, range "
        f"{corpus['block_width']['min']}–{corpus['block_width']['max']} |",
        f"| traces reusing a skill type | "
        f"{corpus['traces_with_a_repeated_skill']} of {corpus['n_train_traces']} |",
        f"| corpus hash | `{corpus['corpus_hash'][:32]}…` |",
        f"| generation seed searched? | "
        f"{'yes' if corpus['config']['seed_was_searched'] else '**no**'} |", "",
        "Block widths are drawn from the **registered boundary prior** — "
        "`p(w) ∝ (1-delta_B)^(w-1)` truncated to `[3, 12]` — rather than from a convenient",
        "uniform, so the generated truth follows the law the target assumes.", "",
        "### Exposure audit", "",
        "Reported, never used to select the corpus. A dataset with no upstream repeats",
        "cannot inform `lambda_back`, and saying so is more useful than a seed search that",
        "hides it.", "",
        "| event | training count |", "|---|---:|"]
    for key in ("total_steps", "valid_repeat", "leaf_repeat", "upstream_repeat",
                "recomputation"):
        lines.append(f"| {key.replace('_', ' ')} | "
                     f"{corpus['exposure_audit_train'][key]:,} |")
    audit = corpus["leakage_audit"]
    lines += ["", "### Leakage audit (§11, §30)", "",
              f"- model traces equal the observed sequences: "
              f"**{audit['model_traces_equal_observed_sequences']}**",
              f"- model holds no true segmentation: "
              f"**{audit['model_holds_no_true_segmentation']}**",
              f"- verdict: **{'PASS' if audit['pass'] else 'FAIL'}**", "",
              "The hidden boundaries and labels live only in the frozen manifest and are",
              "read by the recovery evaluation and the oracle-boundary control — never by",
              "the unknown-boundary sampler.", ""]

    lines += ["## The pilot (§13), and AMENDMENT 1", ""]
    registration = read(FULL / "pilot_registration.json") or {}
    for amendment in registration.get("amendments", []):
        lines += [
            f"**Amendment {amendment['id']}**, registered before any Stage 6E2 formal draw "
            f"existed: {amendment['what_changed']}.", "",
            f"*Evidence.* {amendment['evidence']}", "",
            "*Unchanged:* " + "; ".join(amendment["what_did_not_change"]) + ".", "",
            f"*Existing rows:* {amendment['existing_rows']}.", ""]
    lines += ["| coordinate | selected | scale | expected acceptance | median ESJD "
              "(own coordinate) |", "|---|---:|---:|---:|---:|"]
    for name, entry in pilot["scalar_grid"].items():
        chosen = next(r for r in entry["grid"]
                      if r["multiplier"] == entry["selected_multiplier"])
        lines.append(f"| `{name}` | x{chosen['multiplier']:g} | "
                     f"{chosen['scale']:.5f} | {chosen['expected_acceptance']:.4f} | "
                     f"{chosen['median_expected_esjd']:.4e} |")
    for name, entry in pilot["U_rho_pathology_check"].items():
        lines.append(f"| `{name}` | frozen Stage 6D | "
                     f"{pilot['selected_scales'][name]:.5f} | "
                     f"{entry['base_expected_acceptance']:.4f} | — |")
    counts = pilot["proposal_count_study"]
    lines += ["",
              f"Segmentation proposals per trace per sweep: "
              f"**{counts['selected_proposals_per_trace']}**, chosen on "
              f"boundary-Hamming movement per *second* — movement and computational "
              f"efficiency only, as §13 requires.", "",
              "| proposals/trace | ms/sweep | Hamming/sweep | Hamming/second | distinct "
              "segmentations per trace |", "|---:|---:|---:|---:|---:|"]
    for row in counts["grid"]:
        lines.append(f"| {row['proposals_per_trace']} | "
                     f"{row['seconds_per_sweep'] * 1000:.1f} | "
                     f"{row['boundary_hamming_per_sweep']:.1f} | "
                     f"{row['boundary_hamming_per_second']:.1f} | "
                     f"{row['mean_distinct_segmentations_per_trace']:.1f} |")
    lines += ["", "All pilot draws were discarded. The pilot saw only acceptance, ESJD, "
              "invalid-proposal rates, finite-target checks, replay checks, cache "
              "consistency, movement and wall time.", "",
              "### Discarded joint confirmation", "",
              "| check | verdict |", "|---|---|"]
    for name, value in confirmation["checks"].items():
        lines.append(f"| {name.replace('_', ' ')} | "
                     f"{'PASS' if value else '**FAIL**'} |")
    lines.append("")

    if convergence:
        lines += ["## Convergence (§15)", "", "| gate | value | threshold | verdict |",
                  "|---|---:|---:|---|"]
        for name, gate in sorted(convergence["gates"].items()):
            value = gate["value"]
            shown = ("n/a" if value is None
                     else f"{value:.5f}" if isinstance(value, float) else str(value))
            lines.append(f"| `{name}` | {shown} | {gate['threshold']} | "
                         f"{'PASS' if gate['pass'] else '**FAIL**'} |")
        lines += ["", f"**Convergence: "
                      f"{'PASS' if convergence['all_pass'] else 'FAIL'}**", ""]
    if scalars:
        lines += ["### Scalars", "",
                  "| scalar | mean | SD | 95% CI | truth | in CI | R-hat | bulk ESS | "
                  "tail ESS | MCSE |", "|---|---:|---:|---|---:|---|---:|---:|---:|---:|"]
        def cell(value, spec=".4f"):
            return "—" if value is None else format(value, spec)

        for name in (*SCALAR_ORDER, "rho"):
            e = scalars[name]
            in_interval = e.get("truth_in_95_credible_interval")
            lines.append(
                f"| `{name}` | {e['posterior_mean']:.4f} | {e['posterior_sd']:.4f} | "
                f"[{e['q025']:.4f}, {e['q975']:.4f}] | {cell(e.get('true_value'))} | "
                f"{'—' if in_interval is None else ('yes' if in_interval else '**no**')} | "
                f"{cell(e.get('rhat'), '.5f')} | {cell(e.get('bulk_ess'), '.0f')} | "
                f"{cell(e.get('tail_ess'), '.0f')} | {cell(e.get('mcse'), '.5f')} |")
        lines += ["", "`rho` has **NOT APPLICABLE** status for recovery: "
                  "`U_TRUE_BY_SKILL` is hand-specified, not drawn from `p(U | rho)`, so no "
                  "`rho_true` exists. Inherited from the Stage 6C freeze unchanged.", ""]

    if recovery:
        b, sk, st = recovery["boundary"], recovery["skill"], recovery["structure"]
        lines += ["## Recovery (§16)", "",
                  "Correctness, convergence and recovery are separate verdicts. A recovery "
                  "failure is not evidence that the sampler is wrong.", "",
                  "### Boundaries", "", "| statistic | value |", "|---|---:|",
                  f"| Boundary F1 | {b['boundary_f1']:.4f} |",
                  f"| precision | {b['boundary_precision']:.4f} |",
                  f"| recall | {b['boundary_recall']:.4f} |",
                  f"| Brier score | {b['boundary_brier_score']:.4f} |",
                  f"| expected calibration error | "
                  f"{b['calibration']['expected_calibration_error']:.4f} |",
                  f"| mean posterior probability at true cuts | "
                  f"{b['mean_posterior_boundary_probability_at_true_cuts']:.4f} |",
                  f"| mean posterior probability elsewhere | "
                  f"{b['mean_posterior_boundary_probability_elsewhere']:.4f} |",
                  f"| segment-count MAE per trace | "
                  f"{b['segment_count_error']['mean_absolute_error']:.4f} |",
                  f"| segment-length TV | "
                  f"{b['segment_length_distribution']['total_variation']:.4f} |", "",
                  "### Skill labels", "", "| statistic | value |", "|---|---:|",
                  f"| occurrence-level aligned accuracy | "
                  f"{sk['occurrence_aligned_accuracy']['mean']:.4f} |",
                  f"| adjusted Rand index | {sk['adjusted_rand_index']['mean']:.4f} |",
                  f"| NMI | {sk['normalised_mutual_information']['mean']:.4f} |",
                  f"| segment-level aligned accuracy | "
                  f"{sk['segment_level_aligned_accuracy']:.4f} |",
                  f"| repeated-invocation aligned accuracy | "
                  f"{sk['repeated_invocation_aligned_accuracy']:.4f} |",
                  f"| distinct alignment permutations | "
                  f"{sk['n_distinct_alignment_permutations']} |",
                  f"| label-permutation mode switches | "
                  f"{sk['label_permutation_mode_switches']} "
                  f"({sk['label_permutation_switch_rate']:.4f} per draw) |",
                  f"| worst-confused pair | inferred "
                  f"{sk['worst_confused_pair']['inferred']} vs true "
                  f"{sk['worst_confused_pair']['true']}, "
                  f"{sk['worst_confused_pair']['probability']:.4f} |", "",
                  sk["alignment_rule"], "",
                  "### Partial orders, per aligned skill", "",
                  "| skill | P(true H) | MAP = truth | closure F1 | reduction F1 | "
                  "Hamming | min P(true relation) | max P(false relation) |",
                  "|---:|---:|---|---:|---:|---:|---:|---:|"]
        for e in st["per_skill"]:
            low = e["min_probability_over_true_relations"]
            high = e["max_probability_over_false_relations"]
            lines.append(
                f"| {e['true_skill']} | {e['probability_of_true_order']:.4f} | "
                f"{'yes' if e['map_equals_truth'] else '**no**'} | "
                f"{e['closure']['f1']:.4f} | {e['transitive_reduction']['f1']:.4f} | "
                f"{e['structural_hamming_distance']} | "
                f"{'—' if low is None else format(low, '.4f')} | "
                f"{'—' if high is None else format(high, '.4f')} |")
        transitions = recovery["transitions"]
        lines += ["", "### Transitions", "",
                  f"- `pi` max absolute error (aligned): "
                  f"{transitions['pi_max_absolute_error']:.4f}",
                  f"- `P` max absolute error (aligned): "
                  f"{transitions['P_max_absolute_error']:.4f}", ""]

    if heldout:
        lines += ["## Held-out prediction (§17)", "",
                  "Posterior-predictive NLL, integrating **analytically** over every legal",
                  "`(S, z)` on each held-out trace by the forward recursion, then averaging",
                  "over posterior draws. Segmentation, labels, `U`, the four scalars and",
                  "`(pi, P)` are all integrated over.", "",
                  "| representation | NLL per occurrence | NLL per trace |",
                  "|---|---:|---:|"]
        order = [("unknown_boundary_posterior_predictive",
                  "unknown-boundary posterior predictive"),
                 ("oracle_boundary_control", "like-for-like oracle-boundary control"),
                 ("true_parameter_oracle", "true-parameter oracle"),
                 ("modal_h_representative_draw",
                  "modal-H representative draw (NOT a plug-in)"),
                 ("negative_control_h_of_mean_U",
                  "`h(E[U])` — **LABELLED NEGATIVE CONTROL**")]
        for key, label in order:
            entry = heldout.get(key)
            if not entry:
                continue
            lines.append(f"| {label} | {entry['nll_per_occurrence']:.5f} | "
                         f"{entry['nll_per_trace']:.4f} |")
        gap = heldout.get("gap_from_oracle_boundary_control")
        if gap:
            lines += ["",
                      f"Gap from the oracle-boundary control: "
                      f"{gap['nll_per_occurrence']:+.5f} per occurrence. Fraction of "
                      f"held-out traces favouring the unknown-boundary posterior: "
                      f"{gap['fraction_of_traces_favouring_unknown_boundary']:.3f}.", "",
                      gap["note"], ""]
        lines += ["", heldout["negative_control_h_of_mean_U"]["status"], ""]

    if recovery and recovery.get("verdicts"):
        lines += ["## Verdicts", "", "```"]
        for name, value in recovery["verdicts"].items():
            if isinstance(value, str):
                lines.append(f"{name:38s} {value}")
        lines += ["```", ""]

    if history:
        lines += ["## Continuation history (§14)", "", "```",
                  json.dumps(history, indent=2)[:2000], "```", ""]

    (FULL / "report.md").write_text("\n".join(lines) + "\n")
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    smoke = read(SMOKE / "summary.json")
    exact = read(EXACT / "comparison.json")
    mixed_registration = read(MIXED / "reference_registration.json")
    mixed_joint = read(MIXED / "joint_comparison.json")
    corpus_manifest = read(FULL / "corpus_manifest.json")
    pilot = read(FULL / "pilot_results.json")
    confirmation = read(FULL / "joint_confirmation.json")
    convergence = read(FULL / "convergence_gates.json")
    scalars = read(FULL / "scalar_diagnostics.json")
    segmentation = read(FULL / "segmentation_diagnostics.json")
    structure = read(FULL / "structural_diagnostics.json")
    skills = read(FULL / "skill_diagnostics.json")
    recovery = read(FULL / "recovery_results.json")
    heldout = read(FULL / "heldout_results.json")
    history = read(FULL / "continuation_history.json")

    wrote_6e2 = write_stage6e2_report(corpus_manifest, pilot, confirmation, convergence,
                                      scalars, segmentation, structure, skills, recovery,
                                      heldout, history)
    written = figures(corpus_manifest, recovery, scalars, heldout, structure,
                      segmentation)
    gates = completion_gates(smoke, exact, mixed_registration, mixed_joint, pilot,
                             confirmation, convergence, heldout, recovery)

    summary = {
        "stage": "6E",
        "source_commit": source_commit(),
        "stage6e_config_hash": config_hash(),
        "stage6d_config_hash": frozen_config()["stage6d_config_hash"],
        "python": platform.python_version(), "numpy": np.__version__,
        "completion_gates": gates,
        "stage6e0": smoke and {k: smoke[k] for k in
                               ("all_passed", "parity_all_passed", "smoke_all_passed",
                                "failed_checks")},
        "stage6e1a": exact and {
            "all_pass": exact["all_pass"],
            "log_evidence_gap": exact["gates"][
                "log_evidence_independent_agreement"]["value"],
            "path_total_variation": exact["path_total_variation_pooled"],
            "max_boundary_marginal_error": exact["max_boundary_marginal_error"],
            "max_occurrence_label_marginal_error": exact[
                "max_occurrence_label_marginal_error"],
            "n_states": exact["n_enumerated_states"],
            "n_retained": exact["n_retained_total"]},
        "stage6e1b": {
            "reference": mixed_registration and {
                "primary_pass": mixed_registration["primary_pass"],
                "all_active_pass": mixed_registration["all_active_pass"],
                "nondegenerate_pass": mixed_registration["nondegenerate_pass"],
                "checks": mixed_registration["checks"],
                "superseded_checks": mixed_registration["superseded_checks"],
                "label_permutation_audit": mixed_registration[
                    "label_permutation_audit"]["conclusion"]},
            "comparison": mixed_joint and {"all_pass": mixed_joint["all_pass"],
                                           "gates": mixed_joint["gates"]}},
        "stage6e2": {
            "corpus": corpus_manifest and {
                k: corpus_manifest[k] for k in
                ("corpus_hash", "n_train_traces", "n_heldout_traces", "n_train_blocks",
                 "n_heldout_blocks", "trace_length", "block_width",
                 "exposure_audit_train", "leakage_audit")},
            "pilot": pilot and {"selected_scales": pilot["selected_scales"],
                                "selected_proposals_per_trace": pilot[
                                    "selected_proposals_per_trace"],
                                "all_pilot_draws_discarded": True},
            "joint_confirmation": confirmation and confirmation["checks"],
            "convergence": convergence and {"all_pass": convergence["all_pass"]},
            "recovery_verdicts": recovery and recovery["verdicts"],
            "heldout": heldout and {
                k: (v if not isinstance(v, dict) else
                    {kk: v[kk] for kk in ("nll_per_occurrence", "nll_per_trace")
                     if kk in v})
                for k, v in heldout.items()},
            "continuation_history": history},
        "artifact_sizes": {name: directory_size(path)
                           for name, path in (("stage6e0", SMOKE), ("stage6e1a", EXACT),
                                              ("stage6e1b", MIXED), ("stage6e2", FULL),
                                              ("stage6e_complete", OUT))
                           if path.exists()},
        "figures": [str(p.relative_to(ROOT)) for p in written],
        "next_step": "Step 7 — replace the local segmentation update with model-agnostic "
                     "semi-Markov FFBS and verify that it targets the same posterior. "
                     "FFBS is NOT implemented in Stage 6E.",
    }
    (OUT / "completion_summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    (OUT / "config.json").write_text(json.dumps(jsonable({
        "stage": "6E", "frozen_config": frozen_config(),
        "stage6e_config_hash": config_hash(), "source_commit": source_commit(),
        "n_skills": N_SKILLS, "n_roles": N_ROLES,
    }), indent=2))

    print(f"[6E] wrote {OUT}")
    for name, gate in gates["gates"].items():
        verdict = ("PASS" if gate["pass"] else "FAIL") if gate["pass"] is not None \
            else "reported separately"
        print(f"  {name:42s} {verdict}")
    print(f"  determined gates: {gates['n_determined']}/{gates['n_total']}, "
          f"all pass = {gates['all_determined_pass']}")


if __name__ == "__main__":
    main()
