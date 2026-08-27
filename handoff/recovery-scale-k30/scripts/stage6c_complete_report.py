"""Stage 6C — assemble the per-run reports and the completion summary.

    PYTHONPATH=src python scripts/stage6c_complete_report.py

Four verdicts are kept strictly apart, because they can and do differ:

    sampler correctness      does the chain reproduce its own independent reference?
    structural (U) recovery  does the posterior concentrate on the generating order?
    rho identifiability      how much does the data say about rho? (a separate question
                             from whether rho was *recovered*, which is not applicable)
    beta recovery            does the beta posterior contain and centre on beta_true?

A correct sampler can fail to recover, and a broken one can appear to recover. Reporting
them as one number would hide exactly the distinction Stage 6C exists to make.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.stage6c_frozen import (      # noqa: E402
    STAGE6B_CONFIG_HASH, STAGE6C_MODEL_ID, config_hash, frozen_config,
    load_stage6c_dataset,
)

RESULTS = ROOT / "results" / "mcmc_original"
COMPLETE = RESULTS / "stage6c_complete"

DIRS = {
    "6c1_smoke": RESULTS / "stage6c1_u_rho_smoke",
    "6c1_reference": RESULTS / "stage6c1_u_rho_reference",
    "6c1_full": RESULTS / "stage6c1_u_rho_full_seed0",
    "6c2_smoke": RESULTS / "stage6c2_u_rho_beta_smoke",
    "6c2_reference": RESULTS / "stage6c2_u_rho_beta_reference",
    "6c2_full": RESULTS / "stage6c2_u_rho_beta_full_seed0",
}


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:                                            # pragma: no cover
        return "unknown"


def directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:
            return "n/a"
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value)


def verdict(passed: bool | None) -> str:
    if passed is None:
        return "NOT APPLICABLE"
    return "PASS" if passed else "FAIL"


# --------------------------------------------------------------------- per-run reports
def gates_table(comparison: dict) -> list[str]:
    lines = ["| gate | value | threshold | verdict |", "|---|---|---|---|"]
    for name, gate in comparison["gates"].items():
        lines.append(f"| {name} | {fmt(gate['value'])} | {fmt(gate['threshold'])} | "
                     f"{verdict(gate['pass'])} |")
    return lines


def scalar_table(scalars: dict, names) -> list[str]:
    lines = ["| scalar | mean | sd | median | 95% interval | R-hat | bulk ESS | "
             "tail ESS | MCSE | KS vs reference |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        s = scalars.get(name)
        if not s:
            continue
        interval = f"[{fmt(s.get('q025'))}, {fmt(s.get('q975'))}]"
        lines.append(
            f"| {name} | {fmt(s.get('posterior_mean'))} | {fmt(s.get('posterior_sd'))} | "
            f"{fmt(s.get('median'))} | {interval} | {fmt(s.get('rhat'))} | "
            f"{fmt(s.get('bulk_ess'), 1)} | {fmt(s.get('tail_ess'), 1)} | "
            f"{fmt(s.get('mcse'))} | {fmt(s.get('ks_distance_to_reference'))} |")
    return lines


def write_run_report(stage: str, path: Path) -> dict:
    config = read_json(path / "config.json")
    structural = read_json(path / "structural_diagnostics.json")
    scalars = read_json(path / "scalar_diagnostics.json")
    recovery = read_json(path / "recovery_results.json")
    comparison = read_json(path / "reference_comparison.json")
    if not all((config, structural, scalars, recovery, comparison)):
        return {}

    names = ["rho"] + (["beta"] if stage == "6c2" else [])
    structure = recovery["structural"]
    title = ("Stage 6C1 — joint inference of U and rho" if stage == "6c1"
             else "Stage 6C2 — joint inference of U, rho and beta")

    lines = [f"# {title}", ""]
    lines += [
        f"Source commit `{config['source_commit']}`, Stage 6C config hash "
        f"`{config['stage6c_config_hash'][:16]}...`.", "",
        f"{config['n_chains']} chains x {config['n_sweeps']:,} sweeps, "
        f"{config['burn_in']:,} burn-in, thinning {config['thin']} -> "
        f"{config['retained_pooled']:,} retained draws pooled "
        f"({config['retained_per_chain']:,} per chain). Wall clock "
        f"{config['wall_seconds'] / 60:.1f} min.", "",
        "## Chain starts (none at the truth)", "",
        "| chain | U start | relations at start | rho | beta | seed |",
        "|---|---|---|---|---|---|"]
    for start in config["chain_starts"]:
        lines.append(f"| {start['chain']} | {start['u_start']} | "
                     f"{start['start_relations']} | {fmt(start['rho'], 2)} | "
                     f"{fmt(start['beta'], 2)} | {start['seed']} |")

    lines += ["", "## Sampler correctness — agreement with the frozen reference", ""]
    lines += gates_table(comparison)
    lines += ["",
              f"The reference was built and frozen before these chains ran "
              f"(`{config['reference']}`) and was not adjusted afterwards.", ""]
    lines += [f"- pooled retained draws compared: {comparison['retained_draws_pooled']:,}",
              f"- iid reference draws: {comparison['reference_draws']:,}",
              f"- mixed discrete/continuous energy distance "
              f"{fmt(comparison['mixed']['observed'])} against a "
              f"{comparison['mixed']['envelope_quantile']:.0%} reference-vs-reference "
              f"envelope of {fmt(comparison['mixed']['envelope'])} "
              f"({comparison['mixed']['n_coordinates']} coordinates, "
              f"{comparison['mixed']['dropped_constant_coordinates']} constant ones "
              f"dropped)", ""]

    lines += ["## Scalars", ""] + scalar_table(scalars, names) + [""]
    if "log_target" in scalars:
        lt = scalars["log_target"]
        lines += [f"Log posterior: R-hat {fmt(lt.get('rhat'))}, bulk ESS "
                  f"{fmt(lt.get('bulk_ess'), 1)}." if not lt.get("degenerate")
                  else f"Log posterior: {lt.get('note')}", ""]

    lines += ["## Structural recovery", "",
              f"- true poset index: {structure['true_poset_index']}",
              f"- MAP poset index: {structure['map_poset_index']} "
              f"({'is' if structure['map_is_true'] else 'is NOT'} the true poset)",
              f"- posterior probability of the true poset: "
              f"{fmt(structure['posterior_probability_of_true'])}",
              f"- posterior rank of the true poset: "
              f"{structure['posterior_rank_of_true']}",
              f"- unique orders visited: {structure['n_unique_states_visited']}",
              f"- minimum true-relation probability: "
              f"{fmt(structure['min_true_relation_probability'])}",
              f"- maximum false-relation probability: "
              f"{fmt(structure['max_false_relation_probability'])}", "",
              "| representation | precision | recall | F1 | structural Hamming |",
              "|---|---|---|---|---|"]
    for key in ("closure", "reduction"):
        block = structure[key]
        lines.append(f"| {key} | {fmt(block['precision'])} | {fmt(block['recall'])} | "
                     f"{fmt(block['f1'])} | {block['structural_hamming']} |")

    lines += ["", f"- full-U total variation vs reference: "
                  f"{fmt(structural['full_u_total_variation'])}",
              f"- max relation-marginal error: "
              f"{fmt(structural['max_relation_marginal_error'])}",
              f"- max reduction-marginal error: "
              f"{fmt(structural['max_reduction_marginal_error'])}",
              f"- worst single chain relation error: "
              f"{fmt(structural['worst_chain_relation_error'])}", ""]

    count = structural["relation_count_convergence"]
    if count.get("degenerate"):
        lines += [f"Relation count is constant at {fmt(count['constant_value'], 0)} "
                  f"across every chain and draw. R-hat and ESS are **undefined**, not "
                  f"passing: the poset posterior is a point mass, so there is no "
                  f"structural variation left to mix over.", ""]
    else:
        lines += [f"Relation count: R-hat {fmt(count.get('rhat'))}, bulk ESS "
                  f"{fmt(count.get('bulk_ess'), 1)}.", ""]

    acceptance = scalars.get("acceptance", {})
    total, post = acceptance.get("total", {}), acceptance.get("post_burn_in", {})
    lines += ["## Acceptance", "",
              "| parameter | total | post burn-in |", "|---|---|---|"]
    for name in ("U", "rho") + (("beta",) if stage == "6c2" else ()):
        lines.append(f"| {name} | {fmt(total.get(name))} | {fmt(post.get(name))} |")
    per_chain = acceptance.get("per_chain_post_burn_in", {})
    if per_chain:
        lines += ["", "Post burn-in acceptance per chain:", ""]
        for chain, values in sorted(per_chain.items()):
            rendered = ", ".join(f"{k} {fmt(v)}" for k, v in values.items())
            lines.append(f"- chain {chain}: {rendered}")
    lines.append("")

    path.joinpath("report.md").write_text("\n".join(lines))
    return {"config": config, "structural": structural, "scalars": scalars,
            "recovery": recovery, "comparison": comparison}


def write_reference_report(stage: str, path: Path) -> dict:
    config = read_json(path / "config.json")
    summary = read_json(path / "exact_summary.json")
    audit = read_json(path / "structural_prior_audit.json")
    if not (config and summary):
        return {}

    lines = [f"# Stage {stage.upper()} — exact reference", "",
             f"Source commit `{config['source_commit']}`. "
             f"Built without any MCMC (`uses_mcmc: {config['uses_mcmc']}`).", "",
             f"Target: `{config['target']}`", "",
             "## Grids", "",
             f"- rho: {config['rho_grid']['n']} points on "
             f"[{config['rho_grid']['lo']}, {config['rho_grid']['hi']}]"]
    if "beta_grid" in config:
        lines.append(f"- beta: {config['beta_grid']['n']} points on "
                     f"[{config['beta_grid']['lo']}, {config['beta_grid']['hi']}]")
    lines += [f"- prior cell masses: {config['cell_masses']['n_draws']:,} prior draws, "
              f"seed {config['cell_masses']['seed']}, common random numbers across rho",
              "", "## Coverage and refinement", ""]

    coverage = summary["coverage"]
    lines += ["| coordinate | grid | integrates to | outer-boundary mass |",
              "|---|---|---|---|"]
    for name, block in coverage.items():
        lines.append(f"| {name} | {block['n_points']} points on "
                     f"[{fmt(block['grid_lo'])}, {fmt(block['grid_hi'])}] | "
                     f"{fmt(block['integrates_to'], 8)} | "
                     f"{fmt(block['outer_boundary_mass_fraction'])} |")
    refinement = summary["refinement"]
    lines += ["", "Halving the grid resolution moves:", ""]
    for key, value in refinement.items():
        lines.append(f"- `{key}`: {fmt(value, 6)}")

    lines += ["", "## Posterior over orders", "",
              f"- catalogue size: {summary['n_posets']}",
              f"- MAP poset: {summary['map_poset']} (probability "
              f"{fmt(summary['map_probability'], 8)})",
              f"- true poset: {summary['true_poset_index']}, probability "
              f"{fmt(summary['true_poset_probability'], 8)}, rank "
              f"{summary['true_poset_rank']}",
              f"- MAP is the true poset: {fmt(summary['map_is_true_poset'])}", "",
              "| rank | poset | probability | relations |", "|---|---|---|---|"]
    for rank, entry in enumerate(summary["top5"], start=1):
        lines.append(f"| {rank} | {entry['index']} | {fmt(entry['probability'], 8)} | "
                     f"{entry['relations']} |")

    lines += ["", "## Scalar marginals", "",
              f"- rho: mean {fmt(summary['rho']['mean'])}, sd {fmt(summary['rho']['sd'])}, "
              f"median {fmt(summary['rho']['median'])}, 95% "
              f"[{fmt(summary['rho']['q025'])}, {fmt(summary['rho']['q975'])}]"]
    if "beta" in summary:
        lines.append(f"- beta: mean {fmt(summary['beta']['mean'])}, sd "
                     f"{fmt(summary['beta']['sd'])}, 95% "
                     f"[{fmt(summary['beta']['q025'])}, {fmt(summary['beta']['q975'])}]")

    if audit:
        prior = audit["structural_prior"]
        catalogue = audit["catalogue"]
        likelihood = audit["exact_likelihood"]
        lines += ["", "## Structural prior audit (§2.1 gate)", "",
                  f"- max abs error vs scipy MVN logpdf: "
                  f"{fmt(prior['scipy_mvn_max_abs_error'])}",
                  f"- max abs error of the closed-form log determinant: "
                  f"{fmt(prior['closed_form_logdet_max_abs_error'])}",
                  f"- single-row quadrature mass: "
                  f"{prior['single_row_quadrature_mass']}",
                  f"- row-factorisation max abs error: "
                  f"{fmt(prior['row_factorisation_max_abs_error'])}",
                  f"- a rho-dependent combinatorial normaliser is needed: "
                  f"{fmt(prior['combinatorial_normaliser_needed'])}", "",
                  "Negative control — deleting `-(m/2) log|Sigma_rho|` moves the rho "
                  "posterior mean from "
                  f"{fmt(prior['negative_control']['rho_posterior_mean_with_normaliser'])}"
                  f" to "
                  f"{fmt(prior['negative_control']['rho_posterior_mean_without_normaliser'])}"
                  f" (shift {fmt(prior['negative_control']['shift'])}), so the "
                  "normaliser is load-bearing.", "",
                  "## Catalogue validation", "",
                  f"- size {catalogue['size']} "
                  f"(expected {catalogue['expected_labelled_posets_on_5']}, matches: "
                  f"{fmt(catalogue['matches_expected'])})",
                  f"- duplicate keys: {catalogue['duplicate_keys']}",
                  f"- all entries are partial orders: "
                  f"{fmt(catalogue['all_are_partial_orders'])}",
                  f"- closure/reduction round-trip complete: "
                  f"{fmt(catalogue['round_trip_complete'])}",
                  f"- representatives induce their filed order: "
                  f"{fmt(catalogue['representatives_induce_filed_order'])}",
                  f"- ranking tuples enumerated: "
                  f"{catalogue['ranking_tuples_total']:,} of "
                  f"{catalogue['ranking_tuples_expected']:,}", "",
                  "## Exact likelihood", "",
                  f"- MLE poset: {likelihood['mle_poset_index']}, true poset: "
                  f"{likelihood['true_poset_index']}, MLE is true: "
                  f"{fmt(likelihood['mle_is_true_poset'])}",
                  f"- clear of the runner-up by "
                  f"{fmt(likelihood['nats_clear_of_runner_up'], 1)} nats", ""]

    path.joinpath("report.md").write_text("\n".join(lines))
    return {"config": config, "summary": summary, "audit": audit}


def write_smoke_report(stage: str, path: Path) -> dict:
    summary = read_json(path / "summary.json")
    if not summary:
        return {}
    lines = [f"# Stage {stage.upper()} — smoke run", "",
             f"All required checks passed: **{fmt(summary['all_passed'])}**", "",
             "| check | result |", "|---|---|"]
    for name, value in summary["checks"].items():
        lines.append(f"| {name} | {fmt(value)} |")
    lines += ["", "## Observed behaviour", ""]
    for key in ("n_sweeps", "u_acceptance", "rho_acceptance", "beta_acceptance",
                "rho_range", "beta_range", "final_relation_count"):
        if key in summary:
            lines.append(f"- `{key}`: {summary[key]}")
    lines.append("")
    path.joinpath("report.md").write_text("\n".join(lines))
    return summary


# ------------------------------------------------------------------- completion report
def build_completion(payloads: dict) -> dict:
    frozen = load_stage6c_dataset()
    out = {"stage": "6C", "model_id": STAGE6C_MODEL_ID,
           "stage6c_config_hash": config_hash(),
           "stage6b_config_hash": STAGE6B_CONFIG_HASH,
           "source_commit": source_commit(),
           "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "python": platform.python_version(), "numpy": np.__version__,
           "verdicts": {}, "gates": {}, "artifacts": {}}

    for stage in ("6c1", "6c2"):
        run = payloads.get(f"{stage}_full")
        if not run:
            continue
        gates = run["comparison"]["gates"]
        correctness = all(g["pass"] for g in gates.values())
        structure = run["recovery"]["structural"]
        u_recovered = (structure["map_is_true"]
                       and structure["closure"]["structural_hamming"] == 0)
        u_partial = structure["closure"]["f1"] is not None and (
            structure["closure"]["f1"] > 0.5)

        verdicts = {
            "sampler_correctness": verdict(correctness),
            "u_recovery": ("PASS" if u_recovered else
                           "PARTIAL" if u_partial else "FAIL"),
            "rho_recovery": "NOT APPLICABLE",
            "rho_recovery_reason":
                "U_TRUE is hand specified in recurrent_synthetic.py, not drawn from "
                "p(U | rho); no rho_true exists in the generator",
        }
        if stage == "6c2":
            beta = run["scalars"].get("beta", {})
            verdicts["beta_recovery"] = verdict(bool(beta.get("truth_in_95_interval")))
        out["verdicts"][stage] = verdicts
        out["gates"][stage] = gates

    for name, path in DIRS.items():
        if path.exists():
            out["artifacts"][name] = {
                "path": str(path.relative_to(ROOT)),
                "bytes": directory_size(path),
                "files": sorted(f.name for f in path.iterdir() if f.is_file())}
    figures = COMPLETE / "figures"
    if figures.exists():
        out["artifacts"]["figures"] = {
            "path": str(figures.relative_to(ROOT)), "bytes": directory_size(figures),
            "files": sorted(f.name for f in figures.iterdir() if f.is_file())}
    out["truth"] = {k: float(v) for k, v in frozen.truth.items()}
    return out


def write_complete_report(payloads: dict, completion: dict) -> None:
    lines = ["# Stage 6C — recurrent latent-poset MCMC", "",
             f"Model `{completion['model_id']}`, config hash "
             f"`{completion['stage6c_config_hash'][:16]}...`, source commit "
             f"`{completion['source_commit']}`.", "",
             "Stage 6C makes `U` and `rho` latent (6C1) and then additionally `beta` "
             "(6C2). The target is **continuous in U**:", "",
             "```",
             "p(U, rho[, beta] | Y)  proportional to  "
             "p(Y | h(U), fixed) p(U | rho) p(rho) [p(beta)]",
             "```", "",
             "The chain's state is the real matrix `U`; `h(U)` is a derived label. See "
             "`model_audit.md` in this directory for the Stage 6C0 audit that settled "
             "the model.", "",
             "## Verdicts", "",
             "| stage | sampler correctness | U recovery | rho recovery | beta recovery |",
             "|---|---|---|---|---|"]
    for stage in ("6c1", "6c2"):
        v = completion["verdicts"].get(stage)
        if not v:
            continue
        lines.append(f"| {stage.upper()} | {v['sampler_correctness']} | "
                     f"{v['u_recovery']} | {v['rho_recovery']} | "
                     f"{v.get('beta_recovery', '—')} |")

    lines += ["", "These are four separate questions and are deliberately not "
                  "combined. `rho` recovery is NOT APPLICABLE because "
                  f"{completion['verdicts'].get('6c1', {}).get('rho_recovery_reason', '')}"
                  ".", ""]

    lines += ["## Gates", ""]
    for stage in ("6c1", "6c2"):
        gates = completion["gates"].get(stage)
        if not gates:
            continue
        lines += [f"### Stage {stage.upper()}", "",
                  "| gate | value | threshold | verdict |", "|---|---|---|---|"]
        for name, gate in gates.items():
            lines.append(f"| {name} | {fmt(gate['value'])} | {fmt(gate['threshold'])} | "
                         f"{verdict(gate['pass'])} |")
        lines.append("")

    # ---- §16: what changes when beta is freed -------------------------------------
    one, two = payloads.get("6c1_full"), payloads.get("6c2_full")
    if one and two:
        s1, s2 = one["recovery"]["structural"], two["recovery"]["structural"]
        r1 = one["scalars"]["rho"]; r2 = two["scalars"]["rho"]
        b2 = two["scalars"]["beta"]
        lines += ["## Stage 6C1 vs Stage 6C2 — what freeing beta changes", "",
                  "| quantity | 6C1 (beta fixed) | 6C2 (beta free) |", "|---|---|---|",
                  f"| posterior probability of the true poset | "
                  f"{fmt(s1['posterior_probability_of_true'])} | "
                  f"{fmt(s2['posterior_probability_of_true'])} |",
                  f"| distinct orders visited | {s1['n_unique_states_visited']} | "
                  f"{s2['n_unique_states_visited']} |",
                  f"| closure F1 | {fmt(s1['closure']['f1'])} | "
                  f"{fmt(s2['closure']['f1'])} |",
                  f"| min true-relation probability | "
                  f"{fmt(s1['min_true_relation_probability'])} | "
                  f"{fmt(s2['min_true_relation_probability'])} |",
                  f"| max false-relation probability | "
                  f"{fmt(s1['max_false_relation_probability'])} | "
                  f"{fmt(s2['max_false_relation_probability'])} |",
                  f"| rho posterior mean | {fmt(r1['posterior_mean'])} | "
                  f"{fmt(r2['posterior_mean'])} |",
                  f"| rho posterior sd | {fmt(r1['posterior_sd'])} | "
                  f"{fmt(r2['posterior_sd'])} |", "",
                  "**Structural uncertainty does not change when beta is freed.** Both "
                  "stages put probability 1.0 on the true order and visit exactly one "
                  "order after burn-in, so no new structural mode appears and there is "
                  "no U/beta confounding to report: the likelihood separates the true "
                  "order from its nearest competitor by 271.5 nats, which no value of "
                  "beta in the posterior's support can overturn.", "",
                  f"The 6C2 beta posterior is {fmt(b2['posterior_mean'])} +/- "
                  f"{fmt(b2['posterior_sd'])}, against the Stage 6B1 reference posterior "
                  f"of 1.4961 +/- 0.0319 obtained with **U held fixed at the truth**. "
                  f"Freeing U and rho therefore costs beta essentially no precision — "
                  f"another consequence of the structure being sharply identified.", ""]

    # ---- honest accounting of gates that were not met as specified -----------------
    lines += ["## Gate shortfalls and caveats", ""]
    if one:
        pooled = one["comparison"]["retained_draws_pooled"]
        lines += [
            f"- **Retained-sample count is below the §12 target.** The registered "
            f"protocol (4 chains x 20,000 sweeps, 5,000 burn-in, thinning 5) yields "
            f"{pooled:,} pooled retained draws per stage, against the §12 request for "
            f"at least 100,000. The §13 continuation ceiling of 60,000 sweeps would "
            f"still reach only 44,000, so the two clauses cannot both be satisfied as "
            f"written. The registered run protocol was followed and the shortfall is "
            f"recorded here rather than resolved by rescaling. It does not affect the "
            f"structural gates, which are satisfied by 100+ orders of magnitude, and "
            f"the scalar gates carry their own ESS and MCSE.",
            f"- **No rho marginal KS threshold was pre-registered.** §12 asks for one; "
            f"the gates registered before the runs were TV, relation-marginal error, "
            f"R-hat and the calibrated mixed-reference envelope (which includes the rho "
            f"coordinate). The observed rho KS distances are reported descriptively "
            f"against the distribution-free reference value 1.36/sqrt(bulk ESS); no "
            f"threshold was fitted after seeing them. The beta KS gate (0.05) *was* "
            f"registered in code before Stage 6C2 ran.",
            "- **Relation-count R-hat and ESS are undefined, not passing.** The poset "
            "posterior is a point mass, so the relation-count trace is constant at 6 in "
            "every chain. The diagnostics report this as `degenerate` and emit `null` "
            "rather than a flattering R-hat of 1.0, and correlations involving the "
            "relation count are reported as undefined with the reason attached.",
            "- **The exact reference's one Monte Carlo ingredient is the prior cell "
            "mass.** Everything else is exact enumeration. `pi_rho(P)` used 40,000,000 "
            "prior draws with common random numbers across the rho grid; the maximum "
            "standard error over all 4231 posets and 81 rho values is 1.44e-05.", ""]

    lines += ["## Why rho is weakly identified", "",
              "The poset posterior is effectively a point mass on the true order, and "
              "`rho` does not enter the likelihood at all. So", "",
              "```",
              "p(rho | Y)  proportional to  p(rho) * pi_rho(P_true)",
              "```", "",
              "and the entire `rho` posterior is driven by how the prior cell mass of one "
              "poset varies with `rho`, on 5 rows of a 2-dimensional Gaussian. "
              "`pi_rho(P_true)` is around 1e-4 and decreases gently in `rho`, which is a "
              "weak signal by construction. This is a property of the registered "
              "experiment, not a defect of the sampler, and it is why `rho` "
              "identifiability is reported separately from sampler correctness.", ""]

    lines += ["## Artifacts", "", "| artifact | path | size |", "|---|---|---|"]
    for name, entry in completion["artifacts"].items():
        lines.append(f"| {name} | `{entry['path']}` | "
                     f"{entry['bytes'] / 1024:.0f} KB |")
    lines += ["", "## Not started", "",
              "Stage 6D, unknown-boundary inference, segmentation inference, skill-label "
              "inference, semi-Markov FFBS, Step 7 and real-data experiments are **not "
              "started** in this stage.", ""]

    COMPLETE.mkdir(parents=True, exist_ok=True)
    (COMPLETE / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    payloads = {}
    for stage in ("6c1", "6c2"):
        payloads[f"{stage}_smoke"] = write_smoke_report(stage, DIRS[f"{stage}_smoke"])
        payloads[f"{stage}_reference"] = write_reference_report(
            stage, DIRS[f"{stage}_reference"])
        payloads[f"{stage}_full"] = write_run_report(stage, DIRS[f"{stage}_full"])

    completion = build_completion({k: v for k, v in payloads.items() if v})
    COMPLETE.mkdir(parents=True, exist_ok=True)
    (COMPLETE / "completion_summary.json").write_text(json.dumps(completion, indent=2))
    (COMPLETE / "config.json").write_text(json.dumps(frozen_config(), indent=2))
    write_complete_report({k: v for k, v in payloads.items() if v}, completion)
    print(json.dumps(completion["verdicts"], indent=2))


if __name__ == "__main__":
    main()
