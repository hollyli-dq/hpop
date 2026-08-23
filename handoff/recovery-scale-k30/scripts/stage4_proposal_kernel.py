"""Stage 4 — algorithm correctness of the local proposal kernel.

    PYTHONPATH=src python scripts/stage4_proposal_kernel.py
    PYTHONPATH=src python scripts/stage4_proposal_kernel.py --continue-on-failure

Stage 1 validated the posterior with a proposal that needs the global state list.
This validates the moves a real sampler would use — Split, Merge, Shift, Relabel —
against the same exact posterior, plus detailed balance and a negative control.

No synthetic recovery here: that is Milestone B.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import toy  # noqa: E402
from hpop.mcmc_original.diagnostics import total_variation_distance  # noqa: E402
from hpop.mcmc_original.enumerate import build_trace_states, exact_state_table  # noqa: E402
from hpop.mcmc_original.latent_poset import precedence_from_u  # noqa: E402
from hpop.mcmc_original.proposals import (  # noqa: E402
    LocalMoveKernel,
    MoveType,
    run_local_mcmc,
    transition_matrix,
)
from hpop.mcmc_original.targets import SkillEvaluator, log_target_segmentation  # noqa: E402

SEED = 20260808
N_ITERATIONS = 200_000
BURN_IN = 10_000
N_PROPOSAL_DRAWS = 60_000
TV_TOLERANCE = 0.02
BALANCE_TOLERANCE = 1e-12


class StageFailure(Exception):
    def __init__(self, stage, expected, observed, config):
        super().__init__(f"{stage}: expected {expected}, observed {observed}")
        self.stage, self.expected, self.observed, self.config = (
            stage, expected, observed, config,
        )


def check(condition, stage, expected, observed, config):
    if not condition:
        raise StageFailure(stage, expected, observed, config)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def build() -> dict:
    skills = toy.stage4_skills()
    x = toy.S4_TRACE
    evaluators = [SkillEvaluator(s) for s in skills]
    trace_states = build_trace_states(x, skills, evaluators, toy.S4_DELTA_B)
    tables = [evaluators[k].log_table(skills[k].u) for k in range(len(skills))]
    log_pi = toy.uniform_log_pi(len(skills))
    exact = exact_state_table(trace_states, tables, log_pi)
    kernel = LocalMoveKernel(x=x, skills=skills)
    u_by_skill = {k: skills[k].u for k in range(len(skills))}

    def log_target(segmentation):
        return log_target_segmentation(
            x, segmentation, evaluators, u_by_skill, toy.S4_DELTA_B, log_pi
        )

    return {
        "x": x, "skills": skills, "states": list(trace_states.segmentations),
        "exact": exact, "kernel": kernel, "log_target": log_target,
    }


def describe(ctx, segmentation) -> str:
    x = ctx["x"]
    return " + ".join(
        f"{tuple(x[g.start:g.end])}_{toy.S4_SKILL_NAMES[g.skill]}"
        for g in segmentation.segments
    )


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_moves(ctx) -> dict:
    kernel, states = ctx["kernel"], ctx["states"]
    legal = set(states)
    config = {"trace": ctx["x"], "n_states": len(states)}

    availability = {
        move: sum(len(kernel.neighbours(s, move)) for s in states)
        for move in MoveType.ALL
    }
    for move, total in availability.items():
        check(total > 0, f"Stage 4 move {move}", "at least one legal move somewhere",
              total, config)

    for state in states:
        for move in MoveType.ALL:
            for candidate in kernel.neighbours(state, move):
                check(candidate in legal, f"Stage 4 {move} legality",
                      "a legal enumerated state", describe(ctx, candidate), config)
                check(candidate != state, f"Stage 4 {move}", "a different state",
                      "the current state", config)

    # Split <-> Merge, and Shift/Relabel self-inverse
    for state in states:
        for candidate in kernel.neighbours(state, MoveType.SPLIT):
            check(state in kernel.neighbours(candidate, MoveType.MERGE),
                  "Stage 4 split/merge inversion", "merge undoes split",
                  describe(ctx, candidate), config)
        for candidate in kernel.neighbours(state, MoveType.MERGE):
            check(state in kernel.neighbours(candidate, MoveType.SPLIT),
                  "Stage 4 split/merge inversion", "split undoes merge",
                  describe(ctx, candidate), config)
        for move in (MoveType.SHIFT, MoveType.RELABEL):
            for candidate in kernel.neighbours(state, move):
                check(state in kernel.neighbours(candidate, move),
                      f"Stage 4 {move} inversion", f"{move} is its own inverse",
                      describe(ctx, candidate), config)

    return {"availability_total": availability,
            "availability_by_state": [
                {move: len(kernel.neighbours(s, move)) for move in MoveType.ALL}
                for s in states
            ]}


def check_proposal_law(ctx) -> dict:
    kernel, states = ctx["kernel"], ctx["states"]
    config = {"draws": N_PROPOSAL_DRAWS, "seed": SEED}
    rng = np.random.default_rng(SEED)

    worst_mismatch = 0.0
    for state in states:
        law = kernel.proposal_distribution(state)
        check(abs(sum(law.values()) - 1.0) < 1e-12, "Stage 4 proposal law",
              "q(S -> .) sums to 1", sum(law.values()), config)
        counts: Counter = Counter()
        for _ in range(N_PROPOSAL_DRAWS):
            candidate, _ = kernel.sample_proposal(state, rng)
            counts[candidate] += 1
        for candidate, expected in law.items():
            observed = counts[candidate] / N_PROPOSAL_DRAWS
            worst_mismatch = max(worst_mismatch, abs(observed - expected))
    check(worst_mismatch < 0.01, "Stage 4 sampled-vs-computed q",
          "max |sampled - computed| < 0.01", f"{worst_mismatch:.5f}", config)

    worst_asymmetry = 0.0
    for state in states:
        for candidate, forward in kernel.proposal_distribution(state).items():
            if candidate == state:
                continue
            reverse = kernel.proposal_prob(candidate, state)
            check(reverse > 0.0, "Stage 4 reversible support",
                  "q(S'->S) > 0 whenever q(S->S') > 0", 0.0, config)
            worst_asymmetry = max(worst_asymmetry, abs(forward - reverse))
    check(worst_asymmetry > 0.1, "Stage 4 asymmetry",
          "a genuinely asymmetric proposal (> 0.1)", f"{worst_asymmetry:.5f}", config)

    return {"max_sampling_mismatch": worst_mismatch,
            "max_proposal_asymmetry": worst_asymmetry}


def check_balance(ctx) -> dict:
    states, exact, kernel = ctx["states"], ctx["exact"], ctx["kernel"]
    pi = exact["probs"]
    config = {"n_states": len(states), "tolerance": BALANCE_TOLERANCE}

    matrix = transition_matrix(states, exact["log_targets"], kernel)
    check(np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12), "Stage 4 stochasticity",
          "rows sum to 1", float(np.abs(matrix.sum(axis=1) - 1).max()), config)

    flow = pi[:, None] * matrix
    balance_error = float(np.abs(flow - flow.T).max())
    check(balance_error < BALANCE_TOLERANCE, "Stage 4 detailed balance",
          f"< {BALANCE_TOLERANCE}", f"{balance_error:.3e}", config)

    stationarity_error = float(np.abs(pi @ matrix - pi).max())
    check(stationarity_error < BALANCE_TOLERANCE, "Stage 4 stationarity",
          f"< {BALANCE_TOLERANCE}", f"{stationarity_error:.3e}", config)

    values, vectors = np.linalg.eig(matrix.T)
    leading = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    leading = leading / leading.sum()
    eigen_tv = total_variation_distance(leading, pi)
    check(eigen_tv < 1e-10, "Stage 4 leading eigenvector",
          "TV(eigenvector, posterior) < 1e-10", f"{eigen_tv:.3e}", config)

    n = len(states)
    reachable = np.linalg.matrix_power(matrix + np.eye(n), n)
    check(bool(np.all(reachable > 0)), "Stage 4 irreducibility",
          "every state reachable from every state", "not connected", config)

    # negative control: the same kernel with the Hastings ratio dropped
    index = {s: i for i, s in enumerate(states)}
    naive = np.zeros((n, n))
    for i, state in enumerate(states):
        for candidate, forward in kernel.proposal_distribution(state).items():
            if candidate == state or forward <= 0.0:
                continue
            j = index[candidate]
            log_alpha = exact["log_targets"][j] - exact["log_targets"][i]
            naive[i, j] = forward * min(1.0, math.exp(min(0.0, log_alpha)))
        naive[i, i] = 1.0 - naive[i].sum()
    naive_flow = pi[:, None] * naive
    naive_balance = float(np.abs(naive_flow - naive_flow.T).max())
    values, vectors = np.linalg.eig(naive.T)
    naive_leading = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    naive_leading = naive_leading / naive_leading.sum()
    naive_tv = total_variation_distance(naive_leading, pi)
    check(naive_tv > 0.05, "Stage 4 negative control",
          "dropping the Hastings ratio must break stationarity (TV > 0.05)",
          f"{naive_tv:.5f}", config)

    return {
        "detailed_balance_error": balance_error,
        "stationarity_error": stationarity_error,
        "eigenvector_tv": eigen_tv,
        "irreducible": True,
        "transition_matrix": matrix.tolist(),
        "naive_detailed_balance_error": naive_balance,
        "naive_stationary": naive_leading.tolist(),
        "naive_tv": naive_tv,
    }


def check_long_run(ctx) -> dict:
    states, exact, kernel = ctx["states"], ctx["exact"], ctx["kernel"]
    config = {"iterations": N_ITERATIONS, "burn_in": BURN_IN, "seed": SEED}
    rng = np.random.default_rng(SEED)
    result = run_local_mcmc(states[0], ctx["log_target"], kernel,
                            N_ITERATIONS, BURN_IN, rng)
    index = {s: i for i, s in enumerate(states)}
    counts = np.bincount([index[s] for s in result["kept"]], minlength=len(states))
    empirical = counts / counts.sum()
    tv = total_variation_distance(empirical, exact["probs"])

    check(tv < TV_TOLERANCE, "Stage 4 posterior recovery",
          f"TV < {TV_TOLERANCE}", f"{tv:.5f}", config)
    check(len(set(result["kept"])) == len(states), "Stage 4 coverage",
          f"all {len(states)} states visited", len(set(result["kept"])), config)
    for move in MoveType.ALL:
        check(result["accepted_by_move"][move] > 0, f"Stage 4 {move} acceptance",
              "at least one acceptance", result["accepted_by_move"][move], config)

    return {
        "iterations": N_ITERATIONS, "burn_in": BURN_IN, "seed": SEED,
        "total_variation": tv,
        "empirical": empirical.tolist(),
        "exact": exact["probs"].tolist(),
        "max_abs_error": float(np.abs(empirical - exact["probs"]).max()),
        "acceptance_rate": result["acceptance_rate"],
        "proposed_by_move": result["proposed_by_move"],
        "accepted_by_move": result["accepted_by_move"],
        "acceptance_by_move": result["acceptance_by_move"],
        "n_kept": len(result["kept"]),
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:  # pragma: no cover
            return "unknown"
    return {"branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "commit": run("git", "rev-parse", "HEAD"),
            "dirty": run("git", "status", "--porcelain") != ""}


def write_report(ctx, results, decisions, path: Path) -> None:
    lines: list[str] = []
    add = lines.append
    info = results["environment"]

    add("# Stage 4 — algorithm correctness of the local proposal kernel")
    add("")
    add(f"Date: {date.today().isoformat()}")
    add(f"Branch: `{info['git']['branch']}`  Commit: `{info['git']['commit']}`"
        f"{'  (working tree dirty)' if info['git']['dirty'] else ''}")
    add(f"Python {info['python']}, NumPy {info['numpy']}")
    add("")
    add("Stage 1 proved the posterior was right using a proposal that needs the global")
    add("state list. This stage validates the moves a real sampler would actually use —")
    add("**Split, Merge, Shift, Relabel** — against that same exact posterior. No")
    add("synthetic recovery is attempted here; that is Milestone B.")
    add("")

    add("## PASS / FAIL summary")
    add("")
    add("| check | result | headline |")
    add("|---|---|---|")
    for d in decisions:
        add(f"| {d['stage']} | **{d['status']}** | {d['headline']} |")
    add("")

    add("## 1. Why the Stage 0-3 toy could not test this")
    add("")
    add("The Stage 0-3 skills make two of the four moves structurally dead:")
    add("")
    add("- every block matches **at most one** skill, so Relabel has nothing to move to;")
    add("- every B block consumes exactly one CPA-2 label, so the number of segments `L`")
    add("  is pinned by the trace and Split/Merge can never reach a legal state.")
    add("")
    add("Stage 4 therefore uses a purpose-built toy where the supports genuinely overlap.")
    add("")

    add("## 2. The kernel toy")
    add("")
    add(f"Trace `x = {tuple(ctx['x'])}`, `delta_B = {toy.S4_DELTA_B}`, "
        f"`beta = {toy.BETA}`, `epsilon = {toy.EPSILON}`, uniform `pi_k = 1/4`.")
    add("")
    add("| skill | CPA labels | induced order | role in the test |")
    add("|---|---|---|---|")
    order_notes = {
        "A": "0 > 1",
        "D": "antichain (0 and 1 incomparable)",
        "F": "2 > 3",
        "E": "0 > 1 and 2 > 3, the two chains incomparable",
    }
    purpose = {
        "A": "shares a support with D",
        "D": "**makes Relabel live** — same block, different order",
        "F": "splits out of E",
        "E": "support = A's union F's, **makes Split/Merge live**",
    }
    for k, name in enumerate(toy.S4_SKILL_NAMES):
        add(f"| {name} | {ctx['skills'][k].cpa_labels} | {order_notes[name]} "
            f"| {purpose[name]} |")
    add("")
    add("Shift is live because a boundary can slide between the 2-block and 4-block")
    add("tilings, relabelling both adjacent segments as it goes.")
    add("")
    add(f"The trace admits **{len(ctx['states'])} legal segmentations** with `L` in "
        f"{sorted({len(s.segments) for s in ctx['states']})}:")
    add("")
    add("| # | segmentation | exact P(S \\| x) | relabel | split | merge | shift |")
    add("|---|---|---|---|---|---|---|")
    for i, (state, p) in enumerate(zip(ctx["states"], ctx["exact"]["probs"])):
        avail = results["moves"]["availability_by_state"][i]
        add(f"| {i} | {describe(ctx, state)} | {p:.6f} | "
            + " | ".join(str(avail[m]) for m in MoveType.ALL) + " |")
    add("")

    add("## 3. Move-level correctness")
    add("")
    add("Verified for every state and every move:")
    add("")
    add("- every proposed state is support-compatible and tiles the whole trace;")
    add("- no move returns the current state;")
    add("- **Split and Merge are exact inverses** of each other;")
    add("- **Shift and Relabel are their own inverses**.")
    add("")
    add("Total availability across the state space: "
        + ", ".join(f"`{m}` {results['moves']['availability_total'][m]}"
                    for m in MoveType.ALL) + ".")
    add("")

    add("## 4. The proposal law")
    add("")
    add("The kernel exposes `q(S -> S')` explicitly:")
    add("")
    add("```")
    add("q(S -> S') = sum_t  p_t * 1[S' in N_t(S)] / |N_t(S)|")
    add("```")
    add("")
    add("`|N_t(S)|` is a **local** count — it inspects the segments of one state, never")
    add("the global state list — so this is computable in a real sampler.")
    add("")
    add(f"- sampled proposals match the computed law: max gap "
        f"**{results['proposal']['max_sampling_mismatch']:.5f}** over "
        f"{N_PROPOSAL_DRAWS:,} draws per state (criterion < 0.01)")
    add(f"- the proposal is genuinely **asymmetric**: max `|q(S->S') - q(S'->S)|` = "
        f"**{results['proposal']['max_proposal_asymmetry']:.4f}**")
    add("- `q(S->S') > 0` implies `q(S'->S) > 0` for every pair, so every move can be undone")
    add("")
    add("The asymmetry is structural, not incidental: the all-E state offers 4 splits,")
    add("while each state reachable from it offers only 1 merge back.")
    add("")

    add("## 5. Detailed balance and stationarity")
    add("")
    add("Built the exact MH transition matrix `K` over the enumerated space and checked:")
    add("")
    add("| property | value | criterion |")
    add("|---|---|---|")
    b = results["balance"]
    add(f"| rows of K sum to 1 | yes | exact |")
    add(f"| detailed balance, max \\|pi_i K_ij - pi_j K_ji\\| | "
        f"{b['detailed_balance_error']:.3e} | < {BALANCE_TOLERANCE} |")
    add(f"| stationarity, max \\|piK - pi\\| | {b['stationarity_error']:.3e} "
        f"| < {BALANCE_TOLERANCE} |")
    add(f"| TV(leading left eigenvector of K, posterior) | {b['eigenvector_tv']:.3e} "
        f"| < 1e-10 |")
    add(f"| irreducible | yes | all states mutually reachable |")
    add("")

    add("### The negative control")
    add("")
    add("This is what gives the whole stage teeth. The same kernel with the Hastings")
    add("ratio dropped — the mistake a symmetric-proposal implementation would make —")
    add("still runs, still mixes, and still looks healthy, but:")
    add("")
    add(f"- detailed balance breaks: max flow asymmetry **{b['naive_detailed_balance_error']:.5f}**")
    add(f"- its stationary distribution is **TV = {b['naive_tv']:.5f}** away from the posterior")
    add("")
    add("| state | exact | naive-kernel stationary |")
    add("|---|---|---|")
    for i, (p, q) in enumerate(zip(ctx["exact"]["probs"], b["naive_stationary"])):
        add(f"| {i} | {p:.6f} | {q:.6f} |")
    add("")
    add("A trace that happens to induce a symmetric proposal cannot detect this at all,")
    add("which is why the kernel toy was selected for asymmetry rather than convenience.")
    add("")

    add("## 6. Posterior recovery over a long run")
    add("")
    r = results["long_run"]
    add(f"{r['iterations']:,} iterations, {r['burn_in']:,} burn-in, "
        f"{r['n_kept']:,} kept, seed `{r['seed']}`, overall acceptance "
        f"**{r['acceptance_rate']:.4f}**.")
    add("")
    add("| move | proposed | accepted | acceptance |")
    add("|---|---|---|---|")
    for move in MoveType.ALL:
        add(f"| {move} | {r['proposed_by_move'][move]:,} | "
            f"{r['accepted_by_move'][move]:,} | {r['acceptance_by_move'][move]:.4f} |")
    add("")
    add("| state | exact | MCMC | abs error |")
    add("|---|---|---|---|")
    for i, (e, m) in enumerate(zip(r["exact"], r["empirical"])):
        add(f"| {i} | {e:.6f} | {m:.6f} | {abs(m - e):.6f} |")
    add("")
    add(f"**Total variation distance = {r['total_variation']:.6f}** "
        f"(criterion < {TV_TOLERANCE})")
    add(f"Worst single-state error = {r['max_abs_error']:.6f}. All "
        f"{len(ctx['states'])} states visited; every move type proposed *and* accepted.")
    add("")

    add("## 7. Notes and limitations")
    add("")
    for note in results["notes"]:
        add(f"- {note}")
    add("")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="debug only: keep going after a check fails")
    args = parser.parse_args()

    out_dir = ROOT / "results" / "mcmc_original"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "stage4_proposal_kernel.json"
    report_path = out_dir / "stage4_proposal_kernel_report.md"

    ctx = build()
    results = {
        "environment": {"git": git_info(), "python": platform.python_version(),
                        "numpy": np.__version__},
        "trace": list(ctx["x"]),
        "n_states": len(ctx["states"]),
        "states": [describe(ctx, s) for s in ctx["states"]],
        "exact_posterior": ctx["exact"]["probs"].tolist(),
        "notes": [
            "The Stage 0-3 toy cannot exercise this kernel: every block there matches at "
            "most one skill (Relabel dead) and L is pinned by the trace (Split/Merge dead). "
            "Stage 4 uses a purpose-built toy with overlapping supports.",
            "The kernel toy trace was selected for a genuinely asymmetric proposal. "
            "(0,1,2,3,0,1) is accidentally symmetric and (0,1,2,3,2,3,0,1) offers no legal "
            "Shift; on either, a missing Hastings correction would go undetected.",
            "With role counts of 2 and 4, a one-position Shift is never legal on this trace; "
            "shift_moves therefore defaults to allowing any boundary inside the combined "
            "span, which is what keeps the kernel irreducible. max_shift=1 is implemented "
            "and tested, but yields no legal moves here.",
            "This stage validates the kernel on a single fixed trace with U held fixed. It "
            "says nothing about joint S+U sampling with local moves, or about mixing on "
            "long real traces — those belong to Milestone B and later.",
        ],
    }
    decisions: list[dict] = []
    failures: list[StageFailure] = []

    def run_check(name, headline_fn, fn, key):
        try:
            value = fn(ctx)
        except StageFailure as failure:
            decisions.append({"stage": name, "status": "FAIL",
                              "headline": f"expected {failure.expected}, "
                                          f"observed {failure.observed}"})
            print(f"[FAIL] {name}")
            print(f"       check    : {failure.stage}")
            print(f"       expected : {failure.expected}")
            print(f"       observed : {failure.observed}")
            print(f"       config   : {failure.config}")
            print(f"       report   : {report_path}")
            failures.append(failure)
            return None
        results[key] = value
        decisions.append({"stage": name, "status": "PASS", "headline": headline_fn(value)})
        print(f"[PASS] {name}")
        return value

    steps = [
        ("Stage 4A move-level correctness",
         lambda v: "split/merge inverse, shift/relabel self-inverse, all moves legal",
         check_moves, "moves"),
        ("Stage 4B proposal law q(S->S')",
         lambda v: f"sampled matches computed (max gap {v['max_sampling_mismatch']:.5f}), "
                   f"asymmetry {v['max_proposal_asymmetry']:.4f}",
         check_proposal_law, "proposal"),
        ("Stage 4C detailed balance + stationarity",
         lambda v: f"balance {v['detailed_balance_error']:.1e}, "
                   f"naive-kernel control TV {v['naive_tv']:.4f}",
         check_balance, "balance"),
        ("Stage 4D posterior recovery (200k steps)",
         lambda v: f"TV = {v['total_variation']:.5f}", check_long_run, "long_run"),
    ]
    for name, headline, fn, key in steps:
        value = run_check(name, headline, fn, key)
        if value is None and not args.continue_on_failure:
            break

    results["decisions"] = decisions
    json_path.write_text(json.dumps(jsonable(results), indent=2) + "\n")
    write_report(ctx, results, decisions, report_path)

    print()
    print(f"JSON   : {json_path}")
    print(f"Report : {report_path}")
    if failures:
        print(f"\n{len(failures)} check(s) FAILED")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
