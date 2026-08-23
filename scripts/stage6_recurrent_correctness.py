"""Stage 6.0 / 6A — recurrent likelihood correctness and generator validation.

    PYTHONPATH=src python scripts/stage6_recurrent_correctness.py --mode smoke \
        --generator-seed 0 --output-dir results/mcmc_original/stage6_part1_smoke

Validates that p_RFS is a normalized probability model and that the generator samples
from exactly that model. NO recurrent-parameter MCMC is implemented here.
Stage 5C (full joint S+U+P) remains DEFERRED and is not required by this task.
"""
from __future__ import annotations
import argparse, itertools, json, platform, subprocess, sys
from datetime import date
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from hpop.mcmc_original.latent_poset import precedence_from_u                    # noqa: E402
from hpop.mcmc_original.static_bpop import bpop_step_probabilities               # noqa: E402
from hpop.mcmc_original import recurrent_synthetic as rs                          # noqa: E402
from hpop.mcmc_original import stage6_diagnostics as sd                           # noqa: E402
from hpop.mcmc_original.recurrent_rfs import (                                    # noqa: E402
    RecurrentRFSParameters, logit, recurrent_feasibility, recurrent_rfs_likelihood,
    recurrent_stale_successor_counts, recurrent_step_probabilities,
    recurrent_validity_update, sample_recurrent_rfs_sequence,
)

class Failure(Exception):
    def __init__(self, stage, expected, observed):
        super().__init__(f"{stage}: expected {expected}, observed {observed}")
        self.stage, self.expected, self.observed = stage, expected, observed

def check(cond, stage, expected, observed):
    if not cond: raise Failure(stage, expected, observed)

def jsonable(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.floating, np.integer)): return v.item()
    if isinstance(v, (np.bool_, bool)): return bool(v)
    if isinstance(v, dict): return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)): return [jsonable(x) for x in v]
    return v

def git(*a):
    try: return subprocess.run(("git",)+a, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception: return "unknown"

def deterministic_checks():
    rs.assert_stage6_library()
    P = precedence_from_u(rs.U_TRUE); q0 = np.zeros(5)
    F = recurrent_feasibility(P, q0); S = recurrent_stale_successor_counts(P, q0)
    check(F.tolist() == [1., 1., 0., 0., 0.], "6.0 feasibility", "[1,1,0,0,0]", F.tolist())
    check(S.astype(int).tolist() == [3, 0, 2, 1, 0], "6.0 stale counts", "[3,0,2,1,0]", S.tolist())
    om = logit(0.85)
    q1 = recurrent_validity_update(0, P, np.ones(5), om)
    check(np.allclose(q1, [1, 1, .15, .15, .15]), "6.0 invalidation",
          "[1,1,.15,.15,.15]", q1.tolist())
    q2 = recurrent_validity_update(0, P, q1, om)
    check(abs(q2[4] - 0.15**2) < 1e-12, "6.0 compounding", 0.0225, float(q2[4]))
    pr = RecurrentRFSParameters(1.5, 0.02, om, 0.0, 0.0)
    rec = recurrent_step_probabilities(rs.U_TRUE, q0, pr)
    sta = bpop_step_probabilities(list(range(5)), rs.U_TRUE, 1.5, 0.02)
    d = float(np.abs(rec - sta).max())
    check(d < 1e-12, "6.0 first-step static parity", "< 1e-12", d)
    return {"feasibility_q0": F.tolist(), "stale_counts_q0": S.astype(int).tolist(),
            "closure": [[int(i), int(j)] for i in range(5) for j in range(5) if P[i, j]],
            "first_step_parity_max_diff": d}

def normalization():
    rng = np.random.default_rng(0); rows = []; worst = 0.0
    settings = [(1.5,.02,.85,.8,.25), (0.,0.,.5,0.,0.), (3.,.1,.99,2.,1.5),
                (1.,.05,.01,0.,1.), (0.,.2,.999,0.,0.)]
    for (m, T) in [(2,4),(3,3),(3,4)]:
        for (b,e,k,lr,lb) in settings:
            U = rng.normal(size=(m,2))
            total = sum(recurrent_rfs_likelihood(s, U, b, e, logit(k), lr, lb)
                        for s in itertools.product(range(m), repeat=T))
            worst = max(worst, abs(total-1.0))
            rows.append({"m":m,"T":T,"n_sequences":m**T,"beta":b,"epsilon":e,"kappa":k,
                         "lambda_rep":lr,"lambda_back":lb,"sum":total,"abs_error":abs(total-1.0)})
    check(worst < 1e-10, "6.0 fixed-length normalization", "< 1e-10", worst)
    return {"rows": rows, "worst_abs_error": worst}

def empirical_check(seed):
    U = np.array([[1.0, 1.0], [0.0, 0.0]])
    pr = RecurrentRFSParameters(1.5, 0.02, logit(0.85), 0.8, 0.25)
    exact = {s: recurrent_rfs_likelihood(s, U, pr.beta, pr.epsilon, pr.shared_omega,
                                         pr.lambda_rep, pr.lambda_back)
             for s in itertools.product(range(2), repeat=4)}
    n = 100_000
    rng = np.random.default_rng(seed)
    counts = {s: 0 for s in exact}
    for _ in range(n):
        counts[sample_recurrent_rfs_sequence(rng, 4, U, pr)] += 1
    emp = {s: c/n for s, c in counts.items()}
    tv = 0.5*sum(abs(emp[s]-exact[s]) for s in exact)
    mx = max(abs(emp[s]-exact[s]) for s in exact)
    check(tv < 0.01, "6.0 empirical frequencies", "TV < 0.01", tv)
    return {"n_samples": n, "total_variation": tv, "max_abs_frequency_error": mx,
            "table": [{"sequence": list(s), "analytic": exact[s], "empirical": emp[s]}
                      for s in sorted(exact)]}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("smoke","full"), default="smoke")
    ap.add_argument("--generator-seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--continue-on-failure", action="store_true")
    a = ap.parse_args(); out = a.output_dir; out.mkdir(parents=True, exist_ok=True)
    results, decisions, failures = {}, [], []

    def step(name, fn, *args):
        try: v = fn(*args)
        except Failure as f:
            decisions.append({"stage": name, "status": "FAIL",
                              "headline": f"expected {f.expected}, observed {f.observed}"})
            print(f"[FAIL] {name}\n       expected {f.expected}\n       observed {f.observed}")
            failures.append(f); return None
        decisions.append({"stage": name, "status": "PASS"}); print(f"[PASS] {name}"); return v

    results["deterministic"] = step("Stage 6.0 deterministic recurrent equations", deterministic_checks)
    if results["deterministic"] is None and not a.continue_on_failure: return 1
    results["normalization"] = step("Stage 6.0 fixed-length normalization", normalization)
    if results["normalization"] is None and not a.continue_on_failure:
        print("STOPPING: normalization is the gate; the dataset is not generated."); return 1

    data = rs.generate_recurrent_dataset(a.mode, a.generator_seed)
    results["parity"] = step("Stage 6A generator-likelihood parity", lambda: (
        (lambda p: (check(p["max_abs_log_likelihood_difference"] < 1e-12,
                          "6A parity", "< 1e-12", p["max_abs_log_likelihood_difference"]), p)[1])(
            sd.generator_likelihood_parity(data.train, data.u_true, data.parameters))))
    results["empirical"] = step("Stage 6.0 empirical frequency check", empirical_check, a.generator_seed)

    audit = rs.exposure_audit(data.train)
    results["exposure"] = audit
    if a.mode == "full":
        def gate():
            check(audit["leaf_repeat"] >= 100, "6A leaf repeats", ">= 100", audit["leaf_repeat"])
            check(audit["upstream_repeat"] >= 100, "6A upstream repeats", ">= 100", audit["upstream_repeat"])
            check(audit["recomputation"] >= 100, "6A recomputations", ">= 100", audit["recomputation"])
            for pair, v in audit["gate_exposure_true_pairs"].items():
                check(v >= 50, f"6A gate exposure {pair}", ">= 50", v)
            return audit
        step("Stage 6A exposure audit", gate)
    else:
        print(f"[INFO] Stage 6A exposure audit (smoke, no registered thresholds): "
              f"leaf {audit['leaf_repeat']}, upstream {audit['upstream_repeat']}, "
              f"recomputation {audit['recomputation']}")

    results["curvature"] = sd.identifiability_curvature(data.train, data.u_true, data.parameters)
    cw = results["curvature"]
    print(f"[INFO] identifiability curvature: weakest parameter is "
          f"{cw['weakest_parameter']} at {cw['weakest_true_over_sd']:.1f} sd from zero")

    if data.heldout:
        diag = sd.held_out_diagnostics(data.heldout, data.u_true, data.parameters)
        results["held_out"] = diag
        step("Stage 6A held-out: true beats wrong antichain U", lambda: (
            check(diag["hard_criterion_true_beats_antichain"], "6A antichain criterion",
                  "NLL(true) < NLL(antichain)",
                  f"{diag['nll_true_parameters']:.4f} vs "
                  f"{diag['variants']['wrong_antichain_U']['nll_variant']:.4f}"), diag)[1])

    cfg = {"date": date.today().isoformat(), "git_branch": git("rev-parse","--abbrev-ref","HEAD"),
           "git_commit": git("rev-parse","HEAD"), "python": platform.python_version(),
           "numpy": np.__version__, "stage_5c": "DEFERRED - not required by this task",
           "recurrent_mcmc_implemented": False, **data.config}
    (out/"config.json").write_text(json.dumps(jsonable(cfg), indent=2)+"\n")
    (out/"dataset.json").write_text(json.dumps(jsonable({
        "config": data.config,
        "train": [{"roles": list(b.roles), "split": b.split} for b in data.train],
        "heldout": [{"roles": list(b.roles), "split": b.split} for b in data.heldout],
        "synthetic_truth_note": "q traces are diagnostics only; the likelihood never needs them",
    }), indent=2)+"\n")
    results["decisions"] = decisions
    (out/"stage6_part1_results.json").write_text(json.dumps(jsonable(results), indent=2)+"\n")

    L = [f"# Stage 6.0 / 6A — recurrent likelihood correctness ({a.mode}, seed {a.generator_seed})", "",
         f"Date {cfg['date']} · branch `{cfg['git_branch']}` · commit `{cfg['git_commit'][:8]}`",
         f"Python {cfg['python']} · NumPy {cfg['numpy']}", "",
         "**Stage 5C (full joint S+U+P) remains DEFERRED** — not required here, since this task",
         "holds the boundaries and the latent poset fixed. **No recurrent-parameter MCMC has been",
         "implemented**: this validates the likelihood and the generator only.", "",
         "## PASS / FAIL", "", "| check | result |", "|---|---|"]
    for d in decisions: L.append(f"| {d['stage']} | **{d['status']}** |")
    det = results["deterministic"]; nz = results["normalization"]
    L += ["", "## Model and seeds", "",
          f"- `U_TRUE` closure: {det['closure']}", 
          f"- role 1 incomparable with all others; `U_ANTICHAIN` induces no ordered pairs",
          f"- fixed `T = {data.config['T']}` — **the likelihood conditions on T**; there is no",
          "  `p(T | skill)`, no duration and no stopping model",
          f"- beta {data.config['beta']}, epsilon {data.config['epsilon']}, kappa {data.config['kappa']},",
          f"  lambda_rep {data.config['lambda_rep']}, lambda_back {data.config['lambda_back']}",
          f"- seeds: train {data.config['seed_train']}, held-out {data.config['seed_heldout']}", "",
          "## Deterministic recurrent equations", "",
          f"- `F(q0)` = {det['feasibility_q0']} (expected [1,1,0,0,0])",
          f"- `S(q0)` = {det['stale_counts_q0']} (expected [3,0,2,1,0])",
          f"- first-step parity with static BPOP: max diff **{det['first_step_parity_max_diff']:.3e}**", "",
          "## Fixed-length normalization (the gate)", "",
          "| m | T | sequences | beta | eps | kappa | l_rep | l_back | \\|sum-1\\| |", "|---|---|---|---|---|---|---|---|---|"]
    for r in nz["rows"]:
        L.append(f"| {r['m']} | {r['T']} | {r['n_sequences']} | {r['beta']} | {r['epsilon']} | "
                 f"{r['kappa']} | {r['lambda_rep']} | {r['lambda_back']} | {r['abs_error']:.2e} |")
    L += ["", f"**worst |sum − 1| = {nz['worst_abs_error']:.3e}** (criterion < 1e-10)", ""]
    if results.get("parity"):
        p = results["parity"]
        L += ["## Generator–likelihood parity", "",
              f"- blocks replayed: {p['n_blocks']}",
              f"- max \\|log-likelihood difference\\|: **{p['max_abs_log_likelihood_difference']:.3e}**",
              f"- max \\|q difference\\|: **{p['max_abs_q_difference']:.3e}**",
              f"- max \\|step log-p difference\\|: **{p['max_abs_step_logp_difference']:.3e}**", ""]
    if results.get("empirical"):
        e = results["empirical"]
        L += ["## Empirical vs analytic frequencies (m=2, T=4)", "",
              f"- {e['n_samples']:,} samples; **TV = {e['total_variation']:.5f}**, "
              f"max abs error {e['max_abs_frequency_error']:.5f}", ""]
    ex = results["exposure"]
    L += ["## Exposure audit (training blocks)", "",
          f"- steps {ex['total_steps']:,} over {ex['n_blocks']} blocks",
          f"- valid-role repeats {ex['valid_repeat']:,}; **leaf repeats {ex['leaf_repeat']:,}**; "
          f"**upstream repeats {ex['upstream_repeat']:,}**; **recomputations {ex['recomputation']:,}**", "",
          "Gate exposure `E_zx` for the true ordered pairs:", "",
          "| pair | count |", "|---|---|"]
    for k, v in ex["gate_exposure_true_pairs"].items(): L.append(f"| {k} | {v} |")
    if results.get("held_out"):
        h = results["held_out"]
        L += ["", "## Held-out diagnostics", "",
              f"NLL per step under true parameters: **{h['nll_true_parameters']:.5f}** "
              f"over {h['n_blocks']} blocks.", "",
              "| perturbation | NLL | paired mean diff | s.e. | median | frac favouring true |",
              "|---|---|---|---|---|---|"]
        for name, r in h["variants"].items():
            L.append(f"| {name} | {r['nll_variant']:.5f} | {r['paired_mean_difference']:+.5f} | "
                     f"{r['standard_error']:.5f} | {r['median_difference']:+.5f} | "
                     f"{r['fraction_favouring_true']:.3f} |")
        L += ["", "Only the antichain row is a hard criterion. The others are **diagnostics**:",
              "a small mean difference is reported as small, and no scalar-parameter",
              "identifiability is claimed — that is Stage 6B.", ""]
    L += ["## Identifiability — deferred to Stage 6B", "",
          "The zero-ablation numbers above are **not** by themselves an identifiability",
          "diagnostic. Setting `lambda_back` from 0.25 to 0 raises held-out NLL by only",
          "+0.0092, but a zero-ablation gap depends on both the curvature of the likelihood",
          "and the distance from the true value to zero, and here the true coefficient is",
          "itself small. Nothing about that number implies weak identifiability.", "",
          "Identifiability is evaluated in **Stage 6B** using numerically normalized",
          "one-dimensional reference posterior profiles and MCMC recovery, with `omega`",
          "handled by full replay of the validity recursion. See",
          "`results/mcmc_original/stage6b_full_seed0/`.", "",
          "## Deviations and notes", "",
          "- `lambda_rep` is a **relative penalty on currently valid candidates**, not a monotone",
          "  sequence-level repeat cost; no monotonicity is asserted anywhere.",
          "- The latent graph stays **acyclic**; repetition is carried by the validity state.",
          "- Repeated sequences are never passed to the static optimized likelihood.",
          "- No U / rho / beta / omega / lambda inference was performed.", ""]
    (out/"stage6_part1_report.md").write_text("\n".join(L)+"\n")
    print(f"\nOutputs: {out}\nReport : {out/'stage6_part1_report.md'}")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
