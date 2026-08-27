"""Stage 6D2 — §15 convergence, §16 recovery, §13 column permutation, §17 held-out, §20 gates.

    PYTHONPATH=src python scripts/stage6d2_analysis.py --register
    PYTHONPATH=src python scripts/stage6d2_analysis.py --run-dir results/mcmc_original/...

`--register` writes the gate registration and is run **before the formal chains start**,
so no threshold in this file has ever seen a Stage 6D2 draw. `--run-dir` then reads a
completed run and reports against that registration, unchanged.

## Why Stage 6D2 has no reference-comparison gate, and what replaces it

Stage 6D1 is where sampler *correctness* was established, against an independent
scrambled-Sobol reference on a deliberately small model. No such reference exists for the
500-block corpus and none can: prior importance sampling degrades as the likelihood
sharpens, which is exactly why the Stage 6D1 model was made small. Stage 6D2 therefore
answers a different question — does the validated sampler, run on the real corpus,
converge, recover the generating structure, and reproduce what its parents already
established? — and its gates are convergence, parent consistency and recovery.

Parent consistency is the substantive one. Stage 6C established that on this corpus the
induced-order posterior is a **point mass** at the true order. The recurrent likelihood
reads `U` only through `h(U)`, so if that point mass survives in Stage 6D2 the likelihood
seen by the four scalars is *identical* to the one Stage 6B3 saw with `U` pinned at the
truth. The four scalar marginals must therefore agree with Stage 6B3's, and `beta` and
`rho` must agree with Stage 6C2's. Any disagreement is `U`-`beta`, `H`-`omega` or
`rho`-`U` confounding, and these gates are how it would show up.

## The three verdicts are kept apart

    sampler correctness   established in Stage 6D1; not re-litigated here
    convergence           does this run mix on this corpus?
    recovery              does the posterior find the generating U and scalars?

A correct sampler can fail to recover and a broken one can appear to. Reporting them as
one number would erase the distinction Stage 6D exists to make.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.latent_poset import precedence_from_u              # noqa: E402
from hpop.mcmc_original.recurrent_scalar_mcmc import (                     # noqa: E402
    blockwise_recurrent_log_likelihood,
)
from hpop.mcmc_original.recurrent_scalar_posterior import PRIORS           # noqa: E402
from hpop.mcmc_original.stage6c_diagnostics import (                       # noqa: E402
    convergence_block, mode_summary, recovery_metrics,
)
from hpop.mcmc_original.stage6c_exact_reference import build_catalogue     # noqa: E402
from hpop.mcmc_original.stage6d_diagnostics import (                       # noqa: E402
    column_permutation_audit,
)
from hpop.mcmc_original.stage6d_frozen import (                            # noqa: E402
    ACTIVE_6D, SCALAR_ORDER, config_hash, load_stage6d_dataset,
)

RESULTS = ROOT / "results" / "mcmc_original"
REGISTRATION = RESULTS / "stage6d2_gate_registration.json"

# --------------------------------------------------------------- registered thresholds
RHAT_MAX = 1.01
MIN_BULK_ESS = 1_000
MAX_MCSE_OVER_SD = 0.05
ACCEPTANCE_BAND = (0.20, 0.65)
PARENT_MEAN_TOLERANCE_IN_SD = 0.25
MIN_TRUE_ORDER_PROBABILITY = 0.99
PREDICTIVE_DRAWS = 400

# The parent posteriors this run must reproduce. Read from the committed Stage 6B3 and
# Stage 6C2 result files at registration time and frozen into the registration, so a later
# edit to those files cannot silently move a Stage 6D2 gate.
PARENT_FILES = {
    "6b3": RESULTS / "stage6b3_joint4_full_seed0" / "scalar_diagnostics.json",
    "6c2": RESULTS / "stage6c2_u_rho_beta_full_seed0" / "scalar_diagnostics.json",
    "6c2_recovery": RESULTS / "stage6c2_u_rho_beta_full_seed0" / "recovery_results.json",
}


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:                                            # pragma: no cover
        return "unknown"


def _jsonable(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if math.isnan(v) else v
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def parent_posteriors() -> dict:
    """The Stage 6B3 and Stage 6C2 marginals, as committed."""
    b3 = json.loads(PARENT_FILES["6b3"].read_text())["marginals"]
    c2 = json.loads(PARENT_FILES["6c2"].read_text())
    c2r = json.loads(PARENT_FILES["6c2_recovery"].read_text())
    return {
        "stage6b3": {name: {"mean": b3[name]["mcmc"]["mean"],
                            "sd": b3[name]["mcmc"]["sd"]}
                     for name in SCALAR_ORDER},
        "stage6b3_note": "all four scalars free, U pinned at U_TRUE",
        "stage6c2": {name: {"mean": c2[name]["posterior_mean"],
                            "sd": c2[name]["posterior_sd"]}
                     for name in ("rho", "beta")},
        "stage6c2_note": "U, rho and beta free; omega, lambda_rep and lambda_back fixed",
        "stage6c2_structure": {
            "true_poset_index": c2r["structural"]["true_poset_index"],
            "posterior_probability_of_true":
                c2r["structural"]["posterior_probability_of_true"],
            "n_unique_states_visited": c2r["structural"]["n_unique_states_visited"]},
    }


# ------------------------------------------------------------------------ registration
def registration() -> dict:
    return {
        "registered_before_any_stage6d2_draw_existed": True,
        "stage": "6D2",
        "stage6d_config_hash": config_hash(),
        "source_commit": source_commit(),
        "why_no_reference_gate":
            "sampler correctness was established in Stage 6D1 against an independent "
            "scrambled-Sobol reference. No prior-importance reference is achievable on "
            "the 500-block corpus — that is why the Stage 6D1 model was made small — so "
            "Stage 6D2 gates convergence, parent consistency and recovery instead, and "
            "does not restate the correctness claim.",
        "primary_gates": {
            "convergence": {
                "per_coordinate_rhat": {"threshold": RHAT_MAX, "comparison": "<="},
                "log_posterior_rhat": {"threshold": RHAT_MAX, "comparison": "<="},
                "relation_count_rhat": {
                    "threshold": RHAT_MAX, "comparison": "<=",
                    "degenerate_counts_as_pass": True,
                    "why": "Stage 6C established that the induced-order posterior on "
                           "this corpus is a point mass at the true order, so a constant "
                           "relation-count trace is the expected finding. It is recorded "
                           "as `degenerate`, never as an R-hat of 1.0, and the point "
                           "mass is separately gated by the structural-recovery gate."},
                "uncertain_relation_rhat": {
                    "threshold": RHAT_MAX, "comparison": "<=",
                    "not_applicable_if_no_relation_varies": True},
                "min_bulk_ess": {"threshold": MIN_BULK_ESS, "comparison": ">=",
                                 "over": ["rho"] + list(SCALAR_ORDER)},
                "max_mcse_over_posterior_sd": {"threshold": MAX_MCSE_OVER_SD,
                                               "comparison": "<="},
                "post_burn_in_acceptance_band": {
                    "band": list(ACCEPTANCE_BAND), "over": list(ACTIVE_6D),
                    "why": "the band the Stage 6D2 pilot's joint confirmation used; "
                           "carried into the formal run unchanged"},
            },
            "parent_consistency": {
                "tolerance_in_parent_sd": PARENT_MEAN_TOLERANCE_IN_SD,
                "scalars_vs_stage6b3": list(SCALAR_ORDER),
                "rho_vs_stage6c2": ["rho"],
                "structure_vs_stage6c2": {
                    "min_probability_of_the_stage6c2_map_order":
                        MIN_TRUE_ORDER_PROBABILITY},
                "why": "the recurrent likelihood reads U only through h(U). If the "
                       "induced order is the same point mass Stage 6C found, the "
                       "likelihood the scalars see is identical to Stage 6B3's, so the "
                       "marginals must agree. Disagreement is U-beta, H-omega or rho-U "
                       "confounding.",
                "why_rho_belongs_to_the_6c2_comparison":
                    "rho enters only p(U | rho) and the likelihood is constant inside an "
                    "order cell, so conditional on the same point-mass order posterior "
                    "the rho marginal does not depend on the scalars at all. Stage 6C2's "
                    "rho marginal is therefore the right comparator even though Stage "
                    "6C2 fixed three of the scalars.",
                "tolerance_justification":
                    "with bulk ESS in the thousands the Monte Carlo error on a "
                    "posterior mean is under 0.02 parent sd, so 0.25 sd is roughly a "
                    "12-sigma allowance and cannot be met by a genuinely shifted "
                    "posterior",
                "deliberately_not_a_gate": {
                    "beta_vs_stage6c2":
                        "excluded before any Stage 6D2 draw existed. Stage 6C2 holds "
                        "omega, lambda_rep and lambda_back at their registered values "
                        "while Stage 6B3 and Stage 6D2 marginalise over them, and beta "
                        "is correlated with those three: the two parents already "
                        "disagree with each other by 0.40 Stage 6B3 sd (1.48097 against "
                        "1.49648), so no single Stage 6D2 value could satisfy a 0.25 sd "
                        "gate against both. Requiring agreement with a differently "
                        "conditioned posterior would be a gate on a quantity that is not "
                        "supposed to be equal. The contrast is computed and reported "
                        "instead, as a measurement of how much marginalising over the "
                        "other three scalars moves beta."},
            },
            "recovery": {
                "structural": {"closure_f1": 1.0, "structural_hamming": 0,
                               "min_probability_of_the_true_order":
                                   MIN_TRUE_ORDER_PROBABILITY},
                "scalars": {"truth_inside_95_credible_interval": list(SCALAR_ORDER)},
                "rho": "NOT APPLICABLE — U_TRUE is hand specified in "
                       "recurrent_synthetic.py, not drawn from p(U | rho), so no "
                       "generating rho exists. This is permanent and is not a gate.",
            },
        },
        "reported_but_not_gated": {
            "held_out_prediction":
                "§17. Following the Stage 6B convention, held-out numbers are secondary "
                "reporting and never drive a decision — they are not used to choose a "
                "proposal scale, a prior or a threshold.",
            "column_permutation_audit":
                "§13. h(U) is the intersection of the column orderings and Sigma_rho is "
                "column-exchangeable, so the target is invariant under permuting the d "
                "columns. A large signed-contrast R-hat with a small invariant-summary "
                "R-hat is label switching, not non-convergence, so it cannot be a gate. "
                "The symmetry's footprint is reported instead.",
            "entrywise_U_recovery":
                "not claimed anywhere in Stage 6D: the likelihood is piecewise constant "
                "in U and the target is column-exchangeable.",
        },
        "continuation_rule": {
            "initial": {"chains": 4, "sweeps": 30_000, "burn_in": 10_000, "thin": 5},
            "extend_in_blocks_of": 20_000, "ceiling": 100_000,
            "condition": "only if a primary convergence gate fails",
            "determinism": "a continuation re-runs from the same per-chain seeds with a "
                           "larger sweep count, which reproduces the earlier sweeps "
                           "bit-identically; failed attempts are preserved unmodified in "
                           "their own directories and named in continuation_history.json",
        },
        "parent_posteriors": parent_posteriors(),
        "predictive_draws": PREDICTIVE_DRAWS,
        "python": platform.python_version(), "numpy": np.__version__,
    }


# --------------------------------------------------------------------------- analysis
def load_run(run_dir: Path) -> dict:
    with np.load(run_dir / "chains.npz") as z:
        data = {k: z[k] for k in z.files}
    data["config"] = json.loads((run_dir / "config.json").read_text())
    data["scalar_diagnostics"] = json.loads(
        (run_dir / "scalar_diagnostics.json").read_text())
    data["structural_diagnostics"] = json.loads(
        (run_dir / "structural_diagnostics.json").read_text())
    return data


def convergence_section(run: dict) -> dict:
    """§15. Every coordinate, plus the log posterior and the structural summaries."""
    scalars = run["scalar_diagnostics"]
    out = {"per_coordinate": {}, "log_target": scalars["log_target"],
           "acceptance": scalars["acceptance"]}
    for name in ("rho",) + SCALAR_ORDER:
        entry = scalars[name]
        sd = entry.get("posterior_sd")
        mcse = entry.get("mcse")
        out["per_coordinate"][name] = {
            "posterior_mean": entry["posterior_mean"], "posterior_sd": sd,
            "q025": entry["q025"], "q975": entry["q975"],
            "rhat": entry.get("rhat"), "rhat_bulk": entry.get("rhat_bulk"),
            "rhat_tail": entry.get("rhat_tail"),
            "bulk_ess": entry.get("bulk_ess"), "tail_ess": entry.get("tail_ess"),
            "mcse": mcse,
            "mcse_over_sd": (mcse / sd if mcse is not None and sd else None),
            "degenerate": entry.get("degenerate"),
            "acceptance_total": entry.get("acceptance_total"),
            "acceptance_post_burn_in": entry.get("acceptance_post_burn_in"),
        }
    structural = run["structural_diagnostics"]
    out["relation_count"] = structural["relation_count_convergence"]
    out["uncertain_relations"] = {
        "n": structural["n_uncertain_relations"],
        "max_rhat": structural.get("max_relation_rhat"),
        "min_bulk_ess": structural.get("min_relation_bulk_ess"),
        "note": "a relation that never varies has no R-hat; averaging one in would "
                "flatter the report, so only the varying relations are scored",
    }
    out["n_h_states_visited"] = structural["n_h_states_visited"]
    return out


def recovery_section(run: dict, frozen, catalogue) -> dict:
    """§16. Structural, scalar and rho recovery, plus the parent-consistency contrasts."""
    u_by_chain = run["u_draws"].astype(float)
    pooled = u_by_chain.reshape(-1, u_by_chain.shape[2], u_by_chain.shape[3])
    ids = np.array([catalogue.index_of(precedence_from_u(u)) for u in pooled])
    ids_by_chain = ids.reshape(u_by_chain.shape[0], u_by_chain.shape[1])

    structural = recovery_metrics(ids, catalogue, frozen.u_true)
    modes = mode_summary(ids_by_chain, catalogue)

    scalars = run["scalar_diagnostics"]
    scalar_recovery = {}
    for name in SCALAR_ORDER:
        entry = scalars[name]
        truth = float(frozen.truth[name])
        scalar_recovery[name] = {
            "posterior_mean": entry["posterior_mean"],
            "posterior_sd": entry["posterior_sd"],
            "q025": entry["q025"], "q975": entry["q975"],
            "true_value": truth,
            "truth_in_95_interval": bool(entry["q025"] <= truth <= entry["q975"]),
            "error_in_posterior_sd": ((entry["posterior_mean"] - truth)
                                      / entry["posterior_sd"]),
        }
    return {
        "structural": structural,
        "modes": modes,
        "scalars": scalar_recovery,
        "rho": {
            "posterior_mean": scalars["rho"]["posterior_mean"],
            "posterior_sd": scalars["rho"]["posterior_sd"],
            "q025": scalars["rho"]["q025"], "q975": scalars["rho"]["q975"],
            "recovery": "NOT APPLICABLE — no generating value exists",
            "why": "U_TRUE is hand specified in recurrent_synthetic.py rather than drawn "
                   "from p(U | rho); no RHO_TRUE exists and none may be manufactured by "
                   "regenerating a more favourable dataset.",
        },
    }


def parent_consistency(run: dict, parents: dict) -> dict:
    """§16's confounding checks: does freeing U, rho and omega move anything?"""
    scalars = run["scalar_diagnostics"]
    out = {"vs_stage6b3": {}, "vs_stage6c2": {}, "structure_vs_stage6c2": {}}
    for name in SCALAR_ORDER:
        parent = parents["stage6b3"][name]
        difference = scalars[name]["posterior_mean"] - parent["mean"]
        out["vs_stage6b3"][name] = {
            "stage6d2_mean": scalars[name]["posterior_mean"],
            "stage6d2_sd": scalars[name]["posterior_sd"],
            "stage6b3_mean": parent["mean"], "stage6b3_sd": parent["sd"],
            "difference": difference,
            "difference_in_parent_sd": difference / parent["sd"],
            "sd_ratio": scalars[name]["posterior_sd"] / parent["sd"],
        }
    for name in ("rho", "beta"):
        parent = parents["stage6c2"][name]
        difference = scalars[name]["posterior_mean"] - parent["mean"]
        out["vs_stage6c2"][name] = {
            "stage6d2_mean": scalars[name]["posterior_mean"],
            "stage6d2_sd": scalars[name]["posterior_sd"],
            "stage6c2_mean": parent["mean"], "stage6c2_sd": parent["sd"],
            "difference": difference,
            "difference_in_parent_sd": difference / parent["sd"],
            "sd_ratio": scalars[name]["posterior_sd"] / parent["sd"],
            "is_a_gate": name == "rho",
            "note": (None if name == "rho" else
                     "reported contrast, not a gate: Stage 6C2 fixes omega, lambda_rep "
                     "and lambda_back while Stage 6D2 marginalises over them, so the "
                     "two beta marginals are not supposed to be equal. The Stage 6B3 "
                     "row above is the like-for-like comparison."),
        }
    out["parents_disagree_with_each_other"] = {
        "beta_stage6b3_vs_stage6c2_in_stage6b3_sd":
            (parents["stage6c2"]["beta"]["mean"] - parents["stage6b3"]["beta"]["mean"])
            / parents["stage6b3"]["beta"]["sd"],
        "reading": "the gap between the two parents is itself the size of the effect of "
                   "conditioning on omega and the lambdas rather than marginalising "
                   "over them; it is not a defect in either.",
    }
    out["interpretation"] = {
        "U_beta": "beta agreeing with Stage 6B3, which pinned U at the truth while "
                  "marginalising over the same three other scalars, means marginalising "
                  "over U does not move beta: no U-beta confounding.",
        "H_omega": "the induced order agreeing with Stage 6C2's, where omega was fixed, "
                   "means freeing omega does not move the structure: no H-omega "
                   "confounding.",
        "rho_U": "rho agreeing with Stage 6C2's means the extra three free scalars do "
                 "not reach rho, which is expected because rho enters only p(U | rho) "
                 "and the order posterior is unchanged.",
    }
    return out


def structure_vs_parent(run: dict, parents: dict, catalogue) -> dict:
    u_by_chain = run["u_draws"].astype(float)
    pooled = u_by_chain.reshape(-1, u_by_chain.shape[2], u_by_chain.shape[3])
    ids = np.array([catalogue.index_of(precedence_from_u(u)) for u in pooled])
    parent_index = int(parents["stage6c2_structure"]["true_poset_index"])
    probability = float(np.mean(ids == parent_index))
    return {
        "stage6c2_map_poset_index": parent_index,
        "stage6d2_probability_of_that_order": probability,
        "stage6c2_probability_of_that_order":
            parents["stage6c2_structure"]["posterior_probability_of_true"],
        "stage6d2_unique_states_visited": int(len(set(ids.tolist()))),
        "stage6c2_unique_states_visited":
            parents["stage6c2_structure"]["n_unique_states_visited"],
    }


def held_out_section(run: dict, frozen) -> dict:
    """§17. Posterior predictive on the frozen held-out split. Reported, never gated."""
    heldout = frozen.heldout
    u_by_chain = run["u_draws"].astype(float)
    pooled_u = u_by_chain.reshape(-1, u_by_chain.shape[2], u_by_chain.shape[3])
    n = pooled_u.shape[0]
    take = np.linspace(0, n - 1, min(PREDICTIVE_DRAWS, n)).astype(int)

    scalars = {name: run[name].ravel()[take] for name in SCALAR_ORDER}
    per_block = np.empty((take.size, heldout.shape[0]))
    for i, index in enumerate(take):
        per_block[i] = blockwise_recurrent_log_likelihood(
            heldout, pooled_u[index], float(scalars["beta"][i]), frozen.epsilon,
            float(scalars["omega"][i]), float(scalars["lambda_rep"][i]),
            float(scalars["lambda_back"][i]))
    shift = per_block.max(axis=0)
    predictive = shift + np.log(np.exp(per_block - shift).mean(axis=0))

    def score(u, values) -> float:
        return float(blockwise_recurrent_log_likelihood(
            heldout, u, values["beta"], frozen.epsilon, values["omega"],
            values["lambda_rep"], values["lambda_back"]).sum()) / heldout.size

    truth_values = {name: float(frozen.truth[name]) for name in SCALAR_ORDER}
    mean_values = {name: float(run[name].mean()) for name in SCALAR_ORDER}
    prior_values = {name: (PRIORS[name]["shape"] / PRIORS[name]["rate"]
                           if PRIORS[name]["family"] == "gamma"
                           else PRIORS[name]["mean"]) for name in SCALAR_ORDER}

    # A valid plug-in needs a U that *induces* the modal order, not the entrywise mean of
    # the draws. The modal order is identified from the draws alone — no truth is read.
    closures = np.array([precedence_from_u(u) for u in pooled_u])
    keys = [c.tobytes() for c in closures]
    modal_key = max(set(keys), key=keys.count)
    modal_index = keys.index(modal_key)
    modal_u = pooled_u[modal_index]

    # The entrywise mean, kept only as a negative control.
    entrywise_mean_u = pooled_u.mean(axis=0)
    mean_relations = int(precedence_from_u(entrywise_mean_u).sum())
    modal_relations = int(precedence_from_u(modal_u).sum())

    return {
        "n_blocks": int(heldout.shape[0]), "n_steps": int(heldout.size),
        "predictive_draws_used": int(take.size),
        "posterior_predictive_log_score_per_step":
            float(predictive.sum()) / heldout.size,
        "log_score_at_the_generating_truth": score(frozen.u_true, truth_values),
        "log_score_at_the_posterior_mean_scalars_and_a_modal_order_draw":
            score(modal_u, mean_values),
        "log_score_at_the_prior_mean_with_true_U": score(frozen.u_true, prior_values),
        "per_block_predictive_min": float(predictive.min() / heldout.shape[1]),
        "per_block_predictive_max": float(predictive.max() / heldout.shape[1]),
        "entrywise_posterior_mean_U_is_not_a_valid_plug_in": {
            "log_score": score(entrywise_mean_u, mean_values),
            "relations_induced_by_the_entrywise_mean": mean_relations,
            "relations_induced_by_a_modal_order_draw": modal_relations,
            "why": "averaging U entrywise across the posterior collapses the "
                   "incomparabilities: every draw realises the same partial order, but "
                   "their mean lands in a region where all coordinates are strictly "
                   "ordered, so h(mean U) is a denser order than h(U) for any draw. "
                   "The likelihood reads U only through h(U), so this plug-in scores a "
                   "different model. It is reported as a negative control — a concrete "
                   "demonstration of why entrywise U recovery is not claimed — and not "
                   "as a competitor.",
        },
        "note": "secondary reporting only; not used to choose a proposal scale, a prior "
                "or a threshold. The prior-mean row is a floor, not a competitor.",
    }


# ------------------------------------------------------------------------------ gates
def evaluate_gates(convergence: dict, recovery: dict, consistency: dict,
                   structure: dict) -> dict:
    gates = {}

    def add(name, value, threshold, passed, comparison, note=None):
        gates[name] = {"value": value, "threshold": threshold,
                       "comparison": comparison, "pass": bool(passed)}
        if note:
            gates[name]["note"] = note

    for name in ("rho",) + SCALAR_ORDER:
        r = convergence["per_coordinate"][name]["rhat"]
        add(f"{name}_rhat", r, RHAT_MAX, r is not None and r <= RHAT_MAX, "<=")
    lt = convergence["log_target"].get("rhat")
    add("log_posterior_rhat", lt, RHAT_MAX, lt is not None and lt <= RHAT_MAX, "<=")

    rc = convergence["relation_count"]
    if rc.get("degenerate"):
        add("relation_count_rhat", None, RHAT_MAX, True, "<=",
            "degenerate: the relation-count trace is constant at "
            f"{rc.get('constant_value')}, which is the point mass Stage 6C established "
            "on this corpus. R-hat is undefined, not 1.0. The point mass itself is "
            "gated by structural recovery.")
    else:
        add("relation_count_rhat", rc.get("rhat"), RHAT_MAX,
            rc.get("rhat") is not None and rc["rhat"] <= RHAT_MAX, "<=")

    ur = convergence["uncertain_relations"]
    if ur["n"] == 0:
        add("uncertain_relation_rhat", None, RHAT_MAX, True, "<=",
            "no relation varies across the retained draws, so there is no per-relation "
            "R-hat to compute; not applicable rather than passing on a vacuous 1.0")
    else:
        add("uncertain_relation_rhat", ur["max_rhat"], RHAT_MAX,
            ur["max_rhat"] is not None and ur["max_rhat"] <= RHAT_MAX, "<=")

    ess = {n: convergence["per_coordinate"][n]["bulk_ess"]
           for n in ("rho",) + SCALAR_ORDER}
    worst_ess = min(v for v in ess.values() if v is not None)
    add("min_bulk_ess", worst_ess, MIN_BULK_ESS, worst_ess >= MIN_BULK_ESS, ">=",
        f"worst coordinate: {min(ess, key=lambda k: ess[k])}")

    ratios = {n: convergence["per_coordinate"][n]["mcse_over_sd"]
              for n in ("rho",) + SCALAR_ORDER}
    worst_ratio = max(v for v in ratios.values() if v is not None)
    add("max_mcse_over_posterior_sd", worst_ratio, MAX_MCSE_OVER_SD,
        worst_ratio <= MAX_MCSE_OVER_SD, "<=",
        f"worst coordinate: {max(ratios, key=lambda k: ratios[k])}")

    acceptance = convergence["acceptance"]["post_burn_in"]
    outside = {k: v for k, v in acceptance.items()
               if not (ACCEPTANCE_BAND[0] <= v <= ACCEPTANCE_BAND[1])}
    gates["post_burn_in_acceptance_band"] = {
        "value": acceptance, "threshold": list(ACCEPTANCE_BAND),
        "comparison": "inside", "pass": not outside, "outside": outside}

    for name in SCALAR_ORDER:
        d = abs(consistency["vs_stage6b3"][name]["difference_in_parent_sd"])
        add(f"{name}_matches_stage6b3", d, PARENT_MEAN_TOLERANCE_IN_SD,
            d <= PARENT_MEAN_TOLERANCE_IN_SD, "<=")
    d = abs(consistency["vs_stage6c2"]["rho"]["difference_in_parent_sd"])
    add("rho_matches_stage6c2", d, PARENT_MEAN_TOLERANCE_IN_SD,
        d <= PARENT_MEAN_TOLERANCE_IN_SD, "<=",
        "rho enters only p(U | rho), so conditional on the same order posterior it "
        "cannot depend on the scalars Stage 6C2 held fixed")

    p = structure["stage6d2_probability_of_that_order"]
    add("structure_matches_stage6c2", p, MIN_TRUE_ORDER_PROBABILITY,
        p >= MIN_TRUE_ORDER_PROBABILITY, ">=",
        "freeing omega, lambda_rep and lambda_back must not move the induced order")

    closure = recovery["structural"]["closure"]
    add("structural_recovery_closure_f1", closure["f1"], 1.0, closure["f1"] == 1.0, "==")
    add("structural_recovery_hamming", closure["structural_hamming"], 0,
        closure["structural_hamming"] == 0, "==")
    pt = recovery["structural"]["posterior_probability_of_true"]
    add("probability_of_the_true_order", pt, MIN_TRUE_ORDER_PROBABILITY,
        pt >= MIN_TRUE_ORDER_PROBABILITY, ">=")
    for name in SCALAR_ORDER:
        inside = recovery["scalars"][name]["truth_in_95_interval"]
        gates[f"{name}_truth_in_95_interval"] = {
            "value": inside, "threshold": True, "comparison": "==", "pass": bool(inside)}

    failed = [k for k, g in gates.items() if not g["pass"]]
    return {"gates": gates, "n_gates": len(gates), "failed": failed,
            "all_passed": not failed,
            "convergence_gates_failed": [k for k in failed if k.endswith("_rhat")
                                         or k in ("min_bulk_ess",
                                                  "max_mcse_over_posterior_sd",
                                                  "post_burn_in_acceptance_band")]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    if args.register:
        REGISTRATION.parent.mkdir(parents=True, exist_ok=True)
        REGISTRATION.write_text(json.dumps(_jsonable(registration()), indent=2))
        print(f"registered -> {REGISTRATION}")
        return

    if not args.run_dir:
        raise SystemExit("give --register or --run-dir")
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not REGISTRATION.exists():
        raise SystemExit("the gate registration is missing; it must exist before a run "
                         "is analysed")
    reg = json.loads(REGISTRATION.read_text())
    parents = reg["parent_posteriors"]

    frozen = load_stage6d_dataset()
    run = load_run(run_dir)
    catalogue = build_catalogue(frozen.u_true.shape[0], frozen.u_true.shape[1])

    convergence = convergence_section(run)
    recovery = recovery_section(run, frozen, catalogue)
    consistency = parent_consistency(run, parents)
    structure = structure_vs_parent(run, parents, catalogue)
    consistency["structure_vs_stage6c2"] = structure
    audit = column_permutation_audit(run["u_draws"].astype(float))
    held_out = held_out_section(run, frozen)
    verdict = evaluate_gates(convergence, recovery, consistency, structure)

    (run_dir / "convergence_diagnostics.json").write_text(
        json.dumps(_jsonable(convergence), indent=2))
    (run_dir / "recovery_results.json").write_text(
        json.dumps(_jsonable(recovery), indent=2))
    (run_dir / "parent_consistency.json").write_text(
        json.dumps(_jsonable(consistency), indent=2))
    (run_dir / "column_permutation_audit.json").write_text(
        json.dumps(_jsonable(audit), indent=2))
    (run_dir / "heldout_prediction.json").write_text(
        json.dumps(_jsonable(held_out), indent=2))
    (run_dir / "gates.json").write_text(json.dumps(_jsonable({
        "registration_source": str(REGISTRATION.relative_to(ROOT)),
        "stage6d_config_hash": config_hash(), "source_commit": source_commit(),
        **verdict}), indent=2))

    for name, gate in verdict["gates"].items():
        value = gate["value"]
        shown = (f"{value:.6g}" if isinstance(value, float)
                 else ("n/a" if value is None else str(value)))
        print(f"[6d2] {name}: {shown} {gate['comparison']} {gate['threshold']} -> "
              f"{'PASS' if gate['pass'] else 'FAIL'}", flush=True)
    print(f"\n[6d2] {verdict['n_gates']} gates, "
          f"{'ALL PASS' if verdict['all_passed'] else 'FAILED: ' + str(verdict['failed'])}",
          flush=True)


if __name__ == "__main__":
    main()
