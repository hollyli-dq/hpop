"""Stage 6E2 — freeze the trace corpus, build the oracle control, run the discarded pilot.

    PYTHONPATH=src python scripts/stage6e2_corpus_and_pilot.py

Three things happen here, in this order, and none of them may look at a recovery quantity.

## 1. The corpus (§11) and its leakage audit

The Stage 6D corpus is 500 independent blocks with `K = 1` and no trace structure, so §11's
fallback applies and a trace-level corpus is generated and frozen. The generation seed is
fixed before any inference and is **not** searched. What the inference code receives is
`corpus.traces(...)` — the observed CPA occurrence sequences and nothing else;
`leakage_audit` checks that the frozen manifest's hidden truth is absent from what the
sampler is constructed with, rather than trusting the call sites.

## 2. The like-for-like oracle-boundary control (§12)

The Stage 6D2 result was computed on a *different* corpus, so comparing its numbers with
Stage 6E2's would be comparing two different problems. The control instead exposes the true
boundaries and labels of **these** traces and runs the same model with `(S, z)` pinned,
which is the only comparison that isolates the cost of making them latent.

## 3. The fresh pilot (§13)

Stage 6D2's scales are the pilot's **centre, not its conclusion** — a target with latent
`(S, z)` is a different target, and Stage 6D established that a proposal scale is a
property of the target rather than of the kernel. The pilot registers its grid, its
admissible band and its tie-break before running, and may look only at acceptance, ESJD,
invalid-proposal rates, finite-target checks, replay checks and cache consistency. Every
pilot draw is discarded.

**How ESJD is measured.** Rather than run 24 separate chains — one per (parameter,
multiplier) — the pilot runs one chain and, at each sweep, draws one proposal per candidate
multiplier and records its *expected* contribution:

    expected acceptance = min(1, alpha)
    expected ESJD       = min(1, alpha) * (jump in the kernel's own coordinate)^2

This is the standard Rao-Blackwellised estimator of the same two quantities the separate
chains would estimate, it uses the real corpus and the real target, and it consumes only
information §13 permits. The candidate proposals are *measured and discarded*; the chain
itself always advances using the base scale, so the shadow measurements cannot perturb the
trajectory they are measuring.

**Coordinates.** ESJD is measured in the coordinate each kernel actually walks in: `log`
for `beta`, `lambda_rep`, `lambda_back`; the identity for `omega`; `logit` for `rho`.
Measuring a log random walk's jump on the original scale would report a scale-dependent
number and would pick the wrong multiplier.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.fast_segmentation_kernel import (                    # noqa: E402
    FastSegmentationKernel, key_of, segmentation_of,
)
from hpop.mcmc_original.proposals import MoveType                            # noqa: E402
from hpop.mcmc_original.recurrent_scalar_mcmc import build_proposal          # noqa: E402
from hpop.mcmc_original.recurrent_scalar_posterior import log_prior          # noqa: E402
from hpop.mcmc_original.sampler_u import propose_row                         # noqa: E402
from hpop.mcmc_original.stage6c_frozen import (                              # noqa: E402
    log_jacobian_rho, log_rho_prior, log_structural_prior, rho_from_unconstrained,
    rho_to_unconstrained,
)
from hpop.mcmc_original.stage6d_frozen import (                              # noqa: E402
    REGISTERED_SCALES, SCALAR_ORDER, rho_is_in_support,
)
from hpop.mcmc_original.stage6e_block_table import assert_table_matches_scorer  # noqa: E402
from hpop.mcmc_original.stage6e_corpus import (                              # noqa: E402
    CORPUS_SEED, assert_distinct_orders, corpus_hash, exposure_audit_traces,
    generate_corpus,
)
from hpop.mcmc_original.stage6e_frozen import (                              # noqa: E402
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH, N_ROLES, N_SKILLS,
    assert_stage6d_unchanged, config_hash,
)
from hpop.mcmc_original.stage6e_sampler import (                             # noqa: E402
    SkillBlockLikelihood, Stage6ESampler, assert_evaluators_agree,
    boundary_hamming, occurrence_label_changes, segmentation_sweep, sweep_once,
)
from hpop.mcmc_original.stage6e_state import Stage6EModel, Stage6EState      # noqa: E402
from hpop.mcmc_original.transitions import log_transition_matrix             # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "stage6e2_unknown_boundary_full_seed0"

# ------------------------------------------------------- registered pilot definitions
#
# AMENDMENT 1, registered before any Stage 6E2 formal draw existed.
#
# The first pilot pass selected the **upper boundary** of the registered grid for all four
# scalar coordinates, and `lambda_rep` and `lambda_back` were still inadmissible there
# (expected acceptance 0.653 and 0.661, above the 0.60 ceiling). Four coordinates pinned
# against the same edge is evidence that the *search range* was truncated, not that the
# selection rule failed — so the range is extended and the rule is left exactly as it was.
#
# What the amendment does NOT change: the admissible band, the selection statistic, the
# tie-break, the ESJD coordinates, the permitted information, or the proposal-count study.
# What it does change: three more candidates per scalar coordinate.
#
# The existing rows are **preserved verbatim** rather than recomputed, and the new
# candidates are measured in their own pass with their own registered seed, so no recorded
# number moves as a side effect of adding candidates.
ORIGINAL_MULTIPLIER_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
EXTENDED_MULTIPLIERS = (16.0, 32.0, 64.0)
MULTIPLIER_GRID = ORIGINAL_MULTIPLIER_GRID + EXTENDED_MULTIPLIERS
EXTENSION_SEED_OFFSET = 500
ADMISSIBLE_ACCEPTANCE = (0.20, 0.60)
PROPOSAL_COUNT_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)      # of the mean trace length
# U and rho keep their Stage 6D definitions unless a pathology is demonstrated. The
# threshold for "pathology" is registered here, before the pilot runs, and is deliberately
# far wider than the scalar admissibility band: it is a fault detector, not a tuner.
PATHOLOGY_BAND = (0.05, 0.90)
PILOT_SWEEPS = 400
PILOT_COUNT_SWEEPS = 60
CONFIRMATION_SWEEPS = 400
PILOT_SEED = 6_053_101
ESJD_COORDINATE = {"beta": "log", "omega": "identity", "lambda_rep": "log",
                   "lambda_back": "log", "rho": "logit"}


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:                                                # pragma: no cover
        return "unknown"


def jsonable(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if math.isnan(v) else v
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def to_coordinate(name: str, value: float) -> float:
    kind = ESJD_COORDINATE[name]
    if kind == "log":
        return math.log(value)
    if kind == "logit":
        return rho_to_unconstrained(value)
    return float(value)


# --------------------------------------------------------------------- leakage audit
def leakage_audit(corpus, model: Stage6EModel) -> dict:
    """The inference model must contain the observations and nothing else.

    Checked structurally: every trace the model holds equals the corpus's observed role
    sequence, and no attribute of the model or of a freshly built state is equal to any
    hidden truth object. A boundary or label reaching the sampler would be a silent
    correctness failure that every downstream number would then inherit.
    """
    observed = [tuple(t.roles) for t in corpus.train]
    matches = [tuple(a) == tuple(b) for a, b in zip(model.traces, observed)]
    true_keys = {t.true_key() for t in corpus.train}
    exposed = []
    for attribute in vars(model):
        value = getattr(model, attribute)
        if isinstance(value, tuple) and value and isinstance(value[0], tuple):
            if any(v in true_keys for v in value):
                exposed.append(attribute)
    return {
        "n_traces": len(observed),
        "model_traces_equal_observed_sequences": bool(all(matches)),
        "model_holds_no_true_segmentation": not exposed,
        "exposed_attributes": exposed,
        "model_attributes": sorted(vars(model)),
        "true_boundaries_and_labels_live_only_in": "the frozen corpus manifest, used by "
                                                   "the recovery evaluation and by the "
                                                   "oracle-boundary control, never by the "
                                                   "unknown-boundary sampler",
        "pass": bool(all(matches) and not exposed),
    }


# ------------------------------------------------------------------- the pilot chain
def shadow_scalar_measurements(state, model, skill_ll, u, values, total_ll, rng,
                               multipliers=MULTIPLIER_GRID) -> dict:
    """One proposal per (scalar, multiplier), measured and discarded.

    `min(1, alpha)` is the expected acceptance of that proposal and
    `min(1, alpha) * (delta in the walk's own coordinate)^2` its expected squared jump.
    The chain does not move here; these are measurements only.
    """
    out = {}
    for name in SCALAR_ORDER:
        allow_cache = name != "omega"
        current = values[name]
        for multiplier in multipliers:
            proposal = build_proposal(name, REGISTERED_SCALES[name] * multiplier)(
                current, rng)
            prior = log_prior(name, proposal.value)
            if not math.isfinite(prior):
                out[(name, multiplier)] = (0.0, 0.0, 1)
                continue
            trial = dict(values)
            trial[name] = proposal.value
            evaluate = skill_ll.cached if allow_cache else skill_ll.full_replay
            candidate_ll = float(sum(
                evaluate(k, u[k], trial["beta"], trial["omega"], trial["lambda_rep"],
                         trial["lambda_back"])
                for k in range(model.n_skills)))
            if not math.isfinite(candidate_ll):
                # The extended multipliers reach parameter values where the registered
                # likelihood underflows to NaN. Such a proposal is rejected by the
                # sampler, so it must be recorded here as acceptance 0 and counted
                # invalid. Left unguarded it would be far worse than useless:
                # `min(0.0, NaN)` is `0.0`, so `exp(min(0, log_alpha))` would record the
                # proposal as acceptance **1.0** and make the most extreme multiplier
                # look like the best-mixing one.
                out[(name, multiplier)] = (0.0, 0.0, 1)
                continue
            log_alpha = ((prior + candidate_ll)
                         - (log_prior(name, current) + total_ll)
                         + proposal.log_q_reverse_minus_forward)
            acceptance = math.exp(min(0.0, log_alpha)) if math.isfinite(log_alpha) else 0.0
            jump = to_coordinate(name, proposal.value) - to_coordinate(name, current)
            out[(name, multiplier)] = (acceptance, acceptance * jump ** 2, 0)
    return out


def run_pilot(corpus, model: Stage6EModel, n_proposals: int, sweeps: int,
              multipliers=MULTIPLIER_GRID, seed: int = PILOT_SEED,
              measure_u_rho: bool = True) -> dict:
    """One pilot chain with shadow scale measurements. Every draw is discarded."""
    rng = np.random.default_rng(seed)
    sampler = Stage6ESampler(model=model, scales=dict(REGISTERED_SCALES),
                             n_proposals_per_trace=n_proposals, use_block_table=True)
    state = pilot_start(corpus, model, rng)
    skill_ll = SkillBlockLikelihood(traces=model.traces, epsilon=model.epsilon)

    records: dict = {(n, m): [] for n in SCALAR_ORDER for m in multipliers}
    invalid_records: dict = {(n, m): 0 for n in SCALAR_ORDER for m in multipliers}
    base_acceptance = {n: [0, 0] for n in ("U", "rho", *SCALAR_ORDER)}
    rho_shadow = {m: [] for m in ORIGINAL_MULTIPLIER_GRID}
    u_shadow = {m: [] for m in ORIGINAL_MULTIPLIER_GRID}
    finite_targets = True
    began = time.perf_counter()

    for sweep in range(sweeps):
        before = dict(state.proposed), dict(state.accepted)
        state = sweep_once(state, sampler, rng)
        for name in base_acceptance:
            base_acceptance[name][0] += state.proposed.get(name, 0) - before[0].get(name, 0)
            base_acceptance[name][1] += state.accepted.get(name, 0) - before[1].get(name, 0)
        finite_targets &= math.isfinite(state.components["log_target"])

        # shadow measurements at the state the sweep just produced
        skill_ll.set_blocks(state.segmentations, model.n_skills)
        u = np.asarray(state.u_by_skill, dtype=float)
        values = {n: float(getattr(state, n)) for n in SCALAR_ORDER}
        for k in range(model.n_skills):
            skill_ll.refresh(k, u[k], values["omega"])
        total_ll = float(sum(skill_ll.full_replay(k, u[k], values["beta"],
                                                  values["omega"], values["lambda_rep"],
                                                  values["lambda_back"])
                             for k in range(model.n_skills)))
        measured = shadow_scalar_measurements(state, model, skill_ll, u, values,
                                              total_ll, rng, multipliers)
        for key, (acceptance, esjd, invalid) in measured.items():
            records[key].append((acceptance, esjd))
            invalid_records[key] += invalid

        # rho and U shadow measurements, for the pathology check only. The amendment
        # extends the SCALAR grid only, so these keep the originally registered grid and
        # are skipped entirely in the extension pass.
        if not measure_u_rho:
            continue
        structural = {k: log_structural_prior(u[k], state.rho)
                      for k in range(model.n_skills)}
        z = rho_to_unconstrained(state.rho)
        for multiplier in ORIGINAL_MULTIPLIER_GRID:
            candidate = rho_from_unconstrained(
                z + REGISTERED_SCALES["rho"] * multiplier * rng.normal())
            if not rho_is_in_support(candidate):
                rho_shadow[multiplier].append((0.0, 0.0))
                continue
            candidate_structural = sum(log_structural_prior(u[k], candidate)
                                       for k in range(model.n_skills))
            log_alpha = ((candidate_structural - sum(structural.values()))
                         + (log_rho_prior(candidate) - log_rho_prior(state.rho))
                         + (log_jacobian_rho(candidate) - log_jacobian_rho(state.rho)))
            acceptance = math.exp(min(0.0, log_alpha))
            jump = rho_to_unconstrained(candidate) - z
            rho_shadow[multiplier].append((acceptance, acceptance * jump ** 2))
        for multiplier in ORIGINAL_MULTIPLIER_GRID:
            k = int(rng.integers(model.n_skills))
            row = int(rng.integers(u.shape[1]))
            candidate = propose_row(u[k], row, REGISTERED_SCALES["U"] * multiplier, rng)
            candidate_structural = log_structural_prior(candidate, state.rho)
            candidate_ll = skill_ll.full_replay(k, candidate, values["beta"],
                                                values["omega"], values["lambda_rep"],
                                                values["lambda_back"])
            current_ll = skill_ll.full_replay(k, u[k], values["beta"], values["omega"],
                                              values["lambda_rep"], values["lambda_back"])
            log_alpha = ((candidate_ll - current_ll)
                         + (candidate_structural - structural[k]))
            acceptance = math.exp(min(0.0, log_alpha))
            jump = float(np.sum((candidate[row] - u[k][row]) ** 2))
            u_shadow[multiplier].append((acceptance, acceptance * jump))

    runtime = time.perf_counter() - began
    return {
        "records": records, "invalid": invalid_records, "rho_shadow": rho_shadow,
        "u_shadow": u_shadow, "base_acceptance": base_acceptance,
        "finite_targets": finite_targets, "sweeps": sweeps, "runtime_seconds": runtime,
        "final_state": state, "sampler": sampler,
    }


def select_scales(pilot: dict, multipliers=MULTIPLIER_GRID) -> dict:
    """Registered rule: largest median ESJD among admissible, tie to multiplier near 1.

    Unchanged by the amendment. Only the set of candidates it ranges over is larger.
    """
    chosen = {}
    table = {}
    for name in SCALAR_ORDER:
        rows = []
        for multiplier in multipliers:
            values = pilot["records"][(name, multiplier)]
            acceptance = float(np.mean([a for a, _ in values]))
            esjd = float(np.median([e for _, e in values]))
            rows.append({"multiplier": multiplier,
                         "scale": REGISTERED_SCALES[name] * multiplier,
                         "expected_acceptance": acceptance,
                         "median_expected_esjd": esjd,
                         "esjd_coordinate": ESJD_COORDINATE[name],
                         "invalid_proposals": pilot["invalid"][(name, multiplier)],
                         "admissible": bool(ADMISSIBLE_ACCEPTANCE[0] <= acceptance
                                            <= ADMISSIBLE_ACCEPTANCE[1])})
        admissible = [r for r in rows if r["admissible"]]
        if admissible:
            best = max(r["median_expected_esjd"] for r in admissible)
            tied = [r for r in admissible
                    if r["median_expected_esjd"] >= best - 1e-15]
            tied.sort(key=lambda r: (abs(math.log(r["multiplier"])), r["multiplier"]))
            selection, reason = tied[0], "largest median expected ESJD among admissible"
        else:
            # No admissible candidate. The registered fallback is the multiplier whose
            # acceptance is closest to the admissible band, reported as such rather than
            # silently widening the band.
            def distance(r):
                a = r["expected_acceptance"]
                return max(ADMISSIBLE_ACCEPTANCE[0] - a, a - ADMISSIBLE_ACCEPTANCE[1], 0.0)
            selection = min(rows, key=lambda r: (distance(r), abs(math.log(r["multiplier"]))))
            reason = ("NO ADMISSIBLE CANDIDATE: the acceptance band was not met at any "
                      "multiplier; the closest is reported and used, and the band is NOT "
                      "widened")
        chosen[name] = selection["scale"]
        table[name] = {"grid": rows, "selected_multiplier": selection["multiplier"],
                       "selected_scale": selection["scale"], "selection_reason": reason,
                       "any_admissible": bool(admissible)}
    return {"scales": chosen, "table": table}


def pathology_check(pilot: dict) -> dict:
    """Do the Stage 6D `U` and `rho` definitions show a registered efficiency pathology?"""
    out = {}
    for name, shadow in (("rho", pilot["rho_shadow"]), ("U", pilot["u_shadow"])):
        rows = [{"multiplier": m,
                 "expected_acceptance": float(np.mean([a for a, _ in v])),
                 "median_expected_esjd": float(np.median([e for _, e in v]))}
                for m, v in shadow.items()]
        base = next(r for r in rows if r["multiplier"] == 1.0)
        pathological = not (PATHOLOGY_BAND[0] <= base["expected_acceptance"]
                            <= PATHOLOGY_BAND[1])
        out[name] = {"grid": rows, "base_expected_acceptance":
                     base["expected_acceptance"],
                     "pathology_band": list(PATHOLOGY_BAND),
                     "pathology_declared": bool(pathological),
                     "action": ("apply the multiplier grid" if pathological
                                else "keep the frozen Stage 6D definition unchanged")}
    return out


def proposal_count_study(corpus, model: Stage6EModel, counts) -> dict:
    """Movement per sweep and per second at each registered candidate count.

    Only movement and compute enter this: boundary-Hamming distance travelled,
    occurrence-label changes, distinct segmentations visited, acceptance and impossible
    rates by move type, and wall time. No recovery quantity is computed.
    """
    rows = []
    for count in counts:
        rng = np.random.default_rng(PILOT_SEED + 7)
        sampler = Stage6ESampler(model=model, scales=dict(REGISTERED_SCALES),
                                 n_proposals_per_trace=int(count), use_block_table=True)
        state = pilot_start(corpus, model, rng)
        proposed, accepted, invalid = {}, {}, {}
        # Distinct states are counted PER TRACE. A joint state over 100 traces is almost
        # surely new every sweep, so the joint count would just report the sweep count.
        visited = [set() for _ in model.traces]
        hamming = labels_changed = 0
        began = time.perf_counter()
        for _ in range(PILOT_COUNT_SWEEPS):
            before = tuple(key_of(s) for s in state.segmentations)
            state = sweep_once(state, sampler, rng)
            after = tuple(key_of(s) for s in state.segmentations)
            hamming += sum(boundary_hamming(a, b) for a, b in zip(before, after))
            labels_changed += sum(occurrence_label_changes(a, b)
                                  for a, b in zip(before, after))
            for n, key in enumerate(after):
                visited[n].add(key)
        runtime = time.perf_counter() - began
        for bucket, target in ((state.proposed, proposed), (state.accepted, accepted),
                               (state.invalid, invalid)):
            for move in MoveType.ALL:
                target[move] = bucket.get(move, 0)
        rows.append({
            "proposals_per_trace": int(count),
            "sweeps": PILOT_COUNT_SWEEPS,
            "runtime_seconds": runtime,
            "seconds_per_sweep": runtime / PILOT_COUNT_SWEEPS,
            "boundary_hamming_per_sweep": hamming / PILOT_COUNT_SWEEPS,
            "boundary_hamming_per_second": hamming / runtime,
            "label_changes_per_sweep": labels_changed / PILOT_COUNT_SWEEPS,
            "mean_distinct_segmentations_per_trace": float(
                np.mean([len(v) for v in visited])),
            "acceptance_by_move": {m: (accepted[m] / proposed[m] if proposed[m] else None)
                                   for m in MoveType.ALL},
            "impossible_rate_by_move": {
                m: (invalid[m] / proposed[m] if proposed[m] else None)
                for m in MoveType.ALL},
        })
    best = max(rows, key=lambda r: r["boundary_hamming_per_second"])
    return {"grid": rows, "selected_proposals_per_trace": best["proposals_per_trace"],
            "selection_rule": "largest boundary-Hamming movement per SECOND — movement "
                              "and computational efficiency only, exactly as section 13 "
                              "requires. No recovery quantity, no truth, no R-hat.",
            "note": "movement per sweep necessarily increases with the proposal count; "
                    "movement per second is what decides, because a sweep is not a fixed "
                    "unit of compute."}


def pilot_start(corpus, model: Stage6EModel, rng) -> Stage6EState:
    """A random legal segmentation, never the truth. Used only by the discarded pilot."""
    segmentations = []
    for trace in model.traces:
        segmentations.append(segmentation_of(random_legal_key(len(trace),
                                                              model.n_skills, rng)))
    u = rng.normal(scale=1.0, size=(model.n_skills, model.n_roles, 2))
    return Stage6EState(segmentations=tuple(segmentations), u_by_skill=u, rho=0.4,
                        beta=1.2, omega=1.0, lambda_rep=0.7, lambda_back=0.4,
                        pi=np.full(model.n_skills, 1.0 / model.n_skills),
                        transition=uniform_transition(model.n_skills))


def uniform_transition(n_skills: int) -> np.ndarray:
    p = np.full((n_skills, n_skills), 1.0 / (n_skills - 1))
    np.fill_diagonal(p, 0.0)
    return p


def random_legal_key(length: int, n_skills: int, rng) -> tuple:
    """A uniform-ish legal segmentation: widths drawn until the remainder is legal."""
    ends, start, previous = [], 0, None
    while start < length:
        remaining = length - start
        options = [w for w in range(MIN_BLOCK_WIDTH, min(MAX_BLOCK_WIDTH, remaining) + 1)
                   if remaining - w == 0 or remaining - w >= MIN_BLOCK_WIDTH]
        width = int(rng.choice(options))
        skill = int(rng.choice([k for k in range(n_skills) if k != previous]))
        start += width
        ends.append((start, skill))
        previous = skill
    return tuple(ends)


def joint_confirmation(corpus, model: Stage6EModel, scales: dict, n_proposals: int,
                       sweeps: int) -> dict:
    """A discarded confirmation run at the selected settings, before the formal chains."""
    rng = np.random.default_rng(PILOT_SEED + 99)
    sampler = Stage6ESampler(model=model, scales=dict(scales),
                             n_proposals_per_trace=n_proposals, use_block_table=True)
    state = pilot_start(corpus, model, rng)
    targets, finite, legal = [], True, True
    hamming = 0
    visited = [set() for _ in model.traces]
    began = time.perf_counter()
    for _ in range(sweeps):
        before = tuple(key_of(s) for s in state.segmentations)
        state = sweep_once(state, sampler, rng)
        after = tuple(key_of(s) for s in state.segmentations)
        hamming += sum(boundary_hamming(a, b) for a, b in zip(before, after))
        targets.append(state.components["log_target"])
        finite &= math.isfinite(state.components["log_target"])
        # each trace against ITS OWN kernel: the kernels differ by trace length, so
        # checking every key against kernel 0 would check the wrong coverage constraint
        for n, key in enumerate(after):
            visited[n].add(key)
            legal &= sampler._kernels[n].is_legal(key)

    # The table is refreshed at the START of a sweep, for the segmentation phase, and the
    # rest of the sweep then moves U, rho and the scalars — so at the end of a sweep the
    # table and the scorer are legitimately at different parameters and comparing them
    # there compares two different models. The invariant that matters, and the one the
    # sampler relies on, is that AT THE PARAMETERS THE TABLE WAS BUILT AT it equals the
    # registered per-block scorer. So both are put at the final state first.
    scorer = model.scorer_for(state)
    sampler._table.refresh(state.u_by_skill, state.beta, state.omega, state.lambda_rep,
                           state.lambda_back)
    table_parity = assert_table_matches_scorer(sampler._table, scorer, limit=2000)
    evaluator_parity = assert_evaluators_agree(model, state)
    acceptance = {n: (state.accepted.get(n, 0) / state.proposed[n]
                      if state.proposed.get(n) else None)
                  for n in state.proposed}
    checks = {
        "all_targets_finite": bool(finite),
        "every_segmentation_legal": bool(legal),
        "block_table_matches_registered_scorer": bool(table_parity["pass"]),
        "grouped_and_per_block_evaluators_agree": bool(evaluator_parity["pass"]),
        "boundaries_moved": bool(hamming > 0),
        "every_trace_visited_more_than_one_segmentation": bool(
            all(len(v) > 1 for v in visited)),
        "every_move_type_proposed": bool(
            all(state.proposed.get(m, 0) > 0 for m in MoveType.ALL)),
        "every_move_type_accepted": bool(
            all(state.accepted.get(m, 0) > 0 for m in MoveType.ALL)),
    }
    failed = [k for k, v in checks.items() if not v]
    return {
        "sweeps": sweeps, "runtime_seconds": time.perf_counter() - began,
        "checks": checks, "failed_checks": failed, "all_passed": not failed,
        "acceptance": acceptance,
        "boundary_hamming_total": hamming,
        "mean_distinct_segmentations_per_trace": float(
            np.mean([len(v) for v in visited])),
        "min_distinct_segmentations_over_traces": int(min(len(v) for v in visited)),
        "log_target_range": [float(min(targets)), float(max(targets))],
        "block_table_parity": table_parity,
        "evaluator_parity": evaluator_parity,
        "draws_discarded": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-sweeps", type=int, default=PILOT_SWEEPS)
    parser.add_argument("--count-sweeps", type=int, default=PILOT_COUNT_SWEEPS)
    parser.add_argument("--confirmation-sweeps", type=int, default=CONFIRMATION_SWEEPS)
    args = parser.parse_args()
    globals()["PILOT_COUNT_SWEEPS"] = args.count_sweeps

    OUT.mkdir(parents=True, exist_ok=True)
    assert_stage6d_unchanged()

    # ---- 1. corpus ---------------------------------------------------------------------
    corpus = generate_corpus()
    traces = corpus.traces("train")
    model = Stage6EModel(traces=traces, epsilon=corpus.epsilon, delta_b=corpus.delta_b,
                         n_skills=N_SKILLS, n_roles=N_ROLES,
                         min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
                         infer_pi_P=True)
    audit = leakage_audit(corpus, model)
    lengths = [t.length for t in corpus.train]
    widths = [e - s for t in corpus.train
              for s, e in zip([0] + list(t.true_boundaries),
                              list(t.true_boundaries) + [t.length])]
    manifest = {
        "corpus_hash": corpus_hash(corpus),
        "config": corpus.config,
        "n_train_traces": len(corpus.train), "n_heldout_traces": len(corpus.heldout),
        "n_train_blocks": corpus.n_train_blocks,
        "n_heldout_blocks": corpus.n_heldout_blocks,
        "trace_length": {"mean": float(np.mean(lengths)), "min": int(min(lengths)),
                         "max": int(max(lengths))},
        "heldout_trace_length_mean": float(np.mean([t.length for t in corpus.heldout])),
        "block_width": {"mean": float(np.mean(widths)), "min": int(min(widths)),
                        "max": int(max(widths))},
        "blocks_per_trace": {"mean": float(np.mean([t.n_blocks for t in corpus.train])),
                             "support": list(corpus.config
                                             ["blocks_per_trace_support"])},
        "skill_block_counts": {int(k): int(sum(t.true_labels.count(k)
                                               for t in corpus.train))
                               for k in range(N_SKILLS)},
        "traces_with_a_repeated_skill": int(sum(
            1 for t in corpus.train if len(set(t.true_labels)) < len(t.true_labels))),
        "exposure_audit_train": exposure_audit_traces(corpus, "train"),
        "exposure_audit_heldout": exposure_audit_traces(corpus, "heldout"),
        "induced_orders": assert_distinct_orders(),
        "leakage_audit": audit,
        "observed_train": [list(t.roles) for t in corpus.train],
        "observed_heldout": [list(t.roles) for t in corpus.heldout],
        "hidden_true_boundaries_train": [list(t.true_boundaries) for t in corpus.train],
        "hidden_true_labels_train": [list(t.true_labels) for t in corpus.train],
        "hidden_true_boundaries_heldout": [list(t.true_boundaries)
                                           for t in corpus.heldout],
        "hidden_true_labels_heldout": [list(t.true_labels) for t in corpus.heldout],
        "hidden_truth_use": "recovery evaluation and the oracle-boundary control only",
    }
    (OUT / "corpus_manifest.json").write_text(json.dumps(jsonable(manifest), indent=2))
    print(f"[6E2] corpus: {len(corpus.train)} train traces / {corpus.n_train_blocks} "
          f"blocks, {len(corpus.heldout)} held-out / {corpus.n_heldout_blocks} blocks; "
          f"J mean {np.mean(lengths):.1f}; leakage audit "
          f"{'PASS' if audit['pass'] else 'FAIL'}")
    if not audit["pass"]:
        raise SystemExit(f"leakage audit FAILED: {audit}")

    # ---- 2. the pilot ------------------------------------------------------------------
    mean_length = float(np.mean(lengths))
    counts = sorted({max(1, int(round(mean_length * m)))
                     for m in PROPOSAL_COUNT_MULTIPLIERS})
    registration = {
        "registered_before_any_pilot_draw_existed": True,
        "amended_before_any_formal_draw_existed": True,
        "stage": "6E2 pilot",
        "source_commit": source_commit(),
        "starting_scales": dict(REGISTERED_SCALES),
        "starting_scales_status": "the frozen Stage 6D2 scales, used as the pilot's "
                                  "CENTRE only. A target with latent (S, z) is a "
                                  "different target and carries no efficiency guarantee.",
        "multiplier_grid": list(MULTIPLIER_GRID),
        "amendments": [{
            "id": 1,
            "registered_before_any_stage6e2_formal_draw_existed": True,
            "what_changed": "the candidate multiplier grid only, from "
                            f"{list(ORIGINAL_MULTIPLIER_GRID)} to "
                            f"{list(MULTIPLIER_GRID)}",
            "applies_to": list(SCALAR_ORDER),
            "evidence": "all four scalar coordinates selected the grid's upper boundary "
                        "x8, and lambda_rep and lambda_back were still inadmissible "
                        "there (expected acceptance 0.653 and 0.661 against a 0.60 "
                        "ceiling). Four coordinates pinned to the same edge is evidence "
                        "that the search RANGE was truncated, not that the rule failed.",
            "what_did_not_change": ["admissible acceptance band",
                                    "selection statistic (largest median expected ESJD)",
                                    "tie-break", "ESJD coordinates",
                                    "permitted and forbidden information",
                                    "proposal-count study and its selection",
                                    "U and rho policy and their grid"],
            "existing_rows": "preserved verbatim — pass 1 reruns the originally "
                             "registered grid with its original seed, and the new "
                             "candidates are measured in a second pass with their own "
                             "registered seed, so adding candidates cannot move a number "
                             "that was already recorded",
            "extension_seed": PILOT_SEED + EXTENSION_SEED_OFFSET,
            "fallback_if_still_inadmissible_at_x64": "the previously registered "
                                                     "nearest-band fallback; tuning then "
                                                     "stops and the band is NOT widened",
        }],
        "original_multiplier_grid": list(ORIGINAL_MULTIPLIER_GRID),
        "extended_multipliers": list(EXTENDED_MULTIPLIERS),
        "admissible_acceptance": list(ADMISSIBLE_ACCEPTANCE),
        "selection": "largest median expected ESJD among admissible candidates",
        "tie_break": "multiplier closest to 1, then the smaller multiplier",
        "esjd_coordinates": dict(ESJD_COORDINATE),
        "esjd_estimator": "expected ESJD, min(1, alpha) * (coordinate jump)^2, from one "
                          "measured-and-discarded proposal per candidate per sweep",
        "proposal_count_candidates": counts,
        "proposal_count_multipliers": list(PROPOSAL_COUNT_MULTIPLIERS),
        "proposal_count_selection": "largest boundary-Hamming movement per second",
        "U_rho_policy": "keep the frozen Stage 6D definitions unless a pathology is "
                        f"demonstrated: base acceptance outside {PATHOLOGY_BAND}",
        "permitted_pilot_information": ["acceptance", "ESJD", "invalid-proposal rates",
                                        "finite-target checks", "recurrent replay checks",
                                        "cache consistency", "movement", "wall time"],
        "forbidden_pilot_information": ["boundary F1", "skill ARI", "structural recovery",
                                        "generating truth", "held-out NLL",
                                        "posterior means", "credible intervals",
                                        "candidate-run R-hat"],
        "all_pilot_draws_discarded": True,
    }
    (OUT / "pilot_registration.json").write_text(
        json.dumps(jsonable(registration), indent=2))

    print(f"[6E2] proposal-count study over {counts} ...", flush=True)
    counts_result = proposal_count_study(corpus, model, counts)
    selected_count = counts_result["selected_proposals_per_trace"]
    for row in counts_result["grid"]:
        print(f"       n_prop={row['proposals_per_trace']:4d}  "
              f"{row['seconds_per_sweep']*1000:7.1f} ms/sweep  "
              f"hamming/sweep {row['boundary_hamming_per_sweep']:6.2f}  "
              f"hamming/sec {row['boundary_hamming_per_second']:7.2f}  "
              f"states/trace {row['mean_distinct_segmentations_per_trace']:.1f}")
    print(f"[6E2] selected proposals per trace: {selected_count}")

    # ---- pass 1: the originally registered grid, unchanged --------------------------
    print(f"[6E2] scale pilot pass 1 (registered grid "
          f"{list(ORIGINAL_MULTIPLIER_GRID)}), {args.pilot_sweeps} sweeps ...", flush=True)
    pilot = run_pilot(corpus, model, selected_count, args.pilot_sweeps,
                      multipliers=ORIGINAL_MULTIPLIER_GRID, seed=PILOT_SEED,
                      measure_u_rho=True)

    # ---- pass 2: AMENDMENT 1, the extended candidates only ---------------------------
    # Registered before any Stage 6E2 formal draw existed, on the evidence that all four
    # scalar coordinates selected the grid's upper boundary and two were still above the
    # admissible ceiling there. Pass 1's rows are carried through untouched, so extending
    # the search cannot move a number that was already recorded; only new rows appear.
    print(f"[6E2] scale pilot pass 2 (AMENDMENT 1, extended candidates "
          f"{list(EXTENDED_MULTIPLIERS)}), {args.pilot_sweeps} sweeps ...", flush=True)
    extension = run_pilot(corpus, model, selected_count, args.pilot_sweeps,
                          multipliers=EXTENDED_MULTIPLIERS,
                          seed=PILOT_SEED + EXTENSION_SEED_OFFSET, measure_u_rho=False)
    pilot["records"].update(extension["records"])
    pilot["invalid"].update(extension["invalid"])
    pilot["finite_targets"] = bool(pilot["finite_targets"]
                                   and extension["finite_targets"])
    pilot["extension_runtime_seconds"] = extension["runtime_seconds"]

    selection = select_scales(pilot, MULTIPLIER_GRID)
    pathology = pathology_check(pilot)
    for name in SCALAR_ORDER:
        entry = selection["table"][name]
        print(f"       {name:12s} -> multiplier {entry['selected_multiplier']:>5} "
              f"scale {entry['selected_scale']:.5f}  ({entry['selection_reason'][:50]})")
    for name, entry in pathology.items():
        print(f"       {name:12s} base acceptance "
              f"{entry['base_expected_acceptance']:.3f} -> {entry['action']}")

    final_scales = dict(REGISTERED_SCALES)
    final_scales.update(selection["scales"])
    for name, entry in pathology.items():
        if entry["pathology_declared"]:
            admissible = [r for r in entry["grid"]
                          if ADMISSIBLE_ACCEPTANCE[0] <= r["expected_acceptance"]
                          <= ADMISSIBLE_ACCEPTANCE[1]]
            if admissible:
                best = max(admissible, key=lambda r: r["median_expected_esjd"])
                final_scales[name] = REGISTERED_SCALES[name] * best["multiplier"]

    pilot_results = {
        "proposal_count_study": counts_result,
        "scalar_grid": selection["table"],
        "U_rho_pathology_check": pathology,
        "selected_scales": final_scales,
        "selected_proposals_per_trace": selected_count,
        "base_acceptance_observed": {
            n: (v[1] / v[0] if v[0] else None)
            for n, v in pilot["base_acceptance"].items()},
        "all_targets_finite": pilot["finite_targets"],
        "pilot_sweeps": pilot["sweeps"],
        "pilot_runtime_seconds": pilot["runtime_seconds"],
        "all_pilot_draws_discarded": True,
    }

    # ---- 3. discarded joint confirmation ------------------------------------------------
    print(f"[6E2] joint confirmation, {args.confirmation_sweeps} sweeps ...", flush=True)
    confirmation = joint_confirmation(corpus, model, final_scales, selected_count,
                                      args.confirmation_sweeps)
    pilot_results["joint_confirmation_summary"] = confirmation["checks"]
    (OUT / "pilot_results.json").write_text(json.dumps(jsonable(pilot_results), indent=2))
    (OUT / "joint_confirmation.json").write_text(
        json.dumps(jsonable(confirmation), indent=2))

    config = {
        "stage": "6E2", "source_commit": source_commit(),
        "stage6e_config_hash": config_hash(),
        "python": platform.python_version(), "numpy": np.__version__,
        "corpus_hash": manifest["corpus_hash"], "corpus_seed": CORPUS_SEED,
        "model": {"n_skills": N_SKILLS, "n_roles": N_ROLES, "d_latent": 2,
                  "epsilon": corpus.epsilon, "delta_B": DELTA_B,
                  "min_width": MIN_BLOCK_WIDTH, "max_width": MAX_BLOCK_WIDTH,
                  "infer_pi_P": True},
        "selected_scales": final_scales,
        "selected_proposals_per_trace": selected_count,
        "sweep_order": ["(S,z)", "(pi,P)", "U", "rho", "beta", "omega", "lambda_rep",
                        "lambda_back"],
    }
    (OUT / "config.json").write_text(json.dumps(jsonable(config), indent=2))

    print(f"[6E2] joint confirmation "
          f"{sum(confirmation['checks'].values())}/{len(confirmation['checks'])} checks "
          f"-> {'PASS' if confirmation['all_passed'] else 'FAIL'} "
          f"{confirmation['failed_checks']}")
    print(f"[6E2] wrote {OUT}")
    if not confirmation["all_passed"]:
        raise SystemExit(f"joint confirmation FAILED: {confirmation['failed_checks']}")


if __name__ == "__main__":
    main()
