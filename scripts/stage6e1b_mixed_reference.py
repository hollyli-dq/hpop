"""Stage 6E1B — the mixed unknown-boundary reference, and four dispersed joint chains.

    PYTHONPATH=src python scripts/stage6e1b_mixed_reference.py [--stage reference|chains|all]

Stage 6E1A fixed every continuous coordinate and asked whether the move kernel targets the
right distribution over `(S, z)`. Stage 6E1B frees `U`, `rho` and the four recurrent
scalars as well, so the object under test is the *complete mixed* posterior

    p(S, z, U, rho, beta, omega, lambda_rep, lambda_back | x)

and the reference must represent both the discrete and the continuous parts at once.

## Construction

Scrambled Sobol in **prior coordinates** (the validated Stage 6D1 strategy), so the
proposal is exactly the joint prior and the importance weight carries no prior density.
What the weight does carry is the *marginal* segmentation likelihood: for each QMC point,
every legal `(S, z)` is enumerated and summed, giving `Z_n(theta)` in closed form. That is
what makes a mixed reference affordable here — there are only `O(J^2 K)` distinct candidate
blocks, they are cached within a draw, and the enumerated state set has 21 members per
trace.

## `pi` and `P` are fixed for this reference, and deliberately asymmetric

§9 permits fixing them, and they are fixed. They are also chosen with **distinct rows and a
non-uniform `pi`**, which is not cosmetic: with a uniform `pi` and a symmetric `P`, relabelling
the skills together with their `U_k` would be an exact symmetry of the target, every
per-skill summary would be unidentified, and the reference/MCMC comparison of per-skill
`H` would be meaningless. `label_permutation_audit` verifies that no nontrivial
relabelling is a symmetry before any comparison is made.

## Gates

The reference is frozen against the **corrected** Stage 6D RQMC protocol: the primary
precision statistic is `rqmc_se = sd(replicate estimates, ddof=1)/sqrt(R)` and its t-based
half-width. The maximum-over-replicates statistics are computed and reported as superseded
descriptive diagnostics; their thresholds are shown and their failures are reported as
failures, never relabelled.
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

from hpop.mcmc_original.fast_segmentation_kernel import key_of, segmentation_of  # noqa: E402
from hpop.mcmc_original.latent_poset import precedence_from_u                 # noqa: E402
from hpop.mcmc_original.recurrent_rfs import (                                # noqa: E402
    RecurrentRFSParameters, recurrent_step_probabilities, recurrent_validity_update,
)
from hpop.mcmc_original.recurrent_scalar_posterior import (                   # noqa: E402
    cached_batch_log_likelihood,
)
from hpop.mcmc_original.stage6c_diagnostics import convergence_block          # noqa: E402
from hpop.mcmc_original.stage6d_frozen import REGISTERED_SCALES, SCALAR_ORDER # noqa: E402
from hpop.mcmc_original.stage6e_exact import total_variation                  # noqa: E402
from hpop.mcmc_original.stage6e_frozen import (                               # noqa: E402
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH, assert_stage6d_unchanged, config_hash,
)
from hpop.mcmc_original.stage6e_mixed_reference import (                      # noqa: E402
    MixedModel, SCALAR_NAMES, combine_mixed_replicates, h_label_of,
    label_permutation_audit, mixed_replicate, mixed_replicate_summary,
)
from hpop.mcmc_original.stage6e_sampler import run_stage6e_chain              # noqa: E402
from hpop.mcmc_original.stage6e_state import Stage6EModel, Stage6EState       # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "stage6e1b_mixed_reference"

# ----------------------------------------------------------------- registered problem
J = 8
K_SKILLS = 3
M_ROLES = 3
D_LATENT = 2
N_TRACES = 2
EPSILON = 0.02
CORPUS_SEED = 6_052_000

# Deliberately asymmetric: no relabelling is a symmetry, so per-skill summaries are
# identified. Registered before any draw.
PI_FIXED = np.array([0.60, 0.30, 0.10])
P_FIXED = np.array([[0.00, 0.70, 0.30],
                    [0.25, 0.00, 0.75],
                    [0.80, 0.20, 0.00]])

U_GENERATING = np.array([
    [[2.0, 2.0], [1.0, 0.0], [0.0, 1.0]],
    [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0]],
    [[3.0, 3.0], [2.0, 2.0], [0.0, 0.0]],
], dtype=float)
TRUTH = {"beta": 1.5, "omega": 1.7346, "lambda_rep": 0.8, "lambda_back": 0.25}

# ------------------------------------------------------------------------- QMC config
N_REPLICATES = 16
N_POINTS = 2 ** 20
QMC_SEEDS = tuple(6_052_100 + i for i in range(N_REPLICATES))

# The corrected Stage 6D RQMC gates, carried over unchanged.
QUALITY_GATES = {
    "max_rqmc_standard_error": 0.001,
    "max_half_width_95": 0.0025,
    "min_relative_ess": 0.02,
    "max_normalised_weight": 0.001,
    "max_log_evidence_sd": 0.05,
}
SUPERSEDED_GATES = {
    "max_replicate_h_total_variation": 0.003,
    "max_replicate_relation_departure": 0.003,
}
NONDEGENERACY = {
    "segmentation_max_probability_below": 0.90,
    "segmentation_min_states_above_0.01": 3,
    "induced_h_min_states_above_0.01": 3,
}

# ----------------------------------------------------------------------- chain config
N_CHAINS = 4
# Attempt 0 ran 150,000 sweeps and cleared every gate except `induced_h_total_variation`,
# at 0.01050 against 0.01. The diagnosis is recorded in
# `results/.../stage6e1b_mixed_reference_FAILED_attempt0_150k/README.md`: bulk ESS was as
# low as 301 for `lambda_rep`, while every bias-sensitive statistic (five posterior means
# within 0.024 reference SD, the mixed multivariate energy statistic, ten R-hats) was
# clean. `TV(H)` sums 19 cell errors per skill, so it is the statistic most exposed to
# Monte Carlo noise, and at that effective size ~0.01 is the expected magnitude.
#
# The response is more draws and nothing else. The target, the kernel, the proposal
# scales, the reference and the 0.01 gate are all unchanged; only `n` moves, which is the
# one intervention that separates noise from bias rather than hiding the difference.
N_SWEEPS = 600_000
BURN_IN = 120_000
THIN = 10
PROPOSALS_PER_TRACE = J
CHAIN_SEEDS = (6_052_001, 6_052_002, 6_052_003, 6_052_004)


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
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


# ------------------------------------------------------------------ corpus generation
def generate_corpus() -> tuple:
    """Two traces of `J = 8`, generated by the registered equations from `q_0 = 0`."""
    rng = np.random.default_rng(CORPUS_SEED)
    params = RecurrentRFSParameters(beta=TRUTH["beta"], epsilon=EPSILON,
                                    shared_omega=TRUTH["omega"],
                                    lambda_rep=TRUTH["lambda_rep"],
                                    lambda_back=TRUTH["lambda_back"])
    traces, truths = [], []
    for _ in range(N_TRACES):
        widths = (4, 4)
        path = [int(rng.choice(K_SKILLS, p=PI_FIXED))]
        path.append(int(rng.choice(K_SKILLS, p=P_FIXED[path[0]])))
        roles = []
        for width, skill in zip(widths, path):
            u = U_GENERATING[skill]
            precedence = precedence_from_u(u)
            q = np.zeros(M_ROLES)
            for _ in range(width):
                mixed = recurrent_step_probabilities(u, q, params)
                y = int(rng.choice(M_ROLES, p=mixed))
                roles.append(y)
                q = recurrent_validity_update(y, precedence, q, params.shared_omega)
        traces.append(tuple(roles))
        ends, running = [], 0
        for w in widths:
            running += w
            ends.append(running)
        truths.append(tuple(zip(ends, path)))
    return tuple(traces), tuple(truths)


def build_mixed_model(traces) -> MixedModel:
    return MixedModel(traces=traces, n_skills=K_SKILLS, m=M_ROLES, d=D_LATENT,
                      epsilon=EPSILON, pi=PI_FIXED, transition=P_FIXED,
                      delta_b=DELTA_B, min_width=MIN_BLOCK_WIDTH,
                      max_width=MAX_BLOCK_WIDTH)


# -------------------------------------------------------------------- the reference
N_RETAINED_PER_REPLICATE = 500       # iid-equivalent draws, for the mixed statistic only


def build_reference(model: MixedModel) -> dict:
    summaries, retained = [], []
    began = time.perf_counter()
    for i, seed in enumerate(QMC_SEEDS):
        replicate = mixed_replicate(model, N_POINTS, seed,
                                    n_retained=N_RETAINED_PER_REPLICATE)
        summaries.append(mixed_replicate_summary(replicate, model))
        retained.append(replicate["retained"])
        print(f"[6E1B] replicate {i + 1}/{N_REPLICATES} seed {seed}: "
              f"rel ESS {replicate['relative_ess']:.4f}  "
              f"max w {replicate['max_normalised_weight']:.2e}  "
              f"log Z {replicate['log_evidence']:.6f}  "
              f"({time.perf_counter() - began:.0f}s)", flush=True)
    combined = combine_mixed_replicates(summaries, model)
    combined["runtime_seconds"] = time.perf_counter() - began
    combined["retained"] = {
        "closures": np.concatenate([r["closures"] for r in retained]),
        "sampled": np.concatenate([r["sampled"] for r in retained]),
        **{name: np.concatenate([r[name] for r in retained])
           for name in ("rho", *SCALAR_NAMES)},
    }
    return combined


def mixed_multivariate_coordinates(closures, sampled, scalars, model) -> np.ndarray:
    """One row per draw: [closure indicators over all skills, segment counts, scalars].

    A single vector spanning the discrete structure, the segmentation and the continuous
    coordinates, so the statistic below can detect a disagreement in their *joint*
    behaviour that every marginal comparison would miss.
    """
    n = closures.shape[0]
    counts = np.array([[len(model.states[j]) for j in row] for row in sampled],
                      dtype=float)
    return np.column_stack([closures.reshape(n, -1).astype(float), counts,
                            *[scalars[name] for name in ("rho", *SCALAR_NAMES)]])


def mixed_multivariate_statistic(reference_rows, mcmc_rows, seed: int = 5) -> dict:
    """Energy distance against an envelope calibrated on the reference against itself.

    Reused wholesale from the Stage 6B/6D machinery: `standardise` puts every coordinate
    on the reference's own scale, `energy_distance` is the statistic, and
    `calibrate_energy_envelope` gives the null distribution of that statistic when both
    samples genuinely come from the reference. Constant coordinates are dropped and
    counted rather than silently contributing zero.
    """
    from hpop.mcmc_original.stage6b_joint_diagnostics import (
        calibrate_energy_envelope, energy_distance, standardise,
    )
    reference_rows = np.asarray(reference_rows, dtype=float)
    mcmc_rows = np.asarray(mcmc_rows, dtype=float)
    spread = reference_rows.std(axis=0, ddof=1)
    keep = spread > 1e-12
    dropped = int((~keep).sum())
    reference_rows, mcmc_rows = reference_rows[:, keep], mcmc_rows[:, keep]
    centre = reference_rows.mean(axis=0)
    scale = reference_rows.std(axis=0, ddof=1)
    a = standardise(reference_rows, centre, scale)
    b = standardise(mcmc_rows, centre, scale)
    n_x = min(len(a) // 2, len(b))
    n_y = min(len(a) - n_x, len(b))
    observed = float(energy_distance(a[:n_x * 2:2], b[:n_y]))
    envelope = calibrate_energy_envelope(a, n_x=n_x, n_y=n_y, n_replicates=40, seed=seed)
    threshold = float(envelope["envelope"])
    return {
        "statistic": "energy distance on [closure indicators, segment counts, "
                     "standardised scalars]",
        "observed": observed, "envelope": threshold,
        "envelope_quantile": envelope["quantile"],
        "null_mean": float(envelope["mean"]), "null_sd": float(envelope["sd"]),
        "n_reference": int(n_x), "n_mcmc": int(n_y),
        "n_coordinates": int(keep.sum()), "dropped_constant_coordinates": dropped,
        "z_score": float((observed - envelope["mean"]) / max(1e-12, envelope["sd"])),
        "pass": bool(observed <= threshold),
    }


def reference_quality(combined: dict, model: MixedModel) -> dict:
    checks = {
        "max_rqmc_standard_error": {
            "value": combined["precision"]["max_rqmc_standard_error"],
            "threshold": QUALITY_GATES["max_rqmc_standard_error"], "primary": True},
        "max_half_width_95": {
            "value": combined["precision"]["max_half_width_95"],
            "threshold": QUALITY_GATES["max_half_width_95"], "primary": True},
        "min_relative_ess": {
            "value": combined["relative_ess"]["min"],
            "threshold": QUALITY_GATES["min_relative_ess"], "primary": False,
            "comparison": ">="},
        "max_normalised_weight": {
            "value": combined["max_normalised_weight"]["max"],
            "threshold": QUALITY_GATES["max_normalised_weight"], "primary": False},
        "log_evidence_sd": {
            "value": combined["log_evidence"]["sd"],
            "threshold": QUALITY_GATES["max_log_evidence_sd"], "primary": False},
    }
    for name, check in checks.items():
        if check.get("comparison") == ">=":
            check["pass"] = bool(check["value"] >= check["threshold"])
        else:
            check["pass"] = bool(check["value"] <= check["threshold"])

    superseded = {}
    for name, threshold in SUPERSEDED_GATES.items():
        value = combined["superseded_descriptive"][name]
        superseded[name] = {"value": value, "threshold": threshold,
                            "pass": bool(value <= threshold),
                            "status": "SUPERSEDED DESCRIPTIVE DIAGNOSTIC — not a "
                                      "precision measure and not a primary gate"}

    # nondegeneracy
    segmentation = combined["pooled_segmentation"]
    max_p = float(segmentation.max())
    n_above = int((segmentation > 0.01).sum(axis=1).min())
    h_above = [int((p > 0.01).sum()) for p in combined["pooled_h_probability"]]
    nondegenerate = {
        "segmentation_max_probability": {
            "value": max_p,
            "threshold": NONDEGENERACY["segmentation_max_probability_below"],
            "pass": bool(max_p < NONDEGENERACY["segmentation_max_probability_below"])},
        "segmentation_states_above_0.01": {
            "value": n_above,
            "threshold": NONDEGENERACY["segmentation_min_states_above_0.01"],
            "pass": bool(n_above >= NONDEGENERACY["segmentation_min_states_above_0.01"])},
        "induced_h_states_above_0.01": {
            "value": min(h_above), "per_skill": h_above,
            "threshold": NONDEGENERACY["induced_h_min_states_above_0.01"],
            "pass": bool(min(h_above)
                         >= NONDEGENERACY["induced_h_min_states_above_0.01"])},
    }
    return {
        "checks": checks, "superseded_checks": superseded,
        "nondegeneracy": nondegenerate,
        "primary_pass": all(c["pass"] for c in checks.values() if c["primary"]),
        "all_active_pass": all(c["pass"] for c in checks.values()),
        "nondegenerate_pass": all(c["pass"] for c in nondegenerate.values()),
        "label_permutation_audit": label_permutation_audit(model),
        "gate_rationale": {
            "max_rqmc_standard_error":
                "rqmc_se = sd(replicate_estimates, ddof=1)/sqrt(R) is the uncertainty of "
                "the quantity the comparison consumes, the replicate mean. Independent "
                "scrambles make the per-replicate estimates iid, which licenses it.",
            "max_half_width_95":
                "t(0.975, R-1) * rqmc_se. At 2.5e-3 the reference's own 95% uncertainty "
                "occupies at most a quarter of the 0.01 error budget it feeds.",
            "superseded":
                "A maximum over replicates estimates the dispersion of a SINGLE "
                "replicate. It does not shrink as R grows and is not an uncertainty for "
                "the replicate mean. Superseded in Stage 6D1; retained here as a "
                "descriptive diagnostic whose failures are reported as failures.",
            "estimator_choice":
                "The conditional (Rao-Blackwellised) estimator is primary: it uses the "
                "exact conditional p(S,z | theta) rather than one multinomial draw from "
                "it, so it estimates the same quantity with strictly lower variance. The "
                "section 9 sampled estimator is computed and gated alongside it.",
        },
    }


# ------------------------------------------------------------------------- the chains
def dispersed_starts(model: MixedModel) -> list:
    """Four structurally distinct legal starts. None is the generating segmentation."""
    coarse = ((J, 0),)
    fine_a = ((3, 0), (J, 1))
    fine_b = ((5, 2), (J, 0))
    middle = ((4, 1), (J, 2))
    shapes = [
        [fine_a, fine_b],       # chain 0: fine, varied labels
        [coarse, coarse],       # chain 1: coarse
        [middle, fine_a],       # chain 2: mixed
        [fine_b, middle],       # chain 3: structurally distinct intermediate
    ]
    rho_starts = [0.10, 0.35, 0.60, 0.85]
    scalar_starts = [
        {"beta": 0.60, "omega": -1.20, "lambda_rep": 0.30, "lambda_back": 0.80},
        {"beta": 1.20, "omega": 0.50, "lambda_rep": 0.90, "lambda_back": 0.20},
        {"beta": 2.20, "omega": 2.40, "lambda_rep": 1.60, "lambda_back": 1.30},
        {"beta": 3.00, "omega": 4.00, "lambda_rep": 2.40, "lambda_back": 0.05},
    ]
    starts = []
    for chain in range(N_CHAINS):
        rng = np.random.default_rng(90_000 + chain)
        u = rng.normal(scale=1.5, size=(K_SKILLS, M_ROLES, D_LATENT))
        starts.append(Stage6EState(
            segmentations=tuple(segmentation_of(k) for k in shapes[chain]),
            u_by_skill=u, rho=rho_starts[chain], pi=PI_FIXED, transition=P_FIXED,
            **scalar_starts[chain]))
    return starts


def _chain_worker(payload: dict) -> dict:
    """One chain, in its own process. Rebuilds the model rather than unpickling it.

    Every run-length parameter arrives in `payload`. A `spawn` worker re-imports this
    module from scratch, so anything `main` wrote into `globals()` — including a `--sweeps`
    override — would silently not reach it, and the worker would run the module default
    while the log reported the override.
    """
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    chain = payload["chain"]
    traces, _ = generate_corpus()
    model = build_mixed_model(traces)
    start = dispersed_starts(model)[chain]
    began = time.perf_counter()
    result = run_stage6e_chain(
        model=Stage6EModel(traces=model.traces, epsilon=EPSILON, delta_b=DELTA_B,
                           n_skills=K_SKILLS, n_roles=M_ROLES,
                           min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
                           infer_pi_P=False),
        start=start, scales=REGISTERED_SCALES,
        n_proposals_per_trace=payload["n_proposals"], num_sweeps=payload["num_sweeps"],
        burn_in=payload["burn_in"], thin=payload["thin"], seed=CHAIN_SEEDS[chain],
        chain=chain, store_keys=True)
    return {"chain": chain, "seed": CHAIN_SEEDS[chain],
            "u_draws": result.u_draws, "scalars": result.scalars,
            "log_target": result.log_target, "segment_counts": result.segment_counts,
            "relation_counts": result.relation_counts,
            "boundary_keys": result.boundary_keys, "movement": result.movement,
            "acceptance_rates": result.acceptance(),
            "runtime_seconds": time.perf_counter() - began}


class _ChainResult:
    """Minimal stand-in carrying exactly what `compare` reads off a chain.

    The acceptance rates arrive under `acceptance_rates`, not `acceptance`: writing a
    dict into `__dict__` under the same name as the method would shadow it, and the
    failure surfaces only at the very end of a multi-hour run.
    """

    def __init__(self, payload: dict):
        self.__dict__.update(payload)

    def acceptance(self, post_burn_in: bool = True) -> dict:      # noqa: D102
        return self.__dict__["acceptance_rates"]


def run_chains(model: MixedModel) -> list:
    """Four chains, one process each. Seeds and starts are per-chain, so parallel
    execution changes wall time and nothing about the trajectories."""
    from multiprocessing import get_context
    began = time.perf_counter()
    jobs = [{"chain": c, "num_sweeps": N_SWEEPS, "burn_in": BURN_IN, "thin": THIN,
             "n_proposals": PROPOSALS_PER_TRACE} for c in range(N_CHAINS)]
    print(f"[6E1B] {N_CHAINS} chains x {N_SWEEPS:,} sweeps, burn-in {BURN_IN:,}, "
          f"thin {THIN}, in parallel", flush=True)
    with get_context("spawn").Pool(processes=N_CHAINS) as pool:
        payloads = pool.map(_chain_worker, jobs)
    for payload in payloads:
        print(f"[6E1B] chain {payload['chain']} seed {payload['seed']}: "
              f"{len(payload['log_target']):,} retained in "
              f"{payload['runtime_seconds']:.0f}s  acceptance "
              f"{({k: round(v, 3) for k, v in payload['acceptance_rates'].items() if v == v})}",
              flush=True)
    print(f"[6E1B] {N_CHAINS} chains in {time.perf_counter() - began:.0f}s wall",
          flush=True)
    return [_ChainResult(p) for p in payloads]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=["reference", "chains", "all"])
    parser.add_argument("--replicates", type=int, default=N_REPLICATES)
    parser.add_argument("--points", type=int, default=N_POINTS)
    parser.add_argument("--sweeps", type=int, default=N_SWEEPS)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--thin", type=int, default=None)
    args = parser.parse_args()
    globals()["N_REPLICATES"] = args.replicates
    globals()["QMC_SEEDS"] = tuple(6_052_100 + i for i in range(args.replicates))
    globals()["N_POINTS"] = args.points
    registered_sweeps = N_SWEEPS          # read BEFORE the override overwrites it
    globals()["N_SWEEPS"] = args.sweeps
    if args.burn_in is not None:
        globals()["BURN_IN"] = args.burn_in
    elif args.sweeps != registered_sweeps:
        # a shortened smoke run must shorten its burn-in too, or it cannot start
        globals()["BURN_IN"] = max(1, args.sweeps // 5)
    if args.thin is not None:
        globals()["THIN"] = args.thin

    OUT.mkdir(parents=True, exist_ok=True)
    assert_stage6d_unchanged()

    traces, truths = generate_corpus()
    model = build_mixed_model(traces)
    audit = label_permutation_audit(model)
    print(f"[6E1B] traces {traces}")
    print(f"[6E1B] |states| per trace = {len(model.states)}, QMC dimension = "
          f"{model.qmc_dimension}")
    print(f"[6E1B] label permutation audit: {audit['conclusion']}")

    config = {
        "stage": "6E1B", "source_commit": source_commit(),
        "stage6e_config_hash": config_hash(),
        "python": platform.python_version(), "numpy": np.__version__,
        "problem": {"J": J, "n_traces": N_TRACES, "n_skills": K_SKILLS,
                    "m_roles": M_ROLES, "d_latent": D_LATENT, "epsilon": EPSILON,
                    "delta_B": DELTA_B, "min_width": MIN_BLOCK_WIDTH,
                    "max_width": MAX_BLOCK_WIDTH, "corpus_seed": CORPUS_SEED,
                    "pi_fixed": PI_FIXED.tolist(), "P_fixed": P_FIXED.tolist(),
                    "traces": [list(t) for t in traces],
                    "generating_segmentations": [[list(p) for p in t] for t in truths],
                    "generating_note": "recorded only; enters no comparison",
                    "U_generating": U_GENERATING.tolist(), "truth_scalars": TRUTH,
                    "n_enumerated_states_per_trace": len(model.states),
                    "qmc_dimension": model.qmc_dimension,
                    "qmc_coordinates": model.coordinate_names()},
        "latent": ["S", "z", "U", "rho", *SCALAR_ORDER],
        "fixed": ["pi", "P", "delta_B", "epsilon", "K", "m", "d"],
        "pi_P_fixed_why": "section 9 permits it; their conjugate update is defined given "
                          "sampled labels, so adding them to the QMC construction would "
                          "need a Dirichlet inverse CDF and K more nearly-unidentified "
                          "dimensions for no gain in what this reference tests. They are "
                          "INFERRED in Stage 6E2.",
        "qmc": {"sequence": "scrambled Sobol", "n_replicates": args.replicates,
                "n_points_per_replicate": args.points, "seeds": list(QMC_SEEDS),
                "weight": "w = prod_n Z_n(theta), the exact marginal segmentation "
                          "likelihood; the proposal IS the joint prior so no prior "
                          "density remains in the weight"},
        "chains": {"n_chains": N_CHAINS, "sweeps": args.sweeps, "burn_in": BURN_IN,
                   "thin": THIN, "seeds": list(CHAIN_SEEDS),
                   "proposals_per_trace_per_sweep": PROPOSALS_PER_TRACE,
                   "scales": dict(REGISTERED_SCALES)},
    }
    (OUT / "config.json").write_text(json.dumps(jsonable(config), indent=2))

    # ---- reference --------------------------------------------------------------------
    if args.stage in ("reference", "all"):
        combined = build_reference(model)
        quality = reference_quality(combined, model)
        registration = {
            "registered_before_any_mcmc_comparison": True,
            "quality_gates": QUALITY_GATES,
            "superseded_gates": SUPERSEDED_GATES,
            "nondegeneracy_criteria": NONDEGENERACY,
            "estimators": ["conditional (Rao-Blackwellised, PRIMARY)",
                           "sampled (section 9 literal, reported alongside)"],
            **quality,
        }
        (OUT / "reference_registration.json").write_text(
            json.dumps(jsonable(registration), indent=2))
        (OUT / "qmc_summary.json").write_text(json.dumps(jsonable({
            "n_replicates": combined["n_replicates"],
            "log_evidence": combined["log_evidence"],
            "relative_ess": combined["relative_ess"],
            "max_normalised_weight": combined["max_normalised_weight"],
            "precision": combined["precision"],
            "superseded_descriptive": combined["superseded_descriptive"],
            "scalars": combined["scalars"],
            "runtime_seconds": combined["runtime_seconds"],
        }), indent=2))
        np.savez_compressed(
            OUT / "reference_draws.npz",
            segmentation_conditional=combined["pooled_segmentation"],
            segmentation_sampled=combined["pooled_segmentation_sampled"],
            boundary=combined["pooled_boundary"], labels=combined["pooled_labels"],
            segment_counts=combined["pooled_segment_counts"],
            relation=combined["pooled_relation"],
            **{f"h_probability_skill{k}": p
               for k, p in enumerate(combined["pooled_h_probability"])},
            **{f"h_keys_skill{k}": np.array(
                [np.frombuffer(key, dtype=bool) for key in keys])
               for k, keys in enumerate(combined["h_keys"])},
            state_ends=np.array([[e for e, _ in k] + [-1] * (J - len(k))
                                 for k in model.states], dtype=np.int16),
            state_labels=np.array([[s for _, s in k] + [-1] * (J - len(k))
                                   for k in model.states], dtype=np.int8),
            retained_closures=combined["retained"]["closures"],
            retained_sampled=combined["retained"]["sampled"],
            **{f"retained_{name}": combined["retained"][name]
               for name in ("rho", *SCALAR_NAMES)})
        print(f"[6E1B] reference primary_pass={quality['primary_pass']} "
              f"nondegenerate={quality['nondegenerate_pass']} "
              f"max rqmc_se={combined['precision']['max_rqmc_standard_error']:.3e} "
              f"half-width={combined['precision']['max_half_width_95']:.3e} "
              f"min rel ESS={combined['relative_ess']['min']:.4f}")
        if not (quality["primary_pass"] and quality["nondegenerate_pass"]):
            raise SystemExit("Stage 6E1B reference did not meet its registered gates; "
                             "chains not started")

    if args.stage == "reference":
        return

    # ---- chains -----------------------------------------------------------------------
    results = run_chains(model)
    compare(model, results)


def compare(model: MixedModel, results: list) -> None:
    """Every §9 comparison, against the frozen reference."""
    reference = np.load(OUT / "reference_draws.npz", allow_pickle=False)
    registration = json.loads((OUT / "reference_registration.json").read_text())
    state_index = {k: i for i, k in enumerate(model.states)}
    n_states = len(model.states)

    # ---- MCMC segmentation summaries ---------------------------------------------------
    per_chain_segmentation = []
    per_chain_boundary = []
    per_chain_labels = []
    per_chain_counts = []
    for result in results:
        segmentation = np.zeros((model.n_traces, n_states))
        for draw in result.boundary_keys:
            for t, key in enumerate(draw):
                segmentation[t, state_index[key]] += 1.0
        segmentation /= max(1, len(result.boundary_keys))
        per_chain_segmentation.append(segmentation)

        boundary = np.zeros((model.n_traces, J - 1))
        labels = np.zeros((model.n_traces, J, K_SKILLS))
        counts = np.zeros((model.n_traces, J + 1))
        for j, key in enumerate(model.states):
            for t in range(model.n_traces):
                p = segmentation[t, j]
                if p == 0.0:
                    continue
                start = 0
                for end, skill in key:
                    labels[t, start:end, skill] += p
                    start = end
                for end, _ in key[:-1]:
                    boundary[t, end - 1] += p
                counts[t, len(key)] += p
        per_chain_boundary.append(boundary)
        per_chain_labels.append(labels)
        per_chain_counts.append(counts)

    pooled_segmentation = np.mean(per_chain_segmentation, axis=0)
    pooled_boundary = np.mean(per_chain_boundary, axis=0)
    pooled_labels = np.mean(per_chain_labels, axis=0)
    pooled_counts = np.mean(per_chain_counts, axis=0)

    reference_segmentation = reference["segmentation_conditional"]
    reference_sampled = reference["segmentation_sampled"]

    tv_per_trace = [total_variation(pooled_segmentation[t], reference_segmentation[t])
                    for t in range(model.n_traces)]
    tv_per_trace_sampled = [total_variation(pooled_segmentation[t], reference_sampled[t])
                            for t in range(model.n_traces)]
    boundary_error = float(np.abs(pooled_boundary - reference["boundary"]).max())
    label_error = float(np.abs(pooled_labels - reference["labels"]).max())
    count_tv = max(total_variation(pooled_counts[t], reference["segment_counts"][t])
                   for t in range(model.n_traces))

    # ---- induced H, per skill ----------------------------------------------------------
    h_tv, relation_error_by_skill = [], []
    mcmc_h = []
    for k in range(K_SKILLS):
        keys = [h_label_of(u) for result in results for u in result.u_draws[:, k]]
        counts: dict = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        total = sum(counts.values())
        reference_keys = [np.packbits(row).tobytes()
                          for row in reference[f"h_keys_skill{k}"]]
        del reference_keys
        ref_keys_raw = [row.tobytes()
                        for row in reference[f"h_keys_skill{k}"].astype(bool)]
        ref_p = reference[f"h_probability_skill{k}"]
        union = list(ref_keys_raw)
        for key in counts:
            if key not in union:
                union.append(key)
        a = np.array([counts.get(key, 0) / total for key in union])
        b = np.array([ref_p[ref_keys_raw.index(key)] if key in ref_keys_raw else 0.0
                      for key in union])
        h_tv.append(total_variation(a, b))
        mcmc_h.append({"keys": [key.hex() for key in union], "probability": a.tolist()})
        mcmc_relation = np.zeros(M_ROLES * M_ROLES)
        for key, count in counts.items():
            mcmc_relation += (count / total) * np.frombuffer(
                key, dtype=bool).astype(float)
        relation_error_by_skill.append(
            float(np.abs(mcmc_relation - reference["relation"][k]).max()))

    # ---- scalars ------------------------------------------------------------------------
    scalar_names = ("rho", *SCALAR_NAMES)
    scalar_summary = {}
    for name in scalar_names:
        chains = np.array([r.scalars[name] for r in results])
        block = convergence_block(chains, name)
        reference_mean = registration and None
        del reference_mean
        scalar_summary[name] = {
            "mcmc_mean": float(chains.mean()), "mcmc_sd": float(chains.std(ddof=1)),
            "mcmc_median": float(np.median(chains)),
            "mcmc_q025": float(np.quantile(chains, 0.025)),
            "mcmc_q975": float(np.quantile(chains, 0.975)),
            **block}

    qmc = json.loads((OUT / "qmc_summary.json").read_text())
    for name in scalar_names:
        pooled = qmc["scalars"][name]["pooled_summary"]
        summary = scalar_summary[name]
        reference_sd = pooled["sd"]
        summary["reference_mean"] = pooled["mean"]
        summary["reference_sd"] = reference_sd
        summary["mean_gap_in_reference_sd"] = (
            abs(summary["mcmc_mean"] - pooled["mean"]) / reference_sd
            if reference_sd > 0 else float("nan"))
        summary["sd_ratio"] = (summary["mcmc_sd"] / reference_sd
                               if reference_sd > 0 else float("nan"))

    # ---- the mixed multivariate statistic ------------------------------------------------
    reference_rows = mixed_multivariate_coordinates(
        reference["retained_closures"], reference["retained_sampled"],
        {name: reference[f"retained_{name}"] for name in ("rho", *SCALAR_NAMES)}, model)
    mcmc_closures, mcmc_sampled, mcmc_scalars = [], [], {n: [] for n in
                                                         ("rho", *SCALAR_NAMES)}
    for result in results:
        for d in range(len(result.log_target)):
            mcmc_closures.append(np.array(
                [precedence_from_u(result.u_draws[d, k]).reshape(-1)
                 for k in range(K_SKILLS)]))
            mcmc_sampled.append([state_index[key] for key in result.boundary_keys[d]])
        for name in ("rho", *SCALAR_NAMES):
            mcmc_scalars[name].append(result.scalars[name])
    mcmc_rows = mixed_multivariate_coordinates(
        np.array(mcmc_closures), np.array(mcmc_sampled),
        {n: np.concatenate(v) for n, v in mcmc_scalars.items()}, model)
    thinned = mcmc_rows[::max(1, len(mcmc_rows) // len(reference_rows))]
    mixed = mixed_multivariate_statistic(reference_rows, thinned)

    # ---- convergence on the required coordinates ----------------------------------------
    log_target = np.array([r.log_target for r in results])
    n_segments = np.array([r.segment_counts.sum(axis=1) for r in results])
    relation_count = np.array([r.relation_counts.sum(axis=1) for r in results])
    boundary_indicator = np.array(
        [[1.0 if any(e == 4 for e, _ in draw[0]) else 0.0 for draw in r.boundary_keys]
         for r in results])
    varying_relation = None
    for k in range(K_SKILLS):
        for i in range(M_ROLES):
            for j in range(M_ROLES):
                if i == j:
                    continue
                series = np.array([[float(precedence_from_u(u[k])[i, j])
                                    for u in r.u_draws] for r in results])
                if 0.02 < series.mean() < 0.98:
                    varying_relation = (k, i, j, series)
                    break
            if varying_relation:
                break
        if varying_relation:
            break

    convergence = {
        "log_posterior": convergence_block(log_target, "log_posterior"),
        "n_segments": convergence_block(n_segments, "n_segments"),
        "relation_count": convergence_block(relation_count, "relation_count"),
        "selected_boundary_indicator": convergence_block(
            boundary_indicator, "boundary at t=4 in trace 0"),
        "selected_relation_indicator": (
            convergence_block(varying_relation[3],
                              f"relation {varying_relation[:3]}")
            if varying_relation else
            {"degenerate": True, "note": "no relation indicator has posterior probability "
                                         "strictly inside (0.02, 0.98); none is selected",
             "rhat": None}),
        **{name: {k: v for k, v in scalar_summary[name].items()
                  if k in ("rhat", "bulk_ess", "tail_ess", "mcse", "degenerate")}
           for name in scalar_names},
    }

    def rhat_of(block):
        return block.get("rhat") if block and not block.get("degenerate") else None

    rhat_values = {name: rhat_of(block) for name, block in convergence.items()}
    worst_rhat = max((v for v in rhat_values.values() if v is not None), default=None)

    gates = {
        "segmentation_total_variation": {
            "value": max(tv_per_trace), "threshold": 0.01,
            "pass": bool(max(tv_per_trace) < 0.01), "per_trace": tv_per_trace},
        "segmentation_total_variation_vs_sampled_estimator": {
            "value": max(tv_per_trace_sampled), "threshold": 0.01,
            "pass": bool(max(tv_per_trace_sampled) < 0.01),
            "per_trace": tv_per_trace_sampled,
            "note": "section 9's literal sampled reference estimator; the conditional "
                    "estimator above is primary"},
        "max_boundary_marginal_error": {
            "value": boundary_error, "threshold": 0.01,
            "pass": bool(boundary_error < 0.01)},
        "max_occurrence_label_marginal_error": {
            "value": label_error, "threshold": 0.01, "pass": bool(label_error < 0.01)},
        "induced_h_total_variation": {
            "value": max(h_tv), "threshold": 0.01, "pass": bool(max(h_tv) < 0.01),
            "per_skill": h_tv},
        "max_relation_marginal_error": {
            "value": max(relation_error_by_skill), "threshold": 0.01,
            "pass": bool(max(relation_error_by_skill) < 0.01),
            "per_skill": relation_error_by_skill},
        "segment_count_total_variation": {
            "value": count_tv, "threshold": 0.01, "pass": bool(count_tv < 0.01)},
        "mixed_multivariate_reference_statistic": {
            "value": mixed["observed"], "threshold": mixed["envelope"],
            "pass": mixed["pass"], "z_score": mixed["z_score"]},
        **{f"{name}_rhat": {"value": value, "threshold": 1.01,
                            "pass": bool(value is not None and value <= 1.01)}
           for name, value in rhat_values.items() if value is not None},
    }
    for name, value in rhat_values.items():
        if value is None:
            gates[f"{name}_rhat"] = {
                "value": None, "threshold": 1.01, "pass": True,
                "note": "degenerate or not applicable; recorded as such, not as an "
                        "R-hat of 1.0"}
    all_pass = all(g["pass"] for g in gates.values())

    np.savez_compressed(
        OUT / "chains.npz",
        segmentation=np.array(per_chain_segmentation),
        boundary=np.array(per_chain_boundary), labels=np.array(per_chain_labels),
        segment_counts=np.array(per_chain_counts),
        u_draws=np.array([r.u_draws for r in results]),
        **{f"scalar_{n}": np.array([r.scalars[n] for r in results])
           for n in scalar_names},
        log_target=log_target, n_segments=n_segments, relation_counts=relation_count,
        chain_seeds=np.array([r.seed for r in results]),
        runtime_seconds=np.array([r.runtime_seconds for r in results]))

    (OUT / "segmentation_comparison.json").write_text(json.dumps(jsonable({
        "reference_estimator": "conditional (Rao-Blackwellised), primary",
        "total_variation_per_trace": tv_per_trace,
        "total_variation_per_trace_vs_sampled": tv_per_trace_sampled,
        "max_boundary_marginal_error": boundary_error,
        "max_occurrence_label_marginal_error": label_error,
        "segment_count_total_variation": count_tv,
        "reference_segmentation": reference_segmentation,
        "mcmc_segmentation": pooled_segmentation,
        "reference_boundary": reference["boundary"], "mcmc_boundary": pooled_boundary,
        "reference_segment_counts": reference["segment_counts"],
        "mcmc_segment_counts": pooled_counts,
        "per_chain_total_variation": [
            [total_variation(c[t], reference_segmentation[t])
             for t in range(model.n_traces)] for c in per_chain_segmentation],
    }), indent=2))

    (OUT / "structural_comparison.json").write_text(json.dumps(jsonable({
        "induced_h_total_variation_per_skill": h_tv,
        "max_relation_marginal_error_per_skill": relation_error_by_skill,
        "reference_relation_marginal": reference["relation"],
        "mcmc_h": mcmc_h,
        "n_reference_h_states_per_skill": [
            int(len(reference[f"h_probability_skill{k}"])) for k in range(K_SKILLS)],
        "relation_count_convergence": convergence["relation_count"],
        "selected_relation_indicator": convergence["selected_relation_indicator"],
    }), indent=2))

    (OUT / "scalar_comparison.json").write_text(json.dumps(jsonable(scalar_summary),
                                                           indent=2))

    (OUT / "joint_comparison.json").write_text(json.dumps(jsonable({
        "gates": gates, "all_pass": all_pass,
        "mixed_multivariate": mixed,
        "convergence": convergence,
        "acceptance_by_chain": [r.acceptance() for r in results],
        "movement": [r.movement for r in results],
        "retained_per_chain": [len(r.log_target) for r in results],
        "retained_pooled": int(sum(len(r.log_target) for r in results)),
        "runtime_seconds": [r.runtime_seconds for r in results],
        "worst_rhat": worst_rhat,
    }), indent=2))

    for name, gate in gates.items():
        value = gate["value"]
        shown = "n/a" if value is None else f"{value:.6g}"
        print(f"[6E1B] {name:52s} {shown:>12s} (thr {gate['threshold']}) -> "
              f"{'PASS' if gate['pass'] else 'FAIL'}")
    print(f"[6E1B] wrote {OUT}")
    if not all_pass:
        raise SystemExit("Stage 6E1B FAILED: "
                         f"{[k for k, g in gates.items() if not g['pass']]}")


if __name__ == "__main__":
    main()
