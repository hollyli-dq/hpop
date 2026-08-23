"""Stage 6E2 — the formal chains (§14) and the like-for-like oracle-boundary control (§12).

    PYTHONPATH=src python scripts/stage6e2_formal_chains.py --run unknown
    PYTHONPATH=src python scripts/stage6e2_formal_chains.py --run oracle
    PYTHONPATH=src python scripts/stage6e2_formal_chains.py --continue-block 25000

The pilot is over and every pilot draw is discarded; these chains restart from scratch at
the selected scales.

## The oracle-boundary control is the same sampler with `(S, z)` pinned

§12 asks for the frozen Stage 6D kernel run on the oracle blocks of *these* traces, so that
the comparison isolates the cost of making `(S, z)` latent rather than mixing it with a
change of corpus. That is obtained here by running the Stage 6E sampler with **zero
segmentation proposals per trace** and the segmentation pinned at the hidden truth. The
remaining updates are then literally the Stage 6D objects — `sampler_u.propose_row`, the
logit `rho` walk with its `log(rho(1-rho))` Jacobian, and `scalar_mh_step` with
`build_proposal` — and Stage 6E0 §7.1 already established that with `(S, z)` fixed the
Stage 6E target differs from the Stage 6D target by an additive constant that depends on no
inferred coordinate, so every acceptance ratio the control forms is a Stage 6D ratio.

The one addition is the `(pi, P)` Gibbs step, which Stage 6D did not have because it had
`K = 1` and therefore no path prior to infer. It is kept in the control so the control and
the unknown-boundary run differ in exactly one thing: whether `(S, z)` is latent.

## Dispersion

The four starts are structurally distinct and **none is the true segmentation**: a fine
segmentation with varied labels, a coarse legal one, a random legal one, and an
intermediate one built from a different width rule. `U`, `rho`, the four scalars and
`(pi, P)` are dispersed independently, and every start is checked to have finite posterior
density before the chain is allowed to begin.

## Continuation

If the registered convergence gates fail, the *same* chains continue in deterministic
25,000-sweep blocks to a maximum of 150,000, resuming from the saved state and RNG so the
continuation is bit-identical to an uninterrupted run. No chain is ever restarted and
selected, and no scale is changed after recovery has been looked at.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.fast_segmentation_kernel import segmentation_of        # noqa: E402
from hpop.mcmc_original.stage6c_frozen import log_structural_prior             # noqa: E402
from hpop.mcmc_original.stage6d_frozen import SCALAR_ORDER                     # noqa: E402
from hpop.mcmc_original.stage6e_corpus import generate_corpus                  # noqa: E402
from hpop.mcmc_original.stage6e_frozen import (                                # noqa: E402
    MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH, N_ROLES, N_SKILLS, assert_stage6d_unchanged,
)
from hpop.mcmc_original.recurrent_segmentation import log_target_stage6e       # noqa: E402
from hpop.mcmc_original.stage6e_sampler import run_stage6e_chain               # noqa: E402
from hpop.mcmc_original.stage6e_state import Stage6EModel, Stage6EState        # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "stage6e2_unknown_boundary_full_seed0"

N_CHAINS = 4
N_SWEEPS = 50_000
BURN_IN = 15_000
THIN = 5
CONTINUATION_BLOCK = 25_000
MAX_SWEEPS = 150_000
CHAIN_SEEDS = (6_053_201, 6_053_202, 6_053_203, 6_053_204)
START_SEEDS = (6_053_301, 6_053_302, 6_053_303, 6_053_304)


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


# ----------------------------------------------------------------- dispersed segmentations
def fine_key(length: int, n_skills: int, rng) -> tuple:
    """As many minimum-width blocks as fit, labels cycling with a random offset."""
    ends, start, previous, offset = [], 0, None, int(rng.integers(n_skills))
    index = 0
    while start < length:
        remaining = length - start
        width = MIN_BLOCK_WIDTH
        if remaining - width != 0 and remaining - width < MIN_BLOCK_WIDTH:
            width = remaining                      # absorb the tail; still legal
        if width > MAX_BLOCK_WIDTH:
            width = MAX_BLOCK_WIDTH
            if remaining - width < MIN_BLOCK_WIDTH:
                width = remaining - MIN_BLOCK_WIDTH
        skill = (offset + index) % n_skills
        if skill == previous:
            skill = (skill + 1) % n_skills
        start += width
        ends.append((start, skill))
        previous, index = skill, index + 1
    return tuple(ends)


def coarse_key(length: int, n_skills: int, rng) -> tuple:
    """As few blocks as the maximum width allows, labels alternating."""
    ends, start, previous = [], 0, None
    while start < length:
        remaining = length - start
        width = min(MAX_BLOCK_WIDTH, remaining)
        if remaining - width != 0 and remaining - width < MIN_BLOCK_WIDTH:
            width = remaining - MIN_BLOCK_WIDTH
        skill = int(rng.choice([k for k in range(n_skills) if k != previous]))
        start += width
        ends.append((start, skill))
        previous = skill
    return tuple(ends)


def random_key(length: int, n_skills: int, rng) -> tuple:
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


def intermediate_key(length: int, n_skills: int, rng) -> tuple:
    """Alternating short/long widths — structurally unlike the other three."""
    ends, start, previous, toggle = [], 0, None, True
    while start < length:
        remaining = length - start
        width = MIN_BLOCK_WIDTH + 1 if toggle else min(MAX_BLOCK_WIDTH - 2, remaining)
        width = min(width, remaining)
        if remaining - width != 0 and remaining - width < MIN_BLOCK_WIDTH:
            width = remaining
        skill = int(rng.choice([k for k in range(n_skills) if k != previous]))
        start += width
        ends.append((start, skill))
        previous, toggle = skill, not toggle
    return tuple(ends)


START_SHAPES = (fine_key, coarse_key, random_key, intermediate_key)
SHAPE_NAMES = ("fine, varied labels", "coarse", "random legal",
               "intermediate alternating widths")

RHO_STARTS = (0.10, 0.35, 0.60, 0.85)
SCALAR_STARTS = (
    {"beta": 0.60, "omega": -1.00, "lambda_rep": 0.30, "lambda_back": 0.80},
    {"beta": 1.10, "omega": 0.60, "lambda_rep": 0.90, "lambda_back": 0.20},
    {"beta": 2.20, "omega": 2.40, "lambda_rep": 1.60, "lambda_back": 1.20},
    {"beta": 3.00, "omega": 3.80, "lambda_rep": 2.40, "lambda_back": 0.06},
)


def uniform_transition(n_skills: int) -> np.ndarray:
    p = np.full((n_skills, n_skills), 1.0 / (n_skills - 1))
    np.fill_diagonal(p, 0.0)
    return p


def dispersed_start(chain: int, corpus, model: Stage6EModel, oracle: bool) -> Stage6EState:
    rng = np.random.default_rng(START_SEEDS[chain])
    if oracle:
        segmentations = tuple(segmentation_of(t.true_key()) for t in corpus.train)
    else:
        shape = START_SHAPES[chain]
        segmentations = tuple(segmentation_of(shape(len(t), model.n_skills, rng))
                              for t in model.traces)
    u = rng.normal(scale=1.2, size=(model.n_skills, model.n_roles, 2))
    # dispersed (pi, P): a Dirichlet draw per chain rather than the uniform point
    pi = rng.dirichlet(np.full(model.n_skills, 1.0))
    transition = np.zeros((model.n_skills, model.n_skills))
    for h in range(model.n_skills):
        allowed = [k for k in range(model.n_skills) if k != h]
        draw = rng.dirichlet(np.full(len(allowed), 1.0))
        for k, value in zip(allowed, draw):
            transition[h, k] = value
    return Stage6EState(segmentations=segmentations, u_by_skill=u,
                        rho=RHO_STARTS[chain], pi=pi, transition=transition,
                        **SCALAR_STARTS[chain])


def build_model(corpus) -> Stage6EModel:
    return Stage6EModel(traces=corpus.traces("train"), epsilon=corpus.epsilon,
                        delta_b=corpus.delta_b, n_skills=N_SKILLS, n_roles=N_ROLES,
                        min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
                        infer_pi_P=True)


# ------------------------------------------------------------------------- one chain
def _run_one(payload: dict) -> dict:
    """Worker entry point. Rebuilds the model from the corpus rather than unpickling it."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    corpus = generate_corpus()
    model = build_model(corpus)
    chain = payload["chain"]
    oracle = payload["oracle"]
    resume = payload.get("resume")

    if resume is None:
        state = dispersed_start(chain, corpus, model, oracle)
        rng = np.random.default_rng(payload["seed"])
    else:
        state = Stage6EState.from_dict(resume)
        rng = np.random.default_rng(payload["seed"])
        rng.bit_generator.state = state.rng_state

    parts = log_target_stage6e(state, model)
    if not math.isfinite(parts["log_target"]):
        raise ValueError(f"chain {chain}: start has non-finite log target")

    result = run_stage6e_chain(
        model=model, start=state, scales=payload["scales"],
        n_proposals_per_trace=payload["n_proposals"], num_sweeps=payload["num_sweeps"],
        burn_in=payload["burn_in"], thin=THIN, seed=payload["seed"], chain=chain,
        rng=rng, state=(state if resume is not None else None),
        store_labels=True, store_keys=False,
        use_block_table=bool(payload["n_proposals"] > 0),
        progress_every=payload.get("progress_every", 0),
        checkpoint_path=payload.get("checkpoint_path"),
        checkpoint_every=payload.get("checkpoint_every", 0))

    return {
        "chain": chain, "seed": payload["seed"],
        "start_log_target": parts["log_target"],
        "u_draws": result.u_draws, "scalars": result.scalars,
        "pi_draws": result.pi_draws, "transition_draws": result.transition_draws,
        "segment_counts": result.segment_counts,
        "occurrence_labels": result.occurrence_labels,
        "log_target": result.log_target,
        "log_block_likelihood": result.log_block_likelihood,
        "relation_counts": result.relation_counts,
        "proposed": result.proposed, "accepted": result.accepted,
        "invalid": result.invalid,
        "proposed_after_burn_in": result.proposed_after_burn_in,
        "accepted_after_burn_in": result.accepted_after_burn_in,
        "movement": result.movement, "runtime_seconds": result.runtime_seconds,
        "final_state": result.final_state.to_dict(),
        "n_retained": int(len(result.log_target)),
    }


def save_run(results: list, tag: str, meta: dict, append_to=None) -> Path:
    """Write the chain arrays. `append_to` concatenates onto an earlier block's draws.

    A §14 continuation extends the *same* chains, so its draws belong after the ones the
    first block already retained. Writing only the new block would silently discard
    everything before it — the run would look like 25,000 sweeps rather than 75,000, and
    every ESS and R-hat would be computed on the wrong sample.
    """
    path = OUT / (f"chains.npz" if tag == "unknown" else f"oracle_control_chains.npz")
    payload = {
        "u_draws": np.array([r["u_draws"] for r in results]),
        "pi_draws": np.array([r["pi_draws"] for r in results]),
        "transition_draws": np.array([r["transition_draws"] for r in results]),
        "segment_counts": np.array([r["segment_counts"] for r in results]),
        "occurrence_labels": np.array([r["occurrence_labels"] for r in results]),
        "log_target": np.array([r["log_target"] for r in results]),
        "log_block_likelihood": np.array([r["log_block_likelihood"] for r in results]),
        "relation_counts": np.array([r["relation_counts"] for r in results]),
        "chain_seeds": np.array([r["seed"] for r in results]),
        **{f"scalar_{n}": np.array([r["scalars"][n] for r in results])
           for n in (*SCALAR_ORDER, "rho")},
    }
    if append_to is not None and Path(append_to).exists():
        previous = np.load(append_to)
        merged = {}
        for key, value in payload.items():
            if key == "chain_seeds":
                merged[key] = value
                continue
            merged[key] = np.concatenate([previous[key], value], axis=1)
        payload = merged
        print(f"    concatenated onto {previous['log_target'].shape[1]:,} earlier draws "
              f"per chain -> {payload['log_target'].shape[1]:,}")
    np.savez_compressed(path, **payload)
    (OUT / (f"{tag}_run_summary.json")).write_text(json.dumps(jsonable({
        **meta,
        "per_chain": [{k: r[k] for k in
                       ("chain", "seed", "n_retained", "runtime_seconds", "movement",
                        "proposed", "accepted", "invalid", "proposed_after_burn_in",
                        "accepted_after_burn_in", "start_log_target")}
                      for r in results],
        "acceptance_post_burn_in": [
            {k: (r["accepted_after_burn_in"].get(k, 0) / v if v else None)
             for k, v in r["proposed_after_burn_in"].items()} for r in results],
    }), indent=2))
    (OUT / f"{tag}_final_states.json").write_text(json.dumps(
        {str(r["chain"]): r["final_state"] for r in results}, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="unknown", choices=["unknown", "oracle"])
    parser.add_argument("--sweeps", type=int, default=N_SWEEPS)
    parser.add_argument("--burn-in", type=int, default=BURN_IN)
    parser.add_argument("--resume", action="store_true",
                        help="section 14 continuation: extend the completed chains by one "
                             "registered 25,000-sweep block after a convergence gate failed")
    parser.add_argument("--recover", action="store_true",
                        help="crash recovery: resume interrupted chains from their last "
                             "checkpoint to the SAME registered sweep count. This is not a "
                             "section 14 continuation and does not extend the run")
    args = parser.parse_args()
    if args.resume and args.recover:
        raise SystemExit("--resume and --recover are different operations; pick one")

    assert_stage6d_unchanged()
    OUT.mkdir(parents=True, exist_ok=True)
    pilot = json.loads((OUT / "pilot_results.json").read_text())
    scales = {k: float(v) for k, v in pilot["selected_scales"].items()}
    n_proposals = (0 if args.run == "oracle"
                   else int(pilot["selected_proposals_per_trace"]))

    corpus = generate_corpus()
    model = build_model(corpus)

    history_path = OUT / "continuation_history.json"
    history = (json.loads(history_path.read_text()) if history_path.exists()
               else {"unknown": [], "oracle": []})

    resume_states = None
    total_sweeps = args.sweeps
    burn_in = args.burn_in
    if args.recover:
        # Crash recovery. The chains were interrupted; each checkpoint carries its own
        # state AND its own RNG state, so continuing from it is bit-identical to the run
        # that would have happened uninterrupted. The target sweep count is UNCHANGED —
        # this completes the registered run rather than extending it, which is why it is
        # recorded separately from a section 14 continuation.
        directory = OUT / f"{args.run}_checkpoints"
        states, sweeps = {}, []
        for chain in range(N_CHAINS):
            payload = json.loads((directory / f"chain{chain}_checkpoint.json").read_text())
            if payload["n_retained"]:
                raise SystemExit(
                    f"chain {chain} had already retained {payload['n_retained']} draws at "
                    "its checkpoint; recovering would discard them. Resume support for a "
                    "partially retained chain is not implemented, deliberately — silently "
                    "losing draws would be worse than refusing.")
            states[str(chain)] = payload["state"]
            sweeps.append(int(payload["sweep"]))
        resume_states = states
        print(f"[6E2:{args.run}] CRASH RECOVERY from checkpoints at sweeps "
              f"{sorted(set(sweeps))}, continuing to the registered {total_sweeps:,}; "
              f"no draws were retained before the interruption, so nothing is lost")
    elif args.resume:
        states = json.loads((OUT / f"{args.run}_final_states.json").read_text())
        resume_states = states
        previous = max(int(s["iteration"]) for s in states.values())
        total_sweeps = previous + CONTINUATION_BLOCK
        if total_sweeps > MAX_SWEEPS:
            raise SystemExit(f"continuation would exceed the registered maximum of "
                             f"{MAX_SWEEPS} sweeps per chain")
        burn_in = args.burn_in
        print(f"[6E2:{args.run}] continuing from sweep {previous:,} to {total_sweeps:,}")

    payloads = [{
        "chain": c, "seed": CHAIN_SEEDS[c], "oracle": args.run == "oracle",
        "scales": scales, "n_proposals": n_proposals, "num_sweeps": total_sweeps,
        "burn_in": burn_in,
        "resume": (resume_states[str(c)] if resume_states else None),
        "progress_every": max(1, total_sweeps // 20),
        "checkpoint_path": str(OUT / f"{args.run}_checkpoints"),
        "checkpoint_every": max(1, total_sweeps // 10),
    } for c in range(N_CHAINS)]

    print(f"[6E2:{args.run}] {N_CHAINS} chains x {total_sweeps:,} sweeps, burn-in "
          f"{burn_in:,}, thin {THIN}, {n_proposals} segmentation proposals per trace "
          f"per sweep", flush=True)
    for c in range(N_CHAINS):
        print(f"    chain {c}: start shape = "
              f"{'ORACLE (true segmentation, pinned)' if args.run == 'oracle' else SHAPE_NAMES[c]}"
              f", rho0 = {RHO_STARTS[c]}, scalars {SCALAR_STARTS[c]}")

    began = time.perf_counter()
    with get_context("spawn").Pool(processes=N_CHAINS) as pool:
        results = pool.map(_run_one, payloads)
    wall = time.perf_counter() - began

    meta = {
        "stage": "6E2", "run": args.run, "source_commit": source_commit(),
        "python": platform.python_version(), "numpy": np.__version__,
        "n_chains": N_CHAINS, "sweeps": total_sweeps, "burn_in": burn_in, "thin": THIN,
        "chain_seeds": list(CHAIN_SEEDS), "start_seeds": list(START_SEEDS),
        "scales": scales, "n_proposals_per_trace": n_proposals,
        "start_shapes": (["ORACLE true segmentation (pinned)"] * N_CHAINS
                         if args.run == "oracle" else list(SHAPE_NAMES)),
        "rho_starts": list(RHO_STARTS), "scalar_starts": list(SCALAR_STARTS),
        "wall_seconds": wall,
        "retained_per_chain": [r["n_retained"] for r in results],
        "retained_pooled": int(sum(r["n_retained"] for r in results)),
        "oracle_control_note": (
            "segmentation pinned at the hidden truth and zero segmentation proposals, so "
            "the remaining updates are literally the frozen Stage 6D kernels; Stage 6E0 "
            "7.1 established that with (S,z) fixed the Stage 6E target differs from the "
            "Stage 6D target by an additive constant independent of every inferred "
            "coordinate. The (pi,P) Gibbs step is retained so that this control and the "
            "unknown-boundary run differ in exactly one thing."
            if args.run == "oracle" else None),
    }
    # a continuation appends to the block before it; a crash recovery completes the
    # same block and replaces it
    path = save_run(results, args.run, meta,
                    append_to=(OUT / ("chains.npz" if args.run == "unknown"
                                      else "oracle_control_chains.npz"))
                    if args.resume else None)
    history[args.run].append({
        "block": len(history[args.run]) + 1,
        "sweeps_to": total_sweeps, "burn_in": burn_in, "thin": THIN,
        "resumed": bool(args.resume or args.recover),
        "kind": ("section 14 continuation" if args.resume
                 else "crash recovery (same registered sweep count)" if args.recover
                 else "initial registered run"),
        "wall_seconds": wall,
        "retained_pooled": meta["retained_pooled"],
        "scales": scales, "n_proposals_per_trace": n_proposals,
        "why": ("registered 25,000-sweep continuation after a convergence gate failed"
                if args.resume else
                "crash recovery: the chains were interrupted and resumed from their last "
                "checkpoint to the SAME registered sweep count, with each chain's own RNG "
                "state restored. Not a continuation and not an extension."
                if args.recover else "initial registered run"),
    })
    history_path.write_text(json.dumps(jsonable(history), indent=2))

    print(f"[6E2:{args.run}] done in {wall / 3600:.2f} h; "
          f"{meta['retained_pooled']:,} retained draws -> {path} "
          f"({path.stat().st_size / 1e6:.1f} MB)")
    for r in results:
        acceptance = {k: round(r["accepted_after_burn_in"].get(k, 0) / v, 3)
                      for k, v in r["proposed_after_burn_in"].items() if v}
        print(f"    chain {r['chain']}: {r['runtime_seconds'] / 3600:.2f} h, "
              f"acceptance {acceptance}")


if __name__ == "__main__":
    main()
