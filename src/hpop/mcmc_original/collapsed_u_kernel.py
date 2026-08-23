"""The occasional partially-collapsed latent-U structural move, and its sweep composition.

## The move

`collapsed_u_mh_step` is a Metropolis-Hastings update of ONE row of ONE skill's `U_k` —
the registered Stage 6C row proposal (`sampler_u.propose_row`, the production scale, the
same proposal covariance at fixed `rho`, the same prior, the same `h(U)`) — scored against
the **collapsed** likelihood in which the labelled segmentation is summed out exactly:

    log R_U = [ell_coll(U') - ell_coll(U)] + [log p(U'|rho) - log p(U|rho)]
              + [log q(U|U') - log q(U'|U)]

The proposal is symmetric (a mean-zero Gaussian added to one row), so the last bracket is
zero; `test_collapsed_u_kernel.py` verifies that from executable code rather than by
assertion. The move NEVER reads the stored `(S, z)` — the acceptance ratio contains no
conditional-likelihood term, and `test_collapsed_u_ordering.py` proves the decision is
byte-identical under a scrambled stored segmentation.

## Why the composition preserves the target

Write the joint target `pi(U, S, z, phi | X)` with `phi = (rho, theta, pi, P)` and let
`pi_marg(U | phi) = sum_{S,z} pi(U, S, z, phi)`. The scheduled block is

    K((U,S,z) -> (U',S',z')) = M(U -> U') * p(S', z' | X, U', phi)

with `M` the collapsed MH kernel above, which is invariant for `pi_marg` by MH
construction. Integrating the incoming `(S, z)` out (nothing in the block reads them):

    sum_{S,z} pi(U,S,z) K = pi_marg(U) M(U -> U') p(S',z' | U')
                          -> pi_marg(U') p(S',z' | U') = pi(U', S', z')

— the standard partially-collapsed Gibbs argument (marginalise, move, then draw the
discarded coordinates from their exact conditional). The second factor is Step 7B's
`ffbs_sweep_once`, whose FIRST action is to rebuild the candidate tables at the
post-move parameters and draw every trace's `(S, z)` from its exact conditional (Step 7A
FFBS, acceptance 1); only after that fresh draw does the Stage 6E parameter phase — which
conditions on `(S, z)` — run. So the required ordering

    collapsed U MH  ->  exact FFBS refresh of all (S, z)  ->  conditional updates

is exactly what `collapsed_ffbs_sweep_once` executes, and no stale `(S, z)` is consumed
between the collapsed move and the refresh. On sweeps where no collapsed move is
scheduled the function IS `ffbs_sweep_once`, bit for bit.

## What this module deliberately does not do

It does not replace the ordinary conditional U sweep (Stage 6C rows inside `sweep_once`)
— that remains the cheap within-cell mixer, untouched; the collapsed move is the
structural one. It does not modify `semi_markov_ffbs`, `sampler_u`, the recurrent scorer,
or `recurrent_joint_ffbs_mcmc`: the composition is by CALL, in this new module only, so
the validated Step 7A/7B code paths are byte-identical to before this work.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from hpop.mcmc_original.collapsed_u_likelihood import CollapsedULikelihood
from hpop.mcmc_original.fast_segmentation_kernel import key_of
from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.proposals import MoveType
from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import (
    Stage7BSampler, ffbs_sweep_once,
)
from hpop.mcmc_original.sampler_u import propose_row
from hpop.mcmc_original.stage6c_frozen import log_structural_prior
from hpop.mcmc_original.stage6d_frozen import SCALAR_ORDER
from hpop.mcmc_original.stage6e_sampler import Stage6EChainResult, _write_checkpoint
from hpop.mcmc_original.stage6e_state import Stage6EModel, Stage6EState

__all__ = [
    "CollapsedUConfig", "collapsed_u_mh_step", "collapsed_ffbs_sweep_once",
    "is_collapsed_sweep", "run_collapsed_u_chain",
]

MOVE_NAME = "U_collapsed"


@dataclass
class CollapsedUConfig:
    """Configuration of the occasional collapsed move. `every = 0` disables it."""

    every: int = 10                 # provisional default from the C1 audit; NOT tuned
    scale: float = 0.5              # the registered production row scale, unchanged

    def __post_init__(self) -> None:
        if int(self.every) < 0:
            raise ValueError(f"collapsed_u_every must be >= 0, got {self.every}")
        if float(self.scale) <= 0.0:
            raise ValueError(f"scale must be positive, got {self.scale}")


def is_collapsed_sweep(iteration: int, every: int) -> bool:
    """Scheduled on absolute sweep indices, so a resumed chain keeps the same phase."""
    return bool(every) and (int(iteration) + 1) % int(every) == 0


# ------------------------------------------------------------------------- the move
def collapsed_u_mh_step(state: Stage6EState, model: Stage6EModel,
                        likelihood: CollapsedULikelihood, rng,
                        scale: float) -> tuple[Stage6EState, dict]:
    """One collapsed MH update of a uniformly chosen `(skill, row)`. Returns a NEW state
    and the move record; the stored `(S, z)` is neither read nor written."""
    state = state.copy()
    K, m, _ = np.asarray(state.u_by_skill).shape
    skill = int(rng.integers(K))
    row = int(rng.integers(m))
    u_k = np.array(state.u_by_skill[skill], dtype=float)

    began = time.perf_counter()
    candidate = propose_row(u_k, row, float(scale), rng)
    state.proposed[MOVE_NAME] = state.proposed.get(MOVE_NAME, 0) + 1

    record = {"skill": skill, "row": row, "accepted": False, "h_changed": False,
              "invalid": False, "log_alpha": None, "d_log_lik_collapsed": None,
              "d_log_prior": None, "seconds": 0.0}
    candidate_prior = log_structural_prior(candidate, state.rho)
    if not math.isfinite(candidate_prior):
        state.invalid[MOVE_NAME] = state.invalid.get(MOVE_NAME, 0) + 1
        record.update(invalid=True, seconds=time.perf_counter() - began)
        return state, record

    current_prior = log_structural_prior(u_k, state.rho)
    d_coll, candidate_log_z = likelihood.delta_for_candidate(state, skill, candidate)
    d_prior = candidate_prior - current_prior
    # symmetric row proposal at fixed rho: the Hastings term is exactly zero
    log_alpha = d_coll + d_prior
    h_changed = not np.array_equal(precedence_from_u(candidate),
                                   precedence_from_u(u_k))

    accepted = bool(log_alpha >= 0.0 or math.log(rng.random()) < log_alpha)
    if accepted:
        u = np.array(state.u_by_skill, dtype=float, copy=True)
        u[skill] = candidate
        state.u_by_skill = u
        likelihood.commit_candidate(state, candidate_log_z)
        state.accepted[MOVE_NAME] = state.accepted.get(MOVE_NAME, 0) + 1

    record.update(accepted=accepted, h_changed=bool(h_changed),
                  log_alpha=float(log_alpha), d_log_lik_collapsed=float(d_coll),
                  d_log_prior=float(d_prior), seconds=time.perf_counter() - began)
    return state, record


# ------------------------------------------------------------------------ the sweep
def collapsed_ffbs_sweep_once(state: Stage6EState, sampler: Stage7BSampler,
                              likelihood: CollapsedULikelihood, config: CollapsedUConfig,
                              rng) -> tuple[Stage6EState, dict | None]:
    """One sweep. Scheduled sweeps prepend the collapsed move; every sweep then runs the
    unmodified `ffbs_sweep_once`, whose first action is the exact FFBS refresh of every
    `(S, z)` at the (possibly moved) `U` — the immediate refresh the ordering requires."""
    record = None
    if is_collapsed_sweep(state.iteration, config.every):
        state, record = collapsed_u_mh_step(state, sampler.model, likelihood, rng,
                                            config.scale)
    state = ffbs_sweep_once(state, sampler, rng)
    return state, record


# -------------------------------------------------------------------------- the run
def run_collapsed_u_chain(model: Stage6EModel, start: Stage6EState, scales: dict,
                          num_sweeps: int, burn_in: int, thin: int, seed: int,
                          collapsed: CollapsedUConfig | None = None, chain: int = 0,
                          table_source: str = "batched", rng=None,
                          state: Stage6EState | None = None, store_labels: bool = True,
                          store_keys: bool = True, progress_every: int = 0,
                          checkpoint_path=None, checkpoint_every: int = 0,
                          movement_tracker=None) -> Stage6EChainResult:
    """One chain of the partially-collapsed sampler. Storage, bookkeeping and the
    checkpoint format mirror `run_stage7b_chain`; the only difference in the loop is the
    sweep function, and with `collapsed.every = 0` the two chains are draw-identical."""
    if burn_in >= num_sweeps:
        raise ValueError("burn_in must be smaller than num_sweeps")
    if thin < 1:
        raise ValueError("thin must be at least 1")
    collapsed = CollapsedUConfig() if collapsed is None else collapsed

    sampler = Stage7BSampler(model=model, scales=dict(scales),
                             table_source=table_source)
    likelihood = CollapsedULikelihood(model=model)
    rng = np.random.default_rng(seed) if rng is None else rng
    state = (start.copy() if state is None else state)
    state.chain = chain
    for bucket in (state.proposed, state.accepted, state.invalid):
        for move in MoveType.ALL:
            bucket.setdefault(move, 0)
        for name in ("U", "rho", "pi_P", "S_z_ffbs", MOVE_NAME, *SCALAR_ORDER):
            bucket.setdefault(name, 0)

    start_iteration = int(state.iteration)
    capacity = len(range(max(burn_in, start_iteration), num_sweeps, thin)) + 1
    n_traces = len(model.traces)
    K, m, d = np.asarray(state.u_by_skill).shape
    max_length = max(len(t) for t in model.traces)

    u_draws = np.empty((capacity, K, m, d), dtype=np.float32)
    scalars = {n: np.empty(capacity) for n in (*SCALAR_ORDER, "rho")}
    pi_draws = np.empty((capacity, K), dtype=np.float32)
    transition_draws = np.empty((capacity, K, K), dtype=np.float32)
    segment_counts = np.empty((capacity, n_traces), dtype=np.int16)
    relation_counts = np.empty((capacity, K), dtype=np.int16)
    log_target = np.empty(capacity)
    log_blocks = np.empty(capacity)
    boundary_keys: list = []
    labels = (np.full((capacity, n_traces, max_length), -1, dtype=np.int8)
              if store_labels else None)

    post_proposed: dict = {}
    post_accepted: dict = {}
    movement = {"boundary_hamming": 0, "label_changes": 0, "states_changed": 0}
    collapsed_records: list = []

    k = 0
    began = time.perf_counter()
    for i in range(start_iteration, num_sweeps):
        before_p, before_a = dict(state.proposed), dict(state.accepted)
        state, record = collapsed_ffbs_sweep_once(state, sampler, likelihood,
                                                  collapsed, rng)
        if record is not None:
            record["sweep"] = i
            collapsed_records.append(record)
        movement["boundary_hamming"] += state.components.get("boundary_hamming_moved", 0)
        movement["label_changes"] += state.components.get("label_changes_moved", 0)
        movement["states_changed"] += state.components.get("ffbs_states_changed", 0)
        if movement_tracker is not None:
            # observed every sweep, never read by the sampler — as in run_stage7b_chain
            movement_tracker.record(state, [key_of(s) for s in state.segmentations])
        if i >= burn_in:
            for name in state.proposed:
                post_proposed[name] = (post_proposed.get(name, 0)
                                       + state.proposed[name] - before_p.get(name, 0))
                post_accepted[name] = (post_accepted.get(name, 0)
                                       + state.accepted[name] - before_a.get(name, 0))
            if (i - burn_in) % thin == 0 and k < capacity:
                u_draws[k] = state.u_by_skill
                for name in SCALAR_ORDER:
                    scalars[name][k] = getattr(state, name)
                scalars["rho"][k] = state.rho
                pi_draws[k] = state.pi
                transition_draws[k] = state.transition
                segment_counts[k] = state.segment_counts
                for s in range(K):
                    relation_counts[k, s] = int(precedence_from_u(
                        state.u_by_skill[s]).sum())
                log_target[k] = state.components["log_target"]
                log_blocks[k] = state.components["log_block_likelihood"]
                if store_keys:
                    boundary_keys.append(tuple(key_of(s) for s in state.segmentations))
                if labels is not None:
                    for n, arr in enumerate(state.occurrence_labels()):
                        labels[k, n, :len(arr)] = arr
                k += 1
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    chain {chain}: sweep {i + 1:,}/{num_sweeps:,} "
                  f"({time.perf_counter() - began:.0f}s)", flush=True)
        if checkpoint_path and checkpoint_every and (i + 1) % checkpoint_every == 0:
            _write_checkpoint(checkpoint_path, chain, state, i + 1, k, u_draws, scalars,
                              pi_draws, transition_draws, segment_counts,
                              relation_counts, log_target, log_blocks, labels)
    runtime = time.perf_counter() - began

    result = Stage6EChainResult(
        u_draws=u_draws[:k], scalars={n: v[:k] for n, v in scalars.items()},
        pi_draws=pi_draws[:k], transition_draws=transition_draws[:k],
        segment_counts=segment_counts[:k], boundary_keys=boundary_keys,
        occurrence_labels=(labels[:k] if labels is not None else np.empty((0,))),
        log_target=log_target[:k], log_block_likelihood=log_blocks[:k],
        relation_counts=relation_counts[:k],
        proposed=dict(state.proposed), accepted=dict(state.accepted),
        invalid=dict(state.invalid), proposed_after_burn_in=post_proposed,
        accepted_after_burn_in=post_accepted, movement=movement, final_state=state,
        runtime_seconds=runtime, chain=chain, seed=seed)
    result.table_builds = int(sampler.tables.builds)
    result.table_build_seconds = float(sampler.tables.build_seconds)
    result.collapsed_records = collapsed_records
    result.collapsed_config = {"every": int(collapsed.every),
                               "scale": float(collapsed.scale)}
    result.collapsed_likelihood_stats = {
        "evaluations": int(likelihood.evaluations),
        "cache_hits": int(likelihood.cache_hits)}
    return result
