"""Metropolis-Hastings over segmentations, on the enumerated finite state space.

Deliberately the simplest correct sampler: a **symmetric independence proposal**
that picks uniformly among the legal states other than the current one. With ``N``
legal states,

    q(S' | S) = 1 / (N - 1)   for every S' != S

which is symmetric, so the proposal densities cancel and

    log alpha = log_target(S') - log_target(S).

Accept when ``log Uniform(0,1) < min(0, log alpha)``.

No split, merge, shift or relabel moves — those are later work. A trace with a single
legal state has nothing to propose and the update is a no-op.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "propose_other_state",
    "mh_segmentation_step",
    "run_segmentation_mcmc",
    "state_log_prior_constants",
    "run_joint_mcmc",
]


def propose_other_state(n_states: int, current: int, rng: np.random.Generator) -> int:
    """Uniform draw from every state except ``current``; a no-op if there is only one."""
    if n_states < 1:
        raise ValueError("n_states must be at least 1")
    if not (0 <= current < n_states):
        raise ValueError(f"current={current} outside range({n_states})")
    if n_states == 1:
        return current
    candidate = int(rng.integers(n_states - 1))
    return candidate if candidate < current else candidate + 1


def mh_segmentation_step(
    current: int, log_targets: np.ndarray, rng: np.random.Generator
) -> tuple[int, bool, bool]:
    """One symmetric-independence MH step.

    Returns:
        ``(new_state, accepted, proposed)``. ``proposed`` is False when the trace has
        only one legal state, so acceptance rates are not diluted by no-ops.
    """
    n_states = len(log_targets)
    if n_states == 1:
        return current, False, False

    candidate = propose_other_state(n_states, current, rng)
    log_alpha = float(log_targets[candidate]) - float(log_targets[current])
    if log_alpha >= 0.0 or math.log(rng.random()) < log_alpha:
        return candidate, True, True
    return current, False, True


def run_segmentation_mcmc(
    log_targets: np.ndarray,
    n_iterations: int,
    burn_in: int,
    rng: np.random.Generator,
    init: int = 0,
) -> dict:
    """Run the segmentation sampler against a **fixed** log-target vector.

    This is the Stage-1 setting: U is fixed, so the target over the finite state
    space is a constant vector and the chain's only job is to reproduce it.
    """
    log_targets = np.asarray(log_targets, dtype=float)
    if n_iterations <= burn_in:
        raise ValueError("n_iterations must exceed burn_in")

    state = int(init)
    trace = np.empty(n_iterations, dtype=int)
    accepted = 0
    proposed = 0
    for i in range(n_iterations):
        state, was_accepted, was_proposed = mh_segmentation_step(state, log_targets, rng)
        accepted += int(was_accepted)
        proposed += int(was_proposed)
        trace[i] = state

    kept = trace[burn_in:]
    return {
        "trace": trace,
        "kept": kept,
        "n_iterations": n_iterations,
        "burn_in": burn_in,
        "n_accepted": accepted,
        "n_proposed": proposed,
        "acceptance_rate": accepted / proposed if proposed else float("nan"),
    }


def state_log_prior_constants(
    trace_states, log_pi: np.ndarray, log_transition: np.ndarray | None = None
) -> np.ndarray:
    """Boundary prior + skill-path prior per state — the U-independent part.

    Precomputing this is what keeps the joint sampler cheap: only the emission term
    changes when U moves.
    """
    from hpop.mcmc_original.targets import log_path_prior

    return np.array(
        [
            trace_states.log_boundary[s]
            + log_path_prior(trace_states.paths[s], log_pi, log_transition)
            for s in range(trace_states.n_states)
        ],
        dtype=float,
    )


def run_joint_mcmc(
    trace_states_list,
    evaluators,
    init_u,
    rho: float,
    sigma_u: float,
    n_iterations: int,
    burn_in: int,
    thin: int,
    rng: np.random.Generator,
    log_pi: np.ndarray,
    log_transition: np.ndarray | None = None,
) -> dict:
    """Alternate segmentation MH and row-wise U MH — Stage 2B.

    Each iteration:

    1. **Segmentation update.** For every trace, propose another legal state and
       accept by symmetric MH, scored with the *current* U.
    2. **U update.** Aggregate the role sequences the current segmentations assign to
       each skill into permutation counts, then run one row-wise MH sweep per skill.

    No transition matrix, no durations, no composition model.
    """
    from hpop.mcmc_original.sampler_u import log_u_prior, u_row_sweep

    n_skills = len(evaluators)
    u = {k: np.array(init_u[k], dtype=float, copy=True) for k in range(n_skills)}
    constants = [
        state_log_prior_constants(ts, log_pi, log_transition) for ts in trace_states_list
    ]
    states = [int(rng.integers(ts.n_states)) for ts in trace_states_list]

    saved_states: list[list[int]] = []
    saved_u: dict[int, list[np.ndarray]] = {k: [] for k in range(n_skills)}
    log_post = np.empty(n_iterations, dtype=float)
    seg_accepted = seg_proposed = 0
    u_accepted = u_proposed = 0

    for i in range(n_iterations):
        tables = [evaluators[k].log_table(u[k]) for k in range(n_skills)]

        # 1. segmentation update, one MH step per trace
        total_emission = 0.0
        for t, ts in enumerate(trace_states_list):
            emissions = np.array(
                [
                    sum(float(tables[skill][idx]) for skill, idx in descriptor)
                    for descriptor in ts.descriptors
                ],
                dtype=float,
            )
            log_targets = constants[t] + emissions
            states[t], accepted, proposed = mh_segmentation_step(
                states[t], log_targets, rng
            )
            seg_accepted += int(accepted)
            seg_proposed += int(proposed)
            total_emission += float(log_targets[states[t]])

        # 2. U update, given the currently sampled segmentations
        counts = {
            k: np.zeros(len(evaluators[k].permutations), dtype=float)
            for k in range(n_skills)
        }
        for t, ts in enumerate(trace_states_list):
            for skill, idx in ts.descriptors[states[t]]:
                counts[skill][idx] += 1.0

        for k in range(n_skills):
            if counts[k].sum() == 0:
                continue

            def log_likelihood(candidate, k=k):
                return evaluators[k].counts_log_likelihood(candidate, counts[k])

            u[k], _, accepted, proposed = u_row_sweep(
                u[k], log_likelihood, rho, sigma_u, rng
            )
            u_accepted += accepted
            u_proposed += proposed

        log_post[i] = total_emission + sum(log_u_prior(u[k], rho) for k in range(n_skills))
        if i >= burn_in and (i - burn_in) % thin == 0:
            saved_states.append(list(states))
            for k in range(n_skills):
                saved_u[k].append(u[k].copy())

    return {
        "states": np.array(saved_states, dtype=int),
        "u_samples": {k: np.array(v) for k, v in saved_u.items()},
        "log_posterior": log_post,
        "log_posterior_kept": log_post[burn_in:],
        "segmentation_acceptance_rate": (
            seg_accepted / seg_proposed if seg_proposed else float("nan")
        ),
        "u_acceptance_rate": u_accepted / u_proposed if u_proposed else float("nan"),
        "n_iterations": n_iterations,
        "burn_in": burn_in,
        "thin": thin,
    }
