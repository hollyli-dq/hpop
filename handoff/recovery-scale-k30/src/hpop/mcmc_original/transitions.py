"""Skill-to-skill transition matrix with a conjugate Dirichlet Gibbs update.

Stage 3 adds exactly one new object to the model: ``P[h, k]``, the probability that
skill ``h`` is followed by skill ``k``. Nothing else changes — no durations, no
composition model, no recurrence.

Self-transitions are disallowed in this toy (two adjacent segments of the same skill
would be indistinguishable from one longer segment under a support-compatible
tiling), so each row is a distribution over

    allowed_next(h) = {0, ..., K-1} \\ {h}

with an independent Dirichlet(eta, ..., eta) prior on that restricted row. Given the
currently sampled skill paths and their adjacent-transition counts ``C[h, k]``, the
row posterior is conjugate:

    P[h, allowed_next(h)] | S  ~  Dirichlet(eta + C[h, k] : k in allowed_next(h))

which is what :func:`sample_transition_matrix` draws from.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "allowed_next",
    "transition_counts",
    "dirichlet_posterior_params",
    "sample_transition_matrix",
    "posterior_mean_transition_matrix",
    "log_transition_matrix",
    "run_segmentation_transition_gibbs",
]


def allowed_next(h: int, n_skills: int) -> tuple[int, ...]:
    """Skills that may follow ``h`` — everything but ``h`` itself."""
    if not (0 <= h < n_skills):
        raise ValueError(f"h={h} outside range({n_skills})")
    return tuple(k for k in range(n_skills) if k != h)


def transition_counts(paths, n_skills: int) -> np.ndarray:
    """``C[h, k]`` = number of adjacent ``h -> k`` transitions across all paths."""
    counts = np.zeros((n_skills, n_skills), dtype=float)
    for path in paths:
        path = [int(z) for z in path]
        for prev, nxt in zip(path[:-1], path[1:]):
            if not (0 <= prev < n_skills and 0 <= nxt < n_skills):
                raise ValueError(f"path {path} contains a skill outside range({n_skills})")
            if prev == nxt:
                raise ValueError(
                    f"self-transition {prev} -> {nxt} is not allowed in this toy"
                )
            counts[prev, nxt] += 1.0
    return counts


def dirichlet_posterior_params(
    counts: np.ndarray, n_skills: int, eta: float = 1.0
) -> dict[int, tuple[tuple[int, ...], np.ndarray]]:
    """Posterior Dirichlet parameters for each allowable row.

    Returns ``{h: (allowed_next(h), alpha)}`` with ``alpha_k = eta + C[h, k]``.
    """
    if eta <= 0.0:
        raise ValueError(f"eta must be positive, got {eta}")
    out = {}
    for h in range(n_skills):
        allowed = allowed_next(h, n_skills)
        alpha = np.array([eta + counts[h, k] for k in allowed], dtype=float)
        out[h] = (allowed, alpha)
    return out


def sample_transition_matrix(
    counts: np.ndarray, n_skills: int, rng: np.random.Generator, eta: float = 1.0
) -> np.ndarray:
    """One Gibbs draw of ``P`` from the conjugate row posteriors. Diagonal is 0."""
    params = dirichlet_posterior_params(counts, n_skills, eta)
    p = np.zeros((n_skills, n_skills), dtype=float)
    for h, (allowed, alpha) in params.items():
        draw = rng.dirichlet(alpha)
        for k, value in zip(allowed, draw):
            p[h, k] = value
    return p


def posterior_mean_transition_matrix(
    counts: np.ndarray, n_skills: int, eta: float = 1.0
) -> np.ndarray:
    """Analytic posterior mean ``E[P[h,k]] = alpha_k / sum(alpha)``. Diagonal is 0."""
    params = dirichlet_posterior_params(counts, n_skills, eta)
    p = np.zeros((n_skills, n_skills), dtype=float)
    for h, (allowed, alpha) in params.items():
        mean = alpha / alpha.sum()
        for k, value in zip(allowed, mean):
            p[h, k] = value
    return p


def log_transition_matrix(p: np.ndarray) -> np.ndarray:
    """Elementwise log with the forbidden self-transitions sent to -inf."""
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(p, dtype=float))


def run_segmentation_transition_gibbs(
    trace_states_list,
    evaluators,
    u_by_skill,
    n_skills: int,
    log_pi: np.ndarray,
    n_iterations: int,
    burn_in: int,
    thin: int,
    rng: np.random.Generator,
    eta: float = 1.0,
) -> dict:
    """Alternate segmentation MH and Dirichlet Gibbs on ``P`` — optional Stage 3C.

    ``U`` is held **fixed** throughout, so the emission term of every state is a
    constant and the only thing moving the segmentations is the transition context.
    That isolation is the point: it tests the transition machinery without entangling
    it with latent-order inference.
    """
    from hpop.mcmc_original.sampler_segmentation import mh_segmentation_step
    from hpop.mcmc_original.targets import log_path_prior

    tables = [evaluators[k].log_table(u_by_skill[k]) for k in range(n_skills)]
    emissions = [
        np.array(
            [
                sum(float(tables[skill][idx]) for skill, idx in descriptor)
                for descriptor in ts.descriptors
            ],
            dtype=float,
        )
        for ts in trace_states_list
    ]

    states = [int(rng.integers(ts.n_states)) for ts in trace_states_list]
    p = sample_transition_matrix(np.zeros((n_skills, n_skills)), n_skills, rng, eta)

    saved_p: list[np.ndarray] = []
    saved_counts: list[np.ndarray] = []
    accepted = proposed = 0

    for i in range(n_iterations):
        log_p = log_transition_matrix(p)

        # 1. segmentation update conditional on the current P
        for t, ts in enumerate(trace_states_list):
            constants = np.array(
                [log_path_prior(path, log_pi, log_p) for path in ts.paths], dtype=float
            )
            log_targets = ts.log_boundary + constants + emissions[t]
            states[t], was_accepted, was_proposed = mh_segmentation_step(
                states[t], log_targets, rng
            )
            accepted += int(was_accepted)
            proposed += int(was_proposed)

        # 2. count the sampled transitions, 3. Gibbs the transition rows
        counts = transition_counts(
            [ts.paths[states[t]] for t, ts in enumerate(trace_states_list)], n_skills
        )
        p = sample_transition_matrix(counts, n_skills, rng, eta)

        if i >= burn_in and (i - burn_in) % thin == 0:
            saved_p.append(p.copy())
            saved_counts.append(counts.copy())

    saved_p_arr = np.array(saved_p)
    return {
        "p_samples": saved_p_arr,
        "p_mean": saved_p_arr.mean(axis=0),
        "count_samples": np.array(saved_counts),
        "count_mean": np.array(saved_counts).mean(axis=0),
        "final_states": list(states),
        "acceptance_rate": accepted / proposed if proposed else float("nan"),
        "n_iterations": n_iterations,
        "burn_in": burn_in,
        "thin": thin,
    }
