"""Proportional-effort `U` quota: MEAN attempted updates per role vector, flat in `K`.

The constant this rule holds across the ladder is the *expected* number of proposals each
role-level latent vector receives. It is a mean, not a per-role guarantee -- see "What the
constant is, precisely" below before quoting any per-role figure.

## The problem

At a fixed sweep count and fixed cadence, a chain makes the same number of `U` proposals
whatever `K` is, and those proposals are spread over `K * m` role vectors. So each role
vector receives an effort that falls as `1/K`: 166.7 attempts at `K = 3` against 16.7 at
`K = 30`. That gradient runs along the very axis a recovery-versus-`K` study measures, and
after the fact "recovery degrades with `K`" and "`K = 30` received ten times less sampler
effort" cannot be separated.

## The registered rule

Hold the FFBS schedule fixed and scale the number of `U` proposals instead. One registered
parameter sets the effort:

    target_u_attempts_per_role

and the chain's total quota is

    M_K = round(target_u_attempts_per_role * K * m)

Scaling the proposals rather than the sweeps is what makes this affordable. The total `U`
cost is `M_K * (cost per move)` however the moves are distributed, so lengthening the chain
instead would inflate the FFBS bill by the same factor while buying no extra `U` effort --
and the FFBS sweeps were never the deficient resource.

## The proposal unit, verified rather than assumed

`sampler_u.propose_row` sets `candidate[row] = candidate[row] + sigma * normal(size=d)`:
one move perturbs **one complete role-level `d`-vector**, not one scalar coordinate. The
denominator is therefore `K * m`. If the kernel is ever changed to move a single
coordinate, the denominator becomes `d * K * m` and `assert_proposal_unit_is_role_vector`
fails loudly rather than letting the normalisation drift.

## Distributing the quota

## What the constant is, precisely

`M_K / (K * m)` is the **mean scheduled attempts per role vector**, not a per-role
guarantee. The kernel selects `(skill, row)` by uniform random scan
(`rng.integers(K)`, `rng.integers(m)`), so realised counts are
`Multinomial(M_K, uniform over K*m cells)`: mean `M_K/(K*m)`, standard deviation
`sqrt(M_K/(K*m))` -- about 12.9 at a mean of 166.7, at **every** rung.

That distinction matters and must survive into the write-up. What the rule makes constant
across `K` is the *expected* effort per role and its spread; what it does not make
constant is the min and max across roles, which widen with `K` purely because more roles
give more chances at an extreme draw (max/min about 1.25 at `K = 3` against about 1.6 at
`K = 30`). This is an order-statistic effect on a fixed per-role law, categorically
different from the systematic tenfold gradient in the *mean* that the rule removes.

Balanced targeting would make every role's count deterministic, but it would replace the
random-scan kernel with a systematic-scan one -- a different algorithm needing its own
registration and validation. Random scan is standard, valid, and left alone. Realised
per-role counts are therefore reported as **diagnostics** (`attempts_per_role_summary`),
never asserted.

## Distributing the quota

`M_K` is spread over the `E` registered `U`-update events by a cumulative quota,

    moves_at_event_e = floor(e * M_K / E) - floor((e - 1) * M_K / E)

which is exact -- the counts sum to `M_K` by telescoping -- and as even as integers allow,
with every event receiving `floor(M_K/E)` or `ceil(M_K/E)`. A rounded per-event constant
such as `round(K/3)` is **not** used and must never be: at `K = 5, 10, 20` it is not
integral, and flooring or ceiling it reintroduces exactly the unequal per-role effort this
module exists to remove.

The burn-in and retained phases are partitioned at the warm-up boundary of that single
cumulative schedule, so each phase's share is proportional to its event count, exact, and
deterministic. Both are recorded separately: `u_scale` is tuned during warm-up and frozen
afterwards, so proposals made under a moving scale must never be pooled with proposals made
under the frozen one.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ROLE_VECTOR_DIM", "total_quota", "distribute_quota", "quota_schedule",
           "update_events", "assert_proposal_unit_is_role_vector",
           "attempts_per_role_summary"]

#: The dimension of one role-level latent vector in the registered library.
ROLE_VECTOR_DIM = 2


def assert_proposal_unit_is_role_vector(n_roles: int = 6, d: int = ROLE_VECTOR_DIM,
                                        seed: int = 0) -> dict:
    """Confirm one `U` move updates a whole role vector. Raises if it does not.

    The quota's denominator depends on this. Checking it here means a change to the
    proposal kernel surfaces as a failure rather than as a silently wrong normalisation.
    """
    from hpop.mcmc_original.sampler_u import propose_row

    u = np.zeros((int(n_roles), int(d)))
    row = min(3, int(n_roles) - 1)
    candidate = propose_row(u, row, 0.5, np.random.default_rng(int(seed)))
    moved_rows = np.flatnonzero((candidate != u).any(axis=1))
    moved_coords = np.flatnonzero(candidate[row] != u[row])
    if moved_rows.tolist() != [row]:
        raise AssertionError(
            f"a U move touched rows {moved_rows.tolist()}, expected exactly [{row}]")
    if moved_coords.size != int(d):
        raise AssertionError(
            f"a U move changed {moved_coords.size} of {d} coordinates in its row; the "
            f"proposal unit is no longer a complete role vector, so the quota "
            f"denominator must become d*K*m rather than K*m")
    return {"proposal_unit": "one complete role-level vector",
            "coordinates_moved_per_proposal": int(moved_coords.size),
            "denominator": "K * m",
            "denominator_if_scalar_kernel": "d * K * m"}


def total_quota(target_u_attempts_per_role: float, n_skills: int, n_roles: int) -> int:
    """`M_K = round(target * K * m)`, the whole chain's `U` proposal budget."""
    if target_u_attempts_per_role < 0:
        raise ValueError("target_u_attempts_per_role must be non-negative, got "
                         f"{target_u_attempts_per_role}")
    return int(round(float(target_u_attempts_per_role) * int(n_skills) * int(n_roles)))


def distribute_quota(total: int, events: int) -> np.ndarray:
    """Cumulative-quota split: `floor(e*M/E) - floor((e-1)*M/E)` for `e = 1..E`.

    Sums to `total` exactly and differs by at most one between any two events.
    """
    total, events = int(total), int(events)
    if events <= 0:
        if total:
            raise ValueError(f"cannot place {total} proposals in {events} events")
        return np.zeros(0, dtype=np.int64)
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    cumulative = (np.arange(events + 1, dtype=np.int64) * total) // events
    return np.diff(cumulative)


def update_events(sweeps: int, u_every: int) -> np.ndarray:
    """The sweeps at which a registered `U`-update event occurs."""
    sweeps, u_every = int(sweeps), int(u_every)
    if u_every <= 0:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, sweeps, u_every, dtype=np.int64)


def quota_schedule(target_u_attempts_per_role: float, n_skills: int, n_roles: int,
                   sweeps: int, warmup: int, u_every: int) -> dict:
    """The full deterministic schedule, split at the warm-up boundary.

    Returns per-event counts plus the realised attempts per role in each phase, so a run
    can record what it actually did rather than what it intended.
    """
    events = update_events(sweeps, u_every)
    quota = total_quota(target_u_attempts_per_role, n_skills, n_roles)
    per_event = distribute_quota(quota, events.size)
    is_burnin = events < int(warmup)

    denominator = int(n_skills) * int(n_roles)
    burnin_total = int(per_event[is_burnin].sum())
    retained_total = int(per_event[~is_burnin].sum())
    return {
        "target_u_attempts_per_role": float(target_u_attempts_per_role),
        "n_skills": int(n_skills), "n_roles": int(n_roles),
        "denominator_role_vectors": denominator,
        "proposal_unit": "one complete role-level vector (verified, not assumed)",
        "total_quota_M_K": int(quota),
        "events": events, "moves_per_event": per_event,
        "n_events": int(events.size),
        "n_events_burnin": int(is_burnin.sum()),
        "n_events_retained": int((~is_burnin).sum()),
        "burnin_quota": burnin_total,
        "retained_quota": retained_total,
        "mean_attempts_per_role_total": quota / denominator if denominator else 0.0,
        "mean_attempts_per_role_burnin": burnin_total / denominator if denominator else 0.0,
        "mean_attempts_per_role_retained": (retained_total / denominator
                                       if denominator else 0.0),
        "per_role_targeting": ("uniform random scan over (skill, row); the reported "
                              "per-role figures are MEANS, not guarantees"),
        "per_role_sd_expected": (float(np.sqrt(quota / denominator))
                                 if denominator else 0.0),
        "realised_per_role_counts": ("not determined by the schedule; measure with "
                                     "attempts_per_role_summary and report min/median/max"),
        "distribution_rule": ("cumulative quota floor(e*M/E)-floor((e-1)*M/E); NOT a "
                             "rounded per-event constant such as round(K/3), which is "
                             "non-integral at K = 5, 10, 20 and whose floor/ceil "
                             "reintroduces unequal per-role effort"),
        "phase_split_rule": ("the single cumulative schedule partitioned at the warm-up "
                             "boundary, so each phase's share is proportional to its "
                             "event count, exact and deterministic"),
    }


def attempts_per_role_summary(role_counts) -> dict:
    """Realised attempts per role vector: min, median, max, and the spread.

    The spread is the quantity the registered rule constrains. With a cumulative quota it
    is bounded by the unavoidable one-proposal rounding, but it is measured rather than
    asserted because the *selection* of which role to move is random even though the
    *count* of moves is deterministic.
    """
    counts = np.asarray(role_counts, dtype=np.int64).ravel()
    if counts.size == 0:
        return {"n_roles": 0, "min": 0, "median": 0.0, "max": 0, "spread": 0,
                "total": 0, "mean": 0.0}
    return {"n_roles": int(counts.size), "min": int(counts.min()),
            "median": float(np.median(counts)), "max": int(counts.max()),
            "spread": int(counts.max() - counts.min()), "total": int(counts.sum()),
            "mean": float(counts.mean())}
