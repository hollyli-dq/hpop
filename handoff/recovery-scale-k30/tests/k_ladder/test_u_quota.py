"""The registered proportional-effort rule: constant `U` attempts per role vector.

Seven properties are required of this rule. Each is tested here against the thing that
would go wrong without it, not merely against its own restatement.
"""

from __future__ import annotations

import inspect
import io
import re
import tokenize

import numpy as np
import pytest

from hpop.mcmc_cpa import u_quota
from hpop.mcmc_cpa.u_quota import (ROLE_VECTOR_DIM, assert_proposal_unit_is_role_vector,
                                   attempts_per_role_summary, distribute_quota,
                                   quota_schedule, total_quota, update_events)

LADDER = (3, 5, 10, 20, 30)
M_ROLES = 10
SWEEPS, WARMUP, EVERY = 50_000, 20_000, 10


def code_only(module) -> str:
    """Source with every comment and string literal removed.

    A source-level check that looks at raw text will match the module's own prose --
    including the docstring that names the rejected rule in order to reject it. Tokenising
    and dropping COMMENT and STRING leaves executable code only, which is what these
    checks are actually about.
    """
    source = inspect.getsource(module)
    kept = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept)


def code_only_fn(fn) -> str:
    """As `code_only`, for a single function."""
    kept = []
    for token in tokenize.generate_tokens(io.StringIO(inspect.getsource(fn)).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept)


def schedules(target=166.7, sweeps=SWEEPS, warmup=WARMUP, every=EVERY):
    return {k: quota_schedule(target, k, M_ROLES, sweeps, warmup, every)
            for k in LADDER}


# ------------------------------------------------- 1. realised total equals M_K
def test_the_distributed_counts_sum_to_the_quota_exactly():
    for k, s in schedules().items():
        assert int(s["moves_per_event"].sum()) == s["total_quota_M_K"], f"K={k}"


def test_the_quota_is_round_target_times_k_times_m():
    for target in (50.0, 100.0, 166.7):
        for k in LADDER:
            assert total_quota(target, k, M_ROLES) == round(target * k * M_ROLES)


@pytest.mark.parametrize("total,events", [(0, 5), (1, 7), (7, 1), (1667, 100),
                                          (50_010, 5_000), (3, 5), (99, 100)])
def test_the_cumulative_split_is_exact_for_awkward_cases(total, events):
    d = distribute_quota(total, events)
    assert d.size == events
    assert int(d.sum()) == total
    assert (d >= 0).all()


def test_a_quota_cannot_be_placed_in_no_events():
    assert distribute_quota(0, 0).size == 0
    with pytest.raises(ValueError):
        distribute_quota(5, 0)


# ------------------- 2. per-role effort differs across rungs only by rounding
def test_attempts_per_role_are_equal_across_rungs_up_to_rounding():
    """The whole point. The bound is the unavoidable one-proposal rounding of `M_K`,
    which is `0.5 / (K*m)` per role -- far below one proposal."""
    s = schedules()
    per_role = {k: v["mean_attempts_per_role_total"] for k, v in s.items()}
    spread = max(per_role.values()) - min(per_role.values())
    assert spread <= 0.5 / (min(LADDER) * M_ROLES), per_role


def test_each_event_receives_a_count_within_one_of_every_other():
    for k, s in schedules().items():
        counts = s["moves_per_event"]
        assert int(counts.max() - counts.min()) <= 1, f"K={k}: {counts.min()}..{counts.max()}"


def test_the_old_fixed_cadence_really_did_fall_as_one_over_k():
    """The control. Without this rule, effort per role is `sweeps/(every*K*m)` -- a
    tenfold gradient from K=3 to K=30. If this ever stops holding, the motivation for the
    module has changed and the rule should be revisited rather than silently kept."""
    fixed = {k: SWEEPS / (EVERY * k * M_ROLES) for k in LADDER}
    assert fixed[3] / fixed[30] == pytest.approx(10.0)
    assert fixed[3] > fixed[30]


# ------------------------------------------- 3. no floor/ceil of K/3 anywhere
def test_no_rounded_k_over_three_rule_is_used():
    """`round(K/3)` is non-integral at K = 5, 10, 20; flooring or ceiling it reintroduces
    exactly the unequal effort this module removes. Checked in the source, because the
    failure would be invisible in the output at K = 3 and K = 30 where K/3 is integral."""
    code = code_only(u_quota)
    assert not re.search(r"(floor|ceil|round|int)\s*\(\s*[^)]*\bk\b\s*/\s*3",
                         code, re.IGNORECASE), "a K/3 rounding rule appeared in code"
    assert "// 3" not in code, "integer division by 3 appeared in code"
    # and the prose that names the rejected rule is documentation, not an implementation
    assert "round(K/3)" in inspect.getsource(u_quota), \
        "the module should still explain why the rounded rule is rejected"


def test_a_rounded_per_event_constant_would_fail_the_equal_effort_test():
    """Demonstrates the rejected alternative actually is worse, rather than asserting it."""
    per_role = {}
    for k in LADDER:
        moves = max(1, round(k / 3))                 # the rejected rule
        events = update_events(SWEEPS, EVERY).size
        per_role[k] = moves * events / (k * M_ROLES)
    spread = max(per_role.values()) - min(per_role.values())
    assert spread > 1.0, ("round(K/3) was expected to give visibly unequal effort; "
                          f"got {per_role}")


# ---------------------------- 4. burn-in and retained recorded separately
def test_burnin_and_retained_quotas_are_separate_and_sum_to_the_total():
    for k, s in schedules().items():
        assert s["burnin_quota"] + s["retained_quota"] == s["total_quota_M_K"], f"K={k}"
        assert s["burnin_quota"] > 0 and s["retained_quota"] > 0
        assert s["n_events_burnin"] + s["n_events_retained"] == s["n_events"]


def test_the_phase_split_is_proportional_to_event_count():
    for k, s in schedules().items():
        expected = s["total_quota_M_K"] * s["n_events_burnin"] / s["n_events"]
        assert abs(s["burnin_quota"] - expected) <= 1, f"K={k}"


def test_both_phases_have_equal_per_role_effort_across_rungs():
    s = schedules()
    for phase in ("mean_attempts_per_role_burnin", "mean_attempts_per_role_retained"):
        values = [v[phase] for v in s.values()]
        assert max(values) - min(values) <= 0.05, (phase, values)


# --------------------------------- 5. the proposal unit is verified, not assumed
def test_one_u_move_updates_one_complete_role_vector():
    info = assert_proposal_unit_is_role_vector()
    assert info["coordinates_moved_per_proposal"] == ROLE_VECTOR_DIM
    assert info["denominator"] == "K * m"


def test_a_scalar_coordinate_kernel_would_be_caught():
    """If the kernel is ever changed to move one coordinate, the denominator must become
    `d*K*m`. The check must fail rather than silently normalise by the wrong thing."""
    import hpop.mcmc_original.sampler_u as sampler_u

    original = sampler_u.propose_row

    def scalar_kernel(u, row, sigma_u, rng):
        candidate = np.array(u, dtype=float, copy=True)
        candidate[row, 0] += sigma_u * rng.normal()      # ONE coordinate only
        return candidate

    sampler_u.propose_row = scalar_kernel
    try:
        with pytest.raises(AssertionError, match="d\\*K\\*m|complete role vector"):
            assert_proposal_unit_is_role_vector()
    finally:
        sampler_u.propose_row = original


def test_the_denominator_uses_k_times_m_not_two_k_m():
    s = quota_schedule(100.0, 10, M_ROLES, SWEEPS, WARMUP, EVERY)
    assert s["denominator_role_vectors"] == 10 * M_ROLES
    assert s["total_quota_M_K"] == round(100.0 * 10 * M_ROLES)


# ------------------------- 6. realised per-role counts are logged with min/median/max
def test_the_summary_reports_min_median_max_and_spread():
    counts = np.array([[5, 6, 5], [6, 5, 6]])
    summary = attempts_per_role_summary(counts)
    assert summary == {"n_roles": 6, "min": 5, "median": 5.5, "max": 6, "spread": 1,
                       "total": 33, "mean": 5.5}


def test_an_empty_summary_does_not_explode():
    assert attempts_per_role_summary(np.zeros((0, 0)))["n_roles"] == 0


# ------------------------------------- 7. every schedule decision is deterministic
def test_the_schedule_is_deterministic_and_carries_no_randomness():
    a = quota_schedule(166.7, 30, M_ROLES, SWEEPS, WARMUP, EVERY)
    b = quota_schedule(166.7, 30, M_ROLES, SWEEPS, WARMUP, EVERY)
    np.testing.assert_array_equal(a["moves_per_event"], b["moves_per_event"])
    np.testing.assert_array_equal(a["events"], b["events"])
    assert a["total_quota_M_K"] == b["total_quota_M_K"]


def test_the_quota_module_draws_no_random_numbers():
    """The counts must be fixed by the design index alone. Only the *choice* of which
    role to move is random, and that lives in the registered CRN namespace."""
    for fn in (quota_schedule, distribute_quota, total_quota, update_events):
        body = code_only_fn(fn)
        for forbidden in ("default_rng", "shuffle", "permutation", "Generator"):
            assert forbidden not in body, f"{forbidden} appeared in {fn.__name__}"


def test_the_schedule_does_not_depend_on_a_seed_argument():
    signature = inspect.signature(quota_schedule)
    assert not any("seed" in p or "rng" in p for p in signature.parameters)


# --------------- the constant is a MEAN, not a per-role guarantee: pin the distinction
def test_the_kernel_selects_the_role_by_uniform_random_scan():
    """If this ever becomes balanced targeting, the per-role claim may be strengthened --
    but only then, and only deliberately."""
    import hpop.mcmc_cpa.collapsed_u as collapsed_u

    src = inspect.getsource(collapsed_u.collapsed_u_mh_step_cpa)
    assert "rng.integers(k)" in src and "rng.integers(m)" in src, \
        "role selection is no longer a uniform random scan; revisit the per-role claim"


def test_realised_per_role_counts_are_random_not_equal():
    """The rule fixes the expected effort per role. Realised counts are multinomial and
    genuinely spread; claiming every role gets the same number would be false."""
    target, k, m = 166.7, 30, M_ROLES
    quota = total_quota(target, k, m)
    cells = k * m
    counts = np.bincount(np.random.default_rng(11).integers(cells, size=quota),
                         minlength=cells)
    summary = attempts_per_role_summary(counts)
    assert summary["spread"] > 20, ("per-role counts came out suspiciously equal; a "
                                    "random scan should spread them")
    assert summary["max"] / summary["min"] > 1.2


def test_the_expected_spread_is_the_same_at_every_rung():
    """What the rule holds constant across K is the per-role LAW -- mean and sd -- not
    the min and max, which widen with K only as an order-statistic effect."""
    for k in LADDER:
        s = quota_schedule(166.7, k, M_ROLES, SWEEPS, WARMUP, EVERY)
        assert s["mean_attempts_per_role_total"] == pytest.approx(166.7, abs=0.05)
        assert s["per_role_sd_expected"] == pytest.approx(np.sqrt(166.7), rel=0.01)


def test_the_schedule_declares_that_the_figures_are_means():
    s = quota_schedule(166.7, 30, M_ROLES, SWEEPS, WARMUP, EVERY)
    assert "MEANS, not guarantees" in s["per_role_targeting"]
    assert "uniform random scan" in s["per_role_targeting"]


def test_the_module_does_not_claim_equal_per_role_counts():
    """Guards the prose as well as the code: the docstring must not promise what the
    random scan cannot deliver."""
    doc = " ".join((inspect.getdoc(u_quota) or "").split())   # normalise line breaks
    assert "not a per-role guarantee" in doc
    assert "mean scheduled attempts per role vector" in doc
    assert "It is a mean, not a per-role guarantee" in doc
