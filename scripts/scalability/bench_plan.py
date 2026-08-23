"""The registered benchmark plan: every configuration, fixed before the first run.

The plan is a pure function of nothing -- it takes no arguments and reads no clock -- so
the configuration list is identical on a resume, and a partially completed run continues
against the same set it started against. `plan_digest()` pins it, and the driver refuses
to resume onto a state file written under a different digest.

Axes follow the study specification:

    baseline      the existing matched scale, reproduced (Section 8)
    J             trace length, at fixed N, K, A, D                  (Section 9)
    K             skill library size, dense transition dynamics      (Section 10)
    N             corpus size                                        (Section 11)
    D             maximum legal segment width                        (Section 12)
    A_full        role inventory, every skill supported on all A     (Section 13, A)
    A_sparse      role inventory, ten roles per skill                (Section 13, B)
    target        the anticipated real-data operating point          (Section 16)
    target_long   long-trace primitives at the target's K and A      (Section 16)

Conditional points (`J = 1024`, `K = 80`) carry `conditional_on`, and the driver runs them
only when the named predecessor completed inside its budget and inside the memory gate.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_common as bc                                              # noqa: E402

ALL_GROUPS = ("build", "primitives", "cond", "marg")
PRIMITIVE_GROUPS = ("build", "primitives")

# Wall-clock ceilings from Section 15, in seconds, per (configuration, group).
TIMEOUT_STANDARD = 12 * 60
TIMEOUT_TARGET = 30 * 60


def _cfg(axis, label, **kwargs) -> bc.BenchConfig:
    return bc.BenchConfig(axis=axis, label=label, **kwargs)


def baseline_configs() -> list:
    """Section 8. N = 100 with 25 traces at each of J in {24, 32, 40, 48}."""
    mix = tuple(J for J in (24, 32, 40, 48) for _ in range(25))
    return [_cfg("baseline", "baseline_matched_scale", N=100, J=48, K=3, A=5, D_max=12,
                 D_min=3, regime=bc.FULL_SUPPORT, length_mix=mix,
                 timeout_s=TIMEOUT_STANDARD,
                 note="the existing matched FULL-LATENT scale, reproduced on this "
                      "machine; J is a mixture, so the J field records only the "
                      "longest class")]


def j_configs() -> list:
    """Section 9. N = 16 equal-length traces, K = 10, A = 20, D in [3, 12]."""
    out = []
    for J in (24, 48, 96, 192, 384, 768):
        out.append(_cfg("J", f"J_{J}", N=16, J=J, K=10, A=20, D_max=12, D_min=3,
                        regime=bc.FULL_SUPPORT, timeout_s=TIMEOUT_STANDARD,
                        min_reps=15 if J <= 384 else 5))
    out.append(_cfg("J", "J_1024", N=16, J=1024, K=10, A=20, D_max=12, D_min=3,
                    regime=bc.FULL_SUPPORT, timeout_s=TIMEOUT_STANDARD, min_reps=5,
                    note="conditional on J=768"))
    return out


def k_configs() -> list:
    """Section 10. N = 32, J = 128, A = 20, D in [3, 12]."""
    out = [_cfg("K", f"K_{K}", N=32, J=128, K=K, A=20, D_max=12, D_min=3,
                regime=bc.FULL_SUPPORT, timeout_s=TIMEOUT_STANDARD,
                min_reps=15 if K <= 20 else 5)
           for K in (3, 5, 10, 20, 40)]
    out.append(_cfg("K", "K_80", N=32, J=128, K=80, A=20, D_max=12, D_min=3,
                    regime=bc.FULL_SUPPORT, timeout_s=TIMEOUT_STANDARD, min_reps=5,
                    note="conditional on K=40"))
    return out


def n_configs() -> list:
    """Section 11. J = 128, K = 10, A = 20, D in [3, 12]."""
    return [_cfg("N", f"N_{N}", N=N, J=128, K=10, A=20, D_max=12, D_min=3,
                 regime=bc.FULL_SUPPORT, timeout_s=TIMEOUT_STANDARD,
                 min_reps=15 if N <= 128 else 5)
            for N in (1, 8, 16, 32, 64, 128, 256)]


def d_configs() -> list:
    """Section 12. N = 16, J = 192, K = 10, A = 20, D_min = 3."""
    return [_cfg("D", f"D_{D}", N=16, J=192, K=10, A=20, D_max=D, D_min=3,
                 regime=bc.FULL_SUPPORT, timeout_s=TIMEOUT_STANDARD,
                 min_reps=15 if D <= 48 else 5)
            for D in (6, 12, 24, 48, 96)]


def a_configs() -> list:
    """Section 13. N = 16, J = 128, K = 10, D in [3, 12], both support regimes."""
    out = []
    for regime, axis in ((bc.FULL_SUPPORT, "A_full"), (bc.SPARSE_SUPPORT, "A_sparse")):
        for A in (5, 10, 20, 30, 50):
            out.append(_cfg(axis, f"{axis}_{A}", N=16, J=128, K=10, A=A, D_max=12,
                            D_min=3, regime=regime, timeout_s=TIMEOUT_STANDARD,
                            min_reps=15 if A <= 30 else 5))
    return out


def target_configs() -> list:
    """Section 16. The anticipated real-data regime, sparse support as primary."""
    out = [_cfg("target", "target_operating_point", N=100, J=200, K=20, A=50, D_max=12,
                D_min=3, regime=bc.SPARSE_SUPPORT, timeout_s=TIMEOUT_TARGET,
                warmups=50, min_reps=5, max_reps=200,
                op_reps=(("cond_plain", 50, 200, 200),
                         ("marg_plain", 50, 200, 200),
                         ("cond_structural", 2, 5, 50),
                         ("marg_structural", 2, 5, 50),
                         ("emission_build", 2, 5, 15),
                         ("emission_cache_hit", 3, 15, 50),
                         ("forward_batched", 3, 15, 50),
                         ("backward_sample", 3, 15, 50),
                         ("ffbs_complete", 3, 15, 50)),
                note="Section 16 primary target: 50 warm-up and 200 timed plain sweeps "
                     "per arm, structural sweeps capped at 50 and allowed to stop at 5 "
                     "under the 30-minute ceiling")]
    out.append(_cfg("target", "target_operating_point_full_support", N=100, J=200, K=20,
                    A=50, D_max=12, D_min=3, regime=bc.FULL_SUPPORT,
                    timeout_s=TIMEOUT_TARGET, warmups=10, min_reps=5, max_reps=50,
                    note="the same operating point under the full-support stress "
                         "regime, reported beside the sparse primary and never averaged "
                         "with it"))
    out.append(_cfg("target_long", "target_long_J500", N=100, J=500, K=20, A=50,
                    D_max=12, D_min=3, regime=bc.SPARSE_SUPPORT,
                    groups=PRIMITIVE_GROUPS, timeout_s=TIMEOUT_TARGET,
                    warmups=2, min_reps=5, max_reps=15,
                    note="primitives only; expected to fail the memory gate at N=100 "
                         "and to be recorded as skipped rather than attempted"))
    out.append(_cfg("target_long", "target_long_J500_N20", N=20, J=500, K=20, A=50,
                    D_max=12, D_min=3, regime=bc.SPARSE_SUPPORT,
                    groups=PRIMITIVE_GROUPS, timeout_s=TIMEOUT_TARGET,
                    warmups=2, min_reps=5, max_reps=15,
                    note="REDUCED CORPUS. The J=500 long-trace primitive point at the "
                         "largest N the dense score table fits in the memory gate. It "
                         "measures trace length at the target K and A; it is not the "
                         "N=100 operating point and is never reported as one."))
    return out


# Points that run only when their predecessor completed safely (Sections 9 and 10).
CONDITIONAL = {
    "J_1024": {"requires": "J_768",
               "max_rss_fraction_of_physical": 0.40,
               "why": "Section 9 admits J=1024 only if J=768 stayed below 40% of "
                      "physical RAM, finished inside its timeout, and did not swap"},
    "K_80": {"requires": "K_40",
             "max_rss_fraction_of_physical": 0.40,
             "why": "Section 10 admits K=80 only if K=40 completed safely below 40% of "
                    "physical RAM with every primary operation under five minutes"},
}


def full_plan() -> list:
    return (baseline_configs() + j_configs() + k_configs() + n_configs()
            + d_configs() + a_configs() + target_configs())


AXIS_ORDER = ("baseline", "J", "K", "N", "D", "A_full", "A_sparse", "target",
              "target_long")


def tasks_for(configs) -> list:
    """(config, group) work items, in the order the driver will attempt them.

    **Ascending within an axis.** Section 15's rule -- once a point on a monotone axis is
    refused, skip every larger point on it unattempted -- is only meaningful in that
    order, and it is also the only thing that stops one runaway point from eating the
    remaining budget.

    **Round robin across axes.** Taking the smallest unrun point of every axis before any
    axis's second point means a budget that runs out leaves every axis with coverage at
    the small end, rather than three axes complete and four untouched. It also places the
    target operating point -- the study's headline measurement, and the one with the
    largest per-configuration ceiling -- in the first round instead of behind every
    expensive tail on every other axis.

    Group order is rotated per configuration, so no single operation always runs first on
    a freshly warmed machine.

    This ordering is deliberately NOT part of `plan_digest`, which covers the
    configuration set alone. Re-ordering the queue therefore does not invalidate a state
    file: a resume keeps every completed measurement and simply picks the next pending
    task in the new order.
    """
    by_axis: dict = {}
    for index, cfg in enumerate(configs):
        by_axis.setdefault(cfg.axis, []).append((index, cfg))

    ordered = []
    depth = 0
    while any(depth < len(rows) for rows in by_axis.values()):
        for axis in AXIS_ORDER:
            rows = by_axis.get(axis, [])
            if depth < len(rows):
                ordered.append(rows[depth])
        depth += 1

    out = []
    for index, cfg in ordered:
        groups = list(cfg.groups)
        rotation = index % len(groups)
        groups = groups[rotation:] + groups[:rotation]
        for group in groups:
            out.append({"config": cfg, "group": group,
                        "task_id": f"{cfg.label}::{group}"})
    return out


def plan_digest(configs=None) -> str:
    configs = full_plan() if configs is None else configs
    payload = json.dumps([c.as_dict() for c in configs], sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    configs = full_plan()
    tasks = tasks_for(configs)
    print(f"{len(configs)} configurations, {len(tasks)} (configuration, group) tasks")
    print(f"plan digest {plan_digest(configs)}")
    for cfg in configs:
        print(f"  {cfg.axis:<12} {cfg.label:<38} N={cfg.N:<4} J={cfg.J:<5} K={cfg.K:<3} "
              f"A={cfg.A:<3} D={cfg.D_min}-{cfg.D_max:<3} {cfg.regime:<6} "
              f"groups={len(cfg.groups)}")


# ============================================================================
# Optional phase (Section 17): only ever run after every registered point has
# settled, and only with the remaining budget. It adds no new axis and answers
# no new question. It exists to do four things the study specification names:
#
#   1. repeat the target operating point under a second deterministic data seed;
#   2. re-attempt the points the first pass censored, now that a worker killed
#      mid-repetition preserves what it had already measured;
#   3. take a quieter baseline, the machine having gone idle since the first pass;
#   4. tighten intervals on the largest point of each axis.
#
# It is a SEPARATE plan with its own digest and its own state file, so it cannot
# disturb the completed main run. Nothing here may be reported as a new finding;
# it is corroboration, and the reports label it as such.
# ============================================================================

SECOND_SEED = bc.BENCH_SEED + 1


def optional_configs() -> list:
    """Corroboration points, in descending order of what the specification asks for."""
    out = []

    # 1. the target operating point again, second seed, quiet machine
    out.append(_cfg("optional_target_seed2", "optional_target_seed2", N=100, J=200,
                    K=20, A=50, D_max=12, D_min=3, regime=bc.SPARSE_SUPPORT,
                    seed=SECOND_SEED, timeout_s=TIMEOUT_TARGET,
                    warmups=50, min_reps=5, max_reps=200,
                    op_reps=(("cond_plain", 50, 200, 200),
                             ("marg_plain", 50, 200, 200),
                             ("cond_structural", 2, 5, 20),
                             ("marg_structural", 2, 5, 20),
                             ("emission_build", 2, 5, 15),
                             ("emission_cache_hit", 3, 15, 50),
                             ("forward_batched", 3, 15, 50),
                             ("backward_sample", 3, 15, 50),
                             ("ffbs_complete", 3, 15, 50)),
                    note="Section 17.1: the target operating point under a second "
                         "deterministic data seed, on a machine that has since gone "
                         "idle. Corroboration of the primary, not a new measurement."))

    # 2. the three points the first pass censored, with repetition counts set from
    #    what that pass learned about their cost
    out.append(_cfg("optional_retry", "optional_target_full_support_retry", N=100,
                    J=200, K=20, A=50, D_max=12, D_min=3, regime=bc.FULL_SUPPORT,
                    timeout_s=TIMEOUT_TARGET, groups=("build", "cond", "marg"),
                    warmups=1, min_reps=2, max_reps=5,
                    op_reps=(("emission_build", 1, 2, 4),
                             ("emission_cache_hit", 2, 10, 30),
                             ("cond_plain", 3, 10, 40),
                             ("marg_plain", 3, 10, 40),
                             ("cond_structural", 1, 2, 4),
                             ("marg_structural", 1, 2, 4)),
                    note="RETRY of the three groups the first pass lost to the "
                         "30-minute ceiling. Repetition counts are cut to what that "
                         "pass showed is affordable, and every point here is a "
                         "REDUCED-REPETITION measurement."))

    # 3. a quieter baseline
    mix = tuple(J for J in (24, 32, 40, 48) for _ in range(25))
    out.append(_cfg("optional_baseline", "optional_baseline_quiet", N=100, J=48, K=3,
                    A=5, D_max=12, D_min=3, regime=bc.FULL_SUPPORT, length_mix=mix,
                    timeout_s=TIMEOUT_STANDARD, min_reps=25, max_reps=60,
                    note="Section 17.3: the matched scale again on an idle machine, "
                         "for a load-controlled comparison against the first pass."))

    # 4. the largest point of each axis, second seed, for interval width and for a
    #    check that the exponents are not an artifact of one corpus draw
    for label, kwargs in (
            ("optional_J_384_seed2", dict(N=16, J=384, K=10, A=20, D_max=12)),
            ("optional_K_40_seed2", dict(N=32, J=128, K=40, A=20, D_max=12)),
            ("optional_N_128_seed2", dict(N=128, J=128, K=10, A=20, D_max=12)),
            ("optional_D_48_seed2", dict(N=16, J=192, K=10, A=20, D_max=48)),
            ("optional_A_full_50_seed2", dict(N=16, J=128, K=10, A=50, D_max=12)),
    ):
        out.append(_cfg("optional_repeat", label, D_min=3, regime=bc.FULL_SUPPORT,
                        seed=SECOND_SEED, timeout_s=TIMEOUT_STANDARD,
                        min_reps=10, max_reps=30,
                        note="Section 17.2 and 17.4: a large point repeated under a "
                             "second seed on a quiet machine.", **kwargs))
    out.append(_cfg("optional_repeat", "optional_A_sparse_50_seed2", N=16, J=128, K=10,
                    A=50, D_max=12, D_min=3, regime=bc.SPARSE_SUPPORT, seed=SECOND_SEED,
                    timeout_s=TIMEOUT_STANDARD, min_reps=10, max_reps=30,
                    note="Section 17.2 and 17.4: a large point repeated under a second "
                         "seed on a quiet machine."))
    return out


OPTIONAL_AXIS_ORDER = ("optional_target_seed2", "optional_retry", "optional_baseline",
                       "optional_repeat")


def optional_tasks() -> list:
    """Optional-phase work items, in the specification's own priority order."""
    out = []
    configs = optional_configs()
    for index, cfg in enumerate(sorted(configs,
                                       key=lambda c: OPTIONAL_AXIS_ORDER.index(c.axis))):
        groups = list(cfg.groups)
        rotation = index % len(groups)
        groups = groups[rotation:] + groups[:rotation]
        for group in groups:
            out.append({"config": cfg, "group": group,
                        "task_id": f"{cfg.label}::{group}"})
    return out
