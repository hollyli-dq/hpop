"""One benchmarked operation group, in its own process.

    python scripts/scalability/bench_worker.py --config <json> --group <name> --out <json>

A fresh process per group is what makes `ru_maxrss` mean something: the peak resident set
reported here is the peak of *this* configuration and this group, with nothing else in the
address space. It also means a configuration that allocates too much cannot poison the
next one.

## Groups

    build        candidate block / emission table construction, and the cache-hit path
    primitives   optimized forward, frozen backward draw, and the complete FFBS update
    cond         FULL-COND plain and structural sweeps
    marg         FULL-MARG plain and structural sweeps

## Protocol

Warm-ups are untimed. Timed repetitions are interleaved round-robin across the operations
in the group, so a drift in ambient machine load lands on every operation equally instead
of on whichever one happened to run last. Repetitions continue past the minimum, in
batches, until every operation's bootstrap 95% interval for the median has relative
half-width at or below five per cent, up to the configured maximum or the deadline.

Every repetition is recorded. Nothing is averaged before it reaches the result file.

Each repetition starts from an identical state copy under an identically seeded rng, so
repetitions measure the same arithmetic rather than a chain wandering into cheaper or more
expensive regions.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                    # noqa: E402
import bench_common as bc                                             # noqa: E402

CI_TARGET = 0.05
REP_BATCH = 5


# ------------------------------------------------------------------------ measurement
class Timer:
    """Wall clock and process CPU time for one repetition, taken around the call only."""

    def __init__(self):
        self.wall = 0.0
        self.cpu = 0.0

    def __enter__(self):
        self._w = time.perf_counter()
        self._c = time.process_time()
        return self

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self._w
        self.cpu = time.process_time() - self._c
        return False


def _finite(*arrays) -> bool:
    for array in arrays:
        a = np.asarray(array, dtype=float)
        if np.isnan(a).any():
            return False
    return True


# ---------------------------------------------------------------------------- groups
def build_group(cfg, model, state):
    """Emission/block-score table construction, and the H-cache short circuit."""
    from hpop.mcmc_optimized import COUNTERS, FLAGS, HashCachedFFBSBlockTables

    alloc = Timer()
    with alloc:
        tables = HashCachedFFBSBlockTables(model=model, source="batched")
    tables.refresh(state)                                    # one build before timing

    def invalidate():
        # Force the H-keyed short circuit to miss, so the timed call measures a real
        # rebuild. `_ever_built` is the optimized backend's own cache flag, set from
        # outside; no source file is patched. Untimed, and it is two attribute writes.
        tables._ever_built = False
        tables._structural_key = None

    def rebuild():
        tables.refresh(state)

    def cache_hit():
        tables.refresh(state)

    checks = {}

    def invariants():
        built = list(tables.tables_for(state))
        finite_entries = sum(int(np.isfinite(t).sum()) for t in built)
        checks.update({
            "no_nan_in_block_tables": all(not np.isnan(t).any() for t in built),
            "finite_block_entries": finite_entries,
            "expected_finite_block_entries": int(
                bc.legal_block_count(cfg, model)["legal_blocks_times_skills"]),
            "emission_rebuilds": int(COUNTERS.emission_rebuilds),
            "emission_cache_hits": int(COUNTERS.emission_cache_hits),
            "table_source": "batched",
            "flags": FLAGS.snapshot(),
        })
        checks["legal_block_count_matches_geometry"] = (
            checks["finite_block_entries"] == checks["expected_finite_block_entries"])

    return ({"emission_build": rebuild, "emission_cache_hit": cache_hit},
            {"emission_build": invalidate},
            {"emission_table_alloc_seconds": alloc.wall}, invariants, checks)


def primitives_group(cfg, model, state):
    """Optimized forward, the frozen backward draw, and the complete FFBS update."""
    from hpop.mcmc_original.semi_markov_ffbs import backward_sample
    from hpop.mcmc_original.transitions import log_transition_matrix
    from hpop.mcmc_optimized import (COUNTERS, HashCachedFFBSBlockTables,
                                     forward_batched_group)
    from hpop.mcmc_optimized.segmentation import ffbs_segmentation_draw

    tables = HashCachedFFBSBlockTables(model=model, source="batched")
    tables.refresh(state)
    all_tables = list(tables.tables_for(state))
    log_pi = np.log(state.pi)
    log_p = log_transition_matrix(state.transition)

    classes: dict = {}
    for n, table in enumerate(all_tables):
        classes.setdefault(np.asarray(table).shape[0], []).append(n)

    holder = {}

    def forward():
        charts = [None] * len(all_tables)
        for _length, members in sorted(classes.items()):
            for n, chart in zip(members, forward_batched_group(
                    [all_tables[n] for n in members], log_pi, log_p, model.delta_b,
                    model.max_width, model.min_width)):
                charts[n] = chart
        holder["charts"] = charts

    forward()
    charts = holder["charts"]

    def backward():
        rng = np.random.default_rng(20260822)
        holder["blocks"] = [backward_sample(chart, rng) for chart in charts]

    def complete_ffbs():
        rng = np.random.default_rng(20260822)
        holder["ffbs"] = ffbs_segmentation_draw(model, state, tables, rng)

    checks = {}

    def invariants():
        forward()
        backward()
        complete_ffbs()
        alphas = [c.alpha for c in holder["charts"]]
        normalizers = np.array([c.log_normalizer for c in holder["charts"]])
        pattern = all(np.isfinite(a).any() for a in alphas)
        widths_ok, cover_ok, repeats_ok = True, True, True
        for trace, blocks in zip(model.traces, holder["blocks"]):
            cursor, previous = 0, None
            for a, b, k in blocks:
                if a != cursor or not model.min_width <= b - a <= model.max_width:
                    widths_ok = False
                if previous is not None and int(k) == previous:
                    repeats_ok = False
                cursor, previous = b, int(k)
            cover_ok = cover_ok and cursor == len(trace)
        checks.update({
            "no_nan_in_alpha": all(not np.isnan(a).any() for a in alphas),
            "some_finite_alpha_every_trace": bool(pattern),
            "log_normalizers_all_finite": bool(np.isfinite(normalizers).all()),
            "log_normalizer_total": float(normalizers.sum()),
            "backward_draw_widths_legal": bool(widths_ok),
            "backward_draw_covers_trace": bool(cover_ok),
            "backward_draw_no_self_transition": bool(repeats_ok),
            "forward_batched_groups": int(COUNTERS.forward_batched_groups),
            "forward_batched_traces": int(COUNTERS.forward_batched_traces),
            "n_length_classes": len(classes),
        })

    return ({"forward_batched": forward, "backward_sample": backward,
             "ffbs_complete": complete_ffbs}, {}, {}, invariants, checks)


def sweep_group(cfg, model, state, arm: str, prefix: str):
    """Plain and structural sweeps for one arm, each from an identical state copy."""
    from hpop.mcmc_original.matched_full_latent import validate_paths, validate_pi_p
    from hpop.mcmc_optimized import sweep_once

    # One sampler per operation. The samplers carry caches -- the H-keyed emission cache
    # and the collapsed likelihood's current-state cache -- and interleaving two
    # operations through a single sampler would have each one invalidating the other's,
    # measuring cache thrash rather than the steady state a chain actually runs in.
    plain_sampler = bc.build_sampler(cfg, model, arm)
    structural_sampler = bc.build_sampler(cfg, model, arm)

    plain_base = state.copy()
    plain_base.iteration = 0                    # (0 + 1) % 10 != 0  -> no structural move
    structural_base = state.copy()
    structural_base.iteration = 9               # (9 + 1) % 10 == 0  -> structural move

    holder = {}

    def plain():
        rng = np.random.default_rng(20260822)
        holder["plain"] = sweep_once(plain_base.copy(), plain_sampler, rng)

    def structural():
        rng = np.random.default_rng(20260823)
        holder["structural"] = sweep_once(structural_base.copy(), structural_sampler, rng)

    def before_structural():
        """Put the structural sampler in the state a chain would hand it, untimed.

        Two separate things are arranged here, for two different reasons.

        The emission cache is forced to miss. A structural sweep only pays a table
        rebuild when the proposal moves `H = h(U)`; when it does not, the H-keyed cache
        legitimately returns the same bits. Repeating one proposal from one base state
        would let the cache hit from the second repetition onward and report a structural
        sweep that never rebuilds, so the rebuild is forced every time and the measured
        number is the H-moved case -- the upper bound, and the one a cost model should
        use. `emission_build` is measured separately, so the H-unchanged structural sweep
        is recoverable by subtraction.

        The collapsed likelihood's baseline is primed at the base state. In a chain the
        incoming `ell_coll(U)` is already cached from the previous sweep; restarting every
        repetition from an identical state would otherwise charge FULL-MARG an extra
        all-trace forward pass that a real chain does not pay.
        """
        structural_sampler.tables._ever_built = False
        structural_sampler.tables._structural_key = None
        if arm == "FULL-MARG":
            structural_sampler.collapsed_likelihood.log_z_per_trace(structural_base)

    checks = {}

    def invariants():
        plain()
        structural()
        ok = True
        samplers = {"plain": plain_sampler, "structural": structural_sampler}
        for name in ("plain", "structural"):
            new_state, record = holder[name]
            try:
                validate_pi_p(new_state, model)
                validate_paths(new_state, model)
                samplers[name].fixed.assert_unchanged(new_state)
            except AssertionError as error:
                ok = False
                checks[f"{name}_invariant_error"] = str(error)
            components = new_state.components
            finite = all(np.isfinite(float(v)) for v in components.values()
                         if isinstance(v, (int, float)))
            checks[f"{name}_kernel_order"] = list(record["kernel_order"])
            checks[f"{name}_scheduled_structural"] = bool(record["scheduled_structural"])
            checks[f"{name}_components_finite"] = bool(finite)
            checks[f"{name}_log_target"] = float(components.get("log_target", float("nan")))
            p = np.asarray(new_state.transition, dtype=float)
            checks[f"{name}_P_diagonal_exactly_zero"] = bool(
                np.array_equal(np.diag(p), np.zeros(p.shape[0])))
            checks[f"{name}_P_rows_sum_to_one"] = bool(np.allclose(
                p.sum(axis=1), 1.0, atol=1e-12))
        record = holder["structural"][1]["structural_record"]
        checks["structural_move_attempted"] = record is not None
        if record is not None:
            checks["structural_move_accepted"] = bool(record.get("accepted"))
            checks["structural_move_invalid"] = bool(record.get("invalid"))
        checks["all_invariants_pass"] = bool(ok)

    return ({f"{prefix}_plain": plain, f"{prefix}_structural": structural},
            {f"{prefix}_structural": before_structural}, {}, invariants, checks)


GROUPS = {
    "build": lambda cfg, m, s: build_group(cfg, m, s),
    "primitives": lambda cfg, m, s: primitives_group(cfg, m, s),
    "cond": lambda cfg, m, s: sweep_group(cfg, m, s, "FULL-COND", "cond"),
    "marg": lambda cfg, m, s: sweep_group(cfg, m, s, "FULL-MARG", "marg"),
}


# ------------------------------------------------------------------------------ driver
def run(cfg: bc.BenchConfig, group: str, deadline: float,
        out_path: Path | None = None) -> dict:
    from hpop.mcmc_optimized import COUNTERS, FLAGS

    FLAGS.reset()                       # all four optimisations on: optimized_segmental_v1
    COUNTERS.reset()

    started_wall = time.time()
    rss_at_start = bc.peak_rss_bytes()
    load_before = bc.load_average()
    bc.speed_probe()                     # discard the first, it warms the allocator
    probes = [bc.speed_probe()]

    setup = Timer()
    with setup:
        model = bc.build_model(cfg)
        state = bc.build_state(cfg, model)

    geometry = bc.legal_block_count(cfg, model)
    work = bc.forward_work_counts(cfg, model)
    memory = bc.predict_memory(cfg, model, group)
    graph = bc.role_graph_summary(state.u_by_skill)
    preflight = bc.memory_preflight(memory["predicted_process_rss_bytes"])

    if not preflight["allowed"]:
        return {
            "status": "skipped_memory_preflight", "config": cfg.as_dict(), "group": group,
            "preflight": preflight, "predicted_memory": memory, "geometry": geometry,
            "forward_work": work, "role_graph": graph,
            "loadavg_before": load_before, "loadavg_after": bc.load_average(),
            "swap": bc.swapping_now(),
        }

    build = Timer()
    with build:
        operations, pre_hooks, extras, invariants, checks = GROUPS[group](cfg, model,
                                                                          state)

    invariants()
    gc.collect()

    names = list(operations)
    samples = {name: {"wall": [], "cpu": []} for name in names}

    # The result is built by a closure so it can be written after every repetition. A
    # configuration killed by the driver's hard timeout then still leaves the timings it
    # had already taken on disk, instead of nothing at all.
    state_box = {"censored": False, "budget_limited": set(), "warmups_done": {},
                 "warmup_cost": {}, "share": {}, "spent": {},
                 "warmups_truncated": [], "plan": {}}

    def _result(status: str) -> dict:
        done = min((len(samples[n]["wall"]) for n in names), default=0)
        return {
            "status": status,
            "censored": bool(state_box["censored"]),
            "config": cfg.as_dict(),
            "config_id": bc.config_id(cfg),
            "group": group,
            "reps_completed": int(done),
            "reps_completed_per_operation": {n: len(samples[n]["wall"]) for n in names},
            "reduced_repetitions": bool(done < 15),
            "repetition_plan": {n: {"warmups": state_box["plan"].get(n, (0, 0, 0))[0],
                                    "min_reps": state_box["plan"].get(n, (0, 0, 0))[1],
                                    "max_reps": state_box["plan"].get(n, (0, 0, 0))[2]}
                                for n in names},
            "warmups_completed": dict(state_box["warmups_done"]),
            "warmups_truncated_by_budget": sorted(set(state_box["warmups_truncated"])),
            "budget_limited_operations": sorted(state_box["budget_limited"]),
            "warmup_mean_seconds": dict(state_box["warmup_cost"]),
            "time_budget_share_seconds": dict(state_box["share"]),
            "time_spent_per_operation_seconds": dict(state_box["spent"]),
            "warmups": int(cfg.warmups),
            "setup_seconds": setup.wall,
            "group_build_seconds": build.wall,
            "extras": extras,
            "operations": {
                name: {
                    "wall_raw": list(samples[name]["wall"]),
                    "cpu_raw": list(samples[name]["cpu"]),
                    "wall": bc.summarize(samples[name]["wall"]),
                    "cpu": bc.summarize(samples[name]["cpu"]),
                } for name in names},
            "invariants": checks,
            "geometry": geometry,
            "forward_work": work,
            "predicted_memory": memory,
            "role_graph": graph,
            "preflight": preflight,
            "peak_rss_bytes": bc.peak_rss_bytes(),
            "peak_rss_at_process_start_bytes": rss_at_start,
            "process_cpu_seconds_total": bc.process_cpu_seconds(),
            "loadavg_before": load_before,
            "loadavg_after": bc.load_average(),
            "speed_probe_seconds": list(probes),
            "speed_probe_median_seconds": (float(np.median(probes)) if probes
                                           else float("nan")),
            "swap": bc.swapping_now(),
            "wall_seconds_total": time.time() - started_wall,
            "counters": COUNTERS.snapshot(),
            "flags": FLAGS.snapshot(),
        }

    def flush(status: str) -> None:
        bc.atomic_write(Path(out_path),
                        json.dumps(_result(status), indent=2, sort_keys=True,
                                   default=float))

    # Per-operation repetition targets. The plan registers specific counts for the target
    # operating point (Section 16), where a plain sweep and a structural sweep differ in
    # cost by more than an order of magnitude and one blanket count would either starve
    # the cheap operation or overrun the budget on the expensive one.
    overrides = {row[0]: (int(row[1]), int(row[2]), int(row[3])) for row in cfg.op_reps}
    plan = {name: overrides.get(name, (int(cfg.warmups), int(cfg.min_reps),
                                       int(cfg.max_reps))) for name in names}
    state_box["plan"] = plan

    def one(name):
        """One repetition of one operation: untimed preparation, then the timed call."""
        hook = pre_hooks.get(name)
        if hook is not None:
            hook()
        timer = Timer()
        with timer:
            operations[name]()
        return timer

    # Warm-ups are untimed as measurements but their duration is recorded, because it is
    # the only cost estimate available before the budget has to be divided.
    #
    # Two guards, both learned from a configuration that was killed mid-repetition and
    # lost everything it had measured. Warm-ups may not eat more than a quarter of the
    # budget -- ten warm-up rebuilds of a five-minute table is fifty minutes against a
    # thirty-minute ceiling -- and no repetition is started that the remaining time
    # cannot finish, because a repetition killed in flight yields nothing at all.
    started_loop = time.time()
    warmup_ceiling = started_loop + 0.25 * max(0.0, deadline - started_loop)
    warmups_done = state_box["warmups_done"] = {name: 0 for name in names}
    warmup_cost = state_box["warmup_cost"] = {name: 0.0 for name in names}
    warmups_truncated = state_box["warmups_truncated"]
    for name in names:
        elapsed = 0.0
        for index in range(max(1, plan[name][0])):
            now = time.time()
            estimate = elapsed / index if index else 0.0
            if now + 1.15 * estimate > deadline or (index and now > warmup_ceiling):
                warmups_truncated.append(name)
                break
            elapsed += one(name).wall
            warmups_done[name] += 1
        warmup_cost[name] = elapsed / max(1, warmups_done[name])
    probes.append(bc.speed_probe())

    # Divide the remaining budget between the operations in proportion to what each one
    # would need to reach its registered repetition count. Without this the round-robin
    # starves the cheap operation: at the target operating point one structural sweep
    # costs about as much as fifteen plain sweeps, so a plain-for-structural round robin
    # would deliver a few dozen plain sweeps where the plan registers two hundred. Each
    # operation drops out of the rounds when it reaches its maximum, satisfies the
    # interval target, or exhausts its own share -- so interleaving survives for the
    # configurations whose operations cost the same, which is most of them.
    remaining = max(0.0, deadline - time.time())
    desired = {name: plan[name][2] * max(warmup_cost[name], 1e-9) for name in names}
    total_desired = sum(desired.values()) or 1.0
    share = state_box["share"] = {
        name: remaining * desired[name] / total_desired for name in names}
    spent = state_box["spent"] = {name: 0.0 for name in names}

    finished, censored = set(), False
    budget_limited = state_box["budget_limited"]
    while len(finished) < len(names):
        if time.time() > deadline:
            censored = state_box["censored"] = True
            break
        for name in names:                       # interleaved, one rep each per round
            if name in finished:
                continue
            observed = samples[name]["wall"]
            estimate = (float(np.median(observed)) if observed
                        else warmup_cost[name])
            if time.time() + 1.15 * estimate > deadline:
                # starting this repetition would run past the driver's hard kill, and a
                # repetition killed in flight is not a slow measurement, it is no
                # measurement -- so it is refused and the point is marked censored
                finished.add(name)
                censored = state_box["censored"] = True
                continue
            timer = one(name)
            samples[name]["wall"].append(timer.wall)
            samples[name]["cpu"].append(timer.cpu)
            spent[name] += timer.wall
            if out_path is not None:
                flush("partial")
            _warmup, minimum, maximum = plan[name]
            count = len(samples[name]["wall"])
            if count >= maximum:
                finished.add(name)
            elif count >= minimum and bc.bootstrap_median_ci(
                    samples[name]["wall"])["relative_half_width"] <= CI_TARGET:
                finished.add(name)
            elif count >= minimum and spent[name] >= share[name]:
                # Stopped by its budget share, past its registered minimum. That is a
                # complete measurement with a wider interval than the plan hoped for --
                # recorded as budget-limited, and deliberately NOT as censored, because
                # censoring means the deadline cut a point short of usable.
                finished.add(name)
                budget_limited.add(name)
    reps_done = min((len(samples[n]["wall"]) for n in names), default=0)
    probes.append(bc.speed_probe())

    gc.collect()
    result = _result("censored_deadline" if reps_done == 0 else "ok")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--group", required=True, choices=sorted(GROUPS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--deadline-s", type=float, default=720.0)
    args = parser.parse_args()

    payload = json.loads(Path(args.config).read_text())
    payload["length_mix"] = tuple(payload.get("length_mix", ()))
    payload["groups"] = tuple(payload.get("groups", ()))
    cfg = bc.BenchConfig(**payload)

    result = run(cfg, args.group, time.time() + float(args.deadline_s),
                 out_path=Path(args.out))
    bc.atomic_write(Path(args.out), json.dumps(result, indent=2, sort_keys=True,
                                               default=float))
    print(json.dumps({"status": result["status"],
                      "reps": result.get("reps_completed", 0),
                      "peak_rss_mb": round(result.get("peak_rss_bytes", 0) / 1024 ** 2, 1)}))


if __name__ == "__main__":
    main()
