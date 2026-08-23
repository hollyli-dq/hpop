"""Shared pieces for the segmental-inference scalability benchmark.

Nothing in this module edits `hpop.mcmc_original` or `hpop.mcmc_optimized`. It only
*calls* them, with model dimensions the registered constants do not fix. `Stage6EModel`
already takes `n_skills`, `n_roles`, `min_width` and `max_width` as ordinary fields, so a
benchmark configuration is a different argument, never a different model.

## What a configuration means

    N   number of traces in the corpus
    J   trace length (equal for every trace unless a length mix is given)
    K   number of reusable skills
    A   size of the canonical-action / role vocabulary   (`Stage6EModel.n_roles`)
    D   maximum legal segment width                      (`max_width`)

## The two role-support regimes, and why they are reported apart

`full`   every skill's `U_k` is an independent draw over all `A` roles, so the induced
         precedence relation is dense: under two latent columns an unordered pair is
         comparable with probability about one half, giving roughly `A^2 / 4` relations
         and predecessor lists of length about `A / 4`.

`sparse` a skill's support is `min(10, A)` roles drawn deterministically from the seed.
         Roles outside the support share one identical latent row, so they are mutually
         incomparable by construction (`>` is false in both directions on a tie) and the
         relation count collapses. Traces are emitted from the supports, so the corpus is
         what a real CPA vocabulary of size `A` with narrow per-skill support looks like.

The two are never averaged: they are different role graphs, and the emission recursion's
cost is a function of the graph, not of `A` alone. Both regimes' measured relation counts
and predecessor-list sizes are recorded with every configuration.

## Determinism

Every array a configuration needs is a pure function of `(seed, N, J, K, A, D, regime)`.
Re-running a configuration reproduces the same corpus, the same `U`, the same `pi`/`P` and
the same sampler rng stream, which is what makes a resumed run comparable with the part
that ran before the interruption.
"""

from __future__ import annotations

import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Threading is pinned before NumPy is imported anywhere in this process.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results" / "scalability" / "optimized_segmental_v1"

# The one seed the whole study is a function of.
BENCH_SEED = 20260822

LATENT_DIM = 2
FULL_SUPPORT = "full"
SPARSE_SUPPORT = "sparse"
SPARSE_SUPPORT_SIZE = 10

# The registered FULL-LATENT fixed coordinates. Imported rather than copied where the
# import is cheap; these mirror `full_latent_constants` and are asserted against it.
EPSILON = 0.02
DELTA_B = 0.15
MIN_WIDTH_DEFAULT = 3


# --------------------------------------------------------------------- configurations
@dataclass(frozen=True)
class BenchConfig:
    """One benchmarked operating point. Hashable, JSON-serialisable, self-describing."""

    axis: str
    label: str
    N: int
    J: int
    K: int
    A: int
    D_max: int
    D_min: int = MIN_WIDTH_DEFAULT
    regime: str = FULL_SUPPORT
    seed: int = BENCH_SEED
    length_mix: tuple = ()          # empty => N traces all of length J
    groups: tuple = ("build", "primitives", "cond", "marg")
    min_reps: int = 15
    max_reps: int = 50
    warmups: int = 3
    timeout_s: int = 720
    note: str = ""
    # ("operation", warmups, min_reps, max_reps) overrides, for the target operating
    # point where the plan registers a specific repetition count per operation
    op_reps: tuple = ()

    def key(self) -> str:
        return self.label

    def as_dict(self) -> dict:
        out = asdict(self)
        out["length_mix"] = list(self.length_mix)
        out["groups"] = list(self.groups)
        out["op_reps"] = [list(row) for row in self.op_reps]
        return out


def config_id(cfg: BenchConfig) -> str:
    return (f"{cfg.axis}/N{cfg.N}_J{cfg.J}_K{cfg.K}_A{cfg.A}_D{cfg.D_max}"
            f"_{cfg.regime}_s{cfg.seed}")


# ------------------------------------------------------------------ corpus synthesis
def _rng(seed: int, *parts: int) -> np.random.Generator:
    """A child stream addressed by integers, so every array has its own reproducible rng."""
    return np.random.default_rng([int(seed), *[int(p) for p in parts]])


def skill_supports(K: int, A: int, seed: int) -> list:
    """Deterministic per-skill role supports of size `min(10, A)` for the sparse regime."""
    size = min(SPARSE_SUPPORT_SIZE, int(A))
    return [np.sort(_rng(seed, 11, k).choice(int(A), size=size, replace=False))
            for k in range(int(K))]


def make_u(K: int, A: int, regime: str, seed: int, scale: float = 1.5) -> np.ndarray:
    """`U` of shape `(K, A, 2)`; the sparse regime ties the out-of-support rows.

    A tie makes `all(u_x > u_z)` false in both directions, so tied roles carry no
    precedence relation at all. That is the whole mechanism by which the sparse regime
    thins the role graph, and it uses only the model's own definition of precedence.
    """
    K, A = int(K), int(A)
    u = np.empty((K, A, LATENT_DIM), dtype=float)
    for k in range(K):
        rows = float(scale) * _rng(seed, 7, k).standard_normal((A, LATENT_DIM))
        if regime == SPARSE_SUPPORT and A > SPARSE_SUPPORT_SIZE:
            support = skill_supports(K, A, seed)[k]
            outside = np.setdiff1d(np.arange(A), support)
            # One shared off-diagonal row, `(+C, -C)`, for every role outside the
            # support. Tied rows are incomparable to each other (`>` is false both
            # ways), and `(+C, -C)` is incomparable to any row inside the box because
            # it wins the first column and loses the second. So the induced order is
            # exactly the order on the `min(10, A)` support roles, which is what "each
            # skill has ten roles" has to mean for a model whose only notion of a role
            # inventory is its precedence relation.
            far = 8.0 * float(scale)
            rows[outside] = np.array([far, -far])
        u[k] = rows
    return u


def make_traces(N: int, J: int, K: int, A: int, regime: str, seed: int,
                length_mix: tuple = ()) -> tuple:
    """Role sequences. Lengths come from `length_mix` when given, else all equal `J`.

    In the sparse regime a trace is a concatenation of skill-blocks whose roles are drawn
    from that skill's support, which is what a corpus over a size-`A` CPA vocabulary with
    narrow per-skill support actually looks like. In the full regime roles are uniform on
    the whole vocabulary. The arithmetic cost of every measured operation depends on array
    *shapes*, which are identical either way; the role values change only which gate
    pattern the recursion walks.
    """
    N, J, K, A = int(N), int(J), int(K), int(A)
    lengths = list(length_mix) if length_mix else [J] * N
    if len(lengths) != N:
        raise ValueError(f"length_mix has {len(lengths)} entries for N={N}")
    supports = skill_supports(K, A, seed) if regime == SPARSE_SUPPORT else None
    traces = []
    for n, length in enumerate(lengths):
        rng = _rng(seed, 3, n)
        if supports is None:
            traces.append(tuple(int(v) for v in rng.integers(0, A, size=length)))
            continue
        out, cursor = [], 0
        while cursor < length:
            k = int(rng.integers(K))
            width = int(rng.integers(4, 13))
            width = min(width, length - cursor)
            out.extend(int(v) for v in rng.choice(supports[k], size=width))
            cursor += width
        traces.append(tuple(out[:length]))
    return tuple(traces)


def role_graph_summary(u: np.ndarray) -> dict:
    """Measured density of the induced precedence relation, per skill and pooled.

    Recorded for every configuration so the `A` axis can be read against the graph the
    emission recursion actually walks rather than against `A` alone.
    """
    from hpop.mcmc_original.latent_poset import precedence_from_u
    u = np.asarray(u, dtype=float)
    K, A = u.shape[0], u.shape[1]
    relations, pred_sizes, max_pred = [], [], []
    for k in range(K):
        precedence = np.asarray(precedence_from_u(u[k]))
        off = precedence.copy()
        np.fill_diagonal(off, False)
        relations.append(int(off.sum()))
        sizes = off.sum(axis=0)
        pred_sizes.append(float(sizes.mean()))
        max_pred.append(int(sizes.max()) if sizes.size else 0)
    pairs = A * (A - 1)
    return {
        "n_roles": int(A), "n_skills": int(K),
        "relations_per_skill": relations,
        "relation_density_per_skill": [r / pairs if pairs else 0.0 for r in relations],
        "mean_relation_density": float(np.mean([r / pairs for r in relations])
                                       if pairs else 0.0),
        "mean_predecessors_per_role": float(np.mean(pred_sizes)),
        "max_predecessors_per_role": int(max(max_pred)) if max_pred else 0,
    }


# ------------------------------------------------------------------- model and state
def build_model(cfg: BenchConfig):
    """A `Stage6EModel` at the configuration's dimensions. Constructed, never patched."""
    from hpop.mcmc_original.stage6e_state import Stage6EModel
    traces = make_traces(cfg.N, cfg.J, cfg.K, cfg.A, cfg.regime, cfg.seed,
                         cfg.length_mix)
    return Stage6EModel(
        traces=traces, epsilon=EPSILON, delta_b=DELTA_B,
        n_skills=int(cfg.K), n_roles=int(cfg.A),
        min_width=int(cfg.D_min), max_width=int(cfg.D_max),
        infer_pi_P=True, eta_initial=1.0, eta_transition=1.0)


def initial_pi_p(K: int, seed: int):
    """A legal `(pi, P)` with the model's required exactly-zero diagonal."""
    from hpop.mcmc_original.transitions import sample_transition_matrix
    K = int(K)
    rng = _rng(seed, 5, K)
    pi = rng.dirichlet(np.ones(K))
    p = sample_transition_matrix(np.zeros((K, K)), K, rng, 1.0)
    return pi, p


def legal_tiling(length: int, K: int, min_width: int, max_width: int, index: int):
    """A deterministic legal cover; FFBS replaces it on the first draw."""
    from hpop.mcmc_original.types import Segment, Segmentation
    remaining, widths = int(length), []
    while remaining > max_width:
        step = max_width if remaining - max_width >= min_width else remaining - min_width
        widths.append(step)
        remaining -= step
    widths.append(remaining)
    if sum(widths) != length or any(not min_width <= w <= max_width for w in widths):
        raise ValueError(f"no legal tiling for J={length}, "
                         f"widths [{min_width}, {max_width}]")
    segments, start = [], 0
    for position, width in enumerate(widths):
        segments.append(Segment(start, start + width, (index + position) % K))
        start += width
    return Segmentation(tuple(segments))


def build_state(cfg: BenchConfig, model):
    """A legal `Stage6EState` at the configuration's dimensions."""
    from hpop.mcmc_original.full_latent_constants import (
        FIXED_BETA, FIXED_LAMBDA_BACK, FIXED_LAMBDA_REP, FIXED_OMEGA, FIXED_RHO_0)
    from hpop.mcmc_original.stage6e_state import Stage6EState
    pi, p = initial_pi_p(cfg.K, cfg.seed)
    u = make_u(cfg.K, cfg.A, cfg.regime, cfg.seed)
    segmentations = tuple(
        legal_tiling(len(t), cfg.K, cfg.D_min, cfg.D_max, n)
        for n, t in enumerate(model.traces))
    return Stage6EState(
        segmentations=segmentations, u_by_skill=u, rho=float(FIXED_RHO_0),
        beta=float(FIXED_BETA), omega=float(FIXED_OMEGA),
        lambda_rep=float(FIXED_LAMBDA_REP), lambda_back=float(FIXED_LAMBDA_BACK),
        pi=pi, transition=p)


def build_sampler(cfg: BenchConfig, model, arm: str):
    """An `OptimizedFullLatentSampler` at the registered `table_source='batched'`."""
    from hpop.mcmc_original.matched_full_latent import FullLatentConfig, FullLatentFixed
    from hpop.mcmc_optimized import OptimizedFullLatentSampler
    fixed = FullLatentFixed()
    config = FullLatentConfig(arm=arm, structural_cadence=10, structural_scale=0.5,
                              table_source="batched")
    return OptimizedFullLatentSampler(model=model, fixed=fixed, config=config)


# ------------------------------------------------------------------- counted geometry
def legal_block_count(cfg: BenchConfig, model) -> dict:
    """Exact candidate-block geometry, counted from the configuration, not estimated."""
    widths = range(int(cfg.D_min), int(cfg.D_max) + 1)
    per_trace = [sum(max(0, len(t) - w + 1) for w in widths) for t in model.traces]
    total = int(sum(per_trace))
    return {
        "legal_blocks_per_trace_mean": float(np.mean(per_trace)) if per_trace else 0.0,
        "legal_blocks_total": total,
        "legal_blocks_times_skills": total * int(cfg.K),
        "n_legal_widths": len(list(widths)),
        "trace_occurrences": int(sum(len(t) for t in model.traces)),
    }


def forward_work_counts(cfg: BenchConfig, model) -> dict:
    """Forward-recursion work, counted exactly from the recursion's own index ranges.

    `states` is the number of `(b, k)` chart cells; `reductions` is the number of
    `(a, b, k)` predecessor terms the factorised recursion reduces over, plus the `K^2`
    transition reduction it performs once per `(b)`. These are the quantities the
    complexity comparison in Section 18 of the plan is stated against.
    """
    K, D, dmin = int(cfg.K), int(cfg.D_max), int(cfg.D_min)
    states = reductions = transition_terms = 0
    for trace in model.traces:
        J = len(trace)
        states += J * K
        for b in range(1, J + 1):
            lo, hi = max(1, b - D), b - dmin
            width_terms = max(0, hi - lo + 1)
            reductions += width_terms * K
            transition_terms += K * K
    return {"forward_states": int(states),
            "forward_duration_reductions": int(reductions),
            "forward_transition_reductions": int(transition_terms),
            "forward_total_reductions": int(reductions + transition_terms)}


# ------------------------------------------------------------------------ memory model
def predict_memory(cfg: BenchConfig, model, group: str = "marg") -> dict:
    """Predicted bytes from exact array shapes and dtypes, before anything is allocated.

    Three tables of size `(J, J+1, K)` float64 exist in a full sampler at once: the
    `FFBSBlockTables` dense set (shared with its batched builder's target), the
    `FastBlockScoreTable` set the collapsed likelihood owns, and -- inside the batched
    forward -- the `np.stack` copy of one length class. The batched stack is the single
    largest transient and is counted at its worst case, one class holding every trace.

    `projected_banded_bytes` is NOT a measurement of this implementation. It is the
    arithmetic size of a layout that stored only the `D_max - D_min + 1` legal durations
    per start instead of a full `(J+1)` end axis.
    """
    K, D, dmin = int(cfg.K), int(cfg.D_max), int(cfg.D_min)
    lengths = [len(t) for t in model.traces]
    itemsize = 8

    dense_per_trace = [J * (J + 1) * K * itemsize for J in lengths]
    dense_total = int(sum(dense_per_trace))

    # one length class at worst holds every trace
    classes: dict = {}
    for J in lengths:
        classes[J] = classes.get(J, 0) + 1
    batched_stack = int(max(J * (J + 1) * K * itemsize * count
                            for J, count in classes.items()))
    alpha_bytes = int(sum((J + 1) * K * itemsize for J in lengths))
    # the batched core holds alpha and r for a whole class simultaneously
    chart_bytes = 2 * int(max((J + 1) * K * itemsize * count
                              for J, count in classes.items()))

    n_widths = D - dmin + 1
    n_starts = int(sum(max(0, J - dmin + 1) for J in lengths))
    cumulative_bytes = int(K * n_starts * D * itemsize)
    banded = int(sum(max(0, J - dmin + 1) for J in lengths) * n_widths * K * itemsize)

    # `build` and `primitives` construct only the FFBS dense set; `cond` and `marg`
    # build a sampler, which additionally owns the collapsed likelihood's
    # `FastBlockScoreTable` dense set. Counting two sets for a group that allocates one
    # would refuse configurations that are in fact safe, so the count follows the group.
    dense_copies = 2 if group in ("cond", "marg") else 1
    total = (dense_copies * dense_total + batched_stack + chart_bytes + alpha_bytes
             + cumulative_bytes)
    return {
        "group": group,
        "dense_block_table_bytes": dense_total,
        "dense_copies_live_in_this_group": dense_copies,
        "dense_block_table_bytes_all_copies": dense_copies * dense_total,
        "batched_stack_bytes_worst_class": batched_stack,
        "alpha_chart_bytes": alpha_bytes,
        "batched_alpha_r_bytes": chart_bytes,
        "fast_cumulative_bytes": cumulative_bytes,
        "predicted_arrays_total_bytes": int(total),
        "predicted_process_rss_bytes": int(total + 220 * 1024 ** 2),   # interpreter+numpy
        "projected_banded_bytes_NOT_IMPLEMENTED": banded,
        "projected_banded_saving_ratio_NOT_IMPLEMENTED": (
            float(dense_total / banded) if banded else float("nan")),
    }


MEMORY_CAP_BYTES = min(6 * 1024 ** 3, int(0.5 * 17179869184))


def physical_memory_bytes() -> int:
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                  text=True, check=True).stdout.strip())
    except Exception:
        return 0


def memory_cap_bytes() -> int:
    total = physical_memory_bytes()
    return min(6 * 1024 ** 3, int(0.5 * total)) if total else 6 * 1024 ** 3


def swapping_now() -> dict:
    """macOS swap usage. A configuration is refused if the machine is already swapping."""
    try:
        raw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except Exception:
        return {"available": False, "raw": "", "used_mb": 0.0, "swapping": False}
    used = swapins = swapouts = 0.0
    try:
        parts = raw.split()
        used = float(parts[parts.index("used") + 2].rstrip("M"))
    except Exception:
        used = 0.0
    counts = vm_stat_counts()
    swapins, swapouts = counts.get("swapins", 0.0), counts.get("swapouts", 0.0)
    return {"available": True, "raw": raw, "used_mb": used,
            "swapins": swapins, "swapouts": swapouts,
            "available_bytes": counts.get("available_bytes", 0)}


def vm_stat_counts() -> dict:
    """Free/inactive/speculative pages and the cumulative swap counters, from `vm_stat`.

    macOS keeps a compressed swap file in use at all times, so a nonzero `used` figure is
    not evidence of thrashing. The quantity a memory preflight can act on is how much
    physical memory is actually reclaimable right now, which is what this returns.
    """
    try:
        raw = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             check=True).stdout
    except Exception:
        return {}
    page = 4096
    values = {}
    for line in raw.splitlines():
        if "page size of" in line:
            try:
                page = int(line.split("page size of")[1].split()[0])
            except Exception:
                pass
            continue
        if ":" not in line:
            continue
        name, _, value = line.partition(":")
        try:
            values[name.strip().lower()] = float(value.strip().rstrip("."))
        except ValueError:
            continue
    free = values.get("pages free", 0.0)
    inactive = values.get("pages inactive", 0.0)
    speculative = values.get("pages speculative", 0.0)
    purgeable = values.get("pages purgeable", 0.0)
    return {
        "page_size": page,
        "available_bytes": int((free + inactive + speculative + purgeable) * page),
        "free_bytes": int(free * page),
        "swapins": values.get("swapins", 0.0),
        "swapouts": values.get("swapouts", 0.0),
    }


def memory_preflight(predicted_rss: int) -> dict:
    """Decide, before allocating anything, whether a configuration may run.

    Two independent gates, both conservative: the frozen absolute cap of
    `min(6 GB, 50% of physical RAM)`, and a live check that the prediction fits inside
    80% of the memory the kernel reports as currently reclaimable. Neither gate is ever
    relaxed to get a point measured.
    """
    cap = memory_cap_bytes()
    counts = vm_stat_counts()
    available = int(counts.get("available_bytes", 0))
    reasons = []
    if predicted_rss > cap:
        reasons.append(f"predicted RSS {predicted_rss / 2**30:.2f} GiB exceeds the "
                       f"frozen cap {cap / 2**30:.2f} GiB")
    if available and predicted_rss > 0.8 * available:
        reasons.append(f"predicted RSS {predicted_rss / 2**30:.2f} GiB exceeds 80% of "
                       f"the {available / 2**30:.2f} GiB currently reclaimable")
    return {"allowed": not reasons, "reasons": reasons, "cap_bytes": cap,
            "available_bytes": available, "predicted_rss_bytes": int(predicted_rss)}


# ------------------------------------------------------------------------ environment
def load_average() -> list:
    return [float(v) for v in os.getloadavg()]


def peak_rss_bytes() -> int:
    """`ru_maxrss` is bytes on Darwin (kilobytes on Linux); this process is Darwin-only."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def hardware_manifest() -> dict:
    return {
        "cpu_model": _sysctl("machdep.cpu.brand_string"),
        "physical_cores": int(_sysctl("hw.physicalcpu") or 0),
        "logical_cores": int(_sysctl("hw.logicalcpu") or 0),
        "performance_cores": int(_sysctl("hw.perflevel0.physicalcpu") or 0),
        "efficiency_cores": int(_sysctl("hw.perflevel1.physicalcpu") or 0),
        "total_ram_bytes": physical_memory_bytes(),
        "total_ram_gib": round(physical_memory_bytes() / 1024 ** 3, 2),
        "page_size_bytes": int(_sysctl("hw.pagesize") or 0),
        "macos_version": platform.mac_ver()[0],
        "machine": platform.machine(),
        "thermal_note": "Apple silicon exposes no per-core thermal counter to an "
                        "unprivileged process; `pmset -g therm` is recorded instead and "
                        "is empty on this platform when no thermal limit is active.",
        "pmset_therm": _pmset_therm(),
        "memory_cap_bytes": memory_cap_bytes(),
        "swap_at_capture": swapping_now(),
        "loadavg_at_capture": load_average(),
    }


def _pmset_therm() -> str:
    try:
        return subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def software_manifest() -> dict:
    import scipy
    def _git(*args):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return ""
    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_describe_base": _git("rev-parse", "564995efd056d7d33984f0ca1532386e6140ea0c"),
        "worktree": str(ROOT),
        "thread_env": {v: os.environ.get(v) for v in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "backend_under_test": "optimized_segmental_v1 (hpop.mcmc_optimized, all four "
                              "optimisation flags on)",
        "table_source": "batched",
    }


# ------------------------------------------------------------------- machine speed
_PROBE_MATRIX = None
PROBE_REPEATS = 30


def speed_probe(repeats: int = PROBE_REPEATS) -> float:
    """Seconds for a fixed, deterministic float64 workload. A ruler for the machine.

    Process CPU time is the usual defence against ambient load, and it is not enough on
    this hardware. An Apple-silicon machine has performance and efficiency cores, and a
    process moved from one to the other keeps a whole core -- `cpu / wall` stays at one --
    while running at roughly half the speed. Both wall time and CPU time then fall
    together when the machine goes quiet, and nothing in either number reveals it.

    This probe does the same arithmetic every time it is called. Any change in its
    duration is a change in the machine, not in the benchmark, so it makes core-type and
    frequency drift a measured quantity instead of an invisible one.
    """
    global _PROBE_MATRIX
    if _PROBE_MATRIX is None:
        _PROBE_MATRIX = np.random.default_rng(11).standard_normal((192, 192))
    matrix = _PROBE_MATRIX
    began = time.perf_counter()
    checksum = 0.0
    for _ in range(int(repeats)):
        checksum += float((matrix @ matrix)[0, 0])
    elapsed = time.perf_counter() - began
    if not np.isfinite(checksum):
        raise ValueError("speed probe produced a non-finite checksum")
    return elapsed


# --------------------------------------------------------------------------- statistics
def bootstrap_median_ci(samples, resamples: int = 2000, seed: int = 12345,
                        level: float = 0.95) -> dict:
    """Percentile bootstrap interval for the median, with a deterministic rng."""
    data = np.asarray([s for s in samples if np.isfinite(s)], dtype=float)
    if data.size == 0:
        return {"median": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "relative_half_width": float("inf"), "n": 0}
    if data.size == 1:
        value = float(data[0])
        return {"median": value, "lo": value, "hi": value,
                "relative_half_width": float("inf"), "n": 1}
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, data.size, size=(int(resamples), data.size))
    medians = np.median(data[draws], axis=1)
    tail = (1.0 - float(level)) / 2.0
    lo, hi = np.quantile(medians, [tail, 1.0 - tail])
    median = float(np.median(data))
    half = (float(hi) - float(lo)) / 2.0
    return {"median": median, "lo": float(lo), "hi": float(hi),
            "relative_half_width": float(half / median) if median > 0 else float("inf"),
            "n": int(data.size)}


def summarize(samples) -> dict:
    data = np.asarray([s for s in samples if np.isfinite(s)], dtype=float)
    if data.size == 0:
        return {"n": 0}
    ci = bootstrap_median_ci(data)
    return {
        "n": int(data.size), "median": float(np.median(data)),
        "mean": float(data.mean()), "min": float(data.min()), "max": float(data.max()),
        "q25": float(np.quantile(data, 0.25)), "q75": float(np.quantile(data, 0.75)),
        "iqr": float(np.quantile(data, 0.75) - np.quantile(data, 0.25)),
        "p90": float(np.quantile(data, 0.90)),
        "ci_lo": ci["lo"], "ci_hi": ci["hi"],
        "ci_relative_half_width": ci["relative_half_width"],
    }


# ------------------------------------------------------------------------ atomic files
def atomic_write(path: Path, text: str) -> None:
    """Write via a temporary in the same directory, then `os.replace`.

    `os.replace` is atomic within a filesystem, so a reader either sees the whole previous
    file or the whole new one. An interrupted write leaves the temporary behind and the
    real file untouched, which is what makes the resume safe.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)
