"""Step 7B — the Stage 6E joint sampler with the `(S, z)` update replaced by exact FFBS.

Step 7B changes **one kernel** and nothing else. The posterior is the Stage 6E posterior,
term for term; the sweep order is the Stage 6E sweep order; and every update after
`(S, z)` is not merely "equivalent to" Stage 6E's — it *is* Stage 6E's, because
`ffbs_sweep_once` calls `stage6e_sampler.sweep_once` with the local segmentation phase
switched off and lets that function do the rest:

    Stage 6E:  local MH on (S, z)  ->  (pi, P) -> U -> rho -> beta -> omega -> lam_rep -> lam_back
    Step 7B:   FFBS draw of (S, z) ->  [ exactly the same call, n_proposals_per_trace = 0 ]

That structure is deliberate. Re-implementing the parameter phase would have meant
maintaining a second copy of five acceptance ratios and hoping a parity test caught any
drift; delegating means there is only one copy and the parity is a property of the code
rather than of a test. The test is still there — `test_stage7b_joint_sweep` runs both
paths on the same state with the same generator and requires bit-identical output — but it
is a regression, not the load-bearing argument.

## Why the segmentation step has no Hastings ratio

Conditional on `(U, rho, beta, omega, lambda_rep, lambda_back, pi, P)` the per-trace
segmentation posterior is exactly

    p(S_n, z_n | x_n, Theta) proportional to w(S_n, z_n | Theta),

and Step 7A established that the FFBS chart normalises that distribution to machine
precision and that backward sampling draws from it. So this is a **Gibbs** update: the
draw is already from the correct conditional and its acceptance probability is 1 by
construction. There is no proposal density, no reverse move, no acceptance test — and
`test_no_hastings_correction_in_the_segmentation_update` reads this module's source to
keep it that way.

## Conditional independence across traces

Given the global state the traces are independent, so one segmentation sweep draws each
trace's `(S_n, z_n)` from its own chart. The draws are taken **sequentially from the one
chain generator**, in trace order — not from per-trace child generators. That choice is
recorded here because it is what makes a sweep reproducible from a single seed: the whole
sweep, segmentation and parameters alike, consumes one stream in a fixed order.

## The block-table lifecycle, which is the whole efficiency story

FFBS needs `log_block_scores[a, b, k]` for **every candidate block** — `O(J x width x K)`
of them per trace. The parameter updates that follow need the likelihood of the **selected
blocks only**, which is what `SkillBlockLikelihood` already computes. So the candidate
table is built once per segmentation sweep and is never rebuilt for a scalar proposal:

    refresh tables  ->  FFBS draw  ->  mark stale  ->  parameter phase (selected blocks)

`FFBSBlockTables` refuses to be read once stale, and it fingerprints the parameters it was
built at, so a read after any parameter has moved raises instead of silently scoring the
wrong model. `builds` counts refreshes, and the tests assert it equals the number of
segmentation sweeps — one build per sweep, never one per proposal.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from hpop.mcmc_original.block_score_adapters import build_log_block_scores
from hpop.mcmc_original.fast_block_tables import FastBlockScoreTable
from hpop.mcmc_original.fast_segmentation_kernel import key_of, segmentation_of
from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.proposals import MoveType
from hpop.mcmc_original.semi_markov_ffbs import backward_sample, forward
from hpop.mcmc_original.stage6d_frozen import SCALAR_ORDER
from hpop.mcmc_original.stage6e_block_table import BlockScoreTable
from hpop.mcmc_original.stage6e_sampler import (
    Stage6EChainResult, Stage6ESampler, _write_checkpoint, boundary_hamming,
    occurrence_label_changes, sweep_once,
)
from hpop.mcmc_original.stage6e_state import Stage6EModel, Stage6EState
from hpop.mcmc_original.transitions import log_transition_matrix

__all__ = [
    "FFBSBlockTables", "Stage7BSampler", "ffbs_segmentation_draw", "ffbs_sweep_once",
    "key_movement", "run_stage7b_chain", "TABLE_SOURCES",
]

TABLE_SOURCES = ("fast", "batched", "adapter")


# ------------------------------------------------------------------ candidate tables
@dataclass
class FFBSBlockTables:
    """Per-trace `(J, J+1, K)` candidate-block log scores, with an explicit lifecycle.

    `source="batched"` fills the tables from `BlockScoreTable`, the width-bucketed builder
    Stage 6E already validated against the per-block scorer; `source="adapter"` fills them
    with the Step 7A adapter, one `RecurrentBlockScorer.replay` per block. The two are the
    eager and the batched route to the same numbers and `assert_sources_agree` pins them
    against each other — the batched one is the default because it is what makes a
    100-trace sweep affordable, not because it is trusted more.
    """

    model: Stage6EModel
    source: str = "fast"
    _tables: list = field(default_factory=list, repr=False)
    _batched: BlockScoreTable | None = field(default=None, repr=False)
    _fast: FastBlockScoreTable | None = field(default=None, repr=False)
    _index: list = field(default_factory=list, repr=False)
    _fingerprint: tuple | None = field(default=None, repr=False)
    stale: bool = True
    version: int = 0
    builds: int = 0
    reads: int = 0
    build_seconds: float = 0.0
    last_refresh: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.source not in TABLE_SOURCES:
            raise ValueError(f"unknown table source {self.source!r}; "
                             f"expected one of {TABLE_SOURCES}")
        K = int(self.model.n_skills)
        self._tables = [np.full((len(t), len(t) + 1, K), -np.inf)
                        for t in self.model.traces]
        if self.source == "fast":
            # the batched builder owns its own dense tables and writes them in place, so
            # this object hands them out rather than keeping a second copy
            self._fast = FastBlockScoreTable(
                traces=self.model.traces, epsilon=self.model.epsilon, n_skills=K,
                min_width=self.model.min_width, max_width=self.model.max_width,
                n_roles=self.model.n_roles)
            self._tables = self._fast.tables
        if self.source == "batched":
            self._batched = BlockScoreTable(
                traces=self.model.traces, epsilon=self.model.epsilon, n_skills=K,
                min_width=self.model.min_width, max_width=self.model.max_width)
            # one flat index per trace, computed once: (rows in the batched table) ->
            # (a, b, k) in the dense one. Rebuilt never; the block set is fixed by the
            # trace lengths and the width bounds, not by the parameters.
            for n, trace in enumerate(self.model.traces):
                rows, a_ix, b_ix, k_ix = [], [], [], []
                for (tn, a, width), row in self._batched._row.items():
                    if tn != n:
                        continue
                    for k in range(K):
                        rows.append(row)
                        a_ix.append(a)
                        b_ix.append(a + width)
                        k_ix.append(k)
                self._index.append((np.array(k_ix), np.array(rows), np.array(a_ix),
                                    np.array(b_ix)))

    # -- the only writer ------------------------------------------------------------------
    def refresh(self, state: Stage6EState) -> None:
        """Rebuild every candidate score at the state's current parameters."""
        began = time.perf_counter()
        if self.source == "fast":
            self.last_refresh = self._fast.refresh(
                state.u_by_skill, state.beta, state.omega, state.lambda_rep,
                state.lambda_back)
        elif self.source == "batched":
            self._batched.refresh(state.u_by_skill, state.beta, state.omega,
                                  state.lambda_rep, state.lambda_back)
            flat = self._batched._table
            for n, (k_ix, rows, a_ix, b_ix) in enumerate(self._index):
                self._tables[n][a_ix, b_ix, k_ix] = flat[k_ix, rows]
        else:
            scorer = self.model.scorer_for(state)
            for n, trace in enumerate(self.model.traces):
                self._tables[n] = build_log_block_scores(
                    scorer, n, len(trace), self.model.n_skills, self.model.min_width,
                    self.model.max_width)
        self._fingerprint = self._parameters_of(state)
        self.stale = False
        self.version += 1
        self.builds += 1
        self.build_seconds += time.perf_counter() - began

    def mark_stale(self) -> None:
        self.stale = True

    @staticmethod
    def _parameters_of(state: Stage6EState) -> tuple:
        return (np.asarray(state.u_by_skill, dtype=float).tobytes(), float(state.beta),
                float(state.omega), float(state.lambda_rep), float(state.lambda_back))

    def tables_for(self, state: Stage6EState) -> list:
        """The tables, or an exception. Never a silently stale read."""
        if self.stale:
            raise AssertionError(
                "the FFBS candidate tables are stale: they are built once per "
                "segmentation sweep and invalidated as soon as it ends, so a read here "
                "would score a parameter state they were not built at")
        if self._fingerprint != self._parameters_of(state):
            raise AssertionError(
                "the FFBS candidate tables were built at different parameters than the "
                "state now carries")
        self.reads += 1
        return self._tables


def assert_sources_agree(model: Stage6EModel, state: Stage6EState,
                         tolerance: float = 1e-9) -> dict:
    """The batched and eager candidate tables must be the same numbers.

    Same blocks, same `q_0 = 0`, different loop order — so any difference beyond
    floating-point noise is a defect in one of them.
    """
    built = {}
    for source in TABLE_SOURCES:
        tables = FFBSBlockTables(model=model, source=source)
        tables.refresh(state)
        built[source] = [np.array(t, copy=True) for t in tables.tables_for(state)]
    worst = 0.0
    same_support = True
    for a, b in zip(built["batched"], built["adapter"]):
        finite = np.isfinite(a) & np.isfinite(b)
        same_support &= bool((np.isfinite(a) == np.isfinite(b)).all())
        if finite.any():
            worst = max(worst, float(np.abs(a[finite] - b[finite]).max()))
    return {"max_absolute_difference": worst, "same_support": same_support,
            "tolerance": tolerance, "n_traces": len(built["batched"]),
            "pass": bool(worst <= tolerance and same_support)}


# ------------------------------------------------------------------ movement counting
def key_movement(before: tuple, after: tuple) -> tuple:
    """`(boundary Hamming, occurrence-label changes)` between two segmentation keys.

    Identical in value to `boundary_hamming` and `occurrence_label_changes`, which is what
    `test_fast_movement_counters_match_the_stage6e_ones` requires. The difference is that
    those two materialise a per-occurrence numpy array for every trace of every sweep;
    this walks the two span lists once. On a 100-trace sweep that is the difference
    between ~200 array constructions and none, for a quantity that is only a diagnostic.
    """
    cuts_before = {end for end, _ in before[:-1]}
    cuts_after = {end for end, _ in after[:-1]}
    hamming = len(cuts_before ^ cuts_after)

    changes = 0
    i = j = 0
    position = start_before = start_after = 0
    while i < len(before) and j < len(after):
        end_before, skill_before = before[i]
        end_after, skill_after = after[j]
        upto = min(end_before, end_after)
        if skill_before != skill_after:
            changes += upto - position
        position = upto
        if end_before == upto:
            i += 1
            start_before = end_before
        if end_after == upto:
            j += 1
            start_after = end_after
    del start_before, start_after
    return hamming, changes


# ------------------------------------------------------------------ the Gibbs update
def ffbs_segmentation_draw(model: Stage6EModel, state: Stage6EState,
                           tables: FFBSBlockTables, rng) -> dict:
    """Draw `(S_n, z_n)` for every trace from its exact conditional. No accept/reject.

    Returns the new keys, the movement against the incoming segmentation, and each
    trace's `log Z_n(theta)` — the exact marginal segmentation likelihood, which is the
    same quantity the Stage 6E1B reference integrates against and is recorded so the two
    can be compared directly.
    """
    log_pi = np.log(state.pi)
    log_transition = log_transition_matrix(state.transition)
    previous = [key_of(s) for s in state.segmentations]

    keys, normalizers = [], []
    movement = {"boundary_hamming": 0, "label_changes": 0, "states_changed": 0}
    for n, table in enumerate(tables.tables_for(state)):
        chart = forward(table, log_pi, log_transition, model.delta_b, model.max_width,
                        model.min_width)
        blocks = backward_sample(chart, rng)          # exact draw; acceptance is 1
        key = tuple((int(b), int(k)) for _, b, k in blocks)
        keys.append(key)
        normalizers.append(chart.log_normalizer)
        hamming, changes = key_movement(previous[n], key)
        movement["boundary_hamming"] += hamming
        movement["label_changes"] += changes
        movement["states_changed"] += int(key != previous[n])
    return {"keys": tuple(keys), "movement": movement,
            "log_normalizers": np.array(normalizers, dtype=float)}


# ---------------------------------------------------------------------------- the sweep
@dataclass
class Stage7BSampler:
    """The Stage 6E sampler, plus the candidate tables, minus the local segmentation MH."""

    model: Stage6EModel
    scales: dict
    table_source: str = "batched"
    max_shift: int | None = None
    draw_segmentation: bool = True          # test hook: False reproduces Stage 6E exactly
    _stage6e: Stage6ESampler = field(default=None, repr=False)
    _tables: FFBSBlockTables = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # n_proposals_per_trace = 0: the Stage 6E segmentation phase is switched off and
        # this object is used for the phases after it, unmodified.
        self._stage6e = Stage6ESampler(model=self.model, scales=dict(self.scales),
                                       n_proposals_per_trace=0,
                                       max_shift=self.max_shift, use_block_table=False)
        self._tables = FFBSBlockTables(model=self.model, source=self.table_source)

    @property
    def tables(self) -> FFBSBlockTables:
        return self._tables


def ffbs_sweep_once(state: Stage6EState, sampler: Stage7BSampler, rng) -> Stage6EState:
    """One Step 7B global sweep: exact `(S, z)` Gibbs draw, then the Stage 6E phases."""
    state = state.copy()
    ffbs = None
    if sampler.draw_segmentation:
        sampler.tables.refresh(state)
        ffbs = ffbs_segmentation_draw(sampler.model, state, sampler.tables, rng)
        state.segmentations = tuple(segmentation_of(k) for k in ffbs["keys"])
        # the parameter phase is about to move U and the scalars, so nothing built at the
        # current parameters may be read again
        sampler.tables.mark_stale()

    # every update after (S, z) is Stage 6E's own, called rather than reimplemented
    state = sweep_once(state, sampler._stage6e, rng)

    if ffbs is not None:
        state.components["boundary_hamming_moved"] = ffbs["movement"]["boundary_hamming"]
        state.components["label_changes_moved"] = ffbs["movement"]["label_changes"]
        state.components["ffbs_states_changed"] = ffbs["movement"]["states_changed"]
        state.components["ffbs_log_normalizer_total"] = float(
            ffbs["log_normalizers"].sum())
        state.proposed["S_z_ffbs"] = state.proposed.get("S_z_ffbs", 0) + len(
            sampler.model.traces)
        state.accepted["S_z_ffbs"] = state.accepted.get("S_z_ffbs", 0) + len(
            sampler.model.traces)      # a Gibbs draw is accepted by construction
    return state


# ------------------------------------------------------------------------------ the run
def run_stage7b_chain(model: Stage6EModel, start: Stage6EState, scales: dict,
                      num_sweeps: int, burn_in: int, thin: int, seed: int,
                      chain: int = 0, table_source: str = "batched",
                      max_shift: int | None = None, rng=None,
                      state: Stage6EState | None = None, store_labels: bool = True,
                      store_keys: bool = True, draw_segmentation: bool = True,
                      progress_every: int = 0, checkpoint_path=None,
                      checkpoint_every: int = 0, movement_tracker=None
                      ) -> Stage6EChainResult:
    """One Step 7B chain. Storage and bookkeeping mirror `run_stage6e_chain` exactly;
    the only difference in the loop is which sweep function is called."""
    if burn_in >= num_sweeps:
        raise ValueError("burn_in must be smaller than num_sweeps")
    if thin < 1:
        raise ValueError("thin must be at least 1")

    sampler = Stage7BSampler(model=model, scales=dict(scales), table_source=table_source,
                             max_shift=max_shift, draw_segmentation=draw_segmentation)
    rng = np.random.default_rng(seed) if rng is None else rng
    state = (start.copy() if state is None else state)
    state.chain = chain
    for bucket in (state.proposed, state.accepted, state.invalid):
        for move in MoveType.ALL:
            bucket.setdefault(move, 0)
        for name in ("U", "rho", "pi_P", "S_z_ffbs", *SCALAR_ORDER):
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

    k = 0
    began = time.perf_counter()
    for i in range(start_iteration, num_sweeps):
        before_p, before_a = dict(state.proposed), dict(state.accepted)
        state = ffbs_sweep_once(state, sampler, rng)
        movement["boundary_hamming"] += state.components.get("boundary_hamming_moved", 0)
        movement["label_changes"] += state.components.get("label_changes_moved", 0)
        movement["states_changed"] += state.components.get("ffbs_states_changed", 0)
        if movement_tracker is not None:
            # observed every sweep, never read by the sampler: a registered run must not
            # be steered by its own movement diagnostics
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
            # A 50,000-sweep chain on the Stage 6E2 corpus is many hours, so an
            # interruption must not cost the whole run. The state carries its own RNG, so
            # a resume continues bit-identically; the partial draws are written beside it.
            # This is the Stage 6E writer, called rather than reimplemented, so the two
            # checkpoint formats cannot drift.
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
    result.movement_tracker = movement_tracker
    result.table_builds = int(sampler.tables.builds)
    result.table_build_seconds = float(sampler.tables.build_seconds)
    return result


def sweep_cost_breakdown(model: Stage6EModel, state: Stage6EState, scales: dict,
                         n_sweeps: int = 50, seed: int = 0,
                         table_source: str = "batched") -> dict:
    """Where a Step 7B sweep's time goes. Cost only — no draw is retained."""
    sampler = Stage7BSampler(model=model, scales=dict(scales), table_source=table_source)
    rng = np.random.default_rng(seed)
    current = state.copy()

    began = time.perf_counter()
    for _ in range(n_sweeps):
        current = ffbs_sweep_once(current, sampler, rng)
    total = time.perf_counter() - began

    # one isolated chart + draw, at the current parameters
    sampler.tables.refresh(current)
    tables = sampler.tables.tables_for(current)
    log_pi = np.log(current.pi)
    log_p = log_transition_matrix(current.transition)
    chart_began = time.perf_counter()
    charts = [forward(t, log_pi, log_p, model.delta_b, model.max_width, model.min_width)
              for t in tables]
    chart_seconds = time.perf_counter() - chart_began
    draw_began = time.perf_counter()
    for _ in range(100):
        for chart in charts:
            backward_sample(chart, rng)
    draw_seconds = (time.perf_counter() - draw_began) / 100
    sampler.tables.mark_stale()

    return {"n_sweeps": int(n_sweeps), "table_source": table_source,
            "total_seconds": total, "seconds_per_sweep": total / max(1, n_sweeps),
            "block_table_seconds_per_sweep": sampler.tables.build_seconds / max(
                1, sampler.tables.builds),
            "chart_seconds_per_sweep": chart_seconds,
            "backward_draw_seconds_per_sweep": draw_seconds,
            "table_builds": int(sampler.tables.builds),
            "n_traces": len(model.traces),
            "log_normalizer_total": float(sum(c.log_normalizer for c in charts))}
