"""Stage 6E2 — the frozen trace-level synthetic corpus.

## Why a new corpus exists at all

The Stage 6D corpus is 500 **independent blocks** of `T = 20` drawn from a single `U_TRUE`
with `K = 1` skill. It has no trace structure, no skill sequence and no `pi`/`P`, because
Stage 6D had oracle boundaries and oracle labels and therefore needed none. Concatenating
those blocks would not produce a registered trace corpus: it would manufacture a
skill-transition structure the generator never sampled, and any recovery of `P` would then
be recovering an artefact of the concatenation order. §11's fallback applies exactly as
written, and a trace-level corpus is generated and frozen here instead.

## What the generator does, and what it must not do

Each trace is produced by the registered generative model, in order:

1. draw the number of blocks `L_n`;
2. draw the skill path `z_n` from `pi_TRUE` and `P_TRUE` — `P_TRUE` has a zero diagonal, so
   consecutive blocks always carry different skills, which is the registered constraint,
   not a generator convenience;
3. draw each block width from the **registered boundary prior**, `delta_B = 0.15` truncated
   to `[3, 12]`, so the true widths follow the law the target assumes rather than a
   convenient uniform;
4. generate each block by the same recurrent equations the likelihood evaluates, restarting
   `q` at zero for every block.

The generation seed is fixed before any inference and **not** searched. `exposure_audit`
is run afterwards to record whether the corpus contains the repeat, upstream-repeat and
recomputation events the recurrent scalars need, but the seed is not changed if it does
not: that would be selecting a corpus on a property of the answer.

The split is at the **trace** level. Held-out traces share the truth's parameters and
nothing else — no block of a training trace appears in a held-out trace.

## The three skills

`K = 3` reusable skills over `m = 5` roles need three *distinct* nondegenerate induced
orders, or the labels would be unidentifiable for a reason that has nothing to do with the
sampler. The three registered matrices give a chain-plus-isolate, a differently-oriented
chain-plus-isolate, and a fan; `assert_distinct_orders` checks that they are pairwise
different and that none is an antichain or a total order.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_rfs import (
    RecurrentRFSParameters, recurrent_back_costs, recurrent_feasibility,
    recurrent_step_probabilities, recurrent_validity_update,
)
from hpop.mcmc_original.stage6e_frozen import (
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH, N_ROLES, N_SKILLS,
)

__all__ = [
    "U_TRUE_BY_SKILL", "PI_TRUE", "P_TRUE", "SCALAR_TRUTH", "CORPUS_SEED",
    "N_TRAIN_TRACES", "N_HELDOUT_TRACES", "BLOCKS_PER_TRACE",
    "Stage6ETrace", "Stage6ECorpus", "generate_corpus", "assert_distinct_orders",
    "width_distribution", "corpus_hash",
]

# ------------------------------------------------------------------ registered truth
U_TRUE_BY_SKILL = np.array([
    # skill 0: 0 > 2 > 3 > 4 (a chain), role 1 incomparable with everything
    [[4.0, 4.0], [0.0, 5.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0]],
    # skill 1: 1 > 2 > 4 > 0 (a chain on different roles), role 3 incomparable
    [[1.0, 1.0], [4.0, 4.0], [3.0, 3.0], [0.0, 5.0], [2.0, 2.0]],
    # skill 2: a fan -- 4 dominates everything, 2 > 3, roles 0 and 1 otherwise free
    [[3.0, 0.0], [0.0, 3.0], [2.0, 2.0], [1.0, 1.0], [4.0, 4.0]],
], dtype=float)

PI_TRUE = np.array([0.50, 0.30, 0.20])
P_TRUE = np.array([[0.00, 0.65, 0.35],
                   [0.30, 0.00, 0.70],
                   [0.75, 0.25, 0.00]])

SCALAR_TRUTH = {"beta": 1.5, "omega": math.log(0.85 / 0.15),
                "lambda_rep": 0.8, "lambda_back": 0.25}
EPSILON_TRUE = 0.02

CORPUS_SEED = 6_053_000            # fixed before any inference; never searched
N_TRAIN_TRACES = 100
N_HELDOUT_TRACES = 45
BLOCKS_PER_TRACE = (4, 5, 6)       # drawn uniformly; mean 5 -> ~500 training blocks


def assert_distinct_orders(u_by_skill=None) -> dict:
    """Three pairwise-distinct induced orders, none an antichain and none total."""
    u_by_skill = U_TRUE_BY_SKILL if u_by_skill is None else np.asarray(u_by_skill)
    K, m, _ = u_by_skill.shape
    closures = [precedence_from_u(u_by_skill[k]) for k in range(K)]
    maximum = m * (m - 1) // 2
    rows = []
    for k, closure in enumerate(closures):
        count = int(closure.sum())
        rows.append({"skill": k, "relation_count": count,
                     "is_antichain": count == 0, "is_total_order": count == maximum,
                     "pairs": sorted((int(i), int(j)) for i in range(m)
                                     for j in range(m) if closure[i, j])})
    distinct = all(not np.array_equal(closures[a], closures[b])
                   for a in range(K) for b in range(a + 1, K))
    if not distinct:
        raise AssertionError("the registered skills must induce distinct partial orders")
    for row in rows:
        if row["is_antichain"] or row["is_total_order"]:
            raise AssertionError(f"skill {row['skill']} is degenerate: {row}")
    return {"skills": rows, "pairwise_distinct": True,
            "max_possible_relations": maximum}


def width_distribution(delta_b: float = DELTA_B, min_width: int = MIN_BLOCK_WIDTH,
                       max_width: int = MAX_BLOCK_WIDTH) -> np.ndarray:
    """The registered boundary prior's width law, truncated to the legal widths.

    A segment of width `w` costs `(w - 1) log(1 - delta_B)` in non-cut internal positions
    and its terminating cut costs `log delta_B`, so `p(w) ∝ (1 - delta_B)^(w-1)` on
    `[min_width, max_width]`. Drawing the truth from this law rather than from a uniform
    keeps the generated corpus consistent with the target the sampler assumes.
    """
    widths = np.arange(min_width, max_width + 1)
    weights = (1.0 - delta_b) ** (widths - 1)
    return weights / weights.sum()


@dataclass(frozen=True)
class Stage6ETrace:
    roles: tuple                 # the observed CPA occurrence sequence -- the ONLY input
    true_boundaries: tuple       # internal cut positions (hidden)
    true_labels: tuple           # one skill per block (hidden)
    split: str

    @property
    def length(self) -> int:
        return len(self.roles)

    @property
    def n_blocks(self) -> int:
        return len(self.true_labels)

    def true_key(self) -> tuple:
        """`((end, skill), ...)` — the hidden truth in the sampler's own representation."""
        ends = list(self.true_boundaries) + [self.length]
        return tuple(zip(ends, self.true_labels))

    def true_occurrence_labels(self) -> np.ndarray:
        out = np.empty(self.length, dtype=np.int8)
        start = 0
        for end, skill in self.true_key():
            out[start:end] = skill
            start = end
        return out


@dataclass(frozen=True)
class Stage6ECorpus:
    train: tuple
    heldout: tuple
    u_true: np.ndarray
    pi_true: np.ndarray
    p_true: np.ndarray
    scalar_truth: dict
    epsilon: float
    delta_b: float
    seed: int
    config: dict = field(default_factory=dict)

    def traces(self, split: str = "train") -> tuple:
        return tuple(t.roles for t in (self.train if split == "train" else self.heldout))

    def true_keys(self, split: str = "train") -> tuple:
        return tuple(t.true_key() for t in (self.train if split == "train"
                                            else self.heldout))

    @property
    def n_train_blocks(self) -> int:
        return int(sum(t.n_blocks for t in self.train))

    @property
    def n_heldout_blocks(self) -> int:
        return int(sum(t.n_blocks for t in self.heldout))


def _generate_trace(rng, u_by_skill, params, delta_b, split: str) -> Stage6ETrace:
    widths_support = np.arange(MIN_BLOCK_WIDTH, MAX_BLOCK_WIDTH + 1)
    width_p = width_distribution(delta_b)
    n_blocks = int(rng.choice(BLOCKS_PER_TRACE))

    path = [int(rng.choice(N_SKILLS, p=PI_TRUE))]
    for _ in range(n_blocks - 1):
        path.append(int(rng.choice(N_SKILLS, p=P_TRUE[path[-1]])))

    roles, ends, running = [], [], 0
    for skill in path:
        width = int(rng.choice(widths_support, p=width_p))
        u = u_by_skill[skill]
        precedence = precedence_from_u(u)
        q = np.zeros(u.shape[0])                       # q_0 = 0 for EVERY block
        for _ in range(width):
            mixed = recurrent_step_probabilities(u, q, params)
            y = int(rng.choice(u.shape[0], p=mixed))
            roles.append(y)
            q = recurrent_validity_update(y, precedence, q, params.shared_omega)
        running += width
        ends.append(running)
    return Stage6ETrace(roles=tuple(roles), true_boundaries=tuple(ends[:-1]),
                        true_labels=tuple(path), split=split)


def generate_corpus(seed: int = CORPUS_SEED, n_train: int = N_TRAIN_TRACES,
                    n_heldout: int = N_HELDOUT_TRACES,
                    delta_b: float = DELTA_B) -> Stage6ECorpus:
    """Generate and freeze the Stage 6E2 corpus. The seed is fixed, never searched."""
    orders = assert_distinct_orders()
    params = RecurrentRFSParameters(
        beta=SCALAR_TRUTH["beta"], epsilon=EPSILON_TRUE,
        shared_omega=SCALAR_TRUTH["omega"], lambda_rep=SCALAR_TRUTH["lambda_rep"],
        lambda_back=SCALAR_TRUTH["lambda_back"])
    # Separate recorded streams, so the train/test split is at the trace level by
    # construction and no held-out trace can share generator state with a training one.
    rng_train = np.random.default_rng(seed)
    rng_heldout = np.random.default_rng(seed + 10_000)
    train = tuple(_generate_trace(rng_train, U_TRUE_BY_SKILL, params, delta_b, "train")
                  for _ in range(n_train))
    heldout = tuple(_generate_trace(rng_heldout, U_TRUE_BY_SKILL, params, delta_b,
                                    "heldout")
                    for _ in range(n_heldout))
    config = {
        "generator_seed": seed, "seed_train": seed, "seed_heldout": seed + 10_000,
        "n_train_traces": n_train, "n_heldout_traces": n_heldout,
        "blocks_per_trace_support": list(BLOCKS_PER_TRACE),
        "width_law": "registered boundary prior, p(w) proportional to (1-delta_B)^(w-1), "
                     f"truncated to [{MIN_BLOCK_WIDTH}, {MAX_BLOCK_WIDTH}]",
        "width_probabilities": width_distribution(delta_b).tolist(),
        "delta_B": delta_b, "epsilon": EPSILON_TRUE,
        "n_skills": N_SKILLS, "n_roles": N_ROLES,
        "U_true": U_TRUE_BY_SKILL.tolist(), "pi_true": PI_TRUE.tolist(),
        "P_true": P_TRUE.tolist(), "scalar_truth": dict(SCALAR_TRUTH),
        "induced_orders": orders,
        "split_level": "trace",
        "seed_was_searched": False,
        "likelihood_conditions_on_widths": True, "duration_model": None,
    }
    return Stage6ECorpus(train=train, heldout=heldout, u_true=U_TRUE_BY_SKILL,
                         pi_true=PI_TRUE, p_true=P_TRUE,
                         scalar_truth=dict(SCALAR_TRUTH), epsilon=EPSILON_TRUE,
                         delta_b=delta_b, seed=seed, config=config)


def corpus_hash(corpus: Stage6ECorpus) -> str:
    """A hash over everything that defines the corpus, truth included."""
    payload = {
        "train": [[list(t.roles), list(t.true_boundaries), list(t.true_labels)]
                  for t in corpus.train],
        "heldout": [[list(t.roles), list(t.true_boundaries), list(t.true_labels)]
                    for t in corpus.heldout],
        "u_true": corpus.u_true.tolist(), "pi_true": corpus.pi_true.tolist(),
        "p_true": corpus.p_true.tolist(), "scalar_truth": corpus.scalar_truth,
        "epsilon": corpus.epsilon, "delta_b": corpus.delta_b, "seed": corpus.seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def exposure_audit_traces(corpus: Stage6ECorpus, split: str = "train") -> dict:
    """Recurrent-event exposure, replayed per true block from `q_0 = 0`.

    Reported, never used to select the corpus. A dataset with no upstream repeats cannot
    inform `lambda_back`, and saying so is more useful than a seed search that hides it.
    """
    traces = corpus.train if split == "train" else corpus.heldout
    counts = {"valid_repeat": 0, "leaf_repeat": 0, "upstream_repeat": 0,
              "recomputation": 0, "total_steps": 0}
    omega = corpus.scalar_truth["omega"]
    for trace in traces:
        start = 0
        for end, skill in trace.true_key():
            u = corpus.u_true[skill]
            precedence = precedence_from_u(u)
            q = np.zeros(u.shape[0])
            ever = np.zeros(u.shape[0], dtype=bool)
            for y in trace.roles[start:end]:
                counts["total_steps"] += 1
                back = recurrent_back_costs(precedence, q, omega)
                feasible = recurrent_feasibility(precedence, q)
                if q[y] > 0.0:
                    counts["valid_repeat"] += 1
                    if back[y] <= 1e-12:
                        counts["leaf_repeat"] += 1
                    else:
                        counts["upstream_repeat"] += 1
                if ever[y] and q[y] < 1 - 1e-8 and feasible[y] > 0.0:
                    counts["recomputation"] += 1
                ever[y] = True
                q = recurrent_validity_update(y, precedence, q, omega)
            start = end
    return counts
