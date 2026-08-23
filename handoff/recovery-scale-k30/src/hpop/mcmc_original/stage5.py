"""Stage 5 — static multi-trace recovery, reusing the VERIFIED partial-order sampler.

    Stage 5A   oracle segmentation S  ->  infer U_k, rho_k, beta_k (per skill) and P
    Stage 5B   oracle skill library   ->  infer the segmentations S

The update rules for ``U``, ``rho`` and the softmax ``beta`` are **not implemented
here**. They come from the vendored, already-verified ``mcmc_simulation_po``
(``hpop.vendored``). This module only:

1. maps oracle-assigned skill segments into that sampler's input format;
2. runs one independent sampler state per skill;
3. keeps the already-validated Dirichlet transition Gibbs update;
4. keeps the already-validated ``LocalMoveKernel`` for boundaries.

Notation, kept strictly separate (the old sampler calls the latent dimension ``K``):

    n_skills    number of reusable skill types            = 4
    latent_dim  number of columns of each U_k             = 2   (passed as fixed_K)
    skill_id    index of one reusable skill

``fixed_K=latent_dim`` disables the old reversible-jump dimension update. The old
result fields ``K_trace`` / ``K_final`` refer to the **latent product-order
dimension**, never to the size of the skill library; they are re-exported here as
``latent_dim_trace`` / ``latent_dim_final``.

Likelihood branch is ``log_successors_queue_jump`` for both generation and fitting —
the branch in which ``beta`` is a live inverse temperature. ``hpop``'s own
``static_bpop`` is numerically identical to it (verified to 1.8e-15 over 9,600 cases),
so the generator and the fitted model are the same model.
"""

from __future__ import annotations

import contextlib
import io
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from hpop.mcmc_original import toy
from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.static_bpop import (
    bpop_step_probabilities,
    frontier,
    remaining_successor_count,
    sample_bpop_sequence,
)
from hpop.mcmc_original.transitions import (
    allowed_next,
    dirichlet_posterior_params,
    posterior_mean_transition_matrix,
    transition_counts,
)
from hpop.mcmc_original.types import Segment, Segmentation, SkillTemplate
from hpop.vendored import ensure_importable

ensure_importable()
from src.mcmc.hpo_po_hm_mcmc_k_optim import mcmc_simulation_po  # noqa: E402

__all__ = [
    "N_SKILLS", "LATENT_DIM", "NOISE_OPTION", "SKILL_NAMES",
    "SKILL_A", "SKILL_D", "SKILL_F", "SKILL_E",
    "P_TRUE", "PI_TRUE", "stage5_skills", "assert_stage5_library",
    "PODataset", "build_po_dataset_for_skill", "IncompatibleSegment",
    "SyntheticTrace", "generate_corpus",
    "count_support_compatible_tilings", "sample_random_legal_segmentation",
    "run_skill_po_mcmc", "infer_transitions",
]

# ---------------------------------------------------------------------------
# fixed constants — none of these are inferred
# ---------------------------------------------------------------------------

N_SKILLS = 4
LATENT_DIM = 2
NOISE_OPTION = "log_successors_queue_jump"

SKILL_A, SKILL_D, SKILL_F, SKILL_E = 0, 1, 2, 3
SKILL_NAMES = ("A", "D", "F", "E")

LABELS = ((0, 1), (0, 1), (2, 3), (0, 1, 2, 3))
U_A = np.array([[1.0, 1.0], [0.0, 0.0]])
U_D = np.array([[1.0, 0.0], [0.0, 1.0]])
U_F = np.array([[1.0, 1.0], [0.0, 0.0]])
U_E = np.array([[2.0, 2.0], [3.0, -1.0], [1.0, 0.0], [0.0, 1.0]])   # U_ASYM: beta is live
U_TRUE = (U_A, U_D, U_F, U_E)

P_TRUE = np.array([
    [0.00, 0.10, 0.75, 0.15],
    [0.10, 0.00, 0.70, 0.20],
    [0.55, 0.15, 0.00, 0.30],
    [0.35, 0.35, 0.30, 0.00],
])
PI_TRUE = np.full(N_SKILLS, 0.25)
SEGMENTS_PER_TRACE = (3, 4, 5, 6)

# verified sampler hyperparameters (its own defaults / conventions)
DR = 0.9
RHO_PRIOR = 1.0
NOISE_BETA_PRIOR = 1.0
SOFTMAX_BETA_PRIOR = (2.0, 2.0)
SOFTMAX_BETA_STEPSIZE = 0.3
CYCLE_LENGTH = 1000
TRACE_STRIDE = 100          # mcmc_simulation_po stores traces every 100 iterations


def stage5_skills() -> tuple[SkillTemplate, ...]:
    return tuple(
        SkillTemplate(cpa_labels=LABELS[k], u=U_TRUE[k], beta=toy.BETA, epsilon=toy.EPSILON)
        for k in range(N_SKILLS)
    )


def assert_stage5_library() -> None:
    """Structural guards, including that ``beta`` is live for at least one skill."""
    skills = stage5_skills()
    if not precedence_from_u(skills[SKILL_A].u)[0, 1]:
        raise AssertionError("skill A must induce role 0 > role 1")
    if precedence_from_u(skills[SKILL_D].u).any():
        raise AssertionError("skill D must be an antichain")
    if not precedence_from_u(skills[SKILL_F].u)[0, 1]:
        raise AssertionError("skill F must induce role 0 > role 1")

    p_e = precedence_from_u(skills[SKILL_E].u)
    pairs = {(i, j) for i in range(4) for j in range(4) if p_e[i, j]}
    if pairs != {(0, 2), (0, 3)}:
        raise AssertionError(f"skill E must induce exactly {{0>2, 0>3}}, got {sorted(pairs)}")
    if frontier([0, 1, 2, 3], p_e) != (0, 1):
        raise AssertionError("skill E's initial frontier must be {0, 1}")
    if (remaining_successor_count(0, [0, 1, 2, 3], p_e),
            remaining_successor_count(1, [0, 1, 2, 3], p_e)) != (2, 0):
        raise AssertionError("skill E must have S(0)=2, S(1)=0")
    probs = bpop_step_probabilities([0, 1, 2, 3], skills[SKILL_E].u, toy.BETA, toy.EPSILON)
    if abs(probs[0] - probs[1]) < 1e-6:
        raise AssertionError("beta must be live for skill E")

    if not np.allclose(P_TRUE.sum(axis=1), 1.0) or np.any(np.diag(P_TRUE) != 0.0):
        raise AssertionError("P_TRUE must have unit rows and a zero diagonal")
    if np.any(P_TRUE[~np.eye(N_SKILLS, dtype=bool)] <= 0.0):
        raise AssertionError("every allowed transition must be positive")


# ---------------------------------------------------------------------------
# the adapter: oracle segments -> the verified sampler's input format
# ---------------------------------------------------------------------------


class IncompatibleSegment(ValueError):
    """A segment whose CPA multiset does not match the skill's role support."""


@dataclass(frozen=True)
class PODataset:
    """Exactly the three inputs ``mcmc_simulation_po`` expects."""
    items: list[int]
    choice_sets: list[list[int]]
    observed_orders: list[list[int]]
    skill_id: int
    n_roles: int

    def __len__(self) -> int:
        return len(self.observed_orders)


def build_po_dataset_for_skill(traces, segmentations, skill_id: int,
                               skill_template: SkillTemplate) -> PODataset:
    """Collect every segment currently assigned to ``skill_id`` as PO-sampler input.

    For this static model each accepted instance is one complete execution of the
    skill's role set, so every choice set is ``list(range(n_roles))``.

    Raises:
        IncompatibleSegment: if an assigned block's CPA multiset does not equal the
            skill's label set. Silently dropping such a block would quietly change
            the data the verified sampler sees.
    """
    labels = tuple(skill_template.cpa_labels)
    n_roles = len(labels)
    label_to_role = {label: index for index, label in enumerate(labels)}
    if len(label_to_role) != n_roles:
        raise ValueError(f"skill {skill_id} has repeated CPA labels: {labels}")

    items = list(range(n_roles))
    choice_sets: list[list[int]] = []
    observed_orders: list[list[int]] = []

    for trace, segmentation in zip(traces, segmentations):
        observations = trace.observations if hasattr(trace, "observations") else trace
        for segment in segmentation.segments:
            if segment.skill != skill_id:
                continue                                    # never mix skills
            block = tuple(observations[segment.start:segment.end])
            if Counter(block) != Counter(labels):
                raise IncompatibleSegment(
                    f"segment {block} assigned to skill {SKILL_NAMES[skill_id]} "
                    f"does not match its support {labels}")
            observed_orders.append([label_to_role[v] for v in block])
            choice_sets.append(list(items))                 # complete execution

    return PODataset(items=items, choice_sets=choice_sets,
                     observed_orders=observed_orders, skill_id=skill_id, n_roles=n_roles)


# ---------------------------------------------------------------------------
# synthetic corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticTrace:
    observations: tuple[int, ...]
    true_segmentation: Segmentation
    true_skill_path: tuple[int, ...]
    true_role_sequences: tuple[tuple[int, ...], ...]
    split: str
    n_tilings: int

    @property
    def length(self) -> int:
        return len(self.observations)

    @property
    def true_cuts(self) -> frozenset[int]:
        return frozenset(g.end for g in self.true_segmentation.segments[:-1])


def _compatible(observations, start, end, template) -> bool:
    return (end - start == template.n_roles
            and Counter(observations[start:end]) == Counter(template.cpa_labels))


def count_support_compatible_tilings(observations, skills, cap: int = 1_000_000) -> int:
    """Legal tilings, by memoised DP over ``(position, previous skill)``."""
    observations = tuple(observations)
    total = len(observations)
    memo: dict[tuple[int, int], int] = {}

    def ways(position: int, previous: int) -> int:
        if position == total:
            return 1
        key = (position, previous)
        if key in memo:
            return memo[key]
        found = 0
        for k, template in enumerate(skills):
            if k == previous:
                continue
            end = position + template.n_roles
            if end <= total and _compatible(observations, position, end, template):
                found += ways(end, k)
                if found >= cap:
                    found = cap
                    break
        memo[key] = found
        return found

    return ways(0, -1)


def sample_random_legal_segmentation(observations, skills, rng) -> Segmentation:
    """Uniform draw over legal tilings. Reads ONLY the CPA sequence and supports.

    No true boundaries, no true path, no ``U``, no ``P`` — this is what keeps Stage 5B
    free of oracle leakage.
    """
    observations = tuple(observations)
    total = len(observations)
    memo: dict[tuple[int, int], int] = {}

    def ways(position: int, previous: int) -> int:
        if position == total:
            return 1
        key = (position, previous)
        if key in memo:
            return memo[key]
        found = 0
        for k, template in enumerate(skills):
            if k == previous:
                continue
            end = position + template.n_roles
            if end <= total and _compatible(observations, position, end, template):
                found += ways(end, k)
        memo[key] = found
        return found

    if ways(0, -1) == 0:
        raise ValueError(f"no legal segmentation for {observations}")

    segments: list[Segment] = []
    position, previous = 0, -1
    while position < total:
        options, weights = [], []
        for k, template in enumerate(skills):
            if k == previous:
                continue
            end = position + template.n_roles
            if end <= total and _compatible(observations, position, end, template):
                completions = ways(end, k)
                if completions:
                    options.append((k, end))
                    weights.append(float(completions))
        weights = np.array(weights)
        k, end = options[int(rng.choice(len(options), p=weights / weights.sum()))]
        segments.append(Segment(position, end, k))
        position, previous = end, k
    return Segmentation(segments=tuple(segments))


def generate_corpus(n_train: int, n_test: int, seed: int, min_instances: int = 20):
    """Generate traces from ``P_TRUE`` and the true skill library."""
    assert_stage5_library()
    skills = stage5_skills()
    rng = np.random.default_rng(seed)

    def draw(split):
        n_segments = int(rng.choice(SEGMENTS_PER_TRACE))
        path = [int(rng.choice(N_SKILLS, p=PI_TRUE))]
        for _ in range(n_segments - 1):
            path.append(int(rng.choice(N_SKILLS, p=P_TRUE[path[-1]])))
        observations, segments, roles_all = [], [], []
        for skill in path:
            template = skills[skill]
            roles = sample_bpop_sequence(rng, template.u, template.beta, template.epsilon)
            start = len(observations)
            observations.extend(template.cpa_labels[r] for r in roles)
            segments.append(Segment(start, len(observations), skill))
            roles_all.append(roles)
        observations = tuple(observations)
        return SyntheticTrace(observations, Segmentation(segments=tuple(segments)),
                              tuple(path), tuple(roles_all), split,
                              count_support_compatible_tilings(observations, skills))

    train = [draw("train") for _ in range(n_train)]
    extra = 0
    while True:
        counts = Counter(z for t in train for z in t.true_skill_path)
        if all(counts.get(k, 0) >= min_instances for k in range(N_SKILLS)):
            break
        train.append(draw("train"))
        extra += 1
        if extra > 5000:
            raise RuntimeError("coverage unreachable")
    test = [draw("test") for _ in range(n_test)]
    return train, test, {"n_train": len(train), "n_train_extra": extra, "n_test": len(test),
                         "instance_counts": {SKILL_NAMES[k]: int(Counter(
                             z for t in train for z in t.true_skill_path).get(k, 0))
                             for k in range(N_SKILLS)}}


# ---------------------------------------------------------------------------
# Stage 5A
# ---------------------------------------------------------------------------


def run_skill_po_mcmc(dataset: PODataset, num_iterations: int, seed: int,
                      burn_in_fraction: float = 0.3, verbose: bool = False) -> dict:
    """Run the VERIFIED ``mcmc_simulation_po`` for one skill.

    Every update rule (``U`` row proposal from ``Sigma_rho``, the multiplicative ``rho``
    proposal with its ``-log(delta)`` correction, the log-``beta`` random walk with its
    Jacobian, and the dimension-proportional schedule) comes from the vendored sampler.
    Nothing here re-implements them.
    """
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink if not verbose else io.StringIO()):
        result = mcmc_simulation_po(
            num_iterations=num_iterations,
            items=dataset.items,
            choice_sets=dataset.choice_sets,
            observed_orders=dataset.observed_orders,
            dr=DR,
            noise_option=NOISE_OPTION,
            rho_prior=RHO_PRIOR,
            noise_beta_prior=NOISE_BETA_PRIOR,
            K_prior=LATENT_DIM,
            fixed_K=LATENT_DIM,            # disables the reversible-jump dimension update
            random_seed=seed,
            cycle_length=CYCLE_LENGTH,
            epsilon=toy.EPSILON,           # trembling hand stays fixed
            softmax_beta_prior=SOFTMAX_BETA_PRIOR,
            softmax_beta_stepsize=SOFTMAX_BETA_STEPSIZE,
        )

    n_stored = len(result["H_trace"])
    start = int(n_stored * burn_in_fraction)
    kept_h = [np.asarray(h, dtype=bool) for h in result["H_trace"][start:]]
    relation_probability = (np.mean(kept_h, axis=0) if kept_h
                            else np.zeros((dataset.n_roles, dataset.n_roles)))

    update_df = result["update_df"]
    acceptance_by_category = {}
    if len(update_df):
        for category, block in update_df.groupby("category"):
            acceptance_by_category[str(category)] = {
                "n": int(len(block)), "acceptance": float(block["accepted"].mean())}

    return {
        "relation_probability": relation_probability,
        "rho_trace": [float(v) for v in result["rho_trace"][start:]],
        "beta_trace": [float(v) for v in result["softmax_beta_trace"][start:]],
        "latent_dim_trace": [int(v) for v in result["K_trace"]],
        "latent_dim_final": int(result["K_final"]),
        "rho_final": float(result["rho_final"]),
        "beta_final": float(result["softmax_beta_final"]),
        "U_final": np.asarray(result["U_final"]),
        "H_final": np.asarray(result["H_final"], dtype=int),
        "overall_acceptance_rate": float(result["overall_acceptance_rate"]),
        "acceptance_by_category": acceptance_by_category,
        "log_likelihood_currents": [float(v) for v in result["log_likelihood_currents"]],
        "n_stored": n_stored,
        "n_kept": len(kept_h),
        "n_observations": len(dataset),
    }


def infer_transitions(paths, eta: float = 1.0, seed: int = 0) -> dict:
    """The already-validated Dirichlet row Gibbs update. Separate from the PO MCMC."""
    counts = transition_counts(paths, N_SKILLS)
    params = dirichlet_posterior_params(counts, N_SKILLS, eta)
    mean = posterior_mean_transition_matrix(counts, N_SKILLS, eta)
    rng = np.random.default_rng(seed)

    rows = {}
    for h, (allowed, alpha) in params.items():
        draws = rng.dirichlet(alpha, size=20000)
        rows[SKILL_NAMES[h]] = {
            "allowed_next": [SKILL_NAMES[k] for k in allowed],
            "counts": [int(counts[h, k]) for k in allowed],
            "alpha": alpha.tolist(),
            "posterior_mean": [float(mean[h, k]) for k in allowed],
            "true": [float(P_TRUE[h, k]) for k in allowed],
            "ci95": [[float(lo), float(hi)] for lo, hi in
                     zip(np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0))],
        }

    off = ~np.eye(N_SKILLS, dtype=bool)
    row_kl = []
    for h in range(N_SKILLS):
        allowed = [k for k in range(N_SKILLS) if k != h]
        row_kl.append(float(sum(P_TRUE[h, k] * np.log(P_TRUE[h, k] / mean[h, k])
                                for k in allowed if P_TRUE[h, k] > 0)))
    return {
        "counts": counts.astype(int).tolist(),
        "rows": rows,
        "posterior_mean": mean.tolist(),
        "mae": float(np.abs(mean[off] - P_TRUE[off]).mean()),
        "max_abs_error": float(np.abs(mean[off] - P_TRUE[off]).max()),
        "row_kl": row_kl,
        "mean_row_kl": float(np.mean(row_kl)),
    }


# ---------------------------------------------------------------------------
# Stage 5B — oracle U and P, infer the segmentations
# ---------------------------------------------------------------------------


class Stage5BTarget:
    """``log p(x, S | U, P)`` for one trace, with the skill library held fixed.

        (L-1) log delta_B + (T-L) log(1-delta_B)
      + log pi[z_1] + sum_l log P[z_{l-1}, z_l]
      + sum_l log p_BPOP(block_l | U[z_l], beta, epsilon)

    Self-transitions are forbidden and arrive for free: ``P`` has a zero diagonal, so
    ``log P[h,h] = -inf``. Incompatible blocks are ``-inf`` too. Emission terms come
    from precomputed per-skill permutation tables (``U`` is fixed here), cached per
    ``(start, end, skill)``, which is what makes a 20k-sweep corpus run affordable.
    """

    def __init__(self, observations, skills, evaluators, delta_b, log_pi, log_transition):
        self.x = tuple(int(v) for v in observations)
        self.evaluators = evaluators
        self.delta_b = delta_b
        self.log_pi = log_pi
        self.log_transition = log_transition
        self.tables = [evaluators[k].log_table(skills[k].u) for k in range(len(skills))]
        self._cache: dict[tuple[int, int, int], float] = {}

    def _block(self, start, end, skill):
        key = (start, end, skill)
        value = self._cache.get(key)
        if value is None:
            index = self.evaluators[skill].index_of_block(self.x[start:end])
            value = -np.inf if index is None else float(self.tables[skill][index])
            self._cache[key] = value
        return value

    def __call__(self, segmentation) -> float:
        from hpop.mcmc_original.targets import log_boundary_prior, log_path_prior
        segments = segmentation.segments
        path = [g.skill for g in segments]
        total = log_boundary_prior(len(self.x), len(path), self.delta_b)
        total += log_path_prior(path, self.log_pi, self.log_transition)
        if total == -np.inf:
            return -np.inf
        for g in segments:
            value = self._block(g.start, g.end, g.skill)
            if value == -np.inf:
                return -np.inf
            total += value
        return total


def adjusted_rand_index(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    n = a.size
    ua, ub = np.unique(a), np.unique(b)
    table = np.array([[np.sum((a == x) & (b == y)) for y in ub] for x in ua], dtype=float)
    c2 = lambda v: v * (v - 1.0) / 2.0
    sij, si, sj = c2(table).sum(), c2(table.sum(1)).sum(), c2(table.sum(0)).sum()
    total = c2(float(n))
    expected, maximum = si * sj / total, 0.5 * (si + sj)
    return 1.0 if maximum == expected else float((sij - expected) / (maximum - expected))


def run_stage5b(test_traces, skills, n_chains, n_sweeps, burn_in, thin, seed,
                delta_b=None, verbose=False) -> dict:
    """Infer segmentations on the test split with U and P fixed at the truth.

    Uses the Stage-4 ``LocalMoveKernel`` **unchanged**, including its Hastings
    correction, and initialises from ``sample_random_legal_segmentation`` — never from
    the true segmentation.
    """
    from hpop.mcmc_original.proposals import LocalMoveKernel, MoveType, mh_local_step
    from hpop.mcmc_original.targets import SkillEvaluator
    from hpop.mcmc_original.transitions import log_transition_matrix

    delta_b = toy.S4_DELTA_B if delta_b is None else delta_b
    evaluators = [SkillEvaluator(s) for s in skills]
    log_pi = np.log(PI_TRUE)
    log_p = log_transition_matrix(P_TRUE)

    targets = [Stage5BTarget(t.observations, skills, evaluators, delta_b, log_pi, log_p)
               for t in test_traces]

    class _Cached(LocalMoveKernel):
        def __post_init__(self):
            super().__post_init__()
            self._dist: dict = {}

        def proposal_distribution(self, segmentation):
            got = self._dist.get(segmentation)
            if got is None:
                got = super().proposal_distribution(segmentation)
                self._dist[segmentation] = got
            return got

    kernels = [_Cached(x=t.observations, skills=skills) for t in test_traces]

    chains = []
    for chain in range(n_chains):
        rng = np.random.default_rng(seed + 7000 + chain)
        states = [sample_random_legal_segmentation(t.observations, skills, rng)
                  for t in test_traces]
        values = [targets[i](states[i]) for i in range(len(test_traces))]
        if any(v == -np.inf for v in values):
            raise RuntimeError("random initialisation had zero posterior probability")
        saved, proposed, accepted = [], {m: 0 for m in MoveType.ALL}, {m: 0 for m in MoveType.ALL}
        for sweep in range(n_sweeps):
            for i in range(len(test_traces)):
                step = mh_local_step(states[i], targets[i], kernels[i], rng,
                                     current_log_target=values[i])
                states[i], values[i] = step["state"], step["log_target"]
                proposed[step["move"]] += int(step["proposed"])
                accepted[step["move"]] += int(step["accepted"])
            if sweep >= burn_in and (sweep - burn_in) % thin == 0:
                saved.append(list(states))
        chains.append({"saved": saved, "proposed_by_move": proposed,
                       "accepted_by_move": accepted})

    # pooled posterior summaries
    per_trace, pred_cuts, true_cuts = [], set(), set()
    true_labels, mode_labels = [], []
    for i, trace in enumerate(test_traces):
        pooled = [s[i] for c in chains for s in c["saved"]]
        n = len(pooled)
        cut_counts, label_counts = {}, np.zeros((trace.length, len(skills)))
        for segmentation in pooled:
            for g in segmentation.segments[:-1]:
                cut_counts[g.end] = cut_counts.get(g.end, 0) + 1
            for g in segmentation.segments:
                label_counts[g.start:g.end, g.skill] += 1
        boundary = {t: c / n for t, c in sorted(cut_counts.items())}
        predicted = sorted(t for t, p in boundary.items() if p >= 0.5)
        truth = sorted(trace.true_cuts)
        true_cuts |= {(i, t) for t in truth}
        pred_cuts |= {(i, t) for t in predicted}
        labels = np.empty(trace.length, dtype=int)
        for g in trace.true_segmentation.segments:
            labels[g.start:g.end] = g.skill
        true_labels.append(labels)
        mode_labels.append((label_counts / n).argmax(axis=1))
        per_trace.append({"index": i, "length": trace.length, "n_tilings": trace.n_tilings,
                          "true_cuts": truth, "predicted_cuts": predicted,
                          "boundary_posterior": boundary,
                          "mode_labels": (label_counts / n).argmax(axis=1).tolist(),
                          "true_labels": labels.tolist(),
                          "mean_prob_true_cuts": float(np.mean([boundary.get(t, 0.0) for t in truth]))
                          if truth else float("nan")})

    from hpop.mcmc_original.diagnostics import prf1
    truth_flat, pred_flat = np.concatenate(true_labels), np.concatenate(mode_labels)
    proposed = {m: sum(c["proposed_by_move"][m] for c in chains) for m in MoveType.ALL}
    accepted = {m: sum(c["accepted_by_move"][m] for c in chains) for m in MoveType.ALL}
    return {
        "per_trace": per_trace,
        "boundary": prf1(pred_cuts, true_cuts),
        "occurrence_accuracy": float((truth_flat == pred_flat).mean()),
        "adjusted_rand_index": adjusted_rand_index(truth_flat, pred_flat),
        "proposed_by_move": proposed, "accepted_by_move": accepted,
        "acceptance_by_move": {m: (accepted[m] / proposed[m] if proposed[m] else float("nan"))
                               for m in MoveType.ALL},
        "n_chains": n_chains, "n_sweeps": n_sweeps, "burn_in": burn_in, "thin": thin,
    }
