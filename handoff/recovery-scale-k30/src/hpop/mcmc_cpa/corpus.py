"""Corpus generation for the K ladder: traces emitted from per-skill CPA supports.

Section 8 of the ladder preregistration. Given a sealed truth for one rung -- the nested
library's `U` and role maps, plus that rung's own `pi` and `P` -- this draws the training
and held-out traces and then checks the predeclared coverage bands, resampling on failure.

## The emission path was already correct

`matched_synthetic_generator.generate_trace` already maps role to CPA on the way out:

    ell = truth.role_maps[skill]
    cpa.extend(int(ell[role]) for role in roles)

so the *generator* has always consumed role maps. What it lacked was a truth constructor
that would produce non-identity maps -- `sample_prior_truth` builds the identity and
`validate_truth` rejects anything else. The `"requires identity role maps"` assertion in
that module constrains the **scorer**, which `hpop.mcmc_cpa.block_tables` now handles.

This module therefore reuses the sealed primitives unchanged --
`sample_recurrent_rfs_sequence` for the emission, `sample_segmentation_widths` for the
boundaries, `sample_initial_skill` / `sample_next_skill` for the path -- and supplies only
the wiring and the coverage gate. `sample_recurrent_rfs_sequence` reads `u` of shape
`(m, d)` alone, so it is already indifferent to `K` and to `A`.

## What the coverage bands are for

A skill that never appears, or appears with three of its ten roles unused, is not evidence
about library size -- it is a corpus accident. The bands make each rung a *controlled*
recovery problem, and they are functions of the generated latent corpus alone: no
likelihood, no posterior, no recovery metric. A corpus that cannot meet them after the
registered number of attempts is reported as generation-failed. **The thresholds are never
lowered to admit a corpus.**
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from hpop.mcmc_original.matched_segmentation_prior import (sample_segmentation_widths,
                                                           width_sampling_tables)
from hpop.mcmc_original.matched_synthetic_generator import (block_rng, component_rng,
                                                            sample_initial_skill,
                                                            sample_next_skill)
from hpop.mcmc_original.recurrent_rfs import (RecurrentRFSParameters,
                                              sample_recurrent_rfs_sequence)

from .role_maps import RoleMaps

__all__ = ["LadderTrace", "LadderCorpus", "CoverageBands", "generate_ladder_corpus",
           "draw_pi_p", "band_acceptance_rate", "role_exposure", "stationary_of",
           "stationary_ok"]


# ------------------------------------------------------------------ coverage bands
class CoverageBands:
    """Section 8, as data. Every field is a property of the latent corpus alone."""

    __slots__ = ("train_min_instances", "train_min_occurrences", "train_max_occurrences",
                 "train_min_per_role", "test_min_instances", "test_min_occurrences")

    def __init__(self, train_min_instances=30, train_min_occurrences=240,
                 train_max_occurrences=720, train_min_per_role=5,
                 test_min_instances=8, test_min_occurrences=60):
        self.train_min_instances = int(train_min_instances)
        self.train_min_occurrences = int(train_min_occurrences)
        self.train_max_occurrences = int(train_max_occurrences)
        self.train_min_per_role = int(train_min_per_role)
        self.test_min_instances = int(test_min_instances)
        self.test_min_occurrences = int(test_min_occurrences)

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__slots__}


class LadderTrace:
    """One generated trace, with the latent record the coverage check reads."""

    __slots__ = ("index", "split", "cpa", "widths", "labels", "role_blocks")

    def __init__(self, index, split, cpa, widths, labels, role_blocks):
        self.index, self.split = int(index), str(split)
        self.cpa = tuple(int(v) for v in cpa)
        self.widths = tuple(int(v) for v in widths)
        self.labels = tuple(int(v) for v in labels)
        self.role_blocks = tuple(tuple(int(v) for v in b) for b in role_blocks)

    @property
    def length(self) -> int:
        return len(self.cpa)

    def as_dict(self) -> dict:
        return {"index": self.index, "split": self.split, "cpa": list(self.cpa),
                "widths": list(self.widths), "labels": list(self.labels),
                "role_blocks": [list(b) for b in self.role_blocks]}


class LadderCorpus:
    """Train and held-out traces for one (K, replicate), plus the sealed truth digest."""

    __slots__ = ("k", "replicate", "train", "heldout", "role_maps", "pi", "transition",
                 "coverage", "seeds", "attempts")

    def __init__(self, k, replicate, train, heldout, role_maps, pi, transition,
                 coverage, seeds, attempts):
        self.k, self.replicate = int(k), int(replicate)
        self.train, self.heldout = tuple(train), tuple(heldout)
        self.role_maps = role_maps
        self.pi = np.asarray(pi, dtype=float)
        self.transition = np.asarray(transition, dtype=float)
        self.coverage = coverage
        self.seeds = dict(seeds)
        self.attempts = list(attempts)

    def traces(self, split: str) -> tuple:
        return tuple(t.cpa for t in (self.train if split == "train" else self.heldout))

    def observed_digest(self) -> str:
        """Hash of the OBSERVED data only -- no latent state, so it is safe to publish."""
        payload = json.dumps(
            {"train": [list(t.cpa) for t in self.train],
             "heldout": [list(t.cpa) for t in self.heldout]},
            separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def truth_digest(self) -> str:
        """Hash of the sealed latent state. Publish the hash, never the values."""
        payload = json.dumps(
            {"labels": [list(t.labels) for t in self.train + self.heldout],
             "widths": [list(t.widths) for t in self.train + self.heldout],
             "role_maps": self.role_maps.forward.tolist(),
             "pi": self.pi.tolist(), "transition": self.transition.tolist()},
            separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def manifest(self) -> dict:
        """Everything that may be exposed before terminal unsealing: hashes and verdicts."""
        return {
            "K": self.k, "replicate": self.replicate,
            "n_train": len(self.train), "n_heldout": len(self.heldout),
            "trace_length": self.train[0].length if self.train else 0,
            "train_occurrences": sum(t.length for t in self.train),
            "heldout_occurrences": sum(t.length for t in self.heldout),
            "observed_sha256": self.observed_digest(),
            "truth_sha256": self.truth_digest(),
            "role_maps_digest": self.role_maps.digest(),
            "coverage": self.coverage,
            "coverage_bands": CoverageBands().as_dict(),
            "seeds": self.seeds,
            "generation_attempts": self.attempts,
        }


# --------------------------------------------------------------------- the emission
def _emit_trace(master_seed: int, split: str, index: int, length: int,
                u_by_skill, role_maps: RoleMaps, pi, transition,
                params: RecurrentRFSParameters, tables, delta_b, min_width, max_width):
    """One trace: skill path, widths, recurrent roles per block, roles mapped to CPAs."""
    widths = sample_segmentation_widths(
        component_rng(master_seed, split, index, 0), length, delta_b,
        min_width, max_width, tables)

    label_rng = component_rng(master_seed, split, index, 1)
    labels = [sample_initial_skill(label_rng, pi)]
    for _ in range(len(widths) - 1):
        labels.append(sample_next_skill(label_rng, transition, labels[-1]))

    role_blocks, cpa = [], []
    for block, (width, skill) in enumerate(zip(widths, labels)):
        roles = sample_recurrent_rfs_sequence(
            block_rng(master_seed, split, index, block), width,
            u_by_skill[skill], params)
        roles = tuple(int(v) for v in roles)
        role_blocks.append(roles)
        cpa.extend(int(role_maps.forward[skill, r]) for r in roles)

    if len(cpa) != length:
        raise RuntimeError(f"emitted {len(cpa)} symbols, expected {length}")
    return LadderTrace(index, split, cpa, widths, labels, role_blocks)


def _coverage(traces, n_skills: int, n_roles: int, role_maps: RoleMaps) -> dict:
    """Per-skill instance, occurrence and per-role counts. Latent only, no likelihood."""
    instances = np.zeros(n_skills, dtype=np.int64)
    occurrences = np.zeros(n_skills, dtype=np.int64)
    per_role = np.zeros((n_skills, n_roles), dtype=np.int64)
    for trace in traces:
        for width, skill, roles in zip(trace.widths, trace.labels, trace.role_blocks):
            instances[skill] += 1
            occurrences[skill] += int(width)
            for r in roles:
                per_role[skill, r] += 1
    return {"instances": instances, "occurrences": occurrences, "per_role": per_role}


def _bands_met(train_cov, test_cov, bands: CoverageBands) -> tuple:
    reasons = []
    for k, (inst, occ) in enumerate(zip(train_cov["instances"],
                                        train_cov["occurrences"])):
        if inst < bands.train_min_instances:
            reasons.append(f"train skill {k}: {inst} instances < {bands.train_min_instances}")
        if not bands.train_min_occurrences <= occ <= bands.train_max_occurrences:
            reasons.append(
                f"train skill {k}: {occ} occurrences outside "
                f"[{bands.train_min_occurrences}, {bands.train_max_occurrences}]")
        thin = np.flatnonzero(train_cov["per_role"][k] < bands.train_min_per_role)
        if thin.size:
            reasons.append(f"train skill {k}: roles {thin.tolist()} seen fewer than "
                           f"{bands.train_min_per_role} times")
    for k, (inst, occ) in enumerate(zip(test_cov["instances"], test_cov["occurrences"])):
        if inst < bands.test_min_instances:
            reasons.append(f"heldout skill {k}: {inst} instances < {bands.test_min_instances}")
        if occ < bands.test_min_occurrences:
            reasons.append(f"heldout skill {k}: {occ} occurrences < "
                           f"{bands.test_min_occurrences}")
    return (not reasons), reasons


# ------------------------------------------------------------------------ pi and P
def band_acceptance_rate(n_skills: int, trials: int = 2000, seed: int = 0) -> dict:
    """How often a flat-prior draw satisfies criteria 11 and 12. Measured, not assumed.

    Reported by `draw_pi_p` when it gives up, because "rejection sampling failed" is not
    actionable and "the event has probability 5e-4 at this K, so the registered attempt cap
    of 100 cannot reach it" is.
    """
    from hpop.mcmc_original.transitions import sample_transition_matrix
    K = int(n_skills)
    low, high = 0.5 / K, 1.5 / K
    rng = np.random.default_rng(int(seed))
    pi_ok = stat_ok = both = 0
    for _ in range(int(trials)):
        pi = rng.dirichlet(np.ones(K))
        p = sample_transition_matrix(np.zeros((K, K)), K, rng, 1.0)
        a = bool(np.all((pi >= low) & (pi <= high)))
        ok, _reasons, stationary = stationary_ok(pi, p, K)
        b = bool(stationary is not None
                 and np.all((stationary >= low) & (stationary <= high)))
        pi_ok += a
        stat_ok += b
        both += a and b
    return {"n_skills": K, "trials": int(trials),
            "pi_band_rate": pi_ok / trials, "stationary_band_rate": stat_ok / trials,
            "joint_rate": both / trials,
            "expected_attempts": (trials / both) if both else float("inf")}


def draw_pi_p(n_skills: int, seed: int, eta: float = 1.0,
              max_attempts: int = 100) -> tuple:
    """`P` from the registered flat-Dirichlet row model, with `pi` set to `nu(P)`.

    ## Why not a free `pi`

    Each of the `5K` traces draws its first segment from `pi`, so skill `k` receives about
    `5K * pi_k` instances from first segments alone. That term grows **linearly in K**,
    while a balanced skill's total instance count is about `5 * E[L] ~ 71` and does not
    grow with K at all. Measured over 200 draws per rung, an unconstrained flat-Dirichlet
    `pi` leaves a max/min ratio of expected per-skill instances of

        K = 3   2.32       K = 20   2.32
        K = 5   2.65       K = 30   2.09
        K = 10  2.48

    at every rung. A ladder whose whole point is equal evidence per skill cannot carry a
    two-to-threefold imbalance that is itself a function of the draw.

    An earlier note here argued that `pi` "only affects one segment in thirteen" and could
    therefore be left unconstrained. That was wrong: it counted segments within one trace
    and ignored that every one of the `5K` traces contributes a fresh first segment.

    ## What this is, stated honestly

    Setting `pi = nu(P)` makes the skill chain stationary from the first segment, so every
    segment index has marginal `nu` with no reliance on mixing. It is a **ladder-specific
    controlled truth design in the same likelihood family as the confirmatory experiment**,
    not an identical prior-predictive draw: the true `(pi, P)` is no longer an unconditional
    draw from the independent Dirichlet prior the sampler assumes. Given the truth the data
    likelihood is exactly right, so this is not the corpus-rejection mismatch -- but no
    strict prior-predictive calibration claim may be made from these runs.
    """
    from hpop.mcmc_original.transitions import sample_transition_matrix
    K = int(n_skills)
    low, high = 0.5 / K, 1.5 / K
    for attempt in range(int(max_attempts)):
        rng = np.random.default_rng(int(seed) + 1_000_000 * attempt)
        transition = sample_transition_matrix(np.zeros((K, K)), K, rng, float(eta))
        stationary = stationary_of(transition, K)
        if stationary is None:
            continue
        if np.all((stationary >= low) & (stationary <= high)):
            return stationary.copy(), transition          # pi = nu(P)
    rate = band_acceptance_rate(K, trials=500, seed=int(seed))
    raise RuntimeError(
        f"K={K}: no P whose stationary law satisfies the registered balance band in "
        f"{max_attempts} attempts (measured rate {rate['stationary_band_rate']:.3f}). "
        f"Report and stop; do not widen the band.")


def stationary_of(transition, n_skills: int):
    """The unique stationary law of `P`, or None when it is not unique."""
    p = np.asarray(transition, dtype=float)
    values, vectors = np.linalg.eig(p.T)
    unit = np.flatnonzero(np.isclose(values.real, 1.0, atol=1e-9)
                          & (np.abs(values.imag) < 1e-9))
    if unit.size != 1:
        return None
    stationary = np.abs(vectors[:, unit[0]].real)
    return stationary / stationary.sum()


def stationary_ok(pi, transition, n_skills: int) -> tuple:
    """Section 7, criteria 8-12: a unique stationary law, balanced within [0.5/K, 1.5/K]."""
    pi = np.asarray(pi, dtype=float)
    p = np.asarray(transition, dtype=float)
    reasons = []
    if not np.isclose(pi.sum(), 1.0) or np.any(pi <= 0):
        reasons.append("pi is not a positive simplex vector")
    if not np.array_equal(np.diag(p), np.zeros(n_skills)):
        reasons.append("P has a nonzero diagonal")
    values, vectors = np.linalg.eig(p.T)
    unit = np.flatnonzero(np.isclose(values.real, 1.0, atol=1e-9)
                          & (np.abs(values.imag) < 1e-9))
    if unit.size != 1:
        reasons.append(f"stationary law is not unique ({unit.size} unit eigenvalues)")
        return (not reasons), reasons, None
    stationary = np.abs(vectors[:, unit[0]].real)
    stationary = stationary / stationary.sum()
    low, high = 0.5 / n_skills, 1.5 / n_skills
    if np.any(stationary < low) or np.any(stationary > high):
        reasons.append(f"stationary occupancy outside [{low:.4f}, {high:.4f}]")
    if np.any(pi < low) or np.any(pi > high):
        reasons.append(f"pi outside [{low:.4f}, {high:.4f}]")
    return (not reasons), reasons, stationary


def role_exposure(u_by_skill, params, widths=(3, 12), samples: int = 400,
                      seed: int = 0) -> dict:
    """Which roles are rarely emitted, measured over WHOLE segments.

    ## A withdrawn diagnostic, and why

    An earlier version measured the **first-step** emission probability and reported roles
    sitting at the `epsilon / m` floor as "structurally starved". That measure was wrong,
    and provably so: under the registered mixture a role with any predecessor has
    feasibility `F = 0` at `q = 0`, so its first-step probability is *exactly* `epsilon/m`
    for every such role in every draw. It says nothing about whether the role is emitted
    later, once its predecessors have fired -- which is the whole point of a recurrent
    model. Any admissibility rule built on it is unsatisfiable whenever the partial order
    has a single relation, which criterion 3 requires.

    This replaces it with the quantity that actually matters: the expected number of times
    a role appears across segments of the registered widths, obtained by running the
    generative emission forward rather than by inspecting one step of it.
    """
    from hpop.mcmc_original.recurrent_rfs import sample_recurrent_rfs_sequence

    u_by_skill = np.asarray(u_by_skill, dtype=float)
    n_skills, m = u_by_skill.shape[0], u_by_skill.shape[1]
    low, high = int(widths[0]), int(widths[1])
    rng = np.random.default_rng(int(seed))
    per_skill = np.zeros((n_skills, m))
    stderr = np.zeros((n_skills, m))
    at_least_once = np.zeros((n_skills, m))
    for k in range(n_skills):
        draws = np.zeros((int(samples), m))
        for s_i in range(int(samples)):
            width = int(rng.integers(low, high + 1))
            roles = sample_recurrent_rfs_sequence(rng, width, u_by_skill[k], params)
            for r in roles:
                draws[s_i, int(r)] += 1
        per_skill[k] = draws.mean(axis=0)
        stderr[k] = draws.std(axis=0, ddof=1) / np.sqrt(samples)
        at_least_once[k] = (draws > 0).mean(axis=0)
    return {
        "n_skills": int(n_skills), "n_roles": int(m), "segments_per_skill": int(samples),
        "widths_sampled_uniformly_over": [low, high],
        "q0_reset_each_segment": True,
        "expected_count_per_segment": per_skill.tolist(),
        "expected_count_mc_stderr": stderr.tolist(),
        "probability_at_least_once_per_segment": at_least_once.tolist(),
        "probability_mc_stderr": (np.sqrt(at_least_once * (1 - at_least_once)
                                          / samples)).tolist(),
        "per_skill_min_expected_count": per_skill.min(axis=1).tolist(),
        "global_min_expected_count": float(per_skill.min()),
        "note": "two DIFFERENT quantities: expected count per segment, and probability of "
                "at least one occurrence in a segment. Neither is a bound on recovery -- "
                "absence is itself likelihood information, transitivity constrains "
                "unobserved edges, and the prior contributes. Report as a "
                "data-supported observability diagnostic, never as a recovery ceiling. "
                "First-step probability is NOT a valid proxy for either.",
    }


# ------------------------------------------------------------------------- the loop
def generate_ladder_corpus(library, k: int, replicate: int, trace_length: int = 96,
                           params: RecurrentRFSParameters | None = None,
                           delta_b: float = 0.15, min_width: int = 3, max_width: int = 12,
                           bands: CoverageBands | None = None) -> LadderCorpus:
    """One rung's corpus: `N_train = 5 x K`, `N_test = 2 x K`, **generated exactly once**.

    ## There is no acceptance loop, and there must not be one

    An earlier version resampled corpora until per-skill instance, occurrence and per-role
    counts fell inside registered bands. That is a selection bias, not an inefficiency.
    Rejecting until an event `A` holds samples

        p(D | theta, A) = p(D | theta) . 1{A} / P_theta(A)

    while the sampler scores the ordinary `p(D | theta)` with no `-log P_theta(A)` term.
    `P_theta(A)` depends on `U` and the recurrent dynamics -- exactly what the study is
    trying to recover -- so the omission biases every rung differently.

    Every coverage quantity is therefore **measured and reported, never enforced**. The
    same applies to searching for a truth seed that yields agreeable realised counts: that
    is the same bias reached by a different route.
    """
    from hpop.mcmc_original.recurrent_scalar_posterior import TRUE_VALUES

    k, replicate = int(k), int(replicate)
    bands = CoverageBands() if bands is None else bands
    if params is None:
        params = RecurrentRFSParameters(
            beta=float(TRUE_VALUES["beta"]), epsilon=0.02,
            shared_omega=float(TRUE_VALUES["omega"]),
            lambda_rep=float(TRUE_VALUES["lambda_rep"]),
            lambda_back=float(TRUE_VALUES["lambda_back"]))

    u_by_skill, role_maps = library.prefix(k)
    seeds = {"pi_p": 6_510_000 + 100 * k + replicate,
             "train_corpus": 6_520_000 + 100 * k + replicate,
             "heldout_corpus": 6_530_000 + 100 * k + replicate}

    pi, transition = draw_pi_p(k, seeds["pi_p"])
    n_train, n_test = 5 * k, 2 * k
    tables = width_sampling_tables(trace_length, delta_b, min_width, max_width)

    train = [_emit_trace(seeds["train_corpus"], "train", i, trace_length, u_by_skill,
                         role_maps, pi, transition, params, tables,
                         delta_b, min_width, max_width) for i in range(n_train)]
    heldout = [_emit_trace(seeds["heldout_corpus"], "heldout", i, trace_length,
                           u_by_skill, role_maps, pi, transition, params, tables,
                           delta_b, min_width, max_width) for i in range(n_test)]

    train_cov = _coverage(train, k, library.n_roles, role_maps)
    test_cov = _coverage(heldout, k, library.n_roles, role_maps)
    met, unmet = _bands_met(train_cov, test_cov, bands)
    coverage = {
        "train_instances": train_cov["instances"].tolist(),
        "train_occurrences": train_cov["occurrences"].tolist(),
        "train_per_role_min": train_cov["per_role"].min(axis=1).tolist(),
        "train_roles_never_seen": int((train_cov["per_role"] == 0).sum()),
        "heldout_instances": test_cov["instances"].tolist(),
        "heldout_occurrences": test_cov["occurrences"].tolist(),
        "stationary": pi.tolist(),
        "reference_bands": bands.as_dict(),
        "bands_met_AS_A_DIAGNOSTIC_ONLY": bool(met),
        "unmet_conditions": unmet,
        "NOTE": "these are reported, not enforced; the corpus was generated exactly once "
                "and was never rejected on realised counts",
    }
    return LadderCorpus(k, replicate, train, heldout, role_maps, pi, transition,
                        coverage, seeds, [{"generated_once": True}])
