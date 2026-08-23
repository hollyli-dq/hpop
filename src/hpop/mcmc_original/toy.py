"""Toy skill definitions and corpus generation for the Stage 0-3 validation.

Everything here is a *fixed, known* model. The point of the toy is that the exact
posterior can be computed by enumeration, so any sampler can be checked against
ground truth rather than against plausibility.

The local skill model is the original latent partial-order model

    H_k = (M_k, ell_k, U_k, beta_k, epsilon)

where ``ell_k`` maps each role to one CPA label (``cpa_labels``), ``U_k`` induces
the strict partial order by coordinate-wise dominance, and the likelihood of one
complete execution is the Hour-2 BPOP frontier-softmax likelihood. There is no
composition multinomial, no duration model and no recurrence anywhere in this
package.

Two different templates are used for skill B and must never be confused:

    U_B_TOTAL      total order 0 > 1 > 2, used in Stages 0, 1, 2A and 2B
    U_B_ANTICHAIN  no order at all,      used in Stage 3 only

The antichain is what makes Stage 3B's segmentation exactly ambiguous under the
local likelihood, so that the transition context has something to resolve.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.static_bpop import bpop_log_likelihood, sample_bpop_sequence
from hpop.mcmc_original.types import SkillTemplate

__all__ = [
    "SKILL_A",
    "SKILL_B",
    "SKILL_C",
    "BETA",
    "EPSILON",
    "DELTA_B",
    "RHO_U",
    "LATENT_DIM",
    "LABELS_A",
    "LABELS_B",
    "LABELS_C",
    "U_A",
    "U_B_TOTAL",
    "U_B_ANTICHAIN",
    "U_C",
    "PRIMARY_TRACE",
    "skill_a",
    "skill_b_total",
    "skill_b_antichain",
    "skill_c",
    "stage012_skills",
    "stage3_skills",
    "uniform_log_pi",
    "map_cpa_block_to_roles",
    "segment_log_likelihood",
    "sample_skill_execution",
    "make_stage2a_corpus",
    "make_stage2b_corpus",
    "make_stage3c_corpus",
]

# skill identities
SKILL_A = 0
SKILL_B = 1
SKILL_C = 2

# fixed model constants
BETA = 1.5
EPSILON = 0.05
DELTA_B = 0.5
RHO_U = 0.25
LATENT_DIM = 2

# CPA label sets. Labels are unique within a skill, so the CPA -> role map is
# deterministic. A and B share labels {0,1}, which is exactly what creates the
# segmentation ambiguity; C's labels {3,4} are disjoint from both.
LABELS_A = (0, 1)
LABELS_B = (0, 1, 2)
LABELS_C = (3, 4)

U_A = np.array([[1.0, 1.0], [0.0, 0.0]])              # 0 > 1
U_B_TOTAL = np.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])   # 0 > 1 > 2
U_B_ANTICHAIN = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])  # pairwise incomparable
U_C = np.array([[1.0, 1.0], [0.0, 0.0]])              # 3 > 4 (roles 0 > 1)

# the ambiguous trace used in Stage 0 and Stage 3B
PRIMARY_TRACE = (0, 1, 2, 0, 1)


def skill_a() -> SkillTemplate:
    return SkillTemplate(cpa_labels=LABELS_A, u=U_A, beta=BETA, epsilon=EPSILON)


def skill_b_total() -> SkillTemplate:
    return SkillTemplate(cpa_labels=LABELS_B, u=U_B_TOTAL, beta=BETA, epsilon=EPSILON)


def skill_b_antichain() -> SkillTemplate:
    return SkillTemplate(
        cpa_labels=LABELS_B, u=U_B_ANTICHAIN, beta=BETA, epsilon=EPSILON
    )


def skill_c() -> SkillTemplate:
    return SkillTemplate(cpa_labels=LABELS_C, u=U_C, beta=BETA, epsilon=EPSILON)


def stage012_skills() -> tuple[SkillTemplate, ...]:
    """K = 2: skill A and the **total-order** skill B. Stages 0, 1, 2A, 2B."""
    return (skill_a(), skill_b_total())


def stage3_skills() -> tuple[SkillTemplate, ...]:
    """K = 3: skill A, the **antichain** skill B, and skill C. Stage 3 only."""
    return (skill_a(), skill_b_antichain(), skill_c())


def uniform_log_pi(n_skills: int) -> np.ndarray:
    """Uniform first-skill prior ``pi_k = 1/K``, in log space."""
    return np.full(n_skills, -np.log(n_skills))


class IncompatibleBlock(ValueError):
    """Raised when a block's CPA multiset does not match a skill's label set."""


def map_cpa_block_to_roles(block, cpa_labels) -> tuple[int, ...] | None:
    """Map an observed block of CPA labels to this skill's role indices.

    A block is *support-compatible* with a skill only when their CPA multisets are
    exactly equal, so a compatible block is a permutation of the skill's roles.

    Returns:
        The role-index sequence, or ``None`` when the block is not
        support-compatible (the caller turns that into a ``-inf`` log-likelihood).
    """
    block = tuple(int(v) for v in block)
    labels = tuple(int(v) for v in cpa_labels)
    if Counter(block) != Counter(labels):
        return None
    label_to_role = {lab: i for i, lab in enumerate(labels)}
    return tuple(label_to_role[v] for v in block)


def segment_log_likelihood(block, skill_template: SkillTemplate) -> float:
    """``log p_BPOP(role_sequence(block) | U, beta, eps)``, or -inf if incompatible."""
    roles = map_cpa_block_to_roles(block, skill_template.cpa_labels)
    if roles is None:
        return -np.inf
    return bpop_log_likelihood(
        roles, skill_template.u, skill_template.beta, skill_template.epsilon
    )


def sample_skill_execution(
    skill_template: SkillTemplate, rng: np.random.Generator
) -> tuple[int, ...]:
    """Draw one complete execution and return it as observed **CPA labels**."""
    roles = sample_bpop_sequence(
        rng, skill_template.u, skill_template.beta, skill_template.epsilon
    )
    return tuple(skill_template.cpa_labels[r] for r in roles)


def sample_role_execution(
    skill_template: SkillTemplate, rng: np.random.Generator
) -> tuple[int, ...]:
    """Draw one complete execution and return it as **role indices**."""
    return sample_bpop_sequence(
        rng, skill_template.u, skill_template.beta, skill_template.epsilon
    )


def make_stage2a_corpus(
    rng: np.random.Generator, n_a: int = 100, n_b: int = 200
) -> dict[int, list[tuple[int, ...]]]:
    """Independent complete executions with the skill identity known.

    Returns role sequences keyed by skill index — Stage 2A has no segmentation
    uncertainty at all.
    """
    a, b = skill_a(), skill_b_total()
    return {
        SKILL_A: [sample_role_execution(a, rng) for _ in range(n_a)],
        SKILL_B: [sample_role_execution(b, rng) for _ in range(n_b)],
    }


def _concat_trace(
    first: SkillTemplate,
    first_id: int,
    second: SkillTemplate,
    second_id: int,
    rng: np.random.Generator,
) -> dict:
    left = sample_skill_execution(first, rng)
    right = sample_skill_execution(second, rng)
    return {
        "x": left + right,
        "true_cut": len(left),
        "true_path": (first_id, second_id),
    }


def make_stage2b_corpus(
    rng: np.random.Generator, n_ba: int = 20, n_ab: int = 20
) -> list[dict]:
    """Concatenated two-skill traces with known cut and path. Stages 1-2B use K=2."""
    a, b = skill_a(), skill_b_total()
    corpus = [_concat_trace(b, SKILL_B, a, SKILL_A, rng) for _ in range(n_ba)]
    corpus += [_concat_trace(a, SKILL_A, b, SKILL_B, rng) for _ in range(n_ab)]
    return corpus


def make_stage3c_corpus(
    rng: np.random.Generator,
    n_ba: int = 30,
    n_bc: int = 2,
    n_ab: int = 2,
    n_ac: int = 30,
) -> list[dict]:
    """Corpus for the optional Stage 3C, using the **antichain** B and skill C."""
    a, b, c = stage3_skills()
    corpus = [_concat_trace(b, SKILL_B, a, SKILL_A, rng) for _ in range(n_ba)]
    corpus += [_concat_trace(b, SKILL_B, c, SKILL_C, rng) for _ in range(n_bc)]
    corpus += [_concat_trace(a, SKILL_A, b, SKILL_B, rng) for _ in range(n_ab)]
    corpus += [_concat_trace(a, SKILL_A, c, SKILL_C, rng) for _ in range(n_ac)]
    return corpus


def assert_stage3_b_is_antichain() -> None:
    """Guard: Stage 3's B template must induce no precedence whatsoever."""
    p = precedence_from_u(U_B_ANTICHAIN)
    if p.any():
        raise AssertionError(f"U_B_ANTICHAIN is not an antichain:\n{p}")


# ---------------------------------------------------------------------------
# Stage 4 — the proposal-kernel toy
# ---------------------------------------------------------------------------
#
# The Stage 0-3 skills cannot exercise a local move kernel at all:
#
#   * every block matches at most one skill, so Relabel has nothing to move to;
#   * every B block consumes exactly one CPA-2 label, so the number of segments L
#     is pinned by the trace and Split/Merge can never land on a legal state.
#
# Stage 4 therefore needs supports that genuinely overlap. Two changes do it:
#
#   * S4_A and S4_D share the CPA label set {0,1} but carry different orders, so a
#     2-block is compatible with both and Relabel becomes a real move;
#   * S4_E's label set {0,1,2,3} is exactly the union of S4_A's and S4_F's, so a
#     4-block can be merged into one E segment or split into A|F. That makes L vary
#     across legal states, which is what Split/Merge need.
#
# Shift is exercised because a boundary can slide between the 2-block and 4-block
# tilings, reassigning both adjacent skills.

S4_SKILL_A = 0
S4_SKILL_D = 1
S4_SKILL_F = 2
S4_SKILL_E = 3

S4_LABELS_A = (0, 1)
S4_LABELS_D = (0, 1)   # deliberately the same support as A
S4_LABELS_F = (2, 3)
S4_LABELS_E = (0, 1, 2, 3)

U_S4_A = np.array([[1.0, 1.0], [0.0, 0.0]])   # 0 > 1
U_S4_D = np.array([[1.0, 0.0], [0.0, 1.0]])   # antichain: both orders equally likely
U_S4_F = np.array([[1.0, 1.0], [0.0, 0.0]])   # 2 > 3
# 0 > 1 and 2 > 3, with the two chains mutually incomparable
U_S4_E = np.array([[3.0, 0.0], [2.0, -1.0], [0.0, 3.0], [-1.0, 2.0]])

# Tiled by blocks of size 2 and 4, this trace has exactly 11 legal segmentations with
# L in {2, 3, 4}, and all four move types are live on it.
#
# It is chosen for one further property: the proposal law it induces is genuinely
# **asymmetric**. The all-E state has 4 available splits while the states reachable
# from it have only 1 available merge back, so q(S -> S') != q(S' -> S). A kernel that
# assumed symmetry and dropped the Hastings ratio would converge to a stationary
# distribution 0.128 in total variation away from the posterior — while still running,
# still mixing and still looking healthy. Two nearby traces cannot detect that bug at
# all: (0,1,2,3,0,1) is accidentally symmetric, and (0,1,2,3,2,3,0,1) never offers a
# legal Shift. Both were rejected for this reason.
S4_TRACE = (0, 1, 2, 3, 0, 1, 2, 3)
S4_DELTA_B = 0.5


def stage4_skills() -> tuple[SkillTemplate, ...]:
    """K = 4: A and D share a support; E's support is A's union F's."""
    return (
        SkillTemplate(cpa_labels=S4_LABELS_A, u=U_S4_A, beta=BETA, epsilon=EPSILON),
        SkillTemplate(cpa_labels=S4_LABELS_D, u=U_S4_D, beta=BETA, epsilon=EPSILON),
        SkillTemplate(cpa_labels=S4_LABELS_F, u=U_S4_F, beta=BETA, epsilon=EPSILON),
        SkillTemplate(cpa_labels=S4_LABELS_E, u=U_S4_E, beta=BETA, epsilon=EPSILON),
    )


S4_SKILL_NAMES = ("A", "D", "F", "E")
