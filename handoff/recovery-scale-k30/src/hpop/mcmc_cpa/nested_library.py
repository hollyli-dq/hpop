"""One master skill library per replicate, and the nested K ladder cut from it.

Section 5 of the ladder preregistration: draw a single master library at `K_max = 30`,
permute the thirty skills once, and take the first `K` of the permuted order for each rung.
So within a replicate

    K=3  subset  K=5  subset  K=10  subset  K=20  subset  K=30

and the nesting covers CPA supports, role maps, latent utilities and the induced
role-labelled closures together -- not one of them at a time.

## Why nesting rather than an independent draw per rung

An independent library at every `K` would confound library size with library difficulty:
a bad K=30 result could be thirty skills being hard, or those particular thirty being hard.
Nesting removes that. The K=3 rung is *literally three of* the K=30 rung, so a difference
between rungs cannot be a difference of draw.

The cost is that the rungs are no longer independent, which matters for how the ladder is
read: it is a within-replicate comparison, and the two replicates are what carry the
between-truth variation. With exactly two of them, report both points and their range --
never a Gaussian interval through two numbers.

## What is NOT nested

`pi` and `P` are drawn per rung from the registered prior with their own seeds, because a
K=30 transition matrix truncated to its first three rows and columns is not a draw from
the K=3 prior. Truncating and renormalising would quietly change the generative model
between rungs.
"""

from __future__ import annotations

import hashlib

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u

from .role_maps import RoleMaps, sample_role_maps

__all__ = ["MasterLibrary", "draw_master_library", "K_LADDER"]

K_LADDER = (3, 5, 10, 20, 30)


class MasterLibrary:
    """A `K_max`-skill library, already permuted, with the ladder cut from its prefixes."""

    __slots__ = ("u", "role_maps", "permutation", "n_cpa", "replicate", "seeds")

    def __init__(self, u, role_maps: RoleMaps, permutation, n_cpa: int,
                 replicate: int, seeds: dict):
        self.u = np.asarray(u, dtype=float)
        self.role_maps = role_maps
        self.permutation = np.asarray(permutation, dtype=np.int64)
        self.n_cpa = int(n_cpa)
        self.replicate = int(replicate)
        self.seeds = dict(seeds)

    @property
    def k_max(self) -> int:
        return int(self.u.shape[0])

    @property
    def n_roles(self) -> int:
        return int(self.u.shape[1])

    # ------------------------------------------------------------------ the ladder
    def prefix(self, k: int) -> tuple:
        """The first `k` skills: `(U_k, RoleMaps_k)`. Nested by construction."""
        k = int(k)
        if not 1 <= k <= self.k_max:
            raise ValueError(f"K must be in [1, {self.k_max}], got {k}")
        return (self.u[:k].copy(),
                RoleMaps(self.role_maps.forward[:k].copy(), self.n_cpa))

    def closure_bits(self, k: int) -> np.ndarray:
        """Role-labelled transitive-closure bits for the first `k` skills, `(k, m(m-1))`."""
        m = self.n_roles
        off = ~np.eye(m, dtype=bool)
        return np.stack([np.asarray(precedence_from_u(self.u[i]))[off]
                         for i in range(int(k))])

    def library_digest(self, k: int) -> str:
        """Canonical identifier for the first `k` skills.

        Role-labelled support **and** closure bits, per skill, sorted across skills before
        hashing. Sorting makes it invariant to relabelling -- which is what lets it be used
        at K = 30, where the 30! relabellings cannot be enumerated. Section 5 requires the
        support to be part of the identity, not the closure alone: two skills can share a
        closure shape while acting on different CPAs, and those are different skills.
        """
        maps = self.role_maps.forward
        parts = sorted(
            np.ascontiguousarray(np.sort(maps[i])).tobytes()
            + np.packbits(self.closure_bits(k)[i]).tobytes()
            for i in range(int(k)))
        return hashlib.sha256(b"".join(parts)).hexdigest()[:16]

    def as_dict(self) -> dict:
        return {
            "replicate": self.replicate, "k_max": self.k_max,
            "n_roles": self.n_roles, "n_cpa": self.n_cpa,
            "permutation": self.permutation.tolist(),
            "seeds": self.seeds,
            "role_maps": self.role_maps.as_dict(),
            "ladder_digests": {str(k): self.library_digest(k) for k in K_LADDER
                               if k <= self.k_max},
            "relations_per_skill": [int(b.sum()) for b in self.closure_bits(self.k_max)],
        }


# ------------------------------------------------------------------- admissibility
def _admissible(u, maps: RoleMaps) -> tuple:
    """Section 7, criteria 1-7. Functions of the truth alone; no data, no likelihood."""
    m = int(u.shape[1])
    pairs = m * (m - 1) // 2
    reasons = []
    off = ~np.eye(m, dtype=bool)

    closures, supports = [], []
    for k in range(u.shape[0]):
        precedence = np.asarray(precedence_from_u(u[k]))
        strict = precedence.copy()
        np.fill_diagonal(strict, False)
        relations = int(strict.sum())
        if relations < 1:
            reasons.append(f"skill {k}: closure has no relation")
        if relations >= pairs:
            reasons.append(f"skill {k}: {relations} relations is not below {pairs}")
        if np.any(strict & strict.T):
            reasons.append(f"skill {k}: closure is not antisymmetric")
        closures.append(precedence[off].tobytes())
        supports.append(tuple(sorted(maps.forward[k].tolist())))

    labelled = {(c, s) for c, s in zip(closures, supports)}
    if len(labelled) != u.shape[0]:
        reasons.append("two skills share a role-labelled closure")
    if len(set(supports)) != u.shape[0]:
        reasons.append("two skills share a CPA support")
    if any(len(set(s)) != m for s in supports):
        reasons.append("a support does not hold exactly m distinct CPAs")
    return (not reasons), reasons


def draw_master_library(replicate: int, k_max: int = 30, n_roles: int = 10,
                        n_cpa: int = 50, rho: float = 0.5,
                        max_attempts: int = 100) -> tuple:
    """Draw one admissible master library for a replicate, with the registered seeds.

    Rejection sampling against the predeclared admissibility event and nothing else. Every
    attempted seed and every rejection reason is returned, so the record shows what was
    rejected rather than only what survived.
    """
    from hpop.mcmc_original.stage6c_frozen import sigma_rho_matrix

    replicate = int(replicate)
    seeds = {
        "master_structural_truth": 6_500_001 + replicate,
        "master_role_support_library": 6_500_051 + replicate,
        "master_skill_permutation": 6_500_101 + replicate,
    }
    chol = np.linalg.cholesky(sigma_rho_matrix(2, float(rho)))
    maps = sample_role_maps(k_max, n_roles, n_cpa,
                            seed=seeds["master_role_support_library"])

    attempts = []
    for attempt in range(int(max_attempts)):
        seed = seeds["master_structural_truth"] + 1000 * attempt
        rng = np.random.default_rng(seed)
        u = np.stack([np.stack([chol @ rng.standard_normal(2) for _ in range(n_roles)])
                      for _ in range(k_max)])
        ok, reasons = _admissible(u, maps)
        attempts.append({"attempt": attempt, "seed": seed, "accepted": bool(ok),
                         "rejection_reasons": reasons[:4]})
        if ok:
            permutation = np.random.default_rng(
                seeds["master_skill_permutation"]).permutation(k_max)
            library = MasterLibrary(u[permutation],
                                    RoleMaps(maps.forward[permutation], n_cpa),
                                    permutation, n_cpa, replicate, seeds)
            return library, attempts
    raise RuntimeError(
        f"no admissible master library for replicate {replicate} in {max_attempts} "
        f"attempts; do not relax the criteria -- report and stop")


def draw_master_library_v2(replicate: int, k_max: int = 30, n_roles: int = 10,
                           n_cpa: int = 50, rho: float = 0.5,
                           max_candidates: int = 3000) -> tuple:
    """Evidence-admissible master library: skills collected ONE AT A TIME.

    v1 rejected whole 30-skill draws, which is fine for structural criteria that almost
    always hold but infeasible once per-skill evidence admissibility is added (a 40%
    per-skill pass rate makes a whole-library accept a once-in-10^12 event). Here each
    candidate skill is drawn from the same latent prior, kept iff it passes BOTH the
    structural checks and the pair-evidence admissibility of `exposure.evidence_admissible`
    (floors derived from the registered corpus size), and rejected otherwise -- with every
    rejection recorded. Cross-skill distinctness is enforced as candidates are admitted.

    Nesting across the ladder is preserved: rung K is the first K admitted skills, in
    admission order, before the registered permutation.
    """
    from hpop.mcmc_original.stage6c_frozen import sigma_rho_matrix

    from .exposure import evidence_admissible
    from .recovery_regime import REGIME, generation_params

    replicate = int(replicate)
    seeds = {
        "master_structural_truth": 6_900_001 + replicate,
        "master_role_support_library": 6_900_051 + replicate,
        "master_skill_permutation": 6_900_101 + replicate,
    }
    chol = np.linalg.cholesky(sigma_rho_matrix(2, float(rho)))
    maps = sample_role_maps(k_max, n_roles, n_cpa,
                            seed=seeds["master_role_support_library"])

    mean_width = (REGIME.MIN_WIDTH + REGIME.MAX_WIDTH) / 2.0
    expected_instances = REGIME.TRAIN_PER_SKILL * REGIME.TRACE_LENGTH / mean_width
    params = generation_params()

    admitted, closures, attempts = [], set(), []
    candidate = 0
    while len(admitted) < int(k_max):
        if candidate >= int(max_candidates):
            raise RuntimeError(
                f"only {len(admitted)} of {k_max} skills admissible after "
                f"{max_candidates} candidates; report and stop -- do not relax floors")
        rng = np.random.default_rng(seeds["master_structural_truth"]
                                    + 1000 * candidate)
        u_skill = np.stack([chol @ rng.standard_normal(2) for _ in range(n_roles)])
        reasons = []

        closure = np.asarray(precedence_from_u(u_skill))
        relations = int(closure.sum())
        pairs = n_roles * (n_roles - 1)
        if relations < 1:
            reasons.append("closure has no relation")
        if relations >= pairs:
            reasons.append("closure is a total order")
        digest = closure.tobytes()
        if digest in closures:
            reasons.append("duplicate closure")

        if not reasons:
            ok, ev_reasons, profile = evidence_admissible(
                u_skill[None], params, REGIME.DELTA_B, REGIME.MIN_WIDTH,
                REGIME.MAX_WIDTH, REGIME.TRACE_LENGTH, REGIME.EXPOSURE_PROBES,
                seeds["master_structural_truth"] + 500_000 + candidate,
                expected_instances=expected_instances,
                edge_min_expected=5.0, incomp_min_expected_each_way=2.0)
            if not ok:
                reasons.extend(ev_reasons)

        attempts.append({"candidate": candidate, "accepted": not reasons,
                         "rejection_reasons": reasons[:3]})
        if not reasons:
            admitted.append(u_skill)
            closures.add(digest)
        candidate += 1

    u = np.stack(admitted)
    permutation = np.random.default_rng(
        seeds["master_skill_permutation"]).permutation(int(k_max))
    library = MasterLibrary(u[permutation],
                           RoleMaps(maps.forward[permutation], n_cpa),
                           permutation, n_cpa, replicate, seeds)
    accepted = sum(1 for a in attempts if a["accepted"])
    meta = {"scheme": "v2 per-skill evidence admissibility",
            "candidates": candidate, "accepted": accepted,
            "per_skill_acceptance": accepted / candidate,
            "floors": {"edge_min_expected": 5.0,
                       "incomp_min_expected_each_way": 2.0,
                       "expected_instances": expected_instances},
            "attempts": attempts}
    return library, meta
