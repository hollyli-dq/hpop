"""Condition C' chain: the Condition-C chain plus the global skill-transposition.

One transition is added to Condition C and nothing else. The chain class
SUBCLASSES `matched_condition_c.ConditionCChain` and advances the base chain one
sweep at a time, so every piece of validated bookkeeping — retained draws, the
online boundary/occupancy accumulators, the fixed-coordinate assertion, the
checkpoint payload — is reused rather than re-derived. `matched_condition_c.py`
itself is not modified, so the in-flight Condition-C run keeps a byte-identical
import path.

## Frozen transition ordering (from the C' preregistration, commit 9b8e590)

    scheduled skill transposition
        -> [Condition-C sweep] collapsed U move if scheduled
        -> [Condition-C sweep] exact FFBS refresh of ALL (S, z)
        -> [Condition-C sweep] conditional U rows

The preregistration fixes the swap BEFORE the unchanged Condition-C sweep. That
satisfies the requirement that no path-dependent operation may run between a
swap attempt and the FFBS refresh, because the only transition that can
intervene is the collapsed U move, and that move provably never reads the
stored `(S, z)`: its acceptance ratio contains no conditional-likelihood term,
and `test_collapsed_u_ordering.py` pins its decision to be byte-identical under
a scrambled stored segmentation. The first path-dependent operation in the sweep
is the conditional U row phase, which runs strictly after the FFBS refresh.

The refresh runs on EVERY sweep, so it runs after every swap attempt whether the
swap was accepted or rejected; `ffbs_refreshes_after_swap == swap_attempts` is
asserted at every checkpoint.

## Blinding

`SealedTruth` exposes only the FIXED inputs a Condition-C' chain is entitled to
read (the four recurrent scalars, `pi*`, `P*`, `epsilon`, `delta_B`, widths).
Every hidden-truth field — `U*`, `H*`, and the per-trace `S*`, `z*`, role blocks
— raises until `unseal()` is called, which the runner permits only once the
registered stopping condition has been reached.
"""

from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np

from hpop.mcmc_original.collapsed_u_kernel import CollapsedUConfig
from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.matched_condition_b import canonical_h_hash
from hpop.mcmc_original.matched_condition_c import (
    ConditionCChain, ConditionCFixed, ConditionCSampler,
    build_condition_c_model,
)
from hpop.mcmc_original.skill_swap_kernel import (
    SkillSwapConfig, is_swap_sweep, skill_swap_mh_step, unordered_pairs,
)

__all__ = [
    "SealedTruth", "SealedTruthError", "ConditionCPrimeChain",
    "swap_diagnostics",
]


class SealedTruthError(RuntimeError):
    """Raised when hidden truth is read before the registered unsealing point."""


class SealedTruth:
    """Fixed inputs are readable; hidden truth raises until explicitly unsealed.

    Condition C' conditions on `vartheta*`, `pi*`, `P*`, `delta_B*`, `epsilon*`
    and `rho_0` — these are model INPUTS, not latent state, and are exposed.
    `U*`, the induced `H*`, and the per-trace `S*`/`z*`/role blocks are the
    quantities recovery is scored against and stay sealed.
    """

    FIXED_FIELDS = frozenset({
        "beta", "omega", "lambda_rep", "lambda_back", "epsilon", "delta_b",
        "min_width", "max_width", "n_skills", "n_roles", "latent_dim",
        "pi", "transition", "role_maps", "rfs_parameters", "scalars"})
    SEALED_FIELDS = frozenset({"u_by_skill", "precedence", "rho"})

    def __init__(self, truth, sealed: bool = True) -> None:
        object.__setattr__(self, "_truth", truth)
        object.__setattr__(self, "_sealed", bool(sealed))
        object.__setattr__(self, "_reads", [])

    @property
    def sealed(self) -> bool:
        return self._sealed

    def unseal(self, reason: str) -> None:
        """Permitted only by the runner, once the stopping condition is met."""
        object.__setattr__(self, "_sealed", False)
        self._reads.append(("UNSEALED", str(reason)))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self.SEALED_FIELDS and self._sealed:
            raise SealedTruthError(
                f"'{name}' is hidden truth and is sealed until the registered "
                "Condition C' stopping condition permits recovery analysis")
        if name in self.FIXED_FIELDS:
            return getattr(self._truth, name)
        if self._sealed:
            raise SealedTruthError(
                f"'{name}' is not a registered fixed input; refusing to expose "
                "it while sealed")
        return getattr(self._truth, name)

    def fixed_for_condition_c(self, rho_0: float) -> ConditionCFixed:
        """The only construction a sealed chain needs; touches no hidden field."""
        return ConditionCFixed(
            rho_0=float(rho_0), beta=float(self.beta), omega=float(self.omega),
            lambda_rep=float(self.lambda_rep),
            lambda_back=float(self.lambda_back),
            pi=tuple(float(v) for v in self.pi),
            transition=tuple(tuple(float(v) for v in row)
                             for row in self.transition))


class ConditionCPrimeChain(ConditionCChain):
    """Condition-C chain plus the scheduled global skill transposition."""

    def __init__(self, sampler: ConditionCSampler, u_start, seed: int,
                 burn_in: int, thin: int, swap: SkillSwapConfig) -> None:
        super().__init__(sampler, u_start, seed, burn_in, thin)
        self.swap = swap
        self.swap_attempts = 0
        self.swap_accepts = 0
        self.swap_assignment_changes = 0
        self.swap_z_changes = 0
        self.ffbs_refreshes = 0
        self.ffbs_refreshes_after_swap = 0
        self.swap_seconds = 0.0
        self.swap_by_pair = {f"{j}-{k}": {"attempts": 0, "accepts": 0}
                             for j, k in unordered_pairs(
                                 sampler.model.n_skills)}
        self.swap_deltas: list = []          # (pair, delta, accepted, sweep)
        self.accepted_assignment_change_sweeps: list = []

    # ------------------------------------------------------------------ running
    def advance(self, upto: int, checkpoint_path=None, checkpoint_every: int = 0,
                progress_every: int = 0) -> None:
        began = time.perf_counter()
        while self.state.iteration < int(upto):
            record = None
            labels_before = None
            if is_swap_sweep(self.state.iteration, self.swap.every):
                labels_before = [a.copy() for a in
                                 self.state.occurrence_labels()]
                assignment_before = self._assignment()
                t0 = time.perf_counter()
                self.state, record = skill_swap_mh_step(
                    self.state, self.sampler.collapsed_likelihood, self.rng)
                self.swap_seconds += time.perf_counter() - t0
                self.swap_attempts += 1
                key = f"{record['pair'][0]}-{record['pair'][1]}"
                self.swap_by_pair[key]["attempts"] += 1
                self.swap_deltas.append(
                    [key, float(record["d_log_lik_collapsed"]),
                     bool(record["accepted"]), int(self.state.iteration)])
                if record["accepted"]:
                    self.swap_accepts += 1
                    self.swap_by_pair[key]["accepts"] += 1
                    if self._assignment() != assignment_before:
                        self.swap_assignment_changes += 1
                        self.accepted_assignment_change_sweeps.append(
                            int(self.state.iteration))

            # exactly one unchanged Condition-C sweep, with all of its
            # validated bookkeeping; its first action is the FFBS refresh
            super().advance(self.state.iteration + 1)
            self.ffbs_refreshes += 1
            if record is not None:
                self.ffbs_refreshes_after_swap += 1
                if record["accepted"]:
                    after = self.state.occurrence_labels()
                    if any(not np.array_equal(a, b)
                           for a, b in zip(labels_before, after)):
                        self.swap_z_changes += 1

            if checkpoint_path and checkpoint_every and \
                    self.state.iteration % int(checkpoint_every) == 0:
                self.save(checkpoint_path)
            if progress_every and self.state.iteration % progress_every == 0:
                print(f"      chain seed {self.seed}: sweep "
                      f"{self.state.iteration:,} "
                      f"(swap {self.swap_accepts}/{self.swap_attempts}, "
                      f"{time.perf_counter() - began:.0f}s)", flush=True)
        self.assert_ordering_invariant()
        if checkpoint_path:
            self.save(checkpoint_path)

    def _assignment(self) -> tuple:
        return tuple(canonical_h_hash(precedence_from_u(self.state.u_by_skill[k]))
                     for k in range(self.sampler.model.n_skills))

    def assert_ordering_invariant(self) -> None:
        """Every swap attempt, accepted or rejected, was followed by a refresh."""
        if self.ffbs_refreshes_after_swap != self.swap_attempts:
            raise AssertionError(
                "partially-collapsed ordering violated: "
                f"{self.swap_attempts} swap attempts but "
                f"{self.ffbs_refreshes_after_swap} refreshes after a swap")
        if self.ffbs_refreshes < self.swap_attempts:
            raise AssertionError("fewer FFBS refreshes than swap attempts")

    # -------------------------------------------------------------- persistence
    def save(self, path) -> None:
        super().save(path)
        data = dict(np.load(str(path), allow_pickle=False))
        meta = json.loads(str(data["meta"]))
        meta["swap"] = {
            "every": int(self.swap.every),
            "attempts": self.swap_attempts, "accepts": self.swap_accepts,
            "assignment_changes": self.swap_assignment_changes,
            "z_changes": self.swap_z_changes,
            "ffbs_refreshes": self.ffbs_refreshes,
            "ffbs_refreshes_after_swap": self.ffbs_refreshes_after_swap,
            "seconds": self.swap_seconds,
            "by_pair": self.swap_by_pair,
            "assignment_change_sweeps": self.accepted_assignment_change_sweeps,
        }
        data["meta"] = np.array(json.dumps(meta))
        data["swap_deltas"] = np.array(
            [[d[0], str(d[1]), str(int(d[2])), str(d[3])]
             for d in self.swap_deltas], dtype="U24") if self.swap_deltas \
            else np.zeros((0, 4), dtype="U24")
        tmp = str(path) + ".tmp.npz"
        np.savez_compressed(tmp, **data)
        import os
        os.replace(tmp, str(path))

    @classmethod
    def load(cls, path, sampler: ConditionCSampler,
             swap: SkillSwapConfig) -> "ConditionCPrimeChain":
        base = ConditionCChain.load(path, sampler)
        chain = cls.__new__(cls)
        chain.__dict__.update(base.__dict__)
        meta = json.loads(str(np.load(str(path))["meta"]))
        payload = meta.get("swap", {})
        if payload and int(payload.get("every", swap.every)) != int(swap.every):
            raise ValueError(
                f"checkpoint was written at swap cadence "
                f"{payload['every']}, cannot resume at {swap.every}")
        chain.swap = swap
        chain.swap_attempts = int(payload.get("attempts", 0))
        chain.swap_accepts = int(payload.get("accepts", 0))
        chain.swap_assignment_changes = int(payload.get("assignment_changes",
                                                        0))
        chain.swap_z_changes = int(payload.get("z_changes", 0))
        chain.ffbs_refreshes = int(payload.get("ffbs_refreshes", 0))
        chain.ffbs_refreshes_after_swap = int(
            payload.get("ffbs_refreshes_after_swap", 0))
        chain.swap_seconds = float(payload.get("seconds", 0.0))
        chain.swap_by_pair = payload.get(
            "by_pair", {f"{j}-{k}": {"attempts": 0, "accepts": 0}
                        for j, k in unordered_pairs(sampler.model.n_skills)})
        chain.accepted_assignment_change_sweeps = list(
            payload.get("assignment_change_sweeps", []))
        raw = np.load(str(path))
        chain.swap_deltas = ([[r[0], float(r[1]), bool(int(r[2])), int(r[3])]
                              for r in raw["swap_deltas"]]
                             if "swap_deltas" in raw.files else [])
        return chain


def swap_diagnostics(chain: "ConditionCPrimeChain") -> dict:
    """The registered per-chain swap diagnostic block."""
    deltas = chain.swap_deltas
    by_pair = {}
    for key, stats in chain.swap_by_pair.items():
        values = [d[1] for d in deltas if d[0] == key]
        by_pair[key] = {
            "attempts": stats["attempts"], "accepts": stats["accepts"],
            "acceptance": (stats["accepts"] / stats["attempts"]
                           if stats["attempts"] else None),
            "delta_mean": float(np.mean(values)) if values else None,
            "delta_median": float(np.median(values)) if values else None,
            "delta_min": float(np.min(values)) if values else None,
            "delta_max": float(np.max(values)) if values else None,
        }
    gaps = np.diff(chain.accepted_assignment_change_sweeps) \
        if len(chain.accepted_assignment_change_sweeps) > 1 else np.array([])
    anchored = Counter(chain.retained_h_hashes)
    libraries = Counter(tuple(sorted(h)) for h in chain.retained_h_hashes)
    return {
        "swap_attempts": chain.swap_attempts,
        "swap_accepts": chain.swap_accepts,
        "swap_acceptance": (chain.swap_accepts / chain.swap_attempts
                            if chain.swap_attempts else None),
        "accepted_assignment_changes": chain.swap_assignment_changes,
        "accepted_swaps_with_z_reallocation": chain.swap_z_changes,
        "ffbs_refreshes_after_swap": chain.ffbs_refreshes_after_swap,
        "ffbs_refreshes_equals_attempts":
            chain.ffbs_refreshes_after_swap == chain.swap_attempts,
        "by_pair": by_pair,
        "largest_positive_delta": (max((d[1] for d in deltas), default=None)),
        "largest_negative_delta": (min((d[1] for d in deltas), default=None)),
        "seconds_total": chain.swap_seconds,
        "seconds_per_swap": (chain.swap_seconds / chain.swap_attempts
                             if chain.swap_attempts else None),
        "overhead_fraction_of_chain": (chain.swap_seconds / chain.seconds
                                       if chain.seconds else None),
        "sweeps_between_assignment_changes": {
            "n": int(gaps.size),
            "mean": float(gaps.mean()) if gaps.size else None,
            "min": int(gaps.min()) if gaps.size else None,
            "max": int(gaps.max()) if gaps.size else None},
        "distinct_anchored_tuples": len(anchored),
        "distinct_unordered_libraries": len(libraries),
    }
