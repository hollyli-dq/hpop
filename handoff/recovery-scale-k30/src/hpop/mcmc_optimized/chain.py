"""A FULL-LATENT chain that advances on the optimized backend.

`FullLatentChain.advance` calls the reference `full_latent_sweep_once`. Everything else it
does -- retention, summaries, online accumulators, the checkpoint format, the marginal
U/FFBS ordering assertion -- is exactly what the confirmatory run needs, and
`matched_full_latent.py` is pinned by `SOURCE_DIGESTS`, so this subclass overrides only the
sweep call.

It also records the two quantities the preregistration's branch-(a) preconditions need:
the canonical library of the START, and per-chain counts of accepted H-changing moves
during WARM-UP. Both are integers or identifiers about kernel behaviour, not draws; no
warm-up sample enters any posterior summary.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.matched_full_latent import FULL_MARG, FullLatentChain

from .full_latent import sweep_once


def canonical_library(u_by_skill) -> str:
    """The exact canonical closure library: sorted per-skill closures, hashed.

    Invariant to skill relabelling, and exact -- two states share an identifier iff their
    partial orders are equal up to relabelling. This is the preregistered structural
    diagnostic; relation counts are secondary.
    """
    u = np.asarray(u_by_skill, dtype=float)
    blobs = sorted(np.ascontiguousarray(precedence_from_u(u[k])).tobytes()
                   for k in range(u.shape[0]))
    return hashlib.sha256(b"".join(blobs)).hexdigest()[:16]


class OptimizedFullLatentChain(FullLatentChain):
    """`FullLatentChain` advanced by `hpop.mcmc_optimized.sweep_once`."""

    def __init__(self, *args, warmup_sweeps: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_sweeps = int(warmup_sweeps)
        self.start_library = canonical_library(self.state.u_by_skill)
        self.warmup_h_accepts = 0
        self.production_libraries: dict = {}

    def advance(self, upto: int, checkpoint_path=None, checkpoint_every: int = 0,
                progress_every: int = 0) -> None:
        began = last_mark = time.perf_counter()
        while self.state.iteration < int(upto):
            state, info = sweep_once(self.state, self.sampler, self.rng)
            record = info["structural_record"]
            if record is not None:
                self.structural["attempts"] += 1
                self.structural["accepts"] += int(record["accepted"])
                h_accept = int(record["accepted"] and record["h_changed"])
                self.structural["h_accepts"] += h_accept
                if self.state.iteration < self.warmup_sweeps:
                    self.warmup_h_accepts += h_accept
                self.structural["invalid"] += int(record["invalid"])
                if self.sampler.config.arm == FULL_MARG:
                    self.structural["marginal_attempts"] += 1
            for key in ("boundary_hamming", "label_changes", "states_changed"):
                self.movement[key] += int(info["ffbs"]["movement"][key])
            if self.sampler.config.arm == FULL_MARG and info["scheduled_structural"]:
                if info["kernel_order"] != ("marginal_U", "FFBS", "pi_P"):
                    raise AssertionError("marginal attempt was not immediately refreshed")
                self.structural["ffbs_after_marginal"] += 1
            self.state = state
            sweep = self.state.iteration
            if sweep > self.burn_in and (sweep - self.burn_in) % self.thin == 0:
                self._retain()
                library = canonical_library(self.state.u_by_skill)
                self.production_libraries[library] = (
                    self.production_libraries.get(library, 0) + 1)
            now = time.perf_counter()
            if checkpoint_path and checkpoint_every and sweep % int(checkpoint_every) == 0:
                self.seconds += now - last_mark
                last_mark = now
                self.save(checkpoint_path)
            if progress_every and sweep % int(progress_every) == 0:
                print(f"      {self.sampler.config.arm} seed {self.seed}: sweep "
                      f"{sweep:,} ({time.perf_counter() - began:.0f}s segment)",
                      flush=True)
        self.seconds += time.perf_counter() - last_mark
        if self.sampler.config.arm == FULL_MARG and (
                self.structural["marginal_attempts"]
                != self.structural["ffbs_after_marginal"]):
            raise AssertionError("marginal U/FFBS ordering counter mismatch")
        self.state.rng_state = self.rng.bit_generator.state
        if checkpoint_path:
            self.save(checkpoint_path)

    def preconditions(self) -> dict:
        """Evidence for the preregistered branch-(a) preconditions, per chain."""
        return {"start_library": self.start_library,
                "warmup_h_accepts": int(self.warmup_h_accepts),
                "warmup_h_accepts_at_least_one": bool(self.warmup_h_accepts >= 1),
                "production_library_counts": dict(self.production_libraries),
                "production_library_constant":
                    len(self.production_libraries) == 1}
