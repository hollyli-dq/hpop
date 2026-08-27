"""Candidate block-score tables that skip the rebuild when H = h(U) has not moved.

`BlockScoreTable.refresh` reads `u_by_skill` only through

    precedence = np.all(u[:, None, :] > u[None, :, :], axis=2)

and through `u.shape[0]`. The table is therefore a pure function of H = h(U) and the four
recurrent scalars -- not of U. In the FULL-LATENT arms the scalars are held fixed by
`FullLatentFixed`, so the table depends on H alone, and U moves far more often than H does.

Keying the cache on H makes most rebuilds no-ops that return exactly the bits a rebuild
would have produced. This is the only optimisation in the package that is bit-identical:
nothing is recomputed, so nothing can differ.
"""

from __future__ import annotations

import time

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import FFBSBlockTables
from hpop.mcmc_original.stage6e_state import Stage6EState

from .flags import COUNTERS, FLAGS


class HashCachedFFBSBlockTables(FFBSBlockTables):
    """`FFBSBlockTables` plus an H-keyed short circuit on `refresh`.

    Subclassed rather than edited: `recurrent_joint_ffbs_mcmc.py` is one of the ten
    Condition C `SHARED_SOURCES` and must stay byte-identical.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._structural_key = None
        self._ever_built = False

    @staticmethod
    def structural_key(state: Stage6EState) -> tuple:
        u = np.asarray(state.u_by_skill, dtype=float)
        precedence = b"".join(
            np.ascontiguousarray(precedence_from_u(u[k])).tobytes()
            for k in range(u.shape[0]))
        return (precedence, float(state.beta), float(state.omega),
                float(state.lambda_rep), float(state.lambda_back))

    def refresh(self, state: Stage6EState) -> None:
        if not FLAGS.emission_hash_cache:
            COUNTERS.emission_rebuilds += 1
            return super().refresh(state)

        began = time.perf_counter()
        structural = self.structural_key(state)
        if self._ever_built and structural == self._structural_key:
            COUNTERS.emission_cache_hits += 1
            # the stored table IS the table a rebuild would produce, bit for bit
            self._fingerprint = self._parameters_of(state)
            self.stale = False
            # `last_refresh` records what THIS call did, and this call rebuilt nothing;
            # leaving the previous record would report a rebuild that never happened
            self.last_refresh = {
                "rebuilt_skills": [],
                "reused_skills": list(range(int(self.model.n_skills)))}
            self.build_seconds += time.perf_counter() - began
            return None

        COUNTERS.emission_rebuilds += 1
        out = super().refresh(state)
        self._structural_key = structural
        self._ever_built = True
        return out
