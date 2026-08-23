"""The collapsed U likelihood, with its all-trace forward pass batched by length class.

Only `log Z_n` is wanted here, so this is the batched recursion at its cheapest. Subclassed
rather than edited: `collapsed_u_likelihood.py` is one of the Condition C `SHARED_SOURCES`.

A note on what is NOT done here. A MARG structural sweep runs three all-trace forward
passes -- current U, candidate U', then FFBS -- and pi and P are frozen across all three,
so the third duplicates the second on accept and the first on reject. Reusing that chart
would remove a whole pass. It is left alone because the FFBS path uses `BlockScoreTable`
and this path uses `FastBlockScoreTable`, and the two agree to 3.55e-15 but not bitwise;
sharing a chart across them is a design change, not an optimisation.
"""

from __future__ import annotations

import numpy as np

from hpop.mcmc_original.collapsed_u_likelihood import CollapsedULikelihood
from hpop.mcmc_original.stage6e_state import Stage6EState
from hpop.mcmc_original.transitions import log_transition_matrix

from .flags import FLAGS
from .forward import forward_batched_group, forward_dispatch


class BatchedCollapsedULikelihood(CollapsedULikelihood):
    """`CollapsedULikelihood` whose `_forward_all` groups traces by length."""

    def _forward_all(self, state: Stage6EState) -> np.ndarray:
        log_pi = np.log(np.asarray(state.pi, dtype=float))
        log_p = log_transition_matrix(state.transition)
        self.evaluations += 1
        tables = list(self._table.tables)

        if not FLAGS.batched_forward:
            return np.array([
                forward_dispatch(table, log_pi, log_p, self.model.delta_b,
                                 self.model.max_width,
                                 self.model.min_width).log_normalizer
                for table in tables], dtype=float)

        groups: dict = {}
        for n, table in enumerate(tables):
            groups.setdefault(np.asarray(table).shape[0], []).append(n)
        out = np.empty(len(tables), dtype=float)
        for _length, members in sorted(groups.items()):
            charts = forward_batched_group(
                [tables[n] for n in members], log_pi, log_p, self.model.delta_b,
                self.model.max_width, self.model.min_width)
            for n, chart in zip(members, charts):
                out[n] = chart.log_normalizer
        return out
