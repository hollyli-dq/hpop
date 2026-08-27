"""The all-trace FFBS draw, with the forward pass batched by length class.

The backward pass is the frozen `backward_sample`, called in ORIGINAL trace order. That
ordering is not incidental: it is what keeps the rng consumption sequence identical to the
reference. Batching changes where alpha comes from, never the order draws are taken in.
"""

from __future__ import annotations

import numpy as np

from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import key_movement, key_of
from hpop.mcmc_original.semi_markov_ffbs import backward_sample
from hpop.mcmc_original.stage6e_state import Stage6EModel, Stage6EState
from hpop.mcmc_original.transitions import log_transition_matrix

from .flags import FLAGS
from .forward import forward_batched_group, forward_dispatch


def ffbs_segmentation_draw(model: Stage6EModel, state: Stage6EState, tables, rng) -> dict:
    """Draw `(S_n, z_n)` for every trace. Same contract as the reference."""
    log_pi = np.log(state.pi)
    log_transition = log_transition_matrix(state.transition)
    previous = [key_of(s) for s in state.segmentations]
    all_tables = list(tables.tables_for(state))

    precomputed = None
    if FLAGS.batched_forward:
        groups: dict = {}
        for n, table in enumerate(all_tables):
            groups.setdefault(np.asarray(table).shape[0], []).append(n)
        precomputed = [None] * len(all_tables)
        for _length, members in sorted(groups.items()):
            charts = forward_batched_group(
                [all_tables[n] for n in members], log_pi, log_transition,
                model.delta_b, model.max_width, model.min_width)
            for n, chart in zip(members, charts):
                precomputed[n] = chart

    keys, normalizers = [], []
    movement = {"boundary_hamming": 0, "label_changes": 0, "states_changed": 0}
    for n, table in enumerate(all_tables):
        chart = (precomputed[n] if precomputed is not None
                 else forward_dispatch(table, log_pi, log_transition, model.delta_b,
                                       model.max_width, model.min_width))
        blocks = backward_sample(chart, rng)        # the frozen exact draw
        key = tuple((int(b), int(k)) for _, b, k in blocks)
        keys.append(key)
        normalizers.append(chart.log_normalizer)
        hamming, changes = key_movement(previous[n], key)
        movement["boundary_hamming"] += hamming
        movement["label_changes"] += changes
        movement["states_changed"] += int(key != previous[n])
    return {"keys": tuple(keys), "movement": movement,
            "log_normalizers": np.array(normalizers, dtype=float)}
