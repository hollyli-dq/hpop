"""Checkpoint/resume determinism for the partially-collapsed runner.

An uninterrupted run and a checkpoint-then-resume run must agree bit for bit: same
draws, same final state, same RNG state, same move counters, same collapsed schedule
phase. The cadence is keyed on the ABSOLUTE sweep index carried by `state.iteration`,
so a resumed chain schedules its collapsed moves exactly where the uninterrupted one
does — that is what this file pins.
"""

from __future__ import annotations

import json

import numpy as np

from hpop.mcmc_original.collapsed_u_kernel import (
    MOVE_NAME, CollapsedUConfig, run_collapsed_u_chain,
)
from hpop.mcmc_original.stage6e_state import Stage6EState

from mcmc_original.test_collapsed_u_kernel import SCALES, tiny_model, tiny_state

SWEEPS = 24
CHECKPOINT_AT = 12
CADENCE = 5                        # collapsed moves at sweeps 4, 9, 14, 19
SEED = 77


def test_checkpoint_resume_is_bit_identical(tmp_path):
    full = run_collapsed_u_chain(
        model=tiny_model(), start=tiny_state(), scales=SCALES, num_sweeps=SWEEPS,
        burn_in=0, thin=1, seed=SEED, collapsed=CollapsedUConfig(every=CADENCE),
        store_labels=True)

    part1 = run_collapsed_u_chain(
        model=tiny_model(), start=tiny_state(), scales=SCALES,
        num_sweeps=CHECKPOINT_AT, burn_in=0, thin=1, seed=SEED,
        collapsed=CollapsedUConfig(every=CADENCE), store_labels=True,
        checkpoint_path=tmp_path, checkpoint_every=CHECKPOINT_AT)

    payload = json.loads((tmp_path / "chain0_checkpoint.json").read_text())
    assert payload["sweep"] == CHECKPOINT_AT
    restored = Stage6EState.from_dict(payload["state"])
    assert restored.iteration == CHECKPOINT_AT
    rng = np.random.default_rng(SEED)
    rng.bit_generator.state = restored.rng_state

    part2 = run_collapsed_u_chain(
        model=tiny_model(), start=restored, scales=SCALES, num_sweeps=SWEEPS,
        burn_in=0, thin=1, seed=SEED, collapsed=CollapsedUConfig(every=CADENCE),
        store_labels=True, rng=rng, state=restored)

    # draws: part1 + part2 must equal the uninterrupted run exactly
    n1 = len(part1.log_target)
    assert np.array_equal(full.u_draws[:n1], part1.u_draws)
    assert np.array_equal(full.u_draws[n1:], part2.u_draws)
    assert np.array_equal(full.log_target[n1:], part2.log_target)
    for name in full.scalars:
        assert np.array_equal(full.scalars[name][n1:], part2.scalars[name])
    assert full.boundary_keys[n1:] == part2.boundary_keys
    assert np.array_equal(full.segment_counts[n1:], part2.segment_counts)

    # final state: everything the checkpoint carries, bit for bit. `cache_version` is
    # the block scorer's internal cache counter — it restarts with every fresh sampler
    # object and carries no mathematical state, exactly as in the Step 7B2 resume.
    a, b = full.final_state.to_dict(), part2.final_state.to_dict()
    a.pop("cache_version"), b.pop("cache_version")
    assert a == b
    assert a["rng_state"] == b["rng_state"]
    assert a["iteration"] == SWEEPS
    assert a["proposed"][MOVE_NAME] == b["proposed"][MOVE_NAME]

    # the collapsed schedule kept its phase across the resume
    full_schedule = [r["sweep"] for r in full.collapsed_records]
    resumed_schedule = ([r["sweep"] for r in part1.collapsed_records]
                        + [r["sweep"] for r in part2.collapsed_records])
    assert full_schedule == resumed_schedule == [4, 9, 14, 19]
