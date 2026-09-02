"""The gates must be invariant under skill relabelling, and still catch real failures."""

from __future__ import annotations

import numpy as np
import pytest

from hpop.mcmc_cpa.recovery_gates import (chain_statistics, cocluster_probe_pairs,
                                          evaluate_cell)


def synthetic_draws(rng, n_draws=40, n_traces=4, k=3, perm=None):
    """Draws from one synthetic 'chain'; `perm` relabels skills throughout."""
    perm = list(range(k)) if perm is None else list(perm)
    labels, boundaries, u_draws, pi_sorted, spectrum = [], [], [], [], []
    for _ in range(n_draws):
        step_labels, step_bounds = [], []
        for _ in range(n_traces):
            n_seg = int(rng.integers(3, 6))
            raw = [int(rng.integers(k)) for _ in range(n_seg)]
            step_labels.append([perm[s] for s in raw])
            ends = sorted(rng.choice(np.arange(8, 120), size=n_seg - 1,
                                     replace=False).tolist())
            step_bounds.append([int(e) for e in ends])
        labels.append(step_labels)
        boundaries.append(step_bounds)
        u = rng.standard_normal((k, 10, 2))
        u_draws.append(u[np.argsort(perm)].tolist() if perm != list(range(k))
                       else u.tolist())
        pi = np.sort(rng.dirichlet(np.ones(k)))[::-1]
        pi_sorted.append(pi.tolist())
        spectrum.append(np.sort(rng.random(min(k, 5)))[::-1].tolist())
    return {"labels": labels, "boundaries": boundaries, "u": u_draws,
            "pi_sorted": pi_sorted, "p_spectrum": spectrum}


def test_statistics_are_invariant_under_relabelling():
    """The same chain with skills renamed must produce identical statistics."""
    pairs = cocluster_probe_pairs([128] * 4, 50, seed=1)
    base_rng = np.random.default_rng(7)
    plain = synthetic_draws(np.random.default_rng(7), perm=None)
    relabelled = synthetic_draws(np.random.default_rng(7), perm=[2, 0, 1])

    a = chain_statistics(plain, pairs)
    b = chain_statistics(relabelled, pairs)
    assert a["usage"] == b["usage"]
    assert a["cocluster"] == b["cocluster"]
    assert a["relation_sorted"] == b["relation_sorted"]
    assert a["canonical_digests"] == b["canonical_digests"]


def test_a_relabelled_chain_does_not_fail_the_cell():
    """Four chains holding the SAME posterior under different labellings must pass
    exactly as four identically-labelled chains do. This is the defect the earlier
    edge-level gate had, pinned as a requirement."""
    pairs = cocluster_probe_pairs([128] * 4, 50, seed=1)
    perms = [None, [1, 2, 0], [2, 0, 1], [0, 2, 1]]
    stats = [chain_statistics(
        synthetic_draws(np.random.default_rng(100 + i), n_draws=60, perm=p), pairs)
        for i, p in enumerate(perms)]
    verdict_mixed = evaluate_cell(stats, acceptance_retained=0.3)
    stats_same = [chain_statistics(
        synthetic_draws(np.random.default_rng(100 + i), n_draws=60, perm=None), pairs)
        for i in range(4)]
    verdict_same = evaluate_cell(stats_same, acceptance_retained=0.3)
    assert verdict_mixed["passes"] == verdict_same["passes"]
    for key, value in verdict_same["numbers"].items():
        if isinstance(value, float):
            assert verdict_mixed["numbers"][key] == pytest.approx(value, abs=1e-9), key


def test_genuinely_different_posteriors_fail():
    """Chains drawing from DIFFERENT distributions must fail -- invariance must not
    have made the gate blind."""
    pairs = cocluster_probe_pairs([128] * 4, 50, seed=1)

    def biased(rng, hot):
        d = synthetic_draws(rng, n_draws=60)
        # chain-specific bias in relation counts via u scale
        d["u"] = [(np.asarray(u) * hot).tolist() for u in d["u"]]
        return d
    stats = [chain_statistics(biased(np.random.default_rng(200 + i), hot), pairs)
             for i, hot in enumerate((0.2, 0.2, 3.0, 3.0))]
    verdict = evaluate_cell(stats, acceptance_retained=0.3)
    assert not verdict["passes"]
    assert any("relation" in f or "modal" in f for f in verdict["failures"])


def test_acceptance_window_is_enforced():
    pairs = cocluster_probe_pairs([128] * 4, 50, seed=1)
    stats = [chain_statistics(
        synthetic_draws(np.random.default_rng(300 + i), n_draws=60), pairs)
        for i in range(4)]
    ok = evaluate_cell(stats, acceptance_retained=0.30)
    high = evaluate_cell(stats, acceptance_retained=0.75)
    assert "u_acceptance_window" in high["numbers"]
    assert any("acceptance" in f for f in high["failures"])
    assert not any("acceptance" in f for f in ok["failures"])


def test_insufficient_draws_fail_rather_than_pass():
    pairs = cocluster_probe_pairs([128] * 4, 20, seed=1)
    stats = [chain_statistics(
        synthetic_draws(np.random.default_rng(i), n_draws=6), pairs)
        for i in range(4)]
    verdict = evaluate_cell(stats, acceptance_retained=0.3)
    assert not verdict["passes"]
    assert any("insufficient" in f for f in verdict["failures"])


def test_probe_pairs_are_deterministic():
    a = cocluster_probe_pairs([100, 120], 30, seed=9)
    b = cocluster_probe_pairs([100, 120], 30, seed=9)
    c = cocluster_probe_pairs([100, 120], 30, seed=10)
    assert a == b and a != c
