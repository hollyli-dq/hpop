"""Parity gate: `hpop.mcmc_optimized` against `hpop.mcmc_original` on real chain states.

    PYTHONPATH=src python scripts/optimized_backend_parity.py <checkpoint_dir>

The equivalence unit tests use synthetic tables. This gate uses the actual FULL-LATENT
formal checkpoints, which is where the numbers have the real dynamic range. The reference
is the oracle; the optimized backend is the thing on trial.

Checkpoints are opened READ-ONLY. No formal chain file is written.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_full_latent as mfl                     # noqa: E402
from hpop.mcmc_original.semi_markov_ffbs import forward as reference_forward  # noqa: E402
from hpop.mcmc_original.transitions import log_transition_matrix              # noqa: E402
from hpop.mcmc_optimized import (COUNTERS, FLAGS, HashCachedFFBSBlockTables,  # noqa: E402
                                 forward_batched_group, forward_dispatch)

CORPUS_DIR = ROOT / "results" / "mcmc_original" / "matched_synthetic_formal_corpus"
OUT = ROOT / "results" / "mcmc_original" / "full_latent_perf"

CONFIGS = {
    "reference_algorithm": [],
    "O1_inline": ["inline_logsumexp"],
    "O2_cache": ["emission_hash_cache"],
    "O3_factorised": ["factorised_forward"],
    "O4_batched": ["batched_forward"],
    "O1+O2": ["inline_logsumexp", "emission_hash_cache"],
    "O1+O2+O3": ["inline_logsumexp", "emission_hash_cache", "factorised_forward"],
    "all_four": ["inline_logsumexp", "emission_hash_cache", "factorised_forward",
                 "batched_forward"],
}


def build(arm, checkpoint):
    corpus = mfl.load_frozen_observed_corpus(CORPUS_DIR)
    fixed = mfl.FullLatentFixed()
    model = mfl.build_full_latent_model(corpus.train, fixed)
    config = mfl.FullLatentConfig(arm=arm, structural_cadence=10, structural_scale=0.5,
                                  table_source="batched")
    sampler = mfl.FullLatentSampler(model=model, fixed=fixed, config=config)
    state = mfl.FullLatentChain.load(checkpoint, sampler).state.copy()
    sampler.tables.refresh(state)
    return model, sampler, state


def optimized_charts(model, tables, state):
    log_pi = np.log(state.pi)
    log_p = log_transition_matrix(state.transition)
    all_tables = list(tables.tables_for(state))
    if FLAGS.batched_forward:
        groups: dict = {}
        for n, table in enumerate(all_tables):
            groups.setdefault(np.asarray(table).shape[0], []).append(n)
        out = [None] * len(all_tables)
        for _length, members in sorted(groups.items()):
            for n, chart in zip(members, forward_batched_group(
                    [all_tables[n] for n in members], log_pi, log_p, model.delta_b,
                    model.max_width, model.min_width)):
                out[n] = chart
        return out
    return [forward_dispatch(t, log_pi, log_p, model.delta_b, model.max_width,
                             model.min_width) for t in all_tables]


def main() -> None:
    directory = Path(sys.argv[1])
    report = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "arms": {}}
    ok = True

    for arm, name in ((mfl.FULL_COND, "full_cond_0.npz"),
                      (mfl.FULL_MARG, "full_marg_0.npz")):
        model, sampler, state = build(arm, directory / name)
        reference = [reference_forward(t, np.log(state.pi),
                                       log_transition_matrix(state.transition),
                                       model.delta_b, model.max_width, model.min_width)
                     for t in sampler.tables.tables_for(state)]
        ref_alpha = [np.array(c.alpha, copy=True) for c in reference]
        ref_z = np.array([c.log_normalizer for c in reference])
        ref_tables = [np.array(t, copy=True) for t in sampler.tables.tables_for(state)]

        rows = {}
        for label, flags in CONFIGS.items():
            FLAGS.all_off()
            FLAGS.apply(**{f: True for f in flags})
            COUNTERS.reset()

            tables = HashCachedFFBSBlockTables(model=model,
                                               source=sampler.config.table_source)
            tables.refresh(state)
            tables.refresh(state)                      # exercise the cache
            bitwise = all(np.array_equal(a, b) and a.dtype == b.dtype
                          for a, b in zip(ref_tables, tables.tables_for(state)))

            got = optimized_charts(model, tables, state)
            worst_alpha, pattern_ok = 0.0, True
            for a_ref, chart in zip(ref_alpha, got):
                finite = np.isfinite(a_ref)
                if not np.array_equal(finite, np.isfinite(chart.alpha)):
                    pattern_ok = False
                worst_alpha = max(worst_alpha, float(
                    np.max(np.abs(a_ref[finite] - chart.alpha[finite]))))
            worst_z = float(np.max(np.abs(
                ref_z - np.array([c.log_normalizer for c in got]))))

            counters = COUNTERS.snapshot()
            expected = ({"emission_cache_hits"} if "emission_hash_cache" in flags else set())
            expected |= ({"forward_batched_groups"} if "batched_forward" in flags
                         else {"forward_factorised_calls"} if "factorised_forward" in flags
                         else {"forward_inline_calls"} if "inline_logsumexp" in flags
                         else {"forward_reference_calls"})
            fired = all(counters[k] > 0 for k in expected)

            passed = (pattern_ok and worst_alpha < 1e-9 and worst_z < 1e-9 and fired
                      and bitwise)
            ok = ok and passed
            rows[label] = {"max_abs_alpha_error": worst_alpha,
                           "max_abs_logZ_error": worst_z,
                           "inf_pattern_identical": pattern_ok,
                           "emission_tables_bitwise_identical": bitwise,
                           "expected_counters_fired": fired, "counters": counters,
                           "PASS": passed}
        FLAGS.reset()
        report["arms"][arm] = rows

    report["ALL_PASS"] = ok
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "optimized_backend_parity.json").write_text(json.dumps(report, indent=2,
                                                                  sort_keys=True))
    for arm, rows in report["arms"].items():
        print(f"\n=== {arm} ===")
        print(f"  {'config':<20}{'alpha err':>12}{'logZ err':>12}{'inf ok':>8}"
              f"{'tbl bitwise':>13}{'fired':>7}{'PASS':>7}")
        for label, r in rows.items():
            print(f"  {label:<20}{r['max_abs_alpha_error']:>12.2e}"
                  f"{r['max_abs_logZ_error']:>12.2e}{str(r['inf_pattern_identical']):>8}"
                  f"{str(r['emission_tables_bitwise_identical']):>13}"
                  f"{str(r['expected_counters_fired']):>7}{str(r['PASS']):>7}")
    print(f"\nALL_PASS = {ok}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
