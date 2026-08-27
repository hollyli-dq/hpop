"""FULL-LATENT step H — where the resident memory actually is, and how it extrapolates.

    PYTHONPATH=src python scripts/full_latent_memory_decomposition.py [checkpoint.npz]

Measurement only.  Nothing in `src/` is edited; no formal chain file is opened for
writing.  Live arrays are measured with `nbytes`, transient peaks with `tracemalloc`,
and the large grids are EXTRAPOLATED arithmetically rather than allocated.

The question the extrapolation answers is which of J, K, N actually drives the total.
The candidate block-score table is stored densely as (J, J+1, K) per trace, so it carries
a J^2 term, while only the band `min_width <= b - a <= max_width` can ever be finite --
so the same information has an O(J D K) banded form.  Both are reported, because the
difference is what decides whether large J is feasible at all.
"""

from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_full_latent as mfl                     # noqa: E402
from hpop.mcmc_original import semi_markov_ffbs as smf                        # noqa: E402
from hpop.mcmc_original.transitions import log_transition_matrix              # noqa: E402

CORPUS_DIR = ROOT / "results" / "mcmc_original" / "matched_synthetic_formal_corpus"
OUT = ROOT / "results" / "mcmc_original" / "full_latent_perf"
SEED = 909_112_002


def array_bytes(obj, seen=None) -> int:
    """Recursive nbytes over numpy arrays reachable from a container."""
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    total = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            total += array_bytes(k, seen) + array_bytes(v, seen)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            total += array_bytes(v, seen)
    elif hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            total += array_bytes(v, seen)
    elif hasattr(obj, "__slots__"):
        for name in obj.__slots__:
            total += array_bytes(getattr(obj, name, None), seen)
    return total


def main() -> None:
    checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    corpus = mfl.load_frozen_observed_corpus(CORPUS_DIR)
    fixed = mfl.FullLatentFixed()
    model = mfl.build_full_latent_model(corpus.train, fixed)
    config = mfl.FullLatentConfig(arm=mfl.FULL_COND, structural_cadence=10,
                                  structural_scale=0.5, table_source="batched")
    sampler = mfl.FullLatentSampler(model=model, fixed=fixed, config=config)
    if checkpoint is not None and Path(checkpoint).exists():
        chain = mfl.FullLatentChain.load(checkpoint, sampler)
        state, origin = chain.state.copy(), f"checkpoint:{checkpoint.name}"
    else:
        pi, transition = mfl.draw_initial_pi_p(model, SEED)
        state = mfl.initial_full_latent_state(
            model, mfl.make_u_start(0, SEED, 1.0, fixed, model), pi, transition)
        origin = "fresh_start"

    sampler.tables.refresh(state)
    tables = list(sampler.tables.tables_for(state))
    log_pi, log_p = np.log(state.pi), log_transition_matrix(state.transition)

    J_values = [len(t) for t in model.traces]
    K, D, Dmin, N = model.n_skills, model.max_width, model.min_width, len(model.traces)

    emission_bytes = sum(int(np.asarray(t).nbytes) for t in tables)
    charts = [smf.forward(t, log_pi, log_p, model.delta_b, D, Dmin) for t in tables]
    alpha_bytes = sum(int(c.alpha.nbytes) for c in charts)

    tracemalloc.start()
    smf.forward(tables[-1], log_pi, log_p, model.delta_b, D, Dmin)
    _, peak_one_forward = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rng = np.random.default_rng(SEED)
    tracemalloc.start()
    seg = smf.backward_sample(charts[-1], rng)
    _, peak_one_backward = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    measured = {
        "emission_tables_bytes": emission_bytes,
        "forward_alpha_bytes_all_traces": alpha_bytes,
        "state_arrays_bytes": array_bytes(state),
        "model_arrays_bytes": array_bytes(model),
        "sampler_tables_arrays_bytes": array_bytes(sampler.tables),
        "peak_traced_one_forward_bytes": int(peak_one_forward),
        "peak_traced_one_backward_bytes": int(peak_one_backward),
        "segments_last_trace": len(seg),
    }

    # ------------------------------------------------------------- extrapolation
    def dense_emission(J, K, N):
        return N * J * (J + 1) * K * 8

    def banded_emission(J, K, N, D=D, Dmin=Dmin):
        widths = max(0, D - Dmin + 1)
        per_trace = sum(max(0, J - w + 1) for w in range(Dmin, D + 1)) * K
        return N * per_trace * 8 if widths else 0

    def alpha_bytes_for(J, K, N):
        return N * (J + 1) * K * 8

    def batched_terms(J, K, B, D=D):
        return B * D * K * 8

    grid = {}
    for J in (50, 100, 200, 500):
        for Kv in (3, 10, 20):
            grid[f"J{J}_K{Kv}_N100"] = {
                "dense_emission_MB": dense_emission(J, Kv, 100) / 1e6,
                "banded_emission_MB": banded_emission(J, Kv, 100) / 1e6,
                "alpha_all_traces_MB": alpha_bytes_for(J, Kv, 100) / 1e6,
                "batched_terms_25traces_MB": batched_terms(J, Kv, 25) / 1e6,
                "dense_over_banded": (dense_emission(J, Kv, 100)
                                      / max(1, banded_emission(J, Kv, 100))),
            }

    report = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "state_origin": origin,
        "dims": {"N": N, "K": K, "max_width": D, "min_width": Dmin,
                 "J_values": sorted(set(J_values))},
        "measured_bytes": measured,
        "measured_check": {
            "dense_emission_predicted_bytes": sum(J * (J + 1) * K * 8 for J in J_values),
            "banded_emission_predicted_bytes":
                sum(sum(max(0, J - w + 1) for w in range(Dmin, D + 1)) * K * 8
                    for J in J_values),
            "alpha_predicted_bytes": sum((J + 1) * K * 8 for J in J_values),
        },
        "extrapolation_N100": grid,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "memory_decomposition.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
