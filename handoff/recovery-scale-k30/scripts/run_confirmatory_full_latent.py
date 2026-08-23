"""Launch the preregistered optimized FULL-LATENT confirmatory experiment.

    PYTHONPATH=src python scripts/run_confirmatory_full_latent.py --launch --workers 8

Frozen by `results/mcmc_optimized/confirmatory_prereg/PREREG_CONFIRMATORY.md`:
50,000 warm-up sweeps discarded entirely, then 100,000 production sweeps, thin 5,
one terminal gate, no adaptive stopping, no ladder.

Truth stays SEALED: this runner never opens `truth_SEALED.json`, and the corpus loader it
uses reads only the observed CPA arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
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
from hpop.mcmc_optimized import OptimizedFullLatentSampler                    # noqa: E402
from hpop.mcmc_optimized.chain import OptimizedFullLatentChain                # noqa: E402

CORPUS_DIR = ROOT / "results" / "mcmc_optimized" / "confirmatory_corpus"
OUT = ROOT / "results" / "mcmc_optimized" / "confirmatory_run"
CHAIN_DIR = OUT / "chains"

# ------------------------------------------------------------------ frozen schedule
WARMUP = 50_000
PRODUCTION = 100_000
TOTAL = WARMUP + PRODUCTION
THIN = 5
CHECKPOINT_EVERY = 2_000
STRUCTURAL_CADENCE = 10
STRUCTURAL_SCALE = 0.5
TABLE_SOURCE = "batched"

U_SEEDS = (6_304_101, 6_304_102, 6_304_103, 6_304_104)
PIP_SEEDS = (6_306_101, 6_306_102, 6_306_103, 6_306_104)
SCALES = (0.5, 1.0, 2.0, 3.0)
CHAIN_SEEDS = {mfl.FULL_COND: (6_306_201, 6_306_202, 6_306_203, 6_306_204),
               mfl.FULL_MARG: (6_306_211, 6_306_212, 6_306_213, 6_306_214)}

BACKEND_COMMIT = "564995efd056d7d33984f0ca1532386e6140ea0c"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _chain_path(arm: str, index: int) -> Path:
    return CHAIN_DIR / f"{arm.lower().replace('-', '_')}_{index}.npz"


def build(arm: str, index: int):
    corpus = mfl.load_frozen_observed_corpus(CORPUS_DIR)
    fixed = mfl.FullLatentFixed()
    model = mfl.build_full_latent_model(corpus.train, fixed)
    config = mfl.FullLatentConfig(arm=arm, structural_cadence=STRUCTURAL_CADENCE,
                                  structural_scale=STRUCTURAL_SCALE,
                                  table_source=TABLE_SOURCE)
    sampler = OptimizedFullLatentSampler(model=model, fixed=fixed, config=config)
    u_start = mfl.make_u_start(index, U_SEEDS[index], SCALES[index], fixed,
                               model.n_skills, model.n_roles)
    pi, transition = mfl.draw_initial_pi_p(model, PIP_SEEDS[index])
    state = mfl.initial_full_latent_state(model, u_start, pi, transition, fixed)
    probes = mfl.select_truth_free_probes(model.traces, corpus.corpus_hash)
    chain = OptimizedFullLatentChain(
        sampler=sampler, start=state, seed=CHAIN_SEEDS[arm][index],
        burn_in=WARMUP, thin=THIN, probes=probes, warmup_sweeps=WARMUP,
        start_metadata={"u_seed": U_SEEDS[index], "u_scale": SCALES[index],
                        "pip_seed": PIP_SEEDS[index], "start_index": index,
                        "corpus_hash": corpus.corpus_hash})
    return chain


def run_one(task):
    arm, index = task
    chain = build(arm, index)
    path = _chain_path(arm, index)
    chain.advance(TOTAL, checkpoint_path=path, checkpoint_every=CHECKPOINT_EVERY,
                  progress_every=10_000)
    pre = chain.preconditions()
    (OUT / "preconditions" / f"{arm.lower().replace('-','_')}_{index}.json").write_text(
        json.dumps({"arm": arm, "index": index, "seed": chain.seed, **pre},
                   indent=2, sort_keys=True) + "\n")
    return {"arm": arm, "index": index, "seed": chain.seed,
            "sweeps": int(chain.state.iteration),
            "retained_draws": int(chain.retained_draws),
            "seconds": float(chain.seconds), **pre}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    CHAIN_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "preconditions").mkdir(parents=True, exist_ok=True)
    if any(CHAIN_DIR.glob("*.npz")):
        raise SystemExit("confirmatory chains already exist; refusing to relaunch")

    corpus = mfl.load_frozen_observed_corpus(CORPUS_DIR)
    seal = json.loads((CORPUS_DIR / "truth_SEAL.json").read_text())
    launch = {
        "launched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend_commit": BACKEND_COMMIT,
        "prereg_sha256": _sha(ROOT / "results" / "mcmc_optimized"
                              / "confirmatory_prereg" / "PREREG_CONFIRMATORY.md"),
        "corpus_hash_sha256": corpus.corpus_hash,
        "train_hash_sha256": corpus.train_hash,
        "heldout_hash_sha256": corpus.heldout_hash,
        "truth_hash_sha256": seal["truth_sha256"],
        "truth_sealed": True,
        "truth_file_opened_by_runner": False,
        "starts_manifest_sha256": _sha(CORPUS_DIR / "starts_manifest.json"),
        "probes_manifest_sha256": _sha(CORPUS_DIR / "probes_manifest.json"),
        "schedule": {"warmup_discarded": WARMUP, "production": PRODUCTION,
                     "total": TOTAL, "thin": THIN,
                     "checkpoint_every": CHECKPOINT_EVERY},
        "kernel": {"structural_cadence": STRUCTURAL_CADENCE,
                   "structural_scale": STRUCTURAL_SCALE,
                   "table_source": TABLE_SOURCE},
        "seeds": {"u": list(U_SEEDS), "pi_p": list(PIP_SEEDS),
                  "scales": list(SCALES),
                  "cond": list(CHAIN_SEEDS[mfl.FULL_COND]),
                  "marg": list(CHAIN_SEEDS[mfl.FULL_MARG])},
        "initialisation": "dispersed truth-free prior starts; NO old checkpoint and NO "
                          "truth-informed state",
        "orchestrator_pid": os.getpid(),
    }
    (OUT / "launch_record.json").write_text(json.dumps(launch, indent=2, sort_keys=True)
                                            + "\n")
    print(f"[CONFIRMATORY] orchestrator pid {os.getpid()}", flush=True)
    print(f"[CONFIRMATORY] {WARMUP:,} warm-up (discarded) + {PRODUCTION:,} production, "
          f"thin {THIN}", flush=True)

    tasks = [(arm, i) for arm in (mfl.FULL_COND, mfl.FULL_MARG) for i in range(4)]
    with mp.get_context("spawn").Pool(processes=args.workers) as pool:
        results = pool.map(run_one, tasks)

    summary = {"results": results,
               "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)
                                          + "\n")
    print("[CONFIRMATORY] all chains reached the terminal sweep", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
