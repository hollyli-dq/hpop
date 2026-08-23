"""Section 10 of the confirmatory preregistration: held-out negative log likelihood.

    PYTHONPATH=src python scripts/confirmatory_heldout_nll.py

The estimator is fixed by the preregistration and is reproduced here verbatim. For each of
the 45 held-out traces, using production draws only,

    NLL_n = -log( (1/M) * sum_{m=1..M} Z_n(U^(m), pi^(m), P^(m)) )

with `Z_n` the exact all-segmentation marginal likelihood of trace `n` from the frozen
semi-Markov forward recursion -- the same `log_normalizer` the sampler produces -- and `M`
a systematic subsample of **1,000 draws per chain, 4,000 per arm**, at equal spacing
through the production phase.

Computed in the log domain as `logsumexp_m(log Z_n^(m)) - log M`, which is the same number
without overflowing.

## What is fixed here and could not be chosen afterwards

Subsample size and spacing are preregistered. The one degree of freedom the text leaves is
the offset of the systematic sample, taken as the canonical stride-20 grid starting at the
first retained draw: indices 0, 20, ..., 19980 of the 20,000 retained draws. Recorded in
the output so it cannot be quietly changed.

Both arms are computed by identical code on identical traces, and the held-out traces were
never used for inference.

## Supplementary, and labelled as such

The preregistration asks for the two arms. A posterior predictive number is hard to read
without a scale, so the same estimator is also evaluated at the **unsealed truth
parameters** (a single point, not a posterior average). That is descriptive context, it is
**not preregistered**, and it is reported separately from the registered quantities. It
cannot influence the formal verdict, which was fixed before the seal was opened.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")           # one thread: the number must be reproducible

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.full_latent_constants import EPSILON                  # noqa: E402
from hpop.mcmc_original.matched_full_latent import FullLatentFixed            # noqa: E402
from hpop.mcmc_original.semi_markov_ffbs import forward as reference_forward  # noqa: E402
from hpop.mcmc_original.stage6e_frozen import (DELTA_B, MAX_BLOCK_WIDTH,      # noqa: E402
                                               MIN_BLOCK_WIDTH, N_ROLES, N_SKILLS)
from hpop.mcmc_original.stage6e_state import Stage6EModel, Stage6EState       # noqa: E402
from hpop.mcmc_original.transitions import log_transition_matrix              # noqa: E402
from hpop.mcmc_original.types import Segment, Segmentation                    # noqa: E402
from hpop.mcmc_original.collapsed_u_likelihood import CollapsedULikelihood    # noqa: E402
from hpop.mcmc_optimized.likelihood import BatchedCollapsedULikelihood        # noqa: E402

RUN = ROOT / "results" / "mcmc_optimized" / "confirmatory_run"
CORPUS = ROOT / "results" / "mcmc_optimized" / "confirmatory_corpus"
OUT = RUN / "heldout_nll"

ARMS = {"FULL-COND": "full_cond", "FULL-MARG": "full_marg"}
N_CHAINS = 4
DRAWS_PER_CHAIN = 1000            # preregistered
STRIDE_OFFSET = 0                 # canonical systematic grid
BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 6_300_010

# The primary number comes from the FROZEN recursion in `hpop.mcmc_original`, because
# Section 10 names it. The optimized backend agrees to ~1e-14 and is twenty times faster,
# but "we used the frozen recursion" has to be literally true, so the fast path is used
# only as a cross-check on a systematic subsample and its discrepancy is recorded.
CROSS_CHECK_EVERY = 40


# ------------------------------------------------------------------------- corpus
def load_heldout() -> tuple:
    payload = np.load(CORPUS / "heldout_traces.npz", allow_pickle=True)
    count = int(payload["n_traces"][0])
    traces = tuple(tuple(int(v) for v in payload[f"t{n:03d}_cpa"]) for n in range(count))
    return traces


def legal_tiling(length: int, index: int) -> Segmentation:
    """A legal cover. `log Z` does not read it; the state object requires one."""
    remaining, widths = int(length), []
    while remaining > MAX_BLOCK_WIDTH:
        step = (MAX_BLOCK_WIDTH if remaining - MAX_BLOCK_WIDTH >= MIN_BLOCK_WIDTH
                else remaining - MIN_BLOCK_WIDTH)
        widths.append(step)
        remaining -= step
    widths.append(remaining)
    segments, start = [], 0
    for position, width in enumerate(widths):
        segments.append(Segment(start, start + width, (index + position) % N_SKILLS))
        start += width
    return Segmentation(tuple(segments))


def build_model(traces) -> Stage6EModel:
    return Stage6EModel(traces=traces, epsilon=float(EPSILON), delta_b=float(DELTA_B),
                        n_skills=N_SKILLS, n_roles=N_ROLES,
                        min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
                        infer_pi_P=True, eta_initial=1.0, eta_transition=1.0)


def make_state(model, fixed, u, pi, p) -> Stage6EState:
    return Stage6EState(
        segmentations=tuple(legal_tiling(len(t), n)
                            for n, t in enumerate(model.traces)),
        u_by_skill=np.asarray(u, dtype=float), rho=float(fixed.rho_0),
        beta=float(fixed.beta), omega=float(fixed.omega),
        lambda_rep=float(fixed.lambda_rep), lambda_back=float(fixed.lambda_back),
        pi=np.asarray(pi, dtype=float), transition=np.asarray(p, dtype=float))


# --------------------------------------------------------------------- the estimator
def subsample_indices(retained: int) -> np.ndarray:
    stride = retained // DRAWS_PER_CHAIN
    index = np.arange(STRIDE_OFFSET, retained, stride)[:DRAWS_PER_CHAIN]
    if index.size != DRAWS_PER_CHAIN:
        raise AssertionError(f"expected {DRAWS_PER_CHAIN} draws, got {index.size}")
    return index


def log_z_for_arm(arm_prefix: str, model, fixed) -> tuple:
    """`log Z_n` for every subsampled draw of every chain in one arm."""
    likelihood = CollapsedULikelihood(model=model)               # frozen: the primary
    cross = BatchedCollapsedULikelihood(model=model)             # optimized: cross-check
    worst_cross_check, n_cross = 0.0, 0
    rows, provenance = [], []
    for chain in range(N_CHAINS):
        payload = np.load(RUN / "chains" / f"{arm_prefix}_{chain}.npz", allow_pickle=True)
        meta = json.loads(str(payload["meta"]))
        u_draws, pi_draws, p_draws = (payload["u_draws"], payload["pi_draws"],
                                      payload["p_draws"])
        retained = int(u_draws.shape[0])
        if retained != int(meta["retained_draws"]):
            raise AssertionError("retained draw count disagrees with the chain metadata")
        index = subsample_indices(retained)
        began = time.perf_counter()
        for position, draw in enumerate(index):
            state = make_state(model, fixed, u_draws[draw], pi_draws[draw],
                               p_draws[draw])
            values = np.asarray(likelihood.log_z_per_trace(state), dtype=float).copy()
            rows.append(values)
            if position % CROSS_CHECK_EVERY == 0:
                other = np.asarray(cross.log_z_per_trace(state), dtype=float)
                worst_cross_check = max(worst_cross_check,
                                        float(np.max(np.abs(values - other))))
                n_cross += 1
            if position % 200 == 0:
                print(f"    {arm_prefix} chain {chain}: {position}/{len(index)} "
                      f"({time.perf_counter() - began:.0f}s)", flush=True)
        provenance.append({
            "chain": chain, "retained_draws": retained,
            "burn_in_discarded": int(meta["burn_in"]), "thin": int(meta["thin"]),
            "seed": int(meta["seed"]),
            "subsample_indices_first": int(index[0]),
            "subsample_indices_last": int(index[-1]),
            "subsample_stride": int(index[1] - index[0]),
            "n_subsampled": int(index.size),
        })
    return np.vstack(rows), provenance, {"worst_abs_difference": worst_cross_check,
                                         "draws_cross_checked": n_cross}


def nll_from_log_z(log_z: np.ndarray) -> np.ndarray:
    """`-log mean_m Z_n`, in the log domain. `log_z` is (draws, traces)."""
    draws = log_z.shape[0]
    top = log_z.max(axis=0)
    return -(top + np.log(np.exp(log_z - top).sum(axis=0)) - np.log(draws))


def bootstrap_over_traces(per_trace: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = per_trace.size
    draws = rng.integers(0, n, size=(BOOTSTRAP, n))
    means = per_trace[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(per_trace.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
             "total": float(per_trace.sum()),
             "total_ci_lo": float(lo * n), "total_ci_hi": float(hi * n)}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, seed: int) -> dict:
    """Paired over traces: the two arms are evaluated on the same 45 traces."""
    rng = np.random.default_rng(seed)
    difference = a - b
    n = difference.size
    draws = rng.integers(0, n, size=(BOOTSTRAP, n))
    means = difference[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean_difference_per_trace": float(difference.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "total_difference": float(difference.sum()),
            "n_traces_favouring_second": int((difference > 0).sum()),
            "n_traces": int(n)}


def main() -> None:
    began = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    traces = load_heldout()
    model = build_model(traces)
    fixed = FullLatentFixed()
    occurrences = int(sum(len(t) for t in traces))
    print(f"held-out: {len(traces)} traces, {occurrences} CPA occurrences", flush=True)

    results, provenance, per_trace = {}, {}, {}
    for arm, prefix in ARMS.items():
        print(f"  {arm} ...", flush=True)
        log_z, chains, cross_check = log_z_for_arm(prefix, model, fixed)
        if not np.isfinite(log_z).all():
            raise ValueError(f"{arm}: a non-finite log Z was produced")
        nll = nll_from_log_z(log_z)
        per_trace[arm] = nll
        provenance[arm] = chains
        results[arm] = {
            "n_draws_used": int(log_z.shape[0]),
            "per_trace_mean_nll": float(nll.mean()),
            "total_nll": float(nll.sum()),
            "nll_per_cpa_occurrence": float(nll.sum() / occurrences),
            "bootstrap_over_traces": bootstrap_over_traces(nll, BOOTSTRAP_SEED),
            "log_z_mean_over_draws_and_traces": float(log_z.mean()),
            "engine": "frozen hpop.mcmc_original.collapsed_u_likelihood (Section 10 "
                      "names the frozen semi-Markov forward recursion)",
            "optimized_backend_cross_check": {
                **cross_check,
                "tolerance": 1e-10,
                "PASS": bool(cross_check["worst_abs_difference"] <= 1e-10),
            },
        }
        print(f"    total {nll.sum():.3f}   per-trace {nll.mean():.4f}   "
              f"per-occurrence {nll.sum() / occurrences:.4f}", flush=True)

    # --- supplementary, NOT preregistered: the same estimator at the unsealed truth ---
    truth = json.loads((CORPUS / "truth_SEALED.json").read_text())
    supplementary = None
    try:
        likelihood = CollapsedULikelihood(model=model)
        state = make_state(model, fixed, np.asarray(truth["u_by_skill"], dtype=float),
                           np.asarray(truth["pi"], dtype=float),
                           np.asarray(truth["transition"], dtype=float))
        truth_log_z = np.asarray(likelihood.log_z_per_trace(state), dtype=float)
        truth_nll = -truth_log_z
        supplementary = {
            "STATUS": "SUPPLEMENTARY, NOT PREREGISTERED, DESCRIPTIVE ONLY",
            "what": "the same held-out marginal likelihood evaluated at the unsealed "
                    "truth parameters -- a single point, not a posterior average, so it "
                    "is a reference scale and not a competitor to either arm",
            "per_trace_mean_nll": float(truth_nll.mean()),
            "total_nll": float(truth_nll.sum()),
            "nll_per_cpa_occurrence": float(truth_nll.sum() / occurrences),
        }
        print(f"  [supplementary] truth-parameter total {truth_nll.sum():.3f}",
              flush=True)
    except (KeyError, ValueError) as error:
        supplementary = {"STATUS": "unavailable", "reason": str(error)}

    contrast = paired_bootstrap(per_trace["FULL-COND"], per_trace["FULL-MARG"],
                                BOOTSTRAP_SEED + 1)

    report = {
        "preregistration": "results/mcmc_optimized/confirmatory_prereg/"
                           "PREREG_CONFIRMATORY.md, Section 10",
        "estimator": "NLL_n = -log( (1/M) sum_m Z_n(U^(m), pi^(m), P^(m)) ), computed in "
                     "the log domain as -(logsumexp_m log Z_n^(m) - log M)",
        "log_z_source": "exact all-segmentation marginal likelihood from the frozen "
                        "semi-Markov forward recursion, over the candidate block tables "
                        "of the held-out corpus",
        "subsample": {
            "draws_per_chain": DRAWS_PER_CHAIN, "chains_per_arm": N_CHAINS,
            "draws_per_arm": DRAWS_PER_CHAIN * N_CHAINS,
            "spacing": "systematic, stride 20 through the 20,000 retained production "
                       f"draws, offset {STRIDE_OFFSET}",
            "warm_up_discarded": 50_000, "production_sweeps": 100_000, "thin": 5,
        },
        "held_out": {"n_traces": len(traces), "cpa_occurrences": occurrences,
                     "never_used_for_inference": True},
        "model": {"n_skills": N_SKILLS, "n_roles": N_ROLES,
                  "min_width": MIN_BLOCK_WIDTH, "max_width": MAX_BLOCK_WIDTH,
                  "delta_b": float(DELTA_B), "epsilon": float(EPSILON)},
        "arms": results,
        "contrast_FULL_COND_minus_FULL_MARG": {
            **contrast,
            "NOTE": "the contrast implied by computing both arms; the preregistration "
                    "fixes the estimator and both arms, and does not separately register "
                    "this difference. Lower NLL is better, so a positive difference "
                    "favours FULL-MARG.",
        },
        "supplementary_truth_parameter_reference": supplementary,
        "chain_provenance": provenance,
        "bootstrap": {"resamples": BOOTSTRAP, "seed": BOOTSTRAP_SEED,
                      "over": "the 45 held-out traces"},
        "seconds": time.time() - began,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT / "heldout_nll.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    with (OUT / "heldout_nll_per_trace.csv").open("w") as handle:
        handle.write("trace,J,nll_full_cond,nll_full_marg,difference_cond_minus_marg\n")
        for n, trace in enumerate(traces):
            handle.write(f"{n},{len(trace)},{per_trace['FULL-COND'][n]:.9f},"
                         f"{per_trace['FULL-MARG'][n]:.9f},"
                         f"{per_trace['FULL-COND'][n] - per_trace['FULL-MARG'][n]:.9f}\n")
    print(f"\nwrote {OUT}/heldout_nll.json  ({report['seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
