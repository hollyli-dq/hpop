"""FULL-LATENT steps E/G — the DP factorisation and same-length batching, in isolation.

    PYTHONPATH=src python scripts/full_latent_dp_factorisation_bench.py [checkpoint.npz]

Measurement only.  NOTHING in `src/` is edited and the production recursion is not
replaced: the alternatives below are private functions in this file, benchmarked against
`semi_markov_ffbs.forward` on copies of real checkpoint tables and on synthetic tables.

E  the current recursion computes, for every (b, k), a term for every (a, h) pair:
       alpha[b,k] = LSE_{a,h} [ alpha[a,h] + logP[h,k] + score[a,b,k] + width(a,b) ]
   which is O(J D K^2).  Nothing inside the h-sum depends on b, so with
       r[a,k] = LSE_h ( alpha[a,h] + logP[h,k] )        <- O(J K^2) for every a, once
       alpha[b,k] = LSE_a ( r[a,k] + score[a,b,k] + width(a,b) )   <- O(J D K)
   the same numbers come out of O(J K^2 + J D K) work.  `r` is the log-domain form of
   r_a = alpha[a-1,:] @ P.  Equality is asserted, not assumed.

G  the corpus has four length classes of 25 traces each, so the trace axis vectorises
   exactly.  The batched variant runs the same factorised recursion over all traces of
   one length at once.

Timings take the MINIMUM over repeats; an 8-worker formal chain is live on this box.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_full_latent as mfl                     # noqa: E402
from hpop.mcmc_original import semi_markov_ffbs as smf                        # noqa: E402
from hpop.mcmc_original.transitions import log_transition_matrix              # noqa: E402

CORPUS_DIR = ROOT / "results" / "mcmc_original" / "matched_synthetic_formal_corpus"
OUT = ROOT / "results" / "mcmc_original" / "full_latent_perf"
SEED = 909_112_002
NEG = -np.inf


def _lse(a, axis=None):
    """logsumexp that returns -inf for an all -inf slice without warning."""
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        out = logsumexp(a, axis=axis)
    return out


# ------------------------------------------------------- E: the factorised recursion
def forward_factorised(log_block_scores, log_initial_probs, log_transition_matrix_,
                       boundary_prob: float, max_width: int, min_width: int = 1):
    """Mathematically identical to `smf.forward`, at O(J K^2 + J D K)."""
    scores = np.asarray(log_block_scores, dtype=float)
    log_pi = np.asarray(log_initial_probs, dtype=float)
    log_p = np.asarray(log_transition_matrix_, dtype=float)
    J, _, K = scores.shape
    log_db = math.log(float(boundary_prob))
    log_1mdb = math.log1p(-float(boundary_prob))

    alpha = np.full((J + 1, K), NEG)
    r = np.full((J + 1, K), NEG)          # r[a, k] = LSE_h(alpha[a,h] + logP[h,k])
    for b in range(1, J + 1):
        lowest = max(0, b - int(max_width))
        lo, hi = max(1, lowest), b - int(min_width)
        if hi >= lo:
            a_idx = np.arange(lo, hi + 1)
            width_pen = (b - a_idx - 1) * log_1mdb          # (n_a,)
            terms = (r[lo:hi + 1, :] + scores[lo:hi + 1, b, :]
                     + log_db + width_pen[:, None])          # (n_a, K)
            row = _lse(terms, axis=0)
        else:
            row = np.full(K, NEG)
        if lowest == 0 and int(min_width) <= b:
            initial = log_pi + scores[0, b, :] + (b - 1) * log_1mdb
            row = np.logaddexp(row, initial)
        alpha[b, :] = row
        r[b, :] = _lse(alpha[b, :, None] + log_p, axis=0)    # O(K^2), once per b

    log_z = float(_lse(alpha[J])) if np.isfinite(alpha[J]).any() else NEG
    return alpha, log_z


# --------------------------------------------------------- G: the batched recursion
def forward_batched(scores_stack, log_initial_probs, log_transition_matrix_,
                    boundary_prob: float, max_width: int, min_width: int = 1):
    """The same factorised recursion over a stack of equal-length traces at once.

    `scores_stack` is (B, J, J+1, K); returns alpha (B, J+1, K) and log_z (B,)."""
    scores = np.asarray(scores_stack, dtype=float)
    log_pi = np.asarray(log_initial_probs, dtype=float)
    log_p = np.asarray(log_transition_matrix_, dtype=float)
    B, J, _, K = scores.shape
    log_db = math.log(float(boundary_prob))
    log_1mdb = math.log1p(-float(boundary_prob))

    alpha = np.full((B, J + 1, K), NEG)
    r = np.full((B, J + 1, K), NEG)
    for b in range(1, J + 1):
        lowest = max(0, b - int(max_width))
        lo, hi = max(1, lowest), b - int(min_width)
        if hi >= lo:
            a_idx = np.arange(lo, hi + 1)
            width_pen = (b - a_idx - 1) * log_1mdb
            terms = (r[:, lo:hi + 1, :] + scores[:, lo:hi + 1, b, :]
                     + log_db + width_pen[None, :, None])    # (B, n_a, K)
            row = _lse(terms, axis=1)                        # (B, K)
        else:
            row = np.full((B, K), NEG)
        if lowest == 0 and int(min_width) <= b:
            initial = log_pi[None, :] + scores[:, 0, b, :] + (b - 1) * log_1mdb
            row = np.logaddexp(row, initial)
        alpha[:, b, :] = row
        r[:, b, :] = _lse(alpha[:, b, :, None] + log_p[None, :, :], axis=1)

    log_z = _lse(alpha[:, J, :], axis=1)
    return alpha, log_z


# ------------------------------------------------------------------------- fixtures
def real_tables(checkpoint: Path | None):
    corpus = mfl.load_frozen_observed_corpus(CORPUS_DIR)
    fixed = mfl.FullLatentFixed()
    model = mfl.build_full_latent_model(corpus.train, fixed)
    config = mfl.FullLatentConfig(arm=mfl.FULL_COND, structural_cadence=10,
                                  structural_scale=0.5, table_source="batched")
    sampler = mfl.FullLatentSampler(model=model, fixed=fixed, config=config)
    if checkpoint is not None and Path(checkpoint).exists():
        state = mfl.FullLatentChain.load(checkpoint, sampler).state.copy()
        origin = f"checkpoint:{checkpoint.name}"
    else:
        pi, transition = mfl.draw_initial_pi_p(model, SEED)
        state = mfl.initial_full_latent_state(
            model, mfl.make_u_start(0, SEED, 1.0, fixed, model), pi, transition)
        origin = "fresh_start"
    sampler.tables.refresh(state)
    tables = [np.array(t, dtype=float, copy=True)
              for t in sampler.tables.tables_for(state)]
    return {"model": model, "tables": tables, "origin": origin,
            "log_pi": np.log(state.pi),
            "log_p": log_transition_matrix(state.transition)}


def synthetic_table(J: int, K: int, D: int, Dmin: int, rng):
    scores = rng.normal(-2.0, 1.0, size=(J, J + 1, K))
    for a in range(J):
        for b in range(J + 1):
            if not (Dmin <= b - a <= D):
                scores[a, b, :] = NEG
    log_pi = np.log(np.full(K, 1.0 / K))
    log_p = np.log(rng.dirichlet(np.ones(K), size=K))
    return scores, log_pi, log_p


def best_of(fn, repeats: int = 3) -> float:
    best = math.inf
    for _ in range(repeats):
        began = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - began)
    return best


# ------------------------------------------------------------------------ E on real
def part_e_real(fx) -> dict:
    model = fx["model"]
    D, Dmin, db = model.max_width, model.min_width, model.delta_b
    worst = 0.0
    for table in fx["tables"]:
        chart = smf.forward(table, fx["log_pi"], fx["log_p"], db, D, Dmin)
        alpha2, z2 = forward_factorised(table, fx["log_pi"], fx["log_p"], db, D, Dmin)
        finite = np.isfinite(chart.alpha)
        worst = max(worst, float(np.max(np.abs(chart.alpha[finite] - alpha2[finite]))))
        worst = max(worst, abs(chart.log_normalizer - z2))
        if not np.array_equal(finite, np.isfinite(alpha2)):
            raise AssertionError("factorised recursion disagrees on the -inf pattern")

    t_cur = best_of(lambda: [smf.forward(t, fx["log_pi"], fx["log_p"], db, D, Dmin)
                             for t in fx["tables"]])
    t_fac = best_of(lambda: [forward_factorised(t, fx["log_pi"], fx["log_p"], db, D, Dmin)
                             for t in fx["tables"]])
    return {"n_traces": len(fx["tables"]), "max_abs_disagreement": worst,
            "current_seconds": t_cur, "factorised_seconds": t_fac,
            "speedup": t_cur / t_fac}


# --------------------------------------------------------------- E across K (synthetic)
def part_e_scaling(J: int, D: int, Dmin: int, K_values) -> dict:
    rng = np.random.default_rng(SEED)
    rows = {}
    for K in K_values:
        scores, log_pi, log_p = synthetic_table(J, K, D, Dmin, rng)
        chart = smf.forward(scores, log_pi, log_p, 0.15, D, Dmin)
        alpha2, z2 = forward_factorised(scores, log_pi, log_p, 0.15, D, Dmin)
        finite = np.isfinite(chart.alpha)
        disagree = float(np.max(np.abs(chart.alpha[finite] - alpha2[finite])))
        t_cur = best_of(lambda: smf.forward(scores, log_pi, log_p, 0.15, D, Dmin))
        t_fac = best_of(lambda: forward_factorised(scores, log_pi, log_p, 0.15, D, Dmin))

        tracemalloc.start()
        forward_factorised(scores, log_pi, log_p, 0.15, D, Dmin)
        _, peak_fac = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.start()
        smf.forward(scores, log_pi, log_p, 0.15, D, Dmin)
        _, peak_cur = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rows[str(K)] = {
            "current_seconds": t_cur, "factorised_seconds": t_fac,
            "speedup": t_cur / t_fac, "max_abs_disagreement": disagree,
            "ops_current_JDK2": J * D * K * K,
            "ops_factorised_JK2_plus_JDK": J * K * K + J * D * K,
            "op_ratio": (J * D * K * K) / (J * K * K + J * D * K),
            "peak_bytes_current": int(peak_cur), "peak_bytes_factorised": int(peak_fac),
            "lse_calls_current_JK": J * K, "lse_calls_factorised_2J": 2 * J,
        }
    ks = np.array([float(k) for k in rows])
    cur = np.array([rows[k]["current_seconds"] for k in rows])
    fac = np.array([rows[k]["factorised_seconds"] for k in rows])
    return {"J": J, "D": D, "per_K": rows,
            "empirical_exponent_current": float(np.polyfit(np.log(ks), np.log(cur), 1)[0]),
            "empirical_exponent_factorised": float(np.polyfit(np.log(ks), np.log(fac), 1)[0])}


# ------------------------------------------------------------------- G: batching
def part_g(fx) -> dict:
    model = fx["model"]
    D, Dmin, db = model.max_width, model.min_width, model.delta_b
    by_length = defaultdict(list)
    for table in fx["tables"]:
        by_length[table.shape[0]].append(table)

    out = {}
    for J, group in sorted(by_length.items()):
        stack = np.stack(group)
        _, z_batched = forward_batched(stack, fx["log_pi"], fx["log_p"], db, D, Dmin)
        z_seq = np.array([smf.forward(t, fx["log_pi"], fx["log_p"], db, D, Dmin
                                      ).log_normalizer for t in group])
        disagree = float(np.max(np.abs(z_batched - z_seq)))

        t_seq = best_of(lambda: [smf.forward(t, fx["log_pi"], fx["log_p"], db, D, Dmin)
                                 for t in group])
        t_fac = best_of(lambda: [forward_factorised(t, fx["log_pi"], fx["log_p"], db,
                                                    D, Dmin) for t in group])
        t_bat = best_of(lambda: forward_batched(stack, fx["log_pi"], fx["log_p"], db,
                                                D, Dmin))
        tracemalloc.start()
        forward_batched(stack, fx["log_pi"], fx["log_p"], db, D, Dmin)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        out[str(J)] = {
            "n_traces": len(group), "J": J,
            "sequential_current_s": t_seq, "sequential_factorised_s": t_fac,
            "batched_factorised_s": t_bat,
            "speedup_batched_vs_current": t_seq / t_bat,
            "speedup_batched_vs_factorised": t_fac / t_bat,
            "max_abs_disagreement_logZ": disagree,
            "stack_bytes": int(stack.nbytes),
            "batched_peak_traced_bytes": int(peak),
        }
    total_cur = sum(v["sequential_current_s"] for v in out.values())
    total_bat = sum(v["batched_factorised_s"] for v in out.values())
    return {"per_length_class": out, "all_traces_current_s": total_cur,
            "all_traces_batched_s": total_bat, "overall_speedup": total_cur / total_bat}


def main() -> None:
    checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    fx = real_tables(checkpoint)
    model = fx["model"]
    report = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "state_origin": fx["origin"], "loadavg": list(os.getloadavg()),
        "dims": {"K": model.n_skills, "max_width": model.max_width,
                 "min_width": model.min_width, "n_traces": len(fx["tables"])},
        "E_real_corpus": part_e_real(fx),
        "E_scaling_in_K": part_e_scaling(J=40, D=model.max_width,
                                         Dmin=model.min_width,
                                         K_values=(3, 5, 10, 20)),
        "G_batching": part_g(fx),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "dp_factorisation_bench.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
