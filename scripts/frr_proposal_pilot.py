"""Full Recurrent Release — truth-free proposal-scale pilot.

    PYTHONPATH=src python scripts/frr_proposal_pilot.py --sweeps 400

Truth-free by construction: the only quantities inspected are proposal acceptance,
invalid-proposal rate, numerical validity, runtime, and H-changing movement as a reported
liveness quantity. No distance from truth, parameter recovery, held-out NLL or structural
recovery is computed, and the sealed truth file is never opened.

Registered rule, from `recurrent_scalar_mcmc.tune_proposal_scale`: one pilot, acceptance
window (0.20, 0.55), at most ONE adjustment, and the adjustment is set from the pilot's own
posterior spread in the PROPOSAL coordinate (log for beta/lambda_*, identity for omega)
scaled by the optimal one-dimensional random-walk factor 2.38. Base scales are the
registered Stage 6D starting scales; they are re-piloted because proposal scales are
per-corpus. All pilot draws are discarded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_full_latent as mfl                     # noqa: E402
from hpop.mcmc_original.recurrent_scalar_mcmc import PROPOSAL_KIND            # noqa: E402
from hpop.mcmc_original.recurrent_scalar_posterior import PRIORS              # noqa: E402
from hpop.mcmc_original.stage6d_frozen import REGISTERED_SCALES, SCALAR_ORDER # noqa: E402
from hpop.mcmc_optimized.full_recurrent import (FullRecurrentFixed,           # noqa: E402
                                                FullRecurrentSampler, sweep_once)

CORP = ROOT / "results" / "full_recurrent_release" / "corpus"
OUT = ROOT / "results" / "full_recurrent_release" / "pilots"
WINDOW = (0.20, 0.55)
PILOT_SEED = 6_318_001
PILOT_START_SEED = 6_318_101
OPTIMAL_RWM = 2.38

# Registered dispersed recurrent starts (recurrent_scalar_mcmc.REGISTERED_STARTS),
# chain 0 column. Prior-based and truth-free.
PILOT_RECURRENT_START = {"beta": 0.5, "omega": -0.5, "lambda_rep": 0.15,
                         "lambda_back": 0.05}


def build(scales):
    corpus = mfl.load_frozen_observed_corpus(CORP)
    fixed_gen = mfl.FullLatentFixed()          # supplies the model's fixed epsilon/delta_b
    model = mfl.build_full_latent_model(corpus.train, fixed_gen)
    cfg = mfl.FullLatentConfig(arm=mfl.FULL_MARG, structural_cadence=10,
                               structural_scale=0.5, table_source="batched")
    sampler = FullRecurrentSampler(model=model, fixed=FullRecurrentFixed(),
                                   config=cfg, scalar_scales=dict(scales))
    u0 = mfl.make_u_start(0, 6_314_101, 0.5, fixed_gen, model.n_skills, model.n_roles)
    pi, P = mfl.draw_initial_pi_p(model, PILOT_START_SEED)
    state = mfl.initial_full_latent_state(model, u0, pi, P, fixed_gen)
    for name, value in PILOT_RECURRENT_START.items():
        setattr(state, name, float(value))
    return sampler, state, corpus


def run(scales, sweeps, seed):
    sampler, state, _ = build(scales)
    rng = np.random.default_rng(seed)
    acc = {n: 0 for n in SCALAR_ORDER}
    prop = {n: 0 for n in SCALAR_ORDER}
    draws = {n: [] for n in SCALAR_ORDER}
    u_prop = u_acc = u_invalid = h_changed = 0
    finite = True
    began = time.perf_counter()
    for _ in range(sweeps):
        state, info = sweep_once(state, sampler, rng)
        s = info["scalars"]
        for n in SCALAR_ORDER:
            prop[n] += s["proposed"][n]
            acc[n] += s["accepted"][n]
            draws[n].append(s["values"][n])
        rec = info["structural_record"]
        if rec is not None:
            u_prop += 1
            u_acc += int(rec["accepted"])
            u_invalid += int(rec["invalid"])
            h_changed += int(rec["accepted"] and rec["h_changed"])
        finite &= math.isfinite(float(state.components["log_target"]))
    elapsed = time.perf_counter() - began
    return {"seconds": elapsed, "sweeps": sweeps, "ms_per_sweep": 1000 * elapsed / sweeps,
            "finite_log_target_throughout": bool(finite),
            "scalar_acceptance": {n: acc[n] / max(prop[n], 1) for n in SCALAR_ORDER},
            "u_marginal": {"proposed": u_prop, "accepted": u_acc, "invalid": u_invalid,
                           "acceptance": u_acc / max(u_prop, 1),
                           "invalid_rate": u_invalid / max(u_prop, 1),
                           "h_changing_accepted": h_changed},
            "draws": draws}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, default=400)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    base = {n: float(REGISTERED_SCALES[n]) for n in SCALAR_ORDER}
    print(f"pilot 1 at registered base scales: {base}")
    first = run(base, args.sweeps, PILOT_SEED)
    print(f"  {first['ms_per_sweep']:.1f} ms/sweep, finite={first['finite_log_target_throughout']}")
    for n in SCALAR_ORDER:
        print(f"  {n:<12} acceptance {first['scalar_acceptance'][n]:.3f}")
    print(f"  U marginal: acceptance {first['u_marginal']['acceptance']:.3f}, "
          f"invalid {first['u_marginal']['invalid_rate']:.3f}, "
          f"H-changing accepted {first['u_marginal']['h_changing_accepted']}")

    adjusted, final = {}, dict(base)
    burn = args.sweeps // 2
    for n in SCALAR_ORDER:
        rate = first["scalar_acceptance"][n]
        if WINDOW[0] <= rate <= WINDOW[1]:
            adjusted[n] = {"adjusted": False, "acceptance": rate, "scale": base[n]}
            continue
        tail = np.asarray(first["draws"][n][burn:], dtype=float)
        coord = np.log(tail) if PROPOSAL_KIND[n] == "log" else tail
        spread = float(np.std(coord, ddof=1))
        new = float(OPTIMAL_RWM * spread) if spread > 0 else base[n]
        final[n] = new
        adjusted[n] = {"adjusted": True, "acceptance": rate, "scale": new,
                       "posterior_spread_in_proposal_coordinate": spread}
    print(f"\nadjusted scales: {final}")

    second = None
    if any(v["adjusted"] for v in adjusted.values()):
        print("pilot 2 (verification at the adjusted scales)")
        second = run(final, args.sweeps, PILOT_SEED + 1)
        for n in SCALAR_ORDER:
            print(f"  {n:<12} acceptance {second['scalar_acceptance'][n]:.3f}")
        print(f"  {second['ms_per_sweep']:.1f} ms/sweep")

    in_band = {n: WINDOW[0] <= (second or first)["scalar_acceptance"][n] <= WINDOW[1]
               for n in SCALAR_ORDER}
    manifest = {
        "rule": "recurrent_scalar_mcmc.tune_proposal_scale: acceptance window "
                "(0.20, 0.55), at most one adjustment, scale set from the pilot's own "
                "posterior spread in the proposal coordinate x 2.38",
        "base_scales_registered_stage6d": base,
        "window": list(WINDOW), "sweeps_per_pilot": args.sweeps,
        "pilot_seed": PILOT_SEED, "pilot_start_seed": PILOT_START_SEED,
        "truth_free": True,
        "quantities_used": ["scalar acceptance", "U acceptance", "invalid rate",
                            "finite log target", "runtime",
                            "H-changing movement (liveness, reported only)"],
        "pilot_1": {k: v for k, v in first.items() if k != "draws"},
        "adjustment": adjusted,
        "pilot_2": {k: v for k, v in (second or {}).items() if k != "draws"} or None,
        "selected_scales": final,
        "u_row_scale": float(REGISTERED_SCALES["U"]),
        "structural_scale_path_marginal": 0.5,
        "all_in_band": bool(all(in_band.values())), "in_band": in_band,
        "all_pilot_draws_discarded": True,
    }
    (OUT / "PROPOSAL_SCALE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nall in band: {manifest['all_in_band']}  -> {in_band}")
    print(f"wrote {OUT/'PROPOSAL_SCALE_MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
