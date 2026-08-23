"""U-proposal audit at the frozen chain states — diagnosis only, no formal chain touched.

    PYTHONPATH=src python scripts/stage7b2_u_proposal_audit.py [--n 10000] [--probe]

For every chain of both experiments (Step 7B2 FFBS, Stage 6E2 LocalMoveKernel), restore
the latest checkpoint state and draw ~N U-row proposals from the REGISTERED kernel —
`sampler_u.propose_row` at the registered scale 0.5, accepted by
`log_alpha = delta_log_likelihood + delta_log_structural_prior` exactly as `sweep_once`
step 3 computes it — WITHOUT ever updating the state. Each proposal records:

    h_changed        whether h(U') differs from h(U) for the perturbed skill
    d_log_prior      log p(U'|rho) - log p(U|rho)   (structural prior, at the chain's rho)
    d_log_lik        the per-skill full-replay likelihood difference
    log_alpha        their sum (the proposal is symmetric: no Hastings term)
    p_accept         min(1, exp(log_alpha))

The two headline rates, per chain:

    r_propose = P(h(U') != h(U))          — does the kernel even reach the cell boundary?
    r_accept  = E[p_accept | h changed]   — and when it does, does the target let it out?

Their product is the per-proposal escape probability of the occupied structural mode.
The 2x2 accepted/rejected x same-H/changed-H table is reported in expectation (summing
p_accept, low variance) and as one Bernoulli realisation (integer counts).

Negative control, checked per proposal: when h(U') == h(U) the likelihood must not move
(the target reads U only through h(U)); the audit records the max |d_log_lik| over
same-H proposals and fails loudly if it exceeds 1e-9.

The states are read-only. The 6E2 checkpoints belong to the Stage 6E2 experiment and are
only READ; this audit steers nothing — both formal runs continue on their registered
schedules regardless of what it finds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.latent_poset import precedence_from_u                 # noqa: E402
from hpop.mcmc_original.sampler_u import propose_row                          # noqa: E402
from hpop.mcmc_original.stage6c_frozen import log_structural_prior            # noqa: E402
from hpop.mcmc_original.stage6e_corpus import corpus_hash, generate_corpus    # noqa: E402
from hpop.mcmc_original.stage6e_sampler import Stage6ESampler                 # noqa: E402
from hpop.mcmc_original.stage6e_state import Stage6EState                     # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "stage7b2_u_audit"
U_SCALE = 0.5              # the registered scale, identical in both experiments
AUDIT_SEED = 8_150_000     # audit-only; the formal chains' RNGs are never touched

SOURCES = {
    "7b2_ffbs": ROOT / "results" / "mcmc_original" / "stage7b2_full_joint_ffbs"
                / "checkpoints",
    "6e2_local": Path("/Users/dongqing/Desktop/hpop-stage6e/results/mcmc_original"
                      "/stage6e2_unknown_boundary_full_seed0/unknown_checkpoints"),
}


def load_baseline_script():
    import importlib.util
    path = ROOT / "scripts" / "stage6e2_formal_chains.py"
    spec = importlib.util.spec_from_file_location("stage6e2_formal_chains", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_state(state: Stage6EState, model, n_proposals: int, seed: int) -> dict:
    """~n proposals cycled over every (skill, row), all from the SAME frozen state."""
    sampler = Stage6ESampler(model=model, scales={"U": U_SCALE},
                             n_proposals_per_trace=0)
    sampler.prepare(state)
    skill_ll = sampler._skill
    skill_ll.set_blocks(state.segmentations, model.n_skills)

    u = np.array(state.u_by_skill, dtype=float)
    K, m, _ = u.shape
    current_ll = {k: skill_ll.full_replay(k, u[k], state.beta, state.omega,
                                          state.lambda_rep, state.lambda_back)
                  for k in range(K)}
    current_structural = {k: log_structural_prior(u[k], state.rho) for k in range(K)}
    current_h = {k: precedence_from_u(u[k]) for k in range(K)}

    rng = np.random.default_rng(seed)
    rounds = max(1, round(n_proposals / (K * m)))
    records = {name: [] for name in ("skill", "row", "h_changed", "d_log_prior",
                                     "d_log_lik", "log_alpha", "p_accept",
                                     "accept_draw", "relations_new")}
    n_invalid = 0
    began = time.perf_counter()
    for _ in range(rounds):
        for k in range(K):
            for row in range(m):
                candidate = propose_row(u[k], row, U_SCALE, rng)
                cand_structural = log_structural_prior(candidate, state.rho)
                if not math.isfinite(cand_structural):
                    n_invalid += 1
                    continue
                cand_ll = skill_ll.full_replay(k, candidate, state.beta, state.omega,
                                               state.lambda_rep, state.lambda_back)
                if not math.isfinite(cand_ll):
                    n_invalid += 1
                    continue
                h_new = precedence_from_u(candidate)
                d_prior = cand_structural - current_structural[k]
                d_ll = cand_ll - current_ll[k]
                log_alpha = d_ll + d_prior
                records["skill"].append(k)
                records["row"].append(row)
                records["h_changed"].append(not np.array_equal(h_new, current_h[k]))
                records["d_log_prior"].append(d_prior)
                records["d_log_lik"].append(d_ll)
                records["log_alpha"].append(log_alpha)
                records["p_accept"].append(min(1.0, math.exp(min(0.0, log_alpha))))
                records["accept_draw"].append(
                    bool(log_alpha >= 0.0 or math.log(rng.random()) < log_alpha))
                records["relations_new"].append(int(h_new.sum()))
    seconds = time.perf_counter() - began

    arrays = {name: np.asarray(v) for name, v in records.items()}
    changed = arrays["h_changed"]
    same = ~changed
    p_acc = arrays["p_accept"]

    same_h_lik_leak = float(np.abs(arrays["d_log_lik"][same]).max()) if same.any() else 0.0
    if same_h_lik_leak > 1e-9:
        raise AssertionError(
            f"negative control failed: |d_log_lik| = {same_h_lik_leak} on a same-H "
            "proposal, but the target must read U only through h(U)")

    def crossing_quantiles(values):
        if not changed.any():
            return None
        qs = np.quantile(np.asarray(values)[changed], [0.0, 0.25, 0.5, 0.75, 1.0])
        return {q: float(v) for q, v in zip(("min", "q25", "median", "q75", "max"), qs)}

    n = len(p_acc)
    r_propose = float(changed.mean()) if n else float("nan")
    r_accept = float(p_acc[changed].mean()) if changed.any() else None
    return {
        "n_proposals": n, "n_invalid": n_invalid, "seconds": seconds,
        "current_relations_per_skill": [int(current_h[k].sum()) for k in range(K)],
        "current_total_relations": int(sum(current_h[k].sum() for k in range(K))),
        "r_propose": r_propose,
        "r_accept_given_crossing": r_accept,
        "escape_probability_per_proposal": (r_propose * r_accept
                                            if r_accept is not None else 0.0),
        "table_expected": {
            "A_accepted_same_h": float(p_acc[same].sum()),
            "B_rejected_same_h": float((1.0 - p_acc[same]).sum()),
            "C_accepted_changed_h": float(p_acc[changed].sum()),
            "D_rejected_changed_h": float((1.0 - p_acc[changed]).sum()),
        },
        "table_bernoulli": {
            "A_accepted_same_h": int((arrays["accept_draw"] & same).sum()),
            "B_rejected_same_h": int((~arrays["accept_draw"] & same).sum()),
            "C_accepted_changed_h": int((arrays["accept_draw"] & changed).sum()),
            "D_rejected_changed_h": int((~arrays["accept_draw"] & changed).sum()),
        },
        "acceptance_same_h": float(p_acc[same].mean()) if same.any() else None,
        "crossing_decomposition": {
            "d_log_lik": crossing_quantiles(arrays["d_log_lik"]),
            "d_log_prior": crossing_quantiles(arrays["d_log_prior"]),
            "log_alpha": crossing_quantiles(arrays["log_alpha"]),
            "share_killed_by_likelihood": (
                float((np.asarray(arrays["d_log_lik"])[changed]
                       < np.asarray(arrays["d_log_prior"])[changed]).mean())
                if changed.any() else None),
        },
        "r_propose_by_skill_row": {
            f"k{k}_r{row}": float(changed[(arrays["skill"] == k)
                                          & (arrays["row"] == row)].mean())
            for k in range(K) for row in range(m)},
        "negative_control_max_same_h_dloglik": same_h_lik_leak,
        "arrays": arrays,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10_000,
                        help="approximate proposals per chain")
    parser.add_argument("--probe", action="store_true",
                        help="one chain, ~150 proposals, to measure cost and stop")
    args = parser.parse_args()

    corpus = generate_corpus()
    module = load_baseline_script()
    model = module.build_model(corpus)
    OUT.mkdir(parents=True, exist_ok=True)

    results: dict = {"corpus_hash": corpus_hash(corpus), "u_scale": U_SCALE,
                     "audit_seed": AUDIT_SEED, "n_requested_per_chain": args.n,
                     "kernel": "sampler_u.propose_row + the sweep_once step-3 "
                               "acceptance, state never updated",
                     "experiments": {}}
    for name, directory in SOURCES.items():
        results["experiments"][name] = {}
        for chain in range(4):
            path = directory / f"chain{chain}_checkpoint.json"
            if not path.exists():
                results["experiments"][name][str(chain)] = {"missing": str(path)}
                continue
            payload = json.loads(path.read_text())
            state = Stage6EState.from_dict(payload["state"])
            n = 150 if args.probe else args.n
            audit = audit_state(state, model, n, AUDIT_SEED + 100 * chain
                                + (0 if name == "7b2_ffbs" else 1))
            arrays = audit.pop("arrays")
            np.savez_compressed(OUT / f"{name}_chain{chain}_proposals.npz", **arrays)
            audit["checkpoint_sweep"] = payload["sweep"]
            results["experiments"][name][str(chain)] = audit
            print(f"[u-audit] {name} chain {chain} (sweep {payload['sweep']:,}): "
                  f"r_propose {audit['r_propose']:.4f}, "
                  f"r_accept|cross {audit['r_accept_given_crossing']}, "
                  f"escape/proposal {audit['escape_probability_per_proposal']:.2e}, "
                  f"{audit['seconds']:.0f}s", flush=True)
            if args.probe:
                per = audit["seconds"] / max(1, audit["n_proposals"])
                print(f"[u-audit] probe: {per * 1000:.1f} ms/proposal -> "
                      f"{per * args.n * 8 / 60:.0f} min for 8 chains x {args.n:,}")
                return

    (OUT / "audit.json").write_text(json.dumps(results, indent=2))
    print(f"[u-audit] wrote {OUT}")


if __name__ == "__main__":
    main()
