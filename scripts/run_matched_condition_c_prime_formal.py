"""Condition C' formal runner — FROZEN PROTOCOL, NO-LAUNCH BY DEFAULT.

Dry run (safe, the default):
    PYTHONPATH=src .venv/bin/python scripts/run_matched_condition_c_prime_formal.py

Launch (only after Condition C has terminated):
    PYTHONPATH=src .venv/bin/python scripts/run_matched_condition_c_prime_formal.py \
        --protocol condition-c-prime-v1 --launch-formal

Implements the protocol frozen at commit 9b8e590 exactly. It refuses to start a
chain unless BOTH explicit flags are given AND the formal Condition C run has
reached its registered terminal state with no live Condition C processes. It
never signals, pauses or alters Condition C.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_synthetic_generator as msg              # noqa: E402
from hpop.mcmc_original.collapsed_u_kernel import CollapsedUConfig             # noqa: E402
from hpop.mcmc_original.latent_poset import precedence_from_u                  # noqa: E402
from hpop.mcmc_original.matched_condition_b import canonical_h_hash            # noqa: E402
from hpop.mcmc_original.matched_condition_c import (                           # noqa: E402
    ConditionCSampler, build_condition_c_model, initial_condition_c_state,
)
from hpop.mcmc_original.matched_condition_c_prime import (                     # noqa: E402
    ConditionCPrimeChain, SealedTruth, swap_diagnostics,
)
from hpop.mcmc_original.sampler_u import sigma_rho_matrix                      # noqa: E402
from hpop.mcmc_original.skill_swap_kernel import (                             # noqa: E402
    SkillSwapConfig, permutation_invariance_report,
)

PROTOCOL_ID = "condition-c-prime-v1"
PREREG_COMMIT = "9b8e590"
C_LAUNCH_COMMIT = "50eee50"   # the commit the running Condition C loaded
C_DIR = ROOT / "results" / "mcmc_original" / "matched_condition_c"
OUT = ROOT / "results" / "mcmc_original" / "matched_condition_c_prime"
CHAINS = OUT / "formal_chains"
CORPUS_DIR = ROOT / "results" / "mcmc_original" / "matched_synthetic_formal_corpus"

# --------------------------------------------- frozen protocol (from 9b8e590)
RHO_0 = 0.5
GENERATION_SEED = 6_200_001
U_SCALE = 0.5
SCHEDULED_COLLAPSED_SCALE = 1.0
COLLAPSED_CADENCE = 10
SWAP_CADENCE = 50
ARMS = {
    "C-MARG-SWAP": {"collapsed_every": COLLAPSED_CADENCE, "swap_every": SWAP_CADENCE,
                    "seeds": (6_205_011, 6_205_012, 6_205_013, 6_205_014)},
    "C-COND-SWAP": {"collapsed_every": 0, "swap_every": SWAP_CADENCE,
                    "seeds": (6_205_001, 6_205_002, 6_205_003, 6_205_004)},
}
START_SEEDS = (6_204_101, 6_204_102, 6_204_103, 6_204_104)
START_SCALES = (0.5, 1.0, 2.0, 3.0)
CHECKPOINTS = (30_000, 50_000, 75_000, 100_000)
BURN_IN = 10_000
THIN = 5
CHECKPOINT_EVERY = 2_000

SHARED_SOURCES = {
    "matched_generator": "src/hpop/mcmc_original/matched_synthetic_generator.py",
    "formal_corpus_loader": "scripts/generate_matched_formal_corpus.py",
    "semi_markov_ffbs": "src/hpop/mcmc_original/semi_markov_ffbs.py",
    "ordinary_u_row_kernel": "src/hpop/mcmc_original/sampler_u.py",
    "path_marginal_likelihood": "src/hpop/mcmc_original/collapsed_u_likelihood.py",
    "collapsed_u_kernel": "src/hpop/mcmc_original/collapsed_u_kernel.py",
    "recurrent_block_scorer": "src/hpop/mcmc_original/recurrent_segmentation.py",
    "ffbs_joint_sampler": "src/hpop/mcmc_original/recurrent_joint_ffbs_mcmc.py",
    "fast_block_tables": "src/hpop/mcmc_original/fast_block_tables.py",
    "condition_c_composition": "src/hpop/mcmc_original/matched_condition_c.py",
}
NEW_SOURCES = {
    "skill_swap_kernel": "src/hpop/mcmc_original/skill_swap_kernel.py",
    "condition_c_prime_chain": "src/hpop/mcmc_original/matched_condition_c_prime.py",
    "condition_c_prime_runner": "scripts/run_matched_condition_c_prime_formal.py",
}

_WORKER: dict = {}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _sha(rel: str) -> str:
    path = ROOT / rel
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() \
        else "MISSING"


def _unchanged_since(commit: str, rel: str) -> bool:
    """True iff `rel` is byte-identical to its state at `commit`."""
    return subprocess.run(["git", "diff", "--quiet", commit, "--", rel],
                          cwd=ROOT).returncode == 0


def _dump(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True)
                            + "\n")


def condition_c_gate():
    """Import the Condition-C gate function itself, so the gates are identical."""
    spec = importlib.util.spec_from_file_location(
        "run_matched_condition_c_formal",
        ROOT / "scripts/run_matched_condition_c_formal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ======================================================== guards
def condition_c_status() -> dict:
    """Has formal Condition C terminated? Artifact state AND live processes."""
    verdict = C_DIR / "formal_verdict.json"
    terminal = verdict.exists()
    payload = json.loads(verdict.read_text()) if terminal else None
    listing = subprocess.run(["ps", "ax", "-o", "pid=,command="],
                             capture_output=True, text=True).stdout
    live = [line.strip() for line in listing.splitlines()
            if "run_matched_condition_c_formal" in line and "grep" not in line]
    # the orchestrator's workers are anonymous spawn_main processes; treat any
    # live orchestrator as decisive and report worker count for the record
    workers = [line.strip() for line in listing.splitlines()
               if "spawn_main" in line and "grep" not in line]
    return {
        "terminal_artifact": str(verdict.relative_to(ROOT)),
        "terminal_artifact_exists": terminal,
        "classification": (payload or {}).get("classification"),
        "converged": (payload or {}).get("converged"),
        "live_orchestrator_processes": len(live),
        "live_spawn_workers": len(workers),
        "may_launch": bool(terminal and not live),
    }


def assert_may_launch(status: dict) -> None:
    if not status["may_launch"]:
        reason = ("its terminal artifact does not exist yet"
                  if not status["terminal_artifact_exists"]
                  else f"{status['live_orchestrator_processes']} orchestrator "
                       "process(es) are still alive")
        raise SystemExit(
            "Condition C' cannot launch while formal Condition C is still "
            f"active — {reason}. Condition C has not been signalled, paused or "
            "altered.")


# ======================================================== manifests
def build_environment():
    truth = msg.supplied_truth()
    corpus = msg.generate_corpus(
        GENERATION_SEED, tuple((24, 32, 40, 48)[i % 4] for i in range(100)),
        tuple((24, 32, 40, 48)[i % 4] for i in range(45)), truth)
    return truth, corpus


def make_start(index: int) -> np.ndarray:
    rng = np.random.default_rng(START_SEEDS[index])
    chol = np.linalg.cholesky(sigma_rho_matrix(2, RHO_0))
    return np.array([[START_SCALES[index] * (chol @ rng.standard_normal(2))
                      for _ in range(5)] for _ in range(3)])


def target_manifest(sealed: SealedTruth, corpus) -> dict:
    """C' target, and its parity against the frozen Condition-C target."""
    recorded = json.loads((CORPUS_DIR / "corpus_hash.json").read_text())
    c_reg = json.loads((C_DIR / "formal_registration.json").read_text())
    fixed = sealed.fixed_for_condition_c(RHO_0)
    mine = {
        "corpus_hash_sha256": msg.corpus_hash(corpus),
        "truth_hash_sha256": recorded["truth_hash_sha256"],
        "K": int(sealed.n_skills), "d": int(sealed.latent_dim),
        "m_roles": int(sealed.n_roles),
        "rho": RHO_0, "beta": fixed.beta, "omega": fixed.omega,
        "lambda_rep": fixed.lambda_rep, "lambda_back": fixed.lambda_back,
        "pi": list(fixed.pi), "P": [list(r) for r in fixed.transition],
        "delta_B": float(sealed.delta_b), "epsilon": float(sealed.epsilon),
        "min_width": int(sealed.min_width), "max_width": int(sealed.max_width),
        "role_maps": [list(r) for r in sealed.role_maps],
        "trace_lengths_train": [int(t.length) for t in corpus.train],
        "trace_lengths_heldout": [int(t.length) for t in corpus.heldout],
        "u_scale": U_SCALE,
        "scheduled_collapsed_scale": SCHEDULED_COLLAPSED_SCALE,
        "checkpoints": list(CHECKPOINTS), "burn_in": BURN_IN, "thin": THIN,
    }
    parity = {
        "corpus_hash": mine["corpus_hash_sha256"]
        == recorded["corpus_hash_sha256"],
        "truth_hash": mine["truth_hash_sha256"]
        == c_reg["truth_hash_sha256"],
        "u_scale": mine["u_scale"] == c_reg["u_scale"],
        "scheduled_scale": mine["scheduled_collapsed_scale"]
        == c_reg["scheduled_scale"],
        "collapsed_cadence": COLLAPSED_CADENCE == c_reg["cadence"],
        "checkpoints": mine["checkpoints"] == c_reg["checkpoints"],
        "burn_in": mine["burn_in"] == c_reg["burn_in"],
        "thin": mine["thin"] == c_reg["thin"],
        "starts": [list(h) for h in c_reg["paired_starts"]["h_hashes"]]
        == [list(_start_hashes(i)) for i in range(4)],
        "shared_sources_unchanged_since_c_launch": all(
            _unchanged_since(C_LAUNCH_COMMIT, rel)
            for rel in SHARED_SOURCES.values()),
    }
    shared_detail = {key: {"sha256_now": _sha(rel),
                           "unchanged_since_c_launch":
                               _unchanged_since(C_LAUNCH_COMMIT, rel)}
                     for key, rel in SHARED_SOURCES.items()}
    return {"target": mine, "parity_vs_condition_c": parity,
            "shared_source_detail": shared_detail,
            "all_parity_checks_pass": all(parity.values()),
            "single_difference_from_condition_c":
                "one additional transition: a global whole-skill U "
                f"transposition attempted every {SWAP_CADENCE} sweeps; every "
                "other component, parameter, kernel and datum is identical",
            "permutation_invariance_of_fixed_inputs":
                permutation_invariance_report(sealed.pi, sealed.transition)}


def _start_hashes(index: int) -> tuple:
    u = make_start(index)
    return tuple(canonical_h_hash(precedence_from_u(u[k])) for k in range(3))


def seed_manifest(sealed: SealedTruth, corpus) -> dict:
    model = build_condition_c_model(tuple(t.cpa for t in corpus.train))
    fixed = sealed.fixed_for_condition_c(RHO_0)
    rows = []
    for index in range(4):
        u = make_start(index)
        state = initial_condition_c_state(model, u, fixed)
        keys = [tuple((int(s.end), int(s.skill)) for s in seg.segments)
                for seg in state.segmentations]
        rows.append({
            "start_index": index, "start_seed": START_SEEDS[index],
            "start_scale": START_SCALES[index],
            "initial_u_sha256": hashlib.sha256(
                np.ascontiguousarray(u, dtype=float).tobytes()).hexdigest(),
            "initial_H_tuple": list(_start_hashes(index)),
            "initial_Sz_sha256": hashlib.sha256(
                json.dumps(keys, sort_keys=True).encode()).hexdigest(),
            "initial_segment_total": int(sum(len(k) for k in keys)),
            "per_arm_chain_seed": {arm: cfg["seeds"][index]
                                   for arm, cfg in ARMS.items()},
        })
    return {
        "master_seed": None,
        "master_seed_note": "the frozen protocol registers per-chain seeds "
                            "explicitly rather than deriving them from a "
                            "master; all eight are listed below",
        "init_ffbs_seed_note": "there is no separate init-FFBS seed: the "
                               "initial (S, z) is the deterministic legal "
                               "tiling, and the first sweep's exact FFBS "
                               "refresh — driven by the chain seed — replaces "
                               "it from the full conditional",
        "starts_are_preregistered_dispersed": True,
        "initialized_from_condition_c_checkpoint": False,
        "checkpoint_audit_was_diagnostic_only": True,
        "rows": rows,
        "all_chain_seeds": sorted(s for cfg in ARMS.values()
                                  for s in cfg["seeds"]),
    }


def source_manifest() -> dict:
    return {
        "base_commit": _git("rev-parse", "HEAD"),
        "preregistration_commit": PREREG_COMMIT,
        "runner_commit": "recorded at commit time; see git log for this file",
        "shared_with_condition_c": {k: _sha(v)
                                    for k, v in SHARED_SOURCES.items()},
        "new_for_condition_c_prime": {k: _sha(v)
                                      for k, v in NEW_SOURCES.items()},
        "condition_c_sources_modified": False,
        "no_chain_existed_when_frozen": True,
        "experiment_provenance":
            "Condition C' is a PROSPECTIVELY FROZEN FOLLOW-UP experiment, "
            "designed after diagnosing the Condition C failure. Its protocol "
            f"and runner were frozen (preregistration {PREREG_COMMIT}) before "
            "any Condition C' chain existed, but it was NOT preregistered "
            "before Condition C data were observed, and must not be described "
            "as if it were.",
    }


def frozen_configuration(sealed: SealedTruth, corpus) -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "preregistration_commit": PREREG_COMMIT,
        "arms": {arm: {"collapsed_every": cfg["collapsed_every"],
                       "swap_every": cfg["swap_every"],
                       "seeds": list(cfg["seeds"])}
                 for arm, cfg in ARMS.items()},
        "transition_ordering": [
            "scheduled global skill transposition (swap_every = "
            f"{SWAP_CADENCE})",
            "collapsed U move if scheduled (arm's collapsed_every) — provably "
            "never reads the stored (S, z)",
            "exact FFBS refresh of ALL (S, z)",
            "conditional U row sweep (first path-dependent transition)"],
        "ordering_provenance":
            "frozen in the C' preregistration ('the swap runs BEFORE the "
            "unchanged Condition-C sweep'); the conservative fallback ordering "
            "is unnecessary because no path-dependent operation can intervene "
            "between the swap and the refresh",
        "ffbs_refresh_after_every_swap_attempt": True,
        "checkpoints": list(CHECKPOINTS), "burn_in": BURN_IN, "thin": THIN,
        "within_segment_checkpoint_every": CHECKPOINT_EVERY,
        "stopping_rule": "per arm, PASS at two consecutive checkpoints; "
                         "ceiling 100k never extended",
        "convergence_gates": "the Condition-C gate function itself, imported "
                             "and called — anchored per-skill summaries "
                             "retained; NO permutation-invariant primary gate",
        "recovery_blinding": "hidden truth is sealed until the registered "
                             "stopping condition is reached",
        "u_scale": U_SCALE,
        "scheduled_collapsed_scale": SCHEDULED_COLLAPSED_SCALE,
        "rho_0": RHO_0,
    }


# ======================================================== worker
def _advance(args):
    arm, index, upto = args
    if "env" not in _WORKER:
        _WORKER["env"] = build_environment()
    truth, corpus = _WORKER["env"]
    sealed = SealedTruth(truth)
    key = f"sampler_{arm}"
    if key not in _WORKER:
        model = build_condition_c_model(tuple(t.cpa for t in corpus.train))
        _WORKER[key] = ConditionCSampler(
            model=model, fixed=sealed.fixed_for_condition_c(RHO_0),
            u_scale=U_SCALE,
            collapsed=CollapsedUConfig(every=ARMS[arm]["collapsed_every"],
                                       scale=SCHEDULED_COLLAPSED_SCALE))
    sampler = _WORKER[key]
    swap = SkillSwapConfig(every=ARMS[arm]["swap_every"])
    path = CHAINS / f"{arm}_{index}.npz"
    chain = (ConditionCPrimeChain.load(path, sampler, swap) if path.exists()
             else ConditionCPrimeChain(sampler, make_start(index),
                                       seed=ARMS[arm]["seeds"][index],
                                       burn_in=BURN_IN, thin=THIN, swap=swap))
    chain.advance(upto, checkpoint_path=path,
                  checkpoint_every=CHECKPOINT_EVERY, progress_every=10_000)
    chain.assert_ordering_invariant()
    return arm, index, {
        "log_target": list(chain.retained_log_target),
        "log_prior": list(chain.retained_log_prior),
        "rel_counts": list(chain.retained_rel_counts),
        "indicators": np.asarray(chain.retained_indicators, dtype=bool),
        "movement": dict(chain.movement),
        "collapsed": [chain.collapsed_proposed, chain.collapsed_accepted,
                      chain.collapsed_h_accepted],
        "swap": swap_diagnostics(chain),
        "seconds": chain.seconds, "iteration": int(chain.state.iteration),
    }


# ======================================================== main
def run_formal() -> int:
    truth, corpus = build_environment()
    sealed = SealedTruth(truth)
    CHAINS.mkdir(parents=True, exist_ok=True)
    gate_module = condition_c_gate()
    active = {arm: True for arm in ARMS}
    consecutive = {arm: 0 for arm in ARMS}
    stopped_at, log, latest = {}, [], {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for checkpoint in CHECKPOINTS:
            jobs = [(arm, c, checkpoint) for arm in ARMS if active[arm]
                    for c in range(4)]
            if not jobs:
                break
            print(f"== advancing to {checkpoint:,} ==", flush=True)
            for arm, index, rows in pool.map(_advance, jobs):
                latest.setdefault(arm, [None] * 4)[index] = rows
            for arm in ARMS:
                if not active[arm]:
                    continue
                gate = gate_module.arm_gate(latest[arm])
                for row in latest[arm]:
                    if not row["swap"]["ffbs_refreshes_equals_attempts"]:
                        raise AssertionError("ordering invariant violated")
                log.append({"checkpoint": checkpoint, "arm": arm,
                            "pass": gate["pass"], "checks": gate["checks"],
                            "swap": [r["swap"] for r in latest[arm]]})
                _dump(f"formal_gate_{arm}_{checkpoint}.json",
                      {"checks": gate["checks"], "summaries": gate["summaries"],
                       "uncertain": gate["uncertain"], "pass": gate["pass"],
                       "swap_diagnostics": [r["swap"] for r in latest[arm]]})
                print(f"  [{arm}] {checkpoint:,}: "
                      f"{'PASS' if gate['pass'] else 'FAIL'} "
                      f"(max R-hat {gate['checks']['max_rhat']:.4f})",
                      flush=True)
                consecutive[arm] = consecutive[arm] + 1 if gate["pass"] else 0
                if consecutive[arm] >= 2:
                    active[arm] = False
                    stopped_at[arm] = {"checkpoint": checkpoint,
                                       "converged": True}
            if not any(active.values()):
                break
    for arm in ARMS:
        stopped_at.setdefault(arm, {"checkpoint": CHECKPOINTS[-1],
                                    "converged": False})
    _dump("formal_convergence.json", {"checkpoint_log": log,
                                      "stopped_at": stopped_at})
    print("\nCondition C' chains complete. Recovery analysis is sealed until "
          "run separately; stopping state written to formal_convergence.json.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--protocol", default=None)
    args = parser.parse_args()

    truth, corpus = build_environment()
    sealed = SealedTruth(truth)
    config = frozen_configuration(sealed, corpus)
    targets = target_manifest(sealed, corpus)
    seeds = seed_manifest(sealed, corpus)
    sources = source_manifest()
    status = condition_c_status()

    _dump("condition_c_prime_target_manifest.json", targets)
    _dump("condition_c_prime_seed_manifest.json", seeds)
    _dump("condition_c_prime_source_manifest.json", sources)

    launching = bool(args.launch_formal) and args.protocol == PROTOCOL_ID
    dry = {
        "invoked_at_utc": None,
        "frozen_configuration": config,
        "target_parity_all_pass": targets["all_parity_checks_pass"],
        "target_parity": targets["parity_vs_condition_c"],
        "condition_c_status": status,
        "launch_flags": {"--launch-formal": bool(args.launch_formal),
                         "--protocol": args.protocol,
                         "protocol_matches": args.protocol == PROTOCOL_ID},
        "would_launch": launching,
        "chains_started": 0,
        "launch_command": "PYTHONPATH=src .venv/bin/python "
                          "scripts/run_matched_condition_c_prime_formal.py "
                          f"--protocol {PROTOCOL_ID} --launch-formal",
    }
    _dump("condition_c_prime_runner_dry_run.json", dry)

    print(json.dumps({"protocol": PROTOCOL_ID,
                      "arms": {a: {"collapsed_every": c["collapsed_every"],
                                   "swap_every": c["swap_every"],
                                   "seeds": list(c["seeds"])}
                               for a, c in ARMS.items()},
                      "transition_ordering": config["transition_ordering"],
                      "checkpoints": list(CHECKPOINTS),
                      "target_parity_all_pass":
                          targets["all_parity_checks_pass"],
                      "condition_c_terminal": status["terminal_artifact_exists"],
                      "condition_c_live_orchestrators":
                          status["live_orchestrator_processes"],
                      "may_launch": status["may_launch"]}, indent=2))

    if not launching:
        print("\nDRY RUN — no chain started. To launch after Condition C "
              f"terminates:\n  {dry['launch_command']}")
        return 0
    if not targets["all_parity_checks_pass"]:
        raise SystemExit("target parity against Condition C failed; refusing "
                         "to launch")
    assert_may_launch(status)
    print("\nLAUNCHING Condition C' formal chains")
    return run_formal()


if __name__ == "__main__":
    raise SystemExit(main())
