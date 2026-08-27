"""Run the preregistered unanchored FULL-LATENT formal experiment.

This runner never regenerates the corpus and never opens terminal recovery truth.  It
only loads CPA observations through ``load_frozen_observed_corpus``.  Recovery has a
separate opt-in script and is forbidden until this runner has made both arms terminal.

Typical controlled sequence:

    PYTHONPATH=src .venv/bin/python scripts/run_matched_full_latent_formal.py --smoke
    PYTHONPATH=src .venv/bin/python scripts/run_matched_full_latent_formal.py --audit
    PYTHONPATH=src .venv/bin/python scripts/run_matched_full_latent_formal.py --prepare-launch
    # commit the generated launch_manifest.json as the dedicated launch record
    PYTHONPATH=src .venv/bin/python scripts/run_matched_full_latent_formal.py --launch-formal

The explicit prepare/commit/launch separation prevents an accidental formal sweep before
the source, registration, starts, and hashes have a durable launch record.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_full_latent as mfl                 # noqa: E402
from hpop.mcmc_original.stage6b_mcmc_diagnostics import (                 # noqa: E402
    bulk_ess,
    rank_normalized_split_rhat,
    tail_ess,
)


OUT = ROOT / "results" / "mcmc_original" / "matched_full_latent"
CORPUS_DIR = ROOT / "results" / "mcmc_original" / "matched_synthetic_formal_corpus"
PREREG_JSON = ROOT / "PREREG_FULL_LATENT.json"
PREREG_MD = ROOT / "PREREG_FULL_LATENT.md"
CHAIN_DIR = OUT / "formal_chains"
DISCARDED_PREFIX_RECORD = OUT / "DISCARDED_precheckpoint_attempt.json"
PRELAUNCH_AUDIT = OUT / "prelaunch_audit.json"
PRELAUNCH_VALIDATION = OUT / "prelaunch_validation.json"
SMOKE_REPORT = OUT / "smoke" / "smoke.json"

ARMS = {
    mfl.FULL_COND: (6_206_201, 6_206_202, 6_206_203, 6_206_204),
    mfl.FULL_MARG: (6_206_211, 6_206_212, 6_206_213, 6_206_214),
}
U_START_SEEDS = (6_204_101, 6_204_102, 6_204_103, 6_204_104)
U_START_SCALES = (0.5, 1.0, 2.0, 3.0)
PI_P_START_SEEDS = (6_206_101, 6_206_102, 6_206_103, 6_206_104)
STRUCTURAL_CADENCE = 10
STRUCTURAL_SCALE = 0.5
CHECKPOINTS = (30_000, 50_000, 75_000, 100_000)
BURN_IN = 10_000
THIN = 5
CHECKPOINT_EVERY = 2_000
RHAT_GATE = 1.01
ESS_FLOORS = {
    "log_target_bulk": 1000.0,
    "log_target_tail": 500.0,
    "total_relations_bulk": 1000.0,
    "remaining_invariant_bulk": 500.0,
}
PROBE_COUNTS = {"boundary": 32, "coskill": 64, "recovery_coskill": 256}
_WORKER: dict = {}
_TRUTH_FORBIDDEN_MODULES = frozenset({
    "hpop.mcmc_original.recurrent_synthetic",
    "hpop.mcmc_original.matched_synthetic_generator",
    "hpop.mcmc_original.generate_matched_formal_corpus",
})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=False).stdout.strip()


def _git_status() -> str:
    return _git("status", "--short")


def _head_bytes(relative: str) -> bytes:
    """Read one tracked HEAD blob without accepting an uncommitted substitute."""
    result = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT,
                            capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"required launch path is not present in HEAD: {relative}")
    return bytes(result.stdout)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _fixed() -> mfl.FullLatentFixed:
    return mfl.FullLatentFixed()


def _config(arm: str) -> mfl.FullLatentConfig:
    return mfl.FullLatentConfig(arm=arm, structural_cadence=STRUCTURAL_CADENCE,
                                structural_scale=STRUCTURAL_SCALE,
                                table_source="batched")


def _environment():
    """One observed-only formal environment, cached once per process."""
    if "environment" not in _WORKER:
        corpus = mfl.load_frozen_observed_corpus(CORPUS_DIR)
        fixed = _fixed()
        model = mfl.build_full_latent_model(corpus.train, fixed)
        probes = mfl.select_truth_free_probes(
            corpus.train, corpus.corpus_hash,
            boundary_count=PROBE_COUNTS["boundary"],
            coskill_count=PROBE_COUNTS["coskill"],
            recovery_coskill_count=PROBE_COUNTS["recovery_coskill"],
        )
        _WORKER["environment"] = (corpus, fixed, model, probes)
    return _WORKER["environment"]


def _start(index: int, model, fixed) -> tuple:
    u = mfl.make_u_start(index, U_START_SEEDS[index], U_START_SCALES[index], fixed,
                         model.n_skills, model.n_roles, 2)
    pi, p = mfl.draw_initial_pi_p(model, PI_P_START_SEEDS[index])
    return mfl.initial_full_latent_state(model, u, pi, p, fixed), u, pi, p


def _array_hash(array: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _corpus_provenance(corpus: mfl.ObservedCorpus) -> dict:
    return {
        "generator_commit": corpus.generator_commit,
        "corpus_commit": corpus.corpus_commit,
        "corpus_hash": corpus.corpus_hash,
        "train_hash": corpus.train_hash,
        "heldout_hash": corpus.heldout_hash,
        "truth_hash": json.loads((CORPUS_DIR / "corpus_hash.json").read_text())
        ["truth_hash_sha256"],
    }


def start_manifest() -> dict:
    corpus, fixed, model, _ = _environment()
    rows = []
    for index in range(4):
        state, u, pi, p = _start(index, model, fixed)
        paths = [[(s.start, s.end, s.skill) for s in segmentation.segments]
                 for segmentation in state.segmentations]
        rows.append({
            "index": index,
            "u_seed": U_START_SEEDS[index], "u_scale": U_START_SCALES[index],
            "pi_P_seed": PI_P_START_SEEDS[index],
            "state_sha256": mfl.start_state_hash(state),
            "U_sha256": _array_hash(u), "pi_sha256": _array_hash(pi),
            "P_sha256": _array_hash(p),
            "paths_sha256": hashlib.sha256(
                json.dumps(paths, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "n_initial_segments": int(sum(len(s.segments)
                                            for s in state.segmentations)),
            "P_zero_diagonal": bool(np.array_equal(np.diag(p), np.zeros(model.n_skills))),
        })
    return {
        "construction": "Paired across arms; U/S/z/pi/P use no truth alignment or C/C' "
                        "endpoint.",
        "start_rows": rows,
        "pairing": {str(index): {mfl.FULL_COND: rows[index]["state_sha256"],
                                  mfl.FULL_MARG: rows[index]["state_sha256"]}
                    for index in range(4)},
        "corpus_hash_sha256": corpus.corpus_hash,
    }


def _module_path(module: str) -> Path | None:
    """Return the local source file for an importable ``hpop`` module, if any."""
    if not module.startswith("hpop"):
        return None
    relative = Path(*module.split("."))
    source = ROOT / "src" / relative.with_suffix(".py")
    if source.exists():
        return source
    package = ROOT / "src" / relative / "__init__.py"
    return package if package.exists() else None


def _imported_hpop_modules(module: str, path: Path) -> set[str]:
    """Statically resolve local HPOP imports, including function-local imports."""
    tree = ast.parse(path.read_text(), filename=str(path))
    package_parts = module.split(".")[:-1]
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names if alias.name.startswith("hpop"))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level > len(package_parts):
                    continue
                base_parts = package_parts[:len(package_parts) - (node.level - 1)]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if not base.startswith("hpop"):
                continue
            imports.add(base)
            for alias in node.names:
                if alias.name != "*":
                    imports.add(f"{base}.{alias.name}")
    return imports


def _hpop_import_closure() -> dict[str, Path]:
    """Static local HPOP source closure for the formal model and diagnostics."""
    roots = {
        "hpop",
        "hpop.mcmc_original",
        "hpop.mcmc_original.matched_full_latent",
        "hpop.mcmc_original.stage6b_mcmc_diagnostics",
    }
    pending = list(roots)
    resolved: dict[str, Path] = {}
    while pending:
        module = pending.pop()
        if module in resolved:
            continue
        path = _module_path(module)
        if path is None:
            continue
        resolved[module] = path
        for imported in _imported_hpop_modules(module, path):
            if imported not in resolved:
                pending.append(imported)
    return dict(sorted(resolved.items()))


def _assert_runtime_truth_seal() -> dict:
    """Prove a fresh formal-process import has not loaded generator/truth modules.

    The legacy frozen-model files retain *lazy* imports for historical corpus-audit
    functions.  A static source closure intentionally hashes those functions too, but
    only this subprocess check establishes what a live FULL-LATENT process actually
    imports before it reads the observed CPA arrays.
    """
    code = "\n".join((
        "import importlib.util, json, sys",
        "from hpop.mcmc_original import matched_full_latent",  # noqa: F401
        "spec = importlib.util.spec_from_file_location('full_latent_runner_import_probe', "
        + repr(str(ROOT / "scripts" / "run_matched_full_latent_formal.py")) + ")",
        "module = importlib.util.module_from_spec(spec)",
        "spec.loader.exec_module(module)",
        "forbidden = " + repr(sorted(_TRUTH_FORBIDDEN_MODULES)),
        "loaded = sorted(name for name in forbidden if name in sys.modules)",
        "loaded_hpop = sorted(name for name in sys.modules if name.startswith('hpop.mcmc_original'))",
        "print(json.dumps({'loaded_forbidden_modules': loaded, 'loaded_hpop_modules': loaded_hpop}))",
    ))
    environment = dict(os.environ)
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment,
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise AssertionError("truth-seal subprocess import failed: " + result.stderr.strip())
    payload = json.loads(result.stdout)
    if payload["loaded_forbidden_modules"]:
        raise AssertionError("formal import loaded sealed generator/truth modules: "
                             f"{payload['loaded_forbidden_modules']}")
    return {"status": "PASS", "checked_modules": sorted(_TRUTH_FORBIDDEN_MODULES),
            **payload}


def _runtime_manifest() -> dict:
    return {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__}


def _validated_prelaunch_artifact_hashes(expected_source_commit: str | None = None) -> dict:
    """Require the committed-launch inputs to include current PASS audit/test records."""
    required = {
        "prelaunch_audit_sha256": PRELAUNCH_AUDIT,
        "prelaunch_validation_sha256": PRELAUNCH_VALIDATION,
        "smoke_report_sha256": SMOKE_REPORT,
    }
    for label, artifact in required.items():
        if not artifact.is_file():
            raise RuntimeError(f"missing required prelaunch artifact for {label}: {artifact}")
    audit = json.loads(PRELAUNCH_AUDIT.read_text())
    validation = json.loads(PRELAUNCH_VALIDATION.read_text())
    smoke = json.loads(SMOKE_REPORT.read_text())
    if audit.get("truth_seal", {}).get("status") != "PASS":
        raise RuntimeError("prelaunch audit does not certify the runtime truth seal")
    if validation.get("status") != "PASS" or smoke.get("pass") is not True:
        raise RuntimeError("prelaunch validation or nonconfirmatory smoke did not PASS")
    if validation.get("validated_source_hashes") != _source_manifest():
        raise RuntimeError("prelaunch validation was not run against the current source hashes")
    if (expected_source_commit is not None
            and validation.get("validated_source_commit") != expected_source_commit):
        raise RuntimeError("prelaunch validation source commit differs from launch source commit")
    return {label: _sha256(artifact) for label, artifact in required.items()}


def _source_manifest() -> dict:
    # The launch commit pins the full tracked tree.  These hashes freeze the exact
    # recursively-resolved runtime closure as well, which catches a relevant local edit
    # in a user-dirty worktree and makes the observed-only import seal auditable.
    paths = [
        ROOT / "PREREG_FULL_LATENT.json",
        ROOT / "PREREG_FULL_LATENT.md",
        ROOT / "scripts" / "run_matched_full_latent_formal.py",
        ROOT / "scripts" / "run_matched_full_latent_recovery.py",
        ROOT / "src" / "hpop" / "mcmc_original" / "full_latent_recovery.py",
        ROOT / "tests" / "mcmc_original" / "test_matched_full_latent.py",
        ROOT / "tests" / "mcmc_original" / "test_full_latent_recovery.py",
        ROOT / "tests" / "mcmc_original" / "test_full_latent_recovery_driver.py",
        ROOT / "tests" / "mcmc_original" / "test_full_latent_runner.py",
    ]
    paths.extend(_hpop_import_closure().values())
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _assert_registration_matches_code() -> dict:
    prereg = json.loads(PREREG_JSON.read_text())
    fixed = _fixed()
    expected_fixed = prereg["fixed_variables"]
    observed = fixed.as_dict()
    for name, value in (("rho_0", fixed.rho_0), ("beta", fixed.beta),
                        ("omega", fixed.omega), ("lambda_rep", fixed.lambda_rep),
                        ("lambda_back", fixed.lambda_back), ("delta_b", fixed.delta_b),
                        ("epsilon", fixed.epsilon)):
        registered = expected_fixed.get(name, expected_fixed.get(name.replace("_b", "_B")))
        if isinstance(registered, dict):
            registered = registered["value"]
        if not math.isclose(float(value), float(registered), rel_tol=0.0,
                            abs_tol=1e-14):
            raise AssertionError(f"preregistration/code fixed value mismatch for {name}")
    arms = prereg["arms"]
    if set(arms) != {mfl.FULL_COND, mfl.FULL_MARG}:
        raise AssertionError("preregistration arm names do not match runner")
    if prereg["kernel_schedule"]["structural_cadence"] != "every 10 absolute sweeps in both arms; there are no ordinary conditional U row sweeps in either arm.":
        raise AssertionError("unexpected preregistration structural cadence text")
    if STRUCTURAL_CADENCE != 10 or STRUCTURAL_SCALE != 0.5:
        raise AssertionError("runner structural configuration drifted")
    return {"prereg_sha256": _sha256(PREREG_JSON), "fixed": observed,
            "structural_cadence": STRUCTURAL_CADENCE,
            "structural_scale": STRUCTURAL_SCALE}


def prelaunch_audit() -> dict:
    """Truth-free audit of frozen inputs and implementational choices."""
    registration = _assert_registration_matches_code()
    import_seal = _assert_runtime_truth_seal()
    import_closure = sorted(_hpop_import_closure())
    corpus, fixed, model, probes = _environment()
    starts = start_manifest()
    if len(corpus.train) != 100 or len(corpus.heldout) != 45:
        raise AssertionError("formal corpus split does not have the registered trace counts")
    if any(len(probes[key]) != PROBE_COUNTS[key] for key in PROBE_COUNTS):
        raise AssertionError("truth-free probe construction drifted")
    source = (ROOT / "src/hpop/mcmc_original/matched_full_latent.py").read_text()
    forbidden_imports = ("skill_swap_kernel import", "SkillSwapConfig", "3-cycle")
    if any(token in source for token in forbidden_imports):
        raise AssertionError("forbidden rescue kernel appeared in FULL-LATENT source")
    return {
        "condition": "FULL-LATENT",
        "target": "p(S,z,U,pi,P | X, fixed nuisance coordinates)",
        "inferred": ["S", "z", "U", "pi", "P"],
        "fixed": fixed.as_dict(),
        "priors": {"pi": "Dirichlet(1,1,1)",
                   "P": "independent restricted-row Dirichlet(1,1), P_kk=0"},
        "kernel": {
            mfl.FULL_COND: ["conditional U MH", "FFBS", "P Gibbs", "pi Gibbs"],
            mfl.FULL_MARG: ["path-marginal U MH", "FFBS", "P Gibbs", "pi Gibbs"],
            "partially_collapsed_proof": "MARG U reads no stored path; FFBS is first "
                                          "path-dependent step after every attempt."
        },
        "proposal": {"cadence": STRUCTURAL_CADENCE, "scale": STRUCTURAL_SCALE,
                     "selection": "uniform skill and uniform row", "opportunities":
                     "one shared opportunity every 10 sweeps per chain"},
        "starts": starts,
        "convergence": {
            "summaries": ["generic", "boundary", "co-skill", "pi", "P"],
            "rhat": RHAT_GATE, "ess": ESS_FLOORS,
            "checkpoint_ladder": list(CHECKPOINTS), "two_consecutive_passes": True,
            "burn_in": BURN_IN, "thin": THIN,
        },
        "recovery": {"alignment": "closure-Hamming Hungarian; deterministic ties; "
                                   "same mapping for H,pi,P,z",
                     "heldout": "log Z - log C_J then log-mean-exp / occurrence"},
        "corpus": _corpus_provenance(corpus),
        "truth_seal": {
            "status": "PASS",
            "observations": "runner only loads named observed CPA arrays",
            "runtime_import_probe": import_seal,
            "static_hpop_import_closure": import_closure,
        },
        "registration": registration,
        "source_hashes": _source_manifest(),
        "runtime": _runtime_manifest(),
        "git": {"commit": _git("rev-parse", "HEAD"), "status": _git_status()},
        "no_C_prime_state_or_truth_initialization": True,
    }


def _chain_path(arm: str, index: int) -> Path:
    return CHAIN_DIR / f"{arm.lower().replace('-', '_')}_{index}.npz"


def _load_or_create_chain(arm: str, index: int):
    corpus, fixed, model, probes = _environment()
    del corpus
    sampler = mfl.FullLatentSampler(model, fixed, _config(arm))
    path = _chain_path(arm, index)
    if path.exists():
        return mfl.FullLatentChain.load(path, sampler)
    state, _, _, _ = _start(index, model, fixed)
    metadata = {"start_index": index,
                "start_state_sha256": mfl.start_state_hash(state),
                "arm": arm}
    return mfl.FullLatentChain(sampler, state, ARMS[arm][index], BURN_IN, THIN,
                               probes, metadata)


def _advance_worker(payload):
    arm, index, upto = payload
    chain = _load_or_create_chain(arm, index)
    chain.advance(upto, checkpoint_path=_chain_path(arm, index),
                  checkpoint_every=CHECKPOINT_EVERY, progress_every=10_000)
    return {"arm": arm, "index": index, "iteration": chain.state.iteration,
            "seconds": chain.seconds, "retained_draws": chain.retained_draws,
            "structural": chain.structural, "movement": chain.movement}


def advance_formal(arms, upto: int, workers: int = 8) -> list[dict]:
    """Advance all active arm/chain pairs together; checkpoints make interruption safe."""
    tasks = [(arm, index, int(upto)) for arm in arms for index in range(4)]
    outputs = []
    with ProcessPoolExecutor(max_workers=min(len(tasks), int(workers))) as executor:
        future_map = {executor.submit(_advance_worker, task): task for task in tasks}
        for future in as_completed(future_map):
            outputs.append(future.result())
    return sorted(outputs, key=lambda row: row["index"])


def _diag(series_by_chain) -> dict:
    chains = np.asarray(series_by_chain, dtype=float)
    if chains.ndim != 2 or chains.shape[0] != 4 or chains.shape[1] < 4:
        raise ValueError(f"need four nonempty equal-length chains, got {chains.shape}")
    constant = [bool(np.all(chain == chain[0])) for chain in chains]
    if all(constant):
        values = {float(chain[0]) for chain in chains}
        if len(values) == 1:
            return {"rhat": 1.0, "bulk_ess": float(chains.size),
                    "tail_ess": float(chains.size), "degenerate": "constant"}
        return {"rhat": float("inf"), "bulk_ess": 0.0, "tail_ess": 0.0,
                "degenerate": "constant-but-unequal"}
    return {"rhat": float(rank_normalized_split_rhat(chains)["rhat"]),
            "bulk_ess": float(bulk_ess(chains)), "tail_ess": float(tail_ess(chains)),
            "degenerate": None}


def _component_diagnostics(chain_arrays: list[dict]) -> dict:
    """R-hat/ESS for every finite registered invariant scalar component."""
    output = {}
    for name in mfl.FullLatentChain._SUMMARY_KEYS:
        values = [np.asarray(chain[name]) for chain in chain_arrays]
        shape = values[0].shape[1:]
        if any(value.shape[1:] != shape for value in values):
            raise AssertionError(f"summary shape differs across chains: {name}")
        n_components = int(np.prod(shape)) if shape else 1
        for component in range(n_components):
            if shape:
                index = np.unravel_index(component, shape)
                series = [value[(slice(None),) + index] for value in values]
                label = f"{name}[{','.join(str(v) for v in index)}]"
            else:
                series = values
                label = name
            output[label] = _diag(series)
    return output


def arm_gate(arm: str, checkpoint: int) -> dict:
    chains = [_load_or_create_chain(arm, index) for index in range(4)]
    iterations = [int(chain.state.iteration) for chain in chains]
    if iterations != [int(checkpoint)] * 4:
        raise RuntimeError(f"cannot evaluate {arm} gate {checkpoint}: "
                           f"chain iterations are {iterations}")
    arrays = [chain.arrays() for chain in chains]
    summaries = _component_diagnostics(arrays)
    log_target = summaries["log_target"]
    total_relations = summaries["total_relations"]
    auxiliary = [value["bulk_ess"] for name, value in summaries.items()
                 if name not in {"log_target", "total_relations"}]
    # Binary probe summaries can legitimately have a degenerate tail ESS; the inherited
    # registration gates tail ESS only for log target.  Every summary still needs finite
    # rank-normalized R-hat and bulk ESS, and log target needs a finite tail ESS too.
    diagnostics_finite = (
        all(math.isfinite(float(value[metric]))
            for value in summaries.values() for metric in ("rhat", "bulk_ess"))
        and math.isfinite(float(log_target["tail_ess"]))
    )
    h_changes = [chain.structural["h_accepts"] for chain in chains]
    checks = {
        "max_rhat": max(value["rhat"] for value in summaries.values()),
        "log_target_bulk_ess": log_target["bulk_ess"],
        "log_target_tail_ess": log_target["tail_ess"],
        "total_relations_bulk_ess": total_relations["bulk_ess"],
        "minimum_remaining_invariant_bulk_ess": min(auxiliary),
        "all_diagnostics_finite": diagnostics_finite,
        "chains_with_zero_accepted_H_changes": sum(value == 0 for value in h_changes),
        "accepted_H_changes_per_chain": h_changes,
        "marginal_attempts_equal_refreshes": (
            [chain.structural["marginal_attempts"] == chain.structural["ffbs_after_marginal"]
             for chain in chains] if arm == mfl.FULL_MARG else [True] * 4),
    }
    passed = bool(
        checks["all_diagnostics_finite"]
        and
        checks["max_rhat"] <= RHAT_GATE
        and checks["log_target_bulk_ess"] >= ESS_FLOORS["log_target_bulk"]
        and checks["log_target_tail_ess"] >= ESS_FLOORS["log_target_tail"]
        and checks["total_relations_bulk_ess"] >= ESS_FLOORS["total_relations_bulk"]
        and checks["minimum_remaining_invariant_bulk_ess"]
        >= ESS_FLOORS["remaining_invariant_bulk"]
        and checks["chains_with_zero_accepted_H_changes"] == 0
        and all(checks["marginal_attempts_equal_refreshes"])
    )
    return {
        "arm": arm, "checkpoint": int(checkpoint), "chain_iterations": iterations,
        "n_chains": 4, "n_retained_per_chain":
        [int(len(array["log_target"])) for array in arrays],
        "summaries": summaries, "checks": checks, "pass": passed,
        "formal_truth_free_verdict": "PASS" if passed else "FAIL",
        "runtime": [{"seconds": chain.seconds,
                     "seconds_per_sweep": chain.seconds / max(1, chain.state.iteration),
                     "structural": chain.structural, "movement": chain.movement}
                    for chain in chains],
    }


def _gate_path(arm: str, checkpoint: int) -> Path:
    return OUT / f"formal_gate_{arm.lower().replace('-', '_')}_{checkpoint}.json"


def _completed_terminal(arm: str) -> bool:
    return (OUT / f"terminal_{arm.lower().replace('-', '_')}.json").exists()


def _record_terminal_if_needed(arm: str, checkpoint: int, history: list[bool]) -> bool:
    terminal = checkpoint == CHECKPOINTS[-1] or (len(history) >= 2 and history[-1]
                                                  and history[-2])
    if terminal:
        _write_json(OUT / f"terminal_{arm.lower().replace('-', '_')}.json", {
            "arm": arm, "terminal_checkpoint": checkpoint,
            "reason": "two consecutive PASS checkpoints" if checkpoint != CHECKPOINTS[-1]
                      else "registered 100k ceiling",
            "gate_history": history,
            "truth_unsealed": False,
        })
    return terminal


def _read_gate(arm: str, checkpoint: int) -> dict | None:
    path = _gate_path(arm, checkpoint)
    if not path.exists():
        return None
    gate = json.loads(path.read_text())
    if gate.get("arm") != arm or int(gate.get("checkpoint", -1)) != int(checkpoint):
        raise RuntimeError(f"invalid or stale formal gate record: {path}")
    if gate.get("chain_iterations") != [int(checkpoint)] * 4:
        raise RuntimeError(f"gate does not certify exact checkpoint state: {path}")
    return gate


def _gate_history(arm: str) -> list[bool]:
    history = []
    saw_gap = False
    for checkpoint in CHECKPOINTS:
        gate = _read_gate(arm, checkpoint)
        if gate is None:
            saw_gap = True
        elif saw_gap:
            raise RuntimeError(f"noncontiguous formal gate ladder for {arm}")
        else:
            history.append(bool(gate["pass"]))
    return history


def _chain_iterations(arm: str) -> list[int]:
    return [int(_load_or_create_chain(arm, index).state.iteration) for index in range(4)]


def _assert_gate_checkpoint_lineage(arm: str, history: list[bool]) -> None:
    """Require the saved chains to descend from the last persisted gate.

    A gate is a property of the four exact checkpoint states that produced it.  On a
    resume, silently replacing a missing/older checkpoint and advancing it directly to
    the next gate would splice diagnostics from different chains into one stopping
    history.  Fresh arms have no history and may still start at iteration zero.
    """
    if not history:
        return
    previous_checkpoint = int(CHECKPOINTS[len(history) - 1])
    missing = [str(_chain_path(arm, index)) for index in range(4)
               if not _chain_path(arm, index).is_file()]
    if missing:
        raise RuntimeError(
            f"{arm} has gate history through {previous_checkpoint} but is missing its "
            f"checkpoint(s): {missing}; do not restart or splice a gated chain")
    iterations = _chain_iterations(arm)
    if iterations != [previous_checkpoint] * 4:
        raise RuntimeError(
            f"{arm} checkpoint lineage does not match its last recorded "
            f"{previous_checkpoint} gate: {iterations}; do not restart or splice a "
            "gated chain")


def _assert_launch_manifest() -> dict:
    path = OUT / "launch_manifest.json"
    if not path.exists():
        raise RuntimeError("formal launch requires a committed launch_manifest.json; run "
                           "--prepare-launch and commit it first")
    relative = str(path.relative_to(ROOT))
    if _head_bytes(relative) != path.read_bytes():
        raise RuntimeError("formal launch manifest is not byte-identical to the committed "
                           "HEAD launch record; run --prepare-launch and commit it first")
    manifest = json.loads(path.read_text())
    expected = _source_manifest()
    if manifest.get("source_hashes") != expected:
        raise RuntimeError("source hash differs from frozen launch manifest; do not launch")
    for source_relative, digest in manifest["source_hashes"].items():
        source_path = ROOT / source_relative
        if not source_path.is_file() or _sha256(source_path) != digest:
            raise RuntimeError(f"launch source path drifted: {source_relative}")
        if _head_bytes(source_relative) != source_path.read_bytes():
            raise RuntimeError(f"launch source path is not committed at HEAD: {source_relative}")
    if manifest.get("prereg_sha256") != _sha256(PREREG_JSON):
        raise RuntimeError("preregistration hash differs from frozen launch manifest")
    live_corpus = _corpus_provenance(mfl.load_frozen_observed_corpus(CORPUS_DIR))
    if manifest.get("corpus") != live_corpus:
        raise RuntimeError("frozen corpus provenance differs from launch manifest")
    if manifest.get("starts") != start_manifest():
        raise RuntimeError("paired start hashes differ from frozen launch manifest")
    if manifest.get("runtime") != _runtime_manifest():
        raise RuntimeError("Python/NumPy/SciPy runtime differs from frozen launch manifest")
    if manifest.get("source_import_closure") != sorted(_hpop_import_closure()):
        raise RuntimeError("resolved HPOP import closure differs from frozen launch manifest")
    if manifest.get("discarded_precheckpoint_attempt_sha256") != _sha256(
            DISCARDED_PREFIX_RECORD):
        raise RuntimeError("discarded precheckpoint-attempt record differs from launch manifest")
    discarded_relative = str(DISCARDED_PREFIX_RECORD.relative_to(ROOT))
    if _head_bytes(discarded_relative) != DISCARDED_PREFIX_RECORD.read_bytes():
        raise RuntimeError("discarded precheckpoint-attempt record is not committed at HEAD")
    if manifest.get("prelaunch_artifacts") != _validated_prelaunch_artifact_hashes(
            manifest.get("source_commit")):
        raise RuntimeError("prelaunch audit/validation/smoke artifact hashes differ from launch record")
    for artifact in (PRELAUNCH_AUDIT, PRELAUNCH_VALIDATION, SMOKE_REPORT):
        relative_artifact = str(artifact.relative_to(ROOT))
        if _head_bytes(relative_artifact) != artifact.read_bytes():
            raise RuntimeError(f"prelaunch artifact is not committed at HEAD: {relative_artifact}")
    _assert_runtime_truth_seal()
    return manifest


def run_formal(workers: int) -> None:
    _assert_launch_manifest()
    OUT.mkdir(parents=True, exist_ok=True)
    CHAIN_DIR.mkdir(parents=True, exist_ok=True)
    histories = {arm: _gate_history(arm) for arm in ARMS}
    # A crash after writing a passing gate but before writing terminal status must not
    # erase its registered result.  Reconcile it once, without recomputing a gate.
    for arm, history in histories.items():
        if history:
            _record_terminal_if_needed(arm, CHECKPOINTS[len(history) - 1], history)
    # An existing gate may only be extended from the exact four checkpoint states that
    # produced the preceding gate.  In particular, `_load_or_create_chain` must never
    # turn a missing checkpoint into a fresh chain after a gate has been recorded.
    for arm, history in histories.items():
        if history and not _completed_terminal(arm):
            _assert_gate_checkpoint_lineage(arm, history)
    for gate_index, checkpoint in enumerate(CHECKPOINTS):
        active = [arm for arm in ARMS if not _completed_terminal(arm)]
        if not active:
            break
        to_advance = []
        for arm in active:
            if gate_index < len(histories[arm]):
                continue
            if gate_index > len(histories[arm]):
                raise RuntimeError(f"noncontiguous in-memory formal gate history for {arm}")
            iterations = _chain_iterations(arm)
            if any(iteration > checkpoint for iteration in iterations):
                raise RuntimeError(
                    f"{arm} has advanced beyond its missing {checkpoint} gate: {iterations}; "
                    "do not reconstruct a gate from a later state")
            to_advance.append(arm)
        if to_advance:
            print("[FULL-LATENT] advancing " + ", ".join(to_advance)
                  + f" chains to {checkpoint:,} sweeps", flush=True)
            advance_formal(to_advance, checkpoint, workers)
        for arm in to_advance:
            iterations = _chain_iterations(arm)
            if iterations != [checkpoint] * 4:
                raise RuntimeError(f"{arm} did not reach exact checkpoint {checkpoint}: {iterations}")
            gate = arm_gate(arm, checkpoint)
            # The formal truth-free PASS/FAIL is written first; no recovery/held-out
            # code is present in this runner.
            _write_json(_gate_path(arm, checkpoint), gate)
            histories[arm].append(bool(gate["pass"]))
            _record_terminal_if_needed(arm, checkpoint, histories[arm])
            print(f"[{arm}] formal truth-free gate at {checkpoint:,}: "
                  f"{gate['formal_truth_free_verdict']}", flush=True)
    terminal = {arm: _completed_terminal(arm) for arm in ARMS}
    _write_json(OUT / "formal_status.json", {
        "terminal": terminal, "gate_history": histories,
        "truth_unsealed": False,
        "next_step": ("terminal recovery may be explicitly requested" if all(terminal.values())
                      else "resume only from exact checkpoints; no protocol changes"),
    })


def run_smoke() -> dict:
    """Nonconfirmatory observed-prefix smoke; never touches formal recovery truth."""
    corpus = mfl.load_frozen_observed_corpus(CORPUS_DIR)
    traces = corpus.train[:4]
    fixed = _fixed()
    model = mfl.build_full_latent_model(traces, fixed)
    probes = mfl.select_truth_free_probes(traces, corpus.corpus_hash + "-smoke",
                                           boundary_count=6, coskill_count=8,
                                           recovery_coskill_count=12)
    results = {}
    smoke_dir = OUT / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    for offset, arm in enumerate((mfl.FULL_COND, mfl.FULL_MARG)):
        pi, p = mfl.draw_initial_pi_p(model, 6_206_301 + offset)
        u = mfl.make_u_start(0, 6_206_311 + offset, 0.5, fixed,
                             model.n_skills, model.n_roles, 2)
        start = mfl.initial_full_latent_state(model, u, pi, p, fixed)
        config = mfl.FullLatentConfig(arm, structural_cadence=1,
                                      structural_scale=STRUCTURAL_SCALE)
        full = mfl.FullLatentChain(mfl.FullLatentSampler(model, fixed, config), start,
                                   6_206_321 + offset, burn_in=2, thin=1, probes=probes)
        full.advance(20)
        checkpoint = smoke_dir / f"{arm.lower().replace('-', '_')}.npz"
        split = mfl.FullLatentChain(mfl.FullLatentSampler(model, fixed, config), start,
                                    6_206_321 + offset, burn_in=2, thin=1, probes=probes)
        split.advance(10, checkpoint_path=checkpoint, checkpoint_every=5)
        resumed = mfl.FullLatentChain.load(
            checkpoint, mfl.FullLatentSampler(model, fixed, config))
        resumed.advance(20)
        exact_resume = (full.state.to_dict() == resumed.state.to_dict()
                        and all(np.array_equal(full.arrays()[key], resumed.arrays()[key])
                                for key in full.arrays()))
        finite = all(np.isfinite(value).all() for value in (
            np.asarray(full.arrays()["log_target"]), np.asarray(full.pi_draws),
            np.asarray(full.p_draws)))
        pi_moved = not np.array_equal(full.pi_draws[0], full.pi_draws[-1])
        p_moved = not np.array_equal(full.p_draws[0], full.p_draws[-1])
        results[arm] = {
            "no_nan": finite, "pi_moved": pi_moved, "P_moved": p_moved,
            "paths_moved": full.movement["states_changed"] > 0,
            "structural_attempts": full.structural["attempts"],
            "structural_accepts": full.structural["accepts"],
            "exact_save_resume": exact_resume,
            "marginal_attempts_equal_refreshes": (
                full.structural["marginal_attempts"]
                == full.structural["ffbs_after_marginal"]),
        }
    payload = {
        "kind": "NONCONFIRMATORY observed-prefix smoke; no truth recovery",
        "n_traces": len(traces), "sweeps": 20, "results": results,
        "pass": all(all(value for key, value in row.items()
                          if key not in {"structural_accepts"})
                    for row in results.values()),
    }
    _write_json(smoke_dir / "smoke.json", payload)
    return payload


def prepare_launch() -> dict:
    """Create, but do not execute, the committed formal launch record."""
    if any(CHAIN_DIR.glob("*.npz")):
        raise RuntimeError("refusing to prepare a fresh launch over existing chain checkpoints")
    audit = prelaunch_audit()
    _write_json(PRELAUNCH_AUDIT, audit)
    source_commit = _git("rev-parse", "HEAD")
    prelaunch_artifacts = _validated_prelaunch_artifact_hashes(source_commit)
    manifest = {
        "condition": "FULL-LATENT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "git_status_at_prepare": _git_status(),
        "prereg_sha256": _sha256(PREREG_JSON),
        "source_hashes": _source_manifest(),
        "source_import_closure": sorted(_hpop_import_closure()),
        "runtime": _runtime_manifest(),
        "discarded_precheckpoint_attempt_sha256": _sha256(DISCARDED_PREFIX_RECORD),
        "prelaunch_artifacts": prelaunch_artifacts,
        "corpus": audit["corpus"],
        "starts": audit["starts"],
        "chain_seeds": {arm: list(seeds) for arm, seeds in ARMS.items()},
        "proposal": audit["proposal"],
        "checkpoint_ladder": list(CHECKPOINTS),
        "burn_in": BURN_IN, "thin": THIN, "checkpoint_every": CHECKPOINT_EVERY,
        "truth_unsealed": False,
        "no_C_prime_state_or_truth_initialization": True,
    }
    _write_json(OUT / "launch_manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--smoke", action="store_true")
    action.add_argument("--audit", action="store_true")
    action.add_argument("--prepare-launch", action="store_true")
    action.add_argument("--launch-formal", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8")
    if args.smoke:
        print(json.dumps(run_smoke(), indent=2, sort_keys=True))
    elif args.audit:
        audit = prelaunch_audit()
        _write_json(OUT / "prelaunch_audit.json", audit)
        print(json.dumps(audit, indent=2, sort_keys=True))
    elif args.prepare_launch:
        print(json.dumps(prepare_launch(), indent=2, sort_keys=True))
    elif args.launch_formal:
        run_formal(args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
