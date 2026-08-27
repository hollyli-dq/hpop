"""Run the sealed, terminal-only recovery for the FULL-LATENT experiment.

This program is deliberately separate from the formal sampler.  Its first action is
to certify that *both* FULL-COND and FULL-MARG have terminal records under the
registered truth-free stopping rule.  Only after that certificate succeeds does it
import model code, read the formal checkpoints, or open the truth manifest / hidden
NPZ fields.  Thus a partial or still-running formal experiment cannot accidentally
inspect recovery truth through this entry point.

The recovery calculation is fixed by ``PREREG_FULL_LATENT.json``:

* every retained draw is aligned to truth once, using closure-Hamming Hungarian
  matching with the deterministic tie rule in ``full_latent_recovery``;
* that one draw-level mapping is shared by H, pi, and both axes of P;
* segmentation and z recovery use the exact online boundary and co-skill
  accumulators checkpointed during sampling;
* held-out prediction uses every retained U/pi/P draw and computes
  ``log Z - log C_J`` before the per-trace log-mean-exp.

Run only after the formal runner has written both terminal artifacts:

    PYTHONPATH=src .venv/bin/python scripts/run_matched_full_latent_recovery.py \
        --run-terminal-recovery --workers 4

It writes ``terminal_recovery.json`` and ``terminal_recovery_table.md`` under the
formal result directory.  It never modifies formal checkpoints or truth-free gates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
# Child workers need the package path too, but this line itself performs no HPOP import.
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "mcmc_original" / "matched_full_latent"
CORPUS_DIR = ROOT / "results" / "mcmc_original" / "matched_synthetic_formal_corpus"
CHAIN_DIR_NAME = "formal_chains"
ARMS = ("FULL-COND", "FULL-MARG")
CHECKPOINTS = (30_000, 50_000, 75_000, 100_000)
N_CHAINS = 4
BOUNDARY_THRESHOLD = 0.5
CO_SKILL_THRESHOLD = 0.5


def _slug(arm: str) -> str:
    return str(arm).lower().replace("-", "_")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(f"required artifact is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_json_durable(path: Path, payload: dict) -> None:
    """Atomically write and fsync a safety-critical provenance event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    # A directory fsync makes the rename durable on the POSIX filesystem used for the
    # formal run.  Retain portability for filesystems that do not expose O_DIRECTORY.
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _relative_to_root(path: Path, root: Path) -> str:
    """Return a portable repository-relative path, rejecting path escapes."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"artifact lies outside repository root: {path}") from exc


def _head_blob(root: Path, relative_path: str) -> bytes:
    """Read one tracked HEAD blob without touching any HPOP or truth artifact."""
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"], cwd=root,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot read HEAD blob for frozen source {relative_path}: {detail}")
    return bytes(completed.stdout)


def _git_head(root: Path) -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               check=False)
    if completed.returncode != 0:
        raise RuntimeError("cannot resolve repository HEAD for launch provenance")
    return completed.stdout.decode("utf-8").strip()


def _runtime_manifest() -> dict:
    """The runtime identity frozen by the formal launch, without importing HPOP."""
    import scipy

    return {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__}


def _required_launch_starts_and_seeds(launch: dict) -> tuple[dict, dict]:
    """Validate the exact start hash and chain-seed grid before checkpoint loading."""
    starts = launch.get("starts")
    chain_seeds = launch.get("chain_seeds")
    if not isinstance(starts, dict) or not isinstance(starts.get("pairing"), dict):
        raise RuntimeError("frozen launch manifest is missing starts.pairing")
    if not isinstance(chain_seeds, dict):
        raise RuntimeError("frozen launch manifest is missing chain_seeds")
    pairing = starts["pairing"]
    for index in range(N_CHAINS):
        paired = pairing.get(str(index))
        if not isinstance(paired, dict):
            raise RuntimeError(f"frozen launch manifest is missing paired start {index}")
        for arm in ARMS:
            start_hash = paired.get(arm)
            if not isinstance(start_hash, str) or len(start_hash) != 64:
                raise RuntimeError(f"frozen launch manifest has no exact start hash for "
                                   f"{arm} chain {index}")
    for arm in ARMS:
        seeds = chain_seeds.get(arm)
        if not isinstance(seeds, list) or len(seeds) != N_CHAINS:
            raise RuntimeError(f"frozen launch manifest has no four-chain seed list for {arm}")
        for index, seed in enumerate(seeds):
            if isinstance(seed, bool):
                raise RuntimeError(f"frozen launch manifest has invalid {arm} seed {index}")
            try:
                int(seed)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"frozen launch manifest has invalid {arm} seed {index}") from exc
    return pairing, chain_seeds


def assert_frozen_launch_integrity(out: Path = OUT, *, root: Path = ROOT) -> dict:
    """Verify the committed launch record and every frozen source byte before imports.

    The launch manifest itself must still be byte-identical to the HEAD blob.  Every
    registered source hash is then checked against both the current worktree file and
    its HEAD blob.  This catches a local post-launch edit, an uncommitted manifest, or
    a branch/checkout mismatch before model code or terminal truth can be opened.
    """
    out, root = Path(out), Path(root)
    manifest_path = out / "launch_manifest.json"
    manifest_relative = _relative_to_root(manifest_path, root)
    try:
        disk_manifest = manifest_path.read_bytes()
    except FileNotFoundError as exc:
        raise RuntimeError("terminal recovery requires the committed launch_manifest.json") from exc
    if disk_manifest != _head_blob(root, manifest_relative):
        raise RuntimeError("launch manifest differs from its committed HEAD blob")
    try:
        launch = json.loads(disk_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("committed launch manifest is not valid UTF-8 JSON") from exc
    if not isinstance(launch, dict):
        raise RuntimeError("committed launch manifest must be a JSON object")
    expected_runtime = launch.get("runtime")
    actual_runtime = _runtime_manifest()
    if expected_runtime != actual_runtime:
        raise RuntimeError("current Python/NumPy/SciPy runtime differs from the frozen launch")
    source_hashes = launch.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise RuntimeError("frozen launch manifest has no source_hashes")
    driver_path = "scripts/run_matched_full_latent_recovery.py"
    if driver_path not in source_hashes:
        raise RuntimeError("frozen source manifest does not include the terminal recovery driver")
    verified_paths = []
    for relative_path, expected_hash in sorted(source_hashes.items()):
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise RuntimeError("frozen source manifest contains a malformed path/hash entry")
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"frozen source path escapes repository: {relative_path}") from exc
        if not candidate.is_file():
            raise RuntimeError(f"frozen source path is missing from worktree: {relative_path}")
        disk_hash = _sha256_bytes(candidate.read_bytes())
        head_hash = _sha256_bytes(_head_blob(root, relative_path))
        if disk_hash != expected_hash:
            raise RuntimeError(f"frozen source disk hash differs for {relative_path}")
        if head_hash != expected_hash:
            raise RuntimeError(f"frozen source HEAD hash differs for {relative_path}")
        verified_paths.append(relative_path)
    pairing, chain_seeds = _required_launch_starts_and_seeds(launch)
    return {
        "launch": launch,
        "launch_manifest_path": str(manifest_path),
        "launch_manifest_sha256": _sha256_bytes(disk_manifest),
        "head_commit": _git_head(root),
        "launch_source_commit": launch.get("source_commit"),
        "runtime": actual_runtime,
        "verified_source_paths": verified_paths,
        "paired_starts": pairing,
        "chain_seeds": chain_seeds,
    }


def _gate_path(out: Path, arm: str, checkpoint: int) -> Path:
    return out / f"formal_gate_{_slug(arm)}_{int(checkpoint)}.json"


def _terminal_path(out: Path, arm: str) -> Path:
    return out / f"terminal_{_slug(arm)}.json"


def assert_both_arms_terminal(out: Path = OUT) -> dict:
    """Return a truth-free terminal certificate or refuse before any truth I/O.

    This function deliberately only reads records in ``out``.  In particular it does
    not import HPOP modules, open the corpus directory, enumerate NPZ contents, or
    construct a path to the truth manifest.  Keeping this small, standard-library-only
    gate makes the seal auditable and easy to test.
    """
    out = Path(out)
    certificate: dict[str, Any] = {"arms": {}, "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    formal_status_path = out / "formal_status.json"
    formal_status = _read_json(formal_status_path) if formal_status_path.exists() else None
    if formal_status is not None and formal_status.get("truth_unsealed") not in (False, None):
        raise RuntimeError("formal status records truth as already unsealed; refusing a second recovery")

    for arm in ARMS:
        terminal = _read_json(_terminal_path(out, arm))
        if terminal.get("arm") != arm:
            raise RuntimeError(f"terminal artifact names the wrong arm for {arm}")
        if terminal.get("truth_unsealed") is not False:
            raise RuntimeError(f"terminal artifact for {arm} is not a sealed formal record")
        checkpoint = terminal.get("terminal_checkpoint")
        try:
            valid_checkpoint = (not isinstance(checkpoint, bool)
                                and int(checkpoint) in CHECKPOINTS)
        except (TypeError, ValueError):
            valid_checkpoint = False
        if not valid_checkpoint:
            raise RuntimeError(f"terminal artifact for {arm} has an invalid checkpoint")
        checkpoint = int(checkpoint)
        checkpoint_index = CHECKPOINTS.index(checkpoint)
        history = terminal.get("gate_history")
        if (not isinstance(history, list) or len(history) != checkpoint_index + 1
                or any(not isinstance(value, bool) for value in history)):
            raise RuntimeError(f"terminal artifact for {arm} has an invalid gate history")

        gate_records = []
        for index, expected_checkpoint in enumerate(CHECKPOINTS[:checkpoint_index + 1]):
            gate = _read_json(_gate_path(out, arm, expected_checkpoint))
            try:
                gate_checkpoint = int(gate.get("checkpoint", -1))
            except (TypeError, ValueError):
                gate_checkpoint = -1
            if gate.get("arm") != arm or gate_checkpoint != expected_checkpoint:
                raise RuntimeError(f"gate artifact is stale or mismatched for {arm} at "
                                   f"{expected_checkpoint}")
            if gate.get("chain_iterations") != [expected_checkpoint] * N_CHAINS:
                raise RuntimeError(f"gate artifact does not certify four exact {arm} "
                                   f"checkpoints at {expected_checkpoint}")
            if bool(gate.get("pass")) != history[index]:
                raise RuntimeError(f"terminal gate history disagrees with {arm} gate "
                                   f"{expected_checkpoint}")
            gate_records.append({"checkpoint": expected_checkpoint,
                                 "pass": bool(gate.get("pass")),
                                 "path": str(_gate_path(out, arm, expected_checkpoint))})

        # Before the ceiling, only the registered two-consecutive-PASS route can be
        # terminal.  At 100k the preregistered ceiling itself makes the arm terminal.
        if checkpoint != CHECKPOINTS[-1] and not (len(history) >= 2 and history[-1] and history[-2]):
            raise RuntimeError(f"{arm} terminal record does not satisfy two consecutive PASS gates")
        if formal_status is not None:
            if formal_status.get("terminal", {}).get(arm) is not True:
                raise RuntimeError(f"formal status does not certify {arm} as terminal")

        certificate["arms"][arm] = {
            "terminal_path": str(_terminal_path(out, arm)),
            "terminal_checkpoint": checkpoint,
            "reason": terminal.get("reason"),
            "gate_history": history,
            "gates": gate_records,
        }
    certificate["formal_status_path"] = str(formal_status_path) if formal_status is not None else None
    certificate["truth_unsealed_before_recovery"] = False
    return certificate


def record_terminal_recovery_unseal_started(out: Path, certificate: dict) -> dict:
    """Durably record unseal intent before any model import or truth-related I/O.

    A crash after this point leaves a deliberately visible event rather than a silent
    possibility that truth was opened.  The first event is immutable across a restart;
    a restart may reuse it only when it certifies the same terminal records.
    """
    path = Path(out) / "terminal_recovery_unseal_started.json"
    if path.exists():
        existing = _read_json(path)
        existing_certificate = dict(existing.get("terminal_certificate", {}))
        current_certificate = dict(certificate)
        # The certificate's audit timestamp is intentionally fresh on each process
        # start; it is not part of the immutable terminal-artifact binding.
        existing_certificate.pop("checked_at_utc", None)
        current_certificate.pop("checked_at_utc", None)
        if existing_certificate != current_certificate:
            raise RuntimeError("existing unseal-started event certifies different terminal artifacts")
        if existing.get("truth_unsealed") is not True:
            raise RuntimeError("existing unseal-started event is malformed")
        return existing
    event = {
        "event": "FULL-LATENT terminal recovery unseal started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "truth_unsealed": True,
        "truth_opened_at_record_time": False,
        "intent": "terminal recovery may open synthetic truth only after this durable record",
        "terminal_certificate": certificate,
    }
    _write_json_durable(path, event)
    return event


def _post_terminal_imports():
    """Import model/recovery code only after :func:`assert_both_arms_terminal`."""
    from hpop.mcmc_original import matched_full_latent as mfl
    from hpop.mcmc_original import full_latent_recovery as recovery
    return mfl, recovery


def _expected_checkpoint_draws(checkpoint: int, burn_in: int, thin: int) -> int:
    """Retentions obey ``sweep > burn_in`` and ``(sweep-burn_in) % thin == 0``."""
    if checkpoint <= burn_in:
        return 0
    return (int(checkpoint) - int(burn_in)) // int(thin)


def _assert_chain_draw_shapes(chain, model, *, arm: str, index: int,
                              terminal_checkpoint: int, expected_probes: dict,
                              start_hash: str, chain_seed: int) -> None:
    """Validate that a loaded checkpoint is the exact formal retained artifact."""
    if int(chain.state.iteration) != int(terminal_checkpoint):
        raise RuntimeError(f"{arm} chain {index} is at {chain.state.iteration}, not terminal "
                           f"checkpoint {terminal_checkpoint}")
    expected_draws = _expected_checkpoint_draws(terminal_checkpoint, chain.burn_in, chain.thin)
    if chain.retained_draws != expected_draws or expected_draws < 1:
        raise RuntimeError(f"{arm} chain {index} retained {chain.retained_draws} draws; "
                           f"expected {expected_draws}")
    if chain.start_metadata.get("arm") != arm or int(chain.start_metadata.get("start_index", -1)) != index:
        raise RuntimeError(f"{arm} chain {index} has mismatched formal start metadata")
    if chain.start_metadata.get("start_state_sha256") != start_hash:
        raise RuntimeError(f"{arm} chain {index} start hash differs from launch manifest")
    if int(chain.seed) != int(chain_seed):
        raise RuntimeError(f"{arm} chain {index} RNG seed differs from launch manifest")
    if chain.probes != expected_probes:
        raise RuntimeError(f"{arm} chain {index} has truth-free probe drift")

    n = int(chain.retained_draws)
    expected_u_shape = (n, int(model.n_skills), int(model.n_roles), 2)
    expected_pi_shape = (n, int(model.n_skills))
    expected_p_shape = (n, int(model.n_skills), int(model.n_skills))
    if np.asarray(chain.u_draws).shape != expected_u_shape:
        raise RuntimeError(f"{arm} chain {index} has invalid retained U shape")
    if np.asarray(chain.pi_draws).shape != expected_pi_shape:
        raise RuntimeError(f"{arm} chain {index} has invalid retained pi shape")
    if np.asarray(chain.p_draws).shape != expected_p_shape:
        raise RuntimeError(f"{arm} chain {index} has invalid retained P shape")
    if len(chain.relation_indicators) != n:
        raise RuntimeError(f"{arm} chain {index} relation-indicator archive disagrees with draws")
    if len(chain.boundary_sums) != len(model.traces):
        raise RuntimeError(f"{arm} chain {index} boundary accumulator trace count is invalid")
    if any(np.asarray(row).shape != (len(trace) - 1,)
           for row, trace in zip(chain.boundary_sums, model.traces)):
        raise RuntimeError(f"{arm} chain {index} boundary accumulator shape is invalid")
    recovery_count = len(expected_probes["recovery_coskill"])
    if (np.asarray(chain.recovery_coskill_sums).shape != (recovery_count,)
            or np.asarray(chain.recovery_same_segment_sums).shape != (recovery_count,)):
        raise RuntimeError(f"{arm} chain {index} recovery co-skill accumulator shape is invalid")
    if (any(not np.all(np.isfinite(np.asarray(row, dtype=float)))
            for row in chain.boundary_sums)
            or np.any(~np.isfinite(chain.recovery_coskill_sums))
            or np.any(~np.isfinite(chain.recovery_same_segment_sums))):
        raise RuntimeError(f"{arm} chain {index} has non-finite online recovery accumulators")
    for name, values in chain.arrays().items():
        if len(values) != n:
            raise RuntimeError(f"{arm} chain {index} retained summary {name} disagrees with draws")


def _load_exact_formal_chains_after_terminal(out: Path, corpus_dir: Path,
                                             terminal_certificate: dict, launch: dict):
    """Load observed data and all terminal checkpoints after the seal is released."""
    mfl, _ = _post_terminal_imports()
    out, corpus_dir = Path(out), Path(corpus_dir)
    corpus = mfl.load_frozen_observed_corpus(corpus_dir)
    launch_corpus = launch.get("corpus", {})
    for name, value in (("corpus_hash", corpus.corpus_hash),
                        ("train_hash", corpus.train_hash),
                        ("heldout_hash", corpus.heldout_hash)):
        if launch_corpus.get(name) != value:
            raise RuntimeError(f"launch manifest {name} does not match frozen corpus")
    fixed = mfl.FullLatentFixed()
    model = mfl.build_full_latent_model(corpus.train, fixed)
    expected_probes = mfl.select_truth_free_probes(
        corpus.train, corpus.corpus_hash, boundary_count=32, coskill_count=64,
        recovery_coskill_count=256,
    )
    starts, chain_seeds = _required_launch_starts_and_seeds(launch)
    records = {arm: [] for arm in ARMS}
    for arm in ARMS:
        checkpoint = int(terminal_certificate["arms"][arm]["terminal_checkpoint"])
        config = mfl.FullLatentConfig(arm=arm, structural_cadence=10,
                                      structural_scale=0.5, table_source="batched")
        for index in range(N_CHAINS):
            path = out / CHAIN_DIR_NAME / f"{_slug(arm)}_{index}.npz"
            if not path.exists():
                raise RuntimeError(f"terminal formal checkpoint is missing: {path}")
            sampler = mfl.FullLatentSampler(model, fixed, config)
            chain = mfl.FullLatentChain.load(path, sampler)
            expected_start = starts[str(index)][arm]
            expected_seed = int(chain_seeds[arm][index])
            _assert_chain_draw_shapes(chain, model, arm=arm, index=index,
                                      terminal_checkpoint=checkpoint,
                                      expected_probes=expected_probes,
                                      start_hash=expected_start, chain_seed=expected_seed)
            records[arm].append({"index": index, "path": path, "chain": chain})
    return {
        "mfl": mfl,
        "corpus": corpus,
        "fixed": fixed,
        "train_model": model,
        "probes": expected_probes,
        "launch": launch,
        "records": records,
    }


def _load_truth_after_terminal(corpus_dir: Path, observed_train: Iterable[Iterable[int]],
                               expected_truth_hash: str, fixed, recovery) -> dict:
    """Open and validate recovery truth; caller must already hold terminal certificate."""
    corpus_dir = Path(corpus_dir)
    manifest = _read_json(corpus_dir / "truth_manifest.json")
    truth = manifest.get("truth")
    if not isinstance(truth, dict):
        raise RuntimeError("truth manifest has no truth object")
    actual_truth_hash = _sha256_bytes(json.dumps(
        truth, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    if actual_truth_hash != str(expected_truth_hash):
        raise RuntimeError("truth object hash does not match the frozen launch manifest")
    if manifest.get("truth_hash_sha256") != actual_truth_hash:
        raise RuntimeError("truth manifest self-reported hash does not match its truth object")

    truth_u = np.asarray(truth.get("u_by_skill"), dtype=float)
    truth_pi = np.asarray(truth.get("pi"), dtype=float)
    truth_p = np.asarray(truth.get("transition"), dtype=float)
    truth_closures = recovery.closure_stack_from_u(truth_u)
    recorded_closures = np.asarray(truth.get("h_by_skill"), dtype=bool)
    if recorded_closures.shape != truth_closures.shape or not np.array_equal(recorded_closures, truth_closures):
        raise RuntimeError("truth manifest closure representation disagrees with U truth")
    # Reuse the recovery validators rather than accepting malformed pi/P truth.
    recovery.pi_recovery_metrics(truth_pi, truth_pi)
    recovery.transition_recovery_metrics(truth_p, truth_p)
    role_maps = tuple(tuple(int(value) for value in row) for row in truth.get("role_maps", ()))
    identity = tuple(range(truth_u.shape[1]))
    if role_maps != (identity,) * truth_u.shape[0]:
        raise RuntimeError("formal recovery requires the registered identity role maps")
    expected_scalars = {
        "beta": fixed.beta, "omega": fixed.omega,
        "lambda_rep": fixed.lambda_rep, "lambda_back": fixed.lambda_back,
    }
    for name, expected in expected_scalars.items():
        if not math.isclose(float(truth.get("scalars", {}).get(name, float("nan"))),
                            float(expected), rel_tol=0.0, abs_tol=1e-14):
            raise RuntimeError(f"sealed truth {name} differs from the fixed formal model")
    for name, expected in (("epsilon", fixed.epsilon), ("delta_b", fixed.delta_b),
                           ("min_width", 3), ("max_width", 12)):
        if not math.isclose(float(truth.get(name, float("nan"))), float(expected),
                            rel_tol=0.0, abs_tol=1e-14):
            raise RuntimeError(f"sealed truth {name} differs from the fixed formal model")

    observed_train = tuple(tuple(int(v) for v in row) for row in observed_train)
    keys, occurrence_labels = [], []
    train_path = corpus_dir / "train_traces.npz"
    with np.load(train_path, allow_pickle=False) as data:
        n_traces = int(np.asarray(data["n_traces"])[0])
        if n_traces != len(observed_train):
            raise RuntimeError("sealed train truth trace count differs from observed corpus")
        for index, trace in enumerate(observed_train):
            tag = f"t{index:03d}"
            widths = np.asarray(data[f"{tag}_widths"], dtype=int)
            labels = np.asarray(data[f"{tag}_labels"], dtype=int)
            if widths.ndim != 1 or labels.shape != widths.shape or widths.size < 1:
                raise RuntimeError(f"invalid hidden truth segmentation for train trace {index}")
            if np.any(widths < 1) or int(widths.sum()) != len(trace):
                raise RuntimeError(f"hidden truth widths do not cover train trace {index}")
            if np.any(labels < 0) or np.any(labels >= truth_u.shape[0]):
                raise RuntimeError(f"hidden truth labels are out of range for train trace {index}")
            ends = np.cumsum(widths, dtype=int)
            keys.append(tuple((int(end), int(label)) for end, label in zip(ends, labels)))
            occurrence_labels.append(np.repeat(labels, widths).astype(int, copy=False))
    boundaries = recovery.boundary_truth_from_keys(
        keys, trace_lengths=[len(trace) for trace in observed_train])
    return {
        "manifest_path": str(corpus_dir / "truth_manifest.json"),
        "truth_hash_sha256": actual_truth_hash,
        "u_by_skill": truth_u,
        "closures": truth_closures,
        "pi": truth_pi,
        "transition": truth_p,
        "segmentation_keys": keys,
        "occurrence_labels": occurrence_labels,
        "boundaries": boundaries,
    }


def _quantile_summary(values) -> dict:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("cannot summarize an empty or non-finite draw series")
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=0)),
        "q025": float(np.quantile(array, 0.025)),
        "median": float(np.quantile(array, 0.5)),
        "q975": float(np.quantile(array, 0.975)),
    }


def _library_signature(closures: np.ndarray) -> tuple[bytes, ...]:
    """A label-free exact closure-library key for posterior-mode reporting."""
    closures = np.asarray(closures, dtype=bool)
    return tuple(sorted(np.ascontiguousarray(closures[k], dtype=np.uint8).tobytes()
                        for k in range(closures.shape[0])))


def _format_mapping(mapping: Iterable[int]) -> str:
    return "|".join(f"{learned}->{truth}" for learned, truth in enumerate(mapping))


def _draw_level_recovery(records: list[dict], truth: dict, recovery) -> dict:
    """Drawwise shared Hungarian alignment and all H/pi/P recovery summaries."""
    true_closures = truth["closures"]
    true_pi, true_p = truth["pi"], truth["transition"]
    truth_signature = _library_signature(true_closures)
    exact, aggregate_f1, aggregate_hamming = [], [], []
    per_skill_f1 = [[] for _ in range(true_closures.shape[0])]
    per_skill_hamming = [[] for _ in range(true_closures.shape[0])]
    pi_l1, pi_tv, pi_rmse = [], [], []
    p_rmse, p_frobenius, p_mean_row_tv, p_max_row_tv = [], [], [], []
    mapping_counts, tie_counts, library_counts = Counter(), Counter(), Counter()
    per_chain = []

    for record in records:
        chain = record["chain"]
        local_exact, local_f1 = [], []
        for u, pi, transition in zip(chain.u_draws, chain.pi_draws, chain.p_draws):
            learned_closures = recovery.closure_stack_from_u(u)
            # This is intentionally the *one* alignment call for this draw.  Every
            # H/pi/P metric below consumes its jointly aligned output.
            aligned = recovery.align_latent_draw(
                learned_closures, true_closures, pi, transition)
            closure = recovery.closure_recovery_metrics(aligned.closures, true_closures)
            pi_metric = recovery.pi_recovery_metrics(aligned.pi, true_pi)
            p_metric = recovery.transition_recovery_metrics(aligned.transition, true_p)
            is_exact = bool(closure["exact_unordered_library"])
            exact.append(float(is_exact))
            local_exact.append(float(is_exact))
            f1 = float(closure["aggregate_closure"]["closure_f1"])
            aggregate_f1.append(f1)
            local_f1.append(f1)
            aggregate_hamming.append(float(closure["aggregate_closure"]
                                           ["normalized_closure_hamming"]))
            for skill, row in enumerate(closure["per_skill"]):
                per_skill_f1[skill].append(float(row["closure_f1"]))
                per_skill_hamming[skill].append(float(row["normalized_closure_hamming"]))
            pi_l1.append(float(pi_metric["l1_error"]))
            pi_tv.append(float(pi_metric["total_variation_error"]))
            pi_rmse.append(float(pi_metric["rmse"]))
            p_rmse.append(float(p_metric["off_diagonal_rmse"]))
            p_frobenius.append(float(p_metric["frobenius_error"]))
            p_mean_row_tv.append(float(p_metric["mean_row_total_variation"]))
            p_max_row_tv.append(float(p_metric["max_row_total_variation"]))
            mapping_counts[_format_mapping(aligned.alignment.learned_to_truth)] += 1
            tie_counts[str(aligned.alignment.n_optimal_assignments)] += 1
            library_counts[_library_signature(learned_closures)] += 1
        per_chain.append({
            "chain": int(record["index"]),
            "n_retained_draws": int(chain.retained_draws),
            "exact_unordered_library_probability": float(np.mean(local_exact)),
            "mean_aggregate_closure_f1": float(np.mean(local_f1)),
        })

    modal_signature, modal_count = library_counts.most_common(1)[0]
    return {
        "n_retained_draws_total": len(exact),
        "alignment": {
            "rule": "closure-Hamming Hungarian; lexicographically smallest among tied K=3 optima",
            "mapping_frequency": dict(sorted(mapping_counts.items())),
            "n_optimal_assignment_frequency": dict(sorted(tie_counts.items())),
            "shared_for": ["H", "pi", "both axes of P"],
        },
        "unordered_library": {
            "exact_recovery_probability": float(np.mean(exact)),
            "n_exact_draws": int(np.sum(exact)),
            "per_chain": per_chain,
            "posterior_mode": {
                "posterior_probability": float(modal_count / len(exact)),
                "matches_truth_unordered_library": bool(modal_signature == truth_signature),
                "n_distinct_unordered_libraries": int(len(library_counts)),
            },
        },
        "closure": {
            "aggregate_closure_f1": _quantile_summary(aggregate_f1),
            "aggregate_normalized_closure_hamming": _quantile_summary(aggregate_hamming),
            "per_skill": [
                {"truth_skill": skill, "closure_f1": _quantile_summary(per_skill_f1[skill]),
                 "normalized_closure_hamming": _quantile_summary(per_skill_hamming[skill])}
                for skill in range(true_closures.shape[0])
            ],
        },
        "pi": {
            "l1_error": _quantile_summary(pi_l1),
            "total_variation_error": _quantile_summary(pi_tv),
            "rmse": _quantile_summary(pi_rmse),
        },
        "transition": {
            "off_diagonal_rmse": _quantile_summary(p_rmse),
            "frobenius_error": _quantile_summary(p_frobenius),
            "mean_row_total_variation": _quantile_summary(p_mean_row_tv),
            "max_row_total_variation": _quantile_summary(p_max_row_tv),
        },
    }


def _online_path_recovery(records: list[dict], truth: dict, recovery) -> dict:
    """Pool only exact online accumulators; raw sampled paths are never reconstructed."""
    retained = sum(int(record["chain"].retained_draws) for record in records)
    if retained < 1:
        raise RuntimeError("no retained draws available for online path recovery")
    n_traces = len(truth["boundaries"])
    boundary_sums = [np.zeros_like(np.asarray(truth["boundaries"][n]), dtype=float)
                     for n in range(n_traces)]
    co_skill_sums = None
    same_segment_sums = None
    for record in records:
        chain = record["chain"]
        for total, source in zip(boundary_sums, chain.boundary_sums):
            total += np.asarray(source, dtype=float)
        source_co = np.asarray(chain.recovery_coskill_sums, dtype=float)
        source_same = np.asarray(chain.recovery_same_segment_sums, dtype=float)
        co_skill_sums = source_co.copy() if co_skill_sums is None else co_skill_sums + source_co
        same_segment_sums = source_same.copy() if same_segment_sums is None else same_segment_sums + source_same
    boundary = recovery.boundary_recovery_from_accumulators(
        boundary_sums, retained, truth["boundaries"], threshold=BOUNDARY_THRESHOLD)
    pairs = records[0]["chain"].probes["recovery_coskill"]
    same_skill_truth = recovery.co_skill_truth_from_labels(truth["occurrence_labels"], pairs)
    co_skill = recovery.co_skill_recovery_from_accumulators(
        co_skill_sums, retained, same_skill_truth, threshold=CO_SKILL_THRESHOLD)
    # Same-segment posterior probabilities are not a preregistered primary truth metric,
    # but checking their accumulator range ensures all persisted online arrays were read.
    if np.any(same_segment_sums < -1e-10) or np.any(same_segment_sums > retained + 1e-10):
        raise RuntimeError("same-segment online accumulator lies outside retained-draw range")
    return {
        "n_retained_draws_total": retained,
        "boundary": boundary,
        "co_skill": co_skill,
        "source": "exact per-chain online boundary/co-skill accumulators; no raw path archive",
    }


def _heldout_log_z_worker(payload):
    """One post-terminal chain's exact all-draw held-out forward normalisers."""
    (index, u_draws, pi_draws, p_draws, heldout_traces, fixed_values) = payload
    from hpop.mcmc_original import matched_full_latent as mfl
    from hpop.mcmc_original.collapsed_u_likelihood import CollapsedULikelihood
    from hpop.mcmc_original.stage6e_state import Stage6EState

    fixed = mfl.FullLatentFixed(**fixed_values)
    model = mfl.build_full_latent_model(heldout_traces, fixed)
    likelihood = CollapsedULikelihood(model)
    u_draws, pi_draws, p_draws = (np.asarray(u_draws), np.asarray(pi_draws),
                                  np.asarray(p_draws))
    if not (u_draws.shape[0] == pi_draws.shape[0] == p_draws.shape[0]):
        raise RuntimeError("worker received misaligned retained draws")
    state = Stage6EState(segmentations=(), u_by_skill=u_draws[0], rho=fixed.rho_0,
                         beta=fixed.beta, omega=fixed.omega,
                         lambda_rep=fixed.lambda_rep, lambda_back=fixed.lambda_back,
                         pi=pi_draws[0], transition=p_draws[0])
    output = np.empty((u_draws.shape[0], len(heldout_traces)), dtype=float)
    began = time.perf_counter()
    for draw in range(u_draws.shape[0]):
        state.u_by_skill = np.asarray(u_draws[draw], dtype=float)
        state.pi = np.asarray(pi_draws[draw], dtype=float)
        state.transition = np.asarray(p_draws[draw], dtype=float)
        mfl.validate_pi_p(state, model)
        output[draw] = likelihood.log_z_per_trace(state)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("held-out forward recursion returned non-finite normalizers")
    return {"index": int(index), "log_z": output,
            "seconds": float(time.perf_counter() - began),
            "forward_evaluations": int(likelihood.evaluations),
            "cache_hits": int(likelihood.cache_hits)}


def _heldout_recovery(records: list[dict], corpus, fixed, recovery, workers: int) -> dict:
    """Exact all-retained-draw held-out posterior predictive calculation."""
    payloads = [
        (record["index"], np.asarray(record["chain"].u_draws),
         np.asarray(record["chain"].pi_draws), np.asarray(record["chain"].p_draws),
         corpus.heldout, fixed.as_dict())
        for record in records
    ]
    outputs = []
    if int(workers) == 1:
        outputs = [_heldout_log_z_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(payloads))) as executor:
            futures = {executor.submit(_heldout_log_z_worker, payload): payload[0]
                       for payload in payloads}
            for future in as_completed(futures):
                outputs.append(future.result())
    outputs.sort(key=lambda row: row["index"])
    log_z = np.concatenate([row["log_z"] for row in outputs], axis=0)
    result = recovery.heldout_posterior_predictive(
        log_z, [len(trace) for trace in corpus.heldout], delta_b=fixed.delta_b,
        min_width=3, max_width=12,
    )
    result.update({
        "per_chain": [{"chain": row["index"], "n_retained_draws": int(row["log_z"].shape[0]),
                       "seconds": row["seconds"],
                       "forward_evaluations": row["forward_evaluations"],
                       "cache_hits": row["cache_hits"]}
                      for row in outputs],
        "n_retained_draws_total": int(log_z.shape[0]),
        "exact_draw_policy": "all retained U/pi/P draws from all four terminal chains",
    })
    return result


def _runtime_summary(records: list[dict]) -> dict:
    per_chain = []
    for record in records:
        seconds = float(record["chain"].seconds)
        sweeps = int(record["chain"].state.iteration)
        seconds_per_sweep = seconds / max(1, sweeps)
        per_chain.append({
            "chain": int(record["index"]),
            "seconds": seconds,
            "sweeps": sweeps,
            "seconds_per_sweep": seconds_per_sweep,
            "sweeps_per_wall_clock_hour": (float(3600.0 / seconds_per_sweep)
                                            if seconds_per_sweep > 0.0 else None),
        })
    seconds = [row["seconds"] for row in per_chain]
    iterations = [row["sweeps"] for row in per_chain]
    wall_clock_proxy_seconds = float(max(seconds))
    arm_sweeps = int(min(iterations))
    return {
        "per_chain": per_chain,
        "total_chain_seconds": float(sum(seconds)),
        "wall_clock_proxy_seconds": wall_clock_proxy_seconds,
        "arm_sweeps_per_wall_clock_hour": (float(3600.0 * arm_sweeps /
                                                   wall_clock_proxy_seconds)
                                            if wall_clock_proxy_seconds > 0.0 else None),
        "terminal_iterations": iterations,
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _paper_table(per_arm: dict) -> str:
    """Small paper-facing table: only preregistered primary quantities."""
    rows = [
        "| Method | Invariant convergence | Exact unordered library | Closure F1 | Boundary Brier | Co-skill Brier | pi TV error | P off-diag RMSE | Held-out NLL/occ | Runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        result = per_arm[arm]
        h = result["draw_recovery"]
        paths = result["path_recovery"]
        heldout = result["heldout"]
        runtime = result["runtime"]
        terminal_checkpoint = int(result["terminal"]["terminal_checkpoint"])
        convergence = (f"PASS@{terminal_checkpoint // 1000}k"
                       if result["terminal"].get("reason") == "two consecutive PASS checkpoints"
                       else f"ceiling@{terminal_checkpoint // 1000}k")
        rows.append(
            "| {arm} | {convergence} | {exact} | {f1} | {boundary} | {coskill} | {pi} | {p} | {nll} | {runtime} s; {rate} sw/h |".format(
                arm=arm,
                convergence=convergence,
                exact=_fmt(h["unordered_library"]["exact_recovery_probability"]),
                f1=_fmt(h["closure"]["aggregate_closure_f1"]["mean"]),
                boundary=_fmt(paths["boundary"]["boundary_brier_score"]),
                coskill=_fmt(paths["co_skill"]["co_skill_brier_score"]),
                pi=_fmt(h["pi"]["total_variation_error"]["mean"]),
                p=_fmt(h["transition"]["off_diagonal_rmse"]["mean"]),
                nll=_fmt(heldout["heldout_nll_per_occurrence"]),
                runtime=_fmt(runtime["wall_clock_proxy_seconds"], 1),
                rate=_fmt(runtime["arm_sweeps_per_wall_clock_hour"], 0),
            )
        )
    return "\n".join([
        "# FULL-LATENT terminal recovery",
        "",
        "All truth-dependent values below were computed only after both formal arms had terminal truth-free records.",
        "",
        *rows,
        "",
        "`Closure F1`, `pi TV error`, and `P off-diagonal RMSE` are posterior means of draw-level values after one shared closure-Hamming Hungarian alignment per draw. Boundary and co-skill values use pooled exact online accumulators. Held-out NLL uses all retained draws with `log Z - log C_J` before per-trace log-mean-exp.",
        "",
    ])


def run_terminal_recovery(*, out: Path = OUT, corpus_dir: Path = CORPUS_DIR,
                          workers: int = 4, overwrite: bool = False) -> dict:
    """Perform terminal recovery, preserving the strict terminal-before-truth order."""
    if int(workers) < 1 or int(workers) > N_CHAINS:
        raise ValueError(f"workers must be between 1 and {N_CHAINS}")

    # This is intentionally the first operation that can reach artifacts.  Do not move
    # any corpus/truth read or HPOP import above it.
    certificate = assert_both_arms_terminal(Path(out))
    # This durable event is deliberately written *before* the frozen-launch check and
    # before any import that could reach model/corpus code.  If the process dies later,
    # it remains as an auditable indication that terminal recovery was explicitly begun.
    unseal_event = record_terminal_recovery_unseal_started(Path(out), certificate)
    launch_integrity = assert_frozen_launch_integrity(Path(out))
    output_path = Path(out) / "terminal_recovery.json"
    table_path = Path(out) / "terminal_recovery_table.md"
    if (output_path.exists() or table_path.exists()) and not overwrite:
        raise RuntimeError("terminal recovery output already exists; pass --overwrite only "
                           "to recompute the same terminal artifacts")

    loaded = _load_exact_formal_chains_after_terminal(
        Path(out), Path(corpus_dir), certificate, launch_integrity["launch"])
    expected_truth_hash = loaded["launch"].get("corpus", {}).get("truth_hash")
    if not expected_truth_hash:
        raise RuntimeError("frozen launch manifest has no truth hash")
    # The only call that opens truth or hidden NPZ fields comes after terminal and exact
    # checkpoint certification have both succeeded.
    _, recovery = _post_terminal_imports()
    truth = _load_truth_after_terminal(Path(corpus_dir), loaded["corpus"].train,
                                        expected_truth_hash, loaded["fixed"], recovery)

    per_arm = {}
    for arm in ARMS:
        records = loaded["records"][arm]
        draw_recovery = _draw_level_recovery(records, truth, recovery)
        path_recovery = _online_path_recovery(records, truth, recovery)
        heldout = _heldout_recovery(records, loaded["corpus"], loaded["fixed"], recovery, workers)
        per_arm[arm] = {
            "terminal": certificate["arms"][arm],
            "checkpoint_artifacts": [{"chain": record["index"], "path": str(record["path"]),
                                        "retained_draws": int(record["chain"].retained_draws)}
                                       for record in records],
            "draw_recovery": draw_recovery,
            "path_recovery": path_recovery,
            "heldout": heldout,
            "runtime": _runtime_summary(records),
        }

    result = {
        "condition": "FULL-LATENT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "truth_unsealed": True,
        "terminal_certificate": certificate,
        "unseal_started_event": unseal_event,
        "frozen_launch_integrity": {key: value for key, value in launch_integrity.items()
                                     if key not in {"launch", "paired_starts", "chain_seeds"}},
        "truth": {"manifest_path": truth["manifest_path"],
                  "truth_hash_sha256": truth["truth_hash_sha256"]},
        "recovery_protocol": {
            "alignment": "per retained draw closure-Hamming Hungarian; deterministic lexicographic ties; shared mapping for H, pi, both P axes",
            "segmentation": "pooled exact online boundary accumulators, threshold=0.5",
            "co_skill": "pooled exact online co-skill accumulators, threshold=0.5",
            "heldout": "all retained U/pi/P draws; per draw log Z - log C_J; per trace log-mean-exp",
        },
        "per_arm": per_arm,
    }
    _write_json(output_path, result)
    table_path.write_text(_paper_table(per_arm))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-terminal-recovery", action="store_true",
                        help="explicitly unseal and compute recovery after both arms are terminal")
    parser.add_argument("--workers", type=int, default=4,
                        help="post-terminal held-out workers (1..4; default 4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing terminal recovery report")
    args = parser.parse_args(argv)
    if not args.run_terminal_recovery:
        parser.error("--run-terminal-recovery is required; recovery truth is terminal-only")
    result = run_terminal_recovery(workers=args.workers, overwrite=args.overwrite)
    print(json.dumps({
        "condition": result["condition"], "truth_unsealed": result["truth_unsealed"],
        "output": str(OUT / "terminal_recovery.json"),
        "table": str(OUT / "terminal_recovery_table.md"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
