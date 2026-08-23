"""Preflight for the K-recovery scalability package.

    python verify_environment.py            # everything
    python verify_environment.py --integrity  # source integrity only

Three checks, in the order they should be trusted:

1. **Source integrity.** Every shipped file against `SOURCE_INTEGRITY.json`. This is the
   substitute for Section 1's git-ancestry check, which a package shipped without history
   cannot satisfy. See `HANDOFF_NOTES.md`, Blocker 1.
2. **Environment.** Python, NumPy, SciPy and BLAS, against the versions the parity gate was
   verified under. A mismatch is a warning, not a failure -- but the parity gate below is
   then the thing that decides.
3. **Optimized/reference parity.** The registered gate: alpha and log Z within 1e-10,
   identical -inf pattern, identical legal support, valid backward draws, no NaN, valid
   pi/P. This is a hard gate. Nothing should be launched if it fails.

Exit status is 0 only if integrity and parity both pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

EXPECTED = {"numpy": "2.4.6", "scipy": "1.18.0", "python": "3.13.2"}
TOLERANCE = 1e-10


def check_integrity() -> bool:
    manifest_path = ROOT / "SOURCE_INTEGRITY.json"
    if not manifest_path.exists():
        print("FAIL  SOURCE_INTEGRITY.json is missing")
        return False
    manifest = json.loads(manifest_path.read_text())
    bad, missing = [], []
    for relative, digest in manifest["files_sha256"].items():
        path = ROOT / relative
        if not path.exists():
            missing.append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(relative)
    ok = not bad and not missing
    print(f"{'PASS' if ok else 'FAIL'}  source integrity: "
          f"{len(manifest['files_sha256'])} files, {len(bad)} modified, "
          f"{len(missing)} missing")
    for relative in (bad + missing)[:10]:
        print(f"        {relative}")
    print(f"      sealed engines byte-identical to "
          f"{manifest['validated_backend_commit'][:12]}: "
          f"{manifest['sealed_engines_match_validated_commit']}")
    return ok and manifest["sealed_engines_match_validated_commit"]


def check_environment() -> None:
    import numpy
    import scipy
    print(f"      python {platform.python_version()} (expected {EXPECTED['python']})")
    print(f"      numpy  {numpy.__version__} (expected {EXPECTED['numpy']})")
    print(f"      scipy  {scipy.__version__} (expected {EXPECTED['scipy']})")
    try:
        config = numpy.show_config(mode="dicts")
        blas = config.get("Build Dependencies", {}).get("blas", {})
        print(f"      blas   {blas.get('name', 'unknown')} {blas.get('version', '')}")
    except Exception:
        print("      blas   could not be determined")
    print(f"      threads pinned: "
          f"{ {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')} }")
    for name, expected in (("numpy", EXPECTED["numpy"]), ("scipy", EXPECTED["scipy"])):
        actual = sys.modules[name].__version__ if name in sys.modules else None
        if actual and actual != expected:
            print(f"WARN  {name} {actual} != pinned {expected}; the parity gate decides")


def check_parity() -> bool:
    """The registered optimized/reference gate on a small deterministic problem."""
    import numpy as np
    from hpop.mcmc_original.recurrent_joint_ffbs_mcmc import FFBSBlockTables
    from hpop.mcmc_original.semi_markov_ffbs import backward_sample
    from hpop.mcmc_original.semi_markov_ffbs import forward as reference_forward
    from hpop.mcmc_original.stage6e_frozen import (DELTA_B, MAX_BLOCK_WIDTH,
                                                   MIN_BLOCK_WIDTH, N_ROLES, N_SKILLS)
    from hpop.mcmc_original.stage6e_state import Stage6EModel, Stage6EState
    from hpop.mcmc_original.transitions import (log_transition_matrix,
                                                sample_transition_matrix)
    from hpop.mcmc_original.types import Segment, Segmentation
    from hpop.mcmc_optimized import FLAGS
    from hpop.mcmc_optimized.forward import forward_batched_group

    FLAGS.reset()
    rng = np.random.default_rng(20260823)
    traces = tuple(tuple(int(v) for v in rng.integers(0, N_ROLES, size=48))
                   for _ in range(6))
    model = Stage6EModel(traces=traces, epsilon=0.02, delta_b=DELTA_B,
                         n_skills=N_SKILLS, n_roles=N_ROLES,
                         min_width=MIN_BLOCK_WIDTH, max_width=MAX_BLOCK_WIDTH,
                         infer_pi_P=True, eta_initial=1.0, eta_transition=1.0)
    pi = rng.dirichlet(np.ones(N_SKILLS))
    p = sample_transition_matrix(np.zeros((N_SKILLS, N_SKILLS)), N_SKILLS, rng, 1.0)

    def tiling(length, index):
        segments, start, position = [], 0, 0
        while length - start > MAX_BLOCK_WIDTH:
            segments.append(Segment(start, start + MAX_BLOCK_WIDTH,
                                    (index + position) % N_SKILLS))
            start += MAX_BLOCK_WIDTH
            position += 1
        segments.append(Segment(start, length, (index + position) % N_SKILLS))
        return Segmentation(tuple(segments))

    state = Stage6EState(
        segmentations=tuple(tiling(len(t), n) for n, t in enumerate(traces)),
        u_by_skill=rng.standard_normal((N_SKILLS, N_ROLES, 2)), rho=0.5, beta=1.0,
        omega=0.0, lambda_rep=1.0, lambda_back=1.0, pi=pi, transition=p)

    tables = FFBSBlockTables(model=model, source="batched")
    tables.refresh(state)
    built = [np.array(t, copy=True) for t in tables.tables_for(state)]
    log_pi, log_p = np.log(state.pi), log_transition_matrix(state.transition)

    reference = [reference_forward(t, log_pi, log_p, model.delta_b, model.max_width,
                                   model.min_width) for t in built]
    optimized = forward_batched_group(built, log_pi, log_p, model.delta_b,
                                      model.max_width, model.min_width)

    worst_alpha = worst_z = 0.0
    pattern = nan_free = True
    for a, b in zip(reference, optimized):
        ref_alpha, opt_alpha = np.asarray(a.alpha), np.asarray(b.alpha)
        finite = np.isfinite(ref_alpha)
        pattern &= bool(np.array_equal(finite, np.isfinite(opt_alpha)))
        nan_free &= not (np.isnan(ref_alpha).any() or np.isnan(opt_alpha).any())
        if finite.any():
            worst_alpha = max(worst_alpha,
                              float(np.abs(ref_alpha[finite] - opt_alpha[finite]).max()))
        worst_z = max(worst_z, abs(float(a.log_normalizer) - float(b.log_normalizer)))

    draw_rng = np.random.default_rng(7)
    draws_ok = True
    for trace, chart in zip(traces, optimized):
        cursor, previous = 0, None
        for start, end, skill in backward_sample(chart, draw_rng):
            if start != cursor or not MIN_BLOCK_WIDTH <= end - start <= MAX_BLOCK_WIDTH:
                draws_ok = False
            if previous is not None and int(skill) == previous:
                draws_ok = False
            cursor, previous = end, int(skill)
        draws_ok &= cursor == len(trace)

    pi_ok = abs(float(state.pi.sum()) - 1.0) <= 1e-12
    p_ok = bool(np.array_equal(np.diag(state.transition), np.zeros(N_SKILLS)))

    ok = (worst_alpha <= TOLERANCE and worst_z <= TOLERANCE and pattern and nan_free
          and draws_ok and pi_ok and p_ok)
    print(f"{'PASS' if ok else 'FAIL'}  optimized/reference parity")
    print(f"      max |alpha_opt - alpha_ref| = {worst_alpha:.3e}  (tolerance {TOLERANCE:.0e})")
    print(f"      max |logZ_opt  - logZ_ref | = {worst_z:.3e}")
    print(f"      -inf pattern identical: {pattern}   no NaN: {nan_free}")
    print(f"      backward draws valid: {draws_ok}   pi valid: {pi_ok}   P valid: {p_ok}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integrity", action="store_true",
                        help="source integrity only; skip environment and parity")
    args = parser.parse_args()

    print("=== source integrity ===")
    integrity = check_integrity()
    if args.integrity:
        return 0 if integrity else 1
    print("\n=== environment ===")
    check_environment()
    print("\n=== parity gate ===")
    parity = check_parity()
    print(f"\nRESULT: {'READY' if integrity and parity else 'NOT READY'}")
    if not integrity:
        print("  source integrity failed -- do not run anything")
    if not parity:
        print("  parity failed -- Section 0 says stop immediately")
    return 0 if (integrity and parity) else 1


if __name__ == "__main__":
    raise SystemExit(main())
