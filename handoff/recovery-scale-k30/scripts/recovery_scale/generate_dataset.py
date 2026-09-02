#!/usr/bin/env python3
"""Generate and FREEZE the synthetic dataset for the recovery-at-scale experiment.

Materialises, for each production replicate and each rung: the sealed truth (library
prefix + transitions), the training and held-out traces, the evidence profile (IP-Cov,
edge witnessing, all-pairs resolved, nLE per skill), and SHA-256 hashes of every file.
Machines then run against these files; nothing is regenerated on the fleet, so a corpus
mismatch is impossible rather than unlikely.

The TRUTH FILES ARE SEALED: written under truth/ with hashes recorded in the manifest,
to be opened only after the convergence verdicts are frozen. Workers never read truth/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_cpa.corpus import generate_ladder_corpus                    # noqa: E402
from hpop.mcmc_cpa.nested_library import draw_master_library_v2            # noqa: E402
from hpop.mcmc_cpa.recovery_regime import REGIME, generation_params, regime_dict  # noqa: E402
from hpop.mcmc_original.latent_poset import precedence_from_u              # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_profile(corpus, u, maps, K: int) -> dict:
    sys.path.insert(0, "/Users/dongqing/Desktop/hpop/src/hpop/vendored/po_inference_agent")
    try:
        from src.utils.cloud_iac_coverage import (
            CloudIacCriticalPairCoverageAnalyzer as CP)
        from src.utils.po_fun import BasicUtils
        have_bpop = True
    except Exception:
        have_bpop = False
    inv = maps.inverse
    pool = [[] for _ in range(K)]
    for r in corpus.train:
        pos = 0
        for w, s in zip(r.widths, r.labels):
            order, seen = [], set()
            for cp in r.cpa[pos:pos+w]:
                x = int(inv[s, cp])
                if x >= 0 and x not in seen:
                    seen.add(x); order.append(x)
            pool[s].append(order); pos += w
    skills = []
    for k in range(K):
        closure = np.asarray(precedence_from_u(u[k])).astype(int)
        edges = np.argwhere(closure)
        cooc = np.array([sum(1 for o in pool[k] if a in o and b in o)
                         for a, b in edges]) if len(edges) else np.zeros(0, int)
        row = {"skill": k, "instances": len(pool[k]), "edges": int(len(edges)),
               "edge_cooc_min": int(cooc.min()) if cooc.size else None,
               "edge_cooc_median": float(np.median(cooc)) if cooc.size else None,
               "edges_below_min": int((cooc < REGIME.EDGE_COOC_MIN).sum())}
        if have_bpop:
            _, total, cov = CP.critical_pair_coverage(pool[k], closure)
            row["incomparable_pairs"] = int(total)
            row["ip_cov"] = float(cov)
            try:
                row["n_linear_extensions"] = int(
                    BasicUtils.nle(BasicUtils.transitive_reduction(closure)))
            except Exception:
                row["n_linear_extensions"] = None
        ok = bad = 0
        for i in range(10):
            for j in range(i + 1, 10):
                if closure[i, j] or closure[j, i]:
                    good = sum(1 for o in pool[k] if i in o and j in o) \
                        >= REGIME.EDGE_COOC_MIN
                else:
                    mask = 0
                    for o in pool[k]:
                        if i in o and j in o:
                            mask |= 1 if o.index(i) < o.index(j) else 2
                            if mask == 3:
                                break
                    good = mask == 3
                ok += good; bad += (not good)
        row["all_pairs_resolved"] = ok / (ok + bad)
        skills.append(row)
    return {"skills": skills,
            "ip_cov_median": float(np.median([s["ip_cov"] for s in skills]))
            if have_bpop else None,
            "resolved_median": float(np.median([s["all_pairs_resolved"]
                                                for s in skills])),
            "resolved_min": float(min(s["all_pairs_resolved"] for s in skills))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "dataset" / "recovery_scale_v1")
    args = p.parse_args()

    out = args.out
    (out / "truth").mkdir(parents=True, exist_ok=True)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "recovery-scale-dataset/1.0.0",
                "regime": regime_dict(), "replicates": {}, "files_sha256": {}}

    for replicate in REGIME.REPLICATES:
        library, meta = draw_master_library_v2(replicate)
        rep_record = {"library_draw": {k: v for k, v in meta.items()
                                       if k != "attempts"},
                      "library_rejections_recorded": len(meta["attempts"]),
                      "rungs": {}}
        for K in REGIME.K_LADDER:
            u, maps = library.prefix(K)
            corpus = generate_ladder_corpus(
                library, K, replicate, trace_length=REGIME.TRACE_LENGTH,
                params=generation_params(), delta_b=REGIME.DELTA_B,
                min_width=REGIME.MIN_WIDTH, max_width=REGIME.MAX_WIDTH,
                train_per_skill=REGIME.TRAIN_PER_SKILL,
                test_per_skill=REGIME.TEST_PER_SKILL)

            traces_path = out / "traces" / f"rep{replicate}_K{K}.json"
            traces_path.write_text(json.dumps({
                "replicate": replicate, "K": K,
                "train": [{"index": t.index, "cpa": list(t.cpa)}
                          for t in corpus.train],
                "heldout": [{"index": t.index, "cpa": list(t.cpa)}
                            for t in corpus.heldout],
                "coverage_reported_not_enforced": corpus.coverage}, default=str))

            maps_path = out / "traces" / f"rep{replicate}_K{K}_rolemaps.json"
            maps_path.write_text(json.dumps({
                "forward": maps.forward.tolist(), "n_cpa": int(maps.n_cpa),
                "note": "typed supports are part of the OBSERVATION model, not of the "
                        "sealed order truth; workers may read this"}))

            truth_path = out / "truth" / f"rep{replicate}_K{K}.json"
            truth_path.write_text(json.dumps({
                "SEALED": "open only after the convergence verdict is frozen",
                "replicate": replicate, "K": K,
                "u_by_skill": np.asarray(u, dtype=float).tolist(),
                "role_maps_forward": maps.forward.tolist(),
                "pi": corpus.pi.tolist(), "transition": corpus.transition.tolist(),
                "train_segmentations": [{"widths": list(t.widths),
                                         "labels": list(t.labels)}
                                        for t in corpus.train]}))

            profile = evidence_profile(corpus, u, maps, K)
            rep_record["rungs"][str(K)] = {
                "n_train": len(corpus.train), "n_heldout": len(corpus.heldout),
                "evidence": {k: v for k, v in profile.items() if k != "skills"},
                "evidence_per_skill": profile["skills"],
            }
            for path in (traces_path, truth_path, maps_path):
                manifest["files_sha256"][str(path.relative_to(out))] = sha(path)
            print(f"rep{replicate} K={K:>2}: {len(corpus.train)} train traces, "
                  f"IP-Cov med {profile['ip_cov_median']}, resolved "
                  f"med {profile['resolved_median']:.2f} min {profile['resolved_min']:.2f}",
                  flush=True)
        manifest["replicates"][str(replicate)] = rep_record

    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2,
                                                          default=str))
    print(f"\nwrote {out}/dataset_manifest.json "
          f"({len(manifest['files_sha256'])} hashed files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
