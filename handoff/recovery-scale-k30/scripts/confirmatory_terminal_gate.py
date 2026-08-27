"""The single terminal gate for the optimized FULL-LATENT confirmatory experiment.

    PYTHONPATH=src python scripts/confirmatory_terminal_gate.py

Implements PREREG_CONFIRMATORY.md exactly and only. Computed on PRODUCTION draws alone
(the chains' retained draws are production by construction: burn_in = 50,000 warm-up).
TRUTH-FREE: this script never opens truth_SEALED.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.stage6b_mcmc_diagnostics import (                     # noqa: E402
    bulk_ess, mcse_mean, rank_normalized_split_rhat, tail_ess)

RUN = ROOT / "results" / "mcmc_optimized" / "confirmatory_run"
OUT = RUN / "terminal_gate"

RHAT_GATE = 1.01
LOG_TARGET_BULK, LOG_TARGET_TAIL = 1000.0, 500.0
OTHER_BULK, OTHER_TAIL = 400.0, 400.0
LEGACY_BULK, LEGACY_TAIL = 1000.0, 500.0

CLASSES = {
    "log_target": ["log_target"],
    "segmentation": ["total_segments", "mean_segments_per_trace",
                     "mean_segment_length", "sd_segment_length"],
    "boundary": ["boundary_probes"],
    "coskill": ["coskill_probes", "same_segment_probes"],
    "pi": ["sorted_pi", "pi_entropy", "pi_l2"],
    "P": ["P_frobenius", "P_trace2", "P_trace3", "sorted_P_row_entropy",
          "sorted_stationary"],
}
SECONDARY = ["total_relations", "sorted_relation_counts"]   # descriptive only


def diagnose(series_by_chain) -> dict:
    """Registered diagnostics with the preregistered degeneracy branches."""
    chains = np.asarray(series_by_chain, dtype=float)
    constant = [bool(np.all(c == c[0])) for c in chains]
    if all(constant):
        if len({float(c[0]) for c in chains}) == 1:
            return {"branch": "constant_and_equal", "rhat": None,
                    "bulk_ess": None, "tail_ess": None, "mcse": 0.0,
                    "sd": 0.0, "mean": float(chains[0, 0])}
        return {"branch": "constant_within_unequal_across", "rhat": float("inf"),
                "bulk_ess": 0.0, "tail_ess": 0.0, "mcse": float("nan"),
                "sd": float(np.std(chains)), "mean": float(np.mean(chains))}
    b = float(bulk_ess(chains))
    return {"branch": "non_degenerate",
            "rhat": float(rank_normalized_split_rhat(chains)["rhat"]),
            "bulk_ess": b, "tail_ess": float(tail_ess(chains)),
            "mcse": float(mcse_mean(chains)) if b > 0 else float("nan"),
            "sd": float(np.std(chains)), "mean": float(np.mean(chains))}


def gate_for(name: str, d: dict) -> tuple[bool, str]:
    if d["branch"] == "constant_and_equal":
        return True, "degenerate agreement (no ESS floor)"
    if d["branch"] == "constant_within_unequal_across":
        return False, "constant within chains, unequal across -> automatic FAIL"
    bulk = LOG_TARGET_BULK if name == "log_target" else OTHER_BULK
    tail = LOG_TARGET_TAIL if name == "log_target" else OTHER_TAIL
    reasons = []
    if not d["rhat"] <= RHAT_GATE:
        reasons.append(f"rhat {d['rhat']:.4f} > {RHAT_GATE}")
    if not d["bulk_ess"] >= bulk:
        reasons.append(f"bulk ESS {d['bulk_ess']:.1f} < {bulk:.0f}")
    if not d["tail_ess"] >= tail:
        reasons.append(f"tail ESS {d['tail_ess']:.1f} < {tail:.0f}")
    return (not reasons), ("; ".join(reasons) if reasons else "pass")


def library_ids(relation_indicators: np.ndarray, n_skills: int) -> tuple:
    """Exact canonical closure library per draw, as an integer id.

    `n_skills` is read from the chain rather than assumed. It used to be the literal 3 of
    the confirmatory experiment, which reshaped silently and wrongly at any other K -- the
    reshape succeeds whenever the width happens to divide, so the failure would have been
    a plausible-looking wrong answer rather than an error. The divisibility check below
    turns that into a loud one.

    The identifier is a multiset hash: the per-skill closure bytes are **sorted** before
    hashing, so it is invariant to relabelling the skills by construction. That is what
    makes it usable at K = 30, where enumerating the 30! relabellings is hopeless.
    """
    relation_indicators = np.asarray(relation_indicators)
    n, width = relation_indicators.shape
    n_skills = int(n_skills)
    if n_skills < 1:
        raise ValueError(f"n_skills must be positive, got {n_skills}")
    if width % n_skills:
        raise ValueError(
            f"relation-indicator width {width} is not divisible by n_skills "
            f"{n_skills}; the chain and the gate disagree about the model")
    per_skill = width // n_skills
    blocks = relation_indicators.reshape(n, n_skills, per_skill)
    ids = np.empty(n, dtype=np.int64)
    table: dict = {}
    for i in range(n):
        key = b"".join(sorted(np.packbits(blocks[i, k]).tobytes()
                              for k in range(n_skills)))
        ids[i] = table.setdefault(key, len(table))
    return ids, {v: hashlib.sha256(k).hexdigest()[:16] for k, v in table.items()}


def skills_in(chain) -> int:
    """K, from the chain's own `u_draws` of shape (draws, K, m, d)."""
    shape = np.asarray(chain["u_draws"]).shape
    if len(shape) != 4:
        raise ValueError(f"u_draws must be (draws, K, m, d), got {shape}")
    return int(shape[1])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pre = {p.stem: json.loads(p.read_text())
           for p in sorted((RUN / "preconditions").glob("*.json"))}
    report = {"gate": "PREREG_CONFIRMATORY.md single terminal gate",
              "computed_on": "production draws only", "arms": {}}

    for arm, tag in (("FULL-COND", "full_cond"), ("FULL-MARG", "full_marg")):
        data = [np.load(RUN / "chains" / f"{tag}_{i}.npz", allow_pickle=False)
                for i in range(4)]
        rows, failures, degenerate, legacy_short = {}, [], [], []

        for klass, names in CLASSES.items():
            for name in names:
                arrs = [d[f"summary__{name}"] for d in data]
                shape = arrs[0].shape[1:]
                ncomp = int(np.prod(shape)) if shape else 1
                for c in range(ncomp):
                    label = name if not shape else f"{name}[{c}]"
                    series = [a[:, c] if shape else a for a in arrs]
                    dg = diagnose(series)
                    ok, why = gate_for(name, dg)
                    legacy_ok = (dg["branch"] == "constant_and_equal" or
                                 (dg["branch"] == "non_degenerate"
                                  and dg["bulk_ess"] >= LEGACY_BULK
                                  and dg["tail_ess"] >= LEGACY_TAIL
                                  and dg["rhat"] <= RHAT_GATE))
                    rows[label] = {"class": klass, **dg, "pass": ok, "reason": why,
                                   "legacy_1000_500_pass": bool(legacy_ok)}
                    if not ok:
                        failures.append((label, why))
                    if dg["branch"] == "constant_and_equal":
                        degenerate.append(label)
                    if ok and not legacy_ok:
                        legacy_short.append(label)

        # ---- the exact canonical closure library ----
        ids_per_chain, id_maps = [], []
        n_skills = skills_in(data[0])
        if any(skills_in(d) != n_skills for d in data):
            raise ValueError("the four chains disagree about the number of skills")
        for d in data:
            ids, m = library_ids(np.asarray(d["relation_indicators"]), n_skills)
            ids_per_chain.append(ids)
            id_maps.append(m)
        per_chain_sets = [set(np.unique(i).tolist()) for i in ids_per_chain]
        hashes = [{id_maps[c][v] for v in s} for c, s in enumerate(per_chain_sets)]
        all_constant = all(len(s) == 1 for s in per_chain_sets)
        all_equal = all_constant and len(set.union(*hashes)) == 1

        chains_pre = [pre[f"{tag}_{i}"] for i in range(4)]
        starts_distinct = len({p["start_library"] for p in chains_pre})
        warm_ok = all(p["warmup_h_accepts"] >= 1 for p in chains_pre)

        if all_equal:
            branch = "a_constant_and_equal"
            conds = {"starts_dispersed": starts_distinct >= 2,
                     "every_chain_moved_H_in_warmup": warm_ok,
                     "library_constant_and_equal_in_production": True,
                     "all_other_diagnostics_pass": len(failures) == 0}
            lib_pass = all(conds.values())
            lib_reason = ("branch (a) with all preconditions satisfied" if lib_pass
                          else f"branch (a) but preconditions failed: "
                               f"{[k for k, v in conds.items() if not v]}")
            lib_diag = None
        elif all_constant:
            branch = "b_constant_within_unequal_across"
            conds, lib_pass = {}, False
            lib_reason = "constant within chains, unequal across -> automatic FAIL"
            lib_diag = None
        else:
            branch = "c_non_degenerate"
            conds = {}
            lib_diag = diagnose(ids_per_chain)
            lib_pass, lib_reason = gate_for("canonical_library", lib_diag)
        if not lib_pass:
            failures.append(("canonical_library", lib_reason))

        verdict = "PASS" if not failures else "FAIL"
        report["arms"][arm] = {
            "verdict": verdict,
            "n_summaries": len(rows), "n_failures": len(failures),
            "failures": failures[:40],
            "n_degenerate_constant_and_equal": len(degenerate),
            "degenerate_summaries": degenerate[:80],
            "pass_new_but_fail_legacy": legacy_short,
            "canonical_library": {
                "branch": branch, "pass": lib_pass, "reason": lib_reason,
                "preconditions": conds,
                "distinct_libraries_per_chain": [len(s) for s in per_chain_sets],
                "library_hashes_per_chain": [sorted(h) for h in hashes],
                "diagnostics": lib_diag},
            "summaries": rows,
        }

    (OUT / "terminal_gate.json").write_text(json.dumps(report, indent=2,
                                                       sort_keys=True, default=str) + "\n")

    for arm, r in report["arms"].items():
        print(f"\n{'='*78}\n{arm}: {r['verdict']}")
        print(f"  registered summaries : {r['n_summaries']}")
        print(f"  failures             : {r['n_failures']}")
        print(f"  degenerate (const&eq): {r['n_degenerate_constant_and_equal']}")
        print(f"  pass-new/fail-legacy : {len(r['pass_new_but_fail_legacy'])}")
        lib = r["canonical_library"]
        print(f"  canonical library    : {lib['branch']}  -> "
              f"{'PASS' if lib['pass'] else 'FAIL'}  ({lib['reason']})")
        print(f"    distinct libs/chain: {lib['distinct_libraries_per_chain']}")
        if lib["diagnostics"]:
            d = lib["diagnostics"]
            print(f"    rhat {d['rhat']:.4f}  bulk {d['bulk_ess']:.1f}  "
                  f"tail {d['tail_ess']:.1f}")
        if r["failures"]:
            print("  first failures:")
            for label, why in r["failures"][:8]:
                print(f"    {label:<28} {why}")
    print(f"\nwrote {OUT/'terminal_gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
