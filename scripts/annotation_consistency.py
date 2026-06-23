"""Measure CPA annotation consistency — the make-or-break upstream check.

Computes, from `opencode` output(s):
  1. vocabulary saturation (distinct CPAs vs #trajectories; new-CPA rate, early vs late half)
  2. recurrence vs singletons (% occurrences whose label appears in >=2 trajectories)
  3. inter-config agreement (if >=2 inputs: per-trajectory label-set Jaccard, by instance_id)
  4. synonymy proxy (distinct labels vs after-normalization)
  5. APPLY held-out new-rate (if an APPLY-mode output is given: % decision==PROPOSE_NEW)

Label comparison uses normalized strings (a strict lower bound on agreement; true semantic
agreement ideally uses embeddings / LLM adjudication — noted in the report).

Usage:
    PYTHONPATH=src .venv/bin/python scripts/annotation_consistency.py \
        --inputs data/annotated/swe_rebench/cpa_A data/annotated/swe_rebench/cpa_B \
        [--apply data/annotated/swe_rebench/cpa_apply_S2] [--plot docs/annotation_consistency.png]
"""
import argparse, json, os, re
from collections import Counter, defaultdict


def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def load(prefix):
    """prefix.jsonl -> {instance_id: [normalized labels in order]} and flat occurrence list."""
    by = {}
    with open(prefix + ".jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            iid = o.get("_instance_id") or o.get("trajectory_id")
            labs = []
            for c in o.get("cpa_instances", []):
                if c.get("decision") == "ABSTAIN":
                    continue
                lab = c.get("canonical_label") or c.get("candidate_label")
                if lab:
                    labs.append(norm(lab))
            by[iid] = labs
    return by


def saturation(by, shuffles=20):
    """Average #distinct-labels curve over random trajectory orders; report early/late new-rate."""
    import random  # noqa: used only for ordering robustness
    iids = list(by)
    curve = [0.0] * len(iids)
    rng = [i for i in range(len(iids))]
    # deterministic pseudo-shuffles (no Math.random dependency): rotate
    for s in range(shuffles):
        order = iids[s % len(iids):] + iids[:s % len(iids)] if iids else []
        seen = set();
        for k, iid in enumerate(order):
            seen.update(by[iid]); curve[k] += len(seen)
    curve = [c / shuffles for c in curve] if shuffles else curve
    n = len(curve)
    if n >= 4:
        early = (curve[n // 2 - 1] - curve[0]) / max(1, n // 2 - 1)     # new CPAs/traj, first half
        late = (curve[-1] - curve[n // 2 - 1]) / max(1, n - n // 2)      # second half
    else:
        early = late = float("nan")
    return curve, early, late


def main(argv=None):
    ap = argparse.ArgumentParser(description="CPA annotation-consistency metrics.")
    ap.add_argument("--inputs", nargs="+", required=True, help="opencode prefixes (1 or more configs)")
    ap.add_argument("--apply", default=None, help="prefix of an APPLY-mode output (held-out new-rate)")
    ap.add_argument("--plot", default=None, help="optional PNG path for the saturation curve")
    args = ap.parse_args(argv)

    configs = [load(p) for p in args.inputs]
    A = configs[0]
    print("=" * 64)
    print("ANNOTATION CONSISTENCY  ({} config(s), {} trajectories)".format(len(configs), len(A)))
    print("=" * 64)

    # ---- 1. saturation ----
    curve, early, late = saturation(A)
    occ = [l for labs in A.values() for l in labs]
    print("\n[1] vocabulary saturation")
    print("    distinct CPAs: {} over {} occurrences ({} traj)".format(len(set(occ)), len(occ), len(A)))
    print("    new-CPA / trajectory:  first half = {:.2f}   second half = {:.2f}".format(early, late))
    print("    verdict: {}".format("SATURATING (good)" if late < early * 0.5 else
          "still growing — possible label drift" if late == late else "n/a (need more traj)"))

    # ---- 2. recurrence ----
    traj_freq = Counter()
    for labs in A.values():
        for l in set(labs):
            traj_freq[l] += 1
    nonsingleton_occ = sum(1 for l in occ if traj_freq[l] >= 2)
    frac = nonsingleton_occ / len(occ) if occ else 0
    print("\n[2] recurrence vs singletons")
    print("    singleton labels (1 traj): {} / {}".format(sum(1 for l, n in traj_freq.items() if n == 1), len(traj_freq)))
    print("    occurrences using a recurring (>=2 traj) label: {:.0%}".format(frac))
    print("    verdict: {}".format("OK (>=70%)" if frac >= 0.7 else "LOW reuse — weak signal for skills"))

    # ---- 3. inter-config agreement ----
    if len(configs) >= 2:
        B = configs[1]
        shared = [i for i in A if i in B]
        jac = []
        for i in shared:
            sa, sb = set(A[i]), set(B[i])
            jac.append(len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0)
        vocab_overlap = (len(set(occ) & {l for labs in B.values() for l in labs}) /
                         len(set(occ) | {l for labs in B.values() for l in labs})) if occ else 0
        print("\n[3] inter-config agreement ({} shared trajectories)".format(len(shared)))
        print("    mean per-trajectory label-set Jaccard: {:.2f}".format(sum(jac) / len(jac) if jac else 0))
        print("    vocabulary overlap (Jaccard of the two label sets): {:.2f}".format(vocab_overlap))
        print("    verdict: {} (target >=0.6; normalized-string lower bound)".format(
            "OK" if (jac and sum(jac) / len(jac) >= 0.6) else "LOW — labels diverge across configs"))
    else:
        print("\n[3] inter-config agreement: need >=2 configs (pass two --inputs).")

    # ---- 4. synonymy proxy ----
    print("\n[4] synonymy / drift")
    print("    distinct labels: {} (already normalized; embedding/LLM merge would reduce further)".format(len(set(occ))))

    # ---- 5. APPLY held-out new-rate ----
    if args.apply and os.path.exists(args.apply + ".jsonl"):
        ap_by = {}
        with open(args.apply + ".jsonl") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    for c in o.get("cpa_instances", []):
                        ap_by[c.get("occurrence_id")] = c.get("decision")
        decs = list(ap_by.values())
        newr = sum(1 for d in decs if d == "PROPOSE_NEW") / len(decs) if decs else 0
        print("\n[5] APPLY held-out new-rate")
        print("    PROPOSE_NEW on held-out: {:.0%}  ({} occurrences)".format(newr, len(decs)))
        print("    verdict: {}".format("OK (<20%) — library generalizes" if newr < 0.2 else "HIGH — library incomplete/inconsistent"))
    else:
        print("\n[5] APPLY held-out new-rate: pass --apply <prefix> of an APPLY-mode run on held-out traces.")

    if args.plot and len(curve) > 1:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 3.4)); plt.plot(range(1, len(curve) + 1), curve, marker='.', color='#7c9cff')
            plt.xlabel('trajectories annotated'); plt.ylabel('distinct CPAs'); plt.title('CPA vocabulary saturation')
            plt.tight_layout(); plt.savefig(args.plot, dpi=120); print("\nsaturation curve -> " + args.plot)
        except Exception as e:
            print("plot skipped:", e)


if __name__ == "__main__":
    main()
