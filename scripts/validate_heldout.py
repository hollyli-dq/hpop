"""Step 2 — validate the CPA library (v2, 32 declared) on UNSEEN trajectories.

Measures the three APPLY-mode decision rates on a repo-disjoint held-out set:

    MATCH_EXISTING  an existing definition fits          -> library covers the action
    PROPOSE_NEW     no existing definition fits          -> library has a gap
    ABSTAIN         evidence insufficient to decide      -> annotator uncertainty, not a gap

The held-out rate alone is NOT interpretable. A 12% PROPOSE_NEW rate could mean "the library is
incomplete" or "this annotator likes proposing". So we also run an IN-SAMPLE arm over repos the
dictionary WAS built on and report the DELTA. The claim we can defend is comparative:

    PROPOSE_NEW does not rise materially on repos the dictionary never saw
    => the library generalizes beyond its induction sample.

If the held-out rate is high but the in-sample rate is equally high, that is an annotator/prompt
property, not a generalization failure — and the paper must say so.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/validate_heldout.py \
        --heldout data/annotated/swe_rebench/apply_heldout \
        --insample data/annotated/swe_rebench/apply_insample \
        --library rules/cpa_dictionary_v2.json \
        --pilot data/interim/swe_rebench/pilot500.jsonl \
        [--json docs/heldout_validation.json]
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

DECISIONS = ["MATCH_EXISTING", "PROPOSE_NEW", "ABSTAIN"]


def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def load_arm(prefix):
    """Read an opencode APPLY output. Returns (occurrences, trajectories, candidate_updates)."""
    occ, trajs, cands = [], [], []
    path = prefix + ".jsonl"
    if not os.path.exists(path):
        raise SystemExit("missing {} — run the APPLY pass first".format(path))
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            repo = o.get("_repo")
            trajs.append({"instance_id": o.get("_instance_id"), "repo": repo,
                          "resolved": o.get("_resolved"),
                          "n": len(o.get("cpa_instances", []))})
            for c in o.get("cpa_instances", []):
                occ.append({
                    "decision": c.get("decision"),
                    "label": c.get("canonical_label") or c.get("candidate_label"),
                    "conf": c.get("label_confidence"),
                    "repo": repo,
                    "instance_id": o.get("_instance_id"),
                })
            for u in o.get("candidate_library_updates", []):
                cands.append(dict(u, repo=repo, instance_id=o.get("_instance_id")))
    return occ, trajs, cands


def rates(occ):
    c = Counter(o["decision"] for o in occ)
    n = len(occ) or 1
    return {d: {"n": c.get(d, 0), "rate": c.get(d, 0) / n} for d in DECISIONS}, len(occ)


def wilson(k, n, z=1.96):
    """Wilson score interval — correct at the small counts these rates produce (ABSTAIN may be ~0)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - m), min(1.0, c + m))


def per_traj_rate(occ, decision):
    """Mean of per-trajectory rates + its spread. The occurrence-level rate is dominated by long
    trajectories; this says whether PROPOSE_NEW is spread across the corpus or concentrated in a few."""
    by = defaultdict(list)
    for o in occ:
        by[o["instance_id"]].append(o["decision"])
    per = [sum(1 for d in v if d == decision) / len(v) for v in by.values() if v]
    if not per:
        return 0.0, 0.0, 0
    mean = sum(per) / len(per)
    var = sum((x - mean) ** 2 for x in per) / len(per)
    n_any = sum(1 for x in per if x > 0)
    return mean, var ** 0.5, n_any


def main(argv=None):
    ap = argparse.ArgumentParser(description="Held-out CPA library validation (Step 2).")
    ap.add_argument("--heldout", required=True, help="opencode APPLY prefix, repo-disjoint arm")
    ap.add_argument("--insample", default=None, help="opencode APPLY prefix, in-sample control arm")
    ap.add_argument("--library", default="rules/cpa_dictionary_v2.json")
    ap.add_argument("--pilot", default="data/interim/swe_rebench/pilot500.jsonl",
                    help="induction corpus — used to ASSERT repo-disjointness")
    ap.add_argument("--json", default=None, help="write machine-readable results here")
    args = ap.parse_args(argv)

    library = json.load(open(args.library))
    lib_names = {c["name"] for c in library}

    ho_occ, ho_traj, ho_cand = load_arm(args.heldout)
    print("=" * 72)
    print("CPA LIBRARY VALIDATION ON UNSEEN TRAJECTORIES")
    print("=" * 72)
    print("library      : {} ({} CPAs declared)".format(args.library, len(library)))

    # ---- 0. assert repo-disjointness (the whole experiment rests on this) ----
    pilot_repos = set()
    if os.path.exists(args.pilot):
        with open(args.pilot, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pilot_repos.add(json.loads(line).get("repo"))
    ho_repos = {t["repo"] for t in ho_traj}
    leak = ho_repos & pilot_repos
    print("\n[0] repo-disjointness")
    print("    induction corpus : {} repos".format(len(pilot_repos)))
    print("    held-out         : {} trajectories over {} repos".format(len(ho_traj), len(ho_repos)))
    if leak:
        print("    !! LEAK: {} held-out repos are in the induction corpus: {}".format(
            len(leak), ", ".join(sorted(leak)[:5])))
        print("    !! The held-out claim is INVALID until these are removed.")
    else:
        print("    verdict          : CLEAN — zero repo overlap")

    # ---- 1. the three rates ----
    ho_rates, ho_n = rates(ho_occ)
    print("\n[1] decision rates — held-out ({} occurrences)".format(ho_n))
    print("    {:<16} {:>7} {:>8}   {}".format("decision", "n", "rate", "95% CI (Wilson)"))
    for d in DECISIONS:
        lo, hi = wilson(ho_rates[d]["n"], ho_n)
        print("    {:<16} {:>7} {:>7.1%}   [{:.1%}, {:.1%}]".format(
            d, ho_rates[d]["n"], ho_rates[d]["rate"], lo, hi))

    # ---- 2. the control contrast ----
    result = {"library": args.library, "n_declared": len(library),
              "heldout": {"n_occ": ho_n, "n_traj": len(ho_traj), "n_repos": len(ho_repos),
                          "repo_leak": sorted(leak), "rates": ho_rates}}
    if args.insample:
        is_occ, is_traj, is_cand = load_arm(args.insample)
        is_rates, is_n = rates(is_occ)
        print("\n[2] control contrast — in-sample ({} occurrences over {} traj)".format(
            is_n, len(is_traj)))
        print("    {:<16} {:>9} {:>9} {:>9}".format("decision", "held-out", "in-sample", "delta"))
        for d in DECISIONS:
            delta = ho_rates[d]["rate"] - is_rates[d]["rate"]
            print("    {:<16} {:>8.1%} {:>9.1%} {:>+9.1f}pp".format(
                d, ho_rates[d]["rate"], is_rates[d]["rate"], delta * 100))
        dn = ho_rates["PROPOSE_NEW"]["rate"] - is_rates["PROPOSE_NEW"]["rate"]
        lo_h, hi_h = wilson(ho_rates["PROPOSE_NEW"]["n"], ho_n)
        lo_i, hi_i = wilson(is_rates["PROPOSE_NEW"]["n"], is_n)
        overlap = not (hi_h < lo_i or hi_i < lo_h)
        print("\n    PROPOSE_NEW held-out [{:.1%}, {:.1%}] vs in-sample [{:.1%}, {:.1%}]".format(
            lo_h, hi_h, lo_i, hi_i))
        print("    verdict: {}".format(
            "GENERALIZES — CIs overlap; no detectable rise on unseen repos" if overlap else
            "DOES NOT GENERALIZE — PROPOSE_NEW rises {:+.1f}pp on unseen repos".format(dn * 100)))
        print("    NOTE: an equally high rate in BOTH arms is an annotator property, not a library gap.")
        result["insample"] = {"n_occ": is_n, "n_traj": len(is_traj), "rates": is_rates}
        result["delta_propose_new_pp"] = dn * 100
        result["generalizes"] = bool(overlap)
    else:
        print("\n[2] control contrast: SKIPPED (no --insample). The held-out rate has no baseline —")
        print("    a reviewer cannot tell a library gap from an annotator disposition.")

    # ---- 3. concentration ----
    m, sd, n_any = per_traj_rate(ho_occ, "PROPOSE_NEW")
    print("\n[3] is PROPOSE_NEW spread or concentrated? (held-out)")
    print("    per-trajectory rate: mean {:.1%} (sd {:.1%})".format(m, sd))
    print("    trajectories with >=1 PROPOSE_NEW: {}/{}".format(n_any, len(ho_traj)))
    print("    reading: concentrated in few trajectories => a niche gap or a repo artifact;")
    print("             spread across many            => a real, recurring missing CPA.")
    result["heldout"]["propose_new_per_traj_mean"] = m
    result["heldout"]["propose_new_traj_coverage"] = n_any / (len(ho_traj) or 1)

    # ---- 4. WHAT was proposed (the scientifically interesting part) ----
    print("\n[4] proposed labels — held-out (the actual candidate gaps)")
    prop = Counter(norm(o["label"]) for o in ho_occ if o["decision"] == "PROPOSE_NEW" and o["label"])
    prop_traj = defaultdict(set)
    for o in ho_occ:
        if o["decision"] == "PROPOSE_NEW" and o["label"]:
            prop_traj[norm(o["label"])].add(o["instance_id"])
    if not prop:
        print("    (none — the 32-CPA library covered every action on unseen repos)")
    for lab, n in prop.most_common(15):
        print("    {:<40} {:>4} occ / {:>3} traj".format(lab, n, len(prop_traj[lab])))
    print("\n    A label recurring across MANY trajectories is a promote candidate.")
    print("    A label in ONE trajectory is below the appearance threshold — not a category.")
    result["proposed_labels"] = [
        {"label": l, "occurrences": n, "trajectories": len(prop_traj[l])} for l, n in prop.most_common()
    ]

    # ---- 5. synonym check: are proposals real gaps or restatements? ----
    print("\n[5] are the proposals REAL gaps, or synonyms of existing CPAs?")
    lib_norm = {norm(n) for n in lib_names}
    synonyms = [l for l in prop if l in lib_norm]
    if synonyms:
        print("    !! {} proposed labels string-match an EXISTING library CPA: {}".format(
            len(synonyms), ", ".join(synonyms[:5])))
        print("    !! These are matcher failures, not library gaps — they inflate PROPOSE_NEW.")
    else:
        print("    no proposed label string-matches an existing CPA name (crude check only)")
    if ho_cand:
        print("\n    self-reported nearest existing CPA, per candidate_library_updates:")
        for u in ho_cand[:10]:
            print("      {:<32} near: {}".format(
                u.get("candidate_label", "?"), ", ".join(u.get("nearest_existing_labels") or []) or "-"))
        print("\n    If `nearest_existing_labels` is populated and plausible, the proposal is likely a")
        print("    SPLIT of an existing CPA, not a new one — adjudicate before promoting.")
    result["proposals_matching_existing_names"] = synonyms

    # ---- 6. coverage of the declared library ----
    used = {norm(o["label"]) for o in ho_occ if o["decision"] == "MATCH_EXISTING" and o["label"]}
    unused = sorted(n for n in lib_names if norm(n) not in used)
    print("\n[6] declared CPAs never matched on held-out: {}/{}".format(len(unused), len(lib_names)))
    if unused:
        print("    " + ", ".join(unused))
    print("    Expected to include the declared-unseen set (PLAN_APPROACH, APPLY_PATCH, REFACTOR_CODE).")
    result["unmatched_declared"] = unused

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        json.dump(result, open(args.json, "w"), indent=1)
        print("\nmachine-readable results -> " + args.json)


if __name__ == "__main__":
    main()
