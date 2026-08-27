"""Skill segmentation — weak-supervision skill annotation over CPA sequences (no API).

Segments each trajectory's CPA-occurrence sequence into contiguous **skill instances** and assigns a
skill type from rules/skill_library_seed.json, via a transparent phase state-machine (orient →
reproduce → fix → finalize) that disambiguates context-dependent CPAs (e.g. READ_SOURCE/LOCATE_CODE
belong to `explore_and_orient` before the first edit, `localize_and_fix` after). Maximal contiguous
runs of the same skill become one skill instance.

This is a PRIOR / weak supervision for HPOP (which learns the real skills + local/global posets), not
ground truth — analogous to the silver CPA layer. Produces the two-level modelling input:
  <out>.skills.jsonl  per trajectory: global_skill_sequence + skill_instances{type, cpa subsequence}

Usage:
    PYTHONPATH=src .venv/bin/python -m hpop.annotate.skill_segment \
        --sequences data/modelling/swe_rebench/pilot100.sequences.jsonl \
        --library rules/skill_library_seed.json \
        --out data/modelling/swe_rebench/pilot100
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

# CPAs that move the phase forward
_FIX_CPAS = {"EDIT_SOURCE", "ADD_DEBUG_INSTRUMENTATION", "REFACTOR_CODE", "REVERT_CHANGE"}
_REPRO_CPAS = {"REPRODUCE_ISSUE", "WRITE_REPRODUCTION_SCRIPT"}


def cpa_to_skill(cpa, phase):
    """Map a CPA to a skill given the current phase; return (skill_name, new_phase)."""
    if cpa == "INSTALL_DEPENDENCY":
        return "setup_environment", phase
    if cpa in ("SUBMIT_SOLUTION", "CLEANUP_ARTIFACTS"):
        return "verify_and_submit", "finalize"
    if cpa == "WRITE_TEST":
        return "add_regression_test", ("fix" if phase in ("explore", "reproduce") else phase)
    if cpa in _FIX_CPAS:
        return "localize_and_fix", "fix"
    if cpa in _REPRO_CPAS:
        return "reproduce_failure", ("reproduce" if phase in ("explore", "reproduce") else phase)
    if cpa == "DIAGNOSE_FAILURE":
        return ("reproduce_failure" if phase in ("explore", "reproduce") else "localize_and_fix"), phase
    if cpa == "VERIFY_FIX":
        if phase == "finalize":
            return "verify_and_submit", phase
        return ("localize_and_fix" if phase == "fix" else "reproduce_failure"), phase
    if cpa == "RUN_TEST_SUITE":
        if phase == "finalize":
            return "verify_and_submit", phase
        return ("reproduce_failure" if phase in ("explore", "reproduce") else "localize_and_fix"), phase
    if cpa == "EXPLORE_REPOSITORY":
        return "explore_and_orient", phase
    if cpa in ("LOCATE_CODE", "READ_SOURCE"):
        return ("explore_and_orient" if phase == "explore" else "localize_and_fix"), phase
    return "localize_and_fix", phase  # fallback


def _smooth(skills, passes=2):
    """Remove isolated single-position flips: A B A -> A A A (B not flanked by its own kind)."""
    s = list(skills)
    for _ in range(passes):
        for k in range(1, len(s) - 1):
            if s[k] != s[k - 1] and s[k - 1] == s[k + 1]:
                s[k] = s[k - 1]
    return s


def segment(cpa_sequence, smooth=True):
    """CPA label sequence -> list of skill instances [{skill, cpa_indices, cpa_subsequence}]."""
    phase = "explore"
    skills = []
    for cpa in cpa_sequence:
        skill, phase = cpa_to_skill(cpa, phase)
        skills.append(skill)
    if smooth:
        skills = _smooth(skills)
    labelled = [(k, cpa_sequence[k], skills[k]) for k in range(len(cpa_sequence))]
    # merge maximal contiguous runs of the same skill
    insts = []
    i = 0
    while i < len(labelled):
        skill = labelled[i][2]
        j = i
        idxs, subseq = [], []
        while j < len(labelled) and labelled[j][2] == skill:
            idxs.append(labelled[j][0]); subseq.append(labelled[j][1]); j += 1
        insts.append({"skill_type": skill, "cpa_indices": idxs, "cpa_subsequence": subseq})
        i = j
    return insts


def main(argv=None):
    ap = argparse.ArgumentParser(description="Skill segmentation over CPA sequences (weak-supervision prior).")
    ap.add_argument("--sequences", required=True, help="<...>.sequences.jsonl from hpop.extract.sequences")
    ap.add_argument("--library", default="rules/skill_library_seed.json")
    ap.add_argument("--out", required=True, help="output prefix (.skills.jsonl)")
    args = ap.parse_args(argv)

    lib = {s["name"]: s for s in json.load(open(args.library))}
    seqs = [json.loads(l) for l in open(args.sequences) if l.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    skill_freq = Counter()          # skill instances
    skill_traj = Counter()          # trajectories containing the skill
    n_inst = 0
    with open(args.out + ".skills.jsonl", "w", encoding="utf-8") as f:
        for s in seqs:
            insts = segment(s["cpa_sequence"])
            rec = {
                "trajectory_id": s.get("trajectory_id"), "instance_id": s.get("instance_id"),
                "repo": s.get("repo"), "resolved": s.get("resolved"),
                "global_skill_sequence": [si["skill_type"] for si in insts],
                "skill_instances": insts,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for si in insts:
                skill_freq[si["skill_type"]] += 1
            for st in set(si["skill_type"] for si in insts):
                skill_traj[st] += 1
            n_inst += len(insts)

    import statistics as st
    per = [len(json.loads(l)["skill_instances"]) for l in open(args.out + ".skills.jsonl")]
    print("trajectories     : {}  -> {}.skills.jsonl".format(len(seqs), args.out))
    print("skill instances  : {}  (mean {:.1f}/trajectory)".format(n_inst, st.mean(per) if per else 0))
    print("skill library    : {} types used (of {} seed)".format(len(skill_freq), len(lib)))
    print("frequency (instances / trajectories):")
    for k, v in skill_freq.most_common():
        print("   {:>4} / {:>3}   {}".format(v, skill_traj[k], k))


if __name__ == "__main__":
    main()
