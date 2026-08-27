"""Artifact grounding — bind each CPA occurrence to a normalized <role:artifact> and induce the
per-trajectory partial order from artifact flow.

Why: bare CPA-type sequences (EDIT_SOURCE, READ_SOURCE, ...) carry no operands, so a partial order over
them collapses to one global pipeline. Binding the operand (the file/symbol the action touches, taken
from the raw event args and normalized to a portable id + role) lets us order actions that share an
artifact and leave actions on different artifacts incomparable — i.e. a genuine DAG per trajectory.

Label stays portable (EDIT_SOURCE); the artifact id is a basename/symbol, NOT a line number, so it still
recurs across repos. Precedence = same-artifact temporal chain (a ≺ b iff they touch the same artifact
and a precedes b). Cross-artifact pairs are concurrent unless chained transitively.

Input : cpa_rule.jsonl (occurrences w/ source_event_ids) + the raw pilot jsonl (events w/ args).
Output: grounded.jsonl — per traj {grounded_sequence, occurrences[+artifact], precedence_edges, stats}.

Usage:
    PYTHONPATH=src .venv/bin/python -m hpop.extract.ground \
        --annot data/annotated/swe_rebench/cpa_rule.jsonl \
        --raw   data/interim/swe_rebench/pilot500.jsonl \
        --out   data/modelling/swe_rebench/grounded
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

# CPAs whose primary effect is to PRODUCE/mutate an artifact (vs consume/observe it).
OUT_LABELS = {"EDIT_SOURCE", "WRITE_TEST", "WRITE_REPRODUCTION_SCRIPT", "ADD_DEBUG_INSTRUMENTATION",
              "FORMAT_CODE", "REFACTOR_CODE", "REVERT_CHANGE", "APPLY_PATCH"}


def _role(path):
    p = path.lower()
    base = path.rsplit("/", 1)[-1]
    if "conftest" in p or re.search(r"(^|/)test_|_test\.py|/tests?/", p):
        return "test", base
    if re.search(r"repro|reproduce|debug_|scratch|bug_|/tmp/|check_|minimal", p):
        return "repro_script", base
    if re.search(r"setup\.(py|cfg)|pyproject\.toml|\.ini$|\.cfg$|\.toml$|\.ya?ml$|requirements", p):
        return "config", base
    if re.search(r"readme|changelog|contributing|/docs?/|\.md$|\.rst$", p):
        return "doc", base
    return "source", base


def _artifact_of(ev):
    """Return (role, id) for the artifact this event touches, or None."""
    a = ev.get("args") or {}
    tool = ev.get("tool_name") or ""
    cmd = (a.get("command") or ev.get("command") or "")
    if tool == "str_replace_editor":
        path = a.get("path") or ""
        if path and path not in ("/workspace", "/") and "/" in path:
            if path.rstrip("/").count("/") <= 1 and "." not in path.rsplit("/", 1)[-1]:
                return ("dir", path.rstrip("/").rsplit("/", 1)[-1])   # a directory view
            return _role(path)
        return None
    # bash: pull a file/script/test target, else a searched symbol, else an installed package
    m = re.search(r"pytest\s+(?:-\S+\s+)*([^\s|>&]+\.py[\w:./\[\]-]*)", cmd)
    if m:
        return _role(m.group(1).split("::")[0])
    m = re.search(r"python[0-9.]*\s+(?:-\S+\s+)*([^\s|>&]+\.py)\b", cmd)
    if m:
        return _role(m.group(1))
    m = re.search(r"\b(?:pip|pip3)\s+install\s+(?:-\S+\s+)*([A-Za-z][\w.\-]+)", cmd)
    if m:
        return ("dependency", m.group(1).lower())
    m = re.search(r"\b(?:grep|rg)\b\s+(?:-\S+\s+)*[\"']?([A-Za-z_][\w]{2,})", cmd)
    if m:
        return ("symbol", m.group(1))
    return None


def ground_trajectory(ann, raw):
    by_i = {ev["i"]: ev for ev in raw.get("action_tokens", [])}
    occs = []
    for c in ann.get("cpa_instances", []):
        label = c.get("candidate_label")
        idxs = [int(e[1:]) for e in c.get("source_event_ids", [])]
        art = None
        for i in idxs:                       # first event that yields a concrete artifact
            ev = by_i.get(i)
            if ev is None:
                continue
            art = _artifact_of(ev)
            if art:
                break
        role, aid = (art if art else (None, None))
        io = "out" if label in OUT_LABELS else "in"
        token = "{}⟨{}:{}⟩".format(label, role, aid) if art else label
        occs.append({"label": label, "role": role, "artifact": aid, "io": io,
                     "start": c.get("start_event_id"), "token": token})

    # precedence = same-artifact temporal chain (Hasse: consecutive touches)
    chains = defaultdict(list)
    for k, o in enumerate(occs):
        if o["artifact"]:
            chains[(o["role"], o["artifact"])].append(k)
    edges = []
    for key, ks in chains.items():
        for a, b in zip(ks, ks[1:]):
            edges.append([a, b])

    n = len(occs)
    grounded = sum(1 for o in occs if o["artifact"])
    # reachability over the chain-DAG → comparable pairs
    succ = defaultdict(set)
    for a, b in edges:
        succ[a].add(b)
    reach = {}
    def _r(u):
        if u in reach:
            return reach[u]
        s = set(succ[u])
        for v in list(succ[u]):
            s |= _r(v)
        reach[u] = s
        return s
    comparable = sum(len(_r(u)) for u in range(n))
    total_pairs = n * (n - 1) // 2
    return {
        "trajectory_id": ann.get("trajectory_id"), "instance_id": ann.get("_instance_id"),
        "repo": ann.get("_repo"), "resolved": ann.get("_resolved"),
        "grounded_sequence": [o["token"] for o in occs],
        "occurrences": occs,
        "precedence_edges": edges,
        "artifact_chains": {"{}:{}".format(r, a): ks for (r, a), ks in chains.items()},
        "stats": {"n_occ": n, "grounded": grounded, "n_artifacts": len(chains),
                  "chain_edges": len(edges), "comparable_pairs": comparable,
                  "incomparable_pairs": total_pairs - comparable, "total_pairs": total_pairs},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True, help="prefix (.jsonl + .vocab.json)")
    args = ap.parse_args(argv)

    raw_by_id = {}
    for l in open(args.raw):
        if l.strip():
            t = json.loads(l)
            raw_by_id[t.get("trace_id")] = t

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fout = open(args.out + ".jsonl", "w", encoding="utf-8")
    bare_vocab, gnd_vocab, role_freq = set(), set(), Counter()
    tot = Counter()
    n_traj = 0
    for l in open(args.annot):
        if not l.strip():
            continue
        ann = json.loads(l)
        raw = raw_by_id.get(ann.get("_instance_id"))
        if raw is None:
            continue
        g = ground_trajectory(ann, raw)
        fout.write(json.dumps(g, ensure_ascii=False) + "\n")
        n_traj += 1
        for o in g["occurrences"]:
            bare_vocab.add(o["label"]); gnd_vocab.add(o["token"])
            if o["role"]:
                role_freq[o["role"]] += 1
        s = g["stats"]
        for k, v in s.items():
            tot[k] += v
    fout.close()
    json.dump(sorted(gnd_vocab), open(args.out + ".vocab.json", "w"), indent=0)

    cov = 100.0 * tot["grounded"] / max(tot["n_occ"], 1)
    conc = 100.0 * tot["incomparable_pairs"] / max(tot["total_pairs"], 1)
    print("grounded {} trajectories -> {}.jsonl".format(n_traj, args.out))
    print("artifact coverage : {:.0f}% of occurrences carry an artifact ({}/{})".format(cov, tot["grounded"], tot["n_occ"]))
    print("vocabulary        : {} bare CPA labels  ->  {} grounded <label:role:id> tokens".format(len(bare_vocab), len(gnd_vocab)))
    print("partial order     : {:.0f}% of occurrence pairs are INCOMPARABLE (concurrent); {} chain-edges; {} artifacts".format(
        conc, tot["chain_edges"], tot["n_artifacts"]))
    print("roles             : " + ", ".join("{}={}".format(k, v) for k, v in role_freq.most_common()))


if __name__ == "__main__":
    main()
