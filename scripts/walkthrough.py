"""End-to-end worked example: how the data is prepared, what ground truth is, what HPOP infers.

Four parts, all printed from real files and real fits — nothing here is illustrative:

  PART 1  dataset preparation, one real trajectory carried through every stage
  PART 2  synthetic ground truth — what "the true answer" actually is
  PART 3  what HPOP infers on that synthetic trace, aligned against the truth
  PART 4  what HPOP infers on a real SWE-rebench trajectory

Run:
    PYTHONPATH=src .venv/bin/python scripts/walkthrough.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "interim" / "swe_rebench" / "pilot100.jsonl"
ANNOT = ROOT / "data" / "annotated" / "swe_rebench" / "cpa_rule.cpa_instances.jsonl"
SEQ = ROOT / "data" / "modelling" / "swe_rebench" / "pilot100.sequences.jsonl"

TRACE = "tianocore__edk2-pytool-library-372"


def rule(title, ch="="):
    print(f"\n{ch * 96}\n{title}\n{ch * 96}")


def head(title):
    print(f"\n--- {title} " + "-" * max(0, 92 - len(title)))


# =============================================================================================
def part1_dataset():
    rule("PART 1 · DATASET PREPARATION — one real trajectory through every stage")

    raw = next(json.loads(l) for l in RAW.open() if json.loads(l)["trace_id"] == TRACE)
    print(f"trajectory  : {TRACE}")
    print(f"repository  : {raw['repo']}   resolved={raw['resolved']}   exit={raw['exit_status']}")
    print(f"source      : {raw['source']}  (nebius/SWE-rebench-openhands-trajectories)")

    head("STAGE 1 · raw agent events, after ingest (hpop.ingest.swe_rebench)")
    print("The upstream corpus stores tool-call arguments as serialized JSON strings; ingest")
    print("deserializes them and emits one normalized record per action, keeping the event index.")
    for t in raw["action_tokens"][1:4]:
        obs = (t.get("observation") or "")[:78].replace("\n", " ")
        print(f"  e{t['i']:04d}  tool={t['tool_name']:<18} family={t['tool_family']:<8}")
        print(f"         command    : {str(t.get('command'))[:78]}")
        print(f"         observation: {obs}")
        print(f"         after_fail : {t.get('after_fail')}")
    print(f"  ... {raw['num_action_tokens']} action tokens total")

    head("STAGE 2 · CPA occurrences (hpop.annotate.rule_apply -> silver labels)")
    print("Each event span is mapped to one Canonical Procedural Action. NOTE: in this pilot the")
    print("labels are RULE-BASED silver, not the LLM open-coding the paper specifies -- every")
    print("record is MATCH_EXISTING at a constant confidence of 0.75.")
    recs = [json.loads(l) for l in ANNOT.open() if TRACE in l]
    for r in recs[:2] + [x for x in recs if x["outcome"] == "FAILURE"][:1]:
        print(f"  {r['occurrence_id']}")
        print(f"     label={r['canonical_label']:<22} phase={r['phase']:<10} outcome={r['outcome']}")
        print(f"     events={r['source_event_ids']}  decision={r['decision']}  "
              f"conf(label)={r['label_confidence']} conf(boundary)={r['boundary_confidence']}")
        print(f"     evidence: {str(r['evidence'][0])[:80] if r['evidence'] else '-'}")

    head("STAGE 3 · occurrence-level CPA sequence (hpop.extract.sequences)")
    seq = next(json.loads(l) for l in SEQ.open() if l and json.loads(l)["instance_id"] == TRACE)
    cpas = seq["cpa_sequence"]
    print(f"  T = {len(cpas)} occurrences. Repeated labels stay DISTINCT occurrences -- that is the")
    print("  whole point: c_i = c_j does not mean o_i = o_j.")
    for i in range(0, min(len(cpas), 24), 6):
        print("   " + "  ".join(f"{t+i:2d}:{c[:13]:<13}" for t, c in enumerate(cpas[i:i + 6])))
    from collections import Counter
    rep = Counter(cpas)
    print(f"  most repeated: " + ", ".join(f"{k}x{v}" for k, v in rep.most_common(4)))
    print(f"  repeated-occurrence fraction: {1 - len(set(cpas)) / len(cpas):.0%}")

    head("STAGE 4 · seed segments — the admissible boundaries HPOP may merge at")
    print("  The paper uses a phase-guided LLM oversegmentation. This pilot has none attached, so")
    print("  we use the MAXIMAL oversegmentation: one seed per occurrence. Merge-only is then")
    print("  vacuous (every position is an admissible boundary) and no seeding error can leak in.")
    print(f"  J = {len(cpas)} seeds, each of length 1")

    head("STAGE 5 · candidate blocks the model scores")
    print("  A skill instance is any run of adjacent seeds, up to D_max. For D_max=12 the lattice")
    print("  over this trace contains:")
    J, D = len(cpas), 12
    nb = sum(1 for i in range(J) for w in range(1, min(D, J - i) + 1))
    print(f"    {nb} candidate blocks x K_max skills, scored by the recurrent frontier likelihood")
    print(f"    e.g. block (i=8, w=4) = {cpas[8:12]}")
    print("  Exact semi-Markov forward-backward then marginalizes over ALL legal segmentations.")

    head("STAGE 6 · repository-disjoint split")
    rows = [json.loads(l) for l in SEQ.open()]
    print(f"  {len(rows)} trajectories over {len({r['repo'] for r in rows})} repositories, "
          f"{sum(len(r['cpa_sequence']) for r in rows)} occurrences")
    print("  Split is by REPOSITORY, never by trajectory, so test repos are unseen at train time.")
    return cpas


# =============================================================================================
def part2_ground_truth():
    rule("PART 2 · GROUND TRUTH — what 'the true answer' looks like (synthetic)")
    from hpop.synth.generator import sample_corpus, seeds_of

    world, traces = sample_corpus(seed=0, n_traces=40, K_true=4, V=12)
    print("The generator draws a TRUE skill library, a TRUE global order over skill types, and")
    print("then executes them. Everything below is the answer HPOP is asked to recover.\n")

    head("TRUE skill library (4 reusable skills over a 12-CPA vocabulary)")
    for k, s in enumerate(world.skills):
        roles = [world.vocab[r] for r in s.roles]
        edges = [f"{world.vocab[a]}->{world.vocab[b]}" for a, b in s.edges]
        print(f"  SKILL {k}: nodes({len(roles)}) = {', '.join(roles)}")
        print(f"           cover edges = {'; '.join(edges) if edges else '(none)'}")
        print(f"           verify role = {[world.vocab[r] for r in s.verify_roles] or '-'}")

    head("TRUE global order over skill TYPES")
    print("  " + ("; ".join(f"SKILL {a} -> SKILL {b}" for a, b in world.global_edges)
                  or "(no constraints)"))

    head("TRUE trace 0 — the latent program and its observed execution")
    t = traces[0]
    print(f"  T = {len(t.cpas)} occurrences, L = {len(t.skill_labels)} skill instances")
    print(f"  true instance labels : {t.skill_labels}")
    print(f"  true boundaries      : {t.true_boundaries}   (cut positions in occurrence space)")
    for (a, b), k in zip(t.instance_spans, t.skill_labels):
        block = [world.vocab[c] for c in t.cpas[a:b]]
        reruns = len(block) - len(set(block))
        print(f"    span [{a:2d},{b:2d})  SKILL {k}  ({b-a} occ, {reruns} re-executions)")
        print(f"        {' -> '.join(x[:11] for x in block)}")
    return world, traces


# =============================================================================================
def part3_inference(world, traces):
    rule("PART 3 · WHAT HPOP INFERS — aligned against that ground truth")
    from hpop.eval.metrics import (evaluate, match_skills, occurrence_labels,
                                   decoded_to_cpa_spans, transitive_reduction)
    from hpop.inference.hpop import HPOP, HPOPConfig
    from hpop.synth.generator import seeds_of

    split = int(0.7 * len(traces))
    corpus = [seeds_of(x) for x in traces[:split]]
    print("fitting HPOP (K_max=6, D_max=8, 12 EM iterations) ...", flush=True)
    m = HPOP(HPOPConfig(V=12, K_max=6, D_max=8), rng=np.random.default_rng(0))
    m.fit(corpus, iters=12, warmup=3)

    decoded = [m.decode(s) for s in corpus]
    all_t, all_p = [], []
    for tr, segs in zip(traces[:split], decoded):
        T = len(tr.cpas)
        all_p += occurrence_labels(decoded_to_cpa_spans(tr, segs), T).tolist()
        all_t += occurrence_labels([(a, b, k) for (a, b), k in
                                    zip(tr.instance_spans, tr.skill_labels)], T).tolist()
    mapping, _ = match_skills(all_t, all_p, 4, 6)
    res = evaluate(world, traces[:split], decoded, 6, m.D, m.global_structure(corpus))

    head("INFERRED skill library (library slots matched to true skills by Hungarian assignment)")
    inv = {v: k for k, v in mapping.items()}
    for k_true in range(4):
        slot = inv.get(k_true)
        if slot is None:
            print(f"  SKILL {k_true}: NOT RECOVERED")
            continue
        true_cov = transitive_reduction(world.local_matrices()[k_true] > 0)
        pred_cov = transitive_reduction(np.asarray(m.D[slot]) > 0)
        te = {(world.vocab[a], world.vocab[b]) for a, b in zip(*np.where(true_cov))}
        pe = {(world.vocab[a], world.vocab[b]) for a, b in zip(*np.where(pred_cov))}
        print(f"  SKILL {k_true}  <-  slot {slot}   usage {m.counts[slot]:.0f} instances")
        print(f"     recovered : {'; '.join(f'{a}->{b}' for a, b in sorted(te & pe)) or '-'}")
        print(f"     missed    : {'; '.join(f'{a}->{b}' for a, b in sorted(te - pe)) or '-'}")
        print(f"     spurious  : {'; '.join(f'{a}->{b}' for a, b in sorted(pe - te)) or '-'}")

    head("INFERRED vs TRUE segmentation of trace 0")
    tr, segs = traces[0], decoded[0]
    spans = decoded_to_cpa_spans(tr, segs)
    print("   true : " + "  ".join(f"[{a},{b})=S{k}" for (a, b), k in
                                   zip(tr.instance_spans, tr.skill_labels)))
    print("   pred : " + "  ".join(f"[{a},{b})=slot{k}"
                                   f"{'(S' + str(mapping[k]) + ')' if k in mapping else ''}"
                                   for a, b, k in spans))

    head("RECOVERY SCORES on this fit")
    for key in ["skill_ari", "boundary_f1", "local_rel_f1", "local_cover_f1", "global_rel_f1"]:
        print(f"   {key:<18} {res.get(key, float('nan')):.3f}")
    print(f"   {'active library':<18} {len(m.active_skills())}  (true K = 4)")
    return res


# =============================================================================================
def part4_real():
    rule("PART 4 · WHAT HPOP INFERS ON A REAL SWE-REBENCH TRAJECTORY")
    from collections import Counter
    from hpop.inference.hpop import HPOP, HPOPConfig

    rows = [json.loads(l) for l in SEQ.open()]
    counts = Counter(c for r in rows for c in r["cpa_sequence"])
    vocab = [c for c, _ in counts.most_common()]
    idx = {c: i for i, c in enumerate(vocab)}
    for r in rows:
        r["seq"] = [idx[c] for c in r["cpa_sequence"]]

    repos = sorted({r["repo"] for r in rows})
    rng = np.random.default_rng(0)
    rng.shuffle(repos)
    tr_repos = set(repos[:int(0.7 * len(repos))])
    train = [r for r in rows if r["repo"] in tr_repos]
    test = [r for r in rows if r["repo"] not in tr_repos]

    print(f"fitting on {len(train)} train trajectories from {len(tr_repos)} repositories "
          f"(K_max=10, D_max=12, 12 iters) ...", flush=True)
    m = HPOP(HPOPConfig(V=len(vocab), K_max=10, D_max=12, lam_seg=3.0),
             rng=np.random.default_rng(0))
    m.fit([[[c] for c in r["seq"]] for r in train], iters=12, warmup=3)

    head("INFERRED skill library from real traces (no supervision beyond CPA labels)")
    tot = float(sum(np.maximum(m.counts - m.cfg.alpha / m.cfg.K_max, 0)))
    for k in m.active_skills():
        comp = [vocab[i] for i in np.argsort(-m.theta[k])[:4]]
        edges = [f"{vocab[a]}->{vocab[b]}" for a in range(len(vocab)) for b in range(len(vocab))
                 if m.D[k, a, b] > 0]
        share = (m.counts[k] - m.cfg.alpha / m.cfg.K_max) / tot
        print(f"  slot {k}  usage {share:5.1%}  composition: {', '.join(comp)}")
        print(f"          order: {'; '.join(edges) if edges else '(none inferred -- interchangeable)'}")

    head("DECODED held-out trajectory (a repository never seen in training)")
    r = max(test, key=lambda x: len(x["seq"]))
    segs = m.decode([[c] for c in r["seq"]])
    print(f"  {r['instance_id']}  ({r['repo']}, resolved={r['resolved']}, "
          f"{len(r['seq'])} occurrences -> {len(segs)} skill instances)")
    for a, b, k in segs:
        block = [vocab[c] for c in r["seq"][a:b]]
        reruns = len(block) - len(set(block))
        flag = "  <- repair loop" if reruns else ""
        print(f"    [{a:3d},{b:3d})  slot {k}  ({b-a} occ, {reruns} re-executions){flag}")
        print(f"        {' -> '.join(x[:12] for x in block)}")


def dump_json(out=ROOT / "data" / "experiments" / "walkthrough.json"):
    """Re-run every part, capturing structured records so the HTML view is generated, not typed."""
    from hpop.eval.metrics import (evaluate, match_skills, occurrence_labels,
                                   decoded_to_cpa_spans, transitive_reduction)
    from hpop.inference.hpop import HPOP, HPOPConfig
    from hpop.synth.generator import sample_corpus, seeds_of
    from collections import Counter

    doc = {}

    # ---- part 1: real pipeline stages -------------------------------------------------------
    raw = next(json.loads(l) for l in RAW.open() if json.loads(l)["trace_id"] == TRACE)
    recs = [json.loads(l) for l in ANNOT.open() if TRACE in l]
    seq = next(json.loads(l) for l in SEQ.open() if json.loads(l)["instance_id"] == TRACE)
    rows = [json.loads(l) for l in SEQ.open()]
    doc["pipeline"] = {
        "trace": TRACE, "repo": raw["repo"], "resolved": raw["resolved"],
        "exit_status": raw["exit_status"], "n_events": raw["num_action_tokens"],
        "events": [{"i": t["i"], "tool": t["tool_name"], "family": t["tool_family"],
                    "command": str(t.get("command"))[:90],
                    "observation": (t.get("observation") or "")[:90].replace("\n", " "),
                    "after_fail": bool(t.get("after_fail"))}
                   for t in raw["action_tokens"][1:4]],
        "occurrences": [{k: r[k] for k in ("occurrence_id", "canonical_label", "phase", "outcome",
                                           "decision", "label_confidence", "boundary_confidence",
                                           "source_event_ids")} | {"evidence": (r["evidence"] or [""])[0][:90]}
                        for r in recs[:2] + [x for x in recs if x["outcome"] == "FAILURE"][:1]],
        "cpa_sequence": seq["cpa_sequence"],
        "repeat_fraction": 1 - len(set(seq["cpa_sequence"])) / len(seq["cpa_sequence"]),
        "most_repeated": Counter(seq["cpa_sequence"]).most_common(5),
        "corpus": {"n_traj": len(rows), "n_repo": len({r["repo"] for r in rows}),
                   "n_occ": sum(len(r["cpa_sequence"]) for r in rows)},
    }

    # ---- part 2 + 3: synthetic ground truth and inference ------------------------------------
    world, traces = sample_corpus(seed=0, n_traces=40, K_true=4, V=12)
    doc["truth"] = {
        "vocab": world.vocab,
        "skills": [{"id": k, "roles": [world.vocab[r] for r in s.roles],
                    "edges": [[world.vocab[a], world.vocab[b]] for a, b in s.edges],
                    "verify": [world.vocab[r] for r in s.verify_roles]}
                   for k, s in enumerate(world.skills)],
        "global_edges": [[a, b] for a, b in world.global_edges],
        "trace0": {"cpas": [world.vocab[c] for c in traces[0].cpas],
                   "labels": traces[0].skill_labels,
                   "boundaries": traces[0].true_boundaries,
                   "spans": [[a, b] for a, b in traces[0].instance_spans]},
    }

    split = int(0.7 * len(traces))
    corpus = [seeds_of(x) for x in traces[:split]]
    m = HPOP(HPOPConfig(V=12, K_max=6, D_max=8), rng=np.random.default_rng(0))
    m.fit(corpus, iters=12, warmup=3)
    decoded = [m.decode(s) for s in corpus]
    all_t, all_p = [], []
    for tr, segs in zip(traces[:split], decoded):
        T = len(tr.cpas)
        all_p += occurrence_labels(decoded_to_cpa_spans(tr, segs), T).tolist()
        all_t += occurrence_labels([(a, b, k) for (a, b), k in
                                    zip(tr.instance_spans, tr.skill_labels)], T).tolist()
    mapping, _ = match_skills(all_t, all_p, 4, 6)
    res = evaluate(world, traces[:split], decoded, 6, m.D, m.global_structure(corpus))
    inv = {v: k for k, v in mapping.items()}
    inferred = []
    for k_true in range(4):
        slot = inv.get(k_true)
        if slot is None:
            inferred.append({"true": k_true, "slot": None})
            continue
        tc = transitive_reduction(world.local_matrices()[k_true] > 0)
        pc = transitive_reduction(np.asarray(m.D[slot]) > 0)
        te = {(world.vocab[a], world.vocab[b]) for a, b in zip(*np.where(tc))}
        pe = {(world.vocab[a], world.vocab[b]) for a, b in zip(*np.where(pc))}
        inferred.append({"true": k_true, "slot": int(slot),
                         "usage": float(m.counts[slot]),
                         "true_edges": sorted(map(list, te)),
                         "pred_edges": sorted(map(list, pe)),
                         "recovered": sorted(map(list, te & pe)),
                         "missed": sorted(map(list, te - pe)),
                         "spurious": sorted(map(list, pe - te))})
    # inferred global order, expressed in TRUE skill ids so it can be drawn against the truth
    gpred = m.global_structure(corpus)
    gmapped = sorted({(mapping[a], mapping[b]) for a, b in gpred
                      if a in mapping and b in mapping})
    doc["inferred"] = {
        "skills": inferred, "scores": {k: float(v) for k, v in res.items()
                                       if isinstance(v, (int, float))},
        "K_active": int(len(m.active_skills())),
        "global_edges": [list(e) for e in gmapped],
        "global_edges_raw_slots": [[int(a), int(b)] for a, b in gpred],
        "trace0_pred": [[a, b, int(k), mapping.get(int(k))]
                        for a, b, k in decoded_to_cpa_spans(traces[0], decoded[0])],
    }

    # ---- part 4: real inference ---------------------------------------------------------------
    counts = Counter(c for r in rows for c in r["cpa_sequence"])
    vocab = [c for c, _ in counts.most_common()]
    idx = {c: i for i, c in enumerate(vocab)}
    for r in rows:
        r["seq"] = [idx[c] for c in r["cpa_sequence"]]
    repos = sorted({r["repo"] for r in rows})
    rng = np.random.default_rng(0)
    rng.shuffle(repos)
    tr_repos = set(repos[:int(0.7 * len(repos))])
    train = [r for r in rows if r["repo"] in tr_repos]
    test = [r for r in rows if r["repo"] not in tr_repos]
    mr = HPOP(HPOPConfig(V=len(vocab), K_max=10, D_max=12, lam_seg=3.0),
              rng=np.random.default_rng(0))
    mr.fit([[[c] for c in r["seq"]] for r in train], iters=12, warmup=3)
    tot = float(sum(np.maximum(mr.counts - mr.cfg.alpha / mr.cfg.K_max, 0)))
    lib = []
    for k in mr.active_skills():
        lib.append({"slot": int(k),
                    "usage": float((mr.counts[k] - mr.cfg.alpha / mr.cfg.K_max) / tot),
                    "composition": [vocab[i] for i in np.argsort(-mr.theta[k])[:4]],
                    "edges": [[vocab[a], vocab[b]] for a in range(len(vocab))
                              for b in range(len(vocab)) if mr.D[k, a, b] > 0]})
    r = max(test, key=lambda x: len(x["seq"]))
    segs = mr.decode([[c] for c in r["seq"]])
    doc["real"] = {
        "vocab": vocab, "n_train": len(train), "n_train_repo": len(tr_repos),
        "library": lib,
        "decoded": {"instance_id": r["instance_id"], "repo": r["repo"],
                    "resolved": r["resolved"], "n_occ": len(r["seq"]),
                    "segments": [{"a": a, "b": b, "slot": int(k),
                                  "cpas": [vocab[c] for c in r["seq"][a:b]],
                                  "reruns": (b - a) - len(set(r["seq"][a:b]))}
                                 for a, b, k in segs]},
    }

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=2))
    print(f"wrote {out}")
    return doc


if __name__ == "__main__":
    import sys
    if "--json" in sys.argv:
        dump_json()
    else:
        part1_dataset()
        world, traces = part2_ground_truth()
        part3_inference(world, traces)
        part4_real()
        print("\n" + "=" * 96)
