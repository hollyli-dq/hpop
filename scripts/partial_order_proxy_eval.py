"""Proxy evaluation of local partial orders on repeated skill instances.

This is a stronger follow-up to the one-off toy test. It uses the repeated skill instances in
`data/modelling/swe_rebench/pilot100.skills.jsonl` and asks:

1. If we learn a strict chain baseline from training instances, what is the held-out cost?
2. If we instead learn a pairwise-consensus partial order, what is the held-out cost?

Method:
  - Keep only skill instances whose CPA subsequence has unique labels (so a support set is well-defined).
  - Group by `(skill_type, support_set)`.
  - Keep groups with at least 3 instances and at least 2 distinct observed orders.
  - Leave one instance out.
  - Chain baseline = modal training order, converted to a strict chain.
  - Partial-order proxy = for each pair (a,b), add `a -> b` only if *all* training instances put
    `a` before `b`; leave the pair incomparable otherwise.

This is still not full HPOP inference. It is a small, reproducible proxy for the cost tradeoff:
partial orders usually increase entropy on the majority ordering, but improve robustness on genuine
re-orderings.

Run:
    PYTHONPATH=src .venv/bin/python scripts/partial_order_proxy_eval.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from hpop.inference.likelihood import frontier_softmax_logp
from hpop.inference.poset import Poset


ROOT = Path(__file__).resolve().parents[1]
SKILLS_PATH = ROOT / "data" / "modelling" / "swe_rebench" / "pilot100.skills.jsonl"
CPA_VOCAB = 29  # docs/cpa_view.html


def load_skill_groups():
    rows = [json.loads(line) for line in SKILLS_PATH.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        for skill in row["skill_instances"]:
            seq = tuple(skill["cpa_subsequence"])
            if len(seq) != len(set(seq)):
                continue
            support = tuple(sorted(seq))
            groups[(skill["skill_type"], support)].append(seq)
    return groups


def build_pairwise_consensus_poset(support, train_seqs):
    edges = []
    for i, a in enumerate(support):
        for b in support[i + 1:]:
            a_before_b = sum(seq.index(a) < seq.index(b) for seq in train_seqs)
            b_before_a = len(train_seqs) - a_before_b
            if a_before_b == len(train_seqs):
                edges.append((a, b))
            elif b_before_a == len(train_seqs):
                edges.append((b, a))
    return Poset(list(support), edges)


def avg_frontier_size(poset, seq):
    return sum(len(poset.frontier(seq[:t])) for t in range(len(seq))) / len(seq)


def summarize_bucket(name, rows):
    mean_chain = sum(r["chain_logp"] for r in rows) / len(rows)
    mean_poset = sum(r["poset_logp"] for r in rows) / len(rows)
    mean_chain_frontier = sum(r["chain_frontier"] for r in rows) / len(rows)
    mean_poset_frontier = sum(r["poset_frontier"] for r in rows) / len(rows)
    print(f"{name} ({len(rows)} held-out cases)")
    print(f"  mean logp   chain={mean_chain:.4f}  poset={mean_poset:.4f}  delta={mean_poset - mean_chain:.4f}")
    print(f"  mean cost   chain={-mean_chain:.4f}  poset={-mean_poset:.4f}")
    print(f"  wins        poset={sum(r['poset_logp'] > r['chain_logp'] for r in rows)}"
          f"  chain={sum(r['chain_logp'] > r['poset_logp'] for r in rows)}"
          f"  ties={sum(abs(r['poset_logp'] - r['chain_logp']) < 1e-12 for r in rows)}")
    print(f"  frontier    chain={mean_chain_frontier:.3f}  poset={mean_poset_frontier:.3f}"
          f"  poset-vs-29 reduction={100.0 * (1.0 - mean_poset_frontier / CPA_VOCAB):.1f}%")


def main():
    groups = load_skill_groups()
    eval_rows = []
    for key, seqs in groups.items():
        if len(seqs) < 3 or len(set(seqs)) < 2:
            continue
        support = list(key[1])
        for i, heldout in enumerate(seqs):
            train = seqs[:i] + seqs[i + 1:]
            modal_order = Counter(train).most_common(1)[0][0]
            chain = Poset(list(support), list(zip(modal_order, modal_order[1:])))
            poset = build_pairwise_consensus_poset(support, train)
            eval_rows.append({
                "group": key,
                "heldout": heldout,
                "modal_order": modal_order,
                "is_reorder": heldout != modal_order,
                "chain_logp": frontier_softmax_logp(chain, heldout, beta=2.0, eps=0.05),
                "poset_logp": frontier_softmax_logp(poset, heldout, beta=2.0, eps=0.05),
                "chain_frontier": avg_frontier_size(chain, heldout),
                "poset_frontier": avg_frontier_size(poset, heldout),
            })

    all_rows = eval_rows
    reorder_rows = [r for r in eval_rows if r["is_reorder"]]

    print("Partial-order proxy eval on repeated skill instances")
    print("===================================================")
    print("Selection: unique-label skill instances, grouped by (skill_type, support_set),")
    print("           at least 3 instances and at least 2 distinct orders in pilot100.")
    print(f"Candidate groups: {len({r['group'] for r in all_rows})}")
    print()
    summarize_bucket("ALL", all_rows)
    print()
    summarize_bucket("REORDER ONLY", reorder_rows)
    print()
    print("Per-group summary (mean delta = poset_logp - chain_logp):")
    by_group = defaultdict(list)
    for row in all_rows:
        by_group[row["group"]].append(row)
    for group, rows in sorted(by_group.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        delta = sum(r["poset_logp"] - r["chain_logp"] for r in rows) / len(rows)
        print(f"  {group}: n={len(rows):2d}  mean_delta={delta:+.4f}"
              f"  reorders={sum(r['is_reorder'] for r in rows):2d}")


if __name__ == "__main__":
    main()
