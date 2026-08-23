"""Mini empirical test: does a local partial order improve next-step inference?

Uses the open-coded sample in `docs/trajectory_cpa_skill_2.*`:

    run test suite -> {search code symbol || inspect source} -> diagnose failure

The observed trace orders the middle two CPAs one way, but the dependency annotations mark them
`INCOMPARABLE`, so HPOP should treat both orderings as legal. We compare:

1. A strict sequential baseline learned from the observed order only.
2. A local partial-order model built from the annotated dependencies.

Run:
    PYTHONPATH=src .venv/bin/python scripts/mini_partial_order_test.py
"""
from __future__ import annotations

import json
from pathlib import Path

from hpop.inference.likelihood import frontier_softmax_logp
from hpop.inference.poset import Poset


ROOT = Path(__file__).resolve().parents[1]
CPA_PATH = ROOT / "docs" / "trajectory_cpa_skill_2.cpa_instances.jsonl"
DEP_PATH = ROOT / "docs" / "trajectory_cpa_skill_2.dependencies.jsonl"
FULL_CPA_VOCAB = 29  # from docs/cpa_view.html


def load_records(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_models():
    cpas = {row["cpa_instance_id"]: row for row in load_records(CPA_PATH)}
    deps = load_records(DEP_PATH)

    # Mini-skill: run test -> {search || inspect} -> diagnose
    skill_ids = [
        "IAMconsortium__nomenclature-384-CPA002",
        "IAMconsortium__nomenclature-384-CPA003",
        "IAMconsortium__nomenclature-384-CPA004",
        "IAMconsortium__nomenclature-384-CPA005",
    ]
    labels = {cid: cpas[cid]["proposed_cpa"] for cid in skill_ids}
    observed_order = skill_ids[:]  # the order in the trajectory
    swapped_order = [skill_ids[0], skill_ids[2], skill_ids[1], skill_ids[3]]

    chain_edges = list(zip(observed_order, observed_order[1:]))

    poset_edges = []
    incomparables = []
    for dep in deps:
        src, dst = dep["source_cpa_id"], dep["target_cpa_id"]
        if src not in labels or dst not in labels:
            continue
        if dep["proposed_relation"] == "INCOMPARABLE":
            incomparables.append((src, dst))
            continue
        poset_edges.append((src, dst))

    return {
        "labels": labels,
        "observed_order": observed_order,
        "swapped_order": swapped_order,
        "chain": Poset(skill_ids, chain_edges),
        "poset": Poset(skill_ids, poset_edges),
        "incomparables": incomparables,
    }


def frontier_trace(poset: Poset, seq):
    completed = []
    rows = []
    for step in seq:
        frontier = poset.frontier(completed)
        rows.append({
            "completed": completed[:],
            "frontier": frontier,
            "chosen": step,
        })
        completed.append(step)
    return rows


def summarize():
    ctx = build_models()
    observed = ctx["observed_order"]
    swapped = ctx["swapped_order"]
    chain = ctx["chain"]
    poset = ctx["poset"]

    chain_obs = frontier_softmax_logp(chain, observed, beta=2.0, eps=0.05)
    chain_swap = frontier_softmax_logp(chain, swapped, beta=2.0, eps=0.05)
    poset_obs = frontier_softmax_logp(poset, observed, beta=2.0, eps=0.05)
    poset_swap = frontier_softmax_logp(poset, swapped, beta=2.0, eps=0.05)

    poset_frontier_sizes = [
        len(row["frontier"]) for row in frontier_trace(poset, observed)
    ]
    avg_frontier = sum(poset_frontier_sizes) / len(poset_frontier_sizes)
    reduction = 1.0 - (avg_frontier / FULL_CPA_VOCAB)

    print("Mini HPOP partial-order test")
    print("============================")
    print("Source motif: run test suite -> {search code symbol || inspect source} -> diagnose failure")
    print("Annotated incomparable pair:")
    for a, b in ctx["incomparables"]:
        print(f"  - {ctx['labels'][a]} || {ctx['labels'][b]}")

    print("\nOrders under test:")
    print("  observed :", " -> ".join(ctx["labels"][cid] for cid in observed))
    print("  swapped  :", " -> ".join(ctx["labels"][cid] for cid in swapped))

    print("\nExact legal-order counts:")
    print(f"  unrestricted permutations : 24")
    print(f"  strict sequence baseline  : {chain.num_linear_extensions()}")
    print(f"  partial-order model       : {poset.num_linear_extensions()}")

    print("\nFrontier check after `run test suite`:")
    chain_frontier = [ctx["labels"][cid] for cid in chain.frontier([observed[0]])]
    poset_frontier = [ctx["labels"][cid] for cid in poset.frontier([observed[0]])]
    print(f"  chain frontier : {chain_frontier}")
    print(f"  poset frontier : {poset_frontier}")

    print("\nLog-likelihoods (higher is better):")
    print(f"  chain  observed : {chain_obs:.4f}")
    print(f"  chain  swapped  : {chain_swap:.4f}")
    print(f"  poset  observed : {poset_obs:.4f}")
    print(f"  poset  swapped  : {poset_swap:.4f}")
    print(f"  avg held-out chain : {(chain_obs + chain_swap) / 2.0:.4f}")
    print(f"  avg held-out poset : {(poset_obs + poset_swap) / 2.0:.4f}")

    print("\nBranching reduction vs choosing from all 29 CPAs:")
    print(f"  average frontier size : {avg_frontier:.2f}")
    print(f"  reduction             : {100.0 * reduction:.1f}%")


if __name__ == "__main__":
    summarize()
