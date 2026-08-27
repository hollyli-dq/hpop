"""Runnable BPOP/penalty demo (no API key). PYTHONPATH=src .venv/bin/python scripts/inference_demo.py"""
import math
from hpop.inference import Poset, frontier_softmax_logp, crp_predictive, new_skill_logpenalty

print("== frontier-softmax likelihood ==")
h = Poset(["a","b","c","d"], [("a","c"),("b","c"),("c","d")])   # a||b, both before c, c before d
print("linear extensions:", h.num_linear_extensions(), "| incomparable:", h.incomparable_pairs())
for seq in (["a","b","c","d"], ["b","a","c","d"], ["c","a","b","d"]):
    lp = frontier_softmax_logp(h, seq, beta=2.0, eps=0.05)
    print(f"  logp({'-'.join(seq)}) = {lp:7.3f}  {'(order violation)' if not h.is_linear_extension(seq) else ''}")

print("\n== a 'localize_and_fix' skill as a local poset ==")
g = Poset(["SEARCH","INSPECT","EDIT","RUN"], [("SEARCH","INSPECT"),("INSPECT","EDIT"),("EDIT","RUN")])
print("  logp(SEARCH-INSPECT-EDIT-RUN) =", round(frontier_softmax_logp(g, ["SEARCH","INSPECT","EDIT","RUN"], beta=2.0),3))
print("  logp(EDIT-RUN-INSPECT-SEARCH) =", round(frontier_softmax_logp(g, ["EDIT","RUN","INSPECT","SEARCH"], beta=2.0),3), "(violates order)")

print("\n== new-skill penalty: p(open a NEW skill) as the library fills, alpha=0.5 ==")
counts = []
for step, used in enumerate(["A","A","B","A","C","B","A"], 1):
    _, pnew = crp_predictive(counts or [0.0001], 0.5)
    print(f"  before step {step}: library={counts or '∅'}  p(new)={pnew:.3f}")
    # update counts (reuse or open)
    i = {"A":0,"B":1,"C":2}[used]
    while len(counts) <= i: counts.append(0)
    counts[i]+=1
print("  final library sizes:", counts)
for a in (0.2,0.5,1.0,2.0):
    _, pnew = crp_predictive([6,3,1], a)
    print(f"  alpha={a}: p(new | library [6,3,1]) = {pnew:.3f}   (smaller alpha -> more reuse)")
