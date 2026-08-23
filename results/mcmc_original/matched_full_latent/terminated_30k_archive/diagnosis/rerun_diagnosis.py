"""Read-only FULL-LATENT 30k gate diagnosis, reproducible from the archived checkpoints.

Reads ONLY results/.../terminated_30k_archive/checkpoints_30k/*.npz.
Imports ONLY the registered diagnostic functions. No sampler is constructed, no
truth/generator/recovery module is imported, no threshold is altered.

Usage:  PYTHONPATH=src .venv/bin/python <this file>
"""
import sys, os, json, math, hashlib
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hpop.mcmc_original.stage6b_mcmc_diagnostics import (
    bulk_ess, rank_normalized_split_rhat, tail_ess)

CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints_30k")
BURN, THIN, RHAT_GATE = 10000, 5, 1.01
ESS_FLOORS = {"log_target_bulk":1000.0,"log_target_tail":500.0,
              "total_relations_bulk":1000.0,"remaining_invariant_bulk":500.0}
SUMMARY_KEYS = ("log_target","total_relations","sorted_relation_counts","total_segments",
  "mean_segments_per_trace","mean_segment_length","sd_segment_length","boundary_probes",
  "coskill_probes","same_segment_probes","sorted_pi","pi_entropy","pi_l2","P_frobenius",
  "P_trace2","P_trace3","sorted_P_row_entropy","sorted_stationary")

def load(arm):
    out=[]
    for i in range(4):
        d=np.load(os.path.join(CKPT,f"full_{arm}_{i}.npz"), allow_pickle=True)
        rec={k[len("summary__"):]:d[k] for k in d.files if k.startswith("summary__")}
        rec["_rel"]=np.asarray(d["relation_indicators"],dtype=bool)
        rec["_meta"]=json.loads(str(d["meta"]))
        out.append(rec)
    return out

def sweep_of_draw(i): return BURN+(i+1)*THIN

def diag(series):
    ch=np.asarray(series,dtype=float)
    const=[bool(np.all(c==c[0])) for c in ch]
    if all(const):
        vals={float(c[0]) for c in ch}
        if len(vals)==1: return {"rhat":1.0,"bulk_ess":float(ch.size),"tail_ess":float(ch.size),"degenerate":"constant"}
        return {"rhat":float("inf"),"bulk_ess":0.0,"tail_ess":0.0,"degenerate":"constant-but-unequal"}
    return {"rhat":float(rank_normalized_split_rhat(ch)["rhat"]),"bulk_ess":float(bulk_ess(ch)),
            "tail_ess":float(tail_ess(ch)),"degenerate":None}

def components(A, upto=None):
    out={}
    for name in SUMMARY_KEYS:
        vals=[np.asarray(a[name])[:upto] for a in A]
        shape=vals[0].shape[1:]; n=int(np.prod(shape)) if shape else 1
        for c in range(n):
            if shape:
                idx=np.unravel_index(c,shape)
                out[f"{name}[{','.join(map(str,idx))}]"]=diag([v[(slice(None),)+idx] for v in vals])
            else:
                out[name]=diag(vals)
    return out

def libraries(rel):
    """Canonical UNORDERED library hash per draw: sort the 3 skills' 20-bit relation
    vectors, so the result is invariant to skill relabelling. Truth is never consulted."""
    N=rel.shape[0]; r=rel.reshape(N,3,20)
    return np.array([hashlib.sha1(repr(sorted(tuple(int(b) for b in r[t,k]) for k in range(3))).encode()).hexdigest()[:8]
                     for t in range(N)])

report={"burn_in":BURN,"thin":THIN,"rhat_gate":RHAT_GATE,"ess_floors":ESS_FLOORS,"arms":{}}
for arm in ("cond","marg"):
    A=load(arm); S=components(A)
    lt,tr=S["log_target"],S["total_relations"]
    aux=[v["bulk_ess"] for k,v in S.items() if k not in ("log_target","total_relations")]
    finite=(all(math.isfinite(v[m]) for v in S.values() for m in ("rhat","bulk_ess"))
            and math.isfinite(lt["tail_ess"]))
    h=[a["_meta"]["structural"]["h_accepts"] for a in A]
    mok=[a["_meta"]["structural"]["marginal_attempts"]==a["_meta"]["structural"]["ffbs_after_marginal"] for a in A] if arm=="marg" else [True]*4
    maxr=max(v["rhat"] for v in S.values())
    crit={"all_diagnostics_finite":finite,"max_rhat<=1.01":maxr<=RHAT_GATE,
          "log_target_bulk_ess>=1000":lt["bulk_ess"]>=1000,"log_target_tail_ess>=500":lt["tail_ess"]>=500,
          "total_relations_bulk_ess>=1000":tr["bulk_ess"]>=1000,
          "min_remaining_invariant_bulk_ess>=500":min(aux)>=500,
          "chains_with_zero_H_changes==0":sum(x==0 for x in h)==0,
          "marginal_attempts==ffbs_refreshes":all(mok)}
    # canonical library state
    libs=[libraries(a["_rel"]) for a in A]
    per=[]
    for i,hh in enumerate(libs):
        chg=np.nonzero(hh[1:]!=hh[:-1])[0]+1
        per.append({"chain":i,"current":str(hh[-1]),"transitions_after_burn_in":int(len(chg)),
                    "last_transition_sweep":int(sweep_of_draw(int(chg[-1]))) if len(chg) else None,
                    "distinct_libraries_visited":int(len(set(hh)))})
    same=len({p["current"] for p in per})==1
    consensus=None
    if same:
        agree=np.array([len({h_[t] for h_ in libs})==1 for t in range(len(libs[0]))])
        t=len(agree)-1
        while t>0 and agree[t-1]: t-=1
        consensus={"unbroken_since_draw":int(t),"unbroken_since_sweep":int(sweep_of_draw(t)),
                   "fraction_of_window":float((len(agree)-t)/len(agree))}
    report["arms"][arm]={
        "verdict":"PASS" if all(crit.values()) else "FAIL",
        "max_rhat":(None if not math.isfinite(maxr) else maxr),
        "max_rhat_is_infinite":not math.isfinite(maxr),
        "max_rhat_variable":max(S,key=lambda k:S[k]["rhat"] if math.isfinite(S[k]["rhat"]) else 9e99),
        "log_target_bulk_ess":lt["bulk_ess"],"log_target_tail_ess":lt["tail_ess"],
        "total_relations_bulk_ess":tr["bulk_ess"],"min_remaining_invariant_bulk_ess":min(aux),
        "accepted_H_changes_per_chain":h,"failed_criteria":[k for k,v in crit.items() if not v],
        "n_summaries_violating":sum(1 for v in S.values() if not math.isfinite(v["rhat"]) or v["rhat"]>1.01 or v["bulk_ess"]<500),
        "n_summaries_total":len(S),
        "canonical_unordered_library":{"per_chain":per,"all_chains_same_library":same,"consensus":consensus},
    }
    if arm=="marg" and consensus:
        d0=consensus["unbroken_since_draw"]
        post={}
        for k in ("total_relations","sorted_relation_counts[1]","sorted_relation_counts[2]","log_target"):
            if "[" in k:
                nm,ix=k[:-1].split("["); ix=tuple(int(x) for x in ix.split(","))
                s=[np.asarray(a[nm])[(slice(None),)+ix].astype(float) for a in A]
            else:
                s=[np.asarray(a[k]).astype(float) for a in A]
            full=diag(s); pst=diag([x[d0:] for x in s])
            cf={}
            for sw in (30000,50000,75000,100000):
                n=(sw-BURN)//THIN; extra=n-len(s[0])
                ext=[np.concatenate([x,np.full(extra,x[-1])]) if extra>0 else x[:n] for x in s]
                e=diag(ext); cf[str(sw)]={"rhat":e["rhat"],"bulk_ess":e["bulk_ess"]}
            post[k]={"rhat_full_window":full["rhat"],"rhat_post_transient":pst["rhat"],
                     "bulk_ess_full_window":full["bulk_ess"],"bulk_ess_post_transient":pst["bulk_ess"],
                     "counterfactual_frozen_extension":cf}
        report["arms"][arm]["post_transient_and_counterfactual"]=post
        report["arms"][arm]["counterfactual_note"]=(
            "Frozen extension is an EXPLICIT ASSUMPTION (each chain holds its current value), "
            "not observed data. It is applied only to discrete structural invariants that have in "
            "fact been constant and equal across all four chains since the consensus sweep.")

out=os.path.join(os.path.dirname(__file__),"gate_diagnosis_30k.json")
with open(out,"w") as fh: json.dump(report,fh,indent=2,sort_keys=True,default=str)
print(json.dumps(report["arms"]["cond"]["canonical_unordered_library"],indent=1)[:400])
print("WROTE",out)
