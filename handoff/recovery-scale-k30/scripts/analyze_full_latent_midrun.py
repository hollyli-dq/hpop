#!/usr/bin/env python
"""READ-ONLY mid-run truth-recovery analysis of the live FULL-LATENT experiment.

MID-RUN EXPLORATORY DIAGNOSTIC — NOT A REGISTERED CONVERGENCE GATE.

The generating truth was unsealed mid-run at the principal investigator's explicit
instruction (see TRUTH_UNSEAL_midrun.json).  No formal experimental setting was
subsequently changed.

This script only READS durable checkpoint artifacts and the frozen corpus.  It imports
no source file that the live workers use: the registered R-hat/ESS and probe-selection
definitions are vendored verbatim in scripts/_midrun_vendored_registered.py.
"""
from __future__ import annotations
import json, hashlib, math, subprocess, datetime, itertools, sys
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _midrun_vendored_registered as V

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT/"results/mcmc_original/matched_full_latent"
CHAIN = BASE/"formal_chains"
CORP  = ROOT/"results/mcmc_original/matched_synthetic_formal_corpus"
OUT   = BASE/"readonly_midrun_truth_analysis"
OUT.mkdir(parents=True, exist_ok=True)
ARMS = ("FULL-COND","FULL-MARG"); TAG = {"FULL-COND":"full_cond","FULL-MARG":"full_marg"}
ROLE = "ABCDE"; K = 3; M = 5
OFF = [(i,j) for i in range(M) for j in range(M) if i!=j]          # 20 ordered pairs
RHAT_GATE = 1.01
BANNER = "MID-RUN EXPLORATORY DIAGNOSTIC — NOT A REGISTERED CONVERGENCE GATE"

# ── house style ──────────────────────────────────────────────────────────────
SURFACE="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8b8a85"; GRID="#e4e3df"
CHAIN_C=["#2a78d6","#eb6834","#1baf7a","#4a3aa7"]     # validated all-pairs (light)
ARM_C={"FULL-COND":"#eb6834","FULL-MARG":"#2a78d6"}
SKILL_C=["#2a78d6","#eb6834","#1baf7a"]
RED="#e34948"
plt.rcParams.update({
 "figure.facecolor":SURFACE,"axes.facecolor":SURFACE,"savefig.facecolor":SURFACE,
 "axes.edgecolor":GRID,"axes.labelcolor":INK2,"text.color":INK,"xtick.color":INK2,
 "ytick.color":INK2,"grid.color":GRID,"grid.linewidth":0.6,"axes.grid":True,
 "axes.spines.top":False,"axes.spines.right":False,"font.size":8.5,
 "axes.titlesize":9.5,"axes.titleweight":"bold","legend.frameon":False,
 "lines.linewidth":0.9,"pdf.fonttype":42,"ps.fonttype":42,"figure.dpi":110})

def save(fig, stem, note=None):
    fig.text(0.5,-0.004,(note+"   ·   " if note else "")+BANNER,ha="center",va="top",
             fontsize=7,color=MUTED)
    for ext in ("png","pdf"):
        fig.savefig(OUT/f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig); print("  wrote", stem)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*a): return subprocess.run(["git",*a],cwd=ROOT,capture_output=True,text=True).stdout.strip()

# ── 1. live state ────────────────────────────────────────────────────────────
def load_chains():
    data={}
    for arm in ARMS:
        rows=[]
        for c in range(4):
            z=np.load(CHAIN/f"{TAG[arm]}_{c}.npz",allow_pickle=True)
            m=z["meta"].item(); m=json.loads(m) if isinstance(m,(bytes,str)) else m
            r={k[len("summary__"):]:np.asarray(z[k]) for k in z.keys() if k.startswith("summary__")}
            r["pi"]=np.asarray(z["pi_draws"]); r["P"]=np.asarray(z["p_draws"])
            r["rel"]=np.asarray(z["relation_indicators"]).reshape(-1,K,len(OFF))
            r["bnd"]=[np.asarray(z[f"boundary__{n:03d}"]) for n in range(100)]
            r["cosk"]=np.asarray(z["recovery_coskill_sums"])
            r["meta"]=m; r["n"]=int(m["retained_draws"])
            r["sweep"]=m["burn_in"]+np.arange(1,r["n"]+1)*m["thin"]
            r["total_sweep"]=int(m["burn_in"]+r["n"]*m["thin"])
            rows.append(r)
        data[arm]=rows
    return data

# ── 2. truth ─────────────────────────────────────────────────────────────────
def load_truth():
    T=json.loads((CORP/"truth_manifest.json").read_text())
    t=T["truth"]
    H=[np.array(h,int) for h in t["h_by_skill"]]
    vec=np.array([[H[k][i,j] for (i,j) in OFF] for k in range(K)],bool)
    d=np.load(CORP/"train_traces.npz",allow_pickle=False)
    n=int(np.asarray(d["n_traces"])[0])
    traces=[np.asarray(d[f"t{i:03d}_cpa"],int) for i in range(n)]
    widths=[np.asarray(d[f"t{i:03d}_widths"],int) for i in range(n)]
    labels=[np.asarray(d[f"t{i:03d}_labels"],int) for i in range(n)]
    bnd=[]; occ=[]
    for i in range(n):
        J=len(traces[i]); b=np.zeros(J-1,bool)
        b[np.cumsum(widths[i])[:-1]-1]=True; bnd.append(b)
        occ.append(np.repeat(labels[i],widths[i]))       # occurrence-level true skill
    return dict(H=H,vec=vec,pi=np.array(t["pi"]),P=np.array(t["transition"]),
                traces=traces,widths=widths,labels=labels,boundary=bnd,occ=occ,
                h_hashes=t["h_hashes"],truth_hash=T["truth_hash_sha256"],
                corpus_hash=T["corpus_hash_sha256"])

# ── 3. global label alignment (brute force over all 3! permutations) ─────────
PERMS=list(itertools.permutations(range(K)))
def align(rel_draw, truth_vec):
    """One global assignment; cost = closure Hamming.  Deterministic tie-break =
    lexicographically smallest permutation among the minimisers."""
    cost=np.array([[int((rel_draw[k]!=truth_vec[j]).sum()) for j in range(K)] for k in range(K)])
    best=None
    for p in PERMS:                       # p[k] = true skill matched to learned skill k
        tot=sum(cost[k][p[k]] for k in range(K))
        if best is None or tot<best[0]: best=(tot,p)
    return best[1], best[0], cost
def canon(rel_draw):
    """Permutation-invariant canonical library: lexicographically sorted closure rows."""
    rows=sorted(tuple(int(b) for b in r) for r in rel_draw)
    return bytes(bytearray(sum(rows,())))

# ── metric helpers ───────────────────────────────────────────────────────────
def f1_from(al, tv):
    tp=int((al&tv).sum()); fp=int((al&~tv).sum()); fn=int((~al&tv).sum())
    return (2*tp/(2*tp+fp+fn)) if tp else 0.0
def tv_dist(a,b): return float(0.5*np.abs(a-b).sum())
def diag(series):
    """Registered rank-normalized split R-hat / ESS, plus a frozen-chain degeneracy flag.

    When some chains are EXACTLY constant at different values the rank-normalized
    statistic diverges numerically.  That is real information (the chains are stuck),
    but the printed magnitude is meaningless, so it is flagged rather than reported
    as if it were a comparable number."""
    ch=np.asarray(series,float)
    frozen=int(sum(1 for row in ch if np.all(row==row[0])))
    if np.all(ch==ch.flat[0]):
        # every chain constant at the SAME value: perfect agreement, not a pathology
        return {"rhat":1.0,"bulk_ess":float(ch.size),"tail_ess":float(ch.size),
                "frozen_chains":frozen,"degenerate":False,"all_constant_equal":True}
    # any exactly-constant chain alongside disagreement makes the rank statistic unreliable
    return {"rhat":float(V.rank_normalized_split_rhat(ch)["rhat"]),
            "bulk_ess":float(V.bulk_ess(ch)),"tail_ess":float(V.tail_ess(ch)),
            "frozen_chains":frozen,"degenerate":bool(frozen>0),"all_constant_equal":False}

# ═════════════════════════════════════════════════════════════════════════════
def main():
    V.verify_sources(ROOT)
    print("== live state ==")
    D=load_chains(); T=load_truth()
    live=subprocess.run(["bash","-lc","ps ax -o pid,command | grep -c '[m]ultiprocessing-fork'"],
                        capture_output=True,text=True).stdout.strip()
    state={"git_commit":git("rev-parse","HEAD"),
           "sampler_sources_modified":bool(git("status","--short","src/hpop/mcmc_original",
                                               "scripts/run_matched_full_latent_formal.py")),
           "live_worker_processes":int(live or 0),
           "analysed_at_utc":datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00","Z"),
           "chains":{}}
    for arm in ARMS:
        state["chains"][arm]=[{"chain":c,"sweep":r["total_sweep"],"retained":r["n"],
            "hours":round(r["meta"]["seconds"]/3600,3),
            "sec_per_sweep":round(r["meta"]["seconds"]/r["total_sweep"],4),
            "checkpoint_sha256":sha(CHAIN/f"{TAG[arm]}_{c}.npz"),
            "structural":r["meta"]["structural"]} for c,r in enumerate(D[arm])]
        print(f"  {arm}: sweeps={[x['sweep'] for x in state['chains'][arm]]} "
              f"retained={[x['retained'] for x in state['chains'][arm]]}")
    print(f"  workers={state['live_worker_processes']}  sampler modified={state['sampler_sources_modified']}")

    # ── per-draw alignment & recovery ────────────────────────────────────────
    print("== aligning draws ==")
    REC={}
    for arm in ARMS:
        rows=[]
        for c,r in enumerate(D[arm]):
            rel=r["rel"]; n=len(rel)
            sig=np.zeros((n,K),int); ham=np.zeros(n,int); f1=np.zeros(n)
            piA=np.zeros((n,K)); PA=np.zeros((n,K,K)); cid=[]
            for t in range(n):
                p,h,_=align(rel[t],T["vec"]); sig[t]=p; ham[t]=h
                inv=np.argsort(p)                     # true skill j <- learned inv[j]
                al=rel[t][inv]                        # learned rows in TRUE skill order
                f1[t]=f1_from(al,T["vec"])
                piA[t]=r["pi"][t][inv]; PA[t]=r["P"][t][np.ix_(inv,inv)]
                cid.append(canon(rel[t]))
            rows.append(dict(sigma=sig,hamming=ham,f1=f1,pi=piA,P=PA,canon=cid))
        REC[arm]=rows
    # cross-check the brute-force assignment against scipy on a sample
    from scipy.optimize import linear_sum_assignment
    for arm in ARMS:
        r=D[arm][0]["rel"]
        for t in range(0,len(r),max(1,len(r)//50)):
            p,h,cost=align(r[t],T["vec"]); _,cc=linear_sum_assignment(cost)
            assert sum(cost[k][p[k]] for k in range(K))==int(cost[range(K),cc].sum()), "assignment mismatch"
    print("  brute-force 3! assignment verified against linear_sum_assignment")

    truth_canon=canon(T["vec"])
    metrics={"banner":BANNER,"live_state":state,
             "truth":{"K":K,"n_roles":M,"pi_star":T["pi"].tolist(),"P_star":T["P"].tolist(),
                      "relations_per_skill":[int(h.sum()) for h in T["H"]],
                      "sorted_relation_counts":sorted(int(h.sum()) for h in T["H"]),
                      "total_relations":int(sum(int(h.sum()) for h in T["H"])),
                      "h_hashes":T["h_hashes"],"truth_hash":T["truth_hash"],
                      "corpus_hash":T["corpus_hash"],
                      "canonical_library_sha256":hashlib.sha256(truth_canon).hexdigest()},
             "arms":{}}

    # ── per-arm aggregate recovery ───────────────────────────────────────────
    SUM_KEYS=["log_target","total_relations","total_segments","mean_segments_per_trace",
              "mean_segment_length","sd_segment_length","pi_entropy","pi_l2",
              "P_frobenius","P_trace2","P_trace3"]
    VEC_KEYS=["sorted_relation_counts","sorted_pi","sorted_P_row_entropy","sorted_stationary"]
    probes=V.select_truth_free_probes(T["traces"],T["corpus_hash"],32,64,256)["recovery_coskill"]
    true_cosk=np.array([T["occ"][n][i]==T["occ"][n][j] for (n,i,j) in probes],bool)

    for arm in ARMS:
        chains=D[arm]; rec=REC[arm]; nmin=min(r["n"] for r in chains)
        # invariant diagnostics (registered definitions, current draws)
        dg={}
        for k in SUM_KEYS: dg[k]=diag([r[k][:nmin] for r in chains])
        for k in VEC_KEYS:
            for j in range(chains[0][k].shape[1]):
                dg[f"{k}[{j}]"]=diag([r[k][:nmin,j] for r in chains])
        # library states
        libs=[]
        for c in range(4):
            cnt=Counter(rec[c]["canon"]); dom,dn=cnt.most_common(1)[0]
            tail=Counter(rec[c]["canon"][-400:]); tdom,tn=tail.most_common(1)[0]
            libs.append({"chain":c,"distinct_states":len(cnt),
                "dominant_sha256":hashlib.sha256(dom).hexdigest()[:16],
                "dominant_occupancy":round(dn/len(rec[c]["canon"]),4),
                "dominant_equals_truth":bool(dom==truth_canon),
                "tail400_dominant_equals_truth":bool(tdom==truth_canon),
                "tail400_occupancy":round(tn/min(400,len(rec[c]["canon"])),4),
                "exact_library_draw_fraction":round(float(np.mean([x==truth_canon for x in rec[c]["canon"]])),4),
                "tail400_exact_fraction":round(float(np.mean([x==truth_canon for x in rec[c]["canon"][-400:]])),4),
                "mean_closure_hamming":round(float(rec[c]["hamming"].mean()),4),
                "tail400_closure_hamming":round(float(rec[c]["hamming"][-400:].mean()),4),
                "mean_closure_f1":round(float(rec[c]["f1"].mean()),4),
                "tail400_closure_f1":round(float(rec[c]["f1"][-400:].mean()),4)})
        # boundary recovery, all 100 training traces
        pb=[]; tb=[]
        for c in range(4):
            p=np.concatenate([chains[c]["bnd"][n]/chains[c]["n"] for n in range(100)])
            pb.append(p)
        pooled=np.mean(pb,axis=0); tb=np.concatenate([T["boundary"][n] for n in range(100)]).astype(float)
        brier_per_chain=[float(np.mean((p-tb)**2)) for p in pb]
        # co-skill recovery on the 256 registered recovery probes
        ck=[chains[c]["cosk"]/chains[c]["n"] for c in range(4)]
        ck_pool=np.mean(ck,axis=0)
        cbrier=[float(np.mean((c_-true_cosk)**2)) for c_ in ck]
        # pi / P recovery under the SAME per-draw structural alignment
        piA=np.concatenate([rec[c]["pi"] for c in range(4)]); piM=piA.mean(0)
        PAA=np.concatenate([rec[c]["P"] for c in range(4)]); PM=PAA.mean(0)
        offmask=~np.eye(K,dtype=bool)
        metrics["arms"][arm]={
          "sweeps":[r["total_sweep"] for r in chains],
          "retained":[r["n"] for r in chains],
          "sec_per_sweep":round(float(np.mean([r["meta"]["seconds"]/r["total_sweep"] for r in chains])),4),
          "sweeps_per_hour":round(float(np.mean([r["total_sweep"]/(r["meta"]["seconds"]/3600) for r in chains])),1),
          "diagnostics":{k:{**{m:(None if not math.isfinite(v[m]) else round(v[m],4)) for m in ("rhat","bulk_ess","tail_ess")},
                             "frozen_chains":v["frozen_chains"],"degenerate":v["degenerate"],
                             "all_constant_equal":v.get("all_constant_equal",False)} for k,v in dg.items()},
          "max_invariant_rhat":round(max(v["rhat"] for v in dg.values() if math.isfinite(v["rhat"])),4),
          "max_invariant_rhat_excluding_frozen_degenerate":round(max(
              [v["rhat"] for v in dg.values() if math.isfinite(v["rhat"]) and not v["degenerate"]] or [float("nan")]),4),
          "degenerate_frozen_invariants":{k:v["frozen_chains"] for k,v in dg.items() if v["degenerate"]},
          "library":libs,
          "all_four_chains_same_library":len({l["tail400_dominant_equals_truth"] for l in libs})==1 and all(l["tail400_dominant_equals_truth"] for l in libs),
          "boundary_brier_per_chain":[round(b,6) for b in brier_per_chain],
          "boundary_brier_pooled":round(float(np.mean((pooled-tb)**2)),6),
          "coskill_brier_per_chain":[round(b,6) for b in cbrier],
          "coskill_brier_pooled":round(float(np.mean((ck_pool-true_cosk)**2)),6),
          "coskill_mae_pooled":round(float(np.mean(np.abs(ck_pool-true_cosk))),6),
          "pi_posterior_mean_aligned":[round(float(v),4) for v in piM],
          "pi_L1":round(float(np.abs(piM-T["pi"]).sum()),4),
          "pi_TV":round(tv_dist(piM,T["pi"]),4),
          "pi_RMSE":round(float(np.sqrt(np.mean((piM-T["pi"])**2))),4),
          "P_posterior_mean_aligned":[[round(float(v),4) for v in row] for row in PM],
          "P_offdiag_RMSE":round(float(np.sqrt(np.mean((PM-T["P"])[offmask]**2))),4),
          "P_frobenius_error":round(float(np.linalg.norm(PM-T["P"])),4),
          "P_row_TV":[round(tv_dist(PM[i],T["P"][i]),4) for i in range(K)],
        }
        metrics["arms"][arm]["_cache"]={}
        D[arm+"_cache"]={"pb":pb,"pooled":pooled,"tb":tb,"ck":ck,"ck_pool":ck_pool,
                          "true_cosk":true_cosk,"piM":piM,"PM":PM,"dg":dg,"libs":libs,
                          "nmin":nmin,"piA":piA}
        print(f"  {arm}: maxRhat={metrics['arms'][arm]['max_invariant_rhat']:.3f} "
              f"bBrier={metrics['arms'][arm]['boundary_brier_pooled']:.4f} "
              f"cBrier={metrics['arms'][arm]['coskill_brier_pooled']:.4f} "
              f"piTV={metrics['arms'][arm]['pi_TV']:.4f} PF={metrics['arms'][arm]['P_frobenius_error']:.4f}")
    for arm in ARMS: metrics["arms"][arm].pop("_cache",None)

    print("== figures ==")
    TITLE=lambda s:(s+"\n"+BANNER)
    # ── FIGURE A ─────────────────────────────────────────────────────────────
    ROWS=[("log_target","log target"),("total_relations","total closure relations"),
          ("sorted_relation_counts","sorted closure-relation counts"),
          ("total_segments","total inferred segments"),("mean_segment_length","mean segment length"),
          ("pi_entropy","entropy(pi)"),("P_frobenius","||P||_F")]
    fig,axes=plt.subplots(len(ROWS),2,figsize=(11.5,15),sharex="col")
    for r,(key,lab) in enumerate(ROWS):
        lim=[]
        for c,arm in enumerate(ARMS):
            ax=axes[r,c]
            for i,ch in enumerate(D[arm]):
                y=ch[key]
                if y.ndim==2:
                    for j in range(y.shape[1]): ax.plot(ch["sweep"],y[:,j],color=SKILL_C[j],alpha=0.55,lw=0.7)
                    lim+=[y.min(),y.max()]
                else:
                    ax.plot(ch["sweep"],y,color=CHAIN_C[i],alpha=0.85); lim+=[y.min(),y.max()]
            ax.axvline(10000,color=MUTED,lw=0.8,ls=":")
            if r==0: ax.set_title(arm,color=ARM_C[arm],fontsize=11)
            if c==0: ax.set_ylabel(lab,fontsize=8)
            if r==len(ROWS)-1: ax.set_xlabel("sweep")
        lo,hi=min(lim),max(lim); pad=0.05*(hi-lo+1e-9)
        for c in (0,1): axes[r,c].set_ylim(lo-pad,hi+pad)
    fig.legend(handles=[Line2D([],[],color=CHAIN_C[i],lw=2,label=f"chain {i}") for i in range(4)]
        +[Line2D([],[],color=SKILL_C[j],lw=2,ls="-",label=f"sorted component {j+1}") for j in range(3)]
        +[Line2D([],[],color=MUTED,lw=0.8,ls=":",label="burn-in ends (10,000)")],
        loc="upper center",ncol=4,bbox_to_anchor=(0.5,1.0))
    fig.suptitle(TITLE("FULL-LATENT — permutation-invariant posterior summaries, 4 paired chains per arm"),
                 y=1.038,fontsize=12)
    fig.tight_layout(rect=[0,0.008,1,0.995])
    save(fig,"fig_A_invariant_traces","rows share a y-scale across arms")

    # ── FIGURE B ─────────────────────────────────────────────────────────────
    BKEYS=SUM_KEYS+["sorted_relation_counts[0]","sorted_relation_counts[1]","sorted_relation_counts[2]"]
    def series(arm,key,upto):
        ch=D[arm]
        if "[" in key:
            base,idx=key[:-1].split("["); idx=int(idx)
            return [c[base][:upto,idx] for c in ch]
        return [c[key][:upto] for c in ch]
    fig,axes=plt.subplots(4,4,figsize=(13.5,10.5),sharex=True)
    for a,key in enumerate(BKEYS):
        ax=axes.flat[a]
        for arm in ARMS:
            nm=min(c["n"] for c in D[arm]); xs=[];ys=[]
            for m in range(60,nm+1,max(1,nm//40)):
                try: rh=diag(series(arm,key,m))["rhat"]
                except Exception: continue
                if math.isfinite(rh): xs.append(D[arm][0]["sweep"][m-1]); ys.append(rh)
            ax.plot(xs,ys,color=ARM_C[arm])
        ax.axhline(RHAT_GATE,color=RED,lw=1.0,ls="--")
        ax.set_title(key,fontsize=8); ax.set_yscale("log"); ax.set_ylim(0.995,4.0)
        ax.set_yticks([1.0,1.1,1.5,2.0,3.0]); ax.set_yticklabels(["1.00","1.10","1.50","2.00","3.00"])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for a in range(len(BKEYS),16): axes.flat[a].axis("off")
    for ax in axes[-1]: ax.set_xlabel("sweep")
    fig.legend(handles=[Line2D([],[],color=ARM_C[a],lw=2,label=a) for a in ARMS]
        +[Line2D([],[],color=RED,lw=1,ls="--",label=f"registered threshold R-hat = {RHAT_GATE} (reference only)")],
        loc="upper center",ncol=3,bbox_to_anchor=(0.5,1.0))
    fig.suptitle(TITLE("FULL-LATENT — expanding-window rank-normalized split R-hat on each registered invariant"),
                 y=1.045,fontsize=12)
    fig.tight_layout(rect=[0,0.008,1,0.995])
    save(fig,"fig_B_running_rhat","crossing the dashed line is NOT a registered PASS; gates fire only at 30k/50k/75k/100k")

    # ── FIGURE C ─────────────────────────────────────────────────────────────
    allc=Counter()
    for arm in ARMS:
        for c in range(4): allc.update(REC[arm][c]["canon"])
    order=[s for s,_ in allc.most_common()]
    if truth_canon in order: order.insert(0,order.pop(order.index(truth_canon)))
    idx={s:i for i,s in enumerate(order)}
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.4),sharey=True)
    for c_,arm in enumerate(ARMS):
        ax=axes[c_]
        for i in range(4):
            ax.step(D[arm][i]["sweep"],[idx[s] for s in REC[arm][i]["canon"]],where="post",
                    color=CHAIN_C[i],alpha=0.85,lw=1.1)
        ax.axhline(0,color=RED,lw=1.2,ls="--")
        ax.set_title(arm,color=ARM_C[arm]); ax.set_xlabel("sweep")
        ax.set_ylim(-0.6,min(len(order),18)-0.4); ax.invert_yaxis()
    axes[0].set_ylabel("canonical unordered-library state ID\n(0 = TRUE library)")
    fig.legend(handles=[Line2D([],[],color=CHAIN_C[i],lw=2,label=f"chain {i}") for i in range(4)]
        +[Line2D([],[],color=RED,lw=1.2,ls="--",label="state 0 = exact TRUE closure library")],
        loc="upper center",ncol=5,bbox_to_anchor=(0.5,1.0))
    fig.suptitle(TITLE("FULL-LATENT — exact canonical unordered-library state visited over time"),
                 y=1.10,fontsize=12)
    fig.tight_layout(rect=[0,0.01,1,0.99])
    save(fig,"fig_C_exact_library_states","state = lexicographically sorted closure bit-vectors; identical ID means identical library")

    # ── FIGURE D ─────────────────────────────────────────────────────────────
    def draw_poset(ax,h,color,title,sub):
        red=h.copy()
        for i in range(M):
            for j in range(M):
                if h[i,j] and any(h[i,k] and h[k,j] for k in range(M)): red[i,j]=0
        lv=np.zeros(M,int)
        for _ in range(M):
            for i in range(M):
                for j in range(M):
                    if h[i,j]: lv[j]=max(lv[j],lv[i]+1)
        pos={}
        for L in range(lv.max()+1):
            mem=[i for i in range(M) if lv[i]==L]
            for m_,i in enumerate(mem): pos[i]=(m_-(len(mem)-1)/2.0,-L)
        for i in range(M):
            for j in range(M):
                if red[i,j]: ax.add_patch(FancyArrowPatch(pos[i],pos[j],arrowstyle="-|>",
                    mutation_scale=11,shrinkA=17,shrinkB=17,color=INK2,lw=1.3,zorder=1))
        for i,(x,y) in pos.items():
            iso=h[i].sum()==0 and h[:,i].sum()==0
            ax.add_patch(Circle((x,y),0.27,facecolor=SURFACE,edgecolor=MUTED if iso else color,lw=2.1,zorder=3))
            ax.text(x,y,ROLE[i],ha="center",va="center",fontsize=10,zorder=4,color=MUTED if iso else INK)
        ax.set_xlim(-2.1,2.1); ax.set_ylim(-3.7,0.6)   # fixed across panels: equal aspect -> equal heights
        ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_title(title,color=color,fontsize=9.5)
        ax.text(0.5,-0.03,sub,ha="center",va="top",fontsize=7.4,color=MUTED,transform=ax.transAxes)
    def vec2mat(v):
        h=np.zeros((M,M),int)
        for b,(i,j) in zip(v,OFF): h[i,j]=int(b)
        return h
    fig,axes=plt.subplots(3,3,figsize=(11.5,11))
    for k in range(K):
        draw_poset(axes[0,k],T["H"][k],SKILL_C[k],f"TRUE  H*_{k}",
                   f"{int(T['H'][k].sum())} closure relations")
    for r_,arm in enumerate(ARMS):
        dom=Counter(sum([REC[arm][c]["canon"][-400:] for c in range(4)],[])).most_common(1)[0][0]
        # recover a representative aligned draw realising the dominant canonical library
        rep=None
        for c in range(4):
            for t in range(len(REC[arm][c]["canon"])-1,-1,-1):
                if REC[arm][c]["canon"][t]==dom:
                    p=REC[arm][c]["sigma"][t]; inv=np.argsort(p)
                    rep=D[arm][c]["rel"][t][inv]; break
            if rep is not None: break
        for k in range(K):
            h=vec2mat(rep[k]); tvk=T["vec"][k]
            hm=int((rep[k]!=tvk).sum()); f1=f1_from(rep[k].astype(bool),tvk)
            ok="EXACT" if hm==0 else "not exact"
            draw_poset(axes[r_+1,k],h,ARM_C[arm],f"{arm}  ->  matched to H*_{k}",
                       f"{int(h.sum())} rel · F1 {f1:.3f} · Hamming {hm} · {ok}")
    fig.suptitle(TITLE("FULL-LATENT — true reusable partial orders vs the dominant aligned posterior library"),
                 y=1.0,fontsize=12)
    fig.tight_layout(rect=[0,0.01,1,0.965])
    save(fig,"fig_D_truth_vs_learned_posets",
         "dominant canonical library over the last 400 draws (exploratory summary); metrics on transitive closures, diagrams are Hasse reductions")

    # ── FIGURE E ─────────────────────────────────────────────────────────────
    firsts=[]
    for Jt in (24,32,40,48):
        for n in range(100):
            if len(T["traces"][n])==Jt: firsts.append(n); break
    fig,axes=plt.subplots(len(firsts)*2,1,figsize=(13.5,3.3*len(firsts)),
                          gridspec_kw={"height_ratios":[0.72,1.0]*len(firsts),"hspace":0.9})
    ex_rows=[]
    for r_,tid in enumerate(firsts):
        x=T["traces"][tid]; J=len(x); w=T["widths"][tid]; lab=T["labels"][tid]
        ax=axes[2*r_]; ax.set_xlim(-0.5,J-0.5); ax.set_ylim(0,1); ax.axis("off")
        st=0
        for wi,li in zip(w,lab):
            for j in range(st,st+wi):
                ax.add_patch(plt.Rectangle((j-0.45,0.14),0.9,0.74,facecolor=SKILL_C[li],alpha=0.20,
                                           edgecolor=SURFACE,lw=1.2))
                ax.text(j,0.51,ROLE[x[j]],ha="center",va="center",fontsize=7.6,color=INK)
            ax.plot([st-0.5,st+wi-0.5],[0.05,0.05],color=SKILL_C[li],lw=2.6,solid_capstyle="butt")
            st+=wi
        ax.set_title(f"trace {tid} — observed roles with TRUE segmentation/labels (J={J}, {len(w)} true blocks)",
                     loc="left",fontsize=9.5)
        ax=axes[2*r_+1]; ax.set_xlim(-0.5,J-0.5); ax.set_ylim(-0.03,1.06)
        for e in np.cumsum(w)[:-1]: ax.axvline(e-0.5,color=INK,lw=1.3,ls="--",alpha=0.8,zorder=1)
        row={"trace":tid,"J":J,"true_blocks":int(len(w))}
        for arm in ARMS:
            ps=[D[arm][c]["bnd"][tid]/D[arm][c]["n"] for c in range(4)]
            for p in ps: ax.plot(np.arange(J-1)+0.5,p,color=ARM_C[arm],alpha=0.45,lw=0.8,zorder=2)
            pl=np.mean(ps,axis=0)
            ax.plot(np.arange(J-1)+0.5,pl,color=ARM_C[arm],lw=2.2,zorder=3)
            tbi=T["boundary"][tid].astype(float)
            row[f"{arm}_expected_blocks"]=round(float(pl.sum()+1),3)
            row[f"{arm}_boundary_brier"]=round(float(np.mean((pl-tbi)**2)),5)
        ex_rows.append(row)
        ax.set_ylabel("Pr[boundary]",fontsize=8); ax.set_xticks(range(0,J,2))
        if r_==len(firsts)-1: ax.set_xlabel("position j",fontsize=8)
    fig.legend(handles=[Line2D([],[],color=ARM_C[a],lw=2.2,label=f"{a} (thin = 4 chains, thick = pooled)") for a in ARMS]
        +[Line2D([],[],color=INK,lw=1.3,ls="--",label="TRUE boundary")],
        loc="upper center",ncol=3,bbox_to_anchor=(0.5,1.0))
    fig.suptitle(TITLE("FULL-LATENT — posterior segmentation vs truth, first trace of each length class"),
                 y=1.028,fontsize=12)
    save(fig,"fig_E_example_boundaries","trace IDs chosen deterministically (first of each J), not visually")
    metrics["example_traces"]=ex_rows

    # ── FIGURE F ─────────────────────────────────────────────────────────────
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.2))
    ax=axes[0]
    for arm in ARMS:
        ca=D[arm+"_cache"]
        ax.bar([f"{arm}\nch{c}" for c in range(4)],metrics["arms"][arm]["boundary_brier_per_chain"],
               color=ARM_C[arm],width=0.6)
    ax.set_ylabel("boundary Brier (lower better)"); ax.set_title("per-chain boundary Brier, all 100 traces")
    ax.tick_params(axis="x",labelsize=7)
    ax=axes[1]
    bins=np.linspace(0,1,11); ctr=(bins[:-1]+bins[1:])/2
    for arm in ARMS:
        ca=D[arm+"_cache"]; p=ca["pooled"]; t=ca["tb"]
        emp=[t[(p>=bins[i])&(p<bins[i+1])].mean() if ((p>=bins[i])&(p<bins[i+1])).sum()>20 else np.nan
             for i in range(10)]
        ax.plot(ctr,emp,color=ARM_C[arm],marker="o",ms=4,lw=1.6,label=arm)
    ax.plot([0,1],[0,1],color=MUTED,ls=":",lw=1)
    ax.set_xlabel("predicted boundary probability"); ax.set_ylabel("empirical truth frequency")
    ax.set_title("calibration (pooled over chains)"); ax.legend(loc="upper left")
    ax=axes[2]
    for arm in ARMS:
        ca=D[arm+"_cache"]
        ax.bar([f"{arm}\nch{c}" for c in range(4)],metrics["arms"][arm]["coskill_brier_per_chain"],
               color=ARM_C[arm],width=0.6)
    ax.set_ylabel("co-skill Brier (lower better)")
    ax.set_title("per-chain co-skill Brier, 256 registered probe pairs")
    ax.tick_params(axis="x",labelsize=7)
    fig.suptitle(TITLE("FULL-LATENT — corpus-wide boundary and co-clustering recovery"),y=1.06,fontsize=12)
    fig.tight_layout(rect=[0,0.01,1,0.99])
    save(fig,"fig_F_boundary_recovery","boundary metrics use all 100 training traces and every interior position")

    # ── FIGURE G ─────────────────────────────────────────────────────────────
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3))
    ax=axes[0]
    for arm in ARMS:
        ca=D[arm+"_cache"]; c_=ca["ck_pool"]; tc=ca["true_cosk"]
        ax.hist(c_[tc],bins=20,range=(0,1),histtype="step",lw=1.8,color=ARM_C[arm],
                label=f"{arm} · truly same skill")
        ax.hist(c_[~tc],bins=20,range=(0,1),histtype="step",lw=1.0,ls=":",color=ARM_C[arm],
                label=f"{arm} · truly different")
    ax.set_xlabel("posterior Pr[same skill]"); ax.set_ylabel("probe pairs")
    ax.set_title("co-skill posterior separation"); ax.legend(fontsize=7)
    ax=axes[1]
    bins=np.linspace(0,1,11); ctr=(bins[:-1]+bins[1:])/2
    for arm in ARMS:
        ca=D[arm+"_cache"]; p=ca["ck_pool"]; t=ca["true_cosk"].astype(float)
        emp=[t[(p>=bins[i])&(p<bins[i+1])].mean() if ((p>=bins[i])&(p<bins[i+1])).sum()>=5 else np.nan
             for i in range(10)]
        ax.plot(ctr,emp,color=ARM_C[arm],marker="o",ms=4,lw=1.6,label=arm)
    ax.plot([0,1],[0,1],color=MUTED,ls=":",lw=1)
    ax.set_xlabel("predicted Pr[same skill]"); ax.set_ylabel("empirical truth frequency")
    ax.set_title("co-skill calibration"); ax.legend()
    fig.suptitle(TITLE("FULL-LATENT — occurrence-level co-clustering recovery (label invariant)"),y=1.06,fontsize=12)
    fig.tight_layout(rect=[0,0.01,1,0.99])
    save(fig,"fig_G_coclustering_recovery","co-skill indicators need no alignment; they are invariant to skill relabeling")

    # ── FIGURE H ─────────────────────────────────────────────────────────────
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.2))
    ax=axes[0]; w=0.26; xs=np.arange(K)
    ax.bar(xs-w,T["pi"],width=w,color=INK2,label="pi*")
    for a_,arm in enumerate(ARMS):
        ax.bar(xs+a_*w,D[arm+"_cache"]["piM"],width=w,color=ARM_C[arm],label=f"{arm} posterior mean")
    ax.set_xticks(xs,[f"true skill {k}" for k in range(K)]); ax.set_ylabel("probability")
    ax.set_title("aligned pi recovery"); ax.legend(fontsize=7.5)
    ax=axes[1]
    for arm in ARMS:
        piA=D[arm+"_cache"]["piA"]
        parts=ax.violinplot([piA[:,k] for k in range(K)],positions=xs+(0.16 if arm==ARMS[1] else -0.16),
                            widths=0.28,showextrema=False)
        for b in parts["bodies"]: b.set_facecolor(ARM_C[arm]); b.set_alpha(0.55)
    for k in range(K): ax.plot([k-0.35,k+0.35],[T["pi"][k]]*2,color=INK,lw=1.6,ls="--")
    ax.set_xticks(xs,[f"true skill {k}" for k in range(K)])
    ax.set_title("aligned posterior spread (dashed = pi*)")
    ax=axes[2]
    for arm in ARMS:
        for j in range(K):
            ax.plot(D[arm][0]["sweep"],D[arm][0]["sorted_pi"][:,j],color=ARM_C[arm],alpha=0.5,lw=0.7)
    ax.set_title("sorted(pi) — the INVARIANT convergence quantity"); ax.set_xlabel("sweep (chain 0)")
    fig.suptitle(TITLE("FULL-LATENT — pi recovery under the single common structural alignment"),y=1.06,fontsize=12)
    fig.tight_layout(rect=[0,0.01,1,0.99])
    save(fig,"fig_H_pi_recovery","left/middle are TRUTH-ALIGNED recovery; right is permutation-invariant convergence — different questions")

    # ── FIGURE I ─────────────────────────────────────────────────────────────
    fig,axes=plt.subplots(1,4,figsize=(15,3.9))
    mats=[("P*",T["P"]),(f"{ARMS[0]} mean",D[ARMS[0]+'_cache']["PM"]),
          (f"{ARMS[1]} mean",D[ARMS[1]+'_cache']["PM"])]
    for a_,(nm,Mx) in enumerate(mats):
        ax=axes[a_]; ax.imshow(Mx,cmap="Blues",vmin=0,vmax=0.8)
        for i in range(K):
            for j in range(K):
                ax.text(j,i,"0" if i==j else f"{Mx[i,j]:.2f}",ha="center",va="center",fontsize=11,
                        color=(MUTED if i==j else ("#ffffff" if Mx[i,j]>0.45 else INK)))
        ax.set_xticks(range(K),[f"to {j}" for j in range(K)]); ax.set_yticks(range(K),[f"from {i}" for i in range(K)])
        ax.set_title(nm); ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
    ax=axes[3]
    err=np.stack([np.abs(D[a+'_cache']["PM"]-T["P"]) for a in ARMS])
    ax.bar([f"{a}" for a in ARMS],[metrics["arms"][a]["P_frobenius_error"] for a in ARMS],
           color=[ARM_C[a] for a in ARMS],width=0.5)
    ax.set_title("Frobenius error vs P*"); ax.set_ylabel("||P_hat - P*||_F")
    fig.suptitle(TITLE("FULL-LATENT — P recovery under the SAME alignment used for H and pi"),y=1.08,fontsize=12)
    fig.tight_layout(rect=[0,0.01,1,0.99])
    save(fig,"fig_I_P_recovery","no P-specific permutation was chosen; P_kk = 0 is structural")

    # ── FIGURE J ─────────────────────────────────────────────────────────────
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    for ax,mode,xl in ((axes[0],"sweep","sweep"),(axes[1],"hours","wall-clock hours")):
        for arm in ARMS:
            for i,ch in enumerate(D[arm]):
                xv=ch["sweep"] if mode=="sweep" else ch["sweep"]/ch["total_sweep"]*(ch["meta"]["seconds"]/3600)
                ax.plot(xv,ch["log_target"],color=ARM_C[arm],alpha=0.7,lw=0.8)
        ax.set_xlabel(xl); ax.set_ylabel("log target")
    axes[0].set_title("per sweep"); axes[1].set_title("per wall-clock hour")
    fig.legend(handles=[Line2D([],[],color=ARM_C[a],lw=2,label=a) for a in ARMS],
               loc="upper center",ncol=2,bbox_to_anchor=(0.5,1.0))
    fig.suptitle(TITLE("FULL-LATENT — exploration efficiency: both arms target the SAME posterior"),y=1.07,fontsize=12)
    fig.tight_layout(rect=[0,0.01,1,0.99])
    save(fig,"fig_J_cost","wall-clock axis starts after the unretained 10,000-sweep burn-in")

    # ── report ───────────────────────────────────────────────────────────────
    (OUT/"MIDRUN_RECOVERY_METRICS.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n")
    A,B_=metrics["arms"][ARMS[0]],metrics["arms"][ARMS[1]]
    def cell(a,k,f="{}"): return f.format(metrics["arms"][a][k])
    L=[]
    L.append(f"# FULL-LATENT — mid-run truth recovery analysis\n")
    L.append(f"**{BANNER}**\n")
    L.append("Generating truth was unsealed mid-run at the PI's explicit instruction. "
             "No formal experimental setting was subsequently changed.\n")
    L.append("## Provenance\n")
    L.append(f"- analysed at: `{state['analysed_at_utc']}`")
    L.append(f"- git commit: `{state['git_commit']}`")
    L.append(f"- corpus hash: `{metrics['truth']['corpus_hash']}`")
    L.append(f"- truth hash: `{metrics['truth']['truth_hash']}`")
    L.append(f"- truth-unseal event: `results/mcmc_original/matched_full_latent/TRUTH_UNSEAL_midrun.json`")
    L.append(f"- live worker processes at analysis time: **{state['live_worker_processes']}**")
    L.append(f"- sampler/runner sources modified: **{state['sampler_sources_modified']}**")
    L.append("- analysis was READ-ONLY: it opened checkpoints and the frozen corpus for reading only, "
             "started/stopped nothing, and imported no module the live workers use "
             "(registered R-hat/ESS and probe definitions are vendored verbatim in "
             "`scripts/_midrun_vendored_registered.py`, digests re-verified at import).")
    L.append("- no gate file, checkpoint, seed, start, threshold, prior, cadence, scale or datum was written or altered.\n")
    L.append("## Checkpoints analysed\n")
    L.append("| arm | chain | sweep | retained | hours | s/sweep | checkpoint sha256 |")
    L.append("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for r in state["chains"][arm]:
            L.append(f"| {arm} | {r['chain']} | {r['sweep']:,} | {r['retained']} | {r['hours']} | "
                     f"{r['sec_per_sweep']} | `{r['checkpoint_sha256'][:16]}` |")
    tr=metrics["truth"]
    L.append(f"\n## Generating truth\n")
    L.append(f"- K = {tr['K']} skills over {tr['n_roles']} roles A–E")
    L.append(f"- relations per skill (true index order) = {tr['relations_per_skill']}; "
             f"sorted = {tr['sorted_relation_counts']}; total = {tr['total_relations']}")
    L.append(f"- pi* = {tr['pi_star']}")
    L.append(f"- P* = {tr['P_star']}")
    L.append(f"- canonical library sha256 = `{tr['canonical_library_sha256'][:16]}`\n")
    L.append("## MID-RUN EXPLORATORY RECOVERY — FORMAL CONVERGENCE NOT YET ESTABLISHED\n")
    L.append("| Metric | FULL-COND | FULL-MARG |")
    L.append("|---|---|---|")
    def row(name,fn): L.append(f"| {name} | {fn(ARMS[0])} | {fn(ARMS[1])} |")
    row("sweeps per chain",lambda a:f"{metrics['arms'][a]['sweeps'][0]:,}")
    def rh(a):
        v=metrics["arms"][a]["max_invariant_rhat"]
        return (f"{v:.3f}" if v<1e4 else
                f"~1e{int(math.log10(v))} (degenerate: chains frozen at different constants)")
    row("max invariant R-hat",rh)
    row("max invariant R-hat, excluding frozen-degenerate",
        lambda a:f"{metrics['arms'][a]['max_invariant_rhat_excluding_frozen_degenerate']:.3f}")
    row("invariants with >=1 exactly-frozen chain",
        lambda a:f"{len(metrics['arms'][a]['degenerate_frozen_invariants'])}")
    row("log-target R-hat",lambda a:f"{metrics['arms'][a]['diagnostics']['log_target']['rhat']:.3f}")
    row("total-relations R-hat",lambda a:f"{metrics['arms'][a]['diagnostics']['total_relations']['rhat']:.3f}")
    row("chains whose dominant library == TRUE (last 400)",
        lambda a:f"{sum(l['tail400_dominant_equals_truth'] for l in metrics['arms'][a]['library'])} / 4")
    row("closure F1 (mean, last 400)",
        lambda a:f"{np.mean([l['tail400_closure_f1'] for l in metrics['arms'][a]['library']]):.3f}")
    row("closure Hamming (mean, last 400)",
        lambda a:f"{np.mean([l['tail400_closure_hamming'] for l in metrics['arms'][a]['library']]):.2f}")
    row("boundary Brier (pooled, 100 traces)",lambda a:f"{metrics['arms'][a]['boundary_brier_pooled']:.4f}")
    row("co-skill Brier (pooled, 256 probes)",lambda a:f"{metrics['arms'][a]['coskill_brier_pooled']:.4f}")
    row("pi total-variation error",lambda a:f"{metrics['arms'][a]['pi_TV']:.4f}")
    row("P Frobenius error",lambda a:f"{metrics['arms'][a]['P_frobenius_error']:.4f}")
    row("P off-diagonal RMSE",lambda a:f"{metrics['arms'][a]['P_offdiag_RMSE']:.4f}")
    row("sec / sweep",lambda a:f"{metrics['arms'][a]['sec_per_sweep']:.3f}")
    row("sweeps / hour",lambda a:f"{metrics['arms'][a]['sweeps_per_hour']:.0f}")
    L.append("\n### Chain-by-chain exact library status (last 400 retained draws)\n")
    L.append("| arm | chain | distinct states | dominant occupancy | dominant == TRUE | exact-draw fraction | closure F1 | Hamming |")
    L.append("|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for l in metrics["arms"][arm]["library"]:
            L.append(f"| {arm} | {l['chain']} | {l['distinct_states']} | {l['tail400_occupancy']:.2f} | "
                     f"{'YES' if l['tail400_dominant_equals_truth'] else 'no'} | {l['tail400_exact_fraction']:.2f} | "
                     f"{l['tail400_closure_f1']:.3f} | {l['tail400_closure_hamming']:.2f} |")
    L.append("\n### Example traces (first of each length class)\n")
    L.append("| trace | J | true blocks | COND E[blocks] | MARG E[blocks] | COND Brier | MARG Brier |")
    L.append("|---|---|---|---|---|---|---|")
    for r in metrics["example_traces"]:
        L.append(f"| {r['trace']} | {r['J']} | {r['true_blocks']} | {r['FULL-COND_expected_blocks']} | "
                 f"{r['FULL-MARG_expected_blocks']} | {r['FULL-COND_boundary_brier']:.4f} | "
                 f"{r['FULL-MARG_boundary_brier']:.4f} |")
    L.append("\n### Registered-invariant R-hat / ESS (current draws)\n")
    L.append("| invariant | COND R-hat | COND bulk ESS | MARG R-hat | MARG bulk ESS |")
    L.append("|---|---|---|---|---|")
    for k in metrics["arms"][ARMS[0]]["diagnostics"]:
        a=metrics["arms"][ARMS[0]]["diagnostics"][k]; b=metrics["arms"][ARMS[1]]["diagnostics"][k]
        fa="†" if a["degenerate"] else ""; fb="†" if b["degenerate"] else ""
        ra=(f"{a['rhat']:.3g}" if a['rhat'] is not None else "n/a")
        rb=(f"{b['rhat']:.3g}" if b['rhat'] is not None else "n/a")
        L.append(f"| {k} | {ra}{fa} | {a['bulk_ess']} | {rb}{fb} | {b['bulk_ess']} |")
    L.append("\n† = at least one chain is EXACTLY constant over all retained draws while the chains "
             "disagree. The rank-normalized statistic then diverges numerically; the magnitude is "
             "meaningless but the condition itself is the finding (structural coordinates frozen).\n")
    L.append("\n## Figures\n")
    for s in ("fig_A_invariant_traces","fig_B_running_rhat","fig_C_exact_library_states",
              "fig_D_truth_vs_learned_posets","fig_E_example_boundaries","fig_F_boundary_recovery",
              "fig_G_coclustering_recovery","fig_H_pi_recovery","fig_I_P_recovery","fig_J_cost"):
        L.append(f"- `{s}.png` / `{s}.pdf`")
    L.append("\n## Interpretation guard\n")
    L.append("- Neither arm is called converged here. Registered gates fire only at 30k/50k/75k/100k sweeps "
             "and require two consecutive PASSes; none has been evaluated.")
    L.append("- Recovery toward truth and posterior convergence across chains are separate questions. "
             "A chain can sit on the true library and still not be converged.")
    L.append("- Both arms target the SAME posterior. Differences below are exploration efficiency and "
             "mixing, not a different model or a better likelihood.")
    (OUT/"MIDRUN_RECOVERY_REPORT.md").write_text("\n".join(L)+"\n")
    print("  wrote MIDRUN_RECOVERY_REPORT.md / MIDRUN_RECOVERY_METRICS.json")
    return metrics

if __name__=="__main__":
    main()
