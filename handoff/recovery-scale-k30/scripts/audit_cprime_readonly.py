"""Three READ-ONLY audits of the live Condition C' run. Touches nothing formal.

Run:  PYTHONPATH=src .venv/bin/python scripts/audit_cprime_readonly.py

1. Rejection sanity: log alpha = Delta ell_coll exactly (Hastings 0, prior
   difference 0), so NO rejected swap may have Delta ell_coll > 1e-10.
2. Independent recomputation of recorded live swap proposals through a code
   path that never calls skill_swap_kernel.py: witness U from the chain
   checkpoint (valid because ell_coll reads U only through the H tuple, and
   the selected chains hold ONE H tuple across their entire retained series),
   U' built by plain numpy row-block exchange, and ell_coll evaluated with the
   Condition-A SemiMarkovPosterior forward DP — an implementation disjoint
   from the semi_markov_ffbs.forward used inside the live run.
3. Fresh re-implementation of the REGISTERED rank-normalized split-Rhat /
   ESS from the retained draws, compared against the recorded 30k gate
   values; for every infinite indicator, explicit verification that W = 0
   within every chain and B > 0 between chains.

No truth or recovery quantity is read. No formal file is written or modified;
the only output is a new audit artifact.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_synthetic_generator as msg              # noqa: E402
from hpop.mcmc_original.matched_condition_a import SemiMarkovPosterior         # noqa: E402
from hpop.mcmc_original.recurrent_segmentation import RecurrentBlockScorer     # noqa: E402
from hpop.mcmc_original.transitions import log_transition_matrix               # noqa: E402

D = ROOT / "results" / "mcmc_original" / "matched_condition_c_prime"
CHAINS = D / "formal_chains"
OUT = D / "readonly_audits"
GATE_DRAWS = (30_000 - 10_000) // 5


def load(name):
    d = np.load(str(CHAINS / f"{name}.npz"))
    m = json.loads(str(d["meta"]))
    recs = [[str(r[0]), float(r[1]), bool(int(r[2])), int(r[3])]
            for r in d["swap_deltas"]]
    hh = [tuple(str(v) for v in row) for row in d["h_hashes"]]
    return d, m, recs, hh


CHAIN_NAMES = [os.path.basename(p)[:-4]
               for p in sorted(glob.glob(str(CHAINS / "*.npz")))]


# ======================================================================= check 1
def check_rejection_sanity():
    print("=" * 72)
    print("CHECK 1 — no rejected swap may carry Delta ell_coll > 1e-10")
    print("=" * 72)
    worst = -math.inf
    worst_at = None
    n_rej = 0
    for name in CHAIN_NAMES:
        _, _, recs, _ = load(name)
        for pair, delta, accepted, sweep in recs:
            if not accepted:
                n_rej += 1
                if delta > worst:
                    worst, worst_at = delta, (name, pair, sweep)
    ok = worst <= 1e-10
    print(f"  rejected proposals scanned : {n_rej}")
    print(f"  max Delta ell among rejected: {worst:.6e}  at {worst_at}")
    print(f"  verdict: {'PASS' if ok else 'FAIL — a positive-ratio proposal was rejected'}")
    return {"n_rejected_scanned": n_rej, "max_delta_among_rejected": worst,
            "argmax": list(worst_at), "gate": 1e-10, "pass": bool(ok)}


# ======================================================================= check 2
def check_independent_recomputation():
    print("=" * 72)
    print("CHECK 2 — independent recomputation of recorded live swap proposals")
    print("=" * 72)
    truth = msg.supplied_truth()          # fixed inputs only; no hidden field read
    corpus = msg.generate_corpus(
        6_200_001, tuple((24, 32, 40, 48)[i % 4] for i in range(100)),
        tuple((24, 32, 40, 48)[i % 4] for i in range(45)), truth)
    traces = tuple(t.cpa for t in corpus.train)
    log_pi = np.log(np.asarray(truth.pi, dtype=float))
    log_p = log_transition_matrix(np.asarray(truth.transition, dtype=float))

    def ell_coll(u_by_skill):
        scorer = RecurrentBlockScorer(
            traces=traces, epsilon=truth.epsilon,
            u_by_skill=np.asarray(u_by_skill, dtype=float), beta=truth.beta,
            omega=truth.omega, lambda_rep=truth.lambda_rep,
            lambda_back=truth.lambda_back, max_width=truth.max_width,
            min_width=truth.min_width)
        total = 0.0
        for n, trace in enumerate(traces):
            total += SemiMarkovPosterior(
                n, len(trace), scorer, log_pi, log_p, truth.delta_b,
                truth.min_width, truth.max_width).log_z
        return total

    rows, verified_proposals = [], 0
    worst = 0.0
    for name in CHAIN_NAMES:
        d, m, recs, hh = load(name)
        if len(set(hh)) != 1:
            rows.append({"chain": name, "skipped":
                         f"{len(set(hh))} distinct retained H tuples — the "
                         "checkpoint U is not an unambiguous witness for "
                         "historical proposals; covered by the other chains"})
            print(f"  {name}: SKIP ({len(set(hh))} distinct retained H)")
            continue
        u = np.asarray(d["u_by_skill"], dtype=float)
        base = ell_coll(u)
        # group equilibrium-era records: the stored delta must be identical
        # within (chain, pair) because ell depends only on the H tuple
        eq = [r for r in recs if r[3] > 10_000 and not r[2]]
        by_pair = defaultdict(list)
        for pair, delta, _, sweep in eq:
            by_pair[pair].append(delta)
        for pair in sorted(by_pair):
            deltas = by_pair[pair]
            spread = max(deltas) - min(deltas)
            j, k = int(pair[0]), int(pair[2])
            u_prime = np.array(u, copy=True)         # plain numpy exchange —
            u_prime[[j, k]] = u_prime[[k, j]]        # skill_swap_kernel not called
            independent = ell_coll(u_prime) - base
            err = abs(independent - deltas[0])
            worst = max(worst, err, spread)
            verified_proposals += len(deltas)
            rows.append({"chain": name, "pair": pair,
                         "n_recorded_proposals_covered": len(deltas),
                         "stored_delta": deltas[0],
                         "stored_delta_spread_within_group": spread,
                         "independent_delta": independent,
                         "abs_error": err})
            print(f"  {name} swap({pair}): stored {deltas[0]:>10.4f}  "
                  f"independent {independent:>10.4f}  |err| {err:.2e}  "
                  f"covers {len(deltas)} recorded proposals")
    ok = worst <= 1e-10 and verified_proposals >= 20
    print(f"  proposals covered: {verified_proposals}  worst |err|/spread: "
          f"{worst:.2e}  verdict: {'PASS' if ok else 'FAIL'}")
    return {"rows": rows, "recorded_proposals_covered": verified_proposals,
            "worst_abs_error": worst, "gate": 1e-10, "pass": bool(ok),
            "independence": "witness U from checkpoint (valid: ell_coll reads "
                            "U only through H, and each used chain holds one "
                            "retained H tuple); U' by plain numpy row-block "
                            "exchange; ell_coll via the Condition-A "
                            "SemiMarkovPosterior forward DP with the "
                            "production block scorer — skill_swap_kernel.py "
                            "and collapsed_u_likelihood.py are never called",
            "burn_in_era_note": "pre-10k proposals have no retained draws to "
                                "witness the pre-swap cell and are excluded"}


# ======================================================================= check 3
def fresh_rank_normalize(values):
    ranks = stats.rankdata(values, method="average").reshape(values.shape)
    return stats.norm.ppf((ranks - 0.375) / (values.size - 0.25))


def fresh_classic_rhat(chains):
    c = np.atleast_2d(np.asarray(chains, dtype=float))
    m, n = c.shape
    w = float(c.var(axis=1, ddof=1).mean())
    b = float(c.mean(axis=1).var(ddof=1))
    if w == 0:
        return float("nan") if b > 0 else 1.0
    return math.sqrt((n - 1) / n + b / w)


def fresh_split(chains):
    c = np.atleast_2d(np.asarray(chains, dtype=float))
    h = c.shape[1] // 2
    return np.vstack([c[:, :h], c[:, -h:]])


def fresh_rhat(chains):
    c = np.atleast_2d(np.asarray(chains, dtype=float))
    bulk = fresh_classic_rhat(fresh_rank_normalize(fresh_split(c)))
    folded = np.abs(c - np.median(c))
    tail = fresh_classic_rhat(fresh_rank_normalize(fresh_split(folded)))
    return max(bulk, tail)


def fresh_ess(chains):
    c = fresh_rank_normalize(fresh_split(np.atleast_2d(
        np.asarray(chains, dtype=float))))
    m, n = c.shape
    means = c.mean(axis=1, keepdims=True)
    w = c.var(axis=1, ddof=1).mean()
    b = n * c.mean(axis=1).var(ddof=1) if m > 1 else 0.0
    var_hat = (n - 1) / n * w + b / n
    centered = c - means
    rho = []
    for t in range(1, n):
        acov = np.mean([np.dot(centered[i, :-t], centered[i, t:]) / n
                        for i in range(m)])
        rho.append(1.0 - (w - acov) / var_hat)
    rho = np.asarray(rho)
    # Geyer initial monotone positive sequence on pair sums
    tau = 1.0
    prev = math.inf
    t = 0
    while t + 1 < len(rho):
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        pair = min(pair, prev)
        tau += 2.0 * pair
        prev = pair
        t += 2
    total = m * n
    tau = max(tau, 1.0 / math.log10(total)) if total > 10 else max(tau, 1.0)
    return total / tau


def check_gate_reproduction():
    print("=" * 72)
    print("CHECK 3 — fresh reproduction of the registered R-hat / ESS at 30k")
    print("=" * 72)
    out = {}
    for arm in ("C-COND-SWAP", "C-MARG-SWAP"):
        gate = json.loads((D / f"formal_gate_{arm}_30000.json").read_text())
        data = [load(f"{arm}_{i}") for i in range(4)]
        lt = [d[0]["log_target"][:GATE_DRAWS] for d in data]
        lp = [d[0]["log_prior"][:GATE_DRAWS] for d in data]
        rc = [np.asarray(d[0]["rel_counts"][:GATE_DRAWS], float) for d in data]
        ind = [np.asarray(d[0]["indicators"][:GATE_DRAWS], float) for d in data]
        per_skill = [i.reshape(i.shape[0], 3, 20).sum(axis=2) for i in ind]

        arm_rows = []

        def compare(label, series, gate_entry):
            per_chain_const = [np.all(s == s[0]) for s in series]
            if all(per_chain_const):
                values = {float(s[0]) for s in series}
                mine_rhat = 1.0 if len(values) == 1 else float("inf")
                mine_ess = float(sum(len(s) for s in series)) \
                    if len(values) == 1 else 0.0
            else:
                mine_rhat = fresh_rhat(series)
                mine_ess = fresh_ess(series)
            g_rhat, g_ess = gate_entry["rhat"], gate_entry["bulk_ess"]
            if math.isinf(mine_rhat) or math.isinf(g_rhat):
                agree = math.isinf(mine_rhat) == math.isinf(g_rhat)
            else:
                agree = abs(mine_rhat - g_rhat) <= 1e-6 * max(1.0, abs(g_rhat))
            ess_rel = (abs(mine_ess - g_ess) / max(g_ess, 1.0)
                       if not math.isinf(mine_ess) else 0.0)
            arm_rows.append({"summary": label,
                             "gate_rhat": g_rhat, "fresh_rhat": mine_rhat,
                             "rhat_agrees": bool(agree),
                             "gate_bulk_ess": g_ess, "fresh_bulk_ess": mine_ess,
                             "ess_rel_diff": ess_rel})
            print(f"  {arm} {label:24} rhat gate={g_rhat if not math.isinf(g_rhat) else 'inf':>10} "
                  f"fresh={mine_rhat if not math.isinf(mine_rhat) else 'inf':>10} "
                  f"agree={agree}  ess gate={g_ess:.0f} fresh={mine_ess:.0f} "
                  f"(rel {ess_rel:.1%})")

        compare("log_target", lt, gate["summaries"]["log_target"])
        compare("log_prior", lp, gate["summaries"]["log_prior"])
        compare("total_relations", rc, gate["summaries"]["total_relations"])
        for k in range(3):
            compare(f"relations_skill{k}", [c[:, k] for c in per_skill],
                    gate["summaries"][f"relations_skill{k}"])

        # every infinite-R-hat indicator: W = 0 within chains, B > 0 between
        inf_checks = []
        for key, entry in gate["uncertain"].items():
            if entry["rhat"] != float("inf") and entry["rhat"] < 1e308:
                continue
            i = int(key)
            series = [c[:, i] for c in ind]
            w_zero = all(float(np.var(s)) == 0.0 for s in series)
            values = [float(s[0]) for s in series]
            b_pos = len(set(values)) > 1
            inf_checks.append({"indicator": i, "within_variance_all_zero": w_zero,
                               "between_variance_positive": b_pos,
                               "per_chain_constants": values,
                               "consistent_with_inf": bool(w_zero and b_pos)})
        n_bad = sum(1 for r in inf_checks if not r["consistent_with_inf"])
        print(f"  {arm}: {len(inf_checks)} infinite indicators checked, "
              f"{'all W=0 & B>0' if n_bad == 0 else f'{n_bad} INCONSISTENT'}")
        out[arm] = {"summaries": arm_rows, "infinite_indicators": inf_checks,
                    "all_rhat_agree": all(r["rhat_agrees"] for r in arm_rows),
                    "max_ess_rel_diff": max(r["ess_rel_diff"]
                                            for r in arm_rows),
                    "all_inf_consistent": n_bad == 0}
    return out


def main() -> int:
    OUT.mkdir(exist_ok=True)
    c1 = check_rejection_sanity()
    c2 = check_independent_recomputation()
    c3 = check_gate_reproduction()
    payload = {
        "read_only": True,
        "no_truth_or_recovery_read": True,
        "no_formal_file_modified": True,
        "check1_rejection_sanity": c1,
        "check2_independent_recomputation": c2,
        "check3_gate_reproduction": c3,
        "return_transition_wording": {
            "event": "C-COND-SWAP_0 accepted swap(Delta ell = -4.54) at sweep "
                     "12,299 and the exact inverse (+4.54) at sweep 12,499",
            "correct_characterisation":
                "a post-burn-in return transition demonstrating two-way "
                "anchored-assignment mobility across a small (~4.5 nat) "
                "barrier",
            "explicitly_not_claimed":
                "this is NOT evidence of mixing between the two major "
                "Condition-C assignment modes, whose one-step transposition "
                "barriers measure ~83-132 nats in the equilibrium swap-delta "
                "records",
        },
    }
    (OUT / "readonly_audit_30k.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    print("\nartifact:", (OUT / "readonly_audit_30k.json").relative_to(ROOT))
    all_pass = (c1["pass"] and c2["pass"]
                and all(v["all_rhat_agree"] and v["all_inf_consistent"]
                        for v in c3.values()))
    print("OVERALL:", "ALL CHECKS PASS" if all_pass else "A CHECK FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
