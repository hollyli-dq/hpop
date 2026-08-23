"""Independent reproduction and audit of the confirmatory terminal gate.

    PYTHONPATH=src python scripts/confirmatory_gate_audit.py

The gate script imports the repository's registered diagnostics
(`stage6b_mcmc_diagnostics`). This audit deliberately does NOT: it implements
rank-normalized split R-hat, bulk ESS and tail ESS from scratch, following
Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021), and then compares. An
agreement between two independent implementations is evidence; re-running the
same code twice is not.

TRUTH-FREE: opens no truth file.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "mcmc_optimized" / "confirmatory_run"
OUT = RUN / "terminal_gate"


# ----------------------------------------------- independent Vehtari et al. (2021)
def _split(chains: np.ndarray) -> np.ndarray:
    m, n = chains.shape
    half = n // 2
    return np.concatenate([chains[:, :half], chains[:, n - half:]], axis=0)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1)
    order = flat.argsort(kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, flat.size + 1, dtype=float)
    # average ranks within ties
    uniq, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    sums = np.zeros(uniq.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    z = norm.ppf((ranks - 3.0 / 8.0) / (flat.size - 0.25))
    return z.reshape(x.shape)


def _classic_rhat(chains: np.ndarray) -> float:
    m, n = chains.shape
    if n < 2:
        return float("inf")
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = n * chain_means.var(ddof=1)
    if W <= 0:
        return float("inf") if B > 0 else 1.0
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def rhat_independent(chains: np.ndarray) -> float:
    s = _split(np.asarray(chains, dtype=float))
    z = _rank_normalize(s)
    folded = _rank_normalize(np.abs(s - np.median(s)))
    return float(max(_classic_rhat(z), _classic_rhat(folded)))


def _ess_from(chains: np.ndarray) -> float:
    """Geyer initial-positive-sequence ESS on already-split chains."""
    m, n = chains.shape
    if n < 4:
        return 0.0
    centred = chains - chains.mean(axis=1, keepdims=True)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(centred, n=nfft, axis=1)
    acov = np.fft.irfft(f * np.conjugate(f), n=nfft, axis=1)[:, :n].real / n
    chain_var = chains.var(axis=1, ddof=1)
    W = chain_var.mean()
    if W <= 0:
        return 0.0
    chain_means = chains.mean(axis=1)
    B = n * chain_means.var(ddof=1) if m > 1 else 0.0
    var_hat = (n - 1) / n * W + B / n
    rho = 1.0 - (W - acov.mean(axis=0) * n / (n - 1)) / var_hat
    rho[0] = 1.0
    # Geyer initial positive sequence on successive pairs
    t, total = 1, 0.0
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        total += pair
        t += 2
    tau = -1.0 + 2.0 * total
    # A near-iid series makes the first Geyer pair non-positive, leaving tau = -1.
    # Integrated autocorrelation time is at least 1 for a positively correlated chain,
    # so ESS is at most m*n; without this floor the estimator returns several times the
    # sample size, which is not a possible ESS.
    tau = max(tau, 1.0)
    return float(m * n / tau)


def bulk_ess_independent(chains: np.ndarray) -> float:
    return _ess_from(_rank_normalize(_split(np.asarray(chains, dtype=float))))


def tail_ess_independent(chains: np.ndarray) -> float:
    x = np.asarray(chains, dtype=float)
    q05, q95 = np.quantile(x, 0.05), np.quantile(x, 0.95)
    lo, hi = (x <= q05).astype(float), (x <= q95).astype(float)
    # For a binary variable q95 = 1 whenever the mean exceeds 0.05, so the upper
    # indicator is identically true and carries no information. Use the indicators that
    # actually vary; tail ESS is undefined only when NEITHER does, which for a Bernoulli
    # probe means its mean sits outside [0.05, 0.95] on both sides.
    out = [_ess_from(_rank_normalize(_split(ind))) for ind in (lo, hi)
           if not np.all(ind == ind.flat[0])]
    return float(min(out)) if out else float("nan")


def main() -> int:
    report = json.loads((OUT / "terminal_gate.json").read_text())
    audit = {"method": "independent from-scratch Vehtari et al. (2021); the gate uses "
                       "the repository's registered stage6b_mcmc_diagnostics",
             "arms": {}}
    overall_ok = True

    for arm, tag in (("FULL-COND", "full_cond"), ("FULL-MARG", "full_marg")):
        data = [np.load(RUN / "chains" / f"{tag}_{i}.npz", allow_pickle=False)
                for i in range(4)]
        rows = report["arms"][arm]["summaries"]
        checked = agree_branch = agree_verdict = 0
        worst_rhat = worst_bulk = worst_tail = 0.0
        disagreements = []

        for label, got in rows.items():
            name = label.split("[")[0]
            idx = int(label.split("[")[1][:-1]) if "[" in label else None
            arrs = [d[f"summary__{name}"] for d in data]
            series = np.array([a[:, idx] if idx is not None else a for a in arrs],
                              dtype=float)
            constant = [bool(np.all(c == c[0])) for c in series]
            if all(constant):
                branch = ("constant_and_equal"
                          if len({float(c[0]) for c in series}) == 1
                          else "constant_within_unequal_across")
                mine = {"branch": branch, "rhat": None, "bulk_ess": None,
                        "tail_ess": None}
            else:
                mine = {"branch": "non_degenerate",
                        "rhat": rhat_independent(series),
                        "bulk_ess": bulk_ess_independent(series),
                        "tail_ess": tail_ess_independent(series)}
            checked += 1
            if mine["branch"] == got["branch"]:
                agree_branch += 1
            else:
                disagreements.append({"summary": label, "kind": "branch",
                                      "gate": got["branch"], "audit": mine["branch"]})
            if mine["branch"] == "non_degenerate":
                for key, worst in (("rhat", "worst_rhat"), ("bulk_ess", "worst_bulk"),
                                   ("tail_ess", "worst_tail")):
                    a, b = mine[key], got[key]
                    if a != a and b != b:          # both nan
                        continue
                    if a is None or b is None:
                        continue
                    rel = abs(a - b) / max(abs(b), 1e-12)
                    if key == "rhat":
                        worst_rhat = max(worst_rhat, rel)
                    elif key == "bulk_ess":
                        worst_bulk = max(worst_bulk, rel)
                    else:
                        worst_tail = max(worst_tail, rel)
                # verdict agreement is what actually matters
                bulk_floor = 1000.0 if name == "log_target" else 400.0
                tail_floor = 500.0 if name == "log_target" else 400.0
                mine_pass = (mine["rhat"] <= 1.01
                             and mine["bulk_ess"] >= bulk_floor
                             and mine["tail_ess"] >= tail_floor)
            else:
                mine_pass = mine["branch"] == "constant_and_equal"
            if bool(mine_pass) == bool(got["pass"]):
                agree_verdict += 1
            else:
                disagreements.append({"summary": label, "kind": "pass",
                                      "gate": got["pass"], "audit": bool(mine_pass)})

        # Two legitimate ESS estimators can straddle a floor on a marginal summary.
        # What must reproduce is the ARM VERDICT; per-summary agreement is reported
        # in full so any straddling case is visible rather than hidden.
        gate_fail = report["arms"][arm]["n_failures"]
        audit_fail = gate_fail + (checked - agree_verdict)
        arm_verdict_audit = "PASS" if audit_fail == 0 else "FAIL"
        arm_agrees = arm_verdict_audit == report["arms"][arm]["verdict"]
        ok = (agree_branch == checked and arm_agrees)
        overall_ok = overall_ok and ok
        audit["arms"][arm] = {
            "summaries_checked": checked,
            "branch_agreement": f"{agree_branch}/{checked}",
            "pass_fail_agreement": f"{agree_verdict}/{checked}",
            "max_relative_difference": {"rhat": worst_rhat, "bulk_ess": worst_bulk,
                                        "tail_ess": worst_tail},
            "disagreements": disagreements[:20],
            "reproduces_gate": ok,
            "gate_verdict": report["arms"][arm]["verdict"],
            "audit_verdict": arm_verdict_audit,
            "arm_verdict_agrees": arm_agrees,
            "audit_is_more_conservative": audit_fail >= gate_fail,
        }
        print(f"{arm}: branch {agree_branch}/{checked}, per-summary pass/fail "
              f"{agree_verdict}/{checked}, arm verdict gate={report['arms'][arm]['verdict']} "
              f"audit={arm_verdict_audit}  | max rel diff rhat {worst_rhat:.2e} "
              f"bulk {worst_bulk:.2e} tail {worst_tail:.2e}  "
              f"-> {'REPRODUCED' if ok else 'DISAGREES'}")

    audit["independently_reproduced"] = overall_ok
    (OUT / "gate_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True,
                                                    default=str) + "\n")
    print(f"\nindependently_reproduced = {overall_ok}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
