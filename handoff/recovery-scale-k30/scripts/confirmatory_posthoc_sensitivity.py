"""POST-HOC DIAGNOSTIC SENSITIVITY -- NOT THE FORMAL VERDICT.

    PYTHONPATH=src python scripts/confirmatory_posthoc_sensitivity.py

The registered gate failed FULL-MARG on 11 Bernoulli probes whose 5% and 95% quantile
indicators are both constant, making tail ESS 0/0. This asks one counterfactual question:

    if a Bernoulli probe has constant empirical tail indicators, treat tail ESS as NOT
    APPLICABLE and assess it on R-hat, bulk ESS, and the MCSE of its posterior probability

and reports what that alternative rule would conclude.

It changes nothing. The registered verdicts remain FULL-COND = FAIL, FULL-MARG = FAIL, and
this file may not be cited as amending them. The rule below was chosen AFTER seeing which
summaries failed, which is exactly why it cannot carry the verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RUN = ROOT / "results" / "mcmc_optimized" / "confirmatory_run"
OUT = RUN / "posthoc_sensitivity"

RHAT_GATE, OTHER_BULK = 1.01, 400.0
LOG_BULK, LOG_TAIL, OTHER_TAIL = 1000.0, 500.0, 400.0
MCSE_MAX_FOR_PROBABILITY = 0.01      # 1 percentage point on a Bernoulli probability


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gate = json.loads((RUN / "terminal_gate" / "terminal_gate.json").read_text())
    result = {
        "LABEL": "POST-HOC DIAGNOSTIC SENSITIVITY — NOT THE FORMAL VERDICT",
        "registered_verdicts_unchanged": {"FULL-COND": "FAIL", "FULL-MARG": "FAIL"},
        "alternative_rule": (
            "For a Bernoulli probe whose 5% and 95% quantile indicators are both constant, "
            "tail ESS is undefined; treat it as not applicable and require R-hat <= 1.01, "
            "bulk ESS >= 400, and MCSE of the posterior probability <= 0.01."),
        "chosen_after_seeing_results": True,
        "arms": {},
    }

    for arm, tag in (("FULL-COND", "full_cond"), ("FULL-MARG", "full_marg")):
        data = [np.load(RUN / "chains" / f"{tag}_{i}.npz", allow_pickle=False)
                for i in range(4)]
        rows = gate["arms"][arm]["summaries"]
        reclassified, still_failing = [], []

        for label, d in rows.items():
            if d["pass"] or d["branch"] != "non_degenerate":
                if not d["pass"]:
                    still_failing.append((label, d["reason"]))
                continue
            name = label.split("[")[0]
            is_binary = name in ("boundary_probes", "coskill_probes",
                                 "same_segment_probes")
            tail_undefined = d["tail_ess"] is None or d["tail_ess"] != d["tail_ess"]
            if not (is_binary and tail_undefined):
                still_failing.append((label, d["reason"]))
                continue
            idx = int(label.split("[")[1][:-1])
            series = np.array([x[f"summary__{name}"][:, idx] for x in data], dtype=float)
            p_hat = float(series.mean())
            mcse_p = float(np.sqrt(max(p_hat * (1 - p_hat), 0.0) / d["bulk_ess"]))
            ok = (d["rhat"] <= RHAT_GATE and d["bulk_ess"] >= OTHER_BULK
                  and mcse_p <= MCSE_MAX_FOR_PROBABILITY)
            entry = {"summary": label, "rhat": d["rhat"], "bulk_ess": d["bulk_ess"],
                     "posterior_probability": p_hat, "mcse_probability": mcse_p,
                     "would_pass_under_alternative": ok}
            (reclassified if ok else still_failing).append(
                entry if ok else (label, f"alternative rule also fails: mcse_p {mcse_p:.2e}"))

        # the canonical library, re-evaluated with precondition 4 recomputed
        lib = gate["arms"][arm]["canonical_library"]
        lib_note, lib_pass = lib["reason"], lib["pass"]
        if lib["branch"] == "a_constant_and_equal" and not lib["pass"]:
            others_now_pass = len(still_failing) == 0
            lib_pass = all([lib["preconditions"]["starts_dispersed"],
                            lib["preconditions"]["every_chain_moved_H_in_warmup"],
                            lib["preconditions"]["library_constant_and_equal_in_production"],
                            others_now_pass])
            lib_note = ("branch (a); precondition 4 recomputed under the alternative rule: "
                        f"{'satisfied' if others_now_pass else 'still unsatisfied'}")
        alt_fail = len(still_failing) + (0 if lib_pass else 1)
        result["arms"][arm] = {
            "registered_verdict": gate["arms"][arm]["verdict"],
            "registered_failures": gate["arms"][arm]["n_failures"],
            "reclassified_as_not_applicable": reclassified,
            "n_reclassified": len(reclassified),
            "still_failing": still_failing[:40],
            "n_still_failing": len(still_failing),
            "canonical_library_under_alternative": {"pass": lib_pass, "note": lib_note},
            "verdict_under_alternative_rule": "PASS" if alt_fail == 0 else "FAIL",
        }

    (OUT / "posthoc_sensitivity.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")

    print(result["LABEL"])
    print(f"registered verdicts unchanged: {result['registered_verdicts_unchanged']}\n")
    for arm, r in result["arms"].items():
        print(f"{arm}: registered {r['registered_verdict']} "
              f"({r['registered_failures']} failures)")
        print(f"   reclassified as N/A       : {r['n_reclassified']}")
        print(f"   still failing             : {r['n_still_failing']}")
        print(f"   canonical library         : "
              f"{'PASS' if r['canonical_library_under_alternative']['pass'] else 'FAIL'}"
              f"  ({r['canonical_library_under_alternative']['note']})")
        print(f"   VERDICT UNDER ALTERNATIVE : {r['verdict_under_alternative_rule']}"
              "   [sensitivity only]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
