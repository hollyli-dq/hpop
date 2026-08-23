"""Freeze the Condition C' (skill-swap) pre-registration. Launches nothing.

Run:  PYTHONPATH=src .venv/bin/python scripts/condition_c_prime_preregister.py

Condition C' adds ONE transition to the Condition-C kernel: a Metropolis
transposition of two skills' complete utility matrices, scored by the already
validated collapsed likelihood. The diagnosis that motivates it, and the
evidence that it repairs the observed failure, are recorded here BEFORE any
Condition C' chain exists.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original import matched_synthetic_generator as msg              # noqa: E402
from hpop.mcmc_original import skill_swap_kernel as ssk                        # noqa: E402

OUT = ROOT / "results" / "mcmc_original" / "matched_condition_c_prime"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    truth = msg.supplied_truth()
    invariance = ssk.permutation_invariance_report(truth.pi, truth.transition)

    payload = {
        "condition": "C' — joint (S, z, U) inference WITH a skill-swap move",
        "status": "PRE-REGISTERED ONLY — no chain launched; Condition C must "
                  "terminate first",
        "parent_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                        capture_output=True,
                                        text=True).stdout.strip(),

        "diagnosis": {
            "observed_in_condition_c": "at the 30k gate both arms FAIL with "
                                       "R-hat = inf (constant-within-chain, "
                                       "unequal-across-chains). All four "
                                       "C-MARG chains recovered the SAME "
                                       "unlabeled structural library; they "
                                       "differ only in which structure is "
                                       "attached to which anchored skill "
                                       "identity. C-COND chains differ in the "
                                       "library itself.",
            "is_it_label_switching": False,
            "why_not": "pi* and P* are fixed and asymmetric, so no "
                       "non-identity permutation leaves the target invariant; "
                       "the assignments are genuinely different modes, not one "
                       "mode under an arbitrary relabelling",
            "permutation_invariance_of_fixed_inputs": invariance,
            "measured_mode_gap_nats": 124.77,
            "measured_mode_gap_source": "difference in mean retained "
                                        "log-target between C-MARG chains "
                                        "{marg0, marg1, marg3} (-5776.37) and "
                                        "{marg2} (-5651.59) at 36k sweeps; "
                                        "chain-vs-chain, no truth used",
            "rejected_alternative": "a permutation-invariant secondary gate "
                                    "(canonically ordering skills before "
                                    "computing R-hat) was considered and "
                                    "REJECTED: with no symmetry present it "
                                    "would disguise three chains sitting in a "
                                    "125-nat-inferior mode as agreement",
            "note_on_data_scaling": "the gap grows with corpus size (about -1 "
                                    "nat at 4 traces, +3 at 8, +12 at 16, +20 "
                                    "at 24, ~+132 at the 100-trace formal "
                                    "corpus), so more data identifies the "
                                    "assignment more sharply AND deepens the "
                                    "barrier",
        },

        "move": {
            "proposal": "pick an unordered pair (j, k) uniformly from the "
                        "K(K-1)/2 pairs; transpose U_j and U_k entirely",
            "hastings": "0 — the map is an involution and the pair choice is "
                        "uniform, so q(U->U') = q(U'->U)",
            "prior_term": "0 exactly — log p(U|rho) is a sum over skills, so a "
                          "permutation reorders summands; computed and "
                          "asserted at every proposal rather than assumed",
            "acceptance": "log alpha = ell_coll(U') - ell_coll(U), with "
                          "ell_coll = sum_n log Z_n evaluated at the FIXED "
                          "pi*, P*; the collapsed likelihood already "
                          "integrates out (S, z), so the pi/P asymmetry that "
                          "distinguishes the assignments enters automatically",
            "ordering": "the swap runs BEFORE the unchanged Condition-C sweep, "
                        "whose first action is the exact FFBS refresh of every "
                        "(S, z) at the post-swap U — the partially-collapsed "
                        "ordering the invariance argument requires",
            "implementation": "src/hpop/mcmc_original/skill_swap_kernel.py; "
                              "composes by CALL, modifies no validated module",
        },

        "evidence_the_move_repairs_the_observed_failure": {
            "method": "the move was applied to the LIVE Condition-C chain "
                      "checkpoints at 36k sweeps (read-only)",
            "marg0_inferior_assignment": {"swap_0_1": 101.40,
                                          "swap_0_2": 132.14,
                                          "swap_1_2": 110.66,
                                          "outcome": "all three have positive "
                                                     "Delta ell_marg and "
                                                     "therefore essentially "
                                                     "unit MH acceptance"},
            "marg2_superior_assignment": {"swap_0_1": -125.46,
                                          "swap_0_2": -132.14,
                                          "swap_1_2": -104.75,
                                          "outcome": "all three have negative "
                                                     "Delta ell_marg, so marg2 "
                                                     "is LOCALLY MAXIMAL with "
                                                     "respect to the "
                                                     "registered class of "
                                                     "single whole-skill "
                                                     "transpositions"},
            "interpretation_correction_2026_08_18": {
                "superseded_statement": "'marg0 accepts all three swaps and "
                                        "one step lands on marg2's "
                                        "assignment' — imprecise, because it "
                                        "implies every favourable direction "
                                        "reaches marg2",
                "corrected_statement": "The low-target marg0 assignment has "
                                       "multiple strongly favourable global "
                                       "transposition directions; the observed "
                                       "0<->2 transposition directly reaches "
                                       "the higher-target assignment occupied "
                                       "by marg2. The other two transpositions "
                                       "produce different anchored "
                                       "assignments, which were not "
                                       "separately evaluated.",
                "global_optimality_claim": "NOT made — marg2 is local-maximal "
                                           "only within the single "
                                           "whole-skill transposition class",
            },
            "antisymmetry_check": "marg0 swap(0,2) = +132.14 and marg2 "
                                  "swap(0,2) = -132.14 to 8 decimal places, on "
                                  "real chain states",
            "cost": "~5-6 s per proposal on the 100-trace formal corpus",
        },

        "registered_design": {
            "corpus": "the SAME frozen formal corpus "
                      "(dd280a4a09896154..., truth fc41538fd44d170d...); no "
                      "regeneration",
            "fixed_variables": "unchanged from Condition C: rho_0 = 0.5, beta "
                               "1.5, omega 1.7346, lambda_rep 0.8, lambda_back "
                               "0.25, pi*, P*, delta_B 0.15, epsilon 0.02",
            "inferred": ["S", "z", "U"],
            "arms": {"C-MARG-SWAP": "every = 10 collapsed + swap every 50",
                     "C-COND-SWAP": "every = 0 collapsed + swap every 50"},
            "arm_rationale": "running the swap in BOTH arms completes a 2x2 "
                             "with the finished Condition C, separating the "
                             "contribution of path-marginal U moves from that "
                             "of assignment moves",
            "swap_cadence": 50,
            "swap_cadence_rationale": "~5.5 s per proposal against ~2.3 s per "
                                      "sweep gives ~5% overhead; the move only "
                                      "needs to fire often enough to escape "
                                      "once",
            "u_scale": 0.5, "scheduled_collapsed_scale": 1.0,
            "starts": "the SAME four registered dispersed starts (seeds "
                      "6_204_101-104, scales 0.5/1/2/3), paired across arms; "
                      "NOT initialised from any Condition-C chain endpoint",
            "seeds": {"C-MARG-SWAP": [6_205_011, 6_205_012, 6_205_013,
                                      6_205_014],
                      "C-COND-SWAP": [6_205_001, 6_205_002, 6_205_003,
                                      6_205_004]},
            "checkpoints": [30_000, 50_000, 75_000, 100_000],
            "burn_in": 10_000, "thin": 5,
            "convergence_gates": "identical to Condition B/C, unmodified — "
                                 "including the raw per-skill relation counts, "
                                 "which are the summaries that correctly "
                                 "detected this failure",
            "stopping_rule": "per arm, PASS at two consecutive checkpoints; "
                             "ceiling 100k never extended",
            "recovery_metrics": "identical to Condition C, plus swap-move "
                                "movement statistics (proposed / accepted / "
                                "assignment-changed, time to first accepted "
                                "swap)",
            "primary_question": "does supplying the assignment transition make "
                                "the joint posterior mix, and do the chains "
                                "then agree?",
        },

        "collision_ordering_resolution_2026_08_18": {
            "question": "which transition runs first when a sweep is scheduled "
                        "for BOTH the existing collapsed move and the new swap",
            "already_specified_in_this_registration": True,
            "frozen_ordering": [
                "scheduled global skill transposition",
                "collapsed U move if scheduled",
                "exact FFBS refresh of ALL (S, z)",
                "conditional U row sweep"],
            "why_the_conservative_fallback_is_unnecessary":
                "the requirement is that no PATH-DEPENDENT operation may run "
                "between a swap attempt and the FFBS refresh. The only "
                "transition that can intervene is the collapsed U move, whose "
                "acceptance ratio contains no conditional-likelihood term and "
                "which provably never reads the stored (S, z) — pinned by the "
                "validated test_collapsed_u_ordering.py, which shows its "
                "decision is byte-identical under a scrambled stored "
                "segmentation. The first path-dependent transition in the "
                "sweep is the conditional U row phase, which runs strictly "
                "after the refresh.",
            "invariant_asserted_at_every_checkpoint":
                "ffbs_refreshes_after_swap == swap_attempts",
            "composition_function": "matched_condition_c_prime."
                                    "ConditionCPrimeChain.advance, which "
                                    "delegates each sweep to the validated "
                                    "ConditionCChain.advance",
        },
        "amendable_before_launch": ["swap_cadence", "arm set", "seeds"],
        "stop_condition": "this task freezes the protocol only; Condition C "
                          "must reach its own terminal state before any C' "
                          "chain is launched",
    }
    (OUT / "preregistration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"frozen: {OUT / 'preregistration.json'}")
    print(f"  any non-identity permutation a symmetry? "
          f"{invariance['any_non_identity_symmetry']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
