# Condition C — paper-safe result ledger

**CONDITION C TERMINATED AT 75k — FORMAL VERDICT: NOT CONVERGED**

*Deviation.* Condition C was terminated after the third consecutive failed registered checkpoint at 75k sweeps, before the preregistered 100k ceiling. Because the protocol required two consecutive passing checkpoints and only one checkpoint remained, the registered convergence criterion was no longer attainable.

*Non-convergence warning.* Recovery summaries are descriptive diagnostics conditional on non-converged chains and are not interpreted as posterior estimates.

| # | claim (exact wording) | number | source artifact | formal/diagnostic | truth-free/recovery | primary/supplementary | caveat |
|---|---|---|---|---|---|---|---|
| 1 | Neither C-COND nor C-MARG satisfied the registered convergence criterion at 30k, 50k or 75k. | 6 of 6 registered gates FAIL | `condition_c_checkpoint_table.csv` | formal | truth-free | primary | the criterion required two consecutive passing checkpoints |
| 2 | Conditional chains remained separated across distinct structural libraries. | 3 libraries across 4 chains (split 2-1-1) | `condition_c_failure_geometry.json` | diagnostic | truth-free | primary | at the 75k terminal point |
| 3 | Path-marginal chains consistently agreed on the same unordered structural library. | 1 library, all 4 chains | `condition_c_failure_geometry.json` | diagnostic | truth-free | primary | at 30k, 50k and 75k |
| 4 | Path-marginal chains nevertheless remained split across anchored structure-to-skill assignments. | split 3-1 | `condition_c_failure_geometry.json` | diagnostic | truth-free | primary | stable across all three rungs |
| 5 | The anchored assignment gap remained approximately 124-125 nats in typical log target across the registered checkpoints. | 30k: 124.39; 50k: 124.44; 75k: 124.30 | `figure_data/condition_c_assignment_gap.csv` | diagnostic | truth-free | primary | typical log-target difference between modes; NOT a posterior odds ratio — basin volume is unknown |
| 6 | No C-MARG chain crossed the anchored assignment barrier after burn-in through the 75k terminal point. | 0 crossings in 4 chains x 65,000 post-burn-in sweeps | `condition_c_failure_geometry.json` | diagnostic | truth-free | primary | an observation within the registered budget, not an impossibility claim |
| 7 | Accepted path-marginal proposals are not anchored-assignment crossings. | 3,949 accepted, 32 changed the anchored tuple, 0 after burn-in | `figure_data/condition_c_movement.csv` | diagnostic | truth-free | primary | the H-changing accepts all occurred during burn-in |
| 8 | Path marginalisation substantially reduced structural disagreement but did not eliminate global anchored-assignment multimodality within the observed budget. | libraries 3 -> 1; assignment split unchanged at 3-1 | `condition_c_failure_geometry.json` | diagnostic | truth-free | primary | do not write 'solved joint inference' |
| 9 | Under the predefined recovery metrics, every C-MARG chain recovered the true unordered structural library; one of four recovered the true anchored assignment. | 4/4 library; 1/4 anchored | `recovery_75k.json` | diagnostic | recovery-based | primary | Recovery summaries are descriptive diagnostics conditional on non-converged chains and are not interpreted as posterior estimates. |
| 10 | Under the same metrics, one of four C-COND chains recovered the true library. | 1/4 | `recovery_75k.json` | diagnostic | recovery-based | primary | Recovery summaries are descriptive diagnostics conditional on non-converged chains and are not interpreted as posterior estimates. |
| 11 | Per-chain closure F1 separates cleanly by anchored assignment. | C-MARG [0.455, 0.455, 1.0, 0.455]; C-COND [0.248, 0.424, 1.0, 0.288] | `recovery_75k.json` | diagnostic | recovery-based | supplementary | Recovery summaries are descriptive diagnostics conditional on non-converged chains and are not interpreted as posterior estimates. |
| 12 | Held-out oracle-path NLL per occurrence reproduces the generating truth exactly for the single correctly-assigned chain in each arm. | truth 1.0772; C-MARG [3.7003, 3.7003, 1.0772, 3.7003]; antichain baseline 1.7528 | `recovery_75k.json` | diagnostic | recovery-based | supplementary | Recovery summaries are descriptive diagnostics conditional on non-converged chains and are not interpreted as posterior estimates. |

## Forbidden wording

Do **not** write any of the following:

- "path marginalisation solved joint inference"
- "C-MARG converged"
- "C-MARG converged up to permutation"
- "label switching"
- "the bad mode has posterior probability exp(-125) smaller" — basin volume is unknown, and the gap is a typical log-target difference, not an integrated posterior odds ratio
- "the existing move set cannot cross the barrier" — use the observational wording instead
