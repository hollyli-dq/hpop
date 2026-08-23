# FULL-LATENT — mid-run truth recovery analysis

**MID-RUN EXPLORATORY DIAGNOSTIC — NOT A REGISTERED CONVERGENCE GATE**

Generating truth was unsealed mid-run at the PI's explicit instruction. No formal experimental setting was subsequently changed.

## Provenance

- analysed at: `2026-08-22T11:18:22.470098Z`
- git commit: `3244ba43d8bcf8ca16819351019d83057351613a`
- corpus hash: `dd280a4a09896154e167f388edd401a9119ba398167c09404aba5f7743e58ec2`
- truth hash: `fc41538fd44d170df8d0a6401f0c6e6b49d52418c487e22f9e4f45ee047f903e`
- truth-unseal event: `results/mcmc_original/matched_full_latent/TRUTH_UNSEAL_midrun.json`
- live worker processes at analysis time: **8**
- sampler/runner sources modified: **False**
- analysis was READ-ONLY: it opened checkpoints and the frozen corpus for reading only, started/stopped nothing, and imported no module the live workers use (registered R-hat/ESS and probe definitions are vendored verbatim in `scripts/_midrun_vendored_registered.py`, digests re-verified at import).
- no gate file, checkpoint, seed, start, threshold, prior, cadence, scale or datum was written or altered.

## Checkpoints analysed

| arm | chain | sweep | retained | hours | s/sweep | checkpoint sha256 |
|---|---|---|---|---|---|---|
| FULL-COND | 0 | 18,000 | 1600 | 10.716 | 2.1433 | `c1c7e3b2ca002f76` |
| FULL-COND | 1 | 18,000 | 1600 | 10.719 | 2.1439 | `28833909aa7ab8b5` |
| FULL-COND | 2 | 18,000 | 1600 | 10.62 | 2.124 | `171d86a3b09d91dc` |
| FULL-COND | 3 | 18,000 | 1600 | 10.618 | 2.1235 | `4346ef24d87fb61f` |
| FULL-MARG | 0 | 16,000 | 1200 | 11.01 | 2.4772 | `f4740b457d046dba` |
| FULL-MARG | 1 | 16,000 | 1200 | 11.033 | 2.4824 | `7812bc80c5682e67` |
| FULL-MARG | 2 | 16,000 | 1200 | 11.13 | 2.5042 | `a4b5a0d6372bd807` |
| FULL-MARG | 3 | 16,000 | 1200 | 10.934 | 2.4602 | `503e27285efd9cfd` |

## Generating truth

- K = 3 skills over 5 roles A–E
- relations per skill (true index order) = [6, 6, 5]; sorted = [5, 6, 6]; total = 17
- pi* = [0.5, 0.3, 0.2]
- P* = [[0.0, 0.65, 0.35], [0.3, 0.0, 0.7], [0.75, 0.25, 0.0]]
- canonical library sha256 = `312c8f3f1a3de7aa`

## MID-RUN EXPLORATORY RECOVERY — FORMAL CONVERGENCE NOT YET ESTABLISHED

| Metric | FULL-COND | FULL-MARG |
|---|---|---|
| sweeps per chain | 18,000 | 16,000 |
| max invariant R-hat | ~1e15 (degenerate: chains frozen at different constants) | 2.412 |
| max invariant R-hat, excluding frozen-degenerate | 2.834 | 1.260 |
| invariants with >=1 exactly-frozen chain | 4 | 4 |
| log-target R-hat | 2.834 | 1.260 |
| total-relations R-hat | 3.095 | 2.308 |
| chains whose dominant library == TRUE (last 400) | 0 / 4 | 4 / 4 |
| closure F1 (mean, last 400) | 0.737 | 1.000 |
| closure Hamming (mean, last 400) | 8.25 | 0.00 |
| boundary Brier (pooled, 100 traces) | 0.0772 | 0.0538 |
| co-skill Brier (pooled, 256 probes) | 0.1207 | 0.0749 |
| pi total-variation error | 0.1217 | 0.0566 |
| P Frobenius error | 0.1659 | 0.1255 |
| P off-diagonal RMSE | 0.0677 | 0.0512 |
| sec / sweep | 2.134 | 2.481 |
| sweeps / hour | 1687 | 1451 |

### Chain-by-chain exact library status (last 400 retained draws)

| arm | chain | distinct states | dominant occupancy | dominant == TRUE | exact-draw fraction | closure F1 | Hamming |
|---|---|---|---|---|---|---|---|
| FULL-COND | 0 | 1 | 1.00 | no | 0.00 | 0.500 | 16.00 |
| FULL-COND | 1 | 3 | 1.00 | no | 0.00 | 0.667 | 10.00 |
| FULL-COND | 2 | 4 | 1.00 | no | 0.00 | 0.970 | 1.00 |
| FULL-COND | 3 | 4 | 1.00 | no | 0.00 | 0.812 | 6.00 |
| FULL-MARG | 0 | 1 | 1.00 | YES | 1.00 | 1.000 | 0.00 |
| FULL-MARG | 1 | 1 | 1.00 | YES | 1.00 | 1.000 | 0.00 |
| FULL-MARG | 2 | 1 | 1.00 | YES | 1.00 | 1.000 | 0.00 |
| FULL-MARG | 3 | 4 | 1.00 | YES | 1.00 | 1.000 | 0.00 |

### Example traces (first of each length class)

| trace | J | true blocks | COND E[blocks] | MARG E[blocks] | COND Brier | MARG Brier |
|---|---|---|---|---|---|---|
| 0 | 24 | 3 | 3.451 | 3.567 | 0.0801 | 0.0624 |
| 1 | 32 | 6 | 5.427 | 5.52 | 0.0453 | 0.0141 |
| 2 | 40 | 7 | 6.283 | 6.502 | 0.0838 | 0.0576 |
| 3 | 48 | 8 | 7.92 | 7.905 | 0.0589 | 0.0241 |

### Registered-invariant R-hat / ESS (current draws)

| invariant | COND R-hat | COND bulk ESS | MARG R-hat | MARG bulk ESS |
|---|---|---|---|---|
| log_target | 2.83 | 4.603 | 1.26 | 11.1773 |
| total_relations | 3.09† | 4.8842 | 2.31† | 4.9748 |
| total_segments | 1.48 | 7.4776 | 1 | 4620.8082 |
| mean_segments_per_trace | 1.48 | 7.4776 | 1 | 4620.8082 |
| mean_segment_length | 1.48 | 7.4807 | 1 | 4623.8255 |
| sd_segment_length | 1.13 | 19.4306 | 1.01 | 399.3517 |
| pi_entropy | 1.84 | 5.7533 | 1 | 4765.8588 |
| pi_l2 | 1.85 | 5.7162 | 1 | 4748.2107 |
| P_frobenius | 1.25 | 11.4228 | 1.11 | 22.9678 |
| P_trace2 | 1.57 | 6.845 | 1.1 | 24.6447 |
| P_trace3 | 1.57 | 6.8481 | 1.1 | 24.5659 |
| sorted_relation_counts[0] | 4.51† | 4.2385 | 1† | 397.3501 |
| sorted_relation_counts[1] | 4.63e+15† | 4.0201 | 2.41† | 4.8767 |
| sorted_relation_counts[2] | 2.55† | 4.7902 | 1.65† | 6.5908 |
| sorted_pi[0] | 1.67 | 6.3417 | 1 | 4946.7639 |
| sorted_pi[1] | 1.6 | 6.6705 | 1 | 4567.346 |
| sorted_pi[2] | 1.6 | 6.6483 | 1 | 4714.235 |
| sorted_P_row_entropy[0] | 1.35 | 9.1964 | 1.04 | 73.9467 |
| sorted_P_row_entropy[1] | 1.08 | 30.2614 | 1.11 | 23.5626 |
| sorted_P_row_entropy[2] | 1.11 | 22.5585 | 1.03 | 116.377 |
| sorted_stationary[0] | 1.56 | 6.9081 | 1.01 | 3191.8621 |
| sorted_stationary[1] | 1.09 | 27.75 | 1.03 | 85.987 |
| sorted_stationary[2] | 1.58 | 6.7485 | 1.01 | 3366.2856 |

† = at least one chain is EXACTLY constant over all retained draws while the chains disagree. The rank-normalized statistic then diverges numerically; the magnitude is meaningless but the condition itself is the finding (structural coordinates frozen).


## Figures

- `fig_A_invariant_traces.png` / `fig_A_invariant_traces.pdf`
- `fig_B_running_rhat.png` / `fig_B_running_rhat.pdf`
- `fig_C_exact_library_states.png` / `fig_C_exact_library_states.pdf`
- `fig_D_truth_vs_learned_posets.png` / `fig_D_truth_vs_learned_posets.pdf`
- `fig_E_example_boundaries.png` / `fig_E_example_boundaries.pdf`
- `fig_F_boundary_recovery.png` / `fig_F_boundary_recovery.pdf`
- `fig_G_coclustering_recovery.png` / `fig_G_coclustering_recovery.pdf`
- `fig_H_pi_recovery.png` / `fig_H_pi_recovery.pdf`
- `fig_I_P_recovery.png` / `fig_I_P_recovery.pdf`
- `fig_J_cost.png` / `fig_J_cost.pdf`

## Interpretation guard

- Neither arm is called converged here. Registered gates fire only at 30k/50k/75k/100k sweeps and require two consecutive PASSes; none has been evaluated.
- Recovery toward truth and posterior convergence across chains are separate questions. A chain can sit on the true library and still not be converged.
- Both arms target the SAME posterior. Differences below are exploration efficiency and mixing, not a different model or a better likelihood.
