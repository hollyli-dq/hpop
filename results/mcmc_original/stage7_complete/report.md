# Step 7 — status and the Local-vs-FFBS comparison

* Step 7A (FFBS correctness against exact enumeration): **PASS**
* Step 7B0 (joint integration parity and smoke): **PASS**
* Step 7B1 (full joint against the frozen mixed reference): **PASS**
* Step 7B2 (full-corpus mixing comparison): **COMPARED against the frozen baseline**

## Step 7B1 — one posterior, two kernels

| gate | FFBS | LocalMoveKernel | threshold |
|---|---|---|---|
| segmentation_total_variation | 0.00659591 | 0.00460535 | 0.01 |
| segmentation_total_variation_vs_sampled_estimator | 0.00644299 | 0.00541639 | 0.01 |
| max_boundary_marginal_error | 0.00304837 | 0.00151631 | 0.01 |
| max_occurrence_label_marginal_error | 0.00557847 | 0.0023641 | 0.01 |
| induced_h_total_variation | 0.00720684 | 0.00531032 | 0.01 |
| max_relation_marginal_error | 0.00390061 | 0.00387953 | 0.01 |
| segment_count_total_variation | 0.00279293 | 0.000910193 | 0.01 |
| mixed_multivariate_reference_statistic | 0.00287902 | 0.00398879 | 0.004522256615497353 |
| log_posterior_rhat | 1.00008 | 1.00012 | 1.01 |
| n_segments_rhat | 1.00002 | 1.00004 | 1.01 |
| relation_count_rhat | 1.00006 | 1.00014 | 1.01 |
| selected_boundary_indicator_rhat | 1.00004 | 1.00007 | 1.01 |
| selected_relation_indicator_rhat | 1.00001 | 1.00002 | 1.01 |
| rho_rhat | 1.00007 | 1.00023 | 1.01 |
| beta_rhat | 1.00189 | 1.00256 | 1.01 |
| omega_rhat | 1.00032 | 1.00135 | 1.01 |
| lambda_rep_rhat | 1.00325 | 1.00237 | 1.01 |
| lambda_back_rhat | 1.00131 | 1.00063 | 1.01 |

| movement / cost | FFBS | LocalMoveKernel |
|---|---|---|
| wall seconds (4 chains, parallel) | 17144 | 6467 |
| segment-count lag-1 autocorrelation | +0.0807 | +0.0949 |
| relation-count lag-1 autocorrelation | +0.4023 | +0.4026 |
| segment-count bulk ESS | 96,257 | 99,666 |
| minimum bulk ESS over all coordinates | 1,302 | 1,217 |
| minimum ESS / second | 0.1 | 0.2 |
| distinct induced-H states visited | 4640 | 4658 |
| structural mode transitions | 187,478 | 187,331 |
| worst R-hat | 1.00325 | 1.00256 |

Both samplers clear every gate of the same independent reference. The difference is in the kernel's movement and its cost, not in the distribution they target.

## Step 7B2 — the full-corpus comparison

**FFBS DOES NOT ESCAPE the (S,z)-U structural locking: its chains satisfy the same registered conjunction as the baseline's. Falsified, precisely stated: the hypothesis that exact global FFBS updates of (S,z) ALONE resolve the full-joint structural locking. FFBS itself did what it was validated to do — exact conditional sampling of (S,z) (Step 7A), small-reference posterior correctness (Step 7B1), the same target as the LocalMoveKernel — and better (S,z) mixing simply does not propagate to U/H structural mixing, as the freeze manifest anticipated it might not.**

Baseline verdict at its decision point: FAIL / MULTIMODAL per interpretation_rule.json: (S,z)-U structural locking at the registered maximum. FFBS worst invariant R-hat: 2.9399716130923884e+16.

| registered criterion | FFBS (50k) | LocalMoveKernel (frozen final) |
|---|---|---|
| A_frozen_structure | holds | holds |
| B_disagreeing_structure | holds | holds |
| C_log_posterior_gap | holds | holds |

| equal sweeps (both first 50k) | FFBS | LocalMoveKernel |
|---|---|---|
| wall seconds | 95879 (estimated) | 34567 |
| worst invariant R-hat | 29399716130923884.00000 | 9.03748 |
| min invariant bulk ESS | 4 | 4 |
| distinct induced-H states (pooled) | 5 | 6 |
| structural mode transitions | 1 | 2 |
| chains frozen (criterion A basis) | 4 | 4 |

Equal-time and full tables: `comparison_local_vs_ffbs.json`. recovery stays uninterpreted for any sampler whose chains have not converged, exactly as the baseline's interpretation rule registers.

### Conclusion

What is falsified is not FFBS — it performed exactly as validated (exact conditional sampling of (S,z), Step 7A; same-target posterior correctness, Step 7B1; strong global segmentation movement here) — but the hypothesis that exact global FFBS updates of (S,z) alone resolve the full-joint structural locking. Better (S,z) mixing does not propagate to U/H structural mixing: with the segmentation refreshed globally every sweep, 4/4 chains remain structurally frozen, and the U-proposal audit (`stage7b2_u_audit/`) locates the wall in the target itself — cross-cell proposals are frequent (P(H' != H) ~ 0.5) but the conditional likelihood rejects them at 1e-11 to 1e-20. The locking is therefore not kernel locality but strong posterior coupling between the labelled segmentation and the reusable partial-order structure.

This diagnosis motivates the partially collapsed structural update introduced next, which integrates out the labelled segmentation when evaluating latent-U proposals.

