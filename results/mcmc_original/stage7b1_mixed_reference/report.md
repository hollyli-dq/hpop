# Step 7B1 — FFBS inside the full recurrent joint sampler

Status: **PASS**. Step 7B2 (full-corpus mixing comparison): see its own directory.

The `(S, z)` update is an exact FFBS Gibbs draw; every other kernel, the target, the corpus, the starts, the scales, the run length, the gates and the comparison code are Stage 6E1B's, unchanged. The reference is Stage 6E1B's frozen mixed QMC + exact-enumeration reference; no new reference was built.

## The frozen reference, verified from the artifact

| statistic | value | threshold | verdict |
|---|---|---|---|
| max_rqmc_standard_error | 5.6679e-04 | &le; 0.001 | PASS |
| max_half_width_95 | 1.2081e-03 | &le; 0.0025 | PASS |
| min_relative_ess | 4.3861e-02 | &ge; 0.02 | PASS |
| max_normalised_weight | 3.2958e-04 | &le; 0.001 | PASS |
| log_evidence_sd | 3.3824e-03 | &le; 0.05 | PASS |
| max_replicate_h_total_variation (superseded, descriptive) | 7.8396e-03 | 0.003 | FAIL — reported as such |
| max_replicate_relation_departure (superseded, descriptive) | 3.8914e-03 | 0.003 | FAIL — reported as such |

Label permutation audit: no nontrivial relabelling is a symmetry of this target, so per-skill summaries are well posed and raw per-skill R-hat is meaningful

## Gates — the Stage 6E1B gates, unchanged

| gate | FFBS | LocalMoveKernel (frozen) | threshold | verdict |
|---|---|---|---|---|
| segmentation_total_variation | 0.00659591 | 0.00460535 | 0.01 | PASS |
| segmentation_total_variation_vs_sampled_estimator | 0.00644299 | 0.00541639 | 0.01 | PASS |
| max_boundary_marginal_error | 0.00304837 | 0.00151631 | 0.01 | PASS |
| max_occurrence_label_marginal_error | 0.00557847 | 0.0023641 | 0.01 | PASS |
| induced_h_total_variation | 0.00720684 | 0.00531032 | 0.01 | PASS |
| max_relation_marginal_error | 0.00390061 | 0.00387953 | 0.01 | PASS |
| segment_count_total_variation | 0.00279293 | 0.000910193 | 0.01 | PASS |
| mixed_multivariate_reference_statistic | 0.00287902 | 0.00398879 | 0.004522256615497353 | PASS |
| log_posterior_rhat | 1.00008 | 1.00012 | 1.01 | PASS |
| n_segments_rhat | 1.00002 | 1.00004 | 1.01 | PASS |
| relation_count_rhat | 1.00006 | 1.00014 | 1.01 | PASS |
| selected_boundary_indicator_rhat | 1.00004 | 1.00007 | 1.01 | PASS |
| selected_relation_indicator_rhat | 1.00001 | 1.00002 | 1.01 | PASS |
| rho_rhat | 1.00007 | 1.00023 | 1.01 | PASS |
| beta_rhat | 1.00189 | 1.00256 | 1.01 | PASS |
| omega_rhat | 1.00032 | 1.00135 | 1.01 | PASS |
| lambda_rep_rhat | 1.00325 | 1.00237 | 1.01 | PASS |
| lambda_back_rhat | 1.00131 | 1.00063 | 1.01 | PASS |

Worst R-hat: FFBS 1.003246, LocalMoveKernel 1.002556, threshold 1.01.

## Same target, different kernel

| quantity | FFBS | LocalMoveKernel |
|---|---|---|
| sweeps | 600,000 | 600,000 |
| retained (pooled) | 192,000 | 192,000 |
| wall seconds (4 chains, parallel) | 17147 | 6467 |
| seconds per sweep (per chain) | 0.0284 | 0.0108 |
| segmentation acceptance | 1.000 (Gibbs, by construction) | 0.174 relabel / 0.093 split / 0.069 shift |
| boundary changes (chain 0) | 768,628 | 780,733 |
| occurrence-label changes (chain 0) | 3,198,798 | 5,053,011 |

Neither sampler produces a better posterior: both are compared against the same independent reference and both must clear the same gates. What differs is the transition kernel and its cost.

## Scalar ESS and convergence

| coordinate | FFBS bulk ESS | LocalMoveKernel bulk ESS | FFBS R-hat | LocalMoveKernel R-hat |
|---|---|---|---|---|
| log_posterior | 27,942 | 24,910 | 1.00008 | 1.00012 |
| n_segments | 96,257 | 99,666 | 1.00002 | 1.00004 |
| relation_count | 30,657 | 30,026 | 1.00006 | 1.00014 |
| selected_boundary_indicator | 60,125 | 57,741 | 1.00004 | 1.00007 |
| selected_relation_indicator | 50,142 | 47,408 | 1.00001 | 1.00002 |
| rho | 20,288 | 18,659 | 1.00007 | 1.00023 |
| beta | 1,945 | 1,902 | 1.00189 | 1.00256 |
| omega | 3,839 | 3,879 | 1.00032 | 1.00135 |
| lambda_rep | 1,302 | 1,217 | 1.00325 | 1.00237 |
| lambda_back | 6,249 | 6,048 | 1.00131 | 1.00063 |

## Cost

* FFBS sweep: 19.8 ms — candidate tables 5.0 ms, charts 4.2 ms, backward draws 0.10 ms, parameter phase the remainder
* candidate tables built once per sweep: True; 25% of chain wall time
* LocalMoveKernel baseline: 10.8 ms per sweep

Source commit `77093cb5845a9a2dc3472203657fa62fc6222164`.
