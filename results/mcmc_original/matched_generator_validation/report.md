# Matched synthetic generator — validation report

Source commit: `6eb43b355333d07d72d7dffd836b654413702b04` (branch `mcmc-original-latent-poset`)
Generator version: 1.0.0

The generator implements the registered factorization `p(U, rho) p(S, z | delta_B, pi, P) p(X | S, z, h(U), vartheta)` with the exact normalized segmentation prior `p(S | J, delta_B) = delta_B^(L-1) (1-delta_B)^(J-L) / C_J(delta_B)` over contiguous segmentations with widths in [3, 12], sampled by the suffix-DP `G(r)` recursion (exact by telescoping; see `matched_segmentation_prior.py`). The old Stage 6E2 block-count mechanism (`L ~ Uniform{4,5,6}` then iid widths) is preserved untouched in `stage6e_corpus.py` and banned from the new generator by a source-level regression test.

## Registered gates

| gate | value | threshold | pass |
|---|---|---|---|
| dp_normalizer_vs_enumeration_and_reference | 8.882e-16 | 1e-12 | PASS |
| exact_segment_count_two_route_max_error | 7.216e-16 | 1e-12 | PASS |
| exact_boundary_marginal_two_route_max_error | 3.886e-16 | 1e-12 | PASS |
| empirical_segment_count_tv_all_J | 5.357e-03 | 0.01 | PASS |
| empirical_boundary_marginal_max_error_all_J | 3.486e-03 | 0.01 | PASS |
| illegal_segmentation_count | 0 | 0 | PASS |
| tiny_full_state_tv | 2.034e-03 | 0.01 | PASS |
| pi_empirical_max_error | 1.100e-03 | 0.01 | PASS |
| transition_row_empirical_max_error | 1.175e-03 | 0.01 | PASS |
| self_transition_count | 0 | 0 | PASS |
| recurrent_q0_reset_and_block_independence | 3.553e-15 | 1e-10 | PASS |
| complete_data_log_prob_parity | 1.421e-14 | 1e-10 | PASS |
| tiny_logz_parity | 1.776e-15 | 1e-10 | PASS |
| all_negative_controls_detected | {'old_block_count_mechanism': True, 'recurrent_state_leakage': True, 'terminal_block_delta_factor': True, 'self_transitions': True} | all True | PASS |
| reproducibility_byte_identical_and_hashes | True | True | PASS |

## Verdict: ALL GATES PASS

No MCMC, no FFBS, no conditional- or collapsed-U inference, and no Condition A/B/C/D experiment was run. Artifacts in this directory are the complete registered validation record.
