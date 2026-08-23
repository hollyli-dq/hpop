# Stage 6C2 — joint inference of U, rho and beta

Source commit `05ef2c291fa84bbba9960064836729d317d66c31`, Stage 6C config hash `c1545b0b24acad47...`.

4 chains x 20,000 sweeps, 5,000 burn-in, thinning 5 -> 12,000 retained draws pooled (3,000 per chain). Wall clock 15.7 min.

## Chain starts (none at the truth)

| chain | U start | relations at start | rho | beta | seed |
|---|---|---|---|---|---|
| 0 | antichain | 0 | 0.05 | 0.80 | 0 |
| 1 | total_order | 10 | 0.30 | 1.20 | 1 |
| 2 | sparse | 3 | 0.60 | 1.90 | 2 |
| 3 | dense | 10 | 0.90 | 2.60 | 3 |

## Sampler correctness — agreement with the frozen reference

| gate | value | threshold | verdict |
|---|---|---|---|
| full_u_total_variation | 3.627e-115 | 0.0100 | PASS |
| max_relation_marginal_error | 7.255e-115 | 0.0100 | PASS |
| rho_rhat | 1.0029 | 1.0100 | PASS |
| mixed_reference_envelope | 0.0026 | 0.0034 | PASS |
| beta_rhat | 1.0000 | 1.0100 | PASS |
| beta_ks | 0.0112 | 0.0500 | PASS |

The reference was built and frozen before these chains ran (`results/mcmc_original/stage6c2_u_rho_beta_reference`) and was not adjusted afterwards.

- pooled retained draws compared: 12,000
- iid reference draws: 20,000
- mixed discrete/continuous energy distance 0.0026 against a 99% reference-vs-reference envelope of 0.0034 (2 coordinates, 25 constant ones dropped)

## Scalars

| scalar | mean | sd | median | 95% interval | R-hat | bulk ESS | tail ESS | MCSE | KS vs reference |
|---|---|---|---|---|---|---|---|---|---|
| rho | 0.3231 | 0.2216 | 0.2876 | [0.0120, 0.8004] | 1.0029 | 896.6 | 1310.4 | 0.0074 | 0.0124 |
| beta | 1.4965 | 0.0317 | 1.4965 | [1.4339, 1.5591] | 1.0000 | 10005.8 | 10183.9 | 3.170e-04 | 0.0112 |

Log posterior: R-hat 1.0035, bulk ESS 1166.8.

## Structural recovery

- true poset index: 4002
- MAP poset index: 4002 (is the true poset)
- posterior probability of the true poset: 1.0000
- posterior rank of the true poset: 1
- unique orders visited: 1
- minimum true-relation probability: 1.0000
- maximum false-relation probability: 0.0000

| representation | precision | recall | F1 | structural Hamming |
|---|---|---|---|---|
| closure | 1.0000 | 1.0000 | 1.0000 | 0 |
| reduction | 1.0000 | 1.0000 | 1.0000 | 0 |

- full-U total variation vs reference: 3.627e-115
- max relation-marginal error: 7.255e-115
- max reduction-marginal error: 7.255e-115
- worst single chain relation error: 7.255e-115

Relation count is constant at 6 across every chain and draw. R-hat and ESS are **undefined**, not passing: the poset posterior is a point mass, so there is no structural variation left to mix over.

## Acceptance

| parameter | total | post burn-in |
|---|---|---|
| U | 0.3181 | 0.3169 |
| rho | 0.8538 | 0.8521 |
| beta | 0.4401 | 0.4405 |

Post burn-in acceptance per chain:

- chain 0: U 0.3176, rho 0.8533, beta 0.4421
- chain 1: U 0.3218, rho 0.8487, beta 0.4409
- chain 2: U 0.3165, rho 0.8549, beta 0.4394
- chain 3: U 0.3117, rho 0.8518, beta 0.4397
