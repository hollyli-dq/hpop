# Stage 6C1 — joint inference of U and rho

Source commit `05ef2c291fa84bbba9960064836729d317d66c31`, Stage 6C config hash `c1545b0b24acad47...`.

4 chains x 20,000 sweeps, 5,000 burn-in, thinning 5 -> 12,000 retained draws pooled (3,000 per chain). Wall clock 12.0 min.

## Chain starts (none at the truth)

| chain | U start | relations at start | rho | beta | seed |
|---|---|---|---|---|---|
| 0 | antichain | 0 | 0.05 | n/a | 0 |
| 1 | total_order | 10 | 0.30 | n/a | 1 |
| 2 | sparse | 3 | 0.60 | n/a | 2 |
| 3 | dense | 10 | 0.90 | n/a | 3 |

## Sampler correctness — agreement with the frozen reference

| gate | value | threshold | verdict |
|---|---|---|---|
| full_u_total_variation | 1.224e-118 | 0.0100 | PASS |
| max_relation_marginal_error | 2.449e-118 | 0.0100 | PASS |
| rho_rhat | 1.0066 | 1.0100 | PASS |
| mixed_reference_envelope | 0.0013 | 0.0030 | PASS |

The reference was built and frozen before these chains ran (`results/mcmc_original/stage6c1_u_rho_reference`) and was not adjusted afterwards.

- pooled retained draws compared: 12,000
- iid reference draws: 20,000
- mixed discrete/continuous energy distance 0.0013 against a 99% reference-vs-reference envelope of 0.0030 (1 coordinates, 25 constant ones dropped)

## Scalars

| scalar | mean | sd | median | 95% interval | R-hat | bulk ESS | tail ESS | MCSE | KS vs reference |
|---|---|---|---|---|---|---|---|---|---|
| rho | 0.3132 | 0.2133 | 0.2817 | [0.0126, 0.7736] | 1.0066 | 1027.4 | 957.3 | 0.0067 | 0.0249 |

Log posterior: R-hat 1.0022, bulk ESS 1041.3.

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

- full-U total variation vs reference: 1.224e-118
- max relation-marginal error: 2.449e-118
- max reduction-marginal error: 2.449e-118
- worst single chain relation error: 2.449e-118

Relation count is constant at 6 across every chain and draw. R-hat and ESS are **undefined**, not passing: the poset posterior is a point mass, so there is no structural variation left to mix over.

## Acceptance

| parameter | total | post burn-in |
|---|---|---|
| U | 0.3176 | 0.3174 |
| rho | 0.8580 | 0.8586 |

Post burn-in acceptance per chain:

- chain 0: U 0.3153, rho 0.8603
- chain 1: U 0.3263, rho 0.8636
- chain 2: U 0.3163, rho 0.8585
- chain 3: U 0.3118, rho 0.8520
