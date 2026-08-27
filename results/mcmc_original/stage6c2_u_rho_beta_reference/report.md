# Stage 6C2 — exact reference

Source commit `05ef2c291fa84bbba9960064836729d317d66c31`. Built without any MCMC (`uses_mcmc: False`).

Target: `p(P, rho, beta | Y) proportional to p(rho) p(beta) L(P, beta) pi_rho(P)`

## Grids

- rho: 81 points on [0.001, 0.994]
- beta: 241 points on [1.0, 2.2]
- prior cell masses: 40,000,000 prior draws, seed 20250811, common random numbers across rho

## Coverage and refinement

| coordinate | grid | integrates to | outer-boundary mass |
|---|---|---|---|
| rho | 81 points on [0.0010, 0.9940] | 1.00000000 | 0.0227 |
| beta | 241 points on [1.0000, 2.2000] | 1.00000000 | 3.394e-59 |

Halving the grid resolution moves:

- `rho_mean_fine`: 0.325252
- `rho_mean_coarse`: 0.325085
- `rho_mean_abs_change`: 1.669e-04
- `rho_sd_abs_change`: 1.182e-05
- `poset_probability_max_abs_change`: 2.070e-118
- `beta_mean_abs_change`: 1.211e-12
- `beta_sd_abs_change`: 2.191e-13

## Posterior over orders

- catalogue size: 4231
- MAP poset: 4002 (probability 1.00000000)
- true poset: 4002, probability 1.00000000, rank 1
- MAP is the true poset: yes

| rank | poset | probability | relations |
|---|---|---|---|
| 1 | 4002 | 1.00000000 | 6 |
| 2 | 4109 | 7.255e-115 | 7 |
| 3 | 4108 | 1.106e-287 | 8 |
| 4 | 4006 | 3.713e-306 | 7 |
| 5 | 0 | 0.00000000 | 10 |

## Scalar marginals

- rho: mean 0.3253, sd 0.2198, median 0.2940, 95% [0.0147, 0.7919]
- beta: mean 1.4961, sd 0.0319, 95% [1.4336, 1.5591]