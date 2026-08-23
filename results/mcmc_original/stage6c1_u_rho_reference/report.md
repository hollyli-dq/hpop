# Stage 6C1 — exact reference

Source commit `05ef2c291fa84bbba9960064836729d317d66c31`. Built without any MCMC (`uses_mcmc: False`).

Target: `p(P, rho | Y) proportional to p(rho) L(P) pi_rho(P)`

## Grids

- rho: 81 points on [0.001, 0.994]
- prior cell masses: 40,000,000 prior draws, seed 20250811, common random numbers across rho

## Coverage and refinement

| coordinate | grid | integrates to | outer-boundary mass |
|---|---|---|---|
| rho | 81 points on [0.0010, 0.9940] | 1.00000000 | 0.0227 |

Halving the grid resolution moves:

- `rho_mean_fine`: 0.325252
- `rho_mean_coarse`: 0.325085
- `rho_mean_abs_change`: 1.669e-04
- `rho_sd_abs_change`: 1.182e-05
- `poset_probability_max_abs_change`: 6.989e-122

## Posterior over orders

- catalogue size: 4231
- MAP poset: 4002 (probability 1.00000000)
- true poset: 4002, probability 1.00000000, rank 1
- MAP is the true poset: yes

| rank | poset | probability | relations |
|---|---|---|---|
| 1 | 4002 | 1.00000000 | 6 |
| 2 | 4109 | 2.449e-118 | 7 |
| 3 | 4108 | 1.220e-313 | 8 |
| 4 | 0 | 0.00000000 | 10 |
| 5 | 2821 | 0.00000000 | 8 |

## Scalar marginals

- rho: mean 0.3253, sd 0.2198, median 0.2940, 95% [0.0147, 0.7919]

## Structural prior audit (§2.1 gate)

- max abs error vs scipy MVN logpdf: 3.553e-15
- max abs error of the closed-form log determinant: 8.882e-16
- single-row quadrature mass: {'0.1': 1.0, '0.5': 1.0000000000000002, '0.9': 0.999999999999992}
- row-factorisation max abs error: 3.553e-15
- a rho-dependent combinatorial normaliser is needed: no

Negative control — deleting `-(m/2) log|Sigma_rho|` moves the rho posterior mean from 0.1592 to 0.1428 (shift 0.0164), so the normaliser is load-bearing.

## Catalogue validation

- size 4231 (expected 4231, matches: yes)
- duplicate keys: 0
- all entries are partial orders: yes
- closure/reduction round-trip complete: yes
- representatives induce their filed order: yes
- ranking tuples enumerated: 14,400 of 14,400

## Exact likelihood

- MLE poset: 4002, true poset: 4002, MLE is true: yes
- clear of the runner-up by 271.5 nats
