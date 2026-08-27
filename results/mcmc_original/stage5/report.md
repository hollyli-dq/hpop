# Stage 5 — static multi-trace recovery (mcmc-original-latent-poset @ 08e2cb6d)

Date 2026-08-10 · Python 3.13.2 · NumPy 2.4.6

Stage 5A reuses the **verified** vendored `mcmc_simulation_po` per skill; no update
rule for `U`, `rho` or `beta` is reimplemented. Stage 5B reuses the Stage-4
`LocalMoveKernel` unchanged, initialised without oracle information.

## Configuration

- `n_skills` = 4, `latent_dim` = 2 (passed as `fixed_K`; the old
  sampler's `K` is the *latent dimension*, never the library size)
- likelihood branch `log_successors_queue_jump` for generation **and** fitting
- `beta` = 1.5, `epsilon` = 0.05 (fixed), `delta_B` = 0.5
- generator seed 0; corpus {'n_train': 40, 'n_train_extra': 0, 'n_test': 20, 'instance_counts': {'A': 43, 'D': 37, 'F': 60, 'E': 41}}

## Stage 5A — oracle segmentation, infer U/rho/beta and P

| skill | n | acc | rho | beta | latent_dim | true pairs | inferred | exact |
|---|---|---|---|---|---|---|---|---|
| A | 43 | 0.743 | 0.877 | 0.921 | [2] | [(0, 1)] | [(0, 1)] | yes |
| D | 37 | 0.720 | 0.165 | 1.059 | [2] | (none) | (none) | yes |
| F | 60 | 0.743 | 0.916 | 0.652 | [2] | [(0, 1)] | [(0, 1)] | yes |
| E | 41 | 0.483 | 0.011 | 2.486 | [2] | [(0, 2), (0, 3)] | [(0, 2)] | **no** |

Transition matrix: **MAE 0.0654**, max |err| 0.1500, mean row KL 0.0324

## Stage 5B — oracle U and P, infer segmentations

- **Boundary F1 = 0.9655** (P 0.9459, R 0.9859, 70/71 cuts)
- Skill ARI = 0.6670, occurrence accuracy 0.8333

### The ARI decomposed against the Bayes ceiling

| quantity | value |
|---|---|
| Stage 5B ARI, all 20 traces | 0.6670 |
| Stage 5B ARI, the 15 exactly-correct-boundary traces | **0.8872** |
| Oracle ceiling (true boundaries, exact forward-backward label posterior) | **0.8961** |

Conditional on correct boundaries the sampler **attains the oracle ceiling**. The
shortfall against the registered 0.85 gate is therefore two effects, neither of which
is a sampler defect: (i) the ceiling is not 1.0 because skills A and D are
support-matched by design; (ii) 5 of 20 traces have boundary errors.

Oracle recall by skill: A 0.917, D 0.429, F 1.000, E 1.000

| move | proposed | accepted | rate |
|---|---|---|---|
| relabel | 377,792 | 154,963 | 0.4102 |
| split | 169,109 | 30,674 | 0.1814 |
| merge | 326,194 | 30,679 | 0.0941 |
| shift | 155,193 | 85 | 0.0005 |

## Skill E audit — why (0,3) is weakly recovered

41 oracle E instances. Candidate-structure log-likelihood by beta:

| beta | H_true | H_-03 | H_-02 | H_empty | argmax |
|---|---|---|---|---|---|
| 0.5 | -101.02 | -108.63 | -120.03 | -130.30 | H_true |
| 1.0 | -96.82 | -102.92 | -113.91 | -130.30 | H_true |
| 1.5 | -94.76 | -98.54 | -108.96 | -130.30 | H_true |
| 2.0 | -94.22 | -95.43 | -105.16 | -130.30 | H_true |
| 2.5 | -94.64 | -93.48 | -102.39 | -130.30 | H_-03 |
| 3.0 | -95.59 | -92.49 | -100.51 | -130.30 | H_-03 |
| 5.0 | -99.49 | -94.37 | -98.64 | -130.30 | H_-03 |

The ranking **flips between beta 2.0 and 2.5**: above it the structure missing
(0,3) wins. Pinning beta at its true value resolves the pair; inferring it does not:

| beta | mean P(0>3) |
|---|---|
| pinned at 1.5 | **0.981** |
| inferred jointly | **0.438** |

So (0,3) is identified *conditional on beta*; the joint (U, rho, beta) posterior
dissolves it. This is neither a mixing failure (chains started at the true U also
leave) nor missing data (P(0>2) = 1.000 throughout).

## Deviations and open items

- Stage 5B's registered ARI gate (0.85) is **not met** on the full split (0.667). Not
  restated as a pass: the gate is left as registered and the decomposition reported.
- Skill E's (0,3) is **not recovered** under joint beta inference. Reported as open.
- `beta` was pinned in the audit using only the sampler's public parameters
  (`init_state` + a ~zero `softmax_beta_stepsize`); the vendored code is unmodified.
- No joint S+U+P sampler was run. Stages 5A and 5B remain separate by design.

