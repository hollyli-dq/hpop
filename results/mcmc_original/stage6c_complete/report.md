# Stage 6C — recurrent latent-poset MCMC

Model `recurrent-rfs-latent-product-order-v1`, config hash `c1545b0b24acad47...`, source commit `05ef2c291fa84bbba9960064836729d317d66c31`.

Stage 6C makes `U` and `rho` latent (6C1) and then additionally `beta` (6C2). The target is **continuous in U**:

```
p(U, rho[, beta] | Y)  proportional to  p(Y | h(U), fixed) p(U | rho) p(rho) [p(beta)]
```

The chain's state is the real matrix `U`; `h(U)` is a derived label. See `model_audit.md` in this directory for the Stage 6C0 audit that settled the model.

## Verdicts

| stage | sampler correctness | U recovery | rho recovery | beta recovery |
|---|---|---|---|---|
| 6C1 | PASS | PASS | NOT APPLICABLE | — |
| 6C2 | PASS | PASS | NOT APPLICABLE | PASS |

These are four separate questions and are deliberately not combined. `rho` recovery is NOT APPLICABLE because U_TRUE is hand specified in recurrent_synthetic.py, not drawn from p(U | rho); no rho_true exists in the generator.

## Gates

### Stage 6C1

| gate | value | threshold | verdict |
|---|---|---|---|
| full_u_total_variation | 1.224e-118 | 0.0100 | PASS |
| max_relation_marginal_error | 2.449e-118 | 0.0100 | PASS |
| rho_rhat | 1.0066 | 1.0100 | PASS |
| mixed_reference_envelope | 0.0013 | 0.0030 | PASS |

### Stage 6C2

| gate | value | threshold | verdict |
|---|---|---|---|
| full_u_total_variation | 3.627e-115 | 0.0100 | PASS |
| max_relation_marginal_error | 7.255e-115 | 0.0100 | PASS |
| rho_rhat | 1.0029 | 1.0100 | PASS |
| mixed_reference_envelope | 0.0026 | 0.0034 | PASS |
| beta_rhat | 1.0000 | 1.0100 | PASS |
| beta_ks | 0.0112 | 0.0500 | PASS |

## Stage 6C1 vs Stage 6C2 — what freeing beta changes

| quantity | 6C1 (beta fixed) | 6C2 (beta free) |
|---|---|---|
| posterior probability of the true poset | 1.0000 | 1.0000 |
| distinct orders visited | 1 | 1 |
| closure F1 | 1.0000 | 1.0000 |
| min true-relation probability | 1.0000 | 1.0000 |
| max false-relation probability | 0.0000 | 0.0000 |
| rho posterior mean | 0.3132 | 0.3231 |
| rho posterior sd | 0.2133 | 0.2216 |

**Structural uncertainty does not change when beta is freed.** Both stages put probability 1.0 on the true order and visit exactly one order after burn-in, so no new structural mode appears and there is no U/beta confounding to report: the likelihood separates the true order from its nearest competitor by 271.5 nats, which no value of beta in the posterior's support can overturn.

The 6C2 beta posterior is 1.4965 +/- 0.0317, against the Stage 6B1 reference posterior of 1.4961 +/- 0.0319 obtained with **U held fixed at the truth**. Freeing U and rho therefore costs beta essentially no precision — another consequence of the structure being sharply identified.

## Gate shortfalls and caveats

- **Retained-sample count is below the §12 target.** The registered protocol (4 chains x 20,000 sweeps, 5,000 burn-in, thinning 5) yields 12,000 pooled retained draws per stage, against the §12 request for at least 100,000. The §13 continuation ceiling of 60,000 sweeps would still reach only 44,000, so the two clauses cannot both be satisfied as written. The registered run protocol was followed and the shortfall is recorded here rather than resolved by rescaling. It does not affect the structural gates, which are satisfied by 100+ orders of magnitude, and the scalar gates carry their own ESS and MCSE.
- **No rho marginal KS threshold was pre-registered.** §12 asks for one; the gates registered before the runs were TV, relation-marginal error, R-hat and the calibrated mixed-reference envelope (which includes the rho coordinate). The observed rho KS distances are reported descriptively against the distribution-free reference value 1.36/sqrt(bulk ESS); no threshold was fitted after seeing them. The beta KS gate (0.05) *was* registered in code before Stage 6C2 ran.
- **Relation-count R-hat and ESS are undefined, not passing.** The poset posterior is a point mass, so the relation-count trace is constant at 6 in every chain. The diagnostics report this as `degenerate` and emit `null` rather than a flattering R-hat of 1.0, and correlations involving the relation count are reported as undefined with the reason attached.
- **The exact reference's one Monte Carlo ingredient is the prior cell mass.** Everything else is exact enumeration. `pi_rho(P)` used 40,000,000 prior draws with common random numbers across the rho grid; the maximum standard error over all 4231 posets and 81 rho values is 1.44e-05.

## Why rho is weakly identified

The poset posterior is effectively a point mass on the true order, and `rho` does not enter the likelihood at all. So

```
p(rho | Y)  proportional to  p(rho) * pi_rho(P_true)
```

and the entire `rho` posterior is driven by how the prior cell mass of one poset varies with `rho`, on 5 rows of a 2-dimensional Gaussian. `pi_rho(P_true)` is around 1e-4 and decreases gently in `rho`, which is a weak signal by construction. This is a property of the registered experiment, not a defect of the sampler, and it is why `rho` identifiability is reported separately from sampler correctness.

## Artifacts

| artifact | path | size |
|---|---|---|
| 6c1_smoke | `results/mcmc_original/stage6c1_u_rho_smoke` | 6 KB |
| 6c1_reference | `results/mcmc_original/stage6c1_u_rho_reference` | 9914 KB |
| 6c1_full | `results/mcmc_original/stage6c1_u_rho_full_seed0` | 177 KB |
| 6c2_smoke | `results/mcmc_original/stage6c2_u_rho_beta_smoke` | 8 KB |
| 6c2_reference | `results/mcmc_original/stage6c2_u_rho_beta_reference` | 11158 KB |
| 6c2_full | `results/mcmc_original/stage6c2_u_rho_beta_full_seed0` | 329 KB |
| figures | `results/mcmc_original/stage6c_complete/figures` | 1084 KB |

## Not started

Stage 6D, unknown-boundary inference, segmentation inference, skill-label inference, semi-Markov FFBS, Step 7 and real-data experiments are **not started** in this stage.
