# Stage 6E1B — mixed unknown-boundary reference: PASS (18/18 gates)

Stage 6E1A fixed every continuous coordinate. Stage 6E1B frees `U`, `rho` and the four
recurrent scalars as well, so the object under test is the complete mixed posterior

```
p(S, z, U, rho, beta, omega, lambda_rep, lambda_back | x)
```

and the reference has to represent the discrete and the continuous parts at once.

**This is a sampler-correctness result only.** The generating segmentation and the
generating `U` are recorded in `config.json` and enter no comparison. The posterior here
is deliberately weakly identified — 16 observations over 23 latent dimensions — because a
sharp posterior would make the comparison easy for the wrong reason.

## Construction

Scrambled Sobol in **prior coordinates**, the strategy Stage 6D1 validated, so the proposal
is exactly the joint prior and the importance weight carries no prior density. What the
weight carries instead is the *marginal* segmentation likelihood: for every QMC point,
every legal `(S, z)` is enumerated and summed, giving

```
Z_n(theta) = sum over all legal (S_n, z_n) of w(S_n, z_n | theta)
w(theta)   = prod_n Z_n(theta)
```

in closed form. That is what makes a mixed reference affordable: there are only `O(J^2 K)`
distinct candidate blocks per trace, they are computed once per draw, and the enumerated
state set has 21 members per trace.

| | |
|---|---|
| traces | 2, each `J = 8` |
| skills / roles / latent columns | `K = 3`, `m = 3`, `d = 2` |
| enumerated states per trace | 21 |
| QMC dimension | 23 = 1 (`rho`) + 18 (`U`) + 4 (scalars) |
| replicates x points | 16 x 2^20 = 16,777,216 draws |
| `pi`, `P` | **fixed**, and deliberately asymmetric |

### Why `pi` and `P` are fixed, and why they are asymmetric

§9 permits fixing them. They are not tractable to add: their conjugate update is defined
given *sampled* labels, so putting them in the QMC construction would need a Dirichlet
inverse CDF and `K` more nearly-unidentified dimensions for no gain in what this reference
tests. They are **inferred** in Stage 6E2, where the frozen Stage 3 update is the object
under test.

Their asymmetry is load-bearing rather than cosmetic. With a uniform `pi` and a symmetric
`P`, permuting the skill labels together with their `U_k` would be an exact symmetry of
this target, every per-skill summary would be unidentified, and comparing per-skill `H`
between reference and sampler would be meaningless. `label_permutation_audit` checks this
before any comparison is made and reports:

> no nontrivial relabelling is a symmetry of this target, so per-skill summaries are well
> posed and raw per-skill R-hat is meaningful

## Two estimators of the same reference

Given a QMC point, the conditional `p(S_n, z_n | theta, x)` is known **exactly** — it is the
enumerated weight vector normalised by `Z_n`. So there are two estimators of any
segmentation functional:

* **conditional** (Rao-Blackwellised): `sum_i w_i E[f | theta_i]`;
* **sampled**: `sum_i w_i f(S_i, z_i)`, with one `(S_i, z_i)` drawn from that conditional —
  §9's iid-equivalent construction.

They estimate the same quantity, and the sampled one is the conditional one plus a
multinomial draw, so it has strictly larger variance. The **conditional estimator is
registered as primary**: a reference must be more precise than the 0.01 budget it feeds,
and the extra variance is noise added to the reference side, not information. The sampled
estimator is computed and gated alongside it, so §9's construction is carried out and can
be seen to agree rather than being quietly replaced.

## Reference quality — the corrected Stage 6D protocol

| gate | value | threshold | status | verdict |
|---|---:|---:|---|---|
| max RQMC standard error | 5.668e-04 | 1e-3 | **PRIMARY** | PASS |
| max 95% half-width | 1.208e-03 | 2.5e-3 | **PRIMARY** | PASS |
| min relative ESS | 0.04386 | >= 0.02 | active | PASS |
| max normalised weight | 3.296e-04 | 1e-3 | active | PASS |
| log-evidence SD | 3.382e-03 | 0.05 | active | PASS |
| max replicate H total variation | 7.840e-03 | 3e-3 | **SUPERSEDED** | **FAIL** |
| max replicate relation departure | 3.891e-03 | 3e-3 | **SUPERSEDED** | **FAIL** |

`primary_pass = True`, `all_active_pass = True`.

The two superseded statistics **fail**, and are recorded as failures. They are not
relabelled, not averaged into a verdict, and not quietly dropped. They were superseded in
Stage 6D1 for a reason that applies here unchanged: a maximum over `R` replicates estimates
the dispersion of a **single** replicate, does not shrink as `R` grows, and is therefore
not an uncertainty for the replicate mean — which is the quantity the downstream comparison
actually consumes. The statistic that *is* an uncertainty for that quantity,
`rqmc_se = sd(estimates, ddof=1)/sqrt(R)`, passes with a factor of 1.8 in hand, and its
t-based half-width occupies 12% of the 0.01 error budget it feeds.

`log Z = -18.245464` with a standard deviation of 0.00338 across the 16 independent
scrambles.

## Nondegeneracy

| criterion | value | requirement | verdict |
|---|---:|---:|---|
| max `p(S, z \| x)` over states and traces | 0.3378 | < 0.90 | PASS |
| segmentation states above 1% (min over traces) | 14 | >= 3 | PASS |
| induced-`H` states above 1% (min over skills) | 13 | >= 3 | PASS |

The reference is genuinely uncertain in **both** the segmentation and the induced order,
which is what §9 requires and what makes the comparison capable of detecting a defect.

## The chains, and one recorded failed attempt

**Attempt 0** — 4 chains x 150,000 sweeps, 30,000 burn-in, thin 4 (120,000 pooled draws).
It cleared every gate except one:

```
induced_h_total_variation   0.010500   threshold 0.01   FAIL
```

It is preserved in full at
[`../stage6e1b_mixed_reference_FAILED_attempt0_150k/`](../stage6e1b_mixed_reference_FAILED_attempt0_150k/),
with its own README setting out the diagnosis. It is not deleted and not relabelled.

### Why that was Monte Carlo error and not bias

Every statistic that would move if the sampler targeted the wrong distribution was clean:
five posterior means within **0.024 reference standard deviations**, five standard
deviations within **0.8%**, the mixed multivariate energy statistic at z = 0.78 inside its
null, and all ten registered R-hats at or below 1.0042. What was *not* clean was the
effective sample size — **301** for `lambda_rep` out of 120,000 retained draws, an
integrated autocorrelation time near 400.

`TV(H)` is the statistic most exposed to that. It sums 19 cell errors per skill, so its
Monte Carlo noise is roughly `0.5 * sum_i sqrt(p_i / n)`, which at this effective size is
about 0.01 — precisely the observed value. The per-skill figures, 0.01037 / 0.01050 /
0.00667, are the signature of noise at that scale rather than of a systematic displacement.

### What changed for attempt 1, and what did not

**The gate stayed at 0.01 and was not widened.** The model, the target, the move kernel,
the proposal scales and the frozen reference are all untouched. The single change is the
number of draws — 150,000 sweeps to **600,000** — because that is the one intervention that
*distinguishes* the two explanations instead of hiding the difference: Monte Carlo error
falls as `1/sqrt(n)` and should land near 0.005, while a bias would not move at all.

The chains were also switched from sequential to parallel execution. Each chain has its own
registered seed and its own RNG stream, so that changes wall time and nothing else.

### Attempt 1 — 4 chains x 600,000 sweeps, 120,000 burn-in, thin 10 (192,000 pooled)

**Every gate passes.** And the prediction was quantitative, not just directional: with 4x
the draws, every distributional statistic fell by almost exactly `sqrt(4) = 2`.

| gate | attempt 0 | attempt 1 | ratio | threshold |
|---|---:|---:|---:|---:|
| **TV(H)**, max over skills | 0.010500 **FAIL** | **0.005310** | **1.98** | 0.01 |
| TV(S, z), max over traces | 0.009762 | 0.004605 | 2.12 | 0.01 |
| max relation-marginal error | 0.008051 | 0.003880 | 2.08 | 0.01 |
| max boundary-marginal error | 0.004474 | 0.001516 | 2.95 | 0.01 |
| max occurrence-label error | 0.004476 | 0.002364 | 1.89 | 0.01 |
| segment-count TV | 0.007128 | 0.000910 | 7.83 | 0.01 |

A bias does not shrink when you take more draws. Five independent statistics falling by a
factor consistent with `1/sqrt(n)` is the signature of Monte Carlo error and settles the
question the failed attempt raised.

The mixed multivariate statistic — an energy distance on
`[closure indicators, segment counts, standardised scalars]`, 25 non-constant coordinates —
sits at **0.003989** against a self-calibrated 99% envelope of **0.004523**. Every one of
the ten registered R-hats is at or below **1.00256**.

Per-skill `TV(H)`: 0.00408 / 0.00531 / 0.00391. Per-trace `TV(S, z)`: 0.00461 / 0.00390.

### Scalar marginals against the reference

| scalar | MCMC mean | reference mean | gap in reference SD | SD ratio | bulk ESS |
|---|---:|---:|---:|---:|---:|
| `rho` | +0.4735 | +0.4746 | 0.0038 | 0.9999 | 18,659 |
| `beta` | +1.7317 | +1.7321 | **0.0004** | 0.9911 | 1,902 |
| `omega` | -0.2661 | -0.2536 | 0.0063 | 1.0000 | 3,879 |
| `lambda_rep` | +0.5324 | +0.5390 | 0.0179 | 0.9971 | 1,217 |
| `lambda_back` | +0.9139 | +0.9056 | 0.0129 | 1.0044 | 6,048 |

Bulk ESS rose from 301 to 1,217 on the worst coordinate. Every posterior mean now agrees
with the reference to within **0.018 reference standard deviations**, and every standard
deviation to within 0.9%.

One honest qualification: the **per-chain** `TV(S, z)` values (0.0107, 0.0106, 0.0135,
0.0076 on trace 0) straddle 0.01. The registered gate is on the pooled estimate, which is
what four chains are for, and the R-hats confirm the chains agree with each other; but a
single chain of this length would not clear the gate on its own, and the reader should know
that.

## Artifacts

```
config.json                   registered problem, QMC coordinates, chain configuration
reference_registration.json   quality gates, superseded gates, nondegeneracy, label audit
qmc_summary.json              per-replicate log Z, ESS, weights, precision, scalars
reference_draws.npz           pooled segmentation/boundary/label/H/relation reference,
                              plus 8,000 iid-equivalent retained draws for the mixed
                              multivariate statistic
chains.npz                    four dispersed chains
segmentation_comparison.json  TV, boundary and label marginals, per-chain TV
structural_comparison.json    induced-H TV and relation marginals per skill
scalar_comparison.json        scalar marginals against the reference, with R-hat and ESS
joint_comparison.json         every gate, the mixed multivariate statistic, convergence
```
