# Stage 6D0 — audit of the joint oracle-block state

Read from the repository before any Stage 6D inference code was written, and before the
Stage 6D1 reference was constructed. Everything below cites the file it came from. The
Stage 6D brief was written against assumptions this model does not satisfy; §7 records
each divergence and the brief's own clause that resolves it.

## 0. Headline

| the brief allows for | the frozen model actually is |
|---|---|
| `U^(0)` global and `U^(a)` assessor utilities linked by `tau` | **one** `U ∈ R^{5×2}`; no assessor level, no `tau` |
| `rho ~ Beta(1, 1/6)` | `rho ~ Beta(1, 1)` truncated at `1 − 5e-3` = `Uniform(0, 0.995)` |
| a scaling proposal `rho' = 1 − (1−rho)·delta`, ratio `−log delta` | a **logit random walk** carrying `log(rho(1−rho))` |
| `K` = latent columns | the repository writes latent columns as `d`; `K` here means **skills** |

None of these is a defect. Each is the registered Stage 6C state, and the brief directs
that the repository state wins.

## 1. Every active `U` matrix

**One.** `stage6c_frozen.frozen_config()` records `n_skills = 1`, oracle boundaries and
oracle labels, and `recurrent_synthetic.U_TRUE` is a single `(5, 2)` matrix from which all
500 training blocks were generated. There is no `U_k` collection and no assessor level, so
the brief's per-skill product and its `U^(0)`/`U^(a)` hierarchy both have exactly one
factor and are not built.

## 2. Row and column meanings, and the three integers that get conflated

```
m = 5   rows of U       role occurrences
d = 2   columns of U    latent utility coordinates   (the brief calls this K)
K = 1   skills          one U matrix                 (the repository's meaning of K)
        assessors       none
```

These are three different numbers and the code is tested to keep them apart
(`test_stage6d_joint_target.py::test_row_count_latent_dimension_skills_and_assessors_are_distinct`).
`FrozenStage6D` exposes `n_roles = 5`, `latent_dimension = 2`, `n_skills = 1`,
`n_assessors = 0`.

## 3. The role of `rho`

From `sampler_u.sigma_rho_matrix`:

```
Sigma_rho = (1 - rho) I_d + rho 1 1^T
```

so `Var(U[j,k]) = 1` and `Cov(U[j,k], U[j,l]) = rho` for `k ≠ l`. **`rho` is the
equicorrelation between the `d` latent COLUMNS within one row.** Rows are conditionally
independent. `rho` is *not* a covariance between rows, *not* a Bernoulli edge probability,
*not* a prior over a catalogue of partial orders, and needs *no* sum over legal posets:
the prior is a density on `R^{m×d}`, so no combinatorial normaliser arises.

`rho` enters **only** `p(U | rho)`. It appears nowhere in the recurrent likelihood, which
reads `U` only through `h(U)`. A `rho` update therefore consumes zero likelihood
evaluations — asserted by test, and visible in the smoke run's replay counts.

Positive definiteness needs `−1/(d−1) < rho < 1`, i.e. `−1 < rho < 1` at `d = 2`. The
binding constraint is the *prior*, not the covariance: the registered support is
`(0, 0.995)`.

## 4. The role of `tau`

**There is no `tau`.** `tau_in_model` is `False` in both the Stage 6C and Stage 6D frozen
configurations, and no `TAU` constant exists anywhere in `mcmc_original`. The brief's
assessor-residual density

```
r_j^(a) = U_j^(a) − tau U_j^(0)
```

has no referent here, so the corresponding density parity test and the hierarchical
non-centred QMC construction are **not applicable** and are not built. This is recorded
rather than silently skipped.

## 5. The construction of `H = h(U)`

`latent_poset.precedence_from_u` returns the **transitive closure** of coordinate-wise
dominance:

```
i > j   iff   U[i, r] > U[j, r]   for every column r = 1, …, d
```

i.e. the intersection of the `d` column orderings. Properties that matter for Stage 6D:

- **`H` is derived, never state.** The MCMC state is the real matrix `U`. `H` is computed
  for reporting and for the reference comparison only.
- **`H` carries no second prior.** The target has `p(U | rho)` and nothing else structural.
  Adding a `p(H | rho)` term on top would double-count and is explicitly absent.
- **Legality is automatic.** Irreflexivity, asymmetry and transitivity are inherited from
  `>` on the reals, so every real `U` induces a legal strict partial order. There is no
  acyclicity check and no illegal proposal.
- The stored representation is the closure, not the cover relation; the transitive
  reduction is computed separately when a reduction-based statistic is reported.

## 6. Symmetries and non-identifiabilities

These are the reasons entrywise `U` recovery is not claimed anywhere in Stage 6D.

1. **Column permutation.** `h(U)` is the *intersection* of the column orderings, which is
   symmetric under permuting the `d` columns. `Sigma_rho` is exchangeable in the columns,
   so `p(U | rho)` is invariant too. The target is therefore invariant under any column
   permutation, and raw entrywise `U` traces may swap labels between chains without any
   convergence failure. Diagnostics use `H`, relation probabilities and
   permutation-invariant summaries instead.
2. **Common monotone reparameterisation within a column.** `h(U)` depends on each column
   only through its ranking, so any strictly increasing map applied to one column leaves
   `H` — and hence the likelihood — unchanged. The likelihood is **piecewise constant** in
   `U`: it speaks only at order boundaries. Only the prior distinguishes `U` values inside
   one order cell.
3. **Consequence for `rho`.** Because the likelihood is constant on a cell, the posterior
   for `rho` is driven by the prior cell mass alone. This is the mechanism behind Stage
   6C's weak `rho` identifiability and it carries into Stage 6D unchanged.

No attempt is made to remove these symmetries from the production target; the brief
forbids it, and doing so would change the object being validated.

## 7. Divergences from the Stage 6D brief, and the clause that resolves each

| § | brief assumes | frozen reality | resolving clause |
|---|---|---|---|
| 2 | `U^(0)`, `U^(a)`, `tau` | one `U`, no assessors, no `tau` | §2's own fallback: "If the actual Stage 6C model instead has skill-specific `U_k` matrices without assessor hierarchy, use that exact frozen state" |
| 3 | assessor-residual Gaussian term | no assessor level | not applicable; the global-row term is implemented and tested |
| 4 | `rho ~ Beta(1, 1/6)` | `Beta(1, 1)` truncated at `0.995` | §4: "unless the final Stage 6C configuration establishes another prior" — it does |
| 4 | scaling proposal, ratio `−log delta` | logit random walk, `log(rho(1−rho))` | §6 "reuse the frozen Stage 6C rho proposal/update" and §7.2 "require numerical equality" with it — adopting §4's proposal would fail the brief's own parity gate |
| 10.1 | hierarchical non-centred QMC for `U^(a)` | no assessor level | not applicable; the single-`U` non-centred construction `U = L(rho) Z` is built and tested |

The scaling proposal is **not discarded**: `stage6d_frozen.scaling_proposal` and
`scaling_proposal_log_ratio` implement it with the exact `−log(delta)` identity, tested,
and clearly marked as not used in production — so the mathematics the brief asks about is
pinned without displacing the kernel the brief also requires parity with.

## 8. Quantities fixed by oracle information

| fixed | value / source |
|---|---|
| block boundaries `S*` | oracle, from the registered synthetic corpus |
| skill labels `z*` | oracle; one skill, so every block uses the same `U` |
| `epsilon` | `0.02` |
| `q_0` | `zeros(m)`, reset at the start of **every** block |
| likelihood branch | `recurrent-rfs-utility-weighted-frontier-v1`, hash `9ad850f2…e911cc` |
| `tau` | does not exist |

Inferred in Stage 6D: `U`, `rho`, `beta`, `omega`, `lambda_rep`, `lambda_back`.

## 9. What this audit did not find

No model-definition blocker. `p(U | rho)` is a normalised density whose `rho`-dependent
term `−(m/2)log|Sigma_rho|` is present and independently reproduced by an analytic
determinant-and-inverse route (`stage6d_frozen.log_mvn_equicorrelated`, agreeing with the
Cholesky implementation to 1.4e-14). The `rho` prior and all four scalar priors are
registered. Stage 6D proceeds.
