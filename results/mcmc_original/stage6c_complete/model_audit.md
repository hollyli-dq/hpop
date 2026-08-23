# Stage 6C0 — audit of `U`, `rho`, and the structural prior

Read from the repository and the vendored reference implementation before any Stage 6C
code was written. **The registered model is materially different from the discrete-poset
model the Stage 6C brief assumes**, and that difference changes the design of almost every
later section. Everything below cites the file it came from.

## 0. Headline

| brief assumes | registered model actually is |
|---|---|
| MCMC state is a poset / relation set | MCMC state is a **continuous matrix** `U ∈ R^{m×d}` |
| moves add/delete/toggle/reverse relations | moves are a **symmetric Gaussian random walk on one row of `U`** |
| proposals must preserve legality, may be asymmetric | **every real `U` induces a legal partial order**; the proposal is exactly symmetric, Hastings ratio 1 |
| `p(U | rho) ∝ rho^{r(U)}(1-rho)^{M-r(U)}`, needing a combinatorial `Z_m(rho)` | `p(U | rho)` is a **product of multivariate-normal row densities**, already fully normalised |
| a rho-dependent normaliser is probably missing | the rho-dependent normaliser is `log det Sigma_rho`, and it is **already present and exact** |

The §2.1 hard gate is therefore **satisfied**, but for a completely different reason than
the brief anticipates. There is no combinatorial normalising constant in this model
because the structural prior is not a relation-counting score.

## 1. One `U` or one `U_k` per skill?

**One `U`.** For Stage 6 the frozen configuration records `skills_per_block = 1`,
`segmentation = "none — boundaries are known and fixed"`, and a single
`U_TRUE` of shape `(5, 2)` (`recurrent_synthetic.py`). All 500 training blocks are
generated from that one `U_TRUE`. So `K = 1`, and the Stage 6C1/6C2 targets are the
single-`U` forms given in the brief's §3.1/§3.2. The per-skill factorisation of §4 does
not apply and is not used; no `U_1 × ... × U_K` tensor is ever formed because there is
only one `U`.

`sampler_u.run_u_mcmc` is already written for "one skill's `U`" (its own docstring), and
`sampler_segmentation.py:201` is what loops it over `u[k]` in the multi-skill Stage 5
setting. Stage 6C reuses the single-`U` entry point.

## 2–4. Representation, MCMC state, canonicalisation

`latent_poset.py` is explicit:

> the latent variable is `U_k ∈ R^{m_k × d}`, one row per role … The induced strict partial
> order is coordinate-wise dominance:  `i > j  iff  U_k[i, r] > U_k[j, r]` for every
> `r = 1, …, d`.

so:

- **Representation of the order**: `precedence_from_u(U)` returns the **transitive
  closure** as a boolean `(m, m)` matrix. Its docstring states this outright — "already
  transitively closed … the full precedence relation, not a Hasse/cover relation" — and
  that "the transitive reduction is … deliberately not computed here". There is no
  transitive-reduction code anywhere in `mcmc_original`.
- **The MCMC state is `U` itself**, the real matrix — not a poset id, not an adjacency
  list, not a bit mask. The poset is a *derived* quantity.
- **Canonicalisation is not required and does not exist.** Two different `U` matrices
  inducing the same order are not "two representations of one state" that must be merged;
  they are genuinely distinct points of the state space with different prior density. The
  closure matrix is nevertheless a perfectly good *canonical label for the induced order*,
  and Stage 6C uses `precedence_from_u(U).tobytes()` as the catalogue key. That is a
  reporting device, not the sampler's state.
- **Legality is automatic.** As `latent_poset.py` puts it: irreflexivity, asymmetry and
  transitivity "are inherited from `>` on the reals coordinate by coordinate … moving `U`
  can only ever move between valid partial orders." There is no acyclicity check to
  perform and no illegal proposal to reject.

**Consequence for the brief's §7.** "Preserve partial-order legality", "canonicalise the
resulting state", and "the number of legal moves may differ between `U` and `U'`" are all
vacuous here: the move is `U'[j,:] = U[j,:] + sigma_U · eta`, `eta ~ N(0, I_d)`, whose
density is symmetric in `(U, U')` by construction. `log q(U|U') - log q(U'|U) = 0`
**exactly**, not approximately, and this is proved rather than assumed in
`test_stage6c_u_kernel.py`.

## 5–7. `rho`: support, sharing, and where it acts

From `sampler_u.sigma_rho_matrix`:

```
Sigma_rho = (1 - rho) I_d + rho 1 1^T
```

with eigenvalues `1 + (d-1)rho` (once) and `1 - rho` (`d-1` times), so positive
definiteness requires

```
-1/(d-1) < rho < 1      ->   for d = 2:  -1 < rho < 1
```

**`rho` is not confined to (0,1) by the covariance.** It is confined by the *prior*: the
vendored `StatisticalUtils.dRprior(rho, fac)` is a `Beta(1, fac)` density truncated at
`1 - tol` (`tol = 5e-3`) and renormalised by subtracting the log CDF at that point;
`stage5.py:100` registers `RHO_PRIOR = 1.0`, which is the `fac` argument, so the
registered prior is

```
rho ~ Beta(1, 1) truncated to (0, 1 - 5e-3)   =   Uniform(0, 0.995)
```

- **Shared, not skill-specific**, for Stage 6C: there is one `U`, so the question is moot;
  in the Stage 5 multi-skill code `rho` is likewise a single scalar passed to every
  skill's row sweep (`sampler_segmentation.py:201`).
- **`rho` appears only in `p(U | rho)`.** It enters `Sigma_rho` and nothing else. The
  recurrent likelihood reads `U` only through `precedence_from_u(U)` and never sees `rho`.
  A test asserts that changing `rho` does not change the recurrent log likelihood.

## 8. What the structural prior counts

**Nothing structural.** It is not a count of closure relations, Hasse edges, or candidate
ordered pairs. It is a Gaussian density on the latent coordinates:

```
p(U | rho) = prod_{i=1}^{m} N(U[i,:]; 0, Sigma_rho)
```

`rho` controls the *correlation between the d coordinates of a role*, which in turn
controls how likely the induced order is to be dense: at `rho -> 1` the two coordinates
coincide, every pair becomes comparable and the order tends to a total order; at
`rho -> 0` the coordinates are independent and the order is sparse. So `rho` does have a
structural meaning, but it acts through the geometry of the embedding, not through a
relation-counting exponent.

## 2.1 The rho-normalisation gate — **PASSED, and here is why**

The brief's worry is that the code might hold only an unnormalised score
`\tilde p(U|rho)`, whose `rho`-dependent normaliser cancels when `rho` is fixed but not
when `rho` is inferred. That failure mode cannot occur here, because the prior is a
*density on a fixed continuous space*, not a score on a combinatorial set. Its normaliser
is the Gaussian one and it is written out explicitly. `sampler_u.log_u_prior`:

```python
chol    = np.linalg.cholesky(sigma)
log_det = 2.0 * float(np.log(np.diag(chol)).sum())
...
return -0.5 * (m * d * math.log(2.0 * math.pi) + m * log_det + quadratic)
```

The `m * log_det` term **is** the `rho`-dependent normaliser, and it is present. In closed
form,

```
log det Sigma_rho = (d - 1) log(1 - rho) + log(1 + (d - 1) rho)
```

which the vendored `log_U_prior_optimized` writes identically as
`(K-1)*log1p(-rho) + log1p(-rho + rho*K)`. Two independent implementations agree.

**Validation performed** (`test_stage6c_structural_prior.py`):

1. `log_u_prior` integrates to 1 over `R^{m×d}` — checked by high-accuracy quadrature for
   `m=1, d=2` and by Monte Carlo for `m=5, d=2`, across the `rho` grid.
2. It agrees with `scipy.stats.multivariate_normal.logpdf` summed over rows, to 1e-10.
3. It agrees with the vendored closed-form `log_U_prior_optimized` to 1e-10.
4. **Negative control**: an implementation with the `m * log_det` term deleted is shown to
   give a *different and wrong* `rho` posterior — it shifts the `rho` marginal mean by a
   large, quantified amount. This is the analogue of the Stage 6B1 missing-Jacobian
   control, and it demonstrates that the normaliser is not decorative.

**No conjugacy is claimed for `rho`.** The conditional `p(rho | U)` under
`Uniform(0,0.995) × prod_i N(U_i; 0, Sigma_rho)` is not a standard family, so `rho` is
updated by Metropolis-Hastings on a logit random walk with the explicit Jacobian, not by a
Gibbs step.

## 9–10 and §2.2. Does `rho_true` exist? **NO.**

`recurrent_synthetic.py`:

```python
U_TRUE = np.array([[4.0, 4.0], [0.0, 5.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0]])
```

This is **hand specified**, not drawn from `p(U | rho_true)`. It is chosen to realise one
particular poset — `assert_stage6_library` asserts the closure is exactly
`{(0,2),(0,3),(0,4),(2,3),(2,4),(3,4)}` with role 1 incomparable to everything — and its
entries (0 to 5) are wildly atypical of `N(0, Sigma_rho)`, whose coordinates have unit
variance. There is no `rho_true` anywhere in the generator, and no `RHO_TRUE` constant
exists in the repository.

Therefore, per the brief's §2.2:

```
rho sampler correctness : testable, and tested
rho synthetic recovery  : NOT APPLICABLE — no generating value exists
```

Two further points, stated now rather than after seeing results:

- Even if a `rho_true` existed, this dataset carries **one** `U` draw, i.e. 5 rows of a
  2-dimensional Gaussian. That is 5 effective observations for `rho`. `rho` would be
  weakly identified by construction.
- Because `U_TRUE` was hand-picked and is far out in the tail of the prior, the posterior
  for `rho` is driven by how well `Sigma_rho` explains *those specific* coordinates. This
  is a statement about the prior fitting a fixed point, not about recovery.

**No more favourable dataset was generated.** The registered Stage 6 corpus is retained
exactly as-is.

## Consequences for the Stage 6C design

1. **The sampler state is `U` (continuous) and `rho`** — plus `beta` in 6C2. The poset
   catalogue is used for the *reference* and for *reporting*, never as the chain's state.
2. **The U proposal is Stage 2A's `propose_row`, reused unchanged.** Symmetric, so the
   Hastings term is exactly zero. No legality filter, no canonicalisation step.
3. **The exact reference is possible and is genuinely exact in the likelihood**, because
   the likelihood depends on `U` only through the induced order, and for `m = 5, d = 2`
   the reachable set of orders is **complete and finite**: enumerating all `5! × 5! =
   14,400` pairs of coordinate rankings yields **4231 distinct labelled posets**, which is
   exactly the number of labelled partial orders on 5 elements. Every poset on ≤ 5
   elements has order dimension ≤ 2, so nothing is unreachable. The catalogue is therefore
   exhaustive by enumeration, not by sampling.
4. **The only Monte Carlo in the reference is the prior cell mass**
   `pi_rho(P) = P_{U ~ N(0,Sigma_rho)}[h(U) = P]`, which uses the prior alone — no data,
   no likelihood, no chain. Its standard error is reported alongside every gate it feeds.
5. **Detailed balance on a tiny complete state space** (brief §7) is performed on the
   *induced-order chain* of a reduced model, since the actual state space is continuous
   and has no finite transition matrix.

## What this audit did not find

No model-definition blocker. `p(U | rho)` is well defined and normalised, the `rho` prior
is registered, and the likelihood branch is the frozen Stage 6B one
(`recurrent-rfs-utility-weighted-frontier-v1`, hash
`9ad850f22065d85f6cfd855443395d06b8a566e8c8dcdd4f9d85b1f031e911cc`, `epsilon = 0.02`,
`q_0 = zeros(m)`, oracle blocks, oracle labels, `T = 20`, 500 training blocks).
Stage 6C proceeds.
