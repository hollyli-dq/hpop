# Stage 6E1A — exact segmentation-only reference: PASS

Every continuous coordinate is fixed. `S` and `z` are the only latent quantities, so the
posterior lives on a finite set that can be enumerated exactly — there is no Monte Carlo
error on the reference side, and no reference sampler to be wrong in the same way as the
one under test.

**This is a sampler-correctness result only.** The generating segmentation is recorded in
`config.json` and enters no comparison.

## The problem, frozen before any draw

| | |
|---|---|
| trace length `J` | 8 |
| skills `K` | 3 |
| roles `m` / latent columns `d` | 3 / 2 |
| block width | `3 <= w <= 12` (registered) |
| `delta_B` | 0.15 (registered) |
| fixed | `U` (three distinct induced orders), `rho = 0.3`, `beta = 1.5`, `omega = 1.7346`, `lambda_rep = 0.8`, `lambda_back = 0.25`, `pi` uniform, `P` uniform off-diagonal |
| legal states | **21** — width `>= 3` over `J = 8` admits only `L in {1, 2}` |

Selection was registered before sampling: scan seeds `0..49` ascending, take the **first**
meeting `max p < 0.90` and `#{p > 0.01} >= 3`. Seed **0** met both, so no search occurred.
No recovery quantity entered the choice.

## Gates

| gate | value | threshold | verdict |
|---|---:|---:|---|
| `log Z` enumeration vs forward recursion | 8.882e-16 | 1e-10 | PASS |
| enumerated weight vs registered direct target | 3.553e-15 | 1e-12 | PASS |
| fast kernel vs `Stage6EMoveKernel`, all 21 states | 0.000e+00 | 1e-15 | PASS |
| kernel-reachable support vs enumerated support | 21 of 21 | equal | PASS |
| nondegeneracy: `max p(S,z\|x)` | 0.4748 | < 0.90 | PASS |
| nondegeneracy: `#{p > 0.01}` | 7 | >= 3 | PASS |
| retained draws | 380,000 | >= 100,000 | PASS |
| **TV( path )** | **1.470e-03** | 0.01 | **PASS** |
| **max boundary-marginal error** | **2.839e-04** | 0.01 | **PASS** |
| **max occurrence-label marginal error** | **1.066e-03** | 0.01 | **PASS** |
| max labelled-segment marginal error | 9.929e-04 | 0.01 | PASS |
| segment-count TV | 6.776e-05 | 0.01 | PASS |
| max expected-transition-count error | 1.109e-04 | 0.01 | PASS |

All 13 pass. Per-chain path TV: 0.00278, 0.00493, 0.00361, 0.00170 — every chain clears
the gate on its own, not only pooled.

## `log Z` by two routes that share no code path

```
enumeration        -7.939698706440
forward recursion  -7.939698706440
gap                 8.882e-16
```

Route 1 lists all 21 states and scores each by the registered decomposition. Route 2
recurses over positions and never materialises a state, using the fact that `L-1` cuts
each cost `log delta_B` while a width-`w` segment contributes `w-1` non-cut internal
positions, so `sum_l (w_l - 1) = J - L` reproduces `targets.log_boundary_prior` exactly.

The forward recursion is **not** a sampler and is not FFBS. It cross-checks `log Z` and
nothing else; the reference `(S, z)` draws in Stage 6E1B come from the *enumerated*
conditional, and the production segmentation update remains the registered move kernel.

## Move-type behaviour

| move | proposed | accepted | rate | impossible |
|---|---:|---:|---:|---:|
| relabel | 400,499 | 202,939 | 0.50672 | 0 |
| split | 400,068 | 24,265 | 0.06065 | 30,755 |
| merge | 399,974 | 24,269 | 0.06068 | 368,936 |
| shift | 399,459 | 7,900 | 0.01978 | 368,570 |

Merge and shift are *impossible* from a one-segment state, and the posterior puts 0.923 on
`L = 1`, so they are unavailable most of the time. That is the registered kernel behaving
correctly rather than a defect: an empty neighbourhood contributes no-op mass to
`q(current -> current)`, `proposal_distribution` accounts for it, and the Hastings ratios of
the *other* move types are formed from that correctly normalised `q`. Split and merge
accept at rates equal to four significant figures (0.06065 and 0.06068 over ~400,000
proposals each), which is the detailed-balance signature one wants to see between a move
and its reverse.

## One honest qualification

The segment-count distribution is concentrated: `p(L = 1) = 0.9228`, `p(L = 2) = 0.0772`.
The registered nondegeneracy criteria are about the *state* distribution and are met
comfortably (7 states above 1%, no state above 48%), but the number-of-segments coordinate
is not richly explored by this problem, because `J = 8` with a minimum width of 3 admits
only `L in {1, 2}` at all. The `J <= 8` bound is §8's, and the width bound is registered,
so this is a property of the specification rather than a choice made here. Stage 6E1B and
Stage 6E2 exercise the segment-count coordinate over longer traces.

## Artifacts

```
config.json            registered problem, selection scan, chain configuration
exact_reference.npz    all 21 states, log weights, probabilities, both log Z, all marginals
chains.npz             pooled and per-chain empirical distributions and marginals
comparison.json        every gate, per-chain TV, top states, acceptance by move type
```

Runtime: 23.0 s for 4 chains x 400,000 sweeps (15 us per Metropolis step).
