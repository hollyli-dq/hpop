# Hour 2 — static BPOP frontier-softmax likelihood

Date: 2026-08-08
Branch: `mcmc-original-latent-poset`

## 1. Hour-1 prerequisite status

Checked before writing any Hour-2 code:

```
PYTHONPATH=src python -m pytest tests/mcmc_original -q
474 passed in 2.18s
```

`precedence_from_u` is present and passing, returning the transitive closure of the
strict partial order induced by coordinate-wise dominance. Hour 2 was built on top of
it without modifying it.

## 2. Files created / modified

Created:

```
src/hpop/mcmc_original/static_bpop.py
tests/mcmc_original/test_static_bpop.py
scripts/stage00_check_bpop.py
results/mcmc_original/hour02_report.md   (this file)
```

Modified:

```
src/hpop/mcmc_original/__init__.py       (additive: export the static_bpop API)
```

`latent_poset.py` and `types.py` were **not** touched. No file outside
`src/hpop/mcmc_original/` was modified; in particular the pre-existing
`hpop.inference.likelihood` is untouched and is not imported by the new module.

## 3. Mathematical definitions implemented

For a skill with roles `M = {0, ..., m-1}` and latent `U in R^{m x d}`, with
`precedence[i, j]` True iff `i > j` (i.e. **i is a predecessor of j** and must occur
first):

**Remaining set.** After emitting `y_1, ..., y_{t-1}`,

```
R_t = M \ {y_1, ..., y_{t-1}}
```

**Frontier.** The currently executable roles:

```
F_t = { x in R_t : no z in R_t \ {x} has precedence[z, x] }
```

i.e. `x` is executable iff none of its predecessors is still waiting. Returned in
ascending index order. A non-empty `R_t` always has a non-empty `F_t` under a valid
strict finite partial order; an empty frontier raises `RuntimeError`.

**Remaining successor count.** For `x in R_t`:

```
S_t(x) = #{ z in R_t \ {x} : precedence[x, z] }
```

Counted against the **full precedence relation** (the transitive closure), not a
transitive reduction, and restricted to roles still remaining — so it shrinks as the
execution proceeds.

**Successor utility.**

```
Q_t(x) = log(1 + S_t(x))
```

**Frontier softmax.** Normalised over the frontier only:

```
softmax_F(x) = exp(beta * Q_t(x)) / sum_{v in F_t} exp(beta * Q_t(v))
```

computed with the standard max-shift (`logits -= logits.max()`) for stability.

**Trembling-hand mixture.** The one-step distribution, for `x in R_t`:

```
p(y_t = x | y_<t, U, beta, eps) = (1 - eps) * 1[x in F_t] * softmax_F(x)
                                  + eps / |R_t|
```

and exactly 0 for `x` not in `R_t`. The two components carry mass `(1-eps)` and `eps`
respectively, so the step normalises by construction. With `eps > 0` every remaining
role keeps strictly positive probability, so order violations are never impossible.

**Complete likelihood.** For a complete execution `y` (a permutation of `M`):

```
log p(y | U, beta, eps) = sum_t log p(y_t | y_<t, U, beta, eps)
```

`bpop_likelihood` returns `exp` of this, mapping `-inf` to `0.0` rather than
overflowing.

## 4. Known fork example

```
U_FORK = [[ 1.0,  0.0],
          [ 0.0,  1.0],
          [-1.0, -1.0]]
```

induces

```
0 > 2,   1 > 2,   0 || 1     (0 and 1 incomparable)
```

verified directly against `precedence_from_u` in `test_fork_poset_structure`.

Frontiers:

| remaining | frontier |
|---|---|
| {0,1,2} | (0, 1) |
| {1,2} | (1,) |
| {0,2} | (0,) |
| {2} | (2,) |

Initial successor counts and utilities:

| role | S(x) | Q(x) |
|---|---|---|
| 0 | 1 | log 2 = 0.693147 |
| 1 | 1 | log 2 = 0.693147 |
| 2 | 0 | 0 |

One-step probabilities at `beta = 1.5`, `eps = 0.05`:

```
p(y_1 = 0) = 0.4916666667      (0.95 * 0.5 + 0.05/3)
p(y_1 = 1) = 0.4916666667
p(y_1 = 2) = 0.0166666667      (0.05/3, pure trembling hand)
sum        = 1.0000000000000000
```

Roles 0 and 1 have identical `Q`, so the frontier softmax is uniform over them for
**any** beta — the fork's initial split is beta-independent.

## 5. Exact permutation probabilities (fork, beta = 1.5, eps = 0.05)

| execution | probability | status |
|---|---|---|
| (0, 1, 2) | 0.479375000000 | linear extension |
| (1, 0, 2) | 0.479375000000 | linear extension |
| (0, 2, 1) | 0.012291666667 | order violation |
| (1, 2, 0) | 0.012291666667 | order violation |
| (2, 0, 1) | 0.008333333333 | order violation |
| (2, 1, 0) | 0.008333333333 | order violation |

**Total = 0.9999999999999999**, `|total - 1| = 1.110e-16`.

At `eps = 0` the same model gives `p(0,1,2) = p(1,0,2) = 0.5` exactly and 0 for the
other four, for every beta in `{0.0, 0.5, 1.5, 5.0}` — tested.

## 6. Confirmation that the complete likelihood sums to one

Confirmed. The likelihood is a proper sequential distribution over permutations, and
this is checked exhaustively rather than assumed:

- three known models (chain, fork, antichain) x 4 betas x 3 epsilons = **36** cases,
  each summing `m!` executions, all within `1e-10` of 1;
- **2880** randomised cases (see §7), all within `1e-10` of 1;
- a degenerate model with duplicated `U` rows (many mutually incomparable roles),
  across the full beta x epsilon grid;
- `all_permutation_probabilities` (the cached enumerator) agrees with the reference
  step-by-step `bpop_likelihood` path to `1e-15` on every permutation tested.

Additional exactness checks that passed:

- total order, `eps = 0`: `p(0,1,2,3) = 1` exactly, every other permutation 0;
- antichain: every one of the 3! executions has probability exactly 1/6, for every
  beta and every epsilon;
- `beta = 0`: the frontier choice is uniform, `1/|F_t|`;
- the three values the Stage-0 toy spec pins down reproduce exactly at
  `beta = 1.5, eps = 0.05`:
  `p_A((0,1)) = 0.975`, `p_B((0,1,2)) = 0.9425`, `p_B((2,0,1)) = 0.01625`.

## 7. Randomised cases checked

| check | grid | cases |
|---|---|---|
| full-likelihood normalisation | m in {2,3,4,5} x d in {1,2,3} x 20 draws x beta in {0.0,0.5,1.5,5.0} x eps in {0.0,0.01,0.1} | **2880** |
| one-step normalisation | same grid, m distinct execution prefixes each | **10080** |
| frontier non-emptiness | m in {2,3,4,5} x d in {1,2,3} x 20 draws, all 2^m - 1 non-empty remaining sets | all subsets |

Across the full-likelihood check, **109,440** individual permutation probabilities
were enumerated and summed.

## 8. New test counts

```
PYTHONPATH=src python -m pytest tests/mcmc_original/test_static_bpop.py -q
78 passed in 7.25s
```

78 new tests, 0 failures. Package total:

```
PYTHONPATH=src python -m pytest tests/mcmc_original -q
552 passed in 5.70s          (474 Hour 1 + 78 Hour 2)
```

## 9. Full project test status

```
PYTHONPATH=src python -m pytest -q --ignore=archive
615 passed, 1 warning, 36 subtests passed in 40.71s
```

615 = 63 pre-existing + 474 Hour 1 + 78 Hour 2. Zero failures, zero regressions.

Two pre-existing conditions, unchanged by this work and not repaired (both predate
Hour 1):

- a bare `pytest -q` from the repo root also collects `archive/tests/`, which fails
  with **8 collection errors** (`No module named 'hpop.posets'`, `'hpop.synthetic'`,
  `'hpop.model'`, ...). The archived package no longer exists. Use `--ignore=archive`
  or target `tests/`.
- `RuntimeWarning: divide by zero encountered in log` from
  `src/hpop/inference/semi_markov.py:36` during one pre-existing test.

## 10. Demonstration script output

`PYTHONPATH=src python scripts/stage00_check_bpop.py` runs clean and ends with

```
  total = 0.9999999999999999
  |total - 1| = 1.110e-16

[PASS] complete BPOP likelihood sums to 1
```

## 11. Numerical issues discovered

None that affect correctness.

1. The permutation total prints as `0.9999999999999999` rather than a bit-exact
   `1.0`. This is ordinary double-precision summation error at `1.1e-16`, eleven
   orders of magnitude inside the `1e-10` acceptance tolerance. It is not corrected
   or hidden.
2. The softmax uses a max-shift, so `beta = 5.0` with large successor counts does not
   overflow. Checked as part of the randomised grid.
3. `eps = 0` makes order-violating executions exactly zero-probability, so
   `bpop_log_likelihood` returns `-inf` (not a `math domain error`) and
   `bpop_likelihood` returns `0.0`. Tested directly.
4. `epsilon` is validated as `0 <= eps < 1`; `eps = 1` is rejected rather than
   silently producing a pure-noise model, matching the `SkillTemplate` contract from
   Hour 1.

## 12. Deviations from the Hour-2 specification

1. **One extra public function.** `all_permutation_probabilities(u, beta, epsilon)`
   was added beyond the specified API. Tasks 9 and 11 both require enumerating every
   permutation's probability, and doing that through the per-step path alone is
   `O(m! * m^3)` in Python, which made the required 2880-case grid slow. It caches the
   one-step distribution per remaining-set (there are only `2^m`) and is verified
   against the reference `bpop_likelihood` path to `1e-15`, so it is an optimisation,
   not a second implementation of the math.
2. **`remaining_successor_count` requires `x` to be in `remaining`.** The spec defines
   `S_t(x)` only for candidates in `R_t`; querying outside that raises `ValueError`
   rather than returning a silently meaningless count.
3. **Successor utility is recomputed per step.** `S_t(x)` counts only *remaining*
   dominated roles, exactly as Task 2 specifies. Note this differs from the older
   `hpop.inference.likelihood`, which computes `Q` once over the whole poset. Both
   agree whenever the frontier is a singleton or all frontier utilities are tied —
   which covers the chain, fork and antichain examples above — but they differ in
   general. The new module follows the Hour-2 spec.
4. **Scope correction mid-task.** Work had begun on the Stage 0–3 toy ladder
   (`bpop.py`, `targets.py`) before the Hour-2 instruction arrived. Those two files
   were deleted, since boundary priors and segmentation joints are explicitly out of
   scope here. Nothing from them survives in this deliverable.

## 13. Explicitly NOT implemented

No claim is made that any of the following works — none of it exists yet:
segmentation, boundary variables or boundary priors, split/merge/shift/relabel moves,
MCMC of any kind, `phi_k` or any composition likelihood, skill-transition matrix `P`,
duration model `p(d|k)`, recurrence, validity states `q_t`, failure-triggered repair,
background/leak states, FFBS, EM, GRPO, unknown `K`, directly-learned graph edges.

Hour 2 delivers exactly one object: `p_BPOP(y | U, beta, epsilon)` for a single static
skill instance.
