# Hour 1 — Original latent partial-order representation `U -> h(U)`

Date: 2026-08-08

## 1. Git branch

`mcmc-original-latent-poset`

The branch already existed and was already checked out, so no new branch was created.
No existing file was modified. The three files showing as modified in `git status`
(`src/hpop/annotate/opencode.py`, `src/hpop/ingest/swe_rebench.py`,
`tests/test_inference.py`) were already dirty before this work began and were left
untouched.

## 2. Existing test-suite status BEFORE changes

`pytest` was not installed in `.venv`; it was installed (`pytest 9.1.1`) so the suite
could be run at all. No project code was changed to do this.

Baseline, project suite:

```
PYTHONPATH=src python -m pytest tests -q
63 passed, 1 warning, 36 subtests passed in 25.51s
```

- Passing: 63 tests (+36 subtests)
- Failing: 0
- Pre-existing warning: `RuntimeWarning: divide by zero encountered in log` from
  `src/hpop/inference/semi_markov.py:36` during
  `tests/test_semi_markov.py::TestConstraints::test_single_skill_cannot_tile_more_than_D_max_seeds`.
  Not repaired — unrelated to this task.

Pre-existing collection failure, **not** repaired (unrelated, stale code):
a bare `pytest -q` from the repo root also collects `archive/tests/`, which fails
with 8 import errors (`No module named 'hpop.posets'`, `'hpop.synthetic'`,
`'hpop.model'`, etc.). The archived package no longer exists. All runs below
therefore target `tests/` (or use `--ignore=archive`).

## 3. Files created

```
src/hpop/mcmc_original/__init__.py
src/hpop/mcmc_original/types.py
src/hpop/mcmc_original/latent_poset.py
tests/mcmc_original/__init__.py
tests/mcmc_original/test_types.py
tests/mcmc_original/test_latent_poset.py
results/mcmc_original/hour01_report.md   (this file)
notebooks/mcmc_original_walkthrough.ipynb  (executed walkthrough; since extended to
                                            cover Hours 1-2 and Stages 0-4, 11 figures)
```

Nothing under `src/hpop/inference/`, `src/hpop/synth/`, or the existing `tests/*.py`
was read-modified-written. The new package imports nothing from the old
implementation, and the old implementation imports nothing from it.

## 4. Exact definition implemented

For skill k the latent variable is `U_k ∈ R^{m_k × d}`: one row per role, one column
per coordinate of a d-dimensional product order. The induced strict partial order is
coordinate-wise dominance:

```
i ≻_U j   iff   U[i, r] > U[j, r]   for every r = 1, ..., d
```

implemented as

```python
p = np.all(u[:, None, :] > u[None, :, :], axis=-1)
np.fill_diagonal(p, False)
```

returning a `(m, m)` boolean matrix.

`P` is the **transitive closure** — the full precedence relation, not a cover/Hasse
relation. No transitive reduction is computed; that is a later visualisation concern.

Order structure is never represented as learnable edges and never needs an acyclicity
check: every real `U` induces a valid strict partial order by construction, because
irreflexivity, asymmetry and transitivity are inherited coordinate-wise from `>` on
the reals.

Companion queries:

- `predecessors(P, x) = {z : P[z, x]}`, returned as a tuple in ascending index order
- `successors(P, x)   = {z : P[x, z]}`, likewise
- `incomparable(P, i, j)` is True iff `i != j` and neither `P[i, j]` nor `P[j, i]`

## 5. Unit-test results

```
PYTHONPATH=src python -m pytest tests/mcmc_original -q
474 passed in 0.45s
```

- `tests/mcmc_original/test_latent_poset.py` — 0 failures
- `tests/mcmc_original/test_types.py` — 0 failures

Randomised coverage: `m ∈ {2, 3, 4, 5, 8}` × `d ∈ {1, 2, 3, 4}` (20 configurations),
20 independent draws per configuration, fixed seed `20260808` (derived seeds
`+1 … +7` for the separate randomised properties). Transitivity is checked with an
explicit triple loop, asymmetry with a double loop, since `m` is tiny.

## 6. Confirmation of the partial-order axioms

Verified over all deterministic examples and all randomised draws above (continuous
Gaussian `U`, and separately a coarsely quantised integer `U` that deliberately
produces many ties):

- **Irreflexive** — `P[i, i]` is False for every `i`. Confirmed.
- **Asymmetric** — `P[i, j]` implies `not P[j, i]`. Confirmed.
- **Transitive** — `P[i, j]` and `P[j, k]` imply `P[i, k]`. Confirmed by triple loop.

Known incomparable example:

```
U = [[ 1.0,  0.0],
     [ 0.0,  1.0],
     [-1.0, -1.0]]
```

gives `P[0,2] == True`, `P[1,2] == True`, `P[0,1] == False`, `P[1,0] == False`, and
`incomparable(P, 0, 1) == True`. Confirmed, along with
`predecessors(P, 2) == (0, 1)`, `successors(P, 0) == (2,)`.

Known total-order example:

```
U = [[3,3], [2,2], [1,1], [0,0]]
```

gives the chain `0 ≻ 1 ≻ 2 ≻ 3` *and* the closure pairs `0 ≻ 2`, `0 ≻ 3`, `1 ≻ 3`
— 6 True entries in total, no backward entries. Confirmed.

Additional properties verified: the vectorised implementation agrees entry-by-entry
with the literal `all(u[i, r] > u[j, r] for r in range(d))` definition; `h(U)` is
invariant to any coordinate-wise strictly-increasing rescaling of `U`; permuting
roles permutes `P` accordingly; for every distinct pair exactly one of
`P[i,j]`, `P[j,i]`, `incomparable(i,j)` holds; ties on any single coordinate block
dominance.

## 7. Full project test-suite status AFTER changes

```
PYTHONPATH=src python -m pytest tests -q
537 passed, 1 warning, 36 subtests passed in 37.57s
```

537 = 63 pre-existing + 474 new. Zero failures; zero regressions. The one warning is
the same pre-existing `semi_markov.py` log-of-zero warning recorded in §2.
`pytest -q --ignore=archive` gives the identical 537 passed. A bare `pytest -q` still
fails to collect `archive/tests/` exactly as it did before these changes.

## 8. Issues and ambiguities encountered

1. **`pytest` was absent from `.venv`.** Installed `pytest 9.1.1` so a baseline could
   be measured. No other dependency or source file was touched.
2. **`archive/tests/` breaks bare `pytest`.** Pre-existing and out of scope; recorded
   rather than fixed.
3. **`SkillTemplate` uses `@dataclass(frozen=True, eq=False)`.** Deviation from the
   literal `frozen=True` in the spec. With `eq=True`, the generated `__eq__` compares
   `u` and returns an *array*, and the generated `__hash__` raises on an ndarray.
   `eq=False` makes instances compare and hash by identity, which is the only
   sane behaviour until an explicit array-aware equality is needed. `Segment` and
   `Segmentation` use plain `frozen=True` as specified.
4. **`SkillTemplate.u` is copied and frozen.** `__post_init__` stores
   `np.array(u, dtype=float, copy=True)` with `writeable=False`, so the dataclass is
   genuinely immutable and freezing it cannot mutate the caller's array. This is
   stricter than the spec, not different from it.
5. **Contiguity vs. the separate "no overlap" / "strictly increasing ends" rules.**
   For `Segmentation`, `segments[i].end == segments[i+1].start` together with
   `end > start` already implies both. Both are still checked explicitly so the
   invariants are asserted rather than merely implied, and so error messages can
   distinguish a gap from an overlap.
6. **Ties.** `U[i, r] == U[j, r]` on any single coordinate blocks dominance in both
   directions, making `i` and `j` incomparable. This follows directly from strict
   `>` and is what the spec asks for; noted because it means a `U` with repeated
   values yields a sparser order than one might expect. Tested explicitly.
7. **`np.fill_diagonal(p, False)` is redundant** given strict `>`, but is kept so
   irreflexivity does not depend on floating-point comparison behaviour.

## 9. Explicitly NOT implemented (later stages)

No claim is made that any of the following work — none of it exists yet:
BPOP/frontier likelihood, MCMC of any kind, boundary sampling, split/merge moves,
segmentation inference, synthetic recovery, composition-only `phi_k`, multinomial
skill-composition likelihood, skill-transition matrix `P`, duration model `p(d|k)`,
recurrence, background/leak, FFBS, EM, GRPO, or global posets.

The only statistical object implemented in Hour 1 is `U_k -> h(U_k)`.
