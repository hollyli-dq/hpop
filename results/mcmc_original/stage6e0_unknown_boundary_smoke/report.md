# Stage 6E0 — parity and smoke: PASS

Unknown-boundary target audit, parity with both parents, recurrent block-score
integration, and the joint smoke. **Stage 6E0 only. Stage 6E1 and Stage 6E2 are not
started, and no Stage 6E tag exists.**

Read [`../stage6e_complete/model_audit.md`](../stage6e_complete/model_audit.md) first: it
records the four findings that determine this stage, two of which change what Stage 6E can
reuse.

## Results

| check | value | tolerance | verdict |
|---|---:|---:|---|
| §7.1 oracle-segmentation parity with Stage 6D | worst residual 7.105e-14 | 1e-9 | PASS |
| §7.2 segmentation-kernel parity | worst 1.776e-15 over 18 comparisons | 1e-12 | PASS |
| §7.3 transition-update parity | counts, rows and diagonal all match Stage 3 | exact | PASS |
| §7.4 joint smoke | 26/26 checks | — | PASS |

### §7.1 is a constant, computed not fitted

With `S = S*` and `z = z*` pinned, the Stage 6E target differs from the Stage 6D target by
an additive constant — the boundary prior plus the path prior — which depends on no
continuous coordinate. The check is therefore the strong one: the *same* constant at every
`theta`, so every difference a sampler consumes agrees exactly. Six `theta` were drawn,
one at the registered values and five dispersed; the worst residual against the explicitly
computed constant is 7.1e-14.

### §7.2 the Hastings term is real

All 18 comparisons carry a **non-zero** Hastings term. Split and Merge change the
neighbourhood size, so symmetry is never assumed; the ratio is reconstructed from the
inherited Stage 5 neighbourhood counting and agrees to 1.8e-15.

### §7.3 no terminal transition

`sum(L_n - 1)` transitions are counted for `sum(L_n)` segments over `N` traces — a terminal
transition would give `sum(L_n)`. Self-transition counts are exactly zero, the sampled
diagonal is exactly zero, and `alpha = eta + C` holds on the allowed entries only.

### §7.4 what the smoke demonstrates

All four move types propose, and each accepts *and* rejects. Boundaries move, labels move,
`U`, `rho`, `pi`, `P` and all four recurrent scalars move. `q_0` is zero at the start of
every candidate block, checked across the whole segmentation periodically rather than once.
A cached score equals a fresh uncached replay to 1e-12; scoring is order-invariant (A, B, A
returns the same value); scoring candidates does not bump the cache version, so a rejected
proposal leaves the accepted cache untouched. Every retained segmentation is legal. The
state serialises, reloads and resumes bit-identically.

## Cache design

The score depends on `(trace, a, b, k, U_k, beta, omega, lambda_rep, lambda_back, epsilon,
config)`. Rather than key on four floats, the scorer holds a monotone integer `version`
bumped by any parameter change, so the key is the all-integer
`(version, trace, start, end, skill)` and correctness never depends on float equality. The
cache is written only by an explicit `set_parameters`, never by an evaluation.

## Not done

Stage 6E1A, Stage 6E1B, Stage 6E2, the registered test areas, the remaining artifact
directories, figures and the tag. Nothing here claims sampler correctness for the
unknown-boundary posterior: that is what Stage 6E1 exists to establish.
