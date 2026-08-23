# Stage 6E — audit of the unknown-boundary model

Read from the repository before any Stage 6E inference code was written. Everything below
cites the file it came from. Four findings determine the whole stage, and two of them
change what Stage 6E can reuse.

## 0. Headline

| the Stage 6E brief assumes | the registered repository actually has |
|---|---|
| a Stage 6D corpus that may already be trace-level | **500 independent blocks**, `K = 1` skill, no traces, no `pi`/`P` |
| `LocalMoveKernel` reusable as-is | its legality predicate is BPOP-specific and admits **no** recurrent block |
| `pi, P` possibly fixed | they belong to the registered target and are **inferred** |
| possible terminal transition | there is none, and none is added |

## 1. `pi` and `P` are inferred, using the frozen Stage 3 updates

`targets.log_target_segmentation` takes `log_pi` and `log_transition`; `transitions.py`
carries `transition_counts`, `dirichlet_posterior_params`, `sample_transition_matrix` and
`posterior_mean_transition_matrix`, which Stage 3 validated and Part III of the
walkthrough derives. They are part of the registered final target.

They were absent from Stage 6D for a single reason: that stage had `K = 1` skill and
oracle labels, so the path prior had nothing to say. Stage 6E restores them rather than
inventing them. `eta = 1.0` for both rows and initial state.

## 2. `P` forbids self-transitions — a registered constraint with real consequences

`transitions.allowed_next(h, K)` returns `tuple(k for k in range(K) if k != h)`, and
`sample_transition_matrix` leaves the diagonal at exactly `0`. So:

- consecutive segments must carry **different** skills;
- a segmentation repeating a label across a boundary has log target `-inf`.

This is not a Stage 6E choice, and it is load-bearing. Without it a segment could be cut
anywhere at no cost in the path prior, and the segmentation would be badly
non-identifiable. The Stage 6E move kernel excludes such states from the neighbourhood
rather than proposing them and rejecting on `-inf`, so the proposal stays supported on the
target's support and the Hastings ratio is finite in both directions.

## 3. There is no terminal transition

`targets.log_path_prior` emits `log pi[z_1]` and then exactly one `log P[z_{l-1}, z_l]`
per adjacent pair. Nothing is emitted for ending a path. None is added here.

## 4. The Stage 5 compatibility predicate cannot represent the recurrent state space

This is the one place Stage 6E cannot reuse a Stage 5 component verbatim, and it is worth
stating precisely because the brief's default is to reuse.

`proposals.compatible_skills` calls `toy.map_cpa_block_to_roles`, whose contract is:

```python
if Counter(block) != Counter(labels):
    return None
```

A block is compatible with a skill only when their multisets are **equal** — so a legal
Stage 5 block is a *permutation* of that skill's roles, visiting each exactly once. That
is the BPOP model.

The recurrent model is defined by the opposite. `lambda_rep` and `lambda_back` exist
because roles recur; the Stage 6D corpus is blocks of `T = 20` over `m = 5` roles, so
every block repeats roles fifteen times over. Under the Stage 5 predicate **no recurrent
block is compatible with any skill**, every neighbourhood is empty, and the kernel cannot
move.

The registered escape clause applies — "unless the existing kernel cannot represent the
registered state space" — and is taken as narrowly as possible.
`recurrent_segmentation.Stage6EMoveKernel` subclasses `proposals.LocalMoveKernel` and
overrides **`neighbours` alone**. `proposal_distribution`, `proposal_prob`,
`sample_proposal` and `mh_local_step` are inherited unchanged, so the neighbourhood
counting and the Hastings ratio remain the objects Stage 4 verified against an exact
transition matrix. What replaces the predicate is the registered width bound
(`3 <= width <= 12`) plus the no-repeat constraint from finding 2; every skill shares the
same role alphabet, so a legal-width block may carry any skill.

**Consequence for cost.** The Stage 5 predicate made neighbourhoods tiny. Without it they
grow as `O(J x K^2)` per move type, and `proposal_distribution` materialises all of them
on both the forward and reverse side of every step. On a 12-position trace with `K = 3`
the four move types already yield 2 / 0 / 4 / 16 neighbours; at trace lengths near 100
this is the dominant cost of Stage 6E and the reason the formal chains are expensive.

## 5. The Stage 6D corpus is not trace-level, so a new corpus is required

`recurrent_synthetic.generate_recurrent_dataset` produces `RecurrentBlock` objects with a
fixed externally supplied `T`, and `"full"` mode gives **500 train + 200 held-out
independent blocks of `T = 20`**, all from the single `U_TRUE` — `K = 1`. There is no
trace structure, no skill sequence and no `pi`/`P`.

Section 11's fallback therefore applies exactly as written: concatenating those blocks
would not produce a registered trace corpus, and pretending otherwise would fabricate a
skill-transition structure the generator never sampled. A trace-level Stage 6E corpus must
be generated and frozen from the existing recurrent block generator plus the registered
skill-transition model.

## 6. What is fixed, and what becomes latent

```
newly latent in 6E   S_{1:N}  (block boundaries)      z_{1:N}  (skill labels)
still inferred       U_{1:K}, rho, beta, omega, lambda_rep, lambda_back, pi, P
fixed                epsilon = 0.02, K = 3 skills, delta_B = 0.15,
                     min width 3, max width 12, q_0 = zeros(m) per candidate block
does not exist       tau, a second prior on H, a skill-specific duration model
```

`H_k = h(U_k)` remains derived, never state, and carries no second prior — inherited from
the Stage 6D audit unchanged.

## 7. Recurrent block scoring: `q_0 = 0` and a versioned cache

Every candidate block is scored by a complete replay from `q_0 = 0`. A candidate block is
a *hypothesis* about where an execution starts, so letting `q` leak in from the left would
score a different model and would make the answer depend on evaluation order. A boundary
move changes two adjacent blocks; both are rescored from zero and neither inherits from
the other.

The score depends on `(trace, a, b, k, U_k, beta, omega, lambda_rep, lambda_back, epsilon,
config)`. Rather than key a dictionary on four floats — slow, and silently wrong if a
scalar returns to a bit-identical previous value — the scorer holds a monotone integer
`version`. Any parameter change bumps it and clears the cache, so the key is the
all-integer `(version, trace, start, end, skill)` and correctness never depends on float
equality. The cache is written only by an explicit `set_parameters`, never by an
evaluation, so a rejected proposal cannot leave anything behind.

## 8. Verified before any inference was written

```
Stage 6D config hash unchanged      ebd5effd5b32e68a...  (asserted, not assumed)
q_0 zero at the start of every block                     True
order invariance  score A, score B, score A again        equal
cached value equals a fresh uncached replay              equal to 1e-15
a scalar change bumps the version and clears the cache   True
neighbourhoods contain only legal segmentations          True
the proposal distribution sums to 1                      True
```

## 9. What this audit did not find

No model-definition blocker. The target is well defined, every component of the registered
weight has a referent in committed code, and the two divergences from the brief's defaults
(a new trace corpus; a replaced legality predicate) are both forced by the registered
model rather than chosen.
