# Stage 4 — algorithm correctness of the local proposal kernel

Date: 2026-08-09
Branch: `mcmc-original-latent-poset`  Commit: `08e2cb6deeec1634d6c91667368b736a5427f2e0`  (working tree dirty)
Python 3.13.2, NumPy 2.4.6

Stage 1 proved the posterior was right using a proposal that needs the global
state list. This stage validates the moves a real sampler would actually use —
**Split, Merge, Shift, Relabel** — against that same exact posterior. No
synthetic recovery is attempted here; that is Milestone B.

## PASS / FAIL summary

| check | result | headline |
|---|---|---|
| Stage 4A move-level correctness | **PASS** | split/merge inverse, shift/relabel self-inverse, all moves legal |
| Stage 4B proposal law q(S->S') | **PASS** | sampled matches computed (max gap 0.00470), asymmetry 0.1875 |
| Stage 4C detailed balance + stationarity | **PASS** | balance 1.7e-18, naive-kernel control TV 0.1283 |
| Stage 4D posterior recovery (200k steps) | **PASS** | TV = 0.00416 |

## 1. Why the Stage 0-3 toy could not test this

The Stage 0-3 skills make two of the four moves structurally dead:

- every block matches **at most one** skill, so Relabel has nothing to move to;
- every B block consumes exactly one CPA-2 label, so the number of segments `L`
  is pinned by the trace and Split/Merge can never reach a legal state.

Stage 4 therefore uses a purpose-built toy where the supports genuinely overlap.

## 2. The kernel toy

Trace `x = (0, 1, 2, 3, 0, 1, 2, 3)`, `delta_B = 0.5`, `beta = 1.5`, `epsilon = 0.05`, uniform `pi_k = 1/4`.

| skill | CPA labels | induced order | role in the test |
|---|---|---|---|
| A | (0, 1) | 0 > 1 | shares a support with D |
| D | (0, 1) | antichain (0 and 1 incomparable) | **makes Relabel live** — same block, different order |
| F | (2, 3) | 2 > 3 | splits out of E |
| E | (0, 1, 2, 3) | 0 > 1 and 2 > 3, the two chains incomparable | support = A's union F's, **makes Split/Merge live** |

Shift is live because a boundary can slide between the 2-block and 4-block
tilings, relabelling both adjacent segments as it goes.

The trace admits **11 legal segmentations** with `L` in [2, 3, 4]:

| # | segmentation | exact P(S \| x) | relabel | split | merge | shift |
|---|---|---|---|---|---|---|
| 0 | (0, 1)_A + (2, 3)_F + (0, 1)_A + (2, 3)_F | 0.201094 | 2 | 0 | 3 | 0 |
| 1 | (0, 1)_A + (2, 3)_F + (0, 1)_D + (2, 3)_F | 0.103125 | 2 | 0 | 3 | 0 |
| 2 | (0, 1)_A + (2, 3)_F + (0, 1, 2, 3)_E | 0.106504 | 1 | 2 | 1 | 1 |
| 3 | (0, 1)_A + (2, 3, 0, 1)_E + (2, 3)_F | 0.106504 | 1 | 2 | 0 | 3 |
| 4 | (0, 1)_D + (2, 3)_F + (0, 1)_A + (2, 3)_F | 0.103125 | 2 | 0 | 3 | 0 |
| 5 | (0, 1)_D + (2, 3)_F + (0, 1)_D + (2, 3)_F | 0.052885 | 2 | 0 | 3 | 0 |
| 6 | (0, 1)_D + (2, 3)_F + (0, 1, 2, 3)_E | 0.054617 | 1 | 2 | 1 | 1 |
| 7 | (0, 1)_D + (2, 3, 0, 1)_E + (2, 3)_F | 0.054617 | 1 | 2 | 0 | 3 |
| 8 | (0, 1, 2, 3)_E + (0, 1)_A + (2, 3)_F | 0.106504 | 1 | 2 | 1 | 2 |
| 9 | (0, 1, 2, 3)_E + (0, 1)_D + (2, 3)_F | 0.054617 | 1 | 2 | 1 | 2 |
| 10 | (0, 1, 2, 3)_E + (0, 1, 2, 3)_E | 0.056407 | 0 | 4 | 0 | 0 |

## 3. Move-level correctness

Verified for every state and every move:

- every proposed state is support-compatible and tiles the whole trace;
- no move returns the current state;
- **Split and Merge are exact inverses** of each other;
- **Shift and Relabel are their own inverses**.

Total availability across the state space: `relabel` 14, `split` 16, `merge` 16, `shift` 12.

## 4. The proposal law

The kernel exposes `q(S -> S')` explicitly:

```
q(S -> S') = sum_t  p_t * 1[S' in N_t(S)] / |N_t(S)|
```

`|N_t(S)|` is a **local** count — it inspects the segments of one state, never
the global state list — so this is computable in a real sampler.

- sampled proposals match the computed law: max gap **0.00470** over 60,000 draws per state (criterion < 0.01)
- the proposal is genuinely **asymmetric**: max `|q(S->S') - q(S'->S)|` = **0.1875**
- `q(S->S') > 0` implies `q(S'->S) > 0` for every pair, so every move can be undone

The asymmetry is structural, not incidental: the all-E state offers 4 splits,
while each state reachable from it offers only 1 merge back.

## 5. Detailed balance and stationarity

Built the exact MH transition matrix `K` over the enumerated space and checked:

| property | value | criterion |
|---|---|---|
| rows of K sum to 1 | yes | exact |
| detailed balance, max \|pi_i K_ij - pi_j K_ji\| | 1.735e-18 | < 1e-12 |
| stationarity, max \|piK - pi\| | 1.388e-17 | < 1e-12 |
| TV(leading left eigenvector of K, posterior) | 2.567e-16 | < 1e-10 |
| irreducible | yes | all states mutually reachable |

### The negative control

This is what gives the whole stage teeth. The same kernel with the Hastings
ratio dropped — the mistake a symmetric-proposal implementation would make —
still runs, still mixes, and still looks healthy, but:

- detailed balance breaks: max flow asymmetry **0.01775**
- its stationary distribution is **TV = 0.12832** away from the posterior

| state | exact | naive-kernel stationary |
|---|---|---|
| 0 | 0.201094 | 0.221509 |
| 1 | 0.103125 | 0.114380 |
| 2 | 0.106504 | 0.058404 |
| 3 | 0.106504 | 0.104284 |
| 4 | 0.103125 | 0.112753 |
| 5 | 0.052885 | 0.058254 |
| 6 | 0.054617 | 0.030256 |
| 7 | 0.054617 | 0.053548 |
| 8 | 0.106504 | 0.071997 |
| 9 | 0.054617 | 0.036552 |
| 10 | 0.056407 | 0.138061 |

A trace that happens to induce a symmetric proposal cannot detect this at all,
which is why the kernel toy was selected for asymmetry rather than convenience.

## 6. Posterior recovery over a long run

200,000 iterations, 10,000 burn-in, 190,000 kept, seed `20260808`, overall acceptance **0.6745**.

| move | proposed | accepted | acceptance |
|---|---|---|---|
| relabel | 46,766 | 31,704 | 0.6779 |
| split | 27,191 | 22,865 | 0.8409 |
| merge | 39,206 | 22,866 | 0.5832 |
| shift | 24,320 | 15,297 | 0.6290 |

| state | exact | MCMC | abs error |
|---|---|---|---|
| 0 | 0.201094 | 0.200489 | 0.000605 |
| 1 | 0.103125 | 0.104053 | 0.000927 |
| 2 | 0.106504 | 0.105026 | 0.001477 |
| 3 | 0.106504 | 0.106158 | 0.000346 |
| 4 | 0.103125 | 0.102237 | 0.000889 |
| 5 | 0.052885 | 0.052716 | 0.000169 |
| 6 | 0.054617 | 0.054168 | 0.000449 |
| 7 | 0.054617 | 0.054700 | 0.000083 |
| 8 | 0.106504 | 0.108521 | 0.002017 |
| 9 | 0.054617 | 0.055747 | 0.001130 |
| 10 | 0.056407 | 0.056184 | 0.000222 |

**Total variation distance = 0.004157** (criterion < 0.02)
Worst single-state error = 0.002017. All 11 states visited; every move type proposed *and* accepted.

## 7. Notes and limitations

- The Stage 0-3 toy cannot exercise this kernel: every block there matches at most one skill (Relabel dead) and L is pinned by the trace (Split/Merge dead). Stage 4 uses a purpose-built toy with overlapping supports.
- The kernel toy trace was selected for a genuinely asymmetric proposal. (0,1,2,3,0,1) is accidentally symmetric and (0,1,2,3,2,3,0,1) offers no legal Shift; on either, a missing Hastings correction would go undetected.
- With role counts of 2 and 4, a one-position Shift is never legal on this trace; shift_moves therefore defaults to allowing any boundary inside the combined span, which is what keeps the kernel irreducible. max_shift=1 is implemented and tested, but yields no legal moves here.
- This stage validates the kernel on a single fixed trace with U held fixed. It says nothing about joint S+U sampling with local moves, or about mixing on long real traces — those belong to Milestone B and later.

