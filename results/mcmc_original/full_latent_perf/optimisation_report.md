# FULL-LATENT forward-path optimisation — measured marginal speedups

Branch `perf-forward-optimisation`, worktree `/Users/dongqing/Desktop/hpop-perf`.
The live 8-worker formal chain was never touched: `/Users/dongqing/Desktop/hpop/src` is
byte-identical throughout, which matters because `--workers 8` uses multiprocessing spawn
and would re-import edited source if any worker respawned.

Everything is flag-gated in `perf_flags.py`. With no flags set the engine is call-for-call
what it was, so this branch is safe to merge without changing any running configuration.

## The four optimisations

| flag | mechanism | site |
|---|---|---|
| O1 `inline_logsumexp` | scipy `logsumexp` -> the same reduction written out | `semi_markov_ffbs._reduce` |
| O2 `emission_hash_cache` | skip the table rebuild when H = h(U) is unchanged | `FFBSBlockTables.refresh` |
| O3 `factorised_forward` | `r[a,k] = LSE_h(alpha[a,h] + logP[h,k])`, O(JK^2+JDK) | `_factorised_core` |
| O4 `batched_forward` | the factorised recursion over a whole length class | `forward_batched_group` |

O4 is wired into both `ffbs_segmentation_draw` and `CollapsedULikelihood._forward_all`. The
backward loop still runs in original trace order, so the rng is consumed in exactly the same
sequence — batching changes where alpha comes from, never the draw order.

## Parity, checked before any timing

8 configurations x 2 arms, against the unflagged engine, on real checkpoint state:

| config | alpha error | log Z error | -inf pattern | emission tables |
|---|---|---|---|---|
| O2 alone | **0.00e+00** | **0.00e+00** | identical | **bitwise identical** |
| O1 alone | 2.84e-14 | 1.42e-14 | identical | bitwise identical |
| O3, O4, all four | 4.26e-14 | <=3.55e-14 | identical | bitwise identical |

Each configuration's counters are asserted non-zero, so a flag that silently did nothing
cannot pass parity by being inert. `perf_parity.json`.

## Independent measurements (each optimisation alone)

Plain sweep, median of 15 interleaved rounds, wall seconds.

| | COND | | MARG | |
|---|---|---|---|---|
| baseline | 1.362 s | 1.00x | 1.346 s | 1.00x |
| O1 alone | 0.522 s | 2.61x | 0.506 s | 2.66x |
| O2 alone | 1.184 s | 1.15x | 1.171 s | 1.15x |
| O3 alone | 0.398 s | 3.43x | 0.385 s | 3.50x |
| O4 alone | 0.221 s | 6.17x | 0.219 s | 6.14x |

**The product of the isolated speedups is 63.5x. The measured cumulative result is 27.3x.**
Multiplying would have overstated the outcome by 2.3x, because O1, O3 and O4 all attack the
same scipy dispatch cost and cannot each remove it.

## Cumulative stack, with marginal contributions

Plain sweep, median of 15 interleaved rounds. "Marginal" is against the row above.

### FULL-COND

| step | wall median | cumulative | **marginal** | ms saved by this step |
|---|---|---|---|---|
| baseline | 1.346 s | 1.00x | - | - |
| +O1 inline logsumexp | 0.517 s | 2.60x | **2.602x** | 828 |
| +O2 emission cache | 0.346 s | 3.89x | **1.495x** | 171 |
| +O3 factorised | 0.219 s | 6.15x | **1.581x** | 127 |
| +O4 batched | 0.049 s | **27.31x** | **4.442x** | 170 |

### FULL-MARG, plain sweep

| step | wall median | cumulative | **marginal** | ms saved |
|---|---|---|---|---|
| baseline | 1.346 s | 1.00x | - | - |
| +O1 | 0.518 s | 2.60x | **2.595x** | 827 |
| +O2 | 0.341 s | 3.95x | **1.522x** | 178 |
| +O3 | 0.217 s | 6.19x | **1.567x** | 123 |
| +O4 | 0.047 s | **28.65x** | **4.625x** | 170 |

### FULL-MARG, structural sweep (3 all-trace forward passes)

| step | wall median | cumulative | **marginal** | ms saved |
|---|---|---|---|---|
| baseline | 3.659 s | 1.00x | - | - |
| +O1 | 1.158 s | 3.16x | **3.160x** | 2,501 |
| +O2 | 1.008 s | 3.63x | **1.149x** | 150 |
| +O3 | 0.636 s | 5.76x | **1.586x** | 373 |
| +O4 | 0.121 s | **30.26x** | **5.256x** | 515 |

O1 is worth more on a structural sweep (3.16x vs 2.60x) because that sweep runs three
forward passes rather than one, so the dispatch cost it removes is three times larger.

## Where the time goes now

Re-profiled with all four flags, 80 sweeps, same 7-bucket decomposition. COND 54.7 ms/sweep:

| phase | ms | % |
|---|---|---|
| `ffbs_backward` | 23.2 | **42.5%** |
| batched forward | 15.1 | 27.6% |
| `target_full_replay` | 10.4 | 19.0% |
| emission | 0.1 | 0.1% |
| everything else | 5.9 | 10.8% |

`ffbs_table_builds = 1` over 80 sweeps: the H-hash cache hit 79 of 80.

**The bottleneck has moved to backward sampling**, the one part of the forward/backward pair
not touched here. `backward_sample` calls `predecessor_terms` once per drawn segment (~556
per sweep) through the same un-batched Python path the forward recursion used to. Emission
construction, which the item-1 profile ranked second, is now 0.1%.

## Verification against the existing audit suite

`perf_flags` reads `HPOP_PERF_FLAGS` at import, so the project's own audits can be run
against the optimised paths without editing a single test. The tests do not know the flags
exist, which is what makes them a fair check.

| run | result |
|---|---|
| `tests/mcmc_original -k "ffbs or semi or markov or collapsed or block"`, flags OFF | 104 passed |
| same selection, all four flags ON | 104 passed |
| parity gate, 8 configs x 2 arms | ALL_PASS |

**The flags-on run caught a real defect on its first attempt**, which is the reason it was
worth doing. `test_rho_pi_and_P_invalidate_no_block_score_column` failed for all three of
rho, pi and P. The block scores were correct; the diagnostic was not. On a cache hit the
guard returned early without touching `last_refresh`, so that field still held the previous
call's record and reported `rebuilt_skills = [0, 1, 2]` for a call that rebuilt nothing.
Anything reading it -- the test, and `scripts/stage7b2_optimisation_report.py` -- would have
been told a rebuild happened when none did. Fixed by writing the faithful record
(`rebuilt_skills = []`, `reused_skills = all`) on the cache-hit path.

## What was measured but NOT implemented

**MARG's redundant third forward pass.** Confirmed at counter level (`lse_calls` 30,900 =
3 x 10,300 on structural sweeps) and mathematically: pi and P are frozen across the
structural step and FFBS, so pass 3 duplicates pass 2 on accept and pass 1 on reject. After
O4 the two extra passes cost ~74 ms on a structural sweep, so eliminating one saves ~37 ms
there, ~3.7 ms amortised — about 8% of a 47 ms MARG sweep.

It is not implemented because it is the only change here that is not self-contained: the
FFBS path uses `BlockScoreTable` and the collapsed path uses `FastBlockScoreTable`, and the
two agree to 3.55e-15 but **not bitwise**. Reusing a chart across them perturbs the chain at
~16 ULP on top of the ~1e-14 already introduced. Doing it properly means making both paths
share one table object, which is a design change rather than an optimisation. Your call.

## Reproducibility, stated plainly

O2 is bit-identical: it skips recomputation and returns the very bits a rebuild would have
produced. **O1, O3 and O4 are not.** They re-associate floating-point sums, moving alpha by
~1e-14. A categorical draw within 1e-14 of a boundary can fall the other way, after which
the sample path diverges. The posterior is unchanged; the realised chain is not reproducible
against a run made without the flags.

So: O2 can be enabled on an existing formal chain without breaking bit-reproducibility.
O1, O3 and O4 belong to a new frozen launch.

## Artifacts

| file | contents |
|---|---|
| `perf_parity.json` | the parity gate, 8 configs x 2 arms |
| `perf_cumulative_bench.json` | independent + cumulative + structural blocks |
| `sweep_profile_FINAL_all4.json` | re-profile with all four flags |
| `scripts/perf_parity_check.py` | the gate |
| `scripts/perf_cumulative_bench.py` | the interleaved benchmark |
| `src/hpop/mcmc_original/perf_flags.py` | flags, counters, and the exactness note |

## Caveat on absolute numbers

The baseline here is 1.346 s/sweep at load ~8; item 1 measured 1.554 s at load ~20 and
Part A 2.582 s. Absolute times remain contention-dependent and there is still no idle-box
baseline. Every ratio above is from configurations interleaved inside one process, which is
what makes them comparable; the absolute seconds are not portable.
