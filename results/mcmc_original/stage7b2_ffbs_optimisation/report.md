# Step 7B2 — exact optimisation of the FFBS conditional computation

Status: **parity PASS**, performance target **not met**. Sweep 1.686s -> 1.375s (1.23x) against a 1.20s primary target.

## The headline finding, which reframes the task

The brief attributed about 75% of an FFBS sweep to *candidate block-table and forward-chart construction* and set the block table as the primary target. The profile splits that 75% very unevenly:

| phase | baseline | share of sweep | can it be optimised here? |
|---|---|---|---|
| forward chart | 1.586s | 91% | **no — frozen Step 7A engine** |
| block-score construction | 0.229s | 13% | yes |
| parameter phase | 0.045s | 3% | frozen Stage 6E kernels |
| backward draw | 0.020s | 1% | no — frozen engine |

At function level the picture is starker: `semi_markov_ffbs.forward` is ~81% of the sweep, and **~60% of the whole sweep is `scipy.special.logsumexp`**, called 9,097 times per sweep from inside the frozen recursion. The block table — the brief's primary target — was 13%.

So the 1.20s target is not reachable under this task's constraints. Even an instantaneous block table leaves ~1.20s of frozen chart plus the frozen parameter phase. That is a statement about where the time is, not about whether FFBS is correct.

## What was optimised anyway

* new src/hpop/mcmc_original/fast_block_tables.py: one recurrent trajectory per (trace, start) instead of one per candidate block, with every width read off a cumulative sum
* candidate layout precomputed once per (traces, min_width, max_width) and cached; no Python tuples rebuilt per sweep
* candidates sorted by descending remaining length so each step operates on a contiguous view rather than a masked array
* dense per-trace tables allocated once and written in place
* skill-local invalidation: U_k rebuilds only column k; the four global scalars rebuild all; rho, pi and P rebuild none
* key_movement replaces two per-occurrence array constructions per trace per sweep with one linear walk
* stage6e_sampler.segmentation_sweep gains the upstream zero-proposal fast path, skipping a target evaluation whose value is discarded

The block table is the one large win available. A block score is a sum of per-step emissions along a trajectory that starts at `a` with `q_0 = 0`, and **nothing in that trajectory depends on the block end**. Previous builders replayed the shared prefix once per width; this one replays each start once and reads every width off a cumulative sum:

* 182,925 candidate-steps per skill -> 31,488 (5.8x less arithmetic)
* 25,490 candidates from 2,999 starts, across 100 traces and 3 skills

## Measured, alternating inside one process

Cross-run comparison is not usable on this machine: the forward chart is unchanged code and its wall time still varies by 40% between runs, because the Stage 6E2 baseline is competing for cores. Every number below alternates the two implementations inside one process and takes medians.

| quantity | before | after | ratio |
|---|---|---|---|
| global sweep | 1.686s | 1.375s | 1.23x |
| block table | 214.4 ms | 40.1 ms | 5.3x |
| sweep with only the zero-proposal fast path | 1.535s | — | — |

Table parity in that same comparison: max absolute difference **0.0e+00**.

Against the targets: primary 1.20s **missed** (1.375s), preferred 1.00s missed, stretch 0.80s missed. The LocalMoveKernel baseline runs at ~0.69 s per sweep, so FFBS remains roughly 2.0x its cost per sweep on this corpus.

## Exact parity

Across 27 problem shapes (J = 8, 24, 48, 96; K = 1, 2, 3; three parameter settings), tolerance 1e-10:

| comparison | worst observed |
|---|---|
| block score vs the width-bucketed builder | 0.000e+00 |
| block score vs `RecurrentBlockScorer.replay` | 1.421e-14 |
| max relative likelihood difference | 0.000e+00 |
| FFBS log Z | 0.000e+00 |
| exact DP marginals | 0.000e+00 |

Finite/-inf support identical in every shape: True. 3 combination(s) skipped as having no legal path at all (K = 1 with J > max_width forbids every segmentation).

## Negative controls

| injected fault | detected | observed effect |
|---|---|---|
| recurrent state leaks across candidates | True | 2.761e-01 |
| omega does not invalidate the table | True | 1.780e+00 |
| U_k does not invalidate its own column | True | 2.434e+01 |

The honest implementation's gap against the same reference is 0.000e+00, so the controls are separating a real fault from floating-point noise rather than from nothing.

One control had to be repaired to be meaningful: a uniform shift of U_k would NOT be observable: the likelihood sees U only through the precedence closure h(U), so control 3 uses a perturbation that actually moves the induced order

## What would actually reach the target

The remaining cost is concentrated in one primitive inside the frozen engine: `logsumexp` over a handful of terms, called once per `(position, skill)` per trace per sweep. A vectorised chart — one that forms the predecessor terms for all `(b, k)` of a trace as arrays and reduces them without per-cell scipy calls — is the only change that would move the sweep materially, and it is precisely what this task forbids (no engine edit, no second FFBS implementation). That is the right call for a validated engine, and it means the decision is the user's: authorising a vectorised chart, validated entry-for-entry against the frozen one, is the next lever. Nothing in this task should be read as evidence that it is safe to skip that validation.

Frozen engine sha256 `8150bb8235eb159d5e2f08ada7c698c3...` — unchanged: True.

Source commit `cacc1634b14ef0f4cbc54d22824cd43e76f6c4ac`.
