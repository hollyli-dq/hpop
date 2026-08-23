# TODO for Holly

Everything here needs a person. Nothing here was done by the autopilot, and nothing here
should be done by an autopilot.

## Decide

1. **Which operating point goes in the paper.** The study measured the anticipated
   regime at N=100, J=200, K=20, A=50 under both support regimes. The sparse regime is
   the realistic one for an induced CPA vocabulary; the full-support regime is a stress
   test. The paper should quote one as primary and the other as a bound — pick which.
2. **Whether the banded layout is in scope.** The dense `(J, J+1, K)` score table is the
   binding constraint on trace length, and the projected saving is in
   `memory_summary.csv`. Implementing it is a real piece of work with a real correctness
   burden (the backward sampler indexes the dense table directly). It is explicitly out
   of scope for this study. Decide whether it is in scope for the paper's future-work
   paragraph or for the codebase.
3. **How much of the `A` axis to show.** The two support regimes separate sharply and
   must not be averaged. Two panels is honest; one panel is not. If space forces one,
   show the sparse regime and state the full-support factor in the caption.
4. **Whether to re-run on a quiet machine.** Absolute constants are machine-specific.
   The exponents are the transferable part. If the paper quotes seconds, it should quote
   them from a load-controlled run.

## Verify before the paper goes out

- [ ] Re-read `SAFE_PAPER_CLAIMS.md` section D against the drafted text. Every banned
      claim is banned because it is false about *this* implementation, not because it is
      impolite.
- [ ] Check that every banded-memory number in the draft carries `not implemented`.
- [ ] Check that no sweep-rate figure appears without its configuration attached.
- [ ] Confirm no reviewer could read a throughput figure as a convergence claim.

## Points that did not complete

- `target_long_J500::build` — skipped_memory: predicted RSS 7.79 GiB exceeds the frozen cap 6.00 GiB; predicted RSS 7.79 GiB exceeds 80% of the 7.40 GiB currently reclaimable
- `target_long_J500::primitives` — skipped_memory: predicted RSS 7.79 GiB exceeds the frozen cap 6.00 GiB; predicted RSS 7.79 GiB exceeds 80% of the 7.35 GiB currently reclaimable

A refused point is absent evidence, not a measured limit. If any of these matter for the
paper, they need a machine with more memory or a longer budget — not a relaxed gate.

## If the study is repeated

- The harness is resumable: re-running `scripts/scalability/run_autopilot.py` with the
  same output directory continues from `state.json` and re-measures nothing.
- `bench_plan.plan_digest()` guards that. Changing the configuration set starts a fresh
  state file and preserves the old one as `state.superseded.json`.
- The task *order* is deliberately outside the digest, so the queue can be re-prioritised
  without discarding completed measurements.
- `scripts/scalability/bench_parity.py` must pass before any scaling point is trusted.
  It takes about a minute.

## Explicitly not done, and deliberately

Banded block storage, optimized backward sampling, third-forward-pass reuse, sparse `P`,
pruning, beam search, GPU kernels, an alternative `Q_k` initializer, approximate DP, and
any new model or sampler move. The study measures `optimized_segmental_v1` as committed at
`564995efd056d7d33984f0ca1532386e6140ea0c`.
