# Superseded Stage-5 run (kept only as evidence)

`stage5_smoke/` and `stage5_full_seed0/` were produced by a Stage-5 implementation that
**reimplemented** the U update instead of reusing the verified partial-order sampler.
That implementation has been deleted; Stage 5 is being rebuilt on the vendored
`mcmc_simulation_po` (see `src/hpop/vendored/`).

These outputs are retained for one finding that is a property of the **skill library**,
not of any sampler, and will recur unless the library changes:

> Skills A and D share the CPA support {0,1}. D is an antichain, so it emits the ordered
> permutation (0,1) about half the time. For such a block the likelihood ratio is
> p_A/p_D = 0.975/0.500 = 1.95 — the *correct* posterior favours A over the true D by
> roughly 2:1. Measured consequences on the full seed-0 test split:
>
>   * D recall 0.43, versus A 0.88 / F 0.97 / E 0.85
>   * skill ARI 0.667 (boundary F1 was 0.966, so segmentation itself was fine)
>   * on 10 of 20 test traces the posterior STRICTLY PREFERS a non-true segmentation,
>     by up to 3.0 in log target
>
> Where the target points away from the truth, no sampler can recover it. If Stage 5B's
> skill metrics are to be meaningful, A and D must be made distinguishable.
