# Preregistration — final synthetic component-identifiability experiments

**Status: NOT LAUNCHED.** Written before any Condition-A or Condition-B computation on the
confirmatory corpus. No draw, pilot sweep or metric exists for either condition.

Backend `564995efd056d7d33984f0ca1532386e6140ea0c`. Reference oracle `hpop.mcmc_original`,
unmodified; `semi_markov_ffbs.py` = `8150bb8235eb159d5e2f08ada7c698c383f4c7fd31f3b882e218b207dd135486`.

## 1. Scientific questions

**A.** When the reusable partial-order library is known, can the model recover the latent
skill boundaries and reusable-skill assignments? Oracle `H*` → infer `(S, z)`.

**B.** When every skill invocation and reusable-skill label is known, can the model recover
the reusable partial-order library? Oracle `(S*, z*)` → infer `U`, evaluate `H = h(U)`.

Together with the completed fully-latent confirmatory experiment these give three
conditions on **one** corpus and **one** truth, so they are directly comparable.

## 2. Corpus and truth — identical to the confirmatory run

| item | value |
|---|---|
| corpus hash | `3e3aa6533bd7951f9b2ed1dfa050e9d07f1b2e96b0e3914e044010130e9acdfa` |
| truth hash | `effccc91114d9647f87859aaa7ee219e3bc664bb41d2c6389844ea64346a5e8e` |
| train / held-out | 100 / 45, `J` cycling (24, 32, 40, 48) |
| truth law | `truth ~ p(truth | rho = 0.5, A)` |

Truth was legitimately unsealed after the terminal commit `f9799f9f`. **A and B are oracle
component-identifiability experiments: using truth as a fixed input is intentional and is
the whole design.** Truth is never used to tune proposal scales, thresholds, run lengths,
seeds or metrics.

The model, recurrent likelihood, CPA alphabet, role maps, `K = 3`, nuisance parameters and
held-out split are unchanged.

## 3. Authoritative variable scope

| | infer | fix |
|---|---|---|
| **A** | `S`, `z` | `U*`, `theta*`, `pi*`, `P*`, `delta_B*`, `epsilon*` |
| **B** | `U` (evaluate `H = h(U)`) | `S*`, `z*`, `theta*`, `rho = rho_0 = 0.5` |

**Condition B does not infer or recover `rho`.** It infers `U` at fixed `rho_0` and
evaluates induced partial-order recovery. A manuscript audit found no statement claiming
otherwise: `results_matched_synthetic_ab.tex` already states the target as
`p(U | X, S*, z*, vartheta*, rho_0)` and that "rho was *fixed* at rho_0 = 0.5 ... rho was
not a recovery target". No historical artifact is altered.

## 4. Scoped truth access

| component | may read | must not read |
|---|---|---|
| A runner | `H*` (= `U*`) and fixed globals | — |
| A evaluation module | `S*`, `z*` recovery truth | — |
| B target / sampler | `S*`, `z*`, fixed globals | **`H*` / `U*`** |
| B proposal-scale pilot | `S*`, `z*`, fixed globals | **`H*` / `U*`** |
| B recovery evaluation | `H*` | — (opened only after the truth-free terminal report is committed) |

Satisfied by construction and asserted by test: `ConditionBTarget` holds only an
`OracleBlockLikelihood` (built from oracle blocks and the fixed scalars) and `rho_0`;
`U*` never enters it. `build_target` reads `n_skills, beta, epsilon, omega, lambda_rep,
lambda_back` only. A test asserts the B target and chain give identical values when `U*` is
replaced by an arbitrary array, proving no dependence on it.

## 5. Seeds — all verified unused before registration

| purpose | seed(s) |
|---|---|
| A: FFBS draws for nonlinear summaries | `6_301_001` |
| A: tiny enumeration cross-check | `6_300_777` |
| B: pilot chains | `6_302_201`, `6_302_202` |
| B: pilot starts | `6_302_211`, `6_302_212` |
| B: formal chains | `6_302_001`–`6_302_004` |
| B: dispersed starts | `6_302_101`–`6_302_104` at scales (0.5, 1.0, 2.0, 3.0) |

## 6. Condition A protocol — exact dynamic programming, not MCMC

A is the exact semi-Markov posterior `p(S, z | x, U*, theta*, pi*, P*, delta_B*, epsilon*)`.
Traces are conditionally independent, so the posterior is computed **exactly per trace** by
forward/backward DP. **No chains, no burn-in, no production length, no thinning, and no
convergence gate**; inventing one would misrepresent the method.

Exact from the DP: block, boundary, occurrence-label and transition marginals; the
segment-count posterior; the Viterbi MAP labelled path; the posterior probability of the
true path and the true segmentation; the exact path entropy.

**5,000 FFBS draws per trace** (seed `6_301_001`) are used **only** for nonlinear summaries
that cannot be computed exactly from marginals. An exact marginal is never replaced by
Monte Carlo merely to produce a chain.

**Tiny enumeration cross-check** retained: seed `6_300_777`, 200,000 draws, trace lengths
6, 7, 10.

### A metrics — historical only (Ruling 3)

boundary AUROC; boundary Brier; boundary ECE (historical 10-bin definition); boundary
classification at the historical 0.5 threshold; occurrence modal-label accuracy; mean
posterior probability of the true label; per-trace ARI; MAP segment-count accuracy; path
evidence; occurrence NLL versus the historical prior reference (`NullScorer`); held-out
boundary AUROC.

**No new oracle-structure held-out path-marginal NLL is added.** Co-skill recovery is the
primary `z` metric because it is invariant to skill relabelling. A high boundary
probability at a *nearby* position is not counted as boundary recovery.

## 7. Condition B protocol

### 7.1 Truth-free proposal-scale pilot

Historical grid and protocol: multipliers `(0.25, 0.5, 1.0, 2.0, 4.0, 8.0)` on base
`SIGMA_U`, **3,000 sweeps per candidate**, 2 pilot chains, acceptance band `[0.20, 0.60]`.

Quantities exposed to selection are truth-free only: U-row acceptance, invalid-proposal
rate, H-changing move count, expected squared jump distance, numerical validity
(finite log-target). **True `H`, closure recovery, F1/Hamming, posterior probability of
truth, held-out prediction and any scientific comparison are not computed during the
pilot.** `H*` is not exposed to scale selection.

**Selection rule — the historical one, which exists and therefore governs:** retain
candidates with pooled acceptance in `[0.20, 0.60]` **and** finite log-target, then choose
the candidate with **maximum ESJD per proposal**.

ESJD is truth-free but was not in the enumerated list of permitted pilot quantities; it is
used because the historical selection rule requires it, and this is recorded as a
deliberate, declared inclusion.

**Failure rule (this supersedes the historical fallback):** if no candidate enters the
band, **stop before the formal run and report pilot failure**. The historical code instead
fell back to "closest to band, band not widened"; that fallback is **not** used here.

Every pilot draw is discarded. The formal chains start afresh from the preregistered
dispersed starts.

### 7.2 Sampling — final production-phase protocol

This is the **final Condition-B rerun under the final production-phase diagnostic
protocol**. It is **not** a byte-for-byte replication of the historical 10k-retained-window
ladder, and will not be described as one.

| item | value |
|---|---|
| warm-up | **50,000 sweeps, discarded completely** |
| production | **100,000 sweeps** |
| thinning | 5 |
| chains | 4, dispersed structural starts |
| kernel | `sampler_u.u_row_sweep` arithmetic, symmetric Gaussian row proposal |
| terminal gate | one, after production; **no adaptive stopping** |

### 7.3 Convergence gate

Rank-normalized split R-hat, pooled across chains, production draws only.

| summary | R-hat | bulk ESS | tail ESS |
|---|---|---|---|
| log target | ≤ 1.01 | ≥ 1000 | ≥ 500 |
| all other non-degenerate registered diagnostics | ≤ 1.01 | ≥ 400 | ≥ 400 |

**Exact canonical closure library** — the authoritative structural object, not relation
counts:

* **(a) constant and equal** across every production chain and draw → degenerate
  cross-chain agreement; **no ESS floor** for the library variable;
* **(b) constant within chains, unequal across** → **automatic FAIL**;
* **(c) non-degenerate** → ordinary R-hat and ESS gate.

Branch (a) is permitted **only if** all four hold: the four starts are structurally
dispersed; every chain accepts at least one H-changing move during warm-up; the exact
canonical closure library (not only relation counts) is equal; all remaining registered
diagnostics pass. Warm-up contributes only integer counters and start identifiers, never a
draw.

Condition B receives a **formal truth-free MCMC convergence verdict** under this gate. That
verdict is separate from truth-recovery quality and is committed **before `H*` is opened**.

### 7.4 B recovery metrics

exact canonical unordered-library posterior probability; exact labelled-library recovery
after the one frozen alignment rule; mean and per-skill transitive-closure F1; closure
Hamming; incomparable-pair precision/recall/F1; posterior relation calibration/Brier;
permutation-invariant R-hat/ESS; accepted H-changing moves per chain; held-out NLL under
oracle paths against the historical antichain and total-order baselines.

## 8. Recovery verdicts (Ruling 4)

For both A and B: report **all raw metrics with uncertainty/MCSE**, and issue **no new
formal binary recovery PASS/PARTIAL label**. What the historical rule would have returned
is reported descriptively under the heading

> **LEGACY-RULE SENSITIVITY — NOT THE FORMAL VERDICT**

because those thresholds were calibrated on the previous supplied truth and are not the
formal verdict for the new prior-drawn truth.

## 9. MCSE definitions

For any quantity estimated from draws, `MCSE = posterior_sd / sqrt(ESS)` with bulk ESS on
production draws. For a Bernoulli quantity with posterior probability `p`,
`MCSE = sqrt(p(1-p)/ESS)`. Exact DP quantities in Condition A carry **no** MCSE and are
reported as exact; only the FFBS-based nonlinear summaries carry one. Every paper-facing
posterior quantity is reported with its MCSE.

## 10. Numerical equivalence

For both conditions, the optimized path is validated against the frozen reference on a
deterministic subset: identical legal support and `-inf` pattern; forward log-normalizer
error ≤ 1e-10; block-score error ≤ 1e-10; same truth and corpus hashes; no sealed source
modified; all harness tests pass; final metrics regenerate exactly from the stored
CSV/JSON. Observed discrepancies are expected near 1e-14 and the actual values are
recorded. Bit-identical MCMC trajectories are **not** required of algorithms that
re-associate floating-point sums.

## 11. Output artifacts

```
results/mcmc_optimized/final_component_A/
    FINAL_REPORT.md  metrics.json  boundary_metrics.csv  coskill_metrics.csv
    provenance.json  figures/
results/mcmc_optimized/final_component_B/
    FINAL_REPORT.md  metrics.json  chain_diagnostics.json  relation_metrics.csv
    provenance.json  figures/
results/mcmc_optimized/final_synthetic_summary/
    FINAL_SYNTHETIC_REPORT.md  final_synthetic_metrics.json
    final_synthetic_table.csv  provenance.json
paper/tables/tab_component_identifiability.tex
paper/figures/fig_component_identifiability.{pdf,png}
```

Figure panels use a deterministic trace rule fixed here: **the first trace of each length
class** (24, 32, 40, 48) in frozen corpus order. No cherry-picking.

## 12. Stopping and failure rules

1. A is exact and runs once; there is nothing to stop.
2. If the B pilot admits no candidate in `[0.20, 0.60]`, **stop and report pilot failure**.
3. B runs 50,000 + 100,000 once. No extension, no ladder, no adaptive stopping.
4. B's truth-free convergence verdict is committed before `H*` is opened.
5. If a numerical-equivalence check in §10 fails, stop and report before any metric is
   published.
6. No new sampler, rescue kernel, dataset, metric or threshold may be added. No historical
   verdict or sealed source may be modified.

## 13. Paper-facing language

The method is **"Ours / path-marginal inference"**; the conditional variant appears only as
**"w/o path marginalisation"**. A and B are **not** framed as another COND-vs-MARG
competition. Permitted component claims are of the form "with oracle reusable structures,
the exact segmental posterior recovers skill boundaries and occurrence-level
co-clustering", and "with oracle labelled paths, the posterior recovers the reusable
partial-order library". The claim "every inferred parameter improves" is **not** permitted:
the confirmatory result shows `pi` point-estimate recovery is mixed.
