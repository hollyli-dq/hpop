# HPOP — Hierarchical Bayesian Partial-Order Model

A hierarchical Bayesian model: **Actions → Local Partial Orders (skills) → Global Partial Order**,
with a reusable **skill library** under a nonparametric prior that **penalizes creating new skills**.
It is the hierarchical (Dirichlet-Process-over-posets) extension of BPOP (`docs/_bpop.txt`); the
flat single-level model is BPOP. PDAF outputs (`docs/DESIGN.md`, `rules/`) supply priors / weak
supervision. See [[hpop-partial-order-model]] memory for the connection to the published papers.

## 1. Levels and objects

- **Skill = a local partial order.** Library `L = {P_1,…,P_R}`. Each `P_r` is over its CPA support,
  BPOP-parameterized by latent `U^(r) ∈ R^{m_r × K_r}`: the poset is the intersection of `K_r`
  realizers (poset dimension), `K_r ~ trunc-Poisson(λ)`, `U^(r) ~ Gaussian`. This is BPOP's prior,
  one instance per skill.
- **Workflow = a partial order over skill instances.** Instances `s_1,…,s_T` with types
  `z_i ∈ {1,…,R, new}`; global partial order `Q` over the instances, its own BPOP latent `V`.
- **Phases** are a cognitive TAG on each CPA (drive the PDAF rules), NOT a modeling layer. No phrase layer.

## 2. Generative process (per trace)

1. **Skill-type assignment via CRP (this is the new-skill penalty):**
   ```
   P(z_i = r   | z_<i) = n_r       / (i-1+α)     # reuse: cheap, ∝ usage count n_r
   P(z_i = new | z_<i) = α         / (i-1+α)     # new skill: costs ∝ α  (the penalty)
   ```
   A new type draws a fresh poset `P_{r*} ~ G_0` and so also pays a Bayesian-Occam cost (it must
   explain enough data to justify its own parameters). Pitman–Yor(α,d) variant for power-law
   library growth: `P(new) ∝ α + dK`, `P(existing r) ∝ n_r − d`.
2. **Global PO:** emit the order of skill instances as a stochastic linear extension of `Q` via the
   global robust frontier-softmax (temperature `β_g`, ε noise).
3. **Local PO:** within each instance `i`, emit its CPA subsequence as a stochastic linear extension
   of `P_{z_i}` via the local robust frontier-softmax (`β_ℓ`, ε).

## 3. Joint log-likelihood (per trace)

```
log p(trace) =  Σ_i log CRP(z_i | z_<i, α)          ← NEW-SKILL PENALTY (global likelihood)
             +  log p(skill order | Q)               ← global frontier-softmax
             +  Σ_i log p(actions_i | P_{z_i})        ← local frontier-softmax
```
The penalty sits in the global likelihood and charges `−log α` (plus the `G_0` Occam cost of a fresh
poset) every time a new skill is opened instead of reused. `α` is the single tunable knob; smaller α
⇒ stronger reuse / fewer new skills. Corpus-level sharing of the library across traces = a
**hierarchical DP** (HDP-over-posets).

Frontier-softmax (both levels): at step `t`, the frontier `F_t` = minimal not-yet-completed elements;
choose the next element ∝ `exp(β · Q_succ(·))` over `F_t`, with an ε trembling-hand term so noisy
traces don't get zero likelihood. Cost `O(T·|A|)` — avoids the #P-complete linear-extension count.

## 4. PDAF priors / weak supervision (the integration point)

- `local_orders.jsonl` → informative prior on each `U^(r)` (typed CPA edges; `INCOMPARABLE` → direct
  `∥` evidence; confidence → prior strength). Replaces BPOP's uniform 1/3 per-pair prior.
- `global_orders.jsonl` → prior on `Q`.
- `skill_name` + `SKILL_LIBRARY` → seed the library and bias the CRP assignment `z_i`.
- The annotator's **`skill_is_new` flag is the empirical CRP "new-table" event** — PDAF tells the
  model when to consider opening a new skill.

## 5. Inference (sketch)

Collapsed Gibbs over `z` (DP-mixture style; data term = the local frontier-softmax fit), split/merge
moves on skills, RJMCMC on each `K_r` and `K_g`, MH on `β_ℓ, β_g, α`; semi-Markov decoding for
segmentation (HPOP already uses this). Initialize from PDAF priors. Recovered structure compiles into
a frontier executor (parallel where `∥`, no re-planning) for fast inference.

## 6. Build mapping

- `src/hpop/inference/` — two-level frontier-softmax likelihood + latent-U posets + CRP/PY library
  prior with the new-skill penalty; MCMC. (Reuse archived `linext` for exact-NLE / queue-jump baselines.)
- `src/hpop/extract/` — turn `local_orders.jsonl` / `global_orders.jsonl` into the prior edge
  structures (`H_r`, `Q`) the model consumes.
- `src/hpop/library/` — the skill library `{P_r}` (store, index, reuse), with usage counts `n_r`.
