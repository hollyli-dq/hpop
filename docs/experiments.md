# HPOP experiments: implementation, recovery study, and findings

This document records the first executable test of the manuscript's two new modelling components:
the **recurrent relaxed frontier likelihood** and the **merge-only semi-Markov skill segmentation**.
Before this work `src/hpop/inference/` contained only the flat, non-recurrent, hard-poset BPOP
frontier likelihood, so none of the manuscript's claims could be run.

**Headline.** On synthetic data with known ground truth the model works and the new components earn
their place: joint HPOP recovers skill assignments at ARI 0.921 and local order at edge F1 0.848,
within reach of the oracle-boundary ceiling, and removing the recurrent validity state costs
0.17 local edge F1. On the real SWE-rebench pilot HPOP beats every model in its own family but
**loses to a plain bigram by 0.35 nats per occurrence**, and it does *not* beat a bigram at
distinguishing legal reorderings from dependency violations either. The manuscript's plan to
headline next-occurrence likelihood against n-gram baselines will not survive contact with these
numbers; Secs. 5–7 say what to claim instead. One caveat matters: the real-data CPAs are rule-based
**silver** labels, which biases that comparison toward the bigram (Sec. 1b). Implementing the
equations also surfaced seven specific issues in the manuscript (Sec. 7), each with a test.

## 1. What was built

| Module | Contents |
| --- | --- |
| `src/hpop/inference/recurrent.py` | Validity state `q_t`, soft invalidation `J~ = D · sigmoid(omega)`, recurrent frontier `F~^RFS`, successor utility, repeat and backward-jump costs, step softmax, sequence likelihood, analytic `theta` gradient. |
| `src/hpop/inference/semi_markov.py` | Exact forward/backward and Viterbi over the merge-only seed lattice with the `z_l != z_{l+1}` constraint and the `D_max` width cap, plus a brute-force enumerator used as a test oracle. |
| `src/hpop/inference/hpop.py` | Variational EM: exact structured E-step, `theta` gradient ascent, weighted pairwise-consensus structure estimation with likelihood-guided pruning, Dirichlet(`alpha/K_max`) library sparsity, normalized held-out likelihood, type-level global order. Batched over the library. |
| `src/hpop/synth/generator.py` | Ground-truth skill library, type-level global DAG, exogenous failure-triggered repair loops, simulated LLM oversegmentation with tunable boundary recall. |
| `src/hpop/eval/metrics.py` | Boundary P/R/F1, skill ARI, Hungarian slot matching, closure- and cover-edge F1, global edge F1, library-size error. |
| `tests/test_recurrent.py`, `tests/test_semi_markov.py`, `tests/test_hpop.py` | 52 tests, including exact agreement with brute-force enumeration. |

Reproduce:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python scripts/exp_synthetic_recovery.py --seeds 5 --traces 40 --iters 12
PYTHONPATH=src .venv/bin/python scripts/exp_boundary_recall.py  --seeds 3 --traces 30 --iters 12
PYTHONPATH=src .venv/bin/python scripts/exp_order_invariance.py --seeds 6 --traces 40 --swaps-per-trace 25
PYTHONPATH=src .venv/bin/python scripts/exp_real_pilot.py --lam-seg 3.0 --iters 18 --K-max 10 --D-max 12
PYTHONPATH=src .venv/bin/python scripts/exp_real_pilot.py --n-train 16   # data-efficiency point
```

Raw outputs are written to `data/experiments/`.

## 1b. Data sources and setup

**Real data.** `data/modelling/swe_rebench/pilot100.sequences.jsonl`, derived from
`nebius/SWE-rebench-openhands-trajectories`: 100 trajectories over 87 repositories, 50 resolved /
50 unresolved, **5,012 CPA occurrences** (mean 50.1 per trace, range 23–89) over a **16-type CPA
vocabulary** — so each local partial order has 16 candidate nodes. Split is repository-disjoint at
split seed 0: 68 train trajectories / 60 repos / 3,316 occurrences → 32 test trajectories /
27 *unseen* repos / 1,696 occurrences. Seeds are the maximal oversegmentation, one per occurrence.

> **The real-data CPA labels are silver, not gold.** They come from `hpop.annotate.rule_apply`
> (annotator tag `rule_apply_seed-v0`), a deterministic rule annotator that maps each event to a seed
> CPA using tool name, command keywords, failure context and one bit of state. All 5,012 occurrences
> are `MATCH_EXISTING` at a constant confidence of 0.75; there is no LLM open-coding, no human
> adjudication, no `PROPOSE_NEW` or `ABSTAIN`, and the vocabulary is the 16-entry *seed* library
> (`rules/cpa_library_seed.json`), not the induced 29/32-CPA dictionary used elsewhere in this repo.
> A deterministic tool→label rule makes local transitions unusually predictable, so **Sec. 5 is
> biased in the bigram's favour by an unknown amount.** Rerunning it on LLM-annotated CPAs is the
> most important follow-up before the negative result is treated as settled.

**Synthetic data.** `src/hpop/synth/generator.py`, 5 seeds × 40 traces = 200 traces,
**6,922 occurrences**, 900 skill instances, over a **12-type CPA vocabulary**. Ground truth per seed:
K = 4 skills with **3–5 role nodes** each (mean 4.2) and mean 3.5 cover edges, plus a type-level
global DAG over the 4 skills. Each trace has 3–6 skill instances, ~40 occurrences and ~16 seed
segments. 28 train / 12 test traces per seed.

**Model configuration**, identical across every method compared within a table:

| Setting | Synthetic (Secs. 3, 4, 6) | Real pilot (Sec. 5) | Meaning |
| --- | --- | --- | --- |
| `V` | 12 | 16 | nodes in each local partial order |
| `K_max` | 6 | 10 | library slots (truncation level) |
| `K₊` inferred | ≈5 (true 4) | 9 active, 17 learned edges | slots above 1% expected usage |
| `D_max` | 8 | 12 | max seeds merged into one instance |
| `lam_seg` | 1.0 | 3.0 | per-instance penalty (swept on real data) |
| EM iterations | 12 | 18 | exact E-step + M-step |

Fixed throughout: `beta=1.5`, `lam_rep=1.5`, `lam_back=0.5`, `omega=2.5` (σ(ω)=0.92), `eps=0.02`,
`alpha=1.0`, `lam_edge=0.8`, `edge_threshold=0.90`, `min_support=0.25`. Ablations flip exactly one
flag: `use_order=False` gives the HSMM, `use_recurrence=False` sets σ(ω)=0.

**The four experiments, as run:**

| # | Experiment | Data | Scale | Question |
| --- | --- | --- | --- | --- |
| 1 | Synthetic recovery (Sec. 3) | synthetic, ground truth | 5 seeds × 40 traces | boundaries, assignments, local + global order |
| 2 | Initialization audit (Sec. 4) | synthetic, recall swept | 3 seeds × 30 traces × 5 recall levels | when merge-only breaks |
| 3 | Real transfer + data efficiency (Sec. 5) | SWE-rebench pilot, silver CPAs | 100 traj, repo-disjoint, 4 train sizes | held-out prediction on unseen repos |
| 4 | Reorder invariance (Sec. 6) | synthetic, ground-truth incomparability | 6 seeds × 40 traces, 441 valid + 978 invalid swaps | invariance to legal reorderings |

All error bars are ±95% CI across independent generator seeds, not across traces. Every method in a
table sees the same held-out representation; `HPOP, oracle boundaries` receives true boundaries at
training time only. Real-data results are a **single** split — no error bars across splits.

## 2. Correctness checks

The semi-Markov lattice is verified against exhaustive enumeration of every legal
(composition, labelling) pair for six lattice shapes: log-partition, all segment marginals, Viterbi
path and Viterbi value agree to `1e-9`. Posterior marginals sum to exactly 1 over the blocks
covering each seed position, and the closed-form entropy `H = log Z - E_q[score]` matches the
enumerated entropy.

For the recurrent likelihood, Eq. (frontier-equivalence) is checked directly: with invalidation
switched off, `{x : F^RFS(x) = 1, q(x) = 0}` equals the BPOP frontier of the completed set at every
step of a non-repeating trace. The analytic `theta` gradient matches finite differences to 4 decimal
places, and the batched library likelihood matches the single-skill implementation to `1e-9`.

## 3. Synthetic recovery (fills `tab:synthetic`)

5 seeds, 40 traces each (28 train / 12 test), 4 true skills, 12 CPA types, `K_max = 6`,
`D_max = 8`, 12 EM iterations. Mean trace length ~40 CPA occurrences; **39% of occurrences inside a
skill instance are re-executions**, so the corpus genuinely exercises the recurrent likelihood.
Mean ± 95% CI over seeds.

| Method | Skill ARI ↑ | Boundary F1 ↑ | Local edge F1 ↑ | Global edge F1 ↑ | NLL/occ ↓ | \|K₊−K\| ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| HSMM (segmentation, no order) | 0.800 ± 0.054 | 0.831 ± 0.068 | 0.000 | 0.693 ± 0.362 | 1.661 ± 0.024 | 1.80 |
| Flat poset (K=1, no segmentation) | 0.000 | 0.000 | 0.216 ± 0.298 | 0.000 | 2.207 ± 0.126 | 3.00 |
| HPOP, oracle boundaries | 0.948 ± 0.037 | 0.911 ± 0.052 | 0.875 ± 0.092 | 0.705 ± 0.366 | 1.397 ± 0.099 | 0.60 |
| **HPOP, joint** | **0.921 ± 0.058** | **0.891 ± 0.067** | **0.848 ± 0.072** | 0.693 ± 0.362 | **1.399 ± 0.089** | 1.20 |
| HPOP, no recurrence (ablation) | 0.876 ± 0.015 | 0.870 ± 0.025 | 0.679 ± 0.101 | 0.633 ± 0.364 | 1.511 ± 0.060 | 1.20 |

Readings:

1. **Merge-only segmentation is nearly free.** Joint inference reaches 97% of the oracle-boundary
   ARI (0.921 vs 0.948) and the same held-out NLL, when the seed set has full boundary recall.
2. **The recurrent validity state is what buys the local order.** Removing invalidation costs
   0.17 local edge F1 (0.848 → 0.679) and 0.11 nats/occurrence. Without it, every re-execution is
   pure repeat cost, so the estimator cannot tell a genuine dependency from a loop artefact.
3. **Order and segmentation each carry distinct information.** HSMM matches HPOP on boundaries
   (composition alone segments well) but recovers no order and is 0.26 nats worse; the flat poset
   fails outright.
4. **Global edge F1 is underpowered** in this configuration (CI ±0.36): with 4 skills the
   ground-truth global DAG often has 1–2 edges, so a single miss moves F1 by 0.5. This column
   should not be reported without a denser global structure or more skills.

## 4. Merge-only initialization audit

The manuscript requires this check: a true boundary absent from the seed set is unrecoverable by
construction. Sweeping the simulated LLM's boundary recall (3 seeds, 30 traces each):

| target recall | seed boundary recall (ceiling) | model boundary recall | Boundary F1 | Skill ARI | Local edge F1 | NLL/occ |
| --- | --- | --- | --- | --- | --- | --- |
| 1.00 | 1.000 | 0.906 | 0.889 | 0.940 | 0.886 | 1.364 |
| 0.90 | 0.957 | 0.896 | 0.886 | 0.915 | 0.901 | 1.411 |
| 0.75 | 0.781 | 0.710 | 0.724 | 0.821 | 0.712 | 1.579 |
| 0.50 | 0.545 | 0.347 | 0.358 | 0.592 | 0.542 | 1.686 |
| 0.25 | 0.304 | 0.179 | 0.184 | 0.568 | 0.362 | 1.783 |

The model tracks the analytic ceiling closely down to ~0.78 seed recall and then falls *below* it:
missing boundaries do not merely cap segmentation, they corrupt the skills learned from the
oversized blocks, which in turn degrades the segmentation of the rest. The practical threshold is
around **0.9 seed boundary recall**; below it the merge-only restriction should be relaxed or the
prompt made more conservative, exactly as the manuscript's audit paragraph anticipates.

## 5. Real SWE-rebench pilot (100 annotated trajectories)

100 OpenHands trajectories with **rule-based silver CPA labels** (see Sec. 1b — this qualifies the
result below), 16 CPA types, mean length 50 occurrences,
**74% of occurrences per trajectory are repeats of a label seen earlier in that trajectory**.
Repository-disjoint split: 68 train trajectories over 60 repositories, 32 test trajectories over 27
unseen repositories (1,696 held-out occurrences), so this is the manuscript's cross-repository
transfer setting with the library frozen. The pilot has no LLM phase segmentation attached and
consecutive labels almost never repeat (mean run length 1.01), so we use the maximal
oversegmentation — one seed per occurrence — which makes the merge-only restriction vacuous and
isolates the model from seeding error.

Held-out NLL in nats per CPA occurrence (lower is better):

| Model | NLL/occ | vs uniform | vs bigram |
| --- | --- | --- | --- |
| Uniform over 16 CPAs | 2.773 | 0.000 | −1.093 |
| Unigram | 2.350 | 0.423 | −0.670 |
| **Bigram (add-0.5)** | **1.680** | **1.093** | **0.000** |
| Flat poset (K=1) | 2.553 | 0.220 | −0.873 |
| HSMM (composition only) | 2.128 | 0.644 | −0.449 |
| HPOP, no recurrence | 2.087 | 0.686 | −0.407 |
| HPOP | 2.030 | 0.743 | −0.350 |

(HPOP rows at the best setting found, `lambda_seg = 3`, `K_max = 10`, `D_max = 12`, 18 iterations;
baselines are setting-independent.)

**This is a negative result and it is robust.** HPOP orders correctly against everything in its own
family — it beats the flat poset by 0.52 nats, the composition-only HSMM by 0.10, and its own
no-recurrence ablation by 0.06 — but a plain add-0.5 **bigram beats it by 0.35 nats**. We swept
`lambda_seg` over {0.25, 1, 3, 6}, `K_max` over {8, 10}, `D_max` over {10, 12} and 12/18 EM
iterations; the ordering never changed, so this is not under-training or a hyperparameter artefact.

The reason is structural. Inside a skill, `log F~` is ~0 wherever the learned poset has no edge, so
with sparse local structure the step distribution collapses towards a per-skill unigram with a mild
validity adjustment. A partial order deliberately *discards* the local sequential regularity
(`EDIT_SOURCE` then `VERIFY_FIX`) that a bigram captures for free. Decoded held-out instances average
6.9 CPAs and **75.7% of them re-execute a role**, which strongly supports the repeated-execution
motivation for the recurrent likelihood even though it does not close the predictive gap.

**Implication for the manuscript.** Sec. `sec:experiments` currently lists held-out
next-occurrence log likelihood as a headline metric and n-gram models as baseline (i). On this
evidence that framing will lose. What survives is structure recovery (Sec. 3), compression relative
to order-free and flat models, and an interpretable reusable library — not next-action prediction,
and, as Sec. 6 shows, not reorder-invariance either. Either the claim is reframed, or the
within-skill likelihood gains a local sequential component alongside the partial order (for example
a per-skill transition term restricted to the frontier-feasible set, which would keep the
order semantics while recovering what the bigram exploits).

**The usual defence — "structured models win when data is scarce" — does not hold here either.**
Subsampling the training set (same repository-disjoint test set, same settings) leaves the gap flat:

| train trajectories | unigram | bigram | HSMM | HPOP no-rec | HPOP | HPOP − bigram |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 2.361 | 1.832 | 2.168 | 2.164 | 2.166 | +0.333 |
| 16 | 2.354 | 1.754 | 2.124 | 2.093 | 2.086 | +0.332 |
| 32 | 2.346 | 1.703 | 2.089 | 2.061 | 2.059 | +0.357 |
| 68 (all) | 2.350 | 1.680 | 2.128 | 2.087 | 2.030 | +0.350 |

Both models improve with data and the gap stays at ~0.34 nats throughout, so the bigram's advantage
is representational, not a large-sample effect: with 16 CPA types, local transition structure is
cheap to estimate and a partial order deliberately throws it away.

The learned library is readable and matches software-agent procedure, which is evidence for the
skill-induction half of the model even where prediction loses:

| Skill | Usage | Composition (top roles) | Learned cover edges |
| --- | --- | --- | --- |
| 7 | 18.9% | WRITE_TEST, RUN_TEST_SUITE, VERIFY_FIX, REPRODUCE_ISSUE | WRITE_TEST → VERIFY_FIX → REPRODUCE_ISSUE |
| 9 | 16.9% | READ_SOURCE, LOCATE_CODE, DIAGNOSE_FAILURE, EXPLORE_REPOSITORY | — (order-free localization) |
| 5 | 16.8% | VERIFY_FIX, REPRODUCE_ISSUE, EDIT_SOURCE, RUN_TEST_SUITE | EDIT_SOURCE → VERIFY_FIX → {REPRODUCE_ISSUE, RUN_TEST_SUITE} |
| 4 | 16.3% | RUN_TEST_SUITE, READ_SOURCE, DIAGNOSE_FAILURE, REPRODUCE_ISSUE | READ_SOURCE → RUN_TEST_SUITE; EXPLORE_REPOSITORY → {REPRODUCE_ISSUE, DIAGNOSE_FAILURE} |
| 8 | 12.4% | VERIFY_FIX, WRITE_REPRODUCTION_SCRIPT, READ_SOURCE, EDIT_SOURCE | WRITE_REPRODUCTION_SCRIPT → VERIFY_FIX |
| 2 | 7.0% | SUBMIT_SOLUTION, CLEANUP_ARTIFACTS, REPRODUCE_ISSUE, RUN_TEST_SUITE | CLEANUP_ARTIFACTS → SUBMIT_SOLUTION |

`EDIT_SOURCE → VERIFY_FIX`, `WRITE_TEST → VERIFY_FIX`, `WRITE_REPRODUCTION_SCRIPT → VERIFY_FIX` and
`CLEANUP_ARTIFACTS → SUBMIT_SOLUTION` are all genuine procedural dependencies recovered without
supervision, and skill 9 is correctly left order-free — searching and reading during localization
really are interchangeable.

## 6. Invariance to legal reorderings (synthetic, ground-truth incomparability)

Held-out next-occurrence likelihood rewards memorizing the serialization the agent happened to use.
The claim that actually motivates partial orders is different: two executions that differ only in
the order of *incomparable* actions are the same program and should score alike, while an order that
breaks a genuine dependency should not. We test this non-circularly — incomparability is read from
the **ground-truth** local posets, never from the fitted model. For each held-out trace we swap
adjacent occurrences that are truly incomparable (*valid swap*) or truly ordered (*invalid swap*)
and measure the change in per-occurrence NLL. 6 seeds, 40 traces, all available swaps
(10–133 valid and 97–241 invalid swaps per seed).

| Model | valid swap ΔNLL (want ≈ 0) | invalid swap ΔNLL (want > 0) | discrimination |
| --- | --- | --- | --- |
| Bigram | 0.0014 ± 0.0064 | 0.1042 ± 0.0159 | 0.1028 ± 0.0148 |
| HSMM | −0.0005 ± 0.0009 | 0.0007 ± 0.0008 | 0.0012 ± 0.0005 |
| HPOP | 0.0004 ± 0.0005 | 0.0905 ± 0.0144 | 0.0901 ± 0.0146 |

Paired per-seed differences against the bigram: valid-swap cost −0.0010 ± 0.0063 (no difference),
discrimination −0.0127 ± 0.0127 (HPOP marginally *worse*).

**The hypothesis that HPOP wins here is not supported.** HPOP is invariant to legal reorderings —
0.0004 ± 0.0005 nats, indistinguishable from zero and five times more tightly concentrated across
seeds than the bigram — and it strongly rejects dependency violations. But the bigram does the same
thing about as well: with 12 CPA types and enough traces, it simply observes incomparable pairs in
both orders and learns both transitions. The one unambiguous result is the HSMM row: with
segmentation and composition but no order, discrimination collapses to 0.001, confirming that the
partial order — not the skill segmentation — is what makes HPOP violation-sensitive.

A structured model should win over a bigram where transitions are *unobserved* rather than plentiful
— but the data-efficiency sweep in Sec. 5 tested that directly on real traces and found the gap flat
from 8 to 68 training trajectories. With a CPA vocabulary this small, the transition table is simply
cheap to estimate. If the partial-order representation is to beat sequence models on likelihood, it
will have to be at much larger vocabularies, or by adding the local transition term back.

## 7. Findings for the manuscript

These came out of implementing the equations as written. Each is backed by a test.

**(a) Eq. (recurrent-step-likelihood) has no noise floor.** With a hard precedence matrix — or any
relaxed `D_U` that saturates at 1 — `F~^RFS(x)` is exactly 0 whenever a predecessor is stale, so an
order-violating observation has probability 0 and the log-likelihood is `-inf`. BPOP avoids this
with an epsilon trembling-hand term; the recurrent likelihood as written drops it. The
implementation mixes in `eps/M` and documents `eps = 0` as the manuscript's exact equation.
*Test:* `test_eps_zero_gives_violations_zero_probability`.

**(b) Invalidation cannot start a repair loop.** `J~(z,x) = D_U(z,x) · sigmoid(omega_zx)` is masked
by precedence, so invalidation only flows *forward*: executing an upstream node staleifies its
descendants. A **failed test therefore cannot invalidate the edit that preceded it** — the very
edit–test–repair cycle the section is motivated by. Under the model the loop's *restart* is an
unexplained move charged `lambda_rep`; only the recompute cascade that follows it is explained. In
software traces the trigger is observable (the test's exit status), so the natural fix is to
condition invalidation, or `lambda_rep`, on the observed outcome of the preceding occurrence.
*Test:* `test_invalidation_only_flows_along_precedence`.

**(c) `lambda_rep` is not a local repeat cost.** It penalizes *every* currently-valid item at every
step, not just the one that repeats. Raising it therefore suppresses all valid competitors and can
*increase* the likelihood of a trace containing repeats — verified on a 5-step trace where
`lambda_rep = 4` scores higher than `lambda_rep = 0`. The parameter is a global reshaping of the
step distribution and should not be interpreted, or identified, as "the cost of repeating".
*Test:* `test_repeat_cost_does_not_lower_whole_sequence_likelihood`.

**(d) The ELBO's segmentation prior is unnormalized.** Eq. (structured-segmentation-elbo) carries
`sum_l [log pi_{z_l} - lambda_seg]` without dividing by its partition function over the seed
lattice, so the implied prior over segmentations does not sum to 1 and raw `log Z` is not a
likelihood. It is still a usable training objective, but held-out numbers are not comparable across
models or `lambda_seg` settings. The implementation reports
`log p(x) = log Z(score) - log Z(prior-only score)`, which is the exact marginal likelihood.
*Tests:* `test_uniform_model_gives_exact_uniform_likelihood` (recovers `-T log V` exactly) and
`test_likelihood_is_invariant_to_the_segment_penalty`.

**(e) The amortized inference network is unnecessary.** Both the model and the variational family
in Eq. (structured-semi-markov-posterior) are semi-Markov over the same merge-only lattice, so
setting the potential `eta_phi(a,b,k)` equal to the expected generative segment score
`l_n(a,b,k)` of Eq. (expected-generative-segment-score) makes `q_phi` the *exact* conditional
posterior, computable in `O(J · D_max · K^2)`. The paper can drop the inference network and its
hyperparameters and state that the structured E-step is exact — a strictly stronger claim. (As
written, `l_n` is defined and then never connected to `eta_phi`; this closes that gap.)

**(f) The per-skill weight has no composition term.** Fig. (overview) describes each skill by "a
CPA-composition distribution *and* a revisitable local partial order", but
Eq. (recurrent-unnormalized-weight) contains only order terms. Two skills over disjoint CPA sets are
then distinguished only through their order structure, which is weak: EM initialized at
`theta = 0` produces a near-uniform emission model and a flat segmentation posterior. The
implementation adds per-skill composition logits `theta_k` (with `theta = 0` reproducing the
manuscript exactly); note that composition alone already gets HSMM to 0.80 ARI, so this term is
doing much of the identification work and belongs in the equations.

**(g) `K = 1` is degenerate under Assumption (ass:no-self-transition).** With a single skill slot,
the no-adjacent-same-label rule makes every multi-segment tiling illegal, so a trace longer than
`D_max` seeds has zero probability. Harmless in practice, worth a footnote.

## 8. Scope and limitations of this study

* **The real-data CPA layer is silver.** Deterministic rule labels, constant confidence 0.75, seed
  vocabulary rather than the induced dictionary (Sec. 1b). Rule-derived labels have artificially
  regular local transitions, so Sec. 5 is biased in the bigram's favour by an unknown amount.
  Rerunning on LLM-annotated CPAs is the most important follow-up.
* Real-data results come from a **single** repository-disjoint split (seed 0); only the synthetic
  experiments have seed-level error bars.
* The generator is deliberately **not** the fitted model: repair loops are triggered exogenously by
  verification failures (see (b)), so the recovery numbers are measured under model mismatch rather
  than in a well-specified setting.
* Structure estimation is a weighted first-occurrence pairwise-consensus initializer followed by
  likelihood-guided pruning under the cover-edge penalty, not the reparameterized-gradient
  optimization of the relaxed `D_U` the manuscript describes (no autodiff in this environment). The
  relaxed likelihood is implemented and tested; only its gradient-based *fitting* is substituted.
* The global order is estimated at the **skill-type** level using the same recurrent machinery, not
  at the instance level. The recurrent likelihood already represents repeated instances of a type
  without cycles, which suggests the instance-level poset may be avoidable; this deserves a direct
  comparison before the paper commits to either.
* Held-out NLL under the MAP-free marginalization conditions on the seed boundaries, which are
  observed input for every method compared.
