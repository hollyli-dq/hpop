# FULL-LATENT preregistration

Status: preregistered; the only preliminary prefix was discarded before a checkpoint,
gate, or retained formal draw. The corrected formal launch is pending.

This experiment asks: **When skill identities are learned jointly rather than externally anchored, does exact path marginalization enable reliable recovery of the reusable partial-order library, segmentation, skill co-clustering, and transition dynamics?**

It was designed after Conditions C/C′ showed that fixed asymmetric \(\pi^*,P^*\) create an anchored structure-to-index assignment problem. FULL-LATENT instead targets the matched Stage6E finite-Markov posterior

\[
p(S,z,U,\pi,P\mid X,\rho_0,\beta^*,\omega^*,\lambda_{rep}^*,
\lambda_{back}^*,\delta_B^*,\epsilon^*),
\]

with \(S,z,U,\pi,P\) inferred jointly and evaluated modulo one global latent-label permutation. The authoritative model is `src/hpop/mcmc_original/stage6e_frozen.py`: \(K=3\), \(\pi\sim\mathrm{Dir}(1,1,1)\), and independently \(P_{h,-h}\sim\mathrm{Dir}(1,1)\), with \(P_{hh}=0\) exactly and no terminal transition.

The fixed coordinates are \(\rho_0=0.5\), \(\beta=1.5\), \(\omega=1.7346010553881064\), \(\lambda_{rep}=0.8\), \(\lambda_{back}=0.25\), \(\delta_B=0.15\), and \(\epsilon=0.02\). The corpus’s supplied truth has no generating rho; 0.5 is the already-registered Condition B/C isolation value.

## Arms and valid composition

Both arms make exactly one uniformly chosen `(skill,row)` Gaussian U proposal every 10 sweeps, at scale 0.5. This is the parent registered generic row scale. C’s confounded additive schedule is not reused.

- FULL-COND scores that proposal conditional on the current explicit path.
- FULL-MARG scores it with \(\sum_n\log Z_n(U;\pi,P)\), where the forward DP uses the current learned \(\pi,P\).

Both then execute exact FFBS of all \((S,z)\), P Gibbs, and pi Gibbs. In FULL-MARG nothing path-dependent occurs between the marginal MH decision and FFBS, including after a rejection. Thus the standard partially collapsed composition is valid: marginal U MH leaves \(p(U\mid X,\pi,P)\) invariant, FFBS restores the joint path conditional, and the two Gibbs updates condition on those fresh paths.

The frozen corpus is `results/mcmc_original/matched_synthetic_formal_corpus`, with corpus hash `dd280a4a09896154e167f388edd401a9119ba398167c09404aba5f7743e58ec2`, train hash `717b77a4cff740c7811c9becb6e43af68d286dcc1c6c6ba6feb475208fef6541`, held-out hash `4d586fc5e26ec356cdb00d496a64be9d1071ee8ef9d112ab89d1246c46bee23f`, and truth hash `fc41538fd44d170df8d0a6401f0c6e6b49d52418c487e22f9e4f45ee047f903e`. Online code reads only CPA arrays; truth recovery is separate and terminal-only.

## Prelaunch integrity amendment

The initial zero-checkpoint prefix is recorded in
`results/mcmc_original/matched_full_latent/DISCARDED_precheckpoint_attempt.json` and
contributes no scientific result. It was halted after finding resume gate bookkeeping and
legacy import-seal defects. Before the valid formal launch, gate records were made atomic
and exact-checkpoint-specific; a fresh-process probe now prohibits generator/truth module
imports; retained U draws use exact float64 storage; and the terminal-only recovery driver
is frozen. These corrections do not change the target, data, priors, starts, kernel
schedule, thresholds, or no-rescue rule.

There are four paired starts per arm. U uses prior C’s wrong-structure seeds `6204101–6204104` and scales `(0.5,1,2,3)`; pi/P use new truth-free prior-draw seeds `6206101–6206104`. Chain seeds are `6206201–6206204` for FULL-COND and `6206211–6206214` for FULL-MARG. No C/C′ terminal state or truth alignment initializes any chain.

## Gates and recovery

Burn-in is 10,000 sweeps, thinning 5, and gates are at 30k, 50k, 75k, and 100k. An arm stops after two consecutive passes; 100k is a hard ceiling. The inherited thresholds are rank-normalized split-\(\hat R\le1.01\), log-target bulk/tail ESS at least 1000/500, total-relations bulk ESS at least 1000, and other finite invariant structural/probe summaries bulk ESS at least 500. Each chain must accept at least one H-changing structural proposal.

Gates use only label-invariant summaries: generic structural/segmentation summaries, a fixed corpus-hash-selected boundary probe set, co-skill probes, sorted pi statistics, and simultaneous-permutation invariants of P. Every summary is tested over all six K=3 permutations before launch. Raw indexed skills are never a primary convergence gate.

Only after both arms are terminal is truth unsealed. Matching uses closure-Hamming Hungarian assignment with deterministic lexicographic tie-breaking, and the same assignment aligns H, pi, P, and z. The terminal report includes unordered library recovery, closure F1/Hamming, boundary and co-skill recovery, pi/P errors, and exact held-out posterior-predictive NLL per occurrence using \(\log Z_n-\log C_{J_n}\) followed by log-mean-exp across posterior draws.

No global permutation/swap move will be added after launch. No rescues, tempering, changes to data/model/K/fixed nuisances/scales/thresholds, truth-informed starts, or held-out tuning are permitted.
