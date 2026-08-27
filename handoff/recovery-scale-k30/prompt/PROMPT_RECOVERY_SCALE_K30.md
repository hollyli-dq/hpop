# One-week K-recovery scalability experiment — execution prompt

Saved verbatim. Give this to Claude Code (or a human operator) on the target machine.

**Before running it, read `HANDOFF_NOTES.md` in the package root.** Section 1 has been
amended for this code-only package and the amendment is marked inline. Three further
items in the shipped code still need work before a run: the model does not support
A != m, the terminal gate is hard-wired to K = 3, and the nested master library is
unimplemented. All are documented there.

---

We need a ONE-WEEK RECOVERY-SCALABILITY EXPERIMENT for the final segmental
partial-order skill model.

This is a formal synthetic RECOVERY study, not a runtime-only benchmark.

The scientific question is:

    When trace length and per-skill evidence are controlled, how does recovery
    degrade as the reusable skill-library size grows from K=3 to K=30?

The registered skill-library ladder is:

    K in {3, 5, 10, 20, 30}

Trace length is fixed:

    J = 96

Do NOT add a long-trace axis. J must remain fixed for every formal condition.

Run only the paper's actual inference method:

    optimized path-marginal segmental partial-order inference

Do NOT run the conditional ablation on this grid. The small matched experiment
already provides that ablation.

============================================================
0. ONE-WEEK AUTONOMOUS EXECUTION CONTRACT
============================================================

I authorize an unattended run lasting at most 7 calendar days.

Do not ask me routine questions after launch.

Total hard wall-clock limit:

    168 hours

Reserve:

    at least 12 hours for terminal diagnostics, truth recovery, held-out NLL,
    figures, reports and integrity checks.

Formal sampling and pilots must therefore finish by:

    launch time + 156 hours

Use a resumable state machine with:

    atomic state.json
    atomic checkpoint files
    events.jsonl
    heartbeat.json
    progress.md
    failure_manifest.json
    deadline_manifest.json

Every result-producing operation must be restartable from its last durable
state without reseeding.

Maximum automatic retries:

    2 per failed process

Retries must:
- restore the exact saved RNG state;
- restore the exact chain state;
- preserve the same seed;
- preserve all previous draws;
- never restart from truth or a favourable posterior state.

If the hard deadline is reached:
- stop at the next durable checkpoint;
- preserve all completed and partial configurations;
- mark unfinished conditions CENSORED;
- do not shorten production retrospectively;
- do not pretend censored runs are comparable to completed runs.

Stop immediately only for:
- optimized/reference parity failure;
- source-seal violation;
- truth leakage;
- corrupted checkpoint;
- invalid probability/NaN in the sampler;
- irreconcilable model/generator mismatch;
- predicted unsafe memory use;
- absence of the required optimized backend.

============================================================
1. REPOSITORY AND BRANCH ISOLATION
============================================================

[AMENDED for the code-only package. The original text required a git ancestry
check against commit 564995efd056d7d33984f0ca1532386e6140ea0c. This repository
ships the code without the source repository's history, deliberately: no
historical experiment is needed to run this study, and the source history
carries a 176.5 MB obsolete blob that exceeds GitHub's hard limit. Removing that
blob would rewrite 90 commits and change the very hash the check verifies. The
substitute below tests file CONTENT, which is a stronger statement than
reachability. The original wording is preserved in HANDOFF_NOTES.md.]

Verify the optimized backend by content rather than by ancestry:

    python verify_environment.py

This must print RESULT: READY. It asserts that every shipped source file matches
SOURCE_INTEGRITY.json, and in particular that

    src/hpop/mcmc_original/
    src/hpop/mcmc_optimized/

are byte-identical to the trees of validated backend commit

    564995efd056d7d33984f0ca1532386e6140ea0c

as recorded when the package was built (199 files, zero drift, verified with
git hash-object against that commit).

If integrity fails, stop and report that the package is corrupt or has been
edited. Do not recreate the optimized backend manually.

Initialise the working branch:

    branch:
        recovery-scale-k30-v1

    git checkout -b recovery-scale-k30-v1

Record the package's own base commit and the validated backend commit that
SOURCE_INTEGRITY.json pins.

The package must:
- include the optimized backend (verified above);
- include the confirmatory generator and diagnostic machinery;
- leave every sealed reference source unchanged.

There are no historical experiment artifacts in this package and none are
required. This study generates its own sealed truths and corpora as its first
step.

Do not edit:
- mcmc_original sealed sources;
- historical A/B/C/C-prime artifacts;
- previous confirmatory artifacts;
- old scalability artifacts;
- the paper's main TeX files.

New harness code, preregistration, results and paper-ready snippets may be
added only on recovery-scale-k30-v1.

============================================================
2. HARDWARE AND ENVIRONMENT PREFLIGHT
============================================================

Record:

- operating system;
- CPU model;
- physical and logical cores;
- performance/efficiency core topology if applicable;
- total and available RAM;
- swap;
- Python version;
- NumPy version;
- SciPy version;
- BLAS implementation;
- git commit;
- filesystem free space.

Set for every worker:

    OMP_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1
    MKL_NUM_THREADS=1
    NUMEXPR_NUM_THREADS=1
    VECLIB_MAXIMUM_THREADS=1

Verify no other formal HPOP job is active.

Do not kill or renice an unknown existing job. If one exists, report and stop
before launch.

Run:

1. optimized-backend focused tests;
2. generator/inference parity tests;
3. terminal-gate tests;
4. recovery-alignment tests;
5. the complete project suite if it completes within 90 minutes.

The optimized/reference parity gate must satisfy:

    max |alpha_opt - alpha_ref| <= 1e-10
    max |logZ_opt - logZ_ref| <= 1e-10
    identical -inf pattern
    identical legal support
    valid backward draws
    no NaN
    valid pi and P

Record actual observed discrepancies.

============================================================
3. FIXED SCIENTIFIC DESIGN
============================================================

The only formal scaling axis is:

    K = 3, 5, 10, 20, 30

Fix for every formal condition:

    J = 96
    A = 50
    role support size m_k = 10 for every skill
    D_min = 3
    D_max = 12
    rho = 0.5
    recurrent nuisance parameters = the final confirmatory values
    structural-update cadence = 1 per 10 sweeps
    thin = 5
    checkpoint every 2,000 sweeps

Use the same probabilistic model and priors as the final optimized
FULL-LATENT confirmatory experiment.

Do not infer:
- K;
- CPA vocabulary;
- role inventories;
- role-to-CPA maps;
- rho;
- recurrent nuisance parameters.

Infer jointly:

    S, z, U -> H, pi, P

No conditional arm.
No alternative sampler.
No new move.
No sparse-P model.
No approximate DP.
No Q_k initializer.
No banded-memory implementation.
No tuning based on recovery.

============================================================
4. COVERAGE-MATCHED CORPUS SIZES
============================================================

For each K use:

    N_train = 5 K
    N_test  = 2 K
    J       = 96 for every trace

Therefore:

| K  | N_train | N_test | train occurrences | expected train occurrences/skill |
|----|---------|--------|-------------------|----------------------------------|
| 3  | 15      | 6      | 1,440             | approximately 480                |
| 5  | 25      | 10     | 2,400             | approximately 480                |
| 10 | 50      | 20     | 4,800             | approximately 480                |
| 20 | 100     | 40     | 9,600             | approximately 480                |
| 30 | 150     | 60     | 14,400            | approximately 480                |

This design is fixed.

Do not reduce N at larger K to save time.
Do not increase N after observing weak recovery.
Do not vary J across K.

The claim tested is conditional on approximately coverage-matched skills.

============================================================
5. TWO NESTED MASTER-TRUTH REPLICATES
============================================================

Use exactly two independent master-truth replicates:

    replicate r in {0,1}

For each replicate, generate one master library with:

    K_max = 30
    A = 50
    m_k = 10

Apply one fixed random permutation to the 30 skills before defining the nested
ladder.

For each K, use the first K skills of that permuted master library.

Thus, within one replicate:

    K=3 subset K=5 subset K=10 subset K=20 subset K=30

The nested object includes:
- CPA supports;
- role maps;
- latent utility matrices;
- induced role-labelled transitive closures.

Generate pi and P separately for each K from the same registered model prior,
using preregistered K-specific seeds. Do not obtain them by truncating and
renormalizing the K=30 matrix unless that is already the exact registered
generator.

The authoritative exact-library representation must include:

    role-labelled CPA support
    +
    transitive-closure relation bits

not closure relation counts alone.

============================================================
6. FIXED SEED SCHEME
============================================================

Use this deterministic seed scheme unless it collides with an existing formal
registration; if collision occurs, stop and report rather than silently
changing it.

For replicate r in {0,1}:

    master structural truth:
        6_500_001 + r

    master role-support library:
        6_500_051 + r

    master skill permutation:
        6_500_101 + r

For each K and replicate r:

    pi/P truth:
        6_510_000 + 100*K + r

    train corpus:
        6_520_000 + 100*K + r

    held-out corpus:
        6_530_000 + 100*K + r

    dispersed-start base:
        6_540_000 + 100*K + 10*r

    formal-chain base:
        6_550_000 + 100*K + 10*r

Use four start seeds:

    start_base + {1,2,3,4}

Use four chain seeds:

    chain_base + {1,2,3,4}

Use dispersed U-start scales:

    {0.5, 1.0, 2.0, 3.0}

Scale-pilot seeds for K:

    6_560_000 + 100*K + candidate_index

Write the full expanded seed table to the preregistration.

============================================================
7. MASTER-TRUTH ADMISSIBILITY
============================================================

The master truth is a draw from the registered prior conditional on the
following predeclared admissibility event.

For every skill:

1. the role map is injective;
2. the induced closure is a valid strict partial order;
3. relation count is at least 1;
4. relation count is strictly below m_k(m_k-1)/2;
5. all 30 role-labelled skill closures are pairwise distinct;
6. all 30 CPA supports are distinct;
7. no support contains fewer or more than 10 roles.

For each K-specific pi/P:

8. pi is valid;
9. P is non-negative, row-normalized and has zero diagonal;
10. the stationary occupancy vector exists and is unique;
11. each stationary skill probability lies in:
        [0.5/K, 1.5/K]
12. each initial pi component lies in:
        [0.5/K, 1.5/K]

These conditions create a controlled, approximately balanced recovery problem.
The paper must state that recovery scaling is evaluated under balanced
per-skill evidence.

No admissibility condition may inspect:

- model recovery;
- closure F1/Hamming;
- posterior likelihood;
- convergence;
- held-out NLL;
- path-marginal versus conditional performance;
- sampler acceptance;
- distance from an inferred state.

Record every attempted truth seed and every rejection reason.

Maximum master-truth attempts:

    100 per replicate

If no admissible truth is obtained, stop before formal launch.

============================================================
8. CORPUS-COVERAGE ADMISSIBILITY
============================================================

Generate formal train and held-out corpora before launching any sampler.

For every K, replicate and skill require:

Training:
- at least 30 true skill instances;
- between 240 and 720 CPA occurrences assigned to that skill;
- every one of its 10 roles appears at least 5 times.

Held-out:
- at least 8 true skill instances;
- at least 60 CPA occurrences assigned to that skill.

These criteria are functions only of the generated latent corpus and are fixed
before inference.

They do not use posterior recovery or likelihood.

Maximum corpus-generation attempts:

    100 per K/replicate

Record every attempt and rejection reason.

If a corpus fails after 100 attempts:
- do not lower the coverage threshold;
- mark the configuration generation-failed;
- stop before any formal sampler is launched.

Generate all 10 formal corpora first:

    5 K values x 2 replicates

Freeze:

- master-truth hashes;
- K-specific truth hashes;
- train hashes;
- held-out hashes;
- role-map hashes;
- start hashes;
- seed manifest.

Do not print, plot or return any truth value.
Only hashes and admissibility verdicts may be exposed before terminal
unsealing.

Because truths are nested, no truth may be unsealed until all formal K runs
for both replicates have terminal truth-free reports.

============================================================
9. PREREGISTRATION BEFORE PILOT OR FORMAL SAMPLING
============================================================

Create:

    results/recovery_scalability/k_ladder_v1/
        PREREG_RECOVERY_SCALE_K30.md
        PREREG_RECOVERY_SCALE_K30.json
        SEED_MANIFEST.json
        CORPUS_MANIFEST.json
        TRUTH_HASH_MANIFEST.json
        HARDWARE_MANIFEST.json
        SOFTWARE_MANIFEST.json

The preregistration must contain:

- scientific question;
- K ladder;
- fixed J;
- N_train/N_test rule;
- two nested replicates;
- all seeds;
- truth and corpus admissibility;
- model target;
- fixed nuisance parameters;
- proposal pilot;
- runtime schedule selection;
- sampling schedule;
- convergence summaries;
- degeneracy rules;
- recovery metrics;
- held-out NLL estimator;
- truth-unseal rule;
- deadline/censoring rule;
- output files;
- paper claim boundaries.

Commit before the scale pilot:

    "Preregister K=3--30 recovery-scalability ladder"

Do not launch any formal chain before this commit exists.

============================================================
10. TRUTH-FREE PROPOSAL-SCALE PILOT
============================================================

Run one pilot per K, not one pilot per replicate.

Use replicate-0 training observations, but do not expose H*, S* or z* to:

- scale selection;
- pilot diagnostics;
- pilot logs.

The sampler may use only the observed CPA traces and the ordinary model.

Candidate scale multipliers:

    {0.25, 0.5, 1, 2, 4, 8}

Run:

    600 pilot sweeps per candidate
    structural cadence = 1/10

Use only:

- U-row proposal acceptance;
- invalid-proposal rate;
- H-changing acceptance count;
- numerical validity;
- runtime.

Selection rule:

1. retain candidates with pooled U-row acceptance in [0.20,0.60];
2. choose the candidate closest to 0.40;
3. break an exact tie in favour of the smaller scale;
4. if no candidate is in range, pilot FAILS and that K is not launched.

Do not use H-changing acceptance as the selection objective; report it only as
a liveness diagnostic.

Discard all pilot states and draws.

Both formal replicates at the same K use the selected scale.

Commit selected scales before the first formal run:

    "Freeze truth-free proposal scales for K-recovery ladder"

============================================================
11. NEW-MACHINE RUNTIME AND MEMORY CALIBRATION
============================================================

Before formal launch, benchmark each K on pilot state:

    20 plain sweeps
    3 structural sweeps

Then run a short K=30 eight-worker contention test:

    100 sweeps per chain
    two replicates x four chains
    no result retention

Measure:

- plain sec/sweep;
- structural sec/sweep;
- amortized sec/sweep at cadence 1/10;
- peak RSS per worker;
- aggregate peak RSS;
- machine-speed probe;
- load average;
- CPU utilization;
- swapping.

Estimate the complete formal wall-clock for every K under the intended
concurrency.

Preferred schedule A:

    warm-up = 20,000 sweeps
    production = 30,000 sweeps
    total = 50,000 sweeps per chain

Uniform fallback schedule B:

    warm-up = 15,000 sweeps
    production = 25,000 sweeps
    total = 40,000 sweeps per chain

Choose schedule A if:

    projected pilots + all formal sampling <= 132 hours

Choose schedule B only if:

    schedule A > 132 hours
    AND schedule B <= 144 hours

The schedule choice must:
- be made before any formal chain starts;
- use timing and memory only;
- apply uniformly to every K and replicate;
- be committed before launch;
- never use recovery or truth.

If schedule B is projected above 144 hours:
    do not launch;
    report that the machine is insufficient for the registered one-week study.

Memory rule:

If 8 workers are predicted to exceed 75% of physical RAM:
- test one replicate at a time using four workers;
- recompute the deadline projection.

If sequential replicates cannot finish within the selected schedule:
    do not launch.

Do not lower N, K, J, chain count or replicate count to fit the machine.

============================================================
12. FORMAL SAMPLING PROTOCOL
============================================================

For every K and replicate:

    four dispersed chains
    optimized path-marginal backend
    structural cadence = 1/10
    thin = 5
    checkpoint every 2,000 sweeps
    selected proposal scale for that K
    fixed schedule A or B
    one terminal gate after production
    no adaptive stopping

Production draws only are used for:

- convergence diagnostics;
- posterior recovery;
- posterior predictive evaluation.

Warm-up draws are discarded completely.

Warm-up may retain integer liveness counters, including accepted H-changing
moves, but no warm-up draw enters a posterior summary.

Run configurations largest-first:

    K=30
    K=20
    K=10
    K=5
    K=3

Run the two replicates of one K simultaneously if the memory gate permits:

    2 replicates x 4 chains = 8 workers

Do not run different K values simultaneously.

============================================================
13. CONVERGENCE AND SAMPLING-ADEQUACY GATE
============================================================

Use production draws only.

For every non-degenerate registered summary require:

    rank-normalized split R-hat <= 1.01

For log target require:

    bulk ESS >= 1000
    tail ESS >= 500

For other non-degenerate summaries require:

    bulk ESS >= 400
    tail ESS >= 400

Register at least:

- log_target;
- total_segments;
- mean_segments_per_trace;
- mean_segment_length;
- sd_segment_length;
- total_relations;
- sorted per-skill relation counts;
- pi_entropy;
- pi_l2;
- every sorted_pi component;
- P Frobenius norm;
- trace(P^2);
- trace(P^3);
- sorted P-row entropies;
- sorted stationary probabilities;
- 32 deterministic truth-free boundary probes;
- 64 deterministic truth-free co-skill probes.

Select boundary and co-skill probes by a deterministic hash rule based on the
corpus hash. Freeze probe IDs before sampling.

============================================================
14. DISCRETE AND BINARY DEGENERACY RULES
============================================================

The primary structural object is the exact canonical role-labelled closure
library.

For any discrete structural summary:

A. constant and equal across all production chains and draws:
       degenerate cross-chain agreement;
       no R-hat/ESS requirement for that summary;

B. constant within chains but unequal across chains:
       automatic FAIL;

C. non-degenerate:
       ordinary R-hat and ESS requirements.

Branch A is allowed only if:
- all four chains started from structurally distinct libraries;
- every chain accepted at least one H-changing move during warm-up;
- the exact role-labelled canonical library is equal, not merely relation
  counts;
- all remaining registered diagnostics pass.

For a Bernoulli boundary/co-skill probe:

1. if constant and equal across all production chains:
       degenerate agreement;

2. if constant within chains but unequal:
       automatic FAIL;

3. if non-degenerate but the empirical tail-quantile indicator is constant,
   making tail ESS equal to 0/0:
       tail ESS is NOT APPLICABLE;
       require R-hat <= 1.01;
       require bulk/mean ESS >= 400;
       require posterior-probability MCSE <= 0.01;

4. otherwise:
       ordinary R-hat, bulk ESS and tail ESS gates.

This rule is prospective and must be implemented and tested before launch.

A formal convergence FAIL does not prohibit recovery evaluation after the
fixed schedule. Convergence and truth recovery must be reported separately.

============================================================
15. UNATTENDED ORCHESTRATION
============================================================

Launch the experiment fully detached.

On macOS:
    use caffeinate plus start_new_session=True

On Linux:
    use start_new_session=True and systemd-inhibit if available

Verify:
- orchestrator PPID/PGID;
- worker PIDs;
- workers actively use CPU;
- no shell-owned process group;
- sleep inhibition active;
- heartbeat updating;
- checkpoints atomic;
- exact resume tested.

Use one durable orchestration state:

    phase
    current K
    current replicate
    current chain sweeps
    selected schedule
    selected scales
    attempts
    last checkpoint hashes
    deadline remaining
    estimated completion time

Do not use broad pkill patterns.
Signal only recorded PIDs/process groups.

Preserve partial results after every checkpoint.

============================================================
16. TRUTH-FREE TERMINAL PHASE
============================================================

For each K/replicate, after all four chains complete:

1. hash final checkpoints and production draws;
2. compute the registered truth-free convergence gate;
3. independently reproduce the gate;
4. commit/store the terminal truth-free report;
5. record convergence PASS/FAIL and every failed summary;
6. do not open truth.

Because the master truth is nested, wait until all completed/censored
conditions have terminal truth-free reports before opening any truth.

If a run is censored by the hard deadline:
- write a truth-free censored report;
- preserve its draws;
- do not call it converged or failed at the full registered budget.

============================================================
17. RECOVERY AFTER GLOBAL UNSEALING
============================================================

Only after every condition has a committed truth-free terminal report:

    unseal both master truths and all K-specific corpus truths at once

For every posterior draw, use one structural matching from learned skills to
true skills based on role-labelled closure Hamming distance.

Apply the SAME permutation to:

    H
    z
    pi
    rows and columns of P

Do not choose separate favourable alignments for different quantities.

Primary structural metrics:

1. posterior probability of the exact canonical true library;
2. posterior fraction of true skills recovered exactly;
3. macro transitive-closure F1;
4. mean normalized closure Hamming per skill;
5. incomparable-pair F1;
6. relation Brier score and calibration;
7. posterior modal-library diversity;
8. time to first exact-library visit;
9. persistence after first exact-library visit.

Primary segmental metrics:

1. boundary Brier;
2. boundary AUROC;
3. co-skill Brier;
4. ARI;
5. posterior block-count error;
6. segment-length error.

Global dynamics:

1. pi total-variation error;
2. P Frobenius error;
3. P row-wise total variation.

Prediction:

1. held-out NLL per CPA occurrence;
2. difference from the truth plug-in reference;
3. MCSE.

For held-out NLL use exactly:

    500 posterior draws per replicate
    125 equally spaced production draws per chain

Use the same log-mean-exp predictive estimator as the final confirmatory
experiment.

Do not increase posterior draws after observing noisy results.

============================================================
18. ORACLE-H SEGMENTATION REFERENCE
============================================================

After truth unsealing, compute one cheap secondary reference for each K and
replicate:

    exact p(S,z | x,H*,pi*,P*,fixed nuisance parameters)

using the exact semi-Markov dynamic program.

This is not another MCMC experiment.

Use it to report the best identifiable boundary/co-skill recovery available
when the reusable library is known.

Its purpose is to distinguish:

- structure-inference failure;
- irreducible segmentation ambiguity.

Label it:

    ORACLE-H IDENTIFIABILITY REFERENCE

Do not use it to tune the formal runs.

============================================================
19. RECOVERY VERSUS COVERAGE
============================================================

For each true skill record:

- number of train invocations;
- number of train occurrences;
- per-role occurrence counts;
- closure relation count;
- exact recovery indicator;
- closure F1;
- closure Hamming.

Plot:

    per-skill closure F1
    versus
    observed invocation count

and:

    exact skill recovery
    versus
    occurrence coverage

This analysis is performed only after truth unsealing.

It must not be used to exclude difficult skills or corpora.

============================================================
20. AGGREGATION ACROSS REPLICATES
============================================================

There are exactly two independent master-truth replicates.

For each K:

- plot both replicate points;
- report their mean;
- report their range;
- do not present a Gaussian standard-error bar based on only two truths;
- do not claim broad generator-level concentration.

For trace-level predictive/segmentation metrics, bootstrap traces within each
replicate, but keep the two truth replicates visually and numerically separate.

The primary trend is descriptive recovery degradation across:

    K = 3,5,10,20,30

============================================================
21. PAPER FIGURES
============================================================

Create PNG and vector PDF versions of:

1. fig_recovery_vs_K
   Panels:
   - exact canonical-library posterior mass;
   - exact recovered skills / K;
   - macro closure F1;
   - normalized closure Hamming.

2. fig_segmental_recovery_vs_K
   Panels:
   - boundary Brier;
   - co-skill Brier;
   - ARI;
   - block-count error.

3. fig_prediction_vs_K
   - held-out NLL per occurrence;
   - gap to truth plug-in reference.

4. fig_sampling_adequacy_vs_K
   - formal PASS/FAIL;
   - max R-hat;
   - minimum ESS;
   - accepted H-changing moves.

5. fig_anytime_recovery_K20_K30
   - exact-library occupancy versus sweeps;
   - macro closure F1 versus sweeps;
   - wall-clock and normalized compute axes.

6. fig_coverage_vs_recovery
   - per-skill recovery versus actual evidence.

Use:
- K on the horizontal axis;
- two replicate markers at every K;
- a thin line through replicate means;
- no misleading confidence interval from two replicates;
- readable ICLR typography;
- no conditional-ablation curves.

============================================================
22. PAPER TABLES AND TEXT
============================================================

Create:

    paper/tables/tab_recovery_scalability_K.tex
    paper/figures/fig_recovery_vs_K_include.tex
    paper/figures/fig_segmental_recovery_vs_K_include.tex

Do not edit the paper main TeX automatically.

Create:

    RECOVERY_SCALABILITY_SECTION_DRAFT.tex

Safe possible conclusions must depend on the observed results.

If recovery remains strong through K=30:

    "Under approximately matched per-skill evidence, reusable-library and
    segmental recovery remain accurate as K grows from 3 to 30."

If whole-library exact recovery falls but per-skill recovery remains high:

    "Exact joint-library recovery becomes increasingly stringent with K,
    while most individual reusable structures remain accurately recovered."

If recovery deteriorates materially:

    "The optimized exact sampler remains computationally executable, but
    statistical recovery degrades beyond [observed K] under the tested
    per-skill evidence budget."

Never write:
- recovery scales to arbitrary K;
- K=30 formally converged if its gate failed;
- two replicates establish population-level certainty;
- all parameters improve uniformly;
- exact-library failure implies every skill is wrong.

============================================================
23. REQUIRED ARTIFACT TREE
============================================================

Create:

results/recovery_scalability/k_ladder_v1/
    prereg/
    manifests/
    pilots/
    runs/
        K03/
            replicate_0/
            replicate_1/
        K05/
        K10/
        K20/
        K30/
    terminal_truth_free/
    recovery/
    heldout/
    oracle_H/
    figures/
    tables/
    logs/
    state.json
    events.jsonl
    heartbeat.json
    progress.md
    FINAL_REPORT.md
    FINAL_METRICS.json
    FINAL_METRICS.csv
    CENSORED_RUNS.md
    LIMITATIONS.md
    SAFE_PAPER_CLAIMS.md
    TODO_FOR_HOLLY.md

Store raw chain arrays outside ordinary git if they are large.

Commit:
- code;
- preregistration;
- manifests;
- hashes;
- compact diagnostic/recovery artifacts;
- reports;
- figures;
- LaTeX snippets.

Do not commit multi-gigabyte raw checkpoints to normal git unless the repository
already uses an appropriate large-file mechanism.

Every raw file must nevertheless be SHA-256 hashed and listed in a freeze
manifest.

============================================================
24. COMMIT POLICY
============================================================

Use separate commits:

1. preregistration;
2. sealed master truths/corpora manifests;
3. frozen scale pilots;
4. schedule/runtime decision;
5. terminal truth-free results for each K;
6. global truth unseal and recovery;
7. held-out NLL;
8. paper figures/tables.

Suggested messages:

    "Preregister K=3--30 recovery-scalability ladder"
    "Freeze nested recovery-scaling corpora and truth hashes"
    "Freeze truth-free proposal scales for recovery ladder"
    "Launch one-week K-recovery scalability study"
    "Add truth-free terminal K-recovery diagnostics"
    "Add K-ladder structural and segmental recovery"
    "Add K-ladder held-out predictive evaluation"
    "Add paper-ready recovery-scalability figures"

Do not merge the branch automatically.

============================================================
25. FINAL REPORT AFTER ONE WEEK
============================================================

Report:

1. exact elapsed time;
2. machine specification;
3. base and branch commits;
4. selected schedule A or B;
5. selected proposal scale for every K;
6. completed and censored configurations;
7. convergence status for every K/replicate;
8. exact-library posterior mass for every K/replicate;
9. exact recovered skills / K;
10. macro closure F1 and Hamming;
11. boundary and co-skill recovery;
12. pi and P recovery;
13. held-out NLL;
14. oracle-H reference;
15. time to first exact library;
16. structural persistence;
17. runtime and peak RSS;
18. recovery-versus-coverage result;
19. paths and hashes of all figures/tables;
20. safe paper paragraph;
21. strongest limitation;
22. confirmation no conditional arm was run;
23. confirmation no truth was opened before all truth-free terminal reports;
24. confirmation no historical sealed source or verdict was modified.

============================================================
26. LAUNCH
============================================================

Proceed autonomously.

First:
- complete preflight;
- write and commit preregistration;
- generate and seal all master truths/corpora;
- run pilots;
- perform runtime/memory calibration;
- select and commit schedule A or B.

If every hard gate passes:
    launch K=30 first, fully detached.

Verify:
- orchestrator PID/PPID/PGID;
- worker PIDs;
- sleep inhibition;
- active CPU use;
- checkpoint/heartbeat paths;
- estimated completion time.

Then continue automatically:

    K=30 -> K=20 -> K=10 -> K=5 -> K=3
    -> terminal truth-free reports
    -> global unseal
    -> recovery/NLL/oracle-H
    -> figures/tables/reports
    -> final commits

Do not stop to ask routine questions.
