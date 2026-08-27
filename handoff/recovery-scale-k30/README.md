# hpop — K-recovery scalability package

Code for a one-week formal synthetic **recovery** study: how does recovery of a reusable
skill library degrade as `K` grows from 3 to 30, with trace length and per-skill evidence
held fixed?

This package ships **code without git history**, because the source repository cannot be
pushed to GitHub (a 176.5 MB blob in its history exceeds the 100 MB hard limit, and
rewriting history would change the commit hashes the provenance record depends on).

## Read this first

    HANDOFF_NOTES.md

There are four blockers between this package and a run, and one design issue that changes
how the result must be interpreted. None of them is discoverable from the prompt alone.

## Quick start

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python verify_environment.py

`verify_environment.py` must print `RESULT: READY` before anything else happens. It checks
source integrity, the environment, and the optimized/reference parity gate. On the source
machine it reports:

    source integrity: 199 files, 0 modified, 0 missing
    sealed engines byte-identical to 564995efd056: True
    max |alpha_opt - alpha_ref| = 2.842e-14   (tolerance 1e-10)
    max |logZ_opt  - logZ_ref | = 1.421e-14

Then give `prompt/PROMPT_RECOVERY_SCALE_K30.md` to the operator or agent — **after**
resolving Blocker 1, which otherwise stops the run at Section 1.

## What is here

    prompt/                 the execution prompt, verbatim
    HANDOFF_NOTES.md        blockers, design issue, smaller caveats
    verify_environment.py   integrity + environment + parity preflight
    SOURCE_INTEGRITY.json   SHA-256 of every shipped file; origin commits
    requirements.txt        pinned versions the parity gate was verified under
    src/hpop/               the model, the sealed reference engine, the optimized backend
    scripts/                corpus generation, chain runner, terminal gate, recovery,
                            held-out NLL, post-hoc sensitivity
    scripts/harness_reference/
                            resumable orchestration proven over a 154-configuration
                            unattended run: atomic state, memory preflight, partial-result
                            flushing, fixed-work machine-speed probe
    tests/                  the tests that do not depend on results artifacts

## Provenance

    optimized backend      564995efd056d7d33984f0ca1532386e6140ea0c
    sealed reference       hpop.mcmc_original, byte-identical to that commit

`src/hpop/mcmc_original/` is the numerical oracle and must not be edited. The optimized
backend in `src/hpop/mcmc_optimized/` is a separate package precisely so the sealed sources
stay untouched; every routine in it is verified against the reference, and where they
disagree the reference is right by definition.

Source commits for the shipped scripts are recorded in `SOURCE_INTEGRITY.json`.

## What this package deliberately does not contain

No results, no corpora, no chains, no truth files. The study generates its own sealed
truths and corpora as its first step; shipping any would risk truth leakage into a study
whose entire design is truth-free until terminal unsealing.
