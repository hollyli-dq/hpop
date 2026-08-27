# Test baseline on a clean clone

What the shipped suite is expected to do here, so an operator running the prompt's
Section 2 preflight has a definite thing to compare against rather than a guess.

## Expected

    pytest tests -q     ->     1721 passed, 63 skipped, 0 failed     (~16 min, 1 thread)

    python verify_environment.py     ->     RESULT: READY

Anything else is a real problem. In particular **any failure is a real failure** — the
tests that cannot run here are skipped by name, not left to fail.

1784 tests are collected: 1595 from the source repository's suite, of which 63 skip, plus
189 in `tests/k_ladder/` for the code this package adds, all of which run.

| | collected | passed | skipped |
|---|---|---|---|
| source-repository suite | 1595 | 1532 | 63 |
| `tests/k_ladder/` | 189 | 189 | 0 |
| **total** | **1784** | **1721** | **63** |

## Verified against the source repository

The same source-repository suite there, which has the historical artifacts:

    1594 passed, 0 failed

The 63 skips here are 21 that skip in the source repository too, plus 42 listed in
`tests/HISTORICAL_AUDIT_SKIPS.txt`.

## Audit of all 63 skips (read before production, not assumed)

Every skip was enumerated with `pytest tests -rs` and read. They fall into two groups and
nothing else:

| n | module | reason |
|---|---|---|
| 24 | `test_condition_c_terminal.py` | historical artifacts / source-repo git history |
| 9 | `test_stage6b1_diagnostics.py` | historical artifacts |
| 8 | `test_stage7b_mixed_reference.py` | the frozen Stage 6E1B reference is absent here |
| 4 | `test_stage6b1_recurrent_target.py` | historical artifacts |
| 3 | `test_stage6e_pipeline.py` | the Stage 6E2 pilot was never run here |
| 2 | `test_matched_condition_c_prime.py` | historical artifacts |
| 2 | `test_stage6e_exact_and_reference.py` | Stage 6E1B reference not built |
| 2 | `test_stage7a_exact_posterior.py` | historical artifacts |
| 2 | `test_stage7b_mixed_reference.py` | Step 7B1 never run here |
| 1 | `test_collapsed_u_start0_probe.py` | historical artifacts |
| 1 | `test_stage6b_joint_diagnostics.py` | `stage6b2_joint3_smoke` never run |
| 1 | `test_stage6b_joint_diagnostics.py` | `stage6b3_joint4_smoke` never run |
| 1 | `test_stage6e_exact_and_reference.py` | Stage 6E1A never run |
| 1 | `test_stage6e_exact_and_reference.py` | Stage 6E1B chains never run |
| 1 | `test_stage6e_pipeline.py` | Stage 6E2 analysis never run |
| 1 | `test_stage6e_pipeline.py` | Stage 6E2 corpus never frozen |
| **63** | | **42 historical-record audits + 21 upstream-stage dependencies** |

**None of the 63 touches the K-ladder.** All 63 live in `tests/mcmc_original/`, and every
one of the ten modules involved contains zero references to `hpop.mcmc_cpa` — so the CPA
vocabulary layer, the nested library, the shared-Gamma coupling, the three-arm runner, the
skill-local table refresh and the common-random-number machinery are covered entirely by
tests that **do** run here.

**What the skips do cost.** Ten of them (`test_stage7b_mixed_reference.py`, and
`test_stage6e_exact_and_reference.py`) are reference-comparison tests: they check the
engine against a frozen reference that this code-only package does not ship. That class of
validation therefore does not run here. It is not left uncovered — `verify_environment.py`
runs its own optimized-versus-reference parity gate on every clean clone, which is the same
kind of check on the same engine (observed 2.842e-14 against a 1e-10 tolerance) — but the
substitution should be known rather than discovered later. The remaining 53 audit the
*record* of past experiments, which this package deliberately does not carry.

## What the 42 are, and why they are skipped

They audit the *record* of past experiments, not the model. They read files under
`results/mcmc_original/**`, or ask git about commits that are not in this repository's
history. Thirty-nine fail with `FileNotFoundError`; three are assertions about the source
repository's git history, such as `terminal commit must be readable`.

They live in six modules:

| module | skipped | also present and passing |
|---|---|---|
| `test_condition_c_terminal.py` | 24 | 3 |
| `test_stage6b1_diagnostics.py` | 9 | — |
| `test_stage6b1_recurrent_target.py` | 4 | — |
| `test_stage7a_exact_posterior.py` | 2 | — |
| `test_matched_condition_c_prime.py` | 2 | 5 |
| `test_collapsed_u_start0_probe.py` | 1 | 2 |

Those six modules hold 97 tests. Only 42 depend on the missing artifacts; the other 55 run
and pass here, which is why the skip list is **per test rather than per module** — skipping
the modules wholesale would have discarded 55 working tests and weakened the preflight.

## Two files were removed rather than skipped

`test_collapsed_u_expanded_audit.py` and `test_collapsed_u_fast_audit.py` raise
`FileNotFoundError` during *collection*, before any skip marker can apply, so they could
not be handled by the skip list. They are audits of stored collapsed-U chains and test
nothing about the engine.

## Regenerating the skip list

    pytest tests -q | grep '^FAILED' | sed 's/^FAILED //; s/ - .*//' | sort \
        > tests/HISTORICAL_AUDIT_SKIPS.txt

`tests/conftest.py` reports any entry that matches no test at the end of the run, so a
stale list is visible rather than silent.

## What is NOT skipped

Every test of the model, the sealed reference engine (`hpop.mcmc_original`), the optimized
backend (`hpop.mcmc_optimized`), the generator, the block scorer, the semi-Markov
forward/backward recursions and the collapsed-U kernel runs and must pass. The
optimized/reference parity gate is additionally re-checked by `verify_environment.py` on
every invocation.
