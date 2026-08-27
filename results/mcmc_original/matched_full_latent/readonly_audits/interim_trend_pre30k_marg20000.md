# Exploratory FULL-LATENT interim trend diagnostic

**Nonformal / truth-free / read-only.** The registered 30k gate remains authoritative; this artifact neither writes a gate nor changes the running experiment.

Snapshot UTC: `2026-08-22T13:25:54.411644+00:00`.

## Current registered-summary diagnostics

| Arm | Sweep | Retained/chain | log-target R-hat | total-relations R-hat | max invariant R-hat | worst invariant |
|---|---:|---:|---:|---:|---:|---|
| FULL-COND | 22000 | 2400 | 2.753 | 3.228 | inf | `sorted_relation_counts[1]` |

### FULL-COND: worst invariant

`sorted_relation_counts[1]`; type: **constant-but-different across chains**.

| Chain | posterior mean | recent 2k-sweep mean | recent SD | unique values |
|---:|---:|---:|---:|---:|
| 0 | 4 | 4 | 0 | 1 |
| 1 | 5 | 5 | 0 | 1 |
| 2 | 5 | 5 | 0 | 1 |
| 3 | 5 | 5 | 0 | 1 |
| FULL-MARG | 20000 | 2000 | 1.131 | 1.4 | 1.4092847473118995 | `sorted_relation_counts[1]` |

### FULL-MARG: worst invariant

`sorted_relation_counts[1]`; type: **slowly drifting toward agreement**.

| Chain | posterior mean | recent 2k-sweep mean | recent SD | unique values |
|---:|---:|---:|---:|---:|
| 0 | 6 | 6 | 0 | 1 |
| 1 | 6 | 6 | 0 | 1 |
| 2 | 6 | 6 | 0 | 1 |
| 3 | 5.7515 | 6 | 0 | 2 |

## Previous vs current

| Arm | metric | previous | current | factor |
|---|---|---:|---:|---:|
| FULL-COND | log_target_rhat | 2.6047347918485837 | 2.7528427534873647 | 1.0568610524581155 |
| FULL-COND | total_relations_rhat | 3.1006876133146517 | 3.228465444033358 | 1.0412095143574012 |
| FULL-COND | max_invariant_rhat | inf | inf | None |
| FULL-COND | worst invariant | `sorted_relation_counts[1]` | `sorted_relation_counts[1]` | changed=False |
| FULL-MARG | log_target_rhat | 1.3820092889755906 | 1.1314972966036674 | 0.818733495953841 |
| FULL-MARG | total_relations_rhat | 3.073502216840513 | 1.4000428979871118 | 0.45552038007843787 |
| FULL-MARG | max_invariant_rhat | 4.58257569495584 | 1.4092847473118995 | 0.30753114430016637 |
| FULL-MARG | worst invariant | `sorted_relation_counts[2]` | `sorted_relation_counts[1]` | changed=True |

## Scope

Only registered, permutation-invariant checkpoint summaries were read.  No raw skill-indexed trace is interpreted, and this diagnostic opened no synthetic truth or held-out recovery.  No formal source, running process, checkpoint, threshold, seed, scale, cadence, or datum was modified.

This audit attests only to its own truth-free scope.  It does not make a global experiment truth-seal claim; a separately recorded mid-run unseal event exists and was not opened by this audit.
