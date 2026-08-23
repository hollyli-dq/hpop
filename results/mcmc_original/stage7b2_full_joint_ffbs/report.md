# Step 7B2 — FFBS full-joint chains on the frozen Stage 6E2 corpus

**The comparison is deferred.** These are the FFBS chains only. The Stage 6E2 baseline was at 75,000 sweeps of its registered ladder when these chains launched, and the scientific comparison must use its frozen final state, not that intermediate rung. Nothing in this directory reads a baseline chain artifact.

## Run

* 4 chains x 50,000 sweeps, burn-in 15,000, thin 5
* seeds [7063201, 7063202, 7063203, 7063204], starts reconstructed with the baseline's own constructor
* corpus `02be246edf9bd4f4...`, 100 training traces
* proposal scales the baseline's, untuned: {'U': 0.5, 'rho': 0.5, 'beta': 0.40872, 'omega': 1.45704, 'lambda_rep': 0.6648, 'lambda_back': 1.51552}
* wall 19.10 h, 1.370 s per sweep per chain

## Structural movement (the Step 7B2 question)

| chain | H changes | distinct H | sweeps to first H change | relation-count within-chain SD | major H transitions |
|---|---|---|---|---|---|
| 0 | 0 | 1 | None | 0.0000 | 0 |
| 1 | 0 | 1 | None | 0.0000 | 0 |
| 2 | 0 | 1 | None | 0.0000 | 0 |
| 3 | 1 | 2 | 1814 | 0.0000 | 1 |

Chains whose structure is frozen by Stage 6E2's criterion A (within-chain relation-count sd < 0.01): 4 of 4. Largest between-chain gap in mean relation count: 6.000.

## Segmentation movement

| chain | boundary Hamming / sweep | label changes / sweep | co-clustering movement / sweep |
|---|---|---|---|
| 0 | 546.70 | 1343.84 | 0.02943 |
| 1 | 518.52 | 901.26 | 0.02692 |
| 2 | 518.50 | 1030.89 | 0.02683 |
| 3 | 474.80 | 755.90 | 0.02519 |

## Permutation-invariant convergence

Worst R-hat over the invariant summaries: 2.9399716130923884e+16 against a 1.01 threshold (FAIL). No truth alignment is used anywhere in this path.

## Not interpreted here

* recovery and held-out prediction: they require the convergence verdict and the frozen baseline, so they are not computed in this run;
* any Local-vs-FFBS claim: the baseline is still advancing its ladder.

Source commit `2ea872421ff7b790948e5a89105866ff25738b86`.
