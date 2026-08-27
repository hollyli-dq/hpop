# Collapsed-U start[0] focused probe

**START-0 BASIN-SPECIFIC KERNEL INTERACTION SUPPORTED**

4 chains, identical registered start[0], seeds [8155001, 8155002, 8155003, 8155004], 600,000 sweeps each, collapsed move every 10. Windows and verdict rules frozen before launch (registration.json). Run 1 and rep2 remain recorded failures.

## Historical frozen gate (unchanged, NOT the decision statistic)
observed 0.003718 vs frozen envelope 0.004522 -> PASS

## Per-chain energy z by window
| chain | early | middle | late |
|---|---|---|---|
| 0 | +1.67 | +2.64 | +1.47 |
| 1 | +3.40 | +4.08 | +4.26 |
| 2 | +5.37 | +7.69 | +2.03 |
| 3 | +0.85 | +3.26 | +0.80 |

## Chain-balanced energy (prospective)
| window | observed | envelope | z | inside |
|---|---|---|---|---|
| early | 0.004067 | 0.004522 | +0.95 | True |
| middle | 0.004943 | 0.004522 | +2.51 | False |
| late | 0.003509 | 0.004522 | -0.04 | True |

## beta / lambda_rep drift (offset in reference SD; SE-standardised in scalar_drift.json)
| chain | beta early | beta late | l_rep early | l_rep late |
|---|---|---|---|---|
| 0 | -0.018 | -0.098 | -0.041 | -0.025 |
| 1 | -0.078 | +0.113 | -0.082 | +0.148 |
| 2 | -0.160 | +0.013 | -0.161 | +0.042 |
| 3 | -0.039 | -0.049 | +0.013 | -0.013 |

## Verdict conditions
```json
{
  "A": {
    "late_balanced_inside_envelope": true,
    "at_most_one_late_z_above_2": false,
    "median_late_z_leq_1p5": false,
    "beta_3of4_late_offset_se_below_2": true,
    "lambda_rep_3of4_late_offset_se_below_2": true,
    "beta_median_abs_contracts": false,
    "lambda_rep_median_abs_contracts": true,
    "controls_clean": true,
    "late_ess_finite_and_moving": true
  },
  "B": {
    "three_late_z_above_2_with_median_above_2": false,
    "late_balanced_outside_envelope": false,
    "beta_3of4_same_direction_above_2": false,
    "lambda_rep_3of4_same_direction_above_2": false,
    "no_shrink_early_to_late": true,
    "joint_beta_lambda_rep_displacement_with_clean_controls": false
  }
}
```

Recommended next (NOT launched): D0-D4 scalar-release decomposition (collapsed U + FFBS with all scalars fixed; release beta only; lambda_rep only; both; then the rest) to localise the interaction (NOT launched here)
