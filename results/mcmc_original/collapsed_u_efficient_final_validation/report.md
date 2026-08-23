# Sequential collapsed-U validation

**COLLAPSED-U KERNEL VALIDATED**

Burn-in 100,000 (verified), seeds [8158001, 8158002, 8158003, 8158004], cadence 10, thin 10.

| checkpoint | verdict | components |
|---|---|---|
| 150k | INCONCLUSIVE | A_frozen_gates, B_primary_energy, C_sensitivity_agrees, D_rhat, E_ess |
| 200k | FAIL | A_frozen_gates, D_rhat, E_ess |
| 250k | FAIL | A_frozen_gates, D_rhat, E_ess |
| 300k | FAIL | A_frozen_gates, D_rhat, E_ess |
| 400k | PASS | all pass |
| 500k | PASS | all pass |

Total wall: 2.52 h.
