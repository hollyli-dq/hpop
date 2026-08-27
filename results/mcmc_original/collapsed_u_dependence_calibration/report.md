# Dependence-aware calibration of the per-chain energy statistic

**ESTIMATOR ARTIFACT / SERIAL DEPENDENCE SUPPORTED**

mu0 (7B1 late mean) = 0.010189; block lengths [2, 5, 10] (L from max psi IACT = 4.9); 150 circular moving-block replicates; threshold z > 2.33.

| start[0] chain | T_late | z_dep (l=2) | z_dep (l=5) | z_dep (l=10) |
|---|---|---|---|---|
| 0 | 0.008600 | -0.93 | -0.78 | -0.65 |
| 1 | 0.012796 | +0.87 | +0.74 | +0.61 |
| 2 | 0.009799 | -0.20 | -0.17 | -0.13 |
| 3 | 0.007831 | -1.45 | -1.13 | -0.96 |

Chains over +2.33 by length: {'2': 0, '5': 0, '10': 0}
Control A (7B1 leave-one-out): within +-2.33 {'2': 4, '5': 4, '10': 4} -> PASS
Control B (finite-state exact): within +-2.33 {'2': 4, '5': 4, '10': 4} -> PASS

Historical verdicts unchanged; kernel unchanged; no new production chains; matched-synthetic paused.
