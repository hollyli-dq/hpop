# Condition C — registered checkpoint trajectory

Primary analysis uses draws through sweep 75,000 only (13,000 retained draws per chain).

| rung | arm | registered verdict | max anchored rhat | log target bulk ess | total relations bulk ess | uncertain relation indicators | distinct unordered libraries | library split | anchored assignment split | assignment gap nats | accepted cross h per chain |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 30000 | cond | FAIL | inf | 4.9 | 4.0 | 30 | 4 | 1-1-1-1 | 1-1-1-1 | 467.86 | 24/14/22/20 |
| 30000 | marg | FAIL | inf | 7.1 | 16000.0 | 18 | 1 | 4 | 3-1 | 124.39 | 24/22/23/35 |
| 50000 | cond | FAIL | inf | 5.2 | 4.2 | 30 | 4 | 1-1-1-1 | 1-1-1-1 | 433.29 | 24/17/22/20 |
| 50000 | marg | FAIL | inf | 7.1 | 32000.0 | 18 | 1 | 4 | 3-1 | 124.44 | 24/22/23/35 |
| 75000 | cond | FAIL | inf | 5.4 | 4.2 | 30 | 3 | 2-1-1 | 1-1-1-1 | 316.89 | 24/18/22/20 |
| 75000 | marg | FAIL | inf | 7.1 | 52000.0 | 18 | 1 | 4 | 3-1 | 124.3 | 24/22/23/35 |

Terminal-rung path-marginal disagreement (from the untruncatable accumulators, supplementary):

- cond: max pairwise boundary-marginal difference 0.933, occurrence-marginal difference 1.0, segment-total spread 26
- marg: max pairwise boundary-marginal difference 0.58, occurrence-marginal difference 1.0, segment-total spread 29
