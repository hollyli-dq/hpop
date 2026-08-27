# Condition B — structure identifiability under oracle paths

Parent commit `b199374baaf3f795ce5ee6dca16b7478bd07a3b9` &middot; corpus `dd280a4a09896154…` &middot; rho fixed at 0.5 (registered correction; rho* = null)

## Classification: **STRUCTURE STRONGLY IDENTIFIABLE UNDER ORACLE PATHS**

- stopped at 50000 sweeps (two consecutive checkpoint passes)
- max R-hat 1.0009, log-posterior bulk ESS 3675
- selected sigma_u 0.5 (acceptance 0.314)

| skill | closure F1 | incomparable F1 | reduction F1 | p(H*) | modal = H* |
|---|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 | True |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | True |
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | True |

- min true-relation marginal 1.000; max false 0.000
- held-out oracle-path NLL/occ: posterior mean 1.0772, predictive 1.0772, truth 1.0772, total-order 4.0398, antichain 1.7528

Raw entrywise U error is not evaluated: the latent product-order coordinates are exchangeable and the likelihood reads U only through h(U). Level C (rho) is not evaluated by registered correction.

STOPPED as registered: no Condition C/D, no FFBS, no collapsed-U, no scalar inference.
