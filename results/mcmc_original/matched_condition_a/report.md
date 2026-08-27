# Condition A — path identifiability under oracle structures

Source commit: `8ca828153e8e263bf4ea4823e45a53fa454037ad` &middot; corpus `dd280a4a09896154…` &middot; truth `fc41538fd44d170d…`

## Verdict: **PATH STRONGLY IDENTIFIABLE**

Exact per-trace semi-Markov posterior over (S, z); everything else fixed to generating truth. No MCMC.

| metric (all traces) | posterior | prior |
|---|---|---|
| boundary Brier | 0.0516 | 0.1083 |
| boundary NLL | 0.1662 | 0.3629 |
| boundary AUROC | 0.9561 | 0.5787 |
| boundary AUPRC | 0.7929 | 0.1501 |
| boundary ECE | 0.0084 | 0.0003 |
| occurrence mean p(c*) | 0.8543 | 0.3395 |
| occurrence modal accuracy | 0.8984 | 0.3838 |

- boundary Brier reduction vs prior: 52.4%; NLL reduction 54.2%; occurrence NLL reduction 77.1%
- mean per-trace ARI 0.747, NMI 0.766
- MAP segment-count accuracy 0.703; mean |E[L] - L*| 0.439
- MAP labelled path exactly equals truth on 20.7% of traces; median true-path posterior 0.0479; median true-segmentation posterior 0.0479
- exact posterior path entropy mean 3.48 nats vs prior 14.02 (mean reduction 10.54 nats)
- FFBS: 5000 iid draws/trace, 15571 draws/s, no burn-in, no thinning, no R-hat; 0 illegal draws

Note on interpretation: with a zero-diagonal P the occurrence-label vector determines the labelled path bijectively, so p(c*-vector) = p(S*, z*). Low exact-path probability on long traces coexists with accurate marginals when near-equivalent paths share mass; the verdict rule was frozen in preregistration.json before inference.

STOPPED as registered: no Condition B/C/D, no U/rho/scalar inference.
