# Collapsed-U fast audit (C0)

Chains: 7B2 FFBS chain 0 (sweep 43,000), 6E2 Local chain 0 (sweep 135,000); 100 cross-H proposals from the exact production U kernel (scale 0.5), seed 8151500. Corpus hash matches the frozen audit.

```text
                              Conditional U       Collapsed U
--------------------------------------------------------------
median cross-H delta log L          -265.6               -58.6
max cross-H log alpha                -27.0                25.0
mean cross-H acceptance           2.05e-14            4.96e-02
expected escapes / 50k            9.06e-09            2.07e+04
```

Barrier reduction (coll - cond): median +214.8 nats, q2.5 +20.4, q97.5 +1261.0.

**COLLAPSED-U MECHANISM VIABLE — EXPAND AUDIT**

Checks: current-state parity (fast vs adapter tables + log Z) <= 1e-10; enumeration parity on the shortest trace at U and U' <= 1e-10; incremental vs full rebuild <= 1e-10; q0 reset bit-identical; same-H conditional dLL = 0; Hastings term numerically 0. Details in comparison.json.
