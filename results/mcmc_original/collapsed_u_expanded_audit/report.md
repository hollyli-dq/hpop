# Collapsed-U expanded audit (C1)

8 frozen chains, 300 cross-H proposals each from the exact production U kernel (scale 0.5); seeds registered in config.json; r_cross measured from each chain's raw stream.

| Experiment | Chain | H hash | r_cross | med dLL_cond | med dLL_coll | med reduction | P(a>1%) | P(a>10%) | P(loga>=0) | mean a | E/50k | tail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7b2_ffbs | 0 | 78b96e48 | 0.578 | -427.5 | -61.0 | +357.0 | 0.020 | 0.013 | 0.013 | 1.35e-02 | 5.86e+03 | YES |
| 7b2_ffbs | 1 | 9a660e64 | 0.415 | -211.6 | -76.2 | +135.4 | 0.093 | 0.093 | 0.093 | 9.33e-02 | 2.90e+04 | YES |
| 7b2_ffbs | 2 | 988422dc | 0.612 | -235.7 | -88.7 | +131.4 | 0.057 | 0.040 | 0.040 | 4.03e-02 | 1.85e+04 | YES |
| 7b2_ffbs | 3 | f08954e5 | 0.629 | -195.1 | -92.6 | +101.2 | 0.040 | 0.030 | 0.020 | 2.20e-02 | 1.04e+04 | YES |
| 6e2_local | 0 | cf423445 | 0.510 | -113.8 | -33.4 | +99.3 | 0.177 | 0.163 | 0.120 | 1.42e-01 | 5.45e+04 | YES |
| 6e2_local | 1 | 976d3e1e | 0.569 | -260.7 | -104.3 | +128.5 | 0.077 | 0.067 | 0.067 | 6.68e-02 | 2.85e+04 | YES |
| 6e2_local | 2 | b62b567a | 0.504 | -273.0 | -93.1 | +210.0 | 0.100 | 0.090 | 0.087 | 9.02e-02 | 3.41e+04 | YES |
| 6e2_local | 3 | fdb0cc9b | 0.490 | -145.7 | -34.7 | +93.9 | 0.170 | 0.113 | 0.107 | 1.10e-01 | 4.04e+04 | YES |

## Pooled (after the per-chain table)
```json
{
  "all": {
    "n_cross": 2400,
    "r_cross_mean": 0.5385012449714877,
    "median_d_ll_conditional": -225.75222015611564,
    "median_d_ll_collapsed": -69.30124255553966,
    "median_barrier_reduction": 131.74510192860413,
    "p_alpha_gt_1pct": 0.09166666666666666,
    "p_alpha_gt_10pct": 0.07625,
    "p_log_alpha_ge_0": 0.06833333333333333,
    "mean_alpha_collapsed": 0.07228743270661536,
    "expected_escapes_50k_mean": 27657.084885288532,
    "expected_escapes_50k_min": 5861.223192948735
  },
  "ffbs": {
    "n_cross": 1200,
    "r_cross_mean": 0.5585370392465677,
    "median_d_ll_conditional": -235.65456312014362,
    "median_d_ll_collapsed": -75.1116722222941,
    "median_barrier_reduction": 157.32691085411426,
    "p_alpha_gt_1pct": 0.0525,
    "p_alpha_gt_10pct": 0.04416666666666667,
    "p_log_alpha_ge_0": 0.041666666666666664,
    "mean_alpha_collapsed": 0.042287544513095715,
    "expected_escapes_50k_mean": 15946.861498419907,
    "expected_escapes_50k_min": 5861.223192948735
  },
  "local": {
    "n_cross": 1200,
    "r_cross_mean": 0.5184654506964076,
    "median_d_ll_conditional": -185.4977540988191,
    "median_d_ll_collapsed": -60.82630872107659,
    "median_barrier_reduction": 114.31533951516798,
    "p_alpha_gt_1pct": 0.13083333333333333,
    "p_alpha_gt_10pct": 0.10833333333333334,
    "p_log_alpha_ge_0": 0.095,
    "mean_alpha_collapsed": 0.10228732090013501,
    "expected_escapes_50k_mean": 39367.308272157155,
    "expected_escapes_50k_min": 28538.048158632966
  }
}
```

## Structural distance
```json
{
  "d_h=1": {
    "count": 1564,
    "median_d_ll_conditional": -144.52490591529545,
    "median_d_ll_collapsed": -53.53288358987657,
    "median_barrier_reduction": 92.72852944647526,
    "mean_alpha_collapsed": 0.09888429328043052,
    "p_alpha_gt_1pct": 0.12148337595907928,
    "p_log_alpha_ge_0": 0.09462915601023018
  },
  "d_h=2": {
    "count": 643,
    "median_d_ll_conditional": -345.4866875058283,
    "median_d_ll_collapsed": -104.99742231400089,
    "median_barrier_reduction": 239.20498413582288,
    "mean_alpha_collapsed": 0.024626444463130726,
    "p_alpha_gt_1pct": 0.041990668740279936,
    "p_log_alpha_ge_0": 0.02021772939346812
  },
  "d_h>=3": {
    "count": 193,
    "median_d_ll_conditional": -1182.098136275395,
    "median_d_ll_collapsed": -193.86093411692053,
    "median_barrier_reduction": 864.3277634462125,
    "mean_alpha_collapsed": 0.015544041531038728,
    "p_alpha_gt_1pct": 0.015544041450777202,
    "p_log_alpha_ge_0": 0.015544041450777202
  }
}
```

Chains with a useful tail (pre-registered criteria): 8/8.

**COLLAPSED-U ROBUST ACROSS MODES**
