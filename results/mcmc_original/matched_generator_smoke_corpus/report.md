# Matched-generator smoke corpus

Source commit: `6eb43b355333d07d72d7dffd836b654413702b04`
Master seed: 6100001 (fixed before generation, never searched)
Corpus hash: `81d6393e45870ff7e2ec1e65d272eacab61c7075a6e3fbe9b4dd07818885a3b3`

- 20 training traces (118 blocks, lengths [24, 32, 40, 48])
- 10 held-out traces (50 blocks), split at the trace level
- truth: supplied mode (Stage 6E2 registered configuration), K=3, m=5, d=2, delta_B=0.15, epsilon=0.02, widths [3, 12]
- generator/inference complete-data log-probability parity on this exact corpus: max |diff| = 1.421e-14 (< 1e-10)

This is a SMOKE corpus for serialization, manifest and scorer-parity checks. It is not the headline matched-synthetic corpus, and no inference has been or should be run on it under this task.
