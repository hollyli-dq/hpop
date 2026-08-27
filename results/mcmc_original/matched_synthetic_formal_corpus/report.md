# Formal matched-synthetic corpus (frozen)

Source commit: `8ca828153e8e263bf4ea4823e45a53fa454037ad` &middot; generator commit `8ca8281`
Master seed: 6200001 (registered before generation; never used by any prior run)
Corpus hash: `dd280a4a09896154e167f388edd401a9119ba398167c09404aba5f7743e58ec2`
Truth hash: `fc41538fd44d170df8d0a6401f0c6e6b49d52418c487e22f9e4f45ee047f903e`

- 100 training traces: exactly 25 each at J = 24, 32, 40, 48
- 45 held-out traces: lengths cycling (24,32,40,48) -> counts 12/11/11/11
- 785 true blocks; skill frequencies {0: 0.3822, 1: 0.3045, 2: 0.3134}
- repeat-occurrence frequency 0.5121

## Section 4 validation (all PASS)
- illegal widths / cover mismatches / self-transitions / non-finite block likelihoods: 0 / 0 / 0 / 0
- q_0-reset replay max error: 3.553e-15 (< 1e-10)
- generator/scorer complete-data log-probability parity: 1.421e-14 (< 1e-10)
- deterministic save/load parity: True

This corpus is FROZEN. Do not regenerate it because realized block counts, skill counts, or recovery difficulty look inconvenient.
