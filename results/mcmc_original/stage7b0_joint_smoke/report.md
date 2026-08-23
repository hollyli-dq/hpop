# Step 7B0 — FFBS joint integration parity and smoke

Status: **PASS**. This is an integration check, not a correctness result; Step 7B1 is where the sampler meets an independent reference.

## Parity

| check | statistic | value | threshold | verdict |
|---|---|---|---|---|
| FFBS conditional vs exact enumeration | max abs log-Z error | 3.553e-15 | 1e-10 | PASS |
| FFBS conditional vs exact enumeration | max conditional TV | 1.793e-15 | 1e-12 | PASS |
| swept state vs direct Stage 6E target | max abs difference | 1.421e-14 | 1e-9 | PASS |
| Stage 6E kernels, segmentation draw off | worst coordinate difference | 0.000e+00 | 0 (bit-identical) | PASS |
| eager vs batched candidate tables | max abs difference | 1.776e-15 | 1e-9 | PASS |
| q_0 = 0 per candidate block | rescore difference | 0 | 0 | PASS |

## Cache lifecycle

* candidate tables built 26 times over 25 segmentation sweeps (plus one taken to test the fingerprint guard): one build per sweep, none per scalar proposal
* a read after the sweep ends raises: True
* a read at moved parameters raises: True
* sweep cost: batched 19.8 ms (table 5.1 ms, charts 5.6 ms, draws 0.15 ms), eager adapter 41.6 ms

## Smoke

* 3,000 sweeps in 56s, 240 retained
* `(S, z)` changed on 3,799 trace-draws; 21 distinct segmentation keys; 3,940 boundary changes; 16,700 occurrence-label changes
* 19 distinct induced `H` states visited
* every scalar moves: {'beta': 0.8154, 'omega': 1.1859, 'lambda_rep': 0.3404, 'lambda_back': 0.7832, 'rho': 0.288}
* FFBS acceptance is exactly 1 by construction: True
* no NaNs: True; deterministic: True; resume bit-identical: True; state round-trip: True
* `pi`/`P` see the new labels: counts differ across the draw (True), `P` diagonal exactly zero (True), transition counts equal the number of adjacent segment pairs (True)

A smoke run demonstrates movement. It is not evidence that the sampler targets the right distribution — that is Step 7B1.

Source commit `77093cb5845a9a2dc3472203657fa62fc6222164`.
