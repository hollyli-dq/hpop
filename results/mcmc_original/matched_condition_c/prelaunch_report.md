# Condition C — pre-launch record

Integration commit `2f146bbce52ca482546a92af2b15b1338fdceed4` on branch `taskbench-external-poset` (merge of Condition B `34873d8` and the collapsed-U validation `58f005e`).

## Small-reference equality (C-COND vs C-MARG, same posterior)
- relation-marginal max |diff| 0.0583 (0 of 60 exceed their MCSE gates)
- boundary max |diff| 0.0303 (gate 0.05); occurrence max |diff| 0.0323 (gate 0.05)
- mean log-target diff 0.154 (gate 1.581)
- verdict: PASS

## Frozen scales (efficiency-only pilot; recovery never inspected)
- conditional U row scale: 0.5 (acceptance 0.334)
- scheduled collapsed scale: 1.0 (acceptance 0.233)
- cadence: c = 10 (fixed by amendment, not tuned)

Conditions A and B, the formal corpus, the smoke corpus and the generator validation are unchanged. NO formal Condition-C chain was launched.
