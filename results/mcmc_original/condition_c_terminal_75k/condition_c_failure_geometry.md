# Condition C — failure geometry

Primary analysis uses draws through sweep 75,000
(13,000 retained draws per chain).

## The progression

**C-COND** — conditional/local structural updates only. Chains remain in
**3 distinct unordered
structural libraries** across 4 chains
(split 2-1-1).
Separated at the structural-library level.

**C-MARG** — path marginalisation added. All four chains agree on
**1 unordered structural
library**, but split
3-1 across anchored
assignments. Shares one unordered structural library; separated in the assignment of that library to anchored skill identities.

The phenomenon is **anchored structure-to-skill assignment multimodality**. It
is not label switching: the fixed `pi*` and `P*` are not invariant under any
non-identity permutation, so the competing assignments are distinct posterior
states rather than one state under an arbitrary relabelling.

## Anchored-assignment gap in C-MARG, by registered checkpoint

| rung | assignment gap (nats) |
|---|---|
| 30,000 | 124.39 |
| 50,000 | 124.44 |
| 75,000 | 124.30 |

The gap is stable across all three registered checkpoints.

## C-COND per chain

| chain | anchored H tuple | unordered library | mean log target | assignment changes after burn-in | last change sweep | path-marginal accepted/proposed | of which changed H |
|---|---|---|---|---|---|---|---|
| cond0 | 9b988e·221530·199042 | 199042·221530·9b988e | -6134.78 | 3 | 25370 | 0/0 | 0 |
| cond1 | 2959d0·0340aa·199042 | 0340aa·199042·2959d0 | -5969.13 | 4 | 51185 | 0/0 | 0 |
| cond2 | 199042·0340aa·2959d0 | 0340aa·199042·2959d0 | -5652.24 | 0 | — | 0/0 | 0 |
| cond3 | 2959d0·4c4174·199042 | 199042·2959d0·4c4174 | -6119.70 | 0 | — | 0/0 | 0 |

## C-MARG per chain

| chain | anchored H tuple | unordered library | mean log target | assignment changes after burn-in | last change sweep | path-marginal accepted/proposed | of which changed H |
|---|---|---|---|---|---|---|---|
| marg0 | 2959d0·0340aa·199042 | 0340aa·199042·2959d0 | -5776.09 | 0 | — | 953/7600 | 7 |
| marg1 | 2959d0·0340aa·199042 | 0340aa·199042·2959d0 | -5776.32 | 0 | — | 985/7600 | 6 |
| marg2 | 199042·0340aa·2959d0 | 0340aa·199042·2959d0 | -5652.00 | 0 | — | 1003/7600 | 5 |
| marg3 | 2959d0·0340aa·199042 | 0340aa·199042·2959d0 | -5776.48 | 0 | — | 1008/7600 | 14 |

## Accepted path-marginal proposals are not assignment crossings

C-MARG accepted **3,949** path-marginal
proposals in total, of which **32**
changed the anchored `H` tuple at all — and
**0** changed the
anchored assignment after burn-in. C-COND, which schedules no path-marginal
move, recorded
7 assignment changes
after burn-in.

## Language

no claim of mathematical impossibility is made; the observation is that no retained C-MARG chain traversed the anchored-assignment barrier within the observed run through 75,000 sweeps. No claim of mathematical impossibility is made: local
row moves could in principle traverse the barrier through a sequence of
low-probability intermediate states.
