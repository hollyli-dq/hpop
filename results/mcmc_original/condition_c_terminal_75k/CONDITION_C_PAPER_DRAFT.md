# Condition C — paper text

## A. Main-paper paragraph

Conditions A and B establish component identifiability: with the reusable
structures fixed to truth the latent paths are strongly identifiable, and with
the paths fixed to truth the structures are strongly identifiable. Condition C
makes the segmentation, the skill labels and the utility matrices jointly
latent on the same matched corpus, comparing a conditional structural kernel
(C-COND) against one that additionally marginalises the paths out of the
structural acceptance ratio (C-MARG). Neither arm satisfied the registered
convergence criterion at 30k, 50k or 75k sweeps. The two failures differ in
kind. C-COND chains remain separated at the structural-library level, occupying
3 distinct unordered
libraries across four chains. C-MARG chains all identify the *same* unordered
library, and are separated only in how that library is assigned to the anchored
skill identities, splitting
3-1 across two
assignments whose typical log-target levels differ by
124-124 nats at every registered checkpoint. Because the
initial distribution and transition matrix are held fixed and are not invariant
under any non-identity permutation of skill identities, these assignments are
distinct posterior states rather than an arbitrary relabelling. Path
marginalisation therefore removes the structural-library barrier but leaves a
coordinated anchored-assignment barrier that no retained C-MARG chain traversed
after burn-in within the registered budget.

## B. Appendix paragraph

The two barriers are distinguishable in the movement diagnostics. Across the
terminal 75k budget the C-MARG arm accepted
3,949 path-marginal proposals, of which
32 changed the anchored `H`
tuple at all and
0 did so after burn-in:
accepted path-marginal proposals are overwhelmingly within-cell refinements
rather than assignment crossings. C-COND, which schedules no such move,
recorded 7
assignment changes after burn-in, all of them relocations between distinct
libraries rather than convergence toward a common one. Under the recovery
metrics predefined before unsealing, and read strictly as descriptive
diagnostics conditional on non-converged chains,
4 of four C-MARG
chains recovered the true unordered library against
1 of four in
C-COND, while in both arms exactly one chain recovered the true anchored
assignment. Per-chain closure F1 separates accordingly, and the held-out
oracle-path negative log-likelihood of the single correctly-assigned chain
reproduces the generating truth to four decimal places (1.0772 nats per
occurrence) against 1.7528 for an antichain baseline.

## C. Table caption

Registered checkpoint trajectory for Condition C. Both arms failed the frozen
convergence gate at every rung; the gate is reported exactly as registered,
with anchored per-skill summaries and no permutation-invariant substitute. The
right-hand columns give the truth-free explanatory diagnosis: the number of
distinct unordered structural libraries held across the four chains of each
arm, the anchored-assignment split, and the typical log-target gap between the
two C-MARG assignments.

## D. Figure caption

Condition C failure geometry. Conditional structural updates (C-COND) leave the
four chains in different unordered structural libraries. Adding path
marginalisation (C-MARG) brings all four chains onto a common library, but they
remain split 3-1 across
assignments of that library to the anchored skill identities, separated by
124-124 nats of typical log target and stable across all
three registered checkpoints. Because the fixed initial and transition
distributions are not permutation invariant, the two assignments are distinct
posterior states.

## E. Limitation sentence

Condition C was terminated after the third consecutive failed registered
checkpoint at 75k sweeps rather than at the preregistered 100k ceiling, because
the two-consecutive-pass criterion could no longer be satisfied; the reported
recovery quantities are descriptive diagnostics conditional on non-converged
chains and are not posterior estimates, and the observed absence of
anchored-assignment crossings is a statement about the registered budget rather
than a claim of impossibility.

## F. Transition sentence motivating Condition C'

Because the residual barrier is a coordinated reassignment of whole structures
between anchored identities rather than a local perturbation of any one of
them, we prospectively froze a follow-up experiment that adds a single global
transposition move, scored by the same path-marginal likelihood, and report it
separately as Condition C'.
