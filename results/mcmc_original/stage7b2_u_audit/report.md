# U-proposal audit at the frozen chain states

~10,000 proposals per chain from the REGISTERED U kernel (`sampler_u.propose_row`,
scale 0.5, `log_alpha = dLL + dPrior`, no Hastings term), drawn at the latest
checkpoint state of every chain of both experiments. Diagnosis only: no state was
updated, no formal chain touched. Full per-proposal arrays in
`*_chain*_proposals.npz`, machine-readable summary in `audit.json`.

## The two rates

| experiment | chain | rel/skill | r_propose | r_accept \| crossing | escape / proposal |
|---|---|---|---:|---:|---:|
| 7B2 FFBS | 0 | [1,3,5] | 0.530 | 8.9e-18 | 4.7e-18 |
| 7B2 FFBS | 1 | [5,5,2] | 0.544 | 1.5e-20 | 8.1e-21 |
| 7B2 FFBS | 2 | [3,3,3] | 0.573 | 1.1e-11 | 6.3e-12 |
| 7B2 FFBS | 3 | [3,5,7] | 0.446 | 1.1e-20 | 4.9e-21 |
| 6E2 Local | 0 | [2,3,3] | 0.589 | 1.5e-13 | 8.7e-14 |
| 6E2 Local | 1 | [2,5,2] | 0.455 | 3.3e-12 | 1.5e-12 |
| 6E2 Local | 2 | [4,2,2] | 0.538 | 1.7e-17 | 9.4e-18 |
| 6E2 Local | 3 | [3,4,6] | 0.488 | 2.1e-20 | 1.0e-20 |

**The proposal is not the bottleneck.** Roughly half of all single-row proposals
already induce a different order `h(U')`. **The target is.** A crossing proposal's
acceptance probability is 1e-11 to 1e-20; with 15 U proposals per sweep the
per-sweep escape probability is at most ~1e-10, i.e. no feasible run length
(10^5 sweeps) escapes a mode this kernel family occupies.

## What kills the crossings

`killed-by-LL = 1.00` in every chain: the structural prior moves by fractions of a
nat (median ~-0.15, max +3.5) while the likelihood drops by a cliff — median
dLL -150 to -394 nats, and **the best crossing proposal out of 10,000 is still
-21 nats down** (max log_alpha -20.7 to -40.6). Neighbouring order cells reachable
by perturbing ONE row are all likelihood-catastrophic; the genuinely competitive
orders (the modes the OTHER chains occupy, hundreds of nats apart in log target)
differ in many rows at once and are simply not adjacent under this kernel.

## Controls

* Same-H proposals accept at 0.73-0.76 in every chain — the within-cell walk is
  healthy and matches the formal runs' observed U acceptance.
* Negative control: max |dLL| over same-H proposals is exactly 0.0 in all 8 chains —
  the target reads `U` only through `h(U)`, as registered.

## Reading

The (S,z)-U locking is a property of the TARGET's geometry — the conditional
`p(U | S, z, Theta)` on a 3,200-block corpus puts ~200-nat walls between order
cells — not of either segmentation kernel. This is mechanism-level confirmation of
what the intermediate 7B2 chains already suggested: exact FFBS on `(S, z)` cannot
free `U`, because no amount of segmentation refreshment changes the fact that the
U kernel's only exits are single-row perturbations into likelihood cliffs. A Step 8
kernel must jump BETWEEN competitive orders directly (order-level exchange /
mode-jumping proposals with a coordinated multi-row U update, or tempering on the
structural coordinate).

Audit seed 8,150,000; corpus hash `02be246e...` (matches both experiments);
U scale 0.5 (registered in both). The 6E2 checkpoints were read, not written;
neither formal run was altered or steered by this audit.
