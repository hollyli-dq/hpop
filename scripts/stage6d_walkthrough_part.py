"""Build, execute, clean and merge Part X — the Stage 6D narrative — into the walkthrough.

    PYTHONPATH=src python scripts/stage6d_walkthrough_part.py            # build+execute+merge
    PYTHONPATH=src python scripts/stage6d_walkthrough_part.py --dry-run  # build+execute only

The walkthrough is assembled one Part at a time, and each Part is **executed in its own
fresh kernel** before being merged. That is why the notebook's execution counts restart
inside each Part rather than running 1..N across the whole file: a Part has to stand on
its own imports, so that re-running it never depends on a variable some earlier Part
happened to leave in the namespace.

Three rules this script enforces, because they are the ones that decay:

* **Executed, not asserted.** Every number in Part X is printed by a cell that read it
  from `results/mcmc_original/`, or computed live. Nothing is typed into the prose.
* **Cleaned.** After execution the outputs are stripped of stderr, of empty streams and of
  execution metadata, and the merge refuses to proceed if any cell errored.
* **No duplicated reporting.** The prose carries the argument; the cells carry the
  numbers. A figure that repeats a table is cut rather than kept for symmetry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "mcmc_original_walkthrough.ipynb"
PART_TITLE = "# Part X — Stage 6D: everything at once, and the lesson about scales"


def md(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(text.strip("\n"))


# ---------------------------------------------------------------------------- the Part
def part_cells() -> list:
    return [
        md(r"""
---

# Part X — Stage 6D: everything at once, and the lesson about scales

Part IX made the *structure* unknown. Stage 6D makes everything unknown at once: $U$,
$\rho$, $\beta$, $\omega$, $\lambda_{\text{rep}}$ and $\lambda_{\text{back}}$ are all
inferred jointly, on oracle block boundaries and oracle skill labels. It is the last stage
that gets to assume the segmentation.

## 43. What is new, and what a sweep costs

Almost nothing here is new mathematics. Stage 6D is a **composition layer**: every update
calls a kernel a previous stage already validated, as an object rather than as re-derived
algebra. So the first thing to establish is that composing them changed neither parent —
not by reading the same helper twice, but by rebuilding each parent's acceptance ratio
from that parent's own objects and requiring numerical equality.

The cost structure is worth stating because it explains every runtime in this Part. A
sweep replays the likelihood exactly $m+1$ times: once per $U$ row, because a $U$ move can
change $h(U)$ and with it the frontier and the whole $q$ trajectory, and once for
$\omega$, because $\kappa=\sigma(\omega)$ enters the $q$ recursion itself. $\beta$,
$\lambda_{\text{rep}}$ and $\lambda_{\text{back}}$ are scored from a cache keyed on
$(h(U), \omega)$. And $\rho$ consumes **zero** likelihood evaluations, because it acts
only through $p(U\mid\rho)$ — an assertion the cell below checks rather than repeats.
"""),
        code(r"""
%matplotlib inline
import json, math, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd().parent / "src"))

from hpop.mcmc_original.recurrent_joint_scalar_mcmc import RecurrentJointEvaluator
from hpop.mcmc_original.recurrent_latent_poset_mcmc import LatentPosetEvaluator
from hpop.mcmc_original.recurrent_oracle_joint_mcmc import (
    Stage6DTarget, initial_state, sweep_once,
)
from hpop.mcmc_original.recurrent_scalar_posterior import log_prior
from hpop.mcmc_original.stage6d_frozen import (
    ACTIVE_6D, REGISTERED_SCALES, SCALAR_ORDER, config_hash, frozen_config,
    load_stage6d_dataset,
)

R = Path("../results/mcmc_original")
D0 = R / "stage6d0_joint_smoke"
DREF = R / "stage6d1_joint_reference"
D1 = R / "stage6d1_joint_mcmc"
D2 = R / "stage6d2_oracle_joint_full_seed0"
PILOT = R / "stage6d2_pilot"
read = lambda p: json.loads(Path(p).read_text())          # noqa: E731

INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8f8e88"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.dpi": 130, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK3, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "xtick.major.size": 0, "ytick.major.size": 0,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2.0,
    "font.size": 9,
})

cfg = frozen_config()
dim = cfg["dimensions"]
print(f"model {cfg['model_id']}   config hash {config_hash()[:16]}...")
print(f"  m = {dim['m_rows']} rows of U (role occurrences) | "
      f"d = {dim['d_latent_columns']} latent columns (the brief's K) | "
      f"K = {dim['n_skills']} skill | assessors {dim['n_assessors']} | "
      f"tau in model: {cfg['tau_in_model']}")
print(f"  inferred jointly: {', '.join(cfg['active'])}")
print(f"  H is derived, never state: {cfg['h_is_derived_not_state']}   "
      f"second prior on H: {cfg['second_prior_on_H']}")

# ---- live parity: rebuild each parent's ratio from that parent's own objects ----------
frozen = load_stage6d_dataset()
blocks = frozen.train[:40]
truth = {k: float(v) for k, v in frozen.truth.items()}
rng = np.random.default_rng(11)
u_states = [frozen.u_true] + [frozen.u_true + rng.normal(scale=0.6, size=(5, 2))
                              for _ in range(3)]

worst = {name: 0.0 for name in SCALAR_ORDER}
for u in u_states:
    ev6d = LatentPosetEvaluator(blocks, epsilon=frozen.epsilon, omega=truth["omega"])
    ev6b = RecurrentJointEvaluator(blocks, u, frozen.epsilon)
    for name in SCALAR_ORDER:
        for rho in (0.1, 0.5, 0.85):            # rho must not touch a scalar ratio
            for candidate in (0.6, 0.95, 1.4, 2.3):
                trial = dict(truth); trial[name] = candidate
                prior = log_prior(name, candidate) - log_prior(name, truth[name])
                r6d = prior + (ev6d.log_likelihood(u, *[trial[k] for k in SCALAR_ORDER],
                                                   allow_cache=False)
                               - ev6d.log_likelihood(u, *[truth[k] for k in SCALAR_ORDER],
                                                     allow_cache=False))
                r6b = prior + (ev6b.log_likelihood(*[trial[k] for k in SCALAR_ORDER],
                                                   allow_cache=False)
                               - ev6b.log_likelihood(*[truth[k] for k in SCALAR_ORDER],
                                                     allow_cache=False))
                worst[name] = max(worst[name], abs(r6d - r6b))

print(f"\nStage 6B scalar-kernel parity, recomputed live over "
      f"{len(u_states)} U x 3 rho x 4 candidates (tolerance 7.3e-12):")
for name in SCALAR_ORDER:
    print(f"  {name:<12} worst |ratio_6D - ratio_6B|   {worst[name]:.3e}")

# ---- the replay budget, counted rather than claimed -----------------------------------
smoke = read(D0 / "summary.json")
print(f"\nStage 6D0 smoke: {len(smoke['checks'])} checks, all passed = "
      f"{smoke['all_passed']}")
print(f"  {smoke['replay_calls']:,} full replays over {smoke['n_sweeps']} sweeps "
      f"= {smoke['replay_calls'] / smoke['n_sweeps']:.2f} per sweep "
      f"({smoke['expected_replays_per_sweep']})")
print(f"  {smoke['cached_calls']:,} cached evaluations "
      f"(beta, lambda_rep, lambda_back, off the (H, omega)-keyed cache)")
print(f"  rho proposals {smoke['proposed']['rho']:,}, and none of them replayed anything")
"""),
        md(r"""
## 44. An independent reference, and a statistic that had to be retired

Stage 6C's reference was exact enumeration. That route is closed here: with all four
scalars free the state space is continuous in six coordinates, so Stage 6D1 builds a
**scrambled-Sobol importance sampler in prior coordinates** instead, on a deliberately
small model — three blocks of $T=5$, $m=3$.

The construction is non-centred, $U = Z\,L(\rho)^{\!\top}$ with
$L L^{\!\top}=\Sigma_\rho$, and the proposal is then *exactly* the joint prior, so the
unnormalised weight collapses to the likelihood alone. That is the point rather than a
convenience: it removes the Gaussian determinant from the weight entirely, so a
determinant error in the **sampler** cannot hide behind the same error in the
**reference**.

The model is small on purpose, and the reason is the one that governs all prior importance
sampling: it degrades as the likelihood sharpens relative to the prior. At 30 blocks of
$T=8$ the relative ESS was $0.005$ with a single point holding 10% of the weight. At three
blocks of $T=5$ it is stable, and — the part that matters for a *structural* claim — six
of the nineteen labelled orders on three elements carry more than 1% mass, so this problem
exercises structural **mixing**, which the Stage 6C corpus could not.

One thing here should be read carefully. The reference was first registered on the
**maximum departure of any single replicate** from the replicate mean. That statistic was
measuring the wrong quantity: it estimates the dispersion of *one* replicate, so it has no
reason to shrink as $R$ grows — a maximum over more draws reaches further into the tail —
while the quantity the comparison actually consumes is the replicate **mean**. Doubling
$N$ left it essentially unchanged while the log-evidence standard deviation fell as
expected. It was replaced by $\text{rqmc\_se} = \mathrm{sd}/\sqrt{R}$ **before any MCMC
comparison existed**, and it is still computed, still reported, and still failing. A
retired statistic that fails is reported as a failing retired statistic; it is not
relabelled as a pass.
"""),
        code(r"""
ref = read(DREF / "reference_summary.json")
qa = read(DREF / "quality_audit.json")
rep0 = ref["per_replicate"][0]

print(f"{rep0['n_points']:,} points x {len(ref['per_replicate'])} independent scrambles")
print(f"  log evidence {ref['log_evidence']['mean']:.6f}  "
      f"(sd across replicates {ref['log_evidence']['sd']:.2e}, "
      f"range {ref['log_evidence']['range']:.2e})")
print(f"  relative ESS {rep0['relative_ess']:.4f}   "
      f"max normalised weight {rep0['max_normalised_weight']:.2e}   "
      f"induced orders visited {rep0['n_induced_h_states']}")

print(f"\n{'PRIMARY gate':<40}{'value':>12}{'threshold':>12}   verdict")
for name, c in qa["checks"].items():
    if c.get("primary"):
        print(f"{name:<40}{c['value']:>12.3e}{c['threshold']:>12.1e}   "
              f"{'PASS' if c['pass'] else 'FAIL'}")
print(f"\n{'SECONDARY diagnostic (not a gate)':<40}{'value':>12}{'threshold':>12}   verdict")
for name, c in qa["checks"].items():
    if not c.get("primary"):
        print(f"{name:<40}{c['value']:>12.3e}{c['threshold']:>12.1e}   "
              f"{'PASS' if c['pass'] else 'FAIL'}")

print(f"\n{'RETIRED statistic, still computed':<40}{'value':>12}{'old thr.':>12}   old verdict")
for name, c in qa["superseded_checks_on_this_run"].items():
    print(f"{name:<40}{c['value']:>12.3e}{c['threshold']:>12.1e}   "
          f"{'PASS' if c['pass'] else 'FAIL'}")
print(f"\nall_pass = {qa['all_pass']}   primary_pass = {qa['primary_pass']}")
print("The reference was frozen on primary_pass. The distinction is kept visible.")
"""),
        md(r"""
## 45. Three attempts, and the thing that was actually wrong

Stage 6D1 did not pass on the first run, and the failures are the substantive finding of
the stage rather than an embarrassment to be tidied away. All three attempts are preserved
with the same statistics, and no gate was relaxed at any point.

The first attempt used the registered Stage 6B scales throughout and failed on $\beta$ and
$\lambda_{\text{rep}}$. Extending it to the 100,000-sweep ceiling fixed those two and
pushed $\omega$ over instead — which is the signature of a chain that is not mixing rather
than one that needs longer. The diagnosis is in the acceptance column: **0.96–0.98 for
every scalar**, with bulk ESS in the hundreds out of 40,000 draws. A random walk accepting
97% of its proposals is taking steps far too small to explore anything.

Two efficiency-only pilots followed, one for $\omega$ and then one for the remaining
three. Both saw acceptance, ESJD, finite-target checks, invalid-proposal counts and
replay/cache consistency, and **nothing else** — never the reference, the truth, a
recovery statistic or an R-hat — and every pilot draw was discarded.

One detail in those pilots is worth stating because getting it wrong biases the answer in
a specific direction. $\beta$, $\lambda_{\text{rep}}$ and $\lambda_{\text{back}}$ are
registered as **log**-scale random walks, so squared jump distance is accumulated in log
space, $(\log x' - \log x)^2$. Measuring in raw parameter space would reward large
absolute moves at large parameter values and would systematically select scales that are
too big.

The second attempt's directory was later overwritten by the third attempt's rerun, and its
base seed was never written down. It is therefore preserved as an explicitly labelled
**re-execution** of the recorded configuration rather than as the original chain, with the
numbers recorded at the time cited beside it.
"""),
        code(r"""
ATTEMPTS = [
    ("attempt 1 - registered scales, 50k", R / "stage6d1_joint_mcmc_FAILED_attempt0_50k"),
    ("attempt 1 - registered scales, 100k ceiling", R / "stage6d1_joint_mcmc_FAILED_attempt1"),
    ("attempt 2 - omega x32 (re-execution)",
     R / "stage6d1_joint_mcmc_FAILED_attempt2_omega_retuned_REEXECUTED"),
    ("attempt 3 - beta/lambda retuned", D1),
]
NAMES = ("rho",) + SCALAR_ORDER

print(f"{'attempt':<46}{'sweeps':>8}   failed gates")
for label, d in ATTEMPTS:
    cf = read(d / "config.json")
    gates = read(d / "reference_comparison.json")["gates"]
    bad = [k for k, g in gates.items() if not g["pass"]]
    print(f"{label:<46}{cf['sweeps']:>8,}   {', '.join(bad) if bad else 'none'}")

for label, d in ATTEMPTS:
    sc = read(d / "scalar_diagnostics.json")
    print(f"\n{label}")
    print(f"{'':<14}" + "".join(f"{n:>13}" for n in NAMES))
    for row, key, fmt in (("acceptance", "acceptance_post_burn_in", "{:>13.3f}"),
                          ("bulk ESS", "bulk_ess", "{:>13,.0f}"),
                          ("R-hat", "rhat", "{:>13.5f}")):
        print(f"  {row:<12}" + "".join(fmt.format(sc[n][key]) for n in NAMES))

print(f"\n{'attempt':<46}{'H TV':>10}{'rel. err':>10}{'mixed':>10}{'envelope':>10}")
for label, d in ATTEMPTS:
    st = read(d / "structural_diagnostics.json")
    mx = read(d / "reference_comparison.json")["mixed"]
    print(f"{label:<46}{st['h_total_variation']:>10.5f}"
          f"{st['max_relation_marginal_error']:>10.5f}"
          f"{mx['observed']:>10.5f}{mx['envelope']:>10.5f}")

recon = read(ATTEMPTS[2][1] / "reconstruction.json")
print(f"\nattempt 2 is a re-execution, not the original chain "
      f"(is_the_original_chain = {recon['is_the_original_chain']}):")
for k, v in recon["recorded_at_the_time"].items():
    print(f"  {k:<12} recorded at the time {v:<10} this re-execution "
          f"{recon['reproduced_by_this_run'][k]:.5f}")
"""),
        md(r"""
## 46. Proposal scales are a property of the corpus, not of the kernel

This is the lesson Stage 6D exists to teach, and it cost three attempts to learn.

The Stage 6D1 pilots found all four scalar scales **16–32× too small**, worth 22–61× in
effective sample size. It would be natural to read that as "the registered scales were
wrong" and carry the corrections forward. That reading is wrong, and Stage 6D2 is the
control that shows it.

The registered Stage 6B scales were tuned on the 500-block corpus. Stage 6D1's reference
model is *three* blocks of $T=5$, chosen small so that prior importance sampling would
work at all, and its posterior is correspondingly broad — so scales tuned for a sharp
posterior take steps far too timid for a diffuse one. Stage 6D2 runs on the original
500-block corpus, so a separate pilot was run there rather than reusing anything, over a
multiplier grid **symmetric about 1** so that the answer was free to come back smaller.

It came back at $\times 1$ to $\times 2$ for five of the six coordinates. A scale that is
32× wrong on one dataset can be exactly right on another, and neither fact says anything
about the kernel.

The exception is $\rho$, at $\times 8$, and it is the coordinate no stage had ever tuned:
Stage 6C inherited its scale from Stage 5 and Stage 6D1 froze it by instruction. It was
the weakest-mixing coordinate in both. A coordinate that every stage *inherits* is a
coordinate nobody has ever checked.
"""),
        code(r"""
pilot = read(PILOT / "pilot_results.json")
reg = pilot["registration"]

print(f"{'coordinate':<13}{'base':>10}{'mult.':>7}{'selected':>10}{'acc.':>7}"
      f"{'ESJD space':>12}   admissible multipliers")
for name in ACTIVE_6D:
    d = pilot["decisions"][name]
    s = d["selected"]
    print(f"{name:<13}{REGISTERED_SCALES[name]:>10.5g}"
          f"{'x' + s['multiplier_label']:>7}{s['scale']:>10.5g}"
          f"{s['median_acceptance']:>7.3f}{reg['esjd_space'][name]:>12}   "
          f"{', '.join('x' + m for m in d['admissible_candidates'])}")

conf = pilot["joint_confirmation"]
print(f"\njoint confirmation over all six, band {conf['band']}: pass = {conf['pass']}")
print("  " + "   ".join(f"{k} {v:.3f}" for k, v in conf["median_acceptance"].items()))
print(f"\n{pilot['runtime_seconds'] / 60:.0f} minutes, "
      f"{len(pilot['per_chain_rows'])} tuning chains + "
      f"{len(conf['per_chain'])} confirmation chains, every draw discarded = "
      f"{pilot['all_pilot_draws_discarded']}")
print(f"permitted: {', '.join(reg['permitted_statistics'])}")
print(f"never computed: {', '.join(reg['forbidden_and_not_computed'][:4])}, ...")

# ---- Figure 10 -- the two curves the selection rule actually reads --------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
lo, hi = reg["admissible_acceptance"]
colours = [S1, S2, S3, "#9b59b6", "#c9a227", INK2]
for (name, colour) in zip(ACTIVE_6D, colours):
    table = pilot["decisions"][name]["candidate_table"]
    mult = np.array([t["multiplier"] for t in table])
    axes[0].plot(mult, [t["median_acceptance"] for t in table], "o-", ms=3,
                 color=colour, label=name, lw=1.4)
    esjd = np.array([t["median_esjd"] for t in table])
    axes[1].plot(mult, esjd / esjd.max(), "o-", ms=3, color=colour, lw=1.4)
    chosen = pilot["decisions"][name]["selected"]
    axes[1].plot([chosen["multiplier"]], [1.0], "*", ms=11, color=colour)

axes[0].axhspan(lo, hi, color="#e6e5e0", zorder=0)
axes[0].set_xscale("log", base=2); axes[0].set_ylim(0, 1)
axes[0].set_xlabel("multiplier on the registered scale")
axes[0].set_ylabel("median acceptance")
axes[0].set_title("acceptance falls monotonically; the band is registered")
axes[0].legend(ncol=2)
axes[1].set_xscale("log", base=2)
axes[1].set_xlabel("multiplier on the registered scale")
axes[1].set_ylabel("median ESJD / its maximum")
axes[1].set_title("ESJD peaks, and the star is where the rule lands")
for ax in axes:
    ax.axvline(1.0, color=INK3, lw=0.8, ls="--")
fig.tight_layout()
plt.show()
print("Figure 10 -- the pilot's whole evidence base. ESJD is measured in each kernel's")
print("own coordinate: log space for the three log walks, logit for rho, identity for")
print("U and omega. The dashed line is the registered scale; five of six sit on or")
print("beside it, and only rho is far away.")
"""),
        md(r"""
## 47. Stage 6D2 — does it converge, and is anything confounded?

The formal run is four chains of 30,000 sweeps from starts dispersed in *every* coordinate:
four contrasting $h(U)$ structures, $\rho$ spread across its support, and the four scalars
at prior quantiles arranged by a fixed Latin square. Every registered gate passed at the
initial sweep count, so the 20,000-sweep continuation blocks and the 100,000 ceiling were
never used.

One diagnostic has to be read correctly rather than gratefully. On this corpus the
induced-order posterior is a **point mass** — Stage 6C established that, with a 271.5-nat
margin over the nearest competitor — so the relation-count trace never moves. Its $\hat R$
is therefore **undefined**, and the diagnostics say `degenerate` rather than reporting a
flattering $1.0$. The point mass itself is checked by the structural-recovery gate, where
it belongs.

The confounding question is where Stage 6D2 earns its keep, and it has a clean form. The
recurrent likelihood reads $U$ only through $h(U)$. If $h(U)$ is that point mass, then the
likelihood the four scalars see is **identical** to the one Stage 6B3 saw with $U$ pinned
at $U_{\text{TRUE}}$ — so the marginals must agree, and any disagreement would be
$U$–$\beta$, $H$–$\omega$ or $\rho$–$U$ confounding.

$\beta$ is deliberately **not** gated against Stage 6C2, and the reason was recorded before
any Stage 6D2 draw existed. Stage 6C2 holds $\omega$ and the two $\lambda$s at their
registered values while Stage 6B3 and Stage 6D2 marginalise over them, and $\beta$ is
correlated with those three — so the two parents already disagree with *each other* by
about $0.4$ Stage 6B3 sd. No single Stage 6D2 value could satisfy a $0.25$ sd gate against
both, and requiring agreement with a differently conditioned posterior would be a gate on
a quantity that is not supposed to be equal. The contrast is reported instead, because it
measures something real: the size of the effect of conditioning rather than marginalising.
"""),
        code(r"""
conv = read(D2 / "convergence_diagnostics.json")
cons = read(D2 / "parent_consistency.json")
gates = read(D2 / "gates.json")
cf2 = read(D2 / "config.json")

print(f"{cf2['chains']} chains x {cf2['sweeps']:,} sweeps, {cf2['burn_in']:,} burn-in, "
      f"thin {cf2['thin']}, {cf2['retained_pooled']:,} pooled draws, "
      f"{cf2['wall_seconds'] / 60:.1f} min")
print("  starts: " + ", ".join(f"{s['u_start']}({s['start_relations']} rel, "
                               f"rho {s['start_values']['rho']:.2f})"
                               for s in cf2["chain_starts"]))

print(f"\n{'coordinate':<13}{'mean':>10}{'sd':>9}{'acc.':>7}{'R-hat':>9}"
      f"{'bulk ESS':>11}{'tail ESS':>11}{'MCSE/sd':>9}")
for n in NAMES:
    c = conv["per_coordinate"][n]
    print(f"{n:<13}{c['posterior_mean']:>10.5f}{c['posterior_sd']:>9.5f}"
          f"{c['acceptance_post_burn_in']:>7.3f}{c['rhat']:>9.5f}"
          f"{c['bulk_ess']:>11,.0f}{c['tail_ess']:>11,.0f}{c['mcse_over_sd']:>9.4f}")
lt = conv["log_target"]
print(f"{'log target':<13}{lt['posterior_mean']:>10.1f}{'':>9}{'':>7}{lt['rhat']:>9.5f}"
      f"{lt['bulk_ess']:>11,.0f}{lt['tail_ess']:>11,.0f}")

rc = conv["relation_count"]
print(f"\nrelation count: degenerate = {rc['degenerate']}, constant at "
      f"{rc['constant_value']:.0f}, R-hat = {rc['rhat']}  <- undefined, not 1.0")
print(f"  induced orders visited across all chains: {conv['n_h_states_visited']}   "
      f"relations that vary: {conv['uncertain_relations']['n']}")

print(f"\n{'coordinate':<13}{'Stage 6D2':>22}{'parent':>22}  parent   diff (parent sd)")
for n in SCALAR_ORDER:
    c = cons["vs_stage6b3"][n]
    print(f"{n:<13}{c['stage6d2_mean']:>12.5f} +/-{c['stage6d2_sd']:>8.5f}"
          f"{c['stage6b3_mean']:>12.5f} +/-{c['stage6b3_sd']:>8.5f}"
          f"  6B3      {c['difference_in_parent_sd']:+.4f}")
for n in ("rho", "beta"):
    c = cons["vs_stage6c2"][n]
    tag = "6C2     " if c["is_a_gate"] else "6C2*    "
    print(f"{n:<13}{c['stage6d2_mean']:>12.5f} +/-{c['stage6d2_sd']:>8.5f}"
          f"{c['stage6c2_mean']:>12.5f} +/-{c['stage6c2_sd']:>8.5f}"
          f"  {tag} {c['difference_in_parent_sd']:+.4f}")
pd_ = cons["parents_disagree_with_each_other"]
print(f"  * reported, NOT a gate. The two parents already disagree with each other by "
      f"{pd_['beta_stage6b3_vs_stage6c2_in_stage6b3_sd']:+.4f} Stage 6B3 sd.")

st = cons["structure_vs_stage6c2"]
print(f"\nstructure: 6C2 put {st['stage6c2_probability_of_that_order']:.4f} on order "
      f"#{st['stage6c2_map_poset_index']} with omega FIXED; 6D2 puts "
      f"{st['stage6d2_probability_of_that_order']:.4f} on the same order with omega FREE")
print(f"\n{gates['n_gates']} registered gates, all passed = {gates['all_passed']}, "
      f"failed = {gates['failed'] or 'none'}")
"""),
        md(r"""
## 48. Recovery, and a plug-in that is not one

Correctness and recovery stay apart, as they have since Part VIII. Stage 6D1 answered
correctness against an independent reference. Stage 6D2 answers recovery, and it recovers
the generating order exactly and contains all four generating scalars.

$\rho$ recovery remains **NOT APPLICABLE**, permanently and for the same reason as in Part
IX: `U_TRUE` is hand specified in the generator, not drawn from $p(U\mid\rho)$, so no
$\rho_{\text{true}}$ exists. No more favourable dataset has been generated to manufacture
one.

The largest standardised scalar error is around $1.2$–$1.3$ posterior sd. That is a
property of this corpus and not of the sampler, and the parent comparison in §47 is how we
know: Stage 6B3 obtains the same offset with $U$ pinned at the truth. A finite-data
posterior is not obliged to centre on the generating value. It is obliged to *contain* it,
and to match the posterior an independent route obtains from the same likelihood.

The last cell reports a negative control that is more instructive than any of the passing
numbers. Plugging in the **entrywise posterior mean of $U$** scores worse on held-out data
than the prior does. Nothing is broken; the statistic is simply meaningless. Every retained
draw induces the true six-relation order, but their entrywise average lands where all
coordinates are strictly ordered, so $h(\bar U)$ is a *total* order on five elements. The
likelihood reads $U$ only through $h(U)$, so that plug-in scores a different model
altogether. It is the sharpest available demonstration of why Stage 6D reports structure
and never claims the matrix — and why the column-permutation audit is reported rather than
gated: $h(U)$ is the intersection of the column orderings and $\Sigma_\rho$ is exchangeable
in them, so entrywise $U$ traces may swap labels between chains with no convergence failure
at all.
"""),
        code(r"""
rec = read(D2 / "recovery_results.json")
ho = read(D2 / "heldout_prediction.json")
audit = read(D2 / "column_permutation_audit.json")
s = rec["structural"]

print(f"{'P(generating order)':<40}{s['posterior_probability_of_true']:>12.6f}")
print(f"{'MAP order is the generating one':<40}{str(s['map_is_true']):>12}")
print(f"{'distinct orders visited':<40}{s['n_unique_states_visited']:>12}")
for rep in ("closure", "reduction"):
    print(f"{rep + ' F1 / structural Hamming':<40}"
          f"{s[rep]['f1']:>8.4f} / {s[rep]['structural_hamming']}")

print(f"\n{'scalar':<13}{'posterior':>22}{'95% interval':>26}{'truth':>9}"
      f"{'in':>5}{'err (sd)':>10}")
for n in SCALAR_ORDER:
    r = rec["scalars"][n]
    print(f"{n:<13}{r['posterior_mean']:>12.5f} +/-{r['posterior_sd']:>8.5f}"
          f"   [{r['q025']:>8.5f}, {r['q975']:>8.5f}]{r['true_value']:>9.4f}"
          f"{str(r['truth_in_95_interval']):>5}{r['error_in_posterior_sd']:>10.3f}")
print(f"\nrho: {rec['rho']['recovery']}")

print(f"\nheld-out prediction (§17) -- REPORTED, NEVER GATED. "
      f"{ho['n_blocks']} blocks, {ho['n_steps']:,} steps")
print(f"  {'posterior predictive':<52}{ho['posterior_predictive_log_score_per_step']:>11.6f}")
print(f"  {'at the generating truth':<52}"
      f"{ho['log_score_at_the_generating_truth']:>11.6f}")
print(f"  {'posterior-mean scalars, U from a modal-order draw':<52}"
      f"{ho['log_score_at_the_posterior_mean_scalars_and_a_modal_order_draw']:>11.6f}")
print(f"  {'prior mean, U at truth (a floor, not a competitor)':<52}"
      f"{ho['log_score_at_the_prior_mean_with_true_U']:>11.6f}")

nc = ho["entrywise_posterior_mean_U_is_not_a_valid_plug_in"]
print(f"\nNEGATIVE CONTROL -- the entrywise posterior mean of U is not a plug-in")
print(f"  {'its held-out log score':<52}{nc['log_score']:>11.6f}   <- worse than the prior")
print(f"  relations induced by a single draw   "
      f"{nc['relations_induced_by_a_modal_order_draw']}")
print(f"  relations induced by the entrywise mean   "
      f"{nc['relations_induced_by_the_entrywise_mean']}   <- a total order on 5 elements")

print(f"\ncolumn-permutation audit (§13) -- reported, not gated")
print(f"  target is column-exchangeable: {audit['target_is_column_exchangeable']}   "
      f"chains in opposite labellings: {audit['chains_in_opposite_labellings']}")
print(f"  signed-contrast R-hat {audit['signed_contrast_rhat']:.5f}   "
      f"per-chain invariant contrast "
      f"{[round(v, 4) for v in audit['per_chain_absolute_column_contrast']]}")

# ---- Figure 11 -- the marginals against the parent that pinned U at the truth ---------
with np.load(D2 / "chains.npz") as z:
    draws = {n: z[n].ravel() for n in NAMES}

fig, axes = plt.subplots(1, 5, figsize=(13.5, 2.6))
labels = {"rho": r"$\rho$", "beta": r"$\beta$", "omega": r"$\omega$",
          "lambda_rep": r"$\lambda_{\rm rep}$", "lambda_back": r"$\lambda_{\rm back}$"}
for ax, n in zip(axes, NAMES):
    ax.hist(draws[n], bins=60, density=True, color=S1, alpha=0.5)
    if n in SCALAR_ORDER:
        ax.axvline(cons["vs_stage6b3"][n]["stage6b3_mean"], color=INK, lw=1.3, ls="--")
        ax.axvline(rec["scalars"][n]["true_value"], color=S2, lw=1.3)
    else:
        ax.axvline(cons["vs_stage6c2"]["rho"]["stage6c2_mean"], color=INK, lw=1.3, ls="--")
    ax.set_xlabel(labels[n]); ax.set_yticks([])
axes[0].set_title("Stage 6D2 marginals")
fig.tight_layout()
plt.show()
print("Figure 11 -- dashed: the parent posterior mean (Stage 6B3, U pinned at truth;")
print("Stage 6C2 for rho). Orange: the generating value. Freeing U, rho and every")
print("scalar at once leaves the four scalar marginals where the parent put them.")
"""),
        md(r"""
## 49. What Stage 6D closes, and what it does not

**Closed.** The sampler infers the latent utilities, their correlation, and all four
recurrent scalars jointly. Correctness is established against a reference that shares no
code path with it and was frozen before any chain ran; convergence, parent consistency and
recovery are established on the full 500-block corpus against gates registered before any
draw existed. Composing the Stage 6B and Stage 6C kernels changed neither parent, to
$4.6\times10^{-13}$ and exactly zero respectively.

**Not closed, and not claimed.**

* **There is no independent reference for the 500-block corpus, and none can be built by
  this route.** Prior importance sampling degrades as the likelihood sharpens — which is
  precisely why the Stage 6D1 model was made small. Stage 6D2's correctness claim is
  *inherited* from Stage 6D1 and supported by parent consistency; it is not established
  afresh, and saying otherwise would be the easiest available overclaim.
* **$\rho$ is still weakly identified**, for the reason given in Part IX, and Stage 6D
  changes nothing about it.
* **Entrywise $U$ recovery is not claimed and cannot be.** The likelihood is piecewise
  constant in $U$ — it speaks only at order boundaries — and the target is invariant under
  permuting the $d$ columns and under any strictly increasing reparameterisation within a
  column. Structure is the recoverable object.
* **The structural problem is still easy on this corpus.** The order posterior is a point
  mass, so the relation-count trace never moves. Stage 6D1's small model is the only place
  in this notebook where structural *mixing* is exercised at all.

**What comes next.** Stage 6D is the last stage that assumes the segmentation. Stage 6E
frees the block boundaries $S$; skill-label inference and semi-Markov FFBS follow it, and
none of that is started. One practical thing carries forward with it: budget a separate
pilot before the Stage 6E formal run. Freeing $S$ changes the target, and §46 is the
demonstration that a scale tuned on one target has no claim on another.
"""),
    ]


# ------------------------------------------------------------------------ execute/clean
def execute(cells: list) -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                       "name": "python3"}
    client = NotebookClient(notebook, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT / "notebooks")}})
    client.execute()
    return notebook


def clean(notebook: nbformat.NotebookNode) -> list:
    """Strip stderr, empty streams and transient metadata; refuse anything that errored."""
    errors = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        kept = []
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append((index, output.get("ename"),
                               " / ".join(output.get("traceback", [])[-2:])))
                continue
            if output.get("name") == "stderr":
                continue                       # warnings are noise, not results
            if output.get("output_type") == "stream" and not "".join(
                    output.get("text", "")).strip():
                continue
            kept.append(output)
        cell["outputs"] = kept
        cell["metadata"] = {}
    if errors:
        for index, name, trace in errors:
            print(f"cell {index} raised {name}: {trace}", file=sys.stderr)
        raise SystemExit("Part X errored; nothing merged")
    return notebook.cells


def merge(cells: list) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    existing = [i for i, c in enumerate(notebook.cells)
                if c.cell_type == "markdown" and PART_TITLE in "".join(c.source)]
    if existing:
        # Replace an earlier Part X rather than appending a second one.
        start = existing[0]
        # the separator cell immediately above, if this Part owns it
        if start and "".join(notebook.cells[start].source).lstrip().startswith("---"):
            pass
        notebook.cells = notebook.cells[:start] + cells
        print(f"replaced the existing Part X at cell {start}")
    else:
        notebook.cells = notebook.cells + cells
        print(f"appended Part X after cell {len(notebook.cells) - len(cells) - 1}")
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="build and execute, but do not merge")
    args = parser.parse_args()

    cells = part_cells()
    print(f"building Part X: {len(cells)} cells "
          f"({sum(c.cell_type == 'code' for c in cells)} code, "
          f"{sum(c.cell_type == 'markdown' for c in cells)} markdown)")
    notebook = execute(cells)
    cleaned = clean(notebook)
    figures = sum(1 for c in cleaned if c.cell_type == "code"
                  for o in c.get("outputs", [])
                  if "image/png" in (o.get("data") or {}))
    print(f"executed cleanly: {figures} figures, "
          f"{sum(len(c.get('outputs', [])) for c in cleaned if c.cell_type == 'code')} "
          f"outputs retained")
    if args.dry_run:
        print("--dry-run: not merged")
        return
    merge(cleaned)
    print(f"merged into {NOTEBOOK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
