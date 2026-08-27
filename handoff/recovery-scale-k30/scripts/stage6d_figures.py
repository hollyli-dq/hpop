"""Stage 6D — figures for the joint oracle-block runs.

    PYTHONPATH=src python scripts/stage6d_figures.py

Seven figures, each answering a question the tables answer awkwardly:

* `6d1_traces`        did four dispersed starts reach one place, and does everything move?
* `6d1_structure`     does the induced-`H` posterior sit on the frozen QMC reference?
* `6d1_marginals`     do the five continuous marginals sit on the reference's?
* `6d2_traces`        the same question on the full corpus, where `H` is a point mass
* `6d2_marginals`     do the scalars reproduce Stage 6B3's, which had `U` at the truth?
* `6d2_structure`     is the recovered relation matrix the generating one?
* `scale_retuning`    the §G story: what mis-scaled proposals cost, and what fixed them

The reference is drawn as an importance-weighted cloud, never as an unweighted one: the
QMC reference carries weights and plotting it without them would draw the prior.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.latent_poset import precedence_from_u            # noqa: E402
from hpop.mcmc_original.stage6d_frozen import (                          # noqa: E402
    SCALAR_ORDER, load_stage6d_dataset,
)

RESULTS = ROOT / "results" / "mcmc_original"
FIGURES = RESULTS / "stage6d_complete" / "figures"
CHAIN_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#9b59b6")
REFERENCE_COLOUR = "#333333"
TRUTH_COLOUR = "#cc2222"
NAMES = ("rho",) + SCALAR_ORDER
LABELS = {"rho": r"$\rho$", "beta": r"$\beta$", "omega": r"$\omega$",
          "lambda_rep": r"$\lambda_{\mathrm{rep}}$",
          "lambda_back": r"$\lambda_{\mathrm{back}}$"}


def load_npz(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def weighted_histogram(axis, values, weights, bins, colour, label):
    counts, edges = np.histogram(values, bins=bins, weights=weights, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    axis.plot(centres, counts, color=colour, lw=1.6, label=label)


# ------------------------------------------------------------------------------- 6D1
def figure_6d1_traces(chains: dict, out: Path) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(11, 12), sharex=True)
    axes[0].set_ylabel("relations")
    for c in range(chains["relation_counts"].shape[0]):
        axes[0].plot(chains["relation_counts"][c], lw=0.5, alpha=0.75,
                     color=CHAIN_COLOURS[c % 4], label=f"chain {c}")
        for row, name in enumerate(NAMES, start=1):
            axes[row].plot(chains[name][c], lw=0.4, alpha=0.7,
                           color=CHAIN_COLOURS[c % 4])
    for row, name in enumerate(NAMES, start=1):
        axes[row].set_ylabel(LABELS[name])
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("retained draw")
    fig.suptitle("Stage 6D1 — four dispersed starts, all six coordinates free", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_6d1_structure(chains: dict, reference: dict, structural: dict,
                         out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    mcmc = np.array(structural["h_probability_aligned_mcmc"])
    ref = np.array(structural["h_probability_aligned_reference"])
    # Rank by reference mass and drop the tail neither distribution reaches: 19 labelled
    # orders on 3 elements, most of them empty, would otherwise be 14 blank columns.
    order = np.argsort(-ref)
    keep = np.maximum(ref[order], mcmc[order]) > 1e-3
    order = order[keep]
    index = np.arange(order.size)
    axes[0].bar(index - 0.2, ref[order], width=0.4, color=REFERENCE_COLOUR,
                label="QMC reference")
    axes[0].bar(index + 0.2, mcmc[order], width=0.4, color=CHAIN_COLOURS[0],
                label="MCMC")
    axes[0].set_xlabel("induced order $H = h(U)$, reference-ranked")
    axes[0].set_ylabel("posterior probability")
    axes[0].set_title(f"TV = {structural['h_total_variation']:.5f}")
    axes[0].legend(fontsize=8)

    m = chains["u_draws"].shape[2]
    mcmc_relation = np.array(structural["relation_marginal"]).reshape(m, m)
    ref_relation = reference["pooled_relation_marginal"].reshape(m, m)
    for axis, matrix, title in ((axes[1], ref_relation, "reference $P(i > j)$"),
                                (axes[2], mcmc_relation, "MCMC $P(i > j)$")):
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_title(title)
        axis.set_xticks(range(m))
        axis.set_yticks(range(m))
        for i in range(m):
            for j in range(m):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                          fontsize=8, color="w" if matrix[i, j] < 0.6 else "k")
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.suptitle("Stage 6D1 — induced structure against the frozen reference")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_6d1_marginals(chains: dict, reference: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
    weights = reference["pooled_weights"].astype(float)
    for axis, name in zip(axes, NAMES):
        pooled = chains[name].ravel()
        low = min(pooled.min(), float(np.min(reference[f"pooled_{name}"])))
        high = max(pooled.max(), float(np.max(reference[f"pooled_{name}"])))
        # the reference is importance weighted; an unweighted histogram draws the prior
        span = np.linspace(low, high, 60)
        weighted_histogram(axis, reference[f"pooled_{name}"].astype(float), weights,
                           span, REFERENCE_COLOUR, "QMC reference")
        axis.hist(pooled, bins=span, density=True, alpha=0.45,
                  color=CHAIN_COLOURS[0], label="MCMC")
        axis.set_xlabel(LABELS[name])
        axis.set_yticks([])
    axes[0].legend(fontsize=8)
    fig.suptitle("Stage 6D1 — continuous marginals against the weighted QMC reference")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------------------- 6D2
def figure_6d2_traces(chains: dict, out: Path) -> None:
    fig, axes = plt.subplots(7, 1, figsize=(11, 13.5), sharex=True)
    for c in range(chains["relation_counts"].shape[0]):
        axes[0].plot(chains["relation_counts"][c], lw=0.6, alpha=0.8,
                     color=CHAIN_COLOURS[c % 4], label=f"chain {c}")
        for row, name in enumerate(NAMES, start=1):
            axes[row].plot(chains[name][c], lw=0.4, alpha=0.7,
                           color=CHAIN_COLOURS[c % 4])
        axes[6].plot(chains["log_target"][c], lw=0.4, alpha=0.7,
                     color=CHAIN_COLOURS[c % 4])
    axes[0].set_ylabel("relations")
    axes[0].set_ylim(-0.5, 10.5)
    for row, name in enumerate(NAMES, start=1):
        axes[row].set_ylabel(LABELS[name])
    axes[6].set_ylabel("log target")
    axes[0].legend(ncol=4, fontsize=8, loc="lower right")
    axes[-1].set_xlabel("retained draw")
    fig.suptitle("Stage 6D2 — 500 oracle blocks; the induced order is a point mass",
                 y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_6d2_marginals(chains: dict, consistency: dict, frozen, out: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
    for axis, name in zip(axes, NAMES):
        pooled = chains[name].ravel()
        axis.hist(pooled, bins=60, density=True, alpha=0.55, color=CHAIN_COLOURS[0],
                  label="Stage 6D2")
        if name in SCALAR_ORDER:
            parent = consistency["vs_stage6b3"][name]
            axis.axvline(parent["stage6b3_mean"], color=REFERENCE_COLOUR, lw=1.4,
                         ls="--", label="Stage 6B3 mean ($U$ at truth)")
            axis.axvline(float(frozen.truth[name]), color=TRUTH_COLOUR, lw=1.2,
                         label="truth")
        else:
            parent = consistency["vs_stage6c2"]["rho"]
            axis.axvline(parent["stage6c2_mean"], color=REFERENCE_COLOUR, lw=1.4,
                         ls="--", label="Stage 6C2 mean")
        axis.set_xlabel(LABELS[name])
        axis.set_yticks([])
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.suptitle("Stage 6D2 — marginals against the parent posteriors and the truth")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_6d2_structure(chains: dict, recovery: dict, frozen, out: Path) -> None:
    m = frozen.u_true.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    true_closure = precedence_from_u(frozen.u_true).astype(float)
    posterior = np.array(recovery["structural"]["posterior_mean_relation_matrix"])
    for axis, matrix, title in ((axes[0], true_closure, "generating $h(U_{true})$"),
                                (axes[1], posterior, "posterior $P(i > j)$")):
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_title(title)
        axis.set_xticks(range(m))
        axis.set_yticks(range(m))
        for i in range(m):
            for j in range(m):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                          fontsize=8, color="w" if matrix[i, j] < 0.6 else "k")
        fig.colorbar(image, ax=axis, fraction=0.046)

    difference = np.abs(posterior - true_closure)
    image = axes[2].imshow(difference, vmin=0.0, vmax=1.0, cmap="magma")
    axes[2].set_title(f"|difference|, max {difference.max():.3g}")
    axes[2].set_xticks(range(m))
    axes[2].set_yticks(range(m))
    fig.colorbar(image, ax=axes[2], fraction=0.046)
    fig.suptitle("Stage 6D2 — structural recovery against the generating order")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------- the §G story
def figure_scale_retuning(history: dict, pilot: dict, out: Path) -> None:
    """What under-scaled proposals cost on the small model, and what 6D2 chose instead."""
    gain = history["efficiency_gain"]
    names = [n for n in ("beta", "omega", "lambda_rep", "lambda_back") if n in gain]
    before = [gain[n]["bulk_ess_before"] for n in names]
    after = [gain[n]["bulk_ess_after"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    index = np.arange(len(names))
    axes[0].bar(index - 0.2, before, width=0.4, color="#999999",
                label="registered Stage 6B scales")
    axes[0].bar(index + 0.2, after, width=0.4, color=CHAIN_COLOURS[0],
                label="Stage 6D1 pilot scales")
    axes[0].set_yscale("log")
    axes[0].set_xticks(index)
    axes[0].set_xticklabels(names, rotation=15)
    axes[0].set_ylabel("bulk ESS (log scale)")
    axes[0].set_title("Stage 6D1: a 16-32x under-scaled proposal costs 22-61x in ESS")
    axes[0].legend(fontsize=8)
    for i, (b, a) in enumerate(zip(before, after)):
        axes[0].text(i + 0.2, a * 1.1, f"x{a / b:.0f}", ha="center", fontsize=8)

    coordinates = list(pilot["selected_multipliers"])
    multipliers = [float(pilot["decisions"][c]["selected"]["multiplier"])
                   for c in coordinates]
    acceptance = [pilot["decisions"][c]["selected"]["median_acceptance"]
                  for c in coordinates]
    axes[1].bar(np.arange(len(coordinates)), multipliers, color=CHAIN_COLOURS[2])
    axes[1].axhline(1.0, color=REFERENCE_COLOUR, lw=1.0, ls="--",
                    label="registered Stage 6B/6C scale")
    axes[1].set_yscale("log", base=2)
    axes[1].set_xticks(np.arange(len(coordinates)))
    axes[1].set_xticklabels(coordinates, rotation=15)
    axes[1].set_ylabel("selected multiplier (log$_2$)")
    axes[1].set_title("Stage 6D2 pilot: the full-data posterior wants different steps")
    axes[1].legend(fontsize=8)
    for i, (mult, acc) in enumerate(zip(multipliers, acceptance)):
        axes[1].text(i, mult * 1.08, f"{acc:.2f}", ha="center", fontsize=8)
    fig.suptitle("Proposal scales are a property of the corpus, not of the kernel")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage6d2-dir", default="stage6d2_oracle_joint_full_seed0")
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    frozen = load_stage6d_dataset()
    written = []

    run_6d1 = RESULTS / "stage6d1_joint_mcmc"
    if (run_6d1 / "chains.npz").exists():
        chains = load_npz(run_6d1 / "chains.npz")
        reference = load_npz(RESULTS / "stage6d1_joint_reference" / "qmc_replicates.npz")
        structural = read_json(run_6d1 / "structural_diagnostics.json")
        figure_6d1_traces(chains, FIGURES / "6d1_traces.png")
        figure_6d1_structure(chains, reference, structural,
                             FIGURES / "6d1_structure.png")
        figure_6d1_marginals(chains, reference, FIGURES / "6d1_marginals.png")
        written += ["6d1_traces.png", "6d1_structure.png", "6d1_marginals.png"]

    run_6d2 = RESULTS / args.stage6d2_dir
    if (run_6d2 / "chains.npz").exists():
        chains = load_npz(run_6d2 / "chains.npz")
        figure_6d2_traces(chains, FIGURES / "6d2_traces.png")
        written.append("6d2_traces.png")
        consistency = read_json(run_6d2 / "parent_consistency.json")
        if consistency:
            figure_6d2_marginals(chains, consistency, frozen,
                                 FIGURES / "6d2_marginals.png")
            written.append("6d2_marginals.png")
        recovery = read_json(run_6d2 / "recovery_results.json")
        if recovery:
            figure_6d2_structure(chains, recovery, frozen,
                                 FIGURES / "6d2_structure.png")
            written.append("6d2_structure.png")

    history = read_json(run_6d1 / "continuation_history.json")
    pilot = read_json(RESULTS / "stage6d2_pilot" / "pilot_results.json")
    if history and pilot:
        figure_scale_retuning(history, pilot, FIGURES / "scale_retuning.png")
        written.append("scale_retuning.png")

    print(f"wrote {len(written)} figures to {FIGURES}")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
