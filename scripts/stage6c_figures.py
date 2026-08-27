"""Stage 6C — figures for the latent-poset runs.

    PYTHONPATH=src python scripts/stage6c_figures.py

Four figures, each answering a question the numbers alone answer awkwardly:

* traces      — did the dispersed starts converge to one place, and does anything move?
* structure   — does the posterior relation matrix match the reference, and the truth?
* marginals   — does the rho (and beta) marginal sit on its frozen reference?
* identifiability — how flat is `pi_rho(P_true)` in rho? This is the figure that explains
  why rho is weakly identified even when the sampler is exactly right, and it is drawn
  from the prior cell masses alone, with no chain involved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.latent_poset import precedence_from_u          # noqa: E402
from hpop.mcmc_original.stage6c_exact_reference import build_catalogue  # noqa: E402
from hpop.mcmc_original.stage6c_frozen import load_stage6c_dataset      # noqa: E402

RESULTS = ROOT / "results" / "mcmc_original"
CHAIN_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#9b59b6")
REFERENCE_COLOUR = "#333333"


def _load(stage: str):
    run = ("stage6c1_u_rho_full_seed0" if stage == "6c1"
           else "stage6c2_u_rho_beta_full_seed0")
    ref = ("stage6c1_u_rho_reference" if stage == "6c1"
           else "stage6c2_u_rho_beta_reference")
    with np.load(RESULTS / run / "chains.npz") as z:
        chains = {k: z[k] for k in z.files}
    with np.load(RESULTS / ref / "exact_reference.npz") as z:
        reference = {k: z[k] for k in z.files}
    return chains, reference


def figure_traces(stage: str, chains: dict, out: Path) -> None:
    rows = 3 if stage == "6c2" else 2
    fig, axes = plt.subplots(rows, 1, figsize=(11, 2.6 * rows), sharex=True)
    axes = np.atleast_1d(axes)

    for c in range(chains["relation_counts"].shape[0]):
        axes[0].plot(chains["relation_counts"][c], lw=0.7, alpha=0.8,
                     color=CHAIN_COLOURS[c % 4], label=f"chain {c}")
        axes[1].plot(chains["rho"][c], lw=0.7, alpha=0.8, color=CHAIN_COLOURS[c % 4])
        if stage == "6c2":
            axes[2].plot(chains["beta"][c], lw=0.7, alpha=0.8,
                         color=CHAIN_COLOURS[c % 4])
    axes[0].set_ylabel("relations in h(U)")
    axes[0].legend(ncol=4, fontsize=8, frameon=False)
    axes[1].set_ylabel(r"$\rho$")
    if stage == "6c2":
        axes[2].set_ylabel(r"$\beta$")
    axes[-1].set_xlabel("retained draw")
    fig.suptitle(f"Stage {stage.upper()} — traces from four dispersed starts", y=0.99)
    fig.tight_layout()
    fig.savefig(out / f"{stage}_traces.png", dpi=140)
    plt.close(fig)


def figure_structure(stage: str, chains: dict, reference: dict, out: Path) -> None:
    catalogue = build_catalogue(5, 2)
    frozen = load_stage6c_dataset()
    m = catalogue.m
    ids = chains["poset_ids"].ravel()
    mcmc = catalogue.closures[ids].reshape(ids.size, -1).mean(axis=0).reshape(m, m)
    ref = np.asarray(reference["relation_marginal"]).reshape(m, m)
    truth = precedence_from_u(frozen.u_true).astype(float)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, matrix, title in zip(
            axes, (truth, ref, mcmc, np.abs(mcmc - ref)),
            ("truth  h(U_TRUE)", "reference  P(i > j)", "MCMC  P(i > j)",
             "|MCMC - reference|")):
        image = ax.imshow(matrix, vmin=0, vmax=1 if "|" not in title else 0.01,
                          cmap="viridis" if "|" not in title else "magma")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(m)); ax.set_yticks(range(m))
        ax.set_xlabel("j"); ax.set_ylabel("i")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(f"Stage {stage.upper()} — closure relation marginals", y=1.02)
    fig.tight_layout()
    fig.savefig(out / f"{stage}_structure.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def figure_marginals(stage: str, chains: dict, reference: dict, out: Path) -> None:
    names = ["rho"] + (["beta"] if stage == "6c2" else [])
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 3.6))
    axes = np.atleast_1d(axes)
    frozen = load_stage6c_dataset()

    for ax, name in zip(axes, names):
        pooled = chains[name].ravel()
        grid = reference[f"{name}_grid" if f"{name}_grid" in reference
                         else "rho_grid" if name == "rho" else "beta_grid"]
        density = reference[f"{name}_marginal_density"]
        ax.hist(pooled, bins=60, density=True, alpha=0.55, color=CHAIN_COLOURS[0],
                label="MCMC (pooled)")
        ax.plot(grid, density, color=REFERENCE_COLOUR, lw=2, label="exact reference")
        if name == "beta":
            ax.axvline(frozen.truth["beta"], color="#c0392b", ls="--", lw=1.4,
                       label=r"$\beta_{\mathrm{true}}$")
        ax.set_xlabel(rf"$\{name}$" if name in ("rho", "beta") else name)
        ax.set_ylabel("density")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Stage {stage.upper()} — scalar marginals against the frozen reference",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out / f"{stage}_marginals.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def figure_identifiability(out: Path) -> None:
    """Why rho is weakly identified: the prior cell mass of the true poset is nearly flat.

    Drawn from the prior cell masses alone — no data, no likelihood, no chain. Because the
    poset posterior is effectively a point mass at `P_true`, `p(rho | Y)` is proportional
    to `p(rho) * pi_rho(P_true)`, so the shape of the curve below *is* the rho posterior
    up to normalisation.
    """
    path = RESULTS / "stage6c1_u_rho_reference" / "exact_reference.npz"
    with np.load(path) as z:
        rho_grid, masses = z["rho_grid"], z["cell_masses"]
        standard_error = z["cell_mass_se"]
        rho_density = z["rho_marginal_density"]

    catalogue = build_catalogue(5, 2)
    frozen = load_stage6c_dataset()
    true_index = catalogue.index_of(precedence_from_u(frozen.u_true))
    pi_true = masses[:, true_index]
    pi_se = standard_error[:, true_index]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(rho_grid, pi_true, color=CHAIN_COLOURS[0], lw=2)
    axes[0].fill_between(rho_grid, pi_true - 2 * pi_se, pi_true + 2 * pi_se,
                         color=CHAIN_COLOURS[0], alpha=0.25, label=r"$\pm 2$ SE")
    axes[0].set_xlabel(r"$\rho$")
    axes[0].set_ylabel(r"$\pi_\rho(P_{\mathrm{true}})$")
    axes[0].set_title("prior cell mass of the true poset", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].plot(rho_grid, rho_density, color=REFERENCE_COLOUR, lw=2)
    axes[1].set_xlabel(r"$\rho$")
    axes[1].set_ylabel("density")
    axes[1].set_title(r"exact $p(\rho \mid Y)$", fontsize=10)

    fig.suptitle("Stage 6C — rho identifiability comes from the prior cell mass alone",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out / "rho_identifiability.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=RESULTS / "stage6c_complete" / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    made = []
    for stage in ("6c1", "6c2"):
        try:
            chains, reference = _load(stage)
        except FileNotFoundError as error:
            print(f"[{stage}] skipped: {error}")
            continue
        figure_traces(stage, chains, args.out)
        figure_structure(stage, chains, reference, args.out)
        figure_marginals(stage, chains, reference, args.out)
        made += [f"{stage}_traces.png", f"{stage}_structure.png",
                 f"{stage}_marginals.png"]
    figure_identifiability(args.out)
    made.append("rho_identifiability.png")
    print(json.dumps({"figures": made, "directory": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
