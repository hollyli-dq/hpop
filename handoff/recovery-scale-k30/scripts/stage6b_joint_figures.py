"""Stage 6B2 / 6B3 — trace plots, marginal overlays, and pairwise posterior plots.

    PYTHONPATH=src python scripts/stage6b_joint_figures.py \
        --result-dir results/mcmc_original/stage6b2_joint3_full_seed0

Three figures per stage, each answering a different question:

* traces — did the chains mix, and did the dispersed starts converge to one place?
* marginals — does each one-dimensional posterior sit on its reference?
* pairs — does the *dependence* match, which four marginals cannot establish?

The reference contours are computed by integrating the frozen grid over the remaining
coordinates, so the pair plots compare the sampler against the same object the gates use.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hpop.mcmc_original.stage6b_joint_reference import normalise_log_density   # noqa: E402

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8f8e88"
CHAIN_COLOURS = [INK, S1, S2, S3]

plt.rcParams.update({
    "figure.dpi": 130, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK3, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.2,
})


def load(result_dir: Path):
    config = json.loads((result_dir / "config.json").read_text())
    active = tuple(config["active"])
    summary = json.loads((result_dir / "reference_summary.json").read_text())["summary"]
    stored = np.load(result_dir / "chains.npz")
    chains = {n: stored[f"draws_{n}"] for n in active}
    payload = np.load(result_dir / "joint_reference.npz")
    axes_z = {n: payload[f"axis_z_{n}"] for n in active}
    axes_value = {n: payload[f"axis_value_{n}"] for n in active}
    density, _ = normalise_log_density(
        active, axes_z, np.asarray(payload["log_density_z"], dtype=float))
    return config, active, summary, chains, axes_z, axes_value, density


def marginal_density(active, axes_z, axes_value, density, index):
    """Reference marginal on the PARAMETER scale, by integrating out the other axes."""
    marginal = density
    for axis in range(len(active) - 1, -1, -1):
        if axis != index:
            marginal = np.trapezoid(marginal, axes_z[active[axis]], axis=axis)
    z = axes_z[active[index]]
    marginal = marginal / np.trapezoid(marginal, z)
    value = axes_value[active[index]]
    # p(value) = p(z) * |dz/dvalue|; for z = log(value) that is 1/value
    jacobian = np.gradient(z, value)
    return value, marginal * jacobian


def pair_density(active, axes_z, axes_value, density, i, j):
    marginal = density
    for axis in range(len(active) - 1, -1, -1):
        if axis not in (i, j):
            marginal = np.trapezoid(marginal, axes_z[active[axis]], axis=axis)
    # after integrating, the remaining axes keep their relative order
    if i > j:
        marginal = marginal.T
        i, j = j, i
    return axes_value[active[i]], axes_value[active[j]], marginal


def figure_traces(path, active, chains, summary, config):
    n = len(active)
    fig, axes = plt.subplots(n, 1, figsize=(9.5, 1.9 * n), sharex=True)
    axes = np.atleast_1d(axes)
    show = min(1500, chains[active[0]].shape[1])
    for ax, name in zip(axes, active):
        for c in range(chains[name].shape[0]):
            ax.plot(chains[name][c, :show], color=CHAIN_COLOURS[c % 4], lw=0.5,
                    alpha=0.8, label=f"chain {c}" if name == active[0] else None)
        ax.axhline(summary["mean"][name], color=S2, lw=1.2, ls="--")
        ax.set_ylabel(name)
    axes[0].legend(ncol=4, loc="upper right")
    axes[-1].set_xlabel(f"kept draw (first {show} of {chains[active[0]].shape[1]}, "
                        f"post burn-in, thinned)")
    fig.suptitle(f"Stage 6{config['stage'].upper()} — chains from dispersed starts\n"
                 "dashed: reference posterior mean", x=0.01, y=0.995, ha="left",
                 va="top", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_marginals(path, active, chains, summary, axes_z, axes_value, density, config):
    n = len(active)
    cols = min(n, 2)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for k, name in enumerate(active):
        ax = axes[k]
        pooled = chains[name].ravel()
        ax.hist(pooled, bins=70, density=True, color=S1, alpha=0.30, edgecolor="none",
                label="MCMC")
        value, marginal = marginal_density(active, axes_z, axes_value, density, k)
        ax.plot(value, marginal, color=INK, lw=1.5, label="reference")
        ax.axvline(config["frozen_config"]["truth"][name], color=S2, lw=1.2, ls="--",
                   label="truth")
        lo, hi = summary["q025"][name], summary["q975"][name]
        pad = 2.5 * (hi - lo)
        ax.set_xlim(lo - pad / 3, hi + pad / 3)
        ax.set_yticks([])
        ax.set_title(name)
    for spare in axes[n:]:
        spare.axis("off")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Stage 6{config['stage'].upper()} — marginals against the independent "
                 f"reference", x=0.01, y=0.995, ha="left", va="top", fontsize=10,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_pairs(path, active, chains, axes_z, axes_value, density, config):
    pairs = [(i, j) for i in range(len(active)) for j in range(i + 1, len(active))]
    cols = min(3, len(pairs))
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    step = max(1, chains[active[0]].size // 3000)
    for k, (i, j) in enumerate(pairs):
        ax = axes[k]
        for c in range(chains[active[i]].shape[0]):
            ax.scatter(chains[active[i]][c][::step], chains[active[j]][c][::step],
                       s=2.0, alpha=0.25, color=CHAIN_COLOURS[c % 4], linewidths=0,
                       label=f"chain {c}" if k == 0 else None)
        x, y, joint = pair_density(active, axes_z, axes_value, density, i, j)
        levels = np.quantile(joint, [0.80, 0.95, 0.995])
        ax.contour(x, y, joint.T, levels=levels, colors=[INK], linewidths=1.0)
        ax.set_xlabel(active[i]); ax.set_ylabel(active[j])
    for spare in axes[len(pairs):]:
        spare.axis("off")
    if len(pairs) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, loc="upper right", markerscale=4)
    fig.suptitle(f"Stage 6{config['stage'].upper()} — pairwise posterior, chains coloured "
                 f"separately\ncontours: the frozen reference integrated over the other "
                 f"coordinates", x=0.01, y=0.995, ha="left", va="top", fontsize=10,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-dir", type=Path, required=True)
    a = ap.parse_args()

    config, active, summary, chains, axes_z, axes_value, density = load(a.result_dir)
    figures = a.result_dir / "figures"
    figures.mkdir(exist_ok=True)

    figure_traces(figures / "traces.png", active, chains, summary, config)
    figure_marginals(figures / "marginals.png", active, chains, summary, axes_z,
                     axes_value, density, config)
    figure_pairs(figures / "pairs.png", active, chains, axes_z, axes_value, density,
                 config)
    for name in ("traces.png", "marginals.png", "pairs.png"):
        print(f"  {figures / name}  ({(figures / name).stat().st_size / 1000:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
