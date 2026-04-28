"""Panel (c) — MC wall loss CDF + histogram."""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.constants import COLOR_NAVY, COLOR_RED, COLOR_STEEL, MAX_WL_FRAC, PIPE_WT


def _ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(direction="out", length=4, width=0.7)
    ax.grid(True, linewidth=0.35, color="#cccccc", alpha=0.7)
    ax.set_axisbelow(True)


def plot_mc_distribution(mc_result, out_path="assets/figures/panel_c_mc_distribution.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wl = mc_result["wall_loss"]
    limit = PIPE_WT * 1000 * MAX_WL_FRAC
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax = axes[0]
    _ax(ax)
    sorted_wl = np.sort(wl)
    cdf = np.arange(1, len(sorted_wl) + 1) / len(sorted_wl)
    ax.step(sorted_wl, cdf, color=COLOR_NAVY, linewidth=1.2, where="post", label="CDF")
    ax.axvline(
        limit, color=COLOR_RED, linewidth=0.9, linestyle=":", label=f"20% WT = {limit:.2f} mm"
    )
    for pct, lbl in [(50, "P50"), (90, "P90"), (95, "P95")]:
        v = np.percentile(wl, pct)
        ax.axvline(v, color=COLOR_STEEL, linewidth=0.6, alpha=0.5)
        ax.text(
            v + 0.002,
            pct / 100 - 0.07,
            f"{lbl}\n{v:.3f}",
            fontsize=7.5,
            color=COLOR_STEEL,
            va="top",
            fontfamily="DejaVu Sans",
        )
    ax.set_xlabel("Wall loss at assessment horizon [mm]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_ylabel("Cumulative probability [—]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_title(
        "(c)  Wall Loss CDF — MC LHS $N$ = 10,000",
        fontsize=10,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        loc="left",
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    _ax(ax)
    bins = np.linspace(wl.min(), wl.max(), 50)
    ax.hist(wl, bins=bins, color=COLOR_STEEL, alpha=0.75, label="Wall loss distribution")
    ax.axvline(limit, color=COLOR_RED, linewidth=1.0, linestyle=":", label="20% WT limit")
    ax.set_xlabel("Wall loss [mm]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_ylabel("Count [—]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_title(
        "(c')  Wall Loss Histogram",
        fontsize=10,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        loc="left",
    )
    ax.legend(fontsize=8, frameon=False)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path
