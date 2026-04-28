"""
Panels (a) moisture heatmap, (b) temperature heatmap - radial x time.
Both shown as 2D (r_nodes x t_steps) images.
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(direction="out", length=4, width=0.7)
    ax.grid(True, linewidth=0.35, color="#cccccc", alpha=0.7)
    ax.set_axisbelow(True)


def plot_field_heatmaps(result, out_path="assets/figures/panel_ab_fields.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mesh = result["mesh"]
    r = mesh["r"] * 1000  # mm
    ins_sl = mesh["ins_slice"]
    r_ins = r[ins_sl.start : ins_sl.stop]
    t = result["t_yr"]
    theta = result["theta"][:, ins_sl.start : ins_sl.stop]
    T = result["T"][:, ins_sl.start : ins_sl.stop] - 273.15  # C

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    _ax(ax)
    im = ax.pcolormesh(t, r_ins, theta.T, cmap="Blues", vmin=0, vmax=0.8, shading="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\theta_w$ [vol fraction]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_xlabel("Time [yr]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_ylabel("Radial position [mm]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_title(
        "(a)  Moisture Field $\\theta_w(r, t)$ — Insulation Annulus",
        fontsize=10,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        loc="left",
    )

    ax = axes[1]
    _ax(ax)
    im2 = ax.pcolormesh(t, r_ins, T.T, cmap="RdYlGn_r", shading="auto")
    cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label(r"$T$ [°C]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_xlabel("Time [yr]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_ylabel("Radial position [mm]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_title(
        "(b)  Temperature Field $T(r, t)$ — Insulation Annulus",
        fontsize=10,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        loc="left",
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path
