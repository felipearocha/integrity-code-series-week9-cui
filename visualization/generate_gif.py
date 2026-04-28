"""
GIF: Animated moisture front evolution in insulation annulus.
24 frames, ACCELERATED TIMESCALE.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from src.constants import COLOR_CHARCOAL, COLOR_NAVY, COLOR_RED


def generate_gif(result, out_path="assets/animations/cui_moisture_front.gif", n_frames=24, fps=6):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mesh = result["mesh"]
    r = mesh["r"] * 1000  # mm
    ins_sl = mesh["ins_slice"]
    r_ins = r[ins_sl.start : ins_sl.stop]
    t_yr = result["t_yr"]
    theta = result["theta"][:, ins_sl.start : ins_sl.stop]
    wl = result["wl"]
    n_steps = len(t_yr)
    frame_idx = np.linspace(0, n_steps - 1, n_frames, dtype=int)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_linewidth(0.7)
        ax.tick_params(direction="out", length=3, width=0.7)
        ax.grid(True, linewidth=0.3, color="#cccccc", alpha=0.6)

    from src.constants import THETA_CRIT

    (line_moist,) = axes[0].plot([], [], color=COLOR_NAVY, linewidth=1.5)
    fill_wet = [axes[0].fill_between([], [], alpha=0.0)]
    (line_wl,) = axes[1].plot([], [], color=COLOR_RED, linewidth=1.5)
    time_txt = axes[0].text(
        0.02,
        0.92,
        "",
        transform=axes[0].transAxes,
        fontsize=9,
        color=COLOR_CHARCOAL,
        fontfamily="DejaVu Sans",
    )
    axes[0].set_xlim(r_ins[0], r_ins[-1])
    axes[0].set_ylim(0, 0.85)
    axes[0].set_ylabel("Moisture content theta_w [vol frac]", fontsize=9, fontfamily="DejaVu Sans")
    axes[0].set_title(
        "CUI Moisture Front — Insulation Annulus [ACCELERATED TIMESCALE]",
        fontsize=10,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        loc="left",
    )
    axes[0].axhline(
        THETA_CRIT,
        color=COLOR_RED,
        linewidth=0.8,
        linestyle=":",
        label=f"theta_crit = {THETA_CRIT}",
    )
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_xlim(t_yr[0], t_yr[-1])
    axes[1].set_ylim(0, max(wl.max() * 1.2, 0.01))
    axes[1].set_xlabel("Time [yr]", fontsize=9, fontfamily="DejaVu Sans")
    axes[1].set_ylabel("Wall loss [mm]", fontsize=9, fontfamily="DejaVu Sans")
    axes[1].set_title(
        "Cumulative Wall Loss at Pipe Surface",
        fontsize=10,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        loc="left",
    )
    fig.text(
        0.5,
        0.01,
        "[ACCELERATED: 1 frame = ~0.4 yr  |  Total = 10 yr design life]",
        ha="center",
        fontsize=7.5,
        color="#888888",
        fontfamily="DejaVu Sans",
    )

    def init():
        line_moist.set_data([], [])
        line_wl.set_data([], [])
        time_txt.set_text("")
        return line_moist, line_wl, time_txt

    def update(frame_k):
        k = frame_idx[frame_k]
        line_moist.set_data(r_ins, theta[k])
        fill_wet[0].remove()
        fill_wet[0] = axes[0].fill_between(
            r_ins, theta[k], THETA_CRIT, where=theta[k] > THETA_CRIT, alpha=0.20, color=COLOR_NAVY
        )
        line_wl.set_data(t_yr[: k + 1], wl[: k + 1])
        time_txt.set_text(f"t = {t_yr[k]:.1f} yr")
        return line_moist, line_wl, time_txt

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init, blit=False, interval=1000 // fps
    )
    writer = animation.PillowWriter(fps=fps)
    ani.save(out_path, writer=writer, dpi=110)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path
