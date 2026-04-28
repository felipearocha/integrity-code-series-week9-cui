"""
Panels (d) sensitivity tornado, (e) surrogate parity,
(f) iso-risk contour, (g) FAD, (h) inverse solution.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src.constants import COLOR_CHARCOAL, COLOR_NAVY, COLOR_RED, COLOR_STEEL, COLOR_TEAL, P_OP_BAR


def _ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(direction="out", length=4, width=0.7)
    ax.grid(True, linewidth=0.35, color="#cccccc", alpha=0.7)
    ax.set_axisbelow(True)


def plot_sensitivity(rho_dict, out_path="assets/figures/panel_d_sensitivity.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    names = list(rho_dict.keys())
    rhos = list(rho_dict.values())
    order = sorted(range(len(rhos)), key=lambda i: abs(rhos[i]))
    names = [names[i] for i in order]
    rhos = [rhos[i] for i in order]
    colors = [COLOR_RED if r > 0 else COLOR_STEEL for r in rhos]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    _ax(ax)
    bars = ax.barh(names, rhos, color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color="black", linewidth=0.7)
    for bar, r in zip(bars, rhos):
        ax.text(
            r + 0.005 * np.sign(r),
            bar.get_y() + bar.get_height() / 2,
            f"{r:.3f}",
            va="center",
            ha="left" if r >= 0 else "right",
            fontsize=8,
            fontfamily="DejaVu Sans",
            color=COLOR_CHARCOAL,
        )
    ax.set_xlabel("Spearman rank correlation [—]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_title(
        "(d)  Sensitivity Tornado — Wall Loss at Assessment Horizon",
        fontsize=10,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        loc="left",
    )
    p1 = mpatches.Patch(color=COLOR_RED, label="Positive")
    p2 = mpatches.Patch(color=COLOR_STEEL, label="Negative")
    ax.legend(handles=[p1, p2], fontsize=8, frameon=False)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_surrogate(surr, out_path="assets/figures/panel_e_surrogate.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax = axes[0]
    _ax(ax)
    ax.scatter(
        surr["y_test"], surr["y_pred_test"], s=4, alpha=0.3, color=COLOR_STEEL, rasterized=True
    )
    mn = min(surr["y_test"].min(), surr["y_pred_test"].min())
    mx = max(surr["y_test"].max(), surr["y_pred_test"].max())
    ax.plot([mn, mx], [mn, mx], color=COLOR_RED, linewidth=0.9, label="1:1")
    ax.set_xlabel("Physics wall loss [mm]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_ylabel("Surrogate prediction [mm]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_title(
        f"(e)  GBR Surrogate Parity\nR2={surr['r2_test']:.4f}  MAE={surr['mae_test']:.4f} mm",
        fontsize=10,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        loc="left",
    )
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    _ax(ax)
    names = surr["feature_names"]
    imp = surr["feature_importance"]
    order = np.argsort(imp)
    ax.barh(
        [names[i] for i in order],
        [imp[i] for i in order],
        color=COLOR_TEAL,
        edgecolor="none",
        height=0.6,
    )
    ax.set_xlabel("Permutation importance [delta R2]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_title(
        "(e')  Feature Importance",
        fontsize=10,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        loc="left",
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_iso_risk(mc_result, out_path="assets/figures/panel_f_iso_risk.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    from src.constants import MAX_WL_FRAC, PIPE_WT

    params = mc_result["params"]
    wl = mc_result["wall_loss"]
    limit = PIPE_WT * 1000 * MAX_WL_FRAC
    theta_c = params["theta_crit"]
    S_arr = np.log10(params["S_mag"] + 1e-15)
    failed = (wl >= limit).astype(float)
    # 2D bins
    tc_bins = np.linspace(theta_c.min(), theta_c.max(), 20)
    sm_bins = np.linspace(S_arr.min(), S_arr.max(), 20)
    pof_grid = np.full((19, 19), np.nan)
    for i in range(19):
        for j in range(19):
            mask = (
                (theta_c >= tc_bins[i])
                & (theta_c < tc_bins[i + 1])
                & (S_arr >= sm_bins[j])
                & (S_arr < sm_bins[j + 1])
            )
            if mask.sum() > 3:
                pof_grid[i, j] = failed[mask].mean()
    tc_mid = 0.5 * (tc_bins[:-1] + tc_bins[1:])
    sm_mid = 0.5 * (sm_bins[:-1] + sm_bins[1:])
    from scipy.interpolate import griddata

    valid = ~np.isnan(pof_grid)
    if valid.sum() > 4:
        TT, SS = np.meshgrid(tc_mid, sm_mid)
        pts = np.column_stack([TT[valid.T], SS[valid.T]])
        vals = pof_grid[valid]
        g_tc = np.linspace(tc_mid.min(), tc_mid.max(), 40)
        g_sm = np.linspace(sm_mid.min(), sm_mid.max(), 40)
        GT, GS = np.meshgrid(g_tc, g_sm)
        pof_i = griddata(pts, vals, (GT, GS), method="linear")
    else:
        g_tc, g_sm, GT, GS, pof_i = tc_mid, sm_mid, TT, SS, pof_grid

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    _ax(ax)
    cf = ax.contourf(
        g_tc, g_sm, pof_i, levels=np.linspace(0, 1, 21), cmap="RdYlGn_r", vmin=0, vmax=1
    )
    cs = ax.contour(
        g_tc,
        g_sm,
        pof_i,
        levels=[0.01, 0.05, 0.10],
        colors=[COLOR_TEAL, COLOR_NAVY, COLOR_RED],
        linewidths=0.9,
    )
    ax.clabel(cs, fmt=lambda x: f"PoF={x:.0%}", fontsize=8)
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04).set_label(
        "PoF [—]", fontsize=10, fontfamily="DejaVu Sans"
    )
    ax.set_xlabel("theta_crit [vol fraction]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_ylabel("log10(S_mag) [m3/m3/s]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_title(
        "(f)  Iso-Risk Contour — Go/No-Go at Assessment Horizon",
        fontsize=10,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        loc="left",
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_fad(result, out_path="assets/figures/panel_g_fad.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    from src.fad_assessment import Lr_max_value, fad_curve, fad_trajectory

    wl = result["wl"]
    t = result["t_yr"]
    traj = fad_trajectory(P_OP_BAR, wl)
    Lr_max = Lr_max_value()
    Lr_line = np.linspace(0, Lr_max * 1.1, 400)
    Kr_line = fad_curve(Lr_line, Lr_max)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    _ax(ax)
    ax.plot(Lr_line, Kr_line, color=COLOR_NAVY, linewidth=1.2, label="API 579-1 Level 2 Option B")
    ax.axvline(
        Lr_max, color=COLOR_NAVY, linewidth=0.7, linestyle="--", label=f"Lr_max = {Lr_max:.3f}"
    )
    ax.fill_between(Lr_line, Kr_line, 1.0, alpha=0.08, color=COLOR_RED, label="Unacceptable")
    sc = ax.scatter(traj["Lr"], traj["Kr"], c=t, cmap="viridis", s=10, zorder=4, rasterized=True)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label(
        "Time [yr]", fontsize=10, fontfamily="DejaVu Sans"
    )
    ax.plot(traj["Lr"][0], traj["Kr"][0], "o", color=COLOR_TEAL, markersize=8, label="t=0")
    ax.plot(
        traj["Lr"][-1], traj["Kr"][-1], "s", color=COLOR_RED, markersize=8, label=f"t={t[-1]:.0f}yr"
    )
    ax.set_xlim(0, max(Lr_max * 1.15, traj["Lr"].max() * 1.1))
    ax.set_ylim(0, max(traj["Kr"].max(), 1.0) * 1.15)
    ax.set_xlabel("Lr = sigma_ref / sigma_Y [—]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_ylabel("Kr = KI / Kmat [—]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_title(
        "(g)  Failure Assessment Diagram — API 579-1 Level 2",
        fontsize=10,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        loc="left",
    )
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_inverse(inv_results, out_path="assets/figures/panel_h_inverse.png"):
    """Panel h: synthetic DFOS inverse reconstruction for 3 test cases."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax = axes[0]
    _ax(ax)
    colors = [COLOR_NAVY, COLOR_TEAL, COLOR_RED]
    for k, (res, col) in enumerate(zip(inv_results, colors)):
        ax.scatter(
            [res["S_true"] * 1e6],
            [res["S_opt"] * 1e6],
            color=col,
            s=60,
            zorder=4,
            label=f"Case {k + 1}: error={abs(res['S_opt'] - res['S_true']) * 1e6:.2f} ppm",
        )
    mn_val = min(r["S_true"] for r in inv_results) * 1e6
    mx_val = max(r["S_true"] for r in inv_results) * 1e6
    ax.plot([mn_val, mx_val], [mn_val, mx_val], color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("True S_moisture [1e-6 m3/m3/s]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_ylabel("Recovered S_moisture [1e-6 m3/m3/s]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_title(
        "(h)  Inverse Recovery — DFOS Synthetic Tests",
        fontsize=10,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        loc="left",
    )
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    _ax(ax)
    for k, (res, col) in enumerate(zip(inv_results, colors)):
        ax.bar(k + 1, res["misfit_opt"], color=col, width=0.5, label=f"Case {k + 1}")
    ax.set_xlabel("Test case [—]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_ylabel("Residual misfit J(S_opt) [K^2]", fontsize=10, fontfamily="DejaVu Sans")
    ax.set_title(
        "(h')  Inverse Misfit", fontsize=10, fontweight="bold", fontfamily="DejaVu Sans", loc="left"
    )
    ax.legend(fontsize=8, frameon=False)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path
