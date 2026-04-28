"""API 579-1 Level 2 Option B FAD — identical structure to Week 8, adapted for X52."""

import numpy as np

from src.constants import KMAT, PIPE_OD, PIPE_WT, STEEL_SMYS, STEEL_UTS


def Lr_max_value():
    return 0.5 * (1.0 + STEEL_UTS / STEEL_SMYS)


def fad_curve(Lr, Lr_max=None):
    if Lr_max is None:
        Lr_max = Lr_max_value()
    Lr = np.asarray(Lr, dtype=float)
    Kr = np.zeros_like(Lr)
    v = Lr <= Lr_max
    Kr[v] = (1.0 + 0.5 * Lr[v] ** 2) ** (-0.5) * (0.3 + 0.7 * np.exp(-0.65 * Lr[v] ** 6))
    return Kr


def assessment_point(P_bar, wl_mm, Kmat=KMAT, D=PIPE_OD, t_nom=PIPE_WT):
    P_MPa = P_bar * 0.1
    t_rem = max(t_nom - wl_mm * 1e-3, 1e-4)
    sigma_h = P_MPa * D / (2.0 * t_rem)
    a_m = wl_mm * 1e-3
    x = min(a_m / t_rem, 0.99)
    F = 1.12 - 0.231 * x + 10.55 * x**2 - 21.72 * x**3 + 30.39 * x**4
    KI = sigma_h * np.sqrt(np.pi * a_m) * F
    Lr = sigma_h / (STEEL_SMYS * 1e-6)
    Kr = KI / Kmat
    return float(Lr), float(Kr)


def fad_trajectory(P_bar, wall_loss_arr, Kmat=KMAT):
    n = len(wall_loss_arr)
    Lr_arr = np.zeros(n)
    Kr_arr = np.zeros(n)
    Lr_max = Lr_max_value()
    for k, wl in enumerate(wall_loss_arr):
        Lr, Kr = assessment_point(P_bar, wl, Kmat)
        Lr_arr[k], Kr_arr[k] = Lr, Kr
    Kr_bound = fad_curve(Lr_arr, Lr_max)
    status = np.where(Kr_arr <= Kr_bound, "acceptable", "unacceptable")
    return {"Lr": Lr_arr, "Kr": Kr_arr, "Kr_boundary": Kr_bound, "Lr_max": Lr_max, "status": status}
