"""
Philip-de Vries hygrothermal moisture transport — robust 1D FDM.

Uses explicit Euler (stable with small dt) in insulation domain.
Boundary:
  outer ins node: Dirichlet THETA_SAT at holiday
  inner ins node: zero-flux (pipe surface does not store moisture)
"""

import numpy as np

from src.constants import BETA_THETA, D_THETA0, THETA_SAT


def D_theta(theta_w):
    """Isothermal moisture diffusivity [m^2 s^-1]."""
    tw = np.clip(theta_w, 0.0, THETA_SAT)
    return D_THETA0 * np.exp(BETA_THETA * tw)


def D_T_coeff(theta_w, T):
    """Philip-de Vries thermal diffusivity [m^2 s^-1 K^-1]."""
    tw = np.clip(theta_w, 0.0, THETA_SAT)
    return D_THETA0 * 0.05 * tw * (T / 373.15)


def moisture_step(mesh, theta_w, T, S_moisture, dt, theta_outer=THETA_SAT):
    """
    Advance moisture field by dt [s] using explicit Euler.
    Stable when dt < dr^2 / (2 * D_max).

    theta_outer: Dirichlet value at outer insulation face. Default THETA_SAT
    is a fully wet holiday; lower values represent partial holidays
    (smaller defect area or limited water availability at the cladding hole).
    """
    r = mesh["r"]
    ins_sl = mesh["ins_slice"]
    i0 = ins_sl.start
    i1 = ins_sl.stop - 1  # inclusive last insulation node

    if i1 - i0 < 2:
        return theta_w.copy()

    theta_new = theta_w.copy()

    for i in range(i0, i1 + 1):
        if i == i0:
            # Inner BC: reflecting (zero-flux) — moisture accumulates against pipe wall
            theta_new[i] = theta_new[i + 1] if (i + 1 <= i1) else theta_w[i]
            continue
        if i == i1:
            # Outer BC: Dirichlet theta_outer (parametric holiday) — fixed
            theta_new[i] = theta_outer
            continue

        drm = r[i] - r[i - 1]
        drp = r[i + 1] - r[i]
        rc = r[i]

        Dm = 0.5 * (D_theta(theta_w[i - 1]) + D_theta(theta_w[i]))
        Dp = 0.5 * (D_theta(theta_w[i]) + D_theta(theta_w[i + 1]))

        # Thermal contribution (explicit)
        DT_m = 0.5 * (D_T_coeff(theta_w[i - 1], T[i - 1]) + D_T_coeff(theta_w[i], T[i]))
        DT_p = 0.5 * (D_T_coeff(theta_w[i], T[i]) + D_T_coeff(theta_w[i + 1], T[i + 1]))
        dTdr_m = (T[i] - T[i - 1]) / drm
        dTdr_p = (T[i + 1] - T[i]) / drp

        # Radial diffusion: (1/r) d/dr(r*D*d_theta/dr)
        flux_m = Dm * 0.5 * (r[i - 1] + r[i]) / 2.0 * (theta_w[i] - theta_w[i - 1]) / drm
        flux_p = Dp * 0.5 * (r[i] + r[i + 1]) / 2.0 * (theta_w[i + 1] - theta_w[i]) / drp
        # Thermal flux (moisture toward hot pipe: positive dT/dr toward inner -> DT > 0)
        flux_T_m = DT_m * 0.5 * (r[i - 1] + r[i]) / 2.0 * dTdr_m
        flux_T_p = DT_p * 0.5 * (r[i] + r[i + 1]) / 2.0 * dTdr_p

        dr_avg = 0.5 * (drm + drp)
        div_theta = (flux_p - flux_m) / (rc * dr_avg)
        div_T = (flux_T_p - flux_T_m) / (rc * dr_avg)

        theta_new[i] = theta_w[i] + dt * (div_theta + div_T + S_moisture[i])

    theta_new = np.clip(theta_new, 0.0, THETA_SAT)
    return theta_new


def apply_holiday_bc(theta_w, mesh, theta_outer=THETA_SAT):
    theta_new = theta_w.copy()
    theta_new[mesh["ins_slice"].stop - 1] = theta_outer
    return theta_new


def max_stable_dt(mesh):
    """CFL stability criterion for explicit moisture diffusion."""
    r = mesh["r"]
    ins_sl = mesh["ins_slice"]
    r_ins = r[ins_sl.start : ins_sl.stop]
    dr_min = np.min(np.diff(r_ins))
    D_max = D_theta(THETA_SAT)
    return 0.4 * dr_min**2 / D_max
