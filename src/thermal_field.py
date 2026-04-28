"""
Fourier heat conduction solver — 1D radial (axisymmetric).

PDE: rho_eff*cp_eff * dT/dt = (1/r)*d/dr(r*lambda_eff*dT/dr) + Q_corr
BCs:
  T(r_i) = T_process           (Dirichlet inner wall)
  -lambda*dT/dr|r_o_clad = h*(T-T_inf) + eps*sigma*(T^4-T_inf^4)  (Robin outer)

Solved with Crank-Nicolson (theta=0.5) on the radial grid.
"""

import numpy as np
from scipy.linalg import solve_banded

from src.constants import (
    CLAD_EMISS,
    H_CONV,
    INS_CP_DRY,
    INS_K_DRY,
    INS_RHO_DRY,
    SIGMA_SB,
    STEEL_K,
    STEEL_RHO,
    T_AMBIENT,
    T_PROCESS,
    WATER_CP,
    WATER_RHO,
)


def lambda_eff(theta_w, is_ins):
    """Effective thermal conductivity [W m^-1 K^-1]."""
    if not is_ins:
        return STEEL_K
    return INS_K_DRY + 0.6 * theta_w  # linear moisture correction [ASSUMED]


def rho_cp_eff(theta_w, is_ins):
    """Volumetric heat capacity [J m^-3 K^-1]."""
    if not is_ins:
        return STEEL_RHO * 500.0  # steel cp ~ 500 J kg^-1 K^-1
    rho_dry = INS_RHO_DRY * INS_CP_DRY
    rho_water = WATER_RHO * WATER_CP * theta_w
    return rho_dry + rho_water


def build_thermal_system(mesh, T, theta_w, Q_corr, dt):
    """
    Build tridiagonal system for Crank-Nicolson thermal step.
    Returns (ab, rhs) for scipy solve_banded (banded form).
    """
    r = mesh["r"]
    nr = mesh["nr"]
    ins_sl = mesh["ins_slice"]
    ins_mask = np.zeros(nr, dtype=bool)
    ins_mask[ins_sl] = True

    lam = np.array([lambda_eff(theta_w[i], ins_mask[i]) for i in range(nr)])
    rhocp = np.array([rho_cp_eff(theta_w[i], ins_mask[i]) for i in range(nr)])

    # Half-point conductivities
    lam_half = 0.5 * (lam[:-1] + lam[1:])
    dr = np.diff(r)
    r_half = 0.5 * (r[:-1] + r[1:])

    # Build full matrix (not banded yet) using explicit storage
    diag = np.zeros(nr)
    upper = np.zeros(nr - 1)
    lower = np.zeros(nr - 1)
    rhs = np.zeros(nr)

    # Inner BC (Dirichlet)
    diag[0] = 1.0
    rhs[0] = T_PROCESS

    # Interior nodes (Crank-Nicolson)
    for i in range(1, nr - 1):
        drm = dr[i - 1]  # dr to left
        drp = dr[i]  # dr to right
        rm = r_half[i - 1]
        rp = r_half[i]
        lm = lam_half[i - 1]
        lp = lam_half[i]
        rc = rhocp[i]
        rc_dt = rc / dt
        coeff_m = lm * rm / (drm * r[i])
        coeff_p = lp * rp / (drp * r[i])

        lower[i - 1] = -0.5 * coeff_m
        diag[i] = rc_dt + 0.5 * (coeff_m + coeff_p)
        upper[i] = -0.5 * coeff_p

        # RHS (explicit half from current step)
        rhs[i] = (
            (rc_dt - 0.5 * (coeff_m + coeff_p)) * T[i]
            + 0.5 * coeff_m * T[i - 1]
            + 0.5 * coeff_p * T[i + 1]
            + Q_corr[i]
        )

    # Outer BC (Robin: convection + radiation), linearise radiation
    i = nr - 1
    drm = dr[-1]
    rm = r_half[-1]
    lm = lam_half[-1]
    rc = rhocp[i]
    coeff_m = lm * rm / (drm * r[i])
    # Linearised radiation: h_rad = eps*sigma*(T^3 + T^2*Tinf + T*Tinf^2 + Tinf^3)
    h_rad = CLAD_EMISS * SIGMA_SB * (T[i] ** 2 + T_AMBIENT**2) * (T[i] + T_AMBIENT)
    h_tot = H_CONV + h_rad
    rc_dt = rc / dt
    # Robin: lambda*dT/dr = h_tot*(T - T_amb) at outer face
    # FD approximation: lambda*(T[n] - T[n-1])/dr = h_tot*(T[n] - T_amb)
    lower[i - 1] = -lm / drm
    diag[i] = lm / drm + h_tot
    rhs[i] = h_tot * T_AMBIENT

    # Pack into banded form (ab[0] = upper, ab[1] = diag, ab[2] = lower)
    ab = np.zeros((3, nr))
    ab[0, 1:] = upper
    ab[1, :] = diag
    ab[2, :-1] = lower

    return ab, rhs


def thermal_step(mesh, T, theta_w, Q_corr, dt):
    """Advance thermal field by one timestep dt [s]."""
    ab, rhs = build_thermal_system(mesh, T, theta_w, Q_corr, dt)
    T_new = solve_banded((1, 1), ab, rhs)
    T_new[0] = T_PROCESS  # enforce Dirichlet
    return T_new


def steady_state_temperature(mesh, theta_w=None):
    """
    Compute steady-state radial temperature profile by iterating
    thermal_step until convergence (dt=1e6 s pseudo-transient).
    """
    nr = mesh["nr"]
    if theta_w is None:
        theta_w = np.zeros(nr)
    T = np.linspace(T_PROCESS, T_AMBIENT, nr)
    Q = np.zeros(nr)
    for _ in range(2000):
        T_new = thermal_step(mesh, T, theta_w, Q, dt=1e4)
        if np.max(np.abs(T_new - T)) < 1e-6:
            return T_new
        T = T_new
    return T
