"""
Butler-Volmer electrochemical kinetics and Faraday wall loss.

i_corr = i0(T) * [exp(alpha_a*F*eta/(RT)) - exp(-alpha_c*F*eta/(RT))]
       = 0  if theta_w <= theta_crit

dWT/dt = -M_Fe/(n*F*rho_steel) * i_corr
"""

import numpy as np

from src.constants import (
    ALPHA_A,
    ALPHA_C,
    EA_FE,
    ETA_MIXED,
    FARADAY,
    FE_M,
    FE_N,
    I0_REF,
    R_GAS,
    STEEL_RHO,
    T_REF_EC,
    THETA_CRIT,
)

SEC_PER_YR = 365.25 * 24.0 * 3600.0


def i0(T):
    """Exchange current density [A m^-2] via Arrhenius."""
    return I0_REF * np.exp(-EA_FE / R_GAS * (1.0 / T - 1.0 / T_REF_EC))


def i_corr(T, theta_w, eta=ETA_MIXED, theta_crit=THETA_CRIT):
    """
    Corrosion current density [A m^-2] at pipe inner surface.
    T, theta_w: scalars or arrays.
    """
    T = np.asarray(T, dtype=float)
    theta_w = np.asarray(theta_w, dtype=float)
    i0_val = i0(T)
    bv = i0_val * (
        np.exp(ALPHA_A * FARADAY * eta / (R_GAS * T))
        - np.exp(-ALPHA_C * FARADAY * eta / (R_GAS * T))
    )
    active = (theta_w > theta_crit).astype(float)
    return bv * active


def wall_loss_rate(T, theta_w, eta=ETA_MIXED, theta_crit=THETA_CRIT):
    """
    Wall loss rate [mm yr^-1] via Faraday's law.
    dWT/dt = M_Fe/(n*F*rho) * i_corr  [m/s -> mm/yr]
    """
    ic = i_corr(T, theta_w, eta, theta_crit)
    v_ms = ic * FE_M / (FE_N * FARADAY * STEEL_RHO)
    return v_ms * 1e3 * SEC_PER_YR


def faraday_step(wl_mm, T_inner, theta_inner, dt_s, eta=ETA_MIXED, theta_crit=THETA_CRIT):
    """
    Advance wall loss [mm] by dt_s seconds.
    T_inner: temperature at pipe inner surface [K]
    theta_inner: moisture at inner insulation face [vol fraction]
    Returns new wall loss [mm].
    """
    ic = i_corr(T_inner, theta_inner, eta, theta_crit)
    v_ms = ic * FE_M / (FE_N * FARADAY * STEEL_RHO)
    dwl = v_ms * 1e3 * dt_s  # m/s -> mm, * dt_s [s] -> mm
    return wl_mm + dwl


def heat_from_corrosion(T_inner, theta_inner, eta=ETA_MIXED, theta_crit=THETA_CRIT):
    """
    Q_corr [W m^-2] heat flux from anodic dissolution at pipe surface.
    Q = i_corr * |eta|
    """
    ic = i_corr(T_inner, theta_inner, eta, theta_crit)
    return ic * abs(eta)
