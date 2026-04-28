"""
Strang operator splitting — CUI thermohygro-electrochemical simulation.

Outer timestep: dt_outer (30 days nominal)
Inner moisture sub-stepping: moisture_step uses stable explicit Euler sub-steps.
Thermal and electrochemical: one solve per outer step.
"""

import numpy as np

from src.constants import PIPE_WT, T_PROCESS, THETA_CRIT, THETA_INIT, THETA_SAT
from src.electrochemistry import faraday_step, heat_from_corrosion
from src.moisture_field import apply_holiday_bc, max_stable_dt, moisture_step
from src.thermal_field import steady_state_temperature, thermal_step

S_REF_DEFAULT = 1.0e-6  # reference holiday source magnitude [m^3/m^3/s]


def _holiday_theta_outer(S_mag, S_ref=S_REF_DEFAULT):
    """
    Map source magnitude S_mag [m^3/m^3/s] to effective Dirichlet outer
    moisture content. Models a partial holiday: fraction f = S_mag/S_ref
    of the cladding circumference is wetted at THETA_SAT, the rest at
    THETA_INIT. Saturates at S_mag >= S_ref (full holiday).
    """
    if S_mag <= 0.0:
        return THETA_INIT
    f = min(S_mag / S_ref, 1.0)
    return f * THETA_SAT + (1.0 - f) * THETA_INIT


def _moisture_substep(mesh, theta, T, S, dt_outer, theta_outer=THETA_SAT):
    """Sub-step moisture field for stability."""
    dt_stab = max_stable_dt(mesh) * 0.4
    n_sub = max(1, int(dt_outer / dt_stab) + 1)
    dt_sub = dt_outer / n_sub
    for _ in range(n_sub):
        theta = apply_holiday_bc(theta, mesh, theta_outer=theta_outer)
        theta = moisture_step(mesh, theta, T, S, dt_sub, theta_outer=theta_outer)
        theta = np.clip(theta, 0.0, 0.80)
    return theta


def _Q_array(mesh, T_inner, theta_inner, eta=0.15):
    """Distribute corrosion heat flux q [W m^-2] into volumetric source
    Q [W m^-3] at the first interior steel node, using the mesh-derived
    cell thickness instead of a hardcoded constant."""
    Q = np.zeros(mesh["nr"])
    q = heat_from_corrosion(T_inner, theta_inner, eta)
    if mesh["nr"] >= 2:
        dr_cell = mesh["r"][1] - mesh["r"][0]
        Q[1] = q / dr_cell
    return Q


def run_coupled(
    mesh,
    T0,
    theta0,
    dt_s,
    n_steps,
    holiday=True,
    eta=0.15,
    theta_crit=THETA_CRIT,
    S_mag=S_REF_DEFAULT,
    S_ref=S_REF_DEFAULT,
):
    """
    Strang-split coupled simulation.

    holiday: if False, force dry outer BC (theta_outer=THETA_INIT) regardless
             of S_mag. If True, S_mag controls the outer Dirichlet via
             _holiday_theta_outer(S_mag, S_ref).
    S_mag:   moisture source magnitude [m^3/m^3/s]. Default S_REF_DEFAULT
             (1e-6) yields a fully wet holiday (theta_outer = THETA_SAT) —
             matches legacy baseline behaviour.
    S_ref:   reference magnitude defining "full holiday". S_mag >= S_ref
             saturates the outer BC at THETA_SAT.

    Returns T_hist(n+1, nr), theta_hist(n+1, nr), wl_hist(n+1), t_yr(n+1).
    """
    T = T0.copy()
    theta = theta0.copy()
    wl_mm = 0.0
    nr = mesh["nr"]
    ins_sl = mesh["ins_slice"]

    T_hist = np.zeros((n_steps + 1, nr))
    theta_hist = np.zeros((n_steps + 1, nr))
    wl_hist = np.zeros(n_steps + 1)
    t_hist = np.arange(n_steps + 1) * dt_s / (365.25 * 24 * 3600)

    T_hist[0] = T
    theta_hist[0] = theta

    if holiday:
        theta_outer = _holiday_theta_outer(S_mag, S_ref)
    else:
        theta_outer = THETA_INIT
    S = np.zeros(nr)  # volumetric source unused; outer Dirichlet carries S_mag

    for k in range(n_steps):
        T_inner = T[1]
        theta_inner = theta[ins_sl.start]

        # EC half
        wl_mm = faraday_step(wl_mm, T_inner, theta_inner, dt_s / 2, eta, theta_crit)
        wl_mm = min(wl_mm, PIPE_WT * 1000)

        # HY half (sub-stepped)
        theta = _moisture_substep(mesh, theta, T, S, dt_s / 2, theta_outer)

        # TH full
        Q = _Q_array(mesh, T_inner, theta_inner, eta)
        T = thermal_step(mesh, T, theta, Q, dt_s)
        T = np.clip(T, 200.0, T_PROCESS + 20.0)

        # HY half (sub-stepped)
        theta = _moisture_substep(mesh, theta, T, S, dt_s / 2, theta_outer)

        # EC half
        T_inner = T[1]
        theta_inner = theta[ins_sl.start]
        wl_mm = faraday_step(wl_mm, T_inner, theta_inner, dt_s / 2, eta, theta_crit)
        wl_mm = min(wl_mm, PIPE_WT * 1000)

        T_hist[k + 1] = T
        theta_hist[k + 1] = theta
        wl_hist[k + 1] = wl_mm

    return {"T": T_hist, "theta": theta_hist, "wl": wl_hist, "t_yr": t_hist, "mesh": mesh}


def run_baseline(dt_days=30.0, n_years=10.0):
    """
    Deterministic baseline: one cladding holiday, nominal parameters.
    Uses coarse mesh for speed; fine mesh available by increasing n_r_ins.
    """
    mesh = build_mesh_baseline()
    nr = mesh["nr"]
    theta0 = np.full(nr, THETA_INIT)
    T0 = steady_state_temperature(mesh, theta0)
    dt_s = dt_days * 24.0 * 3600.0
    n_steps = int(n_years * 365.25 / dt_days)
    result = run_coupled(mesh, T0, theta0, dt_s, n_steps, holiday=True)
    assert result["wl"][0] == 0.0, "wl(t=0) must be zero"
    assert np.all(np.diff(result["wl"]) >= -1e-9), "wl must be monotone"
    return result


def build_mesh_baseline():
    from src.geometry import build_mesh

    return build_mesh(n_r_steel=3, n_r_ins=8, n_r_clad=2, n_z=8)
