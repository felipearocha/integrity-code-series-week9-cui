"""
DFOS-informed inverse moisture source identification.

Given: T_obs at outer cladding from DFOS sensor (single point, 1D radial).
Find:  S_magnitude — strength of moisture source at the holiday.

Forward map: run_coupled(S_mag) -> T_clad(t_end). S_mag controls a
partial-holiday outer Dirichlet BC via _holiday_theta_outer in
src/coupled_solver.py — see Eq. (2c) in README.

Misfit:  J(S0) = (T_clad_model - T_clad_obs)^2 + lambda_reg * S0^2

Solver:  golden-section line search on J(S0) over [S_lo, S_hi].
The inverse is uniquely identifiable only on S in [0, S_ref]; above
S_ref the BC saturates at THETA_SAT and J becomes flat. Callers should
cap S_hi <= S_ref.

NOTE: a full adjoint formulation is out of scope; only the 1D
single-parameter inverse is implemented.
"""

import numpy as np

from src.constants import THETA_INIT
from src.thermal_field import steady_state_temperature


def _T_clad_from_S(S_mag, mesh, dt_s, n_steps, eta=-0.15, S_ref=1.0e-6):
    """Run forward model and return outer cladding temperature [K] at t_end.

    S_mag is now wired through to run_coupled, which maps it to a partial-
    holiday outer Dirichlet:
      theta_outer = (S_mag/S_ref)*THETA_SAT + (1 - S_mag/S_ref)*THETA_INIT
    (clamped at S_mag >= S_ref). Higher S_mag => more moisture penetrates =>
    higher effective insulation conductivity => warmer outer cladding.
    """
    from src.coupled_solver import run_coupled

    nr = mesh["nr"]
    theta0 = np.full(nr, THETA_INIT)
    T0 = steady_state_temperature(mesh, theta0)
    result = run_coupled(
        mesh, T0, theta0, dt_s, n_steps, holiday=(S_mag > 0), eta=eta, S_mag=S_mag, S_ref=S_ref
    )
    return result["T"][-1, -1]


def misfit(S_mag, T_obs, mesh, dt_s, n_steps, lambda_reg=1e-12, eta=-0.15):
    """
    Scalar misfit J(S0) for single-parameter inverse.
    T_obs: observed outer cladding temperature [K]
    """
    T_model = _T_clad_from_S(S_mag, mesh, dt_s, n_steps, eta)
    return (T_model - T_obs) ** 2 + lambda_reg * S_mag**2


def solve_inverse(
    T_obs,
    mesh,
    dt_s,
    n_steps,
    S_lo=0.0,
    S_hi=1e-5,
    lambda_reg=1e-12,
    tol=1e-10,
    max_iter=50,
    eta=-0.15,
):
    """
    Golden-section search for optimal S_mag minimising misfit J(S0).

    Returns
    -------
    dict: S_opt, T_model_opt, misfit_opt, iterations
    """
    phi = (np.sqrt(5) - 1) / 2.0
    a, b = S_lo, S_hi

    for it in range(max_iter):
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        Jc = misfit(c, T_obs, mesh, dt_s, n_steps, lambda_reg, eta)
        Jd = misfit(d, T_obs, mesh, dt_s, n_steps, lambda_reg, eta)
        if Jc < Jd:
            b = d
        else:
            a = c
        if abs(b - a) < tol:
            break

    S_opt = (a + b) / 2.0
    T_model_opt = _T_clad_from_S(S_opt, mesh, dt_s, n_steps, eta)

    return {
        "S_opt": S_opt,
        "T_model_opt": T_model_opt,
        "T_obs": T_obs,
        "misfit_opt": misfit(S_opt, T_obs, mesh, dt_s, n_steps, lambda_reg, eta),
        "iterations": it + 1,
        "lambda_reg": lambda_reg,
    }


def synthetic_dfos_observation(mesh, dt_s, n_steps, S_true, noise_K=0.05, seed=7):
    """
    Generate synthetic DFOS measurement from forward model + Gaussian noise.
    noise_K: DFOS temperature measurement noise std [K] [ASSUMED] 0.05 K
    """
    rng = np.random.default_rng(seed)
    T_true = _T_clad_from_S(S_true, mesh, dt_s, n_steps)
    return T_true + rng.normal(0.0, noise_K)
