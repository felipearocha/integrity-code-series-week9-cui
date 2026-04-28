"""
Monte Carlo / LHS uncertainty quantification for Week 9 CUI.

Parameters sampled:
  S_mag      : moisture source magnitude [m^3/m^3/s]  LogNormal
  theta_crit : electrolyte threshold [vol frac]        Uniform
  i0_ref     : exchange current density [A/m^2]        LogNormal
  E_a        : activation energy [J/mol]               Normal
  lambda_eff : insulation conductivity [W/m/K]         Uniform
  L_defect   : holiday size (scaling)                  Uniform
"""

import numpy as np
from scipy.stats import lognorm, norm, spearmanr
from scipy.stats import uniform as sp_uniform

from src.constants import (
    DESIGN_LIFE,
    EA_FE,
    I0_REF,
    MAX_WL_FRAC,
    PIPE_WT,
)


def latin_hypercube_sample(n, d, seed=42):
    rng = np.random.default_rng(seed)
    result = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        result[:, j] = (perm + rng.uniform(size=n)) / n
    return result


def _u_to_params(u):
    S_mag = lognorm.ppf(u[:, 0], s=0.8, scale=1e-6)
    theta_crit = sp_uniform.ppf(u[:, 1], loc=0.03, scale=0.07)
    i0_ref = lognorm.ppf(u[:, 2], s=0.5, scale=I0_REF)
    E_a = norm.ppf(u[:, 3], loc=EA_FE, scale=5000.0)
    lambda_eff = sp_uniform.ppf(u[:, 4], loc=0.030, scale=0.030)
    L_defect = sp_uniform.ppf(u[:, 5], loc=0.5, scale=2.0)
    return {
        "S_mag": S_mag,
        "theta_crit": theta_crit,
        "i0_ref": i0_ref,
        "E_a": E_a,
        "lambda_eff": lambda_eff,
        "L_defect": L_defect,
    }


def _wl_from_params(S_mag_k, theta_crit_k, i0_ref_k, E_a_k, lambda_eff_k, t_yr):
    """
    Fast analytical wall loss estimate for MC inner loop.
    Avoids running full PDE per sample — uses physics-based closed-form.

    Time for electrolyte film to form: t_wet = f(S_mag, theta_crit)
    Once wet: i_corr(T_process, theta_crit) * Faraday integration
    """
    from src.constants import FARADAY, FE_M, FE_N, R_GAS, STEEL_RHO, T_PROCESS

    SEC_PER_YR = 365.25 * 24.0 * 3600.0

    # Time to reach theta_crit from dry: theta_crit / S_mag [s]
    t_wet_s = max(theta_crit_k / max(S_mag_k, 1e-15), 0.0)
    t_wet_yr = t_wet_s / SEC_PER_YR

    if t_wet_yr >= t_yr:
        return 0.0  # never reaches threshold

    # Corrosion duration [yr]
    t_corr_yr = t_yr - t_wet_yr

    # i_corr at T_process using sampled parameters
    i0_val = i0_ref_k * np.exp(-E_a_k / R_GAS * (1.0 / T_PROCESS - 1.0 / 298.15))
    from src.constants import ALPHA_A, ALPHA_C, ETA_MIXED

    bv = i0_val * (
        np.exp(ALPHA_A * FARADAY * ETA_MIXED / (R_GAS * T_PROCESS))
        - np.exp(-ALPHA_C * FARADAY * ETA_MIXED / (R_GAS * T_PROCESS))
    )
    bv = max(bv, 0.0)

    v_ms = bv * FE_M / (FE_N * FARADAY * STEEL_RHO)
    wl_mm = v_ms * 1e3 * t_corr_yr * SEC_PER_YR

    # Scale by L_defect and lambda (thermal effect proxy)
    lambda_scale = max(0.030 / max(lambda_eff_k, 1e-4), 0.5)
    return float(np.clip(wl_mm * lambda_scale, 0, PIPE_WT * 1000))


def run_monte_carlo(n_samples=10_000, t_assess_yr=None, n_t=40, seed=42):
    if t_assess_yr is None:
        t_assess_yr = DESIGN_LIFE
    u = latin_hypercube_sample(n_samples, 6, seed)
    params = _u_to_params(u)
    t_years = np.linspace(0, t_assess_yr, n_t)

    wall_loss_final = np.zeros(n_samples)
    censored = np.zeros(n_samples, dtype=bool)
    PoF_t = np.zeros(n_t)

    for k in range(n_samples):
        wl_traj = np.array(
            [
                _wl_from_params(
                    params["S_mag"][k],
                    params["theta_crit"][k],
                    params["i0_ref"][k],
                    params["E_a"][k],
                    params["lambda_eff"][k],
                    t,
                )
                for t in t_years
            ]
        )
        limit = PIPE_WT * 1000 * MAX_WL_FRAC
        wall_loss_final[k] = wl_traj[-1]
        if wl_traj[-1] >= PIPE_WT * 1000:
            censored[k] = True
        PoF_t += (wl_traj >= limit).astype(float)

    PoF_t /= n_samples
    return {
        "params": params,
        "wall_loss": wall_loss_final,
        "WT_nom_mm": np.full(n_samples, PIPE_WT * 1000),
        "censored": censored,
        "PoF_t": PoF_t,
        "t_years": t_years,
        "pof_final": float(PoF_t[-1]),
        "n_samples": n_samples,
    }


def spearman_sensitivity(mc_result):
    wl = mc_result["wall_loss"]
    rho_dict = {}
    for name, arr in mc_result["params"].items():
        rho, _ = spearmanr(arr, wl)
        rho_dict[name] = float(rho)
    return dict(sorted(rho_dict.items(), key=lambda x: abs(x[1]), reverse=True))
