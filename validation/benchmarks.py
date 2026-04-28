"""
Analytical benchmarks for Week 9 CUI simulation.
Each benchmark compares numerical output to a closed-form reference.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import (
    ALPHA_A,
    ALPHA_C,
    EA_FE,
    ETA_MIXED,
    FARADAY,
    FE_M,
    FE_N,
    H_CONV,
    I0_REF,
    INS_K_DRY,
    INS_THICK,
    PIPE_OD,
    R_GAS,
    STEEL_K,
    STEEL_RHO,
    T_AMBIENT,
    T_PROCESS,
    T_REF_EC,
)


def benchmark_steady_state_T():
    """
    Analytical steady-state T for thin insulation annulus:
    T_outer = T_inner - (T_inner-T_amb)*R_ins/(R_ins+R_conv)
    where R_ins = ln(r_o+delta)/r_o / (2*pi*L*k_ins) per unit length.
    For slab approx: T_outer ~ T_inner - (T_inner-T_amb)*(k_ins/delta)/(k_ins/delta + h_conv)
    """
    import numpy as np

    from src.geometry import build_mesh
    from src.thermal_field import steady_state_temperature

    mesh = build_mesh(n_r_steel=3, n_r_ins=12, n_r_clad=2)
    T = steady_state_temperature(mesh)
    # Slab approximation: R_cond = delta/k = 0.075/0.040 = 1.875 m^2 K/W
    # R_conv = 1/h = 1/10 = 0.1 m^2 K/W
    # T_outer = T_inner - (T_inner-T_amb)*R_conv/(R_cond+R_conv)
    R_cond = INS_THICK / INS_K_DRY
    R_conv = 1.0 / H_CONV
    T_outer_analytic = T_PROCESS - (T_PROCESS - T_AMBIENT) * R_cond / (R_cond + R_conv)
    T_outer_numeric = T[-1]
    err = abs(T_outer_numeric - T_outer_analytic)
    print(
        f"T_outer analytic={T_outer_analytic:.2f} K, numeric={T_outer_numeric:.2f} K, err={err:.2f} K"
    )
    assert err < 15.0, f"Steady-state T error {err:.1f} K exceeds 15 K (slab approximation)"


def benchmark_faraday_mass_balance():
    """
    Faraday's law: 1 A/m^2 Fe dissolution -> 1.163 mm/yr.
    Identical to Week 8 — physics unchanged.
    """
    SEC_PER_YR = 365.25 * 24 * 3600
    v_mm_yr = 1.0 * FE_M / (FE_N * FARADAY * STEEL_RHO) * 1e3 * SEC_PER_YR
    v_ref = 1.163
    err = abs(v_mm_yr - v_ref) / v_ref
    print(f"Faraday: 1 A/m^2 -> {v_mm_yr:.4f} mm/yr (ref {v_ref}), err={err * 100:.3f}%")
    assert err < 0.01


def benchmark_butler_volmer_tafel():
    """
    At large |eta|: BV -> Tafel: i = i0*exp(alpha*F*eta/RT)
    For alpha=0.5, eta=0.4V, T=298K: ratio = exp(0.5*96485*0.4/8.314/298) = exp(7.77) ~ 2360
    """
    from src.electrochemistry import i0, i_corr

    T = 298.15
    eta_large = 0.4
    theta_wet = 0.3
    i0_val = i0(T)
    i_bv = i_corr(T, theta_wet, eta=eta_large)
    tafel = i0_val * np.exp(ALPHA_A * FARADAY * eta_large / (R_GAS * T))
    ratio = i_bv / tafel
    print(f"BV Tafel check: BV={i_bv:.4e} A/m^2, Tafel={tafel:.4e}, ratio={ratio:.4f}")
    assert abs(ratio - 1.0) < 0.05, f"BV deviates from Tafel by {abs(ratio - 1) * 100:.1f}%"


def benchmark_i0_arrhenius():
    """
    Arrhenius: ln(i0(T2)/i0(T1)) = Ea/R*(1/T1 - 1/T2)
    """
    from src.electrochemistry import i0

    T1, T2 = 298.15, 363.15
    ratio_numeric = i0(T2) / i0(T1)
    ratio_analytic = np.exp(EA_FE / R_GAS * (1 / T1 - 1 / T2))
    err = abs(ratio_numeric - ratio_analytic) / ratio_analytic
    print(
        f"Arrhenius: numeric={ratio_numeric:.4f}, analytic={ratio_analytic:.4f}, err={err * 100:.4f}%"
    )
    assert err < 1e-6


def benchmark_moisture_diffusion_scaling():
    """
    D_theta(theta_sat) / D_theta(0) = exp(BETA_THETA * THETA_SAT).
    This is a direct check of the Philip-de Vries parameterisation.
    """
    from src.constants import BETA_THETA, D_THETA0, THETA_SAT
    from src.moisture_field import D_theta

    ratio = D_theta(THETA_SAT) / D_theta(0.0)
    expected = np.exp(BETA_THETA * THETA_SAT)
    err = abs(ratio - expected) / expected
    print(f"D_theta ratio: {ratio:.4f} (expected {expected:.4f}), err={err * 100:.6f}%")
    assert err < 1e-6


if __name__ == "__main__":
    print("=" * 60)
    print("WEEK 9 BENCHMARKS")
    print("=" * 60)
    benchmark_steady_state_T()
    benchmark_faraday_mass_balance()
    benchmark_butler_volmer_tafel()
    benchmark_i0_arrhenius()
    benchmark_moisture_diffusion_scaling()
    print("\nAll benchmarks passed.")
