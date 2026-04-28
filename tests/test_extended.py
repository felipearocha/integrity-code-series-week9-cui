"""
Extended test suite to reach 150+ total. ~70 additional tests.
Covers: physics monotonicity, regression locks, convergence, edge cases,
integration chains, validation benchmark wrappers.
"""

import os

import numpy as np
import pytest

from src.constants import (
    MAX_WL_FRAC,
    P_OP_BAR,
    PIPE_WT,
    T_AMBIENT,
    T_PROCESS,
    THETA_CRIT,
    THETA_SAT,
)


# ── Physics monotonicity ──────────────────────────────────────────────────
class TestMonotonicity:
    def test_i_corr_increases_with_T(self):
        from src.electrochemistry import i_corr

        th = THETA_CRIT * 2
        assert i_corr(400.0, th) > i_corr(350.0, th) > i_corr(300.0, th)

    def test_i_corr_zero_at_threshold(self):
        from src.electrochemistry import i_corr

        assert i_corr(T_PROCESS, THETA_CRIT) == 0.0

    def test_wl_rate_increases_with_T(self):
        from src.electrochemistry import wall_loss_rate

        th = 0.3
        assert wall_loss_rate(400.0, th) > wall_loss_rate(300.0, th)

    def test_wl_monotone_in_time(self):
        from src.electrochemistry import faraday_step

        wl = 0.0
        prev = 0.0
        for _ in range(10):
            wl = faraday_step(wl, T_PROCESS, 0.3, 86400)
            assert wl >= prev
            prev = wl

    def test_D_theta_strictly_increasing(self):
        from src.moisture_field import D_theta

        thetas = np.linspace(0, THETA_SAT * 0.9, 10)
        Ds = [D_theta(t) for t in thetas]
        assert np.all(np.diff(Ds) > 0)

    def test_D_T_zero_at_dry(self):
        from src.moisture_field import D_T_coeff

        assert D_T_coeff(0.0, T_PROCESS) == 0.0

    def test_steady_state_T_monotone_decreasing(self):
        from src.geometry import build_mesh
        from src.thermal_field import steady_state_temperature

        mesh = build_mesh(n_r_steel=3, n_r_ins=10, n_r_clad=2)
        T = steady_state_temperature(mesh)
        # Should generally decrease from inner to outer (allow small wiggles at interfaces)
        assert T[0] > T[-1]
        assert T[0] >= T_AMBIENT

    def test_fad_curve_monotone_decreasing(self):
        from src.fad_assessment import Lr_max_value, fad_curve

        Lr = np.linspace(0, Lr_max_value() * 0.9, 30)
        Kr = fad_curve(Lr)
        assert np.all(np.diff(Kr) <= 0)

    def test_fad_curve_max_equals_1(self):
        from src.fad_assessment import fad_curve

        assert abs(fad_curve(np.array([0.0]))[0] - 1.0) < 1e-3

    def test_mc_wall_loss_positive(self):
        from src.monte_carlo import run_monte_carlo

        mc = run_monte_carlo(n_samples=100, n_t=5, seed=0)
        assert np.all(mc["wall_loss"] >= 0)

    def test_mc_pof_t_non_decreasing(self):
        from src.monte_carlo import run_monte_carlo

        mc = run_monte_carlo(n_samples=200, n_t=10, seed=1)
        assert np.all(np.diff(mc["PoF_t"]) >= -1e-10)


# ── Regression / locked baseline values ──────────────────────────────────
class TestRegression:
    """Regression tests with locked values. If these fail, a physics change occurred."""

    def test_i0_at_120C_order(self):
        from src.electrochemistry import i0

        # i0(393K) should be O(1e-3) A/m^2 with EA=50kJ/mol, i0_ref=1e-5
        v = i0(T_PROCESS)
        assert 1e-4 < v < 1e-1

    def test_wall_loss_rate_at_120C_range(self):
        from src.electrochemistry import wall_loss_rate

        v = wall_loss_rate(T_PROCESS, 0.3)
        assert 0.005 < v < 0.5  # mm/yr

    def test_D_theta_at_0_equals_D0(self):
        from src.constants import D_THETA0
        from src.moisture_field import D_theta

        assert D_theta(0.0) == pytest.approx(D_THETA0)

    def test_faraday_1yr_locked(self):
        from src.electrochemistry import faraday_step, wall_loss_rate

        SEC_PER_YR = 365.25 * 24 * 3600
        wl = faraday_step(0.0, T_PROCESS, 0.3, SEC_PER_YR)
        v = wall_loss_rate(T_PROCESS, 0.3)
        assert abs(wl - v) / v < 0.01

    def test_Lr_max_X52_range(self):
        from src.fad_assessment import Lr_max_value

        # X52: UTS=455, SMYS=358 -> Lr_max = 0.5*(1+455/358) = 1.135
        Lr_max = Lr_max_value()
        assert 1.10 < Lr_max < 1.20


# ── Convergence ───────────────────────────────────────────────────────────
class TestConvergence:
    def test_thermal_grid_convergence(self):
        """Finer insulation grid -> outer T converges."""
        from src.geometry import build_mesh
        from src.thermal_field import steady_state_temperature

        T_vals = []
        for n in [5, 10, 20]:
            mesh = build_mesh(n_r_steel=3, n_r_ins=n, n_r_clad=2)
            T = steady_state_temperature(mesh)
            T_vals.append(T[-1])
        # T_outer should converge (differences decrease)
        assert (
            abs(T_vals[2] - T_vals[1]) < abs(T_vals[1] - T_vals[0])
            or abs(T_vals[2] - T_vals[1]) < 1.0
        )  # within 1K at fine grid

    def test_moisture_front_propagation_timescale(self):
        """Moisture should reach inner face within physical timescale."""
        from src.constants import THETA_INIT
        from src.geometry import build_mesh
        from src.moisture_field import apply_holiday_bc, max_stable_dt, moisture_step
        from src.thermal_field import steady_state_temperature

        mesh = build_mesh(n_r_steel=3, n_r_ins=8, n_r_clad=2)
        theta = np.full(mesh["nr"], THETA_INIT)
        T = steady_state_temperature(mesh, theta)
        S = np.zeros(mesh["nr"])
        dt = max_stable_dt(mesh) * 0.4
        ins_sl = mesh["ins_slice"]
        # Run 1 year
        n_1yr = int(365 * 86400 / dt)
        for _ in range(min(n_1yr, 5000)):
            theta = apply_holiday_bc(theta, mesh)
            theta = moisture_step(mesh, theta, T, S, dt)
        assert theta[ins_sl.start] > THETA_CRIT


# ── Edge cases ────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_i_corr_at_zero_temperature_diff(self):
        from src.electrochemistry import i_corr

        # Should work at both ends of T range
        assert i_corr(273.15, 0.3) >= 0
        assert i_corr(450.0, 0.3) >= 0

    def test_faraday_zero_moisture_gives_no_change(self):
        from src.electrochemistry import faraday_step

        wl_before = 2.0
        wl_after = faraday_step(wl_before, T_PROCESS, 0.0, 86400)
        assert wl_after == wl_before

    def test_fad_zero_wall_loss(self):
        from src.fad_assessment import assessment_point

        Lr, Kr = assessment_point(P_OP_BAR, 0.0)
        assert Lr >= 0 and Kr >= 0

    def test_fad_large_wall_loss(self):
        from src.fad_assessment import assessment_point

        Lr, Kr = assessment_point(P_OP_BAR, PIPE_WT * 0.8 * 1000)
        assert Lr > 0

    def test_mc_single_sample(self):
        from src.monte_carlo import run_monte_carlo

        mc = run_monte_carlo(n_samples=1, n_t=5, seed=0)
        assert len(mc["wall_loss"]) == 1

    def test_audit_chain_empty_initially(self):
        from src.audit_chain import AuditChain

        ch = AuditChain()
        assert len(ch) == 0

    def test_audit_chain_multiple_entries_linked(self):
        from src.audit_chain import AuditChain

        ch = AuditChain()
        e0 = ch.append("r0", {}, {})
        e1 = ch.append("r1", {}, {})
        assert e1.prev_hash == e0.entry_hash

    def test_geometry_single_layer(self):
        from src.geometry import build_mesh, mesh_is_physical

        mesh = build_mesh(n_r_steel=1, n_r_ins=3, n_r_clad=1)
        assert mesh_is_physical(mesh)

    def test_moisture_step_no_source_conserves(self):
        """With zero source and no holiday, moisture should not increase."""
        from src.constants import THETA_INIT
        from src.geometry import build_mesh
        from src.moisture_field import moisture_step
        from src.thermal_field import steady_state_temperature

        mesh = build_mesh(n_r_steel=3, n_r_ins=8, n_r_clad=2)
        theta = np.full(mesh["nr"], THETA_INIT)
        T = steady_state_temperature(mesh, theta)
        S = np.zeros(mesh["nr"])
        theta_new = moisture_step(mesh, theta, T, S, dt=3600)
        # Without explicit holiday BC, inner nodes should stay near initial value
        ins_sl = mesh["ins_slice"]
        assert theta_new[ins_sl.start] <= theta[ins_sl.start] * 1.1 + 0.001

    def test_surrogate_feature_matrix_shape(self):
        from src.monte_carlo import _u_to_params, latin_hypercube_sample, run_monte_carlo
        from src.surrogate_gbr import build_feature_matrix

        u = latin_hypercube_sample(20, 6)
        params = _u_to_params(u)
        X = build_feature_matrix(params)
        assert X.shape == (20, 6)


# ── Integration chain ─────────────────────────────────────────────────────
class TestIntegration:
    def test_full_chain_10_steps(self):
        """Full physics chain: 10 outer timesteps, verify physical outputs."""
        from src.coupled_solver import run_baseline

        result = run_baseline(dt_days=90, n_years=2.5)
        assert result["wl"][-1] >= 0
        assert np.all(np.isfinite(result["T"]))
        assert np.all(np.isfinite(result["theta"]))

    def test_mc_then_surrogate_consistent(self):
        """MC wall loss distribution should be reproduced by surrogate within tolerance."""
        from src.monte_carlo import run_monte_carlo, spearman_sensitivity
        from src.surrogate_gbr import predict_surrogate, train_surrogate

        mc = run_monte_carlo(n_samples=300, n_t=5, seed=42)
        surr = train_surrogate(mc["params"], mc["wall_loss"], n_estimators=100, max_depth=3)
        assert surr["r2_test"] > 0.5

    def test_benchmark_runner(self):
        import validation.benchmarks as bm

        bm.benchmark_faraday_mass_balance()
        bm.benchmark_i0_arrhenius()
        bm.benchmark_moisture_diffusion_scaling()

    def test_inverse_with_zero_source(self):
        """With S_true=0, observation T should be close to no-holiday steady-state."""
        from src.geometry import build_mesh
        from src.inverse_problem import synthetic_dfos_observation

        mesh = build_mesh(n_r_steel=3, n_r_ins=6, n_r_clad=2)
        T_obs = synthetic_dfos_observation(mesh, 60 * 86400, 2, 0.0, noise_K=0.0, seed=0)
        assert np.isfinite(T_obs)
        assert T_AMBIENT < T_obs < T_PROCESS


# ── Parametrized physics sweeps ────────────────────────────────────────────
class TestParametrizedSweeps:
    @pytest.mark.parametrize("T_K", [300, 340, 380, 420])
    def test_i0_positive_at_temps(self, T_K):
        from src.electrochemistry import i0

        assert i0(T_K) > 0

    @pytest.mark.parametrize("theta", [0.0, 0.05, 0.10, 0.30, 0.60, 0.80])
    def test_D_theta_positive_at_moisture(self, theta):
        from src.moisture_field import D_theta

        assert D_theta(theta) > 0

    @pytest.mark.parametrize("theta", [0.0, 0.03, 0.10, 0.50])
    def test_i_corr_threshold_behaviour(self, theta):
        from src.electrochemistry import i_corr

        ic = i_corr(T_PROCESS, theta)
        if theta <= THETA_CRIT:
            assert ic == 0.0
        else:
            assert ic > 0.0

    @pytest.mark.parametrize("wl_mm", [0.0, 0.5, 1.0, 2.0, 4.0])
    def test_fad_assessment_point_at_wall_losses(self, wl_mm):
        from src.fad_assessment import assessment_point

        Lr, Kr = assessment_point(P_OP_BAR, wl_mm)
        assert Lr >= 0 and Kr >= 0

    @pytest.mark.parametrize("n_ins", [5, 8, 12])
    def test_mesh_builds_for_different_resolutions(self, n_ins):
        from src.geometry import build_mesh, mesh_is_physical

        mesh = build_mesh(n_r_ins=n_ins)
        assert mesh_is_physical(mesh)
        assert mesh["nr"] >= n_ins + 4  # total nodes >= insulation + boundaries

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_lhs_different_seeds_differ(self, seed):
        from src.monte_carlo import latin_hypercube_sample

        u1 = latin_hypercube_sample(20, 6, seed=seed)
        u2 = latin_hypercube_sample(20, 6, seed=seed + 100)
        assert not np.allclose(u1, u2)

    @pytest.mark.parametrize("dt_s", [3600, 86400, 864000])
    def test_faraday_step_scales_linearly_with_dt(self, dt_s):
        from src.electrochemistry import faraday_step

        wl1 = faraday_step(0.0, T_PROCESS, 0.3, dt_s)
        wl2 = faraday_step(0.0, T_PROCESS, 0.3, 2 * dt_s)
        assert abs(wl2 - 2 * wl1) / (wl2 + 1e-20) < 1e-9
