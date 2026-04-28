"""Tests for moisture and electrochemistry modules. ~40 tests."""

import numpy as np
import pytest

from src.constants import BETA_THETA, D_THETA0, PIPE_WT, T_AMBIENT, T_PROCESS, THETA_CRIT, THETA_SAT
from src.electrochemistry import faraday_step, heat_from_corrosion, i0, i_corr, wall_loss_rate
from src.geometry import build_mesh
from src.moisture_field import D_T_coeff, D_theta, apply_holiday_bc, max_stable_dt, moisture_step
from src.thermal_field import steady_state_temperature


@pytest.fixture(scope="module")
def mesh():
    return build_mesh(n_r_steel=3, n_r_ins=8, n_r_clad=2)


@pytest.fixture(scope="module")
def T_ss(mesh):
    return steady_state_temperature(mesh)


class TestMoisture:
    def test_D_theta_positive(self):
        assert D_theta(0.0) > 0

    def test_D_theta_at_dry_equals_D0(self):
        assert D_theta(0.0) == pytest.approx(D_THETA0)

    def test_D_theta_increases_with_moisture(self):
        assert D_theta(0.5) > D_theta(0.1) > D_theta(0.0)

    def test_D_theta_at_sat(self):
        assert D_theta(THETA_SAT) == pytest.approx(D_THETA0 * np.exp(BETA_THETA * THETA_SAT))

    def test_D_T_positive(self):
        assert D_T_coeff(0.3, T_PROCESS) > 0

    def test_D_T_zero_at_dry(self):
        assert D_T_coeff(0.0, T_PROCESS) == 0.0

    def test_D_T_increases_with_moisture(self):
        assert D_T_coeff(0.5, T_PROCESS) > D_T_coeff(0.1, T_PROCESS)

    def test_max_stable_dt_positive(self, mesh):
        assert max_stable_dt(mesh) > 0

    def test_max_stable_dt_finite(self, mesh):
        assert np.isfinite(max_stable_dt(mesh))

    def test_moisture_step_shape(self, mesh, T_ss):
        nr = mesh["nr"]
        theta = np.full(nr, 0.05)
        S = np.zeros(nr)
        theta_new = moisture_step(mesh, theta, T_ss, S, dt=100.0)
        assert theta_new.shape == (nr,)

    def test_moisture_step_bounds(self, mesh, T_ss):
        nr = mesh["nr"]
        theta = np.full(nr, 0.3)
        S = np.zeros(nr)
        theta_new = moisture_step(mesh, theta, T_ss, S, dt=100.0)
        assert np.all(theta_new >= 0) and np.all(theta_new <= THETA_SAT)

    def test_holiday_bc_sets_outer_to_sat(self, mesh):
        theta = np.zeros(mesh["nr"])
        theta_new = apply_holiday_bc(theta, mesh)
        assert theta_new[mesh["ins_slice"].stop - 1] == THETA_SAT

    def test_holiday_bc_does_not_change_inner(self, mesh):
        theta = np.zeros(mesh["nr"])
        theta_new = apply_holiday_bc(theta, mesh)
        assert theta_new[0] == 0.0

    def test_moisture_propagates_inward(self, mesh, T_ss):
        """With holiday BC, inner insulation face should eventually exceed THETA_CRIT."""
        nr = mesh["nr"]
        theta = np.full(nr, 0.01)
        S = np.zeros(nr)
        dt = max_stable_dt(mesh) * 0.4
        for _ in range(500):
            theta = apply_holiday_bc(theta, mesh)
            theta = moisture_step(mesh, theta, T_ss, S, dt)
        ins_start = mesh["ins_slice"].start
        assert theta[ins_start] > THETA_CRIT


class TestElectrochemistry:
    def test_i0_positive(self):
        assert i0(T_PROCESS) > 0

    def test_i0_increases_with_T(self):
        assert i0(T_PROCESS) > i0(T_AMBIENT)

    def test_i0_arrhenius_ratio(self):
        # i0(353K)/i0(298K) should follow Arrhenius with Ea=50kJ/mol
        from src.constants import EA_FE, R_GAS, T_REF_EC

        ratio = i0(353.15) / i0(298.15)
        expected = np.exp(-EA_FE / R_GAS * (1 / 353.15 - 1 / T_REF_EC))
        assert abs(ratio - expected) / expected < 0.01

    def test_i_corr_zero_below_threshold(self):
        assert i_corr(T_PROCESS, THETA_CRIT * 0.5) == 0.0

    def test_i_corr_positive_above_threshold(self):
        assert i_corr(T_PROCESS, THETA_CRIT * 2) > 0

    def test_i_corr_increases_with_T(self):
        th = THETA_CRIT * 2
        assert i_corr(T_PROCESS, th) > i_corr(T_AMBIENT, th)

    def test_wall_loss_rate_zero_dry(self):
        assert wall_loss_rate(T_PROCESS, 0.0) == 0.0

    def test_wall_loss_rate_positive_wet(self):
        assert wall_loss_rate(T_PROCESS, 0.3) > 0

    def test_wall_loss_rate_mm_yr_range(self):
        v = wall_loss_rate(T_PROCESS, 0.3)
        assert 1e-4 < v < 10.0  # physical range for CUI

    def test_faraday_step_increases_wl(self):
        wl = faraday_step(0.0, T_PROCESS, 0.3, 86400.0)
        assert wl > 0.0

    def test_faraday_step_zero_dry(self):
        assert faraday_step(0.0, T_PROCESS, 0.0, 86400.0) == 0.0

    def test_faraday_step_monotone_with_time(self):
        wl1 = faraday_step(0.0, T_PROCESS, 0.3, 86400.0)
        wl2 = faraday_step(0.0, T_PROCESS, 0.3, 172800.0)
        assert wl2 > wl1

    def test_heat_from_corrosion_non_negative(self):
        assert heat_from_corrosion(T_PROCESS, 0.3) >= 0

    def test_heat_zero_dry(self):
        assert heat_from_corrosion(T_PROCESS, 0.0) == 0.0

    def test_faraday_vs_wall_loss_rate_consistency(self):
        """faraday_step for 1 year should match wall_loss_rate."""
        SEC_PER_YR = 365.25 * 24 * 3600
        v_yr = wall_loss_rate(T_PROCESS, 0.3)
        wl = faraday_step(0.0, T_PROCESS, 0.3, SEC_PER_YR)
        assert abs(wl - v_yr) / v_yr < 0.01
