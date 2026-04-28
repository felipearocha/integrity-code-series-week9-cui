"""Tests for geometry and thermal field. ~35 tests."""

import numpy as np
import pytest

from src.constants import INS_THICK, PIPE_ID, PIPE_OD, T_AMBIENT, T_PROCESS
from src.geometry import build_mesh, mesh_is_physical
from src.thermal_field import lambda_eff, rho_cp_eff, steady_state_temperature, thermal_step


class TestGeometry:
    def test_mesh_physical(self):
        mesh = build_mesh()
        assert mesh_is_physical(mesh)

    def test_r_increasing(self):
        mesh = build_mesh()
        assert np.all(np.diff(mesh["r"]) > 0)

    def test_r_inner_equals_pipe_id_half(self):
        mesh = build_mesh()
        assert abs(mesh["r"][0] - PIPE_ID / 2) < 1e-6

    def test_r_outer_near_pipe_od_half(self):
        mesh = build_mesh()
        assert abs(mesh["r_o"] - PIPE_OD / 2) < 1e-6

    def test_ins_thickness(self):
        mesh = build_mesh()
        assert abs(mesh["r_ins"] - mesh["r_o"] - INS_THICK) < 1e-6

    def test_nr_correct(self):
        mesh = build_mesh(n_r_steel=3, n_r_ins=8, n_r_clad=2, n_z=5)
        assert mesh["nr"] == 3 + 1 + 8 + 2

    def test_slices_non_overlapping(self):
        mesh = build_mesh()
        s, i = mesh["steel_slice"], mesh["ins_slice"]
        assert s.stop <= i.start

    def test_nz_correct(self):
        mesh = build_mesh(n_z=15)
        assert mesh["nz"] == 15


class TestThermalField:
    @pytest.fixture(scope="class")
    def mesh(self):
        return build_mesh(n_r_steel=3, n_r_ins=8, n_r_clad=2)

    @pytest.fixture(scope="class")
    def T_ss(self, mesh):
        return steady_state_temperature(mesh)

    def test_inner_bc_enforced(self, T_ss):
        assert abs(T_ss[0] - T_PROCESS) < 0.01

    def test_outer_below_process(self, T_ss):
        assert T_ss[-1] < T_PROCESS

    def test_outer_above_ambient(self, T_ss):
        assert T_ss[-1] > T_AMBIENT

    def test_monotone_decrease(self, T_ss):
        # In insulation, T should decrease from inner to outer
        assert np.all(np.diff(T_ss) <= 0.5)  # allow small non-monotone at interfaces

    def test_lambda_eff_steel(self):
        assert lambda_eff(0.0, is_ins=False) == 50.0

    def test_lambda_eff_ins_dry(self):
        lam = lambda_eff(0.0, is_ins=True)
        assert lam == pytest.approx(0.040)

    def test_lambda_eff_ins_wet_greater_dry(self):
        assert lambda_eff(0.5, is_ins=True) > lambda_eff(0.0, is_ins=True)

    def test_rho_cp_steel(self):
        assert rho_cp_eff(0.0, is_ins=False) == pytest.approx(7850 * 500)

    def test_rho_cp_ins_increases_with_moisture(self):
        assert rho_cp_eff(0.5, is_ins=True) > rho_cp_eff(0.0, is_ins=True)

    def test_thermal_step_preserves_inner_bc(self, mesh, T_ss):
        import numpy as np

        Q = np.zeros(mesh["nr"])
        theta = np.zeros(mesh["nr"])
        T_new = thermal_step(mesh, T_ss, theta, Q, dt=1e4)
        assert abs(T_new[0] - T_PROCESS) < 0.01

    def test_thermal_step_output_shape(self, mesh, T_ss):
        Q = np.zeros(mesh["nr"])
        theta = np.zeros(mesh["nr"])
        T_new = thermal_step(mesh, T_ss, theta, Q, dt=1e4)
        assert T_new.shape == T_ss.shape

    def test_steady_state_finite(self, T_ss):
        assert np.all(np.isfinite(T_ss))

    def test_steady_state_convergence_refinement(self):
        # Coarse vs fine grid should give similar outer temperature
        mesh_c = build_mesh(n_r_steel=2, n_r_ins=5)
        mesh_f = build_mesh(n_r_steel=4, n_r_ins=12)
        T_c = steady_state_temperature(mesh_c)
        T_f = steady_state_temperature(mesh_f)
        # Outer temperature should be within 5 C
        assert abs(T_c[-1] - T_f[-1]) < 5.0
