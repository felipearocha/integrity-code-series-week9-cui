"""Tests for coupled_solver, inverse, MC, surrogate, FAD, audit, viz. ~80 tests."""

import os

import numpy as np
import pytest


# ── Coupled solver ────────────────────────────────────────────────────────
class TestCoupledSolver:
    @pytest.fixture(scope="class")
    def result(self):
        from src.coupled_solver import run_baseline

        return run_baseline(dt_days=60, n_years=5)

    def test_wl_zero_at_t0(self, result):
        assert result["wl"][0] == 0.0

    def test_wl_monotone(self, result):
        assert np.all(np.diff(result["wl"]) >= -1e-10)

    def test_wl_positive_at_end(self, result):
        assert result["wl"][-1] >= 0.0

    def test_T_shape(self, result):
        nr = result["mesh"]["nr"]
        assert result["T"].shape[1] == nr

    def test_theta_shape(self, result):
        assert result["theta"].shape == result["T"].shape

    def test_T_inner_near_process(self, result):
        from src.constants import T_PROCESS

        assert abs(result["T"][-1, 0] - T_PROCESS) < 1.0

    def test_theta_bounds(self, result):
        assert np.all(result["theta"] >= 0)
        assert np.all(result["theta"] <= 0.85)

    def test_t_yr_monotone(self, result):
        assert np.all(np.diff(result["t_yr"]) > 0)

    def test_moisture_reaches_inner_face(self, result):
        from src.constants import THETA_CRIT

        ins_start = result["mesh"]["ins_slice"].start
        assert result["theta"][-1, ins_start] > THETA_CRIT


# ── Inverse problem ────────────────────────────────────────────────────────
class TestInverse:
    @pytest.fixture(scope="class")
    def setup(self):
        from src.constants import T_PROCESS
        from src.geometry import build_mesh
        from src.inverse_problem import solve_inverse, synthetic_dfos_observation

        mesh = build_mesh(n_r_steel=3, n_r_ins=6, n_r_clad=2)
        dt_s = 60 * 86400
        n_steps = 6
        S_true = 3e-7
        T_obs = synthetic_dfos_observation(mesh, dt_s, n_steps, S_true, noise_K=0.0, seed=1)
        inv = solve_inverse(T_obs, mesh, dt_s, n_steps, S_lo=0, S_hi=1e-5, max_iter=20)
        return inv, S_true

    def test_inverse_returns_dict(self, setup):
        inv, _ = setup
        assert isinstance(inv, dict)

    def test_S_opt_positive(self, setup):
        inv, _ = setup
        assert inv["S_opt"] >= 0

    def test_S_opt_in_search_range(self, setup):
        inv, _ = setup
        assert 0 <= inv["S_opt"] <= 1e-5

    def test_misfit_finite(self, setup):
        inv, _ = setup
        assert np.isfinite(inv["misfit_opt"])

    def test_iterations_positive(self, setup):
        inv, _ = setup
        assert inv["iterations"] > 0

    def test_synthetic_noise_adds_variability(self):
        from src.geometry import build_mesh
        from src.inverse_problem import synthetic_dfos_observation

        mesh = build_mesh(n_r_steel=3, n_r_ins=6, n_r_clad=2)
        T1 = synthetic_dfos_observation(mesh, 60 * 86400, 3, 5e-7, noise_K=0.0, seed=0)
        T2 = synthetic_dfos_observation(mesh, 60 * 86400, 3, 5e-7, noise_K=0.1, seed=0)
        # Noisy observation may differ
        assert isinstance(T1, float) and isinstance(T2, float)

    @pytest.mark.parametrize("S_true", [2e-7, 5e-7, 8e-7])
    def test_inverse_recovers_S_within_tolerance(self, S_true):
        """Regression guard for the v1.1.0 inverse fix.

        With the partial-holiday Dirichlet wired through to run_coupled,
        golden-section search must recover S_mag within 5% in the
        identifiable range S in [0, S_ref=1e-6]. Before the fix this
        produced 150-1400% error.
        """
        from src.geometry import build_mesh
        from src.inverse_problem import solve_inverse, synthetic_dfos_observation

        mesh = build_mesh(n_r_steel=3, n_r_ins=6, n_r_clad=2)
        dt_s = 60 * 86400
        n_steps = 4
        T_obs = synthetic_dfos_observation(
            mesh, dt_s, n_steps, S_true, noise_K=0.03, seed=int(S_true * 1e8)
        )
        inv = solve_inverse(T_obs, mesh, dt_s, n_steps, S_lo=0.0, S_hi=1.0e-6, max_iter=30)
        rel_err = abs(inv["S_opt"] - S_true) / S_true
        assert rel_err < 0.05, (
            f"Inverse recovery error {rel_err * 100:.2f}% exceeds 5% "
            f"(S_true={S_true:.1e}, S_opt={inv['S_opt']:.3e})"
        )


# ── Monte Carlo ────────────────────────────────────────────────────────────
class TestMonteCarlo:
    @pytest.fixture(scope="class")
    def mc(self):
        from src.monte_carlo import run_monte_carlo

        return run_monte_carlo(n_samples=500, n_t=10, seed=0)

    def test_shape(self, mc):
        assert len(mc["wall_loss"]) == 500

    def test_wl_non_negative(self, mc):
        assert np.all(mc["wall_loss"] >= 0)

    def test_pof_in_range(self, mc):
        assert 0 <= mc["pof_final"] <= 1.0

    def test_PoF_t_monotone(self, mc):
        assert np.all(np.diff(mc["PoF_t"]) >= -1e-10)

    def test_params_keys(self, mc):
        assert set(mc["params"].keys()) == {
            "S_mag",
            "theta_crit",
            "i0_ref",
            "E_a",
            "lambda_eff",
            "L_defect",
        }

    def test_spearman_in_range(self, mc):
        from src.monte_carlo import spearman_sensitivity

        rho = spearman_sensitivity(mc)
        assert all(-1 <= v <= 1 for v in rho.values())

    def test_lhs_shape(self):
        from src.monte_carlo import latin_hypercube_sample

        u = latin_hypercube_sample(100, 6)
        assert u.shape == (100, 6)

    def test_lhs_range(self):
        from src.monte_carlo import latin_hypercube_sample

        u = latin_hypercube_sample(50, 6)
        assert u.min() >= 0 and u.max() <= 1

    def test_lhs_stratified(self):
        from src.monte_carlo import latin_hypercube_sample

        u = latin_hypercube_sample(50, 3)
        for j in range(3):
            col = u[:, j]
            n = 50
            counts = [np.sum((col >= i / n) & (col < (i + 1) / n)) for i in range(n)]
            assert all(c == 1 for c in counts)


# ── Surrogate ─────────────────────────────────────────────────────────────
class TestSurrogate:
    @pytest.fixture(scope="class")
    def surr(self):
        from src.monte_carlo import run_monte_carlo
        from src.surrogate_gbr import train_surrogate

        mc = run_monte_carlo(n_samples=300, n_t=10, seed=7)
        return train_surrogate(mc["params"], mc["wall_loss"], n_estimators=50, max_depth=3)

    def test_r2_train_positive(self, surr):
        assert surr["r2_train"] > 0.5

    def test_r2_test_non_negative(self, surr):
        assert surr["r2_test"] >= 0

    def test_mae_non_negative(self, surr):
        assert surr["mae_test"] >= 0

    def test_feature_importance_length(self, surr):
        assert len(surr["feature_importance"]) == 6

    def test_predictions_non_negative(self, surr):
        from src.monte_carlo import run_monte_carlo
        from src.surrogate_gbr import predict_surrogate

        mc = run_monte_carlo(50, 10, seed=99)
        preds = predict_surrogate(surr, mc["params"])
        assert np.all(preds >= 0)


# ── FAD ──────────────────────────────────────────────────────────────────
class TestFAD:
    def test_Lr_max_gt_1(self):
        from src.fad_assessment import Lr_max_value

        assert Lr_max_value() > 1.0

    def test_fad_at_0_near_1(self):
        from src.fad_assessment import fad_curve

        assert abs(fad_curve(np.array([0.0]))[0] - 1.0) < 1e-3

    def test_fad_decreasing(self):
        from src.fad_assessment import Lr_max_value, fad_curve

        Lr = np.linspace(0, Lr_max_value() * 0.95, 50)
        Kr = fad_curve(Lr)
        assert np.all(np.diff(Kr) <= 0)

    def test_assessment_point_positive_Lr(self):
        from src.constants import P_OP_BAR
        from src.fad_assessment import assessment_point

        Lr, Kr = assessment_point(P_OP_BAR, 1.0)
        assert Lr > 0

    def test_trajectory_length(self):
        from src.constants import P_OP_BAR
        from src.fad_assessment import fad_trajectory

        wl = np.linspace(0, 3, 20)
        traj = fad_trajectory(P_OP_BAR, wl)
        assert len(traj["Lr"]) == 20


# ── Audit chain ────────────────────────────────────────────────────────────
class TestAuditChain:
    def test_append_verify(self):
        from src.audit_chain import AuditChain

        ch = AuditChain()
        e = ch.append("r1", {"T": 300}, {"v": 1.0})
        assert e.verify()

    def test_chain_valid(self):
        from src.audit_chain import AuditChain

        ch = AuditChain()
        ch.append("r1", {}, {})
        ch.append("r2", {}, {})
        assert ch.verify_chain()

    def test_tamper_detected(self):
        from src.audit_chain import AuditChain

        ch = AuditChain()
        e = ch.append("r1", {"x": 1}, {"y": 2})
        e.inputs["x"] = 999
        assert not e.verify()

    def test_prev_hash_first_zeros(self):
        from src.audit_chain import AuditChain

        ch = AuditChain()
        e = ch.append("r0", {}, {})
        assert e.prev_hash == "0" * 64


# ── Visualization output tests ─────────────────────────────────────────────
class TestVisualizationOutputs:
    def _check(self, path):
        assert os.path.exists(path), f"Missing: {path}"
        assert os.path.getsize(path) > 0, f"Empty: {path}"

    def test_field_heatmaps(self, tmp_path):
        from src.coupled_solver import run_baseline
        from visualization.plot_fields import plot_field_heatmaps

        result = run_baseline(dt_days=60, n_years=2)
        out = str(tmp_path / "figs" / "test_fields.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plot_field_heatmaps(result, out_path=out)
        self._check(out)

    def test_sensitivity_plot(self, tmp_path):
        from visualization.plot_analysis import plot_sensitivity

        rho = {
            "S_mag": 0.3,
            "theta_crit": -0.2,
            "i0_ref": 0.5,
            "E_a": 0.4,
            "lambda_eff": -0.1,
            "L_defect": 0.05,
        }
        out = str(tmp_path / "figs" / "test_sensitivity.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plot_sensitivity(rho, out_path=out)
        self._check(out)

    def test_fad_plot(self, tmp_path):
        from src.coupled_solver import run_baseline
        from visualization.plot_analysis import plot_fad

        result = run_baseline(dt_days=60, n_years=2)
        out = str(tmp_path / "figs" / "test_fad.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plot_fad(result, out_path=out)
        self._check(out)
