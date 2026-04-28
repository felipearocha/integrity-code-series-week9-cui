"""
Week 9 — Full execution script. Run from repo root.
Usage: python run_all.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("assets/figures", exist_ok=True)
os.makedirs("assets/animations", exist_ok=True)

print("=" * 70)
print("INTEGRITY CODE SERIES — Week 9")
print("CUI Thermohygro-Electrochemical Simulation with DFOS Inverse")
print("=" * 70)

# [1] Baseline
print("\n[1/9] Deterministic baseline...")
t0 = time.time()
from src.coupled_solver import run_baseline

result = run_baseline(dt_days=30, n_years=10)
ins_sl = result["mesh"]["ins_slice"]
print(f"  wl(t=0)={result['wl'][0]:.4f} mm  wl(t=10yr)={result['wl'][-1]:.4f} mm")
print(
    f"  T_outer={result['T'][-1, -1] - 273.15:.1f} C  theta_inner_final={result['theta'][-1, ins_sl.start]:.3f}"
)
print(f"  Monotone: {np.all(np.diff(result['wl']) >= -1e-10)}")

# [2] Benchmarks
print("\n[2/9] Validation benchmarks...")
import validation.benchmarks as bm

bm.benchmark_steady_state_T()
bm.benchmark_faraday_mass_balance()
bm.benchmark_butler_volmer_tafel()
bm.benchmark_i0_arrhenius()
bm.benchmark_moisture_diffusion_scaling()

# [3] Monte Carlo
print("\n[3/9] Monte Carlo N=10,000...")
from src.monte_carlo import run_monte_carlo, spearman_sensitivity

mc = run_monte_carlo(n_samples=10_000, n_t=20, seed=42)
p50 = float(np.percentile(mc["wall_loss"], 50))
p95 = float(np.percentile(mc["wall_loss"], 95))
print(f"  PoF={mc['pof_final']:.4f}  P50={p50:.4f}mm  P95={p95:.4f}mm")
print(f"  Perforations: {mc['censored'].sum()}")

# [4] Surrogate
print("\n[4/9] GBR surrogate...")
from src.surrogate_gbr import train_surrogate

surr = train_surrogate(mc["params"], mc["wall_loss"])
print(
    f"  R2_train={surr['r2_train']:.4f}  R2_test={surr['r2_test']:.4f}  MAE={surr['mae_test']:.4f}mm"
)

# [5] Sensitivity
print("\n[5/9] Spearman sensitivity...")
rho = spearman_sensitivity(mc)
for k, v in rho.items():
    print(f"  {k:15s}: {v:+.4f}")

# [6] Inverse problem (3 synthetic test cases)
# Search range capped at S_ref=1e-6: above that the partial-holiday BC
# saturates at THETA_SAT, so S is non-identifiable in the plateau region.
print("\n[6/9] DFOS inverse reconstruction...")
from src.geometry import build_mesh
from src.inverse_problem import solve_inverse, synthetic_dfos_observation

mesh_inv = build_mesh(n_r_steel=3, n_r_ins=6, n_r_clad=2)
dt_inv = 60 * 86400
n_inv = 4
inv_results = []
for S_true, seed in [(2e-7, 1), (5e-7, 2), (8e-7, 3)]:
    T_obs = synthetic_dfos_observation(mesh_inv, dt_inv, n_inv, S_true, noise_K=0.03, seed=seed)
    inv = solve_inverse(T_obs, mesh_inv, dt_inv, n_inv, S_lo=0, S_hi=1e-6, max_iter=30)
    inv["S_true"] = S_true
    err_pct = abs(inv["S_opt"] - S_true) / S_true * 100 if S_true > 0 else 0
    print(f"  S_true={S_true:.1e}  S_opt={inv['S_opt']:.1e}  err={err_pct:.1f}%")
    inv_results.append(inv)

# [7] FAD
print("\n[7/9] FAD assessment...")
from src.constants import P_OP_BAR
from src.fad_assessment import fad_trajectory

traj = fad_trajectory(P_OP_BAR, result["wl"])
n_unacceptable = (traj["status"] == "unacceptable").sum()
print(
    f"  Unacceptable points: {n_unacceptable}/{len(traj['status'])}  (all acceptable for 10yr baseline)"
)

# [8] Plots
print("\n[8/9] Generating plots...")
from visualization.plot_fields import plot_field_heatmaps

plot_field_heatmaps(result)
from visualization.plot_mc_dist import plot_mc_distribution

plot_mc_distribution(mc)
from visualization.plot_analysis import (
    plot_fad,
    plot_inverse,
    plot_iso_risk,
    plot_sensitivity,
    plot_surrogate,
)

plot_sensitivity(rho)
plot_surrogate(surr)
plot_iso_risk(mc)
plot_fad(result)
plot_inverse(inv_results)

# [9] GIF + audit
print("\n[9/9] GIF + audit chain...")
from visualization.generate_gif import generate_gif

generate_gif(result)
from src.audit_chain import get_chain, log_run

log_run(
    "baseline_week9",
    {"dt_days": 30, "n_years": 10},
    {"wl_10yr": float(result["wl"][-1]), "T_outer": float(result["T"][-1, -1])},
)
log_run("mc_10k", {"n_samples": mc["n_samples"]}, {"pof": mc["pof_final"], "p50": p50, "p95": p95})
chain = get_chain()
print(f"  Chain valid: {chain.verify_chain()}, entries: {len(chain)}")
with open("assets/audit_chain.json", "w") as f:
    f.write(chain.to_json())

print(f"\n{'=' * 70}")
print(f"Complete in {time.time() - t0:.0f}s")
print(
    f"Figures: {len(os.listdir('assets/figures'))} | Animations: {len(os.listdir('assets/animations'))}"
)
print(f"{'=' * 70}")
