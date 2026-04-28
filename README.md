# ICS2 Week 9 — CUI Coupled Thermohygro-Electrochemical Simulation

[![CI](https://github.com/felipearocha/integrity-code-series-week9-cui/actions/workflows/ci.yml/badge.svg)](https://github.com/felipearocha/integrity-code-series-week9-cui/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-151%20passing-brightgreen.svg)](#testing)
[![License](https://img.shields.io/badge/license-Research--Educational-lightgrey.svg)](LICENSE)

Three-way coupled physics-first simulator for **Corrosion Under Insulation (CUI)**
on insulated carbon-steel process piping. Simulates moisture ingress through a
cladding holiday, hygrothermal transport to the hot pipe surface, and Butler-
Volmer corrosion kinetics — all solved together via Strang operator splitting.
Includes a DFOS-informed inverse problem to recover holiday source magnitude
from outer-cladding temperature observations.

## Quickstart

```bash
git clone https://github.com/felipearocha/integrity-code-series-week9-cui.git
cd REPO
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py        # ~70 s on a laptop; produces 7 figures + 1 GIF + audit_chain.json
pytest tests/ -v         # 151 tests
python validation/benchmarks.py   # 5 analytical benchmarks
```

## Problem Statement

Corrosion Under Insulation (CUI) on carbon steel process piping causes ~60% of pipe
leaks in documented refinery case studies (AMPP 2022). It progresses unseen beneath
insulation systems for years. PHMSA Docket 2026-1520 (April 24, 2026) places external
corrosion threats on hazardous liquid pipelines under active regulatory scrutiny.
API RP 583 identifies 50-175 C as the critical temperature window for CUI on carbon steel.

This package simulates three-way coupled physics: heat conduction through the insulation
annulus, hygrothermal moisture transport driven by capillary and thermal gradients, and
Butler-Volmer electrochemical corrosion at the steel surface — all solved together via
Strang operator splitting. A DFOS-informed inverse problem recovers moisture source
locations from outer cladding temperature measurements.

---

## Governing Equations

### Field 1 — Fourier Heat Conduction (Eq. 1)

```
rho_eff(theta_w)*cp_eff(theta_w)*dT/dt = div(lambda_eff(theta_w)*grad(T)) + Q_corr
```

Boundary conditions:
```
T(r_i) = T_process                                      (1a) inner wall Dirichlet
-lambda*dT/dr|_outer = h_conv*(T-T_inf) + eps*sigma*(T^4-T_inf^4)  (1b) Robin outer
dT/dz|_ends = 0                                         (1c) adiabatic axial ends
```

### Field 2 — Philip-de Vries Hygrothermal Moisture Transport (Eq. 2)

```
d(theta_w)/dt = div(D_theta(theta_w)*grad(theta_w)) + div(D_T(theta_w,T)*grad(T))
```

Where:
```
D_theta(theta_w) = D_theta0 * exp(beta_theta * theta_w)             (2a)
D_T(theta_w, T)  = D_vap_atm * xi(theta_w) * f(theta_w, T)          (2b)
theta_w|_holiday = f * THETA_SAT + (1 - f) * THETA_INIT             (2c) partial-holiday Dirichlet
                   with f = min(S_mag / S_ref, 1)
J_w.n|_intact    = 0                                                (2d) zero-flux
theta_w|_t=0     = THETA_INIT [ASSUMED]                             (2e)
```

The partial-holiday BC (2c) models the cladding defect as a fraction `f`
of the outer circumference at saturation; `S_mag = S_ref` (default
`S_REF_DEFAULT = 1e-6 m^3/m^3/s`) is a fully wet holiday, `S_mag = 0` is
intact cladding. Above `S_ref` the BC saturates and S_mag is non-
identifiable from T_clad alone, so the inverse problem (Eq. 6) is
restricted to the unique-recovery range `S ∈ [0, S_ref]`.

### Field 3 — Butler-Volmer Electrochemical Kinetics (Eq. 3)

```
i_corr = i0(T) * [exp(alpha_a*F*eta/(RT)) - exp(-alpha_c*F*eta/(RT))]  if theta_w > theta_crit
       = 0                                                                 if theta_w <= theta_crit

i0(T) = i0_ref * exp(-Ea/R * (1/T - 1/T_ref))                   (3a)
```

### Wall Loss — Faraday's Law (Eq. 4)

```
dWT/dt = -M_Fe/(n*F*rho_steel) * i_corr(T, theta_w)
```

### Strang Operator Splitting (Eq. 5)

```
u^{n+1} = L_EC(dt/2) o L_HY(dt/2) o L_TH(dt) o L_HY(dt/2) o L_EC(dt/2) u^n
```

### Tikhonov Inverse Problem (Eq. 6)

```
min_S J(S) = (T_obs - T_model(S))^2 + lambda_reg * S^2     S in [0, S_ref]
```

Single-parameter 1D inverse for `S_mag`, solved by golden-section line
search on `J(S)` in `src/inverse_problem.py`. Recovery error is &lt;1% on
the identifiable range `S in [0, S_ref]` for synthetic DFOS cases with
0.03 K measurement noise (see `[6/9]` in `run_all.py`). Above S_ref the
model BC saturates and the inverse is non-injective by construction.
A full adjoint formulation is out of scope.

### Monte Carlo / PoF (Eq. 9-10)

```
xi = {S_mag, theta_crit, i0_ref, E_a, lambda_eff, L_defect}  N_MC = 10,000 (LHS)
PoF(t*) = (1/N) * sum( 1[WT_k(t*) > 0.20 * t_nom] )
```

### FAD — API 579-1 Level 2 Option B (Eq. 12-13)

```
f(Lr) = [1 + 0.5*Lr^2]^(-1/2) * [0.3 + 0.7*exp(-0.65*Lr^6)]
Lr_max = 0.5*(1 + UTS/SMYS)
```

---

## Repository Structure

```
integrity_code_series_week9_cui_thermohygro/
├── run_all.py
├── requirements.txt
├── README.md
├── EXECUTION_ORDER.md
├── conftest.py
├── src/
│   ├── constants.py          Physical constants, [ASSUMED] flags
│   ├── geometry.py           2D axisymmetric FDM mesh
│   ├── thermal_field.py      Fourier PDE solver (Crank-Nicolson)
│   ├── moisture_field.py     Philip-de Vries solver (explicit, sub-stepped)
│   ├── electrochemistry.py   Butler-Volmer + Faraday wall loss
│   ├── coupled_solver.py     Strang operator splitting, baseline runner
│   ├── inverse_problem.py    Tikhonov inverse, DFOS synthetic tests
│   ├── monte_carlo.py        LHS sampling, MC propagation, Spearman
│   ├── surrogate_gbr.py      GBR surrogate, parity metrics
│   ├── fad_assessment.py     API 579-1 Level 2 FAD
│   └── audit_chain.py        SHA-256 hash-linked run log
├── validation/
│   └── benchmarks.py
├── visualization/
│   ├── plot_fields.py
│   ├── plot_mc_dist.py
│   ├── plot_analysis.py
│   └── generate_gif.py
├── tests/
│   ├── test_geometry_thermal.py
│   ├── test_moisture_electrochemistry.py
│   └── test_remaining.py
├── assets/figures/           8 static PNG panels (300 DPI)
├── assets/animations/        cui_moisture_front.gif
├── assets/audit_chain.json
└── notebooks/
```

---

## [ASSUMED] Parameter Flags

| Parameter | Value | Basis |
|-----------|-------|-------|
| D_theta0 (mineral wool) | 6.0e-11 m^2/s | Literature porous media, mineral wool [ASSUMED] |
| beta_theta | 5.0 | D_theta exponential slope [ASSUMED] |
| theta_crit | 0.05 vol fraction | API RP 583 qualitative guidance [ASSUMED] |
| i0_ref at 25 C | 1.0e-5 A/m^2 | General Fe dissolution literature [ASSUMED] |
| E_a Fe dissolution | 50 kJ/mol | General electrochemistry literature [ASSUMED] |
| eta_mixed | 0.15 V | Mixed potential anodic overpotential [ASSUMED] |
| K_mat X52 | 70 MPa sqrt(m) | Sweet service estimate [ASSUMED] |
| D_T coupling | 0.05 * D_theta0 * theta_w * T_norm | Philip-de Vries simplified [ASSUMED] |
| Holiday source S | 1.0e-6 m^3/m^3/s | Continuous wetting rate [ASSUMED] |

---

## Escalation Table (vs. Week 8)

| Dimension | Week 8 | Week 9 |
|-----------|--------|--------|
| PDE count | 1 (PR EOS + chemistry, algebraic) | 3 coupled PDEs (Fourier + PdV + BV) |
| Geometry | 1D spatial pipeline | 2D axisymmetric annulus |
| Inverse problem | None | DFOS-informed Tikhonov inversion |
| Operator method | Sequential | Strang splitting (2nd order) |
| Sensor integration | None | DFOS noise model + synthetic tests |

---

## Cybersecurity Summary

STRIDE threat model applied to DFOS sensor network feeding inverse solver:

| Threat | Mitigation |
|--------|-----------|
| Spoofing (DFOS fiber injection) | Cryptographic signing at acquisition unit |
| Tampering (inverse solver input) | Hash-verified I/O; audit_chain.json |
| Repudiation | Immutable SHA-256 chain per run |
| Information Disclosure | OT-segment VLAN for DFOS network |
| Denial of Service | Rate limiting + GBR surrogate fallback |
| Elevation of Privilege | Surrogate out-of-bounds triggers human review |
| Data Poisoning | Physics residual validation before GBR retraining |

---

## License

Research and educational use only. Not for operational fitness-for-service decisions
without independent engineering review and site-specific validation.
API RP 583, API 579-1, PHMSA regulations take precedence over model output.
[ASSUMED] parameters must be validated against site-specific inspection data.

---

Integrity Code Series — ICS2 | Week 9 | Physics-first. Verification over visibility.
