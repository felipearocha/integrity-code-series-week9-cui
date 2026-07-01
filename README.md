# ICS2 Week 9 — CUI Coupled Thermohygro-Electrochemical Simulation

[![CI](https://github.com/felipearocha/integrity-code-series-week9-cui/actions/workflows/ci.yml/badge.svg)](https://github.com/felipearocha/integrity-code-series-week9-cui/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests: 154 passing](https://img.shields.io/badge/tests-154%20passing-brightgreen.svg)](tests)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20172508.svg)](https://doi.org/10.5281/zenodo.20172508)

Three-way coupled physics-first simulator for **Corrosion Under Insulation (CUI)**
on insulated carbon-steel process piping. Simulates moisture ingress through a
cladding holiday, hygrothermal transport to the hot pipe surface, and Butler-
Volmer corrosion kinetics — all solved together via Strang operator splitting.
Includes a DFOS-informed inverse problem to recover holiday source magnitude
from outer-cladding temperature observations.

## Integrity Code Series

Part of an ongoing series of physics-first integrity simulators by Felipe Rocha:

| # | Repo | Domain |
|---|---|---|
| Week 3 | [Integrity-code-series-3](https://github.com/felipearocha/Integrity-code-series-3) | F1 lap simulation (six coupled ODEs) |
| Week 6 | [integrity-code-series-week6-smartphone-galvanic](https://github.com/felipearocha/Integrity-code-series-week6-smartphone-galvanic) | Smartphone galvanic corrosion (Laplace + Butler-Volmer) |
| Week 7 | [integrity_code_series_week7_h2_lferw](https://github.com/felipearocha/integrity_code_series_week7_h2_lferw) | LF-ERW H2 conversion (B31.12 + NACE TM0316) |
| Week 8 | [integrity-code-series-week8-creep-fatigue-heater](https://github.com/felipearocha/integrity-code-series-week8-creep-fatigue-heater) | Creep-fatigue 9Cr-1Mo (Norton/Omega + Coffin-Manson) |
| **Week 9** | **[integrity-code-series-week9-cui](https://github.com/felipearocha/integrity-code-series-week9-cui)** | **CUI thermohygro-electrochemical (3 PDEs, Strang) — this repo** |
| Week 10 | [integrity-code-series-week-10_nnph_scc](https://github.com/felipearocha/integrity-code-series-week-10_nnph_scc) | NNpHSCC full-physics (Chen-Sutherby-Xing + BS 7910) |
| Week 11 | [integrity-code-series-week11-erosion-corrosion-multiphase](https://github.com/felipearocha/integrity-code-series-week11-erosion-corrosion-multiphase) | Erosion-corrosion multiphase (NORSOK M-506 + DNV-RP-O501 + G119 + API 579) |
| Bonus | [Vibration-Accelerated-Corrosion-Coupled-Mechano-Electrochemical-Simulation](https://github.com/felipearocha/Vibration-Accelerated-Corrosion-Coupled-Mechano-Electrochemical-Simulation) | Vibration-accelerated corrosion (SDOF + Butler-Volmer + Archard) |
| Bonus | [synthetic-integrity-digital-twin-piml](https://github.com/felipearocha/synthetic-integrity-digital-twin-piml) | Physics-informed neural-network surrogate |
| Bonus | [integrity-data-foundation](https://github.com/felipearocha/integrity-data-foundation) | Engineering data validation baseline |

## Quickstart

```bash
git clone https://github.com/felipearocha/integrity-code-series-week9-cui.git
cd REPO
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py        # ~70 s on a laptop; produces 7 figures + 1 GIF + audit_chain.json
pytest tests/ -v         # 154 tests
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

[**view the full rendered reference**](https://htmlpreview.github.io/?https://github.com/felipearocha/integrity-code-series-week9-cui/blob/main/docs/equations.html)

Full rendered (MathJax) reference: **[docs/equations.html](docs/equations.html)** — open in any browser.

The eight headline governing equations, rendered natively on GitHub:

**Field 1 — Fourier heat conduction (Eq. 1):**

$$\rho_{\text{eff}}(\theta_w)\,c_{p,\text{eff}}(\theta_w)\,\frac{\partial T}{\partial t} \;=\; \nabla\!\cdot\!\bigl(\lambda_{\text{eff}}(\theta_w)\,\nabla T\bigr) \;+\; Q_{\text{corr}}$$

**Field 2 — Philip-de Vries hygrothermal moisture transport (Eq. 2):**

$$\frac{\partial \theta_w}{\partial t} \;=\; \nabla\!\cdot\!\bigl(D_\theta(\theta_w)\,\nabla \theta_w\bigr) \;+\; \nabla\!\cdot\!\bigl(D_T(\theta_w,T)\,\nabla T\bigr)$$

**Field 3 — Butler-Volmer electrochemical kinetics (Eq. 3):**

$$i_{\text{corr}} = \begin{cases} i_0(T)\left[\,e^{\,\alpha_a F \eta /(RT)} - e^{-\alpha_c F \eta /(RT)}\,\right], & \theta_w > \theta_{\text{crit}} \\[4pt] 0, & \theta_w \le \theta_{\text{crit}} \end{cases}$$

**Wall loss — Faraday's law (Eq. 4):**

$$\frac{dWT}{dt} = -\frac{M_{\text{Fe}}}{n\,F\,\rho_{\text{steel}}}\; i_{\text{corr}}(T, \theta_w)$$

**Strang operator splitting (Eq. 5):**

$$u^{\,n+1} = L_{\text{EC}}\!\left(\tfrac{\Delta t}{2}\right) \circ L_{\text{HY}}\!\left(\tfrac{\Delta t}{2}\right) \circ L_{\text{TH}}(\Delta t) \circ L_{\text{HY}}\!\left(\tfrac{\Delta t}{2}\right) \circ L_{\text{EC}}\!\left(\tfrac{\Delta t}{2}\right)\, u^{\,n}$$

**Tikhonov inverse problem (Eq. 6):**

$$\min_{S}\; J(S) = \bigl(T_{\text{obs}} - T_{\text{model}}(S)\bigr)^{2} + \lambda_{\text{reg}}\,S^{2}, \qquad S \in [0, S_{\text{ref}}]$$

**Monte Carlo / probability of failure (Eq. 9-10):**

$$\mathrm{PoF}(t^{*}) = \frac{1}{N}\sum_{k=1}^{N} \mathbf{1}\!\left[\, WT_k(t^{*}) > 0.20\, t_{\text{nom}} \,\right]$$

**Failure Assessment Diagram — API 579-1 Level 2 Option B (Eq. 12-13):**

$$f(L_r) = \left[\,1 + 0.5\,L_r^{2}\,\right]^{-1/2} \left[\,0.3 + 0.7\,e^{-0.65\,L_r^{6}}\,\right]$$

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
├── docs/
│   └── equations.html        Rendered (MathJax) governing-equations reference
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
│   ├── test_extended.py
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

## Cybersecurity (STRIDE)

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

## Anti-Hallucination Note

Every physical parameter carries an explicit provenance tag. Constants read directly from
a standard or the literature are stated as such; parameters anchored to literature but not
calibrated to a specific site are flagged `[ASSUMED]` in `src/constants.py` and collected
in the **[ASSUMED] Parameter Flags** table above. In the Integrity Code Series tiering this
maps to:

- **T1 [SOURCE]** — values fixed by physics or a cited standard (the API 579-1 Level 2
  Option B FAD curve, Faraday's law, the API RP 583 50-175 C CUI window).
- **T2 [SOURCE]** — quantities derived from T1 inputs (effective insulation properties, the
  golden-section Tikhonov recovery restricted to `S ∈ [0, S_ref]`).
- **T3 [ASSUMED]** — practitioner / literature estimates that are not site-calibrated: the
  mineral-wool moisture diffusivity `D_theta0` and slope `beta_theta`, the electrolyte
  threshold `theta_crit`, the Fe-dissolution `i0_ref` / `E_a` / mixed overpotential, and the
  `D_T` coupling form.

No equation, constant, or citation in this repository or in `docs/equations.html` is
introduced beyond what the model actually implements.

---

## Disclaimer

Research tool only. Not for design, fitness-for-service, or safety-critical decisions without site-specific calibration and independent PE review.

Additionally: API RP 583, API 579-1, and PHMSA regulations take precedence over model
output, and `[ASSUMED]` parameters must be validated against site-specific inspection data
before any operational use.

---

## License

MIT — Felipe Rocha. See [LICENSE](LICENSE).

---

Integrity Code Series — ICS2 | Week 9 | Physics-first. Verification over visibility.
---

## How to Cite

If this software contributes to your work, please cite both the software (this repository) and the underlying methods it implements.

**Software (archived release):**

> Rocha, F. (2026). *Integrity Code Series - Week 9 - CUI Coupled Thermohygro-Electrochemical Simulation* (Version 1.1.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20172508

**BibTeX:**

```bibtex
@software{rocha_2026_week9,
  author       = {Rocha, Felipe},
  title        = {{Integrity Code Series - Week 9 - CUI Coupled Thermohygro-Electrochemical Simulation}},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.1.1},
  doi          = {10.5281/zenodo.20172508},
  url          = {https://doi.org/10.5281/zenodo.20172508}
}
```

The two DOIs Zenodo provides are:

| DOI                                  | What it points to                                                  |
|--------------------------------------|--------------------------------------------------------------------|
| `10.5281/zenodo.20172508` (concept)   | Always resolves to the latest version - use this for citation.     |
| `10.5281/zenodo.20172509` (version)   | Pinned to v1.1.1 specifically - use when reproducibility matters.  |

A machine-readable citation file is also available in [`CITATION.cff`](CITATION.cff) - GitHub will display a "Cite this repository" widget at the top right of the repo page that exports BibTeX / APA / RIS automatically.
