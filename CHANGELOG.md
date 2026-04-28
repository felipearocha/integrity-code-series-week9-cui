# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [1.1.0] — 2026-04-28

### Fixed

- **Inverse problem (critical):** the forward model `_T_clad_from_S` was
  independent of `S_mag` because the holiday source magnitude was hardcoded
  in `run_coupled` (`S[ins_sl.stop-1] = 1.0e-6`) and `S_mag` was only used
  as a boolean (`holiday=(S_mag > 0)`). Golden-section search converged to
  the upper search bound, producing 150–1400% recovery error in
  `run_all.py [6/9]`. Fixed by parameterising the outer Dirichlet BC
  through `_holiday_theta_outer(S_mag, S_ref)`:
  `theta_outer = f * THETA_SAT + (1 - f) * THETA_INIT, f = min(S_mag/S_ref, 1)`.
  Recovery error is now < 1% on the identifiable range `S ∈ [0, S_ref]`
  with 0.03 K DFOS noise.
- **Magic number in `coupled_solver._Q_array`:** corrosion heat-flux to
  volumetric source conversion used a hardcoded `0.002` m. Replaced with
  the mesh-derived first-cell thickness `mesh["r"][1] - mesh["r"][0]`.
- **Misleading docstring in `inverse_problem.py`:** removed the claim
  about an "adjoint gradient formulation" — only golden-section is
  implemented. The docstring now describes the actual single-parameter
  1D inverse and its identifiability range.
- **Duplicate inline comments in `constants.py`:** cleaned `D_THETA0`,
  `BETA_THETA`, `ETA_MIXED` annotations.

### Changed

- `run_coupled` accepts `S_mag` and `S_ref` keyword arguments. Defaults
  (`S_mag=S_ref=1e-6`) preserve legacy baseline behaviour exactly
  (`wl(10yr) = 0.1360 mm` unchanged).
- `moisture_step` and `apply_holiday_bc` accept `theta_outer=THETA_SAT`
  as a keyword argument. Default preserves prior behaviour.
- `run_all.py` `[6/9]` inverse search range capped at `S_hi = S_ref = 1e-6`
  with synthetic test cases `{2e-7, 5e-7, 8e-7}` — all in the identifiable
  range. Previous range `[0, 3e-6]` produced spurious "recovery" inside
  the saturation plateau.

### Added

- `LICENSE` file (RES-EDU-1.0) formalising the Research and Educational
  Use clause that previously appeared only as prose in `README.md`.
- `CHANGELOG.md` (this file).
- `CONTRIBUTING.md` with development setup and contribution guidelines.
- `pyproject.toml` with project metadata, ruff and pytest configuration.
- `.gitignore` covering Python build artifacts, generated assets, IDE
  files, and `__pycache__`.
- `.github/workflows/ci.yml` — GitHub Actions CI matrix
  (Python 3.11/3.12/3.13 × Ubuntu/macOS/Windows): ruff lint, pytest with
  coverage, validation benchmarks, and a `run_all.py` smoke test that
  uploads the generated figures and audit chain as artefacts.
- `.github/workflows/release.yml` — tag-triggered release workflow that
  attaches a sanitised source archive and the most recent CI artefacts.
- `.github/dependabot.yml` for weekly dependency and Actions updates.
- `.github/ISSUE_TEMPLATE/` (bug, feature, physics-question) and
  `.github/PULL_REQUEST_TEMPLATE.md`.
- `.pre-commit-config.yaml` running ruff (lint + format) and basic
  hygiene hooks before every commit.
- Inverse-recovery regression test in `tests/test_remaining.py`
  asserting `|S_opt - S_true| / S_true < 0.05` for at least one
  identifiable test case — locks the fix as a regression guard.

### Documentation

- `README.md` updated to reflect the partial-holiday Dirichlet model in
  Eq. (2c) and the `S ∈ [0, S_ref]` identifiability constraint on the
  inverse problem in Eq. (6).

## [1.0.0] — 2026-04-24

### Added

- Initial public scaffold: 3-PDE coupled CUI simulator, Strang operator
  splitting, 151 tests, 5 analytical benchmarks, Monte Carlo / LHS UQ,
  GBR surrogate, API 579-1 Level 2 FAD, SHA-256 audit chain, 7 panel
  figures + 1 GIF.

### Known issues (resolved in 1.1.0)

- Inverse problem reported spurious recovery — see 1.1.0 Fixed.
