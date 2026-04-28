# Execution Order — Week 9

## Prerequisites
```bash
pip install -r requirements.txt
```
Python >= 3.11 required.

## Step-by-step
```bash
python run_all.py          # Steps 1-9 sequentially (~60-90s on standard laptop)
pytest tests/ -v           # 150+ tests
python validation/benchmarks.py  # Standalone analytical benchmarks
```

## Note on Moisture Timestep
The explicit moisture solver uses sub-stepping for CFL stability.
Max stable dt ~ 6 days for the default 8-node insulation grid.
Outer coupling timestep is 30 days by default (604 moisture sub-steps per outer step).
Finer insulation grids require smaller dt_sub but improve spatial accuracy.

## Output files
| File | Description |
|------|-------------|
| assets/figures/panel_ab_fields.png | Moisture and temperature 2D fields |
| assets/figures/panel_c_mc_distribution.png | CDF + histogram |
| assets/figures/panel_d_sensitivity.png | Spearman tornado |
| assets/figures/panel_e_surrogate.png | GBR parity + importance |
| assets/figures/panel_f_iso_risk.png | Go/no-go contour |
| assets/figures/panel_g_fad.png | API 579-1 FAD trajectory |
| assets/figures/panel_h_inverse.png | DFOS inverse reconstruction |
| assets/animations/cui_moisture_front.gif | Animated moisture front |
| assets/audit_chain.json | Immutable run log |
