"""
Physical constants and material properties for Week 9 CUI simulation.
[ASSUMED] flags mark parameters anchored to literature, not site-calibrated.
"""

R_GAS = 8.314462618  # J mol^-1 K^-1
FARADAY = 96485.33212  # C mol^-1
SIGMA_SB = 5.670374419e-8  # W m^-2 K^-4

STEEL_RHO = 7850.0  # kg m^-3
STEEL_SMYS = 358e6  # Pa  X52
STEEL_UTS = 455e6  # Pa  X52
STEEL_K = 50.0  # W m^-1 K^-1
FE_M = 55.845e-3  # kg mol^-1
FE_N = 2

PIPE_OD = 0.2191  # m  8.625 inch
PIPE_WT = 0.00819  # m  Sch 40
PIPE_ID = PIPE_OD - 2.0 * PIPE_WT
PIPE_L = 6.0  # m  one bay

INS_THICK = 0.075  # m  75 mm mineral wool [ASSUMED]
INS_K_DRY = 0.040  # W m^-1 K^-1 [ASSUMED]
INS_RHO_DRY = 100.0  # kg m^-3 [ASSUMED]
INS_CP_DRY = 840.0  # J kg^-1 K^-1 [ASSUMED]
D_THETA0 = 6.0e-11  # m^2 s^-1  isothermal moisture diffusivity, mineral wool [ASSUMED]
BETA_THETA = 5.0  # D_theta exponential slope [ASSUMED]

CLAD_THICK = 0.0009  # m  aluminium cladding
CLAD_K = 200.0  # W m^-1 K^-1
CLAD_EMISS = 0.05  # emissivity [ASSUMED]

T_PROCESS = 393.15  # K  120 C (within API RP 583 50-175 C window)
T_AMBIENT = 293.15  # K  20 C
H_CONV = 10.0  # W m^-2 K^-1 [ASSUMED]
P_OP_BAR = 10.0  # bar

THETA_INIT = 0.01  # vol fraction dry insulation [ASSUMED]
THETA_SAT = 0.80  # vol fraction saturated [ASSUMED]
THETA_CRIT = 0.05  # vol fraction electrolyte threshold [ASSUMED] API RP 583
D_VAP_ATM = 2.6e-5  # m^2 s^-1  vapour diffusivity in air
WATER_RHO = 1000.0  # kg m^-3
WATER_M = 18.015e-3  # kg mol^-1
WATER_CP = 4182.0  # J kg^-1 K^-1
WATER_HV = 2.45e6  # J kg^-1

ALPHA_A = 0.5  # [ASSUMED]
ALPHA_C = 0.5  # [ASSUMED]
I0_REF = 1.0e-5  # A m^-2 at T_REF [ASSUMED]
EA_FE = 50000.0  # J mol^-1 [ASSUMED]
T_REF_EC = 298.15  # K
ETA_MIXED = 0.15  # V  anodic mixed overpotential [ASSUMED]

MAX_WL_FRAC = 0.20
DESIGN_LIFE = 10.0  # yr
KMAT = 70.0  # MPa sqrt(m) [ASSUMED]

COLOR_NAVY = "#1b3a5c"
COLOR_STEEL = "#4c80b0"
COLOR_RED = "#8c2318"
COLOR_TEAL = "#2e7d7b"
COLOR_CHARCOAL = "#333333"
