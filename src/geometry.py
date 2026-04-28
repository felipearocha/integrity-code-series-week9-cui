"""
2D axisymmetric finite-difference mesh for CUI simulation.

Domain radial layers:
  [r_i, r_o]           steel pipe wall
  [r_o, r_o+INS_THICK] insulation annulus
  [r_o+INS, r_clad]    aluminium cladding

Axial: z in [0, PIPE_L], uniform grid.
"""

import numpy as np

from src.constants import CLAD_THICK, INS_THICK, PIPE_ID, PIPE_L, PIPE_OD


def build_mesh(n_r_steel=4, n_r_ins=20, n_r_clad=2, n_z=30):
    """
    Build 1D radial node array and axial node array.
    Returns dict with r_nodes, z_nodes, layer_indices.
    """
    r_i = PIPE_ID / 2.0
    r_o = PIPE_OD / 2.0
    r_ins = r_o + INS_THICK
    r_clad = r_ins + CLAD_THICK

    r_steel = np.linspace(r_i, r_o, n_r_steel + 1)
    r_insul = np.linspace(r_o, r_ins, n_r_ins + 1)[1:]
    r_cladg = np.linspace(r_ins, r_clad, n_r_clad + 1)[1:]

    r_nodes = np.concatenate([r_steel, r_insul, r_cladg])
    z_nodes = np.linspace(0.0, PIPE_L, n_z)

    idx_steel_end = n_r_steel  # last steel index
    idx_ins_start = n_r_steel + 1
    idx_ins_end = n_r_steel + n_r_ins
    idx_clad_start = n_r_steel + n_r_ins + 1

    return {
        "r": r_nodes,
        "z": z_nodes,
        "nr": len(r_nodes),
        "nz": n_z,
        "r_i": r_i,
        "r_o": r_o,
        "r_ins": r_ins,
        "r_clad": r_clad,
        "steel_slice": slice(0, idx_steel_end + 1),
        "ins_slice": slice(idx_ins_start, idx_ins_end + 1),
        "clad_slice": slice(idx_clad_start, len(r_nodes)),
    }


def mesh_is_physical(mesh):
    r = mesh["r"]
    return bool(np.all(np.diff(r) > 0) and r[0] > 0)
