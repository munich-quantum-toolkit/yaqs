# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Frozen configuration for the individual-gates publication campaign."""

from __future__ import annotations

import math
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "output"
REPO_ROOT = EXPERIMENT_DIR.parents[1]

CAMPAIGN_ID = "individual_gates_v2_sites_1_10"
N = 12
Q0, Q1 = 1, 10  # zero-based; paper sites (2, 11)
SEEDS = (11, 22, 33)
BOND_PROFILE = [1, 2, 4, 8, 8, 8, 8, 8, 8, 8, 4, 2, 1]
CHI_MAX_VALUES = (8, 16)  # original Pauli/CNOT production-threshold campaign
DTYPE = "complex128"

SVD_THRESHOLD = 1e-13  # production-threshold campaign
KRYLOV_TOL = 1e-12
GATE_LIBRARY_SPLIT_CUTOFF = 1e-14
MIN_KEEP = 1
TRUNC_MODE = "discarded_weight"
TDVP_MODE = "2site"
N_SUB_MAIN = 1

# Effective-zero SVD threshold: production rejects literal 0.
EFFECTIVE_ZERO_SVD_THRESHOLD = 1e-300

METHODS = (
    "gate_local_2tdvp",  # DigitalSimParams gate_mode="full-tdvp"
    "mpo_zipup",  # gate_mode="mpo"
    "tebd_swap",  # gate_mode="swaps"
)
METHOD_TO_GATE_MODE = {
    "gate_local_2tdvp": "full-tdvp",
    "mpo_zipup": "mpo",
    "tebd_swap": "swaps",
}
DIRECT_METHODS = ("mpo_zipup", "tebd_swap")

PAULI_GATES = ("rxx", "ryy", "rzz")
# x = theta / (2π); theta = 2π x exactly (no wrapping).
X_VALUES = (
    0.0,
    1e-4,
    3.162277660e-4,
    1e-3,
    3.162277660e-3,
    1e-2,
    0.1,
    0.25,
)

# CNOT orientations: (control, target). First is the main-text orientation.
CNOT_ORIENTATIONS = (
    (Q0, Q1),
    (Q1, Q0),
)

# Fresh CNOT-versus-χ_max dataset (main-text panel d). Forward orientation only.
CNOT_RANK_CHI_VALUES = (8, 10, 12, 14, 16)
CNOT_RANK_TDVP_N_SUB = (1, 16, 128, 256)  # 128 is refinement-control only
CNOT_RANK_DISPLAY_N_SUB = (1, 16, 256)  # 256 = fine resolution (not "converged")
CNOT_RANK_CONTROL, CNOT_RANK_TARGET = Q0, Q1
CNOT_RANK_SVD_THRESHOLD = EFFECTIVE_ZERO_SVD_THRESHOLD

# Controlled-refinement diagnostic (forward CNOT, all campaign seeds, χ=8).
REFINEMENT_SEEDS = SEEDS
REFINEMENT_CONTROL_SEED = REFINEMENT_SEEDS[0]
REFINEMENT_CHI = 8
REFINEMENT_CONTROL, REFINEMENT_TARGET = Q0, Q1
REFINEMENT_N_SUB = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
REFINEMENT_FINE_N_SUB = 1024
REFINEMENT_SVD_THRESHOLD = EFFECTIVE_ZERO_SVD_THRESHOLD
REFINEMENT_KRYLOV_CONTROL_TOL = 1e-14

EXPECTED_PAULI_ROWS = len(PAULI_GATES) * len(SEEDS) * len(CHI_MAX_VALUES) * len(X_VALUES) * len(METHODS)
EXPECTED_CNOT_ROWS = len(CNOT_ORIENTATIONS) * len(SEEDS) * len(CHI_MAX_VALUES) * len(METHODS)
EXPECTED_CAMPAIGN_ROWS = EXPECTED_PAULI_ROWS + EXPECTED_CNOT_ROWS  # 468
EXPECTED_CNOT_RANK_DIRECT = len(CNOT_RANK_CHI_VALUES) * len(SEEDS) * len(DIRECT_METHODS)  # 30
EXPECTED_CNOT_RANK_TDVP = len(CNOT_RANK_CHI_VALUES) * len(SEEDS) * len(CNOT_RANK_TDVP_N_SUB)  # 60
EXPECTED_CNOT_RANK_ROWS = EXPECTED_CNOT_RANK_DIRECT + EXPECTED_CNOT_RANK_TDVP  # 90

# Kept byte-stable for resumable task hashes of existing refinement tasks.
CX_GENERATOR_BRANCH = (
    "H_CX = (pi/4)(I-Z_c)⊗(I-X_t) = (pi/2)(I-U_CX); U_CX(s)=expm(-i s H_CX) for s in [0,1]; expm(-i H_CX)=CX"
)
CX_PATH_NOTE = "TDVP substeps divide U_CX(s)=expm(-i s H_CX) for s in [0,1]; do not construct fractional CNOT gates."

VALIDATION_MATRIX_TOL = 1e-13
VALIDATION_FIDELITY_TOL = 1e-12
# Discarded-weight events below this are treated as numerical zero.
POSITIVE_WEIGHT_EPS = 1e-30


def theta_from_x(x: float) -> float:
    """Return ``theta = 2π x`` exactly."""
    return float(2.0 * math.pi * x)
