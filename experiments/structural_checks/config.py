# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Frozen configuration for the structural projector validation suite."""

from __future__ import annotations

import math
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "output"

N = 8
D = 2
CHI = 4
MPS_SEEDS = (101, 102, 103)
GENERATOR_SEED = 20260731
DTYPE = "complex128"

REL_TOL = 1e-10
ABS_ACTION_FLOOR = 1e-8
EXTERIOR_TERM_FLOOR = 1e-6
SIGMA_MIN_REL = 1e-8
SIGMA_DISC_REL = 1e-10
OBSTRUCTION_P2_TOL = 1e-12

# Product-state RXX obstruction (zero-based sites (0,3) = paper X_1 X_4).
OBSTRUCTION_N = 4
OBSTRUCTION_SITES = (0, 3)
OBSTRUCTION_THETA = 0.5 * math.pi  # π/2
PRODUCT_SWEEP_CONFIGS = (
    {"chi_max": 2, "n_sub": 1},
    {"chi_max": 2, "n_sub": 8},
    {"chi_max": 32, "n_sub": 1},
    {"chi_max": 32, "n_sub": 8},
)
PRODUCTION_STALL_DIST_TOL = 1e-12
PRODUCTION_STALL_INFIDELITY_TOL = 1e-12
DISCARDED_WEIGHT_TOL = 1e-14

FIXED_RANK_GEOMETRIES = {
    "interior": (2, 5),
    "left_boundary": (0, 4),
    "right_boundary": (3, 7),
}

TWO_SITE_GEOMETRIES = {
    "adjacent_interior": (3, 4),
    "separated_interior": (2, 5),
    "left_boundary": (0, 4),
    "right_boundary": (3, 7),
}

NN_GEOMETRIES = {
    "left_boundary": (0, 1),
    "interior": (3, 4),
    "right_boundary": (6, 7),
}

EXTERIOR_SEED = 101
EXTERIOR_SITES = (2, 5)


def bond_profile(n: int = N, chi: int = CHI) -> list[int]:
    """Exact MPS bond profile ``chi_c = min(2**c, 2**(n-c), chi)``.

    Returns:
        Bond dimensions at cuts ``0, ..., n``.
    """
    return [min(2**c, 2 ** (n - c), chi) for c in range(n + 1)]
