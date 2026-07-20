# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Configuration for the main-text single RZZ gate benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from gate_runtime import (
    KRYLOV_TOL,
    L_DEFAULT,
    SVD_THRESHOLD,
    TARGET_BOND_PROFILE,
    TDVP_MODE,
    TRUNC_MODE,
)

PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PACKAGE_DIR / "output"
EXPERIMENTS_DIR = PACKAGE_DIR.parent
FIGURES_DIR = EXPERIMENTS_DIR / "figures"
FIGURE_STEM = "figure_single_gate_main_text"

BENCHMARK_ID = "single_gate_main_text"
SEED = 11
GATE_TYPE = "rzz"
Q0, Q1 = 2, 9
SEPARATION = abs(Q1 - Q0)
CHI0 = max(TARGET_BOND_PROFILE)
METHODS = ("hybrid_tdvp", "tebd_swap", "mpo_zipup", "variational_mpo")

CHI_SCAN_LADDER = (8, 12, 16, 24, 32, 64, 128)
INTERMEDIATE_LADDER = (8, 12, 16, 24, 32, 48, 64, 96, 128)
CHI_SCAN_X = (1e-3, 1e-2, 0.1, 0.3)
FULL_INFIDELITY_THRESHOLD = 1e-10
ANGLE_TDVP_SUBSTEPS = 64
SUBSTEP_ANGLE_X = 0.1
SUBSTEP_VALUES = (1, 2, 4, 8, 16, 32, 64)
SPECIAL_X = (0.25, 0.5, 1.0)
PLOT_FLOOR = 1e-12
FIT_X_MIN = 1e-4
FIT_X_MAX = 1e-2


def build_generic_angle_grid(*, n_points: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Log-spaced generic angles excluding special landmarks."""
    x_log = np.logspace(-4.0, 0.0, n_points)
    special = set(SPECIAL_X)
    mask = np.array([not any(abs(x - s) < 1e-12 for s in special) for x in x_log])
    x_values = x_log[mask]
    return x_values, 2.0 * np.pi * x_values


def build_special_angles() -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(SPECIAL_X, dtype=float)
    return x_values, 2.0 * np.pi * x_values


def pick_intermediate_chi(low: int, full: int) -> int:
    """Pick predetermined log-scale midpoint from ``INTERMEDIATE_LADDER``."""
    if full <= low:
        msg = f"Need full > low, got low={low} full={full}"
        raise ValueError(msg)
    target = float(np.sqrt(low * full))
    candidates = [chi for chi in INTERMEDIATE_LADDER if low < chi < full]
    if not candidates:
        msg = f"No intermediate ladder value between {low} and {full}"
        raise ValueError(msg)
    return min(candidates, key=lambda chi: abs(np.log(chi) - np.log(target)))


def production_config(*, chi_low: int, chi_mid: int, chi_full: int) -> dict[str, Any]:
    x_gen, _ = build_generic_angle_grid()
    x_spec, _ = build_special_angles()
    return {
        "benchmark_id": BENCHMARK_ID,
        "seed": SEED,
        "L": L_DEFAULT,
        "gate_type": GATE_TYPE,
        "pair": [Q0, Q1],
        "separation": SEPARATION,
        "target_bond_profile": TARGET_BOND_PROFILE,
        "chi0": CHI0,
        "chi_low": chi_low,
        "chi_intermediate": chi_mid,
        "chi_full": chi_full,
        "methods": list(METHODS),
        "angle_tdvp_substeps": ANGLE_TDVP_SUBSTEPS,
        "generic_x_values": [float(x) for x in x_gen],
        "special_x_values": [float(x) for x in x_spec],
        "substep_angle_x": SUBSTEP_ANGLE_X,
        "substep_values": list(SUBSTEP_VALUES),
        "svd_threshold": SVD_THRESHOLD,
        "krylov_tol": KRYLOV_TOL,
        "trunc_mode": TRUNC_MODE,
        "tdvp_mode": TDVP_MODE,
        "full_infidelity_threshold": FULL_INFIDELITY_THRESHOLD,
    }
