# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Predetermined settings for the fixed-resource 2D circuit benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import path_setup  # noqa: F401
from gate_runtime import KRYLOV_TOL, SVD_THRESHOLD, TRUNC_MODE, TDVP_MODE

PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PACKAGE_DIR / "output"

BENCHMARK_ID = "fixed_resources_2d"
NUM_ROWS = 4
NUM_COLS = 4
NUM_QUBITS = NUM_ROWS * NUM_COLS

# Model parameters (predetermined; not tuned after runs).
ISING_J = 1.0
ISING_H = 1.0  # transverse field; library name is ``g``
HEISENBERG_J = 1.0
HEISENBERG_H = 0.0  # isotropic Heisenberg without extra field

DT = 0.1
TMAX_INITIAL = 3.0
TMAX_EXTENDED = 5.0
RELIABILITY_THRESHOLD = 1e-2

CHI_MAIN = 32
CHI_CONTROL = 256
CONTROL_STEPS = 2  # short high-capacity check (χ=256)

# TEBD/MPO at high χ should match the exact circuit near machine precision.
CONTROL_INF_TOL_EXACT = 1e-8
# Hybrid TDVP with one Krylov substep retains finite integrator error even at high χ.
CONTROL_INF_TOL_TDVP = 1e-3
# Heisenberg @ χ=256 (all methods) is too slow for routine validation; Ising alone
# is the high-capacity gate/ordering control. Heisenberg correctness is covered by
# gate-identity, nontrivial evolution, and the χ=32 trajectories.
CONTROL_MODELS = ("ising",)
CONTROL_SKIP: tuple[tuple[str, str], ...] = ()

TDVP_SUBSTEPS = 1
PLOT_FLOOR = 1e-12
T0_INFIDELITY_TOL = 1e-12

MODELS = ("ising", "heisenberg")
# Variational MPO omitted: per-gate fair-start VMPS proved prohibitively slow on these
# 16-qubit trajectories (hours per Trotter step); left for a dedicated follow-up.
METHODS = ("hybrid_tdvp", "tebd_swap", "mpo_zipup")
SKIPPED_METHODS = ("variational_mpo",)

# Difference from library defaults documented in validation output.
LIBRARY_NOTE = (
    "Snake MPS ordering and even/odd edge colouring follow "
    "`mqt.yaqs.core.libraries.circuit_library.create_2d_*`. "
    "This benchmark uses second-order Suzuki–Trotter (Strange: bond/2, field, bond/2) "
    "rather than the first-order schedules in those helpers. "
    "Ising uses g=h=1.0 (not the 0.5 test default). Heisenberg is isotropic J=1 with h=0. "
    "Variational MPO is omitted due to prohibitive wall-clock cost on full 2D circuits."
)


def timesteps_for_tmax(tmax: float) -> int:
    return int(round(tmax / DT))


def production_config(*, tmax: float) -> dict[str, Any]:
    return {
        "benchmark_id": BENCHMARK_ID,
        "grid": [NUM_ROWS, NUM_COLS],
        "num_qubits": NUM_QUBITS,
        "ising": {"J": ISING_J, "h": ISING_H},
        "heisenberg": {"J": HEISENBERG_J, "h": HEISENBERG_H},
        "dt": DT,
        "tmax": tmax,
        "timesteps": timesteps_for_tmax(tmax),
        "trotter_order": 2,
        "chi_main": CHI_MAIN,
        "chi_control": CHI_CONTROL,
        "control_steps": CONTROL_STEPS,
        "control_inf_tol_exact": CONTROL_INF_TOL_EXACT,
        "control_inf_tol_tdvp": CONTROL_INF_TOL_TDVP,
        "control_models": list(CONTROL_MODELS),
        "control_skip": [list(p) for p in CONTROL_SKIP],
        "tdvp_substeps": TDVP_SUBSTEPS,
        "reliability_threshold": RELIABILITY_THRESHOLD,
        "svd_threshold": SVD_THRESHOLD,
        "krylov_tol": KRYLOV_TOL,
        "trunc_mode": TRUNC_MODE,
        "tdvp_mode": TDVP_MODE,
        "initial_states": {"ising": "|0...0>", "heisenberg": "checkerboard_neel_snake"},
        "methods": list(METHODS),
        "skipped_methods": list(SKIPPED_METHODS),
        "library_note": LIBRARY_NOTE,
    }
