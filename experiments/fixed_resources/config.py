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
from gate_runtime import KRYLOV_TOL, SVD_THRESHOLD, TDVP_MODE, TRUNC_MODE

PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PACKAGE_DIR / "output"
CORRECTED_OUTPUT_DIR = PACKAGE_DIR / "output_corrected"
FIGURES_DIR = PACKAGE_DIR.parent / "figures"
FIGURE_STEM = "figure_circuit_fixed_chi"

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
# Panel (b) display window (full trajectories still go to TMAX_INITIAL).
TFIM_TRAJ_PLOT_TMAX = 2.0
RELIABILITY_THRESHOLD = 1e-2
THRESHOLD_SENSITIVITY = (1e-3, 1e-2, 1e-1)

CHI_MAIN = 32
CHI_CONTROL = 256
CONTROL_STEPS = 2  # short high-capacity check (χ=256)

# TEBD/MPO at high χ should match the exact circuit near machine precision.
CONTROL_INF_TOL_EXACT = 1e-8
# Hybrid TDVP with finite Krylov/substep error retains a small integrator residual.
CONTROL_INF_TOL_TDVP = 1e-3
CONTROL_MODELS = ("ising",)
CONTROL_SKIP: tuple[tuple[str, str], ...] = ()

# Filled after subdivision validation; do not transfer n=1 from the single-gate study.
TDVP_SUBSTEPS = 2  # chosen by subdivision validation (stable vs n=4 on TFIM/Heisenberg)
SUBDIVISION_NS = (1, 2, 4, 8, 16, 64)
SUBDIVISION_TFIM_STEPS = 20  # t ≤ 2.0
SUBDIVISION_HEIS_STEPS = 1  # first-step control; multi-step horizon tested in production

PLOT_FLOOR = 1e-12
T0_INFIDELITY_TOL = 1e-12

MODELS = ("ising", "heisenberg")
METHODS = ("hybrid_tdvp", "tebd_swap", "mpo_zipup")
VARIATIONAL_CONTROL_METHODS = ("variational_mpo",)

CHI_HORIZON = (2, 4, 8, 12, 16, 24, 32, 48, 64)
CHI_HEISENBERG = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)

LIBRARY_NOTE = (
    "Snake MPS ordering and even/odd edge colouring follow "
    "`mqt.yaqs.core.libraries.circuit_library.create_2d_*`. "
    "This benchmark uses second-order Suzuki–Trotter (Strange: bond/2, field, bond/2) "
    "rather than the first-order schedules in those helpers. "
    "Ising uses g=h=1.0 (not the 0.5 test default). Heisenberg is isotropic J=1 with h=0. "
    "Hybrid TDVP: nearest-neighbor two-qubit gates use TEBD; non-adjacent gates use "
    "2-site TDVP with tdvp_sweeps=n fractional-time substeps. "
    "Exact reference is dense application of the identical Trotter circuit."
)


def timesteps_for_tmax(tmax: float) -> int:
    return int(round(tmax / DT))


def production_config(*, tmax: float, tdvp_substeps: int | None = None) -> dict[str, Any]:
    n = TDVP_SUBSTEPS if tdvp_substeps is None else int(tdvp_substeps)
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
        "chi_horizon": list(CHI_HORIZON),
        "chi_heisenberg": list(CHI_HEISENBERG),
        "control_steps": CONTROL_STEPS,
        "control_inf_tol_exact": CONTROL_INF_TOL_EXACT,
        "control_inf_tol_tdvp": CONTROL_INF_TOL_TDVP,
        "control_models": list(CONTROL_MODELS),
        "control_skip": [list(p) for p in CONTROL_SKIP],
        "tdvp_substeps": n,
        "reliability_threshold": RELIABILITY_THRESHOLD,
        "threshold_sensitivity": list(THRESHOLD_SENSITIVITY),
        "svd_threshold": SVD_THRESHOLD,
        "krylov_tol": KRYLOV_TOL,
        "trunc_mode": TRUNC_MODE,
        "tdvp_mode": TDVP_MODE,
        "initial_states": {"ising": "|0...0>", "heisenberg": "checkerboard_neel_snake"},
        "methods": list(METHODS),
        "library_note": LIBRARY_NOTE,
        "comparison_type": "fixed_bond_dimension_cap",
        "not_yet": "fixed_memory_or_runtime_frontier",
    }
