# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Settings for the 4×4 TFIM resource-frontier benchmark."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import path_setup  # noqa: F401
from gate_runtime import KRYLOV_TOL, SVD_THRESHOLD, TDVP_MODE, TRUNC_MODE

PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PACKAGE_DIR / "output"
FIXED_RESOURCES_OUTPUT = PACKAGE_DIR.parent / "fixed_resources" / "output"

BENCHMARK_ID = "resource_frontier_tfim_4x4"
NUM_ROWS = 4
NUM_COLS = 4
NUM_QUBITS = NUM_ROWS * NUM_COLS

ISING_J = 1.0
ISING_H = 1.0
HEISENBERG_J = 1.0
HEISENBERG_H = 0.0

DT = 0.1
RELIABILITY_THRESHOLD = 1e-2
TARGET_STEPS = 15
TMAX = TARGET_STEPS * DT  # 1.5
TDVP_SUBSTEPS = 1

BYTES_PER_COMPLEX128 = 16
MIB = 1024.0 * 1024.0

# Validated χ ladder from fixed_resources dense scan.
CHI_INGEST = (2, 4, 8, 12, 16, 24, 32, 48, 64)
# Adaptive high-capacity ladder for TEBD/MPO only.
CHI_HIGH = (96, 128, 192, 256)

METHODS = ("hybrid_tdvp", "tebd_swap", "mpo_zipup")
SKIPPED_METHODS = ("variational_mpo",)

TIMING_REPEATS = 3
TIMING_NEAR_OPTIMAL_FRAC = 0.20
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

PLOT_FLOOR = 1e-12
T0_INFIDELITY_TOL = 1e-12

LIBRARY_NOTE = (
    "Same 4×4 TFIM circuit, |0…0> initial state, snake MPS ordering, and second-order "
    "Strange Trotter (Δt=0.1) as experiments/fixed_resources. Exact reference is same-circuit "
    "dense statevector evolution. Variational MPO omitted."
)


def apply_thread_limits() -> None:
    for key, value in THREAD_ENV.items():
        os.environ[key] = value


def production_config() -> dict[str, Any]:
    return {
        "benchmark_id": BENCHMARK_ID,
        "grid": [NUM_ROWS, NUM_COLS],
        "num_qubits": NUM_QUBITS,
        "model": "ising",
        "ising": {"J": ISING_J, "h": ISING_H},
        "dt": DT,
        "tmax": TMAX,
        "target_steps": TARGET_STEPS,
        "trotter_order": 2,
        "reliability_threshold": RELIABILITY_THRESHOLD,
        "tdvp_substeps": TDVP_SUBSTEPS,
        "chi_ingest": list(CHI_INGEST),
        "chi_high": list(CHI_HIGH),
        "methods": list(METHODS),
        "skipped_methods": list(SKIPPED_METHODS),
        "bytes_per_complex128": BYTES_PER_COMPLEX128,
        "svd_threshold": SVD_THRESHOLD,
        "krylov_tol": KRYLOV_TOL,
        "trunc_mode": TRUNC_MODE,
        "tdvp_mode": TDVP_MODE,
        "thread_env": dict(THREAD_ENV),
        "timing_repeats": TIMING_REPEATS,
        "library_note": LIBRARY_NOTE,
    }
