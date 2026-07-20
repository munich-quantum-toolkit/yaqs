# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Settings for the 4×4 TFIM TDVP substep convergence audit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import path_setup  # noqa: F401
from gate_runtime import KRYLOV_TOL, SVD_THRESHOLD, TDVP_MODE, TRUNC_MODE

PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PACKAGE_DIR / "output"
RESOURCE_FRONTIER_OUTPUT = PACKAGE_DIR.parent / "resource_frontier" / "output"

BENCHMARK_ID = "convergence_tfim_tdvp_substeps"
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
TMAX = TARGET_STEPS * DT
TDVP_SUBSTEPS = 1  # default for shared trajectory imports

BYTES_PER_COMPLEX128 = 16

CHI_VALUES = (16, 32, 64)
# Probe χ first when extending the substep ladder beyond the cached set.
CHI_PROBE = 32
SUBSTEPS_INITIAL = (1, 2, 4)
SUBSTEPS_EXTENDED = 8
# Further ladder rungs (not hashed — changing these must not invalidate the SQLite cache).
SUBSTEPS_N16 = 16
SUBSTEPS_N32 = 32

# Convergence criteria (n vs 2n)
ABS_INF_TOL = 1e-4
REL_INF_TOL = 0.05
# Halt if median D_8/D_4 over the comparison window is not clearly < 1.
D_RATIO_DECREASE_MAX = 1.0

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

PLOT_FLOOR = 1e-12


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
        "chi_values": list(CHI_VALUES),
        "substeps_initial": list(SUBSTEPS_INITIAL),
        "substeps_extended": SUBSTEPS_EXTENDED,
        "abs_inf_tol": ABS_INF_TOL,
        "rel_inf_tol": REL_INF_TOL,
        "svd_threshold": SVD_THRESHOLD,
        "krylov_tol": KRYLOV_TOL,
        "trunc_mode": TRUNC_MODE,
        "tdvp_mode": TDVP_MODE,
        "bytes_per_complex128": BYTES_PER_COMPLEX128,
        "thread_env": dict(THREAD_ENV),
        "method": "hybrid_tdvp",
        "initial_state": "|0...0>",
        "library_note": (
            "Identical 4×4 TFIM Strang circuit and gate_runtime TDVP settings as "
            "experiments/fixed_resources and experiments/resource_frontier. "
            "tdvp_sweeps=n splits each two-qubit gate into n equal-angle substeps."
        ),
    }


def config_hash() -> str:
    payload = json.dumps(production_config(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_key(*, chi: int, substeps: int) -> str:
    return f"chi{chi}_n{substeps}_{config_hash()}"
