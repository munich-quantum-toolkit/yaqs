# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Frozen configuration for the circuit and resource benchmark campaign."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Model = Literal["ising", "heisenberg"]
Geometry = Literal["1d", "2d"]
Method = Literal["gate_local_2tdvp", "mpo_contract_compress", "tebd_swap"]


@dataclass(frozen=True)
class BenchmarkCase:
    """Physical model and geometry for one circuit trajectory.

    Attributes:
        key: Stable machine-readable case identifier.
        label: Compact display label.
        model: Hamiltonian family.
        geometry: One-dimensional chain or two-dimensional square lattice.
        rows: Number of physical lattice rows.
        cols: Number of physical lattice columns.
        initial_state: Product-state protocol used for this model.
    """

    key: str
    label: str
    model: Model
    geometry: Geometry
    rows: int
    cols: int
    initial_state: Literal["zeros", "neel"]

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            msg = "Benchmark lattice dimensions must be positive."
            raise ValueError(msg)
        if self.geometry == "1d" and self.rows != 1:
            msg = "The 1D benchmark convention requires rows=1."
            raise ValueError(msg)
        expected_state = "zeros" if self.model == "ising" else "neel"
        if self.initial_state != expected_state:
            msg = f"{self.model} requires initial_state={expected_state!r}."
            raise ValueError(msg)

    @property
    def n_qubits(self) -> int:
        """Return the number of spin sites in the case."""
        return self.rows * self.cols


EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "output"
REPO_ROOT = EXPERIMENT_DIR.parents[1]

CAMPAIGN_ID = "circuit_benchmarks_v1"
N = 16
GRID_ROWS = 4
GRID_COLS = 4

CASES: dict[str, BenchmarkCase] = {
    "ising_1d": BenchmarkCase("ising_1d", "1D Ising", "ising", "1d", 1, N, "zeros"),
    "heisenberg_1d": BenchmarkCase(
        "heisenberg_1d", "1D Heisenberg", "heisenberg", "1d", 1, N, "neel"
    ),
    "ising_2d": BenchmarkCase(
        "ising_2d", "2D Ising", "ising", "2d", GRID_ROWS, GRID_COLS, "zeros"
    ),
    "heisenberg_2d": BenchmarkCase(
        "heisenberg_2d", "2D Heisenberg", "heisenberg", "2d", GRID_ROWS, GRID_COLS, "neel"
    ),
}
CASE_KEYS = tuple(CASES)

# Hamiltonian convention:
#   H_Ising = -J sum_<ij> Z_i Z_j - g sum_i X_i
#   H_Heisenberg = -J sum_<ij> (X_i X_j + Y_i Y_j + Z_i Z_j)
# with open boundaries. Qiskit R_P(theta)=exp(-i theta P/2), hence theta=-2 c dt.
ISING_J = 1.0
ISING_G = 1.0
HEISENBERG_J = 1.0
HEISENBERG_FIELD = 0.0

DT = 0.1
N_STEPS = 30
T_MAX = DT * N_STEPS
TROTTER_ORDER = 2

METHODS: tuple[Method, ...] = (
    "gate_local_2tdvp",
    "mpo_contract_compress",
    "tebd_swap",
)
METHOD_TO_GATE_MODE = {
    "gate_local_2tdvp": "full-tdvp",
    "mpo_contract_compress": "mpo",
    "tebd_swap": "swaps",
}

CHI_MAIN = 32
CHI_GRID = (4, 8, 16, 24, 32, 64, 128, 192, 256)
FRONTIER_CASE_KEY = "ising_2d"
FRONTIER_STEPS = 15
FRONTIER_TARGET_STEPS = (5, 10, 15)

RELIABILITY_THRESHOLD = 1e-2
THRESHOLD_SENSITIVITY = (5e-3, 1e-2, 2e-2)
TDVP_SUBSTEP_CANDIDATES = (1, 2, 4)
TDVP_PRODUCTION_SUBSTEPS = 2
TDVP_RESOLUTION_CASE_STEPS = {"ising_2d": 15, "heisenberg_2d": 1}

SVD_THRESHOLD = 1e-13
KRYLOV_TOL = 1e-12
TRUNC_MODE = "discarded_weight"
TDVP_MODE = "2site"

TIMING_WARMUPS = 1
TIMING_REPEATS = 3


def time_for_step(step: int) -> float:
    """Return physical time after a one-based number of Trotter steps."""
    if step < 0:
        msg = "step must be non-negative."
        raise ValueError(msg)
    return float(step * DT)
