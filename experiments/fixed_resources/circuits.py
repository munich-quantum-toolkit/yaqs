# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Second-order 2D Trotter schedules with snake MPS ordering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from config import DT, HEISENBERG_H, HEISENBERG_J, ISING_H, ISING_J, NUM_COLS, NUM_ROWS
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag

GateName = Literal["rx", "rz", "rxx", "ryy", "rzz"]


@dataclass(frozen=True)
class GateOp:
    """One circuit gate in schedule order."""

    name: GateName
    qubits: tuple[int, ...]
    theta: float

    def to_dag_node(self, length: int):
        qc = QuantumCircuit(length)
        if len(self.qubits) == 1:
            getattr(qc, self.name)(self.theta, self.qubits[0])
        else:
            getattr(qc, self.name)(self.theta, self.qubits[0], self.qubits[1])
        return next(iter(circuit_to_dag(qc).topological_op_nodes()))


@dataclass(frozen=True)
class TrotterStep:
    """Gates for one complete second-order Trotter step."""

    index: int
    gates: tuple[GateOp, ...]


def site_index(row: int, col: int, *, num_cols: int = NUM_COLS) -> int:
    if row % 2 == 0:
        return row * num_cols + col
    return row * num_cols + (num_cols - 1 - col)


def _ising_bond_gates(beta: float, *, num_rows: int = NUM_ROWS, num_cols: int = NUM_COLS) -> list[GateOp]:
    gates: list[GateOp] = []
    for row in range(num_rows):
        for col in range(0, num_cols - 1, 2):
            q1, q2 = site_index(row, col, num_cols=num_cols), site_index(row, col + 1, num_cols=num_cols)
            gates.append(GateOp("rzz", (q1, q2), beta))
        for col in range(1, num_cols - 1, 2):
            q1, q2 = site_index(row, col, num_cols=num_cols), site_index(row, col + 1, num_cols=num_cols)
            gates.append(GateOp("rzz", (q1, q2), beta))
    for col in range(num_cols):
        for row in range(0, num_rows - 1, 2):
            q1, q2 = site_index(row, col, num_cols=num_cols), site_index(row + 1, col, num_cols=num_cols)
            gates.append(GateOp("rzz", (q1, q2), beta))
        for row in range(1, num_rows - 1, 2):
            q1, q2 = site_index(row, col, num_cols=num_cols), site_index(row + 1, col, num_cols=num_cols)
            gates.append(GateOp("rzz", (q1, q2), beta))
    return gates


def _ising_field_gates(alpha: float, *, num_rows: int = NUM_ROWS, num_cols: int = NUM_COLS) -> list[GateOp]:
    return [
        GateOp("rx", (site_index(row, col, num_cols=num_cols),), alpha)
        for row in range(num_rows)
        for col in range(num_cols)
    ]


def _heisenberg_field_gates(
    theta_z: float, *, num_rows: int = NUM_ROWS, num_cols: int = NUM_COLS
) -> list[GateOp]:
    if abs(theta_z) < 1e-15:
        return []
    return [
        GateOp("rz", (site_index(row, col, num_cols=num_cols),), theta_z)
        for row in range(num_rows)
        for col in range(num_cols)
    ]


def _heisenberg_bond_gates(
    theta_xx: float,
    theta_yy: float,
    theta_zz: float,
    *,
    num_rows: int = NUM_ROWS,
    num_cols: int = NUM_COLS,
) -> list[GateOp]:
    gates: list[GateOp] = []
    for gate_name, theta in (("rzz", theta_zz), ("rxx", theta_xx), ("ryy", theta_yy)):
        for row in range(num_rows):
            for col in range(0, num_cols - 1, 2):
                q1, q2 = site_index(row, col, num_cols=num_cols), site_index(row, col + 1, num_cols=num_cols)
                gates.append(GateOp(gate_name, (q1, q2), theta))
            for col in range(1, num_cols - 1, 2):
                q1, q2 = site_index(row, col, num_cols=num_cols), site_index(row, col + 1, num_cols=num_cols)
                gates.append(GateOp(gate_name, (q1, q2), theta))
        for col in range(num_cols):
            for row in range(0, num_rows - 1, 2):
                q1, q2 = site_index(row, col, num_cols=num_cols), site_index(row + 1, col, num_cols=num_cols)
                gates.append(GateOp(gate_name, (q1, q2), theta))
            for row in range(1, num_rows - 1, 2):
                q1, q2 = site_index(row, col, num_cols=num_cols), site_index(row + 1, col, num_cols=num_cols)
                gates.append(GateOp(gate_name, (q1, q2), theta))
    return gates


def build_ising_schedule(
    *, timesteps: int, num_rows: int = NUM_ROWS, num_cols: int = NUM_COLS
) -> tuple[TrotterStep, ...]:
    """Second-order Suzuki–Trotter: bond(dt/2), field(dt), bond(dt/2)."""
    alpha = -2.0 * DT * ISING_H
    beta = -2.0 * DT * ISING_J
    half = beta / 2.0
    steps: list[TrotterStep] = []
    for idx in range(timesteps):
        gates = (
            _ising_bond_gates(half, num_rows=num_rows, num_cols=num_cols)
            + _ising_field_gates(alpha, num_rows=num_rows, num_cols=num_cols)
            + _ising_bond_gates(half, num_rows=num_rows, num_cols=num_cols)
        )
        steps.append(TrotterStep(index=idx, gates=tuple(gates)))
    return tuple(steps)


def build_heisenberg_schedule(
    *, timesteps: int, num_rows: int = NUM_ROWS, num_cols: int = NUM_COLS
) -> tuple[TrotterStep, ...]:
    """Second-order Suzuki–Trotter for isotropic Heisenberg (Jxx=Jyy=Jzz=J, h=0)."""
    theta = -2.0 * DT * HEISENBERG_J
    theta_z = -2.0 * DT * HEISENBERG_H
    half = theta / 2.0
    steps: list[TrotterStep] = []
    for idx in range(timesteps):
        gates = (
            _heisenberg_bond_gates(half, half, half, num_rows=num_rows, num_cols=num_cols)
            + _heisenberg_field_gates(theta_z, num_rows=num_rows, num_cols=num_cols)
            + _heisenberg_bond_gates(half, half, half, num_rows=num_rows, num_cols=num_cols)
        )
        steps.append(TrotterStep(index=idx, gates=tuple(gates)))
    return tuple(steps)


def build_qiskit_circuit(
    model: str, *, timesteps: int, num_rows: int = NUM_ROWS, num_cols: int = NUM_COLS
) -> QuantumCircuit:
    """Materialize the full schedule as a Qiskit circuit for cross-checks."""
    schedule = (
        build_ising_schedule(timesteps=timesteps, num_rows=num_rows, num_cols=num_cols)
        if model == "ising"
        else build_heisenberg_schedule(timesteps=timesteps, num_rows=num_rows, num_cols=num_cols)
    )
    qc = QuantumCircuit(num_rows * num_cols)
    for step in schedule:
        for gate in step.gates:
            if len(gate.qubits) == 1:
                getattr(qc, gate.name)(gate.theta, gate.qubits[0])
            else:
                getattr(qc, gate.name)(gate.theta, gate.qubits[0], gate.qubits[1])
    return qc


def neel_basis_string(*, num_rows: int = NUM_ROWS, num_cols: int = NUM_COLS) -> str:
    """Checkerboard Néel product state in snake MPS order."""
    chain = ["0"] * (num_rows * num_cols)
    for row in range(num_rows):
        for col in range(num_cols):
            q = site_index(row, col, num_cols=num_cols)
            chain[q] = "0" if (row + col) % 2 == 0 else "1"
    return "".join(chain)
