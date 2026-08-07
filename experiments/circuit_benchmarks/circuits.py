# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Circuit definitions for the four publication benchmark trajectories.

The dense reference and every MPS method consume the same ordered gate list.
The schedules therefore compare tensor-network update error, not Trotter error.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from .config import DT, HEISENBERG_J, ISING_G, ISING_J, N_STEPS, BenchmarkCase

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qiskit.dagcircuit import DAGOpNode

GateName = Literal["rx", "rxx", "ryy", "rzz"]
Edge = tuple[int, int]


@dataclass(frozen=True)
class GateOp:
    """One gate in the fixed Trotter ordering."""

    name: GateName
    qubits: tuple[int, ...]
    theta: float

    def __post_init__(self) -> None:
        expected_arity = 1 if self.name == "rx" else 2
        if len(self.qubits) != expected_arity:
            msg = f"{self.name} requires {expected_arity} qubits."
            raise ValueError(msg)
        if len(set(self.qubits)) != len(self.qubits):
            msg = "A gate cannot act twice on the same qubit."
            raise ValueError(msg)

    def append_to(self, circuit: QuantumCircuit) -> None:
        """Append this operation to a Qiskit circuit."""
        getattr(circuit, self.name)(float(self.theta), *self.qubits)

    def to_dag_node(self, n_qubits: int) -> DAGOpNode:
        """Compile this operation to the node consumed by YAQS gate updates."""
        circuit = QuantumCircuit(n_qubits)
        self.append_to(circuit)
        return next(iter(circuit_to_dag(circuit).topological_op_nodes()))


@dataclass(frozen=True)
class EdgeMatching:
    """Disjoint physical edges that form one interaction layer."""

    label: str
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        sites = [site for edge in self.edges for site in edge]
        if len(sites) != len(set(sites)):
            msg = f"Edge layer {self.label!r} is not a matching."
            raise ValueError(msg)


@dataclass(frozen=True)
class GateLayer:
    """One exactly commuting gate group in a product-formula step."""

    label: str
    time_fraction: float
    gates: tuple[GateOp, ...]


@dataclass(frozen=True)
class TrotterStep:
    """One symmetric second-order Trotter step."""

    index: int
    layers: tuple[GateLayer, ...]

    @property
    def gates(self) -> tuple[GateOp, ...]:
        """Return the step as the exact ordered gate sequence."""
        return tuple(gate for layer in self.layers for gate in layer.gates)


def snake_index(row: int, col: int, num_cols: int) -> int:
    """Map a physical grid coordinate to the snaking MPS chain index."""
    if row < 0 or col < 0 or col >= num_cols:
        msg = f"Invalid grid coordinate ({row}, {col}) for {num_cols} columns."
        raise ValueError(msg)
    if row % 2 == 0:
        return row * num_cols + col
    return row * num_cols + (num_cols - 1 - col)


def edge_matchings(case: BenchmarkCase) -> tuple[EdgeMatching, ...]:
    """Return a deterministic edge coloring of the open interaction graph."""
    if case.geometry == "1d":
        even = tuple((site, site + 1) for site in range(0, case.n_qubits - 1, 2))
        odd = tuple((site, site + 1) for site in range(1, case.n_qubits - 1, 2))
        return (EdgeMatching("even", even), EdgeMatching("odd", odd))

    horizontal_even: list[Edge] = []
    horizontal_odd: list[Edge] = []
    vertical_even: list[Edge] = []
    vertical_odd: list[Edge] = []
    for row in range(case.rows):
        for col in range(case.cols - 1):
            edge = tuple(sorted((snake_index(row, col, case.cols), snake_index(row, col + 1, case.cols))))
            (horizontal_even if col % 2 == 0 else horizontal_odd).append(edge)
    for col in range(case.cols):
        for row in range(case.rows - 1):
            edge = tuple(sorted((snake_index(row, col, case.cols), snake_index(row + 1, col, case.cols))))
            (vertical_even if row % 2 == 0 else vertical_odd).append(edge)
    return (
        EdgeMatching("horizontal_even", tuple(horizontal_even)),
        EdgeMatching("horizontal_odd", tuple(horizontal_odd)),
        EdgeMatching("vertical_even", tuple(vertical_even)),
        EdgeMatching("vertical_odd", tuple(vertical_odd)),
    )


def physical_edges(case: BenchmarkCase) -> tuple[Edge, ...]:
    """Return every open-boundary nearest-neighbor edge exactly once."""
    return tuple(edge for matching in edge_matchings(case) for edge in matching.edges)


def _gate_layer(
    label: str,
    gate_name: GateName,
    edges: Sequence[Edge],
    theta: float,
    time_fraction: float,
    *,
    reverse: bool = False,
) -> GateLayer:
    ordered_edges = tuple(reversed(edges)) if reverse else tuple(edges)
    gates = tuple(GateOp(gate_name, edge, theta) for edge in ordered_edges)
    return GateLayer(label, time_fraction, gates)


def _ising_layers(case: BenchmarkCase, dt: float) -> tuple[GateLayer, ...]:
    bond_angle = -2.0 * ISING_J * dt
    field_angle = -2.0 * ISING_G * dt
    edges = physical_edges(case)
    forward = _gate_layer("ising_bonds", "rzz", edges, bond_angle / 2.0, 0.5)
    field = GateLayer(
        "transverse_field",
        1.0,
        tuple(GateOp("rx", (site,), field_angle) for site in range(case.n_qubits)),
    )
    backward = _gate_layer(
        "ising_bonds", "rzz", edges, bond_angle / 2.0, 0.5, reverse=True
    )
    return (forward, field, backward)


def _heisenberg_layers(case: BenchmarkCase, dt: float) -> tuple[GateLayer, ...]:
    full_angle = -2.0 * HEISENBERG_J * dt
    groups = tuple(
        (f"{gate_name}:{matching.label}", gate_name, matching.edges)
        for gate_name in ("rzz", "rxx", "ryy")
        for matching in edge_matchings(case)
    )
    if not groups:
        return ()

    half_layers = tuple(
        _gate_layer(label, gate_name, edges, full_angle / 2.0, 0.5)
        for label, gate_name, edges in groups[:-1]
    )
    center_label, center_name, center_edges = groups[-1]
    center = _gate_layer(center_label, center_name, center_edges, full_angle, 1.0)
    backward = tuple(
        _gate_layer(label, gate_name, edges, full_angle / 2.0, 0.5, reverse=True)
        for label, gate_name, edges in reversed(groups[:-1])
    )
    return (*half_layers, center, *backward)


def build_trotter_step(case: BenchmarkCase, *, index: int = 0, dt: float = DT) -> TrotterStep:
    """Build one time-reversal-symmetric second-order product-formula step."""
    if dt <= 0.0:
        msg = "dt must be positive."
        raise ValueError(msg)
    layers = _ising_layers(case, dt) if case.model == "ising" else _heisenberg_layers(case, dt)
    return TrotterStep(index=index, layers=layers)


def build_schedule(
    case: BenchmarkCase, *, steps: int = N_STEPS, dt: float = DT
) -> tuple[TrotterStep, ...]:
    """Build the complete, fixed gate schedule for one benchmark case."""
    if steps < 1:
        msg = "steps must be at least one."
        raise ValueError(msg)
    return tuple(build_trotter_step(case, index=index, dt=dt) for index in range(steps))


def step_qiskit_circuit(step: TrotterStep, n_qubits: int) -> QuantumCircuit:
    """Materialize one ordered Trotter step as a Qiskit circuit."""
    circuit = QuantumCircuit(n_qubits)
    for gate in step.gates:
        gate.append_to(circuit)
    return circuit


def build_qiskit_circuit(
    case: BenchmarkCase,
    *,
    schedule: Sequence[TrotterStep] | None = None,
    steps: int = N_STEPS,
    dt: float = DT,
) -> QuantumCircuit:
    """Materialize the exact schedule used by all comparison methods."""
    selected = build_schedule(case, steps=steps, dt=dt) if schedule is None else tuple(schedule)
    circuit = QuantumCircuit(case.n_qubits)
    for step in selected:
        circuit.compose(step_qiskit_circuit(step, case.n_qubits), inplace=True)
    return circuit


def circuit_fingerprint(case: BenchmarkCase, schedule: Sequence[TrotterStep]) -> str:
    """Return a stable SHA256 fingerprint of the physical protocol and gate order."""
    payload = {
        "case": {
            "key": case.key,
            "model": case.model,
            "geometry": case.geometry,
            "rows": case.rows,
            "cols": case.cols,
            "initial_state": case.initial_state,
        },
        "steps": [
            {
                "index": step.index,
                "layers": [
                    {
                        "label": layer.label,
                        "time_fraction": layer.time_fraction,
                        "gates": [
                            {"name": gate.name, "qubits": gate.qubits, "theta": gate.theta}
                            for gate in layer.gates
                        ],
                    }
                    for layer in step.layers
                ],
            }
            for step in schedule
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
