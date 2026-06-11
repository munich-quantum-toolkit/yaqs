# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Stress benchmark for dense-log TDVP generic UnitaryGate paths."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.simulation_parameters import GateMode, Observable, StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import Z


@dataclass(frozen=True)
class BenchmarkCase:
    """Benchmark circuit specification."""

    name: str
    num_qubits: int
    depth: int
    long_range: bool
    mixed_neighbors: bool = False
    reversed_neighbor: bool = False


def random_dense_two_qubit_unitary(seed: int, scale: float = 0.35) -> np.ndarray:
    """Build a deterministic dense two-qubit unitary.

    Args:
        seed: Random seed for the Hermitian generator.
        scale: Overall generator scale.

    Returns:
        Dense ``4x4`` unitary matrix.
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    generator = scale * (raw + raw.conj().T) / 2
    return np.asarray(expm(-1j * generator), dtype=np.complex128)


def build_circuit(case: BenchmarkCase) -> QuantumCircuit:
    """Build a deterministic stress circuit.

    Args:
        case: Benchmark specification.

    Returns:
        Qiskit circuit containing generic ``UnitaryGate`` operations.
    """
    qc = QuantumCircuit(case.num_qubits)
    for layer in range(case.depth):
        for qubit in range(case.num_qubits):
            qc.rx(0.03 * (layer + 1) * (qubit + 1), qubit)
            qc.rz(-0.02 * (layer + 1) * (qubit + 1), qubit)

        if case.long_range:
            qc.append(UnitaryGate(random_dense_two_qubit_unitary(1000 + layer)), [0, case.num_qubits - 1])
            if case.num_qubits > 3:
                qc.append(UnitaryGate(random_dense_two_qubit_unitary(2000 + layer)), [1, case.num_qubits - 2])
            if case.mixed_neighbors:
                neighbor_sites = (
                    [case.num_qubits - 1, case.num_qubits - 2]
                    if case.reversed_neighbor
                    else [case.num_qubits - 2, case.num_qubits - 1]
                )
                qc.append(
                    UnitaryGate(random_dense_two_qubit_unitary(5000 + layer)),
                    neighbor_sites,
                )
        else:
            for left in range(0, case.num_qubits - 1, 2):
                qc.append(UnitaryGate(random_dense_two_qubit_unitary(3000 + 10 * layer + left)), [left, left + 1])
            for left in range(1, case.num_qubits - 1, 2):
                qc.append(UnitaryGate(random_dense_two_qubit_unitary(4000 + 10 * layer + left)), [left, left + 1])
    return qc


def run_yaqs(circuit: QuantumCircuit, gate_mode: GateMode) -> np.ndarray:
    """Run YAQS and return the final statevector.

    Args:
        circuit: Circuit to simulate.
        gate_mode: Digital two-qubit gate mode.

    Returns:
        Final statevector from YAQS.
    """
    params = StrongSimParams(
        observables=[Observable(Z(), 0)],
        gate_mode=gate_mode,
        preset="exact",
        svd_threshold=1e-12,
        max_bond_dim=None,
        get_state=True,
    )
    result = Simulator(parallel=False, show_progress=False).run(
        State(circuit.num_qubits, initial="zeros"), circuit, params
    )
    assert result.output_state is not None
    return result.output_state.mps.to_vec()


def fidelity(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return squared state overlap fidelity."""
    return float(abs(np.vdot(reference, candidate)) ** 2)


def benchmark_case(case: BenchmarkCase, repeats: int = 3) -> dict[str, object]:
    """Benchmark one circuit case.

    Args:
        case: Benchmark specification.
        repeats: Number of timing repeats per method.

    Returns:
        Timing and fidelity metrics for hybrid and full dense-log TDVP.
    """
    circuit = build_circuit(case)
    reference = np.asarray(Statevector(circuit).data, dtype=np.complex128)
    metrics: dict[str, object] = {
        "num_qubits": case.num_qubits,
        "depth": case.depth,
        "long_range": case.long_range,
        "mixed_neighbors": case.mixed_neighbors,
        "reversed_neighbor": case.reversed_neighbor,
        "num_unitary_gates": sum(1 for instruction in circuit.data if instruction.operation.name == "unitary"),
    }

    for label, gate_mode in [("dense_log", "tdvp"), ("full_dense_log", "full-tdvp")]:
        timings = []
        fidelities = []
        for _ in range(repeats):
            start = time.perf_counter()
            candidate = run_yaqs(circuit, cast("GateMode", gate_mode))
            timings.append(time.perf_counter() - start)
            fidelities.append(fidelity(reference, candidate))
        metrics[label] = {
            "median_seconds": float(np.median(timings)),
            "min_fidelity": float(np.min(fidelities)),
            "max_infidelity": float(1.0 - np.min(fidelities)),
        }

    dense_time = cast("dict[str, float]", metrics["dense_log"])["median_seconds"]
    full_dense_time = cast("dict[str, float]", metrics["full_dense_log"])["median_seconds"]
    metrics["full_dense_log_vs_hybrid_tdvp_speedup"] = dense_time / full_dense_time
    return metrics


def main() -> None:
    """Run all benchmark cases and save a JSON report."""
    cases = [
        BenchmarkCase("nearest_4q_depth4", num_qubits=4, depth=4, long_range=False),
        BenchmarkCase("long_range_4q_depth4", num_qubits=4, depth=4, long_range=True),
        BenchmarkCase("mixed_4q_depth4", num_qubits=4, depth=4, long_range=True, mixed_neighbors=True),
        BenchmarkCase(
            "mixed_reversed_4q_depth4",
            num_qubits=4,
            depth=4,
            long_range=True,
            mixed_neighbors=True,
            reversed_neighbor=True,
        ),
        BenchmarkCase("long_range_6q_depth3", num_qubits=6, depth=3, long_range=True),
        BenchmarkCase("mixed_6q_depth3", num_qubits=6, depth=3, long_range=True, mixed_neighbors=True),
    ]
    results = {case.name: benchmark_case(case) for case in cases}
    output_path = Path(__file__).with_name("unitary_gate_tdvp_stress_benchmark.json")
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
