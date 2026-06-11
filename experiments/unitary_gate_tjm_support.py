# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small experiment for native Qiskit UnitaryGate support in digital TJM."""

from __future__ import annotations

import json
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


def dense_two_qubit_unitary() -> np.ndarray:
    """Build a deterministic dense two-qubit unitary.

    Returns:
        Dense ``4x4`` unitary matrix.
    """
    generator = np.array(
        [
            [0.2, 0.1 + 0.3j, -0.2j, 0.4],
            [0.1 - 0.3j, -0.1, 0.25, 0.05j],
            [0.2j, 0.25, 0.3, -0.15 + 0.1j],
            [0.4, -0.05j, -0.15 - 0.1j, -0.4],
        ],
        dtype=np.complex128,
    )
    return np.asarray(expm(-1j * generator), dtype=np.complex128)


def yaqs_statevector(circuit: QuantumCircuit, gate_mode: GateMode) -> np.ndarray:
    """Run YAQS and return the final statevector.

    Args:
        circuit: Circuit to simulate.
        gate_mode: Digital two-qubit gate mode.

    Returns:
        Final statevector from the YAQS MPS output state.
    """
    params = StrongSimParams(
        observables=[Observable(Z(), 0)],
        gate_mode=gate_mode,
        preset="exact",
        svd_threshold=1e-12,
        max_bond_dim=None,
        get_state=True,
    )
    state = State(circuit.num_qubits, initial="zeros")
    result = Simulator(parallel=False, show_progress=False).run(state, circuit, params)
    assert result.output_state is not None
    return result.output_state.mps.to_vec()


def fidelity(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return squared state overlap fidelity."""
    return float(abs(np.vdot(reference, candidate)) ** 2)


def run_case(circuit: QuantumCircuit, gate_mode: str) -> float:
    """Run one comparison case against Qiskit.

    Args:
        circuit: Circuit to compare.
        gate_mode: Digital two-qubit gate mode.

    Returns:
        Squared state overlap fidelity.
    """
    reference = np.asarray(Statevector(circuit).data, dtype=np.complex128)
    candidate = yaqs_statevector(circuit, cast("GateMode", gate_mode))
    return fidelity(reference, candidate)


def main() -> None:
    """Run the experiment and write exact-agreement diagnostics."""
    unitary = dense_two_qubit_unitary()

    nearest = QuantumCircuit(2)
    nearest.h(0)
    nearest.append(UnitaryGate(unitary), [0, 1])
    nearest.rz(0.17, 1)

    long_range = QuantumCircuit(3)
    long_range.rx(0.23, 1)
    long_range.append(UnitaryGate(unitary), [0, 2])
    long_range.ry(-0.31, 0)

    results = {
        "nearest_dense_log_tdvp_fidelity": run_case(nearest, "tdvp"),
        "nearest_full_dense_log_tdvp_fidelity": run_case(nearest, "full-tdvp"),
        "nearest_mpo_fidelity": run_case(nearest, "mpo"),
        "long_range_dense_log_tdvp_fidelity": run_case(long_range, "tdvp"),
        "long_range_full_dense_log_tdvp_fidelity": run_case(long_range, "full-tdvp"),
        "long_range_mpo_fidelity": run_case(long_range, "mpo"),
    }
    results["all_close_to_one"] = all(np.isclose(value, 1.0, atol=1e-10) for value in results.values())

    output_path = Path(__file__).with_name("unitary_gate_tjm_support.json")
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
