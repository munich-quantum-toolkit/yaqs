# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import pytest
from qiskit.circuit import QuantumCircuit

from mqt.yaqs.core.libraries.circuit_library import create_Heisenberg_circuit, create_Ising_circuit


def test_create_Ising_circuit_valid_even() -> None:
    # Use an even number of qubits.
    circ = create_Ising_circuit(L=4, J=1, g=0.5, dt=0.1, timesteps=1)

    # Check that the output is a QuantumCircuit with the right number of qubits.
    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == 4

    # Count the gates by name.
    op_names = [instr.operation.name for instr in circ.data]
    rx_count = op_names.count("rx")
    rzz_count = op_names.count("rzz")

    assert rx_count == 4
    assert rzz_count == 3


def test_create_Ising_circuit_valid_odd() -> None:
    # Use an odd number of qubits.
    circ = create_Ising_circuit(L=5, J=1, g=0.5, dt=0.1, timesteps=1)

    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == 5

    # For L=5:
    #   - The rx loop adds 5 gates.
    #   - The even-site loop: 5//2 = 2 iterations → 2*2 = 4 CX and 2 RZ.
    #   - The odd-site loop: range(1, 5//2) → 1 iteration → 2 CX and 1 RZ.
    #   - The extra clause (since 5 is odd and not 1) adds 2 CX and 1 RZ.
    # Total: 5 rx, (4+2+2)=8 CX, and (2+1+1)=4 RZ.
    op_names = [instr.operation.name for instr in circ.data]
    rx_count = op_names.count("rx")
    rzz_count = op_names.count("rzz")

    assert rx_count == 5
    assert rzz_count == 4


def test_create_Heisenberg_circuit_valid_even() -> None:
    # Use an even number of qubits.
    circ = create_Heisenberg_circuit(L=4, Jx=1, Jy=1, Jz=1, h=0.5, dt=0.1, timesteps=1)

    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == 4

    # Check that the circuit contains the expected types of gates.
    op_names = [instr.operation.name for instr in circ.data]
    for gate in ["rz", "rzz", "rxx", "ryy"]:
        assert gate in op_names


def test_create_Heisenberg_circuit_valid_odd() -> None:
    # Use an odd number of qubits.
    circ = create_Heisenberg_circuit(L=5, Jx=1, Jy=1, Jz=1, h=0.5, dt=0.1, timesteps=1)

    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == 5

    op_names = [instr.operation.name for instr in circ.data]
    for gate in ["rz", "rzz", "rxx", "ryy"]:
        assert gate in op_names
