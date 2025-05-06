# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the quantum circuit generation functions in the YAQS circuit library.

This module contains tests for verifying the correctness of quantum circuits generated
by the functions `create_ising_circuit` and `create_Heisenberg_circuit` from the YAQS circuit library.

The tests ensure:
- Circuits have the correct number of qubits (both even and odd cases).
- Circuits contain the expected quantum gates (e.g., rx, rz, rzz, rxx, ryy).
- Gate counts match expected values based on circuit parameters (J, g, Jx, Jy, Jz, h, dt, timesteps).

These tests help maintain consistency and correctness of circuit generation used
in quantum simulations within the YAQS project.
"""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit

from mqt.yaqs.core.libraries.circuit_library import (
    create_1d_fermi_hubbard_circuit,
    create_2d_fermi_hubbard_circuit,
    create_2d_heisenberg_circuit,
    create_2d_ising_circuit,
    create_heisenberg_circuit,
    create_ising_circuit,
    nearest_neighbour_random_circuit,
)


def test_create_ising_circuit_valid_even() -> None:
    """Test that create_ising_circuit returns a valid circuit for an even number of qubits.

    This test creates an Ising circuit with L=4 qubits using parameters J, g, dt, and timesteps.
    It verifies that:
      - The resulting object is a QuantumCircuit with 4 qubits.
      - The expected number of rx and rzz gates are present (4 rx gates and 3 rzz gates).
    """
    circ = create_ising_circuit(L=4, J=1, g=0.5, dt=0.1, timesteps=1)

    # Check that the output is a QuantumCircuit with the correct number of qubits.
    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == 4

    # Count the gates by name.
    op_names = [instr.operation.name for instr in circ.data]
    rx_count = op_names.count("rx")
    rzz_count = op_names.count("rzz")

    assert rx_count == 4
    assert rzz_count == 3


def test_create_ising_circuit_valid_odd() -> None:
    """Test that create_ising_circuit returns a valid circuit for an odd number of qubits.

    This test creates an Ising circuit with L=5 qubits and verifies that:
      - The output is a QuantumCircuit with 5 qubits.
      - The counts of rx and rzz gates match the expected numbers for an odd-length circuit
        (5 rx gates and 4 rzz gates).
    """
    circ = create_ising_circuit(L=5, J=1, g=0.5, dt=0.1, timesteps=1)

    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == 5

    # For L=5:
    #   - rx loop adds 5 gates.
    #   - Even-site loop: 5//2 = 2 iterations → adds 4 CX and 2 RZ.
    #   - Odd-site loop: range(1, 5//2) → 1 iteration → adds 2 CX and 1 RZ.
    #   - Extra clause for odd number adds 2 CX and 1 RZ.
    # Total expected: 5 rx, 8 CX, and 4 rzz gates.
    op_names = [instr.operation.name for instr in circ.data]
    rx_count = op_names.count("rx")
    rzz_count = op_names.count("rzz")

    assert rx_count == 5
    assert rzz_count == 4


def test_create_heisenberg_circuit_valid_even() -> None:
    """Test that create_heisenberg_circuit returns a valid circuit for an even number of qubits.

    This test creates a Heisenberg circuit with L=4 qubits using parameters Jx, Jy, Jz, h, dt, and timesteps.
    It verifies that the resulting circuit is a QuantumCircuit with 4 qubits and that it contains the expected gate
    types (rz, rzz, rxx, and ryy).
    """
    circ = create_heisenberg_circuit(L=4, Jx=1, Jy=1, Jz=1, h=0.5, dt=0.1, timesteps=1)

    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == 4

    op_names = [instr.operation.name for instr in circ.data]
    for gate in ["rz", "rzz", "rxx", "ryy"]:
        assert gate in op_names, f"Gate {gate} not found in the circuit."


def test_create_heisenberg_circuit_valid_odd() -> None:
    """Test that create_heisenberg_circuit returns a valid circuit for an odd number of qubits.

    This test creates a Heisenberg circuit with L=5 qubits using parameters Jx, Jy, Jz, h, dt, and timesteps.
    It verifies that the resulting circuit is a QuantumCircuit with 5 qubits and that it contains the expected gate
    types (rz, rzz, rxx, and ryy).
    """
    circ = create_heisenberg_circuit(L=5, Jx=1, Jy=1, Jz=1, h=0.5, dt=0.1, timesteps=1)

    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == 5

    op_names = [instr.operation.name for instr in circ.data]
    for gate in ["rz", "rzz", "rxx", "ryy"]:
        assert gate in op_names, f"Gate {gate} not found in the circuit."


def test_create_ising_circuit_periodic_even() -> None:
    """Test that create_ising_circuit returns a valid periodic circuit for an even number of qubits.

    This test creates an Ising circuit with L=4 qubits and periodic boundary conditions.
    It verifies that:
      - The circuit is a QuantumCircuit with 4 qubits.
      - The additional long-range rzz gate between the first and last qubit is present.
      - The gate counts are adjusted accordingly (4 rx gates and 4 rzz gates).
    """
    circ = create_ising_circuit(L=4, J=1, g=0.5, dt=0.1, timesteps=1, periodic=True)

    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == 4

    op_names = [instr.operation.name for instr in circ.data]
    rx_count = op_names.count("rx")
    rzz_count = op_names.count("rzz")

    # Without periodic: 3 rzz gates for L=4, periodic adds one more.
    assert rx_count == 4
    assert rzz_count == 4


def test_create_ising_circuit_periodic_odd() -> None:
    """Test that create_ising_circuit returns a valid periodic circuit for an odd number of qubits.

    This test creates an Ising circuit with L=5 qubits and periodic boundary conditions.
    It verifies that:
      - The circuit is a QuantumCircuit with 5 qubits.
      - The additional long-range rzz gate between the first and last qubit is present.
      - The gate counts are adjusted accordingly (5 rx gates and 5 rzz gates).
    """
    circ = create_ising_circuit(L=5, J=1, g=0.5, dt=0.1, timesteps=1, periodic=True)

    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == 5

    op_names = [instr.operation.name for instr in circ.data]
    rx_count = op_names.count("rx")
    rzz_count = op_names.count("rzz")

    # Without periodic: 4 rzz gates for L=5, periodic adds one more.
    assert rx_count == 5
    assert rzz_count == 5


def test_create_2d_ising_circuit_2x3() -> None:
    """Test that create_2d_ising_circuit returns a valid circuit for a rectangular grid.

    This test creates a 2D Ising circuit for a grid with a specified number of rows and columns.
    It verifies that:
      - The circuit is a QuantumCircuit with total qubits equal to num_rows * num_cols.
      - Each qubit receives an rx gate in every timestep.
      - The circuit contains rzz gates for the interactions.
    """
    num_rows = 2
    num_cols = 3
    total_qubits = num_rows * num_cols
    circ = create_2d_ising_circuit(num_rows, num_cols, J=1, g=0.5, dt=0.1, timesteps=1)

    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == total_qubits

    op_names = [instr.operation.name for instr in circ.data]
    # Check that each qubit gets one rx rotation in the single timestep.
    assert op_names.count("rx") == total_qubits
    # Check that rzz gates are present.
    assert "rzz" in op_names


def test_create_2d_ising_circuit_3x2() -> None:
    """Test that create_2d_ising_circuit returns a valid circuit for a rectangular grid.

    This test creates a 2D Ising circuit for a grid with a specified number of rows and columns.
    It verifies that:
      - The circuit is a QuantumCircuit with total qubits equal to num_rows * num_cols.
      - Each qubit receives an rx gate in every timestep.
      - The circuit contains rzz gates for the interactions.
    """
    num_rows = 3
    num_cols = 2
    total_qubits = num_rows * num_cols
    circ = create_2d_ising_circuit(num_rows, num_cols, J=1, g=0.5, dt=0.1, timesteps=1)

    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == total_qubits

    op_names = [instr.operation.name for instr in circ.data]
    # Check that each qubit gets one rx rotation in the single timestep.
    assert op_names.count("rx") == total_qubits
    # Check that rzz gates are present.
    assert "rzz" in op_names


def test_create_2d_heisenberg_circuit_2x3() -> None:
    """Test that create_2d_heisenberg_circuit returns a valid circuit for a rectangular grid.

    This test creates a 2D Heisenberg circuit for a grid with a specified number of rows and columns.
    It verifies that:
      - The circuit is a QuantumCircuit with total qubits equal to num_rows * num_cols.
      - Each qubit receives an rx gate in every timestep.
      - The circuit contains rxx gates for the interactions.
      - The circuit contains ryy gates for the interactions.
      - The circuit contains rzz gates for the interactions.
    """
    num_rows = 2
    num_cols = 3
    total_qubits = num_rows * num_cols
    circ = create_2d_heisenberg_circuit(num_rows, num_cols, Jx=1, Jy=1, Jz=1, h=0.5, dt=0.1, timesteps=1)

    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == total_qubits

    op_names = [instr.operation.name for instr in circ.data]
    # Check that each qubit gets one rx rotation in the single timestep.
    assert op_names.count("rz") == total_qubits
    assert "rxx" in op_names
    assert "ryy" in op_names
    assert "rzz" in op_names


def test_create_2d_heisenberg_circuit_3x2() -> None:
    """Test that create_2d_heisenberg_circuit returns a valid circuit for a rectangular grid.

    This test creates a 2D Heisenberg circuit for a grid with a specified number of rows and columns.
    It verifies that:
      - The circuit is a QuantumCircuit with total qubits equal to num_rows * num_cols.
      - Each qubit receives an rx gate in every timestep.
      - The circuit contains rxx gates for the interactions.
      - The circuit contains ryy gates for the interactions.
      - The circuit contains rzz gates for the interactions.
    """
    num_rows = 3
    num_cols = 2
    total_qubits = num_rows * num_cols
    circ = create_2d_heisenberg_circuit(num_rows, num_cols, Jx=1, Jy=1, Jz=1, h=0.5, dt=0.1, timesteps=1)

    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == total_qubits

    op_names = [instr.operation.name for instr in circ.data]
    # Check that each qubit gets one rx rotation in the single timestep.
    assert op_names.count("rz") == total_qubits
    assert "rxx" in op_names
    assert "ryy" in op_names
    assert "rzz" in op_names


def test_create_2d_fermi_hubbard_circuit_3x2() -> None:
    """Test that create_2d_fermi_hubbard_circuit returns a valid circuit for a rectangular grid.

    This test creates a 2D Fermi-Hubbard circuit for a grid with a specified number of rows and columns.
    It verifies that:
      - The circuit is a QuantumCircuit with total qubits equal to 2 * num_rows * num_cols.
      - The circuit contains phase gates for the chemical potential term.
      - The number of phase gates is equal to twice the number of qubits.
      - The circuit contains controlled phase gates for the onsite term.
      - The circuit contains rx gates for the long-range interactions.
      - The circuit contains ry gates for the long-range interactions.
      - The circuit contains rz gates for the long-range interactions.
    """
    num_rows = 3
    num_cols = 2
    total_qubits = 2 * num_rows * num_cols
    circ = create_2d_fermi_hubbard_circuit(
        num_rows, num_cols, u=0.5, t=1.0, mu=0.5, num_trotter_steps=1, dt=0.1, timesteps=1
    )

    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == total_qubits

    op_names = [instr.operation.name for instr in circ.data]
    # Check that the phase gates from the chemical potential term
    # are present for every timestep and trotter step (two per trotter step)
    assert op_names.count("p") == 2 * total_qubits
    # Check that the controlled phase gates from the onsite interaction term
    # are present for every timestep and trotter step (two per trotter step)
    assert op_names.count("cp") == total_qubits
    # Check that the rotation gates from the hopping terms are present
    assert "rx" in op_names
    assert "ry" in op_names
    assert "rz" in op_names


def test_create_2d_fermi_hubbard_circuit_3x2() -> None:
    """Test that create_1d_fermi_hubbard_circuit returns a valid circuit.

    This test creates a 1D Fermi-Hubbard circuit with a specified number lattice sites.
    It verifies that:
      - The circuit is a QuantumCircuit with total qubits equal to 2 * num_sites.
      - The circuit contains phase gates for the chemical potential term.
      - The number of phase gates is equal to twice the number of qubits.
      - The circuit contains controlled phase gates for the onsite term.
      - The circuit contains rxx gates for the hopping term.
      - The circuit contains ryy gates for the hopping term.
    """
    num_sites = 4
    total_qubits = 2 * num_sites
    circ = create_1d_fermi_hubbard_circuit(
        num_sites, u=0.5, t=1.0, mu=0.5, num_trotter_steps=1, dt=0.1, timesteps=1
    )

    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == total_qubits

    op_names = [instr.operation.name for instr in circ.data]
    # Check that the phase gates from the chemical potential term
    # are present for every timestep and trotter step (two per trotter step)
    assert op_names.count("p") == 2 * total_qubits
    # Check that the controlled phase gates from the onsite interaction term
    # are present for every timestep and trotter step (two per trotter step)
    assert op_names.count("cp") == total_qubits
    # Check that the rotation gates from the hopping terms are present
    op_names.count("rxx") == 2 * (num_sites - 1)
    op_names.count("rzz") == 2 * (num_sites - 1)


def test_nearest_neighbour_random_circuit_structure() -> None:
    """Nearest-neighbour random circuit: correct qubit count, gate counts and barriers."""
    n_qubits = 3
    layers = 2
    qc = nearest_neighbour_random_circuit(n_qubits, layers, seed=42)

    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == n_qubits

    names = [instr.operation.name for instr in qc.data]
    # one U-gate per qubit per layer
    assert names.count("u") == n_qubits * layers
    # one barrier at end of each layer
    assert names.count("barrier") == layers

    # count CZ+CX per layer:
    expected_pairs = []
    for layer in range(layers):
        if layer % 2 == 0:
            expected_pairs.append(len([(i, i + 1) for i in range(1, n_qubits - 1, 2)]))
        else:
            expected_pairs.append(len([(i, i + 1) for i in range(0, n_qubits - 1, 2)]))
    assert names.count("cz") + names.count("cx") == sum(expected_pairs)


def test_nearest_seed_reproducibility() -> None:
    """Circuits generated with the same seed must be identical."""
    qc1 = nearest_neighbour_random_circuit(4, 3, seed=123)
    qc2 = nearest_neighbour_random_circuit(4, 3, seed=123)
    # comparing OpenQASM is a quick way to verify structural equality
    assert qc1 == qc2
