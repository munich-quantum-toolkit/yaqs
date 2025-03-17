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

import numpy as np
import scipy as sp

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator

from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import SquareLattice, BoundaryCondition
from qiskit_nature.second_q.mappers import JordanWignerMapper

from mqt.yaqs.core.libraries.circuit_library import (
    create_2d_heisenberg_circuit,
    create_2d_ising_circuit,
    create_heisenberg_circuit,
    create_ising_circuit,
    create_2D_Fermi_Hubbard_circuit
)

from mqt.yaqs.circuits.reference_implementation.FH_reference import create_Fermi_Hubbard_model_qutip


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


def test_create_2D_Fermi_Hubbard_circuit_equal_qiskit():
    # Define the FH model parameters
    t = 1.0         # kinetic hopping
    mu = 0.5        # chemical potential
    u = 4.0         # onsite interaction
    Lx, Ly = 2, 2   # lattice dimensions
    timesteps = 1
    dt = 0.1
    num_trotter_steps = 10

    # yaqs implementation
    model = {'name': '2D_Fermi_Hubbard', 'Lx': Lx, 'Ly': Ly, 'mu': mu, 'u': u, 't': t, 'num_trotter_steps': num_trotter_steps}
    circuit = create_2D_Fermi_Hubbard_circuit(model, dt=dt, timesteps=timesteps)
    U_yaqs = Operator(circuit).to_matrix()

    # Qiskit implementation
    square_lattice = SquareLattice(rows=Lx, cols=Ly, boundary_condition=BoundaryCondition.OPEN)
    fh_hamiltonian = FermiHubbardModel(
        square_lattice.uniform_parameters(
            uniform_interaction=t,
            uniform_onsite_potential=mu,
        ),
        onsite_interaction=u,
    )
    mapper = JordanWignerMapper()
    qubit_jw_op = mapper.map(fh_hamiltonian.second_q_op())
    H_qiskit = qubit_jw_op.to_matrix()
    U_qiskit = sp.linalg.expm(-1j*dt*timesteps*H_qiskit)

    # Calculate error
    error = np.linalg.norm(U_qiskit - U_yaqs, 2)
    print("|U_qiskit - U_yaqs| = " + str(error))
    assert error <= 10e-3

def test_create_2D_Fermi_Hubbard_circuit_equal_qutip():
    # Define the FH model parameters
    t = 1.0         # kinetic hopping
    mu = 0.5        # chemical potential
    u = 4.0         # onsite interaction
    Lx, Ly = 2, 2   # lattice dimensions
    timesteps = 1
    dt = 0.1
    num_trotter_steps = 10

    # yaqs implementation
    model = {'name': '2D_Fermi_Hubbard', 'Lx': Lx, 'Ly': Ly, 'mu': -mu, 'u': u, 't': -t, 'num_trotter_steps': num_trotter_steps}
    circuit = create_2D_Fermi_Hubbard_circuit(model, dt=dt, timesteps=timesteps)
    U_yaqs = Operator(circuit).to_matrix()

    # QuTiP implementation
    H_qutip = create_Fermi_Hubbard_model_qutip(Lx, Ly, u, -t, mu)
    U_qutip = sp.linalg.expm(-1j*dt*timesteps*H_qutip.full())

    # Calculate error
    error = np.linalg.norm(U_qutip - U_yaqs, 2)
    print("|U_qutip - U_yaqs| = " + str(error))
    assert error <= 10e-3