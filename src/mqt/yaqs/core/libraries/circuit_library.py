# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Library of useful quantum circuits.

This module provides functions for creating quantum circuits that simulate
the dynamics of the Ising and Heisenberg models. The functions create_ising_circuit
and create_Heisenberg_circuit construct Qiskit QuantumCircuit objects based on specified
parameters such as the number of qubits, interaction strengths, time steps, and total simulation time.
These circuits are used to simulate the evolution of quantum many-body systems under the
respective Hamiltonians.
"""

from __future__ import annotations

# ignore non-lowercase argument names for physics notation
# ruff: noqa: N803
import numpy as np
from qiskit.circuit import QuantumCircuit

from .circuit_library_utils import add_random_single_qubit_rotation


def create_ising_circuit(
    L: int, J: float, g: float, dt: float, timesteps: int, *, periodic: bool = False
) -> QuantumCircuit:
    """Ising Trotter circuit with optional periodic boundary conditions.

    Create a quantum circuit for simulating the Ising model. When periodic is True,
    a long-range rzz gate is added between the last and first qubits in each timestep.

    Args:
        L (int): Number of qubits in the circuit.
        J (float): Coupling constant for the ZZ interaction.
        g (float): Transverse field strength.
        dt (float): Time step for the simulation.
        timesteps (int): Number of time steps to simulate.
        periodic (bool, optional): If True, add a long-range gate between qubits 0 and L-1.
                                   Defaults to False.

    Returns:
        QuantumCircuit: A quantum circuit representing the Ising model evolution.
    """
    # Angle on X rotation
    alpha = -2 * dt * g
    # Angle on ZZ rotation
    beta = -2 * dt * J

    circ = QuantumCircuit(L)
    for _ in range(timesteps):
        # Apply RX rotations on all qubits.
        for site in range(L):
            circ.rx(theta=alpha, qubit=site)

        # Even-odd nearest-neighbor interactions.
        for site in range(L // 2):
            circ.rzz(beta, qubit1=2 * site, qubit2=2 * site + 1)

        # Odd-even nearest-neighbor interactions.
        for site in range(1, L // 2):
            circ.rzz(beta, qubit1=2 * site - 1, qubit2=2 * site)

        # For odd L > 1, handle the last pair.
        if L % 2 != 0 and L != 1:
            circ.rzz(beta, qubit1=L - 2, qubit2=L - 1)

        # If periodic, add an additional long-range gate between qubit L-1 and qubit 0.
        if periodic and L > 1:
            circ.rzz(beta, qubit1=0, qubit2=L - 1)

    return circ


def create_2d_ising_circuit(
    num_rows: int, num_cols: int, J: float, g: float, dt: float, timesteps: int
) -> QuantumCircuit:
    """2D Ising Trotter circuit on a rectangular grid using a snaking MPS ordering.

    Args:
        num_rows (int): Number of rows in the qubit grid.
        num_cols (int): Number of columns in the qubit grid.
        J (float): Coupling constant for the ZZ interaction.
        g (float): Transverse field strength.
        dt (float): Time step for the simulation.
        timesteps (int): Number of Trotter steps.

    Returns:
        QuantumCircuit: A quantum circuit representing the 2D Ising model evolution with MPS-friendly ordering.
    """
    total_qubits = num_rows * num_cols
    circ = QuantumCircuit(total_qubits)

    # Define a helper function to compute the snaking index.
    def site_index(row: int, col: int) -> int:
        # For even rows, map left-to-right; for odd rows, map right-to-left.
        if row % 2 == 0:
            return row * num_cols + col
        return row * num_cols + (num_cols - 1 - col)

    # Single-qubit rotation and ZZ interaction angles.
    alpha = -2 * dt * g
    beta = -2 * dt * J

    for _ in range(timesteps):
        # Apply RX rotations to all qubits according to the snaking order.
        for row in range(num_rows):
            for col in range(num_cols):
                q = site_index(row, col)
                circ.rx(alpha, q)

        # Horizontal interactions: within each row, apply rzz gates between adjacent qubits.
        for row in range(num_rows):
            # Even bonds in the row.
            for col in range(0, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rzz(beta, q1, q2)
            # Odd bonds in the row.
            for col in range(1, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rzz(beta, q1, q2)

        # Vertical interactions: between adjacent rows.
        for col in range(num_cols):
            # Even bonds vertically.
            for row in range(0, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rzz(beta, q1, q2)
            # Odd bonds vertically.
            for row in range(1, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rzz(beta, q1, q2)

    return circ


def create_heisenberg_circuit(
    L: int, Jx: float, Jy: float, Jz: float, h: float, dt: float, timesteps: int
) -> QuantumCircuit:
    """Heisenberg Trotter circuit.

    Create a quantum circuit for simulating the Heisenberg model.

    Args:
        L (int): Number of qubits (sites) in the circuit.
        Jx (float): Coupling constant for the XX interaction.
        Jy (float): Coupling constant for the YY interaction.
        Jz (float): Coupling constant for the ZZ interaction.
        h (float): Magnetic field strength.
        dt (float): Time step for the simulation.
        timesteps (int): Number of time steps to simulate.

    Returns:
        QuantumCircuit: A quantum circuit representing the Heisenberg model evolution.
    """
    theta_xx = -2 * dt * Jx
    theta_yy = -2 * dt * Jy
    theta_zz = -2 * dt * Jz
    theta_z = -2 * dt * h

    circ = QuantumCircuit(L)
    for _ in range(timesteps):
        # Z application
        for site in range(L):
            circ.rz(phi=theta_z, qubit=site)

        # ZZ application
        for site in range(L // 2):
            circ.rzz(theta=theta_zz, qubit1=2 * site, qubit2=2 * site + 1)

        for site in range(1, L // 2):
            circ.rzz(theta=theta_zz, qubit1=2 * site - 1, qubit2=2 * site)

        if L % 2 != 0 and L != 1:
            circ.rzz(theta=theta_zz, qubit1=L - 2, qubit2=L - 1)

        # XX application
        for site in range(L // 2):
            circ.rxx(theta=theta_xx, qubit1=2 * site, qubit2=2 * site + 1)

        for site in range(1, L // 2):
            circ.rxx(theta=theta_xx, qubit1=2 * site - 1, qubit2=2 * site)

        if L % 2 != 0 and L != 1:
            circ.rxx(theta=theta_xx, qubit1=L - 2, qubit2=L - 1)

        # YY application
        for site in range(L // 2):
            circ.ryy(theta=theta_yy, qubit1=2 * site, qubit2=2 * site + 1)

        for site in range(1, L // 2):
            circ.ryy(theta=theta_yy, qubit1=2 * site - 1, qubit2=2 * site)

        if L % 2 != 0 and L != 1:
            circ.ryy(theta=theta_yy, qubit1=L - 2, qubit2=L - 1)

    return circ


def create_2d_heisenberg_circuit(
    num_rows: int, num_cols: int, Jx: float, Jy: float, Jz: float, h: float, dt: float, timesteps: int
) -> QuantumCircuit:
    """2D Heisenberg Trotter circuit on a rectangular grid using a snaking MPS ordering.

    Args:
        num_rows (int): Number of rows in the qubit grid.
        num_cols (int): Number of columns in the qubit grid.
        Jx (float): Coupling constant for the XX interaction.
        Jy (float): Coupling constant for the YY interaction.
        Jz (float): Coupling constant for the ZZ interaction.
        h (float): Single-qubit Z-field strength.
        dt (float): Time step for the simulation.
        timesteps (int): Number of Trotter steps.

    Returns:
        QuantumCircuit: A quantum circuit representing the 2D Heisenberg model evolution
                       with MPS-friendly ordering.
    """
    total_qubits = num_rows * num_cols
    circ = QuantumCircuit(total_qubits)

    # Define a helper function to compute the snaking index.
    def site_index(row: int, col: int) -> int:
        # For even rows, map left-to-right; for odd rows, map right-to-left.
        if row % 2 == 0:
            return row * num_cols + col
        return row * num_cols + (num_cols - 1 - col)

    # Define the Trotter angles
    theta_xx = -2.0 * dt * Jx
    theta_yy = -2.0 * dt * Jy
    theta_zz = -2.0 * dt * Jz
    theta_z = -2.0 * dt * h

    for _ in range(timesteps):
        # (1) Apply single-qubit Z rotations to all qubits
        for row in range(num_rows):
            for col in range(num_cols):
                q = site_index(row, col)
                circ.rz(theta_z, q)

        # (2) ZZ interactions
        # Horizontal even bonds
        for row in range(num_rows):
            for col in range(0, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rzz(theta_zz, q1, q2)
        # Horizontal odd bonds
        for row in range(num_rows):
            for col in range(1, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rzz(theta_zz, q1, q2)
        # Vertical even bonds
        for col in range(num_cols):
            for row in range(0, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rzz(theta_zz, q1, q2)
        # Vertical odd bonds
        for col in range(num_cols):
            for row in range(1, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rzz(theta_zz, q1, q2)

        # (3) XX interactions
        # Horizontal even bonds
        for row in range(num_rows):
            for col in range(0, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rxx(theta_xx, q1, q2)
        # Horizontal odd bonds
        for row in range(num_rows):
            for col in range(1, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rxx(theta_xx, q1, q2)
        # Vertical even bonds
        for col in range(num_cols):
            for row in range(0, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rxx(theta_xx, q1, q2)
        # Vertical odd bonds
        for col in range(num_cols):
            for row in range(1, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rxx(theta_xx, q1, q2)

        # (4) YY interactions
        # Horizontal even bonds
        for row in range(num_rows):
            for col in range(0, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.ryy(theta_yy, q1, q2)
        # Horizontal odd bonds
        for row in range(num_rows):
            for col in range(1, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.ryy(theta_yy, q1, q2)
        # Vertical even bonds
        for col in range(num_cols):
            for row in range(0, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.ryy(theta_yy, q1, q2)
        # Vertical odd bonds
        for col in range(num_cols):
            for row in range(1, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.ryy(theta_yy, q1, q2)

    return circ


def nearest_neighbour_random_circuit(
    n_qubits: int,
    layers: int,
    seed: int = 42,
) -> QuantumCircuit:
    """Creates a random circuit with single and two-qubit nearest-neighbor gates.

    Gates are sampled following the prescription in https://arxiv.org/abs/2002.07730.

    Returns:
        A `QuantumCircuit` on `n_qubits` implementing `layers` of alternating
        random single-qubit rotations and nearest-neighbor CZ/CX entanglers.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)

    for layer in range(layers):
        # Single-qubit random rotations
        for qubit in range(n_qubits):
            add_random_single_qubit_rotation(qc, qubit, rng)

        # Two-qubit entangling gates
        if layer % 2 == 0:
            # Even layer → pair (1,2), (3,4), ...
            pairs = [(i, i + 1) for i in range(1, n_qubits - 1, 2)]
        else:
            # Odd layer → pair (0,1), (2,3), ...
            pairs = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]

        for q1, q2 in pairs:
            if rng.random() < 0.5:
                qc.cz(q1, q2)
            else:
                qc.cx(q1, q2)

        qc.barrier()
    return qc
