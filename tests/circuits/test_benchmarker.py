# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for benchmarker.

This file tests various aspects of the circuit benchmarker.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

from mqt.yaqs.circuits.benchmarker import run

plt.show = lambda: None


def test_run_default_parameters() -> None:
    """Benchmarker default parameters.

    Test the run function with default parameters.
    This test creates a simple quantum circuit with 3 qubits, applies a Hadamard gate
    to the first qubit, and then applies CNOT gates between the first and second qubits,
    and the second and third qubits. The circuit is then passed to the run function to
    benchmark its performance with default parameters.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    plt.close("all")
    figs_before = len(plt.get_fignums())

    # Create a simple quantum circuit for testing
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Run the benchmarker with default parameters

    run(qc)
    figs_after = len(plt.get_fignums())
    assert figs_after > figs_before, "Expected a matplotlib plot to be created."
    plt.close("all")


def test_run_custom_parameters() -> None:
    """Benchmarker custom parameters.

    Test the `run` function with custom parameters on a simple quantum circuit.
    This test creates a quantum circuit with 3 qubits, applies a Hadamard gate to the first qubit,
    and then applies CNOT gates between the first and second qubits, and the second and third qubits.
    It then defines custom parameters for `max_bond_dims`, `window_sizes`, and `thresholds`, and
    runs the `run` function with these parameters.
    Custom Parameters:
    - max_bond_dims: List of maximum bond dimensions to test.
    - window_sizes: List of window sizes to test.
    - thresholds: List of thresholds to test.
    The purpose of this test is to ensure that the `run` function can handle custom parameters
    correctly and produce the expected results.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    plt.close("all")
    figs_before = len(plt.get_fignums())

    # Create a simple quantum circuit for testing
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Define custom parameters
    max_bond_dims = [2, 4]
    window_sizes = [0, 1]
    thresholds = [0.001, 1e-06]

    # Run the benchmarker with custom parameters
    run(qc, max_bond_dims=max_bond_dims, window_sizes=window_sizes, thresholds=thresholds)
    figs_after = len(plt.get_fignums())
    assert figs_after > figs_before, "Expected a matplotlib plot to be created."
    plt.close("all")


def test_run_dots_style() -> None:
    """Benchmarker dots.

    Test the run function with a simple quantum circuit using the 'dots' style.
    This test creates a simple quantum circuit with 3 qubits, applies a Hadamard gate
    to the first qubit, and then applies CNOT gates between the first and second qubits,
    and the second and third qubits. The circuit is then run using the 'dots' style.
    The purpose of this test is to verify that the run function correctly handles
    the 'dots' style for visualizing the quantum circuit.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    plt.close("all")
    figs_before = len(plt.get_fignums())

    # Create a simple quantum circuit for testing
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Run the benchmarker with dots style
    run(qc, style="dots")
    figs_after = len(plt.get_fignums())
    assert figs_after > figs_before, "Expected a matplotlib plot to be created."
    plt.close("all")


def test_run_planes_style() -> None:
    """Benchmarker planes.

    Test the benchmarker with the 'planes' style.
    This function creates a simple quantum circuit with 3 qubits, applies a Hadamard gate to the first qubit,
    and then applies CNOT gates between the first and second qubits, and the second and third qubits.
    It then runs the benchmarker with the 'planes' style to verify its functionality.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    plt.close("all")
    figs_before = len(plt.get_fignums())

    # Create a simple quantum circuit for testing
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Run the benchmarker with planes style
    run(qc, style="planes")
    figs_after = len(plt.get_fignums())
    assert figs_after > figs_before, "Expected a matplotlib plot to be created."
    plt.close("all")
