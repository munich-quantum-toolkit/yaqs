# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module provides tests for benchmarking an arbitrary quantum circuit using the Tensor Jump Method (TJM).
The tests compare exact simulation results from Qiskit's Aersimulator with approximate simulations using various
simulation parameters such as maximum bond dimension, window size, and SVD threshold. It verifies that the simulation
metrics (absolute error and runtime) are computed correctly and that the generated 3D plots accurately represent the
performance of the approximate simulation methods.
"""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
from qiskit.quantum_info import Operator, Pauli, Statevector
from qiskit_aer import Aersimulator

from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def run(
    input_ircuit: QuantumCircuit,
    style: str = "dots",
    max_bond_dims: list[int] | None = None,
    window_sizes: list[int] | None = None,
    thresholds: list[float] | None = None,
) -> None:
    """Benchmark an arbitrary quantum circuit by comparing a Qiskit exact simulation
    with approximate simulations using various simulation parameters.

    Parameters:
      input_ircuit (QuantumCircuit): A Qiskit circuit to be benchmarked.
      max_bond_dims (list): List of maximum bond dimensions to test.
      window_sizes (list): List of window sizes to test.
      thresholds (list): List of SVD thresholds to test.

    The function runs the circuit with a Qiskit Aersimulator to obtain the exact expectation
    value of a Pauli Z measurement on the middle qubit, and then simulates the same circuit using
    yaqs's simulator over a grid of simulation parameters. It then groups the results by bond
    dimension and plots 3D "planes" in the window/threshold space (with –log₁₀(threshold) as one axis)
    that are stacked along the bond dimension axis. The face color of each plane encodes the absolute
    error (left) or runtime (right).
    """
    # Create a deep copy for the exact simulation and prepare it.
    if thresholds is None:
        thresholds = [0.001, 1e-06, 1e-09, 1e-12, 1e-16]
    if window_sizes is None:
        window_sizes = [0, 1, 2, 3, 4]
    if max_bond_dims is None:
        max_bond_dims = [2, 4, 8, 16, 32]
    circuit_exact = copy.deepcopy(input_ircuit)
    circuit_exact.reverse_bits()
    circuit_exact.save_statevector()  # Instruct the simulator to save the statevector.
    num_qubits = circuit_exact.num_qubits

    # Run the exact simulation using Qiskit's Aersimulator.
    simulator = Aersimulator(method="statevector")
    result = simulator.run(circuit_exact).result()
    qiskit_state = result.get_statevector(circuit_exact)

    # Construct an observable: Z on the middle qubit (I on others).
    pauli_list = ["I"] * num_qubits
    pauli_list[num_qubits // 2] = "Z"
    pauli_string = "".join(pauli_list)
    op = Operator(Pauli(pauli_string).to_matrix())

    sv = Statevector(qiskit_state)
    exact_result = np.real(sv.expectation_value(op))

    # Set up the simulation state and noise model for yaqs.
    state = MPS(num_qubits, state="zeros")
    noise_model = None
    N = 1  # number of simulation samples

    # Prepare lists to store parameters and results.
    bond_list = []
    window_list = []
    threshold_list = []
    errors = []
    runtimes = []

    # Loop over all combinations of simulation parameters.
    for window in window_sizes:
        if window > num_qubits:
            break
        for bond_dim in max_bond_dims:
            if bond_dim > 2 ** (num_qubits // 2):
                break
            for threshold in thresholds:
                # Create a fresh copy of the circuit for each simulation.
                circuit_copy = copy.deepcopy(input_ircuit)
                measurements = [Observable("z", num_qubits // 2)]
                sim_params = StrongSimParams(measurements, N, bond_dim, threshold=threshold, window_size=window)

                start_time = time.time()
                simulator.run(state, circuit_copy, sim_params, noise_model)
                runtime = time.time() - start_time

                # Compute the absolute error with respect to the exact result.
                error = np.abs(sim_params.observables[0].results - exact_result)

                # Record the simulation parameters and results.
                bond_list.append(bond_dim)
                window_list.append(window)
                threshold_list.append(threshold)
                errors.append(error)
                runtimes.append(runtime)

    if style == "dots":
        # Convert thresholds to -log10(threshold) for visualization.
        log_thresholds = [-np.log10(th) for th in threshold_list]

        # Create a figure with two 3D subplots.
        fig = plt.figure(figsize=(16, 7))

        # 3D scatter plot for error.
        ax1 = fig.add_subplot(121, projection="3d")
        sc1 = ax1.scatter(
            bond_list, window_list, log_thresholds, c=errors, cmap="viridis", s=60, norm=LogNorm(vmin=1e-12, vmax=1e-2)
        )
        ax1.set_xlabel("Max Bond Dimension", fontsize=11)
        ax1.set_ylabel("Window Size", fontsize=11)
        ax1.set_zlabel("-log10(Threshold)", fontsize=11)
        ax1.set_title("Error vs. Simulation Parameters", fontsize=14)
        ax1.grid(True, linestyle="--", alpha=0.5)
        cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1, aspect=20)
        cbar1.set_label("Absolute Error (log scale)", fontsize=11)
        cbar1.ax.tick_params(labelsize=10)

        # 3D scatter plot for runtime.
        ax2 = fig.add_subplot(122, projection="3d")
        sc2 = ax2.scatter(bond_list, window_list, log_thresholds, c=runtimes, cmap="magma", s=60)
        ax2.set_xlabel("Max Bond Dimension", fontsize=11)
        ax2.set_ylabel("Window Size", fontsize=11)
        ax2.set_zlabel("-log10(Threshold)", fontsize=11)
        ax2.set_title("Runtime vs. Simulation Parameters", fontsize=14)
        ax2.grid(True, linestyle="--", alpha=0.5)
        cbar2 = fig.colorbar(sc2, ax=ax2, pad=0.1, aspect=20)
        cbar2.set_label("Runtime (s)", fontsize=11)
        cbar2.ax.tick_params(labelsize=10)

        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.zaxis.set_major_locator(MaxNLocator(integer=True))

        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.zaxis.set_major_locator(MaxNLocator(integer=True))

        # Overall figure title and layout adjustments.
        fig.suptitle("Performance Metrics Benchmark", fontsize=18, fontweight="bold")
        plt.tight_layout(pad=4.0)
        plt.show()

    elif style == "planes":
        # Convert thresholds to -log10(threshold) for visualization.
        log_thresholds = [-np.log10(th) for th in threshold_list]
        # === Group data by bond dimension ===
        # Filter out values that were skipped during simulation.
        used_windows = [w for w in window_sizes if w <= num_qubits]
        used_bond_dims = [bd for bd in max_bond_dims if bd <= 2 ** (num_qubits // 2)]
        n_windows = len(used_windows)
        n_thresholds = len(thresholds)

        # Create dictionaries to hold 2D arrays (plane data) for each used bond dimension.
        error_dict = {bd: np.zeros((n_windows, n_thresholds)) for bd in used_bond_dims}
        runtime_dict = {bd: np.zeros((n_windows, n_thresholds)) for bd in used_bond_dims}

        # Fill the dictionaries with the simulation results.
        k = 0
        for i in range(len(used_windows)):
            for bd in used_bond_dims:
                for j, _ in enumerate(thresholds):
                    error_dict[bd][i, j] = errors[k]
                    runtime_dict[bd][i, j] = runtimes[k]
                    k += 1

        # Create the 2D grid for the window and threshold plane.
        # Here, y-axis: window size, z-axis: -log10(threshold)
        Y, Z = np.meshgrid(used_windows, [-np.log10(th) for th in thresholds], indexing="ij")

        # === Plotting ===
        fig = plt.figure(figsize=(16, 7))

        # --- Error Plot ---
        ax1 = fig.add_subplot(121, projection="3d")
        norm_err = LogNorm(vmin=1e-12, vmax=1e-2)
        cmap_err = plt.cm.viridis

        for bd in used_bond_dims:
            # Create a plane at x = bond_dim.
            X_plane = np.full(Y.shape, bd)
            facecolors = cmap_err(norm_err(error_dict[bd]))
            ax1.plot_surface(
                X_plane,
                Y,
                Z,
                rstride=1,
                cstride=1,
                facecolors=facecolors,
                shade=False,
                edgecolor="k",
                linewidth=0.3,
                antialiased=True,
            )
        ax1.set_xlabel("Max Bond Dimension", fontsize=11)
        ax1.set_ylabel("Window Size", fontsize=11)
        ax1.set_zlabel("-log10(Threshold)", fontsize=11)
        ax1.set_title("Error vs. Simulation Parameters", fontsize=14)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Set tick positions using the actual values:
        # x-axis (bond dimensions)
        ax1.set_xticks(used_bond_dims)
        ax1.set_xticklabels(used_bond_dims)
        # y-axis (window sizes)
        ax1.set_yticks(used_windows)
        ax1.set_yticklabels(used_windows)
        # z-axis (-log10(threshold)); these will be integers (e.g. 3, 6, 9, ...)
        z_vals = [np.round(-np.log10(th)) for th in thresholds]
        ax1.set_zticks(z_vals)
        ax1.set_zticklabels(z_vals)

        mappable_err = plt.cm.ScalarMappable(norm=norm_err, cmap=cmap_err)
        mappable_err.set_array([])
        cbar1 = fig.colorbar(mappable_err, ax=ax1, pad=0.1, aspect=20)
        cbar1.set_label("Absolute Error (log scale)", fontsize=11)
        cbar1.ax.tick_params(labelsize=10)

        # --- Runtime Plot ---
        ax2 = fig.add_subplot(122, projection="3d")
        norm_rt = Normalize(vmin=min(runtimes), vmax=max(runtimes))
        cmap_rt = plt.cm.magma

        for bd in used_bond_dims:
            X_plane = np.full(Y.shape, bd)
            facecolors_rt = cmap_rt(norm_rt(runtime_dict[bd]))
            ax2.plot_surface(
                X_plane,
                Y,
                Z,
                rstride=1,
                cstride=1,
                facecolors=facecolors_rt,
                shade=False,
                edgecolor="k",
                linewidth=0.3,
                antialiased=True,
            )
        ax2.set_xlabel("Max Bond Dimension", fontsize=11)
        ax2.set_ylabel("Window Size", fontsize=11)
        ax2.set_zlabel("-log10(Threshold)", fontsize=11)
        ax2.set_title("Runtime vs. Simulation Parameters", fontsize=14)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # Set tick positions using the actual values:
        ax2.set_xticks(used_bond_dims)
        ax2.set_xticklabels(used_bond_dims)
        ax2.set_yticks(used_windows)
        ax2.set_yticklabels(used_windows)
        ax2.set_zticks(z_vals)
        ax2.set_zticklabels(z_vals)

        mappable_rt = plt.cm.ScalarMappable(norm=norm_rt, cmap=cmap_rt)
        mappable_rt.set_array([])
        cbar2 = fig.colorbar(mappable_rt, ax=ax2, pad=0.1, aspect=20)
        cbar2.set_label("Runtime (s)", fontsize=11)
        cbar2.ax.tick_params(labelsize=10)

        fig.suptitle("Performance Metrics Benchmark", fontsize=18, fontweight="bold")
        plt.tight_layout(pad=4.0)
        plt.show()
