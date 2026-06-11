# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run an 8-qubit trotterized Ising circuit with and without X noise."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]
import numpy as np
from qiskit.circuit import QuantumCircuit

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
from mqt.yaqs.core.libraries.gate_library import Z

NUM_QUBITS = 8
TIMESTEPS = 10
DT = 0.1
J = 1.0
G = 0.5
NOISE_STRENGTH = 0.01
OBSERVED_QUBIT = 4  # zero-based index for the fifth qubit
RANDOM_SEED = 11


def build_sampled_ising_circuit() -> QuantumCircuit:
    """Build a ten-step Ising circuit sampled after each timestep.

    Returns:
        The Ising circuit with labelled sampling barriers between timesteps.
    """
    circuit = QuantumCircuit(NUM_QUBITS)
    one_step = create_ising_circuit(L=NUM_QUBITS, J=J, g=G, dt=DT, timesteps=1)

    for step in range(TIMESTEPS):
        circuit.compose(one_step, inplace=True)
        if step < TIMESTEPS - 1:
            circuit.barrier(label="SAMPLE_OBSERVABLES")

    return circuit


def run_simulation(circuit: QuantumCircuit, noise_model: NoiseModel | None, num_traj: int) -> np.ndarray:
    """Run a strong digital TJM simulation.

    Args:
        circuit: The circuit to simulate.
        noise_model: Optional noise model for the simulation.
        num_traj: Number of circuit tensor jump trajectories.

    Returns:
        The fifth-qubit Z expectation value at each sampled timestep.
    """
    state = State(NUM_QUBITS, initial="zeros", pad=2)
    sim_params = StrongSimParams(
        observables=[Observable(Z(), OBSERVED_QUBIT)],
        num_traj=num_traj,
        sample_layers=True,
        random_seed=RANDOM_SEED,
        preset="balanced",
        gate_mode="mpo",
    )
    result = Simulator(parallel=True, show_progress=True).run(state, circuit, sim_params, noise_model)
    return np.real(result.expectation_values[0])


def save_plot(
    times: np.ndarray,
    noiseless_z: np.ndarray,
    single_site_x_z: np.ndarray,
    single_site_x_and_xx_z: np.ndarray,
    output_path: Path,
) -> None:
    """Save a matplotlib line plot for the fifth-qubit Z expectation."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, noiseless_z, "o-", label="No noise, 1 trajectory")
    ax.plot(times, single_site_x_z, "s-", label="Single-site X noise, 200 trajectories")
    ax.plot(times, single_site_x_and_xx_z, "^-", label="Single-site X + NN XX noise, 200 trajectories")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\langle Z_5 \rangle$")
    ax.set_title("8-qubit trotterized Ising circuit")
    ax.set_xticks(times)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_csv(
    times: np.ndarray,
    noiseless_z: np.ndarray,
    single_site_x_z: np.ndarray,
    single_site_x_and_xx_z: np.ndarray,
    output_path: Path,
) -> None:
    """Save the sampled expectation values as CSV."""
    rows = ["time,noiseless_z,single_site_x_noise_z,single_site_x_and_nn_xx_noise_z"]
    rows.extend(
        f"{time:.1f},{noiseless:.10f},{single_site_x:.10f},{single_site_x_and_xx:.10f}"
        for time, noiseless, single_site_x, single_site_x_and_xx in zip(
            times, noiseless_z, single_site_x_z, single_site_x_and_xx_z, strict=True
        )
    )
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    """Run all simulations and plot the fifth-qubit Z expectation."""
    circuit = build_sampled_ising_circuit()
    single_site_x_noise_processes = [
        {"name": "pauli_x", "sites": [site], "strength": NOISE_STRENGTH} for site in range(NUM_QUBITS)
    ]
    nearest_neighbor_xx_noise_processes = [
        {"name": "crosstalk_xx", "sites": [site, site + 1], "strength": NOISE_STRENGTH}
        for site in range(NUM_QUBITS - 1)
    ]

    single_site_x_noise = NoiseModel(single_site_x_noise_processes)
    single_site_x_and_xx_noise = NoiseModel(single_site_x_noise_processes + nearest_neighbor_xx_noise_processes)

    noiseless_z = run_simulation(circuit, noise_model=None, num_traj=1)
    single_site_x_z = run_simulation(circuit, noise_model=single_site_x_noise, num_traj=200)
    single_site_x_and_xx_z = run_simulation(circuit, noise_model=single_site_x_and_xx_noise, num_traj=200)

    times = np.linspace(0.0, TIMESTEPS * DT, TIMESTEPS + 1)
    output_dir = Path(__file__).parent
    save_plot(
        times,
        noiseless_z,
        single_site_x_z,
        single_site_x_and_xx_z,
        output_dir / "eight_qubit_ising_x_noise_z5.png",
    )
    save_csv(
        times,
        noiseless_z,
        single_site_x_z,
        single_site_x_and_xx_z,
        output_dir / "eight_qubit_ising_x_noise_z5.csv",
    )


if __name__ == "__main__":
    main()
