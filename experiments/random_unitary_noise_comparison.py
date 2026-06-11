# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compare a random generic-UnitaryGate circuit with and without X/XX noise."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from scipy.linalg import expm

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import Z

NUM_QUBITS = 4
DEPTH = 6
NOISY_TRAJECTORIES = 100
NOISE_STRENGTH = 0.01
OBSERVED_QUBIT = 1
RANDOM_SEED = 37


def random_two_qubit_unitary(rng: np.random.Generator, scale: float = 0.25) -> np.ndarray:
    """Build one deterministic random dense two-qubit unitary.

    Returns:
        Dense ``4x4`` unitary matrix.
    """
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    generator = scale * (raw + raw.conj().T) / 2
    return np.asarray(expm(-1j * generator), dtype=np.complex128)


def build_random_unitary_circuit() -> QuantumCircuit:
    """Build a small sampled circuit containing generic two-qubit ``UnitaryGate`` operations.

    Returns:
        Sampled random circuit.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    circuit = QuantumCircuit(NUM_QUBITS)

    for layer in range(DEPTH):
        for qubit in range(NUM_QUBITS):
            circuit.rx(float(rng.uniform(-0.4, 0.4)), qubit)
            circuit.rz(float(rng.uniform(-0.4, 0.4)), qubit)

        if layer % 2 == 0:
            circuit.append(UnitaryGate(random_two_qubit_unitary(rng)), [0, 1])
            circuit.append(UnitaryGate(random_two_qubit_unitary(rng)), [2, 3])
        else:
            circuit.append(UnitaryGate(random_two_qubit_unitary(rng)), [1, 2])
            circuit.append(UnitaryGate(random_two_qubit_unitary(rng)), [0, 3])

        if layer < DEPTH - 1:
            circuit.barrier(label="SAMPLE_OBSERVABLES")

    return circuit


def single_site_x_noise() -> NoiseModel:
    """Build single-site X noise on every qubit.

    Returns:
        Noise model with local X processes.
    """
    return NoiseModel([{"name": "pauli_x", "sites": [site], "strength": NOISE_STRENGTH} for site in range(NUM_QUBITS)])


def single_site_x_and_nearest_neighbor_xx_noise() -> NoiseModel:
    """Build single-site X plus nearest-neighbor XX crosstalk noise.

    Returns:
        Noise model with local X and nearest-neighbor XX processes.
    """
    processes = [{"name": "pauli_x", "sites": [site], "strength": NOISE_STRENGTH} for site in range(NUM_QUBITS)]
    processes.extend(
        {"name": "crosstalk_xx", "sites": [site, site + 1], "strength": NOISE_STRENGTH}
        for site in range(NUM_QUBITS - 1)
    )
    return NoiseModel(processes)


def run_simulation(circuit: QuantumCircuit, noise_model: NoiseModel | None, num_traj: int) -> np.ndarray:
    """Run the circuit and return sampled Z expectation values on the observed qubit.

    Returns:
        Sampled Z expectation values.
    """
    params = StrongSimParams(
        observables=[Observable(Z(), OBSERVED_QUBIT)],
        num_traj=num_traj,
        sample_layers=True,
        random_seed=RANDOM_SEED,
        preset="exact",
        gate_mode="full-tdvp",
    )
    result = Simulator(parallel=True, show_progress=True).run(
        State(NUM_QUBITS, initial="zeros", pad=2),
        circuit,
        params,
        noise_model,
    )
    return np.real(result.expectation_values[0])


def save_plot(
    layers: np.ndarray,
    noiseless_z: np.ndarray,
    single_site_x_z: np.ndarray,
    single_site_x_and_xx_z: np.ndarray,
    output_path: Path,
) -> None:
    """Save a Matplotlib plot for the sampled expectation values."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, noiseless_z, "o-", label="No noise, 1 trajectory")
    ax.plot(layers, single_site_x_z, "s-", label=f"X noise, {NOISY_TRAJECTORIES} trajectories")
    ax.plot(layers, single_site_x_and_xx_z, "^-", label=f"X + NN XX noise, {NOISY_TRAJECTORIES} trajectories")
    ax.set_xlabel("Sampled circuit layer")
    ax.set_ylabel(r"$\langle Z_2 \rangle$")
    ax.set_title("Random generic-UnitaryGate circuit")
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_csv(
    layers: np.ndarray,
    noiseless_z: np.ndarray,
    single_site_x_z: np.ndarray,
    single_site_x_and_xx_z: np.ndarray,
    output_path: Path,
) -> None:
    """Save the sampled expectation values as CSV."""
    rows = ["layer,noiseless_z,single_site_x_noise_z,single_site_x_and_nn_xx_noise_z"]
    rows.extend(
        f"{int(layer)},{noiseless:.10f},{single_site_x:.10f},{single_site_x_and_xx:.10f}"
        for layer, noiseless, single_site_x, single_site_x_and_xx in zip(
            layers, noiseless_z, single_site_x_z, single_site_x_and_xx_z, strict=True
        )
    )
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    """Run the three simulations and write plot/CSV outputs."""
    circuit = build_random_unitary_circuit()

    noiseless_z = run_simulation(circuit, noise_model=None, num_traj=1)
    single_site_x_z = run_simulation(circuit, noise_model=single_site_x_noise(), num_traj=NOISY_TRAJECTORIES)
    single_site_x_and_xx_z = run_simulation(
        circuit,
        noise_model=single_site_x_and_nearest_neighbor_xx_noise(),
        num_traj=NOISY_TRAJECTORIES,
    )

    layers = np.arange(DEPTH + 1)
    output_dir = Path(__file__).parent
    save_plot(
        layers,
        noiseless_z,
        single_site_x_z,
        single_site_x_and_xx_z,
        output_dir / "random_unitary_noise_comparison.png",
    )
    save_csv(
        layers,
        noiseless_z,
        single_site_x_z,
        single_site_x_and_xx_z,
        output_dir / "random_unitary_noise_comparison.csv",
    )


if __name__ == "__main__":
    main()
