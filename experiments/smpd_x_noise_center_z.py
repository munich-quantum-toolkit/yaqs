# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run a small SMPD-style sequential circuit with several X-noise strengths."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]
import numpy as np
from qiskit.circuit import QuantumCircuit

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.circuit_library import create_sequential_matrix_product_disentangler_circuit
from mqt.yaqs.core.libraries.gate_library import Z

NUM_QUBITS = 5
CENTER_QUBIT = NUM_QUBITS // 2
NOISY_TRAJECTORIES = 200
RANDOM_SEED = 137
NOISE_STRENGTHS = (0.0, 0.01, 0.1)
SEQUENTIAL_BONDS = tuple((site, site + 1) for site in range(NUM_QUBITS - 1))


def noise_free(strength: float) -> bool:
    """Return whether a noise strength should be treated as noiseless."""
    return abs(strength) < 1e-15


def random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Build a deterministic random dense unitary.

    Returns:
        Dense unitary matrix.
    """
    matrix = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(matrix)
    phases = np.diag(r) / np.abs(np.diag(r))
    return np.asarray(q * phases, dtype=np.complex128)


def build_sampled_smpd_circuit() -> tuple[QuantumCircuit, list[str]]:
    """Build a sampled SMPD-style state-preparation circuit.

    Returns:
        Tuple of the circuit and labels for the sampled expectation-value columns.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    circuit = QuantumCircuit(NUM_QUBITS)
    layer_labels = ["input"]

    initial_layer = [(random_unitary(2, rng), (site,)) for site in range(NUM_QUBITS)]
    circuit.compose(create_sequential_matrix_product_disentangler_circuit(NUM_QUBITS, initial_layer), inplace=True)
    circuit.barrier(label="SAMPLE_OBSERVABLES")
    layer_labels.append("single-qubit layer")

    sequential_gates = [(random_unitary(4, rng), sites) for sites in SEQUENTIAL_BONDS]
    for gate_index, gate in enumerate(sequential_gates):
        circuit.compose(create_sequential_matrix_product_disentangler_circuit(NUM_QUBITS, [gate]), inplace=True)
        layer_labels.append(f"sequential {gate[1][0]}-{gate[1][1]}")
        if gate_index < len(sequential_gates) - 1:
            circuit.barrier(label="SAMPLE_OBSERVABLES")

    return circuit, layer_labels


def x_noise_model(strength: float) -> NoiseModel | None:
    """Build local X noise on every qubit.

    Returns:
        Noise model for positive strengths, otherwise ``None``.
    """
    if noise_free(strength):
        return None
    return NoiseModel([{"name": "pauli_x", "sites": [site], "strength": strength} for site in range(NUM_QUBITS)])


def run_simulation(circuit: QuantumCircuit, strength: float) -> np.ndarray:
    """Run the sampled circuit for one noise strength.

    Returns:
        Sampled center-qubit Z expectation values.
    """
    params = StrongSimParams(
        observables=[Observable(Z(), CENTER_QUBIT)],
        num_traj=1 if noise_free(strength) else NOISY_TRAJECTORIES,
        sample_layers=True,
        random_seed=RANDOM_SEED,
        preset="exact",
        gate_mode="full-tdvp",
    )
    result = Simulator(parallel=True, show_progress=True).run(
        State(NUM_QUBITS, initial="zeros", pad=2),
        circuit,
        params,
        x_noise_model(strength),
    )
    return np.real(result.expectation_values[0])


def save_plot(layer_labels: list[str], z_values: dict[float, np.ndarray], output_path: Path) -> None:
    """Save the center-qubit Z expectation plot."""
    x_axis = np.arange(len(layer_labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for strength, values in z_values.items():
        label = "no noise" if noise_free(strength) else rf"X noise $\gamma={strength}$"
        ax.plot(x_axis, values, marker="o", linewidth=2, label=label)

    ax.set_xlabel("SMPD state-preparation step")
    ax.set_ylabel(rf"$\langle Z \rangle$ on center qubit, site {CENTER_QUBIT}")
    ax.set_title("SMPD sequential circuit under local X noise")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(layer_labels, rotation=20, ha="right")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_csv(layer_labels: list[str], z_values: dict[float, np.ndarray], output_path: Path) -> None:
    """Save sampled expectation values as CSV."""
    rows = ["layer,no_noise_z,x_noise_0_01_z,x_noise_0_1_z"]
    for idx, label in enumerate(layer_labels):
        rows.append(f"{label},{z_values[0.0][idx]:.10f},{z_values[0.01][idx]:.10f},{z_values[0.1][idx]:.10f}")
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    """Run the SMPD noise comparison experiment."""
    circuit, layer_labels = build_sampled_smpd_circuit()
    z_values = {strength: run_simulation(circuit, strength) for strength in NOISE_STRENGTHS}

    output_dir = Path(__file__).parent
    save_plot(layer_labels, z_values, output_dir / "smpd_x_noise_center_z.png")
    save_csv(layer_labels, z_values, output_dir / "smpd_x_noise_center_z.csv")


if __name__ == "__main__":
    main()
