# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Prepare an 8-qubit Gaussian target state with deterministic Krotov optimization."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]
import numpy as np

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovOptions,
    ParameterizedCircuit,
    ParameterizedGate,
    state_preparation_metrics,
    train_krotov_state_preparation_batch,
)
from mqt.yaqs.optimization.krotov import forward_states

NUM_QUBITS = 8
NUM_LAYERS = 3
MU = 0.5
SIGMA = 0.1
RANDOM_SEED = 123
ENTANGLING_INIT_SCALE = 0.01
MAX_ITERATIONS = 400
BATCH_STEP_SIZE = 0.4
BATCH_DECAY = 0.005


def gaussian_target_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the normalized Gaussian target state from the requested parameters.

    Returns:
        Tuple ``(x, amplitudes, probabilities)``.
    """
    num_entries = 2**NUM_QUBITS
    x_grid = np.linspace(0.0, 1.0, num_entries)
    amplitudes = np.exp(-((x_grid - MU) ** 2) / (4.0 * SIGMA**2))
    amplitudes /= np.linalg.norm(amplitudes)
    probabilities = np.abs(amplitudes) ** 2
    return x_grid, amplitudes.astype(np.complex128), probabilities


def append_u3_layer(gates: list[ParameterizedGate], site: int, base_index: int) -> None:
    """Append a trainable ``rz-ry-rz`` one-qubit unitary."""
    gates.extend((
        ParameterizedGate("rz", (site,), param_index=base_index + 2),
        ParameterizedGate("ry", (site,), param_index=base_index),
        ParameterizedGate("rz", (site,), param_index=base_index + 1),
    ))


def append_two_qubit_block(gates: list[ParameterizedGate], left_site: int, right_site: int, base_index: int) -> None:
    """Append a trainable scalar KAK-style two-qubit block."""
    append_u3_layer(gates, left_site, base_index)
    append_u3_layer(gates, right_site, base_index + 3)
    gates.extend((
        ParameterizedGate("rxx", (left_site, right_site), param_index=base_index + 6),
        ParameterizedGate("ryy", (left_site, right_site), param_index=base_index + 7),
        ParameterizedGate("rzz", (left_site, right_site), param_index=base_index + 8),
    ))


def create_right_sweep_smpd_ansatz(num_qubits: int, num_layers: int) -> ParameterizedCircuit:
    """Create a Krotov-compatible scalar right-sweep SMPD-style ansatz.

    Each layer follows the right-canonical SMPD gate order: nearest-neighbor
    blocks from left to right, followed by a final one-qubit unitary on the last
    qubit. A trainable product-state layer is prepended to avoid starting nearly
    orthogonal to the localized target.

    Returns:
        Parameterized circuit with one scalar parameter per primitive gate.
    """
    gates: list[ParameterizedGate] = []
    parameter_index = 0

    for site in range(num_qubits):
        append_u3_layer(gates, site, parameter_index)
        parameter_index += 3

    for _layer in range(num_layers):
        for site in range(num_qubits - 1):
            append_two_qubit_block(gates, site, site + 1, parameter_index)
            parameter_index += 9
        append_u3_layer(gates, num_qubits - 1, parameter_index)
        parameter_index += 3

    return ParameterizedCircuit(num_qubits=num_qubits, gates=gates, num_params=parameter_index)


def initialize_near_target_peak(circuit: ParameterizedCircuit, target_probabilities: np.ndarray) -> np.ndarray:
    """Initialize the product layer at the target peak with small entangling noise.

    Returns:
        Initial trainable parameter vector.
    """
    peak_index = int(np.argmax(target_probabilities))
    theta = np.zeros(circuit.num_params, dtype=np.float64)
    for site in range(NUM_QUBITS):
        if (peak_index >> site) & 1:
            theta[3 * site] = np.pi

    rng = np.random.default_rng(RANDOM_SEED)
    theta[3 * NUM_QUBITS :] = rng.normal(
        scale=ENTANGLING_INIT_SCALE,
        size=circuit.num_params - 3 * NUM_QUBITS,
    )
    return theta


def final_statevector(circuit: ParameterizedCircuit, theta: np.ndarray) -> np.ndarray:
    """Evaluate the final optimized statevector.

    Returns:
        Dense final statevector.
    """
    states = forward_states(circuit, theta, np.array([], dtype=np.float64), MPS(NUM_QUBITS), KrotovOptions().truncation)
    return states[-1].to_vec()


def save_trace_csv(result_trace: dict[str, list[float | int | str]], output_path: Path) -> None:
    """Save the Krotov optimization trace."""

    def format_row(idx: int) -> str:
        return ",".join((
            str(result_trace["step"][idx]),
            str(result_trace["phase"][idx]),
            f"{float(result_trace['loss'][idx]):.16e}",
            f"{float(result_trace['fidelity'][idx]):.16e}",
            f"{float(result_trace['step_size'][idx]):.16e}",
            f"{float(result_trace['gradient_norm'][idx]):.16e}",
            f"{float(result_trace['update_norm'][idx]):.16e}",
        ))

    rows = ["iteration,phase,loss,fidelity,step_size,gradient_norm,update_norm"]
    rows.extend(format_row(idx) for idx in range(len(result_trace["step"])))
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def save_plot(
    x_grid: np.ndarray,
    target_probabilities: np.ndarray,
    prepared_probabilities: np.ndarray,
    fidelities: np.ndarray,
    output_path: Path,
) -> None:
    """Save fidelity history and the final probability distribution comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(np.arange(len(fidelities)), fidelities, linewidth=2)
    axes[0].set_xlabel("Krotov iteration")
    axes[0].set_ylabel("target-state fidelity")
    axes[0].set_title("State-preparation convergence")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x_grid, target_probabilities, linewidth=2, label="target")
    axes[1].plot(x_grid, prepared_probabilities, "--", linewidth=2, label="prepared")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("probability")
    axes[1].set_title("Final distribution")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Run the Krotov Gaussian state-preparation experiment."""
    x_grid, target_state, target_probabilities = gaussian_target_state()
    circuit = create_right_sweep_smpd_ansatz(NUM_QUBITS, NUM_LAYERS)
    initial_theta = initialize_near_target_peak(circuit, target_probabilities)

    initial_loss, initial_fidelity = state_preparation_metrics(circuit, initial_theta, target_state)
    options = KrotovOptions(
        max_iterations=MAX_ITERATIONS,
        batch_step_size=BATCH_STEP_SIZE,
        batch_schedule="inverse",
        batch_decay=BATCH_DECAY,
    )
    result = train_krotov_state_preparation_batch(
        circuit,
        target_state,
        initial_theta=initial_theta,
        options=options,
    )

    optimized_state = final_statevector(circuit, result.theta)
    prepared_probabilities = np.abs(optimized_state) ** 2
    final_loss = float(result.trace["loss"][-1])
    final_fidelity = float(result.trace["fidelity"][-1])

    output_dir = Path(__file__).parent
    trace_path = output_dir / "krotov_gaussian_smpd_state_preparation_trace.csv"
    plot_path = output_dir / "krotov_gaussian_smpd_state_preparation.png"
    save_trace_csv(result.trace, trace_path)
    save_plot(x_grid, target_probabilities, prepared_probabilities, np.asarray(result.trace["fidelity"]), plot_path)
    sys.stdout.write(
        "\n".join((
            f"num_qubits={NUM_QUBITS}, num_layers={NUM_LAYERS}",
            f"num_params={circuit.num_params}, num_gates={len(circuit.gates)}",
            f"initial_loss={initial_loss:.12f}, initial_fidelity={initial_fidelity:.12f}",
            f"final_loss={final_loss:.12f}, final_fidelity={final_fidelity:.12f}",
            f"trace_csv={trace_path}",
            f"plot_png={plot_path}",
            "",
        ))
    )


if __name__ == "__main__":
    main()
