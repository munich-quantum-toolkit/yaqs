# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Four-qubit low-initial-fidelity Krotov state-preparation diagnostic."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mqt.yaqs.core.data_structures.mps import MPS  # noqa: E402
from mqt.yaqs.core.data_structures.noise_model import NoiseModel  # noqa: E402
from mqt.yaqs.optimization import (  # noqa: E402
    KrotovTJMOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    noisy_state_preparation_contribution,
    noisy_state_preparation_metrics,
    state_preparation_contribution,
    state_preparation_metrics,
)
from mqt.yaqs.optimization.krotov import forward_states  # noqa: E402

NUM_QUBITS = 4
NUM_LAYERS = 3
NUM_STEPS = int(os.environ.get("YAQS_FOUR_QUBIT_STEPS", "30"))
GAMMA = 0.002
TARGET_SEED = 123


def build_ansatz(num_qubits: int, num_layers: int) -> ParameterizedCircuit:
    """Build a small nearest-neighbor ansatz with a reachable random target.

    Returns:
        Parameterized circuit with one scalar parameter per primitive gate.
    """
    gates: list[ParameterizedGate] = []
    parameter_index = 0
    for _layer in range(num_layers):
        for site in range(num_qubits):
            gates.append(ParameterizedGate("ry", (site,), param_index=parameter_index))
            parameter_index += 1
            gates.append(ParameterizedGate("rz", (site,), param_index=parameter_index))
            parameter_index += 1
        for site in range(num_qubits - 1):
            gates.append(ParameterizedGate("rxx", (site, site + 1), param_index=parameter_index))
            parameter_index += 1
            gates.append(ParameterizedGate("ryy", (site, site + 1), param_index=parameter_index))
            parameter_index += 1
            gates.append(ParameterizedGate("rzz", (site, site + 1), param_index=parameter_index))
            parameter_index += 1

    for site in range(num_qubits):
        gates.append(ParameterizedGate("ry", (site,), param_index=parameter_index))
        parameter_index += 1
        gates.append(ParameterizedGate("rz", (site,), param_index=parameter_index))
        parameter_index += 1

    return ParameterizedCircuit(num_qubits, gates, num_params=parameter_index)


def build_target(circuit: ParameterizedCircuit, truncation: KrotovTruncation) -> np.ndarray:
    """Generate a reachable four-qubit target state.

    Returns:
        Dense target statevector.
    """
    rng = np.random.default_rng(TARGET_SEED)
    target_theta = rng.normal(scale=1.2, size=circuit.num_params)
    return forward_states(circuit, target_theta, np.array([]), MPS(NUM_QUBITS), truncation)[-1].to_vec()


def build_noise_model() -> NoiseModel:
    """Build weak local Pauli X/Y/Z noise on every qubit.

    Returns:
        Local Pauli noise model.
    """
    return NoiseModel([
        {"name": name, "sites": [site], "strength": GAMMA}
        for site in range(NUM_QUBITS)
        for name in ("pauli_x", "pauli_y", "pauli_z")
    ])


def run_noiseless(
    circuit: ParameterizedCircuit,
    initial_theta: np.ndarray,
    target: np.ndarray,
    truncation: KrotovTruncation,
) -> list[dict[str, float | int | str]]:
    """Run deterministic Krotov from the bad initial state.

    Returns:
        Per-iteration trace rows.
    """
    theta = initial_theta.copy()
    rows: list[dict[str, float | int | str]] = []
    for iteration in range(NUM_STEPS + 1):
        loss, fidelity = state_preparation_metrics(circuit, theta, target, truncation=truncation)
        rows.append({
            "method": "noiseless",
            "iteration": iteration,
            "fixed_noisy_fidelity": "",
            "noiseless_fidelity": fidelity,
            "loss": loss,
            "gradient_norm": 0.0,
        })
        if iteration < NUM_STEPS:
            contribution, _loss, _fidelity = state_preparation_contribution(
                circuit,
                theta,
                target,
                MPS(NUM_QUBITS),
                truncation,
            )
            rows[-1]["gradient_norm"] = float(np.linalg.norm(contribution))
            theta -= 0.3 * contribution
    return rows


def run_noisy(
    circuit: ParameterizedCircuit,
    initial_theta: np.ndarray,
    target: np.ndarray,
    noise_model: NoiseModel,
    truncation: KrotovTruncation,
    *,
    update: Literal["independent", "cross"],
    train_trajectories: int,
    validation_trajectories: int,
) -> list[dict[str, float | int | str]]:
    """Run noisy Krotov and validate every step on fixed trajectory maps.

    Returns:
        Per-iteration trace rows.
    """
    theta = initial_theta.copy()
    train_options = KrotovTJMOptions(
        num_trajectories=train_trajectories,
        random_seed=222,
        trajectory_update=update,
        apply_noise_to="two-qubit",
    )
    validation_options = KrotovTJMOptions(
        num_trajectories=validation_trajectories,
        random_seed=999,
        trajectory_update=update,
        apply_noise_to="two-qubit",
    )
    _contribution, _loss, _fidelity, validation_trajectories_data = noisy_state_preparation_contribution(
        circuit,
        initial_theta,
        target,
        noise_model,
        validation_options,
        MPS(NUM_QUBITS),
        truncation,
    )
    fixed_validation_maps = [trajectory.noise_maps for trajectory in validation_trajectories_data]

    rows: list[dict[str, float | int | str]] = []
    for iteration in range(NUM_STEPS + 1):
        loss, fixed_noisy_fidelity, _trajectory_fidelities = noisy_state_preparation_metrics(
            circuit,
            theta,
            target,
            noise_model,
            validation_options,
            truncation=truncation,
            fixed_noise_maps=fixed_validation_maps,
        )
        _noiseless_loss, noiseless_fidelity = state_preparation_metrics(
            circuit,
            theta,
            target,
            truncation=truncation,
        )
        rows.append({
            "method": f"noisy-{update}",
            "iteration": iteration,
            "fixed_noisy_fidelity": fixed_noisy_fidelity,
            "noiseless_fidelity": noiseless_fidelity,
            "loss": loss,
            "gradient_norm": 0.0,
        })
        if iteration < NUM_STEPS:
            contribution, _loss, _fidelity, _trajectories = noisy_state_preparation_contribution(
                circuit,
                theta,
                target,
                noise_model,
                train_options,
                MPS(NUM_QUBITS),
                truncation,
                iteration=iteration,
            )
            rows[-1]["gradient_norm"] = float(np.linalg.norm(contribution))
            theta -= 0.2 * contribution
    return rows


def save_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    """Save diagnostic trace rows."""
    fieldnames = ["method", "iteration", "fixed_noisy_fidelity", "noiseless_fidelity", "loss", "gradient_norm"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict[str, float | int | str]], path: Path) -> None:
    """Plot fidelity improvement from the bad initial state."""
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for method, color in [
        ("noiseless", "#005f73"),
        ("noisy-independent", "#9b2226"),
        ("noisy-cross", "#ca6702"),
    ]:
        method_rows = [row for row in rows if row["method"] == method]
        x = np.asarray([int(row["iteration"]) for row in method_rows])
        if method == "noiseless":
            y = np.asarray([float(row["noiseless_fidelity"]) for row in method_rows])
            label = "noiseless fidelity"
        else:
            y = np.asarray([float(row["fixed_noisy_fidelity"]) for row in method_rows])
            label = method.replace("-", " ") + " fixed validation"
        ax.plot(x, y, linewidth=2.2, label=label, color=color)

    ax.set_xlabel("Krotov step")
    ax.set_ylabel("target-state fidelity")
    ax.set_title("Four-qubit low-initial-fidelity state preparation")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Run the four-qubit Krotov diagnostic."""
    truncation = KrotovTruncation()
    circuit = build_ansatz(NUM_QUBITS, NUM_LAYERS)
    target = build_target(circuit, truncation)
    initial_theta = np.zeros(circuit.num_params, dtype=np.float64)
    noise_model = build_noise_model()

    rows = run_noiseless(circuit, initial_theta, target, truncation)
    rows.extend(
        run_noisy(
            circuit,
            initial_theta,
            target,
            noise_model,
            truncation,
            update="independent",
            train_trajectories=16,
            validation_trajectories=32,
        )
    )
    rows.extend(
        run_noisy(
            circuit,
            initial_theta,
            target,
            noise_model,
            truncation,
            update="cross",
            train_trajectories=4,
            validation_trajectories=12,
        )
    )

    output_dir = Path(__file__).parent
    csv_path = output_dir / f"krotov_four_qubit_low_fidelity_test_steps{NUM_STEPS}.csv"
    plot_path = output_dir / f"krotov_four_qubit_low_fidelity_test_steps{NUM_STEPS}.png"
    save_csv(rows, csv_path)
    save_plot(rows, plot_path)

    initial_noiseless = float(rows[0]["noiseless_fidelity"])
    final_noiseless = float([row for row in rows if row["method"] == "noiseless"][-1]["noiseless_fidelity"])
    independent_rows = [row for row in rows if row["method"] == "noisy-independent"]
    cross_rows = [row for row in rows if row["method"] == "noisy-cross"]
    sys.stdout.write(
        "\n".join((
            f"num_qubits={NUM_QUBITS}, num_layers={NUM_LAYERS}, num_params={circuit.num_params}",
            f"gamma={GAMMA}",
            f"initial_noiseless_fidelity={initial_noiseless:.12f}",
            f"final_noiseless_fidelity={final_noiseless:.12f}",
            ("noisy_independent_fixed_validation="
            f"{float(independent_rows[0]['fixed_noisy_fidelity']):.12f}"
            f" -> {float(independent_rows[-1]['fixed_noisy_fidelity']):.12f}"),
            ("noisy_cross_fixed_validation="
            f"{float(cross_rows[0]['fixed_noisy_fidelity']):.12f}"
            f" -> {float(cross_rows[-1]['fixed_noisy_fidelity']):.12f}"),
            f"trace_csv={csv_path}",
            f"plot_png={plot_path}",
            "",
        ))
    )


if __name__ == "__main__":
    main()
