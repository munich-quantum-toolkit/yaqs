# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Sweep noisy four-qubit Krotov state preparation over Pauli noise strengths."""

# ruff: noqa: PLC2701

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.krotov_four_qubit_low_fidelity_test import (  # noqa: E402
    NUM_LAYERS,
    NUM_QUBITS,
    build_ansatz,
)
from mqt.yaqs.core.data_structures.mps import MPS  # noqa: E402
from mqt.yaqs.core.data_structures.noise_model import NoiseModel  # noqa: E402
from mqt.yaqs.optimization import (  # noqa: E402
    KrotovTJMOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    noisy_state_preparation_contribution,
    noisy_state_preparation_metrics,
    state_preparation_metrics,
)
from mqt.yaqs.optimization.krotov import (  # noqa: E402
    _noisy_state_preparation_batch_epoch,
    _noisy_state_preparation_online_update,
    _resolve_target_state,
    _state_preparation_batch_epoch,
    _state_preparation_online_update,
    forward_states,
)

NUM_STEPS = int(os.environ.get("YAQS_SWEEP_STEPS", "80"))
NUM_GAMMAS = int(os.environ.get("YAQS_SWEEP_POINTS", "10"))
GAMMA_MIN = float(os.environ.get("YAQS_SWEEP_GAMMA_MIN", "0.001"))
GAMMA_MAX = float(os.environ.get("YAQS_SWEEP_GAMMA_MAX", "0.1"))
TARGET_SEED = int(os.environ.get("YAQS_TARGET_SEED", "123"))
SWITCH_ITERATION = int(os.environ.get("YAQS_SWEEP_SWITCH_ITERATION", "20"))
ONLINE_STEP_SIZE = float(os.environ.get("YAQS_SWEEP_ONLINE_STEP_SIZE", "0.3"))
BATCH_STEP_SIZE = float(os.environ.get("YAQS_SWEEP_BATCH_STEP_SIZE", "0.2"))
NUM_WORKERS = int(os.environ.get("YAQS_SWEEP_WORKERS", str(os.cpu_count() or 1)))


def build_target(circuit: ParameterizedCircuit, truncation: KrotovTruncation, target_seed: int) -> np.ndarray:
    """Generate a reachable four-qubit target state from one random seed.

    Returns:
        Dense target statevector.
    """
    rng = np.random.default_rng(target_seed)
    target_theta = rng.normal(scale=1.2, size=circuit.num_params)
    return forward_states(circuit, target_theta, np.array([]), MPS(NUM_QUBITS), truncation)[-1].to_vec()


def build_noise_model(gamma: float) -> NoiseModel:
    """Build local Pauli X/Y/Z noise on every qubit.

    Returns:
        Local Pauli noise model for the requested strength.
    """
    return NoiseModel([
        {"name": name, "sites": [site], "strength": gamma}
        for site in range(NUM_QUBITS)
        for name in ("pauli_x", "pauli_y", "pauli_z")
    ])


def run_noiseless_reference(
    circuit: ParameterizedCircuit,
    target: np.ndarray,
    initial_theta: np.ndarray,
    truncation: KrotovTruncation,
) -> tuple[float, float]:
    """Run the deterministic hybrid reference for the same number of steps.

    Returns:
        Initial and final noiseless fidelities.
    """
    theta = initial_theta.copy()
    target_mps = _resolve_target_state(target, circuit.num_qubits)
    initial_fidelity = state_preparation_metrics(circuit, theta, target, truncation=truncation)[1]
    for iteration in range(1, NUM_STEPS + 1):
        if iteration <= SWITCH_ITERATION:
            theta, _contribution = _state_preparation_online_update(
                circuit,
                theta,
                target_mps,
                ONLINE_STEP_SIZE,
                MPS(NUM_QUBITS),
                truncation,
            )
        else:
            theta, _gradient_norm, _loss, _fidelity = _state_preparation_batch_epoch(
                circuit,
                theta,
                target_mps,
                BATCH_STEP_SIZE,
                MPS(NUM_QUBITS),
                truncation,
            )
    final_fidelity = state_preparation_metrics(circuit, theta, target, truncation=truncation)[1]
    return initial_fidelity, final_fidelity


def noisy_options(
    update: Literal["independent", "cross"],
    num_trajectories: int,
    seed: int,
) -> KrotovTJMOptions:
    """Build trajectory options shared across the sweep.

    Returns:
        TJM options for one noisy run.
    """
    return KrotovTJMOptions(
        num_trajectories=num_trajectories,
        random_seed=seed,
        trajectory_update=update,
        apply_noise_to="two-qubit",
    )


def run_noisy_strength(
    circuit: ParameterizedCircuit,
    target: np.ndarray,
    initial_theta: np.ndarray,
    truncation: KrotovTruncation,
    gamma: float,
    update: Literal["independent", "cross"],
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    """Run one noisy hybrid optimizer variant for one noise strength.

    Returns:
        Detailed per-iteration rows and a compact summary row.
    """
    if update == "independent":
        train_trajectories = 16
        validation_trajectories = 32
    else:
        train_trajectories = 4
        validation_trajectories = 12

    train_options = noisy_options(update, train_trajectories, 222)
    validation_options = noisy_options(update, validation_trajectories, 999)
    noise_model = build_noise_model(gamma)
    target_mps = _resolve_target_state(target, circuit.num_qubits)

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

    theta = initial_theta.copy()
    detail_rows: list[dict[str, float | int | str]] = []
    best_validation_fidelity = 0.0
    best_iteration = 0
    final_validation_fidelity = 0.0
    final_noiseless_fidelity = 0.0

    for iteration in range(NUM_STEPS + 1):
        _loss, fixed_validation_fidelity, _trajectory_fidelities = noisy_state_preparation_metrics(
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
        if fixed_validation_fidelity > best_validation_fidelity:
            best_validation_fidelity = fixed_validation_fidelity
            best_iteration = iteration

        gradient_norm = 0.0
        if iteration < NUM_STEPS:
            phase_iteration = iteration + 1
            if phase_iteration <= SWITCH_ITERATION:
                theta_next, contribution = _noisy_state_preparation_online_update(
                    circuit,
                    theta,
                    target_mps,
                    noise_model,
                    train_options,
                    ONLINE_STEP_SIZE,
                    MPS(NUM_QUBITS),
                    truncation,
                    phase_iteration,
                )
                gradient_norm = float(np.linalg.norm(contribution))
            else:
                theta_next, gradient_norm, _loss, _fidelity = _noisy_state_preparation_batch_epoch(
                    circuit,
                    theta,
                    target_mps,
                    noise_model,
                    train_options,
                    BATCH_STEP_SIZE,
                    MPS(NUM_QUBITS),
                    truncation,
                    phase_iteration,
                )

        detail_rows.append({
            "gamma": gamma,
            "target_seed": TARGET_SEED,
            "method": update,
            "iteration": iteration,
            "fixed_validation_fidelity": fixed_validation_fidelity,
            "noiseless_fidelity": noiseless_fidelity,
            "gradient_norm": gradient_norm,
            "train_trajectories": train_trajectories,
            "validation_trajectories": validation_trajectories,
        })
        final_validation_fidelity = fixed_validation_fidelity
        final_noiseless_fidelity = noiseless_fidelity
        if iteration < NUM_STEPS:
            theta = theta_next

    summary_row: dict[str, float | int | str] = {
        "gamma": gamma,
        "target_seed": TARGET_SEED,
        "method": update,
        "initial_validation_fidelity": float(detail_rows[0]["fixed_validation_fidelity"]),
        "final_validation_fidelity": final_validation_fidelity,
        "best_validation_fidelity": best_validation_fidelity,
        "best_iteration": best_iteration,
        "final_noiseless_fidelity": final_noiseless_fidelity,
        "train_trajectories": train_trajectories,
        "validation_trajectories": validation_trajectories,
    }
    return detail_rows, summary_row


def save_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    """Save rows as CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_plot(
    summary_rows: list[dict[str, float | int | str]],
    noiseless_initial: float,
    noiseless_final: float,
    path: Path,
) -> None:
    """Plot final and best fixed-validation fidelity versus noise strength."""
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for method, color in [("independent", "#9b2226"), ("cross", "#ca6702")]:
        rows = [row for row in summary_rows if row["method"] == method]
        gammas = np.asarray([float(row["gamma"]) for row in rows])
        final = np.asarray([float(row["final_validation_fidelity"]) for row in rows])
        best = np.asarray([float(row["best_validation_fidelity"]) for row in rows])
        ax.plot(gammas, final, marker="o", linewidth=2.0, color=color, label=f"{method} final")
        ax.plot(gammas, best, marker="x", linestyle="--", linewidth=1.6, color=color, label=f"{method} best")

    ax.axhline(noiseless_initial, color="#6c757d", linestyle=":", linewidth=1.6, label="initial noiseless")
    ax.axhline(noiseless_final, color="#005f73", linestyle=":", linewidth=1.8, label=f"{NUM_STEPS}-step noiseless")
    ax.set_xlabel("Pauli noise strength gamma")
    ax.set_ylabel("fixed-validation target-state fidelity")
    ax.set_title(f"Four-qubit noisy hybrid Krotov sweep, target seed {TARGET_SEED}")
    ax.set_xlim(GAMMA_MIN, GAMMA_MAX)
    ax.set_xticks(np.linspace(GAMMA_MIN, GAMMA_MAX, NUM_GAMMAS))
    ax.tick_params(axis="x", labelrotation=30)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Run the noise-strength sweep."""
    start = time.perf_counter()
    truncation = KrotovTruncation()
    circuit = build_ansatz(NUM_QUBITS, NUM_LAYERS)
    target = build_target(circuit, truncation, TARGET_SEED)
    initial_theta = np.zeros(circuit.num_params, dtype=np.float64)
    gammas = np.linspace(GAMMA_MIN, GAMMA_MAX, NUM_GAMMAS)

    noiseless_initial, noiseless_final = run_noiseless_reference(circuit, target, initial_theta, truncation)
    detail_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    sys.stdout.write(
        f"num_qubits={NUM_QUBITS}, num_layers={NUM_LAYERS}, num_params={circuit.num_params}, "
        f"steps={NUM_STEPS}, switch={SWITCH_ITERATION}, online_step={ONLINE_STEP_SIZE}, "
        f"batch_step={BATCH_STEP_SIZE}, target_seed={TARGET_SEED}, workers={NUM_WORKERS}\n"
    )
    sys.stdout.write(f"gammas={','.join(f'{gamma:.6g}' for gamma in gammas)}\n")
    sys.stdout.write(f"noiseless_reference={noiseless_initial:.12f}->{noiseless_final:.12f}\n")
    sys.stdout.flush()

    updates: tuple[Literal["independent", "cross"], ...] = ("independent", "cross")
    jobs: list[tuple[int, float, Literal["independent", "cross"]]] = [
        (gamma_index, float(gamma), update)
        for gamma_index, gamma in enumerate(gammas, start=1)
        for update in updates
    ]
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(run_noisy_strength, circuit, target, initial_theta, truncation, gamma, update): (
                gamma_index,
                gamma,
                update,
            )
            for gamma_index, gamma, update in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            _gamma_index, gamma, update = futures[future]
            rows, summary = future.result()
            detail_rows.extend(rows)
            summary_rows.append(summary)
            percent = 100.0 * completed / len(jobs)
            sys.stdout.write(
                f"progress={percent:6.2f}% gamma={gamma:.6g} update={update} "
                f"final={float(summary['final_validation_fidelity']):.12f} "
                f"best={float(summary['best_validation_fidelity']):.12f} "
                f"elapsed={time.perf_counter() - start:.1f}s\n"
            )
            sys.stdout.flush()

    summary_rows.sort(key=lambda row: (float(row["gamma"]), str(row["method"])))
    detail_rows.sort(key=lambda row: (float(row["gamma"]), str(row["method"]), int(row["iteration"])))

    output_dir = Path(__file__).parent
    filename_prefix = f"krotov_four_qubit_hybrid_noise_strength_sweep_seed{TARGET_SEED}_steps{NUM_STEPS}"
    detail_path = output_dir / f"{filename_prefix}_detail.csv"
    summary_path = output_dir / f"{filename_prefix}_summary.csv"
    plot_path = output_dir / f"{filename_prefix}.png"
    save_csv(detail_rows, detail_path)
    save_csv(summary_rows, summary_path)
    save_summary_plot(summary_rows, noiseless_initial, noiseless_final, plot_path)

    sys.stdout.write(
        "\n".join((
            f"detail_csv={detail_path}",
            f"summary_csv={summary_path}",
            f"plot_png={plot_path}",
            f"elapsed_seconds={time.perf_counter() - start:.3f}",
            "",
        ))
    )


if __name__ == "__main__":
    main()
