# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Noisy Krotov state preparation for the 8-qubit Gaussian SMPD ansatz."""

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.krotov_gaussian_smpd_state_preparation import (  # noqa: E402
    BATCH_DECAY,
    BATCH_STEP_SIZE,
    MAX_ITERATIONS,
    NUM_LAYERS,
    NUM_QUBITS,
    create_right_sweep_smpd_ansatz,
    gaussian_target_state,
    initialize_near_target_peak,
)
from mqt.yaqs.core.data_structures.mps import MPS  # noqa: E402
from mqt.yaqs.core.data_structures.noise_model import NoiseModel  # noqa: E402
from mqt.yaqs.optimization import (  # noqa: E402
    KrotovNoiseMap,
    KrotovOptions,
    KrotovTJMOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    noisy_state_preparation_contribution,
    train_krotov_state_preparation_batch,
)
from mqt.yaqs.optimization.krotov import forward_tjm_trajectory  # noqa: E402

GAMMA = float(os.environ.get("YAQS_GAMMA", "0.02"))
NUM_TRAJECTORIES = 50
NOISY_OPTIMIZATION_STEPS = int(os.environ.get("YAQS_NOISY_OPTIMIZATION_STEPS", "50"))
NOISY_STEP_SIZE = 0.1
RANDOM_SEED = 321
BASELINE_FIDELITY = 0.55
TrajectoryUpdateMode = Literal["independent", "cross"]
_TRAJECTORY_UPDATE_ENV = os.environ.get("YAQS_TRAJECTORY_UPDATE", "cross")
if _TRAJECTORY_UPDATE_ENV not in {"independent", "cross"}:
    msg = f"YAQS_TRAJECTORY_UPDATE must be 'independent' or 'cross', got {_TRAJECTORY_UPDATE_ENV!r}."
    raise ValueError(msg)
TRAJECTORY_UPDATE = cast("TrajectoryUpdateMode", _TRAJECTORY_UPDATE_ENV)
MAX_WORKERS = min(NUM_TRAJECTORIES, os.cpu_count() or 1)


def logical_smpd_noise_gate_indices(num_qubits: int, num_layers: int) -> tuple[int, ...]:
    """Return the final primitive index of each logical SMPD two-qubit block.

    The Krotov ansatz decomposes one logical two-qubit SMPD block into local U3
    rotations followed by ``rxx``, ``ryy``, and ``rzz`` primitives. Circuit TJM
    noise should be sampled once for that logical block, so the noise mask points
    to the final ``rzz`` primitive of each block.
    """
    indices: list[int] = []
    gate_index = 3 * num_qubits
    for _layer in range(num_layers):
        for _site in range(num_qubits - 1):
            gate_index += 9
            indices.append(gate_index - 1)
        gate_index += 3
    return tuple(indices)


def build_noise_model(gamma: float) -> NoiseModel:
    """Build the requested local X/Y/Z Pauli noise model.

    Returns:
        Noise model with one Pauli X/Y/Z process on each qubit.
    """
    return NoiseModel([
        {"name": name, "sites": [site], "strength": gamma}
        for site in range(NUM_QUBITS)
        for name in ("pauli_x", "pauli_y", "pauli_z")
    ])


def format_gamma_for_path(gamma: float) -> str:
    """Format a noise strength for stable output filenames.

    Returns:
        Path-safe gamma string.
    """
    return f"{gamma:g}".replace(".", "p")


def load_or_create_warm_start(
    cache_path: Path,
    target_probabilities: np.ndarray,
    target_state: np.ndarray,
) -> np.ndarray:
    """Load the deterministic warm start, or create it if the cache is absent.

    Returns:
        Warm-start trainable parameter vector.
    """
    circuit = create_right_sweep_smpd_ansatz(NUM_QUBITS, NUM_LAYERS)
    if cache_path.exists():
        return np.load(cache_path)

    initial_theta = initialize_near_target_peak(circuit, target_probabilities)
    result = train_krotov_state_preparation_batch(
        circuit,
        target_state,
        initial_theta=initial_theta,
        options=KrotovOptions(
            max_iterations=MAX_ITERATIONS,
            batch_step_size=BATCH_STEP_SIZE,
            batch_schedule="inverse",
            batch_decay=BATCH_DECAY,
        ),
    )
    np.save(cache_path, result.theta)
    return result.theta


def normalized_trajectory_fidelity(target_state: np.ndarray, state: MPS) -> float:
    """Return ``|<target|state>|^2 / <state|state>`` for one trajectory."""
    norm = float(np.real(state.scalar_product(state)))
    if norm <= 0.0:
        return 0.0
    overlap = np.vdot(target_state, state.to_vec())
    return float(abs(overlap) ** 2 / norm)


def apply_dense_operator(
    state: np.ndarray,
    matrix: np.ndarray,
    sites: tuple[int, ...],
    num_qubits: int,
) -> np.ndarray:
    """Apply a one- or two-site operator to a dense YAQS little-endian statevector.

    Returns:
        Updated dense statevector.
    """
    axes = tuple(num_qubits - 1 - site for site in sites)
    tensor = state.reshape((2,) * num_qubits)
    remaining_axes = tuple(axis for axis in range(num_qubits) if axis not in axes)
    permutation = axes + remaining_axes
    inverse_permutation = np.argsort(permutation)
    front_dim = 2 ** len(sites)
    rest_dim = 2 ** (num_qubits - len(sites))
    work = np.transpose(tensor, permutation).reshape(front_dim, rest_dim)
    updated = matrix @ work
    return np.transpose(updated.reshape((2,) * num_qubits), inverse_permutation).reshape(-1)


def apply_dense_noise_adjoint(
    state: np.ndarray,
    noise_map: KrotovNoiseMap,
    num_qubits: int,
) -> np.ndarray:
    """Apply the pathwise adjoint of a realized noise map to a dense costate.

    Returns:
        Updated dense costate.
    """
    current = state
    for matrix, sites in reversed(noise_map.operators):
        current = apply_dense_operator(current, matrix.conj().T, sites, num_qubits)
    return current


def trajectory_seed(base_seed: int, iteration: int, trajectory_index: int) -> int:
    """Build the same deterministic trajectory seed as the optimizer core.

    Returns:
        Integer seed for one trajectory.
    """
    return int(base_seed + 1_000_003 * iteration + trajectory_index)


def trajectory_job(
    args: tuple[
        int,
        int,
        ParameterizedCircuit,
        np.ndarray,
        np.ndarray,
        NoiseModel,
        KrotovTJMOptions,
        KrotovTruncation,
    ],
) -> tuple[int, np.ndarray, float, float, int]:
    """Evaluate one trajectory contribution.

    The single-trajectory contribution is averaged by the parent process. This
    keeps all 50 TJM trajectories parallel while reusing the public noisy Krotov
    contribution function.

    Returns:
        Trajectory index, contribution, raw fidelity, normalized fidelity, and
        number of sampled jumps.
    """
    trajectory_index, iteration, circuit, theta, target_state, noise_model, tjm_options, truncation = args
    single_tjm_options = KrotovTJMOptions(
        num_trajectories=1,
        random_seed=tjm_options.random_seed + trajectory_index,
        dt=tjm_options.dt,
        apply_noise_to=tjm_options.apply_noise_to,
        noisy_gate_indices=tjm_options.noisy_gate_indices,
    )
    contribution, _loss, raw_fidelity, trajectories = noisy_state_preparation_contribution(
        circuit,
        theta,
        target_state,
        noise_model,
        single_tjm_options,
        MPS(circuit.num_qubits),
        truncation,
        iteration=iteration,
    )
    trajectory = trajectories[0]
    normalized_fidelity = normalized_trajectory_fidelity(target_state, trajectory.states[-1])
    jump_count = sum(noise_map.jump_process_index is not None for noise_map in trajectory.noise_maps)
    return trajectory_index, contribution, raw_fidelity, normalized_fidelity, jump_count


def forward_trajectory_job(
    args: tuple[
        int,
        int,
        ParameterizedCircuit,
        np.ndarray,
        np.ndarray,
        NoiseModel,
        KrotovTJMOptions,
        KrotovTruncation,
    ],
) -> tuple[int, np.ndarray, np.ndarray, list[KrotovNoiseMap], float, float, int]:
    """Generate one TJM trajectory and return dense storage for cross updates.

    Returns:
        Trajectory index, dense gate outputs, final vector, realized maps, raw
        fidelity, normalized fidelity, and jump count.
    """
    trajectory_index, iteration, circuit, theta, target_state, noise_model, tjm_options, truncation = args
    rng = np.random.default_rng(trajectory_seed(tjm_options.random_seed, iteration, trajectory_index))
    trajectory = forward_tjm_trajectory(
        circuit,
        theta,
        np.array([], dtype=np.float64),
        MPS(circuit.num_qubits),
        truncation,
        noise_model,
        tjm_options,
        rng,
    )
    gate_outputs = np.asarray([state.to_vec() for state in trajectory.gate_outputs], dtype=np.complex128)
    final_state = trajectory.states[-1]
    final_vector = np.asarray(final_state.to_vec(), dtype=np.complex128)
    raw_fidelity = float(abs(np.vdot(target_state, final_vector)) ** 2)
    norm = float(np.real(final_state.scalar_product(final_state)))
    normalized_fidelity = raw_fidelity / norm if norm > 0.0 else 0.0
    jump_count = sum(noise_map.jump_process_index is not None for noise_map in trajectory.noise_maps)
    return (
        trajectory_index,
        gate_outputs,
        final_vector,
        trajectory.noise_maps,
        raw_fidelity,
        normalized_fidelity,
        jump_count,
    )


def parallel_noisy_contribution(
    circuit: ParameterizedCircuit,
    theta: np.ndarray,
    target_state: np.ndarray,
    noise_model: NoiseModel,
    tjm_options: KrotovTJMOptions,
    truncation: KrotovTruncation,
    iteration: int,
    executor: ProcessPoolExecutor,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute one noisy Krotov contribution from parallel trajectories.

    Returns:
        Contribution, mean normalized fidelity, normalized fidelities, raw
        fidelities, and jump counts.
    """
    jobs = [
        (traj_idx, iteration, circuit, theta, target_state, noise_model, tjm_options, truncation)
        for traj_idx in range(tjm_options.num_trajectories)
    ]
    contribution = np.zeros(circuit.num_params, dtype=np.float64)
    raw_fidelities = np.zeros(tjm_options.num_trajectories, dtype=np.float64)
    normalized_fidelities = np.zeros(tjm_options.num_trajectories, dtype=np.float64)
    jump_counts = np.zeros(tjm_options.num_trajectories, dtype=np.int64)

    for (
        traj_idx,
        traj_contribution,
        raw_fidelity,
        normalized_fidelity,
        jump_count,
    ) in executor.map(trajectory_job, jobs):
        contribution += traj_contribution
        raw_fidelities[traj_idx] = raw_fidelity
        normalized_fidelities[traj_idx] = normalized_fidelity
        jump_counts[traj_idx] = jump_count

    contribution /= tjm_options.num_trajectories
    return contribution, float(np.mean(normalized_fidelities)), normalized_fidelities, raw_fidelities, jump_counts


def dense_cross_costates(
    circuit: ParameterizedCircuit,
    theta: np.ndarray,
    target_state: np.ndarray,
    noise_maps: list[list[KrotovNoiseMap]],
) -> np.ndarray:
    """Build dense stale target costates for every trajectory and gate.

    Returns:
        Array of shape ``(num_trajectories, num_gates, 2**num_qubits)``.
    """
    costates = np.empty((len(noise_maps), len(circuit.gates), target_state.size), dtype=np.complex128)
    x = np.array([], dtype=np.float64)
    for traj_idx, trajectory_maps in enumerate(noise_maps):
        lambda_after_noise = target_state.copy()
        for gate_index in range(len(circuit.gates) - 1, -1, -1):
            gate = circuit.gates[gate_index]
            chi_tilde = apply_dense_noise_adjoint(
                lambda_after_noise,
                trajectory_maps[gate_index],
                circuit.num_qubits,
            )
            costates[traj_idx, gate_index, :] = chi_tilde
            matrix, sites = circuit.gate_matrix(gate, theta, x)
            lambda_after_noise = apply_dense_operator(chi_tilde, matrix.conj().T, sites, circuit.num_qubits)
    return costates


def dense_cross_contribution(
    circuit: ParameterizedCircuit,
    gate_outputs: np.ndarray,
    costates: np.ndarray,
) -> np.ndarray:
    """Evaluate the cross-trajectory update with dense vectorized overlaps.

    Returns:
        Trainable-parameter contribution vector.
    """
    contribution = np.zeros(circuit.num_params, dtype=np.float64)
    scale = 1.0 / (gate_outputs.shape[0] * costates.shape[0])
    for gate_index, gate in enumerate(circuit.gates):
        if gate.param_index is None:
            continue
        operator, sites = circuit.derivative_operator(gate)
        psi = gate_outputs[:, gate_index, :]
        xi = costates[:, gate_index, :]
        derivative_psi = np.asarray([
            apply_dense_operator(state, operator, sites, circuit.num_qubits) for state in psi
        ])
        derivative_overlaps = xi.conj() @ derivative_psi.T
        density_overlaps = psi.conj() @ xi.T
        signal = -gate.angle_scale * 2.0 * float(np.real(np.sum(derivative_overlaps * density_overlaps.T))) * scale
        contribution[gate.param_index] += signal
    return contribution


def parallel_noisy_cross_contribution(
    circuit: ParameterizedCircuit,
    theta: np.ndarray,
    target_state: np.ndarray,
    noise_model: NoiseModel,
    tjm_options: KrotovTJMOptions,
    truncation: KrotovTruncation,
    iteration: int,
    executor: ProcessPoolExecutor,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute one cross-trajectory update from parallel TJM trajectories.

    Returns:
        Contribution, mean normalized fidelity, normalized fidelities, raw
        fidelities, and jump counts.
    """
    jobs = [
        (traj_idx, iteration, circuit, theta, target_state, noise_model, tjm_options, truncation)
        for traj_idx in range(tjm_options.num_trajectories)
    ]
    dim = target_state.size
    gate_outputs = np.empty((tjm_options.num_trajectories, len(circuit.gates), dim), dtype=np.complex128)
    noise_maps: list[list[KrotovNoiseMap]] = [[] for _ in range(tjm_options.num_trajectories)]
    raw_fidelities = np.zeros(tjm_options.num_trajectories, dtype=np.float64)
    normalized_fidelities = np.zeros(tjm_options.num_trajectories, dtype=np.float64)
    jump_counts = np.zeros(tjm_options.num_trajectories, dtype=np.int64)

    for (
        traj_idx,
        trajectory_gate_outputs,
        _final_vector,
        trajectory_noise_maps,
        raw_fidelity,
        normalized_fidelity,
        jump_count,
    ) in executor.map(forward_trajectory_job, jobs):
        gate_outputs[traj_idx, :, :] = trajectory_gate_outputs
        noise_maps[traj_idx] = trajectory_noise_maps
        raw_fidelities[traj_idx] = raw_fidelity
        normalized_fidelities[traj_idx] = normalized_fidelity
        jump_counts[traj_idx] = jump_count

    costates = dense_cross_costates(circuit, theta, target_state, noise_maps)
    contribution = dense_cross_contribution(circuit, gate_outputs, costates)
    return contribution, float(np.mean(normalized_fidelities)), normalized_fidelities, raw_fidelities, jump_counts


def save_trace_csv(trace: list[dict[str, float | int]], trajectory_fidelities: np.ndarray, path: Path) -> None:
    """Save the mean and per-trajectory fidelity history."""
    fieldnames = [
        "iteration",
        "mean_fidelity",
        "raw_mean_fidelity",
        "loss",
        "step_size",
        "gradient_norm",
        "mean_jump_count",
        *[f"traj_{idx}" for idx in range(trajectory_fidelities.shape[1])],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, fidelities in zip(trace, trajectory_fidelities, strict=True):
            writer.writerow({
                **row,
                **{f"traj_{idx}": f"{float(value):.16e}" for idx, value in enumerate(fidelities)},
            })


def save_plot(trace: list[dict[str, float | int]], trajectory_fidelities: np.ndarray, path: Path) -> None:
    """Plot mean fidelity and all individual trajectory fidelities."""
    iterations = np.asarray([int(row["iteration"]) for row in trace])
    mean_fidelities = np.asarray([float(row["mean_fidelity"]) for row in trace])

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for traj_idx in range(trajectory_fidelities.shape[1]):
        ax.plot(iterations, trajectory_fidelities[:, traj_idx], color="#8a8a8a", alpha=0.22, linewidth=0.8)

    ax.plot(iterations, mean_fidelities, color="#005f73", linewidth=2.4, label="mean fidelity")
    ax.axhline(BASELINE_FIDELITY, color="#ae2012", linestyle="--", linewidth=1.5, label="55% baseline")
    ax.set_xlabel("Krotov optimization step")
    ax.set_ylabel("target-state fidelity")
    ax.set_title("Noisy Gaussian SMPD state preparation")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def progress_message(iteration: int, stage: str, stage_fraction: float, start: float) -> str:
    """Build a progress message for long noisy optimization runs.

    Returns:
        Single-line progress string.
    """
    total_iterations = NOISY_OPTIMIZATION_STEPS + 1
    completed = iteration + stage_fraction
    percent = 100.0 * completed / total_iterations
    elapsed = time.perf_counter() - start
    return (
        f"progress={percent:6.2f}% step={iteration:02d}/{NOISY_OPTIMIZATION_STEPS} "
        f"stage={stage} elapsed={elapsed:.1f}s"
    )


def main() -> None:
    """Run the noisy state-preparation experiment."""
    _x_grid, target_state, target_probabilities = gaussian_target_state()
    circuit = create_right_sweep_smpd_ansatz(NUM_QUBITS, NUM_LAYERS)
    output_dir = Path(__file__).parent
    warm_start_path = output_dir / "krotov_gaussian_smpd_warm_start_theta.npy"
    theta = load_or_create_warm_start(warm_start_path, target_probabilities, target_state)

    noise_gate_indices = logical_smpd_noise_gate_indices(NUM_QUBITS, NUM_LAYERS)
    noise_model = build_noise_model(GAMMA)
    tjm_options = KrotovTJMOptions(
        num_trajectories=NUM_TRAJECTORIES,
        random_seed=RANDOM_SEED,
        apply_noise_to="two-qubit",
        noisy_gate_indices=noise_gate_indices,
        trajectory_update=TRAJECTORY_UPDATE,
    )
    truncation = KrotovOptions().truncation

    trace: list[dict[str, float | int]] = []
    trajectory_rows: list[np.ndarray] = []
    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for iteration in range(NOISY_OPTIMIZATION_STEPS + 1):
            sys.stdout.write(progress_message(iteration, "starting", 0.0, start) + "\n")
            sys.stdout.flush()
            if TRAJECTORY_UPDATE == "cross":
                (
                    contribution,
                    mean_fidelity,
                    trajectory_fidelities,
                    raw_fidelities,
                    jump_counts,
                ) = parallel_noisy_cross_contribution(
                    circuit,
                    theta,
                    target_state,
                    noise_model,
                    tjm_options,
                    truncation,
                    iteration,
                    executor,
                )
            else:
                (
                    contribution,
                    mean_fidelity,
                    trajectory_fidelities,
                    raw_fidelities,
                    jump_counts,
                ) = parallel_noisy_contribution(
                    circuit,
                    theta,
                    target_state,
                    noise_model,
                    tjm_options,
                    truncation,
                    iteration,
                    executor,
                )
            gradient_norm = float(np.linalg.norm(contribution))
            raw_mean_fidelity = float(np.mean(raw_fidelities))
            trace.append({
                "iteration": iteration,
                "mean_fidelity": mean_fidelity,
                "raw_mean_fidelity": raw_mean_fidelity,
                "loss": 1.0 - mean_fidelity,
                "step_size": 0.0 if iteration == NOISY_OPTIMIZATION_STEPS else NOISY_STEP_SIZE,
                "gradient_norm": gradient_norm,
                "mean_jump_count": float(np.mean(jump_counts)),
            })
            trajectory_rows.append(trajectory_fidelities)
            sys.stdout.write(
                f"{progress_message(iteration, 'complete', 1.0, start)} "
                f"mean_fidelity={mean_fidelity:.6f} "
                f"raw_mean={raw_mean_fidelity:.6f} grad_norm={gradient_norm:.6f} "
                f"mean_jumps={float(np.mean(jump_counts)):.2f}\n"
            )
            sys.stdout.flush()
            if iteration < NOISY_OPTIMIZATION_STEPS:
                theta -= NOISY_STEP_SIZE * contribution

    trajectory_fidelities = np.vstack(trajectory_rows)
    output_suffix = f"_gamma{format_gamma_for_path(GAMMA)}_{TRAJECTORY_UPDATE}"
    csv_path = output_dir / f"krotov_gaussian_smpd_noisy_state_preparation{output_suffix}_trace.csv"
    plot_path = output_dir / f"krotov_gaussian_smpd_noisy_state_preparation{output_suffix}.png"
    save_trace_csv(trace, trajectory_fidelities, csv_path)
    save_plot(trace, trajectory_fidelities, plot_path)
    np.save(output_dir / f"krotov_gaussian_smpd_noisy{output_suffix}_final_theta.npy", theta)

    best_mean = max(float(row["mean_fidelity"]) for row in trace)
    final_mean = float(trace[-1]["mean_fidelity"])
    sys.stdout.write(
        "\n".join((
            f"num_qubits={NUM_QUBITS}, num_layers={NUM_LAYERS}, num_params={circuit.num_params}",
            f"gamma={GAMMA}, trajectories={NUM_TRAJECTORIES}, workers={MAX_WORKERS}, update={TRAJECTORY_UPDATE}",
            f"logical_noisy_gates={len(noise_gate_indices)}",
            f"best_mean_fidelity={best_mean:.12f}",
            f"final_mean_fidelity={final_mean:.12f}",
            f"elapsed_seconds={time.perf_counter() - start:.3f}",
            f"trace_csv={csv_path}",
            f"plot_png={plot_path}",
            "",
        ))
    )


if __name__ == "__main__":
    main()
