# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Check the noisy SWAP-reset-SWAP quantum-memory witness benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from mqt.yaqs.characterization.memory.operational_memory.response_matrix import (
    assemble_response_matrix,
    compute_spectrum,
)

PAULI_LABELS = ("I", "X", "Y", "Z")
COLUMN_LABELS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
COLUMN_ROWS = np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int64)
COLUMN_SIGNS = np.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float64)
P_CRITICAL = 2.0 / 3.0
DEFAULT_NUM_POINTS = 101
CHECK_TOLERANCE = 5e-12
RESPONSE_MATRIX_UPDATE_COMMIT = "c6744d1ff97ad163b53682e71dda858a3454718b"
PAPER_SUBSECTION = r"\subsection{Certifying quantum memory from response data}"

IDENTITY = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
PAULIS = np.stack((IDENTITY, PAULI_X, PAULI_Y, PAULI_Z))
ZERO_STATE = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
SWAP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.complex128,
)


def partial_trace_system(rho_se: np.ndarray) -> np.ndarray:
    """Trace out the first factor in the ``|S,E>`` basis ordering."""
    tensor = np.asarray(rho_se, dtype=np.complex128).reshape(2, 2, 2, 2)
    return np.trace(tensor, axis1=0, axis2=2)


def partial_trace_environment(rho_se: np.ndarray) -> np.ndarray:
    """Trace out the second factor in the ``|S,E>`` basis ordering."""
    tensor = np.asarray(rho_se, dtype=np.complex128).reshape(2, 2, 2, 2)
    return np.trace(tensor, axis1=1, axis2=3)


def input_states() -> np.ndarray:
    """Return the six Pauli eigenstates in ``+X,-X,+Y,-Y,+Z,-Z`` order."""
    axes = (PAULI_X, PAULI_Y, PAULI_Z)
    return np.stack([(IDENTITY + sign * axis) / 2.0 for axis in axes for sign in (1.0, -1.0)])


def simulate_protocol(rho_input: np.ndarray, p: float) -> tuple[np.ndarray, float]:
    """Evolve one input and return its output and retained zero-outcome probability."""
    rho_se = np.kron(rho_input, ZERO_STATE)
    rho_se = SWAP @ rho_se @ SWAP.conj().T

    system_before_break = partial_trace_environment(rho_se)
    zero_outcome_probability = float(np.trace(ZERO_STATE @ system_before_break).real)
    rho_se = np.kron(ZERO_STATE, partial_trace_system(rho_se))

    rho_system = partial_trace_environment(rho_se)
    rho_se = (1.0 - p) * rho_se + p * np.kron(rho_system, IDENTITY / 2.0)
    rho_se = SWAP @ rho_se @ SWAP.conj().T
    rho_output = partial_trace_environment(rho_se)
    return rho_output, zero_outcome_probability


def pauli_responses(output_states: np.ndarray) -> np.ndarray:
    """Encode normalized outputs as ``(n_histories, one_future, IXYZ)`` responses."""
    responses = np.empty((len(output_states), 1, 4), dtype=np.float64)
    for history, rho_output in enumerate(output_states):
        responses[history, 0] = [float(np.trace(pauli @ rho_output).real) for pauli in PAULIS]
    return responses


def assemble_reference_response_matrix(pauli_ixyz: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Independently assemble future/IXYZ rows and history columns without centering."""
    features = np.asarray(pauli_ixyz, dtype=np.float64)
    probabilities = np.asarray(weights, dtype=np.float64)
    if features.shape != (6, 1, 4) or probabilities.shape != (6, 1):
        msg = f"Expected Pauli shape (6, 1, 4) and weight shape (6, 1), got {features.shape} and {probabilities.shape}."
        raise ValueError(msg)
    weighted = features * probabilities[..., np.newaxis]
    return weighted.transpose(1, 2, 0).reshape(4, 6)


def response_entropy(response_matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return natural-log response entropy, singular values, and squared mode weights."""
    singular_values = np.linalg.svd(response_matrix, compute_uv=False).astype(np.float64)
    resolution = np.finfo(np.float64).eps * max(response_matrix.shape) * singular_values[0]
    squared = np.where(singular_values > resolution, singular_values**2, 0.0)
    total = float(np.sum(squared))
    if total <= 0.0:
        return 0.0, singular_values, np.zeros_like(singular_values)
    mode_weights = squared / total
    positive = mode_weights > 0.0
    entropy = abs(float(-np.sum(mode_weights[positive] * np.log(mode_weights[positive]))))
    return entropy, singular_values, mode_weights


def witness_coefficients() -> np.ndarray:
    """Return the fixed linear-response witness coefficient matrix."""
    coefficients = np.zeros((4, 6), dtype=np.float64)
    coefficients[0] = 1.0 / 36.0
    for column, (row, sign) in enumerate(zip(COLUMN_ROWS, COLUMN_SIGNS, strict=True)):
        coefficients[row, column] = -sign / 12.0
    return coefficients


def evaluate_witness(response_matrix: np.ndarray, coefficients: np.ndarray) -> tuple[float, float, float]:
    """Evaluate average fidelity and the witness in direct and coefficient forms."""
    fidelity_sum = 0.0
    for column, (row, sign) in enumerate(zip(COLUMN_ROWS, COLUMN_SIGNS, strict=True)):
        fidelity_sum += response_matrix[0, column] + sign * response_matrix[row, column]
    average_fidelity = fidelity_sum / 12.0
    witness_direct = 2.0 / 3.0 - average_fidelity
    witness_linear = float(np.sum(coefficients * response_matrix))
    return float(average_fidelity), float(witness_direct), witness_linear


def analytic_entropy(p: float | np.ndarray) -> np.ndarray:
    """Evaluate the analytic entropy with safe handling of zero mode weights."""
    p_array = np.asarray(p, dtype=np.float64)
    lambda_squared = (1.0 - p_array) ** 2
    leading = 1.0 / (1.0 + lambda_squared)
    subleading = lambda_squared / (3.0 * (1.0 + lambda_squared))
    entropy = np.zeros_like(p_array)
    positive_leading = leading > 0.0
    entropy[positive_leading] -= leading[positive_leading] * np.log(leading[positive_leading])
    positive_subleading = subleading > 0.0
    entropy[positive_subleading] -= 3.0 * subleading[positive_subleading] * np.log(subleading[positive_subleading])
    return entropy


def p_grid(num_points: int) -> np.ndarray:
    """Build a grid spanning zero to one with ``p=2/3`` inserted exactly."""
    if num_points < 3:
        msg = f"num_points must be at least 3, got {num_points}."
        raise ValueError(msg)
    values = np.linspace(0.0, 1.0, num_points, dtype=np.float64)
    values[int(np.argmin(np.abs(values - P_CRITICAL)))] = P_CRITICAL
    return np.sort(values)


def run_benchmark(p_values: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Run the exact density-matrix benchmark and validate all analytic predictions."""
    inputs = input_states()
    coefficients = witness_coefficients()
    n_points = len(p_values)
    outputs = np.empty((n_points, 6, 2, 2), dtype=np.complex128)
    pauli_ixyz = np.empty((n_points, 6, 1, 4), dtype=np.float64)
    weights = np.empty((n_points, 6, 1), dtype=np.float64)
    matrices = np.empty((n_points, 4, 6), dtype=np.float64)
    reference_matrices = np.empty_like(matrices)
    singular_values = np.empty((n_points, 4), dtype=np.float64)
    singular_weights = np.empty((n_points, 4), dtype=np.float64)
    yaqs_singular_values = np.empty_like(singular_values)
    entropies = np.empty(n_points, dtype=np.float64)
    yaqs_entropies = np.empty_like(entropies)
    fidelities = np.empty(n_points, dtype=np.float64)
    witnesses = np.empty(n_points, dtype=np.float64)
    witnesses_linear = np.empty(n_points, dtype=np.float64)
    retained_probabilities = np.empty((n_points, 6), dtype=np.float64)

    analytic_outputs = np.empty_like(outputs)
    analytic_spectra = np.empty_like(singular_values)
    for p_index, p in enumerate(p_values):
        for history, rho_input in enumerate(inputs):
            outputs[p_index, history], retained_probabilities[p_index, history] = simulate_protocol(rho_input, float(p))
            analytic_outputs[p_index, history] = (1.0 - p) * rho_input + p * IDENTITY / 2.0

        pauli_ixyz[p_index] = pauli_responses(outputs[p_index])
        weights[p_index, :, 0] = retained_probabilities[p_index]
        matrices[p_index] = assemble_response_matrix(pauli_ixyz[p_index], weights[p_index])
        reference_matrices[p_index] = assemble_reference_response_matrix(pauli_ixyz[p_index], weights[p_index])
        entropies[p_index], singular_values[p_index], singular_weights[p_index] = response_entropy(matrices[p_index])
        yaqs_spectrum = compute_spectrum(matrices[p_index], discarded_weight_threshold=None)
        yaqs_singular_values[p_index] = yaqs_spectrum["singular_values_full"]
        yaqs_entropies[p_index] = yaqs_spectrum["entropy"]
        fidelities[p_index], witnesses[p_index], witnesses_linear[p_index] = evaluate_witness(
            matrices[p_index], coefficients
        )
        lambda_value = 1.0 - p
        analytic_spectra[p_index] = np.array([
            np.sqrt(6.0),
            np.sqrt(2.0) * lambda_value,
            np.sqrt(2.0) * lambda_value,
            np.sqrt(2.0) * lambda_value,
        ])

    analytic_entropies = analytic_entropy(p_values)
    analytic_fidelities = 1.0 - p_values / 2.0
    analytic_witnesses = p_values / 2.0 - 1.0 / 3.0
    critical_index = int(np.flatnonzero(np.isclose(p_values, P_CRITICAL, rtol=0.0, atol=1e-15))[0])
    endpoint_index = int(np.flatnonzero(np.isclose(p_values, 1.0, rtol=0.0, atol=1e-15))[0])

    np.testing.assert_allclose(retained_probabilities, 1.0, rtol=0.0, atol=CHECK_TOLERANCE)
    np.testing.assert_allclose(pauli_ixyz[..., 0], 1.0, rtol=0.0, atol=CHECK_TOLERANCE)
    np.testing.assert_allclose(matrices[:, 0, :], 1.0, rtol=0.0, atol=CHECK_TOLERANCE)
    np.testing.assert_allclose(matrices, reference_matrices, rtol=0.0, atol=CHECK_TOLERANCE)
    np.testing.assert_allclose(outputs, analytic_outputs, rtol=0.0, atol=CHECK_TOLERANCE)
    np.testing.assert_allclose(yaqs_singular_values, singular_values, rtol=0.0, atol=CHECK_TOLERANCE)
    np.testing.assert_allclose(yaqs_entropies, entropies, rtol=0.0, atol=CHECK_TOLERANCE)
    np.testing.assert_allclose(singular_values, analytic_spectra, rtol=0.0, atol=CHECK_TOLERANCE)
    np.testing.assert_allclose(entropies, analytic_entropies, rtol=0.0, atol=CHECK_TOLERANCE)
    np.testing.assert_allclose(fidelities, analytic_fidelities, rtol=0.0, atol=CHECK_TOLERANCE)
    np.testing.assert_allclose(witnesses, analytic_witnesses, rtol=0.0, atol=CHECK_TOLERANCE)
    np.testing.assert_allclose(witnesses_linear, witnesses, rtol=0.0, atol=CHECK_TOLERANCE)
    assert abs(entropies[endpoint_index]) <= CHECK_TOLERANCE
    assert entropies[critical_index] > 0.0
    assert abs(witnesses[critical_index]) <= CHECK_TOLERANCE
    assert np.all(witnesses[p_values < P_CRITICAL] < 0.0)
    assert np.all(witnesses[p_values > P_CRITICAL] > 0.0)

    checks = {
        "max_output_state_abs_error": float(np.max(np.abs(outputs - analytic_outputs))),
        "max_response_matrix_assembly_abs_error": float(np.max(np.abs(matrices - reference_matrices))),
        "max_yaqs_spectrum_singular_value_abs_error": float(np.max(np.abs(yaqs_singular_values - singular_values))),
        "max_yaqs_spectrum_entropy_abs_error": float(np.max(np.abs(yaqs_entropies - entropies))),
        "max_singular_value_abs_error": float(np.max(np.abs(singular_values - analytic_spectra))),
        "max_entropy_abs_error": float(np.max(np.abs(entropies - analytic_entropies))),
        "max_witness_analytic_abs_error": float(np.max(np.abs(witnesses - analytic_witnesses))),
        "max_witness_linear_abs_error": float(np.max(np.abs(witnesses_linear - witnesses))),
        "max_retained_probability_abs_error": float(np.max(np.abs(retained_probabilities - 1.0))),
        "S_V_at_p_1": float(entropies[endpoint_index]),
        "S_V_at_p_2_over_3": float(entropies[critical_index]),
        "w_at_p_2_over_3": float(witnesses[critical_index]),
    }
    arrays = {
        "p": p_values,
        "input_states": inputs,
        "output_states": outputs,
        "pauli_ixyz_ij": pauli_ixyz,
        "weights_ij": weights,
        "response_matrices": matrices,
        "reference_response_matrices": reference_matrices,
        "singular_values": singular_values,
        "singular_weights": singular_weights,
        "yaqs_singular_values": yaqs_singular_values,
        "yaqs_S_V": yaqs_entropies,
        "S_V": entropies,
        "F_av": fidelities,
        "w": witnesses,
        "w_linear": witnesses_linear,
        "witness_coefficients": coefficients,
        "analytic_output_states": analytic_outputs,
        "analytic_singular_values": analytic_spectra,
        "analytic_S_V": analytic_entropies,
        "analytic_F_av": analytic_fidelities,
        "analytic_w": analytic_witnesses,
        "retained_probabilities": retained_probabilities,
        "pauli_labels": np.asarray(PAULI_LABELS),
        "column_labels": np.asarray(COLUMN_LABELS),
    }
    return arrays, checks


def write_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write the requested scalar data with full double precision."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("p", "S_V", "w", "F_av"))
        for p, entropy, witness, fidelity in zip(arrays["p"], arrays["S_V"], arrays["w"], arrays["F_av"], strict=True):
            writer.writerow(tuple(f"{float(value):.17g}" for value in (p, entropy, witness, fidelity)))


def load_csv(path: Path) -> dict[str, np.ndarray]:
    """Load scalar data for plot-only regeneration."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    return {name: np.atleast_1d(data[name]) for name in ("p", "S_V", "w", "F_av")}


def plot(arrays: dict[str, np.ndarray], output_stem: Path) -> None:
    """Plot response entropy and the fixed witness on a shared depolarization axis."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    p_values = arrays["p"]
    dense_p = np.linspace(0.0, 1.0, 1001)
    dense_entropy = analytic_entropy(dense_p)
    dense_witness = dense_p / 2.0 - 1.0 / 3.0
    stride = max(1, len(p_values) // 10)
    marker_indices = np.unique(
        np.concatenate((
            np.arange(0, len(p_values), stride, dtype=np.int64),
            np.flatnonzero(np.isclose(p_values, P_CRITICAL, rtol=0.0, atol=1e-15)),
            np.array([len(p_values) - 1], dtype=np.int64),
        ))
    )

    figure, axes = plt.subplots(2, 1, figsize=(4.4, 5.0), sharex=True, constrained_layout=True)
    line_color = "#8c2d04"
    marker_color = "#d94801"
    for axis in axes:
        axis.axvspan(0.0, P_CRITICAL, color="#e1f0e1", linewidth=0.0)
        axis.axvspan(P_CRITICAL, 1.0, color="#fff0cc", linewidth=0.0)
        axis.axvline(P_CRITICAL, color="0.3", linestyle="--", linewidth=0.9)
        axis.grid(True, axis="y", alpha=0.12, linewidth=0.4)

    axes[0].plot(dense_p, dense_entropy, color=line_color, linewidth=1.7, label="analytic")
    axes[0].plot(
        p_values[marker_indices],
        arrays["S_V"][marker_indices],
        linestyle="none",
        marker="o",
        markersize=4.1,
        markerfacecolor="white",
        markeredgecolor=marker_color,
        markeredgewidth=1.0,
        label="simulation",
    )
    axes[0].set_ylabel(r"Response entropy $S_V$")
    axes[0].set_ylim(-0.035, 1.33)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].text(0.025, 0.985, "(a)", transform=axes[0].transAxes, fontweight="bold", va="top")
    axes[0].text(
        P_CRITICAL - 0.015,
        0.94,
        r"$p=2/3$",
        transform=axes[0].get_xaxis_transform(),
        color="0.25",
        ha="right",
        va="top",
    )

    axes[1].plot(dense_p, dense_witness, color=line_color, linewidth=1.7)
    axes[1].plot(
        p_values[marker_indices],
        arrays["w"][marker_indices],
        linestyle="none",
        marker="o",
        markersize=4.1,
        markerfacecolor="white",
        markeredgecolor=marker_color,
        markeredgewidth=1.0,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.85)
    axes[1].set_ylabel(r"Witness $w$")
    axes[1].set_xlabel(r"Depolarization strength $p$")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(-0.37, 0.22)
    axes[1].set_xticks((0.0, 1.0 / 3.0, P_CRITICAL, 1.0), ("0", r"$1/3$", r"$2/3$", "1"))
    axes[1].text(0.025, 0.985, "(b)", transform=axes[1].transAxes, fontweight="bold", va="top")
    axes[1].text(
        P_CRITICAL / 2.0,
        -0.055,
        "Certified quantum\nmemory",
        color="#216e39",
        ha="center",
        va="center",
        fontsize=8,
    )
    axes[1].text(
        0.80,
        0.155,
        "Classically\nreproducible\nmemory",
        color="#8c510a",
        ha="center",
        va="center",
        fontsize=7.5,
    )
    figure.savefig(output_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def sha256(path: Path) -> str:
    """Return an artifact's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_snapshot() -> dict[str, object]:
    """Record the experiment worktree revision without requiring a clean checkout."""
    repository_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        cwd=repository_root,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        check=False,
        capture_output=True,
        cwd=repository_root,
        text=True,
    ).stdout.splitlines()
    return {"commit_before_run": commit or None, "status_before_run": status}


def response_matrix_implementation_snapshot() -> dict[str, object]:
    """Identify the YAQS response-matrix implementation imported for this run."""
    source_path = Path(inspect.getfile(assemble_response_matrix)).resolve()
    root_process = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        cwd=source_path.parent,
        text=True,
    )
    repository_root = Path(root_process.stdout.strip()) if root_process.returncode == 0 else None
    if repository_root is None:
        source_file = str(source_path)
        commit = None
        source_status: list[str] = []
    else:
        source_file = str(source_path.relative_to(repository_root))
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            cwd=repository_root,
            text=True,
        ).stdout.strip()
        source_status = subprocess.run(
            ["git", "status", "--short", "--", source_file],
            check=False,
            capture_output=True,
            cwd=repository_root,
            text=True,
        ).stdout.splitlines()
    return {
        "source_file": source_file,
        "source_sha256": sha256(source_path),
        "git_commit": commit or None,
        "source_status": source_status,
        "matches_recorded_update_commit": commit == RESPONSE_MATRIX_UPDATE_COMMIT,
    }


def refresh_plot_hashes(manifest_path: Path, output_stem: Path) -> None:
    """Refresh derived-figure hashes after a plot-only run."""
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for suffix in (".pdf", ".png"):
        path = output_stem.with_suffix(suffix)
        manifest["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest["plot_script_sha256"] = sha256(Path(__file__))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def print_checks(checks: dict[str, float], num_points: int) -> None:
    """Print the requested numerical comparisons and endpoint checks."""
    print(f"Simulated {num_points} depolarization values with exact two-qubit density matrices.")
    print(f"max output-state discrepancy: {checks['max_output_state_abs_error']:.3e}")
    print(f"max response-matrix assembly discrepancy: {checks['max_response_matrix_assembly_abs_error']:.3e}")
    print(f"max YAQS spectrum entropy discrepancy: {checks['max_yaqs_spectrum_entropy_abs_error']:.3e}")
    print(f"max singular-value discrepancy: {checks['max_singular_value_abs_error']:.3e}")
    print(f"max entropy discrepancy: {checks['max_entropy_abs_error']:.3e}")
    print(f"max witness discrepancy (simulation vs analytic): {checks['max_witness_analytic_abs_error']:.3e}")
    print(f"max witness discrepancy (direct vs linear): {checks['max_witness_linear_abs_error']:.3e}")
    print(f"max retained-probability discrepancy from one: {checks['max_retained_probability_abs_error']:.3e}")
    print(f"S_V(1) = {checks['S_V_at_p_1']:.12g}")
    print(f"S_V(2/3) = {checks['S_V_at_p_2_over_3']:.12g}")
    print(f"w(2/3) = {checks['w_at_p_2_over_3']:.3e}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("updated_experiments/results/quantum_memory_witness"),
    )
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run or replot the benchmark and save its data, figure, and manifest."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "witness_summary.csv"
    raw_path = output_dir / "raw_responses.npz"
    output_stem = output_dir / "quantum_memory_witness"
    manifest_path = output_dir / "run_manifest.json"

    if args.plot_only:
        plot(load_csv(csv_path), output_stem)
        refresh_plot_hashes(manifest_path, output_stem)
        print(f"Replotted quantum-memory witness benchmark in {output_dir}")
        return

    snapshot = git_snapshot()
    implementation = response_matrix_implementation_snapshot()
    started = time.perf_counter()
    arrays, checks = run_benchmark(p_grid(args.num_points))
    write_csv(csv_path, arrays)
    np.savez_compressed(raw_path, **arrays)
    plot(arrays, output_stem)
    artifacts = [csv_path, raw_path, output_stem.with_suffix(".pdf"), output_stem.with_suffix(".png")]
    manifest = {
        "experiment": "noisy SWAP-reset-SWAP quantum-memory benchmark",
        "paper_subsection": PAPER_SUBSECTION,
        "response_matrix_update_commit": RESPONSE_MATRIX_UPDATE_COMMIT,
        "response_matrix_implementation": implementation,
        "experiment_worktree": snapshot,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "configuration": {
            "basis_order": "|S,E>",
            "column_order": list(COLUMN_LABELS),
            "response_rows": list(PAULI_LABELS),
            "num_p_values": int(len(arrays["p"])),
            "p_includes_2_over_3": True,
            "finite_shots": False,
            "retained_history_probability": 1.0,
            "centered": False,
            "response_orientation": "future_response_rows_history_columns",
            "weight_scope": "complete_retained_record",
            "spectrum_logarithm": "natural",
            "zero_mode_resolution": "eps * max(V.shape) * sigma_max",
            "yaqs_spectrum_discarded_weight_threshold": None,
            "check_tolerance": CHECK_TOLERANCE,
        },
        "witness": {
            "definition": "w = 2/3 - F_av",
            "certification_rule": "w < 0 certifies quantum memory",
            "classification_scope": "p >= 2/3 is classically reproducible only for this known benchmark",
        },
        "checks": checks,
        "runtime_seconds": time.perf_counter() - started,
        "script_sha256": sha256(Path(__file__)),
        "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifacts},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print_checks(checks, len(arrays["p"]))
    print(f"Wrote quantum-memory witness benchmark to {output_dir}")


if __name__ == "__main__":
    main()
