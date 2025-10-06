from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Iterable


# Support running as a script from the repository root
_HERE = os.path.dirname(__file__)
_PARENT = os.path.dirname(_HERE)
for path in (_HERE, _PARENT):
    if path not in sys.path:
        sys.path.insert(0, path)

from qiskit.quantum_info import Pauli
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
from qiskit_aer.noise.errors import PauliLindbladError

import numpy as np
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
import matplotlib.pyplot as plt

from worker_functions.circuits import noncommuting_layer, x_commuting_brickwork
from worker_functions.mean_error import print_mean_errors_against_exact
from worker_functions.plotting import (
    plot_avg_bond_dims,
    plot_series_against_exact,
    plot_stochastic_variances,
)
from worker_functions.qiskit_simulators import run_qiskit_exact, run_qiskit_mps
from worker_functions.yaqs_simulator import build_noise_models, run_yaqs


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

import numpy as np
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
from worker_functions.circuits import noncommuting_layer, x_commuting_brickwork
from worker_functions.mean_error import print_mean_errors_against_exact
from worker_functions.plotting import (
    plot_avg_bond_dims,
    plot_series_against_exact,
    plot_stochastic_variances,
)
from worker_functions.qiskit_simulators import run_qiskit_exact, run_qiskit_mps
from worker_functions.yaqs_simulator import build_noise_models, run_yaqs

@dataclass(frozen=True)
class BenchmarkConfig:
    """Container describing a single circuit configuration."""

    circuit_name: str
    basis_builder: Callable[[int], "QuantumCircuit"]
    num_qubits: int
    num_layers: int
    noise_strengths: tuple[float, ...]


def _setup_noise_models(num_qubits: int, noise_strength: float) -> tuple[
    QiskitNoiseModel,
    tuple[object, object, object, object],
]:
    processes = [
        {"name": "pauli_x", "sites": [i], "strength": noise_strength}
        for i in range(num_qubits)
    ] + [
        {"name": "crosstalk_xx", "sites": [i, i + 1], "strength": noise_strength}
        for i in range(num_qubits - 1)
    ]
    noise_models = build_noise_models(processes)

    qiskit_noise_model = QiskitNoiseModel()
    two_qubit_xx_error = PauliLindbladError(
        [Pauli("IX"), Pauli("XI"), Pauli("XX")],
        [noise_strength, noise_strength, noise_strength],
    )
    for qubit in range(num_qubits - 1):
        qiskit_noise_model.add_quantum_error(
            two_qubit_xx_error,
            ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"],
            [qubit, qubit + 1],
        )
    return qiskit_noise_model, noise_models


def run_benchmark(
    *,
    circuit_name: str,
    basis_builder: Callable[[int], "QuantumCircuit"],
    num_qubits: int,
    num_layers: int,
    noise_strength: float,
    num_traj: int,
    output_dir: Path,
) -> Path:
    """Run the benchmark for a single circuit and persist the results."""
    print(f"=== Running benchmark for {circuit_name} ===")
    basis_circuit = basis_builder(num_qubits)
    qiskit_noise_model, (
        noise_model_normal,
        noise_model_projector,
        noise_model_unitary_2pt,
        noise_model_unitary_gauss,
    ) = _setup_noise_models(num_qubits, noise_strength)

    print("  → running qiskit exact")
    exact = run_qiskit_exact(
        num_qubits,
        num_layers,
        basis_circuit,
        qiskit_noise_model,
        method="density_matrix",
    )
    print("  → running qiskit mps")
    qiskit_mps_expvals, qiskit_mps_bonds, qiskit_mps_var = run_qiskit_mps(
        num_qubits,
        num_layers,
        basis_circuit,
        qiskit_noise_model,
        num_traj=num_traj,
    )
    print("  → running yaqs (standard)")
    yaqs_kwargs = {"num_traj": num_traj, "parallel": False}

    yaqs_results_normal, yaqs_bonds_normal, yaqs_var_normal = run_yaqs(
        basis_circuit,
        num_qubits,
        num_layers,
        noise_model_normal,
        **yaqs_kwargs,
    )
    print("  → running yaqs (projector)")
    yaqs_results_projector, yaqs_bonds_projector, yaqs_var_projector = run_yaqs(
        basis_circuit,
        num_qubits,
        num_layers,
        noise_model_projector,
        **yaqs_kwargs,
    )
    print("  → running yaqs (unitary 2pt)")
    (
        yaqs_results_unitary_2pt,
        yaqs_bonds_unitary_2pt,
        yaqs_var_unitary_2pt,
    ) = run_yaqs(
        basis_circuit,
        num_qubits,
        num_layers,
        noise_model_unitary_2pt,
        **yaqs_kwargs,
    )
    print("  → running yaqs (unitary gauss)")
    (
        yaqs_results_unitary_gauss,
        yaqs_bonds_unitary_gauss,
        yaqs_var_unitary_gauss,
    ) = run_yaqs(
        basis_circuit,
        num_qubits,
        num_layers,
        noise_model_unitary_gauss,
        **yaqs_kwargs,
    )

    series_by_label: dict[str, np.ndarray] = {
        "standard": yaqs_results_normal,
        "projector": yaqs_results_projector,
        "unitary_2pt": yaqs_results_unitary_2pt,
        "unitary_gauss": yaqs_results_unitary_gauss,
        "qiskit_mps": qiskit_mps_expvals,
    }
    mean_errors = print_mean_errors_against_exact(exact, series_by_label)

    plot_series_against_exact(
        exact,
        series_by_label,
        num_qubits=num_qubits,
        num_layers=num_layers,
    )
    plot_stochastic_variances(
        num_layers=num_layers,
        qiskit_var=qiskit_mps_var,
        yaqs_var_by_label={
            "standard": yaqs_var_normal,
            "projector": yaqs_var_projector,
            "unitary_2pt": yaqs_var_unitary_2pt,
            "unitary_gauss": yaqs_var_unitary_gauss,
        },
    )
    plot_avg_bond_dims(
        num_layers=num_layers,
        qiskit_bonds=qiskit_mps_bonds,
        yaqs_bonds_by_label={
            "standard": yaqs_bonds_normal,
            "projector": yaqs_bonds_projector,
            "unitary_2pt": yaqs_bonds_unitary_2pt,
            "unitary_gauss": yaqs_bonds_unitary_gauss,
        },
    )

    plt.close("all")

    payload = {
        "circuit_name": circuit_name,
        "num_qubits": num_qubits,
        "num_layers": num_layers,
        "noise_strength": noise_strength,
        "num_traj": num_traj,
        "exact": exact,
        "series_by_label": series_by_label,
        "qiskit_mps_bonds": qiskit_mps_bonds,
        "yaqs_bonds": {
            "standard": yaqs_bonds_normal,
            "projector": yaqs_bonds_projector,
            "unitary_2pt": yaqs_bonds_unitary_2pt,
            "unitary_gauss": yaqs_bonds_unitary_gauss,
        },
        "stochastic_variances": {
            "qiskit_mps": qiskit_mps_var,
            "standard": yaqs_var_normal,
            "projector": yaqs_var_projector,
            "unitary_2pt": yaqs_var_unitary_2pt,
            "unitary_gauss": yaqs_var_unitary_gauss,
        },
        "mean_abs_errors": mean_errors,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{circuit_name}_L{num_qubits}_layers{num_layers}_noise{noise_strength:.3f}.pkl"
    )
    with output_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"  → saved results to {output_path}")
    return output_path


def _iter_configs() -> Iterable[BenchmarkConfig]:
    """Generate a diverse suite of benchmark configurations."""

    default_strengths = (0.02, 0.1, 0.2)

    yield BenchmarkConfig(
        circuit_name="noncommuting_layer",
        basis_builder=noncommuting_layer,
        num_qubits=6,
        num_layers=8,
        noise_strengths=(0.02, 0.05, 0.1, 0.2),
    )
    yield BenchmarkConfig(
        circuit_name="x_commuting_brickwork_depth1",
        basis_builder=lambda nq: x_commuting_brickwork(nq, 1, add_barriers=False),
        num_qubits=6,
        num_layers=8,
        noise_strengths=default_strengths,
    )
    yield BenchmarkConfig(
        circuit_name="x_commuting_brickwork_depth2",
        basis_builder=lambda nq: x_commuting_brickwork(nq, 2, add_barriers=False),
        num_qubits=6,
        num_layers=8,
        noise_strengths=default_strengths,
    )
    yield BenchmarkConfig(
        circuit_name="ising_trotter",
        basis_builder=lambda nq: create_ising_circuit(nq, 1.0, 0.5, 0.1, 1, periodic=False),
        num_qubits=6,
        num_layers=8,
        noise_strengths=default_strengths + (0.05,),
    )
    yield BenchmarkConfig(
        circuit_name="ising_trotter_deep",
        basis_builder=lambda nq: create_ising_circuit(nq, 1.2, 0.5, 0.05, 2, periodic=False),
        num_qubits=6,
        num_layers=12,
        noise_strengths=(0.05, 0.1, 0.2),
    )
    yield BenchmarkConfig(
        circuit_name="noncommuting_layer_8q",
        basis_builder=noncommuting_layer,
        num_qubits=8,
        num_layers=8,
        noise_strengths=(0.05, 0.1),
    )


def main() -> None:
    num_traj = 40
    saved_files: list[Path] = []
    for config in _iter_configs():
        for noise_strength in config.noise_strengths:
            try:
                saved_files.append(
                    run_benchmark(
                        circuit_name=config.circuit_name,
                        basis_builder=config.basis_builder,
                        num_qubits=config.num_qubits,
                        num_layers=config.num_layers,
                        noise_strength=noise_strength,
                        num_traj=num_traj,
                        output_dir=RESULTS_DIR,
                    )
                )
            except Exception as exc:  # pragma: no cover - diagnostic output
                print(
                    "Benchmark for %s (noise=%s) failed: %s"
                    % (config.circuit_name, noise_strength, exc)
                )

    print("=== Saved benchmark files ===")
    for path in saved_files:
        print(f"  • {path}")


if __name__ == "__main__":
    main()
