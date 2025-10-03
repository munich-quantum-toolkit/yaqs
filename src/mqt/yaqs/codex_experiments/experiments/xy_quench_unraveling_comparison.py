r"""XY-quench unraveling comparison between Qiskit MPS and YAQS.

This script implements the Codex prompt from the user request. It performs a
noisy XY Trotter evolution on a 16-qubit ring and compares the Qiskit digital
MPS simulator against several YAQS unraveling strategies. The experiment keeps
adding trajectories until the maximum standard error of the mean across all
local :math:`\langle Z_i \rangle` observables and time steps drops below the
requested threshold.

Generated artefacts (CSV files, plots, metadata) are stored next to this file
under ``results/`` and ``fig/``.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RXXGate, RYYGate
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
from qiskit_aer.noise.errors import PauliLindbladError

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import Z


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExperimentConfig:
    """Collect all tunable experiment parameters."""

    num_qubits: int = 16
    steps: int = 25
    tau: float = 0.1
    p_noise: float = 0.002
    seed0: int = 1337
    sem_threshold: float = 0.005
    batch_size: int = 8
    max_traj_cap: int = 2000
    max_bond_dim: int = 256


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


class RunningStats:
    """Online Welford accumulator for trajectory statistics."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)

    def update(self, samples: Iterable[np.ndarray]) -> None:
        """Update statistics with one or more sample arrays."""

        for sample in samples:
            arr = np.asarray(sample, dtype=np.float64)
            if arr.shape != self.shape:
                raise ValueError(f"Sample shape {arr.shape} does not match accumulator {self.shape}.")
            self.count += 1
            delta = arr - self.mean
            self.mean += delta / self.count
            delta2 = arr - self.mean
            self.m2 += delta * delta2

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return mean, standard deviation, and SEM arrays."""

        if self.count < 2:
            std = np.zeros_like(self.mean)
        else:
            std = np.sqrt(self.m2 / (self.count - 1))
        if self.count == 0:
            sem = np.full_like(self.mean, np.inf, dtype=np.float64)
        else:
            sem = std / math.sqrt(self.count)
        return self.mean.copy(), std, sem


@contextmanager
def patched_default_rng(seed: int):
    """Temporarily replace ``np.random.default_rng`` with a deterministic generator."""

    original = np.random.default_rng
    generator = np.random.Generator(np.random.PCG64(seed))

    def factory(arg: Optional[int] = None) -> np.random.Generator:
        if arg is None:
            return generator
        return original(arg)

    np.random.default_rng = factory  # type: ignore[assignment]
    try:
        yield
    finally:
        np.random.default_rng = original  # type: ignore[assignment]


def precompute_z_weights(num_qubits: int) -> np.ndarray:
    """Pre-compute ±1 weights for local Z expectation values."""

    dim = 1 << num_qubits
    indices = np.arange(dim, dtype=np.int64)
    weights = np.empty((num_qubits, dim), dtype=np.float64)
    for q in range(num_qubits):
        shift = num_qubits - q - 1
        bits = (indices >> shift) & 1
        weights[q] = 1.0 - 2.0 * bits.astype(np.float64)
    return weights


def compute_local_z_expectations(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    r"""Compute :math:`\langle Z_i \rangle` given a statevector and cached weights."""

    probs = np.abs(data) ** 2
    return weights @ probs


def initial_statevector(bitstring: str) -> Statevector:
    """Return a computational basis statevector for the provided bitstring."""

    index = int(bitstring, 2)
    num_qubits = len(bitstring)
    dim = 1 << num_qubits
    amps = np.zeros(dim, dtype=complex)
    amps[index] = 1.0
    return Statevector(amps)


def build_qubit_pairs(num_qubits: int) -> list[tuple[int, int]]:
    """Return nearest-neighbour pairs with periodic boundary conditions."""

    return [(i, (i + 1) % num_qubits) for i in range(num_qubits)]


def build_xy_trotter_circuit(num_qubits: int, steps: int, tau: float) -> QuantumCircuit:
    """Create the XY first-order Trotter circuit with sampling barriers."""

    circuit = QuantumCircuit(num_qubits)
    circuit.barrier(label="SAMPLE_OBSERVABLES")
    pairs = build_qubit_pairs(num_qubits)
    angle = 2.0 * tau
    for _ in range(steps):
        for i, j in pairs:
            circuit.append(RYYGate(angle), [i, j])
        for i, j in pairs:
            circuit.append(RXXGate(angle), [i, j])
        circuit.barrier(label="SAMPLE_OBSERVABLES")
    return circuit


def gaussian_s_weight(sigma: float, M: int = 11, k: float = 4.0) -> float:
    r"""Helper to compute :math:`E[\sin^2(\theta)]` for Gaussian discretisation."""

    if sigma <= 0.0:
        return 0.0
    theta_max = k * sigma
    thetas_pos = np.linspace(0.0, theta_max, (M + 1) // 2)
    thetas = np.concatenate([-thetas_pos[:0:-1], thetas_pos])
    weights = np.exp(-0.5 * (thetas / sigma) ** 2)
    weights /= weights.sum()
    weights = 0.5 * (weights + weights[::-1])
    return float(np.sum(weights * (np.sin(thetas) ** 2)))


def find_gaussian_sigma(target: float, M: int = 11, k: float = 4.0) -> float:
    """Binary search ``sigma`` so that ``E[sin^2(theta)]`` matches ``target``."""

    if target <= 0.0:
        return 1e-6
    low, high = 1e-6, 2.0
    for _ in range(64):
        mid = 0.5 * (low + high)
        val = gaussian_s_weight(mid, M=M, k=k)
        if val > target:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


# ---------------------------------------------------------------------------
# Noise model construction
# ---------------------------------------------------------------------------


def build_noise_models(num_qubits: int, noise_strength: float) -> tuple[
    NoiseModel,
    NoiseModel,
    NoiseModel,
    NoiseModel,
    QiskitNoiseModel,
]:
    """Create YAQS noise models and the matching Qiskit model."""

    processes = [
        {"name": "pauli_x", "sites": [i], "strength": noise_strength}
        for i in range(num_qubits)
    ]
    processes += [
        {"name": "crosstalk_xx", "sites": [i, i + 1], "strength": noise_strength}
        for i in range(num_qubits - 1)
    ]

    # Base model (default unraveling)
    noise_model_default = NoiseModel(processes)

    # Projector unraveling
    proj_processes = [{**p, "unraveling": "projector"} for p in processes]
    noise_model_projector = NoiseModel(proj_processes)

    # Analog 2-point ±theta
    theta0 = float(np.arcsin(np.sqrt(noise_strength)))
    two_pt_processes = [
        {**p, "unraveling": "unitary_2pt", "theta0": theta0}
        for p in processes
    ]
    noise_model_analog_2pt = NoiseModel(two_pt_processes)

    # Analog Gaussian (match same strength)
    sigma = find_gaussian_sigma(noise_strength)
    gaussian_processes = [
        {**p, "unraveling": "unitary_gauss", "sigma": sigma, "theta_max": 4.0 * sigma, "M": 11}
        for p in processes
    ]
    noise_model_analog_gauss = NoiseModel(gaussian_processes)

    # Qiskit noise model (Pauli-Lindblad approximation)
    qiskit_noise_model = QiskitNoiseModel()
    two_qubit_error = PauliLindbladError(
        [Pauli("IX"), Pauli("XI"), Pauli("XX")],
        [noise_strength, noise_strength, noise_strength],
    )
    for qubit in range(num_qubits - 1):
        qiskit_noise_model.add_quantum_error(
            two_qubit_error,
            ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"],
            [qubit, qubit + 1],
        )

    return (
        noise_model_default,
        noise_model_projector,
        noise_model_analog_2pt,
        noise_model_analog_gauss,
        qiskit_noise_model,
    )


# ---------------------------------------------------------------------------
# Trajectory evaluators
# ---------------------------------------------------------------------------


class QiskitTrajectoryRunner:
    """Generate noisy trajectories using manual Pauli sampling."""

    def __init__(
        self,
        config: ExperimentConfig,
        pairs: list[tuple[int, int]],
        weights: np.ndarray,
        initial: Statevector,
    ) -> None:
        self.config = config
        self.pairs = pairs
        self.weights = weights
        self.initial = np.asarray(initial.data, dtype=np.complex64)
        self.angle = 2.0 * config.tau
        self.simulator = AerSimulator(method="matrix_product_state")
        labels = [a + b for a in "IXYZ" for b in "IXYZ"]
        self.identity_label = "II"
        self.non_identity = [label for label in labels if label != self.identity_label]

    def run(self, traj_seed: int) -> np.ndarray:
        cfg = self.config
        rng = np.random.default_rng(traj_seed)
        circuit = QuantumCircuit(cfg.num_qubits)
        circuit.barrier(label="SAMPLE_OBSERVABLES")
        circuit.save_statevector(label="m0")

        for step in range(cfg.steps):
            for pair in self.pairs:
                circuit.ryy(self.angle, *pair)
                label = self._sample_noise(rng)
                if label != self.identity_label:
                    circuit.pauli(label, pair)
            for pair in self.pairs:
                circuit.rxx(self.angle, *pair)
                label = self._sample_noise(rng)
                if label != self.identity_label:
                    circuit.pauli(label, pair)
            circuit.barrier(label="SAMPLE_OBSERVABLES")
            circuit.save_statevector(label=f"m{step + 1}")

        job = self.simulator.run(
            circuit,
            shots=0,
            seed_simulator=traj_seed,
            initial_statevector=self.initial,
        )
        result = job.result()
        data = result.data(0)
        results = np.empty((cfg.steps + 1, cfg.num_qubits), dtype=np.float64)
        for m in range(cfg.steps + 1):
            state = np.asarray(data[f"m{m}"], dtype=np.complex64)
            results[m] = compute_local_z_expectations(state, self.weights)
        return results.T

    def _sample_noise(self, rng: np.random.Generator) -> str:
        if rng.random() < self.config.p_noise:
            return self.non_identity[rng.integers(len(self.non_identity))]
        return self.identity_label


class YAQSTrajectoryRunner:
    """Execute single YAQS trajectories for different unraveling schemes."""

    def __init__(
        self,
        config: ExperimentConfig,
        circuit: QuantumCircuit,
        noise_model: NoiseModel,
        bitstring: str,
    ) -> None:
        self.config = config
        self.circuit = circuit
        self.noise_model = noise_model
        self.bitstring = bitstring

    def run(self, traj_seed: int) -> np.ndarray:
        cfg = self.config
        observables = [Observable(Z(), i) for i in range(cfg.num_qubits)]
        sim_params = StrongSimParams(
            observables=observables,
            num_traj=1,
            max_bond_dim=cfg.max_bond_dim,
            sample_layers=True,
        )
        state = MPS(
            cfg.num_qubits,
            state="basis",
            basis_string=self.bitstring,
            pad=2,
        )
        with patched_default_rng(traj_seed):
            simulator.run(state, self.circuit, sim_params, self.noise_model, parallel=False)

        trajectories = []
        keep = cfg.steps + 1  # drop final duplicate column
        for observable in sim_params.observables:
            assert observable.trajectories is not None
            traj = np.real(observable.trajectories[0])[: keep]
            trajectories.append(traj)
        return np.stack(trajectories)


# ---------------------------------------------------------------------------
# Result persistence and plotting helpers
# ---------------------------------------------------------------------------


def ensure_directories(base: Path) -> tuple[Path, Path]:
    results_dir = base / "results"
    fig_dir = base / "fig"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, fig_dir


def save_localz_csv(
    path: Path,
    method: str,
    mean: np.ndarray,
    std: np.ndarray,
    sem: np.ndarray,
    n_traj: int,
    runtime: float,
    cfg: ExperimentConfig,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "method",
            "qubit",
            "step",
            "m",
            "tau",
            "N",
            "mean",
            "std",
            "sem",
            "n_traj",
            "runtime_sec",
        ])
        for q in range(cfg.num_qubits):
            for m in range(cfg.steps + 1):
                writer.writerow(
                    [
                        method,
                        q + 1,
                        m,
                        m,
                        cfg.tau,
                        cfg.num_qubits,
                        mean[q, m],
                        std[q, m],
                        sem[q, m],
                        n_traj,
                        runtime,
                    ]
                )


def save_summary_csv(path: Path, records: list[dict], cfg: ExperimentConfig) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "n_traj",
                "runtime_sec",
                "speedup_vs_qiskit",
                "p_noise",
                "trotter_tau",
                "steps",
                "N",
                "seed0",
            ]
        )
        for entry in records:
            writer.writerow(
                [
                    entry["method"],
                    entry["n_traj"],
                    entry["runtime_sec"],
                    entry["speedup"],
                    cfg.p_noise,
                    cfg.tau,
                    cfg.steps,
                    cfg.num_qubits,
                    cfg.seed0,
                ]
            )


def save_run_meta(path: Path, cfg: ExperimentConfig, records: list[dict]) -> None:
    meta = {
        "config": asdict(cfg),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methods": records,
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def plot_heatmaps(fig_dir: Path, methods: dict[str, np.ndarray], cfg: ExperimentConfig) -> None:
    import matplotlib.pyplot as plt

    m_axis = np.arange(cfg.steps + 1)
    q_axis = np.arange(1, cfg.num_qubits + 1)
    for method, matrix in methods.items():
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            extent=[m_axis[0], m_axis[-1], q_axis[0], q_axis[-1]],
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )
        ax.set_xlabel("Trotter step m")
        ax.set_ylabel("Qubit index")
        ax.set_title(f"Local ⟨Z⟩ heatmap – {method}")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("⟨Z⟩")
        ax.grid(False)
        fig.savefig(fig_dir / f"heatmap_localZ_{method}.png", dpi=100)
        plt.close(fig)


def plot_selected_lines(fig_dir: Path, stats: dict[str, tuple[np.ndarray, np.ndarray]], cfg: ExperimentConfig) -> None:
    import matplotlib.pyplot as plt

    qubits = [0, 1, 2, 3]
    methods = list(stats.keys())
    m_axis = np.arange(cfg.steps + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, q in zip(axes, qubits):
        for method in methods:
            mean, sem = stats[method]
            ax.plot(m_axis, mean[q], label=method)
            ax.fill_between(m_axis, mean[q] - sem[q], mean[q] + sem[q], alpha=0.2)
        ax.set_title(f"Qubit {q + 1}")
        ax.set_xlabel("Trotter step m")
        ax.set_ylabel("⟨Z⟩")
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods))
    fig.savefig(fig_dir / "localZ_lines_selected.png", dpi=120)
    plt.close(fig)


def plot_runtime_bars(fig_dir: Path, summary: list[dict]) -> None:
    import matplotlib.pyplot as plt

    methods = [entry["method"] for entry in summary]
    runtimes = [entry["runtime_sec"] for entry in summary]
    labels = [f"{entry['n_traj']} trajs\n×{entry['speedup']:.2f}" for entry in summary]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = ax.bar(methods, runtimes, color="#4477aa")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime comparison and speedups")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, label, ha="center", va="bottom")
    fig.savefig(fig_dir / "runtime_speedup_bars.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------


def run_method(
    runner,
    cfg: ExperimentConfig,
) -> tuple[RunningStats, float, int, np.ndarray, np.ndarray, np.ndarray]:
    """Generic loop that keeps sampling until SEM threshold is met."""

    stats = RunningStats((cfg.num_qubits, cfg.steps + 1))
    start = time.perf_counter()
    total = 0
    while total < cfg.max_traj_cap:
        batch = min(cfg.batch_size, cfg.max_traj_cap - total)
        samples = []
        for offset in range(batch):
            traj_seed = cfg.seed0 + total + offset
            samples.append(runner.run(traj_seed))
        stats.update(samples)
        total += batch
        _, _, sem = stats.finalize()
        if np.max(sem) < cfg.sem_threshold and total > 0:
            break
    runtime = time.perf_counter() - start
    mean, std, sem = stats.finalize()
    return stats, runtime, total, mean, std, sem


def console_summary(records: list[dict]) -> None:
    header = f"{'Method':<24}{'Traj':>8}{'Runtime (s)':>16}{'Speedup':>12}{'Max SEM':>12}"
    print("\n=== XY quench unraveling comparison ===")
    print(header)
    print("-" * len(header))
    for entry in records:
        print(
            f"{entry['method']:<24}{entry['n_traj']:>8d}{entry['runtime_sec']:>16.3f}{entry['speedup']:>12.2f}{entry['max_sem']:>12.4f}"
        )


def main() -> None:
    cfg = ExperimentConfig()
    base_dir = Path(__file__).resolve().parent
    results_dir, fig_dir = ensure_directories(base_dir)

    # Global seeding for deterministic behaviour
    random.seed(cfg.seed0)
    np.random.seed(cfg.seed0)
    # Qiskit 2.x removed ``algorithm_globals``; seeding NumPy suffices for deterministic
    # behaviour of the Aer simulators used here.

    bitstring = "0001000100010001"
    initial = initial_statevector(bitstring)
    weights = precompute_z_weights(cfg.num_qubits)
    pairs = build_qubit_pairs(cfg.num_qubits)
    circuit = build_xy_trotter_circuit(cfg.num_qubits, cfg.steps, cfg.tau)

    (
        nm_default,
        nm_projector,
        nm_2pt,
        nm_gauss,
        qiskit_noise_model,
    ) = build_noise_models(cfg.num_qubits, cfg.p_noise)

    # Keep the Qiskit noise model around for provenance, even though
    # manual sampling is used per the specification.
    _ = qiskit_noise_model

    qiskit_runner = QiskitTrajectoryRunner(cfg, pairs, weights, initial)
    yaqs_default = YAQSTrajectoryRunner(cfg, circuit, nm_default, bitstring)
    yaqs_gaussian = YAQSTrajectoryRunner(cfg, circuit, nm_gauss, bitstring)
    yaqs_2pt = YAQSTrajectoryRunner(cfg, circuit, nm_2pt, bitstring)
    yaqs_projector = YAQSTrajectoryRunner(cfg, circuit, nm_projector, bitstring)

    method_specs = [
        ("qiskit_digital_mps", qiskit_runner),
        ("yaqs_default", yaqs_default),
        ("analog_gaussian", yaqs_gaussian),
        ("analog_2pt", yaqs_2pt),
        ("projector_unraveling", yaqs_projector),
    ]

    summary_records: list[dict] = []
    mean_data: dict[str, np.ndarray] = {}
    lines_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    baseline_runtime = None

    for method, runner in method_specs:
        stats, runtime, n_traj, mean, std, sem = run_method(runner, cfg)
        if baseline_runtime is None:
            baseline_runtime = runtime
        speedup = baseline_runtime / runtime if runtime > 0 else float("inf")
        record = {
            "method": method,
            "n_traj": n_traj,
            "runtime_sec": runtime,
            "speedup": speedup,
            "max_sem": float(np.max(sem)),
        }
        summary_records.append(record)
        mean_data[method] = mean
        lines_data[method] = (mean, sem)

        csv_path = results_dir / f"localZ_{method}.csv"
        save_localz_csv(csv_path, method, mean, std, sem, n_traj, runtime, cfg)

    save_summary_csv(results_dir / "summary.csv", summary_records, cfg)
    save_run_meta(results_dir / "run_meta.json", cfg, summary_records)

    plot_heatmaps(fig_dir, mean_data, cfg)
    plot_selected_lines(fig_dir, lines_data, cfg)
    plot_runtime_bars(fig_dir, summary_records)

    console_summary(summary_records)


if __name__ == "__main__":
    main()

