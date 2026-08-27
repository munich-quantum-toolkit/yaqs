# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Circuit equivalence checker with MPO and dense matrix backends.

This module provides :class:`EquivalenceChecker` for comparing two quantum circuits.
The scalable MPO algorithm is the primary backend; a dense tensorized matrix backend is
available for very small circuits. With ``representation="auto"``, circuits with at most
:data:`DEFAULT_MATRIX_MAX_QUBITS` qubits use the matrix backend and larger circuits use MPO.
Pass ``representation="mpo"`` explicitly for production workloads.

When a :class:`~mqt.yaqs.core.data_structures.noise_model.NoiseModel` is passed to
:meth:`EquivalenceChecker.check`, the checker samples ``num_traj`` explicit Pauli
realizations of the second circuit and runs the standard relative-operator check on
each trajectory. The resulting random-unitary channel is summarized by the Monte Carlo
mean and standard error of the squared trajectory overlaps. This is a stochastic
Pauli-channel comparison, not a general noisy-channel equivalence test.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Literal, TypedDict, overload

import numpy as np
from qiskit.circuit.library import XGate, YGate, ZGate
from qiskit.converters import circuit_to_dag

from .core.data_structures.mpo import MPO
from .core.data_structures.noise_model import NoiseModel, is_pauli, validate_noise_model_for_run
from .core.parallel_utils import WORKER_CTX, available_cpus, reassemble_indexed, run_backend_parallel
from .core.random_utils import make_disorder_rng, make_trajectory_rng
from .digital.utils.contraction_utils import iterate
from .digital.utils.dag_utils import is_digital_noise_opportunity
from .digital.utils.matrix_utils import (
    compose_operator_tensor,
    compute_identity_fidelity,
    strip_final_measurements,
)
from .digital.utils.qasm_utils import load_circuit

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from qiskit.circuit import QuantumCircuit

    from .core.parallel_utils import MPContext

__all__ = [
    "DEFAULT_MATRIX_MAX_QUBITS",
    "EquivalenceCheckResult",
    "EquivalenceChecker",
    "EquivalenceEnsembleResult",
    "Representation",
]

Representation = Literal["auto", "matrix", "mpo"]
DEFAULT_MATRIX_MAX_QUBITS = 7
_PAULI_GATES = {"x": XGate(), "y": YGate(), "z": ZGate()}
_PAULI_MATRICES = {label: np.asarray(gate.to_matrix()) for label, gate in _PAULI_GATES.items()}
_PAULI_PRODUCTS = {
    (left_label, right_label): np.kron(left_matrix, right_matrix)
    for left_label, left_matrix in _PAULI_MATRICES.items()
    for right_label, right_matrix in _PAULI_MATRICES.items()
}
_PAULI_ERROR = "Stochastic Pauli-channel comparison supports recognized YAQS Pauli processes only."


class EquivalenceCheckResult(TypedDict):
    """Return type of :meth:`EquivalenceChecker.check` for a single pair."""

    equivalent: bool
    fidelity: float
    elapsed_time: float
    representation: str
    matrix: NDArray[np.complex128] | None
    mpo: MPO | None
    schmidt_values: NDArray[np.float64] | None
    center_cut_entanglement_entropy: float | None
    global_entanglement_entropy: float | None


class EquivalenceEnsembleResult(TypedDict):
    """Monte Carlo result from a stochastic Pauli-channel comparison.

    ``fidelity`` is the mean of the squared trajectory overlaps, and
    ``fidelity_error`` is its empirical standard error (or ``None`` for one
    trajectory). ``equivalent`` compares that sampled process-fidelity mean with
    the squared configured threshold. It is a finite-sample decision, not an exact
    equivalence certificate.
    """

    equivalent: bool
    fidelity: float
    fidelity_error: float | None
    elapsed_time: float
    representation: str
    num_traj: int
    trajectories: list[EquivalenceCheckResult]
    matrix: NDArray[np.complex128] | None
    mpo: MPO | None
    schmidt_values: NDArray[np.float64] | None
    center_cut_entanglement_entropy: float | None
    global_entanglement_entropy: float | None


def _validate_representation(representation: str) -> Representation:
    """Validate and normalize the representation selector.

    Args:
        representation: Requested backend name.

    Returns:
        A validated ``Representation`` literal.

    Raises:
        ValueError: If ``representation`` is not one of ``auto``, ``matrix``, or ``mpo``.
    """
    allowed = ("auto", "matrix", "mpo")
    if representation not in allowed:
        msg = f"representation must be one of {allowed!r}, got {representation!r}."
        raise ValueError(msg)
    return representation


def _validate_matrix_max_qubits(matrix_max_qubits: int) -> int:
    """Validate the matrix auto-backend qubit cutover.

    Args:
        matrix_max_qubits: Maximum qubit count for ``representation="auto"`` to select matrix.

    Returns:
        The validated non-negative cutover value.

    Raises:
        TypeError: If ``matrix_max_qubits`` is not an ``int``.
        ValueError: If ``matrix_max_qubits`` is negative.
    """
    if isinstance(matrix_max_qubits, bool) or not isinstance(matrix_max_qubits, int):
        msg = f"matrix_max_qubits must be int, got {type(matrix_max_qubits).__name__}."
        raise TypeError(msg)
    if matrix_max_qubits < 0:
        msg = f"matrix_max_qubits must be non-negative, got {matrix_max_qubits}."
        raise ValueError(msg)
    return matrix_max_qubits


def _validate_max_workers(max_workers: int | None) -> int | None:
    """Validate the worker-count cap.

    Args:
        max_workers: Requested cap, or ``None`` for the default.

    Returns:
        The validated cap, or ``None``.

    Raises:
        TypeError: If ``max_workers`` is not ``None`` or a non-boolean ``int``.
        ValueError: If ``max_workers`` is not positive.
    """
    if max_workers is None:
        return None
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        msg = f"max_workers must be int or None, got {type(max_workers).__name__}."
        raise TypeError(msg)
    if max_workers <= 0:
        msg = f"max_workers must be positive, got {max_workers}."
        raise ValueError(msg)
    return max_workers


def _validate_fidelity(fidelity: float) -> float:
    """Validate the normalized root-overlap threshold.

    Args:
        fidelity: Requested threshold.

    Returns:
        The threshold as a built-in ``float``.

    Raises:
        TypeError: If ``fidelity`` is not a non-boolean real number.
        ValueError: If ``fidelity`` is non-finite or outside the interval ``[0, 1]``.
    """
    if isinstance(fidelity, bool) or not isinstance(fidelity, (int, float, np.floating, np.integer)):
        msg = f"fidelity must be a real number, got {type(fidelity).__name__}."
        raise TypeError(msg)
    value = float(fidelity)
    if not math.isfinite(value) or not 0 <= value <= 1:
        msg = f"fidelity must be finite and between 0 and 1 inclusive, got {value}."
        raise ValueError(msg)
    return value


def _validate_num_traj(num_traj: int) -> int:
    """Validate the trajectory count.

    Args:
        num_traj: Requested ensemble size.

    Returns:
        The validated trajectory count.

    Raises:
        TypeError: If ``num_traj`` is not an ``int``.
        ValueError: If ``num_traj`` is less than one.
    """
    if isinstance(num_traj, bool) or not isinstance(num_traj, int):
        msg = f"num_traj must be int, got {type(num_traj).__name__}."
        raise TypeError(msg)
    if num_traj < 1:
        msg = f"num_traj must be at least 1, got {num_traj}."
        raise ValueError(msg)
    return num_traj


def _validate_random_seed(random_seed: int | None) -> int | None:
    """Validate an optional run-level RNG seed.

    Args:
        random_seed: Requested seed, or ``None`` for a non-deterministic stream.

    Returns:
        The validated seed, or ``None``.

    Raises:
        TypeError: If ``random_seed`` is not ``None`` or a non-boolean ``int``.
        ValueError: If ``random_seed`` is negative.
    """
    if random_seed is None:
        return None
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        msg = f"random_seed must be int or None, got {type(random_seed).__name__}."
        raise TypeError(msg)
    if random_seed < 0:
        msg = f"random_seed must be non-negative, got {random_seed}."
        raise ValueError(msg)
    return random_seed


def _has_unsupported_mpo_gates(circuit: QuantumCircuit) -> bool:
    """Return whether ``circuit`` contains a gate the MPO backend cannot apply.

    Args:
        circuit: Circuit to inspect.

    Returns:
        ``True`` if any non-barrier, non-measure instruction acts on more than two qubits.
    """
    return any(
        instruction.operation.num_qubits > 2 and instruction.operation.name not in {"barrier", "measure"}
        for instruction in circuit.data
    )


def _pauli_labels(process: dict[str, Any]) -> tuple[str, ...] | None:
    """Decode labels from a normalized YAQS Pauli process.

    Args:
        process: Process normalized by :class:`NoiseModel`.

    Returns:
        One Pauli label per process site, or ``None`` if unsupported.
    """
    if not is_pauli(process):
        return None

    if len(process["sites"]) == 1:
        matrix = np.asarray(process["matrix"])
        label = max(_PAULI_MATRICES, key=lambda candidate: abs(np.vdot(_PAULI_MATRICES[candidate], matrix)))
        return (label,)

    if "factors" in process:
        return tuple(
            max(_PAULI_MATRICES, key=lambda candidate: abs(np.vdot(_PAULI_MATRICES[candidate], factor)))
            for factor in process["factors"]
        )

    matrix = np.asarray(process["matrix"])
    return max(_PAULI_PRODUCTS.items(), key=lambda item: abs(np.vdot(item[1], matrix)))[0]


def _validate_pauli_noise_model(noise_model: NoiseModel, num_qubits: int) -> None:
    """Validate that a noise model can be materialized as Pauli gates.

    Args:
        noise_model: Sampled noise model to validate.
        num_qubits: Width of the circuit receiving sampled noise.

    Raises:
        ValueError: If scheduled jumps, invalid sites or dimensions, or non-Pauli
            processes are present.
    """
    if noise_model.scheduled_jumps:
        msg = "Stochastic Pauli-channel comparison does not support scheduled jumps."
        raise ValueError(msg)
    validate_noise_model_for_run(noise_model, length=num_qubits, physical_dimensions=2)
    if any(_pauli_labels(process) is None for process in noise_model.processes):
        raise ValueError(_PAULI_ERROR)


def _sample_noisy_circuit(
    circuit: QuantumCircuit,
    noise_model: NoiseModel,
    rng: np.random.Generator,
) -> QuantumCircuit:
    """Sample one Pauli-noise realization of ``circuit``.

    Every gate acting on two or more qubits is a noise opportunity. A process
    participates when its complete support is contained in the gate support.
    At most one process is appended after each such gate.

    Args:
        circuit: Circuit to sample.
        noise_model: Concrete Pauli noise model (distribution-valued strengths
            must already be resolved).
        rng: Random-number generator for event and process draws.

    Returns:
        A copy of ``circuit`` with sampled Pauli gates inserted.
    """
    sampled_circuit = circuit.copy_empty_like()
    for instruction in circuit.data:
        sites = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        qubits = [sampled_circuit.qubits[site] for site in sites]
        clbits = [sampled_circuit.clbits[circuit.find_bit(clbit).index] for clbit in instruction.clbits]
        sampled_circuit.append(instruction.operation.copy(), qubits, clbits)

        if not is_digital_noise_opportunity(instruction.operation):
            continue

        gate_sites = set(sites)
        processes = [process for process in noise_model.processes if set(process["sites"]).issubset(gate_sites)]
        rates = [float(process["strength"]) for process in processes]
        total_rate = sum(rates)
        if not total_rate or rng.random() >= -math.expm1(-total_rate):
            continue

        threshold = float(rng.random()) * total_rate
        cumulative = 0.0
        selected = next(process for process, rate in zip(reversed(processes), reversed(rates), strict=True) if rate > 0)
        for process, rate in zip(processes, rates, strict=True):
            cumulative += rate
            if threshold < cumulative:
                selected = process
                break

        labels = _pauli_labels(selected)
        assert labels is not None
        for label, site in zip(labels, selected["sites"], strict=True):
            sampled_circuit.append(_PAULI_GATES[label], [sampled_circuit.qubits[int(site)]])

    return sampled_circuit


def _mean_or_none(values: list[float | None]) -> float | None:
    """Average a list of optional floats, or return ``None`` if any entry is missing.

    Args:
        values: Per-trajectory scalars.

    Returns:
        The arithmetic mean, or ``None``.
    """
    if any(value is None for value in values):
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _standard_error_or_none(values: NDArray[np.float64]) -> float | None:
    """Estimate the standard error of a sample mean when at least two samples exist.

    Args:
        values: One-dimensional sample values.

    Returns:
        The sample standard deviation divided by the square root of the sample size,
        or ``None`` when the sample contains only one value.
    """
    if values.size < 2:
        return None
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _check_loaded_pair(
    checker: EquivalenceChecker,
    circuit1: QuantumCircuit,
    circuit2: QuantumCircuit,
    backend: Literal["matrix", "mpo"],
    *,
    parallel: bool | None = None,
) -> EquivalenceCheckResult:
    """Run one relative-operator check on already loaded circuits.

    Args:
        checker: Checker supplying thresholds and backend settings.
        circuit1: First (reference) circuit.
        circuit2: Second circuit.
        backend: Resolved ``"matrix"`` or ``"mpo"`` backend.
        parallel: Override for MPO zone-thread parallelism. ``None`` uses
            :attr:`EquivalenceChecker.parallel`.

    Returns:
        A single :class:`EquivalenceCheckResult`.
    """
    use_parallel = checker.parallel if parallel is None else parallel
    start_time = time.time()

    if backend == "matrix":
        composed = compose_operator_tensor(circuit1, circuit2)
        measured_fidelity = compute_identity_fidelity(composed)
        hilbert_dim = 2**circuit1.num_qubits
        return {
            "equivalent": measured_fidelity >= checker.fidelity,
            "fidelity": measured_fidelity,
            "elapsed_time": time.time() - start_time,
            "representation": backend,
            "matrix": composed.reshape(hilbert_dim, hilbert_dim),
            "mpo": None,
            "schmidt_values": None,
            "center_cut_entanglement_entropy": None,
            "global_entanglement_entropy": None,
        }

    circuit1 = strip_final_measurements(circuit1)
    circuit2 = strip_final_measurements(circuit2)
    mpo = MPO.identity(circuit1.num_qubits)
    circuit1_dag = circuit_to_dag(circuit1)
    circuit2_dag = circuit_to_dag(circuit2)
    iterate(
        mpo,
        circuit1_dag,
        circuit2_dag,
        checker.threshold,
        parallel=use_parallel,
        max_workers=checker.max_workers,
        mp_context=checker.mp_context,
    )
    measured_fidelity = mpo.compute_identity_fidelity()
    center_cut = mpo.length // 2
    return {
        "equivalent": measured_fidelity >= checker.fidelity,
        "fidelity": measured_fidelity,
        "elapsed_time": time.time() - start_time,
        "representation": backend,
        "matrix": None,
        "mpo": mpo,
        "schmidt_values": mpo.compute_schmidt_spectrum(center_cut),
        "center_cut_entanglement_entropy": mpo.compute_entanglement_entropy(center_cut),
        "global_entanglement_entropy": sum(mpo.compute_entanglement_entropy(cut) for cut in range(1, mpo.length)),
    }


def _run_noisy_check_trajectory(
    traj_idx: int,
    circuit1: QuantumCircuit,
    circuit2: QuantumCircuit,
    noise_model: NoiseModel,
    random_seed: int | None,
    checker: EquivalenceChecker,
    backend: Literal["matrix", "mpo"],
) -> EquivalenceCheckResult:
    """Sample one noisy ``circuit2`` realization and run a single relative check.

    Args:
        traj_idx: Trajectory index used to seed the sampler.
        circuit1: Clean reference circuit.
        circuit2: Circuit to sample Pauli noise onto.
        noise_model: Concrete Pauli noise model.
        random_seed: Optional run-level seed.
        checker: Checker supplying thresholds and backend settings.
        backend: Resolved ``"matrix"`` or ``"mpo"`` backend.

    Returns:
        One :class:`EquivalenceCheckResult` for the sampled pair. Stored operators
        are dropped after diagnostics are extracted.
    """
    rng = make_trajectory_rng(traj_idx, base_seed=random_seed)
    noisy2 = _sample_noisy_circuit(circuit2, noise_model, rng)
    result = _check_loaded_pair(checker, circuit1, noisy2, backend, parallel=False)
    result["mpo"] = None
    result["matrix"] = None
    return result


def _ensemble_trajectory_worker(traj_idx: int) -> EquivalenceCheckResult:
    """Process-pool worker: sample one noisy circuit and check it serially.

    Returns:
        One :class:`EquivalenceCheckResult` for trajectory ``traj_idx``.
    """
    checker = EquivalenceChecker(
        threshold=WORKER_CTX["threshold"],
        fidelity=WORKER_CTX["fidelity"],
        representation=WORKER_CTX["backend"],
        matrix_max_qubits=WORKER_CTX["matrix_max_qubits"],
        parallel=False,
        max_workers=1,
        mp_context=WORKER_CTX["mp_context"],
    )
    return _run_noisy_check_trajectory(
        traj_idx,
        WORKER_CTX["circuit1"],
        WORKER_CTX["circuit2"],
        WORKER_CTX["noise_model"],
        WORKER_CTX["random_seed"],
        checker,
        WORKER_CTX["backend"],
    )


class EquivalenceChecker:
    """Public entry point for circuit equivalence checking.

    The MPO backend is the primary, scalable method; the matrix backend is intended for
    very small qubits counts. Owns numerical thresholds and backend selection. The two
    circuits to compare are passed per call to :meth:`check`. Supplying a supported
    :class:`NoiseModel` performs a stochastic Pauli-channel comparison, not a general
    noisy-channel equivalence test.

    Attributes:
        threshold: Singular-value truncation threshold used during SVD in the MPO update.
        fidelity: Root-overlap threshold for a noiseless check. Stochastic Pauli
            comparisons compare their process-fidelity estimate with the square of
            this threshold.
        representation: Backend selection (``"auto"``, ``"matrix"``, or ``"mpo"``).
        matrix_max_qubits: Qubit count cutover for ``representation="auto"``.
        parallel: Whether to parallelize noiseless MPO pair updates and stochastic Pauli
            trajectory ensembles (default ``True``).
        max_workers: Maximum worker threads for noiseless MPO checks, and the process-pool
            cap for stochastic Pauli trajectory ensembles. Process pools are also capped
            by ``num_traj``.
        mp_context: Multiprocessing start method when a stochastic Pauli ensemble process
            pool is used.
    """

    def __init__(
        self,
        *,
        threshold: float = 1e-13,
        fidelity: float = 1 - 1e-13,
        representation: Representation = "auto",
        matrix_max_qubits: int = DEFAULT_MATRIX_MAX_QUBITS,
        parallel: bool = True,
        max_workers: int | None = None,
        mp_context: MPContext = "auto",
    ) -> None:
        """Initialize the checker with numerical thresholds and backend options.

        Args:
            threshold: SVD truncation threshold in the MPO update (default ``1e-13``).
            fidelity: Minimum root overlap for a noiseless identity check (default
                ``1 - 1e-13``). Stochastic Pauli comparisons square it to obtain the
                corresponding process-fidelity threshold.
            representation: ``"auto"`` picks matrix for ``num_qubits <= matrix_max_qubits``, else MPO;
                ``"matrix"`` or ``"mpo"`` force that backend.
            matrix_max_qubits: Cutover for ``representation="auto"`` (default ``7``).
            parallel: Enable parallel checkerboard MPO pair updates on noiseless checks
                (effective only from 12 qubits upward) and process-pool execution of
                stochastic Pauli trajectory ensembles (default ``True``).
            max_workers: Cap on worker threads for noiseless MPO checks, and on processes for
                stochastic Pauli trajectory ensembles. Process pools use at most
                ``num_traj`` workers.
            mp_context: Start method when a stochastic Pauli ensemble process pool is used.
        """
        self.threshold = threshold
        self.fidelity = _validate_fidelity(fidelity)
        self.representation = _validate_representation(representation)
        self.matrix_max_qubits = _validate_matrix_max_qubits(matrix_max_qubits)
        self.parallel = parallel
        self.max_workers = _validate_max_workers(max_workers)
        self.mp_context = mp_context

    def _resolve_representation(self, num_qubits: int) -> Literal["matrix", "mpo"]:
        """Choose the concrete backend for a given circuit width.

        Args:
            num_qubits: Number of qubits in the circuits being compared.

        Returns:
            ``"matrix"`` or ``"mpo"`` according to ``representation`` and ``matrix_max_qubits``.
        """
        if self.representation == "matrix":
            return "matrix"
        if self.representation == "mpo":
            return "mpo"
        return "matrix" if num_qubits <= self.matrix_max_qubits else "mpo"

    def _run_noisy_ensemble(
        self,
        circuit1: QuantumCircuit,
        circuit2: QuantumCircuit,
        noise_model: NoiseModel,
        num_traj: int,
        random_seed: int | None,
        backend: Literal["matrix", "mpo"],
    ) -> EquivalenceEnsembleResult:
        """Sample noisy ``circuit2`` trajectories and aggregate relative-operator checks.

        Args:
            circuit1: Clean reference circuit.
            circuit2: Circuit sampled stochastically on each trajectory.
            noise_model: Concrete Pauli noise model.
            num_traj: Ensemble size.
            random_seed: Optional run-level seed.
            backend: Resolved backend.

        Returns:
            A Monte Carlo :class:`EquivalenceEnsembleResult`. The process-fidelity
            standard error is ``None`` when ``num_traj`` is one.
        """
        start_time = time.time()
        workers = 1
        if self.parallel and num_traj > 1:
            resolved_workers = self.max_workers if self.max_workers is not None else max(1, available_cpus() - 1)
            workers = min(num_traj, resolved_workers)

        if workers > 1:
            payload = {
                "circuit1": circuit1,
                "circuit2": circuit2,
                "noise_model": noise_model,
                "random_seed": random_seed,
                "threshold": self.threshold,
                "fidelity": self.fidelity,
                "backend": backend,
                "matrix_max_qubits": self.matrix_max_qubits,
                "mp_context": self.mp_context,
            }
            by_idx = dict(
                run_backend_parallel(
                    _ensemble_trajectory_worker,
                    payload=payload,
                    n_jobs=num_traj,
                    max_workers=workers,
                    show_progress=False,
                    desc="Noisy EC trajectories",
                    mp_context=self.mp_context,
                )
            )
            trajectories = reassemble_indexed(by_idx, num_traj, label="Stochastic Pauli-channel ensemble")
        else:
            trajectories = [
                _run_noisy_check_trajectory(
                    traj_idx,
                    circuit1,
                    circuit2,
                    noise_model,
                    random_seed,
                    self,
                    backend,
                )
                for traj_idx in range(num_traj)
            ]

        # A trajectory reports |Tr(Q_r)| / d; random-unitary channel process fidelity
        # is the ensemble mean of its square.
        process_fidelity_samples = np.square(np.asarray([traj["fidelity"] for traj in trajectories], dtype=np.float64))
        mean_fidelity = float(np.mean(process_fidelity_samples))
        process_fidelity_threshold = self.fidelity**2
        return {
            "equivalent": mean_fidelity >= process_fidelity_threshold,
            "fidelity": mean_fidelity,
            "fidelity_error": _standard_error_or_none(process_fidelity_samples),
            "elapsed_time": time.time() - start_time,
            "representation": backend,
            "num_traj": num_traj,
            "trajectories": trajectories,
            "matrix": None,
            "mpo": None,
            "schmidt_values": None,
            "center_cut_entanglement_entropy": _mean_or_none([
                traj["center_cut_entanglement_entropy"] for traj in trajectories
            ]),
            "global_entanglement_entropy": _mean_or_none([
                traj["global_entanglement_entropy"] for traj in trajectories
            ]),
        }

    @overload
    def check(
        self,
        circuit1: QuantumCircuit | str | Path,
        circuit2: QuantumCircuit | str | Path,
        *,
        noise_model: None = None,
        num_traj: int = 1,
        random_seed: int | None = None,
    ) -> EquivalenceCheckResult: ...

    @overload
    def check(
        self,
        circuit1: QuantumCircuit | str | Path,
        circuit2: QuantumCircuit | str | Path,
        *,
        noise_model: NoiseModel,
        num_traj: int = 1,
        random_seed: int | None = None,
    ) -> EquivalenceEnsembleResult: ...

    def check(
        self,
        circuit1: QuantumCircuit | str | Path,
        circuit2: QuantumCircuit | str | Path,
        *,
        noise_model: NoiseModel | None = None,
        num_traj: int = 1,
        random_seed: int | None = None,
    ) -> EquivalenceCheckResult | EquivalenceEnsembleResult:
        """Compare two quantum circuits, optionally under sampled Pauli noise.

        On the noiseless path, circuits that differ only up to global phase and numerical
        error have a composed operator ``U2† U1`` that approximates the identity and are
        reported as equivalent.

        When ``noise_model`` is set, Pauli noise is sampled onto ``circuit2`` only.
        Each of ``num_traj`` independent realizations is checked with the same
        relative operator ``U_noisy† U_ideal`` used for a noiseless pair. The returned
        ensemble reports a Monte Carlo estimate of the random-unitary Pauli channel's
        process fidelity, its empirical standard error when ``num_traj > 1``, and
        trajectory-averaged operator-entanglement diagnostics. This is not a general
        noisy-channel equivalence test, and its threshold comparison is a sample-level
        decision rather than an equivalence certificate.
        With ``parallel=True``, noisy worker count is capped by both ``num_traj`` and
        ``max_workers``. A process pool is used only when that count is greater than one,
        and each worker uses serial ``iterate``. ``parallel=False`` keeps the noisy
        ensemble in-process and serial. MPO zone threads are not used on the noisy path.

        Args:
            circuit1: First quantum circuit. Accepts a :class:`~qiskit.circuit.QuantumCircuit`,
                a ``Path`` to an OpenQASM file, or a ``str`` — either a filesystem path or raw
                OpenQASM 2/3 source (when the first substantive line declares ``OPENQASM``).
                Prefer file paths when the program uses ``include`` directives. OpenQASM 3
                requires ``pip install mqt-yaqs[qasm3]``.
            circuit2: Second quantum circuit (must have the same number of qubits).
                Accepts the same types as ``circuit1``. When ``noise_model`` is set, this is
                the circuit that is sampled stochastically.
            noise_model: Optional YAQS noise model restricted to normalized one-site X/Y/Z
                processes and two-site Pauli products. ``None`` runs a single noiseless
                check. Distribution-valued strengths are resolved once per call.
            num_traj: Number of stochastic circuit realizations when ``noise_model`` is set.
                Must be ``1`` when ``noise_model`` is ``None``.
            random_seed: Optional run-level seed for disorder sampling and per-trajectory
                circuit draws. Must be non-negative; ``None`` uses non-deterministic streams.

        Returns:
            :class:`EquivalenceCheckResult` for a noiseless pair, or
            :class:`EquivalenceEnsembleResult` when ``noise_model`` is set. Both include
            ``equivalent`` and ``fidelity``; for a stochastic Pauli result these are the
            sampled process-fidelity decision and estimate. The ensemble additionally
            includes ``fidelity_error``, ``num_traj``, and the individual ``trajectories``.

        Raises:
            ValueError: If the circuits have different numbers of qubits, contain mid-circuit
                measurements, contain gates on more than two qubits on the MPO backend,
                ``num_traj`` is used without ``noise_model``, ``num_traj`` is less than one,
                ``random_seed`` is negative, or the noise model is not a supported Pauli model.
            TypeError: If ``num_traj``, ``random_seed``, or ``noise_model`` has an invalid type.
        """
        num_traj = _validate_num_traj(num_traj)
        random_seed = _validate_random_seed(random_seed)
        if noise_model is None:
            if num_traj != 1:
                msg = "num_traj must be 1 when noise_model is None."
                raise ValueError(msg)
        elif not isinstance(noise_model, NoiseModel):
            msg = f"noise_model must be NoiseModel or None, got {type(noise_model).__name__}."
            raise TypeError(msg)

        circuit1 = load_circuit(circuit1)
        circuit2 = load_circuit(circuit2)

        if circuit1.num_qubits != circuit2.num_qubits:
            msg = "Circuits must have the same number of qubits."
            raise ValueError(msg)

        backend = self._resolve_representation(circuit1.num_qubits)
        if backend == "mpo" and (_has_unsupported_mpo_gates(circuit1) or _has_unsupported_mpo_gates(circuit2)):
            msg = (
                "representation='mpo' does not support gates acting on more than two qubits; "
                "use representation='matrix'. The matrix fallback for unknown unitaries "
                "supports at most eight qubits."
            )
            raise ValueError(msg)

        if noise_model is None:
            return _check_loaded_pair(self, circuit1, circuit2, backend)

        noise_model = noise_model.sample(rng=make_disorder_rng(base_seed=random_seed))
        _validate_pauli_noise_model(noise_model, circuit2.num_qubits)
        return self._run_noisy_ensemble(
            circuit1,
            circuit2,
            noise_model,
            num_traj,
            random_seed,
            backend,
        )
