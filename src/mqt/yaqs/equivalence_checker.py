# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compare circuits with MPO or dense matrix backends.

:class:`EquivalenceChecker` uses dense matrices up to
:data:`DEFAULT_MATRIX_MAX_QUBITS` qubits and MPOs above that limit by default.
Checks can include sampled noise on the second circuit.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Literal, TypedDict, overload

import numpy as np
from qiskit.circuit.library import XGate, YGate, ZGate
from qiskit.converters import circuit_to_dag

from .core._validation import validate_bool, validate_choice, validate_finite_real, validate_integer
from .core.data_structures.mpo import MPO
from .core.data_structures.noise_model import NoiseModel, is_pauli, validate_noise_model_for_run
from .core.parallel_utils import WORKER_CTX, available_cpus, reassemble_indexed, run_backend_parallel
from .core.random_utils import make_disorder_rng, make_trajectory_rng
from .digital.utils.contraction_utils import iterate
from .digital.utils.dag_utils import convert_matrix_layout, is_digital_noise_opportunity
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
_MP_CONTEXTS: tuple[MPContext, ...] = ("fork", "spawn", "auto")
_PAULI_GATES = {"x": XGate(), "y": YGate(), "z": ZGate()}
_PAULI_MATRICES = {label: np.asarray(gate.to_matrix()) for label, gate in _PAULI_GATES.items()}
_PAULI_PRODUCTS = {
    (left_label, right_label): np.kron(left_matrix, right_matrix)
    for left_label, left_matrix in _PAULI_MATRICES.items()
    for right_label, right_matrix in _PAULI_MATRICES.items()
}
_PAULI_ERROR = "The noise model contains a process that is not supported for circuit sampling."


class _CheckResultBase(TypedDict):
    """Fields shared by single-pair and noisy-ensemble results."""

    equivalent: bool
    fidelity: float
    elapsed_time: float
    representation: str
    matrix: NDArray[np.complex128] | None
    mpo: MPO | None
    schmidt_values: NDArray[np.float64] | None
    center_cut_entanglement_entropy: float | None
    global_entanglement_entropy: float | None


class EquivalenceCheckResult(_CheckResultBase):
    """Result for one circuit pair, with relative operator ``O = U1 U2†``.

    ``fidelity`` is ``|Tr(O)| / d``, where ``d`` is the Hilbert-space dimension.
    A value of one means equivalence up to global phase, within numerical error.
    ``equivalent`` compares this overlap with the configured threshold.

    Attributes:
        equivalent: Whether ``fidelity`` meets the checker's threshold.
        fidelity: Phase-insensitive overlap; each result type defines its estimator.
        elapsed_time: Check duration in seconds.
        representation: Selected backend, ``"matrix"`` or ``"mpo"``.
        matrix: Relative operator for a noiseless matrix check, otherwise ``None``.
            Site zero is the least-significant subsystem, matching Qiskit.
        mpo: Relative operator for a noiseless MPO check, otherwise ``None``.
            Dense conversion uses the same site order as ``matrix``.
        schmidt_values: Center-cut operator Schmidt spectrum for MPO checks, otherwise ``None``.
        center_cut_entanglement_entropy: Center-cut operator entropy for MPO checks, otherwise ``None``.
        global_entanglement_entropy: Sum of operator entropies across all internal MPO cuts, otherwise ``None``.
    """


class _OptionalEnsembleFields(TypedDict, total=False):
    """Noisy result fields included only when requested."""

    trajectories: list[EquivalenceCheckResult]


class EquivalenceEnsembleResult(_CheckResultBase, _OptionalEnsembleFields):
    """Monte Carlo result from a check with noise on the second circuit.

    ``fidelity`` is the root-mean-square trajectory overlap, on the noiseless
    scale. ``equivalent`` compares this estimate with the configured threshold;
    a finite sample does not give an exact equivalence certificate.
    MPO diagnostics are trajectory averages with zero-padded Schmidt spectra.
    ``matrix`` and ``mpo`` are ``None``. See :class:`EquivalenceCheckResult`
    for shared fields.

    Attributes:
        fidelity_error: Delta-method Monte Carlo standard error; zero if all
            sampled overlaps are zero, or ``None`` for a single trajectory.
        num_traj: Number of sampled circuit realizations.
        trajectories: Individual results, included only with ``return_trajectories=True``.
    """

    fidelity_error: float | None
    num_traj: int


_NoiseOutcome = tuple[float, tuple[str, ...]]
_SupportNoisePlan = tuple[tuple[int, ...], tuple[_NoiseOutcome, ...]]
_CircuitNoisePlan = tuple[tuple[_SupportNoisePlan, ...], ...]


def _validate_representation(representation: str) -> Representation:
    """Validate and normalize the representation selector.

    Args:
        representation: Requested backend name.

    Returns:
        A validated ``Representation`` literal.

    """
    allowed: tuple[Representation, ...] = ("auto", "matrix", "mpo")
    return validate_choice(representation, name="representation", allowed=allowed)


def _validate_matrix_max_qubits(matrix_max_qubits: int) -> int:
    """Validate the matrix auto-backend qubit cutover.

    Args:
        matrix_max_qubits: Maximum qubit count for ``representation="auto"`` to select matrix.

    Returns:
        The validated non-negative cutover value.

    """
    return validate_integer(matrix_max_qubits, name="matrix_max_qubits", minimum=0)


def _validate_max_workers(max_workers: int | None) -> int | None:
    """Validate the worker-count cap.

    Args:
        max_workers: Requested cap, or ``None`` for the default.

    Returns:
        The validated cap, or ``None``.

    """
    if max_workers is None:
        return None
    return validate_integer(max_workers, name="max_workers", minimum=1)


def _validate_threshold(threshold: float) -> float:
    """Validate the non-negative SVD truncation threshold.

    Args:
        threshold: Candidate truncation threshold.

    Returns:
        The finite threshold as a Python :class:`float`.

    Raises:
        ValueError: If ``threshold`` is negative.
    """
    normalized = validate_finite_real(threshold, name="threshold")
    if normalized < 0:
        msg = f"threshold must be non-negative, got {normalized}."
        raise ValueError(msg)
    return normalized


def _validate_fidelity(fidelity: float) -> float:
    """Validate the finite root-overlap threshold.

    Args:
        fidelity: Candidate root-overlap threshold.

    Returns:
        The finite threshold as a Python :class:`float`.

    Raises:
        ValueError: If ``fidelity`` is non-finite or outside ``[0, 1]``.
    """
    try:
        normalized = validate_finite_real(fidelity, name="fidelity")
    except ValueError:
        msg = f"fidelity must be finite and between 0 and 1 inclusive, got {fidelity}."
        raise ValueError(msg) from None
    if not 0 <= normalized <= 1:
        msg = f"fidelity must be finite and between 0 and 1 inclusive, got {normalized}."
        raise ValueError(msg)
    return normalized


def _pauli_labels(process: dict[str, Any]) -> tuple[str, ...] | None:
    """Decode one label per site from a normalized Pauli process.

    Args:
        process: Normalized noise process.

    Returns:
        Pauli labels, or ``None`` when the process is unsupported.
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


def _build_circuit_noise_plan(
    circuit: QuantumCircuit,
    noise_model: NoiseModel,
) -> _CircuitNoisePlan:
    """Precompute gate-local noise data shared by every trajectory.

    Args:
        circuit: Circuit whose instructions define noise opportunities.
        noise_model: Concrete noise model to validate and plan.

    Returns:
        Eligible support groups for each instruction. An empty tuple means that
        the instruction has no applicable noise.

    Raises:
        ValueError: If the model cannot be sampled or same-support probabilities
            sum to more than one.
    """
    if noise_model.scheduled_jumps:
        msg = "Scheduled jumps are not supported for circuit-sampled equivalence checks."
        raise ValueError(msg)
    validate_noise_model_for_run(noise_model, length=circuit.num_qubits, physical_dimensions=2)

    grouped_processes: dict[tuple[int, ...], list[tuple[tuple[str, ...], float]]] = {}
    for process in noise_model.processes:
        labels = _pauli_labels(process)
        if labels is None:
            raise ValueError(_PAULI_ERROR)
        probability = float(process["strength"])
        if not probability:
            continue
        sites = tuple(int(site) for site in process["sites"])
        grouped_processes.setdefault(sites, []).append((labels, probability))

    support_plans: list[_SupportNoisePlan] = []
    for sites, processes in grouped_processes.items():
        total_probability = math.fsum(probability for _, probability in processes)
        if total_probability > 1.0:
            msg = (
                "Circuit-sampled equivalence checking requires process strengths "
                f"sharing support {sites} to sum to at most 1, got {total_probability}."
            )
            raise ValueError(msg)
        outcomes: list[_NoiseOutcome] = []
        cumulative_probability = 0.0
        for labels, probability in processes:
            cumulative_probability += probability
            outcomes.append((cumulative_probability, labels))
        support_plans.append((sites, tuple(outcomes)))

    instruction_plans: list[tuple[_SupportNoisePlan, ...]] = []
    for instruction in circuit.data:
        if not is_digital_noise_opportunity(instruction.operation):
            instruction_plans.append(())
            continue
        gate_sites = {circuit.find_bit(qubit).index for qubit in instruction.qubits}
        instruction_plans.append(
            tuple((sites, outcomes) for sites, outcomes in support_plans if all(site in gate_sites for site in sites))
        )
    return tuple(instruction_plans)


def _sample_noisy_circuit(
    circuit: QuantumCircuit,
    noise_plan: _CircuitNoisePlan,
    rng: np.random.Generator,
) -> QuantumCircuit:
    """Sample one Pauli-noise realization of ``circuit``.

    Every supported two-qubit gate is a noise opportunity. Processes with the
    same support are mutually exclusive probability branches, while distinct
    supports are sampled independently.

    Args:
        circuit: Circuit to sample.
        noise_plan: Per-instruction noise data precomputed for ``circuit``.
        rng: Random-number generator for categorical support draws.

    Returns:
        A copy of ``circuit`` with sampled Pauli gates inserted.
    """
    sampled_circuit = circuit.copy_empty_like()
    for instruction, instruction_plan in zip(circuit.data, noise_plan, strict=True):
        sites = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        qubits = [sampled_circuit.qubits[site] for site in sites]
        clbits = [sampled_circuit.clbits[circuit.find_bit(clbit).index] for clbit in instruction.clbits]
        sampled_circuit.append(instruction.operation.copy(), qubits, clbits)

        for support_sites, outcomes in instruction_plan:
            draw = float(rng.random())
            for cumulative_probability, labels in outcomes:
                if draw < cumulative_probability:
                    for label, site in zip(labels, support_sites, strict=True):
                        sampled_circuit.append(_PAULI_GATES[label], [sampled_circuit.qubits[site]])
                    break

    return sampled_circuit


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
        matrix = convert_matrix_layout(composed.reshape(hilbert_dim, hilbert_dim))
        return {
            "equivalent": measured_fidelity >= checker.fidelity,
            "fidelity": measured_fidelity,
            "elapsed_time": time.time() - start_time,
            "representation": backend,
            "matrix": matrix,
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
    noise_plan: _CircuitNoisePlan,
    random_seed: int | None,
    checker: EquivalenceChecker,
    backend: Literal["matrix", "mpo"],
) -> EquivalenceCheckResult:
    """Sample one noisy ``circuit2`` realization and run a single relative check.

    Args:
        traj_idx: Trajectory index used to seed the sampler.
        circuit1: Clean reference circuit.
        circuit2: Circuit to sample Pauli noise onto.
        noise_plan: Per-instruction noise data precomputed for ``circuit2``.
        random_seed: Optional run-level seed.
        checker: Checker supplying thresholds and backend settings.
        backend: Resolved ``"matrix"`` or ``"mpo"`` backend.

    Returns:
        One :class:`EquivalenceCheckResult` for the sampled pair. Stored operators
        are dropped after diagnostics are extracted.
    """
    rng = make_trajectory_rng(traj_idx, base_seed=random_seed)
    noisy2 = _sample_noisy_circuit(circuit2, noise_plan, rng)
    result = _check_loaded_pair(checker, circuit1, noisy2, backend, parallel=False)
    result["mpo"] = None
    result["matrix"] = None
    return result


def _ensemble_trajectory_worker(traj_idx: int) -> EquivalenceCheckResult:
    """Sample and check one trajectory in a process-pool worker.

    Returns:
        The trajectory result.
    """
    return _run_noisy_check_trajectory(
        traj_idx,
        WORKER_CTX["circuit1"],
        WORKER_CTX["circuit2"],
        WORKER_CTX["noise_plan"],
        WORKER_CTX["random_seed"],
        WORKER_CTX["checker"],
        WORKER_CTX["backend"],
    )


class EquivalenceChecker:
    """Compare circuits with reusable numerical thresholds and backend settings.

    :meth:`check` accepts noiseless pairs or a noise model for a Monte Carlo comparison.
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
        """Configure numerical thresholds and execution.

        Args:
            threshold: Finite, non-negative SVD truncation threshold for MPO updates.
            fidelity: Minimum root overlap in ``[0, 1]`` for both check modes.
            representation: ``"auto"`` selects matrices up to ``matrix_max_qubits``,
                then MPOs. ``"matrix"`` and ``"mpo"`` select a backend explicitly.
            matrix_max_qubits: Non-negative qubit cutover for automatic selection.
            parallel: Enable threaded noiseless MPO updates and parallel noisy trajectories.
            max_workers: Positive cap on noiseless worker threads or noisy worker processes.
                ``None`` selects the default; noisy pools are also capped by ``num_traj``.
            mp_context: ``"auto"``, ``"fork"``, or ``"spawn"`` for noisy process pools.
        """
        self.threshold = threshold
        self.fidelity = fidelity
        self.representation = representation
        self.matrix_max_qubits = matrix_max_qubits
        self.parallel = parallel
        self.max_workers = max_workers
        self.mp_context = mp_context
        self._validate_settings()

    def _validate_settings(self) -> None:
        """Validate mutable checker settings before backend selection."""
        threshold = _validate_threshold(self.threshold)
        fidelity = _validate_fidelity(self.fidelity)
        representation = _validate_representation(self.representation)
        matrix_max_qubits = _validate_matrix_max_qubits(self.matrix_max_qubits)
        parallel = validate_bool(self.parallel, name="parallel")
        max_workers = _validate_max_workers(self.max_workers)
        mp_context = validate_choice(self.mp_context, name="mp_context", allowed=_MP_CONTEXTS)

        self.threshold = threshold
        self.fidelity = fidelity
        self.representation = representation
        self.matrix_max_qubits = matrix_max_qubits
        self.parallel = parallel
        self.max_workers = max_workers
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
        noise_plan: _CircuitNoisePlan,
        num_traj: int,
        random_seed: int | None,
        backend: Literal["matrix", "mpo"],
        *,
        return_trajectories: bool,
    ) -> EquivalenceEnsembleResult:
        """Sample noisy ``circuit2`` trajectories and aggregate relative-operator checks.

        Args:
            circuit1: Clean reference circuit.
            circuit2: Circuit sampled stochastically on each trajectory.
            noise_plan: Per-instruction noise data precomputed for ``circuit2``.
            num_traj: Ensemble size.
            random_seed: Optional run-level seed.
            backend: Resolved backend.
            return_trajectories: Whether to include individual trajectory results.

        Returns:
            An aggregated :class:`EquivalenceEnsembleResult`.
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
                "noise_plan": noise_plan,
                "random_seed": random_seed,
                "checker": self,
                "backend": backend,
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
            trajectories = reassemble_indexed(by_idx, num_traj, label="Noisy equivalence-checking ensemble")
        else:
            trajectories = [
                _run_noisy_check_trajectory(
                    traj_idx,
                    circuit1,
                    circuit2,
                    noise_plan,
                    random_seed,
                    self,
                    backend,
                )
                for traj_idx in range(num_traj)
            ]

        # Random-unitary process fidelity is mean(a_r**2). Its square root keeps
        # the public result on the noiseless root-overlap scale.
        process_fidelity_samples = np.square(np.asarray([traj["fidelity"] for traj in trajectories], dtype=np.float64))
        fidelity = math.sqrt(float(np.mean(process_fidelity_samples)))
        if num_traj > 1:
            process_fidelity_error = float(np.std(process_fidelity_samples, ddof=1) / np.sqrt(num_traj))
            fidelity_error = process_fidelity_error / (2 * fidelity) if fidelity else 0.0
        else:
            fidelity_error = None

        schmidt_spectra = [
            values.ravel() for trajectory in trajectories if (values := trajectory["schmidt_values"]) is not None
        ]
        mean_schmidt_values = None
        if schmidt_spectra:
            mean_schmidt_values = np.zeros(max(values.size for values in schmidt_spectra), dtype=np.float64)
            for values in schmidt_spectra:
                mean_schmidt_values[: values.size] += values
            mean_schmidt_values /= len(schmidt_spectra)

        center_entropy = global_entropy = None
        if backend == "mpo":
            center_entropy = float(np.mean([traj["center_cut_entanglement_entropy"] for traj in trajectories]))
            global_entropy = float(np.mean([traj["global_entanglement_entropy"] for traj in trajectories]))
        result: EquivalenceEnsembleResult = {
            "equivalent": fidelity >= self.fidelity,
            "fidelity": fidelity,
            "fidelity_error": fidelity_error,
            "elapsed_time": time.time() - start_time,
            "representation": backend,
            "num_traj": num_traj,
            "matrix": None,
            "mpo": None,
            "schmidt_values": mean_schmidt_values,
            "center_cut_entanglement_entropy": center_entropy,
            "global_entanglement_entropy": global_entropy,
        }
        if return_trajectories:
            result["trajectories"] = trajectories
        return result

    @overload
    def check(
        self,
        circuit1: QuantumCircuit | str | Path,
        circuit2: QuantumCircuit | str | Path,
        *,
        noise_model: None = None,
        num_traj: int = 1,
        random_seed: int | None = None,
        return_trajectories: Literal[False] = False,
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
        return_trajectories: bool = False,
    ) -> EquivalenceEnsembleResult: ...

    def check(
        self,
        circuit1: QuantumCircuit | str | Path,
        circuit2: QuantumCircuit | str | Path,
        *,
        noise_model: NoiseModel | None = None,
        num_traj: int = 1,
        random_seed: int | None = None,
        return_trajectories: bool = False,
    ) -> EquivalenceCheckResult | EquivalenceEnsembleResult:
        """Compare two circuits up to global phase, optionally with sampled noise.

        Checks the relative operator ``U1 U2†``. Noise acts on ``circuit2`` only,
        after eligible two-qubit gates. Resolved strengths are direct branch
        probabilities, not Lindblad rates. Processes with the same support are
        mutually exclusive and their probabilities must sum to at most one.
        Distinct supports are sampled independently.

        Args:
            circuit1: Circuit, QASM file path, or OpenQASM 2/3 source accepted by
                :func:`~mqt.yaqs.digital.utils.qasm_utils.load_circuit`.
            circuit2: Second circuit, with the same qubit count and accepted input types.
            noise_model: Optional :class:`~mqt.yaqs.NoiseModel` with processes supported
                by stochastic circuit sampling. Distribution-valued strengths are
                resolved once per call.
            num_traj: Positive number of noisy circuit realizations; must be one without noise.
            random_seed: Non-negative seed for strength sampling and circuit draws,
                or ``None`` for unseeded sampling.
            return_trajectories: Include individual results; requires a noise model.

        Returns:
            :class:`EquivalenceCheckResult` without noise, or
            :class:`EquivalenceEnsembleResult` with root-mean-square overlap and
            Monte Carlo uncertainty. These types describe the fields and threshold decision.

        Raises:
            ValueError: If qubit counts differ, a circuit has mid-circuit measurements,
                the MPO backend encounters a gate on more than two qubits, or an
                ensemble option or noise probability is invalid.
            TypeError: If an ensemble option or noise model has an invalid type.
        """
        self._validate_settings()
        num_traj = validate_integer(num_traj, name="num_traj", minimum=1)
        if random_seed is not None:
            random_seed = validate_integer(random_seed, name="random_seed", minimum=0)
        return_trajectories = validate_bool(return_trajectories, name="return_trajectories")
        if noise_model is None:
            if num_traj != 1:
                msg = "num_traj must be 1 when noise_model is None."
                raise ValueError(msg)
            if return_trajectories:
                msg = "return_trajectories requires a noise_model."
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
        if backend == "mpo" and any(
            instruction.operation.num_qubits > 2 and instruction.operation.name not in {"barrier", "measure"}
            for circuit in (circuit1, circuit2)
            for instruction in circuit.data
        ):
            msg = (
                "representation='mpo' does not support gates acting on more than two qubits; "
                "use representation='matrix'. The matrix fallback for unknown unitaries "
                "supports at most eight qubits."
            )
            raise ValueError(msg)

        if noise_model is None:
            return _check_loaded_pair(self, circuit1, circuit2, backend)

        noise_model = noise_model.sample(rng=make_disorder_rng(base_seed=random_seed))
        noise_plan = _build_circuit_noise_plan(circuit2, noise_model)
        return self._run_noisy_ensemble(
            circuit1,
            circuit2,
            noise_plan,
            num_traj,
            random_seed,
            backend,
            return_trajectories=return_trajectories,
        )
