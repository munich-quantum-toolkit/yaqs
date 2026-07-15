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
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from qiskit.converters import circuit_to_dag

from .core.data_structures.mpo import MPO
from .digital.utils.contraction_utils import iterate
from .digital.utils.matrix_utils import (
    compose_operator_tensor,
    compute_identity_fidelity,
    strip_final_measurements,
)
from .digital.utils.qasm_utils import load_circuit

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray
    from qiskit.circuit import QuantumCircuit

    from .core.parallel_utils import MPContext

__all__ = ["DEFAULT_MATRIX_MAX_QUBITS", "EquivalenceCheckResult", "EquivalenceChecker", "Representation"]

Representation = Literal["auto", "matrix", "mpo"]
DEFAULT_MATRIX_MAX_QUBITS = 7


class EquivalenceCheckResult(TypedDict):
    """Return type of :meth:`EquivalenceChecker.check`."""

    equivalent: bool
    fidelity: float
    elapsed_time: float
    representation: str
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
    return cast("Representation", representation)


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
    """Validate the MPO parallel worker-thread cap.

    Args:
        max_workers: Requested thread cap, or ``None`` for the default.

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


class EquivalenceChecker:
    """Public entry point for circuit equivalence checking.

    The MPO backend is the primary, scalable method; the matrix backend is intended for
    very small qubits counts. Owns numerical thresholds and backend selection. The two
    circuits to compare are passed per call to :meth:`check`.

    Attributes:
        threshold: Singular-value truncation threshold used during SVD in the MPO update.
        fidelity: Fidelity threshold for deciding whether the composed operator is identity-like.
        representation: Backend selection (``"auto"``, ``"matrix"``, or ``"mpo"``).
        matrix_max_qubits: Qubit count cutover for ``representation="auto"``.
        parallel: Whether to use a thread pool for independent MPO pair updates (default ``True``; MPO backend only).
        max_workers: Maximum worker threads when ``parallel`` is True (MPO backend only).
        mp_context: Reserved for future process-pool use (MPO uses threads today).
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
            fidelity: Minimum fidelity to treat the composed operator as identity (default ``1 - 1e-13``).
            representation: ``"auto"`` picks matrix for ``num_qubits <= matrix_max_qubits``, else MPO;
                ``"matrix"`` or ``"mpo"`` force that backend.
            matrix_max_qubits: Cutover for ``representation="auto"`` (default ``7``).
            parallel: Enable thread-pool parallelism for checkerboard MPO pair updates (default ``True``;
                effective only from ``MIN_QUBITS_FOR_MPO_PARALLEL`` qubits upward).
            max_workers: Cap on worker threads for the MPO backend (default: machine CPU count).
            mp_context: Reserved; MPO parallelism uses in-process threads, not processes.
        """
        self.threshold = threshold
        self.fidelity = fidelity
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

    def check(
        self,
        circuit1: QuantumCircuit | str | Path,
        circuit2: QuantumCircuit | str | Path,
    ) -> EquivalenceCheckResult:
        """Check whether two quantum circuits are equivalent.

        If the circuits differ only up to global phase and numerical error, the composed
        operator ``U2† U1`` approximates the identity.

        Args:
            circuit1: First quantum circuit. Accepts a :class:`~qiskit.circuit.QuantumCircuit`,
                a ``Path`` to an OpenQASM file, or a ``str`` — either a filesystem path or raw
                OpenQASM 2/3 source (when the first substantive line declares ``OPENQASM``).
                Prefer file paths when the program uses ``include`` directives. OpenQASM 3
                requires ``pip install mqt-yaqs[qasm3]``.
            circuit2: Second quantum circuit (must have the same number of qubits).
                Accepts the same types as ``circuit1``.

        Returns:
            :class:`EquivalenceCheckResult` with keys ``equivalent`` (bool),
            ``fidelity`` (float, measured overlap with identity), ``elapsed_time`` (float,
            seconds), ``representation`` (``"matrix"`` or ``"mpo"``), ``matrix`` (dense
            composed operator on the matrix backend), ``mpo`` (composed operator on the MPO
            backend), and on the MPO backend also ``schmidt_values``,
            ``center_cut_entanglement_entropy``, and ``global_entanglement_entropy``.
            Backend-specific keys are ``None`` when the other backend ran.

        Raises:
            ValueError: If the circuits have different numbers of qubits or contain mid-circuit measurements.
        """
        circuit1 = load_circuit(circuit1)
        circuit2 = load_circuit(circuit2)

        if circuit1.num_qubits != circuit2.num_qubits:
            msg = "Circuits must have the same number of qubits."
            raise ValueError(msg)

        backend = self._resolve_representation(circuit1.num_qubits)
        start_time = time.time()

        if backend == "matrix":
            composed = compose_operator_tensor(circuit1, circuit2)
            measured_fidelity = compute_identity_fidelity(composed)
            hilbert_dim = 2**circuit1.num_qubits
            return {
                "equivalent": measured_fidelity >= self.fidelity,
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
            self.threshold,
            parallel=self.parallel,
            max_workers=self.max_workers,
            mp_context=self.mp_context,
        )
        measured_fidelity = mpo.compute_identity_fidelity()
        center_cut = mpo.length // 2
        return {
            "equivalent": measured_fidelity >= self.fidelity,
            "fidelity": measured_fidelity,
            "elapsed_time": time.time() - start_time,
            "representation": backend,
            "matrix": None,
            "mpo": mpo,
            "schmidt_values": mpo.compute_schmidt_spectrum(center_cut),
            "center_cut_entanglement_entropy": mpo.compute_entanglement_entropy(center_cut),
            "global_entanglement_entropy": sum(mpo.compute_entanglement_entropy(cut) for cut in range(1, mpo.length)),
        }
