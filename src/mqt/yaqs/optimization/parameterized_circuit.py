# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Parameterized circuit representation for gate-local circuit optimization.

This module provides a lightweight, gate-list representation of parameterized quantum
circuits used by the Krotov-inspired discrete adjoint optimizer. Each circuit factor is
either a fixed gate or a one-parameter trainable gate with a fixed Hermitian generator,
``U_k = exp(-i/2 * alpha_k(theta, x) * G_k)``, where the scalar angle map
``alpha_k(theta, x) = angle_scale * theta[param_index] + angle_offset + data_map(x)``
may combine a trainable parameter with a sample-dependent data encoding.

Gate matrices are produced in *ascending-site* convention: for a two-qubit gate acting
on sites ``(s0, s1)``, the returned ``4 x 4`` matrix acts on the merged physical index
``|q_min, q_max>`` with the lower MPS site as the more significant factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit.library import CXGate
from qiskit.synthesis import OneQubitEulerDecomposer, TwoQubitBasisDecomposer

from ..core.libraries.circuit_library import brickwall_matrix_product_disentangler_bonds
from ..core.libraries.gate_library import GateLibrary

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray
    from qiskit.circuit import QuantumCircuit

_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

#: Derivative operators ``D`` with ``dU/d(alpha) = D @ U`` for the supported
#: one-parameter gates. Two-qubit operators are given in gate-site order.
PARAMETRIC_DERIVATIVE_OPERATORS: dict[str, NDArray[np.complex128]] = {
    "rx": -0.5j * _X,
    "ry": -0.5j * _Y,
    "rz": -0.5j * _Z,
    "p": 1j * np.diag([0.0, 1.0]).astype(np.complex128),
    "rxx": -0.5j * np.kron(_X, _X),
    "ryy": -0.5j * np.kron(_Y, _Y),
    "rzz": -0.5j * np.kron(_Z, _Z),
    "cp": 1j * np.diag([0.0, 0.0, 0.0, 1.0]).astype(np.complex128),
}

#: Gate-library names that take a single rotation-angle parameter.
SINGLE_ANGLE_GATES = frozenset(PARAMETRIC_DERIVATIVE_OPERATORS)

_ONE_QUBIT_DECOMPOSER = OneQubitEulerDecomposer("ZYZ")
_TWO_QUBIT_DECOMPOSER = TwoQubitBasisDecomposer(CXGate())


def brickwall_matrix_product_disentangler_num_parameters(
    num_qubits: int,
    depth: int,
    *,
    initial_single_qubit_layer: bool = False,
) -> int:
    """Return the number of scalar Krotov parameters in the BMPD architecture.

    Args:
        num_qubits: Number of qubits.
        depth: Number of BMPD entangling gates per pair.
        initial_single_qubit_layer: Whether to include a trainable U3 product-state
            preparation layer with three scalar parameters per qubit.

    Returns:
        Number of scalar parameters expected by
        :func:`create_brickwall_matrix_product_disentangler_parameterized_circuit`.
    """
    num_bmpd_gates = sum(len(layer) for layer in brickwall_matrix_product_disentangler_bonds(num_qubits, depth))
    initial_params = 3 * num_qubits if initial_single_qubit_layer else 0
    return initial_params + 9 * num_bmpd_gates


def _swap_two_site_convention(mat: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Permute a two-qubit matrix between ``|q_a, q_b>`` and ``|q_b, q_a>`` ordering.

    Args:
        mat: ``4 x 4`` matrix in ``|q_a, q_b>`` ordering.

    Returns:
        The same operator expressed in ``|q_b, q_a>`` ordering.
    """
    tensor = mat.reshape(2, 2, 2, 2)
    return np.asarray(tensor.transpose(1, 0, 3, 2).reshape(4, 4), dtype=np.complex128)


def _validate_disentangler_sites(sites: Sequence[int], num_qubits: int) -> tuple[int, ...]:
    """Validate one- or two-qubit disentangler sites.

    Returns:
        Validated sites as a tuple.

    Raises:
        ValueError: If the sites are invalid for an SMPD gate.
    """
    site_tuple = tuple(int(site) for site in sites)
    if len(site_tuple) not in {1, 2}:
        msg = f"Only one- and two-qubit SMPD gates are supported, got sites={site_tuple!r}."
        raise ValueError(msg)
    if len(set(site_tuple)) != len(site_tuple):
        msg = f"SMPD gate acts on duplicate sites {site_tuple!r}."
        raise ValueError(msg)
    if any(site < 0 or site >= num_qubits for site in site_tuple):
        msg = f"SMPD gate sites {site_tuple!r} outside range(0, {num_qubits})."
        raise ValueError(msg)
    return site_tuple


def _validate_disentangler_unitary(matrix: NDArray[np.complex128], num_sites: int) -> NDArray[np.complex128]:
    """Validate a dense one- or two-qubit SMPD gate matrix.

    Returns:
        Validated dense unitary matrix.

    Raises:
        ValueError: If the matrix shape or unitarity check fails.
    """
    expected_dim = 2**num_sites
    mat = np.asarray(matrix, dtype=np.complex128)
    if mat.shape != (expected_dim, expected_dim):
        msg = f"Expected a {(expected_dim, expected_dim)} SMPD unitary for {num_sites} sites, got {mat.shape}."
        raise ValueError(msg)
    if not np.allclose(mat.conj().T @ mat, np.eye(expected_dim, dtype=np.complex128), atol=1e-10):
        msg = "SMPD gate matrix must be unitary."
        raise ValueError(msg)
    return mat


@dataclass
class ParameterizedGate:
    """One factor of a parameterized circuit.

    A factor is either a fixed gate (``param_index is None`` and ``data_map is None``)
    or a one-parameter gate whose rotation angle is
    ``angle_scale * theta[param_index] + angle_offset + data_map(x)``.

    Attributes:
        name: Gate name as found in :class:`~mqt.yaqs.core.libraries.gate_library.GateLibrary`.
        sites: Site indices the gate acts on, in gate order (e.g. ``(control, target)``).
        param_index: Index into the trainable parameter vector, or ``None`` for fixed gates.
        angle_scale: Multiplicative factor of the trainable parameter in the angle map.
        angle_offset: Constant offset of the angle map.
        data_map: Optional sample-dependent contribution to the angle map.
        fixed_params: Parameter values for fixed gates constructed with library parameters
            (e.g. a frozen ``u`` gate). Ignored for single-angle gates.
    """

    name: str
    sites: tuple[int, ...]
    param_index: int | None = None
    angle_scale: float = 1.0
    angle_offset: float = 0.0
    data_map: Callable[[NDArray[np.float64]], float] | None = None
    fixed_params: tuple[float, ...] = ()

    @property
    def is_trainable(self) -> bool:
        """Whether this gate carries a trainable parameter.

        Returns:
            ``True`` if the gate angle depends on the trainable parameter vector.
        """
        return self.param_index is not None

    @property
    def is_parametric(self) -> bool:
        """Whether this gate is built from a single rotation angle.

        Returns:
            ``True`` if the gate is a supported one-parameter gate.
        """
        return self.name in SINGLE_ANGLE_GATES


class ParameterizedCircuit:
    """Gate-list representation of a parameterized quantum circuit.

    Attributes:
        num_qubits: Number of qubits (MPS sites).
        gates: Ordered list of circuit factors.
        num_params: Length of the trainable parameter vector.
    """

    def __init__(self, num_qubits: int, gates: list[ParameterizedGate], num_params: int | None = None) -> None:
        """Initializes and validates a parameterized circuit.

        Args:
            num_qubits: Number of qubits in the circuit.
            gates: Ordered list of circuit factors applied left to right.
            num_params: Length of the trainable parameter vector. If ``None``, it is
                inferred as ``max(param_index) + 1``.

        Raises:
            ValueError: If a gate acts on invalid sites, a trainable gate is not a
                supported one-parameter gate, or a parameter index is out of range.
        """
        self.num_qubits = num_qubits
        self.gates = list(gates)

        max_index = -1
        for gate in self.gates:
            if len(gate.sites) not in {1, 2}:
                msg = f"Gate '{gate.name}' must act on one or two sites, got {gate.sites!r}."
                raise ValueError(msg)
            if len(set(gate.sites)) != len(gate.sites):
                msg = f"Gate '{gate.name}' acts on duplicate sites {gate.sites!r}."
                raise ValueError(msg)
            if any(site < 0 or site >= num_qubits for site in gate.sites):
                msg = f"Gate '{gate.name}' acts on sites {gate.sites!r} outside range(0, {num_qubits})."
                raise ValueError(msg)
            if not hasattr(GateLibrary, gate.name):
                msg = f"Gate '{gate.name}' not found in GateLibrary."
                raise ValueError(msg)
            if gate.is_trainable and not gate.is_parametric:
                msg = (
                    f"Trainable gate '{gate.name}' is not a supported one-parameter gate. "
                    f"Supported: {sorted(SINGLE_ANGLE_GATES)}."
                )
                raise ValueError(msg)
            if (gate.data_map is not None or not np.isclose(gate.angle_offset, 0.0)) and not gate.is_parametric:
                msg = f"Gate '{gate.name}' has an angle map but is not a one-parameter gate."
                raise ValueError(msg)
            if gate.param_index is not None:
                max_index = max(max_index, gate.param_index)

        inferred = max_index + 1
        if num_params is None:
            num_params = inferred
        elif num_params < inferred:
            msg = f"num_params={num_params} is smaller than required by parameter indices ({inferred})."
            raise ValueError(msg)
        self.num_params = num_params

    @staticmethod
    def angle(gate: ParameterizedGate, theta: NDArray[np.float64], x: NDArray[np.float64] | None) -> float:
        """Evaluate the scalar angle map of a one-parameter gate.

        Args:
            gate: The circuit factor.
            theta: Trainable parameter vector.
            x: Input sample, required if the gate has a data map.

        Returns:
            The rotation angle ``alpha(theta, x)``.

        Raises:
            ValueError: If the gate has a data map but no sample is provided.
        """
        angle = gate.angle_offset
        if gate.param_index is not None:
            angle += gate.angle_scale * float(theta[gate.param_index])
        if gate.data_map is not None:
            if x is None:
                msg = f"Gate '{gate.name}' has a data map but no input sample was provided."
                raise ValueError(msg)
            angle += float(gate.data_map(x))
        return angle

    def gate_matrix(
        self,
        gate: ParameterizedGate,
        theta: NDArray[np.float64],
        x: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.complex128], tuple[int, ...]]:
        """Build the unitary matrix of a circuit factor in ascending-site convention.

        Args:
            gate: The circuit factor.
            theta: Trainable parameter vector.
            x: Input sample for data-dependent angle maps.

        Returns:
            A tuple ``(matrix, sites)`` where ``sites`` is sorted ascending and
            ``matrix`` acts on the merged index ``|q_min, q_max>`` for two-qubit gates.
        """
        attr = getattr(GateLibrary, gate.name)
        if gate.is_parametric:
            library_gate = attr([self.angle(gate, theta, x)])
        elif gate.fixed_params:
            library_gate = attr(list(gate.fixed_params))
        else:
            library_gate = attr()

        matrix = np.asarray(library_gate.matrix, dtype=np.complex128)
        if len(gate.sites) == 2 and gate.sites[0] > gate.sites[1]:
            matrix = _swap_two_site_convention(matrix)
        return matrix, tuple(sorted(gate.sites))

    @staticmethod
    def derivative_operator(gate: ParameterizedGate) -> tuple[NDArray[np.complex128], tuple[int, ...]]:
        """Return the derivative operator of a trainable gate in ascending-site convention.

        The operator ``D`` satisfies ``dU/d(alpha) = D @ U``, so that the gate-local
        Krotov contribution is ``angle_scale * 2 Re <chi | D | psi_after>``.

        Args:
            gate: A trainable circuit factor.

        Returns:
            A tuple ``(operator, sites)`` with ``sites`` sorted ascending.

        Raises:
            ValueError: If the gate is not a supported one-parameter gate.
        """
        if not gate.is_parametric:
            msg = f"Gate '{gate.name}' has no derivative operator."
            raise ValueError(msg)
        operator = PARAMETRIC_DERIVATIVE_OPERATORS[gate.name]
        if len(gate.sites) == 2 and gate.sites[0] > gate.sites[1]:
            operator = _swap_two_site_convention(operator)
        return operator, tuple(sorted(gate.sites))

    @classmethod
    def from_qiskit(
        cls,
        circuit: QuantumCircuit,
        parameters: list[object] | None = None,
    ) -> ParameterizedCircuit:
        """Convert a Qiskit circuit with symbolic parameters into a parameterized circuit.

        Each gate angle may be an affine expression ``a * p + b`` of at most one free
        Qiskit :class:`~qiskit.circuit.Parameter`. The trainable parameter ordering is
        given by ``parameters`` (defaults to ``circuit.parameters`` order).

        Args:
            circuit: The Qiskit circuit to convert.
            parameters: Trainable parameter ordering; defaults to ``circuit.parameters``.

        Returns:
            The equivalent :class:`ParameterizedCircuit`.

        Raises:
            ValueError: If an instruction is unsupported, an expression involves more
                than one free parameter, or an expression is not affine.
        """
        param_order = list(circuit.parameters) if parameters is None else list(parameters)
        param_positions = {param: idx for idx, param in enumerate(param_order)}

        gates: list[ParameterizedGate] = []
        for instruction in circuit.data:
            name = instruction.operation.name
            if name in {"barrier", "measure"}:
                continue
            sites = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
            params = list(instruction.operation.params)

            free_params = [p for p in params if hasattr(p, "parameters") and p.parameters]
            if not free_params:
                fixed = tuple(float(p) for p in params)
                if name in SINGLE_ANGLE_GATES and fixed:
                    gates.append(ParameterizedGate(name=name, sites=sites, angle_offset=fixed[0]))
                else:
                    gates.append(ParameterizedGate(name=name, sites=sites, fixed_params=fixed))
                continue

            if name not in SINGLE_ANGLE_GATES or len(params) != 1:
                msg = f"Parameterized instruction '{name}' is not a supported one-parameter gate."
                raise ValueError(msg)
            expression = params[0]
            symbols = list(expression.parameters)
            if len(symbols) != 1:
                msg = f"Angle expression {expression} must involve exactly one free parameter."
                raise ValueError(msg)
            symbol = symbols[0]
            if symbol not in param_positions:
                msg = f"Parameter {symbol} not found in the provided parameter ordering."
                raise ValueError(msg)

            gradient = expression.gradient(symbol)
            if hasattr(gradient, "parameters") and gradient.parameters:
                msg = f"Angle expression {expression} is not affine in {symbol}."
                raise ValueError(msg)
            scale = float(np.real(complex(gradient)))
            offset = float(np.real(complex(expression.assign(symbol, 0.0))))
            gates.append(
                ParameterizedGate(
                    name=name,
                    sites=sites,
                    param_index=param_positions[symbol],
                    angle_scale=scale,
                    angle_offset=offset,
                )
            )

        return cls(num_qubits=circuit.num_qubits, gates=gates, num_params=len(param_order))


def _append_u3_parameterization(
    gates: list[ParameterizedGate],
    site: int,
    base_index: int,
    *,
    adjoint: bool,
) -> None:
    """Append a scalar RZ-RY-RZ decomposition of a U3 gate or its adjoint."""
    theta, phi, lam = base_index, base_index + 1, base_index + 2
    if adjoint:
        gates.extend((
            ParameterizedGate("rz", (site,), param_index=phi, angle_scale=-1.0),
            ParameterizedGate("ry", (site,), param_index=theta, angle_scale=-1.0),
            ParameterizedGate("rz", (site,), param_index=lam, angle_scale=-1.0),
        ))
        return

    gates.extend((
        ParameterizedGate("rz", (site,), param_index=lam),
        ParameterizedGate("ry", (site,), param_index=theta),
        ParameterizedGate("rz", (site,), param_index=phi),
    ))


def _append_bmpd_gate_adjoint(
    gates: list[ParameterizedGate],
    sites: Sequence[int],
    base_index: int,
) -> None:
    """Append the scalar state-preparation adjoint of one BMPD two-qubit block."""
    left_site, right_site = sites
    gates.extend((
        ParameterizedGate("rxx", (left_site, right_site), param_index=base_index + 6, angle_scale=-1.0),
        ParameterizedGate("ryy", (left_site, right_site), param_index=base_index + 7, angle_scale=-1.0),
        ParameterizedGate("rzz", (left_site, right_site), param_index=base_index + 8, angle_scale=-1.0),
    ))
    _append_u3_parameterization(gates, left_site, base_index, adjoint=True)
    _append_u3_parameterization(gates, right_site, base_index + 3, adjoint=True)


def _append_numeric_trainable_angle(
    gates: list[ParameterizedGate],
    initial_theta: list[float],
    name: str,
    sites: tuple[int, ...],
    angle: float,
) -> None:
    """Append one trainable scalar gate initialized to a numeric angle."""
    param_index = len(initial_theta)
    initial_theta.append(float(angle))
    gates.append(ParameterizedGate(name, sites, param_index=param_index))


def _append_numeric_u_gate(
    gates: list[ParameterizedGate],
    initial_theta: list[float],
    site: int,
    theta: float,
    phi: float,
    lam: float,
) -> None:
    """Append a numeric U3 gate as trainable scalar RZ-RY-RZ factors."""
    site_tuple = (site,)
    _append_numeric_trainable_angle(gates, initial_theta, "rz", site_tuple, lam)
    _append_numeric_trainable_angle(gates, initial_theta, "ry", site_tuple, theta)
    _append_numeric_trainable_angle(gates, initial_theta, "rz", site_tuple, phi)


def _append_decomposed_instruction(
    gates: list[ParameterizedGate],
    initial_theta: list[float],
    name: str,
    params: Sequence[float],
    sites: tuple[int, ...],
) -> None:
    """Append a Qiskit decomposition instruction in Krotov-compatible form.

    Raises:
        ValueError: If the decomposition emits an unsupported instruction.
    """
    if name in {"barrier", "measure"}:
        return
    if name in {"u", "u3"}:
        if len(params) != 3:
            msg = f"Expected three parameters for Qiskit '{name}' instruction, got {len(params)}."
            raise ValueError(msg)
        _append_numeric_u_gate(gates, initial_theta, sites[0], float(params[0]), float(params[1]), float(params[2]))
        return
    if name in SINGLE_ANGLE_GATES:
        if len(params) != 1:
            msg = f"Expected one parameter for Qiskit '{name}' instruction, got {len(params)}."
            raise ValueError(msg)
        _append_numeric_trainable_angle(gates, initial_theta, name, sites, float(params[0]))
        return
    if not params and hasattr(GateLibrary, name):
        gates.append(ParameterizedGate(name, sites))
        return
    msg = f"Unsupported instruction '{name}' in SMPD KAK decomposition."
    raise ValueError(msg)


def _append_decomposed_circuit(
    gates: list[ParameterizedGate],
    initial_theta: list[float],
    site_tuple: tuple[int, ...],
    matrix: NDArray[np.complex128],
) -> None:
    """Decompose one dense SMPD matrix and append Krotov-compatible factors."""
    if len(site_tuple) == 1:
        decomposed = _ONE_QUBIT_DECOMPOSER(matrix)
    else:
        decomposed = _TWO_QUBIT_DECOMPOSER(_swap_two_site_convention(matrix))

    for instruction in decomposed.data:
        local_sites = tuple(decomposed.find_bit(qubit).index for qubit in instruction.qubits)
        physical_sites = tuple(site_tuple[local_site] for local_site in local_sites)
        _append_decomposed_instruction(
            gates,
            initial_theta,
            instruction.operation.name,
            tuple(float(param) for param in instruction.operation.params),
            physical_sites,
        )


def create_brickwall_matrix_product_disentangler_parameterized_circuit(
    num_qubits: int,
    depth: int,
    *,
    initial_single_qubit_layer: bool = False,
) -> ParameterizedCircuit:
    """Create a Krotov-compatible scalar-parameter BMPD state-preparation circuit.

    This is the parameterized counterpart of
    :func:`mqt.yaqs.core.libraries.circuit_library.create_brickwall_matrix_product_disentangler_ansatz_circuit`.
    Each dense nine-parameter BMPD block
    ``RXX(t6) RYY(t7) RZZ(t8) (U3(t0,t1,t2) x U3(t3,t4,t5))`` is emitted as
    supported one-parameter primitive gates in its state-preparation adjoint form.

    The trainable vector uses the same ordering as the dense BMPD ansatz: optional
    initial U3 product-state parameters first, then all nine-parameter BMPD rows
    flattened in disentangling-layer order with ascending bonds inside each layer.

    Args:
        num_qubits: Number of qubits.
        depth: Number of entangling gates per pair. The architecture has
            ``2 * depth`` brickwall layers.
        initial_single_qubit_layer: Whether to prepend a trainable U3 product-state
            layer with three scalar parameters per qubit.

    Returns:
        A :class:`ParameterizedCircuit` whose trainable factors are all supported
        by the deterministic MPS Krotov optimizer.
    """
    gates: list[ParameterizedGate] = []
    parameter_offset = 0

    if initial_single_qubit_layer:
        for site in range(num_qubits):
            _append_u3_parameterization(gates, site, parameter_offset + 3 * site, adjoint=False)
        parameter_offset = 3 * num_qubits

    row_layers: list[list[tuple[int, tuple[int, int]]]] = []
    row_index = 0
    for layer in brickwall_matrix_product_disentangler_bonds(num_qubits, depth):
        indexed_layer: list[tuple[int, tuple[int, int]]] = []
        for sites in layer:
            indexed_layer.append((row_index, sites))
            row_index += 1
        row_layers.append(indexed_layer)

    for indexed_layer in reversed(row_layers):
        for row, sites in indexed_layer:
            _append_bmpd_gate_adjoint(gates, sites, parameter_offset + 9 * row)

    num_params = brickwall_matrix_product_disentangler_num_parameters(
        num_qubits,
        depth,
        initial_single_qubit_layer=initial_single_qubit_layer,
    )
    return ParameterizedCircuit(num_qubits=num_qubits, gates=gates, num_params=num_params)


def create_sequential_matrix_product_disentangler_parameterized_circuit(
    num_qubits: int,
    dense_gates: Sequence[tuple[NDArray[np.complex128], Sequence[int]]],
) -> tuple[ParameterizedCircuit, NDArray[np.float64]]:
    """Convert an SMPD dense gate sequence into a Krotov-compatible circuit.

    One-qubit dense unitaries are decomposed into trainable ``rz-ry-rz`` factors.
    Two-qubit dense unitaries are decomposed with Qiskit's KAK/Weyl decomposition
    into trainable one-qubit scalar rotations around fixed ``cx`` gates. The returned
    initial parameter vector reproduces the original dense SMPD circuit up to a
    global phase.

    Args:
        num_qubits: Number of qubits.
        dense_gates: SMPD state-preparation gates as ``(matrix, sites)`` tuples,
            in the same convention accepted by
            :func:`mqt.yaqs.core.libraries.circuit_library.create_sequential_matrix_product_disentangler_circuit`.

    Returns:
        A tuple ``(circuit, initial_theta)`` where ``circuit`` is a
        :class:`ParameterizedCircuit` suitable for deterministic MPS Krotov
        optimization and ``initial_theta`` initializes it to the supplied dense
        SMPD circuit.
    """
    gates: list[ParameterizedGate] = []
    initial_theta: list[float] = []

    for matrix, sites in dense_gates:
        site_tuple = _validate_disentangler_sites(sites, num_qubits)
        mat = _validate_disentangler_unitary(matrix, len(site_tuple))
        _append_decomposed_circuit(gates, initial_theta, site_tuple, mat)

    return ParameterizedCircuit(num_qubits=num_qubits, gates=gates, num_params=len(initial_theta)), np.asarray(
        initial_theta,
        dtype=np.float64,
    )


__all__ = [
    "PARAMETRIC_DERIVATIVE_OPERATORS",
    "SINGLE_ANGLE_GATES",
    "ParameterizedCircuit",
    "ParameterizedGate",
    "brickwall_matrix_product_disentangler_num_parameters",
    "create_brickwall_matrix_product_disentangler_parameterized_circuit",
    "create_sequential_matrix_product_disentangler_parameterized_circuit",
]
