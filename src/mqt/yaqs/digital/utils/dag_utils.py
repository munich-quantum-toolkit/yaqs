# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility functions for DAG circuits.

This module implements conversion and processing functions for quantum circuits
using their DAG representations. It provides utilities to:

- Convert a ``DAGCircuit`` into a list of gate objects from the ``GateLibrary``.
- Extract a temporal zone from a ``DAGCircuit`` for specified qubits.
- Determine the maximum distance (in terms of qubit indices) of multi-qubit gates.
- Select starting points for gate application based on a checkerboard pattern.

Qiskit instructions whose ``op.name`` matches one of the 31 entries in
:data:`SUPPORTED_QISKIT_GATE_NAMES` are translated via hardcoded ``GateLibrary``
classes. From three qubits on, a controlled gate is built from its controls and target operator,
and a Pauli rotation or Pauli string from its terms, without the dense matrix of the gate. All
other unitary gates fall back to matrix-backed
:class:`~mqt.yaqs.core.libraries.gate_library.BaseGate` instances on at most eight qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe
from qiskit.circuit import (
    AnnotatedOperation,
    ControlledGate,
    ControlModifier,
    Gate,
    Operation,
    Parameter,
    ParameterExpression,
)
from qiskit.circuit.library import PauliEvolutionGate, PauliGate
from qiskit.converters import dag_to_circuit
from qiskit.dagcircuit import DAGOpNode
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator, SparsePauliOp

from ...core import linalg
from ...core.data_structures._sum_of_products import ExactComplex, sum_of_products_tensors
from ...core.data_structures.mpo import MPO
from ...core.data_structures.mpo_utils import make_identity_site
from ...core.libraries.gate_library import BaseGate, GateLibrary, extend_gate
from ...core.libraries.operator_matrices import (
    IDENTITY,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    PROJECTOR_ONE,
    PROJECTOR_ZERO,
)
from ...core.methods.decompositions import right_qr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from qiskit.circuit import Qubit
    from qiskit.dagcircuit import DAGCircuit

# ``barrier`` is ignored during DAG-to-gate conversion. ``measure`` is rejected
# here because conversion builds a unitary gate list; mid-circuit measurements are
# not representable in that form. Circuit simulation (`process_layer` in
# ``digital_tjm``) still drops ``measure`` nodes from the live DAG before gate
# application so terminal measurements in a Qiskit circuit do not block simulation.
_SKIP_INSTRUCTIONS = frozenset({"barrier"})
_ZONE_SKIP_INSTRUCTIONS = frozenset({"measure", "barrier"})
_REJECTED_INSTRUCTIONS = frozenset({"reset", "delay", "store", "measure"})
# Qiskit ``op.name`` values resolved through ``getattr(GateLibrary, name)``.
SUPPORTED_QISKIT_GATE_NAMES: tuple[str, ...] = (
    "ccx",
    "ccz",
    "cp",
    "cswap",
    "cx",
    "cz",
    "h",
    "i",
    "id",
    "iden",
    "p",
    "rx",
    "ry",
    "rz",
    "rxx",
    "ryy",
    "rzz",
    "s",
    "sdg",
    "swap",
    "sx",
    "sxdg",
    "t",
    "tdg",
    "u",
    "u1",
    "u2",
    "u3",
    "x",
    "y",
    "z",
)
_CONTROL_FLOW_INSTRUCTIONS = frozenset({
    "for_loop",
    "while_loop",
    "if_else",
    "switch_case",
    "break_loop",
    "continue_loop",
})


def convert_matrix_layout(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Convert a Qiskit unitary matrix into YAQS gate storage order.

    The map reverses the qubit axis order within the output and the input index group.
    It is an involution: applying it twice returns the input.

    Args:
        matrix: Dense ``2**n x 2**n`` unitary in Qiskit little-endian ordering.

    Returns:
        Matrix that reshapes to the YAQS gate tensor convention.
    """
    num_qubits = int(np.log2(matrix.shape[0]))
    permutation = (*reversed(range(num_qubits)), *reversed(range(num_qubits, 2 * num_qubits)))
    tensor = np.reshape(matrix, (2,) * (2 * num_qubits)).transpose(permutation)
    return np.asarray(tensor.reshape(matrix.shape), dtype=np.complex128)


def _get_qubit_indices(dag: DAGCircuit | None, node: DAGOpNode) -> list[int]:
    """Return MPS site indices for a DAG operation node.

    Args:
        dag: Parent DAG when available; used for ``find_bit`` lookup. Without it, the index within the register is used.
        node: DAG operation node.

    Returns:
        Qubit indices in circuit order.
    """
    indices: list[int] = []
    for qubit in node.qargs:
        if dag is not None:
            indices.append(dag.find_bit(qubit).index)
        else:
            indices.append(qubit._index)  # ruff:ignore[private-member-access]
    return indices


def _has_unbound_params(op: object) -> bool:
    """Return whether a Qiskit operation still has free symbolic parameters.

    Args:
        op: Qiskit instruction or gate.

    Returns:
        True if any entry in ``op.params`` is an unbound ``Parameter`` or
        ``ParameterExpression`` with remaining free symbols.
    """
    params = getattr(op, "params", ())
    return any(
        isinstance(param, Parameter) or (isinstance(param, ParameterExpression) and bool(param.parameters))
        for param in params
    )


def _is_unitary(matrix: NDArray[np.complex128], *, atol: float = 1e-10) -> bool:
    """Return whether ``matrix`` is unitary within tolerance.

    Args:
        matrix: Square complex operator matrix.
        atol: Absolute tolerance for ``matrix @ matrix.conj().T ≈ I``.

    Returns:
        True if the matrix is unitary within ``atol``.
    """
    dim = matrix.shape[0]
    product = matrix @ matrix.conj().T
    return bool(np.allclose(product, np.eye(dim, dtype=np.complex128), atol=atol))


def _extract_matrix(op: Operation, *, name: str) -> NDArray[np.complex128]:
    """Extract a unitary matrix from a Qiskit operation.

    Args:
        op: Qiskit instruction or gate.
        name: Operation name for error reporting.

    Returns:
        Complex unitary matrix.

    Raises:
        ValueError: If no matrix representation is available.
    """
    to_matrix = getattr(op, "to_matrix", None)
    if callable(to_matrix):
        try:
            matrix = to_matrix()
        except (TypeError, QiskitError) as exc:
            try:
                return np.asarray(Operator(op).data, dtype=np.complex128)
            except (TypeError, QiskitError, ValueError):
                msg = f"Cannot translate Qiskit instruction '{name}': failed to build a matrix representation."
                raise ValueError(msg) from exc
        if matrix is not None:
            return np.asarray(matrix, dtype=np.complex128)

    try:
        return np.asarray(Operator(op).data, dtype=np.complex128)
    except (TypeError, QiskitError, ValueError) as exc:
        msg = f"Cannot translate Qiskit instruction '{name}': no matrix representation available."
        raise ValueError(msg) from exc


def _reject_unsupported(node: DAGOpNode) -> None:
    """Raise for Qiskit instructions that YAQS cannot translate.

    Args:
        node: DAG operation node.

    Raises:
        ValueError: If the instruction is unsupported.
    """
    name = node.op.name
    if name in _REJECTED_INSTRUCTIONS:
        if name == "measure":
            msg = (
                f"Cannot translate Qiskit instruction '{name}': mid-circuit measurements are not supported; "
                "remove measure instructions before conversion."
            )
        else:
            msg = f"Cannot translate Qiskit instruction '{name}': {name} is not supported in YAQS circuit simulation."
        raise ValueError(msg)
    if name in _CONTROL_FLOW_INSTRUCTIONS:
        msg = f"Cannot translate Qiskit instruction '{name}': control-flow operations are not supported."
        raise ValueError(msg)
    if getattr(node.op, "condition", None) is not None:
        msg = f"Cannot translate Qiskit instruction '{name}': conditional operations are not supported."
        raise ValueError(msg)
    if node.cargs:
        msg = f"Cannot translate Qiskit instruction '{name}': classically controlled operations are not supported."
        raise ValueError(msg)


# Bound on the dense matrix fallback: a ``2**n x 2**n`` complex128 matrix reaches
# 1 MB at n = 8 and grows by a factor of 16 per added qubit.
_MAX_MATRIX_FALLBACK_QUBITS = 8
# Narrower gates keep their dense matrix, which the one- and two-site paths read directly.
_MIN_PARTS_QUBITS = 3
_PAULI_LABELS = frozenset("IXYZ")
_PAULI_MATRIX = {"I": IDENTITY, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}
# The number of Pauli strings sets the width of every tensor, and m commuting rotations give 2**m.
_MAX_PAULI_TERMS = 64
# Products of distinct Pauli factors; the identity and equal factors are handled in code.
_PAULI_PRODUCT: dict[tuple[str, str], tuple[complex, str]] = {
    ("X", "Y"): (1j, "Z"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "X"): (1j, "Y"),
    ("X", "Z"): (-1j, "Y"),
}


def is_digital_noise_opportunity(operation: Operation) -> bool:
    """Return whether a Qiskit operation is an eligible two-qubit noise opportunity.

    Args:
        operation: A Qiskit circuit operation.

    Returns:
        ``True`` when ``operation`` is a supported two-qubit unitary.
    """
    if operation.num_qubits != 2:
        return False

    name = operation.name
    if name in _SKIP_INSTRUCTIONS or name in _REJECTED_INSTRUCTIONS or name in _CONTROL_FLOW_INSTRUCTIONS:
        return False
    if getattr(operation, "condition", None) is not None:
        return False
    if _has_unbound_params(operation):
        return False
    if name in SUPPORTED_QISKIT_GATE_NAMES:
        return True
    try:
        matrix = _extract_matrix(operation, name=name)
    except ValueError:
        return False
    return _is_unitary(matrix)


def _translate_matrix(op: Operation, *, name: str, sites: list[int]) -> BaseGate:
    """Build a YAQS gate from a Qiskit operation matrix.

    Args:
        op: Qiskit instruction or gate.
        name: Qiskit operation name to store on the YAQS gate.
        sites: Target qubit indices.

    Returns:
        Matrix-backed YAQS gate without an analytic generator.

    Raises:
        ValueError: If parameters are unbound, the gate is too wide for the matrix
            fallback, or the operator is not unitary.
    """
    if _has_unbound_params(op):
        msg = f"Cannot translate Qiskit gate '{name}': unbound parameters; bind parameters before simulation."
        raise ValueError(msg)

    if len(sites) > _MAX_MATRIX_FALLBACK_QUBITS:
        msg = (
            f"Cannot translate Qiskit gate '{name}': the matrix fallback supports at most "
            f"{_MAX_MATRIX_FALLBACK_QUBITS} qubits, got {len(sites)}; decompose the instruction before simulation."
        )
        raise ValueError(msg)

    matrix = _extract_matrix(op, name=name)
    if not _is_unitary(matrix):
        msg = f"Cannot translate Qiskit gate '{name}': operator is not unitary."
        raise ValueError(msg)

    if len(sites) >= 2:
        matrix = convert_matrix_layout(matrix)

    gate = GateLibrary.custom(matrix)
    gate.name = name
    gate.set_sites(*sites)
    return gate


def _reduce_end_block(
    tensors: list[NDArray[np.complex128]],
) -> tuple[NDArray[np.complex128], list[NDArray[np.complex128]]]:
    """Reduce every bond of an MPO block whose left bond is open, the open bond included.

    Args:
        tensors: MPO tensors with axes ``(out, in, left, right)``; the last right bond is 1.

    Returns:
        The map from the open bond to its reduced bond, and the block with the minimal bond on every cut.
    """
    tensors = list(tensors)
    for i in range(len(tensors) - 1):
        d_out, d_in, left, right = tensors[i].shape
        q_tensor, r_mat = right_qr(tensors[i].reshape(d_out * d_in, left, right))
        tensors[i] = q_tensor.reshape(d_out, d_in, left, -1)
        tensors[i + 1] = oe.contract("kr,abrs->abks", r_mat, tensors[i + 1])
    for i in range(len(tensors) - 1, -1, -1):
        d_out, d_in, left, right = tensors[i].shape
        matrix = tensors[i].transpose(2, 0, 1, 3).reshape(left, d_out * d_in * right)
        u_mat, s_list, v_mat = linalg.svd(matrix, full_matrices=False)
        # Relative 1e-12 is a round-off floor, not a simulation tolerance.
        keep = linalg.truncate(s_list, mode="relative", threshold=1e-12, min_keep=1)
        tensors[i] = v_mat[:keep].reshape(keep, d_out, d_in, right).transpose(1, 2, 0, 3)
        boundary = u_mat[:, :keep] * s_list[:keep]
        if i > 0:
            tensors[i - 1] = oe.contract("abml,lk->abmk", tensors[i - 1], boundary)
    return boundary, tensors


def _control_window_mpo(
    controls: dict[int, int],
    window: list[NDArray[np.complex128]],
    *,
    window_start: int,
    first_site: int,
    last_site: int,
) -> list[NDArray[np.complex128]]:
    """Return the identity plus the controls times a target window, minimal bond past the outer controls.

    Args:
        controls: Control sites mapped to the value each one must read.
        window: Target tensors from ``window_start`` onwards, one per site.
        window_start: First site the window covers.
        first_site: First site of the gate span.
        last_site: Last site of the gate span.

    Returns:
        Tensors ``(out, in, left, right)`` for every site from ``first_site`` to ``last_site``.
    """
    projectors = (PROJECTOR_ZERO, PROJECTOR_ONE)
    window_sites = dict(zip(range(window_start, window_start + len(window)), window, strict=True))
    channel = []
    for site in range(first_site, last_site + 1):
        tensor = window_sites.get(site, make_identity_site(2))
        if site in controls:
            tensor = oe.contract("ab,bclr->aclr", projectors[controls[site]], tensor)
        channel.append(tensor)
    channel_mpo = MPO()
    channel_mpo.custom(channel, transpose=False)
    tensors = (MPO.identity(len(channel)) + channel_mpo).tensors
    # Only the end blocks are reduced: the controls' identity string outweighs U - I by 2**(m/2).
    first_control, last_control = min(controls) - first_site, max(controls) - first_site
    if last_control < len(tensors) - 1:
        boundary, tensors[last_control + 1 :] = _reduce_end_block(tensors[last_control + 1 :])
        tensors[last_control] = oe.contract("abml,lk->abmk", tensors[last_control], boundary)
    if first_control > 0:
        mirrored = [tensor.transpose(0, 1, 3, 2) for tensor in reversed(tensors[:first_control])]
        boundary, mirrored = _reduce_end_block(mirrored)
        tensors[:first_control] = [tensor.transpose(0, 1, 3, 2) for tensor in reversed(mirrored)]
        tensors[first_control] = oe.contract("lk,ablr->abkr", boundary, tensors[first_control])
    return tensors


def _identity_multiple(matrix: NDArray[np.complex128]) -> complex | None:
    """Return ``c`` if ``matrix`` is ``c`` times the identity up to round-off relative to ``matrix``, else ``None``.

    Args:
        matrix: Square matrix.

    Returns:
        The multiple, ``0`` for the zero matrix, or ``None``.
    """
    if not matrix.any():
        return 0j
    multiple = np.trace(matrix) / len(matrix)
    if np.linalg.norm(matrix - multiple * np.eye(len(matrix))) <= 1e-12 * np.linalg.norm(matrix):
        return complex(multiple)
    return None


def _controlled_product_mpo(
    controls: dict[int, int], targets: list[int], factors: list[NDArray[np.complex128]]
) -> list[NDArray[np.complex128]]:
    """Return the MPO of ``I + |b><b| (x) u_1 (x) ... (x) u_k - |b><b| (x) I`` at the minimal bond.

    Args:
        controls: Control sites mapped to the value each one must read.
        targets: Target sites, in the factor order of ``factors``.
        factors: One-qubit factor of each target.

    Returns:
        Tensors ``(out, in, left, right)`` for every site from the first to the last involved site.
    """
    first_site, last_site = min([*controls, *targets]), max([*controls, *targets])
    projectors = (PROJECTOR_ZERO, PROJECTOR_ONE)
    control_ops = {site - first_site: projectors[value] for site, value in controls.items()}
    controlled = dict(control_ops)
    phase = 1.0 + 0j
    for target, factor in zip(targets, factors, strict=True):
        # A factor c I acts on no qubit, so c joins the coefficient of the controlled string.
        shift = _identity_multiple(factor - IDENTITY)
        if shift is None:
            controlled[target - first_site] = factor
        else:
            phase *= 1 + shift
    terms = [(1.0, {}), (phase, controlled), (-1.0, control_ops)]
    return sum_of_products_tensors(terms, last_site - first_site + 1)


def _controlled_gate_mpo(
    controls: dict[int, int], targets: list[int], target_matrix: NDArray[np.complex128]
) -> list[NDArray[np.complex128]]:
    """Return the MPO of ``|b><b| (x) U + (I - |b><b|) (x) I`` at the minimal bond on every cut.

    Args:
        controls: Control sites mapped to the value each one must read.
        targets: Target sites in the factor order of ``target_matrix``.
        target_matrix: Target unitary in YAQS factor order.

    Returns:
        Tensors ``(out, in, left, right)`` for every site from the first to the last involved site.
    """
    first_site, last_site = min([*controls, *targets]), max([*controls, *targets])
    difference = target_matrix - np.eye(len(target_matrix), dtype=np.complex128)
    phase = _identity_multiple(difference)
    if phase is not None:
        return _controlled_product_mpo(controls, targets, [(1 + phase) * IDENTITY] + [IDENTITY] * (len(targets) - 1))
    if len(targets) == 1:
        return _controlled_product_mpo(controls, targets, [target_matrix])
    # An entangling target needs the operator-Schmidt terms of U - I, found by one SVD.
    window = extend_gate(np.reshape(difference, (2,) * (2 * len(targets))), targets)
    return _control_window_mpo(controls, window, window_start=min(targets), first_site=first_site, last_site=last_site)


class _ControlledGate(BaseGate):
    """Controlled gate given by its MPO alone, without a dense ``matrix`` or ``tensor``."""

    def __init__(
        self,
        target: NDArray[np.complex128] | list[NDArray[np.complex128]],
        control_values: list[int],
    ) -> None:
        """Store the target operator, or its one-qubit factors, and the value each control must read.

        Args:
            target: Target unitary in YAQS factor order, or one factor per target.
            control_values: Required value of each control, in control order.
        """
        self._target = target
        self._control_values = control_values
        num_targets = len(target) if isinstance(target, list) else int(np.log2(target.shape[0]))
        self.interaction = len(control_values) + num_targets

    def set_sites(self, *sites: int | list[int]) -> None:
        """Bind the controls, then the targets, and build the MPO.

        Args:
            *sites: Site indices or lists of indices, controls before targets.

        Raises:
            ValueError: If the site count differs from ``interaction``.
        """
        sites_list = []
        for s in sites:
            if isinstance(s, int):
                sites_list.append(s)
            else:
                sites_list.extend(s)

        if len(sites_list) != self.interaction:
            msg = f"Number of sites {len(sites_list)} must be equal to the interaction level {self.interaction}"
            raise ValueError(msg)

        self.sites = sites_list
        num_controls = len(self._control_values)
        controls = dict(zip(sites_list[:num_controls], self._control_values, strict=True))
        targets = sites_list[num_controls:]
        self.mpo_tensors = (
            _controlled_product_mpo(controls, targets, self._target)
            if isinstance(self._target, list)
            else _controlled_gate_mpo(controls, targets, self._target)
        )


def _pauli_product(left: str, right: str) -> tuple[complex, str]:
    """Return ``(phase, label)`` such that ``left @ right == phase * label``.

    Args:
        left: Pauli label on the left.
        right: Pauli label on the right.

    Returns:
        The phase and the resulting Pauli label.
    """
    # The caller drops the identity from a term's factors, so only ``left`` can be the identity.
    if left == "I":
        return 1.0, right
    if left == right:
        return 1.0, "I"
    return _PAULI_PRODUCT[left, right]


def _rotation_terms(
    labels: list[str], coeffs: list[float], time: float
) -> list[tuple[ExactComplex, dict[int, str]]] | None:
    """Expand ``exp(-i time sum_k coeffs_k P_k)`` of commuting ``P_k`` into Pauli strings with exact coefficients.

    Args:
        labels: Pauli labels in Qiskit order, one per term.
        coeffs: Real coefficient of each term.
        time: Evolution time of the gate.

    Returns:
        ``(coefficient, {factor position: Pauli label})`` pairs, or ``None`` above the term budget.
    """
    zero = ExactComplex.of(0)
    accumulated: dict[tuple[tuple[int, str], ...], ExactComplex] = {(): ExactComplex.of(1)}
    for label, coeff in zip(labels, coeffs, strict=True):
        angle = time * coeff
        # Each factor is cos(angle) I - i sin(angle) P_k.
        cosine, sine = ExactComplex.of(np.cos(angle)), ExactComplex.of(-1j * np.sin(angle))
        factors = {factor: op for factor, op in enumerate(reversed(label)) if op != "I"}
        expanded: dict[tuple[tuple[int, str], ...], ExactComplex] = {}
        for key, value in accumulated.items():
            expanded[key] = expanded.get(key, zero) + value * cosine
            merged, phase = dict(key), sine
            for factor, op in factors.items():
                step, result = _pauli_product(merged.get(factor, "I"), op)
                phase *= ExactComplex.of(step)
                if result == "I":
                    merged.pop(factor, None)
                else:
                    merged[factor] = result
            product_key = tuple(sorted(merged.items()))
            expanded[product_key] = expanded.get(product_key, zero) + value * phase
        accumulated = {key: value for key, value in expanded.items() if value}
        if len(accumulated) > _MAX_PAULI_TERMS:
            return None
    return [(value, dict(key)) for key, value in accumulated.items()]


def _pauli_sum_tensors(
    terms: Sequence[tuple[complex | ExactComplex, dict[int, str]]], sites: list[int]
) -> list[NDArray[np.complex128]]:
    """Return the MPO of a sum of Pauli strings over the span of ``sites``, at the minimal bond.

    Args:
        terms: ``(coefficient, {factor position: Pauli label})`` pairs.
        sites: Physical site of each factor position.

    Returns:
        Tensors ``(out, in, left, right)`` from the first to the last site of the span.
    """
    first_site = min(sites)
    strings = [
        (coeff, {sites[factor] - first_site: _PAULI_MATRIX[label] for factor, label in factors.items()})
        for coeff, factors in terms
    ]
    return sum_of_products_tensors(strings, max(sites) - first_site + 1)


class _PauliSumGate(BaseGate):
    """Gate given by a short sum of Pauli strings, without a dense ``matrix`` or ``tensor``."""

    def __init__(self, terms: Sequence[tuple[complex | ExactComplex, dict[int, str]]], num_sites: int) -> None:
        """Store the Pauli-string terms and the number of sites they act on.

        Args:
            terms: ``(coefficient, {factor position: Pauli label})`` pairs.
            num_sites: Number of sites the gate acts on.
        """
        self._terms = terms
        self.interaction = num_sites

    def set_sites(self, *sites: int | list[int]) -> None:
        """Bind the sites in factor order and build the MPO.

        Args:
            *sites: Site indices or lists of indices, in the gate's factor order.

        Raises:
            ValueError: If the site count differs from ``interaction``.
        """
        sites_list: list[int] = []
        for s in sites:
            if isinstance(s, int):
                sites_list.append(s)
            else:
                sites_list.extend(s)

        if len(sites_list) != self.interaction:
            msg = f"Number of sites {len(sites_list)} must be equal to the interaction level {self.interaction}"
            raise ValueError(msg)

        self.sites = sites_list
        self.mpo_tensors = _pauli_sum_tensors(self._terms, sites_list)


def _translate_pauli_evolution(op: PauliEvolutionGate, *, name: str, sites: list[int]) -> BaseGate | None:
    """Build a Pauli-string rotation from its terms, or return ``None`` to fall back.

    Args:
        op: Qiskit Pauli evolution gate.
        name: Qiskit operation name to store on the YAQS gate.
        sites: Qubit indices in the gate's factor order.

    Returns:
        The gate, or ``None`` if its operator is not a commuting real Pauli sum within budget.
    """
    operator = op.operator
    if not isinstance(operator, SparsePauliOp) or operator.num_qubits != len(sites):
        return None
    try:
        time = float(op.time)
    except (TypeError, ValueError):
        return None
    labels = list(operator.paulis.to_labels())
    if any(set(label) - _PAULI_LABELS for label in labels):
        return None
    # Qiskit rejects complex coefficients when the gate is built, so the generator is Hermitian.
    coeffs = [float(complex(value).real) for value in operator.coeffs]
    # Terms i and j commute iff x_i . z_j + z_i . x_j is even, checked for all pairs at once.
    x_bits = operator.paulis.x.astype(np.int32)
    z_bits = operator.paulis.z.astype(np.int32)
    if ((x_bits @ z_bits.T + z_bits @ x_bits.T) % 2).any():
        return None
    terms = _rotation_terms(labels, coeffs, time)
    if terms is None:
        return None
    gate = _PauliSumGate(terms, len(sites))
    gate.name = name
    gate.set_sites(*sites)
    return gate


def _translate_pauli_string(op: PauliGate, *, name: str, sites: list[int]) -> BaseGate | None:
    """Build a Pauli string from its label, or return ``None`` to fall back.

    Args:
        op: Qiskit Pauli gate.
        name: Qiskit operation name to store on the YAQS gate.
        sites: Qubit indices in the gate's factor order.

    Returns:
        The gate, or ``None`` if the label is not a plain Pauli string on these sites.
    """
    label = op.params[0] if op.params else None
    if not isinstance(label, str) or len(label) != len(sites) or set(label) - _PAULI_LABELS:
        return None
    factors = {factor: single for factor, single in enumerate(reversed(label)) if single != "I"}
    gate = _PauliSumGate([(1.0 + 0j, factors)], len(sites))
    gate.name = name
    gate.set_sites(*sites)
    return gate


def _as_controlled(op: Operation) -> tuple[Gate | None, int, int, int] | None:
    """Return ``(base gate, number of controls, control state, number of targets)``, else ``None``.

    Args:
        op: Qiskit operation.

    Returns:
        The controls, base gate and target count, or ``None`` if ``op`` is not a plain controlled gate.
    """
    if isinstance(op, ControlledGate):
        base = op.base_gate
        # Only a multi-target gate declares a target count; other extra qubits are ancillas.
        declared = getattr(op, "num_target_qubits", None)
        num_targets = declared if declared is not None else (0 if base is None else base.num_qubits)
        return base, op.num_ctrl_qubits, op.ctrl_state, num_targets
    # Qiskit 3.0 returns an AnnotatedOperation from Gate.control(); one control modifier is the flat gate.
    if isinstance(op, AnnotatedOperation) and len(op.modifiers) == 1:
        modifier = op.modifiers[0]
        if isinstance(modifier, ControlModifier) and isinstance(op.base_op, Gate):
            num_controls = modifier.num_ctrl_qubits
            ctrl_state = modifier.ctrl_state
            num_targets = op.base_op.num_qubits
            # ``ControlModifier`` normalises its control state to an integer on construction.
            if isinstance(ctrl_state, int):
                return op.base_op, num_controls, ctrl_state, num_targets
    return None


def _target_factors(base: Gate, num_targets: int, *, name: str) -> list[NDArray[np.complex128]] | None:
    """Return the one-qubit factors of a target block that is their tensor product, or ``None``.

    Args:
        base: Target operator of the controlled gate.
        num_targets: Number of target qubits the gate declares.
        name: Qiskit operation name, for the error of a base without a matrix.

    Returns:
        One factor per target, or ``None`` if the block is not a product of one-qubit factors.
    """
    label = base.params[0] if isinstance(base, PauliGate) and base.params else None
    if isinstance(label, str) and len(label) == num_targets:
        # Label character ``label[-1 - j]`` acts on base qubit ``j``, so the factors run reversed.
        # Qiskit validates the label of a ``PauliGate``, so every character is a Pauli.
        return [_PAULI_MATRIX[single] for single in reversed(label)]
    # Any other base acts on one qubit and repeats on every target, as in ``MCMTGate``.
    if base.num_qubits != 1:
        return None
    try:
        factor = _extract_matrix(base, name=name)
    except ValueError:
        return None
    if not _is_unitary(factor):
        return None
    return [factor] * num_targets


def _translate_controlled(
    base: Gate | None,
    *,
    num_controls: int,
    ctrl_state: int,
    num_targets: int,
    name: str,
    sites: list[int],
) -> BaseGate | None:
    """Build a controlled gate from its controls and target operator, or return ``None`` to fall back.

    Args:
        base: Target operator of the controlled gate.
        num_controls: Number of control qubits, which come first in ``sites``.
        ctrl_state: Value the controls must read, as a Qiskit control state.
        num_targets: Number of target qubits the gate declares.
        name: Qiskit operation name to store on the YAQS gate.
        sites: Qubit indices, controls before targets.

    Returns:
        The gate, or ``None`` if it is not controls plus a unitary operator this path can build.
    """
    if base is None or num_controls < 1 or num_targets < 1 or num_controls + num_targets != len(sites):
        return None
    target: NDArray[np.complex128] | list[NDArray[np.complex128]] | None = None
    if num_targets >= 2:
        # A product of one-qubit factors needs no target matrix at any width.
        target = _target_factors(base, num_targets, name=name)
    if target is None and num_targets <= _MAX_MATRIX_FALLBACK_QUBITS and base.num_qubits == num_targets:
        try:
            target_matrix = _extract_matrix(base, name=name)
        except ValueError:
            return None
        if not _is_unitary(target_matrix):
            return None
        target = convert_matrix_layout(target_matrix) if num_targets >= 2 else target_matrix
    if target is None:
        return None
    gate = _ControlledGate(target, [(ctrl_state >> i) & 1 for i in range(num_controls)])
    gate.name = name
    gate.set_sites(*sites)
    return gate


def _translate_node(dag: DAGCircuit | None, node: DAGOpNode) -> BaseGate:
    """Convert a single DAG operation node into a YAQS gate.

    Args:
        dag: Parent DAG when available.
        node: DAG operation node.

    Returns:
        Initialized YAQS gate with sites assigned.

    Raises:
        ValueError: If the instruction cannot be translated.
    """
    name = node.op.name
    _reject_unsupported(node)

    if _has_unbound_params(node.op):
        msg = f"Cannot translate Qiskit gate '{name}': unbound parameters; bind parameters before simulation."
        raise ValueError(msg)

    sites = _get_qubit_indices(dag, node)
    if name not in SUPPORTED_QISKIT_GATE_NAMES:
        op = node.op
        if len(sites) >= _MIN_PARTS_QUBITS:
            controlled = _as_controlled(op)
            if controlled is not None:
                base, num_controls, ctrl_state, num_targets = controlled
                gate = _translate_controlled(
                    base,
                    num_controls=num_controls,
                    ctrl_state=ctrl_state,
                    num_targets=num_targets,
                    name=name,
                    sites=sites,
                )
                if gate is not None:
                    return gate
            if isinstance(op, PauliEvolutionGate):
                gate = _translate_pauli_evolution(op, name=name, sites=sites)
                if gate is not None:
                    return gate
            if isinstance(op, PauliGate):
                gate = _translate_pauli_string(op, name=name, sites=sites)
                if gate is not None:
                    return gate
        return _translate_matrix(op, name=name, sites=sites)

    try:
        gate_cls = getattr(GateLibrary, name)
    except AttributeError:
        return _translate_matrix(node.op, name=name, sites=sites)

    gate_object = gate_cls(node.op.params) if node.op.params else gate_cls()
    gate_object.set_sites(*sites)
    return gate_object


def convert_dag_to_tensor_algorithm(dag: DAGCircuit) -> list[BaseGate]:
    """Convert a DAGCircuit into a list of gate objects from the GateLibrary.

    This function traverses the input DAGCircuit (or a single DAGOpNode) and creates a list of gate objects.
    For each node, it retrieves the corresponding gate class from the GateLibrary, initializes it, sets any
    parameters if present, and assigns the qubit indices (sites) on which the gate acts.

    From three qubits on, controlled gates, Pauli rotations and Pauli strings are built as an MPO from
    their parts. Other unknown unitary Qiskit gates on up to eight qubits are converted from their
    matrix representation.
    ``barrier`` nodes are skipped. ``measure``, ``reset``, ``delay``, classically controlled ops,
    and control-flow instructions are rejected. See :data:`SUPPORTED_QISKIT_GATE_NAMES` for
    hardcoded gate names.

    Note:
        This conversion path rejects ``measure`` because it builds a unitary gate list. Circuit
        simulation removes ``measure`` nodes from the live DAG separately (see ``process_layer`` in
        ``digital_tjm``) so terminal Qiskit measurements do not block evolution.

    Args:
        dag: The DAGCircuit (or a single DAGOpNode) representing a quantum operation.

    Returns:
        A list of gate objects, each with its ``sites`` assigned.
    """
    algorithm: list[BaseGate] = []

    if isinstance(dag, DAGOpNode):
        algorithm.append(_translate_node(None, dag))
    else:
        parent_dag = dag
        for gate in dag.op_nodes():
            name = gate.op.name
            if name in _SKIP_INSTRUCTIONS:
                continue
            algorithm.append(_translate_node(parent_dag, gate))

    return algorithm


def extract_temporal_zone(dag: DAGCircuit, qubits: list[int]) -> DAGCircuit:
    """Extract the temporal zone without modifying ``dag``.

    Args:
        dag: The input DAGCircuit (unchanged).
        qubits: Qubit indices defining the strip ``[min(qubits), max(qubits)]``.

    Returns:
        A new DAGCircuit containing only operations in the temporal zone.
    """
    return _build_temporal_zone(dag.copy(), qubits)


def get_temporal_zone(dag: DAGCircuit, qubits: list[int]) -> DAGCircuit:
    """Extract a temporal zone and remove consumed operations from the input DAG.

    Starts with all qubits in the inclusive strip ``[min(qubits), max(qubits)]``.
    Keeps operations whose qubits remain active. A crossing gate removes
    its overlapping qubits from the active set for subsequent operations.
    Skips measurement and barrier nodes encountered during extraction.

    Args:
        dag: Input DAG, mutated in place.
        qubits: Indices defining the initial strip.

    Returns:
        New DAG containing the extracted operations.
    """
    return _build_temporal_zone(dag, qubits)


def _build_temporal_zone(dag: DAGCircuit, qubits: list[int]) -> DAGCircuit:
    """Build a temporal-zone DAG and remove consumed nodes from ``dag``.

    Returns:
        A new DAGCircuit containing only operations in the temporal zone.
    """
    new_dag = dag.copy_empty_like()
    layers = list(dag.multigraph_layers())
    qubits_to_check: set[Qubit] = set()
    qubits_to_check.update(dag.qubits[qubit] for qubit in range(min(qubits), max(qubits) + 1))

    for layer in layers:
        for node in layer:
            if isinstance(node, DAGOpNode):
                qubit_set = set(node.qargs)

                # If the gate acts entirely within the current cone of qubits.
                if qubit_set <= qubits_to_check:
                    if node.op.name in _ZONE_SKIP_INSTRUCTIONS:
                        dag.remove_op_node(node)
                        continue
                    new_dag.apply_operation_back(node.op, node.qargs)
                    dag.remove_op_node(node)
                else:
                    # For partial overlap, remove the overlapping qubits from the cone.
                    if node.op.name in _ZONE_SKIP_INSTRUCTIONS:
                        dag.remove_op_node(node)
                        continue
                    for item in qubit_set & qubits_to_check:
                        qubits_to_check.remove(item)

        # Stop once no qubits remain in the cone.
        if len(qubits_to_check) == 0:
            break

    return new_dag


def check_longest_gate(dag: DAGCircuit) -> int:
    """Determine the maximum distance between qubits in any multi-qubit gate in the first layer.

    This function inspects the first layer of the DAGCircuit and computes the distance between the first
    and last qubits involved in each multi-qubit gate. The distance is defined in terms of qubit indices.
    A result of 1 or 2 indicates that only nearest-neighbor gates are present.

    Args:
        dag (DAGCircuit): The DAGCircuit to inspect.

    Returns:
        int: The largest distance (in terms of qubit indices) found among the multi-qubit gates in the first layer.
    """
    largest_distance = 1
    first_layer = next(dag.layers(), None)

    if first_layer is not None:
        layer_circuit = dag_to_circuit(first_layer["graph"])
        for gate in layer_circuit.data:
            if gate.operation.num_qubits > 1:
                first_site = layer_circuit.find_bit(gate.qubits[0]).index
                last_site = layer_circuit.find_bit(gate.qubits[-1]).index
                largest_distance = max(largest_distance, abs(first_site - last_site) + 1)

    return largest_distance


def select_starting_point(num_qubits: int, dag: DAGCircuit) -> tuple[range, range]:
    """Determine the starting set of neighboring qubits (even-even or odd-odd) for gate application.

    This function selects a checkerboard pattern for gate application based on the layout of gates in the first
    layer of the DAGCircuit. It returns two ranges of qubit indices that define the groups of neighboring qubits
    to be used as starting points.

    Args:
        num_qubits (int): Total number of qubits (or sites) in the system.
        dag (DAGCircuit): The DAGCircuit used to inspect the first set of gates.

    Returns:
        tuple[range, range]: A tuple containing two ranges:
            - The first range corresponds to the first group of qubits.
            - The second range corresponds to the complementary group.
    """
    assert num_qubits > 1

    first_layer = next(dag.layers(), None)

    first_iterator = range(0, num_qubits - 1, 2)
    second_iterator = range(1, num_qubits - 1, 2)
    odd = False

    if first_layer is not None:
        layer_circuit = dag_to_circuit(first_layer["graph"])
        for gate in layer_circuit.data:
            # If a two-qubit gate appears with an odd-indexed starting qubit, switch the ordering.
            if gate.operation.num_qubits == 2:
                if layer_circuit.find_bit(gate.qubits[0]).index % 2 != 0:
                    odd = True
                break

        if odd:
            first_iterator = range(1, num_qubits - 1, 2)
            second_iterator = range(0, num_qubits - 1, 2)

    return first_iterator, second_iterator
