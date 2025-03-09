# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility functions for MPO-based equivalence checking.

This module provides functions to manipulate tensor network representations of quantum operations.
It includes routines for performing SVD-based decompositions (decompose_theta), applying gates to local tensors
(apply_gate), extracting and applying temporal zones from DAGCircuits (apply_temporal_zone), and updating Matrix
Product Operators (MPO) via layer-wise and long-range gate applications. These utilities facilitate the
conversion of quantum circuits into tensor network algorithms for checking equivalence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe
from qiskit.converters import dag_to_circuit

from .dag_utils import check_longest_gate, convert_dag_to_tensor_algorithm, get_temporal_zone, select_starting_point

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.dagcircuit import DAGCircuit

    from ...core.data_structures.networks import MPO
    from ...core.libraries.gate_library import BaseGate


def decompose_theta(
    theta: NDArray[np.complex128], threshold: float
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Decompose theta during equivalence checking.

    Perform an SVD-based decomposition of the tensor `theta`, truncating singular values below a given threshold,
    and reshape the result into two rank-4 tensors.

    The input tensor is first re-ordered and flattened into a matrix, then decomposed using SVD.
    Singular values below the threshold are discarded. The left singular vectors (U) are reshaped into a tensor
    with shape (dims[0], dims[1], dims[2], num_sv), and the product of the diagonal singular value matrix with
    the right singular vectors (V) is reshaped and transposed into a tensor with shape
    (dims[3], dims[4], num_sv, dims[5]).

    Args:
        theta (NDArray[np.complex128]): The high-rank tensor to decompose.
        threshold (float): The cutoff threshold for singular values.

    Returns:
        tuple[NDArray[np.complex128], NDArray[np.complex128]]:
            A tuple (U, M) of two reshaped tensors derived from the SVD.
    """
    dims = theta.shape
    # Reorder indices before flattening
    theta = np.transpose(theta, (0, 3, 2, 1, 4, 5))
    theta_matrix = np.reshape(theta, (dims[0] * dims[1] * dims[2], dims[3] * dims[4] * dims[5]))

    U, S_list, V = np.linalg.svd(theta_matrix, full_matrices=False)
    S_list = S_list[S_list > threshold]
    U = U[:, : len(S_list)]
    V = V[: len(S_list), :]

    # Reshape U into a tensor of shape (dims[0], dims[1], dims[2], num_sv)
    U = np.reshape(U, (dims[0], dims[1], dims[2], len(S_list)))

    # Compute site tensor M and reshape: first form M = diag(S_list) @ V,
    # then reshape to (num_sv, dims[3], dims[4], dims[5]) and transpose to (dims[3], dims[4], num_sv, dims[5])
    M = np.diag(S_list) @ V
    M = np.reshape(M, (len(S_list), dims[3], dims[4], dims[5]))
    M = np.transpose(M, (1, 2, 0, 3))

    return U, M


def apply_gate(
    gate: BaseGate,
    theta: NDArray[np.complex128],
    site0: int,
    site1: int,
    *,
    conjugate: bool = False,
) -> NDArray[np.complex128]:
    """Apply a single-, two-, or multi-qubit gate from a GateLibrary object to a local tensor `theta`.

    Depending on the gate's interaction type and the dimensionality of `theta`, this function contracts the gate's
    tensor with `theta` according to a predefined pattern. If `conjugate` is True, the gate tensor is conjugated before
    contraction. Identity gates leave `theta` unchanged.

    Args:
        gate (BaseGate): A gate object from the GateLibrary that contains .tensor, .interaction, and .sites attributes.
        theta (NDArray[np.complex128]): The local tensor to update.
        site0 (int): The first qubit (site) index.
        site1 (int): The second qubit (site) index.
        conjugate (bool, optional): Whether to apply the conjugated version of the gate tensor. Defaults to False.

    Returns:
        NDArray[np.complex128]: The updated local tensor after applying the gate.
    """
    # Check gate site usage
    assert gate.interaction in {1, 2}, "Gate interaction must be 1 or 2."

    if gate.interaction == 1:
        assert gate.sites[0] in {site0, site1}, "Single-qubit gate must be on one of the sites."
    elif gate.interaction == 2:
        assert gate.sites[0] in {site0, site1}, "Two-qubit gate must be on the correct pair of sites."
        assert gate.sites[1] in {site0, site1}, "Two-qubit gate must be on the correct pair of sites."

    # For nearest-neighbor gates (theta.ndim == 6)
    assert theta.ndim == 6
    if conjugate:
        theta = np.transpose(theta, (3, 4, 2, 0, 1, 5))

    if gate.name == "I":
        pass  # Identity gate, no action needed.
    elif gate.interaction == 1:
        if gate.sites[0] == site0:
            if conjugate:
                theta = oe.contract("ij, jklmno->iklmno", np.conj(gate.tensor), theta)
            else:
                theta = oe.contract("ij, jklmno->iklmno", gate.tensor, theta)
        elif gate.sites[0] == site1:
            if conjugate:
                theta = oe.contract("ij, kjlmno->kilmno", np.conj(gate.tensor), theta)
            else:
                theta = oe.contract("ij, kjlmno->kilmno", gate.tensor, theta)
    elif gate.interaction == 2:
        if conjugate:
            theta = oe.contract("ijkl, klmnop->ijmnop", np.conj(gate.tensor), theta)
        else:
            theta = oe.contract("ijkl, klmnop->ijmnop", gate.tensor, theta)
    if conjugate:
        theta = np.transpose(theta, (3, 4, 2, 0, 1, 5))

    return theta


def apply_temporal_zone(
    theta: NDArray[np.complex128],
    dag: DAGCircuit,
    qubits: list[int],
    *,
    conjugate: bool = False,
) -> NDArray[np.complex128]:
    """Apply the temporal zone of a DAGCircuit to a local tensor `theta`.

    The temporal zone is the subset of operations extracted from the DAGCircuit that act on the specified qubits.
    This function uses the temporal zone to create a sequence of gate operations (via the GateLibrary) and applies
    them sequentially to `theta`. If conjugate is True, the gates are applied in their conjugated form.

    Args:
        theta (NDArray[np.complex128]): The local tensor to update.
        dag (DAGCircuit): The DAGCircuit from which to extract the temporal zone.
        qubits (list[int]): The qubit indices on which to apply the temporal zone (typically two neighboring qubits).
        conjugate (bool, optional): Whether to apply the gates in conjugated form. Defaults to False.

    Returns:
        NDArray[np.complex128]: The updated tensor after applying the temporal zone.
    """
    n = qubits[0]
    if dag.op_nodes():
        temporal_zone = get_temporal_zone(dag, [n, n + 1])
        tensor_circuit = convert_dag_to_tensor_algorithm(temporal_zone)

        for gate in tensor_circuit:
            theta = apply_gate(gate, theta, n, n + 1, conjugate=conjugate)
    return theta


def update_mpo(mpo: MPO, dag1: DAGCircuit, dag2: DAGCircuit, qubits: list[int], threshold: float) -> None:
    """Update two neighboring MPO tensors by applying gates extracted from two DAGCircuits.

    The function first contracts the two neighboring MPO tensors to form a combined tensor.
    It then applies gate operations from dag1 and dag2 (with appropriate conjugation) via the temporal zone,
    and finally decomposes the updated tensor back into two MPO tensors using an SVD-based truncation.

    Args:
        mpo (MPO): The MPO object whose tensors will be updated.
        dag1 (DAGCircuit): A DAGCircuit containing gates (from the left) to apply.
        dag2 (DAGCircuit): A DAGCircuit containing gates (from the right) to apply.
        qubits (list[int]): List of qubit indices (e.g. [n, n+1]) on which to apply the gates.
        threshold (float): The SVD threshold for truncation.
    """
    n = qubits[0]
    # Contract two neighboring MPO tensors
    theta = oe.contract("abcd, efdg->aecbfg", mpo.tensors[n], mpo.tensors[n + 1])

    # Apply gates from dag1 (G) and dag2 (G')
    if dag1:
        theta = apply_temporal_zone(theta, dag1, qubits, conjugate=False)
    if dag2:
        # When both dag1 and dag2 are provided, use conjugation on dag2's gates
        if dag1 is None:
            theta = apply_temporal_zone(theta, dag2, qubits, conjugate=False)
        else:
            theta = apply_temporal_zone(theta, dag2, qubits, conjugate=True)

    # Decompose the tensor back into two MPO tensors
    mpo.tensors[n], mpo.tensors[n + 1] = decompose_theta(theta, threshold)


def apply_layer(
    mpo: MPO,
    circuit1_dag: DAGCircuit,
    circuit2_dag: DAGCircuit,
    first_iterator: range,
    second_iterator: range,
    threshold: float,
) -> None:
    """Apply a complete layer of gate updates to an MPO in two sweeps.

    The layer is applied by updating MPO tensors on qubit pairs defined by the first_iterator and second_iterator.
    For each pair, the function calls update_mpo to apply the corresponding gates and perform SVD-based truncation.

    Args:
        mpo (MPO): The MPO object to update.
        circuit1_dag (DAGCircuit): The first circuit's DAGCircuit representation.
        circuit2_dag (DAGCircuit): The second circuit's DAGCircuit representation.
        first_iterator (range): Range of starting qubit indices for the first sweep.
        second_iterator (range): Range of starting qubit indices for the second sweep.
        threshold (float): The SVD truncation threshold.
    """
    for n in first_iterator:
        update_mpo(mpo, circuit1_dag, circuit2_dag, [n, n + 1], threshold)

    for n in second_iterator:
        update_mpo(mpo, circuit1_dag, circuit2_dag, [n, n + 1], threshold)


def apply_long_range_layer(mpo: MPO, dag1: DAGCircuit, dag2: DAGCircuit, threshold: float, *, conjugate: bool) -> None:
    """Detect and apply a long-range gate from the first layer of a DAGCircuit to an MPO.

    This function searches for a gate in the specified DAGCircuit (dag1 if not conjugate, else dag2)
    whose qubit distance exceeds 2, and then applies that gate to update the MPO.
    The process involves contracting neighboring MPO tensors, applying the long-range gate,
    and then decomposing the result back into MPO tensors via SVD-based truncation.

    Args:
        mpo (MPO): The MPO object being updated.
        dag1 (DAGCircuit): The first circuit's DAGCircuit.
        dag2 (DAGCircuit): The second circuit's DAGCircuit.
        threshold (float): The SVD threshold for truncation.
        conjugate (bool): If True, apply the gate from dag2 in conjugated form; otherwise, from dag1.
    """
    dag_to_search = dag1 if not conjugate else dag2

    first_layer = next(dag_to_search.layers(), None)
    gate_MPO = None
    distance = None
    location = None
    if first_layer is not None:
        layer_circuit = dag_to_circuit(first_layer["graph"])
        for gate in layer_circuit.data:
            if gate.operation.num_qubits <= 1:
                continue

            distance = np.abs(gate.qubits[0]._index - gate.qubits[-1]._index) + 1  # noqa: SLF001
            if distance <= 2:
                continue

            location = min(gate.qubits[0]._index, gate.qubits[-1]._index)  # noqa: SLF001

            dag = dag2 if conjugate else dag1

            for node in dag.op_nodes():
                if (
                    node.name == gate.operation.name
                    and len(node.qargs) >= 2
                    and node.qargs[0]._index == gate.qubits[0]._index  # noqa: SLF001
                    and node.qargs[1]._index == gate.qubits[1]._index  # noqa: SLF001
                ):
                    gate_MPO = convert_dag_to_tensor_algorithm(node)[0].mpo
                    if conjugate:
                        gate_MPO.rotate(conjugate=True)
                    dag.remove_op_node(node)
                    break
            break

    assert gate_MPO is not None, "Long-range gate MPO not found."
    assert gate_MPO.length <= mpo.length

    if gate_MPO.length == mpo.length:
        sites = range(mpo.length)
    elif location is not None and distance is not None:
        sites = range(location, location + distance)

    # Process even-indexed sites from the gate MPO
    for site_gate_MPO, overall_site in enumerate(sites):
        if site_gate_MPO != len(sites) - 1 and site_gate_MPO % 2 == 0:
            if not conjugate:
                tensor1 = np.transpose(gate_MPO.tensors[site_gate_MPO], (0, 2, 1, 3))
                tensor2 = np.transpose(gate_MPO.tensors[site_gate_MPO + 1], (0, 2, 1, 3))
                tensor3 = np.transpose(mpo.tensors[overall_site], (0, 2, 1, 3))
                tensor4 = np.transpose(mpo.tensors[overall_site + 1], (0, 2, 1, 3))
                theta = oe.contract("abcd,edfg,chij,fjkl->aebhikgl", tensor1, tensor2, tensor3, tensor4)
            else:
                mpo.rotate()
                tensor1 = np.transpose(gate_MPO.tensors[site_gate_MPO], (0, 2, 1, 3))
                tensor2 = np.transpose(gate_MPO.tensors[site_gate_MPO + 1], (0, 2, 1, 3))
                tensor3 = np.transpose(mpo.tensors[overall_site], (0, 2, 1, 3))
                tensor4 = np.transpose(mpo.tensors[overall_site + 1], (0, 2, 1, 3))
                theta = oe.contract("abcd,edfg,chij,fjkl->aebhikgl", tensor1, tensor2, tensor3, tensor4)
                mpo.rotate()

            theta = apply_temporal_zone(theta, dag1, [overall_site, overall_site + 1], conjugate=False)
            theta = apply_temporal_zone(theta, dag2, [overall_site, overall_site + 1], conjugate=True)

            dims = theta.shape
            theta = np.reshape(theta, (dims[0], dims[1], dims[2] * dims[3], dims[4], dims[5], dims[6] * dims[7]))
            mpo.tensors[overall_site], mpo.tensors[overall_site + 1] = decompose_theta(theta, threshold)

            gate_MPO.tensors[site_gate_MPO] = None
            gate_MPO.tensors[site_gate_MPO + 1] = None

        # Process odd-indexed (or hanging) tensor if present.
        if site_gate_MPO == len(sites) - 1 and any(isinstance(tensor, np.ndarray) for tensor in gate_MPO.tensors):
            if not conjugate:
                tensor1 = np.transpose(gate_MPO.tensors[site_gate_MPO], (0, 2, 1, 3))
                tensor2 = np.transpose(mpo.tensors[overall_site], (0, 2, 1, 3))
                theta = oe.contract("abcd,cefg->abefdg", tensor1, tensor2)
            else:
                mpo.rotate()
                tensor1 = np.transpose(gate_MPO.tensors[site_gate_MPO], (0, 2, 1, 3))
                tensor2 = np.transpose(mpo.tensors[overall_site], (0, 2, 1, 3))
                theta = oe.contract("abcd,cefg->febagd", tensor1, tensor2)
                mpo.rotate()

            dims = theta.shape
            theta = np.reshape(theta, (dims[0], dims[1] * dims[2], dims[3], dims[4] * dims[5]))

            tensor1 = np.transpose(mpo.tensors[overall_site - 1], (0, 2, 1, 3))
            theta = oe.contract("abcd, edfg->aebcfg", tensor1, theta)

            theta = apply_temporal_zone(theta, dag1, [overall_site - 1, overall_site], conjugate=False)
            theta = apply_temporal_zone(theta, dag2, [overall_site - 1, overall_site], conjugate=True)

            mpo.tensors[overall_site - 1], mpo.tensors[overall_site] = decompose_theta(theta, threshold)
            gate_MPO.tensors[site_gate_MPO] = None

    assert not any(isinstance(tensor, np.ndarray) for tensor in gate_MPO.tensors), "Not all gate tensors were applied."


def iterate(mpo: MPO, dag1: DAGCircuit, dag2: DAGCircuit, threshold: float) -> None:
    """Iteratively apply layers of gates from two DAGCircuits to an MPO until no gates remain.

    The function selects starting qubit ranges based on the available operations in dag1 or dag2.
    In each iteration, it checks the maximum gate distance. If all gates are nearest-neighbor (distance 1 or 2),
    a standard layer update is applied; otherwise, a specialized long-range update is performed.

    Args:
        mpo (MPO): The MPO object to update.
        dag1 (DAGCircuit): The first circuit's DAGCircuit.
        dag2 (DAGCircuit): The second circuit's DAGCircuit.
        threshold (float): The SVD truncation threshold used during decomposition.
    """
    N = mpo.length

    if dag1.op_nodes():
        first_iterator, second_iterator = select_starting_point(N, dag1)
    else:
        first_iterator, second_iterator = select_starting_point(N, dag2)

    while dag1.op_nodes() or dag2.op_nodes():
        largest_distance1 = check_longest_gate(dag1)
        largest_distance2 = check_longest_gate(dag2)
        # If all gates are nearest-neighbor (distance <= 2), apply the standard layer.
        if largest_distance1 in {1, 2} and largest_distance2 in {1, 2}:
            apply_layer(mpo, dag1, dag2, first_iterator, second_iterator, threshold)
        else:
            # For longer-range gates, decide which DAG to use based on gate distance.
            conjugate = largest_distance2 > largest_distance1
            apply_long_range_layer(mpo, dag1, dag2, threshold, conjugate=conjugate)
