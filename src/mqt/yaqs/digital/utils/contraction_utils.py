# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""DAG-driven MPO update routines for equivalence checking.

Applies temporal zones from paired circuits to an MPO via local contractions, gate updates,
and SVD splits (see :mod:`mqt.yaqs.core.data_structures.mpo_utils` for primitive tensor ops).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe
from qiskit.converters import dag_to_circuit

from ...core.data_structures.mpo import MPO
from ...core.data_structures.mpo_utils import contract_mpo_site_with_mpo_site, decompose_theta
from ...core.parallel_utils import MPContext, available_cpus, limit_worker_threads
from .dag_utils import check_longest_gate, convert_dag_to_tensor_algorithm, get_temporal_zone, select_starting_point

# Below this width, thread-pool overhead usually beats the cost of one SVD pair update.
MIN_QUBITS_FOR_MPO_PARALLEL = 12
_MIN_PAIRS_PER_SWEEP_FOR_PARALLEL = 3

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.dagcircuit import DAGCircuit

    from ...core.libraries.gate_library import BaseGate


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
    assert theta.ndim == 6, f"Expected theta to have 6 dimensions, got {theta.ndim}"
    if conjugate:
        theta = np.transpose(theta, (3, 4, 2, 0, 1, 5))

    if gate.name == "I":
        pass  # Identity gate, no action needed.
    elif gate.interaction == 1:
        if gate.sites[0] == site0:
            if conjugate:
                theta = oe.contract("ij, jklmno->iklmno", np.conj(gate.matrix), theta)
            else:
                theta = oe.contract("ij, jklmno->iklmno", gate.matrix, theta)
        elif gate.sites[0] == site1:
            if conjugate:
                theta = oe.contract("ij, kjlmno->kilmno", np.conj(gate.matrix), theta)
            else:
                theta = oe.contract("ij, kjlmno->kilmno", gate.matrix, theta)
    elif gate.interaction == 2:
        if conjugate:
            theta = oe.contract("ijkl, klmnop->ijmnop", np.conj(gate.tensor), theta)
        else:
            theta = oe.contract("ijkl, klmnop->ijmnop", gate.tensor, theta)
    if conjugate:
        theta = np.transpose(theta, (3, 4, 2, 0, 1, 5))

    return theta


def apply_temporal_zone_gates(
    theta: NDArray[np.complex128],
    gates: list[BaseGate],
    qubits: list[int],
    *,
    conjugate: bool = False,
) -> NDArray[np.complex128]:
    """Apply a pre-extracted list of gates to a local tensor ``theta``.

    Args:
        theta: Local tensor to update.
        gates: Pre-extracted gates to apply in order.
        qubits: Local qubit indices; ``qubits[0]`` is used as the left site ``n``.
        conjugate: If True, apply each gate as its conjugate transpose.

    Returns:
        The updated local tensor after applying all gates.
    """
    n = qubits[0]
    for gate in gates:
        theta = apply_gate(gate, theta, n, n + 1, conjugate=conjugate)
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
        return apply_temporal_zone_gates(theta, tensor_circuit, qubits, conjugate=conjugate)
    return theta


def compute_pair_update(
    tensor_n: NDArray[np.complex128],
    tensor_n1: NDArray[np.complex128],
    gates1: list[BaseGate],
    gates2: list[BaseGate],
    threshold: float,
    qubits: list[int],
    *,
    apply_conjugate_on_second: bool,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Contract two site tensors, apply gate zones, and SVD-decompose back to two sites.

    Args:
        tensor_n: MPO tensor on the left site of the pair.
        tensor_n1: MPO tensor on the right site of the pair.
        gates1: Gates from the first circuit's temporal zone on this pair.
        gates2: Gates from the second circuit's temporal zone on this pair.
        threshold: SVD truncation threshold.
        qubits: The two site indices ``[n, n + 1]`` for gate placement.
        apply_conjugate_on_second: If True, apply ``gates2`` with conjugation.

    Returns:
        Updated ``(tensor_n, tensor_n1)`` after decomposition.
    """
    theta = oe.contract("abcd, efdg->aecbfg", tensor_n, tensor_n1)
    if gates1:
        theta = apply_temporal_zone_gates(theta, gates1, qubits, conjugate=False)
    if gates2:
        theta = apply_temporal_zone_gates(
            theta,
            gates2,
            qubits,
            conjugate=apply_conjugate_on_second,
        )
    return decompose_theta(theta, threshold)


@dataclass(frozen=True)
class _PairUpdateWork:
    """One checkerboard pair: gate zones and site index (serial zone extraction)."""

    site: int
    gates1: tuple[BaseGate, ...]
    gates2: tuple[BaseGate, ...]
    apply_conjugate_on_second: bool


@dataclass(frozen=True)
class _PairUpdateResult:
    """Updated MPO tensors for one checkerboard pair."""

    site: int
    tensor_n: NDArray[np.complex128]
    tensor_n1: NDArray[np.complex128]


def _gather_pair_update_work(
    dag1: DAGCircuit,
    dag2: DAGCircuit,
    pair_iterator: range,
) -> list[_PairUpdateWork]:
    """Extract temporal zones serially for each pair in a sweep.

    Returns:
        One work item per site index in ``pair_iterator``.
    """
    work_items: list[_PairUpdateWork] = []
    dag1_has_ops = bool(dag1.op_nodes())
    for n in pair_iterator:
        qubits = [n, n + 1]
        zone1 = get_temporal_zone(dag1, qubits) if dag1_has_ops else None
        zone2 = get_temporal_zone(dag2, qubits)
        gates1 = convert_dag_to_tensor_algorithm(zone1) if zone1 is not None and zone1.op_nodes() else []
        gates2 = convert_dag_to_tensor_algorithm(zone2) if zone2.op_nodes() else []
        work_items.append(
            _PairUpdateWork(
                site=n,
                gates1=tuple(gates1),
                gates2=tuple(gates2),
                apply_conjugate_on_second=bool(gates2),
            )
        )
    return work_items


def _compute_pair_work(
    mpo: MPO,
    work: _PairUpdateWork,
    threshold: float,
) -> _PairUpdateResult:
    """Run ``compute_pair_update`` for one checkerboard pair (thread-pool worker).

    Returns:
        Updated tensors for the pair's left site index.
    """
    n = work.site
    qubits = [n, n + 1]
    tensor_n, tensor_n1 = compute_pair_update(
        mpo.tensors[n],
        mpo.tensors[n + 1],
        list(work.gates1),
        list(work.gates2),
        threshold,
        qubits,
        apply_conjugate_on_second=work.apply_conjugate_on_second,
    )
    return _PairUpdateResult(site=n, tensor_n=tensor_n, tensor_n1=tensor_n1)


def _apply_pair_update_results(mpo: MPO, results: list[_PairUpdateResult]) -> None:
    for result in results:
        mpo.tensors[result.site] = result.tensor_n
        mpo.tensors[result.site + 1] = result.tensor_n1


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
    gates1: list[BaseGate] = []
    gates2: list[BaseGate] = []
    if dag1.op_nodes():
        zone1 = get_temporal_zone(dag1, qubits)
        gates1 = convert_dag_to_tensor_algorithm(zone1)
    if dag2.op_nodes():
        zone2 = get_temporal_zone(dag2, qubits)
        gates2 = convert_dag_to_tensor_algorithm(zone2)
    apply_conjugate = bool(gates2)
    mpo.tensors[n], mpo.tensors[n + 1] = compute_pair_update(
        mpo.tensors[n],
        mpo.tensors[n + 1],
        gates1,
        gates2,
        threshold,
        qubits,
        apply_conjugate_on_second=apply_conjugate,
    )


def _apply_layer_sweep(
    mpo: MPO,
    circuit1_dag: DAGCircuit,
    circuit2_dag: DAGCircuit,
    pair_iterator: range,
    threshold: float,
    *,
    parallel: bool,
    max_workers: int | None,
    mp_context: MPContext,
    thread_pool: ThreadPoolExecutor | None,
) -> None:
    _ = mp_context
    if not parallel or len(pair_iterator) < _MIN_PAIRS_PER_SWEEP_FOR_PARALLEL:
        for n in pair_iterator:
            update_mpo(mpo, circuit1_dag, circuit2_dag, [n, n + 1], threshold)
        return

    work_items = _gather_pair_update_work(circuit1_dag, circuit2_dag, pair_iterator)
    if thread_pool is None:
        msg = "parallel MPO sweeps require an active thread pool from iterate()."
        raise RuntimeError(msg)

    workers = max_workers if max_workers is not None else available_cpus()
    workers = max(1, min(workers, len(work_items)))

    def _run_one(work: _PairUpdateWork) -> _PairUpdateResult:
        return _compute_pair_work(mpo, work, threshold)

    if workers == 1:
        results = [_run_one(work) for work in work_items]
    else:
        chunks = thread_pool.map(_run_one, work_items)
        results = list(chunks)
    _apply_pair_update_results(mpo, results)


def apply_layer(
    mpo: MPO,
    circuit1_dag: DAGCircuit,
    circuit2_dag: DAGCircuit,
    first_iterator: range,
    second_iterator: range,
    threshold: float,
    *,
    parallel: bool = False,
    max_workers: int | None = None,
    mp_context: MPContext = "auto",
    thread_pool: ThreadPoolExecutor | None = None,
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
        parallel: If True, run disjoint pair tensor updates in a thread pool after serial zone extraction.
        max_workers: Worker thread count when ``parallel`` is True.
        mp_context: Reserved (thread pool is used for MPO parallelism).
        thread_pool: Shared thread pool created by :func:`iterate`.
    """
    _apply_layer_sweep(
        mpo,
        circuit1_dag,
        circuit2_dag,
        first_iterator,
        threshold,
        parallel=parallel,
        max_workers=max_workers,
        mp_context=mp_context,
        thread_pool=thread_pool,
    )
    _apply_layer_sweep(
        mpo,
        circuit1_dag,
        circuit2_dag,
        second_iterator,
        threshold,
        parallel=parallel,
        max_workers=max_workers,
        mp_context=mp_context,
        thread_pool=thread_pool,
    )


def apply_long_range_layer(mpo: MPO, dag1: DAGCircuit, dag2: DAGCircuit, threshold: float, *, conjugate: bool) -> None:
    """Detect and apply a long-range gate from the first layer of a DAGCircuit to an MPO.

    This function searches for a gate in the specified DAGCircuit (dag1 if not conjugate, else dag2)
    whose qubit distance exceeds 2, and then applies that gate to update the MPO.
    The process involves contracting neighboring MPO tensors, applying the long-range gate,
    and then decomposing the result back into MPO tensors via SVD-based truncation.

    Args:
        mpo: The MPO object being updated.
        dag1: The first circuit's DAGCircuit.
        dag2: The second circuit's DAGCircuit.
        threshold: The SVD threshold for truncation.
        conjugate: If True, apply the gate from dag2 in conjugated form; otherwise, from dag1.
    """
    dag_to_search = dag1 if not conjugate else dag2

    first_layer = next(dag_to_search.layers(), None)
    gate_mpo = None
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
                    gate_ = convert_dag_to_tensor_algorithm(node)[0]
                    gate_mpo = MPO.from_gate(gate_, distance)
                    if conjugate:
                        gate_mpo.rotate(conjugate=True)
                    dag.remove_op_node(node)
                    break
            break

    assert gate_mpo is not None, "Long-range gate MPO not found."
    assert gate_mpo.length <= mpo.length

    if gate_mpo.length == mpo.length:
        sites = range(mpo.length)
    else:
        assert location is not None
        assert distance is not None
        sites = range(location, location + distance)

    # Process even-indexed sites from the gate MPO
    for site_gate_mpo, overall_site in enumerate(sites):
        if site_gate_mpo != len(sites) - 1 and site_gate_mpo % 2 == 0:
            if not conjugate:
                tensor1 = np.transpose(gate_mpo.tensors[site_gate_mpo], (0, 2, 1, 3))
                tensor2 = np.transpose(gate_mpo.tensors[site_gate_mpo + 1], (0, 2, 1, 3))
                tensor3 = np.transpose(mpo.tensors[overall_site], (0, 2, 1, 3))
                tensor4 = np.transpose(mpo.tensors[overall_site + 1], (0, 2, 1, 3))
                theta = oe.contract("abcd,edfg,chij,fjkl->aebhikgl", tensor1, tensor2, tensor3, tensor4)
            else:
                mpo.rotate()
                tensor1 = np.transpose(gate_mpo.tensors[site_gate_mpo], (0, 2, 1, 3))
                tensor2 = np.transpose(gate_mpo.tensors[site_gate_mpo + 1], (0, 2, 1, 3))
                tensor3 = np.transpose(mpo.tensors[overall_site], (0, 2, 1, 3))
                tensor4 = np.transpose(mpo.tensors[overall_site + 1], (0, 2, 1, 3))
                theta = oe.contract("abcd,edfg,chij,fjkl->ikhbaelg", tensor1, tensor2, tensor3, tensor4)
                mpo.rotate()

            dims = theta.shape
            theta = np.reshape(theta, (dims[0], dims[1], dims[2] * dims[3], dims[4], dims[5], dims[6] * dims[7]))
            theta = apply_temporal_zone(theta, dag1, [overall_site, overall_site + 1], conjugate=False)
            theta = apply_temporal_zone(theta, dag2, [overall_site, overall_site + 1], conjugate=True)
            mpo.tensors[overall_site], mpo.tensors[overall_site + 1] = decompose_theta(theta, threshold)

            gate_mpo.tensors[site_gate_mpo] = None  # ty: ignore[invalid-assignment]
            gate_mpo.tensors[site_gate_mpo + 1] = None  # ty: ignore[invalid-assignment]

        # Process odd-indexed (or hanging) tensor if present.
        if site_gate_mpo == len(sites) - 1 and any(isinstance(tensor, np.ndarray) for tensor in gate_mpo.tensors):
            if conjugate:
                mpo.rotate()
            theta = contract_mpo_site_with_mpo_site(
                gate_mpo.tensors[site_gate_mpo],
                mpo.tensors[overall_site],
                conjugate=conjugate,
            )
            if conjugate:
                mpo.rotate()
            theta = np.transpose(theta, (0, 2, 1, 3))

            tensor1 = np.transpose(mpo.tensors[overall_site - 1], (0, 2, 1, 3))
            theta = oe.contract("abcd, edfg->aebcfg", tensor1, theta)

            theta = apply_temporal_zone(theta, dag1, [overall_site - 1, overall_site], conjugate=False)
            theta = apply_temporal_zone(theta, dag2, [overall_site - 1, overall_site], conjugate=True)

            mpo.tensors[overall_site - 1], mpo.tensors[overall_site] = decompose_theta(theta, threshold)
            gate_mpo.tensors[site_gate_mpo] = None  # ty: ignore[invalid-assignment]

    assert not any(isinstance(tensor, np.ndarray) for tensor in gate_mpo.tensors), "Not all gate tensors were applied."


def iterate(
    mpo: MPO,
    dag1: DAGCircuit,
    dag2: DAGCircuit,
    threshold: float,
    *,
    parallel: bool = False,
    max_workers: int | None = None,
    mp_context: MPContext = "auto",
) -> None:
    """Iteratively apply layers of gates from two DAGCircuits to an MPO until no gates remain.

    The function selects starting qubit ranges based on the available operations in dag1 or dag2.
    In each iteration, it checks the maximum gate distance. If all gates are nearest-neighbor (distance 1 or 2),
    a standard layer update is applied; otherwise, a specialized long-range update is performed.

    Args:
        mpo (MPO): The MPO object to update.
        dag1 (DAGCircuit): The first circuit's DAGCircuit.
        dag2 (DAGCircuit): The second circuit's DAGCircuit.
        threshold (float): The SVD truncation threshold used during decomposition.
        parallel: If True, parallelize tensor updates within checkerboard sweeps.
        max_workers: Worker process count when ``parallel`` is True.
        mp_context: Multiprocessing context for the process pool.
    """
    length = mpo.length

    if dag1.op_nodes():
        first_iterator, second_iterator = select_starting_point(length, dag1)
    else:
        first_iterator, second_iterator = select_starting_point(length, dag2)

    def _consume_dags(thread_pool: ThreadPoolExecutor | None) -> None:
        layer_parallel = parallel and thread_pool is not None
        while dag1.op_nodes() or dag2.op_nodes():
            largest_distance1 = check_longest_gate(dag1)
            largest_distance2 = check_longest_gate(dag2)
            if largest_distance1 in {1, 2} and largest_distance2 in {1, 2}:
                apply_layer(
                    mpo,
                    dag1,
                    dag2,
                    first_iterator,
                    second_iterator,
                    threshold,
                    parallel=layer_parallel,
                    max_workers=max_workers,
                    mp_context=mp_context,
                    thread_pool=thread_pool,
                )
            else:
                conjugate = largest_distance2 > largest_distance1
                apply_long_range_layer(mpo, dag1, dag2, threshold, conjugate=conjugate)

    use_parallel = parallel and length >= MIN_QUBITS_FOR_MPO_PARALLEL
    if not use_parallel:
        _consume_dags(None)
        return

    limit_worker_threads(1)
    workers = max_workers if max_workers is not None else available_cpus()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        _consume_dags(pool)
