# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for parallel equivalence checking (serial vs parallel agreement)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from mqt.yaqs import EquivalenceChecker
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm, get_temporal_zone, select_starting_point
from mqt.yaqs.digital.utils.equivalence_schedule import gates_have_disjoint_sites, partition_disjoint_gate_batches
from mqt.yaqs.digital.utils.mpo_utils import (
    MIN_QUBITS_FOR_MPO_PARALLEL,
    apply_layer,
    compute_pair_update,
    update_mpo,
)


def _make_n_by_n_circuit(num_qubits: int) -> QuantumCircuit:
    """Build an ``n`` x ``n`` layered circuit (``n`` qubits, ``n`` repetitions).

    Returns:
        A layered circuit with all-qubit ``h`` gates and linear ``cx`` chains.
    """
    qc = QuantumCircuit(num_qubits)
    for _ in range(num_qubits):
        for q in range(num_qubits):
            qc.h(q)
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
    return qc


def test_gates_have_disjoint_sites() -> None:
    """Disjoint-site predicate matches partition batching expectations."""
    x0 = GateLibrary.x()
    x0.set_sites(0)
    x1 = GateLibrary.x()
    x1.set_sites(1)
    cx = GateLibrary.cx()
    cx.set_sites(0, 1)

    assert gates_have_disjoint_sites(x0, x1) is True
    assert gates_have_disjoint_sites(x0, cx) is False


def test_partition_disjoint_gate_batches() -> None:
    """Disjoint single-qubit gates belong to one batch; overlapping gates do not."""
    x0 = GateLibrary.x()
    x0.set_sites(0)
    x1 = GateLibrary.x()
    x1.set_sites(1)
    cx = GateLibrary.cx()
    cx.set_sites(0, 1)

    batches = partition_disjoint_gate_batches([x0, x1])
    assert len(batches) == 1
    assert len(batches[0]) == 2

    batches_overlap = partition_disjoint_gate_batches([x0, cx])
    assert len(batches_overlap) == 2


def test_compute_pair_update_matches_update_mpo_step() -> None:
    """Pure pair kernel reproduces a single update_mpo step on two sites."""
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)
    qc2 = qc1.copy()

    mpo_ref = MPO.identity(2)
    dag1 = circuit_to_dag(qc1)
    dag2 = circuit_to_dag(qc2)
    update_mpo(mpo_ref, dag1, dag2, [0, 1], 1e-12)

    mpo_test = MPO.identity(2)

    dag1b = circuit_to_dag(qc1)
    dag2b = circuit_to_dag(qc2)
    zone1 = get_temporal_zone(dag1b, [0, 1])
    zone2 = get_temporal_zone(dag2b, [0, 1])
    gates1 = convert_dag_to_tensor_algorithm(zone1)
    gates2 = convert_dag_to_tensor_algorithm(zone2)
    t0, t1 = compute_pair_update(
        mpo_test.tensors[0],
        mpo_test.tensors[1],
        gates1,
        gates2,
        1e-12,
        [0, 1],
        apply_conjugate_on_second=True,
    )
    mpo_test.tensors[0] = t0
    mpo_test.tensors[1] = t1

    for a, b in zip(mpo_ref.tensors, mpo_test.tensors, strict=True):
        assert np.allclose(a, b, atol=1e-10)


@pytest.mark.parametrize("parallel", [False, True])
def test_mpo_checker_serial_vs_parallel_small(*, parallel: bool) -> None:
    """MPO equivalence on small circuits (serial path even when parallel=True)."""
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)

    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.cx(0, 1)

    checker = EquivalenceChecker(representation="mpo", parallel=parallel, max_workers=2)
    result = checker.check(qc1, qc2)
    assert result["equivalent"] is True


@pytest.mark.parametrize("num_qubits", [MIN_QUBITS_FOR_MPO_PARALLEL, MIN_QUBITS_FOR_MPO_PARALLEL + 2])
def test_wide_mpo_serial_vs_parallel_equivalent(num_qubits: int) -> None:
    """Wide ``n`` x ``n`` circuits agree between serial and parallel MPO checking."""
    qc = _make_n_by_n_circuit(num_qubits)
    serial = EquivalenceChecker(representation="mpo", parallel=False, threshold=1e-6).check(qc, qc)
    parallel = EquivalenceChecker(
        representation="mpo",
        parallel=True,
        max_workers=2,
        threshold=1e-6,
    ).check(qc, qc)

    assert serial["equivalent"] is True
    assert parallel["equivalent"] is True
    assert serial["equivalent"] == parallel["equivalent"]


def test_wide_mpo_serial_vs_parallel_non_equivalent() -> None:
    """Serial and parallel MPO paths agree on non-equivalent wide circuits."""
    num_qubits = MIN_QUBITS_FOR_MPO_PARALLEL
    qc1 = _make_n_by_n_circuit(num_qubits)
    qc2 = qc1.copy()
    qc2.x(0)

    serial = EquivalenceChecker(representation="mpo", parallel=False, threshold=1e-6).check(qc1, qc2)
    parallel = EquivalenceChecker(
        representation="mpo",
        parallel=True,
        max_workers=2,
        threshold=1e-6,
    ).check(qc1, qc2)

    assert serial["equivalent"] is False
    assert serial["equivalent"] == parallel["equivalent"]


def test_mpo_parallel_max_workers_one_uses_in_process_path() -> None:
    """``max_workers=1`` still runs through the parallel sweep with a thread pool."""
    num_qubits = MIN_QUBITS_FOR_MPO_PARALLEL
    qc = _make_n_by_n_circuit(num_qubits)
    result = EquivalenceChecker(
        representation="mpo",
        parallel=True,
        max_workers=1,
        threshold=1e-6,
    ).check(qc, qc)
    assert result["equivalent"] is True


def test_apply_layer_parallel_requires_thread_pool() -> None:
    """Parallel layer updates without a thread pool raise at runtime."""
    num_qubits = MIN_QUBITS_FOR_MPO_PARALLEL
    qc = _make_n_by_n_circuit(num_qubits)
    dag1 = circuit_to_dag(qc)
    dag2 = circuit_to_dag(qc)
    mpo = MPO.identity(num_qubits)
    first_iterator, second_iterator = select_starting_point(num_qubits, dag1)

    with pytest.raises(RuntimeError, match="thread pool"):
        apply_layer(
            mpo,
            dag1,
            dag2,
            first_iterator,
            second_iterator,
            1e-6,
            parallel=True,
            max_workers=2,
            thread_pool=None,
        )


def test_apply_layer_parallel_with_thread_pool() -> None:
    """Parallel apply_layer succeeds when a thread pool is provided."""
    num_qubits = MIN_QUBITS_FOR_MPO_PARALLEL
    qc = _make_n_by_n_circuit(num_qubits)
    dag1 = circuit_to_dag(qc)
    dag2 = circuit_to_dag(qc)
    mpo_serial = MPO.identity(num_qubits)
    mpo_parallel = MPO.identity(num_qubits)
    first_iterator, second_iterator = select_starting_point(num_qubits, dag1)

    apply_layer(
        mpo_serial,
        dag1,
        dag2,
        first_iterator,
        second_iterator,
        1e-6,
        parallel=False,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        apply_layer(
            mpo_parallel,
            dag1,
            dag2,
            first_iterator,
            second_iterator,
            1e-6,
            parallel=True,
            max_workers=2,
            thread_pool=pool,
        )

    for serial_tensor, parallel_tensor in zip(mpo_serial.tensors, mpo_parallel.tensors, strict=True):
        assert np.allclose(serial_tensor, parallel_tensor, atol=1e-8)


def test_matrix_checker_equivalent_circuits() -> None:
    """Matrix backend reports equivalent circuits."""
    qc1 = QuantumCircuit(3)
    qc1.h(0)
    qc1.cx(0, 1)
    qc1.rz(np.pi / 4, 2)
    qc2 = qc1.copy()

    result = EquivalenceChecker(representation="matrix").check(qc1, qc2)
    assert result["equivalent"] is True
    assert result["representation"] == "matrix"


def test_long_range_mpo_parallel() -> None:
    """Long-range circuits agree between serial and parallel MPO checking."""
    qc1 = QuantumCircuit(3)
    qc1.h(0)
    qc1.cx(0, 2)

    qc2 = qc1.copy()

    serial = EquivalenceChecker(representation="mpo", parallel=False).check(qc1, qc2)
    parallel = EquivalenceChecker(representation="mpo", parallel=True, max_workers=2).check(qc1, qc2)
    assert serial["equivalent"] == parallel["equivalent"]
