# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Regression: small-angle benchmark local-generator path never enriches LR Pauli gates."""

from __future__ import annotations

from scripts.benchmark_small_angle_tdvp_vs_tebd import (
    _apply_two_qubit_local_generator_tdvp,
    _ising_2d_circuit,
    _make_case,
    _run_circuit,
)


def test_local_generator_tdvp_never_enriches_lr_pauli() -> None:
    qc, edge_types = _ising_2d_circuit(lx=3, ly=3, j=1.0, h=0.5, dt=0.005, layers=2)
    case = _make_case(
        family="ising_2d",
        model="ising",
        geometry="2d_row_major",
        circuit_name="test_ising3x3",
        qc=qc,
        edge_types=edge_types,
        lx=3,
        ly=3,
        initial_state="plus",
        dt=0.005,
        layers=2,
    )
    mps, _, _, _, routes, status = _run_circuit(
        case,
        method="local_generator_tdvp",
        svd_threshold=1e-9,
        max_bond_dim=32,
    )
    assert status == "ok"
    assert mps is not None
    assert routes["enriched_lr"] == 0
    assert routes["tdvp_lr"] > 0


def test_apply_local_generator_routes_lr_rzz_to_tdvp() -> None:
    from qiskit.circuit import QuantumCircuit
    from qiskit.converters import circuit_to_dag

    from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
    from mqt.yaqs.core.data_structures.state import State

    qc = QuantumCircuit(4)
    qc.rzz(0.01, 0, 3)
    dag = circuit_to_dag(qc)
    node = next(n for n in dag.topological_op_nodes() if n.op.name == "rzz")

    mps = State(4, initial="zeros", representation="mps").mps
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        svd_threshold=1e-9,
        max_bond_dim=32,
        gate_mode="hybrid",
    )
    swaps, tdvp_lr, enriched_lr = _apply_two_qubit_local_generator_tdvp(mps, node, params)
    assert swaps == 0
    assert tdvp_lr == 1
    assert enriched_lr == 0
