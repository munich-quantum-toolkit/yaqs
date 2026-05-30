#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Calibrate TDVP visibility safety threshold against Qiskit.

Run:

    uv run python -m scripts.calibrate_tdvp_visibility_threshold

This is an internal calibration script (not a public mode comparison API).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector

from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.digital.digital_tjm import (
    apply_pauli_product_rotation_enriched,
    apply_single_qubit_gate,
    apply_two_qubit_gate_tdvp,
    estimate_local_tdvp_projected_norm,
)
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm


@dataclass(frozen=True)
class StepOutcome:
    name: str
    fidelity_error: float


def _fid_err(vec: np.ndarray, ref: np.ndarray) -> float:
    return float(1.0 - abs(np.vdot(ref, vec)) ** 2)


def _mps_vec(state: State) -> np.ndarray:
    return np.asarray(state.mps.to_vec(), dtype=np.complex128)


def _apply_node_inplace(state: State, node, params: StrongSimParams, *, route: str) -> None:
    if len(node.qargs) == 1:
        apply_single_qubit_gate(state.mps, node)
        return
    if len(node.qargs) != 2:
        raise ValueError(f"Unsupported op with {len(node.qargs)} qubits: {node.op.name}")

    gate = convert_dag_to_tensor_algorithm(node)[0]
    if gate.name in {"rxx", "ryy", "rzz"} and route in {"tdvp", "pauli_enriched"} and abs(gate.sites[0] - gate.sites[1]) != 1:
        if route == "tdvp":
            apply_two_qubit_gate_tdvp(state.mps, gate, params)
        else:
            apply_pauli_product_rotation_enriched(state.mps, gate, params)
        return

    # Default behavior for non-target gates: use TEBD/TDVP hybrid via tdvp (safe for 1q; NN 2q isn't used here).
    apply_two_qubit_gate_tdvp(state.mps, gate, params)


def _prefix_statevector(qc_prefix: QuantumCircuit) -> np.ndarray:
    return np.asarray(Statevector(qc_prefix).data, dtype=np.complex128)


def run_circuit(name: str, qc: QuantumCircuit, *, safety_tol: float) -> None:
    print(f"\n=== {name} ===")
    params = StrongSimParams(
        preset="exact",
        gate_mode="hybrid",
        svd_threshold=1e-14,
        tangent_blindness_tol=1e-12,
        tdvp_visibility_safety_tol=safety_tol,
        tdvp_pauli_consistency_check=True,
        tdvp_pauli_consistency_tol=1e-10,
    )

    dag = circuit_to_dag(qc)
    qc_prefix = QuantumCircuit(qc.num_qubits)

    st_adapt = State(qc.num_qubits, initial="zeros", representation="mps")
    st_tdvp = State(qc.num_qubits, initial="zeros", representation="mps")
    st_enr = State(qc.num_qubits, initial="zeros", representation="mps")
    enriched_seen = False

    for node in dag.topological_op_nodes():
        qc_prefix.append(node.op, node.qargs, node.cargs)
        ref = _prefix_statevector(qc_prefix)

        if len(node.qargs) == 2:
            gate = convert_dag_to_tensor_algorithm(node)[0]
            is_lr_pauli = gate.name in {"rxx", "ryy", "rzz"} and abs(gate.sites[0] - gate.sites[1]) != 1
        else:
            is_lr_pauli = False

        # Adaptive route decision on current adaptive state (diagnostic only).
        chosen = "tdvp"
        vis = None
        if is_lr_pauli:
            vis = estimate_local_tdvp_projected_norm(
                st_adapt.mps, gate, params, window_size=1, estimate_update_delta=True
            )
            chosen = vis.recommended_route
            if chosen == "tdvp" and enriched_seen:
                chosen = "pauli_enriched"

        # Apply node under three policies (copies are independent simulation paths).
        _apply_node_inplace(st_adapt, node, params, route=chosen if is_lr_pauli else "tdvp")
        _apply_node_inplace(st_tdvp, node, params, route="tdvp")
        _apply_node_inplace(st_enr, node, params, route="pauli_enriched")
        if is_lr_pauli and chosen == "pauli_enriched":
            enriched_seen = True

        out_adapt = StepOutcome("adaptive", _fid_err(_mps_vec(st_adapt), ref))
        out_tdvp = StepOutcome("tdvp", _fid_err(_mps_vec(st_tdvp), ref))
        out_enr = StepOutcome("enriched", _fid_err(_mps_vec(st_enr), ref))

        if is_lr_pauli and vis is not None:
            sites = tuple(gate.sites)
            print(
                "lr_gate",
                gate.name,
                "sites",
                sites,
                "projected_ratio",
                vis.projected_ratio,
                "update_delta_ratio",
                vis.update_delta_ratio,
                "tdvp_pass",
                out_tdvp.fidelity_error < 1e-10,
                "tdvp_err",
                out_tdvp.fidelity_error,
                "enriched_err",
                out_enr.fidelity_error,
                "chosen",
                chosen,
                "reason",
                vis.route_reason if chosen == vis.recommended_route else "prior enrichment present (guardrail)",
            )

    ref_final = np.asarray(Statevector(qc).data, dtype=np.complex128)
    print("final_err_adaptive", _fid_err(_mps_vec(st_adapt), ref_final))
    print("final_err_tdvp", _fid_err(_mps_vec(st_tdvp), ref_final))
    print("final_err_enriched", _fid_err(_mps_vec(st_enr), ref_final))


def main() -> None:
    safety_tol = 0.9

    qc = QuantumCircuit(8)
    qc.ryy(0.25, 1, 6)
    run_circuit("blind_ryy_vacuum_8q", qc, safety_tol=safety_tol)

    qc = QuantumCircuit(8)
    qc.rx(np.pi / 2, 6)
    qc.ryy(0.25, 1, 6)
    run_circuit("endpoint_prepared_ryy_8q", qc, safety_tol=safety_tol)

    qc = QuantumCircuit(8)
    qc.rzz(0.25, 1, 6)
    run_circuit("rzz_vacuum_8q", qc, safety_tol=safety_tol)

    qc = QuantumCircuit(8)
    qc.ry(np.pi / 4, 0)
    qc.ry(np.pi / 5, 7)
    qc.rzz(0.19, 0, 7)
    qc.rxx(0.21, 1, 6)
    qc.ryy(0.25, 2, 5)
    run_circuit("mixed_axes_disjoint_pairs_8q", qc, safety_tol=safety_tol)

    qc = QuantumCircuit(10)
    qc.ry(np.pi / 4, 1)
    qc.ry(np.pi / 5, 4)
    qc.rzz(0.19, 1, 8)
    qc.rzz(0.27, 4, 9)
    qc.rxx(0.21, 2, 7)
    qc.ryy(0.25, 3, 6)
    run_circuit("minimal_mixed_stack_10q", qc, safety_tol=safety_tol)

    qc = QuantumCircuit(12)
    qc.ry(np.pi / 4, 1)
    qc.ry(np.pi / 4, 4)
    qc.ry(np.pi / 4, 7)
    qc.rzz(0.19, 1, 10)
    qc.rzz(0.27, 4, 11)
    qc.rzz(0.33, 0, 7)
    qc.rxx(0.21, 2, 9)
    qc.ryy(0.25, 3, 8)
    run_circuit("lr_stack_mixed_12q", qc, safety_tol=safety_tol)


if __name__ == "__main__":
    main()

