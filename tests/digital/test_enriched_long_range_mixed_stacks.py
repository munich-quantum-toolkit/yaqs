# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Regression tests for previously failing mixed long-range Pauli rotation stacks."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State


def _run_hybrid(qc: QuantumCircuit) -> tuple[np.ndarray, int]:
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        svd_threshold=1e-14,
        gate_mode="hybrid",
        tangent_blindness_tol=1e-12,
        tdvp_visibility_safety_tol=None,
        tdvp_pauli_consistency_check=True,
        tdvp_pauli_consistency_tol=1e-10,
    )
    sim = Simulator()
    init = State(qc.num_qubits, initial="zeros", representation="mps")
    result = sim.run(init, qc, params)
    assert result.output_state is not None
    vec = np.asarray(result.output_state.mps.to_vec(), dtype=np.complex128)
    return vec, int(result.output_state.mps.get_max_bond())


def _fid_err(qc: QuantumCircuit, vec: np.ndarray) -> float:
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    return float(1.0 - abs(np.vdot(ref, vec)) ** 2)


@pytest.mark.parametrize(
    "qc_builder",
    [
        pytest.param(
            lambda: (lambda qc: (qc.rxx(0.21, 2, 9), qc.ryy(0.25, 3, 8), qc)[-1])(QuantumCircuit(10)),
            id="stack_core_rxx_ryy_vacuum_10q",
        ),
        pytest.param(
            lambda: (lambda qc: (qc.ry(np.pi / 4, 0), qc.ry(np.pi / 5, 7), qc.rzz(0.19, 0, 7), qc.rxx(0.21, 1, 6), qc.ryy(0.25, 2, 5), qc)[-1])(  # noqa: E501
                QuantumCircuit(8)
            ),
            id="mixed_axes_disjoint_pairs_8q",
        ),
        pytest.param(
            lambda: (lambda qc: (qc.ry(np.pi / 4, 1), qc.ry(np.pi / 5, 4), qc.rzz(0.19, 1, 8), qc.rzz(0.27, 4, 9), qc.rxx(0.21, 2, 7), qc.ryy(0.25, 3, 6), qc)[-1])(  # noqa: E501
                QuantumCircuit(10)
            ),
            id="minimal_mixed_stack_10q",
        ),
        pytest.param(
            lambda: (lambda qc: (qc.ry(np.pi / 4, 1), qc.ry(np.pi / 4, 4), qc.ry(np.pi / 4, 7), qc.rzz(0.19, 1, 10), qc.rzz(0.27, 4, 11), qc.rzz(0.33, 0, 7), qc.rxx(0.21, 2, 9), qc.ryy(0.25, 3, 8), qc)[-1])(  # noqa: E501
                QuantumCircuit(12)
            ),
            id="lr_stack_mixed_12q",
        ),
    ],
)
def test_enriched_mixed_long_range_stacks_match_qiskit(qc_builder) -> None:
    qc = qc_builder()
    vec, max_bond = _run_hybrid(qc)
    assert _fid_err(qc, vec) < 1e-10
    # Loose guardrail: ensure we didn't keep doubling bonds without truncation.
    assert max_bond <= 128

