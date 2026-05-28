# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Scaling benchmarks for hybrid simulation with initial bond padding.

Compares hybrid mode against TEBD and Qiskit on larger qubit counts. Documents when
chi=2 initial padding is sufficient and when TDVP sweep count or gate layout limits accuracy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.digital.digital_tjm import apply_single_qubit_gate, apply_two_qubit_gate

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.simulation_parameters import GateMode

_FIDELITY_TOL = 1e-8
_SVD_THRESHOLD = 1e-14
_DEFAULT_TDVP_SWEEPS = 4


def _fidelity(reference: np.ndarray, state: np.ndarray) -> float:
    """Squared overlap fidelity |<reference|state>|^2.

    Args:
        reference: Reference state vector.
        state: State vector to compare.

    Returns:
        Fidelity in ``[0, 1]``.
    """
    return float(abs(np.vdot(reference, state)) ** 2)


def _max_virtual_bond_dim(mps: MPS) -> int:
    """Largest virtual bond dimension in an MPS.

    Returns:
        Maximum of left and right bond dimensions across all sites.
    """
    bond_max = 1
    for tensor in mps.tensors:
        _, chi_l, chi_r = tensor.shape
        bond_max = max(bond_max, chi_l, chi_r)
    return bond_max


def _simulator_statevector(
    qc: QuantumCircuit,
    *,
    gate_mode: GateMode = "hybrid",
    tdvp_sweeps: int = _DEFAULT_TDVP_SWEEPS,
) -> np.ndarray:
    """Run a noiseless circuit simulation via :class:`Simulator`.

    Args:
        qc: Circuit to simulate.
        gate_mode: Two-qubit update mode.
        tdvp_sweeps: TDVP sub-sweeps per long-range gate.

    Returns:
        Final state vector (site ``i`` = Qiskit qubit ``i``).
    """
    sim_params = StrongSimParams(
        observables=[Observable(Z(), 0)],
        gate_mode=gate_mode,
        preset="exact",
        svd_threshold=_SVD_THRESHOLD,
        tdvp_sweeps=tdvp_sweeps,
        get_state=True,
    )
    state = State(qc.num_qubits, initial="zeros")
    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, None)
    assert result.output_state is not None
    return result.output_state.mps.to_vec()


def _direct_hybrid_statevector(
    qc: QuantumCircuit,
    *,
    initial_pad: int | None = None,
    tdvp_sweeps: int = _DEFAULT_TDVP_SWEEPS,
) -> tuple[np.ndarray, int]:
    """Apply a circuit directly to an MPS without simulator auto-padding.

    Args:
        qc: Circuit to simulate.
        initial_pad: Optional bond-dimension padding passed to :class:`MPS` at init.
        tdvp_sweeps: TDVP sub-sweeps per long-range gate.

    Returns:
        Final state vector and the initial virtual bond dimension before gates.
    """
    sim_params = StrongSimParams(
        gate_mode="hybrid",
        preset="exact",
        svd_threshold=_SVD_THRESHOLD,
        tdvp_sweeps=tdvp_sweeps,
        get_state=True,
    )
    mps = MPS(qc.num_qubits, state="zeros", pad=initial_pad)
    initial_bond = _max_virtual_bond_dim(mps)
    dag = circuit_to_dag(qc)
    for node in dag.topological_op_nodes():
        if node.op.num_qubits == 1:
            apply_single_qubit_gate(mps, node)
        else:
            apply_two_qubit_gate(mps, node, sim_params)
    return mps.to_vec(), initial_bond


def _interior_long_range_cnot(n: int) -> QuantumCircuit:
    """Build ``H(mid); CX(mid, n-1)`` with non-adjacent control and target."""
    mid = n // 2
    assert mid + 1 < n - 1, "probe requires a non-nearest-neighbor CNOT"
    qc = QuantumCircuit(n)
    qc.h(mid)
    qc.cx(mid, n - 1)
    return qc


def _double_interior_long_range(n: int) -> QuantumCircuit:
    """Build two overlapping interior long-range CNOTs."""
    qc = QuantumCircuit(n)
    qc.h(1)
    qc.cx(1, n - 1)
    qc.h(2)
    qc.cx(2, n - 2)
    return qc


def _mid_control_to_zero(n: int) -> QuantumCircuit:
    """Build ``H(mid); CX(mid, 0)`` with control far from target."""
    mid = n // 2
    qc = QuantumCircuit(n)
    qc.h(mid)
    qc.cx(mid, 0)
    return qc


@pytest.mark.parametrize("num_qubits", [5, 6, 8, 10, 12])
def test_single_long_range_cnot_hybrid_matches_tebd_and_qiskit(num_qubits: int) -> None:
    """One interior LR CNOT: chi>=2 padding (via Simulator) suffices up to 12 qubits."""
    qc = _interior_long_range_cnot(num_qubits)
    reference = np.asarray(Statevector(qc).data, dtype=np.complex128)
    hybrid = _simulator_statevector(qc, gate_mode="hybrid")
    tebd = _simulator_statevector(qc, gate_mode="tebd")

    assert _fidelity(reference, hybrid) == pytest.approx(1.0, abs=_FIDELITY_TOL)
    assert _fidelity(reference, tebd) == pytest.approx(1.0, abs=_FIDELITY_TOL)
    assert _fidelity(hybrid, tebd) == pytest.approx(1.0, abs=_FIDELITY_TOL)


@pytest.mark.parametrize("num_qubits", [6, 8, 10])
def test_interior_long_range_requires_initial_bond_padding(num_qubits: int) -> None:
    """Without chi>=2 before the LR gate, interior TDVP stays at ~0.5 fidelity."""
    qc = _interior_long_range_cnot(num_qubits)
    reference = np.asarray(Statevector(qc).data, dtype=np.complex128)

    unpadded, initial_bond = _direct_hybrid_statevector(qc, initial_pad=None)
    padded, padded_bond = _direct_hybrid_statevector(qc, initial_pad=2)

    assert initial_bond == 1
    assert padded_bond == 2
    assert _fidelity(reference, unpadded) == pytest.approx(0.5, abs=1e-6)
    assert _fidelity(reference, padded) == pytest.approx(1.0, abs=_FIDELITY_TOL)


def test_nearest_neighbor_cnot_works_without_initial_padding() -> None:
    """Adjacent CNOTs use TEBD in hybrid mode and do not need initial bond padding."""
    qc = QuantumCircuit(4)
    qc.h(2)
    qc.cx(2, 3)
    reference = np.asarray(Statevector(qc).data, dtype=np.complex128)

    unpadded, initial_bond = _direct_hybrid_statevector(qc, initial_pad=None)
    assert initial_bond == 1
    assert _fidelity(reference, unpadded) == pytest.approx(1.0, abs=_FIDELITY_TOL)


@pytest.mark.parametrize("initial_pad", [2, 4, 8, 16])
def test_higher_initial_padding_does_not_fix_double_long_range(initial_pad: int) -> None:
    """Extra initial padding does not rescue sequential interior LR TDVP updates."""
    qc = _double_interior_long_range(6)
    reference = np.asarray(Statevector(qc).data, dtype=np.complex128)
    hybrid = _direct_hybrid_statevector(qc, initial_pad=initial_pad)[0]
    tebd = _simulator_statevector(qc, gate_mode="tebd")

    assert _fidelity(reference, tebd) == pytest.approx(1.0, abs=_FIDELITY_TOL)
    assert _fidelity(reference, hybrid) < 0.55
    assert _fidelity(hybrid, tebd) < 0.55


def test_double_long_range_hybrid_underperforms_tebd() -> None:
    """Two interior LR CNOTs: TEBD is exact, hybrid TDVP path is far from unitary."""
    qc = _double_interior_long_range(8)
    reference = np.asarray(Statevector(qc).data, dtype=np.complex128)
    hybrid = _simulator_statevector(qc, gate_mode="hybrid")
    tebd = _simulator_statevector(qc, gate_mode="tebd")

    assert _fidelity(reference, tebd) == pytest.approx(1.0, abs=_FIDELITY_TOL)
    assert _fidelity(reference, hybrid) == pytest.approx(0.44, abs=0.05)
    assert _fidelity(hybrid, tebd) == pytest.approx(0.44, abs=0.05)


def test_mid_control_default_sweeps_hybrid_below_tebd() -> None:
    """``H(mid); CX(mid, 0)`` at default sweeps: hybrid ~0.96, TEBD exact."""
    qc = _mid_control_to_zero(8)
    reference = np.asarray(Statevector(qc).data, dtype=np.complex128)
    hybrid = _simulator_statevector(qc, gate_mode="hybrid", tdvp_sweeps=_DEFAULT_TDVP_SWEEPS)
    tebd = _simulator_statevector(qc, gate_mode="tebd")

    assert _fidelity(reference, tebd) == pytest.approx(1.0, abs=_FIDELITY_TOL)
    assert _fidelity(reference, hybrid) == pytest.approx(0.962, abs=0.01)
    assert _fidelity(hybrid, tebd) == pytest.approx(0.962, abs=0.01)


@pytest.mark.parametrize(
    ("tdvp_sweeps", "expected_min_fidelity"),
    [
        (1, 0.49),
        (4, 0.95),
        (16, 0.99),
        (32, 0.999),
    ],
)
def test_mid_control_fidelity_improves_with_tdvp_sweeps(
    tdvp_sweeps: int,
    expected_min_fidelity: float,
) -> None:
    """Mid-control LR CNOT fidelity increases with ``tdvp_sweeps`` (pad=2 fixed)."""
    qc = _mid_control_to_zero(8)
    reference = np.asarray(Statevector(qc).data, dtype=np.complex128)
    hybrid, _ = _direct_hybrid_statevector(qc, initial_pad=2, tdvp_sweeps=tdvp_sweeps)
    fidelity = _fidelity(reference, hybrid)
    assert fidelity >= expected_min_fidelity


def test_single_long_range_sweeps_independent_at_pad_two() -> None:
    """One interior LR CNOT at chi=2: fidelity stays at 1 across sweep counts."""
    qc = _interior_long_range_cnot(8)
    reference = np.asarray(Statevector(qc).data, dtype=np.complex128)
    fidelities = [
        _fidelity(reference, _direct_hybrid_statevector(qc, initial_pad=2, tdvp_sweeps=s)[0])
        for s in (1, 2, 4, 8, 16)
    ]
    assert all(f == pytest.approx(1.0, abs=_FIDELITY_TOL) for f in fidelities)


def test_double_long_range_sweeps_do_not_improve_hybrid() -> None:
    """Sequential interior LR CNOTs: more sweeps do not move hybrid toward TEBD."""
    qc = _double_interior_long_range(6)
    reference = np.asarray(Statevector(qc).data, dtype=np.complex128)
    fidelities = [
        _fidelity(reference, _direct_hybrid_statevector(qc, initial_pad=2, tdvp_sweeps=s)[0])
        for s in (1, 4, 16)
    ]
    assert all(f < 0.55 for f in fidelities)
    assert max(fidelities) - min(fidelities) < 0.05
