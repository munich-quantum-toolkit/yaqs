# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Fixed-χ dynamic TDVP regression tests for protected-bond support retention."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.core.methods.tdvp_retained_bonds import protected_bonds_for_two_site_gate
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate_tdvp

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.mps import MPS


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a, b)) ** 2)


def _bond_second_schmidt(mps: MPS, bond: int) -> float:
    spec = mps.get_schmidt_spectrum([bond, bond + 1])
    vals = np.asarray(spec[~np.isnan(spec)], dtype=np.float64)
    norm = float(np.sum(vals**2))
    if norm > 0.0:
        vals /= np.sqrt(norm)
    return float(vals[1]) if len(vals) > 1 else 0.0


def _max_bond(mps: MPS) -> int:
    return max(mps.bond_dimensions()) if mps.length > 1 else 1


def _tdvp_params(*, max_bond_dim: int | None, tdvp_sweeps: int) -> StrongSimParams:
    return StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=max_bond_dim,
        tdvp_sweeps=tdvp_sweeps,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )


def _qiskit_plus_rzz_reference(length: int, theta: float, *, sites: tuple[int, int]) -> np.ndarray:
    qc = QuantumCircuit(length)
    qc.h(range(length))
    qc.rzz(theta, sites[0], sites[1])
    return np.asarray(Statevector(qc).data, dtype=np.complex128)


def _prep_state(name: str, length: int) -> MPS:
    if name == "plus":
        return State(length, initial="x+").mps
    if name == "zeros":
        return State(length, initial="zeros").mps
    if name == "haar":
        return State(length, initial="haar-random").mps
    if name == "low_depth":
        prep_qc = QuantumCircuit(length)
        for i in range(0, length, 2):
            prep_qc.h(i)
        for i in range(length - 1):
            prep_qc.cx(i, i + 1)
        params = StrongSimParams(
            preset="exact",
            get_state=True,
            max_bond_dim=8,
            gate_mode="mpo",
            svd_threshold=1e-14,
            krylov_tol=1e-12,
        )
        result = Simulator(parallel=False, show_progress=False).run(
            State(length, initial="zeros"), prep_qc, params, None
        )
        assert result.output_state is not None
        return result.output_state.mps
    msg = f"Unknown initial state {name!r}"
    raise ValueError(msg)


def _apply_lr_gate(
    mps: MPS,
    gate_name: str,
    theta: float,
    *,
    max_bond_dim: int,
    sweeps: int,
) -> MPS:
    if gate_name == "rzz":
        gate = GateLibrary.rzz([theta])
    elif gate_name == "rxx":
        gate = GateLibrary.rxx([theta])
    else:
        msg = f"Unknown gate {gate_name!r}"
        raise ValueError(msg)
    gate.set_sites(0, mps.length - 1)
    out = copy.deepcopy(mps)
    apply_two_qubit_gate_tdvp(out, gate, _tdvp_params(max_bond_dim=max_bond_dim, tdvp_sweeps=sweeps))
    return out


def _run_circuit(
    prep: MPS,
    qc: QuantumCircuit,
    *,
    max_bond_dim: int,
    sweeps: int,
) -> MPS:
    init = State(prep.length, tensors=[t.copy() for t in prep.tensors])
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=max_bond_dim,
        tdvp_sweeps=sweeps,
        gate_mode="tdvp",
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )
    result = Simulator(parallel=False, show_progress=False).run(init, qc, params, None)
    assert result.output_state is not None
    return result.output_state.mps


@pytest.mark.parametrize("length", [6, 8, 10])
@pytest.mark.parametrize("max_bond_dim", [1, 2, 4, 8])
@pytest.mark.parametrize("initial_state", ["plus", "zeros", "haar", "low_depth"])
@pytest.mark.parametrize("gate_name", ["rzz", "rxx"])
@pytest.mark.parametrize("sweeps", [1, 4, 16])
def test_fixed_chi_lr_gate_respects_cap(
    length: int,
    max_bond_dim: int,
    initial_state: str,
    gate_name: str,
    sweeps: int,
) -> None:
    """Long-range Pauli rotations never exceed the configured χ cap."""
    theta = 0.3
    prep = _prep_state(initial_state, length)
    out = _apply_lr_gate(prep, gate_name, theta, max_bond_dim=max_bond_dim, sweeps=sweeps)
    assert _max_bond(out) <= max_bond_dim
    assert all(dim <= max_bond_dim for dim in out.bond_dimensions())
    out._assert_bond_shapes_consistent(max_bond_dim=max_bond_dim)
    norm = float(np.linalg.norm(out.to_vec()))
    assert norm == pytest.approx(1.0, abs=1e-8)


def _rzz_lr_ladder_circuit(length: int) -> QuantumCircuit:
    """Build a long-range RZZ ladder circuit for regression tests.

    Returns:
        Circuit with ``RZZ`` gates on pairs ``(i, L-1-i)``.
    """
    qc = QuantumCircuit(length)
    for i in range(length // 2):
        partner = length - 1 - i
        if i < partner:
            qc.rzz(0.3, i, partner)
    return qc


def test_fixed_chi_rzz_lr_ladder_no_shape_error() -> None:
    """Multi-gate ladder circuits complete without bond-dimension violations."""
    length = 8
    prep = State(length, initial="zeros").mps
    out = _run_circuit(prep, _rzz_lr_ladder_circuit(length), max_bond_dim=8, sweeps=4)
    assert _max_bond(out) <= 8


def test_fixed_chi_plus_rzz_eight_sweeps64() -> None:
    """χ=8 reproduces the rank-2 RZZ branch for |+⟩ without exceeding the cap."""
    theta = 0.3
    length = 8
    prep = _prep_state("plus", length)
    out = _apply_lr_gate(prep, "rzz", theta, max_bond_dim=8, sweeps=64)
    ref = _qiskit_plus_rzz_reference(length, theta, sites=(0, length - 1))
    assert _fidelity(ref, out.to_vec()) > 0.999
    assert _max_bond(out) <= 8
    assert _bond_second_schmidt(out, 0) > 0.1


def test_fixed_chi_truncates_high_bond_initial_state() -> None:
    """Fixed-χ TDVP truncates incoming MPS bond dimensions before evolving."""
    length = 8
    prep_qc = QuantumCircuit(length)
    for i in range(0, length, 2):
        prep_qc.h(i)
    for i in range(length - 1):
        prep_qc.cx(i, i + 1)
    prep_params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=None,
        gate_mode="mpo",
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )
    prep_state = Simulator(parallel=False, show_progress=False).run(
        State(length, initial="zeros"), prep_qc, prep_params, None
    )
    assert prep_state.output_state is not None
    prep = prep_state.output_state.mps
    assert max(prep.bond_dimensions()) > 1

    gate = GateLibrary.rzz([0.3])
    gate.set_sites(0, length - 1)
    out = _apply_lr_gate(prep, "rzz", 0.3, max_bond_dim=1, sweeps=4)
    assert _max_bond(out) <= 1
    assert float(np.linalg.norm(out.to_vec())) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("initial_state", ["plus", "low_depth"])
@pytest.mark.parametrize("max_bond_dim", [1, 2, 8])
@pytest.mark.parametrize("sweeps", [1, 4, 16, 64])
def test_fixed_chi_norm_stability(
    initial_state: str,
    max_bond_dim: int,
    sweeps: int,
) -> None:
    """Long-range TDVP stays normalized across χ budgets and sweep counts."""
    length = 8
    prep = _prep_state(initial_state, length)
    out = _apply_lr_gate(prep, "rzz", 0.3, max_bond_dim=max_bond_dim, sweeps=sweeps)
    assert out.norm() == pytest.approx(1.0, abs=1e-8)
    out._assert_bond_shapes_consistent(max_bond_dim=max_bond_dim)


def test_fixed_chi_protected_bonds_match_gate_support() -> None:
    """Protected bonds follow the active gate window, not hardcoded anchors."""
    length = 8
    sites = (0, length - 1)
    protected = protected_bonds_for_two_site_gate(sites[0], sites[1], 0, length - 1)
    assert protected == frozenset(range(sites[0], sites[1]))
    assert 0 in protected
    assert length - 2 in protected


def test_fixed_chi_plus_rzz_one_rank1_limited() -> None:
    """χ=1 stays rank-1 limited, does not pad to 2, and does not crash."""
    theta = 0.3
    length = 8
    prep = _prep_state("plus", length)
    out = _apply_lr_gate(prep, "rzz", theta, max_bond_dim=1, sweeps=64)
    ref = _qiskit_plus_rzz_reference(length, theta, sites=(0, length - 1))
    plus = _prep_state("plus", length)
    assert _max_bond(out) == 1
    assert _fidelity(ref, out.to_vec()) == pytest.approx(0.977668, abs=0.01)
    assert _fidelity(plus.to_vec(), out.to_vec()) > 0.999
