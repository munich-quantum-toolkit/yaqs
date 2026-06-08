# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared helpers for TDVP unit and regression tests."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import GateLibrary
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
