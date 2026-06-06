# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Protected-bond support retention and dynamic TDVP routing tests."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate_tdvp

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.mps import MPS
    from mqt.yaqs.core.data_structures.simulation_parameters import GateMode


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a, b)) ** 2)


def _bond_second_schmidt(mps: MPS, bond: int) -> float:
    spec = mps.get_schmidt_spectrum([bond, bond + 1])
    vals = np.asarray(spec[~np.isnan(spec)], dtype=np.float64)
    norm = float(np.sum(vals**2))
    if norm > 0.0:
        vals /= np.sqrt(norm)
    return float(vals[1]) if len(vals) > 1 else 0.0


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


@pytest.mark.parametrize("length", [7, 8, 10])
def test_endpoint_lr_protects_all_crossed_bonds(length: int) -> None:
    """Endpoint long-range RZZ retains rank-2 support on every crossed bond."""
    theta = 0.3
    sites = (0, length - 1)
    gate = GateLibrary.rzz([theta])
    gate.set_sites(*sites)
    sweeps = 256 if length >= 10 else 64

    out = copy.deepcopy(State(length, initial="x+").mps)
    apply_two_qubit_gate_tdvp(out, gate, _tdvp_params(max_bond_dim=None, tdvp_sweeps=sweeps))

    ref = _qiskit_plus_rzz_reference(length, theta, sites=sites)
    assert _fidelity(ref, out.to_vec()) > 0.999
    for bond in range(sites[0], sites[1]):
        assert _bond_second_schmidt(out, bond) > 0.1
    out._assert_bond_shapes_consistent()


def test_dynamic_tdvp_lr_gate_uses_dynamic_mode() -> None:
    """Long-range TDVP gates must route through dynamic TDVP, not the 2-site kernel."""
    length = 8
    theta = 0.3
    gate = GateLibrary.rzz([theta])
    gate.set_sites(0, length - 1)
    out = copy.deepcopy(State(length, initial="x+").mps)
    params = _tdvp_params(max_bond_dim=None, tdvp_sweeps=4)

    def _noop_tdvp(*_args: object, **_kwargs: object) -> None:
        return None

    with patch("mqt.yaqs.digital.digital_tjm._tdvp_dynamic", side_effect=_noop_tdvp) as mock_tdvp:
        apply_two_qubit_gate_tdvp(out, gate, params)
        assert mock_tdvp.call_count == 1


def test_dynamic_tdvp_lr_gate_evolution_not_product() -> None:
    """Long-range RZZ on |+>^L changes the state and opens the first cut."""
    length = 8
    theta = 0.3
    sites = (0, length - 1)
    gate = GateLibrary.rzz([theta])
    gate.set_sites(*sites)

    out = copy.deepcopy(State(length, initial="x+").mps)
    apply_two_qubit_gate_tdvp(out, gate, _tdvp_params(max_bond_dim=None, tdvp_sweeps=64))

    ref = _qiskit_plus_rzz_reference(length, theta, sites=sites)
    plus = copy.deepcopy(State(length, initial="x+").mps)
    assert _fidelity(ref, out.to_vec()) > 0.999
    assert _fidelity(plus.to_vec(), out.to_vec()) < 0.999
    assert _bond_second_schmidt(out, 0) > 0.1


@pytest.mark.parametrize("gate_mode", ["swaps", "mpo", "tdvp"])
@pytest.mark.parametrize(("gate_name", "sites"), [("rzz", (0, 3)), ("rxx", (0, 3))])
def test_digital_gate_modes_small_lr_high_fidelity(
    gate_mode: str,
    gate_name: str,
    sites: tuple[int, int],
) -> None:
    """Small LR/NN circuits stay accurate and cap-compliant across gate modes."""
    length = 4
    theta = 0.3
    qc = QuantumCircuit(length)
    qc.h(range(length))
    if gate_name == "rzz":
        qc.rzz(theta, sites[0], sites[1])
    else:
        qc.rxx(theta, sites[0], sites[1])

    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=8,
        gate_mode=cast("GateMode", gate_mode),
        tdvp_sweeps=16,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )
    result = Simulator(parallel=False, show_progress=False).run(
        State(length, initial="zeros"), qc, params, None
    )
    assert result.output_state is not None
    out = result.output_state.mps
    assert _fidelity(ref, out.to_vec()) > 0.999
    assert out.get_max_bond() <= 8
    out._assert_bond_shapes_consistent(max_bond_dim=8)
