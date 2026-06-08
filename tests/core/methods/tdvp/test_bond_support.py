# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Digital gate TDVP bond-support and routing tests."""

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
from mqt.yaqs.core.methods.tdvp.bond_support import (
    anchor_support_bonds,
    protected_bonds_for_two_site_gate,
)
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate_tdvp

from tests.core.methods.tdvp.conftest import _bond_second_schmidt, _fidelity, _qiskit_plus_rzz_reference, _tdvp_params

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.simulation_parameters import GateMode


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

    with patch("mqt.yaqs.digital.digital_tjm.tdvp", side_effect=_noop_tdvp) as mock_tdvp:
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
    result = Simulator(parallel=False, show_progress=False).run(State(length, initial="zeros"), qc, params, None)
    assert result.output_state is not None
    out = result.output_state.mps
    assert _fidelity(ref, out.to_vec()) > 0.999
    assert out.get_max_bond() <= 8
    out._assert_bond_shapes_consistent(max_bond_dim=8)


def test_fixed_chi_protected_bonds_match_gate_support() -> None:
    """Protected bonds follow the active gate window, not hardcoded anchors."""
    length = 8
    sites = (0, length - 1)
    protected = protected_bonds_for_two_site_gate(sites[0], sites[1], 0, length - 1)
    assert protected == frozenset(range(sites[0], sites[1]))
    assert 0 in protected
    assert length - 2 in protected


@pytest.mark.parametrize(
    ("site0", "site1", "window_left", "window_right", "expected"),
    [
        (0, 6, 0, 8, frozenset({0, 1, 2, 3, 4, 5})),
        (1, 8, 0, 9, frozenset({1, 2, 3, 4, 5, 6, 7})),
        (2, 7, 1, 8, frozenset({1, 2, 3, 4, 5})),
        (0, 3, 1, 4, frozenset({0, 1})),
        (0, 1, 0, 2, frozenset({0})),
    ],
)
def test_protected_bonds_for_long_range_gate(
    site0: int,
    site1: int,
    window_left: int,
    window_right: int,
    expected: frozenset[int],
) -> None:
    """Protected bonds are gate-crossed bonds mapped into the local window."""
    assert protected_bonds_for_two_site_gate(site0, site1, window_left, window_right) == expected


@pytest.mark.parametrize(
    ("num_sites", "crossed", "expected"),
    [
        (8, frozenset({0, 1, 2, 3, 4, 5, 6}), frozenset({0, 1, 2})),
        (10, frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8}), frozenset({0, 1, 2, 3})),
    ],
)
def test_anchor_support_bonds_uses_anchor_half(
    num_sites: int,
    crossed: frozenset[int],
    expected: frozenset[int],
) -> None:
    """Active support applies only to bonds left of the window midpoint."""
    assert anchor_support_bonds(crossed, num_sites) == expected
