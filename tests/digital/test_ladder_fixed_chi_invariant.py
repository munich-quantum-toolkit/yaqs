# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Fixed-χ dynamic TDVP must match uncapped when observed bonds stay below the cap."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate_tdvp, apply_two_qubit_gate_tebd
from tests.core.methods.tdvp.conftest import _fidelity, _max_bond, _tdvp_params

pytestmark = pytest.mark.tdvp_regression

LADDER_LENGTH = 10
LADDER_INIT = "x+"
LADDER_CHI = 64
LADDER_SWEEPS = 1
LADDER_ANGLE = 0.3
INVARIANT_TOL = 1e-10


def _ladder_pairs(length: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for left in range(length // 2):
        right = length - 1 - left
        if left < right:
            pairs.append((left, right))
    return pairs


def _exact_ladder_reference(length: int, num_gates: int) -> np.ndarray:
    qc = QuantumCircuit(length)
    qc.h(range(length))
    for site_a, site_b in _ladder_pairs(length)[:num_gates]:
        qc.rzz(LADDER_ANGLE, site_a, site_b)
    return np.asarray(Statevector(qc).data, dtype=np.complex128)


def _apply_ladder_gates(state: State, *, max_bond_dim: int | None) -> None:
    params = _tdvp_params(max_bond_dim=max_bond_dim, tdvp_sweeps=LADDER_SWEEPS)
    for site_a, site_b in _ladder_pairs(state.length):
        gate = GateLibrary.rzz([LADDER_ANGLE])
        gate.set_sites(site_a, site_b)
        if abs(site_a - site_b) == 1:
            apply_two_qubit_gate_tebd(state.mps, gate, params)
        else:
            apply_two_qubit_gate_tdvp(state.mps, gate, params)


def test_rzz_lr_ladder_enforces_cap_when_uncapped_exceeds() -> None:
    """When uncapped ladder reaches χ above the cap, capped evolution differs and respects χ."""
    from tests.core.methods.tdvp.conftest import _prep_state, _run_circuit
    from tests.digital.test_digital_tjm import _rzz_lr_ladder_circuit

    length = LADDER_LENGTH
    prep = _prep_state("plus", length)
    uncapped = _run_circuit(copy.deepcopy(prep), _rzz_lr_ladder_circuit(length), max_bond_dim=64, sweeps=1)
    capped = _run_circuit(copy.deepcopy(prep), _rzz_lr_ladder_circuit(length), max_bond_dim=3, sweeps=1)

    assert _max_bond(uncapped) > 3
    assert _max_bond(capped) <= 3
    assert abs(float(np.linalg.norm(capped.to_vec())) - 1.0) < 1e-6
    assert _fidelity(uncapped.to_vec(), capped.to_vec()) < 0.99


def test_dynamic_tdvp_truncates_high_chi_incoming_state() -> None:
    """Fixed-χ LR gates truncate incoming MPS bonds above ``max_bond_dim`` before evolving."""
    length = 8
    prep = State(length, initial="x+").mps
    prep._ensure_internal_bond_dims(tuple(range(length - 1)), 8, max_dim=8)
    assert _max_bond(prep) > 4

    gate = GateLibrary.rzz([0.3])
    gate.set_sites(0, length - 1)
    out = copy.deepcopy(prep)
    apply_two_qubit_gate_tdvp(out, gate, _tdvp_params(max_bond_dim=4, tdvp_sweeps=1))
    assert _max_bond(out) <= 4
    assert abs(float(np.linalg.norm(out.to_vec())) - 1.0) < 1e-6


def test_rzz_lr_ladder_fixed_chi_matches_uncapped_when_below_cap() -> None:
    """L=10 |+⟩ ladder: χ=64 matches χ=None when no bond reaches the cap."""
    uncapped = State(LADDER_LENGTH, initial=LADDER_INIT)
    capped = State(LADDER_LENGTH, initial=LADDER_INIT)
    _apply_ladder_gates(uncapped, max_bond_dim=None)
    _apply_ladder_gates(capped, max_bond_dim=LADDER_CHI)

    assert _max_bond(uncapped.mps) <= LADDER_CHI
    assert _max_bond(capped.mps) <= LADDER_CHI

    exact = _exact_ladder_reference(LADDER_LENGTH, len(_ladder_pairs(LADDER_LENGTH)))
    assert _fidelity(exact, uncapped.mps.to_vec()) > 0.9
    assert _fidelity(uncapped.mps.to_vec(), capped.mps.to_vec()) == pytest.approx(1.0, abs=INVARIANT_TOL)


@pytest.mark.parametrize("gate_index", range(5))
def test_rzz_lr_ladder_per_gate_fixed_chi_matches_uncapped(gate_index: int) -> None:
    """Per-gate partial ladder: χ=64 matches χ=None while bonds stay below cap."""
    uncapped = State(LADDER_LENGTH, initial=LADDER_INIT)
    capped = State(LADDER_LENGTH, initial=LADDER_INIT)
    params_u = _tdvp_params(max_bond_dim=None, tdvp_sweeps=LADDER_SWEEPS)
    params_c = _tdvp_params(max_bond_dim=LADDER_CHI, tdvp_sweeps=LADDER_SWEEPS)

    for idx, (site_a, site_b) in enumerate(_ladder_pairs(LADDER_LENGTH)):
        if idx > gate_index:
            break
        gate = GateLibrary.rzz([LADDER_ANGLE])
        gate.set_sites(site_a, site_b)
        for state, params in ((uncapped, params_u), (capped, params_c)):
            if abs(site_a - site_b) == 1:
                apply_two_qubit_gate_tebd(state.mps, gate, params)
            else:
                apply_two_qubit_gate_tdvp(state.mps, gate, params)

    ref = _exact_ladder_reference(LADDER_LENGTH, gate_index + 1)
    assert _max_bond(capped.mps) <= LADDER_CHI
    assert _fidelity(ref, uncapped.mps.to_vec()) == pytest.approx(
        _fidelity(ref, capped.mps.to_vec()), abs=INVARIANT_TOL
    )
    assert _fidelity(uncapped.mps.to_vec(), capped.mps.to_vec()) == pytest.approx(1.0, abs=INVARIANT_TOL)
