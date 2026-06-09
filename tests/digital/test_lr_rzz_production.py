# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Production long-range RZZ physical invariants (seed prep, no support_bonds)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from qiskit.quantum_info import Pauli, Statevector

from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.core.methods.tdvp import tdvp
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate_tdvp, apply_window, construct_generator_mpo
from tests.core.methods.tdvp.conftest import (
    NORM_TOL,
    _fidelity,
    _max_bond,
    _qiskit_plus_rzz_reference,
    _tdvp_params,
)

pytestmark = pytest.mark.tdvp_regression

RZZ_THETA = 0.3
Z_TOL = 1e-8
PRODUCTION_LENGTHS = (6, 10, 14)
ROUND_TRIP_TOL = 1e-10


def _z_expectation(vec: np.ndarray, site: int) -> float:
    num_qubits = int(np.log2(vec.size))
    label = ["I"] * num_qubits
    label[num_qubits - 1 - site] = "Z"
    return float(np.real(Statevector(vec).expectation_value(Pauli("".join(label)))))


def _apply_production_lr_rzz(
    length: int,
    *,
    theta: float = RZZ_THETA,
    max_bond_dim: int | None = None,
    tdvp_sweeps: int = 1,
) -> np.ndarray:
    gate = GateLibrary.rzz([theta])
    gate.set_sites(0, length - 1)
    out = copy.deepcopy(State(length, initial="x+").mps)
    apply_two_qubit_gate_tdvp(out, gate, _tdvp_params(max_bond_dim=max_bond_dim, tdvp_sweeps=tdvp_sweeps))
    return out.to_vec()


def _apply_no_support_baseline(
    length: int,
    *,
    theta: float = RZZ_THETA,
    max_bond_dim: int | None = None,
    tdvp_sweeps: int = 1,
) -> np.ndarray:
    """Dynamic TDVP without seed prep or support_bonds (micro diagnostic baseline).

    Returns:
        State vector after applying the gate.
    """
    gate = GateLibrary.rzz([theta])
    gate.set_sites(0, length - 1)
    prep = copy.deepcopy(State(length, initial="x+").mps)
    mpo, first_site, last_site = construct_generator_mpo(gate, length)
    short_state, short_mpo, window = apply_window(prep, mpo, first_site, last_site, 1)
    params = _tdvp_params(max_bond_dim=max_bond_dim, tdvp_sweeps=tdvp_sweeps)
    tdvp(short_state, short_mpo, params, support_bonds=None)
    for i in range(window[0], window[1] + 1):
        prep.tensors[i] = short_state.tensors[i - window[0]]
    return prep.to_vec()


@pytest.mark.parametrize("length", PRODUCTION_LENGTHS)
def test_single_lr_rzz_z_observables_near_exact(length: int) -> None:
    """Endpoint RZZ on |+⟩^L: all ⟨Z_i⟩ match exact reference."""
    from tests.core.methods.tdvp.conftest import Z_TOL, _assert_z_observables_match

    vec = _apply_production_lr_rzz(length, max_bond_dim=None, tdvp_sweeps=1)
    ref = _qiskit_plus_rzz_reference(length, RZZ_THETA, sites=(0, length - 1))
    _assert_z_observables_match(ref, vec, length, tol=Z_TOL)


@pytest.mark.parametrize("length", PRODUCTION_LENGTHS)
def test_single_lr_rzz_spectator_z_near_zero(length: int) -> None:
    """Interior sites remain unpolarized: |⟨Z_i⟩| ≈ 0 for spectators."""
    vec = _apply_production_lr_rzz(length, max_bond_dim=None, tdvp_sweeps=1)
    for site in range(1, length - 1):
        assert abs(_z_expectation(vec, site)) < Z_TOL


@pytest.mark.parametrize("length", PRODUCTION_LENGTHS)
def test_single_lr_rzz_norm_preserved(length: int) -> None:
    """Production LR RZZ preserves state norm."""
    vec = _apply_production_lr_rzz(length, max_bond_dim=None, tdvp_sweeps=1)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < NORM_TOL


@pytest.mark.parametrize("length", PRODUCTION_LENGTHS)
def test_single_lr_rzz_bond_dim_not_inflated(length: int) -> None:
    """Bond dimension on crossed path stays minimal for endpoint |+⟩ RZZ."""
    gate = GateLibrary.rzz([RZZ_THETA])
    gate.set_sites(0, length - 1)
    out = copy.deepcopy(State(length, initial="x+").mps)
    apply_two_qubit_gate_tdvp(out, gate, _tdvp_params(max_bond_dim=None, tdvp_sweeps=1))
    assert _max_bond(out) <= 2


@pytest.mark.parametrize("length", PRODUCTION_LENGTHS)
def test_lr_rzz_round_trip_restores_plus_state(length: int) -> None:
    """RZZ(theta) followed by RZZ(-theta) returns to |+⟩^L."""
    prep = copy.deepcopy(State(length, initial="x+").mps)
    gate_fwd = GateLibrary.rzz([RZZ_THETA])
    gate_fwd.set_sites(0, length - 1)
    gate_bwd = GateLibrary.rzz([-RZZ_THETA])
    gate_bwd.set_sites(0, length - 1)
    params = _tdvp_params(max_bond_dim=None, tdvp_sweeps=1)
    apply_two_qubit_gate_tdvp(prep, gate_fwd, params)
    apply_two_qubit_gate_tdvp(prep, gate_bwd, params)
    plus = State(length, initial="x+").mps.to_vec()
    assert _fidelity(plus, prep.to_vec()) > 1.0 - ROUND_TRIP_TOL


@pytest.mark.parametrize("length", PRODUCTION_LENGTHS)
def test_lr_rzz_round_trip_restores_z_observables(length: int) -> None:
    """Round-trip LR RZZ restores all single-site ⟨Z_i⟩."""
    prep = copy.deepcopy(State(length, initial="x+").mps)
    gate_fwd = GateLibrary.rzz([RZZ_THETA])
    gate_fwd.set_sites(0, length - 1)
    gate_bwd = GateLibrary.rzz([-RZZ_THETA])
    gate_bwd.set_sites(0, length - 1)
    params = _tdvp_params(max_bond_dim=None, tdvp_sweeps=1)
    apply_two_qubit_gate_tdvp(prep, gate_fwd, params)
    apply_two_qubit_gate_tdvp(prep, gate_bwd, params)
    vec = prep.to_vec()
    for site in range(length):
        assert abs(_z_expectation(vec, site)) < Z_TOL


@pytest.mark.parametrize("length", PRODUCTION_LENGTHS)
def test_production_matches_no_support_baseline(length: int) -> None:
    """Production path (no seed prep, support_bonds=None) matches no-support micro baseline."""
    prod = _apply_production_lr_rzz(length, max_bond_dim=64, tdvp_sweeps=1)
    baseline = _apply_no_support_baseline(length, max_bond_dim=64, tdvp_sweeps=1)
    assert _fidelity(prod, baseline) == pytest.approx(1.0, abs=1e-12)


def test_production_default_tdvp_sweeps_is_one() -> None:
    """StrongSimParams default tdvp_sweeps remains 1 (multi-sweep is opt-in)."""
    params = StrongSimParams(preset="exact", get_state=True)
    assert params.tdvp_sweeps == 1
