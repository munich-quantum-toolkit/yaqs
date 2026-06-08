# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for TDVP integrators and fixed-χ digital regression checks."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable, StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import GateLibrary, Z
from mqt.yaqs.core.methods.tdvp import tdvp
from mqt.yaqs.core.methods.tdvp.primitives import update_site
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate_tdvp, apply_window, construct_generator_mpo

from tests.core.methods.tdvp.conftest import (
    _apply_lr_gate,
    _bond_second_schmidt,
    _fidelity,
    _max_bond,
    _prep_state,
    _qiskit_plus_rzz_reference,
    _run_circuit,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_single_site_tdvp() -> None:
    """Test the single_site_TDVP function."""
    L = 5
    J = 1
    g = 0.5
    H = MPO.ising(L, J, g)

    state = MPS(L, state="zeros")
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        tdvp_mode="1site",
    )
    tdvp(state, H, sim_params)
    assert state.length == L
    for tensor in state.tensors:
        assert isinstance(tensor, np.ndarray)
    canonical_site = state.check_canonical_form()[0]
    assert canonical_site == 0, (
        f"MPS should be site-canonical at site 0 after single-site TDVP, but got canonical site: {canonical_site}"
    )


def test_two_site_tdvp() -> None:
    """Test the two_site_TDVP function."""
    L = 5
    J = 1
    g = 0.5
    H = MPO.ising(L, J, g)

    state = MPS(L, state="zeros")
    ref_mps = deepcopy(state)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        krylov_tol=1e-12,
        preset="exact",
        tdvp_mode="2site",
    )
    tdvp(state, H, sim_params)
    assert state.length == L
    for tensor in state.tensors:
        assert isinstance(tensor, np.ndarray)
    canonical_site = state.check_canonical_form()[0]
    assert canonical_site == 0, (
        f"MPS should be site-canonical at site 0 after two-site TDVP, but got canonical site: {canonical_site}"
    )
    state_vec = ref_mps.to_vec()
    H_mat = H.to_matrix()
    U = expm(-1j * sim_params.dt * H_mat)
    state_vec = U @ state_vec
    found = state.to_vec()
    assert np.allclose(state_vec, found)


def test_two_site_tdvp_circuit_sweep_scaling() -> None:
    """Circuit-mode 2TDVP passes symmetric substeps when tdvp_sweeps>1."""
    L = 4
    H = MPO.ising(L, 1.0, 0.5)
    state = MPS(L, state="zeros")
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], tdvp_sweeps=2, preset="exact", tdvp_mode="2site")

    with patch("mqt.yaqs.core.methods.tdvp.integrators._two_site_tdvp_sweep") as mock_sweep:
        tdvp(state, H, sim_params)
        assert mock_sweep.call_count == 1
        sweep_plan = mock_sweep.call_args.kwargs["sweep_plan"]
        assert len(sweep_plan) == 2
        assert sweep_plan[0] == pytest.approx(0.5)
        assert sweep_plan[1] == pytest.approx(0.5)


def test_two_site_tdvp_circuit_single_sweep_uses_symmetric() -> None:
    """Circuit tdvp_sweeps=1 uses one symmetric substep, matching analog geometry."""
    L = 4
    H = MPO.ising(L, 1.0, 0.5)
    state = MPS(L, state="zeros")
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], tdvp_sweeps=1, preset="exact", tdvp_mode="2site")

    with patch("mqt.yaqs.core.methods.tdvp.integrators._two_site_tdvp_sweep") as mock_sweep:
        tdvp(state, H, sim_params)
        assert mock_sweep.call_args.kwargs["sweep_plan"] == [1.0]


def test_two_site_tdvp_analog_sweep_preservation() -> None:
    """Analog 2TDVP with tdvp_sweeps>1 still integrates over the full dt per step."""
    L = 5
    J = 1
    g = 0.5
    H = MPO.ising(L, J, g)

    state = MPS(L, state="zeros")
    ref_mps = deepcopy(state)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        tdvp_sweeps=2,
        sample_timesteps=True,
        krylov_tol=1e-12,
        preset="exact",
        tdvp_mode="2site",
    )
    tdvp(state, H, sim_params)

    state_vec = ref_mps.to_vec()
    H_mat = H.to_matrix()
    U = expm(-1j * sim_params.dt * H_mat)
    state_vec = U @ state_vec
    found = state.to_vec()
    assert np.allclose(state_vec, found)


def _digital_two_site_rzz_fidelity(
    length: int,
    theta: float,
    *,
    sites: tuple[int, int],
    initial: str = "x+",
) -> float:
    """Fidelity of ``tdvp_mode='2site'`` digital TDVP against Qiskit for one RZZ gate."""
    qc = QuantumCircuit(length)
    qc.h(range(length))
    qc.rzz(theta, sites[0], sites[1])
    reference = np.asarray(Statevector(qc).data, dtype=np.complex128)

    prep = deepcopy(State(length, initial=initial).mps)
    gate = GateLibrary.rzz([theta])
    gate.set_sites(*sites)
    mpo, first_site, last_site = construct_generator_mpo(gate, length)
    window_state, window_mpo, _window = apply_window(prep, mpo, first_site, last_site, 1)
    sim_params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=None,
        tdvp_sweeps=1,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
        tdvp_mode="2site",
    )
    tdvp(window_state, window_mpo, sim_params)
    return float(abs(np.vdot(reference, window_state.to_vec())) ** 2)


@pytest.mark.parametrize("theta", [0.1, 0.3, np.pi / 4])
def test_two_site_digital_rzz_l2_plus_exact(theta: float) -> None:
    """Two-site digital TDVP is exact on ``L=2``, ``|++⟩``, ``RZZ(0,1)`` without truncation."""
    fidelity = _digital_two_site_rzz_fidelity(2, theta, sites=(0, 1))
    assert fidelity == pytest.approx(1.0, abs=1e-12)


def test_two_site_digital_rzz_l2_haar_exact() -> None:
    """Two-site digital TDVP is exact on ``L=2`` Haar random states."""
    length = 2
    prep = MPS(length, state="haar-random")
    prep.normalize()
    theta = 0.3

    qc = QuantumCircuit(length)
    qc.initialize(prep.to_vec().tolist(), range(length))
    qc.rzz(theta, 0, 1)
    reference = np.asarray(Statevector(qc).data, dtype=np.complex128)

    gate = GateLibrary.rzz([theta])
    gate.set_sites(0, 1)
    mpo, first_site, last_site = construct_generator_mpo(gate, length)
    window_state, window_mpo, _window = apply_window(deepcopy(prep), mpo, first_site, last_site, 1)
    sim_params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=None,
        tdvp_sweeps=1,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
        tdvp_mode="2site",
    )
    tdvp(window_state, window_mpo, sim_params)
    fidelity = float(abs(np.vdot(reference, window_state.to_vec())) ** 2)
    assert fidelity == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("gate_name", ["rzz", "rxx", "ryy"])
@pytest.mark.parametrize("theta", [0.1, 0.3, np.pi / 4])
def test_two_site_digital_pauli_pair_l2_plus_exact(gate_name: str, theta: float) -> None:
    """Two-site digital TDVP is exact for diagonal Pauli pair gates on ``L=2``."""
    length = 2
    qc = QuantumCircuit(length)
    qc.h(range(length))
    if gate_name == "rzz":
        qc.rzz(theta, 0, 1)
        gate = GateLibrary.rzz([theta])
    elif gate_name == "rxx":
        qc.rxx(theta, 0, 1)
        gate = GateLibrary.rxx([theta])
    else:
        qc.ryy(theta, 0, 1)
        gate = GateLibrary.ryy([theta])
    reference = np.asarray(Statevector(qc).data, dtype=np.complex128)

    prep = deepcopy(State(length, initial="x+").mps)
    gate.set_sites(0, 1)
    mpo, first_site, last_site = construct_generator_mpo(gate, length)
    window_state, window_mpo, _window = apply_window(prep, mpo, first_site, last_site, 1)
    sim_params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=None,
        tdvp_sweeps=1,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
        tdvp_mode="2site",
    )
    tdvp(window_state, window_mpo, sim_params)
    fidelity = float(abs(np.vdot(reference, window_state.to_vec())) ** 2)
    assert fidelity == pytest.approx(1.0, abs=1e-12)
    assert window_state.norm() == pytest.approx(1.0, abs=1e-12)


def test_two_site_l2_rzz_applies_unit_evolution_time() -> None:
    """One digital 2TDVP substep on ``L=2`` must apply total generator time 1, not 2."""
    theta = 0.3
    prep = deepcopy(State(2, initial="x+").mps)
    gate = GateLibrary.rzz([theta])
    gate.set_sites(0, 1)
    mpo, first_site, last_site = construct_generator_mpo(gate, 2)
    window_state, window_mpo, _window = apply_window(prep, mpo, first_site, last_site, 1)
    sim_params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=None,
        tdvp_sweeps=1,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
        tdvp_mode="2site",
    )

    recorded_dts: list[float] = []

    def _record_update_site(
        left_env: NDArray[np.complex128],
        right_env: NDArray[np.complex128],
        op: NDArray[np.complex128],
        ket: NDArray[np.complex128],
        dt: float,
        *,
        krylov_tol: float,
    ) -> NDArray[np.complex128]:
        recorded_dts.append(dt)
        return update_site(left_env, right_env, op, ket, dt, krylov_tol=krylov_tol)

    with patch("mqt.yaqs.core.methods.tdvp.integrators.update_site", side_effect=_record_update_site):
        tdvp(window_state, window_mpo, sim_params)

    assert recorded_dts == [pytest.approx(1.0)]


def test_lr_rzz_sweep_improves_fidelity() -> None:
    """More ``tdvp_sweeps`` should improve long-range RZZ accuracy on ``|+⟩^{⊗L}``."""
    length = 6
    theta = 0.3
    gate = GateLibrary.rzz([theta])
    gate.set_sites(0, length - 1)
    prep = State(length, initial="x+").mps
    qc = QuantumCircuit(length)
    qc.h(range(length))
    qc.rzz(theta, 0, length - 1)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)

    def _fidelity_mps(out: MPS) -> float:
        return float(abs(np.vdot(ref, out.to_vec())) ** 2)

    fidelities: list[float] = []
    for sweeps in (1, 16, 64):
        out = deepcopy(prep)
        params = StrongSimParams(
            preset="exact",
            get_state=True,
            max_bond_dim=None,
            tdvp_sweeps=sweeps,
            svd_threshold=1e-14,
            krylov_tol=1e-12,
        )
        apply_two_qubit_gate_tdvp(out, gate, params)
        fidelities.append(_fidelity_mps(out))

    assert fidelities[-1] > fidelities[0] + 1e-6


def _rzz_lr_ladder_circuit(length: int) -> QuantumCircuit:
    """Build a long-range RZZ ladder circuit for regression tests."""
    qc = QuantumCircuit(length)
    for i in range(length // 2):
        partner = length - 1 - i
        if i < partner:
            qc.rzz(0.3, i, partner)
    return qc


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
