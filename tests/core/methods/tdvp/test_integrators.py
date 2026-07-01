# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for TDVP integrators (analog and sweep orchestration)."""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: N806

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable, StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import GateLibrary, Z
from mqt.yaqs.core.methods.tdvp import tdvp
from mqt.yaqs.core.methods.tdvp.integrators import sweep_2site
from mqt.yaqs.core.methods.tdvp.primitives import update_site
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate_tdvp, apply_window, construct_generator_mpo
from tests.core.methods.tdvp.conftest import (
    EXACT_FID_TOL,
    GateName,
    _apply_lr_gate,
    _double_theta_reference,
    _fidelity,
    _qiskit_two_site_reference,
    _tdvp_params,
    assert_mps_bond_invariants,
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
    canonical_site = state.orthogonality_center
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
    canonical_site = state.orthogonality_center
    assert canonical_site == 0, (
        f"MPS should be site-canonical at site 0 after two-site TDVP, but got canonical site: {canonical_site}"
    )
    state_vec = ref_mps.to_vec()
    H_mat = H.to_matrix()
    U = expm(-1j * sim_params.dt * H_mat)
    state_vec = U @ state_vec
    found = state.to_vec()
    assert np.allclose(state_vec, found)


def test_2site_sweep_scaling() -> None:
    """Circuit-mode 2TDVP passes symmetric substeps when tdvp_sweeps>1."""
    L = 4
    H = MPO.ising(L, 1.0, 0.5)
    state = MPS(L, state="zeros")
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], tdvp_sweeps=2, preset="exact", tdvp_mode="2site")

    with patch("mqt.yaqs.core.methods.tdvp.integrators.sweep_2site") as mock_sweep:
        tdvp(state, H, sim_params)
        assert mock_sweep.call_count == 1
        sweep_plan = mock_sweep.call_args.kwargs["sweep_plan"]
        assert len(sweep_plan) == 2
        assert sweep_plan[0] == pytest.approx(0.5)
        assert sweep_plan[1] == pytest.approx(0.5)


def test_2site_sweep_symmetric() -> None:
    """Circuit tdvp_sweeps=1 uses one symmetric substep, matching analog geometry."""
    L = 4
    H = MPO.ising(L, 1.0, 0.5)
    state = MPS(L, state="zeros")
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], tdvp_sweeps=1, preset="exact", tdvp_mode="2site")

    with patch("mqt.yaqs.core.methods.tdvp.integrators.sweep_2site") as mock_sweep:
        tdvp(state, H, sim_params)
        assert mock_sweep.call_args.kwargs["sweep_plan"] == [1.0]


def test_2site_tdvp_tracks_center_mid_sweep() -> None:
    """Two-site LTR sweep updates the tracked center before exit reset to site 0."""
    L = 4
    state = MPS(L, state="zeros")
    operator = MPO.ising(L, 1.0, 0.5)
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact", tdvp_mode="2site")
    original_update = MPS.update_center_after_split
    seen: list[int | None] = []

    def recording_update(self: MPS, left: int, right: int, dist: str) -> None:
        original_update(self, left, right, dist)
        seen.append(self.orthogonality_center)

    with patch.object(MPS, "update_center_after_split", recording_update):
        sweep_2site(state, operator, sim_params, step_scale=1.0)
    assert any(center not in {None, 0} for center in seen)
    assert state.orthogonality_center == 0


def test_2site_analog_sweep_dt() -> None:
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


# --- circuit 2TDVP exactness (tdvp_regression) ---


def _evolve_l2_two_site(
    prep: MPS,
    gate_name: GateName,
    theta: float,
    *,
    sites: tuple[int, int] = (0, 1),
) -> MPS:
    """Apply one 2site TDVP gate on a length-2 chain.

    Returns:
        Window MPS after one 2-site TDVP evolution step.

    """
    if gate_name == "rzz":
        gate = GateLibrary.rzz([theta])
    elif gate_name == "rxx":
        gate = GateLibrary.rxx([theta])
    else:
        gate = GateLibrary.ryy([theta])
    gate.set_sites(*sites)
    mpo, first_site, last_site = construct_generator_mpo(gate, prep.length)
    window_state, window_mpo, _window = apply_window(deepcopy(prep), mpo, first_site, last_site, 1)
    params = _tdvp_params(max_bond_dim=None, tdvp_sweeps=1)
    params.tdvp_mode = "2site"
    tdvp(window_state, window_mpo, params)
    return window_state


@pytest.mark.tdvp_regression
@pytest.mark.parametrize("gate_name", ["rzz", "rxx", "ryy"])
@pytest.mark.parametrize("theta", [0.1, 0.3, np.pi / 4])
def test_two_site_l2_plus_exact(gate_name: GateName, theta: float) -> None:
    """Two-site TDVP applies one gate on |++⟩, not two."""
    prep = State(2, initial="x+").mps
    out = _evolve_l2_two_site(prep, gate_name, theta)
    ref = _qiskit_two_site_reference(2, gate_name, theta, sites=(0, 1))
    assert _fidelity(ref, out.to_vec()) == pytest.approx(1.0, abs=EXACT_FID_TOL)
    assert out.norm() == pytest.approx(1.0, abs=EXACT_FID_TOL)
    assert_mps_bond_invariants(out)


@pytest.mark.tdvp_regression
def test_two_site_l2_haar_exact() -> None:
    """Two-site TDVP is exact on a deterministic Haar two-qubit state."""
    prep = MPS(2, state="haar-random")
    prep.normalize()
    theta = 0.3
    out = _evolve_l2_two_site(prep, "rzz", theta)

    qc = QuantumCircuit(2)
    qc.initialize(prep.to_vec().tolist(), range(2))
    qc.rzz(theta, 0, 1)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)

    assert _fidelity(ref, out.to_vec()) == pytest.approx(1.0, abs=EXACT_FID_TOL)
    assert out.norm() == pytest.approx(1.0, abs=EXACT_FID_TOL)


@pytest.mark.tdvp_regression
def test_l2_rzz_no_double_theta() -> None:
    """|++⟩ + RZZ(θ) must match RZZ(θ), not the old RZZ(2θ) duplicate-update bug."""
    theta = 0.3
    prep = State(2, initial="x+").mps
    out = _evolve_l2_two_site(prep, "rzz", theta)
    correct = _qiskit_two_site_reference(2, "rzz", theta, sites=(0, 1))
    doubled = _double_theta_reference(2, theta, sites=(0, 1))
    assert _fidelity(correct, out.to_vec()) == pytest.approx(1.0, abs=EXACT_FID_TOL)
    assert _fidelity(doubled, out.to_vec()) < 1.0 - 1e-6


@pytest.mark.tdvp_regression
def test_l2_unit_time() -> None:
    """One 2TDVP substep integrates total generator time 1, not 2."""
    theta = 0.3
    prep = deepcopy(State(2, initial="x+").mps)
    gate = GateLibrary.rzz([theta])
    gate.set_sites(0, 1)
    mpo, first_site, last_site = construct_generator_mpo(gate, 2)
    window_state, window_mpo, _window = apply_window(prep, mpo, first_site, last_site, 1)
    params = _tdvp_params(max_bond_dim=None, tdvp_sweeps=1)
    params.tdvp_mode = "2site"

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
        """Record local site evolution times and delegate to :func:`update_site`.

        Returns:
            Evolved site tensor from :func:`update_site`.

        """
        recorded_dts.append(dt)
        return update_site(left_env, right_env, op, ket, dt, krylov_tol=krylov_tol)

    with patch("mqt.yaqs.core.methods.tdvp.integrators.update_site", side_effect=_record_update_site):
        tdvp(window_state, window_mpo, params)

    assert recorded_dts == [pytest.approx(1.0)]


@pytest.mark.tdvp_regression
def test_apply_lr_gate_supports_ryy() -> None:
    """Long-range helper accepts RYY in addition to RZZ/RXX."""
    prep = State(2, initial="x+").mps
    out = _apply_lr_gate(prep, "ryy", 0.2, max_bond_dim=2, sweeps=1)
    assert out.norm() == pytest.approx(1.0, abs=1e-10)


@pytest.mark.tdvp_regression
def test_dynamic_sweep_plan() -> None:
    """tdvp_sweeps=N produces N symmetric substeps summing to unit evolution time."""
    length = 6
    gate = GateLibrary.rzz([0.3])
    gate.set_sites(0, length - 1)
    prep = deepcopy(State(length, initial="x+").mps)
    params = _tdvp_params(max_bond_dim=8, tdvp_sweeps=16)

    with patch("mqt.yaqs.core.methods.tdvp.integrators.sweep_2site") as mock_two:
        mock_two.side_effect = lambda *_args, **_kwargs: None
        apply_two_qubit_gate_tdvp(prep, gate, params)
        sweep_plan = mock_two.call_args.kwargs["sweep_plan"]
        assert len(sweep_plan) == 16
        assert sum(sweep_plan) == pytest.approx(1.0)


def test_1site_analog_sweep_plan_integration() -> None:
    """One-site analog TDVP honors tdvp_sweeps via recursive sweep_plan dispatch."""
    length = 4
    hamiltonian = MPO.ising(length, 1.0, 0.5)
    state = MPS(length, state="zeros")
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        sample_timesteps=False,
        krylov_tol=1e-12,
        preset="exact",
        tdvp_sweeps=2,
        tdvp_mode="1site",
    )
    tdvp(state, hamiltonian, sim_params)
    reference = deepcopy(MPS(length, state="zeros"))
    single = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        sample_timesteps=False,
        krylov_tol=1e-12,
        preset="exact",
        tdvp_sweeps=1,
        tdvp_mode="1site",
    )
    tdvp(reference, hamiltonian, single)
    assert _fidelity(reference.to_vec(), state.to_vec()) > 0.99


@pytest.mark.tdvp_regression
def test_dynamic_ising_matches_expm() -> None:
    """Dynamic TDVP integrates a capped Ising chain against an expm reference."""
    length = 4
    dt = 0.1
    hamiltonian = MPO.ising(length, 1.0, 0.5)
    prep = MPS(length, state="x+")
    state = deepcopy(prep)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=dt,
        dt=dt,
        sample_timesteps=False,
        krylov_tol=1e-12,
        preset="exact",
        max_bond_dim=2,
        tdvp_sweeps=1,
        tdvp_mode="dynamic",
    )
    tdvp(state, hamiltonian, sim_params)
    exact_vec = expm(-1j * dt * hamiltonian.to_matrix()) @ prep.to_vec()
    assert _fidelity(exact_vec, state.to_vec()) > 0.5
    assert state.get_max_bond() <= 2


@pytest.mark.tdvp_regression
def test_dynamic_analog_sweep_plan_integration() -> None:
    """Dynamic analog TDVP honors tdvp_sweeps without mocking sweep_dynamic."""
    length = 4
    hamiltonian = MPO.ising(length, 1.0, 0.5)
    state = MPS(length, state="zeros")
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        sample_timesteps=False,
        krylov_tol=1e-12,
        preset="exact",
        max_bond_dim=2,
        tdvp_sweeps=2,
        tdvp_mode="dynamic",
    )
    tdvp(state, hamiltonian, sim_params)
    ref = deepcopy(MPS(length, state="zeros"))
    single = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        sample_timesteps=False,
        krylov_tol=1e-12,
        preset="exact",
        max_bond_dim=2,
        tdvp_sweeps=1,
        tdvp_mode="dynamic",
    )
    tdvp(ref, hamiltonian, single)
    assert _fidelity(ref.to_vec(), state.to_vec()) > 0.99


@pytest.mark.tdvp_regression
def test_fixed_chi_2site_capped_sweep() -> None:
    """Fixed-χ two-site TDVP runs on a capped digital window evolution."""
    length = 4
    gate = GateLibrary.rzz([0.2])
    gate.set_sites(0, length - 1)
    prep = deepcopy(State(length, initial="x+").mps)
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        gate_mode="full-tdvp",
        max_bond_dim=2,
        tdvp_sweeps=1,
        tdvp_mode="2site",
        svd_threshold=1e-10,
        krylov_tol=1e-12,
    )
    apply_two_qubit_gate_tdvp(prep, gate, params)
    assert prep.get_max_bond() <= 2
    assert prep.norm() == pytest.approx(1.0, abs=1e-8)


@pytest.mark.tdvp_regression
def test_dynamic_digital_fixed_chi_renorm() -> None:
    """Fixed-χ dynamic TDVP on a digital window runs end-of-sweep drift renorm."""
    length = 4
    gate = GateLibrary.rzz([0.2])
    gate.set_sites(0, length - 1)
    prep = deepcopy(State(length, initial="x+").mps)
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=2,
        tdvp_sweeps=1,
        tdvp_mode="dynamic",
        svd_threshold=1e-10,
        krylov_tol=1e-12,
    )
    mpo, first_site, last_site = construct_generator_mpo(gate, length)
    short_state, short_mpo, _window = apply_window(prep, mpo, first_site, last_site, 1)
    tdvp(short_state, short_mpo, params)
    assert short_state.get_max_bond() <= 2
    assert short_state.norm() == pytest.approx(1.0, abs=1e-8)
