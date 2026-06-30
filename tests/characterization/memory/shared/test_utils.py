# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: PLC2701 -- white-box tests import private backend helpers

"""Tests for process-tensor simulation utility helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.shared.utils import (
    _apply_backend_unitary_site_zero,
    _apply_cut_preparation_step,
    _evolve_backend_state,
    _initialize_backend_state,
    _reprepare_backend_state_forced,
    _reprepare_site_zero_forced,
    _reprepare_site_zero_vector_forced,
    _single_qubit_unitary_mapping_basis0_to_ket,
    assemble_state_from_expectations,
    extract_site0_rho,
    make_mcwf_static_context,
    make_zero_psi,
    representation_to_solver,
    resolve_characterizer_representation,
    resolve_stochastic_solver,
)
from mqt.yaqs.core.data_structures.mpo import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.libraries.gate_library import X, Y, Z


def test_make_zero_psi() -> None:
    """Product |0...0> state vector has unit amplitude on the all-zero index."""
    for length in (1, 3, 5):
        psi = make_zero_psi(length)
        assert psi.shape == (2**length,)
        assert psi[0] == 1.0
        assert np.count_nonzero(psi) == 1


def test_initialize_backend_state_mcwf_and_tjm() -> None:
    """Backend state initialization dispatches to ndarray (MCWF) or MPS (TJM)."""
    op = MPO.ising(length=1, J=1.0, g=0.5)

    state_mcwf = _initialize_backend_state(op, solver="MCWF")
    assert isinstance(state_mcwf, np.ndarray)
    assert state_mcwf.shape == (2**op.length,)

    state_tjm = _initialize_backend_state(op, solver="TJM")
    assert isinstance(state_tjm, MPS)
    assert state_tjm.length == op.length


def test_extract_site0_rho_from_mps_and_vector() -> None:
    """Single-qubit density extraction should give a 2x2 PSD matrix with non-negative trace."""
    mps = MPS(length=1, state="zeros")
    rho_mps = extract_site0_rho(mps)
    assert rho_mps.shape == (2, 2)
    assert np.real(np.trace(rho_mps)) >= 0.0

    vec = np.zeros(2, dtype=np.complex128)
    vec[0] = 1.0
    rho_vec = extract_site0_rho(vec)
    np.testing.assert_allclose(rho_vec, np.array([[1.0, 0.0], [0.0, 0.0]]))


def test_assemble_state_from_expectations() -> None:
    """Check that assemble_state_from_expectations inverts simple Pauli expectations for |0>."""
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    rho0 = np.outer(psi0, psi0.conj())

    ex = np.trace(X().matrix @ rho0)
    ey = np.trace(Y().matrix @ rho0)
    ez = np.trace(Z().matrix @ rho0)

    rho_rec = assemble_state_from_expectations({"x": ex, "y": ey, "z": ez})
    np.testing.assert_allclose(rho_rec, rho0, atol=1e-12)


def test_evolve_backend_state_tjm_does_not_mutate_caller_params() -> None:
    """TJM evolution copies AnalogSimParams before toggling get_state."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.05, max_bond_dim=8, order=1)
    state = _initialize_backend_state(op, solver="TJM")
    assert isinstance(state, MPS)
    _evolve_backend_state(state, op, None, params, solver="TJM")
    assert getattr(params, "get_state", False) is False


def test_resolve_characterizer_representation_branches() -> None:
    """Representation resolver covers vector, mps, auto, and invalid inputs."""
    assert resolve_characterizer_representation(2, "vector") == "vector"
    assert resolve_characterizer_representation(2, "mps") == "mps"
    assert resolve_characterizer_representation(4, "auto", vector_max_qubits=10) == "vector"
    assert resolve_characterizer_representation(12, "auto", vector_max_qubits=10) == "mps"
    with pytest.raises(ValueError, match="representation must be"):
        resolve_characterizer_representation(1, "bad")  # ty: ignore[invalid-argument-type]


def test_representation_to_solver_and_resolve_stochastic_solver() -> None:
    """Solver resolution honors explicit solver, representation, and legacy params."""
    assert representation_to_solver("vector") == "MCWF"
    assert representation_to_solver("mps") == "TJM"

    params = AnalogSimParams(dt=0.1)
    assert resolve_stochastic_solver(params, solver="TJM") == "TJM"
    assert resolve_stochastic_solver(params, representation="vector", chain_length=1) == "MCWF"
    with pytest.raises(ValueError, match="chain_length"):
        resolve_stochastic_solver(params, representation="mps")

    class _LegacyParams(AnalogSimParams):
        solver: str = "TJM"

    assert resolve_stochastic_solver(_LegacyParams(dt=0.1)) == "TJM"


def test_reprepare_site_zero_helpers_mcwf_and_mps() -> None:
    """Project+reprepare helpers update MCWF vectors and MPS tensors."""
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    x = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    vec = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

    new_vec, prob = _reprepare_site_zero_vector_forced(vec, x, z)
    assert new_vec.shape == (2,)
    assert 0.0 <= prob <= 1.0

    mps = MPS(length=1, state="zeros")
    prob_mps = _reprepare_site_zero_forced(mps, x, z)
    assert 0.0 <= prob_mps <= 1.0

    out_vec, _ = _reprepare_backend_state_forced(vec, x, z, "MCWF")
    assert isinstance(out_vec, np.ndarray)
    out_mps, _ = _reprepare_backend_state_forced(mps, x, z, "TJM")
    assert isinstance(out_mps, MPS)

    with pytest.raises(TypeError, match="MCWF solver requires"):
        _reprepare_backend_state_forced(mps, x, z, "MCWF")


def test_reset_and_unitary_backend_helpers() -> None:
    """Local unitaries dispatch on MCWF and TJM backends."""
    x = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    u = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    vec = np.zeros(4, dtype=np.complex128)
    vec[0] = 1.0

    mps = MPS(length=2, state="zeros")

    u_vec = _apply_backend_unitary_site_zero(vec, u, "MCWF")
    assert isinstance(u_vec, np.ndarray)
    assert u_vec.shape == (4,)
    u_mps = _apply_backend_unitary_site_zero(mps, u, "TJM")
    assert isinstance(u_mps, MPS)

    u_mat = _single_qubit_unitary_mapping_basis0_to_ket(x)
    np.testing.assert_allclose(u_mat[:, 0], x, atol=1e-12)
    zero_ket = np.zeros(2, dtype=np.complex128)
    u_default = _single_qubit_unitary_mapping_basis0_to_ket(zero_ket)
    assert u_default.shape == (2, 2)


def test_cut_preparation_soft_preserves_entanglement_on_two_qubits() -> None:
    """Multi-qubit cut_preparation reprepares site 0 without resetting other sites."""
    op = MPO.ising(length=2, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, order=1, get_state=True)
    params.elapsed_time = 0.1
    static_ctx = make_mcwf_static_context(op, params, noise_model=None)
    vec = _initialize_backend_state(op, solver="MCWF")
    vec = _evolve_backend_state(vec, op, None, params, "MCWF", static_ctx=static_ctx)
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)

    soft, _ = _apply_cut_preparation_step(vec, plus, "MCWF", chain_length=2)
    assert isinstance(vec, np.ndarray)
    assert isinstance(soft, np.ndarray)
    assert not np.allclose(soft, vec, atol=1e-8)


def test_evolve_backend_state_mcwf_path() -> None:
    """MCWF evolution returns an updated dense state vector."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.05, order=1, get_state=True)
    params.elapsed_time = 0.05
    static_ctx = make_mcwf_static_context(op, params, noise_model=None)
    vec = _initialize_backend_state(op, solver="MCWF")
    assert isinstance(vec, np.ndarray)
    out = _evolve_backend_state(vec, op, None, params, "MCWF", static_ctx=static_ctx)
    assert isinstance(out, np.ndarray)
    assert out.shape == vec.shape

    with pytest.raises(TypeError, match="TJM solver requires"):
        _evolve_backend_state(vec, op, None, params, "TJM")


def test_extract_site0_rho_zero_norm_mps() -> None:
    """Degenerate MPS norm returns a zero reduced density matrix."""
    mps = MPS(length=1, state="zeros")
    mps.tensors[0][:] = 0.0
    rho = extract_site0_rho(mps)
    np.testing.assert_allclose(rho, np.zeros((2, 2), dtype=np.complex128))
