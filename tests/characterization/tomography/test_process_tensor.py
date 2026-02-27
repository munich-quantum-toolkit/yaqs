# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for the ProcessTensor class (comb-Υ reconstruction + QMI)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import itertools

import numpy as np
import pytest

from mqt.yaqs.characterization.tomography.process_tensor import ProcessTensor, _vec_to_rho  # noqa: PLC2701

if TYPE_CHECKING:
    from numpy.typing import NDArray


# -------------------------
# Helpers matching production tomography basis
# -------------------------
def _basis_states_4() -> list[NDArray[np.complex128]]:
    """The 4 single-qubit density matrices used in the tomography module."""
    psi_0 = np.array([1, 0], dtype=np.complex128)
    psi_1 = np.array([0, 1], dtype=np.complex128)
    psi_plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    psi_i_plus = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)

    states = [psi_0, psi_1, psi_plus, psi_i_plus]
    return [np.outer(psi, psi.conj()) for psi in states]


def _choi_basis_16() -> tuple[list[NDArray[np.complex128]], list[tuple[int, int]]]:
    """Your basis CP maps A_{p,m} with Choi matrices B_{p,m} = rho_p ⊗ E_m^T."""
    rhos = _basis_states_4()
    choi = []
    idx = []
    for p, rho_p in enumerate(rhos):
        for m, E_m in enumerate(rhos):
            choi.append(np.kron(rho_p, E_m.T))
            idx.append((p, m))
    return choi, idx


def _dual_from_basis(basis: list[NDArray[np.complex128]]) -> list[NDArray[np.complex128]]:
    """Same as calculate_dual_choi_basis in tomography module."""
    dim = basis[0].shape[0]
    frame = np.column_stack([b.reshape(-1) for b in basis])  # (16,16)
    pinv = np.linalg.pinv(frame)
    dual_frame = pinv.conj().T
    return [dual_frame[:, k].reshape(dim, dim) for k in range(dual_frame.shape[1])]


def _check_duality(duals: list[NDArray[np.complex128]], basis: list[NDArray[np.complex128]], atol: float = 1e-10) -> None:
    for i in range(16):
        for j in range(16):
            inner = np.trace(duals[i].conj().T @ basis[j])
            expected = 1.0 if i == j else 0.0
            assert np.isclose(inner, expected, atol=atol), f"<D{i}|B{j}>={inner} expected {expected}"


def _partial_trace_dense(r: NDArray[np.complex128], dims: list[int], keep: list[int]) -> NDArray[np.complex128]:
    """Partial trace keeping subsystems in keep, matches ProcessTensor implementation."""
    keep = sorted(keep)
    n = len(dims)
    reshaped = r.reshape(*dims, *dims)
    trace_out = [i for i in range(n) if i not in keep]
    perm = keep + trace_out
    reshaped = reshaped.transpose(*(perm + [i + n for i in perm]))
    dim_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    dim_out = int(np.prod([dims[i] for i in trace_out])) if trace_out else 1
    reshaped = reshaped.reshape(dim_keep, dim_out, dim_keep, dim_out)
    return np.einsum("a b c b -> a c", reshaped)


# -------------------------
# Existing tests that still apply
# -------------------------
def test_vec_to_rho() -> None:
    """Test the vector to density matrix conversion."""
    psi0 = np.array([1, 0], dtype=complex)
    rho0 = np.outer(psi0, psi0.conj())
    rho_out = _vec_to_rho(rho0.reshape(-1))
    np.testing.assert_allclose(rho_out, rho0, atol=1e-15)

    psi_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho_plus = np.outer(psi_plus, psi_plus.conj())
    rho_out = _vec_to_rho(rho_plus.reshape(-1))
    np.testing.assert_allclose(rho_out, rho_plus, atol=1e-15)

    vec_unnorm = np.array([2, 0, 0, 0], dtype=complex)
    rho_out = _vec_to_rho(vec_unnorm)
    assert np.isclose(np.trace(rho_out), 1.0)
    assert np.isclose(rho_out[0, 0], 1.0)


# -------------------------
# New: Υ reconstruction + comb QMI tests
# -------------------------
def test_reconstruct_comb_choi_shape_and_psd_k1() -> None:
    """For k=1, Υ should be a (2*4) x (2*4) = 8x8 Hermitian PSD operator (up to numerical tolerance)."""
    choi_basis, choi_indices = _choi_basis_16()
    duals = _dual_from_basis(choi_basis)
    _check_duality(duals, choi_basis)

    # Synthetic PT: output is constant maximally mixed regardless of alpha (memoryless table)
    rho_out = 0.5 * np.eye(2, dtype=np.complex128)
    tensor = np.zeros((4, 16), dtype=np.complex128)
    for a in range(16):
        tensor[:, a] = rho_out.reshape(-1)

    weights = np.ones((16,), dtype=np.float64) / 16.0
    pt = ProcessTensor(tensor, weights, [1.0], duals, choi_indices, choi_basis=choi_basis)

    U = pt.reconstruct_comb_choi(check=True, atol=1e-8)
    assert U.shape == (8, 8)

    # Hermitian
    np.testing.assert_allclose(U, U.conj().T, atol=1e-10)

    # PSD (allow tiny numerical negatives)
    lam_min = float(np.linalg.eigvalsh(U).min().real)
    assert lam_min > -1e-8


def test_reconstruct_comb_choi_self_consistency_k2() -> None:
    """For k=2, the reconstructed Υ should reproduce the stored rho_out(alpha1,alpha2) under the forward contraction."""
    choi_basis, choi_indices = _choi_basis_16()
    duals = _dual_from_basis(choi_basis)
    _check_duality(duals, choi_basis)

    # Synthetic PT: depends only on the *first* index (so it has "one-step memory structure")
    # rho(alpha1,alpha2) = rho0 if alpha1 even else rho1
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    rho1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

    tensor = np.zeros((4, 16, 16), dtype=np.complex128)
    for a1, a2 in itertools.product(range(16), repeat=2):
        tensor[:, a1, a2] = (rho0 if (a1 % 2 == 0) else rho1).reshape(-1)

    weights = np.ones((16, 16), dtype=np.float64) / (16.0 * 16.0)
    pt = ProcessTensor(tensor, weights, [1.0, 1.0], duals, choi_indices, choi_basis=choi_basis)

    U = pt.reconstruct_comb_choi(check=True, atol=1e-8)
    assert U.shape == (32, 32)

    # Reproduce stored outputs using the SAME contraction rule as reconstruct_comb_choi uses internally:
    # rho_pred = Tr_past[ Υ (I ⊗ past^T) ] with past = B_{a1} ⊗ B_{a2}
    dim_p = 4**2
    U4 = U.reshape(2, dim_p, 2, dim_p)

    def rho_pred(a1: int, a2: int) -> NDArray[np.complex128]:
        past = np.kron(choi_basis[a1], choi_basis[a2])
        ins = past.T.reshape(dim_p, dim_p)
        return np.einsum("s p a q, q p -> s a", U4, ins)

    # Test a handful
    for a1, a2 in [(0, 0), (1, 7), (2, 3), (15, 15)]:
        r_true = tensor[:, a1, a2].reshape(2, 2)
        r_hat = rho_pred(a1, a2)
        np.testing.assert_allclose(r_hat, r_true, atol=1e-6)


def test_comb_qmi_bounds_and_zero_for_product_like_case() -> None:
    """comb_qmi_from_upsilon should be >=0 and bounded by 2 bits (since F is a qubit). It should be ~0 for a product Υ."""
    choi_basis, choi_indices = _choi_basis_16()
    duals = _dual_from_basis(choi_basis)
    _check_duality(duals, choi_basis)

    # Build a *product* Υ directly by making outputs independent of alpha and equal to a pure state |0><0|.
    # Intuition: Υ ≈ |0><0| ⊗ (something on past), so I(P:F) ~ 0.
    rho_out = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)

    tensor = np.zeros((4, 16, 16), dtype=np.complex128)
    for a1, a2 in itertools.product(range(16), repeat=2):
        tensor[:, a1, a2] = rho_out.reshape(-1)

    weights = np.ones((16, 16), dtype=np.float64) / (16.0 * 16.0)
    pt = ProcessTensor(tensor, weights, [1.0, 1.0], duals, choi_indices, choi_basis=choi_basis)

    qmi = pt.comb_qmi_from_upsilon(base=2, past="all", normalize=True, check_psd=True)
    assert qmi >= -1e-10  # numeric
    assert qmi <= 2.0 + 1e-10

    # This should be very small if Υ factorizes (up to numeric error / convention).
    assert qmi < 1e-6

# -------------------------
# Helpers: basis and duals (match your tomography module)
# -------------------------
def _basis_states_4() -> list[NDArray[np.complex128]]:
    psi_0 = np.array([1, 0], dtype=np.complex128)
    psi_1 = np.array([0, 1], dtype=np.complex128)
    psi_plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    psi_i_plus = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
    states = [psi_0, psi_1, psi_plus, psi_i_plus]
    return [np.outer(psi, psi.conj()) for psi in states]


def _choi_basis_16() -> tuple[list[NDArray[np.complex128]], list[tuple[int, int]]]:
    rhos = _basis_states_4()
    choi_basis: list[NDArray[np.complex128]] = []
    choi_indices: list[tuple[int, int]] = []
    for p, rho_p in enumerate(rhos):
        for m, E_m in enumerate(rhos):
            choi_basis.append(np.kron(rho_p, E_m.T))  # B_{p,m} = rho_p ⊗ E_m^T
            choi_indices.append((p, m))
    return choi_basis, choi_indices


def _dual_from_basis(basis: list[NDArray[np.complex128]]) -> list[NDArray[np.complex128]]:
    dim = basis[0].shape[0]  # 4
    frame = np.column_stack([b.reshape(-1) for b in basis])  # (16,16)
    pinv = np.linalg.pinv(frame)
    dual_cols = pinv.conj().T
    return [dual_cols[:, k].reshape(dim, dim) for k in range(dual_cols.shape[1])]


def _check_duality(duals: list[NDArray[np.complex128]], basis: list[NDArray[np.complex128]], atol: float = 1e-10) -> None:
    for i in range(16):
        for j in range(16):
            inner = np.trace(duals[i].conj().T @ basis[j])
            expected = 1.0 if i == j else 0.0
            assert np.isclose(inner, expected, atol=atol), f"<D{i}|B{j}>={inner} expected {expected}"


def _partial_trace(r: NDArray[np.complex128], dims: list[int], keep: list[int]) -> NDArray[np.complex128]:
    """Partial trace keeping subsystems in keep. dims lists subsystem dims in order."""
    keep = sorted(keep)
    n = len(dims)
    if any(i < 0 or i >= n for i in keep):
        raise ValueError("keep indices out of range")

    reshaped = r.reshape(*dims, *dims)  # (i0.., j0..)
    trace_out = [i for i in range(n) if i not in keep]
    perm = keep + trace_out
    reshaped = reshaped.transpose(*(perm + [i + n for i in perm]))

    dim_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    dim_out = int(np.prod([dims[i] for i in trace_out])) if trace_out else 1
    reshaped = reshaped.reshape(dim_keep, dim_out, dim_keep, dim_out)

    return np.einsum("a b c b -> a c", reshaped)


def _normalize_density(U: NDArray[np.complex128]) -> NDArray[np.complex128]:
    U = 0.5 * (U + U.conj().T)
    tr = np.trace(U)
    assert abs(tr) > 1e-15
    return U / tr


# -------------------------
# Core tests
# -------------------------
def test_reconstruct_upsilon_is_hermitian_and_psd_k2() -> None:
    """Υ should be Hermitian PSD (numerically) for a simple synthetic PT."""
    choi_basis, choi_indices = _choi_basis_16()
    duals = _dual_from_basis(choi_basis)
    _check_duality(duals, choi_basis)

    # k=2 synthetic tensor: constant maximally mixed output regardless of α
    rho_out = 0.5 * np.eye(2, dtype=np.complex128)
    tensor = np.zeros((4, 16, 16), dtype=np.complex128)
    for a1, a2 in itertools.product(range(16), repeat=2):
        tensor[:, a1, a2] = rho_out.reshape(-1)

    weights = np.ones((16, 16), dtype=np.float64) / (16.0 * 16.0)
    pt = ProcessTensor(tensor, weights, [0.0, 0.0], duals, choi_indices, choi_basis=choi_basis)

    U = pt.reconstruct_comb_choi(check=True, atol=1e-8)
    assert U.shape == (32, 32)

    np.testing.assert_allclose(U, U.conj().T, atol=1e-10)
    lam_min = float(np.linalg.eigvalsh(U).min().real)
    assert lam_min > -1e-8


def test_reconstruct_upsilon_self_consistency_k2() -> None:
    """Υ should reproduce stored outputs via the same forward contraction used in reconstruct_comb_choi()."""
    choi_basis, choi_indices = _choi_basis_16()
    duals = _dual_from_basis(choi_basis)
    _check_duality(duals, choi_basis)

    # k=2 synthetic: depends only on first index parity (easy, but nontrivial)
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    rho1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

    tensor = np.zeros((4, 16, 16), dtype=np.complex128)
    for a1, a2 in itertools.product(range(16), repeat=2):
        tensor[:, a1, a2] = (rho0 if (a1 % 2 == 0) else rho1).reshape(-1)

    weights = np.ones((16, 16), dtype=np.float64) / (16.0 * 16.0)
    pt = ProcessTensor(tensor, weights, [0.0, 0.0], duals, choi_indices, choi_basis=choi_basis)

    U = pt.reconstruct_comb_choi(check=True, atol=1e-8)

    # Forward contraction consistent with your reconstruct_comb_choi() doc:
    # rho = Tr_past[ Υ (I ⊗ past^T) ], where past = B_{a1} ⊗ B_{a2}
    dim_p = 4**2
    U4 = U.reshape(2, dim_p, 2, dim_p)

    def rho_pred(a1: int, a2: int) -> NDArray[np.complex128]:
        past = np.kron(choi_basis[a1], choi_basis[a2])
        ins = past.T.reshape(dim_p, dim_p)
        return np.einsum("s p q r, r p -> s q", U4, ins)

    for a1, a2 in [(0, 0), (1, 7), (2, 3), (15, 15), (8, 9)]:
        r_true = tensor[:, a1, a2].reshape(2, 2)
        r_hat = rho_pred(a1, a2)
        np.testing.assert_allclose(r_hat, r_true, atol=1e-6)


def test_comb_qmi_zero_for_trivial_product_case_k2() -> None:
    """If outputs are independent of α, Υ should factorize (approximately) and comb QMI should be ~0."""
    choi_basis, choi_indices = _choi_basis_16()
    duals = _dual_from_basis(choi_basis)
    _check_duality(duals, choi_basis)

    # Constant pure output |0><0| for all α => Υ should look like |0><0| ⊗ Ω_past
    rho_out = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    tensor = np.zeros((4, 16, 16), dtype=np.complex128)
    for a1, a2 in itertools.product(range(16), repeat=2):
        tensor[:, a1, a2] = rho_out.reshape(-1)

    weights = np.ones((16, 16), dtype=np.float64) / (16.0 * 16.0)
    pt = ProcessTensor(tensor, weights, [0.0, 0.0], duals, choi_indices, choi_basis=choi_basis)

    qmi = pt.comb_qmi_from_upsilon(base=2, past="all", normalize=True, check_psd=True)
    assert qmi >= -1e-10
    assert qmi <= 2.0 + 1e-10
    assert qmi < 1e-6


def test_factorization_error_small_for_trivial_case_k2() -> None:
    """Directly test factorization at the density-operator level for trivial tensor."""
    choi_basis, choi_indices = _choi_basis_16()
    duals = _dual_from_basis(choi_basis)

    # Constant maximally mixed output
    rho_out = 0.5 * np.eye(2, dtype=np.complex128)
    tensor = np.zeros((4, 16, 16), dtype=np.complex128)
    for a1, a2 in itertools.product(range(16), repeat=2):
        tensor[:, a1, a2] = rho_out.reshape(-1)

    weights = np.ones((16, 16), dtype=np.float64) / (16.0 * 16.0)
    pt = ProcessTensor(tensor, weights, [0.0, 0.0], duals, choi_indices, choi_basis=choi_basis)

    U = pt.reconstruct_comb_choi(check=True, atol=1e-8)
    rho = _normalize_density(U)

    # dims: [F, step1, step2] = [2, 4, 4]
    dims = [2, 4, 4]
    rho_F = _partial_trace(rho, dims, [0])
    rho_P = _partial_trace(rho, dims, [1, 2])
    rho_prod = np.kron(rho_F, rho_P)

    err = np.linalg.norm(rho - rho_prod)
    assert err < 1e-6

import numpy as np

def _partial_trace(r, dims, keep):
    keep = sorted(keep)
    n = len(dims)
    reshaped = r.reshape(*dims, *dims)
    trace_out = [i for i in range(n) if i not in keep]
    perm = keep + trace_out
    reshaped = reshaped.transpose(*(perm + [i + n for i in perm]))
    dim_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    dim_out = int(np.prod([dims[i] for i in trace_out])) if trace_out else 1
    reshaped = reshaped.reshape(dim_keep, dim_out, dim_keep, dim_out)
    return np.einsum("a b c b -> a c", reshaped)

def _vn_entropy(rho, base=2):
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) < 1e-15:
        return 0.0
    rho = rho / tr
    evals = np.linalg.eigvalsh(rho).real
    evals = np.clip(evals, 0.0, 1.0)
    nz = evals[evals > 1e-15]
    if nz.size == 0:
        return 0.0
    return float(-(nz * (np.log(nz) / np.log(base))).sum())

import pytest
import numpy as np

@pytest.mark.slow
def test_simulated_zero_time_upsilon_is_physical() -> None:
    from mqt.yaqs.core.data_structures.networks import MPO
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
    from mqt.yaqs.characterization.tomography import run

    op = MPO.ising(length=2, J=0.0, g=0.0)

    params = AnalogSimParams(dt=0.1, max_bond_dim=8, order=1)
    params.show_progress = False
    params.get_state = True

    pt = run(
        operator=op,
        sim_params=params,
        timesteps=[0.0, 0.0],
        num_trajectories=1,
        noise_model=None,
    )

    U = pt.reconstruct_comb_choi(check=True, atol=1e-8)
    U = 0.5 * (U + U.conj().T)

    # Hermitian
    np.testing.assert_allclose(U, U.conj().T, atol=1e-10)

    # PSD up to numerical tolerance
    lam_min = float(np.linalg.eigvalsh(U).min().real)
    assert lam_min > -1e-8

@pytest.mark.slow
def test_simulated_zero_time_conditional_memory_vanishes() -> None:
    from mqt.yaqs.core.data_structures.networks import MPO
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
    from mqt.yaqs.characterization.tomography import run

    op = MPO.ising(length=2, J=0.0, g=0.0)

    params = AnalogSimParams(dt=0.1, max_bond_dim=8, order=1)
    params.show_progress = False
    params.get_state = True

    pt = run(
        operator=op,
        sim_params=params,
        timesteps=[0.0, 0.0],
        num_trajectories=1,
        noise_model=None,
    )

    U = pt.reconstruct_comb_choi(check=True, atol=1e-8)
    U = 0.5 * (U + U.conj().T)
    rho = U / np.trace(U)

    # dims order in reconstruct_comb_choi/comb_qmi_from_upsilon: [F, step1, step2] = [2,4,4]
    dims = [2, 4, 4]

    # Define subsystems:
    # A = step1 (index 1), B = F (index 0), C = step2 (index 2)
    # I(A:B|C) = S(AC) + S(BC) - S(C) - S(ABC)
    rho_ABC = rho
    rho_AC = _partial_trace(rho_ABC, dims, keep=[1, 2])
    rho_BC = _partial_trace(rho_ABC, dims, keep=[0, 2])
    rho_C  = _partial_trace(rho_ABC, dims, keep=[2])

    cmi = (
        _vn_entropy(rho_AC, base=2)
        + _vn_entropy(rho_BC, base=2)
        - _vn_entropy(rho_C, base=2)
        - _vn_entropy(rho_ABC, base=2)
    )

    # Numerical tolerance: inversion + eigens can introduce small noise
    assert cmi < 1e-2