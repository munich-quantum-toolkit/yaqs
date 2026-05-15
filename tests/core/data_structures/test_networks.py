# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for network classes.

This module provides unit tests for the Matrix Product State (MPS) class and its associated methods.
It verifies correct initialization, custom tensor assignment, bond dimension computation, network flipping,
orthogonality center shifting, normalization, observable measurement, and overall validity of MPS objects.
These tests ensure that the MPS class functions as expected in various simulation scenarios.
"""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: N806

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe
import pytest
from qiskit.circuit import QuantumCircuit
from scipy.stats import unitary_group

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.simulation_parameters import (
    AnalogSimParams,
    StrongSimParams,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.libraries.gate_library import BaseGate, Destroy, GateLibrary, Id, X, Z

# ---- single-qubit ops ----
_I2 = np.eye(2, dtype=complex)
_X2 = np.array([[0, 1], [1, 0]], dtype=complex)
_Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z2 = np.array([[1, 0], [0, -1]], dtype=complex)


def _swap_gate_4() -> np.ndarray:
    """Construct the two-qubit SWAP matrix in lexicographic basis.

    Returns:
        np.ndarray: The ``4 x 4`` SWAP matrix.
    """
    return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)


def _permuted_periodic_wrap_gate(gate4: np.ndarray) -> np.ndarray:
    """Permute wrap-ordered two-site gates into merged nearest-neighbor ordering.

    Args:
        gate4: Two-site gate in ``|q_{L-1}, q_0>`` ordering.

    Returns:
        np.ndarray: Gate in merged ordering ``|q_0, q_{L-1}>``.
    """
    p_perm = np.zeros((4, 4), dtype=np.complex128)
    for a in range(2):
        for b in range(2):
            idx_merged = 2 * a + b
            idx_bond = 2 * b + a
            p_perm[idx_bond, idx_merged] = 1.0
    g = np.asarray(gate4, dtype=np.complex128)
    return p_perm.conj().T @ g @ p_perm


def _dense_embed_adjacent_two_site(length: int, site_left: int, gate4: np.ndarray) -> np.ndarray:
    """Embed a two-site gate onto neighboring sites in a dense Hilbert space.

    Args:
        length: Number of qubits.
        site_left: Left site index.
        gate4: Two-site gate matrix.

    Returns:
        np.ndarray: Embedded dense operator.
    """
    left_dim = 2**site_left
    right_dim = 2 ** (length - site_left - 2)
    op4 = np.asarray(gate4, dtype=np.complex128)
    return np.asarray(
        np.kron(np.kron(np.eye(left_dim, dtype=np.complex128), op4), np.eye(right_dim, dtype=np.complex128)),
        dtype=np.complex128,
    )


def _dense_embed_periodic_wrap_two_site(length: int, gate4: np.ndarray) -> np.ndarray:
    """Embed a two-site gate on periodic bond ``(L-1, 0)``.

    Args:
        length: Number of qubits.
        gate4: Two-site gate matrix in wrap ordering.

    Returns:
        np.ndarray: Embedded dense operator.
    """
    g = np.asarray(gate4, dtype=np.complex128)
    if length <= 2:
        return np.asarray(g, dtype=np.complex128)
    dim = 2**length
    sw = _swap_gate_4()
    u_fwd = np.eye(dim, dtype=np.complex128)
    for i in range(length - 2):
        u_fwd = _dense_embed_adjacent_two_site(length, i, sw) @ u_fwd
    g_merged = _permuted_periodic_wrap_gate(g)
    g_nn = _dense_embed_adjacent_two_site(length, length - 2, g_merged)
    return np.asarray(u_fwd.conj().T @ g_nn @ u_fwd, dtype=np.complex128)


def _spin_current_bond_matrix(j_coupling: float) -> np.ndarray:
    """Construct an XY spin-current bond matrix.

    Args:
        j_coupling: XY coupling.

    Returns:
        np.ndarray: ``4 x 4`` bond operator.
    """
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    return 0.25 * j_coupling * (np.kron(x, y) - np.kron(y, x))


def _embed_one_body(op: np.ndarray, length: int, i: int) -> np.ndarray:
    """Embed a single-site operator into a length-L qubit Hilbert space.

    Args:
        op: Local 2x2 operator acting on site i.
        length: Total number of sites.
        i: Site index at which to apply the operator.

    Returns:
        Dense (2**length, 2**length) matrix representing I⊗…⊗op_i⊗…⊗I.
    """
    out = np.array([[1.0]], dtype=complex)
    for k in range(length):
        out = np.kron(out, op if k == i else _I2)
    return out


def _embed_two_body(op1: np.ndarray, op2: np.ndarray, length: int, i: int) -> np.ndarray:
    """Embed a nearest-neighbor two-site operator into a length-L qubit Hilbert space.

    Args:
        op1: Local operator acting on site i.
        op2: Local operator acting on site i+1.
        length: Total number of sites.
        i: Left site index of the two-body term.

    Returns:
        Dense (2**length, 2**length) matrix representing
        I⊗…⊗op1_i⊗op2_{i+1}⊗…⊗I.
    """
    out = np.array([[1.0]], dtype=complex)
    for k in range(length):
        if k == i:
            out = np.kron(out, op1)
        elif k == i + 1:
            out = np.kron(out, op2)
        else:
            out = np.kron(out, _I2)
    return out


def _ising_dense(length: int, j_val: float, g: float) -> np.ndarray:
    """Construct the dense Ising Hamiltonian for an open chain.

    The Hamiltonian is
        H = -J sum_i Z_i Z_{i+1} - g sum_i X_i.

    Args:
        length: Number of sites.
        j_val: Nearest-neighbor coupling strength.
        g: Transverse-field strength.

    Returns:
        Dense (2**length, 2**length) Hamiltonian matrix.
    """
    dim = 2**length
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(length - 1):
        H += (-j_val) * _embed_two_body(_Z2, _Z2, length, i)
    for i in range(length):
        H += (-g) * _embed_one_body(_X2, length, i)

    return H


def _heisenberg_dense(length: int, jx: float, jy: float, jz: float, h: float) -> np.ndarray:
    """Construct the dense Heisenberg Hamiltonian for an open chain.

    The Hamiltonian is
        H = -sum_i (Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1}) - h sum_i Z_i.

    Args:
        length: Number of sites.
        jx: XX coupling strength.
        jy: YY coupling strength.
        jz: ZZ coupling strength.
        h: Longitudinal field strength.

    Returns:
        Dense (2**length, 2**length) Hamiltonian matrix.
    """
    dim = 2**length
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(length - 1):
        H += (-jx) * _embed_two_body(_X2, _X2, length, i)
        H += (-jy) * _embed_two_body(_Y2, _Y2, length, i)
        H += (-jz) * _embed_two_body(_Z2, _Z2, length, i)
    for i in range(length):
        H += (-h) * _embed_one_body(_Z2, length, i)

    return H


def _bose_hubbard_dense(length: int, local_dim: int, omega: float, hopping_j: float, hubbard_u: float) -> np.ndarray:
    """Construct the exact dense Bose-Hubbard Hamiltonian for comparison.

    Returns:
        Dense Hamiltonian matrix.
    """
    # Local operators
    a = Destroy(local_dim).matrix
    adag = Destroy(local_dim).dag().matrix
    n = adag @ a
    id_op = np.eye(local_dim, dtype=complex)

    dim = local_dim**length
    H = np.zeros((dim, dim), dtype=complex)

    # Build H term-by-term using Kronecker products
    def embed(op_list: list[np.ndarray]) -> np.ndarray:
        out = np.array([[1.0]], dtype=complex)
        for op in op_list:
            out = np.kron(out, op)
        return out

    # Onsite terms
    for i in range(length):
        op_list = [id_op] * length
        op_list[i] = omega * n + 0.5 * hubbard_u * (n @ (n - id_op))
        H += embed(op_list)

    # Hopping terms
    for i in range(length - 1):
        # adag_i * a_{i+1}
        op_list1 = [id_op] * length
        op_list1[i] = adag
        op_list1[i + 1] = a
        H += -hopping_j * embed(op_list1)

        # a_i * adag_{i+1}
        op_list2 = [id_op] * length
        op_list2[i] = a
        op_list2[i + 1] = adag
        H += -hopping_j * embed(op_list2)

    return H


def _embed_local_ops(length: int, local_dim: int, site_ops: list[np.ndarray]) -> np.ndarray:
    """Embed a list of single-site operators into the full Hilbert space.

    Returns:
        np.ndarray: The embedded operator on the full chain Hilbert space.
    """
    identity = np.eye(local_dim, dtype=complex)
    op_list = [identity] * length
    for site, op in enumerate(site_ops):
        op_list[site] = op
    out = np.array([[1.0]], dtype=complex)
    for op in op_list:
        out = np.kron(out, op)
    return out


def _fermi_hubbard_1d_fermionic_dense(length: int, t: float, u: float) -> np.ndarray:
    r"""Dense 1D Fermi-Hubbard Hamiltonian matching ``MPO.fermi_hubbard_1d``.

    Uses open boundaries and
    :math:`H = -t \\sum_{i,\\sigma} (c^\\dagger_{i,\\sigma} c_{i+1,\\sigma} + \\mathrm{h.c.})
    + U \\sum_i n_{i,\\uparrow} n_{i,\\downarrow}` on local dimension-4 sites
    (basis :math:`|0\\rangle, |\\!\\downarrow\\rangle, |\\!\\uparrow\\rangle, |\\!\\uparrow\\downarrow\\rangle`).

    Returns:
        np.ndarray: Dense Hamiltonian matrix of shape ``(4**length, 4**length)``.
    """
    local_dim = 4
    c = np.array([[0, 1], [0, 0]], dtype=complex)
    c_dag = np.array([[0, 0], [1, 0]], dtype=complex)
    identity2 = np.eye(2, dtype=complex)
    c_up = np.kron(c, identity2)
    c_down = np.kron(identity2, c)
    c_up_dag = np.kron(c_dag, identity2)
    c_down_dag = np.kron(identity2, c_dag)
    n_up = c_up_dag @ c_up
    n_down = c_down_dag @ c_down
    identity4 = np.eye(local_dim, dtype=complex)
    onsite = u * n_up @ n_down

    dim = local_dim**length
    h = np.zeros((dim, dim), dtype=complex)

    for site in range(length):
        site_ops = [identity4] * length
        site_ops[site] = onsite
        h += _embed_local_ops(length, local_dim, site_ops)

    for site in range(length - 1):
        for c_right, c_left in ((c_up, c_up_dag), (c_down, c_down_dag)):
            hop_ops = [identity4] * length
            hop_ops[site] = c_left
            hop_ops[site + 1] = c_right
            h += -t * _embed_local_ops(length, local_dim, hop_ops)

            hop_ops = [identity4] * length
            hop_ops[site] = c_right
            hop_ops[site + 1] = c_left
            h += -t * _embed_local_ops(length, local_dim, hop_ops)

    return h


def _fermi_hubbard_1d_jordan_wigner_dense(num_orbitals: int, t: float, u: float) -> np.ndarray:
    """Dense JW-transformed Fermi-Hubbard on an interleaved spin chain.

    ``num_orbitals`` must be even. Orbitals are ordered 1↑, 1↓, 2↑, 2↓, ...
    and the Hamiltonian matches the docstring of ``MPO.fermi_hubbard_1d(..., jordan_wigner=True)``.

    Returns:
        np.ndarray: Dense Hamiltonian matrix of shape ``(2**num_orbitals, 2**num_orbitals)``.

    Raises:
        ValueError: If ``num_orbitals`` is odd.
    """
    if num_orbitals % 2 != 0:
        msg = "num_orbitals must be even."
        raise ValueError(msg)

    num_sites = num_orbitals // 2
    dim = 2**num_orbitals
    h = np.zeros((dim, dim), dtype=complex)

    def term(ops: list[tuple[int, np.ndarray]]) -> np.ndarray:
        local = [_I2] * num_orbitals
        for index, op in ops:
            local[index] = op
        return _embed_local_ops(num_orbitals, 2, local)

    for site in range(num_sites):
        up = 2 * site
        down = 2 * site + 1
        h += (u / 4) * term([(up, _I2), (down, _I2)])
        h += -(u / 4) * term([(up, _Z2)])
        h += -(u / 4) * term([(down, _Z2)])
        h += (u / 4) * term([(up, _Z2), (down, _Z2)])

    for site in range(num_sites - 1):
        up = 2 * site
        down = 2 * site + 1
        up_next = 2 * (site + 1)
        h += -(t / 2) * term([(up, _X2), (down, _Z2), (up_next, _X2)])
        h += -(t / 2) * term([(up, _Y2), (down, _Z2), (up_next, _Y2)])

        down_next = 2 * (site + 1) + 1
        h += -(t / 2) * term([(down, _X2), (up_next, _Z2), (down_next, _X2)])
        h += -(t / 2) * term([(down, _Y2), (up_next, _Z2), (down_next, _Y2)])

    return h


def untranspose_block(mpo_tensor: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Reverse the transposition of an MPO tensor.

    MPO tensors are stored in the order (sigma, sigma', row, col). This function transposes
    the tensor to the order (row, col, sigma, sigma') so that the first two indices can be interpreted
    as a block matrix of operators.

    Args:
        mpo_tensor (NDArray[np.complex128]): The MPO tensor in (sigma, sigma', row, col) order.

    Returns:
        NDArray[np.complex128]: The MPO tensor in (row, col, sigma, sigma') order.
    """
    return np.transpose(mpo_tensor, (2, 3, 0, 1))


def crandn(
    size: int | tuple[int, ...],
    *args: int,
    seed: np.random.Generator | int | None = None,
) -> NDArray[np.complex128]:
    """Draw random samples from the standard complex normal distribution.

    Args:
        size: The size/shape of the output array.
        args: Additional dimensions for the output array.
        seed: The seed for the random number generator.

    Returns:
        The array of random complex numbers.
    """
    if isinstance(size, int) and len(args) > 0:
        size = (size, *list(args))
    elif isinstance(size, int):
        size = (size,)
    rng = np.random.default_rng(seed)
    # 1 / sqrt(2) is a normalization factor
    return np.asarray(
        (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2),
        dtype=np.complex128,
    )


def random_mps(shapes: list[tuple[int, int, int]], *, normalize: bool = True) -> MPS:
    """Create a random MPS with the given shapes.

    Args:
        shapes (List[Tuple[int, int, int]]): The shapes of the tensors in the
            MPS.
        normalize (bool): Whether to normalize the MPS.

    Returns:
        MPS: The random MPS.
    """
    tensors = [crandn(shape) for shape in shapes]
    mps = MPS(len(shapes), tensors=tensors)
    if normalize:
        mps.normalize()
    return mps


def _expected_uniform_clipped_bonds(length: int, chi_max: int) -> list[int]:
    """Expected qubit bond-dimension schedule for the Haar-random initializer.

    Args:
        length: Number of sites.
        chi_max: Maximum internal bond dimension.

    Returns:
        A list of bond dimensions ``[chi_0, ..., chi_L]`` with open boundaries.
    """
    bonds = [1]
    bonds.extend(min(chi_max, 2 ** min(i, length - i)) for i in range(1, length))
    bonds.append(1)
    return bonds


rng = np.random.default_rng()

##############################################################################
# Tests for the MPO class
##############################################################################


def test_ising_correct_operator() -> None:
    """Verify that the Ising MPO matches the exact dense Hamiltonian."""
    L = 5
    J = 1.0
    g = 0.5

    mpo = MPO.ising(L, J, g)

    assert mpo.length == L
    assert mpo.physical_dimension == 2
    assert len(mpo.tensors) == L

    assert np.allclose(mpo.to_matrix(), _ising_dense(L, J, g), atol=1e-12)


def test_heisenberg_correct_operator() -> None:
    """Verify that the Heisenberg MPO matches the exact dense Hamiltonian."""
    L = 5
    Jx, Jy, Jz, h = 1.0, 0.5, 0.3, 0.2

    mpo = MPO.heisenberg(L, Jx, Jy, Jz, h)

    assert np.allclose(mpo.to_matrix(), _heisenberg_dense(L, Jx, Jy, Jz, h), atol=1e-12)


def test_bose_hubbard_correct_operator() -> None:
    """Verify that the Bose-Hubbard MPO matches the exact dense Hamiltonian."""
    length = 4
    local_dim = 3  # up to 2 bosons per site
    omega = 0.7
    J = 0.2
    U = 1.3

    mpo = MPO.bose_hubbard(
        length=length,
        local_dim=local_dim,
        omega=omega,
        hopping_j=J,
        hubbard_u=U,
    )

    # Basic checks
    assert mpo.length == length
    assert mpo.physical_dimension == local_dim
    assert len(mpo.tensors) == length
    assert all(t.shape[2] <= 4 and t.shape[3] <= 4 for t in mpo.tensors), "Bond dimension should be 4"

    # Dense comparison
    H_dense = _bose_hubbard_dense(length, local_dim, omega, J, U)
    H_mpo = mpo.to_matrix()
    np.testing.assert_allclose(H_mpo, H_dense, atol=1e-8)


def test_fermi_hubbard_1d_correct_operator() -> None:
    """Verify the fermionic 1D Fermi-Hubbard MPO matches the dense Hamiltonian."""
    length = 3
    u, t = 0.5, 1.0

    mpo = MPO.fermi_hubbard_1d(length, t, u)

    assert mpo.length == length
    assert mpo.physical_dimension == 4
    assert len(mpo.tensors) == length
    assert all(tensor.shape[2] <= 6 and tensor.shape[3] <= 6 for tensor in mpo.tensors)

    h_dense = _fermi_hubbard_1d_fermionic_dense(length, t, u)
    np.testing.assert_allclose(mpo.to_matrix(), h_dense, atol=1e-10)


def test_fermi_hubbard_1d_jordan_wigner_correct_operator() -> None:
    """Verify the JW 1D Fermi-Hubbard MPO matches the dense Pauli Hamiltonian."""
    num_orbitals = 4
    u, t = 0.5, 1.0

    mpo = MPO.fermi_hubbard_1d(num_orbitals, t, u, jordan_wigner=True)

    assert mpo.length == num_orbitals
    assert mpo.physical_dimension == 2
    assert len(mpo.tensors) == num_orbitals
    assert all(tensor.shape[2] <= 16 and tensor.shape[3] <= 16 for tensor in mpo.tensors)

    h_dense = _fermi_hubbard_1d_jordan_wigner_dense(num_orbitals, t, u)
    np.testing.assert_allclose(mpo.to_matrix(), h_dense, atol=1e-10)

    with pytest.raises(ValueError, match=re.escape("length must be an even integer ≥ 2 (ordering: 1↑,1↓,2↑,2↓,...).")):
        MPO.fermi_hubbard_1d(length=5, t=t, u=u, jordan_wigner=True)


def test_fermi_hubbard_1d_length_one() -> None:
    """Verify a single fermionic site MPO matches the dense reference."""
    length = 1
    u, t = 0.5, 1.0

    mpo = MPO.fermi_hubbard_1d(length, t, u)

    assert mpo.length == length
    assert mpo.physical_dimension == 4
    assert mpo.tensors[0].shape == (4, 4, 1, 1)

    h_dense = _fermi_hubbard_1d_fermionic_dense(length, t, u)
    np.testing.assert_allclose(mpo.to_matrix(), h_dense, atol=1e-10)


def test_fermi_hubbard_1d_cross_representation() -> None:
    """Onsite terms agree between fermionic and JW MPOs under the site basis map.

      Hopping terms differ between representations (composite fermionic sites vs JW
    qubit chain), so this test uses ``t=0`` to compare only the interaction part.
    """
    u = 0.5
    for length in (1, 2, 3):
        h_ferm = MPO.fermi_hubbard_1d(length, t=0.0, u=u).to_matrix()
        h_jw = MPO.fermi_hubbard_1d(2 * length, t=0.0, u=u, jordan_wigner=True).to_matrix()
        np.testing.assert_allclose(h_ferm, h_jw, atol=1e-10)


def test_identity() -> None:
    """Test that identity initializes an identity MPO correctly.

    This test checks that an identity MPO has the correct length, physical dimension,
    and that each tensor corresponds to the identity operator.
    """
    mpo = MPO()
    length = 3
    pdim = 2

    mpo.identity(length, physical_dimension=pdim)

    assert mpo.length == length
    assert mpo.physical_dimension == pdim
    assert len(mpo.tensors) == length

    for tensor in mpo.tensors:
        assert tensor.shape == (2, 2, 1, 1)
        assert np.allclose(np.squeeze(tensor), Id().matrix)


def test_finite_state_machine() -> None:
    """Test initializing a custom Hamiltonian MPO using user-provided boundary and inner tensors.

    This test creates random tensors for the left boundary, inner sites, and right boundary,
    initializes the MPO with these using finite_state_machine, and verifies that the tensors
    have the expected shapes and values (after appropriate transposition).
    """
    length = 4
    pdim = 2

    left_bound = rng.random(size=(1, 2, pdim, pdim)).astype(np.complex128)
    inner = rng.random(size=(2, 2, pdim, pdim)).astype(np.complex128)
    right_bound = rng.random(size=(2, 1, pdim, pdim)).astype(np.complex128)

    mpo = MPO()
    mpo.finite_state_machine(length, left_bound, inner, right_bound)

    assert mpo.length == length
    assert len(mpo.tensors) == length

    assert mpo.tensors[0].shape == (pdim, pdim, 1, 2)
    for i in range(1, length - 1):
        assert mpo.tensors[i].shape == (pdim, pdim, 2, 2)
    assert mpo.tensors[-1].shape == (pdim, pdim, 2, 1)

    assert np.allclose(mpo.tensors[0], np.transpose(left_bound, (2, 3, 0, 1)))
    for i in range(1, length - 1):
        assert np.allclose(mpo.tensors[i], np.transpose(inner, (2, 3, 0, 1)))
    assert np.allclose(mpo.tensors[-1], np.transpose(right_bound, (2, 3, 0, 1)))


def test_custom() -> None:
    """Test that custom correctly sets up an MPO from a user-provided list of tensors.

    This test provides a list of tensors for the left boundary, middle, and right boundary,
    initializes the MPO, and checks that the shapes and values of the MPO tensors match the inputs.
    """
    length = 3
    pdim = 2
    tensors = [
        rng.random(size=(1, 2, pdim, pdim)).astype(np.complex128),
        rng.random(size=(2, 2, pdim, pdim)).astype(np.complex128),
        rng.random(size=(2, 1, pdim, pdim)).astype(np.complex128),
    ]

    mpo = MPO()
    mpo.custom(tensors)

    assert mpo.length == length
    assert mpo.physical_dimension == pdim
    assert len(mpo.tensors) == length

    for original, created in zip(tensors, mpo.tensors, strict=False):
        assert original.shape == created.shape
        assert np.allclose(original, created)


def test_from_matrix() -> None:
    """Test that from_matrix() constructs a correct MPO.

    This test constructs a dense Bose-Hubbard Hamiltonian and creates an MPO via from_matrix().
    It checks:
    - reconstruction correctness for Bose-Hubbard
    - random matrices at very large bond dimension
    - random matrices at moderately truncated bond dimension
    - all validation error branches (Codecov)
    """
    rng = np.random.default_rng()

    length = 5
    d = 3  # local dimension
    H = _bose_hubbard_dense(length, d, 0.9, 0.6, 0.2)

    Hmpo = MPO.from_matrix(H, d, 4)
    assert np.allclose(H, Hmpo.to_matrix())

    H = rng.random((d**length, d**length)) + 1j * rng.random((d**length, d**length))
    Hmpo = MPO.from_matrix(H, d, 1_000_000)
    assert np.allclose(H, Hmpo.to_matrix())

    length = 6
    H = rng.random((d**length, d**length)) + 1j * rng.random((d**length, d**length))
    Hmpo = MPO.from_matrix(H, d, 728)
    assert np.max(np.abs(H - Hmpo.to_matrix())) < 1e-2

    mat = np.eye(1)
    with pytest.raises(ValueError, match="Physical dimension d must be > 0"):
        MPO.from_matrix(mat, d=0)

    # non-square matrix
    mat = np.zeros((4, 2))
    with pytest.raises(ValueError, match="Matrix must be square"):
        MPO.from_matrix(mat, d=2)

    # d == 1 but matrix not 1x1
    mat = np.eye(4)
    with pytest.raises(ValueError, match="1x1"):
        MPO.from_matrix(mat, d=1)

    # matrix dimension not a power of d
    mat = np.eye(6)
    with pytest.raises(ValueError, match="not a power"):
        MPO.from_matrix(mat, d=2)

    # inferred n < 1 (log(1)/log(100) = 0)
    mat = np.eye(1)
    with pytest.raises(ValueError, match="invalid"):
        MPO.from_matrix(mat, d=100)


def test_to_mps() -> None:
    """Test converting an MPO to an MPS.

    This test initializes an MPO using ising, converts it to an MPS via to_mps,
    and verifies that the resulting MPS has the correct length and that each tensor has been reshaped
    to the expected dimensions.
    """
    length = 3
    J, g = 1.0, 0.5

    mpo = MPO.ising(length, J, g)
    mps = mpo.to_mps()

    assert isinstance(mps, MPS)
    assert mps.length == length

    for i, tensor in enumerate(mps.tensors):
        original_mpo_tensor = mpo.tensors[i]
        pdim2 = original_mpo_tensor.shape[0] * original_mpo_tensor.shape[1]
        bond_in = original_mpo_tensor.shape[2]
        bond_out = original_mpo_tensor.shape[3]
        assert tensor.shape == (pdim2, bond_in, bond_out)


def test_check_if_valid_mpo() -> None:
    """Test that a valid MPO passes the check_if_valid_mpo method without raising errors.

    This test initializes an Ising MPO and calls check_if_valid_mpo, which should validate the MPO.
    """
    length = 4
    J, g = 1.0, 0.5

    mpo = MPO.ising(length, J, g)
    mpo.check_if_valid_mpo()


def test_rotate() -> None:
    """Test the rotate method for an MPO.

    This test checks that rotating an MPO (without conjugation) transposes each tensor as expected,
    and that rotating back with conjugation returns tensors with the original physical dimensions.
    """
    length = 3
    J, g = 1.0, 0.5

    mpo = MPO.ising(length, J, g)
    original_tensors = [t.copy() for t in mpo.tensors]

    mpo.rotate(conjugate=False)
    for orig, rotated in zip(original_tensors, mpo.tensors, strict=False):
        assert rotated.shape == (
            orig.shape[1],
            orig.shape[0],
            orig.shape[2],
            orig.shape[3],
        )
        np.testing.assert_allclose(rotated, np.transpose(orig, (1, 0, 2, 3)))

    mpo.rotate(conjugate=True)
    for tensor in mpo.tensors:
        assert tensor.shape[0:2] == (2, 2)


def test_check_if_identity() -> None:
    """Test that an identity MPO is recognized as identity by check_if_identity.

    This test initializes an identity MPO and verifies that check_if_identity returns True
    when a fidelity threshold is provided.
    """
    mpo = MPO()
    length = 3
    pdim = 2

    mpo.identity(length, pdim)
    fidelity_threshold = 0.9
    assert mpo.check_if_identity(fidelity_threshold) is True


##############################################################################
# Tests for the MPS class
##############################################################################


@pytest.mark.parametrize("state", ["zeros", "ones", "x+", "x-", "y+", "y-", "Neel", "wall", "basis"])
def test_mps_initialization(state: str) -> None:
    """Test that MPS initializes with the correct chain length, physical dimensions, and tensor shapes.

    This test creates an MPS with 4 sites using a specified default state and verifies that each tensor
    is of rank 3 and has dimensions corresponding to the physical dimension and default bond dimensions.

    Args:
        state (str): The default state to initialize (e.g., "zeros", "ones", "x+", etc.).
    """
    length = 4
    pdim = 2
    basis_string = "1001"

    if state == "basis":
        mps = MPS(
            length=length,
            physical_dimensions=[pdim] * length,
            state=state,
            basis_string=basis_string,
        )
    else:
        mps = MPS(length=length, physical_dimensions=[pdim] * length, state=state)

    assert mps.length == length
    assert len(mps.tensors) == length
    assert all(d == pdim for d in mps.physical_dimensions)

    for i, tensor in enumerate(mps.tensors):
        # Check tensor shape
        assert tensor.ndim == 3
        assert tensor.shape == (pdim, 1, 1)

        # Validate state-specific behavior
        vec = tensor[:, 0, 0]
        if state == "zeros":
            expected = np.array([1, 0], dtype=complex)
            np.testing.assert_allclose(vec, expected)
        elif state == "ones":
            expected = np.array([0, 1], dtype=complex)
            np.testing.assert_allclose(vec, expected)
        elif state == "x+":
            expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
            np.testing.assert_allclose(vec, expected)
        elif state == "x-":
            expected = np.array([1, -1], dtype=complex) / np.sqrt(2)
            np.testing.assert_allclose(vec, expected)
        elif state == "y+":
            expected = np.array([1, 1j], dtype=complex) / np.sqrt(2)
            np.testing.assert_allclose(vec, expected)
        elif state == "y-":
            expected = np.array([1, -1j], dtype=complex) / np.sqrt(2)
            np.testing.assert_allclose(vec, expected)
        elif state == "Neel":
            expected = np.array([1, 0], dtype=complex) if i % 2 else np.array([0, 1], dtype=complex)
            np.testing.assert_allclose(vec, expected)
        elif state == "wall":
            expected = np.array([1, 0], dtype=complex) if i < length // 2 else np.array([0, 1], dtype=complex)
            np.testing.assert_allclose(vec, expected)
        elif state == "basis":
            bit = int(basis_string[i])
            expected = np.zeros(pdim, dtype=complex)
            expected[bit] = 1
            np.testing.assert_allclose(vec, expected)


def test_mps_custom_tensors() -> None:
    """Test that an MPS can be initialized with custom tensors.

    This test provides a list of custom rank-3 tensors for an MPS and verifies that the MPS
    retains these tensors correctly.
    """
    length = 3
    pdim = 2
    t1 = rng.random(size=(pdim, 1, 2)).astype(np.complex128)
    t2 = rng.random(size=(pdim, 2, 2)).astype(np.complex128)
    t3 = rng.random(size=(pdim, 2, 2)).astype(np.complex128)
    tensors = [t1, t2, t3]

    mps = MPS(length=length, tensors=tensors, physical_dimensions=[pdim] * length)
    assert mps.length == length
    assert len(mps.tensors) == length
    for i, tensor in enumerate(mps.tensors):
        assert np.allclose(tensor, tensors[i])


def test_flip_network() -> None:
    """Test the flip_network method of MPS.

    This test reverses the order of the MPS tensors and transposes each tensor's bond dimensions.
    Flipping the network twice should restore the original order and tensor values.
    """
    length = 3
    pdim = 2
    t1 = rng.random(size=(pdim, 1, 2)).astype(np.complex128)
    t2 = rng.random(size=(pdim, 2, 2)).astype(np.complex128)
    t3 = rng.random(size=(pdim, 2, 1)).astype(np.complex128)
    original_tensors = [t1, t2, t3]
    mps = MPS(
        length,
        tensors=copy.deepcopy(original_tensors),
        physical_dimensions=[pdim] * length,
    )

    mps.flip_network()
    flipped_tensors = mps.tensors
    assert len(flipped_tensors) == length
    assert flipped_tensors[0].shape == (
        pdim,
        original_tensors[2].shape[2],
        original_tensors[2].shape[1],
    )
    mps.flip_network()
    for orig, now in zip(original_tensors, mps.tensors, strict=False):
        assert np.allclose(orig, now)


def test_shift_orthogonality_center_right() -> None:
    """Test shifting the orthogonality center to the right in an MPS.

    This test verifies that shifting the orthogonality center does not change the rank of the tensors.
    """
    pdim = 2
    shapes = [(pdim, 1, 2), (pdim, 2, 3), (pdim, 3, 3), (pdim, 3, 1)]
    mps = random_mps(shapes)
    mps.set_canonical_form(0)
    assert mps.check_canonical_form() == [0]
    mps.shift_orthogonality_center_right(current_orthogonality_center=0)
    assert mps.check_canonical_form() == [1]
    mps.shift_orthogonality_center_right(current_orthogonality_center=1)
    assert mps.check_canonical_form() == [2]
    mps.shift_orthogonality_center_right(current_orthogonality_center=2)
    assert mps.check_canonical_form() == [3]


def test_shift_orthogonality_center_left() -> None:
    """Test shifting the orthogonality center to the left in an MPS.

    This test ensures that the left shift operation does not alter the rank (3) of the MPS tensors.
    """
    pdim = 2
    shapes = [(pdim, 1, 2), (pdim, 2, 3), (pdim, 3, 3), (pdim, 3, 1)]
    mps = random_mps(shapes)
    mps.set_canonical_form(3)
    assert mps.check_canonical_form() == [3]
    mps.shift_orthogonality_center_left(current_orthogonality_center=3)
    assert mps.check_canonical_form() == [2]
    mps.shift_orthogonality_center_left(current_orthogonality_center=2)
    assert mps.check_canonical_form() == [1]
    mps.shift_orthogonality_center_left(current_orthogonality_center=1)
    assert mps.check_canonical_form() == [0]


@pytest.mark.parametrize("desired_center", [0, 1, 2, 3])
def test_set_canonical_form(desired_center: int) -> None:
    """Test that set_canonical_form correctly sets the MPS into a canonical form without altering tensor shapes.

    This test initializes an MPS with a default state, applies the canonical form procedure, and checks the
    orthogonality.
    """
    pdim = 2
    shapes = [(pdim, 1, 2), (pdim, 2, 4), (pdim, 4, 3), (pdim, 3, 1)]
    mps = random_mps(shapes)
    mps.set_canonical_form(desired_center)
    assert [desired_center] == mps.check_canonical_form()


def test_normalize() -> None:
    """Test that normalize brings an MPS to unit norm without changing tensor ranks.

    This test normalizes an MPS (using 'B' normalization) and verifies that the overall norm is 1 and
    that all tensors remain rank-3.
    """
    length = 4
    pdim = 2
    t1 = rng.random(size=(pdim, 1, 2)).astype(np.complex128)
    t2 = rng.random(size=(pdim, 2, 3)).astype(np.complex128)
    t3 = rng.random(size=(pdim, 3, 3)).astype(np.complex128)
    t4 = rng.random(size=(pdim, 3, 1)).astype(np.complex128)
    mps = MPS(length, [t1, t2, t3, t4], [pdim] * length)

    mps.normalize(form="B")
    assert np.isclose(mps.norm(), 1)
    for tensor in mps.tensors:
        assert tensor.ndim == 3


def test_scalar_product_same_state() -> None:
    """Test that the scalar product of a normalized state with itself equals 1.

    For a normalized product state (here constructed as an MPS in 'random' state), the inner product
    <psi|psi> should be 1.
    """
    psi_mps = MPS(length=3, state="random")
    val = psi_mps.scalar_product(psi_mps)
    np.testing.assert_allclose(val, 1.0, atol=1e-12)


def test_scalar_product_orthogonal_states() -> None:
    """Test that the scalar product between orthogonal product states is 0.

    This test creates two MPS objects initialized in orthogonal states ("zeros" and "ones")
    and verifies that their inner product is 0.
    """
    psi_mps_0 = MPS(length=3, state="zeros")
    psi_mps_1 = MPS(length=3, state="ones")
    val = psi_mps_0.scalar_product(psi_mps_1)
    np.testing.assert_allclose(val, 0.0, atol=1e-12)


def test_scalar_product_partial_site() -> None:
    """Test the scalar product function when specifying a single site.

    For a given site (here site 0 of a 3-site MPS), the scalar product computed by
    scalar_product should equal the direct contraction of the tensor at that site,
    which for a normalized state is 1.
    """
    psi_mps = MPS(length=3, state="x+")
    site = 0
    partial_val = psi_mps.scalar_product(psi_mps, sites=site)
    np.testing.assert_allclose(partial_val, 1.0, atol=1e-12)


def test_local_expect_z_on_zero_state() -> None:
    """Test the local expectation value of the Z observable on a |0> state.

    For the computational basis state |0>, the expectation value of Z is +1.
    This test verifies that local_expect returns +1 for site 0 and site 1 of a 2-qubit MPS
    initialized in the "zeros" state.
    """
    # Pauli-Z in computational basis.
    z = Observable(Z(), 0)

    psi_mps = MPS(length=2, state="zeros")
    val = psi_mps.local_expect(z, sites=0)
    np.testing.assert_allclose(val, 1.0, atol=1e-12)

    z = Observable(Z(), 1)
    val_site1 = psi_mps.local_expect(z, sites=1)
    np.testing.assert_allclose(val_site1, 1.0, atol=1e-12)


def test_local_expect_x_on_plus_state() -> None:
    """Test the local expectation value of the X observable on a |+> state.

    For the |+> state, defined as 1/√2 (|0> + |1>), the expectation value of the X observable is +1.
    This test verifies that local_expect returns +1 for a single-qubit MPS initialized in the "x+" state.
    """
    x = Observable(X(), 0)
    psi_mps = MPS(length=3, state="x+")
    val = psi_mps.local_expect(x, sites=0)
    np.testing.assert_allclose(val, 1.0, atol=1e-12)


def test_mps_apply_local_l2_periodic_wrap_matches_permuted_nn() -> None:
    """For ``L == 2``, wrap-ordered and permuted NN applications must agree."""
    length = 2
    rng = np.random.default_rng(2026)
    g_random = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))).astype(np.complex128)
    gate4 = (g_random + g_random.conj().T) / 2
    g_merged = _permuted_periodic_wrap_gate(gate4)

    mps_wrap = MPS(length, state="random", pad=8)
    mps_wrap.normalize("B")
    mps_nn = copy.deepcopy(mps_wrap)

    mps_wrap.apply_local(Observable(BaseGate(gate4), sites=[length - 1, 0]))
    mps_nn.apply_local(Observable(BaseGate(g_merged), sites=[0, 1]))

    np.testing.assert_allclose(np.asarray(mps_wrap.to_vec()), np.asarray(mps_nn.to_vec()), atol=1e-9)


def test_mps_apply_local_periodic_wrap_matches_dense_expectation() -> None:
    """Periodic-wrap application should reproduce dense expectation values."""
    length = 5
    j_xy = 1.1
    mps = MPS(length, state="random", pad=16)
    mps.normalize("B")
    psi = np.asarray(mps.to_vec(), dtype=np.complex128)

    j_mat = _spin_current_bond_matrix(j_xy)
    j_dense = _dense_embed_periodic_wrap_two_site(length, j_mat)
    obs = Observable(BaseGate(j_mat), sites=[length - 1, 0])

    mps_with_op = copy.deepcopy(mps)
    mps_with_op.apply_local(obs)

    ex_dense = float(np.real(np.vdot(psi, j_dense @ psi)))
    ex_mps = float(np.real(mps.scalar_product(mps_with_op)))
    assert ex_mps == pytest.approx(ex_dense, rel=0, abs=1e-6)


def test_mps_apply_local_non_adjacent_two_site_raises() -> None:
    """Two-site 4x4 observables must be nearest neighbors (or periodic wrap)."""
    length = 4
    mps = MPS(length, state="random", pad=4)
    mps.normalize("B")
    gate4 = np.eye(4, dtype=np.complex128)
    obs = Observable(BaseGate(gate4), sites=[0, 2])
    with pytest.raises(ValueError, match="Only nearest-neighbor two-site observables are currently implemented"):
        mps.apply_local(obs)


def test_mps_apply_local_unsupported_gate_dimension_raises() -> None:
    """Only one-site (2x2) and two-site (4x4) gates are supported."""
    length = 3
    mps = MPS(length, state="random", pad=4)
    mps.normalize("B")
    obs = Observable(BaseGate(np.eye(8, dtype=np.complex128)), sites=[0, 1, 2])
    with pytest.raises(ValueError, match="Local observable must be one-site or nearest-neighbor two-site"):
        mps.apply_local(obs)


def test_mps_mixed_expectation_l2_periodic_wrap_matches_permuted_nn() -> None:
    """For ``L == 2``, wrap-ordered and permuted NN mixed expectations must agree."""
    length = 2
    rng = np.random.default_rng(2026)
    g_random = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))).astype(np.complex128)
    gate4 = (g_random + g_random.conj().T) / 2
    g_merged = _permuted_periodic_wrap_gate(gate4)

    mps_wrap = MPS(length, state="random", pad=8)
    mps_wrap.normalize("B")
    mps_nn = copy.deepcopy(mps_wrap)

    ex_wrap = mps_wrap.mixed_expectation(mps_nn, Observable(BaseGate(gate4), sites=[length - 1, 0]))
    ex_nn_permuted = mps_nn.mixed_expectation(mps_wrap, Observable(BaseGate(g_merged), sites=[0, 1]))
    assert ex_wrap == pytest.approx(ex_nn_permuted, rel=0, abs=1e-9)


def test_mps_mixed_expectation_periodic_wrap_matches_dense_expectation() -> None:
    """Periodic-wrap mixed expectation should reproduce dense expectation values."""
    length = 5
    j_xy = 1.1
    mps = MPS(length, state="random", pad=16)
    mps.normalize("B")
    psi = np.asarray(mps.to_vec(), dtype=np.complex128)

    j_mat = _spin_current_bond_matrix(j_xy)
    j_dense = _dense_embed_periodic_wrap_two_site(length, j_mat)
    obs = Observable(BaseGate(j_mat), sites=[length - 1, 0])

    ex_dense = float(np.real(np.vdot(psi, j_dense @ psi)))
    ex_mps = float(np.real(mps.mixed_expectation(mps, obs)))
    assert ex_mps == pytest.approx(ex_dense, rel=0, abs=1e-6)


def test_measure() -> None:
    """Test that the measure method of an MPS returns the expected observable value.

    This test creates an MPS initialized in the 'x+' state, measures the X observable on site 0,
    and verifies that the measured value is close to 1.
    """
    length = 2
    pdim = 2
    mps = MPS(length=length, physical_dimensions=[pdim] * length, state="x+")
    obs = Observable(X(), 0)
    val = mps.expect(obs)
    assert np.isclose(val, 1)


def test_single_shot() -> None:
    """Test measure_single_shot on an MPS initialized in the |0> state.

    For an MPS representing the state |0> on all qubits, a single-shot measurement should yield 0.
    """
    psi_mps = MPS(length=3, state="zeros")
    val = psi_mps.measure_single_shot()
    np.testing.assert_allclose(val, 0, atol=1e-12)


def test_single_shot_basis() -> None:
    """Test measure_single_shot with different bases.

    Verify that:
    - x+ state measured in X basis always yields 0.
    - x- state measured in X basis always yields 1.
    - y+ state measured in Y basis always yields 0.
    - y- state measured in Y basis always yields 1.
    """
    # X basis tests
    psi_x_plus = MPS(length=1, state="x+")
    for _ in range(10):
        assert psi_x_plus.measure_single_shot(basis="X") == 0

    psi_x_minus = MPS(length=1, state="x-")
    for _ in range(10):
        assert psi_x_minus.measure_single_shot(basis="X") == 1

    # Y basis tests
    psi_y_plus = MPS(length=1, state="y+")
    for _ in range(10):
        assert psi_y_plus.measure_single_shot(basis="Y") == 0

    psi_y_minus = MPS(length=1, state="y-")
    for _ in range(10):
        assert psi_y_minus.measure_single_shot(basis="Y") == 1


def test_measure_shots_basis() -> None:
    """Test measure_shots with different bases."""
    psi_x_plus = MPS(length=1, state="x+")
    results = psi_x_plus.measure_shots(shots=10, basis="X")
    assert results == {0: 10}

    psi_y_plus = MPS(length=1, state="y+")
    results = psi_y_plus.measure_shots(shots=10, basis="Y")
    assert results == {0: 10}

    # Verify that X measurement on Z state gives 50/50
    psi_zero = MPS(length=1, state="zeros")
    results = psi_zero.measure_shots(shots=100, basis="X")
    assert results.get(0, 0) > 0
    assert results.get(1, 0) > 0
    assert sum(results.values()) == 100


def test_inplace_measure() -> None:
    """Test the in-place .measure(site, basis) method.

    Verify that:
    - Measuring a |+> state in Z basis collapses it to |0> or |1>.
    - Measuring a GHZ-like state |00> + |11> collapses the other site.
    """
    # 1. Single qubit collapse
    psi = MPS(length=1, state="x+")
    psi.normalize(form="B")  # Ensure center is at 0
    outcome = psi.measure(site=0, basis="Z")
    assert outcome in {0, 1}
    # Check that expectation value matches the outcome
    expected_val = 1.0 if outcome == 0 else -1.0
    assert np.isclose(psi.expect(Observable(Z(), 0)), expected_val)

    # 2. GHZ state collapse (2 sites)
    psi = MPS(length=2, state="zeros")
    h_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    psi.tensors[0] = oe.contract("ab, bcd->acd", h_gate, psi.tensors[0])

    a = psi.tensors[0]
    b = psi.tensors[1]

    theta = np.tensordot(a, b, axes=(2, 1))  # (d1, l1, d2, r2) = (2, 1, 2, 1)
    theta = theta.transpose(1, 0, 2, 3)  # (l1, d1, d2, r2)
    theta = theta.reshape(1, 4, 1)
    cx_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    theta = oe.contract("ab, cbd->cad", cx_mat, theta)

    u, s, v = np.linalg.svd(theta.reshape(2, 2), full_matrices=False)
    psi.tensors[0] = u.reshape(2, 1, 2)
    psi.tensors[1] = (np.diag(s) @ v).reshape(2, 2, 1)

    psi.normalize(form="B")
    # State is now (|00> + |11>) / sqrt(2)
    # Measure site 0 in Z
    outcome = psi.measure(site=0, basis="Z")
    assert outcome in {0, 1}

    # After measurement, the state should be |00> or |11>.
    # Verification via state vector is robust to normalization/canonical form.
    vec = psi.to_vec()
    expected_vec = np.array([1, 0, 0, 0]) if outcome == 0 else np.array([0, 0, 0, 1])

    # We might have a global phase or sign depending on SVD
    fidelity = np.abs(np.vdot(vec, expected_vec)) ** 2
    assert np.isclose(fidelity, 1.0)

    # 3. Multiple sites and basis
    psi = MPS(length=3, state="zeros")
    psi.normalize(form="B")
    # Measure site 2 in X basis
    outcome = psi.measure(site=2, basis="X")
    assert outcome in {0, 1}

    assert np.isclose(psi.expect(Observable(X(), 2)), 1.0 if outcome == 0 else -1.0)


def test_multi_shot() -> None:
    """Test measure over multiple shots on an MPS initialized in the |1> state.

    This test performs 10 measurement shots on an MPS in the "ones" state and verifies that
    the measurement result for the corresponding basis state (here, 7) is present, while an unexpected
    key (e.g., 0) should not be present.
    """
    psi_mps = MPS(length=3, state="ones")
    shots_dict = psi_mps.measure_shots(shots=10)
    # Assuming that in the "ones" state the measurement outcome is encoded as 7.
    assert shots_dict[7]
    with pytest.raises(KeyError):
        _ = shots_dict[0]


def test_norm() -> None:
    """Test that the norm of an MPS initialized in the 'zeros' state is 1.

    This test checks the norm method of an MPS.
    """
    length = 3
    pdim = 2
    mps = MPS(length=length, physical_dimensions=[pdim] * length, state="zeros")
    val = mps.norm()
    assert val == 1


def test_check_if_valid_mps() -> None:
    """Test that an MPS with consistent bond dimensions passes the validity check.

    This test creates an MPS with carefully constructed tensors and verifies that check_if_valid_mps
    does not raise an exception.
    """
    length = 3
    pdim = 2
    t1 = rng.random(size=(pdim, 1, 2)).astype(np.complex128)
    t2 = rng.random(size=(pdim, 2, 3)).astype(np.complex128)
    t3 = rng.random(size=(pdim, 3, 1)).astype(np.complex128)
    mps = MPS(length, tensors=[t1, t2, t3], physical_dimensions=[pdim] * length)
    mps.check_if_valid_mps()


def test_check_canonical_form_none() -> None:
    """Tests that no canonical form is detected for an MPS in a non-canonical state."""
    mps = random_mps([(2, 1, 2), (2, 2, 3), (2, 3, 1)], normalize=False)
    res = mps.check_canonical_form()
    assert res == []


def test_check_canonical_form_left() -> None:
    """Test that the left canonical form is detected correctly."""
    unitary_mid = unitary_group.rvs(6).reshape((6, 2, 3)).transpose(1, 0, 2)
    unitary_right = unitary_group.rvs(3).reshape(3, 3, 1)
    tensors = [crandn(2, 1, 6), unitary_mid, unitary_right]
    mps = MPS(length=3, tensors=tensors)
    res = mps.check_canonical_form()
    assert 0 in res


def test_check_canonical_form_right() -> None:
    """Test that the right canonical form is detected correctly."""
    unitary_left = unitary_group.rvs(3).astype(np.complex128).reshape(3, 1, 3)
    unitary_mid = unitary_group.rvs(6).astype(np.complex128).reshape((2, 3, 6))
    tensors = [unitary_left, unitary_mid, crandn(2, 6, 1)]
    mps = MPS(length=3, tensors=tensors)
    res = mps.check_canonical_form()
    assert 2 in res


def test_check_canonical_form_middle() -> None:
    """Test that a site canonical form is detected correctly."""
    unitary_left = unitary_group.rvs(3).astype(np.complex128).reshape(3, 1, 3)
    unitary_right = unitary_group.rvs(3).astype(np.complex128).reshape(3, 3, 1)
    tensors = [unitary_left, crandn(2, 3, 3), unitary_right]
    mps = MPS(length=3, tensors=tensors)
    res = mps.check_canonical_form()
    assert 1 in res


def test_check_canonical_form_full() -> None:
    """Test the very special case that all canonical forms are true."""
    delta_left = np.eye(2, dtype=np.complex128).reshape(2, 1, 2)
    delta_right = np.eye(2, dtype=np.complex128).reshape(2, 2, 1)
    delta_mid = np.zeros((2, 2, 2), dtype=np.complex128)
    delta_mid[0, 0, 0] = np.array(1, dtype=np.complex128)
    delta_mid[1, 1, 1] = np.array(1, dtype=np.complex128)
    tensors = [delta_left, delta_mid, delta_right]
    mps = MPS(length=3, tensors=tensors)
    res = mps.check_canonical_form()
    assert res == [0, 1, 2]


def test_convert_to_vector() -> None:
    """Test convert to vector.

    Tests the MPS_to_vector function for various initial states.
    For each state, the expected full state vector is computed as the tensor
    product of the corresponding local state vectors.
    """
    test_states = ["zeros", "ones", "x+", "x-", "y+", "y-"]
    Length = 4  # Use a small number of sites for testing.
    tol = 1e-12

    for state_str in test_states:
        if state_str == "zeros":
            local_state = np.array([1, 0], dtype=complex)
        if state_str == "ones":
            local_state = np.array([0, 1], dtype=complex)
        if state_str == "x+":
            local_state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        if state_str == "x-":
            local_state = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=complex)
        if state_str == "y+":
            local_state = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)], dtype=complex)
        if state_str == "y-":
            local_state = np.array([1 / np.sqrt(2), -1j / np.sqrt(2)], dtype=complex)

        # Create an MPS for the given state.
        mps = MPS(length=Length, state=state_str)
        psi = mps.to_vec()

        # Construct the expected state vector as the Kronecker product of local states.
        local_states = [local_state for _ in range(Length)]

        expected = np.array(1, dtype=complex)
        for state in local_states:
            expected = np.kron(expected, state)

        assert np.allclose(psi, expected, atol=tol)


def test_convert_to_vector_fidelity() -> None:
    """Test convert to vector.

    Tests the MPS_to_vector function for a circuit input
    """
    num_qubits = 3
    circ = QuantumCircuit(num_qubits)
    circ.h(0)
    circ.cx(0, 1)
    state_vector = np.array([0.70710678, 0, 0, 0.70710678, 0, 0, 0, 0])
    # Define the initial state
    state = MPS(num_qubits, state="zeros")

    # Define the simulation parameters
    sim_params = StrongSimParams(
        observables=[Observable(Z(), site) for site in range(num_qubits)],
        get_state=True,
        show_progress=False,
    )
    simulator.run(state, circ, sim_params)
    assert sim_params.output_state is not None
    tdvp_state = sim_params.output_state.to_vec()
    np.testing.assert_allclose(1, np.abs(np.vdot(state_vector, tdvp_state)) ** 2)


def test_convert_to_vector_fidelity_long_range() -> None:
    """Test convert to vector.

    Tests the MPS_to_vector function for a circuit input
    """
    num_qubits = 3
    circ = QuantumCircuit(num_qubits)
    circ.h(0)
    circ.cx(0, 2)
    state_vector = np.array([0.70710678, 0, 0, 0, 0, 0.70710678, 0, 0])

    # Define the initial state
    state = MPS(num_qubits, state="zeros")

    # Define the simulation parameters
    sim_params = StrongSimParams(
        observables=[Observable(Z(), site) for site in range(num_qubits)],
        get_state=True,
        show_progress=False,
    )
    simulator.run(state, circ, sim_params)
    assert sim_params.output_state is not None
    tdvp_state = sim_params.output_state.to_vec()
    np.testing.assert_allclose(1, np.abs(np.vdot(state_vector, tdvp_state)) ** 2)


@pytest.mark.parametrize(("length", "target"), [(6, 16), (7, 7), (9, 8), (10, 3)])
def test_pad_shapes_and_centre(length: int, target: int) -> None:
    """Test that pad_bond_dimension correctly pads the MPS and preserves invariants.

    * the state's norm is unchanged
    * the orthogonality-centre index is [0]
    * every virtual leg has the expected size
      ( powers-of-two "staircase" capped by target_dim )
    """
    mps = MPS(length=length, state="zeros")  # all bonds = 1
    norm_before = mps.norm()

    mps.pad_bond_dimension(target)

    # invariants
    assert np.isclose(mps.norm(), norm_before, atol=1e-12)
    assert mps.check_canonical_form()[0] == 0

    # expected staircase
    for i, T in enumerate(mps.tensors):
        _, chi_l, chi_r = T.shape

        # left (bond i - 1)
        if i == 0:
            left_expected = 1
        else:
            exp_left = min(i, length - i)
            left_expected = min(target, 2**exp_left)

        # right (bond i)
        if i == length - 1:
            right_expected = 1
        else:
            exp_right = min(i + 1, length - 1 - i)
            right_expected = min(target, 2**exp_right)

        assert chi_l == left_expected, f"site {i}: left {chi_l} vs {left_expected}"
        assert chi_r == right_expected, f"site {i}: right {chi_r} vs {right_expected}"


def test_pad_raises_on_shrink() -> None:
    """Test that pad_bond_dimension raises a ValueError when trying to shrink the bond dimension.

    Calling pad_bond_dimension with a *smaller* target than an existing
    bond must raise a ValueError.
    """
    mps = MPS(length=5, state="zeros")
    mps.pad_bond_dimension(4)  # enlarge first

    with pytest.raises(ValueError, match="Target bond dim must be at least current bond dim"):
        mps.pad_bond_dimension(2)  # would shrink - must fail


def test_haar_random_shapes_and_isometries() -> None:
    """Haar-random initializer should produce feasible bonds and site-wise isometries."""
    length = 8
    chi_max = 4
    mps = MPS(length=length, state="haar-random", pad=chi_max)
    expected = _expected_uniform_clipped_bonds(length, chi_max)

    for i, tensor in enumerate(mps.tensors):
        d, chi_l, chi_r = tensor.shape
        assert d == 2
        assert chi_l == expected[i]
        assert chi_r == expected[i + 1]

        q_mat = tensor.reshape(d * chi_l, chi_r)
        ident = np.eye(chi_r, dtype=np.complex128)
        np.testing.assert_allclose(q_mat.conj().T @ q_mat, ident, atol=1e-12)

    assert np.isclose(mps.norm(), 1.0, atol=1e-12)


def test_haar_random_default_pad_is_product_state() -> None:
    """Without pad, Haar-random defaults to χ_max=1 and should be a product state."""
    mps = MPS(length=6, state="haar-random")
    for tensor in mps.tensors:
        assert tensor.shape[1] == 1
        assert tensor.shape[2] == 1

    assert np.isclose(mps.get_entropy([2, 3]), 0.0, atol=1e-12)


def test_haar_random_invalid_pad_raises() -> None:
    """Haar-random initializer should reject non-positive target bond dimensions."""
    with pytest.raises(ValueError, match="Target bond dimension must be at least 1"):
        _ = MPS(length=6, state="haar-random", pad=0)


def test_haar_random_entropy_statistics_vs_random_mps() -> None:
    """Haar-random MPS should show higher mean entropy and lower variance than random tensors."""
    length = 8
    chi_max = 4
    cut = length // 2 - 1
    num_samples = 120

    bonds = _expected_uniform_clipped_bonds(length, chi_max)
    shapes = [(2, bonds[i], bonds[i + 1]) for i in range(length)]
    local_rng = np.random.default_rng(1234)

    rand_entropies = np.empty(num_samples, dtype=np.float64)
    haar_entropies = np.empty(num_samples, dtype=np.float64)

    for idx in range(num_samples):
        tensors = [crandn(shape, seed=local_rng) for shape in shapes]
        rand_state = MPS(length=length, tensors=tensors, physical_dimensions=2)
        rand_state.normalize()
        rand_state.set_canonical_form(cut)
        rand_entropies[idx] = rand_state.get_entropy([cut, cut + 1])

        haar_state = MPS(length=length, state="haar-random", pad=chi_max)
        haar_state.set_canonical_form(cut)
        haar_entropies[idx] = haar_state.get_entropy([cut, cut + 1])

    assert float(np.mean(haar_entropies)) > float(np.mean(rand_entropies))
    assert float(np.std(haar_entropies)) < float(np.std(rand_entropies))


@pytest.mark.parametrize("center", [0, 1, 2, 3])
def test_truncate_preserves_orthogonality_center_and_canonicity(center: int) -> None:
    """Test that truncation preserves the orthogonality center and canonicity.

    This test checks that after truncation, the orthogonality center remains unchanged.
    """
    # build a simple MPS of length 4
    shapes = [(2, 1, 4)] + [(2, 4, 4)] * 2 + [(2, 4, 1)]
    mps = random_mps(shapes)
    # set an arbitrary initial center
    mps.set_canonical_form(center)
    # record the full state-vector for fidelity check
    before_vec = mps.to_vec()
    # record the center and canonical-split
    before_center = mps.check_canonical_form()[0]
    assert before_center == center

    # do a "no-real" truncation (tiny threshold, generous max bond)
    mps.truncate(threshold=1e-16, max_bond_dim=100)
    after_center = mps.check_canonical_form()[0]
    assert after_center == center

    # fidelity of state stays unity
    after_vec = mps.to_vec()
    overlap = np.abs(np.vdot(before_vec, after_vec)) ** 2
    assert np.isclose(overlap, 1.0, atol=1e-12)

    # also check left/right canonicity around that center
    L = mps.length
    for i in range(before_center):
        # left-canonical test
        A = mps.tensors[i]
        conjA = np.conj(A)
        gram = oe.contract("ijk, ijl->kl", conjA, A)
        # identity on the i-th right bond
        assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-12)
    for i in range(before_center + 1, L):
        # right-canonical test
        A = mps.tensors[i]
        conjA = np.conj(A)
        gram = oe.contract("ijk, ilk->jl", A, conjA)
        assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-12)


def test_truncate_reduces_bond_dimensions_and_truncates() -> None:
    """Test that truncation reduces bond dimensions and truncates the MPS.

    This test creates an MPS with large bond dimensions and then truncates it to a smaller size.
    """
    # build an MPS with initially large bonds
    shapes = [(2, 1, 8)] + [(2, 8, 8)] * 3 + [(2, 8, 1)]
    mps = random_mps(shapes)
    # put it into a known canonical form
    mps.set_canonical_form(2)
    # perform a truncation that will cut back to max_bond=3
    mps.truncate(threshold=1e-12, max_bond_dim=3)

    # check validity and that every bond dim <= 3
    mps.check_if_valid_mps()
    for _tensor in mps.tensors:
        pass
    for T in mps.tensors:
        _, bond_left, bond_right = T.shape
        assert bond_left <= 3
        assert bond_right <= 3


def _bell_pair_mps() -> MPS:
    """Auxiliary function to create a Bell-pair MPS.

    Construct a 2-site MPS for the Bell state (|00> + |11>)/√2.
    Contracting the bond yields θ = diag(1/√2, 1/√2).

    Shapes:
        A: (phys=2, left=1, right=2)
        B: (phys=2, left=2, right=1)

    Returns:
        MPS: The product-state MPS.
    """
    A = np.zeros((2, 1, 2), dtype=complex)
    B = np.zeros((2, 2, 1), dtype=complex)

    # A encodes 1/√2 on |0> with bond 0, and 1/√2 on |1> with bond 1
    A[0, 0, 0] = 1 / np.sqrt(2)
    A[1, 0, 1] = 1 / np.sqrt(2)

    # B routes bond 0 -> |0>, bond 1 -> |1>
    B[0, 0, 0] = 1.0
    B[1, 1, 0] = 1.0

    return MPS(length=2, tensors=[A, B], physical_dimensions=[2, 2])


def _product_state_mps(length: int) -> MPS:
    """Construct a product-state MPS |0…0⟩ with all bonds = 1.

    Returns:
        MPS: The product-state MPS.
    """
    pdim = 2
    tensors = []
    for _ in range(length):
        T = np.zeros((pdim, 1, 1), dtype=complex)
        T[0, 0, 0] = 1.0  # |0>
        tensors.append(T)
    return MPS(length=length, tensors=tensors, physical_dimensions=[pdim] * length)


def test_get_max_bond() -> None:
    """get_max_bond reports max over index-0/2 across tensors."""
    # Shapes chosen so the per-tensor max(phys_dim, right_bond) are 3, 4, 2 → global 4
    t1 = np.zeros((2, 1, 3), dtype=complex)
    t2 = np.zeros((2, 3, 4), dtype=complex)
    t3 = np.zeros((2, 4, 1), dtype=complex)
    mps = MPS(length=3, tensors=[t1, t2, t3], physical_dimensions=[2, 2, 2])

    assert mps.get_max_bond() == 4


def test_get_total_bond() -> None:
    """get_total_bond sums internal left bonds over tensors[1:]."""
    # Left bonds (2nd index) of tensors[1:] are 3 and 4 → total 7
    t1 = np.zeros((2, 1, 3), dtype=complex)
    t2 = np.zeros((2, 3, 4), dtype=complex)
    t3 = np.zeros((2, 4, 1), dtype=complex)
    mps = MPS(length=3, tensors=[t1, t2, t3], physical_dimensions=[2, 2, 2])

    assert mps.get_total_bond() == 7


def test_get_cost() -> None:
    """get_cost sums cubes of internal left bonds over tensors[1:]."""
    # Cubes: 3^3 + 4^3 = 27 + 64 = 91
    t1 = np.zeros((2, 1, 3), dtype=complex)
    t2 = np.zeros((2, 3, 4), dtype=complex)
    t3 = np.zeros((2, 4, 1), dtype=complex)
    mps = MPS(length=3, tensors=[t1, t2, t3], physical_dimensions=[2, 2, 2])

    assert mps.get_cost() == 91


def test_get_entropy_zero_for_product_cut() -> None:
    """get_entropy returns 0 on a product state (bond dim = 1)."""
    mps = _product_state_mps(4)
    ent = mps.get_entropy([1, 2])  # nearest-neighbor cut
    assert isinstance(ent, np.float64)
    assert np.isclose(ent, 0.0, atol=1e-12)


def test_get_entropy_bell_pair_ln2() -> None:
    """get_entropy across the Bell cut equals ln(2)."""
    mps = _bell_pair_mps()
    ent = mps.get_entropy([0, 1])
    assert np.isclose(ent, np.log(2.0), atol=1e-12)


def test_get_entropy_asserts_on_non_adjacent_or_wrong_len() -> None:
    """get_entropy asserts on invalid site lists."""
    mps = _product_state_mps(4)
    with pytest.raises(AssertionError):
        _ = mps.get_entropy([1])  # wrong length
    with pytest.raises(AssertionError):
        _ = mps.get_entropy([1, 3])  # non-adjacent


def test_get_schmidt_spectrum_product_padding() -> None:
    """get_schmidt_spectrum returns [1, nan, …] for product cut; length=500."""
    mps = _product_state_mps(3)
    spec = mps.get_schmidt_spectrum([0, 1])

    assert isinstance(spec, np.ndarray)
    assert spec.dtype == np.float64
    assert spec.shape == (500,)
    assert np.isclose(spec[0], 1.0, atol=1e-12)
    # the remainder must be NaN
    assert np.all(np.isnan(spec[1:]))


def test_get_schmidt_spectrum_bell_pair_values_and_padding() -> None:
    """get_schmidt_spectrum on Bell pair yields two equal singular values then NaNs."""
    mps = _bell_pair_mps()
    spec = mps.get_schmidt_spectrum([0, 1])

    assert spec.shape == (500,)
    # Two non-NaN entries ≈ 1/√2, rest NaN
    non_nan = spec[~np.isnan(spec)]
    assert non_nan.size == 2
    assert np.allclose(non_nan, 1 / np.sqrt(2), atol=1e-12)
    assert np.all(np.isnan(spec[2:]))


def test_get_schmidt_spectrum_asserts_on_invalid_sites() -> None:
    """get_schmidt_spectrum asserts on non-adjacent or wrong-length site lists."""
    mps = _product_state_mps(5)
    with pytest.raises(AssertionError):
        _ = mps.get_schmidt_spectrum([2])  # wrong length
    with pytest.raises(AssertionError):
        _ = mps.get_schmidt_spectrum([1, 3])  # non-adjacent


def test_evaluate_observables_diagnostics_and_meta_then_pvm_separately() -> None:
    """Evaluate diagnostics/meta (no PVM) and PVM in separate calls to satisfy params typing/rules.

    For |0000⟩ product MPS:
      - runtime_cost = Σ_{i≥1} bond_left(i)^3 = 1^3 * 3 = 3
      - total_bond  = Σ_{i≥1} bond_left(i)   = 1   * 3 = 3
      - max_bond    = max over (phys_dim/right_bond) = 2
      - entropy(1,2) = 0
      - schmidt_spectrum(1,2) = length-500 vector with [1, nan, ...]
      - pvm("0000") = 1  (checked in a separate params object to avoid mixing)
    """
    mps = _product_state_mps(4)

    # ---- diagnostics + meta (NO PVM here) ----
    diagnostics_and_meta: list[Observable] = [
        Observable(GateLibrary.runtime_cost(), 0),
        Observable(GateLibrary.max_bond(), 0),
        Observable(GateLibrary.total_bond(), 0),
        Observable(GateLibrary.entropy(), [1, 2]),
        Observable(GateLibrary.schmidt_spectrum(), [1, 2]),
    ]
    sim_diag = AnalogSimParams(diagnostics_and_meta, elapsed_time=0.1, dt=0.1, show_progress=False)

    results_diag = np.empty((len(diagnostics_and_meta), 2), dtype=object)
    mps.evaluate_observables(sim_diag, results_diag, column_index=0)

    # Diagnostics
    # Ordering based on sorted_observables
    assert results_diag[2, 0] == 3  # runtime_cost
    assert results_diag[3, 0] == 2  # max_bond
    assert results_diag[4, 0] == 3  # total_bond

    # Entropy
    assert isinstance(results_diag[0, 0], (float, np.floating))
    assert np.isclose(results_diag[0, 0], 0.0, atol=1e-12)

    # Schmidt spectrum
    spec = results_diag[1, 0]
    assert isinstance(spec, np.ndarray)
    assert spec.shape == (500,)
    assert np.isclose(spec[0], 1.0, atol=1e-12)
    assert np.all(np.isnan(spec[1:]))

    # ---- PVM ONLY (no mixing) ----
    pvm_only = [Observable(GateLibrary.pvm("0000"), 0)]
    sim_pvm = AnalogSimParams(pvm_only, elapsed_time=0.1, dt=0.1, show_progress=False)

    results_pvm = np.empty((len(pvm_only), 1), dtype=object)
    mps.evaluate_observables(sim_pvm, results_pvm, column_index=0)

    assert results_pvm[0, 0] == 1


def test_evaluate_observables_local_ops_and_center_shifts() -> None:
    """Evaluate local observables over increasing sites to exercise rightward shifts.

    For |0000⟩:
      - ⟨Z⟩ at sites 0,1,3 is +1
      - ⟨X⟩ at site 2 is 0
    The observable order [Z(0), Z(1), X(2), Z(3)] forces center shifts 0→1→2→3.
    """
    mps = _product_state_mps(4)

    obs_seq: list[Observable] = [
        Observable(GateLibrary.z(), 0),
        Observable(GateLibrary.z(), 1),
        Observable(GateLibrary.x(), 2),
        Observable(GateLibrary.z(), 3),
    ]
    sim_params = AnalogSimParams(obs_seq, elapsed_time=0.1, dt=0.1, show_progress=False)

    results = np.empty((len(obs_seq), 3), dtype=np.float64)
    mps.evaluate_observables(sim_params, results, column_index=2)

    z0, z1, x2, z3 = (results[i, 2] for i in range(4))
    assert np.isclose(z0, 1.0, atol=1e-12)
    assert np.isclose(z1, 1.0, atol=1e-12)
    assert np.isclose(x2, 0.0, atol=1e-12)
    assert np.isclose(z3, 1.0, atol=1e-12)


def test_evaluate_observables_meta_validation_errors() -> None:
    """Meta-observable input validation: wrong length and non-adjacent sites must assert."""
    mps = _product_state_mps(4)

    # Wrong length (entropy expects exactly two adjacent indices)
    sim_bad_len = AnalogSimParams(
        [Observable(GateLibrary.entropy(), [1])],
        elapsed_time=0.1,
        dt=0.1,
        show_progress=False,
    )
    results_len = np.empty((1, 1), dtype=np.float64)
    with pytest.raises(AssertionError):
        mps.evaluate_observables(sim_bad_len, results_len, column_index=0)

    # Non-adjacent Schmidt cut
    sim_non_adj = AnalogSimParams(
        [Observable(GateLibrary.schmidt_spectrum(), [0, 2])],
        elapsed_time=0.1,
        dt=0.1,
        show_progress=False,
    )
    results_adj = np.empty((1, 1), dtype=object)
    with pytest.raises(AssertionError):
        mps.evaluate_observables(sim_non_adj, results_adj, column_index=0)


def test_hamiltonian_raises_on_nonpositive_length() -> None:
    """Hamiltonian input validation: non-positive system size must raise."""
    with pytest.raises(ValueError, match=r"L must be positive\."):
        MPO.hamiltonian(length=0)

    with pytest.raises(ValueError, match=r"L must be positive\."):
        MPO.hamiltonian(length=-3)


def test_hamiltonian_raises_on_invalid_bc() -> None:
    """Hamiltonian input validation: unsupported boundary conditions must raise."""
    with pytest.raises(ValueError, match=r"bc must be 'open' or 'periodic'\."):
        MPO.hamiltonian(length=4, bc="closed")

    with pytest.raises(ValueError, match=r"bc must be 'open' or 'periodic'\."):
        MPO.hamiltonian(length=4, bc="")


def test_hamiltonian_raises_on_invalid_one_body_operator() -> None:
    """Hamiltonian input validation: invalid single-site operator labels must raise."""
    with pytest.raises(ValueError, match=r"Invalid operator 'Q'"):
        MPO.hamiltonian(length=3, one_body=[(1.0, "Q")])


def test_hamiltonian_raises_on_invalid_two_body_operator_left() -> None:
    """Hamiltonian input validation: invalid left two-body operator labels must raise."""
    with pytest.raises(ValueError, match=r"Invalid operator 'Q'"):
        MPO.hamiltonian(length=3, two_body=[(1.0, "Q", "Z")])


def test_hamiltonian_raises_on_invalid_two_body_operator_right() -> None:
    """Hamiltonian input validation: invalid right two-body operator labels must raise."""
    with pytest.raises(ValueError, match=r"Invalid operator 'Q'"):
        MPO.hamiltonian(length=3, two_body=[(1.0, "X", "Q")])


def test_hamiltonian_normalizes_operator_case() -> None:
    """Hamiltonian construction: operator labels are case-insensitive and normalized."""
    _ = MPO.hamiltonian(
        length=2,
        one_body=[(0.5, "x")],
        two_body=[(1.0, "z", "y")],
        bc="open",
        n_sweeps=0,
    )


def test_from_pauli_sum_raises_on_invalid_physical_dimension() -> None:
    """Pauli-sum MPO validation: only physical_dimension=2 is supported."""
    mpo = MPO()
    with pytest.raises(ValueError, match=r"Only physical_dimension=2 is supported"):
        mpo.from_pauli_sum(terms=[(1.0, "Z0")], length=2, physical_dimension=3)


def test_from_pauli_sum_raises_on_nonpositive_length() -> None:
    """Pauli-sum MPO validation: non-positive length must raise."""
    mpo = MPO()
    with pytest.raises(ValueError, match=r"length must be positive\."):
        mpo.from_pauli_sum(terms=[(1.0, "Z0")], length=0)

    with pytest.raises(ValueError, match=r"length must be positive\."):
        mpo.from_pauli_sum(terms=[(1.0, "Z0")], length=-5)


def test_from_pauli_sum_raises_on_site_index_out_of_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pauli-sum MPO validation: parsed site indices outside [0, L-1] must raise."""
    mpo = MPO()

    # Force the parser to return an out-of-bounds site index regardless of spec.
    monkeypatch.setattr(mpo, "_parse_pauli_string", lambda _spec: {99: "Z"})

    with pytest.raises(ValueError, match=r"Site index 99 outside \[0, 3\]\."):
        mpo.from_pauli_sum(terms=[(1.0, "Z0")], length=4)


def test_from_pauli_sum_raises_on_invalid_local_op_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pauli-sum MPO validation: parsed local operator labels must be in _VALID."""
    mpo = MPO()

    # Force the parser to return an invalid label.
    monkeypatch.setattr(mpo, "_parse_pauli_string", lambda _spec: {0: "Q"})

    with pytest.raises(ValueError, match=r"Invalid local op 'Q'"):
        mpo.from_pauli_sum(terms=[(1.0, "Z0")], length=2)


def test_from_pauli_sum_empty_terms_builds_zero_mpo() -> None:
    """Pauli-sum MPO construction: empty term list yields an all-zero MPO with bond dim 1."""
    mpo = MPO()
    mpo.from_pauli_sum(terms=[], length=3, n_sweeps=0)  # n_sweeps=0 keeps it fast

    assert len(mpo.tensors) == 3
    for t in mpo.tensors:
        assert t.shape == (2, 2, 1, 1)
        assert np.allclose(t, 0.0)


def test_compress_raises_on_negative_n_sweeps() -> None:
    """MPO compress input validation: negative n_sweeps must raise."""
    mpo = MPO()
    mpo.tensors = [np.zeros((2, 2, 1, 1), dtype=complex)]
    with pytest.raises(ValueError, match=r"n_sweeps must be >= 0\."):
        mpo.compress(n_sweeps=-1)


def test_compress_raises_on_invalid_directions() -> None:
    """MPO compress input validation: invalid sweep schedule strings must raise."""
    mpo = MPO()
    mpo.tensors = [np.zeros((2, 2, 1, 1), dtype=complex)]
    with pytest.raises(
        ValueError,
        match=r"directions must be one of \{'lr', 'rl', 'lr_rl', 'rl_lr'\}\.",
    ):
        mpo.compress(directions="lr,rl")


def test_compress_n_sweeps_zero_returns_without_calling_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MPO compress control flow: n_sweeps=0 must return without invoking sweeps."""
    mpo = MPO()
    mpo.tensors = [
        np.zeros((2, 2, 1, 1), dtype=complex),
        np.zeros((2, 2, 1, 1), dtype=complex),
    ]

    called = False

    def boom(**_kwargs: object) -> None:
        nonlocal called
        called = True
        msg = "should not be called when n_sweeps=0"
        raise AssertionError(msg)

    monkeypatch.setattr(mpo, "_compress_one_sweep", boom)

    mpo.compress(n_sweeps=0, directions="lr_rl")
    assert called is False


def test_compress_one_sweep_raises_on_invalid_direction() -> None:
    """MPO _compress_one_sweep input validation: direction must be 'lr' or 'rl'."""
    mpo = MPO()
    mpo.tensors = [
        np.zeros((2, 2, 1, 1), dtype=complex),
        np.zeros((2, 2, 1, 1), dtype=complex),
    ]
    with pytest.raises(ValueError, match=r"direction must be 'lr' or 'rl'\."):
        mpo._compress_one_sweep(direction="xx", tol=1e-12, max_bond_dim=None)  # noqa: SLF001


def test_from_pauli_sum_empty_spec_is_identity_term() -> None:
    """Pauli parsing integration: empty spec denotes the identity operator."""
    mpo = MPO()
    mpo.from_pauli_sum(terms=[(1.0, "")], length=2, n_sweeps=0)
    assert len(mpo.tensors) == 2  # construction succeeded


def test_from_pauli_sum_parses_commas_and_normalizes_case() -> None:
    """Pauli parsing integration: commas/whitespace are accepted and labels are case-normalized."""
    mpo = MPO()
    mpo.from_pauli_sum(terms=[(1.0, "x0, y1")], length=2, n_sweeps=0)
    assert len(mpo.tensors) == 2


def test_from_pauli_sum_raises_on_duplicate_site_in_spec() -> None:
    """Pauli parsing integration: duplicate site indices in a spec must raise."""
    mpo = MPO()
    with pytest.raises(ValueError, match=r"Duplicate site 0 in spec"):
        mpo.from_pauli_sum(terms=[(1.0, "X0 Z0")], length=2, n_sweeps=0)


def test_from_pauli_sum_raises_on_invalid_tokens_in_spec() -> None:
    """Pauli parsing integration: invalid tokens in the spec must raise."""
    mpo = MPO()
    with pytest.raises(ValueError, match=r"Invalid token\(s\) in spec"):
        mpo.from_pauli_sum(terms=[(1.0, "X0 Q2")], length=3, n_sweeps=0)

    with pytest.raises(ValueError, match=r"Invalid token\(s\) in spec"):
        mpo.from_pauli_sum(terms=[(1.0, "X0 Y2 garbage")], length=4, n_sweeps=0)
