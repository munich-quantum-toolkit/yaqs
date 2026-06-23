# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: SLF001 -- white-box tests exercise private MPO compression helpers

"""Tests for :class:`mqt.yaqs.core.data_structures.mpo.MPO`."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mpo_utils import make_identity_site
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.data_structures.state_utils import embed_one_site_operator, embed_two_site_factors
from mqt.yaqs.core.libraries.gate_library import Destroy, GateLibrary, Id, Z

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

# ---- single-qubit ops ----
_I2 = np.eye(2, dtype=complex)
_X2 = np.array([[0, 1], [1, 0]], dtype=complex)
_Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z2 = np.array([[1, 0], [0, -1]], dtype=complex)


def _embed_one_body(op: np.ndarray, length: int, i: int) -> np.ndarray:
    """Embed a single-site operator (MPS / Qiskit site-0 LSB convention).

    Returns:
        Dense embedded operator matrix.
    """
    return embed_one_site_operator(np.asarray(op, dtype=np.complex128), length, i)


def _embed_two_body(op1: np.ndarray, op2: np.ndarray, length: int, i: int) -> np.ndarray:
    """Embed a nearest-neighbor two-site product operator.

    Returns:
        Dense embedded operator matrix.
    """
    return embed_two_site_factors(
        np.asarray(op1, dtype=np.complex128),
        np.asarray(op2, dtype=np.complex128),
        length,
        i,
        i + 1,
    )


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
    H = np.zeros((dim, dim), dtype=complex)  # noqa: N806 -- physics Hamiltonian matrix

    for i in range(length - 1):
        H += (-j_val) * _embed_two_body(_Z2, _Z2, length, i)  # noqa: N806
    for i in range(length):
        H += (-g) * _embed_one_body(_X2, length, i)  # noqa: N806

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
    H = np.zeros((dim, dim), dtype=complex)  # noqa: N806 -- physics Hamiltonian matrix

    for i in range(length - 1):
        H += (-jx) * _embed_two_body(_X2, _X2, length, i)  # noqa: N806
        H += (-jy) * _embed_two_body(_Y2, _Y2, length, i)  # noqa: N806
        H += (-jz) * _embed_two_body(_Z2, _Z2, length, i)  # noqa: N806
    for i in range(length):
        H += (-h) * _embed_one_body(_Z2, length, i)  # noqa: N806

    return H


def _bose_hubbard_dense(length: int, local_dim: int, omega: float, hopping_j: float, hubbard_u: float) -> np.ndarray:
    """Construct the exact dense Bose-Hubbard Hamiltonian for comparison.

    Args:
        length: Number of lattice sites.
        local_dim: Local Hilbert-space dimension per site.
        omega: On-site chemical potential.
        hopping_j: Nearest-neighbor hopping amplitude.
        hubbard_u: On-site interaction strength.

    Returns:
        Dense Hamiltonian matrix.
    """
    # Local operators
    a = Destroy(local_dim).matrix
    adag = Destroy(local_dim).dag().matrix
    n = adag @ a
    id_op = np.eye(local_dim, dtype=complex)

    dim = local_dim**length
    H = np.zeros((dim, dim), dtype=complex)  # noqa: N806 -- physics Hamiltonian matrix

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
        H += embed(op_list)  # noqa: N806

    # Hopping terms
    for i in range(length - 1):
        # adag_i * a_{i+1}
        op_list1 = [id_op] * length
        op_list1[i] = adag
        op_list1[i + 1] = a
        H += -hopping_j * embed(op_list1)  # noqa: N806

        # a_i * adag_{i+1}
        op_list2 = [id_op] * length
        op_list2[i] = a
        op_list2[i + 1] = adag
        H += -hopping_j * embed(op_list2)  # noqa: N806

    return H


def _position_grid_local_hamiltonian(
    positions: np.ndarray,
    mass: float,
    omega: float,
    trap_center: float,
    hbar: float,
) -> np.ndarray:
    """Construct the dense one-ion finite-difference Hamiltonian used as a test reference.

    Args:
        positions: One-dimensional numpy array containing the uniform position grid.
        mass: Ion mass used in the kinetic and harmonic potential terms.
        omega: Harmonic trap angular frequency.
        trap_center: Center position of the harmonic trap.
        hbar: Reduced Planck constant used in the kinetic prefactor.

    Returns:
        Dense local Hamiltonian in the position-grid basis.
    """
    d = positions.size
    dx = positions[1] - positions[0]
    second_derivative = (np.diag(np.ones(d - 1), k=-1) - 2.0 * np.eye(d) + np.diag(np.ones(d - 1), k=1)) / dx**2
    kinetic = -(hbar**2 / (2.0 * mass)) * second_derivative
    potential = np.diag(0.5 * mass * omega**2 * (positions - trap_center) ** 2)
    return kinetic + potential


def _embed_local_ops(length: int, local_dim: int, site_ops: list[np.ndarray]) -> np.ndarray:
    """Embed a list of single-site operators into the full Hilbert space.

    Args:
        length: Number of lattice sites.
        local_dim: Local Hilbert-space dimension per site.
        site_ops: One operator per site (length ``length``).

    Returns:
        The embedded operator on the full chain Hilbert space.
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

    Args:
        length: Number of fermionic lattice sites.
        t: Hopping amplitude.
        u: On-site interaction strength.

    Returns:
        Dense Hamiltonian matrix of shape ``(4**length, 4**length)``.
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

    Args:
        num_orbitals: Number of spin orbitals (must be even).
        t: Hopping amplitude.
        u: On-site interaction strength.

    Returns:
        Dense Hamiltonian matrix of shape ``(2**num_orbitals, 2**num_orbitals)``.

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


def dense_operator_schmidt_values(mpo: MPO, cut: int) -> NDArray[np.float64]:
    """Compute Schmidt values from dense contraction for a given MPO cut.

    Args:
        mpo: MPO whose operator Schmidt spectrum is computed.
        cut: Bond index between sites ``cut - 1`` and ``cut``.

    Returns:
        Dense Schmidt singular values for the requested cut.
    """
    mps = mpo.to_mps()

    state = mps.tensors[0][:, 0, :]
    for tensor in mps.tensors[1:]:
        state = np.tensordot(state, tensor, axes=([-1], [1]))

    state = np.squeeze(state, axis=-1)
    left_dim = int(np.prod(state.shape[:cut], dtype=np.int64))
    right_dim = int(np.prod(state.shape[cut:], dtype=np.int64))
    matrix = np.reshape(state, (left_dim, right_dim))
    singular_values = np.linalg.svd(matrix, compute_uv=False, full_matrices=False)
    return np.asarray(singular_values, dtype=np.float64)


def significant_schmidt_values(values: NDArray[np.float64], tol: float = 1e-12) -> NDArray[np.float64]:
    """Return the numerically significant part of a Schmidt spectrum.

    Args:
        values: Schmidt singular values to filter.
        tol: Drop values less than or equal to this threshold.

    Returns:
        Subset of ``values`` strictly greater than ``tol``.
    """
    spectrum = np.asarray(values, dtype=np.float64)
    return spectrum[spectrum > tol]


rng = np.random.default_rng()


def test_ising_correct_operator() -> None:
    """Verify that the Ising MPO matches the exact dense Hamiltonian."""
    L = 5  # noqa: N806 -- chain length matches physics notation
    J = 1.0  # noqa: N806
    g = 0.5

    mpo = MPO.ising(L, J, g)

    assert mpo.length == L
    assert mpo.physical_dimension == 2
    assert len(mpo.tensors) == L

    assert np.allclose(mpo.to_matrix(), _ising_dense(L, J, g), atol=1e-12)


def test_heisenberg_correct_operator() -> None:
    """Verify that the Heisenberg MPO matches the exact dense Hamiltonian."""
    L = 5  # noqa: N806 -- chain length matches physics notation
    Jx, Jy, Jz, h = 1.0, 0.5, 0.3, 0.2  # noqa: N806

    mpo = MPO.heisenberg(L, Jx, Jy, Jz, h)

    assert np.allclose(mpo.to_matrix(), _heisenberg_dense(L, Jx, Jy, Jz, h), atol=1e-12)


def test_bose_hubbard_correct_operator() -> None:
    """Verify that the Bose-Hubbard MPO matches the exact dense Hamiltonian."""
    length = 4
    local_dim = 3  # up to 2 bosons per site
    omega = 0.7
    J = 0.2  # noqa: N806 -- hopping matches MPO.bose_hubbard parameter name
    U = 1.3  # noqa: N806

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
    H_dense = _bose_hubbard_dense(length, local_dim, omega, J, U)  # noqa: N806
    H_mpo = mpo.to_matrix()  # noqa: N806
    np.testing.assert_allclose(H_mpo, H_dense, atol=1e-8)


def test_trapped_ion_one_ion() -> None:
    """Verify the one-ion position-grid MPO against a dense finite-difference reference."""
    positions = np.linspace(-1.5, 1.5, 5, dtype=np.float64)
    mass = 1.7
    omega = 0.8
    trap_center = 0.2
    hbar = 0.9

    mpo = MPO.trapped_ion(
        positions,
        [mass],
        omega,
        trap_center=trap_center,
        hbar=hbar,
        max_bond_dim=1,
    )

    expected = _position_grid_local_hamiltonian(positions, mass, omega, trap_center, hbar)
    assert mpo.length == 1
    assert mpo.physical_dimension == positions.size
    assert mpo.tensors[0].shape == (positions.size, positions.size, 1, 1)
    np.testing.assert_allclose(mpo.to_matrix(), expected, atol=1e-12)


def test_trapped_ion_two_ions() -> None:
    """Verify the exact two-ion MPO including its softened Coulomb interaction."""
    positions = np.linspace(-1.0, 1.0, 4, dtype=np.float64)
    masses = [1.2, 1.8]
    omega = 0.7
    trap_center = -0.1
    hbar = 0.85
    coulomb_strength = 0.6
    softening_length = 0.3

    mpo = MPO.trapped_ion(
        positions,
        masses,
        omega,
        trap_center=trap_center,
        hbar=hbar,
        coulomb_strength=coulomb_strength,
        softening_length=softening_length,
        coulomb_cutoff=0.0,
    )

    h1 = _position_grid_local_hamiltonian(positions, masses[0], omega, trap_center, hbar)
    h2 = _position_grid_local_hamiltonian(positions, masses[1], omega, trap_center, hbar)
    identity = np.eye(positions.size)
    distance = positions[:, None] - positions[None, :]
    coulomb = coulomb_strength / np.sqrt(distance**2 + softening_length**2)
    expected = np.kron(h1, identity) + np.kron(identity, h2) + np.diag(coulomb.ravel())

    assert mpo.length == 2
    assert mpo.physical_dimension == positions.size
    assert mpo.tensors[0].shape[3] == positions.size + 2
    assert mpo.tensors[1].shape[2] == positions.size + 2
    np.testing.assert_allclose(mpo.to_matrix(), expected, atol=1e-11)
    np.testing.assert_allclose(mpo.to_matrix(), mpo.to_matrix().conj().T, atol=1e-12)


def test_trapped_ion_coulomb_truncation() -> None:
    """Verify that max_bond_dim retains the leading Coulomb SVD channels."""
    positions = np.linspace(-2.0, 2.0, 6, dtype=np.float64)
    omega = 0.5
    coulomb_strength = 0.4
    softening_length = positions[1] - positions[0]
    mpo = MPO.trapped_ion(
        positions,
        [1.0, 1.0],
        omega,
        coulomb_strength=coulomb_strength,
        max_bond_dim=4,
    )

    assert mpo.tensors[0].shape[3] == 4
    assert mpo.tensors[1].shape[2] == 4

    local_h = _position_grid_local_hamiltonian(positions, 1.0, omega, 0.0, 1.0)
    identity = np.eye(positions.size)
    dense_coulomb = mpo.to_matrix() - np.kron(local_h, identity) - np.kron(identity, local_h)
    truncated_coulomb = np.diag(dense_coulomb).reshape(positions.size, positions.size)

    distance = positions[:, None] - positions[None, :]
    exact_coulomb = coulomb_strength / np.sqrt(distance**2 + softening_length**2)
    u, singular_values, vh = np.linalg.svd(exact_coulomb, full_matrices=False)
    expected_rank_2 = u[:, :2] @ np.diag(singular_values[:2]) @ vh[:2, :]
    np.testing.assert_allclose(truncated_coulomb, expected_rank_2, atol=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"positions": np.array([0.0, 1.0]), "masses": [1.0], "omega": 1.0}, "at least three"),
        ({"positions": np.array([0.0, 1.0, 3.0]), "masses": [1.0], "omega": 1.0}, "uniformly spaced"),
        ({"positions": np.arange(3.0), "masses": [], "omega": 1.0}, "exactly one or two"),
        ({"positions": np.arange(3.0), "masses": [-1.0], "omega": 1.0}, "finite positive"),
        ({"positions": np.arange(3.0), "masses": [1.0], "omega": -1.0}, "non-negative"),
        (
            {"positions": np.arange(3.0), "masses": [1.0], "omega": 1.0, "coulomb_strength": 1.0},
            "must be zero",
        ),
        (
            {"positions": np.arange(3.0), "masses": [1.0, 1.0], "omega": 1.0, "softening_length": 0.0},
            "finite and positive",
        ),
        ({"positions": np.arange(3.0), "masses": [1.0, 1.0], "omega": 1.0, "max_bond_dim": 1}, "at least 2"),
        (
            {"positions": np.arange(3.0), "masses": [1.0, 1.0], "omega": 1.0, "max_bond_dim": 4.0},
            "must be an integer",
        ),
    ],
)
def test_trapped_ion_validation(kwargs: dict[str, Any], match: str) -> None:
    """Reject malformed trapped-ion grids and physical parameters."""
    with pytest.raises(ValueError, match=match):
        MPO.trapped_ion(**kwargs)


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
    length = 3
    pdim = 2

    mpo = MPO.identity(length, physical_dimension=pdim)

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


def test_custom_without_transpose_sets_physical_dimension() -> None:
    """custom(transpose=False) reads the physical index from axis 2."""
    pdim = 3
    tensors = [
        rng.random(size=(1, 2, pdim, pdim)).astype(np.complex128),
        rng.random(size=(2, 1, pdim, pdim)).astype(np.complex128),
    ]
    mpo = MPO()
    mpo.custom(tensors, transpose=False)
    assert mpo.physical_dimension == pdim


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

    for original, created in zip(tensors, mpo.tensors, strict=True):
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
    length = 5
    d = 3  # local dimension
    H = _bose_hubbard_dense(length, d, 0.9, 0.6, 0.2)  # noqa: N806 -- dense Hamiltonian matrix

    Hmpo = MPO.from_matrix(H, d, 4)  # noqa: N806
    assert np.allclose(H, Hmpo.to_matrix())

    H = rng.random((d**length, d**length)) + 1j * rng.random((d**length, d**length))  # noqa: N806
    Hmpo = MPO.from_matrix(H, d, 1_000_000)  # noqa: N806
    assert np.allclose(H, Hmpo.to_matrix())

    length = 6
    H = rng.random((d**length, d**length)) + 1j * rng.random((d**length, d**length))  # noqa: N806
    Hmpo = MPO.from_matrix(H, d, 728)  # noqa: N806
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


def test_compute_identity_fidelity_and_check_if_identity() -> None:
    """Identity MPO reports unit fidelity and passes the identity check."""
    mpo = MPO.identity(length=3, physical_dimension=2)
    measured = mpo.compute_identity_fidelity()

    assert measured == pytest.approx(1.0, abs=1e-12)
    assert mpo.check_if_identity(0.9) is True
    assert mpo.check_if_identity(1.0 + 1e-12) is False


def test_compute_identity_fidelity_heterogeneous_physical_dimensions() -> None:
    """Identity fidelity normalizes by the product of per-site local dimensions."""
    local_dims = [2, 3, 2]
    mpo = MPO()
    mpo.custom([make_identity_site(d) for d in local_dims], transpose=False)

    measured = mpo.compute_identity_fidelity()

    assert measured == pytest.approx(1.0, abs=1e-12)


def test_compute_entanglement_entropy_identity_is_zero() -> None:
    """Identity MPO has vanishing operator entanglement entropy at the center cut."""
    mpo = MPO.identity(length=4, physical_dimension=2)
    center_cut = mpo.length // 2

    assert mpo.compute_entanglement_entropy(center_cut) == pytest.approx(0.0, abs=1e-12)


def test_compute_entanglement_entropy_is_finite_and_non_negative() -> None:
    """Entropy is finite and non-negative for a deterministic Ising MPO."""
    mpo = MPO.ising(length=4, J=1.0, g=0.7)
    entropy = mpo.compute_entanglement_entropy(2)

    assert np.isfinite(entropy)
    assert entropy >= -1e-12


def test_compute_schmidt_spectrum_and_entropy_match_dense_reference() -> None:
    """Schmidt spectrum and entropy match a dense reference contraction."""
    mpo = MPO.ising(length=4, J=1.0, g=0.7)
    cut = 2
    schmidt = mpo.compute_schmidt_spectrum(cut)
    dense_schmidt = dense_operator_schmidt_values(mpo, cut)
    dense_schmidt = dense_schmidt[dense_schmidt > 1e-12]

    np.testing.assert_allclose(significant_schmidt_values(schmidt), dense_schmidt, rtol=1e-10, atol=1e-12)

    probabilities = np.square(dense_schmidt)
    probabilities /= np.sum(probabilities)
    reference_entropy = -np.sum(probabilities * np.log(probabilities))

    assert mpo.compute_entanglement_entropy(cut) == pytest.approx(reference_entropy, abs=1e-12)


def test_compute_schmidt_spectrum_trivial_cut_returns_frobenius_norm() -> None:
    """Boundary cuts return the operator Frobenius norm with zero entropy."""
    mpo = MPO.ising(length=4, J=1.0, g=0.7)
    fro_norm = float(np.linalg.norm(np.asarray(mpo.to_matrix(), dtype=np.complex128), ord="fro"))

    np.testing.assert_allclose(mpo.compute_schmidt_spectrum(0), np.array([fro_norm]))
    np.testing.assert_allclose(mpo.compute_schmidt_spectrum(mpo.length), np.array([fro_norm]))
    assert mpo.compute_entanglement_entropy(0) == pytest.approx(0.0, abs=1e-12)
    assert mpo.compute_entanglement_entropy(mpo.length) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("invalid_cut", "exc_type"),
    [(True, TypeError), ("left", TypeError), (-1, ValueError), (5, ValueError)],
)
def test_compute_schmidt_spectrum_rejects_invalid_cut(
    invalid_cut: int | str | bool,  # noqa: FBT001
    exc_type: type[Exception],
) -> None:
    """Invalid cut specifiers raise TypeError or ValueError."""
    mpo = MPO.identity(length=4, physical_dimension=2)

    with pytest.raises(exc_type, match="cut"):
        # Intentionally pass an invalid cut type to exercise runtime validation.
        _ = mpo.compute_schmidt_spectrum(invalid_cut)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("invalid_base", [0.0, -2.0, 1.0, np.inf, np.nan])
def test_compute_entanglement_entropy_rejects_invalid_base(invalid_base: float) -> None:
    """Invalid logarithm bases are rejected."""
    mpo = MPO.ising(length=4, J=1.0, g=0.5)

    with pytest.raises(ValueError, match="base"):
        _ = mpo.compute_entanglement_entropy(2, base=invalid_base)


def test_to_mps() -> None:
    """Test converting an MPO to an MPS.

    This test initializes an MPO using ising, converts it to an MPS via to_mps,
    and verifies that the resulting MPS has the correct length and that each tensor has been reshaped
    to the expected dimensions.
    """
    length = 3
    J, g = 1.0, 0.5  # noqa: N806 -- Ising couplings match MPO.ising signature

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
    J, g = 1.0, 0.5  # noqa: N806 -- Ising couplings match MPO.ising signature

    mpo = MPO.ising(length, J, g)
    assert mpo.check_if_valid_mpo() is True


def test_check_if_valid_mpo_detects_bond_mismatch() -> None:
    """Invalid bond dimensions return False instead of asserting."""
    mpo = MPO.ising(3, 1.0, 0.5)
    mpo.tensors[1] = mpo.tensors[1].copy()
    mpo.tensors[1] = np.zeros((2, 2, 2, 99), dtype=np.complex128)
    assert mpo.check_if_valid_mpo() is False


def test_rotate() -> None:
    """Test the rotate method for an MPO.

    This test checks that rotating an MPO (without conjugation) transposes each tensor as expected,
    and that rotating back with conjugation returns tensors with the original physical dimensions.
    """
    length = 3
    J, g = 1.0, 0.5  # noqa: N806 -- Ising couplings match MPO.ising signature

    mpo = MPO.ising(length, J, g)
    original_tensors = [t.copy() for t in mpo.tensors]

    mpo.rotate(conjugate=False)
    for orig, rotated in zip(original_tensors, mpo.tensors, strict=True):
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
    mpo = MPO.identity(length=3, physical_dimension=2)
    fidelity_threshold = 0.9
    assert mpo.check_if_identity(fidelity_threshold) is True


def test_identity_mpo_tensors_are_independent() -> None:
    """Each site tensor in identity() must be a distinct array."""
    mpo = MPO.identity(3, physical_dimension=2)
    assert mpo.tensors[0] is not mpo.tensors[1]


def test_multiply_mps_identity_preserves_state() -> None:
    """Identity MPO.multiply(MPS) leaves the dense state unchanged."""
    length = 4
    state = MPS(length, state="ones")
    state.normalize()
    expected = np.asarray(state.to_vec(), dtype=np.complex128)

    identity = MPO.identity(length)
    identity.multiply(state, compress=False)

    np.testing.assert_allclose(state.to_vec(), expected, atol=1e-10)


def test_multiply_mpo_matches_dense_operator_product() -> None:
    """Site-wise MPO.multiply(MPO) matches the dense matrix product."""
    length = 4
    left = MPO.ising(length, 1.0, 0.5)
    right = MPO.ising(length, 1.0, 0.3)
    reference = left.to_matrix() @ right.to_matrix()

    left.multiply(right, compress=False)
    np.testing.assert_allclose(right.to_matrix(), reference, atol=1e-10)


def test_multiply_mps_with_compression() -> None:
    """``multiply(MPS, compress=True)`` runs an SVD sweep using ``sim_params``."""
    length = 3
    state = MPS(length, state="ones")
    state.normalize()
    gate = GateLibrary.cx()
    gate.set_sites(0, 1)
    gate_mpo = MPO.from_gate(gate, length)
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact")
    gate_mpo.multiply(state, sim_params=sim_params, compress=True)
    state.check_if_valid_mps()
    assert state.orthogonality_center is not None


def test_multiply_mps_invalidates_then_restores_center() -> None:
    """``multiply(MPS)`` clears gauge during apply and ``compress`` restores tracking."""
    length = 3
    state = MPS(length, state="haar-random", pad=4)
    state.normalize("B")
    assert state.orthogonality_center == 0
    gate = GateLibrary.cx()
    gate.set_sites(0, 1)
    gate_mpo = MPO.from_gate(gate, length)
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact")
    gate_mpo.multiply(state, sim_params=sim_params, compress=True)
    assert state.orthogonality_center is not None
    obs = Observable(GateLibrary.z(), 1)
    exp = state.expect(obs)
    assert np.isfinite(exp)
    assert abs(exp) <= 1.0

    no_compress = MPS(length, state="haar-random", pad=4)
    no_compress.normalize("B")
    gate_mpo.multiply(no_compress, sim_params=sim_params, compress=False)
    assert no_compress.orthogonality_center is None


def test_multiply_mps_compress_requires_sim_params() -> None:
    """Compression without ``sim_params`` raises ``ValueError``."""
    state = MPS(2, state="zeros")
    gate = GateLibrary.cx()
    gate.set_sites(0, 1)
    gate_mpo = MPO.from_gate(gate, 2)
    with pytest.raises(ValueError, match="sim_params is required"):
        gate_mpo.multiply(state, compress=True)


def test_multiply_mps_length_mismatch_raises() -> None:
    """MPO and MPS length mismatch raises ``ValueError``."""
    state = MPS(2, state="zeros")
    gate = GateLibrary.cx()
    gate.set_sites(0, 1)
    gate_mpo = MPO.from_gate(gate, 3)
    with pytest.raises(ValueError, match="does not match MPS length"):
        gate_mpo.multiply(state, compress=False)


def test_multiply_invalid_target_type_raises() -> None:
    """``multiply`` rejects non-MPS/MPO targets."""
    mpo = MPO.identity(2)
    with pytest.raises(TypeError, match="multiply expects MPS or MPO"):
        mpo.multiply(object())  # ty: ignore[no-matching-overload]


def test_multiply_mpo_embedded_start_site() -> None:
    """A shorter gate MPO can be embedded at a non-zero ``start_site``."""
    target = MPO.identity(4)
    gate = GateLibrary.cx()
    gate.set_sites(0, 1)
    support = MPO.from_gate(gate, 2)
    support.multiply(target, start_site=1, compress=False)
    assert target.length == 4
    target.check_if_valid_mpo()


def test_multiply_mpo_with_compression() -> None:
    """``multiply(MPO, compress=True)`` runs bond-dimension compression on the target."""
    gate = GateLibrary.cx()
    gate.set_sites(0, 1)
    support = MPO.from_gate(gate, 2)
    target = MPO.identity(2)
    support.multiply(target, compress=True, tol=1e-12, n_sweeps=1)
    target.check_if_valid_mpo()


def test_multiply_mpo_invalid_embed_raises() -> None:
    """Embedding a gate MPO outside the target chain raises ``ValueError``."""
    target = MPO.identity(3)
    gate = GateLibrary.cx()
    gate.set_sites(0, 1)
    support = MPO.from_gate(gate, 2)
    with pytest.raises(ValueError, match="Cannot embed MPO"):
        support.multiply(target, start_site=2, compress=False)


def test_check_if_identity_non_qubit_physical_dimension() -> None:
    """Identity check uses the MPO physical dimension, not the qubit default."""
    mpo = MPO.identity(2, physical_dimension=3)
    assert mpo.check_if_identity(0.9) is True


##############################################################################
def test_pauli_raises_on_nonpositive_length() -> None:
    """Pauli MPO input validation: non-positive system size must raise."""
    with pytest.raises(ValueError, match=r"L must be positive\."):
        MPO.pauli(length=0)

    with pytest.raises(ValueError, match=r"L must be positive\."):
        MPO.pauli(length=-3)


def test_pauli_raises_on_invalid_bc() -> None:
    """Pauli MPO input validation: unsupported boundary conditions must raise."""
    with pytest.raises(ValueError, match=r"bc must be 'open' or 'periodic'\."):
        MPO.pauli(length=4, bc="closed")

    with pytest.raises(ValueError, match=r"bc must be 'open' or 'periodic'\."):
        MPO.pauli(length=4, bc="")


def test_pauli_raises_on_invalid_one_body_operator() -> None:
    """Pauli MPO input validation: invalid single-site operator labels must raise."""
    with pytest.raises(ValueError, match=r"Invalid operator 'Q'"):
        MPO.pauli(length=3, one_body=[(1.0, "Q")])


def test_pauli_raises_on_invalid_two_body_operator_left() -> None:
    """Pauli MPO input validation: invalid left two-body operator labels must raise."""
    with pytest.raises(ValueError, match=r"Invalid operator 'Q'"):
        MPO.pauli(length=3, two_body=[(1.0, "Q", "Z")])


def test_pauli_raises_on_invalid_two_body_operator_right() -> None:
    """Pauli MPO input validation: invalid right two-body operator labels must raise."""
    with pytest.raises(ValueError, match=r"Invalid operator 'Q'"):
        MPO.pauli(length=3, two_body=[(1.0, "X", "Q")])


def test_pauli_normalizes_operator_case() -> None:
    """Pauli MPO construction: operator labels are case-insensitive and normalized."""
    _ = MPO.pauli(
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
        mpo._compress_one_sweep(direction="xx", tol=1e-12, max_bond_dim=None)


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
