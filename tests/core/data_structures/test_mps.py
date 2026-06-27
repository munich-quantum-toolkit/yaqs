# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for :class:`mqt.yaqs.core.data_structures.mps.MPS`."""

# ruff: noqa: N806

from __future__ import annotations

import copy

import numpy as np
import opt_einsum as oe
import pytest
from qiskit.circuit import QuantumCircuit
from scipy.stats import unitary_group

from mqt.yaqs import AnalogSimParams, Observable, Simulator, State, StrongSimParams
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.libraries.gate_library import BaseGate, GateLibrary, X, Z

_I2 = np.eye(2, dtype=complex)
_X2 = np.array([[0, 1], [1, 0]], dtype=complex)
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


def crandn(
    size: int | tuple[int, ...],
    *args: int,
    seed: np.random.Generator | int | None = None,
) -> np.ndarray:
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


def test_local_expect_qudit_observable_num_sites() -> None:
    """A single d=4 qudit observable (number operator) works via ``num_sites=1``.

    The number operator ``diag(0,1,2,3)`` on a single d=4 qudit initialized to
    level 2 should yield expectation value 2.
    """
    number_op = BaseGate(np.diag([0, 1, 2, 3]).astype(np.complex128), num_sites=1)
    observable = Observable(number_op, sites=0)

    mps = MPS(length=1, physical_dimensions=[4], state="zeros")
    mps.tensors[0] = np.zeros((4, 1, 1), dtype=np.complex128)
    mps.tensors[0][2, 0, 0] = 1.0

    val = mps.local_expect(observable, sites=0)
    np.testing.assert_allclose(val, 2.0, atol=1e-12)


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
    results = psi_zero.measure_shots(shots=20, basis="X")
    assert results.get(0, 0) > 0
    assert results.get(1, 0) > 0
    assert sum(results.values()) == 20


def test_single_shot_qudit_mixed_radix_encoding() -> None:
    """measure_single_shot encodes outcomes via mixed-radix, not binary bit-shift, for qudits.

    A mixed-dimension MPS ([2, 3, 4]) deterministically prepared in level (1, 2, 3) must
    measure to ``1*1 + 2*2 + 3*(2*3) = 23`` in the "Z" basis, not the binary bit-shift result.
    """
    mps = MPS(length=3, physical_dimensions=[2, 3, 4], state="zeros")
    mps.tensors[0] = np.zeros((2, 1, 1), dtype=complex)
    mps.tensors[0][1, 0, 0] = 1.0
    mps.tensors[1] = np.zeros((3, 1, 1), dtype=complex)
    mps.tensors[1][2, 0, 0] = 1.0
    mps.tensors[2] = np.zeros((4, 1, 1), dtype=complex)
    mps.tensors[2][3, 0, 0] = 1.0

    rng = np.random.default_rng(0)
    outcome = mps.measure_single_shot(basis="Z", rng=rng)
    assert outcome == 1 * 1 + 2 * 2 + 3 * (2 * 3)


def test_single_shot_qudit_y_basis_is_heisenberg_weyl_eigenbasis() -> None:
    """measure_single_shot's "Y" basis for qudits is the eigenbasis of the Heisenberg-Weyl product X_d @ Z_d.

    A qutrit prepared in the k-th eigenstate of X_d @ Z_d (eigenvalues ordered by ascending
    phase, matching the qubit Y convention for d=2) must measure deterministically to outcome k
    in the "Y" basis.
    """
    dim = 3
    shift = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        shift[(i + 1) % dim, i] = 1
    omega = np.exp(2j * np.pi / dim)
    clock = np.diag(omega ** np.arange(dim))
    eigvals, eigvecs = np.linalg.eig(shift @ clock)
    order = np.argsort(np.angle(eigvals))
    site_rotation = eigvecs[:, order].conj().T

    rng = np.random.default_rng(0)
    for k in range(dim):
        mps = MPS(length=1, physical_dimensions=[dim], state="zeros")
        mps.tensors[0][:, 0, 0] = np.conj(site_rotation[k, :])
        outcomes = {mps.measure_single_shot(basis="Y", rng=rng) for _ in range(20)}
        assert outcomes == {k}


def test_single_shot_qudit_x_basis_is_dft_eigenbasis() -> None:
    """measure_single_shot's "X" basis for qudits is the generalized-Hadamard (DFT) eigenbasis.

    A qutrit prepared in the k-th eigenstate of the generalized shift operator (the k-th column
    of the DFT matrix's conjugate transpose) must measure deterministically to outcome k in the
    "X" basis.
    """
    dim = 3
    omega = np.exp(2j * np.pi / dim)
    indices = np.arange(dim)
    dft_matrix = (omega ** np.outer(indices, indices)) / np.sqrt(dim)

    rng = np.random.default_rng(0)
    for k in range(dim):
        mps = MPS(length=1, physical_dimensions=[dim], state="zeros")
        mps.tensors[0][:, 0, 0] = np.conj(dft_matrix[k, :])
        outcomes = {mps.measure_single_shot(basis="X", rng=rng) for _ in range(20)}
        assert outcomes == {k}


def test_measure_shots_qudit_aggregates_mixed_radix_outcomes() -> None:
    """measure_shots aggregates qudit outcomes consistently with the mixed-radix encoding."""
    mps = MPS(length=2, physical_dimensions=[2, 3], state="zeros")
    mps.tensors[0] = np.zeros((2, 1, 1), dtype=complex)
    mps.tensors[0][1, 0, 0] = 1.0
    mps.tensors[1] = np.zeros((3, 1, 1), dtype=complex)
    mps.tensors[1][2, 0, 0] = 1.0

    results = mps.measure_shots(shots=10, basis="Z")
    assert results == {1 + 2 * 2: 10}


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
    # MPS is now (|00> + |11>) / sqrt(2)
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
    state = State(num_qubits, initial="zeros")

    # Define the simulation parameters
    sim_params = StrongSimParams(
        observables=[Observable(Z(), site) for site in range(num_qubits)],
        get_state=True,
    )
    result = Simulator(show_progress=False).run(state, circ, sim_params)
    assert result.output_state is not None
    tdvp_state = result.output_state.mps.to_vec()
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
    state = State(num_qubits, initial="zeros")

    # Define the simulation parameters
    sim_params = StrongSimParams(
        observables=[Observable(Z(), site) for site in range(num_qubits)],
        get_state=True,
    )
    result = Simulator(show_progress=False).run(state, circ, sim_params)
    assert result.output_state is not None
    tdvp_state = result.output_state.mps.to_vec()
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
    mps.compress(threshold=1e-16, max_bond_dim=100)
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
    mps.compress(threshold=1e-12, max_bond_dim=3)

    # check validity and that every bond dim <= 3
    mps.check_if_valid_mps()
    for _tensor in mps.tensors:
        pass
    for T in mps.tensors:
        _, bond_left, bond_right = T.shape
        assert bond_left <= 3
        assert bond_right <= 3


def test_compress_single_site_returns_immediately() -> None:
    """``compress`` is a no-op on a one-site MPS."""
    mps = MPS(1, state="zeros")
    before = copy.deepcopy(mps.tensors)
    mps.compress(threshold=1e-12)
    for before_tensor, after_tensor in zip(before, mps.tensors, strict=True):
        np.testing.assert_allclose(before_tensor, after_tensor)


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


def test_assert_bond_shapes_consistent_passes_for_valid_mps() -> None:
    """Bond-shape check accepts matching neighbor bond dimensions."""
    mps = MPS(length=3, state="zeros")
    mps.ensure_internal_bond_dims((0,), 2)
    mps.assert_bond_shapes_consistent(max_bond_dim=2)


def test_assert_bond_shapes_consistent_raises_on_mismatch() -> None:
    """Internal bond-shape check rejects inconsistent neighbor bond sizes."""
    t0 = np.zeros((2, 1, 3), dtype=complex)
    t1 = np.zeros((2, 2, 1), dtype=complex)
    t2 = np.zeros((2, 1, 1), dtype=complex)
    mps = MPS(length=3, tensors=[t0, t1, t2], physical_dimensions=[2, 2, 2])
    with pytest.raises(ValueError, match="bond mismatch"):
        mps.assert_bond_shapes_consistent()


def test_ensure_internal_bond_dims_zero_pads_selected_bonds() -> None:
    """Bond padding raises only listed bonds without touching others."""
    mps = MPS(length=3, state="zeros")
    mps.ensure_internal_bond_dims((0,), 2)

    assert mps.tensors[0].shape == (2, 1, 2)
    assert mps.tensors[1].shape == (2, 2, 1)
    assert mps.tensors[2].shape == (2, 1, 1)


def test_ensure_internal_bond_dims_respects_max_dim() -> None:
    """Bond padding does not pad above an explicit max_dim cap."""
    mps = MPS(length=3, state="zeros")
    mps.ensure_internal_bond_dims((0,), 2, max_dim=1)

    assert mps.tensors[0].shape == (2, 1, 1)
    assert mps.tensors[1].shape == (2, 1, 1)


def test_ensure_internal_bond_dims_pads_asymmetric_bond() -> None:
    """Internal bond padding raises the smaller side when only one tensor is short."""
    t0 = np.zeros((2, 1, 4), dtype=complex)
    t1 = np.zeros((2, 2, 1), dtype=complex)
    t2 = np.zeros((2, 1, 1), dtype=complex)
    mps = MPS(length=3, tensors=[t0, t1, t2], physical_dimensions=[2, 2, 2])
    mps.ensure_internal_bond_dims((0,), 4)

    assert mps.tensors[0].shape == (2, 1, 4)
    assert mps.tensors[1].shape == (2, 4, 1)


def test_ensure_internal_bond_dims_raises_on_truncation() -> None:
    """Shrinking a bond via padding helper is rejected; SVD sync is required."""
    t0 = np.zeros((2, 1, 4), dtype=complex)
    t1 = np.zeros((2, 4, 1), dtype=complex)
    t2 = np.zeros((2, 1, 1), dtype=complex)
    mps = MPS(length=3, tensors=[t0, t1, t2], physical_dimensions=[2, 2, 2])
    with pytest.raises(ValueError, match="cannot be truncated"):
        mps.ensure_internal_bond_dims((0,), 2, max_dim=2)


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
        Observable(GateLibrary.entropy(), [1, 2]),
        Observable(GateLibrary.schmidt_spectrum(), [1, 2]),
    ]
    sim_diag = AnalogSimParams(diagnostics_and_meta, elapsed_time=0.1, dt=0.1)

    results_diag = np.empty((len(diagnostics_and_meta), 2), dtype=object)
    mps.evaluate_observables(sim_diag, results_diag, column_index=0)

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
    sim_pvm = AnalogSimParams(pvm_only, elapsed_time=0.1, dt=0.1)

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
    sim_params = AnalogSimParams(obs_seq, elapsed_time=0.1, dt=0.1)

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
    )
    results_len = np.empty((1, 1), dtype=np.float64)
    with pytest.raises(AssertionError):
        mps.evaluate_observables(sim_bad_len, results_len, column_index=0)

    # Non-adjacent Schmidt cut
    sim_non_adj = AnalogSimParams(
        [Observable(GateLibrary.schmidt_spectrum(), [0, 2])],
        elapsed_time=0.1,
        dt=0.1,
    )
    results_adj = np.empty((1, 1), dtype=object)
    with pytest.raises(AssertionError):
        mps.evaluate_observables(sim_non_adj, results_adj, column_index=0)
