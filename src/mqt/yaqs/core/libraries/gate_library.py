# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Library of quantum gates.

This module defines a collection of quantum gate classes used in quantum simulations.
Each gate is implemented as a class derived from BaseGate and includes its matrix representation,
tensor form, interactions, and generator(s). The module provides concrete implementations
for standard gates. The GateLibrary class aggregates all these gate classes for easy access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import linalg
from .operator_matrices import (
    CONTROLLED_X,
    CONTROLLED_Z,
    HADAMARD,
    IDENTITY,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    SWAP_MATRIX,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from qiskit.circuit import Parameter


def split_tensor(tensor: NDArray[np.complex128]) -> list[NDArray[np.complex128]]:
    """Splits a multi-qubit gate tensor into one tensor per site using Singular Value Decomposition (SVD).

    Only round-off-level singular values are discarded, so a weakly entangling gate keeps its full
    operator-Schmidt rank; truncation to a simulation tolerance happens later, in the state gauge.

    Args:
        tensor: A gate tensor of shape ``(2,) * (2 * n)`` for ``n >= 2`` sites, with index
            order ``(out_1, ..., out_n, in_1, ..., in_n)``.

    Returns:
        list[NDArray[np.complex128]]: A list containing one tensor per site resulting from the split.
            Each tensor has shape (2, 2, bond_left, bond_right); the outer bonds are 1.

    Raises:
        ValueError: If ``tensor`` does not encode a gate on at least two qubits.
    """
    num_sites = tensor.ndim // 2
    if tensor.ndim < 4 or tensor.ndim % 2 != 0:
        msg = f"tensor must have an even number of axes for at least two sites, got {tensor.ndim}."
        raise ValueError(msg)
    expected_shape = (2,) * (2 * num_sites)
    if tensor.shape != expected_shape:
        msg = f"tensor must have shape {expected_shape}, got {tensor.shape}."
        raise ValueError(msg)

    # Group the output and input leg of each site: (out_1, in_1, ..., out_n, in_n)
    matrix = np.transpose(tensor, [axis for site in range(num_sites) for axis in (site, num_sites + site)])

    # Split site by site with SVDs, carrying the singular values to the right
    tensors = []
    left_bond = 1
    remaining = np.reshape(matrix, (left_bond * 4, 4 ** (num_sites - 1)))
    for _ in range(num_sites - 1):
        u_mat, s_list, v_mat = linalg.svd(remaining, full_matrices=False)
        # Relative 1e-12 is a round-off floor, not a simulation tolerance.
        keep = linalg.truncate(s_list, mode="relative", threshold=1e-12, min_keep=1)
        s_list = s_list[:keep]
        u_mat = u_mat[:, :keep]
        v_mat = v_mat[:keep, :]
        tensors.append(np.transpose(np.reshape(u_mat, (left_bond, 2, 2, keep)), (1, 2, 0, 3)))
        left_bond = keep
        remaining = np.reshape(np.diag(s_list) @ v_mat, (left_bond * 4, remaining.shape[1] // 4))

    last_tensor = np.transpose(np.reshape(remaining, (left_bond, 2, 2)), (1, 2, 0))
    tensors.append(np.expand_dims(last_tensor, axis=3))
    return tensors


def extend_gate(
    tensor: NDArray[np.complex128],
    sites: list[int],
    physical_dimensions: list[int] | tuple[int, ...] | None = None,
) -> list[NDArray[np.complex128]]:
    """Extends gate to long-range MPO.

    Extends a given gate tensor to a Matrix Product Operator (MPO) by adding identity tensors
    between specified sites.

    Args:
        tensor: The input gate tensor to be extended.
        sites: A list of site indices where the gate tensor is to be applied.
        physical_dimensions: Optional local dimensions for the contiguous span from
            ``min(sites)`` to ``max(sites)``. When omitted, every site uses dimension 2.

    Returns:
        MPO site tensors in ascending site order from ``min(sites)`` to ``max(sites)``,
        including identity tensors between gate sites. Each tensor has axis order
        ``(out, in, bond_left, bond_right)``.

    Raises:
        ValueError: If ``physical_dimensions`` does not match the site span length.

    Notes:
        - The gate axes are permuted to ascending site order before the split, so the sites may
          be given in any order; the returned tensors are ordered by ascending site index.
        - Identity tensors are inserted between non-adjacent sites.
    """
    num_sites = len(sites)
    order = sorted(range(num_sites), key=lambda idx: sites[idx])
    if order != list(range(num_sites)):
        # Permute the gate axes from the declared site order to ascending site order.
        tensor = np.transpose(tensor, [*order, *[num_sites + idx for idx in order]])
    sorted_sites = sorted(sites)
    span_start = sorted_sites[0]
    span_len = sorted_sites[-1] - span_start + 1
    if physical_dimensions is None:
        dimensions = (2,) * span_len
    else:
        dimensions = tuple(physical_dimensions)
        if len(dimensions) != span_len:
            msg = f"Expected {span_len} physical dimensions for gate support, got {len(dimensions)}."
            raise ValueError(msg)

    tensors = split_tensor(tensor)

    # Adds identity tensors between sites
    mpo_tensors = [tensors[0]]
    for idx in range(1, num_sites):
        previous_right_bond = mpo_tensors[-1].shape[3]
        for site in range(sorted_sites[idx - 1] + 1, sorted_sites[idx]):
            dimension = dimensions[site - span_start]
            identity_tensor = np.zeros(
                (dimension, dimension, previous_right_bond, previous_right_bond),
                dtype=np.complex128,
            )
            for bond in range(previous_right_bond):
                identity_tensor[:, :, bond, bond] = np.eye(dimension, dtype=np.complex128)
            mpo_tensors.append(identity_tensor)
        mpo_tensors.append(tensors[idx])

    return mpo_tensors


class BaseGate:
    """Store a local operator and its sites for tensor-network simulation.

    :meth:`set_sites` binds physical sites in the operator's factor order.
    Gate subclasses define their operator and parameter conventions.

    Attributes:
        name: Gate identifier.
        matrix: Dense operator in local factor order.
        interaction: Number of sites the operator acts on.
        tensor: Matrix, or tensor with output axes followed by input axes after site binding.
        sites: Physical site indices in local factor order, assigned by :meth:`set_sites`.
        generator: Product factors for ``exp(-1j * G)``, assigned on site binding
            for gates that support generator-based evolution.
        mpo_tensors: Gate MPO tensors in ascending physical-site order, including
            spectator identities, available after binding two or more sites.
    """

    name: str = "custom"
    matrix: NDArray[np.complex128]
    interaction: int
    tensor: NDArray[np.complex128]
    generator: NDArray[np.complex128] | list[NDArray[np.complex128]]
    sites: list[int]

    def __init__(self, mat: ArrayLike) -> None:
        """Build a qubit operator from a dense matrix without checking unitarity.

        Args:
            mat: Square matrix whose dimension is a power of two.

        Raises:
            ValueError: If the matrix is not square and two-dimensional, or its
                dimension is not a power of two.
        """
        matrix = np.asarray(mat, dtype=np.complex128)
        if matrix.ndim != 2:
            msg = "Matrix must be a 2-D array."
            raise ValueError(msg)
        if matrix.shape[0] != matrix.shape[1]:
            msg = "Matrix must be square"
            raise ValueError(msg)

        dim = matrix.shape[0]
        interaction = int(np.log2(dim))
        if dim < 1 or 2**interaction != dim:
            msg = f"Matrix dimension {dim} must be a power of 2."
            raise ValueError(msg)

        self.matrix = matrix
        self.tensor = matrix
        self.interaction = interaction

    def set_sites(self, *sites: int | list[int]) -> None:
        """Bind sites and build the multi-qubit tensor and MPO representation.

        Args:
            *sites: Site indices or lists of indices, in local operator-factor order.
                Controlled gates list controls before targets. MPO tensors follow
                ascending physical-site order.

        Raises:
            ValueError: If the site count differs from ``interaction`` or multi-qubit
                support is invalid.
        """
        sites_list = []
        for s in sites:
            if isinstance(s, int):
                sites_list.append(s)
            else:
                sites_list.extend(s)

        # enforce the right number of sites
        if len(sites_list) != self.interaction:
            msg = f"Number of sites {len(sites_list)} must be equal to the interaction level {self.interaction}"
            raise ValueError(msg)

        # store as the proper type
        self.sites = sites_list
        if self.interaction >= 2:
            self.tensor = np.reshape(self.matrix, (2,) * (2 * self.interaction))
            self.mpo_tensors = extend_gate(self.tensor, self.sites)

    def __add__(self, other: BaseGate) -> BaseGate:
        """Adds two gates together.

        Args:
            other: The gate to be added.

        Returns:
            A new gate representing the sum of the two gates.

        Raises:
            ValueError: If the gates have different interaction levels.
        """
        if self.interaction != other.interaction:
            msg = "Cannot add gates with different interaction"
            raise ValueError(msg)
        return self._clone_with_matrix(self.matrix + other.matrix)

    def __sub__(self, other: BaseGate) -> BaseGate:
        """Subtracts one gate from another.

        Args:
            other: The gate to be subtracted.

        Returns:
            A new gate representing the difference between the two gates.

        Raises:
            ValueError: If the gates have different interaction levels.
        """
        if self.interaction != other.interaction:
            msg = "Cannot subtract gates with different interaction"
            raise ValueError(msg)
        return self._clone_with_matrix(self.matrix - other.matrix)

    def __mul__(self, other: BaseGate | complex) -> BaseGate:
        """Multiplies two gates or scales a gate by a scalar.

        Args:
            other: The gate or scalar to multiply.

        Returns:
            A new gate representing the product of the two gates or the scaled gate.

        Raises:
            ValueError: If the gates have different interaction levels (when multiplying two gates).
        """
        if isinstance(other, BaseGate):
            if self.interaction != other.interaction:
                msg = "Cannot multiply gates with different interaction"
                raise ValueError(msg)
            return self._clone_with_matrix(self.matrix @ other.matrix)

        return self._clone_with_matrix(self.matrix * other)

    def __rmul__(self, other: BaseGate | complex) -> BaseGate:
        """Multiplies a scalar or another gate with this gate (right multiplication).

        Args:
            other: The gate or scalar to multiply.

        Returns:
            A new gate representing the product.
        """
        return self.__mul__(other)

    def __matmul__(self, other: BaseGate) -> BaseGate:
        """Matrix multiplication using @ operator.

        Args:
            other: The other gate to multiply.

        Returns:
            A new BaseGate resulting from matrix multiplication.
        """
        return self._clone_with_matrix(self.matrix @ other.matrix)

    def _clone_with_matrix(self, matrix: NDArray[np.complex128]) -> BaseGate:
        """Return a gate with the same interaction level and a new matrix.

        Unlike :meth:`__init__`, this bypasses power-of-two dimension validation so
        arithmetic on d-level ladder operators preserves non-qubit matrix sizes.

        Args:
            matrix: Replacement gate matrix.

        Returns:
            New gate with ``interaction`` copied from ``self`` and ``name`` set to
            ``"custom"``.
        """
        clone = BaseGate.__new__(BaseGate)
        clone.matrix = np.asarray(matrix, dtype=np.complex128)
        clone.tensor = clone.matrix
        clone.interaction = self.interaction
        clone.name = "custom"
        return clone

    def dag(self) -> BaseGate:
        """Returns the conjugate transpose (dagger) of the gate.

        Returns:
            A new gate representing the conjugate transpose of this gate.
        """
        return self._clone_with_matrix(np.conj(self.matrix).T)

    def conj(self) -> BaseGate:
        """Returns the complex conjugate of the gate.

        Returns:
            A new gate representing the complex conjugate of this gate.
        """
        return self._clone_with_matrix(np.conj(self.matrix))

    def trans(self) -> BaseGate:
        """Returns the transpose of the gate.

        Returns:
            A new gate representing the transpose of this gate.
        """
        return self._clone_with_matrix(self.matrix.T)

    @classmethod
    def x(cls) -> X:
        """Returns the X gate.

        Returns:
            An instance of the X gate.
        """
        return X()

    @classmethod
    def y(cls) -> Y:
        """Returns the Y gate.

        Returns:
            An instance of the Y gate.
        """
        return Y()

    @classmethod
    def z(cls) -> Z:
        """Returns the Z gate.

        Returns:
            An instance of the Z gate.
        """
        return Z()

    @classmethod
    def h(cls) -> H:
        """Returns the H gate.

        Returns:
            An instance of the H gate.
        """
        return H()

    @classmethod
    def destroy(cls, d: int = 2) -> Destroy:
        """Returns the Destroy gate.

        Args:
            d: number of levels

        Returns:
            An instance of the Destroy gate.
        """
        return Destroy(d)

    @classmethod
    def create(cls, d: int = 2) -> Create:
        """Returns the Create gate.

        Args:
            d: number of levels

        Returns:
            An instance of the Create gate.
        """
        return Create(d)

    @classmethod
    def id(cls) -> Id:
        """Returns the Id gate.

        Returns:
            An instance of the Id gate.
        """
        return Id()

    @classmethod
    def sx(cls) -> SX:
        """Returns the SX gate.

        Returns:
            An instance of the SX gate.
        """
        return SX()

    @classmethod
    def rx(cls, params: list[Parameter]) -> Rx:
        """Returns the RX gate.

        Args:
            params (list[Parameter]): The rotation angle parameter.

        Returns:
            An instance of the RX gate.
        """
        return Rx(params)

    @classmethod
    def ry(cls, params: list[Parameter]) -> Ry:
        """Returns the RY gate.

        Args:
            params: The rotation angle parameter.

        Returns:
            An instance of the RY gate.
        """
        return Ry(params)

    @classmethod
    def rz(cls, params: list[Parameter]) -> Rz:
        """Returns the RZ gate.

        Args:
            params: The rotation angle parameter.

        Returns:
            An instance of the RZ gate.
        """
        return Rz(params)

    @classmethod
    def p(cls, params: list[Parameter]) -> Phase:
        """Returns the Phase gate.

        Args:
            params: The rotation angle parameter.

        Returns:
            An instance of the Phase gate.
        """
        return Phase(params)

    @classmethod
    def u(cls, params: list[Parameter]) -> U:
        """Returns the U gate.

        Args:
            params: The rotation angle parameters.

        Returns:
            An instance of the U gate.
        """
        return U(params)

    @classmethod
    def u2(cls, params: list[Parameter]) -> U2:
        """Returns the U2 gate.

        Args:
            params (list[Parameter]): The rotation angle parameters.

        Returns:
            An instance of the U2 gate.
        """
        return U2(params)

    @classmethod
    def cx(cls) -> CX:
        """Returns the CX gate.

        Returns:
            An instance of the CX gate.
        """
        return CX()

    @classmethod
    def cz(cls) -> CZ:
        """Returns the CZ gate.

        Returns:
            An instance of the CZ gate.
        """
        return CZ()

    @classmethod
    def ccx(cls) -> CCX:
        """Returns the CCX gate.

        Returns:
            An instance of the CCX gate.
        """
        return CCX()

    @classmethod
    def ccz(cls) -> CCZ:
        """Returns the CCZ gate.

        Returns:
            An instance of the CCZ gate.
        """
        return CCZ()

    @classmethod
    def cp(cls, params: list[Parameter]) -> CPhase:
        """Returns the CPhase gate.

        Args:
            params: The rotation angle parameter.

        Returns:
            An instance of the CPhase gate.
        """
        return CPhase(params)

    @classmethod
    def swap(cls) -> SWAP:
        """Returns the SWAP gate.

        Returns:
            An instance of the SWAP gate.
        """
        return SWAP()

    @classmethod
    def cswap(cls) -> CSWAP:
        """Returns the CSWAP gate.

        Returns:
            An instance of the CSWAP gate.
        """
        return CSWAP()

    @classmethod
    def rxx(cls, params: list[Parameter]) -> Rxx:
        """Returns the RXX gate.

        Args:
            params: The rotation angle parameter.

        Returns:
            An instance of the RXX gate.
        """
        return Rxx(params)

    @classmethod
    def ryy(cls, params: list[Parameter]) -> Ryy:
        """Returns the RYY gate.

        Args:
            params: The rotation angle parameter.

        Returns:
            An instance of the RYY gate.
        """
        return Ryy(params)

    @classmethod
    def rzz(cls, params: list[Parameter]) -> Rzz:
        """Returns the RZZ gate.

        Args:
            params: The rotation angle parameter.

        Returns:
            An instance of the RZZ gate.
        """
        return Rzz(params)

    @property
    def mpo_tensors(self) -> list[NDArray[np.complex128]]:
        """List of MPO tensors representing the gate.

        Raises:
            AttributeError: If the gate does not have MPO tensors defined.
        """
        try:
            return self._mpo_tensors
        except AttributeError:
            msg = "This gate does not have MPO tensors defined."
            raise AttributeError(msg) from None

    @mpo_tensors.setter
    def mpo_tensors(self, tensors: list[NDArray[np.complex128]]) -> None:
        self._mpo_tensors = tensors


class X(BaseGate):
    """Pauli-X (NOT) gate."""

    name = "x"

    def __init__(self) -> None:
        """Initializes the Pauli-X gate."""
        super().__init__(PAULI_X.copy())


class Y(BaseGate):
    """Pauli-Y gate."""

    name = "y"

    def __init__(self) -> None:
        """Initializes the Pauli-Y gate."""
        super().__init__(PAULI_Y.copy())


class Z(BaseGate):
    """Pauli-Z gate."""

    name = "z"

    def __init__(self) -> None:
        """Initializes the Pauli-Z gate."""
        super().__init__(PAULI_Z.copy())


class H(BaseGate):
    """Hadamard gate."""

    name = "h"

    def __init__(self) -> None:
        """Initializes the Hadamard gate."""
        super().__init__(HADAMARD.copy())


class Destroy(BaseGate):
    """Truncated annihilation operator with ``a[n-1, n] = sqrt(n)``."""

    name = "destroy"

    def __init__(self, d: int = 2) -> None:
        """Initializes the Destroy gate.

        Args:
            d: Physical dimension.
        """
        mat = np.diag(np.sqrt(np.arange(1, d)), k=1)
        self.matrix = np.asarray(mat, dtype=np.complex128)
        self.tensor = self.matrix
        self.interaction = 1


class Create(BaseGate):
    """Truncated creation operator with ``a†[n, n-1] = sqrt(n)``."""

    name = "create"

    def __init__(self, d: int = 2) -> None:
        """Initializes the Create gate.

        Args:
            d: Physical dimension.
        """
        mat = np.diag(np.sqrt(np.arange(1, d)), k=-1)
        self.matrix = np.asarray(mat, dtype=np.complex128)
        self.tensor = self.matrix
        self.interaction = 1


class Id(BaseGate):
    """Single-qubit identity gate."""

    name = "id"

    def __init__(self) -> None:
        """Initializes the identity gate."""
        super().__init__(IDENTITY.copy())


class SX(BaseGate):
    """Square-root X gate, with ``SX @ SX = X``."""

    name = "sx"

    def __init__(self) -> None:
        """Initializes the square-root X gate."""
        mat = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128)
        super().__init__(mat)


class Sdg(BaseGate):
    """Adjoint S gate ``diag(1, -1j)``."""

    name = "sdg"

    def __init__(self) -> None:
        """Initializes the adjoint S gate."""
        mat = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
        super().__init__(mat)


class S(BaseGate):
    """Phase gate ``diag(1, 1j)``."""

    name = "s"

    def __init__(self) -> None:
        """Initializes the S gate."""
        mat = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
        super().__init__(mat)


class Tdg(BaseGate):
    """Adjoint T gate ``diag(1, exp(-1j * pi / 4))``."""

    name = "tdg"

    def __init__(self) -> None:
        """Initializes the adjoint T gate."""
        mat = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)
        super().__init__(mat)


class T(BaseGate):
    """Phase gate ``diag(1, exp(1j * pi / 4))``."""

    name = "t"

    def __init__(self) -> None:
        """Initializes the T gate."""
        mat = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
        super().__init__(mat)


class SXdg(BaseGate):
    """Adjoint of the square-root X gate."""

    name = "sxdg"

    def __init__(self) -> None:
        """Initializes the adjoint SX gate."""
        mat = 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=np.complex128)
        super().__init__(mat)


class Rx(BaseGate):
    """X-axis rotation ``exp(-1j * theta * X / 2)``."""

    name = "rx"

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the rotation gate about the x-axis.

        Args:
            params: Rotation angle ``[theta]`` in radians.
        """
        self.theta = params[0]
        mat = np.array([
            [np.cos(self.theta / 2), -1j * np.sin(self.theta / 2)],
            [-1j * np.sin(self.theta / 2), np.cos(self.theta / 2)],
        ])
        super().__init__(mat)


class Ry(BaseGate):
    """Y-axis rotation ``exp(-1j * theta * Y / 2)``."""

    name = "ry"

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the rotation gate about the y-axis.

        Args:
            params: Rotation angle ``[theta]`` in radians.
        """
        self.theta = params[0]
        mat = np.array([
            [np.cos(self.theta / 2), -np.sin(self.theta / 2)],
            [np.sin(self.theta / 2), np.cos(self.theta / 2)],
        ])
        super().__init__(mat)


class Rz(BaseGate):
    """Z-axis rotation ``exp(-1j * theta * Z / 2)``."""

    name = "rz"

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the rotation gate about the z-axis.

        Args:
            params: Rotation angle ``[theta]`` in radians.
        """
        self.theta = params[0]
        mat = np.array([
            [np.exp(-1j * self.theta / 2), 0],
            [0, np.exp(1j * self.theta / 2)],
        ])
        super().__init__(mat)


class Phase(BaseGate):
    """Single-qubit phase gate ``diag(1, exp(1j * theta))``."""

    name = "p"

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the phase gate.

        Args:
            params: Rotation angle ``[theta]`` in radians.
        """
        self.theta = params[0]
        mat = np.array([[1, 0], [0, np.exp(1j * self.theta)]])
        super().__init__(mat)


class U2(BaseGate):
    """Single-qubit :class:`U` gate with ``theta = pi / 2``."""

    name = "u2"

    def __init__(self, params: list[Parameter]) -> None:
        """Set the two phase angles of the U2 gate.

        Args:
            params: Angles ``[phi, lambda]`` in radians.
        """
        self.phi, self.lam = params

        inv_sqrt2 = 1 / np.sqrt(2)
        mat = inv_sqrt2 * np.array(
            [[1, -np.exp(1j * self.lam)], [np.exp(1j * self.phi), np.exp(1j * (self.phi + self.lam))]],
            dtype=np.complex128,
        )

        super().__init__(mat)


class U(BaseGate):
    """Single-qubit Euler rotation in Qiskit U convention: ``[theta, phi, lambda]``."""

    name = "u"

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the U3 gate.

        Args:
            params: Rotation angles ``[theta, phi, lambda]`` in radians.
        """
        self.theta, self.phi, self.lam = params
        mat = np.array([
            [np.cos(self.theta / 2), -np.exp(1j * self.lam) * np.sin(self.theta / 2)],
            [
                np.exp(1j * self.phi) * np.sin(self.theta / 2),
                np.exp(1j * (self.phi + self.lam)) * np.cos(self.theta / 2),
            ],
        ])
        super().__init__(mat)


class CX(BaseGate):
    """Controlled-X (CNOT) gate; sites list the control before the target."""

    name = "cx"

    def __init__(self) -> None:
        """Initializes the controlled-NOT (CX) gate."""
        super().__init__(CONTROLLED_X.copy())

    def set_sites(self, *sites: int | list[int]) -> None:
        """Bind sites and the product-form generator; see :meth:`BaseGate.set_sites`.

        Raises:
            ValueError: If the site count differs from ``interaction``.
        """
        sites_list = []
        for s in sites:
            if isinstance(s, int):
                sites_list.append(s)
            else:
                sites_list.extend(s)

        if len(sites_list) != self.interaction:
            msg = f"Number of sites {len(sites_list)} must be equal to the interaction level {self.interaction}"
            raise ValueError(msg)

        self.sites = sites_list
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        # Generator: π / 4 * ((I - Z) ⊗ (I - X))
        self.generator = [
            (np.pi / 4) * np.array([[0, 0], [0, 2]], dtype=np.complex128),
            np.array([[1, -1], [-1, 1]], dtype=np.complex128),
        ]
        self.mpo_tensors = extend_gate(self.tensor, self.sites)
        if self.sites[1] < self.sites[0]:  # Adjust for reverse control/target
            self.tensor = np.transpose(self.tensor, (1, 0, 3, 2))


class CZ(BaseGate):
    """Controlled-Z gate, which changes the sign of the local ``|11>`` state."""

    name = "cz"

    def __init__(self) -> None:
        """Initializes the controlled-Z (CZ) gate."""
        super().__init__(CONTROLLED_Z.copy())

    def set_sites(self, *sites: int | list[int]) -> None:
        """Bind sites and the product-form generator; see :meth:`BaseGate.set_sites`.

        Raises:
            ValueError: If the site count differs from ``interaction``.
        """
        sites_list = []
        for s in sites:
            if isinstance(s, int):
                sites_list.append(s)
            else:
                sites_list.extend(s)

        if len(sites_list) != self.interaction:
            msg = f"Number of sites {len(sites_list)} must be equal to the interaction level {self.interaction}"
            raise ValueError(msg)

        self.sites = sites_list
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        # Generator: π/4 * ((I - Z) ⊗ (I - Z))
        self.generator = [
            (np.pi / 4) * np.array([[0, 0], [0, 2]], dtype=np.complex128),
            np.array([[0, 0], [0, 2]], dtype=np.complex128),
        ]
        self.mpo_tensors = extend_gate(self.tensor, self.sites)
        if self.sites[1] < self.sites[0]:  # Adjust for reverse control/target
            self.tensor = np.transpose(self.tensor, (1, 0, 3, 2))


class CCX(BaseGate):
    """Toffoli gate; sites list the two controls before the target."""

    name = "ccx"

    def __init__(self) -> None:
        """Initializes the double-controlled NOT (CCX) gate."""
        mat = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ])
        super().__init__(mat)

    def set_sites(self, *sites: int | list[int]) -> None:
        """Bind sites and the product-form generator; see :meth:`BaseGate.set_sites`."""
        super().set_sites(*sites)
        # Generator: π/4 * ((I - Z) ⊗ P1 ⊗ (I - X))
        self.generator = [
            (np.pi / 4) * np.array([[0, 0], [0, 2]], dtype=np.complex128),
            np.array([[0, 0], [0, 1]], dtype=np.complex128),
            np.array([[1, -1], [-1, 1]], dtype=np.complex128),
        ]


class CCZ(BaseGate):
    """Double-controlled Z gate, which changes the sign of the local ``|111>`` state."""

    name = "ccz"

    def __init__(self) -> None:
        """Initializes the double-controlled Z (CCZ) gate."""
        mat = np.diag([1, 1, 1, 1, 1, 1, 1, -1])
        super().__init__(mat)

    def set_sites(self, *sites: int | list[int]) -> None:
        """Bind sites and the product-form generator; see :meth:`BaseGate.set_sites`."""
        super().set_sites(*sites)
        # Generator: π/4 * ((I - Z) ⊗ P1 ⊗ (I - Z))
        self.generator = [
            (np.pi / 4) * np.array([[0, 0], [0, 2]], dtype=np.complex128),
            np.array([[0, 0], [0, 1]], dtype=np.complex128),
            np.array([[0, 0], [0, 2]], dtype=np.complex128),
        ]


class CPhase(BaseGate):
    """Controlled phase gate ``diag(1, 1, 1, exp(1j * theta))``."""

    name = "cp"

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the gate.

        Args:
            params: Rotation angle ``[theta]`` in radians.
        """
        self.theta = params[0]
        mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(1j * self.theta)]])
        super().__init__(mat)

    def set_sites(self, *sites: int | list[int]) -> None:
        """Bind sites and the product-form generator; see :meth:`BaseGate.set_sites`.

        Raises:
            ValueError: If the site count differs from ``interaction``.
        """
        sites_list = []
        for s in sites:
            if isinstance(s, int):
                sites_list.append(s)
            else:
                sites_list.extend(s)

        if len(sites_list) != self.interaction:
            msg = f"Number of sites {len(sites_list)} must be equal to the interaction level {self.interaction}"
            raise ValueError(msg)

        self.sites = sites_list
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        # Generator: -θ * (P1 ⊗ P1)
        self.generator = [
            -self.theta * np.array([[0, 0], [0, 1]], dtype=np.complex128),
            np.array([[0, 0], [0, 1]], dtype=np.complex128),
        ]
        self.mpo_tensors = extend_gate(self.tensor, self.sites)


class SWAP(BaseGate):
    """Exchange the two specified qubit states."""

    name = "swap"

    def __init__(self) -> None:
        """Initializes the SWAP gate."""
        super().__init__(SWAP_MATRIX.copy())

    def set_sites(self, *sites: int | list[int]) -> None:
        """Sets the sites for the gate.

        Args:
            *sites: Variable-length argument list specifying site indices.

        Raises:
            ValueError: If the number of sites does not match the interaction level of the gate.
        """
        sites_list = []
        for s in sites:
            if isinstance(s, int):
                sites_list.append(s)
            else:
                sites_list.extend(s)

        if len(sites_list) != self.interaction:
            msg = f"Number of sites {len(sites_list)} must be equal to the interaction level {self.interaction}"
            raise ValueError(msg)

        self.sites = sites_list
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        self.mpo_tensors = extend_gate(self.tensor, self.sites)


class CSWAP(BaseGate):
    """Fredkin gate; sites list the control before the two swapped qubits.

    Has no product-form generator and uses its MPO representation.
    """

    name = "cswap"

    def __init__(self) -> None:
        """Initializes the controlled-SWAP (CSWAP) gate."""
        mat = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])
        super().__init__(mat)


class Rxx(BaseGate):
    """Two-qubit Pauli rotation ``exp(-1j * theta * (X ⊗ X) / 2)``."""

    name = "rxx"
    interaction = 2

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the gate.

        Args:
            params: Rotation angle ``[theta]`` in radians.
        """
        self.theta = params[0]
        mat = np.array([
            [np.cos(self.theta / 2), 0, 0, -1j * np.sin(self.theta / 2)],
            [0, np.cos(self.theta / 2), -1j * np.sin(self.theta / 2), 0],
            [0, -1j * np.sin(self.theta / 2), np.cos(self.theta / 2), 0],
            [-1j * np.sin(self.theta / 2), 0, 0, np.cos(self.theta / 2)],
        ])
        super().__init__(mat)

    def set_sites(self, *sites: int | list[int]) -> None:
        """Bind sites and the product-form generator; see :meth:`BaseGate.set_sites`.

        Raises:
            ValueError: If the site count differs from ``interaction``.
        """
        sites_list = []
        for s in sites:
            if isinstance(s, int):
                sites_list.append(s)
            else:
                sites_list.extend(s)

        if len(sites_list) != self.interaction:
            msg = f"Number of sites {len(sites_list)} must be equal to the interaction level {self.interaction}"
            raise ValueError(msg)

        self.sites = sites_list
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        self.generator = [(self.theta / 2) * np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])]
        self.mpo_tensors = extend_gate(self.tensor, self.sites)


class Ryy(BaseGate):
    """Two-qubit Pauli rotation ``exp(-1j * theta * (Y ⊗ Y) / 2)``."""

    name = "ryy"
    interaction = 2

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the gate.

        Args:
            params: Rotation angle ``[theta]`` in radians.
        """
        self.theta = params[0]
        mat = np.array([
            [np.cos(self.theta / 2), 0, 0, 1j * np.sin(self.theta / 2)],
            [0, np.cos(self.theta / 2), -1j * np.sin(self.theta / 2), 0],
            [0, -1j * np.sin(self.theta / 2), np.cos(self.theta / 2), 0],
            [1j * np.sin(self.theta / 2), 0, 0, np.cos(self.theta / 2)],
        ])
        super().__init__(mat)

    def set_sites(self, *sites: int | list[int]) -> None:
        """Bind sites and the product-form generator; see :meth:`BaseGate.set_sites`.

        Raises:
            ValueError: If the site count differs from ``interaction``.
        """
        sites_list = []
        for s in sites:
            if isinstance(s, int):
                sites_list.append(s)
            else:
                sites_list.extend(s)

        if len(sites_list) != self.interaction:
            msg = f"Number of sites {len(sites_list)} must be equal to the interaction level {self.interaction}"
            raise ValueError(msg)

        self.sites = sites_list
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        self.generator = [(self.theta / 2) * np.array([[0, -1j], [1j, 0]]), np.array([[0, -1j], [1j, 0]])]
        self.mpo_tensors = extend_gate(self.tensor, self.sites)


class Rzz(BaseGate):
    """Two-qubit Pauli rotation ``exp(-1j * theta * (Z ⊗ Z) / 2)``."""

    name = "rzz"
    interaction = 2

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the gate.

        Args:
            params: Rotation angle ``[theta]`` in radians.
        """
        self.theta = params[0]
        mat = np.array([
            [np.cos(self.theta / 2) - 1j * np.sin(self.theta / 2), 0, 0, 0],
            [0, np.cos(self.theta / 2) + 1j * np.sin(self.theta / 2), 0, 0],
            [0, 0, np.cos(self.theta / 2) + 1j * np.sin(self.theta / 2), 0],
            [0, 0, 0, np.cos(self.theta / 2) - 1j * np.sin(self.theta / 2)],
        ])
        super().__init__(mat)

    def set_sites(self, *sites: int | list[int]) -> None:
        """Bind sites and the product-form generator; see :meth:`BaseGate.set_sites`.

        Raises:
            ValueError: If the site count differs from ``interaction``.
        """
        sites_list = []
        for s in sites:
            if isinstance(s, int):
                sites_list.append(s)
            else:
                sites_list.extend(s)

        if len(sites_list) != self.interaction:
            msg = f"Number of sites {len(sites_list)} must be equal to the interaction level {self.interaction}"
            raise ValueError(msg)

        self.sites = sites_list
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        self.generator = [(self.theta / 2) * np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, -1]])]
        self.mpo_tensors = extend_gate(self.tensor, self.sites)


class GateLibrary:
    """Named gate classes for constructing local operators.

    Each attribute is a :class:`BaseGate` subclass. Construct gates with
    ``GateLibrary.rx([theta])`` or ``GateLibrary.cz()``, then bind their sites.
    Aliases include ``u3`` for ``u``, ``u1`` for ``p``, and ``i`` and ``iden`` for ``id``.
    """

    x = X
    y = Y
    z = Z
    sx = SX
    sxdg = SXdg
    s = S
    sdg = Sdg
    t = T
    tdg = Tdg
    h = H
    id = Id
    i = Id
    iden = Id

    rx = Rx
    ry = Ry
    rz = Rz
    u = U
    u3 = U
    u2 = U2
    u1 = Phase

    cx = CX
    cz = CZ
    ccx = CCX
    ccz = CCZ
    swap = SWAP
    cswap = CSWAP

    rxx = Rxx
    ryy = Ryy
    rzz = Rzz

    cp = CPhase
    p = Phase

    destroy = Destroy
    create = Create

    custom = BaseGate
