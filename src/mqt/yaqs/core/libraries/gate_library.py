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

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from qiskit.circuit import Parameter


def split_tensor(tensor: NDArray[np.complex128]) -> list[NDArray[np.complex128]]:
    """Splits a multi-qubit gate tensor into one tensor per site using Singular Value Decomposition (SVD).

    Args:
        tensor: A gate tensor of shape ``(2,) * (2 * n)`` for ``n >= 2`` sites, with index
            order ``(out_1, ..., out_n, in_1, ..., in_n)``.

    Returns:
        list[NDArray[np.complex128]]: A list containing one tensor per site resulting from the split.
            Each tensor has shape (2, 2, bond_left, bond_right); the outer bonds are 1.
    """
    num_sites = tensor.ndim // 2
    assert num_sites >= 2
    assert tensor.shape == (2,) * (2 * num_sites)

    # Group the output and input leg of each site: (out_1, in_1, ..., out_n, in_n)
    matrix = np.transpose(tensor, [axis for site in range(num_sites) for axis in (site, num_sites + site)])

    # Split site by site with SVDs, carrying the singular values to the right
    tensors = []
    left_bond = 1
    remaining = np.reshape(matrix, (left_bond * 4, 4 ** (num_sites - 1)))
    for _ in range(num_sites - 1):
        u_mat, s_list, v_mat = linalg.svd(remaining, full_matrices=False)
        keep = linalg.truncate(s_list, mode="hard_cutoff", threshold=1e-6, min_keep=1)
        s_list = s_list[:keep]
        u_mat = u_mat[:, :keep]
        v_mat = v_mat[:keep, :]
        tensors.append(np.transpose(np.reshape(u_mat, (left_bond, 2, 2, keep)), (1, 2, 0, 3)))
        left_bond = keep
        remaining = np.reshape(np.diag(s_list) @ v_mat, (left_bond * 4, remaining.shape[1] // 4))

    last_tensor = np.transpose(np.reshape(remaining, (left_bond, 2, 2)), (1, 2, 0))
    tensors.append(np.expand_dims(last_tensor, axis=3))
    return tensors


def extend_gate(tensor: NDArray[np.complex128], sites: list[int]) -> list[NDArray[np.complex128]]:
    """Extends gate to long-range MPO.

    Extends a given gate tensor to a Matrix Product Operator (MPO) by adding identity tensors
    between specified sites.

    Args:
        tensor: The input gate tensor to be extended.
        sites: A list of site indices where the gate tensor is to be applied.

    Returns:
        MPO: The resulting Matrix Product Operator with the gate tensor extended over the specified sites.

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

    tensors = split_tensor(tensor)

    # Adds identity tensors between sites
    mpo_tensors = [tensors[0]]
    for idx in range(1, num_sites):
        for _ in range(sorted_sites[idx] - sorted_sites[idx - 1] - 1):
            previous_right_bond = mpo_tensors[-1].shape[3]
            identity_tensor = np.zeros((2, 2, previous_right_bond, previous_right_bond), dtype=np.complex128)
            for i in range(previous_right_bond):
                identity_tensor[:, :, i, i] = np.identity(2)
            mpo_tensors.append(identity_tensor)
        mpo_tensors.append(tensors[idx])

    return mpo_tensors


class BaseGate:
    """Base class representing a quantum gate.

    Attributes:
        name: The name of the gate.
        matrix: The matrix representation of the gate.
        interaction: The interaction type or level of the gate.
        tensor: The tensor representation of the gate.
        generator: The generator(s) for the gate.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites on which the gate acts.
    """

    name: str = "custom"
    matrix: NDArray[np.complex128]
    interaction: int
    tensor: NDArray[np.complex128]
    generator: NDArray[np.complex128] | list[NDArray[np.complex128]]
    sites: list[int]

    def __init__(self, mat: ArrayLike) -> None:
        """Initializes a BaseGate instance with the given matrix.

        Args:
            mat: The matrix representation of the gate.

        Raises:
            ValueError: If the matrix is not a square 2-D array.
            ValueError: If the matrix dimension is not a power of 2.
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

    @classmethod
    def p0(cls) -> P0:
        """Returns the P0 projector.

        Returns:
            An instance of the P0 gate.
        """
        return P0()

    @classmethod
    def p1(cls) -> P1:
        """Returns the P1 projector.

        Returns:
            An instance of the P1 gate.
        """
        return P1()

    @classmethod
    def pvm(cls, bitstring: str) -> PVM:
        """Create a projection-valued measurement (PVM) operator.

        Args:
            bitstring: The computational basis bitstring (e.g., "0101") that the state
                should be projected onto.

        Returns:
            An instance of the PVM gate representing the projection.
        """
        return PVM(bitstring)

    @classmethod
    def entropy(cls) -> Entropy:
        """Create an entropy diagnostic operator.

        This is a meta-observable used to request the bipartite entanglement
        entropy across a given nearest-neighbor cut.

        Returns:
            An instance of the entropy diagnostic gate.
        """
        return Entropy()

    @classmethod
    def schmidt_spectrum(cls) -> SchmidtSpectrum:
        """Create a Schmidt spectrum diagnostic operator.

        This is a meta-observable used to request the Schmidt coefficients
        across a given nearest-neighbor cut, padded or truncated to a fixed length.

        Returns:
            An instance of the Schmidt spectrum diagnostic gate.
        """
        return SchmidtSpectrum()

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
    """Class representing the Pauli-X (NOT) gate.

    Attributes:
        name: The name of the gate ("x").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "x"

    def __init__(self) -> None:
        """Initializes the Pauli-X gate."""
        mat = np.array([[0, 1], [1, 0]])
        super().__init__(mat)


class Y(BaseGate):
    """Class representing the Pauli-Y gate.

    Attributes:
        name: The name of the gate ("y").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "y"

    def __init__(self) -> None:
        """Initializes the Pauli-Y gate."""
        mat = np.array([[0, -1j], [1j, 0]])
        super().__init__(mat)


class Z(BaseGate):
    """Class representing the Pauli-Z gate.

    Attributes:
        name: The name of the gate ("z").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "z"

    def __init__(self) -> None:
        """Initializes the Pauli-Z gate."""
        mat = np.array([[1, 0], [0, -1]])
        super().__init__(mat)


class H(BaseGate):
    """Class representing the Hadamard (H) gate.

    Attributes:
        name: The name of the gate ("h").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "h"

    def __init__(self) -> None:
        """Initializes the Hadamard gate."""
        mat = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]])
        super().__init__(mat)


class Destroy(BaseGate):
    """Class representing the Destroy (annihilation) gate.

    Attributes:
        name: The name of the gate ("destroy").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

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
    """Class representing the Create (creation) gate.

    Attributes:
        name: The name of the gate ("create").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

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
    """Class representing the identity (Id) gate.

    Attributes:
        name: The name of the gate ("id").
        matrix: The 2x2 identity matrix.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "id"

    def __init__(self) -> None:
        """Initializes the identity gate."""
        mat = np.array([[1, 0], [0, 1]])
        super().__init__(mat)


class SX(BaseGate):
    """Class representing the square-root X (SX) gate.

    Attributes:
        name: The name of the gate ("sx").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "sx"

    def __init__(self) -> None:
        """Initializes the square-root X gate."""
        mat = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128)
        super().__init__(mat)


class Sdg(BaseGate):
    """Class representing the adjoint S (S-dagger) gate.

    Attributes:
        name: The name of the gate ("sdg").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "sdg"

    def __init__(self) -> None:
        """Initializes the adjoint S gate."""
        mat = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
        super().__init__(mat)


class S(BaseGate):
    """Class representing the S gate.

    Attributes:
        name: The name of the gate ("s").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "s"

    def __init__(self) -> None:
        """Initializes the S gate."""
        mat = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
        super().__init__(mat)


class Tdg(BaseGate):
    """Class representing the adjoint T (T-dagger) gate.

    Attributes:
        name: The name of the gate ("tdg").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "tdg"

    def __init__(self) -> None:
        """Initializes the adjoint T gate."""
        mat = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)
        super().__init__(mat)


class T(BaseGate):
    """Class representing the T gate.

    Attributes:
        name: The name of the gate ("t").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "t"

    def __init__(self) -> None:
        """Initializes the T gate."""
        mat = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
        super().__init__(mat)


class SXdg(BaseGate):
    """Class representing the adjoint SX (SX-dagger) gate.

    Attributes:
        name: The name of the gate ("sxdg").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "sxdg"

    def __init__(self) -> None:
        """Initializes the adjoint SX gate."""
        mat = 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=np.complex128)
        super().__init__(mat)


class Rx(BaseGate):
    """Class representing a rotation gate about the x-axis.

    Attributes:
        name: The name of the gate ("rx").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).
        theta: The rotation angle parameter.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "rx"

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the rotation gate about the x-axis.

        Args:
            params : list[Parameter]
            A list containing a single rotation angle (`theta`) parameter.
        """
        self.theta = params[0]
        mat = np.array([
            [np.cos(self.theta / 2), -1j * np.sin(self.theta / 2)],
            [-1j * np.sin(self.theta / 2), np.cos(self.theta / 2)],
        ])
        super().__init__(mat)


class Ry(BaseGate):
    """Class representing a rotation gate about the y-axis.

    Attributes:
        name: The name of the gate ("ry").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).
        theta: The rotation angle parameter.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "ry"

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the rotation gate about the y-axis.

        Args:
            params : list[Parameter]
            A list containing a single rotation angle (`theta`) parameter.
        """
        self.theta = params[0]
        mat = np.array([
            [np.cos(self.theta / 2), -np.sin(self.theta / 2)],
            [np.sin(self.theta / 2), np.cos(self.theta / 2)],
        ])
        super().__init__(mat)


class Rz(BaseGate):
    """Class representing a rotation gate about the z-axis.

    Attributes:
        name: The name of the gate ("rz").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).
        theta: The rotation angle parameter.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "rz"

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the rotation gate about the z-axis.

        Args:
            params : list[Parameter]
            A list containing a single rotation angle (`theta`) parameter.
        """
        self.theta = params[0]
        mat = np.array([
            [np.exp(-1j * self.theta / 2), 0],
            [0, np.exp(1j * self.theta / 2)],
        ])
        super().__init__(mat)


class Phase(BaseGate):
    """Class representing a phase gate.

    Attributes:
        name: The name of the gate ("p").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).
        theta: The phase angle parameter.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "p"

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the phase gate.

        Args:
            params : list[Parameter]
            A list containing a single rotation angle (`theta`) parameter.
        """
        self.theta = params[0]
        mat = np.array([[1, 0], [0, np.exp(1j * self.theta)]])
        super().__init__(mat)


class U2(BaseGate):
    """Class representing a U2 gate.

    Attributes:
        name: The name of the gate ("u2").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).
        phi: The first rotation parameter.
        lam: The second rotation parameter.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "u2"

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the U2 gate.

        Args:
            params: list[Parameter]
                A list containing two rotation angles [phi, lambda].
        """
        self.phi, self.lam = params

        inv_sqrt2 = 1 / np.sqrt(2)
        mat = inv_sqrt2 * np.array(
            [[1, -np.exp(1j * self.lam)], [np.exp(1j * self.phi), np.exp(1j * (self.phi + self.lam))]],
            dtype=np.complex128,
        )

        super().__init__(mat)


class U(BaseGate):
    """Class representing a U3 gate.

    Attributes:
        name: The name of the gate ("u").
        matrix: The 2x2 matrix representation of the gate.
        interaction: The interaction level (1 for single-qubit gates).
        tensor: The tensor representation of the gate (same as the matrix).
        theta: The first rotation parameter.
        phi: The second rotation parameter.
        lam: The third rotation parameter.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "u"

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the U3 gate.

        Args:
            params : list[Parameter]
            A list containing a three rotation angle (theta, phi, lambda) parameters.
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
    """Class representing the controlled-NOT (CX) gate.

    Attributes:
        name: The name of the gate ("cx").
        matrix: The 4x4 matrix representation of the gate.
        interaction: The interaction level (2 for two-qubit gates).
        tensor: The tensor representation reshaped to (2, 2, 2, 2).
        generator: The generator for the gate.
        mpo: An MPO representation generated from the gate tensor.
        sites: The control and target sites.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites and updates the tensor and MPO.
    """

    name = "cx"

    def __init__(self) -> None:
        """Initializes the controlled-NOT (CX) gate."""
        mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        super().__init__(mat)

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
        # Generator: π / 4 * ((I - Z) ⊗ (I - X))
        self.generator = [
            (np.pi / 4) * np.array([[0, 0], [0, 2]], dtype=np.complex128),
            np.array([[1, -1], [-1, 1]], dtype=np.complex128),
        ]
        self.mpo_tensors = extend_gate(self.tensor, self.sites)
        if self.sites[1] < self.sites[0]:  # Adjust for reverse control/target
            self.tensor = np.transpose(self.tensor, (1, 0, 3, 2))


class CZ(BaseGate):
    """Class representing the controlled-Z (CZ) gate.

    Attributes:
        name: The name of the gate ("cz").
        matrix: The 4x4 matrix representation of the gate.
        interaction: The interaction level (2 for two-qubit gates).
        tensor: The tensor representation reshaped to (2, 2, 2, 2).
        generator: The generator for the gate.
        sites: The control and target sites.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites and updates the tensor.
    """

    name = "cz"

    def __init__(self) -> None:
        """Initializes the controlled-Z (CZ) gate."""
        mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
        super().__init__(mat)

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
        # Generator: π/4 * ((I - Z) ⊗ (I - X))
        self.generator = [
            (np.pi / 4) * np.array([[0, 0], [0, 2]], dtype=np.complex128),
            np.array([[1, -1], [-1, 1]], dtype=np.complex128),
        ]
        self.mpo_tensors = extend_gate(self.tensor, self.sites)
        if self.sites[1] < self.sites[0]:  # Adjust for reverse control/target
            self.tensor = np.transpose(self.tensor, (1, 0, 3, 2))


class CCX(BaseGate):
    """Class representing the double-controlled NOT (Toffoli, CCX) gate.

    Attributes:
        name: The name of the gate ("ccx").
        matrix: The 8x8 matrix representation of the gate.
        interaction: The interaction level (3 for three-qubit gates).
        tensor: The tensor representation reshaped to (2, 2, 2, 2, 2, 2).
        generator: The generator for the gate.
        mpo_tensors: An MPO representation generated from the gate tensor.
        sites: The two control sites and the target site.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites and updates the tensor and MPO.
    """

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
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2, 2, 2))
        # Generator: π/4 * ((I - Z) ⊗ P1 ⊗ (I - X))
        self.generator = [
            (np.pi / 4) * np.array([[0, 0], [0, 2]], dtype=np.complex128),
            np.array([[0, 0], [0, 1]], dtype=np.complex128),
            np.array([[1, -1], [-1, 1]], dtype=np.complex128),
        ]
        self.mpo_tensors = extend_gate(self.tensor, self.sites)


class CCZ(BaseGate):
    """Class representing the double-controlled Z (CCZ) gate.

    Attributes:
        name: The name of the gate ("ccz").
        matrix: The 8x8 matrix representation of the gate.
        interaction: The interaction level (3 for three-qubit gates).
        tensor: The tensor representation reshaped to (2, 2, 2, 2, 2, 2).
        generator: The generator for the gate.
        mpo_tensors: An MPO representation generated from the gate tensor.
        sites: The two control sites and the target site.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites and updates the tensor and MPO.
    """

    name = "ccz"

    def __init__(self) -> None:
        """Initializes the double-controlled Z (CCZ) gate."""
        mat = np.diag([1, 1, 1, 1, 1, 1, 1, -1])
        super().__init__(mat)

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
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2, 2, 2))
        # Generator: π/4 * ((I - Z) ⊗ P1 ⊗ (I - Z))
        self.generator = [
            (np.pi / 4) * np.array([[0, 0], [0, 2]], dtype=np.complex128),
            np.array([[0, 0], [0, 1]], dtype=np.complex128),
            np.array([[0, 0], [0, 2]], dtype=np.complex128),
        ]
        self.mpo_tensors = extend_gate(self.tensor, self.sites)


class CPhase(BaseGate):
    """Class representing the controlled phase (CPhase) gate.

    Attributes:
        name: The name of the gate ("cp").
        matrix: The 4x4 matrix representation of the gate.
        interaction: The interaction level (2 for two-qubit gates).
        tensor: The tensor representation reshaped to (2, 2, 2, 2).
        generator: The generator for the gate.
        sites: The control and target sites.
        theta: The angle parameter.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites and updates the tensor.
    """

    name = "cp"

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the gate.

        Args:
            params : list[Parameter]
            A list containing a single rotation angle (`theta`) parameter.
        """
        self.theta = params[0]
        mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(1j * self.theta)]])
        super().__init__(mat)

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
        self.generator = [(self.theta / 2) * np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, 0]])]
        self.mpo_tensors = extend_gate(self.tensor, self.sites)


class SWAP(BaseGate):
    """Class representing the SWAP gate.

    Attributes:
        name: The name of the gate ("swap").
        matrix: The 4x4 matrix representation of the gate.
        interaction: The interaction level (2 for two-qubit gates).
        tensor: The tensor representation reshaped to (2, 2, 2, 2).
        sites: The sites involved in the swap.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites and updates the tensor.
    """

    name = "swap"

    def __init__(self) -> None:
        """Initializes the SWAP gate."""
        mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        super().__init__(mat)

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
    """Class representing the controlled-SWAP (Fredkin, CSWAP) gate.

    The SWAP part of the gate has no single-product generator, so the gate carries no
    ``generator`` attribute and is applied via its MPO representation.

    Attributes:
        name: The name of the gate ("cswap").
        matrix: The 8x8 matrix representation of the gate.
        interaction: The interaction level (3 for three-qubit gates).
        tensor: The tensor representation reshaped to (2, 2, 2, 2, 2, 2).
        mpo_tensors: An MPO representation generated from the gate tensor.
        sites: The control site and the two swapped sites.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites and updates the tensor and MPO.
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
    """Class representing a two-qubit rotation gate about the xx-axis.

    Attributes:
        name: The name of the gate ("rxx").
        matrix: The 4x4 matrix representation of the gate.
        interaction: The interaction level (2 for two-qubit gates).
        tensor: The tensor representation reshaped to (2, 2, 2, 2).
        generator: The generator for the gate.
        sites: The control and target sites.
        theta: The angle parameter.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites and updates the tensor.
    """

    name = "rxx"
    interaction = 2

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the gate.

        Args:
            params : list[Parameter]
            A list containing a single rotation angle (`theta`) parameter.
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
        self.generator = [(self.theta / 2) * np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])]
        self.mpo_tensors = extend_gate(self.tensor, self.sites)


class Ryy(BaseGate):
    """Class representing a two-qubit rotation gate about the yy-axis.

    Attributes:
        name: The name of the gate ("ryy").
        matrix: The 4x4 matrix representation of the gate.
        interaction: The interaction level (2 for two-qubit gates).
        tensor: The tensor representation reshaped to (2, 2, 2, 2).
        generator: The generator for the gate.
        sites: The control and target sites.
        theta: The angle parameter.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites and updates the tensor.
    """

    name = "ryy"
    interaction = 2

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the gate.

        Args:
            params : list[Parameter]
            A list containing a single rotation angle (`theta`) parameter.
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
        self.generator = [(self.theta / 2) * np.array([[0, -1j], [1j, 0]]), np.array([[0, -1j], [1j, 0]])]
        self.mpo_tensors = extend_gate(self.tensor, self.sites)


class Rzz(BaseGate):
    """Class representing a two-qubit rotation gate about the zz-axis.

    Attributes:
        name: The name of the gate ("rzz").
        matrix: The 4x4 matrix representation of the gate.
        interaction: The interaction level (2 for two-qubit gates).
        tensor: The tensor representation reshaped to (2, 2, 2, 2).
        generator: The generator for the gate.
        sites: The control and target sites.
        theta: The angle parameter.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites and updates the tensor.
    """

    name = "rzz"
    interaction = 2

    def __init__(self, params: list[Parameter]) -> None:
        """Initializes the gate.

        Args:
            params : list[Parameter]
            A list containing a single rotation angle (`theta`) parameter.
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
        self.generator = [(self.theta / 2) * np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, -1]])]
        self.mpo_tensors = extend_gate(self.tensor, self.sites)


class XX(BaseGate):
    """Class representing an XX operation. Used for two-site correlators.

    Attributes:
        name: The name of the gate ("xx").
        matrix: The 4x4 matrix representation of the gate.
        interaction: The interaction level (2 for two-qubit gates).
        tensor: The tensor representation reshaped to (2, 2, 2, 2).
        mpo: An MPO representation generated from the gate tensor.
        sites: The control and target sites.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites and updates the tensor and MPO.
    """

    name = "xx"

    def __init__(self) -> None:
        """Initializes the XX gate."""
        x = X().matrix
        # two-site operator X ⊗ X
        mat = np.kron(x, x).astype(np.complex128)
        super().__init__(mat)


class YY(BaseGate):
    """Class representing an YY operation. Used for two-site correlators.

    Attributes:
        name: The name of the gate ("yy").
        matrix: The 4x4 matrix representation of the gate.
        interaction: The interaction level (2 for two-qubit gates).
        tensor: The tensor representation reshaped to (2, 2, 2, 2).
        mpo: An MPO representation generated from the gate tensor.
        sites: The control and target sites.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites and updates the tensor and MPO.
    """

    name = "yy"

    def __init__(self) -> None:
        """Initializes the YY gate."""
        y = Y().matrix
        # two-site operator Y ⊗ Y
        mat = np.kron(y, y).astype(np.complex128)
        super().__init__(mat)


class ZZ(BaseGate):
    """Class representing an ZZ operation. Used for two-site correlators.

    Attributes:
        name: The name of the gate ("zz").
        matrix: The 4x4 matrix representation of the gate.
        interaction: The interaction level (2 for two-qubit gates).
        tensor: The tensor representation reshaped to (2, 2, 2, 2).
        mpo: An MPO representation generated from the gate tensor.
        sites: The control and target sites.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites and updates the tensor and MPO.
    """

    name = "zz"

    def __init__(self) -> None:
        """Initializes the ZZ gate."""
        z = Z().matrix
        # two-site operator Z ⊗ Z
        mat = np.kron(z, z).astype(np.complex128)
        super().__init__(mat)


class P0(BaseGate):
    """Class representing the projector onto ``|0⟩⟨0|``.

    Attributes:
        name: The name of the gate ("p0").
        matrix: The 2x2 matrix representation of the projector.
        interaction: The interaction level (1 for single-qubit projectors).
        tensor: The tensor representation of the projector (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the projector is applied.
    """

    name = "p0"

    def __init__(self) -> None:
        """Initializes the ``|0⟩⟨0|`` projector."""
        mat = np.array([[1, 0], [0, 0]], dtype=complex)
        super().__init__(mat)


class P1(BaseGate):
    """Class representing the projector onto ``|1⟩⟨1|``.

    Attributes:
        name: The name of the gate ("p1").
        matrix: The 2x2 matrix representation of the projector.
        interaction: The interaction level (1 for single-qubit projectors).
        tensor: The tensor representation of the projector (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the projector is applied.
    """

    name = "p1"

    def __init__(self) -> None:
        """Initializes the ``|1⟩⟨1|`` projector."""
        mat = np.array([[0, 0], [0, 1]], dtype=complex)
        super().__init__(mat)


class PVM(BaseGate):
    """Class representing a projection-valued measurement.

    Attributes:
        name: The name of the gate ("pvm").
    """

    name = "pvm"

    def __init__(self, bitstring: str) -> None:
        """Initializes the projection."""
        self.bitstring = bitstring

        # Identity array as placeholder for compatibility
        mat = np.array([[1, 0], [0, 1]])
        super().__init__(mat)


class LocalOperator(BaseGate):
    """Custom one-site operator for arbitrary local Hilbert-space dimensions.

    This gate is intended for observables such as position-grid operators on
    qudits or oscillator truncations. Unlike :class:`BaseGate`, it does not
    interpret the matrix dimension as a qubit interaction count.
    """

    name = "local"

    def __init__(self, matrix: ArrayLike) -> None:
        """Create a one-site local operator.

        Args:
            matrix: Square matrix acting on one local site.

        Raises:
            ValueError: If ``matrix`` is not a square two-dimensional array.
        """
        mat = np.asarray(matrix, dtype=np.complex128)
        if mat.ndim != 2:
            msg = "Local operator matrix must be a 2-D array."
            raise ValueError(msg)
        if mat.shape[0] != mat.shape[1]:
            msg = "Local operator matrix must be square."
            raise ValueError(msg)
        self.matrix = mat
        self.tensor = mat
        self.interaction = 1


class Position(LocalOperator):
    """One-site position operator for a supplied position basis."""

    name = "position"

    def __init__(self, *, positions: ArrayLike) -> None:
        """Create a position operator that is diagonal in the supplied basis.

        Args:
            positions: One-dimensional position values defining the local basis.

        Raises:
            ValueError: If ``positions`` is complex or not a non-empty, finite one-dimensional array.
        """
        position_values = np.asarray(positions)
        if np.iscomplexobj(position_values):
            msg = "positions must contain only real values."
            raise ValueError(msg)
        position_values = np.asarray(position_values, dtype=np.float64)
        if position_values.ndim != 1 or position_values.size == 0:
            msg = "positions must be a non-empty one-dimensional array."
            raise ValueError(msg)
        if not np.all(np.isfinite(position_values)):
            msg = "positions must contain only finite values."
            raise ValueError(msg)
        super().__init__(np.diag(position_values))


class Entropy(BaseGate):
    """Meta-observable for bipartite entanglement entropy across a cut.

    The actual entropy is computed from the MPS; this gate serves as a
    typed handle so that high-level code can request this diagnostic via
    the same measurement interface.
    """

    name = "entropy"

    def __init__(self) -> None:
        """Creates a no-op placeholder matrix for BaseGate compatibility."""
        mat = np.array([[1, 0], [0, 1]], dtype=complex)
        super().__init__(mat)

    def set_sites(self, *sites: int | list[int]) -> None:
        """Sets the sites defining the bipartition (i, i+1).

        Args:
            *sites: One or two integers or a list of two integers indicating the cut.
        """
        sites_list: list[int] = []
        for s in sites:
            if isinstance(s, int):
                sites_list.append(s)
            else:
                sites_list.extend(s)
        self.sites = sites_list


class SchmidtSpectrum(BaseGate):
    """Meta-observable for the Schmidt spectrum across a nearest-neighbor cut.

    The spectrum (singular values) is computed from the MPS around the specified
    bond and returned as a fixed-length vector (padded/truncated as needed).
    """

    name = "schmidt_spectrum"

    def __init__(self) -> None:
        """Creates a no-op placeholder matrix for BaseGate compatibility."""
        mat = np.array([[1, 0], [0, 1]], dtype=complex)
        super().__init__(mat)

    def set_sites(self, *sites: int | list[int]) -> None:
        """Sets the sites defining the bipartition (i, i+1).

        Args:
            *sites: One or two integers or a list of two integers indicating the cut.
        """
        sites_list: list[int] = []
        for s in sites:
            if isinstance(s, int):
                sites_list.append(s)
            else:
                sites_list.extend(s)
        self.sites = sites_list


class GateLibrary:
    """A collection of quantum gate classes for use in simulations.

    This library exposes gate **classes** (not instances). Each attribute points to
    a concrete `BaseGate` subclass implementing the corresponding operator. Use
    them like `GateLibrary.rx(theta)` or `GateLibrary.cz()` (depending on your
    constructors), or via any factory utilities you provide.

    Attributes:
        x: Class for the Pauli-X gate.
        y: Class for the Pauli-Y gate.
        z: Class for the Pauli-Z gate.
        sx: Class for the √X (SX) gate.
        sxdg: Class for the adjoint √X (SX-dagger) gate.
        s: Class for the S gate.
        sdg: Class for the adjoint S (S-dagger) gate.
        t: Class for the T gate.
        tdg: Class for the adjoint T (T-dagger) gate.
        h: Class for the Hadamard gate.
        id: Class for the identity gate.
        i: Alias for the identity gate.
        iden: Alias for the identity gate.

        rx: Class for rotation about the X-axis.
        ry: Class for rotation about the Y-axis.
        rz: Class for rotation about the Z-axis.
        u:  Class for the generic single-qubit U gate.
        u3: Alias for the generic single-qubit U gate.
        u2: Class for the U2 (fixed-θ,φ) single-qubit gate.
        u1: Alias for the single-qubit phase gate.

        cx: Class for the controlled-NOT (CNOT) gate.
        cz: Class for the controlled-Z gate.
        ccx: Class for the double-controlled NOT (Toffoli) gate.
        ccz: Class for the double-controlled Z gate.
        swap: Class for the SWAP gate.
        cswap: Class for the controlled-SWAP (Fredkin) gate.

        rxx: Class for two-qubit rotation about XX.
        ryy: Class for two-qubit rotation about YY.
        rzz: Class for two-qubit rotation about ZZ.

        cp: Class for the controlled-phase gate.
        p:  Class for the single-qubit phase gate.

        destroy: Class for the annihilation operator (ladder operator a).
        create:  Class for the creation operator (ladder operator a†).

        xx: Class for the XX interaction (non-parameterized).
        yy: Class for the YY interaction (non-parameterized).
        zz: Class for the ZZ interaction (non-parameterized).

        p0: Class for projector ``|0⟩⟨0|``.
        p1: Class for projector ``|1⟩⟨1|``.
        pvm: Class for projection-valued measurement onto a given bitstring.
        local: Class for arbitrary one-site local operators.
        position: Class for a one-site position operator in a supplied position basis.

        entropy:      Class representing a request for bipartite entanglement entropy across a cut.
        schmidt_spectrum: Class representing a request for the Schmidt spectrum across a cut.

        custom: Base class hook for defining custom gates (falls back to `BaseGate`).
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

    xx = XX
    yy = YY
    zz = ZZ

    p0 = P0
    p1 = P1
    pvm = PVM
    local = LocalOperator
    position = Position

    entropy = Entropy
    schmidt_spectrum = SchmidtSpectrum

    custom = BaseGate
