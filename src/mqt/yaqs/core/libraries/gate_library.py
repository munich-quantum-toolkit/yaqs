# Copyright (c) 2025 Chair for Design Automation, TUM
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

from ..data_structures.networks import MPO

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.circuit import Parameter


def split_tensor(tensor: NDArray[np.complex128]) -> list[NDArray[np.complex128]]:
    """Splits a two-qubit tensor into two tensors using Singular Value Decomposition (SVD).

    Args:
        tensor (NDArray[np.complex128]): A 4-dimensional tensor with shape (2, 2, 2, 2).

    Returns:
        list[NDArray[np.complex128]]: A list containing two tensors resulting from the split.
            - The first tensor has shape (2, 2, bond_dimension, 1).
            - The second tensor has shape (2, 2, bond_dimension, 1).
    """
    assert tensor.shape == (2, 2, 2, 2)

    # Splits two-qubit matrix
    matrix = np.transpose(tensor, (0, 2, 1, 3))
    dims = matrix.shape
    matrix = np.reshape(matrix, (dims[0] * dims[1], dims[2] * dims[3]))
    u_mat, s_list, v_mat = np.linalg.svd(matrix, full_matrices=False)
    s_list = s_list[s_list > 1e-6]
    u_mat = u_mat[:, 0 : len(s_list)]
    v_mat = v_mat[0 : len(s_list), :]

    tensor1 = u_mat
    tensor2 = np.diag(s_list) @ v_mat

    # Reshape into physical dimensions and bond dimension
    tensor1 = np.reshape(tensor1, (2, 2, tensor1.shape[1]))
    tensor2 = np.reshape(tensor2, (tensor2.shape[0], 2, 2))
    tensor2 = np.transpose(tensor2, (1, 2, 0))

    # Add dummy dimension to boundaries
    tensor1 = np.expand_dims(tensor1, axis=2)
    tensor2 = np.expand_dims(tensor2, axis=3)
    return [tensor1, tensor2]


def extend_gate(tensor: NDArray[np.complex128], sites: list[int]) -> MPO:
    """Extends gate to long-range MPO.

    Extends a given gate tensor to a Matrix Product Operator (MPO) by adding identity tensors
    between specified sites.

    Args:
        tensor (NDArray[np.complex128]): The input gate tensor to be extended.
        sites (list[int]): A list of site indices where the gate tensor is to be applied.

    Returns:
        MPO: The resulting Matrix Product Operator with the gate tensor extended over the specified sites.

    Notes:
        - The function handles cases where the input tensor is split into either 2 or 3 tensors.
        - Identity tensors are inserted between the specified sites.
        - If the sites are provided in reverse order, the resulting MPO tensors are reversed and
          transposed accordingly.
    """
    tensors = split_tensor(tensor)
    if len(tensors) == 2:
        # Adds identity tensors between sites
        mpo_tensors = [tensors[0]]
        for _ in range(np.abs(sites[0] - sites[1]) - 1):
            previous_right_bond = mpo_tensors[-1].shape[3]
            identity_tensor = np.zeros((2, 2, previous_right_bond, previous_right_bond))
            for i in range(previous_right_bond):
                identity_tensor[:, :, i, i] = np.identity(2)
            mpo_tensors.append(identity_tensor)
        mpo_tensors.append(tensors[1])

        if sites[1] < sites[0]:
            mpo_tensors.reverse()
            for idx in range(len(mpo_tensors)):
                mpo_tensors[idx] = np.transpose(mpo_tensors[idx], (0, 1, 3, 2))

    elif len(tensors) == 3:
        mpo_tensors = [tensors[0]]
        for _ in range(np.abs(sites[0] - sites[1]) - 1):
            previous_right_bond = mpo_tensors[-1].shape[3]
            identity_tensor = np.zeros((2, 2, previous_right_bond, previous_right_bond))
            for i in range(previous_right_bond):
                identity_tensor[:, :, i, i] = np.identity(2)
            mpo_tensors.append(identity_tensor)
        mpo_tensors.append(tensors[1])
        for _ in range(np.abs(sites[1] - sites[2]) - 1):
            previous_right_bond = mpo_tensors[-1].shape[3]
            identity_tensor = np.zeros((2, 2, previous_right_bond, previous_right_bond))
            for i in range(previous_right_bond):
                identity_tensor[:, :, i, i] = np.identity(2)
            mpo_tensors.append(identity_tensor)
        mpo_tensors.append(tensors[2])

    mpo = MPO()
    mpo.init_custom(mpo_tensors, transpose=False)
    return mpo


class BaseGate:
    """Base class representing a quantum gate.

    Attributes:
        name (str): The name of the gate.
        matrix (NDArray[np.complex128]): The matrix representation of the gate.
        interaction (int): The interaction type or level of the gate.
        tensor (NDArray[np.complex128]): The tensor representation of the gate.
        generator (NDArray[np.complex128] | list[NDArray[np.complex128]]): The generator(s) for the gate.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites on which the gate acts.
    """

    name: str
    matrix: NDArray[np.complex128]
    interaction: int
    tensor: NDArray[np.complex128]
    generator: NDArray[np.complex128] | list[NDArray[np.complex128]]

    def __init__(self, mat: NDArray[np.complex128]) -> None:
        """Initializes a BaseGate instance with the given matrix.

        Args:
            mat (NDArray[np.complex128]): The matrix representation of the gate.

        Raises:
            ValueError: If the matrix is not square.
            ValueError: If the matrix size is not a power of 2.
        """
        if mat.shape[0] != mat.shape[1]:
            msg = "Matrix must be square"
            raise ValueError(msg)

        log = np.log2(mat.shape[0])

        if log.is_integer() is False:
            msg = "Matrix must have a size that is a power of 2"
            raise ValueError(msg)

        self.matrix = mat
        self.tensor = mat
        self.interaction = int(log)

    def set_sites(self, *sites: int) -> None:
        """Sets the sites for the gate.

        Args:
            *sites (int): Variable-length argument list specifying site indices.

        Raises:
            ValueError: If the number of sites does not match the interaction level of the gate.
        """
        sites_list = list(sites)

        if len(sites_list) != self.interaction:
            msg = f"Number of sites {len(sites_list)} must be equal to the interaction level {self.interaction}"
            raise ValueError(msg)
        self.sites: list[int] = sites_list

    def __add__(self, other: BaseGate) -> BaseGate:
        """Adds two gates together.

        Args:
            other (BaseGate): The gate to be added.

        Raises:
            ValueError: If the gates have different interaction levels.

        Returns:
            BaseGate: A new gate representing the sum of the two gates.
        """
        if self.interaction != other.interaction:
            msg = "Cannot add gates with different interaction"
            raise ValueError(msg)
        return BaseGate(self.matrix + other.matrix)

    def __sub__(self, other: BaseGate) -> BaseGate:
        """Subtracts one gate from another.

        Args:
            other (BaseGate): The gate to be subtracted.

        Raises:
            ValueError: If the gates have different interaction levels.

        Returns:
            BaseGate: A new gate representing the difference between the two gates.
        """
        if self.interaction != other.interaction:
            msg = "Cannot subtract gates with different interaction"
            raise ValueError(msg)
        return BaseGate(self.matrix - other.matrix)

    def __mul__(self, other: BaseGate | complex) -> BaseGate:
        """Multiplies two gates or scales a gate by a scalar.

        Args:
            other (BaseGate | complex): The gate or scalar to multiply.

        Raises:
            ValueError: If the gates have different interaction levels (when multiplying two gates).

        Returns:
            BaseGate: A new gate representing the product of the two gates or the scaled gate.
        """
        if isinstance(other, BaseGate):
            if self.interaction != other.interaction:
                msg = "Cannot multiply gates with different interaction"
                raise ValueError(msg)
            return BaseGate(self.matrix @ other.matrix)

        return BaseGate(self.matrix * other)

    def __rmul__(self, other: BaseGate | complex) -> BaseGate:
        """Multiplies a scalar or another gate with this gate (right multiplication).

        Args:
            other (BaseGate | complex): The gate or scalar to multiply.

        Returns:
            BaseGate: A new gate representing the product.
        """
        return self.__mul__(other)

    def dag(self) -> BaseGate:
        """Returns the conjugate transpose (dagger) of the gate.

        Returns:
            BaseGate: A new gate representing the conjugate transpose of this gate.
        """
        return BaseGate(np.conj(self.matrix).T)

    def conj(self) -> BaseGate:
        """Returns the complex conjugate of the gate.

        Returns:
            BaseGate: A new gate representing the complex conjugate of this gate.
        """
        return BaseGate(np.conj(self.matrix))

    def trans(self) -> BaseGate:
        """Returns the transpose of the gate.

        Returns:
            BaseGate: A new gate representing the transpose of this gate.
        """
        return BaseGate(self.matrix.T)

    @classmethod
    def x(cls) -> X:
        """Returns the X gate.

        Returns:
            X: An instance of the X gate.
        """
        return X()

    @classmethod
    def y(cls) -> Y:
        """Returns the Y gate.

        Returns:
            Y: An instance of the Y gate.
        """
        return Y()

    @classmethod
    def z(cls) -> Z:
        """Returns the Z gate.

        Returns:
            Z: An instance of the Z gate.
        """
        return Z()

    @classmethod
    def h(cls) -> H:
        """Returns the H gate.

        Returns:
            H: An instance of the H gate.
        """
        return H()

    @classmethod
    def destroy(cls) -> Destroy:
        """Returns the Destroy gate.

        Returns:
            Destroy: An instance of the Destroy gate.
        """
        return Destroy()

    @classmethod
    def create(cls) -> Create:
        """Returns the Create gate.

        Returns:
            Create: An instance of the Create gate.
        """
        return Create()

    @classmethod
    def id(cls) -> Id:
        """Returns the Id gate.

        Returns:
            Id: An instance of the Id gate.
        """
        return Id()

    @classmethod
    def sx(cls) -> SX:
        """Returns the SX gate.

        Returns:
            SX: An instance of the SX gate.
        """
        return SX()

    @classmethod
    def rx(cls, params: list[Parameter]) -> Rx:
        """Returns the RX gate.

        Args:
            params (list[Parameter]): The rotation angle parameter.

        Returns:
            Rx: An instance of the RX gate.
        """
        return Rx(params)

    @classmethod
    def ry(cls, params: list[Parameter]) -> Ry:
        """Returns the RY gate.

        Args:
            params (list[Parameter]): The rotation angle parameter.

        Returns:
            Ry: An instance of the RY gate.
        """
        return Ry(params)

    @classmethod
    def rz(cls, params: list[Parameter]) -> Rz:
        """Returns the RZ gate.

        Args:
            params (list[Parameter]): The rotation angle parameter.

        Returns:
            Rz: An instance of the RZ gate.
        """
        return Rz(params)

    @classmethod
    def phase(cls, params: list[Parameter]) -> Phase:
        """Returns the Phase gate.

        Args:
            params (list[Parameter]): The rotation angle parameter.

        Returns:
            Phase: An instance of the Phase gate.
        """
        return Phase(params)

    @classmethod
    def u3(cls, params: list[Parameter]) -> U3:
        """Returns the U3 gate.

        Args:
            params (list[Parameter]): The rotation angle parameters.

        Returns:
            U3: An instance of the U3 gate.
        """
        return U3(params)

    @classmethod
    def cx(cls) -> CX:
        """Returns the CX gate.

        Returns:
            CX: An instance of the CX gate.
        """
        return CX()

    @classmethod
    def cz(cls) -> CZ:
        """Returns the CZ gate.

        Returns:
            CZ: An instance of the CZ gate.
        """
        return CZ()

    @classmethod
    def cphase(cls, params: list[Parameter]) -> CPhase:
        """Returns the CPhase gate.

        Args:
            params (list[Parameter]): The rotation angle parameter.

        Returns:
            CPhase: An instance of the CPhase gate.
        """
        return CPhase(params)

    @classmethod
    def swap(cls) -> SWAP:
        """Returns the SWAP gate.

        Returns:
            SWAP: An instance of the SWAP gate.
        """
        return SWAP()

    @classmethod
    def rxx(cls, params: list[Parameter]) -> Rxx:
        """Returns the RXX gate.

        Args:
            params (list[Parameter]): The rotation angle parameter.

        Returns:
            Rxx: An instance of the RXX gate.
        """
        return Rxx(params)

    @classmethod
    def ryy(cls, params: list[Parameter]) -> Ryy:
        """Returns the RYY gate.

        Args:
            params (list[Parameter]): The rotation angle parameter.

        Returns:
            Ryy: An instance of the RYY gate.
        """
        return Ryy(params)

    @classmethod
    def rzz(cls, params: list[Parameter]) -> Rzz:
        """Returns the RZZ gate.

        Args:
            params (list[Parameter]): The rotation angle parameter.

        Returns:
            Rzz: An instance of the RZZ gate.
        """
        return Rzz(params)


class X(BaseGate):
    """Class representing the Pauli-X (NOT) gate.

    Attributes:
        name (str): The name of the gate ("x").
        matrix (NDArray[np.complex128]): The 2x2 matrix representation of the gate.
        interaction (int): The interaction level (1 for single-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation of the gate (same as the matrix).

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
        name (str): The name of the gate ("y").
        matrix (NDArray[np.complex128]): The 2x2 matrix representation of the gate.
        interaction (int): The interaction level (1 for single-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation of the gate (same as the matrix).

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
        name (str): The name of the gate ("z").
        matrix (NDArray[np.complex128]): The 2x2 matrix representation of the gate.
        interaction (int): The interaction level (1 for single-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation of the gate (same as the matrix).

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
        name (str): The name of the gate ("h").
        matrix (NDArray[np.complex128]): The 2x2 matrix representation of the gate.
        interaction (int): The interaction level (1 for single-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation of the gate (same as the matrix).

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
        name (str): The name of the gate ("destroy").
        matrix (NDArray[np.complex128]): The 2x2 matrix representation of the gate.
        interaction (int): The interaction level (1 for single-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation of the gate (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "destroy"

    def __init__(self) -> None:
        """Initializes the Destroy gate."""
        mat = np.array([[0, 1], [0, 0]])
        super().__init__(mat)


class Create(BaseGate):
    """Class representing the Create (creation) gate.

    Attributes:
        name (str): The name of the gate ("create").
        matrix (NDArray[np.complex128]): The 2x2 matrix representation of the gate.
        interaction (int): The interaction level (1 for single-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation of the gate (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "create"

    def __init__(self) -> None:
        """Initializes the Create gate."""
        mat = np.array([[0, 0], [1, 0]])
        super().__init__(mat)


class Id(BaseGate):
    """Class representing the identity (Id) gate.

    Attributes:
        name (str): The name of the gate ("id").
        matrix (NDArray[np.complex128]): The 2x2 identity matrix.
        interaction (int): The interaction level (1 for single-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation of the gate (same as the matrix).

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
        name (str): The name of the gate ("sx").
        matrix (NDArray[np.complex128]): The 2x2 matrix representation of the gate.
        interaction (int): The interaction level (1 for single-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation of the gate (same as the matrix).

    Methods:
        set_sites(*sites: int) -> None:
            Sets the site(s) where the gate is applied.
    """

    name = "sx"

    def __init__(self) -> None:
        """Initializes the square-root X gate."""
        mat = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
        super().__init__(mat)


class Rx(BaseGate):
    """Class representing a rotation gate about the x-axis.

    Attributes:
        name (str): The name of the gate ("rx").
        matrix (NDArray[np.complex128]): The 2x2 matrix representation of the gate.
        interaction (int): The interaction level (1 for single-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation of the gate (same as the matrix).
        theta (Parameter): The rotation angle parameter.

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
        name (str): The name of the gate ("ry").
        matrix (NDArray[np.complex128]): The 2x2 matrix representation of the gate.
        interaction (int): The interaction level (1 for single-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation of the gate (same as the matrix).
        theta (Parameter): The rotation angle parameter.

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
        name (str): The name of the gate ("rz").
        matrix (NDArray[np.complex128]): The 2x2 matrix representation of the gate.
        interaction (int): The interaction level (1 for single-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation of the gate (same as the matrix).
        theta (Parameter): The rotation angle parameter.

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
        name (str): The name of the gate ("p").
        matrix (NDArray[np.complex128]): The 2x2 matrix representation of the gate.
        interaction (int): The interaction level (1 for single-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation of the gate (same as the matrix).
        theta (Parameter): The phase angle parameter.

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


class U3(BaseGate):
    """Class representing a U3 gate.

    Attributes:
        name (str): The name of the gate ("u").
        matrix (NDArray[np.complex128]): The 2x2 matrix representation of the gate.
        interaction (int): The interaction level (1 for single-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation of the gate (same as the matrix).
        theta (float): The first rotation parameter.
        phi (float): The second rotation parameter.
        lam (float): The third rotation parameter.

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
        name (str): The name of the gate ("cx").
        matrix (NDArray[np.complex128]): The 4x4 matrix representation of the gate.
        interaction (int): The interaction level (2 for two-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation reshaped to (2, 2, 2, 2).
        generator (list[NDArray[np.complex128]]): The generator for the gate.
        mpo: An MPO representation generated from the gate tensor.
        sites (list[int]): The control and target sites.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites and updates the tensor and MPO.
    """

    name = "cx"

    def __init__(self) -> None:
        """Initializes the controlled-NOT (CX) gate."""
        mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        super().__init__(mat)

    def set_sites(self, *sites: int) -> None:
        """Sets the sites for the gate.

        Args:
            *sites (int): Variable-length argument list specifying site indices.

        Raises:
            ValueError: If the number of sites does not match the interaction level of the gate.
        """
        sites_list = list(sites)

        if len(sites_list) != self.interaction:
            msg = f"Number of sites {len(sites_list)} must be equal to the interaction level {self.interaction}"
            raise ValueError(msg)

        self.sites: list[int] = sites_list
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        # Generator: π/4 (I-Z ⊗ I-X)
        self.generator = [(np.pi / 4) * np.array([[0, 0], [0, 2]]), np.array([[1, -1], [-1, 1]])]
        self.mpo = extend_gate(self.tensor, self.sites)
        if sites[1] < sites[0]:  # Adjust for reverse control/target
            self.tensor = np.transpose(self.tensor, (1, 0, 3, 2))


class CZ(BaseGate):
    """Class representing the controlled-Z (CZ) gate.

    Attributes:
        name (str): The name of the gate ("cz").
        matrix (NDArray[np.complex128]): The 4x4 matrix representation of the gate.
        interaction (int): The interaction level (2 for two-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation reshaped to (2, 2, 2, 2).
        generator (list[NDArray[np.complex128]]): The generator for the gate.
        sites (list[int]): The control and target sites.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites and updates the tensor.
    """

    name = "cz"

    def __init__(self) -> None:
        """Initializes the controlled-Z (CZ) gate."""
        mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
        super().__init__(mat)

    def set_sites(self, *sites: int) -> None:
        """Sets the sites for the gate.

        Args:
            *sites (int): Variable-length argument list specifying site indices.

        Raises:
            ValueError: If the number of sites does not match the interaction level of the gate.
        """
        sites_list = list(sites)

        if len(sites_list) != self.interaction:
            msg = f"Number of sites {len(sites_list)} must be equal to the interaction level {self.interaction}"
            raise ValueError(msg)

        self.sites: list[int] = sites_list
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        # Generator: π/4 (I-Z ⊗ I-Z)
        self.generator = [(np.pi / 4) * np.array([[0, 0], [0, 2]]), np.array([[1, -1], [-1, 1]])]
        if sites[1] < sites[0]:  # Adjust for reverse control/target
            self.tensor = np.transpose(self.tensor, (1, 0, 3, 2))


class CPhase(BaseGate):
    """Class representing the controlled phase (CPhase) gate.

    Attributes:
        name (str): The name of the gate ("cp").
        matrix (NDArray[np.complex128]): The 4x4 matrix representation of the gate.
        interaction (int): The interaction level (2 for two-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation reshaped to (2, 2, 2, 2).
        generator (list[NDArray[np.complex128]]): The generator for the gate.
        sites (list[int]): The control and target sites.
        theta (Parameter): The angle parameter.

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
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        self.generator = [(self.theta / 2) * np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, 0]])]


class SWAP(BaseGate):
    """Class representing the SWAP gate.

    Attributes:
        name (str): The name of the gate ("swap").
        matrix (NDArray[np.complex128]): The 4x4 matrix representation of the gate.
        interaction (int): The interaction level (2 for two-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation reshaped to (2, 2, 2, 2).
        sites (list[int]): The sites involved in the swap.

    Methods:
        set_sites(*sites: int) -> None:
            Sets the sites and updates the tensor.
    """

    name = "swap"

    def __init__(self) -> None:
        """Initializes the SWAP gate."""
        mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        super().__init__(mat)
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))


class Rxx(BaseGate):
    """Class representing a two-qubit rotation gate about the xx-axis.

    Attributes:
        name (str): The name of the gate ("rxx").
        matrix (NDArray[np.complex128]): The 4x4 matrix representation of the gate.
        interaction (int): The interaction level (2 for two-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation reshaped to (2, 2, 2, 2).
        generator (list[NDArray[np.complex128]]): The generator for the gate.
        sites (list[int]): The control and target sites.
        theta (Parameter): The angle parameter.

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
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        self.generator = [(self.theta / 2) * np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])]


class Ryy(BaseGate):
    """Class representing a two-qubit rotation gate about the yy-axis.

    Attributes:
        name (str): The name of the gate ("ryy").
        matrix (NDArray[np.complex128]): The 4x4 matrix representation of the gate.
        interaction (int): The interaction level (2 for two-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation reshaped to (2, 2, 2, 2).
        generator (list[NDArray[np.complex128]]): The generator for the gate.
        sites (list[int]): The control and target sites.
        theta (Parameter): The angle parameter.

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
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        self.generator = [(self.theta / 2) * np.array([[0, -1j], [1j, 0]]), np.array([[0, -1j], [1j, 0]])]


class Rzz(BaseGate):
    """Class representing a two-qubit rotation gate about the zz-axis.

    Attributes:
        name (str): The name of the gate ("rzz").
        matrix (NDArray[np.complex128]): The 4x4 matrix representation of the gate.
        interaction (int): The interaction level (2 for two-qubit gates).
        tensor (NDArray[np.complex128]): The tensor representation reshaped to (2, 2, 2, 2).
        generator (list[NDArray[np.complex128]]): The generator for the gate.
        sites (list[int]): The control and target sites.
        theta (Parameter): The angle parameter.

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
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        self.generator = [(self.theta / 2) * np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, -1]])]


class GateLibrary:
    """A collection of quantum gate classes for use in simulations.

    Attributes:
        x: Class for the X gate.
        y: Class for the Y gate.
        z: Class for the Z gate.
        sx: Class for the square-root X gate.
        h: Class for the Hadamard gate.
        id: Class for the identity gate.
        rx: Class for the rotation gate about the x-axis.
        ry: Class for the rotation gate about the y-axis.
        rz: Class for the rotation gate about the z-axis.
        u: Class for the U3 gate.
        cx: Class for the controlled-NOT gate.
        cz: Class for the controlled-Z gate.
        swap: Class for the SWAP gate.
        rxx: Class for the rotation gate about the xx-axis.
        ryy: Class for the rotation gate about the yy-axis.
        rzz: Class for the rotation gate about the zz-axis.
        cp: Class for the controlled phase gate.
        p: Class for the phase gate.
    """

    x = X
    y = Y
    z = Z
    sx = SX
    h = H
    id = Id
    rx = Rx
    ry = Ry
    rz = Rz
    u = U3
    cx = CX
    cz = CZ
    swap = SWAP
    rxx = Rxx
    ryy = Ryy
    rzz = Rzz
    cp = CPhase
    p = Phase
