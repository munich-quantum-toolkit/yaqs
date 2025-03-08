# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module implements classes for representing quantum states and operators using tensor networks.
It defines the Matrix Product State (MPS) and Matrix Product Operator (MPO) classes, along with various
methods for network normalization, canonicalization, measurement, and validity checks. These classes and
utilities are essential for simulating quantum many-body systems using tensor network techniques.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe

from ..libraries.gate_library import GateLibrary
from ..methods.operations import local_expval, scalar_product

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .simulation_parameters import Observable


# Convention (sigma, chi_l-1, chi_l)
class MPS:
    def __init__(
        self,
        length: int,
        tensors: list[NDArray[np.complex128]] | None = None,
        physical_dimensions: list[int] | None = None,
        state: str = "zeros",
    ) -> None:
        """Matrix Product State (MPS) class for representing quantum states.

        Attributes:
        length (int): The number of sites in the MPS.
        tensors (list[NDArray[np.complex128]]): List of rank-3 tensors representing the MPS.
        physical_dimensions (list[int]): List of physical dimensions for each site.
        flipped (bool): Indicates if the network has been flipped.

        Methods:
        __init__(length: int, tensors: list[NDArray[np.complex128]] | None = None, physical_dimensions: list[int] | None = None, state: str = "zeros") -> None:
            Initializes the MPS with given length, tensors, physical dimensions, and initial state.
        write_max_bond_dim() -> int:
            Returns the maximum bond dimension in the MPS.
        flip_network() -> None:
            Flips the bond dimensions in the network to allow operations from right to left.
        shift_orthogonality_center_right(current_orthogonality_center: int) -> None:
            Left and right normalizes the MPS around a selected site, shifting the orthogonality center to the right.
        shift_orthogonality_center_left(current_orthogonality_center: int) -> None:
            Left and right normalizes the MPS around a selected site, shifting the orthogonality center to the left.
        set_canonical_form(orthogonality_center: int) -> None:
            Left and right normalizes the MPS around a selected site to set it in canonical form.
        normalize(form: str = "B") -> None:
            Normalizes the MPS in the specified form.
        measure(observable: Observable) -> np.float64:
            Measures the expectation value of an observable at a specified site.
        norm(site: int | None = None) -> np.float64:
            Computes the norm of the MPS, optionally at a specified site.
        write_tensor_shapes() -> None:
            Writes the shapes of the tensors in the MPS.
        check_if_valid_MPS() -> None:
            Checks if the MPS is valid by verifying bond dimensions.
        check_canonical_form() -> list[int]:
            Checks the canonical form of the MPS and returns the orthogonality center(s).
        """
        self.flipped = False
        if tensors is not None:
            assert len(tensors) == length
            self.tensors = tensors
        else:
            self.tensors = []
        self.length = length
        if physical_dimensions is None:
            # Default case is the qubit (2-level) case
            self.physical_dimensions = []
            for _ in range(self.length):
                self.physical_dimensions.append(2)
        else:
            self.physical_dimensions = physical_dimensions
        assert len(self.physical_dimensions) == length

        # Create d-level |0> state
        if not tensors:
            for i, d in enumerate(self.physical_dimensions):
                vector = np.zeros(d, dtype=complex)
                if state == "zeros":
                    # |0>
                    vector[0] = 1
                elif state == "ones":
                    # |1>
                    vector[1] = 1
                elif state == "x+":
                    # |+> = (|0> + |1>)/sqrt(2)
                    vector[0] = 1 / np.sqrt(2)
                    vector[1] = 1 / np.sqrt(2)
                elif state == "x-":
                    # |-> = (|0> - |1>)/sqrt(2)
                    vector[0] = 1 / np.sqrt(2)
                    vector[1] = -1 / np.sqrt(2)
                elif state == "y+":
                    # |+i> = (|0> + i|1>)/sqrt(2)
                    vector[0] = 1 / np.sqrt(2)
                    vector[1] = 1j / np.sqrt(2)
                elif state == "y-":
                    # |-i> = (|0> - i|1>)/sqrt(2)
                    vector[0] = 1 / np.sqrt(2)
                    vector[1] = -1j / np.sqrt(2)
                elif state == "Neel":
                    # |010101...>
                    if i % 2:
                        vector[0] = 1
                    else:
                        vector[1] = 1
                elif state == "wall":
                    # |000111>
                    if i < length // 2:
                        vector[0] = 1
                    else:
                        vector[1] = 1
                elif state == "random":
                    vector[0] = np.random.rand()
                    vector[1] = 1 - vector[0]
                else:
                    msg = "Invalid state string"
                    raise ValueError(msg)

                tensor = np.expand_dims(vector, axis=(0, 1))

                tensor = np.transpose(tensor, (2, 0, 1))
                self.tensors.append(tensor)

            if state == "random":
                self.normalize()

    def write_max_bond_dim(self) -> int:
        """Calculate and return the maximum bond dimension of the tensors in the network.
        This method iterates over all tensors in the network and determines the maximum
        bond dimension by comparing the first and third dimensions of each tensor's shape.
        The global maximum bond dimension is then returned.

        Returns:
            int: The maximum bond dimension found among all tensors in the network.
        """
        global_max = 0
        for tensor in self.tensors:
            local_max = max(tensor.shape[0], tensor.shape[2])
            global_max = max(global_max, local_max)

        return global_max

    def flip_network(self) -> None:
        """Flips the bond dimensions in the network so that we can do operations
            from right to left.

        Args:
            MPS: list of rank-3 tensors
        Returns:
            new_MPS: list of rank-3 tensors with bond dimensions reversed
                    and sites reversed compared to input MPS
        """
        new_tensors = []
        for tensor in self.tensors:
            new_tensor = np.transpose(tensor, (0, 2, 1))
            new_tensors.append(new_tensor)

        new_tensors.reverse()
        self.tensors = new_tensors
        self.flipped = not self.flipped
        # self.orthogonality_center = self.length+1-self.orthogonality_center

    def shift_orthogonality_center_right(self, current_orthogonality_center: int) -> None:
        """Left and right normalizes an MPS around a selected site.

        Args:
            MPS: list of rank-3 tensors with a physical dimension d^2
            selected_site: site of matrix M around which we normalize
        Returns:
            new_MPS: list of rank-3 tensors at each site
        """
        tensor = self.tensors[current_orthogonality_center]
        old_dims = tensor.shape
        matricized_tensor = np.reshape(tensor, (tensor.shape[0] * tensor.shape[1], tensor.shape[2]))
        Q, R = np.linalg.qr(matricized_tensor)
        Q = np.reshape(Q, (old_dims[0], old_dims[1], -1))
        self.tensors[current_orthogonality_center] = Q

        # If normalizing, we just throw away the R
        if current_orthogonality_center + 1 < self.length:
            self.tensors[current_orthogonality_center + 1] = oe.contract(
                "ij, ajc->aic", R, self.tensors[current_orthogonality_center + 1]
            )

    def shift_orthogonality_center_left(self, current_orthogonality_center: int) -> None:
        """Left and right normalizes an MPS around a selected site.

        Args:
            MPS: list of rank-3 tensors with a physical dimension d^2
            selected_site: site of matrix M around which we normalize
        Returns:
            new_MPS: list of rank-3 tensors at each site
        """
        self.flip_network()
        self.shift_orthogonality_center_right(self.length - current_orthogonality_center - 1)
        self.flip_network()

    # TODO(Aaron): Needs to be adjusted based on current orthogonality center
    #       Rather than sweeping the full chain
    def set_canonical_form(self, orthogonality_center: int) -> None:
        """Left and right normalizes an MPS around a selected site
        Args:
            MPS: list of rank-3 tensors with a physical dimension d^2
            selected_site: site of matrix M around which we normalize
        Returns:
            new_MPS: list of rank-3 tensors at each site.
        """

        def sweep_decomposition(orthogonality_center: int) -> None:
            for site, _ in enumerate(self.tensors):
                if site == orthogonality_center:
                    break
                self.shift_orthogonality_center_right(site)

        sweep_decomposition(orthogonality_center)
        self.flip_network()
        flipped_orthogonality_center = self.length - 1 - orthogonality_center
        sweep_decomposition(flipped_orthogonality_center)
        self.flip_network()

    def normalize(self, form: str = "B") -> None:
        """Normalize the network to a specified form.
        This method normalizes the network to the specified form. By default, it normalizes to form "B" (right canonical).
        The normalization process involves flipping the network, setting the canonical form with the
        orthogonality center at the last position, and shifting the orthogonality center to the rightmost position.

        Args:
            form (str): The form to normalize the network to. Default is "B".

        Returns:
            None.
        """
        if form == "B":
            self.flip_network()

        self.set_canonical_form(orthogonality_center=self.length - 1)
        self.shift_orthogonality_center_right(self.length - 1)

        if form == "B":
            self.flip_network()

    def measure(self, observable: Observable) -> np.float64:
        """Measure the expectation value of a given observable.

        Parameters:
        observable (Observable): The observable to measure. It must have a 'site' attribute indicating the site to measure and a 'name' attribute corresponding to a gate in the GateLibrary.

        Returns:
        np.float64: The real part of the expectation value of the observable.

        Raises:
        AssertionError: If the observable's site is out of range or if the imaginary part of the expectation value is not negligible.
        """
        assert observable.site in range(self.length), "State is shorter than selected site for expectation value."
        # Copying done to stop the state from messing up its own canonical form
        E = local_expval(copy.deepcopy(self), getattr(GateLibrary, observable.name)().matrix, observable.site)
        assert E.imag < 1e-13, f"Measurement should be real, '{E.real:16f}+{E.imag:16f}i'."
        return E.real

    def norm(self, site: int | None = None) -> np.float64:
        """Calculate the norm of the state.

        Parameters:
        site (int | None): The specific site to calculate the norm from. If None, the norm is calculated for the entire network.

        Returns:
        np.float64: The norm of the state or the specified site.
        """
        if site is not None:
            return scalar_product(self, self, site).real
        return scalar_product(self, self).real

    def check_if_valid_MPS(self) -> None:
        """Check if the current tensor network is a valid Matrix Product State (MPS).

        This method verifies that the bond dimensions between consecutive tensors
        in the network are consistent. Specifically, it checks that the second
        dimension of each tensor matches the third dimension of the previous tensor.

        Raises:
            AssertionError: If the bond dimensions between consecutive tensors do not match.
        """
        right_bond = self.tensors[0].shape[2]
        for tensor in self.tensors[1::]:
            assert tensor.shape[1] == right_bond
            right_bond = tensor.shape[2]

    def check_canonical_form(self) -> list[int]:
        """ "
        Checks what canonical form a Matrix Product State (MPS) is in, if any.
        This method verifies if the MPS is in left-canonical form, right-canonical form, or mixed-canonical form.
        It returns a list indicating the canonical form status:
        - [0] if the MPS is in left-canonical form.
        - [self.length - 1] if the MPS is in right-canonical form.
        - [index] if the MPS is in mixed-canonical form, where `index` is the position where the form changes.
        - [-1] if the MPS is not in any canonical form.

        Returns:
            list[int]: A list indicating the canonical form status of the MPS.
        """
        A = copy.deepcopy(self.tensors)
        for i, tensor in enumerate(self.tensors):
            A[i] = np.conj(tensor)
        B = self.tensors

        A_truth = []
        B_truth = []
        epsilon = 1e-12
        for i in range(len(A)):
            M = oe.contract("ijk, ijl->kl", A[i], B[i])
            M[epsilon > M] = 0
            test_identity = np.eye(M.shape[0], dtype=complex)
            A_truth.append(np.allclose(M, test_identity))

        for i in range(len(A)):
            M = oe.contract("ijk, ibk->jb", B[i], A[i])
            M[epsilon > M] = 0
            test_identity = np.eye(M.shape[0], dtype=complex)
            B_truth.append(np.allclose(M, test_identity))

        if all(B_truth):
            return [0]
        if all(A_truth):
            return [self.length - 1]

        if not (all(A_truth) and all(B_truth)):
            sites = []
            for i, truth_value in enumerate(A_truth):
                if truth_value:
                    sites.append(i)
                else:
                    break

            for i, truth_value in enumerate(B_truth[len(sites) :], start=len(sites)):
                sites.append(i)
            try:
                return [sites.index(False)]
            except:
                for i, value in enumerate(A_truth):
                    if not value:
                        return [i - 1, i]
        return [-1]


# Convention (sigma, sigma', chi_l,  chi_l+1)
class MPO:
    """Class representing a Matrix Product Operator (MPO) for quantum many-body systems.

    Methods.
    -------
    init_Ising(length: int, J: float, g: float) -> None
        Initializes the MPO for the Ising model with given parameters.
    init_Heisenberg(length: int, Jx: float, Jy: float, Jz: float, h: float) -> None
        Initializes the MPO for the Heisenberg model with given parameters.
    init_identity(length: int, physical_dimension: int = 2) -> None
        Initializes the MPO as an identity operator.
    init_custom_Hamiltonian(length: int, left_bound: NDArray[np.complex128], inner: NDArray[np.complex128], right_bound: NDArray[np.complex128]) -> None
        Initializes the MPO with custom Hamiltonian tensors.
    init_custom(tensors: list[NDArray[np.complex128]], transpose: bool = True) -> None
        Initializes the MPO with custom tensors.
    convert_to_MPS() -> MPS
        Converts the MPO to a Matrix Product State (MPS).
    convert_to_matrix() -> NDArray[np.complex128]
        Converts the MPO to a full matrix representation.
    write_tensor_shapes() -> None
        Writes the shapes of the tensors in the MPO.
    check_if_valid_MPO() -> bool
        Checks if the MPO is valid.
    check_if_identity(fidelity: float) -> bool
        Checks if the MPO represents an identity operator with given fidelity.
    rotate(conjugate: bool = False) -> None
        Rotates the MPO tensors by swapping physical dimensions.
    """

    def init_Ising(self, length: int, J: float, g: float) -> None:
        """Initialize the Ising model as a Matrix Product Operator (MPO).
        This method constructs the MPO representation of the Ising model with
        specified parameters. The MPO has a 3x3 block structure at each site.

        Args:
            length (int): The number of sites in the Ising chain.
            J (float): The coupling constant for the Z interaction.
            g (float): The coupling constant for the X interaction.

        Returns:
            None.
        """
        physical_dimension = 2
        np.zeros((physical_dimension, physical_dimension), dtype=complex)
        identity = np.eye(physical_dimension, dtype=complex)
        X = GateLibrary.x().matrix
        Z = GateLibrary.z().matrix

        # The MPO has a 3x3 block structure at each site:
        # W = [[ I,     -J Z,  -g X ],
        #      [ 0,       0,     Z  ],
        #      [ 0,       0,     I  ]]

        # Left boundary (1x3 block) selects the top row of W:
        # [I, -J Z, -g X]
        left_bound = np.array([identity, -J * Z, -g * X])[np.newaxis, :]

        # Inner tensors (3x3 block):
        inner = np.zeros((3, 3, physical_dimension, physical_dimension), dtype=complex)
        inner[0, 0] = identity
        inner[0, 1] = -J * Z
        inner[0, 2] = -g * X
        inner[1, 2] = Z
        inner[2, 2] = identity

        # Right boundary (3x1 block) selects the last column:
        # [ -g X, Z, I ]^T but we only take the operators that appear there.
        # Actually, at the right boundary we just pick out the last column:
        # (top row: -g X, second row: Z, third row: I)
        right_bound = np.array([[-g * X], [Z], [identity]])

        # Construct the MPO as a list of tensors:
        # Left boundary, (length-2)*inner, right boundary
        self.tensors = [left_bound] + [inner] * (length - 2) + [right_bound]
        for i, tensor in enumerate(self.tensors):
            # left, right, sigma, sigma'
            self.tensors[i] = np.transpose(tensor, (2, 3, 0, 1))

        self.length = length
        self.physical_dimension = physical_dimension

    def init_Heisenberg(self, length: int, Jx: float, Jy: float, Jz: float, h: float) -> None:
        """Initialize the Heisenberg model as a Matrix Product Operator (MPO).
        Parameters:
        length (int): The number of sites in the chain.
        Jx (float): The coupling constant for the X interaction.
        Jy (float): The coupling constant for the Y interaction.
        Jz (float): The coupling constant for the Z interaction.
        h (float): The magnetic field strength.

        Returns:
        None.
        """
        physical_dimension = 2
        zero = np.zeros((physical_dimension, physical_dimension), dtype=complex)
        identity = np.eye(physical_dimension, dtype=complex)
        X = GateLibrary.x().matrix
        Y = GateLibrary.y().matrix
        Z = GateLibrary.z().matrix

        # Left boundary: shape (1,5, d, d)
        # [I, Jx*X, Jy*Y, Jz*Z, h*Z]
        left_bound = np.array([identity, -Jx * X, -Jy * Y, -Jz * Z, -h * Z])[np.newaxis, :]

        # Inner tensor: shape (5,5, d, d)
        # W = [[ I,    Jx*X,  Jy*Y,  Jz*Z,   h*Z ],
        #      [ 0,     0,     0,     0,     X  ],
        #      [ 0,     0,     0,     0,     Y  ],
        #      [ 0,     0,     0,     0,     Z  ],
        #      [ 0,     0,     0,     0,     I  ]]

        inner = np.zeros((5, 5, physical_dimension, physical_dimension), dtype=complex)
        inner[0, 0] = identity
        inner[0, 1] = -Jx * X
        inner[0, 2] = -Jy * Y
        inner[0, 3] = -Jz * Z
        inner[0, 4] = -h * Z
        inner[1, 4] = X
        inner[2, 4] = Y
        inner[3, 4] = Z
        inner[4, 4] = identity

        # Right boundary: shape (5,1, d, d)
        # [0, X, Y, Z, I]^T
        right_bound = np.array([zero, X, Y, Z, identity])[:, np.newaxis]

        # Construct the MPO
        self.tensors = [left_bound] + [inner] * (length - 2) + [right_bound]
        for i, tensor in enumerate(self.tensors):
            # left, right, sigma, sigma'
            self.tensors[i] = np.transpose(tensor, (2, 3, 0, 1))
        self.length = length
        self.physical_dimension = physical_dimension

    def init_identity(self, length: int, physical_dimension: int = 2) -> None:
        """Initializes the network with identity matrices.
        Parameters:
        length (int): The number of identity matrices to initialize.
        physical_dimension (int, optional): The physical dimension of the identity matrices. Default is 2.

        Returns:
        None.
        """
        M = np.eye(2)
        M = np.expand_dims(M, (2, 3))
        self.length = length
        self.physical_dimension = physical_dimension

        self.tensors = []
        for _ in range(length):
            self.tensors.append(M)

    def init_custom_Hamiltonian(
        self,
        length: int,
        left_bound: NDArray[np.complex128],
        inner: NDArray[np.complex128],
        right_bound: NDArray[np.complex128],
    ) -> None:
        """Initialize a custom Hamiltonian as a Matrix Product Operator (MPO).
        This method sets up the Hamiltonian using the provided boundary and inner tensors.
        The tensors are transposed to match the expected shape for MPOs.

        Args:
            length (int): The number of tensors in the MPO.
            left_bound (NDArray[np.complex128]): The tensor at the left boundary.
            inner (NDArray[np.complex128]): The tensor for the inner sites.
            right_bound (NDArray[np.complex128]): The tensor at the right boundary.

        Raises:
            AssertionError: If the MPO is initialized incorrectly.

        Returns:
            None.
        """
        self.tensors = [left_bound] + [inner] * (length - 2) + [right_bound]
        for i, tensor in enumerate(self.tensors):
            # left, right, sigma, sigma'
            self.tensors[i] = np.transpose(tensor, (2, 3, 0, 1))
        assert self.check_if_valid_MPO(), "MPO initialized wrong"
        self.length = len(self.tensors)
        self.physical_dimension = self.tensors[0].shape[0]

    def init_custom(self, tensors: list[NDArray[np.complex128]], transpose: bool = True) -> None:
        """Initialize the custom MPO (Matrix Product Operator) with the given tensors.
        Parameters.
        ----------
        tensors : list[NDArray[np.complex128]]
            A list of tensors to initialize the MPO.
        transpose : bool, optional
            If True, transpose each tensor to the order (2, 3, 0, 1). Default is True.

        Raises:
        ------
        AssertionError
            If the MPO is initialized incorrectly.

        Notes:
        -----
        This method sets the tensors, optionally transposes them, checks if the MPO is valid,
        and initializes the length and physical dimension of the MPO.
        """
        self.tensors = tensors
        if transpose:
            for i, tensor in enumerate(self.tensors):
                # left, right, sigma, sigma'
                self.tensors[i] = np.transpose(tensor, (2, 3, 0, 1))
        assert self.check_if_valid_MPO(), "MPO initialized wrong"
        self.length = len(self.tensors)
        self.physical_dimension = tensors[0].shape[0]

    def convert_to_MPS(self) -> MPS:
        """Converts the current tensor network to a Matrix Product State (MPS) representation.
        This method reshapes each tensor in the network from shape
        (dim1, dim2, dim3, dim4) to (dim1 * dim2, dim3, dim4) and
        returns a new MPS object with the converted tensors.

        Returns:
            MPS: An MPS object containing the reshaped tensors.
        """
        converted_tensors = [
            np.reshape(tensor, (tensor.shape[0] * tensor.shape[1], tensor.shape[2], tensor.shape[3]))
            for tensor in self.tensors
        ]

        return MPS(self.length, converted_tensors)

    def convert_to_matrix(self) -> NDArray[np.complex128]:
        """Converts a list of tensors into a matrix using Einstein summation convention.
        This method iterates over the list of tensors and performs tensor contractions
        using the Einstein summation convention (`oe.constrain`). The resulting tensor is
        then reshaped accordingly. The final matrix is squeezed to ensure the left and
        right bonds are 1.

        Returns:
            NDArray[np.complex128]: The resulting matrix after tensor contractions and reshaping.
        """
        for i, tensor in enumerate(self.tensors):
            if i == 0:
                mat = tensor
            else:
                mat = oe.contract("abcd, efdg->aebfcg", mat, tensor)
                mat = np.reshape(
                    mat, (mat.shape[0] * mat.shape[1], mat.shape[2] * mat.shape[3], mat.shape[4], mat.shape[5])
                )

        # Final left and right bonds should be 1
        return np.squeeze(mat, axis=(2, 3))

    def check_if_valid_MPO(self) -> bool:
        """Check if the current tensor network is a valid Matrix Product Operator (MPO).
        This method verifies the consistency of the bond dimensions between adjacent tensors
        in the network. Specifically, it checks that the right bond dimension of each tensor
        matches the left bond dimension of the subsequent tensor.

        Returns:
            bool: True if the tensor network is a valid MPO, False otherwise.
        """
        right_bond = self.tensors[0].shape[3]
        for tensor in self.tensors[1::]:
            assert tensor.shape[2] == right_bond
            right_bond = tensor.shape[3]
        return True

    def check_if_identity(self, fidelity: float) -> bool:
        """Check if the current MPO (Matrix Product Operator) represents an identity operation
        within a given fidelity threshold.

        Args:
            fidelity (float): The fidelity threshold to determine if the MPO is an identity.

        Returns:
            bool: True if the MPO is considered an identity within the given fidelity, False otherwise.
        """
        identity_MPO = MPO()
        identity_MPO.init_identity(self.length)

        identity_MPS = identity_MPO.convert_to_MPS()
        MPS = self.convert_to_MPS()
        trace = scalar_product(MPS, identity_MPS)

        # Checks if trace is not a singular values for partial trace
        return not np.round(np.abs(trace), 1) / 2**self.length < fidelity

    def rotate(self, conjugate: bool = False) -> None:
        """Rotates the tensors in the network by flipping the physical dimensions.
        This method transposes each tensor in the network along specified axes.
        If the `conjugate` parameter is set to True, it also takes the complex
        conjugate of each tensor before transposing.

        Args:
            conjugate (bool): If True, take the complex conjugate of each tensor
                              before transposing. Default is False.

        Returns:
            None.
        """
        for i, tensor in enumerate(self.tensors):
            if conjugate:
                tensor = np.conj(tensor)
            self.tensors[i] = np.transpose(tensor, (1, 0, 2, 3))
