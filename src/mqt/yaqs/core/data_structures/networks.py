# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

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

    # TODO: Needs to be adjusted based on current orthogonality center
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
        if form == "B":
            self.flip_network()

        self.set_canonical_form(orthogonality_center=self.length - 1)
        self.shift_orthogonality_center_right(self.length - 1)

        if form == "B":
            self.flip_network()

    def measure(self, observable: Observable) -> np.float64:
        assert observable.site in range(self.length), "State is shorter than selected site for expectation value."
        # Copying done to stop the state from messing up its own canonical form
        E = local_expval(copy.deepcopy(self), getattr(GateLibrary, observable.name)().matrix, observable.site)
        assert E.imag < 1e-13, f"Measurement should be real, '{E.real:16f}+{E.imag:16f}i'."
        return E.real

    def norm(self, site: int | None = None) -> np.float64:
        if site is not None:
            return scalar_product(self, self, site).real
        return scalar_product(self, self).real

    def write_tensor_shapes(self) -> None:
        for _tensor in self.tensors:
            pass

    def check_if_valid_MPS(self) -> None:
        right_bond = self.tensors[0].shape[2]
        for tensor in self.tensors[1::]:
            assert tensor.shape[1] == right_bond
            right_bond = tensor.shape[2]

    def check_canonical_form(self) -> list[int] | None:
        """Checks what canonical form an MPS is in if any.

        Args:
            MPS: list of rank-3 tensors with a physical dimension d^2
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
        return None


# Convention (sigma, sigma', chi_l,  chi_l+1)
class MPO:
    def init_Ising(self, length: int, J: float, g: float) -> None:
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
        self.tensors = [left_bound] + [inner] * (length - 2) + [right_bound]
        for i, tensor in enumerate(self.tensors):
            # left, right, sigma, sigma'
            self.tensors[i] = np.transpose(tensor, (2, 3, 0, 1))
        assert self.check_if_valid_MPO(), "MPO initialized wrong"
        self.length = len(self.tensors)
        self.physical_dimension = self.tensors[0].shape[0]

    def init_custom(self, tensors: list[NDArray[np.complex128]], transpose: bool = True) -> None:
        self.tensors = tensors
        if transpose:
            for i, tensor in enumerate(self.tensors):
                # left, right, sigma, sigma'
                self.tensors[i] = np.transpose(tensor, (2, 3, 0, 1))
        assert self.check_if_valid_MPO(), "MPO initialized wrong"
        self.length = len(self.tensors)
        self.physical_dimension = tensors[0].shape[0]

    def convert_to_MPS(self) -> MPS:
        converted_tensors = [
            np.reshape(tensor, (tensor.shape[0] * tensor.shape[1], tensor.shape[2], tensor.shape[3]))
            for tensor in self.tensors
        ]

        return MPS(self.length, converted_tensors)

    def convert_to_matrix(self) -> NDArray[np.complex128]:
        for i, tensor in enumerate(self.tensors):
            if i == 0:
                mat = tensor
            else:
                mat = np.einsum("abcd, efdg->aebfcg", mat, tensor)
                mat = np.reshape(
                    mat, (mat.shape[0] * mat.shape[1], mat.shape[2] * mat.shape[3], mat.shape[4], mat.shape[5])
                )

        # Final left and right bonds should be 1
        return np.squeeze(mat, axis=(2, 3))
        # mat = np.reshape(mat, mat.size)

    def write_tensor_shapes(self) -> None:
        for _tensor in self.tensors:
            pass

    def check_if_valid_MPO(self) -> bool:
        right_bond = self.tensors[0].shape[3]
        for tensor in self.tensors[1::]:
            assert tensor.shape[2] == right_bond
            right_bond = tensor.shape[3]
        return True

    def check_if_identity(self, fidelity: float) -> bool:
        identity_MPO = MPO()
        identity_MPO.init_identity(self.length)

        identity_MPS = identity_MPO.convert_to_MPS()
        MPS = self.convert_to_MPS()
        trace = scalar_product(MPS, identity_MPS)

        # Checks if trace is not a singular values for partial trace
        return not np.round(np.abs(trace), 1) / 2**self.length < fidelity

    def rotate(self, conjugate: bool = False) -> None:
        for i, tensor in enumerate(self.tensors):
            if conjugate:
                tensor = np.conj(tensor)
            self.tensors[i] = np.transpose(tensor, (1, 0, 2, 3))
