# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.circuit import Parameter

    from ..data_structures.networks import MPO


def _split_tensor(tensor: NDArray[np.complex128]) -> list[NDArray[np.complex128]]:
    assert tensor.shape == (2, 2, 2, 2)

    # Splits two-qubit matrix
    matrix = np.transpose(tensor, (0, 2, 1, 3))
    dims = matrix.shape
    matrix = np.reshape(matrix, (dims[0] * dims[1], dims[2] * dims[3]))
    U, S_list, V = np.linalg.svd(matrix, full_matrices=False)
    S_list = S_list[S_list > 1e-6]
    U = U[:, 0 : len(S_list)]
    V = V[0 : len(S_list), :]

    tensor1 = U
    tensor2 = np.diag(S_list) @ V

    # Reshape into physical dimensions and bond dimension from shape
    tensor1 = np.reshape(tensor1, (2, 2, tensor1.shape[1]))
    tensor2 = np.reshape(tensor2, (tensor2.shape[0], 2, 2))
    # tensor2 = np.transpose(tensor2, (1, 0, 2))
    tensor2 = np.transpose(tensor2, (1, 2, 0))

    # Add dummy dimension to boundaries
    tensor1 = np.expand_dims(tensor1, axis=2)
    tensor2 = np.expand_dims(tensor2, axis=3)
    return [tensor1, tensor2]


def _extend_gate(tensor: NDArray[np.complex128], sites: list[int]) -> MPO:
    from ..data_structures.networks import MPO

    tensors = _split_tensor(tensor)
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
            for i, tensor in enumerate(mpo_tensors):
                mpo_tensors[i] = np.transpose(tensor, (0, 1, 3, 2))

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


class X:
    name = "x"
    matrix = np.array([[0, 1], [1, 0]])
    interaction = 1

    tensor = matrix
    # Generator: (π/2) * X
    generator = [(np.pi / 2) * np.array([[0, -1], [-1, 0]])]

    def set_sites(self, site0: int) -> None:
        self.sites = [site0]


class Y:
    name = "y"
    matrix = np.array([[0, -1j], [1j, 0]])
    interaction = 1

    tensor = matrix
    # Generator: (π/2) * Y
    generator = [(np.pi / 2) * np.array([[0, -1j], [1j, 0]])]

    def set_sites(self, site0: int) -> None:
        self.sites = [site0]


class Z:
    name = "z"
    matrix = np.array([[1, 0], [0, -1]])
    interaction = 1

    tensor = matrix
    # Generator: (π/2) * Z
    generator = [(np.pi / 2) * np.array([[1, 0], [0, -1]])]

    def set_sites(self, site0: int) -> None:
        self.sites = [site0]


class H:
    name = "h"
    matrix = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]])
    interaction = 1

    tensor = matrix
    # Generator: (π/2) * 1/2(X + Z)
    generator = [(np.pi / np.sqrt(2)) * np.array([[0.5, 0.5], [0.5, -0.5]])]

    def set_sites(self, site0: int) -> None:
        self.sites = [site0]


class I:
    name = "id"
    matrix = np.array([[1, 0], [0, 1]])
    interaction = 1

    tensor = matrix
    # Generator: 0 matrix
    generator = np.zeros((2, 2))

    def set_sites(self, site0: int) -> None:
        self.sites = [site0]


class SX:
    name = "sx"
    matrix = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
    interaction = 1

    tensor = matrix
    # Generator: (π/4) * X
    generator = [(np.pi / 4) * np.array([[0, 1], [1, 0]])]

    def set_sites(self, site0: int) -> None:
        self.sites = [site0]


class Rx:
    name = "rx"
    interaction = 1

    def set_params(self, params: list[Parameter]) -> None:
        self.theta = params[0]
        self.matrix = np.array([
            [np.cos(self.theta / 2), -1j * np.sin(self.theta / 2)],
            [-1j * np.sin(self.theta / 2), np.cos(self.theta / 2)],
        ])
        self.tensor = self.matrix
        # Generator: (θ/2) * X
        self.generator = [(self.theta / 2) * np.array([[0, -1j], [-1j, 0]])]

    def set_sites(self, site0: int) -> None:
        self.sites = [site0]


class Ry:
    name = "ry"
    interaction = 1

    def set_params(self, params: list[Parameter]) -> None:
        self.theta = params[0]
        self.matrix = np.array([
            [np.cos(self.theta / 2), -np.sin(self.theta / 2)],
            [np.sin(self.theta / 2), np.cos(self.theta / 2)],
        ])
        self.tensor = self.matrix
        # Generator: (θ/2) * Y
        self.generator = [(self.theta / 2) * np.array([[0, -1], [1, 0]])]

    def set_sites(self, site0: int) -> None:
        self.sites = [site0]


class Rz:
    name = "rz"
    interaction = 1

    def set_params(self, params: list[Parameter]) -> None:
        self.theta = params[0]
        self.matrix = np.array([[np.exp(-1j * self.theta / 2), 0], [0, np.exp(1j * self.theta / 2)]])
        self.tensor = self.matrix
        # Generator: (θ/2) * Z
        self.generator = [(self.theta / 2) * np.array([[-1j, 0], [0, 1j]])]

    def set_sites(self, site0: int) -> None:
        self.sites = [site0]


class Phase:
    name = "p"
    interaction = 1

    def set_params(self, params: list[Parameter]) -> None:
        # Phase gate has one parameter theta
        self.theta = params[0]
        self.matrix = np.array([[1, 0], [0, np.exp(1j * self.theta)]])
        self.tensor = self.matrix
        # Generator: (θ/2) * Z
        self.generator = (self.theta / 2) * np.array([[1, 0], [0, -1]])

    def set_sites(self, site0: int) -> None:
        self.sites = [site0]


class U3:
    name = "u"
    interaction = 1

    def set_params(self, params: list[Parameter]) -> None:
        self.theta, self.phi, self.lam = params
        self.matrix = np.array([
            [np.cos(self.theta / 2), -np.exp(1j * self.lam) * np.sin(self.theta / 2)],
            [
                np.exp(1j * self.phi) * np.sin(self.theta / 2),
                np.exp(1j * (self.phi + self.lam)) * np.cos(self.theta / 2),
            ],
        ])
        self.tensor = self.matrix

    def set_sites(self, site0: int) -> None:
        self.sites = [site0]


class CX:
    name = "cx"
    matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    interaction = 2

    def set_sites(self, site0: int, site1: int) -> None:
        self.sites = [site0, site1]
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        # Generator: (π/4) * (I-Z ⊗ I-X)
        self.generator = [(np.pi / 4) * np.array([[0, 0], [0, 2]]), np.array([[1, -1], [-1, 1]])]
        self.mpo = _extend_gate(self.tensor, self.sites)
        if site1 < site0:  # Adjust for reverse control/target
            # self.generator.reverse()
            self.tensor = np.transpose(self.tensor, (1, 0, 3, 2))


class CZ:
    name = "cz"
    matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
    interaction = 2

    def set_sites(self, site0: int, site1: int) -> None:
        self.sites = [site0, site1]
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        # Generator: (π/4) * (I-Z ⊗ I-Z)
        self.generator = [(np.pi / 4) * np.array([[0, 0], [0, 2]]), np.array([[1, -1], [-1, 1]])]

        if site1 < site0:  # Adjust for reverse control/target
            # self.generator.reverse()
            self.tensor = np.transpose(self.tensor, (1, 0, 3, 2))


class CPhase:
    name = "cp"
    interaction = 2

    def set_params(self, params: list[Parameter]) -> None:
        self.theta = params[0]
        self.matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(1j * self.theta)]])
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        # Generator: (θ/2) * (Z ⊗ P), where P = diag(1, 0)
        self.generator = [(self.theta / 2) * np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, 0]])]

    def set_sites(self, site0: int, site1: int) -> None:
        self.sites = [site0, site1]
        if self.interaction > 2:
            self.mpo = _extend_gate(self.tensor, self.sites)
        elif site1 < site0:  # Adjust for reverse control/target
            # self.generator.reverse()
            self.tensor = np.transpose(self.tensor, (1, 0, 3, 2))


class SWAP:
    name = "swap"
    matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    interaction = 2

    def __init__(self) -> None:
        # Generator: (π/4) * (I ⊗ I + X ⊗ X + Y ⊗ Y + Z ⊗ Z)
        # self.generator = (np.pi / 4) * (
        #     np.kron(np.eye(2), np.eye(2)) + # I ⊗ I
        #     np.kron(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])) +  # X ⊗ X
        #     np.kron(np.array([[0, -1j], [1j, 0]]), np.array([[0, -1j], [1j, 0]])) +  # Y ⊗ Y
        #     np.kron(np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, -1]]))  # Z ⊗ Z
        # )
        self.generator = [
            [
                (np.pi / 4) * np.eye(2),
                np.pi / 4 * np.array([[0, 1], [1, 0]]),
                np.pi / 4 * np.array([[0, -1j], [1j, 0]]),
                np.pi / 4 * np.array([[1, 0], [0, -1]]),
            ],
            [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])],
        ]

    def set_sites(self, site0: int, site1: int) -> None:
        self.sites = [site0, site1]
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))


class Rxx:
    name = "rxx"
    interaction = 2

    def set_params(self, params: list[Parameter]) -> None:
        self.theta = params[0]
        self.matrix = np.array([
            [np.cos(self.theta / 2), 0, 0, -1j * np.sin(self.theta / 2)],
            [0, np.cos(self.theta / 2), -1j * np.sin(self.theta / 2), 0],
            [0, -1j * np.sin(self.theta / 2), np.cos(self.theta / 2), 0],
            [-1j * np.sin(self.theta / 2), 0, 0, np.cos(self.theta / 2)],
        ])
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        # Generator: (θ/2) * (X ⊗ X)
        self.generator = [(self.theta / 2) * np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])]

    def set_sites(self, site0: int, site1: int) -> None:
        self.sites = [site0, site1]


class Ryy:
    name = "ryy"
    interaction = 2

    def set_params(self, params: list[Parameter]) -> None:
        self.theta = params[0]
        self.matrix = np.array([
            [np.cos(self.theta / 2), 0, 0, 1j * np.sin(self.theta / 2)],
            [0, np.cos(self.theta / 2), -1j * np.sin(self.theta / 2), 0],
            [0, -1j * np.sin(self.theta / 2), np.cos(self.theta / 2), 0],
            [1j * np.sin(self.theta / 2), 0, 0, np.cos(self.theta / 2)],
        ])
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        # Generator: (θ/2) * (Y ⊗ Y)
        self.generator = [(self.theta / 2) * np.array([[0, -1j], [1j, 0]]), np.array([[0, -1j], [1j, 0]])]

    def set_sites(self, site0: int, site1: int) -> None:
        self.sites = [site0, site1]


class Rzz:
    name = "rzz"
    interaction = 2

    def set_params(self, params: list[Parameter]) -> None:
        self.theta = params[0]
        self.matrix = np.array([
            [np.cos(self.theta / 2) - 1j * np.sin(self.theta / 2), 0, 0, 0],
            [0, np.cos(self.theta / 2) + 1j * np.sin(self.theta / 2), 0, 0],
            [0, 0, np.cos(self.theta / 2) + 1j * np.sin(self.theta / 2), 0],
            [0, 0, 0, np.cos(self.theta / 2) - 1j * np.sin(self.theta / 2)],
        ])
        self.tensor: NDArray[np.complex128] = np.reshape(self.matrix, (2, 2, 2, 2))
        # Generator: (θ/2) * (Z ⊗ Z)
        self.generator = [(self.theta / 2) * np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, -1]])]

    def set_sites(self, site0: int, site1: int) -> None:
        self.sites = [site0, site1]


# class U2:
#     name = 'u2'
#     interaction = 1

#     def set_params(self, params):
#         # U2 gate has parameters phi and lambda
#         self.phi = params[0]
#         self.lam = params[1]
#         self.matrix = np.array([
#             [1, -np.exp(1j * self.lam)],
#             [np.exp(1j * self.phi), np.exp(1j * (self.phi + self.lam))]
#         ]) / np.sqrt(2)
#         self.tensor = self.matrix
#         # Generator: Derived from the U2 unitary matrix
#         self.generator = (1 / np.sqrt(2)) * np.array([
#             [0, -np.exp(-1j * self.lam)],
#             [np.exp(-1j * self.phi), 0]
#         ])

#     def set_sites(self, site0: int):
#         self.sites = [site0]


# class CCX:
#     matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
#                         [0, 1, 0, 0, 0, 0, 0, 0],
#                         [0, 0, 1, 0, 0, 0, 0, 0],
#                         [0, 0, 0, 1, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 1, 0, 0, 0],
#                         [0, 0, 0, 0, 0, 1, 0, 0],
#                         [0, 0, 0, 0, 0, 0, 0, 1],
#                         [0, 0, 0, 0, 0, 0, 1, 0],])
#     # matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
#     #                     [0, 1, 0, 0, 0, 0, 0, 0],
#     #                     [0, 0, 1, 0, 0, 0, 0, 0],
#     #                     [0, 0, 0, 0, 0, 0, 0, 1],
#     #                     [0, 0, 0, 0, 1, 0, 0, 0],
#     #                     [0, 0, 0, 0, 0, 1, 0, 0],
#     #                     [0, 0, 0, 0, 0, 0, 1, 0],
#     #                     [0, 0, 0, 1, 0, 0, 0, 0],])

#     interaction = 3

#     def set_sites(self, site0: int, site1: int, site2: int):
#         self.sites = [site0, site1, site2]
#         self.tensor = np.reshape(self.matrix, (2, 2, 2, 2, 2, 2))

#         self.interaction = np.abs(site0 - site2)+1
#         self.tensor = _extend_gate(self.tensor, self.sites)


class GateLibrary:
    x = X
    y = Y
    z = Z
    sx = SX
    h = H
    id = I
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
    # ccx = CCX
    cp = CPhase
    # u2 = U2
    p = Phase
