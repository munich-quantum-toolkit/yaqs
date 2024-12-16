import numpy as np

from yaqs.general.libraries.tensor_library import TensorLibrary
from yaqs.general.tensor_operations.tensor_operations import scalar_product

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yaqs.general.data_structures.MPS import MPS

# Convention (sigma, sigma', chi_l,  chi_l+1)
class MPO:
    def init_Ising(self, length: int, physical_dimension: int, J: float, g: float):
        zero = np.zeros((physical_dimension, physical_dimension))
        identity = np.identity(physical_dimension)
        X = getattr(TensorLibrary, "x")().matrix
        Z = getattr(TensorLibrary, "z")().matrix

        left_bound = np.array([identity, -J*Z, -g*X])
        left_bound = np.expand_dims(left_bound, 0)
        inner = np.array([np.array([identity, -J*Z, -g*X]),
                        np.array([zero, zero, Z]),
                        np.array([zero, zero, identity])])

        right_bound = np.array([[-g*X],
                                [Z],
                                [identity]])

        self.tensors = [left_bound] + [inner]*(length-2) + [right_bound]
        for i, tensor in enumerate(self.tensors):
            # left, right, sigma, sigma'
            self.tensors[i] = np.transpose(tensor, (2, 3, 0, 1))

        self.length = length
        self.physical_dimension = physical_dimension

    def init_identity(self, length: int):
        M = np.eye(2)
        M = np.expand_dims(M, (2, 3))

        for i in range(length):
            self.tensors[i] = M

    def init_custom(self, length:int, left_bound: np.ndarray, inner: np.ndarray, right_bound: np.ndarray):
        self.tensors = [left_bound] + [inner]*(length-2) + [right_bound]

    def convert_to_MPS(self):
        converted_tensors = []
        for tensor in self.tensors:
            converted_tensors.append(np.reshape(tensor, (tensor.shape[0]*tensor.shape[1], tensor.shape[2], tensor.shape[3])))

        return MPS(self.length, converted_tensors)

    def write_tensor_shapes(self):
        for tensor in self.tensors:
            print(tensor.shape)

    def check_if_valid_MPO(self):
        right_bond = self.tensors[0].shape[3]
        for tensor in self.tensors[1::]:
            assert tensor.shape[2] == right_bond
            right_bond = tensor.shape[3]

    def check_if_identity(self, fidelity: float):
        identity_MPO = MPO()
        identity_MPO.init_identity(self.length)

        identity_MPS = identity_MPO.convert_to_MPS()
        MPS = self.convert_to_MPS()

        trace = scalar_product(MPS, identity_MPS)

        # Checks if trace is not a singular values for partial trace
        if trace.size != 1 or np.round(np.abs(trace), 1) / 2**self.length < fidelity:
            return False
        else:
            return True