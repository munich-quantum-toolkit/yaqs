import numpy as np

from yaqs.general.libraries.tensor_library import TensorLibrary

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

    def init_custom(self, length:int, left_bound: np.ndarray, inner: np.ndarray, right_bound: np.ndarray):
        self.tensors = [left_bound] + [inner]*(length-2) + [right_bound]

    def write_tensor_shapes(self):
        for tensor in self.tensors:
            print(tensor.shape)

    def check_if_valid_MPS(self):
        right_bond = self.tensors[0].shape[3]
        for tensor in self.tensors[1::]:
            assert tensor.shape[2] == right_bond
            right_bond = tensor.shape[3]