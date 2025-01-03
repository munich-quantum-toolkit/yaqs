import numpy as np

from yaqs.general.libraries.tensor_library import TensorLibrary

# Convention (sigma, sigma', chi_l,  chi_l+1)
class MPO:
    def init_Ising(self, length: int, physical_dimension: int, J: float, g: float):
        zero = np.zeros((physical_dimension, physical_dimension), dtype=complex)
        identity = np.eye(physical_dimension, dtype=complex)
        X = getattr(TensorLibrary, "x")().matrix
        Z = getattr(TensorLibrary, "z")().matrix

        # The MPO has a 3x3 block structure at each site:
        # W = [[ I,     -J Z,  -g X ],
        #      [ 0,       0,     Z  ],
        #      [ 0,       0,     I  ]]

        # Left boundary (1x3 block) selects the top row of W:
        # [I, -J Z, -g X]
        left_bound = np.array([identity, -J*Z, -g*X])[np.newaxis, :]

        # Inner tensors (3x3 block):
        inner = np.zeros((3,3,physical_dimension,physical_dimension), dtype=complex)
        inner[0,0] = identity
        inner[0,1] = -J*Z
        inner[0,2] = -g*X
        inner[1,2] = Z
        inner[2,2] = identity

        # Right boundary (3x1 block) selects the last column:
        # [ -g X, Z, I ]^T but we only take the operators that appear there.
        # Actually, at the right boundary we just pick out the last column:
        # (top row: -g X, second row: Z, third row: I)
        right_bound = np.array([[-g*X],
                                [Z],
                                [identity]])

        # Construct the MPO as a list of tensors:
        # Left boundary, (length-2)*inner, right boundary
        self.tensors = [left_bound] + [inner]*(length-2) + [right_bound]
        for i, tensor in enumerate(self.tensors):
            # left, right, sigma, sigma'
            self.tensors[i] = np.transpose(tensor, (2, 3, 0, 1))

        self.length = length
        self.physical_dimension = physical_dimension

    def init_Heisenberg(self, length: int, physical_dimension: int, Jx: float, Jy: float, Jz: float, h: float):
        zero = np.zeros((physical_dimension, physical_dimension), dtype=complex)
        identity = np.eye(physical_dimension, dtype=complex)
        X = getattr(TensorLibrary, "x")().matrix
        Y = getattr(TensorLibrary, "y")().matrix
        Z = getattr(TensorLibrary, "z")().matrix

        # Left boundary: shape (1,5, d, d)
        # [I, Jx*X, Jy*Y, Jz*Z, h*Z]
        left_bound = np.array([identity, -Jx*X, -Jy*Y, -Jz*Z, -h*Z])[np.newaxis, :]

        # Inner tensor: shape (5,5, d, d)
        # W = [[ I,    Jx*X,  Jy*Y,  Jz*Z,   h*Z ],
        #      [ 0,     0,     0,     0,     X  ],
        #      [ 0,     0,     0,     0,     Y  ],
        #      [ 0,     0,     0,     0,     Z  ],
        #      [ 0,     0,     0,     0,     I  ]]

        inner = np.zeros((5,5,physical_dimension,physical_dimension), dtype=complex)
        inner[0,0] = identity
        inner[0,1] = -Jx*X
        inner[0,2] = -Jy*Y
        inner[0,3] = -Jz*Z
        inner[0,4] = -h*Z
        inner[1,4] = X
        inner[2,4] = Y
        inner[3,4] = Z
        inner[4,4] = identity

        # Right boundary: shape (5,1, d, d)
        # [0, X, Y, Z, I]^T
        right_bound = np.array([zero, X, Y, Z, identity])[:, np.newaxis]

        # Construct the MPO
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