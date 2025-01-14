import numpy as np

from yaqs.general.data_structures.networks import MPS
from yaqs.general.libraries.gate_library import GateLibrary
from yaqs.general.operations.operations import scalar_product

# Convention (sigma, sigma', chi_l,  chi_l+1)
class MPO:
    def init_Ising(self, length: int, physical_dimension: int, J: float, g: float):
        zero = np.zeros((physical_dimension, physical_dimension), dtype=complex)
        identity = np.eye(physical_dimension, dtype=complex)
        X = getattr(GateLibrary, "x")().matrix
        Z = getattr(GateLibrary, "z")().matrix

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
        X = getattr(GateLibrary, "x")().matrix
        Y = getattr(GateLibrary, "y")().matrix
        Z = getattr(GateLibrary, "z")().matrix

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

    def init_identity(self, length: int, physical_dimension: int=2):
        M = np.eye(2)
        M = np.expand_dims(M, (2, 3))
        self.length = length
        self.physical_dimension = physical_dimension

        self.tensors = []
        for _ in range(length):
            self.tensors.append(M)

    def init_custom_Hamiltonian(self, length: int, left_bound: np.ndarray, inner: np.ndarray, right_bound: np.ndarray):
        self.tensors = [left_bound] + [inner]*(length-2) + [right_bound]

    def init_custom(self, tensors: list[np.ndarray]):
        self.tensors = tensors
        self.length = len(self.tensors)
        self.physical_dimension = tensors[0].shape[0]

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

    def rotate(self, conjugate: bool=False):
        for i, tensor in enumerate(self.tensors):
            if conjugate:
                tensor = np.conj(tensor)
            self.tensors[i] = np.transpose(tensor, (1, 0, 2, 3))
