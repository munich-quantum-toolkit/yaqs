import numpy as np

import yaqs.general.data_structures.MPO

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yaqs.general.data_structures.MPO import MPO


def _split_tensor(tensor: np.ndarray) -> list[np.ndarray]:
    assert tensor.shape == (2, 2, 2, 2)

    # Splits two-qubit matrix
    matrix = np.transpose(tensor, (0, 2, 1, 3))
    dims = matrix.shape
    matrix = np.reshape(matrix, (dims[0]*dims[1], dims[2]*dims[3]))
    U, S_list, V = np.linalg.svd(matrix, full_matrices=False)
    S_list = S_list[S_list > 1e-6]
    U = U[:, 0:len(S_list)]
    V = V[0:len(S_list), :]

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
    tensors = [tensor1, tensor2]

    return tensors


def _extend_gate(tensor: np.ndarray, sites: list) -> 'MPO':
    tensors = _split_tensor(tensor)
    if len(tensors) == 2:
    # Adds identity tensors between sites
        mpo_tensors = [tensors[0]]
        for _ in range(np.abs(sites[0]-sites[1])-1):
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
        for _ in range(np.abs(sites[0]-sites[1])-1):
            previous_right_bond = mpo_tensors[-1].shape[3]
            identity_tensor = np.zeros((2, 2, previous_right_bond, previous_right_bond))
            for i in range(previous_right_bond):
                identity_tensor[:, :, i, i] = np.identity(2)
            mpo_tensors.append(identity_tensor)
        mpo_tensors.append(tensors[1])
        for _ in range(np.abs(sites[1]-sites[2])-1):
            previous_right_bond = mpo_tensors[-1].shape[3]
            identity_tensor = np.zeros((2, 2, previous_right_bond, previous_right_bond))
            for i in range(previous_right_bond):
                identity_tensor[:, :, i, i] = np.identity(2)
            mpo_tensors.append(identity_tensor)
        mpo_tensors.append(tensors[2])

    mpo = yaqs.general.data_structures.MPO.MPO()
    mpo.init_custom(mpo_tensors)
    return mpo

class X:
    name = 'x'
    matrix = np.array([[0, 1],
                       [1, 0]])
    interaction = 1

    tensor = matrix

    def set_sites(self, site0: int):
        self.sites = [site0]


class Y:
    name = 'y'

    matrix = np.array([[0, -1j],
                       [1j, 0]])
    interaction = 1

    tensor = matrix

    def set_sites(self, site0: int):
        self.sites = [site0]


class Z:
    name = 'z'

    matrix = np.array([[1, 0],
                       [0, -1]])
    interaction = 1

    tensor = matrix

    def set_sites(self, site0: int):
        self.sites = [site0]


class SX:
    name = 'sx'

    matrix = 1/2*np.array([[1+1j, 1-1j],
                       [1-1j, 1+1j]])
    interaction = 1

    tensor = matrix

    def set_sites(self, site0: int):
        self.sites = [site0]


class H:
    name = 'h'

    matrix = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
                       [1/np.sqrt(2), -1/np.sqrt(2)]])
    interaction = 1

    tensor = matrix

    def set_sites(self, site0: int):
        self.sites = [site0]


class I:
    name = 'id'

    matrix = np.array([[1, 0],
                       [0, 1]])
    interaction = 1

    tensor = matrix

    def set_sites(self, site0: int):
        self.sites = [site0]


class U2:
    name = 'u2'

    interaction = 1

    def set_params(self, params):
        self.phi = params[0]
        self.lam = params[1]
        self.matrix = np.array([[1, -(np.cos(self.lam)+1j*np.sin(self.lam))],
                                [np.cos(self.phi)+1j*np.sin(self.phi), np.cos(self.phi+self.lam)+1j*np.sin(self.phi+self.lam)]])
        self.tensor = self.matrix

    def set_sites(self, site0: int):
        self.sites = [site0]


class Phase:
    name = 'p'

    interaction = 1

    def set_params(self, params):
        self.theta = params[0]
        self.matrix = np.array([[1, 0],
                                [0, np.cos(self.theta)+1j*np.sin(self.theta)]])
        self.tensor = self.matrix

    def set_sites(self, site0: int):
        self.sites = [site0]



class Rx:
    name = 'rx'

    interaction = 1

    def set_params(self, params):
        self.theta = params[0]
        self.matrix = np.array([[np.cos(self.theta/2), -1j*np.sin(self.theta/2)],
                                [-1j*np.sin(self.theta/2), np.cos(self.theta/2)]])
        self.tensor = self.matrix

    def set_sites(self, site0: int):
        self.sites = [site0]


class Ry:
    name = 'ry'

    interaction = 1

    def set_params(self, params):
        self.theta = params[0]
        self.matrix = np.array([[np.cos(self.theta/2), -np.sin(self.theta/2)],
                                [np.sin(self.theta/2), np.cos(self.theta/2)]])
        self.tensor = self.matrix

    def set_sites(self, site0: int):
        self.sites = [site0]


class Rz:
    name = 'rz'

    interaction = 1

    def set_params(self, params):
        self.theta = params[0]
        self.matrix = np.array([[np.cos(self.theta/2)-1j*np.sin(self.theta/2), 0],
                                [0, np.cos(self.theta/2)+1j*np.sin(self.theta/2)]])
        self.tensor = self.matrix

    def set_sites(self, site0: int):
        self.sites = [site0]


class CX:
    name = 'cx'

    matrix = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])

    def set_sites(self, site0: int, site1: int):
        self.sites = [site0, site1]
        self.tensor = np.reshape(self.matrix, (2, 2, 2, 2))
        self.interaction = np.abs(site0 - site1)+1

        if self.interaction > 2:
            self.mpo = _extend_gate(self.tensor, self.sites)
        else:
            # If control is above target
            if site1 < site0:
                self.tensor = np.transpose(self.tensor, (1, 0, 3, 2))


class CZ:
    name = 'cz'

    matrix = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, -1]])
    interaction = 2

    def set_sites(self, site0: int, site1: int):
        self.sites = [site0, site1]
        self.tensor = np.reshape(self.matrix, (2, 2, 2, 2))
        self.interaction = np.abs(site0 - site1)+1
        if self.interaction > 2:
            self.mpo = _extend_gate(self.tensor, self.sites)
        else:
            # If control is above target
            if site1 < site0:
                self.tensor = np.transpose(self.tensor, (1, 0, 3, 2))


class CPhase:
    name = 'cp'

    def set_params(self, params):
        self.theta = params[0]

        self.matrix = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, np.cos(self.theta)+1j*np.sin(self.theta)]])

        self.tensor = np.reshape(self.matrix, (2, 2, 2, 2))

    def set_sites(self, site0: int, site1: int):
        assert self.theta, 'Theta must be set first'
        self.sites = [site0, site1]
        self.tensor = np.reshape(self.matrix, (2, 2, 2, 2))
        self.interaction = np.abs(site0 - site1)+1
        if self.interaction > 2:
            self.mpo = _extend_gate(self.tensor, self.sites)
        else:
            # If control is above target
            if site1 < site0:
                self.tensor = np.transpose(self.tensor, (1, 0, 3, 2))

class SWAP:
    name = 'swap'

    matrix = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]])
    interaction = 2

    def set_sites(self, site0: int, site1: int):
        self.sites = [site0, site1]
        self.tensor = np.reshape(self.matrix, (2, 2, 2, 2))
        self.interaction = np.abs(site0 - site1)+1
        if self.interaction > 2:
            self.mpo = _extend_gate(self.tensor, self.sites)


class Rxx:
    name = 'rxx'

    def set_params(self, params):
        self.theta = params[0]

        self.matrix = np.array([[np.cos(self.theta/2), 0, 0, -1j*np.sin(self.theta/2)],
                                [0, np.cos(self.theta/2), -1j*np.sin(self.theta/2), 0],
                                [0, -1j*np.sin(self.theta/2), np.cos(self.theta/2), 0],
                                [-1j*np.sin(self.theta/2), 0, 0, np.cos(self.theta/2)]])

        self.tensor = np.reshape(self.matrix, (2, 2, 2, 2))

    def set_sites(self, site0: int, site1: int):
        assert self.theta, 'Theta must be set first'
        self.sites = [site0, site1]
        self.tensor = np.reshape(self.matrix, (2, 2, 2, 2))
        self.interaction = np.abs(site0 - site1)+1
        if self.interaction > 2:
            self.mpo = _extend_gate(self.tensor, self.sites)


class Ryy:
    name = 'ryy'

    def set_params(self, params):
        self.theta = params[0]

        self.matrix = np.array([[np.cos(self.theta/2), 0, 0, 1j*np.sin(self.theta/2)],
                                [0, np.cos(self.theta/2), -1j*np.sin(self.theta/2), 0],
                                [0, -1j*np.sin(self.theta/2), np.cos(self.theta/2), 0],
                                [1j*np.sin(self.theta/2), 0, 0, np.cos(self.theta/2)]])

        self.tensor = np.reshape(self.matrix, (2, 2, 2, 2))

    def set_sites(self, site0: int, site1: int):
        assert self.theta, 'Theta must be set first'
        self.sites = [site0, site1]
        self.tensor = np.reshape(self.matrix, (2, 2, 2, 2))
        self.interaction = np.abs(site0 - site1)+1
        if self.interaction > 2:
            self.mpo = _extend_gate(self.tensor, self.sites)


class Rzz:
    name = 'rzz'

    def set_params(self, params):
        self.theta = params[0]

        self.matrix = np.array([[np.cos(self.theta/2)-1j*np.sin(self.theta/2), 0, 0, 0],
                                [0, np.cos(self.theta/2)+1j*np.sin(self.theta/2), 0, 0],
                                [0, 0, np.cos(self.theta/2)+1j*np.sin(self.theta/2), 0],
                                [0, 0, 0, np.cos(self.theta/2)-1j*np.sin(self.theta/2)]])

        self.tensor = np.reshape(self.matrix, (2, 2, 2, 2))

    def set_sites(self, site0: int, site1: int):
        assert self.theta, 'Theta must be set first'
        self.sites = [site0, site1]
        self.tensor = np.reshape(self.matrix, (2, 2, 2, 2))
        self.interaction = np.abs(site0 - site1)+1
        if self.interaction > 2:
            self.mpo = _extend_gate(self.tensor, self.sites)




class CCX:
    matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1, 0],])
    # matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 1, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 1, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 1],
    #                     [0, 0, 0, 0, 1, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 1, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 1, 0],
    #                     [0, 0, 0, 1, 0, 0, 0, 0],])

    interaction = 3

    def set_sites(self, site0: int, site1: int, site2: int):
        self.sites = [site0, site1, site2]
        self.tensor = np.reshape(self.matrix, (2, 2, 2, 2, 2, 2))

        self.interaction = np.abs(site0 - site2)+1
        self.tensor = _extend_gate(self.tensor, self.sites)


class TensorLibrary:
    x = X
    y = Y
    z = Z
    sx = SX
    h = H
    id = I
    rx = Rx
    ry = Ry
    rz = Rz
    cx = CX
    cz = CZ
    swap = SWAP
    rxx = Rxx
    ryy = Ryy
    rzz = Rzz
    ccx = CCX
    cp = CPhase
    u2 = U2
    p = Phase