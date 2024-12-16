import copy
import numpy as np
import opt_einsum as oe

from yaqs.general.data_structures.simulation_parameters import Observable
from yaqs.general.libraries.tensor_library import TensorLibrary
from yaqs.physics.tensor_operations.operations import local_expval, scalar_product


# Convention (sigma, chi_l-1, chi_l)
class MPS:
    def __init__(self, length: int, physical_dimensions: list=[], state: str='zeros'):
        self.tensors = []
        self.length = length
        self.physical_dimensions = physical_dimensions
        if not physical_dimensions:
            # Default case is the qubit (2-level) case
            for _ in range(length):
                self.physical_dimensions.append(2)
        assert len(physical_dimensions) == length

        # Create d-level |0> state
        for d in physical_dimensions:
            vector = np.zeros(d)
            if state == 'zeros':
                vector[0] = 1
            elif state == 'ones':
                vector[1] = 1
            else:
                raise ValueError("Invalid state string")

            tensor = np.expand_dims(vector, axis=(0, 1))

            tensor = np.transpose(tensor, (2, 0, 1))
            self.tensors.append(tensor)
        self.flipped = False
        # self.orthogonality_center = 0

    def write_max_bond_dim(self) -> int:
        global_max = 0
        for tensor in self.tensors:
            local_max = max(tensor.shape[0], tensor.shape[2])
            if local_max > global_max:
                global_max = local_max

        return global_max

    def flip_network(self):
        """ Flips the bond dimensions in the network so that we can do operations
            from right to left

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

    def shift_orthogonality_center_right(self, current_orthogonality_center):
        """ Left and right normalizes an MPS around a selected site

        Args:
            MPS: list of rank-3 tensors with a physical dimension d^2
            selected_site: site of matrix M around which we normalize
        Returns:
            new_MPS: list of rank-3 tensors at each site
        """

        tensor = self.tensors[current_orthogonality_center]
        old_dims = tensor.shape
        matricized_tensor = np.reshape(tensor, (tensor.shape[0]*tensor.shape[1], tensor.shape[2]))
        Q, R = np.linalg.qr(matricized_tensor)
        Q = np.reshape(Q, (old_dims[0], old_dims[1], old_dims[2]))
        self.tensors[current_orthogonality_center] = Q

        # If normalizing, we just throw away the R
        if current_orthogonality_center+1 < self.length:
            self.tensors[current_orthogonality_center+1] = oe.contract('ij, ajc->aic', R, self.tensors[current_orthogonality_center+1])

    def shift_orthogonality_center_left(self, current_orthogonality_center):
        """ Left and right normalizes an MPS around a selected site

        Args:
            MPS: list of rank-3 tensors with a physical dimension d^2
            selected_site: site of matrix M around which we normalize
        Returns:
            new_MPS: list of rank-3 tensors at each site
        """
        self.flip_network()
        self.shift_orthogonality_center_right(self.length-current_orthogonality_center-1)
        self.flip_network()

    # TODO: Needs to be adjusted based on current orthogonality center
    #       Rather than sweeping the full chain
    def set_canonical_form(self, orthogonality_center: int):
        """ Left and right normalizes an MPS around a selected site
        Args:
            MPS: list of rank-3 tensors with a physical dimension d^2
            selected_site: site of matrix M around which we normalize
        Returns:
            new_MPS: list of rank-3 tensors at each site
        """
        def sweep_decomposition(orthogonality_center: int):
            for site, _ in enumerate(self.tensors):
                if site == orthogonality_center:
                    break
                self.shift_orthogonality_center_right(site)

        sweep_decomposition(orthogonality_center)
        self.flip_network()
        flipped_orthogonality_center = self.length-1-orthogonality_center
        sweep_decomposition(flipped_orthogonality_center)
        self.flip_network()

    def normalize(self, form: str='B'):
        if form == 'B':
            self.flip_network()
        
        self.set_canonical_form(orthogonality_center=self.length-1)
        self.shift_orthogonality_center_right(self.length-1)

        if form == 'B':
            self.flip_network()

    def measure(self, observable: 'Observable'):
        assert observable.site in range(0, self.length), "State is shorter than selected site for expectation value."
        # Copying done to stop the state from messing up its own canonical form
        return local_expval(copy.deepcopy(self), getattr(TensorLibrary, observable.name)().matrix, observable.site)

    def norm(self):
        return scalar_product(self, self)

    def write_tensor_shapes(self):
        for tensor in self.tensors:
            print(tensor.shape)

    def check_if_valid_MPS(self):
        right_bond = self.tensors[0].shape[2]
        for tensor in self.tensors[1::]:
            assert tensor.shape[1] == right_bond
            right_bond = tensor.shape[2]

    def check_canonical_form(self):
        """ Checks what canonical form an MPS is in if any

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
            M = oe.contract('ijk, ijl->kl', A[i], B[i])
            M[M < epsilon] = 0
            test_identity = np.eye(M.shape[0], dtype=complex)
            A_truth.append(np.allclose(M, test_identity))

        for i in range(len(A)):
            M = oe.contract('ijk, ibk->jb', B[i], A[i])
            M[M < epsilon] = 0
            test_identity = np.eye(M.shape[0], dtype=complex)
            B_truth.append(np.allclose(M, test_identity))

        print(A_truth)
        print(B_truth)
        if all(A_truth):
            print("MPS is left (A) canonical.")
            print("MPS is site canonical at site % d" % (self.length-1))
            return [self.length-1]

        if all(B_truth):
            print("MPS is right (B) canonical.")
            print("MPS is site canonical at site 0")
            return [0]

        if not (all(A_truth) and all(B_truth)):
            sites = []
            for truth_value in A_truth:
                if truth_value:
                    sites.append(truth_value)
                else:
                    break
            for truth_value in B_truth[len(sites):]:
                sites.append(truth_value)

            try:
                return sites.index(False)
                # print("MPS is site canonical at site % d." % sites.index(False))
            except:
                # form = []
                for i, value in enumerate(A_truth):
                    if not value:
                        return [i-1, i]
                    # if A_truth[i]:
                    #     form.append('A')
                    # elif B_truth[i]:
                    #     form.append('B')
                # print("The MPS has the form: ", form)
