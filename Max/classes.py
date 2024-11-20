import numpy as np

from canonical_forms import normalize


class System:
    """
    Contains all the parameters needed to do time-evolution with the superoperator
        and stochastic unraveling

    ...

    Attributes
    ----------
    density_matrix : numpy array
        Density matrix of the full system
    state_vector : numpy array
        State vector of the full system
    L : int
        Number of particles
    operators: list of numpy arrays
        List of local operators acting on system
    coupling_factors: list of floats
        List of gammas for each operator
    H_0: numpy array
        Matrix Hamiltonian for entire system
    H_eff: numpy array
        Matrix Hamiltonian for entire system with
        non-Hermitian part added using operators
    """

    def __init__(self, model, model_params, d, D, density_matrix, state_vector, resonator_state, max_bond_dimension, H_0, H_0_MPO, L, operators, coupling_factors, single_site_state, T1, T2star, temperature, processes, calculate_exact, calculate_state_vector):
        self.model = model
        self.model_params = model_params
        self.d = d
        self.D = D
        self.density_matrix = density_matrix
        self.state_vector = state_vector
        self.resonator_state = resonator_state
        self.max_bond_dimension = max_bond_dimension
        self.L = L
        self.operators = operators
        self.coupling_factors = coupling_factors
        self.H_0 = H_0
        self.H_0_MPO = H_0_MPO
        self.single_site_state = single_site_state
        self.T1 = T1
        self.T2star = T2star
        self.temperature = temperature
        self.processes = processes
        self.calculate_exact = calculate_exact
        self.calculate_state_vector = calculate_state_vector

        qubit_tensor = np.expand_dims(self.single_site_state, axis=(0, 1))
        resonator_tensor = np.expand_dims(self.resonator_state, axis=(0, 1))

        self.MPS = []
        for i in range(self.L):
            psi_tensor = qubit_tensor
            if self.model == 'Transmon' and i in self.model_params['resonator_sites']:
                psi_tensor = resonator_tensor
            if self.model == 'Transmon' and i == 0:
                psi_tensor = np.zeros(d)
                psi_tensor[1] = 1
                psi_tensor =  np.expand_dims(psi_tensor, axis=(0, 1))

            phys_dim = psi_tensor.shape[-1]
            if i == 0:
                pad = min(phys_dim-1, self.max_bond_dimension-1)
                # Pad right bond
                M = np.concatenate((psi_tensor, np.zeros((1, pad, phys_dim))), axis=1)
            elif i == self.L-1:
                pad = min(phys_dim-1, self.max_bond_dimension-1)
                # Pad left bond
                M = np.concatenate((psi_tensor, np.zeros((pad, 1, phys_dim))), axis=0)
            else:
                left_pad = min(self.MPS[i-1].shape[1] - 1, self.max_bond_dimension-1)
                right_pad = min(min(phys_dim**(i+1), phys_dim**(self.L-(i+1)))-1, self.max_bond_dimension-1)
                # TODO: Generalize, remove brute force
                if self.model == 'Transmon':
                    left_pad = min(self.d-1, self.max_bond_dimension-1)
                    right_pad = min(self.d-1, self.max_bond_dimension-1)
                M = np.concatenate((psi_tensor, np.zeros((left_pad, 1, phys_dim))), axis=0)
                M = np.concatenate((M, np.zeros((1+left_pad, right_pad, phys_dim))), axis=1)

            self.MPS.append(M)
        self.MPS = normalize(self.MPS, form='B')

        if (calculate_exact or calculate_state_vector) and self.model is not 'Transmon':
            H_jump = np.zeros(self.H_0.shape)
            for process in self.processes:
                for i, op in enumerate(self.operators[process]):
                    #TODO: Change iteration over local_gammas
                    try:
                        H_jump += self.coupling_factors[process][i]*np.conj(op.T) @ op
                    except:
                        H_jump = np.add(H_jump, self.coupling_factors[process][i]*np.conj(op.T) @ op, out=H_jump, casting="unsafe")


            self.H_non_Hermitian = -1j/2*H_jump
            self.H_eff = self.H_0 + self.H_non_Hermitian


class NoiseModel:
    def __init__(self, times, jump_types, sites, operators):
        self.times = times
        self.jump_types = jump_types
        self.sites = sites
        self.operators = operators