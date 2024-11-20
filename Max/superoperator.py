from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import time

def create_superoperator(Lindblad_operators, H_0, coupling_factors):
    """ Creates a superoperator for given Lindblad operators

    Args:
        Lindblad_operators: list of numpy array
             List of operators being applied in master equation
        H_0: numpy array
            Model Hamiltonian
        coupling_factors: list of floats
             List of gammas used in master equation

    Returns:
        superoperator: numpy array
            Matrix form of Lindblad master equation
    """
    
    identity = scipy.sparse.eye(H_0.shape[0], format='csc')
    H_0_sparse = scipy.sparse.csc_matrix(H_0)
    commutator_term = -1j*(scipy.sparse.kron(H_0_sparse, identity, format='csc') - scipy.sparse.kron(identity, H_0_sparse.T, format='csc'))

    superoperator = commutator_term

    for i, L in enumerate(Lindblad_operators):
        L_sparse = scipy.sparse.csc_matrix(L)
        gamma = coupling_factors[i]

        term_1 = scipy.sparse.kron(L_sparse, np.conj(L_sparse), format='csc')
        term_2 = scipy.sparse.kron(np.conj(L_sparse.T) @ L_sparse, identity, format='csc')
        term_3 = scipy.sparse.kron(identity, L_sparse.T @ np.conj(L_sparse), format='csc')
        superoperator += gamma/2*(2*term_1 - term_2 - term_3)

    return superoperator


def time_evolution_superoperator(system, T, plot_spectrum=False):
    """ Density matrix time evolution according to the master equation
        using the superoperator form

    Args:
        density_matrix: numpy array
            Matrix corresponding to quantum state
        H_0: numpy array
            Model Hamiltonian
        Lindblad_operators: list of nummpy array
            List of operators being applied in master equation
        coupling_factors: list of floats
            Gammas used in master equation
        T: float
            Elapsed time
        plot_spectrum: bool
            Inputs whether to plot spectrum of superoperator matrix

    Returns:
        rho: numyp array
            Evolved density matrix
    """
    def spectrum(eigenvalues):
        density_dict = dict(Counter(eigenvalues))

        data = density_dict.keys()
        # extract real part
        x = [ele.real for ele in data]
        # extract imaginary part
        y = [ele.imag for ele in data]

        z = list(density_dict.values())

        plt.scatter(x, y, c=z, cmap='plasma')
        plt.colorbar(label='Degeneracy')
        plt.ylabel('Imaginary')
        plt.xlabel('Real')
        plt.title("Spectrum of Superoperator")
        plt.show()

    # Superoperator does not care if operators are relaxation, dephasing, etc.
    # so we need to unpack them into an agnostic list
    unpacked_operators = sum(system.operators.values(), [])
    unpacked_coupling_factors = sum(system.coupling_factors.values(), [])

    rho = np.ravel(system.density_matrix)
    rho_sparse = scipy.sparse.csc_matrix(rho)
    L = create_superoperator(unpacked_operators, system.H_0, unpacked_coupling_factors)
    if plot_spectrum:
        L_dense = L.toarray()
        eigenvalues, _ = np.linalg.eig(L_dense)
        print(eigenvalues)
        spectrum(eigenvalues)

    print("Start SO calculation")
    start = time.time()
    rho_prime = scipy.sparse.linalg.expm(T*L) @ rho_sparse.T

    rho_prime = np.reshape(rho_prime, system.density_matrix.shape).toarray()
    rho_prime = rho_prime/np.trace(rho_prime)
    print("Elapsed: ", time.time()-start)
    return rho_prime