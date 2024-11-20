import numpy as np
from initializations import create_system, create_pauli_x
from Max.old_tests import method_comparison_expectation_value
from functions.XXX_TwoSite_comparemthods import method_comparison_expectation_value_max
from MENDL_TN_MCWF import TN_MCWF_MENDL
import pytenet as ptn



mpoH = ptn.ising_mpo(3, 1.0, 0, -0.5)
mpoH.zero_qnumbers()

# initial wavefunction as MPS with random entries
# maximally allowed virtual bond dimensions
D = [1, 2, 2, 1]


local_operator = create_pauli_x(2)
operator = local_operator

# Create an all-zero MPS for a lattice of size L

#MPS3 = create_all_zero_mps_mendl(L)
psi = ptn.MPS(mpoH.qd, [Di*[0] for Di in D], fill=0.0)  # Fill with zeros instead of random
psi.A[0][0, 0, 0] = 1.0  # Set the first element to 1 to represent |0000...>
psi.orthonormalize(mode='left')





# Set up system parameters
model = 'Ising'  # You can change this to 'Atomic', 'Transmon', or 'Leggett' as needed
d = 2  # Local Hilbert space dimension (2 for qubits)
L = 3  # Number of sites
max_bond_dimension = 10
freq = 1.0  # Qubit frequency
T1 = 10.0  # Relaxation time
T2star = 5.0  # Dephasing time
temperature = 0.1
processes = ['relaxation', 'dephasing']  # You can add or remove processes as needed

# Model-specific parameters (for Ising model)
model_params = {
    'J': 1.0,  # Interaction strength
    'g': 0.5   # Transverse field strength
}

# Create the system
system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, 
                    temperature, processes, model_params, 
                    initial_state='+Z', calculate_exact=True, 
                    calculate_state_vector=True)

exp_values, times, output_noise_list = TN_MCWF_MENDL(
    psi=psi, 
    H=mpoH, 
    system=system, 
    num_trajectories=100, 
    T=0.02, 
    dt=0.01, 
    operator=operator, 
    operator_site=0, 
    force_noise=False, 
    input_noise_list=None
)


