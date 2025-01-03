import qutip as qt
import numpy as np
import sys
import matplotlib.pyplot as plt


# Add the path to the lindbladmpo package
sys.path.append('/Users/maximilianfrohlich/lindbladmpo')

# Import the LindbladMPOSolver class
from lindbladmpo.LindbladMPOSolver import LindbladMPOSolver



''' Heisenberg Model + Noise QuTip and Lindblad MPO simulation'''

# Parameters
N = 30  # number of sites
J = 1    # X and Y coupling strength
J_z = 1  # Z coupling strength
h = 0.5  # transverse field strength
gamma_dephasing = 1 / 10.0  # dephasing rate (1/T2star)
gamma_relaxation = 1 / 10.0  # relaxation rate (1/T1)

# Time vector
T = 10
timesteps = 100
t = np.linspace(0, T, timesteps+1)

# Define parameters for LindbladMPOSolver
parameters = {
    "N": N,
    "t_final": T,
    "tau": T / (timesteps),  # time step
    "J": -2*J,  # coupling factor of XX and YY
    "J_z": -2*J_z, 
    "h_z": -2*h,
    "g_0": gamma_relaxation,  # Strength of deexcitation 
    "g_1": gamma_dephasing,  # Strength of dephasing
    "init_product_state": ["+z"]*15 + ["-z"]*15,  # initial state 
    "1q_components": ["X", "Y", "Z"],  # Request x, y, z observables
    "l_x": N,  # Length of the chain
    "l_y": 1,  # Width of the chain (1 for a 1D chain)
    "b_periodic_x": False,  # Open boundary conditions in x-direction
    "b_periodic_y": False,  # Open boundary conditions in y-direction
}

# Create a solver instance and run the simulation
solver = LindbladMPOSolver(parameters)
solver.solve()

# Access the LindbladMPO results
lindblad_mpo_results = solver.result

z_expectation_values_mpo = np.array([[solver.result['obs-1q'][('z', (i,))][1][t] 
                                for t in range(len(solver.result['obs-1q'][('z', (i,))][0]))] for i in range(N)])

import pickle

# Filepath for the pickle file
pickle_filepath = 'lindblad_mpo_results.pkl'

# Data to save (you can include all the relevant results)
data_to_save = {
    'parameters': parameters,  # Simulation parameters
    'result': lindblad_mpo_results,  # Lindblad MPO results
    # 'x_expectation_values_mpo': x_expectation_values_mpo,
    # 'y_expectation_values_mpo': y_expectation_values_mpo,
    'z_expectation_values_mpo': z_expectation_values_mpo
}

# Save data to a pickle file
with open(pickle_filepath, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"Lindblad MPO results saved to {pickle_filepath}")