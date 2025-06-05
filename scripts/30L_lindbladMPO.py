import sys
import numpy as np
import pickle
import os
import time
import qutip as qt

sys.path.append('/Users/maximilianfrohlich/lindbladmpo')
from lindbladmpo.LindbladMPOSolver import LindbladMPOSolver


# Parameters
N = 30 # number of sites
half_N = N // 2  # half the number of sites for initial state
J = 1    # X and Y coupling strength
J_z = 1  # Z coupling strength
h = 1  # transverse field strength
gamma_dephasing = 0.1  # dephasing rate (1/T2star)
gamma_relaxation = 0.1  # relaxation rate (1/T1)

# Time vector
T = 10
timesteps = 100
t = np.linspace(0, T, timesteps+1)

# # Qutip setup

# # Define Pauli matrices
# sx = qt.sigmax()
# sy = qt.sigmay()
# sz = qt.sigmaz()

# # Construct the Ising Hamiltonian
# H = 0
# for i in range(N-1):
#     H += J * qt.tensor([sx if n==i or n==i+1 else qt.qeye(2) for n in range(N)])
#     H += J * qt.tensor([sy if n==i or n==i+1 else qt.qeye(2) for n in range(N)])
#     H += J * qt.tensor([sz if n==i or n==i+1 else qt.qeye(2) for n in range(N)])

# for i in range(N):
#     H += h * qt.tensor([sz if n==i else qt.qeye(2) for n in range(N)])

# # Construct collapse operators
# c_ops = []

# # Dephasing operators
# for i in range(N):
#     c_ops.append(np.sqrt(gamma_dephasing) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(N)]))

# # Relaxation operators
# for i in range(N):
#     c_ops.append(np.sqrt(gamma_relaxation) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(N)]))

# # Initial state
# # psi0 = qt.tensor([qt.basis(2, 0) for _ in range(N)])
# psi0 = qt.tensor([qt.basis(2, 0) for _ in range(half_N)] + [qt.basis(2, 1) for _ in range(half_N)])


# # Define measurement operators
# sz_list = [qt.tensor([sz if n == i else qt.qeye(2) for n in range(N)]) for i in range(N)]

# # Exact Lindblad solution
# result_lindblad = qt.mesolve(H, psi0, t, c_ops, sz_list, progress_bar=True)




# Define parameters for LindbladMPOSolver
parameters = {
    "N": N,
    "t_final": T,
    "tau": T / (timesteps),  # time step
    "J": -2*J,  # coupling factor of XX and YY
    "J_z": -2*J_z, 
    "h_z": -2*h,
    "g_0": gamma_relaxation,  # Strength of deexcitation 
    "g_2": gamma_dephasing,  # Strength of dephasing
    "init_product_state": ["+z"]*half_N + ["-z"]*half_N,  # initial state 
    "1q_components": ["Z"],  # Request x, y, z observables
    "2q_components": [],      # No 2-site observables
    "3q_components": [],      # No 3-site observables
    "l_x": N,  # Length of the chain
    "l_y": 1,  # Width of the chain (1 for a 1D chain)
    "b_periodic_x": False,  # Open boundary conditions in x-direction
    "b_periodic_y": False,  # Open boundary conditions in y-direction
}

# Create a solver instance and run the simulation
solver = LindbladMPOSolver(parameters)
starting_time = time.time()
solver.solve()
simulation_time = time.time() - starting_time
print(f"Simulation completed in {simulation_time:.2f} seconds.")



# Access the LindbladMPO results
lindblad_mpo_results = solver.result


z_expectation_values_mpo = np.array([[solver.result['obs-1q'][('z', (i,))][1][t] 
                                for t in range(len(solver.result['obs-1q'][('z', (i,))][0]))] for i in range(N)])


import matplotlib.pyplot as plt

# # QuTiP: shape is (len(t), N), transpose to (N, len(t))
# z_expectation_values_qutip = np.array(result_lindblad.expect)

pickle_filepath = "/Users/maximilianfrohlich/Documents/GitHub/mqt-yaqs/scripts/lindblad_mpo_results.pkl"


data_to_save = {
    'parameters': parameters,                      # Simulation parameters
    'result': lindblad_mpo_results,               # Lindblad MPO results
    'z_expectation_values_mpo': z_expectation_values_mpo,  # Observables
    'simulation_time': simulation_time            # Total simulation time in seconds
}


with open(pickle_filepath, 'wb') as f:
    pickle.dump(data_to_save, f)

# Plot comparison
plt.figure(figsize=(10, 6))
for i in range(N):
    # plt.plot(t, z_expectation_values_qutip[i], label=f"Qutip - Site {i}", linestyle="--")
    plt.plot(t, z_expectation_values_mpo[i], label=f"MPO - Site {i}", linestyle="-")
    # plt.plot(t, z_expectation_values_mpo[i]-z_expectation_values_qutip[i], label=f"difference - Site {i} (Z)", linestyle=":")

plt.xlabel("Time")
plt.ylabel("Z Expectation Value")
plt.title("Comparison: QuTiP vs Lindblad MPO")
plt.legend(ncol=2, fontsize="small")
plt.grid(True)
plt.tight_layout()
plt.show()

