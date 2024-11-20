import numpy as np
import pytenet as ptn
from initializations import create_system, create_pauli_x, create_pauli_z, create_pauli_y
from Dynamic_TN_MCWF import TN_MCWF_DynamicTDVP
import matplotlib.pyplot as plt
import qutip as qt
import sys

# Add the path to the lindbladmpo package
sys.path.append('/Users/maximilianfrohlich/lindbladmpo')

# Import the LindbladMPOSolver class
from lindbladmpo.LindbladMPOSolver import LindbladMPOSolver



'''Ising model + Noise simulation QuTip and Lindblad MPO'''

#region
# # Parameters
# L = 4  # number of sites
# J = 1.0  # Ising coupling strength
# h = 0.5  # transverse field strength
# gamma_dephasing = 1 / 10.0  # dephasing rate (1/T2star)
# gamma_relaxation = 1 / 10.0  # relaxation rate (1/T1)

# # Time vector
# T = 10
# timesteps = 100
# t = np.linspace(0, T, timesteps)
# print(len(t))



# # Define Pauli matrices
# sx = qt.sigmax()
# sy = qt.sigmay()
# sz = qt.sigmaz()

# # Construct the Ising Hamiltonian
# H = 0
# for i in range(L-1):
#     H += -J * qt.tensor([sz if n==i or n==i+1 else qt.qeye(2) for n in range(L)])
# for i in range(L):
#     H += -h * qt.tensor([sx if n==i else qt.qeye(2) for n in range(L)])

# # Construct collapse operators
# c_ops = []

# # Dephasing operators
# for i in range(L):
#     c_ops.append(np.sqrt(gamma_dephasing) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(L)]))

# # Relaxation operators
# for i in range(L):
#     c_ops.append(np.sqrt(gamma_relaxation) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(L)]))

# # Initial state (all spins up)
# psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])


# # Define measurement operators
# sx_list = [qt.tensor([sx if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]
# sy_list = [qt.tensor([sy if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]
# sz_list = [qt.tensor([sz if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]

# # Exact Lindblad solution
# result_lindblad = qt.mesolve(H, psi0, t, c_ops, sx_list + sy_list + sz_list, progress_bar=True)

# # Define parameters for LindbladMPOSolver
# parameters = {
#     "N": L,
#     "t_final": T,
#     "tau": T / (timesteps - 1),  # time step
#     "J_z": -2*J,
#     "h_x": -2*h,
#     "g_0": gamma_relaxation,  # Strength of deexcitation 
#     "g_2": gamma_dephasing,  # Strength of dephasing
#     "init_product_state": ["+z"] * L,  # initial state 
#     "1q_components": ["X", "Y", "Z"],  # Request x observable
#     "l_x": L,  # Length of the chain
#     "l_y": 1,  # Width of the chain (1 for a 1D chain)
#     "b_periodic_x": False,  # Open boundary conditions in x-direction
#     "b_periodic_y": False,  # Open boundary conditions in y-direction
# }

# # Create a solver instance and run the simulation
# solver = LindbladMPOSolver(parameters)
# solver.solve()

# # Access the LindbladMPO results
# lindblad_mpo_results = solver.result



# # Plot the results
# fig, axs = plt.subplots(3, 1, figsize=(15, 15))

# components = ['X', 'Y', 'Z']
# for idx, component in enumerate(components):
#     ax = axs[idx]
    
#     # Plot Exact Lindblad results
#     for i in range(L):
#         ax.plot(t, result_lindblad.expect[idx*L + i], label=f'Exact ⟨σ{component.lower()}⟩ Spin {i+1}', linestyle='-')
    
#     # Plot LindbladMPO results
#     component_lower = component.lower()
#     for i in range(L):
#         mpo_data = lindblad_mpo_results['obs-1q'][(component_lower, (i,))][1]
#         ax.plot(t, mpo_data, label=f'MPO ⟨σ{component_lower}⟩ Spin {i+1}', linestyle='--')
    
#     ax.set_xlabel('Time')
#     ax.set_ylabel(f'⟨σ{component.lower()}⟩')
#     ax.set_title(f'{component} components')
#     ax.legend()
#     ax.grid(True)

# plt.tight_layout()
# plt.show()

# print("Final expectation values (Exact Lindblad):")
# for i in range(L):
#     print(f"Spin {i+1}: ⟨σx⟩ = {result_lindblad.expect[i][-1]:.4f}, "
#           f"⟨σy⟩ = {result_lindblad.expect[L+i][-1]:.4f}, "
#           f"⟨σz⟩ = {result_lindblad.expect[2*L+i][-1]:.4f}")

# print("\nFinal expectation values (LindbladMPO):")
# for i in range(L):
#     print(f"Spin {i+1}: "
#           f"⟨σx⟩ = {lindblad_mpo_results['obs-1q'][('x', (i,))][1][-1]:.4f}, "
#           f"⟨σy⟩ = {lindblad_mpo_results['obs-1q'][('y', (i,))][1][-1]:.4f}, "
#           f"⟨σz⟩ = {lindblad_mpo_results['obs-1q'][('z', (i,))][1][-1]:.4f}")









#endregion

''' Heisenberg Model + Noise QuTip and Lindblad MPO simulation'''

# Parameters
N = 3  # number of sites
J = 1    # X and Y coupling strength
D_param = J_z = -0.8  # Z coupling strength
h = -0.1  # transverse field strength
gamma_dephasing = 1 / 10.0  # dephasing rate (1/T2star)
gamma_relaxation = 1 / 10.0  # relaxation rate (1/T1)

# Time vector
T = 10
timesteps = 200
t = np.linspace(0, T, timesteps+1)



'''Qutip exact solution'''


#region


# Define Pauli matrices
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()
sm = qt.sigmam()

# Construct the Hamiltonian (Heisenberg XXZ model)
H = 0
for i in range(N - 1):
    H += -J * qt.tensor([sx if n==i or n==i+1 else qt.qeye(2) for n in range(N)])
    H += -J * qt.tensor([sy if n==i or n==i+1 else qt.qeye(2) for n in range(N)])
    H += -D_param * qt.tensor([sz if n==i or n==i+1 else qt.qeye(2) for n in range(N)])
for i in range(N):
    H += -h * qt.tensor([sz if n == i else qt.qeye(2) for n in range(N)])

# Construct collapse operators
c_ops = []
# Dephasing operators
for i in range(N):
    c_ops.append(np.sqrt(gamma_dephasing) * qt.tensor([sz if n == i else qt.qeye(2) for n in range(N)]))
# Relaxation operators
for i in range(N):
    c_ops.append(np.sqrt(gamma_relaxation) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(N)]))

# Initial state (first spin down, rest up)
psi0 = qt.tensor([qt.basis(2, 1)] + [qt.basis(2, 0) for _ in range(N-1)])
# # Initial state (all spins up)
# psi0 = qt.tensor([qt.basis(2, 0) for _ in range(N)])
# # Initial state (up,down,up,down...)
# psi0 = qt.tensor([qt.basis(2, i % 2) for i in range(N)])

# Define measurement operators
sx_list = [qt.tensor([sx if n == i else qt.qeye(2) for n in range(N)]) for i in range(N)]
sy_list = [qt.tensor([sy if n == i else qt.qeye(2) for n in range(N)]) for i in range(N)]
sz_list = [qt.tensor([sz if n == i else qt.qeye(2) for n in range(N)]) for i in range(N)]
e_ops = sx_list + sy_list + sz_list
# Exact Lindblad solution
result_lindblad = qt.mesolve(H, psi0, t, c_ops, e_ops, progress_bar=True)

print('result_lindblad.expect len',len(result_lindblad.expect))
print('result_lindblad.expect lists len',len(result_lindblad.expect[0]))
print('result_lindblad.expect',result_lindblad.expect)
print('result_lindblad.times',result_lindblad.times)




'''Dynamic TJM setup'''

#region

L = N

# TN-MCWF setup
def create_D(L):
    return [1] + [2] * (L - 1) + [1]

D = create_D(N)
mpoH = ptn.heisenberg_xxz_mpo(N, J=-J, D=-D_param, h=h)
mpoH.zero_qnumbers()

# # Initialize the MPS all zero state
# D_mps = [1] + [2] * (N - 1) + [1]
# psi = ptn.MPS(mpoH.qd, [Di * [0] for Di in D_mps], fill=0.0)
# psi.A[0][0, 0, 0] = 1.0  # Set the first element to 1 to represent |0000...>
# psi.orthonormalize(mode='left')

# Initialize MPS for state |100...0⟩ (first spin down, rest up)
d = 2  # local dimension for qubits
D_mps = [1] + [2] * (N - 1) + [1]
psi = ptn.MPS(mpoH.qd, [Di * [0] for Di in D_mps], fill=0.0)
psi.A[0][1, 0, 0] = 1.0  # First site |1⟩ (down)
for i in range(1, L):
    psi.A[i][0, 0, 0] = 1.0  # Rest of sites |0⟩ (up)

# Orthonormalize the MPS
psi.orthonormalize(mode='left')


# Create system for Dynamic TN MCWF
model = 'Heisenberg'
d = 2
max_bond_dimension = 32
freq = 1.0
T1 = 1 / gamma_relaxation
T2star = 1 / gamma_dephasing
temperature = 0.1
processes = ['relaxation', 'dephasing']
model_params = {'J': J, 'J_z': J_z, 'g': h}

system = create_system(model, d, N, max_bond_dimension, freq, T1, T2star, 
                       temperature, processes, model_params, 
                       initial_state='+Z', calculate_exact=False, 
                       calculate_state_vector=False)




print('System initialized for TN-MCWF simulation')

# Define local operator (Pauli Z) and operator site
local_operator = create_pauli_z(d)
operator = local_operator
operator_site = 1  # Measure the first spin
dt = T/timesteps

# Run the TN-MCWF simulation
num_trajectories = 10  # Adjust as needed
stochastic_exp_values, times_sampled, _ = TN_MCWF_DynamicTDVP(
    psi, mpoH, system, num_trajectories=num_trajectories, T=T, dt=dt, max_bond_dim=max_bond_dimension, 
    operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None
)

print("All trajectories calculated and saved.")

#endregion


'''Plot code'''

# # Observable names for easier labeling
# observables = ['x', 'y', 'z']

observables = ['z']

# Define a threshold for determining if values are close to zero (to handle small numerical noise)
threshold = 1e-10  # Adjust this if needed

# # Check for constant zero lists in result_lindblad.expect
# print("Constant zero lists in result_lindblad.expect:")
# for j in range(len(result_lindblad.expect)):
#     observable_name = observables[j // 3]  # Choose 'x' for 0-2, 'y' for 3-5, 'z' for 6-8
#     if np.all(np.abs(result_lindblad.expect[j]) < threshold):
#         print(f"Exact Lindblad Spin {j % 3}, observable {observable_name} is constant zero")

# # Check for constant zero lists in exp_vals_lindbladmpo
# print("\nConstant zero lists in exp_vals_lindbladmpo:")
# for i in range(len(exp_vals_lindbladmpo)):
#     observable_name = observables[i // 3]  # Choose 'x' for 0-2, 'y' for 3-5, 'z' for 6-8
#     if np.all(np.abs(exp_vals_lindbladmpo[i]) < threshold):
#         print(f"Lindblad MPO Spin {i % 3}, observable {observable_name} is constant zero")




plt.figure(figsize=(12,10))

# # plot differences
# for j in range(len(e_ops)):
#     plt.plot(result_lindblad.times, result_lindblad.expect[j]-exp_vals_lindbladmpo[j], label=f'difference, observable {j}')


for i in range(len(z_expectation_values_mpo)):  
    plt.plot(times_sampled, z_expectation_values_mpo[i], label = f'z observable qubit {i}')

# # Plot Exact Lindblad results
# for j in range(len(e_ops)):
#     observable_name = observables[j // 3]  # Choose 'x' for 0-2, 'y' for 3-5, 'z' for 6-8
#     plt.plot(result_lindblad.times, result_lindblad.expect[j], 
#              label=f'Exact Lindblad Spin {j % 3}, observable {observable_name}')

# # Plot MPO results
# for i in range(len(exp_vals_lindbladmpo)):
#     observable_name = observables[i // 3]  # Choose 'x' for 0-2, 'y' for 3-5, 'z' for 6-8
#     plt.plot(result_lindblad.times, exp_vals_lindbladmpo[i], 
#              label=f'Lindblad MPO Spin {i % 3}, observable {observable_name}')
plt.plot(times_sampled, stochastic_exp_values, label = 'TJM')

plt.title('Heisenberg XXZ')
plt.xlabel('time')
plt.ylabel('Expectation values')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
