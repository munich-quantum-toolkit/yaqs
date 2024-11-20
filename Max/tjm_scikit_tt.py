import numpy as np
import scikit_tt.tensor_train as tt
from scikit_tt.tensor_train import TT
import scikit_tt.solvers.ode as ode
import scipy.linalg as lin
import matplotlib.pyplot as plt
import qutip as qt





def tjm(hamiltonian, jump_operator_list, jump_parameter_list, initial_state, time_step, number_of_steps, max_rank):
    """
    Tensor Jump Method (TJM)

    Parameters
    ----------
    hamiltonian : TT
        Hamiltonian of the system
    jump_operator_list : list[list[np.ndarray]] or list[np.ndarray]
        list of jump operators for each dimension; can be either of the form [[K_1,1 ,...], ..., [K_L,1, ...]], where 
        each sublist contains the jump operators for one specific dimension or of the form [K_1, ..., K_m] if the same 
        set of jump operators is applied to every dimension
    jump_parameter_list : list[list[np.ndarray]] or list[np.ndarray]
        prefactors for the jump operators; the form of this list corresponds to jump_operator_list
    initial_state : TT
        initial state for the simulation
    time_step : float
        time step for the simulation
    number_of_steps : int
        number of time steps

    Returns
    -------
    trajectory : list[TT]
        trajectory of computed states
    """
    
    L = hamiltonian.order
    trajectory = []
    state = initial_state.copy()
    trajectory.append(state)

    # construct dissipative rank-one operators
    diss_op_half = tjm_dissipative_operator(L, jump_operator_list, jump_parameter_list, time_step/2)
    diss_op_full = tjm_dissipative_operator(L, jump_operator_list, jump_parameter_list, time_step)
    
    # begin of loop
    state = diss_op_half@state
    state = tjm_jump_process_tdvp(hamiltonian, state, jump_operator_list, jump_parameter_list, time_step, max_rank, 0)
    trajectory.append(state.copy())
    for k in range(1, number_of_steps-1):
 
        state = diss_op_full@state
        state = tjm_jump_process_tdvp(hamiltonian, state, jump_operator_list, jump_parameter_list, time_step, max_rank, k)
        trajectory.append(state.copy())
    state = diss_op_half@state
    state = (1/state.norm())*state
    trajectory.append(state.copy())

    return trajectory




def tjm_jump_process_tdvp(hamiltonian, state, jump_operator_list, jump_parameter_list, time_step, max_rank, time):
    """
    Apply jump process of the Tensor Jump Method (TJM)

    Parameters
    ----------
    hamiltonian : TT
        Hamiltonian of the system
    state : TT
        current state of the simulation
    jump_operator_list : list[list[np.ndarray]] or list[np.ndarray]
        list of jump operators for each dimension; can be either of the form [[K_1,1 ,...], ..., [K_L,1, ...]], where 
        each sublist contains the jump operators for one specific dimension or of the form [K_1, ..., K_m] if the same 
        set of jump operators is applied to every dimension
    jump_parameter_list : list[list[np.ndarray]] or list[np.ndarray]
        prefactors for the jump operators; the form of this list corresponds to jump_operator_list
    time_step : float
        time step for the simulation

    Returns
    -------
    state_evolved : TT
        evolved state after jump process (either by means of TDVP or randomly applied jump operator)
    """
    
    L = state.order

    # create 2d lists if inputs are 1d (e.g. same set of jump operators for each set)
    if isinstance(jump_operator_list[0], list)==False:
        jump_operator_list_org = jump_operator_list.copy()
        jump_operator_list = [jump_operator_list_org.copy() for _ in range(L)]
    if isinstance(jump_parameter_list[0], list)==False:
        jump_parameter_list_org = jump_parameter_list.copy()
        jump_parameter_list = [jump_parameter_list_org.copy() for _ in range(L)]

    # copy initial state
    state_org = state.copy()    
    state = state.ortho_right()

    if max(state.ranks) < max_rank:
        # time evolution by 2TDVP
        state_evolved = ode.tdvp2site(hamiltonian, state, time_step, 1, threshold=0, max_rank=max_rank)[-1]
    else:
        # time evolution by TDVP
        state_evolved = ode.tdvp(hamiltonian, state, time_step, 1)[-1]

    # probability for jump process
    dp = 1-np.linalg.norm(state_evolved.cores[0].flatten())**2

    # draw random epsilon
    epsilon = np.random.rand()

    if dp > epsilon: 

        print('jump in timestep:', time)

        # initialize jump probabilites
        prob_list = []
        for i in range(len(jump_operator_list)):
            prob_list += [[None for _ in range(len(jump_operator_list[i]))]]

        # index list for application of jump operator
        index_list = []

        # compute probabilities
        for i in range(L):
            for j in range(len(prob_list[i])):
                index_list += [[i,j]]
                prob_list[i][j] = state.cores[i].copy()
                prob_list[i][j] = np.tensordot(jump_operator_list[i][j].copy(), prob_list[i][j], axes=(1,1))
                prob_list[i][j] = time_step*jump_parameter_list[i][j]*np.linalg.norm(prob_list[i][j])**2
            if i<len(prob_list)-1:
                state = state.ortho_left(start_index=i, end_index=i)

        # draw index according to computed distribution and apply jump operator
        distribution = np.hstack(prob_list)
        distribution *= 1/np.sum(distribution)
        sample = np.random.choice(len(index_list), p=distribution)
        index = index_list[sample]
        operator = jump_operator_list[index[0]][index[1]]
        state_evolved = state_org
        state_evolved.cores[index[0]] = np.tensordot(np.sqrt(jump_parameter_list[index[0]][index[1]])*jump_operator_list[index[0]][index[1]], state_evolved.cores[index[0]], axes=(1,1))

    # normalize state
    state_evolved = state_evolved.ortho_right()
    norm = np.linalg.norm(state_evolved.cores[0].flatten())
    state_evolved = (1/norm)*state_evolved

    return state_evolved




def tjm_dissipative_operator(L, jump_operator_list, jump_parameter_list, time_step):
    """
    Construct rank-one tensor operator for the dissipative step of the tensor jump method.

    Parameters
    ----------
    L : int
        system size, e.g., number of qubits
    jump_operator_list : list[list[np.ndarray]] or list[np.ndarray]
        list of jump operators for each dimension; can be either of the form [[K_1,1 ,...], ..., [K_L,1, ...]], where 
        each sublist contains the jump operators for one specific dimension or of the form [K_1, ..., K_m] if the same 
        set of jump operators is applied to every dimension
    jump_parameter_list : list[list[np.ndarray]] or list[np.ndarray]
        prefactors for the jump operators; the form of this list corresponds to jump_operator_list
    time_step : float
        time step for the simulation

    Returns
    -------
    op : TT
        dissipative rank-one operator
    """

    # create 2d lists if inputs are 1d (e.g. same set of jump operators for each set)
    if isinstance(jump_operator_list[0], list)==False:
        jump_operator_list_org = jump_operator_list.copy()
        jump_operator_list = [jump_operator_list_org.copy() for _ in range(L)]
    if isinstance(jump_parameter_list[0], list)==False:
        jump_parameter_list_org = jump_parameter_list.copy()
        jump_parameter_list = [jump_parameter_list_org.copy() for _ in range(L)]
    
    # construct dissipative exponential
    cores = [None]*L
    for i in range(L):
        cores[i] = np.zeros([2,2])
        for j in range(len(jump_operator_list[i])):
            cores[i] += jump_parameter_list[i][j]*jump_operator_list[i][j].conj().T@jump_operator_list[i][j]
        cores[i] = lin.expm(-0.5*time_step*cores[i])[None, :, :, None]
    op = TT(cores)
    return op




'''Test routine: '''



# Parameters
N = 10  # number of sites
J = 1.0  # Ising coupling strength
h = g = 0.5  # transverse field strength
gamma_dephasing = 0 # 1 / 1.0  # dephasing rate (1/T2star)
gamma_relaxation = 0 #1 / 1.0  # relaxation rate (1/T1)

# Define Pauli matrices
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

# Construct the Ising Hamiltonian
H = 0
for i in range(N-1):
    H += -J * qt.tensor([sz if n==i or n==i+1 else qt.qeye(2) for n in range(N)])
for i in range(N):
    H += -h * qt.tensor([sx if n==i else qt.qeye(2) for n in range(N)])

# Construct collapse operators
c_ops = []

# Dephasing operators
for i in range(N):
    c_ops.append(np.sqrt(gamma_dephasing) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(N)]))

# Relaxation operators
for i in range(N):
    c_ops.append(np.sqrt(gamma_relaxation) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(N)]))





# Initial state (all spins up)
psi0 = qt.tensor([qt.basis(2, 0) for _ in range(N)])

# # Time vector
T = 1
timesteps = 10
t = np.linspace(0, T, timesteps+1)

print(t)



# Define Z measurement operator for the fifth qubit
sz_fifth = qt.tensor([sz if n==4 else qt.qeye(2) for n in range(N)])



#Exact Lindblad solution
result_lindblad = qt.mesolve(H, psi0, t, c_ops, [sz_fifth], progress_bar=True)

print(result_lindblad.expect[0][-1])






'''start of scikit initialization'''

# chain length
L = 10

# construct Hamiltonian (Ising model)
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)
g = 0.5
J = 1
cores = [None] * L
cores[0] = tt.build_core([[-g * X, - J * Z, I]])
for i in range(1, L - 1):
    cores[i] = tt.build_core([[I, 0, 0], [Z, 0, 0], [-g * X, - J * Z, I]])
cores[-1] = tt.build_core([-g * X, - J * Z, I])
hamiltonian = TT(cores)



# jump operators and parameters
L_1 = np.array([[0, 1], [0, 0]])
L_2 = np.array([[1, 0], [0, -1]])
jump_operator_list = [[L_1, L_2] for _ in range(L)]
jump_parameter_list = [[0, 0] for _ in range(L)]

# initial state
rank = 10
initial_state = tt.unit([2] * L, [0] * L)
for i in range(rank - 1):
    initial_state += tt.unit([2] * L, [0] * L)
initial_state = initial_state.ortho()
initial_state = (1 / initial_state.norm()) * initial_state
print(initial_state.cores)






# time step, number of steps, operator_site and max rank
time_step = 0.1
number_of_steps = 10
max_rank = 16
operator_site = 4
num_trajectories = 100

# observable

observable = tt.eye(dims=[2]*L)
observable.cores[operator_site]=tt.build_core([Z])

exp_vals = []
exp_vals.append(initial_state.transpose(conjugate=True)@observable@initial_state)

# for i in range(timesteps):
#     initial_state = ode.tdvp(hamiltonian, initial_state, time_step, 1)[-1]
#     exp_vals.append(initial_state.transpose(conjugate=True)@observable@initial_state)

for traj in range(num_trajectories):
    print(traj)

    # apply Tensor Jump Method (TJM)
    trajectory = tjm(hamiltonian, jump_operator_list, jump_parameter_list, initial_state, time_step, number_of_steps, max_rank)

    exp_vals_traj = []

    for state in trajectory: 
        state_adjusted = observable @ state
        state_adjusted =  state_adjusted.ortho_right()
        state = state.ortho_right()
        exp_vals_traj.append(state.transpose(conjugate=True)@state_adjusted)
    exp_vals.append(exp_vals_traj)



# Calculate average exp value from trajectories
exp_vals_array = np.array(exp_vals)
mean_values = np.mean(exp_vals_array, axis=0)

# Extract real parts only
mean_values_real = np.real(mean_values)
mean_values_list = mean_values_real.tolist()

print(mean_values_list)

# plot QuTip solution against TJM
plt.figure(figsize=(10, 8))
plt.plot(t, result_lindblad.expect[0], label="exact lindblad QuTip", marker='o', linestyle='-', color='b')  
plt.plot(t, exp_vals, label="scikit-tt TDVP", marker='s', linestyle='--', color='r')  
plt.xlabel("Timesteps")
plt.ylabel("YExp val. z-obs on 5th qubit")
plt.title("Lindblad simulation noisy Ising L=10")
plt.legend()
plt.show()
