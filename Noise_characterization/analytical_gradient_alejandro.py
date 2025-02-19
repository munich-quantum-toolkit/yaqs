
#%%
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from yaqs.core.data_structures.networks import MPO, MPS
from yaqs.core.data_structures.noise_model import NoiseModel
from yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams

from yaqs import Simulator
from dataclasses import dataclass

import time



#%%


@dataclass
class SimulationParameters:
    T: float = 1
    dt: float = 0.1
    L: int = 4
    J: float = 1
    g: float = 0.5
    gamma_rel: float = 0.1
    gamma_deph: float = 0.1



#%%

# t = np.arange(0, 1 + 0.1, 0.1)

# trapezoidal(A_kn_exp_vals[0][0],t) 

#%%


def trapezoidal(y, x):

    integral = np.zeros(len(y))

    integral[0] = 0

    for i in range(1,len(y)):
        integral[i] = integral[i-1] + 0.5*(x[i] - x[i-1])*(y[i] + y[i-1])

    return integral

    

def qutip_traj(sim_params_class: SimulationParameters):

    T = sim_params_class.T
    dt = sim_params_class.dt
    L = sim_params_class.L
    J = sim_params_class.J
    g = sim_params_class.g
    gamma_rel = sim_params_class.gamma_rel
    gamma_deph = sim_params_class.gamma_deph


    t = np.arange(0, T + dt, dt) 

    '''QUTIP Initialization + Simulation'''

    # Define Pauli matrices
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()

    # Construct the Ising Hamiltonian
    H = 0
    for i in range(L-1):
        H += -J * qt.tensor([sz if n==i or n==i+1 else qt.qeye(2) for n in range(L)])
    for i in range(L):
        H += -g * qt.tensor([sx if n==i else qt.qeye(2) for n in range(L)])



    # Construct collapse operators
    c_ops = []
    gammas = []

    # Relaxation operators
    for i in range(L):
        c_ops.append(np.sqrt(gamma_rel) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(L)]))
        gammas.append(gamma_rel)

    # Dephasing operators
    for i in range(L):
        c_ops.append(np.sqrt(gamma_deph) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(L)]))
        gammas.append(gamma_deph)

    # Initial state
    psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])

    # Define measurement operators
    sx_list = [qt.tensor([sx if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]
    sy_list = [qt.tensor([sy if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]
    sz_list = [qt.tensor([sz if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]

    obs_list = sx_list  + sy_list + sz_list


    # Create new set of observables by multiplying every operator in obs_list with every operator in c_ops
    A_kn_list= []
    for i,c_op in enumerate(c_ops):
        for obs in obs_list:
            A_kn_list.append(  (1/gammas[i]) * (c_op.dag()*obs*c_op  -  0.5*obs*c_op.dag()*c_op  -  0.5*c_op.dag()*c_op*obs)   )



    new_obs_list = obs_list + A_kn_list




    n_obs= len(obs_list)
    n_jump= len(c_ops)


    # Exact Lindblad solution
    result_lindblad = qt.mesolve(H, psi0, t, c_ops, new_obs_list, progress_bar=True)

    exp_vals = []
    for i in range(len(new_obs_list)):
        exp_vals.append(result_lindblad.expect[i])
    

    # Separate original and new expectation values
    original_exp_vals = exp_vals[:n_obs]
    new_exp_vals = exp_vals[n_obs:]

    # Reshape new_exp_vals to be a list of lists with dimensions n_jump times n_obs
    A_kn_exp_vals = [new_exp_vals[i * n_obs:(i + 1) * n_obs] for i in range(n_jump)]
    
    # Compute the integral of the new expectation values to obtain the derivatives
    d_On_d_gk = [ [trapezoidal(A_kn_exp_vals[i][j],t)  for j in range(n_obs)] for i in range(n_jump) ]


    return t, original_exp_vals, d_On_d_gk
    



def tjm(sim_params_class: SimulationParameters, N=1000):

    T = sim_params_class.T
    dt = sim_params_class.dt
    L = sim_params_class.L
    J = sim_params_class.J
    g = sim_params_class.g
    gamma_rel = sim_params_class.gamma_rel
    gamma_deph = sim_params_class.gamma_deph


    t = np.arange(0, T + dt, dt) 


    # Define the system Hamiltonian
    d = 2
    H_0 = MPO()
    H_0.init_Ising(L, d, J, g)
    # Define the initial state
    state = MPS(L, state='zeros')

    # Define the noise model
    # gamma_relaxation = noise_params[0]
    # gamma_dephasing = noise_params[1]
    noise_model = NoiseModel(['relaxation', 'dephasing'], [gamma_rel, gamma_deph])

    sample_timesteps = True
    # N = 10
    threshold = 1e-6
    max_bond_dim = 4
    order = 2
    measurements = [Observable('x', site) for site in range(L)]  + [Observable('y', site) for site in range(L)] + [Observable('z', site) for site in range(L)]

    sim_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)
    Simulator.run(state, H_0, sim_params, noise_model)

    tjm_exp_vals = []
    for observable in sim_params.observables:
        tjm_exp_vals.append(observable.results)
        # print(f"Observable at site {observable.site}: {observable.results}")
    # print(tjm_exp_vals)


    return t, tjm_exp_vals



def loss_function(sim_params, ref_traj):
    """
    Calculates the squared distance between corresponding entries of QuTiP and TJM expectation values.

    Args:
        noise_params (list): Noise parameters for the TJM simulation.
        qt_exp_vals (list of arrays): QuTiP expectation values for each site.

    Returns:
        float: The total squared loss.
    """
    
    # Run the TJM simulation with the given noise parameters

    start_time = time.time()
    t, exp_vals_traj, d_On_d_gk = qutip_traj(sim_params)  
    end_time = time.time()
    tjm_time = end_time - start_time
    # print(f"TJM time -> {tjm_time:.4f}")
    
    # Initialize loss
    loss = 0.0
    
    # Ensure both lists have the same structure
    if len(ref_traj) != len(exp_vals_traj):
        raise ValueError("Mismatch in the number of sites between qt_exp_vals and tjm_exp_vals.")

    # Compute squared distance for each site
    for ref_vals, tjm_vals in zip(ref_traj, exp_vals_traj):
        loss += np.sum((np.array(ref_vals) - np.array(tjm_vals)) ** 2)
    

    n_jump = len(d_On_d_gk)
    n_obs = len(d_On_d_gk[0])
    n_t = len(d_On_d_gk[0][0])

    n_gr = n_jump//2


    dJ_d_gr = 0
    dJ_d_gd = 0


    for i in range(n_obs):
        for j in range(n_t):
            # I have to add all the derivatives with respect to the same gamma_relaxation and gamma_dephasing
            for k in range(n_gr):
                # The initial half of the jump operators are relaxation operators
                dJ_d_gr += 2*(exp_vals_traj[i][j] - ref_traj[i][j]) * d_On_d_gk[k][i][j]
                # The second half of the jump operators are dephasing operators
                dJ_d_gd += 2*(exp_vals_traj[i][j] - ref_traj[i][j]) * d_On_d_gk[n_gr + k][i][j]




    return loss, exp_vals_traj, np.array([dJ_d_gr, dJ_d_gd])





def gradient_descent(sim_params, ref_traj, learning_rate=0.01, max_iterations=100, tolerance=1e-10):
    """
    Performs gradient descent to minimize the loss function.

    Args:
        sim_params (SimulationParameters): Initial simulation parameters.
        ref_traj (list): Reference trajectory for comparison.
        learning_rate (float): Learning rate for gradient descent.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Tolerance for convergence.

    Returns:
        SimulationParameters: Optimized simulation parameters.
        list: Loss history.
    """
    loss_history = []

    gr_history = []
    gd_history = []

    gr_history.append(sim_params.gamma_rel)
    gd_history.append(sim_params.gamma_deph)



    dJ_dgr_history = []
    dJ_dgd_history = []


    for iteration in range(max_iterations):
        # Calculate loss and gradients
        loss, exp_vals_traj, dJ_dg = loss_function(sim_params, ref_traj)
        loss_history.append(loss)

        # Check for convergence
        if loss < tolerance:
            print(f"Converged after {iteration} iterations.")
            break

        
        sim_params.gamma_rel -= learning_rate * dJ_dg[0]
        sim_params.gamma_deph -= learning_rate * dJ_dg[1]


        dJ_dgr_history.append(dJ_dg[0])

        dJ_dgd_history.append(dJ_dg[1])
 



        gr_history.append(sim_params.gamma_rel)
        gd_history.append(sim_params.gamma_deph)
        

        print(f"!!!!!!! Iteration {iteration}: Loss = {loss}")

    return loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history



# --- ADAM GRADIENT DESCENT (Modified) ---
def ADAM_gradient_descent(sim_params, ref_traj, learning_rate=0.01, max_iterations=100, tolerance=1e-6):
    """
    Performs Adam gradient descent to minimize the loss function.
    
    Changes made:
    - Added state variables m and v for the first and second moments.
    - Introduced hyperparameters beta1, beta2, and epsilon.
    - Updated parameters using the Adam update rule with bias correction.
    """
    loss_history = []
    gr_history = []
    gd_history = []
    dJ_dgr_history = []
    dJ_dgd_history = []

    gr_history.append(sim_params.gamma_rel)
    gd_history.append(sim_params.gamma_deph)

    # Adam hyperparameters and initialization (NEW)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    m = np.array([0.0, 0.0])  # First moment (for [gamma_rel, gamma_deph])
    v = np.array([0.0, 0.0])  # Second moment (for [gamma_rel, gamma_deph])

    for iteration in range(max_iterations):
        # Calculate loss and gradients (unchanged)
        loss, exp_vals_traj, dJ_dg = loss_function(sim_params, ref_traj)
        loss_history.append(loss)

        if loss < tolerance:
            print(f"Converged after {iteration} iterations.")
            break
        
        # Adam update steps (NEW)
        m = beta1 * m + (1 - beta1) * dJ_dg
        v = beta2 * v + (1 - beta2) * (dJ_dg ** 2)
        m_hat = m / (1 - beta1 ** (iteration + 1))
        v_hat = v / (1 - beta2 ** (iteration + 1))
        update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Update simulation parameters with Adam update (NEW)
        sim_params.gamma_rel -= update[0]
        sim_params.gamma_deph -= update[1]

        # Log gradient updates
        dJ_dgr_history.append(dJ_dg[0])
        dJ_dgd_history.append(dJ_dg[1])
        
        gr_history.append(sim_params.gamma_rel)
        gd_history.append(sim_params.gamma_deph)
        
        print(f"!!!!!!! Iteration {iteration}: Loss = {loss}")

    return loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history




#%%

# Generate reference trajectory
sim_params = SimulationParameters()

t, qt_ref_traj, A_kn_exp_vals=qutip_traj(sim_params)


#%%

# Perform gradient descent

initial_params = SimulationParameters()
initial_params.gamma_rel = 0.15
initial_params.gamma_deph = 0.13


loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history = ADAM_gradient_descent(initial_params, qt_ref_traj, learning_rate=0.1, max_iterations=400)


# Plot 1: Logarithm of the loss history
plt.figure()
plt.plot(np.log(loss_history), label='log(J)')
plt.legend()
plt.grid(True)  # Add grid to the plot
plt.show()  # Display the first plot

# Plot 2: Gamma relaxation and dephasing with a reference line
plt.figure()
plt.plot(gr_history, label='gamma_relaxation')
plt.plot(gd_history, label='gamma_dephasing')
plt.axhline(y=0.1, color='r', linestyle='--', label='gamma_reference')
plt.legend()
plt.grid(True)  # Add grid to the plot
plt.show()  # Display the second plot


# %%

# %%
