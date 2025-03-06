import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import matplotlib.ticker as ticker

from yaqs.core.data_structures.networks import MPO, MPS
from yaqs.core.data_structures.noise_model import NoiseModel
from yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams

from yaqs import Simulator


import time

import importlib
import yaqs

from yaqs.noise_char.optimization import *
# from yaqs.noise_char.propagation import *
from yaqs.noise_char.analytical_gradient_tjm import *

importlib.reload(yaqs.noise_char.optimization)
importlib.reload(yaqs.noise_char.propagation)





def BFGS_char(state, H_0, sim_params, noise_model, ref_traj, traj_der, learning_rate=0.01, max_iterations=200, tolerance=1e-8):
    """
    Parameters:
    sim_params (object): Simulation parameters containing gamma_rel and gamma_deph.
    ref_traj (array-like): Reference trajectory data.
    traj_der (function): Function that runs the simulation and returns the time, 
                         expected values trajectory, and derivatives of the observables 
                         with respect to the noise parameters.
    learning_rate (float, optional): Learning rate for the BFGS optimizer. Default is 0.01.
    max_iterations (int, optional): Maximum number of iterations for the optimization. Default is 200.
    tolerance (float, optional): Tolerance for the convergence criterion. Default is 1e-8.
    Returns:
    tuple: A tuple containing:
        - loss_history (list): History of loss values during optimization.
        - gr_history (list): History of gamma_rel values during optimization.
        - gd_history (list): History of gamma_deph values during optimization.
        - dJ_dgr_history (list): History of gradients with respect to gamma_rel.
        - dJ_dgd_history (list): History of gradients with respect to gamma_deph.
    
    Performs BFGS optimization to minimize the loss function.
    """
    loss_history = []
    gr_history = []
    gd_history = []
    dJ_dgr_history = []
    dJ_dgd_history = []

    gr_history.append(noise_model.strengths[0])
    gd_history.append(noise_model.strengths[1])

    # Initial parameters
    params_old = np.array([noise_model.strengths[0], noise_model.strengths[1]])
    n_params = len(params_old)

    # Initial inverse Hessian approximation
    H_inv = np.eye(n_params)

    I = np.eye(n_params)


    # Calculate first loss and gradients
    loss, exp_vals_traj, grad_old = loss_function_char(state, H_0, sim_params, noise_model, ref_traj, traj_der)
    loss_history.append(loss)



    for iteration in range(max_iterations):

        # Store current parameters and gradients
        # params_old = params.copy()
        # grad_old = dJ_dg.copy()

        # Update parameters
        params_new = params_old - learning_rate * H_inv.dot(grad_old)

        for i in range(n_params):
            if params_new[i] < 0:
                params_new[i] = 0

        # Update simulation parameters
        sim_params.gamma_rel, sim_params.gamma_deph = params_new

        # Calculate new loss and gradients
        loss, exp_vals_traj, grad_new = loss_function_char(state, H_0, sim_params, noise_model, ref_traj, traj_der)
        loss_history.append(loss)

        if loss < tolerance:
            print(f"Converged after {iteration} iterations.")
            break


        # Compute differences
        s = params_new - params_old
        y = grad_new - grad_old

        # Update inverse Hessian approximation using BFGS formula
        rho = 1.0 / (y.dot(s))

        H_inv = (I - rho * np.outer(s, y)).dot(H_inv).dot(I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        # Log history
        dJ_dgr_history.append(grad_new[0])
        dJ_dgd_history.append(grad_new[1])
        gr_history.append(noise_model.strengths[0])
        gd_history.append(noise_model.strengths[1])


        params_old = params_new 
        grad_old = grad_new

        print(f"Iteration {iteration}: Loss = {loss}")

    return loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history


def loss_function_char(state, H_0, sim_params, noise_model, ref_traj, traj_der):
    """
    Compute the loss function and its gradients for the given simulation parameters.
    Parameters:
    sim_params (dict): Dictionary containing the simulation parameters.
    ref_traj (list): List of reference trajectories for comparison.
    traj_der (function): Function that runs the simulation and returns the time, 
                         expected values trajectory, and derivatives of the observables 
                         with respect to the noise parameters.
    Returns:
    tuple: A tuple containing:
        - loss (float): The computed loss value.
        - exp_vals_traj (list): The expected values trajectory from the TJM simulation.
        - gradients (numpy.ndarray): Array containing the gradients of the loss with respect 
                                     to gamma_relaxation and gamma_dephasing.
    """
    
    
    # Run the TJM simulation with the given noise parameters

    start_time = time.time()
   
    traj_der(state, H_0, sim_params, noise_model) 

    t = sim_params.times 
    exp_vals_traj = []
    for observable in sim_params.observables:
        exp_vals_traj.append(observable.results)
    d_On_d_gk = sim_params.d_On_d_gk
    
   
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

if __name__ == '__main__':



    L = 4
    d = 2
    J = 1
    g = 0.5
    H_0 = MPO()
    H_0.init_Ising(L, d, J, g)

    # Define the initial state
    state = MPS(L, state='zeros')

    # Define the noise model
    gamma = 0.1
    noise_model = NoiseModel(['relaxation', 'dephasing'], [0.2, 0.01])
   

    # Define the simulation parameters
    T = 5
    dt = 0.1
    sample_timesteps = True
    N = 500
    max_bond_dim = 4
    threshold = 1e-6
    order = 1
    measurements = [Observable('x', site) for site in range(L)]  + [Observable('y', site) for site in range(L)]  + [Observable('z', site) for site in range(L)]
    initial_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)


    '''QUTIP calculation'''

    qt_params = SimulationParameters()

    qt_params.T = T
    qt_params.dt = dt
    qt_params.L = L
    qt_params.J = J
    qt_params.g = g
    qt_params.gamma_rel = gamma
    qt_params.gamma_deph = gamma
    qt_params.observables = ['x','y', 'z']


    # Generate reference trajectory
    sim_params = SimulationParameters()

    # t, qt_ref_traj,dO, A_kn_exp_vals=qutip_traj(sim_params)
    t, qt_ref_traj,dO = qutip_traj_char(qt_params)



    loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history = BFGS_char(state, H_0, initial_params, noise_model, qt_ref_traj, run_char, learning_rate=0.2, max_iterations=10,tolerance=1e-8)



    plt.plot(np.log(loss_history), label='log(J)')
    plt.legend()


    def exp_formatter(x, pos):
        return f"{np.exp(x):.2e}"  

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(exp_formatter))
    plt.ylabel('Loss J (exponentiated)')

    plt.show()


