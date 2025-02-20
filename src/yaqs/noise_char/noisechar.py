import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import random
import time
import sys
import os

from yaqs.core.data_structures.networks import MPO
from yaqs.core.data_structures.networks import MPS
from yaqs.core.data_structures.noise_model import NoiseModel
from yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams
from yaqs.physics import PhysicsTJM
from yaqs import Simulator



'''TODO: 

Maybe tjm should not be called inside loss function.
Check Learning rate optimization and epsilon in gradient approximation
and try to learn parameters with qutip as comparison

Try KL divergence as loss function
'''


'''run code via:
PYTHONPATH=$(pwd)/src python3 examples/4_Nois
e_characterization.py '''








def tjm(noise_params, L, J, g):

    # Define the system Hamiltonian
    d = 2
    H_0 = MPO()
    H_0.init_Ising(L, d, J, g)
    # Define the initial state
    state = MPS(L, state='zeros')

    # Define the noise model
    gamma_relaxation = noise_params[0]
    gamma_dephasing = noise_params[1]
    noise_model = NoiseModel(['relaxation', 'dephasing'], [gamma_relaxation, gamma_dephasing])

    T = 0.5
    dt = 0.1
    sample_timesteps = True
    N = 1000
    threshold = 1e-6
    max_bond_dim = 4
    order = 2
    measurements = [Observable('x', site) for site in range(L)] + [Observable('y', site) for site in range(L)] + [Observable('z', site) for site in range(L)]
    
    sim_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)
    Simulator.run(state, H_0, sim_params, noise_model)

    tjm_exp_vals = []
    for observable in sim_params.observables:
        tjm_exp_vals.append(observable.results)
        # print(f"Observable at site {observable.site}: {observable.results}")
    # print(tjm_exp_vals)


    return tjm_exp_vals, sim_params

def loss_function(noise_params, real_exp_vals,L,J,g):
    """
    Calculates the squared distance between corresponding entries of QuTiP and TJM expectation values.

    Args:
        noise_params (list): Noise parameters for the TJM simulation.
        real_exp_vals (list of arrays): QuTiP expectation values for each site.

    Returns:
        float: The total squared loss.
    """
    
    # Run the TJM simulation with the given noise parameters

    start_time = time.time()
    tjm_exp_vals, _ = tjm(noise_params,L,J,g)  
    end_time = time.time()
    tjm_time = end_time - start_time
    print(f"TJM time -> {tjm_time:.4f}")
    
    # Initialize loss
    loss = 0.0
    
    # Ensure both lists have the same structure
    if len(real_exp_vals) != len(tjm_exp_vals):
        raise ValueError("Mismatch in the number of sites between real_exp_vals and tjm_exp_vals.")

    # Compute squared distance for each site
    for real_vals, tjm_vals in zip(real_exp_vals, tjm_exp_vals):
        loss += np.sum((np.array(real_vals) - np.array(tjm_vals)) ** 2)
    
    return loss, tjm_exp_vals

def compute_gradient(base_loss, loss_function, noise_params, real_exp_vals, L,J,g, epsilon, gradient_style ='full'):
    """
    Approximates the gradient of the loss function with respect to noise_params using finite forward differences.

    Args:
        loss_function: Function to compute the loss.
        noise_params: Current noise parameters.
        real_exp_vals: QuTiP or Lindblad MPO expectation values.
        epsilon: Small value for finite difference approximation.

    Returns:
        grad: Gradient vector with respect to noise_params.
    """
    grad = np.zeros_like(noise_params)

    if gradient_style == 'full':
        for i in range(len(noise_params)):
            # Perturb parameter i positively and negatively
            params_up = np.copy(noise_params)
            
            params_up[i] += epsilon
            
            
            # Compute loss for perturbed parameters
            loss_up,_ = loss_function(params_up, real_exp_vals, L, J, g)
            
            
            # Approximate gradient
            grad[i] = (loss_up - base_loss) / (epsilon)
            grad_norm = np.linalg.norm(grad)
            print(f'full gradient norm: {grad_norm}')
    if gradient_style == 'stochastic':

        N = len(noise_params)  # Suppose your list has length N
        index = random.randint(0, N-1)
        params_up = np.copy(noise_params)
        params_up[index] += epsilon
        # Compute loss for perturbed parameters
        loss_up,_ = loss_function(params_up, real_exp_vals, L, J, g)
            
            
        # Approximate stochastic gradient
        grad[index] = (loss_up - base_loss) / (epsilon)
        grad_norm = np.linalg.norm(grad)
        print(f'stochastic gradient norm: {grad_norm}')


    return grad

#region
# def gradient_descent(real_exp_vals, init_noise_params, L,J,g, learning_rate=0.1, epochs=100):
#     """
#     Implements stochastic gradient descent to optimize noise parameters.

#     Args:
#         real_exp_vals: QuTiP or LindbladMPO expectation values.
#         init_noise_params: Initial noise parameters.
#         learning_rate: Step size for updates.
#         epochs: Number of iterations.

#     Returns:
#         optimized_params: Optimized noise parameters.
#         loss_history: List of loss values during training.
#     """
#     # Initialize noise parameters
#     noise_params = np.copy(init_noise_params)
#     loss_history = []

#     for epoch in range(epochs):
#         # Compute the loss
#         start_time = time.time()
#         # Compute the loss
#         loss, _ = loss_function(noise_params, real_exp_vals, L, J,g)
#         # End timing
#         end_time = time.time()

#         # Record the loss calculation time
#         loss_time = end_time - start_time
#         print(f"Epoch {epoch + 1}/{epochs}, Loss Calculation Time: {loss_time:.4f} seconds")

        
#         loss_history.append(loss)

#         start_time = time.time()

#         # Compute the gradient
#         grad = compute_gradient(loss, loss_function, noise_params, real_exp_vals, L,J,g)

#         end_time = time.time()

#         gradient_time = end_time - start_time
#         print(f"Epoch {epoch +1}/ {epochs}, Gradient Calculation Time: {gradient_time:.4f} seconds")

#         # Update parameters
#         noise_params -= learning_rate * grad

#         noise_params = [max(p, 0) for p in noise_params]

#         # Print progress
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}, Params: {noise_params}")

#     return noise_params, loss_history
#endregion

def adam_optimizer_update(noise_params, grad, m, v, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Performs a single Adam update for the given parameters.

    Args:
        noise_params (list): Current noise parameters.
        grad (array): Gradient of the loss with respect to noise parameters.
        m (array): Exponential moving average of the gradients (first moment).
        v (array): Exponential moving average of the squared gradients (second moment).
        t (int): Current timestep (epoch).
        learning_rate (float): Learning rate for Adam.
        beta1 (float): Decay rate for the first moment.
        beta2 (float): Decay rate for the second moment.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        updated_params (list): Updated noise parameters.
        m (array): Updated first moment estimate.
        v (array): Updated second moment estimate.
    """
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    updated_params = noise_params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    updated_params = [max(p, 0) for p in updated_params]  # Ensure parameters stay non-negative
    return updated_params, m, v

def adam_optimized_gradient_descent(real_exp_vals, init_noise_params,  L, J, g, epsilon = 1e-2, learning_rate=0.01, epochs=100, gradient_style = 'full'):
    """
    Implements Adam optimizer for minimizing the loss.

    Args:
        real_exp_vals (list): QuTiP expectation values.
        init_noise_params (list): Initial noise parameters.
        learning_rate (float): Step size for updates.
        epochs (int): Number of iterations.

    Returns:
        optimized_params (list): Optimized noise parameters.
        loss_history (list): List of loss values during optimization.
    """
    # Initialize noise parameters
    noise_params = np.copy(init_noise_params)
    m, v = np.zeros_like(noise_params), np.zeros_like(noise_params)  # Initialize moments
    loss_history = []

    for epoch in range(1, epochs + 1):
        # Compute the loss
        loss, _ = loss_function(noise_params, real_exp_vals, L,J,g)
        loss_history.append(loss)

        # Compute the gradient
        grad = compute_gradient(loss, loss_function, noise_params, real_exp_vals,L,J,g, epsilon, gradient_style = gradient_style)

        # Update parameters using Adam
        noise_params, m, v = adam_optimizer_update(noise_params, grad, m, v, epoch, learning_rate)

        # Print progress
        print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}, Params: {noise_params}")

    return noise_params, loss_history





if __name__ == "__main__":

    T = 0.5
    dt = 0.1
    t = np.arange(0, T + dt, dt) 
    L = 4
    J = 1
    g = 0.5
    gamma = 0.1



    '''QUTIP Initialization + Simulation'''

#region

    # # Define Pauli matrices
    # sx = qt.sigmax()
    # sy = qt.sigmay()
    # sz = qt.sigmaz()

    # # Construct the Ising Hamiltonian
    # H = 0
    # for i in range(L-1):
    #     H += -J * qt.tensor([sz if n==i or n==i+1 else qt.qeye(2) for n in range(L)])
    # for i in range(L):
    #     H += -g * qt.tensor([sx if n==i else qt.qeye(2) for n in range(L)])



    # # Construct collapse operators
    # c_ops = []

    # # Relaxation operators
    # for i in range(L):
    #     c_ops.append(np.sqrt(gamma) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(L)]))

    # # Dephasing operators
    # for i in range(L):
    #     c_ops.append(np.sqrt(gamma) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(L)]))

    # # Initial state
    # psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])

    # # Define measurement operators
    # sx_list = [qt.tensor([sx if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]
    # sy_list = [qt.tensor([sy if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]
    # sz_list = [qt.tensor([sz if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]

    # obs_list = sx_list + sy_list + sz_list

    # # Exact Lindblad solution
    # result_lindblad = qt.mesolve(H, psi0, t, c_ops, obs_list, progress_bar=True)
    # real_exp_vals = []
    # for site in range(len(obs_list)):
    #     real_exp_vals.append(result_lindblad.expect[site])

#endregion

    '''Lindblad MPO Initialization + Simulation'''
#region






    # Add the path to the lindbladmpo package
    sys.path.append('/Users/maximilianfrohlich/lindbladmpo')

    # Import the LindbladMPOSolver class
    from lindbladmpo.LindbladMPOSolver import LindbladMPOSolver


    # Time vector
    timesteps = int(T/dt)
    t_mpo = np.linspace(0, T, timesteps)



    # Define parameters for LindbladMPOSolver
    parameters = {
        "N": L,
        "t_final": T,
        "tau": dt,  # time step
        "J_z": -2*J,
        "h_x": -2*g,
        "g_0": gamma,  # Strength of deexcitation 
        "g_2": gamma,  # Strength of dephasing
        "init_product_state": ["+z"] * L,  # initial state 
        "1q_components": ["X", "Y", "Z"],  # Request x,y,z observable
        "l_x": L,  # Length of the chain
        "l_y": 1,  # Width of the chain (1 for a 1D chain)
        "b_periodic_x": False,  # Open boundary conditions in x-direction
        "b_periodic_y": False,  # Open boundary conditions in y-direction
    }

    # Create a solver instance and run the simulation
    solver = LindbladMPOSolver(parameters)
    solver.solve()

    # Access the LindbladMPO results
    lindblad_mpo_results = solver.result

    # Reconstruct the Lindblad MPO observables in the correct order
    real_exp_vals = []
    for obs in ['x', 'y', 'z']:  # Order: x0, x1, ..., y0, y1, ..., z0, z1, ...
        for i in range(L):
            real_exp_vals.append(lindblad_mpo_results['obs-1q'][(obs, (i,))][1])

#endregion

    init_noise_params = [0.2, 0.4]  # Initial guesses for gamma_relaxation and gamma_dephasing
    epsilons = [1e-2, 5*1e-2,0.1]
    optimized_params = [None] * len(epsilons)
    loss_history = [None] * len(epsilons)


    for i, epsilon in enumerate(epsilons): 
        # Run Adam optimization
        optimized_params[i], loss_history[i] = adam_optimized_gradient_descent(real_exp_vals, init_noise_params, L,J,g, epsilon, learning_rate=0.01, epochs=100, gradient_style = 'stochastic')
        print(f"Optimized Parameters: {optimized_params[i]}")

    # # Plot the loss history
    # plt.plot(loss_history)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title("Loss History During Adam Optimization")
    # plt.show()
        # Plot the loss history with a logarithmic scale
    plt.figure(figsize=(8, 5))
    for i in range(len(epsilons)):
        plt.plot(loss_history[i], label=f'Loss epsilon {epsilons[i]}')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss History During Adam Optimization (Log Scale)")
    plt.yscale("log")  # Set the y-axis to logarithmic scale
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Improve visibility
    plt.legend()
    plt.show()





