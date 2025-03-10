import numpy as np
import time 

from scipy.optimize import minimize

def trapezoidal(y, x):

    len_x = len(x)
    len_y = len(y)

    integral = np.zeros(len_y)

    integral[0] = 0

    if len_x != len_y:
        raise ValueError("Mismatch in the number of elements between x and y. len(x) = {len_x} and len(y) = {len_y}")


    for i in range(1,len(y)):
        integral[i] = integral[i-1] + 0.5*(x[i] - x[i-1])*(y[i] + y[i-1])

    return integral




def loss_function(sim_params, ref_traj, traj_der):
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
    # t, exp_vals_traj, d_On_d_gk, A_kn_exp_vals = traj_der(sim_params)  
    t, exp_vals_traj, d_On_d_gk = traj_der(sim_params)  
   
    end_time = time.time()
    tjm_time = end_time - start_time
    # print(f"TJM time -> {tjm_time:.4f}")
    
   
    
    # Ensure both lists have the same structure
    if np.shape(ref_traj) != np.shape(exp_vals_traj):
        raise ValueError("Mismatch in the number of sites between qt_exp_vals and tjm_exp_vals.")


    n_jump_site, n_obs_site, L, nt = np.shape(d_On_d_gk)


    # Initialize loss
    loss = 0.0

    dJ_d_gr = 0
    dJ_d_gd = 0


    for i in range(n_obs_site):
        for j in range(L):
            for k in range(nt):

                loss += (exp_vals_traj[i,j,k] - ref_traj[i,j,k])**2

                # I have to add all the derivatives with respect to the same gamma_relaxation and gamma_dephasing
                dJ_d_gr += 2*(exp_vals_traj[i,j,k] - ref_traj[i,j,k]) * d_On_d_gk[0,i,j,k]

                dJ_d_gd += 2*(exp_vals_traj[i,j,k] - ref_traj[i,j,k]) * d_On_d_gk[1,i,j,k]




    return loss, exp_vals_traj, np.array([dJ_d_gr, dJ_d_gd])




def gradient_descent(sim_params, ref_traj, traj_der, learning_rate=0.01, max_iterations=200, tolerance=1e-8):
    """
    Performs gradient descent to minimize the loss function.

    Args:
        sim_params (SimulationParameters): Initial simulation parameters.
        ref_traj (list): Reference trajectory for comparison.
        traj_der (function): Function that runs the simulation and returns the time, 
                         expected values trajectory, and derivatives of the observables 
                         with respect to the noise parameters.
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
        loss, exp_vals_traj, dJ_dg = loss_function(sim_params, ref_traj, traj_der)
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
def ADAM_gradient_descent(sim_params, ref_traj, traj_der, learning_rate=0.01, max_iterations=200, tolerance=1e-8):
    """
    Parameters:
    sim_params (object): Simulation parameters containing gamma_rel and gamma_deph.
    ref_traj (array-like): Reference trajectory data.
    traj_der (function): Function that runs the simulation and returns the time, 
                         expected values trajectory, and derivatives of the observables 
                         with respect to the noise parameters.
    learning_rate (float, optional): Learning rate for the Adam optimizer. Default is 0.01.
    max_iterations (int, optional): Maximum number of iterations for the optimization. Default is 100.
    tolerance (float, optional): Tolerance for the convergence criterion. Default is 1e-6.
    Returns:
    tuple: A tuple containing:
        - loss_history (list): History of loss values during optimization.
        - gr_history (list): History of gamma_rel values during optimization.
        - gd_history (list): History of gamma_deph values during optimization.
        - dJ_dgr_history (list): History of gradients with respect to gamma_rel.
        - dJ_dgd_history (list): History of gradients with respect to gamma_deph.
    

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
        loss, exp_vals_traj, dJ_dg = loss_function(sim_params, ref_traj, traj_der)
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
        
        print(f"!!!!!!! Iteration = {iteration}, Loss = {loss}, g_r = {sim_params.gamma_rel}, g_d = {sim_params.gamma_deph}")

    return loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history






def BFGS(sim_params, ref_traj, traj_der, learning_rate=0.01, max_iterations=200, tolerance=1e-8):
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

    gr_history.append(sim_params.gamma_rel)
    gd_history.append(sim_params.gamma_deph)

    # Initial parameters
    params_old = np.array([sim_params.gamma_rel, sim_params.gamma_deph])
    n_params = len(params_old)

    # Initial inverse Hessian approximation
    H_inv = np.eye(n_params)

    I = np.eye(n_params)


    # Calculate first loss and gradients
    loss, exp_vals_traj, grad_old = loss_function(sim_params, ref_traj, traj_der)
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
            # if params_new[i] > 1:
            #     params_new[i] = 1

        # Update simulation parameters
        sim_params.gamma_rel, sim_params.gamma_deph = params_new

        # Calculate new loss and gradients
        loss, exp_vals_traj, grad_new = loss_function(sim_params, ref_traj, traj_der)
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
        gr_history.append(sim_params.gamma_rel)
        gd_history.append(sim_params.gamma_deph)


        params_old = params_new
        grad_old = grad_new

        print(f"!!!!!!! Iteration = {iteration}, Loss = {loss}, g_r = {sim_params.gamma_rel}, g_d = {sim_params.gamma_deph}")


    return loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history
