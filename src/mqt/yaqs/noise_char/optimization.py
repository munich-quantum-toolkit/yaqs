import numpy as np
import time 

from scipy.optimize import minimize

import os
import copy

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




def gradient_descent(sim_params_copy, ref_traj, traj_der, learning_rate=0.01, max_iterations=200, tolerance=1e-8, file_name=" "):
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

    sim_params=copy.deepcopy(sim_params_copy)


    loss_history = []

    gr_history = []
    gd_history = []


    dJ_dgr_history = []
    dJ_dgd_history = []


    if os.path.exists(file_name) and file_name != " ":
        os.remove(file_name)


    if file_name != " ":
        with open(file_name, 'w') as file:
            file.write('#  Iter    Loss    Log10(Loss)    Gamma_rel    Gamma_deph \n')


    for iteration in range(max_iterations):
        # Calculate loss and gradients
        loss, exp_vals_traj, dJ_dg = loss_function(sim_params, ref_traj, traj_der)


        if file_name != " ":
            with open(file_name, 'a') as file:
                file.write('    '.join(map(str, [iteration, loss, np.log10(loss),sim_params.gamma_rel, sim_params.gamma_deph ])) + '\n')


        loss_history.append(loss)
        gr_history.append(sim_params.gamma_rel)
        gd_history.append(sim_params.gamma_deph)
        dJ_dgr_history.append(dJ_dg[0])
        dJ_dgd_history.append(dJ_dg[1])


        print(f"!!!!!!! Iteration = {iteration}, Loss = {loss}, g_r = {sim_params.gamma_rel}, g_d = {sim_params.gamma_deph}")

 


        # Check for convergence
        if loss < tolerance:
            print(f"Converged after {iteration} iterations.")
            break       


        sim_params.gamma_rel -= learning_rate * dJ_dg[0]
        sim_params.gamma_deph -= learning_rate * dJ_dg[1]



        if sim_params.gamma_rel < 0:
            sim_params.gamma_rel = 0

        if sim_params.gamma_deph < 0:
            sim_params.gamma_deph = 0





    return loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history




# --- ADAM GRADIENT DESCENT (Modified) ---
def ADAM_gradient_descent(sim_params_copy, ref_traj, traj_der, learning_rate=0.01, max_iterations=200, tolerance=1e-8, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, file_name=" "):
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

    sim_params=copy.deepcopy(sim_params_copy)


    loss_history = []
    gr_history = []
    gd_history = []
    dJ_dgr_history = []
    dJ_dgd_history = []



    if os.path.exists(file_name) and file_name != " ":
        os.remove(file_name)


    if file_name != " ":
        with open(file_name, 'w') as file:
            file.write('#  Iter    Loss    Log10(Loss)    Gamma_rel    Gamma_deph \n')


    # Adam hyperparameters and initialization (NEW)

    m = np.array([0.0, 0.0])  # First moment (for [gamma_rel, gamma_deph])
    v = np.array([0.0, 0.0])  # Second moment (for [gamma_rel, gamma_deph])

    for iteration in range(max_iterations):
        # Calculate loss and gradients (unchanged)
        loss, exp_vals_traj, dJ_dg = loss_function(sim_params, ref_traj, traj_der)
        

        if file_name != " ":
            with open(file_name, 'a') as file:
                file.write('    '.join(map(str, [iteration, loss, np.log10(loss),sim_params.gamma_rel, sim_params.gamma_deph ])) + '\n')


        loss_history.append(loss)
        gr_history.append(sim_params.gamma_rel)
        gd_history.append(sim_params.gamma_deph)
        dJ_dgr_history.append(dJ_dg[0])
        dJ_dgd_history.append(dJ_dg[1])


        print(f"!!!!!!! Iteration = {iteration}, Loss = {loss}, g_r = {sim_params.gamma_rel}, g_d = {sim_params.gamma_deph}")


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

        if sim_params.gamma_rel < 0:
            sim_params.gamma_rel = 0

        if sim_params.gamma_deph < 0:
            sim_params.gamma_deph = 0


        
        



    return loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history


# def line_search(loss_function, loss, sim_params_copy, ref_traj, traj_der, x, p, grad_old):
#     """
#     BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS

#     Args:
#         loss_function (function): The loss function to minimize.
#         sim_params (SimulationParameters): Current simulation parameters.
#         ref_traj (array-like): Reference trajectory data.
#         traj_der (function): Function that computes the trajectory and gradients.
#         x (np.ndarray): Current parameter vector.
#         p (np.ndarray): Search direction.
#         grad_old (np.ndarray): Gradient at the current point.

#     Returns:
#         float: Step size `a` that satisfies the Wolfe conditions.
#     """
#     a = 1  # Initial step size
#     c1 = 1e-4
#     c2 = 0.9

#     sim_params=copy.deepcopy(sim_params_copy)

#     # Compute the current loss and gradient
#     fx=loss


#     # Compute the new parameters
#     x_new = x + a * p

#     while np.any(x_new < 0):
#         a *= 0.5
#         x_new = x + a * p

    
    
#     sim_params.gamma_rel, sim_params.gamma_deph = x_new

#     # Compute the new loss and gradient
#     fx_new, _, grad_new = loss_function(sim_params, ref_traj, traj_der)

#     # Check Wolfe conditions
#     while fx_new > fx + (c1 * a * grad_old.T @ p) or - grad_new.T @ p > - c2 * grad_old.T @ p:
    

#         with open("x_new_a", 'a') as file:
#             file.write('    '.join(map(str, [a, fx_new, fx + (c1 * a * grad_old.T @ p), grad_new.T @ p, c2 * grad_old.T @ p ])) + '\n')

#         # Reduce step size
#         a *= 0.5

#         x_new = x + a * p

#         fx_new, _, grad_new = loss_function(sim_params, ref_traj, traj_der)
        

#     return a



def BFGS(sim_params_copy, ref_traj, traj_der, learning_rate=0, max_iterations=200, tolerance=1e-8, file_name=" "):
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


    sim_params=copy.deepcopy(sim_params_copy)



    loss_history = []
    gr_history = []
    gd_history = []
    dJ_dgr_history = []
    dJ_dgd_history = []

    if os.path.exists(file_name) and file_name != " ":
        os.remove(file_name)

    # Initial parameters
    params_old = np.array([sim_params.gamma_rel, sim_params.gamma_deph])
    n_params = len(params_old)

    # Initial inverse Hessian approximation
    H_inv = np.eye(n_params)

    I = np.eye(n_params)


    # Calculate first loss and gradients
    loss, _, grad_old = loss_function(sim_params, ref_traj, traj_der)



    if file_name != " ":
        with open(file_name, 'w') as file:
            file.write('#  Iter    Loss    Log10(Loss)    Gamma_rel    Gamma_deph \n')



    if file_name != " ":
        with open(file_name, 'a') as file:
            file.write('    '.join(map(str, [0, loss, np.log10(loss),sim_params.gamma_rel, sim_params.gamma_deph ])) + '\n')


    loss_history.append(loss)
    gr_history.append(params_old[0])
    gd_history.append(params_old[1])
    dJ_dgr_history.append(grad_old[0])
    dJ_dgd_history.append(grad_old[1])


    print(f"!!!!!!! Iteration = 0, Loss = {loss}, g_r = {sim_params.gamma_rel}, g_d = {sim_params.gamma_deph}")



    for iteration in range(max_iterations-1):

        # Compute search direction
        p = -H_inv.dot(grad_old)

        a = learning_rate


        # if learning_rate > 0:
        #     a = learning_rate
        # else:
        #     # Perform line search to find step size
        #     a = line_search(loss_function, loss, sim_params, ref_traj, traj_der, params_old, p, grad_old)


        # Update parameters
        params_new = params_old + a * p

        for i in range(n_params):
            if params_new[i] < 0:
                params_new[i] = 0


        # Update simulation parameters
        sim_params.gamma_rel, sim_params.gamma_deph = params_new

        # Calculate new loss and gradients
        loss, _, grad_new = loss_function(sim_params, ref_traj, traj_der)
        
        if file_name != " ":
            with open(file_name, 'a') as file:
                file.write('    '.join(map(str, [iteration+1, loss, np.log10(loss),sim_params.gamma_rel, sim_params.gamma_deph ])) + '\n')


        loss_history.append(loss)
        gr_history.append(params_new[0])
        gd_history.append(params_new[1])
        dJ_dgr_history.append(grad_new[0])
        dJ_dgd_history.append(grad_new[1])


        print(f"!!!!!!! Iteration = {iteration + 1}, Loss = {loss}, g_r = {sim_params.gamma_rel}, g_d = {sim_params.gamma_deph}")
        




        if loss < tolerance:
            print(f"Converged after {iteration} iterations.")
            break


        # Compute differences
        s = a * p
        y = grad_new - grad_old

        
        prod=y.dot(s)

        if prod != 0:
            
            # Update inverse Hessian approximation using BFGS formula
            rho = 1.0 / prod

            H_inv = (I - rho * np.outer(s, y)).dot(H_inv).dot(I - rho * np.outer(y, s)) + rho * np.outer(s, s)


        params_old = params_new
        grad_old = grad_new




    return loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history






def Secant_Penalized_BFGS(sim_params_copy, ref_traj, traj_der, learning_rate=0, max_iterations=200, tolerance=1e-8, Ns=10e8, N0=10e-10, file_name=" "):
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


    sim_params=copy.deepcopy(sim_params_copy)



    loss_history = []
    gr_history = []
    gd_history = []
    dJ_dgr_history = []
    dJ_dgd_history = []

    if os.path.exists(file_name) and file_name != " ":
        os.remove(file_name)

    # Initial parameters
    params_old = np.array([sim_params.gamma_rel, sim_params.gamma_deph])
    n_params = len(params_old)

    # Initial inverse Hessian approximation
    H_inv = np.eye(n_params)

    I = np.eye(n_params)


    # Calculate first loss and gradients
    loss, _, grad_old = loss_function(sim_params, ref_traj, traj_der)



    if file_name != " ":
        with open(file_name, 'w') as file:
            file.write('#  Iter    Loss    Log10(Loss)    Gamma_rel    Gamma_deph \n')



    if file_name != " ":
        with open(file_name, 'a') as file:
            file.write('    '.join(map(str, [0, loss, np.log10(loss),sim_params.gamma_rel, sim_params.gamma_deph ])) + '\n')


    loss_history.append(loss)
    gr_history.append(params_old[0])
    gd_history.append(params_old[1])
    dJ_dgr_history.append(grad_old[0])
    dJ_dgd_history.append(grad_old[1])


    print(f"!!!!!!! Iteration = 0, Loss = {loss}, g_r = {sim_params.gamma_rel}, g_d = {sim_params.gamma_deph}")



    for iteration in range(max_iterations-1):

        # Compute search direction
        p = -H_inv.dot(grad_old)

        a = learning_rate


        # if learning_rate > 0:
        #     a = learning_rate
        # else:
        #     # Perform line search to find step size
        #     a = line_search(loss_function, loss, sim_params, ref_traj, traj_der, params_old, p, grad_old)


        # Update parameters
        params_new = params_old + a * p

        for i in range(n_params):
            if params_new[i] < 0:
                params_new[i] = 0


        # Update simulation parameters
        sim_params.gamma_rel, sim_params.gamma_deph = params_new

        # Calculate new loss and gradients
        loss, _, grad_new = loss_function(sim_params, ref_traj, traj_der)
        
        if file_name != " ":
            with open(file_name, 'a') as file:
                file.write('    '.join(map(str, [iteration+1, loss, np.log10(loss),sim_params.gamma_rel, sim_params.gamma_deph ])) + '\n')


        loss_history.append(loss)
        gr_history.append(params_new[0])
        gd_history.append(params_new[1])
        dJ_dgr_history.append(grad_new[0])
        dJ_dgd_history.append(grad_new[1])


        print(f"!!!!!!! Iteration = {iteration + 1}, Loss = {loss}, g_r = {sim_params.gamma_rel}, g_d = {sim_params.gamma_deph}")
        




        if loss < tolerance:
            print(f"Converged after {iteration} iterations.")
            break


        # Compute differences
        s = a * p
        y = grad_new - grad_old

        
        prod=y.dot(s)

        if prod != 0:
            
            # Update inverse Hessian approximation using BFGS formula

            beta=max(Ns*np.linalg.norm(s) + N0 , 1e-20)

            print(f"N_s*norm(s) = {Ns*np.linalg.norm(s)}, beta = {beta}")

            gamma=1.0/(prod+1/beta)

            omega=1.0/(prod+2/beta)

            H_inv = (I - omega * np.outer(s, y)).dot(H_inv).dot(I - omega * np.outer(y, s))    +     omega * (gamma/omega  + (gamma-omega)*y.dot(H_inv.dot(y))) * np.outer(s, s)


        params_old = params_new
        grad_old = grad_new




    return loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history
