import numpy as np
import time 

from scipy.optimize import minimize

import os
import copy
import pickle
import gc
from collections import Counter

def trapezoidal(y : np.ndarray | list, x : np.ndarray | list) -> np.ndarray:
    """
    Compute the cumulative integral of y with respect to x using the trapezoidal rule.

    Parameters
    ----------
    y : array_like
        Array of function values to be integrated.
    x : array_like
        Array of x-coordinates corresponding to the y values.

    Returns
    -------
    integral : numpy.ndarray
        Array of cumulative integral values at each point in x.

    Raises
    ------
    ValueError
        If the lengths of x and y do not match.

    Notes
    -----
    The function returns the cumulative integral, i.e., the integral from x[0] up to each x[i].
    """



    len_x = len(x)
    len_y = len(y)

    integral = np.zeros(len_y)

    integral[0] = 0

    if len_x != len_y:
        raise ValueError("Mismatch in the number of elements between x and y. len(x) = {len_x} and len(y) = {len_y}")


    for i in range(1,len(y)):
        integral[i] = integral[i-1] + 0.5*(x[i] - x[i-1])*(y[i] + y[i-1])

    return integral






class loss_class:

    n_eval = 0

    x_history = []
    f_history = []
    x_avg_history = []
    diff_avg_history = []
    grad_history = []

    n_avg = 20
    


    def compute_avg(self):
        if len(self.x_history) <= self.n_avg:
            x_avg = np.mean(self.x_history, axis=0)
        else:
            x_avg = np.mean(self.x_history[self.n_avg:], axis=0)

        self.x_avg_history.append(x_avg.copy())
        

    def compute_diff_avg(self):
        if len(self.x_avg_history) > 1:
            diff = np.max(np.abs(self.x_avg_history[-1] - self.x_avg_history[-2]))
            self.diff_avg_history.append(diff)


    def post_process(self, x, f, grad):
        self.n_eval += 1
        self.x_history.append(x)
        self.f_history.append(f)
        self.grad_history.append(grad)

        self.compute_avg()
        self.compute_diff_avg()


        self.write_opt_traj()

        # self.log_garbage()

        if self.print_to_file:
            self.write_to_file(self.history_file_name, self.f_history[-1], self.x_history[-1], self.grad_history[-1])
            self.write_to_file(self.history_avg_file_name, self.f_history[-1], self.x_avg_history[-1], self.grad_history[-1])

    def reset(self):
        self.n_eval = 0
        self.x_history = []
        self.f_history = []
        self.x_avg_history = []
        self.diff_avg_history = []

    def set_history(self, x_history, f_history, x_avg_history, diff_avg_history):
        self.n_eval = len(x_history)
        self.x_history = list(x_history)
        self.f_history = list(f_history)
        self.x_avg_history = list(x_avg_history)
        self.diff_avg_history = list(diff_avg_history)



    def set_file_name(self, file_name, reset):

        self.work_dir = file_name.rsplit("/", 1)[0]

        if self.print_to_file:
            self.history_file_name = file_name+".txt"
            self.history_avg_file_name = file_name+"_avg.txt"

            if reset or not os.path.exists(self.history_file_name) :
                with open(self.history_file_name, "w") as file:
                    file.write("# iter  loss  " + "  ".join([f"x{i+1}" for i in range(self.d)]) + "    "  + "  ".join([f"grad_x{i+1}" for i in range(self.d)]) + "\n")
            if reset or not os.path.exists(self.history_avg_file_name):
                with open(self.history_avg_file_name, "w") as file:
                    file.write("# iter  loss  " + "  ".join([f"x{i+1}_avg" for i in range(self.d)]) + "    " + "  ".join([f"grad_x{i+1}" for i in range(self.d)]) + "\n")
        
            # self.garbage_file_name = self.work_dir + "/garbage.txt"
            # self.garbage_type_file_name = self.work_dir + "/garbage_type.txt"
            # self.unreachable_file_name = self.work_dir + "/unreachable.txt"


            # if reset or not os.path.exists(self.garbage_file_name):
            #     with open(self.garbage_file_name, "w") as file:
            #         file.write("# iter    Collected objects \n")

            # if reset or not os.path.exists(self.unreachable_file_name):
            #     with open(self.unreachable_file_name, "w") as file:
            #         file.write("# iter    Unreachable objects \n")





    def write_to_file(self, file_name, f, x, grad):
        if self.print_to_file:
            with open(file_name, "a") as file:
                file.write(f"{self.n_eval}    {f}  " + "  ".join([f"{x[j]:.6f}" for j in range(self.d)]) + "    " + "  ".join([f"{grad[j]:.6f}" for j in range(self.d)]) + "\n")

    
    def log_garbage(self):
        """
        Log garbage collection information to a file.
        """
        objs = gc.get_objects()
        with open(self.garbage_file_name, "a") as file:

            file.write(f" {self.n_eval}  {len(objs)} \n")
        
        with open(self.garbage_type_file_name, "a") as file:
            # Get all tracked objects
            

            # Count by type name
            type_counts = Counter(type(obj).__name__ for obj in objs)

            # Show the top N most common types
            top_n = 10
            file.write(f"{self.n_eval}  ")
            for typename, count in type_counts.most_common(top_n):
                file.write(f"{typename}:{count}   ")
            file.write("\n")
        
        unreachable = gc.collect()

        with open(self.unreachable_file_name, "a") as file:
            file.write(f"{self.n_eval}  {unreachable} \n")


    def write_opt_traj(self):
        """
        Write the reference trajectory to a file.
        
        Parameters:
        - t: Time array.
        - ref_traj: Reference trajectory data.
        - file_name: Name of the output file.
        """

        n_obs_site, L, n_t = self.exp_vals_traj.shape

        exp_vals_traj_reshaped = self.exp_vals_traj.reshape(-1, self.exp_vals_traj.shape[-1])

        exp_vals_traj_with_t=np.concatenate([np.array([self.t]), exp_vals_traj_reshaped], axis=0)


        ## Saving reference trajectory and gammas
        header =   "t  " +  "  ".join([obs+str(i)   for obs in ["x","y","z"][:n_obs_site] for i in range(L) ])

        np.savetxt(self.work_dir + f"/opt_traj_{self.n_eval}.txt" , exp_vals_traj_with_t.T, header=header, fmt='%.6f')








class loss_class_2d(loss_class):

    def __init__(self, sim_params, ref_traj, traj_der, print_to_file=False):

        self.d = 2  

        self.print_to_file = print_to_file

        self.ref_traj = ref_traj.copy()
        self.traj_der = traj_der
        self.sim_params = copy.deepcopy(sim_params)

        

    def __call__(self, x):

        self.sim_params.set_gammas(x[0], x[1])


        start_time = time.time()

        self.t, self.exp_vals_traj, self.d_On_d_gk, avg_min_max_traj_time = self.traj_der(self.sim_params) 

        end_time = time.time()


        # self.t = t.copy()
        # self.exp_vals_traj = exp_vals_traj.copy() 

        n_jump_site, n_obs_site, L, nt = np.shape(self.d_On_d_gk)


        diff = self.exp_vals_traj - self.ref_traj


        f = np.sum(diff**2)

        ## I reshape diff so it has a shape compatible with d_On_d_gk (n_jump_site, n_obs_site, L, nt) to do elemtwise multiplication.
        ## Then I sum over the n_obs_site, L and nt dimensions to get the gradient for each gamma,
        ##  returning a vector of shape (n_jump_site)
        grad = np.sum(2 * diff.reshape(1,n_obs_site, L, nt) * self.d_On_d_gk, axis=(1,2,3))


        self.post_process(x.copy(),f, grad.copy())

        sim_time = end_time - start_time ## Simulation time

        return f, grad, sim_time, avg_min_max_traj_time




class loss_class_nd(loss_class):

    def __init__(self, sim_params, ref_traj, traj_der, print_to_file=False):

        self.print_to_file = print_to_file

        self.ref_traj = ref_traj.copy()
        self.traj_der = traj_der
        self.sim_params = copy.deepcopy(sim_params)

        self.n_gamma_rel=len(self.sim_params.gamma_rel)
        self.n_gamma_deph=len(self.sim_params.gamma_deph)


        self.d = self.n_gamma_rel + self.n_gamma_deph

        

    def __call__(self, x):


        self.sim_params.set_gammas(x[:self.n_gamma_rel], x[self.n_gamma_rel:])


        start_time = time.time()

        t, exp_vals_traj, d_On_d_gk, avg_min_max_traj_time = self.traj_der(self.sim_params) 


        end_time = time.time()


        self.t = t.copy()
        self.exp_vals_traj = exp_vals_traj.copy() 

        n_jump_site, n_obs_site, L, nt = np.shape(d_On_d_gk)


        diff = exp_vals_traj - self.ref_traj


        f = np.sum(diff**2)

        ## I reshape diff so it has a shape compatible with d_On_d_gk (n_jump_site, n_obs_site, L, nt) to do elemtwise multiplication.
        ## Then I sum over the n_obs_site and nt dimensions to get the gradient for each gamma for each site,
        ##  returning a matrix of shape (n_jump_site, L) which I then flatten obtaining a vector of shape (n_jump_site*L) 
        grad = np.sum(2 * diff.reshape(1,n_obs_site, L, nt) * d_On_d_gk, axis=(1,3)).flatten()

        self.post_process(x.copy(),f, grad.copy())

        sim_time = end_time - start_time  ## Simulation time

        return f, grad, sim_time, avg_min_max_traj_time








def loss_function(sim_params, ref_traj, traj_der, loss_std=0, dJ_d_gr_std=0, dJ_d_gd_std=0):
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


    while True:
        error=np.random.normal(loc=0.0, scale=loss_std)

        if loss + error > 0:
            loss += error
            break

    dJ_d_gr += np.random.normal(loc=0.0, scale=dJ_d_gr_std)
    dJ_d_gd += np.random.normal(loc=0.0, scale=dJ_d_gd_std)




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





def ADAM_loss_class(f, x_copy, alpha=0.05, max_iterations=1000, threshhold = 5e-4, max_n_convergence = 50, tolerance=1e-8, beta1 = 0.5, beta2 = 0.999, epsilon = 1e-8, restart=False, restart_file=None):


    restart_dir = f.work_dir

    # Find the latest restart file in the restart_dir
    if restart_file is None and restart:
        if os.path.isdir(restart_dir):
            restart_files = [f for f in os.listdir(restart_dir) if f.startswith("restart_step_") and f.endswith(".pkl")]
            if restart_files:
                # Sort by step number
                restart_files.sort()
                restart_file = os.path.join(restart_dir, restart_files[-1])



    # Initialization
    if restart:
        if restart_file is None or not os.path.exists(restart_file):
            raise ValueError("Restart file not found or not specified.")
        with open(restart_file, "rb") as handle:
            saved = pickle.load(handle)
        x = saved["x"]
        m = saved["m"]
        v = saved["v"]
        start_iter = saved["iteration"] + 1  # resume from next iteration

        f.set_history(saved["x_history"], saved["f_history"], saved["x_avg_history"], saved["diff_avg_history"])

        f.t = saved["t"].copy()
        f.exp_vals_traj = saved["exp_vals_traj"].copy()

        print(f"Restarting from iteration {saved['iteration']}, loss={saved['loss']:.6f}")


    else:
        # Remove all .pkl files in the folder
        for fname in os.listdir(restart_dir):
            if fname.endswith(".pkl"):
                try:
                    os.remove(os.path.join(restart_dir, fname))
                except Exception as e:
                    print(f"Warning: Could not remove {fname}: {e}")

        x = x_copy.copy()
        d = len(x)
        m = np.zeros(d)
        v = np.zeros(d)
        start_iter = 0

        # Write a header to performance_metric.txt in f.work_dir
        perf_file = os.path.join(f.work_dir, "performance_metric_sec.txt")
        with open(perf_file, "w") as pf:
            pf.write("# iter    opt_step_time    simulation_time    avg_traj_time    min_traj_time    max_traj_time\n")

    


    for i in range(start_iter,max_iterations):
        # Calculate loss and gradients (unchanged)

        start_time = time.time()

        loss, grad, sim_time, avg_min_max_traj_time = f(x)

    
        # Adam update steps (NEW)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        beta1_t = beta1 ** (i + 1)
        beta2_t = beta2 ** (i + 1)


        m_hat = m / (1 - beta1_t)
        v_hat = v / (1 - beta2_t)

        update = alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        # Update simulation parameters with Adam update (NEW)
        x -= update   


        # Ensure non-negativity for the parameters
        x[x < 0] = 0
        x[x > 1] = 1



        
        restart_data = {
            "iteration": i,
            "x": x.copy(),
            "x_loss": x+update,
            "loss": loss,
            "grad": grad.copy(),
            "beta1": beta1,
            "beta2": beta2,
            "m": m.copy(),
            "v": v.copy(),
            "update":update.copy(),
            "f_history": f.f_history.copy(),
            "x_history": f.x_history.copy(),
            "x_avg_history": f.x_avg_history.copy(),
            "diff_avg_history": f.diff_avg_history.copy(),
            "t": f.t.copy(),
            "exp_vals_traj": f.exp_vals_traj.copy(),
        }

        with open(os.path.join(restart_dir, f"restart_step_{i+1:04d}.pkl"), "wb") as handle:
            pickle.dump(restart_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

        end_time = time.time()


        iter_time = end_time - start_time


        with open(perf_file, "a") as pf:
            pf.write(f"  {i}    {iter_time}    {sim_time}    {avg_min_max_traj_time[0]}    {avg_min_max_traj_time[1]}    {avg_min_max_traj_time[2]}\n")
        

        if abs(loss) < tolerance:
            print(f"Loss converged after {i} iterations. Loss={loss}, tolerance={tolerance}")
            break

        # Convergence check
        if len(f.diff_avg_history) > max_n_convergence and all(
                diff < threshhold for diff in f.diff_avg_history[-max_n_convergence:]):
            print(f"Parameters convergence reached at iteration {i}.")
            break
        


    return f.f_history, f.x_history, f.x_avg_history, f.t, f.exp_vals_traj



# --- ADAM GRADIENT DESCENT (Modified) ---
def ADAM_gradient_descent(sim_params_copy, ref_traj, traj_der, learning_rate=0.01, max_iterations=200, tolerance=1e-8, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, file_name=" ", alpha=1, loss_std=0, dJ_d_gr_std=0, dJ_d_gd_std=0):
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
        loss, exp_vals_traj, dJ_dg = loss_function(sim_params, ref_traj, traj_der, loss_std=loss_std, dJ_d_gr_std=dJ_d_gr_std, dJ_d_gd_std=dJ_d_gd_std)
        

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


        a = learning_rate/(1+iteration/alpha)


        update = a * m_hat / (np.sqrt(v_hat) + epsilon)
        
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



def BFGS(sim_params_copy, ref_traj, traj_der, learning_rate=0, max_iterations=200, tolerance=1e-8, file_name=" ", loss_std=0, dJ_d_gr_std=0, dJ_d_gd_std=0):
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
    loss, _, grad_old = loss_function(sim_params, ref_traj, traj_der, loss_std=loss_std, dJ_d_gr_std=dJ_d_gr_std, dJ_d_gd_std=dJ_d_gd_std)



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
        loss, _, grad_new = loss_function(sim_params, ref_traj, traj_der, loss_std=loss_std, dJ_d_gr_std=dJ_d_gr_std, dJ_d_gd_std=dJ_d_gd_std)
        
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






def Secant_Penalized_BFGS(sim_params_copy, ref_traj, traj_der, learning_rate=0, max_iterations=200, tolerance=1e-8, alpha=1, Ns=10e8, N0=10e-10, file_name=" ", loss_std=0, dJ_d_gr_std=0, dJ_d_gd_std=0):
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
    loss, _, grad_old = loss_function(sim_params, ref_traj, traj_der, loss_std=loss_std, dJ_d_gr_std=dJ_d_gr_std, dJ_d_gd_std=dJ_d_gd_std)



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


        
        a = learning_rate/(1+iteration/alpha)

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
        loss, _, grad_new = loss_function(sim_params, ref_traj, traj_der, loss_std=loss_std, dJ_d_gr_std=dJ_d_gr_std, dJ_d_gd_std=dJ_d_gd_std)
        
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

            beta=(Ns/np.sqrt(dJ_d_gr_std**2+dJ_d_gd_std**2))*np.linalg.norm(s) + N0


            print(f"N_s*norm(s) = {Ns*np.linalg.norm(s)}, beta = {beta}, a = {a}")

            gamma=1.0/(prod+1/beta)

            omega=1.0/(prod+2/beta)

            H_inv = (I - omega * np.outer(s, y)).dot(H_inv).dot(I - omega * np.outer(y, s))    +     omega * (gamma/omega  + (gamma-omega)*y.dot(H_inv.dot(y))) * np.outer(s, s)


        params_old = params_new
        grad_old = grad_new




    return loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history




### For Bayesian Optimization
import torch
import botorch
import matplotlib.pyplot as plt
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.logei import qLogNoisyExpectedImprovement


# GP model for noisy observations
class NoisyDerivativeGPModel(ExactGP):
    def __init__(self, train_x, train_y, noise, likelihood):
        self.num_outputs = 1
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[-1]))
        self.likelihood.noise = noise  # fixed noise per point

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def transform_inputs(self, *args, **kwargs):
        if args:
            return args[0]
        elif "X" in kwargs:
            return kwargs["X"]
        else:
            raise ValueError("transform_inputs called without inputs")


class WrappedNoisyDerivativeModel(Model):
    def __init__(self, gp_model):
        super().__init__()
        self.gp = gp_model
        self._dtype = gp_model.train_inputs[0].dtype
        self._device = gp_model.train_inputs[0].device

    def posterior(self, X, output_indices=None, observation_noise=False, **kwargs):
        self.gp.eval()
        mvn = self.gp(X)  # allow gradients to flow!
        return GPyTorchPosterior(mvn)

    def condition_on_observations(self, X, Y, **kwargs):
        raise NotImplementedError("Not needed for this use case.")

    def transform_inputs(self, *args, **kwargs):
        return self.gp.transform_inputs(*args, **kwargs)

    @property
    def num_outputs(self):  # ✅ here’s the fix
        return 1



def bayesian_optimization(sim_params_copy, ref_traj, traj_der, bounds_list, acquisition="UCB", n_init=5, max_iterations=200, tolerance=1e-8, beta=0.1, num_restarts=10, raw_samples=50, file_name=" ", device="cpu", loss_std=0, dJ_d_gr_std=0, dJ_d_gd_std=0):
    """
    Perform Bayesian Optimization with noisy function and gradient observations.

    Args:
        sim_params_copy (object): Simulation parameters containing gamma_rel and gamma_deph.
        ref_traj (array-like): Reference trajectory data.
        traj_der (function): Function that runs the simulation and returns the time, 
                                expected values trajectory, and derivatives of the observables 
                                with respect to the noise parameters.
        d (int): Dimensionality of the input space.
        n_init (int): Number of initial random samples.
        n_iter (int): Number of optimization iterations.
        noise_f (float): Noise level for the function observations.
        noise_g (float): Noise level for the gradient observations.
        beta (float): Exploration-exploitation trade-off parameter for UCB.
        num_restarts (int): Number of restarts for acquisition function optimization.
        raw_samples (int): Number of raw samples for acquisition function optimization.
        file_name (str): File name to save optimization progress.
        device (str): Device to use for computation ("cpu" or "cuda").
        dtype (torch.dtype): Data type for tensors.

    Returns:
        tuple: A tuple containing:
            - loss_history (list): History of loss values during optimization.
            - gr_history (list): History of gamma_rel values during optimization.
            - gd_history (list): History of gamma_deph values during optimization.
    """
    sim_params = copy.deepcopy(sim_params_copy)


    d = 2  # Number of parameters to optimize (gamma_rel and gamma_deph)

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

    # Config
    bounds = torch.tensor(bounds_list, device=device, dtype=torch.double).T  # Transpose to match the shape [[lower1, lower2, ...], [upper1, upper2, ...]]

    # Initial data
    X = torch.empty(n_init, len(bounds_list), device=device, dtype=torch.double, requires_grad=True)
    X_new = torch.empty_like(X)  # Create a new tensor to avoid in-place operations
    for i, (lower, upper) in enumerate(bounds_list):
        X_new[:, i] = torch.rand(n_init, device=device, dtype=torch.double) * (upper - lower) + lower  # Scale and shift random values to the bounds
    X = X_new  # Assign the new tensor to X

    Y_vals = []
    grad_vals = []

    for i in range(n_init):
        sim_params.gamma_rel, sim_params.gamma_deph = X[i].detach().cpu().numpy()
        loss, _, dJ_dg = loss_function(sim_params, ref_traj, traj_der, loss_std=loss_std, dJ_d_gr_std=dJ_d_gr_std, dJ_d_gd_std=dJ_d_gd_std)
        Y_vals.append(-loss)
        grad_vals.append(dJ_dg)

        loss_history.append(loss)
        gr_history.append(sim_params.gamma_rel)
        gd_history.append(sim_params.gamma_deph)
        dJ_dgr_history.append(dJ_dg[0])
        dJ_dgd_history.append(dJ_dg[1])

    Y = torch.tensor(Y_vals, dtype=torch.double, device=device).unsqueeze(-1)  # shape: (n_init, 1)
    grad_Y = torch.tensor(grad_vals, dtype=torch.double, device=device)   

    X_list = [X.detach()]
    Y_list = [Y.detach()]
    grad_list = [grad_Y.detach()]

    for iteration in range(n_init,max_iterations+n_init):
        X_train = torch.cat(X_list, dim=0)
        Y_train = torch.cat(Y_list, dim=0)
        d_train = torch.cat(grad_list, dim=0)

        # Joint input data
        X_func = X_train
        X_grad = X_train.repeat_interleave(d, dim=0)
        X_joint = torch.cat([X_func, X_grad], dim=0)

        # Joint targets
        Y_joint = torch.cat([Y_train.squeeze(-1), d_train.reshape(-1)], dim=0)

        # Create noise vector: same order as Y_joint
        noise_joint = torch.cat([
            torch.full((len(Y_train),), loss_std ** 2, dtype=torch.double),
            torch.full((len(d_train.reshape(-1)),), (max(dJ_d_gr_std,dJ_d_gd_std)) ** 2, dtype=torch.double)
        ])

        # GP model with fixed noise
        likelihood = FixedNoiseGaussianLikelihood(noise=noise_joint)
        model = NoisyDerivativeGPModel(X_joint, Y_joint, noise_joint, likelihood).to(device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        model.train()
        fit_gpytorch_mll(mll)
        model.eval()

        # Wrap the model
        model = WrappedNoisyDerivativeModel(model)

        # Acquisition
        if acquisition == "UCB":
            acq_func = UpperConfidenceBound(model, beta=beta)

        elif acquisition == "qNEI":
            acq_func = qNoisyExpectedImprovement(model, X_baseline=X_joint)
        
        elif acquisition == "qLNEI":
            acq_func = qLogNoisyExpectedImprovement(model, X_baseline=X_joint)

        else:
            raise ValueError(f"Unknown acquisition function: {acquisition}. Valid options are 'UCB', 'qNEI', and 'qLNEI'.")

        candidate, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        # Sample at new point
        candidate.requires_grad_()
        print("Candidate",candidate.detach().cpu().numpy())
        sim_params.gamma_rel, sim_params.gamma_deph = candidate.detach().cpu().numpy()[0]
        loss, _, dJ_dg = loss_function(sim_params, ref_traj, traj_der, loss_std=loss_std, dJ_d_gr_std=dJ_d_gr_std, dJ_d_gd_std=dJ_d_gd_std)

        if file_name != " ":
            with open(file_name, 'a') as file:
                file.write('    '.join(map(str, [iteration, loss, np.log10(loss), sim_params.gamma_rel, sim_params.gamma_deph])) + '\n')

        loss_history.append(loss)
        gr_history.append(sim_params.gamma_rel)
        gd_history.append(sim_params.gamma_deph)
        dJ_dgr_history.append(dJ_dg[0])
        dJ_dgd_history.append(dJ_dg[1])

        print(f"!!!!!!! Iteration = {iteration + 1}, Loss = {loss}, g_r = {sim_params.gamma_rel}, g_d = {sim_params.gamma_deph}")

        X_list.append(candidate.detach())
        Y_list.append(torch.tensor([[-loss]], dtype=torch.double, device=device))
        grad_list.append(torch.tensor([dJ_dg], dtype=torch.double, device=device))

        if loss < tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break

    return loss_history, gr_history, gd_history
