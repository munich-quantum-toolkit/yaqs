import numpy as np
import time 

from scipy.optimize import minimize

import os
import copy
import pickle
import gc
from collections import Counter

from typing import Callable


from mqt.yaqs.noise_char.propagation import SimulationParameters

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
        """
        Computes the average of the parameter history and appends it to the average history.
        If the length of `x_history` is less than or equal to `n_avg`, computes the mean over the entire `x_history`.
        Otherwise, computes the mean over the entries in `x_history` starting from index `n_avg`.
        The computed average is appended to `x_avg_history`.
        Returns:
            None
        """

        if len(self.x_history) <= self.n_avg:
            x_avg = np.mean(self.x_history, axis=0)
        else:
            x_avg = np.mean(self.x_history[self.n_avg:], axis=0)

        self.x_avg_history.append(x_avg.copy())
        

    def compute_diff_avg(self):
        """
        Computes the maximum absolute difference between the last two entries in `x_avg_history`
        and appends the result to `diff_avg_history`.
        This method is intended to track the change in the average values stored in `x_avg_history`
        over successive iterations.
        """

        if len(self.x_avg_history) > 1:
            diff = np.max(np.abs(self.x_avg_history[-1] - self.x_avg_history[-2]))
            self.diff_avg_history.append(diff)


    def post_process(self, x : np.ndarray, f : float, grad : np.ndarray):
        """
        Post-processes the results of an optimization step.
        This method updates the evaluation count and appends the current parameter values,
        function value, and gradient to their respective histories. It then computes the
        average and difference of the optimization trajectory, writes the optimization
        trajectory to file, and optionally logs the latest results to specified files.
        Args:
            x (array-like): The current parameter values.
            f (float): The current function value.
            grad (array-like): The current gradient.
        Side Effects:
            - Increments the evaluation counter (`self.n_eval`).
            - Updates histories: `self.x_history`, `self.f_history`, `self.grad_history`.
            - Computes and updates averages and differences.
            - Writes optimization trajectory to file.
            - Optionally writes the latest results to history files if `self.print_to_file` is True.
        """

        self.n_eval += 1
        self.x_history.append(x)
        self.f_history.append(f)
        self.grad_history.append(grad)

        self.compute_avg()
        self.compute_diff_avg()


        self.write_opt_traj()

        if self.print_to_file:
            self.write_to_file(self.history_file_name, self.f_history[-1], self.x_history[-1], self.grad_history[-1])
            self.write_to_file(self.history_avg_file_name, self.f_history[-1], self.x_avg_history[-1], self.grad_history[-1])

    def reset(self):
        """
        Reset the optimization history and evaluation counter.
        This method clears all stored histories related to the optimization process,
        including the number of function evaluations, parameter vectors, function values,
        averaged parameter vectors, and averaged differences. After calling this method,
        the optimizer's state is as if no evaluations have been performed.
        """

        self.n_eval = 0
        self.x_history = []
        self.f_history = []
        self.x_avg_history = []
        self.diff_avg_history = []

    def set_history(self, x_history : list | np.ndarray, f_history : list | np.ndarray, x_avg_history : list | np.ndarray, diff_avg_history : list | np.ndarray):
        """
        Stores the optimization history data.
        Parameters
        ----------
        x_history : list or array-like
            Sequence of parameter vectors evaluated during the optimization process.
        f_history : list or array-like
            Sequence of objective function values corresponding to each parameter vector in `x_history`.
        x_avg_history : list or array-like
            Sequence of averaged parameter vectors, typically used for tracking the running average during optimization.
        diff_avg_history : list or array-like
            Sequence of differences between consecutive averaged parameter vectors, useful for convergence analysis.
        Notes
        -----
        This method updates the object's history attributes and sets the number of evaluations performed.
        """

        self.n_eval = len(x_history)
        self.x_history = list(x_history)
        self.f_history = list(f_history)
        self.x_avg_history = list(x_avg_history)
        self.diff_avg_history = list(diff_avg_history)



    def set_file_name(self, file_name : str, reset : bool):
        """
        Sets the base file name for storing optimization history and related files.
        Parameters:
            file_name (str): The base file path (excluding extension) to use for output files.
            reset (bool): If True, existing files will be overwritten; if False, files will only be created if they do not exist.
        Side Effects:
            - Sets the working directory (`self.work_dir`) based on the provided file name.
            - If `self.print_to_file` is True:
                - Sets file names for history and average history logs.
                - Initializes or resets these files with appropriate headers if `reset` is True or files do not exist.
            - (Commented out) Optionally sets up additional files for garbage collection and unreachable objects.
        Notes:
            - The number of variables (`self.d`) determines the number of columns in the headers.
            - File extensions and naming conventions are handled within the method.
        """


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
        


    def write_to_file(self, file_name : str, f : float, x : np.ndarray, grad : np.ndarray):
        """
        Writes the current evaluation data to a specified file if file output is enabled.
        Parameters:
            file_name (str): The path to the file where data will be appended.
            f (float): The function value at the current evaluation.
            x (np.ndarray): The current parameter vector.
            grad (np.ndarray): The gradient vector at the current evaluation.
        Notes:
            The output is appended to the file in a formatted line containing:
            - The evaluation count (`self.n_eval`)
            - The function value (`f`)
            - The parameter vector (`x`) with 6 decimal places
            - The gradient vector (`grad`) with 6 decimal places
            Each value is separated by spaces.
            The method only writes to the file if `self.print_to_file` is True.
        """

        if self.print_to_file:
            with open(file_name, "a") as file:
                file.write(f"{self.n_eval}    {f}  " + "  ".join([f"{x[j]:.6f}" for j in range(self.d)]) + "    " + "  ".join([f"{grad[j]:.6f}" for j in range(self.d)]) + "\n")

    

    def write_opt_traj(self):
        """
        Saves the optimized trajectory of expectation values to a text file.
        This method reshapes the `exp_vals_traj` array, concatenates the time array `self.t` as the first row,
        and writes the resulting data to a file named `opt_traj_{self.n_eval}.txt` in the working directory.
        The file includes a header with time and observable labels.
        The output file format:
            - Each column corresponds to a time point or an observable at a specific site.
            - The first column is time (`t`).
            - Subsequent columns are labeled as `x0`, `y0`, `z0`, ..., up to the number of observed sites and system size.
        Attributes used:
            exp_vals_traj (np.ndarray): Array of expectation values with shape (n_obs_site, L, n_t).
            t (np.ndarray): Array of time points.
            work_dir (str): Directory where the output file will be saved.
            n_eval (int): Evaluation index used in the output filename.
        File saved:
            {work_dir}/opt_traj_{n_eval}.txt
        """



        n_obs_site, L, n_t = self.exp_vals_traj.shape

        exp_vals_traj_reshaped = self.exp_vals_traj.reshape(-1, self.exp_vals_traj.shape[-1])

        exp_vals_traj_with_t=np.concatenate([np.array([self.t]), exp_vals_traj_reshaped], axis=0)


        ## Saving reference trajectory and gammas
        header =   "t  " +  "  ".join([obs+str(i)   for obs in ["x","y","z"][:n_obs_site] for i in range(L) ])

        np.savetxt(self.work_dir + f"/opt_traj_{self.n_eval}.txt" , exp_vals_traj_with_t.T, header=header, fmt='%.6f')








class loss_class_2d(loss_class):
    """
    loss_class_nd is a subclass of loss_class designed for optimization in noise characterization of open quantum systems.
    This class encapsulates the objective function and its gradient computation for optimizing noise parameters (relaxation and dephasing rates) in quantum system simulations. 
    It compares simulated trajectories to a reference trajectory and provides the sum of squared differences as the loss, along with its gradient with respect to the noise parameters.
    It is designed for the case of the same noise parameters for each site, in total 2 parameters.
    
    Attributes
    print_to_file : bool
        Flag indicating whether to print output to a file.
    ref_traj : np.ndarray
        Copy of the reference trajectory data.
    traj_der : Callable[[SimulationParameters], tuple]
        Function to compute trajectory derivatives given simulation parameters.
    sim_params : SimulationParameters
        Deep copy of the simulation parameters, including noise rates.
    n_gamma_rel : int
        Number of relaxation rates in the simulation parameters.
    n_gamma_deph : int
        Number of dephasing rates in the simulation parameters.
    d : int
        Total number of noise parameters (relaxation + dephasing).
    Methods
    __init__(sim_params, ref_traj, traj_der, print_to_file=False)
        Initializes the loss_class_nd instance with simulation parameters, reference trajectory, and trajectory derivative function.
    __call__(x: np.ndarray) -> tuple
        Evaluates the objective function and its gradient for the given noise parameters.
        Updates the simulation parameters, runs the trajectory simulation and its derivatives,
        computes the difference between simulated and reference trajectories, and calculates:
            - The objective function value (sum of squared differences)
            - The gradient with respect to the noise parameters
            - The simulation time
            - The average of the minimum and maximum trajectory times
            Array containing the noise parameters to be optimized. The first n_gamma_rel elements
            correspond to relaxation rates, and the remaining elements correspond to dephasing rates.
            Value of the objective function (sum of squared differences).
            Gradient of the objective function with respect to the noise parameters.
            Time taken to run the simulation, in seconds.
            Average of the minimum and maximum trajectory times from the simulation.
    """

    def __init__(self, sim_params: SimulationParameters, ref_traj: np.ndarray, traj_der: Callable[[SimulationParameters],tuple], print_to_file: bool = False):
        """
        Initializes the optimization class for noise characterization.
        Args:
            sim_params (SimulationParameters): The simulation parameters to be used.
            ref_traj (np.ndarray): Reference trajectory as a NumPy array.
            traj_der (Callable[[SimulationParameters], tuple]): A callable that computes the trajectory derivative given simulation parameters.
            print_to_file (bool, optional): If True, output will be printed to a file. Defaults to False.
        Attributes:
            d (int): Dimensionality, set to 2.
            print_to_file (bool): Indicates whether to print output to a file.
            ref_traj (np.ndarray): Copy of the reference trajectory.
            traj_der (Callable): Function to compute trajectory derivative.
            sim_params (SimulationParameters): Deep copy of the simulation parameters.
        """


        self.d = 2  

        self.print_to_file = print_to_file

        self.ref_traj = ref_traj.copy()
        self.traj_der = traj_der
        self.sim_params = copy.deepcopy(sim_params)

        

    def __call__(self, x: np.ndarray) -> tuple:
        """
        Evaluates the objective function and its gradient for the given parameters.
        This method updates the simulation parameters with the provided gamma values,
        runs the trajectory simulation and its derivative, computes the loss (sum of squared
        differences between the simulated and reference trajectories), and calculates the gradient
        of the loss with respect to the gamma parameters. It also measures the simulation time
        and retrieves the average minimum and maximum trajectory times.
        Args:
            x (np.ndarray): Array of gamma parameters to be set in the simulation.
        Returns:
            tuple:
                - f (float): The value of the objective function (sum of squared differences).
                - grad (np.ndarray): The gradient of the objective function with respect to gamma parameters.
                - sim_time (float): The time taken to run the simulation (in seconds).
                - avg_min_max_traj_time (Any): Average minimum and maximum trajectory times (type depends on `traj_der` output).
        """


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
    """
    loss_class_nd is a subclass of loss_class designed for optimization in noise characterization of open quantum systems.
    This class encapsulates the objective function and its gradient computation for optimizing noise parameters (relaxation and dephasing rates) in quantum system simulations. 
    It compares simulated trajectories to a reference trajectory and provides the sum of squared differences as the loss, along with its gradient with respect to the noise parameters.
    It is designed for the case of independent noise parameters for each site, in total 2*L parameters.
    
    Attributes
    print_to_file : bool
        Flag indicating whether to print output to a file.
    ref_traj : np.ndarray
        Copy of the reference trajectory data.
    traj_der : Callable[[SimulationParameters], tuple]
        Function to compute trajectory derivatives given simulation parameters.
    sim_params : SimulationParameters
        Deep copy of the simulation parameters, including noise rates.
    n_gamma_rel : int
        Number of relaxation rates in the simulation parameters.
    n_gamma_deph : int
        Number of dephasing rates in the simulation parameters.
    d : int
        Total number of noise parameters (relaxation + dephasing).
    Methods
    __init__(sim_params, ref_traj, traj_der, print_to_file=False)
        Initializes the loss_class_nd instance with simulation parameters, reference trajectory, and trajectory derivative function.
    __call__(x: np.ndarray) -> tuple
        Evaluates the objective function and its gradient for the given noise parameters.
        Updates the simulation parameters, runs the trajectory simulation and its derivatives,
        computes the difference between simulated and reference trajectories, and calculates:
            - The objective function value (sum of squared differences)
            - The gradient with respect to the noise parameters
            - The simulation time
            - The average of the minimum and maximum trajectory times
            Array containing the noise parameters to be optimized. The first n_gamma_rel elements
            correspond to relaxation rates, and the remaining elements correspond to dephasing rates.
            Value of the objective function (sum of squared differences).
            Gradient of the objective function with respect to the noise parameters.
            Time taken to run the simulation, in seconds.
            Average of the minimum and maximum trajectory times from the simulation.
    """


    def __init__(self, sim_params: SimulationParameters, ref_traj: np.ndarray, traj_der: Callable[[SimulationParameters], tuple], print_to_file: bool = False):
        """
        Initializes the optimization class for noise characterization.
        Args:
            sim_params (SimulationParameters): Simulation parameters containing relaxation and dephasing rates.
            ref_traj (np.ndarray): Reference trajectory data as a NumPy array.
            traj_der (Callable[[SimulationParameters], tuple]): Callable that computes the trajectory derivative given simulation parameters.
            print_to_file (bool, optional): If True, enables printing output to a file. Defaults to False.
        Attributes:
            print_to_file (bool): Flag indicating whether to print output to a file.
            ref_traj (np.ndarray): Copy of the reference trajectory.
            traj_der (Callable): Function to compute trajectory derivatives.
            sim_params (SimulationParameters): Deep copy of the simulation parameters.
            n_gamma_rel (int): Number of relaxation rates in the simulation parameters.
            n_gamma_deph (int): Number of dephasing rates in the simulation parameters.
            d (int): Total number of noise parameters (relaxation + dephasing).
        """


        self.print_to_file = print_to_file

        self.ref_traj = ref_traj.copy()
        self.traj_der = traj_der
        self.sim_params = copy.deepcopy(sim_params)

        self.n_gamma_rel=len(self.sim_params.gamma_rel)
        self.n_gamma_deph=len(self.sim_params.gamma_deph)


        self.d = self.n_gamma_rel + self.n_gamma_deph

        

    def __call__(self, x: np.ndarray) -> tuple:
        """
        Evaluates the objective function and its gradient for the given parameters.
        This method updates the simulation parameters with the provided gamma values,
        runs the trajectory simulation and its derivatives, computes the difference
        between the simulated and reference trajectories, and calculates the objective
        function (sum of squared differences) and its gradient with respect to the
        gamma parameters. It also records the simulation time and average trajectory
        time statistics.
        Parameters
        ----------
        x : np.ndarray
            Array containing the gamma parameters to be optimized. The first
            `self.n_gamma_rel` elements correspond to one set of gammas, and the
            remaining elements correspond to another set.
        Returns
        -------
        f : float
            The value of the objective function (sum of squared differences between
            simulated and reference trajectories).
        grad : np.ndarray
            The gradient of the objective function with respect to the gamma
            parameters, flattened into a 1D array.
        sim_time : float
            The time taken to run the simulation, in seconds.
        avg_min_max_traj_time : float
            The average of the minimum and maximum trajectory times, as returned by
            the trajectory simulation.
        """



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

