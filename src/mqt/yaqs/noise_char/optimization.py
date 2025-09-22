# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module contains the optimization routines for noise characterization."""

from __future__ import annotations

import contextlib
import copy
import pathlib
import pickle  # noqa: S403
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import Observable
    from mqt.yaqs.noise_char.propagation import PropagatorWithGradients


def trapezoidal(y: np.ndarray | list[float] | None, x: np.ndarray | list[float] | None) -> NDArray[np.float64]:
    """Compute the cumulative integral of `y` with respect to `x` using the trapezoidal rule.

    This function applies the trapezoidal rule to compute the cumulative numerical
    integration of discrete data points. The output array contains the running integral
    from the first data point up to each index.

    Args:
        y (array_like of float or None): Dependent variable values at each point in `x`.
            Must be the same length as `x`.
        x (array_like of float or None): Independent variable values corresponding to `y`.
            Must be the same length as `y`.

    Returns:
        numpy.ndarray of float64: Array of cumulative integral values.
        `integral[i]` is the integral from `x[0]` to `x[i]`.

    Raises:
        ValueError: If either `x` or `y` is `None`.
        ValueError: If `x` and `y` have different lengths.

    Notes:
        - The first value of the returned integral is always `0.0`.
        - The trapezoidal rule approximates the area under the curve by summing the
          areas of trapezoids formed between consecutive points.
        - This method assumes that `x` is ordered and that the intervals may be non-uniform.

    Examples:
        >>> import numpy as np
        >>> x = np.array([0, 1, 2, 3])
        >>> y = np.array([0, 1, 4, 9])
        >>> trapezoidal(y, x)
        array([0. , 0.5, 2.5, 7. ])
    """
    if y is None or x is None:
        msg = f"x or y is None. x = {x}, y = {y}"
        raise ValueError(msg)

    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if len(x) != len(y):
        msg = f"Mismatch in the number of elements between x and y. len(x) = {len(x)} and len(y) = {len(y)}"
        raise ValueError(msg)

    integral = np.zeros(len(y), dtype=np.float64)
    integral[0] = 0.0

    for i in range(1, len(y)):
        integral[i] = integral[i - 1] + 0.5 * (x[i] - x[i - 1]) * (y[i] + y[i - 1])

    return integral


class LossClass:
    """A base LossClass to track optimization history and compute averages."""

    n_avg = 20

    def __init__(
        self, *, ref_traj: list[Observable], traj_gradients: PropagatorWithGradients, working_dir: str | Path = '.', print_to_file: bool = False
    ) -> None:
        """Initializes the optimization class for noise characterization.

        Args:
            ref_traj (List[Observable]): A list of Observable objects representing the reference trajectory.
            traj_gradients (PropagatorWithGradients): An object that provides trajectory gradients and supports setting the observable list.
            print_to_file (bool, optional): If True, enables printing output to a file. Defaults to False.

        Attributes:
            n_eval (int): Counter for the number of evaluations performed.
            x_history (list[np.ndarray]): History of parameter vectors evaluated.
            f_history (list[float]): History of function values evaluated.
            x_avg_history (list[np.ndarray]): History of averaged parameter vectors.
            diff_avg_history (list[float]): History of differences between averages.
            grad_history (list[np.ndarray]): History of gradient vectors.
            print_to_file (bool): Indicates whether to print output to a file.
            work_dir (Path): Working directory for file output.
            ref_traj (List[Observable]): Deep copy of the reference trajectory.
            traj_gradients (PropagatorWithGradients): Deep copy of the trajectory gradients object.
            ref_traj_array (np.ndarray): Array of results from the reference trajectory.
            d (int): Dimensionality of the input noise model's processes.
        """
        self.n_eval = 0
        self.x_history: list[np.ndarray] = []
        self.f_history: list[float] = []
        self.x_avg_history: list[np.ndarray] = []
        self.diff_avg_history: list[float] = []
        self.grad_history: list[np.ndarray] = []
        self.print_to_file: bool = print_to_file


        self.ref_traj = copy.deepcopy(ref_traj)
        self.traj_gradients = copy.deepcopy(traj_gradients)

        self.traj_gradients.set_observable_list(ref_traj)

        self.ref_traj_array = copy.deepcopy(np.array([obs.results for obs in self.ref_traj]))

        self.d = len(self.traj_gradients.input_noise_model.processes)

        self.t = copy.deepcopy(self.traj_gradients.sim_params.times)

        self.set_work_dir(path=working_dir)

        self.write_traj(obs_array = self.ref_traj_array, output_file = self.work_dir / "ref_traj.txt")


    def compute_avg(self) -> None:
        """Computes the average of the parameter history and appends it to the average history.

        If the length of `x_history` is less than or equal to `n_avg`, computes the mean over the entire `x_history`.
        Otherwise, computes the mean over the entries in `x_history` starting from index `n_avg`.
        The computed average is appended to `x_avg_history`.


        """
        if len(self.x_history) <= self.n_avg:
            x_avg = np.mean(self.x_history, axis=0)
        else:
            x_avg = np.mean(self.x_history[self.n_avg :], axis=0)

        self.x_avg_history.append(x_avg.copy())

    def compute_diff_avg(self) -> None:
        """Computes the maximum absolute difference between the last two entries in `x_avg_history`.

        This method is intended to track the change in the average values stored in `x_avg_history`
        over successive iterations.
        """
        if len(self.x_avg_history) > 1:
            diff: float = np.max(np.abs(self.x_avg_history[-1] - self.x_avg_history[-2]))
            self.diff_avg_history.append(diff)

    def post_process(self, x: np.ndarray, f: float, grad: np.ndarray) -> None:
        """Post-processes the results of an optimization step.

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

        self.write_traj(obs_array = self.obs_array, output_file = self.work_dir / f"opt_traj_{self.n_eval}.txt")

        print(f"post_process: n_eval = {self.n_eval} x = {self.x_history[-1]} loss = {self.f_history[-1]}")
        if self.print_to_file:
            self.write_to_file(self.history_file_name, self.f_history[-1], self.x_history[-1], self.grad_history[-1])
            self.write_to_file(
                self.history_avg_file_name, self.f_history[-1], self.x_avg_history[-1], self.grad_history[-1]
            )

    def reset(self) -> None:
        """Reset the optimization history and evaluation counter.

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

    def set_history(
        self,
        x_history: list[np.ndarray] | np.ndarray,
        f_history: list[float] | np.ndarray,
        x_avg_history: list[np.ndarray] | np.ndarray,
        diff_avg_history: list[float] | np.ndarray,
    ) -> None:
        """Stores the optimization history data.

        Parameters.
        ----------
        x_history : list or array-like
            Sequence of parameter vectors evaluated during the optimization process.
        f_history : list or array-like
            Sequence of objective function values corresponding to each parameter vector in `x_history`.
        x_avg_history : list or array-like
            Sequence of averaged parameter vectors, typically used for tracking the running average during optimization.
        diff_avg_history : list or array-like
            Sequence of differences between consecutive averaged parameter vectors, useful for convergence analysis.

        Notes:
        -----
        This method updates the object's history attributes and sets the number of evaluations performed.
        """
        self.n_eval = len(x_history)
        self.x_history = [np.copy(x) for x in x_history]
        self.f_history = [float(f) for f in f_history]  # floats are immutable, so copy not needed
        self.x_avg_history = [np.copy(x) for x in x_avg_history]
        self.diff_avg_history = [float(d) for d in diff_avg_history]

    def set_work_dir(self, *, path: str | Path = '.') -> None:
        """Sets the base directory for storing optimization history and related files.

        Parameters:
            path (str | Path): The base directory path to use for output files.

        Side Effects:
            - Sets the working directory (`self.work_dir`) based on the provided path.
            - Sets file names for history and average history logs.
        """
        self.work_dir: Path = Path(path)

        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.history_file_name = self.work_dir / "loss_x_history.txt"
        self.history_avg_file_name = self.work_dir / "loss_x_history_avg.txt"

    def write_to_file(self, file_name: Path, f: float, x: np.ndarray, grad: np.ndarray) -> None:
        """Writes the current evaluation data to a specified file if file output is enabled.

        Parameters:
            file_name (Path): The path to the file where data will be appended.
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
            if not file_name.exists():
                with file_name.open("w", encoding="utf-8") as file:
                    file.write(
                        "# iter  loss  "
                        + "  ".join([f"x{i + 1}" for i in range(self.d)])
                        + "    "
                        + "  ".join([f"grad_x{i + 1}" for i in range(self.d)])
                        + "\n"
                    )
 
            with file_name.open("a", encoding="utf-8") as file:
                file.write(
                    f"{self.n_eval}    {f}  "
                    + "  ".join([f"{x[j]:.6f}" for j in range(self.d)])
                    + "    "
                    + "  ".join([f"{grad[j]:.6f}" for j in range(self.d)])
                    + "\n"
                )

    def write_traj(self, obs_array: np.array, output_file: Path) -> None:
        """Saves the optimized trajectory of expectation values to a text file.

        This method reshapes the `exp_vals_traj` array, concatenates the time array `self.t` as the first row,
        and writes the resulting data to a file named `opt_traj_{self.n_eval}.txt` in the working directory.
        The file includes a header with time and observable labels.
        The output file format:
            - Each column corresponds to a time point or an observable at a specific site.
            - The first column is time (`t`).
            - Subsequent columns are labeled as `x0`, `y0`, `z0`, ..., up to the number of observed
            sites and system size.
        Attributes used:
            exp_vals_traj (np.ndarray): Array of expectation values with shape (n_obs_site, sites, n_t).
            t (np.ndarray): Array of time points.
            work_dir (str): Directory where the output file will be saved.
            n_eval (int): Evaluation index used in the output filename.
        File saved:
            {work_dir}/opt_traj_{n_eval}.txt.
        """
        n_obs, _n_t = np.shape(obs_array)
        exp_vals_traj_with_t = np.concatenate([np.array([self.t]), obs_array], axis=0)

        header = "t  " + "  ".join(["obs_" + str(i) for i in range(n_obs)])

        np.savetxt(output_file, exp_vals_traj_with_t.T, header=header, fmt="%.6f")


    def x_to_noise_model(self, x: np.ndarray) -> NoiseModel:
        """Converts the optimization variable x to a NoiseModel instance."""
        return_model = copy.deepcopy(self.traj_gradients.input_noise_model)

        for i in range(self.d):
            return_model.processes[i]["strength"] = x[i]

        return return_model

    def noise_model_to_x(self, noise_model: NoiseModel) -> np.ndarray:
        """Converts a NoiseModel instance to the optimization variable x."""
        x: np.ndarray = np.zeros(self.d, dtype=np.float64)
        for i in range(self.d):
            x[i] = noise_model.processes[i]["strength"]
        return x

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray, float]:
        """Evaluates the objective function and its gradient for the given parameters.

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
                - avg_min_max_traj_time (Any): Average, minimum and maximum trajectory running times.
        """
        if len(x) != self.d:
            msg = f"Input array must have length {self.d}, got {len(x)}"
            raise ValueError(msg)

        print("x: ", x)

        noise_model = self.x_to_noise_model(x)

        print("Noise model:", noise_model.processes)

        start_time = time.time()

        self.traj_gradients.run(noise_model)

        self.obs_array = copy.deepcopy(self.traj_gradients.obs_array)

        self.d_on_d_gk = copy.deepcopy(self.traj_gradients.d_on_d_gk_array)

        end_time = time.time()

        _n_jump, n_obs, nt = np.shape(self.d_on_d_gk)

        diff = self.obs_array - self.ref_traj_array

        loss: float = np.sum(diff**2)

        # I reshape diff so it has a shape compatible with d_on_d_gk (n_jump_site, n_obs_site, sites, nt)
        #  to do elemtwise multiplication. Then I sum over the n_obs_site, sites and nt dimensions to
        # get the gradient for each gamma, returning a vector of shape (n_jump_site)
        grad_vec = np.sum(2 * diff.reshape(1, n_obs, nt) * self.d_on_d_gk, axis=(1, 2))

        if len(grad_vec) != len(self.traj_gradients.index_list):
            msg = f"Gradient vector length {len(grad_vec)} does not match index list length {len(self.traj_gradients.index_list)}"
            raise ValueError(msg)

        grad = np.bincount(self.traj_gradients.index_list, weights=grad_vec)

        self.post_process(x.copy(), loss, grad.copy())

        sim_time = end_time - start_time  # Simulation time

        return loss, grad, sim_time


def adam_optimizer(
    f: LossClass,
    x_copy: np.ndarray,
    *,
    alpha: float = 0.05,
    max_iterations: int = 1000,
    threshold: float = 5e-4,
    max_n_convergence: int = 50,
    tolerance: float = 1e-8,
    beta1: float = 0.5,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    restart: bool = False,
    restart_file: Path | None = None,
) -> tuple[
    list[float],  # f.f_history: History of loss values.
    list[np.ndarray],  # f.x_history: History of parameter vectors.
    list[np.ndarray],  # f.x_avg_history: History of averaged parameter vectors.
    np.ndarray,  # f.t: Time array from the loss function.
    np.ndarray,  # f.exp_vals_traj: Optimized trajectory of expectation values.
]:
    """Performs Adam optimization on a given loss function with support for checkpointing and restart.

    Args:
        f (loss_class): An instance of a loss function class.
        x_copy (np.ndarray): Initial parameter vector to optimize.
        alpha (float, optional): Learning rate for Adam optimizer. Default is 0.05.
        max_iterations (int, optional): Maximum number of optimization iterations. Default is 1000.
        threshold (float, optional): Threshold for parameter convergence check. Default is 5e-4.
        max_n_convergence (int, optional): Number of consecutive iterations to check for convergence. Default is 50.
        tolerance (float, optional): Absolute loss tolerance for early stopping. Default is 1e-8.
        beta1 (float, optional): Exponential decay rate for the first moment estimates. Default is 0.5.
        beta2 (float, optional): Exponential decay rate for the second moment estimates. Default is 0.999.
        epsilon (float, optional): Small constant for numerical stability. Default is 1e-8.
        restart (bool, optional): Whether to restart optimization from a checkpoint. Default is False.
        restart_file (str, optional): Path to a specific checkpoint file to restart from.
        If None, the latest checkpoint in the working directory is used.

    Returns:
        Tuple[
            List[float],      # f.f_history: History of loss values.
            List[np.ndarray], # f.x_history: History of parameter vectors.
            List[np.ndarray], # f.x_avg_history: History of averaged parameter vectors.
            np.ndarray,       # f.t: Time array from the loss function.
            np.ndarray        # f.exp_vals_traj: Optimized trajectory of expectation values.
        ]

    Raises:
        ValueError: If restart is True but no valid checkpoint file is found.

    Notes:
        - The optimizer saves a checkpoint at every iteration in the working directory specified by `f.work_dir`.
        - All parameter values are clipped to the [0, 1] range after each update.
        - Performance metrics are logged to 'performance_metric_sec.txt' in the working directory.
    """
    restart_dir = f.work_dir

    perf_path = Path(f.work_dir) / "performance_metric_sec.txt"

    # Find the latest restart file in the restart_dir
    if restart_file is None and restart and pathlib.Path(restart_dir).is_dir():
        restart_files = [
            file_path.name
            for file_path in Path(restart_dir).iterdir()
            if file_path.name.startswith("restart_step_") and file_path.suffix == ".pkl"
        ]
        if restart_files:
            # Sort by step number
            restart_files.sort()
            restart_file = Path(restart_dir) / restart_files[-1]

    # Initialization
    if restart and restart_file is not None:
        if not restart_file.exists():
            msg = "Restart file not found."
            raise ValueError(msg)
        # We only load restart files we created ourselves; no untrusted input here.
        with restart_file.open("rb") as handle:
            saved = pickle.load(handle)  # noqa: S301
        x = saved["x"]
        m = saved["m"]
        v = saved["v"]
        start_iter = saved["iteration"] + 1  # resume from next iteration

        f.set_history(saved["x_history"], saved["f_history"], saved["x_avg_history"], saved["diff_avg_history"])

        f.t = saved["t"].copy()
        f.obs_array = saved["obs_traj"].copy()

    else:
        # Remove all .pkl files in the folder
        restart_path = Path(restart_dir)
        for file_path in restart_path.iterdir():
            if file_path.suffix == ".pkl":
                with contextlib.suppress(Exception):
                    file_path.unlink()

        x = x_copy.copy()
        d = len(x)
        m = np.zeros(d)
        v = np.zeros(d)
        start_iter = 0

        # Write a header to performance_metric.txt in f.work_dir

        with perf_path.open("w", encoding="utf-8") as pf:
            pf.write("# iter    opt_step_time    simulation_time    avg_traj_time    min_traj_time    max_traj_time\n")

    for i in range(start_iter, max_iterations):
        # Calculate loss and gradients (unchanged)

        start_time = time.time()

        loss, grad, sim_time = f(x)

        print(i,loss)

        # Adam update steps (NEW)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)

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
            "x_loss": x + update,
            "loss": loss,
            "grad": grad.copy(),
            "beta1": beta1,
            "beta2": beta2,
            "m": m.copy(),
            "v": v.copy(),
            "update": update.copy(),
            "f_history": f.f_history.copy(),
            "x_history": f.x_history.copy(),
            "x_avg_history": f.x_avg_history.copy(),
            "diff_avg_history": f.diff_avg_history.copy(),
            "t": f.t.copy(),
            "obs_traj": f.obs_array.copy(),
        }

        restart_path = Path(restart_dir) / f"restart_step_{i + 1:04d}.pkl"
        with restart_path.open("wb") as handle:
            pickle.dump(restart_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        end_time = time.time()

        iter_time = end_time - start_time

        with perf_path.open("a", encoding="utf-8") as pf:
            pf.write(f"  {i}    {iter_time}    {sim_time}  \n")

        if abs(loss) < tolerance:
            print(f"Stopping optimization because loss {loss} < tolerance {tolerance}")
            break

        # Convergence check
        if len(f.diff_avg_history) > max_n_convergence and all(
            diff < threshold for diff in f.diff_avg_history[-max_n_convergence:]
        ):
            print(f"Stopping optimization because differences {f.diff_avg_history[-max_n_convergence:]} < threshold {threshold}")
            break

    return f.f_history, f.x_history, f.x_avg_history, f.t, f.obs_array
