# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module contains the optimization routines for noise characterization."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from mqt.yaqs.core.data_structures.simulation_parameters import Observable
    from mqt.yaqs.noise_char.propagation import Propagator


def lineal_function_1000(_i: int) -> int:
    """Return a constant value of 1000.

    This function takes an input parameter and returns a fixed value of 1000,
    regardless of the input value.

    Args:
        _i (int): An integer parameter (unused in the calculation).

    Returns:
        int: The constant value 1000.
    """
    return 1000


class LossClass:
    """A base LossClass to track optimization history and compute averages."""

    n_avg = 20

    def __init__(
        self,
        *,
        ref_traj: list[Observable],
        propagator: Propagator,
        working_dir: str | Path = ".",
        num_traj: Callable[[int], int] = lineal_function_1000,
        print_to_file: bool = False,
        return_numeric_gradients: bool = False,
        epsilon: float = 1e-3,
    ) -> None:
        """Initializes the optimization class for noise characterization.

        Args:
            ref_traj (list[Observable]): A list of Observable objects representing the reference trajectory.
            propagator (Propagator): An object that provides the trajectories of some observable list.
            working_dir (str | Path, optional): The directory where output files will be stored. Default ".".
            num_traj (Callable[[int], int], optional): Function to determine number of trajectories based on
                                                    evaluation count. Default lineal_function_1000.
            print_to_file (bool, optional): If True, enables printing output to a file. Default False.
            return_numeric_gradients (bool, optional): If True, compute gradients numerically. Default False.
            epsilon (float, optional): Step size for numerical gradients. Default 1e-3.

        Attributes:
            n_eval (int): Counter for the number of evaluations performed.
            x_history (list[np.ndarray]): History of parameter vectors evaluated.
            f_history (list[float]): History of function values evaluated.
            x_avg_history (list[np.ndarray]): History of averaged parameter vectors.
            diff_avg_history (list[float]): History of differences between averages.
            grad_history (list[np.ndarray]): History of gradient vectors.
            print_to_file (bool): Indicates whether to print output to a file.
            work_dir (Path): Working directory for file output.
            ref_traj (list[Observable]): Deep copy of the reference trajectory.
            propagator (Propagator): Deep copy of the Propagator object.
            ref_traj_array (np.ndarray): Array of results from the reference trajectory.
            d (int): Dimensionality of the input noise model's processes.
            t (np.ndarray): Time array from the propagator's simulation parameters.
            return_numeric_gradients (bool): Whether to return numeric gradients.
            epsilon (float): Step size for numerical gradient computation.
            converged (bool): Flag indicating if convergence has been reached.
            n_avg (int): Number of recent evaluations to average for convergence check.
            n_conv (int): Number of consecutive small differences required for convergence.
            avg_tol (float): Tolerance for average differences in convergence check.
            num_traj (Callable[[int], int]): Function to compute number of trajectories.
        """
        self.n_eval = 0
        self.x_history: list[np.ndarray] = []
        self.f_history: list[float] = []
        self.x_avg_history: list[np.ndarray] = []
        self.diff_avg_history: list[float] = []
        self.grad_history: list[np.ndarray] = []
        self.print_to_file: bool = print_to_file

        self.ref_traj = copy.deepcopy(ref_traj)
        self.propagator = copy.deepcopy(propagator)

        self.propagator.set_observable_list(ref_traj)

        self.ref_traj_array = copy.deepcopy(np.array([obs.results for obs in self.ref_traj]))

        self.d = len(self.propagator.compact_noise_model.compact_processes)

        self.t = copy.deepcopy(self.propagator.sim_params.times)

        self.set_work_dir(path=working_dir)

        self.write_traj(obs_array=self.ref_traj_array, output_file=self.work_dir / "ref_traj.txt")

        self.return_numeric_gradients = return_numeric_gradients

        self.epsilon = epsilon

        self.converged = False

        self.n_avg = 100

        self.n_conv = 20

        self.avg_tol = 1e-6

        self.num_traj = num_traj

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

    def check_convergence(self) -> None:
        """Check if the optimization has converged based on the average differences.

        This method checks if the last `n_conv` entries in `diff_avg_history` are all
        below the tolerance `avg_tol`. If so, it sets the `converged` flag to True.

        The convergence criterion is that all recent differences in the averaged
        parameter history are sufficiently small, indicating that the optimization
        has stabilized.
        """
        if len(self.diff_avg_history) > self.n_conv and all(
            diff < self.avg_tol for diff in self.diff_avg_history[-self.n_conv :]
        ):
            self.converged = True

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

        self.write_traj(obs_array=self.obs_array, output_file=self.work_dir / f"opt_traj_{self.n_eval}.txt")

        if self.print_to_file:
            self.write_to_file(self.history_file_name, self.f_history[-1], self.x_history[-1], self.grad_history[-1])
            self.write_to_file(
                self.history_avg_file_name, self.f_history[-1], self.x_avg_history[-1], self.grad_history[-1]
            )

        self.check_convergence()

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

    def set_work_dir(self, *, path: str | Path = ".") -> None:
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

    def write_traj(self, obs_array: np.ndarray, output_file: Path) -> None:
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

    def x_to_noise_model(self, x: np.ndarray) -> CompactNoiseModel:
        """Converts the optimization variable x to a CompactNoiseModel instance.

        Args:
            x (np.ndarray): The optimization variable representing noise strengths.

        Returns:
            CompactNoiseModel: The corresponding noise model with updated strengths.
        """
        return_processes = copy.deepcopy(self.propagator.compact_noise_model.compact_processes)

        for i in range(self.d):
            return_processes[i]["strength"] = x[i]

        return CompactNoiseModel(return_processes)

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray, float]:
        """Evaluates the objective function and its gradient for the given parameters.

        This method updates the simulation parameters with the provided gamma values,
        runs the trajectory simulation and its derivative, computes the loss (sum of squared
        differences between the simulated and reference trajectories), and calculates the gradient
        of the loss with respect to the gamma parameters. It also measures the simulation time
        and retrieves the average minimum and maximum trajectory times.

        Args:
            x (np.ndarray): Array of gamma parameters to be set in the simulation.

        Raises:
            ValueError: If the length of the input array `x` does not match the expected dimensionality `self.d`.

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

        noise_model = self.x_to_noise_model(x)

        start_time = time.time()

        self.propagator.sim_params.num_traj = self.num_traj(self.n_eval)

        self.propagator.run(noise_model)

        self.obs_array = copy.deepcopy(self.propagator.obs_array)

        end_time = time.time()

        diff = self.obs_array - self.ref_traj_array

        loss: float = np.sum(diff**2)

        sim_time = end_time - start_time  # Simulation time

        if self.return_numeric_gradients:
            grad = np.zeros_like(x)

            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += self.epsilon

                noise_model_plus = self.x_to_noise_model(x_plus)

                self.propagator.run(noise_model_plus)
                obs_array_plus = copy.deepcopy(self.propagator.obs_array)

                diff_plus = obs_array_plus - self.ref_traj_array
                loss_plus = np.sum(diff_plus**2)

                grad[i] = (loss_plus - loss) / self.epsilon

            self.post_process(x.copy(), loss, grad.copy())

            return loss, grad, sim_time

        grad = np.array([0] * self.d)

        self.post_process(x.copy(), loss, grad.copy())

        return loss, grad, sim_time
